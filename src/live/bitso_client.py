"""
============================================================
bitso_client.py - Cliente REST Configurable para Bitso API
============================================================

Provee acceso ligero y sin estado (stateless) a la API REST de Bitso:
1. Velas históricas recientes (OHLCV) para alimentar el DataPipeline
   con únicamente las últimas N barras cerradas (ej. 260 velas diarias).
2. Order Book público en tiempo real (Best Bid / Best Ask) en el
   milisegundo exacto de ejecución para auditar la fricción real.

CONFIGURACIÓN:
--------------
Puedes ajustar endpoints, libro (`btc_mxn`), `time_bucket` y credenciales
en `config/trading_config.yaml` (sección `bitso_api`) o mediante variables
de entorno:
  - BITSO_BASE_URL
  - BITSO_BOOK
  - BITSO_API_KEY
  - BITSO_API_SECRET
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import urlencode, urlparse

import pandas as pd
import requests

logger = logging.getLogger(__name__)


@dataclass
class BitsoConfig:
    """Configuración de conexión para la API REST de Bitso."""

    base_url: str = "https://api.bitso.com/api/v3"
    ohlc_endpoint: str = "/ohlc"
    order_book_endpoint: str = "/order_book/"
    book: str = "btc_mxn"
    time_bucket: int = 86400  # 86400 s = 1 vela diaria
    lookback_bars: int = 260
    warmup_extra_bars: int = 0
    timeout_seconds: float = 15.0
    max_retries: int = 3
    api_key: str = ""
    api_secret: str = ""

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]] = None) -> "BitsoConfig":
        """Construye configuración combinando YAML + variables de entorno."""
        cfg = data or {}
        return cls(
            base_url=os.environ.get(
                "BITSO_BASE_URL",
                str(cfg.get("base_url", "https://api.bitso.com/api/v3")),
            ).rstrip("/"),
            ohlc_endpoint=str(cfg.get("ohlc_endpoint", "/ohlc")),
            order_book_endpoint=str(cfg.get("order_book_endpoint", "/order_book/")),
            book=os.environ.get(
                "BITSO_BOOK",
                str(cfg.get("book", "btc_mxn")),
            ).lower(),
            time_bucket=int(cfg.get("time_bucket", 86400)),
            lookback_bars=int(cfg.get("lookback_bars", 260)),
            warmup_extra_bars=int(cfg.get("warmup_extra_bars", 0)),
            timeout_seconds=float(cfg.get("timeout_seconds", 15.0)),
            max_retries=int(cfg.get("max_retries", 3)),
            api_key=os.environ.get("BITSO_API_KEY", str(cfg.get("api_key", ""))),
            api_secret=os.environ.get(
                "BITSO_API_SECRET", str(cfg.get("api_secret", ""))
            ),
        )


@dataclass(frozen=True)
class OrderBookSnapshot:
    """Instantánea en tiempo real del mejor Bid/Ask del Order Book."""

    book: str
    best_bid: float
    best_ask: float
    best_bid_amount: float
    best_ask_amount: float
    mid_price: float
    spread_mxn: float
    spread_bps: float
    timestamp_utc: str


class BitsoAPIError(RuntimeError):
    """Error al consultar o decodificar la API REST de Bitso."""


class BitsoRESTClient:
    """
    Cliente REST para consultar velas OHLCV y el Order Book en vivo de Bitso.

    Soporta el esquema oficial de Bitso API v3 y también formatos OHLCV
    genéricos en caso de configurar un proxy o endpoint propio.
    """

    def __init__(
        self,
        config: Optional[BitsoConfig] = None,
        session: Optional[requests.Session] = None,
    ):
        self.config = config or BitsoConfig.from_dict()
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "Neuroevolucion-NEAT-PaperTrader/1.0",
            }
        )

    @staticmethod
    def symbol_to_book(symbol: str) -> str:
        """Convierte símbolos como 'BTC-MXN' o 'BTC/MXN' a formato Bitso 'btc_mxn'."""
        cleaned = symbol.strip().lower().replace("-", "_").replace("/", "_")
        return cleaned

    def _build_auth_header(
        self,
        method: str,
        request_path: str,
        body: str = "",
    ) -> dict[str, str]:
        """
        Genera encabezado HMAC-SHA256 oficial de Bitso v3 si hay credenciales.
        Formato: `Authorization: Bitso <key>:<nonce>:<signature>`
        """
        if not self.config.api_key or not self.config.api_secret:
            return {}

        nonce = str(int(time.time() * 1000))
        message = f"{nonce}{method.upper()}{request_path}{body}".encode("utf-8")
        signature = hmac.new(
            self.config.api_secret.encode("utf-8"),
            message,
            hashlib.sha256,
        ).hexdigest()
        return {"Authorization": f"Bitso {self.config.api_key}:{nonce}:{signature}"}

    def _request_json(
        self,
        method: str,
        endpoint: str,
        params: Optional[dict[str, Any]] = None,
        authenticated: bool = False,
    ) -> Any:
        """Ejecuta petición HTTP con reintentos exponenciales y validación JSON."""
        ep = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        url = f"{self.config.base_url}{ep}"

        parsed = urlparse(url)
        query_str = f"?{urlencode(params)}" if params else ""
        request_path = f"{parsed.path}{query_str}"

        headers: dict[str, str] = {}
        if authenticated:
            headers.update(self._build_auth_header(method, request_path))

        last_exc: Optional[Exception] = None
        attempts = max(1, self.config.max_retries)

        for attempt in range(1, attempts + 1):
            try:
                response = self.session.request(
                    method=method.upper(),
                    url=url,
                    params=params,
                    headers=headers,
                    timeout=self.config.timeout_seconds,
                )
                if response.status_code in (429, 500, 502, 503, 504) and attempt < attempts:
                    wait_s = 0.5 * (2 ** (attempt - 1))
                    logger.warning(
                        f"Bitso HTTP {response.status_code} en {ep} "
                        f"(intento {attempt}/{attempts}). Reintentando en {wait_s:.1f}s..."
                    )
                    time.sleep(wait_s)
                    continue

                response.raise_for_status()
                data = response.json()

                # Esquema estándar Bitso v3: {"success": true, "payload": ...}
                if isinstance(data, dict) and "success" in data:
                    if not data.get("success", False):
                        err_info = data.get("error", data)
                        raise BitsoAPIError(f"Bitso API rechazó la solicitud: {err_info}")
                    return data.get("payload")

                return data

            except (requests.RequestException, ValueError) as exc:
                last_exc = exc
                if attempt < attempts:
                    wait_s = 0.5 * (2 ** (attempt - 1))
                    logger.warning(
                        f"Error de red/JSON consultando {url} "
                        f"(intento {attempt}/{attempts}): {exc}"
                    )
                    time.sleep(wait_s)
                else:
                    break

        raise BitsoAPIError(
            f"Fallo al consultar Bitso REST ({url}) tras {attempts} intentos: {last_exc}"
        ) from last_exc

    def fetch_ohlcv(
        self,
        book: Optional[str] = None,
        limit: Optional[int] = None,
        time_bucket: Optional[int] = None,
        exclude_open_candle: bool = True,
        now_utc: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Descarga únicamente las últimas `limit` velas cerradas desde Bitso REST.

        Parameters
        ----------
        book : str, opcional
            Libro de Bitso (ej. 'btc_mxn'). Default: `config.book`.
        limit : int, opcional
            Cantidad exacta de velas cerradas a conservar (ej. 260).
        time_bucket : int, opcional
            Tamaño de vela en segundos (86400 = 1d, 3600 = 1h).
        exclude_open_candle : bool
            Si True, descarta cualquier vela en formación cuyo periodo aún no haya cerrado.
        now_utc : datetime, opcional
            Reloj de referencia en UTC (útil para pruebas deterministas).

        Returns
        -------
        pd.DataFrame
            DataFrame con columnas `['Open', 'High', 'Low', 'Close', 'Volume']`
            indexado por `DatetimeIndex` cronológico UTC-naive.
        """
        target_book = (book or self.config.book).lower()
        target_limit = int(limit if limit is not None else self.config.lookback_bars)
        bucket_sec = int(time_bucket if time_bucket is not None else self.config.time_bucket)

        if target_limit <= 0:
            raise ValueError("limit debe ser mayor a 0")

        ref_now = now_utc or datetime.now(timezone.utc)
        if ref_now.tzinfo is None:
            ref_now = ref_now.replace(tzinfo=timezone.utc)
        end_ms = int(ref_now.timestamp() * 1000)

        # Pedimos un pequeño margen (+10 velas) para cubrir la vela abierta actual
        # o huecos de mantenimiento sin cargar todo el historial plurianual.
        margin_bars = target_limit + 10
        start_ms = end_ms - (margin_bars * bucket_sec * 1000)

        params = {
            "book": target_book,
            "time_bucket": bucket_sec,
            "start": start_ms,
            "end": end_ms,
        }

        payload = self._request_json(
            "GET",
            self.config.ohlc_endpoint,
            params=params,
            authenticated=False,
        )

        df = self._parse_ohlcv_payload(payload)
        if df.empty:
            raise BitsoAPIError(
                f"Bitso devolvió 0 velas OHLCV para book='{target_book}' "
                f"(bucket={bucket_sec}s)."
            )

        if exclude_open_candle:
            cutoff_naive = (
                ref_now.astimezone(timezone.utc).replace(tzinfo=None)
                - pd.Timedelta(seconds=bucket_sec)
            )
            # Una vela iniciada en `t` cierra en `t + bucket_sec`.
            # Por tanto, está cerrada sí y solo sí `t <= ref_now - bucket_sec`.
            df = df.loc[df.index <= cutoff_naive]

        if df.empty:
            raise BitsoAPIError(
                f"No quedaron velas cerradas tras filtrar velas en formación "
                f"(book='{target_book}')."
            )

        df = df.tail(target_limit).copy()
        logger.info(
            f"Bitso OHLCV recibido: {len(df)} velas cerradas | book={target_book} | "
            f"rango=[{df.index[0]} -> {df.index[-1]}]"
        )
        return df

    @staticmethod
    def _parse_ohlcv_payload(payload: Any) -> pd.DataFrame:
        """Convierte el payload JSON de OHLC (Bitso v3 o genérico) a DataFrame limpio."""
        if isinstance(payload, dict):
            # En caso de venir envuelto en {"ohlc": [...]} o {"data": [...]}
            for key in ("ohlc", "candles", "data", "results"):
                if key in payload and isinstance(payload[key], list):
                    payload = payload[key]
                    break

        if not isinstance(payload, list) or len(payload) == 0:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

        rows: list[dict[str, Any]] = []
        for item in payload:
            if isinstance(item, dict):
                raw_ts = (
                    item.get("bucket_start_time")
                    or item.get("timestamp")
                    or item.get("time")
                    or item.get("date")
                )
                open_v = item.get("first_rate", item.get("open", item.get("Open")))
                high_v = item.get("max_rate", item.get("high", item.get("High")))
                low_v = item.get("min_rate", item.get("low", item.get("Low")))
                close_v = item.get("last_rate", item.get("close", item.get("Close")))
                vol_v = item.get("volume", item.get("Volume", 0.0))
            elif isinstance(item, (list, tuple)) and len(item) >= 6:
                raw_ts, open_v, high_v, low_v, close_v, vol_v = item[:6]
            else:
                continue

            if raw_ts is None or close_v is None:
                continue

            ts = BitsoRESTClient._parse_timestamp(raw_ts)
            rows.append(
                {
                    "Date": ts,
                    "Open": float(open_v),
                    "High": float(high_v),
                    "Low": float(low_v),
                    "Close": float(close_v),
                    "Volume": float(vol_v if vol_v is not None else 0.0),
                }
            )

        if not rows:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

        df = pd.DataFrame(rows)
        df = df.drop_duplicates(subset=["Date"], keep="last").sort_values("Date")
        df = df.set_index("Date")
        df.index.name = "Date"

        # Garantizar consistencia básica de precios
        valid_mask = (
            (df["Open"] > 0)
            & (df["High"] > 0)
            & (df["Low"] > 0)
            & (df["Close"] > 0)
            & (df["Volume"] >= 0)
        )
        return df.loc[valid_mask, ["Open", "High", "Low", "Close", "Volume"]]

    @staticmethod
    def _parse_timestamp(raw_ts: Any) -> pd.Timestamp:
        """Convierte epoch (ms o s) o string ISO-8601 a Timestamp UTC-naive."""
        if isinstance(raw_ts, (int, float)):
            val = float(raw_ts)
            unit = "ms" if val > 1e11 else "s"
            return pd.to_datetime(val, unit=unit, utc=True).tz_localize(None)

        text = str(raw_ts).strip()
        if text.isdigit():
            val = float(text)
            unit = "ms" if val > 1e11 else "s"
            return pd.to_datetime(val, unit=unit, utc=True).tz_localize(None)

        ts = pd.to_datetime(text, utc=True)
        return ts.tz_localize(None)

    def fetch_order_book(
        self,
        book: Optional[str] = None,
        aggregate: bool = True,
    ) -> OrderBookSnapshot:
        """
        Consulta el Order Book público en este preciso milisegundo y extrae
        el mejor precio de compra disponible (`best_ask`) y de venta (`best_bid`).
        """
        target_book = (book or self.config.book).lower()
        params = {
            "book": target_book,
            "aggregate": "true" if aggregate else "false",
        }

        payload = self._request_json(
            "GET",
            self.config.order_book_endpoint,
            params=params,
            authenticated=False,
        )
        request_ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")

        if not isinstance(payload, dict):
            raise BitsoAPIError(
                f"Formato inesperado en Order Book para '{target_book}': {type(payload)}"
            )

        asks_raw = payload.get("asks") or []
        bids_raw = payload.get("bids") or []
        if not asks_raw or not bids_raw:
            raise BitsoAPIError(
                f"Order Book vacío o incompleto para '{target_book}' "
                f"(asks={len(asks_raw)}, bids={len(bids_raw)})."
            )

        parsed_asks = [self._parse_book_level(level) for level in asks_raw]
        parsed_bids = [self._parse_book_level(level) for level in bids_raw]

        # Best Ask: el vendedor más barato disponible en este instante
        best_ask, best_ask_amount = min(parsed_asks, key=lambda x: x[0])
        # Best Bid: el comprador que más paga en este instante
        best_bid, best_bid_amount = max(parsed_bids, key=lambda x: x[0])

        if best_bid <= 0 or best_ask <= 0:
            raise BitsoAPIError(
                f"Precios inválidos en Order Book ({target_book}): "
                f"bid={best_bid}, ask={best_ask}"
            )
        if best_ask < best_bid:
            raise BitsoAPIError(
                f"Order Book cruzado/inválido ({target_book}): "
                f"ask={best_ask} < bid={best_bid}"
            )

        mid_price = 0.5 * (best_bid + best_ask)
        spread_mxn = best_ask - best_bid
        spread_bps = (spread_mxn / mid_price) * 10_000.0 if mid_price > 0 else 0.0
        book_ts = str(payload.get("updated_at") or request_ts)

        snapshot = OrderBookSnapshot(
            book=target_book,
            best_bid=best_bid,
            best_ask=best_ask,
            best_bid_amount=best_bid_amount,
            best_ask_amount=best_ask_amount,
            mid_price=mid_price,
            spread_mxn=spread_mxn,
            spread_bps=spread_bps,
            timestamp_utc=book_ts,
        )
        logger.info(
            f"Order Book [{target_book}] | Bid={best_bid:,.2f} MXN | "
            f"Ask={best_ask:,.2f} MXN | Spread={spread_mxn:,.2f} MXN ({spread_bps:.2f} bps)"
        )
        return snapshot

    @staticmethod
    def _parse_book_level(level: Any) -> tuple[float, float]:
        """Extrae `(price, amount)` de un nivel del Order Book."""
        if isinstance(level, dict):
            return float(level["price"]), float(level.get("amount", 0.0))
        if isinstance(level, (list, tuple)) and len(level) >= 2:
            return float(level[0]), float(level[1])
        raise BitsoAPIError(f"Nivel de Order Book no reconocido: {level}")

