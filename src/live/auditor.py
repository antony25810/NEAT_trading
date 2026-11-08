"""
============================================================
auditor.py - Contabilidad de Fricción Real (El Auditor)
============================================================

Responsable de:
1. Consultar el Order Book público de Bitso en el milisegundo exacto
   en que se emite una orden:
   - COMPRA (`BUY`): ejecuta al precio `best_ask` real (el precio al
     que realmente te venden en ese segundo, NO el `Close` genérico).
   - VENTA (`SELL`): ejecuta al precio `best_bid` real (el precio al
     que realmente te compran en ese segundo).
2. Registrar cada operación auditada en `live_trades.csv` incluyendo
   spread real, desviación frente al Close genérico (bps), comisiones
   en MXN y PnL neto real.
3. Persistir de forma atómica el estado del portafolio y la posición
   abierta (`live_state.json`) entre ejecuciones diarias programadas.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.live.bitso_client import BitsoRESTClient, OrderBookSnapshot
from src.trading.position import Position, PositionType, TradeRecord

logger = logging.getLogger(__name__)


LIVE_TRADES_COLUMNS = [
    "trade_id",
    "execution_timestamp_utc",
    "order_book_timestamp_utc",
    "candle_date",
    "symbol",
    "book",
    "side",
    "reason",
    "direction_signal",
    "confidence_signal",
    "size_signal",
    "candle_close",
    "best_bid",
    "best_ask",
    "spread_mxn",
    "spread_bps",
    "execution_price",
    "friction_vs_close_bps",
    "btc_size",
    "capital_allocated_mxn",
    "commission_mxn",
    "realized_pnl_mxn",
    "realized_pnl_pct",
    "bars_held",
    "equity_after_mxn",
]


def serialize_position(position: Position) -> dict[str, Any]:
    """Serializa una instancia `Position` preservando sus niveles dinámicos."""
    return {
        "position_type": int(position.position_type),
        "entry_price": float(position.entry_price),
        "entry_time": str(position.entry_time),
        "capital_allocated": float(position.capital_allocated),
        "net_capital": float(position.net_capital),
        "size": float(position.size),
        "commission": float(position.commission),
        "slippage": float(position.slippage),
        "stop_loss_pct": float(position.stop_loss_pct),
        "take_profit_pct": float(position.take_profit_pct),
        "trailing_stop_pct": float(position.trailing_stop_pct),
        "stop_loss_price": float(position.stop_loss_price),
        "take_profit_price": float(position.take_profit_price),
        "trailing_stop_price": float(position.trailing_stop_price),
        "peak_price": float(position.peak_price),
        "bars_held": int(position.bars_held),
    }


def deserialize_position(data: dict[str, Any]) -> Position:
    """Reconstruye exactamente una `Position` abierta desde JSON."""
    pos = Position(
        position_type=PositionType(int(data["position_type"])),
        entry_price=float(data["entry_price"]),
        entry_time=data["entry_time"],
        capital=float(data["capital_allocated"]),
        commission=float(data.get("commission", 0.0065)),
        slippage=0.0,  # entry_price ya contiene el Ask/Bid real ejecutado
        stop_loss_pct=float(data["stop_loss_pct"]),
        take_profit_pct=float(data["take_profit_pct"]),
        trailing_stop_pct=float(data["trailing_stop_pct"]),
    )
    # Restaurar atributos exactos acumulados día a día
    pos.entry_price = float(data["entry_price"])
    pos.net_capital = float(data["net_capital"])
    pos.size = float(data["size"])
    pos.slippage = float(data.get("slippage", 0.0))
    pos.stop_loss_price = float(data["stop_loss_price"])
    pos.take_profit_price = float(data["take_profit_price"])
    pos.trailing_stop_price = float(data["trailing_stop_price"])
    pos.peak_price = float(data["peak_price"])
    pos.bars_held = int(data.get("bars_held", 0))
    return pos


@dataclass
class PortfolioState:
    """Estado persistente del simulador Paper Trading entre corridas diarias."""

    initial_capital: float = 100_000.0
    current_capital: float = 100_000.0
    peak_equity: float = 100_000.0
    trading_halted: bool = False
    last_processed_candle: Optional[str] = None
    total_completed_trades: int = 0
    next_trade_row_id: int = 1
    open_position: Optional[Position] = None
    last_signal: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_capital": float(self.initial_capital),
            "current_capital": float(self.current_capital),
            "peak_equity": float(self.peak_equity),
            "trading_halted": bool(self.trading_halted),
            "last_processed_candle": self.last_processed_candle,
            "total_completed_trades": int(self.total_completed_trades),
            "next_trade_row_id": int(self.next_trade_row_id),
            "open_position": (
                serialize_position(self.open_position)
                if self.open_position is not None
                else None
            ),
            "last_signal": self.last_signal,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(
                timespec="milliseconds"
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PortfolioState":
        open_pos_raw = data.get("open_position")
        return cls(
            initial_capital=float(data.get("initial_capital", 100_000.0)),
            current_capital=float(data.get("current_capital", 100_000.0)),
            peak_equity=float(data.get("peak_equity", 100_000.0)),
            trading_halted=bool(data.get("trading_halted", False)),
            last_processed_candle=data.get("last_processed_candle"),
            total_completed_trades=int(data.get("total_completed_trades", 0)),
            next_trade_row_id=int(data.get("next_trade_row_id", 1)),
            open_position=(
                deserialize_position(open_pos_raw)
                if isinstance(open_pos_raw, dict)
                else None
            ),
            last_signal=data.get("last_signal"),
        )


class RealFrictionAuditor:
    """
    Auditor de fricción real: consulta el Order Book en vivo al instante de
    operar y registra las ejecuciones reales (Ask en compra, Bid en venta)
    en `live_trades.csv` y `live_state.json`.
    """

    def __init__(
        self,
        trades_csv_path: str | Path = "live_trades.csv",
        state_json_path: str | Path = "live_state.json",
        commission: float = 0.0065,
    ):
        self.trades_csv_path = Path(trades_csv_path)
        self.state_json_path = Path(state_json_path)
        self.commission = float(commission)

    def load_state(self, initial_capital: float = 100_000.0) -> PortfolioState:
        """Carga `live_state.json` si existe; si no, crea estado limpio."""
        if not self.state_json_path.exists():
            logger.info(
                f"Estado nuevo inicializado (capital={initial_capital:,.2f} MXN)."
            )
            return PortfolioState(
                initial_capital=initial_capital,
                current_capital=initial_capital,
                peak_equity=initial_capital,
            )

        with open(self.state_json_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        state = PortfolioState.from_dict(data)
        logger.info(
            f"Estado recuperado desde {self.state_json_path} | "
            f"capital={state.current_capital:,.2f} MXN | "
            f"posicion={'LONG' if state.open_position else 'CASH'} | "
            f"ultima_vela={state.last_processed_candle}"
        )
        return state

    def save_state(self, state: PortfolioState) -> None:
        """Guarda `live_state.json` de manera atómica."""
        self.state_json_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.state_json_path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(state.to_dict(), fh, indent=2, ensure_ascii=False)
        os.replace(tmp_path, self.state_json_path)

    def execute_buy_at_real_ask(
        self,
        state: PortfolioState,
        bitso_client: BitsoRESTClient,
        symbol: str,
        book: str,
        candle_date: str,
        candle_close: float,
        trade_capital_mxn: float,
        stop_loss_pct: float,
        take_profit_pct: float,
        trailing_stop_pct: float,
        direction_signal: float,
        confidence_signal: float,
        size_signal: float,
        reason: str = "signal_long",
    ) -> tuple[Position, OrderBookSnapshot, dict[str, Any]]:
        """
        Consulta el Order Book público en este preciso milisegundo y abre
        una posición LONG al precio `best_ask` real.
        """
        exec_ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        ob = bitso_client.fetch_order_book(book=book)

        real_ask = ob.best_ask
        entry_commission_mxn = trade_capital_mxn * self.commission

        # slippage=0.0 porque `real_ask` YA incorpora el spread real del Order Book
        position = Position(
            position_type=PositionType.LONG,
            entry_price=real_ask,
            entry_time=exec_ts,
            capital=trade_capital_mxn,
            commission=self.commission,
            slippage=0.0,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            trailing_stop_pct=trailing_stop_pct,
        )
        state.open_position = position

        # Valuar inmediatamente contra el `best_bid` real para reflejar spread + comisiones
        unrealized_pnl, _ = position.current_unrealized_pnl(ob.best_bid)
        equity_after = state.current_capital + unrealized_pnl
        state.peak_equity = max(state.peak_equity, equity_after)

        friction_vs_close_bps = (
            ((real_ask - candle_close) / candle_close) * 10_000.0
            if candle_close > 0
            else 0.0
        )

        row = {
            "trade_id": state.next_trade_row_id,
            "execution_timestamp_utc": exec_ts,
            "order_book_timestamp_utc": ob.timestamp_utc,
            "candle_date": candle_date,
            "symbol": symbol,
            "book": book,
            "side": "BUY",
            "reason": reason,
            "direction_signal": round(direction_signal, 6),
            "confidence_signal": round(confidence_signal, 6),
            "size_signal": round(size_signal, 6),
            "candle_close": round(candle_close, 2),
            "best_bid": round(ob.best_bid, 2),
            "best_ask": round(ob.best_ask, 2),
            "spread_mxn": round(ob.spread_mxn, 2),
            "spread_bps": round(ob.spread_bps, 2),
            "execution_price": round(real_ask, 2),
            "friction_vs_close_bps": round(friction_vs_close_bps, 2),
            "btc_size": round(position.size, 8),
            "capital_allocated_mxn": round(trade_capital_mxn, 2),
            "commission_mxn": round(entry_commission_mxn, 2),
            "realized_pnl_mxn": 0.0,
            "realized_pnl_pct": 0.0,
            "bars_held": 0,
            "equity_after_mxn": round(equity_after, 2),
        }
        self._append_csv_row(row)
        state.next_trade_row_id += 1

        logger.info(
            f"[AUDITOR BUY] {symbol} | Ask Real={real_ask:,.2f} MXN "
            f"(Close={candle_close:,.2f}, fricción={friction_vs_close_bps:+.2f} bps) | "
            f"BTC={position.size:.8f} | Capital={trade_capital_mxn:,.2f} MXN | "
            f"Comisión={entry_commission_mxn:,.2f} MXN"
        )
        return position, ob, row

    def execute_sell_at_real_bid(
        self,
        state: PortfolioState,
        bitso_client: BitsoRESTClient,
        symbol: str,
        book: str,
        candle_date: str,
        candle_close: float,
        direction_signal: float,
        confidence_signal: float,
        size_signal: float,
        reason: str,
    ) -> tuple[TradeRecord, OrderBookSnapshot, dict[str, Any]]:
        """
        Consulta el Order Book público en este preciso milisegundo y cierra
        la posición LONG abierta al precio `best_bid` real.
        """
        if state.open_position is None:
            raise RuntimeError("No existe posición abierta para ejecutar SELL.")

        exec_ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        ob = bitso_client.fetch_order_book(book=book)

        real_bid = ob.best_bid
        pos = state.open_position
        gross_exit_mxn = pos.size * real_bid
        exit_commission_mxn = gross_exit_mxn * pos.commission

        trade = pos.close(
            current_price=real_bid,
            current_time=exec_ts,
            reason=reason,
        )

        state.current_capital += trade.pnl
        state.open_position = None
        state.total_completed_trades += 1
        state.peak_equity = max(state.peak_equity, state.current_capital)

        friction_vs_close_bps = (
            ((candle_close - real_bid) / candle_close) * 10_000.0
            if candle_close > 0
            else 0.0
        )

        row = {
            "trade_id": state.next_trade_row_id,
            "execution_timestamp_utc": exec_ts,
            "order_book_timestamp_utc": ob.timestamp_utc,
            "candle_date": candle_date,
            "symbol": symbol,
            "book": book,
            "side": "SELL",
            "reason": reason,
            "direction_signal": round(direction_signal, 6),
            "confidence_signal": round(confidence_signal, 6),
            "size_signal": round(size_signal, 6),
            "candle_close": round(candle_close, 2),
            "best_bid": round(ob.best_bid, 2),
            "best_ask": round(ob.best_ask, 2),
            "spread_mxn": round(ob.spread_mxn, 2),
            "spread_bps": round(ob.spread_bps, 2),
            "execution_price": round(real_bid, 2),
            "friction_vs_close_bps": round(friction_vs_close_bps, 2),
            "btc_size": round(trade.size, 8),
            "capital_allocated_mxn": round(trade.capital_invested, 2),
            "commission_mxn": round(exit_commission_mxn, 2),
            "realized_pnl_mxn": round(trade.pnl, 2),
            "realized_pnl_pct": round(trade.pnl_pct, 6),
            "bars_held": int(trade.bars_held),
            "equity_after_mxn": round(state.current_capital, 2),
        }
        self._append_csv_row(row)
        state.next_trade_row_id += 1

        logger.info(
            f"[AUDITOR SELL] {symbol} ({reason}) | Bid Real={real_bid:,.2f} MXN "
            f"(Close={candle_close:,.2f}, fricción={friction_vs_close_bps:+.2f} bps) | "
            f"PnL Neto={trade.pnl:+,.2f} MXN ({trade.pnl_pct * 100:+.2f}%) | "
            f"Barras={trade.bars_held} | Capital={state.current_capital:,.2f} MXN"
        )
        return trade, ob, row

    def _append_csv_row(self, row: dict[str, Any]) -> None:
        """Agrega una fila auditada a `live_trades.csv`, creando cabecera si es nuevo."""
        self.trades_csv_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = (
            self.trades_csv_path.exists() and self.trades_csv_path.stat().st_size > 0
        )
        with open(self.trades_csv_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=LIVE_TRADES_COLUMNS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

