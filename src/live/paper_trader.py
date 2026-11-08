"""
============================================================
paper_trader.py - Motor de Paper Trading Diario en Vivo
============================================================

Integra los 4 pilares de la fase operativa:
1. El Cerebro (`ChampionBrain`): deserializa `best_genome.json` con
   `Genome.from_dict()` y reconstruye pesos/topología exactos con
   `FeedForwardNetwork.from_genome()`.
2. El Alimentador (`DataPipeline.get_live_data` + `BitsoRESTClient`):
   solicita únicamente las últimas 260 velas diarias cerradas a Bitso
   para nutrir `SMA200` y `RollingNormalizer(window=252)` sin Data Leakage.
3. Motor de Ejecución Diaria (`PaperTrader.run_daily_cycle`):
   evalúa la fila normalizada de la vela recién cerrada pasando
   `active_indicator_names` por `network.activate()` y decodificando
   `(Direction, Confidence, Position Size)`.
4. El Auditor (`RealFrictionAuditor`):
   consulta el Order Book público en ese preciso milisegundo y registra
   compras al precio `Ask` real y ventas al precio `Bid` real en `live_trades.csv`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import yaml

from src.data.pipeline import DataPipeline
from src.live.auditor import PortfolioState, RealFrictionAuditor
from src.live.bitso_client import BitsoConfig, BitsoRESTClient
from src.neat.genome import Genome
from src.neat.network import FeedForwardNetwork
from src.trading.position import PositionType

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NeuralDecision:
    """Salida completa del cerebro neuronal sobre una vela cerrada."""

    direction: float
    confidence: float
    size_scaling: float
    desired_position: Optional[PositionType]  # LONG, SHORT, CASH o None (HOLD)
    effective_position_fraction: float
    active_inputs: dict[str, float]

    @property
    def signal_label(self) -> str:
        if self.desired_position == PositionType.LONG:
            return "LONG"
        if self.desired_position == PositionType.SHORT:
            return "SHORT"
        if self.desired_position == PositionType.CASH:
            return "EXIT_TO_CASH"
        return "HOLD"

    def to_dict(self) -> dict[str, Any]:
        return {
            "direction": round(self.direction, 6),
            "confidence": round(self.confidence, 6),
            "size_scaling": round(self.size_scaling, 6),
            "signal_label": self.signal_label,
            "effective_position_fraction": round(self.effective_position_fraction, 6),
            "active_inputs": {
                k: round(float(v), 6) for k, v in self.active_inputs.items()
            },
        }


@dataclass
class DailyCycleReport:
    """Resumen estructurado de una ejecución diaria del Paper Trader."""

    action: str  # BUY, SELL, HOLD_LONG, HOLD_CASH, DRY_RUN_BUY, DRY_RUN_SELL, SKIPPED_ALREADY_PROCESSED
    candle_date: str
    symbol: str
    book: str
    candle_close: float
    decision: Optional[NeuralDecision]
    reason: str
    execution_price: Optional[float] = None
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None
    spread_bps: Optional[float] = None
    realized_pnl_mxn: Optional[float] = None
    equity_mxn: float = 0.0
    bars_loaded: int = 0
    dry_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "candle_date": self.candle_date,
            "symbol": self.symbol,
            "book": self.book,
            "candle_close": round(self.candle_close, 2),
            "reason": self.reason,
            "execution_price": (
                round(self.execution_price, 2)
                if self.execution_price is not None
                else None
            ),
            "best_bid": round(self.best_bid, 2) if self.best_bid is not None else None,
            "best_ask": round(self.best_ask, 2) if self.best_ask is not None else None,
            "spread_bps": (
                round(self.spread_bps, 2) if self.spread_bps is not None else None
            ),
            "realized_pnl_mxn": (
                round(self.realized_pnl_mxn, 2)
                if self.realized_pnl_mxn is not None
                else None
            ),
            "equity_mxn": round(self.equity_mxn, 2),
            "bars_loaded": self.bars_loaded,
            "dry_run": self.dry_run,
            "decision": self.decision.to_dict() if self.decision is not None else None,
        }


class ChampionBrain:
    """
    1. Deserialización del Campeón (El Cerebro).

    Carga `best_genome.json`, reconstruye el `Genome` y compila su
    `FeedForwardNetwork` exacta (pesos, sesgos, funciones de activación
    y máscara de indicadores evolucionada).
    """

    def __init__(
        self,
        genome: Genome,
        confidence_threshold: float = 0.30,
        direction_threshold: float = 0.20,
        allow_short: bool = False,
        source_path: Optional[str] = None,
    ):
        self.genome = genome
        self.network = FeedForwardNetwork.from_genome(genome)
        self.confidence_threshold = float(confidence_threshold)
        self.direction_threshold = float(direction_threshold)
        self.allow_short = bool(allow_short)
        self.source_path = source_path

        # Pares ordenados (input_node_id, indicator_name) idénticos a Backtester
        self._ordered_inputs: list[tuple[int, str]] = [
            (input_id, name)
            for input_id, name in enumerate(self.genome.indicator_mask)
            if self.genome.indicator_mask[name]
        ]
        self._output_ids: list[int] = sorted(self.genome.output_node_ids)
        if len(self._output_ids) < 3:
            raise ValueError(
                f"El genoma debe tener al menos 3 nodos de salida; encontrados: {self._output_ids}"
            )

    @property
    def active_indicator_names(self) -> list[str]:
        return [name for _, name in self._ordered_inputs]

    @property
    def active_input_ids(self) -> list[int]:
        return [input_id for input_id, _ in self._ordered_inputs]

    @classmethod
    def from_json(
        cls,
        genome_path: str | Path,
        fallback_path: Optional[str | Path] = None,
        confidence_threshold: float = 0.30,
        direction_threshold: float = 0.20,
        allow_short: bool = False,
    ) -> "ChampionBrain":
        """Carga el campeón desde `genome_path` (o `fallback_path` si no existe)."""
        primary = Path(genome_path)
        chosen: Optional[Path] = None

        if primary.exists():
            chosen = primary
        elif fallback_path is not None and Path(fallback_path).exists():
            chosen = Path(fallback_path)
            logger.warning(
                f"No se encontró '{primary}'; cargando campeón de respaldo '{chosen}'."
            )
        else:
            raise FileNotFoundError(
                f"Archivo de genoma campeón no encontrado en '{primary}'"
                + (f" ni en '{fallback_path}'." if fallback_path else ".")
            )

        with open(chosen, "r", encoding="utf-8") as fh:
            raw_dict = json.load(fh)

        genome = Genome.from_dict(raw_dict)
        logger.info(
            f"Campeón cargado: {chosen} | genome_id={genome.genome_id} | "
            f"nodos={len(genome.nodes)} (ocultos={len(genome.hidden_node_ids)}) | "
            f"conexiones_activas={len(genome.active_connections)} | "
            f"indicadores={genome.active_indicator_names}"
        )
        return cls(
            genome=genome,
            confidence_threshold=confidence_threshold,
            direction_threshold=direction_threshold,
            allow_short=allow_short,
            source_path=str(chosen),
        )

    def evaluate_row(self, normalized_row: Mapping[str, Any]) -> NeuralDecision:
        """
        Evalúa la fila normalizada de la vela recién cerrada con `network.activate()`
        y decodifica `(Direction, Confidence, Position Size)`.
        """
        inputs_by_id: dict[int, float] = {}
        active_inputs_by_name: dict[str, float] = {}

        for input_id, ind_name in self._ordered_inputs:
            if ind_name not in normalized_row:
                raise KeyError(
                    f"El indicador activo '{ind_name}' no existe en la fila normalizada."
                )
            val = float(normalized_row[ind_name])
            if np.isnan(val) or np.isinf(val):
                raise ValueError(
                    f"Valor inválido ({val}) en indicador activo '{ind_name}'."
                )
            inputs_by_id[input_id] = val
            active_inputs_by_name[ind_name] = val

        raw_out = self.network.activate(inputs_by_id)
        direction = float(raw_out[self._output_ids[0]])
        confidence = float(raw_out[self._output_ids[1]])
        size_scaling = float(raw_out[self._output_ids[2]])

        desired_pos = self._decode_signal(direction, confidence)
        eff_frac = float(
            self.genome.trading_params.position_size
            * np.clip(size_scaling, 0.2, 1.0)
        )

        return NeuralDecision(
            direction=direction,
            confidence=confidence,
            size_scaling=size_scaling,
            desired_position=desired_pos,
            effective_position_fraction=eff_frac,
            active_inputs=active_inputs_by_name,
        )

    def _decode_signal(
        self,
        direction_signal: float,
        confidence_signal: float,
    ) -> Optional[PositionType]:
        """
        Reglas idénticas a `Backtester._decode_signal`:
        - Confianza < umbral o zona neutral [-0.2, +0.2] -> `None` (HOLD).
        - Dirección > +0.2 y confianza >= 0.3 -> `PositionType.LONG`.
        - Dirección < -0.2 y confianza >= 0.3 -> `PositionType.CASH` en Spot
          (orden activa de venta) o `PositionType.SHORT` si `allow_short=True`.
        """
        if confidence_signal < self.confidence_threshold:
            return None
        if direction_signal > self.direction_threshold:
            return PositionType.LONG
        if direction_signal < -self.direction_threshold:
            return PositionType.SHORT if self.allow_short else PositionType.CASH
        return None


class PaperTrader:
    """
    Orquestador completo del motor diario de Paper Trading en vivo sobre Bitso.
    """

    def __init__(
        self,
        brain: ChampionBrain,
        bitso_client: BitsoRESTClient,
        pipeline: DataPipeline,
        auditor: RealFrictionAuditor,
        symbol: str = "BTC-MXN",
        initial_capital: float = 100_000.0,
        lookback_bars: int = 260,
        warmup_extra_bars: int = 0,
    ):
        self.brain = brain
        self.bitso_client = bitso_client
        self.pipeline = pipeline
        self.auditor = auditor
        self.symbol = symbol
        self.book = bitso_client.symbol_to_book(symbol)
        self.initial_capital = float(initial_capital)
        self.lookback_bars = int(lookback_bars)
        self.warmup_extra_bars = int(warmup_extra_bars)

    @classmethod
    def from_config(
        cls,
        config_path: str | Path = "config/trading_config.yaml",
        genome_override: Optional[str | Path] = None,
        book_override: Optional[str] = None,
        bars_override: Optional[int] = None,
        trades_csv_override: Optional[str | Path] = None,
        state_json_override: Optional[str | Path] = None,
        bitso_client: Optional[BitsoRESTClient] = None,
    ) -> "PaperTrader":
        """Construye `PaperTrader` leyendo `config/trading_config.yaml`."""
        with open(config_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}

        capital_cfg = cfg.get("capital", {})
        costs_cfg = cfg.get("costs", {})
        decision_cfg = cfg.get("decision", {})
        bitso_cfg_dict = dict(cfg.get("bitso_api", {}))
        paper_cfg = cfg.get("paper_trading", {})

        if book_override:
            bitso_cfg_dict["book"] = book_override
        if bars_override is not None:
            bitso_cfg_dict["lookback_bars"] = int(bars_override)

        bitso_config = BitsoConfig.from_dict(bitso_cfg_dict)
        client = bitso_client or BitsoRESTClient(config=bitso_config)

        # Derivar símbolo legible (ej. "btc_mxn" -> "BTC-MXN")
        symbol = bitso_config.book.upper().replace("_", "-")
        interval = "1d" if bitso_config.time_bucket >= 86400 else "1h"

        genome_path = genome_override or paper_cfg.get(
            "champion_genome_path",
            "runs/run_BTC_MXN_1d_20261107_125714_seed2026/window_1/best_genome.json",
        )
        fallback_path = (
            None
            if genome_override
            else paper_cfg.get(
                "fallback_genome_path",
                "models/best_genome_seed2026_w1.json",
            )
        )

        brain = ChampionBrain.from_json(
            genome_path=genome_path,
            fallback_path=fallback_path,
            confidence_threshold=float(decision_cfg.get("confidence_threshold", 0.30)),
            direction_threshold=float(decision_cfg.get("direction_threshold", 0.20)),
            allow_short=bool(decision_cfg.get("allow_short", False)),
        )

        pipeline = DataPipeline(
            symbol=symbol,
            interval=interval,
            normalization_method="zscore",
            normalization_window=252 if interval == "1d" else 720,
        )

        trades_csv = trades_csv_override or paper_cfg.get(
            "trades_csv_path", "live_trades.csv"
        )
        state_json = state_json_override or paper_cfg.get(
            "state_json_path", "live_state.json"
        )
        auditor = RealFrictionAuditor(
            trades_csv_path=trades_csv,
            state_json_path=state_json,
            commission=float(costs_cfg.get("commission", 0.0065)),
        )

        return cls(
            brain=brain,
            bitso_client=client,
            pipeline=pipeline,
            auditor=auditor,
            symbol=symbol,
            initial_capital=float(capital_cfg.get("initial", 100_000.0)),
            lookback_bars=bitso_config.lookback_bars,
            warmup_extra_bars=bitso_config.warmup_extra_bars,
        )

    def run_daily_cycle(
        self,
        force: bool = False,
        dry_run: bool = False,
        now_utc: Optional[datetime] = None,
    ) -> DailyCycleReport:
        """
        Ejecuta el ciclo diario al cierre de vela:
        1. Descarga las últimas `lookback_bars` (260) velas cerradas y normaliza sin leakage.
        2. Verifica idempotencia para no operar dos veces la misma vela diaria.
        3. Actualiza riesgos intrabarra (Stop Loss / Trailing Stop / Take Profit / MaxDD).
        4. Evalúa la red neuronal del campeón (`Direction`, `Confidence`, `Size`).
        5. Si hay orden de compra o venta, consulta el Order Book en tiempo real y
           registra la ejecución al `Ask` real (compra) o `Bid` real (venta) en `live_trades.csv`.
        """
        state = self.auditor.load_state(initial_capital=self.initial_capital)

        df_live = self.pipeline.get_live_data(
            bitso_client=self.bitso_client,
            lookback_bars=self.lookback_bars,
            warmup_extra_bars=self.warmup_extra_bars,
            exclude_open_candle=True,
            required_indicators=self.brain.active_indicator_names,
            now_utc=now_utc,
        )
        raw_bars_count = len(self.pipeline._raw_data) if self.pipeline._raw_data is not None else len(df_live)

        latest_ts = df_live.index[-1]
        candle_date = (
            latest_ts.strftime("%Y-%m-%d")
            if self.pipeline.interval in ("1d", "1D") and hasattr(latest_ts, "strftime")
            else str(latest_ts)
        )
        latest_row = df_live.iloc[-1]

        open_p = float(latest_row["Open"])
        high_p = float(latest_row["High"])
        low_p = float(latest_row["Low"])
        close_p = float(latest_row["Close"])

        # --- Protección de idempotencia ---
        if state.last_processed_candle == candle_date and not force:
            mtm_equity = self._compute_mark_to_market(state, close_p)
            logger.info(
                f"Vela {candle_date} ya fue procesada previamente. "
                f"Omitiendo ejecución duplicada (usa --force para re-evaluar)."
            )
            return DailyCycleReport(
                action="SKIPPED_ALREADY_PROCESSED",
                candle_date=candle_date,
                symbol=self.symbol,
                book=self.book,
                candle_close=close_p,
                decision=None,
                reason="already_processed",
                equity_mxn=mtm_equity,
                bars_loaded=raw_bars_count,
                dry_run=dry_run,
            )

        # --- Paso 1: Evaluar cerebro neuronal sobre la vela recién cerrada ---
        decision = self.brain.evaluate_row(latest_row)
        params = self.brain.genome.trading_params

        logger.info(
            f"[{self.symbol} | {candle_date}] Close={close_p:,.2f} MXN | "
            f"Red -> Dir={decision.direction:+.4f}, Conf={decision.confidence:.4f}, "
            f"Size={decision.size_scaling:.4f} (frac={decision.effective_position_fraction:.2%}) "
            f"=> Señal={decision.signal_label}"
        )

        # --- Paso 2: Gestión de riesgo sobre posición abierta ---
        risk_exit_reason: Optional[str] = None
        if state.open_position is not None:
            gap_reason = state.open_position.check_gap_exit(open_p)
            if gap_reason is not None:
                risk_exit_reason = gap_reason
            else:
                risk_exit_reason = state.open_position.update_bar(high_p, low_p, close_p)

        mtm_equity = self._compute_mark_to_market(state, close_p)
        state.peak_equity = max(state.peak_equity, mtm_equity)
        drawdown = (
            (state.peak_equity - mtm_equity) / state.peak_equity
            if state.peak_equity > 0
            else 1.0
        )
        if drawdown >= params.max_drawdown:
            state.trading_halted = True
            logger.warning(
                f"Cortacircuito Max Drawdown activado: drawdown={drawdown:.2%} "
                f">= límite={params.max_drawdown:.2%}"
            )

        # --- Paso 3: Resolver orden operativa y auditar fricción real ---
        report: DailyCycleReport

        if state.open_position is not None:
            # Determinar si corresponde VENDER (stop, max_drawdown o convicción bajista activa)
            sell_reason: Optional[str] = None
            if risk_exit_reason is not None:
                sell_reason = risk_exit_reason
            elif state.trading_halted:
                sell_reason = "max_drawdown"
            elif decision.desired_position == PositionType.CASH:
                sell_reason = "signal"

            if sell_reason is not None:
                if dry_run:
                    ob = self.bitso_client.fetch_order_book(book=self.book)
                    report = DailyCycleReport(
                        action="DRY_RUN_SELL",
                        candle_date=candle_date,
                        symbol=self.symbol,
                        book=self.book,
                        candle_close=close_p,
                        decision=decision,
                        reason=sell_reason,
                        execution_price=ob.best_bid,
                        best_bid=ob.best_bid,
                        best_ask=ob.best_ask,
                        spread_bps=ob.spread_bps,
                        equity_mxn=mtm_equity,
                        bars_loaded=raw_bars_count,
                        dry_run=True,
                    )
                else:
                    trade, ob, _ = self.auditor.execute_sell_at_real_bid(
                        state=state,
                        bitso_client=self.bitso_client,
                        symbol=self.symbol,
                        book=self.book,
                        candle_date=candle_date,
                        candle_close=close_p,
                        direction_signal=decision.direction,
                        confidence_signal=decision.confidence,
                        size_signal=decision.size_scaling,
                        reason=sell_reason,
                    )
                    report = DailyCycleReport(
                        action="SELL",
                        candle_date=candle_date,
                        symbol=self.symbol,
                        book=self.book,
                        candle_close=close_p,
                        decision=decision,
                        reason=sell_reason,
                        execution_price=ob.best_bid,
                        best_bid=ob.best_bid,
                        best_ask=ob.best_ask,
                        spread_bps=ob.spread_bps,
                        realized_pnl_mxn=trade.pnl,
                        equity_mxn=state.current_capital,
                        bars_loaded=raw_bars_count,
                        dry_run=False,
                    )
            else:
                # Zona neutral [-0.2, 0.2], confianza baja (<0.3) o señal LONG -> HOLD
                report = DailyCycleReport(
                    action="HOLD_LONG",
                    candle_date=candle_date,
                    symbol=self.symbol,
                    book=self.book,
                    candle_close=close_p,
                    decision=decision,
                    reason="hold_open_position",
                    equity_mxn=mtm_equity,
                    bars_loaded=raw_bars_count,
                    dry_run=dry_run,
                )

        else:
            # Actualmente en CASH: solo compramos con convicción alcista activa (LONG)
            if not state.trading_halted and decision.desired_position == PositionType.LONG:
                trade_capital = state.current_capital * decision.effective_position_fraction
                if trade_capital >= 100.0:
                    if dry_run:
                        ob = self.bitso_client.fetch_order_book(book=self.book)
                        report = DailyCycleReport(
                            action="DRY_RUN_BUY",
                            candle_date=candle_date,
                            symbol=self.symbol,
                            book=self.book,
                            candle_close=close_p,
                            decision=decision,
                            reason="signal_long",
                            execution_price=ob.best_ask,
                            best_bid=ob.best_bid,
                            best_ask=ob.best_ask,
                            spread_bps=ob.spread_bps,
                            equity_mxn=state.current_capital,
                            bars_loaded=raw_bars_count,
                            dry_run=True,
                        )
                    else:
                        _, ob, row_csv = self.auditor.execute_buy_at_real_ask(
                            state=state,
                            bitso_client=self.bitso_client,
                            symbol=self.symbol,
                            book=self.book,
                            candle_date=candle_date,
                            candle_close=close_p,
                            trade_capital_mxn=trade_capital,
                            stop_loss_pct=params.stop_loss,
                            take_profit_pct=params.take_profit,
                            trailing_stop_pct=params.trailing_stop,
                            direction_signal=decision.direction,
                            confidence_signal=decision.confidence,
                            size_signal=decision.size_scaling,
                            reason="signal_long",
                        )
                        report = DailyCycleReport(
                            action="BUY",
                            candle_date=candle_date,
                            symbol=self.symbol,
                            book=self.book,
                            candle_close=close_p,
                            decision=decision,
                            reason="signal_long",
                            execution_price=ob.best_ask,
                            best_bid=ob.best_bid,
                            best_ask=ob.best_ask,
                            spread_bps=ob.spread_bps,
                            equity_mxn=float(row_csv["equity_after_mxn"]),
                            bars_loaded=raw_bars_count,
                            dry_run=False,
                        )
                else:
                    report = DailyCycleReport(
                        action="HOLD_CASH",
                        candle_date=candle_date,
                        symbol=self.symbol,
                        book=self.book,
                        candle_close=close_p,
                        decision=decision,
                        reason="insufficient_capital",
                        equity_mxn=state.current_capital,
                        bars_loaded=raw_bars_count,
                        dry_run=dry_run,
                    )
            else:
                hold_reason = (
                    "trading_halted_max_drawdown"
                    if state.trading_halted
                    else "stay_in_cash"
                )
                report = DailyCycleReport(
                    action="HOLD_CASH",
                    candle_date=candle_date,
                    symbol=self.symbol,
                    book=self.book,
                    candle_close=close_p,
                    decision=decision,
                    reason=hold_reason,
                    equity_mxn=state.current_capital,
                    bars_loaded=raw_bars_count,
                    dry_run=dry_run,
                )

        if not dry_run:
            state.last_processed_candle = candle_date
            state.last_signal = decision.to_dict()
            self.auditor.save_state(state)

        return report

    @staticmethod
    def _compute_mark_to_market(state: PortfolioState, mark_price: float) -> float:
        if state.open_position is None:
            return float(state.current_capital)
        unrealized_pnl, _ = state.open_position.current_unrealized_pnl(mark_price)
        return float(state.current_capital + unrealized_pnl)

