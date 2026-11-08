"""
============================================================
backtester.py - Simulador de Trading y Backtesting
============================================================

Ejecuta una simulacion realista de trading sobre datos historicos
usando las señales generadas por la red neuronal de un genoma NEAT.

REGLAS DE SIMULACION:
1. Las decisiones del cierre t-1 se ejecutan en la apertura t.
2. Los gaps se ejecutan al precio de apertura; después se evalúa el rango intrabar.
3. Se cobra comision (ej: 0.65%) y se aplica slippage (ej: 0.1%) tanto en entrada como en salida.
4. El tamaño de la posicion respeta `position_size` del genoma y `position_size_scaling` del output de la red.
5. El max drawdown del genoma detiene nuevas entradas para el resto de la ventana.
6. Si `allow_short=False` (spot, ej. Bitso) una señal bajista significa quedarse en efectivo.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple

from src.neat.genome import Genome
from src.neat.network import FeedForwardNetwork
from src.trading.position import Position, PositionType, TradeRecord
from src.trading.metrics import compute_metrics, calculate_fitness, PerformanceMetrics


class Backtester:
    """
    Motor de backtesting para evaluar un genoma NEAT sobre datos de mercado.
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission: float = 0.0065,
        slippage: float = 0.001,
        confidence_threshold: float = 0.30,
        timeframe: str = "1d",
        min_trades: int = 10,
        fitness_config: Optional[dict] = None,
        allow_short: bool = False,
        direction_threshold: float = 0.2
    ):
        """
        Parameters
        ----------
        initial_capital : float
            Capital inicial en MXN
        commission : float
            Comision porcentual por operacion (0.0065 = 0.65%)
        slippage : float
            Deslizamiento de precio (0.001 = 0.1%)
        confidence_threshold : float
            Minima confianza producida por la red para abrir posicion
        timeframe : str
            "1d" o "1h"
        min_trades : int
            Minimo de trades deseados para evaluar la estrategia
        allow_short : bool
            Si False, las señales de venta en corto se convierten en CASH.
            Un exchange spot (Bitso) no permite vender lo que no se tiene.
        direction_threshold : float
            Valor absoluto minimo del output de direccion para operar.
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.confidence_threshold = confidence_threshold
        self.timeframe = timeframe
        self.min_trades = min_trades
        self.fitness_config = fitness_config or {}
        self.allow_short = allow_short
        self.direction_threshold = direction_threshold

    def evaluate_genome(
        self,
        genome: Genome,
        df_data: pd.DataFrame,
        indicator_cols: list[str]
    ) -> Tuple[float, PerformanceMetrics, pd.Series, list[TradeRecord]]:
        """
        Evalua un genoma NEAT sobre un DataFrame procesado de datos.

        Parameters
        ----------
        genome : Genome
            Genoma a evaluar
        df_data : pd.DataFrame
            DataFrame procesado que contiene columnas de precios OHLCV
            y columnas de indicadores normalizados
        indicator_cols : list[str]
            Lista de nombres de columnas que corresponden a los indicadores

        Returns
        -------
        tuple[float, PerformanceMetrics, pd.Series, list[TradeRecord]]
            (fitness, metricas, curva_de_equidad, lista_de_trades)
        """
        if df_data.empty or len(df_data) < 30:
            empty_met = compute_metrics(pd.Series(), [], self.initial_capital, self.timeframe)
            return -100.0, empty_met, pd.Series(), []

        # --- Paso 1: Mapear inputs del genoma con las columnas del DataFrame ---
        ordered_indicators = [
            (input_id, name)
            for input_id, name in enumerate(genome.indicator_mask)
            if genome.indicator_mask[name]
        ]
        if not ordered_indicators:
            empty_met = compute_metrics(pd.Series(), [], self.initial_capital, self.timeframe)
            return -100.0, empty_met, pd.Series(), []

        # Obtener los indices de las columnas en df_data que se usaran como inputs
        input_matrices = []
        for _, name in ordered_indicators:
            if name in df_data.columns:
                input_matrices.append(df_data[name].values)
            else:
                # Si falta la columna, usar ceros
                input_matrices.append(np.zeros(len(df_data)))

        if not input_matrices:
            empty_met = compute_metrics(pd.Series(), [], self.initial_capital, self.timeframe)
            return -100.0, empty_met, pd.Series(), []

        # Matriz de shape (num_bars, num_active_inputs)
        input_batch = np.column_stack(input_matrices)

        # --- Paso 2: Generar salidas de la red neuronal vectorizadas ---
        network = FeedForwardNetwork.from_genome(genome)
        network_outputs = network.activate_batch(input_batch)
        # network_outputs tiene shape (num_bars, 3)
        # Columna 0: Direction [-1, 1]
        # Columna 1: Confidence [0, 1]
        # Columna 2: Position Size Scaling [0, 1]

        # --- Paso 3: Bucle de simulacion barra por barra ---
        dates = df_data.index
        opens = df_data["Open"].values
        highs = df_data["High"].values
        lows = df_data["Low"].values
        closes = df_data["Close"].values

        current_capital = self.initial_capital
        current_position: Optional[Position] = None
        trades: list[TradeRecord] = []
        equity_history = []
        peak_equity = self.initial_capital
        trading_halted = False
        pending_type: Optional[PositionType] = None
        pending_size = 0.0

        params = genome.trading_params

        for i in range(len(df_data)):
            date = dates[i]
            open_p = opens[i]
            high_p = highs[i]
            low_p = lows[i]
            close_p = closes[i]

            risk_exit_this_bar = False

            # 1. Resolver gaps de una posicion mantenida desde la barra previa.
            if current_position is not None:
                gap_reason = current_position.check_gap_exit(open_p)
                if gap_reason is not None:
                    trade = current_position.close(open_p, date, gap_reason)
                    trades.append(trade)
                    current_capital += trade.pnl
                    current_position = None
                    risk_exit_this_bar = True

            # 2. Ejecutar en la apertura la señal calculada al cierre anterior.
            if trading_halted:
                # Cortacircuito por Max Drawdown: liquidar inmediatamente
                if current_position is not None:
                    trade = current_position.close(open_p, date, "max_drawdown")
                    trades.append(trade)
                    current_capital += trade.pnl
                    current_position = None
            elif pending_type is not None:
                # Orden activa recibida (CASH, LONG o SHORT)
                if pending_type == PositionType.CASH:
                    # Salida activa a efectivo (convicción bajista en spot)
                    if current_position is not None:
                        trade = current_position.close(open_p, date, "signal")
                        trades.append(trade)
                        current_capital += trade.pnl
                        current_position = None
                else:
                    # pending_type es LONG o SHORT
                    if (
                        current_position is not None
                        and current_position.position_type != pending_type
                    ):
                        trade = current_position.close(open_p, date, "signal")
                        trades.append(trade)
                        current_capital += trade.pnl
                        current_position = None

                    if (
                        current_position is None
                        and current_capital > 0
                        and not risk_exit_this_bar
                    ):
                        frac = params.position_size * np.clip(pending_size, 0.2, 1.0)
                        trade_capital = current_capital * frac

                        if trade_capital >= 100.0:
                            current_position = Position(
                                position_type=pending_type,
                                entry_price=open_p,
                                entry_time=date,
                                capital=trade_capital,
                                commission=self.commission,
                                slippage=self.slippage,
                                stop_loss_pct=params.stop_loss,
                                take_profit_pct=params.take_profit,
                                trailing_stop_pct=params.trailing_stop
                            )
            # Si pending_type is None: HOLD. Se mantiene la posición o se permanece en CASH.

            # 3. Evaluar el rango intrabar con prioridad conservadora.
            if current_position is not None:
                exit_reason = current_position.update_bar(
                    high_p, low_p, close_p
                )
                if exit_reason is not None:
                    if exit_reason == "stop_loss":
                        exit_price = current_position.stop_loss_price
                    elif exit_reason == "trailing_stop":
                        exit_price = current_position.trailing_stop_price
                    else:
                        exit_price = current_position.take_profit_price
                    trade = current_position.close(
                        exit_price, date, exit_reason
                    )
                    trades.append(trade)
                    current_capital += trade.pnl
                    current_position = None
                    risk_exit_this_bar = True

            # 4. Marcar a mercado y activar el cortacircuito de drawdown.
            if current_position is not None:
                unrealized_pnl, _ = current_position.current_unrealized_pnl(close_p)
                total_equity = current_capital + unrealized_pnl
            else:
                total_equity = current_capital

            equity_history.append(total_equity)
            peak_equity = max(peak_equity, total_equity)
            drawdown = (
                (peak_equity - total_equity) / peak_equity
                if peak_equity > 0 else 1.0
            )
            if drawdown >= params.max_drawdown:
                trading_halted = True

            # 5. La señal de esta barra queda pendiente para la próxima apertura.
            if trading_halted:
                pending_type = PositionType.CASH
                pending_size = 0.0
            else:
                pending_type, pending_size = self._decode_signal(
                    network_outputs[i]
                )

        # 6. Al finalizar los datos, cerrar cualquier posicion restante
        if current_position is not None:
            final_date = dates[-1]
            final_close = closes[-1]
            exit_reason = "max_drawdown" if trading_halted else "end_of_data"
            trade = current_position.close(final_close, final_date, exit_reason)
            trades.append(trade)
            current_capital += trade.pnl
            equity_history[-1] = current_capital

        equity_series = pd.Series(equity_history, index=dates)

        # --- Paso 4: Calcular metricas y fitness ---
        metrics = compute_metrics(
            equity_series,
            trades,
            initial_capital=self.initial_capital,
            timeframe=self.timeframe
        )

        supported_fitness_keys = {
            "complexity_penalty_coeff",
            "max_annual_turnover",
            "drawdown_soft_limit",
        }
        fitness_kwargs = {
            key: value
            for key, value in self.fitness_config.items()
            if key in supported_fitness_keys
        }
        fitness = calculate_fitness(
            metrics,
            genome,
            min_trades=self.min_trades,
            **fitness_kwargs
        )

        return fitness, metrics, equity_series, trades

    def _decode_signal(
        self,
        outputs: np.ndarray
    ) -> tuple[Optional[PositionType], float]:
        """
        Convierte las salidas de red en una orden para la barra siguiente.

        Returns
        -------
        tuple[Optional[PositionType], float]
            - PositionType.LONG: Señal alcista con convicción
            - PositionType.SHORT: Señal bajista con convicción (en futuros / allow_short=True)
            - PositionType.CASH: Orden activa de salida a efectivo (convicción bajista en spot)
            - None: Zona neutral o indecisión (confianza baja) -> HOLD
        """
        direction_signal, confidence_signal, size_signal = outputs
        if confidence_signal < self.confidence_threshold:
            return None, float(size_signal)
        if direction_signal > self.direction_threshold:
            return PositionType.LONG, float(size_signal)
        if direction_signal < -self.direction_threshold:
            if self.allow_short:
                return PositionType.SHORT, float(size_signal)
            # Spot: conviccion bajista significa orden activa de salir a efectivo.
            return PositionType.CASH, float(size_signal)
        # Zona neutral [-direction_threshold, +direction_threshold] -> HOLD
        return None, float(size_signal)
