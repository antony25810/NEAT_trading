"""
============================================================
benchmarks.py - Estrategias de Referencia (Benchmarks)
============================================================

POR QUE BENCHMARKS?
-------------------
Un resultado de NEAT solo significa algo si se compara contra
alternativas SIMPLES evaluadas en el MISMO periodo y con los MISMOS
costos. Si la red no supera de forma consistente a:

  1. Buy & Hold: comprar al inicio y no hacer nada.
  2. Cruce de medias (SMA rapida > SMA lenta): una regla de tendencia clasica.

entonces la complejidad evolutiva no aporta "edge" real.

REGLAS DE SIMULACION (identicas al Backtester):
- La señal calculada al cierre de t-1 se ejecuta en la apertura de t.
- Comision y slippage en entrada y salida (via Position).
- Sin stop loss / take profit: solo se sigue la señal.
- Cualquier posicion abierta se cierra al ultimo cierre de la ventana.
"""

from typing import Tuple

import numpy as np
import pandas as pd

from src.trading.position import Position, PositionType, TradeRecord
from src.trading.metrics import PerformanceMetrics, compute_metrics


# Niveles "inalcanzables" para desactivar SL/TP/trailing en Position.
_NO_STOP = 1.0          # stop a precio 0 para un long
_NO_TAKE_PROFIT = 1e9   # objetivo practicamente infinito


def simulate_target_positions(
    df: pd.DataFrame,
    target: pd.Series,
    initial_capital: float = 100000.0,
    commission: float = 0.0065,
    slippage: float = 0.001,
    timeframe: str = "1d",
) -> Tuple[PerformanceMetrics, pd.Series, list[TradeRecord]]:
    """
    Simula una estrategia long/cash a partir de una posicion objetivo por barra.

    Parameters
    ----------
    df : pd.DataFrame
        Datos con columnas Open, High, Low, Close (precios reales)
    target : pd.Series
        Posicion objetivo YA DESFASADA que se ejecuta en la apertura de
        cada barra: 1 = estar comprado, 0 = estar en efectivo.
        Debe compartir indice con df.
    initial_capital, commission, slippage : float
        Mismo modelo de costos que el Backtester
    timeframe : str
        "1d" o "1h" (para anualizar metricas)

    Returns
    -------
    tuple
        (metricas, curva_de_equidad, trades)
    """
    if df.empty:
        empty = compute_metrics(pd.Series(dtype=float), [], initial_capital, timeframe)
        return empty, pd.Series(dtype=float), []

    target = target.reindex(df.index).fillna(0.0).astype(float).values
    opens = df["Open"].values
    closes = df["Close"].values
    dates = df.index

    capital = initial_capital
    position = None
    trades: list[TradeRecord] = []
    equity = np.empty(len(df))

    for i in range(len(df)):
        want_long = target[i] > 0.5

        # Ejecutar en la apertura
        if position is not None and not want_long:
            trade = position.close(opens[i], dates[i], "signal")
            trades.append(trade)
            capital += trade.pnl
            position = None
        elif position is None and want_long and capital > 0:
            position = Position(
                position_type=PositionType.LONG,
                entry_price=opens[i],
                entry_time=dates[i],
                capital=capital,
                commission=commission,
                slippage=slippage,
                stop_loss_pct=_NO_STOP,
                take_profit_pct=_NO_TAKE_PROFIT,
                trailing_stop_pct=_NO_STOP,
            )

        if position is not None:
            position.bars_held += 1
            unrealized, _ = position.current_unrealized_pnl(closes[i])
            equity[i] = capital + unrealized
        else:
            equity[i] = capital

    if position is not None:
        trade = position.close(closes[-1], dates[-1], "end_of_data")
        trades.append(trade)
        capital += trade.pnl
        equity[-1] = capital

    equity_series = pd.Series(equity, index=dates)
    metrics = compute_metrics(equity_series, trades, initial_capital, timeframe)
    return metrics, equity_series, trades


def buy_and_hold(
    df: pd.DataFrame,
    initial_capital: float = 100000.0,
    commission: float = 0.0065,
    slippage: float = 0.001,
    timeframe: str = "1d",
) -> Tuple[PerformanceMetrics, pd.Series, list[TradeRecord]]:
    """
    Compra en la primera apertura de la ventana y vende en el ultimo cierre.

    Paga una comision + slippage de entrada y otra de salida.
    """
    target = pd.Series(1.0, index=df.index)
    return simulate_target_positions(
        df, target, initial_capital, commission, slippage, timeframe
    )


def sma_crossover_target(
    close_history: pd.Series,
    fast: int = 50,
    slow: int = 200,
) -> pd.Series:
    """
    Posicion objetivo de un cruce de medias, ya desfasada una barra.

    Se calcula sobre TODA la historia de cierres (las medias necesitan
    `slow` barras previas) y luego se recorta a la ventana deseada.
    El shift(1) garantiza que la señal del cierre t-1 se ejecute en t.
    """
    if fast >= slow:
        raise ValueError("sma_fast debe ser menor que sma_slow")
    sma_fast = close_history.rolling(fast, min_periods=fast).mean()
    sma_slow = close_history.rolling(slow, min_periods=slow).mean()
    signal = (sma_fast > sma_slow).astype(float)
    signal[sma_slow.isna()] = 0.0
    return signal.shift(1).fillna(0.0)


def sma_crossover(
    df: pd.DataFrame,
    close_history: pd.Series,
    fast: int = 50,
    slow: int = 200,
    initial_capital: float = 100000.0,
    commission: float = 0.0065,
    slippage: float = 0.001,
    timeframe: str = "1d",
) -> Tuple[PerformanceMetrics, pd.Series, list[TradeRecord]]:
    """
    Largo cuando SMA(fast) > SMA(slow), efectivo en caso contrario.

    Parameters
    ----------
    df : pd.DataFrame
        Ventana a evaluar (ej. window.test)
    close_history : pd.Series
        Cierres de toda la historia disponible HASTA el final de la ventana
        (solo pasado: las medias moviles son causales).
    """
    target = sma_crossover_target(close_history, fast, slow).reindex(df.index)
    return simulate_target_positions(
        df, target, initial_capital, commission, slippage, timeframe
    )
