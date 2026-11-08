"""
============================================================
metrics.py - Metricas Financieras y Funcion Fitness
============================================================

Este modulo calcula todas las metricas de rendimiento para un backtest
y combina esos resultados con la penalizacion por complejidad del genoma
para generar la funcion de FITNESS que guia la neuroevolucion NEAT.

METRICAS INCLUIDAS:
- Retorno Total (%)
- Sharpe Ratio (anualizado)
- Sortino Ratio (anualizado, enfocado en volatilidad negativa)
- Maximo Drawdown (%)
- Calmar Ratio (Retorno Anualizado / Max Drawdown)
- Win Rate (% de trades ganadores)
- Profit Factor (Ganancias Brutas / Perdidas Brutas)
- Numero de trades
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

from src.neat.genome import Genome
from src.trading.position import TradeRecord


@dataclass
class PerformanceMetrics:
    """
    Contenedor con todas las metricas de rendimiento de un backtest.
    """
    total_return: float        # Retorno total porcentual (0.15 = +15%)
    annualized_return: float   # Retorno anualizado
    sharpe_ratio: float        # Sharpe Ratio (retorno/volatilidad total)
    sortino_ratio: float       # Sortino Ratio (retorno/volatilidad a la baja)
    max_drawdown: float        # Drawdown maximo (0.20 = 20%)
    calmar_ratio: float        # Calmar Ratio
    win_rate: float            # % de trades ganadores (0.55 = 55%)
    profit_factor: float       # Ganancias / Perdidas
    num_trades: int            # Numero total de trades cerrados
    avg_trade_pnl: float       # PnL promedio por trade %
    avg_bars_held: float       # Duracion promedio de trades en barras
    annual_turnover: float     # Capital negociado / capital inicial por ano
    final_equity: float        # Capital final en MXN

    def to_dict(self) -> dict:
        return {
            "total_return_pct": round(self.total_return * 100, 2),
            "annualized_return_pct": round(self.annualized_return * 100, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "sortino_ratio": round(self.sortino_ratio, 3),
            "max_drawdown_pct": round(self.max_drawdown * 100, 2),
            "calmar_ratio": round(self.calmar_ratio, 3),
            "win_rate_pct": round(self.win_rate * 100, 2),
            "profit_factor": round(self.profit_factor, 3),
            "num_trades": self.num_trades,
            "avg_trade_pnl_pct": round(self.avg_trade_pnl * 100, 2),
            "avg_bars_held": round(self.avg_bars_held, 1),
            "annual_turnover": round(self.annual_turnover, 2),
            "final_equity": round(self.final_equity, 2),
        }


def compute_metrics(
    equity_curve: pd.Series,
    trades: list[TradeRecord],
    initial_capital: float = 100000.0,
    timeframe: str = "1d"
) -> PerformanceMetrics:
    """
    Calcula las metricas completas de rendimiento desde la curva de equidad.

    Parameters
    ----------
    equity_curve : pd.Series
        Serie temporal del capital acumulado en cada barra
    trades : list[TradeRecord]
        Lista de todos los trades cerrados durante el backtest
    initial_capital : float
        Capital inicial
    timeframe : str
        "1d" (365 barras/ano para crypto) o "1h" (8760 barras/ano)

    Returns
    -------
    PerformanceMetrics
    """
    if len(equity_curve) == 0:
        return _empty_metrics(initial_capital)

    final_equity = float(equity_curve.iloc[-1])
    total_return = (final_equity - initial_capital) / initial_capital

    # Factor de anualización
    ann_factor = 365.0 if timeframe in ("1d", "1D") else 8760.0
    num_bars = len(equity_curve)
    years = max(num_bars / ann_factor, 1.0 / ann_factor)

    # Retorno anualizado
    if final_equity > 0:
        annualized_return = (final_equity / initial_capital) ** (1.0 / years) - 1.0
    else:
        annualized_return = -1.0

    # Retornos periodicos (por barra)
    returns = equity_curve.pct_change().dropna()

    if len(returns) < 2 or returns.std() == 0:
        sharpe_ratio = 0.0
        sortino_ratio = 0.0
    else:
        # Sharpe Ratio (asumiendo risk-free rate = 0)
        mean_ret = returns.mean()
        std_ret = returns.std()
        sharpe_ratio = float((mean_ret / std_ret) * np.sqrt(ann_factor))

        # Sortino Ratio (Target Downside Deviation con T=0 sobre todas las barras)
        downside_diff = np.minimum(0.0, returns.values)
        downside_std = float(np.sqrt(np.mean(downside_diff ** 2)))
        if downside_std < 1e-8:
            sortino_ratio = sharpe_ratio
        else:
            sortino_ratio = float((mean_ret / downside_std) * np.sqrt(ann_factor))

    # Maximo Drawdown
    cummax = equity_curve.cummax()
    drawdowns = (equity_curve - cummax) / cummax
    max_drawdown = abs(float(drawdowns.min()))

    # Calmar Ratio
    calmar_ratio = (annualized_return / max_drawdown) if max_drawdown > 0 else 0.0

    # Metricas de trades
    num_trades = len(trades)
    if num_trades > 0:
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl < 0]

        win_rate = len(winning_trades) / num_trades

        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))

        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        else:
            profit_factor = 10.0 if gross_profit > 0 else 1.0

        avg_trade_pnl = sum(t.pnl_pct for t in trades) / num_trades
        avg_bars_held = sum(t.bars_held for t in trades) / num_trades
        annual_turnover = (
            sum(t.capital_invested for t in trades)
            / initial_capital
            / years
        )
    else:
        win_rate = 0.0
        profit_factor = 0.0
        avg_trade_pnl = 0.0
        avg_bars_held = 0.0
        annual_turnover = 0.0

    return PerformanceMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        max_drawdown=max_drawdown,
        calmar_ratio=calmar_ratio,
        win_rate=win_rate,
        profit_factor=min(profit_factor, 50.0),
        num_trades=num_trades,
        avg_trade_pnl=avg_trade_pnl,
        avg_bars_held=avg_bars_held,
        annual_turnover=annual_turnover,
        final_equity=final_equity
    )


def calculate_fitness(
    metrics: PerformanceMetrics,
    genome: Genome,
    min_trades: int = 10,
    complexity_penalty_coeff: float = 0.004,
    max_annual_turnover: float = 24.0,
    drawdown_soft_limit: float = 0.20
) -> float:
    """
    Calcula el fitness escalar para el algoritmo NEAT.

    Combina componentes acotados de Sortino, Calmar, retorno anualizado
    y Profit Factor. Penaliza drawdown de forma cuadratica, actividad
    insuficiente, turnover excesivo y complejidad estructural.

    Parameters
    ----------
    metrics : PerformanceMetrics
        Metricas calculadas del backtest
    genome : Genome
        Genoma que genero esta estrategia (para medir complejidad)
    min_trades : int
        Minimo de trades requeridos para no sufrir penalizacion
    complexity_penalty_coeff : float
        Coeficiente de penalizacion por cada gen activo

    Returns
    -------
    float
        Valor de fitness (puede ser negativo si rinde muy mal)
    """
    # Casos inviables conservan gradiente respecto a otros individuos.
    if metrics.final_equity <= 0 or metrics.num_trades == 0:
        return -5.0 - complexity_penalty_coeff * genome.complexity_score

    # Componentes acotados y comparables en [-1, 1].
    sortino_component = np.tanh(metrics.sortino_ratio / 2.0)
    calmar_component = np.tanh(metrics.calmar_ratio / 2.0)
    return_component = np.tanh(metrics.annualized_return / 0.25)
    profit_factor_component = np.tanh(
        np.log(max(metrics.profit_factor, 0.10))
    )

    base_score = (
        0.35 * sortino_component
        + 0.25 * calmar_component
        + 0.25 * return_component
        + 0.15 * profit_factor_component
    )

    trade_adequacy = min(metrics.num_trades / max(min_trades, 1), 1.0)
    trade_penalty = 1.5 * (1.0 - trade_adequacy) ** 2

    drawdown_ratio = metrics.max_drawdown / max(drawdown_soft_limit, 1e-6)
    drawdown_penalty = 0.50 * min(drawdown_ratio, 2.0) ** 2

    turnover_excess = max(
        0.0,
        metrics.annual_turnover - max_annual_turnover
    )
    turnover_penalty = min(
        turnover_excess / max(max_annual_turnover, 1e-6),
        1.0
    ) * 0.25

    complexity_penalty = (
        complexity_penalty_coeff * genome.complexity_score
    )

    fitness = (
        base_score
        - trade_penalty
        - drawdown_penalty
        - turnover_penalty
        - complexity_penalty
    )

    return float(fitness)


def _empty_metrics(initial_capital: float) -> PerformanceMetrics:
    """Metricas por defecto si no hay datos."""
    return PerformanceMetrics(
        total_return=0.0,
        annualized_return=0.0,
        sharpe_ratio=0.0,
        sortino_ratio=0.0,
        max_drawdown=0.0,
        calmar_ratio=0.0,
        win_rate=0.0,
        profit_factor=0.0,
        num_trades=0,
        avg_trade_pnl=0.0,
        avg_bars_held=0.0,
        annual_turnover=0.0,
        final_equity=initial_capital
    )
