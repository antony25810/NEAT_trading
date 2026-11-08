"""
Modulo de trading: posiciones, metricas de rendimiento y backtesting.

Componentes:
  - Position, PositionType, TradeRecord: gestion de posiciones y trades
  - PerformanceMetrics, compute_metrics, calculate_fitness: evaluacion y fitness
  - Backtester: simulador de trading sobre datos de mercado
  - buy_and_hold, sma_crossover: estrategias de referencia
"""

from src.trading.position import Position, PositionType, TradeRecord
from src.trading.metrics import PerformanceMetrics, compute_metrics, calculate_fitness
from src.trading.backtester import Backtester
from src.trading.benchmarks import buy_and_hold, sma_crossover

__all__ = [
    "Position",
    "PositionType",
    "TradeRecord",
    "PerformanceMetrics",
    "compute_metrics",
    "calculate_fitness",
    "Backtester",
    "buy_and_hold",
    "sma_crossover",
]
