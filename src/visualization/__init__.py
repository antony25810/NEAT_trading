"""
Modulo de visualizacion.

Componentes:
  - plot_equity_curve: grafico de equidad vs benchmark
  - plot_fitness_history: grafico de fitness por generacion
  - plot_drawdown: grafico de drawdown
"""

from src.visualization.plots import plot_equity_curve, plot_fitness_history, plot_drawdown

__all__ = [
    "plot_equity_curve",
    "plot_fitness_history",
    "plot_drawdown",
]
