"""
============================================================
plots.py - Visualizacion y Gráficos del Experimento
============================================================

Proporciona funciones para generar gráficos descriptivos usando Matplotlib/Seaborn:
1. Curvas de Equidad (Estrategia OOS vs Buy & Hold)
2. Evolucion del Fitness por Generación (Train vs Val)
3. Grafico de Drawdown
4. Distribucion de Trades
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

# Configurar estilo visual profesional
plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def plot_equity_curve(
    strategy_equity: pd.Series,
    benchmark_prices: Optional[pd.Series] = None,
    title: str = "Curva de Equidad Out-of-Sample",
    save_path: Optional[str] = None,
    benchmarks: Optional[dict[str, pd.Series]] = None
):
    """
    Grafica la curva de equidad de la estrategia comparada con Buy & Hold.

    Parameters
    ----------
    strategy_equity : pd.Series
        Serie de capital acumulado de la estrategia
    benchmark_prices : pd.Series, opcional
        Precios del activo base para calcular Buy & Hold (sin costos).
        Se ignora si se pasa `benchmarks`.
    title : str
        Titulo del grafico
    save_path : str, opcional
        Ruta donde guardar el grafico PNG
    benchmarks : dict[str, pd.Series], opcional
        Curvas de equidad de estrategias de referencia ya simuladas con
        costos (ej. {"buy_and_hold": ..., "sma_crossover": ...}).
    """
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

    # Normalizar estrategia
    initial_strat = strategy_equity.iloc[0] if len(strategy_equity) > 0 else 1.0
    norm_strat = (strategy_equity / initial_strat) * 100.0

    ax.plot(norm_strat.index, norm_strat.values, label="NEAT Strategy", color="#1f77b4", linewidth=2.0)

    bench_styles = {
        "buy_and_hold": ("Buy & Hold (con costos)", "#ff7f0e", "--"),
        "sma_crossover": ("Cruce de SMAs (con costos)", "#2ca02c", ":"),
    }

    if benchmarks:
        for name, equity in benchmarks.items():
            if equity is None or len(equity) == 0:
                continue
            label, color, style = bench_styles.get(name, (name, None, "-."))
            norm = (equity / equity.iloc[0]) * 100.0
            ax.plot(norm.index, norm.values, label=label, color=color,
                    linestyle=style, linewidth=1.5)
    # Benchmark (Buy & Hold)
    elif benchmark_prices is not None and not benchmark_prices.empty:
        # Alinear indices
        common_idx = strategy_equity.index.intersection(benchmark_prices.index)
        if len(common_idx) > 0:
            bench_aligned = benchmark_prices.loc[common_idx]
            norm_bench = (bench_aligned / bench_aligned.iloc[0]) * 100.0
            ax.plot(norm_bench.index, norm_bench.values, label="Buy & Hold (Benchmark)", color="#ff7f0e", linestyle="--", linewidth=1.5)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Capital Normalizado (Base = 100)", fontsize=11)
    ax.set_xlabel("Fecha", fontsize=11)
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_fitness_history(
    history: list[dict],
    title: str = "Evolucion del Fitness por Generación",
    save_path: Optional[str] = None
):
    """
    Grafica la evolucion del fitness en Train y Validation por generacion.

    Parameters
    ----------
    history : list[dict]
        Lista de diccionarios devueltos por Population.evolve()
    title : str
        Titulo del grafico
    save_path : str, opcional
        Ruta donde guardar el grafico
    """
    if not history:
        return

    df_hist = pd.DataFrame(history)

    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=150)

    gens = df_hist["generation"]

    # Fitness Train Best vs Val Fitness
    if "best_fitness" in df_hist.columns:
        ax1.plot(gens, df_hist["best_fitness"], label="Train Best Fitness", color="#2ca02c", linewidth=2.0)

    if "val_fitness" in df_hist.columns:
        ax1.plot(gens, df_hist["val_fitness"], label="Val Mediana Top-K", color="#d62728", linestyle="--", linewidth=2.0)

    if "val_smoothed" in df_hist.columns:
        ax1.plot(gens, df_hist["val_smoothed"], label="Val Suavizada (early stopping)", color="#8c564b", linewidth=1.5)

    if "avg_fitness" in df_hist.columns:
        ax1.plot(gens, df_hist["avg_fitness"], label="Train Avg Fitness", color="#7f7f7f", linestyle=":", alpha=0.7)

    ax1.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax1.set_xlabel("Generación", fontsize=11)
    ax1.set_ylabel("Fitness Score", fontsize=11)
    ax1.legend(loc="upper left", frameon=True, facecolor="white")

    # Especie count en segundo eje Y
    if "num_species" in df_hist.columns:
        ax2 = ax1.twinx()
        ax2.plot(gens, df_hist["num_species"], color="#9467bd", linestyle="-.", alpha=0.5, label="Numero de Especies")
        ax2.set_ylabel("Especies", color="#9467bd", fontsize=11)
        ax2.grid(False)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_drawdown(
    equity_series: pd.Series,
    title: str = "Grafico de Drawdown",
    save_path: Optional[str] = None
):
    """
    Grafica la curva de drawdown en porcentaje.

    Parameters
    ----------
    equity_series : pd.Series
        Serie de equidad de la estrategia
    """
    if len(equity_series) == 0:
        return

    cummax = equity_series.cummax()
    drawdown_pct = ((equity_series - cummax) / cummax) * 100.0

    fig, ax = plt.subplots(figsize=(12, 4), dpi=150)

    ax.fill_between(drawdown_pct.index, drawdown_pct.values, 0, color="#d62728", alpha=0.4, label="Drawdown (%)")
    ax.plot(drawdown_pct.index, drawdown_pct.values, color="#d62728", linewidth=1.0)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Drawdown (%)", fontsize=11)
    ax.set_xlabel("Fecha", fontsize=11)
    ax.set_ylim(bottom=min(-35.0, drawdown_pct.min() * 1.1), top=1.0)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
