"""
============================================================
main.py - Ejecutable Principal del Proyecto Neuroevolucion
============================================================

Punto de entrada principal para entrenar y evaluar el sistema
NEAT sobre pares de criptomonedas (BTC-MXN / ETH-MXN).

Los valores por defecto salen de config/neat_config.yaml y
config/trading_config.yaml. Los argumentos de linea de comandos
solo los SOBRESCRIBEN cuando se indican.

Uso:
----
  python main.py                                   # todo desde los YAML
  python main.py --generations 30 --pop_size 80    # prueba rapida
  python main.py --seeds 1 2 3 4 5 --workers 4     # varias semillas en paralelo
"""

import sys
import os
import json
import yaml
import argparse
import logging
from pathlib import Path
import pandas as pd

# Agregar ruta raiz
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.pipeline import DataPipeline
from src.evolution.engine import WalkForwardEngine
from src.visualization.plots import plot_equity_curve, plot_fitness_history, plot_drawdown

# Configurar logger profesional
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)8s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("NEAT-Trading")


def load_yaml_config(filepath: str) -> dict:
    """Carga un archivo de configuracion YAML."""
    path = Path(filepath)
    if not path.exists():
        logger.warning(f"Archivo de config no encontrado: {filepath}. Usando valores por defecto.")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _pick(cli_value, config_value, default):
    """Prioridad: argumento CLI > YAML > valor por defecto."""
    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    return default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sistema de Trading Neuroevolutivo (NEAT)")
    parser.add_argument("--symbol", type=str, default="BTC-MXN", help="Simbolo del activo (ej: BTC-MXN, ETH-MXN)")
    parser.add_argument("--interval", type=str, default="1d", help="Intervalo temporal ('1d' o '1h')")
    parser.add_argument("--start", type=str, default="2019-01-01", help="Fecha de inicio (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="Fecha de fin (YYYY-MM-DD, default: hoy)")
    parser.add_argument("--pop_size", type=int, default=None, help="Tamano de la poblacion NEAT (default: YAML)")
    parser.add_argument("--generations", type=int, default=None, help="Generaciones por ventana (default: YAML)")
    parser.add_argument("--patience", type=int, default=None, help="Paciencia de Early Stopping (default: YAML)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Semillas a ejecutar (default: YAML)")
    parser.add_argument("--workers", type=int, default=None, help="Procesos de evaluacion en paralelo (default: YAML)")
    parser.add_argument("--neat_config", type=str, default="config/neat_config.yaml")
    parser.add_argument("--trading_config", type=str, default="config/trading_config.yaml")
    return parser.parse_args()


def generate_plots(engine: WalkForwardEngine, summary: dict, args: argparse.Namespace):
    """Genera graficos de equidad, drawdown y fitness de una corrida."""
    run_folder = engine.run_folder
    oos_csv = run_folder / "oos_equity_curve.csv"
    if not oos_csv.exists():
        return

    equity_series = pd.read_csv(oos_csv, index_col=0, parse_dates=True).iloc[:, 0]

    bench_csv = run_folder / "oos_benchmarks.csv"
    benchmarks = None
    if bench_csv.exists():
        bench_df = pd.read_csv(bench_csv, index_col=0, parse_dates=True)
        benchmarks = {col: bench_df[col].dropna() for col in bench_df.columns}

    plot_equity_curve(
        strategy_equity=equity_series,
        benchmarks=benchmarks,
        title=f"Rendimiento Out-of-Sample NEAT ({args.symbol} {args.interval})",
        save_path=str(run_folder / "oos_equity_curve.png")
    )

    plot_drawdown(
        equity_series=equity_series,
        title=f"Drawdown Out-of-Sample NEAT ({args.symbol})",
        save_path=str(run_folder / "drawdown.png")
    )

    # Historial de fitness de cada ventana
    for win in summary.get("windows", []):
        hist = win.get("history", [])
        if hist:
            idx = win["window_index"]
            plot_fitness_history(
                history=hist,
                title=f"Evolucion del Fitness (Ventana {idx} - Train vs Val)",
                save_path=str(run_folder / f"window_{idx}_fitness.png")
            )

    logger.info(f"\n[EXITO] Graficos generados en: {run_folder}")


def print_run_summary(summary: dict, args: argparse.Namespace, run_folder: Path):
    """Imprime el resumen de una corrida (una semilla)."""
    oos_m = summary.get("global_oos_metrics", {})
    bench = summary.get("global_benchmark_metrics", {})
    print("\n" + "#" * 75)
    print(f"# RESUMEN DEL EXPERIMENTO (semilla {summary.get('seed')})")
    print("#" * 75)
    print(f"  Activo:             {args.symbol}")
    print(f"  Intervalo:          {args.interval}")
    print(f"  Ventanas OOS:       {summary.get('num_windows', 0)}")
    print(f"  {'':18}  {'Retorno':>10} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8}")
    rows = [("NEAT", oos_m)] + [
        ("Buy & Hold", bench.get("buy_and_hold", {})),
        ("Cruce SMAs", bench.get("sma_crossover", {})),
    ]
    for name, m in rows:
        print(
            f"  {name:18}  {m.get('total_return_pct', 0.0):>9.2f}% "
            f"{m.get('sharpe_ratio', 0.0):>8.2f} "
            f"{m.get('sortino_ratio', 0.0):>8.2f} "
            f"{m.get('max_drawdown_pct', 0.0):>7.2f}%"
        )
    print(f"  Carpeta de corrida: {run_folder}")
    print("#" * 75 + "\n")


def print_multi_seed_summary(results: list[dict], runs_dir: Path):
    """Agrega los resultados de varias semillas (media y desviacion)."""
    rows = []
    for res in results:
        oos = res["global_oos_metrics"]
        bh = res.get("global_benchmark_metrics", {}).get("buy_and_hold", {})
        rows.append({
            "seed": res["seed"],
            "neat_return_pct": oos.get("total_return_pct", 0.0),
            "neat_sharpe": oos.get("sharpe_ratio", 0.0),
            "neat_max_dd_pct": oos.get("max_drawdown_pct", 0.0),
            "bh_return_pct": bh.get("total_return_pct", 0.0),
            "bh_sharpe": bh.get("sharpe_ratio", 0.0),
        })
    df = pd.DataFrame(rows).set_index("seed")

    print("\n" + "=" * 75)
    print("  RESUMEN MULTI-SEMILLA")
    print("=" * 75)
    print(df.round(2).to_string())
    print("-" * 75)
    print("  Media:")
    print(df.mean().round(2).to_string())
    print("  Desviacion estandar:")
    print(df.std(ddof=1).round(2).to_string() if len(df) > 1 else "  (una sola semilla)")
    beats = int((df["neat_sharpe"] > df["bh_sharpe"]).sum())
    print(f"\n  Semillas donde NEAT supera el Sharpe de Buy & Hold: {beats}/{len(df)}")
    print("=" * 75 + "\n")

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out = runs_dir / f"multi_seed_summary_{timestamp}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "per_seed": df.reset_index().to_dict(orient="records"),
            "mean": df.mean().to_dict(),
            "std": df.std(ddof=1).fillna(0.0).to_dict(),
            "run_folders": [r["run_folder"] for r in results],
        }, f, indent=2)
    logger.info(f"Resumen multi-semilla guardado en: {out}")


def main():
    args = parse_args()

    print("\n" + "=" * 75)
    print("      PROYECTO NEUROEVOLUCION NEAT - ALGORITMOS GENETICOS Y TRADING")
    print("=" * 75)

    # 1. Cargar configuraciones YAML
    neat_config = load_yaml_config(args.neat_config)
    trading_config = load_yaml_config(args.trading_config)
    evo_cfg = neat_config.get("evolution", {})
    wf_cfg = trading_config.get("walk_forward", {})

    pop_size = _pick(args.pop_size, neat_config.get("population", {}).get("size"), 150)
    generations = _pick(args.generations, evo_cfg.get("generations"), 200)
    patience = _pick(args.patience, evo_cfg.get("early_stopping_patience"), 30)
    workers = _pick(args.workers, evo_cfg.get("workers"), 1)
    seeds = _pick(args.seeds, evo_cfg.get("seeds"), [42])
    min_generations = evo_cfg.get("min_generations", 20)

    logger.info(
        f"Config efectiva: pop={pop_size} | gens={generations} | "
        f"patience={patience} | workers={workers} | seeds={seeds}"
    )

    # Los datos se descargan una sola vez y se reutilizan entre semillas.
    pipeline = DataPipeline(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        interval=args.interval
    )

    runs_dir = Path("runs")
    results = []

    for seed in seeds:
        # 2. Inicializar motor Walk-Forward
        engine = WalkForwardEngine(
            symbol=args.symbol,
            start_date=args.start,
            end_date=args.end,
            interval=args.interval,
            pop_size=pop_size,
            generations=generations,
            early_stopping_patience=patience,
            runs_dir=str(runs_dir),
            neat_config=neat_config,
            trading_config=trading_config,
            seed=seed,
            workers=workers,
            min_generations=min_generations,
            pipeline=pipeline
        )

        # 3. Ejecutar experimento Walk-Forward
        summary = engine.run(
            train_years=wf_cfg.get("train_years", 3),
            val_years=wf_cfg.get("val_years", 1),
            test_years=wf_cfg.get("test_years", 1),
            step_years=wf_cfg.get("step_years", 1),
            mode=wf_cfg.get("mode", "sliding")
        )

        if not summary:
            logger.error(f"El experimento (semilla {seed}) no produjo resultados.")
            continue

        # 4. Graficos y resumen de la semilla
        generate_plots(engine, summary, args)
        print_run_summary(summary, args, engine.run_folder)
        summary["run_folder"] = str(engine.run_folder)
        results.append(summary)

    # 5. Resumen agregado si hubo varias semillas
    if len(results) > 1:
        print_multi_seed_summary(results, runs_dir)


if __name__ == "__main__":
    main()
