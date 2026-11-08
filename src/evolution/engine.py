"""
============================================================
engine.py - Motor de Evolucion Walk-Forward End-to-End
============================================================

Este modulo coordina el entrenamiento evolutivo del sistema a traves
de multiples ventanas Walk-Forward (Sliding u Expanding).

FLUJO DE TRABAJO POR VENTANA WALK-FORWARD:
1. Tomar ventana: Train (ej: 3 anos), Validation (1 ano), Test (1 ano).
2. Inicializar Poblacion NEAT.
3. Para cada Generacion (hasta N generaciones o early stopping):
   a. Evaluar cada genoma en datos de Train usando Backtester.
   b. Evaluar los K mejores de Train en Validation.
   c. Registrar candidatos con un puntaje CONSERVADOR: min(train, val).
   d. Early stopping sobre la MEDIANA suavizada de validacion (no el maximo).
   e. Evolucionar la poblacion (especiacion + cruce + mutacion).
4. Al finalizar la ventana, tomar el candidato con mejor puntaje conservador.
5. Evaluar ese genoma en la ventana de TEST (Out-of-Sample estricto).
6. Evaluar Buy & Hold y cruce de SMAs en el mismo TEST como referencia.
7. Guardar metricas, equidad y genoma de la ventana.
8. Encadenar los resultados de Test de todas las ventanas para obtener el rendimiento global.

POR QUE NO ELEGIR EL MAXIMO DE VALIDACION?
------------------------------------------
Con K candidatos por generacion y G generaciones se hacen ~K*G
comparaciones sobre el MISMO año de validacion. El maximo de todas ellas
esta inflado por suerte (sesgo de seleccion) y la validacion se convierte
en parte del entrenamiento. Usar min(train, val) y la mediana suavizada
reduce ese sesgo. El TEST sigue sin tocarse hasta el final.
"""

import json
import logging
import random
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd

from src.data.pipeline import DataPipeline
from src.evolution.evaluation import GenomeEvaluator, selection_score
from src.neat.population import Population
from src.neat.genome import Genome
from src.trading.backtester import Backtester
from src.trading.benchmarks import buy_and_hold, sma_crossover
from src.trading.metrics import compute_metrics
from src.trading.position import TradeRecord

logger = logging.getLogger(__name__)


def chain_equities(
    equities: List[pd.Series],
    initial_capital: float
) -> pd.Series:
    """
    Encadena curvas de equidad de multiples ventanas OOS.

    Cada ventana de test arranca con `initial_capital`, por lo que el
    retorno de la PRIMERA barra de cada ventana se mide contra ese capital
    inicial (antes se perdia porque pct_change() lo dejaba en 0).
    """
    chained_values = []
    chained_indices = []
    current_cap = initial_capital

    for eq in equities:
        if eq is None or len(eq) == 0:
            continue
        returns = eq.pct_change()
        returns.iloc[0] = eq.iloc[0] / initial_capital - 1.0
        for date, ret in returns.items():
            current_cap *= (1.0 + ret)
            chained_indices.append(date)
            chained_values.append(current_cap)

    return pd.Series(chained_values, index=chained_indices, dtype=float)


class WalkForwardEngine:
    """
    Motor de ejecucion de experimentos Walk-Forward de Neuroevolucion.
    """

    def __init__(
        self,
        symbol: str = "BTC-MXN",
        start_date: str = "2019-01-01",
        end_date: Optional[str] = None,
        interval: str = "1d",
        pop_size: int = 100,
        generations: int = 50,
        early_stopping_patience: int = 15,
        runs_dir: str = "runs",
        neat_config: Optional[Dict] = None,
        trading_config: Optional[Dict] = None,
        seed: Optional[int] = None,
        workers: int = 1,
        min_generations: int = 20,
        pipeline: Optional[DataPipeline] = None
    ):
        """
        Parameters
        ----------
        symbol : str
            Simbolo del activo (ej: "BTC-MXN")
        start_date : str
            Fecha de inicio de datos
        end_date : str, opcional
            Fecha fin
        interval : str
            "1d" o "1h"
        pop_size : int
            Tamano de la poblacion NEAT
        generations : int
            Maximo de generaciones por ventana Walk-Forward
        early_stopping_patience : int
            Generaciones sin mejora en Validation antes de detener
        runs_dir : str
            Directorio raiz para guardar experimentos
        neat_config : dict, opcional
            Configuracion NEAT
        trading_config : dict, opcional
            Configuracion de trading
        seed : int, opcional
            Semilla para random y numpy (reproducibilidad)
        workers : int
            Procesos para evaluar genomas en paralelo (1 = secuencial)
        min_generations : int
            Generaciones minimas antes de permitir early stopping
        pipeline : DataPipeline, opcional
            Pipeline ya construido (permite reutilizar datos entre semillas)
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.pop_size = pop_size
        self.generations = generations
        self.early_stopping_patience = early_stopping_patience
        self.runs_dir = Path(runs_dir)
        self.seed = seed
        self.workers = max(1, int(workers))
        self.min_generations = min_generations

        # Configuraciones
        self.neat_config = neat_config or {}
        self.trading_config = trading_config or {}

        # Pipeline de datos
        self.pipeline = pipeline or DataPipeline(
            symbol=symbol,
            start=start_date,
            end=end_date,
            interval=interval
        )

        # Directorio unico de la corrida
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        safe_symbol = symbol.replace("-", "_").replace("=", "_")
        seed_tag = f"_seed{seed}" if seed is not None else ""
        self.run_folder = (
            self.runs_dir / f"run_{safe_symbol}_{interval}_{timestamp}{seed_tag}"
        )
        self.run_folder.mkdir(parents=True, exist_ok=True)

    def _seed_everything(self):
        """Fija las semillas para que la corrida sea reproducible."""
        if self.seed is None:
            return
        random.seed(self.seed)
        np.random.seed(self.seed)
        logger.info(f"Semilla fijada: {self.seed}")

    def _build_backtester(self) -> Backtester:
        """Construye el backtester con los costos y reglas del YAML."""
        costs = self.trading_config.get("costs", {})
        decision = self.trading_config.get("decision", {})
        return Backtester(
            initial_capital=self.trading_config.get(
                "capital", {}
            ).get("initial", 100000.0),
            commission=costs.get("commission", 0.0065),
            slippage=costs.get("slippage", 0.001),
            confidence_threshold=decision.get("confidence_threshold", 0.30),
            timeframe=self.interval,
            min_trades=decision.get("min_trades", 10),
            fitness_config=self.trading_config.get("fitness", {}),
            allow_short=decision.get("allow_short", False),
            direction_threshold=decision.get("direction_threshold", 0.2)
        )

    def run(
        self,
        train_years: int = 3,
        val_years: int = 1,
        test_years: int = 1,
        step_years: int = 1,
        mode: str = "sliding"
    ) -> Dict:
        """
        Ejecuta la corrida Walk-Forward completa.

        Parameters
        ----------
        train_years : int
            Anos para train
        val_years : int
            Anos para val
        test_years : int
            Anos para test
        step_years : int
            Anos de avance por ventana
        mode : str
            "sliding" o "expanding"

        Returns
        -------
        dict
            Resumen completo del experimento
        """
        logger.info(f"=== INICIANDO WALK-FORWARD NEAT ===")
        logger.info(f"Activo: {self.symbol} | Intervalo: {self.interval}")
        logger.info(f"Directorio de salida: {self.run_folder}")
        self._seed_everything()

        # 1. Obtener datos procesados
        df_processed = self.pipeline.get_processed_data()
        indicator_cols = self.pipeline.active_indicators

        logger.info(f"Datos totales: {len(df_processed)} barras, {len(indicator_cols)} indicadores")

        # 2. Generar ventanas Walk-Forward
        try:
            windows = list(self.pipeline.walk_forward(
                train_years=train_years,
                val_years=val_years,
                test_years=test_years,
                step_years=step_years,
                mode=mode
            ))
        except ValueError as exc:
            logger.error(str(exc))
            windows = []

        if not windows:
            logger.error("No se generaron ventanas Walk-Forward. Verifica las fechas de los datos.")
            return {}

        logger.info(f"Total de ventanas Walk-Forward: {len(windows)}")
        if len(windows) < 3:
            logger.warning(
                f"Solo hay {len(windows)} ventana(s) OOS: las conclusiones "
                f"estadisticas seran debiles."
            )

        # Configuracion comun
        backtester = self._build_backtester()
        initial_cap = backtester.initial_capital
        wf_cfg = self.trading_config.get("walk_forward", {})
        fitness_cfg = self.trading_config.get("fitness", {})
        bench_cfg = self.trading_config.get("benchmarks", {})
        validation_top_k = max(1, int(wf_cfg.get("validation_top_k", 5)))
        selection_metric = wf_cfg.get("selection_metric", "min")
        val_smoothing = max(1, int(wf_cfg.get("val_smoothing", 3)))
        sma_fast = int(bench_cfg.get("sma_fast", 50))
        sma_slow = int(bench_cfg.get("sma_slow", 200))

        window_results = []
        oos_equities: List[pd.Series] = []
        oos_trades: List[TradeRecord] = []
        bench_equities: Dict[str, List[pd.Series]] = {
            "buy_and_hold": [], "sma_crossover": []
        }
        bench_trades: Dict[str, List[TradeRecord]] = {
            "buy_and_hold": [], "sma_crossover": []
        }

        # 3. Procesar cada ventana Walk-Forward
        for win_idx, window in enumerate(windows):
            logger.info(f"\n" + "=" * 70)
            logger.info(f"VENTANA WALK-FORWARD #{win_idx}")
            logger.info(f"  Train: {window.train.index[0]} -> {window.train.index[-1]} ({len(window.train)} barras)")
            logger.info(f"  Val:   {window.val.index[0]} -> {window.val.index[-1]} ({len(window.val)} barras)")
            logger.info(f"  Test:  {window.test.index[0]} -> {window.test.index[-1]} ({len(window.test)} barras)")
            logger.info("=" * 70)

            win_folder = self.run_folder / f"window_{win_idx}"
            win_folder.mkdir(exist_ok=True)

            pop = self._create_population(indicator_cols)

            evaluator = GenomeEvaluator(
                backtester=backtester,
                train_data=window.train,
                val_data=window.val,
                indicator_cols=indicator_cols,
                num_windows=fitness_cfg.get("training_subwindows", 3),
                worst_weight=fitness_cfg.get("worst_window_weight", 0.30),
                workers=self.workers,
            )

            with evaluator:
                selected = self._evolve_window(
                    pop, evaluator, validation_top_k,
                    selection_metric, val_smoothing
                )

            best_genome: Genome = selected["genome"]
            history = selected["history"]

            # 4. Evaluacion final en TEST (Out-of-Sample estricto)
            logger.info(f"\n  >>> EVALUANDO MEJOR GENOMA EN TEST (Out-of-Sample) <<<")
            test_fit, test_metrics, test_equity, test_trades = backtester.evaluate_genome(
                best_genome,
                window.test,
                indicator_cols
            )

            logger.info(f"  TEST Results:")
            logger.info(f"    Fitness: {test_fit:.4f}")
            logger.info(f"    Retorno: {test_metrics.total_return*100:.2f}%")
            logger.info(f"    Sharpe:  {test_metrics.sharpe_ratio:.2f}")
            logger.info(f"    Max DD:  {test_metrics.max_drawdown*100:.2f}%")
            logger.info(f"    Trades:  {test_metrics.num_trades}")

            oos_equities.append(test_equity)
            oos_trades.extend(test_trades)

            # 5. Benchmarks en el mismo periodo de TEST
            close_history = df_processed["Close"].loc[:window.test.index[-1]]
            bh_m, bh_eq, bh_tr = buy_and_hold(
                window.test, initial_cap, backtester.commission,
                backtester.slippage, self.interval
            )
            sma_m, sma_eq, sma_tr = sma_crossover(
                window.test, close_history, sma_fast, sma_slow,
                initial_cap, backtester.commission,
                backtester.slippage, self.interval
            )
            bench_equities["buy_and_hold"].append(bh_eq)
            bench_equities["sma_crossover"].append(sma_eq)
            bench_trades["buy_and_hold"].extend(bh_tr)
            bench_trades["sma_crossover"].extend(sma_tr)

            logger.info(
                f"    Benchmarks -> Buy&Hold: {bh_m.total_return*100:.2f}% | "
                f"SMA{sma_fast}/{sma_slow}: {sma_m.total_return*100:.2f}%"
            )

            # Guardar datos de esta ventana
            win_data = {
                "window_index": win_idx,
                "train_dates": [str(window.train.index[0]), str(window.train.index[-1])],
                "val_dates": [str(window.val.index[0]), str(window.val.index[-1])],
                "test_dates": [str(window.test.index[0]), str(window.test.index[-1])],
                "selection": selected["info"],
                "num_val_evaluations": selected["num_val_evaluations"],
                "generations_run": len(history),
                "test_fitness": float(test_fit),
                "test_metrics": test_metrics.to_dict(),
                "benchmarks": {
                    "buy_and_hold": bh_m.to_dict(),
                    "sma_crossover": sma_m.to_dict(),
                },
                "best_genome_summary": best_genome.summary(),
                "history": history
            }
            window_results.append(win_data)

            # Guardar resumen de ventana en JSON
            with open(win_folder / "summary.json", "w", encoding="utf-8") as f:
                json.dump(win_data, f, indent=2, default=_json_default)

            # Guardar mejor genoma de la ventana en JSON
            with open(win_folder / "best_genome.json", "w", encoding="utf-8") as f:
                json.dump(best_genome.to_dict(), f, indent=2, default=_json_default)

        # 6. Calcular Equidad OOS global encadenada
        full_oos_equity = chain_equities(oos_equities, initial_cap)
        global_oos_metrics = compute_metrics(
            full_oos_equity,
            oos_trades,
            initial_capital=initial_cap,
            timeframe=self.interval
        )

        chained_benchmarks = {
            name: chain_equities(eqs, initial_cap)
            for name, eqs in bench_equities.items()
        }
        global_bench_metrics = {
            name: compute_metrics(
                eq, bench_trades[name],
                initial_capital=initial_cap, timeframe=self.interval
            ).to_dict()
            for name, eq in chained_benchmarks.items()
        }

        logger.info(f"\n" + "=" * 70)
        logger.info("=== RENDIMIENTO GLOBAL OUT-OF-SAMPLE (TODAS LAS VENTANAS) ===")
        logger.info(f"  Retorno Total OOS: {global_oos_metrics.total_return*100:.2f}%")
        logger.info(f"  Sharpe Ratio OOS:  {global_oos_metrics.sharpe_ratio:.2f}")
        logger.info(f"  Sortino Ratio OOS: {global_oos_metrics.sortino_ratio:.2f}")
        logger.info(f"  Max Drawdown OOS:  {global_oos_metrics.max_drawdown*100:.2f}%")
        for name, met in global_bench_metrics.items():
            logger.info(
                f"  [{name}] Retorno: {met['total_return_pct']:.2f}% | "
                f"Sharpe: {met['sharpe_ratio']:.2f} | "
                f"MaxDD: {met['max_drawdown_pct']:.2f}%"
            )
        logger.info("=" * 70)

        # Resumen general del experimento
        experiment_summary = {
            "symbol": self.symbol,
            "interval": self.interval,
            "seed": self.seed,
            "num_windows": len(windows),
            "settings": {
                "train_years": train_years,
                "val_years": val_years,
                "test_years": test_years,
                "step_years": step_years,
                "mode": mode,
                "pop_size": self.pop_size,
                "generations": self.generations,
                "early_stopping_patience": self.early_stopping_patience,
                "min_generations": self.min_generations,
                "validation_top_k": validation_top_k,
                "selection_metric": selection_metric,
                "val_smoothing": val_smoothing,
                "allow_short": backtester.allow_short,
            },
            "global_oos_metrics": global_oos_metrics.to_dict(),
            "global_benchmark_metrics": global_bench_metrics,
            "windows": window_results
        }

        with open(self.run_folder / "experiment_summary.json", "w", encoding="utf-8") as f:
            json.dump(experiment_summary, f, indent=2, default=_json_default)

        # Guardar equidades OOS en CSV
        full_oos_equity.to_csv(self.run_folder / "oos_equity_curve.csv")
        pd.DataFrame(chained_benchmarks).to_csv(
            self.run_folder / "oos_benchmarks.csv"
        )

        return experiment_summary

    def _create_population(self, indicator_cols: list[str]) -> Population:
        """Crea e inicializa una poblacion NEAT para una ventana."""
        mutation_config = dict(self.neat_config.get("mutation", {}))
        mutation_config["activation_functions"] = self.neat_config.get(
            "genome", {}
        ).get(
            "activation_functions",
            ["sigmoid", "tanh", "relu", "leaky_relu", "identity"]
        )

        pop = Population(
            pop_size=self.pop_size,
            num_inputs=len(indicator_cols),
            num_outputs=3,
            elitism=self.neat_config.get("population", {}).get("elitism", 2),
            mutation_config=mutation_config,
            speciation_config=self.neat_config.get("speciation", None),
            crossover_config=self.neat_config.get("crossover", None),
            risk_config=self.trading_config.get("risk", None)
        )
        pop.initialize(indicator_names=indicator_cols, initial_active=4)
        return pop

    def _evolve_window(
        self,
        pop: Population,
        evaluator: GenomeEvaluator,
        validation_top_k: int,
        selection_metric: str,
        val_smoothing: int
    ) -> Dict:
        """
        Evoluciona una poblacion en una ventana y selecciona el genoma final.

        Returns
        -------
        dict
            genome, info (por que se eligio), history y num_val_evaluations
        """
        best_score = float("-inf")
        best_genome: Optional[Genome] = None
        best_info: Dict = {}

        best_smoothed_val = float("-inf")
        patience_counter = 0
        val_medians: List[float] = []
        num_val_evaluations = 0
        history = []

        for gen in range(1, self.generations + 1):
            # a. Evaluar cada genoma en Train
            train_fits = evaluator.train_fitness(pop.genomes)
            for genome, fit in zip(pop.genomes, train_fits):
                genome.fitness = fit

            # b. Validar los K mejores de train
            ranked = sorted(
                pop.genomes,
                key=lambda g: (
                    g.fitness if g.fitness is not None else float("-inf")
                ),
                reverse=True
            )
            candidates = ranked[:validation_top_k]
            val_results = evaluator.val_fitness(candidates)
            num_val_evaluations += len(candidates)

            val_fits = [fit for fit, _ in val_results]
            gen_val_median = float(np.median(val_fits))
            best_val_idx = int(np.argmax(val_fits))
            gen_val_best = float(val_fits[best_val_idx])
            best_val_metrics = val_results[best_val_idx][1]

            # c. Seleccion conservadora: min(train, val) por defecto
            for candidate, (val_fit, val_metrics) in zip(candidates, val_results):
                score = selection_score(
                    candidate.fitness, val_fit, selection_metric
                )
                better = score > best_score + 1e-12
                tie_simpler = (
                    best_genome is not None
                    and abs(score - best_score) <= 1e-12
                    and candidate.complexity_score < best_genome.complexity_score
                )
                if better or tie_simpler:
                    best_score = score
                    best_genome = candidate.copy()
                    best_info = {
                        "generation": gen,
                        "selection_metric": selection_metric,
                        "selection_score": float(score),
                        "train_fitness": float(candidate.fitness),
                        "val_fitness": float(val_fit),
                        "val_metrics": val_metrics.to_dict(),
                    }

            # d. Early stopping sobre la mediana suavizada de validacion
            val_medians.append(gen_val_median)
            smoothed_val = float(np.mean(val_medians[-val_smoothing:]))
            if smoothed_val > best_smoothed_val + 1e-9:
                best_smoothed_val = smoothed_val
                patience_counter = 0
            else:
                patience_counter += 1

            train_best_fit = ranked[0].fitness if ranked else -999.0

            # e. Evolucionar y guardar estadisticas de la generacion
            gen_stats = pop.evolve()
            gen_stats["val_fitness"] = gen_val_median
            gen_stats["val_best_fitness"] = gen_val_best
            gen_stats["val_smoothed"] = smoothed_val
            gen_stats["generalization_gap"] = float(train_best_fit - gen_val_median)
            gen_stats["val_sharpe"] = float(best_val_metrics.sharpe_ratio)
            gen_stats["val_return_pct"] = float(best_val_metrics.total_return * 100)
            history.append(gen_stats)

            if gen % 10 == 0 or gen == 1 or patience_counter == 0:
                logger.info(
                    f"  Gen {gen:3d}/{self.generations} | "
                    f"Train Best: {train_best_fit:.3f} | "
                    f"Val Mediana: {gen_val_median:.3f} "
                    f"(suav: {smoothed_val:.3f}) | "
                    f"Sel: {best_score:.3f} | "
                    f"Species: {gen_stats['num_species']}"
                )

            # Early Stopping
            if (
                patience_counter >= self.early_stopping_patience
                and gen >= self.min_generations
            ):
                logger.info(
                    f"  Early stopping activado en generacion {gen} "
                    f"(sin mejora en Val suavizada por {patience_counter} gens)"
                )
                break

        if best_genome is None:
            best_genome = (pop.best_genome_ever or pop.genomes[0]).copy()

        logger.info(
            f"  Genoma seleccionado (gen {best_info.get('generation', '?')}): "
            f"train={best_info.get('train_fitness', float('nan')):.3f}, "
            f"val={best_info.get('val_fitness', float('nan')):.3f} "
            f"tras {num_val_evaluations} evaluaciones en validacion. "
            f"El fitness de validacion del seleccionado es OPTIMISTA; "
            f"solo TEST es fuera de muestra."
        )

        return {
            "genome": best_genome,
            "info": best_info,
            "history": history,
            "num_val_evaluations": num_val_evaluations,
        }

    def _chain_equities(self, equities: List[pd.Series], initial_capital: float) -> pd.Series:
        """
        Encadena curvas de equidad relativas de multiples ventanas OOS.
        """
        return chain_equities(equities, initial_capital)


def _json_default(obj):
    """Convierte tipos de numpy/pandas a tipos serializables en JSON."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    raise TypeError(f"Tipo no serializable: {type(obj)}")
