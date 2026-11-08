"""
============================================================
evaluation.py - Evaluacion de Genomas (secuencial o paralela)
============================================================

Centraliza COMO se calcula el fitness de un genoma para que el motor
Walk-Forward y los procesos trabajadores usen exactamente la misma logica.

PARALELISMO
-----------
El backtest es un bucle Python barra a barra, por lo que evaluar una
poblacion completa es la parte mas lenta del experimento. Como cada
genoma se evalua de forma INDEPENDIENTE y DETERMINISTA, se puede repartir
entre varios procesos sin cambiar los resultados.

En Windows los procesos se crean con "spawn": cada trabajador importa
este modulo desde cero. Por eso los datos (train/val) se envian UNA vez
por ventana mediante el `initializer` del pool y se guardan en una
variable global del proceso trabajador.
"""

from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import numpy as np
import pandas as pd

from src.neat.genome import Genome
from src.trading.backtester import Backtester
from src.trading.metrics import PerformanceMetrics


def evaluate_training_fitness(
    backtester: Backtester,
    genome: Genome,
    train_data: pd.DataFrame,
    indicator_cols: list[str],
    num_windows: int = 3,
    worst_weight: float = 0.30,
) -> float:
    """
    Combina el rendimiento mediano y el peor regimen del entrenamiento.

    train_fitness = (1 - w) * mediana(subperiodos) + w * peor_subperiodo

    Exigir un buen peor caso reduce la probabilidad de seleccionar una
    estrategia que solo funciona en un regimen de mercado.
    """
    num_windows = max(1, int(num_windows))

    if num_windows == 1 or len(train_data) < num_windows * 30:
        fitness, _, _, _ = backtester.evaluate_genome(
            genome, train_data, indicator_cols
        )
        return fitness

    scores = []
    boundaries = np.linspace(0, len(train_data), num_windows + 1, dtype=int)
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        chunk = train_data.iloc[start:end]
        if len(chunk) < 30:
            continue
        score, _, _, _ = backtester.evaluate_genome(
            genome, chunk, indicator_cols
        )
        scores.append(score)

    if not scores:
        return -5.0

    median_score = float(np.median(scores))
    worst_score = float(min(scores))
    return (1.0 - worst_weight) * median_score + worst_weight * worst_score


# =================================================================
# ESTADO DE LOS PROCESOS TRABAJADORES
# =================================================================

_WORKER_STATE: dict = {}


def _init_worker(
    backtester: Backtester,
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    indicator_cols: list[str],
    num_windows: int,
    worst_weight: float,
):
    """Guarda los datos de la ventana actual en el proceso trabajador."""
    _WORKER_STATE.clear()
    _WORKER_STATE.update(
        backtester=backtester,
        train=train_data,
        val=val_data,
        cols=indicator_cols,
        num_windows=num_windows,
        worst_weight=worst_weight,
    )


def _worker_train(genome: Genome) -> float:
    s = _WORKER_STATE
    return evaluate_training_fitness(
        s["backtester"], genome, s["train"], s["cols"],
        s["num_windows"], s["worst_weight"],
    )


def _worker_val(genome: Genome) -> tuple[float, PerformanceMetrics]:
    s = _WORKER_STATE
    fitness, metrics, _, _ = s["backtester"].evaluate_genome(
        genome, s["val"], s["cols"]
    )
    return fitness, metrics


# =================================================================
# EVALUADOR
# =================================================================

class GenomeEvaluator:
    """
    Evalua genomas en train y validacion, secuencialmente o en paralelo.

    Uso:
    ----
    >>> with GenomeEvaluator(bt, train, val, cols, workers=4) as ev:
    ...     train_fits = ev.train_fitness(pop.genomes)
    ...     val_results = ev.val_fitness(candidates)
    """

    def __init__(
        self,
        backtester: Backtester,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        indicator_cols: list[str],
        num_windows: int = 3,
        worst_weight: float = 0.30,
        workers: int = 1,
    ):
        self.backtester = backtester
        self.train_data = train_data
        self.val_data = val_data
        self.indicator_cols = indicator_cols
        self.num_windows = num_windows
        self.worst_weight = worst_weight
        self.workers = max(1, int(workers))
        self._pool: Optional[ProcessPoolExecutor] = None

    def __enter__(self) -> "GenomeEvaluator":
        if self.workers > 1:
            self._pool = ProcessPoolExecutor(
                max_workers=self.workers,
                initializer=_init_worker,
                initargs=(
                    self.backtester,
                    self.train_data,
                    self.val_data,
                    self.indicator_cols,
                    self.num_windows,
                    self.worst_weight,
                ),
            )
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def _chunksize(self, n: int) -> int:
        return max(1, n // (self.workers * 4))

    def train_fitness(self, genomes: list[Genome]) -> list[float]:
        """Fitness de entrenamiento de cada genoma (mismo orden)."""
        if self._pool is None:
            return [
                evaluate_training_fitness(
                    self.backtester, g, self.train_data, self.indicator_cols,
                    self.num_windows, self.worst_weight,
                )
                for g in genomes
            ]
        return list(self._pool.map(
            _worker_train, genomes, chunksize=self._chunksize(len(genomes))
        ))

    def val_fitness(
        self, genomes: list[Genome]
    ) -> list[tuple[float, PerformanceMetrics]]:
        """(fitness, metricas) de validacion de cada genoma (mismo orden)."""
        if self._pool is None:
            results = []
            for g in genomes:
                fitness, metrics, _, _ = self.backtester.evaluate_genome(
                    g, self.val_data, self.indicator_cols
                )
                results.append((fitness, metrics))
            return results
        return list(self._pool.map(_worker_val, genomes))


# =================================================================
# SELECCION CONSERVADORA (anti-sobreajuste a validacion)
# =================================================================

SELECTION_METRICS = ("min", "mean", "val")


def selection_score(
    train_fitness: float,
    val_fitness: float,
    metric: str = "min",
) -> float:
    """
    Puntaje con el que se elige el genoma final de una ventana.

    PROBLEMA QUE RESUELVE
    ---------------------
    Si en cada generacion se validan K candidatos y se guarda el MAXIMO
    historico de validacion, tras G generaciones se han hecho ~K*G
    comparaciones. El ganador suele ser el que tuvo SUERTE en ese año de
    validacion, no el mas robusto (sesgo de seleccion).

    "min"  -> min(train, val): un candidato solo puntua alto si es bueno
              en AMBOS periodos; un val "afortunado" no compensa un train
              mediocre.
    "mean" -> promedio de train y val.
    "val"  -> solo validacion (comportamiento anterior, mas optimista).
    """
    if metric == "min":
        return min(train_fitness, val_fitness)
    if metric == "mean":
        return 0.5 * (train_fitness + val_fitness)
    if metric == "val":
        return val_fitness
    raise ValueError(
        f"selection_metric '{metric}' no soportado. Opciones: {SELECTION_METRICS}"
    )
