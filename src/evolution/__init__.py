"""
Modulo de evolucion Walk-Forward.

Componentes:
  - WalkForwardEngine: motor de ejecucion del experimento completo
  - GenomeEvaluator: evaluacion secuencial o paralela de genomas
  - selection_score: puntaje conservador para elegir el genoma final
  - chain_equities: encadena las curvas OOS de varias ventanas
"""

from src.evolution.engine import WalkForwardEngine, chain_equities
from src.evolution.evaluation import GenomeEvaluator, selection_score

__all__ = [
    "WalkForwardEngine",
    "GenomeEvaluator",
    "selection_score",
    "chain_equities",
]
