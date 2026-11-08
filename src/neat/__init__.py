"""
Modulo NEAT: Neuroevolucion de Topologias Aumentadas.

Implementacion desde cero del algoritmo NEAT (Stanley & Miikkulainen, 2002)
extendido con parametros de trading y mascara de indicadores.

Componentes:
  - Genome: representacion genetica (red + trading + indicadores)
  - FeedForwardNetwork: fenotipo funcional (evalua datos)
  - InnovationTracker: marcadores historicos
  - Population: gestion de la poblacion y ciclo evolutivo
  - SpeciationEngine: agrupacion por similitud

Uso:
----
>>> from src.neat import Population
>>> pop = Population(pop_size=150, num_inputs=18, num_outputs=3)
>>> pop.initialize(indicator_names=["RSI", "MACD", ...])
>>> for gen in range(200):
...     for genome in pop.genomes:
...         network = FeedForwardNetwork.from_genome(genome)
...         genome.fitness = evaluate(network)
...     stats = pop.evolve()
"""

from src.neat.genome import Genome, ConnectionGene, NodeGene, TradingParams
from src.neat.network import FeedForwardNetwork, ACTIVATION_FUNCTIONS
from src.neat.innovation import InnovationTracker
from src.neat.reproduction import mutate, crossover
from src.neat.species import Species, SpeciationEngine
from src.neat.population import Population

__all__ = [
    "Genome",
    "ConnectionGene",
    "NodeGene",
    "TradingParams",
    "FeedForwardNetwork",
    "ACTIVATION_FUNCTIONS",
    "InnovationTracker",
    "mutate",
    "crossover",
    "Species",
    "SpeciationEngine",
    "Population",
]
