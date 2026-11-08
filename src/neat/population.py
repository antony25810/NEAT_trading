"""
============================================================
population.py - Gestion de la Poblacion NEAT
============================================================

Este modulo orquesta todo el ciclo evolutivo de una generacion:
1. Inicializar poblacion de genomas minimos
2. Especiar (agrupar por similitud)
3. Evaluar fitness (externo — lo hace el simulador de trading)
4. Seleccionar (dentro de cada especie)
5. Reproducir (cruce + mutacion)
6. Repetir

Es el modulo de NIVEL MAS ALTO del motor NEAT.
"""

import random
import logging
from typing import Callable, Optional

from src.neat.genome import Genome
from src.neat.innovation import InnovationTracker
from src.neat.reproduction import mutate, crossover
from src.neat.species import SpeciationEngine

logger = logging.getLogger(__name__)


class Population:
    """
    Poblacion de genomas NEAT con especiacion y reproduccion.

    Uso:
    ----
    >>> pop = Population(pop_size=150, num_inputs=18, num_outputs=3)
    >>> pop.initialize(indicator_names=["RSI", "MACD", ...])
    >>>
    >>> for generation in range(200):
    ...     # Evaluar fitness de cada genoma (funcion externa)
    ...     for genome in pop.genomes:
    ...         genome.fitness = evaluate(genome)
    ...
    ...     # Evolucionar: especiar + seleccionar + reproducir
    ...     stats = pop.evolve()
    ...     print(f"Gen {generation}: best={stats['best_fitness']:.2f}")
    """

    def __init__(
        self,
        pop_size: int = 150,
        num_inputs: int = 18,
        num_outputs: int = 3,
        elitism: int = 2,
        mutation_config: Optional[dict] = None,
        speciation_config: Optional[dict] = None,
        crossover_config: Optional[dict] = None,
        risk_config: Optional[dict] = None,
    ):
        """
        Configura la poblacion.

        Parameters
        ----------
        pop_size : int
            Numero de individuos
        num_inputs : int
            Numero de inputs posibles (indicadores)
        num_outputs : int
            Numero de outputs (3: direccion, confianza, tamano)
        elitism : int
            Cuantos mejores genomas de cada especie pasan directamente
            a la siguiente generacion sin modificacion
        mutation_config : dict, opcional
            Probabilidades de mutacion (de neat_config.yaml)
        speciation_config : dict, opcional
            Configuracion de especiacion (threshold, coeficientes, etc.)
        """
        self.pop_size = pop_size
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.elitism = elitism
        self.risk_config = risk_config or {}

        # Config de mutacion (defaults si no se proporcionan)
        self.mutation_config = mutation_config or {
            "weight_perturbation": 0.80,
            "weight_reset": 0.10,
            "weight_perturbation_std": 0.5,
            "bias_perturbation": 0.30,
            "bias_reset": 0.05,
            "bias_perturbation_std": 0.2,
            "add_node": 0.03,
            "remove_node": 0.02,
            "add_connection": 0.05,
            "remove_connection": 0.04,
            "change_activation": 0.03,
            "mutate_stop_loss": 0.10,
            "mutate_take_profit": 0.10,
            "mutate_position_size": 0.10,
            "mutate_trailing_stop": 0.08,
            "mutate_max_drawdown": 0.08,
            "mutate_indicator_mask": 0.05,
            "activation_functions": [
                "sigmoid", "tanh", "relu", "leaky_relu", "identity"
            ],
        }
        self.mutation_config["risk_ranges"] = self.risk_config

        # Config de especiacion
        spec_cfg = speciation_config or {}
        self.speciation = SpeciationEngine(
            compatibility_threshold=spec_cfg.get("compatibility_threshold", 3.0),
            c1=spec_cfg.get("c1", 1.0),
            c2=spec_cfg.get("c2", 1.0),
            c3=spec_cfg.get("c3", 0.4),
            c4=spec_cfg.get("c4", 0.5),
            c5=spec_cfg.get("c5", 0.3),
            min_species_size=spec_cfg.get("min_species_size", 5),
            max_stagnation=spec_cfg.get("max_stagnation", 15),
            target_species_min=spec_cfg.get("target_species_min", 5),
            target_species_max=spec_cfg.get("target_species_max", 12),
            threshold_step=spec_cfg.get("threshold_step", 0.10),
            threshold_min=spec_cfg.get("threshold_min", 0.50),
            threshold_max=spec_cfg.get("threshold_max", 6.00),
            species_grace_period=spec_cfg.get("species_grace_period", 3),
        )

        cross_cfg = crossover_config or {}
        self.crossover_rate = cross_cfg.get("rate", 0.75)
        self.interspecies_rate = cross_cfg.get("interspecies_rate", 0.001)

        # Innovation tracker (compartido por toda la poblacion)
        self.tracker = InnovationTracker()
        self.tracker.set_node_counter(num_inputs + num_outputs)

        # Genomas actuales
        self.genomes: list[Genome] = []

        # Estadisticas
        self.generation: int = 0
        self._genome_id_counter: int = 0
        self._best_genome_ever: Optional[Genome] = None

    def _next_genome_id(self) -> int:
        """Genera un ID unico para un genoma."""
        self._genome_id_counter += 1
        return self._genome_id_counter

    def initialize(
        self,
        indicator_names: list[str],
        initial_active: int = 4
    ):
        """
        Crea la poblacion inicial con genomas minimos.

        Cada genoma empieza con:
        - N indicadores activos al azar (default: 4)
        - Conexiones directas input -> output
        - Sin nodos ocultos
        - Parametros de trading aleatorios

        Parameters
        ----------
        indicator_names : list[str]
            Nombres de todos los indicadores disponibles
        initial_active : int
            Cuantos indicadores activar al inicio por genoma
        """
        assert len(indicator_names) == self.num_inputs, (
            f"Se esperaban {self.num_inputs} indicadores, "
            f"se recibieron {len(indicator_names)}"
        )

        self.genomes = []

        # Reservar deterministicamente todas las conexiones input->output.
        # Asi el mismo innovation number siempre representa los mismos endpoints.
        for input_id in range(self.num_inputs):
            for output_offset in range(self.num_outputs):
                self.tracker.get_connection_innovation(
                    input_id,
                    self.num_inputs + output_offset,
                    generation=0
                )

        for _ in range(self.pop_size):
            genome = Genome(
                genome_id=self._next_genome_id(),
                num_inputs=self.num_inputs,
                num_outputs=self.num_outputs
            )
            genome.initialize_minimal(
                indicator_names=indicator_names,
                initial_active=initial_active,
                tracker=self.tracker,
                generation=0,
                risk_config=self.risk_config
            )
            self.genomes.append(genome)

        # Especiacion inicial
        self.speciation.speciate(self.genomes, generation=0)

        logger.info(
            f"Poblacion inicializada: {len(self.genomes)} genomas, "
            f"{len(self.speciation.species)} especies, "
            f"{initial_active} indicadores activos por genoma"
        )

    def evolve(self) -> dict:
        """
        Ejecuta un ciclo evolutivo completo (una generacion).

        PREREQUISITO: Todos los genomas deben tener fitness asignado
        antes de llamar a este metodo.

        FLUJO:
        1. Especiar la poblacion actual
        2. Calcular fitness ajustado
        3. Determinar offspring por especie
        4. Para cada especie:
           a. Preservar elite (mejores sin modificar)
           b. Generar offspring (cruce + mutacion)
        5. Reemplazar poblacion
        6. Actualizar estadisticas

        Returns
        -------
        dict
            Estadisticas de la generacion
        """
        self.generation += 1

        # Verificar que todos tengan fitness
        unrated = [g for g in self.genomes if g.fitness is None]
        if unrated:
            logger.warning(
                f"{len(unrated)} genomas sin fitness en generacion "
                f"{self.generation}. Asignando fitness = -999"
            )
            for g in unrated:
                g.fitness = -999.0

        # --- Paso 1: Especiar ---
        self.speciation.speciate(self.genomes, self.generation)

        # --- Paso 2: Calcular offspring por especie ---
        offspring_counts = self.speciation.get_offspring_counts(self.pop_size)

        # Estadisticas de la poblacion que realmente fue evaluada.
        stats = self._compute_stats()

        # Actualizar mejor genoma de la historia antes de resetear fitness.
        current_best = self.speciation.get_best_genome()
        if current_best:
            if (self._best_genome_ever is None or
                    (current_best.fitness is not None and
                     (self._best_genome_ever.fitness is None or
                      current_best.fitness > self._best_genome_ever.fitness))):
                self._best_genome_ever = current_best.copy()

        # --- Paso 3: Generar nueva poblacion ---
        new_genomes: list[Genome] = []

        for species_id, sp in self.speciation.species.items():
            n_offspring = offspring_counts.get(species_id, 0)
            if n_offspring == 0 or sp.is_empty:
                continue

            # Ordenar miembros por fitness (mejor primero)
            members = sorted(
                sp.members,
                key=lambda g: g.fitness if g.fitness is not None else float("-inf"),
                reverse=True
            )

            # Elite: los mejores pasan sin modificacion
            n_elite = min(self.elitism, len(members), n_offspring)
            for i in range(n_elite):
                elite = members[i].copy()
                elite.genome_id = self._next_genome_id()
                elite.fitness = None  # resetear para la proxima evaluacion
                new_genomes.append(elite)

            # Pool de candidatos para reproduccion (mejores 50%)
            pool_size = max(2, len(members) // 2)
            pool = members[:pool_size]

            # Generar el resto de offspring
            remaining = n_offspring - n_elite
            for _ in range(remaining):
                child_id = self._next_genome_id()

                if random.random() < self.crossover_rate and len(pool) >= 2:
                    parent_a = random.choice(pool)
                    parent_b = self._select_second_parent(
                        species_id, pool, parent_a
                    )
                    child = crossover(
                        parent_a,
                        parent_b,
                        child_id,
                        risk_ranges=self.risk_config
                    )
                else:
                    # CLONACION + MUTACION: copiar un padre y mutar
                    parent = random.choice(pool)
                    child = parent.copy()
                    child.genome_id = child_id

                # Aplicar mutaciones al hijo
                mutate(child, self.tracker, self.generation,
                       self.mutation_config)

                child.fitness = None
                new_genomes.append(child)

        # Ajustar si generamos de mas o de menos
        while len(new_genomes) < self.pop_size:
            # Clonar un genoma aleatorio de la poblacion actual
            source = random.choice(self.genomes)
            clone = source.copy()
            clone.genome_id = self._next_genome_id()
            clone.fitness = None
            mutate(clone, self.tracker, self.generation, self.mutation_config)
            new_genomes.append(clone)

        while len(new_genomes) > self.pop_size:
            new_genomes.pop()

        # --- Paso 4: Reemplazar poblacion ---
        self.genomes = new_genomes

        stats["best_genome_ever_fitness"] = (
            self._best_genome_ever.fitness
            if self._best_genome_ever and self._best_genome_ever.fitness is not None
            else 0.0
        )
        logger.info(
            f"Gen {self.generation}: "
            f"best={stats['best_fitness']:.4f}, "
            f"avg={stats['avg_fitness']:.4f}, "
            f"species={stats['num_species']}, "
            f"best_complexity={stats['best_complexity']}"
        )

        return stats

    def _select_second_parent(
        self,
        species_id: int,
        local_pool: list[Genome],
        parent_a: Genome
    ) -> Genome:
        """Selecciona pareja local o interespecie segun configuracion."""
        if random.random() < self.interspecies_rate:
            external = [
                genome
                for sid, species in self.speciation.species.items()
                if sid != species_id
                for genome in species.members
            ]
            if external:
                return random.choice(external)

        candidates = [g for g in local_pool if g is not parent_a]
        return random.choice(candidates)

    def _compute_stats(self) -> dict:
        """Calcula estadisticas de la generacion actual."""
        evaluated = [
            g for g in self.genomes if g.fitness is not None
        ]
        fitnesses = [g.fitness for g in evaluated]

        # Buscar el mejor de la generacion anterior (antes del reset)
        best = self.speciation.get_best_genome()

        return {
            "generation": self.generation,
            "best_fitness": (
                best.fitness if best and best.fitness is not None else 0.0
            ),
            "avg_fitness": sum(fitnesses) / len(fitnesses) if fitnesses else 0.0,
            "min_fitness": min(fitnesses) if fitnesses else 0.0,
            "max_fitness": max(fitnesses) if fitnesses else 0.0,
            "num_species": len(self.speciation.species),
            "compatibility_threshold": self.speciation.threshold,
            "species_sizes": {
                sid: sp.size for sid, sp in self.speciation.species.items()
            },
            "best_complexity": best.complexity if best else 0,
            "best_genome_ever_fitness": 0.0,
            "innovation_count": self.tracker.current_innovation,
            "node_count": self.tracker.current_node_id,
        }

    @property
    def best_genome(self) -> Optional[Genome]:
        """Mejor genoma de la generacion actual."""
        evaluated = [g for g in self.genomes if g.fitness is not None]
        if not evaluated:
            return None
        return max(evaluated, key=lambda g: g.fitness)

    @property
    def best_genome_ever(self) -> Optional[Genome]:
        """Mejor genoma encontrado en toda la evolucion."""
        return self._best_genome_ever
