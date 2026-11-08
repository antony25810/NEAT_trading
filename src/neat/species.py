"""
============================================================
species.py - Especiacion NEAT
============================================================

La especiacion agrupa individuos similares en "especies" para
proteger innovaciones. Sin especiacion, un individuo con una
topologia nueva pero fitness bajo moriria inmediatamente.
Con especiacion, compite solo contra individuos similares.
"""

import random
import math
from typing import Optional

from src.neat.genome import Genome


class Species:
    """
    Representa una especie: grupo de genomas similares.

    Cada especie tiene:
    - Un representante (genoma referencia para comparar distancia)
    - Una lista de miembros
    - Un historial de fitness para detectar estancamiento
    """

    def __init__(self, species_id: int, representative: Genome):
        self.species_id = species_id
        self.representative = representative
        self.members: list[Genome] = []

        # Historial de fitness para detectar estancamiento
        self.best_fitness_ever: float = float("-inf")
        self.generations_without_improvement: int = 0

        # Fitness promedio de la especie (para asignar offspring)
        self.avg_fitness: float = 0.0
        self.age: int = 0

    def add_member(self, genome: Genome):
        """Agrega un genoma a esta especie."""
        genome.species_id = self.species_id
        self.members.append(genome)

    def clear_members(self):
        """Limpia la lista de miembros (para re-especiar)."""
        self.members.clear()

    def update_representative(self):
        """
        Actualiza el representante de la especie.

        El representante se elige AL AZAR entre los miembros actuales.
        Esto es lo que dice el paper original: no es el mejor,
        es uno aleatorio. Esto evita que el representante sea siempre
        el mismo y que la especie se vuelva demasiado "estrecha".
        """
        if self.members:
            self.representative = random.choice(self.members)

    def compute_adjusted_fitness(self):
        """
        Calcula el fitness ajustado de cada miembro (fitness sharing).

        EXPLICIT FITNESS SHARING:
        fitness_ajustado = fitness_raw / num_miembros

        Esto garantiza que especies grandes no dominen la poblacion.
        Una especie con 50 miembros y fitness promedio 10 produce
        la misma cantidad de offspring que una de 5 miembros con
        fitness promedio 1 (ambas contribuyen 10/50 = 1/5 = 0.2 por
        miembro).
        """
        n = len(self.members)
        if n == 0:
            return

        total_fitness = 0.0
        for genome in self.members:
            raw = genome.fitness if genome.fitness is not None else 0.0
            genome.adjusted_fitness = raw / n
            total_fitness += raw

        self.avg_fitness = total_fitness / n

    def check_stagnation(self) -> bool:
        """
        Verifica si la especie esta estancada.

        Una especie esta estancada si su MEJOR fitness no ha mejorado
        en max_stagnation generaciones. En ese caso, se extingue
        para liberar espacio a nuevas innovaciones.

        Returns
        -------
        bool
            True si la especie mejoro, False si esta estancada
        """
        current_best = max(
            (g.fitness for g in self.members if g.fitness is not None),
            default=float("-inf")
        )

        if current_best == float("-inf"):
            return True

        if current_best > self.best_fitness_ever:
            self.best_fitness_ever = current_best
            self.generations_without_improvement = 0
            return True
        else:
            self.generations_without_improvement += 1
            return False

    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def is_empty(self) -> bool:
        return len(self.members) == 0

    def get_best(self) -> Optional[Genome]:
        """Retorna el miembro con mejor fitness."""
        if not self.members:
            return None
        return max(self.members,
                   key=lambda g: g.fitness if g.fitness is not None else float("-inf"))

    def __repr__(self) -> str:
        best = self.get_best()
        best_fit = best.fitness if best and best.fitness is not None else "?"
        return (
            f"Species(id={self.species_id}, "
            f"size={self.size}, "
            f"avg_fit={self.avg_fitness:.2f}, "
            f"best_fit={best_fit}, "
            f"stag={self.generations_without_improvement})"
        )


class SpeciationEngine:
    """
    Motor de especiacion: agrupa genomas en especies.

    En cada generacion:
    1. Limpiar miembros de todas las especies
    2. Para cada genoma, comparar con el representante de cada especie
    3. Si la distancia es menor que el threshold, unirlo a esa especie
    4. Si no cabe en ninguna, crear una nueva especie
    5. Eliminar especies vacias o estancadas
    6. Calcular fitness ajustado (sharing)

    Uso:
    ----
    >>> engine = SpeciationEngine(threshold=3.0)
    >>> engine.speciate(genomes, generation=0)
    >>> print(engine.species)
    """

    def __init__(
        self,
        compatibility_threshold: float = 3.0,
        c1: float = 1.0,
        c2: float = 1.0,
        c3: float = 0.4,
        c4: float = 0.5,
        c5: float = 0.3,
        min_species_size: int = 5,
        max_stagnation: int = 15,
        target_species_min: int = 5,
        target_species_max: int = 12,
        threshold_step: float = 0.10,
        threshold_min: float = 0.50,
        threshold_max: float = 6.00,
        species_grace_period: int = 3
    ):
        """
        Parameters
        ----------
        compatibility_threshold : float
            Distancia maxima para pertenecer a la misma especie
        c1, c2, c3, c4, c5 : float
            Coeficientes para la funcion de distancia
        min_species_size : int
            Tamano minimo para que una especie sobreviva
        max_stagnation : int
            Generaciones sin mejora antes de extinguir
        """
        self.threshold = compatibility_threshold
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        self.c5 = c5
        self.min_species_size = min_species_size
        self.max_stagnation = max_stagnation
        self.target_species_min = target_species_min
        self.target_species_max = target_species_max
        self.threshold_step = threshold_step
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        self.species_grace_period = species_grace_period

        self.species: dict[int, Species] = {}
        self._next_species_id: int = 0

    def speciate(self, genomes: list[Genome], generation: int):
        """
        Asigna cada genoma a una especie.

        ALGORITMO:
        1. Limpiar miembros de todas las especies (mantener representantes)
        2. Para cada genoma:
           a. Comparar con el representante de cada especie existente
           b. Si distancia < threshold, unirlo a esa especie
           c. Si no cabe en ninguna, crear especie nueva
        3. Eliminar especies vacias
        4. Actualizar representantes
        5. Calcular fitness ajustado

        Parameters
        ----------
        genomes : list[Genome]
            Todos los genomas de la poblacion
        generation : int
            Generacion actual (para logging)
        """
        # Paso 1: Limpiar miembros
        for sp in self.species.values():
            sp.clear_members()
            sp.age += 1

        # Paso 2: Asignar genomas a especies
        for genome in genomes:
            placed = False

            for sp in self.species.values():
                dist = genome.distance(
                    sp.representative,
                    c1=self.c1, c2=self.c2, c3=self.c3,
                    c4=self.c4, c5=self.c5
                )
                if dist < self.threshold:
                    sp.add_member(genome)
                    placed = True
                    break

            if not placed:
                # Crear nueva especie
                new_sp = Species(self._next_species_id, genome)
                new_sp.add_member(genome)
                self.species[self._next_species_id] = new_sp
                self._next_species_id += 1

        # Paso 3: Eliminar especies vacias
        empty_ids = [
            sid for sid, sp in self.species.items()
            if sp.is_empty
        ]
        for sid in empty_ids:
            del self.species[sid]

        # Paso 4: Verificar estancamiento y especies persistentemente pequenas.
        stagnant_ids = []
        undersized_ids = []
        for sid, sp in self.species.items():
            sp.check_stagnation()
            if sp.generations_without_improvement >= self.max_stagnation:
                stagnant_ids.append(sid)
            if (
                sp.age > self.species_grace_period
                and sp.size < self.min_species_size
            ):
                undersized_ids.append(sid)

        best_species_id = self._current_best_species_id()
        removable = [
            sid for sid in set(stagnant_ids + undersized_ids)
            if sid != best_species_id
        ]
        if len(removable) < len(self.species):
            for sid in removable:
                del self.species[sid]

        # Paso 5: Actualizar representantes
        for sp in self.species.values():
            sp.update_representative()

        # Paso 6: Calcular fitness ajustado
        for sp in self.species.values():
            sp.compute_adjusted_fitness()

        self._adapt_threshold()

    def get_offspring_counts(self, total_population: int) -> dict[int, int]:
        """
        Calcula cuantos offspring le corresponden a cada especie.

        La cantidad es PROPORCIONAL al fitness ajustado promedio
        de la especie. Especies con mejor fitness ajustado
        producen mas offspring.

        Parameters
        ----------
        total_population : int
            Tamano total de la siguiente generacion

        Returns
        -------
        dict[int, int]
            {species_id: num_offspring}
        """
        if not self.species:
            return {}

        raw_scores = {
            sid: sp.avg_fitness for sid, sp in self.species.items()
        }
        min_score = min(raw_scores.values())
        shifted = {
            sid: max(0.0, score - min_score)
            for sid, score in raw_scores.items()
        }

        if sum(shifted.values()) <= 1e-12:
            shifted = {sid: 1.0 for sid in self.species}
        else:
            shifted = {sid: score + 1e-9 for sid, score in shifted.items()}

        total_score = sum(shifted.values())
        quotas = {
            sid: total_population * score / total_score
            for sid, score in shifted.items()
        }
        counts = {sid: math.floor(q) for sid, q in quotas.items()}
        remaining = total_population - sum(counts.values())

        remainders = sorted(
            self.species,
            key=lambda sid: (
                quotas[sid] - counts[sid],
                raw_scores[sid]
            ),
            reverse=True
        )
        for sid in remainders[:remaining]:
            counts[sid] += 1

        best_sid = self._current_best_species_id()
        if best_sid is not None and counts.get(best_sid, 0) == 0:
            donor = max(
                (sid for sid in counts if sid != best_sid),
                key=lambda sid: counts[sid],
                default=None
            )
            if donor is not None and counts[donor] > 0:
                counts[donor] -= 1
                counts[best_sid] = 1

        return counts

    def _current_best_species_id(self) -> Optional[int]:
        """Identifica la especie que contiene al mejor genoma evaluado."""
        best_sid = None
        best_fitness = float("-inf")
        for sid, sp in self.species.items():
            best = sp.get_best()
            if best is not None and best.fitness is not None:
                if best.fitness > best_fitness:
                    best_sid = sid
                    best_fitness = best.fitness
        return best_sid

    def _adapt_threshold(self):
        """Ajusta suavemente el umbral para mantener diversidad util."""
        count = len(self.species)
        if count < self.target_species_min:
            self.threshold = max(
                self.threshold_min, self.threshold - self.threshold_step
            )
        elif count > self.target_species_max:
            self.threshold = min(
                self.threshold_max, self.threshold + self.threshold_step
            )

    def get_best_genome(self) -> Optional[Genome]:
        """Retorna el mejor genoma de toda la poblacion."""
        best = None
        best_fitness = float("-inf")

        for sp in self.species.values():
            sp_best = sp.get_best()
            if sp_best and sp_best.fitness is not None:
                if sp_best.fitness > best_fitness:
                    best = sp_best
                    best_fitness = sp_best.fitness

        return best

    def get_stats(self) -> dict:
        """Estadisticas de especiacion para logging."""
        return {
            "num_species": len(self.species),
            "compatibility_threshold": round(self.threshold, 4),
            "species_sizes": {
                sid: sp.size for sid, sp in self.species.items()
            },
            "species_avg_fitness": {
                sid: round(sp.avg_fitness, 4)
                for sid, sp in self.species.items()
            },
            "species_stagnation": {
                sid: sp.generations_without_improvement
                for sid, sp in self.species.items()
            },
        }
