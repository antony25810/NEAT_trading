"""
============================================================
genome.py - Representacion del Genoma NEAT + Trading
============================================================

QUE ES UN GENOMA EN NEAT?
--------------------------
En biologia, el genoma es la informacion genetica completa de un
organismo. En NEAT, el genoma codifica:
  1. La TOPOLOGIA de la red neuronal (que nodos existen y como se conectan)
  2. Los PESOS de cada conexion

En nuestro proyecto, extendemos el genoma para incluir:
  3. Parametros de GESTION DE RIESGO (stop loss, take profit, etc.)
  4. MASCARA DE INDICADORES (cuales inputs estan activos)

ESTRUCTURA DEL GENOMA
---------------------
El genoma tiene dos tipos de genes:

ConnectionGene: Representa una conexion entre dos nodos.
  - in_node: nodo de entrada
  - out_node: nodo de salida
  - weight: peso de la conexion (multiplicador de la senal)
  - enabled: si la conexion esta activa o desactivada
  - innovation: innovation number (marcador historico)

NodeGene: Representa un nodo (neurona) en la red.
  - node_id: identificador unico
  - node_type: "input", "hidden", o "output"
  - activation: funcion de activacion ("sigmoid", "tanh", etc.)

CODIFICACION DE IDs DE NODOS
-----------------------------
Los IDs de nodos siguen una convencion fija:

  IDs 0 .. num_inputs-1        → Nodos de INPUT (indicadores)
  IDs num_inputs .. num_inputs+num_outputs-1  → Nodos de OUTPUT
  IDs >= num_inputs+num_outputs → Nodos HIDDEN (creados por mutacion)

Ejemplo con 18 inputs y 3 outputs:
  0-17: inputs (RSI=0, MACD=1, ATR=2, ...)
  18: Output_Direccion
  19: Output_Confianza
  20: Output_Tamano
  21+: nodos ocultos (aparecen por mutacion)
"""

import copy
import random
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.neat.innovation import InnovationTracker


# =================================================================
# GENES
# =================================================================

@dataclass
class ConnectionGene:
    """
    Un gen de conexion: enlace dirigido entre dos nodos.

    EN BIOLOGIA: seria como un axon que conecta dos neuronas.
    EN REDES NEURONALES: es el "cable" con un peso multiplicador.

    La senal fluye: in_node --(*weight)--> out_node

    Si enabled=False, la conexion existe en el genoma pero NO
    transmite senal. Esto pasa cuando un nodo se inserta en medio
    de una conexion: la conexion original se desactiva y se crean
    dos nuevas.
    """
    in_node: int
    out_node: int
    weight: float
    enabled: bool
    innovation: int
    disabled_by_indicator: bool = False

    def copy(self) -> "ConnectionGene":
        """Crea una copia independiente de este gen."""
        return ConnectionGene(
            in_node=self.in_node,
            out_node=self.out_node,
            weight=self.weight,
            enabled=self.enabled,
            innovation=self.innovation,
            disabled_by_indicator=self.disabled_by_indicator
        )

    def to_dict(self) -> dict:
        """Serializa el gen de conexion a diccionario."""
        return {
            "in_node": self.in_node,
            "out_node": self.out_node,
            "weight": self.weight,
            "enabled": self.enabled,
            "innovation": self.innovation,
            "disabled_by_indicator": self.disabled_by_indicator,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConnectionGene":
        """Reconstruye un gen de conexion desde un diccionario."""
        return cls(
            in_node=int(data["in_node"]),
            out_node=int(data["out_node"]),
            weight=float(data["weight"]),
            enabled=bool(data["enabled"]),
            innovation=int(data["innovation"]),
            disabled_by_indicator=bool(data.get("disabled_by_indicator", False)),
        )


@dataclass
class NodeGene:
    """
    Un gen de nodo: una neurona en la red.

    Tipos:
    - "input": recibe datos del exterior (indicadores tecnicos).
      No tiene funcion de activacion (identity).
    - "output": produce la decision final (direccion, confianza, tamano).
      Tiene activacion FIJA (tanh para direccion, sigmoid para los demas).
    - "hidden": nodo intermedio creado por mutacion.
      Tiene activacion MUTABLE (puede cambiar entre sigmoid, tanh, relu, etc.)
    """
    node_id: int
    node_type: str   # "input", "output", "hidden"
    activation: str  # "sigmoid", "tanh", "relu", "leaky_relu", "identity"
    bias: float = 0.0

    def copy(self) -> "NodeGene":
        """Crea una copia independiente de este gen."""
        return NodeGene(
            node_id=self.node_id,
            node_type=self.node_type,
            activation=self.activation,
            bias=self.bias
        )

    def to_dict(self) -> dict:
        """Serializa el gen de nodo a diccionario."""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "activation": self.activation,
            "bias": self.bias,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NodeGene":
        """Reconstruye un gen de nodo desde un diccionario."""
        return cls(
            node_id=int(data["node_id"]),
            node_type=str(data["node_type"]),
            activation=str(data["activation"]),
            bias=float(data.get("bias", 0.0)),
        )


# =================================================================
# PARAMETROS DE TRADING
# =================================================================

@dataclass
class TradingParams:
    """
    Parametros de gestion de riesgo codificados en el genoma.

    Estos valores evolucionan junto con la red neuronal.
    Cada individuo tiene su propia combinacion de parametros,
    y la evolucion descubre cuales funcionan mejor.

    Todos los valores son proporciones (0.0 a 1.0 o rangos especificos).
    """
    stop_loss: float = 0.05        # % de perdida para cerrar posicion
    take_profit: float = 0.10      # % de ganancia para cerrar posicion
    trailing_stop: float = 0.03    # % de trailing stop
    max_drawdown: float = 0.20     # Drawdown maximo permitido
    position_size: float = 0.20    # Fraccion del capital por operacion

    def copy(self) -> "TradingParams":
        return TradingParams(
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
            trailing_stop=self.trailing_stop,
            max_drawdown=self.max_drawdown,
            position_size=self.position_size
        )

    def to_dict(self) -> dict:
        return {
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "trailing_stop": self.trailing_stop,
            "max_drawdown": self.max_drawdown,
            "position_size": self.position_size,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TradingParams":
        return cls(
            stop_loss=float(data.get("stop_loss", 0.05)),
            take_profit=float(data.get("take_profit", 0.10)),
            trailing_stop=float(data.get("trailing_stop", 0.03)),
            max_drawdown=float(data.get("max_drawdown", 0.20)),
            position_size=float(data.get("position_size", 0.20)),
        )

    def distance(self, other: "TradingParams") -> float:
        """
        Distancia entre dos conjuntos de parametros de trading.

        Usada para la especiacion: individuos con parametros muy
        diferentes de riesgo son estrategias diferentes y deberian
        estar en especies separadas.

        Returns
        -------
        float
            Suma de diferencias absolutas normalizadas
        """
        diffs = [
            abs(self.stop_loss - other.stop_loss) / 0.14,       # rango [0.01, 0.15]
            abs(self.take_profit - other.take_profit) / 0.28,    # rango [0.02, 0.30]
            abs(self.trailing_stop - other.trailing_stop) / 0.095,  # rango [0.005, 0.10]
            abs(self.max_drawdown - other.max_drawdown) / 0.20,  # rango [0.10, 0.30]
            abs(self.position_size - other.position_size) / 0.45,  # rango [0.05, 0.50]
        ]
        return sum(diffs) / len(diffs)  # promedio normalizado en [0, 1]


# =================================================================
# GENOMA COMPLETO
# =================================================================

class Genome:
    """
    Genoma completo de un individuo: red neuronal + trading + indicadores.

    Este es el objeto central de todo NEAT. Codifica:
    1. Topologia de la red (nodos y conexiones)
    2. Pesos de las conexiones
    3. Parametros de gestion de riesgo
    4. Mascara de indicadores activos

    Uso:
    ----
    >>> genome = Genome(genome_id=0, num_inputs=18, num_outputs=3)
    >>> genome.initialize_minimal(indicator_names=["RSI", "MACD", ...])
    >>> print(genome.summary())
    """

    def __init__(
        self,
        genome_id: int,
        num_inputs: int = 18,
        num_outputs: int = 3
    ):
        """
        Crea un genoma vacio (sin genes).

        Llama a initialize_minimal() despues para crear la red inicial.

        Parameters
        ----------
        genome_id : int
            Identificador unico de este genoma
        num_inputs : int
            Numero total de indicadores posibles (tipicamente 18)
        num_outputs : int
            Numero de salidas (3: direccion, confianza, tamano)
        """
        self.genome_id = genome_id
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        # Genes estructurales
        self.connections: dict[int, ConnectionGene] = {}  # innov -> ConnectionGene
        self.nodes: dict[int, NodeGene] = {}              # node_id -> NodeGene

        # Parametros de trading
        self.trading_params = TradingParams()

        # Mascara de indicadores: {nombre: activo}
        # Los indicadores desactivados NO se conectan a la red
        self.indicator_mask: dict[str, bool] = {}

        # Fitness (se asigna despues de la evaluacion)
        self.fitness: Optional[float] = None
        self.adjusted_fitness: Optional[float] = None

        # Especie a la que pertenece (se asigna durante especiacion)
        self.species_id: Optional[int] = None

    def initialize_minimal(
        self,
        indicator_names: list[str],
        initial_active: int = 4,
        output_activations: Optional[list[str]] = None,
        tracker: Optional["InnovationTracker"] = None,
        generation: int = 0,
        risk_config: Optional[dict] = None
    ):
        """
        Inicializa el genoma con la red mas simple posible.

        COMPLEXIFICACION MINIMA (Pilar 3 de NEAT):
        - Crea nodos de input para TODOS los indicadores
        - Activa solo 'initial_active' indicadores al azar
        - Crea conexiones directas SOLO de inputs activos a outputs
        - SIN nodos ocultos
        - Pesos aleatorios iniciales

        Parameters
        ----------
        indicator_names : list[str]
            Nombres de todos los indicadores disponibles
        initial_active : int
            Cuantos indicadores activar al inicio (default: 4)
        output_activations : list[str], opcional
            Funciones de activacion de los outputs.
            Default: ["tanh", "sigmoid", "sigmoid"]
            (Direccion[-1,1], Confianza[0,1], Tamano[0,1])
        """
        if output_activations is None:
            output_activations = ["tanh", "sigmoid", "sigmoid"]

        assert len(output_activations) == self.num_outputs
        assert len(indicator_names) == self.num_inputs

        # --- Crear nodos de INPUT ---
        # IDs 0 .. num_inputs-1
        for i in range(self.num_inputs):
            self.nodes[i] = NodeGene(
                node_id=i,
                node_type="input",
                activation="identity"  # inputs no transforman la senal
            )

        # --- Crear nodos de OUTPUT ---
        # IDs num_inputs .. num_inputs+num_outputs-1
        for i in range(self.num_outputs):
            node_id = self.num_inputs + i
            self.nodes[node_id] = NodeGene(
                node_id=node_id,
                node_type="output",
                activation=output_activations[i],
                bias=0.0
            )

        # --- Mascara de indicadores ---
        # Garantizar que la tendencia macro (distancia_SMA200 o Pendiente_SMA200) esté siempre activa
        mandatory_indices = []
        macro_indicator = (
            "distancia_SMA200" if "distancia_SMA200" in indicator_names
            else ("Pendiente_SMA200" if "Pendiente_SMA200" in indicator_names else None)
        )
        if macro_indicator is not None:
            mandatory_indices.append(indicator_names.index(macro_indicator))

        remaining_slots = max(0, min(initial_active, self.num_inputs) - len(mandatory_indices))
        other_indices = [idx for idx in range(self.num_inputs) if idx not in mandatory_indices]
        sampled_indices = random.sample(other_indices, min(remaining_slots, len(other_indices)))
        active_indices = mandatory_indices + sampled_indices

        for i, name in enumerate(indicator_names):
            self.indicator_mask[name] = (i in active_indices)

        # --- Conexiones iniciales ---
        # Solo de inputs ACTIVOS a outputs
        for input_idx in active_indices:
            for output_offset in range(self.num_outputs):
                output_id = self.num_inputs + output_offset
                if tracker is not None:
                    innov = tracker.get_connection_innovation(
                        input_idx, output_id, generation
                    )
                else:
                    # Mapeo determinista por endpoints para uso independiente.
                    innov = input_idx * self.num_outputs + output_offset + 1
                weight = random.gauss(0, 1.0)  # peso aleatorio N(0, 1)
                self.connections[innov] = ConnectionGene(
                    in_node=input_idx,
                    out_node=output_id,
                    weight=weight,
                    enabled=True,
                    innovation=innov
                )

        # --- Parametros de trading aleatorios ---
        risk = risk_config or {}
        stop_loss = risk.get("stop_loss", {})
        take_profit = risk.get("take_profit", {})
        trailing_stop = risk.get("trailing_stop", {})
        position_size = risk.get("position_size", {})
        max_drawdown = risk.get("max_drawdown", {})
        if not isinstance(max_drawdown, dict):
            max_drawdown = {"min": 0.20, "max": float(max_drawdown)}

        self.trading_params = TradingParams(
            stop_loss=random.uniform(
                stop_loss.get("min", 0.01), stop_loss.get("max", 0.15)
            ),
            take_profit=random.uniform(
                take_profit.get("min", 0.02), take_profit.get("max", 0.30)
            ),
            trailing_stop=random.uniform(
                trailing_stop.get("min", 0.005), trailing_stop.get("max", 0.10)
            ),
            max_drawdown=random.uniform(
                max_drawdown.get("min", 0.20), max_drawdown.get("max", 0.45)
            ),
            position_size=random.uniform(
                position_size.get("min", 0.40), position_size.get("max", 1.0)
            )
        )

    # =============================================================
    # PROPIEDADES UTILES
    # =============================================================

    @property
    def input_node_ids(self) -> list[int]:
        """IDs de los nodos de input."""
        return [n.node_id for n in self.nodes.values()
                if n.node_type == "input"]

    @property
    def output_node_ids(self) -> list[int]:
        """IDs de los nodos de output."""
        return [n.node_id for n in self.nodes.values()
                if n.node_type == "output"]

    @property
    def hidden_node_ids(self) -> list[int]:
        """IDs de los nodos ocultos."""
        return [n.node_id for n in self.nodes.values()
                if n.node_type == "hidden"]

    @property
    def active_connections(self) -> list[ConnectionGene]:
        """Solo las conexiones habilitadas."""
        return [c for c in self.connections.values() if c.enabled]

    @property
    def active_indicator_names(self) -> list[str]:
        """Nombres de los indicadores activos."""
        return [name for name, active in self.indicator_mask.items() if active]

    @property
    def active_input_ids(self) -> list[int]:
        """IDs de nodos input que tienen indicadores activos."""
        names = list(self.indicator_mask.keys())
        return [i for i, name in enumerate(names)
                if self.indicator_mask.get(name, False)]

    @property
    def complexity(self) -> int:
        """Complejidad del genoma = num_hidden + num_active_connections."""
        return len(self.hidden_node_ids) + len(self.active_connections)

    @property
    def complexity_score(self) -> float:
        """Costo estructural ponderado para favorecer soluciones compactas."""
        return (
            3.0 * len(self.hidden_node_ids)
            + len(self.active_connections)
            + 2.0 * len(self.active_indicator_names)
        )

    # =============================================================
    # DISTANCIA GENOMICA (para especiacion)
    # =============================================================

    def distance(
        self,
        other: "Genome",
        c1: float = 1.0,
        c2: float = 1.0,
        c3: float = 0.4,
        c4: float = 0.5,
        c5: float = 0.3
    ) -> float:
        """
        Calcula la distancia genomica entre dos genomas.

        Esta funcion determina si dos genomas pertenecen a la misma
        especie o no. Combina:
        - Diferencia TOPOLOGICA (genes excess, disjoint, matching weights)
        - Diferencia en PARAMETROS DE TRADING
        - Diferencia en INDICADORES ACTIVOS

        Formula:
                     c1 * E    c2 * D
        distance = -------- + -------- + c3 * W + c4 * T + c5 * I
                      N          N

        Donde:
          E = numero de genes excess
          D = numero de genes disjoint
          N = max(num_genes_A, num_genes_B), minimo 1
          W = diferencia promedio de pesos en genes matching
          T = distancia de parametros de trading [0, 1]
          I = fraccion de indicadores que difieren [0, 1]
          c1..c5 = coeficientes de ponderacion

        Parameters
        ----------
        other : Genome
            Otro genoma para comparar
        c1, c2, c3 : float
            Coeficientes NEAT originales (topologia)
        c4 : float
            Coeficiente para parametros de trading
        c5 : float
            Coeficiente para mascara de indicadores

        Returns
        -------
        float
            Distancia genomica (0 = identicos, mayor = mas diferentes)
        """
        # --- Parte 1: Distancia topologica (NEAT original) ---
        innovs_a = set(self.connections.keys())
        innovs_b = set(other.connections.keys())

        if not innovs_a and not innovs_b:
            # Ambos sin conexiones
            excess = 0
            disjoint = 0
            avg_weight_diff = 0.0
        else:
            matching = innovs_a & innovs_b
            all_innovs = innovs_a | innovs_b
            non_matching = all_innovs - matching

            if innovs_a and innovs_b:
                max_innov_a = max(innovs_a)
                max_innov_b = max(innovs_b)
                threshold = min(max_innov_a, max_innov_b)

                # Excess: genes no-matching que estan MAS ALLA del rango del otro
                # Disjoint: genes no-matching DENTRO del rango de ambos
                excess = sum(1 for i in non_matching if i > threshold)
                disjoint = len(non_matching) - excess
            else:
                excess = len(non_matching)
                disjoint = 0

            # Diferencia promedio de pesos en genes matching
            if matching:
                weight_diffs = [
                    abs(self.connections[i].weight - other.connections[i].weight)
                    for i in matching
                ]
                avg_weight_diff = sum(weight_diffs) / len(weight_diffs)
            else:
                avg_weight_diff = 0.0

        # N = normalizacion por tamano del genoma
        N = max(len(innovs_a), len(innovs_b), 1)

        topo_distance = (c1 * excess / N) + (c2 * disjoint / N) + (c3 * avg_weight_diff)

        # --- Parte 2: Distancia en parametros de trading ---
        trading_distance = self.trading_params.distance(other.trading_params)

        # --- Parte 3: Distancia en mascara de indicadores ---
        if self.indicator_mask and other.indicator_mask:
            total_indicators = len(self.indicator_mask)
            differing = sum(
                1 for name in self.indicator_mask
                if self.indicator_mask.get(name, False) !=
                   other.indicator_mask.get(name, False)
            )
            indicator_distance = differing / max(total_indicators, 1)
        else:
            indicator_distance = 0.0

        # --- Distancia total ---
        return topo_distance + (c4 * trading_distance) + (c5 * indicator_distance)

    # =============================================================
    # UTILIDADES
    # =============================================================

    def copy(self) -> "Genome":
        """Crea una copia profunda e independiente del genoma."""
        new = Genome(
            genome_id=self.genome_id,
            num_inputs=self.num_inputs,
            num_outputs=self.num_outputs
        )
        new.connections = {k: v.copy() for k, v in self.connections.items()}
        new.nodes = {k: v.copy() for k, v in self.nodes.items()}
        new.trading_params = self.trading_params.copy()
        new.indicator_mask = dict(self.indicator_mask)
        new.fitness = self.fitness
        new.adjusted_fitness = self.adjusted_fitness
        new.species_id = self.species_id
        return new

    def to_dict(self) -> dict:
        """Serializa el genoma completo a un diccionario para JSON."""
        return {
            "genome_id": self.genome_id,
            "num_inputs": self.num_inputs,
            "num_outputs": self.num_outputs,
            "fitness": self.fitness,
            "adjusted_fitness": self.adjusted_fitness,
            "species_id": self.species_id,
            "trading_params": self.trading_params.to_dict(),
            "indicator_mask": dict(self.indicator_mask),
            "nodes": {
                str(node_id): node.to_dict()
                for node_id, node in sorted(self.nodes.items())
            },
            "connections": {
                str(innov): conn.to_dict()
                for innov, conn in sorted(self.connections.items())
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Genome":
        """Reconstruye un objeto Genome a partir de un diccionario."""
        genome = cls(
            genome_id=int(data["genome_id"]),
            num_inputs=int(data.get("num_inputs", 18)),
            num_outputs=int(data.get("num_outputs", 3)),
        )
        genome.fitness = (
            float(data["fitness"]) if data.get("fitness") is not None else None
        )
        genome.adjusted_fitness = (
            float(data["adjusted_fitness"])
            if data.get("adjusted_fitness") is not None else None
        )
        genome.species_id = (
            int(data["species_id"]) if data.get("species_id") is not None else None
        )

        if "trading_params" in data:
            genome.trading_params = TradingParams.from_dict(data["trading_params"])

        if "indicator_mask" in data:
            genome.indicator_mask = {
                str(k): bool(v) for k, v in data["indicator_mask"].items()
            }

        if "nodes" in data:
            nodes_data = data["nodes"]
            if isinstance(nodes_data, dict):
                genome.nodes = {
                    int(k): NodeGene.from_dict(v) for k, v in nodes_data.items()
                }
            elif isinstance(nodes_data, list):
                genome.nodes = {
                    int(n["node_id"]): NodeGene.from_dict(n) for n in nodes_data
                }

        if "connections" in data:
            conns_data = data["connections"]
            if isinstance(conns_data, dict):
                genome.connections = {
                    int(k): ConnectionGene.from_dict(v) for k, v in conns_data.items()
                }
            elif isinstance(conns_data, list):
                genome.connections = {
                    int(c["innovation"]): ConnectionGene.from_dict(c) for c in conns_data
                }

        return genome

    def summary(self) -> dict:
        """Resumen del genoma para logging/debug."""
        return {
            "id": self.genome_id,
            "nodes": len(self.nodes),
            "hidden_nodes": len(self.hidden_node_ids),
            "connections": len(self.connections),
            "active_connections": len(self.active_connections),
            "active_indicators": len(self.active_indicator_names),
            "complexity": self.complexity,
            "fitness": self.fitness,
            "species": self.species_id,
            "trading_params": self.trading_params.to_dict(),
            "indicators": self.active_indicator_names,
        }

    def __repr__(self) -> str:
        return (
            f"Genome(id={self.genome_id}, "
            f"nodes={len(self.nodes)}, "
            f"hidden={len(self.hidden_node_ids)}, "
            f"conns={len(self.active_connections)}, "
            f"indicators={len(self.active_indicator_names)}, "
            f"fitness={self.fitness})"
        )
