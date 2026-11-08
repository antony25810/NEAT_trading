"""
============================================================
network.py - Red Neuronal Feedforward desde un Genoma NEAT
============================================================

QUE HACE ESTE MODULO?
---------------------
Toma un Genome (lista de nodos y conexiones) y construye una red
neuronal funcional que puede procesar datos de entrada y producir
una salida.

POR QUE SEPARAR GENOME DE NETWORK?
------------------------------------
El Genome es la REPRESENTACION GENETICA (el ADN).
La Network es el FENOTIPO (el organismo funcional).

El genoma se usa para: cruce, mutacion, distancia, almacenamiento.
La red se usa para: evaluar datos, tomar decisiones de trading.

Separar ambos es mas limpio y eficiente:
- Construyes la red UNA vez desde el genoma
- La usas MILES de veces para evaluar datos (un timestep por barra)

ORDENAMIENTO TOPOLOGICO
-------------------------
Las redes NEAT pueden tener topologias arbitrarias. No son capas
ordenadas como en redes tradicionales. Ejemplo:

  Input_0 ──► Hidden_1 ──► Output
  Input_0 ──────────────── Output  (conexion directa que salta Hidden_1)

Para evaluarla correctamente, necesitamos calcular los nodos en
el ORDEN correcto: primero los que no dependen de nadie (inputs),
luego los que dependen de los inputs, etc.

Esto se llama "ordenamiento topologico" (topological sort) y es
un algoritmo clasico de grafos (algoritmo de Kahn).
"""

import numpy as np
from typing import Optional

from src.neat.genome import Genome, ConnectionGene, NodeGene


# =================================================================
# FUNCIONES DE ACTIVACION
# =================================================================
# Cada funcion toma un valor numerico y produce otro.
# Son las "no-linealidades" que permiten a la red aprender
# patrones complejos. Sin ellas, toda la red seria una
# transformacion lineal (sin importar cuantas capas tenga).

def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid: f(x) = 1 / (1 + e^(-x))

    Rango: (0, 1)
    Forma: curva en S
    Uso: Output de confianza y tamano de posicion [0, 1]

    Caracteristicas:
    - Aplasta valores extremos hacia 0 o 1
    - Gradiente suave (bueno para optimizacion)
    - Satura para |x| > 5 (valor casi constante)
    """
    # Clip para evitar overflow en exp()
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def tanh_activation(x: np.ndarray) -> np.ndarray:
    """
    Tanh: f(x) = tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))

    Rango: (-1, 1)
    Forma: curva en S centrada en 0
    Uso: Output de direccion [-1, +1] (venta a compra)

    Diferencia vs sigmoid:
    - Centrada en 0 (sigmoid esta centrada en 0.5)
    - Valores negativos tienen significado (venta)
    """
    return np.tanh(x)


def relu(x: np.ndarray) -> np.ndarray:
    """
    ReLU: f(x) = max(0, x)

    Rango: [0, +inf)
    Forma: lineal para x > 0, cero para x <= 0
    Uso: Nodos ocultos (la mas popular en deep learning)

    Ventaja: No satura para valores grandes positivos
    Desventaja: "Neuronas muertas" - si x < 0 siempre, la neurona
    nunca se activa y no contribuye a la red.
    """
    return np.maximum(0, x)


def leaky_relu(x: np.ndarray) -> np.ndarray:
    """
    Leaky ReLU: f(x) = x if x > 0 else 0.01 * x

    Rango: (-inf, +inf) pero valores negativos son muy pequenos
    Resuelve el problema de "neuronas muertas" del ReLU:
    incluso para x < 0, hay un gradiente pequeno (0.01).
    """
    return np.where(x > 0, x, 0.01 * x)


def identity(x: np.ndarray) -> np.ndarray:
    """
    Identity: f(x) = x

    Sin transformacion. Pasa el valor tal cual.
    Usada para nodos de input (no queremos transformar los indicadores).
    Tambien puede ser util en nodos ocultos para crear "bypasses lineales".
    """
    return x


# Mapa de nombre -> funcion para busqueda rapida
ACTIVATION_FUNCTIONS = {
    "sigmoid": sigmoid,
    "tanh": tanh_activation,
    "relu": relu,
    "leaky_relu": leaky_relu,
    "identity": identity,
}


# =================================================================
# RED NEURONAL FEEDFORWARD
# =================================================================

class FeedForwardNetwork:
    """
    Red neuronal feedforward construida desde un genoma NEAT.

    COMO FUNCIONA:
    1. Se construye UNA VEZ desde un Genome (metodo from_genome)
    2. Se precomputa el orden de evaluacion (topological sort)
    3. Se puede evaluar MUCHAS veces con datos diferentes (metodo activate)

    La evaluacion es VECTORIZADA: puede procesar multiples timesteps
    de una sola vez usando NumPy, lo cual es MUCHO mas rapido que
    un loop en Python.

    Uso:
    ----
    >>> network = FeedForwardNetwork.from_genome(genome)
    >>> # Evaluar un solo timestep
    >>> outputs = network.activate(input_values)
    >>> # Evaluar multiples timesteps de golpe
    >>> outputs = network.activate_batch(input_matrix)
    """

    def __init__(self):
        # Mapa de node_id a funcion de activacion
        self._activations: dict[int, callable] = {}
        self._biases: dict[int, float] = {}

        # Lista de (in_node, out_node, weight) para conexiones activas
        self._connections: list[tuple[int, int, float]] = []

        # Orden topologico de evaluacion (lista de node_ids)
        self._eval_order: list[int] = []

        # IDs de nodos de input y output
        self._input_ids: list[int] = []
        self._output_ids: list[int] = []

        # IDs de inputs que realmente reciben datos (activos en indicator_mask)
        self._active_input_ids: list[int] = []

        # Para evaluacion: mapa de node_id a indice en el array de valores
        self._node_to_idx: dict[int, int] = {}

    @classmethod
    def from_genome(
        cls,
        genome: Genome,
        active_input_ids: Optional[list[int]] = None
    ) -> "FeedForwardNetwork":
        """
        Construye una red funcional desde un genoma.

        PASOS:
        1. Extraer nodos y conexiones activas del genoma
        2. Asignar funciones de activacion
        3. Calcular orden topologico (en que orden evaluar los nodos)
        4. Preparar estructuras para evaluacion rapida

        Parameters
        ----------
        genome : Genome
            Genoma del cual construir la red
        active_input_ids : list[int], opcional
            IDs de inputs activos. Si None, se calculan del indicator_mask.

        Returns
        -------
        FeedForwardNetwork
            Red lista para evaluar
        """
        net = cls()

        # --- Nodos ---
        net._input_ids = sorted(genome.input_node_ids)
        net._output_ids = sorted(genome.output_node_ids)

        # Inputs activos (los que tienen indicadores activos)
        if active_input_ids is not None:
            net._active_input_ids = sorted(active_input_ids)
        else:
            net._active_input_ids = sorted(genome.active_input_ids)

        # Funciones de activacion
        for node_id, node in genome.nodes.items():
            act_name = node.activation
            if act_name in ACTIVATION_FUNCTIONS:
                net._activations[node_id] = ACTIVATION_FUNCTIONS[act_name]
            else:
                net._activations[node_id] = identity
            net._biases[node_id] = node.bias

        # --- Conexiones activas ---
        net._connections = [
            (c.in_node, c.out_node, c.weight)
            for c in genome.connections.values()
            if c.enabled
        ]

        # --- Orden topologico ---
        net._eval_order = net._topological_sort(genome)

        # --- Mapa de nodo a indice para arrays ---
        all_node_ids = sorted(genome.nodes.keys())
        net._node_to_idx = {nid: idx for idx, nid in enumerate(all_node_ids)}

        return net

    def activate(self, inputs: dict[int, float]) -> dict[int, float]:
        """
        Evalua la red con un conjunto de inputs.

        Parameters
        ----------
        inputs : dict[int, float]
            Mapa de node_id -> valor para cada input activo.
            Ejemplo: {0: 0.5, 1: -0.3, 5: 0.8}  (RSI=0.5, MACD=-0.3, ADX=0.8)

        Returns
        -------
        dict[int, float]
            Mapa de node_id -> valor para cada output.
            Ejemplo: {18: 0.7, 19: 0.4, 20: 0.2}  (Dir=0.7, Conf=0.4, Size=0.2)
        """
        # Valores de cada nodo (inicializados en 0)
        values: dict[int, float] = {}

        # Establecer valores de input
        for node_id in self._input_ids:
            values[node_id] = inputs.get(node_id, 0.0)

        # Evaluar nodos en orden topologico
        for node_id in self._eval_order:
            if node_id in self._input_ids:
                continue  # los inputs ya tienen valor

            # Sumar todas las entradas a este nodo
            incoming_sum = self._biases.get(node_id, 0.0)
            for in_node, out_node, weight in self._connections:
                if out_node == node_id:
                    incoming_sum += values.get(in_node, 0.0) * weight

            # Aplicar funcion de activacion
            act_func = self._activations.get(node_id, identity)
            values[node_id] = float(act_func(np.array(incoming_sum)))

        # Retornar solo los outputs
        return {
            node_id: values.get(node_id, 0.0)
            for node_id in self._output_ids
        }

    def activate_batch(self, input_matrix: np.ndarray) -> np.ndarray:
        """
        Evalua la red para MULTIPLES timesteps de una vez.

        Esta es la version VECTORIZADA, mucho mas rapida que llamar
        activate() en un loop. Usa operaciones NumPy sobre matrices.

        Parameters
        ----------
        input_matrix : np.ndarray
            Matriz de shape (num_timesteps, num_active_inputs).
            Cada fila es un vector de inputs para un timestep.
            Las columnas corresponden a los inputs activos EN ORDEN.

        Returns
        -------
        np.ndarray
            Matriz de shape (num_timesteps, num_outputs).
            Cada fila es el vector de outputs para ese timestep.
        """
        num_timesteps = input_matrix.shape[0]
        num_nodes = len(self._node_to_idx)

        # Crear matriz de valores: (timesteps, num_nodes)
        values = np.zeros((num_timesteps, num_nodes))

        # Establecer valores de input
        for col_idx, input_id in enumerate(self._active_input_ids):
            if col_idx < input_matrix.shape[1]:
                node_idx = self._node_to_idx[input_id]
                values[:, node_idx] = input_matrix[:, col_idx]

        # Evaluar nodos en orden topologico
        for node_id in self._eval_order:
            if node_id in self._input_ids:
                continue

            node_idx = self._node_to_idx[node_id]

            # Sumar todas las entradas ponderadas
            incoming = np.full(
                num_timesteps, self._biases.get(node_id, 0.0), dtype=float
            )
            for in_node, out_node, weight in self._connections:
                if out_node == node_id:
                    in_idx = self._node_to_idx[in_node]
                    incoming += values[:, in_idx] * weight

            # Aplicar activacion
            act_func = self._activations.get(node_id, identity)
            values[:, node_idx] = act_func(incoming)

        # Extraer solo los outputs
        output_indices = [self._node_to_idx[oid] for oid in self._output_ids]
        return values[:, output_indices]

    def _topological_sort(self, genome: Genome) -> list[int]:
        """
        Ordena los nodos en orden topologico para evaluacion correcta.

        ALGORITMO DE KAHN:
        1. Encontrar nodos sin dependencias (inputs) -> cola
        2. Para cada nodo en la cola:
           a. Agregarlo al resultado
           b. Para cada conexion que SALE de este nodo:
              - Reducir el conteo de dependencias del nodo destino
              - Si el destino ya no tiene dependencias, agregarlo a la cola
        3. Repetir hasta que la cola este vacia

        Esto garantiza que cuando evaluamos un nodo, todos sus inputs
        ya tienen valor calculado.

        Parameters
        ----------
        genome : Genome
            Genoma del cual extraer la topologia

        Returns
        -------
        list[int]
            Lista de node_ids en orden de evaluacion
        """
        # Construir grafo de dependencias
        all_nodes = set(genome.nodes.keys())
        in_degree = {n: 0 for n in all_nodes}

        # Contar dependencias (cuantas conexiones LLEGAN a cada nodo)
        active_conns = [(c.in_node, c.out_node)
                        for c in genome.connections.values()
                        if c.enabled]

        for in_node, out_node in active_conns:
            if out_node in in_degree:
                in_degree[out_node] += 1

        # Nodos sin dependencias (inputs y nodos desconectados)
        queue = [n for n in all_nodes if in_degree[n] == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)

            for in_node, out_node in active_conns:
                if in_node == node and out_node in in_degree:
                    in_degree[out_node] -= 1
                    if in_degree[out_node] == 0:
                        queue.append(out_node)

        # Si hay nodos que no se alcanzaron, agregarlos al final
        # (puede pasar con ciclos o nodos desconectados)
        for n in all_nodes:
            if n not in result:
                result.append(n)

        return result

    @property
    def num_active_inputs(self) -> int:
        return len(self._active_input_ids)

    @property
    def num_outputs(self) -> int:
        return len(self._output_ids)

    def get_structure_info(self) -> dict:
        """Informacion de la estructura de la red para debug."""
        return {
            "input_ids": self._input_ids,
            "active_input_ids": self._active_input_ids,
            "output_ids": self._output_ids,
            "num_connections": len(self._connections),
            "eval_order": self._eval_order,
            "total_nodes": len(self._activations),
        }
