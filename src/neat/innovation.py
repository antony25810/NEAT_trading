"""
============================================================
innovation.py - Contador de Innovation Numbers
============================================================

QUE SON LOS INNOVATION NUMBERS?
--------------------------------
Cada vez que una mutacion estructural crea algo NUEVO en la
poblacion (una conexion o un nodo que no existia), se le asigna
un numero unico incremental: el Innovation Number.

Este numero sirve como "DNI" de esa pieza de estructura.
Permite rastrear la HISTORIA de como aparecio cada gen.

POR QUE SON NECESARIOS?
------------------------
El paper de NEAT los necesita para resolver el problema del cruce:
Cuando dos genomas tienen topologias diferentes, como alineas
sus genes para el cruce?

Sin innovation numbers:
  Genoma A: [conn1, conn2, conn3]
  Genoma B: [conn1, conn2, conn3, conn4]
  No sabes si conn2 de A es "la misma" que conn2 de B.

Con innovation numbers:
  Genoma A: [innov=1, innov=3, innov=5]
  Genoma B: [innov=1, innov=2, innov=5, innov=7]
  Ahora sabes: innov=1 y innov=5 son matching (homologos).
  innov=3 es disjoint de A, innov=2 es disjoint de B, innov=7 es excess de B.

REGLA CLAVE: HISTORIAL PERSISTENTE
----------------------------------
La misma estructura recibe siempre el mismo innovation number durante
toda la corrida, aunque reaparezca muchas generaciones despues.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class InnovationRecord:
    """
    Registro de una innovacion (mutacion estructural).

    Guarda los detalles de la mutacion para poder identificar si
    la misma mutacion ocurre de nuevo en la misma generacion.

    Attributes
    ----------
    innovation_number : int
        ID unico global de esta innovacion
    mutation_type : str
        "connection" o "node"
    in_node : int
        ID del nodo de entrada de la conexion
    out_node : int
        ID del nodo de salida de la conexion
    generation : int
        Generacion en la que ocurrio
    """
    innovation_number: int
    mutation_type: str
    in_node: int
    out_node: int
    generation: int


class InnovationTracker:
    """
    Contador global con historial persistente de innovation numbers.

    Esta clase es el "registro civil" de todas las mutaciones estructurales.
    Cada conexion o nodo nuevo recibe un numero unico, y si la misma
    mutacion reaparece en cualquier generacion, recibe el mismo numero.

    Uso:
    ----
    >>> tracker = InnovationTracker()
    >>> # Nueva conexion de nodo 0 (RSI) a nodo 18 (Output_Dir)
    >>> innov = tracker.get_connection_innovation(0, 18, generation=0)
    >>> print(innov)  # 1
    >>> # Otro individuo hace la misma mutacion en la misma generacion
    >>> innov2 = tracker.get_connection_innovation(0, 18, generation=0)
    >>> print(innov2)  # 1 (mismo numero!)
    >>> # Conexion diferente
    >>> innov3 = tracker.get_connection_innovation(1, 18, generation=0)
    >>> print(innov3)  # 2 (nuevo numero)
    """

    def __init__(self):
        # Contador global. Cada nueva innovacion incrementa este valor.
        self._global_counter: int = 0

        # Contador global para IDs de nodos.
        # Los nodos de input y output tienen IDs fijos (0..N_inputs+N_outputs-1).
        # Los nodos ocultos obtienen IDs incrementales a partir de ahi.
        self._node_counter: int = 0

        # Los marcadores historicos deben persistir durante toda la corrida.
        # La misma conexion o division siempre reutiliza la misma innovacion.
        self._connection_innovations: dict[tuple[int, int], int] = {}
        self._innovation_endpoints: dict[int, tuple[int, int]] = {}
        self._split_innovations: dict[int, tuple[int, int, int]] = {}

        # Generacion actual del cache
        self._current_generation: int = -1

        # Historial completo de todas las innovaciones (para debug/analisis)
        self._history: list[InnovationRecord] = []

    def set_node_counter(self, start_id: int):
        """
        Establece el punto de inicio para IDs de nodos ocultos.

        Debe llamarse despues de definir cuantos nodos de input y output hay.
        Los nodos de input tienen IDs 0..num_inputs-1.
        Los nodos de output tienen IDs num_inputs..num_inputs+num_outputs-1.
        Los nodos ocultos empiezan despues.

        Parameters
        ----------
        start_id : int
            Primer ID disponible para nodos ocultos
            (tipicamente num_inputs + num_outputs)
        """
        self._node_counter = start_id

    def get_connection_innovation(
        self,
        in_node: int,
        out_node: int,
        generation: int
    ) -> int:
        """
        Obtiene el innovation number para una nueva conexion.

        Si la misma conexion (mismo in_node, out_node) ya fue creada
        en esta generacion, retorna el mismo innovation number.
        Si es nueva, asigna uno nuevo.

        Parameters
        ----------
        in_node : int
            ID del nodo de entrada
        out_node : int
            ID del nodo de salida
        generation : int
            Generacion actual

        Returns
        -------
        int
            Innovation number para esta conexion
        """
        # Limpiar cache si cambiamos de generacion
        self._maybe_reset_cache(generation)

        key = (in_node, out_node)
        if key in self._connection_innovations:
            return self._connection_innovations[key]

        # Nueva innovacion
        self._global_counter += 1
        innov = self._global_counter

        # Registrar de forma persistente y en el historial
        self._connection_innovations[key] = innov
        self._innovation_endpoints[innov] = key
        self._history.append(InnovationRecord(
            innovation_number=innov,
            mutation_type="connection",
            in_node=in_node,
            out_node=out_node,
            generation=generation
        ))

        return innov

    def register_connection_innovation(
        self,
        in_node: int,
        out_node: int,
        innovation: int
    ):
        """Sincroniza conexiones creadas antes de asociar un tracker."""
        endpoint = (in_node, out_node)
        endpoint_innovation = self._connection_innovations.get(endpoint)
        innovation_endpoint = self._innovation_endpoints.get(innovation)

        if (
            endpoint_innovation is not None
            and endpoint_innovation != innovation
        ):
            raise RuntimeError(
                f"Endpoint {in_node}->{out_node} ya usa "
                f"innovation {endpoint_innovation}"
            )
        if (
            innovation_endpoint is not None
            and innovation_endpoint != endpoint
        ):
            raise RuntimeError(
                f"Innovation {innovation} ya representa "
                f"{innovation_endpoint[0]}->{innovation_endpoint[1]}"
            )

        self._connection_innovations[endpoint] = innovation
        self._innovation_endpoints[innovation] = endpoint
        self._global_counter = max(self._global_counter, innovation)

    def get_node_id(self) -> int:
        """
        Obtiene un nuevo ID unico para un nodo oculto.

        Cada vez que una mutacion "agregar nodo" crea un nodo nuevo,
        necesita un ID unico. Este metodo lo proporciona.

        Returns
        -------
        int
            Nuevo ID de nodo (nunca se repite)
        """
        self._node_counter += 1
        return self._node_counter - 1

    def get_node_innovation(
        self,
        connection_innov: int,
        generation: int,
        in_node: Optional[int] = None,
        out_node: Optional[int] = None
    ) -> tuple[int, int, int]:
        """
        Obtiene innovation numbers para una mutacion "agregar nodo".

        Cuando se agrega un nodo en una conexion existente (A -> B),
        se divide en dos conexiones: A -> NEW y NEW -> B.
        Esto genera:
          - 1 nuevo nodo (con ID unico)
          - 2 nuevas conexiones (cada una con su innovation number)

        Si la misma division ocurre en la misma generacion en otro
        individuo, se reusan los mismos numeros.

        Parameters
        ----------
        connection_innov : int
            Innovation number de la conexion que se esta dividiendo
        generation : int
            Generacion actual
        in_node : int, opcional
            ID del nodo de entrada de la conexion que se divide
        out_node : int, opcional
            ID del nodo de salida de la conexion que se divide

        Returns
        -------
        tuple[int, int, int]
            (new_node_id, innov_in_to_new, innov_new_to_out)
        """
        self._maybe_reset_cache(generation)

        if connection_innov in self._split_innovations:
            return self._split_innovations[connection_innov]

        # Si no se proveyeron in_node/out_node, buscarlos en el mapeo de endpoints
        if in_node is None or out_node is None:
            endpoints = self._innovation_endpoints.get(connection_innov)
            if endpoints is not None:
                in_node, out_node = endpoints

        # Nuevo nodo
        new_node_id = self.get_node_id()

        # Dos nuevas conexiones
        self._global_counter += 1
        innov_in_to_new = self._global_counter
        self._global_counter += 1
        innov_new_to_out = self._global_counter

        # Registrar de forma persistente en los mapeos de conexiones del tracker
        if in_node is not None and out_node is not None:
            self._connection_innovations[(in_node, new_node_id)] = innov_in_to_new
            self._innovation_endpoints[innov_in_to_new] = (in_node, new_node_id)
            self._connection_innovations[(new_node_id, out_node)] = innov_new_to_out
            self._innovation_endpoints[innov_new_to_out] = (new_node_id, out_node)

        # Guardar permanentemente: una misma division es homologa.
        result = (new_node_id, innov_in_to_new, innov_new_to_out)
        self._split_innovations[connection_innov] = result

        # Registrar en historial
        self._history.append(InnovationRecord(
            innovation_number=innov_in_to_new,
            mutation_type="node_split_in",
            in_node=in_node if in_node is not None else -1,
            out_node=new_node_id,
            generation=generation
        ))
        self._history.append(InnovationRecord(
            innovation_number=innov_new_to_out,
            mutation_type="node_split_out",
            in_node=new_node_id,
            out_node=out_node if out_node is not None else -1,
            generation=generation
        ))

        return result

    def _maybe_reset_cache(self, generation: int):
        """Actualiza la generacion observada para estadisticas."""
        if generation != self._current_generation:
            self._current_generation = generation

    @property
    def current_innovation(self) -> int:
        """Innovation number mas reciente asignado."""
        return self._global_counter

    @property
    def current_node_id(self) -> int:
        """Node ID mas reciente asignado."""
        return self._node_counter

    def get_stats(self) -> dict:
        """Estadisticas del tracker para debug."""
        return {
            "total_innovations": self._global_counter,
            "total_node_ids": self._node_counter,
            "current_generation": self._current_generation,
            "connection_history_size": len(self._connection_innovations),
            "split_history_size": len(self._split_innovations),
            "history_size": len(self._history),
        }
