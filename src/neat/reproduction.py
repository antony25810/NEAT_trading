"""
============================================================
reproduction.py - Mutaciones y Cruce NEAT
============================================================

Este modulo implementa TODOS los operadores geneticos:

MUTACIONES (modifican UN genoma):
  - Mutacion de pesos (perturbacion gaussiana o reset)
  - Agregar nodo (divide una conexion existente)
  - Eliminar nodo (remueve un nodo oculto)
  - Agregar conexion (entre nodos no conectados)
  - Eliminar conexion (desactiva una conexion)
  - Cambiar activacion (de un nodo oculto)
  - Mutar parametros de trading (SL, TP, trailing, position size)
  - Mutar mascara de indicadores (activar/desactivar un indicador)

CRUCE (combina DOS genomas):
  - Alineacion por innovation number
  - Herencia de genes matching del padre mas fit
  - Herencia de genes disjoint/excess del padre mas fit
  - Blend de parametros de trading
"""

import random
from typing import Optional

from src.neat.genome import (
    Genome, ConnectionGene, NodeGene, TradingParams
)
from src.neat.innovation import InnovationTracker


# =================================================================
# MUTACIONES
# =================================================================

def mutate(
    genome: Genome,
    tracker: InnovationTracker,
    generation: int,
    config: dict
) -> Genome:
    """
    Aplica mutaciones a un genoma segun las probabilidades del config.

    Cada tipo de mutacion se aplica INDEPENDIENTEMENTE con su propia
    probabilidad. Un genoma puede recibir multiples mutaciones en
    una sola llamada (ej: mutar pesos Y agregar una conexion).

    Parameters
    ----------
    genome : Genome
        Genoma a mutar (se modifica IN-PLACE y se retorna)
    tracker : InnovationTracker
        Tracker global de innovation numbers
    generation : int
        Generacion actual (para el cache del tracker)
    config : dict
        Probabilidades de mutacion (de neat_config.yaml)

    Returns
    -------
    Genome
        El mismo genoma, mutado
    """
    # --- Mutaciones de pesos (las mas comunes) ---
    if random.random() < config.get("weight_perturbation", 0.80):
        mutate_weights(genome, config)

    if random.random() < config.get("bias_perturbation", 0.30):
        mutate_biases(genome, config)

    # --- Mutaciones estructurales (raras pero importantes) ---
    if random.random() < config.get("add_connection", 0.05):
        mutate_add_connection(genome, tracker, generation)

    if random.random() < config.get("add_node", 0.03):
        mutate_add_node(genome, tracker, generation)

    if random.random() < config.get("remove_node", 0.02):
        mutate_remove_node(genome)

    if random.random() < config.get("remove_connection", 0.04):
        mutate_remove_connection(genome)

    if random.random() < config.get("change_activation", 0.03):
        mutate_change_activation(genome, config)

    # --- Mutaciones de parametros de trading ---
    # Perturbaciones gaussianas PEQUENAS relativas al rango de cada parametro.
    risk = config.get("risk_ranges", {})
    rel_std = config.get("trading_param_perturbation_std", 0.05)

    if random.random() < config.get("mutate_stop_loss", 0.10):
        mutate_trading_param(
            genome, "stop_loss", *_range(risk, "stop_loss", 0.01, 0.15),
            relative_std=rel_std
        )

    if random.random() < config.get("mutate_take_profit", 0.10):
        mutate_trading_param(
            genome, "take_profit", *_range(risk, "take_profit", 0.02, 0.30),
            relative_std=rel_std
        )

    if random.random() < config.get("mutate_trailing_stop", 0.08):
        mutate_trading_param(
            genome, "trailing_stop",
            *_range(risk, "trailing_stop", 0.005, 0.10),
            relative_std=rel_std
        )

    if random.random() < config.get("mutate_position_size", 0.10):
        mutate_trading_param(
            genome, "position_size", *_range(risk, "position_size", 0.40, 1.0),
            relative_std=rel_std
        )

    if random.random() < config.get("mutate_max_drawdown", 0.08):
        mutate_trading_param(
            genome, "max_drawdown",
            *_range(risk, "max_drawdown", 0.20, 0.45),
            relative_std=rel_std
        )

    # --- Mutacion de mascara de indicadores ---
    if random.random() < config.get("mutate_indicator_mask", 0.05):
        mutate_indicator_mask(genome, tracker, generation)

    return genome


# ----- Mutacion de pesos -----

def mutate_weights(genome: Genome, config: dict):
    """
    Perturba o resetea los pesos de las conexiones.

    Para cada conexion:
    - Con probabilidad weight_reset: peso = random nuevo
    - Sino: peso += gaussiana(0, std)

    Esto es la mutacion MAS COMUN (80% de probabilidad).
    Es la que realmente "optimiza" la estrategia sin cambiar
    la estructura de la red.
    """
    reset_prob = config.get("weight_reset", 0.10)
    std = config.get("weight_perturbation_std", 0.5)

    for conn in genome.connections.values():
        if random.random() < reset_prob:
            # Reset completo: nuevo peso aleatorio
            conn.weight = random.gauss(0, 1.0)
        else:
            # Perturbacion: ajuste pequeno
            conn.weight += random.gauss(0, std)

        # Clip para evitar pesos extremos
        conn.weight = max(-8.0, min(8.0, conn.weight))


def mutate_biases(genome: Genome, config: dict):
    """Muta biases de nodos ocultos y de salida."""
    reset_prob = config.get("bias_reset", 0.05)
    std = config.get("bias_perturbation_std", 0.2)

    for node in genome.nodes.values():
        if node.node_type == "input":
            continue
        if random.random() < reset_prob:
            node.bias = random.gauss(0, 1.0)
        else:
            node.bias += random.gauss(0, std)
        node.bias = max(-5.0, min(5.0, node.bias))


# ----- Mutaciones estructurales -----

def mutate_add_connection(
    genome: Genome,
    tracker: InnovationTracker,
    generation: int,
    max_attempts: int = 20
):
    """
    Agrega una nueva conexion entre dos nodos no conectados.

    COMO FUNCIONA:
    1. Elegir dos nodos al azar (in_node, out_node)
    2. Verificar que NO estan ya conectados
    3. Verificar que no crearia un ciclo (feedforward only)
    4. Crear la conexion con peso aleatorio y nuevo innovation number

    Parameters
    ----------
    max_attempts : int
        Intentos maximos para encontrar un par valido.
        En redes pequenas hay pocos pares posibles.
    """
    _synchronize_tracker(genome, tracker)
    all_node_ids = list(genome.nodes.keys())
    input_ids = set(genome.input_node_ids)
    output_ids = set(genome.output_node_ids)

    # Pares existentes (para no duplicar)
    existing_pairs = {
        (c.in_node, c.out_node): c for c in genome.connections.values()
    }

    for _ in range(max_attempts):
        # Elegir in_node: puede ser input o hidden
        # Elegir out_node: puede ser hidden o output
        in_node = random.choice(all_node_ids)
        out_node = random.choice(all_node_ids)

        # Validaciones
        if in_node == out_node:
            continue  # no auto-conexiones
        if out_node in input_ids:
            continue  # no conexiones HACIA inputs
        if in_node in output_ids:
            continue  # no conexiones DESDE outputs
        existing = existing_pairs.get((in_node, out_node))
        if existing is not None:
            if not existing.enabled:
                existing.enabled = True
                existing.weight = random.gauss(0, 1.0)
                return
            continue

        # Verificar que no crea ciclo (simple check)
        if _would_create_cycle(genome, in_node, out_node):
            continue

        # Crear la nueva conexion
        innov = tracker.get_connection_innovation(in_node, out_node, generation)
        existing_innov = genome.connections.get(innov)
        if existing_innov is not None:
            if (existing_innov.in_node, existing_innov.out_node) != (
                in_node, out_node
            ):
                raise RuntimeError(
                    f"Colision de innovation {innov}: "
                    f"{existing_innov.in_node}->{existing_innov.out_node} "
                    f"vs {in_node}->{out_node}"
                )
            existing_innov.enabled = True
            return

        genome.connections[innov] = ConnectionGene(
            in_node=in_node,
            out_node=out_node,
            weight=random.gauss(0, 1.0),
            enabled=True,
            innovation=innov
        )
        return  # exito

    # Si no se encontro par valido despues de max_attempts, no pasa nada


def mutate_add_node(
    genome: Genome,
    tracker: InnovationTracker,
    generation: int,
    activation_options: Optional[list[str]] = None
):
    """
    Agrega un nodo oculto dividiendo una conexion existente.

    COMO FUNCIONA (del paper original):
    Conexion existente: A ---(w=0.7)---> B  (se DESACTIVA)
    Se convierte en:    A ---(w=1.0)---> NEW ---(w=0.7)---> B

    El peso de A->NEW es 1.0 (para preservar la senal inicial).
    El peso de NEW->B es el peso original (0.7).
    Esto minimiza el impacto inmediato de agregar el nodo:
    la red produce (casi) el mismo output que antes.

    POR QUE ESTO ES IMPORTANTE?
    Si la red YA funciona bien, agregar un nodo no deberia
    destruir esa funcionalidad. Con w=1.0 y w_original, la senal
    pasa casi igual (salvo la activacion del nuevo nodo).
    """
    _synchronize_tracker(genome, tracker)
    if activation_options is None:
        activation_options = ["sigmoid", "tanh", "relu", "leaky_relu"]

    # Elegir una conexion activa al azar
    active_conns = [c for c in genome.connections.values() if c.enabled]
    if not active_conns:
        return  # nada que dividir

    conn = random.choice(active_conns)

    # Desactivar la conexion original
    conn.enabled = False

    # Obtener IDs del tracker (cache por generacion)
    new_node_id, innov_in, innov_out = tracker.get_node_innovation(
        conn.innovation, generation, conn.in_node, conn.out_node
    )

    if new_node_id not in genome.nodes:
        genome.nodes[new_node_id] = NodeGene(
            node_id=new_node_id,
            node_type="hidden",
            activation=random.choice(activation_options)
        )

    # Crear las dos nuevas conexiones
    # in_node -> new_node (peso = 1.0 para preservar senal)
    if innov_in in genome.connections:
        genome.connections[innov_in].enabled = True
    else:
        genome.connections[innov_in] = ConnectionGene(
            in_node=conn.in_node,
            out_node=new_node_id,
            weight=1.0,
            enabled=True,
            innovation=innov_in
        )

    # new_node -> out_node (peso = peso original)
    if innov_out in genome.connections:
        genome.connections[innov_out].enabled = True
    else:
        genome.connections[innov_out] = ConnectionGene(
            in_node=new_node_id,
            out_node=conn.out_node,
            weight=conn.weight,
            enabled=True,
            innovation=innov_out
        )


def mutate_remove_node(genome: Genome):
    """
    Elimina un nodo oculto al azar y todas sus conexiones.

    NOTA: Solo elimina nodos HIDDEN. Los inputs y outputs son fijos.
    Esta mutacion no existe en el paper original de NEAT, pero es
    util para reducir la complejidad cuando un nodo no aporta.
    """
    hidden_ids = genome.hidden_node_ids
    if not hidden_ids:
        return  # nada que eliminar

    # Elegir un nodo oculto al azar
    node_to_remove = random.choice(hidden_ids)

    # Eliminar todas las conexiones que tocan este nodo
    to_remove = [
        innov for innov, conn in genome.connections.items()
        if conn.in_node == node_to_remove or conn.out_node == node_to_remove
    ]
    for innov in to_remove:
        del genome.connections[innov]

    # Eliminar el nodo
    del genome.nodes[node_to_remove]


def mutate_remove_connection(genome: Genome):
    """
    Desactiva una conexion activa al azar.

    No la elimina del genoma (podria reactivarse en un cruce futuro),
    solo la marca como enabled=False.
    """
    active_conns = [c for c in genome.connections.values() if c.enabled]
    if not active_conns:
        return

    conn = random.choice(active_conns)
    conn.enabled = False


def mutate_change_activation(genome: Genome, config: dict):
    """
    Cambia la funcion de activacion de un nodo oculto al azar.

    Solo afecta nodos HIDDEN. Los outputs tienen activaciones fijas
    (tanh para direccion, sigmoid para confianza/tamano).
    """
    hidden_ids = genome.hidden_node_ids
    if not hidden_ids:
        return

    node_id = random.choice(hidden_ids)
    options = config.get("activation_functions",
                         ["sigmoid", "tanh", "relu", "leaky_relu", "identity"])

    # Elegir una activacion DIFERENTE a la actual
    current = genome.nodes[node_id].activation
    other_options = [a for a in options if a != current]
    if other_options:
        genome.nodes[node_id].activation = random.choice(other_options)


# ----- Mutaciones de parametros de trading -----

def mutate_trading_param(
    genome: Genome,
    param_name: str,
    min_val: float,
    max_val: float,
    relative_std: float = 0.05
):
    """
    Perturba un parametro de trading con ruido gaussiano PEQUENO.

    COMO FUNCIONA:
    - sigma = relative_std * (max_val - min_val)
      Con relative_std=0.05 cada parametro se mueve ~5% de SU rango,
      sin importar si el rango es [0.005, 0.10] o [0.02, 0.30].
    - Si el valor sale del rango se REFLEJA hacia adentro en lugar de
      recortarse. El recorte acumulaba individuos exactamente en los
      bordes (valores extremos); la reflexion los devuelve al interior.

    POR QUE PEQUENO?
    Un salto grande (o un reset aleatorio) puede convertir una estrategia
    buena en una inviable de golpe (ej. stop loss de 1% -> 14%) y el
    individuo se pierde. Con pasos pequenos la evolucion ajusta finamente
    los parametros que ya funcionan.
    """
    span = max_val - min_val
    if span <= 0:
        setattr(genome.trading_params, param_name, min_val)
        return

    current = getattr(genome.trading_params, param_name)
    sigma = relative_std * span
    new_val = _reflect_into_range(
        current + random.gauss(0, sigma), min_val, max_val
    )
    setattr(genome.trading_params, param_name, new_val)


def _reflect_into_range(value: float, min_val: float, max_val: float) -> float:
    """
    Refleja un valor dentro de [min_val, max_val] como un espejo en los bordes.

    Ejemplos con rango [0, 1]:  1.1 -> 0.9,  -0.2 -> 0.2,  0.5 -> 0.5
    """
    span = max_val - min_val
    if span <= 0:
        return min_val
    offset = (value - min_val) % (2.0 * span)
    reflected = min_val + (span - abs(offset - span))
    return max(min_val, min(max_val, reflected))


# ----- Mutacion de mascara de indicadores -----

def mutate_indicator_mask(
    genome: Genome,
    tracker: InnovationTracker,
    generation: int,
    min_active: int = 2
):
    """
    Activa o desactiva un indicador al azar.

    REGLAS:
    - No puede desactivar si quedan <= min_active indicadores activos
    - Al ACTIVAR un indicador, agrega conexiones nuevas a los outputs
    - Al DESACTIVAR, desactiva las conexiones de ese input

    POR QUE ESTO ES IMPORTANTE?
    La evolucion puede DESCUBRIR que indicadores son utiles.
    Si RSI no aporta, eventualmente se desactivara.
    Si ATR resulta util, se activara y se conectara a la red.
    """
    _synchronize_tracker(genome, tracker)
    if not genome.indicator_mask:
        return

    names = list(genome.indicator_mask.keys())
    name = random.choice(names)
    currently_active = genome.indicator_mask[name]

    if currently_active:
        # Desactivar: verificar que no baje del minimo
        num_active = sum(1 for v in genome.indicator_mask.values() if v)
        if num_active <= min_active:
            return  # no desactivar

        genome.indicator_mask[name] = False
        input_id = names.index(name)

        # Desactivar conexiones de este input
        for conn in genome.connections.values():
            if conn.in_node == input_id and conn.enabled:
                conn.enabled = False
                conn.disabled_by_indicator = True
    else:
        # Activar: crear conexiones a los outputs
        genome.indicator_mask[name] = True
        input_id = names.index(name)

        # Restaurar caminos previos y garantizar acceso a todas las salidas.
        for conn in genome.connections.values():
            if conn.in_node == input_id and conn.disabled_by_indicator:
                conn.enabled = True
                conn.disabled_by_indicator = False

        for output_id in genome.output_node_ids:
            innov = tracker.get_connection_innovation(
                input_id, output_id, generation
            )
            existing = genome.connections.get(innov)
            if existing is not None:
                if (existing.in_node, existing.out_node) != (input_id, output_id):
                    raise RuntimeError(f"Colision de innovation {innov}")
                if existing.disabled_by_indicator:
                    existing.enabled = True
                    existing.disabled_by_indicator = False
            else:
                genome.connections[innov] = ConnectionGene(
                    in_node=input_id,
                    out_node=output_id,
                    weight=random.gauss(0, 0.5),
                    enabled=True,
                    innovation=innov
                )


# =================================================================
# CRUCE
# =================================================================

def crossover(
    parent_a: Genome,
    parent_b: Genome,
    child_id: int,
    risk_ranges: Optional[dict] = None
) -> Genome:
    """
    Cruce de dos genomas usando alineacion por innovation number.

    ALGORITMO DEL PAPER ORIGINAL:
    1. Alinear genes por innovation number
    2. Genes MATCHING (misma innovacion en ambos padres):
       - Heredar del padre con MEJOR fitness (o aleatorio si iguales)
    3. Genes DISJOINT y EXCESS (solo en un padre):
       - Heredar del padre con MEJOR fitness
    4. Parametros de trading: blend (promedio) de ambos padres
    5. Mascara de indicadores: herencia del padre mas fit

    POR QUE EL PADRE MAS FIT DOMINA?
    Porque su genoma ha "demostrado" que funciona mejor.
    Los genes exclusivos del padre menos fit podrian ser
    innovaciones utiles O ruido inutil — no lo sabemos.
    Es mas seguro quedarnos con lo que ya funciona.

    Parameters
    ----------
    parent_a : Genome
        Primer padre
    parent_b : Genome
        Segundo padre
    child_id : int
        ID para el genoma hijo

    Returns
    -------
    Genome
        Genoma hijo resultante del cruce
    """
    # Determinar quien es el padre mas fit
    fit_a = parent_a.fitness if parent_a.fitness is not None else 0
    fit_b = parent_b.fitness if parent_b.fitness is not None else 0

    if fit_a > fit_b:
        better, worse = parent_a, parent_b
    elif fit_b > fit_a:
        better, worse = parent_b, parent_a
    else:
        # Fitness igual: el mas pequeno es "mejor" (Navaja de Occam)
        if len(parent_a.connections) <= len(parent_b.connections):
            better, worse = parent_a, parent_b
        else:
            better, worse = parent_b, parent_a

    # Crear genoma hijo
    child = Genome(
        genome_id=child_id,
        num_inputs=better.num_inputs,
        num_outputs=better.num_outputs
    )

    # --- Heredar NODOS ---
    # Todos los nodos del padre mas fit (y cualquier adicional del otro)
    for node_id, node in better.nodes.items():
        child.nodes[node_id] = node.copy()

    # Agregar nodos ocultos del padre peor que no existen en el mejor
    # (necesarios si heredamos conexiones disjoint/excess del peor)
    for node_id, node in worse.nodes.items():
        if node_id not in child.nodes and node.node_type == "hidden":
            child.nodes[node_id] = node.copy()

    # --- Heredar CONEXIONES ---
    innovs_better = set(better.connections.keys())
    innovs_worse = set(worse.connections.keys())
    matching = innovs_better & innovs_worse

    # Genes matching: elegir del mejor (o aleatorio si fitness igual)
    for innov in matching:
        endpoint_a = (
            parent_a.connections[innov].in_node,
            parent_a.connections[innov].out_node
        )
        endpoint_b = (
            parent_b.connections[innov].in_node,
            parent_b.connections[innov].out_node
        )
        if endpoint_a != endpoint_b:
            raise RuntimeError(
                f"Innovation {innov} no es homologa: "
                f"{endpoint_a} vs {endpoint_b}"
            )
        if fit_a == fit_b:
            # Fitness igual: elegir al azar
            source = random.choice([better, worse])
        else:
            source = better
        child.connections[innov] = source.connections[innov].copy()

        # Probabilidad de re-habilitar genes deshabilitados
        if not child.connections[innov].enabled:
            if random.random() < 0.25:
                child.connections[innov].enabled = True

    # Genes disjoint y excess: heredar del padre mas fit
    for innov in innovs_better - matching:
        child.connections[innov] = better.connections[innov].copy()

    # Si fitness es igual, tambien heredar del peor
    if fit_a == fit_b:
        for innov in innovs_worse - matching:
            if innov not in child.connections:
                candidate = worse.connections[innov].copy()
                if (
                    candidate.enabled
                    and _would_create_cycle(
                        child, candidate.in_node, candidate.out_node
                    )
                ):
                    continue
                child.connections[innov] = candidate

    # --- Parametros de trading: BLEND ---
    # Promedio ponderado hacia el padre mas fit (70/30)
    if fit_a == fit_b:
        alpha = 0.5
    else:
        alpha = 0.7  # 70% del mejor, 30% del peor

    risk = risk_ranges or {}
    child.trading_params = TradingParams(
        stop_loss=_blend(better.trading_params.stop_loss,
                         worse.trading_params.stop_loss, alpha,
                         *_range(risk, "stop_loss", 0.01, 0.15)),
        take_profit=_blend(better.trading_params.take_profit,
                           worse.trading_params.take_profit, alpha,
                           *_range(risk, "take_profit", 0.02, 0.30)),
        trailing_stop=_blend(better.trading_params.trailing_stop,
                             worse.trading_params.trailing_stop, alpha,
                             *_range(risk, "trailing_stop", 0.005, 0.10)),
        max_drawdown=_blend(better.trading_params.max_drawdown,
                            worse.trading_params.max_drawdown, alpha,
                            *_range(risk, "max_drawdown", 0.20, 0.45)),
        position_size=_blend(better.trading_params.position_size,
                             worse.trading_params.position_size, alpha,
                             *_range(risk, "position_size", 0.40, 1.0)),
    )

    # --- Mascara de indicadores: heredar del padre mas fit ---
    child.indicator_mask = dict(better.indicator_mask)

    return child


# =================================================================
# FUNCIONES AUXILIARES
# =================================================================

def _blend(val_a: float, val_b: float, alpha: float,
           min_val: float, max_val: float) -> float:
    """Mezcla ponderada de dos valores con clip."""
    result = alpha * val_a + (1 - alpha) * val_b
    return max(min_val, min(max_val, result))


def _range(
    ranges: dict,
    name: str,
    default_min: float,
    default_max: float
) -> tuple[float, float]:
    """Obtiene un rango configurable, aceptando max_drawdown escalar."""
    cfg = ranges.get(name, {})
    if isinstance(cfg, dict):
        return (
            float(cfg.get("min", default_min)),
            float(cfg.get("max", default_max)),
        )
    return default_min, float(cfg)


def _synchronize_tracker(
    genome: Genome,
    tracker: InnovationTracker
):
    """Registra genes preexistentes antes de crear nuevas innovaciones."""
    for innovation, connection in genome.connections.items():
        tracker.register_connection_innovation(
            connection.in_node,
            connection.out_node,
            innovation
        )


def _would_create_cycle(
    genome: Genome,
    from_node: int,
    to_node: int
) -> bool:
    """
    Verifica si agregar una conexion from_node -> to_node
    crearia un ciclo en la red (haciendola no-feedforward).

    Usa BFS (Breadth-First Search): partiendo de to_node,
    sigue las conexiones hacia adelante. Si llega a from_node,
    hay un ciclo.
    """
    visited = set()
    queue = [to_node]

    while queue:
        current = queue.pop(0)
        if current == from_node:
            return True  # encontramos un ciclo!
        if current in visited:
            continue
        visited.add(current)

        # Seguir conexiones activas que SALEN de current
        for conn in genome.connections.values():
            if conn.enabled and conn.in_node == current:
                queue.append(conn.out_node)

    return False
