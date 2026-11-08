"""
Pruebas de regresion del primer refactor (ver CAMBIOS_REFACTOR_NEAT.md, seccion 9).

Cubren:
- Homologia y reserva de innovation numbers.
- Persistencia de divisiones de nodos.
- Reactivacion completa de indicadores.
- Descendencia exacta con fitness negativos.
- Ajuste dinamico del umbral de especies.
- Ejecucion de señales en la apertura siguiente.
- Gaps a traves de stops.
- Cortacircuito de drawdown.
- Anualizacion con calendario crypto.
- Conservacion de fines de semana en BTC-MXN.
"""

import random

import numpy as np
import pandas as pd
import pytest

from src.data.downloader import DataDownloader
from src.neat.genome import Genome
from src.neat.innovation import InnovationTracker
from src.neat.population import Population
from src.neat.reproduction import mutate_indicator_mask
from src.neat.species import Species, SpeciationEngine
from src.trading.backtester import Backtester
from src.trading.metrics import compute_metrics
from src.trading.position import PositionType
from tests.helpers import make_genome, make_prices, patch_network, signal_outputs


def _backtester(**kwargs) -> Backtester:
    params = dict(
        initial_capital=100000.0, commission=0.0, slippage=0.0,
        confidence_threshold=0.3, timeframe="1d", min_trades=1,
        allow_short=False,
    )
    params.update(kwargs)
    return Backtester(**params)


# =================================================================
# NEAT
# =================================================================

def test_innovation_numbers_are_homologous_and_reserved():
    random.seed(1)
    num_inputs, num_outputs = 6, 3
    pop = Population(pop_size=20, num_inputs=num_inputs, num_outputs=num_outputs)
    pop.initialize([f"i{k}" for k in range(num_inputs)], initial_active=3)

    endpoints_by_innovation = {}
    for genome in pop.genomes:
        for innovation, conn in genome.connections.items():
            key = (conn.in_node, conn.out_node)
            assert endpoints_by_innovation.setdefault(innovation, key) == key

    # Todas las conexiones input->output quedan reservadas al inicio.
    assert pop.tracker.current_innovation == num_inputs * num_outputs

    # Una innovacion nueva nunca reutiliza un numero reservado.
    _, innov_in, innov_out = pop.tracker.get_node_innovation(1, generation=1)
    assert innov_in > num_inputs * num_outputs
    assert innov_out > innov_in


def test_node_split_is_persistent_across_generations():
    random.seed(2)
    pop = Population(pop_size=5, num_inputs=4, num_outputs=3)
    pop.initialize([f"i{k}" for k in range(4)], initial_active=2)

    first = pop.tracker.get_node_innovation(3, generation=1)
    later = pop.tracker.get_node_innovation(3, generation=25)
    assert first == later


def test_indicator_reactivation_restores_connections(monkeypatch):
    random.seed(3)
    pop = Population(pop_size=3, num_inputs=5, num_outputs=3)
    names = [f"i{k}" for k in range(5)]
    pop.initialize(names, initial_active=3)
    genome = pop.genomes[0]

    target = genome.active_indicator_names[0]
    input_id = names.index(target)
    monkeypatch.setattr(random, "choice", lambda seq: target)

    # Desactivar: todas sus conexiones quedan apagadas por el indicador.
    mutate_indicator_mask(genome, pop.tracker, generation=1)
    assert genome.indicator_mask[target] is False
    from_input = [c for c in genome.connections.values() if c.in_node == input_id]
    assert from_input and all(not c.enabled for c in from_input)
    assert all(c.disabled_by_indicator for c in from_input)

    # Simular la perdida de una conexion hacia una salida.
    lost = next(c.innovation for c in from_input
                if c.out_node == genome.output_node_ids[0])
    del genome.connections[lost]

    # Reactivar: se restauran y se garantiza acceso a las 3 salidas.
    mutate_indicator_mask(genome, pop.tracker, generation=2)
    assert genome.indicator_mask[target] is True
    from_input = [c for c in genome.connections.values() if c.in_node == input_id]
    assert all(c.enabled for c in from_input)
    assert {c.out_node for c in from_input} == set(genome.output_node_ids)


def test_offspring_counts_are_exact_with_negative_fitness():
    engine = SpeciationEngine()
    fitness_by_species = {0: [-3.0, -2.5], 1: [-1.0, -0.5, -0.7], 2: [-5.0]}
    gid = 0
    for sid, fitnesses in fitness_by_species.items():
        members = []
        for fit in fitnesses:
            g = Genome(genome_id=gid, num_inputs=2, num_outputs=3)
            g.fitness = fit
            members.append(g)
            gid += 1
        sp = Species(sid, members[0])
        for g in members:
            sp.add_member(g)
        sp.compute_adjusted_fitness()
        engine.species[sid] = sp

    for total in (7, 17, 50):
        counts = engine.get_offspring_counts(total)
        assert sum(counts.values()) == total
        assert all(c >= 0 for c in counts.values())
        # La especie con el mejor individuo siempre recibe descendencia.
        assert counts[1] > 0


def test_compatibility_threshold_adapts():
    engine = SpeciationEngine(
        compatibility_threshold=2.0, target_species_min=5,
        target_species_max=12, threshold_step=0.1,
    )
    dummy = Genome(genome_id=0, num_inputs=2, num_outputs=3)

    engine.species = {0: Species(0, dummy)}
    engine._adapt_threshold()
    assert engine.threshold == pytest.approx(1.9)

    engine.species = {i: Species(i, dummy) for i in range(20)}
    engine._adapt_threshold()
    assert engine.threshold == pytest.approx(2.0)


# =================================================================
# BACKTESTER
# =================================================================

def test_signal_executes_at_next_open(monkeypatch):
    n = 40
    opens = np.full(n, 100.0)
    closes = np.full(n, 100.0)
    opens[6] = 110.0
    closes[6] = 110.0
    df = make_prices(opens, closes)
    # Barra 5: compra. Barra 6: orden activa de venta (ejecuta en apertura de la 7).
    patch_network(monkeypatch, signal_outputs(n, long_bars=[5], short_bars=[6]))

    _, _, _, trades = _backtester().evaluate_genome(
        make_genome(), df, ["ind0", "ind1"]
    )

    assert len(trades) == 1
    trade = trades[0]
    # La señal del cierre de la barra 5 se ejecuta en la APERTURA de la 6.
    assert trade.entry_time == df.index[6]
    assert trade.entry_price == pytest.approx(110.0)
    # La orden de venta del cierre de la barra 6 se sale en la apertura de la 7.
    assert trade.exit_time == df.index[7]
    assert trade.exit_reason == "signal"


def test_gap_through_stop_exits_at_open(monkeypatch):
    n = 40
    opens = np.full(n, 100.0)
    opens[7:] = 80.0
    df = make_prices(opens)
    patch_network(monkeypatch, signal_outputs(n, long_bars=range(5, n)))

    genome = make_genome(stop_loss=0.05, trailing_stop=0.5)
    _, _, _, trades = _backtester().evaluate_genome(genome, df, ["ind0", "ind1"])

    first = trades[0]
    assert first.entry_time == df.index[6]
    assert first.exit_reason == "stop_loss"
    assert first.exit_time == df.index[7]
    # Se ejecuta al precio del gap (80), no al nivel teorico del stop (95).
    assert first.exit_price == pytest.approx(80.0)


def test_drawdown_circuit_breaker_halts_trading(monkeypatch):
    n = 40
    opens = np.empty(n)
    closes = np.empty(n)
    price = 100.0
    for i in range(n):
        opens[i] = price
        price *= 0.98
        closes[i] = price
    df = make_prices(opens, closes)
    patch_network(monkeypatch, signal_outputs(n, all_long=True))

    genome = make_genome(max_drawdown=0.10, stop_loss=0.5, trailing_stop=0.5)
    _, _, equity, trades = _backtester().evaluate_genome(
        genome, df, ["ind0", "ind1"]
    )

    # Una sola operacion: se liquida y ya no se vuelve a entrar.
    assert len(trades) == 1
    exit_pos = df.index.get_loc(trades[0].exit_time)
    after = equity.iloc[exit_pos:]
    assert np.allclose(after.values, after.values[0])
    peak = equity.cummax()
    assert ((peak - equity) / peak).max() >= 0.10


# =================================================================
# METRICAS Y DATOS
# =================================================================

def test_annualization_uses_crypto_calendar():
    days = np.arange(1, 366)
    equity = pd.Series(
        100000.0 * 2.0 ** (days / 365.0),
        index=pd.date_range("2024-01-01", periods=365, freq="D"),
    )
    metrics = compute_metrics(equity, [], initial_capital=100000.0, timeframe="1d")
    # 365 barras diarias = 1 año crypto: duplicar el capital = +100% anual.
    assert metrics.annualized_return == pytest.approx(1.0, rel=1e-6)


def test_btc_mxn_keeps_weekends(monkeypatch, tmp_path):
    usd_index = pd.date_range("2024-01-01", "2024-01-14", freq="D")  # lunes a domingo
    df_usd = pd.DataFrame({
        "Open": 40000.0, "High": 41000.0, "Low": 39000.0,
        "Close": np.linspace(40000, 41300, len(usd_index)), "Volume": 1.0,
    }, index=usd_index)
    fx_index = usd_index[usd_index.dayofweek < 5]
    df_fx = pd.DataFrame({
        "Open": 17.0, "High": 17.0, "Low": 17.0,
        "Close": np.linspace(17.0, 17.9, len(fx_index)), "Volume": 0.0,
    }, index=fx_index)

    def fake_download(self, symbol, start="2018-01-01", end=None, interval="1d",
                      use_cache=True, cache_max_age_hours=24):
        return df_usd.copy() if symbol == "BTC-USD" else df_fx.copy()

    monkeypatch.setattr(DataDownloader, "download", fake_download)
    downloader = DataDownloader(cache_dir=tmp_path)
    out = downloader.download_mxn("BTC", start="2024-01-01", end="2024-01-14",
                                  use_cache=False)

    assert len(out) == len(usd_index)  # fines de semana conservados
    saturday = pd.Timestamp("2024-01-06")
    friday = pd.Timestamp("2024-01-05")
    expected = df_usd.loc[saturday, "Close"] * df_fx.loc[friday, "Close"]
    # El sabado usa el ultimo tipo de cambio conocido (viernes), nunca uno futuro.
    assert out.loc[saturday, "Close"] == pytest.approx(expected)


def test_node_split_connection_innovations_consistency():
    """Verifica que al dividir una conexion, los nuevos endpoints queden registrados en el tracker."""
    tracker = InnovationTracker()
    tracker.set_node_counter(23)  # 20 inputs (0..19), 3 outputs (20..22)
    # Registrar conexion 14 -> 20
    conn_innov = tracker.get_connection_innovation(14, 20, generation=0)

    # Dividir conexion 14 -> 20 creando nodo 23
    new_node_id, innov_in, innov_out = tracker.get_node_innovation(
        conn_innov, generation=1, in_node=14, out_node=20
    )
    assert new_node_id == 23

    # get_connection_innovation para 14 -> 23 debe retornar innov_in (no un numero nuevo)
    assert tracker.get_connection_innovation(14, 23, generation=2) == innov_in
    assert tracker.get_connection_innovation(23, 20, generation=2) == innov_out

    # register_connection_innovation no debe lanzar error
    tracker.register_connection_innovation(14, 23, innov_in)
    tracker.register_connection_innovation(23, 20, innov_out)
