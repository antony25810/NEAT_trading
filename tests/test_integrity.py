"""
Pruebas de integridad del sistema:

1. Paridad activate vs activate_batch: la version vectorizada de la red debe
   producir exactamente los mismos outputs que la evaluacion fila por fila.
2. Causalidad de extremo a extremo: alterar el FUTURO (barras 351-400) no debe
   cambiar ni los indicadores, ni las señales, ni el capital hasta la barra 350.
3. Contabilidad intrabarra: un trailing stop pequeño (1%) no debe cerrar la
   posicion antes que el stop loss (5%) cuando la operacion nunca estuvo en
   ganancias.
"""

import random

import numpy as np
import pandas as pd
import pytest

from src.data.pipeline import DataPipeline
from src.neat.genome import Genome, TradingParams
from src.neat.network import FeedForwardNetwork
from src.neat.population import Population
from src.neat.reproduction import mutate_add_connection, mutate_add_node
from src.trading.backtester import Backtester
from src.trading.position import Position, PositionType
from tests.helpers import make_genome, make_prices, patch_network, signal_outputs


# =================================================================
# 1. PARIDAD activate vs activate_batch
# =================================================================

def _complex_genome(seed: int = 11) -> Genome:
    """Genoma con nodos ocultos, conexiones extra, biases y entradas inactivas."""
    random.seed(seed)
    np.random.seed(seed)
    num_inputs = 6
    pop = Population(pop_size=5, num_inputs=num_inputs, num_outputs=3)
    pop.initialize([f"i{k}" for k in range(num_inputs)], initial_active=4)
    genome = pop.genomes[0]

    for gen in range(1, 4):
        mutate_add_node(genome, pop.tracker, generation=gen)
        mutate_add_connection(genome, pop.tracker, generation=gen)

    for node in genome.nodes.values():
        if node.node_type != "input":
            node.bias = random.gauss(0.0, 0.5)
    return genome


def test_activate_batch_matches_activate_row_by_row():
    genome = _complex_genome()
    assert genome.hidden_node_ids, "La prueba requiere al menos un nodo oculto"
    # Debe haber entradas inactivas para verificar el mapeo columna -> input_id.
    assert len(genome.active_input_ids) < genome.num_inputs

    net = FeedForwardNetwork.from_genome(genome)
    active_ids = sorted(genome.active_input_ids)
    output_ids = sorted(genome.output_node_ids)

    rng = np.random.default_rng(0)
    batch = rng.uniform(-1.0, 1.0, size=(100, len(active_ids)))

    batch_out = net.activate_batch(batch)
    assert batch_out.shape == (100, 3)

    row_out = np.empty_like(batch_out)
    for r in range(batch.shape[0]):
        inputs = {nid: float(batch[r, c]) for c, nid in enumerate(active_ids)}
        result = net.activate(inputs)
        row_out[r] = [result[oid] for oid in output_ids]

    assert np.allclose(batch_out, row_out, rtol=1e-12, atol=1e-12)


# =================================================================
# 2. CAUSALIDAD DE EXTREMO A EXTREMO
# =================================================================

N_BARS = 400
CUTOFF = 350  # barras 1..350 (indices 0..349) no deben verse afectadas


def _synthetic_ohlcv(n: int = N_BARS, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.03, n)
    close = 1_000_000.0 * np.exp(np.cumsum(rets))
    open_ = np.r_[close[0], close[:-1]] * (1.0 + rng.normal(0.0, 0.003, n))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.01, n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.01, n)))
    volume = rng.uniform(50.0, 150.0, n)
    index = pd.date_range("2022-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=index,
    )


def _process(raw: pd.DataFrame, tmp_path) -> tuple[pd.DataFrame, list[str]]:
    """Corre el pipeline real (indicadores + normalizacion) sobre datos inyectados."""
    pipeline = DataPipeline(
        symbol="TEST",
        interval="1d",
        normalization_window=40,
        cache_dir=str(tmp_path),
    )
    pipeline._raw_data = raw.copy()  # evita la descarga
    return pipeline.get_processed_data().copy(), pipeline.get_indicator_names()


def _causality_genome(indicator_names: list[str]) -> Genome:
    random.seed(3)
    genome = Genome(genome_id=1, num_inputs=len(indicator_names), num_outputs=3)
    genome.initialize_minimal(indicator_names=indicator_names, initial_active=6)
    genome.trading_params = TradingParams(
        stop_loss=0.08, take_profit=0.20, trailing_stop=0.05,
        max_drawdown=0.50, position_size=0.50,
    )
    return genome


def test_end_to_end_causality_future_prices_do_not_leak(tmp_path):
    raw = _synthetic_ohlcv()
    tampered = raw.copy()
    price_cols = ["Open", "High", "Low", "Close"]
    tampered.iloc[CUTOFF:, tampered.columns.get_indexer(price_cols)] *= 1000.0
    cutoff_date = raw.index[CUTOFF - 1]

    proc_a, names = _process(raw, tmp_path / "a")
    proc_b, names_b = _process(tampered, tmp_path / "b")
    assert names == names_b

    past_a = proc_a.loc[:cutoff_date]
    past_b = proc_b.loc[:cutoff_date]
    assert len(past_a) >= 60, "Muy pocas barras procesadas antes del corte"

    # (a) Indicadores normalizados identicos hasta la barra 350
    pd.testing.assert_frame_equal(past_a, past_b, check_exact=False, rtol=1e-12, atol=1e-12)
    # Sanidad: la alteracion realmente cambio el futuro
    assert not np.allclose(
        proc_a.loc[proc_a.index > cutoff_date, "Close"],
        proc_b.loc[proc_b.index > cutoff_date, "Close"],
    )

    genome = _causality_genome(names)
    active = genome.active_indicator_names

    # (b) Señales de la red identicas hasta la barra 350
    net = FeedForwardNetwork.from_genome(genome)
    signals_a = net.activate_batch(past_a[active].values)
    signals_b = net.activate_batch(past_b[active].values)
    assert np.allclose(signals_a, signals_b, rtol=1e-12, atol=1e-12)

    # (c) Capital y trades identicos hasta la barra 350
    bt = Backtester(
        initial_capital=100000.0, commission=0.0065, slippage=0.001,
        confidence_threshold=0.0, direction_threshold=0.05,
        min_trades=1, allow_short=False,
    )
    _, _, equity_a, trades_a = bt.evaluate_genome(genome, proc_a, names)
    _, _, equity_b, trades_b = bt.evaluate_genome(genome, proc_b, names)

    pd.testing.assert_series_equal(
        equity_a.loc[:cutoff_date], equity_b.loc[:cutoff_date],
        check_exact=False, rtol=1e-12, atol=1e-9,
    )

    closed_a = [t for t in trades_a if t.exit_time <= cutoff_date]
    closed_b = [t for t in trades_b if t.exit_time <= cutoff_date]
    assert len(closed_a) == len(closed_b)
    for ta, tb in zip(closed_a, closed_b):
        assert ta.entry_time == tb.entry_time
        assert ta.exit_time == tb.exit_time
        assert ta.exit_reason == tb.exit_reason
        assert ta.pnl == pytest.approx(tb.pnl, rel=1e-12)

    # La prueba solo es significativa si hubo actividad antes del corte.
    entries_before = [t for t in trades_a if t.entry_time <= cutoff_date]
    assert entries_before, "El genoma no opero antes del corte; la prueba no seria informativa"


# =================================================================
# 3. CONTABILIDAD INTRABARRA (trailing stop vs stop loss)
# =================================================================

def test_small_trailing_stop_does_not_preempt_stop_loss_on_first_bar():
    pos = Position(
        PositionType.LONG,
        entry_price=100.0,
        entry_time="2024-01-01",
        capital=10000.0,
        commission=0.0,
        slippage=0.0,
        stop_loss_pct=0.05,
        take_profit_pct=0.50,
        trailing_stop_pct=0.01,
    )
    assert pos.stop_loss_price == pytest.approx(95.0)
    # El trailing arranca en el piso del stop loss, no en 99.0
    assert pos.trailing_stop_price == pytest.approx(95.0)

    # Primera barra: Low = 98 (-2%). No debe cerrarse por el trailing del 1%.
    assert pos.check_gap_exit(open_price=99.5) is None
    assert pos.update_bar(high=100.0, low=98.0, close=98.5) is None
    assert pos.trailing_stop_price == pytest.approx(95.0)

    # Una caida por debajo de 95 si debe gatillar el STOP LOSS (no trailing).
    assert pos.update_bar(high=98.5, low=94.0, close=94.5) == "stop_loss"
    trade = pos.close(pos.stop_loss_price, "2024-01-03", "stop_loss")
    assert trade.exit_price == pytest.approx(95.0)
    assert trade.pnl == pytest.approx(-500.0)  # -5% de 10,000 sin costos
    assert trade.bars_held == 2


def test_small_trailing_stop_backtester_accounting(monkeypatch):
    """Mismo escenario dentro del Backtester: la salida ocurre en 95 por stop_loss."""
    n = 40
    opens = np.full(n, 100.0)
    closes = np.full(n, 100.0)
    highs = np.full(n, 100.0)
    lows = np.full(n, 99.5)
    lows[6] = 98.0                       # primera barra de la operacion: -2%
    opens[10], closes[10] = 99.0, 94.5   # barra que perfora el stop loss
    highs[10], lows[10] = 99.0, 94.0
    df = make_prices(opens, closes, highs, lows)

    # Compra al cierre de la barra 5 -> entrada en la apertura de la 6; luego HOLD.
    patch_network(monkeypatch, signal_outputs(n, long_bars=[5]))
    genome = make_genome(stop_loss=0.05, trailing_stop=0.01, take_profit=0.50,
                         max_drawdown=0.90, position_size=1.0)
    bt = Backtester(initial_capital=100000.0, commission=0.0, slippage=0.0,
                    min_trades=1, allow_short=False)
    _, _, equity, trades = bt.evaluate_genome(genome, df, ["ind0", "ind1"])

    assert len(trades) == 1
    trade = trades[0]
    assert trade.entry_time == df.index[6]
    assert trade.exit_time == df.index[10]
    assert trade.exit_reason == "stop_loss"
    assert trade.exit_price == pytest.approx(95.0)
    assert trade.pnl == pytest.approx(-5000.0)
    # La barra con Low=98 se marca a mercado sin cerrar la posicion.
    assert equity.iloc[6] == pytest.approx(100000.0)
    assert equity.iloc[-1] == pytest.approx(95000.0)
