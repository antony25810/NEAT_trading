"""
Utilidades compartidas por las pruebas: datos sinteticos, genomas
controlados y una red "falsa" con salidas predefinidas.
"""

import random

import numpy as np
import pandas as pd

from src.neat.genome import Genome, TradingParams
from src.trading import backtester as backtester_module


INDICATORS = ["ind0", "ind1"]


def make_genome(
    num_inputs: int = 2,
    stop_loss: float = 0.5,
    take_profit: float = 1.0,
    trailing_stop: float = 0.5,
    max_drawdown: float = 0.9,
    position_size: float = 1.0,
) -> Genome:
    """Genoma con todos los indicadores activos y riesgo controlado."""
    random.seed(0)
    genome = Genome(genome_id=1, num_inputs=num_inputs, num_outputs=3)
    genome.initialize_minimal(
        indicator_names=[f"ind{i}" for i in range(num_inputs)],
        initial_active=num_inputs,
    )
    genome.trading_params = TradingParams(
        stop_loss=stop_loss,
        take_profit=take_profit,
        trailing_stop=trailing_stop,
        max_drawdown=max_drawdown,
        position_size=position_size,
    )
    return genome


def make_prices(opens, closes=None, highs=None, lows=None,
                start: str = "2024-01-01") -> pd.DataFrame:
    """DataFrame OHLC + indicadores en cero a partir de listas de precios."""
    opens = np.asarray(opens, dtype=float)
    closes = opens.copy() if closes is None else np.asarray(closes, dtype=float)
    if highs is None:
        highs = np.maximum(opens, closes)
    if lows is None:
        lows = np.minimum(opens, closes)
    index = pd.date_range(start, periods=len(opens), freq="D")
    df = pd.DataFrame({
        "Open": opens,
        "High": np.asarray(highs, dtype=float),
        "Low": np.asarray(lows, dtype=float),
        "Close": closes,
        "Volume": 1.0,
    }, index=index)
    for name in INDICATORS:
        df[name] = 0.0
    return df


def signal_outputs(n: int, long_bars=(), short_bars=(), all_long=False,
                   all_short=False) -> np.ndarray:
    """
    Salidas de red (direccion, confianza, tamano) por barra.
    Las barras no indicadas quedan en efectivo (confianza 0).
    """
    out = np.zeros((n, 3))
    out[:, 2] = 1.0
    if all_long:
        out[:, 0] = 1.0
        out[:, 1] = 1.0
    if all_short:
        out[:, 0] = -1.0
        out[:, 1] = 1.0
    for i in long_bars:
        out[i] = [1.0, 1.0, 1.0]
    for i in short_bars:
        out[i] = [-1.0, 1.0, 1.0]
    return out


class _StubNetwork:
    def __init__(self, outputs: np.ndarray):
        self._outputs = outputs

    def activate_batch(self, input_matrix: np.ndarray) -> np.ndarray:
        return self._outputs[: len(input_matrix)]


def patch_network(monkeypatch, outputs: np.ndarray):
    """Hace que el Backtester use salidas predefinidas en lugar de la red."""
    stub_cls = type(
        "StubFeedForward",
        (),
        {"from_genome": staticmethod(lambda genome, *a, **k: _StubNetwork(outputs))},
    )
    monkeypatch.setattr(backtester_module, "FeedForwardNetwork", stub_cls)


def synthetic_market(n: int = 400, n_indicators: int = 4, seed: int = 0) -> pd.DataFrame:
    """Mercado sintetico con indicadores acotados (sin fuga de futuro)."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.001, 0.03, n)
    close = 100.0 * np.exp(np.cumsum(rets))
    open_ = np.r_[close[0], close[:-1]] * (1.0 + rng.normal(0.0, 0.005, n))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.01, n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.01, n)))
    index = pd.date_range("2020-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "Open": open_, "High": high, "Low": low, "Close": close, "Volume": 1.0,
    }, index=index)
    past_momentum = pd.Series(rets, index=index).rolling(5, min_periods=1).mean()
    df["ind0"] = np.tanh(past_momentum.values * 50.0)
    for k in range(1, n_indicators):
        df[f"ind{k}"] = np.tanh(rng.normal(size=n))
    return df
