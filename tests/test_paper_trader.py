"""
Pruebas unitarias y de integración para el módulo de Paper Trading en vivo:
1. Deserialización del Campeón (`ChampionBrain` + `FeedForwardNetwork.from_genome`)
2. Alimentador en Vivo (`BitsoRESTClient` + `DataPipeline.get_live_data` con 260 velas)
3. Motor de Ejecución Diaria (`PaperTrader.run_daily_cycle` + idempotencia + HOLD)
4. Auditor de Fricción Real (`RealFrictionAuditor` con precios Ask/Bid del Order Book)
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.data.normalizer import RollingNormalizer
from src.data.pipeline import DataPipeline
from src.live.auditor import RealFrictionAuditor
from src.live.bitso_client import BitsoConfig, BitsoRESTClient, OrderBookSnapshot
from src.live.paper_trader import ChampionBrain, NeuralDecision, PaperTrader
from src.trading.backtester import Backtester
from src.trading.position import PositionType


def _make_synthetic_bitso_ohlc_payload(
    num_closed_bars: int = 265,
    include_open_bar: bool = True,
    now_utc: datetime = datetime(2026, 11, 8, 18, 5, 0, tzinfo=timezone.utc),
) -> list[dict]:
    """
    Construye un payload realista estilo Bitso v3 `/ohlc` con `bucket_start_time` en ms.
    Una vela diaria iniciada en `t` cierra en `t + 86400s`.
    """
    bucket_sec = 86400
    # Alinear al inicio del día UTC para velas cerradas
    ref_day_start = datetime(
        now_utc.year, now_utc.month, now_utc.day, 0, 0, 0, tzinfo=timezone.utc
    )
    # `ref_day_start` inició hace 18h05m (<24h), por lo tanto SIGUE ABIERTA.
    # La última vela cerrada inició en `ref_day_start - 1 día`.
    last_closed_start_ts = int(ref_day_start.timestamp()) - bucket_sec

    rng = np.random.default_rng(2026)
    base_prices = 1_200_000.0 + np.cumsum(rng.normal(2_500.0, 18_000.0, size=num_closed_bars + 1))
    base_prices = np.maximum(base_prices, 500_000.0)

    payload = []
    for i in range(num_closed_bars):
        start_sec = last_closed_start_ts - (num_closed_bars - 1 - i) * bucket_sec
        close_p = float(base_prices[i])
        open_p = float(close_p * (1.0 + rng.normal(0, 0.006)))
        high_p = float(max(open_p, close_p) * (1.0 + abs(rng.normal(0.01, 0.005))))
        low_p = float(min(open_p, close_p) * (1.0 - abs(rng.normal(0.01, 0.005))))
        vol = float(rng.uniform(35.0, 220.0))
        payload.append(
            {
                "bucket_start_time": start_sec * 1000,
                "first_rate": f"{open_p:.2f}",
                "max_rate": f"{high_p:.2f}",
                "min_rate": f"{low_p:.2f}",
                "last_rate": f"{close_p:.2f}",
                "volume": f"{vol:.8f}",
            }
        )

    if include_open_bar:
        open_bar_sec = int(ref_day_start.timestamp())  # Inició hoy 00:00 UTC, aún no cumple 24h
        p = float(base_prices[-1] * 1.50)  # Salto artificial enorme en vela inconclusa
        payload.append(
            {
                "bucket_start_time": open_bar_sec * 1000,
                "first_rate": f"{p:.2f}",
                "max_rate": f"{p * 1.02:.2f}",
                "min_rate": f"{p * 0.98:.2f}",
                "last_rate": f"{p * 1.01:.2f}",
                "volume": "12.50000000",
            }
        )

    return payload


# =====================================================================
# 1. PRUEBA DEL CEREBRO CAMPEÓN (Deserialización + Paridad Exacta)
# =====================================================================
def test_champion_brain_deserialization_and_parity():
    genome_path = Path("models/best_genome_seed2026_w1.json")
    assert genome_path.exists(), "Debe existir copia versionada del campeón seed2026"

    brain = ChampionBrain.from_json(
        genome_path=genome_path,
        confidence_threshold=0.30,
        direction_threshold=0.20,
        allow_short=False,
    )

    assert brain.genome.genome_id == 2100
    assert brain.active_indicator_names == [
        "retorno_1d",
        "retorno_5d",
        "distancia_SMA200",
        "MACD",
        "Momentum",
        "Volumen_relativo",
    ]
    assert "distancia_SMA200" in brain.active_indicator_names

    # Comprobar paridad matemática exacta entre `ChampionBrain.evaluate_row` (activate)
    # y `FeedForwardNetwork.activate_batch` + `Backtester._decode_signal` en 50 filas
    rng = np.random.default_rng(42)
    n_rows = 50
    matrix = rng.uniform(-0.95, 0.95, size=(n_rows, len(brain.active_indicator_names)))
    batch_outputs = brain.network.activate_batch(matrix)

    backtester = Backtester(
        confidence_threshold=0.30,
        direction_threshold=0.20,
        allow_short=False,
    )

    for idx in range(n_rows):
        row_dict = {
            name: float(matrix[idx, col_i])
            for col_i, name in enumerate(brain.active_indicator_names)
        }
        decision = brain.evaluate_row(row_dict)
        expected_dir, expected_conf, expected_size = batch_outputs[idx]

        assert np.allclose(
            [decision.direction, decision.confidence, decision.size_scaling],
            [expected_dir, expected_conf, expected_size],
            atol=1e-12,
        )

        expected_pos, _ = backtester._decode_signal(batch_outputs[idx])
        assert decision.desired_position == expected_pos


# =====================================================================
# 2. PRUEBA DEL ALIMENTADOR EN VIVO (260 Velas + SMA200 + Cero Leakage)
# =====================================================================
def test_bitso_client_and_260_bars_live_pipeline_no_leakage():
    now_ref = datetime(2026, 11, 8, 18, 5, 0, tzinfo=timezone.utc)
    ohlc_payload = _make_synthetic_bitso_ohlc_payload(
        num_closed_bars=265,
        include_open_bar=True,
        now_utc=now_ref,
    )

    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"success": True, "payload": ohlc_payload}
    mock_session.request.return_value = mock_resp

    client = BitsoRESTClient(
        config=BitsoConfig(book="btc_mxn", lookback_bars=260),
        session=mock_session,
    )

    brain = ChampionBrain.from_json("models/best_genome_seed2026_w1.json")
    pipeline = DataPipeline(symbol="BTC-MXN", interval="1d")

    df_live = pipeline.get_live_data(
        bitso_client=client,
        lookback_bars=260,
        exclude_open_candle=True,
        required_indicators=brain.active_indicator_names,
        now_utc=now_ref,
    )

    # 1) Debe haber solicitado y retenido exactamente 260 velas cerradas
    assert len(pipeline._raw_data) == 260
    # La vela abierta de hoy (2026-11-08 00:00) debe haber sido excluida;
    # la última vela cerrada es 2026-11-07
    assert pipeline._raw_data.index[-1] == pd.Timestamp("2026-11-07 00:00:00")

    # 2) Todos los indicadores del campeón (incluida distancia_SMA200) están presentes y sin NaN
    latest_row = df_live.iloc[-1]
    for ind in brain.active_indicator_names:
        assert ind in latest_row.index
        assert np.isfinite(latest_row[ind]), f"Indicador {ind} no debe ser NaN en vela 260"
        assert -1.0 <= float(latest_row[ind]) <= 1.0

    # 3) Verificar que indicadores rápidos usan exactamente la ventana Z-score de 252 barras
    raw_ind = pipeline.indicator_engine.compute(pipeline._raw_data)
    strict_norm = RollingNormalizer(window=252, method="zscore").normalize_columns(
        raw_ind, ["retorno_1d", "MACD", "Momentum", "Volumen_relativo"]
    )
    for fast_col in ["retorno_1d", "MACD", "Momentum", "Volumen_relativo"]:
        assert np.isclose(
            df_live[fast_col].iloc[-1],
            strict_norm[fast_col].iloc[-1],
            atol=1e-12,
        )

    # 4) Prueba estricta de CERO Data Leakage: agregar 10 velas futuras extremas (*1000)
    # no debe alterar ni un bit de la fila en 2026-11-07
    future_dates = pd.date_range("2026-11-08", periods=10, freq="D")
    future_df = pd.DataFrame(
        {
            "Open": 999_000_000.0,
            "High": 1_200_000_000.0,
            "Low": 800_000_000.0,
            "Close": 1_100_000_000.0,
            "Volume": 99_999.0,
        },
        index=future_dates,
    )
    extended_raw = pd.concat([pipeline._raw_data, future_df])
    df_extended = pipeline.process_live_data(
        extended_raw,
        required_indicators=brain.active_indicator_names,
    )

    target_ts = pd.Timestamp("2026-11-07 00:00:00")
    for col in brain.active_indicator_names:
        assert np.isclose(
            df_live.loc[target_ts, col],
            df_extended.loc[target_ts, col],
            atol=1e-12,
        ), f"Fuga de datos detectada en {col}"


# =====================================================================
# 3. PRUEBA E2E: BUY AL ASK REAL, IDEMPOTENCIA, HOLD Y SELL AL BID REAL
# =====================================================================
def test_paper_trader_buy_ask_hold_idempotency_and_sell_bid(tmp_path: Path):
    trades_csv = tmp_path / "live_trades.csv"
    state_json = tmp_path / "live_state.json"

    brain = ChampionBrain.from_json("models/best_genome_seed2026_w1.json")
    pipeline = DataPipeline(symbol="BTC-MXN", interval="1d")
    auditor = RealFrictionAuditor(
        trades_csv_path=trades_csv,
        state_json_path=state_json,
        commission=0.0065,
    )

    # Construir historial base de 263 velas diarias cerradas
    dates = pd.date_range("2026-02-18", periods=263, freq="D")
    rng = np.random.default_rng(77)
    closes = 1_400_000.0 + np.cumsum(rng.normal(1_000.0, 8_000.0, size=263))
    full_ohlcv = pd.DataFrame(
        {
            "Open": closes * 0.998,
            "High": closes * 1.010,
            "Low": closes * 0.990,
            "Close": closes,
            "Volume": rng.uniform(50.0, 150.0, size=263),
        },
        index=dates,
    )
    # Fijar precios exactos de cierre para los últimos 3 días de prueba
    # Día 1: 2026-11-05 | Día 2: 2026-11-06 | Día 3: 2026-11-07
    full_ohlcv.iloc[-3] = [1_495_000.0, 1_505_000.0, 1_490_000.0, 1_500_000.0, 100.0]
    full_ohlcv.iloc[-2] = [1_500_000.0, 1_540_000.0, 1_495_000.0, 1_530_000.0, 110.0]
    full_ohlcv.iloc[-1] = [1_530_000.0, 1_610_000.0, 1_525_000.0, 1_600_000.0, 125.0]

    mock_client = MagicMock(spec=BitsoRESTClient)
    mock_client.symbol_to_book.side_effect = BitsoRESTClient.symbol_to_book

    trader = PaperTrader(
        brain=brain,
        bitso_client=mock_client,
        pipeline=pipeline,
        auditor=auditor,
        symbol="BTC-MXN",
        initial_capital=100_000.0,
        lookback_bars=260,
    )

    # --- DÍA 1 (2026-11-05): Señal alcista -> COMPRA al Ask real (1,501,800.00) ---
    mock_client.fetch_ohlcv.return_value = full_ohlcv.iloc[:261].tail(260).copy()
    mock_client.fetch_order_book.return_value = OrderBookSnapshot(
        book="btc_mxn",
        best_bid=1_499_200.00,
        best_ask=1_501_800.00,
        best_bid_amount=0.85,
        best_ask_amount=0.60,
        mid_price=1_500_500.00,
        spread_mxn=2_600.00,
        spread_bps=(2_600.0 / 1_500_500.0) * 10_000.0,
        timestamp_utc="2026-11-05T18:00:00.123+00:00",
    )
    brain.evaluate_row = MagicMock(
        return_value=NeuralDecision(
            direction=0.72,
            confidence=0.81,
            size_scaling=1.0,  # Fracción efectiva = position_size * 1.0
            desired_position=PositionType.LONG,
            effective_position_fraction=brain.genome.trading_params.position_size,
            active_inputs={"distancia_SMA200": 0.45},
        )
    )

    rep_day1 = trader.run_daily_cycle()
    assert rep_day1.action == "BUY"
    assert rep_day1.candle_date == "2026-11-05"
    assert rep_day1.candle_close == 1_500_000.00
    assert rep_day1.execution_price == 1_501_800.00  # ¡Precio Ask real, NO el Close!
    assert mock_client.fetch_order_book.call_count == 1

    # --- DÍA 1 REPETIDO: Protección de Idempotencia ---
    rep_dup = trader.run_daily_cycle()
    assert rep_dup.action == "SKIPPED_ALREADY_PROCESSED"
    assert mock_client.fetch_order_book.call_count == 1  # No vuelve a consultar ni comprar

    # --- DÍA 2 (2026-11-06): Zona Neutral (dir=+0.08) -> HOLD sin fricción ---
    mock_client.fetch_ohlcv.return_value = full_ohlcv.iloc[:262].tail(260).copy()
    brain.evaluate_row = MagicMock(
        return_value=NeuralDecision(
            direction=0.08,  # Zona neutral [-0.2, 0.2]
            confidence=0.65,
            size_scaling=0.5,
            desired_position=None,  # HOLD
            effective_position_fraction=brain.genome.trading_params.position_size * 0.5,
            active_inputs={"distancia_SMA200": 0.50},
        )
    )

    rep_day2 = trader.run_daily_cycle()
    assert rep_day2.action == "HOLD_LONG"
    assert rep_day2.candle_date == "2026-11-06"
    assert mock_client.fetch_order_book.call_count == 1  # Cero llamadas al Order Book en HOLD

    saved_state_d2 = auditor.load_state()
    assert saved_state_d2.open_position is not None
    assert saved_state_d2.open_position.bars_held == 1
    assert saved_state_d2.open_position.peak_price == 1_540_000.0  # Actualizó máximo intrabarra

    # --- DÍA 3 (2026-11-07): Convicción Bajista (dir=-0.78, conf=0.88) -> VENTA al Bid real ---
    mock_client.fetch_ohlcv.return_value = full_ohlcv.iloc[:263].tail(260).copy()
    mock_client.fetch_order_book.return_value = OrderBookSnapshot(
        book="btc_mxn",
        best_bid=1_598_400.00,
        best_ask=1_601_200.00,
        best_bid_amount=1.20,
        best_ask_amount=0.95,
        mid_price=1_599_800.00,
        spread_mxn=2_800.00,
        spread_bps=(2_800.0 / 1_599_800.0) * 10_000.0,
        timestamp_utc="2026-11-07T18:00:00.045+00:00",
    )
    brain.evaluate_row = MagicMock(
        return_value=NeuralDecision(
            direction=-0.78,
            confidence=0.88,
            size_scaling=0.6,
            desired_position=PositionType.CASH,
            effective_position_fraction=brain.genome.trading_params.position_size * 0.6,
            active_inputs={"distancia_SMA200": 0.58},
        )
    )

    rep_day3 = trader.run_daily_cycle()
    assert rep_day3.action == "SELL"
    assert rep_day3.reason == "signal"
    assert rep_day3.candle_close == 1_600_000.00
    assert rep_day3.execution_price == 1_598_400.00  # ¡Precio Bid real, NO el Close!

    # Verificar contabilidad exacta en `live_trades.csv`
    with open(trades_csv, "r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    assert len(rows) == 2
    buy_row, sell_row = rows[0], rows[1]

    assert buy_row["side"] == "BUY"
    assert float(buy_row["execution_price"]) == pytest.approx(1_501_800.00)
    assert float(buy_row["best_ask"]) == pytest.approx(1_501_800.00)
    assert float(buy_row["candle_close"]) == pytest.approx(1_500_000.00)

    assert sell_row["side"] == "SELL"
    assert sell_row["reason"] == "signal"
    assert float(sell_row["execution_price"]) == pytest.approx(1_598_400.00)
    assert float(sell_row["best_bid"]) == pytest.approx(1_598_400.00)
    assert int(sell_row["bars_held"]) == 2

    # Comprobar PnL neto exacto con Ask de entrada + 0.65% comisión y Bid de salida + 0.65% comisión
    alloc = 100_000.0 * brain.genome.trading_params.position_size
    btc_bought = (alloc * (1.0 - 0.0065)) / 1_501_800.00
    net_proceeds = (btc_bought * 1_598_400.00) * (1.0 - 0.0065)
    expected_net_pnl = net_proceeds - alloc

    assert float(sell_row["realized_pnl_mxn"]) == pytest.approx(expected_net_pnl, abs=0.05)
    assert float(sell_row["equity_after_mxn"]) == pytest.approx(
        100_000.0 + expected_net_pnl, abs=0.05
    )


# =====================================================================
# 4. PRUEBA DE STOP LOSS CON LIQUIDACIÓN AL BID REAL
# =====================================================================
def test_paper_trader_stop_loss_sells_at_real_bid(tmp_path: Path):
    trades_csv = tmp_path / "live_trades_sl.csv"
    state_json = tmp_path / "live_state_sl.json"

    brain = ChampionBrain.from_json("models/best_genome_seed2026_w1.json")
    pipeline = DataPipeline(symbol="BTC-MXN", interval="1d")
    auditor = RealFrictionAuditor(
        trades_csv_path=trades_csv,
        state_json_path=state_json,
        commission=0.0065,
    )

    dates = pd.date_range("2026-02-19", periods=262, freq="D")
    closes = np.linspace(1_300_000.0, 1_500_000.0, 262)
    full_ohlcv = pd.DataFrame(
        {
            "Open": closes,
            "High": closes * 1.005,
            "Low": closes * 0.995,
            "Close": closes,
            "Volume": 100.0,
        },
        index=dates,
    )
    # Día 1: compra a Ask = 1,500,000. Stop loss del genoma ~11.08% -> ~1,333,781 MXN
    # Día 2: caída abrupta con Low = 1,280,000 (atraviesa stop_loss_price)
    full_ohlcv.iloc[-1] = [1_490_000.0, 1_495_000.0, 1_280_000.0, 1_310_000.0, 350.0]

    mock_client = MagicMock(spec=BitsoRESTClient)
    mock_client.symbol_to_book.side_effect = BitsoRESTClient.symbol_to_book

    trader = PaperTrader(
        brain=brain,
        bitso_client=mock_client,
        pipeline=pipeline,
        auditor=auditor,
        symbol="BTC-MXN",
        initial_capital=100_000.0,
        lookback_bars=260,
    )

    # Día 1: BUY
    mock_client.fetch_ohlcv.return_value = full_ohlcv.iloc[:261].tail(260).copy()
    mock_client.fetch_order_book.return_value = OrderBookSnapshot(
        book="btc_mxn",
        best_bid=1_499_000.0,
        best_ask=1_500_000.0,
        best_bid_amount=1.0,
        best_ask_amount=1.0,
        mid_price=1_499_500.0,
        spread_mxn=1_000.0,
        spread_bps=6.67,
        timestamp_utc="2026-11-06T18:00:00Z",
    )
    brain.evaluate_row = MagicMock(
        return_value=NeuralDecision(
            direction=0.80,
            confidence=0.90,
            size_scaling=1.0,
            desired_position=PositionType.LONG,
            effective_position_fraction=brain.genome.trading_params.position_size,
            active_inputs={},
        )
    )
    assert trader.run_daily_cycle().action == "BUY"

    # Día 2: Aunque la red diga HOLD, el Stop Loss se activa y liquida al Bid real
    mock_client.fetch_ohlcv.return_value = full_ohlcv.iloc[:262].tail(260).copy()
    mock_client.fetch_order_book.return_value = OrderBookSnapshot(
        book="btc_mxn",
        best_bid=1_308_500.0,
        best_ask=1_311_500.0,
        best_bid_amount=2.0,
        best_ask_amount=1.5,
        mid_price=1_310_000.0,
        spread_mxn=3_000.0,
        spread_bps=22.90,
        timestamp_utc="2026-11-07T18:00:00Z",
    )
    brain.evaluate_row = MagicMock(
        return_value=NeuralDecision(
            direction=0.05,
            confidence=0.40,
            size_scaling=0.5,
            desired_position=None,
            effective_position_fraction=0.3,
            active_inputs={},
        )
    )
    rep_sl = trader.run_daily_cycle()
    assert rep_sl.action == "SELL"
    assert rep_sl.reason == "stop_loss"
    assert rep_sl.execution_price == 1_308_500.0
    assert rep_sl.realized_pnl_mxn < 0


# =====================================================================
# 5. PRUEBA DE AUTENTICACIÓN HMAC OPCIONAL EN BITSO CLIENT
# =====================================================================
def test_bitso_hmac_auth_header_generation():
    cfg = BitsoConfig(api_key="my_test_key", api_secret="my_test_secret")
    client = BitsoRESTClient(config=cfg)
    headers = client._build_auth_header("GET", "/api/v3/balance/")
    assert "Authorization" in headers
    assert headers["Authorization"].startswith("Bitso my_test_key:")
    parts = headers["Authorization"].split(":")
    assert len(parts) == 3
    assert len(parts[2]) == 64  # SHA-256 hex digest

