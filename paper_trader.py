#!/usr/bin/env python3
"""
============================================================
paper_trader.py - Ejecutor Diario de Paper Trading (NEAT + Bitso)
============================================================

Diseñado para ejecutarse automáticamente una vez al día mediante un
programador de tareas justo al cierre de la vela diaria (ej. 18:00 UTC
o 00:00 UTC).

FLUJO DE EJECUCIÓN:
-------------------
1. El Cerebro: Carga `best_genome.json` y reconstruye la topología y
   pesos exactos con `FeedForwardNetwork.from_genome()`.
2. El Alimentador: Usa `DataPipeline.get_live_data()` + `BitsoRESTClient`
   para solicitar únicamente las últimas 260 velas diarias cerradas,
   calculando `SMA200` y aplicando `RollingNormalizer(window=252)` sin
   Data Leakage.
3. Motor Diario: Toma la fila normalizada de la vela recién cerrada,
   evalúa `active_indicator_names` con `network.activate()` y decodifica
   `(Direction, Confidence, Position Size)`.
4. El Auditor: Si toca comprar, consulta el Order Book público en ese
   milisegundo y registra la compra en `live_trades.csv` usando el precio
   `Ask` real. Si toca vender (por señal o por stop), registra la venta
   al precio `Bid` real.

EJEMPLOS DE USO:
----------------
  # Revisar el campeón cargado y estado actual (sin llamadas de red)
  python paper_trader.py --status

  # Simulacro en vivo (consulta Bitso OHLC + Order Book sin escribir CSV/estado)
  python paper_trader.py --dry-run

  # Ejecución diaria real (para Programador de Tareas de Windows o cron)
  python paper_trader.py

PROGRAMADOR DE TAREAS EN WINDOWS (Ejemplo 18:00 UTC = 12:00 CST):
-----------------------------------------------------------------
  schtasks /Create /SC DAILY /TN "NEAT_Bitso_PaperTrader" ^
    /TR "\"C:\\Users\\anton\\OneDrive\\Documentos\\Neuroevolucion\\venv\\Scripts\\python.exe\" \"C:\\Users\\anton\\OneDrive\\Documentos\\Neuroevolucion\\paper_trader.py\"" ^
    /ST 12:00 /F
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.live.paper_trader import PaperTrader


def configure_logging(log_file: str | None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        handlers=handlers,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Motor Diario de Paper Trading NEAT con Fricción Real (Bitso REST)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/trading_config.yaml",
        help="Ruta al archivo YAML de configuración (default: config/trading_config.yaml)",
    )
    parser.add_argument(
        "--genome",
        type=str,
        default=None,
        help="Ruta opcional a un archivo best_genome.json específico",
    )
    parser.add_argument(
        "--book",
        type=str,
        default=None,
        help="Libro de Bitso a operar (ej: btc_mxn, eth_mxn)",
    )
    parser.add_argument(
        "--bars",
        type=int,
        default=None,
        help="Número de velas diarias cerradas a solicitar (default en YAML: 260)",
    )
    parser.add_argument(
        "--trades-csv",
        type=str,
        default=None,
        help="Ruta de salida para el registro auditado live_trades.csv",
    )
    parser.add_argument(
        "--state-file",
        type=str,
        default=None,
        help="Ruta del archivo de estado persistente live_state.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Evalúa datos y Order Book real sin modificar live_trades.csv ni live_state.json",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Fuerza la evaluación aunque la última vela cerrada ya figure procesada",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Muestra el resumen del cerebro campeón y el estado actual sin consultar red",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="logs/paper_trader.log",
        help="Archivo de bitácora rotativa/acumulativa (default: logs/paper_trader.log)",
    )
    return parser


def print_status(trader: PaperTrader) -> None:
    state = trader.auditor.load_state(initial_capital=trader.initial_capital)
    summary = {
        "champion_genome": {
            "source_path": trader.brain.source_path,
            "genome_id": trader.brain.genome.genome_id,
            "fitness": trader.brain.genome.fitness,
            "nodes": len(trader.brain.genome.nodes),
            "hidden_nodes": len(trader.brain.genome.hidden_node_ids),
            "active_connections": len(trader.brain.genome.active_connections),
            "active_indicator_names": trader.brain.active_indicator_names,
            "trading_params": trader.brain.genome.trading_params.to_dict(),
        },
        "feeder_config": {
            "base_url": trader.bitso_client.config.base_url,
            "ohlc_endpoint": trader.bitso_client.config.ohlc_endpoint,
            "order_book_endpoint": trader.bitso_client.config.order_book_endpoint,
            "book": trader.book,
            "time_bucket_sec": trader.bitso_client.config.time_bucket,
            "lookback_bars": trader.lookback_bars,
            "normalization_window": trader.pipeline.normalizer.window,
        },
        "portfolio_state": state.to_dict(),
        "audit_files": {
            "trades_csv": str(trader.auditor.trades_csv_path),
            "state_json": str(trader.auditor.state_json_path),
        },
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> int:
    args = build_parser().parse_args()
    configure_logging(None if args.status else args.log_file)

    try:
        trader = PaperTrader.from_config(
            config_path=args.config,
            genome_override=args.genome,
            book_override=args.book,
            bars_override=args.bars,
            trades_csv_override=args.trades_csv,
            state_json_override=args.state_file,
        )

        if args.status:
            print_status(trader)
            return 0

        report = trader.run_daily_cycle(
            force=args.force,
            dry_run=args.dry_run,
        )
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
        return 0

    except Exception as exc:
        logging.getLogger("paper_trader").exception(
            f"Error crítico durante la ejecución diaria: {exc}"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
