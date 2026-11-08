"""
Paquete de operación en vivo / Paper Trading con auditoría de fricción real.
"""

from src.live.auditor import PortfolioState, RealFrictionAuditor
from src.live.bitso_client import BitsoAPIError, BitsoConfig, BitsoRESTClient, OrderBookSnapshot
from src.live.paper_trader import ChampionBrain, DailyCycleReport, NeuralDecision, PaperTrader

__all__ = [
    "BitsoAPIError",
    "BitsoConfig",
    "BitsoRESTClient",
    "OrderBookSnapshot",
    "PortfolioState",
    "RealFrictionAuditor",
    "ChampionBrain",
    "DailyCycleReport",
    "NeuralDecision",
    "PaperTrader",
]

