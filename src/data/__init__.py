"""
Modulo de datos: descarga, indicadores, normalizacion y Walk-Forward.

Uso rapido:
-----------
>>> from src.data import DataPipeline
>>> pipeline = DataPipeline(symbol="BTC-MXN", start="2018-01-01")
>>> for window in pipeline.walk_forward():
...     # window.train, window.val, window.test
...     pass

Componentes individuales:
--------------------------
>>> from src.data.downloader import DataDownloader
>>> from src.data.indicators import IndicatorEngine, INDICATOR_REGISTRY
>>> from src.data.normalizer import RollingNormalizer
>>> from src.data.walk_forward import WalkForwardSplitter
"""

from src.data.pipeline import DataPipeline
from src.data.downloader import DataDownloader
from src.data.indicators import IndicatorEngine, INDICATOR_REGISTRY
from src.data.normalizer import RollingNormalizer
from src.data.walk_forward import WalkForwardSplitter, WalkForwardWindow

__all__ = [
    "DataPipeline",
    "DataDownloader",
    "IndicatorEngine",
    "INDICATOR_REGISTRY",
    "RollingNormalizer",
    "WalkForwardSplitter",
    "WalkForwardWindow",
]
