"""
============================================================
pipeline.py - Pipeline de Datos Completo
============================================================

QUE ES UN PIPELINE?
-------------------
Un pipeline es una CADENA DE PROCESAMIENTO donde la salida de
un paso es la entrada del siguiente. En nuestro caso:

  Datos crudos (OHLCV)
       |
       v
  [1. Descarga] ──► Datos OHLCV limpios
       |
       v
  [2. Indicadores] ──► Datos + indicadores tecnicos
       |
       v
  [3. Normalizacion] ──► Datos normalizados [-1, 1]
       |
       v
  [4. Walk-Forward] ──► Ventanas (train, val, test)

Esta clase orquesta los 4 modulos que creamos antes
(downloader, indicators, normalizer, walk_forward)
en una sola interfaz simple y limpia.

POR QUE UNA CLASE PIPELINE?
----------------------------
Sin pipeline, tendrias que escribir esto cada vez:
  downloader = DataDownloader()
  df = downloader.download(...)
  engine = IndicatorEngine(...)
  df = engine.compute_clean(...)
  normalizer = RollingNormalizer(...)
  df = normalizer.normalize_columns(...)
  splitter = WalkForwardSplitter(...)
  windows = splitter.split(...)

Con pipeline, es simplemente:
  pipeline = DataPipeline(...)
  for window in pipeline.walk_forward():
      # usar window.train, window.val, window.test

Mucho mas limpio y menos propenso a errores.
"""

import logging
from typing import Iterator, Optional

import pandas as pd

from src.data.downloader import DataDownloader
from src.data.indicators import IndicatorEngine, INDICATOR_REGISTRY
from src.data.normalizer import RollingNormalizer
from src.data.walk_forward import (
    WalkForwardSplitter,
    WalkForwardWindow,
    create_daily_splitter,
    create_hourly_splitter
)

logger = logging.getLogger(__name__)


class DataPipeline:
    """
    Pipeline completo: descarga → indicadores → normalizacion → Walk-Forward.
    
    Uso:
    ----
    >>> pipeline = DataPipeline(
    ...     symbol="BTC-MXN",
    ...     start="2018-01-01",
    ...     interval="1d"
    ... )
    >>> 
    >>> # Obtener datos procesados (sin Walk-Forward)
    >>> df = pipeline.get_processed_data()
    >>> 
    >>> # Iterar sobre ventanas Walk-Forward
    >>> for window in pipeline.walk_forward():
    ...     print(f"Train: {len(window.train)} barras")
    ...     print(f"Val: {len(window.val)} barras")
    ...     print(f"Test: {len(window.test)} barras")
    """
    
    def __init__(
        self,
        symbol: str,
        start: str = "2018-01-01",
        end: Optional[str] = None,
        interval: str = "1d",
        indicators: Optional[list[str]] = None,
        normalization_method: str = "zscore",
        normalization_window: Optional[int] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Configura el pipeline de datos.
        
        Parameters
        ----------
        symbol : str
            Simbolo del activo (ej: "BTC-MXN", "ETH-MXN")
        start : str
            Fecha de inicio de los datos ("YYYY-MM-DD")
        end : str, opcional
            Fecha de fin (default: hoy)
        interval : str
            Intervalo temporal: "1d" (diario) o "1h" (horario)
        indicators : list[str], opcional
            Lista de indicadores a calcular. None = todos.
        normalization_method : str
            Metodo: "zscore" o "minmax"
        normalization_window : int, opcional
            Ventana de normalizacion. Default:
            - 252 para datos diarios (~1 ano)
            - 24*30 = 720 para datos horarios (~1 mes)
        cache_dir : str, opcional
            Directorio de cache para datos descargados
        """
        self.symbol = symbol
        self.start = start
        self.end = end
        self.interval = interval
        
        # Componentes del pipeline
        self.downloader = DataDownloader(cache_dir=cache_dir)
        self.indicator_engine = IndicatorEngine(indicators=indicators)
        
        # Determinar ventana de normalizacion segun el intervalo
        if normalization_window is None:
            if interval in ("1d", "1D"):
                normalization_window = 252  # ~1 ano de datos diarios
            elif interval in ("1h", "60m"):
                normalization_window = 24 * 30  # ~1 mes de datos horarios
            else:
                normalization_window = 100  # Default conservador
        
        self.normalizer = RollingNormalizer(
            window=normalization_window,
            method=normalization_method
        )
        
        # Guardar la lista de indicadores activos
        self.active_indicators = self.indicator_engine.active_indicators
        
        # Cache interno del DataFrame procesado
        self._processed_data: Optional[pd.DataFrame] = None
        self._raw_data: Optional[pd.DataFrame] = None
        
        logger.info(
            f"DataPipeline configurado: {symbol} | {interval} | "
            f"{len(self.active_indicators)} indicadores | "
            f"normalizacion: {normalization_method} (window={normalization_window})"
        )
    
    def _is_mxn_pair(self) -> bool:
        """Detecta si el simbolo es un par crypto/MXN sintetico."""
        s = self.symbol.upper()
        return s.endswith("-MXN") or s.endswith("_MXN")

    def get_raw_data(self) -> pd.DataFrame:
        """
        Descarga y retorna datos crudos OHLCV (sin indicadores ni normalizacion).
        
        Si el simbolo es un par crypto/MXN (ej: "BTC-MXN"), automaticamente
        descarga el par USD y el tipo de cambio, y los combina.
        
        Returns
        -------
        pd.DataFrame
            Datos OHLCV limpios
        """
        if self._raw_data is None:
            logger.info(f"Descargando datos crudos para {self.symbol}...")
            
            if self._is_mxn_pair():
                # Extraer el simbolo crypto (ej: "BTC-MXN" -> "BTC")
                crypto = self.symbol.upper().replace("-MXN", "").replace("_MXN", "")
                logger.info(
                    f"Par MXN detectado. Convirtiendo {crypto}-USD * USD/MXN"
                )
                self._raw_data = self.downloader.download_mxn(
                    crypto=crypto,
                    start=self.start,
                    end=self.end,
                    interval=self.interval
                )
            else:
                self._raw_data = self.downloader.download(
                    symbol=self.symbol,
                    start=self.start,
                    end=self.end,
                    interval=self.interval
                )
            
            logger.info(f"Datos crudos: {len(self._raw_data)} barras")
        
        return self._raw_data
    
    def get_processed_data(self) -> pd.DataFrame:
        """
        Retorna datos completos: OHLCV + indicadores + normalizacion.
        
        FLUJO INTERNO:
        1. Descargar OHLCV (o usar cache)
        2. Calcular indicadores tecnicos
        3. Eliminar NaN del calentamiento
        4. Normalizar SOLO los indicadores (no OHLCV)
        
        POR QUE NO NORMALIZAR OHLCV?
        Porque necesitamos los precios REALES para el backtesting.
        Si normalizamos el precio a [-1,1], no podemos calcular
        ganancias/perdidas reales.
        
        Los indicadores SI se normalizan porque son los inputs
        de la red neuronal.
        
        Returns
        -------
        pd.DataFrame
            DataFrame con OHLCV original + indicadores normalizados
        """
        if self._processed_data is not None:
            return self._processed_data
        
        # Paso 1: Datos crudos
        raw = self.get_raw_data()
        logger.info(f"Paso 1/3: Datos crudos: {len(raw)} barras")
        
        # Paso 2: Calcular indicadores (sin eliminar NaN aun)
        df_with_indicators = self.indicator_engine.compute(raw)
        logger.info(
            f"Paso 2/3: Indicadores calculados: "
            f"{len(self.active_indicators)} indicadores"
        )
        
        # Eliminar filas con NaN (periodo de calentamiento)
        rows_before = len(df_with_indicators)
        df_clean = df_with_indicators.dropna()
        rows_dropped = rows_before - len(df_clean)
        logger.info(
            f"  Filas eliminadas por calentamiento: {rows_dropped} "
            f"({rows_dropped / rows_before * 100:.1f}%)"
        )
        
        # Paso 3: Normalizar SOLO los indicadores
        df_normalized = self.normalizer.normalize_columns(
            df_clean,
            columns=self.active_indicators
        )

        # Eliminar el calentamiento de la normalizacion: las primeras barras
        # no tienen historia suficiente para media/std y quedan como NaN.
        # Antes se rellenaban con 0, lo que alimentaba a la red con entradas
        # planas e inventadas al inicio del primer train.
        rows_before_norm = len(df_normalized)
        df_normalized = df_normalized.dropna(subset=self.active_indicators)
        logger.info(
            f"  Filas eliminadas por calentamiento de normalizacion: "
            f"{rows_before_norm - len(df_normalized)}"
        )
        logger.info(
            f"Paso 3/3: Indicadores normalizados. "
            f"Shape final: {df_normalized.shape}"
        )
        
        self._processed_data = df_normalized
        return self._processed_data

    def process_live_data(
        self,
        raw_df: pd.DataFrame,
        required_indicators: Optional[list[str]] = None
    ) -> pd.DataFrame:
        """
        Procesa una ventana acotada de velas recientes (ej. 260 velas diarias)
        para ejecucion en vivo sin Data Leakage.

        DIFERENCIA CON `get_processed_data()`:
        En entrenamiento historico disponemos de miles de barras y podemos hacer
        `dropna()` global antes de normalizar. En vivo solicitamos unicamente las
        ultimas ~260 velas cerradas para minimizar latencia:
        - La SMA200 consume las primeras 199 barras de calentamiento, dejando 61
          valores validos para `distancia_SMA200` (y 41 para `Pendiente_SMA200`).
        - Los indicadores rapidos (`retorno_1d`, `MACD`, `Momentum`, etc.) tienen
          entre 241 y 260 valores validos, por lo que `RollingNormalizer(window=252)`
          aplica el Z-score exacto de 252 barras sobre ellos sin perder historial.
        - Cada indicador se normaliza causalmente sobre su propia serie (solo
          pasado `<= t`, cero Data Leakage) antes de filtrar `NaN` en las columnas
          requeridas por el genoma campeon.

        Parameters
        ----------
        raw_df : pd.DataFrame
            DataFrame OHLCV cronologico con velas cerradas (minimo 220 barras si se
            usan indicadores de 200 periodos; recomendado 260 barras).
        required_indicators : list[str], opcional
            Subconjunto de indicadores activos del genoma que no pueden ser NaN
            en la vela evaluada. Si es None, exige todos `self.active_indicators`.

        Returns
        -------
        pd.DataFrame
            DataFrame con OHLCV reales + indicadores normalizados causalmente.
        """
        if raw_df is None or raw_df.empty:
            raise ValueError("raw_df esta vacio; no se pueden procesar datos en vivo.")

        df_sorted = raw_df.sort_index().copy()
        df_with_indicators = self.indicator_engine.compute(df_sorted)

        # Normalizador causal con ventana identica (252 en 1d), permitiendo que
        # indicadores lentos como SMA200 (61 puntos post-calentamiento en 260 velas)
        # tambien produzcan su Z-score causal sin vaciar el DataFrame, mientras que
        # todos los indicadores con >=252 puntos usan las 252 barras completas.
        live_min_periods = min(20, max(2, self.normalizer.window // 4))
        live_normalizer = RollingNormalizer(
            window=self.normalizer.window,
            method=self.normalizer.method,
            clip_value=self.normalizer.clip_value,
            squash=self.normalizer.squash,
            tanh_scale=self.normalizer.tanh_scale,
            min_periods=live_min_periods,
        )

        df_normalized = live_normalizer.normalize_columns(
            df_with_indicators,
            columns=self.active_indicators,
        )

        subset_cols = (
            [c for c in required_indicators if c in df_normalized.columns]
            if required_indicators
            else list(self.active_indicators)
        )
        df_ready = df_normalized.dropna(subset=subset_cols)
        if df_ready.empty:
            raise ValueError(
                f"Datos insuficientes ({len(raw_df)} barras) para calcular y normalizar "
                f"los indicadores requeridos: {subset_cols}. Solicita al menos 260 velas."
            )

        logger.info(
            f"Datos en vivo procesados: {len(raw_df)} velas crudas -> "
            f"{len(df_ready)} filas listas (ultima vela cerrada: {df_ready.index[-1]})"
        )
        return df_ready

    def get_live_data(
        self,
        bitso_client,
        lookback_bars: int = 260,
        warmup_extra_bars: int = 0,
        exclude_open_candle: bool = True,
        required_indicators: Optional[list[str]] = None,
        now_utc=None,
    ) -> pd.DataFrame:
        """
        Solicita unicamente las ultimas `lookback_bars` velas cerradas a Bitso
        mediante `bitso_client` y ejecuta calculo de indicadores + Z-score rolling.

        Parameters
        ----------
        bitso_client : BitsoRESTClient
            Cliente REST configurado para Bitso.
        lookback_bars : int
            Cantidad de velas cerradas a solicitar (default: 260).
        warmup_extra_bars : int
            Barras adicionales opcionales si se desea mayor historial post-SMA200.
        exclude_open_candle : bool
            Si True, excluye la vela diaria aun abierta para evitar señales intradia falsas.
        required_indicators : list[str], opcional
            Indicadores activos en el genoma campeon.
        now_utc : datetime, opcional
            Timestamp UTC de referencia.

        Returns
        -------
        pd.DataFrame
            DataFrame listo para evaluar la ultima fila cerrada con la red neuronal.
        """
        total_bars = max(1, int(lookback_bars) + max(0, int(warmup_extra_bars)))
        book = bitso_client.symbol_to_book(self.symbol)
        time_bucket = 86400 if self.interval in ("1d", "1D") else 3600

        raw_live = bitso_client.fetch_ohlcv(
            book=book,
            limit=total_bars,
            time_bucket=time_bucket,
            exclude_open_candle=exclude_open_candle,
            now_utc=now_utc,
        )
        self._raw_data = raw_live
        self._processed_data = self.process_live_data(
            raw_live,
            required_indicators=required_indicators,
        )
        return self._processed_data
    
    def walk_forward(
        self,
        train_years: int = 3,
        val_years: int = 1,
        test_years: int = 1,
        step_years: int = 1,
        mode: str = "sliding"
    ) -> Iterator[WalkForwardWindow]:
        """
        Genera ventanas Walk-Forward sobre los datos procesados.
        
        Este es el metodo PRINCIPAL del pipeline. Combina todo:
        descarga, indicadores, normalizacion y Walk-Forward.
        
        Parameters
        ----------
        train_years, val_years, test_years : int
            Duracion de cada segmento en anos
        step_years : int
            Cuanto avanza la ventana entre iteraciones
        mode : str
            "sliding" o "expanding"
            
        Yields
        ------
        WalkForwardWindow
            Cada ventana con datos de train, val, test procesados
        """
        # Obtener datos procesados
        df = self.get_processed_data()
        
        # Crear el splitter segun el intervalo
        if self.interval in ("1d", "1D"):
            splitter = create_daily_splitter(
                train_years=train_years,
                val_years=val_years,
                test_years=test_years,
                step_years=step_years,
                mode=mode
            )
        elif self.interval in ("1h", "60m"):
            # Para datos horarios, convertir anos a meses
            splitter = create_hourly_splitter(
                train_years=train_years,
                val_months=val_years * 12,
                test_months=test_years * 12,
                step_months=step_years * 12,
                mode=mode
            )
        else:
            raise ValueError(
                f"Intervalo '{self.interval}' no soportado para Walk-Forward. "
                f"Usa '1d' o '1h'."
            )
        
        # Informacion sobre las ventanas
        n_windows = splitter.count_windows(len(df))
        logger.info(
            f"Walk-Forward: {n_windows} ventanas | "
            f"mode={mode} | train={train_years}y, val={val_years}y, test={test_years}y"
        )
        
        if n_windows == 0:
            logger.warning(
                f"No hay suficientes datos para Walk-Forward con la configuracion actual. "
                f"Datos disponibles: {len(df)} barras. "
                f"Intenta reducir los periodos de train/val/test."
            )
        
        yield from splitter.split(df)
    
    def get_indicator_names(self) -> list[str]:
        """Retorna los nombres de los indicadores activos."""
        return list(self.active_indicators)
    
    def get_all_indicator_names(self) -> list[str]:
        """Retorna los nombres de TODOS los indicadores disponibles."""
        return list(INDICATOR_REGISTRY.keys())
    
    def info(self) -> dict:
        """
        Retorna informacion completa del pipeline.
        
        Returns
        -------
        dict
            Diccionario con toda la configuracion y estado
        """
        result = {
            "symbol": self.symbol,
            "interval": self.interval,
            "start": self.start,
            "end": self.end,
            "active_indicators": self.active_indicators,
            "num_indicators": len(self.active_indicators),
            "normalization_method": self.normalizer.method,
            "normalization_window": self.normalizer.window,
        }
        
        if self._raw_data is not None:
            result["raw_bars"] = len(self._raw_data)
            result["raw_date_range"] = (
                f"{self._raw_data.index[0]} to {self._raw_data.index[-1]}"
            )
        
        if self._processed_data is not None:
            result["processed_bars"] = len(self._processed_data)
            result["processed_date_range"] = (
                f"{self._processed_data.index[0]} to "
                f"{self._processed_data.index[-1]}"
            )
        
        return result
    
    def reset_cache(self):
        """Limpia el cache interno para forzar re-procesamiento."""
        self._processed_data = None
        self._raw_data = None
        logger.info("Cache del pipeline reseteado")


# =================================================================
# DEMOSTRACION
# =================================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )
    
    print("=" * 70)
    print("DATA PIPELINE - Demostracion Completa")
    print("=" * 70)
    
    # Crear pipeline para BTC-MXN diario
    pipeline = DataPipeline(
        symbol="BTC-MXN",
        start="2018-01-01",
        interval="1d"
    )
    
    # Mostrar info
    print("\nConfiguracion del pipeline:")
    for key, value in pipeline.info().items():
        print(f"  {key}: {value}")
    
    # Obtener datos procesados
    print("\n" + "-" * 70)
    print("DATOS PROCESADOS")
    print("-" * 70)
    
    df = pipeline.get_processed_data()
    print(f"\nShape: {df.shape}")
    print(f"Columnas: {list(df.columns)}")
    print(f"\nPrimeras filas (indicadores normalizados):")
    indicator_cols = pipeline.get_indicator_names()
    print(df[indicator_cols].head())
    
    # Verificar normalizacion
    print(f"\nRango de valores por indicador:")
    for col in indicator_cols:
        print(f"  {col:>20}: [{df[col].min():+.4f}, {df[col].max():+.4f}]  "
              f"mean={df[col].mean():+.4f}")
    
    # Walk-Forward
    print("\n" + "-" * 70)
    print("WALK-FORWARD")
    print("-" * 70)
    
    for window in pipeline.walk_forward(
        train_years=3,
        val_years=1,
        test_years=1,
        mode="sliding"
    ):
        print(f"\n  {window}")
        print(f"    Train shape: {window.train.shape}")
        print(f"    Val shape:   {window.val.shape}")
        print(f"    Test shape:  {window.test.shape}")
    
    print("\n" + "=" * 70)
    print("Pipeline completado exitosamente!")
    print("=" * 70)
