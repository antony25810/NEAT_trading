"""
============================================================
downloader.py — Módulo de Descarga de Datos OHLCV
============================================================

¿QUÉ HACE ESTE MÓDULO?
-----------------------
Descarga datos históricos de precios (OHLCV) desde Yahoo Finance
y los guarda localmente para no tener que re-descargar cada vez.

¿QUÉ ES OHLCV?
---------------
Cada "barra" o "vela" de precio contiene 5 valores fundamentales:
  - Open (Apertura):   Precio al inicio del periodo
  - High (Máximo):     Precio más alto durante el periodo
  - Low (Mínimo):      Precio más bajo durante el periodo
  - Close (Cierre):    Precio al final del periodo
  - Volume (Volumen):  Cantidad de activo negociada durante el periodo

Ejemplo para BTC-MXN en una barra diaria:
  Fecha       Open        High        Low         Close       Volume
  2024-01-15  728,000     745,000     720,000     740,000     1,234.56

¿POR QUÉ CACHE LOCAL?
----------------------
- yfinance tiene límites de requests por hora.
- Descargar datos cada vez que corremos el programa es lento.
- Guardamos los datos en archivos CSV/Parquet una vez y los leemos del disco.
- Parquet es un formato binario comprimido, más rápido y compacto que CSV.

¿POR QUÉ YFINANCE?
-------------------
- Es gratuito (no necesita API key).
- Tiene datos de criptomonedas incluyendo pares en MXN.
- Es la opción más accesible para un proyecto de investigación.
- Limitación: los datos horarios sólo están disponibles para ~730 días.
"""

import os
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

# Configuramos el logger para este módulo.
# Un logger es como un sistema de "print" profesional: permite controlar
# qué mensajes se muestran (info, warning, error) y dónde se guardan.
logger = logging.getLogger(__name__)


# =================================================================
# Constante: Directorio por defecto donde se guardan los datos
# =================================================================
# Path(__file__) obtiene la ruta de ESTE archivo (downloader.py)
# .resolve() la convierte en ruta absoluta
# .parent sube un nivel (de src/data/ a src/)
# .parent.parent sube dos niveles (de src/ a Neuroevolucion/)
# / "data_cache" agrega la carpeta final
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data_cache"


class DataDownloader:
    """
    Descargador de datos OHLCV con cache local.
    
    ¿POR QUÉ UNA CLASE Y NO FUNCIONES SUELTAS?
    --------------------------------------------
    Porque necesitamos mantener "estado" entre llamadas:
    - El directorio de cache (dónde guardar archivos).
    - Configuración compartida.
    
    Una clase agrupa datos + comportamiento relacionado.
    Es como un "objeto" que sabe cómo descargar y cachear datos.
    
    Uso básico:
    -----------
    >>> downloader = DataDownloader()
    >>> df = downloader.download("BTC-MXN", interval="1d")
    >>> print(df.head())
    """
    
    def __init__(self, cache_dir: Optional[str | Path] = None):
        """
        Inicializa el downloader.
        
        Parameters
        ----------
        cache_dir : str o Path, opcional
            Directorio donde guardar los datos descargados.
            Si no se especifica, usa data_cache/ en la raíz del proyecto.
        """
        # Si el usuario no especifica directorio, usamos el default
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        
        # Crear el directorio si no existe
        # exist_ok=True significa: si ya existe, no dar error
        # parents=True significa: crear también los directorios intermedios
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DataDownloader inicializado. Cache en: {self.cache_dir}")
    
    def _get_cache_path(self, symbol: str, interval: str) -> Path:
        """
        Genera la ruta del archivo de cache para un símbolo e intervalo.
        
        ¿POR QUÉ EL UNDERSCORE AL INICIO?
        -----------------------------------
        En Python, un método que empieza con _ es "privado por convención".
        No es estrictamente privado (Python no tiene privacidad real),
        pero indica: "este método es interno, no lo llames desde fuera".
        
        Parameters
        ----------
        symbol : str
            Símbolo del activo (ej: "BTC-MXN")
        interval : str
            Intervalo temporal (ej: "1d", "1h")
            
        Returns
        -------
        Path
            Ruta al archivo parquet de cache
        """
        # Reemplazamos caracteres problemáticos en nombres de archivo
        # "BTC-MXN" → "BTC_MXN"
        safe_symbol = symbol.replace("-", "_").replace("/", "_")
        filename = f"{safe_symbol}_{interval}.parquet"
        return self.cache_dir / filename
    
    def _is_cache_valid(
        self,
        cache_path: Path,
        max_age_hours: int = 24
    ) -> bool:
        """
        Verifica si el archivo de cache existe y no está demasiado viejo.
        
        ¿POR QUÉ VERIFICAR LA EDAD DEL CACHE?
        ---------------------------------------
        Si descargaste datos ayer y hoy hay datos nuevos, el cache está
        desactualizado. Definimos un "tiempo máximo de vida" (max_age_hours).
        Si el archivo es más viejo que eso, lo consideramos inválido
        y re-descargamos.
        
        Parameters
        ----------
        cache_path : Path
            Ruta al archivo de cache
        max_age_hours : int
            Máxima antigüedad permitida en horas
            
        Returns
        -------
        bool
            True si el cache es válido, False si hay que re-descargar
        """
        if not cache_path.exists():
            return False
        
        # Obtener la fecha de última modificación del archivo
        file_mod_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age = datetime.now() - file_mod_time
        
        # ¿Es más viejo que max_age_hours?
        is_valid = age < timedelta(hours=max_age_hours)
        
        if is_valid:
            logger.info(
                f"Cache válido ({age.total_seconds() / 3600:.1f}h de antigüedad): "
                f"{cache_path.name}"
            )
        else:
            logger.info(
                f"Cache expirado ({age.total_seconds() / 3600:.1f}h de antigüedad): "
                f"{cache_path.name}"
            )
        
        return is_valid
    
    def download(
        self,
        symbol: str,
        start: str = "2018-01-01",
        end: Optional[str] = None,
        interval: str = "1d",
        use_cache: bool = True,
        cache_max_age_hours: int = 24
    ) -> pd.DataFrame:
        """
        Descarga datos OHLCV para un símbolo dado.
        
        ¿CÓMO FUNCIONA EL FLUJO?
        -------------------------
        1. ¿Hay cache válido? → Sí → Leer del disco
                               → No → Descargar de Yahoo Finance
        2. Limpiar datos (eliminar filas vacías, verificar tipos)
        3. Guardar en cache para la próxima vez
        4. Retornar DataFrame de Pandas
        
        Parameters
        ----------
        symbol : str
            Símbolo del activo. Ejemplos:
            - "BTC-MXN" para Bitcoin en pesos mexicanos
            - "ETH-MXN" para Ethereum en pesos mexicanos
            - "SPY" para el ETF del S&P 500
        start : str
            Fecha de inicio en formato "YYYY-MM-DD"
        end : str, opcional
            Fecha de fin. Si no se especifica, usa la fecha actual.
        interval : str
            Intervalo temporal. Opciones:
            - "1d" → barras diarias (un dato por día)
            - "1h" → barras horarias (un dato por hora)
            - "5m" → barras de 5 minutos
            NOTA: Yahoo Finance solo tiene ~730 días de datos horarios.
        use_cache : bool
            Si True, intenta leer del cache antes de descargar.
        cache_max_age_hours : int
            Máxima antigüedad del cache en horas.
            
        Returns
        -------
        pd.DataFrame
            DataFrame con columnas: Open, High, Low, Close, Volume
            Índice: DatetimeIndex con la fecha/hora de cada barra
            
        Raises
        ------
        ValueError
            Si no se pudieron descargar datos (símbolo inválido, sin internet, etc.)
        """
        # Si no se especifica fecha fin, usar hoy
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")
        
        logger.info(
            f"Solicitando datos: {symbol} | {interval} | {start} -> {end}"
        )
        
        # --- Paso 1: Verificar cache ---
        cache_path = self._get_cache_path(symbol, interval)
        
        if use_cache and self._is_cache_valid(cache_path, cache_max_age_hours):
            logger.info(f"Leyendo datos del cache: {cache_path}")
            df = pd.read_parquet(cache_path)
            
            # Filtrar por el rango de fechas solicitado
            # (el cache puede tener más datos de los que necesitamos)
            df = self._filter_date_range(df, start, end)
            logger.info(f"Datos del cache: {len(df)} barras")
            return df
        
        # --- Paso 2: Descargar de Yahoo Finance ---
        logger.info(f"Descargando de Yahoo Finance: {symbol}")
        
        try:
            # yf.download() es la función principal de yfinance
            # auto_adjust=True: ajusta precios por splits y dividendos automáticamente
            # progress=False: no mostrar barra de progreso de yfinance (usamos nuestro logger)
            
            # Para datos horarios, Yahoo tiene un límite de ~730 días
            # Ajustamos la fecha de inicio si es necesario
            effective_start = start
            if interval in ("1h", "60m"):
                max_start = (datetime.now() - timedelta(days=729)).strftime("%Y-%m-%d")
                if start < max_start:
                    logger.warning(
                        f"Yahoo Finance solo tiene ~730 días de datos horarios. "
                        f"Ajustando start de {start} a {max_start}"
                    )
                    effective_start = max_start
            
            df = yf.download(
                tickers=symbol,
                start=effective_start,
                end=end,
                interval=interval,
                auto_adjust=True,
                progress=False
            )
        except Exception as e:
            if use_cache and cache_path.exists():
                logger.warning(
                    f"Fallo la descarga para {symbol} ({e}), usando cache existente: {cache_path}"
                )
                df = pd.read_parquet(cache_path)
                return self._filter_date_range(df, start, end)
            raise ValueError(
                f"Error descargando datos para {symbol}: {e}"
            ) from e
        
        # --- Paso 3: Validar datos ---
        if df is None or df.empty:
            if use_cache and cache_path.exists():
                logger.warning(
                    f"No se obtuvieron datos para {symbol}, usando cache existente: {cache_path}"
                )
                df = pd.read_parquet(cache_path)
                return self._filter_date_range(df, start, end)
            raise ValueError(
                f"No se obtuvieron datos para {symbol} "
                f"({effective_start} -> {end}, interval={interval}). "
                f"Verifica que el simbolo sea valido y tengas conexion a internet."
            )
        
        # Limpiar los datos
        df = self._clean_data(df, symbol)
        
        logger.info(
            f"Descargados {len(df)} barras para {symbol} "
            f"({df.index[0]} -> {df.index[-1]})"
        )
        
        # --- Paso 4: Guardar en cache ---
        if use_cache:
            df.to_parquet(cache_path)
            logger.info(f"Datos guardados en cache: {cache_path}")
        
        return df
    
    def _clean_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Limpia y estandariza los datos descargados.
        
        ¿POR QUÉ LIMPIAR LOS DATOS?
        ----------------------------
        Los datos financieros del mundo real son "sucios":
        - Pueden tener filas con valores NaN (datos faltantes).
        - yfinance a veces devuelve un MultiIndex si descargas múltiples tickers.
        - El volumen puede ser 0 en algunos períodos (mercados cerrados).
        
        Es FUNDAMENTAL limpiar antes de calcular indicadores, porque un solo
        NaN puede propagar errores a todos los cálculos posteriores.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame crudo de yfinance
        symbol : str
            Símbolo del activo (para logging)
            
        Returns
        -------
        pd.DataFrame
            DataFrame limpio con columnas estándar
        """
        # yfinance a veces devuelve un MultiIndex en las columnas
        # cuando descargas un solo ticker. Lo aplanamos.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        
        # Asegurar que las columnas tienen los nombres esperados
        expected_columns = ["Open", "High", "Low", "Close", "Volume"]
        
        # Verificar que las columnas existan
        missing = [col for col in expected_columns if col not in df.columns]
        if missing:
            raise ValueError(
                f"Columnas faltantes en datos de {symbol}: {missing}. "
                f"Columnas disponibles: {list(df.columns)}"
            )
        
        # Quedarnos solo con las columnas OHLCV
        df = df[expected_columns].copy()
        
        # Eliminar filas donde TODAS las columnas de precio son NaN
        # (días festivos, etc.)
        price_cols = ["Open", "High", "Low", "Close"]
        df = df.dropna(subset=price_cols, how="all")
        
        # Para las filas restantes, llenar NaN con el valor anterior
        # Esto se llama "forward fill" — si hoy no hay dato, usa el de ayer
        # Es la convención estándar en datos financieros
        rows_before = len(df)
        df[price_cols] = df[price_cols].ffill()
        
        # Volumen: llenar NaN con 0 (si no se negoció, el volumen es 0)
        df["Volume"] = df["Volume"].fillna(0)
        
        # Eliminar cualquier fila restante con NaN (las primeras filas
        # que no pueden ser forward-filled)
        df = df.dropna()
        
        rows_removed = rows_before - len(df)
        if rows_removed > 0:
            logger.info(f"Limpieza: {rows_removed} filas removidas de {symbol}")
        
        # Asegurar que el índice sea DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Ordenar por fecha (por si acaso)
        df = df.sort_index()
        
        # Asegurar tipos numéricos
        for col in expected_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        return df
    
    def _filter_date_range(
        self,
        df: pd.DataFrame,
        start: str,
        end: Optional[str]
    ) -> pd.DataFrame:
        """
        Filtra un DataFrame por rango de fechas.
        
        Usa la sintaxis de slicing de Pandas con fechas, que es muy elegante:
        df[start:end] filtra todas las filas cuyo índice está entre start y end.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con DatetimeIndex
        start : str
            Fecha inicio "YYYY-MM-DD"
        end : str o None
            Fecha fin "YYYY-MM-DD" (None = hasta el último dato)
            
        Returns
        -------
        pd.DataFrame
            DataFrame filtrado
        """
        return df.loc[start:end]
    
    def download_multiple(
        self,
        symbols: list[str],
        **kwargs
    ) -> dict[str, pd.DataFrame]:
        """
        Descarga datos para múltiples símbolos.
        
        Útil para descargar BTC-MXN y ETH-MXN de una sola llamada.
        
        Parameters
        ----------
        symbols : list[str]
            Lista de símbolos (ej: ["BTC-MXN", "ETH-MXN"])
        **kwargs
            Argumentos adicionales pasados a download()
            
        Returns
        -------
        dict[str, pd.DataFrame]
            Diccionario {símbolo: DataFrame}
        """
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.download(symbol, **kwargs)
                logger.info(f"[OK] {symbol}: {len(results[symbol])} barras")
            except ValueError as e:
                logger.error(f"[FAIL] {symbol}: {e}")
        
        return results
    
    def get_info(self, symbol: str) -> dict:
        """
        Obtiene información básica sobre un activo.
        
        Útil para verificar que un símbolo es válido y ver detalles
        como el nombre, la moneda, el exchange, etc.
        
        Parameters
        ----------
        symbol : str
            Símbolo del activo
            
        Returns
        -------
        dict
            Diccionario con información del activo
        """
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Retornar solo los campos más útiles
        useful_fields = [
            "shortName", "longName", "currency", "exchange",
            "marketCap", "quoteType", "regularMarketPrice"
        ]
        
        return {
            field: info.get(field, "N/A")
            for field in useful_fields
        }
    
    def list_cached_files(self) -> list[dict]:
        """
        Lista todos los archivos de datos en el cache.
        
        Returns
        -------
        list[dict]
            Lista de diccionarios con info de cada archivo cacheado
        """
        cached = []
        for filepath in self.cache_dir.glob("*.parquet"):
            stat = filepath.stat()
            cached.append({
                "filename": filepath.name,
                "size_mb": stat.st_size / (1024 * 1024),
                "last_modified": datetime.fromtimestamp(stat.st_mtime).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "path": str(filepath)
            })
        
        return cached

    # =================================================================
    # CONVERSION A MXN
    # =================================================================
    # Yahoo Finance NO tiene pares crypto/MXN directos.
    # Pero SI tiene:
    #   - BTC-USD (Bitcoin en dolares)
    #   - ETH-USD (Ethereum en dolares)
    #   - MXN=X (tipo de cambio USD/MXN)
    #
    # Entonces calculamos:
    #   BTC_MXN = BTC_USD * USD_MXN
    #
    # Esto es exactamente lo que hacen los exchanges como Bitso:
    # toman el precio internacional en USD y lo multiplican por
    # el tipo de cambio.
    # =================================================================

    def download_mxn(
        self,
        crypto: str,
        start: str = "2018-01-01",
        end: Optional[str] = None,
        interval: str = "1d",
        use_cache: bool = True,
        cache_max_age_hours: int = 24
    ) -> pd.DataFrame:
        """
        Descarga datos de un par crypto/MXN convirtiendo desde USD.

        COMO FUNCIONA:
        1. Descarga CRYPTO-USD (ej: BTC-USD)
        2. Descarga MXN=X (tipo de cambio USD/MXN)
        3. Multiplica precios OHLC por el tipo de cambio
        4. Volumen se mantiene en unidades de crypto (no se convierte)

        Parameters
        ----------
        crypto : str
            Simbolo de la crypto: "BTC", "ETH", etc.
        start : str
            Fecha de inicio
        end : str, opcional
            Fecha de fin
        interval : str
            "1d" o "1h"
        use_cache : bool
            Usar cache local
        cache_max_age_hours : int
            Edad maxima del cache

        Returns
        -------
        pd.DataFrame
            DataFrame OHLCV con precios en MXN
        """
        # Simbolo sintetico para el cache
        mxn_symbol = f"{crypto}_MXN"
        cache_path = self._get_cache_path(
            f"{mxn_symbol}_CALENDAR_V2", interval
        )

        # Verificar cache del par sintetico
        if use_cache and self._is_cache_valid(cache_path, cache_max_age_hours):
            logger.info(f"Leyendo {mxn_symbol} del cache")
            df = pd.read_parquet(cache_path)
            # Filtrar SIEMPRE por start; antes solo se filtraba cuando habia
            # fecha fin, y un cache descargado con otro start se usaba completo.
            return self._filter_date_range(df, start, end)

        # Paso 1: Descargar crypto en USD
        usd_symbol = f"{crypto}-USD"
        logger.info(f"Descargando {usd_symbol} para conversion a MXN")
        try:
            df_usd = self.download(
                symbol=usd_symbol,
                start=start,
                end=end,
                interval=interval,
                use_cache=use_cache,
                cache_max_age_hours=cache_max_age_hours
            )

            # Paso 2: Descargar tipo de cambio USD/MXN
            logger.info("Descargando tipo de cambio USD/MXN (MXN=X)")
            df_fx = self.download(
                symbol="MXN=X",
                start=start,
                end=end,
                interval=interval,
                use_cache=use_cache,
                cache_max_age_hours=cache_max_age_hours
            )
        except Exception as e:
            if use_cache and cache_path.exists():
                logger.warning(
                    f"Fallo la actualizacion de {mxn_symbol} ({e}), usando cache existente: {cache_path}"
                )
                df = pd.read_parquet(cache_path)
                return self._filter_date_range(df, start, end)
            raise

        # Paso 3: Mantener el calendario 24/7 de crypto.
        df_usd = df_usd.copy()
        df_fx = df_fx.copy()

        # En fines de semana se usa el ultimo FX conocido, nunca uno futuro.
        fx_close = df_fx["Close"].reindex(df_usd.index).ffill()
        valid_fx = fx_close.notna()
        df_usd = df_usd.loc[valid_fx]
        fx_close = fx_close.loc[valid_fx]
        logger.info(
            f"Calendario crypto conservado: {len(df_usd)} "
            f"(crypto: {len(df_usd)}, FX: {len(df_fx)})"
        )

        if len(df_usd) == 0:
            raise ValueError(
                f"No hay tipo de cambio previo disponible para {usd_symbol}. "
                f"Verifica que ambos tengan datos en el mismo rango."
            )

        # Paso 4: Convertir OHLC a MXN
        # Precio_MXN = Precio_USD * Tipo_Cambio
        df_mxn = df_usd.copy()
        for col in ["Open", "High", "Low", "Close"]:
            df_mxn[col] = df_usd[col] * fx_close

        # Volumen se mantiene en unidades de crypto
        # (no tiene sentido convertir volumen a pesos)

        logger.info(
            f"Conversion completada: {mxn_symbol} | "
            f"{len(df_mxn)} barras | "
            f"Ultimo precio: ${df_mxn['Close'].iloc[-1]:,.2f} MXN"
        )

        # Guardar en cache
        if use_cache:
            df_mxn.to_parquet(cache_path)
            logger.info(f"Cache guardado: {cache_path}")

        return df_mxn


# =================================================================
# BLOQUE if __name__ == "__main__"
# =================================================================
# Este bloque SOLO se ejecuta si corres este archivo directamente:
#   python src/data/downloader.py
# NO se ejecuta si importas el módulo desde otro archivo:
#   from src.data.downloader import DataDownloader
#
# Es perfecto para poner código de prueba/demostración.
# =================================================================
if __name__ == "__main__":
    # Configurar logging para ver los mensajes en consola
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )
    
    # Crear instancia del downloader
    downloader = DataDownloader()
    
    # Descargar BTC-MXN diario
    print("\n" + "=" * 60)
    print("Descargando BTC-MXN (diario)")
    print("=" * 60)
    btc_daily = downloader.download(
        symbol="BTC-MXN",
        start="2020-01-01",
        interval="1d"
    )
    print(f"\nShape: {btc_daily.shape}")
    print(f"Columnas: {list(btc_daily.columns)}")
    print(f"\nPrimeras 5 filas:")
    print(btc_daily.head())
    print(f"\nÚltimas 5 filas:")
    print(btc_daily.tail())
    print(f"\nEstadísticas básicas:")
    print(btc_daily.describe())
    
    # Descargar ETH-MXN diario
    print("\n" + "=" * 60)
    print("Descargando ETH-MXN (diario)")
    print("=" * 60)
    eth_daily = downloader.download(
        symbol="ETH-MXN",
        start="2020-01-01",
        interval="1d"
    )
    print(f"\nShape: {eth_daily.shape}")
    print(f"Primeras 5 filas:")
    print(eth_daily.head())
    
    # Listar archivos cacheados
    print("\n" + "=" * 60)
    print("Archivos en cache:")
    print("=" * 60)
    for cached in downloader.list_cached_files():
        print(f"  {cached['filename']} ({cached['size_mb']:.2f} MB) - {cached['last_modified']}")
