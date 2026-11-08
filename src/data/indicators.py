"""
============================================================
indicators.py - Indicadores Tecnicos para Trading
============================================================

QUE ES UN INDICADOR TECNICO?
-----------------------------
Un indicador tecnico es una formula matematica que se aplica a los
datos de precio/volumen para extraer informacion sobre el estado
del mercado. Son las "variables" que la red neuronal usara para
tomar decisiones.

Piensa en ellos como "sensores" que miden diferentes aspectos:
- TENDENCIA: direccion general del precio? (subiendo/bajando)
- MOMENTUM: que tan rapido se mueve? (acelerando/frenando)
- VOLATILIDAD: que tan bruscos son los movimientos?
- VOLUMEN: cuanta gente esta participando?
- REGIMEN: tipo de mercado actual? (tendencia fuerte/mercado lateral)

POR QUE TANTOS INDICADORES?
----------------------------
Ningun indicador es perfecto. Cada uno captura un aspecto diferente
del mercado. La EVOLUCION decidira cuales son utiles para cada
estrategia activando/desactivando indicadores via indicator_mask.

Ejemplo: En un mercado con tendencia fuerte, las SMAs son utiles.
En un mercado lateral, el RSI funciona mejor.
La red neuronal NO sabe esto de antemano - lo DESCUBRE evolucionando.

NOTA SOBRE DATA LEAKAGE
------------------------
Todos los indicadores aqui se calculan usando SOLO datos pasados.
Por ejemplo, SMA(20) del dia T usa los precios de T-19 hasta T.
Nunca usa el precio de T+1. Esto es fundamental para evitar
"ver el futuro" durante el backtesting.
"""

import numpy as np
import pandas as pd
from typing import Optional


# =================================================================
# CATALOGO DE INDICADORES
# =================================================================
# Este diccionario registra TODOS los indicadores disponibles.
# Cada entrada tiene: nombre, funcion, categoria, descripcion.
# El genoma de cada individuo tendra un "mask" sobre estos indicadores
# para activar/desactivar los que considere utiles.
#
# POR QUE UN REGISTRO?
# Porque el genoma necesita saber cuantos indicadores hay y cuales
# son, para poder evolucionar su indicator_mask. Si agregas un nuevo
# indicador, solo necesitas registrarlo aqui y automaticamente estara
# disponible para la evolucion.
# =================================================================

INDICATOR_REGISTRY = {}


def register_indicator(name: str, category: str, description: str):
    """
    Decorador que registra un indicador en el catalogo global.
    
    QUE ES UN DECORADOR?
    --------------------
    Es una funcion que "envuelve" otra funcion para agregarle
    funcionalidad. Cuando escribes:
    
        @register_indicator("RSI", "momentum", "Relative Strength Index")
        def compute_rsi(df, period=14):
            ...
    
    Es equivalente a:
        def compute_rsi(df, period=14):
            ...
        compute_rsi = register_indicator("RSI", "momentum", "...")(compute_rsi)
    
    El decorador registra la funcion en INDICATOR_REGISTRY y la retorna
    sin modificarla.
    """
    def decorator(func):
        INDICATOR_REGISTRY[name] = {
            "function": func,
            "category": category,
            "description": description,
            "name": name
        }
        return func
    return decorator


# =================================================================
# RETORNOS
# =================================================================
# Los retornos miden CUANTO cambio el precio en un periodo.
# Son la base de casi todo en finanzas.
#
# Formula: retorno = (precio_hoy - precio_hace_N_dias) / precio_hace_N_dias
#
# Ejemplo: Si BTC estaba a $1,000,000 hace 5 dias y hoy esta a $1,050,000
#          retorno_5d = (1,050,000 - 1,000,000) / 1,000,000 = 0.05 = 5%
# =================================================================

@register_indicator("retorno_1d", "retornos", "Retorno porcentual de 1 dia")
def compute_return_1d(df: pd.DataFrame) -> pd.Series:
    """
    Calcula el retorno porcentual de 1 dia.
    
    pct_change(1) calcula: (precio[t] - precio[t-1]) / precio[t-1]
    
    Resultado: Serie con valores como 0.02 (subio 2%), -0.01 (bajo 1%)
    """
    return df["Close"].pct_change(1)


@register_indicator("retorno_5d", "retornos", "Retorno porcentual de 5 dias")
def compute_return_5d(df: pd.DataFrame) -> pd.Series:
    """Retorno de 5 dias. Captura tendencia de corto plazo (1 semana)."""
    return df["Close"].pct_change(5)


@register_indicator("retorno_20d", "retornos", "Retorno porcentual de 20 dias")
def compute_return_20d(df: pd.DataFrame) -> pd.Series:
    """Retorno de 20 dias. Captura tendencia de mediano plazo (~1 mes)."""
    return df["Close"].pct_change(20)


# =================================================================
# TENDENCIA - Simple Moving Averages (SMA)
# =================================================================
# La SMA es el PROMEDIO del precio de los ultimos N periodos.
#
# SMA(20) = promedio de los ultimos 20 cierres
#
# POR QUE LA DISTANCIA A LA SMA Y NO LA SMA DIRECTA?
# ---------------------------------------------------
# Porque el precio de BTC puede ser $500,000 o $2,000,000.
# La SMA(20) seria $490,000 o $1,950,000.
# Esos numeros absolutos no le dicen nada a la red neuronal.
#
# En cambio, la DISTANCIA PORCENTUAL es independiente del precio:
# distancia = (precio - SMA) / SMA
# Si el resultado es 0.05, significa que el precio esta 5% POR ENCIMA
# del promedio. Eso SI es informacion util y comparable entre activos.
# =================================================================

@register_indicator(
    "distancia_SMA20", "tendencia",
    "Distancia porcentual del precio a la SMA de 20 periodos"
)
def compute_distance_sma20(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Distancia porcentual del precio al promedio de 20 dias.
    
    > 0: precio POR ENCIMA del promedio (tendencia alcista)
    < 0: precio POR DEBAJO del promedio (tendencia bajista)
    ~ 0: precio CERCA del promedio (sin tendencia clara)
    """
    sma = df["Close"].rolling(window=period).mean()
    return (df["Close"] - sma) / sma


@register_indicator(
    "distancia_SMA50", "tendencia",
    "Distancia porcentual del precio a la SMA de 50 periodos"
)
def compute_distance_sma50(df: pd.DataFrame, period: int = 50) -> pd.Series:
    """Distancia a SMA(50). Tendencia de mediano plazo (~2.5 meses)."""
    sma = df["Close"].rolling(window=period).mean()
    return (df["Close"] - sma) / sma


@register_indicator(
    "distancia_SMA200", "tendencia",
    "Distancia porcentual del precio a la SMA de 200 periodos"
)
def compute_distance_sma200(df: pd.DataFrame, period: int = 200) -> pd.Series:
    """
    Distancia a SMA(200). Tendencia de LARGO plazo (~10 meses).
    
    Este es el indicador mas popular en Wall Street para determinar
    si estamos en un mercado alcista (bull) o bajista (bear):
    - Precio > SMA(200) → Bull market
    - Precio < SMA(200) → Bear market
    """
    sma = df["Close"].rolling(window=period).mean()
    return (df["Close"] - sma) / sma


# =================================================================
# MOMENTUM - RSI, MACD, ROC, Momentum
# =================================================================
# El momentum mide la VELOCIDAD del cambio de precio.
# Un precio que sube rapido tiene mucho momentum alcista.
# Un precio que se desacelera puede estar por cambiar de direccion.
# =================================================================

@register_indicator("RSI", "momentum", "Relative Strength Index (14 periodos)")
def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    RSI - Relative Strength Index (Indice de Fuerza Relativa)
    
    QUE MIDE?
    ---------
    La FUERZA de los movimientos alcistas vs bajistas recientes.
    Rango: [0, 100]
    
    COMO SE INTERPRETA?
    -------------------
    RSI > 70: "Sobrecomprado" - el precio subio mucho y podria bajar
    RSI < 30: "Sobrevendido"  - el precio bajo mucho y podria subir
    RSI ~ 50: Sin sesgo claro
    
    COMO SE CALCULA?
    ----------------
    1. Calcular las subidas y bajadas diarias
    2. Promediar las subidas y bajadas de los ultimos N dias (EMA)
    3. RS = promedio_subidas / promedio_bajadas
    4. RSI = 100 - (100 / (1 + RS))
    
    POR QUE DIVIDIMOS POR 100 AL FINAL?
    Para que la red neuronal reciba valores en [0, 1] (despues de normalizar).
    """
    # Calcular cambios diarios de precio
    delta = df["Close"].diff()
    
    # Separar subidas (gains) y bajadas (losses)
    # clip: limitar valores. lower=0 → negativos se vuelven 0
    gain = delta.clip(lower=0)    # Solo subidas, bajadas = 0
    loss = (-delta).clip(lower=0)  # Solo bajadas (como positivo), subidas = 0
    
    # Promedio exponencial (EMA) de ganancias y perdidas
    # EMA da mas peso a datos recientes que el promedio simple
    # adjust=False: usa la formula recursiva (mas eficiente)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    # Relative Strength
    rs = avg_gain / avg_loss
    
    # RSI formula
    rsi = 100 - (100 / (1 + rs))
    
    # Devolver en escala 0-100 (se normalizara despues)
    return rsi


@register_indicator("MACD", "momentum", "MACD line (12, 26)")
def compute_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26
) -> pd.Series:
    """
    MACD - Moving Average Convergence Divergence
    
    QUE MIDE?
    ---------
    La DIFERENCIA entre dos promedios exponenciales (EMA):
    MACD = EMA(12) - EMA(26)
    
    INTUICION:
    Si el promedio rapido (12 dias) esta POR ENCIMA del lento (26 dias),
    significa que el precio esta subiendo (MACD positivo).
    Si esta por debajo, el precio esta bajando (MACD negativo).
    
    POR QUE ES UTIL?
    Captura el CAMBIO de tendencia. Cuando el MACD cruza de negativo
    a positivo, es una senal de compra clasica.
    """
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    return ema_fast - ema_slow


@register_indicator("MACD_hist", "momentum", "MACD histogram (signal line 9)")
def compute_macd_histogram(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> pd.Series:
    """
    Histograma del MACD = MACD - Signal Line
    
    La Signal Line es el EMA(9) del MACD.
    El histograma mide que tan rapido esta cambiando el MACD.
    
    Histograma creciente → la tendencia se esta ACELERANDO
    Histograma decreciente → la tendencia se esta FRENANDO
    """
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd - signal_line


@register_indicator("ROC", "momentum", "Rate of Change (12 periodos)")
def compute_roc(df: pd.DataFrame, period: int = 12) -> pd.Series:
    """
    ROC - Rate of Change (Tasa de Cambio)
    
    Formula: ROC = (precio_hoy - precio_hace_N) / precio_hace_N * 100
    
    Es similar al retorno pero expresado en porcentaje y
    tipicamente se usa con periodos mas largos.
    Mide el MOMENTUM puro: que tan rapido cambia el precio.
    """
    return df["Close"].pct_change(period) * 100


@register_indicator("Momentum", "momentum", "Momentum absoluto (10 periodos)")
def compute_momentum(df: pd.DataFrame, period: int = 10) -> pd.Series:
    """
    Momentum clasico = precio_hoy - precio_hace_N
    
    A diferencia del ROC (porcentual), este es ABSOLUTO.
    Es la forma mas simple de medir momentum.
    Para la red neuronal, se normalizara despues.
    """
    return df["Close"].diff(period)


# =================================================================
# VOLATILIDAD - ATR, Bollinger, Volatilidad historica
# =================================================================
# La volatilidad mide cuanto FLUCTUA el precio.
# Alta volatilidad = movimientos grandes (riesgo + oportunidad)
# Baja volatilidad = movimientos pequenos (estabilidad)
#
# Para gestionar riesgo, la volatilidad es FUNDAMENTAL.
# Un stop loss del 5% tiene sentido en un activo con volatilidad
# del 2%, pero es demasiado ajustado si la volatilidad es del 10%.
# =================================================================

@register_indicator(
    "ATR", "volatilidad",
    "Average True Range (14 periodos)"
)
def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    ATR - Average True Range (Rango Verdadero Promedio)
    
    QUE MIDE?
    ---------
    El rango promedio de movimiento diario del precio, considerando gaps.
    
    QUE ES EL TRUE RANGE?
    ---------------------
    El "rango verdadero" del dia es el MAXIMO de:
    1. High - Low (rango normal del dia)
    2. |High - Close_ayer| (gap alcista)
    3. |Low - Close_ayer| (gap bajista)
    
    POR QUE CONSIDERAR GAPS?
    Si BTC cierra a $1,000,000 y abre al dia siguiente a $1,100,000
    (gap de 10%), el rango High-Low del dia NO captura ese movimiento.
    El True Range si lo captura.
    
    Normalizamos dividiendo por Close para hacer el ATR comparable
    entre diferentes niveles de precio.
    """
    high = df["High"]
    low = df["Low"]
    close_prev = df["Close"].shift(1)
    
    # True Range: maximo de los 3 componentes
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Promedio exponencial del True Range
    atr = true_range.ewm(span=period, adjust=False).mean()
    
    # Normalizar por precio (para comparabilidad)
    return atr / df["Close"]


@register_indicator(
    "BB_width", "volatilidad",
    "Ancho de Bandas de Bollinger (20 periodos, 2 std)"
)
def compute_bollinger_width(
    df: pd.DataFrame,
    period: int = 20,
    num_std: float = 2.0
) -> pd.Series:
    """
    Ancho de las Bandas de Bollinger
    
    QUE SON LAS BANDAS DE BOLLINGER?
    ---------------------------------
    Son 3 lineas basadas en la SMA y la desviacion estandar:
    - Banda superior = SMA(20) + 2 * STD(20)
    - Banda media    = SMA(20)
    - Banda inferior = SMA(20) - 2 * STD(20)
    
    El ANCHO de las bandas = (superior - inferior) / media
    
    INTERPRETACION:
    - Bandas estrechas → Baja volatilidad → El precio "se esta comprimiendo"
      (posible explosion de movimiento proximo)
    - Bandas anchas → Alta volatilidad → Grandes movimientos en curso
    
    Este fenomeno se llama "Bollinger Squeeze" y es una de las senales
    mas populares en trading.
    """
    sma = df["Close"].rolling(window=period).mean()
    std = df["Close"].rolling(window=period).std()
    
    upper = sma + num_std * std
    lower = sma - num_std * std
    
    # Ancho relativo (normalizado por la media)
    width = (upper - lower) / sma
    
    return width


@register_indicator(
    "Volatilidad_20d", "volatilidad",
    "Volatilidad historica de 20 dias (desviacion estandar de retornos)"
)
def compute_volatility_20d(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Volatilidad historica = desviacion estandar de los retornos diarios
    
    Es la forma mas clasica de medir volatilidad.
    Se anualiza multiplicando por sqrt(365) para crypto (opera 365 dias)
    o sqrt(252) para acciones (252 dias habiles).
    
    Aqui NO anualizamos porque la red neuronal trabajara con el valor
    diario. La anualizacion es para humanos, no para la red.
    """
    returns = df["Close"].pct_change()
    return returns.rolling(window=period).std()


# =================================================================
# VOLUMEN - OBV, Volumen Relativo
# =================================================================
# El volumen es la CANTIDAD de activo negociada.
# Es la "energia" detras del movimiento de precio.
#
# "El precio es la direccion, el volumen es el combustible"
#
# Un precio que sube con ALTO volumen tiene mas fuerza que
# un precio que sube con bajo volumen.
# =================================================================

@register_indicator(
    "OBV", "volumen",
    "On-Balance Volume"
)
def compute_obv(df: pd.DataFrame) -> pd.Series:
    """
    OBV - On-Balance Volume (Volumen en Balance)
    
    QUE MIDE?
    ---------
    Es una suma acumulativa del volumen, donde:
    - Si el precio SUBE, se SUMA el volumen
    - Si el precio BAJA, se RESTA el volumen
    
    INTUICION:
    Si el OBV sube consistentemente, hay ACUMULACION (mas compras).
    Si el OBV baja consistentemente, hay DISTRIBUCION (mas ventas).
    
    DIVERGENCIA:
    Si el precio sube pero el OBV baja → senal de debilidad
    Si el precio baja pero el OBV sube → senal de fuerza oculta
    
    Normalizamos con Z-score rolling para que sea comparable.
    """
    # Direccion del precio: 1 si subio, -1 si bajo, 0 si igual
    direction = np.sign(df["Close"].diff())
    
    # OBV = suma acumulativa de (volumen * direccion)
    obv = (df["Volume"] * direction).cumsum()
    
    # Normalizar con z-score rolling (para que no dependa de la escala)
    obv_mean = obv.rolling(window=20, min_periods=1).mean()
    obv_std = obv.rolling(window=20, min_periods=1).std()
    obv_std = obv_std.replace(0, 1)  # Evitar division por cero
    
    return (obv - obv_mean) / obv_std


@register_indicator(
    "Volumen_relativo", "volumen",
    "Volumen relativo al promedio de 20 periodos"
)
def compute_relative_volume(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Volumen Relativo = Volumen_hoy / Promedio_volumen(20)
    
    Valores:
    - ~1.0: volumen normal
    - >2.0: volumen inusualmente alto (algo esta pasando)
    - <0.5: volumen inusualmente bajo (poca actividad)
    
    Es mas util que el volumen absoluto porque es RELATIVO.
    Un volumen de 1000 BTC puede ser alto para un dia normal
    pero bajo durante un crash.
    """
    avg_volume = df["Volume"].rolling(window=period).mean()
    avg_volume = avg_volume.replace(0, 1)  # Evitar division por cero
    return df["Volume"] / avg_volume


# =================================================================
# REGIMEN DE MERCADO - ADX, Pendiente SMA200
# =================================================================
# El "regimen" describe el TIPO de mercado actual:
# - Mercado en tendencia: el precio se mueve consistentemente en una direccion
# - Mercado lateral (range-bound): el precio oscila sin direccion clara
#
# Diferentes estrategias funcionan en diferentes regimenes.
# Una estrategia de "seguir la tendencia" fracasa en mercados laterales.
# Una estrategia de "reversion a la media" fracasa en tendencias fuertes.
# =================================================================

@register_indicator(
    "ADX", "regimen",
    "Average Directional Index (14 periodos)"
)
def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    ADX - Average Directional Index (Indice Direccional Promedio)
    
    QUE MIDE?
    ---------
    La FUERZA de la tendencia (NO la direccion).
    Rango: [0, 100]
    
    INTERPRETACION:
    ADX < 20:  Sin tendencia (mercado lateral)
    ADX 20-40: Tendencia moderada
    ADX > 40:  Tendencia fuerte
    ADX > 60:  Tendencia muy fuerte (raro)
    
    COMO SE CALCULA?
    ----------------
    1. Calcular +DM (Directional Movement positivo) y -DM (negativo)
       +DM = High_hoy - High_ayer (si es positivo y mayor que -DM)
       -DM = Low_ayer - Low_hoy (si es positivo y mayor que +DM)
    2. Calcular +DI y -DI (Directional Indicators)
       +DI = EMA(+DM) / ATR * 100
       -DI = EMA(-DM) / ATR * 100
    3. DX = |+DI - -DI| / (+DI + -DI) * 100
    4. ADX = EMA(DX)
    
    POR QUE ES UTIL PARA NOSOTROS?
    Le dice a la red neuronal si VALE LA PENA buscar tendencias
    o si el mercado esta lateral y es mejor no operar.
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    
    # Movimiento direccional
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    # Solo mantener el mayor y si es positivo
    plus_dm = plus_dm.where(
        (plus_dm > minus_dm) & (plus_dm > 0), 0
    )
    minus_dm = minus_dm.where(
        (minus_dm > plus_dm) & (minus_dm > 0), 0
    )
    
    # True Range para normalizar
    close_prev = close.shift(1)
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Suavizado con EMA
    atr = true_range.ewm(span=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr
    
    # DX y ADX
    di_sum = plus_di + minus_di
    di_sum = di_sum.replace(0, 1)  # Evitar division por cero
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    adx = dx.ewm(span=period, adjust=False).mean()
    
    return adx


@register_indicator(
    "Pendiente_SMA200", "regimen",
    "Pendiente de la SMA de 200 periodos"
)
def compute_sma200_slope(df: pd.DataFrame, period: int = 200) -> pd.Series:
    """
    Pendiente de la SMA(200)
    
    QUE MIDE?
    ---------
    La DIRECCION de la tendencia de largo plazo.
    
    - Pendiente positiva → Tendencia alcista de largo plazo
    - Pendiente negativa → Tendencia bajista de largo plazo
    - Pendiente cercana a 0 → Sin tendencia clara
    
    Se calcula como la diferencia porcentual de la SMA(200)
    entre hoy y hace 20 dias (normalizado).
    """
    sma = df["Close"].rolling(window=period).mean()
    
    # Pendiente = cambio de la SMA en los ultimos 20 periodos / valor de la SMA
    slope = sma.diff(20) / sma
    
    return slope


# =================================================================
# CLASE PRINCIPAL: IndicatorEngine
# =================================================================

class IndicatorEngine:
    """
    Motor de calculo de indicadores tecnicos.
    
    Esta clase:
    1. Conoce TODOS los indicadores disponibles (via INDICATOR_REGISTRY)
    2. Calcula los indicadores seleccionados sobre un DataFrame OHLCV
    3. Retorna un DataFrame con todos los indicadores como columnas nuevas
    
    POR QUE UNA CLASE?
    ------------------
    Porque queremos:
    - Seleccionar un subconjunto de indicadores (no siempre todos)
    - Reutilizar la misma configuracion para multiples DataFrames
    - Encapsular la logica de "cuales indicadores calcular"
    
    Uso:
    ----
    >>> engine = IndicatorEngine()  # Todos los indicadores
    >>> df_with_indicators = engine.compute(ohlcv_df)
    >>> print(df_with_indicators.columns)
    
    >>> engine = IndicatorEngine(indicators=["RSI", "MACD", "ATR"])
    >>> df_subset = engine.compute(ohlcv_df)
    """
    
    def __init__(self, indicators: Optional[list[str]] = None):
        """
        Inicializa el motor de indicadores.
        
        Parameters
        ----------
        indicators : list[str], opcional
            Lista de nombres de indicadores a calcular.
            Si None, calcula TODOS los disponibles.
            Ejemplo: ["RSI", "MACD", "ATR", "ADX"]
        """
        if indicators is None:
            # Usar todos los indicadores registrados
            self.active_indicators = list(INDICATOR_REGISTRY.keys())
        else:
            # Verificar que los indicadores solicitados existan
            unknown = [i for i in indicators if i not in INDICATOR_REGISTRY]
            if unknown:
                available = list(INDICATOR_REGISTRY.keys())
                raise ValueError(
                    f"Indicadores desconocidos: {unknown}. "
                    f"Disponibles: {available}"
                )
            self.active_indicators = indicators
    
    def compute(
        self,
        df: pd.DataFrame,
        drop_ohlcv: bool = False
    ) -> pd.DataFrame:
        """
        Calcula todos los indicadores activos sobre un DataFrame OHLCV.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con columnas: Open, High, Low, Close, Volume
        drop_ohlcv : bool
            Si True, elimina las columnas OHLCV originales y deja
            solo los indicadores. Util cuando la red neuronal solo
            necesita los indicadores como input.
            
        Returns
        -------
        pd.DataFrame
            DataFrame original + columnas de indicadores.
            Las filas iniciales tendran NaN (periodo de calentamiento).
        """
        # Trabajar sobre una copia para no modificar el original
        result = df.copy()
        
        for indicator_name in self.active_indicators:
            entry = INDICATOR_REGISTRY[indicator_name]
            func = entry["function"]
            
            # Calcular el indicador y agregarlo como nueva columna
            result[indicator_name] = func(df)
        
        if drop_ohlcv:
            ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
            result = result.drop(columns=ohlcv_cols, errors="ignore")
        
        return result
    
    def compute_clean(
        self,
        df: pd.DataFrame,
        drop_ohlcv: bool = False
    ) -> pd.DataFrame:
        """
        Calcula indicadores y elimina filas con NaN.
        
        POR QUE HAY NaN?
        ----------------
        Cada indicador necesita un "periodo de calentamiento".
        Ejemplo: SMA(200) necesita 200 datos antes de poder calcular
        el primer valor. Las primeras 199 filas seran NaN.
        
        Al eliminar NaN, perdemos esas filas iniciales, pero
        aseguramos que todos los datos sean validos.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame OHLCV
        drop_ohlcv : bool
            Si True, solo retorna indicadores (sin OHLCV)
            
        Returns
        -------
        pd.DataFrame
            DataFrame sin NaN, listo para usar
        """
        result = self.compute(df, drop_ohlcv=drop_ohlcv)
        
        rows_before = len(result)
        result = result.dropna()
        rows_removed = rows_before - len(result)
        
        if rows_removed > 0:
            # Esto es normal y esperado. No es un error.
            pass
        
        return result
    
    @staticmethod
    def list_available() -> list[dict]:
        """
        Lista todos los indicadores disponibles con su informacion.
        
        Returns
        -------
        list[dict]
            Lista de diccionarios con nombre, categoria y descripcion
        """
        return [
            {
                "name": entry["name"],
                "category": entry["category"],
                "description": entry["description"]
            }
            for entry in INDICATOR_REGISTRY.values()
        ]
    
    @staticmethod
    def list_by_category() -> dict[str, list[str]]:
        """
        Agrupa indicadores por categoria.
        
        Returns
        -------
        dict
            {categoria: [lista de nombres de indicadores]}
        """
        categories = {}
        for name, entry in INDICATOR_REGISTRY.items():
            cat = entry["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(name)
        return categories
    
    def get_warmup_period(self) -> int:
        """
        Estima cuantos periodos de "calentamiento" necesitan
        los indicadores activos.
        
        El indicador que necesita mas datos determina el periodo
        de calentamiento total. Si usas SMA(200), necesitas al
        menos 200 periodos de datos antes de tener un valor valido.
        
        Returns
        -------
        int
            Numero estimado de periodos de calentamiento
        """
        # Los indicadores con periodos mas largos:
        # SMA200 necesita 200, volatilidad_20d necesita 20+1, etc.
        # Por seguridad, usamos 250 si incluimos SMA200
        if any("200" in name for name in self.active_indicators):
            return 250
        elif any("50" in name for name in self.active_indicators):
            return 70
        elif any("20" in name for name in self.active_indicators):
            return 30
        else:
            return 20


# =================================================================
# BLOQUE DE DEMOSTRACION
# =================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("INDICADORES TECNICOS DISPONIBLES")
    print("=" * 60)
    
    # Listar todos los indicadores
    for info in IndicatorEngine.list_available():
        print(f"\n  [{info['category'].upper():>12}] {info['name']}")
        print(f"  {'':>14} {info['description']}")
    
    # Agrupar por categoria
    print("\n" + "=" * 60)
    print("POR CATEGORIA:")
    print("=" * 60)
    for category, names in IndicatorEngine.list_by_category().items():
        print(f"\n  {category.upper()}: {', '.join(names)}")
    
    print(f"\nTotal: {len(INDICATOR_REGISTRY)} indicadores disponibles")
    
    # Ejemplo con datos reales (si hay)
    try:
        from src.data.downloader import DataDownloader
        
        print("\n" + "=" * 60)
        print("CALCULANDO INDICADORES SOBRE BTC-MXN...")
        print("=" * 60)
        
        downloader = DataDownloader()
        df = downloader.download("BTC-MXN", start="2022-01-01", interval="1d")
        
        engine = IndicatorEngine()
        df_indicators = engine.compute_clean(df)
        
        print(f"\nBarras originales: {len(df)}")
        print(f"Barras con indicadores (sin NaN): {len(df_indicators)}")
        print(f"Columnas: {list(df_indicators.columns)}")
        print(f"\nPrimeras filas de indicadores:")
        indicator_cols = [c for c in df_indicators.columns if c not in ["Open", "High", "Low", "Close", "Volume"]]
        print(df_indicators[indicator_cols].head())
        
    except Exception as e:
        print(f"\n(No se pudieron calcular indicadores sobre datos reales: {e})")
        print("Esto es normal si no hay datos descargados aun.")
