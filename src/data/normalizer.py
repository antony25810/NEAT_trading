"""
============================================================
normalizer.py - Normalizacion de Datos para la Red Neuronal
============================================================

POR QUE NORMALIZAR?
-------------------
Las redes neuronales funcionan MUCHO mejor cuando todos los inputs
estan en rangos similares. Sin normalizar:

  - BTC precio: ~$1,000,000 MXN
  - RSI: 0-100
  - Retornos: -0.05 a +0.05
  - Volumen: miles de millones

La red le daria toda la atencion al precio (porque es enorme)
e ignoraria las otras variables. Normalizar pone todo en [-1, 1].

METODOS DE NORMALIZACION
-------------------------
1. Z-Score: x_norm = (x - media) / desviacion_estandar
   - Centra en 0, mayoria de valores entre -3 y +3
   - Bueno para datos con distribucion "normal" (campana de Gauss)

2. MinMax: x_norm = 2 * (x - min) / (max - min) - 1
   - Mapea al rango exacto [-1, 1]
   - Bueno para datos con limites conocidos

QUE ES DATA LEAKAGE (FUGA DE INFORMACION)?
-------------------------------------------
Data leakage ocurre cuando usas informacion del FUTURO para
tomar decisiones en el PRESENTE. En normalizacion:

  INCORRECTO (data leakage):
  - Calcular media y std de TODOS los datos (incluido futuro)
  - Normalizar cada punto con esa media/std global
  - Problema: en el punto T, ya "viste" datos de T+1, T+2, etc.

  CORRECTO (rolling):
  - Para normalizar el punto T, usar SOLO datos hasta T
  - Calcular media y std de los ultimos N puntos (T-N hasta T)
  - Asi simulas lo que harias en tiempo real

EJEMPLO VISUAL:
  Datos: [10, 12, 8, 15, 7, 20, 5]
  
  Para normalizar el valor "15" (posicion 3):
  - INCORRECTO: usar media de [10,12,8,15,7,20,5] = 11 (vio el futuro!)
  - CORRECTO: usar media de [10,12,8,15] = 11.25 (solo pasado)

  Con ventana rolling de 3:
  - CORRECTO: usar media de [8,15] = 11.5 (solo ultimos 3 datos)
"""

import numpy as np
import pandas as pd
from typing import Optional


class RollingNormalizer:
    """
    Normalizador con ventana rolling (sin data leakage).
    
    COMO FUNCIONA?
    --------------
    Para cada punto temporal T, calcula la media y desviacion estandar
    usando SOLO los ultimos `window` puntos (de T-window+1 hasta T).
    Luego normaliza:  x_norm = (x - media_rolling) / std_rolling
    
    El resultado se comprime suavemente a (-1, 1) con tanh(z / tanh_scale).
    A diferencia de un recorte duro, tanh conserva el ORDEN de los valores
    extremos: un movimiento de 4 desviaciones sigue siendo distinto de uno
    de 1 desviacion (con un recorte en +-1 ambos valdrian lo mismo).
    
    POR QUE VENTANA ROLLING Y NO ACUMULATIVA?
    ------------------------------------------
    Con normalizacion acumulativa, la media incluye datos muy antiguos
    que pueden ya no ser representativos. El mercado CAMBIA con el tiempo
    (esto se llama "no-estacionariedad"). Una ventana rolling se adapta
    a las condiciones recientes.
    
    Ejemplo:
    - Si BTC tenia volatilidad del 5% hace 3 anos pero ahora es del 2%,
      la ventana rolling captura la volatilidad actual.
    - La acumulativa seguiria promediando el 5% antiguo.
    
    Uso:
    ----
    >>> normalizer = RollingNormalizer(window=252, method="zscore")
    >>> df_norm = normalizer.normalize(df)
    """
    
    # Metodos de normalizacion disponibles
    METHODS = ["zscore", "minmax"]
    SQUASH_MODES = ["tanh", "clip"]
    
    def __init__(
        self,
        window: int = 252,
        method: str = "zscore",
        clip_value: float = 1.0,
        squash: str = "tanh",
        tanh_scale: float = 2.0,
        min_periods: Optional[int] = None
    ):
        """
        Inicializa el normalizador.
        
        Parameters
        ----------
        window : int
            Tamano de la ventana rolling.
            - 252 para datos diarios (~1 ano de trading)
            - 24*365 = 8760 para datos horarios (~1 ano)
            
        method : str
            Metodo de normalizacion: "zscore" o "minmax"
            
        clip_value : float
            Valor maximo/minimo despues de normalizar.
            Con squash="tanh" escala la salida: clip_value * tanh(z / tanh_scale).
            Con squash="clip" los valores fuera de [-clip, +clip] se truncan.
            
        squash : str
            Como llevar el z-score a un rango acotado:
            - "tanh" (default): compresion suave, sin saturacion abrupta.
            - "clip": recorte duro (comportamiento anterior).
            
        tanh_scale : float
            Escala del z-score antes de tanh. Con 2.0:
            z=1 -> 0.46, z=2 -> 0.76, z=3 -> 0.91, z=4 -> 0.96

        min_periods : int, opcional
            Minimo de observaciones validas dentro de la ventana para emitir
            un valor normalizado. Si es None, usa `max(2, window // 4)`.
        """
        if method not in self.METHODS:
            raise ValueError(
                f"Metodo '{method}' no soportado. Opciones: {self.METHODS}"
            )
        if squash not in self.SQUASH_MODES:
            raise ValueError(
                f"Squash '{squash}' no soportado. Opciones: {self.SQUASH_MODES}"
            )
        if tanh_scale <= 0:
            raise ValueError("tanh_scale debe ser positivo")
        
        self.window = window
        self.method = method
        self.clip_value = clip_value
        self.squash = squash
        self.tanh_scale = tanh_scale
        self.min_periods = min_periods

    
    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normaliza todas las columnas numericas del DataFrame.
        
        FLUJO:
        1. Para cada columna numerica:
           a. Calcular media/std (o min/max) rolling
           b. Normalizar: (x - media) / std  o  2*(x-min)/(max-min) - 1
           c. Comprimir con tanh (o recortar) a (-clip_value, +clip_value)
        2. Las columnas no numericas se mantienen sin cambio
        3. Los NaN del periodo de calentamiento se mantienen (el pipeline
           los elimina; NO se rellenan con 0 para no inventar datos)
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con datos a normalizar
            
        Returns
        -------
        pd.DataFrame
            DataFrame normalizado (misma forma que el original)
        """
        result = df.copy()
        
        # Obtener solo columnas numericas
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            series = df[col]
            
            if self.method == "zscore":
                result[col] = self._zscore_rolling(series)
            elif self.method == "minmax":
                result[col] = self._minmax_rolling(series)
        
        return result
    
    def normalize_columns(
        self,
        df: pd.DataFrame,
        columns: list[str]
    ) -> pd.DataFrame:
        """
        Normaliza solo las columnas especificadas.
        
        Util cuando tienes columnas OHLCV que NO quieres normalizar
        (porque las necesitas en su escala original para el backtesting)
        pero SI quieres normalizar los indicadores.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame completo
        columns : list[str]
            Lista de columnas a normalizar
            
        Returns
        -------
        pd.DataFrame
            DataFrame con solo las columnas especificadas normalizadas
        """
        result = df.copy()
        
        for col in columns:
            if col not in df.columns:
                raise ValueError(f"Columna '{col}' no encontrada en el DataFrame")
            
            series = df[col]
            
            if self.method == "zscore":
                result[col] = self._zscore_rolling(series)
            elif self.method == "minmax":
                result[col] = self._minmax_rolling(series)
        
        return result
    
    def _zscore_rolling(self, series: pd.Series) -> pd.Series:
        """
        Normaliza una serie con Z-Score rolling.
        
        FORMULA:
        z = (x - media_rolling) / std_rolling
        
        QUE ES EL Z-SCORE?
        -------------------
        Mide "cuantas desviaciones estandar" esta un valor por encima
        o por debajo de la media.
        
        z =  0: el valor esta exactamente en la media
        z = +1: el valor esta 1 desviacion estandar por encima
        z = -2: el valor esta 2 desviaciones estandar por debajo
        
        ~68% de los datos tienen |z| < 1
        ~95% de los datos tienen |z| < 2
        ~99% de los datos tienen |z| < 3
        
        Antes se recortaba a [-1, 1], lo que saturaba ~1/3 de los datos
        (todo lo que estuviera a mas de 1 std valia lo mismo). Ahora se usa
        tanh(z / tanh_scale): acota la salida pero preserva la diferencia
        entre un movimiento normal y uno extremo.
        
        Parameters
        ----------
        series : pd.Series
            Serie temporal a normalizar
            
        Returns
        -------
        pd.Series
            Serie normalizada en (-clip_value, +clip_value). Las barras de
            calentamiento (menos de min_periods datos) quedan como NaN.
        """
        # min_periods=max(2, window//4): necesitamos al menos algunos datos
        # para calcular una media/std significativa
        min_periods = (
            max(2, int(self.min_periods))
            if self.min_periods is not None
            else max(2, self.window // 4)
        )
        
        rolling_mean = series.rolling(
            window=self.window,
            min_periods=min_periods
        ).mean()
        
        rolling_std = series.rolling(
            window=self.window,
            min_periods=min_periods
        ).std()
        
        # Evitar division por cero: si std = 0, todos los valores son iguales
        # En ese caso, el z-score deberia ser 0 (esta en la media)
        constant_window = rolling_std.eq(0)
        zscore = (series - rolling_mean) / rolling_std.replace(0, np.nan)
        
        # Donde std era 0, poner 0 (valor esta en la media).
        # El calentamiento (std NaN) se deja como NaN.
        zscore = zscore.mask(constant_window, 0.0)
        
        return self._squash(zscore)
    
    def _squash(self, values: pd.Series) -> pd.Series:
        """Lleva valores no acotados a (-clip_value, +clip_value)."""
        if self.squash == "tanh":
            return self.clip_value * np.tanh(values / self.tanh_scale)
        return values.clip(-self.clip_value, self.clip_value)
    
    def _minmax_rolling(self, series: pd.Series) -> pd.Series:
        """
        Normaliza una serie con MinMax rolling al rango [-1, 1].
        
        FORMULA:
        x_norm = 2 * (x - min_rolling) / (max_rolling - min_rolling) - 1
        
        DIFERENCIA CON Z-SCORE:
        - MinMax GARANTIZA que el rango sea [-1, 1] (antes del clip)
        - Z-Score puede dar valores fuera de [-1, 1] que luego se clipean
        - MinMax es mas sensible a outliers (un solo valor extremo
          cambia el rango completo)
        - Z-Score es mas robusto a outliers
        
        Parameters
        ----------
        series : pd.Series
            Serie temporal a normalizar
            
        Returns
        -------
        pd.Series
            Serie normalizada al rango [-1, 1]
        """
        min_periods = (
            max(2, int(self.min_periods))
            if self.min_periods is not None
            else max(2, self.window // 4)
        )
        
        rolling_min = series.rolling(
            window=self.window,
            min_periods=min_periods
        ).min()
        
        rolling_max = series.rolling(
            window=self.window,
            min_periods=min_periods
        ).max()
        
        # Rango = max - min
        range_val = rolling_max - rolling_min
        
        # Si el rango es 0 (todos iguales), evitar division por cero
        constant_window = range_val.eq(0)
        
        # Normalizar a [0, 1] y luego escalar a [-1, 1]
        normalized = 2 * (series - rolling_min) / range_val.replace(0, np.nan) - 1
        
        # Donde el rango era 0, poner 0; el calentamiento queda NaN
        normalized = normalized.mask(constant_window, 0.0)
        
        # MinMax ya esta acotado por construccion: un recorte basta.
        return normalized.clip(-self.clip_value, self.clip_value)
    
    def get_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Retorna estadisticas de los datos normalizados para verificacion.
        
        Util para confirmar que:
        1. Los valores estan en (-1, 1)
        2. No hay columnas con valores constantes (std=0)
        3. La distribucion es razonable (pocos valores saturados)
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame YA normalizado
            
        Returns
        -------
        pd.DataFrame
            Tabla con min, max, mean, std para cada columna
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        # Con tanh nunca se llega exactamente al limite: se considera
        # "saturado" todo valor por encima del 95% del limite.
        saturation_level = 0.95 * self.clip_value
        
        stats = pd.DataFrame({
            "min": df[numeric_cols].min(),
            "max": df[numeric_cols].max(),
            "mean": df[numeric_cols].mean(),
            "std": df[numeric_cols].std(),
            "nulls": df[numeric_cols].isnull().sum(),
            "pct_saturated": [
                ((df[col].abs() >= saturation_level).sum() / len(df) * 100)
                for col in numeric_cols
            ]
        })
        
        return stats.round(4)


def verify_no_leakage(
    df_original: pd.DataFrame,
    df_normalized: pd.DataFrame,
    test_col: str,
    test_index: int,
    window: int,
    tanh_scale: float = 2.0
) -> dict:
    """
    Verifica que NO hay data leakage en la normalizacion.
    
    COMO FUNCIONA?
    --------------
    Para un punto especifico (test_index), recalcula la normalizacion
    usando SOLO datos disponibles hasta ese punto y compara con el
    valor normalizado. Si son iguales, no hay leakage.
    
    PRUEBA ADICIONAL:
    Modifica un dato FUTURO (test_index + 1) y re-normaliza.
    Si el valor normalizado en test_index NO cambia, confirma
    que el futuro no afecta el presente.
    
    Parameters
    ----------
    df_original : pd.DataFrame
        Datos originales sin normalizar
    df_normalized : pd.DataFrame
        Datos ya normalizados
    test_col : str
        Columna a verificar
    test_index : int
        Indice (posicion) del punto a verificar
    window : int
        Tamano de ventana usado en la normalizacion
    tanh_scale : float
        Escala usada en la compresion tanh del normalizador
        
    Returns
    -------
    dict
        Resultado de la verificacion con detalles
    """
    # Obtener el valor normalizado
    norm_value = df_normalized[test_col].iloc[test_index]
    
    # Recalcular usando solo datos hasta test_index
    available_data = df_original[test_col].iloc[:test_index + 1]
    window_data = available_data.iloc[-window:]
    
    manual_mean = window_data.mean()
    manual_std = window_data.std()
    original_value = df_original[test_col].iloc[test_index]
    
    if manual_std == 0:
        manual_zscore = 0
    else:
        manual_zscore = (original_value - manual_mean) / manual_std
    
    manual_zscore = np.tanh(manual_zscore / tanh_scale)
    
    # Verificar que son iguales (o muy cercanos por errores de punto flotante)
    is_equal = np.isclose(norm_value, manual_zscore, atol=1e-6)
    
    # Test adicional: modificar futuro y verificar que no afecta
    if test_index + 1 < len(df_original):
        df_modified = df_original.copy()
        future_index = df_modified.index[test_index + 1]
        df_modified.loc[future_index, test_col] *= 1000
        
        normalizer = RollingNormalizer(
            window=window, method="zscore", tanh_scale=tanh_scale
        )
        df_renorm = normalizer.normalize_columns(df_modified, [test_col])
        
        renorm_value = df_renorm[test_col].iloc[test_index]
        future_independent = np.isclose(norm_value, renorm_value, atol=1e-6)
    else:
        future_independent = True  # No hay futuro que verificar
    
    return {
        "test_index": test_index,
        "column": test_col,
        "normalized_value": norm_value,
        "manual_calculation": manual_zscore,
        "values_match": is_equal,
        "independent_of_future": future_independent,
        "no_leakage": is_equal and future_independent
    }


# =================================================================
# DEMOSTRACION
# =================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("NORMALIZADOR ROLLING - Demostracion")
    print("=" * 60)
    
    # Crear datos sinteticos para demostrar
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=500, freq="D")
    
    # Simular un precio que va de 100 a ~200 con ruido
    trend = np.linspace(100, 200, 500)
    noise = np.random.randn(500) * 5
    prices = trend + noise
    
    df = pd.DataFrame({
        "Close": prices,
        "RSI": np.random.uniform(20, 80, 500),
        "Volume": np.random.uniform(1e6, 1e8, 500)
    }, index=dates)
    
    print("\nDatos originales:")
    print(df.describe().round(2))
    
    # Normalizar con Z-Score rolling
    normalizer = RollingNormalizer(window=50, method="zscore")
    df_norm = normalizer.normalize(df)
    
    print("\nDatos normalizados (Z-Score rolling, window=50):")
    print(normalizer.get_stats(df_norm))
    
    # Verificar que no hay data leakage
    print("\n" + "=" * 60)
    print("VERIFICACION DE DATA LEAKAGE")
    print("=" * 60)
    
    result = verify_no_leakage(
        df_original=df,
        df_normalized=df_norm,
        test_col="Close",
        test_index=300,
        window=50
    )
    
    for key, value in result.items():
        status = "PASS" if (isinstance(value, bool) and value) else ""
        print(f"  {key}: {value} {status}")
    
    if result["no_leakage"]:
        print("\n  RESULTADO: No se detecto data leakage")
    else:
        print("\n  RESULTADO: ALERTA - Posible data leakage detectado!")
