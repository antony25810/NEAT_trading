"""
============================================================
walk_forward.py - Validacion Walk-Forward
============================================================

QUE ES WALK-FORWARD?
--------------------
Walk-Forward es la tecnica de validacion MAS IMPORTANTE en trading
algoritmico. Es como "simular el paso del tiempo" para probar
si una estrategia funciona en el futuro, no solo en el pasado.

EL PROBLEMA QUE RESUELVE:
--------------------------
Imagina que tienes datos de BTC de 2018 a 2025. Si optimizas
tu estrategia con TODOS esos datos y luego la evaluas con los
MISMOS datos, obtendras resultados fantasticos... pero falsos.
Esto se llama OVERFITTING (sobreajuste).

Es como estudiar para un examen con las mismas preguntas
que van a salir. Sacaras 10, pero no aprendiste nada.

COMO FUNCIONA WALK-FORWARD?
----------------------------
Divide los datos en ventanas que "caminan hacia adelante":

Ventana 1:
  [===== TRAIN 2018-2022 =====][= VAL 2023 =][= TEST 2024 =]

Ventana 2 (deslizada 1 ano):
        [===== TRAIN 2019-2023 =====][= VAL 2024 =][= TEST 2025 =]

En cada ventana:
  1. TRAIN: Evolucionas tu poblacion aqui (entrenas)
  2. VAL:   Mides el fitness aqui para seleccionar (validas)
  3. TEST:  Prueba final out-of-sample (NUNCA se toca durante evolucion)

El resultado final es el PROMEDIO de todos los tests.
Si la estrategia funciona bien en TODOS los periodos de test,
es mas probable que funcione en el futuro real.

VENTANA DESLIZANTE vs EXPANSIVA
--------------------------------
DESLIZANTE (sliding):
  [====TRAIN====][VAL][TEST]
        [====TRAIN====][VAL][TEST]
              [====TRAIN====][VAL][TEST]
  
  La ventana de train siempre tiene el MISMO tamano.
  Ventaja: Se adapta a condiciones recientes.
  Desventaja: Pierde datos antiguos.

EXPANSIVA (expanding):
  [====TRAIN====][VAL][TEST]
  [========TRAIN========][VAL][TEST]
  [============TRAIN============][VAL][TEST]
  
  La ventana de train CRECE con cada paso.
  Ventaja: Usa todos los datos disponibles.
  Desventaja: Datos antiguos pueden no ser representativos.

Para crypto, recomendamos DESLIZANTE porque el mercado cambia mucho.
"""

import logging
from dataclasses import dataclass
from typing import Iterator, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardWindow:
    """
    Representa una ventana del Walk-Forward.
    
    QUE ES UN DATACLASS?
    --------------------
    Es una clase simplificada de Python para guardar datos.
    En lugar de escribir __init__, __repr__, etc. a mano,
    Python los genera automaticamente con @dataclass.
    
    Equivalente a:
        class WalkForwardWindow:
            def __init__(self, train, val, test, window_index):
                self.train = train
                self.val = val
                ...
    
    Pero mucho mas limpio.
    
    Attributes
    ----------
    train : pd.DataFrame
        Datos de entrenamiento (donde evoluciona la poblacion)
    val : pd.DataFrame  
        Datos de validacion (para seleccionar la mejor estrategia)
    test : pd.DataFrame
        Datos de prueba out-of-sample (resultado final)
    window_index : int
        Indice de la ventana (0, 1, 2, ...)
    train_start : str
        Fecha de inicio del train (para logging)
    train_end : str
        Fecha de fin del train
    val_start : str
        Fecha de inicio de validacion
    val_end : str
        Fecha de fin de validacion
    test_start : str
        Fecha de inicio del test
    test_end : str
        Fecha de fin del test
    """
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    window_index: int
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str
    
    def __repr__(self) -> str:
        return (
            f"WalkForwardWindow(#{self.window_index} | "
            f"Train: {self.train_start} -> {self.train_end} [{len(self.train)} barras] | "
            f"Val: {self.val_start} -> {self.val_end} [{len(self.val)} barras] | "
            f"Test: {self.test_start} -> {self.test_end} [{len(self.test)} barras])"
        )
    
    def summary(self) -> dict:
        """Retorna un resumen de la ventana como diccionario."""
        return {
            "window": self.window_index,
            "train": f"{self.train_start} -> {self.train_end} ({len(self.train)} barras)",
            "val": f"{self.val_start} -> {self.val_end} ({len(self.val)} barras)",
            "test": f"{self.test_start} -> {self.test_end} ({len(self.test)} barras)",
        }


class WalkForwardSplitter:
    """
    Generador de ventanas Walk-Forward para series temporales financieras.
    
    PRINCIPIO FUNDAMENTAL:
    No retorna COPIAS de los datos, sino VISTAS (slices).
    Esto es eficiente en memoria porque los datos no se duplican.
    
    NOTA SOBRE EL TAMANO DE VENTANA:
    Para datos diarios de crypto (365 dias/ano):
      - train_periods=5*365=1825, val_periods=365, test_periods=365
    Para datos horarios:
      - train_periods=5*365*24=43800, val_periods=8760, test_periods=8760
    
    Uso:
    ----
    >>> splitter = WalkForwardSplitter(
    ...     train_periods=1825,  # 5 anos
    ...     val_periods=365,     # 1 ano
    ...     test_periods=365,    # 1 ano
    ...     step_periods=365     # mover 1 ano entre ventanas
    ... )
    >>> for window in splitter.split(df):
    ...     print(window)
    ...     # Evolucionar con window.train
    ...     # Validar con window.val
    ...     # Evaluar con window.test
    """
    
    def __init__(
        self,
        train_periods: int,
        val_periods: int,
        test_periods: int,
        step_periods: Optional[int] = None,
        mode: str = "sliding",
        min_train_periods: Optional[int] = None
    ):
        """
        Configura el splitter Walk-Forward.
        
        Parameters
        ----------
        train_periods : int
            Numero de barras (filas) para entrenamiento.
            Para datos diarios, 1825 ≈ 5 anos.
            Para datos horarios, 43800 ≈ 5 anos.
            
        val_periods : int
            Numero de barras para validacion.
            
        test_periods : int
            Numero de barras para test out-of-sample.
            
        step_periods : int, opcional
            Cuanto "avanzar" la ventana entre iteraciones.
            Default: test_periods (avanzar un periodo de test completo).
            Esto evita que los periodos de test se solapen.
            
        mode : str
            "sliding" o "expanding"
            - sliding: ventana de train de tamano fijo
            - expanding: ventana de train que crece
            
        min_train_periods : int, opcional
            Minimo de periodos de train requeridos.
            Solo relevante en modo "expanding" donde las primeras
            ventanas podrian ser muy pequenas.
            Default: train_periods (no se reduce)
        """
        if mode not in ("sliding", "expanding"):
            raise ValueError(f"mode debe ser 'sliding' o 'expanding', no '{mode}'")
        
        self.train_periods = train_periods
        self.val_periods = val_periods
        self.test_periods = test_periods
        self.step_periods = step_periods or test_periods
        self.mode = mode
        self.min_train_periods = min_train_periods or train_periods
    
    def split(self, df: pd.DataFrame) -> Iterator[WalkForwardWindow]:
        """
        Genera ventanas Walk-Forward sobre un DataFrame.
        
        ALGORITMO:
        1. Empezar desde el inicio de los datos
        2. Tomar train_periods + val_periods + test_periods barras
        3. Dividir en train, val, test
        4. Yield (retornar sin terminar) la ventana
        5. Avanzar step_periods posiciones
        6. Repetir hasta que no quepan mas ventanas
        
        QUE ES yield?
        -------------
        yield es como return, pero la funcion NO termina.
        Se "pausa" y retorna un valor. La proxima vez que se
        llama, continua desde donde se quedo.
        
        Esto se llama un "generador" y es PERFECTO para nuestro caso:
        No cargamos TODAS las ventanas en memoria a la vez,
        sino una a la vez, conforme las necesitamos.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame completo con datos temporales
            
        Yields
        ------
        WalkForwardWindow
            Ventana con datos de train, val y test
        """
        total_rows = len(df)
        
        # Minimo de filas necesarias para una ventana
        min_required = self.train_periods + self.val_periods + self.test_periods
        
        if total_rows < min_required:
            raise ValueError(
                f"No hay suficientes datos para Walk-Forward. "
                f"Se necesitan al menos {min_required} barras, "
                f"pero solo hay {total_rows}. "
                f"(Train={self.train_periods} + Val={self.val_periods} + "
                f"Test={self.test_periods})"
            )
        
        window_index = 0
        
        if self.mode == "sliding":
            # Ventana deslizante de tamano fijo
            start = 0
            while True:
                train_start = start
                train_end = start + self.train_periods
                val_start_idx = train_end
                val_end = val_start_idx + self.val_periods
                test_start_idx = val_end
                test_end = test_start_idx + self.test_periods
                
                # Verificar que la ventana cabe en los datos
                if test_end > total_rows:
                    break
                
                # Crear la ventana usando iloc (posicion numerica)
                train_data = df.iloc[train_start:train_end]
                val_data = df.iloc[val_start_idx:val_end]
                test_data = df.iloc[test_start_idx:test_end]
                
                window = WalkForwardWindow(
                    train=train_data,
                    val=val_data,
                    test=test_data,
                    window_index=window_index,
                    train_start=str(train_data.index[0].date() if hasattr(train_data.index[0], 'date') else train_data.index[0]),
                    train_end=str(train_data.index[-1].date() if hasattr(train_data.index[-1], 'date') else train_data.index[-1]),
                    val_start=str(val_data.index[0].date() if hasattr(val_data.index[0], 'date') else val_data.index[0]),
                    val_end=str(val_data.index[-1].date() if hasattr(val_data.index[-1], 'date') else val_data.index[-1]),
                    test_start=str(test_data.index[0].date() if hasattr(test_data.index[0], 'date') else test_data.index[0]),
                    test_end=str(test_data.index[-1].date() if hasattr(test_data.index[-1], 'date') else test_data.index[-1]),
                )
                
                logger.info(f"Ventana {window_index}: {window}")
                yield window
                
                # Avanzar la ventana
                start += self.step_periods
                window_index += 1
        
        elif self.mode == "expanding":
            # Ventana que crece: el inicio es siempre 0
            test_start_first = self.train_periods + self.val_periods
            step = self.step_periods
            
            cursor = test_start_first
            while cursor + self.test_periods <= total_rows:
                train_start = 0  # Siempre desde el inicio
                train_end = cursor - self.val_periods
                val_start_idx = train_end
                val_end = cursor
                test_start_idx = cursor
                test_end = cursor + self.test_periods
                
                # Verificar tamano minimo de train
                if (train_end - train_start) < self.min_train_periods:
                    cursor += step
                    continue
                
                train_data = df.iloc[train_start:train_end]
                val_data = df.iloc[val_start_idx:val_end]
                test_data = df.iloc[test_start_idx:test_end]
                
                window = WalkForwardWindow(
                    train=train_data,
                    val=val_data,
                    test=test_data,
                    window_index=window_index,
                    train_start=str(train_data.index[0].date() if hasattr(train_data.index[0], 'date') else train_data.index[0]),
                    train_end=str(train_data.index[-1].date() if hasattr(train_data.index[-1], 'date') else train_data.index[-1]),
                    val_start=str(val_data.index[0].date() if hasattr(val_data.index[0], 'date') else val_data.index[0]),
                    val_end=str(val_data.index[-1].date() if hasattr(val_data.index[-1], 'date') else val_data.index[-1]),
                    test_start=str(test_data.index[0].date() if hasattr(test_data.index[0], 'date') else test_data.index[0]),
                    test_end=str(test_data.index[-1].date() if hasattr(test_data.index[-1], 'date') else test_data.index[-1]),
                )
                
                logger.info(f"Ventana {window_index}: {window}")
                yield window
                
                cursor += step
                window_index += 1
    
    def count_windows(self, total_rows: int) -> int:
        """
        Calcula cuantas ventanas se pueden generar sin crear los datos.
        
        Util para saber cuantas iteraciones de Walk-Forward habra
        antes de iniciar (para barras de progreso, estimacion de tiempo, etc.)
        
        Parameters
        ----------
        total_rows : int
            Numero total de barras en el DataFrame
            
        Returns
        -------
        int
            Numero de ventanas Walk-Forward posibles
        """
        min_required = self.train_periods + self.val_periods + self.test_periods
        if total_rows < min_required:
            return 0
        
        if self.mode == "sliding":
            available = total_rows - min_required
            return 1 + available // self.step_periods
        else:
            # Expanding: mas complejo, lo simulamos
            count = 0
            cursor = self.train_periods + self.val_periods
            while cursor + self.test_periods <= total_rows:
                count += 1
                cursor += self.step_periods
            return count
    
    def summary(self, total_rows: int) -> dict:
        """
        Genera un resumen de la configuracion Walk-Forward.
        
        Parameters
        ----------
        total_rows : int
            Numero total de barras disponibles
            
        Returns
        -------
        dict
            Resumen de la configuracion
        """
        n_windows = self.count_windows(total_rows)
        
        return {
            "mode": self.mode,
            "train_periods": self.train_periods,
            "val_periods": self.val_periods,
            "test_periods": self.test_periods,
            "step_periods": self.step_periods,
            "total_rows": total_rows,
            "num_windows": n_windows,
            "coverage": f"{(n_windows * self.test_periods) / total_rows * 100:.1f}%"
        }


# =================================================================
# FUNCIONES DE CONVENIENCIA
# =================================================================
# Estas funciones simplifican la creacion de splitters para
# casos comunes (datos diarios, horarios, etc.)

def create_daily_splitter(
    train_years: int = 5,
    val_years: int = 1,
    test_years: int = 1,
    step_years: int = 1,
    mode: str = "sliding"
) -> WalkForwardSplitter:
    """
    Crea un splitter para datos DIARIOS de crypto.
    
    Crypto opera 365 dias al ano (a diferencia de acciones: 252 dias).
    
    Parameters
    ----------
    train_years : int
        Anos de entrenamiento (default: 5)
    val_years : int
        Anos de validacion (default: 1)
    test_years : int
        Anos de test (default: 1)
    step_years : int
        Anos que avanza la ventana (default: 1)
    mode : str
        "sliding" o "expanding"
        
    Returns
    -------
    WalkForwardSplitter
        Splitter configurado para datos diarios
    """
    # Crypto opera 365 dias al ano (24/7)
    days_per_year = 365
    
    return WalkForwardSplitter(
        train_periods=train_years * days_per_year,
        val_periods=val_years * days_per_year,
        test_periods=test_years * days_per_year,
        step_periods=step_years * days_per_year,
        mode=mode
    )


def create_hourly_splitter(
    train_years: int = 1,
    val_months: int = 3,
    test_months: int = 3,
    step_months: int = 3,
    mode: str = "sliding"
) -> WalkForwardSplitter:
    """
    Crea un splitter para datos HORARIOS de crypto.
    
    NOTA: Yahoo Finance solo tiene ~730 dias de datos horarios.
    Por eso los periodos son mas cortos que en datos diarios.
    
    Parameters
    ----------
    train_years : int
        Anos de entrenamiento (default: 1)
    val_months : int
        Meses de validacion (default: 3)
    test_months : int
        Meses de test (default: 3)
    step_months : int
        Meses que avanza la ventana (default: 3)
    mode : str
        "sliding" o "expanding"
        
    Returns
    -------
    WalkForwardSplitter
        Splitter configurado para datos horarios
    """
    # Crypto: 24 horas * 365 dias = 8760 barras/ano
    hours_per_year = 24 * 365
    hours_per_month = 24 * 30  # Aproximacion
    
    return WalkForwardSplitter(
        train_periods=train_years * hours_per_year,
        val_periods=val_months * hours_per_month,
        test_periods=test_months * hours_per_month,
        step_periods=step_months * hours_per_month,
        mode=mode
    )


# =================================================================
# DEMOSTRACION
# =================================================================
if __name__ == "__main__":
    import numpy as np
    
    print("=" * 70)
    print("WALK-FORWARD SPLITTER - Demostracion")
    print("=" * 70)
    
    # Crear datos sinteticos: 8 anos de datos diarios
    # (como si fueran datos de BTC de 2017 a 2025)
    dates = pd.date_range("2017-01-01", "2025-01-01", freq="D")
    n = len(dates)
    
    df = pd.DataFrame({
        "Close": np.cumsum(np.random.randn(n) * 0.5) + 100,
        "Volume": np.random.uniform(1e6, 1e8, n)
    }, index=dates)
    
    print(f"\nDatos sinteticos: {len(df)} barras ({df.index[0].date()} a {df.index[-1].date()})")
    
    # --- Modo SLIDING ---
    print("\n" + "-" * 70)
    print("MODO SLIDING (Ventana deslizante)")
    print("-" * 70)
    
    splitter = create_daily_splitter(
        train_years=3,   # 3 anos de train
        val_years=1,      # 1 ano de validacion
        test_years=1,     # 1 ano de test
        step_years=1,     # avanzar 1 ano
        mode="sliding"
    )
    
    summary = splitter.summary(len(df))
    print(f"\nConfiguracion: {summary}")
    
    for window in splitter.split(df):
        print(f"\n  {window}")
    
    # --- Modo EXPANDING ---
    print("\n" + "-" * 70)
    print("MODO EXPANDING (Ventana expansiva)")
    print("-" * 70)
    
    splitter_exp = create_daily_splitter(
        train_years=3,
        val_years=1,
        test_years=1,
        step_years=1,
        mode="expanding"
    )
    
    for window in splitter_exp.split(df):
        print(f"\n  {window}")
    
    # Verificar que no hay solapamiento entre train/val/test
    print("\n" + "-" * 70)
    print("VERIFICACION DE NO SOLAPAMIENTO")
    print("-" * 70)
    
    for window in splitter.split(df):
        train_end_idx = df.index.get_loc(window.train.index[-1])
        val_start_idx = df.index.get_loc(window.val.index[0])
        val_end_idx = df.index.get_loc(window.val.index[-1])
        test_start_idx = df.index.get_loc(window.test.index[0])
        
        # Train debe terminar ANTES de que val empiece
        assert train_end_idx < val_start_idx, "SOLAPAMIENTO train-val!"
        # Val debe terminar ANTES de que test empiece
        assert val_end_idx < test_start_idx, "SOLAPAMIENTO val-test!"
        
        print(f"  Ventana {window.window_index}: Sin solapamiento")
    
    print("\n  Todas las ventanas pasaron la verificacion.")
