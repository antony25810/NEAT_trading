"""
============================================================
position.py - Gestion de Posiciones de Trading
============================================================

Define la estructura de una posicion (Long, Short, Cash) y la
logica para calcular PnL, verificar Stop Loss, Take Profit y
Trailing Stop en cada barra de precios.
"""

from enum import IntEnum
from dataclasses import dataclass
from typing import Optional


class PositionType(IntEnum):
    """Tipo de posicion en el mercado."""
    CASH = 0   # Fuera del mercado (100% efectivo)
    LONG = 1   # Comprado (apostar a que sube)
    SHORT = -1 # Vendido (apostar a que baja)


@dataclass
class TradeRecord:
    """
    Registro individual de un trade completado.

    Attributes
    ----------
    entry_time : str or Timestamp
        Fecha/hora de entrada
    exit_time : str or Timestamp
        Fecha/hora de salida
    position_type : PositionType
        LONG o SHORT
    entry_price : float
        Precio de entrada (con slippage)
    exit_price : float
        Precio de salida (con slippage)
    size : float
        Unidades del activo negociadas
    capital_invested : float
        Capital invertido en MXN
    pnl : float
        Ganancia/perdida neta en MXN (despues de comisiones)
    pnl_pct : float
        Retorno porcentual neto de la operacion
    exit_reason : str
        Razón de salida ("signal", "stop_loss", "take_profit", "trailing_stop", "max_drawdown", "end_of_data")
    bars_held : int
        Numero de barras que se mantuvo la posicion
    """
    entry_time: object
    exit_time: object
    position_type: PositionType
    entry_price: float
    exit_price: float
    size: float
    capital_invested: float
    pnl: float
    pnl_pct: float
    exit_reason: str
    bars_held: int


class Position:
    """
    Representa una posicion abierta en el mercado.

    Mantiene el estado actual de la operacion:
    - Precio de entrada y capital asignado
    - Precios dinamicos de Stop Loss, Take Profit y Trailing Stop
    - Registro del precio maximo/minimo alcanzado (para trailing stop)
    """

    def __init__(
        self,
        position_type: PositionType,
        entry_price: float,
        entry_time: object,
        capital: float,
        commission: float = 0.0065,
        slippage: float = 0.001,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.10,
        trailing_stop_pct: float = 0.03
    ):
        """
        Abre una nueva posicion.

        Parameters
        ----------
        position_type : PositionType
            LONG o SHORT
        entry_price : float
            Precio actual de mercado
        entry_time : object
            Fecha/hora de entrada
        capital : float
            Monto de capital asignado a esta operacion en MXN
        commission : float
            Comision por lado (ej: 0.0065 = 0.65%)
        slippage : float
            Deslizamiento por lado (ej: 0.001 = 0.1%)
        stop_loss_pct : float
            % de perdida maximo
        take_profit_pct : float
            % de ganancia objetivo
        trailing_stop_pct : float
            % de trailing stop desde el precio mas favorable
        """
        self.position_type = position_type
        self.entry_time = entry_time
        self.commission = commission
        self.slippage = slippage
        self.capital_allocated = capital
        self.bars_held = 0

        # Precio efectivo de entrada aplicando slippage
        if position_type == PositionType.LONG:
            self.entry_price = entry_price * (1.0 + slippage)
        else:  # SHORT
            self.entry_price = entry_price * (1.0 - slippage)

        # Monto despues de comision de entrada
        self.net_capital = capital * (1.0 - commission)
        self.size = self.net_capital / self.entry_price

        # Parametros de gestion de riesgo
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_stop_pct = trailing_stop_pct

        # Calcular niveles de precio fijos iniciales
        if position_type == PositionType.LONG:
            self.stop_loss_price = self.entry_price * (1.0 - stop_loss_pct)
            self.take_profit_price = self.entry_price * (1.0 + take_profit_pct)
            self.peak_price = self.entry_price
            self.trailing_stop_price = self.stop_loss_price  # Inicia protegiendo igual que el SL
        else:  # SHORT
            self.stop_loss_price = self.entry_price * (1.0 + stop_loss_pct)
            self.take_profit_price = self.entry_price * (1.0 - take_profit_pct)
            self.peak_price = self.entry_price  # para short, peak es el minimo precio alcanzado
            self.trailing_stop_price = self.stop_loss_price  # Inicia protegiendo igual que el SL

    def update_bar(self, high: float, low: float, close: float) -> Optional[str]:
        """
        Actualiza el estado de la posicion con la barra actual y verifica si se gatilla algun stop.

        Parameters
        ----------
        high : float
            Precio maximo de la barra
        low : float
            Precio minimo de la barra
        close : float
            Precio de cierre de la barra

        Returns
        -------
        Optional[str]
            Razón de salida ("stop_loss", "take_profit", "trailing_stop") o None si sigue abierta.
        """
        self.bars_held += 1

        if self.position_type == PositionType.LONG:
            # Politica conservadora: si una vela toca salida adversa y objetivo,
            # se asume que la salida adversa ocurrio primero.
            if low <= self.stop_loss_price:
                return "stop_loss"
            if low <= self.trailing_stop_price:
                return "trailing_stop"
            if high >= self.take_profit_price:
                return "take_profit"

            # El nuevo maximo solo afecta el trailing de barras posteriores.
            if high > self.peak_price:
                self.peak_price = high
                self.trailing_stop_price = max(
                    self.trailing_stop_price,
                    self.peak_price * (1.0 - self.trailing_stop_pct)
                )

        elif self.position_type == PositionType.SHORT:
            if high >= self.stop_loss_price:
                return "stop_loss"
            if high >= self.trailing_stop_price:
                return "trailing_stop"
            if low <= self.take_profit_price:
                return "take_profit"

            if low < self.peak_price:
                self.peak_price = low
                self.trailing_stop_price = min(
                    self.trailing_stop_price,
                    self.peak_price * (1.0 + self.trailing_stop_pct)
                )

        return None

    def check_gap_exit(self, open_price: float) -> Optional[str]:
        """Detecta stops u objetivos atravesados por un gap de apertura."""
        if self.position_type == PositionType.LONG:
            if open_price <= self.stop_loss_price:
                return "stop_loss"
            if open_price <= self.trailing_stop_price:
                return "trailing_stop"
            if open_price >= self.take_profit_price:
                return "take_profit"
        else:
            if open_price >= self.stop_loss_price:
                return "stop_loss"
            if open_price >= self.trailing_stop_price:
                return "trailing_stop"
            if open_price <= self.take_profit_price:
                return "take_profit"
        return None

    def close(self, current_price: float, current_time: object, reason: str) -> TradeRecord:
        """
        Cierra la posicion y calcula el PnL final con comisiones y slippage.

        Parameters
        ----------
        current_price : float
            Precio al que se ejecuta el cierre
        current_time : object
            Fecha/hora de cierre
        reason : str
            Razon del cierre

        Returns
        -------
        TradeRecord
            Registro completo de la operacion
        """
        # Ajustar precio de ejecucion por slippage
        if self.position_type == PositionType.LONG:
            exit_price = current_price * (1.0 - self.slippage)
            gross_value = self.size * exit_price
            exit_commission = gross_value * self.commission
        else:  # SHORT
            exit_price = current_price * (1.0 + self.slippage)
            gross_value = (
                self.net_capital
                + (self.entry_price - exit_price) * self.size
            )
            exit_commission = self.size * exit_price * self.commission

        # Aplicar comision de salida
        net_value = gross_value - exit_commission
        pnl = net_value - self.capital_allocated
        pnl_pct = pnl / self.capital_allocated

        return TradeRecord(
            entry_time=self.entry_time,
            exit_time=current_time,
            position_type=self.position_type,
            entry_price=self.entry_price,
            exit_price=exit_price,
            size=self.size,
            capital_invested=self.capital_allocated,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            bars_held=self.bars_held
        )

    def current_unrealized_pnl(self, current_price: float) -> tuple[float, float]:
        """
        Calcula el PnL no realizado (flotante) al precio actual descontando comision y slippage de salida.

        Returns
        -------
        tuple[float, float]
            (pnl_monto_mxn, pnl_porcentual)
        """
        if self.position_type == PositionType.LONG:
            exit_price = current_price * (1.0 - self.slippage)
            gross_val = self.size * exit_price
            exit_commission = gross_val * self.commission
        else:
            exit_price = current_price * (1.0 + self.slippage)
            gross_val = (
                self.net_capital
                + (self.entry_price - exit_price) * self.size
            )
            exit_commission = self.size * exit_price * self.commission

        net_val = gross_val - exit_commission
        pnl = net_val - self.capital_allocated
        return pnl, pnl / self.capital_allocated
