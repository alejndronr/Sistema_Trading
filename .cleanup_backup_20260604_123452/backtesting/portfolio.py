"""
Portfolio — Gestión de Portfolio Simulado
==========================================
Mantiene el estado del portfolio durante el backtesting:
  - Balance de capital
  - Posiciones abiertas
  - Historial de trades
  - Curva de equity

Este es el "libro de órdenes" simulado que el BacktestEngine usa
para registrar todas las operaciones de forma consistente.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from config.settings import RISK, SetupQuality, SignalDirection


@dataclass
class Position:
    """Representa una posición abierta."""
    trade_id: str
    symbol: str
    strategy: str
    direction: SignalDirection
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: Optional[float]
    position_size: float      # Número de unidades
    risk_amount: float        # Capital en riesgo
    entry_time: datetime
    setup_quality: SetupQuality = SetupQuality.A
    entry_reason: str = ""

    # Estado de la posición
    tp1_hit: bool = False             # TP1 ya alcanzado (50% cerrado)
    current_sl: float = 0.0          # SL actual (puede moverse a BE o trailing)
    partial_size: float = 0.0        # Tamaño de la porción que queda tras TP1

    def __post_init__(self):
        if self.current_sl == 0.0:
            self.current_sl = self.stop_loss
        if self.partial_size == 0.0:
            self.partial_size = self.position_size

    @property
    def unrealized_pnl(self) -> float:
        """PnL no realizado — requiere precio actual."""
        return 0.0  # Se calcula en el engine con el precio actual

    def calculate_pnl(self, exit_price: float, size: Optional[float] = None) -> float:
        """Calcula el PnL realizado para un tamaño dado."""
        sz = size or self.position_size
        if self.direction == SignalDirection.LONG:
            return (exit_price - self.entry_price) * sz
        else:
            return (self.entry_price - exit_price) * sz


@dataclass
class ClosedTrade:
    """Registro completo de un trade cerrado."""
    trade_id: str
    symbol: str
    strategy: str
    direction: str
    setup_quality: str
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: Optional[float]
    position_size: float
    risk_amount: float
    pnl_usd: float
    pnl_pct: float
    r_multiple: float          # PnL / risk_amount
    entry_time: datetime
    exit_time: datetime
    duration_hours: float
    exit_reason: str
    entry_reason: str
    market_regime: str = "unknown"


class Portfolio:
    """
    Gestiona el portfolio simulado durante el backtesting.
    Registra cada trade con comisiones y slippage realista.
    """

    def __init__(
        self,
        initial_capital: float = RISK.initial_capital,
        maker_fee: float = RISK.maker_fee,
        taker_fee: float = RISK.taker_fee,
    ):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee

        self._open_positions: Dict[str, Position] = {}
        self._closed_trades: List[ClosedTrade] = []
        self._equity_curve: List[dict] = []

    # ── Apertura de posiciones ─────────────────────────────────────────────────

    def open_position(
        self,
        symbol: str,
        strategy: str,
        direction: SignalDirection,
        entry_price: float,
        stop_loss: float,
        take_profit_1: float,
        position_size: float,
        risk_amount: float,
        entry_time: datetime,
        take_profit_2: Optional[float] = None,
        setup_quality: SetupQuality = SetupQuality.A,
        entry_reason: str = "",
        market_regime: str = "unknown",
        slippage_pct: float = 0.001,
    ) -> Optional[str]:
        """
        Abre una nueva posición con slippage y comisión simulados.
        Retorna el trade_id o None si no hay capital suficiente.
        """
        # Aplicar slippage al precio de entrada
        if direction == SignalDirection.LONG:
            real_entry = entry_price * (1 + slippage_pct)
        else:
            real_entry = entry_price * (1 - slippage_pct)

        # Comisión de apertura
        commission = real_entry * position_size * self.taker_fee
        total_cost = real_entry * position_size + commission

        if total_cost > self.capital:
            return None  # Capital insuficiente

        self.capital -= commission  # Deducir solo la comisión (no el notional completo en spot)

        trade_id = str(uuid.uuid4())
        position = Position(
            trade_id=trade_id,
            symbol=symbol,
            strategy=strategy,
            direction=direction,
            entry_price=real_entry,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            position_size=position_size,
            risk_amount=risk_amount,
            entry_time=entry_time,
            setup_quality=setup_quality,
            entry_reason=entry_reason,
        )

        self._open_positions[trade_id] = position
        return trade_id

    # ── Cierre de posiciones ───────────────────────────────────────────────────

    def close_position(
        self,
        trade_id: str,
        exit_price: float,
        exit_time: datetime,
        exit_reason: str = "",
        size_fraction: float = 1.0,  # 1.0 = cerrar todo, 0.5 = cerrar la mitad
        slippage_pct: float = 0.001,
    ) -> Optional[ClosedTrade]:
        """
        Cierra una posición (total o parcialmente).
        Aplica slippage y comisión de salida.
        """
        if trade_id not in self._open_positions:
            return None

        pos = self._open_positions[trade_id]

        # Aplicar slippage al precio de salida
        if pos.direction == SignalDirection.LONG:
            real_exit = exit_price * (1 - slippage_pct)
        else:
            real_exit = exit_price * (1 + slippage_pct)

        size_to_close = pos.partial_size * size_fraction
        commission = real_exit * size_to_close * self.maker_fee

        pnl_gross = pos.calculate_pnl(real_exit, size_to_close)
        pnl_net = pnl_gross - commission

        self.capital += pnl_net

        # Calcular métricas del trade
        risk = pos.risk_amount
        r_multiple = pnl_net / risk if risk > 0 else 0

        duration = (exit_time - pos.entry_time).total_seconds() / 3600

        trade = ClosedTrade(
            trade_id=trade_id,
            symbol=pos.symbol,
            strategy=pos.strategy,
            direction=pos.direction.name,
            setup_quality=pos.setup_quality.value,
            entry_price=pos.entry_price,
            exit_price=real_exit,
            stop_loss=pos.stop_loss,
            take_profit_1=pos.take_profit_1,
            take_profit_2=pos.take_profit_2,
            position_size=size_to_close,
            risk_amount=risk,
            pnl_usd=round(pnl_net, 4),
            pnl_pct=round(pnl_net / self.initial_capital * 100, 4),
            r_multiple=round(r_multiple, 3),
            entry_time=pos.entry_time,
            exit_time=exit_time,
            duration_hours=round(duration, 2),
            exit_reason=exit_reason,
            entry_reason=pos.entry_reason,
        )
        self._closed_trades.append(trade)

        # Si se cerró todo, remover la posición
        if size_fraction >= 1.0:
            del self._open_positions[trade_id]
        else:
            # Actualizar tamaño restante
            pos.partial_size -= size_to_close
            pos.tp1_hit = True

        return trade

    def update_stop_loss(self, trade_id: str, new_sl: float) -> bool:
        """Actualiza el SL de una posición abierta (solo hacia adelante, nunca atrás)."""
        if trade_id not in self._open_positions:
            return False
        pos = self._open_positions[trade_id]
        if pos.direction == SignalDirection.LONG:
            pos.current_sl = max(pos.current_sl, new_sl)  # Solo puede subir
        else:
            pos.current_sl = min(pos.current_sl, new_sl)  # Solo puede bajar
        return True

    def record_equity(self, timestamp: datetime) -> None:
        """Registra el equity actual en la curva."""
        self._equity_curve.append({
            "timestamp": timestamp,
            "capital": self.capital,
            "open_positions": len(self._open_positions),
        })

    # ── Propiedades ───────────────────────────────────────────────────────────

    @property
    def open_positions(self) -> Dict[str, Position]:
        return self._open_positions

    @property
    def closed_trades(self) -> List[ClosedTrade]:
        return self._closed_trades

    @property
    def equity_curve(self) -> pd.DataFrame:
        if not self._equity_curve:
            return pd.DataFrame()
        return pd.DataFrame(self._equity_curve)

    @property
    def total_pnl(self) -> float:
        return self.capital - self.initial_capital

    @property
    def total_return_pct(self) -> float:
        return (self.capital - self.initial_capital) / self.initial_capital * 100

    def get_trades_dataframe(self) -> pd.DataFrame:
        """Retorna todos los trades cerrados como DataFrame."""
        if not self._closed_trades:
            return pd.DataFrame()
        return pd.DataFrame([vars(t) for t in self._closed_trades])
