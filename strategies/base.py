"""
Clase Base de Estrategias
=========================
Define la interfaz común que todas las estrategias deben implementar.
Facilita la composición y el backtesting uniforme de cualquier estrategia.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import pandas as pd

from config.settings import SetupQuality, SignalDirection, StrategyType


@dataclass
class Signal:
    """
    Señal de trading generada por una estrategia.
    Contiene toda la información necesaria para abrir una posición.
    """
    # Identificación
    strategy: StrategyType
    direction: SignalDirection
    symbol: str
    timeframe: str

    # Precios de la operación
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: Optional[float] = None

    # Calidad y contexto
    quality: SetupQuality = SetupQuality.A
    timestamp: Optional[datetime] = None
    confidence: float = 0.5          # 0.0-1.0, para comparar señales simultáneas

    # Razón de entrada (para el journal)
    entry_reason: str = ""
    conditions_met: List[str] = field(default_factory=list)

    # Datos de gestión de posición
    position_size: float = 0.0       # Calculado por el position sizer
    risk_amount: float = 0.0

    # Régimen de mercado en el momento de la señal
    market_regime: str = "unknown"

    @property
    def risk_reward_ratio_tp1(self) -> float:
        """R/R ratio para TP1."""
        risk = abs(self.entry_price - self.stop_loss)
        if risk <= 0:
            return 0.0
        reward = abs(self.take_profit_1 - self.entry_price)
        return reward / risk

    @property
    def risk_reward_ratio_tp2(self) -> Optional[float]:
        """R/R ratio para TP2."""
        if self.take_profit_2 is None:
            return None
        risk = abs(self.entry_price - self.stop_loss)
        if risk <= 0:
            return 0.0
        reward = abs(self.take_profit_2 - self.entry_price)
        return reward / risk

    @property
    def is_valid(self) -> bool:
        """Valida que la señal tiene precios coherentes."""
        if self.entry_price <= 0 or self.stop_loss <= 0 or self.take_profit_1 <= 0:
            return False
        risk = abs(self.entry_price - self.stop_loss)
        if risk <= 0:
            return False
        # Para longs: SL < entry < TP1
        if self.direction == SignalDirection.LONG:
            return self.stop_loss < self.entry_price < self.take_profit_1
        # Para shorts: TP1 < entry < SL
        elif self.direction == SignalDirection.SHORT:
            return self.take_profit_1 < self.entry_price < self.stop_loss
        return False

    def to_dict(self) -> dict:
        """Serializa la señal para el journal."""
        return {
            "strategy": self.strategy.value,
            "direction": self.direction.name,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit_1": self.take_profit_1,
            "take_profit_2": self.take_profit_2,
            "quality": self.quality.value,
            "confidence": self.confidence,
            "entry_reason": self.entry_reason,
            "conditions_met": self.conditions_met,
            "risk_reward_tp1": round(self.risk_reward_ratio_tp1, 2),
            "market_regime": self.market_regime,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


class BaseStrategy(ABC):
    """
    Clase base abstracta para todas las estrategias.
    Hereda e implementa los métodos abstractos para crear una estrategia nueva.
    """

    def __init__(self, symbol: str, timeframe: str = "4h"):
        self.symbol = symbol
        self.timeframe = timeframe

    @property
    @abstractmethod
    def strategy_type(self) -> StrategyType:
        """Tipo de estrategia."""
        ...

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analiza el DataFrame y genera señales de trading.

        Args:
            df: DataFrame OHLCV con indicadores ya calculados

        Returns:
            DataFrame con columnas adicionales:
              - signal: 1 (long), -1 (short), 0 (flat)
              - entry_price: precio de entrada sugerido
              - stop_loss: stop loss calculado
              - take_profit_1: primer target
              - take_profit_2: segundo target (opcional)
              - signal_quality: SetupQuality
              - entry_reason: string descriptivo
        """
        ...

    @abstractmethod
    def is_valid_entry(self, df: pd.DataFrame, idx: int) -> bool:
        """
        Verifica si en el índice dado se cumplen las condiciones de entrada.
        Usado por el backtesting engine para verificar señal por señal.
        """
        ...

    def prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula todos los indicadores necesarios para la estrategia.
        Llamar antes de generate_signals.
        """
        from indicators.trend import TrendIndicators
        from indicators.momentum import MomentumIndicators
        from indicators.volatility import VolatilityIndicators
        from indicators.volume import VolumeIndicators
        from indicators.market_structure import MarketStructureIndicators

        trend = TrendIndicators()
        momentum = MomentumIndicators()
        volatility = VolatilityIndicators()
        volume = VolumeIndicators()
        structure = MarketStructureIndicators()

        df = trend.calculate_all(df)
        df = momentum.calculate_all(df)
        df = volatility.calculate_all(df)
        df = volume.calculate_all(df)
        df = structure.calculate_all(df)

        return df

    def _safe_get(self, df: pd.DataFrame, idx: int, col: str, default=None):
        """Acceso seguro a una columna del DataFrame por índice."""
        if col not in df.columns or idx >= len(df):
            return default
        val = df[col].iloc[idx]
        if pd.isna(val):
            return default
        return val

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(symbol={self.symbol}, tf={self.timeframe})"
