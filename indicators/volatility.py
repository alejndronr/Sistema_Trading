"""
Indicadores de Volatilidad
==========================
ATR, Bollinger Bands, Keltner Channels, BB Squeeze

Convención de columnas:
  - atr: ATR(14)
  - bb_upper, bb_middle, bb_lower, bb_width, bb_pct
  - kc_upper, kc_middle, kc_lower
  - bb_squeeze: True cuando BB está dentro de los Keltner Channels
  - bb_squeeze_candles: número de velas consecutivas en squeeze
  - atr_regime: 'normal' | 'high_vol' | 'extreme_vol'
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import INDICATORS, VolatilityIndicatorParams


class VolatilityIndicators:
    """ATR, Bollinger Bands, Keltner Channels y Squeeze."""

    def __init__(self, params: VolatilityIndicatorParams = INDICATORS.volatility):
        self.p = params

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula todos los indicadores de volatilidad."""
        df = self.atr(df)
        df = self.bollinger_bands(df)
        df = self.keltner_channels(df)
        df = self.bb_squeeze(df)
        df = self.atr_regime(df)
        return df

    # ── ATR ───────────────────────────────────────────────────────────────────

    def atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """ATR (14) con suavizado Wilder (RMA)."""
        df = df.copy()
        period = self.p.atr_period
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr = pd.concat(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)

        df["atr"] = tr.ewm(alpha=1 / period, adjust=False).mean()
        # ATR como porcentaje del precio (para comparar entre activos)
        df["atr_pct"] = df["atr"] / df["close"] * 100
        return df

    # ── Bollinger Bands ───────────────────────────────────────────────────────

    def bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Bollinger Bands (20, 2).
        bb_width:   ancho relativo de las bandas (volatilidad)
        bb_pct:     posición del precio dentro de las bandas (0-1)
        """
        df = df.copy()
        period = self.p.bb_period
        std_mult = self.p.bb_std
        close = df["close"]

        middle = close.rolling(period).mean()
        std = close.rolling(period).std()

        df["bb_middle"] = middle
        df["bb_upper"] = middle + std_mult * std
        df["bb_lower"] = middle - std_mult * std
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]

        # %B: 0 = en banda inferior, 1 = en banda superior
        bb_range = df["bb_upper"] - df["bb_lower"]
        df["bb_pct"] = (close - df["bb_lower"]) / bb_range.replace(0, np.nan)

        # Señales de toque de banda
        df["bb_touch_upper"] = close >= df["bb_upper"] * 0.999
        df["bb_touch_lower"] = close <= df["bb_lower"] * 1.001
        return df

    # ── Keltner Channels ──────────────────────────────────────────────────────

    def keltner_channels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Keltner Channels (20, 2x ATR).
        Se usa junto con BB para detectar el Squeeze de volatilidad.
        """
        df = df.copy()
        period = self.p.keltner_period
        factor = self.p.keltner_factor
        atr_period = self.p.keltner_atr_period
        close = df["close"]

        # EMA como línea central (no SMA)
        middle = close.ewm(span=period, adjust=False).mean()

        # ATR para las bandas
        high = df["high"]
        low = df["low"]
        tr = pd.concat(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        kc_atr = tr.ewm(alpha=1 / atr_period, adjust=False).mean()

        df["kc_middle"] = middle
        df["kc_upper"] = middle + factor * kc_atr
        df["kc_lower"] = middle - factor * kc_atr
        return df

    # ── BB Squeeze ────────────────────────────────────────────────────────────

    def bb_squeeze(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detecta el BB Squeeze: cuando las Bollinger Bands están dentro
        de los Keltner Channels → compresión de volatilidad → breakout inminente.

        bb_squeeze_candles: número de velas consecutivas en squeeze.
        Si > squeeze_min_candles → activar Estrategia 3.
        """
        df = df.copy()
        required = ["bb_upper", "bb_lower", "kc_upper", "kc_lower"]
        for col in required:
            if col not in df.columns:
                df = self.bollinger_bands(df) if "bb_upper" not in df.columns else df
                df = self.keltner_channels(df) if "kc_upper" not in df.columns else df

        # Squeeze: BB completamente dentro de KC
        df["bb_squeeze"] = (df["bb_upper"] < df["kc_upper"]) & (
            df["bb_lower"] > df["kc_lower"]
        )

        # Contar velas consecutivas en squeeze
        squeeze_count = []
        count = 0
        for is_squeeze in df["bb_squeeze"]:
            if is_squeeze:
                count += 1
            else:
                count = 0
            squeeze_count.append(count)

        df["bb_squeeze_candles"] = squeeze_count
        df["bb_squeeze_ready"] = (
            df["bb_squeeze_candles"] >= self.p.squeeze_min_candles
        )

        # Momento de la explosión: fin del squeeze
        prev_squeeze = df["bb_squeeze"].shift(1)
        df["bb_squeeze_release"] = (~df["bb_squeeze"]) & prev_squeeze.fillna(False)
        return df

    # ── ATR Regime ────────────────────────────────────────────────────────────

    def atr_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clasifica el régimen de volatilidad según el ATR vs su media histórica.
        'normal':       ATR < 1.5x su media de 14 periodos
        'high_vol':     ATR entre 1.5x y 2x → reducir posiciones al 50%
        'extreme_vol':  ATR > 3x → NO OPERAR ese activo
        """
        df = df.copy()
        if "atr" not in df.columns:
            df = self.atr(df)

        atr_mean = df["atr"].rolling(14).mean()
        atr_ratio = df["atr"] / atr_mean.replace(0, np.nan)

        conditions = [
            atr_ratio <= 1.5,
            (atr_ratio > 1.5) & (atr_ratio <= 2.0),
            (atr_ratio > 2.0) & (atr_ratio <= 3.0),
        ]
        choices = ["normal", "high_vol", "very_high_vol"]
        df["atr_regime"] = np.select(conditions, choices, default="extreme_vol")
        df["atr_ratio"] = atr_ratio
        df["atr_tradeable"] = atr_ratio <= 3.0  # False → no operar

        return df

    def dynamic_stop_loss(
        self, entry_price: float, atr: float, direction: int, multiplier: float = 1.0
    ) -> float:
        """
        Calcula un Stop Loss dinámico basado en ATR.
        direction: 1 = long (SL abajo), -1 = short (SL arriba)
        """
        if direction == 1:  # Long
            return entry_price - multiplier * atr
        else:  # Short
            return entry_price + multiplier * atr

    def dynamic_take_profit(
        self, entry_price: float, stop_loss: float, rr_ratio: float, direction: int
    ) -> float:
        """
        Calcula Take Profit con un ratio R/R dado.
        rr_ratio: ej. 2.0 = 2:1 R/R
        """
        risk = abs(entry_price - stop_loss)
        if direction == 1:  # Long
            return entry_price + rr_ratio * risk
        else:  # Short
            return entry_price - rr_ratio * risk
