"""
Indicadores de Momentum
=======================
RSI, RSI Divergencias, MACD, Stochastic RSI

Convención de columnas:
  - rsi: RSI(14)
  - rsi_divergence_bullish / rsi_divergence_bearish
  - macd_line, macd_signal, macd_histogram
  - macd_bullish_cross / macd_bearish_cross
  - stoch_rsi_k, stoch_rsi_d
  - stoch_rsi_oversold / stoch_rsi_overbought
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import INDICATORS, MomentumIndicatorParams


class MomentumIndicators:
    """RSI, MACD, Stochastic RSI y sus señales derivadas."""

    def __init__(self, params: MomentumIndicatorParams = INDICATORS.momentum):
        self.p = params

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula todos los indicadores de momentum."""
        df = self.rsi(df)
        df = self.rsi_divergences(df)
        df = self.macd(df)
        df = self.stochastic_rsi(df)
        return df

    # ── RSI ───────────────────────────────────────────────────────────────────

    def rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """RSI (14) con suavizado Wilder."""
        df = df.copy()
        period = self.p.rsi_period
        delta = df["close"].diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        # Wilder's smoothing (equivalente a EWM con alpha=1/period)
        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))
        df["rsi"] = df["rsi"].fillna(50)  # Neutral cuando no hay datos

        # Señales derivadas
        df["rsi_overbought"] = df["rsi"] > self.p.rsi_overbought
        df["rsi_oversold"] = df["rsi"] < self.p.rsi_oversold
        df["rsi_neutral"] = (df["rsi"] >= self.p.rsi_neutral_low) & (
            df["rsi"] <= self.p.rsi_neutral_high
        )
        return df

    # ── RSI Divergencias ──────────────────────────────────────────────────────

    def rsi_divergences(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detecta divergencias regulares y ocultas del RSI.

        Divergencia Bullish Regular:  precio hace lower low, RSI hace higher low
        Divergencia Bearish Regular:  precio hace higher high, RSI hace lower high
        Divergencia Bullish Oculta:   precio hace higher low, RSI hace lower low
        Divergencia Bearish Oculta:   precio hace lower high, RSI hace higher high
        """
        df = df.copy()
        if "rsi" not in df.columns:
            df = self.rsi(df)

        lookback = self.p.divergence_lookback
        n = len(df)

        bull_div = pd.Series(False, index=df.index)
        bear_div = pd.Series(False, index=df.index)
        bull_hidden = pd.Series(False, index=df.index)
        bear_hidden = pd.Series(False, index=df.index)

        for i in range(lookback, n):
            # Ventana de análisis
            window_close = df["close"].iloc[i - lookback : i + 1]
            window_rsi = df["rsi"].iloc[i - lookback : i + 1]
            window_low = df["low"].iloc[i - lookback : i + 1]
            window_high = df["high"].iloc[i - lookback : i + 1]

            curr_close = df["close"].iloc[i]
            curr_rsi = df["rsi"].iloc[i]
            prev_min_close = window_close.iloc[:-1].min()
            prev_max_close = window_close.iloc[:-1].max()
            prev_min_rsi = window_rsi.iloc[:-1].min()
            prev_max_rsi = window_rsi.iloc[:-1].max()
            curr_low = df["low"].iloc[i]
            curr_high = df["high"].iloc[i]
            prev_min_low = window_low.iloc[:-1].min()
            prev_max_high = window_high.iloc[:-1].max()

            # Bullish Regular: precio lower low + RSI higher low → probable reversión al alza
            if curr_low < prev_min_low and curr_rsi > prev_min_rsi:
                if curr_rsi < 50:  # Solo en zona de sobreventa relativa
                    bull_div.iloc[i] = True

            # Bearish Regular: precio higher high + RSI lower high → probable reversión a la baja
            if curr_high > prev_max_high and curr_rsi < prev_max_rsi:
                if curr_rsi > 50:  # Solo en zona de sobrecompra relativa
                    bear_div.iloc[i] = True

            # Bullish Oculta: precio higher low + RSI lower low → continuación alcista
            if curr_low > prev_min_low and curr_rsi < prev_min_rsi:
                bull_hidden.iloc[i] = True

            # Bearish Oculta: precio lower high + RSI higher high → continuación bajista
            if curr_high < prev_max_high and curr_rsi > prev_max_rsi:
                bear_hidden.iloc[i] = True

        df["rsi_divergence_bullish"] = bull_div
        df["rsi_divergence_bearish"] = bear_div
        df["rsi_divergence_bullish_hidden"] = bull_hidden
        df["rsi_divergence_bearish_hidden"] = bear_hidden
        return df

    # ── MACD ──────────────────────────────────────────────────────────────────

    def macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        MACD (12, 26, 9).
        macd_line:      EMA12 - EMA26
        macd_signal:    EMA9 del MACD
        macd_histogram: MACD - Signal
        """
        df = df.copy()
        close = df["close"]

        ema_fast = close.ewm(span=self.p.macd_fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.p.macd_slow, adjust=False).mean()

        df["macd_line"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd_line"].ewm(
            span=self.p.macd_signal, adjust=False
        ).mean()
        df["macd_histogram"] = df["macd_line"] - df["macd_signal"]

        # Señales derivadas
        df["macd_bullish"] = (df["macd_line"] > 0) & (df["macd_histogram"] > 0)
        df["macd_bearish"] = (df["macd_line"] < 0) & (df["macd_histogram"] < 0)
        df["macd_growing"] = df["macd_histogram"] > df["macd_histogram"].shift(1)

        # Crossovers
        prev_hist = df["macd_histogram"].shift(1)
        df["macd_bullish_cross"] = (prev_hist < 0) & (df["macd_histogram"] >= 0)
        df["macd_bearish_cross"] = (prev_hist > 0) & (df["macd_histogram"] <= 0)

        # Divergencias MACD-precio (simplificadas)
        price_higher_high = df["close"] > df["close"].rolling(10).max().shift(1)
        macd_lower_high = df["macd_line"] < df["macd_line"].rolling(10).max().shift(1)
        df["macd_divergence_bearish"] = price_higher_high & macd_lower_high

        price_lower_low = df["close"] < df["close"].rolling(10).min().shift(1)
        macd_higher_low = df["macd_line"] > df["macd_line"].rolling(10).min().shift(1)
        df["macd_divergence_bullish"] = price_lower_low & macd_higher_low

        return df

    # ── Stochastic RSI ────────────────────────────────────────────────────────

    def stochastic_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Stochastic RSI (3, 3, 14, 14).
        Aplica la fórmula del Stocástico sobre el RSI en vez del precio.
        Más sensible que el RSI clásico, ideal para timing fino de entradas.
        """
        df = df.copy()
        if "rsi" not in df.columns:
            df = self.rsi(df)

        rsi = df["rsi"]
        period = self.p.stoch_rsi_period
        smooth_k = self.p.stoch_rsi_smooth_k
        smooth_d = self.p.stoch_rsi_smooth_d

        rsi_min = rsi.rolling(period).min()
        rsi_max = rsi.rolling(period).max()
        rsi_range = rsi_max - rsi_min

        # Evitar división por cero cuando RSI no cambia
        stoch_rsi_raw = (rsi - rsi_min) / rsi_range.replace(0, np.nan)
        stoch_rsi_raw = stoch_rsi_raw.fillna(0.5) * 100

        # Suavizado K y D
        df["stoch_rsi_k"] = stoch_rsi_raw.rolling(smooth_k).mean()
        df["stoch_rsi_d"] = df["stoch_rsi_k"].rolling(smooth_d).mean()

        # Zonas extremas
        df["stoch_rsi_overbought"] = df["stoch_rsi_k"] > 80
        df["stoch_rsi_oversold"] = df["stoch_rsi_k"] < 20

        # Crossover K sobre D (señal de entrada)
        prev_k = df["stoch_rsi_k"].shift(1)
        prev_d = df["stoch_rsi_d"].shift(1)
        df["stoch_rsi_bullish_cross"] = (prev_k < prev_d) & (df["stoch_rsi_k"] >= df["stoch_rsi_d"])
        df["stoch_rsi_bearish_cross"] = (prev_k > prev_d) & (df["stoch_rsi_k"] <= df["stoch_rsi_d"])

        # Divergencia Stoch RSI (para Mean Reversion)
        df["stoch_rsi_bullish_divergence"] = (
            (df["close"] < df["close"].shift(5))
            & (df["stoch_rsi_k"] > df["stoch_rsi_k"].shift(5))
            & df["stoch_rsi_oversold"]
        )
        df["stoch_rsi_bearish_divergence"] = (
            (df["close"] > df["close"].shift(5))
            & (df["stoch_rsi_k"] < df["stoch_rsi_k"].shift(5))
            & df["stoch_rsi_overbought"]
        )

        return df
