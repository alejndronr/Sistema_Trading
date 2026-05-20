"""
Indicadores de Tendencia
========================
EMA 21, EMA 55, EMA 200, Supertrend, ADX

Todos los métodos son estáticos y reciben un DataFrame con columnas OHLCV.
Retornan el mismo DataFrame con columnas adicionales de indicadores.
Convención de nombres de columnas:
  - ema_21, ema_55, ema_200
  - supertrend, supertrend_direction  (1=alcista, -1=bajista)
  - adx, adx_plus_di, adx_minus_di
  - ema_alignment  (True si EMA21 > EMA55 > EMA200)
  - trend_regime   ('bullish' | 'bearish' | 'ranging')
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import INDICATORS, TrendIndicatorParams


class TrendIndicators:
    """Indicadores de tendencia: EMA, Supertrend, ADX."""

    def __init__(self, params: TrendIndicatorParams = INDICATORS.trend):
        self.p = params

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula todos los indicadores de tendencia en una pasada."""
        df = self.ema(df)
        df = self.supertrend(df)
        df = self.adx(df)
        df = self.ema_alignment(df)
        df = self.trend_regime(df)
        return df

    # ── EMA ───────────────────────────────────────────────────────────────────

    def ema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula EMA 21, 55 y 200."""
        close = df["close"]
        df = df.copy()
        df[f"ema_{self.p.ema_fast}"] = (
            close.ewm(span=self.p.ema_fast, adjust=False).mean()
        )
        df[f"ema_{self.p.ema_mid}"] = (
            close.ewm(span=self.p.ema_mid, adjust=False).mean()
        )
        df[f"ema_{self.p.ema_slow}"] = (
            close.ewm(span=self.p.ema_slow, adjust=False).mean()
        )
        return df

    # ── Supertrend ────────────────────────────────────────────────────────────

    def supertrend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Supertrend (ATR 10, factor 3.0).
        Dirección: 1 = precio sobre Supertrend (alcista), -1 = bajista.
        """
        df = df.copy()
        period = self.p.supertrend_atr_period
        factor = self.p.supertrend_factor

        # ATR para Supertrend
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
        atr = tr.ewm(span=period, adjust=False).mean()

        # Bandas básicas
        hl2 = (high + low) / 2
        upper_band = hl2 + factor * atr
        lower_band = hl2 - factor * atr

        # Supertrend con lógica de cambio de dirección
        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=int)

        # Inicializar
        supertrend.iloc[0] = upper_band.iloc[0]
        direction.iloc[0] = -1

        for i in range(1, len(df)):
            prev_upper = upper_band.iloc[i - 1]
            prev_lower = lower_band.iloc[i - 1]
            curr_upper = upper_band.iloc[i]
            curr_lower = lower_band.iloc[i]
            prev_close = close.iloc[i - 1]
            curr_close = close.iloc[i]

            # Ajustar bandas (no permitir que retroceda)
            final_upper = curr_upper if curr_upper < prev_upper or prev_close > prev_upper else prev_upper
            final_lower = curr_lower if curr_lower > prev_lower or prev_close < prev_lower else prev_lower

            prev_st = supertrend.iloc[i - 1]
            prev_dir = direction.iloc[i - 1]

            if prev_st == prev_upper:
                direction.iloc[i] = 1 if curr_close > final_upper else -1
            else:
                direction.iloc[i] = -1 if curr_close < final_lower else 1

            supertrend.iloc[i] = final_lower if direction.iloc[i] == 1 else final_upper

        df["supertrend"] = supertrend
        df["supertrend_direction"] = direction  # 1=alcista, -1=bajista
        return df

    # ── ADX ───────────────────────────────────────────────────────────────────

    def adx(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ADX (14 periodos) con +DI y -DI.
        ADX > 20: tendencia presente.
        ADX > 25: tendencia fuerte.
        ADX > 40: tendencia muy fuerte.
        """
        df = df.copy()
        period = self.p.adx_period
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # True Range
        tr = pd.concat(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)

        # Directional Movement
        dm_plus = high.diff()
        dm_minus = -low.diff()

        dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0.0)
        dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0.0)

        # Smoothed (Wilder's smoothing)
        atr_s = tr.ewm(alpha=1 / period, adjust=False).mean()
        di_plus = 100 * dm_plus.ewm(alpha=1 / period, adjust=False).mean() / atr_s
        di_minus = 100 * dm_minus.ewm(alpha=1 / period, adjust=False).mean() / atr_s

        # ADX
        dx = (100 * (di_plus - di_minus).abs() / (di_plus + di_minus)).replace(
            [np.inf, -np.inf], np.nan
        )
        adx = dx.ewm(alpha=1 / period, adjust=False).mean()

        df["adx"] = adx
        df["adx_plus_di"] = di_plus
        df["adx_minus_di"] = di_minus
        return df

    # ── Señales derivadas ─────────────────────────────────────────────────────

    def ema_alignment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Verifica alineación de EMAs.
        ema_alignment_bullish: EMA21 > EMA55 > EMA200 (alcista)
        ema_alignment_bearish: EMA21 < EMA55 < EMA200 (bajista)
        """
        df = df.copy()
        f, m, s = f"ema_{self.p.ema_fast}", f"ema_{self.p.ema_mid}", f"ema_{self.p.ema_slow}"

        if not all(col in df.columns for col in [f, m, s]):
            df = self.ema(df)

        df["ema_alignment_bullish"] = (df[f] > df[m]) & (df[m] > df[s])
        df["ema_alignment_bearish"] = (df[f] < df[m]) & (df[m] < df[s])
        return df

    def trend_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clasifica el régimen de tendencia por vela.
        'bullish'  → EMAs alcistas + ADX > umbral
        'bearish'  → EMAs bajistas + ADX > umbral
        'ranging'  → ADX < umbral de tendencia
        """
        df = df.copy()
        required = ["adx", "ema_alignment_bullish", "ema_alignment_bearish"]
        for col in required:
            if col not in df.columns:
                df = self.calculate_all(df)
                break

        conditions = [
            df["ema_alignment_bullish"] & (df["adx"] > self.p.adx_trend_threshold),
            df["ema_alignment_bearish"] & (df["adx"] > self.p.adx_trend_threshold),
        ]
        choices = ["bullish", "bearish"]
        df["trend_regime"] = np.select(conditions, choices, default="ranging")
        return df

    def is_strong_trend(self, df: pd.DataFrame) -> pd.Series:
        """Retorna True si hay tendencia fuerte (ADX > 25)."""
        if "adx" not in df.columns:
            df = self.adx(df)
        return df["adx"] > self.p.adx_strong_threshold

    def price_above_ema21(self, df: pd.DataFrame) -> pd.Series:
        """Condición: precio de cierre sobre EMA 21."""
        col = f"ema_{self.p.ema_fast}"
        if col not in df.columns:
            df = self.ema(df)
        return df["close"] > df[col]

    def retrace_to_ema21(self, df: pd.DataFrame, tolerance_atr: float = 0.5) -> pd.Series:
        """
        Detecta retroceso a la EMA 21 (condición de entrada Estrategia 1).
        True cuando el low toca ± tolerance_atr * ATR de la EMA 21.
        """
        col = f"ema_{self.p.ema_fast}"
        if col not in df.columns:
            df = self.ema(df)

        # ATR aproximado: rango promedio de las últimas 14 velas
        atr = (df["high"] - df["low"]).rolling(14).mean()
        lower_band = df[col] - tolerance_atr * atr
        upper_band = df[col] + tolerance_atr * atr

        return (df["low"] <= upper_band) & (df["high"] >= lower_band)
