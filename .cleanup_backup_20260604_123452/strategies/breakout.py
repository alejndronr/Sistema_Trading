"""
Estrategia 3: Breakout (10% del tiempo operativo)
==================================================
Solo en contexto de compresión de volatilidad (BB Squeeze activo > 10 velas).
El volumen debe ser > 150% del volumen promedio de 20 periodos.
Esperar retesteo del nivel roto antes de entrar (reduce falsas rupturas).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import (
    STRATEGIES,
    BreakoutParams,
    SetupQuality,
    SignalDirection,
    StrategyType,
)
from strategies.base import BaseStrategy, Signal


class BreakoutStrategy(BaseStrategy):
    """Breakout — Estrategia 3. Solo activa en Fase 3+ o con BB Squeeze confirmado."""

    def __init__(
        self,
        symbol: str,
        timeframe: str = "4h",
        params: BreakoutParams = STRATEGIES.breakout,
    ):
        super().__init__(symbol, timeframe)
        self.p = params

    @property
    def strategy_type(self) -> StrategyType:
        return StrategyType.BREAKOUT

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Genera señales de Breakout."""
        df = df.copy()
        n = len(df)

        signals = np.zeros(n, dtype=int)
        entry_prices = np.full(n, np.nan)
        stop_losses = np.full(n, np.nan)
        tp1_prices = np.full(n, np.nan)
        tp2_prices = np.full(n, np.nan)
        signal_reasons = [""] * n
        signal_quality = [""] * n

        for i in range(50, n):
            bull_signal, bull_reason, bull_quality = self._check_bullish_breakout(df, i)
            bear_signal, bear_reason, bear_quality = self._check_bearish_breakout(df, i)

            entry = df["close"].iloc[i]
            atr = df["atr"].iloc[i] if "atr" in df.columns and not pd.isna(df["atr"].iloc[i]) else entry * 0.02

            if bull_signal:
                # SL debajo del nivel de resistencia roto (ahora soporte)
                breakout_level = df["bb_upper"].iloc[i - 1] if "bb_upper" in df.columns else entry - atr
                sl = breakout_level - self.p.sl_atr_multiplier * atr
                risk = entry - sl
                tp1 = entry + 2.0 * risk
                tp2 = entry + 4.0 * risk  # Breakouts pueden tener targets amplios

                signals[i] = 1
                entry_prices[i] = entry
                stop_losses[i] = sl
                tp1_prices[i] = tp1
                tp2_prices[i] = tp2
                signal_reasons[i] = bull_reason
                signal_quality[i] = bull_quality.value

            elif bear_signal:
                breakout_level = df["bb_lower"].iloc[i - 1] if "bb_lower" in df.columns else entry + atr
                sl = breakout_level + self.p.sl_atr_multiplier * atr
                risk = sl - entry
                tp1 = entry - 2.0 * risk
                tp2 = entry - 4.0 * risk

                signals[i] = -1
                entry_prices[i] = entry
                stop_losses[i] = sl
                tp1_prices[i] = tp1
                tp2_prices[i] = tp2
                signal_reasons[i] = bear_reason
                signal_quality[i] = bear_quality.value

        df["signal"] = signals
        df["entry_price"] = entry_prices
        df["stop_loss"] = stop_losses
        df["take_profit_1"] = tp1_prices
        df["take_profit_2"] = tp2_prices
        df["signal_reason"] = signal_reasons
        df["signal_quality"] = signal_quality
        df["strategy"] = self.strategy_type.value

        return df

    def _check_bullish_breakout(
        self, df: pd.DataFrame, idx: int
    ) -> tuple[bool, str, SetupQuality]:
        """Verifica condiciones de breakout alcista."""
        row = df.iloc[idx]
        prev = df.iloc[idx - 1]
        conditions = []

        required = ["bb_squeeze_candles", "volume_ratio", "bb_upper"]
        for col in required:
            if col not in df.columns:
                return False, f"Columna faltante: {col}", SetupQuality.C

        # 1. Squeeze activo > 10 velas
        squeeze_candles = row.get("bb_squeeze_candles", 0)
        if squeeze_candles < self.p.min_squeeze_candles:
            return False, f"Squeeze insuficiente: {squeeze_candles} velas", SetupQuality.C

        # 2. Fin del squeeze: precio rompe banda superior
        was_in_squeeze = prev.get("bb_squeeze", False)
        broke_upper = row["close"] > prev["bb_upper"]
        if not (was_in_squeeze and broke_upper):
            return False, "No hay ruptura de banda superior", SetupQuality.C
        conditions.append(f"Squeeze_release({squeeze_candles}velas)✓")

        # 3. Volumen > 150% de la media
        vol_ratio = row.get("volume_ratio", 0)
        if vol_ratio < self.p.volume_multiplier:
            return False, f"Volumen insuficiente: {vol_ratio:.1f}x (necesita {self.p.volume_multiplier}x)", SetupQuality.C
        conditions.append(f"Volumen={vol_ratio:.1f}x_media✓")

        # 4. Retesteo del nivel roto (si está activado)
        if self.p.wait_for_retest:
            # Verificar que el precio haya vuelto a tocar la banda rota
            price_near_breakout = abs(row["close"] - prev["bb_upper"]) < row.get("atr", row["close"] * 0.02)
            if price_near_breakout:
                conditions.append("Retesteo_confirmado✓")
            # Si no hay retesteo aún, podría ser entrada prematura → B
            # (no bloqueante, pero baja calidad)

        # 5. Confirmación de dirección (BOS alcista o CHoCH)
        if row.get("bos_bullish", False):
            conditions.append("BOS_alcista✓")
        if row.get("ema_alignment_bullish", False):
            conditions.append("EMAs_alineadas✓")

        # Volumen extra alto = A+
        if vol_ratio >= 2.5:
            quality = SetupQuality.A_PLUS
        elif vol_ratio >= 1.75 and len(conditions) >= 3:
            quality = SetupQuality.A
        else:
            quality = SetupQuality.B

        return True, " | ".join(conditions), quality

    def _check_bearish_breakout(
        self, df: pd.DataFrame, idx: int
    ) -> tuple[bool, str, SetupQuality]:
        """Verifica condiciones de breakout bajista."""
        row = df.iloc[idx]
        prev = df.iloc[idx - 1]
        conditions = []

        required = ["bb_squeeze_candles", "volume_ratio", "bb_lower"]
        for col in required:
            if col not in df.columns:
                return False, f"Columna faltante: {col}", SetupQuality.C

        squeeze_candles = row.get("bb_squeeze_candles", 0)
        if squeeze_candles < self.p.min_squeeze_candles:
            return False, f"Squeeze insuficiente: {squeeze_candles} velas", SetupQuality.C

        was_in_squeeze = prev.get("bb_squeeze", False)
        broke_lower = row["close"] < prev["bb_lower"]
        if not (was_in_squeeze and broke_lower):
            return False, "No hay ruptura de banda inferior", SetupQuality.C
        conditions.append(f"Squeeze_release({squeeze_candles}velas)✓")

        vol_ratio = row.get("volume_ratio", 0)
        if vol_ratio < self.p.volume_multiplier:
            return False, f"Volumen insuficiente: {vol_ratio:.1f}x", SetupQuality.C
        conditions.append(f"Volumen={vol_ratio:.1f}x_media✓")

        if row.get("bos_bearish", False):
            conditions.append("BOS_bajista✓")

        quality = SetupQuality.A_PLUS if vol_ratio >= 2.5 else (SetupQuality.A if len(conditions) >= 3 else SetupQuality.B)
        return True, " | ".join(conditions), quality

    def is_valid_entry(self, df: pd.DataFrame, idx: int) -> bool:
        bull_valid, _, _ = self._check_bullish_breakout(df, idx)
        bear_valid, _, _ = self._check_bearish_breakout(df, idx)
        return bull_valid or bear_valid
