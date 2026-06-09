"""
Estrategia 2: Mean Reversion (30% del tiempo operativo)
========================================================
Opera contra la tendencia a corto plazo cuando el mercado está en rango.

Condiciones de entrada:
  1. Mercado en rango: ADX < 20, BB estrecho (relativo)
  2. RSI < 35 (para long) o > 65 (para short)
  3. Precio tocando banda inferior/superior de BB
  4. Soporte/resistencia clave coincidente
  5. Divergencia en Stoch RSI

Stop Loss: cierre de vela fuera de la banda BB + 0.5 ATR
Take Profit: EMA 21 (TP1), banda opuesta BB (TP2)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import (
    INDICATORS,
    STRATEGIES,
    MeanReversionParams,
    SetupQuality,
    SignalDirection,
    StrategyType,
)
from strategies.base import BaseStrategy, Signal


class MeanReversionStrategy(BaseStrategy):
    """Mean Reversion — Estrategia 2."""

    def __init__(
        self,
        symbol: str,
        timeframe: str = "4h",
        params: MeanReversionParams = STRATEGIES.mean_reversion,
    ):
        super().__init__(symbol, timeframe)
        self.p = params
        self._ema_fast = INDICATORS.trend.ema_fast  # 21
        
        # Optimización V3: Forzar parámetros exactos
        self.p.max_adx = 20.0
        self.p.rsi_long_threshold = 35.0
        self.p.rsi_short_threshold = 65.0
        self.p.sl_atr_buffer = 1.5  # Respiración vital (1.5 ATR)

    @property
    def strategy_type(self) -> StrategyType:
        return StrategyType.MEAN_REVERSION

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Genera señales de Mean Reversion para todo el DataFrame."""
        df = df.copy()
        n = len(df)

        signals = np.zeros(n, dtype=int)
        entry_prices = np.full(n, np.nan)
        stop_losses = np.full(n, np.nan)
        tp1_prices = np.full(n, np.nan)
        tp2_prices = np.full(n, np.nan)
        signal_reasons = [""] * n
        signal_quality = [""] * n

        # Optimización V3: Macro Filter
        # 1D EMA55 = 1H EMA(1320)
        # 1D EMA200 = 1H EMA(4800)
        # Nota: Asumiendo que ejecutamos en 1H
        ema_55_1d = df["close"].ewm(span=1320, adjust=False).mean()
        ema_200_1d = df["close"].ewm(span=4800, adjust=False).mean()

        for i in range(4800, n):  # Necesitamos warmup de 4800 velas para la EMA200 diaria
            # Optimización V3: Solo Longs permitidos en Macro Bull Market
            macro_bull = ema_55_1d.iloc[i] > ema_200_1d.iloc[i]
            
            long_signal = False
            if macro_bull:
                long_signal, long_reason, long_quality = self._check_long_conditions(df, i)
            
            short_signal = False  # Optimización V3: Deshabilitar cortos

            if long_signal:
                entry = df["close"].iloc[i]
                atr = df["atr"].iloc[i] if not pd.isna(df["atr"].iloc[i]) else entry * 0.02
                bb_lower = df["bb_lower"].iloc[i]
                ema21 = df[f"ema_{self._ema_fast}"].iloc[i]
                bb_upper = df["bb_upper"].iloc[i]

                sl = bb_lower - self.p.sl_atr_buffer * atr
                tp1 = ema21
                tp2 = bb_upper  # Banda opuesta

                if tp1 > entry and tp2 > entry:
                    signals[i] = 1
                    entry_prices[i] = entry
                    stop_losses[i] = sl
                    tp1_prices[i] = tp1
                    tp2_prices[i] = tp2
                    signal_reasons[i] = long_reason
                    signal_quality[i] = long_quality.value

            elif short_signal:
                entry = df["close"].iloc[i]
                atr = df["atr"].iloc[i] if not pd.isna(df["atr"].iloc[i]) else entry * 0.02
                bb_upper = df["bb_upper"].iloc[i]
                ema21 = df[f"ema_{self._ema_fast}"].iloc[i]
                bb_lower = df["bb_lower"].iloc[i]

                sl = bb_upper + self.p.sl_atr_buffer * atr
                tp1 = ema21
                tp2 = bb_lower

                if tp1 < entry and tp2 < entry:
                    signals[i] = -1
                    entry_prices[i] = entry
                    stop_losses[i] = sl
                    tp1_prices[i] = tp1
                    tp2_prices[i] = tp2
                    signal_reasons[i] = short_reason
                    signal_quality[i] = short_quality.value

        df["signal"] = signals
        df["entry_price"] = entry_prices
        df["stop_loss"] = stop_losses
        df["take_profit_1"] = tp1_prices
        df["take_profit_2"] = tp2_prices
        df["signal_reason"] = signal_reasons
        df["signal_quality"] = signal_quality
        df["strategy"] = self.strategy_type.value

        return df

    def _check_long_conditions(
        self, df: pd.DataFrame, idx: int
    ) -> tuple[bool, str, SetupQuality]:
        """Verifica condiciones de entrada long en Mean Reversion."""
        row = df.iloc[idx]
        conditions = []
        required = ["adx", "rsi", "bb_lower", "bb_upper", "stoch_rsi_k", "atr",
                    f"ema_{self._ema_fast}"]

        for col in required:
            if col not in df.columns or pd.isna(row.get(col, float("nan"))):
                return False, f"Columna faltante: {col}", SetupQuality.C

        # 1. Rango: ADX < 20
        adx = row["adx"]
        if adx >= self.p.max_adx:
            return False, f"ADX={adx:.1f} (>20, no es rango)", SetupQuality.C
        conditions.append(f"ADX={adx:.1f}<20✓")

        # 2. RSI < 35
        rsi = row["rsi"]
        if rsi >= self.p.rsi_long_threshold:
            return False, f"RSI={rsi:.1f} no en sobreventa (<35)", SetupQuality.C
        conditions.append(f"RSI={rsi:.1f}<35✓")

        # 3. Precio tocando banda inferior BB
        close = row["close"]
        bb_lower = row["bb_lower"]
        atr = row["atr"]
        if close > bb_lower + 0.5 * atr:
            return False, "Precio no en banda inferior BB", SetupQuality.C
        conditions.append("Precio_en_BB_lower✓")

        # 4. Soporte clave: Order Block alcista o swing low próximo
        in_support = (
            row.get("price_in_bull_ob", False)
            or row.get("liquidity_below", False)
            or (abs(close - row.get("last_swing_low", close)) < atr)
        )
        if in_support:
            conditions.append("Soporte_coincidente✓")

        # 5. Divergencia Stoch RSI
        stoch_div = row.get("stoch_rsi_bullish_divergence", False)
        stoch_oversold = row.get("stoch_rsi_oversold", False)
        stoch_cross = row.get("stoch_rsi_bullish_cross", False)

        if stoch_div or (stoch_oversold and stoch_cross):
            conditions.append("StochRSI_divergencia✓")
        elif stoch_oversold:
            conditions.append("StochRSI_sobreventa~")
        else:
            # Condición débil pero no bloqueante
            pass

        quality = self._classify_long_quality(row, len(conditions))
        return True, " | ".join(conditions), quality

    def _check_short_conditions(
        self, df: pd.DataFrame, idx: int
    ) -> tuple[bool, str, SetupQuality]:
        """Verifica condiciones de entrada short en Mean Reversion."""
        row = df.iloc[idx]
        conditions = []
        required = ["adx", "rsi", "bb_upper", "stoch_rsi_k", "atr"]

        for col in required:
            if col not in df.columns or pd.isna(row.get(col, float("nan"))):
                return False, f"Columna faltante: {col}", SetupQuality.C

        # 1. Rango: ADX < 20
        adx = row["adx"]
        if adx >= self.p.max_adx:
            return False, f"ADX={adx:.1f} (>20)", SetupQuality.C
        conditions.append(f"ADX={adx:.1f}<20✓")

        # 2. RSI > 65
        rsi = row["rsi"]
        if rsi <= self.p.rsi_short_threshold:
            return False, f"RSI={rsi:.1f} no en sobrecompra (>65)", SetupQuality.C
        conditions.append(f"RSI={rsi:.1f}>65✓")

        # 3. Precio tocando banda superior BB
        close = row["close"]
        bb_upper = row["bb_upper"]
        atr = row["atr"]
        if close < bb_upper - 0.5 * atr:
            return False, "Precio no en banda superior BB", SetupQuality.C
        conditions.append("Precio_en_BB_upper✓")

        # 4. Resistencia clave
        in_resistance = (
            row.get("price_in_bear_ob", False)
            or row.get("liquidity_above", False)
        )
        if in_resistance:
            conditions.append("Resistencia_coincidente✓")

        # 5. Divergencia Stoch RSI bajista
        stoch_div = row.get("stoch_rsi_bearish_divergence", False)
        stoch_overbought = row.get("stoch_rsi_overbought", False)
        stoch_cross = row.get("stoch_rsi_bearish_cross", False)

        if stoch_div or (stoch_overbought and stoch_cross):
            conditions.append("StochRSI_div_bearish✓")

        quality = self._classify_short_quality(row, len(conditions))
        return True, " | ".join(conditions), quality

    def _classify_long_quality(self, row, n_conditions: int) -> SetupQuality:
        bonus = 0
        if row.get("stoch_rsi_bullish_divergence", False):
            bonus += 2
        if row.get("rsi_divergence_bullish", False):
            bonus += 1
        if row.get("obv_divergence_bullish", False):
            bonus += 1
        total = n_conditions + bonus
        if total >= 6:
            return SetupQuality.A_PLUS
        elif total >= 4:
            return SetupQuality.A
        elif total >= 3:
            return SetupQuality.B
        return SetupQuality.C

    def _classify_short_quality(self, row, n_conditions: int) -> SetupQuality:
        bonus = 0
        if row.get("stoch_rsi_bearish_divergence", False):
            bonus += 2
        if row.get("rsi_divergence_bearish", False):
            bonus += 1
        total = n_conditions + bonus
        if total >= 6:
            return SetupQuality.A_PLUS
        elif total >= 4:
            return SetupQuality.A
        elif total >= 3:
            return SetupQuality.B
        return SetupQuality.C

    def is_valid_entry(self, df: pd.DataFrame, idx: int) -> bool:
        long_valid, _, _ = self._check_long_conditions(df, idx)
        short_valid, _, _ = self._check_short_conditions(df, idx)
        return long_valid or short_valid
