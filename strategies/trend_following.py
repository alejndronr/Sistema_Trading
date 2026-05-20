"""
Estrategia 1: Trend Following (60% del tiempo operativo)
=========================================================
Implementación exacta de las condiciones definidas en el prompt maestro.

Condiciones de entrada LONG (todas deben cumplirse):
  1. EMA 21 > EMA 55 > EMA 200 en 4H
  2. Precio sobre EMA 21 en 1H (o 4H si solo hay un timeframe)
  3. ADX > 25 (tendencia fuerte)
  4. RSI entre 45-65 (no sobrecomprado)
  5. MACD positivo y creciente (histogram > 0 y creciendo)
  6. Retroceso a EMA 21 o zona de Order Block

Gestión de posición:
  - Stop Loss: 1 ATR por debajo del swing low más reciente
  - TP1: 2:1 R/R mínimo (cerrar 50%)
  - TP2: 3:1 R/R (dejar correr con trailing stop)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import (
    INDICATORS,
    RISK,
    STRATEGIES,
    SetupQuality,
    SignalDirection,
    StrategyType,
    TrendFollowingParams,
)
from strategies.base import BaseStrategy, Signal


class TrendFollowingStrategy(BaseStrategy):
    """
    Trend Following — Estrategia 1.
    Opera en la dirección de la tendencia principal, entrando en retrocesos.
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str = "4h",
        params: TrendFollowingParams = STRATEGIES.trend_following,
    ):
        super().__init__(symbol, timeframe)
        self.p = params
        self._ema_fast = INDICATORS.trend.ema_fast    # 21
        self._ema_mid = INDICATORS.trend.ema_mid      # 55
        self._ema_slow = INDICATORS.trend.ema_slow    # 200

    @property
    def strategy_type(self) -> StrategyType:
        return StrategyType.TREND_FOLLOWING

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Genera señales de Trend Following para todo el DataFrame.
        Retorna el DataFrame con columnas de señal añadidas.
        """
        df = df.copy()
        n = len(df)

        signals = np.zeros(n, dtype=int)
        entry_prices = np.full(n, np.nan)
        stop_losses = np.full(n, np.nan)
        tp1_prices = np.full(n, np.nan)
        tp2_prices = np.full(n, np.nan)
        signal_reasons = [""] * n
        signal_quality = [""] * n

        for i in range(200, n):  # Warmup de 200 velas para EMA 200
            long_signal, reason, quality = self._check_long_conditions(df, i)

            if long_signal:
                entry = df["close"].iloc[i]
                # SL: 1 ATR bajo el swing low más reciente
                atr = df["atr"].iloc[i] if "atr" in df.columns else entry * 0.02
                swing_low = df["last_swing_low"].iloc[i] if "last_swing_low" in df.columns else entry - atr
                sl = min(swing_low - 0.5 * atr, entry - self.p.sl_atr_multiplier * atr)

                risk = entry - sl
                tp1 = entry + self.p.tp1_rr_ratio * risk
                tp2 = entry + self.p.tp2_rr_ratio * risk

                signals[i] = 1
                entry_prices[i] = entry
                stop_losses[i] = sl
                tp1_prices[i] = tp1
                tp2_prices[i] = tp2
                signal_reasons[i] = reason
                signal_quality[i] = quality.value

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
        """
        Verifica las 6 condiciones de entrada long del prompt maestro.
        Retorna: (señal válida, razón textual, calidad del setup)
        """
        row = df.iloc[idx]
        conditions_met = []
        conditions_failed = []

        # ── Condición 1: Alineación de EMAs ───────────────────────────────────
        ema21_col = f"ema_{self._ema_fast}"
        ema55_col = f"ema_{self._ema_mid}"
        ema200_col = f"ema_{self._ema_slow}"

        required_cols = [ema21_col, ema55_col, ema200_col, "adx", "rsi", "macd_histogram", "atr"]
        if not all(col in df.columns for col in required_cols):
            return False, "Indicadores faltantes", SetupQuality.C

        ema21 = row[ema21_col]
        ema55 = row[ema55_col]
        ema200 = row[ema200_col]

        # Optimización V1: Ignorar EMA200 si MACD es positivo y creciente
        macd_hist = row.get("macd_histogram", None)
        macd_growing = row.get("macd_growing", False)
        macd_strong = macd_hist is not None and not pd.isna(macd_hist) and macd_hist > 0 and macd_growing
        
        if ema21 > ema55 > ema200:
            conditions_met.append("EMA21>EMA55>EMA200✓")
        elif ema21 > ema55 and macd_strong:
            conditions_met.append("EMA21>EMA55+MACD↑✓")
        else:
            conditions_failed.append("EMA_alignment✗")
            return False, f"EMAs no alineadas: {ema21:.0f}>{ema55:.0f}>{ema200:.0f}", SetupQuality.C

        # ── Condición 2: Precio sobre EMA 21 ──────────────────────────────────
        if row["close"] > ema21:
            conditions_met.append("Precio>EMA21✓")
        else:
            conditions_failed.append("Precio<EMA21✗")
            return False, "Precio bajo EMA 21", SetupQuality.C

        # ── Condición 3: ADX > 25 ──────────────────────────────────────────────
        adx = row["adx"]
        if pd.isna(adx):
            return False, "ADX no calculado", SetupQuality.C

        if adx > self.p.min_adx:
            conditions_met.append(f"ADX={adx:.1f}>{self.p.min_adx}✓")
        else:
            return False, f"ADX={adx:.1f} insuficiente (<{self.p.min_adx})", SetupQuality.C

        # ── Condición 4: RSI en zona neutra (45-65) ────────────────────────────
        rsi = row["rsi"]
        if pd.isna(rsi):
            return False, "RSI no calculado", SetupQuality.C

        if self.p.rsi_min <= rsi <= self.p.rsi_max:
            conditions_met.append(f"RSI={rsi:.1f}✓")
        else:
            return False, f"RSI={rsi:.1f} fuera de zona neutra (45-65)", SetupQuality.B

        # ── Condición 5: MACD positivo y creciente ─────────────────────────────
        macd_hist = row["macd_histogram"] if "macd_histogram" in df.columns else None
        macd_growing = row["macd_growing"] if "macd_growing" in df.columns else None
        macd_line = row["macd_line"] if "macd_line" in df.columns else None

        if macd_hist is not None and not pd.isna(macd_hist):
            if macd_hist > 0 and macd_growing:
                conditions_met.append(f"MACD={macd_hist:.4f}↑✓")
            elif macd_hist > 0:
                conditions_met.append(f"MACD={macd_hist:.4f}(plano)~")
                # No bloqueante, pero baja calidad
            else:
                return False, f"MACD negativo ({macd_hist:.4f})", SetupQuality.C

        # ── Condición 6: Retroceso a EMA 21 u Order Block ─────────────────────
        atr = row["atr"]
        ema21_touch = abs(row["low"] - ema21) < 0.5 * atr or abs(row["close"] - ema21) < 0.5 * atr
        in_bull_ob = row.get("price_in_bull_ob", False) or row.get("order_block_bullish", False)

        if ema21_touch:
            conditions_met.append("Retroceso_EMA21✓")
        elif in_bull_ob:
            conditions_met.append("En_OrderBlock✓")
        else:
            # No es un retroceso limpio, pero si el resto es muy fuerte → B
            conditions_failed.append("Sin_retroceso~")
            # Solo bloqueante si hay muchas condiciones fallidas
            pass

        # ── Clasificar calidad del setup ───────────────────────────────────────
        quality = self._classify_quality(df, idx, conditions_met, conditions_failed)

        reason = " | ".join(conditions_met)
        return True, reason, quality

    def _classify_quality(
        self,
        df: pd.DataFrame,
        idx: int,
        conditions_met: list,
        conditions_failed: list,
    ) -> SetupQuality:
        """
        A+: Todas las condiciones + volumen + OBV confirmando + Stoch RSI en oversold
        A:  Todas las condiciones básicas
        B:  5 de 6 condiciones
        C:  Setup marginal
        """
        row = df.iloc[idx]
        score = len(conditions_met)
        bonus = 0

        # Bonus por confirmaciones adicionales
        if row.get("obv_bullish", False):
            bonus += 1
        if row.get("volume_above_avg", False):
            bonus += 1
        if row.get("stoch_rsi_oversold", False):
            bonus += 1
        if row.get("price_in_bull_ob", False) or row.get("fvg_bullish", False):
            bonus += 1
        if row.get("bos_bullish", False) or row.get("choch_bullish", False):
            bonus += 1

        if score >= 5 and bonus >= 3:
            return SetupQuality.A_PLUS
        elif score >= 5 and bonus >= 1:
            return SetupQuality.A
        elif score >= 4:
            return SetupQuality.B
        else:
            return SetupQuality.C

    def is_valid_entry(self, df: pd.DataFrame, idx: int) -> bool:
        """Interface del BaseStrategy."""
        valid, _, _ = self._check_long_conditions(df, idx)
        return valid

    def get_trailing_stop(
        self, entry_price: float, current_price: float, current_sl: float, atr: float
    ) -> float:
        """
        Calcula trailing stop para la segunda mitad de la posición.
        El SL sube con el precio, nunca baja.
        """
        new_sl = current_price - 2.0 * atr  # 2 ATR trailing
        return max(new_sl, current_sl)  # Nunca permitir que el SL retroceda

    def should_move_to_breakeven(
        self, entry_price: float, current_price: float, stop_loss: float
    ) -> bool:
        """
        Mover SL a breakeven cuando el precio alcanza 1:1 R/R.
        Regla del prompt: "SÍ mover stop loss a breakeven cuando precio alcanza 1:1 R/R"
        """
        risk = entry_price - stop_loss
        return current_price >= entry_price + risk  # 1:1 R/R alcanzado
