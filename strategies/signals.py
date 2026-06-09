"""
strategies/signals.py — Generación de señales de trading (Trend Following, Mean Reversion & Breakout).
==============================================================================================
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple
import structlog

from config.settings import (
    STRATEGIES,
    INDICATORS,
    SetupQuality,
    SignalDirection,
    StrategyType,
    TrendFollowingParams,
    MeanReversionParams,
    BreakoutParams,
)

log = structlog.get_logger(__name__)

# ── Dataclasses y Clase Base de Estrategias ────────────────────────────────────

@dataclass
class Signal:
    """Señal de trading generada por una estrategia."""
    strategy: StrategyType
    direction: SignalDirection
    symbol: str
    timeframe: str
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: Optional[float] = None
    quality: SetupQuality = SetupQuality.A
    timestamp: Optional[datetime] = None
    confidence: float = 0.5
    entry_reason: str = ""
    conditions_met: List[str] = field(default_factory=list)
    position_size: float = 0.0
    risk_amount: float = 0.0
    market_regime: str = "unknown"

    @property
    def risk_reward_ratio_tp1(self) -> float:
        risk = abs(self.entry_price - self.stop_loss)
        if risk <= 0:
            return 0.0
        return abs(self.take_profit_1 - self.entry_price) / risk

    @property
    def risk_reward_ratio_tp2(self) -> Optional[float]:
        if self.take_profit_2 is None:
            return None
        risk = abs(self.entry_price - self.stop_loss)
        if risk <= 0:
            return 0.0
        return abs(self.take_profit_2 - self.entry_price) / risk

    @property
    def is_valid(self) -> bool:
        if self.entry_price <= 0 or self.stop_loss <= 0 or self.take_profit_1 <= 0:
            return False
        risk = abs(self.entry_price - self.stop_loss)
        if risk <= 0:
            return False
        if self.direction == SignalDirection.LONG:
            return self.stop_loss < self.entry_price < self.take_profit_1
        elif self.direction == SignalDirection.SHORT:
            return self.take_profit_1 < self.entry_price < self.stop_loss
        return False

    def to_dict(self) -> dict:
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
    def __init__(self, symbol: str, timeframe: str = "4h"):
        self.symbol = symbol
        self.timeframe = timeframe

    @property
    @abstractmethod
    def strategy_type(self) -> StrategyType:
        pass

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def is_valid_entry(self, df: pd.DataFrame, idx: int) -> bool:
        pass

    def prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        from indicators.technical import apply_all_indicators
        return apply_all_indicators(df)

    def _safe_get(self, df: pd.DataFrame, idx: int, col: str, default=None):
        if col not in df.columns or idx >= len(df):
            return default
        val = df[col].iloc[idx]
        if pd.isna(val):
            return default
        return val

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(symbol={self.symbol}, tf={self.timeframe})"

# ── Estrategia 1: Trend Following ──────────────────────────────────────────────

class TrendFollowingStrategy(BaseStrategy):
    def __init__(
        self,
        symbol: str,
        timeframe: str = "4h",
        params: TrendFollowingParams = STRATEGIES.trend_following,
    ):
        super().__init__(symbol, timeframe)
        self.p = params
        self._ema_fast = INDICATORS.trend.ema_fast
        self._ema_mid = INDICATORS.trend.ema_mid
        self._ema_slow = INDICATORS.trend.ema_slow

    @property
    def strategy_type(self) -> StrategyType:
        return StrategyType.TREND_FOLLOWING

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Genera señales de entrada vectorizadas (pandas/numpy) para backtesting.
        Calcula todas las condiciones booleanas en bulk sobre el DataFrame completo,
        evitando el bucle Python puro (~50x más rápido en hardware limitado).
        El método _check_long_conditions se preserva para el engine live (fila a fila).
        """
        df = df.copy()
        n = len(df)

        ema21_col  = f"ema_{self._ema_fast}"
        ema55_col  = f"ema_{self._ema_mid}"
        ema200_col = f"ema_{self._ema_slow}"

        required_cols = [ema21_col, ema55_col, ema200_col, "adx", "rsi", "macd_histogram", "atr"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            # Si faltan indicadores, caer al bucle original con aviso
            import warnings
            warnings.warn(f"generate_signals: columnas faltantes {missing}, usando modo lento.")
            return self._generate_signals_slow(df)

        # ── Máscaras vectorizadas ──────────────────────────────────────────────
        ema21  = df[ema21_col].astype(float)
        ema55  = df[ema55_col].astype(float)
        ema200 = df[ema200_col].astype(float)
        close  = df["close"].astype(float)
        low    = df["low"].astype(float)
        adx    = df["adx"].astype(float)
        rsi    = df["rsi"].astype(float)
        atr    = df["atr"].astype(float)
        macd_h = df["macd_histogram"].astype(float)

        macd_growing = df.get("macd_growing", pd.Series(False, index=df.index))
        if isinstance(macd_growing, pd.DataFrame):
            macd_growing = macd_growing.iloc[:, 0]
        macd_growing = macd_growing.astype(bool)

        # Condición 1: alineación EMA
        ema_aligned      = (ema21 > ema55) & (ema55 > ema200)
        macd_strong      = (macd_h > 0) & macd_growing
        ema_cond         = ema_aligned | ((ema21 > ema55) & macd_strong)

        # Condición 2: precio sobre EMA21
        price_above_ema21 = close > ema21

        # Condición 3: ADX suficiente
        adx_cond = adx > self.p.min_adx

        # Condición 4: RSI en zona válida
        rsi_cond = (rsi >= self.p.rsi_min) & (rsi <= self.p.rsi_max)

        # Condición 5: MACD positivo (no negativo)
        macd_cond = macd_h > 0

        # Máscara global de señal larga (a partir de la vela 200 para warmup)
        warmup_mask = pd.Series(False, index=df.index)
        warmup_mask.iloc[200:] = True

        long_mask = warmup_mask & ema_cond & price_above_ema21 & adx_cond & rsi_cond & macd_cond

        # ── Construir SL / TP vectorizados ────────────────────────────────────
        swing_low = (
            df["last_swing_low"].astype(float)
            if "last_swing_low" in df.columns
            else close - atr
        )

        sl_swing  = swing_low - 0.5 * atr
        sl_atr    = close - self.p.sl_atr_multiplier * atr
        stop_loss = np.minimum(sl_swing.values, sl_atr.values)  # el más alejado (más conservador)
        risk      = close.values - stop_loss
        tp1       = close.values + self.p.tp1_rr_ratio * risk
        tp2       = close.values + self.p.tp2_rr_ratio * risk

        # ── Calidad de señal vectorizada (simplificada para velocidad) ─────────
        bonus = np.zeros(n, dtype=int)
        for bonus_col in ["obv_bullish", "volume_above_avg", "stoch_rsi_oversold",
                          "price_in_bull_ob", "fvg_bullish", "bos_bullish", "choch_bullish"]:
            if bonus_col in df.columns:
                bonus += df[bonus_col].astype(bool).astype(int).values

        # Score base = número de condiciones cumplidas (5 posibles)
        score = (ema_aligned.astype(int) + price_above_ema21.astype(int)
                 + adx_cond.astype(int) + rsi_cond.astype(int) + macd_cond.astype(int)).values

        quality_arr = np.where(
            (score >= 5) & (bonus >= 3), "A+",
            np.where(
                (score >= 5) & (bonus >= 1), "A",
                np.where(score >= 4, "B", "C")
            )
        )

        # ── Escribir resultados ────────────────────────────────────────────────
        mask_idx = long_mask.values

        signals_arr = np.zeros(n, dtype=int)
        signals_arr[mask_idx] = 1

        entry_prices = np.full(n, np.nan)
        stop_losses  = np.full(n, np.nan)
        tp1_prices   = np.full(n, np.nan)
        tp2_prices   = np.full(n, np.nan)

        entry_prices[mask_idx] = close.values[mask_idx]
        stop_losses[mask_idx]  = stop_loss[mask_idx]
        tp1_prices[mask_idx]   = tp1[mask_idx]
        tp2_prices[mask_idx]   = tp2[mask_idx]

        # Construir reason string solo donde hay señal (evita trabajo innecesario)
        reason_arr = [""] * n
        for i in np.where(mask_idx)[0]:
            parts = []
            if ema_aligned.iloc[i]:
                parts.append("EMA21>EMA55>EMA200✓")
            elif macd_strong.iloc[i]:
                parts.append("EMA21>EMA55+MACD↑✓")
            if price_above_ema21.iloc[i]:
                parts.append("Precio>EMA21✓")
            if adx_cond.iloc[i]:
                parts.append(f"ADX={adx.iloc[i]:.1f}✓")
            if rsi_cond.iloc[i]:
                parts.append(f"RSI={rsi.iloc[i]:.1f}✓")
            if macd_h.iloc[i] > 0:
                parts.append(f"MACD={macd_h.iloc[i]:.4f}{'↑' if macd_growing.iloc[i] else '~'}✓")
            reason_arr[i] = " | ".join(parts)

        df["signal"]        = signals_arr
        df["entry_price"]   = entry_prices
        df["stop_loss"]     = stop_losses
        df["take_profit_1"] = tp1_prices
        df["take_profit_2"] = tp2_prices
        df["signal_reason"] = reason_arr
        df["signal_quality"] = quality_arr
        df["strategy"]      = self.strategy_type.value
        return df

    def _generate_signals_slow(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback con bucle Python (para casos sin todos los indicadores)."""
        n = len(df)
        signals = np.zeros(n, dtype=int)
        entry_prices = np.full(n, np.nan)
        stop_losses = np.full(n, np.nan)
        tp1_prices = np.full(n, np.nan)
        tp2_prices = np.full(n, np.nan)
        signal_reasons = [""] * n
        signal_quality = [""] * n

        for i in range(200, n):
            long_signal, reason, quality = self._check_long_conditions(df, i)
            if long_signal:
                entry = df["close"].iloc[i]
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

    def _check_long_conditions(self, df: pd.DataFrame, idx: int) -> tuple[bool, str, SetupQuality]:
        row = df.iloc[idx]
        conditions_met = []
        conditions_failed = []

        ema21_col = f"ema_{self._ema_fast}"
        ema55_col = f"ema_{self._ema_mid}"
        ema200_col = f"ema_{self._ema_slow}"

        required_cols = [ema21_col, ema55_col, ema200_col, "adx", "rsi", "macd_histogram", "atr"]
        if not all(col in df.columns for col in required_cols):
            return False, "Indicadores faltantes", SetupQuality.C

        ema21 = row[ema21_col]
        ema55 = row[ema55_col]
        ema200 = row[ema200_col]

        macd_hist = row.get("macd_histogram", None)
        macd_growing = row.get("macd_growing", False)
        macd_strong = macd_hist is not None and not pd.isna(macd_hist) and macd_hist > 0 and macd_growing

        if ema21 > ema55 > ema200:
            conditions_met.append("EMA21>EMA55>EMA200✓")
        elif ema21 > ema55 and macd_strong:
            conditions_met.append("EMA21>EMA55+MACD↑✓")
        else:
            conditions_failed.append("EMA_alignment✗")
            return False, f"EMAs no alineadas", SetupQuality.C

        if row["close"] > ema21:
            conditions_met.append("Precio>EMA21✓")
        else:
            conditions_failed.append("Precio<EMA21✗")
            return False, "Precio bajo EMA 21", SetupQuality.C

        adx = row["adx"]
        if pd.isna(adx):
            return False, "ADX no calculado", SetupQuality.C

        if adx > self.p.min_adx:
            conditions_met.append(f"ADX={adx:.1f}>{self.p.min_adx}✓")
        else:
            return False, f"ADX insuficiente", SetupQuality.C

        rsi = row["rsi"]
        if pd.isna(rsi):
            return False, "RSI no calculado", SetupQuality.C

        if self.p.rsi_min <= rsi <= self.p.rsi_max:
            conditions_met.append(f"RSI={rsi:.1f}✓")
        else:
            return False, f"RSI fuera de zona", SetupQuality.B

        if macd_hist is not None and not pd.isna(macd_hist):
            if macd_hist > 0 and macd_growing:
                conditions_met.append(f"MACD={macd_hist:.4f}↑✓")
            elif macd_hist > 0:
                conditions_met.append(f"MACD={macd_hist:.4f}~")
            else:
                return False, f"MACD negativo", SetupQuality.C

        atr = row["atr"]
        ema21_touch = abs(row["low"] - ema21) < 0.5 * atr or abs(row["close"] - ema21) < 0.5 * atr
        in_bull_ob = row.get("price_in_bull_ob", False) or row.get("order_block_bullish", False)

        if ema21_touch:
            conditions_met.append("Retroceso_EMA21✓")
        elif in_bull_ob:
            conditions_met.append("En_OrderBlock✓")
        else:
            conditions_failed.append("Sin_retroceso~")

        quality = self._classify_quality(df, idx, conditions_met, conditions_failed)
        return True, " | ".join(conditions_met), quality

    def _classify_quality(self, df: pd.DataFrame, idx: int, conditions_met: list, conditions_failed: list) -> SetupQuality:
        row = df.iloc[idx]
        score = len(conditions_met)
        bonus = 0
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
        valid, _, _ = self._check_long_conditions(df, idx)
        return valid

# ── Estrategia 2: Mean Reversion ───────────────────────────────────────────────

class MeanReversionStrategy(BaseStrategy):
    def __init__(
        self,
        symbol: str,
        timeframe: str = "4h",
        params: MeanReversionParams = STRATEGIES.mean_reversion,
    ):
        super().__init__(symbol, timeframe)
        self.p = params
        self._ema_fast = INDICATORS.trend.ema_fast
        self.p.max_adx = 20.0
        self.p.rsi_long_threshold = 35.0
        self.p.rsi_short_threshold = 65.0
        self.p.sl_atr_buffer = 1.5

    @property
    def strategy_type(self) -> StrategyType:
        return StrategyType.MEAN_REVERSION

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        n = len(df)
        signals = np.zeros(n, dtype=int)
        entry_prices = np.full(n, np.nan)
        stop_losses = np.full(n, np.nan)
        tp1_prices = np.full(n, np.nan)
        tp2_prices = np.full(n, np.nan)
        signal_reasons = [""] * n
        signal_quality = [""] * n

        ema_55_1d = df["close"].ewm(span=1320, adjust=False).mean()
        ema_200_1d = df["close"].ewm(span=4800, adjust=False).mean()

        warmup = min(4800, len(df) // 2) if len(df) > 100 else 50

        for i in range(warmup, n):
            macro_bull = True
            if len(df) > 4800:
                macro_bull = ema_55_1d.iloc[i] > ema_200_1d.iloc[i]
            
            long_signal = False
            if macro_bull:
                long_signal, long_reason, long_quality = self._check_long_conditions(df, i)
            
            if long_signal:
                entry = df["close"].iloc[i]
                atr = df["atr"].iloc[i] if not pd.isna(df["atr"].iloc[i]) else entry * 0.02
                bb_lower = df["bb_lower"].iloc[i]
                ema21 = df[f"ema_{self._ema_fast}"].iloc[i]
                bb_upper = df["bb_upper"].iloc[i]

                sl = bb_lower - self.p.sl_atr_buffer * atr
                tp1 = ema21
                tp2 = bb_upper

                if tp1 > entry and tp2 > entry:
                    signals[i] = 1
                    entry_prices[i] = entry
                    stop_losses[i] = sl
                    tp1_prices[i] = tp1
                    tp2_prices[i] = tp2
                    signal_reasons[i] = long_reason
                    signal_quality[i] = long_quality.value

        df["signal"] = signals
        df["entry_price"] = entry_prices
        df["stop_loss"] = stop_losses
        df["take_profit_1"] = tp1_prices
        df["take_profit_2"] = tp2_prices
        df["signal_reason"] = signal_reasons
        df["signal_quality"] = signal_quality
        df["strategy"] = self.strategy_type.value
        return df

    def _check_long_conditions(self, df: pd.DataFrame, idx: int) -> tuple[bool, str, SetupQuality]:
        row = df.iloc[idx]
        conditions = []
        required = ["adx", "rsi", "bb_lower", "bb_upper", "stoch_rsi_k", "atr", f"ema_{self._ema_fast}"]

        for col in required:
            if col not in df.columns or pd.isna(row.get(col, float("nan"))):
                return False, f"Columna faltante: {col}", SetupQuality.C

        adx = row["adx"]
        if adx >= self.p.max_adx:
            return False, f"ADX={adx:.1f} no es rango", SetupQuality.C
        conditions.append(f"ADX={adx:.1f}<20✓")

        rsi = row["rsi"]
        if rsi >= self.p.rsi_long_threshold:
            return False, f"RSI={rsi:.1f} no sobreventa", SetupQuality.C
        conditions.append(f"RSI={rsi:.1f}<35✓")

        close = row["close"]
        bb_lower = row["bb_lower"]
        atr = row["atr"]
        if close > bb_lower + 0.5 * atr:
            return False, "Precio no en banda inferior BB", SetupQuality.C
        conditions.append("Precio_en_BB_lower✓")

        in_support = (
            row.get("price_in_bull_ob", False)
            or row.get("liquidity_below", False)
            or (abs(close - row.get("last_swing_low", close)) < atr)
        )
        if in_support:
            conditions.append("Soporte_coincidente✓")

        stoch_div = row.get("stoch_rsi_bullish_divergence", False)
        stoch_oversold = row.get("stoch_rsi_oversold", False)
        stoch_cross = row.get("stoch_rsi_bullish_cross", False)

        if stoch_div or (stoch_oversold and stoch_cross):
            conditions.append("StochRSI_divergencia✓")
        elif stoch_oversold:
            conditions.append("StochRSI_sobreventa~")

        quality = self._classify_long_quality(row, len(conditions))
        return True, " | ".join(conditions), quality

    def _check_short_conditions(self, df: pd.DataFrame, idx: int) -> tuple[bool, str, SetupQuality]:
        row = df.iloc[idx]
        conditions = []
        required = ["adx", "rsi", "bb_upper", "stoch_rsi_k", "atr"]
        for col in required:
            if col not in df.columns or pd.isna(row.get(col, float("nan"))):
                return False, f"Columna faltante: {col}", SetupQuality.C

        adx = row["adx"]
        if adx >= self.p.max_adx:
            return False, f"ADX={adx:.1f}", SetupQuality.C
        conditions.append(f"ADX={adx:.1f}<20✓")

        rsi = row["rsi"]
        if rsi <= self.p.rsi_short_threshold:
            return False, f"RSI={rsi:.1f} no sobrecompra", SetupQuality.C
        conditions.append(f"RSI={rsi:.1f}>65✓")

        close = row["close"]
        bb_upper = row["bb_upper"]
        atr = row["atr"]
        if close < bb_upper - 0.5 * atr:
            return False, "Precio no en banda superior BB", SetupQuality.C
        conditions.append("Precio_en_BB_upper✓")

        in_resistance = (
            row.get("price_in_bear_ob", False)
            or row.get("liquidity_above", False)
        )
        if in_resistance:
            conditions.append("Resistencia_coincidente✓")

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

# ── Estrategia 3: Breakout ─────────────────────────────────────────────────────

class BreakoutStrategy(BaseStrategy):
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
                breakout_level = df["bb_upper"].iloc[i - 1] if "bb_upper" in df.columns else entry - atr
                sl = breakout_level - self.p.sl_atr_multiplier * atr
                risk = entry - sl
                tp1 = entry + 2.0 * risk
                tp2 = entry + 4.0 * risk
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

    def _check_bullish_breakout(self, df: pd.DataFrame, idx: int) -> tuple[bool, str, SetupQuality]:
        row = df.iloc[idx]
        prev = df.iloc[idx - 1]
        conditions = []
        required = ["bb_squeeze_candles", "volume_ratio", "bb_upper"]
        for col in required:
            if col not in df.columns:
                return False, f"Columna faltante: {col}", SetupQuality.C

        squeeze_candles = row.get("bb_squeeze_candles", 0)
        if squeeze_candles < self.p.min_squeeze_candles:
            return False, f"Squeeze insuficiente: {squeeze_candles}", SetupQuality.C

        was_in_squeeze = prev.get("bb_squeeze", False)
        broke_upper = row["close"] > prev["bb_upper"]
        if not (was_in_squeeze and broke_upper):
            return False, "No hay ruptura", SetupQuality.C
        conditions.append(f"Squeeze_release({squeeze_candles}velas)✓")

        vol_ratio = row.get("volume_ratio", 0)
        if vol_ratio < self.p.volume_multiplier:
            return False, f"Volumen insuficiente: {vol_ratio:.1f}x", SetupQuality.C
        conditions.append(f"Volumen={vol_ratio:.1f}x✓")

        if self.p.wait_for_retest:
            price_near_breakout = abs(row["close"] - prev["bb_upper"]) < row.get("atr", row["close"] * 0.02)
            if price_near_breakout:
                conditions.append("Retesteo✓")

        if row.get("bos_bullish", False):
            conditions.append("BOS_alcista✓")

        if vol_ratio >= 2.5:
            quality = SetupQuality.A_PLUS
        elif vol_ratio >= 1.75 and len(conditions) >= 3:
            quality = SetupQuality.A
        else:
            quality = SetupQuality.B
        return True, " | ".join(conditions), quality

    def _check_bearish_breakout(self, df: pd.DataFrame, idx: int) -> tuple[bool, str, SetupQuality]:
        row = df.iloc[idx]
        prev = df.iloc[idx - 1]
        conditions = []
        required = ["bb_squeeze_candles", "volume_ratio", "bb_lower"]
        for col in required:
            if col not in df.columns:
                return False, f"Columna faltante: {col}", SetupQuality.C

        squeeze_candles = row.get("bb_squeeze_candles", 0)
        if squeeze_candles < self.p.min_squeeze_candles:
            return False, f"Squeeze insuficiente", SetupQuality.C

        was_in_squeeze = prev.get("bb_squeeze", False)
        broke_lower = row["close"] < prev["bb_lower"]
        if not (was_in_squeeze and broke_lower):
            return False, "No hay ruptura", SetupQuality.C
        conditions.append(f"Squeeze_release({squeeze_candles})✓")

        vol_ratio = row.get("volume_ratio", 0)
        if vol_ratio < self.p.volume_multiplier:
            return False, f"Volumen insuficiente", SetupQuality.C
        conditions.append(f"Volumen={vol_ratio:.1f}x✓")

        if row.get("bos_bearish", False):
            conditions.append("BOS_bajista✓")

        quality = SetupQuality.A_PLUS if vol_ratio >= 2.5 else (SetupQuality.A if len(conditions) >= 3 else SetupQuality.B)
        return True, " | ".join(conditions), quality

    def is_valid_entry(self, df: pd.DataFrame, idx: int) -> bool:
        bull_valid, _, _ = self._check_bullish_breakout(df, idx)
        bear_valid, _, _ = self._check_bearish_breakout(df, idx)
        return bull_valid or bear_valid

# ── Funciones Vectorizadas para apply_all_signals ─────────────────────────────

def signal_trend_following(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["close"]

    cond_ema = df.get("ema_alignment_bullish", pd.Series(False, index=df.index)).fillna(False)
    if not cond_ema.any() and "ema_21" in df.columns and "ema_55" in df.columns and "ema_200" in df.columns:
        cond_ema = (df["ema_21"] > df["ema_55"]) & (df["ema_55"] > df["ema_200"])
    elif "ema_bullish" in df.columns:
        cond_ema = df["ema_bullish"].fillna(False)

    cond_adx = df.get("adx", pd.Series(0.0, index=df.index)) > 20
    rsi_val = df.get("rsi", pd.Series(50.0, index=df.index))
    cond_rsi = (rsi_val >= 40.0) & (rsi_val <= 70.0)

    macd_bull = df.get("macd_bullish_cross", df.get("macd_line", pd.Series(0.0, index=df.index)) > 0).fillna(False)
    if "macd_bull" in df.columns:
        macd_bull = df["macd_bull"].fillna(False)
    macd_growing = df.get("macd_histogram", pd.Series(0.0, index=df.index)).diff() > 0
    if "macd_growing" in df.columns:
        macd_growing = df["macd_growing"].fillna(False)
    cond_macd = macd_bull | macd_growing

    ema21_col = "ema_21" if "ema_21" in df.columns else "ema21"
    ema21 = df.get(ema21_col, c).fillna(c)
    cond_pullback = (c <= ema21 * 1.02) & (c >= ema21 * 0.98)

    cond_ob = df.get("ob_bull", pd.Series(False, index=df.index)).fillna(False)
    cond_fvg = df.get("fvg_bull", pd.Series(False, index=df.index)).fillna(False)
    cond_vol = ~df.get("extreme_vol", pd.Series(False, index=df.index)).fillna(True)
    cond_consensus = df.get("consensus_bull", pd.Series(True, index=df.index)).fillna(True)

    df["tf_long_signal"] = (
        cond_ema & cond_adx & cond_rsi & cond_macd & cond_vol
        & (cond_pullback | cond_ob | cond_fvg)
        & cond_consensus
    ).astype(int)

    sig = df["tf_long_signal"].to_numpy().copy()
    for i in range(1, len(sig)):
        if sig[i] and sig[max(0, i - 3):i].any():
            sig[i] = 0
    df["tf_long_signal"] = sig

    atr = df.get("atr", c * 0.015).fillna(c * 0.015)
    low_min_5 = df["low"].rolling(5).min().fillna(df["low"])
    price_risk = c - (low_min_5 - atr * 1.5)
    df["tf_stop_long"] = low_min_5 - atr * 1.5
    df["tf_tp1"] = c + price_risk * 2.0
    df["tf_tp2"] = c + price_risk * 3.0
    return df

def signal_mean_reversion(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["close"]
    rsi = df.get("rsi", pd.Series(50.0, index=df.index)).fillna(50.0)
    adx = df.get("adx", pd.Series(20.0, index=df.index)).fillna(20.0)
    bb_lower = df.get("bb_lower", c).fillna(c)

    cond_adx = adx < 20
    cond_rsi = rsi < 35
    cond_bb = c <= bb_lower * 1.01

    df["mr_long_signal"] = (cond_adx & cond_rsi & cond_bb).astype(int)

    sig = df["mr_long_signal"].to_numpy().copy()
    for i in range(1, len(sig)):
        if sig[i] and sig[max(0, i - 3):i].any():
            sig[i] = 0
    df["mr_long_signal"] = sig

    atr = df.get("atr", c * 0.015).fillna(c * 0.015)
    df["mr_stop_long"] = bb_lower - 1.5 * atr
    df["mr_tp1"] = df.get("ema_21", df.get("ema21", c))
    df["mr_tp2"] = df.get("bb_upper", c)
    return df

def signal_breakout(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["close"]
    bb_squeeze = df.get("bb_squeeze", pd.Series(False, index=df.index)).fillna(False)
    sq_n = bb_squeeze.rolling(20, min_periods=1).sum().fillna(0)
    bb_upper = df.get("bb_upper", c).fillna(c)
    close_above_bb = c > bb_upper
    vol_ratio = df.get("vol_ratio", pd.Series(1.0, index=df.index)).fillna(1.0)
    high_20 = df["high"].shift(1).rolling(20, min_periods=1).max().fillna(df["high"])
    break_20h_high = c > high_20
    macd_bull = df.get("macd_bull", pd.Series(False, index=df.index)).fillna(False)

    score = pd.Series(0.0, index=df.index)
    score += np.where(sq_n >= 10, 25, 0)
    score += np.where(close_above_bb, 20, 0)
    score += np.where(vol_ratio > 2.0, 20, np.where(vol_ratio > 1.5, 12, 0))
    score += np.where(break_20h_high, 15, 0)
    score += np.where(macd_bull, 10, 0)

    absorption = df.get("absorption", pd.Series(0, index=df.index)).fillna(0).astype(bool)
    obv_accel = df.get("obv_accel", pd.Series(False, index=df.index)).fillna(False)
    inst_vol = absorption | obv_accel

    df["bo_long_signal"] = (inst_vol & (score >= 65)).astype(int)

    sig = df["bo_long_signal"].to_numpy().copy()
    for i in range(1, len(sig)):
        if sig[i] and sig[max(0, i - 3):i].any():
            sig[i] = 0
    df["bo_long_signal"] = sig

    atr = df.get("atr", c * 0.015).fillna(c * 0.015)
    prev_upper = df["bb_upper"].shift(1).fillna(bb_upper)
    df["bo_stop_long"] = prev_upper - 2.0 * atr
    df["bo_tp1"] = c + (c - df["bo_stop_long"]) * 2.0
    df["bo_tp2"] = c + (c - df["bo_stop_long"]) * 4.0
    return df

def apply_all_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = signal_trend_following(df)
    df = signal_mean_reversion(df)
    df = signal_breakout(df)

    # Mapeos de compatibilidad con V5/V6 engines
    df["signal_trend"] = df["tf_long_signal"]
    df["signal"] = df["tf_long_signal"]  # Default
    return df
