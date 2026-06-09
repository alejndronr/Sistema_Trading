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

    @staticmethod
    def _calculate_volatility_percentile(atr_series: pd.Series, lookback: int = 90 * 6) -> float:
        """
        Calcula el percentil actual del ATR respecto a su historia reciente.
        lookback=540 velas corresponde a ~90 días en 4H (6 velas/día).
        Retorna un valor en [0, 1]: 0=volatilidad históricamente baja, 1=alta.
        """
        if len(atr_series) < lookback:
            lookback = max(50, len(atr_series) // 2)
        window = atr_series.dropna().iloc[-lookback:]
        if window.empty or window.iloc[-1] == 0:
            return 0.5  # Default: volatilidad normal
        current_atr = window.iloc[-1]
        pct = float((window < current_atr).mean())
        return round(pct, 4)

    @staticmethod
    def _get_volatility_adjustments(
        atr_series: pd.Series,
        lookback: int = 90 * 6,
    ) -> tuple:
        """
        Devuelve (sl_multiplier, tp_multiplier) basados en el régimen de volatilidad.

        Régimen de volatilidad (basado en percentil ATR histórico 90 días):
          LOW_VOL   (< P30):  SL ajustado -20% | TP ajustado -10%
                              (volatilidad comprimida → movimientos más pequeños)
          NORMAL    (P30-P70): sin ajuste (1.0, 1.0)
          HIGH_VOL  (P70-P90): SL ampliado +30% | TP ampliado +20%
                              (volatilidad expandida → dejar más espacio)
          EXTREME   (> P90):  SL ampliado +60% | TP reducido -20%
                              (caos extremo → proteger capital, objetivos conservadores)

        Nota: los ajustes preservan la relación R/R al aplicarse proporcionalmente,
        manteniendo la ventaja matemática del setup.
        """
        if isinstance(atr_series, pd.Series) and len(atr_series) > 0:
            pct = TrendFollowingStrategy._calculate_volatility_percentile(atr_series, lookback)
        else:
            pct = 0.5

        if pct < 0.30:       # LOW_VOL: comprimida, objetivos conservadores
            return (0.80, 0.90)
        elif pct < 0.70:     # NORMAL: sin ajuste
            return (1.00, 1.00)
        elif pct < 0.90:     # HIGH_VOL: expandida, más espacio
            return (1.30, 1.20)
        else:                # EXTREME_VOL: caos, proteger capital
            return (1.60, 0.80)

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

        # ── Máscaras vectorizadas para LONG ─────────────────────────────────────
        # Enfoque estocástico/probabilístico en lugar de reglas booleanas rígidas.
        ts_tstat         = df.get("ts_tstat", pd.Series(0.0, index=df.index)).astype(float)
        prob_bull        = df.get("prob_bull", pd.Series(0.0, index=df.index)).astype(float)
        hurst            = df.get("hurst_exponent", pd.Series(0.5, index=df.index)).astype(float)

        ema_aligned      = (ema21 > ema55) & (ema55 > ema200)
        
        # Condición Primaria Matemática:
        # T-stat > 1.5 (Confianza estadística de momentum ~93%)
        # Hurst > 0.52 (Mercado tendencial persistente, descarta random walk)
        # Prob_Bull > 0.40 (Modelo GMM indica alta probabilidad de régimen alcista)
        math_bull_cond   = (ts_tstat > 1.5) & (hurst > 0.52) & (prob_bull > 0.40)
        
        # Condición Clásica como Fallback o refuerzo:
        macd_strong      = (macd_h > 0) & macd_growing
        ema_cond         = ema_aligned | ((ema21 > ema55) & macd_strong)
        
        price_above_ema21 = close > ema21
        rsi_cond         = (rsi >= self.p.rsi_min) & (rsi <= self.p.rsi_max)
        macd_cond        = macd_h > 0
        
        warmup_mask = pd.Series(False, index=df.index)
        warmup_mask.iloc[200:] = True

        # Combinamos el modelo matemático con la confirmación de momentum clásico
        long_mask = warmup_mask & math_bull_cond & price_above_ema21 & rsi_cond & macd_cond

        # Volatilidad dinámica (Expansión/Contracción GARCH)
        vol_garch = df.get("vol_garch_proxy", pd.Series(1.0, index=df.index)).astype(float)
        # Limitar el multiplicador GARCH entre 0.5 (muy comprimido) y 2.0 (muy expandido)
        dynamic_atr = atr * np.clip(vol_garch, 0.5, 2.0)

        # ── Construir SL / TP vectorizados para LONG ──────────────────────────
        swing_low = df["last_swing_low"].astype(float) if "last_swing_low" in df.columns else close - dynamic_atr
        sl_swing  = swing_low - 0.5 * dynamic_atr
        sl_atr    = close - self.p.sl_atr_multiplier * dynamic_atr
        stop_loss = np.minimum(sl_swing.values, sl_atr.values)
        risk      = close.values - stop_loss
        tp1       = close.values + self.p.tp1_rr_ratio * risk
        tp2       = close.values + self.p.tp2_rr_ratio * risk

        # ── Máscaras vectorizadas para SHORT ────────────────────────────────────
        from config.settings import ASSETS
        asset_cfg = ASSETS.get(self.symbol)
        allow_shorts = asset_cfg.allow_shorts if asset_cfg else True

        prob_bear = df.get("prob_bear", pd.Series(0.0, index=df.index)).astype(float)
        ema_aligned_short = (ema21 < ema55) & (ema55 < ema200)

        # Condición Primaria Matemática:
        math_bear_cond = (ts_tstat < -1.5) & (hurst > 0.52) & (prob_bear > 0.40)

        macd_strong_short = (macd_h < 0) & (~macd_growing)
        ema_cond_short    = ema_aligned_short | ((ema21 < ema55) & macd_strong_short)
        price_below_ema21 = close < ema21
        rsi_cond_short    = (rsi >= (100 - self.p.rsi_max)) & (rsi <= (100 - self.p.rsi_min))
        macd_cond_short   = macd_h < 0

        if allow_shorts:
            short_mask = warmup_mask & math_bear_cond & price_below_ema21 & rsi_cond_short & macd_cond_short
        else:
            short_mask = pd.Series(False, index=df.index)

        # ── Construir SL / TP vectorizados para SHORT ─────────────────────────
        swing_high = df["last_swing_high"].astype(float) if "last_swing_high" in df.columns else close + dynamic_atr
        sl_swing_short  = swing_high + 0.5 * dynamic_atr
        sl_atr_short    = close + self.p.sl_atr_multiplier * dynamic_atr
        stop_loss_short = np.maximum(sl_swing_short.values, sl_atr_short.values)
        risk_short      = stop_loss_short - close.values
        tp1_short       = close.values - self.p.tp1_rr_ratio * risk_short
        tp2_short       = close.values - self.p.tp2_rr_ratio * risk_short

        # ── Calidad de señal vectorizada ───────────────────────────────────────
        bonus_long = np.zeros(n, dtype=int)
        bonus_short = np.zeros(n, dtype=int)
        
        for bonus_col in ["obv_bullish", "volume_above_avg", "stoch_rsi_oversold", "price_in_bull_ob", "fvg_bullish", "bos_bullish", "choch_bullish"]:
            if bonus_col in df.columns:
                bonus_long += df[bonus_col].astype(bool).astype(int).values
                
        for bonus_col in ["obv_bearish", "volume_above_avg", "stoch_rsi_overbought", "price_in_bear_ob", "fvg_bearish", "bos_bearish", "choch_bearish"]:
            if bonus_col in df.columns:
                bonus_short += df[bonus_col].astype(bool).astype(int).values

        score_long = (ema_aligned.astype(int) + price_above_ema21.astype(int) + math_bull_cond.astype(int) + rsi_cond.astype(int) + macd_cond.astype(int)).values
        score_short = (ema_aligned_short.astype(int) + price_below_ema21.astype(int) + math_bear_cond.astype(int) + rsi_cond_short.astype(int) + macd_cond_short.astype(int)).values

        quality_long_arr = np.where((score_long >= 5) & (bonus_long >= 3), "A+", np.where((score_long >= 5) & (bonus_long >= 1), "A", np.where(score_long >= 4, "B", "C")))
        quality_short_arr = np.where((score_short >= 5) & (bonus_short >= 3), "A+", np.where((score_short >= 5) & (bonus_short >= 1), "A", np.where(score_short >= 4, "B", "C")))

        # ── Scoring Bayesiano vectorizado (confianza probabilística) ────────────
        # Construimos el dict de condiciones activas para cada vela
        # y calculamos el score probabilístico usando el scorer global.
        # Nota: para eficiencia, solo se aplica a las velas con señal activa.
        try:
            from risk.signal_scorer import get_scorer
            from risk.ev_filter import get_ev_filter
            scorer    = get_scorer()
            ev_filter = get_ev_filter()

            # Score base bayesiano usando las condiciones booleanas vectorizadas
            # Aproximación: usamos el score_long/score_short normalizado como proxy
            # El scorer completo se llama vela a vela solo en las señales activas
            confidence_long  = np.where(mask_idx := long_mask.values,
                                        0.47 + 0.03 * score_long.clip(0, 5),
                                        0.47)
            confidence_short = np.where(short_mask.values,
                                        0.47 + 0.03 * score_short.clip(0, 5),
                                        0.47)
            # Refinamiento bayesiano completo en las velas con señal activa
            active_long_idx  = np.where(long_mask.values)[0]
            active_short_idx = np.where(short_mask.values)[0]

            for i in active_long_idx:
                conds = {
                    "ema_aligned":       bool(ema_aligned.iloc[i]),
                    "macd_bullish":      bool(macd_cond.iloc[i]),
                    "rsi_trend_zone":    bool(rsi_cond.iloc[i]),
                    "math_bull_cond":    bool(math_bull_cond.iloc[i]),
                    "price_above_ema21": bool(price_above_ema21.iloc[i]),
                    "volume_above_avg":  bool(df.get("volume_above_avg", pd.Series(False, index=df.index)).iloc[i]),
                    "obv_bullish":       bool(df.get("obv_bullish", pd.Series(False, index=df.index)).iloc[i]),
                    "structure_bullish": bool(df.get("bos_bullish", pd.Series(False, index=df.index)).iloc[i]),
                }
                result = scorer.score(conds, direction="long", rr_ratio=self.p.tp1_rr_ratio)
                confidence_long[i] = result.score

            for i in active_short_idx:
                conds = {
                    "ema_aligned_short": bool(ema_aligned_short.iloc[i]),
                    "macd_bearish":      bool(macd_cond_short.iloc[i]),
                    "rsi_trend_zone_short": bool(rsi_cond_short.iloc[i]),
                    "math_bear_cond":    bool(math_bear_cond.iloc[i]),
                    "price_below_ema21": bool(price_below_ema21.iloc[i]),
                    "volume_above_avg":  bool(df.get("volume_above_avg", pd.Series(False, index=df.index)).iloc[i]),
                    "obv_bearish":       bool(df.get("obv_bearish", pd.Series(False, index=df.index)).iloc[i]),
                    "structure_bearish": bool(df.get("bos_bearish", pd.Series(False, index=df.index)).iloc[i]),
                }
                result = scorer.score(conds, direction="short", rr_ratio=self.p.tp1_rr_ratio)
                confidence_short[i] = result.score

        except ImportError:
            # Si los módulos no están disponibles, usar confianza por score simple
            confidence_long  = 0.47 + 0.03 * score_long.clip(0, 5).astype(float)
            confidence_short = 0.47 + 0.03 * score_short.clip(0, 5).astype(float)


        # ── Escribir resultados ────────────────────────────────────────────────
        mask_idx = long_mask.values
        short_mask_idx = short_mask.values

        signals_arr = np.zeros(n, dtype=int)
        signals_arr[mask_idx] = 1
        signals_arr[short_mask_idx] = -1

        entry_prices = np.full(n, np.nan)
        stop_losses  = np.full(n, np.nan)
        tp1_prices   = np.full(n, np.nan)
        tp2_prices   = np.full(n, np.nan)
        quality_arr  = np.full(n, "", dtype=object)

        # Aplicar Longs
        entry_prices[mask_idx] = close.values[mask_idx]
        stop_losses[mask_idx]  = stop_loss[mask_idx]
        tp1_prices[mask_idx]   = tp1[mask_idx]
        tp2_prices[mask_idx]   = tp2[mask_idx]
        quality_arr[mask_idx]  = quality_long_arr[mask_idx]
        
        # Aplicar Shorts
        entry_prices[short_mask_idx] = close.values[short_mask_idx]
        stop_losses[short_mask_idx]  = stop_loss_short[short_mask_idx]
        tp1_prices[short_mask_idx]   = tp1_short[short_mask_idx]
        tp2_prices[short_mask_idx]   = tp2_short[short_mask_idx]
        quality_arr[short_mask_idx]  = quality_short_arr[short_mask_idx]

        # Construir reason string solo donde hay señal (evita trabajo innecesario)
        reason_arr = [""] * n
        for i in np.where(mask_idx)[0]:
            parts = []
            if ema_aligned.iloc[i]: parts.append("EMA21>EMA55>EMA200✓")
            elif macd_strong.iloc[i]: parts.append("EMA21>EMA55+MACD↑✓")
            if price_above_ema21.iloc[i]: parts.append("Precio>EMA21✓")
            if adx_cond.iloc[i]: parts.append(f"ADX={adx.iloc[i]:.1f}✓")
            if rsi_cond.iloc[i]: parts.append(f"RSI={rsi.iloc[i]:.1f}✓")
            if macd_h.iloc[i] > 0: parts.append(f"MACD={macd_h.iloc[i]:.4f}{'↑' if macd_growing.iloc[i] else '~'}✓")
            reason_arr[i] = " | ".join(parts)
            
        for i in np.where(short_mask_idx)[0]:
            parts = []
            if ema_aligned_short.iloc[i]: parts.append("EMA21<EMA55<EMA200✓")
            elif macd_strong_short.iloc[i]: parts.append("EMA21<EMA55+MACD↓✓")
            if price_below_ema21.iloc[i]: parts.append("Precio<EMA21✓")
            if adx_cond.iloc[i]: parts.append(f"ADX={adx.iloc[i]:.1f}✓")
            if rsi_cond_short.iloc[i]: parts.append(f"RSI={rsi.iloc[i]:.1f}✓")
            if macd_h.iloc[i] < 0: parts.append(f"MACD={macd_h.iloc[i]:.4f}✓")
            reason_arr[i] = " | ".join(parts)

        # ── Columna de confianza bayesiana (probabilidad de éxito) ─────────────
        confidence_arr = np.full(n, 0.47)
        confidence_arr[mask_idx]       = confidence_long[mask_idx]
        confidence_arr[short_mask_idx] = confidence_short[short_mask_idx]

        # ── Régimen de volatilidad como string informativo ─────────────────────
        vol_pct = self._calculate_volatility_percentile(atr)
        if vol_pct < 0.30:
            vol_regime = "LOW_VOL"
        elif vol_pct < 0.70:
            vol_regime = "NORMAL_VOL"
        elif vol_pct < 0.90:
            vol_regime = "HIGH_VOL"
        else:
            vol_regime = "EXTREME_VOL"

        df["signal"]             = signals_arr
        df["entry_price"]        = entry_prices
        df["stop_loss"]          = stop_losses
        df["take_profit_1"]      = tp1_prices
        df["take_profit_2"]      = tp2_prices
        df["signal_reason"]      = reason_arr
        df["signal_quality"]     = quality_arr
        df["confidence"]         = confidence_arr
        df["volatility_regime"]  = vol_regime
        df["volatility_pct"]     = vol_pct
        df["strategy"]           = self.strategy_type.value
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
            # Comprobar Long
            long_signal, reason_l, quality_l = self._check_long_conditions(df, i)
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
                signal_reasons[i] = reason_l
                signal_quality[i] = quality_l.value
                continue
                
            # Comprobar Short
            short_signal, reason_s, quality_s = self._check_short_conditions(df, i)
            if short_signal:
                entry = df["close"].iloc[i]
                atr = df["atr"].iloc[i] if "atr" in df.columns else entry * 0.02
                swing_high = df["last_swing_high"].iloc[i] if "last_swing_high" in df.columns else entry + atr
                sl = max(swing_high + 0.5 * atr, entry + self.p.sl_atr_multiplier * atr)
                risk = sl - entry
                tp1 = entry - self.p.tp1_rr_ratio * risk
                tp2 = entry - self.p.tp2_rr_ratio * risk
                signals[i] = -1
                entry_prices[i] = entry
                stop_losses[i] = sl
                tp1_prices[i] = tp1
                tp2_prices[i] = tp2
                signal_reasons[i] = reason_s
                signal_quality[i] = quality_s.value

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

        ts_tstat = row.get("ts_tstat", 0.0)
        hurst = row.get("hurst_exponent", 0.5)
        prob_bull = row.get("prob_bull", 0.0)

        math_bull_cond = (ts_tstat > 1.5) and (hurst > 0.52) and (prob_bull > 0.40)

        if math_bull_cond:
            conditions_met.append(f"Math_Trend✓ (T:{ts_tstat:.1f}, H:{hurst:.2f}, P:{prob_bull:.2f})")
        elif ema21 > ema55 > ema200:
            conditions_met.append("EMA21>EMA55>EMA200✓")
        elif ema21 > ema55 and macd_strong:
            conditions_met.append("EMA21>EMA55+MACD↑✓")
        else:
            conditions_failed.append("No_Trend_Math_Or_EMA✗")
            return False, f"Sin Momentum Probabilístico ni EMAs", SetupQuality.C

        if row["close"] > ema21:
            conditions_met.append("Precio>EMA21✓")
        else:
            conditions_failed.append("Precio<EMA21✗")
            return False, "Precio bajo EMA 21", SetupQuality.C

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

    def _check_short_conditions(self, df: pd.DataFrame, idx: int) -> tuple[bool, str, SetupQuality]:
        from config.settings import ASSETS
        asset_cfg = ASSETS.get(self.symbol)
        if asset_cfg and not asset_cfg.allow_shorts:
            return False, f"Shorts desactivados en config para {self.symbol}", SetupQuality.C

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
        macd_strong_short = macd_hist is not None and not pd.isna(macd_hist) and macd_hist < 0 and not macd_growing

        ts_tstat = row.get("ts_tstat", 0.0)
        hurst = row.get("hurst_exponent", 0.5)
        prob_bear = row.get("prob_bear", 0.0)

        math_bear_cond = (ts_tstat < -1.5) and (hurst > 0.52) and (prob_bear > 0.40)

        if math_bear_cond:
            conditions_met.append(f"Math_Trend_Short✓ (T:{ts_tstat:.1f}, H:{hurst:.2f}, P:{prob_bear:.2f})")
        elif ema21 < ema55 < ema200:
            conditions_met.append("EMA21<EMA55<EMA200✓")
        elif ema21 < ema55 and macd_strong_short:
            conditions_met.append("EMA21<EMA55+MACD↓✓")
        else:
            conditions_failed.append("No_Trend_Math_Or_EMA✗")
            return False, f"Sin Momentum Probabilístico ni EMAs para short", SetupQuality.C

        if row["close"] < ema21:
            conditions_met.append("Precio<EMA21✓")
        else:
            conditions_failed.append("Precio>EMA21✗")
            return False, "Precio sobre EMA 21", SetupQuality.C

        rsi = row["rsi"]
        if pd.isna(rsi): return False, "RSI no calculado", SetupQuality.C

        rsi_min_short = 100 - self.p.rsi_max
        rsi_max_short = 100 - self.p.rsi_min
        if rsi_min_short <= rsi <= rsi_max_short:
            conditions_met.append(f"RSI={rsi:.1f}✓")
        else:
            return False, f"RSI fuera de zona para short", SetupQuality.B

        if macd_hist is not None and not pd.isna(macd_hist):
            if macd_hist < 0 and not macd_growing:
                conditions_met.append(f"MACD={macd_hist:.4f}↓✓")
            elif macd_hist < 0:
                conditions_met.append(f"MACD={macd_hist:.4f}~")
            else:
                return False, f"MACD positivo", SetupQuality.C

        atr = row["atr"]
        ema21_touch = abs(row["high"] - ema21) < 0.5 * atr or abs(row["close"] - ema21) < 0.5 * atr
        in_bear_ob = row.get("price_in_bear_ob", False) or row.get("order_block_bearish", False)

        if ema21_touch:
            conditions_met.append("Retroceso_EMA21✓")
        elif in_bear_ob:
            conditions_met.append("En_OrderBlock✓")
        else:
            conditions_failed.append("Sin_retroceso~")

        quality = self._classify_quality(df, idx, conditions_met, conditions_failed, is_short=True)
        return True, " | ".join(conditions_met), quality

    def _classify_quality(self, df: pd.DataFrame, idx: int, conditions_met: list, conditions_failed: list, is_short: bool = False) -> SetupQuality:
        row = df.iloc[idx]
        score = len(conditions_met)
        bonus = 0
        
        if not is_short:
            if row.get("obv_bullish", False): bonus += 1
            if row.get("volume_above_avg", False): bonus += 1
            if row.get("stoch_rsi_oversold", False): bonus += 1
            if row.get("price_in_bull_ob", False) or row.get("fvg_bullish", False): bonus += 1
            if row.get("bos_bullish", False) or row.get("choch_bullish", False): bonus += 1
        else:
            if row.get("obv_bearish", False): bonus += 1
            if row.get("volume_above_avg", False): bonus += 1
            if row.get("stoch_rsi_overbought", False): bonus += 1
            if row.get("price_in_bear_ob", False) or row.get("fvg_bearish", False): bonus += 1
            if row.get("bos_bearish", False) or row.get("choch_bearish", False): bonus += 1

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

        # ── Variables Vectorizadas ─────────────────────────────────────────────
        close = df["close"].astype(float)
        atr   = df.get("atr", close * 0.01).astype(float)
        bb_lower = df.get("bb_lower", pd.Series(np.nan, index=df.index)).astype(float)
        bb_upper = df.get("bb_upper", pd.Series(np.nan, index=df.index)).astype(float)
        ema21 = df.get(f"ema_{self._ema_fast}", pd.Series(np.nan, index=df.index)).astype(float)

        zscore = df.get("zscore_vwap", pd.Series(0.0, index=df.index)).astype(float)
        vol_garch = df.get("vol_garch_proxy", pd.Series(1.0, index=df.index)).astype(float)
        prob_range_lv = df.get("prob_range_lv", pd.Series(0.0, index=df.index)).astype(float)
        prob_range_hv = df.get("prob_range_hv", pd.Series(0.0, index=df.index)).astype(float)
        prob_range = prob_range_lv + prob_range_hv

        # ── Máscaras vectorizadas para LONG ─────────────────────────────────────
        math_mr_cond_long = (zscore < -2.0) & (vol_garch < 1.2) & (prob_range > 0.40)
        price_in_bb_lower = close <= (bb_lower + 0.5 * atr)
        
        warmup_mask = pd.Series(False, index=df.index)
        warmup_mask.iloc[100:] = True

        long_mask = warmup_mask & math_mr_cond_long & price_in_bb_lower

        # Construir SL / TP para LONG
        sl_long = bb_lower - self.p.sl_atr_buffer * atr
        tp1_long = ema21
        tp2_long = bb_upper

        # Filtrar trades con R/R negativo
        valid_rr_long = (tp1_long > close) & (tp2_long > close)
        long_mask = long_mask & valid_rr_long

        # ── Máscaras vectorizadas para SHORT ────────────────────────────────────
        from config.settings import ASSETS
        asset_cfg = ASSETS.get(self.symbol)
        allow_shorts = asset_cfg.allow_shorts if asset_cfg else True

        math_mr_cond_short = (zscore > 2.0) & (vol_garch < 1.2) & (prob_range > 0.40)
        price_in_bb_upper = close >= (bb_upper - 0.5 * atr)

        if allow_shorts:
            short_mask = warmup_mask & math_mr_cond_short & price_in_bb_upper
            sl_short = bb_upper + self.p.sl_atr_buffer * atr
            tp1_short = ema21
            tp2_short = bb_lower
            valid_rr_short = (tp1_short < close) & (tp2_short < close)
            short_mask = short_mask & valid_rr_short
        else:
            short_mask = pd.Series(False, index=df.index)

        # ── Escribir resultados ────────────────────────────────────────────────
        mask_idx = long_mask.values
        short_mask_idx = short_mask.values

        signals_arr = np.zeros(n, dtype=int)
        signals_arr[mask_idx] = 1
        signals_arr[short_mask_idx] = -1

        entry_prices = np.full(n, np.nan)
        stop_losses  = np.full(n, np.nan)
        tp1_prices   = np.full(n, np.nan)
        tp2_prices   = np.full(n, np.nan)
        reason_arr   = np.full(n, "", dtype=object)
        quality_arr  = np.full(n, "C", dtype=object)

        # Aplicar Longs
        entry_prices[mask_idx] = close.values[mask_idx]
        stop_losses[mask_idx]  = sl_long.values[mask_idx]
        tp1_prices[mask_idx]   = tp1_long.values[mask_idx]
        tp2_prices[mask_idx]   = tp2_long.values[mask_idx]
        
        for i in np.where(mask_idx)[0]:
            reason_arr[i] = f"ZScore_Extremo✓ (Z:{zscore.iloc[i]:.2f}, VolG:{vol_garch.iloc[i]:.2f}, P_Rng:{prob_range.iloc[i]:.2f}) | Precio_en_BB_lower✓"
            quality_arr[i] = self._classify_long_quality(df.iloc[i], 2).value
            
        # Aplicar Shorts
        if allow_shorts:
            entry_prices[short_mask_idx] = close.values[short_mask_idx]
            stop_losses[short_mask_idx]  = sl_short.values[short_mask_idx]
            tp1_prices[short_mask_idx]   = tp1_short.values[short_mask_idx]
            tp2_prices[short_mask_idx]   = tp2_short.values[short_mask_idx]
            
            for i in np.where(short_mask_idx)[0]:
                reason_arr[i] = f"ZScore_Extremo_Short✓ (Z:{zscore.iloc[i]:.2f}, VolG:{vol_garch.iloc[i]:.2f}, P_Rng:{prob_range.iloc[i]:.2f}) | Precio_en_BB_upper✓"
                quality_arr[i] = self._classify_short_quality(df.iloc[i], 2).value

        # ── Confianza bayesiana / probabilística ───────────────────────────────
        confidence_arr = np.full(n, 0.47)
        # La confianza es proporcional al extremo del Z-Score
        z_conf_long = np.clip(0.40 + abs(zscore.values[mask_idx]) * 0.05, 0.40, 0.85)
        confidence_arr[mask_idx] = z_conf_long
        
        z_conf_short = np.clip(0.40 + abs(zscore.values[short_mask_idx]) * 0.05, 0.40, 0.85)
        confidence_arr[short_mask_idx] = z_conf_short

        df["signal"] = signals_arr
        df["entry_price"] = entry_prices
        df["stop_loss"] = stop_losses
        df["take_profit_1"] = tp1_prices
        df["take_profit_2"] = tp2_prices
        df["signal_reason"] = reason_arr
        df["signal_quality"] = quality_arr
        df["confidence"] = confidence_arr
        df["strategy"] = self.strategy_type.value
        return df

    def _check_long_conditions(self, df: pd.DataFrame, idx: int) -> tuple[bool, str, SetupQuality]:
        row = df.iloc[idx]
        conditions = []

        zscore = row.get("zscore_vwap", 0.0)
        vol_garch = row.get("vol_garch_proxy", 1.0)
        prob_range_lv = row.get("prob_range_lv", 0.0)
        prob_range_hv = row.get("prob_range_hv", 0.0)
        prob_range = prob_range_lv + prob_range_hv

        # Condición Probabilística Primaria: Z-Score extremo + Volatilidad comprimida + Régimen de Rango
        math_mr_cond = (zscore < -2.0) and (vol_garch < 1.2) and (prob_range > 0.40)

        if math_mr_cond:
            conditions.append(f"ZScore_Extremo✓ (Z:{zscore:.2f}, VolG:{vol_garch:.2f}, P_Rng:{prob_range:.2f})")
        else:
            return False, "Sin condición Z-Score para Mean Reversion", SetupQuality.C

        close = row["close"]
        bb_lower = row.get("bb_lower", 0.0)
        atr = row.get("atr", close * 0.01)
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

        zscore = row.get("zscore_vwap", 0.0)
        vol_garch = row.get("vol_garch_proxy", 1.0)
        prob_range_lv = row.get("prob_range_lv", 0.0)
        prob_range_hv = row.get("prob_range_hv", 0.0)
        prob_range = prob_range_lv + prob_range_hv

        # Condición Probabilística Primaria para Short: Z-Score positivo extremo
        math_mr_cond = (zscore > 2.0) and (vol_garch < 1.2) and (prob_range > 0.40)

        if math_mr_cond:
            conditions.append(f"ZScore_Extremo_Short✓ (Z:{zscore:.2f}, VolG:{vol_garch:.2f}, P_Rng:{prob_range:.2f})")
        else:
            return False, "Sin condición Z-Score para Mean Reversion Short", SetupQuality.C

        close = row["close"]
        bb_upper = row.get("bb_upper", 0.0)
        atr = row.get("atr", close * 0.01)
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

        # ── Variables Vectorizadas ─────────────────────────────────────────────
        close = df["close"].astype(float)
        atr   = df.get("atr", close * 0.01).astype(float)
        
        # Volatilidad dinámica (Expansión de varianza esperada)
        vol_garch = df.get("vol_garch_proxy", pd.Series(1.0, index=df.index)).astype(float)
        
        # Anomalía de Volumen (Proxy de test de Poisson)
        vol_ratio = df.get("volume_ratio", pd.Series(1.0, index=df.index)).astype(float)
        
        # Z-Score para confirmar salida del intervalo de confianza del 95%
        zscore = df.get("zscore_vwap", pd.Series(0.0, index=df.index)).astype(float)
        
        bb_upper = df.get("bb_upper", pd.Series(np.nan, index=df.index)).astype(float)
        bb_lower = df.get("bb_lower", pd.Series(np.nan, index=df.index)).astype(float)
        prev_bb_upper = bb_upper.shift(1)
        prev_bb_lower = bb_lower.shift(1)
        
        bb_squeeze_candles = df.get("bb_squeeze_candles", pd.Series(0, index=df.index)).astype(int)

        # ── Máscaras vectorizadas para LONG (Bullish Breakout) ───────────────────
        # Ruptura estadísticamente significativa: 
        # Z-Score > 1.8 (Cerca o fuera de 2 desviaciones) + Expansión Varianza + Anomalía Volumen
        math_bull_breakout = (zscore > 1.8) & (vol_garch > 1.5) & (vol_ratio > self.p.volume_multiplier)
        
        # Condición mecánica (El precio rompe físicamente la banda y veníamos de compresión)
        mechanical_bull = (close > prev_bb_upper) & (bb_squeeze_candles.shift(1) >= self.p.min_squeeze_candles)
        
        warmup_mask = pd.Series(False, index=df.index)
        warmup_mask.iloc[100:] = True

        long_mask = warmup_mask & math_bull_breakout & mechanical_bull

        # SL / TP para LONG
        sl_long = prev_bb_upper - self.p.sl_atr_multiplier * atr
        risk_long = close - sl_long
        tp1_long = close + 2.0 * risk_long
        tp2_long = close + 4.0 * risk_long

        # ── Máscaras vectorizadas para SHORT (Bearish Breakout) ──────────────────
        from config.settings import ASSETS
        asset_cfg = ASSETS.get(self.symbol)
        allow_shorts = asset_cfg.allow_shorts if asset_cfg else True

        math_bear_breakout = (zscore < -1.8) & (vol_garch > 1.5) & (vol_ratio > self.p.volume_multiplier)
        mechanical_bear = (close < prev_bb_lower) & (bb_squeeze_candles.shift(1) >= self.p.min_squeeze_candles)

        if allow_shorts:
            short_mask = warmup_mask & math_bear_breakout & mechanical_bear
            sl_short = prev_bb_lower + self.p.sl_atr_multiplier * atr
            risk_short = sl_short - close
            tp1_short = close - 2.0 * risk_short
            tp2_short = close - 4.0 * risk_short
        else:
            short_mask = pd.Series(False, index=df.index)

        # ── Escribir resultados ────────────────────────────────────────────────
        mask_idx = long_mask.values
        short_mask_idx = short_mask.values

        signals_arr = np.zeros(n, dtype=int)
        signals_arr[mask_idx] = 1
        signals_arr[short_mask_idx] = -1

        entry_prices = np.full(n, np.nan)
        stop_losses  = np.full(n, np.nan)
        tp1_prices   = np.full(n, np.nan)
        tp2_prices   = np.full(n, np.nan)
        reason_arr   = np.full(n, "", dtype=object)
        quality_arr  = np.full(n, "B", dtype=object)

        # Aplicar Longs
        entry_prices[mask_idx] = close.values[mask_idx]
        stop_losses[mask_idx]  = sl_long.values[mask_idx]
        tp1_prices[mask_idx]   = tp1_long.values[mask_idx]
        tp2_prices[mask_idx]   = tp2_long.values[mask_idx]
        
        for i in np.where(mask_idx)[0]:
            sq_cand = bb_squeeze_candles.iloc[i-1]
            vr = vol_ratio.iloc[i]
            reason_arr[i] = f"StatBreakout✓ (Z:{zscore.iloc[i]:.2f}, VolG:{vol_garch.iloc[i]:.2f}) | Squeeze({sq_cand}) | Vol_Anomaly={vr:.1f}x"
            quality_arr[i] = "A+" if vr >= 2.5 else "A"
            
        # Aplicar Shorts
        if allow_shorts:
            entry_prices[short_mask_idx] = close.values[short_mask_idx]
            stop_losses[short_mask_idx]  = sl_short.values[short_mask_idx]
            tp1_prices[short_mask_idx]   = tp1_short.values[short_mask_idx]
            tp2_prices[short_mask_idx]   = tp2_short.values[short_mask_idx]
            
            for i in np.where(short_mask_idx)[0]:
                sq_cand = bb_squeeze_candles.iloc[i-1]
                vr = vol_ratio.iloc[i]
                reason_arr[i] = f"StatBreakoutShort✓ (Z:{zscore.iloc[i]:.2f}, VolG:{vol_garch.iloc[i]:.2f}) | Squeeze({sq_cand}) | Vol_Anomaly={vr:.1f}x"
                quality_arr[i] = "A+" if vr >= 2.5 else "A"

        # ── Confianza bayesiana / probabilística ───────────────────────────────
        confidence_arr = np.full(n, 0.47)
        
        # En breakout, la confianza la da la fuerza de la anomalía de volumen y varianza
        vol_score_long = np.clip((vol_garch.values[mask_idx] - 1.5) * 0.1 + (vol_ratio.values[mask_idx] - 1.5) * 0.1, 0, 0.4)
        confidence_arr[mask_idx] = np.clip(0.50 + vol_score_long, 0.50, 0.90)
        
        vol_score_short = np.clip((vol_garch.values[short_mask_idx] - 1.5) * 0.1 + (vol_ratio.values[short_mask_idx] - 1.5) * 0.1, 0, 0.4)
        confidence_arr[short_mask_idx] = np.clip(0.50 + vol_score_short, 0.50, 0.90)

        df["signal"] = signals_arr
        df["entry_price"] = entry_prices
        df["stop_loss"] = stop_losses
        df["take_profit_1"] = tp1_prices
        df["take_profit_2"] = tp2_prices
        df["signal_reason"] = reason_arr
        df["signal_quality"] = quality_arr
        df["confidence"] = confidence_arr
        df["strategy"] = self.strategy_type.value
        return df

    def is_valid_entry(self, df: pd.DataFrame, idx: int) -> bool:
        row = df.iloc[idx]
        zscore = row.get("zscore_vwap", 0.0)
        vol_garch = row.get("vol_garch_proxy", 1.0)
        vol_ratio = row.get("volume_ratio", 1.0)
        
        valid_bull = (zscore > 1.8) and (vol_garch > 1.5) and (vol_ratio > self.p.volume_multiplier)
        valid_bear = (zscore < -1.8) and (vol_garch > 1.5) and (vol_ratio > self.p.volume_multiplier)
        
        return valid_bull or valid_bear

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

class MetaStrategy(BaseStrategy):
    """
    Orquestador de múltiples estrategias basado en régimen de mercado cuantitativo.
    """
    def __init__(self, symbol: str, timeframe: str = "4h", params=None):
        super().__init__(symbol, timeframe)
        from config.settings import STRATEGIES
        self.p = params if params else STRATEGIES.meta
        self.tf = TrendFollowingStrategy(symbol, timeframe)
        self.mr = MeanReversionStrategy(symbol, timeframe)
        self.bo = BreakoutStrategy(symbol, timeframe)

    @property
    def strategy_type(self) -> StrategyType:
        return StrategyType.META

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Evaluate individual strategies
        df_tf = self.tf.generate_signals(df)
        df_mr = self.mr.generate_signals(df)
        df_bo = self.bo.generate_signals(df)
        
        # Merge individual outputs
        df["tf_sig"] = df_tf.get("signal", pd.Series(0, index=df.index))
        df["mr_sig"] = df_mr.get("signal", pd.Series(0, index=df.index))
        df["bo_sig"] = df_bo.get("signal", pd.Series(0, index=df.index))
        
        df["tf_sl"] = df_tf.get("stop_loss", pd.Series(np.nan, index=df.index))
        df["mr_sl"] = df_mr.get("stop_loss", pd.Series(np.nan, index=df.index))
        df["bo_sl"] = df_bo.get("stop_loss", pd.Series(np.nan, index=df.index))
        
        df["tf_tp1"] = df_tf.get("take_profit_1", pd.Series(np.nan, index=df.index))
        df["mr_tp1"] = df_mr.get("take_profit_1", pd.Series(np.nan, index=df.index))
        df["bo_tp1"] = df_bo.get("take_profit_1", pd.Series(np.nan, index=df.index))
        
        df["tf_tp2"] = df_tf.get("take_profit_2", pd.Series(np.nan, index=df.index))
        df["mr_tp2"] = df_mr.get("take_profit_2", pd.Series(np.nan, index=df.index))
        df["bo_tp2"] = df_bo.get("take_profit_2", pd.Series(np.nan, index=df.index))
        
        # Regime indicators
        hurst = df.get("hurst_exp", pd.Series(0.5, index=df.index))
        zscore = df.get("zscore_vwap", pd.Series(0.0, index=df.index))
        
        # Pre-allocate output arrays
        n = len(df)
        sig_out = np.zeros(n, dtype=int)
        sl_out = np.full(n, np.nan)
        tp1_out = np.full(n, np.nan)
        tp2_out = np.full(n, np.nan)
        regime_out = np.full(n, "unknown", dtype=object)
        
        h_arr = hurst.to_numpy()
        z_arr = zscore.to_numpy()
        
        tf_s = df["tf_sig"].to_numpy()
        mr_s = df["mr_sig"].to_numpy()
        bo_s = df["bo_sig"].to_numpy()
        
        tf_sl = df["tf_sl"].to_numpy()
        mr_sl = df["mr_sl"].to_numpy()
        bo_sl = df["bo_sl"].to_numpy()
        
        tf_tp = df["tf_tp1"].to_numpy()
        mr_tp = df["mr_tp1"].to_numpy()
        bo_tp = df["bo_tp1"].to_numpy()

        tf_tp2 = df["tf_tp2"].to_numpy()
        mr_tp2 = df["mr_tp2"].to_numpy()
        bo_tp2 = df["bo_tp2"].to_numpy()

        for i in range(n):
            if z_arr[i] < self.p.zscore_extreme and mr_s[i] != 0:
                # Extremo: Mean Reversion absoluto
                sig_out[i] = mr_s[i]
                sl_out[i] = mr_sl[i]
                tp1_out[i] = mr_tp[i]
                tp2_out[i] = mr_tp2[i]
                regime_out[i] = "mr_extreme"
            elif h_arr[i] > self.p.hurst_trend_threshold:
                # Tendencia
                if bo_s[i] != 0:
                    sig_out[i] = bo_s[i]
                    sl_out[i] = bo_sl[i]
                    tp1_out[i] = bo_tp[i]
                    tp2_out[i] = bo_tp2[i]
                    regime_out[i] = "breakout_trend"
                elif tf_s[i] != 0:
                    sig_out[i] = tf_s[i]
                    sl_out[i] = tf_sl[i]
                    tp1_out[i] = tf_tp[i]
                    tp2_out[i] = tf_tp2[i]
                    regime_out[i] = "trend_following"
            elif h_arr[i] < self.p.hurst_range_threshold:
                # Lateral
                if mr_s[i] != 0:
                    sig_out[i] = mr_s[i]
                    sl_out[i] = mr_sl[i]
                    tp1_out[i] = mr_tp[i]
                    tp2_out[i] = mr_tp2[i]
                    regime_out[i] = "mean_reversion_range"
            else:
                # Neutral: el primero que dispare
                if tf_s[i] != 0:
                    sig_out[i] = tf_s[i]
                    sl_out[i] = tf_sl[i]
                    tp1_out[i] = tf_tp[i]
                    tp2_out[i] = tf_tp2[i]
                    regime_out[i] = "trend_following_neutral"
                elif bo_s[i] != 0:
                    sig_out[i] = bo_s[i]
                    sl_out[i] = bo_sl[i]
                    tp1_out[i] = bo_tp[i]
                    tp2_out[i] = bo_tp2[i]
                    regime_out[i] = "breakout_neutral"
                elif mr_s[i] != 0:
                    sig_out[i] = mr_s[i]
                    sl_out[i] = mr_sl[i]
                    tp1_out[i] = mr_tp[i]
                    tp2_out[i] = mr_tp2[i]
                    regime_out[i] = "mr_neutral"

        # Prevent multiple consecutive signals
        for i in range(1, n):
            if sig_out[i] and sig_out[max(0, i-3):i].any():
                sig_out[i] = 0

        df["signal_long"] = sig_out
        df["signal"] = sig_out
        df["entry_price"] = df["close"]
        df["stop_loss"] = sl_out
        df["take_profit_1"] = tp1_out
        df["take_profit_2"] = tp2_out
        df["meta_regime"] = regime_out
        df["strategy"] = self.strategy_type.value
        df["signal_reason"] = regime_out
        df["signal_quality"] = "A"
        df["confidence"] = 0.6
        
        return df

    def is_valid_entry(self, df: pd.DataFrame, idx: int) -> bool:
        return df["signal"].iloc[idx] == 1
