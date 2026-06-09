"""
Filtro de Régimen de Mercado y Detector de Fase de Ciclo Macro
=============================================================
Clasifica el mercado actual antes de cualquier operación y
analiza la fase del ciclo macro de Bitcoin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd

from config.settings import (
    ASSETS,
    INDICATORS,
    RISK,
    SESSIONS,
    STRATEGIES,
    AssetPriority,
    MarketRegime,
    StrategyType,
)
from config.logging_config import get_logger

logger = get_logger(__name__)


# ── Halvings conocidos (timestamps Unix aproximados) ──────────────────────────
HALVING_DATES = [
    "2012-11-28",  # Halving 1 — 50→25 BTC
    "2016-07-09",  # Halving 2 — 25→12.5 BTC
    "2020-05-11",  # Halving 3 — 12.5→6.25 BTC
    "2024-04-19",  # Halving 4 — 6.25→3.125 BTC (actual)
    "2028-04-01",  # Halving 5 — estimado
]

# ATH conocidos por ciclo (aproximados)
CYCLE_ATHS = {
    "2013": 1242.0,
    "2017": 19891.0,
    "2021": 68789.0,
    "2025": 126200.0,  # Oct 2025
}


@dataclass
class CycleState:
    """Estado del ciclo macro para un símbolo en un momento dado."""
    phase:           str    # BEAR_DEEP/BEAR_RECOVERY/ACCUMULATION/BULL_EARLY/BULL_MATURE/BULL_LATE/DISTRIBUTION
    phase_strength:  float  # 0-1, confianza en la clasificación
    days_since_ath:  int    # días desde el último ATH
    pct_from_ath:    float  # % por debajo del ATH del ciclo
    pct_from_200w:   float  # % respecto a la media de 200 semanas (precio "justo")
    rsi_daily:       float
    rsi_weekly:      float  # RSI sintetizado semanal
    ema_200d:        float  # EMA 200 diaria
    above_200d:      bool   # precio sobre EMA 200 diaria
    trend_weekly:    str    # "up" / "down" / "sideways"
    halving_context: str    # "pre_halving" / "post_halving_early" / "post_halving_late"
    days_to_next_halving: int

    # Estrategias activas en esta fase
    active_strategies: List[str] = field(default_factory=list)
    # Multiplicador de riesgo según la fase
    risk_multiplier:   float = 1.0
    # Score de convicción de la fase (0-100)
    conviction_score:  int   = 0

    @property
    def is_bull(self) -> bool:
        return self.phase in ("BULL_EARLY", "BULL_MATURE", "BULL_LATE")

    @property
    def is_bear(self) -> bool:
        return self.phase in ("BEAR_DEEP", "BEAR_RECOVERY")

    @property
    def is_accumulation(self) -> bool:
        return self.phase in ("ACCUMULATION", "BEAR_RECOVERY")

    @property
    def description(self) -> str:
        desc = {
            "BEAR_DEEP":      "Bear profundo — preservar capital",
            "BEAR_RECOVERY":  "Recuperación desde mínimos — acumular con cautela",
            "ACCUMULATION":   "Acumulación — construir posición",
            "BULL_EARLY":     "Bull temprano — momentum creciente",
            "BULL_MATURE":    "Bull maduro — tendencia confirmada",
            "BULL_LATE":      "Bull tardío — reducir exposición",
            "DISTRIBUTION":   "Distribución — salir del mercado",
        }
        return desc.get(self.phase, "Desconocido")


class CycleDetector:
    """
    Detector de fase de ciclo macro basado en datos OHLCV.
    """

    NEXT_HALVING = pd.Timestamp("2028-04-01", tz="UTC")

    BEAR_DEEP_THRESHOLD   = -0.50
    BEAR_RECOV_THRESHOLD  = -0.35
    ACCUM_THRESHOLD       = -0.20
    BULL_LATE_RSI_W       = 78.0
    DISTRIBUTION_RSI_W    = 85.0

    def __init__(self, lookback_days: int = 365):
        self.lookback = lookback_days

    def _ema(self, s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=n, adjust=False).mean()

    def _rsi(self, s: pd.Series, p: int = 14) -> pd.Series:
        d = s.diff()
        g = d.clip(lower=0).rolling(p).mean()
        l = (-d.clip(upper=0)).rolling(p).mean()
        return 100 - 100 / (1 + g / l.replace(0, np.nan))

    def _to_weekly(self, df_daily: pd.DataFrame) -> pd.DataFrame:
        df = df_daily.copy()
        df.index = pd.to_datetime(df["timestamp"]) if "timestamp" in df.columns else df.index
        weekly = df.resample("W").agg({
            "open":   "first",
            "high":   "max",
            "low":    "min",
            "close":  "last",
            "volume": "sum",
        }).dropna()
        return weekly

    def _classify_halving_context(self, now: pd.Timestamp) -> Tuple[str, int]:
        last_halving = pd.Timestamp("2024-04-19", tz="UTC")
        next_halving = self.NEXT_HALVING

        days_since_last = (now - last_halving).days
        days_to_next    = max(0, (next_halving - now).days)

        if days_since_last < 180:
            context = "post_halving_early"
        elif days_to_next < 180:
            context = "pre_halving"
        else:
            context = "post_halving_late"

        return context, days_to_next

    def detect(self, df_daily: pd.DataFrame) -> CycleState:
        if len(df_daily) < 100:
            return self._default_state()

        df = df_daily.copy().sort_values("timestamp").reset_index(drop=True)
        if df["timestamp"].dtype == np.int64:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        close = df["close"].astype(float)
        now   = df["timestamp"].iloc[-1]
        price = float(close.iloc[-1])

        # Indicadores diarios
        ema_50d  = float(self._ema(close, 50).iloc[-1])
        ema_200d = float(self._ema(close, 200).iloc[-1])
        rsi_14d  = float(self._rsi(close, 14).iloc[-1])
        rsi_28d  = float(self._rsi(close, 28).iloc[-1])

        # ATH del período
        ath_period = float(close.max())
        ath_idx    = close.idxmax()
        ath_date   = df["timestamp"].iloc[ath_idx]
        days_since_ath = max(0, (now - ath_date).days)
        pct_from_ath   = (price - ath_period) / ath_period

        # Media 200 días como proxy de 200-week SMA
        sma_200d = float(close.rolling(min(200, len(close))).mean().iloc[-1])
        pct_from_200d = (price - sma_200d) / sma_200d if sma_200d > 0 else 0

        # Indicadores semanales
        df_w = self._to_weekly(df)
        close_w  = df_w["close"].astype(float)
        rsi_w    = float(self._rsi(close_w, 14).iloc[-1]) if len(close_w) > 14 else 50.0

        if len(close_w) >= 8:
            slope = float(close_w.iloc[-4:].pct_change().mean()) * 100
            trend_weekly = "up" if slope > 1.0 else ("down" if slope < -1.0 else "sideways")
        else:
            trend_weekly = "sideways"

        halving_context, days_to_next = self._classify_halving_context(now)

        scores: Dict[str, int] = {
            "BEAR_DEEP":      0,
            "BEAR_RECOVERY":  0,
            "ACCUMULATION":   0,
            "BULL_EARLY":     0,
            "BULL_MATURE":    0,
            "BULL_LATE":      0,
            "DISTRIBUTION":   0,
        }

        above_200d = price > ema_200d
        above_50d  = price > ema_50d
        above_200w_proxy = price > sma_200d

        if pct_from_ath < self.BEAR_DEEP_THRESHOLD:
            scores["BEAR_DEEP"] += 40
        if not above_200d and not above_50d:
            scores["BEAR_DEEP"] += 25
        if rsi_14d < 35:
            scores["BEAR_DEEP"] += 20
        if rsi_w < 40:
            scores["BEAR_DEEP"] += 15

        if self.BEAR_DEEP_THRESHOLD <= pct_from_ath < self.BEAR_RECOV_THRESHOLD:
            scores["BEAR_RECOVERY"] += 35
        if not above_200d and above_50d:
            scores["BEAR_RECOVERY"] += 25
        if 35 <= rsi_14d <= 50:
            scores["BEAR_RECOVERY"] += 20
        if trend_weekly == "up" and not above_200w_proxy:
            scores["BEAR_RECOVERY"] += 20

        if self.BEAR_RECOV_THRESHOLD <= pct_from_ath < self.ACCUM_THRESHOLD:
            scores["ACCUMULATION"] += 35
        if above_200d and not above_50d:
            scores["ACCUMULATION"] += 20
        if 45 <= rsi_14d <= 60:
            scores["ACCUMULATION"] += 20
        if trend_weekly == "sideways":
            scores["ACCUMULATION"] += 25

        if self.ACCUM_THRESHOLD <= pct_from_ath < -0.05:
            scores["BULL_EARLY"] += 30
        if above_200d and above_50d and halving_context == "post_halving_early":
            scores["BULL_EARLY"] += 30
        if 50 <= rsi_14d <= 65:
            scores["BULL_EARLY"] += 20
        if trend_weekly == "up" and above_200w_proxy:
            scores["BULL_EARLY"] += 20

        if -0.05 <= pct_from_ath <= 0.20:
            scores["BULL_MATURE"] += 40
        if above_200d and above_50d:
            scores["BULL_MATURE"] += 25
        if 60 <= rsi_w < self.BULL_LATE_RSI_W:
            scores["BULL_MATURE"] += 20
        if trend_weekly == "up":
            scores["BULL_MATURE"] += 15

        if rsi_w >= self.BULL_LATE_RSI_W:
            scores["BULL_LATE"] += 50
        if pct_from_ath > 0:
            scores["BULL_LATE"] += 30
        if rsi_14d > 75:
            scores["BULL_LATE"] += 20

        if rsi_w >= self.DISTRIBUTION_RSI_W:
            scores["DISTRIBUTION"] += 50
        if days_since_ath < 30 and rsi_14d < rsi_28d:
            scores["DISTRIBUTION"] += 30
        if pct_from_ath > 0.10:
            scores["DISTRIBUTION"] += 20

        phase = max(scores, key=scores.get)
        total_score = scores[phase]
        phase_strength = min(total_score / 100.0, 1.0)

        strategy_map = {
            "BEAR_DEEP":     {"strategies": ["MeanReversion"],     "risk": 0.25},
            "BEAR_RECOVERY": {"strategies": ["MeanReversion"],     "risk": 0.40},
            "ACCUMULATION":  {"strategies": ["MeanReversion","Breakout"], "risk": 0.60},
            "BULL_EARLY":    {"strategies": ["TrendFollowing","MeanReversion","Breakout"], "risk": 0.85},
            "BULL_MATURE":   {"strategies": ["TrendFollowing","MomentumScalp","Breakout"], "risk": 1.0},
            "BULL_LATE":     {"strategies": ["TrendFollowing","MomentumScalp"],  "risk": 0.70},
            "DISTRIBUTION":  {"strategies": ["MeanReversion"],     "risk": 0.30},
        }

        config = strategy_map.get(phase, {"strategies": ["MeanReversion"], "risk": 0.5})

        return CycleState(
            phase           = phase,
            phase_strength  = round(phase_strength, 2),
            days_since_ath  = days_since_ath,
            pct_from_ath    = round(pct_from_ath * 100, 1),
            pct_from_200w   = round(pct_from_200d * 100, 1),
            rsi_daily       = round(rsi_14d, 1),
            rsi_weekly      = round(rsi_w, 1),
            ema_200d        = round(ema_200d, 4),
            above_200d      = above_200d,
            trend_weekly    = trend_weekly,
            halving_context = halving_context,
            days_to_next_halving = days_to_next,
            active_strategies   = config["strategies"],
            risk_multiplier     = config["risk"],
            conviction_score    = min(total_score, 100),
        )

    def _default_state(self) -> CycleState:
        return CycleState(
            phase="ACCUMULATION", phase_strength=0.3,
            days_since_ath=0, pct_from_ath=0.0, pct_from_200w=0.0,
            rsi_daily=50.0, rsi_weekly=50.0, ema_200d=0.0,
            above_200d=True, trend_weekly="sideways",
            halving_context="post_halving_late", days_to_next_halving=700,
            active_strategies=["MeanReversion","TrendFollowing"],
            risk_multiplier=0.5, conviction_score=30,
        )


def load_daily_ohlcv(symbol: str, db_path: str) -> pd.DataFrame:
    """Carga velas diarias desde SQLite para un símbolo."""
    import sqlite3
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(
        "SELECT timestamp,open,high,low,close,volume FROM ohlcv "
        "WHERE symbol=? AND timeframe='1d' ORDER BY timestamp ASC",
        conn, params=(symbol,)
    )
    conn.close()
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


class RegimeFilter:
    """
    Filtra condiciones de mercado y determina qué estrategia aplicar.
    """

    def detect_regime(self, df: pd.DataFrame, idx: int = -1) -> MarketRegime:
        if idx == -1:
            idx = len(df) - 1

        row = df.iloc[idx]

        # Check de volatilidad extrema
        atr_regime = row.get("atr_regime", "normal")
        atr_tradeable = row.get("atr_tradeable", True)

        if not atr_tradeable or atr_regime == "extreme_vol":
            return MarketRegime.HIGH_VOLATILITY

        # Clasificar tendencia por EMAs y ADX
        ema21_col = f"ema_{INDICATORS.trend.ema_fast}"
        if ema21_col not in df.columns and "ema21" in df.columns:
            ema21_col = "ema21"
        ema55_col = f"ema_{INDICATORS.trend.ema_mid}"
        if ema55_col not in df.columns and "ema55" in df.columns:
            ema55_col = "ema55"
        ema200_col = f"ema_{INDICATORS.trend.ema_slow}"
        if ema200_col not in df.columns and "ema200" in df.columns:
            ema200_col = "ema200"

        required = [ema21_col, ema55_col, ema200_col, "adx"]
        for col in required:
            if col not in df.columns or pd.isna(row.get(col)):
                return MarketRegime.UNKNOWN

        ema21 = row[ema21_col]
        ema55 = row[ema55_col]
        ema200 = row[ema200_col]
        adx = row["adx"]

        # Tendencia alcista fuerte
        if ema21 > ema55 > ema200 and adx > INDICATORS.trend.adx_trend_threshold:
            return MarketRegime.BULLISH_TREND

        # Tendencia bajista fuerte
        if ema21 < ema55 < ema200 and adx > INDICATORS.trend.adx_trend_threshold:
            return MarketRegime.BEARISH_TREND

        # Alta volatilidad (ATR > 2x media) pero no extrema
        if atr_regime in ("high_vol", "very_high_vol"):
            return MarketRegime.HIGH_VOLATILITY

        # Rango: ADX bajo sin tendencia clara
        if adx < INDICATORS.trend.adx_trend_threshold:
            return MarketRegime.RANGE

        return MarketRegime.UNKNOWN

    def get_active_strategies(self, regime: MarketRegime) -> List[StrategyType]:
        if regime == MarketRegime.BULLISH_TREND:
            return [StrategyType.TREND_FOLLOWING, StrategyType.MEAN_REVERSION, StrategyType.META]
        elif regime == MarketRegime.BEARISH_TREND:
            return [StrategyType.TREND_FOLLOWING, StrategyType.META]
        elif regime == MarketRegime.RANGE:
            return [StrategyType.MEAN_REVERSION, StrategyType.META]
        elif regime == MarketRegime.HIGH_VOLATILITY:
            return [StrategyType.TREND_FOLLOWING, StrategyType.BREAKOUT, StrategyType.META]
        return [StrategyType.META]

    def get_position_size_multiplier(self, regime: MarketRegime) -> float:
        return 1.0

    def is_valid_trading_session(self, timestamp: Optional[datetime] = None) -> Tuple[bool, str]:
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        hour = timestamp.hour

        # Evitar baja liquidez Asia
        if SESSIONS.low_liquidity_start <= hour < SESSIONS.low_liquidity_end:
            return False, f"Baja liquidez Asia (UTC {hour:02d}:xx)"

        # Fines de semana
        if timestamp.weekday() >= 5:
            return False, "Fin de semana (spreads altos)"

        # Sesiones de alta liquidez
        london_open = SESSIONS.london_open[0] <= hour < SESSIONS.london_close[0]
        ny_open = SESSIONS.ny_open[0] <= hour < SESSIONS.ny_close[0]

        if london_open:
            return True, "London Open"
        if ny_open:
            return True, "New York Open"

        # Fuera de sesión preferida pero horario razonable
        if SESSIONS.london_close[0] <= hour < SESSIONS.ny_open[0]:
            return True, "Entre sesiones (aceptable)"

        return False, f"Fuera de sesión preferida (UTC {hour:02d}:xx)"

    def is_near_news_event(
        self, timestamp: Optional[datetime] = None, news_times: Optional[List[datetime]] = None
    ) -> bool:
        if not news_times:
            return False

        from datetime import timedelta
        buffer = timedelta(minutes=SESSIONS.news_buffer_minutes)
        ts = timestamp or datetime.now(timezone.utc)

        return any(abs((ts - news_time).total_seconds()) < buffer.total_seconds() for news_time in news_times)

    def passes_asset_filter(self, symbol: str, current_regime: MarketRegime) -> Tuple[bool, str]:
        # Normalizar pares USDT → USDC para permitir backtesting histórico con USDT
        # La operativa en producción siempre usa USDC; esto es solo para el lookup de prioridad
        lookup_symbol = symbol
        if symbol.endswith("/USDT"):
            usdc_equivalent = symbol.replace("/USDT", "/USDC")
            if usdc_equivalent in ASSETS:
                lookup_symbol = usdc_equivalent

        if lookup_symbol not in ASSETS:
            # Par desconocido — permitir con prioridad media por defecto para no bloquear backtesting
            return True, f"{symbol} no está en universo — permitido con prioridad media"

        asset = ASSETS[lookup_symbol]

        if asset.priority == AssetPriority.LOW:
            if current_regime not in [MarketRegime.BULLISH_TREND, MarketRegime.BEARISH_TREND]:
                return False, f"{symbol} (P3) requiere tendencia fuerte. Régimen actual: {current_regime.value}"

        return True, f"{symbol} aprobado (P{asset.priority.value})"

    def full_filter(
        self,
        df: pd.DataFrame,
        symbol: str,
        idx: int = -1,
        timestamp: Optional[datetime] = None,
    ) -> Tuple[bool, MarketRegime, str]:
        regime = self.detect_regime(df, idx)

        if regime == MarketRegime.HIGH_VOLATILITY:
            return False, regime, "ATR extremo: no operar"

        asset_ok, asset_reason = self.passes_asset_filter(symbol, regime)
        if not asset_ok:
            return False, regime, asset_reason

        return True, regime, f"Régimen: {regime.value}"


    def detect_volatility_regime(
        self,
        df: pd.DataFrame,
        lookback_candles: int = 540,  # ~90 días en 4H
    ) -> dict:
        """
        Detecta el régimen estadístico de volatilidad usando el percentil
        histórico del ATR. Complementa el análisis macro del CycleDetector.

        Regímenes:
          LOW_VOL    (ATR < P30): Compresión — esperar expansión, preferir Breakout
          NORMAL_VOL (P30-P70):  Estable   — Trend Following óptimo
          HIGH_VOL   (P70-P90):  Expansión — Mean Reversion (precios exagerados)
          EXTREME_VOL (> P90):   Caos      — No operar, preservar capital

        Returns:
            dict con régimen, percentil ATR, multiplicador de sizing recomendado
            y lista de estrategias preferidas para este régimen.
        """
        atr_col = "atr"
        if atr_col not in df.columns:
            return {
                "regime": "NORMAL_VOL",
                "atr_percentile": 0.5,
                "sizing_multiplier": 1.0,
                "preferred_strategies": ["trend_following", "mean_reversion"],
                "warning": "ATR no disponible en DataFrame",
            }

        atr_series = df[atr_col].dropna().astype(float)
        if len(atr_series) < 50:
            return {
                "regime": "NORMAL_VOL",
                "atr_percentile": 0.5,
                "sizing_multiplier": 1.0,
                "preferred_strategies": ["trend_following", "mean_reversion"],
                "warning": "Insuficientes datos ATR (<50 velas)",
            }

        window = atr_series.iloc[-lookback_candles:] if len(atr_series) > lookback_candles else atr_series
        current_atr = float(atr_series.iloc[-1])
        pct = float((window < current_atr).mean())

        # Clasificación con tabla de trading strategies óptimas por régimen
        if pct < 0.30:
            regime       = "LOW_VOL"
            sizing_mult  = 0.85      # Reducir un poco — movimientos pequeños
            strategies   = ["breakout", "trend_following"]
            description  = "Volatilidad comprimida — esperar expansión, preferir Breakout"
        elif pct < 0.70:
            regime       = "NORMAL_VOL"
            sizing_mult  = 1.0       # Sizing estándar
            strategies   = ["trend_following", "mean_reversion"]
            description  = "Volatilidad normal — Trend Following óptimo"
        elif pct < 0.90:
            regime       = "HIGH_VOL"
            sizing_mult  = 0.75      # Reducir riesgo — precios exagerados
            strategies   = ["mean_reversion"]
            description  = "Volatilidad expandida — Mean Reversion (reversión a media)"
        else:
            regime       = "EXTREME_VOL"
            sizing_mult  = 0.0       # NO OPERAR
            strategies   = []
            description  = "Volatilidad extrema — preservar capital, no operar"

        return {
            "regime":               regime,
            "atr_percentile":       round(pct, 4),
            "current_atr":          round(current_atr, 6),
            "sizing_multiplier":    sizing_mult,
            "preferred_strategies": strategies,
            "description":          description,
        }


def classify_regime(df: pd.DataFrame) -> str:
    """Wrapper a nivel de módulo para clasificar el régimen de mercado."""
    filter_instance = RegimeFilter()
    regime = filter_instance.detect_regime(df)
    return regime.value

