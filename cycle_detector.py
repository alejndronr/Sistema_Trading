"""
cycle_detector.py — Detector de Fase de Ciclo Macro
====================================================
Lee velas diarias y semanales para clasificar en qué fase del ciclo
de 4 años del halving de Bitcoin nos encontramos.

Las fases son:
  BEAR_DEEP      — caída severa (>50% bajo ATH), mercado capitulando
  BEAR_RECOVERY  — recuperación desde mínimos, aún bajo EMAs clave
  ACCUMULATION   — precio consolidando, smart money acumulando
  BULL_EARLY     — impulso inicial post-halving, tendencia girando alcista
  BULL_MATURE    — tendencia alcista confirmada, máximos crecientes
  BULL_LATE      — euforia, RSI extremo, divergencias, cerca del techo
  DISTRIBUTION   — techo de ciclo, distribución institucional

Inputs:
  df_daily  — DataFrame con velas 1D (mínimo 200 velas)
  df_weekly — DataFrame con velas 1W sintetizadas desde 1D

Output:
  CycleState — dataclass con fase, fuerza, métricas clave y estrategias activas
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


# ── Halvings conocidos (timestamps Unix aproximados) ──────────────────────────
# Usamos los precios de cada halving como referencia de ciclo
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
    Detector de fase de ciclo macro basado en datos diarios.
    
    No depende de APIs externas — usa solo los datos OHLCV locales.
    Diseñado para correr una vez al día (no en cada vela 1H).
    """

    # Próximo halving estimado
    NEXT_HALVING = pd.Timestamp("2028-04-01", tz="UTC")

    # Thresholds calibrados con datos históricos 2016-2026
    BEAR_DEEP_THRESHOLD   = -0.50   # >50% bajo ATH = bear profundo
    BEAR_RECOV_THRESHOLD  = -0.35   # 35-50% bajo ATH = recuperación
    ACCUM_THRESHOLD       = -0.20   # 20-35% bajo ATH = acumulación
    BULL_LATE_RSI_W       = 78.0    # RSI semanal >78 = bull tardío
    DISTRIBUTION_RSI_W    = 85.0    # RSI semanal >85 = distribución

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
        """Sintetiza velas semanales desde datos diarios."""
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
        """Determina en qué parte del ciclo de halving estamos."""
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
        """
        Clasifica la fase de ciclo actual usando datos diarios.
        
        Args:
            df_daily: DataFrame con columnas [timestamp, open, high, low, close, volume]
                     Mínimo 200 velas recomendado (200 días = ~7 meses)
        
        Returns:
            CycleState con la fase detectada y métricas asociadas
        """
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

        # ── Indicadores diarios ────────────────────────────────────────────────
        ema_50d  = float(self._ema(close, 50).iloc[-1])
        ema_200d = float(self._ema(close, 200).iloc[-1])
        rsi_14d  = float(self._rsi(close, 14).iloc[-1])
        rsi_28d  = float(self._rsi(close, 28).iloc[-1])

        # ATH del período disponible
        ath_period = float(close.max())
        ath_idx    = close.idxmax()
        ath_date   = df["timestamp"].iloc[ath_idx]
        days_since_ath = max(0, (now - ath_date).days)
        pct_from_ath   = (price - ath_period) / ath_period

        # Media 200 días como proxy de 200-week SMA
        sma_200d = float(close.rolling(min(200, len(close))).mean().iloc[-1])
        pct_from_200d = (price - sma_200d) / sma_200d if sma_200d > 0 else 0

        # ── Indicadores semanales ──────────────────────────────────────────────
        df_w = self._to_weekly(df)
        close_w  = df_w["close"].astype(float)
        rsi_w    = float(self._rsi(close_w, 14).iloc[-1]) if len(close_w) > 14 else 50.0
        ema_50w  = float(self._ema(close_w, 50).iloc[-1]) if len(close_w) > 50 else price
        ema_20w  = float(self._ema(close_w, 20).iloc[-1]) if len(close_w) > 20 else price

        # Tendencia semanal
        if len(close_w) >= 8:
            slope = float(close_w.iloc[-4:].pct_change().mean()) * 100
            trend_weekly = "up" if slope > 1.0 else ("down" if slope < -1.0 else "sideways")
        else:
            trend_weekly = "sideways"

        # Contexto de halving
        halving_context, days_to_next = self._classify_halving_context(now)

        # ── Scoring de fase ────────────────────────────────────────────────────
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

        # BEAR_DEEP: caída severa + por debajo de todas las medias + RSI bajo
        if pct_from_ath < self.BEAR_DEEP_THRESHOLD:
            scores["BEAR_DEEP"] += 40
        if not above_200d and not above_50d:
            scores["BEAR_DEEP"] += 25
        if rsi_14d < 35:
            scores["BEAR_DEEP"] += 20
        if rsi_w < 40:
            scores["BEAR_DEEP"] += 15

        # BEAR_RECOVERY: subiendo desde mínimos pero aún débil
        if self.BEAR_DEEP_THRESHOLD <= pct_from_ath < self.BEAR_RECOV_THRESHOLD:
            scores["BEAR_RECOVERY"] += 35
        if not above_200d and above_50d:  # Recuperando pero bajo 200d
            scores["BEAR_RECOVERY"] += 25
        if 35 <= rsi_14d <= 50:
            scores["BEAR_RECOVERY"] += 20
        if trend_weekly == "up" and not above_200w_proxy:
            scores["BEAR_RECOVERY"] += 20

        # ACCUMULATION: cerca de medias, sin dirección clara
        if self.BEAR_RECOV_THRESHOLD <= pct_from_ath < self.ACCUM_THRESHOLD:
            scores["ACCUMULATION"] += 35
        if above_200d and not above_50d:
            scores["ACCUMULATION"] += 20
        if 45 <= rsi_14d <= 60:
            scores["ACCUMULATION"] += 20
        if trend_weekly == "sideways":
            scores["ACCUMULATION"] += 25

        # BULL_EARLY: girando alcista, EMAs cruzando
        if self.ACCUM_THRESHOLD <= pct_from_ath < -0.05:
            scores["BULL_EARLY"] += 30
        if above_200d and above_50d and halving_context == "post_halving_early":
            scores["BULL_EARLY"] += 30
        if 50 <= rsi_14d <= 65:
            scores["BULL_EARLY"] += 20
        if trend_weekly == "up" and above_200w_proxy:
            scores["BULL_EARLY"] += 20

        # BULL_MATURE: tendencia alcista confirmada, cerca o en ATH
        if -0.05 <= pct_from_ath <= 0.20:  # Cerca o en ATH
            scores["BULL_MATURE"] += 40
        if above_200d and above_50d:
            scores["BULL_MATURE"] += 25
        if 60 <= rsi_w < self.BULL_LATE_RSI_W:
            scores["BULL_MATURE"] += 20
        if trend_weekly == "up":
            scores["BULL_MATURE"] += 15

        # BULL_LATE: RSI extremo, euforia
        if rsi_w >= self.BULL_LATE_RSI_W:
            scores["BULL_LATE"] += 50
        if pct_from_ath > 0:  # En o sobre ATH histórico
            scores["BULL_LATE"] += 30
        if rsi_14d > 75:
            scores["BULL_LATE"] += 20

        # DISTRIBUTION: divergencias + RSI extremo + cerca del techo
        if rsi_w >= self.DISTRIBUTION_RSI_W:
            scores["DISTRIBUTION"] += 50
        if days_since_ath < 30 and rsi_14d < rsi_28d:  # Divergencia bajista
            scores["DISTRIBUTION"] += 30
        if pct_from_ath > 0.10:  # Muy por encima del ATH anterior
            scores["DISTRIBUTION"] += 20

        # Determinar fase ganadora
        phase = max(scores, key=scores.get)
        total_score = scores[phase]
        # Normalizar fuerza
        phase_strength = min(total_score / 100.0, 1.0)

        # ── Estrategias activas por fase ───────────────────────────────────────
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


if __name__ == "__main__":
    import sys
    from pathlib import Path

    db = str(Path(__file__).parent / "data" / "db" / "trading.db")
    detector = CycleDetector()

    symbols = ["BTC/USDC", "ETH/USDC", "SOL/USDC", "BNB/USDC", "LINK/USDC", "AVAX/USDC"]
    print("\n═" * 30)
    print("  Análisis de Ciclo Macro — Todos los pares")
    print("═" * 30)

    for sym in symbols:
        df_d = load_daily_ohlcv(sym, db)
        if df_d.empty or len(df_d) < 50:
            print(f"  {sym}: datos insuficientes")
            continue
        state = detector.detect(df_d)
        strategies = ", ".join(state.active_strategies)
        print(f"\n  {sym}")
        print(f"    Fase:          {state.phase} (convicción {state.conviction_score}/100)")
        print(f"    Descripción:   {state.description}")
        print(f"    Desde ATH:     {state.pct_from_ath:+.1f}% ({state.days_since_ath} días)")
        print(f"    RSI diario:    {state.rsi_daily:.0f} | RSI semanal: {state.rsi_weekly:.0f}")
        print(f"    Sobre EMA200d: {'✅' if state.above_200d else '❌'}")
        print(f"    Tendencia sem: {state.trend_weekly}")
        print(f"    Halving:       {state.halving_context} ({state.days_to_next_halving}d para el próximo)")
        print(f"    Estrategias:   {strategies}")
        print(f"    Risk mult.:    {state.risk_multiplier:.0%}")
