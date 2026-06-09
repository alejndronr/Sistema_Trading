"""
live_engine_v5.py — Motor de Ejecución Cuantitativo V5.0 «Omniscient»
======================================================================
Evolución del V4 con inteligencia de mercado completa:

NUEVAS CAPAS vs V4:
──────────────────────────────────────────────────────────────────────
1. PATRONES DE VELAS (18 patrones japoneses)
   Hammer, Engulfing, Morning/Evening Star, Doji, Marubozu,
   Three Soldiers/Crows, Piercing, Dark Cloud, Harami, Tweezer
   → +5 a +20 puntos por patrón confirmado en contexto de régimen

2. FIGURAS CHARTISTAS (8 figuras)
   H&S, Doble Suelo/Techo, Triángulos (3 tipos), Bull/Bear Flag, Cuña
   → +12 a +25 puntos por figura detectada

3. SOPORTE/RESISTENCIA DINÁMICO
   Clustering manual de swing points (sin sklearn)
   → SL ajustado a zonas S/R reales + score si precio en confluencia

4. FIBONACCI AUTOMÁTICO
   TP1=1.272×, TP2=1.618×, TP3=2.618× del movimiento
   → Targets basados en extensiones Fibonacci, no múltiplos fijos de R

5. BOS/CHoCH MEJORADO
   Break of Structure y Change of Character como filtros de sesgo
   → +18 pts BOS, +22 pts CHoCH en dirección del trade

6. ORDER FLOW PROXY
   OBV vectorizado + CVD acumulado + absorción institucional
   → Confirmación de volumen institucional detrás de la señal

CORRECCIONES DE BUGS (propuesta externa):
──────────────────────────────────────────────────────────────────────
✓ Bug 1 CORREGIDO: Lookahead bias en backtest
  get_structure_context() usa df.iloc[-lookback:] — nunca ve el futuro
  Los swing points tienen delay de 5 velas de confirmación

✓ Bug 2 CORREGIDO: OBV vectorizado
  _obv_vectorized() usa np.sign + np.cumsum — 100% C, sin bucles Python

SCORE TOTAL = Técnicos + Velas + Figuras + SMC + S/R + Fibonacci + OF
Umbral mínimo adaptativo por régimen y estrategia.

Capital: $1,000 USDC · Pares: 6 × USDC
"""

from __future__ import annotations

import asyncio
import math
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog
from dotenv import load_dotenv
from sqlalchemy import text

import ccxt.async_support as ccxt

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

# ── Importar capa de indicadores V5 ───────────────────────────────────────────
try:
    from indicators.indicators_v5 import (
        enrich_v5,
        get_structure_context,
        CandlePatterns,
        Fibonacci,
        SRZones,
    )
    V5_INDICATORS = True
except ImportError:
    V5_INDICATORS = False

from ml.meta_labeler import MetaLabeler
from monitoring.telegram_bot import TelegramBot
from paper_portfolio import PaperPortfolio

log = structlog.get_logger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# ── Constantes ────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

ENGINE_VERSION = "5.0.0-Omniscient"

SYMBOLS: List[str] = [
    "BTC/USDC", "ETH/USDC", "SOL/USDC",
    "BNB/USDC", "LINK/USDC", "AVAX/USDC",
]

TF_STRUCTURE   = "1h"
TF_ENTRY       = "15m"
TF_STRUCT_SECS = 3600
TF_ENTRY_SECS  = 900

MAX_POSITIONS      = 3
COMMISSION_RATE    = 0.001
SLIPPAGE_ESTIMATE  = 0.001
INITIAL_CAPITAL    = float(os.environ.get("INITIAL_CAPITAL", "1000"))

# Riesgo
RISK_PER_TRADE_USD = 10.0
RISK_PCT_TIER_1    = 0.010
RISK_PCT_TIER_2    = 0.015
RISK_PCT_TIER_3    = 0.020
MAX_RISK_PER_TRADE = 0.020
KELLY_FRACTION     = 0.25

# Scores mínimos por estrategia
# Subidos significativamente para corregir el exceso de trades de TF
MIN_SCORE_TREND     = 75   # 75/100: solo setups A o A+
MIN_SCORE_MR        = 68   # era 65 — ligero ajuste
MIN_SCORE_BREAKOUT  = 65   # era 60
MIN_SCORE_MOMENTUM  = 65   # era 60
MIN_SCORE_V5_BONUS  = 15   # bonus MÍNIMO requerido de capas V5

# Confirmación V5 obligatoria: la señal debe tener al menos 1 capa V5 activa
# para entrar. Esto convierte las capas V5 en FILTROS, no en sumadores opcionales.
REQUIRE_V5_CONFIRMATION = True

# Correlación y sesión
CORRELATION_GROUPS: Dict[str, str] = {
    "BTC/USDC": "btc",
    "ETH/USDC": "altcoin", "SOL/USDC": "altcoin",
    "BNB/USDC": "altcoin", "LINK/USDC": "altcoin", "AVAX/USDC": "altcoin",
}
MAX_CORRELATED_POSITIONS = 1
MR_ALLOWED_HOURS_UTC: range = range(0, 8)
MR_ATR_STOP_MULTIPLIER: float = 2.5

# Circuit breakers
CB_DAILY_REDUCE  = -0.030
CB_DAILY_PAUSE   = -0.050
CB_PEAK_SHUTDOWN = -0.080

FAST_LOOP_SECS     = 60
CANDLE_BUFFER      = 8
MAX_RETRIES        = 5
CB_ERR_WINDOW      = 60
CB_ERR_LIMIT       = 4
COOLDOWN_AFTER_SL_MIN = 90


# ══════════════════════════════════════════════════════════════════════════════
# ── Dataclasses ───────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MarketRegime:
    symbol:   str
    regime:   str
    strength: float
    adx:      float
    atr_pct:  float
    trend_up: bool
    vfi:      float

    @property
    def is_tradeable(self) -> bool:
        return self.atr_pct < 8.0

    @property
    def risk_multiplier(self) -> float:
        return {"BULL": 1.0, "RANGE": 0.7, "HIGH_VOL": 0.5}.get(self.regime, 0.4)


@dataclass
class V5Context:
    """Contexto de las capas V5 para un símbolo."""
    candle_bull_score:  float = 0.0
    candle_bear_score:  float = 0.0
    structure_bias:     str   = "neutral"   # bull / bear / neutral
    structure_score:    float = 0.0
    sr_at_zone:         bool  = False
    sr_zone_score:      float = 0.0
    fib_near:           bool  = False
    fib_level:          str   = ""
    chart_bull_score:   float = 0.0
    chart_bear_score:   float = 0.0
    chart_patterns:     List[Dict] = field(default_factory=list)
    obv_rising:         bool  = False
    cvd_rising:         bool  = False
    obv_accelerating:   bool  = False   # OBV subiendo MÁS RÁPIDO que su media
    cvd_positive:       bool  = False   # Delta de volumen positivo neto (>0.1)
    absorption:         bool  = False
    sr_zones:           List[Dict] = field(default_factory=list)
    swing_highs:        List[Dict] = field(default_factory=list)
    swing_lows:         List[Dict] = field(default_factory=list)
    fib_tp1:            float = 0.0
    fib_tp2:            float = 0.0
    fib_tp3:            float = 0.0

    @property
    def total_bull_bonus(self) -> float:
        return (self.candle_bull_score + self.structure_score * 0.5 +
                self.sr_zone_score + self.chart_bull_score +
                (8  if self.fib_near else 0) +
                (8  if self.obv_accelerating else 0) +
                (8  if self.cvd_positive else 0) +
                (12 if self.absorption else 0) +
                (3  if self.obv_rising else 0) +
                (3  if self.cvd_rising else 0))

    @property
    def total_bear_bonus(self) -> float:
        return (self.candle_bear_score + self.structure_score * 0.5 +
                self.sr_zone_score + self.chart_bear_score +
                (8 if self.fib_near else 0))


@dataclass
class SignalResult:
    symbol:       str
    strategy:     str
    direction:    str
    score:        float
    entry_price:  float
    stop_loss:    float
    tp1:          float
    tp2:          float
    tp3:          float
    atr:          float
    regime:       MarketRegime
    quality:      str
    reasons:      List[str] = field(default_factory=list)
    ml_proba:     float = 0.5
    v5_context:   Optional[V5Context] = None


@dataclass
class SymbolCooldown:
    last_sl_time:  float = 0.0
    sl_count_24h:  int   = 0
    last_sl_reset: float = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# ── Indicadores base (del V4, mantenidos) ─────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def _rsi(c: pd.Series, p: int = 14) -> pd.Series:
    d = c.diff()
    g = d.clip(lower=0).rolling(p).mean()
    l = (-d.clip(upper=0)).rolling(p).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def _atr(df: pd.DataFrame, p: int = 14) -> pd.Series:
    hl  = df["high"] - df["low"]
    hpc = (df["high"] - df["close"].shift()).abs()
    lpc = (df["low"]  - df["close"].shift()).abs()
    return pd.concat([hl, hpc, lpc], axis=1).max(axis=1).ewm(span=p, adjust=False).mean()

def _bollinger(c: pd.Series, p: int = 20, s: float = 2.0):
    m = c.rolling(p).mean()
    σ = c.rolling(p).std()
    return m - s*σ, m, m + s*σ

def _macd(c: pd.Series, fast=12, slow=26, sig=9):
    ef = _ema(c, fast); es = _ema(c, slow)
    ml = ef - es; sl = _ema(ml, sig)
    return ml, sl, ml - sl

def _adx(df: pd.DataFrame, p: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    pdm = h.diff().clip(lower=0); mdm = (-l.diff()).clip(lower=0)
    pdm = pdm.where(pdm > mdm, 0); mdm = mdm.where(mdm > pdm, 0)
    atr_raw = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    atr_s   = atr_raw.ewm(span=p, adjust=False).mean()
    pdi = 100*pdm.ewm(span=p,adjust=False).mean()/atr_s.replace(0,np.nan)
    mdi = 100*mdm.ewm(span=p,adjust=False).mean()/atr_s.replace(0,np.nan)
    dx  = (100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan))
    return dx.ewm(span=p, adjust=False).mean().fillna(0)

def _vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["high"]+df["low"]+df["close"])/3
    return (tp*df["volume"]).cumsum()/df["volume"].cumsum().replace(0,np.nan)

def _vfi(df: pd.DataFrame, p: int = 130) -> pd.Series:
    tp = (df["high"]+df["low"]+df["close"])/3
    tp_s = tp.clip(lower=1e-10)
    inter = np.log(tp_s) - np.log(tp_s.shift(1))
    vi = inter.rolling(30).std().fillna(0.01)
    cut = 0.1*vi*df["close"]
    vave = df["volume"].rolling(p).mean().shift(1).fillna(1)
    vmax = vave*2.0
    mf   = tp - tp.shift(1)
    vcp  = np.where(mf>cut, df["volume"], np.where(mf<-cut, -df["volume"], 0.0))
    vf   = pd.Series(vcp, index=df.index).clip(lower=-vmax, upper=vmax)
    return (vf.rolling(p).sum()/vave.replace(0,np.nan)).fillna(0)

def _stoch_rsi(c: pd.Series, p: int = 14) -> pd.Series:
    r = _rsi(c, p)
    return ((r - r.rolling(p).min()) /
            (r.rolling(p).max() - r.rolling(p).min()).replace(0, np.nan))

def _volume_ratio(v: pd.Series, p: int = 20) -> pd.Series:
    return v / v.rolling(p).mean().replace(0, np.nan)


def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Indicadores base (V4) + extensiones V5."""
    df = df.copy()
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]

    df["ema9"]   = _ema(c, 9);  df["ema21"] = _ema(c, 21)
    df["ema55"]  = _ema(c, 55); df["ema200"]= _ema(c, 200)
    df["bb_lower"], df["bb_mid"], df["bb_upper"] = _bollinger(c)
    df["bb_width"]   = (df["bb_upper"]-df["bb_lower"])/df["bb_mid"].replace(0,np.nan)
    df["bb_squeeze"] = df["bb_width"] < df["bb_width"].rolling(50).quantile(0.20)
    df["rsi"]        = _rsi(c, 14); df["rsi_fast"] = _rsi(c, 7)
    df["stoch_rsi"]  = _stoch_rsi(c)
    df["macd"], df["macd_signal"], df["macd_hist"] = _macd(c)
    df["macd_bull"]     = df["macd"] > df["macd_signal"]
    df["macd_growing"]  = df["macd_hist"] > df["macd_hist"].shift(1)
    df["macd_cross_up"] = (df["macd"]>df["macd_signal"])&(df["macd"].shift(1)<=df["macd_signal"].shift(1))
    df["atr"]          = _atr(df); df["atr_pct"] = df["atr"]/c*100
    df["atr_pct_rank"] = df["atr_pct"].rolling(100).rank(pct=True)
    df["adx"]          = _adx(df)
    df["vol_ratio"]    = _volume_ratio(v); df["vol_spike"] = df["vol_ratio"] > 2.0
    df["vfi"]          = _vfi(df);         df["vfi_bull"]  = df["vfi"] > 0
    df["vwap"]         = _vwap(df)
    bearish_c = c < df["open"]; bull_m = c > c.shift(1)
    df["ob_bull"]    = (bearish_c.shift(1) & bull_m).fillna(False)
    df["ob_bull_hi"] = np.where(df["ob_bull"], h.shift(1), np.nan)
    df["ob_bull_lo"] = np.where(df["ob_bull"], l.shift(1), np.nan)
    df["fvg_bull"]   = (l > h.shift(2)).fillna(False)
    df["fvg_bull_hi"]= np.where(df["fvg_bull"], l, np.nan)
    df["fvg_bull_lo"]= np.where(df["fvg_bull"], h.shift(2), np.nan)
    w = 5
    df["swing_hi"] = (h==h.rolling(w*2+1,center=True).max()).astype(int)
    df["swing_lo"] = (l==l.rolling(w*2+1,center=True).min()).astype(int)
    df["trend_up"]   = (df["ema21"]>df["ema55"])&(df["ema55"]>df["ema200"])
    df["trend_down"] = (df["ema21"]<df["ema55"])&(df["ema55"]<df["ema200"])
    df["above_vwap"] = c > df["vwap"]
    df["momentum_3"] = c.pct_change(3)*100; df["momentum_10"] = c.pct_change(10)*100
    score = (df["trend_up"].fillna(False).astype(float)*25 +
             ((df["rsi"].clip(30,70)-30)/40.0)*20 +
             df["macd_bull"].fillna(False).astype(float)*20 +
             (df["adx"].clip(0,50)/50.0)*20 +
             df["vfi_bull"].fillna(False).astype(float)*15)
    df["consensus"] = score.fillna(0)

    # ── Extensiones V5 ────────────────────────────────────────────────────────
    if V5_INDICATORS:
        try:
            df = enrich_v5(df)
        except Exception as exc:
            log.debug("enrich_v5_skip", error=str(exc))

    return df


def detect_regime(df: pd.DataFrame, symbol: str) -> MarketRegime:
    if len(df) < 50:
        return MarketRegime(symbol, "RANGE", 0.4, 20.0, 2.0, False, 0.0)
    last = df.iloc[-1]
    adx      = float(last.get("adx", 20))
    atr_pct  = float(last.get("atr_pct", 2.0))
    atr_rank = float(last.get("atr_pct_rank", 0.5))
    trend_up = bool(last.get("trend_up", False))
    trend_dn = bool(last.get("trend_down", False))
    vfi      = float(last.get("vfi", 0.0))
    if atr_rank > 0.85:
        return MarketRegime(symbol, "HIGH_VOL", 0.9, adx, atr_pct, trend_up, vfi)
    if trend_up and adx > 22 and vfi > 0:
        return MarketRegime(symbol, "BULL", min(1.0,(adx-22)/28+0.4), adx, atr_pct, True, vfi)
    if trend_dn and adx > 20:
        return MarketRegime(symbol, "BEAR", min(1.0,(adx-20)/30+0.3), adx, atr_pct, False, vfi)
    return MarketRegime(symbol, "RANGE", max(0.3,1.0-adx/40), adx, atr_pct, trend_up, vfi)


# ══════════════════════════════════════════════════════════════════════════════
# ── Contexto V5 (sin lookahead bias) ─────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def build_v5_context(df: pd.DataFrame, direction: str) -> V5Context:
    """
    Construye el contexto V5 para la señal actual.
    CRÍTICO: usa solo df.iloc[-lookback:] → sin lookahead bias.
    Los swing points tienen delay de confirmación de 5 velas.
    """
    ctx = V5Context()
    if not V5_INDICATORS or len(df) < 50:
        return ctx

    last = df.iloc[-1]

    # ── Patrones de velas (ya calculados en enrich_v5) ────────────────────
    ctx.candle_bull_score = float(last.get("cp_bull_score", 0))
    ctx.candle_bear_score = float(last.get("cp_bear_score", 0))

    # ── Order Flow ────────────────────────────────────────────────────────
    ctx.obv_rising  = bool(last.get("obv_rising", False))
    ctx.cvd_rising  = bool(last.get("cvd_rising", False))
    ctx.absorption  = bool(last.get("absorption", 0))

    # OBV acelerando: OBV actual > media OBV de las últimas 10 velas
    # (no solo subiendo, sino subiendo más rápido que su tendencia reciente)
    if "obv" in df.columns and len(df) >= 12:
        obv_series  = df["obv"].iloc[-12:]
        obv_now     = float(obv_series.iloc[-1])
        obv_avg10   = float(obv_series.iloc[-11:-1].mean())
        obv_slope   = float(obv_series.diff().iloc[-3:].mean())
        ctx.obv_accelerating = obv_slope > 0 and obv_now > obv_avg10 * 1.005

    # CVD positivo neto: delta de volumen promedio de últimas 5 velas > umbral
    if "delta_vol" in df.columns and len(df) >= 6:
        delta_mean = float(df["delta_vol"].iloc[-5:].mean())
        ctx.cvd_positive = delta_mean > 0.15  # >15% sesgo comprador neto

    # ── Estructura, S/R, Fibonacci (con slice estricto) ───────────────────
    try:
        struct = get_structure_context(df, lookback=min(150, len(df)))
        ctx.structure_bias  = struct["structure"]["bias"]
        ctx.structure_score = float(struct["structure"]["score"])
        ctx.sr_at_zone      = struct["at_zone"]
        ctx.sr_zones        = struct["zones"]
        ctx.swing_highs     = struct["swing_highs"]
        ctx.swing_lows      = struct["swing_lows"]
        ctx.chart_patterns  = struct["chart_patterns"]

        # Score de zona S/R
        if ctx.sr_at_zone and struct["current_zone"]:
            sr = struct["sr_helper"]
            ctx.sr_zone_score = sr.score_for_zone(
                struct["current_zone"], direction
            )

        # Fibonacci: TP por extensiones
        price = struct["price"]
        stop_approx = price * 0.985  # placeholder — se ajusta en estrategia
        fib = struct["fib_helper"]
        tp1, tp2, tp3 = fib.tp_levels(price, stop_approx, direction)
        ctx.fib_tp1, ctx.fib_tp2, ctx.fib_tp3 = tp1, tp2, tp3

        # ¿Precio cerca de nivel Fibonacci?
        fib_r = struct["fib_retrace"]
        if fib_r:
            near = fib.nearest_level(price, fib_r, tol=0.008)
            if near:
                ctx.fib_near  = True
                ctx.fib_level = near[0]

        # Score de figuras chartistas
        for pat in ctx.chart_patterns:
            if pat.get("direction") in ("bull", "neutral"):
                ctx.chart_bull_score += pat.get("score", 0)
            if pat.get("direction") in ("bear", "neutral"):
                ctx.chart_bear_score += pat.get("score", 0)

    except Exception as exc:
        log.debug("v5_context_error", error=str(exc))

    return ctx


# ══════════════════════════════════════════════════════════════════════════════
# ── 4 Estrategias + V5 scoring ────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _quality(score: float) -> str:
    if score >= 90: return "A+"
    if score >= 75: return "A"
    if score >= 60: return "B"
    return "C"


def _fib_levels(entry: float, stop: float, direction: str,
                ctx: V5Context) -> Tuple[float, float, float]:
    """Usa Fibonacci si está disponible, sino múltiplos de R."""
    dist = abs(entry - stop)
    if ctx and ctx.fib_tp1 > 0 and direction == "long":
        # Re-calcular con el SL real
        fib = Fibonacci()
        tp1, tp2, tp3 = fib.tp_levels(entry, stop, direction)
        return tp1, tp2, tp3
    if direction == "long":
        return entry + 2.0*dist, entry + 3.5*dist, entry + 5.0*dist
    return entry - 2.0*dist, entry - 3.5*dist, entry - 5.0*dist


def _sl_adjusted(base_sl: float, direction: str,
                 ctx: V5Context) -> float:
    """Ajusta el SL a la zona S/R más cercana si hay una."""
    if not ctx or not ctx.sr_zones:
        return base_sl
    for z in ctx.sr_zones[:3]:
        if direction == "long" and z["type"] == "support":
            candidate = z["bottom"] * 0.997
            if base_sl < candidate < base_sl * 1.015:
                return candidate  # SL en zona S/R, más preciso
        elif direction == "short" and z["type"] == "resistance":
            candidate = z["top"] * 1.003
            if base_sl * 0.985 < candidate < base_sl:
                return candidate
    return base_sl


def strategy_trend_following_v5(
    df: pd.DataFrame, regime: MarketRegime
) -> Optional[SignalResult]:
    if len(df) < 210:
        return None
    last  = df.iloc[-1]
    c     = float(last["close"])
    score = 0.0
    reasons: List[str] = []

    if bool(last.get("trend_up", False)):    score += 25; reasons.append("EMA_stack_bull")
    adx = float(last.get("adx", 0))
    if adx > 25:   score += 25; reasons.append(f"ADX={adx:.0f}>25")
    elif adx > 20: score += 15; reasons.append(f"ADX={adx:.0f}>20")
    rsi = float(last.get("rsi", 50))
    if 40 <= rsi <= 70: score += 15; reasons.append(f"RSI={rsi:.0f}")
    if bool(last.get("macd_bull", False)) and bool(last.get("macd_growing", False)):
        score += 15; reasons.append("MACD_bull+growing")
    elif bool(last.get("macd_bull", False)):
        score += 8;  reasons.append("MACD_bull")
    ema21     = float(last.get("ema21", c))
    dist_ema  = abs(c - ema21) / ema21 * 100
    if dist_ema < 1.5: score += 10; reasons.append("pullback_ema21_tight")
    elif dist_ema < 3: score += 6;  reasons.append("pullback_ema21")
    if df["ob_bull"].iloc[-5:].any() if "ob_bull" in df.columns else False:
        score += 10; reasons.append("OB_bull")
    if df["fvg_bull"].iloc[-3:].any() if "fvg_bull" in df.columns else False:
        score += 8; reasons.append("FVG_bull")
    if float(last.get("vol_ratio", 1)) > 1.2: score += 5; reasons.append("vol_above_avg")
    if bool(last.get("vfi_bull", False)):      score += 5; reasons.append("VFI_bull")

    # ── Capas V5 ──────────────────────────────────────────────────────────
    ctx = build_v5_context(df, "long")
    v5_bonus = ctx.total_bull_bonus
    if v5_bonus > 0:
        score += min(v5_bonus, 35)  # cap para no inflar demasiado
        if ctx.candle_bull_score > 0:
            reasons.append(f"CANDLE_bull+{ctx.candle_bull_score:.0f}")
        if ctx.structure_bias == "bull":
            reasons.append(f"BOS/CHoCH_bull+{ctx.structure_score:.0f}")
        if ctx.sr_at_zone:
            reasons.append(f"SR_zone+{ctx.sr_zone_score:.0f}")
        if ctx.fib_near:
            reasons.append(f"FIB_{ctx.fib_level}")
        for pat in ctx.chart_patterns:
            if pat.get("direction") == "bull":
                reasons.append(f"{pat['name']}+{pat['score']}")

    if score < MIN_SCORE_TREND:
        return None

    # Confirmación V5 — requiere al menos 1 señal FUERTE o 2 moderadas.
    # Condiciones calibradas para ser True el 20-35% del tiempo:
    #   candle >= 20: solo patrones fuertes (Engulfing, 3 Soldiers, Morning Star)
    #   structure = bull: BOS/CHoCH real detectado
    #   sr_at_zone: precio exactamente en zona S/R conocida
    #   obv_accelerating: OBV acelerando (no solo subiendo)
    if REQUIRE_V5_CONFIRMATION:
        # Señales fuertes (cada una por sí sola es suficiente)
        strong_signal = (
            ctx.candle_bull_score >= 20 or   # patrón fuerte confirmado
            ctx.absorption                    # absorción institucional real
        )
        # Señales moderadas (necesitan 2)
        moderate_signals = [
            ctx.candle_bull_score >= 10,      # patrón moderado
            ctx.structure_bias == "bull",     # estructura alcista
            ctx.sr_at_zone,                   # en zona S/R
            ctx.obv_accelerating,             # OBV acelerando
            ctx.cvd_positive,                 # compradores netos > 15%
        ]
        if not strong_signal and sum(moderate_signals) < 2:
            return None

    # Spot no puede hacer short — TrendFollowing solo en mercados alcistas/laterales
    if regime.regime == "BEAR":
        return None   # En BEAR no hay setup long válido para TrendFollowing
    if regime.regime == "HIGH_VOL":
        score *= 0.7  # Alta volatilidad: reducir tamaño pero no bloquear

    atr  = float(last.get("atr", c * 0.015))
    low5 = float(df["low"].iloc[-5:].min())
    sl   = _sl_adjusted(max(c - 1.5*atr, low5 - 0.5*atr, c*0.985), "long", ctx)
    dist = c - sl
    if dist <= 0:
        return None
    tp1, tp2, tp3 = _fib_levels(c, sl, "long", ctx)

    return SignalResult(
        symbol=regime.symbol, strategy="TrendFollowing",
        direction="long", score=score,
        entry_price=c, stop_loss=sl,
        tp1=tp1, tp2=tp2, tp3=tp3,
        atr=atr, regime=regime, quality=_quality(score),
        reasons=reasons, v5_context=ctx,
    )


def strategy_mean_reversion_v5(
    df: pd.DataFrame, regime: MarketRegime
) -> Optional[SignalResult]:
    if len(df) < 50:
        return None
    # En BEAR: solo entrar en sobreventa extrema con alta confluencia
    # Es el único setup long válido en mercado bajista (comprar el suelo)
    if regime.regime == "BEAR":
        # RSI debe estar en territorio de sobreventa extrema
        rsi_check = float(df.iloc[-1].get("rsi", 50))
        if rsi_check > 32:
            return None  # No es sobreventa suficiente para comprar en BEAR
        # Y la fuerza bajista no puede ser máxima
        if regime.strength > 0.85:
            return None  # Tendencia bajista demasiado fuerte — no coger cuchillos

    last  = df.iloc[-1]
    c     = float(last["close"])
    score = 0.0
    reasons: List[str] = []

    adx = float(last.get("adx", 30))
    if adx < 20:   score += 20; reasons.append(f"ADX={adx:.0f}_range")
    elif adx < 25: score += 12; reasons.append(f"ADX={adx:.0f}")
    rsi = float(last.get("rsi", 50))
    if rsi < 28:   score += 30; reasons.append(f"RSI={rsi:.0f}_extreme")
    elif rsi < 35: score += 20; reasons.append(f"RSI={rsi:.0f}_oversold")
    bb_lower = float(last.get("bb_lower", c)); bb_mid = float(last.get("bb_mid", c))
    dist_bb = (c - bb_lower) / bb_lower * 100 if bb_lower > 0 else 999
    if dist_bb < 0.5:  score += 20; reasons.append("AT_BB_lower")
    elif dist_bb < 2:  score += 12; reasons.append("NEAR_BB_lower")
    stoch = float(last.get("stoch_rsi", 0.5))
    if stoch < 0.15:   score += 15; reasons.append(f"StochRSI={stoch:.2f}_extreme")
    elif stoch < 0.25: score += 8;  reasons.append(f"StochRSI={stoch:.2f}")
    if bool(last.get("macd_cross_up", False)): score += 15; reasons.append("MACD_cross_up")
    elif bool(last.get("macd_growing", False)):score += 8;  reasons.append("MACD_hist_turning")
    if bool(last.get("vfi_bull", False)):       score += 10; reasons.append("VFI_bull_at_extreme")

    # ── Capas V5 ──────────────────────────────────────────────────────────
    ctx = build_v5_context(df, "long")
    v5_bonus = ctx.total_bull_bonus
    # Para MR, las velas de reversión son especialmente valiosas
    if ctx.candle_bull_score > 0:
        score += min(ctx.candle_bull_score * 1.2, 25)
        reasons.append(f"CANDLE_reversal+{ctx.candle_bull_score:.0f}")
    if ctx.sr_at_zone and ctx.sr_zone_score > 0:
        score += ctx.sr_zone_score
        reasons.append(f"SR_support+{ctx.sr_zone_score:.0f}")
    if ctx.fib_near:
        score += 10; reasons.append(f"FIB_{ctx.fib_level}_reversal")

    if score < MIN_SCORE_MR:
        return None

    # MR: requiere confluencia real — precio en zona + señal de reversión
    # La reversión sin soporte es una apuesta, no un trade
    if REQUIRE_V5_CONFIRMATION:
        # Señal fuerte de reversión (por sí sola suficiente con S/R)
        reversal_candle = ctx.candle_bull_score >= 15  # Engulfing, Hammer, Morning Star
        at_support      = ctx.sr_at_zone and ctx.sr_zone_score >= 5
        at_fib          = ctx.fib_near

        # Necesita (vela de reversión + soporte) O (soporte + Fibonacci)
        if not ((reversal_candle and at_support) or
                (at_support and at_fib) or
                (reversal_candle and at_fib)):
            return None

    atr  = float(last.get("atr", c * 0.015))
    sl   = _sl_adjusted(max(bb_lower - MR_ATR_STOP_MULTIPLIER * atr, c * 0.965), "long", ctx)
    dist = c - sl
    if dist <= 0:
        return None
    ema21 = float(last.get("ema21", c * 1.02))
    tp1   = max(c + 1.2 * dist, ema21)
    tp2   = bb_mid
    tp3   = float(last.get("bb_upper", c + 3 * dist))
    # Refinar con Fibonacci si disponible
    if ctx.fib_tp1 > 0:
        tp1, tp2, tp3 = _fib_levels(c, sl, "long", ctx)

    return SignalResult(
        symbol=regime.symbol, strategy="MeanReversion",
        direction="long", score=score,
        entry_price=c, stop_loss=sl,
        tp1=tp1, tp2=tp2, tp3=tp3,
        atr=atr, regime=regime, quality=_quality(score),
        reasons=reasons, v5_context=ctx,
    )


def strategy_breakout_v5(
    df: pd.DataFrame, regime: MarketRegime
) -> Optional[SignalResult]:
    if len(df) < 55:
        return None

    last  = df.iloc[-1]
    c     = float(last["close"])
    score = 0.0
    reasons: List[str] = []

    sq_col = df.get("bb_squeeze", pd.Series(False, index=df.index)).fillna(False)
    sq_n   = int(sq_col.iloc[-20:].sum()) if isinstance(sq_col, pd.Series) else 0
    if sq_n >= 10: score += 25; reasons.append(f"BB_squeeze_{sq_n}v")
    elif sq_n >= 5:score += 12; reasons.append(f"BB_squeeze_{sq_n}v")
    bb_upper = float(last.get("bb_upper", c))
    if c > bb_upper:          score += 20; reasons.append("CLOSE_above_BB")
    vol_ratio = float(last.get("vol_ratio", 1))
    if vol_ratio > 2.0:       score += 20; reasons.append(f"VOL_spike_{vol_ratio:.1f}x")
    elif vol_ratio > 1.5:     score += 12; reasons.append(f"VOL_elev_{vol_ratio:.1f}x")
    high20 = float(df["high"].iloc[-21:-1].max()) if len(df) > 21 else c
    if c > high20:            score += 15; reasons.append("BREAK_20period_high")
    if bool(last.get("macd_bull", False)): score += 10; reasons.append("MACD_confirm")

    # ── Capas V5: absorción + figura de compresión ────────────────────────
    ctx = build_v5_context(df, "long")
    if ctx.absorption:   score += 12; reasons.append("ABSORPTION_detected")
    if ctx.obv_rising:   score += 8;  reasons.append("OBV_rising")
    if ctx.cvd_rising:   score += 8;  reasons.append("CVD_rising")
    for pat in ctx.chart_patterns:
        if "TRIANGLE" in pat.get("name", "") or "FLAG" in pat.get("name", ""):
            score += pat.get("score", 0)
            reasons.append(f"{pat['name']}+{pat['score']}")

    if score < MIN_SCORE_BREAKOUT:
        return None

    # Breakout REQUIERE volumen real — sin volumen es falsa ruptura
    if REQUIRE_V5_CONFIRMATION:
        if not (ctx.absorption or ctx.obv_accelerating):
            return None

    atr  = float(last.get("atr", c * 0.015))
    prev_upper = float(df["bb_upper"].iloc[-2]) if len(df) > 1 else bb_upper
    sl   = _sl_adjusted(max(prev_upper - 1.5*atr, c*0.975), "long", ctx)
    dist = c - sl
    if dist <= 0:
        return None
    tp1, tp2, tp3 = _fib_levels(c, sl, "long", ctx)

    return SignalResult(
        symbol=regime.symbol, strategy="Breakout",
        direction="long", score=score,
        entry_price=c, stop_loss=sl,
        tp1=tp1, tp2=tp2, tp3=tp3,
        atr=atr, regime=regime, quality=_quality(score),
        reasons=reasons, v5_context=ctx,
    )


def strategy_momentum_scalp_v5(
    df: pd.DataFrame, regime: MarketRegime
) -> Optional[SignalResult]:
    # Spot long-only: MomentumScalp tampoco en mercado bajista
    if regime.regime == "BEAR":
        return None
    if len(df) < 30:
        return None
    if regime.atr_pct > 5.0:
        return None

    last  = df.iloc[-1]
    prev  = df.iloc[-2] if len(df) > 1 else last
    c     = float(last["close"])
    score = 0.0
    reasons: List[str] = []

    rsi_n = float(last.get("rsi", 50)); rsi_p = float(prev.get("rsi", 50))
    rsi_f = float(last.get("rsi_fast", 50))
    if rsi_p < 52 and rsi_n > 58 and rsi_f > 60:
        score += 25; reasons.append(f"RSI_accel_{rsi_p:.0f}→{rsi_n:.0f}")
    elif rsi_n > 55 and rsi_n > rsi_p + 3:
        score += 15; reasons.append(f"RSI_momentum_{rsi_n:.0f}")
    vwap = float(last.get("vwap", c))
    if c > vwap * 1.001: score += 20; reasons.append("ABOVE_VWAP")
    e9n = float(last.get("ema9", c)); e21n = float(last.get("ema21", c))
    e9p = float(prev.get("ema9", c)); e21p = float(prev.get("ema21", c))
    if e9p <= e21p and e9n > e21n: score += 20; reasons.append("EMA9_cross_EMA21")
    elif e9n > e21n:               score += 10; reasons.append("EMA9_above_EMA21")
    vol3 = df["volume"].iloc[-3:]
    if len(vol3) == 3 and vol3.is_monotonic_increasing:
        score += 15; reasons.append("VOL_3bar_increasing")
    elif float(last.get("vol_ratio", 1)) > 1.3:
        score += 8;  reasons.append("VOL_above_avg")
    mh = float(last.get("macd_hist", 0))
    if mh > 0 and bool(last.get("macd_growing", False)):
        score += 20; reasons.append("MACD_hist_bull_growing")
    elif mh > 0:
        score += 10; reasons.append("MACD_hist_positive")

    # ── V5: velas de continuación ────────────────────────────────────────
    ctx = build_v5_context(df, "long")
    if ctx.candle_bull_score > 0:
        score += min(ctx.candle_bull_score * 0.8, 15)
        reasons.append(f"CANDLE_momentum+{ctx.candle_bull_score:.0f}")

    if score < MIN_SCORE_MOMENTUM:
        return None

    # Scalp: OBV acelerando Y (CVD positivo O patrón de continuación)
    if REQUIRE_V5_CONFIRMATION:
        if not ctx.obv_accelerating:
            return None  # Sin momentum real de OBV, no es scalp
        if not (ctx.cvd_positive or ctx.candle_bull_score >= 10):
            return None  # Necesita confirmación adicional

    atr  = float(last.get("atr", c * 0.015))
    sl   = _sl_adjusted(max(c - 1.0*atr, c*0.988), "long", ctx)
    dist = c - sl
    if dist <= 0:
        return None
    tp1, tp2, tp3 = _fib_levels(c, sl, "long", ctx)

    return SignalResult(
        symbol=regime.symbol, strategy="MomentumScalp",
        direction="long", score=score,
        entry_price=c, stop_loss=sl,
        tp1=tp1, tp2=tp2, tp3=tp3,
        atr=atr, regime=regime, quality=_quality(score),
        reasons=reasons, v5_context=ctx,
    )


def select_best_signal(signals: List[Optional[SignalResult]]) -> Optional[SignalResult]:
    valid = [s for s in signals if s is not None and s.score >= 55]
    if not valid:
        return None
    regime = valid[0].regime.regime
    PRIORITY = {
        "BULL":     {"TrendFollowing":4,"MomentumScalp":3,"Breakout":2,"MeanReversion":1},
        "RANGE":    {"MeanReversion":4,"MomentumScalp":3,"Breakout":2,"TrendFollowing":1},
        "BEAR":     {"MeanReversion":3,"TrendFollowing":1,"Breakout":1,"MomentumScalp":1},
        "HIGH_VOL": {"TrendFollowing":2,"MeanReversion":2,"Breakout":1,"MomentumScalp":1},
    }
    if regime == "BEAR":
        # Spot long-only: en BEAR solo MeanReversion (comprar en sobreventa extrema)
        # con score muy alto — el resto de estrategias ya retornaron None
        valid = [s for s in valid if s.strategy == "MeanReversion" and s.score >= 72]
    if regime == "HIGH_VOL": valid = [s for s in valid if s.score >= 70]
    if not valid:
        return None
    pm = PRIORITY.get(regime, PRIORITY["RANGE"])
    return max(valid, key=lambda s: pm.get(s.strategy, 1) * 10 + s.score)


# ══════════════════════════════════════════════════════════════════════════════
# ── Sizing (idéntico al V4) ───────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def compute_position_size(
    signal: SignalResult, capital: float,
    open_count: int, daily_pnl_pct: float,
) -> Tuple[float, float, float]:
    if capital < 1500:      risk_usd = RISK_PER_TRADE_USD
    elif capital < 5000:    risk_usd = capital * RISK_PCT_TIER_1
    elif capital < 20_000:  risk_usd = capital * RISK_PCT_TIER_2
    else:                   risk_usd = capital * RISK_PCT_TIER_3
    risk_usd = min(risk_usd, capital * MAX_RISK_PER_TRADE)
    risk_usd *= signal.regime.risk_multiplier
    if daily_pnl_pct < CB_DAILY_REDUCE:
        risk_usd *= 0.5
    quality_mult = {"A+": 1.0, "A": 0.8, "B": 0.6, "C": 0.4}
    risk_usd *= quality_mult.get(signal.quality, 0.6)
    wr = 0.50 + signal.ml_proba * 0.10
    b  = (signal.tp1 - signal.entry_price) / (signal.entry_price - signal.stop_loss) if (signal.entry_price - signal.stop_loss) > 0 else 1
    q  = 1 - wr
    kf = max(0, (wr * b - q) / b) * KELLY_FRACTION
    if kf > 0:
        risk_usd = min(risk_usd, capital * kf)
    risk_usd = max(risk_usd, 3.0)
    dist_sl  = signal.entry_price - signal.stop_loss
    if dist_sl <= 0:
        return 0.0, 0.0, 0.0
    units       = risk_usd / dist_sl
    max_notional= capital * 0.35
    if units * signal.entry_price > max_notional:
        units   = max_notional / signal.entry_price
    notional = units * signal.entry_price
    if notional < 11.0:
        return 0.0, 0.0, 0.0
    return units, risk_usd, notional


# ══════════════════════════════════════════════════════════════════════════════
# ── Motor Principal V5 ───────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

class LiveEngineV5:
    """Motor V5 — hereda toda la infraestructura del V4 + capas V5."""

    def __init__(self, paper_mode: bool = True) -> None:
        self.paper_mode = paper_mode
        self._running   = False
        self._tasks: List[asyncio.Task] = []
        self.state: Dict[str, Any] = {
            "paused": False, "kill": False, "paper_mode": paper_mode,
            "capital": INITIAL_CAPITAL, "open_positions": [],
            "pnl_today": 0.0, "pnl_today_pct": 0.0, "drawdown_pct": 0.0,
            "ml_ready": False, "regimes": {}, "last_signals": {},
            "v5_enabled": V5_INDICATORS,
        }
        self._cooldowns: Dict[str, SymbolCooldown] = {s: SymbolCooldown() for s in SYMBOLS}
        self._df_cache:  Dict[str, pd.DataFrame]   = {}
        self._pending_signals: Dict[str, SignalResult] = {}
        self._recent_errors: List[float] = []
        self._exchange:  Optional[ccxt.Exchange]  = None
        self._portfolio: Optional[PaperPortfolio] = None
        self._bot:       Optional[TelegramBot]    = None
        self._ml:        Optional[MetaLabeler]    = None

    async def start(self) -> None:
        log.info("engine_v5_starting", version=ENGINE_VERSION,
                 paper=self.paper_mode, v5_indicators=V5_INDICATORS,
                 capital=f"${INITIAL_CAPITAL:.0f}", symbols=SYMBOLS)
        self._running = True
        self._exchange = ccxt.binance({
            "apiKey": os.environ.get("BINANCE_API_KEY",""),
            "secret": os.environ.get("BINANCE_API_SECRET",""),
            "options": {"defaultType":"spot"}, "enableRateLimit": True,
        })
        await self._ping_exchange()
        if not self.paper_mode:
            if not await self._validate_api_keys():
                raise RuntimeError("API keys inválidas.")
        raw_db = os.environ.get("DATABASE_URL","")
        if not raw_db:
            raise RuntimeError("DATABASE_URL no configurada.")
        async_url = (raw_db.replace("postgresql://","postgresql+asyncpg://")
                          .replace("postgres://","postgresql+asyncpg://"))
        self._portfolio = PaperPortfolio(initial_capital=INITIAL_CAPITAL, db_url=async_url)
        await self._portfolio.initialize()
        await self._sync_state()
        self._ml = MetaLabeler(model_path=str(PROJECT_ROOT/"ml"/"model.joblib"))
        self.state["ml_ready"] = self._ml.is_ready()
        token   = os.environ.get("TELEGRAM_TOKEN","")
        chat_id = int(os.environ.get("TELEGRAM_ALLOWED_USER_ID","0") or "0")
        if token and chat_id:
            self._bot = TelegramBot(token=token, allowed_chat_id=chat_id)
            await self._bot.start()
            await self._bot.send_startup(
                paper_mode=self.paper_mode,
                capital=self.state["capital"],
                n_positions=len(self.state["open_positions"]),
            )
        asyncio.get_event_loop().add_signal_handler(
            signal.SIGTERM, lambda: asyncio.create_task(self.shutdown("SIGTERM"))
        )
        log.info("engine_v5_live", mode="paper" if self.paper_mode else "LIVE",
                 v5=V5_INDICATORS, strategies=4)
        t1 = asyncio.create_task(self._loop_slow(),  name="loop_slow")
        t2 = asyncio.create_task(self._loop_entry(), name="loop_entry")
        t3 = asyncio.create_task(self._loop_fast(),  name="loop_fast")
        self._tasks = [t1, t2, t3]
        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            log.exception("engine_v5_fatal", error=str(exc))
            if self._bot:
                await self._bot.send_alert(f"💀 Error fatal V5:\n```{exc}```","critical")
            raise

    async def shutdown(self, reason: str = "manual") -> None:
        log.warning("shutdown_v5", reason=reason)
        self._running = False
        for task in self._tasks:
            task.cancel()
        daily       = await self._portfolio.get_daily_stats() if self._portfolio else {}
        session_pnl = daily.get("pnl_today", 0.0)
        await self._save_state()
        if self._exchange:
            await self._exchange.close()
        if self._bot:
            await self._bot.send_shutdown(reason=reason, session_pnl=session_pnl)
            await self._bot.stop()

    async def _loop_slow(self) -> None:
        while self._running:
            wait = _seconds_to_next_close(TF_STRUCT_SECS)
            await asyncio.sleep(wait + CANDLE_BUFFER)
            if self.state["kill"] or self.state["paused"]:
                continue
            for symbol in SYMBOLS:
                try:
                    await self._analyze_symbol_v5(symbol)
                except Exception as exc:
                    log.exception("analysis_error_v5", symbol=symbol, error=str(exc))

    async def _analyze_symbol_v5(self, symbol: str) -> None:
        df = await self._fetch_candles(symbol, TF_STRUCTURE, limit=500)
        if df is None or len(df) < 220:
            return
        try:
            df = enrich_dataframe(df)   # V4 + V5 indicadores
        except Exception as exc:
            log.warning("enrich_error_v5", symbol=symbol, error=str(exc)); return
        regime = detect_regime(df, symbol)
        self.state["regimes"][symbol] = regime
        if not regime.is_tradeable:
            return
        self._df_cache[symbol] = df
        candidates = []
        if regime.regime in ("BULL","RANGE","HIGH_VOL"):
            candidates = [
                strategy_trend_following_v5(df, regime),
                strategy_mean_reversion_v5(df, regime),
                strategy_breakout_v5(df, regime),
                strategy_momentum_scalp_v5(df, regime),
            ]
        elif regime.regime == "BEAR":
            # Spot long-only: en BEAR solo MeanReversion en sobreventa extrema
            # TrendFollowing y MomentumScalp ya retornan None en BEAR
            candidates = [
                strategy_mean_reversion_v5(df, regime),
            ]
        if self.state["ml_ready"] and self._ml:
            for sig in [s for s in candidates if s is not None]:
                try:
                    sig.ml_proba = self._ml.predict_proba(df)
                except Exception:
                    sig.ml_proba = 0.5
        best = select_best_signal(candidates)
        if best:
            log.info("signal_queued_v5", symbol=symbol, strategy=best.strategy,
                     score=f"{best.score:.0f}", quality=best.quality,
                     regime=regime.regime, v5_bonus=f"{best.v5_context.total_bull_bonus:.0f}" if best.v5_context else "0",
                     reasons=",".join(best.reasons[:4]))
            self._pending_signals[symbol] = best
        else:
            self._pending_signals.pop(symbol, None)

    async def _loop_entry(self) -> None:
        while self._running:
            wait = _seconds_to_next_close(TF_ENTRY_SECS)
            await asyncio.sleep(wait + 5)
            if self.state["kill"] or self.state["paused"] or not self._pending_signals:
                continue
            for symbol, signal in list(self._pending_signals.items()):
                try:
                    await self._attempt_entry(symbol, signal)
                except Exception as exc:
                    log.exception("entry_error_v5", symbol=symbol, error=str(exc))

    async def _attempt_entry(self, symbol: str, signal: SignalResult) -> None:
        open_positions = await self._portfolio.get_open_positions()
        open_symbols   = {p["symbol"] for p in open_positions}
        if symbol in open_symbols:
            self._pending_signals.pop(symbol, None); return
        if len(open_positions) >= MAX_POSITIONS:
            return
        # Filtro correlación
        my_group = CORRELATION_GROUPS.get(symbol, "other")
        if my_group != "btc":
            grp_count = sum(1 for p in open_positions
                            if CORRELATION_GROUPS.get(p.get("symbol",""),"other") == my_group)
            if grp_count >= MAX_CORRELATED_POSITIONS:
                return
        # Filtro horario MR
        if "MeanReversion" in signal.strategy:
            if datetime.now(timezone.utc).hour not in MR_ALLOWED_HOURS_UTC:
                return
        # Cooldown
        cooldown = self._cooldowns.get(symbol, SymbolCooldown())
        if time.time() - cooldown.last_sl_time < COOLDOWN_AFTER_SL_MIN * 60:
            return
        # Circuit breakers
        capital  = await self._portfolio.get_current_capital()
        daily    = await self._portfolio.get_daily_stats()
        pnl_pct  = float(daily.get("pnl_today",0)) / capital if capital > 0 else 0
        if pnl_pct <= CB_DAILY_PAUSE:
            return
        ps = await self._get_portfolio_state()
        if ps:
            peak = float(ps.get("peak_capital", capital))
            if peak > 0 and (capital - peak) / peak <= CB_PEAK_SHUTDOWN:
                if self._bot:
                    await self._bot.send_alert("🚨 Drawdown límite — Motor detenido","critical")
                await self.shutdown("peak_drawdown_limit"); return
        # Confirmación 15m
        df_15m = await self._fetch_candles(symbol, TF_ENTRY, limit=100)
        if df_15m is None or len(df_15m) < 30:
            return
        try:
            df_15m = enrich_dataframe(df_15m)
        except Exception:
            return
        last_15m = df_15m.iloc[-1]
        c_15m    = float(last_15m["close"])
        drift    = abs(c_15m - signal.entry_price) / signal.entry_price * 100
        if drift > 1.5:
            self._pending_signals.pop(symbol, None); return
        # Micro-confirmación 1/3 condiciones
        micro = (int(float(last_15m.get("rsi",50)) > 50) +
                 int(bool(last_15m.get("macd_bull",False))) +
                 int(bool(last_15m.get("above_vwap",False))))
        if micro == 0:
            return
        signal.entry_price = c_15m
        units, risk_usd, notional = compute_position_size(signal, capital, len(open_positions), pnl_pct)
        if units <= 0:
            return
        capital_pct = notional / capital * 100 if capital > 0 else 0
        log.info("trade_opening_v5", symbol=symbol, strategy=signal.strategy,
                 score=f"{signal.score:.0f}", quality=signal.quality,
                 entry=f"${c_15m:.4f}", sl=f"${signal.stop_loss:.4f}",
                 risk_usd=f"${risk_usd:.2f}", notional=f"${notional:.2f}",
                 capital_pct=f"{capital_pct:.1f}%", regime=signal.regime.regime,
                 ml=f"{signal.ml_proba:.0%}", v5_bonus=f"{signal.v5_context.total_bull_bonus:.0f}" if signal.v5_context else "0",
                 reasons=",".join(signal.reasons[:5]))
        trade = await self._portfolio.open_position(
            symbol=symbol,
            strategy=f"{signal.strategy}_{signal.regime.regime}",
            entry_price=c_15m, stop_loss=signal.stop_loss,
            tp1=signal.tp1, tp2=signal.tp2,
            units=units, ml_proba=signal.ml_proba,
            direction=signal.direction, regime=signal.regime.regime,
            risk_amount=risk_usd, notional_usd=notional,
        )
        if self._bot:
            await self._bot.send_trade_open(trade)
        await self._log_trade_extended(trade, signal, notional, capital)
        self._pending_signals.pop(symbol, None)
        self.state["last_signals"][symbol] = signal

    async def _loop_fast(self) -> None:
        last_daily = datetime.now(tz=timezone.utc).date()
        while self._running:
            await asyncio.sleep(FAST_LOOP_SECS)
            if self.state["kill"]:
                await self._execute_kill_switch(); break
            try:
                prices = await self._get_current_prices()
            except (ccxt.NetworkError, ccxt.RequestTimeout) as exc:
                await self._handle_network_error(exc, "get_prices"); continue
            closed = await self._portfolio.update_positions(prices)
            for ct in closed:
                reason = ct.get("exit_reason","unknown")
                if self._bot:
                    await self._bot.send_trade_close(ct, reason)
                if "stop" in reason.lower() or "sl" in reason.lower():
                    sym = ct.get("symbol","")
                    if sym in self._cooldowns:
                        self._cooldowns[sym].last_sl_time = time.time()
                        self._cooldowns[sym].sl_count_24h += 1
            capital = await self._portfolio.get_current_capital()
            daily   = await self._portfolio.get_daily_stats()
            pnl_pct = float(daily.get("pnl_today",0)) / capital if capital > 0 else 0
            self.state.update({
                "capital": capital, "pnl_today": float(daily.get("pnl_today",0)),
                "pnl_today_pct": pnl_pct,
                "open_positions": await self._portfolio.get_open_positions(),
            })
            await self._update_heartbeat()
            if self._bot:
                await self._bot.process_updates(self.state)
            now = datetime.now(tz=timezone.utc)
            if now.hour == 23 and now.minute >= 55:
                today = now.date()
                if today != last_daily:
                    last_daily = today
                    if self._bot:
                        await self._bot.send_daily_pnl({
                            **daily, "capital": capital, "drawdown_pct": pnl_pct,
                            "ml_ready": self.state["ml_ready"],
                        })
            await self._reset_daily_trackers()

    # ── Helpers (idénticos al V4) ─────────────────────────────────────────────
    async def _get_portfolio_state(self) -> Optional[Dict]:
        if not self._portfolio or not self._portfolio._engine:
            return None
        try:
            async with self._portfolio._session_factory() as s:
                r = await s.execute(text("SELECT * FROM portfolio_state WHERE id=1"))
                row = r.fetchone()
                return dict(row._mapping) if row else None
        except Exception:
            return None

    async def _sync_state(self) -> None:
        self.state["capital"]        = await self._portfolio.get_current_capital()
        self.state["open_positions"] = await self._portfolio.get_open_positions()

    async def _save_state(self) -> None:
        if not self._portfolio or not self._portfolio._engine:
            return
        try:
            async with self._portfolio._session_factory() as s:
                await s.execute(
                    text("UPDATE portfolio_state SET current_capital=:c,updated_at=NOW() WHERE id=1"),
                    {"c": self.state["capital"]})
                await s.commit()
        except Exception:
            pass

    async def _update_heartbeat(self) -> None:
        if not self._portfolio:
            return
        try:
            async with self._portfolio._session_factory() as s:
                await s.execute(text("""
                    INSERT INTO system_heartbeat(id,last_ping,engine_version,paper_mode)
                    VALUES(1,NOW(),:v,:p)
                    ON CONFLICT(id) DO UPDATE SET last_ping=NOW()
                """), {"v": ENGINE_VERSION, "p": self.paper_mode})
                await s.commit()
        except Exception:
            pass

    async def _log_trade_extended(self, trade: Dict, signal: SignalResult,
                                   notional: float, capital: float) -> None:
        if not self._portfolio or not self._portfolio._engine:
            return
        v5_obs = ""
        if signal.v5_context:
            ctx = signal.v5_context
            v5_obs = (f"v5 candle_bull={ctx.candle_bull_score:.0f} "
                      f"struct_bias={ctx.structure_bias} "
                      f"sr_zone={ctx.sr_at_zone} "
                      f"fib={ctx.fib_level} "
                      f"chart_pats={len(ctx.chart_patterns)}")
        try:
            async with self._portfolio._session_factory() as s:
                await s.execute(text("""
                    INSERT INTO trades_journal
                        (trade_id,strategy,symbol,timeframe,direction,
                         setup_quality,entry_price,stop_loss,
                         tp1,tp2,take_profit_1,units,position_size,
                         risk_amount,entry_reason,market_regime,regime,
                         ml_proba,entry_time,is_backtest,observations)
                    VALUES
                        (:tid,:strat,:sym,:tf,:dir,
                         :qual,:ep,:sl,
                         :tp1,:tp2,:tp1,:size,:size,
                         :risk,:reason,:regime,:regime,
                         :ml,NOW(),FALSE,:obs)
                    ON CONFLICT DO NOTHING
                """), {
                    "tid":    trade.get("id",0),   "strat": signal.strategy,
                    "sym":    signal.symbol,        "tf":    TF_ENTRY,
                    "dir":    signal.direction,
                    "qual":   ord(signal.quality[0])-64 if signal.quality else 0,
                    "ep":     signal.entry_price,   "sl": signal.stop_loss,
                    "tp1":    signal.tp1,            "tp2": signal.tp2,
                    "size":   trade.get("units",0),
                    "risk":   trade.get("risk_amount",0),
                    "reason": " | ".join(signal.reasons[:5]),
                    "regime": signal.regime.regime, "ml": signal.ml_proba,
                    "obs":    (f"notional=${notional:.2f} cap%={notional/capital*100:.1f}% "
                               f"score={signal.score:.0f} quality={signal.quality} {v5_obs}"),
                })
                await s.commit()
        except Exception as exc:
            log.debug("journal_v5_skip", error=str(exc))

    async def _execute_kill_switch(self) -> None:
        log.error("kill_switch_v5")
        try:
            prices = await self._get_current_prices()
        except Exception:
            prices = {}
        closed = await self._portfolio.emergency_close_all(prices)
        for ct in closed:
            if self._bot:
                await self._bot.send_trade_close(ct, "kill_switch")
        await self.shutdown("kill_switch")

    async def _reset_daily_trackers(self) -> None:
        now = datetime.now(tz=timezone.utc)
        if now.hour == 0 and now.minute < 2 and self._portfolio:
            capital = await self._portfolio.get_current_capital()
            try:
                async with self._portfolio._session_factory() as s:
                    await s.execute(
                        text("UPDATE portfolio_state SET daily_start=:c WHERE id=1"),
                        {"c": capital})
                    await s.commit()
            except Exception:
                pass
            for cd in self._cooldowns.values():
                cd.sl_count_24h = 0; cd.last_sl_reset = time.time()

    async def _ping_exchange(self) -> None:
        for attempt in range(1, MAX_RETRIES+1):
            try:
                await self._exchange.fetch_time(); return
            except ccxt.NetworkError:
                await asyncio.sleep(min(2**attempt, 60))
        raise RuntimeError("No se pudo conectar a Binance.")

    async def _validate_api_keys(self) -> bool:
        try:
            perms = await self._exchange.fetch_api_key_permissions()
            if perms.get("enableWithdrawals", False):
                log.error("api_keys_have_withdrawal_DANGER"); return False
            return perms.get("enableSpotAndMarginTrading", False)
        except Exception:
            return False

    async def _fetch_candles(self, symbol: str, tf: str, limit: int = 500) -> Optional[pd.DataFrame]:
        for attempt in range(1, MAX_RETRIES+1):
            try:
                ohlcv = await self._exchange.fetch_ohlcv(symbol, tf, limit=limit)
                df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                return df.dropna()
            except ccxt.NetworkError:
                await asyncio.sleep(min(2**attempt, 60))
            except Exception as exc:
                log.warning("fetch_candles_error_v5", symbol=symbol, tf=tf, e=str(exc))
                return None
        return None

    async def _get_current_prices(self) -> Dict[str, float]:
        prices = {}
        for sym in SYMBOLS:
            try:
                t = await self._exchange.fetch_ticker(sym)
                prices[sym] = float(t["last"])
            except Exception:
                if sym in self._df_cache and len(self._df_cache[sym]) > 0:
                    prices[sym] = float(self._df_cache[sym]["close"].iloc[-1])
        return prices

    async def _handle_network_error(self, exc: Exception, ctx: str) -> None:
        now = time.time()
        self._recent_errors = [t for t in self._recent_errors if now-t < CB_ERR_WINDOW]
        self._recent_errors.append(now)
        log.warning("network_error_v5", context=ctx, count=len(self._recent_errors))
        if len(self._recent_errors) >= CB_ERR_LIMIT:
            self._recent_errors.clear()
            if self._bot:
                await self._bot.send_alert("⚠️ Errores de red — pausa 5min","warn")
            await asyncio.sleep(300)


def _seconds_to_next_close(tf_secs: int) -> float:
    epoch = datetime.now(tz=timezone.utc).timestamp()
    return max(math.ceil(epoch / tf_secs) * tf_secs - epoch, 1.0)


async def _main() -> None:
    paper = os.environ.get("PAPER_MODE","true").lower() not in ("false","0","no")
    engine = LiveEngineV5(paper_mode=paper)
    try:
        await engine.start()
    except KeyboardInterrupt:
        await engine.shutdown("KeyboardInterrupt")
    except Exception:
        await engine.shutdown("fatal_error"); sys.exit(1)


if __name__ == "__main__":
    asyncio.run(_main())
