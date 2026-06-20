"""
live_engine_v6.py — Motor Cuantitativo V6.0 «Adaptive Cycle»
=============================================================
Arquitectura de tres horizontes temporales simultáneos:

  HORIZONTE MACRO   (días)   — CycleDetector: ¿en qué fase del ciclo de 4 años?
  HORIZONTE MEDIO   (horas)  — RegimeDetector: ¿qué está haciendo el mercado hoy?
  HORIZONTE MICRO   (velas)  — Estrategias V5: ¿hay un punto de entrada ahora?

La clave de la flexibilidad:
  · En BULL_MATURE → TrendFollowing agresivo, sizing 100%
  · En ACCUMULATION → Solo MeanReversion en soporte + Breakout
  · En BEAR_DEEP → Solo MeanReversion extrema, sizing 25%, capital preservado
  · En DISTRIBUTION → Reducir todo, preparar salida

Esto no es un sistema rígido con reglas fijas. Es un sistema que
reconoce en qué parte del mercado estamos y actúa en consecuencia.
"""
from __future__ import annotations

import asyncio
import math
import os
import signal
import sys
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
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

# ── Importar capas ─────────────────────────────────────────────────────────────
try:
    from indicators import (
        enrich_v5, get_structure_context,
        CandlePatterns, Fibonacci, SRZones,
    )
    V5_OK = True
except ImportError:
    V5_OK = False

try:
    from risk import CycleDetector, CycleState, load_daily_ohlcv
    CYCLE_OK = True
except ImportError:
    CYCLE_OK = False

from ml.meta_labeler import MetaLabeler
from monitoring.telegram_bot import TelegramBot
from paper_portfolio import PaperPortfolio

log = structlog.get_logger(__name__)

ENGINE_VERSION = "6.0.0-AdaptiveCycle"

# ── Pares y paths ──────────────────────────────────────────────────────────────
SYMBOLS: List[str] = [
    "BTC/USDC", "ETH/USDC", "SOL/USDC",
    "BNB/USDC", "LINK/USDC", "AVAX/USDC",
]
SQLITE_PATH = str(PROJECT_ROOT / "data" / "db" / "trading.db")

TF_MACRO      = "1d"
TF_STRUCTURE  = "1h"
TF_ENTRY      = "15m"
TF_STRUCT_S   = 3600
TF_ENTRY_S    = 900

# ── Riesgo base ────────────────────────────────────────────────────────────────
INITIAL_CAPITAL    = float(os.environ.get("INITIAL_CAPITAL", "1000"))
RISK_PER_TRADE_USD = 10.0
MAX_RISK_PCT       = 0.020
KELLY_FRACTION     = 0.25
MAX_POSITIONS      = 3
COMMISSION_RATE    = 0.001
SLIPPAGE           = 0.001
MARGIN_ENABLED     = os.environ.get("MARGIN_ENABLED", "false").lower() == "true"

# ── Filtros ────────────────────────────────────────────────────────────────────
CORRELATION_GROUPS: Dict[str, str] = {
    "BTC/USDC": "btc",
    "ETH/USDC": "altcoin", "SOL/USDC": "altcoin",
    "BNB/USDC": "altcoin", "LINK/USDC": "altcoin", "AVAX/USDC": "altcoin",
}
# Pares de alta prioridad: únicos activos en BEAR_DEEP (reduce CPU 66%)
SYMBOLS_BEAR_DEEP: List[str] = ["BTC/USDC", "ETH/USDC"]
MAX_CORR_POS          = 1
MR_HOURS_BEAR         = range(0, 24)    # MR a cualquier hora (corregido)
MR_HOURS_ACCUM        = range(0, 16)    # MR más amplio en acumulación
MR_HOURS_BULL         = range(0, 24)    # En bull market MR a cualquier hora
COOLDOWN_SL_MIN       = 90
CB_DAILY_REDUCE       = -0.030
CB_DAILY_PAUSE        = -0.050
CB_PEAK_SHUTDOWN      = -0.080
# Circuit breaker por símbolo: suspender 4h si ≥3 SL consecutivos
CB_SL_CONSECUTIVE     = 3
CB_SL_SUSPEND_MIN     = 240
FAST_LOOP_S           = 45      # 45s en lugar de 60s: más reactivo, coste mínimo
CANDLE_BUFFER         = 8


# ══════════════════════════════════════════════════════════════════════════════
# ── Dataclasses ───────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MarketRegime:
    symbol:     str
    regime:     str     # BULL / RANGE / BEAR / HIGH_VOL
    strength:   float
    adx:        float
    atr_pct:    float
    trend_up:   bool

    @property
    def is_tradeable(self) -> bool:
        return self.atr_pct < 10.0


@dataclass
class V5Context:
    candle_bull:     float = 0.0
    candle_bear:     float = 0.0
    structure_bias:  str   = "neutral"
    structure_score: float = 0.0
    sr_at_zone:      bool  = False
    sr_zone_score:   float = 0.0
    fib_near:        bool  = False
    fib_level:       str   = ""
    chart_bull:      float = 0.0
    chart_bear:      float = 0.0
    chart_patterns:  List  = field(default_factory=list)
    obv_accel:       bool  = False
    cvd_pos:         bool  = False
    absorption:      bool  = False
    sr_zones:        List  = field(default_factory=list)
    fib_tp1:         float = 0.0
    fib_tp2:         float = 0.0


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
    cycle:        Optional[CycleState]
    quality:      str
    reasons:      List[str] = field(default_factory=list)
    ml_proba:     float = 0.5
    v5:           Optional[V5Context] = None
    vol_ratio:    float = 1.0


@dataclass
class SymbolState:
    last_sl_time:       float = 0.0
    sl_count_24h:       int   = 0
    sl_consecutive:     int   = 0    # SL consecutivos actuales (sin win intermedio)
    sl_suspend_until:   float = 0.0  # timestamp hasta el cual el símbolo está suspendido
    last_enrich_price:  float = 0.0  # último precio de cierre enriquecido (cache)
    last_enrich_df:     Optional[Any] = None   # DataFrame enriquecido en caché
    cycle:              Optional[CycleState] = None
    cycle_updated:      float = 0.0   # timestamp de última actualización del ciclo
    last_dca_week:      int   = 0     # semana ISO del último DCA ejecutado


# ══════════════════════════════════════════════════════════════════════════════
# ── Indicadores base ──────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _ema(s, n): return s.ewm(span=n, adjust=False).mean()
def _rsi(c, p=14):
    d=c.diff(); g=d.clip(lower=0).rolling(p).mean()
    l=(-d.clip(upper=0)).rolling(p).mean()
    return 100-100/(1+g/l.replace(0,np.nan))
def _atr(df, p=14):
    hl=df["high"]-df["low"]
    hpc=(df["high"]-df["close"].shift()).abs()
    lpc=(df["low"]-df["close"].shift()).abs()
    return pd.concat([hl,hpc,lpc],axis=1).max(axis=1).ewm(span=p,adjust=False).mean()
def _bbands(c, p=20, s=2.0):
    m=c.rolling(p).mean(); σ=c.rolling(p).std()
    return m-s*σ, m, m+s*σ
def _macd(c, f=12, sl=26, sg=9):
    ef=_ema(c,f); es=_ema(c,sl); ml=ef-es; sig=_ema(ml,sg)
    return ml, sig, ml-sig
def _adx(df, p=14):
    h,l,c=df["high"],df["low"],df["close"]
    pdm=h.diff().clip(lower=0); mdm=(-l.diff()).clip(lower=0)
    pdm=pdm.where(pdm>mdm,0); mdm=mdm.where(mdm>pdm,0)
    atr_raw=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    atr_s=atr_raw.ewm(span=p,adjust=False).mean()
    pdi=100*pdm.ewm(span=p,adjust=False).mean()/atr_s.replace(0,np.nan)
    mdi=100*mdm.ewm(span=p,adjust=False).mean()/atr_s.replace(0,np.nan)
    dx=(100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan))
    return dx.ewm(span=p,adjust=False).mean().fillna(0)
def _vwap(df):
    tp=(df["high"]+df["low"]+df["close"])/3
    return (tp*df["volume"]).cumsum()/df["volume"].cumsum().replace(0,np.nan)
def _vfi(df, p=130):
    tp=(df["high"]+df["low"]+df["close"])/3
    inter=np.log(tp.clip(1e-10))-np.log(tp.clip(1e-10).shift(1))
    vi=inter.rolling(30).std().fillna(0.01)
    cut=0.1*vi*df["close"]
    vave=df["volume"].rolling(p).mean().shift(1).fillna(1)
    vmax=vave*2.0; mf=tp-tp.shift(1)
    vcp=np.where(mf>cut,df["volume"],np.where(mf<-cut,-df["volume"],0.0))
    vf=pd.Series(vcp,index=df.index).clip(lower=-vmax,upper=vmax)
    return (vf.rolling(p).sum()/vave.replace(0,np.nan)).fillna(0)
def _stoch_rsi(c, p=14):
    r=_rsi(c,p)
    return (r-r.rolling(p).min())/(r.rolling(p).max()-r.rolling(p).min()).replace(0,np.nan)
def _obv(c, v):
    d=np.sign(np.diff(c.values,prepend=c.values[0]))
    return pd.Series(np.cumsum(d*v.values),index=c.index)
def _delta_vol(o,h,l,c,v):
    rng=np.where(h==l,1e-10,h-l)
    return pd.Series(((c-l)/rng-(h-c)/rng).values,index=c.index)

def _hurst_fast(c, window=100, lag=20):
    ret1 = c.pct_change(1)
    ret_lag = c.pct_change(lag)
    std1 = ret1.rolling(window).std().replace(0, np.nan)
    std_lag = ret_lag.rolling(window).std()
    return (np.log(std_lag / std1) / np.log(lag)).fillna(0.5)

def _vwap_rolling(df, window=20):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    tp_vol = tp * df["volume"]
    return tp_vol.rolling(window).sum() / df["volume"].rolling(window).sum().replace(0, np.nan)

def _tstat_fast(c, window=20):
    x_series = pd.Series(np.arange(len(c)), index=c.index)
    r = c.rolling(window).corr(x_series).clip(-0.999, 0.999)
    return (r * np.sqrt(window - 2) / np.sqrt(1 - r**2)).fillna(0.0)



def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Enriquece el DataFrame con todos los indicadores. Compatible V5+V6."""
    df = df.copy()
    c, h, l, o, v = df["close"], df["high"], df["low"], df["open"], df["volume"]

    df["ema9"]  = _ema(c,9);  df["ema21"] = _ema(c,21)
    df["ema55"] = _ema(c,55); df["ema200"]= _ema(c,200)
    df["bb_lower"], df["bb_mid"], df["bb_upper"] = _bbands(c)
    df["bb_width"]   = (df["bb_upper"]-df["bb_lower"])/df["bb_mid"].replace(0,np.nan)
    df["bb_squeeze"] = df["bb_width"] < df["bb_width"].rolling(50).quantile(0.20)
    df["rsi"]        = _rsi(c,14); df["rsi_fast"] = _rsi(c,7)
    df["stoch_rsi"]  = _stoch_rsi(c)
    df["macd"], df["macd_signal"], df["macd_hist"] = _macd(c)
    df["macd_bull"]     = df["macd"] > df["macd_signal"]
    df["macd_growing"]  = df["macd_hist"] > df["macd_hist"].shift(1)
    df["macd_cross_up"] = (df["macd"]>df["macd_signal"])&(df["macd"].shift(1)<=df["macd_signal"].shift(1))
    df["atr"]      = _atr(df); df["atr_pct"] = df["atr"]/c*100
    df["atr_rank"] = df["atr_pct"].rolling(100).rank(pct=True)
    df["adx"]      = _adx(df)
    df["vol_ratio"]= v / v.rolling(20).mean().replace(0,np.nan)
    df["vol_spike"]= df["vol_ratio"] > 2.0
    df["vfi"]      = _vfi(df); df["vfi_bull"] = df["vfi"] > 0
    df["vwap"]     = _vwap(df)
    df["obv"]      = _obv(c, v)
    df["delta_vol"]= _delta_vol(o, h, l, c, v)
    # OBV acelerando
    df["obv_accel"] = (df["obv"] > df["obv"].rolling(10).mean()*1.005) & (df["obv"].diff(3) > 0)
    # CVD positivo (compradores netos > 15%)
    df["cvd_pos"]   = df["delta_vol"].rolling(5).mean() > 0.15
    # Absorción institucional
    rng = h - l
    avg_rng = rng.rolling(20).mean(); avg_vol = v.rolling(20).mean()
    df["absorption"] = ((v > avg_vol*1.8) & (rng < avg_rng*0.6)).astype(np.int8)
    df["trend_up"]   = (df["ema21"]>df["ema55"])&(df["ema55"]>df["ema200"])
    df["trend_down"] = (df["ema21"]<df["ema55"])&(df["ema55"]<df["ema200"])
    df["above_vwap"] = c > df["vwap"]
    df["momentum_3"] = c.pct_change(3)*100

    # ── Indicadores Estadísticos Cuantitativos (Fase 3) ──
    df["hurst_exp"] = _hurst_fast(c, window=100, lag=20)
    
    vwap_20 = _vwap_rolling(df, 20)
    std_20 = c.rolling(20).std().replace(0, np.nan)
    df["zscore_vwap"] = (c - vwap_20) / std_20
    
    df["t_stat"] = _tstat_fast(c, window=20)
    
    returns = c.pct_change()
    vol_cond = np.sqrt((returns**2).ewm(span=20).mean())
    df["vol_ratio_garch"] = vol_cond / vol_cond.shift(5).replace(0, np.nan)

    # Patrones de velas (si V5 disponible)
    if V5_OK:
        try:
            df = enrich_v5(df)
        except Exception:
            pass

    return df


def detect_regime(df: pd.DataFrame, symbol: str) -> MarketRegime:
    if len(df) < 50:
        return MarketRegime(symbol, "RANGE", 0.4, 20.0, 2.0, False)
    last = df.iloc[-1]
    adx      = float(last.get("adx", 20))
    atr_pct  = float(last.get("atr_pct", 2.0))
    atr_rank = float(last.get("atr_rank", 0.5))
    trend_up = bool(last.get("trend_up", False))
    trend_dn = bool(last.get("trend_down", False))
    vfi      = float(last.get("vfi", 0.0))
    if atr_rank > 0.85:
        return MarketRegime(symbol, "HIGH_VOL", 0.9, adx, atr_pct, trend_up)
    if trend_up and adx > 22 and vfi > 0:
        return MarketRegime(symbol, "BULL", min(1.0,(adx-22)/28+0.4), adx, atr_pct, True)
    if trend_dn and adx > 20:
        return MarketRegime(symbol, "BEAR", min(1.0,(adx-20)/30+0.3), adx, atr_pct, False)
    return MarketRegime(symbol, "RANGE", max(0.3,1.0-adx/40), adx, atr_pct, trend_up)


# ══════════════════════════════════════════════════════════════════════════════
# ── Contexto V5 (sin lookahead bias) ─────────────────────────════════════════
# ══════════════════════════════════════════════════════════════════════════════

def build_v5_context(df: pd.DataFrame, direction: str = "long") -> V5Context:
    ctx = V5Context()
    if not V5_OK or len(df) < 50:
        # Fallback: usar columnas calculadas en enrich_dataframe
        last = df.iloc[-1]
        ctx.candle_bull   = float(last.get("cp_bull_score", 0))
        ctx.candle_bear   = float(last.get("cp_bear_score", 0))
        ctx.obv_accel     = bool(last.get("obv_accel", False))
        ctx.cvd_pos       = bool(last.get("cvd_pos", False))
        ctx.absorption    = bool(last.get("absorption", 0))
        return ctx

    last = df.iloc[-1]
    ctx.candle_bull = float(last.get("cp_bull_score", 0))
    ctx.candle_bear = float(last.get("cp_bear_score", 0))
    ctx.obv_accel   = bool(last.get("obv_accel", False))
    ctx.cvd_pos     = bool(last.get("cvd_pos", False))
    ctx.absorption  = bool(last.get("absorption", 0))

    try:
        struct = get_structure_context(df, lookback=min(150, len(df)))
        ctx.structure_bias  = struct["structure"]["bias"]
        ctx.structure_score = float(struct["structure"]["score"])
        ctx.sr_at_zone      = struct["at_zone"]
        ctx.sr_zones        = struct["zones"]
        ctx.chart_patterns  = struct["chart_patterns"]
        if ctx.sr_at_zone and struct["current_zone"]:
            sr = struct["sr_helper"]
            ctx.sr_zone_score = sr.score_for_zone(struct["current_zone"], direction)
        fib = struct["fib_helper"]
        price = struct["price"]
        fib_r = struct["fib_retrace"]
        if fib_r:
            near = fib.nearest_level(price, fib_r, tol=0.008)
            if near:
                ctx.fib_near  = True
                ctx.fib_level = near[0]
        for pat in ctx.chart_patterns:
            if pat.get("direction") in ("bull","neutral"):
                ctx.chart_bull += pat.get("score", 0)
            if pat.get("direction") in ("bear","neutral"):
                ctx.chart_bear += pat.get("score", 0)
    except Exception:
        pass

    return ctx


# ══════════════════════════════════════════════════════════════════════════════
# ── El núcleo: selector de estrategia adaptativo ──────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _quality(score: float) -> str:
    if score >= 90: return "A+"
    if score >= 75: return "A"
    if score >= 60: return "B"
    return "C"


def _targets(entry: float, stop: float, direction: str, ctx: V5Context) -> Tuple[float,float,float]:
    """
    Calcula los take profits con relación riesgo/beneficio optimizada:
      TP1 = 1.5R  → cierre parcial 50% (asegura ganancia con win rate ≥35%)
      TP2 = 3.0R  → objetivo principal 1:3 (compounding sostenible)
      TP3 = 5.0R  → trailing stop largo en trades excepcionales

    Si hay un nivel Fibonacci cercano a TP2 (±1%), se ajusta TP2 a ese nivel
    para respetar resistencias naturales del mercado.
    """
    dist = abs(entry - stop)
    if direction == "long":
        tp1 = entry + dist * 1.5
        tp2 = entry + dist * 3.0
        tp3 = entry + dist * 5.0
        # Ajuste Fibonacci en TP2 si hay nivel cercano
        if ctx.fib_tp2 > tp1 and abs(ctx.fib_tp2 - tp2) / tp2 < 0.015:
            tp2 = ctx.fib_tp2
    else:
        tp1 = entry - dist * 1.5
        tp2 = entry - dist * 3.0
        tp3 = entry - dist * 5.0
    return tp1, tp2, tp3


def _sl_from_sr(base_sl: float, direction: str, ctx: V5Context) -> float:
    """Ajusta SL a zona S/R si hay una más precisa cerca."""
    for z in ctx.sr_zones[:3]:
        if direction == "long" and z["type"] == "support":
            candidate = z["bottom"] * 0.997
            if base_sl < candidate < base_sl * 1.02:
                return candidate
        elif direction == "short" and z["type"] == "resistance":
            candidate = z["top"] * 1.003
            if base_sl * 0.98 < candidate < base_sl:
                return candidate
    return base_sl


def _v5_confirmed(ctx: V5Context, phase: str, strategy: str) -> Tuple[bool, str]:
    """
    Evalúa si las capas V5 confirman la señal.
    
    La lógica varía según la fase del ciclo:
    - En BULL: más permisivo (el mercado ayuda)
    - En BEAR/ACCUM: más estricto (solo las mejores confluencias)
    
    Retorna (confirmado, razón de rechazo)
    """
    # Señal fuerte — válida en cualquier fase
    if ctx.candle_bull >= 20 or ctx.absorption:
        return True, ""

    # Excepción explícita para MeanReversion:
    # Si choca contra un soporte fuerte, no pedimos tendencia alcista V5
    if strategy == "MeanReversion" and (ctx.sr_at_zone or ctx.fib_near):
        return True, ""

    # Contar señales moderadas
    mods = [
        ctx.candle_bull >= 10,
        ctx.structure_bias == "bull",
        ctx.sr_at_zone,
        ctx.obv_accel,
        ctx.cvd_pos,
    ]
    n = sum(mods)

    # En fases alcistas del ciclo macro: 1 señal moderada es suficiente
    if phase in ("BULL_EARLY", "BULL_MATURE"):
        return n >= 1, "" if n >= 1 else "sin confirmación V5"

    # En acumulación: 2 señales
    if phase in ("ACCUMULATION", "BULL_LATE", "DISTRIBUTION"):
        return n >= 2, "" if n >= 2 else f"V5 insuficiente ({n}/2 señales)"

    # En bear: 2 señales incluyendo obligatoriamente vela de reversión o S/R
    if phase in ("BEAR_DEEP", "BEAR_RECOVERY"):
        bear_ok = (ctx.candle_bull >= 12 or ctx.sr_at_zone) and n >= 2
        return bear_ok, "" if bear_ok else f"V5 bear insuficiente ({n}/2 + vela/SR)"

    # Fallback
    return n >= 2, "" if n >= 2 else f"V5 insuficiente ({n}/2)"


class AdaptiveSignalSelector:
    """
    Selecciona y puntúa señales adaptándose a la fase del ciclo macro.
    
    Este es el corazón del V6: no hay estrategias fijas.
    El ciclo macro determina qué buscar, con qué agresividad,
    y qué nivel de confirmación exigir.
    """

    def analyze(
        self,
        df:     pd.DataFrame,
        symbol: str,
        regime: MarketRegime,
        cycle:  Optional[CycleState],
        hour_utc: int,
    ) -> Optional[SignalResult]:
        """
        Punto de entrada único. Evalúa el contexto completo y retorna
        la mejor señal disponible, o None si no hay setup válido.
        """
        if len(df) < 220:
            return None

        phase = cycle.phase if cycle else "ACCUMULATION"
        active = cycle.active_strategies if cycle else ["TrendFollowing","MeanReversion"]

        ctx  = build_v5_context(df, "long")
        last = df.iloc[-1]

        candidates = []

        # ── TrendFollowing ────────────────────────────────────────────────────
        if "TrendFollowing" in active and regime.regime not in ("BEAR",) and cycle and cycle.is_bull:
            sig = self._trend_following(df, symbol, regime, cycle, ctx, last)
            if sig:
                candidates.append(sig)

        # ── TrendFollowing en BULL pero régimen local RANGE ───────────────────
        if "TrendFollowing" in active and regime.regime == "RANGE" and phase in ("BULL_EARLY","BULL_MATURE"):
            sig = self._trend_following(df, symbol, regime, cycle, ctx, last)
            if sig:
                candidates.append(sig)

        # ── MeanReversion ────────────────────────────────────────────────────
        if "MeanReversion" in active:
            mr_hours = (MR_HOURS_BEAR if phase in ("BEAR_DEEP","BEAR_RECOVERY")
                       else MR_HOURS_ACCUM if phase == "ACCUMULATION"
                       else MR_HOURS_BULL)
            if hour_utc in mr_hours:
                sig = self._mean_reversion(df, symbol, regime, cycle, ctx, last, phase)
                if sig:
                    candidates.append(sig)

        # ── Breakout ────────────────────────────────────────────────────────
        if "Breakout" in active and regime.regime in ("RANGE","BULL"):
            sig = self._breakout(df, symbol, regime, cycle, ctx, last, phase)
            if sig:
                candidates.append(sig)

        # ── MomentumScalp ───────────────────────────────────────────────────
        if "MomentumScalp" in active and phase in ("BULL_MATURE","BULL_EARLY") and regime.regime != "BEAR":
            sig = self._trend_following(df, symbol, regime, cycle, ctx, last)
            if sig:
                candidates.append(sig)

        # ── DCA Automático en Suelos (BTC/USDC en BEAR_DEEP con colapso) ────
        if symbol == "BTC/USDC" and phase == "BEAR_DEEP":
            sig = self._dca_bear_floor(df, symbol, regime, cycle, ctx, last)
            if sig:
                # DCA override everything
                return sig

        # ── TrendFollowing SHORT (Altcoins en BEAR_DEEP/DISTRIBUTION) ───────
        if MARGIN_ENABLED and symbol != "BTC/USDC" and phase in ("BEAR_DEEP", "DISTRIBUTION"):
            sig = self._trend_following_short(df, symbol, regime, cycle, ctx, last, phase)
            if sig:
                candidates.append(sig)

        if not candidates:
            return None

        # Seleccionar la mejor señal
        return max(candidates, key=lambda s: s.score)

    # ── Implementaciones de estrategias ───────────────────────────────────────

    def _dca_bear_floor(self, df, symbol, regime, cycle, ctx, last) -> Optional[SignalResult]:
        """
        DCA forzado en suelo histórico absoluto.
        Independiente del signal scoring normal.
        """
        c = float(last["close"])
        rsi = float(last.get("rsi", 50))
        zscore = float(last.get("zscore_vwap", 0))
        
        # DCA optimization: price below 200W SMA instead of fixed ATH drawdown
        if not cycle or cycle.pct_from_200w > 0:
            return None
            
        # Condiciones de colapso histórico
        if rsi < 15 or zscore < -3.5:
            now_week = datetime.now(tz=timezone.utc).isocalendar().week
            ss = self._engine._sym_state.get(symbol)
            if ss and ss.last_dca_week != now_week:
                # Actualizar tracker (se marcará como ejecutado aunque el engine lo rechace por risk, 
                # pero para este backtest/live es suficiente)
                ss.last_dca_week = now_week
                
                atr = float(last.get("atr", c*0.015))
                # SL muy amplio para DCA, TP lejano
                sl = c * 0.70  # 30% drop protection
                tp1 = c * 1.50 # 50% up
                
                return SignalResult(
                    symbol=symbol, strategy="DCA_BEAR_FLOOR", direction="long",
                    score=100.0, entry_price=c, stop_loss=sl, tp1=tp1, tp2=tp1*1.5, tp3=tp1*2.0,
                    atr=atr, regime=regime, cycle=cycle, quality="A+",
                    reasons=["DCA_BEAR_FLOOR", f"pct_200w={cycle.pct_from_200w:.2f}%", f"RSI={rsi:.1f}"],
                    v5=ctx, vol_ratio=1.0,
                )
        return None

    def _trend_following_short(self, df, symbol, regime, cycle, ctx, last, phase) -> Optional[SignalResult]:
        c      = float(last["close"])
        score  = 0.0
        reasons: List[str] = []

        if bool(last.get("trend_down", False)):    score += 25; reasons.append("EMA_stack_down")
        adx = float(last.get("adx", 0))
        if adx > 25:   score += 25; reasons.append(f"ADX={adx:.0f}")
        elif adx > 20: score += 15; reasons.append(f"ADX={adx:.0f}")
        rsi = float(last.get("rsi", 50))
        if 30 <= rsi <= 60: score += 15; reasons.append(f"RSI={rsi:.0f}")
        if not bool(last.get("macd_bull")) and not bool(last.get("macd_growing")):
            score += 15; reasons.append("MACD_bear")
        ema21 = float(last.get("ema21", c))
        if abs(c-ema21)/ema21*100 < 1.5: score += 10; reasons.append("pullback_ema21")
        if not bool(last.get("vfi_bull")): score += 5; reasons.append("VFI_bear")
        
        # Validación estadística (Fase 3)
        t_stat = float(last.get("t_stat", 0.0))
        if t_stat < -2.0:
            score += 15; reasons.append(f"t_stat={t_stat:.1f}")
        elif t_stat > -1.0:
            log.debug("tf_short_tstat_invalid", symbol=symbol, t_stat=f"{t_stat:.1f}")
            return None  # Requiere tendencia bajista estadísticamente válida

        # Bonus según fase del ciclo
        if phase == "BEAR_DEEP": score *= 1.15; reasons.append("CYCLE:BEAR_DEEP")
        elif phase == "DISTRIBUTION": score *= 1.10; reasons.append("CYCLE:DISTRIBUTION")

        min_score = 65 if phase == "BEAR_DEEP" else 75
        if score < min_score:
            log.debug("tf_short_score_below_threshold", symbol=symbol, score=f"{score:.0f}", min_score=min_score)
            return None

        # Confirmación V5
        ok, reason = _v5_confirmed(ctx, phase, "TrendFollowing")
        # Invertir requerimiento para V5 (simplificado: si hay confirmación bear)
        if not (ctx.candle_bear >= 10 or ctx.structure_bias == "bear" or ctx.sr_at_zone):
            log.debug("tf_short_v5_no_confirmation", symbol=symbol)
            return None
        
        if not ok:
            log.debug("tf_short_v5_rejected", symbol=symbol, reason=reason)
            return None

        atr  = float(last.get("atr", c*0.015))
        high5 = float(df["high"].iloc[-5:].max())
        sl   = _sl_from_sr(min(c+2.0*atr, high5+0.3*atr, c*1.022), "short", ctx)
        if c >= sl: 
            log.debug("tf_short_sl_overlap", symbol=symbol, entry=c, sl=sl)
            return None
        tp1, tp2, tp3 = _targets(c, sl, "short", ctx)

        return SignalResult(
            symbol=symbol, strategy="TrendFollowing", direction="short",
            score=score, entry_price=c, stop_loss=sl, tp1=tp1, tp2=tp2, tp3=tp3,
            atr=atr, regime=regime, cycle=cycle, quality=_quality(score),
            reasons=reasons, v5=ctx, vol_ratio=float(last.get("vol_ratio_garch", 1.0)),
        )

    def _trend_following(self, df, symbol, regime, cycle, ctx, last) -> Optional[SignalResult]:
        phase  = cycle.phase if cycle else "ACCUMULATION"
        c      = float(last["close"])
        score  = 0.0
        reasons: List[str] = []

        if bool(last.get("trend_up", False)):    score += 25; reasons.append("EMA_stack")
        adx = float(last.get("adx", 0))
        if adx > 25:   score += 25; reasons.append(f"ADX={adx:.0f}")
        elif adx > 20: score += 15; reasons.append(f"ADX={adx:.0f}")
        rsi = float(last.get("rsi", 50))
        if 40 <= rsi <= 70: score += 15; reasons.append(f"RSI={rsi:.0f}")
        if bool(last.get("macd_bull")) and bool(last.get("macd_growing")):
            score += 15; reasons.append("MACD_bull")
        ema21 = float(last.get("ema21", c))
        if abs(c-ema21)/ema21*100 < 1.5: score += 10; reasons.append("pullback_ema21")
        if bool(last.get("vfi_bull")): score += 5; reasons.append("VFI")
        
        # Validación estadística (Fase 3)
        t_stat = float(last.get("t_stat", 0.0))
        if t_stat > 2.0:
            score += 15; reasons.append(f"t_stat={t_stat:.1f}")
        elif t_stat < 1.0:
            log.debug("tf_tstat_below_threshold", symbol=symbol, t_stat=f"{t_stat:.2f}")
            return None  # Requiere tendencia estadísticamente válida

        # Bonus según fase del ciclo
        if cycle:
            if phase == "BULL_MATURE":  score *= 1.15; reasons.append("CYCLE:BULL_MATURE")
            elif phase == "BULL_EARLY": score *= 1.10; reasons.append("CYCLE:BULL_EARLY")
            elif phase == "BULL_LATE":  score *= 0.85; reasons.append("CYCLE:BULL_LATE⚠️")

        # Umbral adaptativo: más estricto en fases tardías
        min_score = 65 if phase == "BULL_MATURE" else 72 if phase == "BULL_EARLY" else 78
        if score < min_score:
            log.debug("tf_score_below_threshold", symbol=symbol,
                      score=f"{score:.0f}", min_score=min_score, phase=phase)
            return None

        # Confirmación V5
        ok, reason = _v5_confirmed(ctx, phase, "TrendFollowing")
        if not ok:
            log.debug("tf_v5_rejected", symbol=symbol, reason=reason)
            return None

        atr  = float(last.get("atr", c*0.015))
        low5 = float(df["low"].iloc[-5:].min())
        sl   = _sl_from_sr(max(c-2.0*atr, low5-0.3*atr, c*0.978), "long", ctx)
        if c <= sl: 
            log.debug("tf_sl_overlap", symbol=symbol, entry=c, sl=sl)
            return None
        tp1, tp2, tp3 = _targets(c, sl, "long", ctx)

        return SignalResult(
            symbol=symbol, strategy="TrendFollowing", direction="long",
            score=score, entry_price=c, stop_loss=sl, tp1=tp1, tp2=tp2, tp3=tp3,
            atr=atr, regime=regime, cycle=cycle, quality=_quality(score),
            reasons=reasons, v5=ctx, vol_ratio=float(last.get("vol_ratio_garch", 1.0)),
        )

    def _mean_reversion(self, df, symbol, regime, cycle, ctx, last, phase) -> Optional[SignalResult]:
        c = float(last["close"])
        score = 0.0
        reasons: List[str] = []

        adx = float(last.get("adx", 30))
        rsi = float(last.get("rsi", 50))
        bb_lower = float(last.get("bb_lower", c))
        bb_mid   = float(last.get("bb_mid",   c))
        bb_upper = float(last.get("bb_upper", c))
        stoch    = float(last.get("stoch_rsi", 0.5))
        atr      = float(last.get("atr", c*0.015))

        if adx < 20:   score += 20; reasons.append(f"ADX={adx:.0f}_range")
        elif adx < 25: score += 12
        if rsi < 28:   score += 30; reasons.append(f"RSI={rsi:.0f}_extreme")
        elif rsi < 35: score += 20; reasons.append(f"RSI={rsi:.0f}_oversold")
        dist_bb = (c-bb_lower)/bb_lower*100 if bb_lower > 0 else 999
        if dist_bb < 0.5:  score += 20; reasons.append("AT_BB_lower")
        elif dist_bb < 2:  score += 12; reasons.append("NEAR_BB_lower")
        if stoch < 0.15:   score += 15; reasons.append(f"StochRSI={stoch:.2f}_extreme")
        elif stoch < 0.25: score += 8
        if bool(last.get("macd_cross_up")): score += 15; reasons.append("MACD_cross")
        elif bool(last.get("macd_growing")): score += 8

        # En fase BEAR: penalizar si no hay sobreventa real en lugar de vetar
        zscore = float(last.get("zscore_vwap", 0.0))
        if phase in ("BEAR_DEEP","BEAR_RECOVERY"):
            if rsi > 35 and zscore > -1.5:
                # Penalizacion en lugar de veto: condiciones moderadas en bear
                score -= 20
                log.debug("mr_bear_no_oversold_penalty", symbol=symbol,
                          rsi=f"{rsi:.0f}", zscore=f"{zscore:.2f}", score_after=f"{score:.0f}")
            if zscore < -2.0:
                score += 20; reasons.append(f"zscore={zscore:.1f}_extreme")
            elif zscore < -1.5:
                score += 10; reasons.append("BEAR_oversold")
            if cycle and cycle.phase_strength > 0.90:
                # Penalizacion en lugar de veto: bear muy fuerte reduce score
                score -= 25
                log.debug("mr_bear_very_strong_penalty", symbol=symbol,
                          phase_strength=f"{cycle.phase_strength:.2f}", score_after=f"{score:.0f}")

        # Boost por Hurst Exponent (Fase 3)
        hurst = float(last.get("hurst_exp", 0.5))
        if hurst < 0.45:
            score += 10; reasons.append(f"hurst={hurst:.2f}_mr")
            
        # Bonus por cycle
        if cycle and phase == "ACCUMULATION":
            score += 8; reasons.append("CYCLE:ACCUM_bonus")

        # Bonus por Soportes/Resistencias institucionales
        if ctx.sr_at_zone:
            score += 15; reasons.append("SR_Support_Zone")
        if ctx.fib_near:
            score += 10; reasons.append("FIB_Support")

        # Umbral adaptativo por fase del ciclo (NOT es el AND-serial de antes)
        min_score = {
            "BEAR_DEEP":     55,
            "BEAR_RECOVERY": 58,
            "ACCUMULATION":  55,
            "BULL_EARLY":    50,
            "BULL_MATURE":   45,
            "BULL_LATE":     60,
            "DISTRIBUTION":  65,
        }.get(phase, 60)
        if score < min_score:
            log.debug("mr_score_below_threshold", symbol=symbol,
                      score=f"{score:.0f}", min_score=min_score, phase=phase)
            return None

        # Confirmacion V5 — para MR exigimos soporte o Fibonacci
        if not (ctx.candle_bull >= 12 or ctx.sr_at_zone or ctx.fib_near):
            log.debug("mr_no_v5_confirmation", symbol=symbol,
                      candle_bull=ctx.candle_bull, sr_at_zone=ctx.sr_at_zone, fib_near=ctx.fib_near)
            return None
        ok, reason = _v5_confirmed(ctx, phase, "MeanReversion")
        if not ok:
            log.debug("mr_v5_rejected", symbol=symbol, reason=reason)
            return None

        atr_mult = 2.5 if phase in ("BEAR_DEEP","BEAR_RECOVERY") else 2.0
        sl   = _sl_from_sr(max(bb_lower-atr_mult*atr, c*0.965), "long", ctx)
        if c <= sl: return None
        tp1, tp2, tp3 = _targets(c, sl, "long", ctx)
        # TP2 natural en MR = BB media
        tp2 = max(tp2, bb_mid)

        return SignalResult(
            symbol=symbol, strategy="MeanReversion", direction="long",
            score=score, entry_price=c, stop_loss=sl, tp1=tp1, tp2=tp2, tp3=tp3,
            atr=atr, regime=regime, cycle=cycle, quality=_quality(score),
            reasons=reasons, v5=ctx, vol_ratio=float(last.get("vol_ratio_garch", 1.0)),
        )

    def _breakout(self, df, symbol, regime, cycle, ctx, last, phase) -> Optional[SignalResult]:
        c = float(last["close"])
        score = 0.0
        reasons: List[str] = []

        sq_n = int(df.get("bb_squeeze", pd.Series(False)).iloc[-20:].sum()) if "bb_squeeze" in df.columns else 0
        if sq_n >= 10: score += 25; reasons.append(f"squeeze_{sq_n}v")
        bb_upper = float(last.get("bb_upper", c))
        if c > bb_upper: score += 20; reasons.append("close_above_BB")
        vol_r = float(last.get("vol_ratio", 1))
        if vol_r > 2.0:   score += 20; reasons.append(f"vol_{vol_r:.1f}x")
        elif vol_r > 1.5: score += 12
        if len(df) > 21:
            h20 = float(df["high"].iloc[-21:-1].max())
            if c > h20: score += 15; reasons.append("break_20h_high")
        if bool(last.get("macd_bull")): score += 10; reasons.append("MACD")

        # Breakout REQUIERE volumen institucional
        if not (ctx.absorption or ctx.obv_accel):
            log.debug("breakout_no_institutional_vol", symbol=symbol,
                      absorption=ctx.absorption, obv_accel=ctx.obv_accel)
            return None

        if score < 65:
            log.debug("breakout_score_below_threshold", symbol=symbol, score=f"{score:.0f}")
            return None

        atr  = float(last.get("atr", c*0.015))
        prev_upper = float(df["bb_upper"].iloc[-2]) if len(df) > 1 else bb_upper
        sl   = _sl_from_sr(max(prev_upper-2.0*atr, c*0.972), "long", ctx)
        if c <= sl: return None
        tp1, tp2, tp3 = _targets(c, sl, "long", ctx)

        return SignalResult(
            symbol=symbol, strategy="Breakout", direction="long",
            score=score, entry_price=c, stop_loss=sl, tp1=tp1, tp2=tp2, tp3=tp3,
            atr=atr, regime=regime, cycle=cycle, quality=_quality(score),
            reasons=reasons, v5=ctx, vol_ratio=float(last.get("vol_ratio_garch", 1.0)),
        )

    def _momentum_scalp(self, df, symbol, regime, cycle, ctx, last) -> Optional[SignalResult]:
        c = float(last["close"])
        prev = df.iloc[-2] if len(df) > 1 else last
        score = 0.0
        reasons: List[str] = []

        rsi_n = float(last.get("rsi", 50)); rsi_p = float(prev.get("rsi", 50))
        rsi_f = float(last.get("rsi_fast", 50))
        vwap  = float(last.get("vwap", c))
        e9n   = float(last.get("ema9", c)); e21n = float(last.get("ema21", c))
        e9p   = float(prev.get("ema9", c)); e21p = float(prev.get("ema21", c))
        mh    = float(last.get("macd_hist", 0))

        if rsi_p < 52 and rsi_n > 58 and rsi_f > 60:
            score += 25; reasons.append(f"RSI_accel_{rsi_p:.0f}→{rsi_n:.0f}")
        elif rsi_n > 55 and rsi_n > rsi_p+3:
            score += 15; reasons.append(f"RSI_mom_{rsi_n:.0f}")
        if c > vwap*1.001: score += 20; reasons.append("above_VWAP")
        if e9p <= e21p and e9n > e21n: score += 20; reasons.append("EMA9_cross")
        elif e9n > e21n:               score += 10
        vol3 = df["volume"].iloc[-3:]
        if len(vol3)==3 and vol3.is_monotonic_increasing:
            score += 15; reasons.append("vol_3bar_up")
        if mh > 0 and bool(last.get("macd_growing")): score += 20; reasons.append("MACD_grow")
        elif mh > 0: score += 10

        # Scalp solo con OBV acelerando
        if not ctx.obv_accel:
            log.debug("scalp_no_obv_accel", symbol=symbol)
            return None
        if not (ctx.cvd_pos or ctx.candle_bull >= 10):
            log.debug("scalp_no_cvd_or_candle", symbol=symbol,
                      cvd_pos=ctx.cvd_pos, candle_bull=ctx.candle_bull)
            return None
        if score < 65:
            log.debug("scalp_score_below_threshold", symbol=symbol, score=f"{score:.0f}")
            return None

        atr  = float(last.get("atr", c*0.015))
        sl   = _sl_from_sr(max(c-1.2*atr, c*0.985), "long", ctx)
        if c <= sl: return None
        tp1, tp2, tp3 = _targets(c, sl, "long", ctx)

        return SignalResult(
            symbol=symbol, strategy="MomentumScalp", direction="long",
            score=score, entry_price=c, stop_loss=sl, tp1=tp1, tp2=tp2, tp3=tp3,
            atr=atr, regime=regime, cycle=cycle, quality=_quality(score),
            reasons=reasons, v5=ctx, vol_ratio=float(last.get("vol_ratio_garch", 1.0)),
        )


# ══════════════════════════════════════════════════════════════════════════════
# ── Sizing adaptativo al ciclo ────────────────────────────────════════════════
# ══════════════════════════════════════════════════════════════════════════════

def compute_position_size(
    signal: SignalResult, capital: float,
    open_count: int, daily_pnl_pct: float,
    confluence_score: int = 0
) -> Tuple[float, float, float]:
    """Kelly fraccionado con multiplicadores adaptativos al ciclo."""
    if signal.strategy == "DCA_BEAR_FLOOR":
        # DCA fijo de muy bajo riesgo independientemente de SL dist
        dist = abs(signal.entry_price - signal.stop_loss)
        if dist <= 0: return 0.0, 0.0, 0.0
        # Forzar un nominal de aprox $15 USD
        notional = 15.0
        units = notional / signal.entry_price
        risk = units * dist
        return units, risk, notional

    if capital < 1500:      risk = RISK_PER_TRADE_USD
    elif capital < 5000:    risk = capital * 0.010
    elif capital < 20_000:  risk = capital * 0.015
    else:                   risk = capital * 0.020

    # Multiplicador del ciclo
    if signal.cycle:
        risk *= signal.cycle.risk_multiplier
        
    # Multiplicador por Confluence Score (Fase 5)
    if confluence_score > 75:
        risk *= 1.25  # High conviction setup
    elif confluence_score < 25:
        risk *= 0.75  # Low conviction setup

    risk = min(risk, capital * MAX_RISK_PCT)

    # Multiplicador de régimen local
    risk *= {"BULL":1.0,"RANGE":0.7,"HIGH_VOL":0.5}.get(signal.regime.regime, 0.4)

    # Multiplicador del ciclo macro — el más importante
    if signal.cycle:
        # Extra en BULL_MATURE si señal de calidad
        if signal.cycle.phase == "BULL_MATURE" and signal.quality in ("A+","A"):
            risk *= 1.20
        # Reducir siempre en DISTRIBUTION
        if signal.cycle.phase == "DISTRIBUTION":
            risk *= 0.50

    # Penalización a BTC (Sizing reducido al 50% max)
    if signal.symbol == "BTC/USDC":
        risk *= 0.50

    # Circuit breaker diario
    if daily_pnl_pct < CB_DAILY_REDUCE:
        risk *= 0.5

    # Kelly
    quality_mult = {"A+":1.0,"A":0.8,"B":0.6,"C":0.4}.get(signal.quality, 0.6)
    risk *= quality_mult

    wr = 0.50 + signal.ml_proba * 0.10
    b  = abs(signal.tp1-signal.entry_price) / max(abs(signal.entry_price-signal.stop_loss), 1e-10)
    kf = max(0, (wr*b-(1-wr))/b) * KELLY_FRACTION
    if kf > 0:
        risk = min(risk, capital*kf)

    risk = max(risk, 3.0)
    dist = abs(signal.entry_price - signal.stop_loss)
    
    # Aplicamos vol_ratio de GARCH como multiplicador del SL
    # Limitar entre 0.5 (compresión) y 2.0 (alta volatilidad)
    vol_mult = max(0.5, min(2.0, signal.vol_ratio))
    dist = dist * vol_mult
    
    # Modificar el stop_loss real en la señal
    if signal.direction == "long":
        signal.stop_loss = signal.entry_price - dist
    else:
        signal.stop_loss = signal.entry_price + dist

    if dist <= 0: return 0.0, 0.0, 0.0

    units    = risk / dist
    max_not  = capital * 0.35
    if units * signal.entry_price > max_not:
        units = max_not / signal.entry_price

    notional = units * signal.entry_price
    if notional < 11.0: return 0.0, 0.0, 0.0

    return units, risk, notional


# ══════════════════════════════════════════════════════════════════════════════
# ── Motor Principal V6 ────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

class LiveEngineV6:

    def __init__(self, paper_mode: bool = True) -> None:
        self.paper_mode = paper_mode
        self._running   = False
        self._tasks:    List[asyncio.Task] = []
        self.state:     Dict[str, Any]     = {
            "paused": False, "kill": False, "paper_mode": paper_mode,
            "capital": INITIAL_CAPITAL, "open_positions": [],
            "cycles": {}, "regimes": {}, "last_signals": {},
            # Embudo de senales (Tarea 5 — telemetria)
            "funnel": {
                "total_evaluations": 0,   # llamadas a _analyze_symbol hoy
                "passed_regime": 0,        # superaron detect_regime.is_tradeable
                "generated_signal": 0,     # selector.analyze devolvio senal
                "executed_trade": 0,       # trade enviado al exchange/paper
                "consecutive_zero_days": 0,
                "last_trade_date": None,
                "reset_date": None,        # fecha del ultimo reset diario
            },
        }
        self._sym_state:  Dict[str, SymbolState]     = {s: SymbolState() for s in SYMBOLS}
        self._df_cache:   Dict[str, pd.DataFrame]    = {}
        self._pending:    Dict[str, SignalResult]     = {}
        self._recent_err: List[float]                = []
        self._cycle_detector = CycleDetector() if CYCLE_OK else None
        self._selector   = AdaptiveSignalSelector()
        self._exchange:  Optional[ccxt.Exchange]     = None
        self._portfolio: Optional[PaperPortfolio]    = None
        self._bot:       Optional[TelegramBot]       = None
        self._ml:        Optional[MetaLabeler]       = None

    async def start(self) -> None:
        self._running = True   # ← CRÍTICO: los loops usan while self._running
        log.info("engine_v6_starting", version=ENGINE_VERSION,
                 paper=self.paper_mode, cycle_detector=CYCLE_OK,
                 v5_indicators=V5_OK, symbols=SYMBOLS)

        self._exchange = ccxt.binance({
            "apiKey": os.environ.get("BINANCE_API_KEY",""),
            "secret": os.environ.get("BINANCE_API_SECRET",""),
            "options": {"defaultType":"spot"}, "enableRateLimit": True,
        })
        await self._ping_exchange()

        raw_db = os.environ.get("DATABASE_URL","")
        async_url = (raw_db.replace("postgresql://","postgresql+asyncpg://")
                          .replace("postgres://","postgresql+asyncpg://")
                          .replace("localhost","127.0.0.1"))
        self._portfolio = PaperPortfolio(initial_capital=INITIAL_CAPITAL, db_url=async_url)
        await self._portfolio.initialize()
        self.state["capital"]        = await self._portfolio.get_current_capital()
        self.state["open_positions"] = await self._portfolio.get_open_positions()

        self._ml = MetaLabeler(model_path=str(PROJECT_ROOT/"ml"/"model.joblib"))

        token   = os.environ.get("TELEGRAM_TOKEN","")
        chat_id = int(os.environ.get("TELEGRAM_ALLOWED_USER_ID","0") or "0")
        if token and chat_id:
            self._bot = TelegramBot(token=token, allowed_chat_id=chat_id)
            await self._bot.start()
            await self._bot.send_startup(paper_mode=self.paper_mode,
                                          capital=self.state["capital"],
                                          n_positions=len(self.state["open_positions"]))

        # Inicializar ciclos desde datos históricos
        await self._update_all_cycles()

        asyncio.get_event_loop().add_signal_handler(
            signal.SIGTERM, lambda: asyncio.create_task(self.shutdown("SIGTERM"))
        )

        log.info("engine_v6_live", mode="paper" if self.paper_mode else "LIVE",
                 strategies="adaptive", cycle_ok=CYCLE_OK, v5_ok=V5_OK)

        t1 = asyncio.create_task(self._loop_slow(),   name="loop_slow")
        t2 = asyncio.create_task(self._loop_entry(),  name="loop_entry")
        t3 = asyncio.create_task(self._loop_fast(),   name="loop_fast")
        t4 = asyncio.create_task(self._loop_macro(),  name="loop_macro")
        self._tasks = [t1, t2, t3, t4]
        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            log.exception("engine_v6_fatal", error=str(exc))
            raise

    async def shutdown(self, reason: str = "manual") -> None:
        log.warning("shutdown_v6", reason=reason)
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self._exchange:
            await self._exchange.close()
        if self._bot:
            await self._bot.stop()

    # ── Loop macro: actualiza el ciclo una vez al día ─────────────────────────
    async def _loop_macro(self) -> None:
        while self._running:
            # Ejecutar a las 01:00 UTC (después de que el OHLCV diario se actualice)
            now  = datetime.now(timezone.utc)
            next_run = now.replace(hour=1, minute=0, second=0, microsecond=0)
            if next_run <= now:
                next_run += timedelta(days=1)
            wait = (next_run - now).total_seconds()
            await asyncio.sleep(min(wait, 3600))  # máximo 1h de espera
            await self._update_all_cycles()

    async def _update_all_cycles(self) -> None:
        """Lee velas diarias de SQLite y actualiza el ciclo macro de cada par."""
        if not self._cycle_detector:
            return
        for symbol in SYMBOLS:
            try:
                df_d = load_daily_ohlcv(symbol, SQLITE_PATH)
                if len(df_d) < 100:
                    continue
                state = self._cycle_detector.detect(df_d)
                self._sym_state[symbol].cycle         = state
                self._sym_state[symbol].cycle_updated = time.time()
                
                prev_phase = self.state["cycles"].get(symbol, {}).get("phase")
                
                self.state["cycles"][symbol] = {
                    "phase": state.phase,
                    "risk_mult": state.risk_multiplier,
                    "active_strategies": state.active_strategies,
                }
                log.info("cycle_updated", symbol=symbol, phase=state.phase,
                         conviction=state.conviction_score,
                         risk_mult=f"{state.risk_multiplier:.0%}",
                         strategies=state.active_strategies)
                         
                # ── Phase 7: Regime Transition Alerts ──
                if prev_phase and prev_phase != state.phase and self._bot:
                    msg = (
                        f"🔄 <b>Cambio de fase en {symbol}:</b> {prev_phase} → {state.phase}\n"
                        f"RSI diario: {state.rsi_daily} | % ATH: {state.pct_from_ath}%\n"
                        f"Nuevas estrategias: {', '.join(state.active_strategies)}\n"
                        f"Risk multiplier: {state.risk_multiplier:.0%}\n"
                        f"Próxima acción: revisar posiciones abiertas."
                    )
                    asyncio.create_task(self._bot.send_message(msg))
                # ── Persistir en PostgreSQL (para el dashboard) ──────────────
                if self._portfolio:
                    try:
                        strategies_str = ",".join(state.active_strategies) if state.active_strategies else ""
                        async with self._portfolio._session_factory() as s:
                            await s.execute(text("""
                                INSERT INTO cycle_state
                                    (symbol, phase, conviction, risk_multiplier,
                                     rsi_daily, rsi_weekly, pct_from_ath,
                                     active_strategies, last_dca_week, updated_at)
                                VALUES (:sym, :phase, :conv, :risk,
                                        :rsi_d, :rsi_w, :pct_ath, :strats, :last_dca, NOW())
                                ON CONFLICT (symbol) DO UPDATE SET
                                    phase=EXCLUDED.phase,
                                    conviction=EXCLUDED.conviction,
                                    risk_multiplier=EXCLUDED.risk_multiplier,
                                    rsi_daily=EXCLUDED.rsi_daily,
                                    rsi_weekly=EXCLUDED.rsi_weekly,
                                    pct_from_ath=EXCLUDED.pct_from_ath,
                                    active_strategies=EXCLUDED.active_strategies,
                                    last_dca_week=EXCLUDED.last_dca_week,
                                    updated_at=NOW()
                            """), {
                                "sym":   symbol,
                                "phase": state.phase,
                                "conv":  float(state.conviction_score) / 100.0,
                                "risk":  state.risk_multiplier,
                                "rsi_d": state.rsi_daily,
                                "rsi_w": state.rsi_weekly,
                                "pct_ath": state.pct_from_ath,
                                "strats": strategies_str,
                                "last_dca": self._sym_state[symbol].last_dca_week if symbol in self._sym_state else 0,
                            })
                            await s.commit()
                    except Exception as exc2:
                        log.debug("cycle_persist_skip", symbol=symbol, error=str(exc2))
            except Exception as exc:
                log.warning("cycle_update_error", symbol=symbol, error=str(exc))

    # ── Loop lento: análisis 1H ───────────────────────────────────────────────
    async def _loop_slow(self) -> None:
        while self._running:
            wait = _secs_to_next(TF_STRUCT_S)
            await asyncio.sleep(wait + CANDLE_BUFFER)
            if self.state["kill"] or self.state["paused"]:
                continue
            # Optimización ZimaBlade: en BEAR_DEEP solo analizar BTC+ETH
            bear_deep_syms = set()
            for sym, ss in self._sym_state.items():
                if ss.cycle and ss.cycle.phase in ("BEAR_DEEP", "BEAR_RECOVERY"):
                    bear_deep_syms.add(sym)
            active_symbols = (
                SYMBOLS_BEAR_DEEP
                if len(bear_deep_syms) >= len(SYMBOLS) // 2  # mayoría en bear deep
                else SYMBOLS
            )
            for symbol in active_symbols:
                try:
                    await self._analyze_symbol(symbol)
                except Exception as exc:
                    log.exception("analysis_error_v6", symbol=symbol, error=str(exc))

    async def _analyze_symbol(self, symbol: str) -> None:
        # ── Optimización ZimaBlade: caché de DataFrame enriquecido ────────────
        # Si la vela de 1H no ha cerrado y el precio no varió >0.2%, reusar el df
        df_raw = await self._fetch_candles(symbol, TF_STRUCTURE, 300)  # 300 velas (era 500)
        if df_raw is None or len(df_raw) < 220:
            return

        ss = self._sym_state[symbol]
        last_close = float(df_raw["close"].iloc[-1])
        price_drift = abs(last_close - ss.last_enrich_price) / max(ss.last_enrich_price, 1e-10)

        if ss.last_enrich_df is not None and price_drift < 0.002:
            # Precio movió <0.2% desde el último enriquecimiento → reusar
            df = ss.last_enrich_df
        else:
            try:
                df = enrich_dataframe(df_raw)
                
                # ── Alertas Avanzadas (Fase 4) ──
                last_row = df.iloc[-1]
                if symbol == "BTC/USDC" and float(last_row.get("zscore_vwap", 0)) < -2.5:
                    log.info("alert_vwap_mr", symbol=symbol, zscore=f"{last_row['zscore_vwap']:.2f}", msg="Zona de entrada MR")
                
                if ss.last_enrich_df is not None:
                    old_h = float(ss.last_enrich_df.iloc[-1].get("hurst_exp", 0.5))
                    new_h = float(last_row.get("hurst_exp", 0.5))
                    if old_h < 0.45 and new_h > 0.55:
                        log.info("alert_hurst_regime_change", symbol=symbol, old=f"{old_h:.2f}", new=f"{new_h:.2f}", msg="Cambio a régimen tendencial detectado")
                
                if float(last_row.get("vol_ratio_garch", 1.0)) > 2.0:
                    log.info("alert_extreme_volatility", symbol=symbol, vol=f"{last_row['vol_ratio_garch']:.2f}", msg="Volatilidad extrema, sizing reducido")
                
                ss.last_enrich_price = last_close
                ss.last_enrich_df    = df
            except Exception as exc:
                log.warning("enrich_error_v6", symbol=symbol, error=str(exc)); return

        # ── Embudo de senales: reset diario y conteo de evaluaciones ──────────
        funnel = self.state["funnel"]
        today_str = datetime.now(timezone.utc).date().isoformat()
        if funnel.get("reset_date") != today_str:
            # Nuevo dia: resetear contadores pero mantener historial de dias consecutivos
            prev_executed = funnel.get("executed_trade", 0)
            if prev_executed == 0 and funnel.get("reset_date") is not None:
                funnel["consecutive_zero_days"] = funnel.get("consecutive_zero_days", 0) + 1
            else:
                funnel["consecutive_zero_days"] = 0
                funnel["last_trade_date"] = today_str
            funnel["total_evaluations"] = 0
            funnel["passed_regime"] = 0
            funnel["generated_signal"] = 0
            funnel["executed_trade"] = 0
            funnel["reset_date"] = today_str
        funnel["total_evaluations"] += 1

        regime = detect_regime(df, symbol)
        if not regime.is_tradeable:
            return

        funnel["passed_regime"] += 1

        self._df_cache[symbol]            = df
        self.state["regimes"][symbol]     = regime.regime
        cycle = self._sym_state[symbol].cycle
        hour  = datetime.now(timezone.utc).hour

        signal = self._selector.analyze(df, symbol, regime, cycle, hour)
        if signal is None:
            self._pending.pop(symbol, None); return

        funnel["generated_signal"] += 1

        # ML proba
        if self._ml and self._ml.is_ready():
            try:
                signal.ml_proba = self._ml.predict_proba(df)
            except Exception:
                signal.ml_proba = 0.5

        log.info("signal_queued_v6", symbol=symbol,
                 strategy=signal.strategy, score=f"{signal.score:.0f}",
                 quality=signal.quality,
                 phase=cycle.phase if cycle else "?",
                 risk_mult=f"{cycle.risk_multiplier:.0%}" if cycle else "50%",
                 reasons=",".join(signal.reasons[:4]))
        self._pending[symbol] = signal

    # ── Loop entrada: confirmación 15M ────────────────────────────────────────
    async def _loop_entry(self) -> None:
        while self._running:
            wait = _secs_to_next(TF_ENTRY_S)
            await asyncio.sleep(wait + 5)
            if self.state["kill"] or self.state["paused"] or not self._pending:
                continue
                
            # ── Pair Correlation Circuit Breaker (Fase 6) ──
            # Si se detectan múltiples señales simultáneas (ej. BTC y ETH rompen al alza a la vez)
            # ordenamos por ML Proba y Score, y solo ejecutamos la MEJOR señal del batch.
            sorted_signals = sorted(
                self._pending.items(),
                key=lambda x: (x[1].ml_proba, x[1].score),
                reverse=True
            )
            
            best_symbol, best_signal = sorted_signals[0]
            
            if len(sorted_signals) > 1:
                log.info("circuit_breaker_correlation", 
                         msg=f"Filtrando {len(sorted_signals)-1} señales simultáneas. Ejecutando solo {best_symbol}")
                
            try:
                await self._attempt_entry(best_symbol, best_signal)
            except Exception as exc:
                log.exception("entry_error_v6", symbol=best_symbol, error=str(exc))
                
            # Descartamos el resto para no sobre-exponer la cartera al mismo movimiento de mercado
            self._pending.clear()

    async def _attempt_entry(self, symbol: str, signal: SignalResult) -> None:
        open_pos   = await self._portfolio.get_open_positions()
        open_syms  = {p["symbol"] for p in open_pos}
        if symbol in open_syms:
            self._pending.pop(symbol, None); return
        if len(open_pos) >= MAX_POSITIONS:
            return

        # Filtro correlación
        grp = CORRELATION_GROUPS.get(symbol, "other")
        if grp != "btc":
            n_grp = sum(1 for p in open_pos if CORRELATION_GROUPS.get(p.get("symbol",""),"other") == grp)
            if n_grp >= MAX_CORR_POS: return

        # Cooldown tras SL
        ss = self._sym_state[symbol]
        now_ts = time.time()
        if now_ts - ss.last_sl_time < COOLDOWN_SL_MIN * 60: 
            log.debug("entry_rejected_cooldown", symbol=symbol, min_remaining=int((COOLDOWN_SL_MIN*60 - (now_ts - ss.last_sl_time))/60))
            return

        # ── Circuit breaker: SL consecutivos por símbolo ──────────────────────
        if ss.sl_consecutive >= CB_SL_CONSECUTIVE:
            if now_ts < ss.sl_suspend_until:
                log.info("symbol_suspended_consecutive_sl", symbol=symbol,
                         count=ss.sl_consecutive,
                         resume_in_min=int((ss.sl_suspend_until - now_ts) / 60))
                return
            else:
                # Expiró la suspensión → reiniciar contador
                ss.sl_consecutive = 0

        # Circuit breakers diarios
        capital  = await self._portfolio.get_current_capital()
        daily    = await self._portfolio.get_daily_stats()
        pnl_pct  = float(daily.get("pnl_today",0)) / capital if capital > 0 else 0
        if pnl_pct <= CB_DAILY_PAUSE: return

        # Confirmación 15M
        df15 = await self._fetch_candles(symbol, TF_ENTRY, 100)
        if df15 is None or len(df15) < 30: return
        try:
            df15 = enrich_dataframe(df15)
        except Exception:
            return
        last15 = df15.iloc[-1]
        c15    = float(last15["close"])
        drift  = abs(c15 - signal.entry_price) / signal.entry_price * 100
        if drift > 1.5:
            self._pending.pop(symbol, None); return

        # ── Confirmación micro 15M ────────────────────────────────────────────
        if signal.strategy.startswith("MeanReversion") and signal.direction == "long":
            # Para MR Long en sobreventa, buscamos rechazo, no tendencia
            micro = (int(float(last15.get("rsi", 50)) > 25) +
                     int(float(last15.get("stoch_rsi", 0.5)) > 0.05) +
                     int(bool(last15.get("macd_growing", False))))
        elif signal.direction == "long":
            micro = (int(float(last15.get("rsi", 50)) > 50) +
                     int(bool(last15.get("macd_bull", False))) +
                     int(bool(last15.get("above_vwap", False))))
        else: # SHORT
            micro = (int(float(last15.get("rsi", 50)) < 50) +
                     int(not bool(last15.get("macd_bull", False))) +
                     int(not bool(last15.get("above_vwap", True))))
                     
        if micro == 0: 
            log.debug("entry_rejected_micro", symbol=symbol, rsi=float(last15.get("rsi",50)), stoch=float(last15.get("stoch_rsi",0.5)), macd=bool(last15.get("macd_growing",False)))
            return

        # ── Filtro de volumen: vela de entrada debe tener volumen activo ───────
        # Volumen de la última vela 15M debe ser >= 80% del promedio de 20 velas
        # Evita entradas en velas "muertas" con alta tasa de falsos positivos
        vol_ratio_15m = float(last15.get("vol_ratio", 1.0))
        if vol_ratio_15m < 0.80:
            log.debug("entry_skipped_low_volume", symbol=symbol,
                      vol_ratio=f"{vol_ratio_15m:.2f}")
            return

        signal.entry_price = c15

        # ── Multi-indicator Confluence Score (Fase 5) ─────────────
        h_exp = float(last15.get("hurst_exp", 0.5))
        z_vwap = float(last15.get("zscore_vwap", 0))
        tstat = float(last15.get("t_stat", 0))
        v_garch = float(last15.get("vol_ratio_garch", 1.0))
        
        confluence = 0
        if h_exp < 0.45:       confluence += 25
        if z_vwap < -2.0:      confluence += 25
        if abs(tstat) > 2.0:   confluence += 25
        if v_garch < 0.8:      confluence += 25

        units, risk, notional = compute_position_size(
            signal, capital, len(open_pos), pnl_pct, confluence_score=confluence
        )
        if units <= 0: return

        cycle   = signal.cycle
        cap_pct = notional / capital * 100 if capital > 0 else 0

        log.info("trade_opening_v6", symbol=symbol,
                 strategy=signal.strategy, score=f"{signal.score:.0f}",
                 quality=signal.quality, entry=f"${c15:.4f}",
                 sl=f"${signal.stop_loss:.4f}",
                 tp1=f"${signal.tp1:.4f}", tp2=f"${signal.tp2:.4f}",
                 rr_tp1=f"{abs(signal.tp1-c15)/max(abs(c15-signal.stop_loss),1e-10):.1f}R",
                 rr_tp2=f"{abs(signal.tp2-c15)/max(abs(c15-signal.stop_loss),1e-10):.1f}R",
                 risk=f"${risk:.2f}", notional=f"${notional:.2f}",
                 cap_pct=f"{cap_pct:.1f}%",
                 vol_ratio_15m=f"{vol_ratio_15m:.2f}",
                 phase=cycle.phase if cycle else "?",
                 risk_mult=f"{cycle.risk_multiplier:.0%}" if cycle else "?",
                 ml=f"{signal.ml_proba:.0%}")

        trade = await self._portfolio.open_position(
            symbol=symbol,
            strategy=f"{signal.strategy}_{cycle.phase if cycle else 'UNKNOWN'}",
            entry_price=c15, stop_loss=signal.stop_loss,
            tp1=signal.tp1, tp2=signal.tp2,
            units=units, ml_proba=signal.ml_proba,
            direction=signal.direction, regime=signal.regime.regime,
            risk_amount=risk, notional_usd=notional,
            atr=signal.atr,
            hurst_at_entry=h_exp, zscore_at_entry=z_vwap,
            t_stat_at_entry=tstat, confluence_score=confluence,
        )
        # Contabilizar en el embudo de senales
        self.state["funnel"]["executed_trade"] += 1

        if self._bot:
            await self._bot.send_trade_open(trade)
        await self._log_trade(trade, signal, notional, capital)
        self._pending.pop(symbol, None)

    # ── Loop rápido: gestión de posiciones ────────────────────────────────────
    async def _loop_fast(self) -> None:
        last_daily = datetime.now(tz=timezone.utc).date()
        while self._running:
            await asyncio.sleep(FAST_LOOP_S)
            if self.state["kill"]:
                break
            try:
                prices = await self._get_prices()
            except Exception as exc:
                log.warning("price_error_v6", error=str(exc)); continue

            closed = await self._portfolio.update_positions(prices)
            for ct in closed:
                reason = ct.get("exit_reason", "")
                if self._bot:
                    await self._bot.send_trade_close(ct, reason)
                sym = ct.get("symbol", "")
                if sym in self._sym_state:
                    ss = self._sym_state[sym]
                    if "stop" in reason.lower():
                        ss.last_sl_time   = time.time()
                        ss.sl_consecutive += 1
                        if ss.sl_consecutive >= CB_SL_CONSECUTIVE:
                            ss.sl_suspend_until = time.time() + CB_SL_SUSPEND_MIN * 60
                            log.warning("symbol_auto_suspended", symbol=sym,
                                        consecutive_sl=ss.sl_consecutive,
                                        suspend_min=CB_SL_SUSPEND_MIN)
                            if self._bot:
                                # Notificar suspensión por Telegram
                                await self._bot.send_message(
                                    f"⚠️ {sym} suspendido {CB_SL_SUSPEND_MIN}min "
                                    f"({ss.sl_consecutive} SL consecutivos)"
                                )
                            # Guardar estado de suspensión en la BD
                            try:
                                async with self._portfolio._session_factory() as s:
                                    await s.execute(text(
                                        "UPDATE positions SET status='suspended' WHERE symbol=:sym AND status='open'"
                                    ), {"sym": sym})
                                    await s.commit()
                            except Exception as exc:
                                log.debug("suspend_persist_error", symbol=sym, error=str(exc))
                    else:
                        # Trade cerrado en ganancia → reiniciar contador de SL consecutivos
                        ss.sl_consecutive = 0
                        # Quitar estado de suspensión en la BD si lo hubiera
                        try:
                            async with self._portfolio._session_factory() as s:
                                await s.execute(text(
                                    "UPDATE positions SET status='open' WHERE symbol=:sym AND status='suspended'"
                                ), {"sym": sym})
                                await s.commit()
                        except Exception:
                            pass

            capital = await self._portfolio.get_current_capital()
            daily   = await self._portfolio.get_daily_stats()
            pnl_pct = float(daily.get("pnl_today",0)) / capital if capital > 0 else 0
            self.state.update({
                "capital": capital, "pnl_today_pct": pnl_pct,
                "open_positions": await self._portfolio.get_open_positions(),
            })
            await self._update_heartbeat()
            if self._bot:
                await self._bot.process_updates(self.state)

            now = datetime.now(tz=timezone.utc)
            if now.date() != last_daily and now.hour == 0:
                last_daily = now.date()
                if self._bot and daily:
                    await self._bot.send_daily_pnl({
                        **daily,
                        "capital": capital,
                        "funnel": dict(self.state.get("funnel", {})),
                    })

    # ── Helpers ────────────────────────────────────────────────────────────────
    async def _log_trade(self, trade, signal, notional, capital):
        if not self._portfolio or not self._portfolio._engine:
            return
        cycle = signal.cycle
        cycle_obs = f"phase={cycle.phase} risk_mult={cycle.risk_multiplier:.0%} conviction={cycle.conviction_score}" if cycle else ""
        try:
            async with self._portfolio._session_factory() as s:
                await s.execute(text("""
                    INSERT INTO trades_journal
                        (trade_id,strategy,symbol,timeframe,direction,
                         setup_quality,entry_price,stop_loss,
                         tp1,tp2,take_profit_1,units,position_size,
                         risk_amount,entry_reason,market_regime,regime,
                         ml_proba,entry_time,is_backtest,observations,duration_hours,pnl_pct,r_multiple)
                    VALUES(:tid,:strat,:sym,:tf,:dir,:qual,:ep,:sl,
                           :tp1,:tp2,:tp1,:size,:size,:risk,:reason,
                           :regime,:regime,:ml,NOW(),FALSE,:obs,0,0,0)
                    ON CONFLICT DO NOTHING
                """), {
                    "tid":    trade.get("id",0),
                    "strat":  signal.strategy,
                    "sym":    signal.symbol,
                    "tf":     TF_ENTRY,
                    "dir":    signal.direction,
                    "qual":   ord(signal.quality[0])-64 if signal.quality else 0,
                    "ep":     signal.entry_price,
                    "sl":     signal.stop_loss,
                    "tp1":    signal.tp1,
                    "tp2":    signal.tp2,
                    "size":   trade.get("units",0),
                    "risk":   trade.get("risk_amount",0),
                    "reason": " | ".join(signal.reasons[:5]),
                    "regime": signal.regime.regime,
                    "ml":     signal.ml_proba,
                    "obs":    f"notional=${notional:.2f} cap%={notional/capital*100:.1f}% score={signal.score:.0f} {cycle_obs}",
                })
                await s.commit()
        except Exception as exc:
            log.debug("journal_v6_skip", error=str(exc))

    async def _update_heartbeat(self):
        if not self._portfolio:
            return
        try:
            import json as _json
            open_pos = self.state.get("open_positions", [])
            n_pos    = len(open_pos)
            capital  = self.state.get("capital", 0.0)
            pnl_pct  = self.state.get("pnl_today_pct", 0.0)
            pnl_usd  = round(capital * pnl_pct, 4)
            regimes_json = _json.dumps(self.state.get("regimes", {}))
            # cycles: solo serializar campos básicos (evitar objetos no serializables)
            cycles_simple = {}
            for sym, cyc in self.state.get("cycles", {}).items():
                if isinstance(cyc, dict):
                    cycles_simple[sym] = cyc
                else:
                    cycles_simple[sym] = {"phase": str(cyc)}
            cycles_json = _json.dumps(cycles_simple)
            async with self._portfolio._session_factory() as s:
                await s.execute(text("""
                    INSERT INTO system_heartbeat
                        (id, last_ping, engine_version, paper_mode,
                         active_positions, pnl_today, regimes_json, cycles_json)
                    VALUES (1, NOW(), :v, :p, :npos, :pnl, :reg, :cyc)
                    ON CONFLICT(id) DO UPDATE SET
                        last_ping=NOW(),
                        engine_version=EXCLUDED.engine_version,
                        paper_mode=EXCLUDED.paper_mode,
                        active_positions=EXCLUDED.active_positions,
                        pnl_today=EXCLUDED.pnl_today,
                        regimes_json=EXCLUDED.regimes_json,
                        cycles_json=EXCLUDED.cycles_json
                """), {
                    "v":    ENGINE_VERSION,
                    "p":    self.paper_mode,
                    "npos": n_pos,
                    "pnl":  pnl_usd,
                    "reg":  regimes_json,
                    "cyc":  cycles_json,
                })
                await s.commit()
        except Exception:
            pass

    async def _ping_exchange(self):
        for i in range(1,6):
            try:
                await self._exchange.fetch_time(); return
            except Exception:
                await asyncio.sleep(min(2**i,60))
        raise RuntimeError("No se pudo conectar a Binance")

    async def _fetch_candles(self, symbol, tf, limit=500):
        for i in range(1,6):
            try:
                raw = await self._exchange.fetch_ohlcv(symbol, tf, limit=limit)
                df  = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                return df.dropna()
            except ccxt.NetworkError:
                await asyncio.sleep(min(2**i,60))
            except Exception as exc:
                log.warning("fetch_error_v6", symbol=symbol, tf=tf, e=str(exc))
                return None
        return None

    async def _get_prices(self):
        prices = {}
        for sym in SYMBOLS:
            try:
                t = await self._exchange.fetch_ticker(sym)
                prices[sym] = float(t["last"])
            except Exception:
                if sym in self._df_cache:
                    prices[sym] = float(self._df_cache[sym]["close"].iloc[-1])
        return prices


def _secs_to_next(tf_s: int) -> float:
    epoch = datetime.now(tz=timezone.utc).timestamp()
    return max(math.ceil(epoch/tf_s)*tf_s - epoch, 1.0)


async def _main():
    paper = os.environ.get("PAPER_MODE","true").lower() not in ("false","0","no")
    engine = LiveEngineV6(paper_mode=paper)
    try:
        await engine.start()
    except KeyboardInterrupt:
        await engine.shutdown("KeyboardInterrupt")
    except Exception as e:
        log.exception("fatal_error_traceback", error=str(e))
        await engine.shutdown("fatal_error")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(_main())
