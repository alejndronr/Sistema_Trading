"""
live_engine.py — Motor de Ejecución Cuantitativo V4.0 «Adaptive»
=================================================================
Rediseño completo. Filosofía central: GENERAR OPORTUNIDADES, no esperar
condiciones perfectas. El sistema se adapta al mercado en lugar de esperar
que el mercado se adapte al sistema.

CAMBIOS PRINCIPALES vs V3.0:
──────────────────────────────────────────────────────────────────────────
1. MULTI-TIMEFRAME REAL: 1h estructura + 15m timing de entrada
   → Más precisión sin sacrificar frecuencia de señales

2. 4 ESTRATEGIAS CONCURRENTES con pesos dinámicos por régimen:
   · TrendFollowing  — EMA stack + ADX + retroceso (régimen BULL)
   · MeanReversion   — BB extremos + RSI divergencia (régimen RANGE)
   · Breakout        — BB squeeze + volumen + retest (todos)
   · MomentumScalp   — NUEVA: RSI acceleration + VWAP (régimen BULL/RANGE)

3. SCORING DE SEÑALES (0-100) en lugar de filtros binarios
   → Una señal con score 65/100 ENTRA. Antes requería 5 condiciones AND.

4. GESTIÓN DE RIESGO KELLY FRACCIONADO:
   · Kelly completo: f = (p*b - q) / b
   · Usamos Kelly/4 (ultra-conservador para crypto)
   · Nunca supera 2% del capital ni $20 de riesgo en $1000

5. TRAILING STOP DINÁMICO por ATR (no fijo):
   · En BULL: trailing a 1.5 ATR del máximo
   · En RANGE: trailing a 2 ATR del máximo
   · Breakeven automático al 1R

6. COOLDOWN INTELIGENTE: pausa por símbolo (no global) tras SL

7. CIRCUIT BREAKERS ESCALONADOS:
   · -3% diario → reducir tamaño 50%
   · -5% diario → pausar nuevas entradas
   · -8% sobre pico → apagar motor

8. DETECCIÓN DE RÉGIMEN MEJORADA: combina ADX + EMA + ATR percentil
   para clasificar: BULL / BEAR / RANGE / HIGH_VOLATILITY

Capital inicial: $1,000 USDC
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

import ccxt.async_support as ccxt
import numpy as np
import pandas as pd
import structlog
from dotenv import load_dotenv
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from ml.meta_labeler import MetaLabeler
from monitoring.telegram_bot import TelegramBot
from paper_portfolio import PaperPortfolio

log = structlog.get_logger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# ── Constantes del Motor ──────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

ENGINE_VERSION = "4.0.0-Adaptive"

# Universe de pares — ordenados por liquidez/prioridad
SYMBOLS: List[str] = [
    "BTC/USDC",   # tier-1: mayor liquidez, menor spread
    "ETH/USDC",   # tier-1
    "SOL/USDC",   # tier-2: buena volatilidad, suficiente volumen
    "BNB/USDC",   # tier-2
    "LINK/USDC",  # tier-2: buena tendencia
    "AVAX/USDC",  # tier-3: mayor volatilidad = más oportunidades
]

# Timeframes: estructura en 1h, entrada en 15m
TF_STRUCTURE  = "1h"
TF_ENTRY      = "15m"
TF_STRUCT_SECS = 3600
TF_ENTRY_SECS  = 900

# Parámetros de gestión de cartera
MAX_POSITIONS       = 3       # máximo simultáneo
MAX_POSITIONS_SAME  = 1       # max 1 por símbolo (anti-piramidación)
COMMISSION_RATE     = 0.001   # 0.1% Binance spot estándar
SLIPPAGE_ESTIMATE   = 0.001   # 0.1% slippage estimado
INITIAL_CAPITAL     = float(os.environ.get("INITIAL_CAPITAL", "1000"))

# Riesgo por trade
RISK_PER_TRADE_USD  = 10.0    # $10 fijo mientras capital < $1500
RISK_PCT_TIER_1     = 0.010   # 1.0% para $1500-$5000
RISK_PCT_TIER_2     = 0.015   # 1.5% para $5000-$20000
RISK_PCT_TIER_3     = 0.020   # 2.0% para $20000+
MAX_RISK_PER_TRADE  = 0.020   # nunca superar 2% sea cual sea el capital

# Circuit breakers (nunca tocar estos valores)
CB_DAILY_REDUCE     = -0.030  # -3%: reducir tamaño a 50%
CB_DAILY_PAUSE      = -0.050  # -5%: pausar nuevas entradas
CB_PEAK_SHUTDOWN    = -0.080  # -8% sobre pico: apagar motor

# Loops del motor
FAST_LOOP_SECS  = 60    # revisión de posiciones y precios
CANDLE_BUFFER   = 8     # segundos de margen tras cierre de vela
MAX_RETRIES     = 5
CB_ERR_WINDOW   = 60    # segundos para contar errores de red
CB_ERR_LIMIT    = 4     # errores en ventana antes de pause

# Cooldown por símbolo tras stop loss (en minutos)
COOLDOWN_AFTER_SL_MIN = 90

# Score mínimo para entrar — INTENCIONAL: 55/100, no 80/100
MIN_SIGNAL_SCORE = 55

# Score mínimo específico para MeanReversion (más exigente por riesgo de correlación)
MIN_SCORE_MEAN_REVERSION = 65

# Correlación de cartera: máximo activos de alta correlación simultáneos
# BTC, ETH, SOL, BNB, LINK, AVAX tienen correlación > 0.75 entre sí
# Permitir solo 1 posición abierta en altcoins correlacionadas al mismo tiempo
MAX_CORRELATED_POSITIONS = 1

# Grupos de correlación alta (no abrir más de MAX_CORRELATED_POSITIONS del mismo grupo)
CORRELATION_GROUPS: Dict[str, str] = {
    "BTC/USDC": "btc",          # BTC es su propio grupo (menos correlado)
    "ETH/USDC": "altcoin",      # Las altcoins van juntas
    "SOL/USDC": "altcoin",
    "BNB/USDC": "altcoin",
    "LINK/USDC": "altcoin",
    "AVAX/USDC": "altcoin",
}

# Horario permitido para MeanReversion (UTC): solo sesión asiática (baja volatilidad)
# Evita que MR entre durante London/NY overlap donde los movimientos son más direccionales
MR_ALLOWED_HOURS_UTC: range = range(0, 8)   # 00:00–07:59 UTC = sesión asiática

# Multiplicador ATR para SL de MeanReversion (más holgado para evitar stops prematuros)
MR_ATR_STOP_MULTIPLIER: float = 2.5  # antes era 2.0 implícito

# Peso Kelly fraccionado (Kelly/4 = ultra-conservador)
KELLY_FRACTION = 0.25


# ══════════════════════════════════════════════════════════════════════════════
# ── Dataclasses ───────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MarketRegime:
    """Clasificación del régimen de mercado para un símbolo."""
    symbol:   str
    regime:   str      # BULL | BEAR | RANGE | HIGH_VOL
    strength: float    # 0-1: cuánto nos fiamos de la clasificación
    adx:      float
    atr_pct:  float    # ATR / precio * 100
    trend_up: bool     # ema21 > ema55 > ema200
    vfi:      float    # Volume Flow Indicator

    @property
    def is_tradeable(self) -> bool:
        """HIGH_VOL extremo no es operable sin ajuste especial."""
        return self.atr_pct < 8.0  # si ATR > 8% del precio, saltamos

    @property
    def risk_multiplier(self) -> float:
        """Multiplicador del riesgo base según régimen."""
        if self.regime == "BULL":
            return 1.0
        if self.regime == "RANGE":
            return 0.7
        if self.regime == "HIGH_VOL":
            return 0.5
        return 0.4  # BEAR


@dataclass
class SignalResult:
    """Resultado de evaluación de una señal con scoring."""
    symbol:       str
    strategy:     str       # TrendFollowing | MeanReversion | Breakout | MomentumScalp
    direction:    str       # long | short
    score:        float     # 0-100
    entry_price:  float
    stop_loss:    float
    tp1:          float
    tp2:          float
    tp3:          float     # nuevo: target extendido para trenders
    atr:          float
    regime:       MarketRegime
    quality:      str       # A+ | A | B | C
    reasons:      List[str] = field(default_factory=list)
    ml_proba:     float = 0.5


@dataclass
class SymbolCooldown:
    """Rastrea el cooldown por símbolo tras una pérdida."""
    last_sl_time: float = 0.0
    sl_count_24h: int = 0
    last_sl_reset: float = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# ── Funciones de Análisis Técnico (inline, sin imports circulares) ─────────────
# ══════════════════════════════════════════════════════════════════════════════

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl  = df["high"] - df["low"]
    hpc = (df["high"] - df["close"].shift()).abs()
    lpc = (df["low"]  - df["close"].shift()).abs()
    tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _bollinger(close: pd.Series, period: int = 20, std: float = 2.0):
    mid   = close.rolling(period).mean()
    sigma = close.rolling(period).std()
    return mid - std * sigma, mid, mid + std * sigma


def _macd(close: pd.Series, fast=12, slow=26, signal_p=9):
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal    = _ema(macd_line, signal_p)
    hist      = macd_line - signal
    return macd_line, signal, hist


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ADX simplificado — vectorizado."""
    high, low, close = df["high"], df["low"], df["close"]
    plus_dm  = (high.diff()).clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    plus_dm  = plus_dm.where(plus_dm > minus_dm, 0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0)

    tr = _atr(df, period)  # ATR = suavizado de TR
    atr_raw = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr_s = atr_raw.ewm(span=period, adjust=False).mean()

    plus_di  = 100 * plus_dm.ewm(span=period, adjust=False).mean()  / atr_s.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr_s.replace(0, np.nan)

    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    return dx.ewm(span=period, adjust=False).mean().fillna(0)


def _vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP del período."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    cum_vol = df["volume"].cumsum()
    cum_tpv = (tp * df["volume"]).cumsum()
    return cum_tpv / cum_vol.replace(0, np.nan)


def _volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    avg = volume.rolling(period).mean()
    return volume / avg.replace(0, np.nan)


def _stoch_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    rsi = _rsi(close, period)
    min_rsi = rsi.rolling(period).min()
    max_rsi = rsi.rolling(period).max()
    denom = (max_rsi - min_rsi).replace(0, np.nan)
    return (rsi - min_rsi) / denom


def _vfi(df: pd.DataFrame, period: int = 130) -> pd.Series:
    """Volume Flow Indicator."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    tp_safe = tp.clip(lower=1e-10)
    inter = np.log(tp_safe) - np.log(tp_safe.shift(1))
    vinter = inter.rolling(30).std().fillna(0.01)
    cutoff = 0.1 * vinter * df["close"]
    vave   = df["volume"].rolling(period).mean().shift(1).fillna(1)
    vmax   = vave * 2.0
    mf = tp - tp.shift(1)
    vcp = np.where(mf > cutoff, df["volume"],
          np.where(mf < -cutoff, -df["volume"], 0.0))
    vf  = pd.Series(vcp, index=df.index).clip(lower=-vmax, upper=vmax)
    return (vf.rolling(period).sum() / vave.replace(0, np.nan)).fillna(0)


def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula todos los indicadores necesarios sobre el OHLCV raw.
    Completamente vectorizado — optimizado para Atom E3950.
    """
    df = df.copy()
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]

    # ── EMAs ──────────────────────────────────────────────────────────────────
    df["ema9"]   = _ema(c, 9)
    df["ema21"]  = _ema(c, 21)
    df["ema55"]  = _ema(c, 55)
    df["ema200"] = _ema(c, 200)

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    df["bb_lower"], df["bb_mid"], df["bb_upper"] = _bollinger(c, 20, 2.0)
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"].replace(0, np.nan)
    # BB Squeeze: width en percentil 20 de los últimos 50 períodos
    df["bb_squeeze"] = df["bb_width"] < df["bb_width"].rolling(50).quantile(0.20)

    # ── RSI & Stoch RSI ───────────────────────────────────────────────────────
    df["rsi"]      = _rsi(c, 14)
    df["rsi_fast"] = _rsi(c, 7)
    df["stoch_rsi"] = _stoch_rsi(c, 14)

    # ── MACD ──────────────────────────────────────────────────────────────────
    df["macd"], df["macd_signal"], df["macd_hist"] = _macd(c)
    df["macd_bull"]    = df["macd"] > df["macd_signal"]
    df["macd_growing"] = df["macd_hist"] > df["macd_hist"].shift(1)
    df["macd_cross_up"] = (df["macd"] > df["macd_signal"]) & (df["macd"].shift(1) <= df["macd_signal"].shift(1))

    # ── ATR y volatilidad ─────────────────────────────────────────────────────
    df["atr"]     = _atr(df, 14)
    df["atr_pct"] = df["atr"] / c * 100
    # ATR percentil para régimen HIGH_VOL
    df["atr_pct_rank"] = df["atr_pct"].rolling(100).rank(pct=True)

    # ── ADX (fuerza de tendencia) ─────────────────────────────────────────────
    df["adx"] = _adx(df, 14)

    # ── Volumen ───────────────────────────────────────────────────────────────
    df["vol_ratio"] = _volume_ratio(v, 20)
    df["vol_spike"] = df["vol_ratio"] > 2.0   # > 200% del promedio
    df["vfi"]       = _vfi(df, 130)
    df["vfi_bull"]  = df["vfi"] > 0

    # ── VWAP (reset diario aproximado) ────────────────────────────────────────
    df["vwap"] = _vwap(df)

    # ── Order Blocks (custom, sin dependencias) ───────────────────────────────
    bearish_candle = c < df["open"]
    bullish_move   = c > c.shift(1)
    df["ob_bull"]    = (bearish_candle.shift(1) & bullish_move).fillna(False)
    df["ob_bull_hi"] = np.where(df["ob_bull"], h.shift(1), np.nan)
    df["ob_bull_lo"] = np.where(df["ob_bull"], l.shift(1), np.nan)

    # ── Fair Value Gaps ───────────────────────────────────────────────────────
    df["fvg_bull"]    = (l > h.shift(2)).fillna(False)
    df["fvg_bull_hi"] = np.where(df["fvg_bull"], l, np.nan)
    df["fvg_bull_lo"] = np.where(df["fvg_bull"], h.shift(2), np.nan)

    # ── Swing Highs/Lows (ventana 5 velas) ───────────────────────────────────
    w = 5
    df["swing_hi"] = (h == h.rolling(w * 2 + 1, center=True).max()).astype(int)
    df["swing_lo"] = (l == l.rolling(w * 2 + 1, center=True).min()).astype(int)

    # ── Trend structure ───────────────────────────────────────────────────────
    df["trend_up"]    = (df["ema21"] > df["ema55"]) & (df["ema55"] > df["ema200"])
    df["trend_down"]  = (df["ema21"] < df["ema55"]) & (df["ema55"] < df["ema200"])
    df["above_vwap"]  = c > df["vwap"]

    # ── Momentum ─────────────────────────────────────────────────────────────
    df["momentum_3"]  = c.pct_change(3) * 100
    df["momentum_10"] = c.pct_change(10) * 100

    # ── Consensus Score (0-100) ───────────────────────────────────────────────
    score = pd.Series(0.0, index=df.index)
    score += df["trend_up"].fillna(False).astype(float) * 25
    rsi_norm = ((df["rsi"].clip(30, 70) - 30) / 40.0)
    score += rsi_norm * 20
    score += df["macd_bull"].fillna(False).astype(float) * 20
    adx_norm = df["adx"].clip(0, 50) / 50.0
    score += adx_norm * 20
    score += df["vfi_bull"].fillna(False).astype(float) * 15
    df["consensus"] = score.fillna(0)

    return df


def detect_regime(df: pd.DataFrame, symbol: str) -> MarketRegime:
    """
    Clasifica el régimen de mercado en base a múltiples indicadores.
    Usa las últimas velas del DataFrame ya enriquecido.

    Lógica:
      HIGH_VOL → ATR percentil > 85 (volatilidad extrema)
      BULL     → trend_up AND adx > 22 AND vfi > 0
      BEAR     → trend_down AND adx > 20
      RANGE    → todo lo demás
    """
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
        strength = min(1.0, (adx - 22) / 28 + 0.4)
        return MarketRegime(symbol, "BULL", strength, adx, atr_pct, True, vfi)

    if trend_dn and adx > 20:
        strength = min(1.0, (adx - 20) / 30 + 0.3)
        return MarketRegime(symbol, "BEAR", strength, adx, atr_pct, False, vfi)

    return MarketRegime(symbol, "RANGE", max(0.3, 1.0 - adx / 40), adx, atr_pct, trend_up, vfi)


# ══════════════════════════════════════════════════════════════════════════════
# ── Las 4 Estrategias ─────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def strategy_trend_following(df: pd.DataFrame, regime: MarketRegime) -> Optional[SignalResult]:
    """
    Trend Following — LONG en tendencias alcistas.

    Condiciones de score acumulativo (no AND puro):
    - EMA stack alcista            → +25 pts
    - ADX > 20                     → +15 pts (>25: +25 pts)
    - RSI 40-70                    → +15 pts
    - MACD alcista                 → +15 pts
    - Retroceso a EMA21 ±3%        → +10 pts  (zona ideal)
    - Order Block o FVG activo     → +10 pts  (confluencia SMC)
    - Volumen > promedio           → +5 pts
    - VFI bullish                  → +5 pts
    Total máximo                   = 100 pts (mínimo para entrar: 55)
    """
    if len(df) < 210:
        return None

    last = df.iloc[-1]
    c    = float(last["close"])

    # ── Score ──────────────────────────────────────────────────────────────
    score   = 0.0
    reasons = []

    # EMA alignment
    if bool(last.get("trend_up", False)):
        score += 25
        reasons.append("EMA_stack_bull")

    # ADX
    adx = float(last.get("adx", 0))
    if adx > 25:
        score += 25
        reasons.append(f"ADX={adx:.0f}>25")
    elif adx > 20:
        score += 15
        reasons.append(f"ADX={adx:.0f}>20")

    # RSI
    rsi = float(last.get("rsi", 50))
    if 40 <= rsi <= 70:
        score += 15
        reasons.append(f"RSI={rsi:.0f}")

    # MACD
    if bool(last.get("macd_bull", False)) and bool(last.get("macd_growing", False)):
        score += 15
        reasons.append("MACD_bull+growing")
    elif bool(last.get("macd_bull", False)):
        score += 8
        reasons.append("MACD_bull")

    # Retroceso a EMA21
    ema21 = float(last.get("ema21", c))
    dist_ema21 = abs(c - ema21) / ema21 * 100
    if dist_ema21 < 1.5:
        score += 10
        reasons.append("pullback_ema21_tight")
    elif dist_ema21 < 3.0:
        score += 6
        reasons.append("pullback_ema21")

    # SMC: Order Block o FVG reciente
    ob_recent = df["ob_bull"].iloc[-5:].any() if "ob_bull" in df.columns else False
    fvg_recent = df["fvg_bull"].iloc[-3:].any() if "fvg_bull" in df.columns else False
    if ob_recent or fvg_recent:
        score += 10
        reasons.append("SMC_confluence")

    # Volumen
    vol_ratio = float(last.get("vol_ratio", 1.0))
    if vol_ratio > 1.2:
        score += 5
        reasons.append(f"vol={vol_ratio:.1f}x")

    # VFI
    if bool(last.get("vfi_bull", False)):
        score += 5
        reasons.append("VFI_bull")

    if score < MIN_SIGNAL_SCORE:
        return None

    # Régimen: solo BULL o final de BEAR recuperación
    if regime.regime not in ("BULL", "RANGE"):
        score *= 0.7

    # ── Niveles ────────────────────────────────────────────────────────────
    atr  = float(last.get("atr", c * 0.015))
    low5 = float(df["low"].iloc[-5:].min())

    # SL: 1.5 ATR bajo mínimo de 5 velas (nunca < 1% del precio)
    sl_atr    = c - 1.5 * atr
    sl_swing  = low5 - 0.5 * atr
    stop_loss = max(sl_atr, sl_swing, c * 0.985)  # nunca SL > 1.5% del precio

    dist_sl = c - stop_loss
    if dist_sl <= 0:
        return None

    # R/R: 2:1, 3.5:1, 5:1 (tres targets)
    tp1 = c + 2.0 * dist_sl
    tp2 = c + 3.5 * dist_sl
    tp3 = c + 5.0 * dist_sl

    # Calidad del setup
    if score >= 85:
        quality = "A+"
    elif score >= 70:
        quality = "A"
    elif score >= 55:
        quality = "B"
    else:
        quality = "C"

    return SignalResult(
        symbol=regime.symbol, strategy="TrendFollowing",
        direction="long", score=score,
        entry_price=c, stop_loss=stop_loss,
        tp1=tp1, tp2=tp2, tp3=tp3,
        atr=atr, regime=regime, quality=quality, reasons=reasons,
    )


def strategy_mean_reversion(df: pd.DataFrame, regime: MarketRegime) -> Optional[SignalResult]:
    """
    Mean Reversion — LONG en mercados laterales cuando precio está en extremo inferior.

    Score acumulativo:
    - ADX < 25 (mercado no en tendencia)  → +20 pts
    - RSI < 35                             → +20 pts (< 28: +30 pts)
    - Precio tocando BB inferior ±1%       → +20 pts
    - Stoch RSI < 0.2                      → +15 pts
    - Divergencia alcista (MACD cruzando)  → +15 pts
    - VFI positivo (volumen comprador)     → +10 pts
    Total máximo                           = 100 pts
    """
    if len(df) < 50:
        return None

    last = df.iloc[-1]
    c    = float(last["close"])

    # Solo en RANGE (máxima efectividad) o inicio de BULL con sobreventido
    if regime.regime == "BEAR" and regime.strength > 0.7:
        return None  # En bear fuerte, mean reversion es muy peligroso

    score   = 0.0
    reasons = []

    # ADX bajo (mercado lateral)
    adx = float(last.get("adx", 30))
    if adx < 20:
        score += 20
        reasons.append(f"ADX={adx:.0f}<20_range")
    elif adx < 25:
        score += 12
        reasons.append(f"ADX={adx:.0f}<25")

    # RSI sobrevendido
    rsi = float(last.get("rsi", 50))
    if rsi < 28:
        score += 30
        reasons.append(f"RSI={rsi:.0f}_oversold_extreme")
    elif rsi < 35:
        score += 20
        reasons.append(f"RSI={rsi:.0f}_oversold")

    # Bollinger inferior
    bb_lower = float(last.get("bb_lower", c))
    bb_mid   = float(last.get("bb_mid", c))
    dist_bb  = (c - bb_lower) / bb_lower * 100 if bb_lower > 0 else 999
    if dist_bb < 0.5:
        score += 20
        reasons.append("at_BB_lower")
    elif dist_bb < 2.0:
        score += 12
        reasons.append("near_BB_lower")

    # Stoch RSI sobrevendido
    stoch = float(last.get("stoch_rsi", 0.5))
    if stoch < 0.15:
        score += 15
        reasons.append(f"StochRSI={stoch:.2f}_extreme")
    elif stoch < 0.25:
        score += 8
        reasons.append(f"StochRSI={stoch:.2f}")

    # MACD divergencia / cruce alcista
    if bool(last.get("macd_cross_up", False)):
        score += 15
        reasons.append("MACD_cross_up_divergence")
    elif bool(last.get("macd_growing", False)) and not bool(last.get("macd_bull", True)):
        score += 8
        reasons.append("MACD_hist_turning")

    # VFI comprador
    if bool(last.get("vfi_bull", False)):
        score += 10
        reasons.append("VFI_bull_at_extreme")

    # MeanReversion usa umbral más exigente que otras estrategias
    if score < MIN_SCORE_MEAN_REVERSION:
        return None

    # ── Niveles ────────────────────────────────────────────────────────────
    atr       = float(last.get("atr", c * 0.015))
    # SL: 2.5 ATR (más holgado para evitar stops prematuros en correcciones)
    stop_loss = max(bb_lower - MR_ATR_STOP_MULTIPLIER * atr, c * 0.965)

    dist_sl = c - stop_loss
    if dist_sl <= 0:
        return None

    # MR: TP1 = EMA21, TP2 = BB mid, TP3 = BB upper
    ema21 = float(last.get("ema21", c * 1.02))
    tp1   = max(c + 1.2 * dist_sl, ema21)   # al menos 1.2:1
    tp2   = bb_mid
    tp3   = float(last.get("bb_upper", c + 3 * dist_sl))

    # Asegurar R/R mínimo 1.2:1
    if (tp1 - c) < 1.2 * dist_sl:
        tp1 = c + 1.2 * dist_sl

    quality = "A+" if score >= 80 else ("A" if score >= 65 else ("B" if score >= 50 else "C"))

    return SignalResult(
        symbol=regime.symbol, strategy="MeanReversion",
        direction="long", score=score,
        entry_price=c, stop_loss=stop_loss,
        tp1=tp1, tp2=tp2, tp3=tp3,
        atr=atr, regime=regime, quality=quality, reasons=reasons,
    )


def strategy_breakout(df: pd.DataFrame, regime: MarketRegime) -> Optional[SignalResult]:
    """
    Breakout — LONG en rotura de compresión con volumen confirmado.

    Score acumulativo:
    - BB Squeeze activo > 10 velas         → +25 pts
    - Cierre sobre BB superior             → +20 pts
    - Volumen > 150% del promedio          → +20 pts
    - Precio sobre máximo de 20 velas      → +15 pts
    - MACD positivo                        → +10 pts
    - Retest del nivel roto (si aplica)    → +10 pts
    Total máximo                           = 100 pts
    """
    if len(df) < 55:
        return None

    last = df.iloc[-1]
    c    = float(last["close"])

    score   = 0.0
    reasons = []

    # BB Squeeze: cuántas velas consecutivas en compresión
    squeeze_col = df.get("bb_squeeze", pd.Series(False, index=df.index)).fillna(False)
    if isinstance(squeeze_col, pd.Series):
        squeeze_count = int(squeeze_col.iloc[-20:].sum())
    else:
        squeeze_count = 0

    if squeeze_count >= 10:
        score += 25
        reasons.append(f"BB_squeeze_{squeeze_count}velas")
    elif squeeze_count >= 5:
        score += 12
        reasons.append(f"BB_squeeze_{squeeze_count}velas")

    # Rotura sobre BB superior
    bb_upper   = float(last.get("bb_upper", c))
    prev_upper = float(df["bb_upper"].iloc[-2]) if "bb_upper" in df.columns and len(df) > 1 else bb_upper
    if c > bb_upper:
        score += 20
        reasons.append("close_above_BB_upper")
    elif c > prev_upper * 0.999:
        score += 12
        reasons.append("testing_BB_upper")

    # Volumen
    vol_ratio = float(last.get("vol_ratio", 1.0))
    if vol_ratio > 2.0:
        score += 20
        reasons.append(f"vol_spike_{vol_ratio:.1f}x")
    elif vol_ratio > 1.5:
        score += 12
        reasons.append(f"vol_elevated_{vol_ratio:.1f}x")

    # Rotura de máximo de 20 velas
    high20 = float(df["high"].iloc[-21:-1].max()) if len(df) > 21 else c
    if c > high20:
        score += 15
        reasons.append("breakout_20period_high")

    # MACD confirma
    if bool(last.get("macd_bull", False)):
        score += 10
        reasons.append("MACD_confirming")

    # Retest del nivel (precio volvió al nivel roto y rebotó)
    if len(df) > 3:
        prev_low = float(df["low"].iloc[-3:-1].min())
        if bb_upper * 0.995 < prev_low < bb_upper * 1.005:
            score += 10
            reasons.append("retest_confirmed")

    if score < MIN_SIGNAL_SCORE:
        return None

    # ── Niveles ────────────────────────────────────────────────────────────
    atr = float(last.get("atr", c * 0.015))

    # SL: debajo del nivel de rotura (BB upper anterior)
    stop_loss = max(prev_upper - 1.5 * atr, c * 0.975)
    dist_sl   = c - stop_loss
    if dist_sl <= 0:
        return None

    # Breakouts pueden tener targets amplios
    tp1 = c + 2.0 * dist_sl
    tp2 = c + 4.0 * dist_sl
    tp3 = c + 6.0 * dist_sl

    quality = "A+" if score >= 85 else ("A" if score >= 70 else ("B" if score >= 55 else "C"))

    return SignalResult(
        symbol=regime.symbol, strategy="Breakout",
        direction="long", score=score,
        entry_price=c, stop_loss=stop_loss,
        tp1=tp1, tp2=tp2, tp3=tp3,
        atr=atr, regime=regime, quality=quality, reasons=reasons,
    )


def strategy_momentum_scalp(df: pd.DataFrame, regime: MarketRegime) -> Optional[SignalResult]:
    """
    Momentum Scalp — NUEVA estrategia para mercados con impulso claro.
    Optimizada para $1000: capitaliza movimientos de 0.5-1.5% rápidos.

    Lógica: precio sobre VWAP + RSI acelerando desde zona neutral (45-55 → 60+)
    con volumen confirmando. Targets más ajustados, pero señales más frecuentes.

    Score:
    - RSI acelerando (lag1 > 45 y actual > 55)  → +25 pts
    - Precio sobre VWAP                          → +20 pts
    - EMA9 cruzando EMA21 al alza                → +20 pts
    - Volumen creciente 3 velas                  → +15 pts
    - MACD hist positivo y creciendo             → +20 pts
    Total máximo                                 = 100 pts

    Solo activa en BULL y RANGE (no en BEAR ni HIGH_VOL extremo).
    """
    if len(df) < 30:
        return None

    if regime.regime in ("BEAR",) and regime.strength > 0.6:
        return None

    if regime.atr_pct > 5.0:
        return None  # Demasiada volatilidad para scalp

    last    = df.iloc[-1]
    prev    = df.iloc[-2]
    c       = float(last["close"])

    score   = 0.0
    reasons = []

    # RSI acelerando desde neutral
    rsi_now  = float(last.get("rsi", 50))
    rsi_prev = float(prev.get("rsi", 50)) if len(df) > 1 else rsi_now
    rsi_fast = float(last.get("rsi_fast", 50))

    if rsi_prev < 52 and rsi_now > 58 and rsi_fast > 60:
        score += 25
        reasons.append(f"RSI_accel_{rsi_prev:.0f}→{rsi_now:.0f}")
    elif rsi_now > 55 and rsi_now > rsi_prev + 3:
        score += 15
        reasons.append(f"RSI_momentum_{rsi_now:.0f}")

    # Precio sobre VWAP
    vwap = float(last.get("vwap", c))
    if c > vwap * 1.001:
        score += 20
        reasons.append("above_VWAP")

    # EMA9 cruzando EMA21 (corto plazo)
    ema9_now  = float(last.get("ema9", c))
    ema21_now = float(last.get("ema21", c))
    ema9_prev = float(prev.get("ema9", c)) if len(df) > 1 else ema9_now
    ema21_prev = float(prev.get("ema21", c)) if len(df) > 1 else ema21_now

    if ema9_prev <= ema21_prev and ema9_now > ema21_now:
        score += 20
        reasons.append("EMA9_cross_EMA21")
    elif ema9_now > ema21_now:
        score += 10
        reasons.append("EMA9_above_EMA21")

    # Volumen creciente (3 últimas velas)
    vol3 = df["volume"].iloc[-3:]
    if len(vol3) == 3 and vol3.is_monotonic_increasing:
        score += 15
        reasons.append("vol_3bar_increasing")
    elif float(last.get("vol_ratio", 1.0)) > 1.3:
        score += 8
        reasons.append("vol_above_avg")

    # MACD hist positivo y creciendo
    macd_hist = float(last.get("macd_hist", 0))
    if macd_hist > 0 and bool(last.get("macd_growing", False)):
        score += 20
        reasons.append("MACD_hist_bull_growing")
    elif macd_hist > 0:
        score += 10
        reasons.append("MACD_hist_positive")

    if score < MIN_SIGNAL_SCORE:
        return None

    # ── Niveles ajustados (targets más cercanos) ───────────────────────────
    atr  = float(last.get("atr", c * 0.015))

    # SL más ajustado: 1 ATR (scalp — no queremos arriesgar más)
    stop_loss = max(c - 1.0 * atr, c * 0.988)
    dist_sl   = c - stop_loss
    if dist_sl <= 0:
        return None

    # Targets conservadores 1.5:1, 2.5:1, 4:1
    tp1 = c + 1.5 * dist_sl
    tp2 = c + 2.5 * dist_sl
    tp3 = c + 4.0 * dist_sl

    quality = "A+" if score >= 85 else ("A" if score >= 70 else "B")

    return SignalResult(
        symbol=regime.symbol, strategy="MomentumScalp",
        direction="long", score=score,
        entry_price=c, stop_loss=stop_loss,
        tp1=tp1, tp2=tp2, tp3=tp3,
        atr=atr, regime=regime, quality=quality, reasons=reasons,
    )


def select_best_signal(signals: List[SignalResult]) -> Optional[SignalResult]:
    """
    De todas las señales válidas para un símbolo, selecciona la mejor.

    Criterios:
    1. Primero filtra por score >= MIN_SIGNAL_SCORE
    2. Si hay múltiples, prioriza según régimen:
       - BULL:     TrendFollowing > MomentumScalp > Breakout > MeanReversion
       - RANGE:    MeanReversion  > MomentumScalp > Breakout > TrendFollowing
       - BEAR:     MeanReversion solo si score > 70
       - HIGH_VOL: Solo si score > 75 y es TrendFollowing/MeanReversion
    3. Desempate: score más alto
    """
    if not signals:
        return None

    valid = [s for s in signals if s is not None and s.score >= MIN_SIGNAL_SCORE]
    if not valid:
        return None

    regime = valid[0].regime.regime

    # Pesos por régimen y estrategia
    PRIORITY = {
        "BULL":     {"TrendFollowing": 4, "MomentumScalp": 3, "Breakout": 2, "MeanReversion": 1},
        "RANGE":    {"MeanReversion": 4,  "MomentumScalp": 3, "Breakout": 2, "TrendFollowing": 1},
        "BEAR":     {"MeanReversion": 3,  "TrendFollowing": 1, "Breakout": 1, "MomentumScalp": 1},
        "HIGH_VOL": {"TrendFollowing": 2, "MeanReversion": 2, "Breakout": 1, "MomentumScalp": 1},
    }

    priority_map = PRIORITY.get(regime, PRIORITY["RANGE"])

    # Filtros extra por régimen
    if regime == "BEAR":
        valid = [s for s in valid if s.score >= 65]
    if regime == "HIGH_VOL":
        valid = [s for s in valid if s.score >= 70]

    if not valid:
        return None

    # Ordenar: prioridad de estrategia × score
    best = max(valid, key=lambda s: priority_map.get(s.strategy, 1) * 10 + s.score)
    return best


# ══════════════════════════════════════════════════════════════════════════════
# ── Gestión de Riesgo y Sizing ────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def compute_position_size(
    signal:    SignalResult,
    capital:   float,
    open_count: int,
    daily_pnl_pct: float,
) -> Tuple[float, float, float]:
    """
    Calcula el tamaño de posición con Kelly fraccionado + límites de riesgo.

    Returns:
        (units, risk_amount_usd, notional_usd)
        - units:          unidades del activo a comprar
        - risk_amount_usd: capital en riesgo (distancia al SL × units)
        - notional_usd:   capital total invertido (units × entry_price)
    """
    # ── Riesgo base por capital ────────────────────────────────────────────
    if capital < 1500:
        risk_usd = RISK_PER_TRADE_USD  # $10 fijo en fase inicial
    elif capital < 5000:
        risk_usd = capital * RISK_PCT_TIER_1
    elif capital < 20_000:
        risk_usd = capital * RISK_PCT_TIER_2
    else:
        risk_usd = capital * RISK_PCT_TIER_3

    # Nunca superar el límite absoluto
    risk_usd = min(risk_usd, capital * MAX_RISK_PER_TRADE)

    # ── Ajuste por régimen ─────────────────────────────────────────────────
    risk_usd *= signal.regime.risk_multiplier

    # ── Reducción por circuit breaker parcial ─────────────────────────────
    if daily_pnl_pct < CB_DAILY_REDUCE:
        risk_usd *= 0.5

    # ── Reducción por calidad del setup ───────────────────────────────────
    quality_mult = {"A+": 1.0, "A": 0.8, "B": 0.6, "C": 0.4}
    risk_usd *= quality_mult.get(signal.quality, 0.6)

    # ── Kelly fraccionado (estimación win rate histórica ~50% para crypto) ─
    # f* = (p*b - q) / b  donde b = R/R, p = win rate estimado
    estimated_win_rate = 0.50 + signal.ml_proba * 0.10  # ajustado por ML
    b = (signal.tp1 - signal.entry_price) / (signal.entry_price - signal.stop_loss)  # R/R
    q = 1 - estimated_win_rate
    kelly_full = (estimated_win_rate * b - q) / b if b > 0 else 0
    kelly_fraction_used = max(0, kelly_full) * KELLY_FRACTION

    if kelly_fraction_used > 0:
        kelly_risk = capital * kelly_fraction_used
        # Usar el mínimo de Kelly y el riesgo fijo (conservador)
        risk_usd = min(risk_usd, kelly_risk)

    # Mínimo $3 (no entrar por menos)
    risk_usd = max(risk_usd, 3.0)

    # ── Unidades ───────────────────────────────────────────────────────────
    dist_sl = signal.entry_price - signal.stop_loss
    if dist_sl <= 0:
        return 0.0, 0.0, 0.0

    units = risk_usd / dist_sl

    # Capping nocional: no más de 35% del capital en una posición
    max_notional = capital * 0.35
    if units * signal.entry_price > max_notional:
        units = max_notional / signal.entry_price

    # Validar mínimo Binance (~$11)
    notional_usd = units * signal.entry_price
    if notional_usd < 11.0:
        return 0.0, 0.0, 0.0

    return units, risk_usd, notional_usd


# ══════════════════════════════════════════════════════════════════════════════
# ── Motor Principal ───────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

class LiveEngine:
    """
    Motor de trading V4.0 — multi-estrategia, multi-timeframe, adaptativo.

    Arquitectura:
    - _loop_slow: se dispara en cada cierre de vela 1h → análisis + señales
    - _loop_entry: se dispara en cada cierre de vela 15m → entrada precisa
    - _loop_fast: cada 60s → gestión posiciones, trailing stop, heartbeat
    """

    def __init__(self, paper_mode: bool = True) -> None:
        self.paper_mode = paper_mode
        self._running   = False
        self._tasks: List[asyncio.Task] = []

        # ── Estado global ─────────────────────────────────────────────────
        self.state: Dict[str, Any] = {
            "paused":         False,
            "kill":           False,
            "paper_mode":     paper_mode,
            "capital":        INITIAL_CAPITAL,
            "open_positions": [],
            "pnl_today":      0.0,
            "pnl_today_pct":  0.0,
            "drawdown_pct":   0.0,
            "ml_ready":       False,
            "regimes":        {},   # {symbol: MarketRegime}
            "last_signals":   {},   # {symbol: SignalResult}
        }

        # ── Cooldowns por símbolo ──────────────────────────────────────────
        self._cooldowns: Dict[str, SymbolCooldown] = {
            sym: SymbolCooldown() for sym in SYMBOLS
        }

        # ── Cache de DataFrames enriquecidos ──────────────────────────────
        self._df_cache: Dict[str, pd.DataFrame] = {}

        # ── Señales pendientes de timing (esperando vela 15m) ─────────────
        self._pending_signals: Dict[str, SignalResult] = {}

        # ── Circuit breaker de red ────────────────────────────────────────
        self._recent_errors: List[float] = []

        # ── Componentes externos ──────────────────────────────────────────
        self._exchange:  Optional[ccxt.Exchange]  = None
        self._portfolio: Optional[PaperPortfolio] = None
        self._bot:       Optional[TelegramBot]    = None
        self._ml:        Optional[MetaLabeler]    = None

    # ══════════════════════════════════════════════════════════════════════════
    # Arranque y Parada
    # ══════════════════════════════════════════════════════════════════════════

    async def start(self) -> None:
        log.info("engine_starting", version=ENGINE_VERSION, paper=self.paper_mode,
                 capital=f"${INITIAL_CAPITAL:.0f}", symbols=SYMBOLS)

        self._running = True

        # Exchange
        self._exchange = ccxt.binance({
            "apiKey":          os.environ.get("BINANCE_API_KEY", ""),
            "secret":          os.environ.get("BINANCE_API_SECRET", ""),
            "options":         {"defaultType": "spot"},
            "enableRateLimit": True,
        })
        await self._ping_exchange()

        if not self.paper_mode:
            if not await self._validate_api_keys():
                raise RuntimeError("API keys inválidas o permisos de retiro activos.")

        # Portfolio
        raw_db = os.environ.get("DATABASE_URL", "")
        if not raw_db:
            raise RuntimeError("DATABASE_URL no configurada en .env")
        async_url = (raw_db
                     .replace("postgresql://",   "postgresql+asyncpg://")
                     .replace("postgres://",      "postgresql+asyncpg://"))

        self._portfolio = PaperPortfolio(
            initial_capital=INITIAL_CAPITAL,
            db_url=async_url,
        )
        await self._portfolio.initialize()
        await self._sync_state()

        # ML MetaLabeler
        self._ml = MetaLabeler(model_path=str(PROJECT_ROOT / "ml" / "model.joblib"))
        self.state["ml_ready"] = self._ml.is_ready()
        log.info("ml_status", ready=self.state["ml_ready"])

        # Telegram
        token   = os.environ.get("TELEGRAM_TOKEN", "")
        chat_id = int(os.environ.get("TELEGRAM_ALLOWED_USER_ID", "0") or "0")
        if token and chat_id:
            self._bot = TelegramBot(token=token, allowed_chat_id=chat_id)
            await self._bot.start()
            await self._bot.send_startup(
                paper_mode=self.paper_mode,
                capital=self.state["capital"],
                n_positions=len(self.state["open_positions"]),
            )

        # Señal SIGTERM
        asyncio.get_event_loop().add_signal_handler(
            signal.SIGTERM,
            lambda: asyncio.create_task(self.shutdown("SIGTERM")),
        )

        log.info("engine_live", mode="paper" if self.paper_mode else "LIVE",
                 symbols=len(SYMBOLS), strategies=4)

        t_slow  = asyncio.create_task(self._loop_slow(),  name="loop_slow")
        t_entry = asyncio.create_task(self._loop_entry(), name="loop_entry")
        t_fast  = asyncio.create_task(self._loop_fast(),  name="loop_fast")
        self._tasks = [t_slow, t_entry, t_fast]

        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            log.exception("engine_fatal", error=str(exc))
            if self._bot:
                await self._bot.send_alert(f"💀 Error fatal:\n```{exc}```", level="critical")
            raise

    async def shutdown(self, reason: str = "manual") -> None:
        log.warning("shutdown", reason=reason)
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

    # ══════════════════════════════════════════════════════════════════════════
    # Loop Lento — Análisis 1H (cierre de vela)
    # ══════════════════════════════════════════════════════════════════════════

    async def _loop_slow(self) -> None:
        """
        Se ejecuta en cada cierre de vela 1H.
        Función: analizar mercado, detectar régimen, generar señales estructurales.
        Las señales se almacenan en _pending_signals para timing en _loop_entry.
        """
        while self._running:
            wait = _seconds_to_next_close(TF_STRUCT_SECS)
            await asyncio.sleep(wait + CANDLE_BUFFER)

            if self.state["kill"] or self.state["paused"]:
                continue

            log.debug("analysis_cycle_start", n_symbols=len(SYMBOLS))

            for symbol in SYMBOLS:
                try:
                    await self._analyze_symbol(symbol)
                except Exception as exc:
                    log.exception("analysis_error", symbol=symbol, error=str(exc))

    async def _analyze_symbol(self, symbol: str) -> None:
        """Descarga datos 1H, enriquece, detecta régimen y genera señales."""
        # Descargar velas 1H (500 velas ≈ 20 días)
        df = await self._fetch_candles(symbol, TF_STRUCTURE, limit=500)
        if df is None or len(df) < 220:
            return

        try:
            df = enrich_dataframe(df)
        except Exception as exc:
            log.warning("enrich_error", symbol=symbol, error=str(exc))
            return

        regime = detect_regime(df, symbol)
        self.state["regimes"][symbol] = regime

        # No analizar si volatilidad extrema
        if not regime.is_tradeable:
            log.info("symbol_skipped_high_vol", symbol=symbol, atr_pct=regime.atr_pct)
            return

        self._df_cache[symbol] = df

        # Evaluar las 4 estrategias
        candidates: List[Optional[SignalResult]] = []

        if regime.regime in ("BULL", "RANGE", "HIGH_VOL"):
            candidates.append(strategy_trend_following(df, regime))
            candidates.append(strategy_mean_reversion(df, regime))
            candidates.append(strategy_breakout(df, regime))
        elif regime.regime == "BEAR":
            # En bear: solo MR de alta convicción o TF si hay señal muy fuerte
            candidates.append(strategy_mean_reversion(df, regime))
            candidates.append(strategy_trend_following(df, regime))

        valid = [s for s in candidates if s is not None]

        # ML scoring para cada señal válida
        if self.state["ml_ready"] and self._ml:
            for sig in valid:
                try:
                    sig.ml_proba = self._ml.predict_proba(df)
                except Exception:
                    sig.ml_proba = 0.5

        best = select_best_signal(valid)

        if best:
            log.info(
                "signal_queued",
                symbol=symbol, strategy=best.strategy,
                score=f"{best.score:.0f}", quality=best.quality,
                regime=regime.regime, reasons=",".join(best.reasons[:3]),
            )
            self._pending_signals[symbol] = best
        else:
            # Limpiar señal anterior si no hay nueva
            self._pending_signals.pop(symbol, None)

    # ══════════════════════════════════════════════════════════════════════════
    # Loop de Entrada — Timing 15M
    # ══════════════════════════════════════════════════════════════════════════

    async def _loop_entry(self) -> None:
        """
        Se ejecuta en cada cierre de vela 15M.
        Función: si hay señales pendientes del loop 1H, refinar el timing
        con velas 15m y ejecutar la entrada si el contexto lo confirma.
        """
        while self._running:
            wait = _seconds_to_next_close(TF_ENTRY_SECS)
            await asyncio.sleep(wait + 5)

            if self.state["kill"] or self.state["paused"]:
                continue

            if not self._pending_signals:
                continue

            for symbol, signal in list(self._pending_signals.items()):
                try:
                    await self._attempt_entry(symbol, signal)
                except Exception as exc:
                    log.exception("entry_error", symbol=symbol, error=str(exc))

    async def _attempt_entry(self, symbol: str, signal: SignalResult) -> None:
        """
        Verifica condiciones de entrada en 15m y ejecuta si todo está en orden.
        """
        # ── Guards pre-entrada ─────────────────────────────────────────────
        open_positions = await self._portfolio.get_open_positions()
        open_symbols   = {p["symbol"] for p in open_positions}

        if symbol in open_symbols:
            self._pending_signals.pop(symbol, None)
            return  # anti-piramidación

        if len(open_positions) >= MAX_POSITIONS:
            log.debug("max_positions_reached", n=len(open_positions))
            return

        # ── Filtro de correlación de cartera ───────────────────────────────
        # Evita abrir múltiples altcoins correlacionadas al mismo tiempo
        # (el evento del 29 mayo: 3 stops simultáneos en AVAX+LINK+BTC)
        my_group = CORRELATION_GROUPS.get(symbol, "other")
        if my_group != "btc":  # BTC es menos correlado, solo filtrar altcoins
            group_count = sum(
                1 for p in open_positions
                if CORRELATION_GROUPS.get(p.get("symbol", ""), "other") == my_group
            )
            if group_count >= MAX_CORRELATED_POSITIONS:
                log.info(
                    "correlation_filter_skip",
                    symbol=symbol,
                    group=my_group,
                    open_in_group=group_count,
                    max=MAX_CORRELATED_POSITIONS,
                )
                return

        # ── Filtro horario para MeanReversion ─────────────────────────────
        # MR funciona mejor en sesión asiática (baja volatilidad)
        # En London/NY los movimientos son más direccionales y rompen la reversión
        if "MeanReversion" in signal.strategy:
            current_hour_utc = datetime.now(timezone.utc).hour
            if current_hour_utc not in MR_ALLOWED_HOURS_UTC:
                log.info(
                    "mr_session_filter_skip",
                    symbol=symbol,
                    hour_utc=current_hour_utc,
                    allowed=f"00-07 UTC",
                )
                return

        # Cooldown por símbolo
        cooldown = self._cooldowns.get(symbol, SymbolCooldown())
        elapsed  = time.time() - cooldown.last_sl_time
        if elapsed < COOLDOWN_AFTER_SL_MIN * 60:
            remaining = int((COOLDOWN_AFTER_SL_MIN * 60 - elapsed) / 60)
            log.debug("symbol_in_cooldown", symbol=symbol, remaining_min=remaining)
            return

        # Circuit breaker diario
        capital   = await self._portfolio.get_current_capital()
        daily     = await self._portfolio.get_daily_stats()
        pnl_today = float(daily.get("pnl_today", 0.0))
        pnl_pct   = pnl_today / capital if capital > 0 else 0.0

        if pnl_pct <= CB_DAILY_PAUSE:
            log.warning("cb_daily_pause", pnl_pct=f"{pnl_pct:.2%}")
            return

        # Circuit breaker de pico
        ps_df = await self._get_portfolio_state()
        if ps_df:
            peak = float(ps_df.get("peak_capital", capital))
            dd_from_peak = (capital - peak) / peak if peak > 0 else 0
            if dd_from_peak <= CB_PEAK_SHUTDOWN:
                log.error("cb_peak_shutdown", dd=f"{dd_from_peak:.2%}")
                if self._bot:
                    await self._bot.send_alert(
                        f"🚨 Drawdown {dd_from_peak:.2%} sobre pico — Motor detenido", "critical"
                    )
                await self.shutdown("peak_drawdown_limit")
                return

        # ── Confirmación en 15m ────────────────────────────────────────────
        df_15m = await self._fetch_candles(symbol, TF_ENTRY, limit=100)
        if df_15m is None or len(df_15m) < 30:
            return

        try:
            df_15m = enrich_dataframe(df_15m)
        except Exception:
            return

        last_15m = df_15m.iloc[-1]
        c_15m    = float(last_15m["close"])

        # Confirmar que el precio 15m no se ha alejado demasiado del análisis 1H
        price_drift = abs(c_15m - signal.entry_price) / signal.entry_price * 100
        if price_drift > 1.5:
            log.debug("signal_stale_price_drift", symbol=symbol, drift=f"{price_drift:.1f}%")
            self._pending_signals.pop(symbol, None)
            return

        # Micro-confirmación 15m: al menos 2 de 3 condiciones alcistas en 15m
        micro_ok = 0
        if float(last_15m.get("rsi", 50)) > 50:
            micro_ok += 1
        if bool(last_15m.get("macd_bull", False)):
            micro_ok += 1
        if bool(last_15m.get("above_vwap", False)):
            micro_ok += 1

        # Solo requiero 1/3 — no queremos un filtro demasiado duro
        if micro_ok == 0:
            log.debug("micro_confirmation_failed", symbol=symbol, score=micro_ok)
            return

        # ── Sizing ─────────────────────────────────────────────────────────
        # Actualizar entry_price al precio actual 15m (más preciso)
        signal.entry_price = c_15m

        units, risk_usd, notional_usd = compute_position_size(
            signal, capital, len(open_positions), pnl_pct
        )
        if units <= 0:
            log.warning("position_size_zero", symbol=symbol)
            return

        # Calcular % del capital que representa esta posición
        capital_pct = (notional_usd / capital * 100) if capital > 0 else 0

        # ── Ejecutar entrada ───────────────────────────────────────────────
        log.info(
            "trade_opening",
            symbol=symbol, strategy=signal.strategy,
            score=f"{signal.score:.0f}", quality=signal.quality,
            entry=f"${c_15m:.4f}", sl=f"${signal.stop_loss:.4f}",
            tp1=f"${signal.tp1:.4f}",
            # Capital tracking — visible en logs y dashboard
            risk_usd=f"${risk_usd:.2f}",
            notional_usd=f"${notional_usd:.2f}",
            capital_pct=f"{capital_pct:.1f}%",
            capital_total=f"${capital:.2f}",
            regime=signal.regime.regime, ml=f"{signal.ml_proba:.0%}",
            reasons=",".join(signal.reasons[:3]),
        )

        trade = await self._portfolio.open_position(
            symbol=symbol,
            strategy=f"{signal.strategy}_{signal.regime.regime}",
            entry_price=c_15m,
            stop_loss=signal.stop_loss,
            tp1=signal.tp1,
            tp2=signal.tp2,
            units=units,
            ml_proba=signal.ml_proba,
            direction=signal.direction,
            regime=signal.regime.regime,
            risk_amount=risk_usd,
            notional_usd=notional_usd,
        )

        if self._bot:
            await self._bot.send_trade_open(trade)

        # Registrar en journal con calidad, razones y capital invertido
        await self._log_trade_extended(
            trade, signal,
            notional_usd=notional_usd,
            capital_at_entry=capital,
        )

        # Limpiar señal usada
        self._pending_signals.pop(symbol, None)
        self.state["last_signals"][symbol] = signal

    # ══════════════════════════════════════════════════════════════════════════
    # Loop Rápido — Gestión de Posiciones (60s)
    # ══════════════════════════════════════════════════════════════════════════

    async def _loop_fast(self) -> None:
        """
        Cada 60 segundos:
        - Actualiza precios y gestiona posiciones (SL, TP, trailing)
        - Heartbeat a PostgreSQL
        - Resumen diario a las 23:55 UTC
        - Procesa comandos Telegram
        """
        last_daily_date = datetime.now(tz=timezone.utc).date()

        while self._running:
            await asyncio.sleep(FAST_LOOP_SECS)

            # Kill switch
            if self.state["kill"]:
                await self._execute_kill_switch()
                break

            try:
                prices = await self._get_current_prices()
            except (ccxt.NetworkError, ccxt.RequestTimeout) as exc:
                await self._handle_network_error(exc, "get_prices")
                continue

            # Actualizar posiciones con trailing stop dinámico
            closed_trades = await self._portfolio.update_positions(prices)
            for ct in closed_trades:
                reason = ct.get("exit_reason", "unknown")
                if self._bot:
                    await self._bot.send_trade_close(ct, reason)

                # Registrar cooldown si fue SL
                if "stop" in reason.lower() or "sl" in reason.lower():
                    sym = ct.get("symbol", "")
                    if sym in self._cooldowns:
                        self._cooldowns[sym].last_sl_time = time.time()
                        self._cooldowns[sym].sl_count_24h += 1
                        log.info("cooldown_set", symbol=sym,
                                 min=COOLDOWN_AFTER_SL_MIN)

            # Sincronizar estado
            capital  = await self._portfolio.get_current_capital()
            daily    = await self._portfolio.get_daily_stats()
            pnl_pct  = float(daily.get("pnl_today", 0.0)) / capital if capital > 0 else 0

            self.state.update({
                "capital":        capital,
                "pnl_today":      float(daily.get("pnl_today", 0.0)),
                "pnl_today_pct":  pnl_pct,
                "open_positions": await self._portfolio.get_open_positions(),
            })

            await self._update_heartbeat()

            if self._bot:
                await self._bot.process_updates(self.state)

            # Resumen diario
            now = datetime.now(tz=timezone.utc)
            if now.hour == 23 and now.minute >= 55:
                today = now.date()
                if today != last_daily_date:
                    last_daily_date = today
                    if self._bot:
                        await self._bot.send_daily_pnl({
                            **daily, "capital": capital,
                            "drawdown_pct": pnl_pct,
                            "ml_ready": self.state["ml_ready"],
                        })

            await self._reset_daily_trackers_if_needed()

    # ══════════════════════════════════════════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════════════════════════════════════════

    async def _get_portfolio_state(self) -> Optional[Dict]:
        if not self._portfolio or not self._portfolio._engine:
            return None
        try:
            async with self._portfolio._session_factory() as sess:
                r = await sess.execute(text("SELECT * FROM portfolio_state WHERE id = 1"))
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
            async with self._portfolio._session_factory() as sess:
                await sess.execute(
                    text("UPDATE portfolio_state SET current_capital=:c, updated_at=NOW() WHERE id=1"),
                    {"c": self.state["capital"]},
                )
                await sess.commit()
        except Exception:
            pass

    async def _update_heartbeat(self) -> None:
        if not self._portfolio:
            return
        try:
            async with self._portfolio._session_factory() as sess:
                await sess.execute(text("""
                    INSERT INTO system_heartbeat (id, last_ping, engine_version, paper_mode)
                    VALUES (1, NOW(), :v, :p)
                    ON CONFLICT (id) DO UPDATE SET last_ping = NOW()
                """), {"v": ENGINE_VERSION, "p": self.paper_mode})
                await sess.commit()
        except Exception:
            pass

    async def _log_trade_extended(
        self,
        trade:       Dict,
        signal:      SignalResult,
        notional_usd: float = 0.0,
        capital_at_entry: float = 0.0,
    ) -> None:
        """Registra el trade en trades_journal con campos extendidos de V4."""
        if not self._portfolio or not self._portfolio._engine:
            return
        try:
            async with self._portfolio._session_factory() as sess:
                capital_pct_entry = (
                    notional_usd / capital_at_entry * 100
                    if capital_at_entry > 0 else 0
                )
                await sess.execute(text("""
                    INSERT INTO trades_journal
                        (trade_id, strategy, symbol, timeframe, direction,
                         setup_quality, entry_price, stop_loss,
                         tp1, tp2, take_profit_1,
                         units, position_size,
                         risk_amount, entry_reason, market_regime, regime,
                         ml_proba, entry_time, is_backtest,
                         observations)
                    VALUES
                        (:tid, :strat, :sym, :tf, :dir,
                         :qual, :ep, :sl,
                         :tp1, :tp2, :tp1,
                         :size, :size,
                         :risk, :reason, :regime, :regime,
                         :ml, NOW(), FALSE,
                         :obs)
                    ON CONFLICT DO NOTHING
                """), {
                    "tid":    trade.get("id", 0),
                    "strat":  signal.strategy,
                    "sym":    signal.symbol,
                    "tf":     TF_ENTRY,
                    "dir":    signal.direction,
                    "qual":   ord(signal.quality[0]) - 64 if signal.quality else 0,
                    "ep":     signal.entry_price,
                    "sl":     signal.stop_loss,
                    "tp1":    signal.tp1,
                    "tp2":    signal.tp2,
                    "size":   trade.get("units", 0),
                    "risk":   trade.get("risk_amount", 0),
                    "reason": " | ".join(signal.reasons[:5]),
                    "regime": signal.regime.regime,
                    "ml":     signal.ml_proba,
                    "obs":    (
                        f"notional=${notional_usd:.2f} "
                        f"capital_pct={capital_pct_entry:.1f}% "
                        f"capital_at_entry=${capital_at_entry:.2f} "
                        f"score={signal.score:.0f} "
                        f"quality={signal.quality}"
                    ),
                })
                await sess.commit()
        except Exception as exc:
            log.debug("journal_write_skip", error=str(exc))

    async def _execute_kill_switch(self) -> None:
        log.error("kill_switch_activated")
        try:
            prices = await self._get_current_prices()
        except Exception:
            prices = {}
        closed = await self._portfolio.emergency_close_all(prices)
        for ct in closed:
            if self._bot:
                await self._bot.send_trade_close(ct, "kill_switch")
        await self.shutdown("kill_switch")

    async def _reset_daily_trackers_if_needed(self) -> None:
        now = datetime.now(tz=timezone.utc)
        if now.hour == 0 and now.minute < 2 and self._portfolio:
            capital = await self._portfolio.get_current_capital()
            try:
                async with self._portfolio._session_factory() as sess:
                    await sess.execute(
                        text("UPDATE portfolio_state SET daily_start=:c WHERE id=1"),
                        {"c": capital},
                    )
                    await sess.commit()
            except Exception:
                pass

            # Resetear contadores 24h de cooldowns
            for cooldown in self._cooldowns.values():
                cooldown.sl_count_24h = 0
                cooldown.last_sl_reset = time.time()

    async def _ping_exchange(self) -> None:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                await self._exchange.fetch_time()
                log.info("exchange_ok")
                return
            except ccxt.NetworkError:
                await asyncio.sleep(min(2 ** attempt, 60))
        raise RuntimeError("No se pudo conectar a Binance tras varios intentos.")

    async def _validate_api_keys(self) -> bool:
        try:
            perms = await self._exchange.fetch_api_key_permissions()
            if perms.get("enableWithdrawals", False):
                log.error("api_keys_have_withdrawal_permission — RIESGO CRÍTICO")
                return False
            return perms.get("enableSpotAndMarginTrading", False)
        except Exception:
            return False

    async def _fetch_candles(
        self, symbol: str, timeframe: str, limit: int = 500
    ) -> Optional[pd.DataFrame]:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                ohlcv = await self._exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                df    = pd.DataFrame(
                    ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                return df.dropna()
            except ccxt.NetworkError:
                await asyncio.sleep(min(2 ** attempt, 60))
            except Exception as exc:
                log.warning("fetch_candles_error", symbol=symbol, tf=timeframe, e=str(exc))
                return None
        return None

    async def _get_current_prices(self) -> Dict[str, float]:
        prices = {}
        for sym in SYMBOLS:
            try:
                ticker    = await self._exchange.fetch_ticker(sym)
                prices[sym] = float(ticker["last"])
            except Exception:
                # Si falla un símbolo, usar precio cacheado o skip
                if sym in self._df_cache and len(self._df_cache[sym]) > 0:
                    prices[sym] = float(self._df_cache[sym]["close"].iloc[-1])
        return prices

    async def _handle_network_error(self, exc: Exception, context: str) -> None:
        now = time.time()
        self._recent_errors = [t for t in self._recent_errors if now - t < CB_ERR_WINDOW]
        self._recent_errors.append(now)
        log.warning("network_error", context=context, error=str(exc),
                    count=len(self._recent_errors))
        if len(self._recent_errors) >= CB_ERR_LIMIT:
            self._recent_errors.clear()
            log.error("circuit_breaker_network — pausing 5 min")
            if self._bot:
                await self._bot.send_alert("⚠️ Errores de red consecutivos — pausa 5 min", "warn")
            await asyncio.sleep(300)


# ══════════════════════════════════════════════════════════════════════════════
# ── Helpers de tiempo ─────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _seconds_to_next_close(tf_secs: int) -> float:
    """Segundos hasta el próximo cierre de vela del timeframe dado."""
    epoch    = datetime.now(tz=timezone.utc).timestamp()
    next_cls = math.ceil(epoch / tf_secs) * tf_secs
    return max(next_cls - epoch, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# ── Punto de entrada ──────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

async def _main() -> None:
    paper_mode = os.environ.get("PAPER_MODE", "true").lower() not in ("false", "0", "no")
    engine     = LiveEngine(paper_mode=paper_mode)
    try:
        await engine.start()
    except KeyboardInterrupt:
        await engine.shutdown("KeyboardInterrupt")
    except Exception:
        await engine.shutdown("fatal_error")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(_main())
