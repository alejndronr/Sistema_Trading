"""
strategies/signals.py — Generación de señales de trading (Trend Following & Mean Reversion).
========================================================================================
Corrección de over-filtering y optimización para el hardware Atom E3950.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger(__name__)


def signal_trend_following(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estrategia 1: Trend Following (60% del tiempo operativo).
    Corrección del over-filtering: MACD cambia de AND a OR.
    El pullback se amplía de ±0.5% a ±2% (más realista en 4H).
    Añadimos confirmación de Consensus como alternativa al MACD estricto.
    """
    df = df.copy()
    c = df["close"]

    # Condiciones principales — sin cambios
    cond_ema = df.get("ema_alignment_bullish", pd.Series(False, index=df.index)).fillna(False)
    # Si ema_alignment_bullish no existe, usar ema_bullish o calcular de emap
    if not cond_ema.any() and "ema_21" in df.columns and "ema_55" in df.columns and "ema_200" in df.columns:
        cond_ema = (df["ema_21"] > df["ema_55"]) & (df["ema_55"] > df["ema_200"])
    elif "ema_bullish" in df.columns:
        cond_ema = df["ema_bullish"].fillna(False)

    adx_val = df.get("adx", pd.Series(0.0, index=df.index))
    cond_adx = adx_val > 20  # ADX > 20 (era 25, bajar para más señales)

    # RSI: neutral ampliado
    rsi_val = df.get("rsi", pd.Series(50.0, index=df.index))
    cond_rsi = (rsi_val >= 40.0) & (rsi_val <= 70.0)  # RSI 40-70 (era 45-65)

    # MACD: OR en vez de AND — la clave del fix
    macd_bull = df.get("macd_bullish_cross", df.get("macd_line", pd.Series(0.0, index=df.index)) > 0).fillna(False)
    # fallback macd_bull/macd_growing
    if "macd_bull" in df.columns:
        macd_bull = df["macd_bull"].fillna(False)
    
    macd_growing = df.get("macd_histogram", pd.Series(0.0, index=df.index)).diff() > 0
    if "macd_growing" in df.columns:
        macd_growing = df["macd_growing"].fillna(False)

    cond_macd = macd_bull | macd_growing

    # Pullback ampliado: ±2% de EMA21 (en 4H el precio rara vez toca exacto)
    ema21_col = "ema_21" if "ema_21" in df.columns else "ema21"
    ema21 = df.get(ema21_col, c).fillna(c)
    cond_pullback = (c <= ema21 * 1.02) & (c >= ema21 * 0.98)

    cond_ob = df.get("ob_bull", pd.Series(False, index=df.index)).fillna(False)
    cond_fvg = df.get("fvg_bull", pd.Series(False, index=df.index)).fillna(False)
    
    # Extreme volume
    cond_vol = ~df.get("extreme_vol", pd.Series(False, index=df.index)).fillna(True)

    # Filtro de Consensus (suave, de mejora 2)
    cond_consensus = df.get("consensus_bull", pd.Series(True, index=df.index)).fillna(True)

    # Señal: EMA + ADX + RSI + MACD(OR) + zona de entrada (pullback OR OB OR FVG) & consensus
    df["tf_long_signal"] = (
        cond_ema & cond_adx & cond_rsi & cond_macd & cond_vol
        & (cond_pullback | cond_ob | cond_fvg)
        & cond_consensus
    ).astype(int)

    # Cooldown 3 velas — numpy array (pandas 3.x safe)
    sig = df["tf_long_signal"].to_numpy().copy()
    for i in range(1, len(sig)):
        if sig[i] and sig[max(0, i - 3):i].any():
            sig[i] = 0
    df["tf_long_signal"] = sig

    # SL dinámico: 1.5 ATR bajo el mínimo de 5 velas
    atr = df.get("atr", c * 0.015).fillna(c * 0.015)
    low_min_5 = df["low"].rolling(5).min().fillna(df["low"])
    price_risk = c - (low_min_5 - atr * 1.5)
    
    df["tf_stop_long"] = low_min_5 - atr * 1.5
    df["tf_tp1"] = c + price_risk * 2.0   # R/R = 2:1
    df["tf_tp2"] = c + price_risk * 3.0   # R/R = 3:1

    return df


def signal_mean_reversion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estrategia 2: Mean Reversion (30% del tiempo operativo).
    """
    df = df.copy()
    c = df["close"]
    rsi = df.get("rsi", pd.Series(50.0, index=df.index)).fillna(50.0)
    adx = df.get("adx", pd.Series(20.0, index=df.index)).fillna(20.0)
    bb_lower = df.get("bb_lower", c).fillna(c)
    
    cond_adx = adx < 20
    cond_rsi = rsi < 35
    cond_bb = c <= bb_lower * 1.01  # dentro de 1% de la banda inferior
    
    df["mr_long_signal"] = (cond_adx & cond_rsi & cond_bb).astype(int)
    
    # Cooldown 3 velas
    sig = df["mr_long_signal"].to_numpy().copy()
    for i in range(1, len(sig)):
        if sig[i] and sig[max(0, i - 3):i].any():
            sig[i] = 0
    df["mr_long_signal"] = sig
    
    return df


def apply_all_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica todas las estrategias y mapea la señal final para el engine."""
    df = df.copy()
    df = signal_trend_following(df)
    df = signal_mean_reversion(df)
    
    # Mapear a la columna esperada por live_engine.py
    df["signal_trend"] = df["tf_long_signal"]
    return df
