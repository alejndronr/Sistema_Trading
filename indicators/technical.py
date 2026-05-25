"""
indicators/technical.py — Compilación de todos los indicadores técnicos del sistema.
===================================================================================
Incorpora Smart Money Concepts (con fallback), VFI y Consensus Score.
Optimizado para bajo uso de CPU en hardware Atom E3950.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger(__name__)


def add_market_structure(df: pd.DataFrame) -> pd.DataFrame:
    """
    SMC mejorado. Usa smart-money-concepts si está instalado,
    fallback a implementación custom si no.
    """
    try:
        import smartmoneyconcepts as smc_lib
        return _add_market_structure_smc(df, smc_lib)
    except ImportError:
        log.warning("smc_library_not_found_using_fallback")
        return _add_market_structure_custom(df)


def _add_market_structure_smc(df: pd.DataFrame, smc_lib) -> pd.DataFrame:
    """Implementación con librería joshyattridge/smart-money-concepts."""
    ohlc = df[["open", "high", "low", "close"]].copy()

    try:
        # Swing Highs/Lows
        swing = smc_lib.swing_highs_lows(ohlc, swing_length=10)
        df["swing_high"] = swing["HighLow"].eq(1).astype(int)
        df["swing_low"]  = swing["HighLow"].eq(-1).astype(int)

        # BOS y CHoCH
        bos = smc_lib.bos_choch(ohlc, swing, close_break=True)
        df["bos_bull"]   = bos["BOS"].eq(1).fillna(False)
        df["bos_bear"]   = bos["BOS"].eq(-1).fillna(False)
        df["choch_bull"] = bos["CHOCH"].eq(1).fillna(False)
        df["choch_bear"] = bos["CHOCH"].eq(-1).fillna(False)

        # Order Blocks
        ob = smc_lib.ob(ohlc, swing)
        df["ob_bull"] = ob["OB"].eq(1).fillna(False)
        df["ob_bear"] = ob["OB"].eq(-1).fillna(False)
        df["ob_top"]  = ob["Top"].fillna(np.nan)
        df["ob_bot"]  = ob["Bottom"].fillna(np.nan)

        # Fair Value Gaps
        fvg = smc_lib.fvg(ohlc, join_consecutive=True)
        df["fvg_bull"] = fvg["FVG"].eq(1).fillna(False)
        df["fvg_bear"] = fvg["FVG"].eq(-1).fillna(False)
        df["fvg_top"]  = fvg["Top"].fillna(np.nan)
        df["fvg_bot"]  = fvg["Bottom"].fillna(np.nan)

        # Equal Highs/Lows — NUEVO (liquidity pools)
        try:
            liq = smc_lib.liquidity(ohlc, swing, range_percent=0.01)
            df["liq_high"] = liq["Liquidity"].eq(1).fillna(False)
            df["liq_low"]  = liq["Liquidity"].eq(-1).fillna(False)
        except Exception:
            df["liq_high"] = False
            df["liq_low"]  = False

    except Exception as exc:
        log.warning("smc_library_error_using_fallback", error=str(exc))
        return _add_market_structure_custom(df)

    return df


def _add_market_structure_custom(df: pd.DataFrame) -> pd.DataFrame:
    """Fallback: implementación custom original."""
    df = df.copy()
    
    # Asegurar que se calculan los indicadores de estructura originales
    from indicators.market_structure import MarketStructureIndicators
    df = MarketStructureIndicators().calculate_all(df)
    
    # Mapear columnas originales a los nombres simplificados/esperados:
    df["ob_bull"] = df.get("order_block_bullish", False)
    df["ob_bear"] = df.get("order_block_bearish", False)
    df["ob_top"] = df.get("ob_bull_high", np.nan)
    df["ob_bot"] = df.get("ob_bull_low", np.nan)
    
    df["bos_bull"] = df.get("bos_bullish", False)
    df["bos_bear"] = df.get("bos_bearish", False)
    df["choch_bull"] = df.get("choch_bullish", False)
    df["choch_bear"] = df.get("choch_bearish", False)
    
    df["fvg_bull"] = df.get("fvg_bullish", False)
    df["fvg_bear"] = df.get("fvg_bearish", False)
    df["fvg_top"] = df.get("fvg_bull_top", np.nan)
    df["fvg_bot"] = df.get("fvg_bull_bottom", np.nan)
    
    df["liq_high"] = False
    df["liq_low"] = False
    
    return df


def add_vfi(df: pd.DataFrame, period: int = 130) -> pd.DataFrame:
    """
    Volume Flow Indicator — más preciso que OBV para confirmar tendencia.
    Adaptado de freqtrade/technical para Atom E3950.
    Completamente vectorizado, sin loops Python.
    """
    df = df.copy()
    tp = (df["high"] + df["low"] + df["close"]) / 3.0

    # Evitar log(0) con clip
    tp_safe = tp.clip(lower=1e-10)
    inter = np.log(tp_safe) - np.log(tp_safe.shift(1))
    vinter = inter.rolling(30).std().fillna(0.01)

    cutoff = 0.1 * vinter * df["close"]
    vave   = df["volume"].rolling(period).mean().shift(1).fillna(1)
    vmax   = vave * 2.0

    mf = tp - tp.shift(1)

    # Dirección del volumen (vectorizado)
    vcp = np.where(mf > cutoff,  df["volume"],
          np.where(mf < -cutoff, -df["volume"], 0.0))

    vf = pd.Series(vcp, index=df.index)
    vf = vf.clip(lower=-vmax, upper=vmax)

    vave_safe = vave.replace(0, np.nan)
    df["vfi"]      = vf.rolling(period).sum() / vave_safe
    df["vfi"]      = df["vfi"].fillna(0)
    df["vfi_bull"] = df["vfi"] > 0
    return df


def add_consensus(df: pd.DataFrame) -> pd.DataFrame:
    """
    Puntuación 0-100 que agrega múltiples indicadores.
    > 60: sesgo alcista | < 40: sesgo bajista | 40-60: neutral

    Diseñado para el Atom E3950: operaciones vectorizadas simples,
    sin numpy linalg ni scipy.
    """
    df = df.copy()
    weights = {
        "ema_alignment_bullish": 25,   # tendencia principal
        "rsi":         20,   # momentum normalizado
        "macd_bull":   20,   # momentum MACD
        "adx":         20,   # fuerza tendencia
        "vfi_bull":    15,   # volumen real
    }

    score = pd.Series(0.0, index=df.index)
    total_weight = sum(weights.values())

    # Alineación de EMAs
    if "ema_alignment_bullish" in df.columns:
        score += df["ema_alignment_bullish"].fillna(False).astype(float) * weights["ema_alignment_bullish"]
    elif "ema_bullish" in df.columns:
        score += df["ema_bullish"].fillna(False).astype(float) * weights["ema_alignment_bullish"]

    # RSI
    if "rsi" in df.columns:
        rsi_norm = ((df["rsi"].clip(30, 70) - 30) / 40.0)
        score += rsi_norm * weights["rsi"]

    # MACD
    if "macd_bull" not in df.columns:
        if "macd_line" in df.columns:
            df["macd_bull"] = df["macd_line"] > 0
        else:
            df["macd_bull"] = False
    score += df["macd_bull"].fillna(False).astype(float) * weights["macd_bull"]

    # ADX
    if "adx" in df.columns:
        adx_norm = df["adx"].clip(0, 50) / 50.0
        score += adx_norm * weights["adx"]

    # Volume Flow
    if "vfi_bull" in df.columns:
        score += df["vfi_bull"].fillna(False).astype(float) * weights["vfi_bull"]

    df["consensus"]      = (score / total_weight * 100).clip(0, 100)
    df["consensus_bull"] = df["consensus"] >= 55
    df["consensus_bear"] = df["consensus"] <= 40
    return df


def apply_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula la batería completa de indicadores base y avanzados."""
    from indicators.trend import TrendIndicators
    from indicators.momentum import MomentumIndicators
    from indicators.volatility import VolatilityIndicators
    from indicators.volume import VolumeIndicators
    from indicators.market_structure import MarketStructureIndicators

    df = df.copy()
    
    # 1. Batería de indicadores base
    df = TrendIndicators().calculate_all(df)
    df = MomentumIndicators().calculate_all(df)
    df = VolatilityIndicators().calculate_all(df)
    df = VolumeIndicators().calculate_all(df)
    df = MarketStructureIndicators().calculate_all(df)
    
    # Mapeos de compatibilidad
    if "ema_alignment_bullish" in df.columns:
        df["ema_bullish"] = df["ema_alignment_bullish"]

    # 2. Baterías V3.0 añadidas
    df = add_market_structure(df)
    df = add_vfi(df)
    df = add_consensus(df)

    return df
