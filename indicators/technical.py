"""
indicators/technical.py — Compilación de todos los indicadores técnicos del sistema.
===================================================================================
Incorpora Smart Money Concepts (con fallback), VFI, Consensus Score y Capa V5.
Optimizado para bajo uso de CPU en hardware Atom E3950.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger(__name__)

DTYPE = np.float32

# ── Helpers vectorizados para V5 ──────────────────────────────────────────────

def _obv_vectorized(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """OBV 100% vectorizado."""
    direction = np.sign(np.diff(close.astype(DTYPE), prepend=close[0]))
    return np.cumsum(direction * volume.astype(DTYPE))

def _delta_volume(open_: np.ndarray, high: np.ndarray,
                  low: np.ndarray, close: np.ndarray,
                  volume: np.ndarray) -> np.ndarray:
    """Proxy de delta de volumen."""
    rng  = np.where(high == low, 1e-10, high - low)
    buy  = (close - low)  / rng
    sell = (high  - close) / rng
    return (buy - sell).astype(DTYPE)

def _absorption(high: np.ndarray, low: np.ndarray,
                volume: np.ndarray, period: int = 20) -> np.ndarray:
    """Absorción institucional."""
    rng      = (high - low).astype(DTYPE)
    avg_rng  = pd.Series(rng).rolling(period, min_periods=1).mean().values
    avg_vol  = pd.Series(volume.astype(DTYPE)).rolling(period, min_periods=1).mean().values
    return ((volume > avg_vol * 1.8) & (rng < avg_rng * 0.6)).astype(np.int8)

def _find_swing_points(high: np.ndarray, low: np.ndarray,
                       left: int = 5, right: int = 5) -> Tuple[List[Dict], List[Dict]]:
    """Swing points sin lookahead bias."""
    n = len(high)
    swing_highs = []
    swing_lows = []
    for i in range(left, n - right):
        win_h = high[i - left : i + right + 1]
        win_l = low[i  - left : i + right + 1]
        if high[i] == win_h.max():
            swing_highs.append({"index": i, "price": float(high[i])})
        if low[i]  == win_l.min():
            swing_lows.append({"index": i, "price": float(low[i])})
    return swing_highs, swing_lows

def _linear_slope(y: np.ndarray) -> float:
    """Pendiente de regresión lineal simple."""
    n = len(y)
    if n < 2:
        return 0.0
    x   = np.arange(n, dtype=np.float32)
    num = n * np.dot(x, y) - x.sum() * y.sum()
    den = n * np.dot(x, x) - x.sum() ** 2
    return float(num / den) if den != 0 else 0.0

def _hurst_exponent(close: pd.Series, lags: int = 20) -> pd.Series:
    """
    Exponente de Hurst vectorizado usando Variance Ratio proxy.
    Optimizado para Atom E3950 (solo Pandas/NumPy).
    < 0.5 Mean Reverting, 0.5 Random Walk, > 0.5 Trending.
    """
    lag1_var = close.diff().rolling(lags).var()
    lag2_var = close.diff(2).rolling(lags).var()
    # Para evitar división por cero
    lag1_var = lag1_var.replace(0, np.nan)
    # H approx = log2(var(lag2)/var(lag1)) / 2  (Simplified R/S proxy)
    hurst = np.log2(lag2_var / lag1_var) / 2.0
    return hurst.fillna(0.5)

def _zscore_vwap(close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
    """Z-Score del precio contra el VWAP de N periodos."""
    typical_price = close # Simplificación usando Close para mayor velocidad
    vwap = (typical_price * volume).rolling(period).sum() / volume.rolling(period).sum()
    std = close.rolling(period).std()
    std = std.replace(0, np.nan)
    return (close - vwap) / std

def _volatility_garch_proxy(returns: pd.Series, span: int = 20) -> pd.Series:
    """
    Proxy de GARCH(1,1) usando EWMA de retornos al cuadrado.
    Retorna el ratio de aceleración de la varianza.
    """
    ret_sq = returns ** 2
    vol_ema = ret_sq.ewm(span=span).mean()
    # Volatility Ratio (Aceleración)
    vol_ratio = vol_ema / vol_ema.shift(5).replace(0, np.nan)
    return vol_ratio.fillna(1.0)

def _ts_momentum(close: pd.Series, period: int = 20) -> Tuple[pd.Series, pd.Series]:
    """
    Time Series Momentum: Pendiente y T-Stat de la regresión lineal sobre 20 velas.
    Implementación vectorizada rápida.
    """
    x = np.arange(period)
    x_mean = x.mean()
    x_var = x.var() * period
    
    # Usando rolling apply de numpy es lento en python, 
    # Usaremos una aproximación rápida con la diferencia entre la EMA rápida y lenta normalizada
    # o implementamos una regresión lineal rolling de pandas pura.
    
    # Solución ultra-rápida (Atom E3950 proxy de momentum)
    # En lugar de regresión lineal completa, usamos el ratio de la EMA rápida vs lenta
    # o simplemente (close - close.shift(period)) / period como slope
    
    # Pendiente: Cambio de precio promedio por vela
    slope = (close - close.shift(period)) / period
    
    # Standard Error proxy: volatilidad normalizada por raiz del periodo
    se = close.rolling(period).std() / np.sqrt(period)
    
    # T-Stat proxy
    t_stat = slope / se.replace(0, np.nan)
    return slope.fillna(0.0), t_stat.fillna(0.0)

def _regime_hmm_proxy(close: pd.Series, atr: pd.Series, period: int = 60) -> pd.DataFrame:
    """
    Proxy de Hidden Markov Model / Gaussian Mixture.
    Devuelve la probabilidad continua (0.0 - 1.0) de pertenecer a 4 regímenes:
    bull_trend, bear_trend, high_vol_range, low_vol_range.
    """
    ret = close.pct_change()
    
    # 1. Componentes de distribución
    ret_mean = ret.rolling(period).mean()
    ret_std  = ret.rolling(period).std().replace(0, np.nan)
    
    # Normalizamos el retorno medio y la volatilidad (Z-Scores empíricos cortos)
    z_ret = ret_mean / (ret_std / np.sqrt(period))
    z_ret = z_ret.fillna(0)
    
    atr_norm = atr / close
    atr_mean = atr_norm.rolling(period*3).mean()
    atr_std  = atr_norm.rolling(period*3).std().replace(0, np.nan)
    z_vol = (atr_norm - atr_mean) / atr_std
    z_vol = z_vol.fillna(0)
    
    # 2. Funciones de Activación Probabilística (Sigmoid)
    prob_bull = 1 / (1 + np.exp(-(z_ret - 1.0)))  
    prob_bear = 1 / (1 + np.exp(z_ret + 1.0))
    prob_range = 1.0 - np.maximum(prob_bull, prob_bear)
    
    # Separación por Volatilidad
    prob_high_vol = 1 / (1 + np.exp(-(z_vol - 0.5))) 
    prob_low_vol  = 1.0 - prob_high_vol
    
    # 3. Probabilidades Conjuntas
    p_bull = prob_bull
    p_bear = prob_bear
    p_range_hv = prob_range * prob_high_vol
    p_range_lv = prob_range * prob_low_vol
    
    # 4. Normalización (Softmax sum=1.0)
    total = p_bull + p_bear + p_range_hv + p_range_lv
    total = total.replace(0, np.nan)
    
    return pd.DataFrame({
        "prob_bull": p_bull / total,
        "prob_bear": p_bear / total,
        "prob_range_hv": p_range_hv / total,
        "prob_range_lv": p_range_lv / total
    }).fillna(0.25)


# ── Capa V5 Clases ────────────────────────────────────────────────────────────

class CandlePatterns:
    @staticmethod
    def _body(o: pd.Series, c: pd.Series) -> pd.Series:
        return (c - o).abs()

    @staticmethod
    def _rng(h: pd.Series, l: pd.Series) -> pd.Series:
        return (h - l).replace(0, np.nan)

    @staticmethod
    def _upper_shadow(o: pd.Series, h: pd.Series, c: pd.Series) -> pd.Series:
        return h - pd.concat([o, c], axis=1).max(axis=1)

    @staticmethod
    def _lower_shadow(o: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
        return pd.concat([o, c], axis=1).min(axis=1) - l

    @classmethod
    def detect_all(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        o, h, l, c = df["open"], df["high"], df["low"], df["close"]
        body       = cls._body(o, c)
        rng        = cls._rng(h, l)
        upper      = cls._upper_shadow(o, h, c)
        lower      = cls._lower_shadow(o, l, c)
        bull       = c > o
        bear       = c < o

        body_pct = body / rng

        df["cp_hammer"] = ((body_pct < 0.35) & (lower > 2 * body) & (upper < body) & bear.shift(1).fillna(False)).astype(np.int8) * 15
        df["cp_inv_hammer"] = ((body_pct < 0.35) & (upper > 2 * body) & (lower < body) & bear.shift(1).fillna(False)).astype(np.int8) * 12
        df["cp_engulf_bull"] = (bear.shift(1).fillna(False) & bull & (c > o.shift(1)) & (o < c.shift(1))).astype(np.int8) * 15
        doji_mid = body_pct.shift(1) < 0.30
        df["cp_morning_star"] = (bear.shift(2).fillna(False) & doji_mid.fillna(False) & bull & (c > ((o.shift(2) + c.shift(2)) / 2))).astype(np.int8) * 18
        df["cp_piercing"] = (bear.shift(1).fillna(False) & bull & (c > (o.shift(1) + c.shift(1)) / 2) & (c < o.shift(1))).astype(np.int8) * 14
        df["cp_harami_bull"] = (bear.shift(1).fillna(False) & bull & (o > c.shift(1)) & (c < o.shift(1))).astype(np.int8) * 10
        df["cp_tweezer_bot"] = (bear.shift(1).fillna(False) & bull & ((l - l.shift(1)).abs() / l.shift(1) < 0.002)).astype(np.int8) * 12
        df["cp_3white"] = (bull & bull.shift(1).fillna(False) & bull.shift(2).fillna(False) & (c > c.shift(1)) & (c.shift(1) > c.shift(2))).astype(np.int8) * 20
        df["cp_shooting_star"] = ((body_pct < 0.35) & (upper > 2 * body) & (lower < body) & bull.shift(1).fillna(False)).astype(np.int8) * 15
        df["cp_hanging_man"] = ((body_pct < 0.35) & (lower > 2 * body) & (upper < body) & bull.shift(1).fillna(False)).astype(np.int8) * 15
        df["cp_engulf_bear"] = (bull.shift(1).fillna(False) & bear & (c < o.shift(1)) & (o > c.shift(1))).astype(np.int8) * 15
        df["cp_evening_star"] = (bull.shift(2).fillna(False) & doji_mid.fillna(False) & bear & (c < ((o.shift(2) + c.shift(2)) / 2))).astype(np.int8) * 18
        df["cp_dark_cloud"] = (bull.shift(1).fillna(False) & bear & (c < (o.shift(1) + c.shift(1)) / 2) & (c > o.shift(1))).astype(np.int8) * 14
        df["cp_harami_bear"] = (bull.shift(1).fillna(False) & bear & (o < c.shift(1)) & (c > o.shift(1))).astype(np.int8) * 10
        df["cp_3black"] = (bear & bear.shift(1).fillna(False) & bear.shift(2).fillna(False) & (c < c.shift(1)) & (c.shift(1) < c.shift(2))).astype(np.int8) * 20
        df["cp_doji"] = (body_pct < 0.10).astype(np.int8) * 8
        df["cp_marubozu_bull"] = (bull & (body_pct > 0.90)).astype(np.int8) * 10
        df["cp_marubozu_bear"] = (bear & (body_pct > 0.90)).astype(np.int8) * 10

        bull_cols = ["cp_hammer","cp_inv_hammer","cp_engulf_bull","cp_morning_star",
                     "cp_piercing","cp_harami_bull","cp_tweezer_bot","cp_3white",
                     "cp_marubozu_bull"]
        bear_cols = ["cp_shooting_star","cp_hanging_man","cp_engulf_bear",
                     "cp_evening_star","cp_dark_cloud","cp_harami_bear",
                     "cp_3black","cp_marubozu_bear"]

        df["cp_bull_score"] = df[[c for c in bull_cols if c in df.columns]].sum(axis=1)
        df["cp_bear_score"] = df[[c for c in bear_cols if c in df.columns]].sum(axis=1)
        df["cp_net_score"]  = df["cp_bull_score"] - df["cp_bear_score"]

        return df

class ChartPatterns:
    def __init__(self, tol: float = 0.025):
        self.tol = tol

    def _near(self, a: float, b: float) -> bool:
        return abs(a - b) / max(abs(b), 1e-10) <= self.tol

    def head_and_shoulders(self, swing_highs: List[Dict]) -> List[Dict]:
        results = []
        if len(swing_highs) < 3:
            return results
        for i in range(len(swing_highs) - 2):
            ls, head, rs = (swing_highs[i]["price"], swing_highs[i+1]["price"], swing_highs[i+2]["price"])
            if head <= ls or head <= rs:
                continue
            if not self._near(ls, rs):
                continue
            neckline  = (ls + rs) / 2
            target    = neckline - (head - neckline)
            results.append({
                "name": "HEAD_AND_SHOULDERS", "direction": "bear",
                "score": 25, "neckline": neckline, "target": target,
                "idx": swing_highs[i+2]["index"],
            })
        return results

    def inv_head_and_shoulders(self, swing_lows: List[Dict]) -> List[Dict]:
        results = []
        if len(swing_lows) < 3:
            return results
        for i in range(len(swing_lows) - 2):
            ls, head, rs = (swing_lows[i]["price"], swing_lows[i+1]["price"], swing_lows[i+2]["price"])
            if head >= ls or head >= rs:
                continue
            if not self._near(ls, rs):
                continue
            neckline = (ls + rs) / 2
            target   = neckline + (neckline - head)
            results.append({
                "name": "INV_HEAD_AND_SHOULDERS", "direction": "bull",
                "score": 25, "neckline": neckline, "target": target,
                "idx": swing_lows[i+2]["index"],
            })
        return results

    def double_bottom(self, swing_lows: List[Dict]) -> List[Dict]:
        results = []
        if len(swing_lows) < 2:
            return results
        for i in range(len(swing_lows) - 1):
            b1, b2 = swing_lows[i]["price"], swing_lows[i+1]["price"]
            if not self._near(b1, b2):
                continue
            results.append({
                "name": "DOUBLE_BOTTOM", "direction": "bull",
                "score": 20, "target": b2 + abs(b2 - b1) * 2,
                "idx": swing_lows[i+1]["index"],
            })
        return results

    def double_top(self, swing_highs: List[Dict]) -> List[Dict]:
        results = []
        if len(swing_highs) < 2:
            return results
        for i in range(len(swing_highs) - 1):
            t1, t2 = swing_highs[i]["price"], swing_highs[i+1]["price"]
            if not self._near(t1, t2):
                continue
            results.append({
                "name": "DOUBLE_TOP", "direction": "bear",
                "score": 20, "target": t2 - abs(t2 - t1) * 2,
                "idx": swing_highs[i+1]["index"],
            })
        return results

    def triangle(self, swing_highs: List[Dict], swing_lows: List[Dict], n: int = 5) -> List[Dict]:
        results = []
        if len(swing_highs) < n or len(swing_lows) < n:
            return results
        h_prices = np.array([x["price"] for x in swing_highs[-n:]], dtype=DTYPE)
        l_prices = np.array([x["price"] for x in swing_lows[-n:]],  dtype=DTYPE)
        h_slope = _linear_slope(h_prices)
        l_slope = _linear_slope(l_prices)

        if h_slope < -0.001 and l_slope > 0.001:
            results.append({
                "name": "SYMMETRICAL_TRIANGLE", "direction": "neutral",
                "score": 12, "breakout_imminent": True, "idx": swing_highs[-1]["index"],
            })
        elif abs(h_slope) < 0.001 and l_slope > 0.001:
            results.append({
                "name": "ASCENDING_TRIANGLE", "direction": "bull",
                "score": 18, "idx": swing_highs[-1]["index"],
            })
        elif h_slope < -0.001 and abs(l_slope) < 0.001:
            results.append({
                "name": "DESCENDING_TRIANGLE", "direction": "bear",
                "score": 18, "idx": swing_lows[-1]["index"],
            })
        elif h_slope > 0.001 and l_slope > h_slope * 1.5:
            results.append({
                "name": "RISING_WEDGE", "direction": "bear",
                "score": 15, "idx": swing_highs[-1]["index"],
            })
        elif h_slope < -0.001 and l_slope < h_slope * 1.5:
            results.append({
                "name": "FALLING_WEDGE", "direction": "bull",
                "score": 15, "idx": swing_lows[-1]["index"],
            })
        return results

    def flag_pattern(self, close: np.ndarray, swing_highs: List[Dict], swing_lows: List[Dict], lookback: int = 20) -> List[Dict]:
        results = []
        if len(close) < lookback + 5:
            return results
        impulse = (close[-1] - close[-6]) / close[-6] * 100
        if impulse > 3.0:
            recent = close[-lookback:]
            slope  = _linear_slope(recent.astype(DTYPE))
            if -0.005 < slope < 0:
                target = close[-1] + abs(close[-6] - close[-11])
                results.append({
                    "name": "BULL_FLAG", "direction": "bull",
                    "score": 18, "target": target, "idx": len(close) - 1,
                })
        elif impulse < -3.0:
            recent = close[-lookback:]
            slope  = _linear_slope(recent.astype(DTYPE))
            if 0 < slope < 0.005:
                results.append({
                    "name": "BEAR_FLAG", "direction": "bear",
                    "score": 18, "idx": len(close) - 1,
                })
        return results

    def scan_all(self, df: pd.DataFrame, swing_highs: List[Dict], swing_lows: List[Dict]) -> List[Dict]:
        c = df["close"].values
        all_patterns = []
        all_patterns += self.head_and_shoulders(swing_highs)
        all_patterns += self.inv_head_and_shoulders(swing_lows)
        all_patterns += self.double_bottom(swing_lows)
        all_patterns += self.double_top(swing_highs)
        all_patterns += self.triangle(swing_highs, swing_lows)
        all_patterns += self.flag_pattern(c, swing_highs, swing_lows)
        return all_patterns

class SRZones:
    def __init__(self, min_touches: int = 2, cluster_pct: float = 0.003):
        self.min_touches = min_touches
        self.cluster_pct = cluster_pct

    def find(self, swing_highs: List[Dict], swing_lows: List[Dict], current_price: float) -> List[Dict]:
        all_pts = sorted(
            [(p["price"], "high") for p in swing_highs] +
            [(p["price"], "low")  for p in swing_lows],
            key=lambda x: x[0],
        )
        if len(all_pts) < self.min_touches:
            return []
        clusters = []
        current = [all_pts[0]]
        for i in range(1, len(all_pts)):
            if (abs(all_pts[i][0] - current[-1][0]) / max(current[-1][0], 1e-10) <= self.cluster_pct):
                current.append(all_pts[i])
            else:
                if len(current) >= self.min_touches:
                    clusters.append(current)
                current = [all_pts[i]]
        if len(current) >= self.min_touches:
            clusters.append(current)

        zones = []
        for cluster in clusters:
            prices = [p[0] for p in cluster]
            center = float(np.mean(prices))
            top = float(max(prices))
            bottom = float(min(prices))
            touches = len(cluster)
            strength = min(touches / 6.0, 1.0)
            if current_price > top * 1.005:
                zone_type = "support"
            elif current_price < bottom * 0.995:
                zone_type = "resistance"
            else:
                highs_n = sum(1 for p in cluster if p[1] == "high")
                zone_type = "resistance" if highs_n > touches / 2 else "support"
            zones.append({
                "center": center, "top": top, "bottom": bottom,
                "touches": touches, "strength": strength, "type": zone_type,
            })
        return sorted(zones, key=lambda z: abs(z["center"] - current_price))

    def nearest(self, zones: List[Dict], price: float, n: int = 3) -> List[Dict]:
        return zones[:n]

    def at_zone(self, zones: List[Dict], price: float, tol: float = 0.005) -> Tuple[bool, Optional[Dict]]:
        for z in zones:
            if z["bottom"] * (1 - tol) <= price <= z["top"] * (1 + tol):
                return True, z
        return False, None

    def score_for_zone(self, zone: Dict, direction: str) -> int:
        base = int(zone["strength"] * 15)
        if (direction == "long"  and zone["type"] == "support") or \
           (direction == "short" and zone["type"] == "resistance"):
            return base + 5
        return 0

class Fibonacci:
    RETRACE_LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    EXTENSION_LEVELS = [1.0, 1.272, 1.618, 2.0, 2.618]

    def retracement(self, high: float, low: float) -> Dict[str, float]:
        diff = high - low
        return {f"{r:.3f}": high - r * diff for r in self.RETRACE_LEVELS}

    def extension(self, low: float, high: float) -> Dict[str, float]:
        diff = high - low
        return {f"{e:.3f}": high + (e - 1.0) * diff for e in self.EXTENSION_LEVELS}

    def nearest_level(self, price: float, levels: Dict[str, float], tol: float = 0.005) -> Optional[Tuple[str, float]]:
        best_dist = float("inf")
        best_name = None
        best_price = None
        for name, lvl_price in levels.items():
            dist = abs(price - lvl_price) / max(abs(lvl_price), 1e-10)
            if dist < best_dist:
                best_dist = dist
                best_name = name
                best_price = lvl_price
        if best_dist <= tol:
            return best_name, best_price
        return None

    def tp_levels(self, entry: float, stop: float, direction: str) -> Tuple[float, float, float]:
        dist = abs(entry - stop)
        if direction == "long":
            tp1 = entry + 1.272 * dist
            tp2 = entry + 1.618 * dist
            tp3 = entry + 2.618 * dist
        else:
            tp1 = entry - 1.272 * dist
            tp2 = entry - 1.618 * dist
            tp3 = entry - 2.618 * dist
        return tp1, tp2, tp3

class MarketStructure:
    def analyze(self, swing_highs: List[Dict], swing_lows: List[Dict], close: np.ndarray) -> Dict:
        events = []
        bias = "neutral"
        if not swing_highs or not swing_lows:
            return {"bias": bias, "events": events, "score": 0}
        last_sh = swing_highs[-1]["price"]
        last_sl = swing_lows[-1]["price"]
        prev_sh = swing_highs[-2]["price"] if len(swing_highs) > 1 else last_sh
        prev_sl = swing_lows[-2]["price"]  if len(swing_lows)  > 1 else last_sl
        current = float(close[-1])

        if current > last_sh:
            events.append({"type": "BOS_BULL", "score": 18, "level": last_sh})
            bias = "bull"
        elif current < last_sl:
            events.append({"type": "BOS_BEAR", "score": 18, "level": last_sl})
            bias = "bear"

        if current > prev_sh and last_sl < prev_sl:
            events.append({"type": "CHOCH_BULL", "score": 22, "level": prev_sh})
            bias = "bull"
        elif current < prev_sl and last_sh > prev_sh:
            events.append({"type": "CHOCH_BEAR", "score": 22, "level": prev_sl})
            bias = "bear"

        return {"bias": bias, "events": events, "score": sum(e["score"] for e in events)}

def enrich_v5(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    o = df["open"].values.astype(DTYPE)
    h = df["high"].values.astype(DTYPE)
    l = df["low"].values.astype(DTYPE)
    c = df["close"].values.astype(DTYPE)
    v = df["volume"].values.astype(DTYPE)

    df["obv"]        = _obv_vectorized(c, v)
    df["delta_vol"]  = _delta_volume(o, h, l, c, v)
    df["absorption"] = _absorption(h, l, v)
    df["obv_rising"] = df["obv"] > df["obv"].shift(3)
    df["cvd_cumul"]  = df["delta_vol"].cumsum()
    df["cvd_rising"] = df["cvd_cumul"] > df["cvd_cumul"].shift(5)
    df = CandlePatterns.detect_all(df)
    return df

def get_structure_context(df: pd.DataFrame, lookback: int = 100) -> Dict:
    window = df.iloc[-lookback:].copy()
    h_arr = window["high"].values
    l_arr = window["low"].values
    c_arr = window["close"].values
    price = float(c_arr[-1])

    swing_highs, swing_lows = _find_swing_points(h_arr, l_arr, left=5, right=5)
    ms = MarketStructure()
    structure = ms.analyze(swing_highs, swing_lows, c_arr)

    sr = SRZones(min_touches=2, cluster_pct=0.003)
    zones = sr.find(swing_highs, swing_lows, price)
    at_zone, current_zone = sr.at_zone(zones, price)

    fib = Fibonacci()
    fib_retrace = {}
    fib_extend = {}
    if swing_highs and swing_lows:
        last_high = swing_highs[-1]["price"]
        last_low  = swing_lows[-1]["price"]
        fib_retrace = fib.retracement(last_high, last_low)
        fib_extend  = fib.extension(last_low, last_high)

    cp = ChartPatterns(tol=0.025)
    chart_patterns = cp.scan_all(window, swing_highs, swing_lows)

    return {
        "swing_highs":    swing_highs,
        "swing_lows":     swing_lows,
        "structure":      structure,
        "zones":          zones,
        "at_zone":        at_zone,
        "current_zone":   current_zone,
        "fib_retrace":    fib_retrace,
        "fib_extend":     fib_extend,
        "chart_patterns": chart_patterns,
        "price":          price,
        "sr_helper":      sr,
        "fib_helper":     fib,
    }

# ── Clases de Indicadores Individuales para compatibilidad de tests ───────────

from config.settings import INDICATORS, TrendIndicatorParams, MomentumIndicatorParams, VolatilityIndicatorParams, VolumeIndicatorParams, MarketStructureParams

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def _rsi(c: pd.Series, p: int = 14) -> pd.Series:
    d = c.diff()
    g = d.clip(lower=0).rolling(p).mean()
    l = (-d.clip(upper=0)).rolling(p).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def _atr(df: pd.DataFrame, p: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hpc = (df["high"] - df["close"].shift()).abs()
    lpc = (df["low"] - df["close"].shift()).abs()
    return pd.concat([hl, hpc, lpc], axis=1).max(axis=1).ewm(span=p, adjust=False).mean()

def _bbands(c: pd.Series, p: int = 20, s: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    m = c.rolling(p).mean()
    std = c.rolling(p).std()
    return m - s * std, m, m + s * std

def _macd(c: pd.Series, f: int = 12, sl: int = 26, sg: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ef = _ema(c, f)
    es = _ema(c, sl)
    ml = ef - es
    sig = _ema(ml, sg)
    return ml, sig, ml - sig

def _adx(df: pd.DataFrame, p: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    pdm = h.diff().clip(lower=0)
    mdm = (-l.diff()).clip(lower=0)
    pdm = pdm.where(pdm > mdm, 0)
    mdm = mdm.where(mdm > pdm, 0)
    atr_raw = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr_s = atr_raw.ewm(span=p, adjust=False).mean()
    pdi = 100 * pdm.ewm(span=p, adjust=False).mean() / atr_s.replace(0, np.nan)
    mdi = 100 * mdm.ewm(span=p, adjust=False).mean() / atr_s.replace(0, np.nan)
    dx = (100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan))
    return dx.ewm(span=p, adjust=False).mean().fillna(0)

class TrendIndicators:
    def __init__(self, params: TrendIndicatorParams = INDICATORS.trend):
        self.p = params

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.ema(df)
        df = self.supertrend(df)
        df = self.adx(df)
        df = self.ema_alignment(df)
        df = self.trend_regime(df)
        return df

    def ema(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[f"ema_{self.p.ema_fast}"] = _ema(df["close"], self.p.ema_fast)
        df[f"ema_{self.p.ema_mid}"] = _ema(df["close"], self.p.ema_mid)
        df[f"ema_{self.p.ema_slow}"] = _ema(df["close"], self.p.ema_slow)
        return df

    def supertrend(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        period = self.p.supertrend_atr_period
        factor = self.p.supertrend_factor
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
        atr_s = tr.ewm(span=period, adjust=False).mean()
        hl2 = (high + low) / 2
        upper_band = hl2 + factor * atr_s
        lower_band = hl2 - factor * atr_s

        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=int)
        supertrend.iloc[0] = upper_band.iloc[0]
        direction.iloc[0] = -1

        for i in range(1, len(df)):
            prev_upper = upper_band.iloc[i - 1]
            prev_lower = lower_band.iloc[i - 1]
            curr_upper = upper_band.iloc[i]
            curr_lower = lower_band.iloc[i]
            prev_close = close.iloc[i - 1]
            curr_close = close.iloc[i]

            final_upper = curr_upper if curr_upper < prev_upper or prev_close > prev_upper else prev_upper
            final_lower = curr_lower if curr_lower > prev_lower or prev_close < prev_lower else prev_lower

            prev_st = supertrend.iloc[i - 1]
            prev_dir = direction.iloc[i - 1]

            if prev_st == prev_upper:
                direction.iloc[i] = 1 if curr_close > final_upper else -1
            else:
                direction.iloc[i] = -1 if curr_close < final_lower else 1

            supertrend.iloc[i] = final_lower if direction.iloc[i] == 1 else final_upper

        df["supertrend"] = supertrend
        df["supertrend_direction"] = direction
        return df

    def adx(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        period = self.p.adx_period
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
        dm_plus = high.diff()
        dm_minus = -low.diff()
        dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0.0)
        dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0.0)

        atr_s = tr.ewm(alpha=1 / period, adjust=False).mean()
        di_plus = 100 * dm_plus.ewm(alpha=1 / period, adjust=False).mean() / atr_s
        di_minus = 100 * dm_minus.ewm(alpha=1 / period, adjust=False).mean() / atr_s
        dx = (100 * (di_plus - di_minus).abs() / (di_plus + di_minus)).replace([np.inf, -np.inf], np.nan)
        adx_val = dx.ewm(alpha=1 / period, adjust=False).mean()

        df["adx"] = adx_val
        df["adx_plus_di"] = di_plus
        df["adx_minus_di"] = di_minus
        return df

    def ema_alignment(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        f, m, s = f"ema_{self.p.ema_fast}", f"ema_{self.p.ema_mid}", f"ema_{self.p.ema_slow}"
        if not all(col in df.columns for col in [f, m, s]):
            df = self.ema(df)
        df["ema_alignment_bullish"] = (df[f] > df[m]) & (df[m] > df[s])
        df["ema_alignment_bearish"] = (df[f] < df[m]) & (df[m] < df[s])
        return df

    def trend_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        required = ["adx", "ema_alignment_bullish", "ema_alignment_bearish"]
        for col in required:
            if col not in df.columns:
                df = self.calculate_all(df)
                break
        conditions = [
            df["ema_alignment_bullish"] & (df["adx"] > self.p.adx_trend_threshold),
            df["ema_alignment_bearish"] & (df["adx"] > self.p.adx_trend_threshold),
        ]
        choices = ["bullish", "bearish"]
        df["trend_regime"] = np.select(conditions, choices, default="ranging")
        return df

    def is_strong_trend(self, df: pd.DataFrame) -> pd.Series:
        if "adx" not in df.columns:
            df = self.adx(df)
        return df["adx"] > self.p.adx_strong_threshold

    def price_above_ema21(self, df: pd.DataFrame) -> pd.Series:
        col = f"ema_{self.p.ema_fast}"
        if col not in df.columns:
            df = self.ema(df)
        return df["close"] > df[col]

    def retrace_to_ema21(self, df: pd.DataFrame, tolerance_atr: float = 0.5) -> pd.Series:
        col = f"ema_{self.p.ema_fast}"
        if col not in df.columns:
            df = self.ema(df)
        atr = (df["high"] - df["low"]).rolling(14).mean()
        lower_band = df[col] - tolerance_atr * atr
        upper_band = df[col] + tolerance_atr * atr
        return (df["low"] <= upper_band) & (df["high"] >= lower_band)

class MomentumIndicators:
    def __init__(self, params: MomentumIndicatorParams = INDICATORS.momentum):
        self.p = params

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.rsi(df)
        df = self.rsi_divergences(df)
        df = self.macd(df)
        df = self.stochastic_rsi(df)
        return df

    def rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["rsi"] = _rsi(df["close"], self.p.rsi_period)
        df["rsi"] = df["rsi"].fillna(50)
        df["rsi_overbought"] = df["rsi"] > self.p.rsi_overbought
        df["rsi_oversold"] = df["rsi"] < self.p.rsi_oversold
        df["rsi_neutral"] = (df["rsi"] >= self.p.rsi_neutral_low) & (df["rsi"] <= self.p.rsi_neutral_high)
        return df

    def rsi_divergences(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "rsi" not in df.columns:
            df = self.rsi(df)
        lookback = self.p.divergence_lookback
        n = len(df)
        bull_div = pd.Series(False, index=df.index)
        bear_div = pd.Series(False, index=df.index)
        bull_hidden = pd.Series(False, index=df.index)
        bear_hidden = pd.Series(False, index=df.index)

        for i in range(lookback, n):
            window_close = df["close"].iloc[i - lookback : i + 1]
            window_rsi = df["rsi"].iloc[i - lookback : i + 1]
            window_low = df["low"].iloc[i - lookback : i + 1]
            window_high = df["high"].iloc[i - lookback : i + 1]

            curr_close = df["close"].iloc[i]
            curr_rsi = df["rsi"].iloc[i]
            prev_min_close = window_close.iloc[:-1].min()
            prev_max_close = window_close.iloc[:-1].max()
            prev_min_rsi = window_rsi.iloc[:-1].min()
            prev_max_rsi = window_rsi.iloc[:-1].max()
            curr_low = df["low"].iloc[i]
            curr_high = df["high"].iloc[i]
            prev_min_low = window_low.iloc[:-1].min()
            prev_max_high = window_high.iloc[:-1].max()

            if curr_low < prev_min_low and curr_rsi > prev_min_rsi:
                if curr_rsi < 50:
                    bull_div.iloc[i] = True
            if curr_high > prev_max_high and curr_rsi < prev_max_rsi:
                if curr_rsi > 50:
                    bear_div.iloc[i] = True
            if curr_low > prev_min_low and curr_rsi < prev_min_rsi:
                bull_hidden.iloc[i] = True
            if curr_high < prev_max_high and curr_rsi > prev_max_rsi:
                bear_hidden.iloc[i] = True

        df["rsi_divergence_bullish"] = bull_div
        df["rsi_divergence_bearish"] = bear_div
        df["rsi_divergence_bullish_hidden"] = bull_hidden
        df["rsi_divergence_bearish_hidden"] = bear_hidden
        return df

    def macd(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        ml, sig, hist = _macd(df["close"], self.p.macd_fast, self.p.macd_slow, self.p.macd_signal)
        df["macd_line"] = ml
        df["macd_signal"] = sig
        df["macd_histogram"] = hist
        df["macd_bullish"] = (df["macd_line"] > 0) & (df["macd_histogram"] > 0)
        df["macd_bearish"] = (df["macd_line"] < 0) & (df["macd_histogram"] < 0)
        df["macd_growing"] = df["macd_histogram"] > df["macd_histogram"].shift(1)

        prev_hist = df["macd_histogram"].shift(1)
        df["macd_bullish_cross"] = (prev_hist < 0) & (df["macd_histogram"] >= 0)
        df["macd_bearish_cross"] = (prev_hist > 0) & (df["macd_histogram"] <= 0)

        price_higher_high = df["close"] > df["close"].rolling(10).max().shift(1)
        macd_lower_high = df["macd_line"] < df["macd_line"].rolling(10).max().shift(1)
        df["macd_divergence_bearish"] = price_higher_high & macd_lower_high

        price_lower_low = df["close"] < df["close"].rolling(10).min().shift(1)
        macd_higher_low = df["macd_line"] > df["macd_line"].rolling(10).min().shift(1)
        df["macd_divergence_bullish"] = price_lower_low & macd_higher_low
        return df

    def stochastic_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "rsi" not in df.columns:
            df = self.rsi(df)
        rsi = df["rsi"]
        period = self.p.stoch_rsi_period
        smooth_k = self.p.stoch_rsi_smooth_k
        smooth_d = self.p.stoch_rsi_smooth_d

        rsi_min = rsi.rolling(period).min()
        rsi_max = rsi.rolling(period).max()
        rsi_range = rsi_max - rsi_min

        stoch_rsi_raw = (rsi - rsi_min) / rsi_range.replace(0, np.nan)
        stoch_rsi_raw = stoch_rsi_raw.fillna(0.5) * 100

        df["stoch_rsi_k"] = stoch_rsi_raw.rolling(smooth_k).mean()
        df["stoch_rsi_d"] = df["stoch_rsi_k"].rolling(smooth_d).mean()
        df["stoch_rsi_overbought"] = df["stoch_rsi_k"] > 80
        df["stoch_rsi_oversold"] = df["stoch_rsi_k"] < 20

        prev_k = df["stoch_rsi_k"].shift(1)
        prev_d = df["stoch_rsi_d"].shift(1)
        df["stoch_rsi_bullish_cross"] = (prev_k < prev_d) & (df["stoch_rsi_k"] >= df["stoch_rsi_d"])
        df["stoch_rsi_bearish_cross"] = (prev_k > prev_d) & (df["stoch_rsi_k"] <= df["stoch_rsi_d"])
        df["stoch_rsi_bullish_divergence"] = (df["close"] < df["close"].shift(5)) & (df["stoch_rsi_k"] > df["stoch_rsi_k"].shift(5)) & df["stoch_rsi_oversold"]
        df["stoch_rsi_bearish_divergence"] = (df["close"] > df["close"].shift(5)) & (df["stoch_rsi_k"] < df["stoch_rsi_k"].shift(5)) & df["stoch_rsi_overbought"]
        return df

class VolatilityIndicators:
    def __init__(self, params: VolatilityIndicatorParams = INDICATORS.volatility):
        self.p = params

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.atr(df)
        df = self.bollinger_bands(df)
        df = self.keltner_channels(df)
        df = self.bb_squeeze(df)
        df = self.atr_regime(df)
        return df

    def atr(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["atr"] = _atr(df, self.p.atr_period)
        df["atr_pct"] = df["atr"] / df["close"] * 100
        return df

    def bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        period = self.p.bb_period
        std_mult = self.p.bb_std
        close = df["close"]
        middle = close.rolling(period).mean()
        std = close.rolling(period).std()
        df["bb_middle"] = middle
        df["bb_upper"] = middle + std_mult * std
        df["bb_lower"] = middle - std_mult * std
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        bb_range = df["bb_upper"] - df["bb_lower"]
        df["bb_pct"] = (close - df["bb_lower"]) / bb_range.replace(0, np.nan)
        df["bb_touch_upper"] = close >= df["bb_upper"] * 0.999
        df["bb_touch_lower"] = close <= df["bb_lower"] * 1.001
        return df

    def keltner_channels(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        period = self.p.keltner_period
        factor = self.p.keltner_factor
        atr_period = self.p.keltner_atr_period
        close = df["close"]
        middle = close.ewm(span=period, adjust=False).mean()
        high = df["high"]
        low = df["low"]
        tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
        kc_atr = tr.ewm(alpha=1 / atr_period, adjust=False).mean()
        df["kc_middle"] = middle
        df["kc_upper"] = middle + factor * kc_atr
        df["kc_lower"] = middle - factor * kc_atr
        return df

    def bb_squeeze(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        required = ["bb_upper", "bb_lower", "kc_upper", "kc_lower"]
        for col in required:
            if col not in df.columns:
                df = self.bollinger_bands(df) if "bb_upper" not in df.columns else df
                df = self.keltner_channels(df) if "kc_upper" not in df.columns else df
        df["bb_squeeze"] = (df["bb_upper"] < df["kc_upper"]) & (df["bb_lower"] > df["kc_lower"])
        squeeze_count = []
        count = 0
        for is_squeeze in df["bb_squeeze"]:
            if is_squeeze:
                count += 1
            else:
                count = 0
            squeeze_count.append(count)
        df["bb_squeeze_candles"] = squeeze_count
        df["bb_squeeze_ready"] = (df["bb_squeeze_candles"] >= self.p.squeeze_min_candles)
        prev_squeeze = df["bb_squeeze"].shift(1)
        df["bb_squeeze_release"] = (~df["bb_squeeze"]) & prev_squeeze.fillna(False)
        return df

    def atr_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "atr" not in df.columns:
            df = self.atr(df)
        atr_mean = df["atr"].rolling(14).mean()
        atr_ratio = df["atr"] / atr_mean.replace(0, np.nan)
        conditions = [
            atr_ratio <= 1.5,
            (atr_ratio > 1.5) & (atr_ratio <= 2.0),
            (atr_ratio > 2.0) & (atr_ratio <= 3.0),
        ]
        choices = ["normal", "high_vol", "very_high_vol"]
        df["atr_regime"] = np.select(conditions, choices, default="extreme_vol")
        df["atr_ratio"] = atr_ratio
        df["atr_tradeable"] = atr_ratio <= 3.0
        return df

    def dynamic_stop_loss(self, entry_price: float, atr: float, direction: int, multiplier: float = 1.0) -> float:
        return entry_price - multiplier * atr if direction == 1 else entry_price + multiplier * atr

    def dynamic_take_profit(self, entry_price: float, stop_loss: float, rr_ratio: float, direction: int) -> float:
        risk = abs(entry_price - stop_loss)
        return entry_price + rr_ratio * risk if direction == 1 else entry_price - rr_ratio * risk

class VolumeIndicators:
    def __init__(self, params: VolumeIndicatorParams = INDICATORS.volume):
        self.p = params

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.obv(df)
        df = self.vwap(df)
        df = self.cvd(df)
        df = self.volume_analysis(df)
        df = self.volume_profile(df)
        return df

    def obv(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close = df["close"]
        volume = df["volume"]
        direction = np.where(close > close.shift(1), 1, np.where(close < close.shift(1), -1, 0))
        obv_val = (volume * direction).cumsum()
        df["obv"] = obv_val
        period = self.p.obv_signal_period
        df["obv_ema"] = pd.Series(obv_val).ewm(span=period, adjust=False).mean().values
        df["obv_bullish"] = df["obv"] > df["obv_ema"]
        lookback = 14
        price_higher = close > close.shift(lookback)
        obv_lower = df["obv"] < df["obv"].shift(lookback)
        df["obv_divergence_bearish"] = price_higher & obv_lower
        price_lower = close < close.shift(lookback)
        obv_higher = df["obv"] > df["obv"].shift(lookback)
        df["obv_divergence_bullish"] = price_lower & obv_higher
        return df

    def vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        tp_volume = typical_price * df["volume"]
        if "timestamp" in df.columns:
            df["timestamp_dt"] = pd.to_datetime(df["timestamp"])
            if self.p.vwap_anchor == "D":
                group_key = df["timestamp_dt"].dt.date
            else:
                group_key = df["timestamp_dt"].dt.isocalendar().week
            cumsum_tpv = tp_volume.groupby(group_key).cumsum()
            cumsum_vol = df["volume"].groupby(group_key).cumsum()
        else:
            cumsum_tpv = tp_volume.cumsum()
            cumsum_vol = df["volume"].cumsum()
        df["vwap"] = cumsum_tpv / cumsum_vol.replace(0, np.nan)
        df["price_above_vwap"] = df["close"] > df["vwap"]
        df["vwap_distance_pct"] = (df["close"] - df["vwap"]) / df["vwap"] * 100
        return df

    def cvd(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close = df["close"]
        hl_range = df["high"] - df["low"]
        bullish_frac = np.where(hl_range > 0, (close - df["low"]) / hl_range, 0.5)
        buy_vol = df["volume"] * bullish_frac
        sell_vol = df["volume"] * (1 - bullish_frac)
        delta = buy_vol - sell_vol
        df["cvd"] = delta.cumsum()
        df["volume_delta"] = delta
        df["cvd_bullish"] = df["cvd"] > df["cvd"].shift(1)
        df["cvd_trend"] = df["cvd"].rolling(10).mean()
        df["cvd_divergence"] = ((df["close"] > df["close"].shift(5)) & (df["cvd"] < df["cvd"].shift(5))) | ((df["close"] < df["close"].shift(5)) & (df["cvd"] > df["cvd"].shift(5)))
        return df

    def volume_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        period = self.p.volume_avg_period
        multiplier = self.p.volume_breakout_multiplier
        df["volume_avg"] = df["volume"].rolling(period).mean()
        df["volume_ratio"] = df["volume"] / df["volume_avg"].replace(0, np.nan)
        df["volume_above_avg"] = df["volume_ratio"] > multiplier
        df["volume_spike"] = df["volume_ratio"] > 2.5
        df["volume_decreasing"] = df["volume"].rolling(3).mean() < df["volume"].rolling(10).mean()
        return df

    def volume_profile(self, df: pd.DataFrame, n_bins: int = None, value_area_pct: float = 0.70) -> pd.DataFrame:
        df = df.copy()
        n_bins = n_bins or self.p.volume_profile_bins
        if len(df) < n_bins:
            df["vp_poc"] = np.nan
            df["vp_vah"] = np.nan
            df["vp_val"] = np.nan
            return df
        price_min = df["low"].min()
        price_max = df["high"].max()
        bin_edges = np.linspace(price_min, price_max, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        volume_by_bin = np.zeros(n_bins)
        for _, row in df.iterrows():
            row_low, row_high, row_vol = row["low"], row["high"], row["volume"]
            if row_high == row_low:
                bin_idx = np.searchsorted(bin_edges, row_low, side="right") - 1
                bin_idx = min(max(bin_idx, 0), n_bins - 1)
                volume_by_bin[bin_idx] += row_vol
            else:
                for j in range(n_bins):
                    overlap_low = max(row_low, bin_edges[j])
                    overlap_high = min(row_high, bin_edges[j + 1])
                    if overlap_high > overlap_low:
                        frac = (overlap_high - overlap_low) / (row_high - row_low)
                        volume_by_bin[j] += row_vol * frac
        poc_idx = np.argmax(volume_by_bin)
        poc_price = bin_centers[poc_idx]
        total_vol = volume_by_bin.sum()
        target_vol = total_vol * value_area_pct
        cum_vol = volume_by_bin[poc_idx]
        lo_idx, hi_idx = poc_idx, poc_idx
        while cum_vol < target_vol:
            can_go_up = hi_idx < n_bins - 1
            can_go_down = lo_idx > 0
            if can_go_up and can_go_down:
                if volume_by_bin[hi_idx + 1] >= volume_by_bin[lo_idx - 1]:
                    hi_idx += 1
                    cum_vol += volume_by_bin[hi_idx]
                else:
                    lo_idx -= 1
                    cum_vol += volume_by_bin[lo_idx]
            elif can_go_up:
                hi_idx += 1
                cum_vol += volume_by_bin[hi_idx]
            elif can_go_down:
                lo_idx -= 1
                cum_vol += volume_by_bin[lo_idx]
            else:
                break
        df["vp_poc"] = poc_price
        df["vp_vah"] = bin_centers[hi_idx]
        df["vp_val"] = bin_centers[lo_idx]
        df["price_in_value_area"] = (df["close"] >= df["vp_val"]) & (df["close"] <= df["vp_vah"])
        df["price_above_poc"] = df["close"] > poc_price
        return df

class MarketStructureIndicators:
    def __init__(self, params: MarketStructureParams = INDICATORS.structure):
        self.p = params

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.swing_points(df)
        df = self.market_structure(df)
        df = self.order_blocks(df)
        df = self.fair_value_gaps(df)
        df = self.liquidity_zones(df)
        return df

    def swing_points(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        lookback = self.p.swing_lookback
        n = len(df)
        swing_highs = pd.Series(False, index=df.index)
        swing_lows = pd.Series(False, index=df.index)
        swing_high_prices = pd.Series(np.nan, index=df.index)
        swing_low_prices = pd.Series(np.nan, index=df.index)
        for i in range(lookback, n - lookback):
            window_highs = df["high"].iloc[i - lookback : i + lookback + 1]
            window_lows = df["low"].iloc[i - lookback : i + lookback + 1]
            if df["high"].iloc[i] == window_highs.max():
                swing_highs.iloc[i] = True
                swing_high_prices.iloc[i] = df["high"].iloc[i]
            if df["low"].iloc[i] == window_lows.min():
                swing_lows.iloc[i] = True
                swing_low_prices.iloc[i] = df["low"].iloc[i]
        df["swing_high"] = swing_highs
        df["swing_low"] = swing_lows
        df["swing_high_price"] = swing_high_prices
        df["swing_low_price"] = swing_low_prices
        df["last_swing_high"] = df["swing_high_price"].ffill()
        df["last_swing_low"] = df["swing_low_price"].ffill()
        return df

    def market_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "last_swing_high" not in df.columns:
            df = self.swing_points(df)
        df["bos_bullish"] = df["high"] > df["last_swing_high"].shift(1)
        df["bos_bearish"] = df["low"] < df["last_swing_low"].shift(1)

        def detect_choch(df_in: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
            n = len(df_in)
            choch_bull = pd.Series(False, index=df_in.index)
            choch_bear = pd.Series(False, index=df_in.index)
            lookback = 20
            for i in range(lookback, n):
                window = df_in.iloc[i - lookback : i]
                swing_highs_in_window = window[window["swing_high"]]["swing_high_price"]
                swing_lows_in_window = window[window["swing_low"]]["swing_low_price"]
                if len(swing_highs_in_window) >= 2:
                    prev_is_bearish = swing_highs_in_window.iloc[-1] < swing_highs_in_window.iloc[0]
                    if prev_is_bearish and df_in["bos_bullish"].iloc[i]:
                        choch_bull.iloc[i] = True
                if len(swing_lows_in_window) >= 2:
                    prev_is_bullish = swing_lows_in_window.iloc[-1] > swing_lows_in_window.iloc[0]
                    if prev_is_bullish and df_in["bos_bearish"].iloc[i]:
                        choch_bear.iloc[i] = True
            return choch_bull, choch_bear

        df["choch_bullish"], df["choch_bearish"] = detect_choch(df)
        return df

    def order_blocks(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        n = len(df)
        ob_bull_high = pd.Series(np.nan, index=df.index)
        ob_bull_low = pd.Series(np.nan, index=df.index)
        ob_bear_high = pd.Series(np.nan, index=df.index)
        ob_bear_low = pd.Series(np.nan, index=df.index)
        ob_bullish = pd.Series(False, index=df.index)
        ob_bearish = pd.Series(False, index=df.index)
        lookback = self.p.order_block_lookback
        impulse_threshold = 0.005
        for i in range(2, n):
            if i >= 3:
                if all(df["close"].iloc[i - j] > df["open"].iloc[i - j] for j in range(0, 3)):
                    move_pct = (df["close"].iloc[i] - df["close"].iloc[i - 3]) / df["close"].iloc[i - 3]
                    if move_pct > impulse_threshold:
                        for k in range(i - 3, max(i - lookback, 0), -1):
                            if df["close"].iloc[k] < df["open"].iloc[k]:
                                ob_bullish.iloc[k] = True
                                ob_bull_high.iloc[k] = df["high"].iloc[k]
                                ob_bull_low.iloc[k] = df["low"].iloc[k]
                                break
                if all(df["close"].iloc[i - j] < df["open"].iloc[i - j] for j in range(0, 3)):
                    move_pct = (df["close"].iloc[i - 3] - df["close"].iloc[i]) / df["close"].iloc[i - 3]
                    if move_pct > impulse_threshold:
                        for k in range(i - 3, max(i - lookback, 0), -1):
                            if df["close"].iloc[k] > df["open"].iloc[k]:
                                ob_bearish.iloc[k] = True
                                ob_bear_high.iloc[k] = df["high"].iloc[k]
                                ob_bear_low.iloc[k] = df["low"].iloc[k]
                                break
        df["order_block_bullish"] = ob_bullish
        df["order_block_bearish"] = ob_bearish
        df["ob_bull_high"] = ob_bull_high
        df["ob_bull_low"] = ob_bull_low
        df["ob_bear_high"] = ob_bear_high
        df["ob_bear_low"] = ob_bear_low
        df["price_in_bull_ob"] = (ob_bullish.cummax() & (df["close"] >= df["ob_bull_low"].ffill()) & (df["close"] <= df["ob_bull_high"].ffill()))
        df["price_in_bear_ob"] = (ob_bearish.cummax() & (df["close"] >= df["ob_bear_low"].ffill()) & (df["close"] <= df["ob_bear_high"].ffill()))
        return df

    def fair_value_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        n = len(df)
        fvg_bull = pd.Series(False, index=df.index)
        fvg_bear = pd.Series(False, index=df.index)
        fvg_bull_top = pd.Series(np.nan, index=df.index)
        fvg_bull_bottom = pd.Series(np.nan, index=df.index)
        fvg_bear_top = pd.Series(np.nan, index=df.index)
        fvg_bear_bottom = pd.Series(np.nan, index=df.index)
        for i in range(2, n):
            high_2 = df["high"].iloc[i - 2]
            low_0 = df["low"].iloc[i]
            low_2 = df["low"].iloc[i - 2]
            high_0 = df["high"].iloc[i]
            if low_0 > high_2:
                fvg_bull.iloc[i] = True
                fvg_bull_top.iloc[i] = low_0
                fvg_bull_bottom.iloc[i] = high_2
            if high_0 < low_2:
                fvg_bear.iloc[i] = True
                fvg_bear_top.iloc[i] = low_2
                fvg_bear_bottom.iloc[i] = high_0
        df["fvg_bullish"] = fvg_bull
        df["fvg_bearish"] = fvg_bear
        df["fvg_bull_top"] = fvg_bull_top
        df["fvg_bull_bottom"] = fvg_bull_bottom
        df["fvg_bear_top"] = fvg_bear_top
        df["fvg_bear_bottom"] = fvg_bear_bottom
        df["price_in_bull_fvg"] = (fvg_bull.cummax() & (df["close"] >= df["fvg_bull_bottom"].ffill()) & (df["close"] <= df["fvg_bull_top"].ffill()))
        df["price_in_bear_fvg"] = (fvg_bear.cummax() & (df["close"] >= df["fvg_bear_bottom"].ffill()) & (df["close"] <= df["fvg_bear_top"].ffill()))
        return df

    def liquidity_zones(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "swing_high" not in df.columns:
            df = self.swing_points(df)
        threshold = self.p.liquidity_threshold_atr
        atr_approx = (df["high"] - df["low"]).rolling(14).mean()
        liquidity_above = pd.Series(False, index=df.index)
        liquidity_below = pd.Series(False, index=df.index)
        swing_high_levels = []
        swing_low_levels = []
        for i in range(len(df)):
            atr = atr_approx.iloc[i] if not pd.isna(atr_approx.iloc[i]) else 0.01 * df["close"].iloc[i]
            tol = threshold * atr
            if df["swing_high"].iloc[i]:
                swing_high_levels.append(df["high"].iloc[i])
                if len(swing_high_levels) > 20:
                    swing_high_levels.pop(0)
            if df["swing_low"].iloc[i]:
                swing_low_levels.append(df["low"].iloc[i])
                if len(swing_low_levels) > 20:
                    swing_low_levels.pop(0)
            curr_price = df["close"].iloc[i]
            highs_above = [h for h in swing_high_levels if h > curr_price]
            if len(highs_above) >= 2:
                max_h, min_h = max(highs_above), min(highs_above)
                if max_h - min_h < 2 * tol:
                    liquidity_above.iloc[i] = True
            lows_below = [l for l in swing_low_levels if l < curr_price]
            if len(lows_below) >= 2:
                max_l, min_l = max(lows_below), min(lows_below)
                if max_l - min_l < 2 * tol:
                    liquidity_below.iloc[i] = True
        df["liquidity_above"] = liquidity_above
        df["liquidity_below"] = liquidity_below
        df["equal_highs"] = (df["swing_high"] & (df["swing_high_price"] - df["swing_high_price"].shift(1)).abs() < df["swing_high_price"] * 0.001)
        df["equal_lows"] = (df["swing_low"] & (df["swing_low_price"] - df["swing_low_price"].shift(1)).abs() < df["swing_low_price"] * 0.001)
        return df

    def get_nearest_support_resistance(self, df: pd.DataFrame, current_price: float, n_levels: int = 3) -> dict:
        levels = {"resistance": [], "support": []}
        if "swing_high_price" in df.columns:
            recent_highs = df["swing_high_price"].dropna().values
            resistance = [h for h in recent_highs if h > current_price]
            support = [h for h in recent_highs if h < current_price]
            levels["resistance"].extend(sorted(resistance)[:n_levels])
            levels["support"].extend(sorted(support, reverse=True)[:n_levels])
        if "ob_bull_low" in df.columns:
            ob_supports = df["ob_bull_low"].dropna().values
            support_obs = [s for s in ob_supports if s < current_price]
            levels["support"].extend(sorted(support_obs, reverse=True)[:n_levels])
        if "ob_bear_high" in df.columns:
            ob_resist = df["ob_bear_high"].dropna().values
            resist_obs = [r for r in ob_resist if r > current_price]
            levels["resistance"].extend(sorted(resist_obs)[:n_levels])
        levels["resistance"] = sorted(set(levels["resistance"]))[:n_levels]
        levels["support"] = sorted(set(levels["support"]), reverse=True)[:n_levels]
        return levels

# ── Funciones V3/Consensus ───────────────────────────────────────────────────

def add_market_structure(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    high, low, close, open_ = df["high"], df["low"], df["close"], df["open"]
    swing_len = 5
    df["swing_high"] = (high == high.rolling(swing_len * 2 + 1, center=True).max()).astype(int)
    df["swing_low"] = (low == low.rolling(swing_len * 2 + 1, center=True).min()).astype(int)
    bearish_candle = close < open_
    bullish_move = close > close.shift(1)
    df["ob_bull"] = (bearish_candle.shift(1) & bullish_move).fillna(False)
    df["ob_bear"] = (~bearish_candle.shift(1).astype(bool) & ~bullish_move.astype(bool)).fillna(False)
    df["ob_top"]  = np.where(df["ob_bull"], high.shift(1), np.nan)
    df["ob_bot"]  = np.where(df["ob_bull"], low.shift(1),  np.nan)
    df["fvg_bull"] = (low > high.shift(2)).fillna(False)
    df["fvg_bear"] = (high < low.shift(2)).fillna(False)
    df["fvg_top"]  = np.where(df["fvg_bull"], low,         np.nan)
    df["fvg_bot"]  = np.where(df["fvg_bull"], high.shift(2), np.nan)
    prev_high = high.rolling(10).max().shift(1)
    prev_low  = low.rolling(10).min().shift(1)
    df["bos_bull"]   = (close > prev_high).fillna(False)
    df["bos_bear"]   = (close < prev_low).fillna(False)
    df["choch_bull"] = (df["bos_bull"] & df["swing_low"].shift(1).astype(bool)).fillna(False)
    df["choch_bear"] = (df["bos_bear"] & df["swing_high"].shift(1).astype(bool)).fillna(False)
    df["liq_high"] = df["swing_high"].astype(bool)
    df["liq_low"]  = df["swing_low"].astype(bool)
    return df

def add_vfi(df: pd.DataFrame, period: int = 130) -> pd.DataFrame:
    df = df.copy()
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    tp_safe = tp.clip(lower=1e-10)
    inter = np.log(tp_safe) - np.log(tp_safe.shift(1))
    vinter = inter.rolling(30).std().fillna(0.01)
    cutoff = 0.1 * vinter * df["close"]
    vave   = df["volume"].rolling(period).mean().shift(1).fillna(1)
    vmax   = vave * 2.0
    mf = tp - tp.shift(1)
    vcp = np.where(mf > cutoff,  df["volume"], np.where(mf < -cutoff, -df["volume"], 0.0))
    vf = pd.Series(vcp, index=df.index).clip(lower=-vmax, upper=vmax)
    vave_safe = vave.replace(0, np.nan)
    df["vfi"]      = vf.rolling(period).sum() / vave_safe
    df["vfi"]      = df["vfi"].fillna(0)
    df["vfi_bull"] = df["vfi"] > 0
    return df

def add_consensus(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    weights = {
        "ema_alignment_bullish": 25,
        "rsi":         20,
        "macd_bull":   20,
        "adx":         20,
        "vfi_bull":    15,
    }
    score = pd.Series(0.0, index=df.index)
    total_weight = sum(weights.values())

    if "ema_alignment_bullish" in df.columns:
        score += df["ema_alignment_bullish"].fillna(False).astype(float) * weights["ema_alignment_bullish"]
    elif "ema_bullish" in df.columns:
        score += df["ema_bullish"].fillna(False).astype(float) * weights["ema_alignment_bullish"]

    if "rsi" in df.columns:
        rsi_norm = ((df["rsi"].clip(30, 70) - 30) / 40.0)
        score += rsi_norm * weights["rsi"]

    if "macd_bull" not in df.columns:
        if "macd_line" in df.columns:
            df["macd_bull"] = df["macd_line"] > 0
        else:
            df["macd_bull"] = False
    score += df["macd_bull"].fillna(False).astype(float) * weights["macd_bull"]

    if "adx" in df.columns:
        adx_norm = df["adx"].clip(0, 50) / 50.0
        score += adx_norm * weights["adx"]

    if "vfi_bull" in df.columns:
        score += df["vfi_bull"].fillna(False).astype(float) * weights["vfi_bull"]

    df["consensus"]      = (score / total_weight * 100).clip(0, 100)
    df["consensus_bull"] = df["consensus"] >= 55
    df["consensus_bear"] = df["consensus"] <= 40
    return df

# ── Main Entrypoint ───────────────────────────────────────────────────────────

def apply_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c, h, l, o, v = df["close"], df["high"], df["low"], df["open"], df["volume"]

    # 1. Batería de indicadores individuales (clases de compatibilidad)
    df = TrendIndicators().calculate_all(df)
    df = MomentumIndicators().calculate_all(df)
    df = VolatilityIndicators().calculate_all(df)
    df = VolumeIndicators().calculate_all(df)
    df = MarketStructureIndicators().calculate_all(df)

    # 2. Mapeos de compatibilidad de nombres
    if "ema_alignment_bullish" in df.columns:
        df["ema_bullish"] = df["ema_alignment_bullish"]

    # 3. Indicadores V3/Consensus
    df = add_market_structure(df)
    df = add_vfi(df)
    df = add_consensus(df)

    # 4. Indicadores V5 (Velas + CVD + Absorción)
    df = enrich_v5(df)

    # 5. Compatibilidad V6 inline name mapping
    df["ema9"]   = _ema(c, 9)
    df["ema21"]  = _ema(c, 21)
    df["ema55"]  = _ema(c, 55)
    df["ema200"] = _ema(c, 200)

    # Asegurar ambos estilos en el df
    df["ema_9"]   = df["ema9"]
    df["ema_21"]  = df["ema21"]
    df["ema_55"]  = df["ema55"]
    df["ema_200"] = df["ema200"]

    # 6. Indicadores Probabilísticos / Matemáticos (Fase 2)
    # Regime Proxy (HMM/GMM)
    atr_series = df["atr"] if "atr" in df.columns else _atr(h, l, c)
    regime_probs = _regime_hmm_proxy(c, atr_series)
    for col in regime_probs.columns:
        df[col] = regime_probs[col]
        
    # Hurst Exponent (Trend persistence)
    df["hurst_exponent"] = _hurst_exponent(c, lags=30)
    
    # Time Series Momentum & T-Stat
    ts_slope, ts_tstat = _ts_momentum(c, period=20)
    df["ts_momentum"] = ts_slope
    df["ts_tstat"] = ts_tstat
    
    # Z-Score VWAP (Mean Reversion Arbitrage)
    df["zscore_vwap"] = _zscore_vwap(c, v, period=50)

    # Volatility GARCH proxy (Aceleración de varianza para Breakouts)
    ret = c.pct_change()
    df["vol_garch_proxy"] = _volatility_garch_proxy(ret, span=20)

    # Bollinger Bands
    df["bb_mid"] = df["bb_middle"]

    # MACD
    df["macd"] = df["macd_line"]
    df["macd_hist"] = df["macd_histogram"]

    # Volatilidad
    df["atr_pct"] = df["atr"] / c * 100
    df["atr_rank"] = df["atr_pct"].rolling(100).rank(pct=True)

    # CVD
    df["delta_vol"] = _delta_volume(o.values.astype(DTYPE), h.values.astype(DTYPE),
                                    l.values.astype(DTYPE), c.values.astype(DTYPE),
                                    v.values.astype(DTYPE))
    df["cvd_pos"] = df["delta_vol"].rolling(5).mean() > 0.15

    # 6. Atom-Compatible Quantitative Indicators
    # 6.1 Hurst Exponent (Vectorized)
    df["hurst_exp"] = _hurst_exponent(df["close"])
    
    # 6.2 Z-Score VWAP
    if "vwap" not in df.columns:
        # Fallback to simple VWMA if VWAP wasn't calculated by VolatilityIndicators
        tp = (h + l + c) / 3
        df["vwap"] = (tp * v).rolling(20).sum() / v.rolling(20).sum()
    df["zscore_vwap"] = _zscore_vwap(df["close"], df["volume"])
    
    # 6.3 Volatility GARCH Proxy (EWMA)
    returns = df["close"].pct_change().fillna(0)
    df["vol_ratio_garch"] = _volatility_garch_proxy(returns)
    
    # 6.4 Time Series Momentum (TSM)
    slope, tstat = _ts_momentum(df["close"])
    df["ts_momentum_slope"] = slope
    df["ts_momentum_tstat"] = tstat

    # OBV Accel
    df["obv_accel"] = (df["obv"] > df["obv"].rolling(10).mean()*1.005) & (df["obv"].diff(3) > 0)

    # Trend status
    df["trend_up"]   = (df["ema21"]>df["ema55"]) & (df["ema55"]>df["ema200"])
    df["trend_down"] = (df["ema21"]<df["ema55"]) & (df["ema55"]<df["ema200"])
    df["above_vwap"] = c > df["vwap"]
    df["momentum_3"] = c.pct_change(3) * 100

    return df
