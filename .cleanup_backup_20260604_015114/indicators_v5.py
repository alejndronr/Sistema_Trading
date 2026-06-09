"""
indicators/indicators_v5.py — Capa de indicadores V5
=====================================================
Extiende enrich_dataframe() del live_engine V4 con:

  · 18 patrones de velas japonesas (numpy puro, sin TA-Lib)
  · 8 figuras chartistas (H&S, doble suelo/techo, flags, triángulos, cuña)
  · S/R dinámico por clustering manual (sin sklearn)
  · Fibonacci automático (retroceso + extensión)
  · BOS / CHoCH mejorado
  · Order Flow proxy (delta volumen, absorción)
  · OBV vectorizado 100% (corrige bug de la propuesta)

Diseño para Atom E3950:
  · float32 donde es posible
  · Sin bucles Python en el path caliente
  · Sin sklearn, sin TA-Lib
  · Compatible con el live_engine V4 existente

Uso:
    from indicators.indicators_v5 import enrich_v5, detect_candle_patterns,
         detect_chart_patterns, find_sr_zones, fibonacci_levels
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

DTYPE = np.float32


# ══════════════════════════════════════════════════════════════════════════════
# ── Helpers vectorizados (todos sin bucles Python) ────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _obv_vectorized(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """
    OBV 100% vectorizado — corrige el bucle for de la propuesta original.
    np.sign + np.cumsum ejecutan en C puro (~1000x más rápido en Atom).
    """
    direction = np.sign(np.diff(close.astype(DTYPE), prepend=close[0]))
    return np.cumsum(direction * volume.astype(DTYPE))


def _delta_volume(open_: np.ndarray, high: np.ndarray,
                  low: np.ndarray, close: np.ndarray,
                  volume: np.ndarray) -> np.ndarray:
    """
    Proxy de delta de volumen (order flow aproximado sin datos de tick).
    Buying pressure: fracción de la vela que cierra arriba del rango.
    Delta = buying_pressure - selling_pressure ∈ [-1, 1]
    """
    rng  = np.where(high == low, 1e-10, high - low)
    buy  = (close - low)  / rng   # fracción alcista
    sell = (high  - close) / rng  # fracción bajista
    return (buy - sell).astype(DTYPE)


def _absorption(high: np.ndarray, low: np.ndarray,
                volume: np.ndarray, period: int = 20) -> np.ndarray:
    """
    Absorción: volumen alto con rango pequeño = institucionales absorbiendo.
    Vectorizado con rolling numpy.
    """
    rng      = (high - low).astype(DTYPE)
    avg_rng  = pd.Series(rng).rolling(period, min_periods=1).mean().values
    avg_vol  = pd.Series(volume.astype(DTYPE)).rolling(period, min_periods=1).mean().values
    return ((volume > avg_vol * 1.8) & (rng < avg_rng * 0.6)).astype(np.int8)


def _find_swing_points(high: np.ndarray, low: np.ndarray,
                       left: int = 5, right: int = 5) -> Tuple[List, List]:
    """
    Swing highs/lows sin lookahead bias — usa solo datos hasta el índice actual.
    IMPORTANTE: devuelve puntos con right-bars de delay (no puede ver el futuro).
    Compatible con backtesting y live (el punto se confirma right velas después).
    """
    n = len(high)
    swing_highs: List[Dict] = []
    swing_lows:  List[Dict] = []

    # El rango va hasta n-right para no usar datos futuros en la confirmación
    for i in range(left, n - right):
        win_h = high[i - left : i + right + 1]
        win_l = low[i  - left : i + right + 1]
        if high[i] == win_h.max():
            swing_highs.append({"index": i, "price": float(high[i])})
        if low[i]  == win_l.min():
            swing_lows.append({"index": i, "price": float(low[i])})

    return swing_highs, swing_lows


def _linear_slope(y: np.ndarray) -> float:
    """Pendiente de regresión lineal simple (mínimos cuadrados, vectorizado)."""
    n = len(y)
    if n < 2:
        return 0.0
    x   = np.arange(n, dtype=np.float32)
    num = n * np.dot(x, y) - x.sum() * y.sum()
    den = n * np.dot(x, x) - x.sum() ** 2
    return float(num / den) if den != 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# ── 18 Patrones de velas (numpy puro) ─────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

class CandlePatterns:
    """
    18 patrones de velas japonesas implementados sin TA-Lib ni bucles.
    Todos los métodos operan sobre arrays completos (vectorizados).
    Devuelven Series de scores (0 = no detectado, >0 = score del patrón).
    """

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
        """
        Detecta todos los patrones y los añade como columnas al DataFrame.
        Retorna el DataFrame con columnas 'cp_*' añadidas.
        """
        o, h, l, c = df["open"], df["high"], df["low"], df["close"]
        body       = cls._body(o, c)
        rng        = cls._rng(h, l)
        upper      = cls._upper_shadow(o, h, c)
        lower      = cls._lower_shadow(o, l, c)
        bull       = c > o
        bear       = c < o

        body_pct = body / rng   # fracción del cuerpo respecto al rango

        # ── Reversión alcista ─────────────────────────────────────────────────

        # Hammer: cuerpo pequeño arriba, mecha inferior larga
        df["cp_hammer"] = (
            (body_pct < 0.35) &
            (lower > 2 * body) &
            (upper < body) &
            bear.shift(1).fillna(False)  # contexto bajista previo
        ).astype(np.int8) * 15

        # Inverted Hammer
        df["cp_inv_hammer"] = (
            (body_pct < 0.35) &
            (upper > 2 * body) &
            (lower < body) &
            bear.shift(1).fillna(False)
        ).astype(np.int8) * 12

        # Bullish Engulfing
        df["cp_engulf_bull"] = (
            bear.shift(1).fillna(False) & bull &
            (c > o.shift(1)) & (o < c.shift(1))
        ).astype(np.int8) * 15

        # Morning Star (3 velas)
        doji_mid = body_pct.shift(1) < 0.30
        df["cp_morning_star"] = (
            bear.shift(2).fillna(False) &
            doji_mid.fillna(False) &
            bull &
            (c > ((o.shift(2) + c.shift(2)) / 2))
        ).astype(np.int8) * 18

        # Piercing Line
        df["cp_piercing"] = (
            bear.shift(1).fillna(False) & bull &
            (c > (o.shift(1) + c.shift(1)) / 2) &
            (c < o.shift(1))
        ).astype(np.int8) * 14

        # Bullish Harami
        df["cp_harami_bull"] = (
            bear.shift(1).fillna(False) & bull &
            (o > c.shift(1)) & (c < o.shift(1))
        ).astype(np.int8) * 10

        # Tweezer Bottom
        df["cp_tweezer_bot"] = (
            bear.shift(1).fillna(False) & bull &
            ((l - l.shift(1)).abs() / l.shift(1) < 0.002)
        ).astype(np.int8) * 12

        # Three White Soldiers
        df["cp_3white"] = (
            bull & bull.shift(1).fillna(False) & bull.shift(2).fillna(False) &
            (c > c.shift(1)) & (c.shift(1) > c.shift(2))
        ).astype(np.int8) * 20

        # ── Reversión bajista ─────────────────────────────────────────────────

        # Shooting Star
        df["cp_shooting_star"] = (
            (body_pct < 0.35) &
            (upper > 2 * body) &
            (lower < body) &
            bull.shift(1).fillna(False)
        ).astype(np.int8) * 15

        # Hanging Man
        df["cp_hanging_man"] = (
            (body_pct < 0.35) &
            (lower > 2 * body) &
            (upper < body) &
            bull.shift(1).fillna(False)
        ).astype(np.int8) * 15

        # Bearish Engulfing
        df["cp_engulf_bear"] = (
            bull.shift(1).fillna(False) & bear &
            (c < o.shift(1)) & (o > c.shift(1))
        ).astype(np.int8) * 15

        # Evening Star (3 velas)
        df["cp_evening_star"] = (
            bull.shift(2).fillna(False) &
            doji_mid.fillna(False) &
            bear &
            (c < ((o.shift(2) + c.shift(2)) / 2))
        ).astype(np.int8) * 18

        # Dark Cloud Cover
        df["cp_dark_cloud"] = (
            bull.shift(1).fillna(False) & bear &
            (c < (o.shift(1) + c.shift(1)) / 2) &
            (c > o.shift(1))
        ).astype(np.int8) * 14

        # Bearish Harami
        df["cp_harami_bear"] = (
            bull.shift(1).fillna(False) & bear &
            (o < c.shift(1)) & (c > o.shift(1))
        ).astype(np.int8) * 10

        # Three Black Crows
        df["cp_3black"] = (
            bear & bear.shift(1).fillna(False) & bear.shift(2).fillna(False) &
            (c < c.shift(1)) & (c.shift(1) < c.shift(2))
        ).astype(np.int8) * 20

        # ── Continuación / Indecisión ─────────────────────────────────────────

        # Doji
        df["cp_doji"] = (body_pct < 0.10).astype(np.int8) * 8

        # Marubozu alcista (sin mechas)
        df["cp_marubozu_bull"] = (
            bull & (body_pct > 0.90)
        ).astype(np.int8) * 10

        # Marubozu bajista
        df["cp_marubozu_bear"] = (
            bear & (body_pct > 0.90)
        ).astype(np.int8) * 10

        # ── Score acumulado de velas ───────────────────────────────────────────
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

    @staticmethod
    def context_multiplier(pattern_score: float, regime: str,
                           direction: str) -> float:
        """
        Amplifica o reduce el score según la confluencia régimen-dirección-patrón.
        Un patrón alcista en régimen BULL vale 1.3×, en BEAR vale 0.6×.
        """
        if direction == "long":
            if regime == "BULL":
                return 1.3
            elif regime == "BEAR":
                return 0.6
            else:
                return 1.0
        else:  # short
            if regime == "BEAR":
                return 1.3
            elif regime == "BULL":
                return 0.6
            else:
                return 1.0


# ══════════════════════════════════════════════════════════════════════════════
# ── Figuras chartistas ────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

class ChartPatterns:
    """
    8 figuras chartistas detectadas algorítmicamente.
    Opera sobre los swing points — sin lookahead bias porque los swing points
    ya vienen con el delay de confirmación (right=5 velas).
    """

    def __init__(self, tol: float = 0.025):
        self.tol = tol  # 2.5% de tolerancia para igualdad de precios

    def _near(self, a: float, b: float) -> bool:
        return abs(a - b) / max(abs(b), 1e-10) <= self.tol

    def head_and_shoulders(
        self, swing_highs: List[Dict]
    ) -> List[Dict]:
        """H&S bajista: 3 picos con el central más alto."""
        results = []
        if len(swing_highs) < 3:
            return results
        for i in range(len(swing_highs) - 2):
            ls, head, rs = (swing_highs[i]["price"],
                            swing_highs[i+1]["price"],
                            swing_highs[i+2]["price"])
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

    def inv_head_and_shoulders(
        self, swing_lows: List[Dict]
    ) -> List[Dict]:
        """H&S invertido alcista."""
        results = []
        if len(swing_lows) < 3:
            return results
        for i in range(len(swing_lows) - 2):
            ls, head, rs = (swing_lows[i]["price"],
                            swing_lows[i+1]["price"],
                            swing_lows[i+2]["price"])
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
        """Doble suelo (W): 2 mínimos similares."""
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
        """Doble techo (M): 2 máximos similares."""
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

    def triangle(
        self, swing_highs: List[Dict], swing_lows: List[Dict], n: int = 5
    ) -> List[Dict]:
        """
        Triángulos: simétrico, ascendente, descendente, cuña.
        Detecta por pendientes de trendlines de máximos y mínimos.
        """
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
                "score": 12, "breakout_imminent": True,
                "idx": swing_highs[-1]["index"],
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

    def flag_pattern(
        self,
        close: np.ndarray,
        swing_highs: List[Dict],
        swing_lows:  List[Dict],
        lookback:    int = 20,
    ) -> List[Dict]:
        """
        Flag alcista/bajista: impulso fuerte + consolidación en canal.
        Detecta el mástil y la bandera por pendientes.
        """
        results = []
        if len(close) < lookback + 5:
            return results

        # Impulso reciente (últimas 5 velas)
        impulse = (close[-1] - close[-6]) / close[-6] * 100

        if impulse > 3.0:  # Mástil alcista fuerte
            # Verificar consolidación (bandera): pendiente suave negativa
            recent = close[-lookback:]
            slope  = _linear_slope(recent.astype(DTYPE))
            if -0.005 < slope < 0:
                target = close[-1] + abs(close[-6] - close[-11])
                results.append({
                    "name": "BULL_FLAG", "direction": "bull",
                    "score": 18, "target": target,
                    "idx": len(close) - 1,
                })
        elif impulse < -3.0:  # Mástil bajista
            recent = close[-lookback:]
            slope  = _linear_slope(recent.astype(DTYPE))
            if 0 < slope < 0.005:
                results.append({
                    "name": "BEAR_FLAG", "direction": "bear",
                    "score": 18, "idx": len(close) - 1,
                })

        return results

    def scan_all(
        self,
        df: pd.DataFrame,
        swing_highs: List[Dict],
        swing_lows:  List[Dict],
    ) -> List[Dict]:
        """Ejecuta todos los detectores y retorna la lista unificada."""
        c = df["close"].values
        all_patterns: List[Dict] = []
        all_patterns += self.head_and_shoulders(swing_highs)
        all_patterns += self.inv_head_and_shoulders(swing_lows)
        all_patterns += self.double_bottom(swing_lows)
        all_patterns += self.double_top(swing_highs)
        all_patterns += self.triangle(swing_highs, swing_lows)
        all_patterns += self.flag_pattern(c, swing_highs, swing_lows)
        return all_patterns


# ══════════════════════════════════════════════════════════════════════════════
# ── Soporte / Resistencia dinámico ────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

class SRZones:
    """
    S/R dinámico por clustering manual (sin sklearn/DBSCAN).
    Agrupa swing points dentro del cluster_pct% en zonas.
    """

    def __init__(self, min_touches: int = 2, cluster_pct: float = 0.003):
        self.min_touches  = min_touches
        self.cluster_pct  = cluster_pct

    def find(
        self,
        swing_highs: List[Dict],
        swing_lows:  List[Dict],
        current_price: float,
    ) -> List[Dict]:
        all_pts = sorted(
            [(p["price"], "high") for p in swing_highs] +
            [(p["price"], "low")  for p in swing_lows],
            key=lambda x: x[0],
        )
        if len(all_pts) < self.min_touches:
            return []

        # Agrupar puntos dentro del cluster_pct%
        clusters: List[List] = []
        current: List       = [all_pts[0]]
        for i in range(1, len(all_pts)):
            if (abs(all_pts[i][0] - current[-1][0]) / max(current[-1][0], 1e-10)
                    <= self.cluster_pct):
                current.append(all_pts[i])
            else:
                if len(current) >= self.min_touches:
                    clusters.append(current)
                current = [all_pts[i]]
        if len(current) >= self.min_touches:
            clusters.append(current)

        zones: List[Dict] = []
        for cluster in clusters:
            prices  = [p[0] for p in cluster]
            center  = float(np.mean(prices))
            top     = float(max(prices))
            bottom  = float(min(prices))
            touches = len(cluster)
            strength = min(touches / 6.0, 1.0)

            # Tipo: soporte si precio sobre la zona, resistencia si debajo
            if current_price > top * 1.005:
                zone_type = "support"
            elif current_price < bottom * 0.995:
                zone_type = "resistance"
            else:
                highs_n = sum(1 for p in cluster if p[1] == "high")
                zone_type = "resistance" if highs_n > touches / 2 else "support"

            zones.append({
                "center":   center,
                "top":      top,
                "bottom":   bottom,
                "touches":  touches,
                "strength": strength,
                "type":     zone_type,
            })

        return sorted(zones, key=lambda z: abs(z["center"] - current_price))

    def nearest(self, zones: List[Dict], price: float, n: int = 3) -> List[Dict]:
        return zones[:n]

    def at_zone(
        self, zones: List[Dict], price: float, tol: float = 0.005
    ) -> Tuple[bool, Optional[Dict]]:
        for z in zones:
            if z["bottom"] * (1 - tol) <= price <= z["top"] * (1 + tol):
                return True, z
        return False, None

    def score_for_zone(self, zone: Dict, direction: str) -> int:
        """Score adicional si el precio está en confluencia S/R."""
        base = int(zone["strength"] * 15)
        if (direction == "long"  and zone["type"] == "support") or \
           (direction == "short" and zone["type"] == "resistance"):
            return base + 5
        return 0


# ══════════════════════════════════════════════════════════════════════════════
# ── Fibonacci ─────────────────────────────────────────────────────════════────
# ══════════════════════════════════════════════════════════════════════════════

class Fibonacci:
    """Niveles de Fibonacci para retroceso y extensión."""

    RETRACE_LEVELS  = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    EXTENSION_LEVELS= [1.0, 1.272, 1.618, 2.0, 2.618]

    def retracement(self, high: float, low: float) -> Dict[str, float]:
        diff = high - low
        return {f"{r:.3f}": high - r * diff for r in self.RETRACE_LEVELS}

    def extension(self, low: float, high: float) -> Dict[str, float]:
        diff = high - low
        return {f"{e:.3f}": high + (e - 1.0) * diff for e in self.EXTENSION_LEVELS}

    def nearest_level(
        self, price: float, levels: Dict[str, float], tol: float = 0.005
    ) -> Optional[Tuple[str, float]]:
        """Retorna el nivel Fibonacci más cercano al precio, si está en rango."""
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

    def tp_levels(
        self, entry: float, stop: float, direction: str
    ) -> Tuple[float, float, float]:
        """
        Calcula TP1, TP2, TP3 basados en extensiones Fibonacci del movimiento.
        """
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


# ══════════════════════════════════════════════════════════════════════════════
# ── BOS / CHoCH mejorado ─────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

class MarketStructure:
    """
    Break of Structure (BOS) y Change of Character (CHoCH).
    Detecta cambios de sesgo en tiempo real sin lookahead bias.
    """

    def analyze(
        self,
        swing_highs: List[Dict],
        swing_lows:  List[Dict],
        close: np.ndarray,
    ) -> Dict:
        """
        Retorna el sesgo actual (bull/bear/neutral) y los últimos eventos
        de estructura. Solo usa datos hasta el último índice disponible.
        """
        events: List[Dict] = []
        bias = "neutral"

        if not swing_highs or not swing_lows:
            return {"bias": bias, "events": events, "score": 0}

        last_sh = swing_highs[-1]["price"]
        last_sl = swing_lows[-1]["price"]
        prev_sh = swing_highs[-2]["price"] if len(swing_highs) > 1 else last_sh
        prev_sl = swing_lows[-2]["price"]  if len(swing_lows)  > 1 else last_sl
        current = float(close[-1])

        # BOS alcista: precio rompe el último swing high
        if current > last_sh:
            events.append({"type": "BOS_BULL", "score": 18, "level": last_sh})
            bias = "bull"

        # BOS bajista
        elif current < last_sl:
            events.append({"type": "BOS_BEAR", "score": 18, "level": last_sl})
            bias = "bear"

        # CHoCH alcista: después de BOS bajista, precio rompe máximo anterior
        if current > prev_sh and last_sl < prev_sl:
            events.append({"type": "CHOCH_BULL", "score": 22, "level": prev_sh})
            bias = "bull"

        # CHoCH bajista
        elif current < prev_sl and last_sh > prev_sh:
            events.append({"type": "CHOCH_BEAR", "score": 22, "level": prev_sl})
            bias = "bear"

        total_score = sum(e["score"] for e in events)
        return {"bias": bias, "events": events, "score": total_score}


# ══════════════════════════════════════════════════════════════════════════════
# ── enrich_v5: extiende enrich_dataframe del V4 ──────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def enrich_v5(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade todas las columnas V5 al DataFrame ya enriquecido por enrich_dataframe().
    Diseñado para ser llamado DESPUÉS de enrich_dataframe().

    Nuevas columnas añadidas:
      obv              — On-Balance Volume vectorizado
      delta_vol        — Proxy de order flow (-1 a +1)
      absorption       — Detección de absorción institucional
      cp_*             — 18 patrones de velas (scores)
      cp_bull_score    — Score alcista acumulado de velas
      cp_bear_score    — Score bajista acumulado de velas
      cp_net_score     — Score neto de velas
    """
    o = df["open"].values.astype(DTYPE)
    h = df["high"].values.astype(DTYPE)
    l = df["low"].values.astype(DTYPE)
    c = df["close"].values.astype(DTYPE)
    v = df["volume"].values.astype(DTYPE)

    # ── Order Flow ────────────────────────────────────────────────────────────
    df["obv"]        = _obv_vectorized(c, v)
    df["delta_vol"]  = _delta_volume(o, h, l, c, v)
    df["absorption"] = _absorption(h, l, v)

    # OBV momentum (divergencia simple)
    df["obv_rising"] = df["obv"] > df["obv"].shift(3)

    # CVD acumulado (proxy institucional)
    df["cvd_cumul"]  = df["delta_vol"].cumsum()
    df["cvd_rising"] = df["cvd_cumul"] > df["cvd_cumul"].shift(5)

    # ── Patrones de velas ─────────────────────────────────────────────────────
    df = CandlePatterns.detect_all(df)

    return df


def get_structure_context(
    df: pd.DataFrame,
    lookback: int = 100,
) -> Dict:
    """
    Calcula el contexto de estructura de mercado para las últimas `lookback` velas.
    Retorna swing points, zonas S/R, sesgo BOS/CHoCH y niveles Fibonacci.

    IMPORTANTE: Esta función opera sobre el slice df[-lookback:] para evitar
    lookahead bias. El resultado es válido SOLO para el momento actual (última vela).
    """
    window = df.iloc[-lookback:].copy()

    h_arr = window["high"].values
    l_arr = window["low"].values
    c_arr = window["close"].values
    price = float(c_arr[-1])

    # Swing points (con delay de confirmación de 5 velas — sin lookahead)
    swing_highs, swing_lows = _find_swing_points(h_arr, l_arr, left=5, right=5)

    # Estructura de mercado
    ms = MarketStructure()
    structure = ms.analyze(swing_highs, swing_lows, c_arr)

    # Zonas S/R
    sr = SRZones(min_touches=2, cluster_pct=0.003)
    zones = sr.find(swing_highs, swing_lows, price)
    at_zone, current_zone = sr.at_zone(zones, price)

    # Fibonacci desde último swing alto/bajo
    fib = Fibonacci()
    fib_retrace  = {}
    fib_extend   = {}
    fib_tp_levels = (0.0, 0.0, 0.0)

    if swing_highs and swing_lows:
        last_high = swing_highs[-1]["price"]
        last_low  = swing_lows[-1]["price"]
        fib_retrace = fib.retracement(last_high, last_low)
        fib_extend  = fib.extension(last_low, last_high)

    # Figuras chartistas
    cp = ChartPatterns(tol=0.025)
    chart_patterns = cp.scan_all(window, swing_highs, swing_lows)

    return {
        "swing_highs":    swing_highs,
        "swing_lows":     swing_lows,
        "structure":      structure,        # bias, events, score
        "zones":          zones,            # lista de zonas S/R
        "at_zone":        at_zone,
        "current_zone":   current_zone,
        "fib_retrace":    fib_retrace,
        "fib_extend":     fib_extend,
        "chart_patterns": chart_patterns,
        "price":          price,
        "sr_helper":      sr,
        "fib_helper":     fib,
    }
