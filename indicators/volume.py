"""
Indicadores de Volumen
======================
OBV, VWAP, CVD (Cumulative Volume Delta), Volume Profile

Convención de columnas:
  - obv: On Balance Volume
  - obv_ema: EMA del OBV (señal de tendencia)
  - vwap: VWAP diario
  - cvd: Cumulative Volume Delta
  - volume_avg: volumen promedio de 20 periodos
  - volume_above_avg: True si volumen > media × multiplicador
  - vp_poc: Point of Control (precio con mayor volumen negociado)
  - vp_vah: Value Area High
  - vp_val: Value Area Low
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import INDICATORS, VolumeIndicatorParams


class VolumeIndicators:
    """OBV, VWAP, CVD, Volume Profile."""

    def __init__(self, params: VolumeIndicatorParams = INDICATORS.volume):
        self.p = params

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula todos los indicadores de volumen."""
        df = self.obv(df)
        df = self.vwap(df)
        df = self.cvd(df)
        df = self.volume_analysis(df)
        df = self.volume_profile(df)
        return df

    # ── OBV (On Balance Volume) ───────────────────────────────────────────────

    def obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        OBV acumula volumen cuando el precio sube y lo resta cuando baja.
        La tendencia del OBV confirma (o diverge de) la tendencia del precio.
        """
        df = df.copy()
        close = df["close"]
        volume = df["volume"]

        direction = np.where(
            close > close.shift(1), 1, np.where(close < close.shift(1), -1, 0)
        )
        obv = (volume * direction).cumsum()
        df["obv"] = obv

        # EMA del OBV para señal de tendencia
        period = self.p.obv_signal_period
        df["obv_ema"] = pd.Series(obv).ewm(span=period, adjust=False).mean().values

        # OBV por encima de su EMA = presión compradora dominante
        df["obv_bullish"] = df["obv"] > df["obv_ema"]

        # Divergencia OBV-precio (simplificada, usando ventana de 14 velas)
        lookback = 14
        price_higher = close > close.shift(lookback)
        obv_lower = df["obv"] < df["obv"].shift(lookback)
        df["obv_divergence_bearish"] = price_higher & obv_lower

        price_lower = close < close.shift(lookback)
        obv_higher = df["obv"] > df["obv"].shift(lookback)
        df["obv_divergence_bullish"] = price_lower & obv_higher

        return df

    # ── VWAP ──────────────────────────────────────────────────────────────────

    def vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        VWAP diario (Volume Weighted Average Price).
        Se resetea cada día (o semana si anchor='W').
        Actúa como zona de precio justo intradiario.
        """
        df = df.copy()

        # Precio típico = (H + L + C) / 3
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        tp_volume = typical_price * df["volume"]

        if "timestamp" in df.columns:
            # Determinar grupo de anclaje (día o semana)
            if self.p.vwap_anchor == "D":
                group_key = df["timestamp"].dt.date
            else:  # Weekly
                group_key = df["timestamp"].dt.isocalendar().week

            # VWAP = suma(TP*V) / suma(V) acumulado desde el inicio del período
            cumsum_tpv = tp_volume.groupby(group_key).cumsum()
            cumsum_vol = df["volume"].groupby(group_key).cumsum()
        else:
            cumsum_tpv = tp_volume.cumsum()
            cumsum_vol = df["volume"].cumsum()

        df["vwap"] = cumsum_tpv / cumsum_vol.replace(0, np.nan)

        # Precio relativo al VWAP
        df["price_above_vwap"] = df["close"] > df["vwap"]
        df["vwap_distance_pct"] = (df["close"] - df["vwap"]) / df["vwap"] * 100
        return df

    # ── CVD (Cumulative Volume Delta) ─────────────────────────────────────────

    def cvd(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        CVD = diferencia entre volumen comprador y vendedor acumulado.

        Aproximación con datos OHLCV (sin datos de tape real):
        - Si close > open: volumen comprador
        - Si close < open: volumen vendedor
        - Si close == open: dividir 50/50

        Para datos de tick reales (Fase 2+), esto se reemplaza con el delta real.
        """
        df = df.copy()
        close = df["close"]
        open_ = df["open"]
        volume = df["volume"]
        hl_range = df["high"] - df["low"]

        # Fracción compradora estimada (Tick Rule aproximado)
        bullish_frac = np.where(
            hl_range > 0,
            (close - df["low"]) / hl_range,
            0.5,
        )
        buy_vol = volume * bullish_frac
        sell_vol = volume * (1 - bullish_frac)

        delta = buy_vol - sell_vol
        df["cvd"] = delta.cumsum()
        df["volume_delta"] = delta  # Delta de cada vela

        # CVD creciente = presión compradora dominante
        df["cvd_bullish"] = df["cvd"] > df["cvd"].shift(1)
        df["cvd_trend"] = df["cvd"].rolling(10).mean()
        df["cvd_divergence"] = (
            (df["close"] > df["close"].shift(5)) & (df["cvd"] < df["cvd"].shift(5))
        ) | (
            (df["close"] < df["close"].shift(5)) & (df["cvd"] > df["cvd"].shift(5))
        )

        return df

    # ── Volume Analysis ────────────────────────────────────────────────────────

    def volume_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Análisis básico de volumen: media, picos, volumen relativo.
        """
        df = df.copy()
        period = self.p.volume_avg_period
        multiplier = self.p.volume_breakout_multiplier

        df["volume_avg"] = df["volume"].rolling(period).mean()
        df["volume_ratio"] = df["volume"] / df["volume_avg"].replace(0, np.nan)
        df["volume_above_avg"] = df["volume_ratio"] > multiplier
        df["volume_spike"] = df["volume_ratio"] > 2.5  # Spike extremo

        # Volumen decreciente en tendencia (posible agotamiento)
        df["volume_decreasing"] = df["volume"].rolling(3).mean() < df["volume"].rolling(10).mean()

        return df

    # ── Volume Profile ────────────────────────────────────────────────────────

    def volume_profile(
        self,
        df: pd.DataFrame,
        n_bins: int = None,
        value_area_pct: float = 0.70,
    ) -> pd.DataFrame:
        """
        Volume Profile simplificado para datos OHLCV.
        Calcula POC, VAH y VAL para la ventana completa de datos.

        Para un Volume Profile dinámico (rolling), ver volume_profile_rolling().

        POC (Point of Control): nivel de precio con mayor volumen negociado
        VAH (Value Area High): límite superior del 70% del volumen
        VAL (Value Area Low):  límite inferior del 70% del volumen
        """
        df = df.copy()
        n_bins = n_bins or self.p.volume_profile_bins

        if len(df) < n_bins:
            df["vp_poc"] = np.nan
            df["vp_vah"] = np.nan
            df["vp_val"] = np.nan
            return df

        # Construir el perfil con todos los datos del DataFrame
        price_min = df["low"].min()
        price_max = df["high"].max()
        bin_edges = np.linspace(price_min, price_max, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Distribuir el volumen de cada vela proporcionalmente a los bins
        volume_by_bin = np.zeros(n_bins)
        for _, row in df.iterrows():
            row_low, row_high, row_vol = row["low"], row["high"], row["volume"]
            if row_high == row_low:
                # Toda la vela en un bin
                bin_idx = np.searchsorted(bin_edges, row_low, side="right") - 1
                bin_idx = min(max(bin_idx, 0), n_bins - 1)
                volume_by_bin[bin_idx] += row_vol
            else:
                # Distribuir proporcionalmente
                for j in range(n_bins):
                    overlap_low = max(row_low, bin_edges[j])
                    overlap_high = min(row_high, bin_edges[j + 1])
                    if overlap_high > overlap_low:
                        frac = (overlap_high - overlap_low) / (row_high - row_low)
                        volume_by_bin[j] += row_vol * frac

        # POC: bin con mayor volumen
        poc_idx = np.argmax(volume_by_bin)
        poc_price = bin_centers[poc_idx]

        # Value Area: 70% del volumen total centrado en el POC
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

        vah = bin_centers[hi_idx]
        val = bin_centers[lo_idx]

        # Asignar el mismo valor a todas las filas (perfil global del período)
        df["vp_poc"] = poc_price
        df["vp_vah"] = vah
        df["vp_val"] = val

        # Precio relativo al Value Area
        df["price_in_value_area"] = (df["close"] >= val) & (df["close"] <= vah)
        df["price_above_poc"] = df["close"] > poc_price

        return df

    def volume_profile_rolling(
        self,
        df: pd.DataFrame,
        window: int = 100,
    ) -> pd.DataFrame:
        """
        Volume Profile dinámico: recalcula el perfil en cada ventana de N velas.
        Más costoso computacionalmente pero más preciso para backtesting.
        """
        df = df.copy()
        poc_list = []
        vah_list = []
        val_list = []

        for i in range(len(df)):
            if i < window:
                poc_list.append(np.nan)
                vah_list.append(np.nan)
                val_list.append(np.nan)
                continue

            window_df = df.iloc[i - window : i].copy()
            result = self.volume_profile(window_df)
            poc_list.append(result["vp_poc"].iloc[-1])
            vah_list.append(result["vp_vah"].iloc[-1])
            val_list.append(result["vp_val"].iloc[-1])

        df["vp_poc_rolling"] = poc_list
        df["vp_vah_rolling"] = vah_list
        df["vp_val_rolling"] = val_list
        return df
