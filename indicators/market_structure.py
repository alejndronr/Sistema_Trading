"""
Estructura de Mercado — Smart Money Concepts
============================================
CHoCH (Change of Character), BOS (Break of Structure),
Order Blocks, Fair Value Gaps (FVG), Liquidity Pools

Metodología Smart Money: analiza el comportamiento institucional
a través de la estructura de swing highs/lows y zonas de liquidez.

Convención de columnas:
  - swing_high / swing_low: True en velas de swing
  - swing_high_price / swing_low_price: precio del swing
  - choch_bullish / choch_bearish: Change of Character
  - bos_bullish / bos_bearish: Break of Structure
  - order_block_bullish / order_block_bearish: zonas OB
  - fvg_bullish / fvg_bearish: Fair Value Gaps activos
  - liquidity_above / liquidity_below: zonas de liquidez
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import INDICATORS, MarketStructureParams


class MarketStructureIndicators:
    """Smart Money Concepts: estructura de mercado, Order Blocks, FVG."""

    def __init__(self, params: MarketStructureParams = INDICATORS.structure):
        self.p = params

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula todos los indicadores de estructura de mercado."""
        df = self.swing_points(df)
        df = self.market_structure(df)
        df = self.order_blocks(df)
        df = self.fair_value_gaps(df)
        df = self.liquidity_zones(df)
        return df

    # ── Swing Highs y Lows ────────────────────────────────────────────────────

    def swing_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detecta Swing Highs y Swing Lows.
        Un Swing High es una vela cuyo high es el máximo de las N velas adyacentes.
        Un Swing Low es una vela cuyo low es el mínimo de las N velas adyacentes.
        """
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
            curr_high = df["high"].iloc[i]
            curr_low = df["low"].iloc[i]

            if curr_high == window_highs.max():
                swing_highs.iloc[i] = True
                swing_high_prices.iloc[i] = curr_high

            if curr_low == window_lows.min():
                swing_lows.iloc[i] = True
                swing_low_prices.iloc[i] = curr_low

        df["swing_high"] = swing_highs
        df["swing_low"] = swing_lows
        df["swing_high_price"] = swing_high_prices
        df["swing_low_price"] = swing_low_prices

        # Forward-fill para tener siempre el último swing disponible
        df["last_swing_high"] = df["swing_high_price"].ffill()
        df["last_swing_low"] = df["swing_low_price"].ffill()

        return df

    # ── CHoCH y BOS ───────────────────────────────────────────────────────────

    def market_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Break of Structure (BOS): continuación de tendencia.
          BOS Bullish: precio rompe un swing high previo en tendencia alcista.
          BOS Bearish: precio rompe un swing low previo en tendencia bajista.

        Change of Character (CHoCH): posible inversión de tendencia.
          CHoCH Bullish: en tendencia bajista, precio rompe swing high → reversión.
          CHoCH Bearish: en tendencia alcista, precio rompe swing low → reversión.
        """
        df = df.copy()
        if "last_swing_high" not in df.columns:
            df = self.swing_points(df)

        # BOS Bullish: nuevo high sobre el último swing high
        df["bos_bullish"] = df["high"] > df["last_swing_high"].shift(1)
        # BOS Bearish: nuevo low bajo el último swing low
        df["bos_bearish"] = df["low"] < df["last_swing_low"].shift(1)

        # Para CHoCH necesitamos detectar la tendencia previa
        # Aproximación: si los últimos N swings son highs decrecientes → bajista
        def detect_choch(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
            n = len(df)
            choch_bull = pd.Series(False, index=df.index)
            choch_bear = pd.Series(False, index=df.index)
            lookback = 20

            for i in range(lookback, n):
                window = df.iloc[i - lookback : i]

                # Tendencia previa bajista: swings highs decrecientes
                swing_highs_in_window = window[window["swing_high"]]["swing_high_price"]
                swing_lows_in_window = window[window["swing_low"]]["swing_low_price"]

                if len(swing_highs_in_window) >= 2:
                    prev_is_bearish = swing_highs_in_window.iloc[-1] < swing_highs_in_window.iloc[0]
                    if prev_is_bearish and df["bos_bullish"].iloc[i]:
                        choch_bull.iloc[i] = True

                if len(swing_lows_in_window) >= 2:
                    prev_is_bullish = swing_lows_in_window.iloc[-1] > swing_lows_in_window.iloc[0]
                    if prev_is_bullish and df["bos_bearish"].iloc[i]:
                        choch_bear.iloc[i] = True

            return choch_bull, choch_bear

        df["choch_bullish"], df["choch_bearish"] = detect_choch(df)
        return df

    # ── Order Blocks ──────────────────────────────────────────────────────────

    def order_blocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Order Blocks: última vela de dirección contraria antes de un movimiento impulsivo.

        Order Block Bullish (demanda): última vela bajista antes de un impulso alcista.
          → Zona donde los institucionales compraron.

        Order Block Bearish (oferta): última vela alcista antes de un impulso bajista.
          → Zona donde los institucionales vendieron.

        Los OBs son zonas de soporte/resistencia de alta probabilidad.
        """
        df = df.copy()

        n = len(df)
        ob_bull_high = pd.Series(np.nan, index=df.index)
        ob_bull_low = pd.Series(np.nan, index=df.index)
        ob_bear_high = pd.Series(np.nan, index=df.index)
        ob_bear_low = pd.Series(np.nan, index=df.index)
        ob_bullish = pd.Series(False, index=df.index)
        ob_bearish = pd.Series(False, index=df.index)

        lookback = self.p.order_block_lookback
        impulse_threshold = 0.005  # 0.5% mínimo para considerar impulso

        for i in range(2, min(n, lookback)):
            # Detectar impulso alcista: 3 velas alcistas consecutivas
            if all(df["close"].iloc[i - j] > df["open"].iloc[i - j] for j in range(0, 3)):
                move_pct = (df["close"].iloc[i] - df["close"].iloc[i - 3]) / df["close"].iloc[i - 3]
                if move_pct > impulse_threshold:
                    # Buscar la última vela bajista antes del impulso
                    for k in range(i - 3, max(i - lookback, 0), -1):
                        if df["close"].iloc[k] < df["open"].iloc[k]:
                            ob_bullish.iloc[k] = True
                            ob_bull_high.iloc[k] = df["high"].iloc[k]
                            ob_bull_low.iloc[k] = df["low"].iloc[k]
                            break

            # Detectar impulso bajista: 3 velas bajistas consecutivas
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

        # Precio dentro de un OB activo
        df["price_in_bull_ob"] = (
            ob_bullish.cummax()  # OB detectado en algún momento pasado
            & (df["close"] >= df["ob_bull_low"].ffill())
            & (df["close"] <= df["ob_bull_high"].ffill())
        )
        df["price_in_bear_ob"] = (
            ob_bearish.cummax()
            & (df["close"] >= df["ob_bear_low"].ffill())
            & (df["close"] <= df["ob_bear_high"].ffill())
        )

        return df

    # ── Fair Value Gaps (FVG) ─────────────────────────────────────────────────

    def fair_value_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fair Value Gaps (Imbalances): huecos de precio entre velas.
        Ocurren cuando hay un gap entre el high de vela[i-2] y el low de vela[i].

        FVG Bullish: high[i-2] < low[i] → hueco alcista
        FVG Bearish: low[i-2] > high[i] → hueco bajista

        Los FVGs tienden a ser "rellenados" por el precio en el futuro.
        Son zonas de soporte/resistencia de alta probabilidad.
        """
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

            # FVG Bullish: gap entre la vela de hace 2 y la vela actual
            if low_0 > high_2:
                fvg_bull.iloc[i] = True
                fvg_bull_top.iloc[i] = low_0
                fvg_bull_bottom.iloc[i] = high_2

            # FVG Bearish
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

        # Precio dentro de un FVG activo (no rellenado aún)
        df["price_in_bull_fvg"] = (
            fvg_bull.cummax()
            & (df["close"] >= df["fvg_bull_bottom"].ffill())
            & (df["close"] <= df["fvg_bull_top"].ffill())
        )
        df["price_in_bear_fvg"] = (
            fvg_bear.cummax()
            & (df["close"] >= df["fvg_bear_bottom"].ffill())
            & (df["close"] <= df["fvg_bear_top"].ffill())
        )

        return df

    # ── Liquidity Zones ────────────────────────────────────────────────────────

    def liquidity_zones(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detecta zonas de liquidez (stops acumulados).

        Liquidity Above: conjunto de swing highs próximos (zona donde hay stop-loss
          de shorts y buy stops → precio tiende a ir a buscar esa liquidez).

        Liquidity Below: swing lows próximos (zona de stop-loss de longs
          y sell stops).

        También detecta Equal Highs/Lows (niveles exactamente iguales = trampa).
        """
        df = df.copy()
        if "swing_high" not in df.columns:
            df = self.swing_points(df)

        threshold = self.p.liquidity_threshold_atr
        # ATR aproximado para definir "proximity"
        atr_approx = (df["high"] - df["low"]).rolling(14).mean()

        # Zonas de liquidez: swing highs/lows agrupados dentro de 1 ATR
        liquidity_above = pd.Series(False, index=df.index)
        liquidity_below = pd.Series(False, index=df.index)

        swing_high_levels: List[float] = []
        swing_low_levels: List[float] = []

        for i in range(len(df)):
            atr = atr_approx.iloc[i] if not pd.isna(atr_approx.iloc[i]) else 0.01 * df["close"].iloc[i]
            tol = threshold * atr

            if df["swing_high"].iloc[i]:
                swing_high_levels.append(df["high"].iloc[i])
                # Mantener solo los últimos 20 swings
                if len(swing_high_levels) > 20:
                    swing_high_levels.pop(0)

            if df["swing_low"].iloc[i]:
                swing_low_levels.append(df["low"].iloc[i])
                if len(swing_low_levels) > 20:
                    swing_low_levels.pop(0)

            curr_price = df["close"].iloc[i]

            # Liquidez arriba: hay swing highs agrupados por encima del precio actual
            highs_above = [h for h in swing_high_levels if h > curr_price]
            if len(highs_above) >= 2:
                max_h, min_h = max(highs_above), min(highs_above)
                if max_h - min_h < 2 * tol:  # Agrupados
                    liquidity_above.iloc[i] = True

            # Liquidez abajo: swing lows agrupados por debajo
            lows_below = [l for l in swing_low_levels if l < curr_price]
            if len(lows_below) >= 2:
                max_l, min_l = max(lows_below), min(lows_below)
                if max_l - min_l < 2 * tol:
                    liquidity_below.iloc[i] = True

        df["liquidity_above"] = liquidity_above
        df["liquidity_below"] = liquidity_below

        # Equal Highs / Equal Lows (dentro de tolerancia del 0.1%)
        df["equal_highs"] = (
            df["swing_high"]
            & (df["swing_high_price"] - df["swing_high_price"].shift(1)).abs()
            < df["swing_high_price"] * 0.001
        )
        df["equal_lows"] = (
            df["swing_low"]
            & (df["swing_low_price"] - df["swing_low_price"].shift(1)).abs()
            < df["swing_low_price"] * 0.001
        )

        return df

    def get_nearest_support_resistance(
        self, df: pd.DataFrame, current_price: float, n_levels: int = 3
    ) -> dict:
        """
        Retorna los niveles de soporte/resistencia más cercanos al precio actual.
        Combina swing highs/lows, Order Blocks y FVGs.
        """
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

        # Deduplicar y ordenar
        levels["resistance"] = sorted(set(levels["resistance"]))[:n_levels]
        levels["support"] = sorted(set(levels["support"]), reverse=True)[:n_levels]

        return levels
