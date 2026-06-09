"""
Filtro de Régimen de Mercado
=============================
Clasifica el mercado actual antes de cualquier operación.
Determina qué estrategia está activa según el régimen detectado.

Regímenes:
  BULLISH_TREND:   EMA21>EMA55>EMA200 en 4H + ADX>25 → Estrategia 1 activa
  BEARISH_TREND:   EMA21<EMA55<EMA200 en 4H + ADX>25 → Solo shorts (Futures) o CASH
  RANGE:           ADX<20 → Estrategia 2 activa
  HIGH_VOLATILITY: ATR>2x su media de 14 → reducir posición al 50%
  EXTREME_VOLATILITY: ATR>3x → NO operar

Además filtra por:
  - Prioridad del activo (1, 2, 3)
  - Sesiones de alta liquidez
  - Ventanas de noticias macro
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional, Tuple

import pandas as pd

from config.settings import (
    ASSETS,
    INDICATORS,
    RISK,
    SESSIONS,
    STRATEGIES,
    AssetPriority,
    MarketRegime,
    StrategyType,
)
from config.logging_config import get_logger

logger = get_logger(__name__)


class RegimeFilter:
    """
    Filtra condiciones de mercado y determina qué estrategia aplicar.
    Es el primer check antes de evaluar cualquier señal.
    """

    def detect_regime(self, df: pd.DataFrame, idx: int = -1) -> MarketRegime:
        """
        Clasifica el régimen de mercado en el índice dado.
        Por defecto usa la última vela disponible.
        """
        if idx == -1:
            idx = len(df) - 1

        row = df.iloc[idx]

        # ── Check de volatilidad extrema (máxima prioridad) ────────────────────
        atr_regime = row.get("atr_regime", "normal")
        atr_tradeable = row.get("atr_tradeable", True)

        if not atr_tradeable or atr_regime == "extreme_vol":
            return MarketRegime.HIGH_VOLATILITY

        # ── Clasificar tendencia por EMAs y ADX ───────────────────────────────
        ema21_col = f"ema_{INDICATORS.trend.ema_fast}"
        ema55_col = f"ema_{INDICATORS.trend.ema_mid}"
        ema200_col = f"ema_{INDICATORS.trend.ema_slow}"

        required = [ema21_col, ema55_col, ema200_col, "adx"]
        for col in required:
            if col not in df.columns or pd.isna(row.get(col)):
                return MarketRegime.UNKNOWN

        ema21 = row[ema21_col]
        ema55 = row[ema55_col]
        ema200 = row[ema200_col]
        adx = row["adx"]

        # Tendencia alcista fuerte
        if ema21 > ema55 > ema200 and adx > INDICATORS.trend.adx_trend_threshold:
            return MarketRegime.BULLISH_TREND

        # Tendencia bajista fuerte
        if ema21 < ema55 < ema200 and adx > INDICATORS.trend.adx_trend_threshold:
            return MarketRegime.BEARISH_TREND

        # Alta volatilidad (ATR > 2x media) pero no extrema
        if atr_regime in ("high_vol", "very_high_vol"):
            return MarketRegime.HIGH_VOLATILITY

        # Rango: ADX bajo sin tendencia clara
        if adx < INDICATORS.trend.adx_trend_threshold:
            return MarketRegime.RANGE

        return MarketRegime.UNKNOWN

    def get_active_strategies(self, regime: MarketRegime) -> List[StrategyType]:
        """
        Retorna las estrategias activas para el régimen dado.
        Implementa exactamente la lógica del prompt maestro.
        """
        if regime == MarketRegime.BULLISH_TREND:
            # Estrategia 1 activa, Estrategia 2 solo en dirección de tendencia
            return [StrategyType.TREND_FOLLOWING, StrategyType.MEAN_REVERSION]

        elif regime == MarketRegime.BEARISH_TREND:
            # Solo shorts (Futures en Fase 3+), o CASH en Spot
            # En Fase 1 (backtesting): solo short via Trend Following
            return [StrategyType.TREND_FOLLOWING]

        elif regime == MarketRegime.RANGE:
            # Estrategia 2 activa, Estrategia 1 desactivada
            return [StrategyType.MEAN_REVERSION]

        elif regime == MarketRegime.HIGH_VOLATILITY:
            # Optimización V1: Operar en alta volatilidad (el SL de 1.5 ATR ajusta el tamaño)
            return [StrategyType.TREND_FOLLOWING, StrategyType.BREAKOUT]

        return []

    def get_position_size_multiplier(self, regime: MarketRegime) -> float:
        """
        Retorna el multiplicador de tamaño de posición según el régimen.
        Optimización V1: No se penaliza por HIGH_VOLATILITY, el SL ajustado se encarga.
        """
        return 1.0

    def is_valid_trading_session(self, timestamp: Optional[datetime] = None) -> Tuple[bool, str]:
        """
        Verifica si el timestamp cae en una sesión de alta liquidez.
        Retorna (válido, razón).

        Sesiones válidas (UTC):
          - London Open:   08:00-10:00
          - New York Open: 13:00-16:00
          - Overlap:       13:00-17:00
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        hour = timestamp.hour

        # Evitar baja liquidez Asia
        if SESSIONS.low_liquidity_start <= hour < SESSIONS.low_liquidity_end:
            return False, f"Baja liquidez Asia (UTC {hour:02d}:xx)"

        # Fines de semana (spreads más altos)
        if timestamp.weekday() >= 5:  # 5=sábado, 6=domingo
            return False, "Fin de semana (spreads altos)"

        # Sesiones de alta liquidez
        london_open = SESSIONS.london_open[0] <= hour < SESSIONS.london_close[0]
        ny_open = SESSIONS.ny_open[0] <= hour < SESSIONS.ny_close[0]

        if london_open:
            return True, "London Open"
        if ny_open:
            return True, "New York Open"

        # Fuera de sesión preferida pero horario razonable
        if SESSIONS.london_close[0] <= hour < SESSIONS.ny_open[0]:
            return True, "Entre sesiones (aceptable)"

        return False, f"Fuera de sesión preferida (UTC {hour:02d}:xx)"

    def is_near_news_event(
        self, timestamp: Optional[datetime] = None, news_times: Optional[List[datetime]] = None
    ) -> bool:
        """
        Verifica si el timestamp está dentro de la ventana de noticias macro.
        En Fase 1 (backtesting) esto puede ignorarse; en Fase 2+ conectar con un
        calendario económico (ej: ForexFactory API).
        """
        if not news_times:
            return False

        from datetime import timedelta
        buffer = timedelta(minutes=SESSIONS.news_buffer_minutes)
        ts = timestamp or datetime.now(timezone.utc)

        return any(abs((ts - news_time).total_seconds()) < buffer.total_seconds() for news_time in news_times)

    def passes_asset_filter(self, symbol: str, current_regime: MarketRegime) -> Tuple[bool, str]:
        """
        Filtra activos según su prioridad y el régimen de mercado.
        Prioridad 3 solo en tendencia fuerte.
        """
        if symbol not in ASSETS:
            return False, f"Activo {symbol} no está en el universo permitido"

        asset = ASSETS[symbol]

        if asset.priority == AssetPriority.LOW:
            if current_regime not in [MarketRegime.BULLISH_TREND, MarketRegime.BEARISH_TREND]:
                return False, f"{symbol} (P3) requiere tendencia fuerte. Régimen actual: {current_regime.value}"

        return True, f"{symbol} aprobado (P{asset.priority.value})"

    def full_filter(
        self,
        df: pd.DataFrame,
        symbol: str,
        idx: int = -1,
        timestamp: Optional[datetime] = None,
    ) -> Tuple[bool, MarketRegime, str]:
        """
        Filtro completo: régimen + sesión + activo.
        Retorna (puede_operar, régimen, razón).
        """
        # 1. Detectar régimen
        regime = self.detect_regime(df, idx)

        # 2. Alta volatilidad extrema → bloquear
        if regime == MarketRegime.HIGH_VOLATILITY:
            return False, regime, "ATR extremo: no operar"

        # 3. Filtro de activo según prioridad
        asset_ok, asset_reason = self.passes_asset_filter(symbol, regime)
        if not asset_ok:
            return False, regime, asset_reason

        # 4. Sesión de trading (solo para live, en backtest se puede omitir)
        # session_ok, session_reason = self.is_valid_trading_session(timestamp)

        logger.debug(
            "regime_detected",
            symbol=symbol,
            regime=regime.value,
        )

        return True, regime, f"Régimen: {regime.value}"
