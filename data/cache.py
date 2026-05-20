"""
In-Memory Cache — Datos en RAM para el Engine Live
===================================================
Para Fase 2+ (live trading): mantiene los últimos N candles en RAM
para evitar lecturas constantes del HDD mecánico del ZimaBlade.

En Fase 1 (backtesting): no es crítico, pero se usa para
no recalcular indicadores en cada iteración del loop de backtesting.
"""

from __future__ import annotations

import threading
from collections import deque
from datetime import datetime
from typing import Deque, Dict, Optional

import pandas as pd

from config.logging_config import get_logger
from config.settings import TIMEFRAME_MINUTES

logger = get_logger(__name__)


class OHLCVCache:
    """
    Cache thread-safe de datos OHLCV en memoria.
    Usa deques con maxlen para mantener solo las últimas N velas.
    """

    # Número de velas a mantener en cache por par/timeframe
    # Suficiente para calcular todos los indicadores incluyendo EMA 200
    DEFAULT_MAX_CANDLES = 500

    def __init__(self, max_candles: int = DEFAULT_MAX_CANDLES):
        self.max_candles = max_candles
        self._cache: Dict[str, Dict[str, Deque]] = {}
        self._lock = threading.RLock()
        self._last_update: Dict[str, datetime] = {}

    def _make_key(self, symbol: str, timeframe: str) -> tuple:
        return (symbol, timeframe)

    def update(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        """Carga o actualiza el cache con un DataFrame completo."""
        key = self._make_key(symbol, timeframe)

        with self._lock:
            if key not in self._cache:
                self._cache[key] = {
                    col: deque(maxlen=self.max_candles)
                    for col in ["timestamp", "open", "high", "low", "close", "volume"]
                }

            # Cargar solo las últimas max_candles filas
            tail = df.tail(self.max_candles)
            for col in ["timestamp", "open", "high", "low", "close", "volume"]:
                self._cache[key][col].clear()
                self._cache[key][col].extend(tail[col].tolist())

            self._last_update[str(key)] = datetime.utcnow()

        logger.debug(
            "cache_updated",
            symbol=symbol,
            timeframe=timeframe,
            candles=len(df.tail(self.max_candles)),
        )

    def append_candle(
        self,
        symbol: str,
        timeframe: str,
        timestamp: datetime,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ) -> None:
        """Agrega una sola vela nueva al cache (para WebSocket live)."""
        key = self._make_key(symbol, timeframe)

        with self._lock:
            if key not in self._cache:
                self._cache[key] = {
                    col: deque(maxlen=self.max_candles)
                    for col in ["timestamp", "open", "high", "low", "close", "volume"]
                }

            cache = self._cache[key]

            # Si la última vela tiene el mismo timestamp, actualizar (vela en formación)
            if cache["timestamp"] and cache["timestamp"][-1] == timestamp:
                cache["high"][-1] = max(cache["high"][-1], high)
                cache["low"][-1] = min(cache["low"][-1], low)
                cache["close"][-1] = close
                cache["volume"][-1] = volume
            else:
                cache["timestamp"].append(timestamp)
                cache["open"].append(open_)
                cache["high"].append(high)
                cache["low"].append(low)
                cache["close"].append(close)
                cache["volume"].append(volume)

    def get_dataframe(
        self, symbol: str, timeframe: str, n_candles: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """
        Retorna los datos del cache como DataFrame.

        Args:
            n_candles: Si se especifica, retorna solo las últimas N velas.
        """
        key = self._make_key(symbol, timeframe)

        with self._lock:
            if key not in self._cache or not self._cache[key]["timestamp"]:
                return None

            data = {col: list(self._cache[key][col]) for col in ["timestamp", "open", "high", "low", "close", "volume"]}

        df = pd.DataFrame(data)

        if n_candles:
            df = df.tail(n_candles).reset_index(drop=True)

        return df

    def is_warm(self, symbol: str, timeframe: str, min_candles: int = 200) -> bool:
        """Verifica si el cache tiene suficientes velas para calcular indicadores."""
        key = self._make_key(symbol, timeframe)
        with self._lock:
            if key not in self._cache:
                return False
            return len(self._cache[key]["timestamp"]) >= min_candles

    def get_latest_price(self, symbol: str, timeframe: str) -> Optional[float]:
        """Retorna el último precio de cierre disponible."""
        df = self.get_dataframe(symbol, timeframe, n_candles=1)
        if df is not None and not df.empty:
            return float(df["close"].iloc[-1])
        return None

    def clear(self, symbol: Optional[str] = None, timeframe: Optional[str] = None) -> None:
        """Limpia el cache — todo, o solo un par/timeframe específico."""
        with self._lock:
            if symbol and timeframe:
                key = self._make_key(symbol, timeframe)
                self._cache.pop(key, None)
            else:
                self._cache.clear()
                self._last_update.clear()

    def get_stats(self) -> dict:
        """Retorna estadísticas del cache (para monitoreo)."""
        with self._lock:
            return {
                "entries": len(self._cache),
                "pairs": list({k[0] for k in self._cache}),
                "timeframes": list({k[1] for k in self._cache}),
                "total_candles": sum(
                    len(v["timestamp"]) for v in self._cache.values()
                ),
                "last_updates": dict(self._last_update),
            }


# Instancia global del cache (singleton)
# Importar en otros módulos: from data.cache import ohlcv_cache
ohlcv_cache = OHLCVCache()
