"""
Data Fetcher — Descarga de Datos OHLCV via ccxt
================================================
Responsabilidades:
  - Descargar datos OHLCV históricos de Binance via ccxt
  - Gestionar paginación (Binance devuelve máx. 1000 velas por request)
  - Rate limiting automático para no ser baneado
  - Detección y relleno de gaps en los datos
  - Persistir en SQLite via storage.py

Uso:
    from data.fetcher import DataFetcher

    fetcher = DataFetcher()
    df = await fetcher.fetch_ohlcv("BTC/USDT", "4h", since_days=730)
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import ccxt
import ccxt.async_support as ccxt_async
import pandas as pd
from tqdm import tqdm

from config.logging_config import get_logger
from config.settings import (
    ASSETS,
    TIMEFRAME_MINUTES,
    AssetPriority,
    env_settings,
)
from data.storage import OHLCVStorage

logger = get_logger(__name__)


class DataFetcher:
    """
    Descarga datos OHLCV históricos de Binance.

    Soporta descarga síncrona (para scripts) y asíncrona (para el engine live).
    Los datos se almacenan automáticamente en SQLite para evitar re-descargas.
    """

    # Límite de Binance: 1000 velas por request
    CANDLES_PER_REQUEST = 1000
    # Pausa entre requests para respetar rate limits
    REQUEST_DELAY_MS = 200

    def __init__(
        self,
        use_testnet: bool = False,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ):
        self.storage = OHLCVStorage()
        self._use_testnet = use_testnet

        # Exchange síncrono (para backtesting / scripts)
        exchange_config: dict = {
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        }

        # Agregar credenciales si están disponibles (no necesarias para datos públicos)
        _key = api_key or env_settings.binance_api_key
        _secret = api_secret or env_settings.binance_api_secret
        if _key and _secret:
            exchange_config["apiKey"] = _key
            exchange_config["secret"] = _secret

        if use_testnet:
            exchange_config["options"]["urls"] = {
                "api": "https://testnet.binance.vision/api"
            }

        self.exchange: ccxt.Exchange = ccxt.binance(exchange_config)
        logger.info("data_fetcher_initialized", exchange="binance", testnet=use_testnet)

    # ── API Pública (sin autenticación) ───────────────────────────────────────

    def fetch_ohlcv_sync(
        self,
        symbol: str,
        timeframe: str,
        since_days: int = 730,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """
        Descarga datos OHLCV históricos de forma síncrona.

        Args:
            symbol: Par de trading (ej: "BTC/USDT")
            timeframe: Timeframe ccxt (ej: "4h", "1h", "15m")
            since_days: Número de días hacia atrás para descargar
            show_progress: Mostrar barra de progreso tqdm

        Returns:
            DataFrame con columnas: timestamp, open, high, low, close, volume
        """
        # 1. Intentar cargar desde caché local primero
        cached = self.storage.load_ohlcv(symbol, timeframe)
        if cached is not None and not cached.empty:
            # Verificar si el caché tiene datos suficientemente recientes
            latest = cached["timestamp"].max()
            now = pd.Timestamp.now(tz="UTC")
            gap_minutes = (now - latest).total_seconds() / 60
            tf_minutes = TIMEFRAME_MINUTES.get(timeframe, 60)

            if gap_minutes < tf_minutes * 3:
                logger.info(
                    "ohlcv_loaded_from_cache",
                    symbol=symbol,
                    timeframe=timeframe,
                    rows=len(cached),
                )
                return cached

            # Hay un gap — descargar solo datos nuevos
            logger.info(
                "ohlcv_cache_outdated_fetching_delta",
                symbol=symbol,
                timeframe=timeframe,
                gap_hours=round(gap_minutes / 60, 1),
            )
            since_ms = int(latest.timestamp() * 1000)
            new_data = self._download_range(symbol, timeframe, since_ms, show_progress=False)
            if not new_data.empty:
                combined = pd.concat([cached, new_data]).drop_duplicates("timestamp")
                combined = combined.sort_values("timestamp").reset_index(drop=True)
                self.storage.save_ohlcv(combined, symbol, timeframe)
                return combined
            return cached

        # 2. Descarga completa
        since_dt = datetime.now(timezone.utc) - timedelta(days=since_days)
        since_ms = int(since_dt.timestamp() * 1000)

        logger.info(
            "ohlcv_download_start",
            symbol=symbol,
            timeframe=timeframe,
            since=since_dt.isoformat(),
        )

        df = self._download_range(symbol, timeframe, since_ms, show_progress=show_progress)

        if not df.empty:
            self.storage.save_ohlcv(df, symbol, timeframe)
            logger.info(
                "ohlcv_download_complete",
                symbol=symbol,
                timeframe=timeframe,
                rows=len(df),
            )

        return df

    def _download_range(
        self,
        symbol: str,
        timeframe: str,
        since_ms: int,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """
        Descarga un rango de datos OHLCV manejando la paginación de Binance.
        Binance devuelve máximo 1000 velas por request — hacemos loops.
        """
        all_candles: List[list] = []
        current_since = since_ms
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

        tf_minutes = TIMEFRAME_MINUTES.get(timeframe, 60)
        tf_ms = tf_minutes * 60 * 1000
        expected_candles = max(1, (now_ms - since_ms) // tf_ms)

        pbar = tqdm(
            total=expected_candles,
            desc=f"Descargando {symbol} {timeframe}",
            unit="velas",
            disable=not show_progress,
        )

        try:
            while current_since < now_ms:
                try:
                    candles = self.exchange.fetch_ohlcv(
                        symbol,
                        timeframe=timeframe,
                        since=current_since,
                        limit=self.CANDLES_PER_REQUEST,
                    )
                except ccxt.NetworkError as e:
                    logger.warning("network_error_retrying", error=str(e))
                    time.sleep(5)
                    continue
                except ccxt.RateLimitExceeded:
                    logger.warning("rate_limit_exceeded_sleeping")
                    time.sleep(30)
                    continue

                if not candles:
                    break

                all_candles.extend(candles)
                pbar.update(len(candles))

                # La siguiente iteración empieza desde la última vela + 1 tf
                last_ts = candles[-1][0]
                current_since = last_ts + tf_ms

                # Si devolvió menos velas de las esperadas, llegamos al final
                if len(candles) < self.CANDLES_PER_REQUEST:
                    break

                # Rate limiting cortés
                time.sleep(self.REQUEST_DELAY_MS / 1000)

        finally:
            pbar.close()

        if not all_candles:
            logger.warning("no_data_returned", symbol=symbol, timeframe=timeframe)
            return pd.DataFrame()

        return self._candles_to_dataframe(all_candles)

    @staticmethod
    def _candles_to_dataframe(candles: List[list]) -> pd.DataFrame:
        """Convierte la lista de velas ccxt a un DataFrame limpio y tipado."""
        df = pd.DataFrame(
            candles,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

        # Casting explícito de tipos
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)

        return df

    def fetch_multiple_pairs(
        self,
        symbols: List[str],
        timeframe: str,
        since_days: int = 730,
    ) -> dict[str, pd.DataFrame]:
        """
        Descarga datos de múltiples pares secuencialmente.
        Retorna diccionario {symbol: DataFrame}.
        """
        results: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            if symbol not in ASSETS:
                logger.warning("symbol_not_in_universe", symbol=symbol)
                continue
            try:
                df = self.fetch_ohlcv_sync(symbol, timeframe, since_days)
                if not df.empty:
                    results[symbol] = df
            except Exception as e:
                logger.error("fetch_failed", symbol=symbol, error=str(e))

        return results

    def fetch_multiple_timeframes(
        self,
        symbol: str,
        timeframes: List[str],
        since_days: int = 730,
    ) -> dict[str, pd.DataFrame]:
        """
        Descarga múltiples timeframes para un mismo par.
        Retorna diccionario {timeframe: DataFrame}.
        """
        results: dict[str, pd.DataFrame] = {}
        for tf in timeframes:
            try:
                df = self.fetch_ohlcv_sync(symbol, tf, since_days)
                if not df.empty:
                    results[tf] = df
            except Exception as e:
                logger.error("fetch_failed", symbol=symbol, timeframe=tf, error=str(e))
        return results

    def validate_data_quality(self, df: pd.DataFrame, timeframe: str) -> dict:
        """
        Valida la calidad de los datos descargados.
        Detecta gaps, valores nulos y anomalías de precio.

        Returns:
            Dict con estadísticas de calidad y lista de problemas encontrados.
        """
        issues = []
        tf_minutes = TIMEFRAME_MINUTES.get(timeframe, 60)
        expected_gap = timedelta(minutes=tf_minutes)

        # Detectar gaps temporales
        time_diffs = df["timestamp"].diff().dropna()
        gaps = time_diffs[time_diffs > expected_gap * 1.5]
        if not gaps.empty:
            issues.append(f"{len(gaps)} gaps temporales detectados")

        # Valores nulos
        null_counts = df.isnull().sum()
        if null_counts.any():
            issues.append(f"Valores nulos: {null_counts.to_dict()}")

        # Precios imposibles (high < low, close fuera de rango)
        invalid_prices = df[df["high"] < df["low"]]
        if not invalid_prices.empty:
            issues.append(f"{len(invalid_prices)} velas con high < low")

        # Volumen cero
        zero_volume = df[df["volume"] == 0]
        if len(zero_volume) > len(df) * 0.01:  # > 1% de velas con vol = 0
            issues.append(f"{len(zero_volume)} velas con volumen cero")

        quality_score = max(0, 100 - len(issues) * 10)

        return {
            "total_candles": len(df),
            "date_range": f"{df['timestamp'].min()} → {df['timestamp'].max()}",
            "gaps_count": len(gaps),
            "null_count": int(null_counts.sum()),
            "quality_score": quality_score,
            "issues": issues,
            "is_valid": quality_score >= 80,
        }

    def close(self):
        """Cierra la sesión del exchange si es necesario."""
        pass

    def __enter__(self) -> "DataFetcher":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
