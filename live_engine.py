"""
live_engine.py — Motor de ejecución principal (Paper Trading & Live).
=====================================================================
Corre 24/7 en el ZimaBlade. Conecta todos los módulos del sistema:
datos, indicadores, señales, ML, riesgo, portfolio y Telegram.

Arquitectura de dos loops:
    loop_slow: se ejecuta en cada cierre de vela 4H → genera señales
    loop_fast: se ejecuta cada 60s → gestiona SL/TP, heartbeat, comandos Telegram

Uso:
    PAPER_MODE=true python live_engine.py     # paper trading (por defecto)
    PAPER_MODE=false python live_engine.py    # REAL (no usar sin 4 semanas de paper)
"""

from __future__ import annotations

import asyncio
import math
import os
import signal
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import ccxt.async_support as ccxt
import pandas as pd
import structlog
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from ml.meta_labeler import MetaLabeler, FEATURES
from monitoring.telegram_bot import TelegramBot
from paper_portfolio import PaperPortfolio

log = structlog.get_logger(__name__)

# ── Versión del motor ──────────────────────────────────────────────────────────
ENGINE_VERSION = "2.0.0"

# ── Parámetros fijos (reflejo de los del Master Prompt) ───────────────────────
SYMBOLS          = ["BTC/USDT", "ETH/USDT"]
TIMEFRAME        = "4h"
TIMEFRAME_SECS   = 4 * 3600
MAX_POSITIONS    = 3
ML_THRESHOLD     = 0.60
RISK_PCT         = 0.01           # 1% del capital por trade
ATR_STOP_MULT    = 1.5            # 1.5 ATR de stop (no 1.0 — lección aprendida)
COMMISSION_RATE  = 0.001
HEARTBEAT_EVERY  = 60             # segundos entre heartbeats
FAST_LOOP_SECS   = 60
MAX_RETRIES      = 6              # backoff exponencial: 1+2+4+8+16+32 = 63s máx
CIRCUIT_BREAK_ERRORS = 3         # errores en 60s → pausa 5 min


class LiveEngine:
    """
    Motor principal de paper trading / ejecución en vivo.

    Lanza dos corutinas concurrentes:
        - loop_slow: generación de señales en cada vela 4H
        - loop_fast: monitorización de SL/TP y comandos Telegram
    """

    def __init__(self, paper_mode: bool = True) -> None:
        """
        Args:
            paper_mode: si True (por defecto), simula órdenes sin tocar Binance.
                        En False, ejecuta órdenes reales → ¡requiere validación previa!
        """
        self.paper_mode     = paper_mode
        self._running       = False
        self._tasks: List[asyncio.Task] = []

        # Estado compartido (mutable por Telegram bot)
        self.state: Dict[str, Any] = {
            "paused":         False,
            "kill":           False,
            "paper_mode":     paper_mode,
            "capital":        0.0,
            "open_positions": [],
            "pnl_today":      0.0,
            "drawdown_pct":   0.0,
            "ml_ready":       False,
        }

        # Red circuit breaker
        self._recent_errors: List[float] = []

        # Inicializar componentes (setup async en start())
        self._exchange: Optional[ccxt.Exchange] = None
        self._portfolio: Optional[PaperPortfolio] = None
        self._bot: Optional[TelegramBot] = None
        self._ml: Optional[MetaLabeler] = None
        self._db_engine: Optional[Any] = None

    # ══════════════════════════════════════════════════════════════════════════
    # Arranque y parada
    # ══════════════════════════════════════════════════════════════════════════

    async def start(self) -> None:
        """
        Secuencia de arranque completa:
            1. Conectar Binance (ping)
            2. Verificar permisos API
            3. Cargar estado desde PostgreSQL
            4. Cargar modelo ML
            5. Enviar "🚀 Engine arrancado" por Telegram
            6. Lanzar loops
        """
        log.info("engine_starting", version=ENGINE_VERSION, paper_mode=self.paper_mode)
        self._running = True

        # ── 1. Configurar ccxt ────────────────────────────────────────────────
        self._exchange = ccxt.binance({
            "apiKey":    os.environ.get("BINANCE_API_KEY", ""),
            "secret":    os.environ.get("BINANCE_API_SECRET", ""),
            "options":   {"defaultType": "spot"},
            "enableRateLimit": True,
        })

        await self._ping_exchange()

        # ── 2. Verificar permisos API ─────────────────────────────────────────
        if not self.paper_mode:
            api_ok = await self._validate_api_keys()
            if not api_ok:
                log.error("invalid_api_keys_aborting")
                raise RuntimeError("API keys inválidas o con permisos de retiro. Abortando.")

        # ── 3. Cargar estado desde PostgreSQL ─────────────────────────────────
        db_url = os.environ.get("DATABASE_URL", "")
        if not db_url:
            raise RuntimeError("DATABASE_URL no configurada en .env")

        # asyncpg requiere postgresql+asyncpg:// como esquema
        async_db_url = db_url.replace("postgresql://", "postgresql+asyncpg://") \
                              .replace("postgres://",   "postgresql+asyncpg://")

        self._portfolio = PaperPortfolio(
            initial_capital=float(os.environ.get("INITIAL_CAPITAL", "300")),
            db_url=async_db_url,
        )
        await self._portfolio.initialize()
        await self._load_state_from_db()

        # ── 4. Cargar modelo ML ───────────────────────────────────────────────
        self._ml = MetaLabeler(model_path=str(PROJECT_ROOT / "ml" / "model.joblib"))
        self.state["ml_ready"] = self._ml.is_ready()
        if not self.state["ml_ready"]:
            log.warning(
                "ml_model_not_ready",
                detail="Operando con umbral permisivo (proba=0.5)",
            )

        # ── 5. Telegram ───────────────────────────────────────────────────────
        token    = os.environ.get("TELEGRAM_TOKEN", "")
        chat_id  = int(os.environ.get("TELEGRAM_ALLOWED_USER_ID", "0") or "0")
        if token and chat_id:
            self._bot = TelegramBot(token=token, allowed_chat_id=chat_id)
            await self._bot.start()
            await self._bot.send_startup(
                paper_mode=self.paper_mode,
                capital=self.state["capital"],
                n_positions=len(self.state["open_positions"]),
            )
        else:
            log.warning("telegram_not_configured_running_silently")

        # ── 6. Registrar SIGTERM para shutdown limpio ─────────────────────────
        loop = asyncio.get_event_loop()
        loop.add_signal_handler(
            signal.SIGTERM,
            lambda: asyncio.create_task(self.shutdown("SIGTERM")),
        )

        # ── 7. Lanzar loops ───────────────────────────────────────────────────
        log.info("engine_running", symbols=SYMBOLS, timeframe=TIMEFRAME)
        t_slow = asyncio.create_task(self._loop_slow(), name="loop_slow")
        t_fast = asyncio.create_task(self._loop_fast(), name="loop_fast")
        self._tasks = [t_slow, t_fast]

        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            log.info("tasks_cancelled")
        except Exception as exc:
            log.exception("engine_unhandled_exception", error=str(exc))
            if self._bot:
                await self._bot.send_alert(
                    f"Error crítico no capturado:\n```{exc}```", level="critical"
                )
            raise

    async def shutdown(self, reason: str = "manual") -> None:
        """
        Apagado limpio:
            1. Cancelar tasks
            2. Guardar estado en PostgreSQL
            3. Cerrar conexiones
            4. Notificar Telegram
        """
        log.warning("engine_shutdown_initiated", reason=reason)
        self._running = False

        for task in self._tasks:
            task.cancel()

        # PnL de sesión
        daily = await self._portfolio.get_daily_stats() if self._portfolio else {}
        session_pnl = daily.get("pnl_today", 0.0)

        await self._save_state_to_db()

        if self._exchange:
            await self._exchange.close()

        if self._bot:
            await self._bot.send_shutdown(reason=reason, session_pnl=session_pnl)
            await self._bot.stop()

        log.info("engine_shutdown_complete", reason=reason)

    # ══════════════════════════════════════════════════════════════════════════
    # Loop lento — generación de señales (cada cierre de vela 4H)
    # ══════════════════════════════════════════════════════════════════════════

    async def _loop_slow(self) -> None:
        """Genera señales en cada cierre de vela 4H."""
        while self._running:
            # Esperar al próximo cierre de vela
            wait_secs = self._next_4h_candle_close()
            log.debug("loop_slow_waiting", seconds=int(wait_secs))
            await asyncio.sleep(wait_secs + 5)   # +5s margen para que la vela cierre

            if self.state["kill"]:
                break
            if self.state["paused"]:
                log.info("loop_slow_paused_skipping_signal")
                continue

            for symbol in SYMBOLS:
                try:
                    await self._process_symbol(symbol)
                except ccxt.NetworkError as exc:
                    await self._handle_network_error(exc, context=f"process_symbol({symbol})")
                except ccxt.ExchangeError as exc:
                    log.error("exchange_error_skipping_cycle", symbol=symbol, error=str(exc))
                    if self._bot:
                        await self._bot.send_alert(
                            f"Exchange error en {symbol}: {exc}", level="warning"
                        )
                except Exception as exc:
                    log.exception("loop_slow_unexpected_error", symbol=symbol, error=str(exc))
                    if self._bot:
                        await self._bot.send_alert(
                            f"Error inesperado en {symbol}:\n```{exc}```", level="critical"
                        )
                    raise   # Re-raise para que systemd haga restart

    async def _process_symbol(self, symbol: str) -> None:
        """
        Procesa un símbolo en el cierre de la vela 4H:
            1. Descarga velas → 2. Indicadores → 3. Señales → 4. ML → 5. Ejecuta
        """
        # ── 1. Descargar velas ────────────────────────────────────────────────
        df = await self._fetch_candles(symbol, TIMEFRAME, limit=500)
        if df is None or len(df) < 200:
            log.warning("insufficient_candles", symbol=symbol, n=len(df) if df is not None else 0)
            return

        # ── 2. Indicadores ────────────────────────────────────────────────────
        try:
            from indicators.technical import apply_all_indicators
            df = apply_all_indicators(df)
        except ImportError:
            log.error("indicators_module_not_found")
            return

        # ── 3. Señales ────────────────────────────────────────────────────────
        try:
            from strategies.signals import apply_all_signals
            df = apply_all_signals(df)
        except ImportError:
            log.error("signals_module_not_found")
            return

        df = df.dropna()
        if df.empty:
            return

        last = df.iloc[-1]

        # ── 4. Verificar señal en la última vela ──────────────────────────────
        signal_col = "signal_trend"    # columna producida por signals.py
        if signal_col not in df.columns or not bool(last.get(signal_col, False)):
            log.debug("no_signal", symbol=symbol)
            return

        # ── 4a. Régimen de mercado ─────────────────────────────────────────────
        try:
            from risk.regime_filter import RegimeFilter
            regime_ok = RegimeFilter().is_tradeable(df, strategy="trend_following")
            if not regime_ok:
                log.info("regime_filter_blocked", symbol=symbol)
                return
        except ImportError:
            log.warning("regime_filter_not_found_skipping_check")

        # ── 4b. Circuit breaker de drawdown ───────────────────────────────────
        capital = await self._portfolio.get_current_capital()
        daily   = await self._portfolio.get_daily_stats()
        pnl_day = daily["pnl_today"]
        dd_pct  = pnl_day / capital if capital > 0 else 0
        if dd_pct < -0.03:   # 3% drawdown diario → no operar
            log.warning("daily_drawdown_circuit_breaker", dd_pct=f"{dd_pct:.2%}")
            return

        # ── 4c. Máximo de posiciones simultáneas ──────────────────────────────
        open_positions = await self._portfolio.get_open_positions()
        if len(open_positions) >= MAX_POSITIONS:
            log.info("max_positions_reached", current=len(open_positions), max=MAX_POSITIONS)
            return

        # ── 4d. MetaLabeler ───────────────────────────────────────────────────
        features = self._extract_features(last)
        ml_proba = self._ml.predict_proba(features) if self._ml else 0.5

        if ml_proba < ML_THRESHOLD:
            log.info("signal_filtered_by_ml", symbol=symbol, proba=f"{ml_proba:.2%}")
            if self._bot:
                await self._bot.send_signal_filtered(symbol, "trend_following", ml_proba)
            return

        # ── 5. Calcular SL, TP y position size ────────────────────────────────
        price = float(last["close"])
        atr   = float(last.get("atr", price * 0.015))

        stop_loss = price - ATR_STOP_MULT * atr
        tp1       = price + 2.0 * (price - stop_loss)   # R/R = 2:1
        tp2       = price + 3.0 * (price - stop_loss)   # R/R = 3:1

        risk_amount = capital * RISK_PCT
        units = risk_amount / (price - stop_loss) if (price - stop_loss) > 0 else 0
        if units <= 0:
            log.warning("invalid_position_size", symbol=symbol)
            return

        # ── 6. Abrir posición ─────────────────────────────────────────────────
        trade = await self._portfolio.open_position(
            symbol=symbol,
            strategy="trend_following",
            entry_price=price,
            stop_loss=stop_loss,
            tp1=tp1,
            tp2=tp2,
            units=units,
            ml_proba=ml_proba,
        )

        # ── 7. Notificar Telegram ─────────────────────────────────────────────
        if self._bot:
            await self._bot.send_trade_open(trade)

        log.info(
            "trade_opened",
            symbol=symbol,
            price=f"${price:.2f}",
            sl=f"${stop_loss:.2f}",
            tp1=f"${tp1:.2f}",
            units=f"{units:.6f}",
            ml=f"{ml_proba:.1%}",
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Loop rápido — monitorización 60s (SL/TP, heartbeat, comandos Telegram)
    # ══════════════════════════════════════════════════════════════════════════

    async def _loop_fast(self) -> None:
        """Monitorización cada 60 segundos."""
        last_daily_summary = datetime.now(tz=timezone.utc).date()

        while self._running:
            await asyncio.sleep(FAST_LOOP_SECS)

            if self.state["kill"]:
                log.warning("kill_switch_activated")
                prices = await self._get_current_prices()
                closed = await self._portfolio.emergency_close_all(prices)
                for ct in closed:
                    if self._bot:
                        await self._bot.send_trade_close(ct, "kill_switch")
                await self.shutdown("kill_switch")
                break

            # ── 1. Obtener precios actuales ────────────────────────────────────
            try:
                prices = await self._get_current_prices()
            except ccxt.NetworkError as exc:
                await self._handle_network_error(exc, "get_current_prices")
                continue

            # ── 2. Verificar SL/TP ────────────────────────────────────────────
            closed_trades = await self._portfolio.update_positions(prices)
            for ct in closed_trades:
                if self._bot:
                    await self._bot.send_trade_close(ct, ct["exit_reason"])

            # ── 3. Actualizar estado compartido ───────────────────────────────
            capital = await self._portfolio.get_current_capital()
            daily   = await self._portfolio.get_daily_stats()
            positions = await self._portfolio.get_open_positions()

            self.state["capital"]        = capital
            self.state["pnl_today"]      = daily["pnl_today"]
            self.state["open_positions"] = positions

            # Drawdown del día
            if capital > 0:
                self.state["drawdown_pct"] = daily["pnl_today"] / capital

            # ── 4. Heartbeat en BD ────────────────────────────────────────────
            await self._update_heartbeat()

            # ── 5. Procesar comandos Telegram ─────────────────────────────────
            if self._bot:
                await self._bot.process_updates(self.state)

            # ── 6. Resumen diario (una vez al día a las 23:59 UTC) ────────────
            now = datetime.now(tz=timezone.utc)
            if now.hour == 23 and now.minute >= 55:
                today = now.date()
                if today != last_daily_summary:
                    last_daily_summary = today
                    if self._bot:
                        await self._bot.send_daily_pnl({
                            **daily,
                            "capital":      capital,
                            "drawdown_pct": self.state["drawdown_pct"],
                            "ml_ready":     self.state["ml_ready"],
                            "ml_trained_at": self._ml.get_metadata().get(
                                "trained_at", "—"
                            ) if self._ml else "—",
                        })

            # ── 7. Resetear DD tracker si cambió el día ────────────────────────
            await self._maybe_reset_period_trackers()

    # ══════════════════════════════════════════════════════════════════════════
    # Helpers de persistencia
    # ══════════════════════════════════════════════════════════════════════════

    async def _load_state_from_db(self) -> None:
        """Recupera capital y posiciones abiertas desde PostgreSQL."""
        capital = await self._portfolio.get_current_capital()
        positions = await self._portfolio.get_open_positions()
        self.state["capital"] = capital
        self.state["open_positions"] = positions
        log.info(
            "state_loaded",
            capital=f"${capital:.2f}",
            open_positions=len(positions),
        )

    async def _save_state_to_db(self) -> None:
        """Guarda el estado actual del portfolio en PostgreSQL."""
        if not self._portfolio or not self._portfolio._engine:
            return
        async with self._portfolio._session_factory() as sess:
            await sess.execute(
                text("""
                    UPDATE portfolio_state
                    SET current_capital = :cap, updated_at = NOW()
                    WHERE id = 1
                """),
                {"cap": self.state["capital"]},
            )
            await sess.commit()
        log.info("state_saved_to_db", capital=f"${self.state['capital']:.2f}")

    async def _update_heartbeat(self) -> None:
        """Actualiza el timestamp de heartbeat en la BD."""
        if not self._portfolio:
            return
        try:
            async with self._portfolio._session_factory() as sess:
                await sess.execute(
                    text("""
                        INSERT INTO system_heartbeat (id, last_ping, engine_version, paper_mode)
                        VALUES (1, NOW(), :version, :paper)
                        ON CONFLICT (id) DO UPDATE
                          SET last_ping = NOW()
                    """),
                    {"version": ENGINE_VERSION, "paper": self.paper_mode},
                )
                await sess.commit()
        except Exception as exc:
            log.warning("heartbeat_update_failed", error=str(exc))

    async def _maybe_reset_period_trackers(self) -> None:
        """Resetea daily_start / weekly_start en la BD si cambió el período."""
        if not self._portfolio:
            return
        now = datetime.now(tz=timezone.utc)
        # Resetear daily_start a medianoche UTC
        if now.hour == 0 and now.minute < 2:
            capital = await self._portfolio.get_current_capital()
            async with self._portfolio._session_factory() as sess:
                await sess.execute(
                    text("""
                        UPDATE portfolio_state
                        SET daily_start = :cap
                        WHERE id = 1
                    """),
                    {"cap": capital},
                )
                await sess.commit()
            log.info("daily_tracker_reset", capital=f"${capital:.2f}")

    # ══════════════════════════════════════════════════════════════════════════
    # Helpers de red y datos
    # ══════════════════════════════════════════════════════════════════════════

    async def _ping_exchange(self) -> None:
        """Verifica conectividad con Binance."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                await self._exchange.fetch_time()
                log.info("exchange_connected", exchange="binance")
                return
            except ccxt.NetworkError as exc:
                delay = min(2 ** attempt, 60)
                log.warning(
                    "exchange_ping_failed",
                    attempt=attempt,
                    retry_in=delay,
                    error=str(exc),
                )
                await asyncio.sleep(delay)
        raise RuntimeError("No se pudo conectar a Binance tras múltiples intentos.")

    async def _validate_api_keys(self) -> bool:
        """
        Verifica que las API keys tengan permisos de spot trading pero NO retiros.

        Returns:
            True si las keys son válidas y seguras.
        """
        try:
            permissions = await self._exchange.fetch_api_key_permissions()
            can_trade    = permissions.get("enableSpotAndMarginTrading", False)
            can_withdraw = permissions.get("enableWithdrawals", False)

            if can_withdraw:
                log.error("api_key_has_withdrawal_permission_DANGEROUS")
                if self._bot:
                    await self._bot.send_alert(
                        "🚨 API key tiene permiso de RETIROS. Desactívalo en Binance.", "critical"
                    )
                return False

            if not can_trade:
                log.error("api_key_missing_spot_trading_permission")
                return False

            log.info("api_keys_validated", can_trade=can_trade, can_withdraw=can_withdraw)
            return True

        except ccxt.AuthenticationError as exc:
            log.error("api_keys_invalid", error=str(exc))
            return False
        except Exception as exc:
            log.error("api_key_validation_error", error=str(exc))
            return False

    async def _fetch_candles(
        self, symbol: str, timeframe: str, limit: int = 500
    ) -> Optional[pd.DataFrame]:
        """
        Descarga velas OHLCV de Binance con backoff exponencial.

        Returns:
            DataFrame con columnas: timestamp, open, high, low, close, volume.
        """
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                ohlcv = await self._exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                df = pd.DataFrame(
                    ohlcv,
                    columns=["timestamp", "open", "high", "low", "close", "volume"],
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                return df
            except ccxt.NetworkError as exc:
                delay = min(2 ** attempt, 60)
                log.warning(
                    "fetch_candles_retry",
                    symbol=symbol,
                    attempt=attempt,
                    retry_in=delay,
                    error=str(exc),
                )
                await asyncio.sleep(delay)
        log.error("fetch_candles_failed_all_retries", symbol=symbol)
        return None

    async def _get_current_prices(self) -> Dict[str, float]:
        """
        Obtiene el último precio de todos los símbolos activos.

        Returns:
            {symbol: last_price}
        """
        prices: Dict[str, float] = {}
        for symbol in SYMBOLS:
            ticker = await self._exchange.fetch_ticker(symbol)
            prices[symbol] = float(ticker["last"])
        return prices

    async def _handle_network_error(
        self, exc: Exception, context: str
    ) -> None:
        """
        Registra errores de red y activa circuit breaker si hay 3 en 60s.
        En ese caso pausa 5 minutos.
        """
        import time
        now = time.time()
        self._recent_errors = [t for t in self._recent_errors if now - t < 60]
        self._recent_errors.append(now)

        log.warning("network_error", context=context, error=str(exc))

        if len(self._recent_errors) >= CIRCUIT_BREAK_ERRORS:
            log.error("circuit_breaker_triggered_pausing_5min")
            if self._bot:
                await self._bot.send_alert(
                    f"Circuit breaker activado: 3 errores de red en 60s. "
                    f"Pausa de 5 minutos.", level="warning"
                )
            self._recent_errors.clear()
            await asyncio.sleep(300)   # 5 minutos

    # ══════════════════════════════════════════════════════════════════════════
    # Helpers de ML y tiempo
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _extract_features(row: pd.Series) -> Dict[str, float]:
        """
        Extrae el vector de features de la última vela para el MetaLabeler.

        Args:
            row: última fila del DataFrame con indicadores calculados.

        Returns:
            dict compatible con MetaLabeler.predict_proba().
        """
        price = float(row.get("close", 1))
        ema21 = float(row.get("ema_21", price))
        ema55 = float(row.get("ema_55", price))
        atr   = float(row.get("atr", price * 0.015))
        vol   = float(row.get("volume", 0))
        vol_ma = float(row.get("volume_ma20", vol or 1))
        cvd   = float(row.get("cvd", 0))
        bb_up = float(row.get("bb_upper", price * 1.02))
        bb_lo = float(row.get("bb_lower", price * 0.98))
        bb_mid = float(row.get("bb_mid", price))
        regime_raw = str(row.get("regime", "RANGE")).upper()
        regime_map = {"BULL": 1, "RANGE": 0, "HIGH_VOL": -1}

        return {
            "rsi":             float(row.get("rsi", 50)),
            "adx":             float(row.get("adx", 20)),
            "atr_pct":         atr / price * 100 if price > 0 else 0,
            "ema21_dist_pct":  (price - ema21) / ema21 * 100 if ema21 > 0 else 0,
            "ema55_dist_pct":  (price - ema55) / ema55 * 100 if ema55 > 0 else 0,
            "macd_hist":       float(row.get("macd_hist", 0)),
            "bb_width":        (bb_up - bb_lo) / bb_mid if bb_mid > 0 else 0,
            "vol_ratio":       vol / vol_ma if vol_ma > 0 else 1,
            "cvd_bull":        1.0 if cvd > 0 else 0.0,
            "regime_encoded":  float(regime_map.get(regime_raw, 0)),
        }

    @staticmethod
    def _next_4h_candle_close() -> float:
        """
        Calcula los segundos hasta el próximo cierre de vela 4H UTC.

        Ejemplo: si son las 07:35 UTC, la próxima vela cierra a las 08:00 UTC
        → devuelve 25 * 60 = 1500 segundos.
        """
        now = datetime.now(tz=timezone.utc)
        epoch_secs = now.timestamp()
        next_close = math.ceil(epoch_secs / TIMEFRAME_SECS) * TIMEFRAME_SECS
        return max(next_close - epoch_secs, 1)


# ══════════════════════════════════════════════════════════════════════════════
# Entrypoint
# ══════════════════════════════════════════════════════════════════════════════

async def _main() -> None:
    paper_mode_env = os.environ.get("PAPER_MODE", "true").lower()
    paper_mode = paper_mode_env not in ("false", "0", "no")

    if not paper_mode:
        log.warning(
            "LIVE_MODE_ACTIVE",
            message="PAPER_MODE=false — el sistema ejecutará órdenes REALES en Binance.",
        )

    engine = LiveEngine(paper_mode=paper_mode)
    try:
        await engine.start()
    except KeyboardInterrupt:
        log.info("keyboard_interrupt")
        await engine.shutdown("KeyboardInterrupt")
    except Exception as exc:
        log.exception("fatal_error", error=str(exc))
        await engine.shutdown("fatal_error")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(_main())
