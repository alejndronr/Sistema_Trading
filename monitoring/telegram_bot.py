"""
monitoring/telegram_bot.py — Bot de Telegram de control institucional.
=======================================================================
Se integra en live_engine.py, NO es un proceso separado.
Envía alertas pasivas (trades, señales, PnL) y procesa comandos activos
del operador (/status, /pause, /resume, /kill).

Seguridad:
    - Solo responde al TELEGRAM_ALLOWED_USER_ID configurado en .env.
    - Los mensajes de IDs no autorizados se descartan en silencio (sin log).
    - El kill switch requiere dos mensajes: /kill + CONFIRMAR (máx. 60s).
    - Si el token es inválido, el sistema opera sin bot (no es fallo crítico).
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict, List, Optional

import aiohttp
import structlog

log = structlog.get_logger(__name__)

ML_THRESHOLD = 0.60   # Umbral por defecto; live_engine puede pasar el suyo


class TelegramBot:
    """
    Cliente HTTP asíncrono para la Telegram Bot API.

    Envía mensajes y hace long-polling para recibir comandos.
    No bloquea el event loop: todos los métodos son async.
    """

    # Prefijo de la API
    _BASE = "https://api.telegram.org/bot{token}/{method}"

    def __init__(self, token: str, allowed_chat_id: int) -> None:
        """
        Args:
            token:          token del bot obtenido de @BotFather.
            allowed_chat_id: único ID autorizado para enviar comandos.
        """
        self._token = token
        self._allowed_chat_id = allowed_chat_id
        self._session: Optional[aiohttp.ClientSession] = None
        self._offset: int = 0
        self._kill_pending: bool = False
        self._kill_timestamp: float = 0.0
        self._ready: bool = False

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """
        Inicializa la sesión HTTP y verifica el token con getMe.
        Si el token es inválido, loguea error pero NO lanza excepción
        (el motor debe operar aunque el bot falle).
        """
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15),
        )
        try:
            data = await self._call("getMe", {})
            if data and data.get("ok"):
                bot_name = data["result"].get("username", "?")
                log.info("telegram_bot_ready", bot_username=bot_name)
                self._ready = True
            else:
                log.error("telegram_invalid_token", response=data)
        except Exception as exc:
            log.error("telegram_start_failed", error=str(exc))
            # No relanzar: el motor sigue sin bot

    async def stop(self) -> None:
        """Cierra la sesión HTTP."""
        if self._session and not self._session.closed:
            await self._session.close()
        self._ready = False

    # ── Envío de mensajes ─────────────────────────────────────────────────────

    async def send_trade_open(self, trade: Dict[str, Any]) -> None:
        """
        Alerta cuando se abre una posición virtual o real.

        Args:
            trade: dict con symbol, entry_price, stop_loss, tp1, tp2,
                   units, risk_amount, strategy, ml_proba.
        """
        symbol = trade.get("symbol", "—")
        entry  = trade.get("entry_price", 0)
        sl     = trade.get("stop_loss", 0)
        tp1    = trade.get("tp1", 0)
        tp2    = trade.get("tp2", 0)
        risk   = trade.get("risk_amount", 0)
        ml     = trade.get("ml_proba", 0)
        strat  = trade.get("strategy", "—")

        msg = (
            f"🟢 *TRADE ABIERTO*\n"
            f"Par: `{symbol}` | Estrategia: `{strat}`\n"
            f"Entrada: `${entry:.2f}`\n"
            f"SL: `${sl:.2f}` | TP1: `${tp1:.2f}` | TP2: `${tp2:.2f}`\n"
            f"Riesgo: `${risk:.2f}` (1%)\n"
            f"Confianza ML: `{ml:.1%}` ✅"
        )
        await self._send(msg)

    async def send_trade_close(self, trade: Dict[str, Any], reason: str) -> None:
        """
        Alerta cuando se cierra una posición.

        Args:
            trade:  dict con pnl, pnl_pct, r_multiple, symbol, strategy.
            reason: razón de cierre (stop_loss, tp1, tp2, breakeven, kill).
        """
        pnl     = trade.get("pnl", 0)
        pnl_pct = trade.get("pnl_pct", 0)
        r_mult  = trade.get("r_multiple", 0)
        symbol  = trade.get("symbol", "—")

        if pnl > 0:
            emoji = "🟢"
            outcome = "PROFIT"
        elif abs(pnl) < 0.01:
            emoji = "🟡"
            outcome = "BREAKEVEN"
        else:
            emoji = "🔴"
            outcome = "LOSS"

        msg = (
            f"{emoji} *TRADE CERRADO — {outcome}*\n"
            f"Par: `{symbol}` | Razón: `{reason}`\n"
            f"PnL: `{pnl:+.2f}` USD (`{pnl_pct:+.2%}`)\n"
            f"R-Múltiple: `{r_mult:+.2f}R`"
        )
        await self._send(msg)

    async def send_signal_filtered(
        self,
        symbol: str,
        strategy: str,
        ml_proba: float,
        threshold: float = ML_THRESHOLD,
    ) -> None:
        """
        Alerta silenciosa cuando una señal es rechazada por el filtro ML.
        """
        msg = (
            f"⚪ *SEÑAL SILENCIADA*\n"
            f"`{symbol}` [`{strategy}`]\n"
            f"ML probabilidad: `{ml_proba:.1%}` (umbral `{threshold:.0%}`)"
        )
        await self._send(msg)

    async def send_daily_pnl(self, stats: Dict[str, Any]) -> None:
        """
        Resumen nocturno de PnL y estado del sistema.

        Args:
            stats: dict con capital, pnl_today, trades_today, wins_today,
                   losses_today, drawdown_pct, ml_ready, ml_trained_at.
        """
        capital   = stats.get("capital", 0)
        pnl       = stats.get("pnl_today", 0)
        trades    = stats.get("trades_today", 0)
        wins      = stats.get("wins_today", 0)
        losses    = stats.get("losses_today", 0)
        dd        = stats.get("drawdown_pct", 0)
        ml_ready  = "✅" if stats.get("ml_ready") else "⚠️ Sin entrenar"
        ml_date   = stats.get("ml_trained_at", "—")

        trend = "📈" if pnl >= 0 else "📉"
        msg = (
            f"*📊 Resumen diario*\n"
            f"{trend} PnL: `{pnl:+.2f}` USD\n"
            f"Capital: `${capital:.2f}`\n"
            f"Trades: `{trades}` (W:{wins} / L:{losses})\n"
            f"Drawdown: `{dd:.2%}`\n"
            f"Modelo ML: {ml_ready} | Entrenado: `{ml_date}`"
        )
        await self._send(msg)

    async def send_alert(self, message: str, level: str = "info") -> None:
        """
        Envía una alerta genérica.

        Args:
            message: texto libre.
            level:   "info" | "warning" | "critical".
        """
        prefix = {
            "info":     "ℹ️",
            "warning":  "⚠️",
            "critical": "🚨",
        }.get(level, "ℹ️")

        msg = f"{prefix} {message}"
        await self._send(msg, pin=(level == "critical"))

    async def send_heartbeat_fail(self) -> None:
        """Alerta de heartbeat perdido."""
        await self._send(
            "🚨 *HEARTBEAT PERDIDO*\nEl engine no respondió en los últimos 60s.\n"
            "Verificar: `journalctl -fu trading-engine`",
            pin=True,
        )

    async def send_startup(self, paper_mode: bool, capital: float, n_positions: int) -> None:
        """Alerta de arranque del motor."""
        mode_label = "📄 PAPER" if paper_mode else "🔴 LIVE"
        msg = (
            f"🚀 *Engine arrancado* | {mode_label}\n"
            f"Capital: `${capital:.2f}`\n"
            f"Posiciones recuperadas: `{n_positions}`"
        )
        await self._send(msg)

    async def send_shutdown(self, reason: str, session_pnl: float) -> None:
        """Alerta de parada del motor."""
        msg = (
            f"⛔ *Engine detenido*\n"
            f"Razón: `{reason}`\n"
            f"PnL de sesión: `{session_pnl:+.2f}` USD"
        )
        await self._send(msg)

    # ── Procesamiento de comandos (polling) ───────────────────────────────────

    async def process_updates(self, engine_state: Dict[str, Any]) -> Optional[str]:
        """
        Hace long-polling de updates de Telegram (timeout=5s, no bloquea).
        Procesa los comandos del operador autorizado y muta engine_state.

        Args:
            engine_state: dict compartido con el motor. Claves relevantes:
                          "paused" (bool), "kill" (bool), "capital" (float),
                          "open_positions" (list), "pnl_today" (float),
                          "drawdown_pct" (float), "ml_ready" (bool).

        Returns:
            El comando procesado como string, o None si no hubo nada.
        """
        if not self._ready or not self._session:
            return None

        updates = await self._get_updates(timeout=5)
        processed_cmd: Optional[str] = None

        for update in updates:
            self._offset = update["update_id"] + 1
            message = update.get("message", {})
            if not message:
                continue

            from_id = message.get("from", {}).get("id")
            chat_id = message.get("chat", {}).get("id")
            text    = (message.get("text") or "").strip()

            # ── Verificación de autorización ──────────────────────────────────
            if from_id != self._allowed_chat_id:
                # Silencio total ante usuarios no autorizados
                continue

            log.info("telegram_command_received", text=text[:60])

            # ── Kill switch en dos pasos ───────────────────────────────────────
            if self._kill_pending:
                if time.time() - self._kill_timestamp > 60:
                    self._kill_pending = False
                    await self._reply(chat_id, "⏱ Confirmación expirada. /kill cancelado.")
                    continue

                if text.upper() == "CONFIRMAR":
                    self._kill_pending = False
                    engine_state["kill"] = True
                    await self._reply(
                        chat_id,
                        "⛔ *KILL SWITCH activado.*\n"
                        "Cerrando posiciones y apagando el motor..."
                    )
                    processed_cmd = "kill"
                else:
                    self._kill_pending = False
                    await self._reply(chat_id, "✅ Kill cancelado. El motor sigue activo.")
                continue

            # ── Comandos estándar ─────────────────────────────────────────────
            cmd = text.split()[0].lower() if text else ""

            if cmd == "/status":
                await self._cmd_status(chat_id, engine_state)
                processed_cmd = "status"

            elif cmd == "/pause":
                engine_state["paused"] = True
                await self._reply(
                    chat_id,
                    "⏸ *Motor pausado.*\n"
                    "No se abrirán nuevas posiciones.\n"
                    "SL/TP siguen monitorizados.\n"
                    "Usa /resume para reanudar."
                )
                log.warning("engine_paused_by_telegram")
                processed_cmd = "pause"

            elif cmd == "/resume":
                if engine_state.get("paused"):
                    engine_state["paused"] = False
                    await self._reply(chat_id, "▶️ *Motor reanudado.* Operativa normal.")
                    log.info("engine_resumed_by_telegram")
                else:
                    await self._reply(chat_id, "ℹ️ El motor ya está activo.")
                processed_cmd = "resume"

            elif cmd == "/kill":
                self._kill_pending = True
                self._kill_timestamp = time.time()
                await self._reply(
                    chat_id,
                    "⛔ *Kill Switch*\n\n"
                    "Esta acción cerrará *todas las posiciones* a precio de mercado "
                    "y detendrá el motor.\n\n"
                    "Responde exactamente `CONFIRMAR` en los próximos 60 segundos.\n"
                    "Cualquier otra respuesta cancela la operación."
                )
                processed_cmd = "kill_requested"

            elif cmd.startswith("/"):
                await self._reply(
                    chat_id,
                    "ℹ️ Comando no reconocido.\n"
                    "Comandos disponibles: /status /pause /resume /kill"
                )

            # Mensajes sin "/" → ignorar en silencio

        return processed_cmd

    # ── Handlers internos de comandos ─────────────────────────────────────────

    async def _cmd_status(
        self, chat_id: int, engine_state: Dict[str, Any]
    ) -> None:
        """Envía el estado completo del sistema al chat del operador."""
        capital    = engine_state.get("capital", 0)
        paused     = engine_state.get("paused", False)
        pnl        = engine_state.get("pnl_today", 0)
        dd         = engine_state.get("drawdown_pct", 0)
        ml_ready   = "✅" if engine_state.get("ml_ready") else "⚠️ Sin entrenar"
        paper      = "📄 PAPER" if engine_state.get("paper_mode", True) else "🔴 LIVE"
        positions  = engine_state.get("open_positions", [])
        state_str  = "⏸ PAUSADO" if paused else "▶️ ACTIVO"

        pos_lines: List[str] = []
        for p in positions:
            pos_lines.append(
                f"  • `{p.get('symbol')}` @ `${p.get('entry_price', 0):.2f}` "
                f"SL `${p.get('stop_loss', 0):.2f}` TP1 `${p.get('tp1', 0):.2f}`"
            )
        pos_text = "\n".join(pos_lines) if pos_lines else "  _Ninguna_"

        msg = (
            f"*📡 Estado — {paper}*\n\n"
            f"Motor: {state_str}\n"
            f"Capital: `${capital:.2f}`\n"
            f"PnL hoy: `{pnl:+.2f}` USD\n"
            f"Drawdown: `{dd:.2%}`\n"
            f"Modelo ML: {ml_ready}\n\n"
            f"*Posiciones abiertas ({len(positions)}):*\n{pos_text}"
        )
        await self._reply(chat_id, msg)

    # ── Helpers HTTP ──────────────────────────────────────────────────────────

    async def _send(self, text: str, pin: bool = False) -> None:
        """Envía un mensaje al canal principal (self._allowed_chat_id)."""
        if not self._ready or not self._session:
            log.debug("telegram_bot_not_ready_skipping_send")
            return
        await self._reply(self._allowed_chat_id, text, pin=pin)

    async def _reply(self, chat_id: int, text: str, pin: bool = False) -> None:
        """Envía un mensaje a un chat_id específico."""
        if not self._session:
            return
        payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
        result = await self._call("sendMessage", payload)

        if pin and result and result.get("ok"):
            msg_id = result["result"]["message_id"]
            await self._call("pinChatMessage", {"chat_id": chat_id, "message_id": msg_id})

    async def _get_updates(self, timeout: int = 5) -> List[Dict[str, Any]]:
        """Hace long-polling a getUpdates."""
        params = {
            "timeout": timeout,
            "offset":  self._offset,
            "allowed_updates": ["message"],
        }
        data = await self._call("getUpdates", params, method="GET")
        return data.get("result", []) if data and data.get("ok") else []

    async def _call(
        self,
        method: str,
        payload: Dict[str, Any],
        http_method: str = "POST",
    ) -> Optional[Dict[str, Any]]:
        """
        Llama a un método de la Telegram Bot API.

        Returns:
            JSON de respuesta o None si hay error de red/HTTP.
        """
        if not self._session:
            return None
        url = self._BASE.format(token=self._token, method=method)
        try:
            if http_method == "GET":
                async with self._session.get(url, params=payload) as resp:
                    return await resp.json()
            else:
                async with self._session.post(url, json=payload) as resp:
                    return await resp.json()
        except aiohttp.ClientError as exc:
            log.warning("telegram_http_error", method=method, error=str(exc))
            return None
        except asyncio.TimeoutError:
            log.warning("telegram_timeout", method=method)
            return None
