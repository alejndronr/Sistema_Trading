"""monitoring/telegram_bot.py — Bot de control institucional para Paper‑Trading.

Funcionalidades:
    Alertas pasivas (llamadas directas, sin polling):
        • alert_trade_executed()    — trade abierto por el motor
        • alert_signal_rejected()   — señal rechazada por el filtro ML
        • alert_daily_summary()     — resumen nocturno de PnL y drawdown

    Comandos activos (polling en background):
        /status   — capital, posiciones abiertas, PnL del día, drawdown, modelo ML
        /pause    — pausa nuevas entradas sin cerrar posiciones existentes
        /resume   — reanuda la operativa normal
        /kill     — solicita confirmación y apaga el motor elegantemente

Seguridad:
    • Solo responde a TELEGRAM_ALLOWED_USER_ID (variable de entorno)
    • Ante cualquier remitente no autorizado, respuesta genérica sin info

Dependencias:
    pip install python-telegram-bot>=20.7

Uso:
    Ejecutar `start_command_listener(engine_ref, portfolio_ref)` como
    tarea asyncio ANTES de llamar a `live_engine.run()`.

    from monitoring.telegram_bot import start_command_listener
    asyncio.create_task(start_command_listener(engine, portfolio))
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from typing import Any, Mapping, Optional

import requests

from config.settings import env_settings, ML_CONFIG

log = logging.getLogger(__name__)

# ── Constantes de seguridad ────────────────────────────────────────────────────
# Se carga desde .env; si no existe, el bot acepta todos los usuarios (peligroso)
_ALLOWED_USER_ID: Optional[int] = (
    int(os.environ.get("TELEGRAM_ALLOWED_USER_ID", 0)) or None
)

# Estado compartido de pausa (consultado por live_engine.signal_worker)
_PAUSED: bool = False
# Referencia al kill‑switch del motor (se inyecta al iniciar el listener)
_kill_switch_ref: Any = None
# Referencia a la cartera virtual (para /status y /kill)
_portfolio_ref: Any = None
# Token de confirmación de kill en curso (evita kills accidentales)
_pending_kill: bool = False


def is_paused() -> bool:
    """Devuelve True si el motor está en pausa (sin nuevas entradas)."""
    return _PAUSED


# ══════════════════════════════════════════════════════════════════════════════
# Low‑level HTTP helpers
# ══════════════════════════════════════════════════════════════════════════════

def _token() -> str:
    return env_settings.telegram_bot_token

def _chat_id() -> str:
    return env_settings.telegram_chat_id


def _post(method: str, payload: Mapping[str, Any]) -> Optional[dict]:
    """POST a la Telegram Bot API. Devuelve el JSON de respuesta o None."""
    if not _token() or not _chat_id():
        log.warning("Telegram credentials no configuradas — omitiendo notificación.")
        return None
    url = f"https://api.telegram.org/bot{_token()}/{method}"
    data = {"chat_id": _chat_id(), **payload}
    try:
        resp = requests.post(url, json=data, timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        log.error("Telegram POST falló (%s): %s", method, exc)
        return None


def _reply(chat_id: int | str, text: str, parse_mode: str = "Markdown") -> None:
    """Responde a un chat específico (puede ser diferente al canal de alertas)."""
    if not _token():
        return
    url = f"https://api.telegram.org/bot{_token()}/sendMessage"
    try:
        requests.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": parse_mode}, timeout=5)
    except Exception as exc:
        log.error("_reply falló: %s", exc)


# ══════════════════════════════════════════════════════════════════════════════
# Alertas pasivas (llamadas directas desde el motor)
# ══════════════════════════════════════════════════════════════════════════════

def alert_trade_executed(
    symbol: str,
    direction: str,
    entry_price: float,
    stop_loss: float,
    take_profit_1: float,
    ml_probability: float,
) -> None:
    """Alerta cuando un trade pasa el filtro ML y se abre la posición virtual."""
    rr = round(abs(take_profit_1 - entry_price) / abs(entry_price - stop_loss), 2)
    emoji = "🟢" if direction.upper() == "LONG" else "🔴"
    txt = (
        f"{emoji} *Trade ejecutado*\n"
        f"*Par*: `{symbol}`\n"
        f"*Dirección*: `{direction}`\n"
        f"*Entrada*: `{entry_price:.2f}`\n"
        f"*Stop‑Loss*: `{stop_loss:.2f}`\n"
        f"*Take‑Profit*: `{take_profit_1:.2f}` (R/R `{rr}x`)\n"
        f"*Confianza ML*: `{ml_probability:.1%}` ✅"
    )
    _post("sendMessage", {"text": txt, "parse_mode": "Markdown"})


def alert_signal_rejected(
    symbol: str,
    ml_probability: float,
    threshold: float | None = None,
) -> None:
    """Alerta silenciosa cuando una señal es rechazada por el filtro ML."""
    thr = threshold if threshold is not None else ML_CONFIG.confidence_threshold
    txt = (
        f"⚠️ *Señal rechazada*\n"
        f"`{symbol}` — Prob ML: `{ml_probability:.1%}` (umbral: `{thr:.0%}`)"
    )
    _post("sendMessage", {"text": txt, "parse_mode": "Markdown"})


def alert_trade_closed(
    symbol: str,
    direction: str,
    entry_price: float,
    exit_price: float,
    pnl_usd: float,
    exit_reason: str,
) -> None:
    """Alerta cuando el motor cierra una posición (SL o TP)."""
    emoji = "✅" if pnl_usd >= 0 else "🛑"
    txt = (
        f"{emoji} *Posición cerrada*\n"
        f"*Par*: `{symbol}` | *Dir*: `{direction}`\n"
        f"*Entrada*: `{entry_price:.2f}` → *Salida*: `{exit_price:.2f}`\n"
        f"*PnL*: `{pnl_usd:+.2f}` USD\n"
        f"*Razón*: `{exit_reason}`"
    )
    _post("sendMessage", {"text": txt, "parse_mode": "Markdown"})


def alert_daily_summary(
    date: datetime,
    pnl_usd: float,
    drawdown_pct: float,
    open_trades: int,
) -> None:
    """Resumen nocturno enviado al cierre de cada sesión."""
    trend = "📈" if pnl_usd >= 0 else "📉"
    txt = (
        f"*📊 Resumen diario — {date:%Y‑%m‑%d}*\n"
        f"{trend} *PnL del día*: `{pnl_usd:+.2f}` USD\n"
        f"*Drawdown actual*: `{drawdown_pct:.2%}`\n"
        f"*Posiciones abiertas*: `{open_trades}`"
    )
    _post("sendMessage", {"text": txt, "parse_mode": "Markdown"})


def alert_engine_stopped(reason: str = "kill switch") -> None:
    """Notifica que el motor se ha detenido."""
    _post("sendMessage", {
        "text": f"⛔️ *Motor detenido*\nRazón: `{reason}`",
        "parse_mode": "Markdown",
    })


# ══════════════════════════════════════════════════════════════════════════════
# Polling loop — escucha comandos activos del operador
# ══════════════════════════════════════════════════════════════════════════════

async def start_command_listener(kill_switch, portfolio) -> None:
    """
    Corrutina asyncio que hace long‑polling a la Bot API y despacha
    los comandos /status, /pause, /resume y /kill.

    Args:
        kill_switch: instancia de ``execution.failsafes.KillSwitch``
        portfolio:   instancia de ``execution.paper_portfolio.PaperPortfolio``
    """
    global _kill_switch_ref, _portfolio_ref, _PAUSED, _pending_kill
    _kill_switch_ref = kill_switch
    _portfolio_ref   = portfolio

    offset: int = 0
    log.info("Telegram command listener iniciado.")

    while not kill_switch.is_active():
        updates = await _get_updates(offset)
        for update in updates:
            offset = update["update_id"] + 1
            await _dispatch(update)
        await asyncio.sleep(2)   # polling cada 2 segundos


async def _get_updates(offset: int) -> list:
    """Llama a getUpdates con long‑polling de 30 segundos."""
    if not _token():
        await asyncio.sleep(10)
        return []
    url = f"https://api.telegram.org/bot{_token()}/getUpdates"
    params = {"timeout": 30, "offset": offset, "allowed_updates": ["message"]}
    try:
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None,
            lambda: requests.get(url, params=params, timeout=35),
        )
        data = resp.json()
        return data.get("result", []) if data.get("ok") else []
    except Exception as exc:
        log.debug("getUpdates error: %s", exc)
        return []


async def _dispatch(update: dict) -> None:
    """Identifica el comando y lo despacha al handler correspondiente."""
    global _pending_kill

    message = update.get("message", {})
    if not message:
        return

    from_user = message.get("from", {})
    user_id   = from_user.get("id")
    chat_id   = message.get("chat", {}).get("id")
    text      = (message.get("text") or "").strip()

    # ── Verificación de seguridad ──────────────────────────────────────────
    if _ALLOWED_USER_ID and user_id != _ALLOWED_USER_ID:
        _reply(chat_id, "ℹ️ Sistema de trading — canal privado.")
        log.warning("Comando recibido de user_id no autorizado: %d", user_id)
        return

    log.info("Comando recibido de uid=%d: %s", user_id, text[:80])

    # ── CONFIRMAR kill pendiente ───────────────────────────────────────────
    if _pending_kill:
        if text.upper() == "CONFIRMAR":
            _pending_kill = False
            await _handle_kill_confirmed(chat_id)
        else:
            _pending_kill = False
            _reply(chat_id, "✅ Kill cancelado. El motor sigue activo.")
        return

    # ── Despachar comando ──────────────────────────────────────────────────
    cmd = text.split()[0].lower() if text else ""

    if cmd == "/status":
        await _cmd_status(chat_id)
    elif cmd == "/pause":
        await _cmd_pause(chat_id)
    elif cmd == "/resume":
        await _cmd_resume(chat_id)
    elif cmd == "/kill":
        await _cmd_kill_request(chat_id)
    elif cmd.startswith("/"):
        _reply(chat_id, "ℹ️ Comando no reconocido. Comandos disponibles: /status /pause /resume /kill")
    # Mensajes sin "/" se ignoran en silencio


# ── Handlers de cada comando ─────────────────────────────────────────────────

async def _cmd_status(chat_id: int) -> None:
    """Muestra el estado completo del sistema."""
    from pathlib import Path
    from config.settings import ML_CONFIG

    portfolio = _portfolio_ref
    open_trades = portfolio.get_open_trades() if portfolio else []
    capital     = portfolio.capital if portfolio else 0.0

    # Comprobar si existe el modelo ML entrenado
    model_path = ML_CONFIG.model_dir / ML_CONFIG.model_filename
    model_ok   = "✅ Cargado" if model_path.exists() else "❌ No encontrado"
    mtime      = (
        datetime.fromtimestamp(model_path.stat().st_mtime).strftime("%Y-%m-%d")
        if model_path.exists() else "—"
    )

    engine_state = "⏸ PAUSADO" if _PAUSED else "▶️ ACTIVO"

    trades_txt = ""
    for t in open_trades:
        trades_txt += (
            f"  • `{t['symbol']}` {t['direction']} @ `{t['entry_price']:.2f}`"
            f"  SL `{t['stop_loss']:.2f}`  TP `{t['take_profit_1']:.2f}`\n"
        )
    trades_txt = trades_txt or "  _Ninguna_"

    txt = (
        f"*📡 Estado del sistema — {datetime.utcnow():%Y‑%m‑%d %H:%M} UTC*\n\n"
        f"*Capital*: `${capital:.2f}`\n"
        f"*Motor*: {engine_state}\n"
        f"*Modelo ML*: {model_ok} (actualizado: {mtime})\n"
        f"*Umbral ML*: `{ML_CONFIG.confidence_threshold:.0%}`\n\n"
        f"*Posiciones abiertas* ({len(open_trades)}):\n{trades_txt}"
    )
    _reply(chat_id, txt)


async def _cmd_pause(chat_id: int) -> None:
    """Pausa nuevas entradas sin cerrar posiciones existentes."""
    global _PAUSED
    if _PAUSED:
        _reply(chat_id, "ℹ️ El motor ya está en pausa. Usa /resume para reanudar.")
        return
    _PAUSED = True
    log.warning("Motor PAUSADO por comando Telegram (uid relacionado).")
    _reply(chat_id,
        "⏸ *Motor pausado*\n"
        "No se abrirán nuevas posiciones.\n"
        "Las posiciones existentes siguen monitorizadas.\n"
        "Usa /resume para reanudar."
    )


async def _cmd_resume(chat_id: int) -> None:
    """Reanuda la operativa normal."""
    global _PAUSED
    if not _PAUSED:
        _reply(chat_id, "ℹ️ El motor ya está activo. No hay nada que reanudar.")
        return
    _PAUSED = False
    log.info("Motor REANUDADO por comando Telegram.")
    _reply(chat_id, "▶️ *Motor reanudado* — volviendo a la operativa normal.")


async def _cmd_kill_request(chat_id: int) -> None:
    """Solicita confirmación antes de ejecutar el kill switch."""
    global _pending_kill
    _pending_kill = True
    _reply(
        chat_id,
        "⛔️ *KILL SWITCH*\n\n"
        "Esta acción:\n"
        "1. Cierra todas las posiciones a precio de mercado\n"
        "2. Detiene el motor de trading\n"
        "3. Guarda el estado en la base de datos\n\n"
        "*¿Estás seguro?* Responde exactamente: `CONFIRMAR`\n"
        "_(Cualquier otra respuesta cancela el kill)_"
    )


async def _handle_kill_confirmed(chat_id: int) -> None:
    """Ejecuta el kill switch: cierra posiciones y para el motor."""
    global _kill_switch_ref, _portfolio_ref

    _reply(chat_id, "⛔️ Ejecutando kill switch — cerrando posiciones...")
    log.critical("KILL SWITCH activado por comando Telegram.")

    # 1️⃣ Obtener precio actual para cerrar posiciones a mercado
    portfolio = _portfolio_ref
    if portfolio:
        open_trades = portfolio.get_open_trades()
        if open_trades:
            # Intentamos obtener el precio actual vía ccxt
            try:
                import ccxt
                exchange = ccxt.binance({"enableRateLimit": True})
                for trade in list(open_trades):
                    symbol = trade["symbol"]
                    ticker = exchange.fetch_ticker(symbol)
                    price  = float(ticker["last"])
                    portfolio.close_trade(
                        trade_id=trade["trade_id"],
                        exit_price=price,
                        exit_reason="kill_switch",
                    )
                    log.info("Posición %s cerrada a %.2f por kill switch.", symbol, price)
                exchange.close()
            except Exception as exc:
                log.error("Error cerrando posiciones en kill switch: %s", exc)
                _reply(chat_id, f"⚠️ Error cerrando posiciones: `{exc}`")
        else:
            log.info("Kill switch: no había posiciones abiertas.")

    # 2️⃣ Activar el kill switch del motor
    if _kill_switch_ref:
        _kill_switch_ref.activate()

    _reply(
        chat_id,
        "✅ *Kill switch ejecutado*\n"
        "Todas las posiciones cerradas. Motor detenido.\n"
        "El servicio se reiniciará en 30s (RestartSec en systemd).\n"
        "Para evitar el reinicio: `sudo systemctl stop trading-engine`"
    )
    alert_engine_stopped("kill switch via Telegram")
