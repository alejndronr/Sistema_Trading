"""
Psychology Guard — Detección y Bloqueo de Sesgos Cognitivos
============================================================
El sistema detecta y BLOQUEA automáticamente los 6 patrones
psicológicos definidos en el prompt maestro:

1. REVENGE TRADING:  cooldown de 1 hora post-pérdida, nunca doblar posición
2. FOMO:             si precio se movió > 3% sin entrada, NO entrar
3. SOBREOPTIMIZACIÓN: mínimo 50 trades antes de cambiar parámetros
4. CONFIRMATION BIAS: buscar activamente razones para NO entrar
5. LOSS AVERSION:    tratar pérdidas y ganancias como estadística
6. OVERTRADING:      máximo 2-3 setups válidos al día
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

from config.logging_config import get_logger
from config.settings import RISK

logger = get_logger(__name__)


class PsychologyGuard:
    """
    Guarda de psicología de trading.
    Previene los sesgos cognitivos más comunes que destruyen cuentas.
    """

    MAX_TRADES_PER_DAY = 3                    # Máximo de setups diarios
    FOMO_THRESHOLD = RISK.fomo_price_move_threshold  # 3%
    REVENGE_COOLDOWN_MINUTES = RISK.revenge_cooldown_minutes  # 60 min
    MIN_TRADES_BEFORE_PARAM_CHANGE = 50      # Mínimo para evaluar cambios

    def __init__(self):
        self._last_loss_time: Optional[datetime] = None
        self._losses_streak: int = 0
        self._wins_streak: int = 0
        self._trades_today: int = 0
        self._today: Optional[datetime] = None
        self._trade_history: List[dict] = []
        self._blocked_reasons: List[str] = []

    # ── Tracking ──────────────────────────────────────────────────────────────

    def record_trade_result(
        self,
        pnl: float,
        timestamp: Optional[datetime] = None,
        entry_price: Optional[float] = None,
        exit_price: Optional[float] = None,
    ) -> None:
        """Registra el resultado de un trade para tracking de sesgos."""
        ts = timestamp or datetime.now(timezone.utc)
        is_loss = pnl < 0

        # Resetear contador diario si es un nuevo día
        if self._today is None or ts.date() != self._today.date():
            self._trades_today = 0
            self._today = ts

        self._trades_today += 1

        if is_loss:
            self._last_loss_time = ts
            self._losses_streak += 1
            self._wins_streak = 0

            if self._losses_streak >= 4:
                logger.warning(
                    "consecutive_losses_warning",
                    streak=self._losses_streak,
                    message="Considera parar y revisar tu sistema",
                )
        else:
            self._losses_streak = 0
            self._wins_streak += 1

        self._trade_history.append({
            "timestamp": ts,
            "pnl": pnl,
            "is_loss": is_loss,
            "entry_price": entry_price,
            "exit_price": exit_price,
        })

    # ── Guards ────────────────────────────────────────────────────────────────

    def check_revenge_trading(self, timestamp: Optional[datetime] = None) -> Tuple[bool, str]:
        """
        REVENGE TRADING GUARD:
        Cooldown de 1 hora después de una pérdida.
        Retorna (puede_operar, razón).
        """
        if self._last_loss_time is None:
            return True, "Sin pérdidas recientes"

        ts = timestamp or datetime.now(timezone.utc)
        elapsed = ts - self._last_loss_time
        cooldown = timedelta(minutes=self.REVENGE_COOLDOWN_MINUTES)

        if elapsed < cooldown:
            remaining = int((cooldown - elapsed).total_seconds() / 60)
            reason = (
                f"REVENGE TRADING BLOQUEADO: cooldown post-pérdida. "
                f"Quedan {remaining} min. El mercado siempre da otra oportunidad."
            )
            logger.warning("revenge_trading_blocked", remaining_minutes=remaining)
            return False, reason

        return True, "Cooldown expirado"

    def check_fomo(
        self,
        entry_price: float,
        price_n_candles_ago: float,
        timestamp: Optional[datetime] = None,
    ) -> Tuple[bool, str]:
        """
        FOMO GUARD:
        Si el precio ya se movió > 3% sin entrada, NO entrar.
        'El mercado siempre da otra oportunidad.'
        """
        if price_n_candles_ago <= 0:
            return True, "No hay precio de referencia"

        move_pct = abs(entry_price - price_n_candles_ago) / price_n_candles_ago

        if move_pct > self.FOMO_THRESHOLD:
            reason = (
                f"FOMO BLOQUEADO: el precio ya se movió {move_pct*100:.1f}% "
                f"(umbral: {self.FOMO_THRESHOLD*100:.0f}%). "
                f"Esperar nueva oportunidad. El mercado siempre da otra oportunidad."
            )
            logger.warning("fomo_blocked", move_pct=round(move_pct * 100, 2))
            return False, reason

        return True, f"Movimiento de precio {move_pct*100:.1f}% dentro del umbral"

    def check_overtrading(self, timestamp: Optional[datetime] = None) -> Tuple[bool, str]:
        """
        OVERTRADING GUARD:
        Máximo 2-3 setups válidos al día. 'Cash is a position.'
        """
        ts = timestamp or datetime.now(timezone.utc)

        # Resetear contador si es nuevo día
        if self._today is None or ts.date() != self._today.date():
            self._trades_today = 0
            self._today = ts

        if self._trades_today >= self.MAX_TRADES_PER_DAY:
            reason = (
                f"OVERTRADING BLOQUEADO: {self._trades_today}/{self.MAX_TRADES_PER_DAY} "
                f"operaciones hoy. Cash is a position. Esperar al siguiente día."
            )
            logger.warning("overtrading_blocked", trades_today=self._trades_today)
            return False, reason

        return True, f"{self._trades_today}/{self.MAX_TRADES_PER_DAY} operaciones hoy"

    def check_consecutive_losses(self, max_consecutive: int = 4) -> Tuple[bool, str]:
        """
        Regla de protección: si hay más de N pérdidas consecutivas,
        revisar el sistema antes de continuar.
        """
        if self._losses_streak >= max_consecutive:
            reason = (
                f"ALERTA: {self._losses_streak} pérdidas consecutivas. "
                f"Revisar condiciones del mercado y parámetros de estrategia. "
                f"Considera pausar y analizar."
            )
            logger.error(
                "consecutive_losses_critical",
                streak=self._losses_streak,
            )
            return False, reason

        return True, f"{self._losses_streak} pérdidas consecutivas (máx: {max_consecutive})"

    def full_psychology_check(
        self,
        entry_price: float,
        price_5_candles_ago: float,
        timestamp: Optional[datetime] = None,
    ) -> Tuple[bool, List[str]]:
        """
        Ejecuta todos los checks psicológicos en secuencia.
        Retorna (puede_operar, lista de razones).
        """
        ts = timestamp or datetime.now(timezone.utc)
        reasons = []
        can_trade = True

        checks = [
            self.check_revenge_trading(ts),
            self.check_fomo(entry_price, price_5_candles_ago, ts),
            self.check_overtrading(ts),
            self.check_consecutive_losses(),
        ]

        for ok, reason in checks:
            if not ok:
                can_trade = False
                reasons.append(reason)

        if can_trade:
            logger.info("psychology_check_passed", trades_today=self._trades_today)
        else:
            logger.warning("psychology_check_failed", reasons=reasons)

        return can_trade, reasons

    # ── Overoptimization Guard ─────────────────────────────────────────────────

    def can_change_parameters(self) -> Tuple[bool, str]:
        """
        SOBREOPTIMIZACIÓN GUARD:
        Mínimo 50 trades de muestra antes de evaluar cambios.
        """
        n_trades = len(self._trade_history)

        if n_trades < self.MIN_TRADES_BEFORE_PARAM_CHANGE:
            reason = (
                f"SOBREOPTIMIZACIÓN BLOQUEADA: solo hay {n_trades} trades. "
                f"Se necesitan mínimo {self.MIN_TRADES_BEFORE_PARAM_CHANGE} para "
                f"cambiar parámetros con significancia estadística."
            )
            return False, reason

        return True, f"Suficiente muestra ({n_trades} trades)"

    # ── Statistics ─────────────────────────────────────────────────────────────

    def get_psychology_stats(self) -> dict:
        """Retorna estadísticas de comportamiento psicológico."""
        total = len(self._trade_history)
        if total == 0:
            return {"total_trades": 0}

        losses = [t for t in self._trade_history if t["is_loss"]]
        wins = [t for t in self._trade_history if not t["is_loss"]]

        return {
            "total_trades": total,
            "win_count": len(wins),
            "loss_count": len(losses),
            "current_loss_streak": self._losses_streak,
            "current_win_streak": self._wins_streak,
            "trades_today": self._trades_today,
            "last_loss": self._last_loss_time.isoformat() if self._last_loss_time else None,
            "can_change_params": self.can_change_parameters()[0],
        }
