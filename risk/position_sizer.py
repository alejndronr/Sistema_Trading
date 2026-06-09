"""
Position Sizer — Cálculo de Tamaño de Posición
================================================
Implementa exactamente la fórmula del prompt maestro:

  Position Size = (Capital × Riesgo%) / (Precio entrada - Stop Loss)

  Ejemplo: $300 capital, $10 riesgo fijo, entrada $100, SL $97
    → $10 / ($100 - $97) = $10 / $3 = 3.33 unidades

Escala de riesgo por capital (compounding):
  $0 - $1,000      → Riesgo fijo: $10/trade (fase de aprendizaje)
  $1,000 - $5,000  → 1% por trade
  $5,000 - $20,000 → 1.5% por trade
  $20,000+         → 2% por trade (nunca más)

También implementa el circuit breaker de drawdown:
  3% diario  → parar el día
  6% semanal → revisar sistema
  10% mensual → modo solo-estudio
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Dict, Optional

from config.logging_config import get_logger
from config.settings import RISK, SetupQuality

logger = get_logger(__name__)


class DrawdownTracker:
    """Registra pérdidas diarias/semanales/mensuales para el circuit breaker."""

    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.peak_capital = initial_capital
        self.current_capital = initial_capital
        self._daily_start: Dict[date, float] = {}
        self._weekly_start: Dict[int, float] = {}  # ISO week number
        self._monthly_start: Dict[tuple, float] = {}  # (year, month)

    def record_capital(self, capital: float, timestamp: Optional[datetime] = None) -> None:
        """Actualiza el capital actual y registra picos."""
        ts = timestamp or datetime.now(timezone.utc)
        today = ts.date()
        week = ts.isocalendar()[1]
        month = (ts.year, ts.month)

        # Inicializar el capital de inicio de período si no existe
        if today not in self._daily_start:
            self._daily_start[today] = capital
        if week not in self._weekly_start:
            self._weekly_start[week] = capital
        if month not in self._monthly_start:
            self._monthly_start[month] = capital

        self.current_capital = capital
        if capital > self.peak_capital:
            self.peak_capital = capital

    def get_daily_drawdown(self, timestamp: Optional[datetime] = None) -> float:
        """Retorna el drawdown del día actual como fracción (0.03 = 3%)."""
        ts = timestamp or datetime.now(timezone.utc)
        today = ts.date()
        start = self._daily_start.get(today, self.current_capital)
        if start <= 0:
            return 0.0
        return max(0.0, (start - self.current_capital) / start)

    def get_weekly_drawdown(self, timestamp: Optional[datetime] = None) -> float:
        ts = timestamp or datetime.now(timezone.utc)
        week = ts.isocalendar()[1]
        start = self._weekly_start.get(week, self.current_capital)
        if start <= 0:
            return 0.0
        return max(0.0, (start - self.current_capital) / start)

    def get_monthly_drawdown(self, timestamp: Optional[datetime] = None) -> float:
        ts = timestamp or datetime.now(timezone.utc)
        month = (ts.year, ts.month)
        start = self._monthly_start.get(month, self.current_capital)
        if start <= 0:
            return 0.0
        return max(0.0, (start - self.current_capital) / start)

    def get_max_drawdown_from_peak(self) -> float:
        """Drawdown máximo desde el pico histórico."""
        if self.peak_capital <= 0:
            return 0.0
        return max(0.0, (self.peak_capital - self.current_capital) / self.peak_capital)

    def check_circuit_breakers(self, timestamp: Optional[datetime] = None) -> Dict[str, bool]:
        """
        Verifica todos los circuit breakers.
        Retorna dict con {nombre: True si disparado}.
        """
        ts = timestamp or datetime.now(timezone.utc)
        daily_dd = self.get_daily_drawdown(ts)
        weekly_dd = self.get_weekly_drawdown(ts)
        monthly_dd = self.get_monthly_drawdown(ts)

        triggered = {
            "daily": daily_dd >= RISK.max_daily_drawdown_pct,
            "weekly": weekly_dd >= RISK.max_weekly_drawdown_pct,
            "monthly": monthly_dd >= RISK.max_monthly_drawdown_pct,
        }

        for name, is_triggered in triggered.items():
            if is_triggered:
                dd_pct = {"daily": daily_dd, "weekly": weekly_dd, "monthly": monthly_dd}[name]
                logger.warning(
                    "circuit_breaker_triggered",
                    period=name,
                    drawdown_pct=round(dd_pct * 100, 2),
                    capital=self.current_capital,
                )

        return triggered

    def can_trade(self, timestamp: Optional[datetime] = None) -> tuple[bool, str]:
        """
        Verifica si se puede operar según las reglas de drawdown.
        Retorna (puede_operar, razón).
        """
        ts = timestamp or datetime.now(timezone.utc)
        triggered = self.check_circuit_breakers(ts)

        if triggered["monthly"]:
            dd = self.get_monthly_drawdown(ts)
            return False, f"Drawdown mensual {dd*100:.1f}% > {RISK.max_monthly_drawdown_pct*100:.0f}% → MODO SOLO-ESTUDIO"

        if triggered["weekly"]:
            dd = self.get_weekly_drawdown(ts)
            return False, f"Drawdown semanal {dd*100:.1f}% > {RISK.max_weekly_drawdown_pct*100:.0f}% → Revisar sistema"

        if triggered["daily"]:
            dd = self.get_daily_drawdown(ts)
            return False, f"Drawdown diario {dd*100:.1f}% > {RISK.max_daily_drawdown_pct*100:.0f}% → Parar el día"

        return True, "Circuit breakers OK"


class PositionSizer:
    """
    Calcula el tamaño óptimo de posición según las reglas del prompt maestro.
    Incorpora Kelly Criterion fraccional para ajuste dinámico del sizing.
    Thread-safe: cada instancia mantiene su propio estado.
    """

    # ── Configuración Kelly ───────────────────────────────────────────────────
    KELLY_FRACTION: float  = 0.25   # Cuarto-Kelly (conservador, estándar de fondos)
    KELLY_LOOKBACK: int    = 30     # Trades para estimar Kelly dinámico
    KELLY_MIN_TRADES: int  = 10     # Mínimo de trades para activar Kelly

    def __init__(self, initial_capital: float = RISK.initial_capital):
        self.capital = initial_capital
        self.drawdown_tracker = DrawdownTracker(initial_capital)
        self._open_positions: int = 0
        # Historial rolling de R-múltiples para Kelly dinámico
        self._trade_r_history: list = []

    def update_capital(self, new_capital: float, timestamp: Optional[datetime] = None) -> None:
        """Actualiza el capital tras cerrar una posición."""
        self.capital = new_capital
        self.drawdown_tracker.record_capital(new_capital, timestamp)

    def record_trade_result(self, r_multiple: float) -> None:
        """
        Registra el resultado de un trade en R-múltiples para el cálculo Kelly.
        Mantiene solo los últimos KELLY_LOOKBACK trades (rolling window).
        """
        self._trade_r_history.append(r_multiple)
        if len(self._trade_r_history) > self.KELLY_LOOKBACK:
            self._trade_r_history.pop(0)

    def kelly_fraction(self) -> float:
        """
        Calcula el porcentaje óptimo de capital a arriesgar según Kelly Criterion.

        Fórmula: Kelly% = (WR × AvgWin_R - LR × AvgLoss_R) / AvgWin_R
        Se aplica el cuarto-Kelly (× 0.25) para máxima seguridad.

        Devuelve 1.0 (sin ajuste) si no hay suficientes datos.
        El resultado está capeado en [0.5, 2.0] para evitar extremos.
        """
        if len(self._trade_r_history) < self.KELLY_MIN_TRADES:
            return 1.0   # Sin datos suficientes: usar sizing estándar

        r_arr = [r for r in self._trade_r_history]
        wins   = [r for r in r_arr if r > 0]
        losses = [r for r in r_arr if r <= 0]

        if not wins or not losses:
            return 1.0

        win_rate  = len(wins) / len(r_arr)
        loss_rate = 1.0 - win_rate
        avg_win_r = sum(wins) / len(wins)
        avg_loss_r = abs(sum(losses) / len(losses))  # Positivo

        if avg_win_r <= 0 or avg_loss_r <= 0:
            return 1.0

        # Kelly% = (WR × b - LR) / b  donde b = avg_win / avg_loss en R
        b = avg_win_r / avg_loss_r
        kelly_pct = (win_rate * b - loss_rate) / b

        # Cuarto-Kelly para máxima robustez
        fractional_kelly = kelly_pct * self.KELLY_FRACTION

        # Normalizar como multiplicador del sizing estándar
        # Kelly% / riesgo_base_pct da el multiplicador
        base_risk_pct = RISK.get_risk_amount(self.capital) / self.capital
        kelly_multiplier = fractional_kelly / base_risk_pct if base_risk_pct > 0 else 1.0

        # Capear en [0.5, 2.0]: nunca reducir más del 50%, nunca doblar
        return round(max(0.5, min(2.0, kelly_multiplier)), 3)

    def get_risk_amount(self, quality: SetupQuality = SetupQuality.A, confidence: float = 0.5) -> float:
        """
        Retorna el monto en riesgo según la escala de capital del prompt maestro,
        ajustado dinámicamente por el Kelly Criterion.

        $0-$1,000:      $10 fijo (aprendizaje)
        $1,000-$5,000:  1%
        $5,000-$20,000: 1.5%
        $20,000+:       2%

        Setups A+: hasta 2x el riesgo base (nunca supera 2% del capital).
        Kelly ajusta el resultado final (×0.5 a ×2.0 según historial).
        """
        base_risk = RISK.get_risk_amount(self.capital)

        if quality == SetupQuality.A_PLUS:
            # A+ puede usar hasta 2x, con cap en 2% del capital
            base_risk = min(base_risk * 2, self.capital * 0.02)

        # Proxy de Kelly basado en Confidence (Score Bayesiano)
        # Confidence va de ~0.40 a 0.95. 0.5 = neutro (1x). 0.8+ = agresivo (hasta 2x).
        kelly_confidence_mult = max(0.5, min(2.0, confidence * 2.0))

        # Ajuste Kelly dinámico basado en historial
        kelly_history_mult = self.kelly_fraction()
        
        # Combinamos ambos (promedio)
        kelly_mult = (kelly_confidence_mult + kelly_history_mult) / 2.0
        
        kelly_adjusted = base_risk * kelly_mult

        # Límite absoluto: nunca más del 2% del capital
        max_risk = self.capital * 0.02
        return min(kelly_adjusted, max_risk)

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss_price: float,
        quality: SetupQuality = SetupQuality.A,
        symbol_priority: int = 1,
        regime_multiplier: float = 1.0,
        confidence: float = 0.5,
    ) -> Dict:
        """
        Calcula el tamaño de posición completo.
        """
        if entry_price <= 0 or stop_loss_price <= 0:
            raise ValueError("Precios inválidos (deben ser > 0)")

        price_diff = abs(entry_price - stop_loss_price)
        if price_diff <= 0:
            raise ValueError("Diferencia precio-SL debe ser > 0")

        risk_amount = self.get_risk_amount(quality, confidence) * regime_multiplier

        # Position Size = Riesgo / (Entrada - SL)
        position_size = risk_amount / price_diff

        # Valor de la posición en USD
        position_value = position_size * entry_price

        # En spot: no podemos gastar más capital del que tenemos.
        # Reducir el tamaño si el notional supera el capital disponible.
        max_position_value = self.capital * 0.95  # Dejar 5% como buffer para comisiones
        if position_value > max_position_value:
            position_size = max_position_value / entry_price
            position_value = max_position_value

        result = {
            "position_size": round(position_size, 6),
            "position_value_usd": round(position_value, 2),
            "risk_amount": round(risk_amount, 2),
            "risk_pct": round(risk_amount / self.capital * 100, 2),
            "entry_price": entry_price,
            "stop_loss": stop_loss_price,
            "price_risk_per_unit": round(price_diff, 8),
            "capital": self.capital,
            "setup_quality": quality.value,
        }

        logger.info(
            "position_sized",
            **{k: v for k, v in result.items() if k != "capital"},
        )

        return result

    def can_open_position(self) -> tuple[bool, str]:
        """
        Verifica si se puede abrir una nueva posición.
        Checks: max 3 posiciones simultáneas + circuit breakers.
        """
        if self._open_positions >= RISK.max_open_positions:
            return False, f"Máximo de posiciones simultáneas alcanzado ({RISK.max_open_positions})"

        can_trade, reason = self.drawdown_tracker.can_trade()
        if not can_trade:
            return False, reason

        return True, "OK"

    def register_position_opened(self) -> None:
        """Registra la apertura de una posición."""
        self._open_positions += 1

    def register_position_closed(self, pnl: float, timestamp: Optional[datetime] = None) -> None:
        """Registra el cierre de una posición y actualiza el capital."""
        self._open_positions = max(0, self._open_positions - 1)
        self.update_capital(self.capital + pnl, timestamp)

    @property
    def open_positions(self) -> int:
        return self._open_positions

    def get_status(self) -> Dict:
        """Retorna un resumen del estado actual del position sizer."""
        return {
            "capital": round(self.capital, 2),
            "open_positions": self._open_positions,
            "risk_per_trade": round(self.get_risk_amount(), 2),
            "daily_drawdown_pct": round(self.drawdown_tracker.get_daily_drawdown() * 100, 2),
            "weekly_drawdown_pct": round(self.drawdown_tracker.get_weekly_drawdown() * 100, 2),
            "max_drawdown_pct": round(self.drawdown_tracker.get_max_drawdown_from_peak() * 100, 2),
            "peak_capital": round(self.drawdown_tracker.peak_capital, 2),
        }


def calculate_position_size(
    entry_price: float,
    stop_loss_price: float,
    quality: SetupQuality = SetupQuality.A,
    symbol_priority: int = 1,
    regime_multiplier: float = 1.0,
    capital: float = RISK.initial_capital,
) -> Dict:
    """Wrapper a nivel de módulo para calcular tamaño de posición."""
    sizer = PositionSizer(initial_capital=capital)
    return sizer.calculate_position_size(
        entry_price=entry_price,
        stop_loss_price=stop_loss_price,
        quality=quality,
        symbol_priority=symbol_priority,
        regime_multiplier=regime_multiplier,
    )
