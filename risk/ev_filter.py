"""
risk/ev_filter.py — Filtro de Valor Esperado (Expected Value)
=============================================================
Garantiza matemáticamente que cada trade ejecutado tiene una
ventaja estadística positiva antes de entrar al mercado.

Fundamento:
  EV = P(win) × Reward_R - P(loss) × 1.0

  Si EV > 0 → el juego es favorable a largo plazo.
  Si EV ≤ 0 → el trade destruye capital en expectativa.

Ejemplo real:
  Setup con P(win)=0.47, R/R=2.0:
    EV = 0.47 × 2.0 - 0.53 × 1.0 = 0.94 - 0.53 = +0.41R ✅

  Setup con P(win)=0.35, R/R=2.0:
    EV = 0.35 × 2.0 - 0.65 × 1.0 = 0.70 - 0.65 = +0.05R ❌ (muy cerca del break-even)

El filtro aplica un margen de seguridad por encima del break-even matemático
para compensar slippage, comisiones y el error en la estimación de P(win).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class EVResult:
    """Resultado del cálculo de Valor Esperado para un trade."""
    expected_value_r: float         # EV en múltiplos de R
    p_win: float                    # Probabilidad estimada de ganar
    p_loss: float                   # Probabilidad estimada de perder
    reward_r: float                 # Ratio R/R del setup
    breakeven_winrate: float        # Win rate mínimo para EV=0
    passes_filter: bool             # ¿El trade supera el filtro?
    rejection_reason: str = ""      # Razón de rechazo (si aplica)

    @property
    def edge_pct(self) -> float:
        """Ventaja sobre el break-even como porcentaje del riesgo."""
        return self.p_win - self.breakeven_winrate

    def __repr__(self) -> str:
        status = "✅ PASA" if self.passes_filter else f"❌ RECHAZA ({self.rejection_reason})"
        return (
            f"EVResult(EV={self.expected_value_r:+.3f}R, "
            f"P(win)={self.p_win:.1%}, "
            f"BE={self.breakeven_winrate:.1%}, "
            f"edge={self.edge_pct:+.1%}, "
            f"{status})"
        )


class EVFilter:
    """
    Filtro de Valor Esperado para decisiones de trading.

    El filtro aplica tres comprobaciones antes de aprobar un trade:
    1. EV > MIN_EV_R (ventaja positiva con margen de seguridad)
    2. P(win) > breakeven_winrate × (1 + MIN_EDGE_MARGIN)
    3. R/R >= MIN_RR_RATIO (mínimo ratio de recompensa/riesgo)

    Esto garantiza que el sistema solo opera cuando las matemáticas
    están inequívocamente a su favor.
    """

    # Configuración del filtro (conservadora por defecto)
    MIN_EV_R: float         = 0.10   # EV mínimo en R (break-even + 10% de seguridad)
    MIN_EDGE_MARGIN: float  = 0.03   # Margen mínimo sobre el break-even win rate
    MIN_RR_RATIO: float     = 1.5    # Ratio R/R mínimo aceptable
    MAX_RR_RATIO: float     = 8.0    # Ratio R/R máximo (evitar setups irreales)
    MIN_P_WIN: float        = 0.35   # Probabilidad mínima absoluta (evitar Hail Mary)

    def calculate(
        self,
        p_win: float,
        reward_r: float,
        commission_r: float = 0.001,  # ~0.1% por comisión de Binance
    ) -> EVResult:
        """
        Calcula el EV y decide si el trade es ejecutable.

        Args:
            p_win: Probabilidad de ganar (del BayesianSignalScorer)
            reward_r: Ratio R/R del setup (e.g., 2.0 significa TP a 2R)
            commission_r: Coste de comisión en múltiplos de R
                         (aprox. 0.001 para Binance 0.1%)

        Returns:
            EVResult con la decisión y todos los detalles matemáticos
        """
        p_win  = float(np.clip(p_win, 0.0, 1.0))
        p_loss = 1.0 - p_win

        # EV ajustado por comisiones
        ev = p_win * reward_r - p_loss * 1.0 - commission_r

        # Win rate mínimo para EV = 0 (break-even matemático)
        # Solución de: p × R - (1-p) × 1 = 0  →  p = 1/(1+R)
        breakeven_wr = 1.0 / (1.0 + reward_r) if reward_r > 0 else 1.0

        # ── Comprobaciones del filtro ─────────────────────────────────────────
        passes = True
        reason = ""

        if p_win < self.MIN_P_WIN:
            passes = False
            reason = f"P(win)={p_win:.1%} < mínimo {self.MIN_P_WIN:.0%}"

        elif reward_r < self.MIN_RR_RATIO:
            passes = False
            reason = f"R/R={reward_r:.1f} < mínimo {self.MIN_RR_RATIO:.1f}"

        elif reward_r > self.MAX_RR_RATIO:
            passes = False
            reason = f"R/R={reward_r:.1f} > máximo {self.MAX_RR_RATIO:.1f} (setup irreal)"

        elif ev < self.MIN_EV_R:
            passes = False
            reason = f"EV={ev:+.3f}R < mínimo +{self.MIN_EV_R:.2f}R"

        elif (p_win - breakeven_wr) < self.MIN_EDGE_MARGIN:
            passes = False
            reason = (
                f"Edge={p_win - breakeven_wr:+.1%} < "
                f"margen mínimo +{self.MIN_EDGE_MARGIN:.0%}"
            )

        return EVResult(
            expected_value_r=round(ev, 4),
            p_win=round(p_win, 4),
            p_loss=round(p_loss, 4),
            reward_r=round(reward_r, 4),
            breakeven_winrate=round(breakeven_wr, 4),
            passes_filter=passes,
            rejection_reason=reason,
        )

    def batch_evaluate(
        self,
        signals: list,
        p_win_default: float = 0.47,
    ) -> list:
        """
        Evalúa una lista de señales en batch.
        Cada señal debe tener atributos: confidence, risk_reward_ratio_tp1.
        Retorna la lista filtrada solo con los trades válidos.
        """
        approved = []
        for signal in signals:
            p_win   = getattr(signal, "confidence", p_win_default)
            rr      = getattr(signal, "risk_reward_ratio_tp1", 2.0)
            result  = self.calculate(p_win=p_win, reward_r=rr)
            if result.passes_filter:
                approved.append(signal)
        return approved

    @staticmethod
    def breakeven_analysis(rr_range: list = None) -> str:
        """
        Genera una tabla de win rates mínimos necesarios para cada R/R.
        Útil para entender el trade-off entre frecuencia y calidad.
        """
        if rr_range is None:
            rr_range = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]

        lines = ["── Break-Even Win Rate por R/R Ratio ──────────────────"]
        lines.append(f"  {'R/R':>6} {'Break-even WR':>16} {'Min P(win) para EV>0.1R':>24}")
        lines.append("  " + "-" * 48)
        for rr in rr_range:
            be = 1.0 / (1.0 + rr)
            min_p = (0.1 + rr) / (rr + 1.0 + 0.0)  # Aproximación
            min_p = (1.0 + 0.1) / (rr + 1.0)        # p*rr - (1-p) = 0.1
            lines.append(f"  {rr:>6.1f} {be:>15.1%}  {min_p:>23.1%}")
        return "\n".join(lines)


# Instancia global
_ev_filter: Optional[EVFilter] = None


def get_ev_filter() -> EVFilter:
    """Retorna la instancia global del filtro EV (lazy init)."""
    global _ev_filter
    if _ev_filter is None:
        _ev_filter = EVFilter()
    return _ev_filter


def calculate_expected_value(p_win: float, rr_ratio: float) -> float:
    """
    Función de conveniencia para calcular el EV de un trade.
    Retorna el EV en múltiplos de R.
    """
    return get_ev_filter().calculate(p_win, rr_ratio).expected_value_r
