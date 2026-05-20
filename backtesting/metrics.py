"""
Métricas de Rendimiento del Backtest
=====================================
Calcula todas las métricas definidas en el prompt maestro:

  - Win Rate (objetivo ≥ 45%)
  - Risk/Reward promedio (objetivo ≥ 2:1)
  - Expectancy = (WR × Avg Win) - (LR × Avg Loss)
  - Profit Factor = Ganancias brutas / Pérdidas brutas (objetivo > 1.5)
  - Sharpe Ratio mensual y anual (objetivo > 1.0)
  - Max Drawdown
  - Consecutivas pérdidas máximas
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class BacktestMetrics:
    """Contenedor de todas las métricas de rendimiento."""

    # ── Core Metrics (del prompt maestro) ─────────────────────────────────────
    total_trades: int = 0
    win_rate: float = 0.0              # % trades ganadores
    loss_rate: float = 0.0
    win_count: int = 0
    loss_count: int = 0

    avg_win_usd: float = 0.0           # Ganancia promedio en USD
    avg_loss_usd: float = 0.0          # Pérdida promedio en USD
    avg_r_multiple: float = 0.0        # R promedio por trade
    avg_win_r: float = 0.0
    avg_loss_r: float = 0.0

    expectancy: float = 0.0            # Expectancy en USD/trade
    profit_factor: float = 0.0        # Ganancias brutas / Pérdidas brutas
    total_return_pct: float = 0.0
    total_pnl_usd: float = 0.0

    # ── Risk Metrics ──────────────────────────────────────────────────────────
    max_drawdown_pct: float = 0.0      # Max drawdown como fracción (0.10 = 10%)
    max_drawdown_usd: float = 0.0
    max_drawdown_duration_days: float = 0.0
    avg_drawdown_pct: float = 0.0

    # ── Time Metrics ──────────────────────────────────────────────────────────
    sharpe_ratio_monthly: float = 0.0
    sharpe_ratio_annual: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # ── Streak Analysis ───────────────────────────────────────────────────────
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    avg_trade_duration_hours: float = 0.0

    # ── Breakdown por Exit Reason ──────────────────────────────────────────────
    sl_hit_count: int = 0
    tp1_hit_count: int = 0
    tp2_hit_count: int = 0

    # ── Breakdown por Setup Quality ────────────────────────────────────────────
    quality_stats: dict = None  # type: ignore

    @classmethod
    def from_trades(
        cls,
        trades_df: pd.DataFrame,
        equity_curve: pd.DataFrame,
        initial_capital: float,
    ) -> "BacktestMetrics":
        """
        Calcula todas las métricas a partir de un DataFrame de trades cerrados.
        """
        m = cls()

        if trades_df is None or trades_df.empty:
            return m

        m.total_trades = len(trades_df)
        if m.total_trades == 0:
            return m

        pnl = trades_df["pnl_usd"].values
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]

        m.win_count = len(wins)
        m.loss_count = len(losses)
        m.win_rate = m.win_count / m.total_trades if m.total_trades > 0 else 0
        m.loss_rate = 1.0 - m.win_rate

        m.avg_win_usd = float(np.mean(wins)) if len(wins) > 0 else 0.0
        m.avg_loss_usd = float(np.mean(losses)) if len(losses) > 0 else 0.0
        m.total_pnl_usd = float(np.sum(pnl))
        m.total_return_pct = m.total_pnl_usd / initial_capital * 100

        # Expectancy
        m.expectancy = (
            m.win_rate * m.avg_win_usd + m.loss_rate * m.avg_loss_usd
        )

        # Profit Factor
        gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
        gross_loss = float(abs(np.sum(losses))) if len(losses) > 0 else 0.0001
        m.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # R multiple
        if "r_multiple" in trades_df.columns:
            r = trades_df["r_multiple"].values
            m.avg_r_multiple = float(np.mean(r))
            wins_r = r[r > 0]
            losses_r = r[r < 0]
            m.avg_win_r = float(np.mean(wins_r)) if len(wins_r) > 0 else 0
            m.avg_loss_r = float(np.mean(losses_r)) if len(losses_r) > 0 else 0

        # Duration
        if "duration_hours" in trades_df.columns:
            m.avg_trade_duration_hours = float(trades_df["duration_hours"].mean())

        # Streaks
        m.max_consecutive_wins, m.max_consecutive_losses = cls._calculate_streaks(pnl)

        # Exit reasons
        if "exit_reason" in trades_df.columns:
            reasons = trades_df["exit_reason"].value_counts()
            m.sl_hit_count = int(reasons.get("stop_loss", 0))
            m.tp1_hit_count = int(reasons.get("take_profit_1", 0))
            m.tp2_hit_count = int(reasons.get("take_profit_2", 0))

        # Quality breakdown
        if "setup_quality" in trades_df.columns:
            m.quality_stats = {}
            for quality in ["A+", "A", "B", "C"]:
                qt = trades_df[trades_df["setup_quality"] == quality]
                if not qt.empty:
                    qt_wins = qt[qt["pnl_usd"] > 0]
                    m.quality_stats[quality] = {
                        "count": len(qt),
                        "win_rate": len(qt_wins) / len(qt) if len(qt) > 0 else 0,
                        "avg_pnl": float(qt["pnl_usd"].mean()),
                        "total_pnl": float(qt["pnl_usd"].sum()),
                    }

        # Drawdown (usando equity curve)
        if equity_curve is not None and not equity_curve.empty:
            m.max_drawdown_pct, m.max_drawdown_usd, m.max_drawdown_duration_days = (
                cls._calculate_drawdown(equity_curve, initial_capital)
            )

        # Sharpe / Sortino / Calmar
        if equity_curve is not None and not equity_curve.empty and len(equity_curve) > 1:
            m.sharpe_ratio_monthly, m.sharpe_ratio_annual = cls._calculate_sharpe(
                equity_curve
            )
            m.sortino_ratio = cls._calculate_sortino(equity_curve)
            m.calmar_ratio = (
                m.total_return_pct / 100 / m.max_drawdown_pct
                if m.max_drawdown_pct > 0
                else float("inf")
            )

        return m

    @staticmethod
    def _calculate_streaks(pnl: np.ndarray) -> tuple[int, int]:
        """Calcula rachas máximas de wins y losses consecutivos."""
        max_wins = max_losses = current_wins = current_losses = 0

        for p in pnl:
            if p > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)

        return max_wins, max_losses

    @staticmethod
    def _calculate_drawdown(
        equity_curve: pd.DataFrame, initial_capital: float
    ) -> tuple[float, float, float]:
        """
        Calcula el drawdown máximo, su valor en USD y su duración.
        Returns: (max_dd_pct, max_dd_usd, max_dd_duration_days)
        """
        capital = equity_curve["capital"].values
        timestamps = equity_curve["timestamp"].values if "timestamp" in equity_curve.columns else None

        peak = capital[0]
        max_dd = 0.0
        max_dd_usd = 0.0
        max_dd_duration = 0.0
        dd_start_ts = None

        for i, c in enumerate(capital):
            if c > peak:
                peak = c
                dd_start_ts = timestamps[i] if timestamps is not None else None

            dd = (peak - c) / peak if peak > 0 else 0
            dd_usd = peak - c

            if dd > max_dd:
                max_dd = dd
                max_dd_usd = dd_usd

                if timestamps is not None and dd_start_ts is not None:
                    try:
                        curr_ts = pd.Timestamp(timestamps[i])
                        start_ts = pd.Timestamp(dd_start_ts)
                        duration = (curr_ts - start_ts).days
                        max_dd_duration = max(max_dd_duration, duration)
                    except Exception:
                        pass

        return max_dd, max_dd_usd, max_dd_duration

    @staticmethod
    def _calculate_sharpe(
        equity_curve: pd.DataFrame, risk_free_rate: float = 0.05
    ) -> tuple[float, float]:
        """
        Calcula Sharpe Ratio mensual y anual.
        Usa los retornos de la equity curve.
        risk_free_rate: tasa libre de riesgo anual (5% por defecto).
        """
        capital = equity_curve["capital"].values
        if len(capital) < 2:
            return 0.0, 0.0

        returns = np.diff(capital) / capital[:-1]
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0, 0.0

        # Asumir retornos por vela. Anualizar asumiendo velas de 4H (6 velas/día, ~252 días/año)
        candles_per_year = 252 * 6  # Para 4H
        rf_per_candle = risk_free_rate / candles_per_year

        avg_return = np.mean(returns)
        std_return = np.std(returns)

        sharpe = (avg_return - rf_per_candle) / std_return * np.sqrt(candles_per_year)
        sharpe_monthly = sharpe / np.sqrt(12)

        return round(sharpe_monthly, 3), round(sharpe, 3)

    @staticmethod
    def _calculate_sortino(equity_curve: pd.DataFrame) -> float:
        """Sortino Ratio — como Sharpe pero solo penaliza la volatilidad negativa."""
        capital = equity_curve["capital"].values
        if len(capital) < 2:
            return 0.0

        returns = np.diff(capital) / capital[:-1]
        downside = returns[returns < 0]

        if len(downside) == 0 or np.std(downside) == 0:
            return float("inf")

        candles_per_year = 252 * 6
        avg_return = np.mean(returns)
        downside_std = np.std(downside)

        return round(avg_return / downside_std * np.sqrt(candles_per_year), 3)

    def summary(self) -> str:
        """Tabla resumen de todas las métricas."""
        lines = [
            f"{'Métrica':<30} {'Valor':>12} {'Objetivo':>12}",
            "-" * 55,
            f"{'Total Trades':<30} {self.total_trades:>12}",
            f"{'Win Rate':<30} {self.win_rate*100:>11.1f}% {'≥45%':>12}",
            f"{'Win Count / Loss Count':<30} {self.win_count:>5}/{self.loss_count:<5}",
            f"{'Avg Win (USD)':<30} {self.avg_win_usd:>12.2f}",
            f"{'Avg Loss (USD)':<30} {self.avg_loss_usd:>12.2f}",
            f"{'Avg R Multiple':<30} {self.avg_r_multiple:>12.2f}",
            f"{'Expectancy (USD/trade)':<30} {self.expectancy:>12.2f}",
            f"{'Profit Factor':<30} {self.profit_factor:>12.2f} {'≥1.5':>12}",
            f"{'Total Return':<30} {self.total_return_pct:>11.1f}%",
            f"{'Total PnL (USD)':<30} {self.total_pnl_usd:>12.2f}",
            "-" * 55,
            f"{'Max Drawdown':<30} {self.max_drawdown_pct*100:>11.1f}% {'≤15%':>12}",
            f"{'Max Drawdown (USD)':<30} {self.max_drawdown_usd:>12.2f}",
            f"{'Max DD Duration (days)':<30} {self.max_drawdown_duration_days:>12.0f}",
            "-" * 55,
            f"{'Sharpe Ratio (Annual)':<30} {self.sharpe_ratio_annual:>12.2f} {'≥1.0':>12}",
            f"{'Sortino Ratio':<30} {self.sortino_ratio:>12.2f}",
            f"{'Calmar Ratio':<30} {self.calmar_ratio:>12.2f}",
            "-" * 55,
            f"{'Max Consecutive Wins':<30} {self.max_consecutive_wins:>12}",
            f"{'Max Consecutive Losses':<30} {self.max_consecutive_losses:>12}",
            f"{'Avg Trade Duration (h)':<30} {self.avg_trade_duration_hours:>12.1f}",
            "-" * 55,
            f"{'SL Hit':<30} {self.sl_hit_count:>12}",
            f"{'TP1 Hit':<30} {self.tp1_hit_count:>12}",
            f"{'TP2 Hit':<30} {self.tp2_hit_count:>12}",
        ]
        return "\n".join(lines)
