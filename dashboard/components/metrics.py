"""
dashboard/components/metrics.py
Cálculo de métricas de trading a partir de DataFrames de trades.
"""
import numpy as np
import pandas as pd
from typing import Tuple


def compute_metrics(df: pd.DataFrame) -> dict:
    """
    Calcula todas las métricas relevantes desde un DataFrame de trades.
    Maneja tanto nomenclatura antigua como nueva.
    """
    if df.empty:
        return _empty_metrics()

    pnl = pd.to_numeric(df.get("pnl", pd.Series([], dtype=float)), errors="coerce").dropna()
    if pnl.empty:
        return _empty_metrics()

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    n = len(pnl)

    win_rate = len(wins) / n if n else 0.0
    avg_win = wins.mean() if not wins.empty else 0.0
    avg_loss = abs(losses.mean()) if not losses.empty else 0.0

    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    # Sharpe (anual, asumiendo trade = unidad de tiempo ~4h)
    returns = pnl / 1000.0  # Normalizar por capital base
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0.0

    # Sortino
    downside = returns[returns < 0]
    sortino = (returns.mean() / downside.std() * np.sqrt(252)) if not downside.empty and downside.std() > 0 else 0.0

    # Max Drawdown
    cumulative = pnl.cumsum()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max)
    max_dd_usd = abs(drawdown.min()) if not drawdown.empty else 0.0
    max_dd_pct = abs((drawdown / (running_max + 1e-9)).min()) if not drawdown.empty else 0.0

    # Calmar
    total_return = pnl.sum()
    calmar = (total_return / max_dd_usd) if max_dd_usd > 0 else 0.0

    # Rachas
    binary = (pnl > 0).astype(int).values
    max_wins, max_losses = _streak_analysis(binary)

    # R multiple
    r_col = df.get("r_multiple", None)
    avg_r = pd.to_numeric(r_col, errors="coerce").mean() if r_col is not None else 0.0

    # Duration
    dur_col = df.get("duration_hours", None)
    avg_dur = pd.to_numeric(dur_col, errors="coerce").mean() if dur_col is not None else 0.0

    return {
        "total_trades": n,
        "win_count": len(wins),
        "loss_count": len(losses),
        "win_rate": win_rate,
        "avg_win_usd": avg_win,
        "avg_loss_usd": avg_loss,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "total_pnl_usd": pnl.sum(),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "max_dd_usd": max_dd_usd,
        "max_dd_pct": max_dd_pct,
        "max_consecutive_wins": max_wins,
        "max_consecutive_losses": max_losses,
        "avg_r": avg_r if not np.isnan(avg_r) else 0.0,
        "avg_duration_h": avg_dur if not np.isnan(avg_dur) else 0.0,
    }


def compute_metrics_30d(df: pd.DataFrame) -> dict:
    """Métricas del último mes."""
    if df.empty:
        return _empty_metrics()
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=30)
    col = "entry_time" if "entry_time" in df.columns else df.columns[0]
    recent = df[pd.to_datetime(df[col], utc=True) >= cutoff]
    return compute_metrics(recent)


def _streak_analysis(binary: np.ndarray) -> Tuple[int, int]:
    """Calcula rachas máximas de victorias y pérdidas."""
    if len(binary) == 0:
        return 0, 0
    max_w = max_l = cur_w = cur_l = 0
    for b in binary:
        if b == 1:
            cur_w += 1
            cur_l = 0
            max_w = max(max_w, cur_w)
        else:
            cur_l += 1
            cur_w = 0
            max_l = max(max_l, cur_l)
    return max_w, max_l


def _empty_metrics() -> dict:
    return {
        "total_trades": 0, "win_count": 0, "loss_count": 0,
        "win_rate": 0.0, "avg_win_usd": 0.0, "avg_loss_usd": 0.0,
        "profit_factor": 0.0, "expectancy": 0.0, "total_pnl_usd": 0.0,
        "gross_profit": 0.0, "gross_loss": 0.0,
        "sharpe_ratio": 0.0, "sortino_ratio": 0.0, "calmar_ratio": 0.0,
        "max_dd_usd": 0.0, "max_dd_pct": 0.0,
        "max_consecutive_wins": 0, "max_consecutive_losses": 0,
        "avg_r": 0.0, "avg_duration_h": 0.0,
    }
