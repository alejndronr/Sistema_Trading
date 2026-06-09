"""
dashboard/components/charts.py
Funciones reutilizables de gráficos Plotly para el dashboard.
Optimizado para Atom E3950: sin animaciones, máximo 500 velas en candlestick.
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# ── Paleta de colores ──────────────────────────────────────────────────────────
COLORS = {
    "bg": "#0E1117",
    "text": "#FAFAFA",
    "win": "#00C851",
    "loss": "#FF4444",
    "warn": "#FFBB33",
    "blue": "#2196F3",
    "orange": "#FF8800",
    "gray": "#6C757D",
    "grid": "#1E2130",
}

PHASE_COLORS = {
    "BEAR_DEEP": "#FF1744",
    "BEAR_RECOVERY": "#FF6D00",
    "ACCUMULATION": "#FFD600",
    "BULL_EARLY": "#76FF03",
    "BULL_MATURE": "#00E676",
    "BULL_LATE": "#FFAB00",
    "DISTRIBUTION": "#D50000",
}

_LAYOUT_BASE = dict(
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor=COLORS["bg"],
    font=dict(color=COLORS["text"], family="Inter, sans-serif", size=12),
    xaxis=dict(showgrid=True, gridcolor=COLORS["grid"], zeroline=False),
    yaxis=dict(showgrid=True, gridcolor=COLORS["grid"], zeroline=False),
    margin=dict(l=40, r=20, t=40, b=30),
    hoverlabel=dict(bgcolor="#1E2130", font_color=COLORS["text"]),
)


def _layout(**kwargs) -> dict:
    d = _LAYOUT_BASE.copy()
    d.update(kwargs)
    return d


# ── Equity Curve ───────────────────────────────────────────────────────────────

def equity_curve_chart(df: pd.DataFrame, period: str = "ALL") -> go.Figure:
    """Curva de equity con área de drawdown. df debe tener: entry_time, equity, drawdown, pnl."""
    if df.empty:
        return _empty_fig("Sin datos de equity")

    # Filtrar período
    df = _filter_period(df, "entry_time", period)
    if df.empty:
        return _empty_fig("Sin datos en el período seleccionado")

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.04,
    )

    # Línea de equity
    fig.add_trace(go.Scatter(
        x=df["entry_time"], y=df["equity"],
        mode="lines", name="Capital",
        line=dict(color=COLORS["blue"], width=2),
    ), row=1, col=1)

    # Capital inicial
    initial = df["equity"].iloc[0]
    fig.add_hline(y=initial, line_dash="dash", line_color=COLORS["gray"],
                  annotation_text=f"Inicial ${initial:,.0f}", row=1, col=1)

    # Puntos de trades
    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] <= 0]
    if not wins.empty:
        fig.add_trace(go.Scatter(
            x=wins["entry_time"], y=wins["equity"],
            mode="markers", name="Ganador",
            marker=dict(color=COLORS["win"], size=5, symbol="circle"),
        ), row=1, col=1)
    if not losses.empty:
        fig.add_trace(go.Scatter(
            x=losses["entry_time"], y=losses["equity"],
            mode="markers", name="Perdedor",
            marker=dict(color=COLORS["loss"], size=5, symbol="circle"),
        ), row=1, col=1)

    # Drawdown
    fig.add_trace(go.Scatter(
        x=df["entry_time"], y=df["drawdown"] * 100,
        mode="lines", name="Drawdown %",
        fill="tozeroy",
        line=dict(color=COLORS["loss"], width=1),
        fillcolor="rgba(255,68,68,0.2)",
    ), row=2, col=1)

    fig.update_layout(
        **_layout(height=500, showlegend=True, legend=dict(orientation="h", y=1.05)),
        yaxis=dict(title="Capital USD", tickformat="$,.0f", showgrid=True, gridcolor=COLORS["grid"]),
        yaxis2=dict(title="Drawdown %", tickformat=".1f", showgrid=True, gridcolor=COLORS["grid"]),
    )
    return fig


def pnl_histogram(df: pd.DataFrame) -> go.Figure:
    """Histograma de distribución de PnL por trade."""
    if df.empty or "pnl" not in df.columns:
        return _empty_fig("Sin datos")

    pnl = pd.to_numeric(df["pnl"], errors="coerce").dropna()
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]

    fig = go.Figure()
    if not wins.empty:
        fig.add_trace(go.Histogram(
            x=wins, name="Ganadores", nbinsx=25,
            marker_color=COLORS["win"], opacity=0.8,
        ))
    if not losses.empty:
        fig.add_trace(go.Histogram(
            x=losses, name="Perdedores", nbinsx=25,
            marker_color=COLORS["loss"], opacity=0.8,
        ))

    fig.update_layout(
        **_layout(height=300, barmode="overlay"),
        xaxis_title="PnL (USD)",
        yaxis_title="Frecuencia",
        showlegend=True,
    )
    return fig


def strategy_bar_chart(df: pd.DataFrame) -> go.Figure:
    """Bar chart de Win Rate / Profit Factor / PnL total por estrategia."""
    if df.empty or "strategy" not in df.columns:
        return _empty_fig("Sin datos")

    pnl = pd.to_numeric(df["pnl"], errors="coerce")
    df = df.copy()
    df["pnl"] = pnl
    df["win"] = pnl > 0

    grouped = df.groupby("strategy").agg(
        total_trades=("pnl", "count"),
        wins=("win", "sum"),
        total_pnl=("pnl", "sum"),
        gross_profit=("pnl", lambda x: x[x > 0].sum()),
        gross_loss=("pnl", lambda x: abs(x[x <= 0].sum())),
    ).reset_index()

    grouped["win_rate"] = (grouped["wins"] / grouped["total_trades"] * 100).round(1)
    grouped["profit_factor"] = (grouped["gross_profit"] / grouped["gross_loss"].replace(0, 1e-9)).round(2)

    fig = make_subplots(rows=1, cols=3, subplot_titles=["Win Rate %", "Profit Factor", "PnL USD"])

    for i, (col, fmt) in enumerate([("win_rate", ".1f%"), ("profit_factor", ".2f"), ("total_pnl", "$,.0f")], 1):
        colors = [COLORS["win"] if v >= 0 else COLORS["loss"] for v in grouped[col]]
        fig.add_trace(go.Bar(
            x=grouped["strategy"], y=grouped[col],
            marker_color=colors, showlegend=False,
            text=[f"{v:.2f}" for v in grouped[col]],
            textposition="outside",
        ), row=1, col=i)

    fig.update_layout(**_layout(height=350))
    return fig


def exit_reason_pie(df: pd.DataFrame) -> go.Figure:
    """Pie chart de razones de salida."""
    if df.empty or "exit_reason" not in df.columns:
        return _empty_fig("Sin datos")

    counts = df["exit_reason"].value_counts().reset_index()
    counts.columns = ["reason", "count"]

    color_map = {
        "stop_loss": COLORS["loss"],
        "tp1_partial": "#4CAF50",
        "tp2": COLORS["win"],
        "backtest_end": COLORS["gray"],
        "trailing_stop": COLORS["orange"],
    }
    colors = [color_map.get(r, COLORS["blue"]) for r in counts["reason"]]

    fig = go.Figure(go.Pie(
        labels=counts["reason"], values=counts["count"],
        marker_colors=colors, hole=0.4,
        textinfo="label+percent",
        insidetextorientation="radial",
    ))
    fig.update_layout(**_layout(height=320, showlegend=True))
    return fig


def ml_roc_history(df: pd.DataFrame) -> go.Figure:
    """Gráfico de línea con evolución histórica del ROC-AUC."""
    if df.empty:
        return _empty_fig("Sin datos de retrains")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["retrain_date"], y=df["roc_auc"],
        mode="lines+markers", name="ROC-AUC",
        line=dict(color=COLORS["blue"], width=2),
        marker=dict(size=8),
    ))
    if "cv_f1" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["retrain_date"], y=df["cv_f1"],
            mode="lines+markers", name="CV F1",
            line=dict(color=COLORS["orange"], width=2),
        ))
    fig.add_hline(y=0.65, line_dash="dash", line_color=COLORS["warn"],
                  annotation_text="Objetivo 0.65")
    fig.update_layout(
        **_layout(height=300),
        xaxis_title="Fecha", yaxis_title="Score",
        yaxis_range=[0, 1],
    )
    return fig


def ml_calibration_scatter(df: pd.DataFrame) -> go.Figure:
    """Scatter: ml_proba vs resultado real (calibración del modelo)."""
    if df.empty or "ml_proba" not in df.columns:
        return _empty_fig("Sin datos ML")

    pnl = pd.to_numeric(df["pnl"], errors="coerce")
    proba = pd.to_numeric(df["ml_proba"], errors="coerce")
    mask = proba.notna() & pnl.notna()
    result = (pnl[mask] > 0).astype(int)

    fig = go.Figure()

    # Puntos ganadores vs perdedores
    for val, name, color in [(1, "Ganador", COLORS["win"]), (0, "Perdedor", COLORS["loss"])]:
        subset = proba[mask][result == val]
        if not subset.empty:
            fig.add_trace(go.Scatter(
                x=subset, y=[val + np.random.uniform(-0.05, 0.05) for _ in range(len(subset))],
                mode="markers", name=name,
                marker=dict(color=color, size=6, opacity=0.6),
            ))

    # Línea de calibración perfecta
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                  line=dict(color=COLORS["gray"], dash="dash"))

    fig.update_layout(
        **_layout(height=300),
        xaxis_title="ML Proba", yaxis_title="Resultado Real",
        xaxis_range=[0, 1], yaxis_range=[-0.2, 1.2],
    )
    return fig


def candlestick_chart(df: pd.DataFrame, symbol: str, max_candles: int = 500) -> go.Figure:
    """Gráfico de velas + EMA50/200 + RSI en subplot."""
    if df.empty:
        return _empty_fig(f"Sin datos para {symbol}")

    # Limitar velas para el Atom
    df = df.tail(max_candles).copy()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.04)

    # Velas
    fig.add_trace(go.Candlestick(
        x=df.index if "timestamp" not in df.columns else pd.to_datetime(df["timestamp"], unit="ms"),
        open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name=symbol,
        increasing_line_color=COLORS["win"],
        decreasing_line_color=COLORS["loss"],
    ), row=1, col=1)

    # EMAs
    for period, color in [(50, COLORS["orange"]), (200, COLORS["blue"])]:
        ema = df["close"].ewm(span=period).mean()
        x = df.index if "timestamp" not in df.columns else pd.to_datetime(df["timestamp"], unit="ms")
        fig.add_trace(go.Scatter(
            x=x, y=ema, mode="lines", name=f"EMA{period}",
            line=dict(color=color, width=1.5),
        ), row=1, col=1)

    # RSI
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).ewm(span=14).mean()
    loss = -delta.where(delta < 0, 0).ewm(span=14).mean()
    rs = gain / loss.replace(0, 1e-9)
    rsi = 100 - (100 / (1 + rs))
    x = df.index if "timestamp" not in df.columns else pd.to_datetime(df["timestamp"], unit="ms")

    fig.add_trace(go.Scatter(
        x=x, y=rsi, mode="lines", name="RSI 14",
        line=dict(color=COLORS["warn"], width=1.5),
    ), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color=COLORS["loss"], row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color=COLORS["win"], row=2, col=1)

    fig.update_layout(**_layout(height=500, title=symbol))
    fig.update_xaxes(rangeslider_visible=False)
    return fig


def mini_equity_24h(df: pd.DataFrame) -> go.Figure:
    """Mini curva de equity últimas 24h."""
    if df.empty:
        return _empty_fig("Sin datos últimas 24h")

    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=24)
    col = "entry_time" if "entry_time" in df.columns else df.columns[0]
    recent = df[pd.to_datetime(df[col], utc=True) >= cutoff]
    if recent.empty:
        recent = df.tail(20)

    fig = go.Figure(go.Scatter(
        x=recent[col], y=recent["equity"],
        mode="lines", fill="tonexty",
        line=dict(color=COLORS["blue"], width=2),
        fillcolor="rgba(33,150,243,0.15)",
    ))
    fig.update_layout(
        **_layout(height=150),
        showlegend=False,
        margin=dict(l=10, r=10, t=5, b=5),
        xaxis=dict(showticklabels=False),
    )
    return fig


# ── Utilidades ─────────────────────────────────────────────────────────────────

def _empty_fig(msg: str = "Sin datos") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        **_layout(height=200),
        annotations=[dict(text=msg, x=0.5, y=0.5, showarrow=False,
                          font=dict(size=14, color=COLORS["gray"]))],
    )
    return fig


def _filter_period(df: pd.DataFrame, col: str, period: str) -> pd.DataFrame:
    periods = {"1S": 7, "1M": 30, "3M": 90, "6M": 180, "1A": 365}
    if period not in periods or col not in df.columns:
        return df
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=periods[period])
    return df[pd.to_datetime(df[col], utc=True) >= cutoff]
