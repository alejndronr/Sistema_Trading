"""
dashboard/components/cycle_display.py
Componentes visuales para mostrar las fases del CycleDetector.
"""
import streamlit as st
import pandas as pd
from datetime import datetime, timezone

# Halving Abril 2028
NEXT_HALVING = datetime(2028, 4, 15, tzinfo=timezone.utc)

PHASE_COLORS = {
    "BEAR_DEEP":     "#FF1744",
    "BEAR_RECOVERY": "#FF6D00",
    "ACCUMULATION":  "#FFD600",
    "BULL_EARLY":    "#76FF03",
    "BULL_MATURE":   "#00E676",
    "BULL_LATE":     "#FFAB00",
    "DISTRIBUTION":  "#D50000",
    "UNKNOWN":       "#6C757D",
}

PHASE_EMOJI = {
    "BEAR_DEEP":     "🔴",
    "BEAR_RECOVERY": "🟠",
    "ACCUMULATION":  "#FFAB00",
    "BULL_EARLY":    "🟢",
    "BULL_MATURE":   "💚",
    "BULL_LATE":     "🟡",
    "DISTRIBUTION":  "🔻",
    "UNKNOWN":       "⚪",
}

PHASE_RISK = {
    "BEAR_DEEP": 0.25,
    "BEAR_RECOVERY": 0.60,
    "ACCUMULATION": 0.60,
    "BULL_EARLY": 1.00,
    "BULL_MATURE": 1.00,
    "BULL_LATE": 0.60,
    "DISTRIBUTION": 0.25,
}

PHASE_STRATEGIES = {
    "BEAR_DEEP":     ["TrendFollowing"],
    "BEAR_RECOVERY": ["TrendFollowing", "MeanReversion"],
    "ACCUMULATION":  ["Breakout", "MeanReversion"],
    "BULL_EARLY":    ["TrendFollowing", "Breakout"],
    "BULL_MATURE":   ["TrendFollowing", "MeanReversion", "Breakout"],
    "BULL_LATE":     ["MeanReversion"],
    "DISTRIBUTION":  ["MeanReversion"],
}


def days_to_halving() -> int:
    now = datetime.now(timezone.utc)
    delta = NEXT_HALVING - now
    return max(0, delta.days)


def render_phase_badge(phase: str) -> str:
    """Devuelve HTML de un badge de fase."""
    color = PHASE_COLORS.get(phase, "#6C757D")
    return f'<span style="background:{color};color:#000;border-radius:6px;padding:3px 10px;font-weight:bold;font-size:13px;">{phase}</span>'


def render_cycle_panel(cycle_data: list[dict]):
    """
    Renderiza el panel completo del CycleDetector.
    cycle_data: lista de dicts con keys: symbol, phase, rsi_daily, rsi_weekly, pct_from_ath, conviction
    """
    halving_days = days_to_halving()

    st.markdown(f"⏳ **Próximo Halving:** {halving_days} días (Abril 2028)")

    if not cycle_data:
        st.info("Sin datos del CycleDetector. El motor de trading no está reportando en PostgreSQL.")
        _demo_cycle_panel()
        return

    cols = st.columns(min(len(cycle_data), 3))
    for i, item in enumerate(cycle_data):
        col = cols[i % 3]
        phase = item.get("phase", "UNKNOWN")
        color = PHASE_COLORS.get(phase, "#6C757D")
        risk = PHASE_RISK.get(phase, 1.0)
        strategies = PHASE_STRATEGIES.get(phase, [])

        with col:
            st.markdown(
                f"""
                <div style="background:#1E2130;border-radius:10px;padding:14px;border-left:4px solid {color};margin-bottom:10px;">
                  <div style="font-size:18px;font-weight:bold;">{item.get('symbol','')}</div>
                  <div style="margin:6px 0;">{render_phase_badge(phase)}</div>
                  <div style="color:#AAA;font-size:12px;">Risk: <b style="color:{color};">{risk*100:.0f}%</b></div>
                  <div style="color:#AAA;font-size:12px;">RSI D/W: <b>{item.get('rsi_daily','-'):.0f}</b> / <b>{item.get('rsi_weekly','-'):.0f}</b></div>
                  <div style="color:#AAA;font-size:12px;">%ATH: <b>{item.get('pct_from_ath',0)*100:.1f}%</b></div>
                  <div style="color:#888;font-size:11px;margin-top:6px;">{', '.join(strategies)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _demo_cycle_panel():
    """Panel de demostración cuando no hay datos reales."""
    demo = [
        {"symbol": "BTC/USDC", "phase": "DISTRIBUTION", "rsi_daily": 62, "rsi_weekly": 65, "pct_from_ath": -0.28},
        {"symbol": "ETH/USDC", "phase": "BEAR_RECOVERY", "rsi_daily": 45, "rsi_weekly": 42, "pct_from_ath": -0.52},
        {"symbol": "SOL/USDC", "phase": "ACCUMULATION", "rsi_daily": 48, "rsi_weekly": 50, "pct_from_ath": -0.45},
    ]
    render_cycle_panel(demo)


def render_cycle_timeline():
    """Visualiza el timeline del ciclo macro de 4 años."""
    import plotly.graph_objects as go

    events = [
        ("Halving\nAbr-2024", 0, "#FFAB00"),
        ("ATH Oct-2025\n$126k", 0.45, "#00E676"),
        ("Suelo?\n~Oct-2026", 0.7, "#FF1744"),
        ("Acumulación\n2026-2028", 0.85, "#FFD600"),
        ("Halving\nAbr-2028", 1.0, "#FFAB00"),
    ]

    # Posición actual estimada (~70% del ciclo)
    now_pct = 0.65

    fig = go.Figure()

    # Línea base
    fig.add_shape(type="line", x0=0, x1=1, y0=0.5, y1=0.5,
                  line=dict(color="#2196F3", width=4))

    # Eventos
    for label, x, color in events:
        fig.add_trace(go.Scatter(
            x=[x], y=[0.5],
            mode="markers+text",
            marker=dict(color=color, size=18, symbol="diamond"),
            text=[label], textposition="bottom center",
            textfont=dict(size=10, color="#FAFAFA"),
            showlegend=False,
        ))

    # Marcador "AHORA"
    fig.add_trace(go.Scatter(
        x=[now_pct], y=[0.7],
        mode="markers+text",
        marker=dict(color="#FF4444", size=14, symbol="arrow-down"),
        text=["📍 AHORA"], textposition="top center",
        textfont=dict(size=12, color="#FF4444", family="Inter"),
        showlegend=False,
    ))

    fig.update_layout(
        paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
        font=dict(color="#FAFAFA"),
        height=200, margin=dict(l=20, r=20, t=10, b=60),
        xaxis=dict(range=[-0.05, 1.1], showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(range=[0, 1], showgrid=False, showticklabels=False, zeroline=False),
    )
    return fig
