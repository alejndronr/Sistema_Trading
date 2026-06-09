"""
dashboard/pages/5_cycle.py
Análisis de Ciclo Macro: timeline, velas diarias por par, indicadores de ciclo.
"""
import sys
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np

from dashboard.components.db import query_sqlite, query_pg
from dashboard.components.charts import candlestick_chart, COLORS, PHASE_COLORS, _layout, _empty_fig
from dashboard.components.cycle_display import (
    render_cycle_timeline, days_to_halving, PHASE_COLORS as PCOL, PHASE_RISK, PHASE_STRATEGIES
)

# ── CSS ───────────────────────────────────────────────────────────────────────
CSS_PATH = Path(__file__).parent.parent / "assets" / "style.css"
if CSS_PATH.exists():
    with open(CSS_PATH) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("# 🌐 Análisis de Ciclo Macro")
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# TIMELINE DEL CICLO ACTUAL
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📅 Timeline del Ciclo 2024-2028</div>', unsafe_allow_html=True)

fig_timeline = render_cycle_timeline()
st.plotly_chart(fig_timeline, use_container_width=True, config={"displayModeBar": False})

halving_days = days_to_halving()
col_hv1, col_hv2, col_hv3 = st.columns(3)
col_hv1.metric("⏳ Días al Halving", f"{halving_days} días")
col_hv2.metric("📅 Fecha Halving", "Abril 2028")
col_hv3.metric("🎯 ATH estimado BTC", "~$126k (Oct 2025)")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# INDICADORES DE CICLO POR PAR
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Indicadores de Ciclo por Par</div>', unsafe_allow_html=True)

# Intentar datos reales de la BD
cycle_data = []
try:
    from dashboard.components.db import get_cycle_states
    cycle_df = get_cycle_states()
    if not cycle_df.empty:
        cycle_data = cycle_df.to_dict("records")
except Exception:
    pass

if not cycle_data:
    # Datos de ejemplo para cuando el motor no está corriendo
    cycle_data = [
        {"symbol": "BTC/USDC", "phase": "DISTRIBUTION", "conviction": 72, "rsi_daily": 62, "rsi_weekly": 65, "pct_from_ath": -0.28, "ema200_dist": 1.45},
        {"symbol": "ETH/USDC", "phase": "BEAR_RECOVERY", "conviction": 65, "rsi_daily": 45, "rsi_weekly": 42, "pct_from_ath": -0.52, "ema200_dist": -0.12},
        {"symbol": "DOT/USDC", "phase": "ACCUMULATION", "conviction": 58, "rsi_daily": 48, "rsi_weekly": 50, "pct_from_ath": -0.71, "ema200_dist": -0.08},
        {"symbol": "AVAX/USDC", "phase": "ACCUMULATION", "conviction": 60, "rsi_daily": 52, "rsi_weekly": 48, "pct_from_ath": -0.60, "ema200_dist": 0.02},
        {"symbol": "ADA/USDC", "phase": "BEAR_RECOVERY", "conviction": 55, "rsi_daily": 44, "rsi_weekly": 46, "pct_from_ath": -0.68, "ema200_dist": -0.05},
        {"symbol": "NEAR/USDC", "phase": "ACCUMULATION", "conviction": 50, "rsi_daily": 50, "rsi_weekly": 49, "pct_from_ath": -0.75, "ema200_dist": 0.01},
    ]
    st.caption("ℹ️ Mostrando datos de demostración (motor offline)")

# Tabla resumen
tbl_rows = []
for item in cycle_data:
    phase = item.get("phase", "UNKNOWN")
    tbl_rows.append({
        "Par": item.get("symbol", ""),
        "Fase": phase,
        "Convicción": f"{item.get('conviction', 0):.0f}%",
        "RSI D": f"{item.get('rsi_daily', 0):.0f}",
        "RSI W": f"{item.get('rsi_weekly', 0):.0f}",
        "% ATH": f"{item.get('pct_from_ath', 0)*100:.1f}%",
        "vs EMA200": f"{item.get('ema200_dist', 0)*100:.1f}%",
        "Risk Mult": f"{PHASE_RISK.get(phase, 1.0)*100:.0f}%",
        "Estrategias": ", ".join(PHASE_STRATEGIES.get(phase, ["-"])),
    })

tbl_df = pd.DataFrame(tbl_rows)

def phase_color_style(val):
    color = PCOL.get(str(val).upper(), "#6C757D")
    return f"background-color: {color}20; color: {color}; font-weight: bold"

if not tbl_df.empty:
    styled_tbl = tbl_df.style.applymap(phase_color_style, subset=["Fase"])
    st.dataframe(styled_tbl, use_container_width=True, hide_index=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICOS DE VELAS DIARIAS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🕯️ Velas Diarias por Par (90 días)</div>', unsafe_allow_html=True)

MAIN_SYMBOLS = ["BTC/USDC", "ETH/USDC", "DOT/USDC"]
sel_sym = st.selectbox("Seleccionar par:", [s.get("symbol") for s in cycle_data], index=0)

ohlcv_df = query_sqlite(
    "SELECT timestamp, open, high, low, close, volume FROM ohlcv WHERE symbol = ? AND timeframe = '1d' ORDER BY timestamp DESC LIMIT 90",
    params=(sel_sym,)
)

if not ohlcv_df.empty:
    ohlcv_df = ohlcv_df.sort_values("timestamp")
    fig_candle = candlestick_chart(ohlcv_df, sel_sym, max_candles=90)
    st.plotly_chart(fig_candle, use_container_width=True, config={"displayModeBar": True})
else:
    st.info(f"Sin datos OHLCV diarios para {sel_sym}. Ejecuta `fetch_data.py` para descargar datos históricos.")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# PROYECCIÓN DEL CICLO (informativa)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🔮 Proyección del Ciclo (Referencia Histórica)</div>', unsafe_allow_html=True)

st.info("""
**⚠️ Disclaimer:** Esta proyección es puramente informativa basada en patrones históricos. 
No constituye predicción ni consejo de inversión.
""")

col_prj1, col_prj2 = st.columns(2)

with col_prj1:
    st.markdown("##### Duración histórica por fase (ciclos anteriores)")
    hist_phases = {
        "BULL_EARLY":    ("~4 meses", "Post-halving, primer rally"),
        "BULL_MATURE":   ("~6 meses", "Tendencia alcista sostenida"),
        "BULL_LATE":     ("~3 meses", "Euforia y correcciones"),
        "DISTRIBUTION":  ("~3-6 meses", "Distribución institucional"),
        "BEAR_DEEP":     ("~6-12 meses", "Capitulación y suelo"),
        "BEAR_RECOVERY": ("~3-6 meses", "Recuperación inicial"),
        "ACCUMULATION":  ("~6-12 meses", "Smart money acumula"),
    }
    for phase, (duration, desc) in hist_phases.items():
        color = PCOL.get(phase, "#6C757D")
        st.markdown(
            f'<div style="display:flex;align-items:center;margin:4px 0;">'
            f'<span style="width:12px;height:12px;background:{color};border-radius:50%;display:inline-block;margin-right:8px;"></span>'
            f'<b style="color:{color};width:140px;">{phase}</b> '
            f'<span style="color:#AAA;">{duration} — {desc}</span></div>',
            unsafe_allow_html=True
        )

with col_prj2:
    st.markdown("##### Eventos clave del ciclo actual")
    events = [
        ("✅", "Halving Bitcoin", "Abril 2024", "Completado"),
        ("✅", "ATH ~$126k", "Octubre 2025", "Completado"),
        ("⏳", "Suelo del Bear", "Est. Oct-Dec 2026", "Pendiente"),
        ("⏳", "Acumulación", "2026-2028", "Pendiente"),
        ("⏳", "Próximo Halving", "Abril 2028", f"En {halving_days} días"),
    ]
    for icon, event, date, status in events:
        color = "#00C851" if icon == "✅" else "#FFBB33"
        st.markdown(
            f'{icon} **{event}** — {date} <span style="color:{color};font-size:12px;">({status})</span>',
            unsafe_allow_html=True
        )
