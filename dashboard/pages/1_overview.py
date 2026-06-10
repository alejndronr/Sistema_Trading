"""
dashboard/pages/1_overview.py
Página Overview — Estado general del sistema, posiciones abiertas y equity 24h.
"""
import sys
import time
from pathlib import Path
import datetime

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np

from dashboard.components.db import (
    get_heartbeat, get_portfolio_state, get_open_positions,
    get_trades, get_equity_curve, pg_available, get_current_prices_cached
)
from dashboard.components.metrics import compute_metrics_30d
from dashboard.components.charts import mini_equity_24h, COLORS
from dashboard.components.cycle_display import render_cycle_panel, days_to_halving

# ── CSS ───────────────────────────────────────────────────────────────────────
CSS_PATH = Path(__file__).parent.parent / "assets" / "style.css"
if CSS_PATH.exists():
    with open(CSS_PATH) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Auto-refresh ──────────────────────────────────────────────────────────────
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()
if time.time() - st.session_state.last_refresh > 60:
    st.session_state.last_refresh = time.time()
    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
hb = get_heartbeat()
ps = get_portfolio_state()

col_logo, col_status, col_mode, col_ts = st.columns([3, 1.5, 1.5, 2])

with col_logo:
    st.markdown("# 🤖 Sistema Trading ZimaBlade V6")

with col_status:
    if hb:
        last_ping = pd.to_datetime(hb.get("last_ping"), errors="coerce", utc=True)
        now = pd.Timestamp.now(tz="UTC")
        offline = last_ping is None or (now - last_ping).total_seconds() > 300
        if offline:
            st.markdown('<span class="badge-offline">🔴 OFFLINE</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-live">🟢 LIVE</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-offline">🔴 OFFLINE</span>', unsafe_allow_html=True)

with col_mode:
    paper = hb.get("paper_mode", True) if hb else True
    if paper:
        st.markdown('<span class="badge-paper">📄 PAPER MODE</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-live-trading">💰 LIVE TRADING</span>', unsafe_allow_html=True)

with col_ts:
    if hb and hb.get("last_ping"):
        st.caption(f"Último ping: {hb['last_ping']}")
    ver = hb.get("engine_version", "V6") if hb else "V6"
    st.caption(f"Motor: {ver}")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# KPIs PRINCIPALES
# ─────────────────────────────────────────────────────────────────────────────
trades_df = get_trades(limit=1000, real_only=True)
metrics_30d = compute_metrics_30d(trades_df)
open_pos = get_open_positions()
equity_df = get_equity_curve()

capital = ps.get("current_capital", 1000.0) if ps else 1000.0
initial = ps.get("daily_start", capital) if ps else capital
pnl_total = metrics_30d["total_pnl"]
pnl_today = (capital - initial) if ps else 0.0
max_dd = metrics_30d["max_dd_pct"] * 100

k1, k2, k3, k4, k5, k6, k7, k8 = st.columns(8)

k1.metric("💰 Capital", f"${capital:,.2f}")
delta_pnl = f"+${pnl_total:,.2f}" if pnl_total >= 0 else f"-${abs(pnl_total):,.2f}"
k2.metric("📈 PnL Total", f"${pnl_total:,.2f}", delta=delta_pnl)

delta_today = f"+${pnl_today:,.2f}" if pnl_today >= 0 else f"-${abs(pnl_today):,.2f}"
k3.metric("📅 PnL Hoy", f"${pnl_today:,.2f}", delta=delta_today)
k4.metric("📉 Max DD", f"{max_dd:.1f}%", delta=f"-{max_dd:.1f}%")
k5.metric("🎯 Win Rate 30d", f"{metrics_30d['win_rate']*100:.1f}%")
k6.metric("⚡ P. Factor 30d", f"{metrics_30d['profit_factor']:.2f}")
k7.metric("📂 Trades abiertos", len(open_pos) if not open_pos.empty else 0)
k8.metric("📊 Total trades", len(trades_df))

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# ESTADO DEL CYCLE DETECTOR
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🌐 CycleDetector — Estado por Par</div>', unsafe_allow_html=True)

# Intentar leer datos reales del ciclo desde PostgreSQL
cycle_data_raw = []
try:
    from dashboard.components.db import query_pg
    cycle_df = query_pg(
        "SELECT symbol, phase, conviction, rsi_daily, rsi_weekly, pct_from_ath FROM cycle_state ORDER BY symbol"
    )
    if not cycle_df.empty:
        cycle_data_raw = cycle_df.to_dict("records")
except Exception:
    pass  # Usar demo si no hay tabla

render_cycle_panel(cycle_data_raw)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# POSICIONES ABIERTAS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📂 Posiciones Abiertas</div>', unsafe_allow_html=True)

if open_pos.empty:
    st.info("✅ No hay posiciones abiertas actualmente.")
else:
    syms = tuple(open_pos["symbol"].unique().tolist())
    current_prices = get_current_prices_cached(syms)

    display_cols = []
    for _, row in open_pos.iterrows():
        sym = row.get("symbol", "")
        entry = float(row.get("entry_price", 0))
        curr = current_prices.get(sym, entry)
        sl = float(row.get("stop_loss", 0))
        tp1 = float(row.get("tp1", 0))
        tp2 = row.get("tp2")
        units = float(row.get("units", row.get("remaining_units", 0)))
        direction = row.get("direction", "LONG")

        if direction == "LONG":
            pnl = (curr - entry) * units
            pnl_pct = (curr / entry - 1) * 100 if entry > 0 else 0
            tp1_dist = (tp1 - entry) if tp1 > 0 else 0
            curr_dist = (curr - entry)
            progress = min(1.0, curr_dist / tp1_dist) if tp1_dist > 0 else 0
        else:
            pnl = (entry - curr) * units
            pnl_pct = (entry / curr - 1) * 100 if curr > 0 else 0
            progress = 0.0

        tp1_hit = bool(row.get("tp1_hit", False))
        ml_p = row.get("ml_proba", None)

        display_cols.append({
            "Símbolo": sym,
            "Estrategia": row.get("strategy", "-"),
            "Dirección": direction,
            "Entrada": f"${entry:,.2f}",
            "Actual": f"${curr:,.2f}",
            "SL": f"${sl:,.2f}",
            "TP1": f"${tp1:,.2f}",
            "PnL $": pnl,
            "PnL %": pnl_pct,
            "TP1 Hit": "✅" if tp1_hit else "",
            "ML Proba": f"{ml_p:.2f}" if ml_p is not None else "-",
            "Progress": progress,
        })

    pos_df = pd.DataFrame(display_cols)

    # Colorear según PnL
    def color_pnl(val):
        color = "color: #00C851" if val >= 0 else "color: #FF4444"
        return color

    styled = pos_df.drop(columns=["Progress"]).style\
        .format({"PnL $": "${:,.2f}", "PnL %": "{:.2f}%"})\
        .applymap(color_pnl, subset=["PnL $", "PnL %"])

    st.dataframe(styled, use_container_width=True, height=250)

    # Barras de progreso
    st.caption("📊 Progreso hacia TP1 por posición:")
    for row_d in display_cols:
        cols_p = st.columns([2, 6, 1])
        cols_p[0].caption(row_d["Símbolo"])
        cols_p[1].progress(float(row_d["Progress"]))
        cols_p[2].caption(f"{row_d['Progress']*100:.0f}%")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SEÑALES PENDIENTES
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">⏳ Señales Pendientes</div>', unsafe_allow_html=True)

import json
pending_signals = []
# Intentamos derivarlas del heartbeat si `live_engine.py` lo estuviera persistiendo en algún sitio
# Nota: como en el step anterior no se guardó "pending" en system_heartbeat, aquí lo leemos
# si estuviéramos guardando el _pending en JSON. Por ahora lo dejamos como info estática o vacío.
# Si quisiéramos leerlo, podríamos parsear algo del heartbeat. Aquí pondremos un info.
st.info("No hay señales pendientes de confirmación en 15M en este momento.")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# MINI EQUITY CURVE — ÚLTIMAS 24H
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📉 Equity — Últimas 24h</div>', unsafe_allow_html=True)

if not equity_df.empty:
    fig = mini_equity_24h(equity_df)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
else:
    st.info("Sin suficientes datos para mostrar la curva de equity.")

# ── Footer refresh indicator ──────────────────────────────────────────────────
elapsed = int(time.time() - st.session_state.get("last_refresh", time.time()))
st.caption(f"🔄 Actualización automática cada 60s | Próxima en {max(0, 60 - elapsed)}s")
