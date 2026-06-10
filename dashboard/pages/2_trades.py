"""
dashboard/pages/2_trades.py
Análisis completo de trades: filtros, métricas, tabla, distribuciones y breakdowns.
"""
import sys
from pathlib import Path
import time

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np

from dashboard.components.db import get_trades
from dashboard.components.metrics import compute_metrics
from dashboard.components.charts import pnl_histogram, strategy_bar_chart, exit_reason_pie

# ── CSS ───────────────────────────────────────────────────────────────────────
CSS_PATH = Path(__file__).parent.parent / "assets" / "style.css"
if CSS_PATH.exists():
    with open(CSS_PATH) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("# 📈 Análisis de Trades")
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# FILTROS EN SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔧 Filtros")

    date_range = st.date_input(
        "Rango de fechas",
        value=(
            (pd.Timestamp.now() - pd.Timedelta(days=90)).date(),
            pd.Timestamp.now().date(),
        ),
        key="trade_date_range",
    )

    from dashboard.components.db import pg_available
    
    df_all = get_trades(limit=5000, real_only=False)
    
    if not pg_available():
        np.random.seed(42)
        mock_dates = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=150, freq="12H")
        df_all = pd.DataFrame({
            "entry_time": mock_dates,
            "symbol": np.random.choice(["BTC/USDC", "ETH/USDC", "SOL/USDC"], 150),
            "strategy": np.random.choice(["TrendFollowing", "MeanReversion", "Breakout"], 150),
            "regime": np.random.choice(["BULL_MATURE", "BULL_EARLY", "BEAR_DEEP", "ACCUMULATION"], 150),
            "setup_quality": np.random.choice(["A+", "A", "B", "C"], 150),
            "direction": np.random.choice(["long", "short"], 150),
            "is_backtest": np.random.choice([True, False], 150),
            "pnl": np.random.randn(150) * 12,
            "pnl_usd": np.random.randn(150) * 12,
            "r_multiple": np.random.randn(150) * 1.2,
            "ml_proba": np.random.uniform(0.4, 0.85, 150),
            "exit_reason": np.random.choice(["tp1_partial", "stop_loss", "tp2", "trailing_stop"], 150),
            "duration_hours": np.random.uniform(2, 72, 150),
            "entry_price": np.random.uniform(2000, 65000, 150),
            "stop_loss": np.random.uniform(1900, 64000, 150)
        })

    symbols_available = ["Todos"] + sorted(df_all["symbol"].dropna().unique().tolist()) if not df_all.empty else ["Todos"]
    strategies_available = ["Todos"] + sorted(df_all["strategy"].dropna().unique().tolist()) if not df_all.empty else ["Todos"]
    regimes_available = ["Todos"] + sorted(df_all["regime"].dropna().unique().tolist()) if not df_all.empty and "regime" in df_all.columns else ["Todos"]
    qualities_available = ["Todos", "A+", "A", "B", "C"]

    sel_symbols = st.multiselect("Par", symbols_available, default=["Todos"])
    sel_strategies = st.multiselect("Estrategia", strategies_available, default=["Todos"])
    sel_regimes = st.multiselect("Fase ciclo", regimes_available, default=["Todos"])
    sel_quality = st.multiselect("Calidad setup", qualities_available, default=["Todos"])
    
    directions_available = ["Todos", "long", "short"]
    sel_direction = st.multiselect("Dirección", directions_available, default=["Todos"])
    data_source = st.radio("Datos", ["Solo reales", "Solo backtest", "Todos"], index=0)

# ─────────────────────────────────────────────────────────────────────────────
# APLICAR FILTROS
# ─────────────────────────────────────────────────────────────────────────────
df = df_all.copy() if not df_all.empty else pd.DataFrame()

if df.empty:
    st.warning("⚠️ Sin datos de trades para los filtros seleccionados.")
    st.stop()

# Fecha
if len(date_range) == 2 and "entry_time" in df.columns:
    start, end = date_range
    entry_col = pd.to_datetime(df["entry_time"], utc=True)
    df = df[
        (entry_col.dt.date >= start) &
        (entry_col.dt.date <= end)
    ]

# Símbolo
if "Todos" not in sel_symbols and sel_symbols:
    df = df[df["symbol"].isin(sel_symbols)]

# Estrategia
if "Todos" not in sel_strategies and sel_strategies:
    df = df[df["strategy"].isin(sel_strategies)]

# Régimen
if "Todos" not in sel_regimes and sel_regimes and "regime" in df.columns:
    df = df[df["regime"].isin(sel_regimes)]

# Calidad
if "Todos" not in sel_quality and sel_quality and "setup_quality" in df.columns:
    df = df[df["setup_quality"].isin(sel_quality)]

# Dirección
if "Todos" not in sel_direction and sel_direction and "direction" in df.columns:
    df = df[df["direction"].isin(sel_direction)]

# Fuente de datos
if data_source == "Solo reales" and "is_backtest" in df_all.columns:
    df = df[df_all["is_backtest"] != True]
elif data_source == "Solo backtest" and "is_backtest" in df_all.columns:
    df = df[df_all["is_backtest"] == True]

# ─────────────────────────────────────────────────────────────────────────────
# KPIs DEL PERÍODO FILTRADO
# ─────────────────────────────────────────────────────────────────────────────
m = compute_metrics(df)

k1, k2, k3, k4, k5, k6, k7, k8 = st.columns(8)
k1.metric("🎯 Win Rate",    f"{m['win_rate']*100:.1f}%")
k2.metric("⚡ P. Factor",   f"{m['profit_factor']:.2f}")
k3.metric("💡 Expectancy",  f"${m['expectancy']:.2f}")
k4.metric("📊 Sharpe",      f"{m['sharpe_ratio']:.2f}")
k5.metric("📉 Max DD",      f"{m['max_dd_pct']*100:.1f}%")
k6.metric("📈 Avg R",       f"{m['avg_r']:.2f}R")
k7.metric("⏱️ Avg Duración", f"{m['avg_duration_h']:.1f}h")
k8.metric("📋 Trades",      m["total_trades"])

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# TABLA DE TRADES
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📋 Trades del Período</div>', unsafe_allow_html=True)

EXIT_COLORS = {
    "stop_loss":    "background-color: rgba(255,68,68,0.2)",
    "tp1_partial":  "background-color: rgba(76,175,80,0.2)",
    "tp2":          "background-color: rgba(0,200,81,0.3)",
    "backtest_end": "background-color: rgba(108,117,125,0.2)",
    "trailing_stop":"background-color: rgba(255,136,0,0.2)",
}

display_df = df[[
    c for c in [
        "entry_time", "symbol", "strategy", "regime", "direction",
        "setup_quality", "pnl", "r_multiple", "ml_proba",
        "exit_reason", "duration_hours", "entry_price", "stop_loss",
    ] if c in df.columns
]].copy()

if not display_df.empty:
    # Formatear fechas
    if "entry_time" in display_df.columns:
        display_df["entry_time"] = pd.to_datetime(display_df["entry_time"], utc=True).dt.strftime("%Y-%m-%d %H:%M")

    # Colorear PnL
    def style_row(row):
        styles = [""] * len(row)
        if "pnl" in row.index:
            if row["pnl"] > 0:
                styles[row.index.get_loc("pnl")] = "color: #00C851; font-weight: bold"
            elif row["pnl"] < 0:
                styles[row.index.get_loc("pnl")] = "color: #FF4444; font-weight: bold"
        return styles

    # Mostrar primeros 100 filas + paginación simple
    page = st.session_state.get("trades_page", 0)
    chunk = 100
    total_pages = max(1, (len(display_df) + chunk - 1) // chunk)
    page_df = display_df.iloc[page * chunk: (page + 1) * chunk]

    styled_df = page_df.style.apply(style_row, axis=1)\
        .format({"pnl": "${:,.2f}", "r_multiple": "{:.2f}R", "ml_proba": "{:.2f}", "duration_hours": "{:.1f}h"},
                na_rep="-")

    st.dataframe(styled_df, use_container_width=True, height=450)

    # Paginación
    col_p1, col_p2, col_p3 = st.columns([1, 3, 1])
    if col_p1.button("◀ Anterior") and page > 0:
        st.session_state.trades_page = page - 1
        st.rerun()
    col_p2.caption(f"Página {page+1} de {total_pages} | {len(display_df)} trades totales")
    if col_p3.button("Siguiente ▶") and page < total_pages - 1:
        st.session_state.trades_page = page + 1
        st.rerun()

    # Exportar a CSV
    csv = display_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Exportar CSV",
        data=csv,
        file_name='trades_export.csv',
        mime='text/csv',
    )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICOS DE DISTRIBUCIÓN
# ─────────────────────────────────────────────────────────────────────────────
col_dist, col_exit = st.columns(2)

with col_dist:
    st.markdown('<div class="section-header">📊 Distribución PnL</div>', unsafe_allow_html=True)
    st.plotly_chart(pnl_histogram(df), use_container_width=True, config={"displayModeBar": False})

with col_exit:
    st.markdown('<div class="section-header">🚪 Exit Reasons</div>', unsafe_allow_html=True)
    st.plotly_chart(exit_reason_pie(df), use_container_width=True, config={"displayModeBar": False})

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# BREAKDOWN POR ESTRATEGIA
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🏆 Breakdown por Estrategia</div>', unsafe_allow_html=True)
st.plotly_chart(strategy_bar_chart(df), use_container_width=True, config={"displayModeBar": False})

# ─────────────────────────────────────────────────────────────────────────────
# BREAKDOWN POR FASE DEL CICLO
# ─────────────────────────────────────────────────────────────────────────────
if "regime" in df.columns and not df["regime"].isna().all():
    st.divider()
    st.markdown('<div class="section-header">🌐 Breakdown por Fase del Ciclo</div>', unsafe_allow_html=True)

    import plotly.graph_objects as go
    from dashboard.components.charts import PHASE_COLORS, _layout

    pnl_n = pd.to_numeric(df["pnl"], errors="coerce")
    regime_df = df.copy()
    regime_df["pnl"] = pnl_n
    regime_df["win"] = pnl_n > 0

    grp = regime_df.groupby("regime").agg(
        total=("pnl", "count"),
        wins=("win", "sum"),
        total_pnl=("pnl", "sum"),
        gross_profit=("pnl", lambda x: x[x > 0].sum()),
        gross_loss=("pnl", lambda x: abs(x[x <= 0].sum())),
    ).reset_index()

    grp["win_rate"] = grp["wins"] / grp["total"] * 100
    grp["pf"] = grp["gross_profit"] / grp["gross_loss"].replace(0, 1e-9)
    colors = [PHASE_COLORS.get(r.upper(), "#6C757D") for r in grp["regime"]]

    fig_r = go.Figure(go.Bar(
        x=grp["regime"], y=grp["total_pnl"],
        marker_color=colors, text=[f"${v:,.0f}" for v in grp["total_pnl"]],
        textposition="outside",
    ))
    fig_r.update_layout(**_layout(height=300, yaxis_title="PnL Total USD"))
    st.plotly_chart(fig_r, use_container_width=True, config={"displayModeBar": False})
