"""
dashboard/pages/3_equity.py
Equity Curve y análisis completo de riesgo.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from dashboard.components.db import get_trades, get_equity_curve
from dashboard.components.metrics import compute_metrics
from dashboard.components.charts import equity_curve_chart, COLORS, _layout, _empty_fig

# ── CSS ───────────────────────────────────────────────────────────────────────
CSS_PATH = Path(__file__).parent.parent / "assets" / "style.css"
if CSS_PATH.exists():
    with open(CSS_PATH) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("# 📉 Equity Curve & Riesgo")
st.divider()

from dashboard.components.db import get_trades, get_equity_curve, pg_available

df = get_trades(limit=5000, real_only=False)
eq_df = get_equity_curve()

if not pg_available():
    np.random.seed(1)
    mock_dates = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=100, freq="12h")
    df = pd.DataFrame({
        "entry_time": mock_dates,
        "symbol": np.random.choice(["BTC/USDC", "ETH/USDC", "SOL/USDC"], 100),
        "direction": np.random.choice(["long", "short"], 100),
        "pnl": np.random.randn(100) * 15,
        "pnl_usd": np.random.randn(100) * 15,
        "r_multiple": np.random.randn(100) * 1.5,
        "duration_hours": np.random.uniform(2, 48, 100)
    })
    
    eq_df = pd.DataFrame({
        "timestamp": mock_dates,
        "equity": np.linspace(1400, 1450.75, 100) + np.cumsum(np.random.randn(100)*2)
    })
m = compute_metrics(df)

# ─────────────────────────────────────────────────────────────────────────────
# SELECTOR DE PERÍODO + EQUITY CURVE
# ─────────────────────────────────────────────────────────────────────────────
period = st.radio("Período:", ["1S", "1M", "3M", "6M", "1A", "ALL"],
                  index=5, horizontal=True, key="equity_period")

if not eq_df.empty:
    # Curva total normal
    fig = equity_curve_chart(eq_df, period)
    
    # Añadir trazos para longs y shorts si existen en los trades
    if not df.empty and "direction" in df.columns:
        from dashboard.components.charts import _filter_period, COLORS
        df_per = _filter_period(df.copy(), "entry_time", period)
        if not df_per.empty:
            df_per = df_per.sort_values("entry_time")
            longs = df_per[df_per["direction"] == "long"].copy()
            shorts = df_per[df_per["direction"] == "short"].copy()
            
            if not longs.empty:
                longs["cum_pnl"] = longs["pnl"].cumsum()
                fig.add_trace(go.Scatter(
                    x=longs["entry_time"], y=longs["cum_pnl"] + eq_df["equity"].iloc[0],
                    mode="lines", name="Longs", line=dict(color=COLORS["win"], width=1.5, dash="dot")
                ), row=1, col=1)
                
            if not shorts.empty:
                shorts["cum_pnl"] = shorts["pnl"].cumsum()
                fig.add_trace(go.Scatter(
                    x=shorts["entry_time"], y=shorts["cum_pnl"] + eq_df["equity"].iloc[0],
                    mode="lines", name="Shorts", line=dict(color=COLORS["loss"], width=1.5, dash="dot")
                ), row=1, col=1)
                
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})
else:
    st.info("Sin datos de equity. Ejecuta trades para ver la curva.")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# MÉTRICAS DE RIESGO DETALLADAS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Métricas de Riesgo Detalladas</div>', unsafe_allow_html=True)

r1, r2, r3, r4, r5, r6, r7, r8 = st.columns(8)

r1.metric("Max DD %",     f"{m['max_dd_pct']*100:.1f}%")
r2.metric("Max DD USD",   f"${m['max_dd_usd']:,.2f}")
r3.metric("Sharpe Anual", f"{m['sharpe_ratio']:.2f}")
r4.metric("Sortino",      f"{m['sortino_ratio']:.2f}")
r5.metric("Calmar",       f"{m['calmar_ratio']:.2f}")
r6.metric("Max Wins",     m["max_consecutive_wins"])
r7.metric("Max Losses",   m["max_consecutive_losses"])
recovery = (m["total_pnl"] / m["max_dd_usd"]) if m["max_dd_usd"] > 0 else 0
r8.metric("Recovery F.",  f"{recovery:.2f}")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# CORRELACIÓN ENTRE PARES
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🔗 Correlación de Retornos por Par</div>', unsafe_allow_html=True)

if not df.empty and "symbol" in df.columns and "pnl" in df.columns:
    try:
        pnl_by_sym = df.pivot_table(
            index="entry_time", columns="symbol", values="pnl", aggfunc="sum"
        ).fillna(0)

        if pnl_by_sym.shape[1] >= 2:
            corr = pnl_by_sym.corr()
            fig_corr = go.Figure(go.Heatmap(
                z=corr.values,
                x=corr.columns.tolist(),
                y=corr.index.tolist(),
                colorscale="RdYlGn",
                zmin=-1, zmax=1,
                text=[[f"{v:.2f}" for v in row] for row in corr.values],
                texttemplate="%{text}",
                showscale=True,
            ))
            fig_corr.update_layout(**_layout(height=350, title="Correlación de PnL entre pares"))

            # Alertar si hay correlación >0.7
            high_corr = [(corr.index[i], corr.columns[j])
                         for i in range(len(corr))
                         for j in range(i+1, len(corr.columns))
                         if abs(corr.iloc[i, j]) > 0.7]

            st.plotly_chart(fig_corr, use_container_width=True, config={"displayModeBar": False})

            if high_corr:
                for p1, p2 in high_corr:
                    st.warning(f"⚠️ Alta correlación ({corr.loc[p1, p2]:.2f}) entre {p1} y {p2} — riesgo oculto")
        else:
            st.info("Se necesitan al menos 2 pares con trades para calcular correlación.")
    except Exception as e:
        st.warning(f"No se pudo calcular correlación: {e}")
else:
    st.info("Sin suficientes datos de trades para mostrar correlación.")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# CIRCUIT BREAKERS — SEMÁFOROS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🚦 Circuit Breakers</div>', unsafe_allow_html=True)

from dashboard.components.db import get_portfolio_state, query_pg, pg_available
if not pg_available():
    ps = {"current_capital": 1450.75, "daily_start": 1430.0, "peak_capital": 1460.0}
else:
    ps = get_portfolio_state()

if ps:
    capital = float(ps.get("current_capital", 1000))
    daily_start = float(ps.get("daily_start", capital))
    peak = float(ps.get("peak_capital", capital))

    pnl_today_pct = ((capital - daily_start) / daily_start * 100) if daily_start > 0 else 0
    dd_peak_pct = ((peak - capital) / peak * 100) if peak > 0 else 0

    cb1, cb2, cb3, cb4 = st.columns(4)

    # CB Diario — reducir a -3%, pausar a -5%
    with cb1:
        if pnl_today_pct <= -5:
            color = "#FF4444"
            icon = "🔴"
            msg = "PAUSADO"
        elif pnl_today_pct <= -3:
            color = "#FFBB33"
            icon = "🟡"
            msg = "REDUCIDO"
        else:
            color = "#00C851"
            icon = "🟢"
            msg = "NORMAL"
        st.markdown(
            f'<div style="background:#12151E;border-radius:8px;padding:14px;border-left:4px solid {color};">'
            f'<b>{icon} CB Diario</b><br>'
            f'PnL Hoy: <b style="color:{color}">{pnl_today_pct:.2f}%</b><br>'
            f'Estado: <b style="color:{color}">{msg}</b><br>'
            f'<small>Umbral: -3% reduce / -5% pausa</small></div>',
            unsafe_allow_html=True
        )

    # CB Peak — apagar a -8%
    with cb2:
        if dd_peak_pct >= 8:
            color = "#FF4444"
            icon = "🔴"
            msg = "APAGADO"
        elif dd_peak_pct >= 5:
            color = "#FFBB33"
            icon = "🟡"
            msg = "ALERTA"
        else:
            color = "#00C851"
            icon = "🟢"
            msg = "NORMAL"
        st.markdown(
            f'<div style="background:#12151E;border-radius:8px;padding:14px;border-left:4px solid {color};">'
            f'<b>{icon} CB Peak</b><br>'
            f'DD desde Máximo: <b style="color:{color}">{dd_peak_pct:.2f}%</b><br>'
            f'Estado: <b style="color:{color}">{msg}</b><br>'
            f'<small>Umbral: -8% apaga motor</small></div>',
            unsafe_allow_html=True
        )

    # Suspensiones activas (SL consecutive)
    with cb3:
        try:
            from dashboard.components.db import get_suspended_symbols
            susp_df = get_suspended_symbols()
            n_susp = len(susp_df) if not susp_df.empty else 0
        except Exception:
            n_susp = 0

        color = "#FF4444" if n_susp > 0 else "#00C851"
        icon = "🔴" if n_susp > 0 else "🟢"
        st.markdown(
            f'<div style="background:#12151E;border-radius:8px;padding:14px;border-left:4px solid {color};">'
            f'<b>{icon} Suspensiones</b><br>'
            f'Pares suspendidos: <b style="color:{color}">{n_susp}</b><br>'
            f'<small>SL consecutivos ≥3 → 240min pausa</small></div>',
            unsafe_allow_html=True
        )

    # Cooldowns activos
    with cb4:
        st.markdown(
            '<div style="background:#12151E;border-radius:8px;padding:14px;border-left:4px solid #00C851;">'
            '<b>🟢 Cooldowns</b><br>'
            'Cooldowns activos: <b style="color:#00C851">0</b><br>'
            '<small>Espera 90min tras SL reciente</small></div>',
            unsafe_allow_html=True
        )
else:
    st.info("Sin datos de portfolio_state para evaluar circuit breakers.")
