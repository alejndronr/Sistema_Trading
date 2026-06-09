"""
dashboard/pages/4_ml.py
Panel de Machine Learning: estado del modelo, historial retrains, calibración.
"""
import sys
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from dashboard.components.db import get_trades, get_ml_retrain_log
from dashboard.components.charts import ml_roc_history, ml_calibration_scatter, COLORS, _layout, _empty_fig, pnl_histogram

# ── CSS ───────────────────────────────────────────────────────────────────────
CSS_PATH = Path(__file__).parent.parent / "assets" / "style.css"
if CSS_PATH.exists():
    with open(CSS_PATH) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("# 🤖 Machine Learning")
st.divider()

retrain_df = get_ml_retrain_log()
trades_df = get_trades(limit=3000, real_only=True)

# ─────────────────────────────────────────────────────────────────────────────
# ESTADO DEL MODELO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Estado del Modelo</div>', unsafe_allow_html=True)

col_state, col_retrain_btn = st.columns([3, 1])

with col_state:
    if not retrain_df.empty:
        latest = retrain_df.iloc[0]
        m1, m2, m3, m4, m5, m6 = st.columns(6)

        roc = float(latest.get("roc_auc", 0))
        f1 = float(latest.get("cv_f1", 0))
        n_samp = int(latest.get("n_samples", 0))
        top_feat = latest.get("top_feature", "-")
        thresh = float(latest.get("threshold", 0.40))
        retrain_date = latest.get("retrain_date", "-")

        roc_color = COLORS["win"] if roc >= 0.65 else COLORS["warn"] if roc >= 0.55 else COLORS["loss"]

        m1.metric("📅 Último Retrain", str(retrain_date)[:10] if retrain_date else "-")
        m2.metric("🎯 ROC-AUC", f"{roc:.3f}", delta=f"{'✅ OK' if roc >= 0.65 else '⚠️ Bajo'}")
        m3.metric("📊 CV F1", f"{f1:.3f}")
        m4.metric("🎚️ Threshold", f"{thresh:.2f}")
        m5.metric("🔢 Muestras", f"{n_samp:,}")
        m6.metric("⭐ Top Feature", top_feat)

        # Distribución ganadores/perdedores
        if not trades_df.empty and "pnl" in trades_df.columns:
            pnl_n = pd.to_numeric(trades_df["pnl"], errors="coerce").dropna()
            n_wins = (pnl_n > 0).sum()
            n_loss = (pnl_n <= 0).sum()
            total = n_wins + n_loss
            if total > 0:
                st.caption(f"Dataset: {n_wins} ganadores ({n_wins/total*100:.1f}%) / {n_loss} perdedores ({n_loss/total*100:.1f}%)")
    else:
        st.info("Sin historial de retrains. Ejecuta `ml/retrain_model.py` para empezar.")

with col_retrain_btn:
    st.markdown("### 🔄 Retrain Manual")
    if st.button("🔄 Ejecutar Retrain ML", use_container_width=True, type="secondary"):
        with st.spinner("Ejecutando retrain del modelo ML..."):
            ml_script = ROOT / "ml" / "retrain_model.py"
            if ml_script.exists():
                result = subprocess.run(
                    [sys.executable, str(ml_script), "--min-trades", "30"],
                    capture_output=True, text=True, timeout=120,
                )
                if result.returncode == 0:
                    st.success("✅ Retrain completado exitosamente.")
                    st.code(result.stdout[-1000:] if result.stdout else "")
                    st.cache_data.clear()
                else:
                    st.error(f"❌ Error en retrain: {result.stderr[-500:]}")
            else:
                st.warning(f"Script no encontrado: {ml_script}")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# HISTORIAL DE RETRAINS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📈 Historial de Retrains</div>', unsafe_allow_html=True)

col_hist_tbl, col_hist_chart = st.columns([2, 3])

with col_hist_tbl:
    if not retrain_df.empty:
        display_cols = [c for c in ["retrain_date", "roc_auc", "cv_f1", "n_samples", "top_feature", "threshold"] if c in retrain_df.columns]
        st.dataframe(
            retrain_df[display_cols].style.format(
                {"roc_auc": "{:.3f}", "cv_f1": "{:.3f}", "threshold": "{:.2f}", "n_samples": "{:,}"},
                na_rep="-"
            ),
            height=300, use_container_width=True
        )
    else:
        st.info("Sin historial disponible.")

with col_hist_chart:
    if not retrain_df.empty and "roc_auc" in retrain_df.columns:
        retrain_plot = retrain_df.copy()
        retrain_plot["retrain_date"] = pd.to_datetime(retrain_plot["retrain_date"], errors="coerce")
        st.plotly_chart(ml_roc_history(retrain_plot.dropna(subset=["retrain_date"])),
                        use_container_width=True, config={"displayModeBar": False})
    else:
        st.plotly_chart(_empty_fig("Sin datos de historial ROC-AUC"), use_container_width=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# ML PROBA DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🎲 Distribución ML Proba</div>', unsafe_allow_html=True)

col_proba, col_calib = st.columns(2)

with col_proba:
    if not trades_df.empty and "ml_proba" in trades_df.columns:
        proba = pd.to_numeric(trades_df["ml_proba"], errors="coerce").dropna()
        pnl_n = pd.to_numeric(trades_df["pnl"], errors="coerce")
        wins_mask = pnl_n > 0

        fig_p = go.Figure()
        fig_p.add_trace(go.Histogram(
            x=proba[wins_mask.reindex(proba.index, fill_value=False)],
            name="Ganadores", nbinsx=20, marker_color=COLORS["win"], opacity=0.8
        ))
        fig_p.add_trace(go.Histogram(
            x=proba[~wins_mask.reindex(proba.index, fill_value=True)],
            name="Perdedores", nbinsx=20, marker_color=COLORS["loss"], opacity=0.8
        ))
        from dashboard.components.charts import _layout as lay
        fig_p.update_layout(**lay(height=300, barmode="overlay",
                                  xaxis_title="ML Proba", yaxis_title="Frecuencia"))
        st.plotly_chart(fig_p, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("Sin datos de ML proba en trades.")

with col_calib:
    st.markdown("##### Calibración del Modelo")
    if not trades_df.empty and "ml_proba" in trades_df.columns:
        st.plotly_chart(ml_calibration_scatter(trades_df), use_container_width=True,
                        config={"displayModeBar": False})
        st.caption("📖 Cuanto más cercana a la diagonal = mejor calibrado")
    else:
        st.info("Sin datos suficientes para calibración.")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">⭐ Feature Importance</div>', unsafe_allow_html=True)

model_meta_path = ROOT / "ml" / "model_metadata.json"
features = {}

if model_meta_path.exists():
    try:
        with open(model_meta_path) as f:
            meta = json.load(f)
        features = meta.get("feature_importance", {})
    except Exception:
        pass

if features:
    feat_df = pd.DataFrame(list(features.items()), columns=["Feature", "Importance"])\
                .sort_values("Importance", ascending=True)

    fig_fi = go.Figure(go.Bar(
        y=feat_df["Feature"], x=feat_df["Importance"],
        orientation="h", marker_color=COLORS["blue"],
        text=[f"{v:.4f}" for v in feat_df["Importance"]],
        textposition="outside",
    ))
    from dashboard.components.charts import _layout as lay
    fig_fi.update_layout(**lay(height=max(300, len(features) * 22), xaxis_title="Importance"))
    st.plotly_chart(fig_fi, use_container_width=True, config={"displayModeBar": False})
else:
    st.info(f"Feature importance no disponible. Ejecuta retrain para generar `ml/model_metadata.json`.")
    # Placeholder con features conocidas
    placeholder_features = [
        "rsi_14", "adx_14", "ema_21_55_cross", "volume_ratio", "atr_14",
        "bb_width", "macd_signal", "hurst_90", "zscore_vwap", "supertrend_dir",
        "ema_200_dist", "rsi_weekly", "price_vs_ema55", "obv_trend", "stoch_rsi_k",
    ]
    st.caption("Features utilizadas en el modelo (importancias pendientes de retrain):")
    st.write(", ".join(placeholder_features))
