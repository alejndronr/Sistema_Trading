"""
dashboard/app.py
Entry point del dashboard — ZimaBlade Sistema Trading V6
"""
import time
import sys
from pathlib import Path

# ── Path setup (antes de cualquier import local) ───────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st

# ── Page Config (DEBE ser la primera llamada a st.*) ──────────────────────────
st.set_page_config(
    page_title="🤖 Sistema Trading ZimaBlade V6",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "Sistema de Trading Algorítmico Cuantitativo — ZimaBlade V6",
    },
)

# ── CSS ───────────────────────────────────────────────────────────────────────
CSS_PATH = Path(__file__).parent / "assets" / "style.css"
if CSS_PATH.exists():
    with open(CSS_PATH) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Auto-refresh cada 60s ──────────────────────────────────────────────────────
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

if time.time() - st.session_state.last_refresh > 60:
    st.session_state.last_refresh = time.time()
    st.rerun()

# ── Sidebar global ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 ZimaBlade V6")
    st.markdown("---")
    st.markdown("**Navegación**")
    st.markdown("""
    - [📊 Overview](/) 
    - [📈 Trades](/trades)
    - [📉 Equity & Riesgo](/equity)
    - [🤖 Machine Learning](/ml)
    - [🌐 Ciclo Macro](/cycle)
    - [⚙️ Sistema](/system)
    """)

    st.markdown("---")
    if st.button("🔄 Refrescar ahora", use_container_width=True):
        st.session_state.last_refresh = time.time()
        st.cache_data.clear()
        st.rerun()

    last = st.session_state.get("last_refresh", time.time())
    elapsed = int(time.time() - last)
    st.caption(f"Próxima actualización en: {max(0, 60 - elapsed)}s")

# ── Main page content (Overview) ──────────────────────────────────────────────
# La página principal se importa desde pages/1_overview.py cuando Streamlit
# maneja la navegación. Este archivo sirve como punto de entrada.
st.markdown("# 🤖 Sistema Trading ZimaBlade V6")
st.markdown("*Selecciona una página del sidebar izquierdo para comenzar.*")

# Mostrar estado rápido
from dashboard.components.db import get_heartbeat, pg_available
import datetime

col1, col2, col3 = st.columns(3)

with col1:
    if pg_available():
        st.success("✅ PostgreSQL conectado")
    else:
        st.error("❌ PostgreSQL no disponible — modo demo")

with col2:
    hb = get_heartbeat()
    if hb:
        st.success(f"✅ Motor: {hb.get('engine_version', 'V6')}")
    else:
        st.warning("⚠️ Motor offline o sin heartbeat")

with col3:
    st.info(f"🕐 {datetime.datetime.now().strftime('%H:%M:%S')}")
