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

pg_ok = pg_available()
hb = get_heartbeat()

pg_pill = f"""<div class="status-pill status-{'success' if pg_ok else 'warning'}">
    <div class="pulsing-dot" style="color: {'#34D399' if pg_ok else '#FBBF24'}"></div>
    {'PostgreSQL Live' if pg_ok else 'Local Demo Mode'}
</div>"""

engine_pill = f"""<div class="status-pill status-{'success' if hb else 'warning'}">
    <div class="pulsing-dot" style="color: {'#34D399' if hb else '#FBBF24'}"></div>
    {'Engine Active (' + hb.get('engine_version', 'V6') + ')' if hb else 'Engine Offline (Demo)'}
</div>"""

st.markdown(f'<div style="display: flex; gap: 10px; margin-bottom: 2rem;">{pg_pill}{engine_pill}</div>', unsafe_allow_html=True)
