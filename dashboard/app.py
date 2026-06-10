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

# ── Main page content (Overview) ──────────────────────────────────────────────
# Redirigir automáticamente a la página de Overview para que actúe como Home.
st.switch_page("pages/1_overview.py")
