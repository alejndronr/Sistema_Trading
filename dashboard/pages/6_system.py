"""
dashboard/pages/6_system.py
Sistema & Configuración: recursos ZimaBlade, servicios systemd, DB, logs y controles.
"""
import sys
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd

from dashboard.components.db import query_pg, query_sqlite, pg_available

# ── CSS ───────────────────────────────────────────────────────────────────────
CSS_PATH = Path(__file__).parent.parent / "assets" / "style.css"
if CSS_PATH.exists():
    with open(CSS_PATH) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("# ⚙️ Sistema & Configuración")
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# RECURSOS DEL SISTEMA (ZimaBlade Atom E3950)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🖥️ Estado del Sistema ZimaBlade</div>', unsafe_allow_html=True)

@st.cache_data(ttl=10)
def get_system_stats() -> dict:
    stats = {}
    try:
        import psutil
        cpu = psutil.cpu_percent(interval=0.5)
        ram = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        stats["cpu_pct"] = cpu
        stats["ram_pct"] = ram.percent
        stats["ram_total_gb"] = ram.total / 1e9
        stats["ram_used_gb"] = ram.used / 1e9
        stats["disk_pct"] = disk.percent
        stats["disk_total_gb"] = disk.total / 1e9
        stats["disk_used_gb"] = disk.used / 1e9
    except ImportError:
        stats = {"cpu_pct": 0, "ram_pct": 0, "ram_total_gb": 16, "ram_used_gb": 0,
                 "disk_pct": 0, "disk_total_gb": 1500, "disk_used_gb": 0}
    except Exception:
        pass

    # Temperatura (Linux)
    try:
        temp_path = Path("/sys/class/thermal/thermal_zone0/temp")
        if temp_path.exists():
            stats["cpu_temp"] = int(temp_path.read_text().strip()) / 1000
    except Exception:
        pass

    # Uptime del servicio
    try:
        res = subprocess.run(
            ["systemctl", "show", "trading-engine", "--property=ActiveEnterTimestamp"],
            capture_output=True, text=True, timeout=5
        )
        stats["engine_uptime_raw"] = res.stdout.strip()
    except Exception:
        pass

    return stats


sys_stats = get_system_stats()

s1, s2, s3, s4 = st.columns(4)

cpu = sys_stats.get("cpu_pct", 0)
cpu_color = "#FF4444" if cpu > 80 else "#FFBB33" if cpu > 60 else "#00C851"
s1.metric("🖥️ CPU", f"{cpu:.1f}%")
s1.progress(min(1.0, cpu / 100))

ram = sys_stats.get("ram_pct", 0)
ram_color = "#FF4444" if ram > 85 else "#FFBB33" if ram > 70 else "#00C851"
s2.metric("💾 RAM", f"{ram:.1f}%  ({sys_stats.get('ram_used_gb',0):.1f}/{sys_stats.get('ram_total_gb',16):.0f} GB)")
s2.progress(min(1.0, ram / 100))

disk = sys_stats.get("disk_pct", 0)
s3.metric("💿 Disco", f"{disk:.1f}%  ({sys_stats.get('disk_used_gb',0):.0f}/{sys_stats.get('disk_total_gb',1500):.0f} GB)")
s3.progress(min(1.0, disk / 100))

temp = sys_stats.get("cpu_temp")
if temp:
    temp_color = "#FF4444" if temp > 75 else "#FFBB33" if temp > 65 else "#00C851"
    s4.metric("🌡️ Temp CPU", f"{temp:.1f}°C")
else:
    s4.metric("🌡️ Temp CPU", "N/D")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SERVICIOS SYSTEMD
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🔧 Servicios Systemd</div>', unsafe_allow_html=True)

SERVICES = [
    "trading-engine",
    "trading-dashboard",
    "ohlcv-update.timer",
    "trading-retrain.timer",
]


@st.cache_data(ttl=15)
def get_service_status(service: str) -> str:
    try:
        result = subprocess.run(
            ["systemctl", "is-active", service],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


for svc in SERVICES:
    status = get_service_status(svc)
    is_active = status == "active"
    color = "#00C851" if is_active else "#FF4444"
    icon = "🟢" if is_active else "🔴"

    col_svc, col_status, col_btn = st.columns([3, 2, 2])

    with col_svc:
        st.markdown(f"**{svc}**")

    with col_status:
        st.markdown(f'{icon} <span style="color:{color};">{status.upper()}</span>', unsafe_allow_html=True)

    with col_btn:
        if svc == "trading-engine":
            if st.button(f"🔄 Restart {svc}", key=f"restart_{svc}"):
                st.session_state[f"confirm_restart_{svc}"] = True

            if st.session_state.get(f"confirm_restart_{svc}"):
                confirmed = st.checkbox("⚠️ Confirmar restart del motor", key=f"chk_restart_{svc}")
                if confirmed:
                    try:
                        subprocess.run(["systemctl", "restart", svc], timeout=10)
                        st.success(f"✅ {svc} reiniciado.")
                        st.session_state[f"confirm_restart_{svc}"] = False
                        st.cache_data.clear()
                    except Exception as e:
                        st.error(f"Error: {e}")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# BASE DE DATOS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🗄️ Base de Datos</div>', unsafe_allow_html=True)

col_pg, col_sqlite = st.columns(2)

with col_pg:
    st.markdown("##### PostgreSQL — trading_db")
    if pg_available():
        try:
            size_df = query_pg(
                "SELECT pg_size_pretty(pg_database_size('trading_db')) AS size"
            )
            trades_count = query_pg(
                "SELECT COUNT(*) as total, SUM(CASE WHEN is_backtest THEN 1 ELSE 0 END) as backtest FROM trades_journal"
            )
            st.markdown(f"📦 Tamaño BD: **{size_df.iloc[0]['size'] if not size_df.empty else 'N/D'}**")
            if not trades_count.empty:
                row = trades_count.iloc[0]
                real_trades = int(row["total"]) - int(row.get("backtest", 0) or 0)
                st.markdown(f"📊 Trades reales: **{real_trades:,}**")
                st.markdown(f"🧪 Backtest: **{int(row.get('backtest', 0) or 0):,}**")
        except Exception as e:
            st.warning(f"Error consultando PG: {e}")
    else:
        st.error("❌ PostgreSQL no disponible")

with col_sqlite:
    st.markdown("##### SQLite — trading.db (OHLCV)")
    sqlite_path = ROOT / "data" / "db" / "trading.db"
    if sqlite_path.exists():
        size_bytes = sqlite_path.stat().st_size
        st.markdown(f"📦 Tamaño: **{size_bytes / 1e6:.1f} MB**")

        ohlcv_summary = query_sqlite(
            "SELECT symbol, timeframe, COUNT(*) as candles, MAX(timestamp) as last_ts FROM ohlcv GROUP BY symbol, timeframe ORDER BY symbol, timeframe"
        )
        if not ohlcv_summary.empty:
            ohlcv_summary["last_date"] = pd.to_datetime(
                ohlcv_summary["last_ts"], unit="ms", utc=True
            ).dt.strftime("%Y-%m-%d")

            now_ms = int(time.time() * 1000)
            ohlcv_summary["stale"] = (now_ms - ohlcv_summary["last_ts"]) > 86400000 * 2

            def color_stale(val):
                return "color: #FF4444" if val else "color: #00C851"

            st.dataframe(
                ohlcv_summary[["symbol", "timeframe", "candles", "last_date", "stale"]]\
                    .style.applymap(color_stale, subset=["stale"])\
                    .format({"candles": "{:,}"}),
                height=250, use_container_width=True
            )
        else:
            st.info("Sin datos OHLCV en SQLite.")
    else:
        st.warning(f"trading.db no encontrado en {sqlite_path}")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN ACTIVA (.env)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">⚙️ Configuración Activa (.env)</div>', unsafe_allow_html=True)

env_path = ROOT / ".env"
if env_path.exists():
    HIDDEN_KEYS = {"API_KEY", "API_SECRET", "SECRET", "PASSWORD", "TOKEN"}
    env_lines = []
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, val = line.partition("=")
                key_upper = key.strip().upper()
                if any(h in key_upper for h in HIDDEN_KEYS):
                    val = "●●●●●●●●"
                env_lines.append(f"{key.strip()} = {val.strip()}")
    st.code("\n".join(env_lines), language="ini")
else:
    st.info(f"No se encontró .env en {env_path}")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# LOG VIEWER — journalctl
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📜 Log Viewer — trading-engine</div>', unsafe_allow_html=True)

log_level = st.selectbox("Filtrar nivel:", ["Todos", "info", "warning", "error"], index=0)

if st.button("🔃 Actualizar logs"):
    st.cache_data.clear()

@st.cache_data(ttl=30)
def get_logs(level: str) -> str:
    try:
        cmd = ["journalctl", "-u", "trading-engine", "-n", "50", "--no-pager",
               "--output=short-precise"]
        if level != "Todos":
            cmd += ["-p", level]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return res.stdout or res.stderr or "(sin logs disponibles)"
    except FileNotFoundError:
        # No systemd (en Windows/dev)
        log_file = ROOT / "logs" / "trading.log"
        if log_file.exists():
            lines = log_file.read_text().splitlines()[-50:]
            return "\n".join(lines)
        return "(journalctl no disponible en este sistema)"
    except Exception as e:
        return f"Error: {e}"


raw_logs = get_logs(log_level)

# Colorear logs por nivel
colored_lines = []
for line in raw_logs.splitlines():
    line_lower = line.lower()
    if "error" in line_lower or "critical" in line_lower:
        colored_lines.append(f'<span class="log-error">{line}</span>')
    elif "warning" in line_lower or "warn" in line_lower:
        colored_lines.append(f'<span class="log-warning">{line}</span>')
    else:
        colored_lines.append(f'<span class="log-info">{line}</span>')

st.markdown(
    f'<div class="log-viewer">{"<br>".join(colored_lines)}</div>',
    unsafe_allow_html=True
)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# PANEL DE CONTROL DEL MOTOR
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🎮 Control del Motor</div>', unsafe_allow_html=True)

st.warning("⚠️ **ZONA DE PELIGRO** — Todas las acciones requieren confirmación explícita.", icon="⚠️")

ctrl_cols = st.columns(4)

# Restart
with ctrl_cols[0]:
    if st.button("🔄 Reiniciar Motor", use_container_width=True):
        st.session_state["ctrl_restart"] = True
    if st.session_state.get("ctrl_restart"):
        if st.checkbox("✅ Confirmar restart", key="chk_ctrl_restart"):
            try:
                subprocess.run(["systemctl", "restart", "trading-engine"], timeout=10)
                st.success("Motor reiniciado.")
                st.session_state["ctrl_restart"] = False
            except Exception as e:
                st.error(f"Error: {e}")

# Pausar entradas
with ctrl_cols[1]:
    if st.button("⏸️ Pausar Entradas", use_container_width=True):
        st.session_state["ctrl_pause"] = True
    if st.session_state.get("ctrl_pause"):
        if st.checkbox("✅ Confirmar pausa", key="chk_ctrl_pause"):
            try:
                query_pg("UPDATE system_heartbeat SET paper_mode = TRUE WHERE id = 1")
                st.success("Entradas pausadas (paper mode activado).")
                st.session_state["ctrl_pause"] = False
            except Exception as e:
                st.error(f"Error: {e}")

# Actualizar OHLCV
with ctrl_cols[2]:
    if st.button("📥 Actualizar OHLCV", use_container_width=True):
        st.session_state["ctrl_ohlcv"] = True
    if st.session_state.get("ctrl_ohlcv"):
        if st.checkbox("✅ Confirmar descarga", key="chk_ctrl_ohlcv"):
            dl_script = ROOT / "scripts" / "fetch_data.py"
            if dl_script.exists():
                with st.spinner("Descargando OHLCV..."):
                    res = subprocess.run([sys.executable, str(dl_script)],
                                        capture_output=True, text=True, timeout=300)
                    if res.returncode == 0:
                        st.success("OHLCV actualizado.")
                    else:
                        st.error(f"Error: {res.stderr[-300:]}")
            else:
                st.warning(f"Script no encontrado: {dl_script}")
            st.session_state["ctrl_ohlcv"] = False

# ── KILL SWITCH ───────────────────────────────────────────────────────────────
with ctrl_cols[3]:
    st.markdown('<div style="border: 2px solid #FF4444; border-radius: 8px; padding: 8px;">', unsafe_allow_html=True)
    kill_1 = st.checkbox("🚨 Confirmar KILL", key="kill_chk_1")
    kill_2 = st.checkbox("⚠️ Estoy seguro", key="kill_chk_2")

    kill_btn = st.button("🚨 KILL SWITCH", type="primary", use_container_width=True,
                          disabled=not (kill_1 and kill_2))
    if kill_btn and kill_1 and kill_2:
        try:
            subprocess.run(["systemctl", "stop", "trading-engine"], timeout=10)
            st.error("🛑 Motor DETENIDO.")
        except Exception as e:
            st.error(f"Error al detener: {e}")
    st.markdown("</div>", unsafe_allow_html=True)
