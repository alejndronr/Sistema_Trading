"""
dashboard/app.py — ZimaBlade Trading HQ Dashboard V3
=====================================================
Correcciones aplicadas sobre V2 (análisis de revisión):

  ✓ CCXT Binance para precios — elimina dependencia de CoinGecko y rate-limit 429
  ✓ Fallback graceful: si CCXT falla, intenta CoinGecko; si falla, último precio BD
  ✓ Métricas globales por query de agregación separada (no limitadas por LIMIT 500)
  ✓ Timestamps → Europe/Madrid en todas las tablas y gráficos
  ✓ Pares USDT (los que usa el motor V4 en producción)
  ✓ Precio cacheado en sesión para no golpear Binance en cada rerun

Columnas exactas de trades_journal (verificadas):
  id, symbol, strategy, entry_time, exit_time, entry_price, exit_price,
  stop_loss, tp1, tp2, units, pnl, pnl_pct, r_multiple, exit_reason,
  ml_proba, regime, commission_paid, trade_id

Usuario BD: trading · Host: 127.0.0.1
"""

from __future__ import annotations

import csv
import io
import os
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
# ── Config ─────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

_raw        = os.environ.get("DATABASE_URL",
              "postgresql://trading:@127.0.0.1:5432/trading_db")
DB_URL      = (_raw.replace("+asyncpg", "").replace("localhost", "127.0.0.1"))

INITIAL_CAPITAL = float(os.environ.get("INITIAL_CAPITAL", "1000"))
TZ_LOCAL        = "Europe/Madrid"

# Pares del motor V4 — USDT (los que están en producción)
SYMBOLS: List[str] = [
    "BTC/USDC", "ETH/USDC", "SOL/USDC",
    "BNB/USDC", "LINK/USDC", "AVAX/USDC",
]

# Gate live (mismos criterios que live_engine V4)
GATE = {
    "win_rate":      {"label": "Win Rate",       "target": 45.0, "unit": "%",     "op": "gte"},
    "profit_factor": {"label": "Profit Factor",  "target": 1.3,  "unit": "",      "op": "gte"},
    "max_drawdown":  {"label": "Max Drawdown",   "target": -8.0, "unit": "%",     "op": "lte"},
    "min_trades":    {"label": "Trades cerrados","target": 20,   "unit": "",      "op": "gte"},
    "heartbeat":     {"label": "Heartbeat 24h",  "target": 0,    "unit": "fallos","op": "lte"},
    "period_weeks":  {"label": "Período mínimo", "target": 4,    "unit": "sem",   "op": "gte"},
}

PRICE_CACHE_TTL = 30   # segundos entre llamadas a Binance
MAX_OPEN_DISPLAY = 3   # máximo de posiciones simultáneas del motor V4
PRICE_CACHE_KEY = "_price_cache"
PRICE_TIME_KEY  = "_price_time"


# ══════════════════════════════════════════════════════════════════════════════
# ── Motor de precios: CCXT Binance → fallback CoinGecko → fallback BD ─────────
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_ccxt(symbols: List[str]) -> Optional[Dict[str, Dict]]:
    """
    Obtiene tickers de Binance vía ccxt (una sola llamada fetch_tickers).
    No requiere API key para precios públicos.
    Timeout 6s — si falla, devuelve None para activar el fallback.
    """
    try:
        import ccxt
        exchange = ccxt.binance({
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
        # fetch_tickers con lista específica = una sola petición HTTP
        tickers = exchange.fetch_tickers(symbols)
        result  = {}
        for sym in symbols:
            t = tickers.get(sym, {})
            result[sym] = {
                "price":      float(t.get("last",   0.0) or 0.0),
                "change_24h": float(t.get("percentage", 0.0) or 0.0),
                "bid":        float(t.get("bid",    0.0) or 0.0),
                "ask":        float(t.get("ask",    0.0) or 0.0),
                "volume_24h": float(t.get("baseVolume", 0.0) or 0.0),
                "source":     "binance",
            }
        return result
    except Exception as exc:
        st.toast(f"⚠️ CCXT: {exc} — intentando fallback", icon="⚠️")
        return None


def _fetch_coingecko(symbols: List[str]) -> Optional[Dict[str, Dict]]:
    """Fallback a CoinGecko si CCXT falla."""
    CG_MAP = {
        "BTC/USDC": "bitcoin",  "ETH/USDC": "ethereum",
        "SOL/USDC": "solana",   "BNB/USDC": "binancecoin",
        "LINK/USDC":"chainlink","AVAX/USDC":"avalanche-2",
        "XRP/USDC": "ripple",   "ADA/USDC": "cardano",
    }
    try:
        import requests
        ids  = ",".join(dict.fromkeys(CG_MAP.get(s, s.split("/")[0].lower()) for s in symbols))
        resp = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": ids, "vs_currencies": "usd",
                    "include_24hr_change": "true"},
            timeout=8,
        )
        if resp.status_code == 429:
            return None   # rate-limited, no reintentar
        data   = resp.json()
        result = {}
        for sym in symbols:
            cg   = CG_MAP.get(sym, sym.split("/")[0].lower())
            coin = data.get(cg, {})
            result[sym] = {
                "price":      float(coin.get("usd", 0.0) or 0.0),
                "change_24h": float(coin.get("usd_24h_change", 0.0) or 0.0),
                "bid":        0.0, "ask": 0.0, "volume_24h": 0.0,
                "source":     "coingecko",
            }
        return result
    except Exception:
        return None


def _prices_from_db(symbols: List[str]) -> Dict[str, Dict]:
    """
    Último recurso: precio de la última vela OHLCV en BD.
    Siempre devuelve algo (nunca falla).
    """
    result = {s: {"price": 0.0, "change_24h": 0.0, "bid": 0.0,
                  "ask": 0.0, "volume_24h": 0.0, "source": "db_fallback"}
              for s in symbols}
    try:
        engine = get_engine()
        if engine is None:
            return result
        for sym in symbols:
            df = qry("""
                SELECT close FROM ohlcv
                WHERE symbol=:s AND timeframe='1h'
                ORDER BY timestamp DESC LIMIT 2
            """, {"s": sym})
            if not df.empty:
                result[sym]["price"] = float(df["close"].iloc[0])
                if len(df) > 1:
                    prev = float(df["close"].iloc[1])
                    cur  = float(df["close"].iloc[0])
                    result[sym]["change_24h"] = (cur - prev) / prev * 100 if prev else 0
    except Exception:
        pass
    return result


def get_prices(symbols: List[str] = None) -> Dict[str, Dict]:
    """
    Obtiene precios con caché en session_state (TTL = PRICE_CACHE_TTL segundos).
    Cadena: Binance CCXT → CoinGecko → último OHLCV en BD.
    """
    if symbols is None:
        symbols = SYMBOLS

    now      = time.time()
    cached   = st.session_state.get(PRICE_CACHE_KEY)
    cached_t = st.session_state.get(PRICE_TIME_KEY, 0)

    if cached and (now - cached_t) < PRICE_CACHE_TTL:
        return cached

    data = _fetch_ccxt(symbols)
    if data is None:
        data = _fetch_coingecko(symbols)
    if data is None:
        data = _prices_from_db(symbols)

    st.session_state[PRICE_CACHE_KEY] = data
    st.session_state[PRICE_TIME_KEY]  = now
    return data


# ══════════════════════════════════════════════════════════════════════════════
# ── Base de datos ──────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_engine():
    try:
        e = create_engine(DB_URL, pool_pre_ping=True,
                          pool_size=3, max_overflow=2,
                          connect_args={"connect_timeout": 5})
        with e.connect() as c:
            c.execute(text("SELECT 1"))
        return e
    except Exception as ex:
        st.warning(f"⚠️ PostgreSQL no disponible: {ex}")
        return None


def qry(sql: str, params: dict = None) -> pd.DataFrame:
    e = get_engine()
    if e is None:
        return pd.DataFrame()
    try:
        with e.connect() as c:
            r = c.execute(text(sql), params or {})
            return pd.DataFrame(r.fetchall(), columns=list(r.keys()))
    except Exception as ex:
        st.error(f"Query error: {ex}")
        return pd.DataFrame()


def write(sql: str, params: dict = None) -> bool:
    e = get_engine()
    if e is None:
        return False
    try:
        with e.begin() as c:
            c.execute(text(sql), params or {})
        return True
    except Exception as ex:
        st.error(f"Write error: {ex}")
        return False


def ensure_manual_tables():
    write("""
        CREATE TABLE IF NOT EXISTS manual_investments (
            id          SERIAL PRIMARY KEY,
            symbol      VARCHAR(20)   NOT NULL,
            amount      DECIMAL(18,8) NOT NULL,
            buy_price   DECIMAL(18,8) NOT NULL,
            buy_date    DATE          NOT NULL DEFAULT CURRENT_DATE,
            exchange    VARCHAR(50)   DEFAULT 'Binance',
            tx_type     VARCHAR(30)   DEFAULT 'buy',
            notes       TEXT          DEFAULT '',
            status      VARCHAR(10)   DEFAULT 'open',
            created_at  TIMESTAMPTZ   DEFAULT NOW()
        );
        CREATE TABLE IF NOT EXISTS manual_closings (
            id              SERIAL PRIMARY KEY,
            investment_id   INT REFERENCES manual_investments(id) ON DELETE CASCADE,
            symbol          VARCHAR(20)   NOT NULL,
            amount_sold     DECIMAL(18,8) NOT NULL,
            buy_price       DECIMAL(18,8) NOT NULL,
            sell_price      DECIMAL(18,8) NOT NULL,
            sell_date       DATE          NOT NULL DEFAULT CURRENT_DATE,
            pnl_usd         DECIMAL(12,4),
            exchange        VARCHAR(50),
            notes           TEXT,
            created_at      TIMESTAMPTZ DEFAULT NOW()
        );
    """)


# ══════════════════════════════════════════════════════════════════════════════
# ── Helpers ────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def fmt_usd(v: float, sign: bool = True) -> str:
    s = "+" if v > 0 and sign else ""
    return f"{s}${v:,.2f}"

def fmt_pct(v: float) -> str:
    s = "+" if v > 0 else ""
    return f"{s}{v:.2f}%"

def to_madrid(series: pd.Series) -> pd.Series:
    """
    Convierte columna de timestamps UTC → Europe/Madrid.
    Maneja: naive, UTC, ya localizadas, y objetos date puros.
    utc=True coerce mezclas de timezones a UTC antes de convertir.
    """
    s = pd.to_datetime(series, errors="coerce", utc=True)
    return s.dt.tz_convert(TZ_LOCAL)

def passes_gate(key: str, val: float) -> bool:
    c = GATE[key]
    return val >= c["target"] if c["op"] == "gte" else val <= c["target"]

def color_val(v: float) -> str:
    return "#1D9E75" if v > 0 else ("#E24B4A" if v < 0 else "gray")


# ══════════════════════════════════════════════════════════════════════════════
# ── Métricas globales (query de agregación, sin LIMIT) ────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def global_metrics() -> Dict[str, Any]:
    """
    Calcula métricas sobre TODOS los trades cerrados.
    Query de agregación SQL: no carga filas en memoria, no afectado por LIMIT.
    """
    df = qry("""
        WITH trades_agrupados AS (
            SELECT
                symbol,
                DATE_TRUNC('minute', entry_time) AS entry_minute,
                SUM(COALESCE(pnl, pnl_usd))      AS pnl_total,
                MAX(ABS(COALESCE(r_multiple, 0))) AS r_abs,
                BOOL_OR(COALESCE(pnl, pnl_usd) > 0) AS es_ganador,
                MAX(exit_time)                   AS last_exit
            FROM trades_journal
            WHERE exit_time IS NOT NULL
            GROUP BY symbol, DATE_TRUNC('minute', entry_time)
        )
        SELECT
            COUNT(*)                                               AS total,
            SUM(CASE WHEN es_ganador THEN 1 ELSE 0 END)           AS wins,
            COALESCE(SUM(CASE WHEN pnl_total > 0
                THEN pnl_total ELSE 0 END), 0)                    AS gross_win,
            COALESCE(ABS(SUM(CASE WHEN pnl_total <= 0
                THEN pnl_total ELSE 0 END)), 0)                   AS gross_loss,
            COALESCE(SUM(pnl_total), 0)                           AS total_pnl,
            COALESCE(MAX(pnl_total), 0)                           AS best,
            COALESCE(MIN(pnl_total), 0)                           AS worst,
            COALESCE(AVG(CASE WHEN es_ganador THEN r_abs
                ELSE -r_abs END), 0)                              AS avg_r
        FROM trades_agrupados
    """)
    if df.empty:
        return dict(total=0, win_rate=0.0, profit_factor=0.0,
                    avg_r=0.0, total_pnl=0.0, best=0.0, worst=0.0)

    r = df.iloc[0]
    total      = int(r["total"])
    wins       = int(r["wins"])
    gross_win  = float(r["gross_win"])
    gross_loss = float(r["gross_loss"])
    win_rate   = (wins / total * 100) if total > 0 else 0.0
    pf         = (gross_win / gross_loss) if gross_loss > 0 else float("inf")

    return dict(
        total        = total,
        win_rate     = round(win_rate, 2),
        profit_factor= round(pf, 3),
        avg_r        = round(float(r["avg_r"]), 2),
        total_pnl    = round(float(r["total_pnl"]), 2),
        best         = round(float(r["best"]), 2),
        worst        = round(float(r["worst"]), 2),
    )


def drawdown_from_db() -> float:
    """
    Calcula el drawdown actual desde portfolio_state.
    (capital_actual - peak_capital) / peak_capital * 100
    """
    df = qry("SELECT current_capital, peak_capital FROM portfolio_state WHERE id=1")
    if df.empty:
        return 0.0
    cap  = float(df["current_capital"].iloc[0])
    peak = float(df["peak_capital"].iloc[0])
    return (cap - peak) / peak * 100 if peak > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# ── UI: estilos ────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="ZimaBlade Trading HQ",
    page_icon="📈",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
.mono{font-family:'JetBrains Mono',monospace;font-size:13px;}
div[data-testid="metric-container"]{
  background:rgba(128,128,128,.05);
  border:1px solid rgba(128,128,128,.12);
  border-radius:10px;padding:14px 18px;}
div[data-testid="metric-container"] label{
  font-size:11px!important;text-transform:uppercase;
  letter-spacing:.5px;color:rgba(128,128,128,.8)!important;}
div[data-testid="metric-container"] [data-testid="stMetricValue"]{
  font-size:24px!important;font-weight:600!important;}
button[data-baseweb="tab"][aria-selected="true"]{
  font-weight:600!important;border-bottom-color:#1D9E75!important;
  color:#1D9E75!important;}
.info-box{
  background:rgba(29,158,117,.08);border-left:3px solid #1D9E75;
  border-radius:0 6px 6px 0;padding:10px 14px;font-size:13px;margin-bottom:12px;}
.warn-box{
  background:rgba(239,159,39,.08);border-left:3px solid #EF9F27;
  border-radius:0 6px 6px 0;padding:10px 14px;font-size:13px;margin-bottom:12px;}
.price-row{
  display:flex;align-items:center;justify-content:space-between;
  padding:7px 0;border-bottom:1px solid rgba(128,128,128,.08);}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding-top:1.5rem!important;}
</style>
""", unsafe_allow_html=True)

# ── Inicialización ──────────────────────────────────────────────────────────────
ensure_manual_tables()

# ── Datos del header (siempre frescos, sin caché) ──────────────────────────────
ps  = qry("SELECT current_capital, peak_capital FROM portfolio_state WHERE id=1")
hb  = qry("SELECT last_ping, engine_version, paper_mode FROM system_heartbeat WHERE id=1")

capital   = float(ps["current_capital"].iloc[0]) if not ps.empty else INITIAL_CAPITAL
peak      = float(ps["peak_capital"].iloc[0])    if not ps.empty else INITIAL_CAPITAL
eng_ver   = str(hb["engine_version"].iloc[0])    if not hb.empty else "—"
paper     = bool(hb["paper_mode"].iloc[0])        if not hb.empty else True
last_ping = pd.to_datetime(hb["last_ping"].iloc[0], utc=True) if not hb.empty else None

# Estado del motor
if last_ping:
    if last_ping.tzinfo is None:
        last_ping = last_ping.tz_localize("UTC")
    since_min = (datetime.now(timezone.utc) - last_ping.tz_convert(timezone.utc)).total_seconds() / 60
    hb_icon   = "🟢" if since_min < 5 else ("🟡" if since_min < 60 else "🔴")
    hb_txt    = f"{hb_icon} Motor activo hace {int(since_min)}m"
else:
    hb_txt = "⚪ Sin datos"

mode_bg  = "#FEF3C7" if paper else "#D1FAE5"
mode_txt = "PAPER"   if paper else "LIVE"

st.markdown(f"""
<div style="display:flex;align-items:center;gap:12px;
     padding-bottom:16px;border-bottom:1px solid rgba(128,128,128,.15);
     margin-bottom:20px;">
  <span style="font-size:28px;">📈</span>
  <h1 style="font-size:22px;font-weight:600;margin:0;">ZimaBlade Trading HQ</h1>
  <span style="font-size:11px;font-weight:600;padding:3px 10px;border-radius:999px;
               background:{mode_bg};color:#333;">{mode_txt} · v{eng_ver}</span>
  <span style="font-size:12px;color:gray;margin-left:auto;">
    {hb_txt} &nbsp;·&nbsp; Capital: <strong>${capital:,.2f}</strong>
  </span>
</div>
""", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
t1,t2,t3,t4,t5,t6 = st.tabs([
    "🏠 Resumen", "💼 Portafolio", "🤖 Bot · Journal",
    "👤 Inversiones Manuales", "🛡️ Gate Live", "📋 Fiscal",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RESUMEN
# ══════════════════════════════════════════════════════════════════════════════
with t1:

    # Métricas globales (query de agregación, no limitadas por LIMIT 500)
    m      = global_metrics()
    dd_pct = drawdown_from_db()
    prices = get_prices()

    # Fuente del precio para mostrar en UI
    src_label = next((v["source"] for v in prices.values()), "—")
    src_color = "#1D9E75" if src_label == "binance" else (
                "#EF9F27" if src_label == "coingecko" else "#999")

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    _delta_capital = capital - INITIAL_CAPITAL
    # delta numérico → Streamlit colorea verde si >0, rojo si <0 automáticamente
    c1.metric("Capital Bot",
              value=f"${capital:,.2f}",
              delta=round(_delta_capital, 2))
    _delta_pnl = m["total_pnl"]
    c2.metric("PnL Total",
              value=f"${_delta_pnl:,.2f}",
              delta=round(_delta_pnl / INITIAL_CAPITAL * 100, 2) if INITIAL_CAPITAL else 0)
    c3.metric("Win Rate",        f"{m['win_rate']:.1f}%",
              delta="✓ ≥45%" if m["win_rate"] >= 45 else "✗ <45%")
    c4.metric("Profit Factor",   f"{m['profit_factor']:.2f}",
              delta="✓ ≥1.3" if m["profit_factor"] >= 1.3 else "✗ <1.3")
    c5.metric("Drawdown",        f"{dd_pct:.2f}%",
              delta_color="inverse" if dd_pct < -5 else "normal")
    c6.metric("Trades cerrados", str(m["total"]))

    st.divider()

    col_l, col_r = st.columns([2, 1])

    with col_l:
        # Curva de capital (últimas 500 velas cerradas para el gráfico)
        df_equity = qry("""
            SELECT
                MAX(exit_time)             AS exit_time,
                SUM(COALESCE(pnl, pnl_usd)) AS pnl
            FROM trades_journal
            WHERE exit_time IS NOT NULL
            GROUP BY symbol, DATE_TRUNC('minute', entry_time)
            ORDER BY MAX(exit_time) ASC
            LIMIT 500
        """)

        if df_equity.empty:
            st.markdown("""<div class="info-box">
            🔍 <strong>El motor V4 está analizando el mercado.</strong><br>
            Analiza 6 pares cada hora con score mínimo 55/100.
            Los primeros trades aparecerán aquí en cuanto se alcance el umbral.<br><br>
            <strong>Diagnóstico rápido:</strong><br>
            <code>journalctl -u trading-engine --since "1 hour ago" | grep -E "signal_queued|trade_opening|score"</code>
            </div>""", unsafe_allow_html=True)
        else:
            df_equity["exit_time"] = to_madrid(df_equity["exit_time"])
            df_equity["capital"]   = INITIAL_CAPITAL + df_equity["pnl"].astype(float).cumsum()

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_equity["exit_time"], y=df_equity["capital"],
                mode="lines", line=dict(color="#1D9E75", width=2),
                fill="tozeroy", fillcolor="rgba(29,158,117,.07)",
                name="Capital",
            ))
            fig.add_hline(y=INITIAL_CAPITAL, line_dash="dash",
                          line_color="rgba(128,128,128,.4)",
                          annotation_text="Capital inicial")
            fig.update_layout(
                title="Curva de capital (Europe/Madrid)",
                height=280, margin=dict(l=0,r=0,t=36,b=0),
                yaxis_tickprefix="$",
                xaxis_title=None,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
            )
            st.plotly_chart(fig, width='stretch')

    with col_r:
        # Precios en vivo con fuente indicada
        st.markdown(
            f"**Precios en vivo** &nbsp;"
            f"<span style='font-size:11px;color:{src_color};'>"
            f"● {src_label}</span>",
            unsafe_allow_html=True,
        )
        st.caption(f"Caché {PRICE_CACHE_TTL}s · {TZ_LOCAL}")

        for sym in SYMBOLS:
            p    = prices.get(sym, {})
            px_  = p.get("price", 0.0)
            chg  = p.get("change_24h", 0.0)
            col  = color_val(chg)
            sign = "▲" if chg >= 0 else "▼"
            ticker = sym.replace("/USDC", "")
            st.markdown(
                f"<div class='price-row'>"
                f"<span><strong>{ticker}</strong></span>"
                f"<span class='mono'>${px_:,.4f} "
                f"<span style='color:{col};font-size:11px;'>{sign}{abs(chg):.2f}%</span>"
                f"</span></div>",
                unsafe_allow_html=True,
            )

        if st.button("🔄 Actualizar precios", width='stretch'):
            # Limpiar caché de precios de session_state
            st.session_state.pop(PRICE_CACHE_KEY, None)
            st.session_state.pop(PRICE_TIME_KEY, None)
            st.rerun()

    st.divider()

    # Posiciones abiertas del bot
    df_open = qry("""
        SELECT symbol, strategy, direction, entry_time, entry_price,
               stop_loss, tp1, tp2,
               units, remaining_units,
               risk_amount, tp1_hit,
               ml_proba
        FROM positions WHERE status='open' ORDER BY entry_time DESC
    """)
    st.subheader(f"🤖 Posiciones abiertas del bot ({len(df_open)})")

    if df_open.empty:
        st.info("Sin posiciones abiertas. El motor analiza señales en cada cierre de vela 1H.")
    else:
        df_open["entry_time"] = to_madrid(df_open["entry_time"])
        rows_o = []
        for _, p in df_open.iterrows():
            sym  = str(p["symbol"])
            cur  = prices.get(sym, {}).get("price", float(p["entry_price"]))
            unreal = float(p["units"]) * (cur - float(p["entry_price"]))
            notional_pos  = float(p["units"]) * float(p["entry_price"])
            cap_pct_pos   = notional_pos / capital * 100 if capital > 0 else 0
            rem_units     = float(p.get("remaining_units") or p["units"])
            unreal_remain = rem_units * (cur - float(p["entry_price"]))
            rows_o.append({
                "Par":            sym,
                "Estrategia":     p.get("strategy","—"),
                "Dirección":      str(p.get("direction","long")).upper(),
                "Entrada":        f"${float(p['entry_price']):,.4f}",
                "Actual":         f"${cur:,.4f}",
                "SL":             f"${float(p['stop_loss']):,.4f}",
                "TP1":            f"${float(p['tp1']):,.4f}",
                "TP2":            f"${float(p['tp2']):,.4f}",
                "TP1 hit":        "✅" if p.get("tp1_hit") else "⬜",
                "Unidades":       round(float(p["units"]), 6),
                "Unid. restantes":round(rem_units, 6),
                "Capital inv.":   f"${notional_pos:,.2f}",
                "% capital":      f"{cap_pct_pos:.1f}%",
                "Riesgo USD":     f"${float(p['risk_amount']):,.2f}",
                "PnL no real.":   round(unreal_remain, 2),
                "ML proba":       f"{float(p['ml_proba']):.0%}" if p.get("ml_proba") else "—",
                "Apertura (MAD)": str(p["entry_time"]),
            })
        df_od = pd.DataFrame(rows_o)
        def _cpnl(v):
            try: return f"color:{color_val(float(v))};font-weight:500"
            except: return ""
        for _col in ["entry_price","stop_loss","tp1","tp2","units","risk_amount","ml_proba"]:
            if _col in df_open.columns:
                df_open[_col] = pd.to_numeric(df_open[_col], errors="coerce")
        st.dataframe(
            df_od.style.map(_cpnl, subset=["PnL no real."]),
            width='stretch', hide_index=True,
        )

    # ── Resumen de exposición de capital ─────────────────────────────────
    if not df_open.empty and rows_o:
        df_exp = pd.DataFrame(rows_o)
        total_notional = sum(
            float(str(r).replace("$","").replace(",",""))
            for r in df_exp["Capital inv."]
        )
        total_risk = sum(
            float(str(r).replace("$","").replace(",",""))
            for r in df_exp["Riesgo USD"]
        )
        pct_exposed = total_notional / capital * 100 if capital > 0 else 0

        st.markdown("**Exposición de capital**")
        ex1, ex2, ex3, ex4 = st.columns(4)
        ex1.metric("Capital en posiciones", f"${total_notional:,.2f}",
                   delta=f"{pct_exposed:.1f}% del total")
        ex2.metric("Capital libre",
                   f"${max(0, capital - total_notional):,.2f}",
                   delta=f"{max(0, 100 - pct_exposed):.1f}% disponible")
        ex3.metric("Riesgo total expuesto", f"${total_risk:,.2f}",
                   delta=f"{total_risk/capital*100:.1f}% del capital" if capital > 0 else "—",
                   delta_color="inverse")
        ex4.metric("Posiciones abiertas", f"{len(df_open)}/{MAX_OPEN_DISPLAY}")
        st.divider()

    # Actividad reciente
    df_recent = qry("""
        SELECT symbol, strategy,
               COALESCE(pnl, pnl_usd) AS pnl,
               exit_reason,
               COALESCE(regime, market_regime) AS regime,
               r_multiple,
               exit_time
        FROM trades_journal
        WHERE exit_time IS NOT NULL
        ORDER BY exit_time DESC LIMIT 5
    """)
    if not df_recent.empty:
        st.subheader("⚡ Últimas 5 operaciones cerradas")
        df_recent["exit_time"] = to_madrid(df_recent["exit_time"])
        df_recent["pnl"]       = df_recent["pnl"].astype(float)
        st.dataframe(df_recent, width='stretch', hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PORTAFOLIO COMBINADO
# ══════════════════════════════════════════════════════════════════════════════
with t2:
    st.subheader("💼 Portafolio combinado: bot + manuales")

    df_op2  = qry("SELECT symbol, strategy, direction, entry_price, units, risk_amount, tp1, tp2, remaining_units FROM positions WHERE status='open'")
    df_man2 = qry("SELECT symbol, amount, buy_price, exchange FROM manual_investments WHERE status='open'")

    all_syms = list({
        *(df_op2["symbol"].tolist()  if not df_op2.empty  else []),
        *(df_man2["symbol"].tolist() if not df_man2.empty else []),
    }) or SYMBOLS
    px2 = get_prices(all_syms)

    rows2 = []
    for _, p in (df_op2.iterrows() if not df_op2.empty else pd.DataFrame().iterrows()):
        sym  = str(p["symbol"])
        cur  = px2.get(sym, {}).get("price", float(p["entry_price"]))
        cost = float(p["units"]) * float(p["entry_price"])
        val  = float(p["units"]) * cur
        pnl  = val - cost
        rows2.append({
            "Activo": sym.replace("/USDC",""), "Tipo":"🤖 Bot",
            "Unidades": round(float(p["units"]),6),
            "Precio compra": f"${float(p['entry_price']):,.4f}",
            "Precio actual": f"${cur:,.4f}",
            "Valor USD": round(val,2), "Inversión": round(cost,2),
            "PnL USD": round(pnl,2),
            "PnL %": round(pnl/cost*100 if cost else 0, 2),
        })
    for _, t in (df_man2.iterrows() if not df_man2.empty else pd.DataFrame().iterrows()):
        sym  = str(t["symbol"])
        cur  = px2.get(sym, {}).get("price", float(t["buy_price"]))
        cost = float(t["amount"]) * float(t["buy_price"])
        val  = float(t["amount"]) * cur
        pnl  = val - cost
        rows2.append({
            "Activo": sym.replace("/USDC",""), "Tipo":"👤 Manual",
            "Unidades": round(float(t["amount"]),6),
            "Precio compra": f"${float(t['buy_price']):,.4f}",
            "Precio actual": f"${cur:,.4f}",
            "Valor USD": round(val,2), "Inversión": round(cost,2),
            "PnL USD": round(pnl,2),
            "PnL %": round(pnl/cost*100 if cost else 0, 2),
        })

    if rows2:
        df_pf = pd.DataFrame(rows2)
        tot_inv = df_pf["Inversión"].sum()
        tot_val = df_pf["Valor USD"].sum()
        tot_pnl = df_pf["PnL USD"].sum()

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Capital invertido", f"${tot_inv:,.2f}")
        c2.metric("Valor liquidativo", f"${tot_val:,.2f}")
        c3.metric("PnL no realizado",  fmt_usd(tot_pnl),
                  delta=fmt_pct(tot_pnl/tot_inv*100) if tot_inv else "—")
        c4.metric("Posiciones abiertas", len(rows2))

        def _cpf(v):
            try: return f"color:{color_val(float(v))};font-weight:500"
            except: return ""
        st.dataframe(
            df_pf.style.map(_cpf, subset=["PnL USD","PnL %"]),
            width='stretch', hide_index=True,
        )

        ca, cb = st.columns(2)
        with ca:
            agg = df_pf.groupby("Activo")["Valor USD"].sum().reset_index()
            fig = px.pie(agg, names="Activo", values="Valor USD",
                         title="Distribución por activo", hole=0.5,
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(height=260, margin=dict(l=0,r=0,t=36,b=0),
                              paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width='stretch')
        with cb:
            pba    = df_pf.groupby("Activo")["PnL USD"].sum().reset_index()
            colors = [color_val(v) for v in pba["PnL USD"]]
            fig2   = go.Figure(go.Bar(x=pba["Activo"], y=pba["PnL USD"],
                                      marker_color=colors))
            fig2.update_layout(title="PnL por activo", height=260,
                               margin=dict(l=0,r=0,t=36,b=0),
                               yaxis_tickprefix="$",
                               plot_bgcolor="rgba(0,0,0,0)",
                               paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, width='stretch')
    else:
        st.info("Sin posiciones abiertas. El motor analizará señales en el próximo cierre de vela.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — BOT JOURNAL
# ══════════════════════════════════════════════════════════════════════════════
with t3:
    st.subheader("🤖 Journal del bot — operaciones reales")

    # Métricas globales (sin LIMIT — query de agregación)
    mg = global_metrics()
    mc1,mc2,mc3,mc4,mc5,mc6 = st.columns(6)
    mc1.metric("Trades cerrados",  str(mg["total"]))
    mc2.metric("Win Rate",         f"{mg['win_rate']:.1f}%")
    mc3.metric("Profit Factor",    f"{mg['profit_factor']:.2f}")
    mc4.metric("R medio",          f"{mg['avg_r']:.2f}R")
    mc5.metric("Mejor trade",      fmt_usd(mg["best"]))
    mc6.metric("Peor trade",       fmt_usd(mg["worst"]))

    st.divider()

    # Filtros
    with st.expander("🔍 Filtros", expanded=False):
        fc1,fc2,fc3,fc4 = st.columns(4)
        f_sym   = fc1.text_input("Par (ej. BTC)")
        f_strat = fc2.text_input("Estrategia")
        f_reg   = fc3.selectbox("Régimen", ["","BULL","RANGE","BEAR","HIGH_VOL"])
        f_cl    = fc4.checkbox("Solo cerrados", value=False)

    where  = ["1=1"]
    params = {}
    if f_sym:
        where.append("symbol ILIKE :sym"); params["sym"] = f"%{f_sym}%"
    if f_strat:
        where.append("strategy ILIKE :strat"); params["strat"] = f"%{f_strat}%"
    if f_reg:
        where.append("regime = :reg"); params["reg"] = f_reg
    if f_cl:
        where.append("exit_time IS NOT NULL")

    # LIMIT 500 solo para la tabla visual
    df_j = qry(f"""
        SELECT id, symbol, strategy,
               COALESCE(direction,'') AS direction,
               entry_time, exit_time,
               entry_price, exit_price,
               tp1, tp2,
               COALESCE(units, 0)             AS units,
               COALESCE(pnl, pnl_usd)         AS pnl,
               pnl_pct, r_multiple, exit_reason,
               ml_proba,
               COALESCE(regime, market_regime) AS regime,
               commission_paid, setup_quality,
               risk_amount, duration_hours,
               entry_reason, observations
        FROM trades_journal
        WHERE {' AND '.join(where)}
        ORDER BY entry_time DESC
        LIMIT 500
    """, params)

    if df_j.empty:
        st.markdown("""<div class="info-box">
        📭 <strong>0 trades en trades_journal.</strong><br>
        El motor está corriendo (heartbeat activo). Las señales se generan
        al cierre de cada vela 1H con score ≥ 55/100.<br><br>
        Para ver qué está pasando en tiempo real:
        </div>""", unsafe_allow_html=True)
        st.code("""# Ver logs del motor en tiempo real
journalctl -u trading-engine -f | grep -E "signal_queued|trade_opening|score|regime"

# Si ves "analysis_cycle_start" sin "signal_queued"
# → ningún par alcanza score 55. Mercado sin señal clara.

# Si ves "signal_queued" sin "trade_opening"
# → micro-confirmación 15m fallando. Normal si hay ruido.

# Si ves "trade_opening" pero la BD sigue en 0
# → revisar _log_trade_extended y permisos de tabla trades_journal""",
                language="bash")
    else:
        # Convertir timestamps a Madrid
        for col in ["entry_time", "exit_time"]:
            if col in df_j.columns:
                df_j[col] = to_madrid(df_j[col]).dt.strftime("%Y-%m-%d %H:%M")

        cols_show = [c for c in [
            "symbol","strategy","direction",
            "entry_price","exit_price","tp1","tp2","units",
            "pnl","pnl_pct","r_multiple",
            "regime","ml_proba","setup_quality",
            "exit_reason","entry_reason",
            "risk_amount","duration_hours",
            "commission_paid","observations",
            "entry_time","exit_time",
        ] if c in df_j.columns]

        def _sj(v):
            try: return f"color:{color_val(float(v))};font-weight:500"
            except: return ""

        st.dataframe(
            df_j[cols_show].style.map(
                _sj, subset=[c for c in ["pnl","pnl_pct","r_multiple"] if c in df_j.columns]
            ),
            width='stretch', hide_index=True, height=420,
        )

        # Nota sobre tp1_partial
        st.caption(
            "ℹ️ Las métricas agrupan TP1 parcial + TP2 del mismo trade en un único resultado. "
            "La tabla muestra todas las filas de la BD para trazabilidad completa."
        )
        total_global = mg["total"]
        if total_global > 500:
            st.caption(f"Mostrando 500 de {total_global} trades.")

        # Gráficos
        ga, gb = st.columns(2)
        closed_j = df_j[df_j["exit_time"].notna() & df_j["pnl"].notna()].copy()
        closed_j["pnl"] = pd.to_numeric(closed_j["pnl"], errors="coerce")

        with ga:
            if not closed_j.empty:
                fig = px.histogram(closed_j, x="pnl", nbins=20,
                                   title="Distribución PnL",
                                   color_discrete_sequence=["#1D9E75"])
                fig.update_layout(height=240, margin=dict(l=0,r=0,t=36,b=0),
                                  xaxis_tickprefix="$",
                                  plot_bgcolor="rgba(0,0,0,0)",
                                  paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, width='stretch')
        with gb:
            if not closed_j.empty and "strategy" in closed_j.columns:
                sc  = closed_j.groupby("strategy").size().reset_index(name="count")
                fig2 = px.pie(sc, names="strategy", values="count",
                              title="Trades por estrategia", hole=0.5,
                              color_discrete_sequence=["#1D9E75","#378ADD","#EF9F27","#7F77DD"])
                fig2.update_layout(height=240, margin=dict(l=0,r=0,t=36,b=0),
                                   paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig2, width='stretch')

        csv_out = df_j.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Descargar journal CSV", csv_out,
                           f"journal_{date.today()}.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — INVERSIONES MANUALES
# ══════════════════════════════════════════════════════════════════════════════
with t4:
    st.subheader("👤 Inversiones manuales")
    col_f, col_t = st.columns([1, 2])

    with col_f:
        st.markdown("#### ➕ Registrar compra")
        with st.form("add_manual", clear_on_submit=True):
            ms  = st.text_input("Par (ej. BTC/USDC)").upper().strip()
            mc1_f, mc2_f = st.columns(2)
            ma  = mc1_f.number_input("Cantidad", min_value=0.0, format="%.8f")
            mp  = mc2_f.number_input("Precio USD", min_value=0.0, format="%.4f")
            md  = st.date_input("Fecha", value=date.today())
            me  = st.selectbox("Exchange",
                               ["Binance","Coinbase","Kraken","Bybit","Wallet","Otro"])
            mtt = st.selectbox("Tipo",
                               ["buy","transfer_in","airdrop","staking_reward"])
            mn  = st.text_area("Notas", height=60)
            if ma > 0 and mp > 0:
                st.info(f"💰 Total estimado: **${ma*mp:,.2f}**")
            if st.form_submit_button("💾 Guardar", width='stretch'):
                if not ms or ma <= 0 or mp <= 0:
                    st.error("Rellena símbolo, cantidad y precio.")
                else:
                    ok = write("""
                        INSERT INTO manual_investments
                            (symbol,amount,buy_price,buy_date,exchange,tx_type,notes,status)
                        VALUES (:s,:a,:p,:d,:e,:t,:n,'open')
                    """, {"s":ms,"a":ma,"p":mp,"d":md,"e":me,"t":mtt,"n":mn})
                    if ok:
                        st.success(f"✅ {ms} guardado")
                        st.rerun()

        st.divider()
        st.markdown("#### 🔴 Registrar venta")
        df_mo = qry("""
            SELECT id, symbol, amount, buy_price, buy_date, exchange
            FROM manual_investments WHERE status='open' ORDER BY buy_date DESC
        """)
        if df_mo.empty:
            st.info("Sin posiciones manuales abiertas.")
        else:
            opts = {
                f"{r['symbol']} · {r['amount']} @ ${float(r['buy_price']):.4f}": r
                for _, r in df_mo.iterrows()
            }
            with st.form("close_manual"):
                sel_lbl = st.selectbox("Posición", list(opts.keys()))
                sel_inv = opts[sel_lbl]
                max_a   = float(sel_inv["amount"])
                sc1, sc2 = st.columns(2)
                sa  = sc1.number_input("Cantidad vendida", min_value=0.0,
                                        max_value=max_a, value=max_a, format="%.8f")
                sp  = sc2.number_input("Precio venta USD", min_value=0.0, format="%.4f")
                sd  = st.date_input("Fecha venta", value=date.today())
                sn  = st.text_input("Notas")
                if sp > 0 and sa > 0:
                    pnl_est = sa * (sp - float(sel_inv["buy_price"]))
                    st.info(f"PnL estimado: **{fmt_usd(pnl_est)}**")
                if st.form_submit_button("✅ Registrar venta", width='stretch'):
                    if sp <= 0 or sa <= 0:
                        st.error("Precio y cantidad obligatorios.")
                    else:
                        pnl_r = sa * (sp - float(sel_inv["buy_price"]))
                        write("""
                            INSERT INTO manual_closings
                                (investment_id,symbol,amount_sold,buy_price,
                                 sell_price,sell_date,pnl_usd,exchange,notes)
                            VALUES (:ii,:sym,:a,:bp,:sp,:sd,:pnl,:ex,:n)
                        """, {
                            "ii":int(sel_inv["id"]), "sym":str(sel_inv["symbol"]),
                            "a":sa, "bp":float(sel_inv["buy_price"]),
                            "sp":sp, "sd":sd, "pnl":pnl_r,
                            "ex":str(sel_inv.get("exchange","Binance")), "n":sn,
                        })
                        if sa >= max_a - 1e-9:
                            write("UPDATE manual_investments SET status='closed' WHERE id=:id",
                                  {"id":int(sel_inv["id"])})
                        else:
                            write("UPDATE manual_investments SET amount=:a WHERE id=:id",
                                  {"a": max_a - sa, "id": int(sel_inv["id"])})
                        st.success("✅ Venta registrada")
                        st.rerun()

    with col_t:
        st.markdown("#### 📊 Posiciones abiertas")
        df_mo2 = qry("""
            SELECT id, symbol, amount, buy_price, buy_date, exchange, notes
            FROM manual_investments WHERE status='open' ORDER BY buy_date DESC
        """)
        if df_mo2.empty:
            st.info("Sin inversiones manuales.")
        else:
            syms_m = tuple(df_mo2["symbol"].unique())
            px_m   = get_prices(list(syms_m))
            rows_m = []
            for _, t in df_mo2.iterrows():
                sym  = str(t["symbol"])
                cur  = px_m.get(sym, {}).get("price", float(t["buy_price"]))
                pnl  = float(t["amount"]) * (cur - float(t["buy_price"]))
                cost = float(t["amount"]) * float(t["buy_price"])
                pct  = pnl / cost * 100 if cost > 0 else 0
                rows_m.append({
                    "ID":       int(t["id"]),
                    "Activo":   sym.replace("/USDC",""),
                    "Cantidad": float(t["amount"]),
                    "Compra":   f"${float(t['buy_price']):,.4f}",
                    "Actual":   f"${cur:,.4f}",
                    "PnL USD":  round(pnl, 2),
                    "PnL %":    round(pct, 2),
                    "Exchange": str(t.get("exchange","—")),
                    "Fecha":    str(t["buy_date"]),
                })
            df_rm = pd.DataFrame(rows_m)
            def _cm(v):
                try: return f"color:{color_val(float(v))};font-weight:500"
                except: return ""
            st.dataframe(
                df_rm.style.map(_cm, subset=["PnL USD","PnL %"]),
                width='stretch', hide_index=True,
            )
            tot_m = df_rm["PnL USD"].sum()
            st.markdown(
                f"**PnL no realizado:** "
                f"<span style='color:{color_val(tot_m)};font-weight:600;'>"
                f"{fmt_usd(tot_m)}</span>",
                unsafe_allow_html=True,
            )

        st.divider()
        st.markdown("#### 📜 Historial de ventas")
        df_cl = qry("""
            SELECT symbol, amount_sold, buy_price, sell_price,
                   pnl_usd, sell_date, exchange, notes
            FROM manual_closings ORDER BY sell_date DESC
        """)
        if df_cl.empty:
            st.info("Sin ventas registradas aún.")
        else:
            df_cl["pnl_usd"] = df_cl["pnl_usd"].astype(float)
            def _cc(v):
                try: return f"color:{color_val(float(v))};font-weight:500"
                except: return ""
            st.dataframe(
                df_cl.style.map(_cc, subset=["pnl_usd"]),
                width='stretch', hide_index=True,
            )
            tot_r = float(df_cl["pnl_usd"].sum())
            st.markdown(
                f"**Total ganancia realizada:** "
                f"<span style='color:{color_val(tot_r)};font-weight:600;'>"
                f"{fmt_usd(tot_r)}</span>",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — GATE LIVE
# ══════════════════════════════════════════════════════════════════════════════
with t5:
    st.subheader("🛡️ Gate de validación pre-live")
    st.markdown("""<div class="info-box">
    Todos los criterios deben cumplirse antes de activar capital real.
    Las métricas se calculan sobre <strong>todos</strong> los trades cerrados (sin LIMIT).
    </div>""", unsafe_allow_html=True)

    mg5    = global_metrics()
    dd5    = drawdown_from_db()

    weeks_in = 0.0
    df_first = qry("""
        SELECT MIN(entry_time) AS first_trade FROM trades_journal
        WHERE exit_time IS NOT NULL
    """)
    if not df_first.empty and df_first["first_trade"].iloc[0] is not None:
        first = pd.to_datetime(df_first["first_trade"].iloc[0], utc=True)
        weeks_in = (datetime.now(timezone.utc) - first).days / 7

    current_vals = {
        "win_rate":      mg5["win_rate"],
        "profit_factor": mg5["profit_factor"],
        "max_drawdown":  dd5,
        "min_trades":    mg5["total"],
        "heartbeat":     0,
        "period_weeks":  round(weeks_in, 1),
    }

    pass_count = sum(passes_gate(k, v) for k, v in current_vals.items())
    total_c    = len(current_vals)
    st.progress(pass_count / total_c,
                text=f"{pass_count}/{total_c} criterios superados")

    if pass_count == total_c:
        st.success("🚀 ¡Todos los criterios superados! El sistema está listo para capital real.")
    else:
        st.warning(f"⚠️ Faltan {total_c - pass_count} criterio(s) para activar modo live.")

    st.divider()

    for key, crit in GATE.items():
        val = current_vals.get(key, 0)
        ok  = passes_gate(key, val)
        icon = "✅" if ok else ("⏳" if val == 0 and key in ("min_trades","period_weeks") else "❌")
        target_str = f"{'≥' if crit['op']=='gte' else '≤'} {crit['target']}{crit['unit']}"
        c_i, c_n, c_v, c_t = st.columns([0.5, 3, 2, 2])
        c_i.markdown(f"<span style='font-size:20px;'>{icon}</span>",
                     unsafe_allow_html=True)
        c_n.markdown(f"**{crit['label']}**")
        col_v = "#1D9E75" if ok else "#E24B4A"
        c_v.markdown(
            f"<span style='color:{col_v};font-weight:600;'>"
            f"{val:.1f}{crit['unit']}</span>",
            unsafe_allow_html=True,
        )
        c_t.markdown(
            f"<span style='color:gray;font-size:12px;'>{target_str}</span>",
            unsafe_allow_html=True,
        )
        st.divider()

    # RAM bare-metal desde /proc/meminfo
    try:
        with open("/proc/meminfo") as f:
            mi = dict(line.split(":") for line in f.read().splitlines() if ":" in line)
        total_gb = int(mi["MemTotal"].strip().split()[0])    / 1e6
        avail_gb = int(mi["MemAvailable"].strip().split()[0])/ 1e6
        used_gb  = total_gb - avail_gb
        pct_used = used_gb / total_gb * 100
        ram_ok   = used_gb < 4.0
        st.markdown(
            f"**💾 RAM ZimaBlade:** `{used_gb:.2f} GB` / `{total_gb:.1f} GB` "
            f"({pct_used:.0f}%) {'✅' if ram_ok else '⚠️ >4GB'}",
        )
        st.progress(min(pct_used/100, 1.0))
    except Exception:
        st.caption("RAM: no disponible (ejecutar en el servidor)")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — FISCAL
# ══════════════════════════════════════════════════════════════════════════════
with t6:
    st.subheader("📋 Informes fiscales · AEAT España")
    st.markdown("""<div class="info-box">
    <strong>Método FIFO</strong> obligatorio por la AEAT.
    CSV compatible con <strong>Koinly</strong> y <strong>CoinTracking</strong>.
    Timestamps en <strong>Europe/Madrid</strong>.
    </div>""", unsafe_allow_html=True)

    # Bot: compras y ventas cerradas
    df_bot_f = qry("""
        SELECT entry_time AS fecha, symbol,
               'BUY'  AS tipo, 'Bot' AS fuente,
               COALESCE(units, position_size) AS cantidad,
               entry_price AS precio_usd,
               COALESCE(units, position_size)*entry_price AS total_usd,
               COALESCE(units, position_size)*entry_price AS coste_fifo,
               NULL::numeric AS ganancia,
               commission_paid,
               strategy AS notas
        FROM trades_journal
        UNION ALL
        SELECT exit_time, symbol,
               'SELL', 'Bot',
               COALESCE(units, position_size), exit_price,
               COALESCE(units, position_size)*exit_price,
               COALESCE(units, position_size)*entry_price,
               COALESCE(pnl, pnl_usd),
               commission_paid, exit_reason
        FROM trades_journal
        WHERE exit_time IS NOT NULL
    """)

    # Manuales: compras + ventas
    df_man_f = qry("""
        SELECT buy_date AS fecha, symbol,
               tx_type AS tipo, 'Manual' AS fuente,
               amount AS cantidad, buy_price AS precio_usd,
               amount*buy_price AS total_usd,
               amount*buy_price AS coste_fifo,
               NULL::numeric AS ganancia,
               NULL::numeric AS commission_paid,
               notes AS notas
        FROM manual_investments
        UNION ALL
        SELECT sell_date, symbol,
               'SELL', 'Manual',
               amount_sold, sell_price,
               amount_sold*sell_price,
               amount_sold*buy_price,
               pnl_usd,
               NULL, notes
        FROM manual_closings
    """)

    frames = [df for df in [df_bot_f, df_man_f] if not df.empty]
    if not frames:
        st.info("Sin operaciones aún. Los informes aparecerán cuando el bot cierre sus primeros trades.")
    else:
        df_fiscal = pd.concat(frames, ignore_index=True)
        df_fiscal["fecha"] = pd.to_datetime(df_fiscal["fecha"], errors="coerce", utc=True)
        # Convertir a Madrid para fiscal
        df_fiscal["fecha"] = to_madrid(df_fiscal["fecha"]).dt.date
        df_fiscal = df_fiscal.sort_values("fecha").reset_index(drop=True)

        # Métricas fiscales del año actual
        year_now = date.today().year
        df_year  = df_fiscal[pd.to_datetime(df_fiscal["fecha"], errors="coerce", utc=True).dt.year == year_now]
        sells    = df_year[df_year["tipo"]=="SELL"].copy()
        sells["ganancia"] = pd.to_numeric(sells["ganancia"], errors="coerce").fillna(0)
        gains  = sells[sells["ganancia"] > 0]["ganancia"].sum()
        losses = sells[sells["ganancia"] <= 0]["ganancia"].sum()
        net    = gains + losses
        irpf   = net * 0.19 if net > 0 else 0

        fi1,fi2,fi3,fi4 = st.columns(4)
        fi1.metric(f"Ganancias {year_now}", fmt_usd(gains))
        fi2.metric(f"Pérdidas {year_now}",  fmt_usd(losses))
        fi3.metric("Base imponible neta",   fmt_usd(net))
        fi4.metric("IRPF estimado ~19%",    f"~${irpf:,.2f}")
        st.caption("⚠️ Estimación orientativa. Consulta a un asesor fiscal.")

        st.divider()
        # Castear columnas numéricas a float64 para compatibilidad Arrow/PyArrow
        for _col in ["cantidad", "precio_usd", "total_usd", "coste_fifo",
                     "ganancia", "commission_paid"]:
            if _col in df_fiscal.columns:
                df_fiscal[_col] = pd.to_numeric(df_fiscal[_col], errors="coerce")
        st.dataframe(df_fiscal, width='stretch',
                     hide_index=True, height=320)

        # Koinly CSV
        def to_koinly(df: pd.DataFrame) -> bytes:
            buf = io.StringIO()
            w   = csv.writer(buf)
            w.writerow([
                "Date","Sent Amount","Sent Currency",
                "Received Amount","Received Currency",
                "Fee Amount","Fee Currency",
                "Net Worth Amount","Net Worth Currency",
                "Label","Description","TxHash",
            ])
            for _, r in df.iterrows():
                ds   = str(r["fecha"]) + " 00:00 UTC"
                sym  = str(r["symbol"]).replace("/USDC","")
                fee  = r.get("commission_paid","") or ""
                note = r.get("notas","") or ""
                if str(r["tipo"]).upper() in ("BUY","TRANSFER_IN","AIRDROP","STAKING_REWARD"):
                    w.writerow([ds, r["total_usd"],"USD", r["cantidad"],sym,
                                fee,"USD", r["total_usd"],"USD","",note,""])
                else:
                    w.writerow([ds, r["cantidad"],sym, r["total_usd"],"USD",
                                fee,"USD", r["total_usd"],"USD","",note,""])
            return buf.getvalue().encode("utf-8")

        ce1, ce2 = st.columns(2)
        with ce1:
            st.download_button(
                "📥 Exportar Koinly CSV",
                to_koinly(df_fiscal),
                f"koinly_{year_now}_{date.today()}.csv",
                "text/csv",
                width='stretch',
            )
        with ce2:
            st.download_button(
                "📥 CSV completo (raw)",
                df_fiscal.to_csv(index=False).encode("utf-8"),
                f"fiscal_completo_{year_now}_{date.today()}.csv",
                "text/csv",
                width='stretch',
            )

        # Gráfico por activo
        if not sells.empty:
            by_asset = (sells.groupby("symbol")["ganancia"]
                        .sum().reset_index()
                        .sort_values("ganancia", ascending=False))
            fig_tax = go.Figure(go.Bar(
                x=by_asset["symbol"].str.replace("/USDC",""),
                y=by_asset["ganancia"],
                marker_color=[color_val(v) for v in by_asset["ganancia"]],
            ))
            fig_tax.update_layout(
                title=f"Ganancias/pérdidas realizadas por activo ({year_now})",
                height=240, margin=dict(l=0,r=0,t=36,b=0),
                yaxis_tickprefix="$",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_tax, width='stretch')

        with st.expander("📌 Guía Koinly · AEAT España"):
            st.markdown(f"""
1. Descarga el **CSV Koinly** de arriba (timestamps en UTC, requerido por Koinly).
2. En Koinly → **Importar** → **CSV personalizado** → mapea las columnas.
3. Koinly aplicará FIFO automáticamente.
4. Exporta el informe PDF para adjuntarlo a tu declaración.

**Tramos IRPF {year_now} (base del ahorro):**
| Tramo | Tipo |
|---|---|
| Hasta 6.000 € | 19% |
| 6.001 – 50.000 € | 21% |
| 50.001 – 200.000 € | 23% |
| Más de 200.000 € | 27% |

**Modelo 721:** obligatorio si tienes >50.000 € en exchanges extranjeros a 31/12.
            """)
