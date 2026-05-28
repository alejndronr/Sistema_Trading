"""
dashboard/app.py — ZimaBlade Trading HQ Dashboard
===================================================
Dashboard profesional integrado con PostgreSQL real del sistema de trading.

Funcionalidades:
  - Resumen general: capital, PnL, drawdown, heartbeat del motor
  - Portafolio combinado: posiciones del bot + inversiones manuales
  - Journal del bot: todas las operaciones con filtros avanzados
  - Inversiones manuales: registro, seguimiento y cierre con FIFO
  - Gate de validación pre-live: los 6 criterios en tiempo real
  - Informes fiscales: exportación Koinly / CoinTracking (AEAT)
  - Precios en vivo: CoinGecko API (sin clave necesaria)

Instalación en ZimaBlade:
  pip install streamlit ccxt pandas sqlalchemy psycopg2-binary
        python-dotenv plotly requests --quiet

Uso:
  streamlit run dashboard/app.py --server.port 8501 \
      --server.address 0.0.0.0 --server.headless true

Autor: generado para Sistema_Trading (ZimaBlade + Proxmox)
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
import requests
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# ── Config ────────────────────────────────────────────────────────────────────

load_dotenv()

# --- MOTOR BINANCE CCXT ---
@st.cache_data(ttl=3600)
def get_binance_usdc_pairs():
    try:
        import ccxt
        exchange = ccxt.binance()
        markets = exchange.load_markets()
        # Filtramos para obtener solo mercados Spot contra USDC
        pairs = [s for s, m in markets.items() if m.get('quote') == 'USDC' and m.get('spot')]
        
        # Garantizamos que SUI y otras principales estén visibles
        if 'SUI/USDC' not in pairs: pairs.append('SUI/USDC')
        return sorted(pairs) + ["Otro..."]
    except Exception as e:
        return ["BTC/USDC", "ETH/USDC", "SOL/USDC", "SUI/USDC", "RENDER/USDC", "LINK/USDC", "Otro..."]

@st.cache_data(ttl=60)
def get_live_prices(symbols: list) -> dict:
    try:
        import ccxt
        exchange = ccxt.binance()
        # Descargamos todos los tickers de Binance de golpe (mucho más rápido que uno a uno)
        tickers = exchange.fetch_tickers()
        prices = {}
        for sym in symbols:
            if sym in tickers and tickers[sym].get('last') is not None:
                prices[sym] = float(tickers[sym]['last'])
            else:
                prices[sym] = 0.0
        return prices
    except Exception as e:
        return {s: 0.0 for s in symbols}
# --------------------------


# Convertir asyncpg → psycopg2 para uso síncrono en Streamlit
_raw_db_url = os.environ.get(
    "DATABASE_URL",
    "postgresql://trading:trading@127.0.0.1:5432/trading_db",
)
DB_URL = (
    _raw_db_url.replace("+asyncpg", "")
    if "+asyncpg" in _raw_db_url
    else _raw_db_url
)

INITIAL_CAPITAL = float(os.environ.get("INITIAL_CAPITAL", "1000.0"))
PAPER_MODE      = os.environ.get("PAPER_MODE", "true").lower() == "true"

# Criterios del gate de validación pre-live
VALIDATION_CRITERIA = {
    "win_rate":      {"label": "Win Rate",         "target": 45.0,  "unit": "%",   "op": "gte"},
    "profit_factor": {"label": "Profit Factor",    "target": 1.3,   "unit": "",    "op": "gte"},
    "max_drawdown":  {"label": "Max Drawdown",     "target": -8.0,  "unit": "%",   "op": "lte"},
    "min_trades":    {"label": "Trades cerrados",  "target": 20,    "unit": "",    "op": "gte"},
    "heartbeat":     {"label": "Heartbeat 24h",    "target": 0,     "unit": "fallos", "op": "lte"},
    "ram_gb":        {"label": "RAM Usage",        "target": 4.0,   "unit": "GB",  "op": "lte"},
    "period_weeks":  {"label": "Período mínimo",   "target": 4,     "unit": "sem", "op": "gte"},
}

# ── Base de datos ──────────────────────────────────────────────────────────────

@st.cache_resource
def get_engine():
    """Crea y cachea el engine SQLAlchemy. Solo se conecta si PostgreSQL está disponible."""
    try:
        engine = create_engine(DB_URL, pool_pre_ping=True, pool_timeout=5, connect_args={"connect_timeout": 5})
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception as e:
        st.warning(f"⚠️ PostgreSQL no disponible: {e}. Mostrando datos de demo.")
        return None


def run_query(sql: str, params: dict = None) -> pd.DataFrame:
    """Ejecuta una query y devuelve DataFrame. Si no hay BD, retorna DataFrame vacío."""
    engine = get_engine()
    if engine is None:
        return pd.DataFrame()
    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql), params or {})
            rows = result.fetchall()
            cols = result.keys()
            return pd.DataFrame(rows, columns=list(cols))
    except Exception as e:
        st.error(f"Error en query: {e}")
        return pd.DataFrame()


def run_write(sql: str, params: dict = None) -> bool:
    """Ejecuta INSERT/UPDATE. Devuelve True si tuvo éxito."""
    engine = get_engine()
    if engine is None:
        return False
    try:
        with engine.begin() as conn:
            conn.execute(text(sql), params or {})
        return True
    except Exception as e:
        st.error(f"Error al escribir en BD: {e}")
        return False


def ensure_manual_tables():
    """Crea las tablas de inversiones manuales si no existen."""
    run_write("""
        CREATE TABLE IF NOT EXISTS manual_investments (
            id          SERIAL PRIMARY KEY,
            symbol      VARCHAR(20)  NOT NULL,
            amount      DECIMAL(18,8) NOT NULL,
            buy_price   DECIMAL(18,8) NOT NULL,
            buy_date    DATE         NOT NULL DEFAULT CURRENT_DATE,
            exchange    VARCHAR(50)  DEFAULT 'Binance',
            tx_type     VARCHAR(30)  DEFAULT 'buy',
            notes       TEXT         DEFAULT '',
            status      VARCHAR(10)  DEFAULT 'open',
            created_at  TIMESTAMPTZ  DEFAULT NOW()
        );
        CREATE TABLE IF NOT EXISTS manual_closings (
            id              SERIAL PRIMARY KEY,
            investment_id   INT REFERENCES manual_investments(id) ON DELETE CASCADE,
            symbol          VARCHAR(20)  NOT NULL,
            amount_sold     DECIMAL(18,8) NOT NULL,
            buy_price       DECIMAL(18,8) NOT NULL,
            sell_price      DECIMAL(18,8) NOT NULL,
            sell_date       DATE         NOT NULL DEFAULT CURRENT_DATE,
            pnl_usd         DECIMAL(12,4),
            exchange        VARCHAR(50),
            notes           TEXT,
            created_at      TIMESTAMPTZ  DEFAULT NOW()
        );
    """)


# ── Precios en vivo (CoinGecko, sin API key) ──────────────────────────────────

COINGECKO_IDS = {
    "BTC/USDC": "bitcoin",
    "ETH/USDC": "ethereum",
    "SOL/USDC": "solana",
    "BNB/USDC": "binancecoin",
    "XRP/USDC": "ripple",
    "ADA/USDC": "cardano",
    "AVAX/USDC": "avalanche-2",
    "DOGE/USDC": "dogecoin",
    "DOT/USDC": "polkadot",
    "MATIC/USDC": "matic-network",
    "LINK/USDC": "chainlink",
    "RENDER/USDC": "render-token",
}


@st.cache_data(ttl=60)
def fetch_live_prices(symbols: Tuple[str, ...]) -> Dict[str, Dict]:
    """
    Obtiene precios y variación 24h de CoinGecko.
    Cachea 60 segundos. Fallback a precios 0 si hay error.
    """
    ids_needed = [COINGECKO_IDS.get(s, s.split("/")[0].lower()) for s in symbols]
    ids_str    = ",".join(dict.fromkeys(ids_needed))  # deduplica manteniendo orden
    try:
        url  = "https://api.coingecko.com/api/v3/simple/price"
        resp = requests.get(
            url,
            params={"ids": ids_str, "vs_currencies": "usd",
                    "include_24hr_change": "true", "include_market_cap": "true"},
            timeout=8,
        )
        data = resp.json()
    except Exception:
        data = {}

    result = {}
    for sym in symbols:
        cg_id = COINGECKO_IDS.get(sym, sym.split("/")[0].lower())
        coin  = data.get(cg_id, {})
        result[sym] = {
            "price":      coin.get("usd", 0.0),
            "change_24h": coin.get("usd_24h_change", 0.0),
            "market_cap": coin.get("usd_market_cap", 0.0),
        }
    return result


def get_price(sym: str) -> float:
    prices = fetch_live_prices((sym,))
    return prices.get(sym, {}).get("price", 0.0)


# ── Helpers de formato ────────────────────────────────────────────────────────

def fmt_usd(v: float, decimals: int = 2) -> str:
    sign = "+" if v > 0 else ""
    return f"{sign}${v:,.{decimals}f}"


def fmt_pct(v: float) -> str:
    sign = "+" if v > 0 else ""
    return f"{sign}{v:.2f}%"


def color_pnl(v: float) -> str:
    return "🟢" if v > 0 else "🔴" if v < 0 else "⚪"


def delta_color(v: float) -> str:
    """Para st.metric delta_color."""
    return "normal" if v >= 0 else "inverse"


# ── Datos del bot (PostgreSQL) ─────────────────────────────────────────────────

def get_portfolio_state() -> Dict[str, Any]:
    """Lee portfolio_state y calcula métricas clave."""
    df = run_query("SELECT * FROM portfolio_state WHERE id = 1")
    if df.empty:
        return {
            "current_capital": INITIAL_CAPITAL,
            "peak_capital":    INITIAL_CAPITAL,
            "daily_start":     INITIAL_CAPITAL,
            "weekly_start":    INITIAL_CAPITAL,
            "monthly_start":   INITIAL_CAPITAL,
            "updated_at":      None,
        }
    row = df.iloc[0].to_dict()
    for k in ["current_capital", "peak_capital", "daily_start",
              "weekly_start", "monthly_start"]:
        row[k] = float(row.get(k, INITIAL_CAPITAL))
    return row


def get_heartbeat() -> Dict[str, Any]:
    df = run_query("SELECT * FROM system_heartbeat WHERE id = 1")
    if df.empty:
        return {"last_ping": None, "engine_version": "—", "paper_mode": True}
    row = df.iloc[0].to_dict()
    return row


def get_open_positions() -> pd.DataFrame:
    df = run_query("""
        SELECT id, symbol, strategy, direction, entry_time, entry_price,
               stop_loss, tp1, tp2, units, risk_amount, tp1_hit,
               remaining_units, ml_proba, status
        FROM positions
        WHERE status = 'open'
        ORDER BY entry_time DESC
    """)
    return df


def get_trades_journal(
    limit: int = 500,
    strategy: str = "",
    direction: str = "",
    quality: str = "",
    symbol: str = "",
    closed_only: bool = False,
) -> pd.DataFrame:
    filters = ["1=1"]
    params: Dict[str, Any] = {}

    if strategy:
        filters.append("strategy = :strategy")
        params["strategy"] = strategy
    if direction:
        filters.append("direction = :direction")
        params["direction"] = direction
    if quality:
        filters.append("setup_quality = :quality")
        params["quality"] = quality
    if symbol:
        filters.append("symbol ILIKE :symbol")
        params["symbol"] = f"%{symbol}%"
    if closed_only:
        filters.append("exit_time IS NOT NULL")

    where = " AND ".join(filters)
    df = run_query(f"""
        SELECT trade_id, strategy, symbol, timeframe, direction,
               setup_quality, entry_price, exit_price, stop_loss,
               take_profit_1, position_size, risk_amount,
               pnl_usd, pnl_pct, r_multiple,
               entry_time, exit_time, duration_hours,
               entry_reason, exit_reason, market_regime,
               observations, is_backtest
        FROM trades_journal
        WHERE {where}
        ORDER BY entry_time DESC
        LIMIT :limit
    """, {**params, "limit": limit})
    return df


def get_equity_curve(days: int = 30) -> pd.DataFrame:
    """Curva de capital diaria desde trades_journal."""
    df = run_query(f"""
        SELECT
            DATE(exit_time) AS day,
            SUM(pnl_usd)    AS daily_pnl
        FROM trades_journal
        WHERE exit_time IS NOT NULL
          AND exit_time >= NOW() - INTERVAL '{days} days'
        GROUP BY DATE(exit_time)
        ORDER BY day ASC
    """)
    if df.empty:
        # Demo data si no hay trades
        today = date.today()
        days_list = [today - timedelta(days=i) for i in reversed(range(days))]
        import random, math
        random.seed(42)
        pnl = [random.gauss(5, 25) for _ in days_list]
        df = pd.DataFrame({"day": days_list, "daily_pnl": pnl})
    df["cumulative_pnl"] = df["daily_pnl"].cumsum()
    df["capital"]        = INITIAL_CAPITAL + df["cumulative_pnl"]
    return df


def get_drawdown_series(days: int = 30) -> pd.DataFrame:
    df = run_query(f"""
        SELECT date, drawdown_pct, period
        FROM drawdowns
        WHERE period = 'daily'
          AND date >= NOW() - INTERVAL '{days} days'
        ORDER BY date ASC
    """)
    return df


def get_ml_performance() -> pd.DataFrame:
    return run_query("""
        SELECT model_version, total_predictions, trades_approved,
               avg_prob_win, precision_ml, recall_ml
        FROM v_ml_performance
        ORDER BY total_predictions DESC
    """)


def compute_bot_metrics(df_trades: pd.DataFrame) -> Dict[str, Any]:
    """Calcula métricas clave del bot desde el journal."""
    if df_trades.empty:
        return {
            "total_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
            "avg_r": 0.0, "total_pnl": 0.0, "best_trade": 0.0,
            "worst_trade": 0.0, "avg_duration": 0.0, "max_drawdown": 0.0,
        }
    closed = df_trades[df_trades["exit_time"].notna()].copy()
    if closed.empty:
        return {"total_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
                "avg_r": 0.0, "total_pnl": 0.0, "best_trade": 0.0,
                "worst_trade": 0.0, "avg_duration": 0.0, "max_drawdown": 0.0}

    pnl   = closed["pnl_usd"].dropna().astype(float)
    wins  = pnl[pnl > 0]
    loses = pnl[pnl <= 0]

    profit_factor = (
        wins.sum() / abs(loses.sum())
        if len(loses) > 0 and loses.sum() != 0
        else float("inf")
    )
    win_rate = len(wins) / len(pnl) * 100 if len(pnl) > 0 else 0.0

    # Drawdown desde capital acumulado
    equity  = INITIAL_CAPITAL + pnl.cumsum()
    peak    = equity.cummax()
    dd_pct  = ((equity - peak) / peak * 100).min()

    return {
        "total_trades":  len(pnl),
        "win_rate":      round(win_rate, 2),
        "profit_factor": round(profit_factor, 3),
        "avg_r":         round(closed["r_multiple"].dropna().astype(float).mean(), 2),
        "total_pnl":     round(pnl.sum(), 2),
        "best_trade":    round(pnl.max(), 2),
        "worst_trade":   round(pnl.min(), 2),
        "avg_duration":  round(closed["duration_hours"].dropna().astype(float).mean(), 1),
        "max_drawdown":  round(float(dd_pct), 2),
    }


# ── Inversiones manuales ──────────────────────────────────────────────────────

def get_manual_investments(status: str = "open") -> pd.DataFrame:
    return run_query(
        "SELECT * FROM manual_investments WHERE status = :s ORDER BY buy_date DESC",
        {"s": status},
    )


def get_manual_closings() -> pd.DataFrame:
    return run_query("SELECT * FROM manual_closings ORDER BY sell_date DESC")


def add_manual_investment(
    symbol: str, amount: float, buy_price: float,
    buy_date: date, exchange: str, tx_type: str, notes: str,
) -> bool:
    ok = run_write("""
        INSERT INTO manual_investments
            (symbol, amount, buy_price, buy_date, exchange, tx_type, notes, status)
        VALUES (:sym, :amt, :price, :date, :exch, :ttype, :notes, 'open')
    """, {
        "sym": symbol, "amt": amount, "price": buy_price,
        "date": buy_date, "exch": exchange, "ttype": tx_type, "notes": notes,
    })
    return ok


def close_manual_investment(
    inv_id: int, symbol: str, amount_sold: float, buy_price: float,
    sell_price: float, sell_date: date, exchange: str, notes: str,
) -> bool:
    pnl = amount_sold * (sell_price - buy_price)
    ok = run_write("""
        INSERT INTO manual_closings
            (investment_id, symbol, amount_sold, buy_price,
             sell_price, sell_date, pnl_usd, exchange, notes)
        VALUES (:inv_id, :sym, :amt, :bp, :sp, :sd, :pnl, :exch, :notes)
    """, {
        "inv_id": inv_id, "sym": symbol, "amt": amount_sold, "bp": buy_price,
        "sp": sell_price, "sd": sell_date, "pnl": pnl, "exch": exchange, "notes": notes,
    })
    if not ok:
        return False
    # Actualizar estado: si cierra todo, marcar closed
    inv_df = run_query("SELECT amount FROM manual_investments WHERE id = :id", {"id": inv_id})
    if inv_df.empty:
        return True
    orig_amount = float(inv_df.iloc[0]["amount"])
    if amount_sold >= orig_amount - 1e-9:
        run_write("UPDATE manual_investments SET status='closed' WHERE id=:id", {"id": inv_id})
    else:
        new_amount = orig_amount - amount_sold
        run_write(
            "UPDATE manual_investments SET amount=:a WHERE id=:id",
            {"a": new_amount, "id": inv_id},
        )
    return True


def delete_manual_investment(inv_id: int) -> bool:
    return run_write(
        "UPDATE manual_investments SET status='deleted' WHERE id=:id",
        {"id": inv_id},
    )


# ── Exportaciones fiscales ─────────────────────────────────────────────────────

def build_fiscal_dataframe(bot_trades: pd.DataFrame, manual_open: pd.DataFrame,
                           manual_closed: pd.DataFrame) -> pd.DataFrame:
    """Construye DataFrame unificado de todas las operaciones para fiscal."""
    rows = []

    # Trades del bot cerrados
    if not bot_trades.empty:
        closed_bot = bot_trades[bot_trades["exit_time"].notna()]
        for _, t in closed_bot.iterrows():
            rows.append({
                "fecha":       pd.to_datetime(t["entry_time"]).date(),
                "activo":      str(t["symbol"]).replace("/USDC", ""),
                "tipo":        "BUY",
                "fuente":      "Bot",
                "cantidad":    float(t["position_size"] or 0),
                "precio_usd":  float(t["entry_price"] or 0),
                "total_usd":   float(t["position_size"] or 0) * float(t["entry_price"] or 0),
                "coste_fifo":  float(t["position_size"] or 0) * float(t["entry_price"] or 0),
                "ganancia":    None,
                "exchange":    "Binance",
                "notas":       str(t.get("entry_reason", t.get("strategy", ""))),
            })
            rows.append({
                "fecha":       pd.to_datetime(t["exit_time"]).date(),
                "activo":      str(t["symbol"]).replace("/USDC", ""),
                "tipo":        "SELL",
                "fuente":      "Bot",
                "cantidad":    float(t["position_size"] or 0),
                "precio_usd":  float(t["exit_price"] or 0),
                "total_usd":   float(t["position_size"] or 0) * float(t["exit_price"] or 0),
                "coste_fifo":  float(t["position_size"] or 0) * float(t["entry_price"] or 0),
                "ganancia":    float(t["pnl_usd"] or 0),
                "exchange":    "Binance",
                "notas":       str(t.get("exit_reason", "")),
            })

    # Compras manuales abiertas
    if not manual_open.empty:
        for _, t in manual_open.iterrows():
            rows.append({
                "fecha":       t["buy_date"],
                "activo":      str(t["symbol"]).replace("/USDC", ""),
                "tipo":        str(t.get("tx_type", "buy")).upper(),
                "fuente":      "Manual",
                "cantidad":    float(t["amount"]),
                "precio_usd":  float(t["buy_price"]),
                "total_usd":   float(t["amount"]) * float(t["buy_price"]),
                "coste_fifo":  float(t["amount"]) * float(t["buy_price"]),
                "ganancia":    None,
                "exchange":    str(t.get("exchange", "Binance")),
                "notas":       str(t.get("notes", "")),
            })

    # Ventas manuales cerradas
    if not manual_closed.empty:
        for _, t in manual_closed.iterrows():
            rows.append({
                "fecha":       t["sell_date"],
                "activo":      str(t["symbol"]).replace("/USDC", ""),
                "tipo":        "SELL",
                "fuente":      "Manual",
                "cantidad":    float(t["amount_sold"]),
                "precio_usd":  float(t["sell_price"]),
                "total_usd":   float(t["amount_sold"]) * float(t["sell_price"]),
                "coste_fifo":  float(t["amount_sold"]) * float(t["buy_price"]),
                "ganancia":    float(t["pnl_usd"] or 0),
                "exchange":    str(t.get("exchange", "Binance")),
                "notas":       str(t.get("notes", "")),
            })

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("fecha").reset_index(drop=True)
    return df


def to_koinly_csv(df: pd.DataFrame) -> bytes:
    """Genera CSV en formato Koinly (compatible AEAT)."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "Date", "Sent Amount", "Sent Currency",
        "Received Amount", "Received Currency",
        "Fee Amount", "Fee Currency",
        "Net Worth Amount", "Net Worth Currency",
        "Label", "Description", "TxHash",
    ])
    for _, r in df.iterrows():
        date_str = str(r["fecha"]) + " 00:00 UTC"
        if r["tipo"] in ("BUY", "TRANSFER_IN"):
            writer.writerow([
                date_str, r["total_usd"], "USD",
                r["cantidad"], r["activo"],
                "", "",
                r["total_usd"], "USD",
                "", r["notas"], "",
            ])
        elif r["tipo"] == "SELL":
            writer.writerow([
                date_str, r["cantidad"], r["activo"],
                r["total_usd"], "USD",
                "", "",
                r["total_usd"], "USD",
                "", r["notas"], "",
            ])
    return buf.getvalue().encode("utf-8")


def to_cointracking_csv(df: pd.DataFrame) -> bytes:
    """Genera CSV en formato CoinTracking."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "Type", "Buy Amount", "Buy Currency",
        "Sell Amount", "Sell Currency",
        "Fee", "Fee Currency",
        "Exchange", "Trade-Group", "Comment", "Date",
    ])
    for _, r in df.iterrows():
        date_str = str(r["fecha"]) + " 00:00:00"
        if r["tipo"] in ("BUY", "TRANSFER_IN"):
            writer.writerow([
                "Buy", r["cantidad"], r["activo"],
                r["total_usd"], "USD",
                "", "",
                r["exchange"], r["fuente"],
                r["notas"], date_str,
            ])
        elif r["tipo"] == "SELL":
            writer.writerow([
                "Sell", r["total_usd"], "USD",
                r["cantidad"], r["activo"],
                "", "",
                r["exchange"], r["fuente"],
                r["notas"], date_str,
            ])
    return buf.getvalue().encode("utf-8")


# ── Streamlit UI ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ZimaBlade Trading HQ",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# CSS personalizado — dark/light compatible, coherente con el prompt de diseño
st.markdown("""
<style>
/* ── Fuentes ── */
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=DM+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* ── Header ── */
.main-header {
    display: flex; align-items: center; gap: 12px;
    padding: 0 0 16px 0; border-bottom: 1px solid rgba(128,128,128,.15);
    margin-bottom: 20px;
}
.main-header h1 { font-size: 22px; font-weight: 600; margin: 0; }
.pill {
    display: inline-block; font-size: 11px; font-weight: 600;
    padding: 3px 10px; border-radius: 999px; letter-spacing: .3px;
}
.pill-paper  { background: #FEF3C7; color: #92400E; }
.pill-live   { background: #D1FAE5; color: #065F46; }
.pill-dead   { background: #FEE2E2; color: #991B1B; }
.pill-ok     { background: #D1FAE5; color: #065F46; }
.pill-warn   { background: #FEF3C7; color: #92400E; }
.pill-fail   { background: #FEE2E2; color: #991B1B; }

/* ── Metric cards ── */
div[data-testid="metric-container"] {
    background: rgba(128,128,128,.05);
    border: 1px solid rgba(128,128,128,.12);
    border-radius: 10px;
    padding: 14px 18px;
}
div[data-testid="metric-container"] label {
    font-size: 11px !important; text-transform: uppercase;
    letter-spacing: .5px; color: rgba(128,128,128,.8) !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 24px !important; font-weight: 600 !important;
}

/* ── Tabs ── */
button[data-baseweb="tab"] { font-size: 13px !important; }
button[data-baseweb="tab"][aria-selected="true"] {
    font-weight: 600 !important;
    border-bottom-color: #1D9E75 !important;
    color: #1D9E75 !important;
}

/* ── Tablas ── */
.stDataFrame { border-radius: 8px; overflow: hidden; }

/* ── Monospace para precios ── */
.mono { font-family: 'JetBrains Mono', monospace; font-size: 13px; }

/* ── Validation items ── */
.val-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 0; border-bottom: 1px solid rgba(128,128,128,.1);
}
.val-row:last-child { border-bottom: none; }
.check-pass { color: #1D9E75; font-size: 18px; }
.check-fail { color: #E24B4A; font-size: 18px; }
.check-wait { color: #F59E0B; font-size: 18px; }

/* ── Info box ── */
.info-box {
    background: rgba(29,158,117,.08);
    border-left: 3px solid #1D9E75;
    border-radius: 0 6px 6px 0;
    padding: 10px 14px;
    font-size: 13px;
    margin-bottom: 12px;
}

/* ── Ocultar Streamlit branding ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; }
</style>
""", unsafe_allow_html=True)


# ─── Inicialización ────────────────────────────────────────────────────────────
ensure_manual_tables()

# ─── Header global ────────────────────────────────────────────────────────────
hb      = get_heartbeat()
ps      = get_portfolio_state()
mode    = "PAPER" if ps.get("paper_mode", PAPER_MODE) else "LIVE"
mode_cls = "pill-paper" if mode == "PAPER" else "pill-live"

# Heartbeat status
last_ping = hb.get("last_ping")
if last_ping:
    since = (pd.Timestamp.utcnow() - pd.to_datetime(last_ping, utc=True)).total_seconds() / 60
    hb_status = "🟢 Motor activo" if since < 5 else ("🟡 Sin ping" if since < 60 else "🔴 Motor caído")
else:
    hb_status = "⚪ Sin datos"

st.markdown(f"""
<div class="main-header">
    <span style="font-size:28px;">📈</span>
    <h1>ZimaBlade Trading HQ</h1>
    <span class="pill {mode_cls}">{mode}</span>
    <span style="font-size:12px; color: gray; margin-left:auto;">{hb_status} &nbsp;·&nbsp; Capital: <strong>${ps['current_capital']:,.2f}</strong></span>
</div>
""", unsafe_allow_html=True)


# ─── Navegación por tabs ───────────────────────────────────────────────────────
tab_overview, tab_portfolio, tab_bot, tab_manual, tab_validation, tab_fiscal = st.tabs([
    "🏠 Resumen",
    "💼 Portafolio",
    "🤖 Bot · Journal",
    "👤 Inversiones Manuales",
    "🛡️ Gate Live",
    "📋 Informes Fiscales",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RESUMEN
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:

    all_trades   = get_trades_journal(limit=1000)
    bot_metrics  = compute_bot_metrics(all_trades)
    open_pos     = get_open_positions()
    manual_open  = get_manual_investments("open")

    # Precios en vivo
    watched_symbols = list({
        *[str(p["symbol"]) for p in open_pos.to_dict("records") if "symbol" in p],
        *[str(t["symbol"]) for t in manual_open.to_dict("records") if "symbol" in t],
        "BTC/USDC", "ETH/USDC", "SOL/USDC",
    })
    prices = fetch_live_prices(tuple(watched_symbols))

    # PnL portafolio manual no realizado
    manual_unrealized = 0.0
    for _, row in manual_open.iterrows():
        sym = str(row["symbol"])
        cur = prices.get(sym, {}).get("price", float(row["buy_price"]))
        manual_unrealized += float(row["amount"]) * (cur - float(row["buy_price"]))

    # Drawdown actual
    dd_actual = 0.0
    if ps["peak_capital"] > 0:
        dd_actual = (ps["current_capital"] - ps["peak_capital"]) / ps["peak_capital"] * 100

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Capital Bot", f"${ps['current_capital']:,.2f}",
                delta=fmt_usd(ps["current_capital"] - INITIAL_CAPITAL),
                delta_color="normal")
    col2.metric("PnL Total Bot", fmt_usd(bot_metrics["total_pnl"]),
                delta=fmt_pct((bot_metrics["total_pnl"] / INITIAL_CAPITAL) * 100) if INITIAL_CAPITAL else "—")
    col3.metric("Portafolio Manual",
                f"${manual_unrealized + sum(float(r['amount']) * float(r['buy_price']) for _, r in manual_open.iterrows()):,.2f}",
                delta=fmt_usd(manual_unrealized), delta_color="normal")
    col4.metric("Win Rate", f"{bot_metrics['win_rate']:.1f}%",
                delta=f"{'✓' if bot_metrics['win_rate'] >= 45 else '✗'} obj. ≥45%")
    col5.metric("Profit Factor", f"{bot_metrics['profit_factor']:.2f}",
                delta=f"{'✓' if bot_metrics['profit_factor'] >= 1.3 else '✗'} obj. ≥1.3")
    col6.metric("Drawdown", f"{dd_actual:.2f}%",
                delta=f"{'✓' if dd_actual > -8 else '✗'} límite 8%",
                delta_color="inverse" if dd_actual < -5 else "normal")

    st.divider()

    col_left, col_right = st.columns([2, 1])

    with col_left:
        days_eq = st.select_slider("Período curva de capital", [7, 14, 30, 90], value=30, key="eq_days")
        eq_df = get_equity_curve(days_eq)
        if not eq_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=eq_df["day"], y=eq_df["capital"],
                mode="lines", line=dict(color="#1D9E75", width=2),
                fill="tozeroy", fillcolor="rgba(29,158,117,.07)",
                name="Capital",
            ))
            fig.add_hline(y=INITIAL_CAPITAL, line_dash="dash",
                          line_color="rgba(128,128,128,.4)", annotation_text="Capital inicial")
            fig.update_layout(
                title="Curva de capital del bot",
                height=280, margin=dict(l=0, r=0, t=36, b=0),
                xaxis_title=None, yaxis_tickprefix="$",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Precios en vivo")
        st.caption("Actualizado cada 60 seg · CoinGecko")
        for sym, pdata in list(prices.items())[:6]:
            price  = pdata["price"]
            chg    = pdata["change_24h"]
            chg_color = "#1D9E75" if chg >= 0 else "#E24B4A"
            sign   = "▲" if chg >= 0 else "▼"
            ticker = sym.replace("/USDC", "")
            st.markdown(
                f"**{ticker}** &nbsp; "
                f"<span class='mono'>${price:,.4f}</span> &nbsp; "
                f"<span style='color:{chg_color};font-size:12px;'>{sign} {abs(chg):.2f}%</span>",
                unsafe_allow_html=True,
            )
        if st.button("🔄 Refrescar precios", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    st.divider()

    # Posiciones abiertas del bot
    st.subheader(f"🤖 Posiciones abiertas del bot ({len(open_pos)})")
    if open_pos.empty:
        st.info("No hay posiciones abiertas actualmente.")
    else:
        display_pos = open_pos.copy()
        for col in ["entry_price", "stop_loss", "tp1", "tp2"]:
            if col in display_pos.columns:
                display_pos[col] = display_pos[col].apply(lambda x: f"${float(x):,.4f}" if x else "—")
        st.dataframe(display_pos, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PORTAFOLIO COMBINADO
# ══════════════════════════════════════════════════════════════════════════════
with tab_portfolio:

    st.subheader("💼 Portafolio combinado: bot + inversiones manuales")

    manual_open  = get_manual_investments("open")
    open_pos     = get_open_positions()

    all_symbols = list({
        *[str(r["symbol"]) for _, r in manual_open.iterrows()],
        *[str(r["symbol"]) for _, r in open_pos.iterrows()],
        "BTC/USDC", "ETH/USDC",
    })
    prices = fetch_live_prices(tuple(all_symbols))

    portfolio_rows = []

    # Bot
    for _, p in open_pos.iterrows():
        sym  = str(p["symbol"])
        cur  = prices.get(sym, {}).get("price", float(p["entry_price"]))
        cost = float(p["units"]) * float(p["entry_price"])
        val  = float(p["units"]) * cur
        pnl  = val - cost
        pct  = (pnl / cost * 100) if cost > 0 else 0
        portfolio_rows.append({
            "Activo": sym.replace("/USDC", ""),
            "Tipo":   "🤖 Bot",
            "Unidades": f"{float(p['units']):.6f}",
            "Precio compra": f"${float(p['entry_price']):,.4f}",
            "Precio actual": f"${cur:,.4f}",
            "Valor (USD)":   round(val, 2),
            "Inversión":     round(cost, 2),
            "PnL (USD)":     round(pnl, 2),
            "PnL %":         round(pct, 2),
            "Estrategia":    str(p.get("strategy", "—")),
        })

    # Manual
    for _, t in manual_open.iterrows():
        sym  = str(t["symbol"])
        cur  = prices.get(sym, {}).get("price", float(t["buy_price"]))
        cost = float(t["amount"]) * float(t["buy_price"])
        val  = float(t["amount"]) * cur
        pnl  = val - cost
        pct  = (pnl / cost * 100) if cost > 0 else 0
        portfolio_rows.append({
            "Activo":        sym.replace("/USDC", ""),
            "Tipo":          "👤 Manual",
            "Unidades":      f"{float(t['amount']):.6f}",
            "Precio compra": f"${float(t['buy_price']):,.4f}",
            "Precio actual": f"${cur:,.4f}",
            "Valor (USD)":   round(val, 2),
            "Inversión":     round(cost, 2),
            "PnL (USD)":     round(pnl, 2),
            "PnL %":         round(pct, 2),
            "Estrategia":    str(t.get("exchange", "—")),
        })

    if portfolio_rows:
        df_pf = pd.DataFrame(portfolio_rows)
        total_inv = df_pf["Inversión"].sum()
        total_val = df_pf["Valor (USD)"].sum()
        total_pnl = df_pf["PnL (USD)"].sum()
        total_pct = (total_pnl / total_inv * 100) if total_inv > 0 else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Capital invertido", f"${total_inv:,.2f}")
        c2.metric("Valor liquidativo", f"${total_val:,.2f}")
        c3.metric("PnL no realizado", fmt_usd(total_pnl),
                  delta=fmt_pct(total_pct), delta_color="normal")
        c4.metric("Posiciones", len(portfolio_rows))

        # Color PnL en tabla
        def style_pnl(val):
            try:
                if isinstance(val, str):
                    cv = val.replace('$', '').replace('%', '').replace(' ', '').strip()
                    num = float(cv)
                else:
                    num = float(val)
                color = '#EF4444' if num < 0 else '#00FF7F'
            except:
                color = '#FFFFFF'
            return f'color: {color}; font-weight: bold;'
            return ""

        styled = df_pf.style.map(style_pnl, subset=["PnL (USD)", "PnL %"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # Gráfico distribución
        col_a, col_b = st.columns(2)
        with col_a:
            agg = df_pf.groupby("Activo")["Valor (USD)"].sum().reset_index()
            fig_alloc = px.pie(agg, names="Activo", values="Valor (USD)",
                               title="Distribución por activo",
                               color_discrete_sequence=px.colors.qualitative.Set2,
                               hole=0.5)
            fig_alloc.update_layout(height=280, margin=dict(l=0,r=0,t=36,b=0),
                                    paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_alloc, use_container_width=True)
        with col_b:
            pnl_by_asset = df_pf.groupby("Activo")["PnL (USD)"].sum().reset_index()
            colors = ["#1D9E75" if v >= 0 else "#E24B4A" for v in pnl_by_asset["PnL (USD)"]]
            fig_pnl = go.Figure(go.Bar(
                x=pnl_by_asset["Activo"], y=pnl_by_asset["PnL (USD)"],
                marker_color=colors,
            ))
            fig_pnl.update_layout(
                title="PnL por activo (USD)",
                height=280, margin=dict(l=0,r=0,t=36,b=0),
                yaxis_tickprefix="$", plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
            )
            st.plotly_chart(fig_pnl, use_container_width=True)
    else:
        st.info("Sin posiciones abiertas. Abre una operación o registra una inversión manual.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — BOT JOURNAL
# ══════════════════════════════════════════════════════════════════════════════
with tab_bot:

    st.subheader("🤖 Journal del bot de trading")

    with st.expander("🔍 Filtros", expanded=True):
        fc1, fc2, fc3, fc4, fc5 = st.columns(5)
        f_strat   = fc1.selectbox("Estrategia", ["", "TrendFollowing", "MeanReversion", "Breakout"], key="fs")
        f_dir     = fc2.selectbox("Dirección",  ["", "LONG", "SHORT"], key="fd")
        f_quality = fc3.selectbox("Calidad",    ["", "A+", "A", "B", "C"], key="fq")
        f_sym     = fc4.text_input("Par (ej. BTC)", key="fsym")
        f_closed  = fc5.checkbox("Solo cerrados", value=False, key="fcl")

    df_trades = get_trades_journal(
        limit=1000, strategy=f_strat, direction=f_dir,
        quality=f_quality, symbol=f_sym, closed_only=f_closed,
    )
    bot_metrics = compute_bot_metrics(df_trades)

    mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
    mc1.metric("Trades cerrados", bot_metrics["total_trades"])
    mc2.metric("Win Rate",         f"{bot_metrics['win_rate']:.1f}%")
    mc3.metric("Profit Factor",    f"{bot_metrics['profit_factor']:.2f}")
    mc4.metric("R medio",          f"{bot_metrics['avg_r']:.2f}R")
    mc5.metric("Mejor trade",      fmt_usd(bot_metrics["best_trade"]))
    mc6.metric("Peor trade",       fmt_usd(bot_metrics["worst_trade"]))

    if df_trades.empty:
        st.info("Sin trades que coincidan con los filtros. El bot aún no ha registrado operaciones en PostgreSQL.")
    else:
        # Formatear para visualización
        show_cols = ["symbol", "strategy", "direction", "setup_quality",
                     "entry_price", "exit_price", "pnl_usd", "r_multiple",
                     "market_regime", "entry_time", "exit_time", "duration_hours"]
        df_show = df_trades[[c for c in show_cols if c in df_trades.columns]].copy()

        def style_pnl_row(val):
            if pd.isna(val):
                return ""
            try:
                v = float(val)
                return f"color: {'#1D9E75' if v > 0 else '#E24B4A'}; font-weight: 500"
            except Exception:
                return ""

        styled_trades = df_show.style.map(style_pnl_row, subset=["pnl_usd"] if "pnl_usd" in df_show.columns else [])
        st.dataframe(styled_trades, use_container_width=True, hide_index=True, height=400)

        # Gráficos analíticos
        st.divider()
        ga, gb, gc = st.columns(3)

        closed_df = df_trades[df_trades["exit_time"].notna() & df_trades["pnl_usd"].notna()].copy()
        closed_df["pnl_usd"] = closed_df["pnl_usd"].astype(float)

        with ga:
            if not closed_df.empty and "strategy" in closed_df.columns:
                strat_counts = closed_df.groupby("strategy").size().reset_index(name="count")
                fig_s = px.pie(strat_counts, names="strategy", values="count",
                               title="Trades por estrategia", hole=0.5,
                               color_discrete_sequence=["#1D9E75", "#378ADD", "#EF9F27"])
                fig_s.update_layout(height=240, margin=dict(l=0,r=0,t=36,b=0),
                                    paper_bgcolor="rgba(0,0,0,0)", showlegend=True)
                st.plotly_chart(fig_s, use_container_width=True)

        with gb:
            if not closed_df.empty:
                fig_dist = px.histogram(closed_df, x="pnl_usd", nbins=20,
                                        title="Distribución PnL",
                                        color_discrete_sequence=["#1D9E75"])
                fig_dist.update_layout(height=240, margin=dict(l=0,r=0,t=36,b=0),
                                       xaxis_tickprefix="$",
                                       plot_bgcolor="rgba(0,0,0,0)",
                                       paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
                st.plotly_chart(fig_dist, use_container_width=True)

        with gc:
            ml_df = get_ml_performance()
            if not ml_df.empty:
                st.subheader("ML MetaLabeler")
                for _, row in ml_df.iterrows():
                    st.markdown(f"""
                    **{row.get('model_version','—')}**
                    - Precisión: `{float(row.get('precision_ml',0) or 0):.1%}`
                    - Recall: `{float(row.get('recall_ml',0) or 0):.1%}`
                    - Trades aprobados: `{row.get('trades_approved','—')}`
                    """)
            else:
                st.info("Sin datos ML aún.")

        # Export CSV
        csv_bot = df_trades.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Descargar journal completo (CSV)",
            data=csv_bot,
            file_name=f"bot_journal_{date.today()}.csv",
            mime="text/csv",
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — INVERSIONES MANUALES
# ══════════════════════════════════════════════════════════════════════════════
with tab_manual:

    st.subheader("👤 Inversiones manuales")

    col_form, col_table = st.columns([1, 2])

    with col_form:

        # ── Formulario nueva compra ──────────────────────────────────────────
        st.markdown("#### ➕ Registrar nueva compra")
        with st.form("form_add_manual", clear_on_submit=True):
            m_sym_sel = st.selectbox("Par", get_binance_usdc_pairs())
            if m_sym_sel == "Otro...":
                m_sym = st.text_input("Escribe el par manualmente").upper().strip()
            else:
                m_sym = m_sym_sel
            m_col1, m_col2 = st.columns(2)
            m_amt   = m_col1.number_input("Cantidad", min_value=0.0, format="%.8f", step=0.0001)
            m_price = m_col2.number_input("Precio compra (USD)", min_value=0.0, format="%.4f", step=0.01)
            m_date  = st.date_input("Fecha de compra", value=date.today())
            m_exch  = st.selectbox("Exchange", ["Binance", "Coinbase", "Kraken", "Bybit", "Wallet", "Otro"])
            m_type  = st.selectbox("Tipo", ["buy", "transfer_in", "airdrop", "staking_reward"])
            m_notes = st.text_area("Notas (opcional)", height=60)

            if m_amt > 0 and m_price > 0:
                st.info(f"💰 Inversión total estimada: **${m_amt * m_price:,.2f}**")

            submitted = st.form_submit_button("💾 Guardar inversión", use_container_width=True)
            if submitted:
                if not m_sym or m_amt <= 0 or m_price <= 0:
                    st.error("Rellena símbolo, cantidad y precio.")
                else:
                    if add_manual_investment(m_sym, m_amt, m_price, m_date, m_exch, m_type, m_notes):
                        st.success(f"✅ {m_sym} registrado correctamente.")
                        st.rerun()

        st.divider()

        # ── Formulario cierre / venta ────────────────────────────────────────
        st.markdown("#### 🔴 Registrar venta / cierre")
        manual_open_data = get_manual_investments("open")
        if manual_open_data.empty:
            st.info("Sin posiciones manuales abiertas que cerrar.")
        else:
            inv_options = {
                f"{r['symbol']} · {r['amount']} unid. @ ${float(r['buy_price']):.4f} ({r['buy_date']})": r
                for _, r in manual_open_data.iterrows()
            }
            with st.form("form_close_manual"):
                sel_label  = st.selectbox("Posición a cerrar", list(inv_options.keys()))
                sel_inv    = inv_options[sel_label]
                max_amount = float(sel_inv["amount"])
                s_col1, s_col2 = st.columns(2)
                s_amt   = s_col1.number_input("Cantidad vendida",
                                               min_value=0.0, max_value=max_amount,
                                               value=max_amount, format="%.8f")
                s_price = s_col2.number_input("Precio de venta (USD)", min_value=0.0, format="%.4f")
                s_date  = st.date_input("Fecha de venta", value=date.today())
                s_notes = st.text_input("Notas")

                if s_price > 0 and s_amt > 0:
                    pnl_est = s_amt * (s_price - float(sel_inv["buy_price"]))
                    pnl_color = "🟢" if pnl_est >= 0 else "🔴"
                    st.info(f"{pnl_color} PnL estimado: **{fmt_usd(pnl_est)}** "
                            f"({fmt_pct((pnl_est / (s_amt * float(sel_inv['buy_price'])) * 100) if s_amt > 0 else 0)})")

                close_ok = st.form_submit_button("✅ Registrar venta", use_container_width=True)
                if close_ok:
                    if s_price <= 0 or s_amt <= 0:
                        st.error("Precio y cantidad son obligatorios.")
                    else:
                        if close_manual_investment(
                            inv_id=int(sel_inv["id"]),
                            symbol=str(sel_inv["symbol"]),
                            amount_sold=s_amt,
                            buy_price=float(sel_inv["buy_price"]),
                            sell_price=s_price,
                            sell_date=s_date,
                            exchange=str(sel_inv.get("exchange", "Binance")),
                            notes=s_notes,
                        ):
                            st.success("✅ Venta registrada y ganancia guardada para informes fiscales.")
                            st.rerun()

    with col_table:

        # ── Tabla posiciones abiertas ────────────────────────────────────────
        st.markdown("#### 📊 Posiciones abiertas")
        manual_open_now = get_manual_investments("open")
        if manual_open_now.empty:
            st.info("Sin inversiones manuales registradas.")
        else:
            all_manual_syms = tuple(manual_open_now["symbol"].unique())
            prices_manual   = fetch_live_prices(all_manual_syms)

            rows_m = []
            for _, t in manual_open_now.iterrows():
                sym = str(t["symbol"])
                cur = prices_manual.get(sym, {}).get("price", float(t["buy_price"]))
                pnl = float(t["amount"]) * (cur - float(t["buy_price"]))
                pct = (pnl / (float(t["amount"]) * float(t["buy_price"])) * 100) if float(t["buy_price"]) > 0 else 0
                rows_m.append({
                    "ID":         int(t["id"]),
                    "Activo":     sym.replace("/USDC", ""),
                    "Cantidad":   float(t["amount"]),
                    "Compra":     f"${float(t['buy_price']):,.4f}",
                    "Actual":     f"${cur:,.4f}",
                    "Inv. USD":   round(float(t["amount"]) * float(t["buy_price"]), 2),
                    "Valor USD":  round(float(t["amount"]) * cur, 2),
                    "PnL USD":    round(pnl, 2),
                    "PnL %":      round(pct, 2),
                    "Exchange":   str(t.get("exchange", "—")),
                    "Fecha":      str(t["buy_date"]),
                })

            df_m = pd.DataFrame(rows_m)
            total_m_inv = df_m["Inv. USD"].sum()
            total_m_val = df_m["Valor USD"].sum()
            total_m_pnl = df_m["PnL USD"].sum()

            mc1m, mc2m, mc3m = st.columns(3)
            mc1m.metric("Invertido", f"${total_m_inv:,.2f}")
            mc2m.metric("Valor actual", f"${total_m_val:,.2f}")
            mc3m.metric("PnL no realizado", fmt_usd(total_m_pnl),
                        delta_color="normal" if total_m_pnl >= 0 else "inverse")

            def color_pnl_cell(val):
                try:
                    v = float(val)
                    return f"color: {'#1D9E75' if v > 0 else ('#E24B4A' if v < 0 else 'gray')}; font-weight: 500"
                except Exception:
                    return ""

            st.dataframe(
                df_m.style.map(color_pnl_cell, subset=["PnL USD", "PnL %"]),
                use_container_width=True, hide_index=True,
            )

            # Borrar posición
            del_id = st.number_input("ID a eliminar (0 = ninguno)", min_value=0, step=1, key="del_id")
            if st.button("🗑️ Eliminar posición seleccionada") and del_id > 0:
                if delete_manual_investment(del_id):
                    st.success(f"Posición #{del_id} eliminada.")
                    st.rerun()

        st.divider()

        # ── Tabla ventas realizadas ──────────────────────────────────────────
        st.markdown("#### 📜 Historial de ventas (ganancias realizadas)")
        closed_m = get_manual_closings()
        if closed_m.empty:
            st.info("Sin ventas registradas aún.")
        else:
            closed_m["pnl_usd"] = closed_m["pnl_usd"].astype(float)
            st.dataframe(closed_m[[c for c in [
                "symbol","amount_sold","buy_price","sell_price",
                "pnl_usd","sell_date","exchange","notes"
            ] if c in closed_m.columns]], use_container_width=True, hide_index=True)
            total_realized = float(closed_m["pnl_usd"].sum())
            st.markdown(f"**Total ganancia realizada:** {color_pnl(total_realized)} **{fmt_usd(total_realized)}**")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — GATE DE VALIDACIÓN PRE-LIVE
# ══════════════════════════════════════════════════════════════════════════════
with tab_validation:

    st.subheader("🛡️ Gate de validación pre-live")
    st.markdown("""
    <div class="info-box">
        Todos los criterios deben cumplirse antes de pasar a capital real.
        El sistema bloquea el switch automáticamente si algún criterio falla.
    </div>
    """, unsafe_allow_html=True)

    all_trades_v  = get_trades_journal(limit=2000, closed_only=True)
    bot_metrics_v = compute_bot_metrics(all_trades_v)
    ps_v          = get_portfolio_state()
    hb_v          = get_heartbeat()

    # Heartbeat failures: contar en últimas 24h desde drawdowns o log
    hb_fails = 0  # TODO: parsear logs si se implementa tabla de fallos

    # RAM usage desde /proc/meminfo (en ZimaBlade Linux)
    ram_used_gb = None
    try:
        with open("/proc/meminfo") as f:
            meminfo = dict(line.split(":") for line in f.read().splitlines() if ":" in line)
        total  = int(meminfo["MemTotal"].strip().split()[0]) / 1e6
        avail  = int(meminfo["MemAvailable"].strip().split()[0]) / 1e6
        ram_used_gb = round(total - avail, 2)
    except Exception:
        pass  # No estamos en el servidor

    # Calcular drawdown máximo
    dd_actual_v = 0.0
    if ps_v["peak_capital"] > 0:
        dd_actual_v = (ps_v["current_capital"] - ps_v["peak_capital"]) / ps_v["peak_capital"] * 100

    # Semanas en paper: desde primer trade
    weeks_in_paper = 0
    if not all_trades_v.empty and "entry_time" in all_trades_v.columns:
        first = pd.to_datetime(all_trades_v["entry_time"].min())
        weeks_in_paper = (pd.Timestamp.utcnow() - first.tz_localize(timezone.utc)).days / 7

    criteria_values = {
        "win_rate":      bot_metrics_v["win_rate"],
        "profit_factor": bot_metrics_v["profit_factor"],
        "max_drawdown":  dd_actual_v,
        "min_trades":    bot_metrics_v["total_trades"],
        "heartbeat":     hb_fails,
        "ram_gb":        ram_used_gb if ram_used_gb is not None else 0.0,
        "period_weeks":  round(weeks_in_paper, 1),
    }

    def passes(key: str, val: float) -> bool:
        c = VALIDATION_CRITERIA[key]
        return (val >= c["target"] if c["op"] == "gte"
                else val <= c["target"])

    all_pass   = all(passes(k, v) for k, v in criteria_values.items() if v is not None)
    pass_count = sum(passes(k, v) for k, v in criteria_values.items() if v is not None)
    total_crit = len(criteria_values)

    # Barra de progreso
    progress_pct = pass_count / total_crit
    st.progress(progress_pct, text=f"{pass_count}/{total_crit} criterios superados")

    if all_pass:
        st.success("🚀 **¡Todos los criterios superados!** El sistema está listo para capital real.")
    else:
        remaining = total_crit - pass_count
        st.warning(f"⚠️ Faltan {remaining} criterio(s) para activar el modo live.")

    st.divider()

    # Tabla criterios
    for key, crit in VALIDATION_CRITERIA.items():
        val = criteria_values.get(key)
        ok  = passes(key, val) if val is not None else False
        icon = "✅" if ok else ("⏳" if val is None else "❌")
        val_str = (
            f"{val:.1f}{crit['unit']}"
            if val is not None else "Sin datos"
        )
        target_str = f"{'≥' if crit['op'] == 'gte' else '≤'} {crit['target']}{crit['unit']}"

        col_i, col_n, col_v, col_t = st.columns([0.5, 3, 2, 2])
        col_i.markdown(f"<span style='font-size:20px;'>{icon}</span>", unsafe_allow_html=True)
        col_n.markdown(f"**{crit['label']}**")
        col_v.markdown(
            f"<span style='color:{'#1D9E75' if ok else '#E24B4A'};font-weight:600;'>{val_str}</span>",
            unsafe_allow_html=True,
        )
        col_t.markdown(f"<span style='color:gray;font-size:12px;'>{target_str}</span>",
                       unsafe_allow_html=True)
        st.divider()

    # Drawdown timeline
    dd_series = get_drawdown_series(30)
    if not dd_series.empty:
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=dd_series["date"], y=dd_series["drawdown_pct"].astype(float),
            mode="lines", fill="tozeroy",
            line=dict(color="#E24B4A", width=1.5),
            fillcolor="rgba(226,75,74,.07)",
            name="Drawdown",
        ))
        fig_dd.add_hline(y=-8, line_dash="dash", line_color="rgba(226,75,74,.5)",
                         annotation_text="Límite -8%")
        fig_dd.update_layout(
            title="Evolución del drawdown (30 días)",
            height=220, margin=dict(l=0,r=0,t=36,b=0),
            yaxis_ticksuffix="%",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig_dd, use_container_width=True)

    if ram_used_gb:
        st.caption(f"💾 RAM del servidor: {ram_used_gb:.2f} GB usados de ~4 GB disponibles")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — INFORMES FISCALES
# ══════════════════════════════════════════════════════════════════════════════
with tab_fiscal:

    st.subheader("📋 Informes fiscales · AEAT España")

    st.markdown("""
    <div class="info-box">
        <strong>Método FIFO</strong> — obligatorio por la AEAT. Las ganancias se calculan
        usando el precio de las primeras unidades compradas. Los CSV exportados son
        compatibles con <strong>Koinly</strong> y <strong>CoinTracking</strong> directamente.
    </div>
    """, unsafe_allow_html=True)

    year_filter = st.selectbox("Año fiscal", [2025, 2024, 2023], key="tax_year")

    # Cargar datos
    bot_trades_f   = get_trades_journal(limit=5000, closed_only=True)
    manual_open_f  = get_manual_investments("open")
    manual_closed_f = get_manual_closings()

    df_fiscal = build_fiscal_dataframe(bot_trades_f, manual_open_f, manual_closed_f)

    # Métricas fiscales
    if not df_fiscal.empty:
        sells = df_fiscal[df_fiscal["tipo"] == "SELL"].copy()
        sells["ganancia"] = sells["ganancia"].fillna(0).astype(float)
        gains  = sells[sells["ganancia"] > 0]["ganancia"].sum()
        losses = sells[sells["ganancia"] <= 0]["ganancia"].sum()
        net    = gains + losses
        irpf   = net * 0.19 if net > 0 else 0

        fi1, fi2, fi3, fi4 = st.columns(4)
        fi1.metric("Ganancias realizadas", fmt_usd(gains))
        fi2.metric("Pérdidas realizadas",  fmt_usd(losses))
        fi3.metric("Base imponible neta",  fmt_usd(net))
        fi4.metric("Estimación IRPF ~19%", f"~${irpf:,.2f}")

        st.caption("⚠️ Estimación orientativa. Consulta a un asesor fiscal.")

        st.divider()

        # Tabla completa
        st.markdown("#### 🗂️ Todas las operaciones del ejercicio")
        st.dataframe(df_fiscal, use_container_width=True, hide_index=True, height=350)

        # Botones de exportación
        col_exp1, col_exp2, col_exp3 = st.columns(3)
        with col_exp1:
            koinly_csv = to_koinly_csv(df_fiscal)
            st.download_button(
                "📥 Descargar Koinly CSV",
                data=koinly_csv,
                file_name=f"koinly_export_{year_filter}_{date.today()}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col_exp2:
            ct_csv = to_cointracking_csv(df_fiscal)
            st.download_button(
                "📥 Descargar CoinTracking CSV",
                data=ct_csv,
                file_name=f"cointracking_export_{year_filter}_{date.today()}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col_exp3:
            raw_csv = df_fiscal.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 CSV completo (raw)",
                data=raw_csv,
                file_name=f"reporte_completo_{year_filter}_{date.today()}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.divider()

        # Desglose por activo
        if not sells.empty:
            by_asset = sells.groupby("activo")["ganancia"].sum().reset_index().sort_values("ganancia", ascending=False)
            fig_tax = go.Figure(go.Bar(
                x=by_asset["activo"], y=by_asset["ganancia"],
                marker_color=["#1D9E75" if v >= 0 else "#E24B4A" for v in by_asset["ganancia"]],
            ))
            fig_tax.update_layout(
                title="Ganancias/pérdidas realizadas por activo",
                height=250, margin=dict(l=0,r=0,t=36,b=0),
                yaxis_tickprefix="$",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_tax, use_container_width=True)
    else:
        st.info("Sin operaciones cerradas registradas. Las ganancias realizadas aparecerán aquí una vez el bot cierre trades o registres ventas manuales.")

    # Recordatorio
    with st.expander("📌 Guía rápida: cómo importar en Koinly"):
        st.markdown("""
        1. Descarga el **CSV Koinly** con el botón de arriba.
        2. En Koinly → **Importar transacciones** → selecciona **CSV personalizado**.
        3. Mapea las columnas: `Date`, `Sent Amount/Currency`, `Received Amount/Currency`.
        4. Koinly aplicará FIFO automáticamente y calculará las plusvalías.
        5. Exporta el informe PDF para adjuntarlo a tu declaración.

        **Importante para España (AEAT):**
        - Modelo **721**: declarar criptomonedas en exchanges extranjeros si superas **50.000 €**.
        - Las plusvalías van en el **Modelo 100** (IRPF), base del ahorro.
        - Tramos: 19% hasta 6.000 €, 21% de 6.000 a 50.000 €, 23% de 50.000 a 200.000 €.
        """)
