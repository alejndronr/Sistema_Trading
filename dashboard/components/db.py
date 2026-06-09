"""
dashboard/components/db.py
Conexiones cacheadas a PostgreSQL (datos de trading) y SQLite (OHLCV).
"""
import os
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    import psycopg2
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

# ── Rutas ──────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent.parent
SQLITE_PATH = ROOT_DIR / "data" / "db" / "trading.db"


# ── PostgreSQL ─────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_pg_connection():
    """Conexión persistente a PostgreSQL (cached_resource = 1 instancia total)."""
    if not HAS_PSYCOPG2:
        return None
    try:
        conn = psycopg2.connect(
            host=os.environ.get("DB_HOST", "127.0.0.1"),
            port=int(os.environ.get("DB_PORT", 5432)),
            database=os.environ.get("DB_NAME", "trading_db"),
            user=os.environ.get("DB_USER", "trading"),
            password=os.environ.get("DB_PASSWORD", ""),
            connect_timeout=5,
        )
        return conn
    except Exception as e:
        return None


@st.cache_data(ttl=30, show_spinner=False)
def query_pg(sql: str, params=None) -> pd.DataFrame:
    """Ejecuta una query en PostgreSQL y devuelve DataFrame (TTL=30s)."""
    conn = get_pg_connection()
    if conn is None:
        return pd.DataFrame()
    try:
        if conn.closed:
            # Reconectar si se cerró
            st.cache_resource.clear()
            conn = get_pg_connection()
        return pd.read_sql(sql, conn, params=params)
    except Exception as e:
        st.warning(f"⚠️ PostgreSQL query error: {e}")
        return pd.DataFrame()


def execute_pg(sql: str, params=None) -> None:
    """Ejecuta una query en PostgreSQL sin cachear (para INSERT/UPDATE/DELETE)."""
    conn = get_pg_connection()
    if conn is None:
        return
    try:
        if conn.closed:
            st.cache_resource.clear()
            conn = get_pg_connection()
        with conn.cursor() as cur:
            cur.execute(sql, params)
        conn.commit()
    except Exception as e:
        st.warning(f"⚠️ PostgreSQL execute error: {e}")
        if conn and not conn.closed:
            conn.rollback()


def pg_available() -> bool:
    """True si PostgreSQL está disponible."""
    conn = get_pg_connection()
    return conn is not None and not conn.closed


# ── SQLite OHLCV ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def query_sqlite(sql: str, params=None) -> pd.DataFrame:
    """Ejecuta query en SQLite (OHLCV). TTL=5min (datos menos volátiles)."""
    if not SQLITE_PATH.exists():
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(str(SQLITE_PATH), timeout=10)
        df = pd.read_sql(sql, conn, params=params)
        conn.close()
        return df
    except Exception as e:
        return pd.DataFrame()


# ── Queries predefinidas ───────────────────────────────────────────────────────

@st.cache_data(ttl=30, show_spinner=False)
def get_heartbeat() -> dict:
    df = query_pg(
        "SELECT last_ping, engine_version, paper_mode, active_positions, pnl_today, regimes_json, cycles_json FROM system_heartbeat WHERE id = 1"
    )
    if df.empty:
        return {}
    return df.iloc[0].to_dict()


@st.cache_data(ttl=30, show_spinner=False)
def get_portfolio_state() -> dict:
    df = query_pg(
        "SELECT current_capital, peak_capital, daily_start FROM portfolio_state WHERE id = 1"
    )
    if df.empty:
        return {}
    return df.iloc[0].to_dict()


@st.cache_data(ttl=15, show_spinner=False)
def get_open_positions() -> pd.DataFrame:
    return query_pg(
        "SELECT * FROM positions WHERE status = 'open' ORDER BY id DESC"
    )

@st.cache_data(ttl=15, show_spinner=False)
def get_suspended_symbols() -> pd.DataFrame:
    return query_pg(
        "SELECT symbol FROM positions WHERE status = 'suspended'"
    )

@st.cache_data(ttl=30, show_spinner=False)
def get_cycle_states() -> pd.DataFrame:
    return query_pg(
        "SELECT * FROM cycle_state ORDER BY symbol"
    )


@st.cache_data(ttl=30, show_spinner=False)
def get_trades(limit: int = 1000, real_only: bool = True) -> pd.DataFrame:
    where = "WHERE is_backtest = FALSE OR is_backtest IS NULL" if real_only else ""
    sql = f"""
        SELECT
            entry_time, exit_time, symbol, strategy, direction, setup_quality,
            COALESCE(pnl, pnl_usd) AS pnl,
            COALESCE(units, position_size) AS units,
            COALESCE(regime, market_regime) AS regime,
            COALESCE(tp1, take_profit_1) AS tp1,
            r_multiple, ml_proba, exit_reason, duration_hours,
            entry_reason, observations, entry_price, stop_loss
        FROM trades_journal
        {where}
        ORDER BY entry_time DESC
        LIMIT {limit}
    """
    df = query_pg(sql)
    if not df.empty:
        for col in ["entry_time", "exit_time"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
        if "pnl" in df.columns:
            df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce").fillna(0)
    return df


@st.cache_data(ttl=60, show_spinner=False)
def get_ml_retrain_log() -> pd.DataFrame:
    return query_pg(
        "SELECT * FROM ml_retrain_log ORDER BY retrain_date DESC LIMIT 50"
    )


@st.cache_data(ttl=10, show_spinner=False)
def get_current_prices_cached(symbols: list) -> dict:
    """Intenta conseguir los precios por CCXT; si falla, usa el último de PostgreSQL cacheado"""
    try:
        import ccxt
        exchange = ccxt.binance({"options": {"defaultType": "spot"}})
        prices = {}
        # Para evitar spam de requests si son muchos, en el dashboard
        # típicamente se usará para los que están en open_positions o ciclo.
        for sym in symbols:
            try:
                t = exchange.fetch_ticker(sym)
                prices[sym] = float(t["last"])
            except Exception:
                pass
        return prices
    except Exception:
        return {}


@st.cache_data(ttl=300, show_spinner=False)
def get_ohlcv_summary() -> pd.DataFrame:
    return query_sqlite(
        """
        SELECT symbol, timeframe,
               COUNT(*) as candles,
               MIN(timestamp) as from_ms,
               MAX(timestamp) as to_ms
        FROM ohlcv
        GROUP BY symbol, timeframe
        ORDER BY symbol, timeframe
        """
    )


@st.cache_data(ttl=120, show_spinner=False)
def get_equity_curve() -> pd.DataFrame:
    """Construye la curva de equity a partir de trades cerrados."""
    df = get_trades(limit=5000, real_only=False)
    if df.empty or "pnl" not in df.columns:
        return pd.DataFrame()
    df = df.sort_values("entry_time")
    df["cumulative_pnl"] = df["pnl"].cumsum()
    # Capital base (intentar desde portfolio_state)
    ps = get_portfolio_state()
    initial = ps.get("current_capital", 1000.0) - df["cumulative_pnl"].iloc[-1] if ps else 1000.0
    df["equity"] = initial + df["cumulative_pnl"]
    df["peak"] = df["equity"].cummax()
    df["drawdown"] = (df["equity"] - df["peak"]) / df["peak"]
    return df[["entry_time", "exit_time", "equity", "peak", "drawdown", "pnl"]].copy()
