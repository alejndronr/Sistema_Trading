"""
diagnose_bot.py — Diagnóstico completo del motor de trading
============================================================
Ejecutar en el servidor para identificar por qué el bot no opera:

    source venv/bin/activate
    python diagnose_bot.py

Analiza:
  1. Estado del motor (heartbeat, pausa, kill)
  2. Capital y circuit breakers
  3. Por qué las estrategias no generan señales
  4. Estado del cooldown por símbolo
  5. Si hay posiciones abiertas que bloquean nuevas entradas
  6. Calidad de las señales de los últimos 3 días
"""

import os
import sys
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

# ── Conexión BD ────────────────────────────────────────────────────────────────
raw = os.environ.get("DATABASE_URL", "")
db_url = (raw.replace("+asyncpg", "")
             .replace("localhost", "127.0.0.1"))
engine = create_engine(db_url, connect_args={"connect_timeout": 5})

def qry(sql, params=None):
    with engine.connect() as c:
        r = c.execute(text(sql), params or {})
        return pd.DataFrame(r.fetchall(), columns=list(r.keys()))

INITIAL_CAPITAL = float(os.environ.get("INITIAL_CAPITAL", "1000"))

print("═" * 60)
print("  🔍 Diagnóstico del Motor de Trading")
print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("═" * 60)

# ══════════════════════════════════════════════════════════════════════
# 1. HEARTBEAT — ¿Está vivo el motor?
# ══════════════════════════════════════════════════════════════════════
print("\n[1/6] Estado del motor (heartbeat)")
try:
    hb = qry("SELECT last_ping, engine_version, paper_mode FROM system_heartbeat WHERE id=1")
    if hb.empty:
        print("  ❌ Sin heartbeat — el motor no está corriendo o nunca arrancó")
    else:
        last_ping = pd.to_datetime(hb["last_ping"].iloc[0], utc=True)
        since_min = (datetime.now(timezone.utc) - last_ping).total_seconds() / 60
        eng_ver   = hb["engine_version"].iloc[0]
        paper     = hb["paper_mode"].iloc[0]
        icon = "✅" if since_min < 5 else ("⚠️" if since_min < 60 else "❌")
        print(f"  {icon} Último ping: hace {since_min:.0f} minutos")
        print(f"     Motor: {eng_ver} | Modo: {'PAPER' if paper else '⚠️ LIVE'}")
        if since_min > 10:
            print(f"  ⚠️  ALERTA: Motor sin responder desde {since_min:.0f} min")
            print("     → sudo systemctl restart trading-engine")
except Exception as e:
    print(f"  ❌ Error consultando heartbeat: {e}")

# ══════════════════════════════════════════════════════════════════════
# 2. CAPITAL Y CIRCUIT BREAKERS
# ══════════════════════════════════════════════════════════════════════
print("\n[2/6] Capital y circuit breakers")
try:
    ps = qry("SELECT current_capital, peak_capital, daily_start FROM portfolio_state WHERE id=1")
    if ps.empty:
        print("  ⚠️  Sin portfolio_state")
    else:
        cap        = float(ps["current_capital"].iloc[0])
        peak       = float(ps["peak_capital"].iloc[0])
        daily_st   = float(ps["daily_start"].iloc[0]) if ps["daily_start"].iloc[0] else cap
        dd_pct     = (cap - peak) / peak * 100 if peak > 0 else 0
        daily_pnl  = (cap - daily_st) / daily_st * 100 if daily_st > 0 else 0

        print(f"  Capital actual:  ${cap:,.2f}")
        print(f"  Capital inicial: ${INITIAL_CAPITAL:,.2f}")
        print(f"  PnL total:       ${cap - INITIAL_CAPITAL:,.2f} ({(cap-INITIAL_CAPITAL)/INITIAL_CAPITAL*100:.1f}%)")
        print(f"  Drawdown pico:   {dd_pct:.2f}%")
        print(f"  PnL hoy:         {daily_pnl:.2f}%")

        problems = []
        if daily_pnl <= -5.0:
            problems.append(f"❌ CB_DAILY_PAUSE activo ({daily_pnl:.1f}% ≤ -5%) — motor NO entra trades")
        elif daily_pnl <= -3.0:
            problems.append(f"⚠️  CB_DAILY_REDUCE activo ({daily_pnl:.1f}% ≤ -3%) — sizing al 50%")
        if dd_pct <= -8.0:
            problems.append(f"❌ CB_PEAK_SHUTDOWN activo ({dd_pct:.1f}% ≤ -8%) — motor APAGADO")

        if problems:
            for p in problems:
                print(f"  {p}")
        else:
            print("  ✅ Sin circuit breakers activos")
except Exception as e:
    print(f"  ❌ Error: {e}")

# ══════════════════════════════════════════════════════════════════════
# 3. POSICIONES ABIERTAS — ¿Hay MAX_POSITIONS?
# ══════════════════════════════════════════════════════════════════════
print("\n[3/6] Posiciones abiertas")
try:
    pos = qry("SELECT symbol, strategy, entry_price, entry_time, units FROM positions WHERE status='open'")
    if pos.empty:
        print("  ✅ Sin posiciones abiertas (no bloquea entradas)")
    else:
        print(f"  {len(pos)} posición(es) abierta(s):")
        for _, p in pos.iterrows():
            et = pd.to_datetime(p["entry_time"], utc=True)
            hrs = (datetime.now(timezone.utc) - et).total_seconds() / 3600
            print(f"    • {p['symbol']} | {p['strategy']} | {hrs:.0f}h abierta")
        if len(pos) >= 3:
            print("  ⚠️  MAX_POSITIONS (3) alcanzado — motor NO puede abrir más")
except Exception as e:
    print(f"  ❌ Error: {e}")

# ══════════════════════════════════════════════════════════════════════
# 4. ÚLTIMOS TRADES — ¿Cuándo fue el último trade?
# ══════════════════════════════════════════════════════════════════════
print("\n[4/6] Historial de trades recientes")
try:
    recent = qry("""
        SELECT symbol, strategy, entry_time, exit_time, pnl,
               exit_reason, is_backtest
        FROM trades_journal
        WHERE is_backtest = FALSE OR is_backtest IS NULL
        ORDER BY entry_time DESC
        LIMIT 10
    """)
    if recent.empty:
        print("  ⚠️  Sin trades reales en la BD")
    else:
        last_real = pd.to_datetime(recent["entry_time"].iloc[0], utc=True)
        days_ago  = (datetime.now(timezone.utc) - last_real).days
        print(f"  Último trade real: hace {days_ago} día(s)")
        for _, t in recent.iterrows():
            et  = pd.to_datetime(t["entry_time"], utc=True).strftime("%m/%d %H:%M")
            pnl = float(t["pnl"]) if t["pnl"] is not None else 0.0
            icon = "✅" if pnl > 0 else "❌"
            print(f"    {icon} {et} | {t['symbol']} | {t['strategy'][:20]} | ${pnl:+.2f}")
except Exception as e:
    print(f"  ❌ Error: {e}")

# ══════════════════════════════════════════════════════════════════════
# 5. ANÁLISIS DE SEÑALES EN VIVO — ¿Por qué no hay señales?
# ══════════════════════════════════════════════════════════════════════
print("\n[5/6] Análisis de señales en vivo (diagnóstico de estrategias)")

try:
    # Importar el motor V5 o V4
    try:
        from live_engine_v5 import (
            enrich_dataframe, detect_regime,
            strategy_trend_following_v5 as stf,
            strategy_mean_reversion_v5  as smr,
            strategy_breakout_v5        as sbo,
            build_v5_context,
        )
        engine_name = "V5"
    except ImportError:
        from live_engine import (
            enrich_dataframe, detect_regime,
            strategy_trend_following as stf,
            strategy_mean_reversion  as smr,
            strategy_breakout        as sbo,
        )
        engine_name = "V4"
        build_v5_context = None

    print(f"  Motor: {engine_name}")

    # Cargar OHLCV desde SQLite
    db_path = ROOT / "data" / "db" / "trading.db"
    conn    = sqlite3.connect(str(db_path))

    symbols = ["BTC/USDC", "ETH/USDC", "SOL/USDC", "BNB/USDC", "LINK/USDC", "AVAX/USDC"]

    for sym in symbols:
        df_raw = pd.read_sql(
            "SELECT timestamp,open,high,low,close,volume FROM ohlcv "
            f"WHERE symbol=? AND timeframe='1h' ORDER BY timestamp DESC LIMIT 300",
            conn, params=(sym,)
        )
        if len(df_raw) < 220:
            print(f"  ⚠️  {sym}: datos insuficientes ({len(df_raw)} velas)")
            continue

        df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], unit="ms", utc=True)
        df_raw = df_raw.sort_values("timestamp").reset_index(drop=True)

        try:
            df = enrich_dataframe(df_raw)
        except Exception as e:
            print(f"  ❌ {sym}: error en indicadores — {e}")
            continue

        regime = detect_regime(df, sym)
        last   = df.iloc[-1]

        # Scores individuales
        rsi  = float(last.get("rsi", 50))
        adx  = float(last.get("adx", 20))
        trend_up = bool(last.get("trend_up", False))

        # Intentar generar señal
        sig_tf  = stf(df, regime)
        sig_mr  = smr(df, regime)
        sig_bo  = sbo(df, regime)

        tf_score = sig_tf.score if sig_tf else "—"
        mr_score = sig_mr.score if sig_mr else "—"
        bo_score = sig_bo.score if sig_bo else "—"

        # Contexto V5 si disponible
        v5_info = ""
        if build_v5_context and engine_name == "V5":
            ctx = build_v5_context(df, "long")
            v5_info = (f" | cp_bull={ctx.candle_bull_score:.0f}"
                       f" obv_acc={ctx.obv_accelerating}"
                       f" cvd_pos={ctx.cvd_positive}"
                       f" sr={ctx.sr_at_zone}")

        best_any = sig_tf or sig_mr or sig_bo
        icon = "🟢" if best_any else "⭕"
        print(f"  {icon} {sym:<12} | {regime.regime:<8} "
              f"| ADX={adx:.0f} RSI={rsi:.0f} trend={'↑' if trend_up else '↓'} "
              f"| TF={tf_score} MR={mr_score} BO={bo_score}"
              f"{v5_info}")

    conn.close()

except Exception as e:
    print(f"  ❌ Error en análisis de señales: {e}")
    import traceback; traceback.print_exc()

# ══════════════════════════════════════════════════════════════════════
# 6. DIAGNÓSTICO DE LOGS — últimos mensajes relevantes
# ══════════════════════════════════════════════════════════════════════
print("\n[6/6] Últimas entradas del journal de systemd")
print("  Ejecuta manualmente para ver logs completos:")
print("  journalctl -u trading-engine --since '4 days ago' | grep -E")
print("  'signal_queued|trade_opening|circuit_breaker|paused|engine_v5_live|analysis_cycle'")
print()
print("  Comandos de diagnóstico rápido:")
print("  # ¿El motor está analizando?")
print("  journalctl -u trading-engine -f | grep 'analysis_cycle\\|signal'")
print()
print("  # ¿Hay errores?")
print("  journalctl -u trading-engine --since '1 hour ago' | grep -i 'error\\|exception\\|critical'")
print()
print("  # ¿El motor está pausado desde Telegram?")
print("  journalctl -u trading-engine --since '4 days ago' | grep -i 'paused\\|pause\\|resumed'")

print("\n" + "═" * 60)
print("  ✅ Diagnóstico completado")
print("═" * 60)
