"""
scripts/backtest_v5.py — Backtester Multi-Año con CycleDetector
===============================================================
Backtesta el motor V6 sobre todo el histórico disponible.
A diferencia del backtest anterior, aquí el CycleDetector
determina en cada vela qué estrategias están activas y con
qué multiplicador de riesgo.

Uso:
    python scripts/backtest_v5.py                    # todo el histórico
    python scripts/backtest_v5.py --months 12        # último año
    python scripts/backtest_v5.py --dry-run          # sin guardar en BD
    python scripts/backtest_v5.py --retrain          # reentrenar ML al terminar
    python scripts/backtest_v5.py --report-by-phase  # breakdown por fase
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

# Importar motor V6
try:
    from live_engine_v6 import (
        enrich_dataframe, detect_regime,
        AdaptiveSignalSelector, compute_position_size,
        SignalResult, MarketRegime,
        SYMBOLS, INITIAL_CAPITAL, COMMISSION_RATE, SLIPPAGE,
        MAX_CORR_POS, CORRELATION_GROUPS, COOLDOWN_SL_MIN,
    )
    print("✅ Lógica V6 importada desde live_engine_v6.py")
except ImportError as e:
    print(f"✗ No se pudo importar live_engine_v6.py: {e}")
    sys.exit(1)

try:
    from cycle_detector import CycleDetector, CycleState, load_daily_ohlcv
    CYCLE_OK = True
    print("✅ CycleDetector importado")
except ImportError:
    CYCLE_OK = False
    print("⚠️  CycleDetector no disponible — backtest sin ciclos")

SQLITE_PATH = str(PROJECT_ROOT / "data" / "db" / "trading.db")
BACKTEST_WEIGHT = 0.3   # peso de trades de backtest en el retrain


# ══════════════════════════════════════════════════════════════════════════════
# ── Carga de datos ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def load_ohlcv(symbol: str, timeframe: str, months: Optional[int] = None) -> pd.DataFrame:
    conn = sqlite3.connect(SQLITE_PATH)
    query = ("SELECT timestamp,open,high,low,close,volume FROM ohlcv "
             f"WHERE symbol=? AND timeframe=? ORDER BY timestamp ASC")
    df = pd.read_sql_query(query, conn, params=(symbol, timeframe))
    conn.close()
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    if months:
        cutoff = datetime.now(timezone.utc) - timedelta(days=months*30)
        df = df[df["timestamp"] >= cutoff]
    return df.reset_index(drop=True)


def load_daily_snapshot(symbol: str, as_of_date: pd.Timestamp) -> pd.DataFrame:
    """
    Carga velas diarias hasta una fecha específica (sin lookahead).
    La columna timestamp en SQLite es bigint (ms Unix).
    """
    conn = sqlite3.connect(SQLITE_PATH)
    # Convertir fecha a ms Unix para comparar con bigint de la BD
    as_of_ms = int(as_of_date.timestamp() * 1000)
    df = pd.read_sql_query(
        "SELECT timestamp,open,high,low,close,volume FROM ohlcv "
        "WHERE symbol=? AND timeframe='1d' "
        "AND timestamp <= ? ORDER BY timestamp ASC",
        conn,
        params=(symbol, as_of_ms)
    )
    conn.close()
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# ── Portfolio de backtesting ───────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

class BacktestPosition:
    def __init__(self, trade_id, symbol, strategy, entry_idx, entry_price,
                 stop_loss, tp1, tp2, units, risk, notional, entry_time,
                 regime, cycle_phase, ml_proba, entry_reason):
        self.trade_id    = trade_id
        self.symbol      = symbol
        self.strategy    = strategy
        self.entry_idx   = entry_idx
        self.entry_price = entry_price
        self.current_sl  = stop_loss
        self.stop_loss   = stop_loss
        self.tp1         = tp1
        self.tp2         = tp2
        self.units       = units
        self.remaining   = units
        self.risk        = risk
        self.notional    = notional
        self.entry_time  = entry_time
        self.regime      = regime
        self.cycle_phase = cycle_phase
        self.ml_proba    = ml_proba
        self.entry_reason= entry_reason
        self.tp1_hit     = False


class BacktestPortfolio:
    def __init__(self, capital: float):
        self.initial   = capital
        self.capital   = capital
        self.peak      = capital
        self.open:     Dict[str, BacktestPosition] = {}
        self.closed:   List[Dict]                  = []
        self.equity:   List[float]                 = []
        self.daily_pnl: Dict[str, float]           = {}

    def open_position(self, pos: BacktestPosition) -> None:
        commission = pos.entry_price * pos.units * COMMISSION_RATE
        self.capital -= commission
        self.open[pos.trade_id] = pos

    def update(self, timestamp: pd.Timestamp,
               prices: Dict[str, Dict]) -> List[Dict]:
        closed_now = []
        for tid in list(self.open.keys()):
            pos  = self.open[tid]
            sym  = pos.symbol
            info = prices.get(sym, {})
            high = info.get("high", info.get("close", pos.entry_price))
            low  = info.get("low",  info.get("close", pos.entry_price))

            # Stop Loss
            if low <= pos.current_sl:
                ct = self._close(tid, pos.current_sl, timestamp, "stop_loss", 1.0)
                if ct: closed_now.append(ct)
                continue

            # TP1 parcial (50%)
            if not pos.tp1_hit and high >= pos.tp1:
                ct = self._close(tid, pos.tp1, timestamp, "tp1_partial", 0.5)
                if ct: closed_now.append(ct)
                pos.tp1_hit = True
                pos.current_sl = pos.entry_price * 1.001  # breakeven

            # TP2 completo
            if pos.tp1_hit and high >= pos.tp2:
                ct = self._close(tid, pos.tp2, timestamp, "tp2", 1.0)
                if ct: closed_now.append(ct)
                continue

            # Trailing stop tras TP1
            if pos.tp1_hit:
                trail = pos.entry_price * 0.015
                new_sl = high - trail
                if new_sl > pos.current_sl:
                    pos.current_sl = new_sl

        return closed_now

    def _close(self, tid: str, price: float, ts: pd.Timestamp,
               reason: str, fraction: float) -> Optional[Dict]:
        if tid not in self.open:
            return None
        pos  = self.open[tid]
        exit_price = price * (1 - SLIPPAGE)
        size       = pos.remaining * fraction
        commission = exit_price * size * COMMISSION_RATE
        pnl_gross  = (exit_price - pos.entry_price) * size
        pnl_net    = pnl_gross - commission
        self.capital += pnl_net
        self.peak     = max(self.peak, self.capital)

        dur_h   = (ts - pos.entry_time).total_seconds() / 3600
        r_mult  = pnl_net / pos.risk if pos.risk > 0 else 0
        pnl_pct = pnl_net / self.initial * 100

        trade = {
            "trade_id":    pos.trade_id + ("_p" if reason == "tp1_partial" else ""),
            "symbol":      pos.symbol,
            "strategy":    pos.strategy,
            "cycle_phase": pos.cycle_phase,
            "regime":      pos.regime,
            "direction":   "long",
            "entry_price": pos.entry_price,
            "exit_price":  round(exit_price, 8),
            "stop_loss":   pos.stop_loss,
            "tp1":         pos.tp1,
            "tp2":         pos.tp2,
            "units":       round(size, 8),
            "pnl_usd":     round(pnl_net, 4),
            "pnl_pct":     round(pnl_pct, 6),
            "r_multiple":  round(r_mult, 3),
            "risk_amount": pos.risk,
            "notional":    pos.notional,
            "duration_hours": round(dur_h, 2),
            "exit_reason": reason,
            "entry_reason": pos.entry_reason,
            "ml_proba":    pos.ml_proba,
            "entry_time":  pos.entry_time,
            "exit_time":   ts,
        }
        self.closed.append(trade)

        if fraction >= 1.0:
            del self.open[tid]
        else:
            pos.remaining -= size
        return trade

    def close_all(self, prices: Dict, ts: pd.Timestamp) -> List[Dict]:
        closed = []
        for tid in list(self.open.keys()):
            pos   = self.open[tid]
            price = prices.get(pos.symbol, {}).get("close", pos.entry_price)
            ct    = self._close(tid, price, ts, "backtest_end", 1.0)
            if ct: closed.append(ct)
        return closed

    @property
    def drawdown_pct(self) -> float:
        if self.peak <= 0: return 0.0
        return (self.capital - self.peak) / self.peak * 100

    @property
    def return_pct(self) -> float:
        return (self.capital - self.initial) / self.initial * 100


# ══════════════════════════════════════════════════════════════════════════════
# ── Motor de backtest principal ────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(
    df_map:   Dict[str, pd.DataFrame],
    capital:  float,
    verbose:  bool = True,
) -> Tuple[BacktestPortfolio, List[Dict]]:

    portfolio  = BacktestPortfolio(capital)
    selector   = AdaptiveSignalSelector()
    detector   = CycleDetector() if CYCLE_OK else None
    all_trades: List[Dict] = []

    # Ciclo macro inicial — precalcular para todos los pares antes del loop
    cycle_cache: Dict[str, CycleState] = {}
    cycle_updated_at: Dict[str, int]   = {}

    if detector:
        print("  Precalculando ciclos iniciales...")
        for sym in df_map.keys():
            # Usar los primeros 200 días disponibles como snapshot inicial
            first_ts = df_map[sym]["timestamp"].iloc[WARMUP]
            df_daily_init = load_daily_snapshot(sym, first_ts)
            if len(df_daily_init) >= 50:
                try:
                    cycle_cache[sym] = detector.detect(df_daily_init)
                except Exception:
                    pass

    ref_sym = next((s for s in ["BTC/USDC","ETH/USDC"] if s in df_map), list(df_map.keys())[0])
    ref_df  = df_map[ref_sym]
    n       = len(ref_df)
    WARMUP  = 220

    ts_index: Dict[str, Dict] = {}
    for sym, df in df_map.items():
        ts_index[sym] = {ts: i for i, ts in enumerate(df["timestamp"])}

    iterator = tqdm(range(WARMUP, n), desc="Backtesting V6",
                    unit="velas", disable=not verbose)

    for i in iterator:
        ref_row   = ref_df.iloc[i]
        timestamp = ref_row["timestamp"]
        hour_utc  = timestamp.hour

        # Precios de la vela actual
        prices: Dict[str, Dict] = {}
        for sym, df in df_map.items():
            idx = ts_index[sym].get(timestamp)
            if idx is None: continue
            row = df.iloc[idx]
            prices[sym] = {
                "close": float(row["close"]),
                "high":  float(row["high"]),
                "low":   float(row["low"]),
            }

        # Actualizar posiciones
        closed_now = portfolio.update(timestamp, prices)
        all_trades.extend(closed_now)
        portfolio.equity.append(portfolio.capital)

        # Cooldowns activos
        active_cooldowns: Dict[str, float] = {}

        # Actualizar ciclo macro cada 24 velas (una vez al día)
        if detector and i % 24 == 0:
            for sym in df_map.keys():
                df_daily = load_daily_snapshot(sym, timestamp)
                if len(df_daily) >= 100:
                    try:
                        cycle_cache[sym]      = detector.detect(df_daily)
                        cycle_updated_at[sym] = i
                    except Exception:
                        pass

        # Circuit breakers
        dd = portfolio.drawdown_pct
        if dd <= -10.0:
            break  # Max drawdown del backtest

        # Buscar nuevas señales
        open_syms  = {pos.symbol for pos in portfolio.open.values()}
        open_groups: Dict[str, int] = {}
        for pos in portfolio.open.values():
            g = CORRELATION_GROUPS.get(pos.symbol, "other")
            open_groups[g] = open_groups.get(g, 0) + 1

        if len(portfolio.open) >= 3:
            continue

        for sym in SYMBOLS:
            if sym not in df_map: continue
            if sym in open_syms: continue

            # Correlación
            grp = CORRELATION_GROUPS.get(sym, "other")
            if grp != "btc" and open_groups.get(grp, 0) >= MAX_CORR_POS:
                continue

            idx = ts_index[sym].get(timestamp)
            if idx is None or idx < WARMUP: continue

            window = df_map[sym].iloc[max(0, idx-300):idx+1]
            if len(window) < 220: continue

            # Ciclo y régimen
            cycle  = cycle_cache.get(sym)
            regime = detect_regime(window, sym)
            if not regime.is_tradeable: continue

            # Señal
            try:
                sig = selector.analyze(window, sym, regime, cycle, hour_utc)
            except Exception:
                continue
            if sig is None: continue

            # Sizing
            daily_pnl_pct = 0.0  # simplificado en backtest
            units, risk, notional = compute_position_size(
                sig, portfolio.capital, len(portfolio.open), daily_pnl_pct
            )
            if units <= 0: continue

            # Abrir posición
            tid = str(uuid.uuid4())[:8]
            pos = BacktestPosition(
                trade_id=tid, symbol=sym,
                strategy=f"{sig.strategy}_{cycle.phase if cycle else 'UNKNOWN'}",
                entry_idx=i, entry_price=sig.entry_price,
                stop_loss=sig.stop_loss, tp1=sig.tp1, tp2=sig.tp2,
                units=units, risk=risk, notional=notional,
                entry_time=timestamp,
                regime=regime.regime,
                cycle_phase=cycle.phase if cycle else "UNKNOWN",
                ml_proba=sig.ml_proba,
                entry_reason=" | ".join(sig.reasons[:4]),
            )
            portfolio.open_position(pos)
            open_groups[grp] = open_groups.get(grp, 0) + 1
            open_syms.add(sym)

    # Cerrar posiciones abiertas
    final_prices = {s: {"close": float(df.iloc[-1]["close"]), "high": float(df.iloc[-1]["high"]), "low": float(df.iloc[-1]["low"])} for s, df in df_map.items()}
    remaining = portfolio.close_all(final_prices, ref_df.iloc[-1]["timestamp"])
    all_trades.extend(remaining)

    return portfolio, all_trades


# ══════════════════════════════════════════════════════════════════════════════
# ── Reporte detallado ──────────────────────────════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

def print_report(portfolio: BacktestPortfolio, trades: List[Dict],
                 by_phase: bool = False) -> None:
    closed = [t for t in trades if t["exit_reason"] != "backtest_end"]
    if not closed:
        print("\n  ⚠️  Sin trades cerrados."); return

    pnls  = [t["pnl_usd"] for t in closed]
    wins  = [p for p in pnls if p > 0]
    loses = [p for p in pnls if p <= 0]
    wr    = len(wins)/len(pnls)*100 if pnls else 0
    pf    = sum(wins)/abs(sum(loses)) if loses and sum(loses)!=0 else float("inf")
    avg_r = sum(t["r_multiple"] for t in closed)/len(closed) if closed else 0

    print(f"\n{'═'*62}")
    print(f"  BACKTEST V6 — RESULTADOS")
    print(f"{'═'*62}")
    print(f"  Capital inicial:   ${portfolio.initial:,.2f}")
    print(f"  Capital final:     ${portfolio.capital:,.2f}  ({portfolio.return_pct:+.2f}%)")
    print(f"  Max Drawdown:      {portfolio.drawdown_pct:.2f}%")
    print(f"")
    print(f"  Trades cerrados:   {len(closed)}")
    print(f"  Win Rate:          {wr:.1f}%  {'✅' if wr>=45 else '❌'} (obj ≥45%)")
    print(f"  Profit Factor:     {pf:.2f}  {'✅' if pf>=1.3 else '❌'} (obj ≥1.3)")
    print(f"  R medio:           {avg_r:+.2f}R")
    print(f"  PnL neto:          ${sum(pnls):.2f}")
    print(f"  Mejor trade:       ${max(pnls):.2f}")
    print(f"  Peor trade:        ${min(pnls):.2f}")

    # Por estrategia
    print(f"\n  Por estrategia:")
    by_strat: Dict[str, List] = {}
    for t in closed:
        s = t["strategy"].split("_")[0]
        by_strat.setdefault(s,[]).append(t["pnl_usd"])
    for strat, sp in sorted(by_strat.items()):
        sw = len([p for p in sp if p>0])
        icon = "✅" if sum(sp)>0 else "❌"
        print(f"    {icon} {strat:<20} {len(sp):>4} trades  "
              f"WR:{sw/len(sp)*100:>4.0f}%  PnL:${sum(sp):>8.2f}")

    # Por fase del ciclo
    if by_phase:
        print(f"\n  Por fase del ciclo:")
        by_p: Dict[str, List] = {}
        for t in closed:
            p = t.get("cycle_phase","UNKNOWN")
            by_p.setdefault(p,[]).append(t["pnl_usd"])
        for phase, pp in sorted(by_p.items()):
            pw = len([p for p in pp if p>0])
            icon = "✅" if sum(pp)>0 else "❌"
            print(f"    {icon} {phase:<20} {len(pp):>4} trades  "
                  f"WR:{pw/len(pp)*100:>4.0f}%  PnL:${sum(pp):>8.2f}")

    print(f"{'═'*62}")


# ══════════════════════════════════════════════════════════════════════════════
# ── Guardar en PostgreSQL ──────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def save_to_db(engine, trades: List[Dict]) -> int:
    if not trades: return 0
    inserted = 0
    with engine.begin() as conn:
        for t in trades:
            try:
                conn.execute(text("""
                    INSERT INTO trades_journal (
                        trade_id, symbol, strategy, timeframe, direction,
                        entry_price, exit_price, stop_loss, tp1, tp2, take_profit_1,
                        units, position_size, pnl, pnl_usd, pnl_pct, r_multiple,
                        risk_amount, duration_hours, exit_reason, entry_reason,
                        market_regime, regime, ml_proba,
                        entry_time, exit_time, is_backtest, observations
                    ) VALUES (
                        :tid, :sym, :strat, '1h', 'long',
                        :ep, :xp, :sl, :tp1, :tp2, :tp1,
                        :size, :size, :pnl, :pnl, :pnl_pct, :r,
                        :risk, :dur, :xreason, :ereason,
                        :regime, :regime, :ml,
                        :et, :xt, TRUE,
                        :obs
                    ) ON CONFLICT DO NOTHING
                """), {
                    "tid":    t["trade_id"],
                    "sym":    t["symbol"],
                    "strat":  t["strategy"],
                    "ep":     t["entry_price"],  "xp": t["exit_price"],
                    "sl":     t["stop_loss"],
                    "tp1":    t["tp1"],          "tp2": t["tp2"],
                    "size":   t["units"],
                    "pnl":    t["pnl_usd"],      "pnl_pct": t["pnl_pct"],
                    "r":      t["r_multiple"],
                    "risk":   t["risk_amount"],
                    "dur":    t["duration_hours"],
                    "xreason":t["exit_reason"],  "ereason": t["entry_reason"],
                    "regime": t["regime"],
                    "ml":     t["ml_proba"],
                    "et":     t["entry_time"],   "xt": t["exit_time"],
                    "obs":    (f"BACKTEST_V6 weight={BACKTEST_WEIGHT} "
                               f"phase={t.get('cycle_phase','?')} "
                               f"notional=${t.get('notional',0):.2f}"),
                })
                inserted += 1
            except Exception:
                pass
    return inserted


# ══════════════════════════════════════════════════════════════════════════════
# ── CLI ───────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Backtester V6 — Motor Adaptativo de Ciclos")
    parser.add_argument("--months",        type=int,   default=None,
                        help="Meses de histórico (None = todo)")
    parser.add_argument("--pairs",         nargs="+",  default=None)
    parser.add_argument("--capital",       type=float, default=INITIAL_CAPITAL)
    parser.add_argument("--dry-run",       action="store_true")
    parser.add_argument("--retrain",       action="store_true")
    parser.add_argument("--report-by-phase",action="store_true")
    parser.add_argument("--quiet",         action="store_true")
    args = parser.parse_args()

    symbols = args.pairs or SYMBOLS
    verbose = not args.quiet
    months_str = f"{args.months} meses" if args.months else "HISTÓRICO COMPLETO"

    print(f"\n{'═'*62}")
    print(f"  🤖 Backtester V6 — Adaptive Cycle")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═'*62}")
    print(f"  Pares:      {', '.join(symbols)}")
    print(f"  Período:    {months_str}")
    print(f"  Capital:    ${args.capital:,.2f}")
    print(f"  Ciclos:     {'✅ CycleDetector activo' if CYCLE_OK else '❌ Sin ciclos'}")
    print(f"  Modo:       {'DRY-RUN' if args.dry_run else 'GUARDANDO EN PostgreSQL'}")

    # ── 1. Cargar datos ────────────────────────────────────────────────────────
    print(f"\n[1/4] Cargando OHLCV 1H...")
    df_map: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = load_ohlcv(sym, "1h", args.months)
        if len(df) < 250:
            print(f"  ⚠️  {sym}: {len(df)} velas — insuficiente"); continue
        print(f"  ✓ {sym}: {len(df):,} velas "
              f"({df['timestamp'].iloc[0].date()} → {df['timestamp'].iloc[-1].date()})")
        df_map[sym] = df

    if not df_map:
        print("  ✗ Sin datos. Ejecuta download_full_history.sh primero.")
        sys.exit(1)

    # ── 2. Calcular indicadores ────────────────────────────────────────────────
    print(f"\n[2/4] Calculando indicadores V6...")
    for sym in list(df_map.keys()):
        try:
            df_map[sym] = enrich_dataframe(df_map[sym])
            print(f"  ✓ {sym}")
        except Exception as e:
            print(f"  ✗ {sym}: {e}")
            del df_map[sym]

    if not df_map:
        print("  ✗ Error en todos los pares.")
        sys.exit(1)

    # ── 3. Ejecutar backtest ───────────────────────────────────────────────────
    print(f"\n[3/4] Ejecutando backtest event-driven con CycleDetector...")
    t0 = time.time()
    portfolio, all_trades = run_backtest(df_map, args.capital, verbose=verbose)
    elapsed = time.time() - t0
    print(f"\n  Completado en {elapsed:.1f}s")
    print_report(portfolio, all_trades, by_phase=args.report_by_phase)

    # ── 4. Guardar ────────────────────────────────────────────────────────────
    closed = [t for t in all_trades if t["exit_reason"] != "backtest_end"]
    if args.dry_run:
        print(f"\n[4/4] DRY-RUN — {len(closed)} trades NO guardados.")
    else:
        print(f"\n[4/4] Guardando {len(closed)} trades en PostgreSQL...")
        raw = os.environ.get("DATABASE_URL","")
        url = raw.replace("+asyncpg","").replace("localhost","127.0.0.1")
        db_engine = create_engine(url, connect_args={"connect_timeout":10})
        ins = save_to_db(db_engine, closed)
        print(f"  ✅ {ins} trades insertados (is_backtest=TRUE)")

    # ── 5. Retrain ────────────────────────────────────────────────────────────
    if args.retrain and not args.dry_run:
        print(f"\n[5/5] Iniciando retrain ML...")
        import subprocess
        r = subprocess.run(
            [sys.executable, "ml/retrain_model.py","--initial","--min-trades","50"],
            cwd=str(PROJECT_ROOT), capture_output=False,
        )
        print("  ✅ Retrain completado" if r.returncode==0 else "  ⚠️  Retrain con errores")

    print(f"\n{'═'*62}")
    print(f"  ✅ Backtest V6 finalizado")
    print(f"{'═'*62}\n")


if __name__ == "__main__":
    main()
