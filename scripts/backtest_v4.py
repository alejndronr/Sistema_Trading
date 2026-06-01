"""
scripts/backtest_v4.py — Backtester con lógica exacta del Motor V4
===================================================================
Usa el mismo código de análisis que live_engine.py V4:
  · enrich_dataframe()      — todos los indicadores inline
  · detect_regime()         — clasificación de régimen
  · strategy_*()            — las 4 estrategias con scoring 0-100
  · select_best_signal()    — selección por régimen
  · compute_position_size() — Kelly fraccionado
  · Filtro correlación      — máx 1 altcoin simultánea
  · Filtro horario MR       — solo sesión asiática 00-07 UTC

Guarda trades en PostgreSQL (trades_journal, is_backtest=TRUE)
para que retrain_model.py los use directamente.

Uso:
    python scripts/backtest_v4.py                          # últimos 3 meses, todos los pares
    python scripts/backtest_v4.py --months 6              # 6 meses
    python scripts/backtest_v4.py --pairs BTC/USDC ETH/USDC  # pares específicos
    python scripts/backtest_v4.py --dry-run               # sin guardar en BD
    python scripts/backtest_v4.py --retrain               # reentrenar ML al terminar
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
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

# ══════════════════════════════════════════════════════════════════════════════
# ── Importar lógica V4 directamente del live_engine ───────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
# En lugar de reimplementar, importamos las funciones del live_engine V4.
# Si por algún motivo falla el import (cambios futuros), usamos fallback inline.

try:
    from live_engine import (
        enrich_dataframe,
        detect_regime,
        strategy_trend_following,
        strategy_mean_reversion,
        strategy_breakout,
        strategy_momentum_scalp,
        select_best_signal,
        compute_position_size,
        SignalResult,
        MarketRegime,
        SYMBOLS,
        MIN_SIGNAL_SCORE,
        MIN_SCORE_MEAN_REVERSION,
        MAX_CORRELATED_POSITIONS,
        CORRELATION_GROUPS,
        MR_ALLOWED_HOURS_UTC,
        INITIAL_CAPITAL,
        COMMISSION_RATE,
        SLIPPAGE_ESTIMATE,
    )
    print("✅ Lógica V4 importada desde live_engine.py")
    V4_IMPORT_OK = True
except ImportError as e:
    print(f"⚠️  No se pudo importar live_engine.py ({e})")
    print("    Asegúrate de ejecutar desde /home/trading/sistema_trading/")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# ── Config del backtester ──────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

BACKTEST_SYMBOLS: List[str] = SYMBOLS   # mismos que el motor

# Peso de trades de backtest vs reales en el retrain (0.3 = 30%)
BACKTEST_SAMPLE_WEIGHT = 0.3

# Comisiones y slippage (reales de Binance spot)
COMMISSION     = COMMISSION_RATE     # 0.001
SLIPPAGE       = SLIPPAGE_ESTIMATE   # 0.001


# ══════════════════════════════════════════════════════════════════════════════
# ── Base de datos ──────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def get_engine():
    raw = os.environ.get("DATABASE_URL", "")
    if not raw:
        raise RuntimeError("DATABASE_URL no configurada en .env")
    url = (raw.replace("+asyncpg", "")
              .replace("postgresql+asyncpg://", "postgresql://")
              .replace("localhost", "127.0.0.1"))
    e = create_engine(url, pool_pre_ping=True,
                      connect_args={"connect_timeout": 10})
    with e.connect() as c:
        c.execute(text("SELECT 1"))
    return e


def load_ohlcv_sqlite(symbol: str, timeframe: str, months: int) -> pd.DataFrame:
    """Carga OHLCV desde SQLite (donde download_data.py guarda los datos)."""
    import sqlite3
    db_path = PROJECT_ROOT / "data" / "db" / "trading.db"
    if not db_path.exists():
        return pd.DataFrame()

    since = datetime.now(timezone.utc) - timedelta(days=months * 30)
    since_str = since.strftime("%Y-%m-%d %H:%M:%S")

    conn = sqlite3.connect(str(db_path))
    try:
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv
            WHERE symbol = ? AND timeframe = ?
              AND timestamp >= ?
            ORDER BY timestamp ASC
        """
        df = pd.read_sql_query(query, conn,
                               params=(symbol, timeframe, since_str))
    finally:
        conn.close()

    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.dropna().reset_index(drop=True)


def save_backtest_trades(engine, trades: List[Dict]) -> int:
    """
    Guarda trades del backtest en trades_journal con is_backtest=TRUE.
    Usa UPSERT para evitar duplicados si se re-ejecuta el backtest.
    Retorna el número de trades insertados.
    """
    if not trades:
        return 0

    inserted = 0
    with engine.begin() as conn:
        for t in trades:
            try:
                conn.execute(text("""
                    INSERT INTO trades_journal (
                        trade_id, symbol, strategy, timeframe, direction,
                        setup_quality, entry_price, exit_price, stop_loss,
                        tp1, tp2, take_profit_1,
                        units, position_size,
                        pnl, pnl_usd, pnl_pct, r_multiple,
                        risk_amount, duration_hours,
                        exit_reason, entry_reason,
                        market_regime, regime,
                        ml_proba, entry_time, exit_time,
                        is_backtest, observations
                    ) VALUES (
                        :tid, :sym, :strat, :tf, :dir,
                        :qual, :ep, :xp, :sl,
                        :tp1, :tp2, :tp1,
                        :size, :size,
                        :pnl, :pnl, :pnl_pct, :r,
                        :risk, :dur,
                        :xreason, :ereason,
                        :regime, :regime,
                        :ml, :et, :xt,
                        TRUE, :obs
                    )
                    ON CONFLICT DO NOTHING
                """), {
                    "tid":     t["trade_id"],
                    "sym":     t["symbol"],
                    "strat":   t["strategy"],
                    "tf":      t["timeframe"],
                    "dir":     t["direction"],
                    "qual":    t.get("setup_quality_int", 0),
                    "ep":      t["entry_price"],
                    "xp":      t["exit_price"],
                    "sl":      t["stop_loss"],
                    "tp1":     t["tp1"],
                    "tp2":     t.get("tp2"),
                    "size":    t["units"],
                    "pnl":     t["pnl_usd"],
                    "pnl_pct": t["pnl_pct"],
                    "r":       t["r_multiple"],
                    "risk":    t["risk_amount"],
                    "dur":     t["duration_hours"],
                    "xreason": t["exit_reason"],
                    "ereason": t["entry_reason"],
                    "regime":  t["regime"],
                    "ml":      t.get("ml_proba", 0.5),
                    "et":      t["entry_time"],
                    "xt":      t["exit_time"],
                    "obs":     (
                        f"BACKTEST weight={BACKTEST_SAMPLE_WEIGHT} "
                        f"score={t.get('score',0):.0f} "
                        f"notional=${t.get('notional_usd',0):.2f}"
                    ),
                })
                inserted += 1
            except Exception as ex:
                pass  # Duplicado o error puntual — continuar

    return inserted


# ══════════════════════════════════════════════════════════════════════════════
# ── Motor de backtesting V4 ───────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

class BacktestPortfolio:
    """
    Portfolio simulado para el backtester V4.
    Gestión de posiciones con TP1 parcial (50%) + TP2 completo,
    trailing stop, breakeven y comisiones/slippage reales.
    """

    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.capital         = initial_capital
        self.peak_capital    = initial_capital
        self.open_positions: Dict[str, Dict] = {}
        self.closed_trades:  List[Dict]       = []
        self.equity_curve:   List[Dict]       = []

    def open(self, signal: SignalResult, units: float, risk_usd: float,
             notional_usd: float, timestamp: datetime, timeframe: str) -> Optional[str]:
        """Abre posición con slippage y comisión de entrada."""
        entry_real = signal.entry_price * (1 + SLIPPAGE)  # slippage largo
        commission  = entry_real * units * COMMISSION

        if notional_usd + commission > self.capital * 0.95:
            return None  # Protección: no usar más del 95% del capital

        self.capital -= commission
        tid = str(uuid.uuid4())[:8]

        self.open_positions[tid] = {
            "trade_id":    tid,
            "symbol":      signal.symbol,
            "strategy":    signal.strategy,
            "timeframe":   timeframe,
            "direction":   signal.direction,
            "quality":     signal.quality,
            "score":       signal.score,
            "entry_price": entry_real,
            "stop_loss":   signal.stop_loss,
            "current_sl":  signal.stop_loss,
            "tp1":         signal.tp1,
            "tp2":         signal.tp2,
            "tp3":         signal.tp3,
            "units":       units,
            "remaining":   units,
            "risk_amount": risk_usd,
            "notional_usd":notional_usd,
            "entry_time":  timestamp,
            "regime":      signal.regime.regime,
            "ml_proba":    signal.ml_proba,
            "entry_reason":" | ".join(signal.reasons[:4]),
            "tp1_hit":     False,
        }
        return tid

    def update(self, timestamp: datetime, prices: Dict[str, float]) -> List[Dict]:
        """
        Actualiza todas las posiciones con el precio actual.
        Cierra por SL, TP1 parcial o TP2.
        Retorna lista de trades cerrados en esta vela.
        """
        closed_this_bar = []

        for tid in list(self.open_positions.keys()):
            pos   = self.open_positions[tid]
            sym   = pos["symbol"]
            price = prices.get(sym)
            if price is None:
                continue

            high = prices.get(f"{sym}_high", price)
            low  = prices.get(f"{sym}_low",  price)

            # ── Stop Loss ────────────────────────────────────────────────────
            if low <= pos["current_sl"]:
                ct = self._close(tid, pos["current_sl"], timestamp, "stop_loss",
                                 fraction=1.0)
                if ct:
                    closed_this_bar.append(ct)
                continue

            # ── TP1 parcial (50%) ────────────────────────────────────────────
            if not pos["tp1_hit"] and high >= pos["tp1"]:
                ct = self._close(tid, pos["tp1"], timestamp, "tp1_partial",
                                 fraction=0.5)
                if ct:
                    closed_this_bar.append(ct)
                    pos["tp1_hit"] = True
                    # Mover SL a breakeven (entrada + comisión)
                    be = pos["entry_price"] * (1 + COMMISSION)
                    pos["current_sl"] = max(pos["current_sl"], be)

            # ── TP2 completo (resto) ─────────────────────────────────────────
            if pos["tp1_hit"] and high >= pos["tp2"]:
                ct = self._close(tid, pos["tp2"], timestamp, "tp2",
                                 fraction=1.0)
                if ct:
                    closed_this_bar.append(ct)
                continue

            # ── Trailing stop tras TP1 (1.5 ATR del máximo) ─────────────────
            # Simplificado: si el precio sube >1% tras TP1, subir SL
            if pos["tp1_hit"]:
                trail_dist = pos["entry_price"] * 0.01
                new_sl = high - trail_dist
                if new_sl > pos["current_sl"]:
                    pos["current_sl"] = new_sl

        return closed_this_bar

    def _close(self, tid: str, exit_price: float, exit_time: datetime,
               exit_reason: str, fraction: float) -> Optional[Dict]:
        if tid not in self.open_positions:
            return None
        pos = self.open_positions[tid]

        exit_real = exit_price * (1 - SLIPPAGE)  # slippage salida
        size      = pos["remaining"] * fraction
        commission = exit_real * size * COMMISSION

        pnl_gross = (exit_real - pos["entry_price"]) * size
        pnl_net   = pnl_gross - commission
        self.capital += pnl_net
        self.peak_capital = max(self.peak_capital, self.capital)

        dur_h = (exit_time - pos["entry_time"]).total_seconds() / 3600
        r_mult = pnl_net / pos["risk_amount"] if pos["risk_amount"] > 0 else 0

        qual_map = {"A+": 95, "A": 80, "B": 65, "C": 50}

        trade = {
            "trade_id":          pos["trade_id"] + ("_tp1" if exit_reason == "tp1_partial" else ""),
            "symbol":            pos["symbol"],
            "strategy":          pos["strategy"],
            "timeframe":         pos["timeframe"],
            "direction":         pos["direction"],
            "setup_quality":     pos["quality"],
            "setup_quality_int": qual_map.get(pos["quality"], 65),
            "score":             pos["score"],
            "entry_price":       pos["entry_price"],
            "exit_price":        exit_real,
            "stop_loss":         pos["stop_loss"],
            "tp1":               pos["tp1"],
            "tp2":               pos["tp2"],
            "units":             size,
            "risk_amount":       pos["risk_amount"],
            "notional_usd":      pos["notional_usd"],
            "pnl_usd":           round(pnl_net, 4),
            "pnl_pct":           round(pnl_net / self.initial_capital * 100, 6),
            "r_multiple":        round(r_mult, 3),
            "duration_hours":    round(dur_h, 2),
            "exit_reason":       exit_reason,
            "entry_reason":      pos["entry_reason"],
            "regime":            pos["regime"],
            "ml_proba":          pos["ml_proba"],
            "entry_time":        pos["entry_time"],
            "exit_time":         exit_time,
        }
        self.closed_trades.append(trade)

        if fraction >= 1.0:
            del self.open_positions[tid]
        else:
            pos["remaining"] -= size

        return trade

    def close_all(self, prices: Dict[str, float], timestamp: datetime) -> List[Dict]:
        """Cierra todas las posiciones al final del backtest."""
        closed = []
        for tid in list(self.open_positions.keys()):
            pos   = self.open_positions[tid]
            price = prices.get(pos["symbol"], pos["entry_price"])
            ct    = self._close(tid, price, timestamp, "backtest_end", 1.0)
            if ct:
                closed.append(ct)
        return closed

    @property
    def total_return_pct(self) -> float:
        return (self.capital - self.initial_capital) / self.initial_capital * 100

    @property
    def max_drawdown_pct(self) -> float:
        if self.peak_capital <= 0:
            return 0.0
        return (self.capital - self.peak_capital) / self.peak_capital * 100


def run_backtest_v4(
    df_map:    Dict[str, pd.DataFrame],
    timeframe: str,
    capital:   float,
    verbose:   bool = True,
) -> Tuple[BacktestPortfolio, List[Dict]]:
    """
    Ejecuta el backtesting multi-par con la lógica exacta del motor V4.

    df_map: {symbol: DataFrame OHLCV enriquecido con indicadores}
    Itera vela a vela (event-driven) en todos los símbolos simultáneamente.
    """
    portfolio = BacktestPortfolio(capital)
    all_trades: List[Dict] = []

    # Alinear todos los DataFrames en el mismo índice temporal
    # Usar BTC como referencia de timestamps
    ref_sym = next((s for s in ["BTC/USDC", "ETH/USDC"] if s in df_map), list(df_map.keys())[0])
    ref_df  = df_map[ref_sym]
    n_candles = len(ref_df)

    if verbose:
        print(f"\n  Período: {ref_df['timestamp'].iloc[0].date()} → {ref_df['timestamp'].iloc[-1].date()}")
        print(f"  Velas:   {n_candles:,} | Pares: {len(df_map)} | Capital: ${capital:,.2f}")
        print()

    # Pre-construir índice timestamp → posición para cada df
    ts_index: Dict[str, Dict] = {}
    for sym, df in df_map.items():
        ts_index[sym] = {ts: i for i, ts in enumerate(df["timestamp"])}

    # Mínimo de velas para que los indicadores estén calientes
    WARMUP = 210

    iterator = tqdm(range(WARMUP, n_candles), desc="Backtesting V4",
                    unit="velas", disable=not verbose)

    for i in iterator:
        ref_row   = ref_df.iloc[i]
        timestamp = ref_row["timestamp"]
        hour_utc  = timestamp.hour

        # Construir precios actuales (high/low para TP/SL)
        current_prices: Dict[str, float] = {}
        for sym, df in df_map.items():
            ts_idx = ts_index[sym].get(timestamp)
            if ts_idx is None:
                continue
            row = df.iloc[ts_idx]
            current_prices[sym]              = float(row["close"])
            current_prices[f"{sym}_high"]    = float(row["high"])
            current_prices[f"{sym}_low"]     = float(row["low"])

        # ── 1. Actualizar posiciones abiertas (SL/TP) ─────────────────────
        closed_now = portfolio.update(timestamp, current_prices)
        all_trades.extend(closed_now)

        # ── 2. Buscar nuevas señales ────────────────────────────────────────
        # Estado actual de correlación
        open_syms   = {pos["symbol"] for pos in portfolio.open_positions.values()}
        open_groups: Dict[str, int] = {}
        for pos in portfolio.open_positions.values():
            grp = CORRELATION_GROUPS.get(pos["symbol"], "other")
            open_groups[grp] = open_groups.get(grp, 0) + 1

        if len(portfolio.open_positions) >= 3:
            continue  # MAX_POSITIONS = 3

        for sym in BACKTEST_SYMBOLS:
            if sym not in df_map:
                continue
            if sym in open_syms:
                continue  # Anti-piramidación

            # Filtro correlación
            grp = CORRELATION_GROUPS.get(sym, "other")
            if grp != "btc" and open_groups.get(grp, 0) >= MAX_CORRELATED_POSITIONS:
                continue

            ts_idx = ts_index[sym].get(timestamp)
            if ts_idx is None or ts_idx < WARMUP:
                continue

            df_sym = df_map[sym]
            window = df_sym.iloc[max(0, ts_idx - 300): ts_idx + 1]

            if len(window) < 210:
                continue

            # Detectar régimen
            regime = detect_regime(window, sym)
            if not regime.is_tradeable:
                continue

            # Evaluar las 4 estrategias
            candidates = []
            if regime.regime in ("BULL", "RANGE", "HIGH_VOL"):
                candidates.append(strategy_trend_following(window, regime))
                candidates.append(strategy_mean_reversion(window, regime))
                candidates.append(strategy_breakout(window, regime))
                candidates.append(strategy_momentum_scalp(window, regime))
            elif regime.regime == "BEAR":
                candidates.append(strategy_mean_reversion(window, regime))
                candidates.append(strategy_trend_following(window, regime))

            # Filtro horario MeanReversion
            valid = []
            for sig in candidates:
                if sig is None:
                    continue
                if "MeanReversion" in sig.strategy:
                    if hour_utc not in MR_ALLOWED_HOURS_UTC:
                        continue
                valid.append(sig)

            best = select_best_signal(valid)
            if best is None:
                continue

            # Sizing
            daily_pnl_pct = (portfolio.capital - capital) / capital
            units, risk_usd, notional_usd = compute_position_size(
                best, portfolio.capital,
                len(portfolio.open_positions), daily_pnl_pct
            )
            if units <= 0:
                continue

            # Actualizar entry_price al precio real de la vela
            real_price = current_prices.get(sym, best.entry_price)
            best.entry_price = real_price

            # Abrir posición
            tid = portfolio.open(best, units, risk_usd, notional_usd,
                                 timestamp, timeframe)
            if tid:
                # Actualizar grupos abiertos
                open_groups[grp] = open_groups.get(grp, 0) + 1
                open_syms.add(sym)

        # Registrar equity cada 24 velas
        if i % 24 == 0:
            portfolio.equity_curve.append({
                "timestamp": timestamp,
                "capital":   portfolio.capital,
            })

    # Cerrar posiciones abiertas al final
    final_prices = {sym: float(df.iloc[-1]["close"]) for sym, df in df_map.items()}
    remaining = portfolio.close_all(final_prices, ref_df.iloc[-1]["timestamp"])
    all_trades.extend(remaining)

    return portfolio, all_trades


# ══════════════════════════════════════════════════════════════════════════════
# ── Métricas y reporte ────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def print_report(portfolio: BacktestPortfolio, trades: List[Dict],
                 months: int) -> None:
    """Imprime resumen del backtest."""
    closed = [t for t in trades if t["exit_reason"] != "backtest_end"]

    if not closed:
        print("\n  ⚠️  Sin trades cerrados en el período.")
        return

    pnls  = [t["pnl_usd"] for t in closed]
    wins  = [p for p in pnls if p > 0]
    loses = [p for p in pnls if p <= 0]

    win_rate = len(wins) / len(pnls) * 100 if pnls else 0
    pf       = sum(wins) / abs(sum(loses)) if loses and sum(loses) != 0 else float("inf")
    avg_r    = sum(t["r_multiple"] for t in closed) / len(closed) if closed else 0

    # Por estrategia
    by_strat: Dict[str, List[float]] = {}
    for t in closed:
        s = t["strategy"]
        by_strat.setdefault(s, []).append(t["pnl_usd"])

    print(f"\n{'═'*60}")
    print(f"  BACKTEST V4 — RESULTADOS ({months} meses)")
    print(f"{'═'*60}")
    print(f"  Capital inicial:  ${portfolio.initial_capital:,.2f}")
    print(f"  Capital final:    ${portfolio.capital:,.2f}  ({portfolio.total_return_pct:+.2f}%)")
    print(f"  Max Drawdown:     {portfolio.max_drawdown_pct:.2f}%")
    print(f"")
    print(f"  Trades cerrados:  {len(closed)}")
    print(f"  Win Rate:         {win_rate:.1f}%  {'✅' if win_rate >= 45 else '❌'} (obj ≥45%)")
    print(f"  Profit Factor:    {pf:.2f}  {'✅' if pf >= 1.3 else '❌'} (obj ≥1.3)")
    print(f"  R medio:          {avg_r:+.2f}R")
    print(f"  PnL neto:         ${sum(pnls):.2f}")
    print(f"  Mejor trade:      ${max(pnls):.2f}")
    print(f"  Peor trade:       ${min(pnls):.2f}")
    print(f"")
    print(f"  Por estrategia:")
    for strat, strat_pnls in sorted(by_strat.items()):
        strat_wins = len([p for p in strat_pnls if p > 0])
        strat_wr   = strat_wins / len(strat_pnls) * 100
        strat_pnl  = sum(strat_pnls)
        icon = "✅" if strat_pnl > 0 else "❌"
        print(f"    {icon} {strat:<35} {len(strat_pnls):>3} trades  "
              f"WR:{strat_wr:>4.0f}%  PnL:${strat_pnl:>7.2f}")
    print(f"{'═'*60}")


# ══════════════════════════════════════════════════════════════════════════════
# ── Punto de entrada ──────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Backtester V4 — misma lógica que live_engine.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python scripts/backtest_v4.py                        # 3 meses, todos los pares
  python scripts/backtest_v4.py --months 6             # 6 meses
  python scripts/backtest_v4.py --pairs BTC/USDC       # solo BTC
  python scripts/backtest_v4.py --dry-run              # sin guardar en BD
  python scripts/backtest_v4.py --retrain              # reentrenar ML al terminar
        """,
    )
    parser.add_argument("--months",   type=int,   default=3,
                        help="Meses de histórico (default: 3)")
    parser.add_argument("--pairs",    nargs="+",  default=None,
                        help="Pares específicos (default: todos)")
    parser.add_argument("--capital",  type=float, default=INITIAL_CAPITAL,
                        help=f"Capital inicial (default: ${INITIAL_CAPITAL:.0f})")
    parser.add_argument("--timeframe",default="1h",
                        help="Timeframe (default: 1h)")
    parser.add_argument("--dry-run",  action="store_true",
                        help="No guardar en PostgreSQL")
    parser.add_argument("--retrain",  action="store_true",
                        help="Ejecutar retrain ML al terminar")
    parser.add_argument("--quiet",    action="store_true",
                        help="Menos output")
    args = parser.parse_args()

    symbols = args.pairs or BACKTEST_SYMBOLS
    verbose = not args.quiet

    print(f"\n{'═'*60}")
    print(f"  🤖 Backtester V4 — Motor Adaptativo")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═'*60}")
    print(f"  Pares:      {', '.join(symbols)}")
    print(f"  Período:    {args.months} meses")
    print(f"  Timeframe:  {args.timeframe}")
    print(f"  Capital:    ${args.capital:,.2f}")
    print(f"  Modo:       {'DRY-RUN (sin guardar)' if args.dry_run else 'GUARDANDO EN PostgreSQL'}")

    # ── 1. Cargar datos OHLCV desde SQLite ────────────────────────────────────
    print(f"\n[1/4] Cargando datos OHLCV ({args.months} meses)...")
    df_map: Dict[str, pd.DataFrame] = {}

    for sym in symbols:
        df_raw = load_ohlcv_sqlite(sym, args.timeframe, args.months)
        if df_raw.empty or len(df_raw) < 250:
            print(f"  ⚠️  {sym}: datos insuficientes ({len(df_raw)} velas) — skip")
            continue
        print(f"  ✓ {sym}: {len(df_raw):,} velas "
              f"({df_raw['timestamp'].iloc[0].date()} → "
              f"{df_raw['timestamp'].iloc[-1].date()})")
        df_map[sym] = df_raw

    if not df_map:
        print("\n  ✗ Sin datos disponibles. Ejecuta primero:")
        print("    python scripts/download_data.py --pairs BTC/USDC ETH/USDC ... --years 0.25")
        sys.exit(1)

    # ── 2. Calcular indicadores V4 ────────────────────────────────────────────
    print(f"\n[2/4] Calculando indicadores V4 ({len(df_map)} pares)...")
    for sym in list(df_map.keys()):
        try:
            df_map[sym] = enrich_dataframe(df_map[sym])
            print(f"  ✓ {sym}: indicadores OK")
        except Exception as e:
            print(f"  ✗ {sym}: error en indicadores ({e}) — eliminado")
            del df_map[sym]

    if not df_map:
        print("  ✗ Error en todos los pares.")
        sys.exit(1)

    # ── 3. Ejecutar backtesting ───────────────────────────────────────────────
    print(f"\n[3/4] Ejecutando backtest event-driven...")
    t0 = time.time()

    portfolio, all_trades = run_backtest_v4(
        df_map, args.timeframe, args.capital, verbose=verbose
    )

    elapsed = time.time() - t0
    print(f"\n  Completado en {elapsed:.1f}s")

    # Mostrar reporte
    print_report(portfolio, all_trades, args.months)

    # ── 4. Guardar en PostgreSQL ──────────────────────────────────────────────
    closed_final = [t for t in all_trades if t["exit_reason"] != "backtest_end"]

    if args.dry_run:
        print(f"\n[4/4] DRY-RUN — {len(closed_final)} trades NO guardados en BD.")
    else:
        print(f"\n[4/4] Guardando {len(closed_final)} trades en PostgreSQL...")
        try:
            engine = get_engine()
            inserted = save_backtest_trades(engine, closed_final)
            print(f"  ✅ {inserted} trades insertados en trades_journal (is_backtest=TRUE)")

            if inserted < len(closed_final):
                dupes = len(closed_final) - inserted
                print(f"  ℹ️  {dupes} ya existían (duplicados ignorados)")
        except Exception as ex:
            print(f"  ✗ Error guardando en BD: {ex}")
            print("    Los trades están disponibles en memoria — revisa la conexión.")

    # ── 5. Retrain ML ──────────────────────────────────────────────────────────
    if args.retrain and not args.dry_run:
        print(f"\n[5/5] Iniciando retrain ML...")
        import subprocess
        result = subprocess.run(
            [sys.executable, "ml/retrain_model.py", "--initial", "--min-trades", "30"],
            cwd=str(PROJECT_ROOT),
            capture_output=False,
        )
        if result.returncode == 0:
            print("  ✅ Retrain completado")
        else:
            print("  ⚠️  Retrain terminó con errores — revisa los logs")

    print(f"\n{'═'*60}")
    print(f"  ✅ Backtest V4 finalizado")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
