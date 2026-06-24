"""
backtesting/audit_real_signals.py — Auditoria de Senales Reales (Fee-Aware)
=============================================================================
Inspirado en el patron de Krypt Trader: calcular el edge NETO despues de
comisiones sobre senales REALES generadas por el sistema en paper trading,
no sobre backtests historicos.

Diferencia clave con walk_forward.py:
  - walk_forward.py testea la ESTRATEGIA sobre datos historicos de mercado
  - audit_real_signals.py toma los trades REALES del paper trading y calcula
    si el edge observado sobrevive a las comisiones y al slippage real

Uso:
    python backtesting/audit_real_signals.py
    python backtesting/audit_real_signals.py --min-trades 30
    python backtesting/audit_real_signals.py --export audit_YYYYMMDD.json

Requiere: >= 30 trades reales (is_backtest=FALSE) en PostgreSQL (trades_journal)
          para que las metricas tengan significancia estadistica.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

# ── Constantes de Binance Spot ─────────────────────────────────────────────────
TAKER_FEE   = 0.001    # 0.10% — taker fee en Binance Spot sin BNB discount
MAKER_FEE   = 0.0008   # 0.08% — maker fee (ordenes limit que se rellenen)
# El sistema usa market orders en entradas y salidas → 2 × TAKER_FEE por trade
ROUND_TRIP_FEE = 2 * TAKER_FEE   # 0.20% total por trade


def _load_trades_postgres() -> Optional[pd.DataFrame]:
    """
    Carga trades reales desde PostgreSQL (is_backtest=FALSE).
    Retorna None si no hay conexion o no hay datos.
    """
    try:
        from sqlalchemy import create_engine, text

        raw_db = os.environ.get("DATABASE_URL", "")
        if not raw_db or raw_db.startswith("sqlite"):
            return None

        url = (raw_db.replace("postgresql://", "postgresql+psycopg2://")
                     .replace("postgres://", "postgresql+psycopg2://")
                     .replace("localhost", "127.0.0.1"))

        engine = create_engine(url, connect_args={"connect_timeout": 5})
        with engine.connect() as conn:
            df = pd.read_sql(text("""
                SELECT
                    trade_id, strategy, symbol, direction,
                    entry_price, exit_price, stop_loss,
                    take_profit_1 AS tp1, take_profit_2 AS tp2,
                    units, position_size, risk_amount,
                    pnl_usd, pnl_pct, r_multiple,
                    entry_time, exit_time, duration_hours,
                    entry_reason, exit_reason,
                    market_regime, ml_proba,
                    is_backtest, created_at
                FROM trades_journal
                WHERE is_backtest = FALSE
                  AND exit_price IS NOT NULL
                ORDER BY entry_time ASC
            """), conn)
        return df if not df.empty else None
    except Exception as e:
        print(f"  [WARN] No se pudo cargar desde PostgreSQL: {e}")
        return None


def _load_trades_sqlite() -> Optional[pd.DataFrame]:
    """
    Fallback: cargar trades desde SQLite local.
    Util para pruebas en entorno de desarrollo.
    """
    try:
        import sqlite3
        db_path = ROOT / "data" / "db" / "trading.db"
        conn = sqlite3.connect(str(db_path))
        df = pd.read_sql("""
            SELECT * FROM trades_journal
            WHERE is_backtest = 0
              AND exit_price IS NOT NULL
            ORDER BY entry_time ASC
        """, conn)
        conn.close()
        return df if not df.empty else None
    except Exception as e:
        print(f"  [WARN] No se pudo cargar desde SQLite: {e}")
        return None


def compute_fee_adjusted_pnl(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula el PnL neto despues de comisiones reales de Binance.

    Para cada trade:
      fee_entry  = entry_price * units * TAKER_FEE
      fee_exit   = exit_price  * units * TAKER_FEE
      pnl_gross  = (exit_price - entry_price) * units  [long]
                   (entry_price - exit_price) * units  [short]
      pnl_net    = pnl_gross - fee_entry - fee_exit
      slippage   = pnl_reported - pnl_gross  (diferencia entre lo que el sistema
                   creia que gano y lo que realmente gano antes de fees)
    """
    df = df.copy()

    # Asegurar tipos correctos
    for col in ["entry_price", "exit_price", "units", "pnl_usd"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["notional_entry"] = df["entry_price"] * df["units"]
    df["notional_exit"]  = df["exit_price"]  * df["units"]
    df["fee_entry"]      = df["notional_entry"] * TAKER_FEE
    df["fee_exit"]       = df["notional_exit"]  * TAKER_FEE
    df["total_fees_usd"] = df["fee_entry"] + df["fee_exit"]

    # PnL bruto recalculado
    is_long  = df["direction"].str.lower() == "long"
    df["pnl_gross_recalc"] = np.where(
        is_long,
        (df["exit_price"] - df["entry_price"]) * df["units"],
        (df["entry_price"] - df["exit_price"]) * df["units"],
    )

    # Slippage implicito: diferencia entre PnL reportado y PnL recalculado
    df["slippage_usd"] = df["pnl_usd"] - df["pnl_gross_recalc"]

    # PnL neto real: gross recalculado menos fees
    df["pnl_net_usd"]   = df["pnl_gross_recalc"] - df["total_fees_usd"]
    df["pnl_net_pct"]   = df["pnl_net_usd"] / df["notional_entry"]

    return df


def compute_roc_auc_real(df: pd.DataFrame) -> Optional[float]:
    """
    Calcula el ROC-AUC del MetaLabeler sobre senales REALES.
    Compara ml_proba (probabilidad predicha) vs resultado real (win/loss).

    Si ml_proba es NaN o no existe, retorna None.
    """
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        print("  [WARN] scikit-learn no disponible. Instalar con: pip install scikit-learn")
        return None

    if "ml_proba" not in df.columns:
        return None

    df_ml = df[["ml_proba", "pnl_net_usd"]].dropna()
    if len(df_ml) < 10:
        return None

    y_true = (df_ml["pnl_net_usd"] > 0).astype(int)
    y_score = df_ml["ml_proba"].clip(0, 1)

    if y_true.nunique() < 2:
        return None  # No hay variedad en outcomes

    return float(roc_auc_score(y_true, y_score))


def compute_metrics(df: pd.DataFrame) -> dict:
    """Calcula el conjunto completo de metricas de la auditoria."""
    n_trades = len(df)
    wins     = (df["pnl_net_usd"] > 0).sum()
    losses   = (df["pnl_net_usd"] <= 0).sum()
    win_rate = wins / n_trades if n_trades > 0 else 0

    gross_profit = df[df["pnl_net_usd"] > 0]["pnl_net_usd"].sum()
    gross_loss   = abs(df[df["pnl_net_usd"] <= 0]["pnl_net_usd"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    avg_win  = df[df["pnl_net_usd"] > 0]["pnl_net_usd"].mean() if wins > 0 else 0
    avg_loss = df[df["pnl_net_usd"] <= 0]["pnl_net_usd"].mean() if losses > 0 else 0

    total_fees = df["total_fees_usd"].sum()
    total_pnl_gross = df["pnl_gross_recalc"].sum()
    total_pnl_net   = df["pnl_net_usd"].sum()
    fee_drag_pct    = total_fees / abs(total_pnl_gross) * 100 if total_pnl_gross != 0 else 0

    # Drawdown
    cumulative = df["pnl_net_usd"].cumsum()
    rolling_max = cumulative.cummax()
    drawdown    = cumulative - rolling_max
    max_dd      = float(drawdown.min())

    # Expectancy por trade
    expectancy = total_pnl_net / n_trades if n_trades > 0 else 0

    # ROC-AUC real
    roc_auc_real = compute_roc_auc_real(df)

    # Metricas por estrategia
    by_strategy = {}
    for strat in df["strategy"].unique():
        sub = df[df["strategy"] == strat]
        sub_wins = (sub["pnl_net_usd"] > 0).sum()
        sub_loss_amt = abs(sub[sub["pnl_net_usd"] <= 0]["pnl_net_usd"].sum())
        sub_pf = sub[sub["pnl_net_usd"] > 0]["pnl_net_usd"].sum() / sub_loss_amt if sub_loss_amt > 0 else float("inf")
        by_strategy[strat] = {
            "trades": len(sub),
            "win_rate": float(sub_wins / len(sub)),
            "profit_factor": float(sub_pf),
            "total_pnl_net": float(sub["pnl_net_usd"].sum()),
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_trades": int(n_trades),
        "date_range": {
            "from": str(df["entry_time"].min()),
            "to":   str(df["entry_time"].max()),
        },
        "summary": {
            "total_pnl_gross_usd": float(total_pnl_gross),
            "total_fees_usd": float(total_fees),
            "total_pnl_net_usd": float(total_pnl_net),
            "fee_drag_pct": float(fee_drag_pct),
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "avg_win_usd": float(avg_win),
            "avg_loss_usd": float(avg_loss),
            "max_drawdown_usd": float(max_dd),
            "expectancy_per_trade": float(expectancy),
        },
        "ml_model": {
            "roc_auc_real_trades": roc_auc_real,
            "interpretation": (
                "Edge significativo" if roc_auc_real and roc_auc_real >= 0.60
                else "Edge marginal" if roc_auc_real and roc_auc_real >= 0.55
                else "Sin edge confirmado" if roc_auc_real
                else "No calculable (datos insuficientes)"
            ),
        },
        "slippage": {
            "total_slippage_usd": float(df["slippage_usd"].sum()),
            "avg_slippage_per_trade_usd": float(df["slippage_usd"].mean()),
            "slippage_pct_of_pnl": float(
                df["slippage_usd"].sum() / abs(total_pnl_gross) * 100
            ) if total_pnl_gross != 0 else 0,
        },
        "by_strategy": by_strategy,
        "validation_status": _check_validation_criteria(n_trades, profit_factor, max_dd, roc_auc_real),
    }


def _check_validation_criteria(n_trades, pf, max_dd, roc_auc) -> dict:
    """
    Verifica los criterios del DISCLAIMER para aumentar capital.
    """
    criteria = {
        "min_50_trades": {"required": 50, "actual": n_trades, "passed": n_trades >= 50},
        "pf_above_110": {"required": 1.10, "actual": round(pf, 3), "passed": pf >= 1.10},
        "drawdown_below_8pct": {"required": -0.08, "actual": None, "passed": False},  # En USD
        "roc_auc_above_55": {"required": 0.55, "actual": roc_auc, "passed": bool(roc_auc and roc_auc >= 0.55)},
    }
    all_passed = all(v["passed"] for v in criteria.values())
    return {
        "criteria": criteria,
        "ready_to_increase_capital": all_passed,
        "verdict": "APTO para aumentar capital" if all_passed else "NO APTO — ver criterios fallidos",
    }


def print_report(metrics: dict) -> None:
    """Imprime el reporte en formato legible."""
    s = metrics["summary"]
    ml = metrics["ml_model"]
    slip = metrics["slippage"]
    val = metrics["validation_status"]

    print("\n" + "=" * 65)
    print("  AUDITORIA DE SENALES REALES — ZimaBlade V6")
    print(f"  {metrics['generated_at']}")
    print("=" * 65)
    print(f"\n  Trades analizados: {metrics['n_trades']}")
    print(f"  Periodo: {metrics['date_range']['from']} → {metrics['date_range']['to']}")

    print("\n  PnL:")
    print(f"    Bruto:           ${s['total_pnl_gross_usd']:+.2f}")
    print(f"    Comisiones:      ${s['total_fees_usd']:.2f} ({s['fee_drag_pct']:.1f}% del PnL bruto)")
    print(f"    Neto (real):     ${s['total_pnl_net_usd']:+.2f}")

    print("\n  Metricas de calidad:")
    print(f"    Win Rate:        {s['win_rate']:.1%}")
    print(f"    Profit Factor:   {s['profit_factor']:.3f}")
    print(f"    Avg Win:         ${s['avg_win_usd']:+.2f}")
    print(f"    Avg Loss:        ${s['avg_loss_usd']:+.2f}")
    print(f"    Max Drawdown:    ${s['max_drawdown_usd']:.2f}")
    print(f"    Expectancy:      ${s['expectancy_per_trade']:+.2f}/trade")

    print("\n  Modelo ML:")
    print(f"    ROC-AUC real:    {ml['roc_auc_real_trades']:.4f}" if ml["roc_auc_real_trades"] else "    ROC-AUC real:    N/A")
    print(f"    Interpretacion:  {ml['interpretation']}")

    print("\n  Slippage:")
    print(f"    Total:           ${slip['total_slippage_usd']:+.2f}")
    print(f"    Por trade:       ${slip['avg_slippage_per_trade_usd']:+.2f}")
    print(f"    % del PnL:       {slip['slippage_pct_of_pnl']:.1f}%")

    print("\n  Por estrategia:")
    for strat, data in metrics["by_strategy"].items():
        print(f"    {strat:20s} trades={data['trades']:3d} WR={data['win_rate']:.0%} PF={data['profit_factor']:.2f} PnL=${data['total_pnl_net']:+.2f}")

    print("\n  Criterios de validacion (DISCLAIMER):")
    for name, crit in val["criteria"].items():
        status = "✓" if crit["passed"] else "✗"
        print(f"    [{status}] {name}: {crit['actual']} (req: {crit['required']})")

    verdict_icon = "✅" if val["ready_to_increase_capital"] else "❌"
    print(f"\n  {verdict_icon} Veredicto: {val['verdict']}")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Auditoria fee-aware de senales reales de paper trading"
    )
    parser.add_argument("--min-trades", type=int, default=5,
                        help="Numero minimo de trades para proceder (default: 5)")
    parser.add_argument("--export", type=str, default=None,
                        help="Exportar resultados a JSON (ej: audit_20260624.json)")
    args = parser.parse_args()

    print("\nCargando trades reales...")

    # Intentar PostgreSQL primero, SQLite como fallback
    df = _load_trades_postgres()
    if df is None:
        print("  PostgreSQL no disponible, intentando SQLite...")
        df = _load_trades_sqlite()

    if df is None or len(df) == 0:
        print("""
  ❌ No hay trades reales disponibles.
  El sistema aun no ha ejecutado trades con is_backtest=FALSE.

  Esto es esperado si el sistema lleva menos de 30 dias en paper trading.
  Revisar con: journalctl -u trading-engine | grep trade_opening_v6
        """)
        sys.exit(0)

    print(f"  {len(df)} trades reales encontrados.")

    if len(df) < args.min_trades:
        print(f"""
  ⚠️  Solo {len(df)} trades disponibles (minimo solicitado: {args.min_trades}).
  Las metricas no tienen significancia estadistica con tan pocos datos.
  Se muestran como referencia pero no deben usarse para tomar decisiones.
        """)

    # Calcular PnL ajustado por fees
    df = compute_fee_adjusted_pnl(df)

    # Calcular metricas
    metrics = compute_metrics(df)

    # Mostrar reporte
    print_report(metrics)

    # Exportar si se solicita
    if args.export:
        export_path = Path(args.export)
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)
        print(f"  Resultados exportados a: {export_path.resolve()}")


if __name__ == "__main__":
    main()
