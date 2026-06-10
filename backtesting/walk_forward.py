"""
Walk-Forward Validation para el Sistema de Trading V6
===================================================
Itera ventanas temporales deslizantes sobre el dataset histórico.
Entrena en N meses, prueba en M meses. 
Evalúa consistencia Out-of-Sample de parámetros y/o rentabilidad neta.
"""

import sys
import pandas as pd
import numpy as np
import sqlite3
from datetime import timedelta
from dateutil.relativedelta import relativedelta

sys.path.insert(0, ".")
from backtesting.engine import BacktestEngine
from strategies.signals import TrendFollowingStrategy
from config.logging_config import get_logger

logger = get_logger("walk_forward")

TRAIN_MONTHS = 6
TEST_MONTHS = 3
SYMBOL = "BTC/USDC"
TIMEFRAME = "1h"

def load_data():
    conn = sqlite3.connect("data/db/trading.db")
    query = f"SELECT * FROM ohlcv WHERE symbol='{SYMBOL}' AND timeframe='{TIMEFRAME}' ORDER BY timestamp"
    df = pd.read_sql(query, conn)
    conn.close()
    if df.empty:
        return df
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("datetime", inplace=True)
    return df

def run_walk_forward(auto_mode: bool = False):
    df = load_data()
    if df.empty:
        logger.error("No hay datos en la BD.")
        return

    start_date = df.index.min()
    end_date = df.index.max()
    
    current_train_start = start_date
    
    results = []

    logger.info("="*60)
    logger.info(f" WALK-FORWARD VALIDATION ({SYMBOL})")
    logger.info(f" Train: {TRAIN_MONTHS}m | Test: {TEST_MONTHS}m")
    logger.info("="*60)

    window_idx = 1
    while True:
        train_end = current_train_start + relativedelta(months=TRAIN_MONTHS)
        test_end = train_end + relativedelta(months=TEST_MONTHS)
        
        if test_end > end_date:
            # Ya no hay datos suficientes para un ciclo completo de test
            break

        # Train data (para el ML o grid search, aquí solo reportaremos el performance base para simplificar)
        df_train = df[(df.index >= current_train_start) & (df.index < train_end)].copy()
        df_test = df[(df.index >= train_end) & (df.index < test_end)].copy()
        
        logger.info(f"Window {window_idx}: Train [{current_train_start.date()} a {train_end.date()}] -> Test [{train_end.date()} a {test_end.date()}]")

        # Configurar engine
        engine = BacktestEngine(initial_capital=1000.0)
        strategy = TrendFollowingStrategy(SYMBOL, timeframe=TIMEFRAME)
        
        # Ejecutar Test (Out of Sample)
        # Nota: Normalmente se pasaría un diccionario `best_params` sacado del train
        result = engine.run(symbol=SYMBOL, df=df_test, strategy=strategy, timeframe=TIMEFRAME, show_progress=False)
        m = result.metrics
        
        logger.info(f"   => OOS Resultado | Win Rate: {m.win_rate*100:.1f}% | Profit Factor: {m.profit_factor:.2f} | Total PnL: ${m.total_pnl_usd:.2f} | Max DD: {m.max_drawdown_pct*100:.1f}%")
        
        results.append({
            "window": window_idx,
            "win_rate": m.win_rate,
            "profit_factor": m.profit_factor,
            "pnl": m.total_pnl_usd,
            "max_dd": m.max_drawdown_pct
        })

        # Deslizar ventana temporal
        current_train_start += relativedelta(months=TEST_MONTHS)
        window_idx += 1

    if results:
        res_df = pd.DataFrame(results)
        
        report = []
        report.append("="*60)
        report.append(" RESULTADO CONSOLIDADO OUT-OF-SAMPLE")
        report.append("="*60)
        report.append(f"Promedio Win Rate:      {res_df['win_rate'].mean()*100:.1f}%")
        report.append(f"Promedio Profit Factor: {res_df['profit_factor'].mean():.2f}")
        report.append(f"Suma Total PnL (OOS):   ${res_df['pnl'].sum():.2f}")
        report.append(f"Max Drawdown Promedio:  {res_df['max_dd'].mean()*100:.1f}%")
        
        consistente = (res_df["win_rate"] >= 0.45).mean() >= 0.5
        eval_str = "ESTABLE y CONSISTENTE OK" if consistente else "INESTABLE FAIL"
        report.append(f"\n=> Evaluacion: {eval_str}")
        
        report_text = "\n".join(report)
        for line in report:
            logger.info(line)
            
        if auto_mode:
            import os
            os.makedirs("logs", exist_ok=True)
            report_path = "logs/walk_forward_report.txt"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_text)
            logger.info(f"Reporte auto-generado guardado en {report_path}")
            # Aquí iría la integración con Telegram
            # import requests ...
    else:
        logger.warning("No hubo suficientes ventanas para evaluar.")

if __name__ == "__main__":
    auto = "--auto" in sys.argv
    run_walk_forward(auto_mode=auto)
