import os
import sys
import subprocess
from datetime import datetime, timezone
import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.settings import env_settings
from config.logging_config import get_logger

log = get_logger("auto_retrain")

def check_performance_and_retrain():
    # Convertir asyncpg URL a psycopg2 o usar sqlite (asumimos que SQLAlchemy puede conectarse sincronamente para lectura rapida)
    db_url = env_settings.database_url
    sync_url = db_url.replace("+asyncpg", "+psycopg2") if "+asyncpg" in db_url else db_url
    
    try:
        engine = create_engine(sync_url)
        with engine.connect() as conn:
            # ── Phase 7: Retrain Conditionals ──
            try:
                open_pos = conn.execute(text("SELECT count(*) FROM positions WHERE status='open'")).scalar()
                if open_pos and open_pos > 0:
                    log.info("auto_retrain_skipped", reason="open_positions_active")
                    return
            except Exception:
                pass
                
            try:
                last_retrain = conn.execute(text("SELECT max(retrain_date) FROM ml_retrain_log")).scalar()
                if last_retrain:
                    if isinstance(last_retrain, str):
                        last_retrain = pd.to_datetime(last_retrain)
                    if last_retrain.tzinfo is None:
                        last_retrain = last_retrain.replace(tzinfo=timezone.utc)
                    if (datetime.now(timezone.utc) - last_retrain).total_seconds() < 72 * 3600:
                        log.info("auto_retrain_skipped", reason="cooldown_72h_not_met")
                        return
            except Exception:
                pass

            # Traer los ultimos 50 trades para evaluar performance
            query = text("SELECT symbol, pnl_pct, pnl_usd FROM trades_journal ORDER BY exit_time DESC LIMIT 50")
            df = pd.read_sql(query, conn)
    except Exception as e:
        log.error("auto_retrain_db_error", error=str(e))
        return

    if len(df) < 20:
        log.info("auto_retrain_skipped_few_trades", count=len(df))
        return

    wins = len(df[df["pnl_usd"] > 0])
    total = len(df)
    win_rate = wins / total
    
    profit = df[df["pnl_usd"] > 0]["pnl_usd"].sum()
    loss = abs(df[df["pnl_usd"] < 0]["pnl_usd"].sum())
    profit_factor = profit / loss if loss > 0 else 999.0

    log.info("performance_metrics_checked", win_rate=f"{win_rate:.1%}", profit_factor=f"{profit_factor:.2f}", trades=total)

    # Adaptive Thresholds: Si WR baja del 40% o PF baja de 1.05 en una ventana de 50 trades
    if win_rate < 0.40 or profit_factor < 1.05:
        log.warning("performance_degraded_triggering_retrain", win_rate=f"{win_rate:.1%}", profit_factor=f"{profit_factor:.2f}")
        
        try:
            # Ejecutar script de retrain
            result = subprocess.run(
                [sys.executable, "ml/retrain_model.py", "--force"], 
                cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                log.info("auto_retrain_success")
            else:
                log.error("auto_retrain_failed", stderr=result.stderr)
        except Exception as e:
            log.exception("auto_retrain_process_error", error=str(e))
    else:
        log.info("performance_ok_no_retrain_needed")

if __name__ == "__main__":
    check_performance_and_retrain()
