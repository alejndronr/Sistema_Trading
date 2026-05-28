"""
ml/retrain_model.py — Re-entrenamiento mensual autónomo del MetaLabeler.
=========================================================================
Ejecutado por systemd timer el día 1 de cada mes a las 02:00 AM UTC.

Flujo:
    1. Lock file para evitar ejecuciones paralelas
    2. Cargar 18 meses de trades desde PostgreSQL (tabla trades_journal)
    3. Cargar OHLCV correspondiente desde tabla ohlcv
    4. Aplicar indicadores y señales
    5. Entrenar MetaLabeler
    6. Comparar con modelo anterior → guardar solo si mejora
    7. Notificar por Telegram
    8. Exit 0 si OK, 1 si error crítico

Uso:
    python ml/retrain_model.py
    python ml/retrain_model.py --initial    # primer entrenamiento (sin comparación)
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
import signal
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import aiohttp
import pandas as pd
import structlog
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Aseguramos que el root del proyecto esté en sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.meta_labeler import MetaLabeler, MODEL_VERSION

load_dotenv(PROJECT_ROOT / ".env")

log = structlog.get_logger(__name__)

# ── Constantes ─────────────────────────────────────────────────────────────────
LOCK_FILE = Path("/tmp/retrain.lock")
TIMEOUT_SECONDS = 1800          # 30 minutos máximo
LOOKBACK_MONTHS = 18
MIN_TRADES = 30

# ── Timeout handler ────────────────────────────────────────────────────────────
def _timeout_handler(signum: int, frame: Any) -> None:  # noqa: ANN001
    log.error("retrain_timeout", max_seconds=TIMEOUT_SECONDS)
    sys.exit(1)


# ── Telegram helper (síncrono, sin depender del bot completo) ──────────────────
def _send_telegram(message: str) -> None:
    """Envía un mensaje de Telegram de forma síncrona usando urllib."""
    import urllib.request
    import urllib.parse

    token = os.environ.get("TELEGRAM_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        log.warning("telegram_not_configured_skipping_notify")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = urllib.parse.urlencode({
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown",
    }).encode()

    try:
        req = urllib.request.Request(url, data=payload, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status != 200:
                log.warning("telegram_send_failed", status=resp.status)
    except Exception as exc:
        log.warning("telegram_send_error", error=str(exc))


# ── Carga de datos desde PostgreSQL ────────────────────────────────────────────
def _load_trades_from_db(engine_ignorado, since):
    import pandas as pd
    from sqlalchemy import create_engine
    # Forzamos una conexión síncrona local absoluta
    sync_engine = create_engine("postgresql://trading_user:DS0GCdCE7eBVMypeZJ1Ig@127.0.0.1:5432/trading_db")
    try:
        df = pd.read_sql("SELECT * FROM trades", sync_engine)
        if 'entry_time' in df.columns:
            df['entry_time'] = pd.to_datetime(df['entry_time'])
            if since:
                df = df[df['entry_time'] >= pd.to_datetime(since)]
        return df
    except Exception as e:
        print(f"Error leyendo base de datos (ML): {e}")
        return pd.DataFrame()


def _load_ohlcv_from_db(
    engine: Any,
    symbol: str,
    timeframe: str,
    since: datetime,
) -> pd.DataFrame:
    """
    Carga datos OHLCV desde PostgreSQL.

    Returns:
        DataFrame con columnas: timestamp (int), open, high, low, close, volume.
    """
    query = text("""
        SELECT timestamp, open, high, low, close, volume
        FROM ohlcv
        WHERE symbol = :symbol
          AND timeframe = :timeframe
          AND timestamp >= :since
        ORDER BY timestamp ASC
    """)
    since_ts = int(since.timestamp() * 1000)  # milisegundos (Binance format)
    with engine.connect() as conn:
        result = conn.execute(query, {
            "symbol": symbol,
            "timeframe": timeframe,
            "since": since_ts,
        })
        df = pd.DataFrame(result.fetchall(), columns=result.keys())

    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    log.info(
        "ohlcv_loaded",
        symbol=symbol,
        timeframe=timeframe,
        n_candles=len(df),
    )
    return df


def _load_previous_metrics(model_path: Path) -> Optional[Dict[str, float]]:
    """Carga las métricas del modelo anterior desde model_metadata.json."""
    meta_path = model_path.parent / "model_metadata.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        return meta.get("metrics", None)
    except Exception as exc:
        log.warning("metadata_load_failed", error=str(exc))
        return None


def _model_improved(old_metrics: Dict[str, float], new_metrics: Dict[str, float]) -> bool:
    """
    Compara métricas: el nuevo modelo mejora si tiene mejor F1 en CV.
    Criterio: cv_f1_mean del nuevo >= cv_f1_mean del anterior.
    """
    old_f1 = old_metrics.get("cv_f1_mean", 0.0)
    new_f1 = new_metrics.get("cv_f1_mean", 0.0)
    log.info(
        "model_comparison",
        old_cv_f1=f"{old_f1:.4f}",
        new_cv_f1=f"{new_f1:.4f}",
        improved=new_f1 >= old_f1,
    )
    return new_f1 >= old_f1


# ── Función principal ──────────────────────────────────────────────────────────
def main(initial: bool = False) -> int:
    """
    Punto de entrada del script de reentrenamiento.

    Args:
        initial: si True, omite la comparación con el modelo anterior.

    Returns:
        0 si éxito, 1 si error crítico.
    """
    # ── Registrar timeout SIGALRM ──────────────────────────────────────────────
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(TIMEOUT_SECONDS)

    # ── Obtener lock exclusivo ─────────────────────────────────────────────────
    lock_fd = open(LOCK_FILE, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log.error("retrain_already_running", lock_file=str(LOCK_FILE))
        return 1

    log.info("retrain_started", initial=initial, model_version=MODEL_VERSION)

    database_url = os.environ.get("DATABASE_URL", "")
    if not database_url:
        log.error("DATABASE_URL_not_set")
        _send_telegram("🚨 *Retrain FALLIDO*\n`DATABASE_URL` no configurada.")
        return 1

    try:
        engine = create_engine(database_url, pool_pre_ping=True)

        # ── 1. Determinar ventana temporal ────────────────────────────────────
        since = datetime.now(tz=timezone.utc) - timedelta(days=LOOKBACK_MONTHS * 30)

        # ── 2. Cargar trades ──────────────────────────────────────────────────
        trades_df = _load_trades_from_db(engine, since)

        if len(trades_df) < MIN_TRADES:
            msg = (
                f"⚠️ *Retrain omitido*\n"
                f"Solo {len(trades_df)} trades en los últimos {LOOKBACK_MONTHS} meses "
                f"(mínimo {MIN_TRADES})."
            )
            log.warning("insufficient_trades", n_trades=len(trades_df), min_required=MIN_TRADES)
            _send_telegram(msg)
            return 0  # No es error crítico, simplemente no hay suficientes datos

        trades = trades_df.to_dict("records")

        # ── 3. Cargar OHLCV para BTC/USDT 4H ─────────────────────────────────
        ohlcv_df = _load_ohlcv_from_db(engine, "BTC/USDT", "4h", since)

        if ohlcv_df.empty:
            log.error("ohlcv_empty", symbol="BTC/USDT", timeframe="4h")
            _send_telegram("🚨 *Retrain FALLIDO*\nNo hay datos OHLCV en la BD.")
            return 1

        # ── 4. Aplicar indicadores + señales ──────────────────────────────────
        try:
            from indicators.technical import apply_all_indicators
            from strategies.signals import apply_all_signals

            ohlcv_df = apply_all_indicators(ohlcv_df)
            ohlcv_df = apply_all_signals(ohlcv_df)
            ohlcv_df = ohlcv_df.dropna()
        except ImportError as exc:
            log.error("indicators_import_failed", error=str(exc))
            _send_telegram(f"🚨 *Retrain FALLIDO*\nError importando indicadores: `{exc}`")
            return 1

        # ── 5. Entrenar ───────────────────────────────────────────────────────
        model_path = PROJECT_ROOT / "ml" / "model.joblib"
        labeler = MetaLabeler(model_path=str(model_path))

        # Si hay modelo anterior, guardar backup antes de entrenar
        backup_path = model_path.parent / "model_backup.joblib"
        if model_path.exists():
            import shutil
            shutil.copy2(model_path, backup_path)
            log.info("model_backup_created", backup=str(backup_path))

        old_metrics = _load_previous_metrics(model_path)

        new_metrics = labeler.train(trades, ohlcv_df)

        if not new_metrics.get("trained", False):
            log.warning("training_skipped", reason=new_metrics.get("error", "unknown"))
            _send_telegram(
                f"⚠️ *Retrain omitido*\n{new_metrics.get('error', 'Datos insuficientes')}"
            )
            return 0

        # ── 6. Comparar con modelo anterior ───────────────────────────────────
        if not initial and old_metrics is not None:
            if not _model_improved(old_metrics, new_metrics):
                # Revertir al backup
                if backup_path.exists():
                    import shutil
                    shutil.copy2(backup_path, model_path)
                    log.warning(
                        "model_reverted_to_backup",
                        old_f1=old_metrics.get("cv_f1_mean"),
                        new_f1=new_metrics.get("cv_f1_mean"),
                    )
                msg = (
                    f"⚠️ *Retrain sin mejora — modelo anterior conservado*\n"
                    f"F1 anterior: `{old_metrics.get('cv_f1_mean', '—'):.4f}`\n"
                    f"F1 nuevo:    `{new_metrics.get('cv_f1_mean', '—'):.4f}`\n"
                    f"N samples: `{new_metrics.get('n_samples')}`"
                )
                _send_telegram(msg)
                return 0

        # ── 7. Notificar éxito por Telegram ───────────────────────────────────
        cv_f1 = new_metrics.get("cv_f1_mean", 0)
        cv_acc = new_metrics.get("cv_accuracy_mean", 0)
        n = new_metrics.get("n_samples", 0)
        fi = new_metrics.get("feature_importance", {})
        top_feat = list(fi.keys())[:3]

        msg = (
            f"✅ *Modelo ML re-entrenado*\n"
            f"Fecha: `{datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC`\n"
            f"N trades: `{n}`\n"
            f"CV Accuracy: `{cv_acc:.3f}`\n"
            f"CV F1: `{cv_f1:.3f}`\n"
            f"Top features: `{', '.join(top_feat)}`"
        )
        _send_telegram(msg)
        log.info("retrain_completed", cv_f1=f"{cv_f1:.4f}", n_samples=n)
        return 0

    except Exception as exc:  # Excepción inesperada → error crítico
        log.exception("retrain_critical_error", error=str(exc))
        _send_telegram(f"🚨 *Retrain ERROR CRÍTICO*\n```{exc}```")

        # Intentar restaurar backup si existe
        model_path = PROJECT_ROOT / "ml" / "model.joblib"
        backup_path = model_path.parent / "model_backup.joblib"
        if backup_path.exists() and not model_path.exists():
            import shutil
            shutil.copy2(backup_path, model_path)
            log.info("model_restored_from_backup_after_error")

        return 1

    finally:
        # Liberar lock
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()
            LOCK_FILE.unlink(missing_ok=True)
        except Exception:
            pass
        signal.alarm(0)   # cancelar timeout


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-entrenamiento mensual del MetaLabeler.")
    parser.add_argument(
        "--initial",
        action="store_true",
        help="Primer entrenamiento: omite la comparación con modelo anterior.",
    )
    args = parser.parse_args()
    sys.exit(main(initial=args.initial))
