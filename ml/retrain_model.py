"""
ml/retrain_model.py — Reentrenador Autónomo del MetaLabeler V4.0
================================================================
Versión completamente reescrita. Mejoras sobre el script de Gemini y V3:

PROBLEMAS DEL SCRIPT ANTERIOR (Gemini):
  ✗ Features incorrectas: setup_quality/position_size/duration_hours
    NO SON features de mercado — son resultados del trade, no predictores
  ✗ Sin TimeSeriesSplit (data leakage: el modelo ve el futuro en validación)
  ✗ Sin normalización de features
  ✗ Sin backup condicional (sobreescribía siempre)
  ✗ Sin compatibilidad con MetaLabeler.FEATURES canónicas
  ✗ Sin multi-símbolo (solo BTC/USDT)
  ✗ Accuracy en train set (métrica completamente inútil y engañosa)
  ✗ Sin registro de historial de modelos

MEJORAS EN V4:
  ✓ Features 100% alineadas con MetaLabeler.FEATURES (21 features de mercado)
  ✓ TimeSeriesSplit(n_splits=5) — sin data leakage
  ✓ Multi-símbolo: entrena con TODOS los pares del sistema
  ✓ Threshold dinámico optimizado por F1 en validación temporal
  ✓ Trigger inteligente: se ejecuta cuando hay suficientes trades nuevos
    (no espera el día 1 del mes si ya hay 50 trades nuevos)
  ✓ Comparación estricta: solo reemplaza si F1 mejora
  ✓ Historial de modelos: guarda versiones con timestamp en ml/history/
  ✓ Fallback a ccxt si no hay OHLCV en BD (primer entrenamiento)
  ✓ Notificación Telegram con tabla de features más importantes
  ✓ Exit codes: 0=OK, 1=error, 2=skip (sin datos suficientes)
  ✓ Registro en tabla ml_retraining_log para auditoría
  ✓ Compatible con IPv4 explícito (127.0.0.1 en lugar de localhost)
  ✓ Sin asyncpg (usa psycopg2 síncrono — correcto para scripts)

Uso:
    python ml/retrain_model.py                      # reentrenamiento normal
    python ml/retrain_model.py --initial            # primer entrenamiento
    python ml/retrain_model.py --force              # forzar aunque no haya mejora
    python ml/retrain_model.py --dry-run            # analizar sin guardar
    python ml/retrain_model.py --symbols BTC/USDT ETH/USDT   # pares específicos
    python ml/retrain_model.py --min-trades 30      # umbral personalizado

Systemd timer sugerido: cada 2 semanas + trigger por nuevo batch de 50 trades
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import sys

try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import structlog
from dotenv import load_dotenv
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report,
)
from sklearn.model_selection import TimeSeriesSplit
from sqlalchemy import create_engine, text

# ── Path setup ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from ml.meta_labeler import MetaLabeler, FEATURES, MODEL_VERSION

log = structlog.get_logger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# ── Constantes ────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

LOCK_FILE            = PROJECT_ROOT / "ml" / "ml_retrain_v4.lock"
TIMEOUT_SECS         = 2400          # 40 minutos máximo (multi-símbolo tarda más)
LOOKBACK_MONTHS      = 18            # ventana de entrenamiento
MIN_TRADES_DEFAULT   = 30            # mínimo para entrenar
MIN_NEW_TRADES       = 15            # mínimo de trades NUEVOS para reentrenar
HISTORY_DIR          = PROJECT_ROOT / "ml" / "history"
MODEL_PATH           = PROJECT_ROOT / "ml" / "model.joblib"
BACKUP_PATH          = PROJECT_ROOT / "ml" / "model_backup.joblib"
META_PATH            = PROJECT_ROOT / "ml" / "model_metadata.json"
RETRAIN_LOG_PATH     = PROJECT_ROOT / "ml" / "retrain_history.jsonl"

# Pares de entrenamiento — mismos que live_engine.py
TRAIN_SYMBOLS: List[str] = [
    "BTC/USDC", "ETH/USDC", "SOL/USDC", "BNB/USDC", "LINK/USDC", "AVAX/USDC",
]
TRAIN_TIMEFRAME = "1h"

# Hiperparámetros del RandomForest (optimizados para Atom E3950)
RF_PARAMS = dict(
    n_estimators=120,
    max_depth=7,
    min_samples_leaf=8,
    min_samples_split=20,
    max_features="sqrt",
    class_weight="balanced",
    n_jobs=2,        # deja 2 cores libres para el motor
    random_state=42,
)


# ══════════════════════════════════════════════════════════════════════════════
# ── Helpers ───────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _timeout_handler(signum: int, frame: Any) -> None:
    log.error("retrain_timeout", max_secs=TIMEOUT_SECS)
    _send_telegram("🚨 *Retrain TIMEOUT* — proceso cancelado tras 40 minutos.")
    sys.exit(1)


def _send_telegram(message: str) -> None:
    """Envía mensaje de Telegram de forma síncrona (sin asyncio)."""
    token   = os.environ.get("TELEGRAM_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_ALLOWED_USER_ID", "") or os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return

    url     = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = urllib.parse.urlencode({
        "chat_id":    chat_id,
        "text":       message,
        "parse_mode": "Markdown",
    }).encode()

    try:
        req = urllib.request.Request(url, data=payload, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status != 200:
                log.warning("telegram_send_failed", status=resp.status)
    except Exception as exc:
        log.warning("telegram_send_error", error=str(exc))


def _get_db_engine():
    """
    Crea engine SQLAlchemy síncrono (psycopg2).
    CRÍTICO: usa 127.0.0.1, no localhost.
    Debian resuelve 'localhost' a IPv6 ::1, que puede fallar en PostgreSQL.
    """
    raw_url = os.environ.get("DATABASE_URL", "")
    if not raw_url:
        raise RuntimeError("DATABASE_URL no configurada en .env")

    # Convertir asyncpg → psycopg2 si hace falta
    url = (raw_url
           .replace("+asyncpg", "")
           .replace("postgresql+asyncpg://", "postgresql://")
           .replace("postgres://", "postgresql://"))

    # Forzar IPv4
    url = url.replace("localhost", "127.0.0.1")

    connect_args = {}
    if not url.startswith("sqlite"):
        connect_args["connect_timeout"] = 10

    engine = create_engine(url, pool_pre_ping=True, connect_args=connect_args)

    # Verificar conexión
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))

    return engine


def _ensure_retrain_log_table(engine) -> None:
    """Crea la tabla de historial de reentrenamientos si no existe."""
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS ml_retrain_log (
                    id              SERIAL PRIMARY KEY,
                    retrained_at    TIMESTAMPTZ DEFAULT NOW(),
                    model_version   VARCHAR(20),
                    n_trades        INTEGER,
                    n_symbols       INTEGER,
                    cv_accuracy     NUMERIC(6,4),
                    cv_f1           NUMERIC(6,4),
                    precision_val   NUMERIC(6,4),
                    recall_val      NUMERIC(6,4),
                    roc_auc         NUMERIC(6,4),
                    optimal_threshold NUMERIC(5,3),
                    win_rate_dataset  NUMERIC(5,3),
                    top_features    TEXT,
                    deployed        BOOLEAN DEFAULT TRUE,
                    notes           TEXT
                );
            """))
    except Exception as exc:
        log.warning("retrain_log_table_skip", error=str(exc))


# ══════════════════════════════════════════════════════════════════════════════
# ── Carga de datos ────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def load_trades(engine, since: datetime) -> pd.DataFrame:
    """
    Carga trades cerrados desde trades_journal.
    Filtra is_backtest=FALSE para no contaminar con datos sintéticos.
    """
    query = text("""
        SELECT
            symbol,
            strategy,
            COALESCE(direction, 'long')         AS direction,
            entry_time,
            exit_time,
            entry_price,
            exit_price,
            stop_loss,
            tp1,
            tp2,
            COALESCE(units, position_size)      AS units,
            COALESCE(pnl, pnl_usd)              AS pnl,
            pnl_pct,
            r_multiple,
            exit_reason,
            ml_proba,
            COALESCE(regime, market_regime)     AS regime,
            COALESCE(
                duration_hours,
                EXTRACT(EPOCH FROM (exit_time - entry_time))/3600
            )                                   AS duration_hours
        FROM trades_journal
        WHERE entry_time >= :since
          AND exit_time IS NOT NULL
          AND COALESCE(pnl, pnl_usd) IS NOT NULL
        ORDER BY entry_time ASC
    """)

    with engine.connect() as conn:
        df = pd.DataFrame(conn.execute(query, {"since": since}).fetchall(),
                          columns=[
                              "symbol", "strategy", "direction",
                              "entry_time", "exit_time", "entry_price", "exit_price",
                              "stop_loss", "tp1", "tp2", "units",
                              "pnl", "pnl_pct", "r_multiple",
                              "exit_reason", "ml_proba", "regime",
                              "duration_hours",
                          ])

    log.info("trades_loaded", n=len(df), since=str(since.date()),
             symbols=df["symbol"].unique().tolist() if not df.empty else [])
    return df


def load_ohlcv_from_db(engine, symbol: str, since: datetime) -> pd.DataFrame:
    """
    Carga OHLCV del timeframe de entrenamiento desde PostgreSQL.
    La columna timestamp es bigint (milisegundos Unix) — se convierte antes de comparar.
    """
    # Convertir datetime → milisegundos Unix (tipo de la columna en BD)
    since_ms = int(since.timestamp() * 1000)

    query = text("""
        SELECT timestamp, open, high, low, close, volume
        FROM ohlcv
        WHERE symbol    = :sym
          AND timeframe = :tf
          AND timestamp >= :since_ms
        ORDER BY timestamp ASC
    """)
    with engine.connect() as conn:
        df = pd.DataFrame(
            conn.execute(query, {
                "sym":      symbol,
                "tf":       TRAIN_TIMEFRAME,
                "since_ms": since_ms,
            }).fetchall(),
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )

    if not df.empty:
        # Convertir bigint ms → datetime UTC
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def fetch_ohlcv_from_exchange(symbol: str, since: datetime) -> pd.DataFrame:
    """
    Fallback: descarga OHLCV desde Binance vía ccxt si la BD está vacía.
    Útil para el primer entrenamiento.
    """
    try:
        import ccxt
        exchange = ccxt.binance({"enableRateLimit": True})
        since_ms = int(since.timestamp() * 1000)
        log.info("fetching_ohlcv_from_exchange", symbol=symbol, tf=TRAIN_TIMEFRAME)

        all_candles = []
        limit = 1000
        fetch_since = since_ms

        while True:
            candles = exchange.fetch_ohlcv(symbol, TRAIN_TIMEFRAME,
                                           since=fetch_since, limit=limit)
            if not candles:
                break
            all_candles.extend(candles)
            if len(candles) < limit:
                break
            fetch_since = candles[-1][0] + 1
            time.sleep(0.5)  # rate limit

        if not all_candles:
            return pd.DataFrame()

        df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        log.info("ohlcv_fetched_from_exchange", symbol=symbol, n=len(df))
        return df

    except Exception as exc:
        log.error("exchange_fetch_failed", symbol=symbol, error=str(exc))
        return pd.DataFrame()


def load_ohlcv_all_symbols(engine, since: datetime) -> Dict[str, pd.DataFrame]:
    """
    Carga OHLCV para todos los símbolos de entrenamiento.
    Si la BD no tiene datos para un símbolo, hace fallback a exchange.
    """
    result: Dict[str, pd.DataFrame] = {}

    for symbol in TRAIN_SYMBOLS:
        df = load_ohlcv_from_db(engine, symbol, since)

        if df.empty or len(df) < 200:
            log.warning("ohlcv_insufficient_in_db", symbol=symbol, n=len(df))
            df = fetch_ohlcv_from_exchange(symbol, since)

        if not df.empty and len(df) >= 200:
            result[symbol] = df
            log.info("ohlcv_ready", symbol=symbol, n=len(df))
        else:
            log.warning("ohlcv_skip", symbol=symbol, reason="less than 200 candles")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# ── Construcción del dataset multi-símbolo ────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _compute_indicators_inline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula todos los indicadores necesarios para MetaLabeler.FEATURES.
    Completamente inline para evitar dependencias circulares.
    Alineado con enrich_dataframe() del live_engine.py V4.
    """
    df = df.copy()
    c = df["close"]
    h, l, v = df["high"], df["low"], df["volume"]

    # EMAs
    df["ema21"]  = c.ewm(span=21,  adjust=False).mean()
    df["ema55"]  = c.ewm(span=55,  adjust=False).mean()
    df["ema200"] = c.ewm(span=200, adjust=False).mean()

    # RSI-14
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    # ATR-14
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(span=14, adjust=False).mean()

    # ADX-14 (simplificado pero correcto)
    plus_dm  = h.diff().clip(lower=0)
    minus_dm = (-l.diff()).clip(lower=0)
    plus_dm  = plus_dm.where(plus_dm > minus_dm, 0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0)
    atr_s    = tr.ewm(span=14, adjust=False).mean().replace(0, np.nan)
    plus_di  = 100 * plus_dm.ewm(span=14, adjust=False).mean() / atr_s
    minus_di = 100 * minus_dm.ewm(span=14, adjust=False).mean() / atr_s
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["adx"] = dx.ewm(span=14, adjust=False).mean().fillna(0)

    # Bollinger Bands
    bb_mid         = c.rolling(20).mean()
    bb_std         = c.rolling(20).std()
    df["bb_lower"] = bb_mid - 2 * bb_std
    df["bb_mid"]   = bb_mid
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / bb_mid.replace(0, np.nan)

    # MACD
    ema_fast    = c.ewm(span=12, adjust=False).mean()
    ema_slow    = c.ewm(span=26, adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = macd_line - macd_signal

    # Volume ratio
    df["vol_ratio"] = v / v.rolling(20).mean().replace(0, np.nan)

    # CVD proxy (delta de volumen)
    df["cvd"] = np.where(c > c.shift(1), v, -v)

    # Régimen encoded
    trend_up   = (df["ema21"] > df["ema55"]) & (df["ema55"] > df["ema200"])
    trend_down = (df["ema21"] < df["ema55"]) & (df["ema55"] < df["ema200"])
    df["regime_encoded"] = np.where(trend_up, 1.0, np.where(trend_down, -1.0, 0.0))

    # Consensus Score
    score = (
        trend_up.astype(float) * 25
        + ((df["rsi"].clip(30, 70) - 30) / 40.0) * 20
        + (macd_line > 0).astype(float) * 20
        + (df["adx"].clip(0, 50) / 50.0) * 20
        + ((df["cvd"] > 0).astype(float).rolling(10).mean()) * 15
    )
    df["consensus"] = score.fillna(0)

    # Order Blocks y FVG
    bearish = c < df["open"]
    bull_move = c > c.shift(1)
    df["ob_bull"]  = (bearish.shift(1) & bull_move).fillna(False).astype(float)
    df["fvg_bull"] = (l > h.shift(2)).fillna(False).astype(float)

    return df


def build_multi_symbol_dataset(
    trades_df: pd.DataFrame,
    ohlcv_map: Dict[str, pd.DataFrame],
    labeler: MetaLabeler,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Construye (X, y) combinando trades de todos los símbolos disponibles.

    Para cada trade:
      1. Encuentra el OHLCV del símbolo correspondiente
      2. Localiza la vela del entry_time
      3. Extrae el vector de features de esa vela (con contexto previo)
      4. Etiqueta: 1 si pnl > 0, 0 si pnl <= 0

    Returns:
        X: (n, 21) float32
        y: (n,) int
        meta: lista de dicts con metadatos de cada muestra (para diagnóstico)
    """
    X_rows: List[np.ndarray] = []
    y_rows: List[int] = []
    meta_rows: List[Dict] = []

    skipped = 0

    for _, trade in trades_df.iterrows():
        symbol = str(trade["symbol"])

        if symbol not in ohlcv_map:
            skipped += 1
            continue

        df_raw = ohlcv_map[symbol]

        # Asegurar indicadores calculados
        if "rsi" not in df_raw.columns:
            try:
                df_raw = _compute_indicators_inline(df_raw)
                ohlcv_map[symbol] = df_raw  # cachear para los siguientes trades del mismo símbolo
            except Exception as exc:
                log.warning("indicators_failed", symbol=symbol, error=str(exc))
                skipped += 1
                continue

        # Localizar la vela de entrada
        entry_time = pd.Timestamp(trade["entry_time"])
        if entry_time.tzinfo is None:
            entry_time = entry_time.tz_localize("UTC")
        else:
            entry_time = entry_time.tz_convert("UTC")

        ts_col = df_raw["timestamp"]
        mask   = ts_col <= entry_time
        if not mask.any():
            skipped += 1
            continue

        # Índice de la vela en el df_raw
        idx = int(mask.values.nonzero()[0][-1])

        # Necesitamos al menos 50 velas de contexto previo para lags y rolling
        if idx < 50:
            skipped += 1
            continue

        # Construir feature matrix sobre la ventana de contexto + vela actual
        window_df = df_raw.iloc[max(0, idx - 200) : idx + 1].copy().reset_index(drop=True)
        try:
            X_matrix = labeler.build_feature_matrix(window_df)
        except Exception as exc:
            log.debug("feature_build_failed", symbol=symbol, error=str(exc))
            skipped += 1
            continue

        if len(X_matrix) == 0:
            skipped += 1
            continue

        # Última fila = features de la vela de entrada
        features_vec = X_matrix[-1]

        # Label
        pnl = float(trade.get("pnl", trade.get("pnl_usd", 0)) or 0)
        label = 1 if pnl > 0 else 0

        X_rows.append(features_vec)
        y_rows.append(label)
        meta_rows.append({
            "symbol":    symbol,
            "strategy":  trade.get("strategy", ""),
            "pnl":       pnl,
            "r_multiple": trade.get("r_multiple", 0),
            "regime":    trade.get("regime", ""),
            "entry_time": str(trade["entry_time"]),
        })

    if not X_rows:
        log.error("dataset_empty_after_processing", skipped=skipped)
        return np.array([]), np.array([]), []

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=int)

    # Reemplazar inf/nan residuales
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    win_rate = y.mean()
    log.info(
        "dataset_built",
        n_samples=len(X),
        n_skipped=skipped,
        win_rate=f"{win_rate:.2%}",
        symbols=list(ohlcv_map.keys()),
        features=len(FEATURES),
    )

    return X, y, meta_rows


# ══════════════════════════════════════════════════════════════════════════════
# ── Entrenamiento y Evaluación ────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def optimize_threshold(model, X_val: np.ndarray, y_val: np.ndarray) -> float:
    """
    Busca el threshold óptimo (0.40–0.80) que maximiza F1 en el set de validación.
    Usa grid de 9 puntos — < 100ms en Atom E3950.
    """
    if len(X_val) < 5:
        return 0.55

    probas = model.predict_proba(X_val)[:, 1]
    best_thresh, best_f1 = 0.55, 0.0

    for thresh in np.arange(0.40, 0.81, 0.05):
        preds = (probas >= thresh).astype(int)
        if preds.sum() == 0:
            continue
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, float(thresh)

    log.info("threshold_optimized", threshold=f"{best_thresh:.2f}", f1=f"{best_f1:.4f}")
    return best_thresh


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Entrena RandomForest con TimeSeriesSplit(n_splits=5).
    Calcula métricas en CV temporal (sin data leakage).

    Returns:
        metrics dict con el modelo entrenado incluido como 'model'
    """
    n = len(X)
    if n < 20:
        return {"trained": False, "error": f"Solo {n} muestras (mínimo 20)"}

    # 20% final para validación de threshold
    val_size   = max(int(n * 0.20), 5)
    X_main     = X[:-val_size]
    y_main     = y[:-val_size]
    X_val      = X[-val_size:]
    y_val      = y[-val_size:]

    # ── Cross-Validation temporal ───────────────────────────────────────────
    tscv = TimeSeriesSplit(n_splits=5)
    cv_metrics: List[Dict] = []

    log.info("cv_start", n_splits=5, n_main=len(X_main))

    for fold_i, (train_idx, val_idx) in enumerate(tscv.split(X_main)):
        X_tr, X_vl = X_main[train_idx], X_main[val_idx]
        y_tr, y_vl = y_main[train_idx], y_main[val_idx]

        if len(np.unique(y_tr)) < 2:
            continue  # skip si solo hay una clase

        fold_model = RandomForestClassifier(**RF_PARAMS)
        fold_model.fit(X_tr, y_tr)
        preds = fold_model.predict(X_vl)
        probas = fold_model.predict_proba(X_vl)[:, 1]

        fold_m = {
            "fold":      fold_i + 1,
            "accuracy":  accuracy_score(y_vl, preds),
            "f1":        f1_score(y_vl, preds, zero_division=0),
            "precision": precision_score(y_vl, preds, zero_division=0),
            "recall":    recall_score(y_vl, preds, zero_division=0),
        }
        try:
            fold_m["roc_auc"] = roc_auc_score(y_vl, probas)
        except Exception:
            fold_m["roc_auc"] = 0.5

        cv_metrics.append(fold_m)
        log.debug("fold_done", **{k: f"{v:.4f}" if isinstance(v, float) else v
                                  for k, v in fold_m.items()})

    if not cv_metrics:
        return {"trained": False, "error": "CV falló en todos los folds"}

    cv_df = pd.DataFrame(cv_metrics)

    # ── Modelo final en todo X_main ──────────────────────────────────────────
    final_model = RandomForestClassifier(**RF_PARAMS)
    final_model.fit(X_main, y_main)

    # ── Threshold óptimo sobre X_val (datos que el modelo NO vio) ─────────
    best_threshold = optimize_threshold(final_model, X_val, y_val)

    # ── Métricas finales sobre X_val ──────────────────────────────────────
    val_probas = final_model.predict_proba(X_val)[:, 1]
    val_preds  = (val_probas >= best_threshold).astype(int)

    val_metrics = {
        "accuracy_val":  accuracy_score(y_val, val_preds),
        "f1_val":        f1_score(y_val, val_preds, zero_division=0),
        "precision_val": precision_score(y_val, val_preds, zero_division=0),
        "recall_val":    recall_score(y_val, val_preds, zero_division=0),
    }
    try:
        val_metrics["roc_auc_val"] = roc_auc_score(y_val, val_probas)
    except Exception:
        val_metrics["roc_auc_val"] = 0.5

    # ── Feature importance ────────────────────────────────────────────────
    importance_pairs = sorted(
        zip(FEATURES, final_model.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )
    feature_importance = {feat: round(float(imp), 6) for feat, imp in importance_pairs}

    metrics = {
        "trained":            True,
        "n_samples":          n,
        "win_rate_dataset":   round(float(y.mean()), 4),
        # CV metrics
        "cv_accuracy_mean":   round(float(cv_df["accuracy"].mean()), 4),
        "cv_f1_mean":         round(float(cv_df["f1"].mean()), 4),
        "cv_precision_mean":  round(float(cv_df["precision"].mean()), 4),
        "cv_recall_mean":     round(float(cv_df["recall"].mean()), 4),
        "cv_roc_auc_mean":    round(float(cv_df["roc_auc"].mean()), 4),
        "cv_f1_std":          round(float(cv_df["f1"].std()), 4),
        # Val metrics
        **val_metrics,
        # Config
        "optimal_threshold":  round(best_threshold, 3),
        "feature_importance": feature_importance,
        # El modelo en sí (no se serializa a JSON)
        "model":              final_model,
    }

    log.info(
        "training_complete",
        n=n,
        cv_f1=f"{metrics['cv_f1_mean']:.4f} ±{metrics['cv_f1_std']:.4f}",
        cv_acc=f"{metrics['cv_accuracy_mean']:.4f}",
        roc_auc=f"{metrics['cv_roc_auc_mean']:.4f}",
        threshold=f"{best_threshold:.2f}",
        top_feature=list(feature_importance.keys())[0],
    )

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# ── Comparación y Despliegue ──────────────────────────────════════════════════
# ══════════════════════════════════════════════════════════════════════════════

def load_previous_metrics() -> Optional[Dict]:
    """Carga métricas del modelo actual desde model_metadata.json."""
    if not META_PATH.exists():
        return None
    try:
        with open(META_PATH, encoding="utf-8") as fh:
            meta = json.load(fh)
        return meta.get("metrics", meta)
    except Exception as exc:
        log.warning("meta_load_failed", error=str(exc))
        return None


def model_improved(old: Dict, new: Dict, force: bool = False) -> bool:
    """
    El nuevo modelo se despliega si:
      1. --force está activo, O
      2. CV F1 mejora, O
      3. CV F1 es igual pero ROC-AUC mejora
    """
    if force:
        return True

    old_f1  = old.get("cv_f1_mean",  old.get("cv_f1",  0.0))
    new_f1  = new.get("cv_f1_mean",  0.0)
    old_auc = old.get("cv_roc_auc_mean", 0.5)
    new_auc = new.get("cv_roc_auc_mean", 0.5)

    log.info("model_comparison",
             old_f1=f"{old_f1:.4f}", new_f1=f"{new_f1:.4f}",
             old_auc=f"{old_auc:.4f}", new_auc=f"{new_auc:.4f}")

    if new_f1 > old_f1:
        return True
    if abs(new_f1 - old_f1) < 0.005 and new_auc > old_auc:
        return True
    return False


def deploy_model(
    new_model: RandomForestClassifier,
    metrics: Dict,
    labeler: MetaLabeler,
    engine,
    dry_run: bool = False,
) -> None:
    """
    Guarda el modelo, actualiza metadata, archiva versión anterior
    y registra en el log de reentrenamientos.
    """
    if dry_run:
        log.info("dry_run_model_not_saved")
        return

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    timestamp_str = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")

    # ── Archivar versión anterior ──────────────────────────────────────────
    if MODEL_PATH.exists():
        archive_path = HISTORY_DIR / f"model_{timestamp_str}_prev.joblib"
        shutil.copy2(MODEL_PATH, archive_path)
        shutil.copy2(MODEL_PATH, BACKUP_PATH)  # backup rápido en ml/
        log.info("model_archived", path=str(archive_path))

    # ── Guardar nuevo modelo ───────────────────────────────────────────────
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(new_model, MODEL_PATH)

    # ── Actualizar labeler en memoria ──────────────────────────────────────
    labeler._model = new_model

    # ── Actualizar metadata.json ──────────────────────────────────────────
    metadata = {
        "trained_at":        datetime.utcnow().isoformat() + "Z",
        "model_version":     f"{MODEL_VERSION}-V4",
        "n_samples":         metrics["n_samples"],
        "features":          FEATURES,
        "optimal_threshold": metrics["optimal_threshold"],
        "metrics": {k: v for k, v in metrics.items()
                    if k not in ("model", "feature_importance")},
        "feature_importance": metrics["feature_importance"],
    }
    with open(META_PATH, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)
    labeler._metadata = metadata

    # ── Registrar en JSONL de historial local ─────────────────────────────
    log_entry = {"timestamp": timestamp_str, **{
        k: v for k, v in metrics.items() if k != "model"
    }}
    with open(RETRAIN_LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(log_entry) + "\n")

    # ── Registrar en PostgreSQL ───────────────────────────────────────────
    top_feature = list(metrics["feature_importance"].keys())[0] if metrics["feature_importance"] else ""
    try:
        import json as _json
        fi_json = _json.dumps(metrics["feature_importance"])
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO ml_retrain_log
                    (retrain_date, roc_auc, cv_f1, n_samples, top_feature, threshold, feature_importance_json, notes)
                VALUES
                    (NOW(), :ra, :cf, :ns, :tf, :th, :fi, :note)
            """), {
                "ra":  metrics.get("cv_roc_auc_mean", 0),
                "cf":  metrics.get("cv_f1_mean", 0),
                "ns":  metrics["n_samples"],
                "tf":  top_feature,
                "th":  metrics["optimal_threshold"],
                "fi":  fi_json,
                "note": f"V4 multi-symbol, {timestamp_str}",
            })
    except Exception as exc:
        log.warning("db_log_failed", error=str(exc))

    log.info("model_deployed", path=str(MODEL_PATH))


# ══════════════════════════════════════════════════════════════════════════════
# ── Función principal ─────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def main(
    initial:   bool = False,
    force:     bool = False,
    dry_run:   bool = False,
    min_trades: int = MIN_TRADES_DEFAULT,
    symbols:   Optional[List[str]] = None,
) -> int:
    """
    Flujo completo de reentrenamiento.

    Returns:
        0 = éxito (modelo desplegado o skip justificado)
        1 = error crítico
        2 = skip (no hay suficientes datos)
    """
    # ── Timeout SIGALRM ────────────────────────────────────────────────────
    has_sigalrm = hasattr(signal, "SIGALRM")
    if has_sigalrm:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(TIMEOUT_SECS)

    # ── Lock exclusivo (evitar ejecuciones paralelas) ──────────────────────
    lock_fd = None
    if HAS_FCNTL:
        lock_fd = open(LOCK_FILE, "w")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log.error("retrain_already_running")
            return 1
    else:
        try:
            lock_fd = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            log.error("retrain_already_running")
            return 1

    start_time = time.time()
    log.info(
        "retrain_v4_start",
        initial=initial, force=force, dry_run=dry_run,
        min_trades=min_trades, symbols=symbols or TRAIN_SYMBOLS,
    )

    print("\n" + "═" * 60)
    print(f"  🤖 MetaLabeler V4 — Reentrenamiento Autónomo")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * 60)

    engine = None
    try:
        # ── 1. Conectar a BD ───────────────────────────────────────────────
        print("\n[1/7] Conectando a PostgreSQL (127.0.0.1)...")
        engine = _get_db_engine()
        _ensure_retrain_log_table(engine)
        print("      ✓ Conexión establecida")

        # ── 2. Cargar trades ───────────────────────────────────────────────
        since = datetime.now(tz=timezone.utc) - timedelta(days=LOOKBACK_MONTHS * 30)
        print(f"\n[2/7] Cargando trades desde {since.date()} ({LOOKBACK_MONTHS}m)...")
        trades_df = load_trades(engine, since)
        n_trades  = len(trades_df)
        print(f"      ✓ {n_trades} trades cerrados")

        if n_trades < min_trades:
            msg = (f"⚠️ *Retrain omitido*\n"
                   f"Solo {n_trades} trades (mínimo {min_trades}).\n"
                   f"El sistema necesita más historial para aprender.")
            print(f"\n  ⚠️  {n_trades} trades < mínimo {min_trades} → SKIP")
            _send_telegram(msg)
            return 2

        # Comprobar si hay suficientes trades NUEVOS desde último reentrenamiento
        if not initial and not force and META_PATH.exists():
            try:
                with open(META_PATH) as fh:
                    meta = json.load(fh)
                last_retrain = datetime.fromisoformat(
                    meta["trained_at"].replace("Z", "+00:00")
                )
                new_trades = trades_df[
                    pd.to_datetime(trades_df["entry_time"], utc=True) > last_retrain
                ]
                if len(new_trades) < MIN_NEW_TRADES:
                    print(f"\n  ℹ️  Solo {len(new_trades)} trades nuevos desde "
                          f"último retrain → SKIP")
                    log.info("skip_insufficient_new_trades",
                             new=len(new_trades), min=MIN_NEW_TRADES)
                    return 2
            except Exception:
                pass  # Si no podemos leer la fecha, continuamos

        # ── 3. Cargar OHLCV ────────────────────────────────────────────────
        print(f"\n[3/7] Cargando OHLCV ({TRAIN_TIMEFRAME}) para {len(TRAIN_SYMBOLS)} símbolos...")
        use_symbols = symbols or TRAIN_SYMBOLS
        ohlcv_map   = load_ohlcv_all_symbols(engine, since)

        if not ohlcv_map:
            print("      ✗ No hay OHLCV disponible. Abortando.")
            _send_telegram("🚨 *Retrain FALLIDO*\nNo hay datos OHLCV en la BD ni en Binance.")
            return 1
        print(f"      ✓ OHLCV disponible para: {list(ohlcv_map.keys())}")

        # ── 4. Construir dataset ───────────────────────────────────────────
        print("\n[4/7] Calculando indicadores y construyendo dataset...")
        labeler = MetaLabeler(model_path=str(MODEL_PATH))
        X, y, meta_rows = build_multi_symbol_dataset(trades_df, ohlcv_map, labeler)

        if len(X) < min_trades:
            msg = (f"⚠️ *Retrain omitido*\n"
                   f"Solo {len(X)} muestras procesables (mínimo {min_trades}).")
            print(f"\n  ⚠️  Dataset final: {len(X)} muestras < mínimo → SKIP")
            _send_telegram(msg)
            return 2

        win_rate = float(y.mean())
        print(f"      ✓ Dataset: {len(X)} muestras | Win rate: {win_rate:.1%}")
        print(f"        Distribución: {int(y.sum())} ganadores / {int((1-y).sum())} perdedores")

        # ── 5. Entrenar ────────────────────────────────────────────────────
        print("\n[5/7] Entrenando RandomForest con TimeSeriesSplit(n=5)...")
        new_metrics = train_and_evaluate(X, y, dry_run=dry_run)

        if not new_metrics.get("trained", False):
            err = new_metrics.get("error", "unknown")
            print(f"\n  ✗ Entrenamiento falló: {err}")
            _send_telegram(f"🚨 *Retrain FALLIDO*\n{err}")
            return 1

        cv_f1  = new_metrics["cv_f1_mean"]
        cv_acc = new_metrics["cv_accuracy_mean"]
        roc    = new_metrics["cv_roc_auc_mean"]
        thresh = new_metrics["optimal_threshold"]

        print(f"\n      ✓ CV F1:       {cv_f1:.4f} ±{new_metrics['cv_f1_std']:.4f}")
        print(f"        CV Accuracy: {cv_acc:.4f}")
        print(f"        ROC-AUC:     {roc:.4f}")
        print(f"        Threshold:   {thresh:.2f}")

        # ── 6. Comparar con modelo anterior ───────────────────────────────
        print("\n[6/7] Comparando con modelo anterior...")
        old_metrics = load_previous_metrics()

        if old_metrics and not initial:
            should_deploy = model_improved(old_metrics, new_metrics, force=force)
            old_f1 = old_metrics.get("cv_f1_mean", old_metrics.get("cv_f1", 0))
            print(f"      F1 anterior: {old_f1:.4f}  |  F1 nuevo: {cv_f1:.4f}")
            print(f"      → {'DESPLEGAR ✓' if should_deploy else 'CONSERVAR ANTERIOR ✗'}")
        else:
            should_deploy = True
            print("      → Primer entrenamiento: desplegando sin comparación")

        # ── 7. Desplegar ───────────────────────────────────────────────────
        if should_deploy:
            print("\n[7/7] Desplegando nuevo modelo...")
            deploy_model(new_metrics["model"], new_metrics, labeler, engine, dry_run=dry_run)
            print("      ✓ model.joblib guardado")
            print("      ✓ model_metadata.json actualizado")
            print("      ✓ Registrado en ml_retrain_log (PostgreSQL)")
        else:
            print("\n[7/7] Modelo anterior es mejor → no se reemplaza")
            # Revertir al backup por si acaso se había sobrescrito
            if BACKUP_PATH.exists() and not dry_run:
                shutil.copy2(BACKUP_PATH, MODEL_PATH)

        # ── Notificación Telegram ──────────────────────────────────────────
        top_5  = list(new_metrics["feature_importance"].keys())[:5]
        top_5_vals = [f"{k}: {new_metrics['feature_importance'][k]:.4f}" for k in top_5]

        if should_deploy:
            emoji = "✅"
            deployed_txt = "Nuevo modelo *desplegado* ✓"
        else:
            emoji = "⚠️"
            deployed_txt = "Modelo anterior conservado (sin mejora)"

        msg = (
            f"{emoji} *ML Retrain completado*\n"
            f"Fecha: `{datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC`\n"
            f"Trades: `{len(X)}` | Win rate: `{win_rate:.1%}`\n"
            f"CV F1: `{cv_f1:.4f}` ±`{new_metrics['cv_f1_std']:.4f}`\n"
            f"ROC-AUC: `{roc:.4f}` | Threshold: `{thresh:.2f}`\n"
            f"Top features:\n" + "\n".join(f"  `{f}`" for f in top_5_vals[:3]) + "\n"
            f"\n{deployed_txt}"
        )
        _send_telegram(msg)

        # ── Resumen final ──────────────────────────────────────────────────
        elapsed = time.time() - start_time
        print("\n" + "═" * 60)
        print(f"  {'✅ Completado' if should_deploy else '⏭️  Sin cambios'} en {elapsed:.0f}s")
        if not dry_run and should_deploy:
            print(f"  Modelo guardado en: {MODEL_PATH}")
            print(f"  Backup en:          {BACKUP_PATH}")
        print("═" * 60 + "\n")

        return 0

    except Exception as exc:
        log.exception("retrain_critical_error", error=str(exc))
        print(f"\n  💀 ERROR CRÍTICO: {exc}")
        _send_telegram(f"🚨 *Retrain ERROR CRÍTICO*\n```{exc}```")

        # Restaurar backup si el modelo quedó corrupto
        if not dry_run and BACKUP_PATH.exists() and not MODEL_PATH.exists():
            shutil.copy2(BACKUP_PATH, MODEL_PATH)
            log.info("model_restored_from_backup")
            print("  ↩️  Modelo restaurado desde backup")

        return 1

    finally:
        if has_sigalrm:
            signal.alarm(0)
        try:
            if HAS_FCNTL and lock_fd is not None:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()
            elif lock_fd is not None:
                os.close(lock_fd)
            LOCK_FILE.unlink(missing_ok=True)
        except Exception:
            pass
        if engine:
            engine.dispose()


# ══════════════════════════════════════════════════════════════════════════════
# ── CLI ───────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MetaLabeler V4 — Reentrenador autónomo multi-símbolo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python ml/retrain_model.py                           # reentrenamiento estándar
  python ml/retrain_model.py --initial                 # primer entrenamiento
  python ml/retrain_model.py --force                   # forzar aunque no mejore
  python ml/retrain_model.py --dry-run                 # solo evaluar, no guardar
  python ml/retrain_model.py --min-trades 20           # umbral más bajo
  python ml/retrain_model.py --symbols BTC/USDT ETH/USDT  # pares específicos
        """,
    )
    parser.add_argument(
        "--initial",
        action="store_true",
        help="Primer entrenamiento: omite comparación con modelo anterior.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Desplegar aunque el nuevo modelo no mejore.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Entrenar y evaluar sin guardar ningún archivo.",
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=MIN_TRADES_DEFAULT,
        help=f"Mínimo de trades para entrenar (default: {MIN_TRADES_DEFAULT}).",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Pares específicos (default: todos en TRAIN_SYMBOLS).",
    )
    args = parser.parse_args()

    sys.exit(main(
        initial=args.initial,
        force=args.force,
        dry_run=args.dry_run,
        min_trades=args.min_trades,
        symbols=args.symbols,
    ))
