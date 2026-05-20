"""
ml/meta_labeler.py — MetaLabeler: "portero" de señales con RandomForest.
=========================================================================
No predice el precio. Predice si una señal de entrada concreta tiene
probabilidad de éxito > umbral configurado (por defecto 60%).

Flujo:
    1. build_dataset()  → construye (X, y) a partir de trades históricos
    2. train()          → entrena RF con TimeSeriesSplit, guarda modelo + metadata
    3. predict_proba()  → probabilidad de éxito para una señal en vivo
    4. is_ready()       → True si el modelo está entrenado y cargado
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import structlog
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit

log = structlog.get_logger(__name__)

# ── Features canónicas (mismo orden siempre para el modelo) ────────────────────
FEATURES: List[str] = [
    "rsi",            # RSI-14 actual
    "adx",            # Fuerza de tendencia (ADX-14)
    "atr_pct",        # ATR / precio_actual * 100 (volatilidad normalizada)
    "ema21_dist_pct", # (precio - EMA21) / EMA21 * 100
    "ema55_dist_pct", # (precio - EMA55) / EMA55 * 100
    "macd_hist",      # Histograma MACD
    "bb_width",       # (BB_upper - BB_lower) / BB_mid (compresión)
    "vol_ratio",      # volume / volume.rolling(20).mean()
    "cvd_bull",       # CVD alcista: 1 si vol_delta > 0, else 0
    "regime_encoded", # BULL=1, RANGE=0, HIGH_VOL=-1
]

MODEL_VERSION = "1.0.0"
MIN_SAMPLES = 30


class MetaLabeler:
    """
    Portero ML para señales de trading usando RandomForestClassifier.

    Usa TimeSeriesSplit (no K-Fold aleatorio) para respetar el orden
    temporal de los datos y evitar filtración de información futura.
    """

    def __init__(self, model_path: str = "ml/model.joblib") -> None:
        """
        Args:
            model_path: ruta donde se guarda/carga el modelo entrenado.
        """
        self.model_path = Path(model_path)
        self.metadata_path = self.model_path.parent / "model_metadata.json"
        self._model: Optional[RandomForestClassifier] = None
        self._metadata: Dict[str, Any] = {}

        # Intentar cargar modelo existente al arrancar
        if self.model_path.exists():
            self._load_model()

    # ── Construcción del dataset ───────────────────────────────────────────────

    def build_dataset(
        self,
        trades: List[Dict[str, Any]],
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Construye (X, y) alineando trades históricos con el snapshot de
        indicadores en el momento de la señal de entrada.

        Args:
            trades: lista de dicts con campos entry_time, exit_price, tp1.
                    Cada dict debe tener 'entry_time' como datetime o timestamp.
            df:     DataFrame con OHLCV + indicadores (columnas en FEATURES).
                    Index o columna 'timestamp' en UTC.

        Returns:
            (X, y) donde y=1 si trade en profit, y=0 si en pérdida.
        """
        if "timestamp" not in df.columns:
            raise ValueError("df debe contener columna 'timestamp'")

        # Asegurar timestamp como DatetimeIndex para búsqueda eficiente
        df_indexed = df.set_index("timestamp")
        df_indexed.index = pd.to_datetime(df_indexed.index, utc=True)

        rows: List[Dict[str, float]] = []
        labels: List[int] = []

        for trade in trades:
            entry_time = pd.Timestamp(trade["entry_time"]).tz_localize("UTC") \
                if pd.Timestamp(trade["entry_time"]).tzinfo is None \
                else pd.Timestamp(trade["entry_time"]).tz_convert("UTC")

            # Buscar la vela más cercana al entry_time (hacia atrás)
            idx_candidates = df_indexed.index[df_indexed.index <= entry_time]
            if idx_candidates.empty:
                log.debug("skip_trade_no_candle", entry_time=str(entry_time))
                continue

            candle = df_indexed.loc[idx_candidates[-1]]

            # Extraer features; rellenar NaN con 0 (no queremos entrenar en NaN)
            feature_row: Dict[str, float] = {}
            for feat in FEATURES:
                val = candle.get(feat, np.nan) if isinstance(candle, pd.Series) else np.nan
                feature_row[feat] = float(val) if pd.notna(val) else 0.0

            rows.append(feature_row)

            # Label: 1 si el trade fue profitable, 0 si no
            pnl = trade.get("pnl", None)
            if pnl is not None:
                labels.append(1 if float(pnl) > 0 else 0)
            else:
                # Fallback: comparar exit_price con entry_price
                entry_price = float(trade.get("entry_price", 0))
                exit_price = float(trade.get("exit_price", 0))
                labels.append(1 if exit_price > entry_price else 0)

        if not rows:
            return pd.DataFrame(columns=FEATURES), pd.Series(dtype=int)

        X = pd.DataFrame(rows, columns=FEATURES)
        y = pd.Series(labels, name="target")

        log.info(
            "dataset_built",
            n_samples=len(X),
            win_rate=f"{y.mean():.2%}",
            features=FEATURES,
        )
        return X, y

    # ── Entrenamiento ──────────────────────────────────────────────────────────

    def train(
        self,
        trades: List[Dict[str, Any]],
        df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Entrena el RandomForest con los trades históricos y guarda el modelo.

        Usa TimeSeriesSplit(n_splits=5) para validación correcta en series
        temporales (sin filtración de información futura).

        Args:
            trades: lista de trades del backtesting / journal.
            df:     DataFrame con OHLCV + indicadores completos.

        Returns:
            dict con métricas: accuracy, precision, recall, f1, n_samples,
            feature_importance, cv_scores.

        Raises:
            ValueError: si n_samples < MIN_SAMPLES.
        """
        X, y = self.build_dataset(trades, df)

        if len(X) < MIN_SAMPLES:
            log.warning(
                "insufficient_samples",
                n_samples=len(X),
                min_required=MIN_SAMPLES,
                action="skipping_training",
            )
            return {
                "error": f"Solo {len(X)} muestras (mínimo {MIN_SAMPLES})",
                "n_samples": len(X),
                "trained": False,
            }

        # ── Cross-validation temporal ──────────────────────────────────────────
        tscv = TimeSeriesSplit(n_splits=5)
        cv_accuracies: List[float] = []
        cv_f1s: List[float] = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            fold_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=6,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
            fold_model.fit(X_tr, y_tr)
            preds = fold_model.predict(X_val)
            cv_accuracies.append(accuracy_score(y_val, preds))
            cv_f1s.append(f1_score(y_val, preds, zero_division=0))
            log.debug(
                "cv_fold_done",
                fold=fold + 1,
                accuracy=f"{cv_accuracies[-1]:.3f}",
                f1=f"{cv_f1s[-1]:.3f}",
            )

        # ── Entrenamiento final en todo el dataset ─────────────────────────────
        final_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        final_model.fit(X, y)
        final_preds = final_model.predict(X)

        # Métricas en el dataset completo (referencia, no validación real)
        metrics: Dict[str, Any] = {
            "accuracy":          round(float(accuracy_score(y, final_preds)), 4),
            "precision":         round(float(precision_score(y, final_preds, zero_division=0)), 4),
            "recall":            round(float(recall_score(y, final_preds, zero_division=0)), 4),
            "f1":                round(float(f1_score(y, final_preds, zero_division=0)), 4),
            "cv_accuracy_mean":  round(float(np.mean(cv_accuracies)), 4),
            "cv_f1_mean":        round(float(np.mean(cv_f1s)), 4),
            "n_samples":         len(X),
            "win_rate_dataset":  round(float(y.mean()), 4),
            "feature_importance": self._importance_dict(final_model),
            "trained":           True,
        }

        # ── Guardar modelo y metadata ──────────────────────────────────────────
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_model, self.model_path)
        self._model = final_model

        self._metadata = {
            "trained_at":        datetime.utcnow().isoformat() + "Z",
            "model_version":     MODEL_VERSION,
            "n_samples":         len(X),
            "features":          FEATURES,
            "metrics":           {
                k: v for k, v in metrics.items()
                if k != "feature_importance"
            },
            "feature_importance": metrics["feature_importance"],
        }
        with open(self.metadata_path, "w", encoding="utf-8") as fh:
            json.dump(self._metadata, fh, indent=2, ensure_ascii=False)

        log.info(
            "model_trained",
            n_samples=len(X),
            cv_accuracy=f"{metrics['cv_accuracy_mean']:.3f}",
            cv_f1=f"{metrics['cv_f1_mean']:.3f}",
            model_path=str(self.model_path),
        )
        return metrics

    # ── Predicción en vivo ─────────────────────────────────────────────────────

    def predict_proba(self, features: Dict[str, float]) -> float:
        """
        Calcula la probabilidad de éxito para una señal en vivo.

        Args:
            features: dict con las claves de FEATURES y sus valores actuales.

        Returns:
            Probabilidad [0.0 - 1.0]. Devuelve 0.5 si el modelo no está listo
            (modo permisivo: no bloquea señales mientras no hay datos suficientes).
        """
        if not self.is_ready():
            log.debug("model_not_ready_using_default_proba", default=0.5)
            return 0.5

        row = [features.get(feat, 0.0) for feat in FEATURES]
        X = np.array(row, dtype=float).reshape(1, -1)

        # predict_proba → [[P(0), P(1)]] — nos interesa P(1)
        proba = float(self._model.predict_proba(X)[0][1])  # type: ignore[union-attr]

        log.debug("ml_proba", proba=f"{proba:.3f}", features=features)
        return proba

    # ── Consultas de estado ────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        """True si el modelo está entrenado y cargado en memoria."""
        return self._model is not None

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Devuelve las top-10 features ordenadas por importancia (descendente).

        Returns:
            dict {feature_name: importance_score}. Vacío si el modelo no existe.
        """
        if not self.is_ready():
            return {}
        return self._importance_dict(self._model)  # type: ignore[arg-type]

    def get_metadata(self) -> Dict[str, Any]:
        """Devuelve la metadata del modelo cargada desde model_metadata.json."""
        if self._metadata:
            return self._metadata
        if self.metadata_path.exists():
            with open(self.metadata_path, encoding="utf-8") as fh:
                self._metadata = json.load(fh)
        return self._metadata

    # ── Helpers privados ───────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Carga el modelo desde disco."""
        try:
            self._model = joblib.load(self.model_path)
            log.info("model_loaded", path=str(self.model_path))
        except Exception as exc:
            log.error("model_load_failed", path=str(self.model_path), error=str(exc))
            self._model = None

    @staticmethod
    def _importance_dict(model: RandomForestClassifier) -> Dict[str, float]:
        """Convierte feature_importances_ del modelo en un dict ordenado (top-10)."""
        pairs = sorted(
            zip(FEATURES, model.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        )
        return {feat: round(float(imp), 6) for feat, imp in pairs[:10]}
