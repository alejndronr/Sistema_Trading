"""
ml/meta_labeler.py — MetaLabeler: "portero" de señales con RandomForest.
=========================================================================
No predice el precio. Predice si una señal de entrada concreta tiene
probabilidad de éxito > umbral configurado (dinámico, optimizado en entrenamiento).
Diseñado para el hardware de bajos recursos Intel Atom E3950.
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

# ── Features canónicas V2 (mismo orden siempre para el modelo) ──────────────────
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
    "consensus",      # Consensus Score (0-100)
    "rsi_lag1",       # RSI lag 1
    "rsi_lag3",       # RSI lag 3
    "adx_lag1",       # ADX lag 1
    "close_chg_1",    # % cambio close 1 vela
    "close_chg_3",    # % cambio close 3 velas
    "rsi_roll5_mean", # Media rolling 5 rsi
    "rsi_roll5_std",  # Desviación rolling 5 rsi
    "vol_roll10_mean",# Media rolling 10 volumen
    "ob_bull_recent5",# OB bullish reciente en 5 velas
    "fvg_bull_recent3",# FVG bullish reciente en 3 velas
]

MODEL_VERSION = "2.0.0"
MIN_SAMPLES = 30


class MetaLabeler:
    """
    Portero ML para señales de trading usando RandomForestClassifier.
    Optimizado para el Atom E3950 (n_estimators=100, max_depth=6, n_jobs=2).
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

    def build_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """
        Construye matriz de features vectorizada.
        Diseñada para Atom E3950: sin loops Python, sin scipy.
        Procesa 500 filas en < 50ms en el hardware objetivo.

        Args:
            df: DataFrame con indicadores ya calculados (apply_all_indicators)

        Returns:
            numpy array shape (n_samples, n_features), float32 para menor RAM
        """
        price = df["close"].values.astype(float)
        n = len(price)

        def col(name: str, default: float = 0.0) -> np.ndarray:
            """Extrae columna con default seguro."""
            if name in df.columns:
                return df[name].fillna(default).values.astype(float)
            return np.full(n, default, dtype=float)

        rsi = col("rsi", 50.0)
        adx = col("adx", 20.0)
        atr = col("atr", price * 0.015)
        
        # Buscar ema_21 o ema21
        ema21_col = "ema_21" if "ema_21" in df.columns else "ema21"
        ema21 = col(ema21_col, price)
        
        # Buscar ema_55 o ema55
        ema55_col = "ema_55" if "ema_55" in df.columns else "ema55"
        ema55 = col(ema55_col, price)
        
        macd_h = col("macd_histogram", col("macd_hist", 0.0))
        bb_up = col("bb_upper", price * 1.02)
        bb_lo = col("bb_lower", price * 0.98)
        bb_mid = col("bb_mid", price)
        vol = col("volume", 1.0)
        cvd = col("cvd", 0.0)
        
        # Régimen
        regime_raw = df.get("regime", df.get("trend_regime", pd.Series("RANGE", index=df.index)))
        regime_map = {
            "BULL_TREND": 1.0, "BULL": 1.0, "bullish": 1.0,
            "RANGE": 0.0, "ranging": 0.0,
            "HIGH_VOL": -0.5, "high_volatility": -0.5,
            "BEAR_TREND": -1.0, "BEAR": -1.0, "bearish": -1.0
        }
        regime = regime_raw.map(regime_map).fillna(0.0).values.astype(float)
        consensus = col("consensus", 50.0)

        # Volume moving average (vectorizado)
        vol_s = pd.Series(vol)
        vol_ma = vol_s.rolling(20, min_periods=1).mean().values
        vol_ma = np.where(vol_ma == 0.0, 1.0, vol_ma)

        # Features base
        safe_price = np.where(price == 0.0, 1e-10, price)
        safe_ema21 = np.where(ema21 == 0.0, 1e-10, ema21)
        safe_ema55 = np.where(ema55 == 0.0, 1e-10, ema55)
        safe_bbmid = np.where(bb_mid == 0.0, 1e-10, bb_mid)

        f_rsi = rsi
        f_adx = adx
        f_atr_pct = atr / safe_price * 100
        f_ema21_dist = (price - ema21) / safe_ema21 * 100
        f_ema55_dist = (price - ema55) / safe_ema55 * 100
        f_macd_hist = macd_h
        f_bb_width = (bb_up - bb_lo) / safe_bbmid
        f_vol_ratio = vol / vol_ma
        f_cvd_bull = (cvd > 0.0).astype(float)
        f_regime = regime
        f_consensus = consensus

        # Lag features (shift con padding)
        def shift_pad(arr: np.ndarray, num: int, fill: float = 0.0) -> np.ndarray:
            out = np.empty_like(arr)
            out[:num] = fill
            out[num:] = arr[:-num]
            return out

        f_rsi_lag1 = shift_pad(rsi, 1, 50.0)
        f_rsi_lag3 = shift_pad(rsi, 3, 50.0)
        f_adx_lag1 = shift_pad(adx, 1, 20.0)
        f_chg1 = np.diff(price, prepend=price[0]) / safe_price * 100
        f_chg3 = np.where(
            np.arange(n) >= 3,
            (price - shift_pad(price, 3, price[0])) / safe_price * 100,
            0.0
        )

        # Rolling stats (vectorizado con pandas)
        rsi_s = pd.Series(rsi)
        f_rsi_mean5 = rsi_s.rolling(5, min_periods=1).mean().values
        f_rsi_std5 = rsi_s.rolling(5, min_periods=1).std().fillna(5.0).values
        f_vol_mean10 = vol_s.rolling(10, min_periods=1).mean().values

        # SMC recientes (rolling max vectorizado)
        ob_bull = col("ob_bull", 0.0)
        fvg_bull = col("fvg_bull", 0.0)
        f_ob_recent = pd.Series(ob_bull).rolling(5, min_periods=1).max().values
        f_fvg_recent = pd.Series(fvg_bull).rolling(3, min_periods=1).max().values

        # Stack en matriz
        X = np.column_stack([
            f_rsi, f_adx, f_atr_pct, f_ema21_dist, f_ema55_dist,
            f_macd_hist, f_bb_width, f_vol_ratio, f_cvd_bull, f_regime,
            f_consensus,
            f_rsi_lag1, f_rsi_lag3, f_adx_lag1, f_chg1, f_chg3,
            f_rsi_mean5, f_rsi_std5, f_vol_mean10,
            f_ob_recent, f_fvg_recent,
        ]).astype(np.float32)  # float32 → mitad de RAM que float64

        # Reemplazar inf/nan
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Z-score por columna (vectorizado, sin scipy)
        mu = X.mean(axis=0)
        std = X.std(axis=0)
        std = np.where(std == 0.0, 1.0, std)
        X = (X - mu) / std

        return X

    def build_dataset(
        self,
        trades: List[Dict[str, Any]],
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Construye (X, y) alineando trades históricos con el matrix de features.
        """
        if "timestamp" not in df.columns:
            raise ValueError("df debe contener columna 'timestamp'")

        # Calcular la matriz completa de features
        X_matrix = self.build_feature_matrix(df)
        
        # Crear un mapa de timestamp -> index fila de features
        df_indexed = df.copy()
        df_indexed["feature_idx"] = np.arange(len(df_indexed))
        df_indexed = df_indexed.set_index("timestamp")
        df_indexed.index = pd.to_datetime(df_indexed.index, utc=True)

        rows: List[np.ndarray] = []
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
            f_idx = int(candle.get("feature_idx") if isinstance(candle, pd.Series) else candle["feature_idx"].iloc[-1])

            rows.append(X_matrix[f_idx])

            # Label: 1 si el trade fue profitable, 0 si no
            pnl = trade.get("pnl", None)
            if pnl is not None:
                labels.append(1 if float(pnl) > 0 else 0)
            else:
                entry_price = float(trade.get("entry_price", 0))
                exit_price = float(trade.get("exit_price", 0))
                labels.append(1 if exit_price > entry_price else 0)

        if not rows:
            return pd.DataFrame(columns=FEATURES), pd.Series(dtype=int)

        X = pd.DataFrame(np.array(rows), columns=FEATURES)
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
        Usa TimeSeriesSplit(n_splits=5) para validación correcta.
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

        # Usar conjunto de validación final para optimizar el threshold
        val_size = max(int(len(X) * 0.2), 5)
        X_train_cv, X_val_cv = X.iloc[:-val_size], X.iloc[-val_size:]
        y_train_cv, y_val_cv = y.iloc[:-val_size], y.iloc[-val_size:]

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_cv)):
            X_tr, X_val = X_train_cv.iloc[train_idx], X_train_cv.iloc[val_idx]
            y_tr, y_val = y_train_cv.iloc[train_idx], y_train_cv.iloc[val_idx]

            fold_model = RandomForestClassifier(
                n_estimators=100,    # máximo para Atom
                max_depth=6,
                min_samples_leaf=10,
                max_features="sqrt",
                class_weight="balanced",
                n_jobs=2,            # dejar 2 cores libres
                random_state=42,
            )
            fold_model.fit(X_tr, y_tr)
            preds = fold_model.predict(X_val)
            cv_accuracies.append(accuracy_score(y_val, preds))
            cv_f1s.append(f1_score(y_val, preds, zero_division=0))

        # ── Entrenamiento final en todo el dataset ─────────────────────────────
        final_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_leaf=10,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=2,
            random_state=42,
        )
        final_model.fit(X, y)
        final_preds = final_model.predict(X)

        # Optimizar threshold dinámico en el set de validación
        best_thresh = self._optimize_threshold(final_model, X_val_cv.values, y_val_cv.values)

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
            "optimal_threshold": best_thresh,
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
            optimal_threshold=best_thresh,
            model_path=str(self.model_path),
        )
        return metrics

    def _optimize_threshold(
        self, model, X_val: np.ndarray, y_val: np.ndarray
    ) -> float:
        """
        Grid de 9 puntos (0.40→0.80, paso 0.05).
        En Atom E3950: < 100ms para 500 muestras.
        """
        if len(X_val) == 0:
            return 0.60
            
        probas = model.predict_proba(X_val)[:, 1]
        best_thresh, best_f1 = 0.60, 0.0

        for thresh in np.arange(0.40, 0.81, 0.05):
            preds = (probas >= thresh).astype(int)
            if preds.sum() == 0:
                continue
            f1 = f1_score(y_val, preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thresh = f1, float(thresh)

        log.info("threshold_optimized", threshold=f"{best_thresh:.2f}", f1=f"{best_f1:.4f}")
        return best_thresh

    # ── Predicción en vivo ─────────────────────────────────────────────────────

    def predict_proba(self, df: pd.DataFrame) -> float:
        """
        Recibe el DataFrame COMPLETO (no solo la última fila)
        para poder calcular lag features y rolling stats.
        Devuelve la probabilidad de la última fila.
        """
        if not self.is_ready():
            log.debug("model_not_ready_using_default_proba", default=0.5)
            return 0.5

        try:
            X = self.build_feature_matrix(df)
            if len(X) == 0:
                return 0.5
            # predict_proba → [[P(0), P(1)]] — nos interesa P(1) de la última fila
            proba = float(self._model.predict_proba(X[-1:])[:, 1][0])
            log.debug("ml_proba", proba=f"{proba:.3f}")
            return proba
        except Exception as exc:
            log.warning("predict_proba_failed", error=str(exc))
            return 0.5

    # ── Consultas de estado ────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        """True si el modelo está entrenado y cargado en memoria."""
        return self._model is not None

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Devuelve las top-10 features ordenadas por importancia (descendente).
        """
        if not self.is_ready():
            return {}
        return self._importance_dict(self._model)

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
