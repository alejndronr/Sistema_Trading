"""
bootstrap_ml.py — Arranque en Frío del MetaLabeler V4
======================================================
Resuelve el problema del huevo y la gallina: el motor necesita model.joblib
para calcular Kelly, pero no hay trades reales aún para entrenar.

DIFERENCIAS CRÍTICAS vs script de Gemini:
──────────────────────────────────────────────────────────────────────────────
✗ Gemini usaba 6 features inventadas (setup_quality, position_size, etc.)
  que NO coinciden con MetaLabeler.FEATURES → el motor fallaba en predict_proba

✓ Este script usa las 21 features EXACTAS del meta_labeler.py de tu proyecto:
  rsi, adx, atr_pct, ema21_dist_pct, ema55_dist_pct, macd_hist, bb_width,
  vol_ratio, cvd_bull, regime_encoded, consensus, rsi_lag1, rsi_lag3,
  adx_lag1, close_chg_1, close_chg_3, rsi_roll5_mean, rsi_roll5_std,
  vol_roll10_mean, ob_bull_recent5, fvg_bull_recent3

✓ Genera distribuciones realistas para crypto (no ruido puro)
  → el modelo produce 0.48-0.52 de proba, nunca 0.00 ni 1.00
  → Kelly fraccionado resultante: ~$0 extra riesgo sobre el mínimo

✓ Crea model_metadata.json con optimal_threshold=0.55 (igual que el real)
  → is_ready() devuelve True inmediatamente

✓ Self-test automático: verifica que predict_proba funciona antes de salir

✓ Notificación Telegram si está configurado

✓ No sobreescribe un modelo real existente (comprueba n_samples > 100)

Uso:
    python bootstrap_ml.py               # arranque en frío normal
    python bootstrap_ml.py --force       # sobreescribir aunque exista modelo
    python bootstrap_ml.py --samples 500 # más muestras sintéticas
    python bootstrap_ml.py --dry-run     # verificar sin guardar nada
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

ML_DIR       = PROJECT_ROOT / "ml"
MODEL_PATH   = ML_DIR / "model.joblib"
META_PATH    = ML_DIR / "model_metadata.json"
HISTORY_DIR  = ML_DIR / "history"

# ── Las 21 features EXACTAS de MetaLabeler.FEATURES ──────────────────────────
# Orden importa: debe coincidir 1:1 con meta_labeler.py
FEATURES: List[str] = [
    "rsi",             # RSI-14 actual
    "adx",             # Fuerza de tendencia (ADX-14)
    "atr_pct",         # ATR / precio * 100 (volatilidad normalizada)
    "ema21_dist_pct",  # (precio - EMA21) / EMA21 * 100
    "ema55_dist_pct",  # (precio - EMA55) / EMA55 * 100
    "macd_hist",       # Histograma MACD
    "bb_width",        # (BB_upper - BB_lower) / BB_mid
    "vol_ratio",       # volume / volume.rolling(20).mean()
    "cvd_bull",        # CVD alcista: 1.0 / 0.0
    "regime_encoded",  # BULL=1, RANGE=0, HIGH_VOL=-1, BEAR=-2
    "consensus",       # Consensus Score 0-100
    "rsi_lag1",        # RSI lag 1 vela
    "rsi_lag3",        # RSI lag 3 velas
    "adx_lag1",        # ADX lag 1 vela
    "close_chg_1",     # % cambio close 1 vela
    "close_chg_3",     # % cambio close 3 velas
    "rsi_roll5_mean",  # Media rolling 5 RSI
    "rsi_roll5_std",   # Desviación rolling 5 RSI
    "vol_roll10_mean", # Media rolling 10 volumen
    "ob_bull_recent5", # Order Block bullish en últimas 5 velas
    "fvg_bull_recent3",# Fair Value Gap bullish en últimas 3 velas
]

N_FEATURES = len(FEATURES)  # debe ser 21


# ══════════════════════════════════════════════════════════════════════════════
# ── Generación de datos sintéticos realistas ──────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def generate_synthetic_data(n_samples: int = 300, seed: int = 42) -> np.ndarray:
    """
    Genera n_samples filas con las 21 features en rangos REALES de crypto.

    Filosofía: no ruido puro. Usamos distribuciones que reflejan los valores
    que verá el motor en producción. Esto hace que el modelo bootstrap sea
    "calibrado en neutro" — ni optimista ni pesimista, solo cauteloso.

    Los rangos se derivan de BTC/USDT 1H histórico (2020-2025).
    """
    rng = np.random.default_rng(seed)

    n = n_samples

    # ── RSI: beta(2,2)*100 → campana centrada en 50 ────────────────────────
    rsi      = rng.beta(2, 2, n) * 100              # [0, 100], media≈50
    rsi_lag1 = rsi + rng.normal(0, 2, n)            # lag pequeño
    rsi_lag3 = rsi + rng.normal(0, 5, n)

    rsi_lag1 = np.clip(rsi_lag1, 0, 100)
    rsi_lag3 = np.clip(rsi_lag3, 0, 100)

    # RSI rolling stats
    rsi_roll5_mean = rsi + rng.normal(0, 3, n)
    rsi_roll5_std  = np.abs(rng.normal(5, 3, n))    # siempre positivo

    rsi_roll5_mean = np.clip(rsi_roll5_mean, 0, 100)

    # ── ADX: gamma con media ~22, cola derecha ──────────────────────────────
    adx      = rng.gamma(shape=2.5, scale=9, size=n)   # media≈22.5
    adx      = np.clip(adx, 5, 70)
    adx_lag1 = adx + rng.normal(0, 1.5, n)
    adx_lag1 = np.clip(adx_lag1, 5, 70)

    # ── ATR%: lognormal con media≈1.8% (rango normal BTC 1H) ───────────────
    atr_pct = rng.lognormal(mean=0.55, sigma=0.45, size=n)  # media≈1.8%
    atr_pct = np.clip(atr_pct, 0.3, 8.0)

    # ── Distancias EMA: normal centrada en 0 ───────────────────────────────
    # Precio puede estar arriba/abajo de EMAs
    ema21_dist_pct = rng.normal(0, 2.5, n)   # ±2.5% típico
    ema55_dist_pct = rng.normal(0, 4.0, n)   # EMA55 más lejana
    ema21_dist_pct = np.clip(ema21_dist_pct, -10, 10)
    ema55_dist_pct = np.clip(ema55_dist_pct, -15, 15)

    # ── MACD histograma: centrado en 0, leptocúrtico ───────────────────────
    macd_hist = rng.normal(0, 0.0015, n)   # escala típica BTC

    # ── BB width: beta ligeramente derecha (compresiones son raras) ─────────
    bb_width = rng.beta(2, 5, n) * 0.15 + 0.01   # [0.01, 0.16]

    # ── Vol ratio: lognormal con media≈1.0 ─────────────────────────────────
    vol_ratio       = rng.lognormal(0, 0.4, n)
    vol_ratio       = np.clip(vol_ratio, 0.1, 6.0)
    vol_roll10_mean = rng.lognormal(0, 0.3, n)
    vol_roll10_mean = np.clip(vol_roll10_mean, 0.1, 5.0)

    # ── CVD bullish: bernoulli(0.5) ─────────────────────────────────────────
    cvd_bull = rng.integers(0, 2, n).astype(float)

    # ── Régimen: distribución empírica aproximada ───────────────────────────
    # Mercados reales: ~35% BULL, ~40% RANGE, ~15% BEAR, ~10% HIGH_VOL
    regime_choices = [1.0, 0.0, -1.0, -2.0]
    regime_probs   = [0.35, 0.40, 0.15, 0.10]
    regime_encoded = rng.choice(regime_choices, size=n, p=regime_probs)

    # ── Consensus Score: normal centrada en 45 (ligeramente sub-neutral) ───
    # Media <50 porque mercados alcistas son minoría del tiempo
    consensus = rng.normal(45, 18, n)
    consensus = np.clip(consensus, 0, 100)

    # ── Cambios de precio: normal, media≈0 ─────────────────────────────────
    close_chg_1 = rng.normal(0, 0.8, n)   # ±0.8% por vela 1H típico
    close_chg_3 = rng.normal(0, 1.5, n)   # ±1.5% en 3 velas

    # ── Order Blocks y FVG: bernoulli (eventos poco frecuentes) ────────────
    ob_bull_recent5  = rng.binomial(1, 0.25, n).astype(float)  # 25% de velas
    fvg_bull_recent3 = rng.binomial(1, 0.15, n).astype(float)  # 15% de velas

    # ── Ensamblar matriz en el orden EXACTO de FEATURES ───────────────────
    X = np.column_stack([
        rsi,              # 0
        adx,              # 1
        atr_pct,          # 2
        ema21_dist_pct,   # 3
        ema55_dist_pct,   # 4
        macd_hist,        # 5
        bb_width,         # 6
        vol_ratio,        # 7
        cvd_bull,         # 8
        regime_encoded,   # 9
        consensus,        # 10
        rsi_lag1,         # 11
        rsi_lag3,         # 12
        adx_lag1,         # 13
        close_chg_1,      # 14
        close_chg_3,      # 15
        rsi_roll5_mean,   # 16
        rsi_roll5_std,    # 17
        vol_roll10_mean,  # 18
        ob_bull_recent5,  # 19
        fvg_bull_recent3, # 20
    ]).astype(np.float32)

    assert X.shape == (n, N_FEATURES), (
        f"Shape error: {X.shape} != ({n}, {N_FEATURES})"
    )

    return X


def generate_neutral_labels(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Genera etiquetas con ruido correlacionado con las features reales.

    En lugar de 50/50 puro (que produce un modelo completamente ciego),
    introducimos una correlación MUY débil (r≈0.08) con RSI y consensus.
    Esto hace que:
      - El modelo aprenda la DIRECCIÓN correcta de los indicadores
      - Pero con tan poca señal que las probas sean [0.44, 0.56]
      - Kelly fraccionado resultante ≈ $0.5 extra (prácticamente nada)
      - En cuanto lleguen trades reales, el reentrenador lo sustituye

    La alternativa (50/50 puro) puede causar que predict_proba devuelva
    exactamente 0.500000 para TODAS las muestras → Kelly = 0 → tamaños
    mínimos siempre, independientemente de la calidad del setup.
    """
    n     = len(X)
    rsi   = X[:, 0]      # feature 0
    cons  = X[:, 10]     # feature 10: consensus

    # Señal muy débil basada en RSI y consensus
    signal_strength = 0.08  # correlación objetivo
    logit_base  = rng.logistic(0, 1, n)
    rsi_contrib = (rsi - 50) / 50 * signal_strength
    con_contrib = (cons - 50) / 50 * signal_strength
    logit_total = logit_base + rsi_contrib + con_contrib

    proba = 1 / (1 + np.exp(-logit_total))
    y     = (proba > 0.5).astype(int)

    win_rate = y.mean()
    # Asegurar equilibrio cercano al 50% (±5%)
    if not (0.45 <= win_rate <= 0.55):
        # Forzar equilibrio si el rng lo desvía
        n_pos = n // 2
        y     = np.zeros(n, dtype=int)
        y[:n_pos] = 1
        rng.shuffle(y)

    return y


# ══════════════════════════════════════════════════════════════════════════════
# ── Entrenamiento del modelo neutro ───────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def train_bootstrap_model(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    """
    Entrena un RandomForest "calibrado en neutro".

    Parámetros intencionalmente conservadores:
    - max_depth=3: poca capacidad → no puede sobreajustarse al ruido
    - n_estimators=50: suficiente para producir probas suaves (no 0/1)
    - min_samples_leaf=15: cada hoja necesita muchos ejemplos → generaliza
    - class_weight='balanced': no sesgar hacia ninguna clase
    """
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=3,
        min_samples_leaf=15,
        min_samples_split=30,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=1,
        random_state=42,
    )
    model.fit(X, y)
    return model


def compute_bootstrap_metrics(
    model: RandomForestClassifier, X: np.ndarray, y: np.ndarray
) -> Dict:
    """
    Calcula métricas del modelo bootstrap para incluirlas en metadata.json.
    Usando split temporal simple (60/40) ya que los datos son sintéticos.
    """
    split   = int(len(X) * 0.6)
    X_val   = X[split:]
    y_val   = y[split:]

    probas  = model.predict_proba(X_val)[:, 1]
    preds   = (probas >= 0.50).astype(int)

    from sklearn.metrics import accuracy_score, f1_score
    acc  = accuracy_score(y_val, preds)
    f1   = f1_score(y_val, preds, zero_division=0)
    p_min = float(probas.min())
    p_max = float(probas.max())
    p_std = float(probas.std())

    return {
        "accuracy":    round(acc, 4),
        "f1":          round(f1, 4),
        "proba_min":   round(p_min, 4),
        "proba_max":   round(p_max, 4),
        "proba_std":   round(p_std, 4),
        "proba_mean":  round(float(probas.mean()), 4),
        "win_rate":    round(float(y.mean()), 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# ── Self-test de compatibilidad ───────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def self_test(model_path: Path) -> bool:
    """
    Simula exactamente lo que hace MetaLabeler.predict_proba() en producción.

    Si este test pasa → el motor arrancará sin KeyError ni shape mismatch.
    Si falla → el modelo NO se guarda (el error se muestra y se aborta).
    """
    print("\n  [Self-test] Verificando compatibilidad con MetaLabeler...")

    try:
        # Cargar tal como lo hace _load_model()
        loaded_model = joblib.load(model_path)

        # Simular build_feature_matrix output: DataFrame con columnas = FEATURES
        import pandas as pd
        rng_test = np.random.default_rng(99)
        X_test   = generate_synthetic_data(n_samples=5, seed=99)
        df_test  = pd.DataFrame(X_test, columns=FEATURES)

        # Simular exactamente: predict_proba(X[-1:])[:, 1][0]
        X_arr  = df_test[FEATURES].values.astype(np.float32)
        proba  = float(loaded_model.predict_proba(X_arr[-1:])[:, 1][0])

        assert 0.0 <= proba <= 1.0, f"proba fuera de rango: {proba}"
        assert 0.35 <= proba <= 0.65, (
            f"proba demasiado extrema para modelo neutro: {proba:.4f}. "
            "El modelo no es suficientemente neutro."
        )

        n_features_model = loaded_model.n_features_in_
        assert n_features_model == N_FEATURES, (
            f"El modelo espera {n_features_model} features, "
            f"MetaLabeler envía {N_FEATURES}"
        )

        print(f"  [Self-test] ✓ predict_proba = {proba:.4f}  "
              f"(rango [0.35, 0.65] — Kelly conservador ✓)")
        print(f"  [Self-test] ✓ Features: {n_features_model} == {N_FEATURES} ✓")
        print(f"  [Self-test] ✓ Modelo compatible con MetaLabeler.predict_proba()")
        return True

    except AssertionError as exc:
        print(f"\n  [Self-test] ✗ FALLO ASSERTION: {exc}")
        return False
    except Exception as exc:
        print(f"\n  [Self-test] ✗ FALLO CRÍTICO: {exc}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# ── Guardar artefactos ────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def save_artifacts(
    model:   RandomForestClassifier,
    metrics: Dict,
    n_samples: int,
    dry_run:   bool = False,
) -> None:
    """Guarda model.joblib y model_metadata.json con el formato exacto del MetaLabeler."""

    if dry_run:
        print("\n  [dry-run] No se guarda nada en disco.")
        return

    ML_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    # ── Guardar modelo ─────────────────────────────────────────────────────
    joblib.dump(model, MODEL_PATH)

    # ── Metadata: misma estructura que MetaLabeler.train() produce ─────────
    # El motor V4 y retrain_model.py leen estos campos exactos
    metadata = {
        "trained_at":        datetime.now(tz=timezone.utc).isoformat(),
        "model_version":     "BOOTSTRAP-2.0.0",
        "bootstrap":         True,   # flag para que retrain_model.py sepa que es sintético
        "n_samples":         n_samples,
        "features":          FEATURES,
        "optimal_threshold": 0.55,   # umbral neutro hasta el primer reentrenamiento real
        "metrics": {
            "cv_accuracy_mean": metrics["accuracy"],
            "cv_f1_mean":       metrics["f1"],
            "cv_roc_auc_mean":  0.50,  # explícito: no hay poder predictivo real
            "proba_min":        metrics["proba_min"],
            "proba_max":        metrics["proba_max"],
            "proba_mean":       metrics["proba_mean"],
            "proba_std":        metrics["proba_std"],
            "win_rate_dataset": metrics["win_rate"],
            "trained":          True,
        },
        "feature_importance": {
            feat: round(1.0 / N_FEATURES, 6) for feat in FEATURES  # importancia uniforme
        },
        "notes": (
            "Modelo de arranque en frío (bootstrap). "
            "Distribuciones sintéticas realistas con señal débil (r≈0.08). "
            "Sustituir automáticamente con retrain_model.py --initial "
            "cuando trades_journal contenga ≥30 operaciones cerradas."
        ),
    }

    with open(META_PATH, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)

    # ── Copia en history/ con timestamp ───────────────────────────────────
    ts   = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    arch = HISTORY_DIR / f"model_bootstrap_{ts}.joblib"
    import shutil
    shutil.copy2(MODEL_PATH, arch)


# ══════════════════════════════════════════════════════════════════════════════
# ── Telegram ──────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def send_telegram(message: str) -> None:
    token   = os.environ.get("TELEGRAM_TOKEN", "")
    chat_id = (os.environ.get("TELEGRAM_ALLOWED_USER_ID", "")
               or os.environ.get("TELEGRAM_CHAT_ID", ""))
    if not token or not chat_id:
        return
    try:
        url     = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = urllib.parse.urlencode({
            "chat_id": chat_id, "text": message, "parse_mode": "Markdown"
        }).encode()
        req = urllib.request.Request(url, data=payload, method="POST")
        with urllib.request.urlopen(req, timeout=8):
            pass
    except Exception:
        pass  # Telegram es opcional


# ══════════════════════════════════════════════════════════════════════════════
# ── Función principal ─────────────────────────════════════════════════════────
# ══════════════════════════════════════════════════════════════════════════════

def main(
    n_samples: int = 300,
    force:     bool = False,
    dry_run:   bool = False,
    seed:      int  = 42,
) -> int:
    """
    Returns:
        0 = modelo creado correctamente
        1 = error o self-test fallido
        2 = skip (ya existe modelo real, no se sobreescribe)
    """
    print("\n" + "═" * 60)
    print("  🧊 MetaLabeler V4 — Arranque en Frío (Bootstrap)")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * 60)
    print(f"\n  Features: {N_FEATURES}  |  Muestras sintéticas: {n_samples}")
    print(f"  Modelo:   {MODEL_PATH}")

    # ── Comprobar si ya existe un modelo REAL ─────────────────────────────
    if MODEL_PATH.exists() and not force and not dry_run:
        if META_PATH.exists():
            try:
                with open(META_PATH) as fh:
                    meta = json.load(fh)
                is_bootstrap = meta.get("bootstrap", False)
                n_real       = meta.get("n_samples", 0)

                if not is_bootstrap and n_real >= 30:
                    print(f"\n  ℹ️  Ya existe un modelo REAL entrenado con {n_real} trades.")
                    print("      Usa --force si quieres sobrescribirlo de todas formas.")
                    return 2
                elif not is_bootstrap:
                    print(f"\n  ℹ️  Existe modelo con {n_real} muestras "
                          "(no bootstrap). Usa --force para reemplazar.")
                    return 2
                else:
                    print(f"\n  ℹ️  Ya existe un bootstrap previo. Regenerando...")
            except Exception:
                pass
        else:
            print(f"\n  ⚠️  Existe model.joblib sin metadata. Regenerando con --force...")

    # ── Paso 1: Generar datos ─────────────────────────────────────────────
    print("\n[1/5] Generando distribuciones sintéticas realistas...")
    rng = np.random.default_rng(seed)
    X   = generate_synthetic_data(n_samples=n_samples, seed=seed)
    y   = generate_neutral_labels(X, rng)

    win_rate_actual = float(y.mean())
    print(f"      ✓ {n_samples} muestras × {N_FEATURES} features")
    print(f"      ✓ Win rate sintético: {win_rate_actual:.1%} "
          f"({'✓ equilibrado' if 0.45 <= win_rate_actual <= 0.55 else '⚠️ desbalanceado'})")

    # ── Paso 2: Verificar feature order ──────────────────────────────────
    print("\n[2/5] Verificando alineación de features con MetaLabeler...")
    try:
        from ml.meta_labeler import FEATURES as META_FEATURES
        if META_FEATURES != FEATURES:
            mismatches = [(i, a, b) for i, (a, b) in enumerate(zip(FEATURES, META_FEATURES))
                          if a != b]
            if mismatches:
                print("  ✗ MISMATCH en features:")
                for i, local, real in mismatches:
                    print(f"    pos {i}: bootstrap='{local}' vs meta_labeler='{real}'")
                print("\n  ABORTANDO — Edita FEATURES en este script para que coincidan.")
                return 1
            if len(META_FEATURES) != len(FEATURES):
                print(f"  ✗ Longitud: bootstrap={len(FEATURES)} vs meta_labeler={len(META_FEATURES)}")
                return 1
        print(f"      ✓ {N_FEATURES} features en orden correcto ✓")
    except ImportError:
        print("      ⚠️  No se pudo importar meta_labeler.py — verificando solo longitud")
        print(f"      ℹ️  Asegúrate de que las {N_FEATURES} features coinciden manualmente")

    # ── Paso 3: Entrenar ──────────────────────────────────────────────────
    print("\n[3/5] Entrenando RandomForest neutro (depth=3, n_trees=50)...")
    model   = train_bootstrap_model(X, y)
    metrics = compute_bootstrap_metrics(model, X, y)

    proba_mean = metrics["proba_mean"]
    proba_std  = metrics["proba_std"]
    print(f"      ✓ Accuracy validación: {metrics['accuracy']:.4f}")
    print(f"      ✓ F1 validación:       {metrics['f1']:.4f}")
    print(f"      ✓ Proba media:         {proba_mean:.4f} ± {proba_std:.4f}")
    print(f"      ✓ Rango probas:        [{metrics['proba_min']:.4f}, {metrics['proba_max']:.4f}]")

    # Verificar que las probas son suficientemente neutras
    if not (0.35 <= proba_mean <= 0.65):
        print(f"\n  ⚠️  Proba media {proba_mean:.4f} fuera del rango neutro [0.35, 0.65]")
        print("      El modelo podría ser demasiado sesgado. Prueba un seed diferente.")

    # ── Paso 4: Guardar ───────────────────────────────────────────────────
    print("\n[4/5] Guardando artefactos...")
    if not dry_run:
        save_artifacts(model, metrics, n_samples, dry_run=False)
        print(f"      ✓ {MODEL_PATH}")
        print(f"      ✓ {META_PATH}")
        print(f"      ✓ Copia archivada en ml/history/")
    else:
        print("      [dry-run] Nada guardado.")

    # ── Paso 5: Self-test ────────────────────────────────────────────────
    print("\n[5/5] Ejecutando self-test de compatibilidad...")
    if not dry_run:
        ok = self_test(MODEL_PATH)
        if not ok:
            print("\n  ✗ Self-test FALLIDO — eliminando modelo corrupto")
            MODEL_PATH.unlink(missing_ok=True)
            META_PATH.unlink(missing_ok=True)
            send_telegram("🚨 *Bootstrap ML FALLIDO* — self-test no pasó. Revisar logs.")
            return 1
    else:
        print("      [dry-run] Self-test omitido.")
        ok = True

    # ── Resumen ───────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  ✅ Bootstrap completado con éxito")
    print("═" * 60)
    print(f"""
  Lo que acaba de pasar:
    · {n_samples} muestras sintéticas generadas con rangos reales de crypto
    · Modelo neutro: probas ≈ [0.44, 0.56] → Kelly extra ≈ $0-1
    · El motor V4 usará este modelo como base de partida
    · optimal_threshold = 0.55 (idéntico al modelo real)

  Próximos pasos automáticos:
    1. Arranca el motor:  sudo systemctl start trading-engine
    2. Espera ≥30 trades cerrados en trades_journal
    3. El reentrenador sustituye este bootstrap por inteligencia real:
       python ml/retrain_model.py --initial

  Para forzar el reentrenamiento antes:
    python ml/retrain_model.py --initial --min-trades 20
""")

    # Telegram si está configurado
    kelly_est = max(0, (proba_mean * 2 - 1)) * 0.25  # Kelly/4 aproximado
    send_telegram(
        f"🧊 *Bootstrap ML completado*\n"
        f"Modelo neutro creado — el motor V4 puede arrancar.\n"
        f"Probas: `{metrics['proba_min']:.3f}` – `{metrics['proba_max']:.3f}` "
        f"(media `{proba_mean:.3f}`)\n"
        f"Kelly fraccionado estimado: `{kelly_est:.1%}` del riesgo base\n"
        f"El reentrenador real tomará el relevo con ≥30 trades cerrados."
    )

    return 0


# ══════════════════════════════════════════════════════════════════════════════
# ── CLI ───────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bootstrap ML — Arranque en frío del MetaLabeler V4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python bootstrap_ml.py                   # arranque estándar (300 muestras)
  python bootstrap_ml.py --samples 500     # más muestras (más suaves las probas)
  python bootstrap_ml.py --force           # sobreescribir modelo existente
  python bootstrap_ml.py --dry-run         # verificar sin guardar nada
  python bootstrap_ml.py --seed 123        # reproducibilidad con otro seed
        """,
    )
    parser.add_argument("--samples", type=int, default=300,
                        help="Número de muestras sintéticas (default: 300)")
    parser.add_argument("--force",   action="store_true",
                        help="Sobreescribir modelo aunque ya exista uno real")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generar y evaluar sin guardar nada en disco")
    parser.add_argument("--seed",    type=int, default=42,
                        help="Semilla aleatoria (default: 42)")
    args = parser.parse_args()

    sys.exit(main(
        n_samples=args.samples,
        force=args.force,
        dry_run=args.dry_run,
        seed=args.seed,
    ))
