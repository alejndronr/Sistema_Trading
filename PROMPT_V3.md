# PROMPT MAESTRO v3.0 — MEJORA INCREMENTAL CON REPOS EXTERNOS
# Sistema de Trading Algorítmico — ZimaBlade
# Agente: Antigravity | Modo: mejora sobre sistema EN PRODUCCIÓN

---

## ⚠️ LEE ESTO ANTES DE TOCAR UNA SOLA LÍNEA

El sistema está corriendo en producción 24/7. Un crash no controlado
significa posiciones abiertas sin monitorización. Cada archivo modificado
debe pasar ast.parse() antes de ser entregado. Cada nueva dependencia
debe tener un fallback silencioso si falla el import.

---

## HARDWARE REAL — RESTRICCIONES ABSOLUTAS

```
CPU:  Intel Atom E3950 @ 1.60 GHz — 4 cores MUY LENTOS
      Sin AVX2. Sin instrucciones vectoriales avanzadas.
      Rendimiento ~10x inferior a un i7 moderno.
RAM:  16 GB — máximo 4 GB para el engine en producción
HDD:  1.5 TB mecánico — latencia de escritura alta (5-10ms por seek)
      I/O secuencial: ~100 MB/s. I/O aleatorio: miserable.
SO:   Debian 12 LXC sobre Proxmox
```

### Lo que esto significa en código:
- RandomForest: máximo n_estimators=100, max_depth=6, n_jobs=2
  (n_jobs=4 satura el Atom y bloquea el loop de trading)
- PROHIBIDO en loops de trading: numpy linalg, scipy stats, modelos sklearn
- PERMITIDO en loop 60s: operaciones pandas vectorizadas sobre <500 filas
- PERMITIDO en background (retraining nocturno): sklearn con límite de tiempo
- PROHIBIDO siempre: LSTM, PyTorch, TensorFlow, LightGBM con 1000+ iteraciones
- HDD: toda escritura asíncrona, gráficos en BytesIO (RAM), nunca en disco

---

## ESTADO ACTUAL DEL SISTEMA — CÓDIGO REAL REVISADO

### Arquitectura (leída en auditoría previa — NO alterar sin razón)
```
Sistema_Trading/
├── config/settings.py          # Pydantic BaseSettings, atr_stop=1.5
├── data/
│   ├── fetcher.py              # ccxt async, SQLite WAL + PostgreSQL
│   └── storage.py              # interfaz unificada BD
├── indicators/
│   └── technical.py            # EMA/RSI/MACD/ATR/BB/ADX/OBV/SMC custom
├── strategies/
│   └── signals.py              # TF(60%) + MR(30%) + BO(10%), cooldown numpy
├── risk/
│   ├── position_sizer.py       # 1% capital, guard div/0, escala por tramos
│   └── regime_filter.py        # BULL_TREND/RANGE/HIGH_VOL/BEAR_TREND
├── backtesting/
│   ├── engine.py               # event-driven, slippage+comisión realistas
│   └── metrics.py              # WR, PF, Sharpe, DD, Expectancy
├── ml/
│   ├── meta_labeler.py         # RF, TimeSeriesSplit(5), umbral 60% fijo
│   └── retrain_model.py        # systemd timer mensual, lock, backup
├── monitoring/
│   └── telegram_bot.py         # aiohttp puro, /status /pause /resume /kill
├── paper_portfolio.py          # SQLAlchemy async, persistencia PostgreSQL
├── live_engine.py              # loop_slow(4H) + loop_fast(60s)
└── deploy.sh                   # Debian 12, 9 pasos
```

### Bugs ya corregidos — NO reintroducir jamás
```python
# 1. asyncpg necesita objeto date, no string
today = date.today()           # CORRECTO
today = date.today().isoformat()  # INCORRECTO — crashea asyncpg

# 2. SUM puede devolver NULL sin COALESCE
COALESCE(SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END), 0)  # CORRECTO
SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END)                # INCORRECTO

# 3. Parámetro de _call() en TelegramBot
await self._call("getUpdates", params, http_method="GET")  # CORRECTO
await self._call("getUpdates", params, method="GET")       # INCORRECTO

# 4. Cooldown de señales en pandas 3.x
sig = df["tf_long_signal"].to_numpy().copy()  # CORRECTO
sig = df["tf_long_signal"].copy()             # INCORRECTO (ChainedAssignment)

# 5. PostgreSQL en Debian — siempre 127.0.0.1 no localhost
DATABASE_URL=postgresql+asyncpg://user:pass@127.0.0.1:5432/db  # CORRECTO
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/db   # INCORRECTO

# 6. .env sin comentarios inline
INITIAL_CAPITAL=300.0           # CORRECTO
INITIAL_CAPITAL=300.0  # comentario  # INCORRECTO — float() falla
```

### Parámetros del sistema — NO cambiar
```
Capital:         300 USD (compounding automático)
Exchange:        Binance Spot (sin futuros ni apalancamiento)
Pares:           BTC/USDT, ETH/USDT
Timeframe:       4H (1H descartado por ruido)
Riesgo/trade:    1% capital (max 2% setups A+)
Stop Loss:       mínimo 1.5 ATR (1.0 ATR es cazado en Binance)
DD diario:       3% → stop día
DD semanal:      6% → revisar sistema
DD mensual:      10% → solo estudio
Max posiciones:  3 simultáneas
Python:          3.11+ — pandas_ta NO funciona, usar `ta`
```

---

## DIAGNÓSTICO: POR QUÉ NO HAY SEÑALES

Antes de añadir mejoras, hay que corregir el problema raíz: el sistema
lleva días sin generar señales. La auditoría reveló que con los filtros
actuales (EMA_bull AND ADX>25 AND RSI 45-65 AND MACD_bull AND MACD_growing
AND pullback a EMA21) solo se generan 2 señales en 4380 velas (2 años).
Esto es over-filtering severo.

### El cuello de botella identificado
```
EMA bullish + ADX strong:    527 velas (12%)
+ RSI neutral:               224 velas (5%)
+ MACD bull AND growing:       6 velas (0.1%)  ← AQUÍ MUERE TODO
+ pullback/OB:                 2 velas (0.05%) ← resultado final
```

MACD bull (>0) Y MACD growing simultáneamente es casi imposible
durante un pullback a la EMA21. Son condiciones contradictorias:
cuando el precio retrocede a la EMA21, el MACD normalmente ya no
está creciendo.

---

## MEJORAS A IMPLEMENTAR — ORDEN ESTRICTO

### MEJORA 0 (CRÍTICA): Corregir el over-filtering de señales
**Archivo**: `strategies/signals.py`
**Prioridad**: MÁXIMA — sin esto el sistema no opera nunca

#### Lógica corregida para signal_trend_following():

```python
def signal_trend_following(df: pd.DataFrame) -> pd.DataFrame:
    """
    Corrección del over-filtering: MACD cambia de AND a OR.
    El pullback se amplía de ±0.5% a ±2% (más realista en 4H).
    Añadimos confirmación de Consensus como alternativa al MACD estricto.
    """
    df = df.copy()
    c = df["close"]

    # Condiciones principales — sin cambios
    cond_ema  = df["ema_bullish"].fillna(False)
    cond_adx  = df["trending_strong"].fillna(False)   # ADX > 20 (era 25, bajar)
    cond_rsi  = df["rsi_neutral"].fillna(False)        # RSI 40-70 (ampliar rango)

    # MACD: OR en vez de AND — la clave del fix
    cond_macd = (
        df["macd_bull"].fillna(False) |    # MACD positivo
        df["macd_growing"].fillna(False)   # O MACD creciendo (no ambos)
    )

    # Pullback ampliado: ±2% de EMA21 (en 4H el precio rara vez toca exacto)
    cond_pullback = (
        (c <= df["ema21"] * 1.02) &
        (c >= df["ema21"] * 0.98)
    )
    cond_ob  = df["ob_bull"].fillna(False)
    cond_fvg = df["fvg_bull"].fillna(False)  # añadir FVG como zona de entrada
    cond_vol = ~df["extreme_vol"].fillna(True)

    # Señal: EMA + ADX + RSI + MACD(OR) + zona de entrada (pullback OR OB OR FVG)
    df["tf_long_signal"] = (
        cond_ema & cond_adx & cond_rsi & cond_macd & cond_vol
        & (cond_pullback | cond_ob | cond_fvg)
    ).astype(int)

    # Cooldown 3 velas — numpy array (pandas 3.x safe)
    sig = df["tf_long_signal"].to_numpy().copy()
    for i in range(1, len(sig)):
        if sig[i] and sig[max(0, i - 3):i].any():
            sig[i] = 0
    df["tf_long_signal"] = sig

    # SL dinámico: 1.5 ATR bajo el mínimo de 5 velas
    price_risk = c - (df["low"].rolling(5).min() - df["atr"] * 1.5)
    df["tf_stop_long"] = df["low"].rolling(5).min() - df["atr"] * 1.5
    df["tf_tp1"] = c + price_risk * 2.0   # R/R = 2:1
    df["tf_tp2"] = c + price_risk * 3.0   # R/R = 3:1

    return df
```

#### Ajustes en config/settings.py:
```python
# Cambiar estos valores en IndicatorConfig o BacktestConfig:
adx_threshold: int = 20          # era 25, bajar para más señales
rsi_neutral_low: float = 40.0    # era 45
rsi_neutral_high: float = 70.0   # era 65
pullback_pct: float = 0.02       # era 0.005 (±0.5%)
```

#### Diagnóstico post-fix esperado:
Con estos cambios, el sistema debería generar entre 15-30 señales
por cada 500 velas (4H), lo que en producción equivale a 2-5 trades
por semana en BTC+ETH. Suficiente para estadística sin sobreoperar.

---

### MEJORA 1: Smart Money Concepts con fallback
**Repo**: `joshyattridge/smart-money-concepts`
**Archivo**: `indicators/technical.py`
**Install**: `pip install smart-money-concepts`

Reemplazar `add_market_structure()` con versión que usa la librería
externa si está disponible, o cae al código custom si no lo está.
El fallback es OBLIGATORIO — nunca crashear por una librería opcional.

```python
def add_market_structure(df: pd.DataFrame) -> pd.DataFrame:
    """
    SMC mejorado. Usa smart-money-concepts si está instalado,
    fallback a implementación custom si no.
    """
    try:
        import smartmoneyconcepts as smc_lib
        return _add_market_structure_smc(df, smc_lib)
    except ImportError:
        log.warning("smc_library_not_found_using_fallback")
        return _add_market_structure_custom(df)


def _add_market_structure_smc(df: pd.DataFrame, smc_lib) -> pd.DataFrame:
    """Implementación con librería joshyattridge/smart-money-concepts."""
    ohlc = df[["open", "high", "low", "close"]].copy()

    try:
        # Swing Highs/Lows
        swing = smc_lib.swing_highs_lows(ohlc, swing_length=10)
        df["swing_high"] = swing["HighLow"].eq(1).astype(int)
        df["swing_low"]  = swing["HighLow"].eq(-1).astype(int)

        # BOS y CHoCH
        bos = smc_lib.bos_choch(ohlc, swing, close_break=True)
        df["bos_bull"]   = bos["BOS"].eq(1).fillna(False)
        df["bos_bear"]   = bos["BOS"].eq(-1).fillna(False)
        df["choch_bull"] = bos["CHOCH"].eq(1).fillna(False)
        df["choch_bear"] = bos["CHOCH"].eq(-1).fillna(False)

        # Order Blocks
        ob = smc_lib.ob(ohlc, swing)
        df["ob_bull"] = ob["OB"].eq(1).fillna(False)
        df["ob_bear"] = ob["OB"].eq(-1).fillna(False)
        df["ob_top"]  = ob["Top"].fillna(np.nan)
        df["ob_bot"]  = ob["Bottom"].fillna(np.nan)

        # Fair Value Gaps
        fvg = smc_lib.fvg(ohlc, join_consecutive=True)
        df["fvg_bull"] = fvg["FVG"].eq(1).fillna(False)
        df["fvg_bear"] = fvg["FVG"].eq(-1).fillna(False)
        df["fvg_top"]  = fvg["Top"].fillna(np.nan)
        df["fvg_bot"]  = fvg["Bottom"].fillna(np.nan)

        # Equal Highs/Lows — NUEVO (liquidity pools)
        try:
            liq = smc_lib.liquidity(ohlc, swing, range_percent=0.01)
            df["liq_high"] = liq["Liquidity"].eq(1).fillna(False)
            df["liq_low"]  = liq["Liquidity"].eq(-1).fillna(False)
        except Exception:
            df["liq_high"] = False
            df["liq_low"]  = False

    except Exception as exc:
        log.warning("smc_library_error_using_fallback", error=str(exc))
        return _add_market_structure_custom(df)

    return df


def _add_market_structure_custom(df: pd.DataFrame) -> pd.DataFrame:
    """Fallback: implementación custom original."""
    # ... (mantener el código custom existente sin cambios)
    # Copiar aquí la función add_market_structure original actual
    # con todas sus columnas: ob_bull, ob_bear, bos_bull, fvg_bull, etc.
    pass
```

**IMPORTANTE**: La función `_add_market_structure_custom` debe contener
el código custom existente completo. No dejar `pass` — copiar el código
actual de `add_market_structure()` tal cual.

---

### MEJORA 2: VFI y Consensus Indicator
**Repo**: `freqtrade/technical` (inspiración, implementación propia)
**Archivo**: `indicators/technical.py`
**Nota CPU Atom**: Calcular solo en el loop lento (4H), nunca en el rápido (60s)

```python
def add_vfi(df: pd.DataFrame, period: int = 130) -> pd.DataFrame:
    """
    Volume Flow Indicator — más preciso que OBV para confirmar tendencia.
    Adaptado de freqtrade/technical para Atom E3950.
    Completamente vectorizado, sin loops Python.
    """
    tp = (df["high"] + df["low"] + df["close"]) / 3.0

    # Evitar log(0) con clip
    tp_safe = tp.clip(lower=1e-10)
    inter = np.log(tp_safe) - np.log(tp_safe.shift(1))
    vinter = inter.rolling(30).std().fillna(0.01)

    cutoff = 0.1 * vinter * df["close"]
    vave   = df["volume"].rolling(period).mean().shift(1).fillna(1)
    vmax   = vave * 2.0

    mf = tp - tp.shift(1)

    # Dirección del volumen (vectorizado)
    vcp = np.where(mf > cutoff,  df["volume"],
          np.where(mf < -cutoff, -df["volume"], 0.0))

    vf = pd.Series(vcp, index=df.index)
    vf = vf.clip(lower=-vmax, upper=vmax)

    vave_safe = vave.replace(0, np.nan)
    df["vfi"]      = vf.rolling(period).sum() / vave_safe
    df["vfi"]      = df["vfi"].fillna(0)
    df["vfi_bull"] = df["vfi"] > 0
    return df


def add_consensus(df: pd.DataFrame) -> pd.DataFrame:
    """
    Puntuación 0-100 que agrega múltiples indicadores.
    > 60: sesgo alcista | < 40: sesgo bajista | 40-60: neutral

    Diseñado para el Atom E3950: operaciones vectorizadas simples,
    sin numpy linalg ni scipy.

    Uso en signals.py: en vez de 6 condiciones AND que matan la
    frecuencia, usar consensus >= 55 como condición adicional suave.
    """
    weights = {
        "ema_bullish": 25,   # tendencia principal
        "rsi":         20,   # momentum normalizado
        "macd_bull":   20,   # momentum MACD
        "adx":         20,   # fuerza tendencia
        "vfi_bull":    15,   # volumen real
    }

    score = pd.Series(0.0, index=df.index)
    total_weight = sum(weights.values())

    if "ema_bullish" in df.columns:
        score += df["ema_bullish"].fillna(False).astype(float) * weights["ema_bullish"]

    if "rsi" in df.columns:
        rsi_norm = ((df["rsi"].clip(30, 70) - 30) / 40.0)
        score += rsi_norm * weights["rsi"]

    if "macd_bull" in df.columns:
        score += df["macd_bull"].fillna(False).astype(float) * weights["macd_bull"]

    if "adx" in df.columns:
        adx_norm = df["adx"].clip(0, 50) / 50.0
        score += adx_norm * weights["adx"]

    if "vfi_bull" in df.columns:
        score += df["vfi_bull"].fillna(False).astype(float) * weights["vfi_bull"]

    df["consensus"]      = (score / total_weight * 100).clip(0, 100)
    df["consensus_bull"] = df["consensus"] >= 55
    df["consensus_bear"] = df["consensus"] <= 40
    return df
```

#### Integrar en apply_all_indicators() al final:
```python
def apply_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # ... (mantener todas las llamadas actuales)
    df = add_vfi(df)        # AÑADIR
    df = add_consensus(df)  # AÑADIR
    return df
```

#### Usar Consensus en signal_trend_following() como condición suave:
```python
# En strategies/signals.py, añadir a las condiciones de TF:
cond_consensus = df.get("consensus_bull", pd.Series(True, index=df.index))

df["tf_long_signal"] = (
    cond_ema & cond_adx & cond_rsi & cond_macd & cond_vol
    & (cond_pullback | cond_ob | cond_fvg)
    & cond_consensus   # AÑADIR — filtro suave, no bloquea si consensus=NaN
).astype(int)
```

---

### MEJORA 3: MetaLabeler v2 — Features avanzadas
**Repo**: `asavinov/intelligent-trading-bot` (inspiración)
**Archivo**: `ml/meta_labeler.py`
**Nota CPU Atom**: n_estimators=100 máximo, n_jobs=2

#### Constante actualizada:
```python
FEATURES_V2 = [
    # Base (existentes)
    "rsi", "adx", "atr_pct", "ema21_dist_pct", "ema55_dist_pct",
    "macd_hist", "bb_width", "vol_ratio", "cvd_bull", "regime_encoded",
    # Lag (nuevas)
    "rsi_lag1", "rsi_lag3", "adx_lag1",
    "close_chg_1", "close_chg_3",
    # Rolling (nuevas)
    "rsi_roll5_mean", "rsi_roll5_std", "vol_roll10_mean",
    # Consensus (de Mejora 2)
    "consensus",
    # SMC (de Mejora 1)
    "ob_bull_recent5", "fvg_bull_recent3",
]
```

#### Método build_feature_matrix() — TOTALMENTE VECTORIZADO:
```python
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
    price  = df["close"].values.astype(float)
    n      = len(price)

    def col(name: str, default: float = 0.0) -> np.ndarray:
        """Extrae columna con default seguro."""
        if name in df.columns:
            return df[name].fillna(default).values.astype(float)
        return np.full(n, default, dtype=float)

    rsi    = col("rsi", 50)
    adx    = col("adx", 20)
    atr    = col("atr", price * 0.015)
    ema21  = col("ema_21", price)
    ema55  = col("ema_55", price)
    macd_h = col("macd_hist", 0)
    bb_up  = col("bb_upper", price * 1.02)
    bb_lo  = col("bb_lower", price * 0.98)
    bb_mid = col("bb_mid", price)
    vol    = col("volume", 1)
    cvd    = col("cvd", 0)
    regime_raw = df.get("regime", pd.Series("RANGE", index=df.index))
    regime_map = {"BULL_TREND": 1.0, "BULL": 1.0,
                  "RANGE": 0.0, "HIGH_VOL": -0.5, "BEAR_TREND": -1.0}
    regime = regime_raw.map(regime_map).fillna(0).values.astype(float)
    consensus = col("consensus", 50)

    # Volume moving average (vectorizado)
    vol_s   = pd.Series(vol)
    vol_ma  = vol_s.rolling(20, min_periods=1).mean().values
    vol_ma  = np.where(vol_ma == 0, 1, vol_ma)

    # Features base
    safe_price = np.where(price == 0, 1e-10, price)
    safe_ema21 = np.where(ema21 == 0, 1e-10, ema21)
    safe_ema55 = np.where(ema55 == 0, 1e-10, ema55)
    safe_bbmid = np.where(bb_mid == 0, 1e-10, bb_mid)

    f_rsi         = rsi
    f_adx         = adx
    f_atr_pct     = atr / safe_price * 100
    f_ema21_dist  = (price - ema21) / safe_ema21 * 100
    f_ema55_dist  = (price - ema55) / safe_ema55 * 100
    f_macd_hist   = macd_h
    f_bb_width    = (bb_up - bb_lo) / safe_bbmid
    f_vol_ratio   = vol / vol_ma
    f_cvd_bull    = (cvd > 0).astype(float)
    f_regime      = regime
    f_consensus   = consensus

    # Lag features (shift con padding)
    def shift_pad(arr: np.ndarray, n: int, fill: float = 0) -> np.ndarray:
        out = np.empty_like(arr)
        out[:n] = fill
        out[n:] = arr[:-n]
        return out

    f_rsi_lag1    = shift_pad(rsi, 1, 50)
    f_rsi_lag3    = shift_pad(rsi, 3, 50)
    f_adx_lag1    = shift_pad(adx, 1, 20)
    f_chg1        = np.diff(price, prepend=price[0]) / safe_price * 100
    f_chg3        = np.where(
        np.arange(n) >= 3,
        (price - shift_pad(price, 3, price[0])) / safe_price * 100,
        0.0
    )

    # Rolling stats (vectorizado con pandas)
    rsi_s         = pd.Series(rsi)
    f_rsi_mean5   = rsi_s.rolling(5, min_periods=1).mean().values
    f_rsi_std5    = rsi_s.rolling(5, min_periods=1).std().fillna(5).values
    f_vol_mean10  = vol_s.rolling(10, min_periods=1).mean().values

    # SMC recientes (rolling max vectorizado)
    ob_bull       = col("ob_bull", 0)
    fvg_bull      = col("fvg_bull", 0)
    f_ob_recent   = pd.Series(ob_bull).rolling(5, min_periods=1).max().values
    f_fvg_recent  = pd.Series(fvg_bull).rolling(3, min_periods=1).max().values

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
    mu  = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    X   = (X - mu) / std

    return X
```

#### RandomForest actualizado para Atom E3950:
```python
# En el método train() de MetaLabeler:
model = RandomForestClassifier(
    n_estimators=100,    # máximo para Atom (era probablemente 200)
    max_depth=6,
    min_samples_leaf=10,
    max_features="sqrt",
    class_weight="balanced",
    n_jobs=2,            # dejar 2 cores para el engine
    random_state=42,
)
```

#### Threshold optimization (grid ligero para Atom):
```python
def _optimize_threshold(
    self, model, X_val: np.ndarray, y_val: np.ndarray
) -> float:
    """
    Grid de 9 puntos (0.40→0.80, paso 0.05).
    En Atom E3950: < 100ms para 500 muestras.
    """
    from sklearn.metrics import f1_score
    probas = model.predict_proba(X_val)[:, 1]
    best_thresh, best_f1 = 0.50, 0.0

    for thresh in np.arange(0.40, 0.81, 0.05):
        preds = (probas >= thresh).astype(int)
        if preds.sum() == 0:
            continue
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, float(thresh)

    log.info("threshold_optimized", threshold=f"{best_thresh:.2f}", f1=f"{best_f1:.4f}")
    return best_thresh
```

#### Guardar y cargar threshold en model_metadata.json:
```python
# Al guardar el modelo, incluir el threshold en metadata:
metadata = {
    "trained_at": datetime.utcnow().isoformat(),
    "model_version": MODEL_VERSION,
    "n_samples": n_samples,
    "optimal_threshold": best_thresh,   # NUEVO
    "metrics": metrics_dict,
}
# En live_engine.py, leer el threshold:
meta = self._ml.get_metadata()
ML_THRESHOLD = meta.get("optimal_threshold", 0.60)
```

#### predict_proba() actualizado:
```python
def predict_proba(self, df: pd.DataFrame) -> float:
    """
    Recibe el DataFrame COMPLETO (no solo la última fila)
    para poder calcular lag features y rolling stats.
    Devuelve la probabilidad de la última fila.
    """
    if self._model is None:
        return 0.5   # modo permisivo si no hay modelo

    try:
        X = self.build_feature_matrix(df)
        if len(X) == 0:
            return 0.5
        proba = self._model.predict_proba(X[-1:])[:, 1][0]
        return float(proba)
    except Exception as exc:
        log.warning("predict_proba_failed", error=str(exc))
        return 0.5
```

---

### MEJORA 4: Visualización de trades por Telegram
**Repo**: `matplotlib/mplfinance`
**Archivo**: `monitoring/telegram_bot.py`
**Restricción crítica**: En BytesIO (RAM), nunca en disco (HDD mecánico)

```python
# Añadir al inicio de telegram_bot.py:
import io
try:
    import matplotlib
    matplotlib.use("Agg")    # headless, sin display
    import mplfinance as mpf
    _MPLFINANCE_AVAILABLE = True
except ImportError:
    _MPLFINANCE_AVAILABLE = False

# Añadir método a la clase TelegramBot:
async def send_trade_chart(
    self,
    trade: dict,
    df: pd.DataFrame,
    reason: str,
) -> None:
    """
    Genera gráfico de velas con entry/SL/TP marcados.
    Completamente en RAM (BytesIO) — nunca toca el HDD mecánico.
    Si mplfinance no está instalado, el método no hace nada (no crashea).
    """
    if not self._ready or not self._session or not _MPLFINANCE_AVAILABLE:
        return

    try:
        # Últimas 50 velas (suficiente contexto, ligero para el Atom)
        chart_df = df.tail(50).copy()

        # Preparar índice temporal
        if "timestamp" in chart_df.columns:
            chart_df.index = pd.to_datetime(chart_df["timestamp"], utc=True)
        chart_df = chart_df[["open", "high", "low", "close", "volume"]].copy()
        chart_df.columns = ["Open", "High", "Low", "Close", "Volume"]
        chart_df = chart_df.dropna()

        if len(chart_df) < 10:
            return

        entry = trade.get("entry_price", 0)
        sl    = trade.get("stop_loss", 0)
        tp1   = trade.get("tp1", 0)
        tp2   = trade.get("tp2", 0)
        pnl   = trade.get("pnl", 0)
        sym   = trade.get("symbol", "")

        # Solo niveles válidos
        levels  = [v for v in [entry, sl, tp1, tp2] if v > 0]
        colors  = ["white", "red", "lime", "cyan"][:len(levels)]
        styles  = ["--", "-", "--", "--"][:len(levels)]

        hlines = dict(hlines=levels, colors=colors,
                      linestyle=styles, linewidths=[1]*len(levels))

        title = f"{sym} | {reason} | PnL: {pnl:+.2f} USD"

        style = mpf.make_mpf_style(
            base_mpf_style="nightclouds",
            rc={"font.size": 8},
        )

        # Generar en BytesIO (RAM pura)
        buf = io.BytesIO()
        mpf.plot(
            chart_df,
            type="candle",
            style=style,
            title=title,
            volume=True,
            hlines=hlines if levels else {},
            figsize=(9, 5),   # más pequeño para menos RAM
            savefig=dict(fname=buf, dpi=80, bbox_inches="tight"),
        )
        buf.seek(0)
        img_bytes = buf.read()
        buf.close()  # liberar RAM inmediatamente

        # Enviar por Telegram
        url  = self._BASE.format(token=self._token, method="sendPhoto")
        form = aiohttp.FormData()
        form.add_field("chat_id", str(self._allowed_chat_id))
        form.add_field("caption", title)
        form.add_field("photo", img_bytes,
                       filename="trade.png", content_type="image/png")

        async with self._session.post(url, data=form) as resp:
            if resp.status != 200:
                log.warning("chart_send_failed", status=resp.status)

    except Exception as exc:
        # NUNCA crashear el engine por un gráfico fallido
        log.warning("chart_generation_failed", error=str(exc))
```

---

### MEJORA 5: live_engine.py — adaptaciones necesarias

#### 5a. Pasar DataFrame completo al MetaLabeler (no solo la última fila):
```python
# En _process_symbol(), sustituir:
# ANTES:
features = self._extract_features(last)
ml_proba = self._ml.predict_proba(features) if self._ml else 0.5

# DESPUÉS:
ml_proba = self._ml.predict_proba(df) if self._ml else 0.5
# (predict_proba ahora recibe el DataFrame completo y extrae la última fila)
```

#### 5b. Leer threshold dinámico del modelo:
```python
# En start(), después de cargar el modelo ML:
if self._ml and self._ml.is_ready():
    meta = self._ml.get_metadata()
    self._ml_threshold = meta.get("optimal_threshold", 0.60)
    log.info("ml_threshold_loaded", threshold=self._ml_threshold)
else:
    self._ml_threshold = 0.60

# En _process_symbol(), usar self._ml_threshold en vez de ML_THRESHOLD constante
if ml_proba < self._ml_threshold:
    ...
```

#### 5c. Enviar chart al cerrar un trade:
```python
# En _loop_fast(), después de send_trade_close():
for ct in closed_trades:
    if self._bot:
        await self._bot.send_trade_close(ct, ct["exit_reason"])
        # AÑADIR:
        if hasattr(self, '_last_df') and self._bot:
            symbol = ct.get("symbol", "")
            df_chart = self._last_df.get(symbol)
            if df_chart is not None:
                await self._bot.send_trade_chart(ct, df_chart, ct["exit_reason"])

# En _process_symbol(), guardar el DataFrame para los charts:
if not hasattr(self, '_last_df'):
    self._last_df = {}
self._last_df[symbol] = df   # guardar referencia al df con indicadores
```

---

### MEJORA 6: requirements.txt actualizado

```text
# Core (existentes — no cambiar versiones)
ccxt>=4.3.0
pandas>=2.2.0
ta>=0.11.0
numpy>=1.26.0
scikit-learn>=1.4.0
joblib>=1.3.0
psycopg2-binary>=2.9.9
aiohttp>=3.9.0
websockets>=12.0
python-telegram-bot>=21.0
plotly>=5.20.0
python-dotenv>=1.0.0
structlog>=24.0.0
sqlalchemy>=2.0.0
asyncpg>=0.29.0
pydantic-settings>=2.0.0
tqdm>=4.0.0

# NUEVAS — opcionales con fallback
smart-money-concepts>=0.0.10   # fallback: código custom
mplfinance>=0.12.10b0          # fallback: no charts por Telegram

# NO añadir — incompatibles con Atom E3950:
# lightgbm        → demasiado pesado para Atom sin GPU
# pytorch          → imposible sin GPU
# tensorflow       → imposible sin GPU
```

---

### MEJORA 7: deploy.sh — verificaciones adicionales

Añadir al paso 5 del deploy, después de `pip install -r requirements.txt`:

```bash
echo "Verificando librerías opcionales..."

$VENV/bin/python -c "import smartmoneyconcepts; print('[OK] smart-money-concepts')" \
    2>/dev/null || echo "[WARN] smart-money-concepts no disponible — usando fallback custom"

$VENV/bin/python -c "import mplfinance; print('[OK] mplfinance — charts habilitados')" \
    2>/dev/null || echo "[WARN] mplfinance no disponible — charts Telegram desactivados"

# Test de rendimiento en Atom E3950 (verificar que el RF no es demasiado lento)
$VENV/bin/python -c "
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
X = np.random.rand(500, 21).astype(np.float32)
y = (X[:, 0] > 0.5).astype(int)
rf = RandomForestClassifier(n_estimators=100, max_depth=6, n_jobs=2, random_state=42)
t0 = time.time()
rf.fit(X, y)
elapsed = time.time() - t0
if elapsed < 30:
    print(f'[OK] RF training: {elapsed:.1f}s (dentro del límite)')
else:
    print(f'[WARN] RF training: {elapsed:.1f}s (muy lento — reducir n_estimators a 50)')
"
```

---

## ORDEN DE ENTREGA — ESTRICTO

```
1. strategies/signals.py      ← CRÍTICO: sin esto el sistema no opera
2. config/settings.py         ← ajustar ADX y RSI thresholds
3. requirements.txt           ← añadir dependencias opcionales
4. indicators/technical.py    ← SMC + VFI + Consensus
5. ml/meta_labeler.py         ← Features v2 + Atom-safe RF + threshold opt
6. monitoring/telegram_bot.py ← charts en BytesIO
7. live_engine.py             ← pasar df completo a ML + charts + threshold dinámico
8. deploy.sh                  ← verificaciones opcionales
```

---

## VERIFICACIONES OBLIGATORIAS POST-ENTREGA

```bash
# 1. Syntax check completo
python -c "
import ast, glob, sys
errors = []
for f in glob.glob('**/*.py', recursive=True):
    if 'venv' in f or '__pycache__' in f:
        continue
    try:
        ast.parse(open(f, encoding='utf-8').read())
    except SyntaxError as e:
        errors.append(f'{f}: {e}')
        print(f'ERR {f}: {e}')
if not errors:
    print(f'OK — todos los archivos sin errores de sintaxis')
sys.exit(len(errors))
"

# 2. Import check
python -c "
import sys; sys.path.insert(0, '.')
from indicators.technical import apply_all_indicators, add_vfi, add_consensus
from strategies.signals import apply_all_signals
from ml.meta_labeler import MetaLabeler
from monitoring.telegram_bot import TelegramBot
from paper_portfolio import PaperPortfolio
import live_engine
print('OK — todos los imports resuelven')
"

# 3. Test de señales (verificar que ya no hay over-filtering)
python -c "
import sys, warnings; sys.path.insert(0,'.'); warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
# Generar 500 velas sintéticas
np.random.seed(42)
n = 500
closes = 50000 * np.cumprod(1 + np.random.randn(n)*0.01)
df = pd.DataFrame({
    'open': closes*0.999, 'high': closes*1.002,
    'low': closes*0.998, 'close': closes,
    'volume': np.random.lognormal(12, 0.5, n),
    'timestamp': pd.date_range('2024-01-01', periods=n, freq='4h')
})
from indicators.technical import apply_all_indicators
from strategies.signals import apply_all_signals
df = apply_all_indicators(df)
df = apply_all_signals(df)
tf = df['tf_long_signal'].sum()
mr = df['mr_long_signal'].sum()
print(f'Señales TF: {tf} | MR: {mr} | Total: {tf+mr}')
if tf + mr < 5:
    print('WARN: muy pocas señales — over-filtering persiste')
elif tf + mr > 50:
    print('WARN: demasiadas señales — under-filtering')
else:
    print('OK — frecuencia de señales correcta')
"

# 4. SMC availability
python -c "
try:
    import smartmoneyconcepts as smc
    print('OK — smart-money-concepts disponible')
except ImportError:
    print('INFO — smart-money-concepts no instalado, usando fallback custom')
"

# 5. Performance test para Atom E3950
python -c "
import time, numpy as np
from sklearn.ensemble import RandomForestClassifier
X = np.random.rand(300, 21).astype(np.float32)
y = (X[:, 0] > 0.5).astype(int)
rf = RandomForestClassifier(n_estimators=100, max_depth=6, n_jobs=2, random_state=42)
t = time.time()
rf.fit(X, y)
elapsed = time.time() - t
print(f'RF training en hardware objetivo: {elapsed:.2f}s')
print('OK' if elapsed < 60 else 'WARN: demasiado lento, reducir n_estimators')
"
```

---

## LO QUE ESTÁ PROHIBIDO — RESUMEN FINAL

```
PROHIBIDO por CPU Atom E3950:
  ✗ n_estimators > 100 en RandomForest
  ✗ LightGBM, XGBoost, CatBoost con > 100 iteraciones
  ✗ LSTM, GRU, Transformer, PyTorch, TensorFlow
  ✗ scipy.optimize o scipy.stats en el loop de trading
  ✗ numpy linalg operations en el loop de 60 segundos
  ✗ n_jobs > 2 en sklearn (bloquea los 4 cores del Atom)

PROHIBIDO por HDD mecánico:
  ✗ Escritura de archivos en loops de trading
  ✗ Generar imágenes/CSV/logs a disco en tiempo real
  ✗ Gráficos mplfinance guardados en disco (usar BytesIO)
  ✗ SQLite en modo WAL con sync=FULL (usar NORMAL o OFF)
  ✗ Múltiples conexiones de BD en paralelo en el loop rápido

PROHIBIDO por RAM 16GB (4GB disponibles para engine):
  ✗ Cargar DataFrames de más de 10.000 filas en memoria durante trading
  ✗ Modelos ML > 500MB en RAM
  ✗ Cachear velas de múltiples timeframes simultáneamente
  ✗ Docker con múltiples servicios pesados simultáneos

PROHIBIDO por estabilidad del sistema en producción:
  ✗ Cambiar el esquema de PostgreSQL sin migración
  ✗ Modificar la interfaz pública de PaperPortfolio sin adaptar live_engine
  ✗ Eliminar columnas existentes de DataFrames (solo añadir)
  ✗ Cambiar DATABASE_URL format sin actualizar paper_portfolio.py
```

---

## SIGUIENTE FASE — CUANDO HAYA 3+ MESES DE DATOS

Una vez validado en live con datos reales, la siguiente iteración implementará:
- Walk-forward validation en retrain_model.py (backtesting.py como referencia)
- Dashboard web con FastAPI + HTMX (sin React, mínimo consumo en Atom)
- HyperOpt automático de parámetros con Optuna (ligero, compatible con Atom)
- Segunda estrategia: Mean Reversion mejorada con régimen de rango confirmado

---

FIN DEL PROMPT MAESTRO v3.0
Sistema de Trading — ZimaBlade (Atom E3950)
Capital: 300 USD | Modo: PAPER → LIVE tras 4 semanas validación
```
