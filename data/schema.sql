-- ══════════════════════════════════════════════════════════════════════════════
-- Sistema de Trading — Schema PostgreSQL (Fase 2)
-- ══════════════════════════════════════════════════════════════════════════════
-- Ejecutar:
--   psql -U trading -d trading_db -f schema.sql
-- O via docker compose (se ejecuta automáticamente en primer boot):
--   docker compose exec postgres psql -U trading -d trading_db -f /schema/schema.sql
-- ══════════════════════════════════════════════════════════════════════════════

-- Extensiones útiles
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";   -- Para gen_random_uuid()

-- ── Tabla OHLCV ───────────────────────────────────────────────────────────────
-- Almacena las velas históricas de precios (OHLCV) para todos los pares
-- Upsert-ready: UNIQUE (symbol, timeframe, timestamp) → ON CONFLICT DO UPDATE

CREATE TABLE IF NOT EXISTS ohlcv (
    id          BIGSERIAL    PRIMARY KEY,
    symbol      VARCHAR(20)  NOT NULL,
    timeframe   VARCHAR(5)   NOT NULL,
    timestamp   TIMESTAMPTZ  NOT NULL,
    open        DOUBLE PRECISION NOT NULL,
    high        DOUBLE PRECISION NOT NULL,
    low         DOUBLE PRECISION NOT NULL,
    close       DOUBLE PRECISION NOT NULL,
    volume      DOUBLE PRECISION NOT NULL,

    CONSTRAINT uq_ohlcv_key UNIQUE (symbol, timeframe, timestamp)
);

-- Índice principal para consultas de rango (el más frecuente en backtesting)
CREATE INDEX IF NOT EXISTS idx_ohlcv_lookup
    ON ohlcv (symbol, timeframe, timestamp ASC);

-- Índice para consultas de símbolo solo (ej: listar todos los timeframes de BTC)
CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol
    ON ohlcv (symbol);

COMMENT ON TABLE ohlcv IS 'Velas OHLCV históricas. Upsert por (symbol, timeframe, timestamp).';

-- ── Tabla TRADES ──────────────────────────────────────────────────────────────
-- Journal completo de trades (backtest + live paper trading)

CREATE TABLE IF NOT EXISTS trades (
    id              BIGSERIAL    PRIMARY KEY,
    trade_id        UUID         NOT NULL DEFAULT gen_random_uuid() UNIQUE,

    -- Identificación del trade
    strategy        VARCHAR(30)  NOT NULL,
    symbol          VARCHAR(20)  NOT NULL,
    timeframe       VARCHAR(5)   NOT NULL,
    direction       VARCHAR(5)   NOT NULL CHECK (direction IN ('LONG', 'SHORT')),
    setup_quality   VARCHAR(3)   CHECK (setup_quality IN ('A+', 'A', 'B', 'C')),

    -- Precios
    entry_price     DOUBLE PRECISION NOT NULL,
    stop_loss       DOUBLE PRECISION NOT NULL,
    take_profit_1   DOUBLE PRECISION NOT NULL,
    take_profit_2   DOUBLE PRECISION,
    exit_price      DOUBLE PRECISION,

    -- Posición y riesgo
    position_size   DOUBLE PRECISION NOT NULL,
    risk_amount     DOUBLE PRECISION NOT NULL,

    -- Resultado
    pnl_usd         DOUBLE PRECISION,
    pnl_pct         DOUBLE PRECISION,
    r_multiple      DOUBLE PRECISION,

    -- Timing
    entry_time      TIMESTAMPTZ,
    exit_time       TIMESTAMPTZ,
    duration_hours  DOUBLE PRECISION,

    -- Journal
    entry_reason    TEXT,
    exit_reason     TEXT,
    observations    TEXT,
    market_regime   VARCHAR(30),

    -- Meta
    is_backtest     BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trades_symbol   ON trades (symbol);
CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades (strategy);
CREATE INDEX IF NOT EXISTS idx_trades_entry    ON trades (entry_time);
CREATE INDEX IF NOT EXISTS idx_trades_backtest ON trades (is_backtest);

COMMENT ON TABLE trades IS 'Journal completo de trades. is_backtest=TRUE para backtest, FALSE para live.';

-- ── Tabla DRAWDOWNS ───────────────────────────────────────────────────────────
-- Registro de drawdowns para el circuit breaker

CREATE TABLE IF NOT EXISTS drawdowns (
    id                       BIGSERIAL   PRIMARY KEY,
    date                     TIMESTAMPTZ NOT NULL,
    period                   VARCHAR(10) NOT NULL CHECK (period IN ('daily', 'weekly', 'monthly')),
    drawdown_pct             DOUBLE PRECISION NOT NULL,
    capital_at_peak          DOUBLE PRECISION NOT NULL,
    capital_current          DOUBLE PRECISION NOT NULL,
    circuit_breaker_triggered BOOLEAN DEFAULT FALSE,

    CONSTRAINT uq_drawdown_date_period UNIQUE (date, period)
);

CREATE INDEX IF NOT EXISTS idx_drawdowns_date ON drawdowns (date DESC);

COMMENT ON TABLE drawdowns IS 'Registro de drawdowns para activar circuit breakers.';

-- ── Tabla ML_PREDICTIONS ──────────────────────────────────────────────────────
-- Registro de predicciones del MetaLabelModel para auditoría y análisis

CREATE TABLE IF NOT EXISTS ml_predictions (
    id                  BIGSERIAL    PRIMARY KEY,
    trade_id            UUID         REFERENCES trades(trade_id) ON DELETE SET NULL,
    predicted_at        TIMESTAMPTZ  DEFAULT NOW(),
    symbol              VARCHAR(20)  NOT NULL,
    timeframe           VARCHAR(5)   NOT NULL,
    signal_timestamp    TIMESTAMPTZ  NOT NULL,

    -- Features usadas (snapshot del mercado en el momento de la señal)
    feat_adx            DOUBLE PRECISION,
    feat_rsi            DOUBLE PRECISION,
    feat_macd_histogram DOUBLE PRECISION,
    feat_macd_line      DOUBLE PRECISION,
    feat_dist_ema21_pct DOUBLE PRECISION,
    feat_dist_ema55_pct DOUBLE PRECISION,
    feat_dist_ema200_pct DOUBLE PRECISION,
    feat_atr            DOUBLE PRECISION,
    feat_atr_ratio      DOUBLE PRECISION,
    feat_cvd_pressure   DOUBLE PRECISION,

    -- Predicción del modelo
    prob_win            DOUBLE PRECISION NOT NULL,  -- P(label=1), rango [0,1]
    threshold_used      DOUBLE PRECISION NOT NULL,  -- Umbral en el momento
    trade_approved      BOOLEAN NOT NULL,           -- prob_win >= threshold_used

    -- Resultado real (se actualiza cuando cierra el trade)
    actual_outcome      BOOLEAN,   -- TRUE=ganador, FALSE=perdedor, NULL=pendiente
    model_version       VARCHAR(50)  -- Nombre del archivo .joblib usado
);

CREATE INDEX IF NOT EXISTS idx_ml_pred_symbol ON ml_predictions (symbol, signal_timestamp);

COMMENT ON TABLE ml_predictions IS 'Auditoría de predicciones del MetaLabelModel. Permite análisis de drift del modelo.';

-- ── Vistas útiles ─────────────────────────────────────────────────────────────

-- Vista resumen de datos disponibles (equivale a get_available_data())
CREATE OR REPLACE VIEW v_data_summary AS
SELECT
    symbol,
    timeframe,
    COUNT(*)    AS candles,
    MIN(timestamp) AS from_date,
    MAX(timestamp) AS to_date,
    MAX(timestamp) - MIN(timestamp) AS data_span
FROM ohlcv
GROUP BY symbol, timeframe
ORDER BY symbol, timeframe;

-- Vista de performance del ML por período
CREATE OR REPLACE VIEW v_ml_performance AS
SELECT
    model_version,
    COUNT(*)                        AS total_predictions,
    SUM(CASE WHEN trade_approved THEN 1 ELSE 0 END) AS trades_approved,
    AVG(prob_win)::NUMERIC(5,3)     AS avg_prob_win,
    -- Precision: de los trades aprobados, cuántos ganaron
    SUM(CASE WHEN trade_approved AND actual_outcome THEN 1 ELSE 0 END)::FLOAT
        / NULLIF(SUM(CASE WHEN trade_approved THEN 1 ELSE 0 END), 0) AS precision_ml,
    -- Recall: de los ganadores, cuántos fueron aprobados
    SUM(CASE WHEN trade_approved AND actual_outcome THEN 1 ELSE 0 END)::FLOAT
        / NULLIF(SUM(CASE WHEN actual_outcome THEN 1 ELSE 0 END), 0) AS recall_ml
FROM ml_predictions
WHERE actual_outcome IS NOT NULL
GROUP BY model_version;

-- ── Grants (para usuario de aplicación) ──────────────────────────────────────
-- Si usas un usuario separado para la app (recomendado en producción):
-- GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO trading_app;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO trading_app;

-- ── Fin del schema ────────────────────────────────────────────────────────────
\echo 'Schema de Trading DB inicializado correctamente.'
