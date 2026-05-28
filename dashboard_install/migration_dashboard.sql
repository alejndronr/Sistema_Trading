-- ══════════════════════════════════════════════════════════════════════════════
-- migration_dashboard.sql — Tablas adicionales para el Dashboard
-- ══════════════════════════════════════════════════════════════════════════════
-- Ejecutar UNA sola vez en el servidor:
--   psql -U trading -d trading_db -f dashboard/migration_dashboard.sql
-- ══════════════════════════════════════════════════════════════════════════════

-- ── Tabla: inversiones manuales (posiciones abiertas) ─────────────────────────
CREATE TABLE IF NOT EXISTS manual_investments (
    id          SERIAL       PRIMARY KEY,
    symbol      VARCHAR(20)  NOT NULL,
    amount      DECIMAL(18,8) NOT NULL,
    buy_price   DECIMAL(18,8) NOT NULL,
    buy_date    DATE         NOT NULL DEFAULT CURRENT_DATE,
    exchange    VARCHAR(50)  DEFAULT 'Binance',
    tx_type     VARCHAR(30)  DEFAULT 'buy',  -- buy | transfer_in | airdrop | staking_reward
    notes       TEXT         DEFAULT '',
    status      VARCHAR(10)  DEFAULT 'open', -- open | closed | deleted
    created_at  TIMESTAMPTZ  DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_manual_inv_symbol ON manual_investments(symbol);
CREATE INDEX IF NOT EXISTS idx_manual_inv_status ON manual_investments(status);

COMMENT ON TABLE manual_investments IS
'Inversiones manuales del usuario (no del bot). Precio + cantidad + fecha para cálculo PnL y FIFO fiscal.';

-- ── Tabla: cierres/ventas manuales ────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS manual_closings (
    id              SERIAL       PRIMARY KEY,
    investment_id   INT          REFERENCES manual_investments(id) ON DELETE CASCADE,
    symbol          VARCHAR(20)  NOT NULL,
    amount_sold     DECIMAL(18,8) NOT NULL,
    buy_price       DECIMAL(18,8) NOT NULL,  -- coste base FIFO
    sell_price      DECIMAL(18,8) NOT NULL,
    sell_date       DATE         NOT NULL DEFAULT CURRENT_DATE,
    pnl_usd         DECIMAL(12,4),           -- ganancia/pérdida realizada
    exchange        VARCHAR(50),
    notes           TEXT,
    created_at      TIMESTAMPTZ  DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_manual_closings_symbol ON manual_closings(symbol);
CREATE INDEX IF NOT EXISTS idx_manual_closings_date   ON manual_closings(sell_date);

COMMENT ON TABLE manual_closings IS
'Registro de ventas/cierres de inversiones manuales. Cada fila = ganancia realizada para Hacienda.';

-- ── Vista: resumen portafolio manual ──────────────────────────────────────────
CREATE OR REPLACE VIEW v_manual_portfolio AS
SELECT
    mi.symbol,
    mi.exchange,
    SUM(mi.amount)                          AS total_amount,
    AVG(mi.buy_price)                       AS avg_buy_price,
    SUM(mi.amount * mi.buy_price)           AS total_invested,
    MIN(mi.buy_date)                        AS first_buy,
    MAX(mi.buy_date)                        AS last_buy,
    COUNT(*)                                AS num_purchases
FROM manual_investments mi
WHERE mi.status = 'open'
GROUP BY mi.symbol, mi.exchange
ORDER BY total_invested DESC;

COMMENT ON VIEW v_manual_portfolio IS
'Portafolio manual agregado por activo/exchange con precio medio ponderado.';

-- ── Vista fiscal unificada ────────────────────────────────────────────────────
-- Combina trades del bot + ventas manuales para informe AEAT
CREATE OR REPLACE VIEW v_fiscal_operations AS
-- Trades cerrados del bot (compra + venta)
SELECT
    DATE(tj.entry_time)      AS fecha,
    tj.symbol,
    'BUY'                    AS tipo,
    'Bot'                    AS fuente,
    tj.position_size         AS cantidad,
    tj.entry_price           AS precio_usd,
    tj.position_size * tj.entry_price AS total_usd,
    tj.position_size * tj.entry_price AS coste_fifo,
    NULL::DOUBLE PRECISION   AS ganancia,
    'Binance'                AS exchange,
    tj.strategy              AS notas
FROM trades_journal tj
WHERE tj.exit_time IS NOT NULL
UNION ALL
SELECT
    DATE(tj.exit_time),
    tj.symbol,
    'SELL', 'Bot',
    tj.position_size,
    tj.exit_price,
    tj.position_size * tj.exit_price,
    tj.position_size * tj.entry_price,
    tj.pnl_usd,
    'Binance',
    tj.exit_reason
FROM trades_journal tj
WHERE tj.exit_time IS NOT NULL
UNION ALL
-- Compras manuales
SELECT
    mi.buy_date,
    mi.symbol,
    UPPER(mi.tx_type), 'Manual',
    mi.amount,
    mi.buy_price,
    mi.amount * mi.buy_price,
    mi.amount * mi.buy_price,
    NULL,
    mi.exchange,
    mi.notes
FROM manual_investments mi
UNION ALL
-- Ventas manuales
SELECT
    mc.sell_date,
    mc.symbol,
    'SELL', 'Manual',
    mc.amount_sold,
    mc.sell_price,
    mc.amount_sold * mc.sell_price,
    mc.amount_sold * mc.buy_price,
    mc.pnl_usd,
    mc.exchange,
    mc.notes
FROM manual_closings mc
ORDER BY fecha ASC;

COMMENT ON VIEW v_fiscal_operations IS
'Vista fiscal unificada: bot + manual. Para exportación Koinly/CoinTracking.';

\echo '✅ Migration dashboard completada. Tablas: manual_investments, manual_closings.'
\echo '   Vistas: v_manual_portfolio, v_fiscal_operations.'
