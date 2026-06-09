-- 1. Dar permisos totales al usuario 'trading' sobre las tablas nuevas
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO trading;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO trading;

-- 2. Añadir las columnas avanzadas a trades_journal (no borra datos)
ALTER TABLE trades_journal ADD COLUMN IF NOT EXISTS trade_id VARCHAR(50);
ALTER TABLE trades_journal ADD COLUMN IF NOT EXISTS pnl_usd FLOAT DEFAULT 0.0;
ALTER TABLE trades_journal ADD COLUMN IF NOT EXISTS strategy VARCHAR(100) DEFAULT 'General';
ALTER TABLE trades_journal ADD COLUMN IF NOT EXISTS timeframe VARCHAR(20);
ALTER TABLE trades_journal ADD COLUMN IF NOT EXISTS setup_quality INT;
ALTER TABLE trades_journal ADD COLUMN IF NOT EXISTS stop_loss FLOAT;
ALTER TABLE trades_journal ADD COLUMN IF NOT EXISTS take_profit_1 FLOAT;
ALTER TABLE trades_journal ADD COLUMN IF NOT EXISTS position_size FLOAT;
ALTER TABLE trades_journal ADD COLUMN IF NOT EXISTS risk_amount FLOAT;
ALTER TABLE trades_journal ADD COLUMN IF NOT EXISTS r_multiple FLOAT;
ALTER TABLE trades_journal ADD COLUMN IF NOT EXISTS duration_hours FLOAT;
ALTER TABLE trades_journal ADD COLUMN IF NOT EXISTS entry_reason TEXT;
ALTER TABLE trades_journal ADD COLUMN IF NOT EXISTS exit_reason TEXT;
ALTER TABLE trades_journal ADD COLUMN IF NOT EXISTS market_regime VARCHAR(100);
ALTER TABLE trades_journal ADD COLUMN IF NOT EXISTS is_backtest BOOLEAN DEFAULT FALSE;

-- 3. Arreglar la anomalía del Drawdown (Capital actual mayor que capital máximo)
UPDATE portfolio_state 
SET peak_capital = GREATEST(peak_capital, current_capital);

-- Aseguramos que todo parta en 1000 limpio
UPDATE portfolio_state 
SET current_capital = 1000.0, 
    peak_capital = 1000.0, 
    daily_start = 1000.0, 
    weekly_start = 1000.0, 
    monthly_start = 1000.0 
WHERE id = 1 AND current_capital < 1000.0;
