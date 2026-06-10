"""
paper_portfolio.py — Portfolio virtual para Paper Trading.
==========================================================
Simula la ejecución de órdenes sin tocar Binance.
Persiste el estado en PostgreSQL para sobrevivir a reinicios del servidor.

En paper_mode=True, live_engine.py usa esta clase en lugar de ccxt directamente.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

log = structlog.get_logger(__name__)

# ── Constantes de costes ───────────────────────────────────────────────────────
COMMISSION_RATE  = 0.001   # 0.1% Binance maker/taker
SLIPPAGE_RATE    = 0.001   # 0.1% slippage estimado
ATR_TRAILING_MULT = 1.5    # Trailing stop = 1.5 ATR bajo el máximo post-TP1


class PaperPortfolio:
    """
    Cartera virtual de paper trading con persistencia en PostgreSQL.

    El estado completo (capital, posiciones, journal) se guarda en la BD
    para que el motor pueda continuar tras un reinicio sin perder datos.
    """

    def __init__(self, initial_capital: float, db_url: str) -> None:
        """
        Args:
            initial_capital: capital inicial en USD (solo se usa si no hay
                             estado previo en la BD).
            db_url:          URL de conexión a PostgreSQL (asyncpg).
                             Ejemplo: postgresql+asyncpg://user:pass@host/db
        """
        self._initial_capital = initial_capital
        self._capital: float = initial_capital
        self._db_url = db_url
        self._engine = create_async_engine(db_url, pool_pre_ping=True)
        self._session_factory = sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )
        self._max_price: Dict[int, float] = {}   # position_id → precio máximo visto (longs)
        self._min_price: Dict[int, float] = {}   # position_id → precio mínimo visto (shorts)

    # ── Inicialización ─────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """
        Crea las tablas necesarias si no existen y carga el estado previo.
        Debe llamarse una vez al arrancar el motor.
        """
        await self._create_tables()
        await self._load_state()
        log.info(
            "paper_portfolio_initialized",
            capital=f"${self._capital:.2f}",
        )

    async def _create_tables(self) -> None:
        """Aplica el schema SQL si las tablas no existen."""
        ddl = """
        CREATE TABLE IF NOT EXISTS positions (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            strategy VARCHAR(30) NOT NULL,
            direction VARCHAR(10) DEFAULT 'long',
            entry_time TIMESTAMPTZ NOT NULL,
            entry_price DECIMAL(18,8) NOT NULL,
            stop_loss DECIMAL(18,8) NOT NULL,
            tp1 DECIMAL(18,8) NOT NULL,
            tp2 DECIMAL(18,8) NOT NULL,
            units DECIMAL(18,8) NOT NULL,
            risk_amount DECIMAL(10,4) NOT NULL,
            tp1_hit BOOLEAN DEFAULT FALSE,
            remaining_units DECIMAL(18,8) NOT NULL,
            binance_order_id VARCHAR(50),
            ml_proba DECIMAL(5,4),
            atr_entry DECIMAL(18,8) DEFAULT 0,  -- ATR real en el momento de apertura
            regime VARCHAR(30),
            status VARCHAR(20) DEFAULT 'open'
        );

        CREATE TABLE IF NOT EXISTS trades_journal (
            id SERIAL PRIMARY KEY,
            trade_id INTEGER,
            symbol VARCHAR(20) NOT NULL,
            strategy VARCHAR(50) NOT NULL,
            timeframe VARCHAR(10),
            direction VARCHAR(10) DEFAULT 'long',
            entry_time TIMESTAMPTZ NOT NULL,
            exit_time TIMESTAMPTZ,
            entry_price DECIMAL(18,8),
            exit_price DECIMAL(18,8),
            stop_loss DECIMAL(18,8),
            tp1 DECIMAL(18,8),
            tp2 DECIMAL(18,8),
            tp2 DECIMAL(18,8),
            units DECIMAL(18,8),
            pnl DECIMAL(10,4),
            pnl_pct DECIMAL(8,6),
            r_multiple DECIMAL(6,3),
            exit_reason VARCHAR(30),
            ml_proba DECIMAL(5,4),
            regime VARCHAR(30),
            commission_paid DECIMAL(10,6),
            setup_quality SMALLINT,
            duration_hours DECIMAL(8,2),
            is_backtest BOOLEAN DEFAULT FALSE,
            risk_amount DECIMAL(10,4),
            entry_reason TEXT,
            observations TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_tj_entry_time ON trades_journal(entry_time DESC);
        CREATE INDEX IF NOT EXISTS idx_tj_symbol     ON trades_journal(symbol);
        CREATE INDEX IF NOT EXISTS idx_tj_strategy   ON trades_journal(strategy);

        CREATE TABLE IF NOT EXISTS portfolio_state (
            id INTEGER PRIMARY KEY DEFAULT 1,
            current_capital DECIMAL(12,4) NOT NULL,
            daily_start DECIMAL(12,4) NOT NULL,
            weekly_start DECIMAL(12,4) NOT NULL,
            monthly_start DECIMAL(12,4) NOT NULL,
            peak_capital DECIMAL(12,4) NOT NULL,
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS system_heartbeat (
            id INTEGER PRIMARY KEY DEFAULT 1,
            last_ping TIMESTAMPTZ DEFAULT NOW(),
            engine_version VARCHAR(30),
            paper_mode BOOLEAN,
            active_positions INTEGER DEFAULT 0,
            pnl_today DECIMAL(10,4) DEFAULT 0,
            regimes_json TEXT,
            cycles_json TEXT
        );

        CREATE TABLE IF NOT EXISTS cycle_state (
            symbol VARCHAR(20) PRIMARY KEY,
            phase VARCHAR(30) NOT NULL,
            conviction DECIMAL(5,3) DEFAULT 0,
            risk_multiplier DECIMAL(4,2) DEFAULT 1.0,
            rsi_daily DECIMAL(6,2),
            rsi_weekly DECIMAL(6,2),
            pct_from_ath DECIMAL(8,4),
            active_strategies TEXT,
            last_dca_week INTEGER DEFAULT 0,
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS ml_retrain_log (
            id SERIAL PRIMARY KEY,
            retrain_date TIMESTAMPTZ DEFAULT NOW(),
            roc_auc DECIMAL(6,4),
            cv_f1 DECIMAL(6,4),
            n_samples INTEGER,
            top_feature VARCHAR(50),
            threshold DECIMAL(5,3),
            feature_importance_json TEXT,
            notes TEXT
        );

        CREATE TABLE IF NOT EXISTS ohlcv (
            symbol VARCHAR(20) NOT NULL,
            timeframe VARCHAR(10) NOT NULL,
            timestamp BIGINT NOT NULL,
            open DECIMAL(18,8), high DECIMAL(18,8),
            low DECIMAL(18,8), close DECIMAL(18,8),
            volume DECIMAL(20,4),
            PRIMARY KEY (symbol, timeframe, timestamp)
        );
        CREATE INDEX IF NOT EXISTS idx_ohlcv_ts
            ON ohlcv(symbol, timeframe, timestamp);
        """
        async with self._engine.begin() as conn:
            for stmt in ddl.split(";"):
                stmt = stmt.strip()
                if stmt:
                    await conn.execute(text(stmt))

        # ── ALTER TABLE para añadir columnas si la tabla ya existía (migraciones) ──
        migrations = [
            "ALTER TABLE trades_journal ADD COLUMN IF NOT EXISTS trade_id INTEGER",
            "ALTER TABLE trades_journal ADD COLUMN IF NOT EXISTS timeframe VARCHAR(10)",
            "ALTER TABLE trades_journal ADD COLUMN IF NOT EXISTS direction VARCHAR(10) DEFAULT 'long'",
            "ALTER TABLE trades_journal ADD COLUMN IF NOT EXISTS tp1 DECIMAL(18,8)",
            "ALTER TABLE trades_journal ADD COLUMN IF NOT EXISTS tp2 DECIMAL(18,8)",
            "ALTER TABLE trades_journal ADD COLUMN IF NOT EXISTS units DECIMAL(18,8)",
            "ALTER TABLE trades_journal ADD COLUMN IF NOT EXISTS pnl DECIMAL(10,4)",
            "ALTER TABLE trades_journal ADD COLUMN IF NOT EXISTS pnl_pct DECIMAL(8,6)",
            "ALTER TABLE trades_journal ADD COLUMN IF NOT EXISTS r_multiple DECIMAL(6,3)",
            "ALTER TABLE trades_journal ADD COLUMN IF NOT EXISTS commission_paid DECIMAL(10,6)",
            "ALTER TABLE trades_journal ADD COLUMN IF NOT EXISTS regime VARCHAR(30)",
            "ALTER TABLE trades_journal ADD COLUMN IF NOT EXISTS setup_quality SMALLINT",
            "ALTER TABLE trades_journal ADD COLUMN IF NOT EXISTS duration_hours DECIMAL(8,2)",
            "ALTER TABLE trades_journal ADD COLUMN IF NOT EXISTS is_backtest BOOLEAN DEFAULT FALSE",
            "ALTER TABLE trades_journal ADD COLUMN IF NOT EXISTS risk_amount DECIMAL(10,4)",
            "ALTER TABLE trades_journal ADD COLUMN IF NOT EXISTS entry_reason TEXT",
            "ALTER TABLE trades_journal ADD COLUMN IF NOT EXISTS observations TEXT",
            "ALTER TABLE system_heartbeat ADD COLUMN IF NOT EXISTS active_positions INTEGER DEFAULT 0",
            "ALTER TABLE system_heartbeat ADD COLUMN IF NOT EXISTS pnl_today DECIMAL(10,4) DEFAULT 0",
            "ALTER TABLE system_heartbeat ADD COLUMN IF NOT EXISTS regimes_json TEXT",
            "ALTER TABLE system_heartbeat ADD COLUMN IF NOT EXISTS cycles_json TEXT",
            "ALTER TABLE trades_journal ADD COLUMN IF NOT EXISTS hurst_at_entry DECIMAL(6,4)",
            "ALTER TABLE trades_journal ADD COLUMN IF NOT EXISTS zscore_at_entry DECIMAL(6,4)",
            "ALTER TABLE trades_journal ADD COLUMN IF NOT EXISTS t_stat_at_entry DECIMAL(6,4)",
            "ALTER TABLE trades_journal ADD COLUMN IF NOT EXISTS confluence_score INTEGER",
            "ALTER TABLE positions ADD COLUMN IF NOT EXISTS hurst_at_entry DECIMAL(6,4)",
            "ALTER TABLE positions ADD COLUMN IF NOT EXISTS zscore_at_entry DECIMAL(6,4)",
            "ALTER TABLE positions ADD COLUMN IF NOT EXISTS t_stat_at_entry DECIMAL(6,4)",
            "ALTER TABLE positions ADD COLUMN IF NOT EXISTS confluence_score INTEGER",
            "ALTER TABLE cycle_state ADD COLUMN IF NOT EXISTS last_dca_week INTEGER DEFAULT 0",
        ]
        async with self._engine.begin() as conn:
            for stmt in migrations:
                try:
                    await conn.execute(text(stmt))
                except Exception:
                    pass  # Ignora si ya existe

        log.debug("db_tables_ensured")

    async def _load_state(self) -> None:
        """Carga el capital actual desde portfolio_state."""
        async with self._session_factory() as session:
            result = await session.execute(
                text("SELECT current_capital FROM portfolio_state WHERE id = 1")
            )
            row = result.fetchone()
            if row:
                self._capital = float(row[0])
                log.info("state_loaded_from_db", capital=f"${self._capital:.2f}")
            else:
                # Primera vez: inicializar
                await session.execute(
                    text("""
                        INSERT INTO portfolio_state
                            (id, current_capital, daily_start, weekly_start,
                             monthly_start, peak_capital)
                        VALUES (1, :cap, :cap, :cap, :cap, :cap)
                        ON CONFLICT (id) DO NOTHING
                    """),
                    {"cap": self._initial_capital},
                )
                await session.commit()
                self._capital = self._initial_capital
                log.info("state_initialized", capital=f"${self._capital:.2f}")

    # ── API pública ────────────────────────────────────────────────────────────

    async def open_position(
        self,
        symbol: str,
        strategy: str,
        entry_price: float,
        stop_loss: float,
        tp1: float,
        tp2: float,
        units: float,
        ml_proba: float,
        direction: str = "long",
        regime: Optional[str] = None,
        risk_amount: Optional[float] = None,
        notional_usd: Optional[float] = None,
        atr: float = 0.0,    # ATR real en el momento de apertura (para trailing stop)
        hurst_at_entry: Optional[float] = None,
        zscore_at_entry: Optional[float] = None,
        t_stat_at_entry: Optional[float] = None,
        confluence_score: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Simula apertura de una posición.
        Aplica comisión (0.1%) y slippage (0.1%) al precio de entrada.

        Returns:
            dict con todos los campos de la posición creada.
        """
        # Precio de entrada real: slippage + comisión
        fill_price = entry_price * (1 + SLIPPAGE_RATE + COMMISSION_RATE) \
            if direction == "long" \
            else entry_price * (1 - SLIPPAGE_RATE - COMMISSION_RATE)

        commission = units * fill_price * COMMISSION_RATE
        
        # risk_amount: preferir el valor calculado por compute_position_size en el motor;
        # si no se pasa (None), calcularlo desde la distancia al SL.
        _risk_computed = abs(fill_price - stop_loss) * units
        _risk_to_store = risk_amount if risk_amount is not None else _risk_computed

        now = datetime.now(tz=timezone.utc)

        async with self._session_factory() as session:
            result = await session.execute(
                text("""
                    INSERT INTO positions
                        (symbol, strategy, direction, entry_time, entry_price,
                         stop_loss, tp1, tp2, units, risk_amount, tp1_hit,
                         remaining_units, ml_proba, atr_entry, regime, status,
                         hurst_at_entry, zscore_at_entry, t_stat_at_entry, confluence_score)
                    VALUES
                        (:symbol, :strategy, :direction, :entry_time, :entry_price,
                         :stop_loss, :tp1, :tp2, :units, :risk_amount, FALSE,
                         :units, :ml_proba, :atr_entry, :regime, 'open',
                         :hurst, :zscore, :tstat, :confluence)
                    RETURNING id
                """),
                {
                    "symbol":      symbol,
                    "strategy":    strategy,
                    "direction":   direction,
                    "entry_time":  now,
                    "entry_price": fill_price,
                    "stop_loss":   stop_loss,
                    "tp1":         tp1,
                    "tp2":         tp2,
                    "units":       units,
                    "risk_amount": _risk_to_store,
                    "ml_proba":    ml_proba,
                    "atr_entry":   atr if atr > 0 else fill_price * 0.015,
                    "regime":      regime,
                    "hurst":       hurst_at_entry,
                    "zscore":      zscore_at_entry,
                    "tstat":       t_stat_at_entry,
                    "confluence":  confluence_score,
                },
            )
            position_id = result.fetchone()[0]
            await session.commit()

        self._max_price[position_id] = fill_price
        self._min_price[position_id] = fill_price

        trade = {
            "id":           position_id,
            "symbol":       symbol,
            "strategy":     strategy,
            "direction":    direction,
            "entry_time":   now.isoformat(),
            "entry_price":  fill_price,
            "stop_loss":    stop_loss,
            "tp1":          tp1,
            "tp2":          tp2,
            "units":        units,
            "remaining_units": units,
            "risk_amount":  risk_amount,
            "ml_proba":     ml_proba,
            "commission":   commission,
            "status":       "open",
        }
        log.info(
            "position_opened",
            id=position_id,
            symbol=symbol,
            direction=direction,
            entry=f"${fill_price:.2f}",
            sl=f"${stop_loss:.2f}",
            tp1=f"${tp1:.2f}",
            units=f"{units:.6f}",
            ml=f"{ml_proba:.1%}",
        )
        return trade

    async def update_positions(
        self, current_prices: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Revisa todas las posiciones abiertas contra los precios actuales.
        Simula high/low usando ±0.1% del precio de cierre para detectar
        si SL o TP fueron tocados en el período.

        Args:
            current_prices: {symbol: last_price}

        Returns:
            Lista de trades cerrados en este ciclo.
        """
        closed: List[Dict[str, Any]] = []

        async with self._session_factory() as session:
            result = await session.execute(
                text("SELECT * FROM positions WHERE status = 'open'")
            )
            positions = [dict(row._mapping) for row in result.fetchall()]

        for pos in positions:
            symbol = pos["symbol"]
            if symbol not in current_prices:
                continue

            price  = current_prices[symbol]
            pos_id = pos["id"]
            low    = price * (1 - SLIPPAGE_RATE)    # simulamos low del período
            high   = price * (1 + SLIPPAGE_RATE)

            sl   = float(pos["stop_loss"])
            tp1  = float(pos["tp1"])
            tp2  = float(pos["tp2"])
            tp1_hit = bool(pos["tp1_hit"])
            remaining = float(pos["remaining_units"])

            # Actualizar precio máximo para trailing stop
            self._max_price[pos_id] = max(self._max_price.get(pos_id, price), price)

            direction = pos.get("direction", "long")

            if direction == "long":
                # ── SL hit ────────────────────────────────────────────────────
                if low <= sl:
                    closed_trade = await self.close_position(pos_id, sl, "stop_loss")
                    closed.append(closed_trade)
                    continue

                # ── TP1 hit (parcial: cerrar 50%) ─────────────────────────────
                if not tp1_hit and high >= tp1:
                    units_to_close = remaining * 0.5
                    await self._partial_close(session, pos_id, tp1, units_to_close)
                    # Mover SL a breakeven
                    entry = float(pos["entry_price"])
                    await session.execute(
                        text("""
                            UPDATE positions
                            SET tp1_hit = TRUE,
                                stop_loss = :be,
                                remaining_units = :rem
                            WHERE id = :pid
                        """),
                        {"be": entry, "rem": remaining - units_to_close, "pid": pos_id},
                    )
                    await session.commit()
                    log.info("tp1_hit_partial_close", id=pos_id, symbol=symbol)
                    continue

                # ── TP2 hit (resto de la posición) ────────────────────────────
                if tp1_hit and high >= tp2:
                    closed_trade = await self.close_position(pos_id, tp2, "tp2")
                    closed.append(closed_trade)
                    continue

                # ── Trailing stop post-TP1 ─────────────────────────────────
                if tp1_hit:
                    peak = self._max_price.get(pos_id, price)
                    # Usar ATR real guardado en el momento de apertura.
                    # Fallback: 1.5% del precio pico si no está disponible.
                    atr_saved = float(pos.get("atr_entry", 0) or 0)
                    atr_for_trail = atr_saved if atr_saved > 0 else peak * 0.015
                    trailing_sl = peak - ATR_TRAILING_MULT * atr_for_trail
                    if trailing_sl > sl and low <= trailing_sl:
                        closed_trade = await self.close_position(
                            pos_id, trailing_sl, "trailing_stop"
                        )
                        closed.append(closed_trade)
            elif direction == "short":
                # ── SL hit ────────────────────────────────────────────────────
                if high >= sl:
                    closed_trade = await self.close_position(pos_id, sl, "stop_loss")
                    closed.append(closed_trade)
                    continue

                # ── TP1 hit (parcial: cerrar 50%) ─────────────────────────────
                if not tp1_hit and low <= tp1:
                    units_to_close = remaining * 0.5
                    await self._partial_close(session, pos_id, tp1, units_to_close)
                    # Mover SL a breakeven
                    entry = float(pos["entry_price"])
                    await session.execute(
                        text("""
                            UPDATE positions
                            SET tp1_hit = TRUE,
                                stop_loss = :be,
                                remaining_units = :rem
                            WHERE id = :pid
                        """),
                        {"be": entry, "rem": remaining - units_to_close, "pid": pos_id},
                    )
                    await session.commit()
                    log.info("tp1_hit_partial_close_short", id=pos_id, symbol=symbol)
                    continue

                # ── TP2 hit (resto de la posición) ────────────────────────────
                if tp1_hit and low <= tp2:
                    closed_trade = await self.close_position(pos_id, tp2, "tp2")
                    closed.append(closed_trade)
                    continue

                # ── Trailing stop post-TP1 ─────────────────────────────────
                if tp1_hit:
                    # Actualizamos precio mínimo
                    self._min_price[pos_id] = min(self._min_price.get(pos_id, price), price)
                    trough = self._min_price[pos_id]
                    atr_saved = float(pos.get("atr_entry", 0) or 0)
                    atr_for_trail = atr_saved if atr_saved > 0 else trough * 0.015
                    trailing_sl = trough + ATR_TRAILING_MULT * atr_for_trail
                    if trailing_sl < sl and high >= trailing_sl:
                        closed_trade = await self.close_position(
                            pos_id, trailing_sl, "trailing_stop"
                        )
                        closed.append(closed_trade)

        return closed

    async def close_position(
        self, position_id: int, exit_price: float, reason: str
    ) -> Dict[str, Any]:
        """
        Cierra una posición, calcula PnL y mueve al trades_journal.

        Args:
            position_id: ID de la posición en tabla positions.
            exit_price:  precio de salida (simulado o de mercado).
            reason:      razón de cierre (stop_loss, tp1, tp2, trailing_stop, kill).

        Returns:
            dict con resumen completo del trade cerrado.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                text("SELECT * FROM positions WHERE id = :pid"),
                {"pid": position_id},
            )
            pos = dict(result.fetchone()._mapping)

        entry_price = float(pos["entry_price"])
        units       = float(pos["remaining_units"])
        commission  = units * exit_price * COMMISSION_RATE
        now         = datetime.now(tz=timezone.utc)

        # PnL bruto
        direction = pos.get("direction", "long")
        if direction == "long":
            pnl = (exit_price - entry_price) * units - commission
        else:
            pnl = (entry_price - exit_price) * units - commission

        pnl_pct   = pnl / (entry_price * units) if entry_price * units > 0 else 0
        risk_amt   = float(pos["risk_amount"])
        r_multiple = pnl / risk_amt if risk_amt > 0 else 0

        async with self._session_factory() as session:
            # Mover a journal
            await session.execute(
                text("""
                    INSERT INTO trades_journal
                        (trade_id, symbol, strategy, timeframe, direction,
                         entry_time, exit_time, entry_price, exit_price,
                         stop_loss, tp1, tp2, units, pnl, pnl_pct, r_multiple,
                         commission_paid, regime, setup_quality, duration_hours,
                         is_backtest, risk_amount, ml_proba, entry_reason, exit_reason, observations,
                         hurst_at_entry, zscore_at_entry, t_stat_at_entry, confluence_score)
                    VALUES
                        (:tid, :sym, :strat, '1h', :dir,
                         :et, :xt, :ep, :xp,
                         :sl, :tp1, :tp2, :units, :pnl, :pnl_pct, :r_mult,
                         :comm, :regime, 3, :dur,
                         FALSE, :risk, :ml_proba, 'auto', :reason, :obs,
                         :hurst, :zscore, :tstat, :confluence)
                """),
                {
                    "tid":          pos["id"],
                    "sym":          pos["symbol"],
                    "strat":        pos["strategy"],
                    "dir":          direction,
                    "et":           pos["entry_time"],
                    "xt":           now,
                    "ep":           entry_price,
                    "xp":           exit_price,
                    "sl":           pos["stop_loss"],
                    "tp1":          pos["tp1"],
                    "tp2":          pos["tp2"],
                    "units":        units,
                    "pnl":          pnl,
                    "pnl_pct":      pnl_pct,
                    "r_mult":       r_multiple,
                    "comm":         commission,
                    "regime":       pos.get("regime"),
                    "dur":          0,
                    "risk":         risk_amt,
                    "ml_proba":     pos.get("ml_proba"),
                    "reason":       reason,
                    "obs":          pos.get("observations", ""),
                    "hurst":        pos.get("hurst_at_entry"),
                    "zscore":       pos.get("zscore_at_entry"),
                    "tstat":        pos.get("t_stat_at_entry"),
                    "confluence":   pos.get("confluence_score"),
                },
            )
            # Marcar posición como cerrada
            await session.execute(
                text("UPDATE positions SET status = 'closed' WHERE id = :pid"),
                {"pid": position_id},
            )
            # Actualizar capital
            self._capital += pnl
            await session.execute(
                text("""
                    UPDATE portfolio_state
                    SET current_capital = :cap,
                        peak_capital = GREATEST(peak_capital, :cap),
                        updated_at = NOW()
                    WHERE id = 1
                """),
                {"cap": self._capital},
            )
            await session.commit()

        self._max_price.pop(position_id, None)

        closed = {
            "id":          position_id,
            "symbol":      pos["symbol"],
            "strategy":    pos["strategy"],
            "direction":   direction,
            "entry_price": entry_price,
            "exit_price":  exit_price,
            "units":       units,
            "pnl":         pnl,
            "pnl_pct":     pnl_pct,
            "r_multiple":  r_multiple,
            "exit_reason": reason,
            "ml_proba":    pos.get("ml_proba"),
        }
        log.info(
            "position_closed",
            id=position_id,
            symbol=pos["symbol"],
            reason=reason,
            pnl=f"{pnl:+.2f}",
            r=f"{r_multiple:+.2f}R",
        )
        return closed

    async def _partial_close(
        self, session: AsyncSession, position_id: int, price: float, units: float
    ) -> None:
        """Registra un cierre parcial de posición en el journal."""
        result = await session.execute(
            text("SELECT * FROM positions WHERE id = :pid"), {"pid": position_id}
        )
        pos = dict(result.fetchone()._mapping)
        commission = units * price * COMMISSION_RATE
        entry_price = float(pos["entry_price"])
        pnl = (price - entry_price) * units - commission

        await session.execute(
            text("""
                INSERT INTO trades_journal
                    (symbol, strategy, entry_time, exit_time, entry_price,
                     exit_price, units, pnl, exit_reason, ml_proba, commission_paid,
                     hurst_at_entry, zscore_at_entry, t_stat_at_entry, confluence_score)
                VALUES
                    (:symbol, :strategy, :entry_time, NOW(), :entry_price,
                     :exit_price, :units, :pnl, 'tp1_partial', :ml_proba, :commission,
                     :hurst, :zscore, :tstat, :confluence)
            """),
            {
                "symbol":      pos["symbol"],
                "strategy":    pos["strategy"],
                "entry_time":  pos["entry_time"],
                "entry_price": entry_price,
                "exit_price":  price,
                "units":       units,
                "pnl":         pnl,
                "ml_proba":    pos.get("ml_proba"),
                "commission":  commission,
                "hurst":       pos.get("hurst_at_entry"),
                "zscore":      pos.get("zscore_at_entry"),
                "tstat":       pos.get("t_stat_at_entry"),
                "confluence":  pos.get("confluence_score"),
            },
        )
        self._capital += pnl

    # ── Consultas ─────────────────────────────────────────────────────────────

    async def get_current_capital(self) -> float:
        """Devuelve el capital actual en memoria (sincronizado con la BD)."""
        return self._capital

    async def get_open_positions(self) -> List[Dict[str, Any]]:
        """Devuelve todas las posiciones abiertas."""
        async with self._session_factory() as session:
            result = await session.execute(
                text("SELECT * FROM positions WHERE status = 'open' ORDER BY entry_time")
            )
            return [dict(row._mapping) for row in result.fetchall()]

    async def get_daily_stats(self) -> Dict[str, Any]:
        """
        Estadísticas del día actual.

        Returns:
            dict: trades_today, pnl_today, wins_today, losses_today.
        """
        today = date.today()
        async with self._session_factory() as session:
            result = await session.execute(
                text("""
                    SELECT
                        COUNT(*)                          AS trades,
                        COALESCE(SUM(pnl), 0)            AS pnl,
                        COALESCE(SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END), 0) AS wins,
                        COALESCE(SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END), 0) AS losses
                    FROM trades_journal
                    WHERE exit_time::date = :today
                """),
                {"today": today},
            )
            row = result.fetchone()

        return {
            "trades_today": int(row[0]) if row else 0,
            "pnl_today":    float(row[1]) if row else 0.0,
            "wins_today":   int(row[2]) if row else 0,
            "losses_today": int(row[3]) if row else 0,
        }

    async def emergency_close_all(
        self, current_prices: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Kill switch: cierra todas las posiciones abiertas a precio de mercado.
        Usado por el comando /kill de Telegram.

        Args:
            current_prices: {symbol: last_price}

        Returns:
            Lista de trades cerrados.
        """
        log.warning("emergency_close_all_triggered")
        positions = await self.get_open_positions()
        closed: List[Dict[str, Any]] = []

        for pos in positions:
            symbol = pos["symbol"]
            price  = current_prices.get(symbol, float(pos["entry_price"]))
            closed_trade = await self.close_position(pos["id"], price, "kill_switch")
            closed.append(closed_trade)

        log.warning("emergency_close_all_done", n_closed=len(closed))
        return closed
