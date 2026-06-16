"""
Data Storage — SQLite / PostgreSQL Interface
=============================================
Gestiona la persistencia de datos OHLCV con soporte dual:
  - SQLite   (Fase 1: desarrollo local, pruebas)
  - PostgreSQL (Fase 2+: producción en ZimaBlade / Proxmox)

El dialect se detecta automáticamente desde DATABASE_URL en .env.
Solo cambiar la variable de entorno para migrar entre motores.

Cambios Fase 2:
  - Upsert nativo PostgreSQL: INSERT … ON CONFLICT DO UPDATE
  - Pool de conexiones para concurrencia (asyncpg preparado)
  - SQLite PRAGMAs solo aplicados cuando dialect == sqlite
  - get_database_size_mb() soporta ambos dialectos
  - TradeRecord y DrawdownRecord eliminan sqlite_autoincrement en PG
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from typing import Generator, List, Optional

import pandas as pd
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config.logging_config import get_logger
from config.settings import env_settings

logger = get_logger(__name__)


# ── ORM Models ─────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


class OHLCVRecord(Base):
    """Una vela OHLCV individual."""

    __tablename__ = "ohlcv"
    __table_args__ = (
        UniqueConstraint("symbol", "timeframe", "timestamp", name="uq_ohlcv_key"),
    )

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    symbol: str = Column(String(20), nullable=False, index=True)
    timeframe: str = Column(String(5), nullable=False, index=True)
    timestamp: datetime = Column(DateTime(timezone=True), nullable=False, index=True)
    open: float = Column(Float, nullable=False)
    high: float = Column(Float, nullable=False)
    low: float = Column(Float, nullable=False)
    close: float = Column(Float, nullable=False)
    volume: float = Column(Float, nullable=False)


class TradeRecord(Base):
    """Registro completo de un trade para el journal."""

    __tablename__ = "trades"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    # Identificación
    trade_id: str = Column(String(36), unique=True, nullable=False)  # UUID
    strategy: str = Column(String(30), nullable=False)
    symbol: str = Column(String(20), nullable=False, index=True)
    timeframe: str = Column(String(5), nullable=False)
    direction: str = Column(String(5), nullable=False)  # LONG / SHORT
    setup_quality: str = Column(String(3), nullable=True)

    # Precios
    entry_price: float = Column(Float, nullable=False)
    stop_loss: float = Column(Float, nullable=False)
    take_profit_1: float = Column(Float, nullable=False)
    take_profit_2: float = Column(Float, nullable=True)
    exit_price: float = Column(Float, nullable=True)

    # Posición
    position_size: float = Column(Float, nullable=False)
    risk_amount: float = Column(Float, nullable=False)

    # Resultado
    pnl_usd: float = Column(Float, nullable=True)
    pnl_pct: float = Column(Float, nullable=True)
    r_multiple: float = Column(Float, nullable=True)  # R obtenido

    # Timing
    entry_time: datetime = Column(DateTime(timezone=True), nullable=True)
    exit_time: datetime = Column(DateTime(timezone=True), nullable=True)
    duration_hours: float = Column(Float, nullable=True)

    # Razones (journal)
    entry_reason: str = Column(Text, nullable=True)
    exit_reason: str = Column(Text, nullable=True)
    observations: str = Column(Text, nullable=True)

    # Metadata
    created_at: datetime = Column(DateTime(timezone=True), default=datetime.utcnow)
    is_backtest: bool = Column(Integer, default=1)  # 1=backtest, 0=live

    # Regime at time of trade
    market_regime: str = Column(String(20), nullable=True)


class DrawdownRecord(Base):
    """Registro de drawdowns para el circuit breaker."""

    __tablename__ = "drawdowns"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    date: datetime = Column(DateTime(timezone=True), nullable=False, index=True)
    period: str = Column(String(10), nullable=False)  # daily / weekly / monthly
    drawdown_pct: float = Column(Float, nullable=False)
    capital_at_peak: float = Column(Float, nullable=False)
    capital_current: float = Column(Float, nullable=False)
    circuit_breaker_triggered: bool = Column(Integer, default=0)


# ── Storage Class ──────────────────────────────────────────────────────────────

class OHLCVStorage:
    """
    Interface para persistir y consultar datos OHLCV.
    Compatible con SQLite (Fase 1) y PostgreSQL (Fase 2+).

    Upsert strategy:
      - SQLite:     INSERT OR REPLACE (vía session.merge)
      - PostgreSQL: INSERT … ON CONFLICT (symbol, timeframe, timestamp)
                    DO UPDATE SET open=…, high=…, low=…, close=…, volume=…
    """

    def __init__(self, database_url: Optional[str] = None):
        url = database_url or env_settings.database_url
        self._dialect = self._detect_dialect(url)

        connect_args = {}
        engine_kwargs: dict = {"echo": False}

        if self._dialect == "sqlite":
            connect_args = {
                "check_same_thread": False,
                "timeout": 30,
            }
        elif self._dialect == "postgresql":
            # Pool de conexiones para concurrencia en producción
            engine_kwargs.update({
                "pool_size": 5,
                "max_overflow": 10,
                "pool_pre_ping": True,    # Detectar conexiones muertas
                "pool_recycle": 3600,     # Reciclar cada hora
            })

        self.engine = create_engine(
            url,
            connect_args=connect_args,
            **engine_kwargs,
        )

        self._session_factory = sessionmaker(
            bind=self.engine,
            expire_on_commit=False,
        )

        # Crear tablas si no existen (SQLAlchemy DDL)
        Base.metadata.create_all(self.engine)

        # SQLite-only optimizations
        if self._dialect == "sqlite":
            self._apply_sqlite_optimizations()

        logger.info(
            "storage_initialized",
            dialect=self._dialect,
            database=url.split("///")[-1] if "///" in url else url.split("@")[-1],
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _detect_dialect(url: str) -> str:
        """Detecta el dialect de la BD a partir de la URL."""
        url_lower = url.lower()
        if url_lower.startswith("postgresql") or url_lower.startswith("postgres"):
            return "postgresql"
        return "sqlite"

    def _apply_sqlite_optimizations(self) -> None:
        """PRAGMAs de SQLite optimizados para HDD mecánico del ZimaBlade."""
        optimizations = [
            "PRAGMA journal_mode=WAL",          # Write-Ahead Logging (mejor concurrencia)
            "PRAGMA synchronous=NORMAL",         # Balance entre velocidad y seguridad
            "PRAGMA cache_size=-64000",          # 64MB cache en RAM
            "PRAGMA temp_store=MEMORY",          # Tablas temporales en RAM
            "PRAGMA mmap_size=268435456",        # 256MB memory-mapped I/O
        ]
        with self.engine.connect() as conn:
            for pragma in optimizations:
                conn.execute(text(pragma))
            conn.commit()

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Context manager para sesiones de base de datos."""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # ── OHLCV Operations ──────────────────────────────────────────────────────

    def save_ohlcv(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        batch_size: int = 500,
    ) -> int:
        """
        Guarda datos OHLCV en la base de datos usando upsert.

        - PostgreSQL: INSERT … ON CONFLICT DO UPDATE (atómico, eficiente)
        - SQLite:     session.merge() equivalente a INSERT OR REPLACE

        Returns:
            Número de filas procesadas (upserted).
        """
        if df.empty:
            return 0

        total_processed = 0

        if self._dialect == "postgresql":
            total_processed = self._save_ohlcv_postgres(df, symbol, timeframe, batch_size)
        else:
            total_processed = self._save_ohlcv_sqlite(df, symbol, timeframe, batch_size)

        logger.info(
            "ohlcv_saved",
            symbol=symbol,
            timeframe=timeframe,
            rows=total_processed,
            dialect=self._dialect,
        )
        return total_processed

    def _save_ohlcv_postgres(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        batch_size: int,
    ) -> int:
        """
        Upsert nativo para PostgreSQL usando INSERT … ON CONFLICT DO UPDATE.
        Mucho más eficiente que session.merge() para inserciones masivas.
        """
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        total = 0
        table = OHLCVRecord.__table__

        for i in range(0, len(df), batch_size):
            batch = df.iloc[i : i + batch_size]
            rows = []
            for _, row in batch.iterrows():
                ts = row["timestamp"]
                if not hasattr(ts, "tzinfo") or ts.tzinfo is None:
                    ts = pd.Timestamp(ts, tz="UTC")

                rows.append({
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "timestamp": ts.to_pydatetime(),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                })

            stmt = pg_insert(table).values(rows)
            # ON CONFLICT → actualizar OHLCV (puede haber correcciones de exchange)
            stmt = stmt.on_conflict_do_update(
                constraint="uq_ohlcv_key",
                set_={
                    "open":   stmt.excluded.open,
                    "high":   stmt.excluded.high,
                    "low":    stmt.excluded.low,
                    "close":  stmt.excluded.close,
                    "volume": stmt.excluded.volume,
                },
            )

            with self.engine.begin() as conn:
                conn.execute(stmt)

            total += len(rows)

        return total

    def _save_ohlcv_sqlite(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        batch_size: int,
    ) -> int:
        """
        Upsert nativo para SQLite usando INSERT ... ON CONFLICT DO UPDATE.
        Requiere SQLAlchemy >= 1.4 y SQLite >= 3.24.
        """
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert

        total = 0
        table = OHLCVRecord.__table__

        for i in range(0, len(df), batch_size):
            batch = df.iloc[i : i + batch_size]
            rows = []
            for _, row in batch.iterrows():
                ts = row["timestamp"]
                if not hasattr(ts, "tzinfo") or ts.tzinfo is None:
                    ts = pd.Timestamp(ts, tz="UTC")

                rows.append({
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "timestamp": ts.to_pydatetime(),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                })

            stmt = sqlite_insert(table).values(rows)
            # ON CONFLICT DO UPDATE nativo de SQLite
            stmt = stmt.on_conflict_do_update(
                index_elements=["symbol", "timeframe", "timestamp"],
                set_={
                    "open":   stmt.excluded.open,
                    "high":   stmt.excluded.high,
                    "low":    stmt.excluded.low,
                    "close":  stmt.excluded.close,
                    "volume": stmt.excluded.volume,
                },
            )

            with self.engine.begin() as conn:
                conn.execute(stmt)

            total += len(rows)

        return total

    def load_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Carga datos OHLCV de la base de datos.

        Returns:
            DataFrame con los datos, o None si no hay datos para ese par/timeframe.
        """
        with self.get_session() as session:
            query = session.query(OHLCVRecord).filter(
                OHLCVRecord.symbol == symbol,
                OHLCVRecord.timeframe == timeframe,
            )

            if start_date:
                query = query.filter(OHLCVRecord.timestamp >= start_date)
            if end_date:
                query = query.filter(OHLCVRecord.timestamp <= end_date)

            query = query.order_by(OHLCVRecord.timestamp)
            records = query.all()

        if not records:
            return None

        data = {
            "timestamp": [r.timestamp for r in records],
            "open":      [r.open for r in records],
            "high":      [r.high for r in records],
            "low":       [r.low for r in records],
            "close":     [r.close for r in records],
            "volume":    [r.volume for r in records],
        }

        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df

    def get_available_data(self) -> pd.DataFrame:
        """Retorna un resumen de todos los datos disponibles en la BD."""
        with self.get_session() as session:
            result = session.execute(
                text("""
                    SELECT symbol, timeframe,
                           COUNT(*) as candles,
                           MIN(timestamp) as from_date,
                           MAX(timestamp) as to_date
                    FROM ohlcv
                    GROUP BY symbol, timeframe
                    ORDER BY symbol, timeframe
                """)
            )
            rows = result.fetchall()

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(
            rows,
            columns=["symbol", "timeframe", "candles", "from_date", "to_date"],
        )

    # ── Trade Journal Operations ───────────────────────────────────────────────

    def save_trade(self, trade_data: dict) -> str:
        """
        Guarda un trade en el journal.
        Retorna el trade_id.
        """
        import uuid
        trade_data.setdefault("trade_id", str(uuid.uuid4()))

        with self.get_session() as session:
            record = TradeRecord(**trade_data)
            session.add(record)

        logger.info(
            "trade_saved",
            trade_id=trade_data["trade_id"],
            symbol=trade_data.get("symbol"),
            direction=trade_data.get("direction"),
        )
        return trade_data["trade_id"]

    def load_trades(
        self,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
        is_backtest: Optional[bool] = None,
        since: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Carga trades del journal con filtros opcionales."""
        with self.get_session() as session:
            query = session.query(TradeRecord)

            if symbol:
                query = query.filter(TradeRecord.symbol == symbol)
            if strategy:
                query = query.filter(TradeRecord.strategy == strategy)
            if is_backtest is not None:
                query = query.filter(TradeRecord.is_backtest == int(is_backtest))
            if since:
                query = query.filter(TradeRecord.entry_time >= since)

            query = query.order_by(TradeRecord.entry_time)
            records = query.all()

        if not records:
            return pd.DataFrame()

        return pd.DataFrame(
            [
                {c.name: getattr(r, c.name) for c in TradeRecord.__table__.columns}
                for r in records
            ]
        )

    def get_database_size_mb(self) -> float:
        """
        Retorna el tamaño de la BD en MB.
        Soporta SQLite (stat del archivo) y PostgreSQL (pg_database_size).
        """
        if self._dialect == "postgresql":
            try:
                with self.engine.connect() as conn:
                    result = conn.execute(
                        text("SELECT pg_database_size(current_database()) / 1048576.0")
                    )
                    return float(result.scalar() or 0)
            except Exception:
                return 0.0
        else:
            db_path = env_settings.database_url.replace("sqlite:///", "")
            from pathlib import Path
            p = Path(db_path)
            if p.exists():
                return p.stat().st_size / (1024 * 1024)
            return 0.0

    @property
    def dialect(self) -> str:
        """Retorna el dialect de la BD ('sqlite' o 'postgresql')."""
        return self._dialect
