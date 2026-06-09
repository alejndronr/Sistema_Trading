"""
Trade Logger — Journal Automático de Operaciones
================================================
Registra cada trade con todos los campos del prompt maestro:
  - Fecha, par, dirección, timeframe
  - Razón de entrada (qué señales confluyen)
  - Precio entrada, SL, TP1, TP2
  - Resultado ($, %, R)
  - Razón de salida
  - Calidad del setup (A+, A, B, C)
  - Observaciones post-trade
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Optional

from config.logging_config import get_logger
from config.settings import LOGS_DIR, ROOT_DIR
from data.storage import OHLCVStorage

logger = get_logger(__name__)

JOURNAL_CSV = ROOT_DIR / "journal" / "trades.csv"
JOURNAL_CSV.parent.mkdir(parents=True, exist_ok=True)


class TradeLogger:
    """
    Registra trades en:
    1. SQLite (via OHLCVStorage) para queries y análisis
    2. CSV plano para revisión manual y export a Koinly/CoinTracking
    """

    CSV_HEADERS = [
        "trade_id", "date", "pair", "direction", "timeframe", "strategy",
        "entry_price", "stop_loss", "take_profit_1", "take_profit_2",
        "exit_price", "position_size", "risk_amount",
        "pnl_usd", "pnl_pct", "r_multiple",
        "entry_time", "exit_time", "duration_hours",
        "entry_reason", "exit_reason", "setup_quality",
        "market_regime", "observations", "is_backtest",
    ]

    def __init__(self, use_db: bool = True, use_csv: bool = True):
        self.use_db = use_db
        self.use_csv = use_csv
        self._storage = OHLCVStorage() if use_db else None

        # Inicializar CSV si no existe
        if use_csv and not JOURNAL_CSV.exists():
            self._init_csv()

    def _init_csv(self) -> None:
        with open(JOURNAL_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_HEADERS)
            writer.writeheader()

    def log_trade(
        self,
        trade_id: str,
        symbol: str,
        direction: str,
        timeframe: str,
        strategy: str,
        entry_price: float,
        stop_loss: float,
        take_profit_1: float,
        exit_price: float,
        position_size: float,
        risk_amount: float,
        pnl_usd: float,
        entry_time: datetime,
        exit_time: datetime,
        entry_reason: str = "",
        exit_reason: str = "",
        setup_quality: str = "A",
        take_profit_2: Optional[float] = None,
        market_regime: str = "unknown",
        observations: str = "",
        is_backtest: bool = True,
    ) -> None:
        """
        Registra un trade completo en el journal.
        Llamar después de que una posición se cierra.
        """
        duration_h = (exit_time - entry_time).total_seconds() / 3600
        pnl_pct = pnl_usd / (entry_price * position_size) * 100 if position_size > 0 else 0
        r_multiple = pnl_usd / risk_amount if risk_amount > 0 else 0

        trade_data = {
            "trade_id": trade_id,
            "strategy": strategy,
            "symbol": symbol,
            "timeframe": timeframe,
            "direction": direction,
            "setup_quality": setup_quality,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit_1": take_profit_1,
            "take_profit_2": take_profit_2,
            "exit_price": exit_price,
            "position_size": position_size,
            "risk_amount": risk_amount,
            "pnl_usd": round(pnl_usd, 4),
            "pnl_pct": round(pnl_pct, 4),
            "r_multiple": round(r_multiple, 3),
            "entry_time": entry_time,
            "exit_time": exit_time,
            "duration_hours": round(duration_h, 2),
            "entry_reason": entry_reason,
            "exit_reason": exit_reason,
            "market_regime": market_regime,
            "observations": observations,
            "is_backtest": is_backtest,
        }

        # Guardar en SQLite
        if self.use_db and self._storage:
            try:
                self._storage.save_trade(trade_data)
            except Exception as e:
                logger.error("trade_db_save_failed", trade_id=trade_id, error=str(e))

        # Guardar en CSV
        if self.use_csv:
            try:
                self._append_to_csv(trade_data)
            except Exception as e:
                logger.error("trade_csv_save_failed", trade_id=trade_id, error=str(e))

        logger.info(
            "trade_logged",
            trade_id=trade_id[:8],
            symbol=symbol,
            pnl=round(pnl_usd, 2),
            r=round(r_multiple, 2),
            quality=setup_quality,
        )

    def _append_to_csv(self, trade_data: dict) -> None:
        """Agrega un trade al CSV del journal."""
        row = {
            "trade_id": trade_data.get("trade_id", ""),
            "date": trade_data.get("entry_time", "").strftime("%Y-%m-%d") if hasattr(trade_data.get("entry_time"), "strftime") else "",
            "pair": trade_data.get("symbol", ""),
            "direction": trade_data.get("direction", ""),
            "timeframe": trade_data.get("timeframe", ""),
            "strategy": trade_data.get("strategy", ""),
            "entry_price": trade_data.get("entry_price", ""),
            "stop_loss": trade_data.get("stop_loss", ""),
            "take_profit_1": trade_data.get("take_profit_1", ""),
            "take_profit_2": trade_data.get("take_profit_2", ""),
            "exit_price": trade_data.get("exit_price", ""),
            "position_size": trade_data.get("position_size", ""),
            "risk_amount": trade_data.get("risk_amount", ""),
            "pnl_usd": trade_data.get("pnl_usd", ""),
            "pnl_pct": trade_data.get("pnl_pct", ""),
            "r_multiple": trade_data.get("r_multiple", ""),
            "entry_time": str(trade_data.get("entry_time", "")),
            "exit_time": str(trade_data.get("exit_time", "")),
            "duration_hours": trade_data.get("duration_hours", ""),
            "entry_reason": trade_data.get("entry_reason", ""),
            "exit_reason": trade_data.get("exit_reason", ""),
            "setup_quality": trade_data.get("setup_quality", ""),
            "market_regime": trade_data.get("market_regime", ""),
            "observations": trade_data.get("observations", ""),
            "is_backtest": int(trade_data.get("is_backtest", 1)),
        }

        with open(JOURNAL_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_HEADERS)
            writer.writerow(row)

    @property
    def journal_path(self) -> Path:
        return JOURNAL_CSV
