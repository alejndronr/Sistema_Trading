"""
Logging Configuration — Sistema de Trading
==========================================
Configura structlog con salida formateada y coloreada para desarrollo,
y JSON estructurado para producción (Grafana/ELK stack en Fase 2+).
"""

import logging
import sys
from pathlib import Path
from typing import Any

import structlog

from config.settings import LOGS_DIR, env_settings


def setup_logging() -> None:
    """
    Inicializa structlog. Llamar una vez al inicio de cada proceso.
    - En desarrollo: salida bonita y coloreada en consola.
    - En producción: JSON estructurado para ingesta por Grafana/ELK.
    """
    log_level = getattr(logging, env_settings.log_level.upper(), logging.INFO)
    is_production = env_settings.environment == "production"

    # ── File handler (siempre activo) ──────────────────────────────────────────
    log_file = LOGS_DIR / "trading.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(log_level)

    # ── Console handler ────────────────────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    logging.basicConfig(
        level=log_level,
        handlers=[file_handler, console_handler],
        format="%(message)s",
    )

    # ── Structlog processors ───────────────────────────────────────────────────
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.processors.StackInfoRenderer(),
    ]

    if is_production:
        # JSON para producción
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Rich console output para desarrollo
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Factory para obtener un logger con el nombre del módulo.

    Uso:
        logger = get_logger(__name__)
        logger.info("trade_opened", symbol="BTC/USDT", price=65000)
    """
    return structlog.get_logger(name)
