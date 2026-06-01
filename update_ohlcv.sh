#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# update_ohlcv.sh — Actualización incremental de velas OHLCV (ejecución diaria)
# ══════════════════════════════════════════════════════════════════════════════
# Se ejecuta una vez al día a las 02:30 UTC (sesión asiática, bajo tráfico).
# Descarga solo las velas nuevas desde el último timestamp guardado en BD.
# El fetcher detecta automáticamente el gap y hace upsert — nunca duplica.
#
# Uso manual:
#   ./update_ohlcv.sh           # actualización incremental
#   ./update_ohlcv.sh --status  # ver estado de la BD sin descargar
#   ./update_ohlcv.sh --full    # forzar descarga completa de 3 meses
# ══════════════════════════════════════════════════════════════════════════════

set -uo pipefail

TRADING_DIR="/home/trading/sistema_trading"
VENV="$TRADING_DIR/venv"

PAIRS="BTC/USDC ETH/USDC SOL/USDC BNB/USDC LINK/USDC AVAX/USDC"
TIMEFRAMES="1h 15m"

# ── Solo status ───────────────────────────────────────────────────────────────
if [ "${1:-}" = "--status" ]; then
    source "$VENV/bin/activate"
    python "$TRADING_DIR/scripts/download_data.py" --status
    exit 0
fi

# ── Descarga forzada completa (3 meses) ───────────────────────────────────────
if [ "${1:-}" = "--full" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Descarga completa 3 meses..."
    source "$VENV/bin/activate"
    python "$TRADING_DIR/scripts/download_data.py" \
        --pairs $PAIRS \
        --timeframes $TIMEFRAMES \
        --years 0.25
    exit $?
fi

# ── Actualización incremental (modo normal) ───────────────────────────────────
# --years 0.05 ≈ 18 días, pero el fetcher detecta el último timestamp
# y solo descarga desde ahí → en la práctica descarga ~24h de velas nuevas
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Actualizando velas OHLCV (incremental)..."

source "$VENV/bin/activate"

python "$TRADING_DIR/scripts/download_data.py" \
    --pairs $PAIRS \
    --timeframes $TIMEFRAMES \
    --years 0.05

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ OHLCV actualizado"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ Error en actualización (exit $EXIT_CODE)"
fi

exit $EXIT_CODE
