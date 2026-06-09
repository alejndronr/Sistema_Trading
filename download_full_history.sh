#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# download_full_history.sh — Descarga histórico completo de todos los pares
# ══════════════════════════════════════════════════════════════════════════════
# Descarga el máximo histórico disponible en Binance para pares USDC.
# BTC/USDC disponible desde ~2019, altcoins desde ~2020-2021.
# Timeframes: 1h (análisis) + 1d (ciclos macro)
#
# Uso:
#   bash download_full_history.sh
# ══════════════════════════════════════════════════════════════════════════════

set -uo pipefail
TRADING_DIR="/home/trading/sistema_trading"
VENV="$TRADING_DIR/venv"

echo "══════════════════════════════════════════════════════"
echo "  ZimaBlade — Descarga histórico completo USDC"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "══════════════════════════════════════════════════════"

source "$VENV/bin/activate"
cd "$TRADING_DIR"

# Pares y años disponibles en Binance
declare -A YEARS=(
    ["BTC/USDC"]="5"     # Desde ~2019
    ["ETH/USDC"]="4"     # Desde ~2020
    ["BNB/USDC"]="4"
    ["SOL/USDC"]="3"     # Desde ~2021
    ["LINK/USDC"]="4"
    ["AVAX/USDC"]="3"
)

TOTAL_CANDLES=0

for PAIR in "BTC/USDC" "ETH/USDC" "BNB/USDC" "SOL/USDC" "LINK/USDC" "AVAX/USDC"; do
    YRS="${YEARS[$PAIR]}"
    echo ""
    echo "── $PAIR ($YRS años) ──"

    # 1H — para estrategias y backtest
    echo "  Descargando 1H..."
    python scripts/download_data.py \
        --pairs "$PAIR" \
        --timeframes 1h \
        --years "$YRS" \
        2>&1 | tail -2

    # 1D — para análisis de ciclos macro (halving BTC)
    echo "  Descargando 1D..."
    python scripts/download_data.py \
        --pairs "$PAIR" \
        --timeframes 1d \
        --years "$YRS" \
        2>&1 | tail -2
done

echo ""
echo "══════════════════════════════════════════════════════"
echo "  Estado final de la BD:"
echo "══════════════════════════════════════════════════════"
python scripts/download_data.py --status

echo ""
echo "✅ Descarga completada"
