#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# install_v5.sh — Instala el Motor V5 en el ZimaBlade
# ══════════════════════════════════════════════════════════════════════════════
set -uo pipefail

TRADING_DIR="/home/trading/sistema_trading"
VENV="$TRADING_DIR/venv"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "══════════════════════════════════════════════════════"
echo "  ZimaBlade — Instalación Motor V5 + Indicadores V5"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "══════════════════════════════════════════════════════"

# ── 1. Backup del motor actual ────────────────────────────────────────────────
echo ""
echo "[1/5] Backup del motor actual..."
TS=$(date +%Y%m%d_%H%M%S)
BACKUP="$TRADING_DIR/.v4_backup_$TS"
mkdir -p "$BACKUP"
cp "$TRADING_DIR/live_engine.py"               "$BACKUP/" 2>/dev/null || true
cp "$TRADING_DIR/scripts/backtest_v4.py"       "$BACKUP/" 2>/dev/null || true
echo "      ✓ Backup en $BACKUP"

# ── 2. Copiar archivos V5 ─────────────────────────────────────────────────────
echo ""
echo "[2/5] Copiando archivos V5..."

# indicators_v5.py → indicators/
mkdir -p "$TRADING_DIR/indicators"
if [ -f "$SCRIPT_DIR/indicators_v5.py" ]; then
    cp "$SCRIPT_DIR/indicators_v5.py" "$TRADING_DIR/indicators/indicators_v5.py"
    echo "      ✓ indicators/indicators_v5.py"
else
    echo "      ⚠️  indicators_v5.py no encontrado junto a este script"
fi

# live_engine_v5.py → raíz del proyecto
if [ -f "$SCRIPT_DIR/live_engine_v5.py" ]; then
    cp "$SCRIPT_DIR/live_engine_v5.py" "$TRADING_DIR/live_engine_v5.py"
    echo "      ✓ live_engine_v5.py"
fi

# backtest_v4.py actualizado → scripts/
if [ -f "$SCRIPT_DIR/backtest_v4.py" ]; then
    cp "$SCRIPT_DIR/backtest_v4.py" "$TRADING_DIR/scripts/backtest_v4.py"
    echo "      ✓ scripts/backtest_v4.py"
fi

# __init__.py para el módulo indicators
if [ ! -f "$TRADING_DIR/indicators/__init__.py" ]; then
    touch "$TRADING_DIR/indicators/__init__.py"
    echo "      ✓ indicators/__init__.py creado"
fi

# ── 3. Verificar sintaxis ─────────────────────────────────────────────────────
echo ""
echo "[3/5] Verificando sintaxis de los archivos..."
source "$VENV/bin/activate"

for f in \
    "$TRADING_DIR/indicators/indicators_v5.py" \
    "$TRADING_DIR/live_engine_v5.py" \
    "$TRADING_DIR/scripts/backtest_v4.py"
do
    if [ -f "$f" ]; then
        python -c "import ast; ast.parse(open('$f').read()); print('  ✓ $(basename $f)')" \
            2>/dev/null || echo "  ✗ Error de sintaxis en $(basename $f)"
    fi
done

# ── 4. Actualizar servicio systemd para usar V5 ───────────────────────────────
echo ""
echo "[4/5] Actualizando servicio systemd..."

CURRENT_EXEC=$(grep "ExecStart" /etc/systemd/system/trading-engine.service 2>/dev/null | head -1)
if echo "$CURRENT_EXEC" | grep -q "live_engine.py"; then
    sudo sed -i 's|live_engine\.py|live_engine_v5.py|g' \
        /etc/systemd/system/trading-engine.service
    echo "      ✓ Servicio actualizado: live_engine.py → live_engine_v5.py"
else
    echo "      ℹ️  Servicio ya usa V5 o no se detectó live_engine.py"
fi

sudo systemctl daemon-reload
echo "      ✓ systemd recargado"

# ── 5. Reiniciar ──────────────────────────────────────────────────────────────
echo ""
echo "[5/5] Reiniciando motor..."
sudo systemctl restart trading-engine

sleep 4
if systemctl is-active --quiet trading-engine; then
    echo "      ✓ Motor V5 activo"
else
    echo "      ✗ Motor no arrancó — revisar logs:"
    echo "        journalctl -u trading-engine -n 30 --no-pager"
    exit 1
fi

echo ""
echo "══════════════════════════════════════════════════════"
echo "  ✅ Motor V5 instalado y corriendo"
echo "══════════════════════════════════════════════════════"
echo ""
echo "  Verificar arranque:"
echo "    journalctl -u trading-engine -f | grep -E 'v5|engine_v5|signal_queued|V5'"
echo ""
echo "  Ejecutar backtest V5 (últimos 3 meses):"
echo "    source $VENV/bin/activate"
echo "    python scripts/backtest_v4.py --months 3 --retrain"
echo ""
echo "  Backup V4 en: $BACKUP"
echo ""
