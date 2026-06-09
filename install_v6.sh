#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# install_v6.sh — Instala el Motor V6 AdaptiveCycle en ZimaBlade
# ══════════════════════════════════════════════════════════════════════════════
set -uo pipefail

TRADING_DIR="/home/trading/sistema_trading"
VENV="$TRADING_DIR/venv"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "══════════════════════════════════════════════════════"
echo "  ZimaBlade — Instalación Motor V6 AdaptiveCycle"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "══════════════════════════════════════════════════════"

# ── 1. Backup ─────────────────────────────────────────────────────────────────
echo ""
echo "[1/5] Backup del motor actual..."
TS=$(date +%Y%m%d_%H%M%S)
BACKUP="$TRADING_DIR/.v5_backup_$TS"
mkdir -p "$BACKUP"
for f in live_engine_v5.py scripts/backtest_v4.py; do
    [ -f "$TRADING_DIR/$f" ] && cp "$TRADING_DIR/$f" "$BACKUP/" 2>/dev/null || true
done
echo "      ✓ Backup en $BACKUP"

# ── 2. Copiar archivos V6 ─────────────────────────────────────────────────────
echo ""
echo "[2/5] Copiando archivos V6..."

copy_file() {
    local src="$SCRIPT_DIR/$1"
    local dst="$TRADING_DIR/$2"
    if [ -f "$src" ]; then
        mkdir -p "$(dirname "$dst")"
        cp "$src" "$dst"
        echo "      ✓ $2"
    else
        echo "      ⚠️  $1 no encontrado"
    fi
}

copy_file "cycle_detector.py"   "cycle_detector.py"
copy_file "live_engine_v6.py"   "live_engine_v6.py"
copy_file "backtest_v5.py"      "scripts/backtest_v5.py"

# ── 3. Verificar sintaxis ─────────────────────────────────────────────────────
echo ""
echo "[3/5] Verificando sintaxis..."
source "$VENV/bin/activate"

for f in cycle_detector.py live_engine_v6.py scripts/backtest_v5.py; do
    fp="$TRADING_DIR/$f"
    if [ -f "$fp" ]; then
        python -c "import ast; ast.parse(open('$fp').read()); print('  ✓ $f')" \
            2>/dev/null || echo "  ✗ Error en $f"
    fi
done

# Verificar que CycleDetector puede leer datos diarios
echo ""
echo "  Verificando CycleDetector con datos reales..."
python - << 'PYEOF'
import sys
sys.path.insert(0, '/home/trading/sistema_trading')
from cycle_detector import CycleDetector, load_daily_ohlcv

db = '/home/trading/sistema_trading/data/db/trading.db'
detector = CycleDetector()
df = load_daily_ohlcv('BTC/USDC', db)
if len(df) < 50:
    print('  ⚠️  Datos diarios insuficientes — ejecuta download_full_history.sh')
else:
    state = detector.detect(df)
    print(f'  ✅ CycleDetector OK')
    print(f'     BTC/USDC → Fase: {state.phase} (convicción {state.conviction_score}/100)')
    print(f'     Desde ATH: {state.pct_from_ath:+.1f}% | RSI diario: {state.rsi_daily:.0f}')
    print(f'     Estrategias activas: {state.active_strategies}')
    print(f'     Risk multiplier: {state.risk_multiplier:.0%}')
PYEOF

# ── 4. Actualizar servicio systemd ────────────────────────────────────────────
echo ""
echo "[4/5] Actualizando servicio systemd..."

if [ -f /etc/systemd/system/trading-engine.service ]; then
    sudo sed -i 's|live_engine_v5\.py|live_engine_v6.py|g' \
        /etc/systemd/system/trading-engine.service
    sudo sed -i 's|live_engine\.py|live_engine_v6.py|g' \
        /etc/systemd/system/trading-engine.service
    sudo systemctl daemon-reload
    echo "      ✓ Servicio actualizado → live_engine_v6.py"
else
    echo "      ⚠️  Servicio systemd no encontrado — actualizar manualmente"
fi

# ── 5. Reiniciar y verificar ──────────────────────────────────────────────────
echo ""
echo "[5/5] Reiniciando motor V6..."
sudo systemctl restart trading-engine
sleep 5

if systemctl is-active --quiet trading-engine; then
    echo "      ✓ Motor V6 activo"
    journalctl -u trading-engine -n 3 --no-pager | grep -E "v6|6.0.0|AdaptiveCycle" || true
else
    echo "      ✗ Motor no arrancó — ver logs:"
    echo "        journalctl -u trading-engine -n 20 --no-pager"
    exit 1
fi

echo ""
echo "══════════════════════════════════════════════════════"
echo "  ✅ Motor V6 AdaptiveCycle instalado"
echo "══════════════════════════════════════════════════════"
echo ""
echo "  Verificar ciclos detectados:"
echo "    source $VENV/bin/activate"
echo "    python cycle_detector.py"
echo ""
echo "  Backtest histórico completo (todos los datos):"
echo "    python scripts/backtest_v5.py --report-by-phase --dry-run"
echo ""
echo "  Backtest + retrain:"
echo "    python scripts/backtest_v5.py --report-by-phase --retrain"
echo ""
echo "  Monitor del motor en vivo:"
echo "    journalctl -u trading-engine -f | grep -E 'v6|cycle_updated|signal_queued|trade_opening'"
echo ""
