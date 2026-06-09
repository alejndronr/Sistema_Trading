#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# cleanup_v5.sh — Limpieza y estructura definitiva post-V5
# ══════════════════════════════════════════════════════════════════════════════
# Elimina archivos basura de versiones anteriores y reorganiza el proyecto.
# Hace backup de todo antes de borrar.
#
# Uso:
#   chmod +x cleanup_v5.sh
#   bash cleanup_v5.sh          # modo interactivo (pide confirmación)
#   bash cleanup_v5.sh --force  # borra sin preguntar
# ══════════════════════════════════════════════════════════════════════════════

set -uo pipefail

TRADING_DIR="/home/trading/sistema_trading"
BACKUP_DIR="$TRADING_DIR/.cleanup_backup_$(date +%Y%m%d_%H%M%S)"
FORCE="${1:-}"

echo "══════════════════════════════════════════════════════"
echo "  ZimaBlade — Limpieza y estructura definitiva V5"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "══════════════════════════════════════════════════════"

# ── 1. Backup preventivo ──────────────────────────────────────────────────────
echo ""
echo "[1/4] Creando backup de seguridad..."
mkdir -p "$BACKUP_DIR"

# Solo backupeamos los archivos que vamos a tocar/mover
BACKUP_FILES=(
    "live_engine.py"
    "indicators_v5.py"
    "backtest_v4.py"
    "dashboard_install"
    "backtesting"
    "strategies"
    "api.py"
    "upgrade_to_binance.py"
    "migrar_fifo.py"
    "fix_colores.py"
    "fix_indent.py"
    "fix_drawdowns.sql"
    "fix_db_final.sql"
    "parche_db.sql"
    "parche_db_2.sql"
    "ml_patch.py"
    "bootstrap_ml.py"
    "docker-compose.yml"
    "web"
    "reports"
    "journal"
    "execution"
    "risk"
)

for f in "${BACKUP_FILES[@]}"; do
    src="$TRADING_DIR/$f"
    if [ -e "$src" ]; then
        cp -r "$src" "$BACKUP_DIR/" 2>/dev/null || true
    fi
done
echo "      ✓ Backup en $BACKUP_DIR"

# ── 2. Mover indicators_v5.py al lugar correcto ───────────────────────────────
echo ""
echo "[2/4] Reorganizando estructura..."

# indicators_v5.py está en la raíz — debe estar en indicators/
if [ -f "$TRADING_DIR/indicators_v5.py" ]; then
    mkdir -p "$TRADING_DIR/indicators"
    mv "$TRADING_DIR/indicators_v5.py" "$TRADING_DIR/indicators/indicators_v5.py"
    echo "      ✓ indicators_v5.py → indicators/indicators_v5.py"
fi

# backtest_v4.py está en la raíz — debe estar en scripts/
if [ -f "$TRADING_DIR/backtest_v4.py" ]; then
    cp "$TRADING_DIR/backtest_v4.py" "$TRADING_DIR/scripts/backtest_v4.py"
    rm "$TRADING_DIR/backtest_v4.py"
    echo "      ✓ backtest_v4.py → scripts/backtest_v4.py"
fi

# Asegurar __init__.py en indicators/
touch "$TRADING_DIR/indicators/__init__.py"
echo "      ✓ indicators/__init__.py"

# ── 3. Eliminar archivos obsoletos ───────────────────────────────────────────
echo ""
echo "[3/4] Eliminando archivos obsoletos..."

# ── Motores anteriores (sustituidos por live_engine_v5.py) ────────────────────
OBSOLETE_FILES=(
    # Motor V1/V2 — sustituido por live_engine_v5.py
    "live_engine.py"

    # Parches y fixes de versiones anteriores (ya integrados)
    "fix_colores.py"
    "fix_indent.py"
    "fix_meta_labeler.py"
    "ml_patch.py"

    # SQLs de migración ya ejecutados
    "fix_drawdowns.sql"
    "fix_db_final.sql"
    "parche_db.sql"
    "parche_db_2.sql"
    "migrar_fifo.py"

    # Scripts de upgrade ya ejecutados
    "upgrade_to_binance.py"
    "bootstrap_ml.py"

    # Infraestructura no usada
    "docker-compose.yml"
    "api.py"

    # Carpetas obsoletas
    "dashboard_install"   # el dashboard definitivo está en dashboard/
    "web"                 # nunca se usó
    "reports"             # vacío
    "execution"           # lógica movida al motor V5
    "risk"                # lógica integrada en live_engine_v5.py

    # Carpetas de backtesting V1/V2 (sustituidas por scripts/backtest_v4.py)
    "backtesting"
    "strategies"          # estrategias ahora están en live_engine_v5.py

    # Nombres extraños (dependencias instaladas mal)
    "=1.35.0"
    "=2.31.0"
    "=5.20.0"
)

for f in "${OBSOLETE_FILES[@]}"; do
    target="$TRADING_DIR/$f"
    if [ -e "$target" ]; then
        if [ "$FORCE" = "--force" ]; then
            rm -rf "$target"
            echo "      🗑  $f"
        else
            echo "      📋 Marcar para borrar: $f"
        fi
    fi
done

if [ "$FORCE" != "--force" ]; then
    echo ""
    echo "  ⚠️  MODO PREVIEW — los archivos NO han sido borrados."
    echo "      Revisa la lista y ejecuta con --force para confirmar:"
    echo "      bash cleanup_v5.sh --force"
fi

# ── 4. Estructura final ───────────────────────────────────────────────────────
echo ""
echo "[4/4] Estructura definitiva del proyecto:"
echo ""
echo "  sistema_trading/"
echo "  ├── live_engine_v5.py       ← Motor principal V5"
echo "  ├── paper_portfolio.py       ← Portfolio paper trading"
echo "  ├── diagnose_bot.py          ← Diagnóstico del sistema"
echo "  ├── .env                     ← Configuración (no tocar)"
echo "  ├── requirements.txt"
echo "  ├── indicators/"
echo "  │   ├── __init__.py"
echo "  │   └── indicators_v5.py    ← Capas V5 (patrones, S/R, Fib)"
echo "  ├── ml/"
echo "  │   ├── meta_labeler.py     ← ML MetaLabeler"
echo "  │   ├── retrain_model.py    ← Reentrenamiento automático"
echo "  │   └── model.joblib        ← Modelo entrenado"
echo "  ├── monitoring/"
echo "  │   └── telegram_bot.py     ← Bot de Telegram"
echo "  ├── dashboard/"
echo "  │   └── app.py              ← Dashboard Streamlit"
echo "  ├── scripts/"
echo "  │   ├── backtest_v4.py      ← Backtester V5"
echo "  │   ├── download_data.py    ← Descarga OHLCV"
echo "  │   └── run_backtest.py     ← CLI backtest V2 (legacy)"
echo "  ├── data/"
echo "  │   └── db/trading.db       ← SQLite OHLCV"
echo "  ├── config/"
echo "  │   └── settings.py"
echo "  ├── systemd/"
echo "  │   ├── trading-engine.service"
echo "  │   ├── trading-dashboard.service"
echo "  │   ├── ohlcv-update.service/.timer"
echo "  │   └── trading-retrain.service/.timer"
echo "  ├── logs/"
echo "  ├── tests/"
echo "  └── venv/"
echo ""
echo "══════════════════════════════════════════════════════"
if [ "$FORCE" = "--force" ]; then
    echo "  ✅ Limpieza completada"
else
    echo "  ℹ️  Preview completado — usa --force para ejecutar"
fi
echo "══════════════════════════════════════════════════════"
