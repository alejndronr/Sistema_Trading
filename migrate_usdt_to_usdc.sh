#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# migrate_usdt_to_usdc.sh — V3
# Migra pares USDT → USDC en código Python y PostgreSQL.
# Usuario BD: trading · Host: 127.0.0.1
# ══════════════════════════════════════════════════════════════════════════════

# NOTA: NO usar set -e porque grep devuelve exit 1 cuando no encuentra nada
set -uo pipefail

TRADING_DIR="/home/trading/sistema_trading"
BACKUP_DIR="$TRADING_DIR/.usdt_backup_$(date +%Y%m%d_%H%M%S)"

echo "══════════════════════════════════════════════════════"
echo "  ZimaBlade — Migración USDT → USDC (V3)"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "══════════════════════════════════════════════════════"

# ── 1. Backup ──────────────────────────────────────────────────────────────
echo ""
echo "[1/4] Creando backup..."
mkdir -p "$BACKUP_DIR"

FILES_TO_BACKUP=(
    "config/settings.py"
    "live_engine.py"
    "ml/retrain_model.py"
    "scripts/download_data.py"
    "scripts/run_backtest.py"
    "scripts/run_optimization.py"
    "backtesting/engine.py"
    "dashboard/app.py"
    "dashboard_install/app.py"
)

for f in "${FILES_TO_BACKUP[@]}"; do
    src="$TRADING_DIR/$f"
    if [ -f "$src" ]; then
        dst_dir="$BACKUP_DIR/$(dirname $f)"
        mkdir -p "$dst_dir"
        cp "$src" "$dst_dir/$(basename $f)"
        echo "      backed up: $f"
    fi
done
echo "      ✓ Backup en $BACKUP_DIR"

# ── 2. Migración de código ─────────────────────────────────────────────────
echo ""
echo "[2/4] Migrando pares USDT → USDC en código..."

do_migrate() {
    local file="$TRADING_DIR/$1"
    if [ -f "$file" ]; then
        sed -i 's|/USDT"|/USDC"|g' "$file"
        sed -i "s|/USDT'|/USDC'|g" "$file"
        echo "      ✓ $1"
    else
        echo "      ⚠️  $1 no encontrado (skip)"
    fi
}

do_migrate "config/settings.py"
do_migrate "live_engine.py"
do_migrate "ml/retrain_model.py"
do_migrate "scripts/download_data.py"
do_migrate "scripts/run_backtest.py"
do_migrate "scripts/run_optimization.py"
do_migrate "backtesting/engine.py"
do_migrate "dashboard/app.py"
do_migrate "dashboard_install/app.py"

# ── 3. Verificar residuos ─────────────────────────────────────────────────
echo ""
echo "[3/4] Verificando migración de código..."

# || true evita que grep exit-1 (sin resultados) mate el script
REMAINING=$(grep -rn "/USDT" "$TRADING_DIR" \
    --include="*.py" \
    --exclude-dir=__pycache__ \
    2>/dev/null | grep -v ".usdt_backup_" | grep '"/USDT"\|'"'"'/USDT'"'" | wc -l || true)

MIGRATED=$(grep -rn "/USDC" "$TRADING_DIR" \
    --include="*.py" \
    --exclude-dir=__pycache__ \
    2>/dev/null | grep -v ".usdt_backup_" | grep '"/USDC"\|'"'"'/USDC'"'" | wc -l || true)

REMAINING=${REMAINING:-0}
MIGRATED=${MIGRATED:-0}

echo "      Pares USDC en código: $MIGRATED"
echo "      Pares USDT restantes: $REMAINING"

if [ "$REMAINING" -gt 0 ]; then
    echo "      Archivos con USDT aún presentes:"
    grep -rn "/USDT" "$TRADING_DIR" \
        --include="*.py" \
        --exclude-dir=__pycache__ \
        2>/dev/null \
        | grep -v ".usdt_backup_" \
        | grep '"/USDT"\|'"'"'/USDT'"'" \
        | sed "s|$TRADING_DIR/||" \
        | head -10 || true
fi

echo "      ✓ Verificación completada"

# ── 4. Migrar PostgreSQL ───────────────────────────────────────────────────
echo ""
echo "[4/4] Migrando simbología en PostgreSQL..."

ENV_FILE="$TRADING_DIR/.env"

if [ ! -f "$ENV_FILE" ]; then
    echo "      ✗ .env no encontrado"
    echo "        Ejecuta manualmente:"
    echo "        psql -U trading -d trading_db -h 127.0.0.1"
    echo "        UPDATE ohlcv SET symbol=REPLACE(symbol,'/USDT','/USDC') WHERE symbol LIKE '%/USDT';"
    echo ""
    echo "══════════════════════════════════════════════════════"
    echo "  ✅ Código migrado. BD pendiente de migración manual."
    echo "══════════════════════════════════════════════════════"
    exit 0
fi

# Parsear DATABASE_URL con Python (más robusto que bash puro)
read -r DB_USER DB_PASS DB_HOST DB_PORT DB_NAME << PYOUT
$(python3 - "$ENV_FILE" << 'PYEOF'
import re, sys

env_file = sys.argv[1]
url = ""
with open(env_file) as f:
    for line in f:
        line = line.strip()
        if line.startswith("DATABASE_URL"):
            url = line.split("=", 1)[1].strip().strip('"').strip("'").split("#")[0].strip()
            break

# Eliminar prefijo asyncpg
url = url.replace("+asyncpg", "").replace("postgresql+asyncpg://", "postgresql://")

m = re.match(r"postgresql://([^:@]+):?([^@]*)@([^:/]+):?(\d+)?/([^\s?]+)", url)
if m:
    user  = m.group(1)
    pwd   = m.group(2)
    host  = m.group(3)
    port  = m.group(4) or "5432"
    dbname= m.group(5)
    # Forzar IPv4
    if host in ("localhost", "::1"):
        host = "127.0.0.1"
    print(user, pwd, host, port, dbname)
else:
    # Fallback a valores seguros
    print("trading", "", "127.0.0.1", "5432", "trading_db")
PYEOF
)
PYOUT

echo "      Usuario:       $DB_USER"
echo "      Host:          $DB_HOST:$DB_PORT"
echo "      Base de datos: $DB_NAME"

# Ejecutar migración SQL
PGPASSWORD="$DB_PASS" psql \
    -U "$DB_USER" \
    -d "$DB_NAME" \
    -h "$DB_HOST" \
    -p "$DB_PORT" \
    -v ON_ERROR_STOP=0 \
    -c "UPDATE ohlcv            SET symbol = REPLACE(symbol,'/USDT','/USDC') WHERE symbol LIKE '%/USDT';" \
    -c "UPDATE trades_journal   SET symbol = REPLACE(symbol,'/USDT','/USDC') WHERE symbol LIKE '%/USDT';" \
    -c "UPDATE positions        SET symbol = REPLACE(symbol,'/USDT','/USDC') WHERE symbol LIKE '%/USDT';" \
    -c "UPDATE manual_investments SET symbol = REPLACE(symbol,'/USDT','/USDC') WHERE symbol LIKE '%/USDT';" \
    -c "UPDATE manual_closings  SET symbol = REPLACE(symbol,'/USDT','/USDC') WHERE symbol LIKE '%/USDT';" \
    -c "SELECT 'ohlcv'            AS tabla, COUNT(*) FILTER (WHERE symbol LIKE '%/USDC') AS usdc, COUNT(*) FILTER (WHERE symbol LIKE '%/USDT') AS usdt_restantes FROM ohlcv
        UNION ALL
        SELECT 'trades_journal',   COUNT(*) FILTER (WHERE symbol LIKE '%/USDC'), COUNT(*) FILTER (WHERE symbol LIKE '%/USDT') FROM trades_journal
        UNION ALL
        SELECT 'positions',        COUNT(*) FILTER (WHERE symbol LIKE '%/USDC'), COUNT(*) FILTER (WHERE symbol LIKE '%/USDT') FROM positions
        UNION ALL
        SELECT 'manual_investments',COUNT(*) FILTER (WHERE symbol LIKE '%/USDC'), COUNT(*) FILTER (WHERE symbol LIKE '%/USDT') FROM manual_investments;"

PSQL_EXIT=$?

echo ""
if [ $PSQL_EXIT -eq 0 ]; then
    echo "      ✓ PostgreSQL migrado correctamente"
else
    echo "      ⚠️  psql terminó con código $PSQL_EXIT"
    echo "        Ejecuta manualmente si hay errores de auth:"
    echo "        psql -U $DB_USER -d $DB_NAME -h $DB_HOST -p $DB_PORT"
fi

# ── Resumen ────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════"
echo "  ✅ Migración completada"
echo "══════════════════════════════════════════════════════"
echo ""
echo "  Próximos pasos:"
echo "    sudo systemctl restart trading-engine trading-dashboard"
echo "    journalctl -u trading-engine -f | grep -E 'USDC|engine_live'"
echo ""
echo "  Re-descargar OHLCV histórico en USDC (recomendado):"
echo "    source venv/bin/activate"
echo "    python scripts/download_data.py \\"
echo "      --pairs BTC/USDC ETH/USDC SOL/USDC BNB/USDC LINK/USDC AVAX/USDC \\"
echo "      --timeframes 1h 15m --years 2"
echo ""
echo "  Backup en: $BACKUP_DIR"
echo ""
