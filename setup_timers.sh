#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# setup_timers.sh — Configura los 2 timers automáticos del sistema
# ══════════════════════════════════════════════════════════════════════════════
#
# Timer 1: ohlcv-update   — cada día a las 02:30 UTC
#          Descarga velas nuevas (1H + 15M) de forma incremental.
#          02:30 = sesión asiática, ZimaBlade en reposo, mínimo impacto.
#
# Timer 2: trading-retrain — domingos 03:00 UTC (30 min después del OHLCV)
#          Reentrena el MetaLabeler si hay ≥ 30 trades nuevos.
#          Se ejecuta después del update para tener datos frescos.
#
# Uso:
#   chmod +x setup_timers.sh
#   sudo ./setup_timers.sh
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

TRADING_DIR="/home/trading/sistema_trading"
VENV="$TRADING_DIR/venv"

echo "══════════════════════════════════════════════════════"
echo "  ZimaBlade — Configuración de Timers Automáticos"
echo "══════════════════════════════════════════════════════"

# ── Copiar scripts al directorio del proyecto ─────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "$SCRIPT_DIR/update_ohlcv.sh" ]; then
    if [ "$SCRIPT_DIR/update_ohlcv.sh" != "$TRADING_DIR/update_ohlcv.sh" ]; then
        cp "$SCRIPT_DIR/update_ohlcv.sh" "$TRADING_DIR/update_ohlcv.sh"
        echo "      ✓ update_ohlcv.sh copiado"
    else
        echo "      ✓ update_ohlcv.sh ya en su sitio"
    fi
    chmod +x "$TRADING_DIR/update_ohlcv.sh"
else
    echo "      ⚠️  update_ohlcv.sh no encontrado junto a este script"
fi

# ── Service 1: OHLCV update ───────────────────────────────────────────────────
echo ""
echo "[1/4] Creando ohlcv-update.service..."
cat > /etc/systemd/system/ohlcv-update.service << SERVICE
[Unit]
Description=Actualización incremental OHLCV — ZimaBlade
After=network.target
Wants=network.target

[Service]
Type=oneshot
User=trading
WorkingDirectory=${TRADING_DIR}
Environment=PATH=${VENV}/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=${TRADING_DIR}/update_ohlcv.sh
StandardOutput=journal
StandardError=journal
SyslogIdentifier=ohlcv-update
TimeoutStartSec=600

[Install]
WantedBy=multi-user.target
SERVICE
echo "      ✓ ohlcv-update.service"

# ── Timer 1: cada día 02:30 UTC ───────────────────────────────────────────────
echo "[2/4] Creando ohlcv-update.timer (02:30 UTC diario)..."
cat > /etc/systemd/system/ohlcv-update.timer << TIMER
[Unit]
Description=Actualización diaria OHLCV — ZimaBlade
Requires=ohlcv-update.service

[Timer]
# 02:30 UTC = sesión asiática, motor en reposo, mínimo impacto en ZimaBlade
OnCalendar=*-*-* 02:30:00 UTC
Persistent=true
RandomizedDelaySec=120

[Install]
WantedBy=timers.target
TIMER
echo "      ✓ ohlcv-update.timer (02:30 UTC)"

# ── Service 2: ML retrain ─────────────────────────────────────────────────────
echo "[3/4] Creando trading-retrain.service..."
cat > /etc/systemd/system/trading-retrain.service << SERVICE
[Unit]
Description=Reentrenamiento MetaLabeler — ZimaBlade
After=postgresql.service ohlcv-update.service
Wants=postgresql.service

[Service]
Type=oneshot
User=trading
WorkingDirectory=${TRADING_DIR}
Environment=PATH=${VENV}/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=${VENV}/bin/python ml/retrain_model.py
StandardOutput=journal
StandardError=journal
SyslogIdentifier=trading-retrain
TimeoutStartSec=2400

[Install]
WantedBy=multi-user.target
SERVICE
echo "      ✓ trading-retrain.service"

# ── Timer 2: domingos 03:00 UTC (30 min después del OHLCV) ───────────────────
echo "[4/4] Creando trading-retrain.timer (domingos 03:00 UTC)..."
cat > /etc/systemd/system/trading-retrain.timer << TIMER
[Unit]
Description=Retrain semanal MetaLabeler — ZimaBlade
Requires=trading-retrain.service

[Timer]
# Domingo 03:00 UTC = 30 min después del update de OHLCV → datos frescos
OnCalendar=Sun *-*-* 03:00:00 UTC
Persistent=true
RandomizedDelaySec=300

[Install]
WantedBy=timers.target
TIMER
echo "      ✓ trading-retrain.timer (domingos 03:00 UTC)"

# ── Activar ───────────────────────────────────────────────────────────────────
echo ""
echo "Activando timers..."
systemctl daemon-reload

systemctl enable ohlcv-update.timer
systemctl start  ohlcv-update.timer

systemctl enable trading-retrain.timer
systemctl start  trading-retrain.timer

# ── Estado ────────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════"
echo "  Próximas ejecuciones programadas:"
echo "══════════════════════════════════════════════════════"
systemctl list-timers ohlcv-update.timer trading-retrain.timer --no-pager

echo ""
echo "══════════════════════════════════════════════════════"
echo "  ✅ Timers configurados"
echo "══════════════════════════════════════════════════════"
echo ""
echo "  Comandos útiles:"
echo ""
echo "  Ver todos los timers activos:"
echo "    systemctl list-timers"
echo ""
echo "  Forzar actualización OHLCV ahora (sin esperar las 02:30):"
echo "    sudo systemctl start ohlcv-update.service"
echo "    journalctl -u ohlcv-update -f"
echo ""
echo "  Forzar retrain ahora:"
echo "    sudo systemctl start trading-retrain.service"
echo "    journalctl -u trading-retrain -f"
echo ""
echo "  Ver logs de actualizaciones anteriores:"
echo "    journalctl -u ohlcv-update --since yesterday"
echo ""
echo "  Estado de velas en BD:"
echo "    source $VENV/bin/activate"
echo "    python scripts/download_data.py --status"
echo ""
