#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# install_dashboard.sh — Instala el dashboard en tu ZimaBlade (Debian 12 LXC)
# ══════════════════════════════════════════════════════════════════════════════
# Uso:
#   chmod +x install_dashboard.sh
#   ./install_dashboard.sh
#
# Requisitos: sistema_trading ya instalado en /home/trading/sistema_trading
#             PostgreSQL activo y accesible en 127.0.0.1:5432
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

TRADING_DIR="/home/trading/sistema_trading"
VENV="$TRADING_DIR/venv"
DASH_DIR="$TRADING_DIR/dashboard"
SERVICE_NAME="trading-dashboard"
PORT=8501

echo "══════════════════════════════════════════"
echo "  ZimaBlade Trading HQ — Dashboard Setup"
echo "══════════════════════════════════════════"

# ── 1. Verificar que el directorio del proyecto existe ─────────────────────
if [ ! -d "$TRADING_DIR" ]; then
    echo "[ERROR] No se encuentra $TRADING_DIR"
    echo "        Asegúrate de que el sistema_trading está instalado."
    exit 1
fi

# ── 2. Activar el virtualenv y añadir dependencias del dashboard ───────────
echo "[1/5] Instalando dependencias del dashboard..."
source "$VENV/bin/activate"

pip install \
    streamlit>=1.35.0 \
    plotly>=5.20.0 \
    requests>=2.31.0 \
    --quiet

echo "      ✓ streamlit, plotly, requests instalados"

# ── 3. Copiar app.py al directorio dashboard ──────────────────────────────
echo "[2/5] Desplegando dashboard/app.py..."
mkdir -p "$DASH_DIR"
cp "$(dirname "$0")/app.py" "$DASH_DIR/app.py"
echo "      ✓ $DASH_DIR/app.py creado"

# ── 4. Crear configuración de Streamlit ───────────────────────────────────
echo "[3/5] Configurando Streamlit..."
mkdir -p "$DASH_DIR/.streamlit"
cat > "$DASH_DIR/.streamlit/config.toml" << 'TOML'
[server]
port            = 8501
address         = "0.0.0.0"
headless        = true
enableCORS      = false
enableXsrfProtection = false
maxUploadSize   = 10

[theme]
primaryColor    = "#1D9E75"
backgroundColor = "#0F1117"
secondaryBackgroundColor = "#1A1F2E"
textColor       = "#FAFAFA"
font            = "sans serif"

[browser]
gatherUsageStats = false
TOML
echo "      ✓ Tema oscuro configurado en .streamlit/config.toml"

# ── 5. Crear servicio systemd ─────────────────────────────────────────────
echo "[4/5] Creando servicio systemd..."
sudo tee /etc/systemd/system/${SERVICE_NAME}.service > /dev/null << SERVICE
[Unit]
Description=ZimaBlade Trading HQ Dashboard
Documentation=https://github.com/alejndronr/Sistema_Trading
After=network.target postgresql.service

[Service]
Type=simple
User=trading
WorkingDirectory=${TRADING_DIR}/dashboard
EnvironmentFile=${TRADING_DIR}/.env
Environment=PATH=${VENV}/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=${VENV}/bin/streamlit run app.py \
    --server.port ${PORT} \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=trading-dashboard

[Install]
WantedBy=multi-user.target
SERVICE

echo "      ✓ /etc/systemd/system/${SERVICE_NAME}.service creado"

# ── 6. Activar y arrancar el servicio ─────────────────────────────────────
echo "[5/5] Activando y arrancando el dashboard..."
sudo systemctl daemon-reload
sudo systemctl enable "${SERVICE_NAME}"
sudo systemctl restart "${SERVICE_NAME}"

sleep 3

if systemctl is-active --quiet "${SERVICE_NAME}"; then
    echo ""
    echo "══════════════════════════════════════════"
    echo "  ✅ Dashboard desplegado con éxito"
    echo "══════════════════════════════════════════"
    # Obtener IP del contenedor
    IP=$(hostname -I | awk '{print $1}')
    echo "  🌐 URL: http://${IP}:${PORT}"
    echo "  📊 Abre esa URL en tu navegador"
    echo ""
    echo "  Comandos útiles:"
    echo "    journalctl -u ${SERVICE_NAME} -f    # logs en tiempo real"
    echo "    systemctl status ${SERVICE_NAME}     # estado del servicio"
    echo "    systemctl restart ${SERVICE_NAME}    # reiniciar"
    echo "══════════════════════════════════════════"
else
    echo ""
    echo "  ❌ El servicio no arrancó correctamente."
    echo "  Revisa los logs:"
    echo "    journalctl -u ${SERVICE_NAME} -n 50 --no-pager"
    exit 1
fi
