#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# deploy.sh — Sistema de Trading Algorítmico
# Ubuntu 24.04 LTS / Proxmox LXC (ZimaBlade)
#
# Uso:
#   chmod +x deploy.sh
#   sudo ./deploy.sh
#
# El script es idempotente: puedes ejecutarlo varias veces sin duplicar nada.
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── Colores ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*"; exit 1; }
step()    { echo -e "\n${BOLD}━━ $* ━━${RESET}"; }

# ── Comprobaciones previas ─────────────────────────────────────────────────────
[[ $EUID -ne 0 ]] && error "Ejecuta el script como root: sudo ./deploy.sh"
[[ "$(lsb_release -is 2>/dev/null)" != "Ubuntu" ]] \
    && warn "Este script está optimizado para Ubuntu 24.04. Continúa bajo tu responsabilidad."

# ── Variables configurables ────────────────────────────────────────────────────
DEPLOY_USER="${SUDO_USER:-trading}"          # Usuario sin privilegios
PROJECT_DIR="/home/${DEPLOY_USER}/trading"   # Ruta de instalación
VENV_DIR="${PROJECT_DIR}/venv"
REPO_URL="https://github.com/TU_USUARIO/Sistema_Trading.git"  # ← ajustar
DB_NAME="trading_db"
DB_USER="trading"
DB_PASS="trading_secure_$(openssl rand -hex 8)"   # contraseña aleatoria
PYTHON="python3.11"
SERVICE_NAME="trading-engine"
ML_RETRAIN_TIMER="trading-ml-retrain"

step "1/9 — Actualizando sistema e instalando dependencias"
apt-get update -qq
apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev \
    python3-pip \
    postgresql postgresql-contrib \
    git curl build-essential libpq-dev \
    htop nano ufw lsb-release ca-certificates \
    > /dev/null 2>&1
success "Dependencias del sistema instaladas"

step "2/9 — Creando usuario del sistema '${DEPLOY_USER}'"
if ! id "${DEPLOY_USER}" &>/dev/null; then
    useradd -m -s /bin/bash -G sudo "${DEPLOY_USER}"
    success "Usuario '${DEPLOY_USER}' creado"
else
    success "Usuario '${DEPLOY_USER}' ya existe"
fi

step "3/9 — Configurando PostgreSQL"
systemctl enable postgresql --now

PG_VERSION=$(pg_lsclusters -h | awk 'NR==1{print $1}')
info "Versión PostgreSQL detectada: ${PG_VERSION}"

# Crear usuario y BD si no existen
sudo -u postgres psql -tc "SELECT 1 FROM pg_roles WHERE rolname='${DB_USER}'" \
    | grep -q 1 || sudo -u postgres psql \
        -c "CREATE USER ${DB_USER} WITH PASSWORD '${DB_PASS}';"

sudo -u postgres psql -tc "SELECT 1 FROM pg_database WHERE datname='${DB_NAME}'" \
    | grep -q 1 || sudo -u postgres psql \
        -c "CREATE DATABASE ${DB_NAME} OWNER ${DB_USER};"

# Aplicar permisos y ejecutar schema
sudo -u postgres psql -d "${DB_NAME}" \
    -c "GRANT ALL PRIVILEGES ON DATABASE ${DB_NAME} TO ${DB_USER};" \
    -c "GRANT ALL ON SCHEMA public TO ${DB_USER};"

if [[ -f "${PROJECT_DIR}/data/schema.sql" ]]; then
    sudo -u postgres psql -U "${DB_USER}" -d "${DB_NAME}" \
        -f "${PROJECT_DIR}/data/schema.sql" && success "Schema SQL aplicado"
else
    warn "schema.sql no encontrado — SQLAlchemy creará las tablas al iniciar"
fi
success "PostgreSQL configurado (DB: ${DB_NAME}, User: ${DB_USER})"

step "4/9 — Clonando el repositorio"
if [[ -d "${PROJECT_DIR}/.git" ]]; then
    info "Repositorio ya existe — haciendo git pull"
    sudo -u "${DEPLOY_USER}" git -C "${PROJECT_DIR}" pull --ff-only
else
    sudo -u "${DEPLOY_USER}" git clone "${REPO_URL}" "${PROJECT_DIR}"
fi
success "Código actualizado en ${PROJECT_DIR}"

step "5/9 — Creando entorno virtual e instalando dependencias Python"
if [[ ! -d "${VENV_DIR}" ]]; then
    sudo -u "${DEPLOY_USER}" "${PYTHON}" -m venv "${VENV_DIR}"
fi
sudo -u "${DEPLOY_USER}" "${VENV_DIR}/bin/pip" install --upgrade pip -q
sudo -u "${DEPLOY_USER}" "${VENV_DIR}/bin/pip" install -r "${PROJECT_DIR}/requirements.txt" -q
success "Entorno virtual listo en ${VENV_DIR}"

step "6/9 — Configurando variables de entorno"
ENV_FILE="${PROJECT_DIR}/.env"

if [[ -f "${ENV_FILE}" ]]; then
    warn ".env ya existe — no se sobreescribe. Revisa manualmente si es necesario."
else
    cp "${PROJECT_DIR}/.env.example" "${ENV_FILE}"
    # Inyectamos la URL de PostgreSQL generada
    sed -i "s|DATABASE_URL=.*|DATABASE_URL=postgresql://${DB_USER}:${DB_PASS}@localhost:5432/${DB_NAME}|" "${ENV_FILE}"
    chown "${DEPLOY_USER}:${DEPLOY_USER}" "${ENV_FILE}"
    chmod 600 "${ENV_FILE}"

    echo ""
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
    echo -e "${YELLOW}  ACCIÓN REQUERIDA: edita el archivo .env y rellena:${RESET}"
    echo -e "${YELLOW}  - BINANCE_API_KEY / BINANCE_API_SECRET${RESET}"
    echo -e "${YELLOW}  - TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID${RESET}"
    echo -e "${YELLOW}  - TELEGRAM_ALLOWED_USER_ID  (tu Telegram numeric ID)${RESET}"
    echo -e "${YELLOW}  Archivo: ${ENV_FILE}${RESET}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
    echo ""
    read -rp "Presiona ENTER cuando hayas editado el .env para continuar... "
fi

step "7/9 — Descarga de datos históricos (BTC/USDT + ETH/USDT · 4H · 2 años)"
sudo -u "${DEPLOY_USER}" bash -c "
    cd '${PROJECT_DIR}'
    source '${VENV_DIR}/bin/activate'
    python scripts/download_data.py \
        --pairs BTC/USDT ETH/USDT \
        --timeframes 4h \
        --years 2 \
        && echo 'Datos descargados correctamente'
" || warn "Descarga de datos falló — revisa tu API key y conexión a internet"

step "8/9 — Entrenamiento inicial del modelo ML"
sudo -u "${DEPLOY_USER}" bash -c "
    cd '${PROJECT_DIR}'
    source '${VENV_DIR}/bin/activate'
    python scripts/run_ml_pipeline.py --symbol BTC/USDT --timeframe 4h \
        && echo 'Modelo ML entrenado y guardado'
" || warn "Entrenamiento ML falló — el motor arrancará sin filtro ML activo"

step "9/9 — Instalando servicios systemd"

# trading-engine.service (motor principal)
cp "${PROJECT_DIR}/systemd/trading-engine.service" /etc/systemd/system/
# Ajustar rutas y usuario en el service file
sed -i "s|/home/ale/|/home/${DEPLOY_USER}/|g"      /etc/systemd/system/trading-engine.service
sed -i "s|User=ale|User=${DEPLOY_USER}|g"           /etc/systemd/system/trading-engine.service
sed -i "s|Group=ale|Group=${DEPLOY_USER}|g"         /etc/systemd/system/trading-engine.service

# ML retrain timer
cp "${PROJECT_DIR}/systemd/trading-ml-retrain.service" /etc/systemd/system/
cp "${PROJECT_DIR}/systemd/trading-ml-retrain.timer"   /etc/systemd/system/
sed -i "s|/home/ale/|/home/${DEPLOY_USER}/|g"      /etc/systemd/system/trading-ml-retrain.service
sed -i "s|User=ale|User=${DEPLOY_USER}|g"           /etc/systemd/system/trading-ml-retrain.service

systemctl daemon-reload
systemctl enable "${SERVICE_NAME}"
systemctl enable "${ML_RETRAIN_TIMER}.timer"

# Arrancamos el motor
systemctl start "${SERVICE_NAME}" && success "Motor de trading arrancado" \
    || warn "No se pudo arrancar el motor — revisa: journalctl -u ${SERVICE_NAME}"

# ── Resumen final ──────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}"
echo "╔══════════════════════════════════════════════════════════╗"
echo "║         SISTEMA DE TRADING — DEPLOY COMPLETADO          ║"
echo "╠══════════════════════════════════════════════════════════╣"
printf "║  %-25s %-30s ║\n" "Usuario:"         "${DEPLOY_USER}"
printf "║  %-25s %-30s ║\n" "Directorio:"      "${PROJECT_DIR}"
printf "║  %-25s %-30s ║\n" "Base de datos:"   "${DB_NAME} @ localhost"
printf "║  %-25s %-30s ║\n" "Motor:"           "$(systemctl is-active ${SERVICE_NAME})"
printf "║  %-25s %-30s ║\n" "ML retrain:"      "$(systemctl is-enabled ${ML_RETRAIN_TIMER}.timer)"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Comandos útiles:                                        ║"
echo "║  journalctl -fu trading-engine     # logs en tiempo real ║"
echo "║  systemctl status trading-engine   # estado del motor    ║"
echo "║  systemctl stop  trading-engine    # parada manual       ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo -e "${RESET}"
echo -e "${YELLOW}NOTA: La contraseña de PostgreSQL generada es:${RESET}"
echo -e "      ${BOLD}${DB_PASS}${RESET}"
echo -e "${YELLOW}Está guardada en ${ENV_FILE} — guárdala en un lugar seguro.${RESET}"
