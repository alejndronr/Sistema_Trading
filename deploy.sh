#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# deploy.sh — Instalación completa del Sistema de Trading en Ubuntu 24.04
# ZimaBlade / Proxmox LXC
#
# Uso: sudo bash deploy.sh
# El script es idempotente: se puede ejecutar varias veces sin duplicar nada.
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail
IFS=$'\n\t'

# ── Colores (sin dependencias externas) ───────────────────────────────────────
RED='\e[31m'; GREEN='\e[32m'; YELLOW='\e[33m'; CYAN='\e[36m'
BOLD='\e[1m'; RESET='\e[0m'

ok()   { echo -e "${GREEN}[OK]${RESET}   $*"; }
fail() { echo -e "${RED}[FAIL]${RESET} $*"; }
warn() { echo -e "${YELLOW}[WARN]${RESET} $*"; }
info() { echo -e "${CYAN}[INFO]${RESET} $*"; }
step() { echo -e "\n${BOLD}${CYAN}━━ PASO $* ━━${RESET}"; }

# ── Registro del estado de cada componente para el panel final ─────────────────
declare -A STATUS
STATUS[python_venv]="PENDING"
STATUS[postgresql]="PENDING"
STATUS[datos_historicos]="PENDING"
STATUS[modelo_ml]="PENDING"
STATUS[servicio_engine]="PENDING"
STATUS[telegram_bot]="PENDING"

# ── Variables configurables ────────────────────────────────────────────────────
TRADING_USER="trading"
INSTALL_DIR="/home/${TRADING_USER}/sistema_trading"
VENV_DIR="${INSTALL_DIR}/venv"
PYTHON="python3.11"
DB_NAME="trading_db"
DB_USER="trading_user"
DB_PASS="$(openssl rand -base64 16 | tr -d '/+=' | head -c 24)"
ENGINE_SERVICE="trading-engine"
RETRAIN_TIMER="trading-retrain"
REPO_SRC="$(pwd)"   # directorio desde donde se lanza el script = repo clonado

# ══════════════════════════════════════════════════════════════════════════════
# PASO 1 — Validaciones previas
# ══════════════════════════════════════════════════════════════════════════════
step "1/9 — Validaciones previas"

# Root
[[ $EUID -ne 0 ]] && { fail "Ejecuta con sudo: sudo bash deploy.sh"; exit 1; }
ok "Ejecutando como root"

# Ubuntu (no bloquear, solo advertir)
DISTRO=$(lsb_release -is 2>/dev/null || echo "Unknown")
RELEASE=$(lsb_release -rs 2>/dev/null || echo "0")
if [[ "$DISTRO" != "Ubuntu" ]]; then
    warn "Distro no es Ubuntu ($DISTRO). Continúa bajo tu responsabilidad."
elif [[ "$RELEASE" != "24.04" ]]; then
    warn "Ubuntu $RELEASE detectado (recomendado 24.04). Continuando..."
else
    ok "Ubuntu 24.04 detectado"
fi

# Internet
if ping -c1 -W3 8.8.8.8 &>/dev/null; then
    ok "Conexión a internet disponible"
else
    fail "Sin conexión a internet. Verifica la red."
    exit 1
fi

# Repo presente
if [[ ! -f "${REPO_SRC}/live_engine.py" ]]; then
    fail "No se encontró live_engine.py en $(pwd). Ejecuta deploy.sh desde el directorio del repo."
    exit 1
fi
ok "Repositorio encontrado en ${REPO_SRC}"

# ══════════════════════════════════════════════════════════════════════════════
# PASO 2 — Dependencias del sistema
# ══════════════════════════════════════════════════════════════════════════════
step "2/9 — Dependencias del sistema"

apt-get update -qq
apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev \
    postgresql postgresql-contrib libpq-dev \
    git curl build-essential lsb-release ca-certificates \
    nano htop > /dev/null 2>&1

ok "Python 3.11, PostgreSQL, git y herramientas instalados"

# ══════════════════════════════════════════════════════════════════════════════
# PASO 3 — Crear usuario sin privilegios
# ══════════════════════════════════════════════════════════════════════════════
step "3/9 — Usuario del sistema '${TRADING_USER}'"

if id "${TRADING_USER}" &>/dev/null; then
    info "Usuario '${TRADING_USER}' ya existe"
else
    useradd -m -s /bin/bash "${TRADING_USER}"
    ok "Usuario '${TRADING_USER}' creado"
fi

# Copiar repo al directorio del usuario (si no está ya allí)
if [[ "${REPO_SRC}" != "${INSTALL_DIR}" ]]; then
    mkdir -p "${INSTALL_DIR}"
    cp -r "${REPO_SRC}/." "${INSTALL_DIR}/"
    chown -R "${TRADING_USER}:${TRADING_USER}" "${INSTALL_DIR}"
    ok "Repo copiado a ${INSTALL_DIR}"
else
    info "El repo ya está en ${INSTALL_DIR}"
fi

# ══════════════════════════════════════════════════════════════════════════════
# PASO 4 — Configurar PostgreSQL
# ══════════════════════════════════════════════════════════════════════════════
step "4/9 — PostgreSQL"

systemctl enable postgresql --now 2>/dev/null || true
sleep 2   # Esperar a que PostgreSQL arranque

# Crear usuario
sudo -u postgres psql -tc "SELECT 1 FROM pg_roles WHERE rolname='${DB_USER}'" | grep -q 1 || \
    sudo -u postgres psql -c "CREATE USER ${DB_USER} WITH PASSWORD '${DB_PASS}';" > /dev/null

# Crear BD
sudo -u postgres psql -tc "SELECT 1 FROM pg_database WHERE datname='${DB_NAME}'" | grep -q 1 || \
    sudo -u postgres psql -c "CREATE DATABASE ${DB_NAME} OWNER ${DB_USER};" > /dev/null

# Permisos
sudo -u postgres psql -d "${DB_NAME}" \
    -c "GRANT ALL PRIVILEGES ON DATABASE ${DB_NAME} TO ${DB_USER}; GRANT ALL ON SCHEMA public TO ${DB_USER};" \
    > /dev/null

# Aplicar schema desde heredoc (todas las tablas del ARCHIVO 6)
sudo -u postgres psql -U "${DB_USER}" -d "${DB_NAME}" > /dev/null <<'SQL_EOF'
CREATE TABLE IF NOT EXISTS positions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    strategy VARCHAR(30) NOT NULL,
    direction VARCHAR(10) DEFAULT 'long',
    entry_time TIMESTAMPTZ NOT NULL,
    entry_price DECIMAL(18,8) NOT NULL,
    stop_loss DECIMAL(18,8) NOT NULL,
    tp1 DECIMAL(18,8) NOT NULL,
    tp2 DECIMAL(18,8) NOT NULL,
    units DECIMAL(18,8) NOT NULL,
    risk_amount DECIMAL(10,4) NOT NULL,
    tp1_hit BOOLEAN DEFAULT FALSE,
    remaining_units DECIMAL(18,8) NOT NULL,
    binance_order_id VARCHAR(50),
    ml_proba DECIMAL(5,4),
    status VARCHAR(20) DEFAULT 'open'
);
CREATE TABLE IF NOT EXISTS trades_journal (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    strategy VARCHAR(30) NOT NULL,
    entry_time TIMESTAMPTZ NOT NULL,
    exit_time TIMESTAMPTZ,
    entry_price DECIMAL(18,8),
    exit_price DECIMAL(18,8),
    stop_loss DECIMAL(18,8),
    tp1 DECIMAL(18,8),
    tp2 DECIMAL(18,8),
    units DECIMAL(18,8),
    pnl DECIMAL(10,4),
    pnl_pct DECIMAL(8,6),
    r_multiple DECIMAL(6,3),
    exit_reason VARCHAR(30),
    ml_proba DECIMAL(5,4),
    regime VARCHAR(20),
    commission_paid DECIMAL(10,6)
);
CREATE TABLE IF NOT EXISTS portfolio_state (
    id INTEGER PRIMARY KEY DEFAULT 1,
    current_capital DECIMAL(12,4) NOT NULL,
    daily_start DECIMAL(12,4) NOT NULL,
    weekly_start DECIMAL(12,4) NOT NULL,
    monthly_start DECIMAL(12,4) NOT NULL,
    peak_capital DECIMAL(12,4) NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS system_heartbeat (
    id INTEGER PRIMARY KEY DEFAULT 1,
    last_ping TIMESTAMPTZ DEFAULT NOW(),
    engine_version VARCHAR(20),
    paper_mode BOOLEAN
);
CREATE TABLE IF NOT EXISTS ohlcv (
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    timestamp BIGINT NOT NULL,
    open DECIMAL(18,8), high DECIMAL(18,8),
    low DECIMAL(18,8), close DECIMAL(18,8),
    volume DECIMAL(20,4),
    PRIMARY KEY (symbol, timeframe, timestamp)
);
CREATE INDEX IF NOT EXISTS idx_ohlcv_ts ON ohlcv(symbol, timeframe, timestamp);
SQL_EOF

DATABASE_URL="postgresql://${DB_USER}:${DB_PASS}@127.0.0.1:5432/${DB_NAME}"
ok "PostgreSQL configurado — DATABASE_URL inyectada"
STATUS[postgresql]="OK"

# ══════════════════════════════════════════════════════════════════════════════
# PASO 5 — Entorno virtual Python
# ══════════════════════════════════════════════════════════════════════════════
step "5/9 — Entorno virtual Python 3.11"

if [[ ! -d "${VENV_DIR}" ]]; then
    sudo -u "${TRADING_USER}" "${PYTHON}" -m venv "${VENV_DIR}"
fi

sudo -u "${TRADING_USER}" "${VENV_DIR}/bin/pip" install --upgrade pip -q
sudo -u "${TRADING_USER}" "${VENV_DIR}/bin/pip" install \
    -r "${INSTALL_DIR}/requirements.txt" -q

info "Verificando librerías opcionales..."

sudo -u "${TRADING_USER}" "${VENV_DIR}/bin/python" -c "import smartmoneyconcepts; print('[OK] smart-money-concepts')" \
    2>/dev/null || warn "smart-money-concepts no disponible — usando fallback custom"

sudo -u "${TRADING_USER}" "${VENV_DIR}/bin/python" -c "import mplfinance; print('[OK] mplfinance — charts habilitados')" \
    2>/dev/null || warn "mplfinance no disponible — charts Telegram desactivados"

# Test de rendimiento en Atom E3950 (verificar que el RF no es demasiado lento)
info "Ejecutando test de rendimiento RandomForest en virtualenv..."
sudo -u "${TRADING_USER}" "${VENV_DIR}/bin/python" -c "
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
X = np.random.rand(500, 21).astype(np.float32)
y = (X[:, 0] > 0.5).astype(int)
rf = RandomForestClassifier(n_estimators=100, max_depth=6, n_jobs=2, random_state=42)
t0 = time.time()
rf.fit(X, y)
elapsed = time.time() - t0
if elapsed < 30:
    print(f'[OK] RF training: {elapsed:.1f}s (dentro del límite)')
else:
    print(f'[WARN] RF training: {elapsed:.1f}s (muy lento — reducir n_estimators a 50)')
" 2>/dev/null || warn "No se pudo ejecutar el test de rendimiento RF"

if "${VENV_DIR}/bin/python" -c "import ccxt, pandas, sklearn" 2>/dev/null; then
    ok "Entorno virtual listo con todas las dependencias"
    STATUS[python_venv]="OK"
else
    fail "Faltan dependencias en el venv"
    STATUS[python_venv]="FAIL"
fi

# ══════════════════════════════════════════════════════════════════════════════
# PASO 6 — Configurar .env
# ══════════════════════════════════════════════════════════════════════════════
step "6/9 — Configuración de variables de entorno"

ENV_FILE="${INSTALL_DIR}/.env"

if [[ -f "${ENV_FILE}" ]]; then
    warn ".env ya existe — actualizando solo DATABASE_URL"
    sed -i "s|^DATABASE_URL=.*|DATABASE_URL=${DATABASE_URL}|" "${ENV_FILE}"
else
    cp "${INSTALL_DIR}/.env.example" "${ENV_FILE}"
    sed -i "s|DATABASE_URL=.*|DATABASE_URL=${DATABASE_URL}|" "${ENV_FILE}"
    chown "${TRADING_USER}:${TRADING_USER}" "${ENV_FILE}"
    chmod 600 "${ENV_FILE}"
fi

echo ""
echo -e "${YELLOW}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${YELLOW}  ACCIÓN REQUERIDA — Edita el archivo .env con tus credenciales:${RESET}"
echo -e "${YELLOW}  Archivo: ${ENV_FILE}${RESET}"
echo ""
echo -e "${YELLOW}  Variables que DEBES rellenar manualmente:${RESET}"
echo -e "${YELLOW}    BINANCE_API_KEY       — tu API key de Binance${RESET}"
echo -e "${YELLOW}    BINANCE_API_SECRET    — tu API secret de Binance${RESET}"
echo -e "${YELLOW}    TELEGRAM_TOKEN        — token del bot de @BotFather${RESET}"
echo -e "${YELLOW}    TELEGRAM_CHAT_ID      — tu ID numérico de Telegram${RESET}"
echo -e "${YELLOW}    TELEGRAM_ALLOWED_USER_ID — igual que TELEGRAM_CHAT_ID${RESET}"
echo -e "${YELLOW}    PAPER_MODE=true       — mantener true las primeras 4 semanas${RESET}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""
read -rp "Presiona ENTER cuando hayas guardado el .env para continuar..."

# ══════════════════════════════════════════════════════════════════════════════
# PASO 7 — Descarga de datos históricos (timeout 10 min)
# ══════════════════════════════════════════════════════════════════════════════
step "7/9 — Descarga de datos históricos (BTC/USDT + ETH/USDT · 4H · 2 años)"

CANDLES_OK=false
if timeout 600 sudo -u "${TRADING_USER}" bash -c "
    cd '${INSTALL_DIR}'
    source '${VENV_DIR}/bin/activate'
    python scripts/download_data.py \
        --pairs 'BTC/USDT' 'ETH/USDT' \
        --timeframes 4h \
        --years 2
" 2>&1; then
    ok "Datos históricos descargados"
    STATUS[datos_historicos]="OK"
    CANDLES_OK=true
else
    warn "Descarga de datos falló o tardó más de 10 min — el motor arrancará sin datos OHLCV"
    warn "Puedes reintentarlo con: python scripts/download_data.py --pairs BTC/USDT --timeframes 4h --years 2"
    STATUS[datos_historicos]="WARN"
fi

# ══════════════════════════════════════════════════════════════════════════════
# PASO 8 — Entrenamiento inicial del modelo ML (timeout 5 min)
# ══════════════════════════════════════════════════════════════════════════════
step "8/9 — Entrenamiento inicial del modelo ML"

if [[ "${CANDLES_OK}" == "true" ]]; then
    if timeout 300 sudo -u "${TRADING_USER}" bash -c "
        cd '${INSTALL_DIR}'
        source '${VENV_DIR}/bin/activate'
        python scripts/run_backtest.py --symbol 'BTC/USDT' --strategy trend_following --save-trades 2>&1
        python ml/retrain_model.py --initial 2>&1
    "; then
        ok "Modelo ML entrenado y guardado en ml/model.joblib"
        STATUS[modelo_ml]="OK"
    else
        warn "Entrenamiento ML falló — el motor arrancará con proba=0.5 (modo permisivo)"
        STATUS[modelo_ml]="WARN"
    fi
else
    warn "Entrenamiento ML omitido (sin datos históricos)"
    STATUS[modelo_ml]="SKIP"
fi

# ══════════════════════════════════════════════════════════════════════════════
# PASO 9 — Instalar servicios systemd
# ══════════════════════════════════════════════════════════════════════════════
step "9/9 — Servicios systemd"

# Ajustar rutas en los service files
for svc_file in trading-engine.service trading-retrain.service; do
    if [[ -f "${INSTALL_DIR}/systemd/${svc_file}" ]]; then
        sed -i \
            -e "s|/home/trading/sistema_trading|${INSTALL_DIR}|g" \
            -e "s|User=trading|User=${TRADING_USER}|g" \
            -e "s|Group=trading|Group=${TRADING_USER}|g" \
            "${INSTALL_DIR}/systemd/${svc_file}"
        cp "${INSTALL_DIR}/systemd/${svc_file}" "/etc/systemd/system/"
    fi
done

if [[ -f "${INSTALL_DIR}/systemd/trading-retrain.timer" ]]; then
    cp "${INSTALL_DIR}/systemd/trading-retrain.timer" "/etc/systemd/system/"
fi

systemctl daemon-reload
systemctl enable "${ENGINE_SERVICE}" 2>/dev/null
systemctl enable "${RETRAIN_TIMER}.timer" 2>/dev/null
systemctl start "${ENGINE_SERVICE}" || true

sleep 10
if systemctl is-active --quiet "${ENGINE_SERVICE}"; then
    ok "Servicio ${ENGINE_SERVICE} activo"
    STATUS[servicio_engine]="OK"
else
    fail "Servicio ${ENGINE_SERVICE} no arrancó"
    STATUS[servicio_engine]="FAIL"
    warn "Revisa los logs: journalctl -fu ${ENGINE_SERVICE}"
fi

# Test básico de Telegram (curl al bot)
TELEGRAM_TOKEN_CHECK=$(grep -oP '(?<=^TELEGRAM_TOKEN=).+' "${ENV_FILE}" 2>/dev/null || true)
if [[ -n "${TELEGRAM_TOKEN_CHECK}" && "${TELEGRAM_TOKEN_CHECK}" != "tu_bot_token_de_botfather" ]]; then
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
        "https://api.telegram.org/bot${TELEGRAM_TOKEN_CHECK}/getMe" 2>/dev/null || echo "000")
    if [[ "${HTTP_CODE}" == "200" ]]; then
        ok "Telegram bot responde"
        STATUS[telegram_bot]="OK"
    else
        warn "Telegram bot no responde (HTTP ${HTTP_CODE}) — verifica el token"
        STATUS[telegram_bot]="WARN"
    fi
else
    warn "TELEGRAM_TOKEN no configurado"
    STATUS[telegram_bot]="SKIP"
fi

# ══════════════════════════════════════════════════════════════════════════════
# PANEL FINAL
# ══════════════════════════════════════════════════════════════════════════════
print_status() {
    local label="$1"
    local key="$2"
    case "${STATUS[$key]}" in
        OK)      echo -e "  ${GREEN}[OK]${RESET}   ${label}" ;;
        FAIL)    echo -e "  ${RED}[FAIL]${RESET} ${label}" ;;
        WARN)    echo -e "  ${YELLOW}[WARN]${RESET} ${label}" ;;
        SKIP)    echo -e "  ${CYAN}[SKIP]${RESET} ${label}" ;;
        PENDING) echo -e "  [????] ${label}" ;;
    esac
}

echo ""
echo -e "${BOLD}${GREEN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║      SISTEMA DE TRADING — DEPLOY COMPLETADO                 ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo -e "${RESET}"

print_status "Python 3.11 venv + dependencias"    python_venv
print_status "PostgreSQL conectado"               postgresql
print_status "Datos históricos descargados"        datos_historicos
print_status "Modelo ML entrenado"                 modelo_ml
print_status "Servicio trading-engine activo"      servicio_engine
print_status "Telegram bot respondiendo"           telegram_bot

echo ""
echo -e "${BOLD}Contraseña PostgreSQL generada (guárdala en un lugar seguro):${RESET}"
echo -e "  ${YELLOW}${DB_PASS}${RESET}"
echo ""
echo -e "${BOLD}Comandos de monitorización:${RESET}"
echo "  journalctl -fu trading-engine          # logs en tiempo real"
echo "  systemctl status trading-engine        # estado del servicio"
echo "  systemctl list-timers | grep retrain   # próximo reentrenamiento"
echo ""
echo -e "${BOLD}${CYAN}Para monitorizar desde tu móvil, usa Telegram:${RESET}"
echo "  /status   → estado del portfolio"
echo "  /pause    → pausar nuevas entradas"
echo "  /kill     → apagado de emergencia"
echo ""
echo -e "${GREEN}${BOLD}¡Sistema listo! Paper Trading iniciado.${RESET}"
echo ""
