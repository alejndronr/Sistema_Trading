# Sistema de Trading Algorítmico de Criptomonedas (ZimaBlade v6.0 - Adaptive Cycle)

Este repositorio contiene la versión consolidada, optimizada y auditada del sistema de trading algorítmico cuantitativo para criptomonedas, desplegado 24/7 en un servidor **ZimaBlade** (Intel Atom E3950, 16GB RAM, HDD mecánico, Debian 12 LXC sobre Proxmox).

Filosofía del sistema: **"Preservar capital primero. Crecer segundo."**

> ⚠️ **IMPORTANTE:** Antes de usar este sistema, leer [DISCLAIMER.md](DISCLAIMER.md).
> El sistema está en validación activa. No existe evidencia out-of-sample suficiente de edge positivo.
> No aumentar capital ni activar MARGIN_ENABLED sin completar la validación documentada en el DISCLAIMER.

---

## 🚀 Inicio Rápido e Instalación

### 1. Clonar e Instalar Dependencias

```bash
# En Windows (desarrollo local)
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# En ZimaBlade (Debian 12 LXC)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configurar Variables de Entorno

Copia el archivo de ejemplo y rellena los valores reales (especialmente API Keys de Binance y Tokens del Bot de Telegram):

```bash
cp .env.example .env
```

### 3. Despliegue Automatizado en ZimaBlade

Para instalar y arrancar los servicios en producción, ejecuta el script de despliegue (`deploy.sh`), el cual configura la base de datos PostgreSQL local en `127.0.0.1`, restaura el esquema y crea los servicios de `systemd`:

```bash
bash deploy.sh
```

---

## 📁 Estructura del Proyecto Consolidado

Tras la auditoría y limpieza física, el proyecto mantiene una arquitectura modular estructurada en paquetes limpios:

```
Sistema_Trading/
├── config/                  # ⚙️ Configuración central y logs
│   ├── settings.py          # Parámetros del sistema, Pydantic BaseSettings, límites de riesgo
│   └── logging_config.py    # Log estructurado con structlog
│
├── data/                    # 📥 Capa de adquisición y persistencia
│   ├── fetcher.py           # Descarga asíncrona de datos OHLCV de Binance via CCXT
│   ├── storage.py           # Interfaz de persistencia (PostgreSQL/SQLite)
│   ├── cache.py             # Caché en memoria RAM para live trading
│   └── schema.sql           # Esquema de base de datos para PostgreSQL
│
├── indicators/              # 📈 Cálculo optimizado de indicadores
│   └── technical.py         # Batería consolidada de indicadores (SMC, VFI, Consensus, V5)
│
├── strategies/              # 🎯 Estrategias cuantitativas
│   └── signals.py           # Lógica consolidada de señales (TrendFollowing, MeanReversion, Breakout)
│
├── risk/                    # 🛡️ Gestión de riesgo y régimen de mercado
│   ├── position_sizer.py    # Control del tamaño de posición y drawdown (circuit breakers)
│   └── regime_filter.py     # Detección del ciclo macro y clasificación de régimen
│
├── backtesting/             # 🏎️ Motor de simulación histórica
│   ├── engine.py            # Motor event-driven a nivel de vela, portafolio y optimizador
│   └── metrics.py           # Cálculo de métricas (Sharpe, Profit Factor, Win Rate, Drawdowns)
│
├── ml/                      # 🧠 Capa de Machine Learning (Meta-Labeling)
│   ├── meta_labeler.py      # Filtro RandomForest para clasificar probabilidad de señal ganadora
│   └── retrain_model.py     # Script mensual autónomo de reentrenamiento (compatible con Win/Linux)
│
├── monitoring/              # 🔔 Monitoreo e interfaces de control
│   └── telegram_bot.py      # Bot interactivo de Telegram para control y notificaciones
│
├── scripts/                 # 🛠️ Herramientas CLI de ejecución
│   ├── download_data.py     # Descarga de históricos de datos
│   ├── run_backtest.py      # Ejecución de backtests históricos
│   └── run_optimization.py  # Optimización de hiperparámetros
│
├── tests/                   # 🧪 Pruebas unitarias
│   ├── test_indicators.py   # Validación matemática de indicadores
│   └── test_backtesting.py  # Validación del motor y métricas de backtesting
│
├── live_engine.py           # 🚀 Motor principal de ejecución productiva (Adaptive Cycle)
├── paper_portfolio.py       # 📄 Portafolio virtual en producción para paper trading
├── deploy.sh                # 📦 Script de despliegue automatizado para ZimaBlade
└── requirements.txt         # Dependencias del proyecto
```

---

## ⚙️ Modos de Operación

### 1. Ejecución del Bot en Producción (ZimaBlade)

El bot corre como un servicio daemonizado gestionado por `systemd`. Para iniciarlo de manera manual:

```bash
# Modo Paper Trading (Valores por defecto de PAPER_MODE=true en .env)
python live_engine.py

# Modificar modo de operación mediante variables de entorno
PAPER_MODE=true python live_engine.py
```

### 2. Backtesting Histórico

Puedes simular el rendimiento de una estrategia con datos históricos previamente descargados:

```bash
# Ejecutar Backtest con estrategia de seguimiento de tendencia en BTC/USDC
python scripts/run_backtest.py --symbol BTC/USDC --strategy trend_following --report

# Ejecutar Backtest multi-par con descarga automática de datos omitidos
python scripts/run_backtest.py --all-pairs --strategy trend_following --download --report
```

### 3. Optimización de Hiperparámetros

Ejecuta búsquedas en cuadrícula para encontrar los mejores parámetros de indicadores y stop loss:

```bash
python scripts/run_optimization.py
```

### 4. Reentrenamiento del Modelo ML

Para reentrenar de forma manual el clasificador de señales (Meta-Labeler):

```bash
python ml/retrain_model.py
```

---

## 🛡️ Parámetros Críticos y Gestión de Riesgo

El sistema implementa reglas estrictas que preservan el capital:

*   **Riesgo por Trade:** Máximo 1% del capital acumulado por operación (compounding dinámico). En cuentas pequeñas (< $1000) se puede fijar una cantidad fija (ej. $10).
*   **Stop Loss Estricto:** Mínimo de 1.5 ATR. Ningún trade se ejecuta sin SL.
*   **Límite de Exposición:** Máximo de 3 posiciones abiertas simultáneamente en toda la cartera.
*   **Circuit Breakers (Drawdown Tracker):**
    *   Si el drawdown diario supera el **3%**, el sistema pausa las operaciones del día.
    *   Si el drawdown semanal supera el **6%**, el sistema se bloquea para revisión.
    *   Si el drawdown mensual supera el **10%**, el bot entra en modo de solo estudio de mercado.

---

## 🧪 Verificación y Control de Calidad

El proyecto incluye un script de auditoría y verificación integral que valida que ningún cambio afecte el flujo del sistema:

1.  **Syntax Check:** Validación mediante `ast.parse` de cada archivo `.py`.
2.  **Import Resolution Check:** Asegura que todos los módulos se importan correctamente.
3.  **Unit Tests:** Ejecución completa de `pytest`.
4.  **Signal Pipeline Test:** Generación de al menos 3 señales con datos sintéticos.

Para ejecutar los tests unitarios tradicionales de forma individual:

```bash
python -m pytest tests/ -v
```

---

*Sistema de Trading Algorítmico v6.0 — Consolidado y Listo para Producción 24/7*
