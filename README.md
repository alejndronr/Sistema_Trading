# Sistema de Trading Algorítmico de Criptomonedas

> **Fase 1: Backtesting Engine** — Motor de backtesting institucional para estrategias de trading en Binance

---

## 🚀 Inicio Rápido

### 1. Instalar dependencias

```bash
# En Windows (local)
python -m venv venv
venv\Scripts\activate

# En ZimaBlade (Ubuntu)
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configurar variables de entorno

```bash
cp .env.example .env
# Editar .env con tu capital y (para Fase 2+) tus API keys
```

### 3. Descargar datos históricos

```bash
# BTC/USDT y ETH/USDT, 2 años, timeframes 4H + 1H + 15M
python scripts/download_data.py --pairs BTC/USDT ETH/USDT --timeframes 4h 1h 15m --years 2

# Verificar datos descargados
python scripts/download_data.py --status
```

### 4. Ejecutar backtesting

```bash
# Trend Following en BTC/USDT (estrategia principal)
python scripts/run_backtest.py --symbol BTC/USDT --strategy trend_following --report

# Mean Reversion en ETH/USDT
python scripts/run_backtest.py --symbol ETH/USDT --strategy mean_reversion --report

# Multi-par con descarga automática
python scripts/run_backtest.py --all-pairs --strategy trend_following --download --report
```

---

## 📁 Estructura del Proyecto

```
Sistema_Trading/
├── config/
│   ├── settings.py          # ⚙️  Configuración central (capital, pares, parámetros)
│   └── logging_config.py    # 📋 Logs con structlog
│
├── data/
│   ├── fetcher.py           # 📥 Descarga OHLCV de Binance via ccxt
│   ├── storage.py           # 💾 SQLite (Fase 1) / PostgreSQL (Fase 2+)
│   └── cache.py             # ⚡ Cache en RAM para live trading
│
├── indicators/
│   ├── trend.py             # 📈 EMA 21/55/200, Supertrend, ADX
│   ├── momentum.py          # 💫 RSI, MACD, Stoch RSI + divergencias
│   ├── volatility.py        # 📊 ATR, Bollinger Bands, Keltner, BB Squeeze
│   ├── volume.py            # 📦 OBV, VWAP, CVD, Volume Profile
│   └── market_structure.py  # 🏗️  CHoCH, BOS, Order Blocks, FVG, Liquidity
│
├── strategies/
│   ├── base.py              # 🔧 Clase base + dataclass Signal
│   ├── trend_following.py   # ✅ Estrategia 1 — Trend Following (60%)
│   ├── mean_reversion.py    # ✅ Estrategia 2 — Mean Reversion (30%)
│   └── breakout.py          # ✅ Estrategia 3 — Breakout (10%)
│
├── risk/
│   ├── position_sizer.py    # 💰 Position sizing + circuit breakers drawdown
│   ├── regime_filter.py     # 🎯 Clasificación de régimen de mercado
│   └── psychology.py        # 🧠 Anti-revenge, anti-FOMO, anti-overtrading
│
├── backtesting/
│   ├── engine.py            # 🏎️  Motor event-driven a nivel de vela
│   ├── portfolio.py         # 📁 Portfolio simulado (comisiones + slippage)
│   ├── metrics.py           # 📊 Win Rate, PF, Sharpe, Drawdown, etc.
│   └── report.py            # 📄 Informe HTML interactivo con Plotly
│
├── journal/
│   └── trade_logger.py      # 📝 Journal SQLite + CSV
│
├── scripts/
│   ├── download_data.py     # CLI: descargar datos históricos
│   └── run_backtest.py      # CLI: ejecutar backtesting
│
└── tests/
    ├── test_indicators.py
    └── test_backtesting.py
```

---

## 📊 Estrategias Implementadas

### Estrategia 1: Trend Following (60% del tiempo)

| Condición | Valor |
|-----------|-------|
| EMA alignment en 4H | EMA21 > EMA55 > EMA200 |
| Precio | Sobre EMA 21 en 1H |
| ADX | > 25 (tendencia fuerte) |
| RSI | Entre 45-65 (zona neutra) |
| MACD | Positivo y creciente |
| Entrada | Retroceso a EMA21 u Order Block |
| Stop Loss | 1 ATR bajo swing low reciente |
| TP1 | 2:1 R/R (cerrar 50%) |
| TP2 | 3:1 R/R (trailing stop) |

### Estrategia 2: Mean Reversion (30% del tiempo)

| Condición | Valor |
|-----------|-------|
| Régimen | ADX < 20, mercado en rango |
| RSI | < 35 (long) o > 65 (short) |
| Precio | Tocando banda inferior/superior BB |
| Confirmación | Soporte/resistencia + Divergencia Stoch RSI |
| Stop Loss | Fuera de BB + 0.5 ATR |
| TP1 | EMA 21 |
| TP2 | Banda opuesta BB |

### Estrategia 3: Breakout (10% del tiempo)

| Condición | Valor |
|-----------|-------|
| Prerequisito | BB Squeeze activo > 10 velas |
| Volumen | > 150% del promedio de 20 periodos |
| Entrada | Retesteo del nivel roto |

---

## 💰 Gestión de Riesgo

```
Capital inicial: 300€ → Riesgo fijo $10/trade

Escala de compounding:
  $0   - $1,000  → $10 fijo (aprendizaje)
  $1K  - $5K     → 1% por trade
  $5K  - $20K    → 1.5% por trade
  $20K+          → 2% por trade (máximo)

Circuit Breakers (automáticos):
  > 3% drawdown diario  → Parar el día
  > 6% drawdown semanal → Revisar sistema
  > 10% drawdown mes    → Modo solo-estudio
```

---

## 🎯 Objetivos de Fase 1 (Backtesting)

| Métrica | Objetivo | Descripción |
|---------|----------|-------------|
| Win Rate | ≥ 45% | Al menos 45 de cada 100 trades ganadores |
| Profit Factor | ≥ 1.5 | $1.50 ganados por cada $1 perdido |
| Sharpe Ratio | ≥ 1.0 | Retorno ajustado por riesgo aceptable |
| Max Drawdown | ≤ 15% | Máxima caída desde el pico |

---

## 🧪 Tests

```bash
# Ejecutar todos los tests
pytest tests/ -v

# Con cobertura
pytest tests/ --cov=. --cov-report=html
```

---

## 🔄 Fases del Sistema

| Fase | Estado | Descripción |
|------|--------|-------------|
| **Fase 1** | ✅ **ACTIVA** | Backtesting con 2 años de datos |
| **Fase 2** | ⏳ Pendiente | Paper Trading en tiempo real |
| **Fase 3** | ⏳ Pendiente | Live Trading Micro ($100-$500) |
| **Fase 4** | ⏳ Pendiente | Scaling con compounding |

---

## 📋 Activos del Universo

| Prioridad | Pares | Condición |
|-----------|-------|-----------|
| P1 — Alta liquidez | BTC/USDT, ETH/USDT | Siempre disponibles |
| P2 — Media | SOL/USDT, BNB/USDT, AVAX/USDT | Volatilidad media |
| P3 — Baja (solo tendencia) | LINK/USDT, DOT/USDT, MATIC/USDT | Solo ADX > 25 |

---

## ⚙️ Infraestructura (ZimaBlade)

- **OS**: Ubuntu Server 24.04 LTS (LXC en Proxmox)
- **BD Fase 1**: SQLite con WAL mode + 64MB cache RAM
- **BD Fase 2+**: PostgreSQL
- **Monitoring**: Grafana + InfluxDB
- **Alertas**: Telegram Bot (Fase 2+)
- **Scheduler**: APScheduler / systemd timer

---

## 📝 Notas Legales (España)

Las ganancias por trading de criptomonedas tributan como **ganancias patrimoniales (IRPF)**.
El journal CSV es compatible con **Koinly** y **CoinTracking** para el cálculo fiscal.
Ver Modelo 720 si activos en exchange extranjero > €50,000.

---

*Sistema de Trading Algorítmico v1.0 — Fase 1: Backtesting*
