"""
Sistema de Trading — Configuración Central
=========================================
Todos los parámetros del sistema están definidos aquí.
Para ajustar el comportamiento del bot, modifica esta clase.
NUNCA hardcodees parámetros en otros módulos — importa siempre desde aquí.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# ── Rutas del proyecto ─────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data" / "db"
LOGS_DIR = ROOT_DIR / "logs"
REPORTS_DIR = ROOT_DIR / "reports" / "output"

# Crear directorios si no existen
for _dir in [DATA_DIR, LOGS_DIR, REPORTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)


# ── Enums ──────────────────────────────────────────────────────────────────────

class AssetPriority(Enum):
    """Clasificación de activos por liquidez y riesgo."""
    HIGH = 1       # BTC/USDT, ETH/USDT
    MEDIUM = 2     # SOL/USDT, BNB/USDT, AVAX/USDT
    LOW = 3        # LINK/USDT, DOT/USDT, MATIC/USDT


class MarketRegime(Enum):
    """Régimen de mercado detectado por el filtro de régimen."""
    BULLISH_TREND = "bullish_trend"
    BEARISH_TREND = "bearish_trend"
    RANGE = "range"
    HIGH_VOLATILITY = "high_volatility"
    UNKNOWN = "unknown"


class StrategyType(Enum):
    """Tipos de estrategia disponibles."""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"


class SignalDirection(Enum):
    """Dirección de la señal de trading."""
    LONG = 1
    SHORT = -1
    FLAT = 0


class SetupQuality(Enum):
    """Calidad del setup (para el journal)."""
    A_PLUS = "A+"
    A = "A"
    B = "B"
    C = "C"


# ── Configuración de entorno (.env) ────────────────────────────────────────────

class EnvSettings(BaseSettings):
    """Variables de entorno sensibles (cargadas desde .env)."""

    # Binance
    binance_api_key: str = Field(default="", alias="BINANCE_API_KEY")
    binance_api_secret: str = Field(default="", alias="BINANCE_API_SECRET")
    binance_testnet: bool = Field(default=True, alias="BINANCE_TESTNET")

    # Telegram
    telegram_bot_token: str = Field(default="", alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str = Field(default="", alias="TELEGRAM_CHAT_ID")

    # Database
    database_url: str = Field(
        default=f"sqlite:///{DATA_DIR}/trading.db",
        alias="DATABASE_URL"
    )

    # Sistema
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    environment: str = Field(default="development", alias="ENVIRONMENT")
    initial_capital: float = Field(default=1000.0, alias="INITIAL_CAPITAL")

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        populate_by_name=True,
    )


# ── Parámetros de Activos ──────────────────────────────────────────────────────

@dataclass
class AssetConfig:
    """Configuración por activo."""
    symbol: str
    priority: AssetPriority
    max_slippage: float          # Porcentaje máximo de slippage tolerado
    min_volume_24h_usd: float    # Volumen mínimo en USD para operar
    allow_shorts: bool = True    # Permite ventas en corto (Margin) para este activo


ASSETS: Dict[str, AssetConfig] = {
    # Prioridad 1 — Alta liquidez y rentabilidad validada (Sharpe ancla)
    "BTC/USDC": AssetConfig("BTC/USDC", AssetPriority.HIGH, 0.001, 100_000_000, allow_shorts=False),
    "ETH/USDC": AssetConfig("ETH/USDC", AssetPriority.HIGH, 0.001, 100_000_000),
    "DOT/USDC": AssetConfig("DOT/USDC", AssetPriority.HIGH, 0.002, 50_000_000),

    # Prioridad 2 — Estructurales, volatilidad media para EV
    "AVAX/USDC": AssetConfig("AVAX/USDC", AssetPriority.MEDIUM, 0.002, 50_000_000),
    "ADA/USDC": AssetConfig("ADA/USDC", AssetPriority.MEDIUM, 0.002, 50_000_000),
    "NEAR/USDC": AssetConfig("NEAR/USDC", AssetPriority.MEDIUM, 0.002, 50_000_000),
    "SUI/USDC": AssetConfig("SUI/USDC", AssetPriority.MEDIUM, 0.002, 50_000_000),

    # Prioridad 3 — Volatilidad Extrema / Mean Reversion
    "WLD/USDC": AssetConfig("WLD/USDC", AssetPriority.LOW, 0.003, 30_000_000),
    "TAO/USDC": AssetConfig("TAO/USDC", AssetPriority.LOW, 0.003, 30_000_000),
    "AAVE/USDC": AssetConfig("AAVE/USDC", AssetPriority.LOW, 0.003, 30_000_000),
}

# Pares ordenados por prioridad (para el backtest)
PRIORITY_1_PAIRS = [s for s, a in ASSETS.items() if a.priority == AssetPriority.HIGH]
PRIORITY_2_PAIRS = [s for s, a in ASSETS.items() if a.priority == AssetPriority.MEDIUM]
PRIORITY_3_PAIRS = [s for s, a in ASSETS.items() if a.priority == AssetPriority.LOW]
ALL_PAIRS = list(ASSETS.keys())


# ── Timeframes ─────────────────────────────────────────────────────────────────

@dataclass
class TimeframeConfig:
    """Configuración de timeframes y su jerarquía."""
    # Análisis macro (contexto)
    macro: List[str] = field(default_factory=lambda: ["1w", "1d"])
    # Análisis de tendencia
    trend: List[str] = field(default_factory=lambda: ["4h", "1h"])
    # Timing de entrada/salida
    entry: List[str] = field(default_factory=lambda: ["15m", "5m"])
    # Todos los timeframes necesarios
    all: List[str] = field(default_factory=lambda: ["1w", "1d", "4h", "1h", "15m", "5m"])
    # Timeframe principal para señales de trading
    primary: str = "4h"
    # Timeframe de ejecución
    execution: str = "15m"


TIMEFRAMES = TimeframeConfig()

# Mapeo ccxt → minutos (para ordenar y calcular duración)
TIMEFRAME_MINUTES: Dict[str, int] = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "6h": 360, "12h": 720,
    "1d": 1440, "3d": 4320, "1w": 10080,
}


# ── Parámetros de Indicadores ──────────────────────────────────────────────────

@dataclass
class TrendIndicatorParams:
    ema_fast: int = 21
    ema_mid: int = 55
    ema_slow: int = 200
    supertrend_atr_period: int = 10
    supertrend_factor: float = 3.0
    adx_period: int = 14
    adx_trend_threshold: float = 20.0   # ADX > 20 para operar
    adx_strong_threshold: float = 20.0  # ADX > 20 para tendencia fuerte (era 25)
    pullback_pct: float = 0.02           # ±2% de pullback (era 0.005)


@dataclass
class MomentumIndicatorParams:
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    rsi_neutral_low: float = 40.0    # Zona neutra para trend following (era 45)
    rsi_neutral_high: float = 70.0   # (era 65)
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    stoch_rsi_period: int = 14
    stoch_rsi_smooth_k: int = 3
    stoch_rsi_smooth_d: int = 3
    divergence_lookback: int = 14    # Velas hacia atrás para detectar divergencias


@dataclass
class VolatilityIndicatorParams:
    atr_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    keltner_period: int = 20
    keltner_atr_period: int = 10
    keltner_factor: float = 2.0
    # BB Squeeze: BB dentro de Keltner = compresión de volatilidad
    squeeze_min_candles: int = 10    # Mínimo de velas en squeeze para breakout


@dataclass
class VolumeIndicatorParams:
    vwap_anchor: str = "D"           # D=Daily, W=Weekly
    obv_signal_period: int = 21      # EMA del OBV para señal
    volume_profile_bins: int = 24    # Número de bins para el Volume Profile
    volume_avg_period: int = 20      # Periodo para el volumen promedio
    volume_breakout_multiplier: float = 1.5  # Vol > 1.5x media = breakout


@dataclass
class MarketStructureParams:
    swing_lookback: int = 10         # Velas para detectar swing highs/lows
    fvg_min_size_atr: float = 0.3   # FVG mínimo en múltiplos de ATR
    order_block_lookback: int = 50   # Velas para detectar Order Blocks
    liquidity_threshold_atr: float = 1.0  # Distancia a zona de liquidez


@dataclass
class IndicatorParams:
    trend: TrendIndicatorParams = field(default_factory=TrendIndicatorParams)
    momentum: MomentumIndicatorParams = field(default_factory=MomentumIndicatorParams)
    volatility: VolatilityIndicatorParams = field(default_factory=VolatilityIndicatorParams)
    volume: VolumeIndicatorParams = field(default_factory=VolumeIndicatorParams)
    structure: MarketStructureParams = field(default_factory=MarketStructureParams)


INDICATORS = IndicatorParams()


# ── Parámetros de Estrategias ──────────────────────────────────────────────────

@dataclass
class TrendFollowingParams:
    """Parámetros exactos de Estrategia 1: Trend Following."""
    # Condiciones de entrada
    min_adx: float = 20.0             # era 25
    rsi_min: float = 40.0             # era 45
    rsi_max: float = 70.0             # era 65
    pullback_pct: float = 0.02        # era 0.005 (±2%)
    # Gestión de trade
    sl_atr_multiplier: float = 1.0    # SL = 1 ATR bajo swing low
    tp1_rr_ratio: float = 2.0         # TP1 = 2:1 R/R mínimo
    tp2_rr_ratio: float = 3.0         # TP2 = 3:1 R/R
    tp1_close_pct: float = 0.5        # Cerrar 50% en TP1
    trailing_ema_period: int = 21     # EMA para el trailing stop (21 o 55)
    # Confluencia: señal válida si alineada en ≥ 2 timeframes superiores
    min_timeframe_confluence: int = 2


@dataclass
class MeanReversionParams:
    """Parámetros exactos de Estrategia 2: Mean Reversion."""
    max_adx: float = 20.0             # Mercado en rango
    rsi_long_threshold: float = 35.0  # RSI < 35 para long
    rsi_short_threshold: float = 65.0 # RSI > 65 para short
    sl_atr_buffer: float = 0.5        # SL = fuera de BB + 0.5 ATR
    tp1_target: str = "ema21"         # TP1 en EMA 21
    tp2_target: str = "bb_opposite"   # TP2 en banda opuesta


@dataclass
class BreakoutParams:
    """Parámetros exactos de Estrategia 3: Breakout."""
    min_squeeze_candles: int = 10     # BB Squeeze activo > 10 velas
    volume_multiplier: float = 1.5   # Vol > 150% de la media de 20 periodos
    wait_for_retest: bool = True      # Esperar retesteo del nivel roto
    sl_atr_multiplier: float = 1.5   # Stop más amplio en breakouts


@dataclass
class StrategyParams:
    trend_following: TrendFollowingParams = field(default_factory=TrendFollowingParams)
    mean_reversion: MeanReversionParams = field(default_factory=MeanReversionParams)
    breakout: BreakoutParams = field(default_factory=BreakoutParams)


STRATEGIES = StrategyParams()


# ── Gestión de Riesgo ──────────────────────────────────────────────────────────

@dataclass
class RiskParams:
    """Reglas absolutas de gestión de riesgo (del prompt maestro)."""

    # Capital base (se actualiza dinámicamente, pero parte aquí)
    initial_capital: float = 300.0    # EUR

    # Riesgo por operación según escala de capital
    # $0-$1,000 → $10 fijo (aprendizaje)
    fixed_risk_usd: float = 10.0      # Fase 1: riesgo fijo
    risk_pct_tier1: float = 0.01      # 1% → $1,000-$5,000
    risk_pct_tier2: float = 0.015     # 1.5% → $5,000-$20,000
    risk_pct_tier3: float = 0.02      # 2% → $20,000+

    # Umbrales de capital para los tiers
    tier1_min: float = 1_000.0
    tier2_min: float = 5_000.0
    tier3_min: float = 20_000.0

    # Límites de operaciones simultáneas
    max_open_positions: int = 3

    # Circuit breakers de drawdown
    max_daily_drawdown_pct: float = 0.03    # 3% → parar el día
    max_weekly_drawdown_pct: float = 0.06   # 6% → revisar sistema
    max_monthly_drawdown_pct: float = 0.10  # 10% → modo solo-estudio

    # Anti-FOMO: no entrar si el precio ya se movió más del X%
    fomo_price_move_threshold: float = 0.03  # 3%

    # Revenge trading: cooldown tras pérdida
    revenge_cooldown_minutes: int = 60

    # Comisiones Binance (Spot)
    maker_fee: float = 0.001  # 0.1%
    taker_fee: float = 0.001  # 0.1%

    # Slippage simulado (para backtesting realista)
    slippage_pct_priority1: float = 0.001   # 0.1%
    slippage_pct_priority2: float = 0.002   # 0.2%

    def get_risk_amount(self, capital: float) -> float:
        """Retorna el monto de riesgo en USD/EUR según el capital actual."""
        if capital < self.tier1_min:
            return self.fixed_risk_usd
        elif capital < self.tier2_min:
            return capital * self.risk_pct_tier1
        elif capital < self.tier3_min:
            return capital * self.risk_pct_tier2
        else:
            return capital * self.risk_pct_tier3

    def calculate_position_size(
        self,
        capital: float,
        entry_price: float,
        stop_loss_price: float,
        setup_quality: SetupQuality = SetupQuality.A,
    ) -> float:
        """
        Fórmula del prompt maestro:
        Position Size = (Capital × Riesgo%) / (Precio entrada - Stop Loss)
        """
        risk_amount = self.get_risk_amount(capital)

        # Setups A+ pueden usar hasta 2x el riesgo normal
        if setup_quality == SetupQuality.A_PLUS:
            risk_amount = min(risk_amount * 2, capital * 0.02)

        price_diff = abs(entry_price - stop_loss_price)
        if price_diff <= 0:
            raise ValueError(f"Diferencia precio-SL inválida: {price_diff}")

        return risk_amount / price_diff


RISK = RiskParams(initial_capital=300.0)


# ── Sesiones de Trading ────────────────────────────────────────────────────────

@dataclass
class TradingSessionConfig:
    """Horarios de alta liquidez (UTC)."""
    london_open: tuple = (8, 0)
    london_close: tuple = (10, 0)
    ny_open: tuple = (13, 0)
    ny_close: tuple = (17, 0)
    overlap_start: tuple = (13, 0)
    overlap_end: tuple = (17, 0)

    # Horas a evitar (manipulación frecuente)
    low_liquidity_start: int = 0    # 00:00 UTC
    low_liquidity_end: int = 6      # 06:00 UTC

    # No operar 30 min antes/después de noticias macro
    news_buffer_minutes: int = 30


SESSIONS = TradingSessionConfig()


# ── Configuración del Backtest ─────────────────────────────────────────────────

@dataclass
class BacktestConfig:
    """Parámetros del motor de backtesting."""
    # Datos históricos: mínimo 2 años como especifica el prompt
    default_years_back: int = 2
    # Warmup period para indicadores (velas descartadas al inicio)
    warmup_candles: int = 200   # Suficiente para EMA 200

    # Objetivo mínimo de Fase 1
    min_profit_factor: float = 1.5
    min_win_rate: float = 0.45
    min_sharpe_ratio: float = 1.0
    max_drawdown_target: float = 0.15  # 15%

    # Walk-forward optimization
    in_sample_pct: float = 0.70      # 70% para training
    out_sample_pct: float = 0.30     # 30% para validación


BACKTEST = BacktestConfig()


# ── Machine Learning (Fase 2) ──────────────────────────────────────────────────

@dataclass
class MLConfig:
    """
    Configuración del módulo de Machine Learning (Meta-Labeling).

    El MetaLabelModel actúa como filtro predictivo sobre las señales de
    la Estrategia 1 (Trend Following). Solo se ejecuta el trade si la
    probabilidad predicha de éxito supera `confidence_threshold`.

    Hiperparámetros conservadores (anti-overfitting para datasets pequeños):
      - n_estimators: 200 árboles (suficiente para estabilizar la varianza)
      - max_depth: 6 (evitar árboles muy profundos que memorizan ruido)
      - min_samples_leaf: 10 (cada hoja necesita al menos 10 ejemplos)
      - class_weight: 'balanced' (compensar desequilibrio win/loss)
    """
    # Tipo de modelo (solo random_forest en Fase 2 — sin XGBoost)
    model_type: str = "random_forest"

    # Umbral de confianza: solo operar si P(win) >= threshold
    # Ajustar según el Precision-Recall tradeoff en producción
    confidence_threshold: float = 0.60   # 60%

    # Hiperparámetros RandomForestClassifier
    n_estimators: int = 200              # Número de árboles
    max_depth: int = 6                   # Profundidad máxima
    min_samples_leaf: int = 10           # Mínimo de muestras por hoja
    random_state: int = 42              # Semilla de aleatoriedad (reproducibilidad)

    # Split temporal (porcentaje para train)
    # Con 2 años de datos de 4H: 75% ≈ 18 meses IS, 25% ≈ 6 meses OOS
    train_split_pct: float = 0.75

    # CVD: ventana de velas para calcular la presión de volumen acumulada
    cvd_lookback_candles: int = 5

    # Directorio donde se persiste el modelo entrenado
    model_dir: Path = field(default_factory=lambda: ROOT_DIR / "ml" / "models")

    # Nombre del archivo del modelo guardado
    model_filename: str = "meta_label_rf.joblib"


ML_CONFIG = MLConfig()


# ── Instancia global de configuración de entorno ───────────────────────────────
# Se usa en todos los módulos: from config.settings import env_settings
try:
    env_settings = EnvSettings()
except Exception:
    # En entornos sin .env (CI, tests), usar valores por defecto
    env_settings = EnvSettings(_env_file=None)  # type: ignore
