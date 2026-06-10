"""
Motor de Backtesting, Portfolio Simulado, Reportes y Optimización
================================================================
"""

from __future__ import annotations

import time
import uuid
import itertools
import multiprocessing
import concurrent.futures
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Type, Any, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from jinja2 import Template
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backtesting.metrics import BacktestMetrics
from config.logging_config import get_logger
from config.settings import (
    ASSETS,
    BACKTEST,
    INDICATORS,
    RISK,
    STRATEGIES,
    REPORTS_DIR,
    AssetPriority,
    MarketRegime,
    SetupQuality,
    SignalDirection,
    StrategyType,
)
from risk.position_sizer import PositionSizer
from risk.regime_filter import RegimeFilter
from risk.signal_scorer import get_scorer
from risk.ev_filter import get_ev_filter
from strategies.signals import BaseStrategy, TrendFollowingStrategy, MetaStrategy

logger = get_logger(__name__)


# ── Clases de Portfolio (originalmente de portfolio.py) ────────────────────────

@dataclass
class Position:
    """Representa una posición abierta."""
    trade_id: str
    symbol: str
    strategy: str
    direction: SignalDirection
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: Optional[float]
    position_size: float      # Número de unidades
    risk_amount: float        # Capital en riesgo
    entry_time: datetime
    setup_quality: SetupQuality = SetupQuality.A
    entry_reason: str = ""

    # Estado de la posición
    tp1_hit: bool = False             # TP1 ya alcanzado (50% cerrado)
    current_sl: float = 0.0          # SL actual (puede moverse a BE o trailing)
    partial_size: float = 0.0        # Tamaño de la porción que queda tras TP1

    def __post_init__(self):
        if self.current_sl == 0.0:
            self.current_sl = self.stop_loss
        if self.partial_size == 0.0:
            self.partial_size = self.position_size

    @property
    def unrealized_pnl(self) -> float:
        """PnL no realizado — requiere precio actual."""
        return 0.0  # Se calcula en el engine con el precio actual

    def calculate_pnl(self, exit_price: float, size: Optional[float] = None) -> float:
        """Calcula el PnL realizado para un tamaño dado."""
        sz = size or self.position_size
        if self.direction == SignalDirection.LONG:
            return (exit_price - self.entry_price) * sz
        else:
            return (self.entry_price - exit_price) * sz


@dataclass
class ClosedTrade:
    """Registro completo de un trade cerrado."""
    trade_id: str
    symbol: str
    strategy: str
    direction: str
    setup_quality: str
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: Optional[float]
    position_size: float
    risk_amount: float
    pnl_usd: float
    pnl_pct: float
    r_multiple: float          # PnL / risk_amount
    entry_time: datetime
    exit_time: datetime
    duration_hours: float
    exit_reason: str
    entry_reason: str
    market_regime: str = "unknown"


class Portfolio:
    """
    Gestiona el portfolio simulado durante el backtesting.
    Registra cada trade con comisiones y slippage realista.
    """

    def __init__(
        self,
        initial_capital: float = RISK.initial_capital,
        maker_fee: float = RISK.maker_fee,
        taker_fee: float = RISK.taker_fee,
    ):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee

        self._open_positions: Dict[str, Position] = {}
        self._closed_trades: List[ClosedTrade] = []
        self._equity_curve: List[dict] = []

    # ── Apertura de posiciones ─────────────────────────────────────────────────

    def open_position(
        self,
        symbol: str,
        strategy: str,
        direction: SignalDirection,
        entry_price: float,
        stop_loss: float,
        take_profit_1: float,
        position_size: float,
        risk_amount: float,
        entry_time: datetime,
        take_profit_2: Optional[float] = None,
        setup_quality: SetupQuality = SetupQuality.A,
        entry_reason: str = "",
        market_regime: str = "unknown",
        slippage_pct: float = 0.001,
    ) -> Optional[str]:
        """
        Abre una nueva posición con slippage y comisión simulados.
        Retorna el trade_id o None si no hay capital suficiente.
        """
        # Aplicar slippage al precio de entrada
        if direction == SignalDirection.LONG:
            real_entry = entry_price * (1 + slippage_pct)
        else:
            real_entry = entry_price * (1 - slippage_pct)

        # Comisión de apertura
        commission = real_entry * position_size * self.taker_fee
        total_cost = real_entry * position_size + commission

        if total_cost > self.capital:
            return None  # Capital insuficiente

        self.capital -= commission  # Deducir solo la comisión (no el notional completo en spot)

        trade_id = str(uuid.uuid4())
        position = Position(
            trade_id=trade_id,
            symbol=symbol,
            strategy=strategy,
            direction=direction,
            entry_price=real_entry,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            position_size=position_size,
            risk_amount=risk_amount,
            entry_time=entry_time,
            setup_quality=setup_quality,
            entry_reason=entry_reason,
        )

        self._open_positions[trade_id] = position
        return trade_id

    # ── Cierre de posiciones ───────────────────────────────────────────────────

    def close_position(
        self,
        trade_id: str,
        exit_price: float,
        exit_time: datetime,
        exit_reason: str = "",
        size_fraction: float = 1.0,  # 1.0 = cerrar todo, 0.5 = cerrar la mitad
        slippage_pct: float = 0.001,
    ) -> Optional[ClosedTrade]:
        """
        Cierra una posición (total o parcialmente).
        Aplica slippage y comisión de salida.
        """
        if trade_id not in self._open_positions:
            return None

        pos = self._open_positions[trade_id]

        # Aplicar slippage al precio de salida
        if pos.direction == SignalDirection.LONG:
            real_exit = exit_price * (1 - slippage_pct)
        else:
            real_exit = exit_price * (1 + slippage_pct)

        size_to_close = pos.partial_size * size_fraction
        commission = real_exit * size_to_close * self.maker_fee

        pnl_gross = pos.calculate_pnl(real_exit, size_to_close)
        pnl_net = pnl_gross - commission

        self.capital += pnl_net

        # Calcular métricas del trade
        risk = pos.risk_amount
        r_multiple = pnl_net / risk if risk > 0 else 0

        _exit_ts = pd.to_datetime(exit_time) if isinstance(exit_time, str) else exit_time
        _entry_ts = pd.to_datetime(pos.entry_time) if isinstance(pos.entry_time, str) else pos.entry_time
        duration = (_exit_ts - _entry_ts).total_seconds() / 3600

        trade = ClosedTrade(
            trade_id=trade_id,
            symbol=pos.symbol,
            strategy=pos.strategy,
            direction=pos.direction.name,
            setup_quality=pos.setup_quality.value,
            entry_price=pos.entry_price,
            exit_price=real_exit,
            stop_loss=pos.stop_loss,
            take_profit_1=pos.take_profit_1,
            take_profit_2=pos.take_profit_2,
            position_size=size_to_close,
            risk_amount=risk,
            pnl_usd=round(pnl_net, 4),
            pnl_pct=round(pnl_net / self.initial_capital * 100, 4),
            r_multiple=round(r_multiple, 3),
            entry_time=pos.entry_time,
            exit_time=exit_time,
            duration_hours=round(duration, 2),
            exit_reason=exit_reason,
            entry_reason=pos.entry_reason,
        )
        self._closed_trades.append(trade)

        # Si se cerró todo, remover la posición
        if size_fraction >= 1.0:
            del self._open_positions[trade_id]
        else:
            # Actualizar tamaño restante
            pos.partial_size -= size_to_close
            pos.tp1_hit = True

        return trade

    def update_stop_loss(self, trade_id: str, new_sl: float) -> bool:
        """Actualiza el SL de una posición abierta (solo hacia adelante, nunca atrás)."""
        if trade_id not in self._open_positions:
            return False
        pos = self._open_positions[trade_id]
        if pos.direction == SignalDirection.LONG:
            pos.current_sl = max(pos.current_sl, new_sl)  # Solo puede subir
        else:
            pos.current_sl = min(pos.current_sl, new_sl)  # Solo puede bajar
        return True

    def record_equity(self, timestamp: datetime) -> None:
        """Registra el equity actual en la curva."""
        self._equity_curve.append({
            "timestamp": timestamp,
            "capital": self.capital,
            "open_positions": len(self._open_positions),
        })

    # ── Propiedades ───────────────────────────────────────────────────────────

    @property
    def open_positions(self) -> Dict[str, Position]:
        return self._open_positions

    @property
    def closed_trades(self) -> List[ClosedTrade]:
        return self._closed_trades

    @property
    def equity_curve(self) -> pd.DataFrame:
        if not self._equity_curve:
            return pd.DataFrame()
        return pd.DataFrame(self._equity_curve)

    @property
    def total_pnl(self) -> float:
        return self.capital - self.initial_capital

    @property
    def total_return_pct(self) -> float:
        return (self.capital - self.initial_capital) / self.initial_capital * 100

    def get_trades_dataframe(self) -> pd.DataFrame:
        """Retorna todos los trades cerrados como DataFrame."""
        if not self._closed_trades:
            return pd.DataFrame()
        return pd.DataFrame([vars(t) for t in self._closed_trades])


# ── Clases del Motor de Backtesting (originalmente de engine.py) ──────────────

@dataclass
class BacktestResult:
    """Resultado completo de un backtest."""
    symbol: str
    strategy: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_trades: int
    portfolio: Portfolio
    metrics: BacktestMetrics
    runtime_seconds: float
    # ── Capas cuantitativas ─────────────────────────────────────────────
    bootstrap_stats: dict = field(default_factory=dict)   # Capa 6: p-value
    volatility_regime: str = "NORMAL_VOL"                 # Capa 3/5: régimen ATR
    kelly_multiplier: float = 1.0                         # Capa 2: Kelly calibrado

    @property
    def total_return_pct(self) -> float:
        return (self.final_capital - self.initial_capital) / self.initial_capital * 100

    def passes_phase1_criteria(self) -> bool:
        """Verifica si el backtest cumple los objetivos de Fase 1."""
        m = self.metrics
        return (
            m.profit_factor >= BACKTEST.min_profit_factor
            and m.win_rate >= BACKTEST.min_win_rate
            and m.sharpe_ratio_annual >= BACKTEST.min_sharpe_ratio
            and m.max_drawdown_pct <= BACKTEST.max_drawdown_target
        )

    def summary(self) -> str:
        """Resumen textual del resultado."""
        m = self.metrics
        lines = [
            f"\n{'='*60}",
            f"BACKTEST: {self.symbol} | {self.strategy} | {self.timeframe}",
            f"{'='*60}",
            f"Período: {self.start_date.date()} → {self.end_date.date()}",
            f"Capital: ${self.initial_capital:.0f} → ${self.final_capital:.2f} ({self.total_return_pct:+.1f}%)",
            f"Trades:  {self.total_trades}",
            f"",
            f"── Métricas Objetivo (Fase 1) ──",
            f"Win Rate:      {m.win_rate*100:.1f}%  (objetivo: >=45%) {'✅' if m.win_rate >= 0.45 else '❌'}",
            f"Profit Factor: {m.profit_factor:.2f}   (objetivo: >=1.5) {'✅' if m.profit_factor >= 1.5 else '❌'}",
            f"Sharpe Ratio:  {m.sharpe_ratio_annual:.2f}   (objetivo: >=1.0) {'✅' if m.sharpe_ratio_annual >= 1.0 else '❌'}",
            f"Max Drawdown:  {m.max_drawdown_pct*100:.1f}%  (objetivo: <=15%) {'✅' if m.max_drawdown_pct <= 0.15 else '❌'}",
            f"",
            f"── Métricas Adicionales ──",
            f"Expectancy:    ${m.expectancy:.2f}/trade",
            f"Avg R:         {m.avg_r_multiple:.2f}R",
            f"Max streak L:  {m.max_consecutive_losses}",
        ]

        # ── Sección cuantitativa ───────────────────────────────────────
        if self.bootstrap_stats:
            bs = self.bootstrap_stats
            sig_icon = "\u2705" if bs.get("significant") else "\u26a0\ufe0f"
            lines += [
                f"",
                f"── Análisis Estadístico (Bootstrap 1000x) ──",
                f"Significancia:  {sig_icon} {'EDGE REAL (p<0.05)' if bs.get('significant') else 'NO SIGNIFICATIVO (posible suerte)'}",
                f"p-value:        {bs.get('p_value', '?'):.4f}   (objetivo: <0.05)",
                f"PF Observado:   {bs.get('observed_pf', '?'):.3f}",
                f"PF Aleatorio P95: {bs.get('bootstrap_pf_p95', '?'):.3f}",
            ]
            if bs.get("warning"):
                lines.append(f"⚠️  {bs['warning']}")

        pass_text = "✅ SÍ" if self.passes_phase1_criteria() else "❌ NO"
        lines += [
            f"",
            f"Régimen Volatilidad: {self.volatility_regime}",
            f"Kelly Multiplier:    {self.kelly_multiplier:.2f}x",
            f"",
            f"PASA FASE 1: {pass_text}",
            f"{'='*60}\n",
        ]
        return "\n".join(lines)


class BacktestEngine:
    """
    Motor de backtesting event-driven a nivel de vela.
    Procesa cada vela cronológicamente y gestiona posiciones abiertas.
    """

    def __init__(self, initial_capital: float = RISK.initial_capital):
        self.initial_capital = initial_capital
        self.regime_filter = RegimeFilter()

    def run(
        self,
        symbol: str,
        df: pd.DataFrame,
        strategy: BaseStrategy,
        timeframe: str = "4h",
        prepare_indicators: bool = True,
        show_progress: bool = True,
        apply_session_filter: bool = False,  # Desactivado en backtest por defecto
    ) -> BacktestResult:
        """
        Ejecuta el backtest para un par y estrategia dados.
        """
        start_time = time.time()
        logger.info(
            "backtest_start",
            symbol=symbol,
            strategy=strategy.strategy_type.value,
            timeframe=timeframe,
            candles=len(df),
        )

        # 1. Preparar indicadores
        if prepare_indicators:
            logger.info("calculating_indicators")
            df = strategy.prepare_dataframe(df)

        # 2. Generar señales
        logger.info("generating_signals")
        df = strategy.generate_signals(df)

        # 3. Eliminar período de warmup
        df = df.iloc[BACKTEST.warmup_candles:].reset_index(drop=True)

        if len(df) < 50:
            raise ValueError(f"Datos insuficientes después del warmup: {len(df)} velas")

        # 4. Inicializar portfolio y sizers
        portfolio = Portfolio(
            initial_capital=self.initial_capital,
            maker_fee=RISK.maker_fee,
            taker_fee=RISK.taker_fee,
        )
        sizer = PositionSizer(initial_capital=self.initial_capital)

        # 5. Determinar slippage según prioridad del activo
        asset_priority = ASSETS.get(symbol)
        slippage = (
            RISK.slippage_pct_priority1
            if asset_priority and asset_priority.priority == AssetPriority.HIGH
            else RISK.slippage_pct_priority2
        )

        # 6. Loop principal por vela
        start_date = pd.to_datetime(df["timestamp"].iloc[0], unit="ms" if isinstance(df["timestamp"].iloc[0], (int, float)) else None)
        end_date = pd.to_datetime(df["timestamp"].iloc[-1], unit="ms" if isinstance(df["timestamp"].iloc[-1], (int, float)) else None)

        iterator = tqdm(
            range(1, len(df)),
            desc=f"Backtesting {symbol}",
            unit="velas",
            disable=not show_progress,
        )

        for i in iterator:
            row = df.iloc[i]
            ts = row.get("datetime", pd.to_datetime(row["timestamp"], unit="ms" if isinstance(row["timestamp"], (int, float)) else None))
            if isinstance(ts, str): ts = pd.to_datetime(ts)
            high = row["high"]
            low = row["low"]
            close = row["close"]

            # Actualizar equity
            if i % 10 == 0:
                portfolio.record_equity(ts)
                sizer.update_capital(portfolio.capital, ts)

            # Obtener ema21 para trailing stop
            ema21_col = f"ema_{INDICATORS.trend.ema_fast}"
            if ema21_col not in df.columns and "ema21" in df.columns:
                ema21_col = "ema21"
            ema21 = row.get(ema21_col, None)

            # 6a. Gestionar posiciones abiertas (verificar SL/TP)
            positions_to_process = list(portfolio.open_positions.items())
            for trade_id, pos in positions_to_process:
                self._manage_open_position(
                    portfolio=portfolio,
                    sizer=sizer,
                    trade_id=trade_id,
                    pos=pos,
                    high=high,
                    low=low,
                    close=close,
                    timestamp=ts,
                    atr=row.get("atr", close * 0.02) if "atr" in df.columns else close * 0.02,
                    ema21=ema21,
                    slippage=slippage,
                    row=row,
                )

            # 6b. Verificar circuit breakers antes de abrir nuevas posiciones
            can_trade, cb_reason = sizer.drawdown_tracker.can_trade(ts)
            if not can_trade:
                continue

            # 6c. Verificar límite de posiciones simultáneas
            if len(portfolio.open_positions) >= RISK.max_open_positions:
                continue

            # 6d. Evaluar señal en la vela actual
            signal = int(row.get("signal", 0))
            if signal == 0:
                continue

            # 6e. Filtro de régimen de mercado
            can_operate, regime, regime_reason = self.regime_filter.full_filter(
                df, symbol, idx=i
            )
            if not can_operate:
                continue

            # 6f. Filtro de régimen para estrategia
            active_strategies = self.regime_filter.get_active_strategies(regime)
            if strategy.strategy_type not in active_strategies:
                continue

            # 6g. Calcular tamaño de posición
            entry_price = row.get("entry_price", close)
            stop_loss = row.get("stop_loss", 0)
            tp1 = row.get("take_profit_1", 0)
            tp2 = row.get("take_profit_2", None)

            if pd.isna(entry_price) or pd.isna(stop_loss) or stop_loss <= 0:
                continue

            # Validar coherencia de precios
            if signal == 1:  # Long
                if not (stop_loss < entry_price < tp1):
                    continue
            elif signal == -1:  # Short
                if not (tp1 < entry_price < stop_loss):
                    continue

            quality_str = row.get("signal_quality", "A")
            quality_map = {"A+": SetupQuality.A_PLUS, "A": SetupQuality.A, "B": SetupQuality.B, "C": SetupQuality.C}
            quality = quality_map.get(quality_str, SetupQuality.A)

            # Multiplicador de régimen (alta volatilidad → 50%)
            regime_mult = self.regime_filter.get_position_size_multiplier(regime)

            try:
                sizing = sizer.calculate_position_size(
                    entry_price=entry_price,
                    stop_loss_price=stop_loss,
                    quality=quality,
                    symbol_priority=asset_priority.priority.value if asset_priority else 1,
                    regime_multiplier=regime_mult,
                )
            except ValueError:
                continue

            # 6g.5 ── Filtro de Valor Esperado (EV) y Scoring Bayesiano ────────────
            # Calcula la probabilidad bayesiana y el EV del setup.
            # Rechaza trades que no tienen ventaja matemática demostrable.
            try:
                confidence = float(row.get("confidence", 0.47))
                rr_tp1 = abs(tp1 - entry_price) / abs(entry_price - stop_loss) if abs(entry_price - stop_loss) > 0 else 2.0

                ev_result = get_ev_filter().calculate(
                    p_win=confidence,
                    reward_r=rr_tp1,
                )

                if not ev_result.passes_filter:
                    logger.debug(
                        "trade_rejected_ev",
                        symbol=symbol,
                        reason=ev_result.rejection_reason,
                        p_win=round(confidence, 3),
                        ev=ev_result.expected_value_r,
                        rr=round(rr_tp1, 2),
                    )
                    continue

            except Exception:
                # Si el filtro falla por cualquier razón, continuar sin filtrar
                pass

            # 6h. Abrir posición
            direction = SignalDirection.LONG if signal == 1 else SignalDirection.SHORT
            trade_id = portfolio.open_position(
                symbol=symbol,
                strategy=strategy.strategy_type.value,
                direction=direction,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit_1=tp1,
                take_profit_2=tp2 if not pd.isna(tp2) else None,
                position_size=sizing["position_size"],
                risk_amount=sizing["risk_amount"],
                entry_time=ts,
                setup_quality=quality,
                entry_reason=str(row.get("signal_reason", "")),
                market_regime=regime.value,
                slippage_pct=slippage,
            )

            if trade_id:
                sizer.register_position_opened()
                logger.info(
                    "position_opened",
                    trade_id=trade_id[:8],
                    symbol=symbol,
                    direction=direction.name,
                    entry=round(entry_price, 4),
                    sl=round(stop_loss, 4),
                    tp1=round(tp1, 4),
                    size=sizing["position_size"],
                    quality=quality.value,
                )

        # 7. Cerrar posiciones abiertas al final del período
        final_close = df["close"].iloc[-1]
        final_ts = df["timestamp"].iloc[-1]
        for trade_id in list(portfolio.open_positions.keys()):
            portfolio.close_position(
                trade_id=trade_id,
                exit_price=final_close,
                exit_time=final_ts,
                exit_reason="end_of_backtest",
                slippage_pct=slippage,
            )

        portfolio.record_equity(final_ts)

        # 8. Calcular métricas
        trades_df = portfolio.get_trades_dataframe()
        equity_df = portfolio.equity_curve
        metrics = BacktestMetrics.from_trades(
            trades_df=trades_df,
            equity_curve=equity_df,
            initial_capital=self.initial_capital,
        )

        # 8.5 ── Kelly: alimentar historial de R-múltiples al sizer ──────────────
        # Esto permite que el Kelly Criterion se calibre con los resultados reales
        # del backtest, reflejando el edge real del sistema en este activo.
        if not trades_df.empty and "r_multiple" in trades_df.columns:
            for r_val in trades_df["r_multiple"].values:
                sizer.record_trade_result(float(r_val))
            kelly_mult = sizer.kelly_fraction()
            logger.info(
                "kelly_calibrated",
                symbol=symbol,
                n_trades=len(trades_df),
                kelly_multiplier=kelly_mult,
                win_rate=round(metrics.win_rate * 100, 1),
            )

        # 8.6 ── Bootstrap Significance Test ──────────────────────────────────
        # Verifica si los resultados son estadísticamente significativos
        # o simplemente suerte en una secuencia aleatoria.
        bootstrap_stats = {}
        if not trades_df.empty and "pnl_usd" in trades_df.columns:
            try:
                from backtesting.metrics import bootstrap_significance
                bootstrap_stats = bootstrap_significance(
                    pnl_series=trades_df["pnl_usd"].values,
                    n_simulations=1000,
                )
            except Exception:
                pass

        runtime = time.time() - start_time

        # Extraer régimen de volatilidad de la última vela con señal
        vol_regime = "NORMAL_VOL"
        if "volatility_regime" in df.columns:
            vol_series = df["volatility_regime"].dropna()
            if not vol_series.empty:
                vol_regime = str(vol_series.iloc[-1])

        kelly_mult_final = sizer.kelly_fraction()

        result = BacktestResult(
            symbol=symbol,
            strategy=strategy.strategy_type.value,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_capital=portfolio.capital,
            total_trades=len(trades_df),
            portfolio=portfolio,
            metrics=metrics,
            runtime_seconds=runtime,
            bootstrap_stats=bootstrap_stats,
            volatility_regime=vol_regime,
            kelly_multiplier=kelly_mult_final,
        )


        logger.info(
            "backtest_complete",
            symbol=symbol,
            trades=len(trades_df),
            final_capital=round(portfolio.capital, 2),
            profit_factor=round(metrics.profit_factor, 2),
            win_rate=round(metrics.win_rate * 100, 1),
            runtime_s=round(runtime, 1),
        )

        return result

    def _manage_open_position(
        self,
        portfolio: Portfolio,
        sizer: PositionSizer,
        trade_id: str,
        pos: Position,
        high: float,
        low: float,
        close: float,
        timestamp: datetime,
        atr: float,
        ema21: Optional[float],
        slippage: float,
        row: pd.Series,
    ) -> None:
        """
        Gestiona una posición abierta vela a vela.
        """
        direction = pos.direction
        entry = pos.entry_price
        sl = pos.current_sl

        if direction == SignalDirection.LONG:
            # Verificar Stop Loss (precio tocó el low)
            if low <= sl:
                exit_price = sl
                trade = portfolio.close_position(
                    trade_id=trade_id,
                    exit_price=exit_price,
                    exit_time=timestamp,
                    exit_reason="stop_loss",
                    slippage_pct=slippage,
                )
                if trade:
                    sizer.register_position_closed(trade.pnl_usd, timestamp)
                return

            # Verificar TP1 (porcentaje configurable)
            if not pos.tp1_hit and high >= pos.take_profit_1:
                # Obtener el porcentaje dinámico de configuración
                strategy_config = getattr(STRATEGIES, pos.strategy, None)
                close_pct = getattr(strategy_config, "tp1_close_pct", 0.5) if strategy_config else 0.5

                trade = portfolio.close_position(
                    trade_id=trade_id,
                    exit_price=pos.take_profit_1,
                    exit_time=timestamp,
                    exit_reason="take_profit_1",
                    size_fraction=close_pct,
                    slippage_pct=slippage,
                )
                if trade:
                    sizer.register_position_closed(trade.pnl_usd, timestamp)
                    # Mover SL a breakeven para asegurar la operación libre de riesgo
                    portfolio.update_stop_loss(trade_id, entry)
                return

            # TP2 (Take Profit Final)
            if pos.tp1_hit:
                if pos.take_profit_2 and high >= pos.take_profit_2:
                    trade = portfolio.close_position(
                        trade_id=trade_id,
                        exit_price=pos.take_profit_2,
                        exit_time=timestamp,
                        exit_reason="take_profit_2",
                        slippage_pct=slippage,
                    )
                    if trade:
                        sizer.register_position_closed(trade.pnl_usd, timestamp)

            # Breakeven: mover SL si precio alcanza 1:1 R/R
            elif not pos.tp1_hit:
                risk = entry - pos.stop_loss
                if close >= entry + risk:
                    portfolio.update_stop_loss(trade_id, entry)


        elif direction == SignalDirection.SHORT:
            # Stop Loss al alza
            if high >= sl:
                trade = portfolio.close_position(
                    trade_id=trade_id,
                    exit_price=sl,
                    exit_time=timestamp,
                    exit_reason="stop_loss",
                    slippage_pct=slippage,
                )
                if trade:
                    sizer.register_position_closed(trade.pnl_usd, timestamp)
                return

            # TP1 (precio cae hasta TP1)
            if not pos.tp1_hit and low <= pos.take_profit_1:
                strategy_config = getattr(STRATEGIES, pos.strategy, None)
                close_pct = getattr(strategy_config, "tp1_close_pct", 0.5) if strategy_config else 0.5

                trade = portfolio.close_position(
                    trade_id=trade_id,
                    exit_price=pos.take_profit_1,
                    exit_time=timestamp,
                    exit_reason="take_profit_1",
                    size_fraction=close_pct,
                    slippage_pct=slippage,
                )
                if trade:
                    sizer.register_position_closed(trade.pnl_usd, timestamp)
                    # Mover SL a breakeven
                    portfolio.update_stop_loss(trade_id, entry)
                return

            # TP2 (Take Profit Final)
            elif pos.tp1_hit:
                if pos.take_profit_2 and low <= pos.take_profit_2:
                    trade = portfolio.close_position(
                        trade_id=trade_id,
                        exit_price=pos.take_profit_2,
                        exit_time=timestamp,
                        exit_reason="take_profit_2",
                        slippage_pct=slippage,
                    )
                    if trade:
                        sizer.register_position_closed(trade.pnl_usd, timestamp)

    def run_multi_asset(
        self,
        symbols: List[str],
        dfs: Dict[str, pd.DataFrame],
        strategy_class: Type[BaseStrategy],
        timeframe: str = "4h",
        show_progress: bool = True,
    ) -> Dict[str, BacktestResult]:
        """
        Ejecuta el backtest para múltiples activos.
        Retorna diccionario {symbol: BacktestResult}.
        """
        results = {}
        for symbol in symbols:
            if symbol not in dfs:
                logger.warning("no_data_for_symbol", symbol=symbol)
                continue
            try:
                strategy = strategy_class(symbol, timeframe)
                result = self.run(
                    symbol=symbol,
                    df=dfs[symbol].copy(),
                    strategy=strategy,
                    timeframe=timeframe,
                    show_progress=show_progress,
                )
                results[symbol] = result
                print(result.summary())
            except Exception as e:
                logger.error("backtest_failed", symbol=symbol, error=str(e))

        return results


# ── Generador de Reportes (originalmente de report.py) ────────────────────────

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Backtest Report — {{ symbol }} {{ strategy }}</title>
<style>
  :root {
    --bg: #0f1117; --surface: #1a1d27; --accent: #4f7cff;
    --green: #00c896; --red: #ff4d4f; --yellow: #ffd700;
    --text: #e0e0e0; --muted: #6b7280;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; }
  .header { background: linear-gradient(135deg, var(--surface), #242837); padding: 2rem 3rem; border-bottom: 1px solid #2d3148; }
  .header h1 { font-size: 1.8rem; color: var(--accent); margin-bottom: 0.5rem; }
  .header p { color: var(--muted); font-size: 0.9rem; }
  .cards { display: flex; flex-wrap: wrap; gap: 1rem; padding: 2rem 3rem 1rem; }
  .card { background: var(--surface); border-radius: 12px; padding: 1.2rem 1.5rem;
          flex: 1; min-width: 160px; border: 1px solid #2d3148; }
  .card .label { font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.4rem; }
  .card .value { font-size: 1.5rem; font-weight: 700; }
  .card .badge { font-size: 0.7rem; padding: 2px 8px; border-radius: 20px; margin-top: 0.3rem; display: inline-block; }
  .green { color: var(--green); } .red { color: var(--red); } .yellow { color: var(--yellow); }
  .badge-pass { background: rgba(0,200,150,0.15); color: var(--green); border: 1px solid var(--green); }
  .badge-fail { background: rgba(255,77,79,0.15); color: var(--red); border: 1px solid var(--red); }
  .section { padding: 1rem 3rem; }
  .section h2 { font-size: 1.1rem; color: var(--muted); margin-bottom: 1rem; font-weight: 500; }
  .chart-container { background: var(--surface); border-radius: 12px; padding: 1rem; border: 1px solid #2d3148; margin-bottom: 1.5rem; }
  table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
  thead { background: #242837; }
  th { padding: 0.7rem 1rem; text-align: left; color: var(--muted); font-weight: 500; }
  td { padding: 0.5rem 1rem; border-bottom: 1px solid #1e2130; }
  tr:hover { background: rgba(79,124,255,0.05); }
  .phase1 { padding: 1rem 3rem 2rem; }
  .phase1 .result { font-size: 1.3rem; font-weight: 700; padding: 1rem 2rem; border-radius: 10px; display: inline-block; }
  .pass { background: rgba(0,200,150,0.15); color: var(--green); border: 1px solid var(--green); }
  .fail { background: rgba(255,77,79,0.15); color: var(--red); border: 1px solid var(--red); }
  .footer { padding: 1.5rem 3rem; color: var(--muted); font-size: 0.8rem; border-top: 1px solid #2d3148; }
</style>
</head>
<body>
<div class="header">
  <h1>📊 Backtest Report — {{ symbol }} | {{ strategy }} | {{ timeframe }}</h1>
  <p>Período: {{ start_date }} → {{ end_date }} &nbsp;|&nbsp; Capital inicial: ${{ initial_capital }} &nbsp;|&nbsp; Generado: {{ generated_at }}</p>
</div>

<div class="cards">
  <div class="card">
    <div class="label">Total Trades</div>
    <div class="value">{{ metrics.total_trades }}</div>
  </div>
  <div class="card">
    <div class="label">Win Rate</div>
    <div class="value {% if metrics.win_rate >= 0.45 %}green{% else %}red{% endif %}">
      {{ "%.1f"|format(metrics.win_rate * 100) }}%
    </div>
    <div class="badge {% if metrics.win_rate >= 0.45 %}badge-pass{% else %}badge-fail{% endif %}">
      Obj: >=45%
    </div>
  </div>
  <div class="card">
    <div class="label">Profit Factor</div>
    <div class="value {% if metrics.profit_factor >= 1.5 %}green{% else %}red{% endif %}">
      {{ "%.2f"|format(metrics.profit_factor) }}
    </div>
    <div class="badge {% if metrics.profit_factor >= 1.5 %}badge-pass{% else %}badge-fail{% endif %}">
      Obj: >=1.5
    </div>
  </div>
  <div class="card">
    <div class="label">Sharpe (Anual)</div>
    <div class="value {% if metrics.sharpe_ratio_annual >= 1.0 %}green{% else %}red{% endif %}">
      {{ "%.2f"|format(metrics.sharpe_ratio_annual) }}
    </div>
    <div class="badge {% if metrics.sharpe_ratio_annual >= 1.0 %}badge-pass{% else %}badge-fail{% endif %}">
      Obj: >=1.0
    </div>
  </div>
  <div class="card">
    <div class="label">Max Drawdown</div>
    <div class="value {% if metrics.max_drawdown_pct <= 0.15 %}green{% else %}red{% endif %}">
      {{ "%.1f"|format(metrics.max_drawdown_pct * 100) }}%
    </div>
    <div class="badge {% if metrics.max_drawdown_pct <= 0.15 %}badge-pass{% else %}badge-fail{% endif %}">
      Obj: <=15%
    </div>
  </div>
  <div class="card">
    <div class="label">Expectancy</div>
    <div class="value {% if metrics.expectancy > 0 %}green{% else %}red{% endif %}">
      ${{ "%.2f"|format(metrics.expectancy) }}
    </div>
  </div>
  <div class="card">
    <div class="label">Return Total</div>
    <div class="value {% if metrics.total_return_pct > 0 %}green{% else %}red{% endif %}">
      {{ "%+.1f"|format(metrics.total_return_pct) }}%
    </div>
  </div>
  <div class="card">
    <div class="label">Avg R Multiple</div>
    <div class="value {% if metrics.avg_r_multiple > 0 %}green{% else %}red{% endif %}">
      {{ "%.2f"|format(metrics.avg_r_multiple) }}R
    </div>
  </div>
</div>

<div class="section">
  <h2>Equity Curve</h2>
  <div class="chart-container">{{ equity_chart }}</div>
  <h2>Drawdown</h2>
  <div class="chart-container">{{ drawdown_chart }}</div>
</div>

<div class="section">
  <h2>Distribución de Retornos</h2>
  <div class="chart-container">{{ distribution_chart }}</div>
</div>

<div class="section">
  <h2>Journal de Trades</h2>
  <div class="chart-container" style="overflow-x:auto;">
    <table>
      <thead>
        <tr>
          <th>#</th><th>Fecha</th><th>Par</th><th>Dir</th><th>Calidad</th>
          <th>Entrada</th><th>SL</th><th>TP1</th><th>Salida</th>
          <th>Tamaño</th><th>PnL ($)</th><th>PnL (%)</th><th>R</th>
          <th>Duración</th><th>Razón Salida</th>
        </tr>
      </thead>
      <tbody>
        {% for t in trades %}
        <tr>
          <td>{{ loop.index }}</td>
          <td>{{ t.entry_time }}</td>
          <td>{{ t.symbol }}</td>
          <td class="{% if t.direction == 'LONG' %}green{% else %}red{% endif %}">{{ t.direction }}</td>
          <td>{{ t.setup_quality }}</td>
          <td>{{ "%.4f"|format(t.entry_price) }}</td>
          <td>{{ "%.4f"|format(t.stop_loss) }}</td>
          <td>{{ "%.4f"|format(t.take_profit_1) }}</td>
          <td>{{ "%.4f"|format(t.exit_price) }}</td>
          <td>{{ "%.6f"|format(t.position_size) }}</td>
          <td class="{% if t.pnl_usd > 0 %}green{% else %}red{% endif %}">
            {{ "%+.2f"|format(t.pnl_usd) }}
          </td>
          <td class="{% if t.pnl_pct > 0 %}green{% else %}red{% endif %}">
            {{ "%+.2f"|format(t.pnl_pct) }}%
          </td>
          <td class="{% if t.r_multiple > 0 %}green{% else %}red{% endif %}">
            {{ "%.2f"|format(t.r_multiple) }}R
          </td>
          <td>{{ "%.1f"|format(t.duration_hours) }}h</td>
          <td>{{ t.exit_reason }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
</div>

<div class="phase1">
  <h2 style="margin-bottom:1rem;">Veredicto — Objetivos Fase 1</h2>
  <div class="result {% if passes %}pass{% else %}fail{% endif %}">
    {% if passes %}✅ PASA FASE 1 — Listo para Paper Trading{% else %}❌ NO PASA FASE 1 — Optimizar estrategia{% endif %}
  </div>
</div>

<div class="footer">
  Sistema de Trading Algorítmico | Fase 1: Backtesting | Capital: ${{ initial_capital }} | {{ generated_at }}
</div>
</body>
</html>"""


class BacktestReporter:
    """Genera informes HTML interactivos de backtesting."""

    def generate(
        self,
        result: BacktestResult,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Genera el informe HTML completo.
        Retorna la ruta al archivo generado.
        """
        if output_path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_{result.symbol.replace('/', '_')}_{result.strategy}_{ts}.html"
            output_path = REPORTS_DIR / filename

        # Asegurar que el directorio de reportes exista
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        trades_df = result.portfolio.get_trades_dataframe()
        equity_df = result.portfolio.equity_curve

        # Generar gráficos Plotly
        equity_chart = self._equity_chart(equity_df, result.initial_capital)
        drawdown_chart = self._drawdown_chart(equity_df)
        distribution_chart = self._distribution_chart(trades_df)

        # Preparar datos para el template
        trades_list = []
        if not trades_df.empty:
            for _, row in trades_df.iterrows():
                t = row.to_dict()
                # Formatear timestamps
                for ts_col in ["entry_time", "exit_time"]:
                    if ts_col in t and hasattr(t[ts_col], "strftime"):
                        t[ts_col] = t[ts_col].strftime("%Y-%m-%d %H:%M")
                trades_list.append(type("Trade", (), t)())

        template = Template(_HTML_TEMPLATE)
        html = template.render(
            symbol=result.symbol,
            strategy=result.strategy,
            timeframe=result.timeframe,
            start_date=result.start_date.strftime("%Y-%m-%d") if result.start_date else "N/A",
            end_date=result.end_date.strftime("%Y-%m-%d") if result.end_date else "N/A",
            initial_capital=f"{result.initial_capital:.0f}",
            final_capital=f"{result.final_capital:.2f}",
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
            metrics=result.metrics,
            equity_chart=equity_chart,
            drawdown_chart=drawdown_chart,
            distribution_chart=distribution_chart,
            trades=trades_list,
            passes=result.passes_phase1_criteria(),
        )

        output_path.write_text(html, encoding="utf-8")
        logger.info("report_generated", path=str(output_path))
        print(f"\n📊 Informe generado: {output_path}")
        return output_path

    def _equity_chart(self, equity_df: pd.DataFrame, initial_capital: float) -> str:
        """Equity curve interactiva con Plotly."""
        if equity_df is None or equity_df.empty:
            return "<p>Sin datos de equity</p>"

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=equity_df["timestamp"],
            y=equity_df["capital"],
            mode="lines",
            name="Capital",
            line=dict(color="#4f7cff", width=2),
            fill="tozeroy",
            fillcolor="rgba(79,124,255,0.1)",
        ))
        fig.add_hline(
            y=initial_capital,
            line_dash="dash",
            line_color="#6b7280",
            annotation_text=f"Capital inicial ${initial_capital:.0f}",
        )
        fig.update_layout(
            **self._dark_layout("Equity Curve"),
            yaxis_title="Capital ($)",
        )
        return fig.to_html(full_html=False, include_plotlyjs="cdn")

    def _drawdown_chart(self, equity_df: pd.DataFrame) -> str:
        """Drawdown chart."""
        if equity_df is None or equity_df.empty:
            return "<p>Sin datos</p>"

        capital = equity_df["capital"].values
        peak = np.maximum.accumulate(capital)
        drawdown = (capital - peak) / np.where(peak > 0, peak, 1) * 100

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=equity_df["timestamp"],
            y=drawdown,
            mode="lines",
            name="Drawdown",
            line=dict(color="#ff4d4f", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(255,77,79,0.15)",
        ))
        fig.add_hline(y=-15, line_dash="dash", line_color="#ffd700",
                      annotation_text="Límite Fase 1 (-15%)")
        fig.update_layout(
            **self._dark_layout("Drawdown (%)"),
            yaxis_title="Drawdown (%)",
        )
        return fig.to_html(full_html=False, include_plotlyjs=False)

    def _distribution_chart(self, trades_df: pd.DataFrame) -> str:
        """Histograma de distribución de PnL + scatter de R multiples."""
        if trades_df is None or trades_df.empty:
            return "<p>Sin trades</p>"

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Distribución PnL ($)", "R Multiples por Trade"]
        )

        pnl = trades_df["pnl_usd"].values
        colors = ["#00c896" if p > 0 else "#ff4d4f" for p in pnl]

        fig.add_trace(go.Histogram(
            x=pnl,
            nbinsx=30,
            name="PnL",
            marker_color=colors,
        ), row=1, col=1)

        if "r_multiple" in trades_df.columns:
            r = trades_df["r_multiple"].values
            r_colors = ["#00c896" if v > 0 else "#ff4d4f" for v in r]
            fig.add_trace(go.Bar(
                x=list(range(len(r))),
                y=r,
                name="R",
                marker_color=r_colors,
            ), row=1, col=2)

        fig.update_layout(**self._dark_layout("Distribución de Retornos"))
        return fig.to_html(full_html=False, include_plotlyjs=False)

    @staticmethod
    def _dark_layout(title: str) -> dict:
        return {
            "title": title,
            "paper_bgcolor": "#1a1d27",
            "plot_bgcolor": "#1a1d27",
            "font": {"color": "#e0e0e0", "family": "Segoe UI"},
            "xaxis": {"gridcolor": "#2d3148", "showgrid": True},
            "yaxis": {"gridcolor": "#2d3148", "showgrid": True},
            "showlegend": False,
            "height": 350,
            "margin": {"t": 40, "b": 30, "l": 50, "r": 20},
        }


# ── Clase de Optimización e Hilos de Trabajo (originalmente de optimizer.py) ──

def run_single_backtest(params: Dict[str, Any], dfs: Dict[str, pd.DataFrame], symbol: str) -> Dict[str, Any]:
    """
    Worker function para ejecutar un backtest individual.
    Como corre en un proceso separado, podemos modificar STRATEGIES libremente.
    """
    # 1. Aplicar parámetros
    tf = params["timeframe"]
    df_train = dfs[tf]
    
    # Mutar configuración global iterando el prefijo
    for key, val in params.items():
        if key == "timeframe" or key == "strategy": continue
        if key.startswith("tf_"):
            setattr(STRATEGIES.trend_following, key.replace("tf_", ""), val)
        elif key.startswith("meta_"):
            setattr(STRATEGIES.meta, key.replace("meta_", ""), val)
    
    # 2. Inicializar motor
    engine = BacktestEngine(initial_capital=BACKTEST.initial_capital if hasattr(BACKTEST, 'initial_capital') else 1000.0)
    
    # 3. Correr backtest silencioso
    strat_name = params.get("strategy", "trend_following")
    if strat_name == "meta":
        strategy = MetaStrategy(symbol, timeframe=tf, params=STRATEGIES.meta)
    else:
        strategy = TrendFollowingStrategy(symbol, timeframe=tf, params=STRATEGIES.trend_following)
    
    result = engine.run(
        symbol=symbol,
        df=df_train.copy(),
        strategy=strategy,
        timeframe=tf,
        show_progress=False
    )
    
    # 4. Calcular fitness
    # Constraint dura: max drawdown <= 15%
    if result.metrics.max_drawdown_pct > 0.15:
        fitness = -1.0
    else:
        fitness = result.metrics.profit_factor
        
    return {
        "params": params,
        "fitness": fitness,
        "result": result
    }


class HyperparameterOptimizer:
    def __init__(self, dfs: Dict[str, pd.DataFrame], symbol: str, in_sample_months: int = 18):
        self.dfs = dfs
        self.symbol = symbol
        
        self.dfs_train = {}
        self.dfs_test = {}
        
        for tf, df in self.dfs.items():
            df_copy = df.copy()
            df_copy['datetime'] = pd.to_datetime(df_copy['timestamp'], unit='ms', utc=True)
            df_copy.set_index('datetime', inplace=True)
            df_copy.sort_index(inplace=True)
            
            # Split point
            end_date = df_copy.index.max()
            split_date = end_date - pd.DateOffset(months=6) # 6 months test
            
            df_train = df_copy[df_copy.index <= split_date].copy()
            df_test = df_copy[df_copy.index > split_date].copy()
            
            # Revertir index para el engine
            df_train.reset_index(inplace=True)
            df_test.reset_index(inplace=True)
            
            self.dfs_train[tf] = df_train
            self.dfs_test[tf] = df_test
        
        logger.info(f"Train/Test split configurado para timeframes: {list(self.dfs.keys())}")
        
    def optimize(self, param_grid: Dict[str, List[Any]]) -> List[Dict]:
        """
        Realiza GridSearchCV en paralelo sobre el df_train.
        """
        # Generar combinaciones
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        logger.info(f"Iniciando Grid Search paralela con {len(combinations)} combinaciones...")
        
        results = []
        
        # Ejecución paralela
        num_cores = multiprocessing.cpu_count()
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
            # Submitir tareas
            futures = [
                executor.submit(run_single_backtest, combo, self.dfs_train, self.symbol)
                for combo in combinations
            ]
            
            # Recolectar resultados con barra de progreso
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                try:
                    res = future.result()
                    results.append(res)
                except Exception as e:
                    logger.error(f"Error en worker: {e}")
                
                completed += 1
                if completed % 20 == 0 or completed == len(combinations):
                    logger.info(f"Progreso: {completed}/{len(combinations)} completados.")
                    
        # Ordenar resultados por fitness (Profit Factor)
        results.sort(key=lambda x: x["fitness"], reverse=True)
        
        # Filtrar los que no pasaron la restricción (fitness = -1.0)
        valid_results = [r for r in results if r["fitness"] > 0]
        
        logger.info(f"Optimización completada. {len(valid_results)} combinaciones superaron el filtro de Drawdown (<=15%).")
        
        return valid_results
        
    def validate_out_of_sample(self, best_params: Dict[str, Any]) -> Any:
        """
        Valida el mejor set de parámetros en el bloque Out-of-Sample.
        """
        logger.info(f"Validando Out-of-Sample con parámetros: {best_params}")
        
        # Usamos df_test completo (asegurando el overlap para indicadores)
        # Necesitamos el warmup
        warmup_candles = 200
        dfs_test_full = {}
        for tf in self.dfs_test.keys():
            df_warmup = self.dfs_train[tf].tail(warmup_candles)
            df_test_full = pd.concat([df_warmup, self.dfs_test[tf]]).copy()
            dfs_test_full[tf] = df_test_full
        
        res = run_single_backtest(best_params, dfs_test_full, self.symbol)
        return res

# ── CLI para invocar Backtest desde consola ──────────────────────────────────
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ejecutar backtest del motor.")
    parser.add_argument("--dry-run", action="store_true", help="Ejecutar sin guardar reporte")
    parser.add_argument("--report-by-phase", action="store_true", help="Mostrar desglose por fase del ciclo")
    args = parser.parse_args()
    
    logger.info("Iniciando validación de backtest (Phase 3 Integration)...")
    
    # Cargar datos para 'BTC/USDC' como ejemplo de validación
    symbol = "BTC/USDC"
    # Requerirá que sqlite tenga los datos. Usamos load_daily_ohlcv u otra función
    import sqlite3
    conn = sqlite3.connect("data/db/trading.db")
    query = f"SELECT * FROM ohlcv WHERE symbol='{symbol}' AND timeframe='1h' ORDER BY timestamp"
    df = pd.read_sql(query, conn)
    conn.close()
    
    if df.empty:
        logger.warning("No hay datos en DB para el backtest. Finalizando.")
        exit(0)
        
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("datetime", inplace=True)
    
    # Inicializar el engine
    engine = BacktestEngine(initial_capital=1000.0)
    strategy = TrendFollowingStrategy(symbol, timeframe="1h")
    
    # Correr
    result = engine.run(symbol=symbol, df=df, strategy=strategy, timeframe="1h")
    
    # Imprimir resumen de métricas
    print("\n" + "="*60)
    print(" BACKTEST METRICS SUMMARY")
    print("="*60)
    print(result.metrics.summary())
    print("="*60 + "\n")
    
    if args.report_by_phase:
        print("=> Desglose por fases no implementado aún en BacktestMetrics.\n")

