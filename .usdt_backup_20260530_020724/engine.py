"""
Motor de Backtesting
====================
Simula la ejecución de estrategias sobre datos históricos OHLCV.

Características:
  - Event-driven a nivel de vela (no vectorizado puro, para realismo)
  - Soporte multi-asset (varios pares simultáneos)
  - Gestión completa de posiciones: TP1/TP2, trailing stop, breakeven
  - Comisiones Binance (0.1%) + slippage realista
  - Circuit breakers de drawdown integrados
  - Integración con todos los módulos: indicadores, estrategias, riesgo

Uso:
    engine = BacktestEngine(initial_capital=300)
    results = engine.run(
        symbol="BTC/USDT",
        df=ohlcv_df,
        strategy=TrendFollowingStrategy("BTC/USDT"),
    )
    print(results.metrics.summary())
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Type

import pandas as pd
from tqdm import tqdm

from backtesting.metrics import BacktestMetrics
from backtesting.portfolio import Portfolio, Position
from config.logging_config import get_logger
from config.settings import (
    ASSETS,
    BACKTEST,
    INDICATORS,
    RISK,
    STRATEGIES,
    AssetPriority,
    MarketRegime,
    SetupQuality,
    SignalDirection,
    StrategyType,
)
from risk.position_sizer import PositionSizer
from risk.regime_filter import RegimeFilter
from strategies.base import BaseStrategy

logger = get_logger(__name__)


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
            f"Win Rate:      {m.win_rate*100:.1f}%  (objetivo: ≥45%) {'✅' if m.win_rate >= 0.45 else '❌'}",
            f"Profit Factor: {m.profit_factor:.2f}   (objetivo: ≥1.5) {'✅' if m.profit_factor >= 1.5 else '❌'}",
            f"Sharpe Ratio:  {m.sharpe_ratio_annual:.2f}   (objetivo: ≥1.0) {'✅' if m.sharpe_ratio_annual >= 1.0 else '❌'}",
            f"Max Drawdown:  {m.max_drawdown_pct*100:.1f}%  (objetivo: ≤15%) {'✅' if m.max_drawdown_pct <= 0.15 else '❌'}",
            f"",
            f"── Métricas Adicionales ──",
            f"Expectancy:    ${m.expectancy:.2f}/trade",
            f"Avg R:         {m.avg_r_multiple:.2f}R",
            f"Max streak L:  {m.max_consecutive_losses}",
            f"",
            f"PASA FASE 1: {'✅ SÍ' if self.passes_phase1_criteria() else '❌ NO'}",
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

        Args:
            symbol: Par de trading (ej: "BTC/USDT")
            df: DataFrame OHLCV
            strategy: Instancia de estrategia
            timeframe: Timeframe del DataFrame
            prepare_indicators: Si True, calcula todos los indicadores
            show_progress: Mostrar barra de progreso
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
        start_date = df["timestamp"].iloc[0]
        end_date = df["timestamp"].iloc[-1]

        iterator = tqdm(
            range(1, len(df)),
            desc=f"Backtesting {symbol}",
            unit="velas",
            disable=not show_progress,
        )

        for i in iterator:
            row = df.iloc[i]
            ts = row["timestamp"]
            high = row["high"]
            low = row["low"]
            close = row["close"]

            # Actualizar equity
            if i % 10 == 0:
                portfolio.record_equity(ts)
                sizer.update_capital(portfolio.capital, ts)

            # Obtener ema21 para trailing stop
            ema21_col = f"ema_{INDICATORS.trend.ema_fast}"
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

        runtime = time.time() - start_time
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
        Gestiona una posición abierta vela a vela:
        - Verifica si se tocó SL o TP1/TP2
        - Mueve SL a breakeven cuando se alcanza 1:1 R/R
        - Aplica trailing stop después de TP1
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

            # Verificar TP1 (50% a cerrar)
            if not pos.tp1_hit and high >= pos.take_profit_1:
                trade = portfolio.close_position(
                    trade_id=trade_id,
                    exit_price=pos.take_profit_1,
                    exit_time=timestamp,
                    exit_reason="take_profit_1",
                    size_fraction=0.5 if pos.strategy == StrategyType.MEAN_REVERSION.value else 0.3,
                    slippage_pct=slippage,
                )
                if trade:
                    sizer.register_position_closed(trade.pnl_usd, timestamp)
                    # Mover SL a breakeven
                    portfolio.update_stop_loss(trade_id, entry)
                return

            # TP2 con trailing stop
            if pos.tp1_hit:
                # Trailing stop usando EMA configurada
                trailing_ema_period = getattr(STRATEGIES.trend_following, 'trailing_ema_period', 21)
                trailing_ema_col = f"trend_ema_{trailing_ema_period}"
                trailing_ema_val = row.get(trailing_ema_col)
                
                if trailing_ema_val is not None and not np.isnan(trailing_ema_val):
                    # Trailing stop para LONG sube con la EMA (nunca baja)
                    new_trailing_sl = trailing_ema_val
                    # Solo actualizamos si el nuevo SL es mejor que el actual
                    if new_trailing_sl > sl:
                        portfolio.update_stop_loss(trade_id, new_trailing_sl)

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
                trade = portfolio.close_position(
                    trade_id=trade_id,
                    exit_price=pos.take_profit_1,
                    exit_time=timestamp,
                    exit_reason="take_profit_1",
                    size_fraction=0.5 if pos.strategy == StrategyType.MEAN_REVERSION.value else 0.3,
                    slippage_pct=slippage,
                )
                if trade:
                    sizer.register_position_closed(trade.pnl_usd, timestamp)
                    portfolio.update_stop_loss(trade_id, entry)

            elif pos.tp1_hit:
                trailing_ema_period = getattr(STRATEGIES.trend_following, 'trailing_ema_period', 21)
                trailing_ema_col = f"trend_ema_{trailing_ema_period}"
                trailing_ema_val = row.get(trailing_ema_col)
                
                if trailing_ema_val is not None and not np.isnan(trailing_ema_val):
                    new_trailing_sl = trailing_ema_val
                    if new_trailing_sl < sl:
                        portfolio.update_stop_loss(trade_id, new_trailing_sl)

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
