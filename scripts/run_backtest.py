"""
Script CLI Principal — Ejecutar Backtesting
============================================
Uso:
    # Backtest básico (BTC/USDT, Trend Following, 4H, 2 años)
    python scripts/run_backtest.py

    # Con opciones
    python scripts/run_backtest.py --symbol ETH/USDT --strategy mean_reversion --timeframe 1h

    # Multi-par
    python scripts/run_backtest.py --all-pairs --strategy trend_following --report

    # Con descarga automática de datos
    python scripts/run_backtest.py --symbol BTC/USDT --download --years 2 --report
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.logging_config import setup_logging
from config.settings import ALL_PAIRS, PRIORITY_1_PAIRS, RISK


def get_strategy(strategy_name: str, symbol: str, timeframe: str):
    """Factory de estrategias por nombre."""
    from strategies.signals import (
        TrendFollowingStrategy,
        MeanReversionStrategy,
        BreakoutStrategy,
    )

    strategies = {
        "trend_following": TrendFollowingStrategy,
        "mean_reversion": MeanReversionStrategy,
        "breakout": BreakoutStrategy,
    }

    cls = strategies.get(strategy_name)
    if not cls:
        raise ValueError(f"Estrategia desconocida: {strategy_name}. Opciones: {list(strategies.keys())}")

    return cls(symbol=symbol, timeframe=timeframe)


def main():
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Motor de Backtesting — Sistema de Trading Algorítmico"
    )
    parser.add_argument("--symbol", default="BTC/USDC", help="Par a analizar")
    parser.add_argument("--all-pairs", action="store_true", help="Backtest en todos los pares")
    parser.add_argument("--priority1", action="store_true", help="Solo pares Prioridad 1 (BTC+ETH)")
    parser.add_argument(
        "--strategy", default="trend_following",
        choices=["trend_following", "mean_reversion", "breakout"],
        help="Estrategia a usar"
    )
    parser.add_argument("--timeframe", default="4h", help="Timeframe del backtest")
    parser.add_argument("--years", type=float, default=2.0, help="Años de histórico")
    parser.add_argument("--capital", type=float, default=1000.0,
                        help=f"Capital inicial (por defecto: $1000)")
    parser.add_argument("--download", action="store_true",
                        help="Descargar datos antes del backtest si no están en la BD")
    parser.add_argument("--report", action="store_true",
                        help="Generar informe HTML al finalizar")
    parser.add_argument("--no-progress", action="store_true",
                        help="Desactivar barras de progreso")
    parser.add_argument("--save-trades", action="store_true",
                        help="Guardar trades en el journal CSV/SQLite")

    args = parser.parse_args()

    from data.fetcher import DataFetcher
    from data.storage import OHLCVStorage
    from backtesting.engine import BacktestEngine, BacktestReporter

    # Determinar pares a analizar
    if args.all_pairs:
        symbols = ALL_PAIRS
    elif args.priority1:
        symbols = PRIORITY_1_PAIRS
    else:
        symbols = [s.strip() for s in args.symbol.split(",") if s.strip()]

    since_days = int(args.years * 365)
    show_progress = not args.no_progress

    print(f"\n{'='*60}")
    print(f"  SISTEMA DE TRADING — BACKTEST ENGINE")
    print(f"{'='*60}")
    print(f"  Estrategia:  {args.strategy}")
    print(f"  Pares:       {symbols}")
    print(f"  Timeframe:   {args.timeframe}")
    print(f"  Período:     {args.years} año(s)")
    print(f"  Capital:     ${args.capital:.0f}")
    print(f"{'='*60}\n")

    storage = OHLCVStorage()
    engine = BacktestEngine(initial_capital=args.capital)
    results = {}

    for symbol in symbols:
        print(f"\n📊 Procesando {symbol}...")

        # Cargar datos
        df = storage.load_ohlcv(symbol, args.timeframe)

        if df is None or df.empty:
            if args.download:
                print(f"   📥 Descargando datos históricos de {symbol}...")
                with DataFetcher() as fetcher:
                    df = fetcher.fetch_ohlcv_sync(
                        symbol=symbol,
                        timeframe=args.timeframe,
                        since_days=since_days,
                        show_progress=show_progress,
                    )
            else:
                print(f"   ⚠️  Sin datos para {symbol} {args.timeframe}")
                print(f"   💡 Ejecuta: python scripts/download_data.py --pairs {symbol} --timeframes {args.timeframe}")
                continue

        if df is None or df.empty:
            print(f"   ❌ No se pudieron obtener datos para {symbol}")
            continue

        print(f"   📈 {len(df):,} velas | {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")

        try:
            strategy = get_strategy(args.strategy, symbol, args.timeframe)
            result = engine.run(
                symbol=symbol,
                df=df.copy(),
                strategy=strategy,
                timeframe=args.timeframe,
                show_progress=show_progress,
            )
            results[symbol] = result
            print(result.summary())

            # Generar informe HTML
            if args.report:
                reporter = BacktestReporter()
                report_path = reporter.generate(result)
                print(f"   📄 Informe: {report_path}")

            # Guardar trades en journal
            if args.save_trades:
                from journal.trade_logger import TradeLogger
                logger = TradeLogger()
                trades_df = result.portfolio.get_trades_dataframe()
                for _, trade in trades_df.iterrows():
                    try:
                        logger.log_trade(
                            trade_id=trade["trade_id"],
                            symbol=trade["symbol"],
                            direction=trade["direction"],
                            timeframe=args.timeframe,
                            strategy=trade["strategy"],
                            entry_price=trade["entry_price"],
                            stop_loss=trade["stop_loss"],
                            take_profit_1=trade["take_profit_1"],
                            exit_price=trade["exit_price"],
                            position_size=trade["position_size"],
                            risk_amount=trade["risk_amount"],
                            pnl_usd=trade["pnl_usd"],
                            entry_time=trade["entry_time"],
                            exit_time=trade["exit_time"],
                            exit_reason=trade["exit_reason"],
                            setup_quality=trade["setup_quality"],
                            is_backtest=True,
                        )
                    except Exception as e:
                        pass
                print(f"   📝 Trades guardados en: {logger.journal_path}")

        except Exception as e:
            import traceback
            print(f"   ❌ Error en backtest de {symbol}: {e}")
            if "--debug" in sys.argv:
                traceback.print_exc()

    # Resumen final multi-par
    if len(results) > 1:
        print(f"\n{'='*60}")
        print(f"  RESUMEN MULTI-PAR")
        print(f"{'='*60}")
        print(f"  {'Par':<15} {'Trades':>7} {'WR%':>7} {'PF':>7} {'Sharpe':>8} {'MaxDD%':>8} {'Pasa Fase1':>12}")
        print(f"  {'-'*15} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*8} {'-'*12}")
        for symbol, result in results.items():
            m = result.metrics
            pasa = "✅ SÍ" if result.passes_phase1_criteria() else "❌ NO"
            print(
                f"  {symbol:<15} {m.total_trades:>7} {m.win_rate*100:>6.1f}% "
                f"{m.profit_factor:>7.2f} {m.sharpe_ratio_annual:>8.2f} "
                f"{m.max_drawdown_pct*100:>7.1f}% {pasa:>12}"
            )
        print(f"{'='*60}\n")

    print("\n✅ Backtesting completado.")


if __name__ == "__main__":
    main()
