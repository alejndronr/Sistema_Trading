"""
Backtesting Histórico con Pares USDT — Análisis Multi-Régimen
=============================================================
Script independiente para testear el sistema con pares USDT que tienen
el máximo histórico disponible en Binance (desde 2017/2018).

IMPORTANTE: La operativa en producción sigue siendo con USDC.
Este script es SOLO para backtesting y análisis de robustez.

Pares con histórico máximo en Binance:
  - BTC/USDT  → desde 2017  (~9 años)
  - ETH/USDT  → desde 2017  (~9 años)
  - BNB/USDT  → desde 2017  (~9 años)
  - SOL/USDT  → desde 2020  (~5 años)
  - AVAX/USDT → desde 2020  (~5 años)
  - LINK/USDT → desde 2019  (~6 años)
  - DOT/USDT  → desde 2020  (~5 años)
  - MATIC/USDT→ desde 2019  (~6 años)

Uso:
    # Descarga + backtest completo (recomendado, primera vez)
    python scripts/backtest_usdt_historical.py --download --report

    # Solo backtest (si ya tienes los datos descargados)
    python scripts/backtest_usdt_historical.py --report

    # Solo una estrategia y par concreto
    python scripts/backtest_usdt_historical.py --symbol BTC/USDT --strategy trend_following --report

    # Comparar las 3 estrategias en BTC
    python scripts/backtest_usdt_historical.py --symbol BTC/USDT --all-strategies --report

    # Ver qué datos hay ya descargados
    python scripts/backtest_usdt_historical.py --status
"""

import argparse
import sys
import time
from pathlib import Path

# Asegurar que el directorio raíz está en el PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.logging_config import setup_logging
from data.fetcher import DataFetcher
from data.storage import OHLCVStorage

# ── Universo USDT para backtesting histórico ──────────────────────────────────
# Pares ordenados por antigüedad de datos disponibles en Binance
USDT_PAIRS = [
    "BTC/USDT",    # Desde 2017-08 — cubre 2 ciclos alcistas + 2 bajistas
    "ETH/USDT",    # Desde 2017-08 — referencia clave del mercado
    "BNB/USDT",    # Desde 2017-11 — alto histórico
    "LINK/USDT",   # Desde 2019-01 — altcoin con buen histórico
    "MATIC/USDT",  # Desde 2019-04
    "SOL/USDT",    # Desde 2020-09 — mercado alcista 2021, bajista 2022, recuperación 2023-2024
    "AVAX/USDT",   # Desde 2020-09
    "DOT/USDT",    # Desde 2020-08
]

# Estrategias disponibles en el sistema
STRATEGIES = ["trend_following", "mean_reversion", "breakout"]

# Años máximos a intentar descargar por par (Binance limita a lo disponible)
MAX_YEARS = 9


def get_strategy(strategy_name: str, symbol: str, timeframe: str):
    """Factory de estrategias — mismo que en producción."""
    from strategies.signals import (
        TrendFollowingStrategy,
        MeanReversionStrategy,
        BreakoutStrategy,
    )
    mapping = {
        "trend_following": TrendFollowingStrategy,
        "mean_reversion": MeanReversionStrategy,
        "breakout": BreakoutStrategy,
    }
    cls = mapping.get(strategy_name)
    if not cls:
        raise ValueError(f"Estrategia '{strategy_name}' desconocida. Opciones: {STRATEGIES}")
    return cls(symbol=symbol, timeframe=timeframe)


def print_header(pairs, strategy, timeframe, capital, years):
    """Imprime cabecera del backtest."""
    print(f"\n{'='*65}")
    print(f"  BACKTEST HISTÓRICO MULTI-RÉGIMEN — PARES USDT")
    print(f"{'='*65}")
    print(f"  Estrategia : {strategy}")
    print(f"  Pares      : {pairs}")
    print(f"  Timeframe  : {timeframe}")
    print(f"  Histórico  : hasta {years} año(s) (máximo disponible)")
    print(f"  Capital    : ${capital:,.0f}")
    print(f"  ⚠️  NOTA: Backtesting en USDT. Producción en USDC.")
    print(f"{'='*65}\n")


def download_usdt_data(pairs, timeframe, years, validate=False):
    """Descarga el máximo histórico posible para todos los pares USDT."""
    print(f"\n🚀 Iniciando descarga de datos históricos USDT")
    print(f"   Intentando {years} años de histórico por par")
    print(f"   (Binance solo devuelve hasta donde hay datos disponibles)\n")

    failed = []
    total_candles = 0
    since_days = int(years * 365)

    with DataFetcher() as fetcher:
        for symbol in pairs:
            print(f"\n📥 {symbol} — {timeframe}")
            try:
                t0 = time.time()
                df = fetcher.fetch_ohlcv_sync(
                    symbol=symbol,
                    timeframe=timeframe,
                    since_days=since_days,
                    show_progress=True,
                )
                elapsed = time.time() - t0

                if df is None or df.empty:
                    print(f"   ⚠️  Sin datos devueltos para {symbol}")
                    failed.append(symbol)
                    continue

                date_from = df["timestamp"].min().date()
                date_to   = df["timestamp"].max().date()
                candles   = len(df)
                total_candles += candles

                print(f"   ✅ {candles:,} velas | {date_from} → {date_to} ({elapsed:.1f}s)")

                if validate:
                    quality = fetcher.validate_data_quality(df, timeframe)
                    score = quality.get("quality_score", "?")
                    gaps  = quality.get("gaps_count", "?")
                    print(f"   📊 Calidad: {score}/100 | Gaps detectados: {gaps}")
                    for issue in quality.get("issues", []):
                        print(f"      ⚠️  {issue}")

            except KeyboardInterrupt:
                print("\n⛔ Descarga interrumpida por el usuario")
                sys.exit(0)
            except Exception as e:
                print(f"   ❌ Error: {e}")
                failed.append(symbol)

    print(f"\n{'─'*50}")
    print(f"📦 Total velas descargadas : {total_candles:,}")
    if failed:
        print(f"❌ Pares fallidos          : {failed}")
    print(f"{'─'*50}\n")
    return failed


def run_backtest_for_pair(engine, symbol, timeframe, strategy_name, storage, show_progress=True):
    """Ejecuta el backtest para un par y estrategia, devuelve result o None."""
    df = storage.load_ohlcv(symbol, timeframe)

    if df is None or df.empty:
        print(f"   ⚠️  Sin datos locales para {symbol} {timeframe}")
        print(f"   💡 Ejecuta primero con --download")
        return None

    date_from = df["timestamp"].min().date()
    date_to   = df["timestamp"].max().date()
    years_covered = (df["timestamp"].max() - df["timestamp"].min()).days / 365.25
    print(f"   📈 {len(df):,} velas | {date_from} → {date_to} ({years_covered:.1f} años)")

    strategy = get_strategy(strategy_name, symbol, timeframe)

    from backtesting.engine import BacktestEngine
    result = engine.run(
        symbol=symbol,
        df=df.copy(),
        strategy=strategy,
        timeframe=timeframe,
        show_progress=show_progress,
    )
    return result


def print_single_result(result, symbol):
    """Imprime el resumen de un resultado individual."""
    m = result.metrics
    pasa = "✅ SÍ" if result.passes_phase1_criteria() else "❌ NO"
    print(f"\n   📊 Resultado para {symbol}:")
    print(f"      Trades totales : {m.total_trades}")
    print(f"      Win Rate       : {m.win_rate*100:.1f}%")
    print(f"      Profit Factor  : {m.profit_factor:.2f}")
    print(f"      Sharpe (anual) : {m.sharpe_ratio_annual:.2f}")
    print(f"      Max Drawdown   : {m.max_drawdown_pct*100:.1f}%")
    print(f"      Retorno total  : {m.total_return_pct*100:.1f}%")
    print(f"      Pasa Fase 1    : {pasa}")


def print_multi_summary(results):
    """Tabla resumen multi-par."""
    print(f"\n{'='*75}")
    print(f"  RESUMEN MULTI-PAR — ROBUSTEZ MULTI-RÉGIMEN")
    print(f"{'='*75}")
    print(f"  {'Par':<14} {'Años':<6} {'Trades':<8} {'WR%':<7} {'PF':<7} {'Sharpe':<8} {'MaxDD%':<8} {'Ret%':<8} {'Fase1'}")
    print(f"  {'-'*14} {'-'*5} {'-'*7} {'-'*6} {'-'*6} {'-'*7} {'-'*7} {'-'*7} {'-'*6}")
    for symbol, (result, years_covered) in results.items():
        m = result.metrics
        pasa = "✅" if result.passes_phase1_criteria() else "❌"
        print(
            f"  {symbol:<14} {years_covered:<6.1f} {m.total_trades:<8} "
            f"{m.win_rate*100:<6.1f}% {m.profit_factor:<7.2f} "
            f"{m.sharpe_ratio_annual:<8.2f} {m.max_drawdown_pct*100:<7.1f}% "
            f"{m.total_return_pct*100:<8.1f}% {pasa}"
        )
    print(f"{'='*75}\n")


def main():
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Backtesting Multi-Régimen con Pares USDT (máximo histórico)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--symbol", default=None,
                        help="Par específico a testear (ej: BTC/USDT). Por defecto: todos los pares USDT")
    parser.add_argument("--strategy", default="trend_following",
                        choices=STRATEGIES, help="Estrategia a probar")
    parser.add_argument("--all-strategies", action="store_true",
                        help="Probar las 3 estrategias (solo válido con --symbol)")
    parser.add_argument("--timeframe", default="4h",
                        help="Timeframe del backtest (default: 4h)")
    parser.add_argument("--capital", type=float, default=1000.0,
                        help="Capital inicial en USD (default: 1000)")
    parser.add_argument("--years", type=float, default=MAX_YEARS,
                        help=f"Años de histórico a intentar descargar (default: {MAX_YEARS})")
    parser.add_argument("--download", action="store_true",
                        help="Descargar datos antes del backtest")
    parser.add_argument("--validate", action="store_true",
                        help="Validar calidad de datos al descargar")
    parser.add_argument("--report", action="store_true",
                        help="Generar informe HTML interactivo por par")
    parser.add_argument("--status", action="store_true",
                        help="Mostrar datos USDT disponibles en la BD y salir")
    parser.add_argument("--no-progress", action="store_true",
                        help="Desactivar barras de progreso")

    args = parser.parse_args()

    # ── Determinar pares ──────────────────────────────────────────────────────
    if args.symbol:
        sym = args.symbol.upper()
        if not sym.endswith("/USDT"):
            print(f"⚠️  Añadiendo /USDT al símbolo: {sym}/USDT")
            sym = f"{sym}/USDT"
        pairs = [sym]
    else:
        pairs = USDT_PAIRS

    # ── Modo status ───────────────────────────────────────────────────────────
    if args.status:
        storage = OHLCVStorage()
        available = storage.get_available_data()
        usdt_data = available[available["symbol"].str.endswith("/USDT")] if not available.empty else available
        if usdt_data.empty:
            print("\n📭 No hay datos USDT en la base de datos.")
            print("   Ejecuta: python scripts/backtest_usdt_historical.py --download\n")
        else:
            print("\n📊 Datos USDT disponibles en la base de datos:")
            print(usdt_data.to_string(index=False))
            print(f"\n💾 Tamaño total de BD: {storage.get_database_size_mb():.2f} MB\n")
        return

    # ── Descarga ──────────────────────────────────────────────────────────────
    if args.download:
        failed = download_usdt_data(pairs, args.timeframe, args.years, args.validate)
        if failed:
            print(f"⚠️  Algunos pares fallaron en la descarga: {failed}")
            print("    El backtest continuará con los datos disponibles.\n")

    # ── Backtesting ───────────────────────────────────────────────────────────
    strategies_to_run = STRATEGIES if args.all_strategies and args.symbol else [args.strategy]

    for strategy_name in strategies_to_run:
        print_header(pairs, strategy_name, args.timeframe, args.capital, args.years)

        from backtesting.engine import BacktestEngine, BacktestReporter
        storage = OHLCVStorage()
        engine  = BacktestEngine(initial_capital=args.capital)
        results = {}  # symbol -> (result, years_covered)

        for symbol in pairs:
            print(f"\n📊 Procesando {symbol} [{strategy_name}]...")

            try:
                result = run_backtest_for_pair(
                    engine, symbol, args.timeframe, strategy_name,
                    storage, show_progress=not args.no_progress
                )
                if result is None:
                    continue

                df_loaded = storage.load_ohlcv(symbol, args.timeframe)
                years_covered = (
                    (df_loaded["timestamp"].max() - df_loaded["timestamp"].min()).days / 365.25
                    if df_loaded is not None and not df_loaded.empty else 0.0
                )

                results[symbol] = (result, years_covered)
                print_single_result(result, symbol)

                # Generar informe HTML
                if args.report:
                    try:
                        reporter = BacktestReporter()
                        report_path = reporter.generate(result)
                        print(f"   📄 Informe HTML: {report_path}")
                    except Exception as e:
                        print(f"   ⚠️  No se pudo generar informe HTML: {e}")

            except KeyboardInterrupt:
                print("\n⛔ Backtest interrumpido por el usuario")
                break
            except Exception as e:
                import traceback
                print(f"   ❌ Error en backtest de {symbol}: {e}")
                if "--debug" in sys.argv:
                    traceback.print_exc()

        # Tabla resumen multi-par
        if len(results) > 1:
            print_multi_summary(results)
        elif len(results) == 1:
            print("\n✅ Backtest completado para el par seleccionado.")

    print("\n✅ Proceso de backtesting histórico USDT finalizado.\n")


if __name__ == "__main__":
    main()
