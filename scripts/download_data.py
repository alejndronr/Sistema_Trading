"""
Script CLI — Descarga de Datos Históricos
==========================================
Uso:
    python scripts/download_data.py --pairs BTC/USDT ETH/USDT --timeframes 4h 1h 15m --years 2
    python scripts/download_data.py --all-pairs --timeframes 4h --years 2
    python scripts/download_data.py --status  # Ver datos disponibles en BD
"""

import argparse
import sys
from pathlib import Path

# Asegurar que el directorio raíz está en el path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.logging_config import setup_logging
from config.settings import ALL_PAIRS, TIMEFRAMES
from data.fetcher import DataFetcher
from data.storage import OHLCVStorage


def main():
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Descarga datos OHLCV históricos de Binance para el backtesting"
    )
    parser.add_argument(
        "--pairs", nargs="+", default=["BTC/USDC", "ETH/USDC"],
        help="Pares a descargar (ej: BTC/USDT ETH/USDT)"
    )
    parser.add_argument(
        "--all-pairs", action="store_true",
        help="Descargar todos los pares del universo"
    )
    parser.add_argument(
        "--timeframes", nargs="+", default=["4h", "1h", "15m"],
        help="Timeframes a descargar (ej: 4h 1h 15m)"
    )
    parser.add_argument(
        "--years", type=float, default=2.0,
        help="Años de histórico a descargar (por defecto: 2)"
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Mostrar datos disponibles en la base de datos"
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Validar la calidad de los datos después de descargar"
    )

    args = parser.parse_args()

    # Mostrar estado de la BD
    if args.status:
        storage = OHLCVStorage()
        available = storage.get_available_data()
        if available.empty:
            print("📭 No hay datos en la base de datos. Ejecuta el script sin --status para descargar.")
        else:
            print("\n📊 Datos disponibles en la base de datos:")
            print(available.to_string(index=False))
            print(f"\n💾 Tamaño de BD: {storage.get_database_size_mb():.2f} MB")
        return

    pairs = ALL_PAIRS if args.all_pairs else args.pairs
    since_days = int(args.years * 365)

    print(f"\n🚀 Iniciando descarga de datos históricos")
    print(f"   Pares: {pairs}")
    print(f"   Timeframes: {args.timeframes}")
    print(f"   Período: {args.years} año(s) ({since_days} días)\n")

    with DataFetcher() as fetcher:
        total_downloaded = 0
        failed = []

        for symbol in pairs:
            for timeframe in args.timeframes:
                print(f"\n📥 {symbol} — {timeframe}")
                try:
                    df = fetcher.fetch_ohlcv_sync(
                        symbol=symbol,
                        timeframe=timeframe,
                        since_days=since_days,
                        show_progress=True,
                    )

                    if df.empty:
                        print(f"   ⚠️  Sin datos para {symbol} {timeframe}")
                        failed.append(f"{symbol}/{timeframe}")
                        continue

                    print(f"   ✅ {len(df):,} velas | {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
                    total_downloaded += len(df)

                    # Validación de calidad
                    if args.validate:
                        quality = fetcher.validate_data_quality(df, timeframe)
                        print(f"   📊 Calidad: {quality['quality_score']}/100 | Gaps: {quality['gaps_count']}")
                        if quality["issues"]:
                            for issue in quality["issues"]:
                                print(f"      ⚠️  {issue}")

                except KeyboardInterrupt:
                    print("\n⛔ Descarga interrumpida por el usuario")
                    sys.exit(0)
                except Exception as e:
                    print(f"   ❌ Error: {e}")
                    failed.append(f"{symbol}/{timeframe}")

    print(f"\n{'='*50}")
    print(f"✅ Descarga completada: {total_downloaded:,} velas totales")
    if failed:
        print(f"❌ Fallidos: {failed}")
    print(f"{'='*50}\n")

    # Mostrar resumen final de la BD
    storage = OHLCVStorage()
    available = storage.get_available_data()
    if not available.empty:
        print("📊 Resumen de la base de datos:")
        print(available.to_string(index=False))
        print(f"\n💾 Tamaño de BD: {storage.get_database_size_mb():.2f} MB")


if __name__ == "__main__":
    main()
