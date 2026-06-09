import argparse
import sys
from pathlib import Path

# Configurar path para imports locales
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.logging_config import setup_logging, get_logger

from backtesting.engine import HyperparameterOptimizer
from data.storage import OHLCVStorage
from config.settings import ALL_PAIRS

def main():
    setup_logging()
    logger = get_logger(__name__)
    
    parser = argparse.ArgumentParser(description="Hyperparameter Optimizer - Motor de Búsqueda de Alfa")
    parser.add_argument("--symbol", type=str, default="BTC/USDC", help="Par a optimizar")
    args = parser.parse_args()

    logger.info("Iniciando Optimización V4", symbol=args.symbol)
    
    # Cargar datos desde SQLite
    storage = OHLCVStorage()
    
    dfs = {}
    timeframes = ["1h", "4h"]
    for tf in timeframes:
        logger.info(f"Cargando datos para {args.symbol} en {tf}...")
        df = storage.load_ohlcv(args.symbol, tf)
        if df is None or df.empty:
            logger.error(f"No hay datos para {args.symbol} en {tf}.")
            return
        dfs[tf] = df
        
    optimizer = HyperparameterOptimizer(dfs=dfs, symbol=args.symbol)
    
    # Definir Parameter Space según plan (Grid Atómico)
    param_grid = {
        "strategy": ["meta"],
        "timeframe": ["4h"],
        "meta_hurst_trend_threshold": [0.52, 0.55, 0.60],
        "meta_hurst_range_threshold": [0.40, 0.45],
        "tf_min_adx": [20.0, 25.0],
        "tf_sl_atr_multiplier": [1.5, 2.0]
    }
    
    # 1. Optimizar In-Sample
    valid_results = optimizer.optimize(param_grid)
    
    if not valid_results:
        logger.warning("Ninguna configuración superó la restricción de Max Drawdown <= 15%.")
        return
        
    logger.info("=== TOP 3 PARÁMETROS IN-SAMPLE ===")
    top_3 = valid_results[:3]
    for i, res in enumerate(top_3):
        params = res['params']
        metrics = res['result'].metrics
        logger.info(f"Rank {i+1} | PF: {metrics.profit_factor:.2f} | WR: {metrics.win_rate*100:.1f}% | DD: {metrics.max_drawdown_pct*100:.1f}% | Params: {params}")
        
    logger.info("\n=== PRUEBA OUT-OF-SAMPLE ===")
    for i, res in enumerate(top_3):
        logger.info(f"Probando Rank {i+1} OOS...")
        oos_res = optimizer.validate_out_of_sample(res['params'])
        metrics_oos = oos_res['result'].metrics
        logger.info(f"OOS Rank {i+1} | PF: {metrics_oos.profit_factor:.2f} | WR: {metrics_oos.win_rate*100:.1f}% | DD: {metrics_oos.max_drawdown_pct*100:.1f}%")

if __name__ == "__main__":
    main()
