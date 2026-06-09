import itertools
import multiprocessing
import concurrent.futures
from typing import Dict, List, Any, Tuple
import pandas as pd
from config.logging_config import get_logger
logger = get_logger(__name__)
import numpy as np

from backtesting.engine import BacktestEngine
from config.settings import TrendFollowingParams, STRATEGIES, BACKTEST
from strategies.trend_following import TrendFollowingStrategy

def run_single_backtest(params: Dict[str, Any], dfs: Dict[str, pd.DataFrame], symbol: str) -> Dict[str, Any]:
    """
    Worker function para ejecutar un backtest individual.
    Como corre en un proceso separado, podemos modificar STRATEGIES libremente.
    """
    # 1. Aplicar parámetros
    tf = params["timeframe"]
    df_train = dfs[tf]
    
    # Mutar configuración global (seguro porque es un proceso hijo)
    STRATEGIES.trend_following.min_adx = params["adx"]
    STRATEGIES.trend_following.sl_atr_multiplier = params["sl_atr"]
    STRATEGIES.trend_following.tp1_rr_ratio = params["tp1_rr"]
    STRATEGIES.trend_following.trailing_ema_period = params["trailing_ema"]
    
    # 2. Inicializar motor
    engine = BacktestEngine(initial_capital=BACKTEST.initial_capital if hasattr(BACKTEST, 'initial_capital') else 1000.0)
    
    # 3. Correr backtest silencioso
    strategy = TrendFollowingStrategy(symbol, timeframe=tf, params=STRATEGIES.trend_following)
    
    # Logging mute (structlog uses logging levels, so we just pass for now or capture output,
    # but we can't easily disable specific loggers without diving into stdlib logging config)
    # We will just let them log to the file.
    
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
