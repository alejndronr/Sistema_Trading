import re

archivo = 'ml/retrain_model.py'
with open(archivo, 'r') as f:
    codigo = f.read()

# Buscamos la función _load_trades_from_db y la reemplazamos entera
nueva_funcion = """def _load_trades_from_db(engine_ignorado, since):
    import pandas as pd
    from sqlalchemy import create_engine
    # Forzamos una conexión síncrona local absoluta
    sync_engine = create_engine("postgresql://trading_user:tu_password_seguro@127.0.0.1:5432/trading_db")
    try:
        df = pd.read_sql("SELECT * FROM trades", sync_engine)
        if 'entry_time' in df.columns:
            df['entry_time'] = pd.to_datetime(df['entry_time'])
            if since:
                df = df[df['entry_time'] >= pd.to_datetime(since)]
        return df
    except Exception as e:
        print(f"Error leyendo base de datos (ML): {e}")
        return pd.DataFrame()
"""

# Reemplazamos la función original (usamos expresiones regulares para atraparla toda)
codigo_modificado = re.sub(r'def _load_trades_from_db.*?return df\n', nueva_funcion, codigo, flags=re.DOTALL)

with open(archivo, 'w') as f:
    f.write(codigo_modificado)

print("✅ Función de base de datos parcheada para usar conector síncrono local.")
