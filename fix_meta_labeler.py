"""
fix_meta_labeler.py — Parchea build_feature_matrix para pandas 3.x
===================================================================
El error: fillna() ya no acepta arrays numpy como valor por defecto.
La solución: separar el caso array-default del scalar-default.

Ejecutar en el servidor:
    python fix_meta_labeler.py
"""
from pathlib import Path
import sys

META_PATH = Path("/home/trading/sistema_trading/ml/meta_labeler.py")

if not META_PATH.exists():
    print(f"✗ No encontrado: {META_PATH}")
    sys.exit(1)

with open(META_PATH) as f:
    src = f.read()

# ── Fix 1: col() con default array — pandas 3.x no acepta ndarray en fillna ──
# El problema: col("atr", price * 0.015) pasa un array como default
# La solución: si la columna no existe, devolver el array directamente
# Si existe, usar fillna con escalar 0 y luego reemplazar los NaN con el array

OLD_COL = '''        def col(name: str, default: float = 0.0) -> np.ndarray:
            """Extrae columna con default seguro."""
            if name in df.columns:
                return df[name].fillna(default).values.astype(float)
            return np.full(n, default, dtype=float)'''

NEW_COL = '''        def col(name: str, default=0.0) -> np.ndarray:
            """
            Extrae columna con default seguro.
            pandas 3.x: fillna() no acepta arrays numpy, solo escalares.
            Si default es array (ej. price * 0.015), lo manejamos manualmente.
            """
            if name in df.columns:
                arr = df[name].values.astype(float)
                if isinstance(default, np.ndarray):
                    # Reemplazar NaN con el array de defaults
                    nan_mask = np.isnan(arr)
                    arr[nan_mask] = default[nan_mask]
                    return arr
                else:
                    # Escalar: fillna seguro
                    return df[name].fillna(float(default)).values.astype(float)
            # Columna no existe
            if isinstance(default, np.ndarray):
                return default.astype(float)
            return np.full(n, float(default), dtype=float)'''

if OLD_COL in src:
    src = src.replace(OLD_COL, NEW_COL)
    print("✅ col() — fix pandas 3.x fillna con ndarray")
else:
    print("⚠️  col() no encontrado exacto — buscando variante")
    # Buscar la función col directamente
    if "def col(name: str, default: float = 0.0)" in src:
        src = src.replace(
            "def col(name: str, default: float = 0.0)",
            "def col(name: str, default=0.0)"
        )
        # Reemplazar la lógica interna
        OLD_INNER = '''            if name in df.columns:
                return df[name].fillna(default).values.astype(float)
            return np.full(n, default, dtype=float)'''
        NEW_INNER = '''            if name in df.columns:
                arr = df[name].values.astype(float)
                if isinstance(default, np.ndarray):
                    nan_mask = np.isnan(arr)
                    arr[nan_mask] = default[nan_mask]
                    return arr
                return df[name].fillna(float(default)).values.astype(float)
            if isinstance(default, np.ndarray):
                return default.astype(float)
            return np.full(n, float(default), dtype=float)'''
        if OLD_INNER in src:
            src = src.replace(OLD_INNER, NEW_INNER)
            print("✅ col() — fix aplicado (variante)")

# ── Fix 2: macd_h usa col() anidado — aplanar para evitar tipo incorrecto ────
OLD_MACD = '        macd_h = col("macd_histogram", col("macd_hist", 0.0))'
NEW_MACD = '''        # pandas 3.x: no anidar col() — resolver en dos pasos
        if "macd_histogram" in df.columns:
            macd_h = col("macd_histogram", 0.0)
        else:
            macd_h = col("macd_hist", 0.0)'''

if OLD_MACD in src:
    src = src.replace(OLD_MACD, NEW_MACD)
    print("✅ macd_h — col() anidado aplanado")

# ── Fix 3: regime_raw.map().fillna() — puede recibir NaN de map ──────────────
OLD_REGIME = '        regime = regime_raw.map(regime_map).fillna(0.0).values.astype(float)'
NEW_REGIME = '''        regime_mapped = regime_raw.map(regime_map)
        # fillna con escalar explícito (pandas 3.x compatible)
        regime = regime_mapped.fillna(0.0).values.astype(float)'''

if OLD_REGIME in src:
    src = src.replace(OLD_REGIME, NEW_REGIME)
    print("✅ regime fillna — explícito")

# Backup y guardar
import shutil
backup = META_PATH.with_suffix(".py.bak")
shutil.copy2(META_PATH, backup)
print(f"   Backup: {backup}")

with open(META_PATH, "w") as f:
    f.write(src)

# Verificar sintaxis
import ast
try:
    ast.parse(src)
    print("✅ Sintaxis OK")
except SyntaxError as e:
    print(f"❌ Error de sintaxis: {e}")
    # Restaurar backup
    shutil.copy2(backup, META_PATH)
    print("   Backup restaurado")
    sys.exit(1)

print(f"\n✅ meta_labeler.py parcheado correctamente")
print(f"   Ahora ejecuta: python ml/retrain_model.py --initial --min-trades 30")
