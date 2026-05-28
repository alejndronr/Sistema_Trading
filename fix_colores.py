import re
path = "dashboard/app.py"

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

new_style_func = """def style_pnl(val):
    try:
        if isinstance(val, str):
            clean_val = val.replace("$", "").replace("%", "").replace(" ", "").strip()
            num = float(clean_val)
        else:
            num = float(val)
        
        # Rojo si es negativo, Verde si es positivo o cero
        color = "#EF4444" if num < 0 else "#00FF7F"
    except:
        color = "#FFFFFF"
        
    return f"color: {color}; font-weight: bold;" """

# Buscamos la función antigua y la reemplazamos
pattern = re.compile(r'def style_pnl\(val\):.*?return [^\n]+', re.DOTALL)
content = pattern.sub(new_style_func, content, count=1)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

print("✅ Lógica de colores de PnL corregida. Rojo para pérdidas, Verde para ganancias.")
