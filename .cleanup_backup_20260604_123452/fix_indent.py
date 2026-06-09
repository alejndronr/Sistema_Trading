import re

with open('dashboard/app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Función que inyecta el código con la indentación exacta
def replacer(match):
    indent = match.group(1) # Capturamos los espacios originales antes del "def"
    func_lines = [
        "def style_pnl(val):",
        "    try:",
        "        if isinstance(val, str):",
        "            cv = val.replace('$', '').replace('%', '').replace(' ', '').strip()",
        "            num = float(cv)",
        "        else:",
        "            num = float(val)",
        "        color = '#EF4444' if num < 0 else '#00FF7F'",
        "    except:",
        "        color = '#FFFFFF'",
        "    return f'color: {color}; font-weight: bold;'"
    ]
    # Reconstruimos el bloque respetando estrictamente los espacios
    return "\n".join([indent + line for line in func_lines])

# Buscamos la función y aplicamos el reemplazo
pattern = re.compile(r'^([ \t]*)def style_pnl\(val\):.*?return [^\n]+', re.DOTALL | re.MULTILINE)
content = pattern.sub(replacer, content, count=1)

with open('dashboard/app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Indentación reparada con precisión milimétrica.")
