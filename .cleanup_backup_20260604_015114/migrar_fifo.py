import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
db_url = os.environ.get("DATABASE_URL", "").replace("+asyncpg", "").replace("+aiosqlite", "")
engine = create_engine(db_url)

# A. Añadir la columna 'type' a la tabla existente sin perder datos
with engine.connect() as conn:
    try:
        conn.execute(text("ALTER TABLE manual_trades_v2 ADD COLUMN type VARCHAR DEFAULT 'COMPRA'"))
        conn.commit()
        print("Columna 'type' añadida a PostgreSQL.")
    except Exception as e:
        print("La columna 'type' ya existe.")

# B. Parchear el HTML visualmente para añadir el selector y la métrica de Ganancias Realizadas
html_path = "web/dashboard.html"
with open(html_path, "r", encoding="utf-8") as f:
    html = f.read()

target_select = '<div class="form-grid">\n            <div class="form-group">\n              <label class="form-label">Criptomoneda</label>'
replace_select = '''<div class="form-grid">
            <div class="form-group">
              <label class="form-label">Tipo de Operación</label>
              <select class="form-select" id="add-type">
                <option value="COMPRA">COMPRA (Añadir activo)</option>
                <option value="VENTA">VENTA (Reducir posición)</option>
              </select>
            </div>
          </div>
          <div class="form-grid">
            <div class="form-group">
              <label class="form-label">Criptomoneda</label>'''

if "Tipo de Operación" not in html:
    html = html.replace(target_select, replace_select)

html = html.replace('<label class="form-label">Precio de compra (USD)</label>', '<label class="form-label">Precio de ejecución (USD)</label>')
html = html.replace('<th>Precio compra</th>', '<th>Precio ejec.</th>')
html = html.replace('<tr><th>Activo</th><th>Cantidad</th><th>Precio compra</th>', '<tr><th>Activo</th><th>Tipo</th><th>Cantidad</th><th>Precio ejec.</th>')

target_metric = '<div class="metric-sub" id="total-pnl-pct">0.00%</div>\n          </div>'
replace_metric = '<div class="metric-sub" id="total-pnl-pct">0.00%</div>\n          </div>\n          <div class="metric">\n            <div class="metric-label">Beneficio Realizado</div>\n            <div class="metric-value" id="realized-pnl">$0.00</div>\n            <div class="metric-sub">Asegurado (FIFO)</div>\n          </div>'

if "Beneficio Realizado" not in html:
    html = html.replace(target_metric, replace_metric)

with open(html_path, "w", encoding="utf-8") as f:
    f.write(html)
print("Dashboard HTML actualizado.")
