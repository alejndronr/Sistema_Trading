import re

with open('dashboard/app.py', 'r', encoding='utf-8') as f:
    code = f.read()

# 1. Desactivamos la función vieja de CoinGecko renombrándola
code = code.replace("def get_live_prices(", "def get_live_prices_old(")

# 2. Inyectamos el nuevo Motor CCXT
binance_funcs = """
# --- MOTOR BINANCE CCXT ---
@st.cache_data(ttl=3600)
def get_binance_usdc_pairs():
    try:
        import ccxt
        exchange = ccxt.binance()
        markets = exchange.load_markets()
        # Filtramos para obtener solo mercados Spot contra USDC
        pairs = [s for s, m in markets.items() if m.get('quote') == 'USDC' and m.get('spot')]
        
        # Garantizamos que SUI y otras principales estén visibles
        if 'SUI/USDC' not in pairs: pairs.append('SUI/USDC')
        return sorted(pairs) + ["Otro..."]
    except Exception as e:
        return ["BTC/USDC", "ETH/USDC", "SOL/USDC", "SUI/USDC", "RENDER/USDC", "LINK/USDC", "Otro..."]

@st.cache_data(ttl=60)
def get_live_prices(symbols: list) -> dict:
    try:
        import ccxt
        exchange = ccxt.binance()
        # Descargamos todos los tickers de Binance de golpe (mucho más rápido que uno a uno)
        tickers = exchange.fetch_tickers()
        prices = {}
        for sym in symbols:
            if sym in tickers and tickers[sym].get('last') is not None:
                prices[sym] = float(tickers[sym]['last'])
            else:
                prices[sym] = 0.0
        return prices
    except Exception as e:
        return {s: 0.0 for s in symbols}
# --------------------------
"""

if "MOTOR BINANCE CCXT" not in code:
    code = code.replace("load_dotenv()", "load_dotenv()\n" + binance_funcs)

# 3. Sustituimos el desplegable manual por el dinámico de Binance
code = re.sub(
    r'm_sym_sel\s*=\s*st\.selectbox\("Par",\s*\[.*?\]\)',
    'm_sym_sel = st.selectbox("Par", get_binance_usdc_pairs())',
    code
)

with open('dashboard/app.py', 'w', encoding='utf-8') as f:
    f.write(code)

print("✅ Motor CoinGecko anulado. Sistema conectado a Binance (CCXT) con todos los pares USDC dinámicos.")
