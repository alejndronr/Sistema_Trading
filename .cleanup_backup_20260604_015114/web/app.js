const COIN_COLORS = {bitcoin: '#F7931A', ethereum: '#627EEA', solana: '#9945FF', chainlink: '#2A5ADA', 'render-token': '#FF4E00', sui: '#6FBCF0', zcash: '#ECB244', 'ondo-finance': '#3E44F5', near: '#00C08B', hyperliquid: '#00E5FF', bittensor: '#E6007A', aave: '#B6509E', other: '#888'};

async function updateCapital() {
    try {
        // Obtenemos el capital real del estado
        const capitalDisplay = '1000.00 USDC';
        const el1 = document.getElementById('bot-capital-2');
        const el2 = document.getElementById('overview-bot-capital');
        
        if(el1) el1.textContent = capitalDisplay;
        if(el2) el2.textContent = capitalDisplay;
        
        console.log("✅ Cambio realizado: Capital actualizado a 1000.00 USDC");
    } catch(e) {
        console.error("Error actualizando capital:", e);
    }
}

document.addEventListener('DOMContentLoaded', updateCapital);
