# DISCLAIMER — Sistema de Trading Algorítmico ZimaBlade V6

> **Leer antes de aumentar capital, activar margin o recomendar este sistema a terceros.**

---

## 1. Sobre el Edge Real del Sistema

Este sistema es **experimental en producción activa**. No existe evidencia out-of-sample
suficiente para afirmar que tiene un edge positivo consistente ajustado por comisiones.

### MetaLabeler (Filtro ML)

| Metrica | Valor actual | Referencia |
|---------|-------------|-----------|
| ROC-AUC en train | ~0.56-0.62 | >0.60 para senal debil |
| ROC-AUC en test | No medido aun | Necesita >30 senales reales |
| Trades paper reales | 0 (sistema operativo desde 2026) | Minimo 50-100 para validar |

Un ROC-AUC de 0.56 es marginalmente mejor que azar (0.50). Esto significa que el modelo
filtra ruido, pero no elimina trades perdedores. La estrategia de base (MeanReversion,
TrendFollowing) debe tener edge propio: el ML amplifica o reduce sizing, no crea edge
donde no existe.

### CycleDetector

El CycleDetector identifica fases del ciclo macro de Bitcoin (BEAR_DEEP, ACCUMULATION,
BULL_EARLY, etc.) basandose en el analisis de 4 ciclos historicos de Bitcoin (2013, 2017,
2020, 2024). Esta es una heuristica estadistica, NO una ley fisica:

- Los ciclos pasados reflejan condiciones regulatorias, de adopcion y liquidez muy diferentes
  a las actuales. Un quinto ciclo no tiene por que seguir el mismo patron.
- El conviction_score mide coherencia interna de los indicadores, no probabilidad de acierto.
- La clasificacion de fase puede tardar 2-5 dias en confirmar cambios de regimen reales
  (persistence filter), causando entradas tardias en movimientos abruptos.

### Estrategias

Las estrategias (MeanReversion, TrendFollowing, Breakout, MomentumScalp) son implementaciones
cuantitativas de heuristicas conocidas. Su performance depende criticamente de las condiciones:

- MeanReversion funciona en mercados de rango. En tendencias fuertes produce perdidas.
- TrendFollowing funciona en tendencias. En mercados laterales produce muchos falsos positivos.
- Breakout tiene una tasa de falsos positivos alta (~70-80%) sin volumen institucional confirmado.
- Ninguna ha sido validada con mas de 50 trades reales en este sistema especifico.

---

## 2. Sobre el Backtest

El backtest walk_forward.py y backtesting/engine.py tienen limitaciones conocidas:

1. **In-sample contamination:** Los parametros de los indicadores fueron ajustados sobre datos
   historicos que incluyen los mismos periodos del backtest. Esto infla artificialmente las metricas.

2. **Slippage y comisiones aproximadas:** Se usa COMMISSION_RATE=0.001 (0.1%) y SLIPPAGE=0.001.
   En mercados de baja liquidez, el slippage real puede ser 3-5x mayor.

3. **Sin datos de libro de ordenes:** Las estrategias asumen ejecucion al precio de cierre de vela.
   En la realidad esto puede no ser posible en el tamano de posicion calculado.

4. **El backtest no prueba la infraestructura:** Un Profit Factor 1.2 en backtest no garantiza que
   el motor en vivo ejecute esas mismas senales por diferencias de timing, datos faltantes o bugs.

**Regla de validacion del proyecto:**
Un sistema se considera candidato a aumentar capital cuando:
- Tiene >= 50 trades reales en paper (no backtest)
- Profit Factor paper >= 1.10 durante >= 30 dias consecutivos
- audit_real_signals.py confirma ROC-AUC en senales reales >= 0.55
- Maximo drawdown paper <= 8%

Actualmente NINGUNA de estas condiciones se ha cumplido.

---

## 3. Sobre Margin Trading

MARGIN_ENABLED=false en produccion. Si se activa:

- Los shorts en altcoins implican riesgo de liquidacion ilimitado si el precio sube contra la posicion.
- Binance Cross Margin puede liquidar TODAS las posiciones de la cuenta, no solo la causante.
- El sistema no tiene logica de gestion de funding rate. En shorts > 8h, el funding puede erosionar el PnL.

No activar MARGIN_ENABLED=true hasta:
- Completar los criterios de validacion del punto 2
- Entender completamente como funciona Cross Margin en Binance
- Backtestear estrategias de short con datos reales de funding rate

---

## 4. Sobre la Infraestructura

- Intel Atom E3950: en alta carga con multiples senales simultaneas puede haber latencia que afecte el timing.
- La conexion a internet del ZimaBlade no tiene redundancia garantizada. Una caida de red mientras
  hay posiciones abiertas puede impedir que el sistema gestione SL/TP.
- Las posiciones en paper portfolio (PostgreSQL) persisten, pero en live trading las ordenes en Binance
  deben monitorizarse manualmente si el motor cae.

---

## 5. Tabla de Riesgos

| Riesgo | Probabilidad | Impacto | Mitigacion actual |
|--------|-------------|---------|-------------------|
| Edge positivo no confirmado | Alto | Capital | Paper trading obligatorio |
| Overfitting del backtest | Alto | Confianza falsa | walk_forward con OOS |
| Fallo de infraestructura | Medio | Posiciones no gestionadas | Systemd auto-restart |
| Liquidacion (margin) | Alto si activo | Total | MARGIN_ENABLED=false |
| Degradacion del modelo ML | Medio | Edge reducido | auto_retrain mensual |
| Datos stale del exchange | Bajo | Trade en precio incorrecto | Drift check 1.5% |

---

## 6. Uso Responsable

Este codigo se publica con fines educativos y de investigacion personal. No es asesoramiento
financiero. Las perdidas de capital son posibles y probables durante las fases de calibracion.

Antes de aumentar capital por encima del inicial ($300-1000), ejecutar:

```bash
# Auditoria de senales reales (requiere >= 30 trades paper)
python backtesting/audit_real_signals.py

# Walk-forward con datos OOS mas recientes
python backtesting/walk_forward.py --auto --report

# Diagnostico del estado actual
python diagnose_signals.py
```

---

*Ultima actualizacion: 2026-06-24 | Version del sistema: V6.0.0-AdaptiveCycle*
*Ver tambien: CALIBRATION.md para documentacion de umbrales de filtros.*
