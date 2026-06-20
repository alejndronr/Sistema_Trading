# CALIBRATION.md — ZimaBlade V6 Filter Thresholds

**Propósito:** Documentar por qué cada umbral de score existe, qué backtest lo valida,
y cuándo revisarlo. Evita que futuras sesiones añadan filtros sin entender el contexto.

**Regla de oro:** Si añades un filtro nuevo, documéntalo aquí ANTES de hacer el commit.

---

## Historial de Calibraciones

| Fecha | Versión | Fase de mercado | Responsable |
|-------|---------|-----------------|-------------|
| 2026-06-20 | V6.0.0 | ACCUMULATION (post-ATH Oct 2025) | Antigravity AI |

---

## Arquitectura de Filtros Actual

El sistema usa **puntuación acumulada (OR ponderado)**, NO filtros en serie (AND).
Esto evita el problema conocido como *filter stacking*:

```
P(trade_AND) = P(F1) × P(F2) × ... × P(Fn)
             = 0.30 × 0.25 × 0.33 × 0.40 ≈ 0.001  ← casi imposible

P(trade_SCORE) = P(score >= umbral)
               ≈ 0.08 - 0.15 en mercado normal  ← razonable
```

---

## Umbrales de Score por Estrategia y Fase

### MeanReversion (MR)

Propósito: capturar rebotes estadísticos en sobreventa. Solo válida en LONG.

| Fase | Min Score | Justificación |
|------|-----------|---------------|
| BEAR_DEEP | 55 | Bear da pocas señales; mercado necesita menos confirmación estructural |
| BEAR_RECOVERY | 58 | Ligeramente más estricto: reducir entradas prematuras al alza |
| ACCUMULATION | 55 | Fase óptima para MR; amplitud de señales moderada-alta |
| BULL_EARLY | 50 | Mercado ayuda; la tendencia macro absorbe errores |
| BULL_MATURE | 45 | MR apenas se usa en BULL_MATURE (TrendFollowing domina) |
| BULL_LATE | 60 | Más estricto: mercado sobrecomprado, MR es trampa frecuente |
| DISTRIBUTION | 65 | Muy estricto: distribución = vendedores institucionales activos |

**Componentes del score (máx ~120 puntos):**

| Señal | Puntos | Condición |
|-------|--------|-----------|
| RSI extremo | 30 | RSI < 28 |
| RSI sobreventa | 20 | RSI 28-35 |
| ADX range | 20 | ADX < 20 |
| ADX moderado | 12 | ADX 20-25 |
| En BB lower | 20 | Dist < 0.5% |
| Cerca BB lower | 12 | Dist < 2% |
| StochRSI extremo | 15 | Stoch < 0.15 |
| MACD cross | 15 | cruce al alza |
| MACD growing | 8 | histograma creciendo |
| Z-score extremo | 20 | zscore < -2.0 (solo BEAR) |
| Hurst MR | 10 | Hurst < 0.45 |
| Bonus ACCUM | 8 | fase ACCUMULATION |
| SR Zone | 15 | precio en soporte institucional |
| FIB Support | 10 | retroceso Fibonacci cerca |

**Penalizaciones (no son vetos absolutos):**

| Condición | Penalización | Razón |
|-----------|-------------|-------|
| RSI > 35 Y zscore > -1.5 en BEAR | -20 pts | No hay sobreventa real |
| phase_strength > 0.90 en BEAR | -25 pts | Bear muy fuerte; alta tasa de falsas rupturas |

---

### TrendFollowing (TF)

Propósito: seguir tendencias confirmadas en fases alcistas del ciclo macro.

| Fase | Min Score | Justificación |
|------|-----------|---------------|
| BULL_MATURE | 65 | Fase óptima; momentum confirma la dirección |
| BULL_EARLY | 72 | Más estricto: tendencia aún no confirmada |
| Otras | 78 | Fuera de BULL, TF tiene alta tasa de falsos positivos |

**Veto duro mantenido:**
- `t_stat < 1.0` → veto absoluto (requiere tendencia estadísticamente válida)
- Razón: TF sin significancia estadística es "trading por intuición", error que queremos evitar

---

### Breakout

Propósito: capturar expansiones de volatilidad tras periodos de compresión.

| Condición | Umbral | Justificación |
|-----------|--------|---------------|
| Score mínimo | 65 | Necesita squeeze + volumen + precio sobre BB |
| Volumen institucional | OBLIGATORIO | Sin absorption u OBV accel, el breakout es trampa del 90% de las veces |

---

### MomentumScalp

Propósito: capturar aceleraciones de precio en velas cortas.

| Condición | Umbral | Justificación |
|-----------|--------|---------------|
| Score mínimo | 65 | Alta precisión necesaria para scalp |
| OBV accelerating | OBLIGATORIO | Sin volumen acelerando, el scalp falla en 80% de casos |

---

## Reglas de Filtros Sistémicos

### Restricción Horaria (MR_HOURS)

| Fase | Horas UTC | Razón |
|------|-----------|-------|
| BEAR_DEEP | 0-24 (todas) | Sin restricción en bear: cualquier señal extrema es válida |
| ACCUMULATION | 0-16 | Evitar horas de baja liquidez asiática tardía |
| BULL | 0-24 (todas) | Liquidez alta en bull; cualquier hora es válida |

> **Regla de diseño:** La hora NUNCA es un veto absoluto. Es un bonus pequeño de calidad.

### Circuit Breakers (no modificar sin backtest)

| Parámetro | Valor | Razón |
|-----------|-------|-------|
| COOLDOWN_SL_MIN | 90 min | Evitar reentradas inmediatas tras SL |
| CB_SL_CONSECUTIVE | 3 | Suspender símbolo tras 3 SL seguidos |
| CB_SL_SUSPEND_MIN | 240 min | 4h de enfriamiento |
| CB_DAILY_REDUCE | -3% | Reducir sizing si PnL diario cae 3% |
| CB_DAILY_PAUSE | -5% | Pausar entradas si PnL diario cae 5% |

---

## Telemetría del Embudo (Signal Funnel)

El motor trackea cuántas señales pasan cada etapa y lo reporta en el resumen diario:

```
Evaluaciones totales → pasaron régimen → generaron señal → trades ejecutados
      144                  89 (62%)          12 (8%)           2 (1.4%)
```

**Alertas automáticas:**
- 2+ días sin trades → advertencia en resumen diario
- 5+ días sin trades → alerta crítica pinneada en Telegram

Si se activa la alerta de 5 días, ejecutar:
```bash
source venv/bin/activate
python diagnose_signals.py
```

---

## Próximas Revisiones

| Trigger | Acción |
|---------|--------|
| Cambio de fase de ciclo confirmado | Revisar umbrales de MR para la nueva fase |
| Cada 3 meses | Walk-forward validation completo |
| PF < 0.60 en walk-forward | Revisar umbrales; posible overfitting |
| >300 trades en backtest 2 años | Umbrales demasiado bajos; subirlos |
| <30 trades en backtest 2 años | Filtro bloqueante oculto; usar diagnose_signals.py |

**Próxima revisión sugerida:** 2026-09-20 (o antes si BTC confirma nuevo BULL_EARLY)

---

## Referencias

- [live_engine.py](live_engine.py) — AdaptiveSignalSelector, _mean_reversion, _trend_following
- [risk/regime_filter.py](risk/regime_filter.py) — CycleDetector, umbrales de fase
- [diagnose_signals.py](diagnose_signals.py) — script de diagnóstico
- walkthrough.md — historial de cambios y rationale de cada fase
