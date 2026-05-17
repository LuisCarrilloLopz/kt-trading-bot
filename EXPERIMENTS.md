# Experimentos y decisiones de diseño

Este proyecto nace de la revisión crítica de `trading-bot-project` V2.3. Cada cambio
documentado abajo corrige un bug concreto o cambia una decisión que se mostró
sub-óptima en los entrenamientos previos.

---

## V1.0 — "Clean slate" (baseline kt)

**Fecha:** 2026-05-16
**Baseline previo:** `trading-bot-project` V2.3 (LSTM + "DSR" + randomize_start + clean eval)

---

### Bugs corregidos del baseline

#### F1 — `truncated` ahora se reporta correctamente

`step()` distingue entre **terminación** (ruina) y **truncamiento** (fin del CSV o
fin de episodio por `MAX_EPISODE_STEPS`).

PPO/GAE usan `V(s_T)=0` sólo en terminación; en truncamiento bootstraps con la
estimación del crítico. Antes, todos los fines de episodio se reportaban como
`done=True, truncated=False` → V(s_T)=0 incluso al llegar al fin del CSV → sesgo
sistemático en la estimación de valor de los últimos pasos.

```python
# Antes: return obs, reward, done, False, info
# Ahora: return obs, reward, terminated, truncated, info
#   - terminated=True sólo en ruina (net_worth ≤ 0.5 * initial_balance)
#   - truncated=True al fin del CSV o al alcanzar MAX_EPISODE_STEPS
```

#### F2 — Fees aplicadas sobre **notional**, no sobre `gross_return`

En el baseline, cerrar un short con pérdidas hacía `gross_return * (1 - fee)` sobre
un valor que podía ser negativo, lo que **reducía la pérdida** (el agente
recibía descuento por perder).

```python
# Ahora — fee sobre notional bruto en ambas patas
self.cash += colateral + pnl - abs(buyback_notional) * TRADING_FEE
```

#### F3 — Reward sin exploit por inactividad

El "DSR" del baseline dividía por una vol móvil que **decaía si el agente no
operaba** (`step_ret = 0` durante varios pasos → `rolling_std_ret → 1e-8`).
Cualquier movimiento posterior daba reward máximo por puro artefacto numérico.
Además, el comentario decía "DSR" pero la fórmula implementada era un z-score
de un paso, no el Differential Sharpe Ratio de Moody-Saffell.

Se sustituye por un reward interpretable:

```
reward = REWARD_LOG_RET_SCALE * log_return            # 1% step → +1
       - REWARD_DRAWDOWN_COEF * drawdown²             # 50% DD → -1.25
       - REWARD_CHURN_COEF    * 1[position changed]   # cada cambio → -0.05
```

Acotado por `clip(-10, 10)`. Las constantes están en `src/core/config.py`.

#### F4 — Observation incluye position size, unrealized PnL y drawdown

Antes la obs sólo tenía `current_position ∈ {-1, 0, +1}` y `cash/initial_balance`.
Con volatility targeting activo, el `position_size` varía 10× entre regímenes y
el agente estaba ciego a su exposición real. Nuevas features:

| # | Feature | Rango típico |
|---|---|---|
| 7 | `position_value / initial_balance` | signed, [-1, 1] |
| 8 | `unrealized_pnl / initial_balance` | [-0.5, 0.5] |
| 9 | `drawdown` (vs peak del episodio) | [0, 1] |

Total: 10 features (antes 8). Todas pre-escaladas para que `VecNormalize` no
deba normalizar la obs (`norm_obs=False`).

#### F5 — Critic LSTM activado (`enable_critic_lstm=True`)

El reward es path-dependent (drawdown depende del peak histórico del episodio).
Sin LSTM en el crítico, V(s) se estima desde un único timestep y no puede
capturar contexto temporal → GAE ruidoso → updates inestables.

---

### Cambios estructurales

#### S1 — Multi-asset (BTC + ETH) en un solo env

`MultiAssetTradingEnv` recibe `dict[str, DataFrame]` y samplea un activo en cada
`reset()`. La LSTM ve regímenes muy distintos (BTC bull 2020-21, crash 2022,
ETH merge 2022, etc.) en lugar de memorizar la trayectoria de un único activo.
Single env, sin overhead de procesos.

#### S2 — `randomize_start` cubre todo el dataset, con `max_episode_steps` fijo

Antes: `max_start = len(df) // 2` (sólo la primera mitad se usaba como inicio).
Ahora: `max_start = len(df) - MAX_EPISODE_STEPS - 1`, con episodios de longitud
fija (4000 steps ≈ 6 semanas en velas 15m). Distribución uniforme sobre todo el
train set, sin sesgo regional.

#### S3 — `market_vol` pre-computado en el loader

Antes: `df.iloc[step-30:step]['log_ret'].std()` en cada step (O(30) + overhead
pandas). Ahora: columna pre-computada al cargar. Beneficio: ~2× velocidad de step.

#### S4 — `best_model` seleccionado por **Sharpe ratio** del equity curve

`EvalCallback` del baseline guardaba por `mean_reward`, que es un proxy ruidoso
del objetivo real (P&L ajustado por riesgo). `SharpeEvalCallback`:

1. Sincroniza `obs_rms`/`ret_rms` del train env al eval env (sin fuga).
2. Rolloutea `N_EVAL_EPISODES` episodios (con LSTM states tracking).
3. Reconstruye el equity curve a partir de `info["net_worth"]`.
4. Calcula Sharpe/Sortino/MDD/Return anualizados y los loguea a TB.
5. Guarda `best_model` y `vec_normalize.pkl` cuando Sharpe medio mejora.

---

### Hiperparámetros vs V2.3

| Parámetro | V2.3 | V1.0 (kt) | Motivo |
|---|---|---|---|
| `n_steps` | 2048 | **4096** | Más contexto temporal para LSTM |
| `batch_size` | 128 | **256** | Acompañar `n_steps` ↑ |
| `enable_critic_lstm` | False | **True** | F5 — reward path-dependent |
| `obs features` | 8 | **10** | F4 — position_value, unrealized_pnl, drawdown |
| `reward` | "DSR" z-score | log_ret + DD + churn | F3 — interpretable, sin exploit |
| `multi-asset` | sólo BTC | **BTC + ETH** | S1 — diversidad de regímenes |
| `randomize_start` | half-dataset | full minus episode len | S2 |
| `eval metric` | `mean_reward` | **Sharpe ratio** | S4 — alineado con objetivo |
| `norm_obs` | True | **False** | features pre-escaladas; sin drift |
| `MAX_EPISODE_STEPS` | None (full CSV) | **4000** | Episodios de longitud consistente |
| Termination handling | `done=True` siempre | `terminated` vs `truncated` | F1 — bootstrap correcto |
| `learning_rate` | `3e-4 → 0` lineal | igual | mantenido |
| `ent_coef` | 0.05 → 0.005 lineal | igual | mantenido |
| `n_epochs` | 10 | igual | mantenido |
| `gamma` | 0.99 (default) | **0.99 explícito** | sin cambio efectivo |
| `gae_lambda` | 0.95 (default) | **0.95 explícito** | sin cambio efectivo |

---

---

## V1.1 — "Derivative DD fix"

**Fecha:** 2026-05-16
**Archivos modificados:** `src/envs/trading_env.py`, `src/core/config.py`
**Baseline:** V1.0 (run abortado en `logs/tensorboard/v1.0-broken-dd/` a 1.08M steps)

### Diagnóstico desde TensorBoard (V1.0 @ 1.08M steps)

Análisis del log del run V1.0 cuando llevaba el 36% del entrenamiento:

| Métrica | Valor | Lectura |
|---|---|---|
| `train/explained_variance` | 0.97 | Crítico excelente |
| `train/clip_fraction` | 0.16 | PPO sano |
| `train/value_loss` | 0.008 | Convergido |
| **`eval/total_return`** | **−34.6 %** | Pierde 1/3 del capital |
| **`eval/max_drawdown`** | **49.5 %** | Roza el umbral de ruina |
| **`eval/sharpe_ratio`** | **−3.3** | Mejorando lento desde −4.4 |
| **`eval/mean_reward`** | **−1584** | Dominado por shaping erróneo |

PPO estaba sano (`explained_variance=0.97`); el bug era de **reward shaping**, no de optimización.

### Bug detectado

`REWARD_DRAWDOWN_COEF * drawdown²` se aplicaba **cada step** que el agente
estuviera en DD. Con un DD típico de 30% durante 3000 de los 4000 steps de un
episodio: penalty acumulado = `3000 × 5 × 0.09 = 1350`, que dominaba el ~85%
del reward total (vs. ~−41 del `log_ret` por una caída del 34%).

Consecuencias:
- El reward signal estaba sesgado a negativo: el techo de la política óptima posible aún era < 0.
- El agente no tenía incentivo asimétrico a salir rápido del DD (mismo coste por step independientemente).
- El crítico aprendió perfectamente que "incluso la mejor política pierde" → estancamiento.

### Fix aplicado

```python
# Antes (V1.0):
reward -= REWARD_DRAWDOWN_COEF * (drawdown ** 2)   # crónico, cada step en DD

# Después (V1.1):
new_drawdown = max(0.0, drawdown - self.prev_drawdown)
reward -= REWARD_DRAWDOWN_COEF * new_drawdown      # one-shot, sólo en aumento de DD
self.prev_drawdown = drawdown
```

Total DD penalty por episodio queda acotado por `REWARD_DRAWDOWN_COEF * max_drawdown_alcanzado`. Con coef=20, un MDD del 50% cuesta como máximo −10 reward (vs ~−1500 antes).

### Cambios

| Parámetro | V1.0 | V1.1 | Motivo |
|---|---|---|---|
| Fórmula DD | `coef * drawdown²` por step | `coef * max(0, ΔDD)` por step | One-shot vs crónico |
| `REWARD_DRAWDOWN_COEF` | 5 | **20** | Compensar la menor frecuencia de firing |
| `prev_drawdown` en state | — | **añadido** | Necesario para derivada |

El log V1.0 se conserva en `logs/tensorboard/v1.0-broken-dd/` para comparar contra V1.1 en TensorBoard.

---

---

## V1.2 — "Crítico ampliado"

**Fecha:** 2026-05-17
**Archivos modificados:** `src/core/config.py`
**Baseline:** V1.1 abortado a 1.31M (logs en `logs/tensorboard/v1.1/`)

### Diagnóstico V1.1 (1.31M steps) y backtest 2025

**TensorBoard (1.25M):**
- `eval/sharpe_ratio` cayó: pico +7.7 → estabilizado ~+1.04 (últ. 25%)
- `eval/max_drawdown` máximo: 36.7 % (vs V1.1 inicial ~12 %)
- **`train/explained_variance`: 0.50 → 0.22** (crítico empeorando)
- `train/value_loss` doblado: 0.10 → 0.24
- `train/entropy_loss` recuperó: -0.18 → -0.42 (re-exploración tras colapso de modo)

**Backtest 2025 (datos no vistos, ~3 meses):**

| | best_model BTC | best_model ETH | last_ckpt BTC | last_ckpt ETH |
|---|---|---|---|---|
| Return | −15.1 % | −14.3 % | −8.3 % | **−50.4 %** |
| vs B&H | −9.9 % | **+16.7 %** | −3.1 % | −19.4 % |
| MDD | 16.8 % | 18.1 % | 17.6 % | **50.6 %** |
| Trades | 112 | 226 | 499 | 828 |
| NEUTRAL % | 94 % | 89 % | 78 % | 54 % |

Conclusión:
- `best_model` es **defensivo** (89-94 % NEUTRAL), no aporta alpha direccional.
- `last_ckpt` confirmó la degradación: 4-5× más activo, casi ruina en ETH (MDD 50 %).
- El criterio "best por Sharpe" salvó el modelo de la degradación posterior.
- El crítico subdimensionado explica el patrón: GAE ruidoso → policy oscila → no converge.

### Cambios

| Parámetro | V1.1 | V1.2 | Motivo |
|---|---|---|---|
| `lstm_hidden_size` | 128 | **256** | Duplicar capacidad LSTM (actor + crítico) para modelar reward path-dependent |
| `net_arch.vf` | [64, 64] | **[128, 128]** | 4× capacidad MLP del crítico |
| `net_arch.pi` | [64, 64] | [64, 64] | Sin cambio (actor estable) |
| `vf_coef` | 0.5 | **1.0** | Más peso al value loss en el gradiente |
| `MAX_EPISODE_STEPS` | 4000 | **2000** | Velas son **horarias** (no 15min), 4000 = 5.5 meses (demasiado); 2000 = ~3 meses |
| `CANDLES_PER_YEAR` (metrics) | 35040 (15m) | **8760 (1h)** | Bug detectado: Sharpe reportados estaban inflados ×√4 = 2 |

### Hipótesis a validar en V1.2

1. `train/explained_variance` debería mantenerse > 0.5 todo el run, no degradarse a 0.2.
2. `train/value_loss` debería bajar (no doblarse).
3. `eval/sharpe_ratio` debería estabilizarse en un valor positivo sin oscilaciones de >2 puntos.
4. La política debería tomar **más posiciones direccionales** (no quedarse 90 % NEUTRAL) si efectivamente aprende timing.

Si V1.2 sigue degradando el crítico, el problema no es de capacidad sino de **señal de reward**: pasaríamos a V1.3 con reward shaping diferente o features adicionales.

---

---

## V1.3 — "Más información, mismo modelo"

**Fecha:** 2026-05-17
**Archivos modificados:** `src/data/loader.py`, `src/envs/trading_env.py`
**Baseline:** V1.2 abortado a 1.57M (logs en `logs/tensorboard/v1.2/`)

### Diagnóstico V1.2 (1.57M steps)

V1.2 amplió el crítico (LSTM 256, vf [128,128], vf_coef 1.0) buscando estabilizar `explained_variance`. Resultado mixto:

**El crítico mejoró:**
- `train/explained_variance`: 0.16 → 0.34 (en V1.1 cayó 0.50 → 0.22 al mismo punto)
- `train/value_loss`: 0.13 vs V1.1 0.24
- `train/entropy_loss`: -0.49 vs V1.1 -0.42 (no colapsó)

**Pero la política empeoró:**
- `eval/sharpe_ratio` (último): −2.49 (V1.1: +0.52 corregido)
- `eval/total_return` (último): −17.7 % (V1.1: +20.4 %)
- `eval/max_drawdown` (último 25%): 21.0 % (V1.1: ~13 %)
- `train/approx_kl`: 0.039 (V1.1: 0.016) — updates demasiado agresivos
- `train/clip_fraction`: 0.19 (V1.1: 0.11)

**Best_model V1.2:** Sharpe máx +2.58 (escala corregida) vs V1.1 best +3.86 → V1.2 final es PEOR.

### Hipótesis confirmada

El bottleneck **no era capacidad del crítico**. Ahora V(s) se modela mejor pero la política sigue sin alpha. Conclusión: **las features no contienen información direccional suficiente** o la policy explora mal el espacio.

### Cambios

| Cambio | V1.2 | V1.3 |
|---|---|---|
| `N_FEATURES` | 10 | **18** (+8) |
| Arquitectura | igual | igual (LSTM 256, vf [128,128], vf_coef 1.0) |
| Reward | igual | igual (sin cambios) |

**Nuevas features añadidas:**

| # | Feature | Fuente | Rango típico | Por qué |
|---|---|---|---|---|
| 10 | `macd_norm` | MACD line / close × 100 | ±1 | Momentum medio plazo |
| 11 | `macd_hist_norm` | MACD - signal / close × 100 | ±1 | Slope del momentum |
| 12 | `atr_norm` | ATR(14) / close × 100 | 0.5-3 | Volatilidad realizada |
| 13 | `hour_sin` | sin(2π·hour/24) | [-1, 1] | Estacionalidad intradía |
| 14 | `hour_cos` | cos(2π·hour/24) | [-1, 1] | Estacionalidad intradía |
| 15 | `log_ret_lag1` | log_ret(t-1) × 100 | ±5 | Reversión/momentum 1h |
| 16 | `log_ret_lag5` | log_ret(t-5) × 100 | ±5 | Reversión/momentum 5h |
| 17 | `log_ret_lag10` | log_ret(t-10) × 100 | ±5 | Reversión/momentum 10h |

Todas pre-calculadas en `loader.py` para mantener `step()` O(1).

### Hipótesis a validar en V1.3

1. **Si V1.3 mejora `eval/sharpe_ratio` y `eval/total_return`** → confirmado que el bottleneck era información. Las features de momentum/temporales aportaban señal direccional faltante.
2. **Si V1.3 NO mejora** → el problema es el reward shape (no incentiva timing) o el horizonte temporal (LSTM no captura patrones más largos que `n_steps`). → V1.4 con reward asimétrico.

### Incompatibilidades

Best_models y vec_normalize de V1.0-V1.2 **NO son cargables en V1.3** (obs shape distinto: 10 → 18). Para backtest comparativo, hay que mantener el código de cada versión por separado (o checkout a un tag). Recomendado: tagear V1.2 antes de seguir si quieres poder volver a evaluar sus modelos.

---

## Backlog (V1.4+)

### Prioridad 1
- [ ] **Multi-seed runs** — entrenar con seeds {13, 42, 77} y reportar media ± std de Sharpe en eval. PPO tiene varianza brutal entre seeds; un solo run no es señal.
- [ ] **Validación reproducibilidad** — dos runs con misma seed deben dar idénticos resultados (sanity check del seeding).

### Prioridad 2
- [ ] **Dict obs space + `norm_obs_keys`** — separar features de mercado (que sí podrían normalizarse) de features de portfolio (que no deben).
- [ ] **SubprocVecEnv (4 workers)** — paralelismo wall-clock; 4 envs distintos sampleando assets independientemente.
- [ ] **Borrow fee para shorts** — 0.01%/8h sobre notional, replicando Binance.
- [ ] **Rebalanceo intra-posición** — ajustar `position_size` cuando la vol cambia significativamente, sin cerrar/reabrir (ahorrar fees).

### Prioridad 3
- [ ] **Asset embedding** — input categórico (BTC/ETH/...) para que la policy pueda diferenciar.
- [ ] **Más features** — MACD, ATR, hora del día, día de la semana.
- [ ] **DSR real de Moody-Saffell** — implementación completa para comparar con el reward actual.
- [ ] **Backtest 2025 automatizado** — script `src/backtest.py` que carga `best_model` y reporta CAGR vs buy-and-hold.
