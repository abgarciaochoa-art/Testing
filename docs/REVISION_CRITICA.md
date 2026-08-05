# Revisión crítica adversarial — consolidación

Fecha: 2026-08-05. Cuatro revisores adversariales auditaron el repo en busca de
defectos que hagan mentir a un backtest. Este documento consolida la
verificación de sus 12 hallazgos: cada uno fue re-verificado leyendo el código
y, cuando traía snippet, reproducido antes de tocar nada. Resultado global:
**11 CONFIRMADOS** (9 arreglados, 2 pendientes de diseño) y **1 DESCARTADO**.

Suite final tras los arreglos: **1.281 tests en verde** (`python3 -m pytest -m
"not network" -p no:cacheprovider -q`).

---

## CRÍTICO

### C1. `earnings_alpha/pipeline.py` (run_event_pipeline, ~línea 855) — sin filtro de universo PIT — CONFIRMADO, ARREGLADO

- **Defecto**: el docstring promete "universo PIT + calendario", pero
  `run_event_pipeline` pasaba `normalize_events(ctx.events)` tal cual a
  features, modelo, rejilla y estudio de eventos. `EventContext` no tenía
  campo de universo y ningún paso comprobaba pertenencia al índice.
- **Escenario de resultado falso**: con el dataset real
  (`load_consensus_events` sobre `consenso_master.parquet`) entran las
  historias completas PRE-inclusión de los miembros actuales. Verificado
  contra el parquet: TSLA tiene 165 eventos, 87 anteriores a su inclusión
  (2020-12-21); NFLX 34 pre-2010; MRNA 13. El universo efectivo es "empresas
  que ACABARÁN entrando al S&P 500": selección condicionada al futuro que
  infla retorno medio por evento y hit rate.
- **Arreglo**: nuevo parámetro opcional `universe: UniverseProvider` en
  `run_event_pipeline`. Tras `normalize_events`, los eventos se filtran a
  emisores con pertenencia PIT en su `event_date` (vía `members_on`, con
  caché por fecha); los descartes quedan en
  `params["n_events_outside_universe"]`. Sin universo, el resultado lo
  declara en `params["universe_filter"] = "SIN FILTRO: ..."` (aviso, no
  error, para no romper el banco sintético). Verificado funcionalmente con
  un proveedor de prueba (44 eventos descartados, resto del pipeline
  intacto).

## GRAVE

### G1. `earnings_alpha/factors/surprise.py` (_pit_consensus, línea ~385) — consenso del día del anuncio AMC admitido — CONFIRMADO (reproducido), ARREGLADO

- **Defecto**: el filtro era `as_of < tradable_date`. Para AMC/UNKNOWN el
  `tradable_date` es la sesión siguiente, así que la foto de consenso con
  `as_of == día del anuncio` (capturada en la pasada de cierre, ya revisada
  hacia el actual) pasaba el filtro y se convertía en el "consenso previo".
- **Reproducción**: evento AMC 2024-05-16 21:00 UTC, actual 2.00, consenso
  pre-anuncio 1.50, foto del día del anuncio 1.98 → el código devolvía
  `consensus=1.98, surprise=0.02` en lugar de `1.50 / 0.50`. Sesgo
  correlacionado con la sesión y con la magnitud real de la sorpresa:
  colapsa SUE/DOUBLE/PEAD hacia 0 de forma no aleatoria.
- **Arreglo**: `_pit_consensus` corta ahora por la **fecha ET del anuncio**
  (`announced_at` → `to_naive_utc` + `eastern_offsets_for`), con fallback al
  `tradable_date` cuando `announced_at` falta o es NaT (comportamiento
  anterior, correcto para BMO/DMH). Verificado: AMC devuelve 1.50/0.50, BMO
  conserva la foto de la víspera, el fallback sin `announced_at` no cambia.

### G2. `earnings_alpha/stats/ic.py` (restrict_to_universe, línea ~492) — panel de pertenencia disperso casado por igualdad exacta — CONFIRMADO (reproducido), ARREGLADO

- **Defecto**: `membership_panel` devuelve por defecto un panel DISPERSO (una
  fila por snapshot de composición); `restrict_to_universe` casaba por
  igualdad exacta de fechas y ponía `keep=False` a toda sesión sin snapshot,
  descartándola en silencio. `pipeline.py` pasa exactamente ese panel
  disperso a `cross_sectional_ic`.
- **Reproducción**: junio de 2008, AAPL miembro ininterrumpido: de 21
  sesiones de entrada solo 15 sobrevivían (`is_member` es True en las
  descartadas). La IC se calculaba sobre una submuestra sesgada hacia días de
  reconstitución, con `n_periods` menor y t de Newey-West distorsionado.
- **Arreglo**: forward-fill de la pertenencia sobre las fechas del panel
  antes de stackear (misma semántica PIT que `engine._membership_matrix`).
  Tras el arreglo las 21 fechas sobreviven. Protege a todos los llamantes,
  no solo a `pipeline.py`.

### G3. `earnings_alpha/backtest/engine.py` (línea ~407) — el retorno del hueco de cotización se destruye — CONFIRMADO (reproducido), ARREGLADO

- **Defecto**: `rcc = adjc.pct_change(fill_method=None)` da NaN también en la
  sesión de REAPERTURA tras un hueco (precio previo NaN), y el bucle diario
  convierte todo NaN de posición viva en retorno 0: el movimiento acumulado
  en la suspensión no se difiere, desaparece. Los huecos preceden típicamente
  reaperturas con caídas severas (preludio de delisting): sesgo direccional
  que infla la pata larga.
- **Reproducción**: cartera con un nombre suspendido 4 sesiones que reabre a
  −20%: bruto acumulado +1,60% con `held_gap_days=5`, cuando la contabilidad
  correcta da ≈ −8,5%.
- **Arreglo**: los retornos de tenencia del bucle (`rcc`, `ron`) se calculan
  sobre `adjc.ffill()` (la reapertura realiza `precio/último_conocido − 1`;
  el hueco solo difiere el P&L); `valid_price`, `can_trade`, `last_valid` y
  la sigma de costes (`sig_m`) siguen usando la serie SIN puentear, y
  `held_gap_days` sigue auditando los días sin precio. Tras el arreglo el
  mismo escenario da −8,40% con `held_gap_days=5`.

### G4. `earnings_alpha/backtest/engine.py` (rama next_open, línea ~585) — posición viva sin `open` pierde el retorno del día — CONFIRMADO (reproducido), ARREGLADO

- **Defecto**: en día de ejecución con `execution='next_open'`, una posición
  viva cuyo `open` falta ese día (pero cuyo `close` existe) recibía
  `r_on=0` y `r_id=0`, y al día siguiente la deriva usa `adjc[t+1]/adjc[t]`:
  el retorno close(t−1)→close(t) desaparecía permanentemente, sin rastro en
  `held_gap_days` (que solo miraba `rcc`).
- **Reproducción**: nombre al ~25% de peso perdiendo 5%/día con `open` NaN:
  la cartera reportaba ≈ −1% total. Tras el arreglo reporta −53,6%.
- **Arreglo**: en día de ejecución, para nombres con `ron` no finito pero
  retorno cierre-a-cierre finito, el retorno c2c se atribuye al tramo
  overnight de la posición vieja (`r_id` queda 0; `can_trade` ya impedía
  operar el nombre). Combinado con G3 (el `ron` puentea el cierre previo),
  ningún retorno de posición viva se evapora ya en esta rama.

### G5. `earnings_alpha/stats/validation.py` (purge_and_embargo, línea ~281) — embargo solo tras el último bloque de test — CONFIRMADO (reproducido), ARREGLADO

- **Defecto**: el embargo usaba `test_end = test_g1.max()`: con test NO
  contiguo (el caso normal de `CombinatorialPurgedCV`, que pasa k grupos
  separados como un único test) las observaciones inmediatamente posteriores
  a cada bloque anterior al último NO se embargaban y entraban al train.
- **Reproducción**: `test = [10:20) ∪ [60:70)`, `embargo=5`: las posiciones
  20–24 seguían en train; solo 70–74 se embargaban. Con CPCV N=12, k=2, 55
  de 66 splits tienen test no adyacente: con etiquetas serialmente
  correlacionadas, la distribución de Sharpes OOS por trayectoria sale
  optimista (criterio 11 de ARCHITECTURE evaluado sobre trayectorias
  infladas). `PurgedKFold` no estaba afectado.
- **Arreglo**: embargo tras el fin de CADA segmento (vectorizado con
  `np.unique(test_g1)` + `searchsorted`); es un superconjunto seguro del
  embargo por bloque (las posiciones interiores ya están excluidas por test
  o purga). Verificado: 20–24 y 70–74 fuera, 25 y 75 dentro; bloque único
  contiguo se comporta idéntico a antes.

### G6. `earnings_alpha/events/eventstudy.py` (_returns_matrix, línea ~122) — prioridad de `log_return` (retorno de precio) sobre `adj_close` — CONFIRMADO (reproducido), ARREGLADO

- **Defecto**: `_returns_matrix` prefería `log_return` asumiéndolo retorno
  total canónico, pero el panel sintético (`data/synthetic.py:1505`) define
  `log_return = log(close·split_factor/prev_close)`: retorno de PRECIO que
  incluye la caída mecánica del ex-dividendo — exactamente el sesgo que el
  propio docstring decía evitar.
- **Reproducción**: 25 tickers 2018-2021: CAAR(+30) = 0.0097 con
  `log_return` frente a 0.0118 con `adj_close` (−21 pb, ~18% del efecto).
  Los ex-dividendo sintéticos caen sistemáticamente en la ventana
  post-evento, doblando la curva CAAR/PEAD a la baja en el tramo de drift.
- **Arreglo**: prioridad invertida: `adj_close` (retorno total por
  construcción CRSP) > `log_return` > `close`, con el porqué documentado en
  el docstring. Los tests de eventstudy (que son cualitativos: Q5 > Q1,
  determinismo) pasan.

### G7. `earnings_alpha/data/estimates.py` (ExternalConsensusProvider, línea ~1033) — sesgo de supervivencia del dataset real, declarado pero no cuantificado ni bloqueado — CONFIRMADO (verificado empíricamente), PENDIENTE (documentación endurecida)

- **Defecto verificado contra el parquet**: 81,5% de las filas (46.432 de
  56.971) proceden del snapshot 2022-11-30; LEH, BSC, WAMUQ, ENE/ENRNQ, FNM,
  FRE, MER, WB, NCC no existen en el fichero; cobertura de los 445 miembros
  reales del índice a 2008-06-30: **59,6%** (vs ~93% del último snapshot).
  La cobertura decae hacia atrás eliminando exactamente a los quebrados de
  2008-2009: los eventos con peores retornos post-anuncio de la muestra.
- **Escenario de resultado falso**: cualquier backtest/estudio 1995-2021
  sobre este dataset infla retorno medio, hit rate y CAAR de quintiles bajos
  (una estrategia larga tras miss parece rentable porque sus peores
  contrapartidas no están en los datos).
- **Hecho ahora**: los docstrings de `ExternalConsensusProvider` y
  `CONSENSUS_PIT_WARNING` incluyen el dato duro medido (59,6% en 2008,
  tickers ausentes) y remiten al filtro de universo de C1
  (`run_event_pipeline(..., universe=...)`), que mitiga el sesgo de
  pre-inclusión aunque no puede resucitar los eventos ausentes.
- **PENDIENTE (diseño)**: cuantificación programática — un
  `coverage_vs_universe(universe)` en el proveedor y un umbral bloqueable
  (p. ej. `DataQualityError` con cobertura PIT < 85% del rango pedido) en
  `load_consensus_events`. Ningún filtro puede corregir el sesgo (los datos
  no existen): el objetivo es que el usuario no pueda ignorarlo.

### G8. `earnings_alpha/events/preevent.py` (InformedTradingScore.score, línea ~929) — z-score por cohorte con eventos futuros — CONFIRMADO (reproducido), PENDIENTE (limitación declarada)

- **Defecto**: `cohort_labels` fusiona fechas HACIA ADELANTE hasta
  `min_size` eventos; el z-score (y el demeaning por (cohorte, sector)) de
  un evento en T usa media y sigma de eventos anunciados días DESPUÉS de T.
  El score negociable en T no es computable en T.
- **Reproducción**: perturbar la feature de un evento del 2022-01-07 cambia
  el score de un evento del 2022-01-03 de −0.1349 a −0.2696.
- **Por qué no se arregla aquí**: el arreglo correcto (estandarización
  retrospectiva por ventana [T−k, T] o momentos de la cohorte anterior) es
  un cambio de diseño que altera el score en todo el banco sintético, sus
  métricas de detección validadas contra `leaked_event_ids()` y los pesos
  por defecto; excede el riesgo acotado de una consolidación.
- **Hecho ahora**: warning explícito en el docstring de la clase: el score
  debe tratarse como disponible el ÚLTIMO día de su cohorte, no en la
  medianoche de T.
- **PENDIENTE (diseño)**: opción (a) ventana retrospectiva, (b) momentos de
  la cohorte anterior, o (c) propagar `available_at = cierre de cohorte`
  para que `pit.assert_no_lookahead` lo detecte.

### G9. `earnings_alpha/events/preevent.py` (residualize_features, línea ~452) — OLS por cohorte con eventos futuros — CONFIRMADO, PENDIENTE (limitación declarada)

- **Defecto**: misma fuga intra-cohorte que G8 aplicada a la regresión de
  ortogonalización: el residuo de un evento en T depende de coeficientes
  estimados con eventos posteriores de la misma cohorte. Mitigado dentro de
  `SurpriseModel.evaluate` por la purga de spans; el uso directo de las
  features residualizadas como señal en fecha de evento hereda el
  look-ahead.
- **Hecho ahora**: warning explícito en el docstring.
- **PENDIENTE (diseño)**: regresión expansiva/retrospectiva dentro de la
  temporada o estimación en la cohorte anterior aplicada fuera de muestra.
  Mismo motivo de aplazamiento que G8 (comparten `cohort_labels`).

## MENOR

### M1. `earnings_alpha/data/shortinterest.py` (línea ~866) — `tierIdentifier` ausente defaulteaba al retardo corto — CONFIRMADO, ARREGLADO

- **Defecto**: `str(item.get("tierIdentifier", "T1") or "T1")` asignaba el
  retardo de publicación CORTO (14 días, Tier 1) a los registros sin tier,
  contra la política del propio módulo ("equivocarse hacia pronto fabrica
  look-ahead"). Acotado (todo el S&P 500 es Tier 1), pero si FINRA renombrara
  el campo TODAS las filas caerían en el default optimista sin error visible.
- **Arreglo**: default invertido al lado conservador: sin tier → `"OTC"`
  (28 días), con el porqué comentado en el código. El fixture de tests trae
  siempre `tierIdentifier`, así que el comportamiento con datos bien
  formados no cambia.

### M2. `earnings_alpha/stats/performance.py` (sharpe_ratio, línea ~517) — `horizon` aceptado y silenciosamente ignorado con Mertens/Lo — CONFIRMADO, ARREGLADO

- **Defecto**: `horizon` solo se usaba con `ci_method='bootstrap'`; con
  `'mertens'` (default) y `'lo'` se ignoraba sin aviso: IC ~√h veces
  demasiado estrecho y PSR inflado con retornos solapados, mientras
  `performance_summary` propagaba `horizon` dando impresión de dependencia
  tratada.
- **Arreglo**: con `horizon > 1` y `ci_method` ∈ {mertens, lo} se emite
  `UserWarning` explícito y el método queda etiquetado
  `sharpe_<m>_iid_sin_correccion_h<h>` en `SharpeResult.method` (y por tanto
  en la fila del tearsheet): la ausencia de corrección queda declarada, que
  es lo que exige §1.3. La corrección del SE por varianza de largo plazo
  (Newey-West sobre los excesos) queda como mejora posible, no imprescindible
  una vez el método está etiquetado y la fila bootstrap (que sí corrige)
  convive en el mismo tearsheet.

## DESCARTADO

### D1. `earnings_alpha/pit/asof.py` (tradable_date, línea 308) — DMH tratado como sesión del mismo día — DESCARTADO

- **Alegación**: un anuncio DMH nominal de las 12:30 ET queda disponible
  desde la medianoche de su propia sesión; un backtest que ejecute EN LA
  APERTURA capturaría el salto del anuncio.
- **Por qué se descarta**: el comportamiento es un contrato documentado y
  testeado, y **ningún consumidor del repo ejecuta en la apertura de la
  fecha de la señal**:
  - `tests/test_pit.py:478` afirma explícitamente "DMH → misma sesión"; el
    docstring de `tradable_date` documenta la regla y su razón (el mercado
    está abierto: el resto de la sesión es operable, lo cual es cierto para
    ejecución al cierre).
  - `CrossSectionalBacktest` decide en `d` y ejecuta en `d+1` (apertura o
    cierre siguientes); `stats.forward_returns` impone `execution_lag=1`;
    `EventBacktest` tiene `dmh_policy` obligatoria (`exclude`/`delay`) que
    excluye o retrasa al cierre los DMH en τ=0
    (`tests/test_event_backtest.py:178`).
  - La línea de ARCHITECTURE ("AMC/DMH-post-cierre → siguiente sesión") lee
    como "AMC, y DMH anunciado tras el cierre", consistente con la política.
  - Magnitud: 23 de 47.779 eventos del dataset real (~0,05%).
  El riesgo residual existe solo para un consumidor EXTERNO que ejecute al
  open de la fecha de señal sin retardo, algo que el repo prohíbe en todas
  sus rutas. No es un defecto que haga mentir a un backtest de este repo.

---

## Resumen de cambios en código

| Fichero | Cambio |
|---|---|
| `earnings_alpha/factors/surprise.py` | Corte PIT del consenso por fecha ET del anuncio (fallback: tradable_date); advertencia de supervivencia con cifras medidas |
| `earnings_alpha/stats/ic.py` | `restrict_to_universe` forward-fillea la pertenencia sobre las fechas del panel |
| `earnings_alpha/backtest/engine.py` | Retornos de tenencia sobre precios puenteados (ffill); atribución c2c a posiciones vivas sin `open` en día de ejecución next_open; `sig_m`/`held_gap_days` siguen sobre la serie sin puentear |
| `earnings_alpha/stats/validation.py` | Embargo tras cada segmento de test (CPCV con test no contiguo) |
| `earnings_alpha/events/eventstudy.py` | `_returns_matrix` prioriza `adj_close` sobre `log_return` |
| `earnings_alpha/pipeline.py` | `run_event_pipeline(universe=...)`: filtro PIT de pertenencia con auditoría en `params` |
| `earnings_alpha/data/shortinterest.py` | Tier FINRA ausente → retardo largo (OTC, 28 días) |
| `earnings_alpha/stats/performance.py` | Warning + etiqueta de método cuando `horizon>1` se ignora (Mertens/Lo) |
| `earnings_alpha/events/preevent.py` | Warnings de look-ahead intra-cohorte en `InformedTradingScore` y `residualize_features` (arreglo de diseño pendiente) |
| `earnings_alpha/data/estimates.py` | Advertencia de supervivencia con cifras medidas y remisión al filtro de universo |

## Pendientes documentados

1. **G7**: cuantificación y umbral bloqueable de cobertura de
   `consenso_master.parquet` contra `SP500Universe` por año.
2. **G8/G9**: estandarización y residualización retrospectivas (o
   `available_at = cierre de cohorte`) en `events/preevent.py`.
3. **M2 (mejora)**: SE de Sharpe con corrección Newey-West para
   `ci_method='mertens'` con `horizon>1`, además del aviso ya emitido.
