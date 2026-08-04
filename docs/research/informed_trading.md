# Detección de la huella de negociación informada antes de anuncios de resultados

**Ámbito.** Revisión de literatura orientada a implementación para el módulo `earnings_alpha.events`
(ángulo B del proyecto), en particular para `PreEventFeatures` según `docs/ARCHITECTURE.md` §3.5.
Todo lo que sigue usa **exclusivamente datos públicos**: precios y volúmenes consolidados, cadenas de
opciones, short interest agregado de FINRA, ficheros de transparencia OTC/ATS de FINRA y formularios
Form 4 ya presentados en EDGAR. El objetivo es **detectar la huella estadística** que deja la
negociación informada en variables observables, no acceder a información material no pública.
Ninguna señal descrita aquí requiere ni admite información privilegiada.

**Fecha de compilación:** 2026-08-04.

---

## 0. Nota metodológica sobre las fuentes de esta revisión

El entorno de desarrollo tiene el egress restringido y `WebFetch` devolvió **HTTP 403 para todos los
dominios** (incluidos arXiv, SSRN, escholarship y repositorios institucionales). La revisión se ha
construido a partir de **26 búsquedas web distintas** cuyos resultados sintetizan resúmenes y
abstracts. Consecuencia práctica, y hay que asumirla con honestidad:

- Las **fórmulas** marcadas como *derivadas* han sido reconstruidas analíticamente aquí y además
  **verificadas numéricamente** (comparación contra la definición directa y simulación Monte Carlo).
  Son fiables, y donde la verificación ha contradicho a la práctica habitual de la literatura se dice
  explícitamente (§3.3 y §5.3).
- Las **cifras de magnitud de efecto** proceden de abstracts. Las que no he podido contrastar con el
  texto completo van marcadas con ⚠ y **deben verificarse contra el paper primario** antes de usarse
  para calibrar nada. Están recogidas también en `docs/OPEN_QUESTIONS.md` (sección de verificación
  bibliográfica) cuando procede.

**Dos hallazgos propios de esta revisión**, ambos con consecuencias directas de implementación y
ambos comprobados numéricamente:

1. **§3.3** — La corrección de autocorrelación al z-score de turnover debe usar el tamaño muestral
   efectivo **exacto** de un AR(1), no la aproximación asintótica al uso. Con `rho = 0,8` y `k = 10`,
   suponer independencia infla el z-score por un factor **2,3**.
2. **§5.3** — La factorización clásica de la verosimilitud de PIN (la que implementa buena parte de
   la literatura) **desborda `float64` con los recuentos de operaciones típicos del S&P 500**
   (falla ya con B = S = 3.000). La forma correcta es log-sum-exp sobre los tres regímenes.

---

## 1. Resumen ejecutivo: tabla de señales

Convención de signo del repo (`ARCHITECTURE.md` §3.4): **valor mayor = más alcista**. La columna
"signo" indica la relación esperada entre la señal *tal y como se define en su sección* y el retorno
del evento / la sorpresa.

| # | Señal | Signo esperado | Horizonte | Magnitud típica reportada | Fiabilidad | Datos mínimos |
|---|---|---|---|---|---|---|
| 1 | `turnover_zscore`, `abnormal_volume_k` | **Ambiguo** (magnitud, no dirección) | [T−20, T−1] → [T, T+1] | Predice *magnitud* de la reacción, no signo. Prima de anuncio ≈ +60 pb/mes para anunciantes (Frazzini–Lamont) ⚠ | Media | OHLCV diario |
| 2 | SUV (volumen no explicado por el retorno) | **Positivo** sobre el drift | [T−10, T−1] → [T+1, T+60] | Relación positiva documentada con PEAD; magnitud del orden de 1 pp trimestral ⚠ | Media | OHLCV diario |
| 3 | `order_imbalance_proxy` (Lee–Ready) | **Positivo** | [T−5, T−1] → [T, T+1] | Flujo institucional más positivo antes de sorpresas positivas (Campbell–Ramadorai–Schwartz) | Alta con intradía, **baja** con datos diarios | Tick data (ideal) / OHLCV (degradado) |
| 4 | PIN | Condicional (no direccional) | Trimestral | 10 pp de PIN ≈ 2,5 %/año de retorno esperado (EHO 2002) ⚠ — impugnado por Duarte–Young | **Baja** para eventos; sirve como control | Recuentos de trades firmados |
| 5 | VPIN | Condicional (no direccional) | Intradía–días | Sin poder predictivo incremental sobre volatilidad (Andersen–Bondarenko) | **Baja** | Tick data |
| 6 | `pre_event_car_k` | **Positivo** con reversión | [T−20, T−1] → [T, T+1] | Fuertemente confundido con momentum, prima de anuncio y sobre-extrapolación | Media, requiere ortogonalización | OHLCV + índice |
| 7 | `short_interest_delta` | **Negativo** | Quincenal → 1–3 meses | Decil alto vs. bajo: −1,16 %/mes (Boehmer–Jones–Zhang, flujo diario) ⚠ | Alta a horizonte largo, **baja** a T−1 por el retardo de publicación | FINRA short interest |
| 8 | `off_exchange_share_delta` | **Negativo o nulo** | Semanal → semanas | Sin efecto direccional robusto; teoría (Zhu 2014) predice *caída* de la cuota dark ante flujo informado | **Baja** como señal de evento (retardo 2–4 semanas) | FINRA OTC Transparency |
| 9 | `insider_net_buy_form4` (oportunista) | **Positivo** | 1–12 meses | 82 pb/mes VW y 180 pb/mes EW (Cohen–Malloy–Pomorski); >100 pb/mes alfa 4F (Ali–Hirshleifer) ⚠ | Alta a horizonte largo; **nula** en [T−30, T−1] por blackout | Form 4 EDGAR |
| 10 | `vol_spread` (Cremers–Weinbaum) | **Positivo** | 1 semana | Q5−Q1 ≈ 50 pb/semana, decreciente en el tiempo ⚠ | Media; débil en eventos *programados* | Cadena de opciones con IV |
| 11 | `iv_skew_25delta` | **Negativo** | 1–6 meses | Smirk más pronunciado vs. más plano: −10,9 %/año ajustado por riesgo ⚠ | Media | Cadena de opciones |
| 12 | `put_call_volume_ratio` / O/S | **Negativo** (¡contraintuitivo!) | 1 semana | Decil bajo − decil alto de O/S = +0,34 %/semana (Johnson–So) ⚠ | Media | Volumen de opciones + acciones |
| 13 | `oi_buildup_calls/puts` | Débil | Días | Sin poder incremental una vez controlado el retorno pre-evento (literatura de opciones/desacuerdo) ⚠ | Baja | Open interest OCC |

**Lectura de conjunto.** Ninguna señal individual es un detector. La literatura fiable dice tres
cosas: (a) el flujo *firmado* es lo informativo, no el volumen bruto; (b) para eventos **programados**
como los resultados, casi todas las señales pierden potencia respecto a eventos no programados;
(c) casi toda la aparente predictibilidad de las señales de volumen desaparece al controlar por el
retorno pre-evento. El diseño correcto es un **compuesto ortogonalizado** con validación temporal
estricta, no una regla sobre una única variable.

---

## 2. Marco: qué se puede detectar con datos públicos

El modelo de referencia es Kyle (1985) / Glosten–Milgrom (1985): un agente informado reparte su orden
en el tiempo para no revelar su señal, y el creador de mercado aprende del flujo agregado. Las
huellas observables son entonces:

1. **Volumen** por encima de lo normal (el informado añade volumen).
2. **Desequilibrio direccional** del flujo (el informado opera en un solo sentido).
3. **Deriva de precio** anticipada (el aprendizaje del creador de mercado mueve el precio antes del
   anuncio).
4. **Migración de venue** (opciones si hay coste de préstamo, lit vs. dark según riesgo de ejecución).
5. **Deterioro de liquidez** (spread, profundidad, impacto).

El punto (5) es donde la evidencia empírica **contradice al modelo**. Kacperczyk y Pagnotta (2019),
usando más de 5.000 operaciones documentadas en expedientes de litigio de la SEC (1995–2015),
encuentran que en los días en que hay informados presentes el volumen y la volatilidad son
anormalmente altos pero **los spreads son un 10 % (acciones) y un 20 % (opciones) MÁS ESTRECHOS**
de lo normal ⚠. La explicación: los informados usan órdenes limitadas (proveen liquidez en lugar de
consumirla) y **eligen deliberadamente días con alto volumen no informado** para esconderse.

Dos consecuencias de diseño, ambas importantes:

- **No usar el ensanchamiento del spread como detector.** El signo empírico es el contrario.
- **El volumen anormal está parcialmente endogeneizado**: el informado elige días que ya tienen
  volumen alto, así que parte del "run-up" que mediríamos es la *causa* de que operen, no su
  *consecuencia*. Esto reduce la relación señal/ruido de todas las features de volumen bruto.

---

## 3. Run-up de volumen y z-score de turnover

### 3.1 Definiciones base

Turnover diario:

```
turn[i,t] = volume[i,t] / shares_outstanding[i,t]
```

`shares_outstanding` debe ser **point-in-time** (recompras, emisiones, splits). Si no se dispone de
él PIT, el sustituto aceptable es el volumen en dólares normalizado por capitalización, o
simplemente comparar el volumen contra su propia distribución histórica (opción 3.3), que no
necesita denominador.

Ajinkya y Jain (1989) documentan que los errores de predicción del volumen bruto están fuertemente
sesgados a la derecha, mientras que la **transformación logarítmica es aproximadamente normal**.
Por tanto se trabaja siempre con:

```
x[i,t] = ln( turn[i,t] + c ),   c = 0.000255
```

La constante `c = 0.000255` es la usada en la literatura de eventos de volumen para evitar `ln(0)`
en días sin negociación.

### 3.2 Ajuste de mercado

El volumen tiene un factor común muy fuerte (días de FOMC, vencimientos trimestrales, rebalanceos de
índice, festivos parciales). Siguiendo a Garfinkel y Sokobin (2006), se usa turnover **ajustado por
mercado**, restando la mediana de la sección cruzada del universo PIT de ese día:

```
xadj[i,t] = x[i,t] - median_{j in universe(t)} x[j,t]
```

La mediana, no la media: en un día de anuncios masivos la media está dominada por los propios
anunciantes.

### 3.3 Ventana base y z-score

Ventana de estimación: **[T−70, T−21]** en días de sesión (50 sesiones), y se **excluyen** de ella
las ventanas [−2, +2] de cualquier evento de resultados anterior del mismo emisor, de días de
inclusión/exclusión del índice y de splits. La ventana base termina en T−21 para no solaparse con la
ventana de detección.

```
mu[i]    = mean( xadj[i,t] , t in base )
sigma[i] = std ( xadj[i,t] , t in base , ddof=1 )
```

**Turnover z-score sobre la ventana pre-evento de k sesiones:**

```
zbar[i,k] = ( mean_{tau=-k..-1} xadj[i,tau] - mu[i] ) / se[i,k]
```

El denominador es donde casi todo el mundo se equivoca. Bajo iid sería `sigma[i]/sqrt(k)`, pero el
volumen diario tiene autocorrelación de primer orden alta (ρ del orden de 0,4–0,7 en acciones
grandes). Dos correcciones válidas:

**(a) Empírica (recomendada).** Calcular en la ventana base la media móvil de k días de `xadj` y
tomar su desviación típica muestral:

```
se[i,k] = std( rolling_mean_k( xadj[i, base] ), ddof=1 )
```

Esto absorbe la autocorrelación sin modelarla. Requiere `len(base) >= 2k + 10`; si no, lanzar
`InsufficientHistory`.

**(b) Tamaño muestral efectivo AR(1).** Si `x` es AR(1) con autocorrelación `rho`, la varianza de la
media de k observaciones **consecutivas** es exactamente:

```
Var( media_k ) = ( sigma^2 / k^2 ) * [ k + 2 * sum_{j=1..k-1} (k - j) * rho^j ]

n_eff(k, rho)  = k^2 / ( k + 2 * sum_{j=1..k-1} (k - j) * rho^j )
se[i,k]        = sigma[i] / sqrt( n_eff(k, rho) )
```

**Usar la forma exacta, no la asintótica.** La aproximación asintótica que circula habitualmente,
`n_eff ≈ k*(1-rho)/(1+rho)`, es una expresión de *k→∞* y para las k que nos interesan (5, 10, 20)
está materialmente sesgada, además de poder devolver `n_eff < 1`, que carece de sentido. Verificación
por simulación Monte Carlo (2·10⁶ observaciones por celda):

| rho | k | `n_eff` empírico | exacto (fórmula) | asintótico |
|---|---|---|---|---|
| 0,3 | 10 | 5,783 | **5,765** | 5,385 |
| 0,6 | 5 | 1,914 | **1,911** | 1,250 |
| 0,6 | 20 | 5,530 | **5,517** | 5,000 |
| 0,8 | 5 | 1,382 | **1,381** | 0,556 (¡< 1!) |
| 0,8 | 10 | 1,844 | **1,842** | 1,111 |

Nótese la magnitud del problema que se corrige: con `rho = 0,8` y `k = 10`, la hipótesis iid daría
`n_eff = 10` cuando el valor real es **1,84**. El z-score ingenuo estaría inflado por un factor
`sqrt(10/1,84) ≈ 2,3`. Esto por sí solo convierte un t de 1,0 en un t de 2,3 y fabrica significancia
donde no la hay.

Usar (a) por defecto; (b) con la forma exacta como *fallback* si el histórico no llega para (a).

### 3.4 Ratio de volumen anormal (`volume_runup`, `abnormal_volume_5/10/20d`)

Versión en ratio, robusta y directamente interpretable:

```
AVR[i,k] = mean_{tau=-k..-1} volume[i,tau] / median_{t in base} volume[i,t]
```

`AVR = 1` es normalidad; `AVR = 1.8` es un 80 % de volumen extra. Mediana en el denominador porque
la media del volumen la dominan tres o cuatro días atípicos.

Versión logarítmica, que es la que entra en modelos lineales:

```
abnvol[i,k] = ln( mean_{tau=-k..-1} volume[i,tau] ) - ln( median_{t in base} volume[i,t] )
```

Y `abnormal_volume_5d`, `abnormal_volume_10d`, `abnormal_volume_20d` son `zbar[i,k]` con k = 5, 10,
20 (usar el z-score, no el ratio, como feature de modelo: es comparable entre acciones).

### 3.5 SUV — Standardized Unexpected Volume

Garfinkel y Sokobin (2006) y Garfinkel (2009) aíslan el volumen **no explicado por el movimiento de
precio**, que es el candidato natural a proxy de divergencia de opinión / actividad informada. Se
estima en la ventana base:

```
x[i,t] = a[i] + b1[i] * Rpos[i,t] + b2[i] * Rneg[i,t] + e[i,t]

Rpos[i,t] = max(r[i,t], 0)
Rneg[i,t] = max(-r[i,t], 0)
```

La asimetría (dos coeficientes en lugar de uno sobre `|r|`) captura el hecho empírico de que el
volumen reacciona distinto a subidas y bajadas de la misma magnitud.

Para cada día de la ventana pre-evento se calcula el residuo **fuera de muestra** y se estandariza
con la desviación típica residual **dentro de muestra**:

```
SUV[i,tau] = ( x[i,tau] - (a[i] + b1[i]*Rpos[i,tau] + b2[i]*Rneg[i,tau]) ) / s_e[i]
```

Agregación pre-evento:

```
SUV[i,k] = ( sum_{tau=-k..-1} SUV[i,tau] ) / sqrt(k)
```

(mismo caveat de autocorrelación: si se quiere el z exacto, dividir por la desviación típica
empírica de la suma móvil de k días en la ventana base, no por `sqrt(k)`).

**Signo y horizonte.** Garfinkel y Sokobin encuentran que el volumen inesperado en la ventana de
anuncio se relaciona **positivamente** con el drift posterior (60 días). Como feature pre-evento el
signo es menos claro y debe estimarse, no imponerse.

### 3.6 EVIDENCIA CONTRARIA CRÍTICA: el volumen *baja* antes de los anuncios programados

Chae (2005, *Journal of Finance* 60(1), 413–442) es el resultado que invalida la intuición ingenua.
Encuentra que el **volumen acumulado DISMINUYE antes de anuncios programados**, y que la disminución
es **mayor cuanto mayor es la asimetría de información**. El mecanismo: los *discretionary liquidity
traders* (que sí pueden elegir cuándo operar) posponen sus operaciones hasta después del anuncio para
no ser seleccionados adversamente. En cambio, antes de anuncios **no programados** el volumen sube
drásticamente y apenas se relaciona con las proxies de asimetría informativa.

Implicaciones de implementación, y son de primer orden:

1. **La línea base correcta no es "un mes normal cualquiera", sino "la ventana pre-anuncio de este
   mismo emisor en trimestres anteriores"**. Si se compara la ventana [T−20, T−1] contra una base
   [T−70, T−21] genérica, la señal medirá sobre todo *cuánto se secó el volumen*, un efecto de
   liquidez, no de información. Recomendación operativa: calcular ambos, y añadir como feature
   adicional la diferencia respecto al mismo z del trimestre anterior del mismo emisor:
   `turnover_zscore_vs_own_prior_quarters = zbar[i,k] − mean(zbar[i,k] de los 4 eventos previos)`.
2. Un run-up positivo en un evento **programado** es mucho más anómalo que el mismo run-up en un
   evento no programado, porque el patrón incondicional es negativo. Esto **favorece** al detector,
   pero solo si la línea base está condicionada al tiempo-evento.
3. Chae también documenta que los creadores de mercado aumentan la sensibilidad del precio antes de
   *todos* los anuncios, extrayendo información de *timing* del libro. Es decir, parte del ajuste
   pre-anuncio ocurre **sin** volumen anormal.

### 3.7 Ratio de run-up (Keown–Pinkerton)

Métrica clásica de la literatura de fusiones, trasladable como diagnóstico de filtración:

```
runup_ratio[i] = CAR[i, -N, -1] / CAR[i, -N, +1]
```

En M&A la literatura clásica sitúa en torno al 40–50 % la fracción del movimiento total que ocurre
antes del anuncio ⚠. En resultados programados el valor incondicional debería ser mucho menor; un
`runup_ratio` alto **y del mismo signo que la sorpresa** es la firma más directa de anticipación.
Ojo: el ratio es inestable cuando `CAR[-N,+1] ≈ 0` — winsorizar o usar solo eventos con
`|CAR[-N,+1]| > 2 * sigma_AR`.

---

## 4. Desequilibrio de órdenes (order imbalance) y sus proxies

### 4.1 Referencia: Lee–Ready con datos intradía

Lee y Ready (1991) clasifican cada operación comparándola con el punto medio de la mejor cotización
vigente `M[t] = (bid + ask)/2`:

```
si P[t] > M[t]        -> compra iniciada (buy)
si P[t] < M[t]        -> venta iniciada (sell)
si P[t] == M[t]       -> regla del tick
```

**Regla del tick** (aplicable también sin cotizaciones): comparar `P[t]` con el último precio
*distinto*:

```
si P[t] > P_last_diff -> buy      (uptick)
si P[t] < P_last_diff -> sell     (downtick)
```

El paper original desfasa las cotizaciones 5 segundos (era de reporte manual); con datos modernos el
desfase correcto es 0 y debe determinarse empíricamente por mercado y por época — es un parámetro,
no una constante.

Medidas agregadas por día:

```
OIB_num[i,t]    = (n_buys - n_sells) / (n_buys + n_sells)
OIB_vol[i,t]    = (Vbuy - Vsell) / (Vbuy + Vsell)
OIB_dollar[i,t] = ($buy - $sell) / ($buy + $sell)
```

Agregado pre-evento:

```
OIB[i,k] = sum_{tau=-k..-1} (Vbuy - Vsell) / sum_{tau=-k..-1} (Vbuy + Vsell)
```

y su z-score contra la distribución de la ventana base, igual que en §3.3.

**Precisión.** Ellis, Michaely y O'Hara (2000) sobre datos de Nasdaq: la regla de cotización acierta
un **76,4 %**, la regla del tick un **77,66 %** y Lee–Ready un **81,05 %** ⚠. Todas las reglas
fallan sobre todo en operaciones **dentro** del spread, en operaciones grandes, en periodos de alto
volumen y en ECN — exactamente el subconjunto donde opera el informado. Chakrabarty, Pascual y
Shkilko documentan que los desequilibrios calculados con la regla del tick "carecen de precisión
suficiente" en la era electrónica. El error de clasificación **atenúa** el coeficiente de la señal
(sesgo hacia cero), no lo invierte, pero puede reducirlo a la mitad.

### 4.2 Bulk Volume Classification (BVC)

Easley, López de Prado y O'Hara proponen no clasificar operación a operación sino repartir el volumen
de una *barra* de forma probabilística:

```
Vbuy[tau]  = V[tau] * Phi( dP[tau] / sigma_dP )
Vsell[tau] = V[tau] - Vbuy[tau] = V[tau] * ( 1 - Phi( dP[tau] / sigma_dP ) )

dP[tau]   = P[tau] - P[tau-1]           (cambio de precio entre barras consecutivas)
sigma_dP  = std( dP ) estimada sobre la muestra de barras (p.ej. acción-mes)
Phi       = CDF de la normal estándar
```

En la versión de *Discerning Information from Trade Data* se sustituye `Phi` por la CDF de una
t de Student con ν grados de libertad, con ν estimado o fijado (los valores empleados en el paper
original son atípicamente bajos y hay que verificarlos ⚠). La versión normal es la que se recomienda
implementar por defecto: es más estable y la diferencia empírica es de segundo orden.

Desequilibrio BVC de la barra:

```
OIB_bvc[tau] = (Vbuy - Vsell) / V = 2 * Phi( dP[tau] / sigma_dP ) - 1
```

**Trampa matemática que hay que documentar en el código.** Con **barras diarias**,
`OIB_bvc[t]` es una **función determinista y monótona del retorno diario estandarizado**. Es decir:
a frecuencia diaria, el desequilibrio BVC no aporta *ninguna* información más allá del retorno; es un
retorno "aplastado" por una sigmoide. Su único valor añadido aparece al **ponderar por volumen**:

```
OIB_bvc[i,k] = sum_{tau=-k..-1} V[i,tau] * ( 2*Phi( dP[i,tau]/sigma_dP[i] ) - 1 )
               / sum_{tau=-k..-1} V[i,tau]
```

Esta sí es distinta del CAR pre-evento, porque pondera cada día por su volumen. Debe ortogonalizarse
frente a `pre_event_car_k` antes de usarse (§10.2), y hay que probar formalmente que aporta IC
incremental; si no lo aporta, se elimina.

### 4.3 Proxy con OHLCV: Close Location Value

Con solo OHLCV diario hay un proxy que **no** es función del retorno cierre-a-cierre y por tanto sí
aporta información independiente: la posición del cierre dentro del rango del día.

```
CLV[i,t] = ( (C - L) - (H - C) ) / (H - L)        en [-1, +1],  0 si H == L
SV[i,t]  = CLV[i,t] * V[i,t]                       (volumen firmado)

OIB_clv[i,k] = sum_{tau=-k..-1} SV[i,tau] / sum_{tau=-k..-1} V[i,tau]
```

Es la construcción subyacente a la línea de acumulación/distribución de Chaikin. **Aviso de
honestidad: no tiene validación académica como proxy de order imbalance.** Se propone porque es
implementable con los datos disponibles y porque es linealmente independiente del retorno diario. La
validación obligatoria antes de usarla en producción es: sobre un subconjunto donde exista tick data,
calcular la correlación de rangos entre `OIB_clv` y el `OIB_vol` real de Lee–Ready. Si la correlación
no supera ~0,4, la feature se descarta. Este experimento está anotado en `docs/OPEN_QUESTIONS.md`.

### 4.4 Regla del tick sobre cierres diarios (NO recomendada)

```
SV[i,t] = sign( close[i,t] - close[i,t-1] ) * V[i,t]
```

Es la regla del tick aplicada a barras diarias. Se documenta aquí solo para desaconsejarla: está
mecánicamente correlacionada con el signo del retorno, con lo que su agregado pre-evento es casi
colineal con el signo de `pre_event_car_k` y no aporta nada. Si se implementa, debe ser como
*baseline* contra el que comparar, nunca como feature del modelo final.

### 4.5 Desequilibrio minorista (BJZZ) — para cuando haya TAQ

Boehmer, Jones, Zhang y Zhang (2021) identifican operaciones minoristas en el tape usando la
**mejora de precio en subcéntimos** que dan los mayoristas a las órdenes minoristas internalizadas.
Para operaciones reportadas al TRF (código de mercado `D`):

```
Z[i,t] = 100 * mod( P[i,t], 0.01 )

Z in (0.0, 0.4)  -> minorista VENDEDOR iniciado
Z in (0.6, 1.0)  -> minorista COMPRADOR iniciado
resto            -> sin clasificar

MROIB[i,t] = (Vbuy_ret - Vsell_ret) / (Vbuy_ret + Vsell_ret)
```

**Caveat fuerte.** Barber, Huang, Jorion, Odean y Schwarz (2024) documentan tasas de error
materiales de este algoritmo, especialmente en los años recientes ⚠, a medida que cambian las
prácticas de internalización y la estructura de ticks. No usar sin re-validar por época.

Relevancia direccional: Kaniel, Liu, Saar y Titman (2012, *JF* 67, 639–680) encuentran que la compra
neta minorista **antes** del anuncio predice el retorno del anuncio, y que después los minoristas
operan en dirección **opuesta** al retorno pre-evento y a la sorpresa (toma de beneficios), lo que
contribuye al PEAD. Alrededor de la mitad de los retornos anormales en los tres meses siguientes es
atribuible a información privada ⚠.

### 4.6 Referencia direccional institucional

- Campbell, Ramadorai y Schwartz (2009, *JFE* 92(1), 66–91), "Caught on tape": infieren el flujo
  institucional diario del TAQ calibrando qué distribución de tamaños de operación mejor predice los
  13-F trimestrales. El flujo de órdenes es **más positivo antes de anuncios positivos** — los
  institucionales anticipan las sorpresas.
- Irvine, Lipson y Puckett (2007, *RFS* 20, 741–768), "Tipping": documentan volumen anormalmente alto
  y **desequilibrio comprador anormalmente grande empezando 5 días antes** de la publicación de
  recomendaciones iniciales de compra. Es la plantilla temporal exacta que buscamos replicar con
  datos públicos: la huella aparece en torno a **T−5**, no antes.

**Decisión de diseño derivada:** la ventana de detección principal debe ser **[T−5, T−1]**, con
[T−10, T−1] y [T−20, T−1] como ventanas secundarias. Ventanas más largas diluyen la señal.

---

## 5. PIN — Probability of Informed Trading

### 5.1 Modelo (Easley, Kiefer, O'Hara y Paperman, 1996)

Cada día de negociación:

- con probabilidad `alpha` ocurre un evento informativo;
- condicionado a que ocurra, la noticia es **mala** con probabilidad `delta` y **buena** con `1-delta`;
- los no informados compran con intensidad Poisson `eps_B` y venden con `eps_S`, siempre;
- los informados llegan con intensidad `mu`: **compran** si la noticia es buena, **venden** si es mala.

Verosimilitud de un día con `B` compras y `S` ventas:

```
L(theta | B,S) = (1-alpha)      * pois(B; eps_B)      * pois(S; eps_S)
               + alpha*delta    * pois(B; eps_B)      * pois(S; eps_S + mu)
               + alpha*(1-delta)* pois(B; eps_B + mu) * pois(S; eps_S)

pois(n; lam) = exp(-lam) * lam^n / n!
```

sobre D días: `L(theta) = prod_d L(theta | B_d, S_d)`, con
`theta = (alpha, delta, mu, eps_B, eps_S)` y `alpha, delta in [0,1]`, `mu, eps_B, eps_S >= 0`.

**Medida final:**

```
PIN = alpha * mu / ( alpha * mu + eps_B + eps_S )
```

Con la restricción habitual `eps_B = eps_S = eps` queda `PIN = alpha*mu / (alpha*mu + 2*eps)`.

### 5.2 Factorización numéricamente estable (derivada aquí)

La verosimilitud directa desborda para `B, S > ~150` (típico en el S&P 500, donde hay miles de
operaciones al día). La factorización estándar (Easley–Hvidkjaer–O'Hara 2010; Lin–Ke 2011) se obtiene
sacando factor común `exp(-eps_B-eps_S) * (mu+eps_B)^B * (mu+eps_S)^S / (B! S!)`. Definiendo:

```
x_b = eps_B / (mu + eps_B)          en (0,1]
x_s = eps_S / (mu + eps_S)          en (0,1]
M   = min(B,S) + max(B,S)/2
```

se obtiene (derivación verificable por sustitución directa):

```
ln L(theta|B,S) = -eps_B - eps_S
                + B*ln(mu + eps_B) + S*ln(mu + eps_S)
                + M*( ln(x_b) + ln(x_s) )
                + ln[  (1-alpha)        * x_b^(B-M) * x_s^(S-M)
                     + alpha*delta      * exp(-mu) * x_b^(B-M) * x_s^(-M)
                     + alpha*(1-delta)  * exp(-mu) * x_b^(-M)  * x_s^(S-M) ]
                - ln(B!) - ln(S!)
```

Los términos `-ln(B!) - ln(S!)` no dependen de `theta` y se omiten en la optimización. El truco de
`M` mantiene los exponentes de `x_b, x_s` acotados, evitando el subdesbordamiento.

**Verificación numérica realizada.** Esta factorización se ha comprobado contra la verosimilitud
directa sobre 500 combinaciones aleatorias de `(theta, B, S)`: error máximo **2,8·10⁻¹³**. La
derivación es correcta.

### 5.3 Pero la factorización clásica NO sirve para el S&P 500 — usar logsumexp

Comprobación adicional, y es la que cambia la decisión de implementación: la factorización con `M`
**desborda para los recuentos de operaciones típicos de nuestro universo**. Con
`theta = (0.3, 0.4, 200, 800, 850)`:

| B | S | verosimilitud directa | factorización `M` | logsumexp |
|---|---|---|---|---|
| 900 | 850 | −14,5100 | −14,5100 | −14,5100 |
| 3.000 | 3.000 | `-inf` (subdesborde) | **`OverflowError`** | −2.940,79 |
| 20.000 | 19.000 | `-inf` (subdesborde) | **`OverflowError`** | −81.810,28 |

La causa: con `B ≈ S = N` se tiene `M = 1,5N`, y el término `x_s^(-M)` con `x_s < 1` crece como
`exp(1,5·N·|ln x_s|)`, que desborda `float64` alrededor de `N ≈ 2.000`. Una acción del S&P 500 tiene
rutinariamente decenas de miles de operaciones diarias. Esta es exactamente la razón por la que
existe literatura específica sobre estimación de PIN en valores muy negociados (Ersan y Alıcı, 2016).

**Forma recomendada para este repo: log-sum-exp sobre los tres regímenes.** Definiendo
`lxb = ln(eps_B/(mu+eps_B))` y `lxs = ln(eps_S/(mu+eps_S))` (ambos ≤ 0):

```
t1 = log1p(-alpha)              + B*lxb + S*lxs           # sin evento
t2 = ln(alpha) + ln(delta)      - mu + B*lxb              # noticia mala
t3 = ln(alpha) + log1p(-delta)  - mu + S*lxs              # noticia buena

ln L(theta|B,S) = -eps_B - eps_S
                + B*ln(mu + eps_B) + S*ln(mu + eps_S)
                - lgamma(B+1) - lgamma(S+1)
                + logsumexp([t1, t2, t3])
```

`logsumexp` = `scipy.special.logsumexp`; `lgamma` = `scipy.special.gammaln`. Nunca se exponencia un
número grande, así que no hay desbordamiento a ninguna escala de `B, S`. Coincide con la
verosimilitud directa con error máximo **2,4·10⁻¹³** en el mismo test de 500 casos, y sigue siendo
finita en los casos donde tanto la directa como la factorización con `M` fallan (tabla anterior).

Usar `log1p(-alpha)` y `log1p(-delta)` en lugar de `ln(1-alpha)`, `ln(1-delta)` preserva la precisión
cuando el optimizador explora `alpha, delta → 1`, que es donde acaban muchos arranques.

**Estimación práctica:**
- Optimizador: `scipy.optimize.minimize` con `method="L-BFGS-B"` y cotas; o `"SLSQP"`.
- **Múltiples arranques obligatorios** (la superficie tiene óptimos locales): usar el esquema de
  valores iniciales en rejilla de Yan–Zhang o Ersan–Alıcı, con al menos 10–50 arranques por
  acción-trimestre y quedarse con el máximo global.
- Ventana: un trimestre (~60 sesiones) por acción. Con menos de ~40 días, `InsufficientHistory`.
- Determinismo: la rejilla de arranques debe derivarse de `seed` (contrato §0.4).

### 5.4 Signo, horizonte, magnitud y evidencia contraria

- **PIN no es direccional**: mide intensidad de asimetría informativa, no su signo. Como feature de
  evento entra como *interacción* (amplifica el signo de otras señales), nunca sola.
- Easley, Hvidkjaer y O'Hara (2002, *JF*): una diferencia de **10 puntos porcentuales de PIN entre dos
  acciones implica 2,5 % anual de diferencia en retorno esperado** ⚠ (NYSE 1983–1998).
- **EVIDENCIA CONTRARIA (fuerte).** Duarte y Young (2009, *JFE*) descomponen PIN en `APIN`
  (información asimétrica) y `PSOS` (probabilidad de shock simétrico de flujo, es decir iliquidez), y
  muestran vía Fama–MacBeth que **el componente de información asimétrica NO está valorado y el de
  iliquidez SÍ**. Conclusión: PIN cotiza porque es una proxy de iliquidez. Usar PIN como "medida de
  negociación informada" sin descomponerlo es dudoso.
- **Limitación operativa decisiva para este proyecto**: PIN requiere recuentos diarios de operaciones
  **firmadas**, es decir tick data + Lee–Ready. Con OHLCV diario **no se puede estimar PIN**. Y su
  frecuencia natural (trimestral) es incompatible con una ventana de 5 días. Recomendación: implementar
  `PIN` como feature *opcional*, calculada por acción-trimestre y desfasada un trimestre completo
  (`available_at = fin del trimestre anterior`), usada como **variable de condicionamiento**, y lanzar
  `ProviderUnavailable` si no hay proveedor intradía.

---

## 6. VPIN — Volume-Synchronized PIN

### 6.1 Cálculo

Easley, López de Prado y O'Hara (2011, 2012) sustituyen el reloj de calendario por un **reloj de
volumen**:

1. **Barras de tiempo finas** (típicamente 1 minuto) con precio `P` y volumen `V`.
2. **Bucket de volumen** de tamaño fijo `V_bucket`. El valor estándar es
   `V_bucket = ADV / 50` (50 buckets por día en promedio). Las barras que cruzan un límite de bucket
   se **parten proporcionalmente** por volumen.
3. Dentro de cada bucket `tau`, clasificar con BVC (§4.2) sobre las barras de 1 minuto:

```
Vbuy[tau]  = sum_{i in tau} V[i] * Phi( dP[i] / sigma_dP )
Vsell[tau] = V_bucket - Vbuy[tau]
```

`sigma_dP` = desviación típica de los cambios de precio entre barras consecutivas, estimada sobre la
muestra (p.ej. acción-mes).

4. **VPIN sobre los últimos `n` buckets** (estándar `n = 50`):

```
VPIN = sum_{tau=1}^{n} | Vsell[tau] - Vbuy[tau] |  /  ( n * V_bucket )
```

`VPIN in [0,1]`. Se actualiza con cada nuevo bucket (no con cada día). La justificación teórica es
que, bajo el modelo PIN, `E| Vsell - Vbuy | ≈ alpha*mu`, mientras que `n*V_bucket` aproxima
`alpha*mu + 2*eps`, con lo que VPIN es un estimador momento-a-momento de PIN sin necesidad de
maximizar verosimilitud.

5. **Transformación habitual para uso como alarma:** la CDF empírica de VPIN sobre una ventana móvil
   (p.ej. un año) — `VPIN_CDF > 0.9` es el umbral de "toxicidad" usado en la literatura.

### 6.2 Uso en este proyecto y evidencia contraria

- **EVIDENCIA CONTRARIA (fuerte).** Andersen y Bondarenko (2014, *Journal of Financial Markets*),
  "VPIN and the flash crash": VPIN es esencialmente una proxy de intensidad de negociación y
  volatilidad, **sin poder predictivo sobre la volatilidad futura más allá de benchmarks triviales**,
  y su aparente anticipación del flash crash de mayo de 2010 es un artefacto del reloj de volumen y
  del uso de información *ex post* en la calibración (el `sigma_dP` y el `ADV` del propio periodo).
- **Trampa de look-ahead específica de VPIN**, y es la que hunde muchas implementaciones: tanto
  `V_bucket = ADV/50` como `sigma_dP` suelen estimarse **sobre toda la muestra**. Eso es look-ahead
  puro. En este repo ambos parámetros deben calcularse con una ventana **móvil y estrictamente
  anterior** (p.ej. ADV y sigma de los 21 días previos), y estar cubiertos por un test que compare
  el resultado con y sin look-ahead.
- **Degradación con barras diarias.** Es tentador construir un "VPIN diario" con buckets de un día.
  No hacerlo sin advertirlo: con una barra por bucket, `|Vsell - Vbuy| / V = |2*Phi(dP/sigma) - 1|`,
  que es una **función del valor absoluto del retorno estandarizado**. El VPIN diario es, por
  construcción, una medida de volatilidad y no de información — lo que confirma la crítica de
  Andersen–Bondarenko en su versión extrema. Si se implementa, debe etiquetarse explícitamente como
  degradado y no puede entrar en el modelo junto a features de volatilidad.

**Recomendación**: `VPIN` se implementa contra un proveedor intradía; sin él, `ProviderUnavailable`.
No inventar una versión diaria y llamarla VPIN.

---

## 7. Retornos anormales acumulados pre-evento

### 7.1 Cálculo (alineado con `abnormal_returns` del contrato §3.5)

Modelo de mercado estimado en `estimation = (-250, -40)` sesiones (por defecto del contrato), de modo
que la ventana de estimación **termina antes** de la ventana de detección:

```
r[i,t] = a[i] + b[i]*r_m[t] + e[i,t]              (OLS sobre la ventana de estimación, L observaciones)
AR[i,tau] = r[i,tau] - ( a[i] + b[i]*r_m[tau] )
CAR[i,k]  = sum_{tau=-k..-1} AR[i,tau]            -> pre_event_car_5 / _10 / _20
```

**Estandarización de Patell (1976)**, que corrige por el error de estimación de `a, b` y hace
comparables eventos con distinta volatilidad:

```
s2[i,tau] = s2_e[i] * ( 1 + 1/L + (r_m[tau] - rbar_m)^2 / sum_{t in est} (r_m[t] - rbar_m)^2 )
SAR[i,tau] = AR[i,tau] / sqrt( s2[i,tau] )
SCAR[i,k]  = sum_{tau=-k..-1} SAR[i,tau] / sqrt(k)
```

Para el contraste en sección cruzada usar **Boehmer, Musumeci y Poulsen (1991)** (test
estandarizado con varianza inducida por el evento), no el t de Patell puro: alrededor de resultados
la varianza aumenta y el test de Patell sobre-rechaza masivamente.

Requisitos mínimos: al menos 120 observaciones válidas en la ventana de estimación y `r_m` del mismo
calendario PIT; en otro caso `InsufficientHistory`.

### 7.2 Signo, horizonte y — sobre todo — confusores

Signo esperado: **positivo** (el precio sube antes de una sorpresa positiva) si hay anticipación. Pero
`pre_event_car_k` es la feature más contaminada de todo el conjunto:

1. **Momentum.** A 20 días, el CAR pre-evento es momentum de corto plazo, con su propio signo
   conocido (reversión a 1 mes, continuación a 12 meses).
2. **Prima de anuncio de resultados.** Frazzini y Lamont (2007): comprar todas las acciones que
   anuncian en el mes siguiente y vender las que no rinde **más de 60 pb/mes**, con una prima
   estimada de **más del 7 % anual** ⚠. Cualquier estrategia que esté larga en anunciantes hereda
   esta prima y la confunde con alfa del detector. **El benchmark correcto es una cartera de
   anunciantes, no el mercado.**
3. **Sobre-extrapolación pre-anuncio.** La literatura documenta que las empresas con alto entusiasmo
   inversor pre-anuncio muestran ajustes de precio menores cuando baten expectativas y **mayores
   cuando decepcionan** ⚠. Es decir, la relación entre run-up de precio y retorno del anuncio no es
   monótona: hay un componente de reversión.
4. **Colinealidad con el resto de features.** Casi todas las señales de flujo (§4) están correlacionadas
   con el CAR pre-evento por construcción.

**Regla operativa:** `pre_event_car_k` entra en el modelo, pero **todas las demás features de flujo se
ortogonalizan contra ella** antes de la combinación (§10.2). Si una feature pierde todo su IC tras
la ortogonalización, no aporta nada — y esto le pasa a más señales de las que la literatura primaria
sugiere (§9.e).

---

## 8. Fuentes de flujo con retardo institucional

### 8.1 Short interest (FINRA, quincenal)

**Definiciones.**

```
SIR[i,t]  = shares_short[i,t] / shares_outstanding[i,t]        (short interest ratio)
DTC[i,t]  = shares_short[i,t] / ADV20[i,t]                     (days to cover)

dSI[i]    = SIR[i, ultimo snapshot publicado antes de T] - SIR[i, snapshot anterior]
dSI_z[i]  = ( dSI[i] - mean(dSI[i, 8 snapshots previos]) ) / std(dSI[i, 8 snapshots previos])
```

`short_interest_delta` = `-dSI_z[i]` para respetar la convención "mayor = más alcista" del repo,
puesto que el signo económico es negativo.

**Signo, horizonte, magnitud.**
- **Negativo y robusto**: más short interest → peores retornos y sorpresas más negativas.
- Boehmer, Jones y Zhang (2008, *JF*), "Which shorts are informed?": el decil más shorteado rinde
  **1,16 % mensual menos** que el menos shorteado ⚠, y la predicción se mantiene a horizontes de
  hasta 3 meses. (Nota: usan **flujo diario** de ventas en corto de NYSE SuperDOT, no el short
  interest quincenal — el flujo es más informativo que el stock.)
- Christophe, Ferri y Angel (2004, *JF*): las ventas en corto en los **5 días previos** al anuncio se
  relacionan negativamente con el retorno post-anuncio (Nasdaq, otoño de 2000). Muestra pequeña y
  periodo muy corto: tratar como evidencia sugerente, no concluyente.
- Akbas, Boehmer, Ertürk y Sorescu (2017, *Financial Management*): el short interest alto predice
  sorpresas de resultados negativas y revisiones a la baja del consenso.
- Se documenta **variación cíclica**: los cortos "seleccionan" (stock-picking) en expansiones y
  "temporizan" (market-timing) en recesiones ⚠, de modo que el IC de la señal no es estacionario.

**Trampas de implementación — esta es la sección que más backtests rompe.**

1. **`available_at` ≠ fecha de referencia.** El snapshot es una foto en la *settlement date* de
   referencia (la liquidación del día 15, o la última del mes). Las firmas reportan hasta las **18:00
   ET del 2.º día hábil posterior** a esa fecha, y **FINRA compila y publica en torno al 7.º–8.º día
   hábil posterior**. Indexar por la fecha de referencia introduce **~8 días hábiles de look-ahead**,
   suficiente para atravesar íntegro el intervalo [T−5, T−1] de un anuncio. `available_at` = fecha de
   publicación, sin excepciones.
2. **Cambio del ciclo de liquidación.** La correspondencia entre *settlement date* y *trade date* ha
   cambiado: T+3 hasta septiembre de 2017, T+2 hasta el 28 de mayo de 2024, T+1 desde entonces. Un
   backtest largo debe mapear settlement→trade con el ciclo vigente en cada fecha.
3. **Denominador PIT.** `shares_outstanding` debe ser la cifra vigente y ajustada por splits ocurridos
   entre snapshots; si no, aparecen saltos espurios de `dSI` de magnitud arbitraria.
4. **Short interest ≠ short volume.** El fichero diario de *short sale volume* de FINRA es una
   variable de **flujo dominada por la intermediación** (los creadores de mercado y mayoristas venden
   en corto continuamente como fontanería de inventario), y además solo marca el lado vendedor de las
   operaciones publicadas, sin las compensatorias. La propia FINRA lo advierte en su *Information
   Notice* de 10/05/2019. **No usar el short volume diario como proxy de posicionamiento
   direccional**, y en particular no usar el *off-exchange short volume ratio*, que es casi
   íntegramente cobertura de internalizadores.
5. **Confusores de posición.** Arbitraje de convertibles, arbitraje de fusiones, arbitraje de índices,
   cobertura de creación/reembolso de ETF y operaciones de captura de dividendo generan short interest
   sin contenido informativo. Filtrar (o al menos marcar) emisores con convertibles vivos y objetivos
   de operaciones corporativas anunciadas.
6. **Coste de préstamo.** El canal informativo lo media la tasa de préstamo (hard-to-borrow), que no
   es pública gratuitamente. Su ausencia limita la interpretación; anotado como pregunta abierta.

### 8.2 Cuota de volumen off-exchange / ATS (FINRA, semanal)

**Datos.** FINRA *OTC Transparency* publica, por valor y por MPID, el volumen semanal ejecutado en
cada ATS, y también el volumen de los *Non-ATS* (mayoristas/internalizadores) por firma por encima de
un umbral. Calendario de publicación:

- **NMS Tier 1** (donde están todos los componentes del S&P 500): **retardo de 2 semanas**.
- Resto de NMS y OTC Equity Securities: **retardo de 4 semanas**.

**Métrica.**

```
off_share[i,w] = ( ats_volume[i,w] + nonats_volume[i,w] ) / consolidated_volume[i,w]

off_exchange_share_delta[i] = ( off_share[i, ultima semana publicada antes de T]
                                - mean( off_share[i, 12 semanas previas] ) )
                              / std( off_share[i, 12 semanas previas] )
```

**Trampa de doble conteo.** FINRA reporta las acciones de cada operación **una sola vez**. El volumen
consolidado del denominador debe contarse igualmente una vez. Mezclar convenciones produce cuotas
por encima del 100 % o sistemáticamente infladas al doble. Verificar la convención del proveedor de
volumen consolidado antes de dividir.

**Signo esperado: negativo o nulo, no positivo.** Es lo contrario de la intuición popular ("mucho
volumen dark = dinero inteligente acumulando"):

- Zhu (2014, *RFS*): los informados **se concentran en el mercado lit**, porque en el dark pool
  todos los informados están en el mismo lado y el riesgo de no ejecución es máximo precisamente
  cuando se tiene información. La predicción teórica es que la **cuota dark CAE** cuando aumenta la
  negociación informada.
- Comerton-Forde y Putniņš (2015, *JFE* 118(1), 70–92): las operaciones dark son **menos informadas**
  que las lit; niveles bajos de dark trading no-bloque son benignos o beneficiosos para la eficiencia
  informativa y niveles altos son perjudiciales (relación no lineal, con umbral ⚠); las operaciones
  de **bloque** en dark **no** perjudican el descubrimiento de precios.
- En ventas en corto, las ejecutadas en bolsa son significativamente **más informativas** que las
  ejecutadas en dark pools ⚠.

**Trampa PIT decisiva.** Con retardo de 2 semanas sobre datos semanales, para un evento en `T` la
última semana **publicada** cubre negociación de hace ~3–5 semanas. Es decir:
`off_exchange_share_delta` **no puede ser una feature de la ventana [T−5, T−1]**. Solo es utilizable
como variable de condicionamiento lenta (régimen de fragmentación del valor). Debe implementarse con
`available_at` = fecha de publicación y hay que aceptar que su ventana efectiva es
[T−35, T−15] aproximadamente. Documentarlo en el docstring de la feature para que nadie la
malinterprete después.

### 8.3 Compras netas de insiders (Form 4)

**Datos.** SEC EDGAR, Form 4 en XML desde junio de 2003. Plazo de presentación: **antes del final del
2.º día hábil siguiente a la transacción** (Sarbanes–Oxley §403, en vigor desde el 29/08/2002); antes
de SOX era el día 10 del mes siguiente. Las presentaciones tardías existen y son frecuentes; el emisor
debe divulgarlas bajo el epígrafe "Delinquent Section 16(a) Reports" del Item 405 de la Regulation S-K.

`available_at` = **timestamp de aceptación en EDGAR**, nunca la fecha de la transacción. Usar la
fecha de transacción introduce entre 2 días y varios meses de look-ahead.

**Filtrado de códigos de transacción — el error más común de la literatura aplicada.**

```
CONSERVAR:  P  (compra en mercado abierto)
            S  (venta en mercado abierto)

EXCLUIR:    A  (concesión/premio)               M  (ejercicio de opciones)
            F  (retención de acciones para impuestos)
            G  (donación)                       C  (conversión)
            D  (disposición a favor del emisor)  ...
```

Sin este filtro, la "venta de insiders" está dominada por combinaciones M+S (ejercitar y vender) y
por F (liquidación neta para impuestos), que son mecánicas y **no informativas**. Un backtest de
insider selling sin filtro de códigos mide el calendario de vesting, no información.

Además, desde el 1 de abril de 2023 el Form 4 incluye una **casilla de plan 10b5-1** con su fecha de
adopción. Las operaciones bajo plan preexistente deben excluirse o ponderarse a la baja.

**Métrica.**

```
NPR[i, K]  = ( buy_shares - sell_shares ) / ( buy_shares + sell_shares )     en [-1, +1]
NPV[i, K]  = ( buy_usd - sell_usd ) / ( ADV_usd[i] * K )                     normalizado por liquidez
```

agregando **solo los Form 4 con `accepted_at <= T-1`** y transacción dentro de [T−K, T−1].

**Clasificación rutinario/oportunista (Cohen, Malloy y Pomorski, 2012, *JF*).** Un insider es
**rutinario** si operó en el **mismo mes natural** en cada uno de los 3 años anteriores; en otro caso
es **oportunista**. Requiere ≥3 años de historia del insider; sin ella, `InsufficientHistory` o
etiqueta `unknown` (no asumir oportunista por defecto: sesgaría la señal).

**Magnitudes.**
- Cohen–Malloy–Pomorski: cartera basada solo en operaciones **oportunistas**: **82 pb/mes**
  ponderada por valor, **180 pb/mes** equiponderada; los rutinarios rinden **≈ 0** ⚠. Los
  oportunistas predicen los retornos de futuros anuncios de resultados, previsiones de analistas y
  previsiones de la dirección; los rutinarios no.
- Ali y Hirshleifer (2017, *JFE* 126(3), 490–515): identifican **empresas oportunistas** por la
  rentabilidad histórica de las operaciones de sus insiders alrededor de anuncios trimestrales; la
  estrategia rinde **alfa 4-factores > 1 %/mes ponderada por valor**, significativa **también en el
  lado corto** ⚠. Esas empresas tienen además más gestión de resultados, reexpresiones, acciones de
  la SEC y litigios.

**Trampa de horizonte — decisiva para nuestra ventana.** Ke, Huddart y Petroni (2003, *JAE*)
muestran que las ventas de insiders aumentan **de 3 a 9 trimestres antes** de una ruptura en una serie
de crecimientos de beneficios, pero que **apenas hay ventas anormales en los 2 trimestres
inmediatamente anteriores**, consistente con la evitación deliberada del riesgo legal. El retorno
anormal medio en los 32 días previos e incluyendo el anuncio de la ruptura es del **−4,29 %** ⚠.

A esto se suman los **blackout periods** corporativos: la mayoría de emisores prohíben operar desde
~2 semanas antes del cierre del trimestre hasta 1–2 días después del anuncio.

**Conclusión operativa:** `insider_net_buy_form4` calculado sobre [T−30, T−1] será **casi siempre
cero** y no aportará nada. La feature debe calcularse sobre **[T−180, T−1]** (o de forma equivalente
sobre los 2–4 trimestres previos), separando `insider_net_buy_form4_opportunistic` de
`insider_net_buy_form4_routine`, y con el volumen del blackout excluido explícitamente para no
confundir "no operó porque no sabía nada" con "no podía operar".

---

## 9. Señales de opciones: resumen y evidencia contraria

Esta sección se limita a lo necesario para el argumento de evidencia contraria; el estudio detallado
de opciones corresponde a `data.options`.

**a) Volatility spread (Cremers y Weinbaum, 2010, *JFQA* 45, 335–367).**

```
vol_spread[i,t] = sum_j w[j] * ( IV_call[i,j,t] - IV_put[i,j,t] )
```

sobre pares `j` casados por (strike, vencimiento); `w[j]` proporcional al open interest medio de la
call y la put del par. Signo **positivo**: calls relativamente caras → retornos superiores. Magnitud:
las acciones con calls relativamente caras superan a las de puts relativamente caras en **~50 pb por
semana** ⚠, con predictibilidad **decreciente** a lo largo de la muestra (consistente con
mispricing que se arbitra con el tiempo). Mayor predictibilidad con opciones líquidas y acción
ilíquida, y en entornos informativos más asimétricos.

**b) Volatility skew / smirk (Xing, Zhang y Zhao, 2010, *JFQA*).**

```
iv_skew[i,t] = IV_put_OTM[i,t] - IV_call_ATM[i,t]
```

con la put OTM elegida por moneyness `K/S` más próximo a **0,95** y la call ATM por `K/S` más próximo
a **1,00**. Signo **negativo**: el quintil con smirk más pronunciado rinde **~10,9 % anual menos**
ajustado por riesgo ⚠, con persistencia de al menos 6 meses, y son además las empresas que sufren
los **peores shocks de beneficios en el trimestre siguiente**.

**c) Acumulación de volatility spread pre-anuncio (Atilgan, 2014).** El desalineamiento entre IV de
calls y puts **crece monótonamente durante el mes previo** al anuncio, su dirección coincide con el
signo del retorno del anuncio, y el spread anormal acumulado tiene poder predictivo significativo
sobre dicho retorno. Más fuerte con opciones líquidas, entorno informativo más asimétrico y acción
ilíquida.

### 9.1 EVIDENCIA CONTRARIA sobre volumen de opciones

Esta es la parte que más contradice la intuición del "unusual options activity":

**d) Johnson y So (2012, *JFE* 106(2), 262–286): el ratio O/S predice retornos NEGATIVAMENTE.**

```
OS[i,t] = ( option_volume_contracts[i,t] * 100 ) / stock_volume_shares[i,t]
```

El **decil más bajo de O/S supera al más alto en 0,34 % por semana** (~19,3 % anualizado) ⚠. Es
decir, **más volumen de opciones relativo predice retornos MENORES**, no mayores. Mecanismo propuesto:
los costes de venta en corto empujan a los informados con noticias **negativas** hacia el mercado de
opciones, de modo que un pico de O/S es, en promedio, una señal bajista. La señal es más fuerte
cuando el coste de préstamo es alto o el apalancamiento de la opción es bajo.

**e) Roll, Schwartz y Subrahmanyam (2010, *JFE*).** O/S es más alto **alrededor** de los anuncios de
resultados y un O/S más alto predice **retornos anormales post-anuncio MENORES**.

**f) Ge, Lin y Pearson (2016, *JFE*).** Matizan el mecanismo: la predictibilidad procede del volumen
de operaciones que **abren** posiciones y son **direccionales**; ni el apalancamiento ni
exclusivamente el coste de préstamo lo explican. Implicación práctica dura: el volumen de opciones
**sin desglose open/close ni dirección** (que es lo único que da el feed público de OPRA) es una
proxy ruidosa. El desglose fiable requiere datos de pago (ISE/CBOE Open-Close).

**g) Jin, Livnat y Zhang (2012, *JAR* 50(2), 401–432).** La ventaja informativa de las medidas de
opciones (volatility spread, skew) sobre los retornos de corto plazo del evento **se concentra en
eventos NO PROGRAMADOS**. Para anuncios **programados** —que es exactamente nuestro caso— es mucho
más débil. Razón económica: la fecha se conoce, la producción de información pública se intensifica y
el informado pierde la ventaja de sincronización.

**h) Poder predictivo que se desvanece al controlar por el retorno.** En la literatura de
opciones/desacuerdo se documenta que el turnover anormal de opciones pre-anuncio parece predecir los
retornos anormales post-anuncio, pero que **la predictibilidad desaparece una vez se controla por los
retornos pre-anuncio** ⚠ (atribución exacta a verificar: la referencia más probable es Choy y Wei,
2012, *Journal of Banking & Finance*). Este es **el test de referencia** que debemos ejecutar sobre
cada una de nuestras features de flujo.

**i) `oi_buildup_calls/puts`: caveats.** El open interest lo publica la OCC tras el proceso nocturno,
de modo que el OI del día `t` está disponible la **mañana de t+1** (`available_at = t+1 pre-apertura`).
Además el ΔOI es **neto**: no distingue aperturas compradoras de aperturas vendedoras sin datos de
dirección. Un aumento de OI de calls es igual de compatible con acumulación alcista informada que con
venta de calls cubiertas.

**j) `iv_term_slope`: caveat mecánico.** La IV del vencimiento frontal **sube mecánicamente** al
acercarse el anuncio porque incorpora la varianza del evento; no es una señal, es aritmética. Hay que
"des-eventizar" la estructura temporal (extraer la varianza del evento con el modelo estándar de dos
componentes) o comparar la pendiente contra la del **mismo emisor en el mismo tau de trimestres
anteriores**.

---

## 10. Filtraciones documentadas y qué nos enseñan

### 10.1 Los casos

**El hackeo de los newswires (2010–2015).** El caso más relevante del mundo para este proyecto. La
SEC imputó a **32 demandados** (comunicado 2015-163) por un esquema en el que hackers ucranianos y
rusos accedieron a los sistemas de Marketwired (2010–2013), PR Newswire (2010–2012) y Business Wire
(2015), obteniendo comunicados —incluidos anuncios de resultados— **después de que las empresas los
enviaran al newswire y antes de su publicación**. Se usaron alrededor de **800 comunicados** para
operar, con beneficios ilícitos superiores a **100 millones de dólares**. La ventana de ventaja
temporal iba de **horas a tres días**.

**Akey, Grégoire y Martineau (2022, *JFE* 143(3), 1162–1184), "Price revelation from insider trading".**
Explotan ese episodio como experimento natural. Hallazgos con consecuencias directas de diseño:

- Los informados **seleccionan** dónde operar: prefieren empresas **grandes**, con **alta cobertura de
  analistas** y **líquidas**. Esto es exactamente lo contrario de donde un detector ingenuo esperaría
  encontrar filtraciones (small caps opacas). Nuestro universo (S&P 500) es, afortunadamente, el
  universo que ellos eligen.
- Operan cuando la señal **cuantitativa** (cifras) y la **cualitativa** (texto) apuntan en la misma
  dirección, y cuando hay una brecha grande entre precio y valor revelado.
- **Los precios incorporan la sorpresa principalmente vía revisiones de cotización, no vía
  operaciones.** Es decir, buena parte de la huella está en el **quote**, no en el **trade** — y el
  quote no está en OHLCV diario. Esto acota estructuralmente lo que un detector con datos diarios
  puede ver.

**Xie (2025/2026, *Journal of Accounting Research*), "Informed Trade of Earnings Announcements".**
Un cartel operó con acceso anticipado a más de **1.000 anuncios de resultados** entre 2011 y 2015.
Prefirieron anuncios con **mayores sorpresas de beneficios y de ventas** respecto al consenso, con
**guidance cuantitativo** y con **sentimiento más extremo**. Resultado sorprendente y muy útil: pese a
la previsión perfecta, **rindieron pobremente** respecto a estrategias hipotéticas con la misma
información. Traducción para nosotros: incluso una filtración real deja una huella **más débil de lo
que el tamaño de la sorpresa sugeriría**, porque el informado está limitado por capital, ejecución y
riesgo de detección. El límite superior de detectabilidad es más bajo de lo que uno esperaría.

**Kacperczyk y Pagnotta (2019, *RFS*), "Chasing private information".** Ya citado en §2: más de 5.000
operaciones de expedientes de la SEC (1995–2015); en los días informados hay volumen y volatilidad
anormalmente altos, **spreads más estrechos** (−10 % acciones, −20 % opciones) ⚠, y los informados
**eligen días de alto volumen no informado**. La información en el mercado de **opciones** es una
señal generalmente **más fuerte** que la del mercado de acciones.

**Tipping documentado (Irvine, Lipson y Puckett, 2007, *RFS*).** Volumen anormal y desequilibrio
comprador anormal **desde 5 días antes** de recomendaciones iniciales de compra; los institucionales
que compraron antes obtuvieron beneficios anormales positivos. La intensidad del desequilibrio se
relaciona con características que **solo se conocen leyendo el informe** (identidad del analista,
si es strong buy) — prueba de filtración de contenido, no de inferencia.

**Riesgo cibernético como canal.** Trabajos recientes ("digital insiders") documentan que las
empresas con menores puntuaciones de mitigación de riesgo ciber presentan **mayor volumen anormal de
acciones y de opciones y mayor volatilidad intradía** en las semanas previas a los anuncios ⚠. Sugiere
una feature de condicionamiento externa (score de ciberriesgo) que hoy no tenemos.

### 10.2 Lo que estos casos enseñan al detector

1. **La ventana es corta**: horas a 3 días en el caso de los newswires; ~5 días en el tipping. La
   ventana principal de detección es **[T−5, T−1]**, y para el caso newswire incluso **[T−1, T−1]**.
2. **El objetivo son empresas grandes y líquidas** — es decir, exactamente el S&P 500. Bien.
3. **Los informados se esconden en días de alto volumen no informado**, lo que confunde
   sistemáticamente el volumen anormal.
4. **Los spreads se estrechan, no se ensanchan.**
5. **Buena parte de la revelación va por cotizaciones, no por operaciones** → techo estructural para
   detectores basados en OHLCV.
6. **La huella es más débil de lo que sugiere el tamaño de la sorpresa** (Xie): calibrar expectativas
   de potencia a la baja.

---

## 11. Evidencia contraria consolidada

Recopilada para que quede en un solo lugar y ninguna se pierda al implementar:

| Creencia común | Evidencia contraria | Referencia |
|---|---|---|
| Alto volumen pre-anuncio = informados operando | El volumen **cae** antes de anuncios *programados*, y cae **más** cuanto mayor la asimetría informativa | Chae (2005) |
| Mucho volumen de opciones = señal alcista | O/S alto predice retornos **más bajos**; +0,34 %/semana para el decil bajo vs. alto | Johnson y So (2012); Roll et al. (2010) |
| Las opciones anticipan los resultados | La ventaja informativa de las medidas de opciones se concentra en eventos **no programados**; en resultados es débil | Jin, Livnat y Zhang (2012) |
| El turnover anormal de opciones predice el evento | La predictibilidad **desaparece** al controlar por el retorno pre-anuncio ⚠ | Literatura opciones/desacuerdo (Choy y Wei, 2012, a verificar) |
| Spreads se ensanchan cuando hay informados | Spreads **10 %/20 % más estrechos** en días informados documentados por la SEC | Kacperczyk y Pagnotta (2019) |
| PIN mide información asimétrica | PIN cotiza porque proxy de **iliquidez**; el componente de información no está valorado | Duarte y Young (2009) |
| VPIN detecta toxicidad y anticipa crashes | Sin poder predictivo incremental sobre volatilidad; la "predicción" del flash crash es artefacto de calibración *ex post* | Andersen y Bondarenko (2014) |
| Mucho volumen dark = acumulación informada | Las operaciones dark son **menos informadas**; la teoría predice que los informados **huyen** del dark | Zhu (2014); Comerton-Forde y Putniņš (2015) |
| El short volume diario mide posicionamiento bajista | Es **flujo dominado por intermediación**, no posición; la propia FINRA lo advierte | FINRA Information Notice 10/05/2019 |
| Las ventas de insiders antes de resultados avisan | Los insiders **evitan deliberadamente** los 2 trimestres previos; la señal está 3–9 trimestres antes | Ke, Huddart y Petroni (2003) |
| El desequilibrio con regla del tick es fiable | Precisión 76–81 % y decreciente; los desequilibrios "carecen de precisión suficiente" | Ellis, Michaely y O'Hara (2000); Chakrabarty et al. |
| Un CAR pre-evento positivo es alfa del detector | Los anunciantes ganan **>60 pb/mes** solo por anunciar (prima de anuncio) | Frazzini y Lamont (2007) |
| El run-up de precio predice monótonamente el retorno del anuncio | Hay **sobre-extrapolación**: mayor castigo a la decepción tras un run-up alto ⚠ | Literatura de sobre-extrapolación pre-anuncio |

---

## 12. Construcción, validación y potencia del detector

### 12.1 Estandarización en sección cruzada por cohorte

Los eventos de resultados se **agrupan en el tiempo** (la temporada de resultados concentra ~40 % de
los eventos en tres semanas de cada trimestre). Los niveles calendario no son comparables. Toda
feature debe z-scorearse **dentro de la cohorte de anunciantes de la misma fecha de evento** (o de la
misma semana si la cohorte diaria es pequeña, con mínimo de ~20 eventos; si no, usar la ventana de
±5 días hábiles).

### 12.2 Ortogonalización obligatoria

Antes de combinar, cada feature candidata `f` se residualiza:

```
f[i] = g0 + g1*pre_event_car_20[i] + g2*ln(mktcap[i]) + g3*ln(ADV_usd[i])
     + g4*realized_vol_60d[i] + sum_s g_s * sector_dummy[s,i] + u[i]

f_orth[i] = u[i]
```

Regresión **por cohorte de fecha**. Una feature que pierde todo su IC tras esto es un duplicado del
CAR pre-evento y debe eliminarse. Este es exactamente el test de §11 fila 4.

### 12.3 Validación temporal

- **CV purgada con embargo** (López de Prado): purgar del entrenamiento todo evento cuya ventana
  [−pre, +post] solape la ventana de test, y aplicar embargo ≥ `post` (60 sesiones por defecto) para
  que el PEAD de un evento de entrenamiento no se filtre al test.
- **Errores estándar agrupados por fecha de anuncio** (los eventos del mismo día comparten shocks de
  mercado y de sector). Sin clustering, los t se inflan por un factor sustancial.
- **Corrección por comparaciones múltiples** (Benjamini–Hochberg) sobre toda la batería de features:
  con ~20 features y umbral 5 %, se espera 1 falso positivo por puro azar.

### 12.4 Potencia y tasa base — el cálculo que evita autoengaños

Sea `p` la prevalencia real de eventos con filtración material, `Se` la sensibilidad del detector y
`Sp` su especificidad. La **precisión** (valor predictivo positivo) es:

```
PPV = p*Se / ( p*Se + (1-p)*(1-Sp) )
```

Con `p = 0,02`, `Se = 0,80`, `Sp = 0,90`:

```
PPV = 0,016 / (0,016 + 0,098) = 0,14
```

**El 86 % de las alertas serían falsos positivos** con un detector que suena excelente en el papel.
Conclusiones: (a) el output de `PreEventFeatures` debe ser un **score continuo**, no una alerta
binaria; (b) toda evaluación debe reportar **precision–recall** (y AUC-PR), no solo AUC-ROC, que es
engañosa con clases desbalanceadas; (c) el uso legítimo del score no es "señalar filtraciones" sino
**ponderar la exposición** en la cartera de eventos.

### 12.5 Validación contra el mercado sintético

`data.synthetic.SyntheticMarket(seed)` genera un subconjunto configurable de eventos "con filtración"
con run-up de volumen. Eso permite el test que de verdad importa: **¿el detector detecta?**

Protocolo mínimo, sin red y determinista:

1. Generar N eventos con fracción de filtración `q` conocida e intensidad de run-up `lambda`.
2. Calcular `PreEventFeatures` y el score compuesto.
3. Reportar AUC-PR y la curva de potencia frente a `lambda`: **la menor intensidad de filtración
   detectable con potencia 0,8 al 5 % de FDR**. Ese número es el resultado de calidad del módulo.
4. Test de no-look-ahead: recalcular con todos los `available_at` desplazados +1 sesión; el AUC-PR
   debe **degradarse**. Si no cambia, hay una fuga temporal en alguna feature.

---

## 13. Tabla PIT: `available_at` correcto por fuente

| Fuente | Fecha "natural" del dato | `available_at` correcto | Retardo típico |
|---|---|---|---|
| OHLCV diario | sesión `t` | cierre de `t` | 0 |
| Open interest de opciones (OCC) | sesión `t` | pre-apertura de `t+1` | 1 sesión |
| Cadena de opciones / IV | snapshot de `t` | cierre de `t` | 0 |
| 8-K Item 2.02 (resultados) | `announced_at` | timestamp de aceptación en EDGAR | minutos |
| Form 4 | fecha de transacción | timestamp de aceptación en EDGAR | 0–2 días hábiles (más si es tardío) |
| Short interest FINRA | settlement date de referencia | **fecha de publicación** (~7.º–8.º día hábil posterior) | 8–12 días naturales |
| FINRA OTC/ATS Transparency (Tier 1) | semana `w` | fin de `w` + **2 semanas** | 14–21 días |
| FINRA OTC/ATS (Tier 2 / OTC) | semana `w` | fin de `w` + **4 semanas** | 28–35 días |
| Consenso de analistas | `as_of` del snapshot | `as_of` | 0 (si el proveedor es PIT) |
| Fundamentales XBRL | `period_end` | fecha de presentación del filing | 20–60 días |

Regla general del repo: **nunca indexar por la fecha del hecho económico; siempre por la fecha de
publicabilidad.**

---

## 14. Mapeo a `PreEventFeatures` del contrato

| Campo del contrato (§3.5) | Fórmula / sección | Notas |
|---|---|---|
| `volume_runup` | `AVR[i,k]`, §3.4 | k = 5 por defecto; base condicionada a tiempo-evento (§3.6) |
| `turnover_zscore` | `zbar[i,k]`, §3.3 con `se` empírica | ajustado por mercado (§3.2) |
| `abnormal_volume_5/10/20d` | `zbar[i,k]`, k ∈ {5,10,20}, §3.3–3.4 | añadir `SUV[i,k]` (§3.5) como variante |
| `order_imbalance_proxy` | `OIB_bvc[i,k]` (§4.2) y `OIB_clv[i,k]` (§4.3) | Lee–Ready (§4.1) si hay proveedor intradía; validar CLV antes de usar |
| `pre_event_car_5/10/20d` | `CAR[i,k]` / `SCAR[i,k]`, §7.1 | estimación (−250, −40); Patell + BMP |
| `short_interest_delta` | `-dSI_z[i]`, §8.1 | `available_at` = fecha de publicación |
| `off_exchange_share_delta` | §8.2 | ventana efectiva ≈ [T−35, T−15]; documentar |
| `insider_net_buy_form4` | `NPR[i,180]`, §8.3, separando oportunista/rutinario | filtro de códigos P/S obligatorio |
| `put_call_volume_ratio`, O/S | §9.1(d) | **signo negativo** |
| `oi_buildup_calls/puts` | §9.1(i) | `available_at` = `t+1` pre-apertura |
| `iv_skew_25delta` | §9(b) | signo negativo |
| `vol_spread` | §9(a) | signo positivo |
| `iv_term_slope` | §9.1(j) | des-eventizar |
| `analyst_revision_drift` | fuera de ámbito de este informe | ver investigación de `data.estimates` |

Features adicionales que esta revisión recomienda añadir al conjunto:

- `turnover_zscore_vs_own_prior_quarters` (§3.6): corrige el sesgo de Chae. **La más importante de
  las que faltan.**
- `suv_5/10/20d` (§3.5): volumen no explicado por el retorno; ortogonal por construcción al CAR.
- `runup_ratio` (§3.7): fracción del movimiento total que ocurre antes.
- `insider_net_buy_form4_opportunistic` / `_routine` (§8.3): la clasificación es donde está toda la
  señal.
- `pin_quarterly` y `vpin` como features **opcionales** que lanzan `ProviderUnavailable` sin
  proveedor intradía (§5.4, §6.2).

---

## 15. Preguntas abiertas (candidatas a `docs/OPEN_QUESTIONS.md`)

1. Verificar contra el texto completo todas las magnitudes marcadas ⚠ (WebFetch bloqueado en este
   entorno).
2. Validar empíricamente `OIB_clv` (§4.3) contra `OIB_vol` de Lee–Ready sobre un subconjunto con tick
   data. Umbral de aceptación propuesto: correlación de rangos ≥ 0,4.
3. Confirmar la atribución exacta del resultado "el poder predictivo del turnover de opciones
   desaparece al controlar por el retorno pre-anuncio" (§9.1.h).
4. Determinar los grados de libertad de la t de Student usados en la BVC original y si merecen la
   pena frente a la normal (§4.2).
5. Decidir la política para el ciclo de liquidación variable (T+3 / T+2 / T+1) en el mapeo
   settlement→trade del short interest (§8.1).
6. Obtener el umbral cuantitativo de la relación no lineal entre cuota dark y eficiencia informativa
   de Comerton-Forde y Putniņš, y su transferibilidad al mercado estadounidense (§8.2).
7. Evaluar si merece la pena una fuente de coste de préstamo (hard-to-borrow) para mediar la señal
   de short interest (§8.1, trampa 6).
8. Cuantificar la degradación del algoritmo BJZZ por época antes de habilitar `MROIB` (§4.5).

---

## 16. Referencias

**Microestructura y medidas de información**

- Easley, D., Kiefer, N. M., O'Hara, M., Paperman, J. B. (1996). "Liquidity, Information, and
  Infrequently Traded Stocks". *Journal of Finance* 51(4), 1405–1436.
- Easley, D., Hvidkjaer, S., O'Hara, M. (2002). "Is Information Risk a Determinant of Asset Returns?".
  *Journal of Finance* 57(5), 2185–2221. https://onlinelibrary.wiley.com/doi/abs/10.1111/1540-6261.00493
  · PDF: https://www.edegan.com/pdfs/Easley%20Hvidkjaer%20OHara%20(2002)%20-%20Is%20Information%20Risk%20A%20Determinant%20Of%20Asset%20Returns.pdf
- Easley, D., Hvidkjaer, S., O'Hara, M. (2010). "Factoring Information into Returns". *JFQA*.
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=556079
- Duarte, J., Young, L. (2009). "Why is PIN priced?". *Journal of Financial Economics* 91(2), 119–138.
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1024588 ·
  http://www.ruf.rice.edu/~jgsfss/duarte_092507.pdf
- Ersan, O., Ghachem, M. (2023). "PINstimation: An R Package for Estimating Probability of Informed
  Trading Models". *The R Journal*. https://journal.r-project.org/articles/RJ-2023-044/ ·
  https://www.pinstimation.com/
- Easley, D., López de Prado, M., O'Hara, M. (2012). "Flow Toxicity and Liquidity in a High-Frequency
  World". *Review of Financial Studies* 25(5), 1457–1493.
- Easley, D., López de Prado, M., O'Hara, M. (2016). "Discerning Information from Trade Data".
  *Journal of Financial Economics*. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1989555
- Andersen, T. G., Bondarenko, O. (2014). "VPIN and the flash crash". *Journal of Financial Markets*.
  https://www.sciencedirect.com/science/article/abs/pii/S1386418113000189
- Lee, C. M. C., Ready, M. J. (1991). "Inferring Trade Direction from Intraday Data". *Journal of
  Finance* 46(2), 733–746.
- Ellis, K., Michaely, R., O'Hara, M. (2000). "A test of the accuracy of the Lee/Ready trade
  classification algorithm". *Journal of International Financial Markets, Institutions & Money*.
  https://www.sciencedirect.com/science/article/abs/pii/S1042443100000482
- Chakrabarty, B., Pascual, R., Shkilko, A. (2015). "Evaluating trade classification algorithms: Bulk
  volume classification versus the tick rule and the Lee-Ready algorithm". *JFM*.
  https://www.sciencedirect.com/science/article/abs/pii/S1386418115000415
- Chordia, T., Roll, R., Subrahmanyam, A. (2002). "Order imbalance, liquidity, and market returns".
  *JFE*. https://www.cis.upenn.edu/~mkearns/finread/Chordia_buy-sell_orders.pdf
- Chordia, T., Subrahmanyam, A. (2004). "Order imbalance and individual stock returns: Theory and
  evidence". *JFE*. https://www.anderson.ucla.edu/documents/areas/fac/finance/36-00.pdf
- Boehmer, E., Jones, C. M., Zhang, X., Zhang, X. (2021). "Tracking Retail Investor Activity".
  *Journal of Finance* 76(5), 2249–2305.
- Barber, B., Huang, X., Jorion, P., Odean, T., Schwarz, C. (2024). "A (Sub)penny for Your Thoughts:
  Tracking Retail Investor Activity in TAQ". *Journal of Finance*. ⚠ (verificar cifras)

**Volumen alrededor de anuncios**

- Chae, J. (2005). "Trading Volume, Information Asymmetry, and Timing Information". *Journal of
  Finance* 60(1), 413–442.
  https://onlinelibrary.wiley.com/doi/full/10.1111/j.1540-6261.2005.00734.x
- Ajinkya, B. B., Jain, P. C. (1989). "The behavior of daily stock market trading volume". *Journal of
  Accounting and Economics*. https://www.sciencedirect.com/science/article/abs/pii/0165410189900189
- Cready, W. M., Ramanan, R. (1991). "The power of tests employing log-transformed volume in detecting
  abnormal trading". *JAE*. https://www.sciencedirect.com/science/article/abs/pii/0165410191900059
- Garfinkel, J. A., Sokobin, J. (2006). "Volatility, Volume, and the Post-Earnings-Announcement Drift".
  *Journal of Accounting Research*. https://www.biz.uiowa.edu/faculty/jgarfinkel/pubs/drift_JAR.pdf
- Frazzini, A., Lamont, O. A. (2007). "The Earnings Announcement Premium and Trading Volume". NBER
  WP 13090. https://www.nber.org/papers/w13090 ·
  https://w4.stern.nyu.edu/finance/docs/pdfs/Seminars/063w-lamont.pdf
- Patell, J. M. (1976). "Corporate Forecasts of Earnings Per Share and Stock Price Behavior: Empirical
  Tests". *Journal of Accounting Research* 14(2), 246–276.
- Boehmer, E., Musumeci, J., Poulsen, A. B. (1991). "Event-study methodology under conditions of
  event-induced variance". *Journal of Financial Economics* 30(2), 253–272.

**Negociación informada documentada y filtraciones**

- Kacperczyk, M., Pagnotta, E. (2019). "Chasing Private Information". *Review of Financial Studies*.
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2695197 ·
  https://www.eief.it/files/2016/09/marcin-kacperczyk.pdf
- Kacperczyk, M., Pagnotta, E. (2024). "Legal Risk and Insider Trading". *Journal of Finance* 79(1),
  305–355. https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.13299
- Akey, P., Grégoire, V., Martineau, C. (2022). "Price revelation from insider trading: Evidence from
  hacked earnings news". *Journal of Financial Economics* 143(3), 1162–1184.
  https://www.sciencedirect.com/science/article/pii/S0304405X21005237 ·
  código y datos: https://github.com/vgreg/hacked_earnings_jfe
- Xie, C. (2026). "Informed Trade of Earnings Announcements". *Journal of Accounting Research*.
  https://onlinelibrary.wiley.com/doi/10.1111/1475-679x.70032 ·
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5354711
- SEC (2015). "SEC Charges 32 Defendants in Scheme to Trade on Hacked News Releases". Press Release
  2015-163. https://www.sec.gov/newsroom/press-releases/2015-163
- Irvine, P., Lipson, M., Puckett, A. (2007). "Tipping". *Review of Financial Studies* 20(3), 741–768.
  https://doi.org/10.2139/ssrn.642341
- Campbell, J. Y., Ramadorai, T., Schwartz, A. (2009). "Caught on tape: Institutional trading, stock
  returns, and earnings announcements". *JFE* 92(1), 66–91.
  https://dash.harvard.edu/bitstream/1/2609649/2/Campbell_CaughtOnTape.pdf
- Kaniel, R., Liu, S., Saar, G., Titman, S. (2012). "Individual Investor Trading and Return Patterns
  around Earnings Announcements". *Journal of Finance* 67(2), 639–680.
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1782553

**Ventas en corto y venues**

- Boehmer, E., Jones, C. M., Zhang, X. (2008). "Which Shorts Are Informed?". *Journal of Finance*
  63(2), 491–527. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=855044
- Christophe, S. E., Ferri, M. G., Angel, J. J. (2004). "Short-Selling Prior to Earnings
  Announcements". *Journal of Finance* 59(4), 1845–1875.
  https://www.semanticscholar.org/paper/Short-Selling-Prior-to-Earnings-Announcements-Christophe-Ferri/a8294d152c39dbcd17abd88628c9839aead12bdf
- Akbas, F., Boehmer, E., Ertürk, B., Sorescu, S. (2017). "Short Interest, Returns, and Unfavorable
  Fundamental Information". *Financial Management* 46(2), 455–486.
  https://onlinelibrary.wiley.com/doi/abs/10.1111/fima.12144
- Engelberg, J. E., Reed, A. V., Ringgenberg, M. C. (2012). "How are shorts informed? Short sellers,
  news, and information processing". *JFE* 105(2), 260–278.
  https://rady.ucsd.edu/faculty/directory/engelberg/pub/portfolios/SHORT_NEWS.pdf
- Comerton-Forde, C., Putniņš, T. J. (2015). "Dark trading and price discovery". *JFE* 118(1), 70–92.
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2183392 ·
  https://conference.nber.org/confer/2012/MMf12/Comerton-Forde_Putnins.pdf
- Zhu, H. (2014). "Do Dark Pools Harm Price Discovery?". *Review of Financial Studies* 27(3), 747–789.
- FINRA (2019). "Understanding Short Sale Volume Data on FINRA's Website". Information Notice
  10/05/2019. https://www.finra.org/rules-guidance/notices/information-notice-051019
- FINRA. "Short Interest Reporting". https://www.finra.org/filing-reporting/regulatory-filing-systems/short-interest
- FINRA. "Equity Short Interest Data Glossary".
  https://www.finra.org/finra-data/browse-catalog/equity-short-interest/glossary
- FINRA. "OTC Transparency (ATS and Non-ATS) Data Website User Guide".
  https://www.finra.org/sites/default/files/OTC-transparency-website-user-guide-v5.pdf
- FINRA. "OTC Transparency — API Specifications".
  https://www.finra.org/sites/default/files/OTC-Transparency-Data-File-Download-API-v04.pdf
- FINRA. "Daily Short Sale Volume Files".
  https://www.finra.org/finra-data/browse-catalog/short-sale-volume-data/daily-short-sale-volume-files

**Insiders (Form 4)**

- Ke, B., Huddart, S., Petroni, K. (2003). "What insiders know about future earnings and how they use
  it: Evidence from insider trades". *Journal of Accounting and Economics* 35(3), 315–346.
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=373560
- Cohen, L., Malloy, C., Pomorski, L. (2012). "Decoding Inside Information". *Journal of Finance*
  67(3), 1009–1043. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1692517 ·
  https://www.nber.org/system/files/working_papers/w16454/w16454.pdf
- Ali, U., Hirshleifer, D. (2017). "Opportunism as a firm and managerial trait: Predicting insider
  trading profits and misconduct". *JFE* 126(3), 490–515.
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2635257
- U.S. Congress (2002). Sarbanes–Oxley Act, §403 (plazo de 2 días hábiles para el Form 4).
  https://corpgov.law.harvard.edu/2009/10/30/sox-and-insider-trades/

**Opciones**

- Cremers, M., Weinbaum, D. (2010). "Deviations from Put-Call Parity and Stock Return Predictability".
  *JFQA* 45(2), 335–367. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=968237
- Xing, Y., Zhang, X., Zhao, R. (2010). "What Does the Individual Option Volatility Smirk Tell Us
  About Future Equity Returns?". *JFQA* 45(3), 641–662.
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1107464 ·
  https://eng.pbcsf.tsinghua.edu.cn/__local/A/24/EA/815CC33800CAA0F7D894E9292A1_6CC8B38B_27B25.pdf
- Roll, R., Schwartz, E., Subrahmanyam, A. (2010). "O/S: The relative trading activity in options and
  stock". *JFE* 96(1), 1–17. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1410091 ·
  https://www.anderson.ucla.edu/documents/areas/fac/finance/options_volume_rev3.pdf
- Johnson, T. L., So, E. C. (2012). "The option to stock volume ratio and future returns". *JFE*
  106(2), 262–286. https://www.travislakejohnson.com/pdfs/Johnson%20So%20OS%202012%20(JFE).pdf
- Ge, L., Lin, T.-C., Pearson, N. D. (2016). "Why does the option to stock volume ratio predict stock
  returns?". *JFE* 120(3), 601–622.
  https://www.sciencedirect.com/science/article/abs/pii/S0304405X16000167
- Jin, W., Livnat, J., Zhang, Y. (2012). "Option Prices Leading Equity Prices: Do Option Traders Have
  an Information Advantage?". *Journal of Accounting Research* 50(2), 401–432.
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1737796
- Atilgan, Y. (2014). "Volatility spreads and earnings announcement returns". *Journal of Banking &
  Finance* 38, 205–215. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1512046
- Choy, S.-K., Wei, J. (2012). "Option trading: Information or differences of opinion?". *Journal of
  Banking & Finance*. https://www.sciencedirect.com/science/article/abs/pii/S0378426612001082
- Chan, K., Ge, L., Lin, T.-C. y otros. "Informed Options Trading Before Corporate Events".
  *Annual Review of Financial Economics*.
  https://www.annualreviews.org/doi/10.1146/annurev-financial-012820-033052

**Metodología de validación**

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. (CV purgada, embargo,
  Sharpe deflactado.)
- Bailey, D. H., López de Prado, M. (2014). "The Deflated Sharpe Ratio". *Journal of Portfolio
  Management* 40(5), 94–107.
- Benjamini, Y., Hochberg, Y. (1995). "Controlling the False Discovery Rate". *JRSS-B* 57(1), 289–300.
