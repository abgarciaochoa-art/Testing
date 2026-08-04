# Señales del mercado de opciones alrededor de anuncios de resultados

**Ámbito.** Revisión de literatura orientada a implementación para los módulos
`earnings_alpha.data.options` y `earnings_alpha.events` (ángulo B del proyecto). Cubre las seis
familias de señales que `docs/ARCHITECTURE.md` §3.5 exige en `PreEventFeatures`
(`vol_spread`, `iv_skew_25delta`, `put_call_volume_ratio`, `oi_buildup_calls/puts`,
`iv_term_slope`) más las que la literatura añade y el contrato no nombra explícitamente
(O/S, movimiento esperado del straddle, volatilidad implícita del evento).

Para cada señal se da: **fórmula exacta e implementable**, ventana pre-evento óptima, signo
esperado, poder predictivo reportado, problemas prácticos y **cómo calcularla con datos de fin de
día**, que es lo único asequible al retail.

Este documento **complementa** `docs/research/informed_trading.md` §9 (que trata las señales de
opciones sólo en lo necesario para el argumento de evidencia contraria) y
`docs/research/data_sources.md` §7 (proveedores de cadenas). Aquí está el detalle de cálculo.

**Fecha de compilación:** 2026-08-04.

---

## 0. Nota metodológica sobre las fuentes y sobre lo que está verificado

`WebFetch` devuelve **HTTP 403 en todos los dominios** en este contenedor (comprobado contra SSRN,
Columbia Business School, Wikipedia, OptionMetrics y repositorios institucionales). La revisión
bibliográfica se ha construido con **18 búsquedas web** que sintetizan resúmenes y abstracts. En
consecuencia:

- Las **cifras de magnitud de efecto** proceden de abstracts y de textos secundarios. Las que no he
  podido contrastar contra el texto completo van marcadas con ⚠ y **deben verificarse contra el
  paper primario** antes de calibrar nada con ellas.
- Los **detalles finos de construcción** de las medidas académicas (filtros exactos de moneyness,
  de días a vencimiento, umbrales de open interest) **no aparecen en los abstracts**. Donde no los
  he podido confirmar, propongo un filtro razonado, lo marco como **[propuesto, no del paper]** y lo
  llevo a §15 (preguntas abiertas).
- Las **fórmulas y derivaciones marcadas [verificado]** se han comprobado numéricamente en este
  entorno. El código está reproducido en el propio documento para que cualquiera pueda repetirlo.

### 0.1 Cinco hallazgos propios de esta revisión, todos verificados numéricamente

Son los que cambian decisiones de implementación, no adorno:

1. **§3.4 — El coste de préstamo contamina el `vol_spread` con una fórmula cerrada.**
   El sesgo es exactamente `VS ≈ −f·√T/φ(0) = −2,5066·f·√T`. Con una comisión de préstamo de
   50 pb y 30 días, eso son **0,36 puntos de volatilidad**; con una acción difícil de tomar
   prestada al 15 %, **10,8 puntos**. La señal completa de Cremers-Weinbaum vale 1-2 puntos.
   Invirtiendo la fórmula se obtiene además un **estimador gratuito de la comisión de préstamo**.
2. **§2.4 — Un desfase de 1 minuto entre la cotización de la opción y el cierre de la acción
   destruye la señal.** El sesgo inducido es `VS ≈ −ε/(φ(0)·√T) = −2,5066·ε/√T`. Un movimiento de
   **10 pb** del subyacente en el último minuto genera **0,88 puntos de vol a 30 días y 1,81 a 7
   días**: del mismo orden que la señal entera. Cuantifica exactamente lo que denuncia Wallmeier
   (2024). Consecuencia: **usar siempre el forward implícito de la cadena, nunca el cierre de la
   acción**, y **no calcular `vol_spread` con vencimientos cortos**.
3. **§3.5 — El ejercicio anticipado americano sesga el `vol_spread` de forma sistemática y
   negativa**, y explota fuera del dinero: −0,13 puntos de vol en ATM a 60 días, pero **−1,8 puntos
   en K/S = 1,20 y −6,6 en K/S = 1,30**. Esto justifica por sí solo un filtro duro de moneyness.
4. **§8.2 — La regla retail del "0,85 × straddle" es incorrecta y va en la dirección contraria.**
   El straddle ATM es *exactamente* `E^Q|S_T − F|`, y el movimiento de una sigma es
   **1,2533 × straddle**, no 0,85 ×. El 0,85 mezcla dos ajustes distintos (quitar el valor temporal
   no-evento, y convertir `E|X|` en `σ`) y aplica el primero sin hacer el segundo.
5. **§4.2 y §8.3 — Interpolar mal cuesta más que la señal.** Coger "el strike más cercano al 25
   delta" con malla de 5 $ produce **1,9 puntos de error** en el risk reversal; el punto fijo
   delta↔strike lo baja a **0,02**. Simétricamente, para el straddle ATM ocurre lo contrario de lo
   que aconseja el folclore: **interpolar linealmente en el strike es peor que coger el strike más
   cercano** (el straddle tiene un mínimo en `K = F`, luego la recta siempre sobreestima).

---

## 1. Resumen ejecutivo: tabla de señales

Convención de signo del repo (`ARCHITECTURE.md` §3.4): **valor mayor = más alcista**. La columna
"signo" es la relación esperada entre la señal *tal como se define en su sección* y el retorno del
evento.

| # | Señal | §  | Signo | Ventana pre-evento | Magnitud reportada | Fiabilidad | ¿Viable con EOD? |
|---|---|---|---|---|---|---|---|
| 1 | `vol_spread` (Cremers-Weinbaum) | §3 | **+** | [T−5, T−1], acumulado desde T−20 | Q5−Q1 ≈ **50 pb/semana** ⚠; en ventana de anuncio **>1,5 %** (Atilgan) ⚠ | Media-alta, **si** se corrige préstamo | **Sí**, con las tres correcciones de §3.7 |
| 2 | `d_vol_spread` (cambio del anterior) | §3.2 | **+** | Δ en [T−20, T−1] | Predice más que el nivel ⚠ | Media | Sí |
| 3 | `iv_skew_xzz` (Xing-Zhang-Zhao) | §4.1 | **−** | Semanal, persiste 6 meses | Q5−Q1 ≈ **−10,9 %/año** ajustado ⚠ | Media | Sí |
| 4 | `iv_skew_25delta` (risk reversal) | §4.2 | **+** (RR alto = alcista) | [T−10, T−1] | Ver §4.3: no es intercambiable con la anterior | Media | Sí, con punto fijo delta↔strike |
| 5 | `put_call_volume_ratio` | §5.1 | **−** | [T−5, T−1] | Pan-Poteshman con volumen *firmado*: **>40 pb al día siguiente**, >1 %/semana ⚠. Sin firmar: mucho más débil | Baja-media sin firmar | Sí (degradada) |
| 6 | `put_call_oi_ratio` | §5.2 | **−** | Δ en [T−10, T−1] | Fodor-Krieger-Doran: el **cambio** del ratio call/put OI predice a pocas semanas ⚠ | Baja-media | Sí |
| 7 | `option_to_stock_volume` (O/S) | §6 | **−** (¡contraintuitivo!) | [T−5, T−1] | Johnson-So: D1−D10 = **+0,34 %/semana** ⚠ (es decir, O/S alto → retorno bajo) | Media | **Sí**, es la más barata de todas |
| 8 | `iv_term_slope` (des-eventizada) | §7 | Ambiguo (magnitud) | [T−10, T−1] | Predice magnitud, no signo | Media | Sí, con dos vencimientos |
| 9 | `event_iv` σ_E (vol implícita del evento) | §7.2 | Ambiguo (magnitud) | T−1 | Base del movimiento esperado | Alta como medida | Sí |
| 10 | `expected_move` (straddle ATM) | §8 | Ambiguo (magnitud) | T−1 | Sobreestima el movimiento realizado: prima de varianza del evento | Alta como medida | Sí |
| 11 | `oi_buildup_calls/puts` (Δ delta-ponderado) | §9 | **+** débil | [T−10, T−1] | Sin poder incremental tras controlar el retorno pre-evento ⚠ | **Baja** | Sí, con retardo T+1 |
| 12 | `iv_crush_expected` | §7.3 | n/a (para el coste) | T−1 | 30-60 % de caída de la IV frontal ⚠ | Alta como medida | Sí |

**Lectura de conjunto — tres advertencias que hay que interiorizar antes de escribir código:**

1. **Jin, Livnat y Zhang (2012)**: la ventaja informativa de las medidas de opciones sobre el
   retorno del evento **se concentra en eventos NO programados**. Los resultados trimestrales son el
   caso *programado* por excelencia. Cabe esperar magnitudes sustancialmente menores que las de la
   literatura de sección cruzada general.
2. **Muravyev, Pearson y Pollet (2025, JFE)**: al menos **dos tercios** de la predictibilidad
   atribuida a `vol_spread` y a `iv_skew` desaparecen si se excluyen las acciones con comisión de
   préstamo alta. No es informed trading: es un **artefacto de medición** por omitir la comisión de
   préstamo del cálculo de la IV. En el S&P 500 el efecto es menor que en small caps (casi todo es
   *general collateral*), lo que en realidad **es una buena noticia para este proyecto**: nuestro
   universo es precisamente aquel en el que el artefacto es pequeño. Pero también implica que las
   magnitudes reportadas en la literatura, estimadas sobre todo el CRSP, **no se trasladan al
   S&P 500**.
3. **Goncalves-Pinto et al. (2020, Management Science)**: buena parte de la predictibilidad de las
   señales de opciones refleja **presión de precios temporal en la acción**, no información en la
   opción. Su medida DOTS (distancia entre el precio implícito por opciones y el precio negociado)
   da 85 pb de alfa al día siguiente ⚠ y **no depende de que se negocien opciones**. Es decir:
   parte de nuestra señal es reversión a corto plazo disfrazada. **Hay que ortogonalizar contra el
   retorno de los últimos 1-5 días.**

---

## 2. Preliminares: cómo se obtiene una IV correcta a partir de una cadena de fin de día

Todo lo demás depende de esto. Una IV mal calculada no es una señal ruidosa: es una señal **con
sesgo sistemático correlacionado con características de la empresa** (dividendo, coste de préstamo,
nivel de precio), que es exactamente lo que un backtest de sección cruzada convierte en falso alfa.

### 2.1 El objeto central es el **forward implícito**, no el precio de la acción

La paridad put-call para opciones europeas sobre una acción con dividendos discretos `D_k` en `t_k`
y comisión de préstamo `f`:

```
C(K,T) − P(K,T) = DF(T) · ( F(T) − K )

con   F(T) = ( S − Σ_k D_k·DF(t_k) ) · e^{(r − f)·T}
```

`F(T)` agrupa **todo** lo que no es volatilidad: tipo, dividendos esperados, comisión de préstamo y
cualquier otra fricción de carry. La idea clave es que **no hace falta estimarlo por partes**: se
puede leer directamente de la cadena.

```python
def implied_forward(
    chain: pd.DataFrame, expiry: date, r: float, valuation_date: date
) -> tuple[float, float]:
    """Extrae (F, DF) de una cadena de fin de día por regresión de paridad put-call.

    Para cada strike K con call y put cotizadas, C(K) − P(K) = DF·F − DF·K es una recta
    en K. La ordenada da DF·F y la pendiente da −DF. Usar varios strikes en lugar de uno
    solo promedia el ruido de la horquilla y detecta strikes corruptos por el residuo.

    Ponderar por 1/(spread_call + spread_put) para que manden los strikes líquidos, y
    restringir a |K/S − 1| <= 0,10, donde el valor temporal de ambas patas es máximo y el
    ejercicio anticipado americano es despreciable (§3.5).

    Referencia: es la construcción estándar del nivel forward en las metodologías de
    índices de volatilidad (Cboe VIX) y la que evita el sesgo de dividendo implícito
    documentado por Wallmeier (2024).
    """
```

Ventajas, todas materiales:

- **Elimina el sesgo de dividendo.** No hay que predecir el dividendo: el mercado ya lo ha
  descontado en el forward. Wallmeier (2024) documenta que usar un dividendo implícito *promedio*
  (lo que hace OptionMetrics para índices europeos) genera **grupos de desviaciones espurias de la
  paridad**.
- **Elimina el sesgo de comisión de préstamo** que denuncian Muravyev, Pearson y Pollet (§3.4).
- **Elimina el desfase de captura** entre la acción y la cadena (§2.4), porque el forward se
  observa en la misma cotización que las opciones.
- **Es un dato en sí mismo.** La comisión de préstamo implícita, `f = r − ln(F/S_adj)/T`, es una
  señal conocida de retornos negativos, y ahora la tenemos gratis (§3.4).

**Regla de implementación:** `data/options.py` debe devolver, además de la cadena, una tabla
`(expiry → F, DF, implied_borrow)` por ticker y fecha. Todas las IV se calculan sobre `F`, con
Black-76, no sobre `S` con Black-Scholes.

### 2.2 Americanas y dividendos discretos: cuándo importa y cuándo no

Las opciones sobre acciones de EE. UU. son **americanas**. OptionMetrics usa un árbol binomial
Cox-Ross-Rubinstein con dividendos discretos y curva cupón cero; Polygon no documenta su modelo con
ese detalle.

La cuantificación de §3.5 dice algo útil: **cerca del dinero la prima de ejercicio anticipado es
pequeña** (0,1-0,3 puntos de vol) y **lejos del dinero es catastrófica** (hasta 6,6 puntos). Por
tanto hay dos estrategias válidas y una inválida:

- **Válida (barata):** Black-76 sobre el forward + filtro duro `0,90 ≤ K/S ≤ 1,10`. El sesgo
  residual es ≈0,1 puntos de vol, homogéneo entre empresas y en gran medida absorbido por la
  estandarización en sección cruzada.
- **Válida (correcta):** árbol CRR americano con dividendos discretos escrowed, y el forward de
  §2.1 como ancla. Necesaria si se quiere usar strikes fuera del rango anterior (por ejemplo, para
  el 25 delta de un subyacente muy volátil).
- **Inválida:** Black-Scholes europeo sobre el spot con dividendo continuo estimado del histórico,
  y sin filtro de moneyness. Es lo que hace la mayoría del código que circula, y produce
  exactamente los sesgos de §3.4 y §3.5 sumados.

### 2.3 Filtros de calidad de la cadena (obligatorios, no opcionales)

Aplicar en este orden, y **registrar cuántos contratos elimina cada uno**: si un filtro tira más del
40 % de la cadena para un ticker-día, ese ticker-día no debe generar señal, debe generar `NaN`.

```
F1  bid > 0                      (un bid de 0 no es un precio, es la ausencia de un precio)
F2  ask > bid                    (cruces = datos corruptos)
F3  (ask − bid)/mid <= 0,50      (horquilla relativa; ver §3.6 para el efecto en la IV)
F4  mid >= 0,05 $                (por debajo, el tick domina el precio)
F5  no violar cotas de no arbitraje sobre el forward:
        call:  DF·max(F − K, 0) <= mid <= DF·F
        put:   DF·max(K − F, 0) <= mid <= DF·K
F6  0 < T <= 365 días
F7  0,90 <= K/S <= 1,10  para vol_spread   (§3.5)
    0,70 <= K/S <= 1,30  para skew         (con pricer americano)
F8  open_interest > 0            (para las medidas ponderadas por OI)
F9  IV converge en [0,01, 5,00]  (si el solver no converge, es NaN, no un valor por defecto)
F10 excluir los 2 días alrededor de un split o de un dividendo especial
    (la cadena queda con strikes no estándar y ratios distintos de 100)
```

**F10 merece énfasis.** Tras un split o un dividendo especial, la OCC ajusta los contratos y el
multiplicador deja de ser 100 acciones. Un `O/S` calculado con multiplicador 100 sobre contratos
ajustados es puro ruido, y aparece justo en fechas que no son aleatorias.

### 2.4 Hora de captura: el problema que nadie mira y que borra la señal — **[verificado]**

Wallmeier (2024, *Journal of Futures Markets* 44, 854-875) documenta que **OptionMetrics registra
los precios de opciones a las 15:59, no a las 16:00**, mientras que el cierre de la acción es de las
16:00. Un minuto. Su conclusión es que ese desfase "incrementa la variabilidad de los spreads de
volatilidad implícita entre puts y calls" y produce distorsiones grandes cuando el mercado se mueve
en los últimos minutos.

Se puede poner número exacto a eso. Si el subyacente usado para calcular la IV se desvía en una
fracción `ε` del subyacente real en el instante de la cotización de las opciones, y llamamos `φ(0) =
1/√(2π) ≈ 0,39894`, entonces (aproximación de primer orden con vega ATM, y `vega_call = vega_put`
para el mismo strike y vencimiento):

```
VS_sesgo  =  IV_call − IV_put  ≈  − ε / ( φ(0) · √T )  =  − 2,5066 · ε / √T
```

Comprobación numérica frente a inversión exacta de Black-76 (S = 100, σ = 35 %, r = 4,5 %):

| ε (desfase del subyacente) | T = 7 d | T = 30 d | T = 60 d | T = 180 d |
|---|---|---|---|---|
| 5 pb | −0,91 pts vol | −0,44 | −0,31 | −0,18 |
| **10 pb** | **−1,81** | **−0,88** | **−0,62** | **−0,36** |
| 20 pb | −3,62 | −1,75 | −1,24 | −0,73 |
| 50 pb | −9,10 | −4,39 | −3,11 | −1,82 |

El error de la aproximación frente a la inversión exacta es < 0,0005 en todos los casos.

**Compárese con la señal:** el spread de volatilidad de Cremers-Weinbaum entre el quintil extremo
alto y el bajo es del orden de **1-2 puntos de volatilidad**. Un movimiento del subyacente de
**10 pb en el último minuto** —absolutamente rutinario— genera 0,88 puntos de ruido a 30 días y
**1,81 a 7 días**. Y no es ruido blanco: está correlacionado con el retorno intradía final, que a su
vez está correlacionado con el retorno del día siguiente. Es decir, **crea predictibilidad espuria
con el signo correcto**.

Tres consecuencias de diseño, no negociables:

1. **Usar el forward implícito de la propia cadena** (§2.1). Absorbe el desfase por completo, porque
   se estima con las mismas cotizaciones que se van a invertir.
2. **No calcular `vol_spread` con vencimientos de menos de ~20 días.** El sesgo escala como `1/√T`.
   Es contraintuitivo —el vencimiento frontal es el que "contiene el evento"— pero para el
   `vol_spread` lo que importa es la asimetría call/put, no el evento.
3. **Registrar la hora de captura** que declara cada proveedor en los metadatos de la caché. Si un
   día se cambia de proveedor, la serie de `vol_spread` tendrá un salto de nivel que se confundirá
   con una señal.

---

## 3. Volatility spread de Cremers-Weinbaum

> Cremers, M. y Weinbaum, D. (2010). "Deviations from Put-Call Parity and Stock Return
> Predictability". *Journal of Financial and Quantitative Analysis* 45(2), 335-367.

### 3.1 Definición exacta

La idea económica: si las opciones fueran europeas y no hubiera fricciones, la paridad put-call
impondría **una única volatilidad implícita** para la call y la put del mismo strike y vencimiento.
Cualquier diferencia es una violación de la paridad, y refleja **presión de demanda direccional**
que el creador de mercado no ha podido arbitrar (Garleanu, Pedersen y Poteshman, 2009).

Sea `P(i,t)` el conjunto de pares `j = (K_j, T_j)` para los que existen **call y put** en la cadena
del ticker `i` en la fecha `t`, ambas superando los filtros de §2.3:

```
                Σ_{j ∈ P(i,t)}  w[i,j,t] · ( IV_call[i,j,t] − IV_put[i,j,t] )
vol_spread[i,t] = ───────────────────────────────────────────────────────────
                              Σ_{j ∈ P(i,t)}  w[i,j,t]

con   w[i,j,t] = ( OI_call[i,j,t] + OI_put[i,j,t] ) / 2
```

Es decir: **media del diferencial de IV call-put sobre pares casados por strike y vencimiento,
ponderada por el open interest medio del par**.

Puntos que hay que fijar y que el abstract no fija — **[propuesto, no del paper]**, ver §15:

- **La ponderación exacta.** "Ponderada por open interest" admite al menos tres lecturas: la media
  `(OI_c + OI_p)/2`, el mínimo `min(OI_c, OI_p)`, o la suma. Propongo la **media**, que es la
  reproducción más citada, y **el mínimo como variante de robustez**: el mínimo penaliza los pares
  en los que una pata es líquida y la otra un cascarón, que son justo los pares cuyo diferencial de
  IV es menos fiable.
- **Rango de vencimientos.** Propongo `20 ≤ DTE ≤ 90` por lo dicho en §2.4 (el suelo de 20 días
  controla el sesgo de no-sincronía) y porque más allá de 90 días las series individuales son
  ilíquidas.
- **Rango de moneyness.** `0,90 ≤ K/S ≤ 1,10`, por §3.5.
- **Mínimo de pares.** Al menos **3 pares válidos**, por §3.6. Con menos, `NaN`.

Signo: **positivo**. `vol_spread` alto (calls relativamente caras) → retornos superiores.

### 3.2 Variantes que la literatura considera más informativas que el nivel

**a) Cambio del spread (`d_vol_spread`).** El nivel de `vol_spread` es persistente y refleja
características estables de la empresa (dividendo, coste de préstamo, sesgo de demanda estructural).
El **cambio** aísla la presión nueva:

```
d_vol_spread[i,t,k] = vol_spread[i,t] − vol_spread[i,t−k]           k ∈ {1, 5, 20}
```

Cremers y Weinbaum reportan que el cambio del spread tiene poder predictivo propio ⚠ (verificar
contra el paper). Para nuestro caso es además la variante **más limpia frente al artefacto de
préstamo** de §3.4: la comisión de préstamo es persistente, luego se cancela en gran parte al tomar
diferencias.

**b) Spread anormal acumulado pre-evento (Atilgan, 2014).** Ésta es *la* variante para el ángulo B.
Atilgan (2014, *Journal of Banking & Finance* 38, 205-215) documenta que la desalineación entre IV
de calls y puts **crece monótonamente durante el mes previo** al anuncio, y que su dirección
coincide con el signo del retorno del anuncio. La construcción:

```
avs[i,t]  = vol_spread[i,t] − mediana_{u ∈ [t−250, t−30]} vol_spread[i,u]     # spread anormal
cavs[i,τ] = Σ_{s = T−τ}^{T−1} avs[i,s]                                        # acumulado
```

Se usa la **mediana** del periodo de referencia, no la media, porque la serie de `vol_spread` tiene
colas gruesas por días con pocos pares válidos. El periodo de referencia **termina en T−30** para
no contaminarlo con la propia acumulación pre-evento, en paralelo exacto a la ventana de estimación
de `abnormal_returns` del contrato (§3.5 de `ARCHITECTURE.md`).

**c) Spread sólo ATM.** Restringir a `|K/S − 1| ≤ 0,025` y al vencimiento más cercano por encima de
20 días. Más ruidoso (un solo par) pero interpretable y libre de decisiones de ponderación.

### 3.3 Ventana, signo y magnitud

| Aspecto | Valor |
|---|---|
| Ventana pre-evento óptima | Nivel: `T−1`. Acumulado: `[T−20, T−1]`, con la mayor parte de la señal en `[T−5, T−1]` |
| Signo | **Positivo** (calls caras → retorno alto) |
| Magnitud, sección cruzada general | Q5 − Q1 ≈ **50-51 pb/semana** ⚠ (las fuentes secundarias citan ambas cifras), decreciente a lo largo de la muestra |
| Magnitud, ventana de anuncio | Atilgan: el quintil de calls caras supera al de puts caras en **más de 1,5 %** en la ventana de dos días del anuncio ⚠ |
| Dónde funciona mejor | **Opciones líquidas + acción ilíquida**, y entornos de información asimétrica. En el S&P 500 tenemos lo primero pero no lo segundo |
| Persistencia | Decreciente en el tiempo, consistente con arbitraje del mispricing |

La combinación "opciones líquidas, acción ilíquida" es un aviso serio: **el S&P 500 es el universo
donde esta señal debería funcionar peor**. Hay que esperar magnitudes bastante menores que las
publicadas.

### 3.4 El problema serio: la comisión de préstamo — **[verificado]**

> Muravyev, D., Pearson, N.D. y Pollet, J.M. (2025). "Why does options market information predict
> stock returns?". *Journal of Financial Economics* 172, 104047.

Su tesis: la predictibilidad de `vol_spread` y `iv_skew` **no es información**, es un **artefacto de
medición**. Al calcular la IV se omite la comisión de préstamo `f`, y como `f` predice retornos
negativos por sí sola, la IV contaminada hereda esa predictibilidad. Excluyendo las acciones de
comisión alta, la predictibilidad **cae al menos dos tercios**.

La derivación cerrada del sesgo, y su verificación:

Con comisión de préstamo `f`, el forward correcto es `F_f = S_adj·e^{(r−f)T}`, pero el investigador
ingenuo usa `F_0 = S_adj·e^{rT}`. La discrepancia de precio que la paridad impone es
`DF·(F_f − F_0) ≈ −S·f·T`. Al invertir call y put con el forward equivocado, y como
`vega_call = vega_put ≡ ν` para el mismo `(K,T)`, la discrepancia se reparte entre las dos IV:

```
ν · ( IV_call − IV_put )  ≈  − S·f·T

y con la vega ATM  ν ≈ S·√T·φ(0) :

      vol_spread_sesgo  ≈  − f·√T / φ(0)  =  − 2,5066 · f · √T
```

Verificación (S = 100, σ = 35 %, r = 4,5 %, ATM; error de la aproximación frente a la inversión
exacta < 4·10⁻⁵ para `f` ≤ 50 pb y < 9·10⁻⁴ en el caso extremo de `f` = 15 % a 60 días — la
aproximación se degrada, como es de esperar, cuando el sesgo deja de ser pequeño):

| Comisión `f` | T = 30 d, sesgo | T = 60 d, sesgo |
|---|---|---|
| 20 pb (general collateral) | −0,14 pts vol | −0,20 |
| 50 pb | −0,36 | −0,51 |
| 200 pb | −1,44 | −2,04 |
| 500 pb | −3,60 | −5,10 |
| 1500 pb (hard to borrow) | −10,75 | −15,16 |

Con la señal completa valiendo 1-2 puntos, una acción al 2 % de comisión ya aporta un sesgo del
tamaño de la señal.

**Dos usos, uno defensivo y otro ofensivo:**

1. **Defensivo — eliminar el artefacto.** Calcular las IV con el forward implícito de §2.1. El
   forward implícito **ya incorpora `f`**, luego el `vol_spread` resultante está limpio por
   construcción. Es gratis y es estrictamente mejor que "excluir acciones de comisión alta".
2. **Ofensivo — la comisión implícita es una señal.** Invirtiendo la fórmula:

```
implied_borrow[i,t,T] = − φ(0) · vol_spread_ATM[i,t,T] / √T
                      = − 0,39894 · vol_spread_ATM[i,t,T] / √T
```

   o, mejor y sin aproximaciones, directamente del forward: `f = r − ln(F/S_adj)/T`. Esto da un
   proxy diario del coste de préstamo **sin ningún proveedor de securities lending**, que es el dato
   caro que `docs/research/data_sources.md` §8.4 no puede permitirse. Signo **negativo** (comisión
   alta → retorno bajo), ventana larga (1-3 meses), y es un complemento natural de
   `short_interest_delta` con la ventaja decisiva de **no tener el retardo de publicación quincenal
   de FINRA**.

   Es probablemente el resultado con mejor relación valor/coste de todo este documento.

**Nota sobre el S&P 500.** Casi todo nuestro universo es *general collateral* (`f` ≈ 10-50 pb), luego
el artefacto es pequeño en media. Pero **no es pequeño en la cola**, y la cola es justo donde
quedarán los extremos del ranking de `vol_spread`. Excluir esas empresas sería tirar señal; hacerlo
bien con el forward implícito no cuesta nada.

### 3.5 El ejercicio anticipado americano — **[verificado]**

Un sesgo distinto del anterior y que nadie corrige. La paridad put-call **no se cumple para
americanas**: `C_am − P_am ≥ DF(F − K)`, porque la put americana vale más que la europea siempre que
`r > 0`, y la call americana vale más que la europea si hay dividendos. Si se invierte con un modelo
europeo, ambas IV quedan infladas, pero **en distinta medida**, y la diferencia va al `vol_spread`.

Cuantificación con árbol CRR de 2000 pasos frente a inversión Black-Scholes europea (S = 100,
σ = 35 %):

| r | q | T | IV_call | IV_put | `vol_spread` (pts vol) |
|---|---|---|---|---|---|
| 4,5 % | 0 % | 30 d | 0,34996 | 0,35208 | **−0,213** |
| 4,5 % | 0 % | 60 d | 0,34996 | 0,35332 | **−0,336** |
| 4,5 % | 2 % | 30 d | 0,34996 | 0,35112 | −0,117 |
| 4,5 % | 5 % | 60 d | 0,35069 | 0,35015 | **+0,054** |
| 1,0 % | 0 % | 30 d | 0,34996 | 0,35031 | −0,035 |

Y, sobre todo, el perfil por moneyness (r = 4,5 %, q = 3 %, T = 60 d):

| K/S | 0,70 | 0,80 | 0,90 | **1,00** | 1,10 | 1,20 | 1,30 |
|---|---|---|---|---|---|---|---|
| `vol_spread` sesgo (pts vol) | **+1,29** | +0,01 | −0,04 | **−0,13** | −0,46 | **−1,82** | **−6,58** |

Tres lecturas:

1. **En ATM el sesgo es pequeño pero no despreciable** (−0,1 a −0,3 puntos), y **depende del
   dividendo y del tipo**, que varían entre empresas y en el tiempo. Es exactamente el tipo de sesgo
   que un ranking de sección cruzada convierte en factor espurio: se estaría ordenando por
   *rentabilidad por dividendo*.
2. **Fuera del dinero explota.** Una put con `K/S = 1,30` produce −6,6 puntos de sesgo. Cualquier
   `vol_spread` que incluya pares muy dentro/fuera del dinero está midiendo el modelo, no el
   mercado.
3. **Con tipos altos el sesgo crece.** En el régimen 2022-2026 (`r` ≈ 4-5 %) el sesgo es 6 veces
   mayor que en 2010-2021 (`r` ≈ 0,5 %). Una serie larga de `vol_spread` calculada con modelo
   europeo tiene una **tendencia inducida por los tipos**.

**Mitigación:** filtro `0,90 ≤ K/S ≤ 1,10` (residual ≈ 0,1 puntos), o pricer CRR americano. El
filtro es la opción sensata para empezar; el pricer, para la versión final.

### 3.6 Ruido de horquilla y número mínimo de pares — **[verificado]**

El precio se observa como punto medio bid-ask. Si la horquilla es `s`, el error típico del punto
medio es del orden de `s/2`, y se traduce en IV dividiendo por la vega:

| K/S | vega por punto de vol | Ruido de IV (pts) con horquilla 0,02 $ | 0,05 $ | 0,10 $ | 0,25 $ |
|---|---|---|---|---|---|
| 0,90 | 0,0723 | 0,138 | 0,346 | 0,691 | 1,728 |
| 0,95 | 0,0981 | 0,102 | 0,255 | 0,510 | 1,275 |
| **1,00** | **0,1139** | **0,088** | **0,219** | **0,439** | **1,097** |
| 1,05 | 0,1041 | 0,096 | 0,240 | 0,480 | 1,200 |
| 1,10 | 0,0702 | 0,142 | 0,356 | 0,712 | 1,780 |

(S = 100, T = 30 d, superficie con smirk realista.)

El ruido del `vol_spread` de **un solo par** es `√2` veces el de una pata, y con `N` pares
promediados cae como `1/√N`:

| N pares (horquilla 0,05 $) | 1 | 3 | 5 | 10 | 20 |
|---|---|---|---|---|---|
| sd(`vol_spread`) en pts vol | 0,310 | 0,179 | 0,139 | 0,098 | 0,069 |

**Regla operativa:** con la señal objetivo en 1-2 puntos de vol, hace falta `sd < 0,2`, es decir
**≥ 3 pares válidos** con horquilla ≤ 0,05 $, o **≥ 10 pares** si las horquillas son de 0,10 $. Con
menos, `NaN`. Esto es lo que justifica el umbral propuesto en §3.1, y no debe relajarse "para tener
más cobertura": relajarlo mete ruido correlacionado con la iliquidez, que es a su vez un factor.

### 3.7 Receta de implementación

```
1. Cargar cadena EOD del ticker-día. Registrar la hora de captura declarada.
2. Por vencimiento con >= 4 strikes casados: estimar (F, DF) por regresión de paridad (§2.1).
   Guardar implied_borrow = r − ln(F/S_adj)/T.
3. Filtros F1..F10 de §2.3, con F7 = [0,90; 1,10].
4. Invertir IV con Black-76 sobre F (o CRR americano si se quiere el rango ancho).
5. Casar pares (K, T) con call y put ambas válidas. Exigir >= 3 pares.
6. vol_spread = media ponderada por (OI_c + OI_p)/2 del diferencial IV_c − IV_p.
7. Derivar d_vol_spread(1, 5, 20) y cavs (§3.2 b).
8. Ortogonalizar contra: retorno de los ultimos 5 dias (Goncalves-Pinto), implied_borrow
   (Muravyev), rentabilidad por dividendo (§3.5) y log(cap). Lo que quede es la señal.
9. Estandarizar en sección cruzada por cohorte de fecha de evento.
```

El paso 8 no es cosmético: los cuatro controles corresponden a las cuatro explicaciones alternativas
documentadas. Si la señal no sobrevive a ellos, no era una señal.

---

## 4. Skew de volatilidad implícita

Dos definiciones que la literatura y la práctica usan indistintamente **y que no lo son**.

### 4.1 Xing-Zhang-Zhao: el smirk de la literatura académica

> Xing, Y., Zhang, X. y Zhao, R. (2010). "What Does the Individual Option Volatility Smirk Tell Us
> About Future Equity Returns?". *Journal of Financial and Quantitative Analysis* 45(3), 641-662.

```
iv_skew_xzz[i,t] = IV_put_OTM[i,t] − IV_call_ATM[i,t]
```

Selección exacta:

- **Put OTM**: la put con moneyness `K/S` más próximo a **0,95**, dentro de la banda de OTM. Xing,
  Zhang y Zhao clasifican como OTM el rango de moneyness **0,80-0,95 para puts** (simétricamente
  1,05-1,20 para calls) y como ATM el rango **0,95-1,05**.
- **Call ATM**: la call con `K/S` más próximo a **1,00**.
- Vencimiento: el más cercano con **al menos 10 días** hasta expiración; ambas patas del **mismo**
  vencimiento.
- Agregación semanal: media de los valores diarios de la semana, o el valor del último día hábil.
  ⚠ El paper usa frecuencia semanal; el detalle exacto de agregación no consta en el abstract.

**Signo: negativo.** Smirk más pronunciado → retornos futuros menores. El quintil con smirk más
pronunciado rinde **≈10,9 % anual menos** ajustado por riesgo ⚠, con persistencia de al menos
6 meses. Y, lo que más importa para este proyecto: **son las empresas que sufren los peores shocks
de beneficios en el trimestre siguiente**.

**Para respetar la convención del repo** (mayor = más alcista), la feature debe almacenarse con el
signo invertido, o documentarse explícitamente. Propongo:

```
iv_skew_xzz_signal[i,t] = − ( IV_put_OTM[i,t] − IV_call_ATM[i,t] )
```

**El matiz decisivo de Van Buskirk.** Van Buskirk ("Volatility Skew, Earnings Announcements, and the
Predictability of Crashes") encuentra que el skew **identifica qué empresas van a sufrir crashes,
pero SÓLO en ventanas cortas de anuncio de resultados**. Fuera de los periodos de resultados el skew
no predice crashes, ni siquiera incluyendo los periodos de guidance de la dirección. El poder
predictivo es incremental respecto a volatilidad histórica, opacidad del reporting e incluso la
propia sorpresa del trimestre.

Esto es **exactamente el ángulo B de este proyecto** y convierte al skew en una de las señales de
opciones más prometedoras aquí, precisamente porque su horizonte natural coincide con nuestra
ventana de evento. Es también el contrapunto a Jin-Livnat-Zhang (§1): no todas las medidas de
opciones pierden potencia en eventos programados; ésta la gana.

### 4.2 Risk reversal 25-delta

Es la definición de mesa de operaciones, y la que pide el contrato (`iv_skew_25delta`):

```
rr25[i,t,T] = IV( Δ_call = +0,25 )  −  IV( Δ_put = −0,25 )
```

Signo **positivo** = calls OTM más caras que puts OTM = sesgo alcista. Es decir, `rr25` **ya
respeta** la convención del repo sin invertir signo, al contrario que `iv_skew_xzz`. En renta
variable el `rr25` es casi siempre **negativo** (las puts son estructuralmente más caras).

El problema es que **el 25 delta no está listado**: hay que localizarlo entre strikes. Y el delta
depende de la IV, que depende del strike. Es un punto fijo.

**Algoritmo correcto (punto fijo delta↔strike):**

```python
def strike_at_delta(
    grid_k: np.ndarray, grid_iv: np.ndarray, F: float, T: float, target_delta: float
) -> float:
    """Resuelve el strike cuyo delta-call de Black-76 es `target_delta`.

    El delta depende de la IV y la IV depende del strike: es un punto fijo que se resuelve
    con un buscador de raíces sobre K, interpolando la IV en log-moneyness a cada paso.
    Convención: la put de delta −0,25 es el strike de delta-call +0,75.

    `grid_k` es log(K/F) ordenado y `grid_iv` la IV observada en esos puntos.
    """
    def objective(K: float) -> float:
        iv = np.interp(np.log(K / F), grid_k, grid_iv)
        v = iv * np.sqrt(T)
        d1 = (np.log(F / K) + 0.5 * v * v) / v
        return float(norm.cdf(d1) - target_delta)

    return brentq(objective, F * np.exp(grid_k[0]) * 1.01, F * np.exp(grid_k[-1]) * 0.99)
```

**Coste de hacerlo mal — [verificado].** Superficie sintética con smirk realista
(`IV(k) = 0,35 − 0,55k + 1,2k²`, `k = log(K/F)`), S = 100, T = 30 d, `rr25` verdadero = **−7,529
puntos de vol**:

| Malla de strikes | Punto fijo delta↔strike | "Strike más cercano al 25Δ" | Interpolación lineal en delta |
|---|---|---|---|
| 1,00 $ | error **0,002** pts | error 0,380 | error 0,013 |
| 2,50 $ | error **0,017** | error 0,971 | error 0,078 |
| 5,00 $ | error **0,020** | error **1,906** | error 0,288 |

El atajo habitual —coger el strike listado cuyo delta esté más cerca de 0,25— produce **1,9 puntos
de error** con mallas de 5 $, que es más que muchas de las señales que queremos medir, y el error
**no es aleatorio**: depende de dónde caiga el precio respecto a la malla, o sea, del precio de la
acción, que es una característica persistente. Ranking espurio garantizado.

Nótese que el punto fijo **es robusto a la malla** (0,02 puntos incluso con strikes de 5 $): el
problema no es la escasez de strikes, es el redondeo al strike listado.

### 4.3 Por qué las dos definiciones no son intercambiables — **[verificado]**

Sobre la misma superficie sintética:

```
iv_skew_xzz  = IV(K/S = 0,95, put) − IV(ATM call)  =  +3,182 pts vol
− rr25                                             =  +7,529 pts vol
```

Miden **la misma pendiente** pero con **brazos de palanca distintos**: XZZ compara un punto a −5 %
de moneyness contra el dinero (un brazo), el `rr25` compara dos puntos simétricos en delta (dos
brazos, y además más lejanos). El cociente entre ambas **no es constante**: depende de la curvatura
de la sonrisa, que varía entre empresas y en el tiempo, y que crece justo antes de resultados.

Consecuencias:

- **No se pueden mezclar en una serie temporal.** Cambiar de definición a mitad de muestra crea un
  salto de nivel y de escala.
- **Las magnitudes publicadas de XZZ no se aplican al `rr25`** ni al revés.
- **Vale la pena calcular las dos** y guardarlas como features distintas: XZZ porque es la que tiene
  respaldo académico con magnitudes documentadas; `rr25` porque es la que exige el contrato, es
  invariante al nivel de precio de la acción y es comparable entre empresas.
- **Añadir la curvatura como tercera feature.** La diferencia entre ambas *es* una medida de
  curvatura, y "curvas de IV cóncavas" son un predictor documentado de **mayor volatilidad realizada
  post-anuncio** (*Review of Finance* 29(4), 963-, 2025: los straddles y strangles delta-neutrales
  rinden significativamente menos ante curvas cóncavas, lo que indica que el mercado cobra una prima
  por el riesgo gamma del evento).

### 4.4 Ventana, signo y magnitud del skew

| Aspecto | XZZ | RR25 |
|---|---|---|
| Signo | **Negativo** (skew alto → retorno bajo) | **Positivo** (RR alto → retorno alto) |
| Ventana | Semanal, persiste 6 meses | [T−10, T−1] |
| Magnitud | Q5−Q1 ≈ −10,9 %/año ajustado por riesgo ⚠ | Sin magnitud publicada directa; usar XZZ como referencia |
| Concentración temporal | **Van Buskirk: sólo predice crashes en ventana de resultados** | Ídem, por construcción análoga |
| Contaminación por préstamo | **Sí** (Muravyev et al.: cae ≥2/3 excluyendo comisión alta) | Sí, misma corrección de §3.4 |

---

## 5. Ratios put/call

### 5.1 Por volumen

```
pcr_volume[i,t] = Σ_j volume_put[i,j,t] / Σ_j volume_call[i,j,t]
```

Con `j` recorriendo **todos** los contratos del ticker (todos los strikes y vencimientos). Es la
definición estándar de un dato de fin de día.

**Problemas, en orden de gravedad:**

1. **La distribución está acotada por abajo en 0 y no por arriba.** La media de un ratio así no
   significa nada. Trabajar con `log(1 + pcr)` o, mejor, con la **cuota de puts**, que sí está
   acotada y es simétrica:

   ```
   put_share[i,t] = Σ volume_put / ( Σ volume_put + Σ volume_call )   ∈ [0,1]
   ```

2. **El nivel es específico de la empresa.** Un valor de 0,7 puede ser altísimo para un ticker y
   bajísimo para otro. **Nunca usar el nivel en sección cruzada**; usar siempre la desviación
   respecto a la propia historia:

   ```
   pcr_z[i,t] = ( put_share[i,t] − media_{[t−60, t−11]} ) / sd_{[t−60, t−11]}
   ```

   El periodo de referencia **termina en `t−11`** para no incluir la propia ventana pre-evento, y
   debe además **excluir las ventanas ±5 sesiones de los eventos anteriores**, porque si no la
   referencia contiene los picos estacionales de resultados y el z-score los infravalora
   sistemáticamente.

3. **El volumen no está firmado.** Ésta es la limitación de fondo. Un contrato negociado puede ser
   apertura compradora, apertura vendedora, cierre comprador o cierre vendedor, y sólo la primera es
   la de un informado alcista. Pan y Poteshman (2006, *RFS* 19(3), 871-908) construyen el ratio
   **sólo con volumen iniciado por compradores que abren posición nueva** y obtienen que las
   acciones de ratio bajo superan a las de ratio alto en **más de 40 pb al día siguiente y más de
   1 % en la semana** ⚠, atribuyéndolo a información no pública. Ese desglose requiere los datos
   **Cboe Open-Close** (`data_sources.md` §7.2: 750 USD el primer año con descuento académico) y
   **no existe en ningún feed EOD gratuito**.

**Signo: negativo** (más puts → retorno menor). **Ventana: [T−5, T−1]**, coherente con Amin y Lee
(1997), que documentan que la actividad de opciones sube **más de un 10 % en los cuatro días
previos** al anuncio y que la dirección de ese flujo previo **anticipa el signo de la noticia**, con
mayor proporción de posiciones largas iniciadas antes de las buenas noticias.

**Versión degradada honesta con EOD.** Sin firmar, se puede refinar de dos maneras que no cuestan
nada:

```
# a) restringir a contratos con contenido direccional real:
#    OTM (donde el apalancamiento es máximo) y vencimiento corto (donde el informado opera)
pcr_otm[i,t] = Σ_{K/S<0,97} vol_put / Σ_{K/S>1,03} vol_call     con DTE <= 45

# b) ponderar por delta para medir exposición direccional, no contratos
pcr_delta[i,t] = Σ_j |Δ_put[j]|·vol_put[j] / Σ_j Δ_call[j]·vol_call[j]
```

La variante (a) tiene respaldo indirecto: Hilliard, Hilliard y Wu (2026, *RQFA* 66(3), 965-992)
construyen una medida que combina el tamaño monetario del cambio de open interest o volumen con la
probabilidad de expirar OTM, y reportan carteras long-short con retornos brutos **superiores al
60 % anual** ⚠. Esa cifra es lo bastante extraordinaria como para exigir replicación antes de
creerla, pero la dirección del diseño —**concentrarse en OTM de vencimiento corto**— es la correcta
y coincide con la intuición de Black (1975).

### 5.2 Por open interest

```
pcr_oi[i,t] = Σ_j open_interest_put[i,j,t] / Σ_j open_interest_call[i,j,t]
```

**El nivel es casi inútil y el cambio es lo informativo.** El nivel de OI está dominado por
posiciones estructurales (calls cubiertas, collars corporativos, coberturas de fondos) que llevan
meses ahí. Fodor, Krieger y Doran (2011, *Financial Markets and Portfolio Management* 25(3),
265-280) encuentran que es **el cambio reciente del ratio call/put de open interest** el que predice
retornos en las semanas siguientes, incluso controlando por factores tradicionales: aumentos grandes
del OI de calls preceden a retornos significativamente mayores, mientras que los aumentos de OI de
puts preceden a retornos menores pero con una relación **considerablemente menos marcada**.

Esa asimetría (calls sí, puts menos) es coherente con Ge, Lin y Pearson (2016) — véase §6.4 — y con
la interpretación de apalancamiento antes que la de restricción de venta en corto.

```
d_pcr_oi[i,t,k] = log( pcr_oi[i,t] ) − log( pcr_oi[i,t−k] )     k ∈ {5, 10}
```

Signo: **negativo** (aumenta el OI de puts respecto al de calls → bajista). **Trampa PIT crítica:**
el open interest del día `t` lo publica la OCC tras el proceso nocturno y **está disponible la
mañana de `t+1`**. `available_at = apertura de t+1`. Sobre ventanas de 5-10 sesiones, un día es un
error relativo enorme. Ver §10.4 y `data_sources.md` §7.3.

### 5.3 Qué esperar realmente

La honestidad obliga a decir que **el ratio put/call sin firmar es la más débil de las señales de
este documento**. Toda la evidencia fuerte (Pan-Poteshman, Ge-Lin-Pearson) se apoya en volumen
firmado y con desglose apertura/cierre. Lo que se puede construir con EOD es una sombra de eso.
Debe entrar en el compuesto con peso pequeño y sometida a la prueba de ortogonalización de §11.

---

## 6. O/S: ratio volumen de opciones a volumen de acciones

> Roll, R., Schwartz, E. y Subrahmanyam, A. (2010). "O/S: The relative trading activity in options
> and stock". *Journal of Financial Economics* 96(1), 1-17.

### 6.1 Fórmula exacta, con las unidades que casi todo el mundo se salta

```
                    option_volume_contracts[i,t] × 100
option_stock[i,t] = ──────────────────────────────────
                       stock_volume_shares[i,t]
```

- `option_volume_contracts` = **suma de todos los contratos negociados** ese día sobre ese
  subyacente, calls y puts, todos los strikes y vencimientos.
- **× 100** convierte contratos a acciones equivalentes. Sin ese factor el ratio sigue siendo
  monótono y el ranking no cambia, pero deja de ser interpretable y **se rompe** en cuanto aparece
  un contrato ajustado por split con multiplicador distinto de 100 (§2.3, F10).
- `stock_volume_shares` = volumen **consolidado** de la acción (todas las plazas, incluido
  off-exchange). Usar sólo el volumen de la plaza primaria infla el ratio de forma correlacionada
  con la cuota off-exchange, que es a su vez una variable de este proyecto
  (`off_exchange_share_delta`): se estaría creando una correlación artificial entre dos features.

**Variante delta-ponderada** (más cercana a la exposición económica real, y la que recomiendo como
complemento):

```
option_stock_delta[i,t] = Σ_j |Δ[i,j,t]| · volume[i,j,t] × 100 / stock_volume_shares[i,t]
```

**Normalización.** Igual que el put/call ratio, el nivel de O/S es específico de la empresa (depende
de si tiene opciones líquidas, del interés minorista, del peso en índices). En sección cruzada hay
que usar `log(O/S)` estandarizado, o el z-score contra la propia historia con la misma construcción
de referencia de §5.1.

### 6.2 La evidencia contradictoria, ordenada

Ésta es la parte que más contradice el folclore del "unusual options activity", y conviene
separarla en cuatro resultados que **no dicen lo mismo**:

**a) Roll, Schwartz y Subrahmanyam (2010).** Tres hallazgos distintos:
   - El O/S **sube alrededor de los anuncios de resultados**.
   - El **retorno absoluto** post-anuncio está **positivamente** relacionado con el O/S
     pre-anuncio → el O/S pre-evento predice la **magnitud** de la reacción. Ésta es una señal
     no direccional, y es la más robusta de las cuatro.
   - Un O/S más alto predice **retornos anormales post-anuncio MENORES** (signo direccional
     **negativo**), lo que interpretan como que la negociación de opciones mejora la eficiencia del
     mercado (menos PEAD).

**b) Johnson y So (2012, *JFE* 106(2), 262-286).** El **decil más bajo de O/S supera al más alto en
0,34 % por semana** (≈19,3 % anualizado) ⚠. Mecanismo propuesto: los costes de venta en corto
empujan a los informados con noticias **negativas** hacia las opciones, luego un pico de O/S es en
promedio **bajista**. La señal es más fuerte cuando el coste de préstamo es alto o el apalancamiento
de la opción es bajo. Además, **el O/S predice noticias de resultados futuras**, lo cual es
directamente relevante para nuestro `SurpriseModel`.

**c) Ge, Lin y Pearson (2016, *JFE* 120(3), 601-622).** Con volumen **firmado** desmontan el
mecanismo de (b): **no** hay evidencia de que las operaciones ligadas a posiciones cortas sintéticas
sean más informativas que las ligadas a largas sintéticas. Lo que predice son **las compras de calls
que abren posición nueva** (el predictor más fuerte), seguidas de las ventas de calls que cierran
posiciones compradas. Concluyen que el canal dominante es el **apalancamiento incorporado** de las
opciones, no la restricción de venta en corto.

**d) Muravyev, Pearson y Pollet (2025).** El marco general: mucho de lo atribuido a información es
comisión de préstamo. Aplica sobre todo a las medidas de IV, pero contamina también la
interpretación de (b), cuyo mecanismo es precisamente el coste de tomar prestado.

### 6.3 Cómo reconciliar todo esto y qué implementar

La lectura consistente con las cuatro piezas:

- El **componente no direccional está bien establecido**: O/S pre-evento alto → reacción de mayor
  magnitud. Es la parte que hay que implementar con confianza, como predictor de `|CAR|`, no de
  `CAR`.
- El **componente direccional negativo existe empíricamente** (a y b coinciden en el signo) pero
  **su mecanismo está en disputa** (b dice cortos, c dice apalancamiento, d dice medición). Un
  efecto cuyo mecanismo nadie sabe explicar es un efecto que hay que ponderar poco y vigilar mucho.
- El **signo direccional es contrario a la intuición retail**. Cualquiera que implemente "mucho
  volumen de opciones = alcista" está implementando el signo equivocado según toda la literatura
  revisada.

Implementación propuesta: **dos features separadas**, no una.

```
os_magnitude[i,t] = z-score de log(O/S) en [T−5, T−1]     → predice |CAR|, entra en el
                                                            modelo de magnitud
os_direction[i,t] = − z-score de log(O/S) en [T−5, T−1]   → predice CAR con signo negativo,
                                                            peso pequeño, vigilada
```

**Ventana: [T−5, T−1]**, coherente con Amin-Lee y con la evidencia de que la actividad de opciones
se concentra en los cinco días previos.

**Es la señal de opciones más barata de todas.** No necesita IV, ni greeks, ni cadena completa: sólo
el volumen agregado de opciones y el volumen de la acción. Cualquier proveedor que dé agregados
diarios sirve. Debería ser la primera que se implemente.

---

## 7. Estructura temporal de la IV y el earnings IV crush

### 7.1 El modelo de dos componentes

> Dubinsky, A. y Johannes, M. (2006). "Earnings Announcements and Equity Options". Working paper,
> Columbia Business School.
> Dubinsky, A., Johannes, M., Kaeck, A. y Seeger, N.J. (2019). "Option Pricing of Earnings
> Announcement Risks". *Review of Financial Studies* 32(2), 646-687.

El precio de la acción sigue una difusión con volatilidad `σ_d`, más un **salto puntual** en la
fecha del anuncio con desviación típica `σ_E` (bajo la medida riesgo-neutral). La varianza total
hasta el vencimiento `T` es entonces **aditiva**:

```
IV(T)² · T  =  σ_d² · T  +  σ_E² · 1{el anuncio cae antes de T}
```

De aquí sale toda la fenomenología conocida:

- La IV del vencimiento frontal **sube mecánicamente** al acercarse el anuncio: `σ_E²` es una
  constante que se reparte entre cada vez menos tiempo, luego `IV(T) = √(σ_d² + σ_E²/T)` crece
  cuando `T → 0`. **Esto no es una señal, es aritmética.** Cualquier feature de "IV subiendo antes
  de resultados" sin des-eventizar está midiendo el calendario.
- La estructura temporal se **invierte** (frontal por encima de lejano) antes del evento.
- Tras el anuncio, `σ_E` desaparece y la IV frontal cae a `σ_d`: eso es el **IV crush**.

### 7.2 Extracción de `σ_E` y `σ_d` con dos vencimientos — **[verificado]**

**Caso A — ambos vencimientos posteriores al anuncio** (el habitual: `T₁` = frontal, `T₂` =
siguiente). Restando las dos ecuaciones de varianza total, `σ_E²` se cancela:

```
σ_d²  =  ( IV₂²·T₂ − IV₁²·T₁ ) / ( T₂ − T₁ )

σ_E²  =  IV₁²·T₁ − σ_d²·T₁   =  T₁·T₂·( IV₁² − IV₂² ) / ( T₂ − T₁ )
```

**Caso B — `T_pre` vence ANTES del anuncio y `T_post` después** (posible gracias a las weeklies, y
mucho más limpio porque no necesita suponer que `σ_d` es plana en todo el tramo):

```
σ_d   =  IV_pre
σ_E²  =  T_post · ( IV_post² − IV_pre² )
```

Verificación con `σ_d` = 30 % y `σ_E` = 5 % reales (la recuperación es exacta hasta precisión de
máquina en ambos casos):

| T₁ | T₂ | IV₁ | IV₂ | `σ_E` recuperada | `σ_d` recuperada | Crush previsto |
|---|---|---|---|---|---|---|
| 7 d | 35 d | 46,94 % | 34,07 % | 0,0500 | 0,3000 | 36,1 % |
| 3 d | 31 d | 62,78 % | 34,56 % | 0,0500 | 0,3000 | 52,2 % |
| 10 d | 45 d | 42,57 % | 33,21 % | 0,0500 | 0,3000 | 29,5 % |
| 21 d | 49 d | 36,53 % | 32,96 % | 0,0500 | 0,3000 | 17,9 % |

**Requisitos y trampas:**

- **`T` debe medirse en años de calendario** para el término difusivo. Usar días de trading es
  defendible para `σ_d` pero entonces hay que ser consistente en las dos ecuaciones o `σ_E` sale
  sesgada.
- **Ambos vencimientos deben contener el mismo número de anuncios.** Si `T₂` es tan lejano que
  incluye *dos* trimestres, la ecuación cambia a `σ_E²·2` en el segundo término y la resta da
  basura. Con vencimientos separados menos de ~80 días naturales el problema no aparece, pero hay
  que **comprobarlo con el calendario de eventos**, no suponerlo. ORATS documenta explícitamente que
  su procedimiento empieza por "aplicar fechas de anuncio precisas para determinar cuántos anuncios
  corresponden a cada vencimiento".
- **`IV₁²·T₁ > IV₂²·T₂` en el numerador de `σ_d²` da negativo** si la estructura temporal está
  invertida por razones ajenas al evento (por ejemplo, tensión de mercado general). En ese caso
  `σ_d²` sale negativa: **devolver `NaN`, no truncar a cero**. Truncar crea un suelo artificial que
  se activa justo en los días de estrés.
- **Usar IV ATM del forward**, no de un strike fijo, para que la comparación entre vencimientos no
  mezcle puntos distintos de la sonrisa.

### 7.3 El crush esperado, en forma cerrada — **[verificado]**

Tras el anuncio, la varianza restante hasta `T₁` es sólo difusiva, luego la IV cae de `IV₁` a
`σ_d`. La caída relativa prevista:

```
iv_crush_expected  =  1 − σ_d / IV₁  =  1 − √( 1 − σ_E² / ( IV₁² · T₁ ) )
```

Ambas expresiones coinciden exactamente. Con `σ_d` = 30 %, `σ_E` = 5 %:

| DTE del vencimiento frontal | 1 d | 2 d | 5 d | 7 d | 14 d | 30 d | 60 d |
|---|---|---|---|---|---|---|---|
| IV₁ | 100,1 % | 73,9 % | 52,2 % | 46,9 % | 39,4 % | 34,7 % | 32,4 % |
| Crush previsto | 70,0 % | 59,4 % | 42,5 % | 36,1 % | 23,8 % | 13,6 % | 7,5 % |

Estas cifras **encajan con lo que se observa empíricamente**: la literatura de mercado reporta
caídas del 30-60 % en la IV frontal de large caps ⚠, y en particular un estudio sobre 4.200 eventos
de 120 subyacentes muy negociados (2021-2025) mide la caída de la IV a 30 días de la straddle ATM
entre el cierre previo y la apertura posterior ⚠. Que un modelo de dos parámetros reproduzca el
rango observado sin calibrar nada es una validación razonable de que el marco es el correcto.

**Uso en el proyecto:** el crush **no es una señal direccional**, es un **componente del coste**. Si
`EventBacktest` (contrato §3.7) va a mantener una posición en opciones a través del evento, el crush
es la mayor parte del P&L y modelarlo mal invalida el resultado. También sirve como filtro de
calidad: un ticker-día cuyo crush previsto sea absurdo (>90 % o <0 %) tiene la cadena mal.

### 7.4 Des-eventizar la pendiente temporal (`iv_term_slope` del contrato)

La feature `iv_term_slope` **tal cual está definida es casi inservible**, por lo dicho en §7.1: el
frontal sube mecánicamente. Hay tres formas de arreglarla, en orden de calidad:

```
# 1. (MEJOR) pendiente de la volatilidad difusiva, ya sin evento
iv_term_slope_ex[i,t] = σ_d(T₂)[i,t] − σ_d(T₁)[i,t]
#    requiere tres vencimientos: dos para extraer σ_E y el tercero para la pendiente

# 2. pendiente cruda menos su valor típico en el mismo tau de trimestres anteriores
iv_term_slope_adj[i,t] = slope[i,t] − mediana_{q ∈ ultimos 8 trimestres} slope[i, tau(t)]
#    controla el efecto calendario sin necesitar tres vencimientos

# 3. (MINIMO) estandarizar en sección cruzada DENTRO de la cohorte de mismo tau
#    Nunca comparar una empresa en T−2 con otra en T−15.
```

La opción 3 es el suelo: **la estandarización por cohorte de `tau` es obligatoria** para cualquier
feature de opciones pre-evento, no sólo para ésta.

**Señales derivadas de `σ_E` que sí son informativas:**

```
event_vol_surprise[i,t] = ( σ_E[i,t] − mediana_{ultimos 8 trimestres} σ_E[i,·,T−1] )
                          / sd_{ultimos 8 trimestres} σ_E[i,·,T−1]
```

Es decir: **¿está el mercado esperando de este trimestre más incertidumbre de lo normal para esta
empresa?** Señal no direccional (predice magnitud). Requiere 8 trimestres de histórico de opciones;
con menos, `InsufficientHistory`.

### 7.5 Lo que también se sabe de la IV alrededor del evento

- **Hann, Kim y Zheng (2019, *Review of Accounting Studies* 24(3), 927-971)**: hay **transferencia
  de información de segundo momento entre empresas del mismo sector**. El cambio de IV del primer
  anunciante del sector se asocia positivamente con el de sus competidores. Implicación directa:
  **el orden de anuncio dentro del sector es una variable**, y la IV de una empresa que aún no ha
  reportado contiene información del anuncio de su competidor. Es una feature natural que el
  contrato no contempla y que merece añadirse: `peer_iv_change_since_first_reporter`.
- **Barth y So (2014)**: hay una **prima de riesgo de volatilidad no diversificable** en los eventos
  de resultados, concentrada en las empresas *bellwether*. Consecuencia práctica: los straddles
  comprados a través del evento pierden en promedio; **la IV pre-evento sobreestima
  sistemáticamente el movimiento realizado**. Cualquier estrategia que compre volatilidad antes de
  resultados parte con viento en contra, y cualquier `expected_move` debe interpretarse como una
  medida riesgo-neutral, no como una previsión.

---

## 8. Movimiento esperado a partir del straddle ATM

### 8.1 La identidad exacta

En el strike `K = F` (forward, no spot), el straddle paga `|S_T − F|`. Por tanto, **sin ninguna
aproximación ni supuesto de modelo**:

```
straddle(K = F, T)  =  DF(T) · E^Q | S_T − F |
```

El straddle ATM **es** el valor actual del movimiento absoluto esperado bajo la medida riesgo-neutral.
No hay que multiplicarlo por nada para obtener eso.

Bajo lognormal, el valor exacto y su aproximación de orden bajo:

```
E^Q|S_T − F|  =  2·F·( 2·Φ(σ√T / 2) − 1 )  ≈  F · σ√T · √(2/π)  =  0,79788 · F · σ√T
```

Verificación **[verificado]** (error de la aproximación < 0,05 % en todos los casos probados):

| σ | T | straddle | `E|ΔS|` exacto | aprox. 0,79788 | 1σ = `F·σ√T` | straddle / 1σ |
|---|---|---|---|---|---|---|
| 30 % | 30 d | 6,8603 | 6,8603 | 6,8624 | 8,6007 | **0,7976** |
| 60 % | 7 d | 6,6278 | 6,6278 | 6,6297 | 8,3091 | **0,7977** |
| 45 % | 14 d | 7,0296 | 7,0296 | 7,0319 | 8,8131 | **0,7976** |
| 25 % | 60 d | 8,0839 | 8,0839 | 8,0874 | 10,1361 | **0,7975** |

### 8.2 Por qué la regla del "0,85 × straddle" es incorrecta — **[verificado]**

La convención retail más difundida dice: *movimiento esperado = 0,85 × straddle ATM*, presentada
como el rango que contiene el movimiento realizado el 68 % de las veces, es decir, una sigma.

**Es incorrecta, y el error va en la dirección contraria.** De la tabla anterior:

```
straddle = 0,7979 × (1 sigma)     ⇒     1 sigma = 1,2533 × straddle
```

El factor correcto para pasar de straddle a una sigma es **1,2533** (= `√(π/2)`), no 0,85. La regla
del 0,85 subestima la sigma en un **32 %**.

De dónde sale la confusión: el 0,85 mezcla **dos ajustes distintos** y aplica sólo uno.

1. El straddle del vencimiento inmediatamente posterior al anuncio contiene, además del salto del
   evento, el **valor temporal de la difusión** de los días restantes. Quitarlo requiere **bajar**
   el straddle: ése es el ajuste que el 0,85 intenta hacer, a ojo.
2. Convertir `E|ΔS|` en `σ` requiere **subir** por 1,2533.

Se aplica el primero de forma aproximada y se olvida el segundo. El resultado es un número que no es
ni el movimiento absoluto esperado (para eso basta el straddle) ni la sigma (para eso hay que
multiplicar por 1,2533) ni el movimiento del evento (para eso está §7.2).

**Lo correcto, y no cuesta más:**

```
# a) movimiento absoluto esperado TOTAL hasta el vencimiento (sin aproximación)
expected_abs_move[i,t,T] = straddle_atm[i,t,T] / DF(T)

# b) una sigma total hasta el vencimiento
one_sigma_total[i,t,T]   = 1,25331 · straddle_atm[i,t,T] / DF(T)

# c) movimiento del EVENTO, aislado de la difusión (esto es lo que se quiere de verdad)
event_sigma[i,t]         = σ_E[i,t]                                # de §7.2
event_expected_abs_move  = 0,79788 · σ_E[i,t] · S[i,t]
```

La (c) es la construcción correcta del "movimiento implícito por resultados", y **requiere dos
vencimientos**, exactamente como hace la industria: ORATS describe su procedimiento como resolver un
"efecto de resultados" implícito que, una vez extraído de cada vencimiento, deja las volatilidades
ex-earnings alineadas en una estructura temporal racional.

### 8.3 Construir el straddle ATM con strikes discretos — **[verificado]**

`K = F` no está listado. Hay tres formas de aproximarlo y **la intuición habitual falla**:

| Método | T = 7 d, malla 5 $ | T = 30 d, malla 5 $ |
|---|---|---|
| Strike listado más cercano a `F` | **−0,034 %** | **−0,144 %** |
| Interpolación **lineal** en `K` | **+0,508 %** | **+0,496 %** |
| Parábola por los 3 strikes más cercanos | −0,010 % | −0,012 % |

**El strike más cercano es mejor que la interpolación lineal.** La razón es geométrica: el straddle,
como función del strike, tiene un **mínimo** en `K ≈ F` (su derivada es `DF·(1 − 2Φ(d₂))`, que se
anula ahí). Una recta trazada entre dos puntos de una función convexa **siempre queda por encima**,
luego la interpolación lineal **sobreestima sistemáticamente** el movimiento esperado, en un 0,5 %
con mallas de 5 $. El error del strike más cercano, en cambio, es un truncamiento a la baja y es
cuatro veces menor.

La parábola por tres puntos es esencialmente exacta (−0,01 %) y cuesta un `polyfit`. **Es la
recomendación.**

### 8.4 Señales derivadas del movimiento esperado

Ninguna es direccional; todas son de magnitud, y son las que alimentan el modelo de `|CAR|`:

```
# ¿espera el mercado más movimiento del habitual PARA ESTA EMPRESA?
implied_move_z[i,t] = ( event_expected_abs_move[i,t] − media_{8 trimestres} )
                      / sd_{8 trimestres}

# ¿tiende esta empresa a moverse más o menos de lo implícito? (prima de varianza del evento)
move_ratio[i,q] = |retorno realizado del evento q| / event_expected_abs_move[i, T_q − 1]
#   La media histórica de move_ratio por empresa es un dato en sí: Barth-So predice < 1
#   en promedio. Empresas con move_ratio persistentemente > 1 son candidatas a comprar
#   volatilidad; es una señal de estrategia de opciones, no de acciones.

# ¿es el evento la mayor parte de la incertidumbre? (limpieza del experimento)
event_share[i,t] = σ_E[i,t]² / ( IV₁[i,t]² · T₁ )     ∈ [0,1]
#   Valores bajos indican que el vencimiento frontal está dominado por ruido no-evento:
#   ese ticker-día es mal candidato para cualquier señal de opciones pre-evento.
```

---

## 9. Acumulación direccional de open interest

### 9.1 Definición

El open interest bruto no vale: hay que medir **exposición direccional**, no contratos. La
construcción que propongo:

```
# cambio de open interest ponderado por delta y por dólares nocionales
oi_buildup[i,t,k] = Σ_j  Δ[i,j,t] · ( OI[i,j,t] − OI[i,j,t−k] ) · 100 · S[i,t]
                    ─────────────────────────────────────────────────────────
                                    market_cap[i,t]

# y desagregado como pide el contrato:
oi_buildup_calls[i,t,k] = Σ_{j ∈ calls} Δ_j · ΔOI_j · 100 · S / market_cap
oi_buildup_puts[i,t,k]  = Σ_{j ∈ puts}  |Δ_j| · ΔOI_j · 100 · S / market_cap
```

con `k ∈ {5, 10}` y `Δ` el delta de Black-76 sobre el forward. La normalización por capitalización
hace la medida comparable entre empresas; sin ella se está midiendo tamaño.

Signo: **positivo** para `oi_buildup_calls`, **negativo** para `oi_buildup_puts`. Combinada:
`oi_buildup_net = oi_buildup_calls − oi_buildup_puts`.

### 9.2 Evidencia y su alcance

Fodor, Krieger y Doran (2011) es el respaldo principal: el cambio del ratio call/put de open
interest predice retornos en las semanas siguientes, con la asimetría ya mencionada (el efecto de
las calls es claro, el de las puts se diluye con controles). Hilliard, Hilliard y Wu (2026) refinan
la idea combinando el tamaño monetario del cambio de OI con la probabilidad de expirar OTM, y
reportan que **en los días previos al anuncio el open interest tiende a aumentar y después de la
noticia los operadores cancelan parte de sus posiciones** — el patrón de acumulación y desmontaje
que buscamos.

Pero la evidencia contraria de `informed_trading.md` §9.1(h) aplica de lleno: **el poder predictivo
del turnover anormal de opciones pre-anuncio desaparece al controlar por el retorno pre-anuncio** ⚠.
Ésta es la prueba que hay que ejecutar sobre `oi_buildup` antes de darla por buena.

### 9.3 La ambigüedad insalvable

**ΔOI es neto y no está firmado.** Un aumento del OI de calls es *exactamente igual* de compatible
con:

- un informado alcista **comprando** calls para abrir, y
- un tenedor de acciones **vendiendo** calls cubiertas (bajista o neutral), y
- un creador de mercado absorbiendo flujo minorista.

No hay forma de distinguirlos sin datos de dirección. Sólo el **Cboe Open-Close Volume Summary**
(`data_sources.md` §7.2) desagrega compras/ventas de apertura y cierre por tipo de participante, y
es el dato con el que se construyó la literatura seminal (Pan y Poteshman, 2006). **Sin él,
`oi_buildup` es una feature de baja fiabilidad y así debe ponderarse.**

Dos mitigaciones parciales que sí se pueden hacer con EOD:

```
# a) restringir a OTM de vencimiento corto, donde la venta cubierta es menos plausible
#    y el apalancamiento del informado es máximo
oi_buildup_otm = restringir a  0,03 < |Δ| < 0,35  y  DTE <= 45

# b) contrastar ΔOI con el volumen del mismo día: si ΔOI ≈ +volumen, casi todo el
#    volumen fue apertura; si ΔOI ≈ −volumen, casi todo fue cierre. Es una firma
#    parcial y gratuita.
open_ratio[i,j,t] = ( OI[i,j,t] − OI[i,j,t−1] ) / max(volume[i,j,t], 1)   ∈ [−1, 1]
```

La (b) es un truco poco explotado y no cuesta nada: `open_ratio ≈ +1` significa que prácticamente
todo el volumen del día abrió posiciones nuevas. No dice **quién** las abrió, pero sí distingue
apertura de cierre, que es la mitad del desglose que da Cboe Open-Close.

### 9.4 Trampa PIT

El open interest de la sesión `t` se conoce la **mañana de `t+1`**. `available_at = apertura de
t+1`. Con ventanas de 5-10 sesiones, tomarlo como conocido en `t` es un error del 10-20 % de la
ventana. `data/options.py` debe fijar:

```
available_at(open_interest de la sesión t)  =  apertura de t+1
available_at(volumen e IV de la sesión t)   =  cierre de t
```

---

## 10. Todo esto con datos de fin de día

Es la sección operativa: qué se puede y qué no con lo que un particular puede pagar.

### 10.1 Qué se pierde exactamente al pasar de intradía a EOD

| Se pierde | Impacto | ¿Hay sustituto EOD? |
|---|---|---|
| Firma del flujo (comprador/vendedor) | **Grave**: mata Pan-Poteshman y el mecanismo de Ge-Lin-Pearson | Parcial: `open_ratio` de §9.3(b) distingue apertura de cierre |
| Desglose apertura/cierre | **Grave** | Parcial: mismo truco |
| Sincronía opción-acción | **Grave** para `vol_spread` (§2.4) | **Sí, completo**: forward implícito (§2.1) |
| Momento intradía del flujo | Medio | No |
| Horquilla en el momento de la operación | Medio | Parcial: horquilla de cierre |
| Volumen por tipo de participante | Alto (sólo Cboe Open-Close) | No |

Lo importante: **el problema más grave para las señales de IV (la sincronía) tiene solución completa
y gratuita con datos EOD**. Los que no la tienen (firma del flujo) afectan sobre todo a las señales
de volumen, que son las más débiles de todas formas. La conclusión es razonablemente optimista:
**las señales de IV son viables con EOD, las de volumen quedan degradadas**.

### 10.2 Receta EOD completa

Suponiendo una cadena EOD por ticker-día con `strike, expiry, right, bid, ask, volume,
open_interest` (con o sin IV del proveedor):

```
PASO 0. Universo y calendario
  - Miembros del S&P 500 en t (universe.members_on).
  - tradable_date de cada evento (pit.tradable_date). Un día de error aquí lo invalida todo.
  - tau = número de sesiones entre t y el evento. TODA feature se estandariza dentro de
    la cohorte de mismo tau.

PASO 1. Por vencimiento: forward implícito
  - Regresión de paridad sobre strikes con call y put (§2.1) -> (F, DF).
  - implied_borrow = r − ln(F / S_adj) / T.          <- señal gratuita, §3.4
  - Si menos de 4 strikes casados: ese vencimiento no se usa.

PASO 2. Calcular las IV uno mismo
  - Black-76 sobre F, con filtro 0,90 <= K/S <= 1,10 (o CRR americano para el rango ancho).
  - NO usar la IV del proveedor sin auditarla: no se sabe qué spot, qué dividendo ni qué
    hora de captura usó. Si se usa, comparar contra la propia en una muestra y documentar
    la diferencia.

PASO 3. Filtros F1..F10 de §2.3, contabilizando descartes por filtro.

PASO 4. Señales de IV
  - vol_spread + d_vol_spread(1,5,20) + cavs               (§3)
  - iv_skew_xzz  (moneyness 0,95 vs 1,00)                  (§4.1)
  - rr25 por punto fijo delta<->strike                     (§4.2)
  - curvatura = relación entre ambas                       (§4.3)

PASO 5. Estructura temporal (necesita >= 2 vencimientos)
  - Comprobar cuántos anuncios cae en cada vencimiento con el calendario de eventos.
  - sigma_E, sigma_d, iv_crush_expected                    (§7.2-7.3)
  - iv_term_slope_ex o _adj                                (§7.4)
  - event_vol_surprise (necesita 8 trimestres)             (§7.4)

PASO 6. Movimiento esperado
  - straddle ATM por parábola de 3 strikes en K = F        (§8.3)
  - expected_abs_move, one_sigma_total, event_expected_abs_move  (§8.2)
  - implied_move_z, move_ratio, event_share                (§8.4)

PASO 7. Señales de volumen y OI
  - option_stock (O/S) y su versión delta-ponderada        (§6.1)
  - put_call_volume_ratio -> put_share -> z-score          (§5.1)
  - put_call_oi_ratio -> d_pcr_oi                          (§5.2)
  - oi_buildup_calls/puts delta-ponderados, con OI de t disponible en t+1  (§9)
  - open_ratio                                              (§9.3 b)

PASO 8. Higiene estadística
  - Winsorizar al 1 %/99 % por fecha.
  - Estandarizar en sección cruzada DENTRO de la cohorte de tau.
  - Ortogonalizar contra: retorno [t−5, t], implied_borrow, rentabilidad por dividendo,
    log(cap), turnover de la acción.
  - Marcar NaN, nunca imputar: en opciones la ausencia de dato está correlacionada con la
    iliquidez, y la iliquidez es un factor de riesgo. Imputar por la media crea alfa falso.
```

### 10.3 Coste y cobertura realistas

De `data_sources.md` §7, aplicado a lo que exige esta receta:

- **Polygon Options Starter (29 USD/mes)** es el mínimo viable: cadena completa con OI diario, IV y
  greeks, llamadas ilimitadas, 2 años de histórico. Dos años del S&P 500 son ≈4.000 eventos, que
  para una señal con `IC` de 0,03 es potencia justa pero no ridícula.
- **ORATS Delayed (99 USD/mes)** aporta superficie suavizada y, sobre todo, `σ_E` ya extraída — pero
  su cuota de 20.000 peticiones/mes obliga a pedir **sólo fechas dentro de ventanas de evento**
  (`data_sources.md` §7.2 hace la aritmética).
- **Lo que no se puede comprar por debajo de ~750 USD/año**: el desglose apertura/cierre de Cboe.
  Sin él, §5 y §9 quedan permanentemente degradadas.

### 10.4 Lo que NO se puede hacer con EOD, dicho claramente

1. **Reproducir Pan-Poteshman.** Su ratio requiere volumen firmado y abierto. Cualquier
   "put/call ratio" EOD es una aproximación mucho más ruidosa, no la misma señal.
2. **Distinguir compra de call de venta de call cubierta** (§9.3).
3. **Medir presión de precios intradía** ni el momento del flujo dentro de la sesión.
4. **Capturar la reacción de la cadena al anuncio AMC** hasta el cierre siguiente: para un evento
   AMC, la primera cadena post-evento es la de `T+1`, ya con el crush consumado.
5. **Usar el OI del propio día `t` como conocido en `t`** (§9.4).

---

## 11. Evidencia contraria consolidada

Reunida en un solo sitio porque es lo que evita construir un backtest bonito y falso:

| # | Hallazgo contrario | Fuente | Consecuencia de diseño |
|---|---|---|---|
| 1 | Dos tercios de la predictibilidad de `vol_spread` e `iv_skew` es comisión de préstamo omitida | Muravyev-Pearson-Pollet (2025, JFE) | Forward implícito obligatorio (§2.1); `implied_borrow` como control **y** como señal |
| 2 | Parte de la predictibilidad es presión de precios en la ACCIÓN, no información en la OPCIÓN | Goncalves-Pinto et al. (2020, MS) | Ortogonalizar contra el retorno de [t−5, t] |
| 3 | La ventaja de las medidas de opciones se concentra en eventos **no programados** | Jin-Livnat-Zhang (2012, JAR) | Rebajar expectativas: resultados = evento programado |
| 4 | O/S alto predice retornos **menores**, no mayores | Roll et al. (2010); Johnson-So (2012) | Signo negativo en `os_direction`; separar magnitud de dirección |
| 5 | El mecanismo de (4) NO es la restricción de venta en corto sino el apalancamiento | Ge-Lin-Pearson (2016, JFE) | El O/S total es proxy ruidosa; lo informativo son compras de calls de apertura |
| 6 | El turnover anormal de opciones pierde poder al controlar por el retorno pre-anuncio | Literatura opciones/desacuerdo ⚠ | **Prueba de referencia** para `oi_buildup` y `pcr` |
| 7 | Los datos EOD tienen un desfase de captura que genera spreads de IV espurios | Wallmeier (2024, JFM) | Cuantificado en §2.4; forward implícito y `DTE >= 20` |
| 8 | El skew **no** predice crashes fuera de la ventana de resultados | Van Buskirk | A favor: nuestro horizonte es justo el bueno |
| 9 | La IV pre-evento sobreestima el movimiento realizado (prima de varianza del evento) | Barth-So (2014); Review of Finance (2025) | `expected_move` es riesgo-neutral, no una previsión; comprar volatilidad tiene viento en contra |
| 10 | El nivel de `vol_spread` predice mejor donde la acción es **ilíquida**: no es nuestro caso | Cremers-Weinbaum (2010) | Esperar magnitudes menores en el S&P 500 |

---

## 12. Mapeo a `PreEventFeatures` del contrato

`ARCHITECTURE.md` §3.5 nombra cinco features de opciones. Correspondencia y propuesta de ampliación:

| Feature del contrato | Sección | Implementación exacta |
|---|---|---|
| `vol_spread` | §3.1 | Media ponderada por OI del diferencial IV call−put sobre pares casados, con forward implícito |
| `iv_skew_25delta` | §4.2 | `rr25` por punto fijo delta↔strike. **Signo positivo = alcista**, ya alineado con la convención |
| `put_call_volume_ratio` | §5.1 | `put_share` z-scored contra referencia [t−60, t−11] excluyendo ventanas de evento |
| `oi_buildup_calls` / `oi_buildup_puts` | §9.1 | ΔOI delta-ponderado y normalizado por capitalización, `available_at = t+1` |
| `iv_term_slope` | §7.4 | **Des-eventizada.** La versión cruda es aritmética del calendario, no señal |

**Ampliaciones propuestas** (para `docs/OPEN_QUESTIONS.md`, propiedad del orquestador; ninguna
requiere cambiar firmas del contrato, sólo añadir columnas al `DataFrame` que devuelve `compute`):

```
implied_borrow            §3.4   comisión de préstamo implícita — sustituye un dato de pago
d_vol_spread_5/20         §3.2   cambio del spread, más limpio que el nivel
cavs                      §3.2   spread anormal acumulado (Atilgan), la variante de evento
iv_skew_xzz               §4.1   la definición con magnitudes publicadas
iv_curvature              §4.3   diferencia entre las dos definiciones de skew
option_stock              §6.1   O/S — la señal más barata de todas
option_stock_delta        §6.1   O/S ponderado por delta
event_iv (σ_E)            §7.2   volatilidad implícita del evento
event_vol_surprise        §7.4   σ_E frente a su historia de 8 trimestres
iv_crush_expected         §7.3   componente de coste para EventBacktest
expected_abs_move         §8.2   movimiento absoluto esperado
implied_move_z            §8.4   movimiento implícito frente a su historia
event_share               §8.4   qué fracción de la varianza frontal es el evento
open_ratio                §9.3   apertura vs cierre, firma parcial gratuita
peer_iv_change            §7.5   contagio de segundo momento intra-sector (Hann-Kim-Zheng)
```

---

## 13. Tabla PIT: `available_at` por dato de opciones

| Dato | Instante en que es públicamente conocible | Nota |
|---|---|---|
| Bid/ask, último precio | Cierre de `t` (hora exacta según proveedor: 15:59 o 16:00) | Registrar la hora en metadatos de caché |
| Volumen de opciones del día | Cierre de `t` | Revisiones de OCC posibles; usar el fichero definitivo |
| **Open interest** | **Apertura de `t+1`** | Proceso nocturno de la OCC (§9.4) |
| IV y greeks del proveedor | Igual que las cotizaciones de las que derivan | Pero **el modelo es desconocido**: auditar |
| Forward implícito / `implied_borrow` | Cierre de `t` | Se deriva de las cotizaciones del mismo instante |
| Contratos existentes (definiciones) | Fecha de listado del contrato | Polygon `?as_of=` da los contratos **que existían** en la fecha: evita el sesgo de mirar hoy la lista |
| Ajustes por split / dividendo especial | Fecha efectiva del ajuste OCC | Marcar ±2 días como inutilizables (§2.3 F10) |

---

## 14. Reproducción de las verificaciones numéricas

Todas las cifras marcadas **[verificado]** salen de este código, que corre sin red con
`numpy`/`scipy`:

```python
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

def black76(F: float, K: float, T: float, sigma: float, cp: int, df: float = 1.0) -> float:
    """Precio Black-76 sobre el forward. cp = +1 call, −1 put."""
    if T <= 0 or sigma <= 0:
        return df * max(cp * (F - K), 0.0)
    v = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * v * v) / v
    return df * cp * (F * norm.cdf(cp * d1) - K * norm.cdf(cp * (d1 - v)))

def implied_vol(price, F, K, T, cp, df=1.0):
    return brentq(lambda s: black76(F, K, T, s, cp, df) - price, 1e-6, 5.0, xtol=1e-13)

# --- §2.4: sesgo por subyacente no sincrono ---------------------------------
S, r, sigma, T, eps = 100.0, 0.045, 0.35, 30 / 365, 0.001
df, F_real, F_used = np.exp(-r * T), S * np.exp(r * T), S * (1 + eps) * np.exp(r * T)
c, p = black76(F_real, S, T, sigma, +1, df), black76(F_real, S, T, sigma, -1, df)
vs = implied_vol(c, F_used, S, T, +1, df) - implied_vol(p, F_used, S, T, -1, df)
assert abs(vs - (-eps / (norm.pdf(0.0) * np.sqrt(T)))) < 5e-5      # -0,00877

# --- §3.4: sesgo por comision de prestamo -----------------------------------
f = 0.02
F_true, F_naive = S * np.exp((r - f) * T), S * np.exp(r * T)
c, p = black76(F_true, S, T, sigma, +1, df), black76(F_true, S, T, sigma, -1, df)
vs = implied_vol(c, F_naive, S, T, +1, df) - implied_vol(p, F_naive, S, T, -1, df)
assert abs(vs - (-f * np.sqrt(T) / norm.pdf(0.0))) < 5e-5          # -0,01441

# --- §7.2 y §7.3: extraccion del evento y crush -----------------------------
sd, se = 0.30, 0.05
T1, T2 = 7 / 365, 35 / 365
iv1, iv2 = np.sqrt(sd**2 + se**2 / T1), np.sqrt(sd**2 + se**2 / T2)
sd_hat = np.sqrt((iv2**2 * T2 - iv1**2 * T1) / (T2 - T1))
se_hat = np.sqrt(iv1**2 * T1 - sd_hat**2 * T1)
assert abs(se_hat - se) < 1e-12 and abs(sd_hat - sd) < 1e-12
assert abs((1 - sd_hat / iv1) - (1 - np.sqrt(1 - se**2 / (iv1**2 * T1)))) < 1e-12  # 36,1 %

# --- §8.1 y §8.2: straddle y movimiento esperado ----------------------------
F = 100.0
straddle = black76(F, F, T, sigma, +1) + black76(F, F, T, sigma, -1)
assert abs(straddle - 2 * F * (2 * norm.cdf(sigma * np.sqrt(T) / 2) - 1)) < 1e-10
assert abs(straddle / (F * sigma * np.sqrt(T)) - np.sqrt(2 / np.pi)) < 1e-3   # 0,7979
```

Las cifras del ejercicio anticipado (§3.5) requieren además un árbol CRR de 2000 pasos con
`q` continuo; la de interpolación 25-delta (§4.2), una superficie sintética
`IV(k) = 0,35 − 0,55k + 1,2k²`.

**Nota de propiedad de ficheros:** la implementación de referencia y sus tests corresponden a los
propietarios de `earnings_alpha/data/options.py` y `earnings_alpha/events/*.py` según
`ARCHITECTURE.md` §2. Este documento no escribe código en el paquete; las verificaciones anteriores
son reproducibles tal cual y deberían convertirse en tests de esos módulos.

---

## 15. Preguntas abiertas

Candidatas a `docs/OPEN_QUESTIONS.md` (fichero propiedad del orquestador):

1. **Ponderación exacta de Cremers-Weinbaum.** ¿`(OI_c + OI_p)/2`, `min(OI_c, OI_p)` o la suma? El
   abstract sólo dice "ponderando cada par put-call por su open interest". **Verificar contra el
   paper primario** (JFQA 45(2), 335-367) antes de fijar la implementación. Impacto: medio; cambia
   el peso relativo de los pares con una pata ilíquida.
2. **Filtros exactos de CW**: rango de DTE, rango de moneyness, mínimo de open interest, mínimo de
   pares. Todos los que uso en §3.1 son **propuestos, no del paper**.
3. **Agregación semanal de Xing-Zhang-Zhao**: ¿media de los días de la semana o valor del último día
   hábil? Y el filtro de DTE exacto (uso ≥10 días).
4. **Magnitudes por verificar** (todas marcadas ⚠): 50 pb/semana de CW; >1,5 % en ventana de
   anuncio de Atilgan; −10,9 %/año de XZZ; 0,34 %/semana de Johnson-So; >40 pb/día de
   Pan-Poteshman; 85 pb de DOTS; >60 % anual de Hilliard-Hilliard-Wu (ésta especialmente).
5. **Atribución de la evidencia (6) de §11** (el turnover de opciones pierde poder al controlar por
   el retorno pre-anuncio). La referencia más probable es Choy y Wei (2012, *JBF*), sin confirmar.
6. **Magnitud real en el S&P 500.** Toda la literatura estima sobre CRSP completo. Dado
   Cremers-Weinbaum (mejor donde la acción es ilíquida) y Muravyev et al. (mejor donde la comisión
   de préstamo es alta), **cabe esperar que las magnitudes en el S&P 500 sean una fracción de las
   publicadas**. Hay que medirlo, no suponerlo, y calibrar las expectativas de potencia estadística
   con el resultado, no con las cifras publicadas.
7. **¿Compensa el pricer CRR americano frente al filtro de moneyness?** §3.5 sugiere que el filtro
   basta para `vol_spread` pero no para el `rr25` de subyacentes muy volátiles, donde el 25 delta
   puede caer fuera de [0,90; 1,10]. Decidir con una medición sobre datos reales.
8. **Auditoría de la IV del proveedor.** Antes de usar la IV de Polygon o EODHD hay que compararla
   con la propia sobre una muestra y documentar la diferencia sistemática. Si el proveedor usa el
   cierre de la acción y nosotros el forward implícito, la diferencia será exactamente la de §2.4 y
   §3.4, y es grande.
9. **Contagio de segundo momento intra-sector** (Hann-Kim-Zheng): merece feature propia
   (`peer_iv_change_since_first_reporter`), pero exige ordenar los anuncios dentro del sector y
   decidir la definición de "peer" (GICS Sub-Industry de la semilla, presumiblemente).

---

## 16. Referencias

### 16.1 Señales de precios de opciones

- **Cremers, M. y Weinbaum, D. (2010).** "Deviations from Put-Call Parity and Stock Return
  Predictability". *Journal of Financial and Quantitative Analysis* 45(2), 335-367.
  [Cambridge Core](https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/abs/deviations-from-putcall-parity-and-stock-return-predictability/D9BA8F97580328AAFD7988B092FE5D50) ·
  [SSRN 968237](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=968237)
- **Xing, Y., Zhang, X. y Zhao, R. (2010).** "What Does the Individual Option Volatility Smirk Tell
  Us About Future Equity Returns?". *JFQA* 45(3), 641-662.
  [Cambridge Core](https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/abs/what-does-the-individual-option-volatility-smirk-tell-us-about-future-equity-returns/ECFD16BA9ACBDC8D577D1BD866FBEA72)
- **Bali, T.G. y Hovakimian, A. (2009).** "Volatility Spreads and Expected Stock Returns".
  *Management Science* 55(11), 1797-1812.
  [INFORMS](https://pubsonline.informs.org/doi/10.1287/mnsc.1090.1063)
- **Atilgan, Y. (2014).** "Volatility spreads and earnings announcement returns". *Journal of
  Banking & Finance* 38, 205-215.
  [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0378426613004081) ·
  [SSRN 1512046](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1512046)
- **Van Buskirk, A.** "Volatility Skew, Earnings Announcements, and the Predictability of Crashes".
  [SSRN 1740513](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1740513)
- **Jin, W., Livnat, J. y Zhang, Y. (2012).** "Option Prices Leading Equity Prices: Do Option Traders
  Have an Information Advantage?". *Journal of Accounting Research* 50(2), 401-432.

### 16.2 Señales de volumen y open interest

- **Roll, R., Schwartz, E. y Subrahmanyam, A. (2010).** "O/S: The relative trading activity in
  options and stock". *Journal of Financial Economics* 96(1), 1-17.
  [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0304405X09002347) ·
  [SSRN 1410091](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1410091)
- **Johnson, T.L. y So, E.C. (2012).** "The option to stock volume ratio and future returns". *JFE*
  106(2), 262-286.
  [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0304405X12000797) ·
  [PDF del autor](https://www.travislakejohnson.com/pdfs/Johnson%20So%20OS%202012%20(JFE).pdf)
- **Ge, L., Lin, T.-C. y Pearson, N.D. (2016).** "Why does the option to stock volume ratio predict
  stock returns?". *JFE* 120(3), 601-622.
  [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0304405X16000167)
- **Pan, J. y Poteshman, A.M. (2006).** "The Information in Option Volume for Future Stock Prices".
  *Review of Financial Studies* 19(3), 871-908.
  [Oxford Academic](https://academic.oup.com/rfs/article-abstract/19/3/871/1646711) ·
  [PDF MIT](https://www.mit.edu/~junpan/volume.pdf)
- **Amin, K.I. y Lee, C.M.C. (1997).** "Option Trading, Price Discovery, and Earnings News
  Dissemination". *Contemporary Accounting Research* 14(2), 153-192.
  [Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1911-3846.1997.tb00531.x)
- **Fodor, A., Krieger, K. y Doran, J. (2011).** "Do option open-interest changes foreshadow future
  equity returns?". *Financial Markets and Portfolio Management* 25(3), 265-280.
  [Springer](https://link.springer.com/article/10.1007/s11408-011-0164-z) ·
  [SSRN 1634065](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1634065)
- **Hilliard, J.E., Hilliard, J. y Wu, Y. (2026).** "Do short-lived options reveal information
  asymmetry? Evidence from open interest and volume signals". *Review of Quantitative Finance and
  Accounting* 66(3), 965-992.
  [Springer](https://link.springer.com/article/10.1007/s11156-025-01427-z)
- **Easley, D., O'Hara, M. y Srinivas, P.S. (1998).** "Option Volume and Stock Prices: Evidence on
  Where Informed Traders Trade". *Journal of Finance* 53(2), 431-465.
- **Black, F. (1975).** "Fact and Fantasy in the Use of Options". *Financial Analysts Journal* 31(4).

### 16.3 Evidencia contraria y explicaciones alternativas

- **Muravyev, D., Pearson, N.D. y Pollet, J.M. (2025).** "Why does options market information predict
  stock returns?". *Journal of Financial Economics* 172 (octubre 2025).
  [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0304405X25001618) ·
  [SSRN 2851560](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2851560) ·
  [paquete de replicación](https://data.mendeley.com/datasets/n73cyx89gs/2) ·
  [nota de prensa Gies](https://giesbusiness.illinois.edu/news/2026/01/07/study--borrowing-fee-is-the-variable-that-debunks-theory-that-options-predict-future-stock-returns)
- **Goncalves-Pinto, L., Grundy, B.D., Hameed, A., van der Heijden, T. y Zhu, Y. (2020).** "Why Do
  Option Prices Predict Stock Returns? The Role of Price Pressure in the Stock Market".
  *Management Science* 66(9), 3903-3926.
  [SSRN 2695145](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2695145)
- **Wallmeier, M. (2024).** "Quality issues of implied volatilities of index and stock options in the
  OptionMetrics IvyDB database". *Journal of Futures Markets* 44(5), 854-875.
  [Wiley](https://onlinelibrary.wiley.com/doi/full/10.1002/fut.22495) ·
  [SSRN 4025257](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4025257)
- **Garleanu, N., Pedersen, L.H. y Poteshman, A.M. (2009).** "Demand-Based Option Pricing". *RFS*
  22(10), 4259-4299. (Fundamento teórico de por qué la demanda direccional mueve la IV.)
- **Bollen, N.P.B. y Whaley, R.E. (2004).** "Does Net Buying Pressure Affect the Shape of Implied
  Volatility Functions?". *Journal of Finance* 59(2), 711-753.

### 16.4 Volatilidad del evento, estructura temporal y crush

- **Dubinsky, A. y Johannes, M. (2006).** "Earnings Announcements and Equity Options". Working paper,
  Columbia Business School.
  [PDF](https://business.columbia.edu/sites/default/files-efs/pubfiles/6051/DJ_2006.pdf) ·
  [SSRN 600593](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=600593)
- **Dubinsky, A., Johannes, M., Kaeck, A. y Seeger, N.J. (2019).** "Option Pricing of Earnings
  Announcement Risks". *Review of Financial Studies* 32(2), 646-687.
  [PDF VU](https://research.vu.nl/ws/portalfiles/portal/108247883/Option_Pricing_of_Earnings_Announcement_Risks.pdf)
- **Barth, M.E. y So, E.C. (2014).** "Non-Diversifiable Volatility Risk and Risk Premiums at Earnings
  Announcements". *The Accounting Review* 89(5), 1579-1607.
  [PDF HBS](https://www.hbs.edu/faculty/Shared%20Documents/events/14/Barth.Risk_Premiums_and_Non-Diversifiable_Earnings.pdf)
- **"Pricing Event Risk: Evidence from Concave Implied Volatility Curves" (2025).** *Review of
  Finance* 29(4), 963-.
  [Oxford Academic](https://academic.oup.com/rof/article/29/4/963/8079062)
- **Hann, R.N., Kim, H. y Zheng, Y. (2019).** "Intra-industry information transfers: evidence from
  changes in implied volatility around earnings announcements". *Review of Accounting Studies* 24(3),
  927-971.
  [PDF Notre Dame CARE](https://care-mendoza.nd.edu/assets/293764/hann_kim_zheng_rast_oct2018.pdf)

### 16.5 Documentación de industria consultada

- **ORATS — "How ORATS Removes Earnings Effect from Implied Volatility"**:
  [blog.orats.com](https://blog.orats.com/how-orats-removes-earnings-effect-from-implied-volatility) ·
  **"Volatility around earnings"**: [orats.com/university](https://orats.com/university/volatility-around-earnings) ·
  **Core Research API**: [docs.orats.io](https://docs.orats.io/datav2-api-guide/core-research.html)
- **OptionMetrics — metodología IvyDB** (árbol CRR con dividendos discretos y curva cupón cero;
  superficie estandarizada por *kernel smoothing* sobre log(DTE) y delta call-equivalente):
  [optionmetrics.com](https://optionmetrics.com/)
- **OCC — Daily Open Interest**:
  [theocc.com](https://www.theocc.com/market-data/market-data-reports/other-market-data-info/batch-processing/daily-open-interest)
- **Cboe — horario de negociación de opciones sobre acciones** (9:30-16:00 ET):
  [cboe.com](https://www.cboe.com/document/tech-spec/document/technical-specifications/equity-options-extended-trading-hours-faq)

---

## 17. Resumen ejecutivo en seis frases

1. De las doce señales revisadas, las tres con mejor relación evidencia/coste para el S&P 500 son
   **`option_stock` (O/S)** —la más barata, sólo necesita dos volúmenes—, **`iv_skew_xzz`** —cuyo
   poder predictivo se concentra precisamente en la ventana de resultados, según Van Buskirk— y
   **`cavs`**, el spread de volatilidad anormal acumulado de Atilgan.
2. El `vol_spread` de Cremers-Weinbaum es implementable y tiene la mejor teoría detrás, pero **sólo
   si se calcula con el forward implícito de la cadena**: sin eso se está midiendo comisión de
   préstamo y desfase de captura, con sesgos que cuantifico en 0,4-10,8 y 0,9-1,8 puntos de
   volatilidad respectivamente, frente a una señal de 1-2 puntos.
3. La corrección anterior regala un dato que en el mercado cuesta dinero: la **comisión de préstamo
   implícita**, señal bajista conocida y sin el retardo quincenal de FINRA.
4. La volatilidad implícita del evento `σ_E` se extrae **exactamente** de dos vencimientos y de ella
   salen el movimiento esperado correcto, el crush previsto y la sorpresa de incertidumbre; el
   folclore del "0,85 × straddle" es incorrecto y el factor correcto para la sigma es 1,2533.
5. Las señales de **volumen** (put/call, O/S direccional, `oi_buildup`) quedan permanentemente
   degradadas con datos EOD porque falta la firma del flujo; el truco de `open_ratio` recupera la
   mitad del desglose gratis, pero la otra mitad cuesta 750 USD/año de Cboe Open-Close.
6. Y la advertencia que gobierna todo: **el signo intuitivo del volumen de opciones está al revés**
   —más actividad de opciones predice retornos *menores*— y **el S&P 500 es el universo donde estas
   señales deberían funcionar peor**, así que las magnitudes publicadas hay que tratarlas como
   techos inalcanzables, no como objetivos.
