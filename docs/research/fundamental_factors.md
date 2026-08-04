# Factores fundamentales cross-section para el S&P 500

**Ámbito.** Ángulo A de la plataforma: factores fundamentales evaluados de forma
continua sobre el universo S&P 500 completo, con horizonte de decisión
semanal (`W-FRI`) a mensual. Este documento fija la **definición matemática
exacta** de cada factor, su fecha de disponibilidad point-in-time, su horizonte de
decaimiento, la magnitud de efecto que cabe esperar, dónde falla por sector y cómo
neutralizarlo.

**Estado de la evidencia.** Todas las cifras que se atribuyen a un artículo llevan
su referencia completa en la §14. Cuando una cifra procede de resúmenes secundarios
y no ha podido verificarse contra el texto original (el proxy de este entorno
bloquea la descarga de PDFs académicos: `WebFetch` devuelve 403 incluso para
`example.com`), va marcada con **[verificar]**. Cuando una magnitud es un *prior de
ingeniería* mío y no un número publicado, va marcada con **[prior]**. Esta
distinción no es cosmética: un backtest calibrado contra un IC inventado es un
backtest inválido.

**Verificación numérica.** Todas las fórmulas de este documento se han implementado
y ejecutado en un script de validación (fuera del repo, en el scratchpad) para
comprobar que son implementables, dimensionalmente consistentes, invariantes de
escala donde deben serlo, y que los requisitos de histórico declarados son los
reales. Los resultados de esa validación se citan en línea donde son relevantes.

---

## 1. Preliminares: notación, PIT y contrato con el repo

### 1.1 Notación

| Símbolo | Significado |
|---|---|
| `i` | empresa (ticker normalizado, `types.normalize_ticker`) |
| `q` | trimestre fiscal reportado (`EarningsEvent.fiscal_quarter`) |
| `t` | fecha de sesión bursátil (índice del panel canónico) |
| `E_{i,q}` | BPA (EPS) diluido antes de extraordinarios, ajustado por splits |
| `R_{i,q}` | ingresos por acción del trimestre `q` |
| `P_{i,t}` | precio de cierre no ajustado en `t` |
| `MC_{i,t}` | capitalización bursátil en `t` |
| `TA`, `CA`, `CL` | activo total, activo corriente, pasivo corriente |
| `NI`, `CFO`, `CAPEX` | resultado neto antes de extraordinarios, flujo de caja operativo, capex |
| `avail(x)` | instante en que `x` fue públicamente conocible (`FundamentalFact.available_at`) |

### 1.2 Las cuatro fechas que nunca deben confundirse

Este es el error número uno de la literatura aplicada y el que invalida más
backtests:

1. **`period_end`** — cierre del trimestre fiscal. **Nunca** es una fecha de señal.
2. **`announced_at`** — instante del comunicado de resultados (8-K item 2.02).
   Trae EPS e ingresos, y a menudo *guidance*, pero **raramente el estado de flujos
   de caja completo**.
3. **`filed_at` / `accepted_at`** — presentación del 10-Q/10-K en EDGAR. Es la
   primera fecha en que están disponibles CFO, capex, activo corriente, deuda a
   largo plazo: es decir, **todo lo que necesitan accruals, F-Score y FCF yield**.
   Llega típicamente 25-45 días *después* del anuncio de resultados.
4. **`tradable_date`** — primera sesión en que la noticia es explotable
   (`pit.tradable_date`). BMO → mismo día; AMC/DMH tras el cierre → sesión
   siguiente.

**Consecuencia operativa, subestimada de forma sistemática:** SUE y sorpresa de
ingresos son negociables el día del `tradable_date` del evento. Accruals, F-Score,
CFO/NI y FCF yield **no**: su `available_at` es el `accepted_at` del 10-Q, y en
S&P 500 la mediana de ese desfase respecto al anuncio ronda las 4-6 semanas
[prior]. Usar el 8-K para computar accruals es look-ahead salvo que se verifique
que ese 8-K concreto incluía estado de flujos de caja.

**Regla de conversión a fecha de sesión.** `available_at` de un filing debe ser la
*acceptance datetime* de EDGAR, no la `filing date`. Si la aceptación es posterior
a las 16:00 ET, la primera fecha negociable es la sesión siguiente.

### 1.3 Contrato con `earnings_alpha`

- Todo factor implementa `factors.Factor` y devuelve `pd.Series` con MultiIndex
  `(date, ticker)`. **Valor mayor = más alcista.** Los factores cuya lectura
  natural es "menos es mejor" (accruals, apalancamiento, crecimiento de activos)
  se devuelven **con el signo cambiado** y el docstring debe decirlo de forma
  explícita.
- Historia insuficiente → `InsufficientHistory`, **nunca** NaN silencioso en el
  agregado ni valor calculado sobre menos observaciones de las exigidas.
- Denominador estructuralmente indefinido para un sector (margen bruto en bancos)
  → **NaN explícito**, no cero ni imputación. Un NaN se excluye del ranking; un
  cero imputado se convierte en una apuesta sectorial involuntaria.
- Ninguna serie fundamental se indexa por `period_end`. El *as-of join* de
  `pit.asof_join` es la única vía admitida.
- Los valores reexpresados (`FundamentalFact.is_restated == True`) se **descartan**:
  el backtest usa la cifra tal y como se reportó por primera vez.

### 1.4 Receta canónica de acondicionamiento (aplicar por fecha, cross-section)

```
1. Filtrar universo PIT              -> universe.members_on(t)
2. Winsorizar al 1% / 99%            -> signals.winsorize(0.01)
3. Z-score cross-section             -> signals.zscore
4. Neutralizar por regresión         -> signals.neutralize(by=["sector","size","beta"])
      residuo = x - X (X'X)^-1 X' x ,  X = [1, dummies GICS, log(MC), beta]
5. Re-estandarizar el residuo        -> signals.zscore
```

**Validado numéricamente** sobre un panel simulado de 503 nombres con 11 sectores:
la máxima media sectorial en valor absoluto pasa de 1.238 a 2.3e-15, y la
correlación con `log(MC)` de +0.342 a 5.4e-15. Es decir, la receta hace lo que
promete a precisión de máquina, siempre que se use `lstsq` y no una inversión
explícita de `X'X` (mal condicionada con dummies casi colineales).

**Orden importante:** winsorizar *antes* de estandarizar. Al revés, un único
outlier (habitual en `E/P` o en `PACC`) infla la desviación típica y comprime todo
el resto de la sección cruzada hacia cero.

### 1.5 Composición sectorial del universo y su coste

Del fichero `data/seed/sp500_constituents.csv` (503 nombres):

| Sector GICS | N | Sector GICS | N |
|---|---:|---|---:|
| Industrials | 81 | Consumer Staples | 34 |
| Financials | 76 | Utilities | 31 |
| Information Technology | 74 | Real Estate | 31 |
| Health Care | 59 | Materials | 26 |
| Consumer Discretionary | 47 | Communication Services | 23 |
| | | Energy | 21 |

**Financials (76) + Real Estate (31) = 107 nombres = 21.3 % del índice.** Este es
el número que hay que tener presente en todo el documento: cualquier factor que sea
estructuralmente indefinido para financieras e inmobiliarias (accruals de capital
circulante, margen bruto, FCF, rotación de activos, apalancamiento comparable)
pierde de entrada más de una quinta parte del universo. Con 11 sectores y una
media de 46 nombres por sector, la neutralización sectorial por regresión es
viable; la neutralización por *demeaning* dentro de Energy (21 nombres) o
Communication Services (23) es ya estadísticamente frágil.

**Aviso PIT sobre el sector.** El fichero semilla trae el sector GICS **de hoy**.
Usarlo para neutralizar en 2005 es un look-ahead leve pero real: GICS creó Real
Estate en 2016 (escindido de Financials) y reestructuró Telecom en Communication
Services en 2018. Debe registrarse en `docs/OPEN_QUESTIONS.md`.

---

## 2. Familia SUE — sorpresa de resultados estandarizada

El repo ya codifica los cuatro denominadores en `types.SurpriseBasis`
(`SIGMA`, `PRICE`, `ABS_ESTIMATE`, `ANALYST_DISPERSION`). Esta sección da la
fórmula exacta de cada uno.

### 2.1 Modelo de expectativas: paseo aleatorio estacional con deriva

Base de las variantes de series temporales (Foster 1977; Foster, Olsen y Shevlin
1984; Bernard y Thomas 1989, 1990):

```
delta_{i,q} = (1/8) * sum_{j=1..8} ( E_{i,q-j} - E_{i,q-j-4} )      # deriva

E_hat_{i,q}  = E_{i,q-4} + delta_{i,q}                              # expectativa

UE_{i,q}     = E_{i,q} - E_hat_{i,q}                                # sorpresa bruta
```

Foster, Olsen y Shevlin (1984) contrastaron modelos ARIMA estacionales más ricos y
concluyeron que el paseo aleatorio estacional con deriva rinde tan bien como ellos,
que es la razón por la que se ha impuesto como estándar. La variante autorregresiva
de Foster (1977) es:

```
E_{i,q} - E_{i,q-4} = phi_i * ( E_{i,q-1} - E_{i,q-5} ) + delta_i + eps_{i,q}
```

estimada por MCO sobre los últimos 20-24 trimestres (mínimo 16). Aporta poco en
S&P 500 y triplica la exigencia de histórico: **no se recomienda** como modelo
primario.

**Requisito de histórico — exacto.** Para `j = 8` la fórmula necesita `E_{i,q-12}`.
Sumando `E_{i,q}` hacen falta **13 observaciones trimestrales consecutivas**
(`q, q-1, ..., q-12`) ≈ 3.25 años. Verificado numéricamente: con 13 observaciones
el cálculo se completa, con 12 debe lanzarse `InsufficientHistory`.

### 2.2 `SurpriseBasis.SIGMA` — SUE clásico

```
sigma_{i,q}   = sd_{j=1..8} [ ( E_{i,q-j} - E_{i,q-j-4} ) - delta_{i,q} ]     (ddof=1)

SUE_sigma     = UE_{i,q} / sigma_{i,q}
```

- **Adimensional e invariante de escala.** Verificado: multiplicar toda la serie de
  EPS por 7 deja `SUE_sigma` idénticamente en +1.062. Esto lo hace inmune a splits
  mal ajustados en el *nivel*, aunque no a splits mal ajustados en *parte* de la
  serie.
- **Modo de fallo crítico:** `sigma → 0`. Una empresa con beneficios muy estables
  (utilities reguladas, consumo básico) produce una desviación típica minúscula y
  un SUE explosivo por un céntimo de diferencia. Verificado: con una serie
  perfectamente estacional el cociente devuelve `NaN`/`inf` sin aviso.
  **Mitigación obligatoria:** suelo en el denominador,
  `sigma_eff = max(sigma, k * P_{i,t^-})` con `k ≈ 0.005`, o alternativamente el
  percentil 10 de la distribución cross-section de `sigma` en esa fecha. Sin suelo,
  el factor es una máquina de generar outliers concentrados en dos sectores.
- **Segundo modo de fallo:** discontinuidades estructurales. Una fusión grande, una
  desinversión o un cambio de ejercicio fiscal rompen la comparabilidad estacional
  y contaminan `delta` y `sigma` durante 8 trimestres. Filtro recomendado:
  descartar el evento si `|Rev_{i,q} / Rev_{i,q-4} - 1| > 0.60` sin que el sector
  entero se mueva de forma parecida [prior].
- **Variante de sigma sobre errores de predicción.** Algunas implementaciones
  definen `sigma` como la desviación típica de los 8 últimos `UE` realizados. Es
  defendible pero exige `E_{i,q-1..q-20}` (21 observaciones ≈ 5.25 años), lo que
  expulsa del universo a toda incorporación reciente al índice. **Se documenta
  pero no se recomienda como variante por defecto.**

### 2.3 `SurpriseBasis.PRICE` — SUE deflactado por precio

```
SUE_price = ( E_{i,q} - E_hat_{i,q} ) / P_{i,t^-}
```

donde `P_{i,t^-}` es el cierre de la sesión **anterior** al `tradable_date`
(alternativa habitual en la literatura: cierre del último día del trimestre `q`;
ambas son PIT-válidas, pero deben elegirse y documentarse una sola vez).

- Magnitud típica: 1e-4 a 1e-2. Verificado: EPS 1.24 vs 1.10 a 180 USD → +0.000778.
  Conviene multiplicar por 100 al reportar, nunca al rankear.
- **Ventaja decisiva:** definido cuando `sigma → 0` y cuando el histórico es corto.
  Es la variante robusta.
- **Contaminación conocida y obligatoria de tratar:** `SUE_price` es
  mecánicamente el numerador de `E/P` dividido por el mismo precio. Correlaciona
  con el factor *value* por construcción. **Debe ortogonalizarse contra
  `earnings_yield`** (`signals.combine(..., orthogonalize)`) antes de combinar, o
  la cartera resultante es value disfrazado de sorpresa.
- Livnat y Mendenhall (2006) advierten además de que el deflactor por precio es
  vulnerable a diferencias intertemporales y transversales en el PER **[verificar]**:
  el mismo `SUE_price` significa cosas distintas en una acción a PER 8 y en una a
  PER 60. La neutralización por sector mitiga parte, no todo.

### 2.4 `SurpriseBasis.ABS_ESTIMATE` — sorpresa relativa a la estimación

```
SUE_abs = ( A_{i,q} - M_{i,q} ) / | M_{i,q} |
```

Es la definición que usa la prensa financiera ("batió en un 8 %"). **No usar como
factor.** Con `M → 0` explota, y el conjunto de empresas con consenso cercano a
cero no es aleatorio (biotech, cíclicas en el suelo del ciclo): la explosión está
correlacionada con el sector y con el estado del ciclo. Se implementa por
compatibilidad con los datos de proveedores y para reproducir titulares, no como
señal.

### 2.5 SUE basado en analistas

```
SUE_analyst = ( A_{i,q} - M_{i,q} ) / P_{i,t^-}
```

- `A_{i,q}`: EPS **actual** de I/B/E/S (base *street* / pro-forma).
- `M_{i,q}`: **mediana** (no media) de las previsiones individuales emitidas o
  confirmadas en los **90 días** previos al anuncio. La mediana es netamente más
  robusta al analista rezagado que no actualiza.

**Cuatro trampas, todas capaces de invalidar el resultado:**

1. **Base contable incoherente.** `A` debe estar en la misma base que `M`. Restar
   un EPS GAAP de un consenso *street* mide la diferencia entre normas contables,
   no una sorpresa. En S&P 500 la brecha GAAP-street es grande y sistemática por
   sector (tecnología: stock-based compensation; farmacia: amortización de
   intangibles de adquisiciones).
2. **Ficheros ajustados por split.** Los ficheros *summary* de I/B/E/S reexpresan
   las previsiones históricas con los factores de split de hoy, y el redondeo a dos
   decimales genera sorpresas espurias que **correlacionan con la propia
   rentabilidad** (Payne y Thomas 2003). Debe usarse el fichero *detail* sin
   ajustar y aplicar el ajuste de split propio. Este sesgo es de los pocos que
   *fabrican* alfa donde no la hay.
3. **Consenso no point-in-time.** Usar el consenso final —revisado *después* del
   anuncio— es look-ahead puro. Debe leerse de `EstimateSnapshot` con
   `as_of <= tradable_date - 1`.
4. **Gestión del consenso.** Las empresas guían a los analistas hacia abajo para
   luego batir por un céntimo. La distribución de sorpresas es asimétrica con moda
   en +0.01 USD. Esto no invalida el factor pero sí recomienda **rankear**, nunca
   usar el valor bruto, y prestar atención a la cola izquierda (los *misses* son
   informativos de forma muy asimétrica).

**Magnitud del efecto:** Livnat y Mendenhall (2006) documentan que el drift
posterior es **significativamente mayor** cuando la sorpresa se define con
previsiones de analistas de I/B/E/S que con un modelo de series temporales sobre
Compustat. Es el argumento empírico principal para preferir `SUE_analyst` cuando
hay datos de consenso, y `SUE_sigma`/`SUE_price` como respaldo cuando no los hay.

### 2.6 `SurpriseBasis.ANALYST_DISPERSION`

```
SUE_disp = ( A_{i,q} - M_{i,q} ) / max( sd_analistas_{i,q} , 0.005 * P_{i,t^-} )
```

Es la variante más cercana a un estadístico t de la sorpresa. Verificado: el suelo
se activa correctamente con `sd = 0` (n = 2 analistas de acuerdo) evitando la
división por cero.

**Advertencia de confusión, importante:** la dispersión de analistas es *por sí
misma* un predictor de rentabilidad —Diether, Malloy y Scherbina (2002) documentan
que mayor dispersión predice *menor* rentabilidad. Al ponerla en el denominador,
`SUE_disp` mezcla dos señales con signos distintos. Solo debe usarse si la
dispersión entra además como control independiente en la combinación.

### 2.7 Resumen SUE

| Variante | Requisito | Robustez | Uso recomendado |
|---|---|---|---|
| `SIGMA` | 13 trim. | media (explota con σ→0) | secundaria, con suelo en σ |
| `PRICE` | 13 trim. | **alta** | respaldo por defecto sin consenso |
| `ABS_ESTIMATE` | consenso | baja | solo reporting |
| analistas / precio | consenso PIT | **alta** | **primaria** |
| `ANALYST_DISPERSION` | consenso + sd | media | solo con control de dispersión |

---

## 3. Sorpresa de ingresos

**Referencia:** Jegadeesh y Livnat (2006), *Revenue surprises and stock returns*.

```
R_{i,q}      = Ingresos_{i,q} / acciones_ajustadas_{i,q}          # POR ACCION

delta^R      = (1/8) * sum_{j=1..8} ( R_{i,q-j} - R_{i,q-j-4} )

SURGE_{i,q}  = ( R_{i,q} - R_{i,q-4} - delta^R ) / sd_{j=1..8}( R_{i,q-j} - R_{i,q-j-4} )
```

Mismo requisito de 13 trimestres que SUE.

**Por qué "por acción" y no ingresos totales — verificado numéricamente.** Sobre
una serie simulada con una recompra del 10 % en los últimos 4 trimestres, `SURGE`
calculado sobre ingresos por acción da **+0.126** y sobre ingresos totales
**−0.814**: signos opuestos. Con ingresos totales el factor mide, en buena parte,
la política de recompra de la empresa —que es un factor distinto y ya documentado
(*net share issuance*)—. Este error es frecuente y silencioso.

**Por qué el factor añade valor sobre SUE:**

- Los ingresos son mucho menos manipulables que el EPS y no están sujetos a la
  gestión del "batir por un céntimo". Una sorpresa de ingresos es una sorpresa de
  demanda; una sorpresa de EPS puede ser una sorpresa de tipo impositivo efectivo,
  de recompras o de provisiones.
- Jegadeesh y Livnat documentan que las sorpresas de ingresos tienen **poder
  explicativo incremental** sobre las de beneficios, y que el drift posterior al
  anuncio es **más fuerte cuando la sorpresa de ingresos va en el mismo sentido que
  la de beneficios**. De ahí la señal compuesta "doble sorpresa":

```
DOUBLE_{i,q} = sign(SUE) * min( |z(SUE)| , |z(SURGE)| )   si sign(SUE) == sign(SURGE)
             = 0                                            en caso contrario
```

  Este constructo (mínimo de las magnitudes con signo común) es deliberadamente
  conservador: solo puntúa cuando ambas señales concuerdan y penaliza la
  discrepancia poniéndola a cero en lugar de promediarla.

**Variante basada en analistas:** `(Rev_actual − Rev_consenso) / P`. La cobertura
de consenso de ingresos es mucho peor que la de EPS, especialmente antes de ~2005;
verificar cobertura antes de usarla como primaria.

**Dónde falla:** bancos y aseguradoras —"ingresos" no es un concepto homogéneo
(ingresos por intereses brutos vs netos vs ingresos totales según el proveedor);
la comparabilidad temporal se rompe si el proveedor cambia de definición.
Inmobiliarias: usar ingresos por rentas. Empresas muy adquisitivas: el crecimiento
inorgánico contamina la comparación estacional exactamente igual que en SUE.

---

## 4. PEAD — deriva posterior al anuncio de resultados

### 4.1 Definición

```
AR_{i,tau}            = r_{i,tau} - r^benchmark_{i,tau}

CAR_i[tau_1, tau_2]   = sum_{tau=tau_1}^{tau_2} AR_{i,tau}
```

con `tau` medido en **sesiones de trading** desde el `tradable_date` (`tau = 0`).
Benchmark admisible, por orden de preferencia:

1. Cartera emparejada por tamaño y book-to-market (estilo Daniel, Grinblatt, Titman
   y Wermers 1997) — el estándar en la literatura contable.
2. Modelo de mercado o FF3, con ventana de estimación `(-250, -40)` que **termina
   antes** de la ventana de evento (así lo fija `events.abnormal_returns`).
3. Exceso sobre el índice — aceptable solo para diagnóstico rápido.

Ventanas canónicas:

- `[0, +1]` o `[+1, +3]`: reacción inmediata (el "ABR" de la literatura de factores).
- `[+2, +60]`: la deriva propiamente dicha, excluyendo el día de anuncio.
- `[+1, +60]`: variante que incluye la reacción; **no usarla para medir drift**,
  porque mezcla reacción y deriva y sobrestima el efecto explotable.

**Como factor continuo** (ángulo A, no ángulo B), PEAD se implementa como:

```
PEAD_{i,t} = z( SUE_{i,q(t)} )  *  1{ 0 < dias_sesion( t - tradable_date_{i,q} ) <= H }
```

es decir: el z-score de la sorpresa del último evento, vivo durante `H` sesiones y
cero después. `H` es el parámetro que hay que calibrar, y §4.3 explica por qué en
2026 debe ser pequeño.

### 4.2 Magnitud histórica

- **Ball y Brown (1968)** documentan el fenómeno por primera vez: los precios siguen
  derivando en la dirección de la sorpresa después del anuncio.
- **Foster, Olsen y Shevlin (1984)**: en 1974-1981, el signo y magnitud del error de
  predicción de beneficios más el tamaño explican el 85 % conjunto de la variación
  de las derivas post-anuncio; la deriva es persistente en todo el periodo, **sin
  concentrarse en ningún subperiodo** **[verificar]**.
- **Bernard y Thomas (1989)**: el diferencial entre el decil superior e inferior de
  SUE es **positivo en 41 de los 48 trimestres** de 1974-1986, y en 11 de los 16
  trimestres con rentabilidad negativa del índice NYSE. El CAR a 60 sesiones crece
  monótonamente con el decil de SUE, y la amplitud entre deciles extremos
  **decrece con el tamaño** —dato central para nosotros—. La cifra habitualmente
  citada para la cartera de cobertura es ~4 % a 60 sesiones (≈18 % anualizado
  rotando trimestralmente) **[verificar]**.
- **Hou, Xue y Zhang (2020)**, con puntos de corte NYSE y ponderación por
  capitalización —la metodología que más se parece a operar en S&P 500—: el decil
  extremo de sorpresa de resultados a horizontes de 6 y 12 meses rinde **0.19 % y
  0.11 % mensual, con t = 1.65 y t = 1.00** respectivamente. **Ninguno de los dos
  supera el umbral de significación.** Solo la versión de mantenimiento a 1 mes
  aguanta mejor. Este es el resultado más relevante de todo el documento para
  calibrar expectativas.

### 4.3 Erosión del PEAD: la evidencia

Este es el punto que más condiciona el diseño de la plataforma.

| Estudio | Hallazgo |
|---|---|
| Chordia, Subrahmanyam y Tong (2014) | Las rentabilidades de una cartera basada en anomalías prominentes **se reducen aproximadamente a la mitad tras la decimalización** (2001). El descenso se explica por AUM de hedge funds, interés corto agregado y rotación. |
| Martineau (2021, *Rest in Peace PEAD*) | En mercados modernos el precio **incorpora íntegramente la sorpresa el día del anuncio**. El PEAD es **inexistente en valores grandes desde ~2006**; solo recientemente ha desaparecido también en microcaps. El descubrimiento de precios ocurre vía cotizaciones de creadores de mercado de alta frecuencia, sin necesidad de operar. |
| Grégoire y Martineau (2021) | Evidencia directa (2011-2015) de que el ajuste tras la sorpresa se produce por cambios de cotización y no por negociación, incluso en el mercado *after-hours* ilíquido. |
| *Why Has PEAD Declined Over Time?* (working paper CEASA/Columbia) | El motor principal de la atenuación en cuatro décadas es la **caída de la persistencia del propio SUE**: las sorpresas de hoy predicen peor las de mañana que las de los años 80. No es solo arbitraje: la señal fundamental subyacente se ha debilitado **[verificar]**. |
| Ng, Rusticus y Verdi (2008) | Los costes de transacción **restringen el arbitraje informado**: la respuesta al anuncio es menor y el drift posterior mayor en empresas con costes altos. Los beneficios de implementar PEAD **se reducen significativamente** tras costes. Corolario: el PEAD residual vive precisamente donde no se puede capturar. |

**Conclusión para este proyecto, sin adornos.** El S&P 500 es el peor universo
posible para el PEAD clásico: son exactamente los valores grandes, líquidos y con
alta cobertura de analistas en los que Martineau documenta que la anomalía murió
hace veinte años. El diseño honesto es:

1. Tratar `H` (horizonte de vida de la señal) como parámetro a estimar, con prior
   de **3-10 sesiones**, no de 60.
2. Reportar el backtest **partido en subperiodos** (pre-2001, 2001-2010, post-2010).
   Un Sharpe agregado 1996-2025 sobre PEAD es engañoso por construcción.
3. Buscar el alfa en la **sección cruzada de la reacción** (qué eventos reaccionan
   de menos) y en el **ángulo B** (huella pre-evento), no en el drift medio.
4. Exigir intervalo de confianza y Sharpe deflactado a toda métrica (§8 del
   contrato de arquitectura).

---

## 5. Momentum de revisiones de analistas

De los factores de esta lista, el que mejor ha aguantado como señal de alta
frecuencia informativa. Cuatro formulaciones, no equivalentes.

### 5.1 Revisión de consenso deflactada por precio (Chan, Jegadeesh y Lakonishok 1996)

```
REV6_{i,t} = sum_{k=0}^{5}  ( F_{i,t-k} - F_{i,t-k-1} ) / P_{i,t-k-1}
```

`F_{i,t}` = consenso medio de BPA a un año vista (FY1) al cierre del mes `t`.
Suma móvil de 6 revisiones mensuales, cada una deflactada por el precio *de su
propio momento* (no por el precio actual: eso reintroduciría el momentum de precio).

Chan, Jegadeesh y Lakonishok muestran que rentabilidad pasada y sorpresa de
beneficios **predicen derivas grandes de forma independiente la una de la otra**, y
que las previsiones de los analistas responden con lentitud a las noticias pasadas,
especialmente en los valores con peor comportamiento reciente. Ese último punto
implica una asimetría explotable en la cola izquierda.

### 5.2 Índice de difusión / amplitud de revisiones

```
DIFF_{i,t} = ( N_up_{i,t} - N_down_{i,t} ) / N_total_{i,t}
```

sobre una ventana móvil de 1 o 3 meses. Acotado en `[-1, +1]`.

- **Ventaja:** no necesita deflactor, es inmune a errores de ajuste por split y muy
  robusto a un analista atípico.
- **Desventaja:** granularidad pobre. Con 4 analistas solo hay 9 valores posibles;
  con 30 analistas (típico en S&P 500) el problema desaparece. Es decir: en nuestro
  universo `DIFF` es *mejor* que en un universo de small caps, al contrario que casi
  todo lo demás en este documento.
- **Regla de higiene:** exigir `N_total >= 3` y contar como "revisión" solo las
  emisiones nuevas dentro de la ventana, no las estimaciones simplemente vigentes.

### 5.3 Revisión ajustada por innovación (Gleason y Lee 2003)

Gleason y Lee documentan que **el mercado no distingue suficientemente** entre
revisiones que aportan información nueva y revisiones que simplemente convergen
hacia el consenso ya existente. Formalmente, dada una revisión del analista `j`
sobre la empresa `i`, de `F_old` a `F_new`, con consenso previo `C_old`:

```
alta innovacion  <=>  F_new > max(F_old, C_old)   o   F_new < min(F_old, C_old)
baja innovacion  <=>  F_new cae entre F_old y C_old   (mero movimiento hacia el consenso)

REVI_{i,j} = ( F_new - F_old ) / P_i        , computado SOLO sobre alta innovacion
```

La deriva post-revisión se concentra en las revisiones de alta innovación. Gleason
y Lee documentan además que el ajuste de precios es **más rápido y completo** en
analistas "celebridad" (*Institutional Investor All-Stars*) y en empresas con mayor
cobertura, y que **una parte sustancial del ajuste retardado ocurre alrededor de los
siguientes anuncios de resultados y de las siguientes revisiones**.

**Implicación directa para nosotros, y es una mala noticia:** en el S&P 500 la
cobertura es máxima y los analistas *All-Star* concentran su atención ahí. Es
precisamente el segmento donde Gleason y Lee documentan que el ajuste es más rápido
y completo. Igual que con el PEAD, hay que esperar magnitudes pequeñas.

### 5.4 Revisión relativa simple

```
dF_{i,t} = ( F^{FY1}_{i,t} - F^{FY1}_{i,t-21} ) / | F^{FY1}_{i,t-21} |
```

Inestable con consenso próximo a cero, por la misma razón que
`SurpriseBasis.ABS_ESTIMATE`. Solo con winsorización agresiva y sabiendo que la
inestabilidad está correlacionada con el sector.

### 5.5 Consideraciones comunes

- **Horizonte y decaimiento:** es un factor **rápido**. El IC es máximo a 1 mes y
  decae con vida media aproximada de 1-3 meses [prior]; a 6 meses la señal está
  esencialmente agotada, coherente con que Chan-Jegadeesh-Lakonishok midan derivas
  sobre ventanas de 6 meses y no de 24.
- **PIT:** exige `EstimateSnapshot` histórico. Sin fotos as-of no hay factor de
  revisiones: reconstruirlo desde el consenso actual es look-ahead de manual.
- **Solapamiento con precio:** las revisiones siguen al precio con retardo. El
  factor debe ortogonalizarse contra momentum de precio a 1-12 meses, o se estará
  comprando momentum con etiqueta fundamental.
- **Dónde falla:** empresas con cobertura muy baja (raras en S&P 500, pero existen
  tras incorporaciones recientes); sectores con consenso dominado por un único
  driver macro —Energía, donde las revisiones son un reflejo mecánico y común de la
  curva del crudo, no información específica de empresa; por eso la neutralización
  sectorial es especialmente importante aquí—.

---

## 6. Piotroski F-Score

**Referencia:** Piotroski (2000), *Value Investing: The Use of Historical Financial
Statement Information to Separate Winners from Losers*, JAR 38 (Supl.): 1-41.

### 6.1 Los nueve componentes, exactos

Sobre el ejercicio fiscal `t`. `ROA` se deflacta por **activo total al inicio del
ejercicio** (`TA_{t-1}`), no por el activo medio ni el de cierre.

**Rentabilidad (4 puntos)**

| # | Indicador | Definición | 1 punto si |
|---|---|---|---|
| 1 | `F_ROA` | `ROA_t = NI_t / TA_{t-1}` | `ROA_t > 0` |
| 2 | `F_CFO` | `CFO_t / TA_{t-1}` | `> 0` |
| 3 | `F_dROA` | `ROA_t − ROA_{t-1}` | `> 0` |
| 4 | `F_ACCRUAL` | `CFO_t/TA_{t-1}` vs `ROA_t` | `CFO_t/TA_{t-1} > ROA_t` |

**Apalancamiento, liquidez y origen de fondos (3 puntos)**

| # | Indicador | Definición | 1 punto si |
|---|---|---|---|
| 5 | `F_dLEVER` | `LEV_t = DeudaLP_t / ((TA_t + TA_{t-1})/2)` | `LEV_t − LEV_{t-1} < 0` (**baja**) |
| 6 | `F_dLIQUID` | `CR_t = CA_t / CL_t` | `CR_t − CR_{t-1} > 0` |
| 7 | `EQ_OFFER` | emisión de acciones ordinarias en el año previo | **NO** emitió |

**Eficiencia operativa (2 puntos)**

| # | Indicador | Definición | 1 punto si |
|---|---|---|---|
| 8 | `F_dMARGIN` | `GM_t = (Ventas_t − COGS_t)/Ventas_t` | `GM_t − GM_{t-1} > 0` |
| 9 | `F_dTURN` | `AT_t = Ventas_t / TA_{t-1}` | `AT_t − AT_{t-1} > 0` |

```
F_SCORE = suma de los 9 indicadores  ∈ {0,...,9}
```

Alto = 8-9; bajo = 0-1.

**Dos errores que circulan por medio internet y que hay que evitar:**

1. Sustituir el componente 1 (`ROA_t > 0`) por "resultado neto positivo" y **omitir
   `F_dROA`**. La lista canónica de Piotroski es
   `{ROA, dROA, CFO, ACCRUAL, dLEVER, dLIQUID, EQ_OFFER, dMARGIN, dTURN}`.
2. **Invertir el signo de `EQ_OFFER`.** Varias fuentes secundarias afirman que se
   otorga el punto si el patrimonio neto *aumenta*. Es al revés: Piotroski otorga
   el punto a la empresa que **no emitió** capital. El razonamiento es explícito en
   el artículo: una empresa en dificultades que capta capital externo señala su
   incapacidad de generar fondos internos, y hacerlo con la cotización deprimida
   subraya su mala situación financiera. Durante la investigación para este
   documento, un resumen automático de fuentes secundarias devolvió precisamente
   la versión invertida; conviene desconfiar de las fuentes terciarias en este
   punto concreto.

### 6.2 Resultados originales y su letra pequeña

Piotroski (2000) documenta que, **dentro del quintil superior de book-to-market**,
comprar los ganadores esperados (`F ≥ 8`) y vender los perdedores (`F ≤ 1`) habría
generado ~23 % anual en 1976-1996; los valores *value* con F alto batieron al
mercado en 13.4 % anual frente al 5.9 % del quintil value completo **[verificar]**.

**La letra pequeña es lo importante para nosotros:** el propio Piotroski documenta
que el efecto **se concentra en empresas pequeñas y medianas, con baja rotación de
la acción y baja cobertura de analistas**. Es la descripción exacta del complemento
del S&P 500. Aplicar F-Score al S&P 500 esperando 23 % anual es un error de lectura
del artículo, no una extrapolación optimista.

### 6.3 Problema de granularidad — cuantificado

Un score entero en `{0,...,9}` sobre 503 nombres produce empates masivos.
**Verificado por simulación** (binomial(9, 0.55), 503 nombres): solo **8 valores
distintos** ocupados y **146 nombres empatados en la moda**. Con eso no se puede
construir un ranking cross-section decente ni una cartera con pesos diferenciados.

**Mitigaciones, por orden de preferencia:**

1. **F-Score continuo.** Sustituir cada indicador binario por el rank-percentil
   cross-section de la magnitud subyacente (`ROA`, `dROA`, `CFO/TA`, `CFO/TA − ROA`,
   `−dLEV`, `dCR`, `−emisión_neta/MC`, `dGM`, `dAT`), promediar los nueve
   percentiles y re-estandarizar. Conserva la lógica de Piotroski y recupera toda
   la granularidad. Es la variante recomendada para este repo.
2. Usar el F-Score binario solo como **filtro de exclusión** (`F <= 3` fuera del
   lado largo), no como score de ranking.
3. Desempatar con un segundo factor (p. ej. `PACC`) dentro de cada nivel de F.

### 6.4 Horizonte, disponibilidad y sectores

- **Frecuencia:** anual. Con datos trimestrales TTM puede refrescarse cada
  trimestre, pero la información nueva por trimestre es escasa: es un factor
  **lento**, con horizonte de decaimiento de ~12 meses.
- **`available_at`:** el `accepted_at` del **10-K**, con el desfase habitual de
  60-90 días desde el cierre del ejercicio. El componente `EQ_OFFER` requiere el
  estado de flujos de caja (emisión de acciones) o el estado de cambios en el
  patrimonio neto: no está en el 8-K.
- **Dónde falla:**
  - **Financieras (76 nombres) e Inmobiliarias (31):** el ratio corriente
    (`F_dLIQUID`) no tiene sentido en un banco (no hay separación corriente/no
    corriente en el balance bancario), el margen bruto (`F_dMARGIN`) no está
    definido (no hay COGS), y la rotación de activos (`F_dTURN`) es un ratio
    trivialmente bajo y no comparable. **Cuatro de los nueve componentes se caen.**
    Recomendación: F-Score = NaN para GICS 40 y 60, o variante bancaria específica
    con ROA, ROE, ratio de eficiencia, cobertura de morosidad y capital regulatorio.
  - **Utilities:** el apalancamiento alto y estable es el modelo de negocio;
    `F_dLEVER` premia mecánicamente la fase de desapalancamiento del ciclo de
    inversión regulatoria y penaliza el crecimiento de base de activos, que es
    justo lo que crea valor en un negocio regulado por RAB.
  - **Biotech sin ingresos:** `F_dMARGIN` y `F_dTURN` indefinidos.
- **Neutralización:** por sector siempre; el F-Score tiene un fuerte componente
  cíclico común (en una recesión, `dROA` y `dMARGIN` caen a la vez en todo un
  sector), por lo que sin neutralizar es en parte una apuesta de ciclo sectorial.

---

## 7. Accruals de Sloan

**Referencia:** Sloan (1996), *Do Stock Prices Fully Reflect Information in Accruals
and Cash Flows About Future Earnings?*, The Accounting Review 71(3): 289-315.

### 7.1 Método de balance (formulación original)

```
ACC_BS_t = [ (dCA_t - dCASH_t) - (dCL_t - dSTD_t - dTXP_t) - DEP_t ]
           / ( (TA_t + TA_{t-1}) / 2 )
```

| Término | Descripción | Compustat |
|---|---|---|
| `dCA` | Δ activo corriente | `ACT` |
| `dCASH` | Δ efectivo e inversiones a corto | `CHE` |
| `dCL` | Δ pasivo corriente | `LCT` |
| `dSTD` | Δ deuda incluida en pasivo corriente | `DLC` |
| `dTXP` | Δ impuestos a pagar | `TXP` |
| `DEP` | dotación a amortización | `DP` |
| deflactor | activo total **medio** | `AT` |

La lógica: capital circulante operativo neto (excluyendo caja y financiación a
corto) menos amortización.

### 7.2 Método de estado de flujos (recomendado)

**Referencia:** Collins y Hribar (2002), *Errors in Estimating Accruals:
Implications for Empirical Research*, JAR 40(1): 105-134.

```
ACC_CF_t = ( NI_t - CFO_t ) / ( (TA_t + TA_{t-1}) / 2 )
```

con `NI` = resultado antes de extraordinarios (`IB`) y `CFO` = flujo operativo del
estado de flujos (`OANCF`) menos partidas extraordinarias y actividades
discontinuadas (`XIDOC`).

**Por qué es obligatorio para el S&P 500.** Collins y Hribar demuestran que el
método de balance está contaminado por operaciones no operativas —fusiones,
adquisiciones, desinversiones y conversión de divisa— que rompen la articulación
entre balance y cuenta de resultados y sesgan la estimación de accruals. El S&P 500
está poblado precisamente por las empresas más adquisitivas y más multinacionales
del mercado. **Verificado numéricamente** sobre un ejemplo con articulación
imperfecta: `ACC_BS = −0.0410` frente a `ACC_CF = −0.0286`, una discrepancia del
43 % en un caso sin siquiera modelar una adquisición.

**Regla del repo:** implementar ambos; usar `ACC_CF` como primario; usar la
discrepancia `|ACC_BS − ACC_CF|` como **bandera de calidad de datos**
(`DataQualityError` o exclusión si excede un umbral) porque señala años con
actividad corporativa que contamina también SUE y el crecimiento de ventas.

### 7.3 Percent accruals

**Referencia:** Hafzalla, Lundholm y Van Winkle (2011), *Percent Accruals*, The
Accounting Review 86(1): 209-236.

```
PACC_t = ( NI_t - CFO_t ) / | NI_t |
```

Hafzalla et al. documentan que una estrategia sobre *percent accruals* genera
rentabilidades de cobertura anuales **significativamente mayores** que la medida
tradicional, y que la mejora viene **sobre todo del lado largo** (empresas con
accruals bajos). Además: `PACC` selecciona mejor las empresas donde la diferencia
entre previsiones sofisticadas e ingenuas es más extrema, **no depende de la
presencia de partidas extraordinarias**, e identifica valores mal valorados
igual de bien en empresas con pérdidas que en empresas con beneficios —donde el
accrual tradicional escalado por activos falla—.

**Verificado:** `PACC` sigue definido con `NI < 0` (ejemplo: `NI = −200`,
`CFO = 50` → `PACC = −1.25`), gracias al valor absoluto en el denominador.

**Recomendación:** `PACC` como primario, `ACC_CF` como secundario y control.

### 7.4 Signo, magnitud, horizonte y sectores

- **Signo en el repo:** `Factor.compute` exige mayor = más alcista.
  **`accruals_factor = −PACC`** (o `−ACC_CF`). Verificado en el script de
  validación. Documentarlo en el docstring es obligatorio: un signo invertido aquí
  produce un backtest con Sharpe negativo simétrico que se lee como "el factor no
  funciona" cuando en realidad funciona al revés.
- **Magnitud original:** Sloan documenta ~10-12 % anual para la cartera larga en
  accruals bajos / corta en accruals altos **[verificar]**. Es una cifra de 1996
  sobre todo el CRSP; véase §13 para la erosión posterior.
- **Persistencia de los componentes** (el mecanismo económico): en la regresión
  `E_{t+1} = α + β_ACC·ACC_t + β_CF·CF_t`, Sloan estima
  `β_ACC ≈ 0.765` frente a `β_CF ≈ 0.855` **[verificar]**. El componente de caja
  del beneficio es más persistente que el de devengo; el mercado fija precios como
  si ambos lo fueran igual. Ese diferencial de persistencia **es** el factor.
- **Horizonte:** anual/trimestral, decaimiento lento (~12 meses). Reequilibrio
  trimestral suficiente; semanal es puro coste de transacción.
- **Disponibilidad:** `accepted_at` del 10-Q/10-K. **Nunca del 8-K**, porque la
  mayoría de comunicados de resultados no incluyen estado de flujos de caja. Es el
  desfase más fácil de olvidar y el más caro.
- **Dónde falla:**
  - **Financieras (GICS 40):** el concepto de capital circulante operativo no
    existe; las variaciones de la cartera crediticia y de trading dominan `CFO`.
    Los accruals de un banco no miden lo que mide el factor. **Excluir.**
  - **Inmobiliarias (GICS 60):** la amortización es enorme y económicamente poco
    significativa; los accruals salen mecánicamente muy negativos, de modo que
    todas las REITs se agolpan en el lado largo del factor sin contenido
    informativo. **Excluir** o usar FFO en lugar de NI.
  - **Utilities:** intensivas en amortización, sesgo similar aunque más leve;
    neutralizar basta.
  - **Empresas con adquisiciones grandes:** contaminación de articulación
    (§7.2). Filtrar por la bandera de discrepancia.

---

## 8. Calidad de beneficios: CFO / NI

### 8.1 Definición y un resultado incómodo

```
EQ_ratio_t = CFO_t / NI_t
```

**Verificado algebraicamente y numéricamente:** cuando `NI > 0`,

```
CFO/NI  =  1 - (NI - CFO)/NI  =  1 - PACC
```

es decir, **`CFO/NI` y percent accruals son transformaciones monótonas la una de la
otra**. No son dos factores: son el mismo factor con dos escalas. En el script de
validación, `CFO/NI = 1.3333` y `1 − PACC = 1.3333` coinciden exactamente. Cualquier
combinación que incluya ambos con pesos independientes está **duplicando la
exposición** y subestimando su riesgo.

Peor aún, con `NI < 0` la equivalencia se rompe y el ratio se comporta mal:
`NI = −200`, `CFO = +50` → `CFO/NI = −0.25`, un valor **negativo** para una empresa
que genera caja, lo que la coloca erróneamente en el lado corto. `1 − PACC` da
`+2.25`, que es la ordenación correcta.

### 8.2 Recomendación

1. Como **factor**, usar `−PACC` (§7.3). No añadir `CFO/NI` como factor
   independiente.
2. Como **diagnóstico** interpretable en informes, reportar `CFO/NI` restringido a
   `NI > 0`, con `NaN` en el resto.
3. Si se quiere una medida de calidad **no redundante** con accruals, usar variantes
   con contenido distinto:

```
CFO_TA_t      = CFO_t / ((TA_t + TA_{t-1})/2)                     # nivel, no devengo
CASH_CONV_t   = ( CFO_t - CAPEX_t ) / EBITDA_t                    # conversion a caja libre
CFO_VOL_t     = sd( CFO_{t-7..t} / TA )  ó  sd(NI/TA)/sd(CFO/TA)  # volatilidad relativa
```

  El cociente de volatilidades `sd(NI)/sd(CFO)` capta *alisamiento de beneficios*
  —una dimensión de calidad genuinamente distinta del nivel de accruals— y es la
  que mejor complementa a `PACC` [prior].

4. Vinculación con Piotroski: `F_ACCRUAL` (componente 4) es exactamente el signo de
   `CFO/TA − ROA`, o sea el signo de `−ACC`. Es otra duplicidad a vigilar al
   combinar F-Score con el factor de accruals: comparten un noveno de la varianza
   del F-Score binario y bastante más en la versión continua.

### 8.3 Sectores y horizonte

Mismos fallos que accruals (§7.4): financieras e inmobiliarias fuera. Horizonte
lento, ~12 meses. Deflactar por activo medio, no por ventas, para que sea comparable
entre sectores intensivos y ligeros en capital.

---

## 9. Margen y tendencia de margen

### 9.1 Niveles

```
GM_t   = ( Ventas_t - COGS_t ) / Ventas_t              # margen bruto
OM_t   = EBIT_t / Ventas_t                             # margen operativo
GPA_t  = ( Ventas_t - COGS_t ) / TA_t                  # gross profitability (Novy-Marx)
```

**Novy-Marx (2013)**, *The other side of value: The gross profitability premium*,
JFE 108(1): 1-28, documenta que el beneficio bruto escalado por **activo total**
(no por fondos propios, y beneficio **bruto**, no neto) tiene aproximadamente el
mismo poder predictivo de la sección cruzada que el book-to-market, y que las
empresas rentables rinden significativamente más que las no rentables **pese a
cotizar más caras**. Además: controlar por rentabilidad **mejora drásticamente el
rendimiento de las estrategias value, especialmente entre los valores más grandes y
líquidos** —es decir, justo en nuestro universo—. Es de los pocos hallazgos de este
documento que *no* se degrada al subir en capitalización. Recibió el premio
Fama-DFA 2013 del JFE.

**Punto clave sobre el numerador:** Novy-Marx usa beneficio **bruto** deliberadamente,
porque es la línea de la cuenta de resultados menos contaminada por decisiones
discrecionales (I+D contabilizado como gasto, provisiones, partidas
extraordinarias). Cuanto más abajo se baja en la cuenta de resultados, más "limpia"
parece la cifra y más ruido de contabilidad discrecional contiene.

### 9.2 Tendencia de margen

Dos implementaciones, **no equivalentes**:

**(a) Diferencia interanual sobre TTM**

```
dGM_t = GM_TTM_t - GM_TTM_{t-4Q}
```

Simple, pero indistingue una mejora sostenida de un salto por ruido de un trimestre.

**(b) t de la pendiente OLS sobre TTM (recomendado)**

```
Ajustar   GM_TTM_{i,q} = alpha + beta * q + eps        sobre las ultimas k = 8..12 TTM
trend_i   = beta_hat / se(beta_hat)
```

**Verificado numéricamente:** sobre dos series con **idéntica pendiente real**
(+0.004/trimestre) y distinto ruido, el estadístico t da **+35.4** para la serie
limpia y **+4.55** para la ruidosa, mientras que la diferencia simple extremo a
extremo da **+0.0476** y **+0.0671** — es decir, la diferencia simple califica
*mejor* a la serie ruidosa, exactamente al revés de lo deseable. La normalización
por el error estándar penaliza el ruido de forma automática, que es justo lo que
hace falta en un ranking cross-section.

Requisito: ≥ 8 observaciones TTM trimestrales → 11 trimestres de datos brutos.

### 9.3 Señales de Lev-Thiagarajan y Abarbanell-Bushee

**Lev y Thiagarajan (1993)**, JAR 31(2): 190-215, formalizan doce señales
fundamentales extraídas de una revisión de la prensa financiera y de informes de
analistas. Las cuatro más usadas se definen como **indicadores de mala noticia**
(un valor positivo es negativo para la acción):

```
INV  = %d Inventarios   - %d Ventas          # inventario creciendo mas que ventas: malo
AR   = %d Cuentas cobrar- %d Ventas          # cobros creciendo mas que ventas: malo
GM   = %d Ventas        - %d Margen bruto    # ventas creciendo mas que margen: malo
SGA  = %d Gastos SG&A   - %d Ventas          # gastos creciendo mas que ventas: malo
```

**El signo es la trampa habitual:** para cumplir el contrato del repo (mayor = más
alcista) el factor es **`−INV`, `−AR`, `−GM`, `−SGA`**.

**Abarbanell y Bushee (1998)**, The Accounting Review 73(1): 19-45, forman carteras
sobre estas señales (más capex, tipo impositivo efectivo, método de valoración de
existencias, salvedades de auditoría y productividad de la plantilla) y obtienen una
rentabilidad anormal acumulada a 12 meses ajustada por tamaño del **13.2 %**, con
una parte significativa concentrada **alrededor de los siguientes anuncios de
resultados** —lo que conecta directamente estas señales con el ángulo B de esta
plataforma: predicen la sorpresa futura, no solo el retorno—.

Véase §13: existe literatura posterior documentando la desaparición de estas
rentabilidades anormales tras la publicación.

### 9.4 Sectores, horizonte y neutralización

- **Horizonte:** lento; el nivel de margen es casi una constante de la empresa
  (autocorrelación anual muy alta), la tendencia se mueve en trimestres.
  Decaimiento ~6-12 meses [prior].
- **Dónde falla:**
  - **Financieras:** no hay COGS ⇒ `GM`, `GPA`, `INV`, `AR` indefinidos.
    31 + 76 = 107 nombres fuera. **NaN, no cero.**
  - **El nivel de margen es casi puro sector.** Software y farmacia superan el 80 %
    de margen bruto estructuralmente; distribución y retail están por debajo del
    20 %. Sin neutralizar sectorialmente, `GM` es un ETF de tecnología con otro
    nombre. La `GPA` de Novy-Marx sufre menos porque el activo total en el
    denominador contrarresta parcialmente, pero también requiere neutralización.
  - **Semiconductores y materiales:** el margen tiene un ciclo propio de 2-4 años;
    la tendencia mide fase del ciclo, no calidad. Neutralizar a nivel de
    *GICS Sub-Industry* (disponible en el fichero semilla) allí donde haya
    suficientes nombres, y si no, a nivel de sector.
  - **Empresas con cambios de política contable en COGS** (reclasificación de
    costes de distribución entre COGS y SG&A): produce un salto artificial en la
    tendencia. La bandera de discrepancia de §7.2 no lo detecta; conviene un filtro
    de salto (`|dGM| > 10 pp` en un trimestre sin cambio de ventas) [prior].

---

## 10. Crecimiento de ventas y aceleración

### 10.1 Definiciones

```
g_TTM_{i,q}  = Rev_TTM_{i,q} / Rev_TTM_{i,q-4} - 1                     # nivel, interanual
g_Q_{i,q}    = Rev_{i,q} / Rev_{i,q-4} - 1                             # trimestral interanual
a_{i,q}      = g_Q_{i,q} - g_Q_{i,q-1}                                 # aceleracion (2a diferencia)
a_std_{i,q}  = a_{i,q} / sd( g_Q_{i,q-8..q-1} )                        # aceleracion estandarizada
```

Sobre ingresos **por acción** por la razón de §3. Requisito: 13 trimestres para
`a_std` (verificado: con 12 debe lanzar `InsufficientHistory`).

### 10.2 El nivel de crecimiento tiene signo NEGATIVO — y esto es contraintuitivo

Este es el punto donde la intuición del inversor discrecional y la evidencia
académica chocan de frente:

- **Lakonishok, Shleifer y Vishny (1994)**, *Contrarian Investment, Extrapolation,
  and Risk*, JF 49(5): 1541-1578: las estrategias value rinden más porque explotan
  el comportamiento subóptimo del inversor típico —que extrapola el crecimiento
  pasado hacia el futuro— y **no** porque sean más arriesgadas. Los valores
  *glamour*, definidos entre otras cosas por alto crecimiento de ventas pasado,
  **rinden menos**.
- **Cooper, Gulen y Schill (2008)**, *Asset Growth and the Cross-Section of Stock
  Returns*, JF 63(4): 1609-1651: la tasa de crecimiento anual del activo es un
  predictor **negativo**, económica y estadísticamente significativo, de la sección
  cruzada de rentabilidades en EE. UU., que **mantiene su capacidad predictiva
  incluso en valores de gran capitalización** y sobrevive al control por
  book-to-market, tamaño, rentabilidades rezagadas, accruals y otras medidas de
  crecimiento. Es uno de los pocos factores documentados que **no** se degrada al
  restringirse a large caps: relevante para nosotros.

**Consecuencia de implementación:**

```
sales_growth_factor  = - z( g_TTM )          # signo negativo, DOCUMENTARLO
asset_growth_factor  = - z( TA_t / TA_{t-1} - 1 )
```

o, alternativamente, ortogonalizar `g_TTM` contra `earnings_yield` y `book_to_market`
y usar el residuo con el signo que salga del estudio interno. Lo que **no** es
defendible es meter el crecimiento de ventas con signo positivo "porque crecer es
bueno".

### 10.3 La aceleración: honestidad sobre el estado de la evidencia

La aceleración (segunda derivada) es una señal **muy popular entre practicantes**
—se asocia al estilo Druckenmiller y a los sistemas tipo CANSLIM, y aparece
implementada en herramientas comerciales como indicador de "Earnings & Sales
Acceleration"— pero su **respaldo en literatura revisada por pares es
considerablemente más débil** que el del nivel de crecimiento o el de los accruals.
No he localizado un artículo de referencia en JF / JFE / JAR / RFS que aísle la
aceleración de ingresos como factor independiente con rentabilidades ajustadas al
riesgo replicadas.

**Tratamiento recomendado en este repo:**

1. Implementarla como **factor exploratorio**, marcado como tal en el docstring.
2. **No** asignarle un prior de IC extraído de la literatura, porque no existe.
3. Validarla con el arsenal completo de `stats`: CV purgada con embargo, Sharpe
   deflactado de Bailey-López de Prado, corrección de Benjamini-Hochberg junto con
   el resto de factores candidatos. Es exactamente el tipo de señal que produce
   falsos positivos si se la evalúa aislada.
4. Racional económico plausible: si el nivel de crecimiento es negativo por
   extrapolación (§10.2) y la aceleración es positiva por infrarreacción a un
   cambio de régimen, la aceleración debe entrar **ortogonalizada contra el nivel**
   o los dos efectos se cancelan parcialmente.

### 10.4 Sectores y horizonte

- **Horizonte:** el nivel es lento (12 meses); la aceleración es rápida por
  construcción (se agota en 1-2 trimestres) [prior].
- **Dónde falla:** empresas muy adquisitivas (el crecimiento inorgánico no es la
  señal que se busca; idealmente usar crecimiento orgánico si el proveedor lo da,
  o filtrar por la bandera de §7.2). Energía y Materiales: el crecimiento de ventas
  es precio de la materia prima, común a todo el sector, no calidad de empresa —
  neutralización sectorial imprescindible—. Financieras: "ventas" no es homogéneo.
  Empresas con estacionalidad extrema (retail navideño): usar siempre TTM o
  comparación interanual, jamás secuencial trimestre a trimestre.

---

## 11. Earnings yield y FCF yield

### 11.1 Definiciones

```
EY_{i,t}     = E_TTM_{i,q(t)} / MC_{i,t}                                     # earnings yield
FCF_TTM      = CFO_TTM - CAPEX_TTM
FCFY_{i,t}   = FCF_TTM_{i,q(t)} / MC_{i,t}

EV_{i,t}     = MC_{i,t} + Deuda_total + Minoritarios + Preferentes - Caja
EBIT_EV      = EBIT_TTM / EV
FCF_EV       = FCF_TTM / EV
```

`q(t)` = último trimestre con `available_at <= t`. `MC_{i,t}` se actualiza a diario;
el numerador solo cambia en las fechas de filing. Esa asimetría es correcta y
deseada: el yield sube cuando el precio cae, y esa es buena parte de la señal.

### 11.2 Yield, nunca múltiplo

**Regla absoluta: usar siempre `E/P`, nunca `P/E`.** Con beneficios cercanos a cero
el `P/E` es discontinuo y salta de `+∞` a `−∞`; el `E/P` es continuo y ordenable en
todo el dominio. **Verificado:** con beneficio de −900 y capitalización 50 000, el
`E/P` es −0.018, perfectamente ordenable en la cola inferior; el `P/E` sería −55.6,
que un ranking naïve colocaría entre las acciones "baratas". Es un error clásico y
caro.

Para las empresas con pérdidas hay dos políticas defendibles, y hay que elegir una
y documentarla:

- (a) dejar `E/P` negativo y ordenar de forma natural (todas las pérdidas al fondo);
- (b) `E/P = NaN` más un factor dummy `is_loss_making` que capture el efecto
  aparte.

La (a) es más simple; la (b) evita que la magnitud de la pérdida —muy ruidosa—
domine la cola.

### 11.3 Por qué EV y no capitalización

`E/P` y `FCF/P` son, inevitablemente, apuestas de estructura de capital: a igualdad
de negocio, una empresa más apalancada muestra mayor `E/P`. `EBIT/EV` y `FCF/EV`
neutralizan la financiación y son más comparables entre sectores. Gray y Vogel
(2012), *Analyzing Valuation Measures: A Performance Horse-Race over the Past 40
Years*, JPM 39(1): 112-121, encuentran que `EBIT/TEV` es la más robusta de las
métricas de valoración habituales **[verificar]**.

**Magnitudes de referencia con metodología conservadora.** Hou, Xue y Zhang (2020),
con puntos de corte NYSE y ponderación por capitalización: el decil extremo de
`cash flow-to-price` rinde en media **0.49 % mensual**, y el de *operating* cash
flow-to-price **0.77 % mensual** —notablemente por debajo de las cifras de los
estudios originales, que usaban ponderación equiponderada y muestras dominadas por
microcaps— **[verificar]**.

### 11.4 Sectores, horizonte y trampas

- **Horizonte:** value es el factor **más lento** de todos los aquí tratados.
  Vida media larga, del orden de años; su decaimiento no se mide en meses. El
  reequilibrio semanal sobre value es casi todo coste de transacción. Para un
  esquema `W-FRI` en el motor de backtest, el value debe entrar con peso estable y
  el turnover debe venir de los factores rápidos.
- **Dónde falla:**
  - **Financieras:** `FCF` carece de sentido —`CFO` incorpora variaciones de la
    cartera de préstamos y de las posiciones de negociación, y `CAPEX` es
    irrelevante—. `EV` tampoco es interpretable: la deuda de un banco es su materia
    prima, no financiación. Para GICS 40, usar `P/B`, `P/TBV` y `ROTE`, con el
    factor de FCF en NaN.
  - **Inmobiliarias:** el resultado neto está deprimido por la amortización
    contable de activos que se aprecian. Usar `FFO`/`AFFO` (`NI + amortización −
    plusvalías por ventas`) en lugar de `NI`, o excluir.
  - **Biotech y tecnología en fase de inversión:** beneficios y FCF negativos por
    diseño; el factor los manda al lado corto sistemáticamente. Ese sesgo puede ser
    rentable o catastrófico según el régimen (fue catastrófico en 2020-2021 y muy
    rentable en 2022). Neutralización sectorial obligatoria.
  - **Energía:** el yield es una función mecánica del precio del crudo con retardo;
    sin neutralizar sectorialmente, un factor de FCF yield es una apuesta larga en
    energía en la parte alta del ciclo, que es justo el peor momento.
  - **Capex cíclico:** `FCF = CFO − CAPEX` castiga a la empresa que está invirtiendo
    para crecer y premia a la que está cosechando. Es coherente con Cooper-Gulen-
    Schill (§10.2), pero conviene ser consciente de que `FCFY` y
    `−asset_growth` están correlacionados y no deben sumarse como si fueran
    independientes.

---

## 12. Apalancamiento, distress y guidance

### 12.1 Apalancamiento y riesgo de quiebra

```
NetDebt_EBITDA = ( Deuda_total - Caja ) / EBITDA_TTM
Debt_Assets    = Deuda_total / TA
IntCov         = EBIT_TTM / Gastos_financieros_TTM
```

**Referencia:** Campbell, Hilscher y Szilagyi (2008), *In Search of Distress Risk*,
JF 63(6): 2899-2939, con datos de EE. UU. 1963-2003. Encuentran que la probabilidad
de quiebra, exclusión de cotización o rating D aumenta con **mayor apalancamiento,
menor rentabilidad, menor capitalización, peores rentabilidades pasadas, mayor
volatilidad pasada, menos caja, mayor market-to-book y menor precio por acción**.
Y el resultado central: **desde 1981 las acciones en dificultades financieras han
ofrecido rentabilidades anormalmente BAJAS**, con volatilidades, betas de mercado y
cargas sobre los factores value y small-cap mucho más altas. Es decir: más riesgo y
menos rentabilidad. Su modelo de forma reducida mide el riesgo de quiebra con más
precisión que el Z-score de Altman (1968) o el O-score de Ohlson (1980), que son las
referencias históricas.

**Signo:** menor apalancamiento y menor probabilidad de quiebra = más alcista.

```
leverage_factor = - z( NetDebt_EBITDA )      # signo negativo
```

**Dónde falla — de forma total:** el apalancamiento es **el factor más
estructuralmente sectorial de todos**. La mediana de deuda/fondos propios del sector
financiero está en otro orden de magnitud que la del resto, porque un banco gana
dinero precisamente endeudándose para prestar. Las utilities están altamente
apalancadas por su base de activos regulada. Sin neutralización sectorial, un factor
de apalancamiento es simplemente una posición corta en Financieras y Utilities.
Recomendación: NaN para GICS 40; neutralización sectorial obligatoria en el resto;
`EBITDA <= 0` ⇒ NaN (el cociente cambia de signo sin sentido económico, mismo
problema que `CFO/NI` en §8.1).

### 12.2 Factores de guidance

La *guidance* es el dato fundamental **más difícil de obtener PIT** y a la vez uno
de los más informativos.

**Definiciones**

```
G_mid            = ( G_low + G_high ) / 2                      # punto medio del rango guiado

GS  (sorpresa)   = ( G_mid - C_prev ) / P                      # C_prev = consenso vigente
                                                               #   ANTES de la guia
dG  (revision)   = ( G_mid_new - G_mid_old ) / P

GW  (amplitud)   = ( G_high - G_low ) / | G_mid |              # incertidumbre; mayor = peor

GD  (dummies)    = { inicia_guia, retira_guia, deja_de_guiar }
```

`GS` es el análogo directo de `SUE_analyst` pero **mirando hacia adelante**, y por
esa razón es a menudo más informativo del retorno del evento que la propia sorpresa
del trimestre cerrado.

**Hechos relevantes de la literatura:**

- **Rogers y Van Buskirk (2013)**, *Bundled forecasts in empirical accounting
  research*, JAE 55(1): 43-65: la gran mayoría de las previsiones de la dirección se
  publican **agrupadas ("bundled") con el anuncio de resultados**. Consecuencia
  metodológica de primer orden para nosotros: **el retorno del día del anuncio no es
  una función limpia de SUE**. Mezcla la sorpresa del trimestre cerrado y la
  sorpresa de guidance sobre el trimestre siguiente, y con frecuencia domina la
  segunda. Esto explica buena parte de la varianza no explicada en las regresiones
  retorno-sobre-SUE y **debe** documentarse en el módulo `events`.
- **Ng, Tuna y Verdi (2013)**, *Management forecast credibility and underreaction to
  news*, RAST 18(4): 956-986: la deriva posterior a una guía es **mayor en las
  empresas con guías históricamente creíbles** (buen historial de precisión). Es
  decir, existe un análogo del PEAD para guidance, condicionado a la reputación del
  emisor. Sugiere una feature: `credibilidad_i = -MAE(G_mid_pasadas vs realizado)`,
  interactuada con `GS`.
- **Anilowski, Feng y Skinner (2007)**, JAE 44(1-2): 36-63: a nivel agregado, la
  guía —especialmente el nivel relativo de guía **a la baja** trimestral— se asocia
  con medidas agregadas de noticias de beneficios, y hay evidencia más modesta de
  asociación con rentabilidades de mercado, concentrada al final de cada trimestre
  natural (cuando se publican la mayoría de preanuncios). Relevante como señal de
  *timing* agregado, no como factor cross-section.
- Existe además literatura sobre la rigidez del hábito de guiar (véase la línea de
  trabajo de Call y coautores sobre si las empresas quedan "atrapadas" emitiendo
  guía trimestral): el **abandono** de la guía es un evento raro y muy informativo,
  lo que respalda incluir `deja_de_guiar` como dummy.

**Obtención de datos — restricción real de este proyecto.** No hay fuente gratuita
de guidance estructurada y point-in-time. La vía viable es parsear los exhibits
`EX-99.1` de los **8-K item 2.02** en EDGAR y extraer el lenguaje de guía
("we expect", "we now anticipate", "full year outlook", rangos numéricos). Es un
problema de NLP no trivial, con alto riesgo de falsos positivos, y **debe anotarse
en `docs/OPEN_QUESTIONS.md`** antes de que ningún backtest dependa de él. Mientras
tanto, un proxy imperfecto pero PIT-limpio y barato es la **revisión del consenso
FY1 en la ventana `[+1, +5]` sesiones tras el anuncio**: recoge la reacción de los
analistas a la guía sin necesidad de leerla.

**Dónde falla:** sectores donde la guía es rara o poco significativa (Financieras
no suelen guiar EPS; Energía guía producción y capex, no beneficios). Empresas que
guían solo anualmente frente a las que guían trimestralmente: la señal no es
comparable, hay que normalizar por horizonte de la guía.

---

## 13. Decaimiento post-publicación: la evidencia y qué implica

Ninguna de las cifras de las secciones anteriores debe usarse como expectativa
para 2026. La evidencia sobre la degradación de los factores publicados es amplia y
consistente:

| Estudio | Hallazgo cuantitativo |
|---|---|
| **McLean y Pontiff (2016)**, JF 71(1): 5-32 | Sobre 97 variables predictoras publicadas: las rentabilidades de cartera son **26 % menores fuera de muestra** y **58 % menores tras la publicación**. La diferencia, **32 puntos porcentuales**, se atribuye a la negociación informada por la publicación académica. |
| **Chordia, Subrahmanyam y Tong (2014)**, JAE 58(1): 41-58 | La mayoría de las anomalías se han atenuado; la rentabilidad media de una estrategia sobre anomalías prominentes **se ha reducido aproximadamente a la mitad tras la decimalización**. Explicado por AUM de hedge funds, interés corto y rotación agregada. |
| **Green, Hand y Zhang (2017)**, RFS 30(12): 4389-4436 | De 94 características, solo **12 son determinantes independientes** en valores no-microcap en 1980-2014, y la predictibilidad **cayó bruscamente en 2003**: desde entonces **solo 2** características siguen siendo determinantes independientes. |
| **Hou, Xue y Zhang (2020)**, RFS 33(5): 2019-2133 | De 452 anomalías replicadas con puntos de corte NYSE y ponderación por capitalización, **el 65 % no supera el umbral \|t\| ≥ 1.96**. La sorpresa de resultados a 6 y 12 meses: 0.19 % y 0.11 % mensual (t = 1.65 y 1.00). |
| **Martineau (2021)**, Critical Finance Review | PEAD **inexistente en valores grandes desde ~2006**. |
| *Why Has PEAD Declined Over Time?* (WP, CEASA/Columbia) | La caída de la **persistencia del SUE** es motor principal de la atenuación del PEAD en cuatro décadas **[verificar]**. |
| *The disappearing abnormal returns to a fundamental signal strategy*, Managerial Finance (2017) | Las rentabilidades anormales de la estrategia de señales fundamentales de Abarbanell-Bushee han desaparecido tras su publicación **[verificar]**. |
| **Ng, Rusticus y Verdi (2008)**, JAR 46(3) | Los beneficios de implementar PEAD se reducen significativamente tras costes de transacción; el drift residual se concentra en los valores con costes altos. |

**Nótese la coincidencia de fechas:** decimalización (2001), caída de
predictibilidad de Green-Hand-Zhang (2003), muerte del PEAD en large caps según
Martineau (~2006). Los tres apuntan al mismo régimen: la primera mitad de los 2000
es una discontinuidad estructural en la sección cruzada de rentabilidades de EE. UU.

### 13.1 Reglas metodológicas que se derivan, y son vinculantes

1. **Partir siempre la muestra.** Todo backtest debe reportar al menos
   `[1996-2000]`, `[2001-2009]`, `[2010-2025]` por separado. Un Sharpe agregado
   1996-2025 sobre cualquier factor de esta lista mezcla dos regímenes distintos y
   es engañoso por construcción. Los datos semilla llegan hasta 2025-08-23, lo que
   permite este corte sin más.
2. **Sharpe deflactado obligatorio.** Bailey y López de Prado, con el número real de
   configuraciones probadas, no con "la que reportamos". Y Benjamini-Hochberg sobre
   el conjunto completo de factores candidatos, no factor a factor.
3. **Puntos de corte NYSE y ponderación por capitalización** en las carteras de
   diagnóstico, para ser comparables con Hou-Xue-Zhang. En S&P 500 esto importa
   menos que en CRSP completo (no hay microcaps), pero la ponderación equiponderada
   sigue sobreponderando la cola pequeña del índice, que es donde vive el poco
   efecto que queda.
4. **Prior bayesiano escéptico.** Ante un factor de esta lista que en nuestro
   backtest S&P 500 post-2010 muestre un t de 2.5, la interpretación por defecto
   debe ser sobreajuste, no descubrimiento. La carga de la prueba está en la CV
   purgada con embargo y el bootstrap estacionario.
5. **Costes realistas.** El `CostModel` del contrato (spread por tramo de liquidez +
   impacto sqrt(participación) + comisión + coste de préstamo) no es un adorno: el
   resultado de Ng-Rusticus-Verdi es que la diferencia entre alfa bruta y neta es
   la diferencia entre anomalía y no anomalía.

---

## 14. Tabla resumen operativa

Los ICs son **rank IC mensuales**. Los marcados **[prior]** son priors de
ingeniería a validar internamente, **no** cifras publicadas: la literatura
académica reporta rentabilidades de cartera, no ICs, y traducir de una a otra
requiere supuestos sobre dispersión que no son universales. La referencia
practicante habitual sitúa los ICs de factores de renta variable documentados en el
rango **0.02-0.05**, siendo 0.05-0.10 excepcional y >0.10 prácticamente inexistente
de forma sostenida.

| Factor | Frecuencia dato | `available_at` | Horizonte / decaimiento | IC esperado (S&P 500, post-2010) | Falla en | Neutralizar por |
|---|---|---|---|---|---|---|
| `SUE_analyst` | trimestral | `tradable_date` evento | 3-10 sesiones | 0.01-0.03 [prior] | large caps líquidas; base GAAP≠street | sector, tamaño, `E/P` |
| `SUE_sigma` | trimestral | `tradable_date` evento | 3-10 sesiones | 0.01-0.02 [prior] | σ→0 (utilities, staples); M&A | sector, tamaño |
| `SUE_price` | trimestral | `tradable_date` evento | 3-10 sesiones | 0.01-0.02 [prior] | correlación con value | sector, tamaño, **`E/P`** |
| `SURGE` (ingresos) | trimestral | `tradable_date` evento | 5-20 sesiones | 0.01-0.03 [prior] | financieras; M&A | sector, tamaño |
| `DOUBLE` (SUE∧SURGE) | trimestral | `tradable_date` evento | 5-20 sesiones | 0.02-0.04 [prior] | ídem | sector, tamaño |
| `PEAD` (SUE con vida H) | evento | `tradable_date` evento | **H = 3-10 sesiones** | 0.00-0.02 [prior] | **large caps: muerto desde ~2006** | sector, tamaño |
| Revisiones (`REV6`) | mensual/diaria | `as_of` snapshot | 1-3 meses | 0.02-0.04 [prior] | alta cobertura; Energía (macro) | sector, **momentum precio** |
| Difusión revisiones | mensual | `as_of` snapshot | 1-3 meses | 0.02-0.03 [prior] | baja cobertura | sector, momentum |
| Piotroski F (binario) | anual | `accepted_at` 10-K | ~12 meses | 0.01-0.02 [prior] | **grandes/líquidas**; financ.; REITs | sector; **empates masivos** |
| Piotroski F (continuo) | trimestral TTM | `accepted_at` 10-Q | ~12 meses | 0.02-0.03 [prior] | financieras; REITs | sector, tamaño |
| `−PACC` (percent accruals) | trimestral | `accepted_at` 10-Q | ~12 meses | 0.02-0.04 [prior] | financieras; REITs; M&A | sector, tamaño |
| `−ACC_CF` | trimestral | `accepted_at` 10-Q | ~12 meses | 0.02-0.03 [prior] | ídem | sector, tamaño |
| `CFO/NI` | trimestral | `accepted_at` 10-Q | — | **redundante con `PACC`** | `NI<0` invierte el signo | no usar como factor |
| `GPA` (Novy-Marx) | trimestral | `accepted_at` 10-Q | 6-12 meses | 0.02-0.04 [prior] | financieras (sin COGS) | sector |
| Tendencia margen (t-OLS) | trimestral | `accepted_at` 10-Q | 6-12 meses | 0.01-0.03 [prior] | financieras; semis (ciclo) | sub-industria si N≥15, si no sector |
| `−g_TTM` (ventas, nivel) | trimestral | `accepted_at` 10-Q | ~12 meses | 0.01-0.03 [prior] | M&A; Energía/Materiales (precio) | sector, `B/M` |
| `−asset_growth` | anual | `accepted_at` 10-K | ~12 meses | 0.02-0.04 [prior] | financieras | sector |
| Aceleración ventas | trimestral | `accepted_at` 10-Q | 1-2 trimestres | **sin prior fiable** | evidencia académica débil | sector, nivel de crecimiento |
| `E/P` | trimestral / diario | `accepted_at` + precio | **años** | 0.02-0.04 [prior] | pérdidas; financieras (usar P/B) | sector, tamaño |
| `FCF/P`, `FCF/EV` | trimestral / diario | `accepted_at` 10-Q | **años** | 0.02-0.04 [prior] | financieras; REITs (usar FFO) | sector, tamaño |
| `EBIT/EV` | trimestral / diario | `accepted_at` 10-Q | **años** | 0.02-0.05 [prior] | financieras | sector, tamaño |
| `−NetDebt/EBITDA` | trimestral | `accepted_at` 10-Q | 6-12 meses | 0.01-0.03 [prior] | **financieras, utilities: estructural** | sector **obligatorio** |
| Guidance `GS` | evento | `announced_at` (bundled) | 5-20 sesiones | 0.02-0.05 [prior] | datos no disponibles PIT gratis | sector |

**Nota sobre la suma de ICs.** Estos factores **no** son independientes. Solapamientos
identificados en este documento y que hay que medir antes de combinar:
`SUE_price`↔`E/P`; `CFO/NI`≡`1−PACC`; `F_ACCRUAL`⊂`F-Score` y `≈ −ACC`;
`FCFY`↔`−asset_growth`; revisiones↔momentum de precio; `SURGE`↔`g_TTM`.
Usar `signals.combine(..., orthogonalize)` o ponderación por IC con matriz de
covarianza estimada, nunca suma de z-scores a pesos iguales.

---

## 15. Recomendaciones concretas para la implementación

### 15.1 Orden de prioridad sugerido

1. **`SUE_price` y `SUE_sigma`** — solo requieren `EarningsEvent` con EPS, que es lo
   que la plataforma ya modela. Sin dependencia de consenso. Es la base sobre la que
   se puede validar todo el resto del pipeline PIT.
2. **`SURGE` y `DOUBLE`** — mismo requisito de datos, alto valor incremental.
3. **`−PACC` y `GPA`** — requieren fundamentales del 10-Q; alto valor, decaimiento
   lento, tolerantes a reequilibrio trimestral.
4. **Revisiones de analistas** — requieren `EstimateSnapshot` histórico, que es la
   dependencia de datos más cara. Alto valor si se consigue.
5. **F-Score continuo, tendencia de margen, yields** — completan la batería.
6. **Guidance** — bloqueado por disponibilidad de datos; documentar en
   `OPEN_QUESTIONS.md`.

### 15.2 Invariantes verificables sin red (para los tests del módulo `factors`)

Estos son contratos comprobables contra `data.synthetic`, no comprobaciones de
rendimiento:

- **Invariancia de escala:** `SUE_sigma(k·EPS) == SUE_sigma(EPS)` para todo `k>0`.
  Verificado en este documento.
- **Historia mínima:** con 12 trimestres, `SUE_*` y `SURGE` deben lanzar
  `InsufficientHistory`; con 13, deben devolver un valor finito.
- **Sensibilidad al día negociable:** desplazar `announced_at` de AMC a BMO debe
  cambiar la fecha de entrada de la señal en exactamente una sesión. Es el test que
  demuestra que `tradable_date` muerde de verdad; si el factor no cambia, el
  as-of join está roto.
- **No look-ahead:** para cualquier `t`, ningún valor del factor puede depender de
  un `FundamentalFact` con `available_at > t`. Test mecánico: recalcular el panel
  truncando los datos en `t` y exigir igualdad con la columna `t` del panel
  completo. Si difiere, `LookAheadError`.
- **Rango:** `F_SCORE ∈ {0..9}` y exactamente 9 componentes. Verificado.
- **Signo:** para el generador sintético, que crea sorpresas correlacionadas con el
  drift posterior, el IC de `SUE` debe ser **positivo**; el de `PACC` sin invertir
  debe ser **negativo** y el de `−PACC` positivo. Es el test que detecta signos
  invertidos, que de otro modo pasan desapercibidos.
- **Neutralización:** tras `neutralize(by=["sector","size"])`, la media por sector
  debe ser ~0 (< 1e-9) y la correlación con `log(MC)` ~0. Verificado a precisión de
  máquina con `lstsq`.
- **NaN estructural:** `GPA`, margen, accruals y FCF deben ser `NaN` —no 0— para
  todos los tickers de GICS 40 y 60. Test: contar NaN esperados ≈ 107 nombres.

### 15.3 Parámetros por defecto propuestos

```python
WINSOR_Q          = 0.01      # por fecha, cross-section
MIN_QUARTERS_SUE  = 13        # verificado: exactamente lo que exige j=1..8
MIN_TTM_TREND     = 8         # observaciones TTM para el t de la pendiente
SIGMA_FLOOR_FRAC  = 0.005     # suelo de sigma como fraccion del precio
CONSENSUS_WINDOW  = 90        # dias previos al anuncio para la mediana de analistas
PEAD_HORIZON_H    = 5         # sesiones; calibrar, prior corto (Martineau 2021)
EXCLUDE_SECTORS   = {"Financials", "Real Estate"}   # para accruals, margen, FCF
```

---

## 16. Referencias

**Sorpresa de resultados y PEAD**

- Ball, R. y Brown, P. (1968). *An Empirical Evaluation of Accounting Income
  Numbers*. Journal of Accounting Research, 6(2), 159-178.
- Foster, G. (1977). *Quarterly Accounting Data: Time-Series Properties and
  Predictive-Ability Results*. The Accounting Review, 52(1), 1-21.
- Foster, G., Olsen, C. y Shevlin, T. (1984). *Earnings Releases, Anomalies, and the
  Behavior of Security Returns*. The Accounting Review, 59(4), 574-603.
- Bernard, V. L. y Thomas, J. K. (1989). *Post-Earnings-Announcement Drift: Delayed
  Price Response or Risk Premium?*. Journal of Accounting Research, 27 (Supl.), 1-36.
- Bernard, V. L. y Thomas, J. K. (1990). *Evidence that Stock Prices Do Not Fully
  Reflect the Implications of Current Earnings for Future Earnings*. Journal of
  Accounting and Economics, 13(4), 305-340.
- Livnat, J. y Mendenhall, R. R. (2006). *Comparing the Post-Earnings Announcement
  Drift for Surprises Calculated from Analyst and Time Series Forecasts*. Journal of
  Accounting Research, 44(1), 177-205.
- Jegadeesh, N. y Livnat, J. (2006). *Revenue Surprises and Stock Returns*. Journal
  of Accounting and Economics, 41(1-2), 147-171.
- Jegadeesh, N. y Livnat, J. (2006). *Post-Earnings-Announcement Drift: The Role of
  Revenue Surprises*. Financial Analysts Journal, 62(2), 22-34. (Véase también
  *Double Surprise into Higher Future Returns*, FAJ 63(4), 2007.)
- Ng, J., Rusticus, T. O. y Verdi, R. S. (2008). *Implications of Transaction Costs
  for the Post-Earnings Announcement Drift*. Journal of Accounting Research, 46(3),
  661-696.
- Martineau, C. (2021). *Rest in Peace Post-Earnings Announcement Drift*. Critical
  Finance Review. https://cfr.ivo-welch.info/published/papers/martineau2021rest.pdf
- Grégoire, V. y Martineau, C. (2021). *How is Earnings News Transmitted to Stock
  Prices?*. Journal of Accounting Research, 60(1), 261-297.
- *Why Has PEAD Declined Over Time? The Role of Earnings News Persistence* (working
  paper, CEASA / Columbia Business School).
  https://business.columbia.edu/sites/default/files-efs/imce-uploads/CEASA/Events%20Page/PEAD_Declined_over_time.pdf

**Revisiones de analistas**

- Chan, L. K. C., Jegadeesh, N. y Lakonishok, J. (1996). *Momentum Strategies*.
  The Journal of Finance, 51(5), 1681-1713.
- Gleason, C. A. y Lee, C. M. C. (2003). *Analyst Forecast Revisions and Market Price
  Discovery*. The Accounting Review, 78(1), 193-225.
- Diether, K. B., Malloy, C. J. y Scherbina, A. (2002). *Differences of Opinion and
  the Cross Section of Stock Returns*. The Journal of Finance, 57(5), 2113-2141.
- Payne, J. L. y Thomas, W. B. (2003). *The Implications of Using Stock-Split
  Adjusted I/B/E/S Data in Empirical Research*. The Accounting Review, 78(4),
  1049-1067.

**Análisis fundamental, calidad y accruals**

- Lev, B. y Thiagarajan, S. R. (1993). *Fundamental Information Analysis*. Journal
  of Accounting Research, 31(2), 190-215.
- Sloan, R. G. (1996). *Do Stock Prices Fully Reflect Information in Accruals and
  Cash Flows About Future Earnings?*. The Accounting Review, 71(3), 289-315.
- Abarbanell, J. S. y Bushee, B. J. (1998). *Abnormal Returns to a Fundamental
  Analysis Strategy*. The Accounting Review, 73(1), 19-45.
- Piotroski, J. D. (2000). *Value Investing: The Use of Historical Financial
  Statement Information to Separate Winners from Losers*. Journal of Accounting
  Research, 38 (Supl.), 1-41.
- Collins, D. W. y Hribar, P. (2002). *Errors in Estimating Accruals: Implications
  for Empirical Research*. Journal of Accounting Research, 40(1), 105-134.
- Richardson, S. A., Sloan, R. G., Soliman, M. T. y Tuna, I. (2005). *Accrual
  Reliability, Earnings Persistence and Stock Prices*. Journal of Accounting and
  Economics, 39(3), 437-485.
- Hafzalla, N., Lundholm, R. y Van Winkle, E. M. (2011). *Percent Accruals*. The
  Accounting Review, 86(1), 209-236.
- Novy-Marx, R. (2013). *The Other Side of Value: The Gross Profitability Premium*.
  Journal of Financial Economics, 108(1), 1-28.
- Asness, C. S., Frazzini, A. y Pedersen, L. H. (2019). *Quality Minus Junk*. Review
  of Accounting Studies, 24(1), 34-112.
- *The disappearing abnormal returns to a fundamental signal strategy*. Managerial
  Finance (2017). https://www.emerald.com/insight/content/doi/10.1108/mf-05-2016-0142

**Valoración, crecimiento y distress**

- Lakonishok, J., Shleifer, A. y Vishny, R. W. (1994). *Contrarian Investment,
  Extrapolation, and Risk*. The Journal of Finance, 49(5), 1541-1578.
- Ohlson, J. A. (1980). *Financial Ratios and the Probabilistic Prediction of
  Bankruptcy*. Journal of Accounting Research, 18(1), 109-131.
- Altman, E. I. (1968). *Financial Ratios, Discriminant Analysis and the Prediction
  of Corporate Bankruptcy*. The Journal of Finance, 23(4), 589-609.
- Campbell, J. Y., Hilscher, J. y Szilagyi, J. (2008). *In Search of Distress Risk*.
  The Journal of Finance, 63(6), 2899-2939.
- Cooper, M. J., Gulen, H. y Schill, M. J. (2008). *Asset Growth and the
  Cross-Section of Stock Returns*. The Journal of Finance, 63(4), 1609-1651.
- Gray, W. R. y Vogel, J. (2012). *Analyzing Valuation Measures: A Performance
  Horse-Race over the Past 40 Years*. The Journal of Portfolio Management, 39(1),
  112-121.

**Guidance**

- Anilowski, C., Feng, M. y Skinner, D. J. (2007). *Does Earnings Guidance Affect
  Market Returns? The Nature and Information Content of Aggregate Earnings
  Guidance*. Journal of Accounting and Economics, 44(1-2), 36-63.
- Rogers, J. L. y Van Buskirk, A. (2013). *Bundled Forecasts in Empirical Accounting
  Research*. Journal of Accounting and Economics, 55(1), 43-65.
- Ng, J., Tuna, I. y Verdi, R. (2013). *Management Forecast Credibility and
  Underreaction to News*. Review of Accounting Studies, 18(4), 956-986.

**Replicación y decaimiento**

- McLean, R. D. y Pontiff, J. (2016). *Does Academic Research Destroy Stock Return
  Predictability?*. The Journal of Finance, 71(1), 5-32.
- Chordia, T., Subrahmanyam, A. y Tong, Q. (2014). *Have Capital Market Anomalies
  Attenuated in the Recent Era of High Liquidity and Trading Activity?*. Journal of
  Accounting and Economics, 58(1), 41-58.
- Green, J., Hand, J. R. M. y Zhang, X. F. (2017). *The Characteristics that Provide
  Independent Information about Average U.S. Monthly Stock Returns*. The Review of
  Financial Studies, 30(12), 4389-4436.
- Hou, K., Xue, C. y Zhang, L. (2020). *Replicating Anomalies*. The Review of
  Financial Studies, 33(5), 2019-2133.
- Daniel, K., Grinblatt, M., Titman, S. y Wermers, R. (1997). *Measuring Mutual Fund
  Performance with Characteristic-Based Benchmarks*. The Journal of Finance, 52(3),
  1035-1058.
- Bailey, D. H. y López de Prado, M. (2014). *The Deflated Sharpe Ratio: Correcting
  for Selection Bias, Backtest Overfitting and Non-Normality*. The Journal of
  Portfolio Management, 40(5), 94-107.

---

## 17. Preguntas abiertas que deben pasar a `docs/OPEN_QUESTIONS.md`

1. **Sector GICS point-in-time.** El fichero semilla trae el sector actual. La
   escisión de Real Estate (2016) y la reestructuración de Communication Services
   (2018) introducen look-ahead en la neutralización de periodos anteriores. ¿Se
   reconstruye un histórico de sectores o se acepta y documenta el sesgo?
2. **`available_at` de fundamentales.** Se necesita la *acceptance datetime* de
   EDGAR, no la fecha de filing. Confirmar que el módulo `data.edgar` la expone.
3. **Desfase 8-K → 10-Q.** Cuantificar empíricamente la mediana del desfase en el
   S&P 500 para calibrar la latencia real de accruals, F-Score y FCF yield.
4. **Guidance PIT.** No hay fuente gratuita estructurada. ¿Se invierte en un parser
   de `EX-99.1` o se usa el proxy de revisión de consenso en `[+1, +5]`?
5. **Base contable de EPS.** Confirmar que el proveedor de estimaciones entrega
   *actual* y *consenso* en la misma base (street), y que los ficheros no están
   ajustados retroactivamente por splits (Payne y Thomas 2003).
6. **Variante bancaria de los factores de calidad.** ¿Se implementa un F-Score
   específico para GICS 40 (ROTE, ratio de eficiencia, capital, morosidad) o se
   dejan los 107 nombres de Financials + Real Estate en NaN?
7. **Verificación de cifras marcadas [verificar].** El proxy de este entorno bloquea
   la descarga de PDFs académicos (403 en `WebFetch`, incluso `example.com`). Las
   magnitudes de Bernard-Thomas, Piotroski, Sloan, Foster-Olsen-Shevlin,
   Hou-Xue-Zhang y Gray-Vogel proceden de resúmenes de búsqueda y deben contrastarse
   contra el texto original en la máquina del usuario antes de usarse para calibrar
   expectativas.
