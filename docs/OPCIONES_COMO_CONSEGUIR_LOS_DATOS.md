# Cómo conseguir el histórico de cadenas de opciones — brokers, vendors de pago y dataset mínimo

**Fecha de investigación:** 2026-08-05 (precios verificados por búsqueda web en esta fecha; todo
lo no confirmado va marcado **[SIN VERIFICAR]**).
**Complementa a** `docs/research/datasets_opciones.md` (veredicto de fuentes abiertas: no hay
backfill gratis con licencia limpia; DoltHub `post-no-preference/options` 2019–2024 como mejor
candidato libre; archivo gratuito ene–jun 2013 de optiondata.org; VIX ya descargado) y a
`docs/research/data_sources.md` §7 (fichas de proveedores). Aquí se cierran las dos vías que
faltaban — **broker (IBKR)** y **pago** — y se calcula el **dataset mínimo con potencia
estadística** para que cada euro compre una respuesta, no un fichero.

**Moneda:** todos los precios en USD, que es como los publican los vendors.

---

## 0. Veredicto en una tabla (léelo primero)

| Vía | ¿Backfill histórico? | ¿Recolección hacia adelante? | Coste de entrada | Veredicto |
|---|---|---|---|---|
| **IBKR API** | **NO** — la API no expone contratos expirados (§1.3) | Sí, con esfuerzo y límites de líneas (§1.4) | ~10–12 USD/mes de datos (a menudo exonerado) | Solo como recolector redundante; **jamás** como fuente de histórico |
| **ThetaData** | Sí, EOD y tick desde 2012-06 | Sí | 40 USD/mes (Starter, 4 años) | **Mejor relación precio/profundidad en cruda** (§2.1) |
| **Databento OPRA** | Sí, desde 2013-04 (trades, NBBO-1m, EOD, OI, definiciones) | Sí | 199 USD/mes (Standard) o pago por uso | **Mejor microestructura**; IV/griegas las calculas tú — que es lo que este repo prefiere (§2.2) |
| **ORATS** | Sí, near-EOD desde 2007 | Sí | 99 USD/mes API · **599 USD una sola vez** todo 2007→hoy | **Mejor compra única**: 599 USD cierran el ángulo B entero (§2.4) |
| **Polygon/Massive** | Sí, desde 2014 (plan alto) | Sí | 29 USD/mes (2 años) | Correcto pero el rebranding deja precios en tránsito (§2.3) |
| **Alpha Vantage** | Sí, cadenas EOD 15+ años | Sí | 0 (25 req/día) · 49,99 USD/mes (75 req/min) | **El backfill de pago más barato** — usable SOLO para precios/volumen/OI, nunca su IV (§2.5) |
| **Cboe DataShop** | Sí, EOD desde 2018 ad-hoc; Open-Close es otra liga | n/a (ficheros) | 400–500 USD · académico 750 USD/año Open-Close | Solo con afiliación académica o para el Open-Close (§2.6) |
| **historicaloptiondata.com** | Sí, desde 2002 | Suscripción | 615 USD (5 años, L1) | Compra única razonable si se quiere 2002–2013; si no, ORATS gana (§2.7) |
| **Intrinio / EODHD** | Parcial / ~1 año | Sí | opaco / ~30 USD | **No** para este proyecto (§2.8, §5) |

**La frase que resume el documento:** el hueco de opciones se cierra con **599 USD una sola vez
(ORATS 2007→hoy)** o con **~40–199 USD/mes** (ThetaData/Databento) si se prefiere cruda; IBKR
**no puede** cerrar el hueco histórico por diseño de su API; y antes de gastar nada, la Fase 0
(0 USD, §4.1) responde ya la primera pregunta con ~9.000–10.000 eventos.

---

## 1. IBKR sin rodeos

El usuario preguntó expresamente por IBKR. La respuesta corta: **IBKR es un bróker con una API de
trading que regala algo de historia, no un vendor de datos**. La respuesta larga:

### 1.1 Lo que la API ofrece para opciones

- **`reqSecDefOptParams`** (TWS API) devuelve, por subyacente, los parámetros de la cadena:
  vencimientos listados y mallas de strikes por *trading class* y exchange. No consume
  suscripción de datos de mercado y no está sujeto al pacing de histórico. Es el punto de
  entrada correcto para enumerar contratos **vigentes** — y solo vigentes.
- **`reqContractDetails`** resuelve cada contrato concreto (conId). Ojo: una cadena entera de un
  ticker del S&P 500 son 1.500–3.000 contratos; enumerarlos todos ya es una operación pesada.
- **`reqHistoricalData`** sobre un contrato de opción concreto devuelve barras (`TRADES`,
  `MIDPOINT`, `BID`, `ASK`, `BID_ASK`) mientras el contrato esté **listado**. Requiere la
  suscripción OPRA activa (los datos históricos exigen la misma suscripción que los de tiempo
  real).
- **`whatToShow=OPTION_IMPLIED_VOLATILITY`** sobre el **subyacente** devuelve la serie histórica
  del índice de IV a 30 días que calcula el propio IBKR, con años de profundidad. Es el único
  "histórico de opciones" real que IBKR regala hacia atrás: sirve como control de nivel de IV,
  pero no contiene skew, ni `vol_spread`, ni OI, ni cadena. Verifícalo en cuenta propia antes de
  contar con él para nada más.
- **`reqMktData` con genericTicks 100/101/104/106** da, en vivo, volumen de opciones, open
  interest, vol histórica e IV por contrato — la materia prima de una foto EOD.

### 1.2 Límites de pacing reales

Documentados por IBKR (TWS API, *Historical Data Limitations*) y confirmados por años de hilos de
usuarios:

- **Máximo 60 peticiones de histórico por cada 10 minutos** (≈8.640/día si se martillea 24/7).
- **Peticiones idénticas separadas menos de 15 s** → violación.
- **6 o más peticiones del mismo contrato+exchange+tick type en 2 s** → violación.
- **`BID_ASK` cuenta doble** a efectos de pacing.
- Desde v9.72 el límite "duro" para barras ≥1 min está formalmente levantado, pero IBKR aplica
  un *soft throttling* que degrada y llega a desconectar el cliente si se abusa. En la práctica,
  60/10 min sigue siendo el presupuesto de planificación honesto.
- Para `reqMktData` el límite es de **líneas simultáneas** (~100 por defecto; ampliable con
  *booster packs* o por volumen de comisiones) y de mensajes/segundo, no el pacing de histórico.
- Los **regulatory snapshots** (`reqMktData(..., regulatorySnapshot=True)`) cuestan
  **0,01 USD cada uno**, con 1 USD/mes de franquicia; si el gasto mensual alcanza el precio de la
  suscripción de esa red, IBKR te suscribe automáticamente el resto del mes.

### 1.3 La pregunta decisiva: contratos expirados

**La API de IBKR no expone datos históricos de opciones expiradas. Ninguno.** El parámetro
`includeExpired=True` que permite recuperar futuros vencidos (hasta ~2 años) **no funciona para
opciones**; la documentación oficial lista "expired options" entre los datos **no disponibles**
por API, y el propio IBKR Quant Blog lo confirma: histórico de opciones "just for the current
active iterations". En cuanto un contrato vence, sus datos desaparecen de la API para siempre.

Consecuencia inmediata para este proyecto: las señales del ángulo B viven en contratos de
**DTE ≤ 90 días** alrededor del anuncio (`options_signals.md` §3.1). *Todos* los contratos que
cubrían las ventanas de earnings de los últimos 3 años están vencidos. La fracción recuperable
del dataset objetivo vía IBKR es, redondeando, **el 0 %**.

### 1.4 Con números: 100 tickers × 3 años de ventanas de earnings

- Eventos: 100 tickers × 3 años × 4 = **1.200 eventos**; ventana T−20..T+5 = 26 sesiones →
  **31.200 ticker-días**.
- Contratos relevantes por evento (vencimientos ≤90 DTE, moneyness 0,7–1,3): ~300 →
  **~360.000 contratos**, todos expirados → **irrecuperables. Fin del cálculo real.**
- Ejercicio contrafactual (si los expirados existieran): 360.000 contratos × (1 petición
  `TRADES` + `BID_ASK` que cuenta doble) = **~1,08 M de peticiones-equivalentes** ÷ 8.640/día =
  **~125 días de API martilleada 24/7**, siendo optimistas con el *soft throttling*. Es decir:
  incluso en el mundo en el que fuera posible, sería una mala idea.
- **Recolección hacia adelante** (la foto EOD diaria del §2 de `docs/RECOLECTOR.md`): 503
  tickers × ~200 contratos filtrados = ~100.000 líneas/día vía `reqMktData` en lotes de ~100
  líneas → **2–4 horas por sesión** dentro de la ventana 16:15–17:00 ET no caben; habría que
  ampliar líneas o reducir el universo/los strikes. Viable como **recolector redundante** de un
  subconjunto (p. ej. los ~30 tickers con evento en ±10 sesiones), no como recolector primario
  de todo el índice.

### 1.5 Coste de suscripciones OPRA en IBKR

- OPRA (US Options) exige tener el **US Securities Snapshot and Futures Value Bundle**
  (no profesional; ~10 USD/mes, exonerado con comisiones suficientes **[SIN VERIFICAR la cifra
  exacta del bundle en 2026]**) y el add-on OPRA Top of Book (histórico ~1,50 USD/mes para no
  profesionales **[SIN VERIFICAR]**; IBKR documenta exoneraciones con umbrales de comisiones de
  5 y 15 USD según nivel).
- Total realista para una cuenta que ya opera algo: **0–12 USD/mes**. El coste no es el
  problema; el problema es el §1.3.

### 1.6 Veredicto IBKR

| Pregunta | Respuesta |
|---|---|
| ¿Backfill histórico? | **NO.** La API no expone opciones expiradas; el 0 % del dataset objetivo es recuperable. |
| ¿Recolección hacia adelante? | **Sí, con matices**: viable para un subconjunto priorizado (tickers con evento próximo), no para 503 tickers × cadena completa dentro de la ventana de captura. Útil como **fuente redundante** del recolector, con la ventaja de que su hora de captura la controlas tú. |
| ¿Coste? | 0–12 USD/mes de datos + los 0,01 USD/snapshot regulatorio si no se suscribe OPRA. |
| ¿Qué sí aprovechar? | `reqSecDefOptParams` para enumerar cadenas vigentes, la serie `OPTION_IMPLIED_VOLATILITY` del subyacente como control de nivel, y la cuenta como *sanity check* en vivo de horquillas cuando llegue el momento de ejecutar. |

---

## 2. Vendors con precio verificado

Regla de lectura: "verificado" = confirmado en la página del vendor o en fuentes primarias
(Federal Register para Cboe) vía búsqueda web el 2026-08-04/05. El proxy de este entorno
devuelve 403 al `WebFetch` directo de casi todos estos dominios; donde solo hay fuentes
secundarias se dice.

### 2.1 ThetaData — la cruda barata con profundidad

- **Planes de opciones** (página de precios del vendor, ago-2026): **Starter 40 USD/mes**
  (cobertura 100 % del mercado USA, peticiones ilimitadas, **4 años** de histórico, intervalos
  de 1 min); **Professional 80 USD/mes** (**8 años**, tick, cada quote NBBO de OPRA, snapshots
  de cadena completa); **tercer nivel 160 USD/mes** (histórico completo; detalles parciales en
  la fuente **[SIN VERIFICAR el desglose exacto]**); comercial desde 250 USD/mes (200 anual).
  ⚠ Un comparador de terceros (FlashAlpha, jun-2026) lista una estructura distinta
  (89/99/129/299): los precios de este vendor **cambian con frecuencia**; confirmar en
  `thetadata.net/pricing` antes de pagar.
- **Profundidad**: opciones desde **2012-06-01** (tape UTP); símbolos solo-CTA desde
  2020-01-01. Incluye quotes bid/ask con tamaños, trades, OI, griegas de 1.º–3.º orden e IV.
- **Modelo de acceso**: terminal local (Java) + API REST/Python contra tu propia máquina;
  descarga masiva permitida dentro del plan, peticiones ilimitadas.
- **Encaja aquí porque**: es cruda de verdad (puedes aplicar §2.1–2.3 de `options_signals.md`:
  forward implícito, Black-76, filtros F1–F10) y 40–80 USD/mes cubren el conjunto MEDIO (§3.4).

### 2.2 Databento OPRA — microestructura seria, pago por lo que usas

- **Planes** (blog oficial + anuncio, jun-2025): **Standard 199 USD/mes** — tiempo real + hasta
  **10 años** de histórico sobre los 18 exchanges de opciones; Plus 1.500; Unlimited 4.000.
  El pago por uso para **live** se retiró el 2025-06-03; el **histórico sigue siendo pago por
  uso en USD/GB**, con tarifas por esquema que no publican en la web — la API
  (`metadata.get_cost` / `metadata.list_unit_price`) da el **presupuesto exacto antes de pagar**.
  Cuentas nuevas: **125 USD de crédito** (página oficial). Licencia OPRA no profesional de
  paso: 1,25 USD/mes.
- **Profundidad**: cobertura completa (todos los esquemas) desde **2023-03-28**; y desde mayo de
  2025, **trades, CBBO-1m (NBBO a minuto), OHLCV, statistics (incluye open interest) y
  definiciones desde 2013-04-01**.
- **Sin IV ni griegas**: las calculas tú. Para este repo eso es una **ventaja**, no un defecto:
  `options_signals.md` §2 exige calcular la IV sobre el forward implícito de la propia cadena
  precisamente para no heredar el modelo opaco de un vendor.
- **Estimación de coste del benchmark del usuario** (solo ventanas T−20..T+5, 100 tickers ×
  3 años; 31.200 ticker-días, ~62 M de registros contrato-día): en DBN, `ohlcv-1d` ≈ 3,5 GB +
  `statistics` ≈ 2–3 GB + `definition` (1 foto/semana) ≈ 3–4 GB → **~8–11 GB**. Al no ser
  públicas las tarifas por GB, la cota honesta es: **o bien 1 mes de Standard (199 USD), o bien
  pago por uso que la propia API cotiza y que el crédito de 125 USD puede cubrir en parte
  [SIN VERIFICAR el USD/GB]**. La descarga por lotes admite filtrar por símbolo padre
  (`AAPL.OPT`) y rango de fechas, así que se paga solo por las ventanas.

### 2.3 Polygon.io / Massive — correcto, pero en plena mudanza de marca

- **Planes de opciones** (tarifario largo 2024-2025, pre-rebranding): **Starter 29 USD/mes**
  (2 años de histórico, llamadas ilimitadas, IV+griegas, OI diario, flat files); Developer
  **79–99 USD/mes** según fuente **[SIN VERIFICAR cuál rige tras el rebranding]** (4 años);
  **Advanced 199 USD/mes** (todo el histórico). Polygon se convirtió en **Massive.com**
  (oct-2025) y el tarifario está en transición: confirmar en `massive.com/pricing`.
- **Profundidad**: OPRA desde **2014** (trades, quotes, agregados diarios por flat files S3).
- **A favor**: `/v3/reference/options/contracts?as_of=` enumera los contratos **que existían**
  en una fecha — resuelve limpio la supervivencia de contratos (§5.2). Flat files = descarga
  masiva sin contar peticiones.
- **En contra**: IV con modelo propio no documentado (misma objeción de siempre: usar sus
  precios, no su IV) y precios en tránsito.

### 2.4 ORATS — la mejor compra única

- **Precios** (páginas del vendor): **Data API 99 USD/mes** (20.000 peticiones/mes, histórico
  *near-EOD* desde **2007**); 199 y 399 USD/mes para más cuota/live; intradía 1-min desde
  ago-2020. Y la pieza clave: **descarga única del histórico completo 2007→hoy por 599 USD**
  (FTP, ventana de descarga de 2 semanas).
- **Qué es**: no cadena cruda sino **superficie suavizada y parametrizada** (IV por delta y
  madurez constantes, skew, curtosis, forward, tipos y dividendos ya resueltos) + resúmenes por
  strike con quotes a ~14 minutos del cierre (su elección declarada para evitar horquillas
  anchas del cierre).
- **La aritmética de cuota** ya está en `data_sources.md` §7.2: 20.000 req/mes = ~40 días de
  trading/ticker/mes; el plan de 99 sirve para ventanas de evento, no para reconstruir el panel.
  **La compra de 599 USD hace irrelevante esa cuota** para el backfill.
- **Cautela**: su hora de captura (~15:46 ET) difiere de un cierre 16:00; mezclar ORATS con otro
  vendor crea el salto de nivel del §2.4 de `options_signals.md`. Documentar `capture_time` en
  la caché, como exige ese informe.

### 2.5 Alpha Vantage HISTORICAL_OPTIONS — el backfill de pago más barato, con dos asteriscos gordos

- **Realidad del plan gratuito** (verificado contra documentación y terceros): la función
  `HISTORICAL_OPTIONS` **sí responde con clave gratuita** (el ejemplo con `apikey=demo` es
  público), con el límite global de **25 peticiones/día**. La nota de
  `datasets_opciones.md` §3.1 ("requiere premium") es **imprecisa en 2026**: lo premium es el
  tiempo real (`REALTIME_OPTIONS`) y el caudal, no el acceso al histórico EOD.
  25 req/día = 25 ticker-días/día → 31.200 ticker-días del conjunto MÍNIMO = **3,4 años**: el
  tier gratuito es inutilizable para backfill en la práctica.
- **Premium**: 49,99 USD/mes (75 req/min) hasta 249,99 (1.200 req/min), sin límite diario.
  1 petición = 1 ticker-día de cadena completa (15+ años de profundidad, con IV y griegas).
  A 75 req/min: MÍNIMO ≈ **7 h**, MEDIO ≈ **35 h**, COMPLETO ≈ **128 h** → **un solo mes de
  49,99 USD** paga cualquiera de los tres conjuntos.
- **Asterisco 1 — la IV y las griegas NO se usan.** `datasets_opciones.md` §3.1 lo midió sobre
  405.804 filas derivadas de este mismo endpoint: violación mediana de paridad put-call de
  **7,8 puntos de IV**, 36 % de pares con >10 pp, flag ITM roto al 100 %. Se compran **bid/ask,
  volumen y OI**; la IV se recalcula en el repo (forward implícito + Black-76).
- **Asterisco 2 — supervivencia**: cobertura de tickers deslistados sin confirmar
  **[SIN VERIFICAR]**; el test de diligencia del §5.2 es obligatorio antes de pagar el mes.
- Términos personales/no redistribución: los ficheros se quedan en la caché local, fuera del repo.

### 2.6 Cboe DataShop — el exchange, con dos productos muy distintos

- **Option EOD Summary** (tarifas efectivas dic-2025, vía Federal Register/DataShop):
  suscripción **500 USD/mes**; **petición ad-hoc de histórico 400 USD** ("per request per
  month", con datos desde **enero de 2018**) — la redacción tarifaria es ambigua entre "400 por
  solicitud" y "400 por mes de datos solicitado" **[SIN VERIFICAR la interpretación; cotizar en
  el carrito de DataShop antes de asumir nada]**. Intradía: 10-min 1.500 USD/mes o 18.000/año;
  1-min 6.000 USD/mes o 72.000/año; ad-hoc 750 y 2.500 respectivamente.
- **Académico**: 50 % de descuento en EOD Summary (mínimo 500 USD) y — lo importante —
  **Open-Close Volume Summary a 750 USD el primer año**.
- **El producto que de verdad importa aquí es el Open-Close**, no el EOD: desagrega el volumen
  en compras/ventas de **apertura vs cierre** por tipo de participante — el dato con el que se
  construyó Pan-Poteshman (2006) y la única forma de tener `put_call_volume_ratio` **firmado**
  (la versión sin firmar es "la más débil de las señales", `options_signals.md` §5.3). Es la
  compra estrella de la Fase 2 si hay afiliación académica.

### 2.7 historicaloptiondata.com (DeltaNeutral) — ficheros desde 2002

- **Precios** (tienda del vendor): 5 años **L1 615 USD**, L2 945, L3 1.415, PRO 7.095;
  **histórico completo desde 2002: PRO 10.595 USD**; el completo en L1 existe como producto
  pero su precio no quedó confirmado **[SIN VERIFICAR; estimar 1.500–2.000 USD]**. EOD de todo
  el mercado USA (~5.900 subyacentes), bid/ask, último, volumen, OI, IV y griegas.
- **Papel**: la alternativa de compra única si se quisiera el tramo **2002–2007** que ORATS no
  tiene (burbuja puntocom tardía + 2008 completo desde antes). Para todo lo demás, 599 USD de
  ORATS dominan a 615+ de DeltaNeutral porque incluyen la superficie ya construida.
- Mismo aviso de siempre: IV/griegas de vendor barato → usar precios, recalcular IV.

### 2.8 Los que quedan

- **optiondata.org / historicaldata.net**: archivos EOD desde mayo-2002 (~24 años), CSV por día,
  con griegas e IV incluidas; **paquetes fijos desde 99 USD**, archivo completo y suscripción
  diaria con precio no confirmado **[SIN VERIFICAR]**. Su corte gratuito ene–jun 2013 (1,8 GB)
  sigue siendo la mejor muestra de calibración de coste cero (ya contemplada en
  `datasets_opciones.md` §5.3).
- **Intrinio**: precios no públicos (rango citado por terceros: desde ~250 USD/mes
  **[SIN VERIFICAR]**), licencia orientada a uso empresarial/display. Para un proyecto de
  investigación personal, fricción de ventas sin ventaja de datos: **descartado**.
- **EODHD**: el add-on de opciones del marketplace (UnicornBay, ~30 USD/mes **[SIN VERIFICAR]**)
  da 40+ campos y ~6.000 tickers pero **~1 año de histórico rodante**: inservible para
  backfill. Descartado para opciones (sigue siendo válido para lo que ya se usa en
  `data_sources.md`).
- **OptionMetrics (IvyDB)**: el patrón oro académico desde 1996; solo institucional vía WRDS,
  precio no publicado. Si en algún momento hay afiliación universitaria, resuelve todo este
  documento de golpe; sin ella, no es una opción.

---

## 3. Dataset mínimo con potencia estadística

Cálculos ejecutados con `scipy 1.x` + `statsmodels` en este entorno
(script: `potencia_opciones.py`, reproducible; test t de dos muestras bilateral y transformación
de Fisher para el IC; α = 0,05, potencia = 0,80).

### 3.1 Tamaños de efecto que queremos poder detectar

De `docs/research/options_signals.md` (§1, §3.3, §6) e `informed_trading.md` (§1):

| Efecto | Magnitud publicada | Supuesto de dispersión | Fuente |
|---|---|---|---|
| `cavs` (vol_spread acumulado) en ventana de anuncio, Q5−Q1 | **+1,5 %** en CAR de 2 días | sd(CAR 2d) ≈ 5,5 % (movimiento implícito típico 4–6 %) | Atilgan (2014) ⚠ |
| Lo mismo, **descontado a S&P 500** | **~+0,5 %** (Muravyev et al. 2025: ≥2/3 del efecto es artefacto de préstamo; en el S&P 500 esperar ~1/3 del efecto bruto) | ídem | §1 y §3.4 de options_signals |
| `vol_spread` semanal, Q5−Q1 | 50 pb/semana | sd semanal ≈ 4 % | Cremers-Weinbaum ⚠ |
| O/S Johnson-So, D1−D10 | 34 pb/semana | ídem | Johnson-So (2012) ⚠ |
| IC objetivo del proyecto por señal | 0,02–0,05 | — | contrato §3.4 |

### 3.2 Cuántos eventos hacen falta (α=0,05, potencia 0,80)

**A. Spread de quintiles (test t dos muestras; "eventos totales" = 5 × n por grupo, porque el
ranking necesita la sección cruzada entera):**

| Efecto | n por quintil | **Eventos totales** | Con Bonferroni m=36 (12 señales × 3 ventanas; cota superior de BH) |
|---|---|---|---|
| Atilgan literatura (1,5 pp / 5,5 %) | 212 | **1.060** | 2.206 |
| Intermedio (1,0 pp) | 476 | **2.379** | 4.947 |
| **S&P 500 realista (0,5 pp)** | 1.900 | **9.502** | **19.748** |
| Cremers-Weinbaum semanal (50 pb / 4 %) | 1.006 | **5.028** | 10.451 |
| Johnson-So O/S (34 pb / 4 %) | 2.174 | **10.868** | 22.587 |

**B. Detección de IC (Fisher):**

| IC | n eventos | con Bonferroni m=36 |
|---|---|---|
| 0,05 | 3.137 | 6.516 |
| 0,04 | 4.903 | 10.186 |
| 0,03 | 8.719 | 18.114 |
| 0,02 | 19.620 | 40.767 |

**C. Puente entre ambas escalas** (señal y retorno ≈ normales): media del quintil extremo de una
N(0,1) = 1,40 ⇒ `Q5−Q1 = 2,80 × IC × sd_retorno`. El efecto "Atilgan literatura" equivale a
IC ≈ 0,097; el "S&P 500 realista" a **IC ≈ 0,032** — coherente con el rango 0,02–0,05 que el
proyecto considera creíble.

**Lectura:** para *replicar* los efectos de la literatura bastan ~1.000–2.500 eventos; para
*detectar lo que de verdad cabe esperar en el S&P 500* hacen falta **~9.500 eventos** (y ~20.000
si se quiere sobrevivir a la corrección por multiplicidad con margen). Ese es el número que
manda sobre cualquier decisión de compra.

### 3.3 Los tres conjuntos objetivo

26 sesiones por evento (T−20..T+5), 4 eventos/ticker/año, cadena completa ≈ 2.000
contratos/ticker-día (≈250 tras los filtros F1–F10 + DTE≤90):

| Conjunto | Definición | Eventos | Ticker-días | Registros (cadena completa) | Tamaño ~parquet | Detecta (Q5−Q1 en CAR 2d) | IC mínimo |
|---|---|---|---|---|---|---|---|
| **MÍNIMO** | 100 tickers × 3 años (2022–2025) | 1.200 | 31.200 | 62 M | ~8 GB | ≥1,41 pp | ~0,09 |
| **MEDIO** | 503 tickers × 3 años | 6.036 | 156.936 | 314 M | ~38 GB | ≥0,63 pp | ~0,04 |
| **COMPLETO** | 503 tickers × 11 años (2013–2024) | 22.132 | 575.432 | 1.151 M | ~138 GB | ≥0,33 pp (y ≈0,5 pp aun con Bonferroni m=36) | ~0,02 |

**Juicio honesto sobre el MÍNIMO:** con 1.200 eventos solo se detectan efectos de tamaño
"literatura completa". Sirve para (a) montar y validar el pipeline entero contra datos reales y
(b) replicar/descartar los efectos grandes; **no** autoriza a declarar muerta una señal de
tamaño realista. El conjunto que responde la pregunta del proyecto es el **MEDIO→COMPLETO**.
Nótese que el benchmark que pidió el usuario (100×3) es exactamente el MÍNIMO.

### 3.4 Cotización de cada conjunto con los precios del §2

| Vendor / vía | MÍNIMO (31,2 k ticker-días) | MEDIO (157 k) | COMPLETO (575 k) |
|---|---|---|---|
| **ORATS** | 2 meses API = 198 USD | **compra única 599 USD** (domina a 8 meses × 99) | **599 USD** (cubre 2007→hoy, sobra histórico) |
| **Alpha Vantage premium** (solo precios/vol/OI) | **50 USD** (~7 h a 75 req/min) | **50 USD** (~35 h) | **50–100 USD** (~128 h, 1–2 meses) |
| **ThetaData** | 1 mes Starter = **40 USD** (4 años cubren 2022–2025) | 1–2 meses Starter/Professional = **40–160 USD** | 1–2 meses del nivel alto = **160–320 USD** (histórico a 2012) |
| **Databento** | 1 mes Standard = **199 USD** o pago por uso de ~8–11 GB (crédito de 125 USD lo cubre en parte) [SIN VERIFICAR USD/GB] | 199–398 USD (~40–55 GB) | 398–597 USD (~150 GB, 2013→) |
| **Polygon/Massive** | Developer ~79–99 USD/mes (Starter solo llega a 2 años) | ~79–199 USD | Advanced 199 USD/mes × 1–2 = **199–398 USD** (2014→) |
| **Cboe DataShop ad-hoc** | ≥400 USD y ambigüedad tarifaria (§2.6) | probablemente miles [SIN VERIFICAR] | ídem; solo tiene sentido la vía académica |
| **DeltaNeutral** | L1 5 años = 615 USD | 615 USD | completo L1 [SIN VERIFICAR ~1,5–2 k] o PRO 10.595 USD |
| **IBKR** | **imposible** (§1.3) | imposible | imposible |

Conclusión de la tabla: **el COMPLETO cuesta entre 50 USD (Alpha Vantage, solo precios crudos y
con diligencia previa) y ~600 USD (ORATS, superficie lista y 2007→)**. No hay ningún escenario
en el que comprar más de ~600 USD de datos EOD esté justificado antes de haber visto señal.

---

## 4. Camino recomendado, por fases y con coste

Principio rector: **cada fase compra la respuesta a una pregunta, no un dataset.** Si la
respuesta es "no", el proyecto para ahí y el dinero no gastado es el beneficio.

### Fase 0 — 0 USD, empieza hoy

1. **DoltHub `post-no-preference/options`** (2019 → 2024-06, ~2.098 símbolos, tabla
   `option_chain` con bid/ask, volumen, OI y griegas; el volcado CSV ronda 6 GB). Descargar
   fuera de este entorno (dolthub.com está bloqueado aquí), **verificar la licencia en la ficha
   del repo** (la mayoría de DoltHub es Creative Commons; está confirmado en su blog que es dato
   comunitario publicado a diario desde 2021) y pasarle los filtros F1–F10 + el test de paridad
   put-call de `options_signals.md` antes de creerle una sola IV.
   Rendimiento esperado: ~500 tickers S&P × 5,5 años × 4 ≈ **9.000–11.000 eventos** → potencia
   para efectos ≥ ~0,5 pp: **la Fase 0 ya alcanza el umbral "S&P 500 realista" del §3.2** si la
   cobertura y calidad aguantan.
2. **optiondata.org ene–jun 2013 gratuito** (1,8 GB): una temporada de resultados completa de
   otra época de volatilidad, como muestra de control externa al periodo DoltHub.
3. **Recolector propio ya especificado** (`docs/RECOLECTOR.md`): cada día sin capturar es panel
   perdido. IBKR entra aquí solo como fuente redundante opcional (§1.6).
4. VIX (ya en el repo) como control de régimen.

**Pregunta que responde la Fase 0:** *¿alguna de las 12 señales EOD sobrevive a la
ortogonalización del §3.7/§11 de options_signals.md con |IC| ≥ 0,04–0,05 en 2019–2024?*
Si ninguna sobrevive con IC ≥ 0,04 en ~10.000 eventos, la probabilidad de que exista un efecto
explotable de 0,5 pp que la Fase 1 fuera a rescatar es baja — y se acabó el gasto.

### Fase 1 — ~600–650 USD, solo si la Fase 0 muestra señal

1. **ORATS, descarga única de 599 USD (2007→hoy)**. Compra exactamente lo que la Fase 0 no
   puede dar: (a) **dos regímenes de crisis** (2008–09, 2020) y 12 años fuera de la muestra de
   DoltHub; (b) superficie a delta y madurez constantes construida con un método uniforme en
   toda la serie (sin salto de proveedor a mitad de muestra); (c) el conjunto COMPLETO del §3.3
   con margen.
2. Opcional (+50 USD): **un mes de Alpha Vantage premium** para descargar cadenas crudas
   (precios/volumen/OI) de una submuestra y **contrastar la superficie de ORATS contra IV
   recalculada en el repo** sobre el forward implícito. Si ORATS y tu Black-76 discrepan
   sistemáticamente, quieres saberlo antes de calibrar nada.

**Pregunta que responde la Fase 1:** *¿la señal de la Fase 0 es estable entre regímenes
(2008/2020 incluidos) y sobrevive con IC ≥ 0,03 en ~20.000 eventos con corrección BH?* Es el
listón del §3.2 para creérsela de verdad.

### Fase 2 — 199 USD/mes o 750 USD/año, solo si la Fase 1 confirma y se va a operar

Elegir según el diagnóstico de la Fase 1:

- Si lo que falta es **flujo firmado** (la mitad fuerte de la literatura: Pan-Poteshman,
  Ge-Lin-Pearson): **Cboe Open-Close Volume Summary, 750 USD el primer año con descuento
  académico** — requiere afiliación; sin ella, su precio estándar hay que cotizarlo en DataShop.
- Si lo que falta es **microestructura/horquilla efectiva** para pasar de IC a P&L neto de
  costes: **Databento Standard 199 USD/mes** (NBBO a minuto 2013→, trades, OI) uno o dos meses
  sobre las ventanas de evento.
- Como recolector primario redundante en producción: **ThetaData Starter/Professional
  40–80 USD/mes**, que además de la foto EOD da el intradía para auditar la hora de captura.

**Pregunta que responde la Fase 2:** *¿el alfa sobrevive a costes de transacción medidos con
horquillas reales, y el flujo firmado añade IC incremental sobre el compuesto EOD?*

---

## 5. Lo que NO recomiendo, y los riesgos

### 5.1 Descartes razonados

| Descartado | Por qué |
|---|---|
| **IBKR como fuente de histórico** | §1.3: la API no expone expirados; el 0 % del dataset objetivo es alcanzable. Cualquier plan que "empiece bajando de IBKR" está muerto antes de empezar. |
| **EODHD para opciones** | ~1 año de histórico rodante: no hay potencia estadística posible (§3.2 exige miles de eventos). |
| **Intrinio** | Precio opaco orientado a empresas; ninguna ventaja de datos sobre ThetaData/Databento/ORATS para este caso de uso. |
| **Republicaciones "gratis" (Kaggle, mirrors de GitHub)** | Ya destripado en `datasets_opciones.md` §3.1: licencia inválida y — medido — IV/griegas rotas. No se usa, no se cita. |
| **DeltaNeutral PRO-HDALL (10.595 USD) y los intradía de DataShop (18–72 k USD/año)** | Compran granularidad que ninguna pregunta de las fases 0–2 necesita. Si algún día hace falta tick histórico, Databento lo da por dos órdenes de magnitud menos. |
| **La IV/griegas de Alpha Vantage (y de cualquier vendor barato) como señal** | Evidencia medida en este repo: violaciones de paridad de 7,8 pp de mediana. Se compran precios; la IV se calcula en casa (forward implícito + Black-76 + filtros F1–F10). |
| **OptionMetrics "porque es el estándar"** | Sin afiliación institucional no hay vía de compra; perseguirlo es coste de oportunidad puro. |
| **Scrapeo masivo de brokers/webs contra sus ToS** | Riesgo legal y de continuidad para ahorrar 40 USD/mes. No. |

### 5.2 Riesgos que hay que gestionar activamente

1. **Calidad de la IV barata.** El patrón se repite en todos los vendors de gama baja: precios
   bid/ask decentes, columnas analíticas malas. Regla del repo: la IV de vendor solo se acepta
   si pasa el test de paridad put-call por pares (diferencia mediana < 1 pp); si no, se
   recalcula. Presupuestar el tiempo de cómputo propio como parte del coste del dato.
2. **Supervivencia de contratos y de subyacentes expirados.** Un feed que enumera "por símbolo
   actual" pierde los deslistados (y el S&P 500 PIT de este repo los necesita: fusiones,
   quiebras). **Test de diligencia antes de pagar a cualquier vendor**: pedir la cadena de
   TWTR en 2022-09, de FRC en 2023-03 y de un ticker con cambio de símbolo (FB→META) y
   comprobar que existen. Polygon (`as_of`) y Databento (`definition` por fecha) lo resuelven
   estructuralmente; en Alpha Vantage y EODHD está **[SIN VERIFICAR]**.
3. **Construir sobre datos que luego no te puedes permitir.** Dos formas de la misma trampa:
   (a) calibrar features contra la *superficie parametrizada* de ORATS y descubrir que sin
   suscripción no puedes reproducirlas hacia adelante — mitigación: el recolector guarda cadena
   cruda y todas las features se derivan con código del repo, de modo que ORATS sea una fuente
   más, no una dependencia; (b) validar con la microestructura de Databento y no poder pagar el
   feed en producción — mitigación: la Fase 2 solo se compra si hay decisión de operar.
4. **Mezcla de horas de captura.** ORATS ~15:46 ET, ThetaData/Databento al cierre, el recolector
   propio 16:15–17:00 ET. `options_signals.md` §2.4 cuantifica que 10 pb de desfase del
   subyacente meten 0,9 pp de ruido en `vol_spread` a 30 días. Toda serie lleva
   `capture_time` en metadatos y **no se concatenan proveedores sin un estudio de empalme**.
5. **Open interest es T+1.** Da igual el vendor: el OI de la sesión T lo publica la OCC la
   mañana de T+1 (`data_sources.md` §7.3). `available_at = apertura de T+1`, siempre.
6. **Licencias y redistribución.** Nada de lo comprado entra en el repositorio git; vive en
   `data/cache/` local. El único dato de opciones redistribuble en el repo sigue siendo el
   fixture Apache-2.0 de Lean y el VIX PDDL.
7. **La multiplicidad no es opcional.** Con 12 señales × 3 ventanas, un "descubrimiento" a
   p=0,04 es ruido esperado. Los números del §3.2 con Bonferroni/BH son el listón real; el
   pipeline de validación del repo (BH en `stats/validation`) debe aplicarse también a este
   ángulo.

---

## 6. Fuentes consultadas (2026-08-04/05)

Vía WebSearch (el proxy devuelve 403 al fetch directo de la mayoría de estos dominios):

- IBKR: `interactivebrokers.github.io/tws-api/historical_limitations.html`;
  IBKR Quant Blog *Historical Options & Futures Data using TWS API* (partes I y II);
  `interactivebrokers.com/en/pricing/market-data-pricing.php`; guías de pacing del Web API;
  hilos de Elite Trader sobre expirados y pacing; `ibkrguides.com` (regulatory snapshots).
- ThetaData: `thetadata.net/pricing`, `thetadata.net/options-data`, docs `Subscriptions`;
  QuantConnect Lean CLI docs (cobertura 2012-06-01 UTP / 2020-01-01 CTA); comparador
  FlashAlpha (jun-2026, precios en conflicto — señalado).
- Databento: blog *Introducing new OPRA pricing plans* (jun-2025), *OPRA improvements coming
  soon* (extensión a 2013-04), `databento.com/datasets/OPRA.PILLAR`, anuncio oficial en X
  (Standard 199 USD, 10 años, 18 exchanges), `databento.com/stocks` (crédito 125 USD).
- Polygon/Massive: `massive.com/pricing` y docs de flat files (histórico 2014→); `apis.io` y
  `apicostcalc.com` (tarifario pre-rebranding); artículo de revisión de Medium.
- ORATS: `orats.com/data-api`, `orats.com/near-eod-data` (599 USD una vez, 2007→, FTP
  2 semanas; captura ~14 min antes del cierre), `orats.com/one-minute-data` (1-min desde
  ago-2020).
- Alpha Vantage: `alphavantage.co/documentation` y `alphavantage.co/premium` (25 req/día
  gratis; 49,99–249,99 USD/mes por caudal; `HISTORICAL_OPTIONS` responde con clave gratuita,
  15+ años).
- Cboe DataShop: `datashop.cboe.com/option-eod-summary`, `.../cboe-options-open-close-volume-summary`;
  Federal Register 2024-00848 (tarifas EOD/intradía/ad-hoc y académico 750 USD).
- DeltaNeutral: `historicaloptiondata.com/shop` y páginas de producto (615/945/1.415/7.095/10.595 USD).
- optiondata.org / historicaldata.net: página principal y `resource.html` (archivo desde
  may-2002, paquetes desde 99 USD, muestra 2013 gratuita).
- DoltHub: blog *Dolt + post-no-preference: Open Data* (2024-09-27) y ficha
  `dolthub.com/repositories/post-no-preference/options`.
- Intrinio: `intrinio.com/pricing`, `intrinio.com/options`, Datarade (rango de precios).
- EODHD: `eodhd.com/marketplace/unicornbay/options` (~1 año de histórico, 40+ campos).

Cálculos de potencia: script `potencia_opciones.py` ejecutado en este entorno con scipy y
statsmodels (test t bilateral vía `TTestIndPower`, IC vía transformación de Fisher); los
supuestos de dispersión (sd CAR 2d = 5,5 %, sd semanal = 4 %) están declarados en §3.1 y son
los dos únicos números de este documento que no proceden ni de un vendor ni de un paper, sino
de calibración razonada — sensibles de revisar cuando la Fase 0 dé las sd empíricas.
