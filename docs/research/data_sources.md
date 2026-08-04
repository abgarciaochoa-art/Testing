# Mapa de fuentes de datos para `earnings-alpha`

**Ámbito.** Este documento inventaría, evalúa y prioriza **todas** las fuentes de datos
que necesitan los dos ángulos de la plataforma sobre el S&P 500 completo:

- **Ángulo A** — factores fundamentales cross-section evaluados todo el año
  (`earnings_alpha/factors/*`).
- **Ángulo B** — la ventana en torno al anuncio de resultados y la detección de la
  huella estadística de negociación informada previa al anuncio
  (`earnings_alpha/events/*`).

Para cada fuente se documenta: **endpoint exacto**, autenticación, límite de tasa,
profundidad histórica, coste (plan gratuito y de pago) y —lo más importante para este
repo— si el dato es **point-in-time (PIT)** o está **reexpresado / contaminado por
revisión posterior**.

---

## 0. Cómo leer este documento

### 0.1 Estado de verificación de cada cifra

El entorno de desarrollo tiene el egress restringido y, además, **`WebFetch` devuelve
403 para cualquier host, incluido `example.com`**: no ha sido posible descargar ni una
sola página de tarifas ni una sola especificación de API en su fuente primaria. Toda la
investigación se ha hecho con `WebSearch`, que devuelve resúmenes sintetizados. Esto
obliga a etiquetar la procedencia de cada afirmación:

| Marca | Significado |
|---|---|
| **[verificado]** | Comprobado *ejecutando código* contra datos que están en este repo. Es la única categoría con certeza total. |
| **[búsqueda]** | Procede de resultados de `WebSearch` de agosto de 2026, coherentes entre sí. Fiabilidad alta para hechos estructurales (existencia de endpoints, esquema de tarificación), **media para precios exactos**. |
| **[verificar]** | Dato de una sola fuente secundaria, o que la búsqueda no consiguió confirmar. **No debe usarse para decidir una compra sin comprobarlo en la web del proveedor.** |
| **[prior]** | Estimación de ingeniería propia, no un número publicado. |

**Advertencia explícita sobre los precios.** Las tarifas de las APIs financieras cambian
varias veces al año y con frecuencia se anuncian con descuentos anuales que confunden la
comparación. Todas las cifras en USD/mes de este documento están sujetas a comprobación
antes de contratar. Ninguna cifra de este documento es un dato inventado: donde no hay
número fiable, se dice *"no publicado"* y no se rellena con una estimación disfrazada.

**Cambio de marca detectado.** `polygon.io` está migrando a `massive.com`; la
documentación y las tarifas ya se sirven desde ese dominio, aunque la API y las claves
siguen siendo las mismas [búsqueda]. El adaptador debe seguir apuntando a
`api.polygon.io` pero conviene registrar el riesgo.

### 0.2 Qué se ha verificado ejecutando código

Se validaron los ficheros semilla del repo con `pandas`, y se comprobó además toda la
aritmética de cuotas y costes citada en el documento (§7.2, §14). El script de
verificación se ejecutó fuera del árbol del repositorio, en el scratchpad de la sesión
(`verify_seed_claims.py`), y **las 26 comprobaciones pasan**: 18 sobre los datos semilla
y 8 sobre los cálculos de cuota. Los resultados están en la §12 y se citan en línea donde
son relevantes. Ninguna de esas cifras es una estimación.

Se deja constancia de que el script no se versiona en el repo por la regla de propiedad
de ficheros: este agente sólo escribe `docs/research/data_sources.md`. Si se quiere
convertir en test permanente, su sitio natural sería `tests/test_seed_integrity.py`,
propiedad del módulo `universe`.

### 0.3 Criterio de evaluación

Cada fuente se juzga contra cinco ejes, en este orden de importancia para el repo:

1. **PIT-idad**: ¿existe un campo que indique *cuándo* el dato fue públicamente
   conocible? Sin él, el dato no puede entrar en una señal (§1 del contrato de
   arquitectura). Un proveedor barato con `available_at` fiable vale más que uno caro
   sin él.
2. **Ausencia de sesgo de supervivencia**: ¿incluye tickers deslistados? El universo del
   repo contiene **644 símbolos que estuvieron en el S&P 500 y ya no están**
   [verificado, §12.2]. Un proveedor que solo sirve los 503 actuales pierde el **57,2 %**
   de los símbolos que alguna vez formaron el panel [verificado].
3. **Profundidad histórica** frente al rango de la semilla (1996-01-02 → 2025-08-23).
4. **Coste** y límite de tasa frente a un universo de ~500-1100 símbolos.
5. **Estabilidad contractual**: API oficial documentada vs. endpoint no oficial.

---

## 1. Qué datos necesita exactamente cada módulo

Esta tabla es la que determina el gasto. Sin ella, la discusión de precios no significa
nada.

| # | Dato | Consumidor en el repo | Criticidad | Frecuencia | Profundidad mínima útil |
|---|---|---|---|---|---|
| 1 | OHLCV diario ajustado + factores de ajuste crudos | `data/prices.py`, todos los factores, `backtest`, `events.abnormal_returns` | **Bloqueante** | diaria | 1996→ (o 2010→ para una v1 honesta) |
| 2 | OHLCV intradía (1 min) | `events.PreEventFeatures.order_imbalance_proxy`, `backtest.EventBacktest` (gap overnight) | Alta | 1 min | ≥5 años |
| 3 | Fundamentales trimestrales as-reported con fecha de presentación | `data/fundamentals.py`, factores de calidad/valor/accruals/F-Score | **Bloqueante** | trimestral | 2009→ (XBRL) |
| 4 | Calendario de resultados con **fecha y hora** (BMO/AMC) | `pit.tradable_date`, `events.event_windows`, todo el ángulo B | **Bloqueante** | por evento | 1996→ deseable, 2004→ suficiente |
| 5 | Consenso de analistas **histórico con vintages** | `factors` (momentum de revisiones), `SUE` analista, `PreEventFeatures.analyst_revision_drift` | Alta (y cara) | diaria/mensual | 10 años |
| 6 | Consenso **en el instante del anuncio** (sólo el último) | `EarningsEvent.eps_estimate`, SUE clásico | **Bloqueante para SUE-analista** | por evento | 10 años |
| 7 | Cadenas de opciones con IV y OI | `data/options.py`, `PreEventFeatures.{vol_spread, iv_skew_25delta, iv_term_slope, oi_buildup_*, put_call_volume_ratio}` | Alta | diaria | ≥5 años |
| 8 | Short interest agregado (FINRA) | `PreEventFeatures.short_interest_delta` | Media | quincenal | 2007→ |
| 9 | Volumen off-exchange / ATS | `PreEventFeatures.off_exchange_share_delta` | Media | semanal (ATS) / diaria (Reg SHO) | 2013→ |
| 10 | SEC EDGAR (8-K 2.02, 10-Q/10-K, Form 4) | `data/edgar.py`, `PreEventFeatures.insider_net_buy_form4` | **Bloqueante** | continua | 1994→ (2004→ para 2.02) |
| 11 | Transcripciones de earnings calls | *(no consumido por el contrato actual)* | Baja | por evento | opcional |
| 12 | Pertenencia histórica al índice | `universe/*`, todo lo demás | **Bloqueante** | diaria | 1996→ |

**Lectura operativa.** Los bloqueantes (1, 3, 4, 10, 12) se cubren **íntegramente con
fuentes gratuitas**. Lo que cuesta dinero es (5), (7) y, en menor medida, (2). Esa es la
conclusión que estructura toda la recomendación de la §14.

---

## 2. El problema PIT, dato por dato

El contrato del repo dice *"toda serie fundamental se indexa por `available_at`, nunca
por `period_end`"*. Pero `available_at` significa algo distinto en cada familia de datos,
y **la mayoría de los proveedores no lo expone**. Esta sección fija la definición
operativa para cada uno.

| Dato | `available_at` correcto | Error típico | Look-ahead que introduce |
|---|---|---|---|
| EPS/ingresos del trimestre | `acceptanceDateTime` del 8-K item 2.02 (o el instante del comunicado) | usar `period_end` | 20-60 días naturales |
| Balance / flujo de caja completo | `filed`/`accepted` del 10-Q o 10-K | usar la fecha del 8-K de resultados | 25-45 días |
| Precio ajustado por splits/dividendos | el ajuste debe reconstruirse con los eventos conocidos **hasta `t`** | usar `adjClose` de hoy | sutil pero real (filtros de precio, tamaño, número de acciones) |
| Consenso de analistas | la fecha de corte del snapshot (`STATPERS` en IBES) | usar el consenso final de la serie | de días a meses; **es el look-ahead más frecuente de esta literatura** |
| Short interest FINRA | fecha de **diseminación** (≈7º día hábil tras el `settlementDate`) | usar `settlementDate` | 9-11 días naturales |
| Volumen ATS FINRA | fecha de publicación = fin de semana + **2 semanas** (Tier 1) | usar `weekStartDate` | **14 días**: destruye cualquier feature pre-evento ingenua |
| Cadena de opciones | cierre de la sesión del `tradeDate` | usar el OI del día siguiente (OCC lo publica por la mañana) | 1 día |
| Pertenencia al índice | fecha efectiva del cambio (no la del anuncio de S&P) | usar la lista de hoy | sesgo de supervivencia estructural |
| Fundamentales reexpresados | *no existe* `available_at` válido | usar `MRQ`/restated | contamina cualquier factor de calidad |

### 2.1 Las tres fechas de EDGAR que no son la misma

Este punto es el que más backtests rompe y merece precisión quirúrgica:

- **`filingDate`** — la fecha *administrativa* del expediente. EDGAR aplica un corte a
  las **17:30 ET**: un envío iniciado después de esa hora (y aceptado) recibe fecha de
  presentación del **siguiente día hábil** [búsqueda]. Excepciones: Formularios 3, 4 y 5
  y correspondencia, que conservan la fecha del día hasta las 22:00 ET [búsqueda].
- **`acceptanceDateTime`** — el instante real (ET) en que EDGAR aceptó el envío. **Es el
  campo que hay que usar**, y está en `https://data.sec.gov/submissions/CIK##########.json`.
- **`period`** / `period_end` — cierre del trimestre reportado. Nunca es una fecha de
  señal.

**Consecuencia concreta y no obvia:** un anuncio AMC a las 16:05 ET recibe
`filingDate` = mismo día; uno a las 18:00 ET recibe `filingDate` = día siguiente. Ambos
son AMC del *mismo* día de negociación y deben producir el **mismo** `tradable_date`. Un
pipeline que use `filingDate` los desplaza de forma inconsistente: unos eventos con un
día de retraso y otros no. Con `acceptanceDateTime` el problema desaparece.

### 2.2 El comunicado precede al 8-K

El 8-K item 2.02 se *presenta* (técnicamente, se "suministra", *furnish*) **después** de
que el comunicado haya salido por el hilo de prensa. El desfase típico va de unos minutos
a un par de horas [prior; no se ha localizado un estudio con la distribución exacta].

Esto tiene una implicación favorable: **`acceptanceDateTime` es una cota superior
conservadora de `announced_at`**. Un backtest que la use nunca opera antes de que la
información fuese pública; como mucho, opera un poco tarde. Para una plataforma cuyo
principio número uno es *"point-in-time o nada"*, ese sesgo es del signo correcto.

El riesgo residual es de **clasificación de sesión**, no de look-ahead:

- comunicado 08:30 ET, 8-K aceptado 09:41 ET → clasificado `DMH` cuando era `BMO`;
- comunicado 16:01 ET, 8-K aceptado 16:35 ET → clasificado `AMC`, correcto;
- comunicado 07:00 ET, 8-K aceptado 07:15 ET → `BMO`, correcto.

Regla propuesta para `data/edgar.py` (decisión de diseño, no un hecho externo):

```text
acc_et = acceptanceDateTime convertido a America/New_York
si acc_et.time() <  09:30  -> Session.BMO
si acc_et.time() >= 16:00  -> Session.AMC
en otro caso               -> Session.DMH  (y marcar baja confianza)
```

...y **cruzar siempre** con el campo `hour`/`time` de un proveedor de calendario
(Finnhub, FMP). Cuando ambos coinciden, confianza alta. Cuando difieren, la política
conservadora del repo (`Session.UNKNOWN` → tratar como AMC) es la que menos daño hace,
porque retrasa la entrada un día en lugar de adelantarla.

---

## 3. (1) Precios y volumen: OHLCV diario e intradía

### 3.1 Por qué el ajuste de precios es un problema PIT y no un detalle

Casi todos los proveedores sirven `adj_close` **retro-ajustado con toda la historia de
splits y dividendos conocida hoy**. Para calcular *retornos* eso es correcto e inocuo.
Para cualquier otra cosa, no:

- un filtro `precio > 5 USD` aplicado sobre `adj_close` de hoy excluye/incluye empresas
  distintas de las que un gestor habría podido filtrar en 2003;
- el *earnings yield* y el *FCF yield* mezclan un EPS as-reported (en acciones de
  entonces) con un precio ajustado (en acciones de ahora) si no se tiene cuidado;
- los umbrales de liquidez (`adv_participation` en `CrossSectionalBacktest`) se calculan
  sobre volumen, que también se reajusta.

Por eso el criterio de selección de proveedor de precios de este repo es:
**se prefiere el que entrega `close` crudo + los factores de ajuste con su fecha**
(Tiingo, Polygon, EODHD) sobre el que sólo entrega `adjClose` (Yahoo). `types.Bar` ya
está diseñado para esto: `close` sin ajustar, `adj_close` opcional.

### 3.2 Fichas por proveedor

#### SEC EDGAR
No sirve precios. Punto. Cualquier arquitectura que lo pretenda está equivocada.

#### yfinance (Yahoo Finance no oficial) — `PROVIDER_ENV_KEYS["yfinance"] = []`

- **Endpoint real** (lo que la librería envuelve):
  `https://query2.finance.yahoo.com/v8/finance/chart/{symbol}?period1={epoch}&period2={epoch}&interval=1d&events=div%2Csplit`
- **Autenticación**: ninguna. Yahoo cerró su API pública en 2017; la librería raspa
  endpoints internos [búsqueda].
- **Límite de tasa**: no documentado y **cambiante**. En 2025-2026 los `429 Too Many
  Requests` se han vuelto sistemáticos en bucles sobre muchos símbolos [búsqueda].
  Mitigaciones habituales: `curl_cffi` con impersonación de navegador, backoff
  exponencial, y bajar a ≤1 símbolo/segundo [prior].
- **Profundidad**: diaria desde 1962 para nombres antiguos. Intradía: 1 m sólo ~30 días
  naturales (y ≤7 días por petición); 5 m/15 m/30 m ~60 días; 1 h ~730 días [verificar].
- **Coste**: 0.
- **PIT**: **no**. `adjclose` está retro-ajustado. No hay `available_at` de ningún tipo.
- **Sesgo de supervivencia**: **grave**. Los tickers deslistados desaparecen o devuelven
  series truncadas. De los 1126 símbolos que alguna vez estuvieron en el índice
  [verificado], los 27 con sufijo `Q` de quiebra (`ENRNQ`, `EKDKQ`, `AAMRQ`,
  `DPHIQ`, `LEHMQ`…) [verificado] son exactamente los que un backtest honesto necesita y
  Yahoo no tiene.
- **Veredicto**: aceptable como *fallback* y para desarrollo local; **inaceptable como
  única fuente de precios de un backtest histórico**. Debe registrarse en
  `ProviderRegistry` con prioridad baja.

#### Stooq — sin credencial

- **Endpoint**: `https://stooq.com/q/d/l/?s={symbol}.us&i=d` → CSV
  (`Date,Open,High,Low,Close,Volume`).
- **Auth**: ninguna. **Tasa**: no documentada; conservador ≤1 req/s [prior].
- **Profundidad**: décadas para US; conserva bastantes deslistados [verificar].
- **PIT**: no (precios ya ajustados, sin factores separados).
- **Veredicto**: excelente **segunda opinión gratuita** para detectar errores del
  proveedor primario (un `DataQualityError` cuando dos fuentes discrepan >X % es una
  validación barata y potente). No sirve como primaria por falta de metadatos.

#### Tiingo — `TIINGO_API_KEY`

- **Endpoints**:
  - EOD: `https://api.tiingo.com/tiingo/daily/{ticker}/prices?startDate=&endDate=&token=`
    → devuelve `close`, `adjClose`, `divCash`, `splitFactor` **por día**.
  - Metadatos y universo: `https://api.tiingo.com/tiingo/daily/{ticker}`
  - Lista completa con fechas de alta/baja:
    `https://apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip`
    (columna `endDate` → identifica deslistados).
  - Intradía IEX: `https://api.tiingo.com/iex/{ticker}/prices?resampleFreq=1min`
- **Auth**: `Authorization: Token <key>` o `?token=`.
- **Tasa**: gratis ~1 000 req/día y 50 símbolos únicos/hora [verificar]; Power
  10 000 req/hora y 40 GB/mes [búsqueda].
- **Profundidad**: EOD desde **1962**, ~80 000 activos [búsqueda]. IEX intradía sólo
  desde ~2017 y **sólo el flujo IEX** (≈2-3 % del volumen consolidado) → inútil para
  microestructura.
- **Coste**: gratis (no comercial, limitado); **Power 30 USD/mes** (uso no comercial);
  **Commercial 50 USD/mes**, mismos límites técnicos + licencia comercial [búsqueda].
  Fundamentales: **add-on aparte**, precio no publicado, hay que hablar con ventas
  [búsqueda].
- **PIT**: **parcial y bueno**: al entregar `divCash` y `splitFactor` por día se puede
  **reconstruir el ajuste tal y como era en `t`**. Es la mejor propiedad de Tiingo para
  este repo.
- **Veredicto**: **mejor relación calidad/precio para el panel diario 1996→**. Es la
  elección del nivel de ~50 USD/mes.

#### Polygon.io / Massive — `POLYGON_API_KEY`

- **Endpoints**:
  - `https://api.polygon.io/v2/aggs/ticker/{t}/range/1/day/{from}/{to}?adjusted=false`
  - `https://api.polygon.io/v2/aggs/ticker/{t}/range/1/minute/{from}/{to}`
  - `https://api.polygon.io/v3/reference/splits?ticker=` y `/v3/reference/dividends?ticker=`
  - `https://api.polygon.io/v3/reference/tickers?active=false` → **listado de
    deslistados**, que es oro para el sesgo de supervivencia.
  - Flat files (S3): `s3://flatfiles/us_stocks_sip/day_aggs_v1/YYYY/MM/YYYY-MM-DD.csv.gz`
- **Auth**: `?apiKey=` o `Authorization: Bearer`.
- **Tasa**: gratis **5 req/min**; todos los planes de pago **llamadas ilimitadas**
  [búsqueda].
- **Coste** (Stocks) [búsqueda, precios a confirmar]:
  | Plan | USD/mes | Latencia | Histórico |
  |---|---|---|---|
  | Basic (gratis) | 0 | fin de día | 2 años [verificar] |
  | Starter | 29 | 15 min diferido | 5 años [verificar] |
  | Developer | 79 | tiempo real IEX | 10 años [verificar] |
  | Advanced | 199 | SIP completo | 20+ años [verificar] |
- **Flat files incluidos en todos los planes de pago** desde 2025/2026 [búsqueda]: es un
  cambio importante, porque descargar 500 símbolos × 30 años por REST es inviable y por
  S3 es trivial. Los ficheros diarios finalizan hacia las **11:00 ET del día siguiente**
  [búsqueda] → `available_at` = T+1 mañana, dato relevante para señales intradía.
- **PIT**: bueno. `adjusted=false` + endpoints de splits/dividendos permiten
  reconstrucción PIT. Incluye deslistados.
- **Veredicto**: **la mejor fuente de intradía asequible**. Su punto débil para este
  proyecto es la profundidad: el plan de 29 USD probablemente no llega a 1996.

#### Alpaca — `ALPACA_API_KEY_ID`, `ALPACA_API_SECRET_KEY`

- **Endpoints**:
  `https://data.alpaca.markets/v2/stocks/bars?symbols=&timeframe=1Day&start=&end=&adjustment=all&feed=sip`
  y `/v1beta1/options/bars`, `/v1beta1/options/snapshots/{underlying}`.
- **Auth**: cabeceras `APCA-API-KEY-ID` / `APCA-API-SECRET-KEY`.
- **Tasa/coste**: gratis = sólo IEX en tiempo real + SIP diferido 15 min, **200 rpm**;
  **Algo Trader Plus 99 USD/mes** = SIP completo + OPRA, **10 000 rpm** [búsqueda].
- **Profundidad**: barras de minuto ~7 años [búsqueda]; el histórico SIP arranca hacia
  2016 [verificar]. **No cubre 1996-2015.**
- **PIT**: parámetro `adjustment` (`raw|split|dividend|all`) → reconstruible.
- **Veredicto**: buena si además se va a **operar** con Alpaca (mismo proveedor para
  datos y ejecución reduce el desfase entre backtest y producción). Como fuente de
  investigación pura, Polygon da más profundidad por menos dinero.

#### EODHD — `EODHD_API_KEY`

- **Endpoints**: `https://eodhd.com/api/eod/{TICKER}.US?from=&to=&fmt=json`,
  `/api/intraday/{TICKER}.US?interval=1m`, `/api/eod-bulk-last-day/US`,
  `/api/fundamentals/{TICKER}.US`, `/api/calendar/earnings`.
- **Auth**: `?api_token=`.
- **Tasa**: gratis 20 req/día; planes de pago **100 000 req/día y 1 000 req/min**
  [búsqueda]. Hay además un coste por *"API call weight"*: algunos endpoints consumen
  más de 1 crédito [verificar].
- **Coste** [búsqueda]: Free 0 · EOD All World **19,99** · All World Extended (EOD +
  intradía 1 m) **29,99** · Fundamentals Data Feed **59,99** · ALL-IN-ONE **99,99**
  USD/mes.
- **Profundidad**: "30+ años" de histórico, 60+ mercados [búsqueda]. Intradía 1 m para US
  desde ~2004 [verificar].
- **PIT**: fundamentales incluyen `filing_date` en parte de los registros [verificar];
  precios con `adjusted_close` y endpoints de splits/dividendos separados.
- **Veredicto**: el **paquete más completo por menos de 100 USD**. Su debilidad es la
  opacidad sobre la construcción de los fundamentales (no documenta si son as-reported o
  reexpresados).

#### Alpha Vantage — `ALPHAVANTAGE_API_KEY`

- **Endpoints**: `https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=`
  (premium desde 2023), `...&function=EARNINGS&symbol=`, `...&function=EARNINGS_CALENDAR&horizon=3month`.
- **Tasa**: gratis **25 req/día y 5 req/min** [búsqueda]. Planes: 49,99 (75 rpm) /
  99,99 (150) / 149,99 (300) / 199,99 (600) / 249,99 (1 200) USD/mes, **sin tope diario**
  [búsqueda].
- **Veredicto para precios**: **descartado**. 25 req/día no permite ni una carga inicial
  de 500 símbolos en menos de tres semanas. Sigue siendo interesante por otra razón, el
  endpoint `EARNINGS` (§5.3).

#### Nasdaq Data Link (ex-Quandl) — `NASDAQ_DATA_LINK_API_KEY`

- **Endpoint**: `https://data.nasdaq.com/api/v3/datatables/SHARADAR/SEP.json?ticker=&api_key=`
- **Tasa**: 50 000 llamadas/día para usuarios autenticados [búsqueda].
- **Aviso**: el dataset gratuito famoso **`WIKI/PRICES` está congelado desde
  2018-03-27** [prior, ampliamente conocido]. Usarlo hoy es un error clásico.
- Los precios útiles son los de **Sharadar SEP** (de pago, §4.4).

#### Databento

- OPRA y equities, **modelo de pago por uso** para histórico + plan **Standard desde
  199 USD/mes** con hasta 10 años de histórico [búsqueda].
- Es el único de la lista con **quotes** de calidad institucional a precio asequible, lo
  que importa si se quiere un `order_imbalance` con regla de Lee-Ready real en vez de un
  proxy.
- **Veredicto**: fuera del presupuesto de los niveles 0-2, pero es la respuesta correcta
  si el proyecto llega a necesitar microestructura de verdad.

### 3.3 Decisión sobre `order_imbalance_proxy`

`PreEventFeatures` pide un `order_imbalance_proxy`. Con datos **diarios** no se puede
clasificar cada operación como iniciada por comprador o vendedor. Hay tres niveles de
aproximación, y conviene implementar el más barato primero:

1. **Sólo OHLCV diario (gratis).** *Close Location Value* de Chaikin:
   `CLV = ((C − L) − (H − C)) / (H − L)`, y el flujo acumulado `CLV × Volume`. Es un
   proxy pobre pero no nulo, y no cuesta nada.
2. **Barras de 1 minuto (Polygon Starter, 29 USD).** Regla de tick de Lee-Ready aplicada
   al cierre de cada minuto: `sign(P_t − P_{t−1})`, con arrastre del signo anterior en
   los empates. Sesgo conocido y acotado; es lo que usa la mayoría de la literatura
   aplicada cuando no tiene quotes.
3. **Trades + NBBO (Databento / Polygon Advanced).** Lee-Ready canónico contra el punto
   medio. Es el único correcto, y cuesta un orden de magnitud más.

Recomendación: implementar (1) y (2) detrás de la misma interfaz, y que el `Provider`
resuelto determine cuál se calcula. Un feature que cambia de definición según el
proveedor **debe** llevarlo en el nombre (`order_imbalance_proxy_clv`,
`order_imbalance_proxy_tick`) para que ningún backtest mezcle dos definiciones sin
enterarse.

---

## 4. (2) Fundamentales trimestrales

### 4.1 SEC EDGAR XBRL: la fuente PIT gratuita, y por qué gana

Esta es la conclusión más importante de todo el documento: **para fundamentales, la
fuente gratuita es también la fuente más point-in-time que existe**. No hay que elegir
entre calidad y coste.

La razón es la estructura de `companyfacts`. Para cada concepto XBRL, EDGAR devuelve
**una lista de hechos**, y cada hecho lleva:

```json
{"start":"2023-10-01","end":"2023-12-30","val":119575000000,
 "accn":"0000320193-24-000006","fy":2024,"fp":"Q1","form":"10-Q",
 "filed":"2024-02-02","frame":"CY2023Q4"}
```

Cuando una empresa **reexpresa** una cifra, el nuevo valor aparece como un hecho
*adicional* para el mismo `(start, end)` con otro `accn` y otro `filed`. Es decir:
**`companyfacts` contiene todos los vintages**. Construir un panel PIT correcto es
entonces mecánico:

> Para el concepto `c`, la empresa `i` y el periodo `p`, el valor conocible en `t` es el
> hecho con `filed` **máximo entre los que cumplen `filed <= t`**; y el valor
> *as-originally-reported* es el de `filed` **mínimo**.

Esto habilita, gratis, dos cosas que normalmente exigen Compustat *Point-in-Time* (un
producto caro): el panel PIT y el estudio de **revisiones contables** como señal en sí
misma. `types.FundamentalFact` ya tiene `is_restated` y `accession` para modelarlo.

#### Endpoints exactos

| Uso | URL |
|---|---|
| Todos los hechos de una empresa | `https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json` |
| Un concepto de una empresa | `https://data.sec.gov/api/xbrl/companyconcept/CIK##########/us-gaap/{Tag}.json` |
| Un concepto, **todas** las empresas, un periodo | `https://data.sec.gov/api/xbrl/frames/us-gaap/{Tag}/USD/CY2024Q1I.json` |
| Metadatos y lista de filings | `https://data.sec.gov/submissions/CIK##########.json` |
| Mapa ticker→CIK (sólo actual) | `https://www.sec.gov/files/company_tickers.json` |
| Mapa con mercado | `https://www.sec.gov/files/company_tickers_exchange.json` |
| Índice diario de filings | `https://www.sec.gov/Archives/edgar/daily-index/{YYYY}/QTR{n}/master.{YYYYMMDD}.idx` |
| Índice trimestral completo | `https://www.sec.gov/Archives/edgar/full-index/{YYYY}/QTR{n}/master.idx` |
| Documento concreto | `https://www.sec.gov/Archives/edgar/data/{cik}/{accn_sin_guiones}/{fichero}` |
| Búsqueda a texto completo (2001→) | `https://efts.sec.gov/LATEST/search-index?q=...&forms=8-K&dateRange=custom&startdt=&enddt=` |
| Volcado masivo de companyfacts | `https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip` |
| Volcado masivo de submissions | `https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip` |
| Financial Statement Data Sets | `https://www.sec.gov/files/dera/data/financial-statement-data-sets/{YYYY}q{n}.zip` |

- **Autenticación**: ninguna, pero **`User-Agent` obligatorio** con nombre y correo
  (`"Nombre Apellido correo@dominio.com"`). Ya está previsto en
  `PROVIDER_ENV_KEYS["sec"] = ["SEC_USER_AGENT"]`.
- **Límite de tasa**: **10 req/s agregadas** sobre `www.sec.gov` + `data.sec.gov` +
  `efts.sec.gov`; superarlo devuelve `429` y puede provocar bloqueo temporal de IP
  [búsqueda]. `Settings.max_requests_per_second = 8.0` deja el margen adecuado.
- **Actualización**: los ZIP masivos se republican **cada noche hacia las 03:00 ET**
  [búsqueda].
- **Profundidad XBRL**: la obligación se implantó por fases entre **2009 y 2011**
  (grandes emisores primero). En la práctica: cobertura sólida del S&P 500 desde
  **2010-2011**; irregular en 2009; **nula antes**. Este es el límite duro del ángulo A
  gratuito.
- **Financial Statement Data Sets**: ZIP trimestrales desde **2009Q1** con `sub.txt`
  (una fila por filing: `adsh`, `cik`, `form`, `period`, `fy`, `fp`, **`filed`**,
  **`accepted`**), `num.txt` (valores), `pre.txt` (presentación), `tag.txt`. Es la vía
  más eficiente para una carga inicial completa: cuatro ficheros al año en vez de 500
  llamadas por empresa.

#### Trampas reales de EDGAR XBRL

1. **Heterogeneidad de etiquetas.** Ingresos aparece como `Revenues`,
   `RevenueFromContractWithCustomerExcludingAssessedTax`,
   `RevenueFromContractWithCustomerIncludingAssessedTax`, `SalesRevenueNet`,
   `SalesRevenueGoodsNet`… según empresa y año. Hace falta una **tabla de prioridad de
   conceptos por concepto lógico**, no un único tag. Lo mismo con `NetIncomeLoss` vs
   `ProfitLoss`, y con `NetCashProvidedByUsedInOperatingActivities` vs su variante
   `...ContinuingOperations`.
2. **Trimestres derivados.** Muchas empresas etiquetan el Q4 sólo dentro del 10-K anual;
   `Q4 = FY − (Q1+Q2+Q3)` hay que calcularlo, y ese cálculo **no debe cruzar
   reexpresiones** (los cuatro sumandos deben venir del mismo vintage).
3. **Duraciones.** Hay que filtrar por `(end − start)` ≈ 90 días para trimestral y
   ≈ 365 para anual; si no, se mezclan acumulados de nueve meses con trimestres.
4. **Ejercicios 52/53 semanas** (distribución, retail): `period_end` se mueve; hay que
   alinear por `fy`/`fp` y no por mes natural.
5. **Enmiendas (`10-K/A`, `10-Q/A`)**: son vintages nuevos legítimos; no deben
   descartarse, deben marcarse `is_restated=True`.
6. **`frames` no es PIT.** El endpoint `frames` selecciona *un* hecho por empresa y
   periodo con su propia heurística. Es cómodo para exploración, **peligroso para
   construir el panel**. Para el panel: `companyfacts` y filtrar por `filed`.

### 4.2 Proveedores comerciales de fundamentales

#### Financial Modeling Prep — `FMP_API_KEY`

- **Endpoints** (API "stable"; los `v3` legacy siguen activos):
  - `https://financialmodelingprep.com/stable/income-statement?symbol=AAPL&period=quarter`
  - `.../balance-sheet-statement`, `.../cash-flow-statement`
  - `.../key-metrics`, `.../ratios`, `.../financial-growth`
- **Ventaja decisiva sobre los demás baratos**: los estados incluyen **`filingDate` y
  `acceptedDate`**, es decir, un `available_at` utilizable directamente. Es la razón por
  la que FMP entra en la recomendación de 50 USD/mes pese a sus otras debilidades.
- **Límites** [búsqueda]: Starter 300 req/min · Premium 750 · Ultimate 3 000. Además un
  **tope de ancho de banda a 30 días**: Free 500 MB · Starter 20 GB · Premium 50 GB ·
  Ultimate 150 GB.
- **Coste**: **no ha sido posible confirmar las cifras exactas** [verificar]. Las
  búsquedas devuelven referencias dispersas: "Premium entre ~79 y ~199 USD/mes"
  [búsqueda] y "desde 99 USD/mes hasta 2 500 USD/año" [búsqueda]. El orden de magnitud
  del escalón Starter está en la banda **20-35 USD/mes** [prior] y el Premium en
  **70-100 USD/mes** [prior]. **Comprobar en `site.financialmodelingprep.com/pricing-plans`
  antes de contratar.**
- **PIT**: parcial. `acceptedDate` es real; los *valores* son los del filing, pero FMP no
  documenta si conserva vintages anteriores tras una reexpresión → hay que asumir que
  **no** y usarlo como fuente de fechas, no como fuente de vintages.

#### Sharadar (SF1) — vía Nasdaq Data Link o "Sharadar Direct"

- **Endpoint**: `https://data.nasdaq.com/api/v3/datatables/SHARADAR/SF1.json?ticker=AAPL&dimension=ARQ&api_key=`
- **Lo que lo hace especial**: la columna **`datekey`** = fecha en que la cifra estuvo
  disponible (fecha del filing), y las **dimensiones**:
  - `ARQ` / `ARY` / `ART` — *as reported*, trimestral / anual / trailing-12M;
  - `MRQ` / `MRY` / `MRT` — *most recent reported*, es decir **reexpresado**.
  Un backtest honesto usa **`AR*`**. Que la distinción esté explícita en el esquema es
  exactamente lo que este repo necesita.
- **Profundidad**: fundamentales desde **1998-2000** según fuente [búsqueda: "history
  reaching back to 1990" en un sitio, "since 1998" en otro → **[verificar]**],
  **16 000+ empresas incluidas las deslistadas**, ~150 indicadores.
- **Datasets hermanos**: `SEP` (precios EOD desde 1998), `TICKERS` (metadatos),
  **`SP500` (altas y bajas del índice desde 1957)**, `SF2` (insiders), `SF3`
  (participaciones institucionales 13F), `ACTIONS`, `EVENTS` (eventos de 8-K desde 1993).
- **Coste**: **no publicado en abierto**; Nasdaq Data Link tarifica por tipo de cuenta
  (Business / Academic / Personal) y exige contacto comercial para algunos feeds
  [búsqueda]. En julio de 2026 Sharadar lanzó **venta directa** en `sharadar.com`, con
  suscripción y baja autoservicio, pero **sin precios en los resultados de búsqueda**
  [búsqueda]. Referencia histórica de mercado: la banda de **100-200 USD/mes** para el
  bundle personal [prior, no confirmado].
- **Veredicto**: si el precio directo cae por debajo de ~150 USD/mes, **es la mejor
  compra individual de todo el proyecto**, porque resuelve de un golpe fundamentales PIT
  pre-2009, precios con deslistados, y constituyentes del S&P 500 desde 1957.

#### Compustat (WRDS) — referencia académica

- `comp.fundq` (trimestral) con **`RDQ`** = *report date of quarterly earnings*, y
  `comp.company`. La variante **Compustat Point-in-Time** (`comp.pit`) conserva vintages
  desde 1987, pero es un producto aparte y caro.
- Acceso sólo institucional; los contratos son anuales y negociados. Una referencia
  citada en foros: ~100 000 USD/año por CRSP + Compustat + fondos en una universidad
  [búsqueda, anecdótico]. No hay tarifa individual.

### 4.3 Resumen de la decisión de fundamentales

| Necesidad | Fuente | Coste |
|---|---|---|
| Panel PIT 2010→ con vintages y reexpresiones | **SEC EDGAR companyfacts + FSDS** | 0 |
| Fecha de disponibilidad rápida sin parsear XBRL | FMP `acceptedDate` | ~25 USD/mes |
| Panel PIT 1998-2009 | Sharadar SF1 `ARQ` | ver arriba |
| Panel PIT 1962-1998 | Compustat PIT (WRDS) | institucional |

---

## 5. (3) Calendario de resultados: fecha **y hora**

Es el dato del que depende `pit.tradable_date`, que el propio contrato califica como *"la
función más crítica del repo"*. Merece por tanto el análisis más cuidadoso.

### 5.1 La jerarquía de fuentes que propone este documento

```text
1. 8-K item 2.02 en EDGAR  -> acceptanceDateTime  (gratis, auditable, cota superior segura)
2. campo `hour`/`time` de Finnhub o FMP           (gratis/barato, confirma o corrige la sesión)
3. calendario prospectivo del proveedor           (para eventos futuros; is_estimated_date=True)
4. Wall Street Horizon                            (institucional; el estándar de oro)
```

La combinación 1+2 es la que se recomienda implementar en `data/estimates.py` y
`data/edgar.py`, con una regla de conciliación explícita y una columna de confianza.

### 5.2 SEC EDGAR 8-K item 2.02 — la base gratuita

- **Qué es**: el item 2.02 (*Results of Operations and Financial Condition*) existe desde
  **agosto de 2004** [búsqueda]; la inmensa mayoría de sus presentaciones son comunicados
  de resultados, con el comunicado como **exhibit EX-99.1** [búsqueda].
- **Cómo obtenerlo**:
  1. `https://data.sec.gov/submissions/CIK##########.json` → arrays paralelos
     `form`, `filingDate`, `acceptanceDateTime`, `accessionNumber`, `primaryDocument`,
     `items` (aquí aparece `"2.02"`). Los filings antiguos están en los ficheros
     adicionales que lista `filings.files[]`.
  2. Filtrar `form == "8-K"` y `"2.02" in items`.
  3. `announced_at` = `acceptanceDateTime` (ET → UTC).
  4. `session` según la regla de la §2.1.
  5. Opcionalmente descargar el EX-99.1 y extraer la hora del *dateline* del comunicado
     para afinar (`"...announced today after the close of market"`, `"/PRNewswire/ -- ...
     4:05 p.m. ET"`).
- **Coste**: 0. **Tasa**: 10 req/s. **Profundidad**: 2004→ para 2.02; 1994→ para el resto
  de EDGAR.
- **Limitaciones honestas**:
  - Es **retrospectivo**: no da fechas futuras. Para el calendario prospectivo hace falta
    otra fuente.
  - Un puñado de empresas no presenta 8-K 2.02 (raro entre las del S&P 500) o lo presenta
    fuera del día del comunicado.
  - Antes de 2004 hay que recurrir a proveedores o a Compustat `RDQ`.

### 5.3 Proveedores de calendario

| Proveedor | Endpoint | Hora BMO/AMC | Histórico | Coste |
|---|---|---|---|---|
| **Finnhub** | `https://finnhub.io/api/v1/calendar/earnings?from=&to=&token=` | **sí**, campo `hour` ∈ {`bmo`,`amc`,`dmh`} | prospectivo + retrospectivo, profundidad no documentada [verificar] | gratis (60 req/min) [búsqueda] |
| **FMP** | `https://financialmodelingprep.com/stable/earnings-calendar?from=&to=` y `.../earnings-calendar-confirmed` | sí (campo `time`); el endpoint *confirmed* trae hora exacta y es de pago | amplio, con fechas estimadas para el futuro | Starter+ |
| **EODHD** | `https://eodhd.com/api/calendar/earnings?from=&to=&api_token=` | parcial [verificar] | sí | 19,99+ |
| **Alpha Vantage** | `?function=EARNINGS_CALENDAR&horizon=3month` (CSV) | **no** | sólo 3/6/12 meses hacia delante | gratis |
| **Alpha Vantage** | `?function=EARNINGS&symbol=` | **no** | `reportedDate`, `reportedEPS`, `estimatedEPS`, `surprise` por trimestre, muchos años atrás | gratis (25 req/día) |
| **Nasdaq (web API)** | `https://api.nasdaq.com/api/calendar/earnings?date=YYYY-MM-DD` | sí (`time-pre-market` / `time-after-hours` / `time-not-supplied`) | sólo ventana reciente | gratis, no oficial |
| **Wall Street Horizon** | API / FTP | **sí, con estado `Confirmed` vs `Unconfirmed`** | ~9 500 empresas | institucional, no publicado |

**Wall Street Horizon** merece un párrafo aparte porque es el estándar frente al que se
mide todo lo demás: distingue explícitamente entre fecha **confirmada por la empresa**,
fecha **recogida pero tentativa** y fecha **estimada por sus analistas a partir del
patrón histórico** [búsqueda]. Ese tri-estado es exactamente lo que
`EarningsEvent.is_estimated_date` modela de forma binaria; conviene recordar que la
versión binaria pierde información.

### 5.4 Fiabilidad: lo que dice la literatura

Este es el punto en que la evidencia académica es más útil, porque el problema es viejo y
está bien documentado:

- **Compustat e IBES discrepan** en la fecha de anuncio; las discrepancias suelen ser
  pequeñas y **se concentran antes de diciembre de 1994** [búsqueda]. Para un backtest
  que arranque en 1996 el problema es menor, pero no nulo.
- **Compustat tiene dos defectos conocidos**: las fechas no siempre son correctas y **la
  hora no siempre está** [búsqueda].
- El **algoritmo de DellaVigna y Pollet (2009)** combina las fechas de Compustat e IBES
  para estimar la fecha real, y **desplaza el evento a la siguiente sesión cuando la hora
  de IBES es posterior al cierre** [búsqueda]. Es, esencialmente, `pit.tradable_date`
  publicado en el *Journal of Finance*; conviene implementarlo con ese nombre en el
  docstring.
- Trabajos recientes **recogen los timestamps de Wall Street Horizon precisamente porque
  desconfían de los de IBES**, y los usan para corregir Compustat cuando el anuncio es
  posterior a las 16:00 [búsqueda]. Es decir: **incluso los datos académicos de pago
  tienen el problema de la hora**, y la comunidad lo resuelve comprando una tercera
  fuente.

**Conclusión operativa.** No existe una fuente única fiable para la hora. La estrategia
correcta —y la que este documento recomienda— es **triangular** y **medir el
desacuerdo**:

- `data/estimates.py` debe devolver, junto a `session`, un campo de **procedencia** y un
  **indicador de acuerdo entre fuentes**;
- el backtest de eventos debe poder **excluir** los eventos con desacuerdo y reportar la
  sensibilidad del resultado a esa exclusión. Si una estrategia sólo funciona con los
  eventos ambiguos incluidos, la estrategia es un artefacto de fechas mal puestas.

### 5.5 Regla de negocio propuesta para `tradable_date`

```text
BMO  (acc_et < 09:30)                 -> tradable_date = misma sesión
AMC  (acc_et >= 16:00)                -> tradable_date = siguiente sesión
DMH  (09:30 <= acc_et < 16:00)        -> siguiente sesión  [decisión conservadora]
UNKNOWN                               -> siguiente sesión  [política del repo]
si el día calculado no es sesión      -> siguiente sesión hábil
```

La decisión de mandar `DMH` a la sesión siguiente es discutible (un anuncio a las 10:00
es negociable ese mismo día), pero un anuncio intradía suele coincidir con una
suspensión de cotización y con un precio de apertura ya contaminado; entrar al cierre del
día siguiente es lo defendible. **Debe ser un parámetro**, no una constante escondida.

---

## 6. (4) Consenso de analistas histórico — el dato difícil

Es el dato más caro y más escaso del proyecto, y conviene ser brutalmente claro:
**no existe ninguna fuente de vintages históricos de consenso por menos de varios cientos
de dólares al mes ni fuera del circuito académico.** Cualquier documento que sugiera lo
contrario está confundiendo *consenso actual* con *consenso histórico*.

### 6.1 Los dos datos que la gente confunde

| | Qué es | Para qué sirve en el repo | Disponibilidad |
|---|---|---|---|
| **Consenso en el anuncio** | el `eps_estimate` vigente el día del evento | SUE, sorpresa de ingresos, `EarningsEvent.eps_estimate` | **barato**: casi todos los proveedores publican una serie histórica de "estimated EPS" por trimestre |
| **Vintages de consenso** | la serie `(as_of, eps_mean, n_analysts, eps_std)` día a día antes del evento | momentum de revisiones, `analyst_revision_drift`, dispersión | **caro o inexistente** |

`types.EstimateSnapshot` está diseñado para el segundo caso (tiene `as_of`). Para el
primero basta con rellenar `EarningsEvent.eps_estimate`.

### 6.2 Opciones reales, de mejor a peor

#### (a) I/B/E/S vía WRDS — el estándar académico

- Ficheros: **`ibes.statsum_epsus`** (Summary History: consenso mensual con `STATPERS`,
  la fecha de corte del snapshot, típicamente el tercer jueves), **`ibes.det_epsus`**
  (Detail History: estimación individual por analista con su fecha), `ibes.actu_epsus`
  (actuales) y las variantes **sin ajustar** (`statsumu_`, `actu_`).
- **Por qué las variantes "unadjusted" importan**: I/B/E/S reajusta retroactivamente las
  estimaciones por splits, y ese reajuste introduce **error de redondeo** y
  **desalineación** cuando el split ocurre entre la fecha de la estimación y la del
  anuncio. Diether, Malloy y Scherbina (2002) documentaron el problema y desde entonces
  la práctica correcta es **usar los ficheros sin ajustar y aplicar el ajuste uno mismo**
  [búsqueda, múltiples fuentes concordantes]. Es un detalle técnico que arruina medidas de
  dispersión y de sorpresa si se ignora.
- **Coste**: WRDS sólo vende contratos **anuales institucionales**; no hay tarifa
  individual [búsqueda]. Para un investigador sin afiliación, la vía es una compra puntual
  al proveedor original, que WRDS reconoce como más barata que un contrato anual
  [búsqueda].
- **Veredicto**: si el usuario tiene **cualquier** afiliación universitaria, esta es la
  respuesta y todo lo demás de esta sección es irrelevante.

#### (b) Zacks — la alternativa comercial con historia larga

- **Profundidad publicada** [búsqueda]: consenso de EPS **anual desde 1979**, **EPS
  trimestral desde 1982**, recomendaciones desde **1985**, estimaciones de ventas y
  precios objetivo desde **2000**.
- Se vende **directo** (`zacksdata.com`) y a través de **WRDS**. Zacks comercializa
  explícitamente datos **point-in-time** para backtesting [búsqueda].
- **Coste**: no publicado; presupuesto a medida [búsqueda]. Banda institucional [prior].

#### (c) Refinitiv/LSEG, FactSet, S&P Capital IQ, Visible Alpha

- Terminales: LSEG Workspace **14 000-24 000 USD/año por puesto**; FactSet
  **12 000-18 000**; S&P Capital IQ Pro **13 000-25 000** [búsqueda]. Los **datafeeds**
  (que es lo que un backtest necesita, no el terminal) se tarifan aparte y no publican
  precio.
- **Visible Alpha** ofrece consenso a nivel de **línea de modelo** (no sólo EPS y
  ventas): segmentos, KPIs operativos. Es lo mejor que existe para sorpresas más finas
  que el EPS, y es puramente institucional.
- **Veredicto**: fuera de alcance. Se documentan para que la comparación sea honesta.

#### (d) Estimize — consenso *crowdsourced*

- Cubre desde ~2011. Su interés académico está establecido: el consenso de la multitud es
  **incrementalmente informativo respecto al de Wall Street**. Nunca ha sido un sustituto
  de I/B/E/S por cobertura, y su modelo de acceso/API ha cambiado varias veces
  [verificar]. Útil como **segunda opinión**, no como base.

#### (e) APIs baratas: qué dan realmente

Aquí es donde hay que ser preciso, porque el marketing es engañoso.

| Proveedor | Endpoint | Qué devuelve de verdad |
|---|---|---|
| FMP | `/stable/analyst-estimates?symbol=` | consenso **actual** para periodos futuros y una fila por periodo pasado, **sin `as_of`** |
| FMP | `/stable/earnings?symbol=` (antes `/earnings-surprises/`) | histórico de `eps_actual` y **`eps_estimated`** por trimestre → **esto sí es el consenso en el anuncio** |
| Finnhub | `/stock/eps-estimate?symbol=&freq=quarterly`, `/stock/revenue-estimate` | consenso actual con `numberAnalysts`; sin vintages |
| Finnhub | `/stock/earnings?symbol=` | sorpresas históricas con `estimate` → consenso en el anuncio |
| EODHD | `/api/calendar/earnings` | incluye `estimate` y `actual` por evento |
| Alpha Vantage | `?function=EARNINGS&symbol=` | `reportedDate`, `reportedEPS`, `estimatedEPS`, `surprise`, `surprisePercentage`, muchos años |

**Diagnóstico.** Las APIs baratas resuelven **(b) el consenso en el anuncio** de forma
razonable, y **no resuelven en absoluto (a) los vintages**. Y hay un matiz adicional
serio: cuando estos proveedores publican `estimatedEPS` para un trimestre de 2012, no
documentan **de qué fecha** es esa estimación ni **qué metodología** de consenso usan
(media/mediana, ventana de inclusión de analistas, tratamiento de partidas
extraordinarias). Dos proveedores dan números distintos para el mismo trimestre. Como
señala la propia literatura de proveedores, *"extraer un snapshot de estimaciones aunque
sea un día tarde contamina la capa de seguimiento con revisiones posteriores al evento"*
[búsqueda].

### 6.3 Las dos salidas honestas sin presupuesto

#### Salida 1 — SUE de series temporales (Foster) en lugar de SUE de analistas

El SUE original de **Foster, Olsen y Shevlin (1984)** no usa analistas en absoluto: usa
un modelo de **paseo aleatorio estacional con deriva** sobre el EPS trimestral,

```text
E[X_q] = X_{q-4} + delta_q ,  delta estimado sobre los ~8-20 trimestres previos
SUE_q  = (X_q - E[X_q]) / sigma(residuos históricos)
```

y **se calcula íntegramente con EDGAR, gratis y con PIT perfecto**. Livnat y Mendenhall
(2006) documentan que el PEAD medido con SUE basado en analistas es **mayor** que el
medido con SUE de series temporales, así que esta salida **degrada la señal pero no la
elimina**, y lo hace de forma cuantificable [búsqueda; magnitud exacta a verificar contra
el artículo original].

Consecuencia de diseño: `SurpriseBasis` ya contempla `SIGMA` y `PRICE`; conviene que el
factor SUE lleve un parámetro `expectation_model ∈ {"analyst", "seasonal_rw"}` y que el
informe reporte **ambos**. Comparar los dos es, además, un resultado interesante en sí.

#### Salida 2 — construir el panel de vintages uno mismo, desde hoy

Esta es la recomendación práctica más valiosa de toda la sección:

> **Arrancar hoy un recolector diario de consenso cuesta ~0 USD y, en doce meses, produce
> un panel de vintages que no se puede comprar por menos de cuatro cifras al año.**

Diseño mínimo (encaja directamente en `data/estimates.py`):

1. Cron diario tras el cierre.
2. Para cada ticker del universo vigente, pedir el consenso del trimestre en curso y del
   siguiente a **dos** proveedores (Finnhub gratis + FMP).
3. Escribir un `EstimateSnapshot` con `as_of = fecha de la captura` (no la que diga el
   proveedor), `source`, y **el payload crudo** para poder reprocesar.
4. Almacenamiento **append-only**, particionado por fecha, en parquet. Nunca reescribir
   una partición pasada — la misma disciplina que `SP500Universe.refresh()`.

Coste: dominado por el rate limit, no por el dinero. 500 tickers × 2 periodos × 2
proveedores = 2 000 llamadas/día; con Finnhub gratis (60 req/min) son ~35 minutos de
reloj [verificado por aritmética].

Limitación que hay que aceptar sin maquillaje: **este panel no tiene historia**. No sirve
para backtestear `analyst_revision_drift` en 2015. Sirve para validar la señal *hacia
delante* y para no tener que comprar nada dentro de un año.

---

## 7. (5) Cadenas de opciones históricas con IV y open interest

Las features del contrato que dependen de esto son cinco de las más discriminantes de
`PreEventFeatures`: `oi_buildup_calls/puts`, `put_call_volume_ratio`, `iv_skew_25delta`,
`vol_spread` (Cremers-Weinbaum) e `iv_term_slope`. Sin opciones, el ángulo B pierde su
mitad más informativa.

### 7.1 Qué hace falta exactamente

- **Por contrato y día**: `strike`, `expiration`, `right`, `bid`, `ask`, `volume`,
  **`open_interest`**, e **IV**.
- **Derivados imprescindibles**: superficie interpolada a **delta constante**
  (25Δ put vs 25Δ call para el skew) y a **madurez constante** (30/60/90 días para la
  pendiente temporal). Interpolar mal la superficie destruye la señal antes de que
  empiece.
- El **`vol_spread` de Cremers-Weinbaum** necesita IV de **call y put del mismo strike y
  vencimiento** (violación de paridad put-call), lo que exige la cadena completa, no
  agregados.

### 7.2 Fichas

#### ORATS — `ORATS_TOKEN`

- **Endpoints**: `https://api.orats.io/datav2/hist/strikes?ticker=&tradeDate=`,
  `/datav2/hist/summaries`, `/datav2/hist/cores`, `/datav2/hist/dailies`,
  `/datav2/hist/ivrank`. Token por query param.
- **Lo que lo diferencia**: ORATS no vende la cadena cruda, vende la **superficie
  suavizada y parametrizada** —skew, curtosis, forecast— comparable entre fechas y entre
  subyacentes [búsqueda]. Para las features del contrato eso es exactamente el trabajo que
  habría que hacer a mano.
- **Profundidad**: EOD desde **2007**; intradía de 1 minuto desde **agosto de 2020**
  [búsqueda].
- **Coste** [búsqueda]: **Delayed Data API 99 USD/mes** (20 000 peticiones/mes) ·
  **Live 199** (100 000/mes) · **Intraday 399**. Cuotas **mensuales**, no por minuto, lo
  que encaja mucho mejor con una carga masiva de backtest que un límite por minuto.
- **Aviso de cuota**: 20 000 req/mes ÷ 500 tickers ≈ **40 días de trading por ticker y
  mes**. Para reconstruir 5 años × 500 tickers hacen falta ~630 000 peticiones ⇒ **~31
  meses de cuota** en el plan de 99 USD. Es decir: el plan barato sirve para **estudiar
  ventanas de evento** (±30 sesiones alrededor de ~4 eventos/año × 500 tickers ≈ 120 000
  req/año) pero **no para reconstruir el panel completo**. Este cálculo [verificado por
  aritmética] es el que debe guiar el diseño del *crawler*: pedir **sólo las fechas dentro
  de ventanas de evento**.
- **Veredicto**: **la mejor compra del nivel de 300 USD/mes.**

#### Cboe DataShop / LiveVol

- **Option EOD Summary**: OHLC, volumen, VWAP y **open interest** por contrato
  [búsqueda].
- **Coste** [búsqueda]: suscripción EOD **500 USD/mes**; petición histórica ad-hoc
  **400 USD** por solicitud/mes; intradía 10 min **1 500/mes** o 18 000/año; intradía
  1 min **6 000/mes** o 72 000/año.
- **Descuento académico**: 50 % sobre precio estándar con mínimo de 500 USD para Option
  EOD Summary; y **750 USD el primer año** para el **Open-Close Volume Summary**
  [búsqueda].
- **El dataset que de verdad importa aquí es el Open-Close Volume Summary**, no el EOD
  Summary: desagrega el volumen en **compras/ventas de apertura y cierre por tipo de
  participante y tamaño de orden**. Es el dato con el que se construyó la literatura
  seminal sobre negociación informada en opciones (Pan y Poteshman, 2006). Para el ángulo
  B de este proyecto, **750 USD/año con descuento académico es probablemente el mejor
  euro gastado de todo el presupuesto** — pero sólo si hay afiliación académica.

#### OptionMetrics IvyDB US (vía WRDS)

- Todas las opciones sobre acciones e índices de EE. UU. **desde enero de 1996**, con
  cierre bid/ask, volumen, **open interest**, y una **superficie de volatilidad
  estandarizada de madurez constante interpolada por delta**, calculada a diario
  [búsqueda]. Incluye curvas de tipos y proyecciones de dividendos.
- Es la fuente canónica de la literatura académica de opciones.
- **Coste**: institucional vía WRDS, no publicado.

#### Polygon Options — `POLYGON_API_KEY`

- **Endpoints**: `https://api.polygon.io/v3/reference/options/contracts?underlying_ticker=&as_of=`,
  `https://api.polygon.io/v3/snapshot/options/{underlying}`,
  `https://api.polygon.io/v2/aggs/ticker/O:{occ_symbol}/range/1/day/{from}/{to}`,
  y flat files `s3://flatfiles/us_options_opra/day_aggs_v1/...`.
- **Options Starter 29 USD/mes**: todos los tickers de opciones de EE. UU., **llamadas
  ilimitadas**, **2 años de histórico**, 15 min de retraso, **greeks e IV**, **open
  interest diario**, agregados de minuto, flat files [búsqueda].
- **Limitaciones**: la IV la calcula Polygon (modelo propio, no documentado con el detalle
  de ORATS); no hay superficie suavizada; y **2 años** de histórico es poco para un
  estudio de eventos con potencia estadística (≈4 000 eventos del S&P 500 en 2 años, lo
  cual, siendo justos, **no es poco**).
- **Veredicto**: la opción **más barata que permite construir las features de opciones**.
  El parámetro **`as_of`** en `/v3/reference/options/contracts` es especialmente valioso:
  devuelve los contratos **que existían** en esa fecha, evitando el sesgo de mirar hoy la
  lista de contratos.

#### Databento OPRA

- **199 USD/mes** (plan Standard) con hasta **10 años** de histórico sobre las 18 bolsas
  de opciones, esquemas `trades`, `cbbo-1m`, `cbbo-1s`, `ohlcv`, `statistics`
  (incluye open interest) y `definition` [búsqueda].
- Sin IV calculada: hay que calcularla uno mismo (lo que, siendo rigurosos, es
  **preferible**: se controla el modelo de dividendos y de tipos).
- **Veredicto**: la mejor opción si se prioriza **profundidad histórica y microestructura**
  sobre analítica precocinada.

#### Tradier — `TRADIER_ACCESS_TOKEN`

- `https://api.tradier.com/v1/markets/options/chains?symbol=&expiration=&greeks=true`
  — greeks e IV **cortesía de ORATS** [búsqueda].
- **No ofrece snapshots históricos de cadena**: sólo `history` y `timesales` por símbolo
  OCC. Reconstruir una cadena histórica exigiría conocer de antemano los símbolos OCC y
  pedirlos uno a uno.
- **Veredicto**: **inútil para backtest**, útil para operar en vivo. Debe registrarse en
  `ProviderRegistry` sólo como proveedor de *snapshot actual*.

#### EODHD opciones

- Marketplace `https://eodhd.com/api/mp/unicornbay/options/contracts` con IV y greeks;
  histórico desde ~2020 [verificar]; add-on de ~29,99 USD/mes [verificar].

### 7.3 Trampa PIT específica de las opciones

El **open interest** que publica la OCC corresponde al cierre de la sesión `T` pero **se
disemina la mañana de `T+1`**. Un feature `oi_buildup` que use el OI de `T` como conocido
en `T` introduce un día de look-ahead. Como las ventanas pre-evento son cortas (5-20
sesiones), un día es un error relativo grande. **`data/options.py` debe fijar
`available_at = apertura de T+1` para el open interest**, y `available_at = cierre de T`
para volumen e IV.

---

## 8. (6) Short interest de FINRA

### 8.1 El calendario, que es la parte que importa

- FINRA exige reportar posiciones cortas **dos veces al mes**: a la fecha de liquidación
  del **día 15** (o la anterior si no es hábil) y a la del **último día hábil** del mes
  [búsqueda].
- Las firmas reportan **antes de las 18:00 ET del segundo día hábil** posterior a la
  fecha de liquidación [búsqueda].
- Los datos se **publican el 7º día hábil** posterior a la fecha de liquidación
  [búsqueda; **[verificar]**, porque FINRA ha ajustado este calendario en el pasado].

**Por tanto**: `available_at` = fecha de diseminación, **no** `settlementDate`. La brecha
es de **9-11 días naturales**. Un `short_interest_delta` indexado por fecha de
liquidación adelanta información más de una semana; en una ventana pre-evento de 10
sesiones, eso es catastrófico.

**Implementación recomendada**: `data/shortinterest.py` no debe *calcular* la fecha de
diseminación con una regla de "7 días hábiles", sino **cargar la tabla oficial de
`Short Interest Reporting Deadlines` de FINRA**, que publica las fechas exactas. Calcular
festivos bursátiles a mano es una fuente de errores de un día, justo el error que el
contrato prohíbe.

### 8.2 Endpoints

| Qué | URL |
|---|---|
| Catálogo y descarga interactiva | `https://www.finra.org/finra-data/browse-catalog/equity-short-interest/data` |
| API de datos (Query API) | `https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest` |
| API alternativa observada | `https://api.dapi.finra.org/api/EquityShortInterest/GetESI?settlementDate=YYYY-MM-DD` [búsqueda] |
| Token OAuth2 | `https://ews.fip.finra.org/fip/rest/ews/oauth2/access_token?grant_type=client_credentials` [verificar] |

- **Autenticación**: la Query API requiere credenciales del **FINRA Developer Center**
  (registro **gratuito**, `client_id`/`client_secret` → token OAuth2 Bearer). El acceso
  anónimo tiene cuotas mucho menores [verificar].
- **Coste**: **0**.
- **Profundidad**: **sólo un año rodante en línea**; lo anterior está en archivos
  descargables [búsqueda]. **Implicación operativa importante: hay que archivar cada
  publicación quincenal en cuanto sale**, o se pierde. Esto encaja con la política
  *append-only* del repo.
- **PIT**: correcto si se usa la fecha de diseminación.

### 8.3 El fichero diario de volumen en corto (Reg SHO) — infravalorado y gratis

- **URL**: `http://regsho.finra.org/CNMSshvol{YYYYMMDD}.txt` (consolidado NMS), y por
  facilidad: `FNSQshvol`, `FNYXshvol`, `FNRAshvol`.
- **Formato**: `Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market`.
- **Frecuencia**: **diaria**, disponible el mismo día tras el cierre o T+1 [verificar].
- **Coste**: 0. **Auth**: ninguna.
- **Profundidad**: histórico descargable de años; los últimos 365 días también en rejilla
  interactiva [búsqueda].
- **Advertencia crítica de interpretación** [búsqueda]: estos ficheros **no están
  consolidados con las bolsas**; cubren únicamente lo reportado a los TRF/ADF/ORF de
  FINRA, es decir, **el mercado off-exchange**. Y, sobre todo, **volumen en corto ≠
  interés corto**: buena parte del volumen en corto es *hedging* de creadores de mercado
  (Blocher y Ringgenberg documentan que interpretar el short volume como sentimiento
  bajista es un error).
- **Por qué merece la pena de todos modos**: es la **única serie diaria y gratuita de
  flujo direccional** disponible. Con `ShortVolume / TotalVolume` se obtiene un ratio
  diario; y con `TotalVolume` frente al volumen consolidado del proveedor de precios se
  obtiene, gratis y con **un día de retraso**, la cuota off-exchange. Ver §9.3.

### 8.4 Alternativas comerciales (préstamo de valores)

- **Ortex**: modela el interés corto a partir de datos de préstamo de valores y lo
  actualiza **intradía**, junto con coste de préstamo, disponibilidad y utilización
  [búsqueda]. Endpoint documentado: `docs.ortex.com` (`index_short_ctb_list`, etc.).
  Precio no publicado en la búsqueda; orientado a retail/semi-profesional [búsqueda].
- **S3 Partners**: equivalente institucional [búsqueda].
- **Valor añadido real**: el **coste de préstamo** (`fee`) es una variable con contenido
  informativo propio y, además, **entra directamente en `CostModel`** (el contrato exige
  "coste de préstamo para cortos", no un 0,05 % fijo). Sin ella, el coste de los cortos es
  un número inventado, que es justo lo que el repo prohíbe.
- **Sustituto gratuito parcial**: la lista de valores *hard-to-borrow* y las tasas
  indicativas de Interactive Brokers son públicas vía FTP [verificar]; sirven para
  calibrar tramos, no para una serie histórica.

---

## 9. (7) Volumen off-exchange y ATS de FINRA

### 9.1 Qué publica FINRA

Dos familias, ambas **semanales por valor**:

- **ATS** (*dark pools*): volumen y número de operaciones por **MPID de cada ATS**.
- **Non-ATS OTC**: volumen ejecutado fuera de bolsa y fuera de ATS — esencialmente
  internalizadores y mayoristas.

Y una tercera, **ATS Blocks**, con las operaciones de bloque, publicada con **un mes de
retraso, el primer lunes** [búsqueda].

### 9.2 Los retrasos de publicación, que son el meollo

[búsqueda, coherente en varias fuentes de FINRA]

| Categoría | Valores | Retraso de publicación |
|---|---|---|
| ATS | NMS **Tier 1** (grandes; todo el S&P 500 lo es) | **2 semanas** |
| ATS | resto de NMS y OTC | 4 semanas |
| Non-ATS | NMS Tier 1 | **2 semanas** |
| Non-ATS | resto de NMS y OTC | 4 semanas |
| ATS Blocks | todos | ~1 mes, primer lunes |

**Consecuencia demoledora para `PreEventFeatures.off_exchange_share_delta`**: para un
evento en `T`, el dato ATS más reciente **conocible** cubre la semana que terminó hacia
`T − 14` días naturales o antes. La feature, tal y como suena, **no se puede calcular con
datos ATS semanales sin look-ahead**.

Hay dos formas correctas de resolverlo y una incorrecta:

- ✅ **Redefinir la feature con la latencia explícita**: `off_exchange_share_delta_lag14`,
  comparando la última semana **publicada** contra su media de las 8 semanas previas. Es
  una señal de régimen, no de la ventana inmediata. Sigue siendo interesante (la
  acumulación institucional empieza semanas antes), pero hay que llamarla por su nombre.
- ✅ **Sustituirla por el proxy diario del §9.3.**
- ❌ Indexar el dato ATS por `weekStartDate`. Introduce **14 días** de look-ahead. Es el
  error que el `asof_join(lag_days=...)` del contrato existe para prevenir; el adaptador
  debe rellenar `available_at` con la **fecha de publicación**, y dejar que `asof_join`
  haga el resto.

### 9.3 El proxy diario que sí funciona, y es gratis

El fichero **Reg SHO diario** (§8.3) contiene `TotalVolume` = **volumen reportado a
FINRA** = volumen off-exchange del símbolo ese día. Combinándolo con el volumen
consolidado del proveedor de precios:

```text
off_exchange_share(t) = FINRA_TotalVolume(t) / Consolidated_Volume(t)
available_at          = T+1  (frente a las 2 semanas del dato ATS)
```

Esto da una serie **diaria**, **gratuita** y con **un día de latencia** de la fracción de
negociación que ocurre fuera de bolsa — que es, conceptualmente, lo que
`off_exchange_share_delta` quiere medir. Pierde el desglose por ATS individual (no se
puede ver *qué* dark pool), pero gana dos semanas de actualidad, y en una ventana
pre-evento de 10 sesiones esa diferencia lo es todo.

**Recomendación**: implementar `off_exchange_share_delta` sobre Reg SHO diario como
definición primaria, y el desglose ATS semanal como feature secundaria y explícitamente
retrasada.

### 9.4 Endpoints

| Qué | URL |
|---|---|
| Portal | `https://www.finra.org/filing-reporting/otc-transparency-data` |
| Query API (resumen semanal) | `https://api.finra.org/data/group/otcMarket/name/weeklySummary` |
| Query API (detalle semanal) | `https://api.finra.org/data/group/otcMarket/name/weeklyDetail` [verificar] |
| Query API (bloques) | `https://api.finra.org/data/group/otcMarket/name/blocksSummary` [verificar] |
| Especificación de descarga | `https://www.finra.org/sites/default/files/OTC-Transparency-Data-File-Download-API-v04.pdf` |

Auth: igual que short interest (credenciales gratuitas del Developer Center). Coste: 0.

---

## 10. (8) SEC EDGAR — resumen consolidado

Ya cubierto en §4.1 (XBRL), §5.2 (8-K 2.02) y §2.1 (fechas). Queda **Form 4**.

### 10.1 Form 4 (insiders) para `insider_net_buy_form4`

- **Descubrimiento**: en `submissions/CIK##########.json` filtrando `form == "4"`; o por
  el índice diario; o por el feed Atom
  `https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=&type=4&output=atom`.
- **Documento**: cada Form 4 tiene un **XML primario de ownership** con
  `nonDerivativeTransaction` y `derivativeTransaction`, cada uno con `transactionDate`,
  `transactionCode`, `transactionShares`, `transactionPricePerShare`,
  `sharesOwnedFollowingTransaction` y **notas al pie** [búsqueda].
- **Códigos relevantes** [búsqueda]: `P` compra en mercado abierto y `S` venta son **los
  únicos con contenido informativo real**; `A` (concesión), `M` (ejercicio de opción),
  `F` (retención para impuestos) y `D` son ruido mecánico de la retribución. Un
  `insider_net_buy` que sume todos los códigos mide el calendario de *vesting*, no
  convicción.
- **Filtro imprescindible**: las ventas bajo **plan 10b5-1** están preprogramadas y no
  informan sobre información privada. El plan se indica en las notas al pie (texto libre)
  y, desde las enmiendas de 2022-2023, en **casillas específicas del formulario**
  [verificar]. Un detector de trading informado que no filtre 10b5-1 está midiendo
  principalmente diversificación patrimonial.
- **PIT**: excelente. Plazo legal de presentación: **2 días hábiles** desde la
  transacción; y los Forms 3/4/5 **conservan la fecha del día hasta las 22:00 ET**
  [búsqueda], así que `acceptanceDateTime` es de nuevo el campo correcto.
- **Coste**: 0.
- Alternativas de pago que ahorran el parseo: EODHD `/api/insider-transactions`, Finnhub
  `/stock/insider-transactions`, Sharadar `SF2`.

---

## 11. (9) Transcripciones de earnings calls

### 11.1 Papel real en este proyecto

**Ninguna feature del contrato de arquitectura las requiere.** Su valor es potencial
(tono, incertidumbre, evasivas en el Q&A, cambios de *guidance* cualitativos) y su timing
las hace **inútiles para el ángulo pre-evento**: la llamada ocurre *después* del
comunicado, típicamente 30-120 minutos después. `available_at` ≥ `announced_at`.

Donde sí encajan es en el **post-evento**: PEAD condicionado al tono de la llamada, y
`cambios de guidance` (que el contrato sí pide en §3.4 de la arquitectura). Para eso, sin
embargo, el 8-K EX-99.1 suele bastar y es gratis.

**Prioridad recomendada: baja.** No debe consumir presupuesto en los niveles 0 y 1.

### 11.2 Opciones

| Proveedor | Endpoint / acceso | Profundidad | Coste |
|---|---|---|---|
| **FMP** | `https://financialmodelingprep.com/stable/earning-call-transcript?symbol=&year=&quarter=` | amplia, años atrás | incluido en planes de pago |
| **API Ninjas** | `https://api.api-ninjas.com/v1/earningscalltranscript?ticker=&year=&quarter=` | **Developer: últimos 5 años**; archivo completo **desde 2005** en Business/Professional [búsqueda] | escalonado, no publicado en la búsqueda |
| **EarningsCall** | SDK oficial Python/JS; segmenta **prepared remarks vs Q&A**, mapea locutores, marcas de tiempo por palabra, audio [búsqueda] | no confirmada | "startup-friendly", no publicado |
| **Quartr** | enterprise, 65 mercados [búsqueda] | la más amplia | institucional |
| **Seeking Alpha** | web, **sin API oficial** | amplia | el raspado infringe sus condiciones |
| **SEC EDGAR** | algunas empresas adjuntan la transcripción como exhibit del 8-K | irregular | 0 |

**Advertencia de licencia** [búsqueda]: las transcripciones suelen venir con
restricciones de redistribución. Para investigación privada no hay problema; para
publicar resultados que incluyan extractos, sí. Conviene registrar la licencia de cada
proveedor junto al dato.

---

## 12. (10) Listas históricas de constituyentes del S&P 500

### 12.1 Lo que ya hay en el repo

| Fichero | Filas | Contenido |
|---|---|---|
| `data/seed/sp500_constituents.csv` | 503 | `Symbol, Security, GICS Sector, GICS Sub-Industry, HQ, Date added, CIK, Founded` |
| `data/seed/sp500_historical_components.csv` | 3 482 | `date, tickers` (cadena separada por comas) |

Ambos derivan del linaje **Clenow (*Trading Evolved*) → `github.com/fja05680/sp500` →
reconciliación con Wikipedia** [búsqueda]. Licencia permisiva, coste 0.

### 12.2 Auditoría ejecutada sobre la semilla — **[verificado]**

Todas las cifras siguientes se obtuvieron ejecutando `pandas` sobre los ficheros del
repo. No son estimaciones.

| Métrica | Valor |
|---|---|
| Rango temporal | **1996-01-02 → 2025-08-23** |
| Filas | 3 482 |
| Días hábiles en el rango | 7 734 |
| **Cobertura de días hábiles** | **45,0 %** → el panel **exige forward-fill** |
| Filas consecutivas con conjunto de miembros **idéntico** | **2 814 de 3 481** |
| Conjuntos de miembros **distintos** | **663** |
| Tamaño del índice: mín / máx / último | 442 / 505 / **503** |
| Símbolos únicos que **alguna vez** pertenecieron (normalizados) | **1 126** |
| Constituyentes actuales | 503 |
| **Símbolos históricos que ya no están** | **644** |
| Símbolos actuales **ausentes** del histórico | **21** |
| Símbolos con sufijo `Q` (quiebra) | 27 (`ENRNQ`, `EKDKQ`, `AAMRQ`, `DPHIQ`, `CPNLQ`…) |
| Altas medias por año | **23,6** |
| CIK nulos en el fichero de constituyentes | **0** |

### 12.3 Tres problemas concretos, verificados, que el módulo `universe` debe resolver

**(a) La semilla histórica está desfasada ~11 meses.** El fichero histórico termina el
**2025-08-23**, pero el fichero de constituyentes contiene **21 empresas con `Date added`
posterior** a esa fecha [verificado]:

```text
IBKR 2025-08-28 · APP/EME/HOOD 2025-09-22 · Q 2025-11-03 · SNDK 2025-11-28
ARES 2025-12-11 · CVNA/FIX/CRH 2025-12-22 · CIEN 2026-02-09
COHR/ECHO/LITE/VRT 2026-03-23 · CASY 2026-04-09 · VEEV 2026-05-07
FDXF 2026-06-01 · FLEX/MRVL 2026-06-22 · HONA 2026-06-29
```

`SP500Universe.refresh()` no es un lujo: sin ella el universo está mal desde
2025-08-24 en adelante. Y su naturaleza *append-only* es correcta precisamente porque el
tramo 1996-2025 no debe reescribirse cuando se añada el tramo 2025-2026.

**(b) Los cambios de ticker se disfrazan de alta+baja.** Comparando el último snapshot
histórico con la lista actual [verificado]:

- `BK` (Bank of New York Mellon) está en 2025-08-23 y **no** en la lista actual;
  `BNY` está en la actual y **no** en el histórico. Es **la misma empresa**.
- Idéntico caso: `MMC` (Marsh & McLennan) → `MRSH`.

Un `universe` ingenuo registraría cuatro eventos (dos bajas, dos altas) donde hay **cero**
cambios de composición, y el backtest cerraría y reabriría posiciones pagando costes
ficticios. **La normalización por ticker no basta: hace falta anclar por CIK.**

**(c) El ancla correcta es el CIK, y es gratis.** El fichero de constituyentes trae CIK
para las 503 empresas, **sin un solo nulo** [verificado]. Para el histórico:

- `https://www.sec.gov/files/company_tickers.json` da el mapa ticker→CIK **actual**;
- `https://data.sec.gov/submissions/CIK##########.json` incluye **`formerNames[]`** con
  las denominaciones anteriores **y sus fechas** → permite reconstruir la deriva de
  nombres, y en muchos casos inferir la de tickers.

Esto convierte `UniverseProvider.cik_for(t, on=date)` en una función implementable con
fuentes gratuitas, que es lo que el contrato pide.

### 12.4 Alternativas y complementos

| Fuente | Qué aporta | Coste |
|---|---|---|
| `fja05680/sp500` (origen de la semilla) | cambios desde 1996; se actualiza cada pocos meses desde Wikipedia [búsqueda] | 0 |
| **Sharadar `SP500`** | **altas y bajas desde 1957** con acción, fecha, ticker, nombre y CIK | de pago |
| **iShares IVV** (holdings diarios CSV) | **pesos reales** del índice, no sólo pertenencia; desde ~2006 | 0 |
| CRSP / Compustat `IDXCST_HIS` (WRDS) | pertenencia académica auditada | institucional |
| S&P Dow Jones Indices | la fuente oficial | institucional, caro |

El truco de **iShares IVV** merece énfasis: el ETF replica el índice con desviación
mínima, publica sus posiciones **a diario y gratis**, y con ello se obtienen **pesos** —
que la semilla no tiene y que hacen falta para construir la cartera de referencia del
modelo de mercado en `abnormal_returns`. Su límite: empieza en la segunda mitad de los
2000 y es T+1.

### 12.5 Limitaciones de la semilla que hay que aceptar

1. **Reconstruida desde Wikipedia** → hereda sus errores y omisiones, especialmente en
   los años 90.
2. **Sin pesos ni float**: no permite reproducir el índice, sólo la pertenencia.
3. **Fechas efectivas aproximadas**: puede haber desfases de uno o dos días entre la
   fecha real de efectividad del cambio y la registrada.
4. **Tickers de la época mezclados con tickers actuales** en algunos vintages (el
   problema (b) de arriba).
5. **45 % de cobertura de días hábiles** [verificado] → obligatorio `reindex` + `ffill`
   antes de construir `membership_panel`, y **prohibido** interpolar hacia atrás.

---

## 13. Matriz comparativa

### 13.1 Cobertura por proveedor

Leyenda: ✔ sólido · ~ parcial o con reservas · ✘ no ofrece · **PIT** = expone un
`available_at` utilizable.

| Proveedor | OHLCV d | Intradía | Fund. | PIT fund. | Calend. | Hora BMO/AMC | Consenso @anuncio | Vintages | Opciones | SI/ATS | Insiders | Deslistados | USD/mes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SEC EDGAR | ✘ | ✘ | ✔ | **✔✔** | ~ (retro) | ✔ | ✘ | ✘ | ✘ | ✘ | ✔ | ✔ | **0** |
| FINRA | ✘ | ✘ | ✘ | — | ✘ | ✘ | ✘ | ✘ | ✘ | **✔✔** | ✘ | ✔ | **0** |
| yfinance | ~ | ~ | ~ | ✘ | ~ | ~ | ~ | ✘ | ~ | ✘ | ✘ | ✘ | **0** |
| Stooq | ✔ | ✘ | ✘ | ✘ | ✘ | ✘ | ✘ | ✘ | ✘ | ✘ | ✘ | ~ | **0** |
| Tiingo | **✔✔** | ~ (IEX) | ~ (add-on) | ✘ | ✘ | ✘ | ✘ | ✘ | ✘ | ✘ | ✘ | ✔ | 30-50 |
| Polygon | ✔ | **✔✔** | ~ | ✘ | ~ | ✘ | ✘ | ✘ | **✔** | ✘ | ✘ | **✔✔** | 29-199 |
| FMP | ✔ | ✔ | ✔ | **✔** | **✔** | ✔ | **✔** | ✘ | ~ | ✘ | ✔ | ~ | ~20-100 |
| EODHD | ✔ | ✔ | ✔ | ~ | ✔ | ~ | ✔ | ✘ | ~ | ✘ | ✔ | ✔ | 20-100 |
| Finnhub | ✔ | ✔ | ✔ | ✘ | **✔** | **✔** | ✔ | ✘ | ~ | ✘ | ✔ | ~ | 0-200 |
| Alpha Vantage | ~ | ~ | ✔ | ✘ | ~ | ✘ | **✔** | ✘ | ~ | ✘ | ✘ | ✘ | 0-250 |
| Alpaca | ✔ | ✔ | ✘ | — | ✘ | ✘ | ✘ | ✘ | ✔ | ✘ | ✘ | ~ | 0-99 |
| Tradier | ~ | ~ | ✘ | — | ✘ | ✘ | ✘ | ✘ | ~ (vivo) | ✘ | ✘ | ✘ | ~10 |
| ORATS | ✘ | ✘ | ✘ | — | ✘ | ✘ | ✘ | ✘ | **✔✔** | ✘ | ✘ | — | 99-399 |
| Cboe DataShop | ✘ | ✘ | ✘ | — | ✘ | ✘ | ✘ | ✘ | **✔✔** | ~ | ✘ | — | 400-6 000 |
| Databento | ✔ | **✔✔** | ✘ | — | ✘ | ✘ | ✘ | ✘ | **✔✔** | ✘ | ✘ | ✔ | 199+ |
| Nasdaq DL / Sharadar | ✔ | ✘ | **✔✔** | **✔✔** | ~ | ✘ | ✘ | ✘ | ✘ | ✘ | ✔ | **✔✔** | no publicado |
| WRDS (IBES+CS+CRSP+OM) | ✔ | ✘ | **✔✔** | **✔✔** | ✔ | ~ | **✔✔** | **✔✔** | **✔✔** | ✘ | ✔ | **✔✔** | institucional |
| Refinitiv/FactSet/CapIQ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | **✔✔** | ✔ | ~ | ✔ | ✔ | 12-25 k/año |

### 13.2 Coste por dato, ordenado por USD por unidad de valor de investigación

| Compra | USD/mes | Qué desbloquea que antes era imposible |
|---|---|---|
| SEC EDGAR + FINRA | **0** | fundamentales PIT, calendario con hora, insiders, short interest, off-exchange |
| Semilla + `fja05680` | **0** | universo sin sesgo de supervivencia |
| Tiingo Power/Commercial | 30-50 | panel de precios 1996→ con factores de ajuste separados |
| FMP Starter | ~20-30 [verificar] | `acceptedDate` en fundamentales + consenso en el anuncio + calendario con hora |
| Polygon Stocks Starter | 29 | minuto para el proxy de order imbalance + lista de deslistados + flat files |
| **ORATS Delayed** | **99** | **todas las features de opciones del contrato, superficie ya suavizada, desde 2007** |
| Polygon Options Starter | 29 | mismas features con 2 años y superficie propia |
| Cboe Open-Close (académico) | ~62 (750/año) | **volumen de apertura comprador/vendedor: la mejor señal de trading informado en opciones** |
| Databento OPRA | 199 | 10 años de opciones + microestructura real |
| WRDS | institucional | vintages de consenso: **lo único que no se puede sustituir** |

---

## 14. Recomendación de stack por nivel de coste

### 14.1 Nivel 0 — **0 USD/mes**

| Necesidad | Proveedor | Notas |
|---|---|---|
| Universo | semilla del repo + `fja05680/sp500` + `refresh()` | resolver renombres por CIK |
| Precios diarios | **Stooq** primario + **yfinance** secundario | discrepancia > umbral ⇒ `DataQualityError` |
| Splits/dividendos, deslistados | **Polygon Basic** (gratis, 5 req/min) | sólo referencia, no barras |
| Fundamentales | **EDGAR `companyfacts` + Financial Statement Data Sets** | PIT real con vintages, 2010→ |
| Calendario + hora | **EDGAR 8-K 2.02 `acceptanceDateTime`** + `hour` de **Finnhub** gratis | triangulación obligatoria |
| Sorpresa | **SUE de series temporales (Foster)** sobre EDGAR | + `EARNINGS` de Alpha Vantage como carga inicial lenta |
| Short interest | **FINRA** quincenal (gratis, registro) | indexar por fecha de diseminación |
| Flujo diario | **FINRA Reg SHO `CNMSshvol`** | short ratio + cuota off-exchange, T+1 |
| ATS | **FINRA OTC Transparency** | retraso de 2 semanas, explícito |
| Insiders | **EDGAR Form 4 XML** | filtrar códigos `P`/`S`, excluir 10b5-1 |
| Opciones | **ninguno** | — |

**Aritmética del cuello de botella de Alpha Vantage** [verificado]: 25 req/día ⇒ 500
tickers en **20 días naturales**. Es viable **una vez**, como carga inicial de
`estimatedEPS` histórico; no es viable para refrescar.

**Qué SÍ se puede investigar en el nivel 0**

- Ángulo A **completo** desde ~2010-2011: valor, calidad, crecimiento, accruals de Sloan,
  F-Score de Piotroski, tendencia de márgenes, apalancamiento, FCF/earnings yield —
  **todos con `available_at` genuino**, que es más de lo que consiguen muchos estudios
  publicados.
- **PEAD con SUE de series temporales**, con sorpresa de ingresos análoga.
- Estudio de eventos completo: `event_windows`, AR/CAR con modelo de mercado, FF3 o media.
- `PreEventFeatures` **parcial**: `volume_runup`, `turnover_zscore`,
  `abnormal_volume_5/10/20d`, `order_imbalance_proxy_clv`, `short_interest_delta`,
  `off_exchange_share_delta` (vía Reg SHO), `insider_net_buy_form4`,
  `pre_event_car_5/10/20d`. **Nueve de las quince** features del contrato.
- Toda la maquinaria de `stats`: IC, Newey-West, CV purgada, Sharpe deflactado,
  Benjamini-Hochberg, bootstrap estacionario.
- Todo `backtest`, `signals` y `reports`.

**Qué NO se puede investigar en el nivel 0**

- Fundamentales anteriores a 2009 ⇒ **no hay crisis de 2008 en la muestra**. Esto no es
  un detalle: un factor de calidad que nunca ha visto una recesión severa está
  insuficientemente validado.
- **SUE frente a consenso de analistas** (sólo frente a modelo). La literatura indica que
  el PEAD medido así es menor [búsqueda] ⇒ se subestimará el efecto.
- **Cero features de opciones**: `vol_spread`, `iv_skew_25delta`, `iv_term_slope`,
  `oi_buildup_calls/puts`, `put_call_volume_ratio`. Es **la pérdida más grave**, porque
  son las features con más respaldo académico para detectar trading informado.
- `analyst_revision_drift`.
- Intradía ⇒ el `EventBacktest` no puede modelar el gap overnight con precios reales de
  apertura fiables; y `order_imbalance` queda en el proxy más burdo.
- Cualquier estimación honesta del coste de préstamo para los cortos.

**Veredicto**: el nivel 0 **no es un juguete**. Permite un trabajo publicable sobre el
ángulo A y sobre la mitad del ángulo B. Es el nivel correcto para las primeras semanas.

---

### 14.2 Nivel 1 — **~50 USD/mes**

**Cesta recomendada** (≈ **52 USD/mes**, sujeta a confirmar el precio de FMP):

| Componente | USD/mes | Qué resuelve |
|---|---|---|
| **Tiingo** Power (o Commercial si hay uso comercial) | 30 (o 50) | precios diarios **1996→**, `divCash` + `splitFactor` ⇒ **ajuste PIT reconstruible**; universo con deslistados |
| **FMP Starter** | ~22 [verificar] | `filingDate`/`acceptedDate` en fundamentales, **calendario con hora**, **`estimatedEPS` histórico por trimestre** |
| Todo el nivel 0 | 0 | sigue siendo la columna vertebral |

**Variante** si se prefiere intradía a profundidad histórica: **Polygon Stocks Starter
(29) + FMP Starter (~22) ≈ 51 USD/mes**. Se gana el minuto (⇒ `order_imbalance_proxy`
por regla de tick) y los deslistados de Polygon; se pierde el tramo 1996-2020 de precios
[el histórico exacto del plan Starter está **[verificar]**].

**Qué añade el nivel 1 sobre el 0**

- **SUE de analistas de verdad**, para todos los eventos históricos que FMP cubra ⇒ se
  puede comparar `seasonal_rw` vs `analyst` y medir la diferencia en lugar de citarla.
- Calendario con hora **cruzado en dos fuentes independientes** (EDGAR + FMP), con
  métrica de desacuerdo ⇒ se puede *medir* la fiabilidad en vez de suponerla.
- Panel de precios limpio hasta 1996, sin sesgo de supervivencia y con ajuste PIT.
- `available_at` de fundamentales sin escribir un parser XBRL completo (aunque conviene
  escribirlo igual: EDGAR sigue siendo la fuente de verdad y FMP la de conveniencia).

**Qué sigue sin poder hacerse**

- **Todo lo de opciones.** Sin cambios respecto al nivel 0.
- **Vintages de consenso** ⇒ `analyst_revision_drift` sigue sin histórico. Aquí es donde
  hay que **arrancar el recolector propio** (§6.3, salida 2): el nivel 1 es exactamente
  el momento correcto para empezar a acumular, porque ya hay dos proveedores.
- Microestructura real (quotes).
- Coste de préstamo histórico.

---

### 14.3 Nivel 2 — **~300 USD/mes**

**Cesta recomendada** (≈ **287 USD/mes**):

| Componente | USD/mes | Qué resuelve |
|---|---|---|
| **ORATS Delayed Data API** | **99** | superficie IV suavizada **desde 2007**: skew 25Δ, pendiente temporal, `vol_spread`, OI |
| **Polygon Stocks Developer** | 79 [verificar] | ~10 años de barras de minuto + flat files S3 + deslistados |
| **FMP Premium** | ~79 [verificar] | consenso, calendario confirmado, transcripciones, mayor cuota |
| **Tiingo** Power | 30 | tramo 1996-2015 que Polygon Developer no alcanza |
| Todo el nivel 0 | 0 | EDGAR y FINRA no se sustituyen nunca |

**Restricción de cuota que condiciona el diseño** [verificado por aritmética]: ORATS
Delayed da 20 000 peticiones/mes. Reconstruir 5 años × 500 tickers de cadenas diarias
requiere ~630 000 peticiones ⇒ **~31 meses**. La consecuencia arquitectónica es directa:
el crawler de opciones **debe** estar dirigido por eventos, no por calendario. Pidiendo
sólo `[T−30, T+10]` alrededor de ~4 eventos/año × 500 tickers ⇒ ~80 000 peticiones/año,
que sí caben (4 meses de cuota, o menos si se prioriza por sector). Esto es exactamente
lo que necesita el ángulo B; el ángulo A no usa opciones.

**Qué añade el nivel 2**

- **Las quince features de `PreEventFeatures`**, ahora sí completas salvo
  `analyst_revision_drift`.
- Las tres señales de opciones con mejor respaldo académico para detectar información
  privada: violación de paridad put-call (`vol_spread`), asimetría de la sonrisa
  (`iv_skew_25delta`) y acumulación de OI antes del anuncio.
- `order_imbalance_proxy` por regla de tick sobre barras de minuto.
- `EventBacktest` con gap overnight modelado con precios de apertura reales.
- Transcripciones para el post-evento y el *guidance*.
- Un **conjunto de validación** para el `SurpriseModel` con features suficientemente ricas
  como para que el ejercicio de purged-CV tenga sentido.

**Qué sigue sin poder hacerse, ni con 300 USD/mes**

- **Vintages históricos de consenso.** Insisto porque es el punto donde más gente se
  engaña: **no se compran a este precio.** Las opciones reales siguen siendo WRDS,
  Zacks directo o autorrecolección.
- **Open-Close Volume de Cboe** (compras de apertura de clientes por tamaño): 500 USD/mes
  a precio de lista, ~750 USD/año con descuento académico [búsqueda].
- Quotes NBBO para Lee-Ready canónico (Databento OPRA/equities, +199).
- Fundamentales anteriores a 2009 (requiere Sharadar o Compustat).
- Fechas **confirmadas** por la empresa (Wall Street Horizon).

---

### 14.4 Fuera de presupuesto, pero decisivo si está disponible

**Si el usuario tiene cualquier afiliación universitaria, la jerarquía cambia por
completo:**

1. **WRDS** con I/B/E/S (`statsum_epsus` + variantes sin ajustar), Compustat, CRSP y
   OptionMetrics resuelve **todo** este documento de golpe, incluidos los vintages.
2. **Cboe DataShop académico**: 50 % de descuento en Option EOD Summary (mínimo 500 USD) y
   **750 USD el primer año** para Open-Close Volume Summary [búsqueda]. A ~62 USD/mes
   equivalentes, **es la mejor relación calidad/precio de toda la investigación de
   trading informado en opciones**, y estaría por debajo del nivel 2.

Merece la pena preguntarlo antes de gastar un euro.

---

## 15. Consecuencias para el código del repo

### 15.1 Prioridades sugeridas para `ProviderRegistry`

Prioridad mayor = se resuelve antes. Son valores de partida, no dogma.

```text
kind="prices_daily"     : tiingo 90 · polygon 80 · eodhd 70 · stooq 40 · yfinance 20 · synthetic 0
kind="prices_intraday"  : polygon 90 · alpaca 70 · tiingo 30 (IEX) · synthetic 0
kind="fundamentals"     : sec 100 · sharadar 80 · fmp 60 · eodhd 50 · synthetic 0
kind="earnings_calendar": sec 100 · fmp 70 · finnhub 60 · eodhd 50 · synthetic 0
kind="estimates"        : self_collected 100 · fmp 60 · finnhub 50 · alphavantage 30 · synthetic 0
kind="options"          : orats 90 · polygon 70 · eodhd 40 · tradier 10 (sólo vivo) · synthetic 0
kind="short_interest"   : finra 100 · synthetic 0
kind="off_exchange"     : finra_regsho 90 · finra_ats 70 · synthetic 0
kind="insiders"         : sec 100 · finnhub 50 · eodhd 50 · synthetic 0
kind="universe"         : seed 100 · sharadar 80
```

**SEC va primero en fundamentales** aunque sea gratis y requiera más trabajo, porque es la
única fuente con vintages. Los de pago son conveniencia, no verdad.

### 15.2 `PROVIDER_ENV_KEYS` se queda corto

`config.PROVIDER_ENV_KEYS` no contempla proveedores que este documento identifica como
necesarios. **No modifico `config.py`** (no es mi fichero); lo dejo anotado:

| Proveedor | Variables que harían falta | Motivo |
|---|---|---|
| `finra` | `FINRA_API_CLIENT_ID`, `FINRA_API_CLIENT_SECRET` | la Query API con OAuth2 tiene cuotas mucho mayores que el acceso anónimo |
| `databento` | `DATABENTO_API_KEY` | opción de microestructura |
| `sharadar` | `SHARADAR_API_KEY` | tras el lanzamiento de venta directa (2026-07) |

Nótese que **`finra` no aparece en absoluto** en el dict actual, y sin embargo aporta tres
de las fuentes gratuitas más valiosas del proyecto (short interest, Reg SHO diario, ATS).

### 15.3 Campos que los adaptadores **deben** rellenar

| Dato | `available_at` correcto | Riesgo si se omite |
|---|---|---|
| `FundamentalFact` | `accepted`/`filed` de EDGAR | 20-60 días de look-ahead |
| `EarningsEvent.announced_at` | `acceptanceDateTime`, **nunca `filingDate`** | desplazamiento inconsistente de 1 día |
| `EstimateSnapshot.as_of` | fecha de **captura propia**, no la que declare el proveedor | contaminación por revisión |
| short interest | fecha de **diseminación** FINRA | 9-11 días |
| ATS semanal | fecha de **publicación** (+2 o +4 semanas) | **14 días** |
| open interest | **apertura de T+1** | 1 día, grave en ventana corta |
| Reg SHO diario | T+1 | 1 día |

### 15.4 Fixtures offline que este documento sugiere grabar

Para cumplir la regla de *offline-testable* sin red, conviene versionar respuestas reales
mínimas (una empresa, unos pocos periodos) de: `submissions`, `companyconcept`,
`companyfacts`, un `master.idx`, un `CNMSshvol` de un día, una respuesta de short interest,
una cadena ORATS de una fecha, y un `calendar/earnings` de Finnhub. Todas son pequeñas y
permiten probar los parsers —que es donde están los bugs reales— sin tocar la red.

---

## 16. Preguntas abiertas

Ordenadas por impacto sobre las decisiones de gasto.

1. **Precios exactos de FMP** (Starter / Premium / Ultimate). No se pudieron confirmar; la
   recomendación del nivel 1 depende de que el escalón de entrada esté por debajo de
   ~30 USD/mes.
2. **Profundidad histórica real de cada plan de Polygon.** "5 años" en Starter y "10" en
   Developer proceden de fuentes secundarias. Si Starter llegara a 10 años, la cesta del
   nivel 1 cambiaría a favor de Polygon frente a Tiingo.
3. **Precio de Sharadar Direct** tras su lanzamiento en julio de 2026. Si está por debajo
   de ~150 USD/mes, reordena el nivel 2 entero (resolvería fundamentales pre-2009,
   precios con deslistados y constituyentes desde 1957 de una sola vez).
4. **Profundidad histórica del `calendar/earnings` de Finnhub** y de su campo `hour`. Es
   el pilar gratuito de la clasificación BMO/AMC junto a EDGAR.
5. **Distribución empírica del desfase entre el comunicado de prensa y la aceptación del
   8-K.** Medirla sobre una muestra propia (comparando el *dateline* del EX-99.1 con
   `acceptanceDateTime`) daría una corrección cuantitativa a `announced_at` y es un
   ejercicio autocontenido y barato.
6. **Confirmar el calendario de diseminación del short interest de FINRA** (¿7º u 8º día
   hábil?) y decidir si se carga la tabla oficial en vez de calcularla.
7. **Nombres exactos de los datasets de la Query API de FINRA** para OTC Transparency
   (`weeklySummary` / `weeklyDetail` / `blocksSummary`) y su esquema de paginación.
8. **Política de licencia de transcripciones** de cada proveedor, si se llegan a usar.
9. **Cuota real de ORATS por endpoint**: ¿una petición a `/hist/strikes` de un ticker-día
   cuenta como 1 o como N según el número de strikes? Cambia el cálculo de la §14.3 por un
   factor grande.
10. **Disponibilidad de afiliación académica del usuario.** Es la pregunta con mayor
    impacto económico de toda la lista: convierte un problema de 300 USD/mes en uno de
    ~60.

---

## 17. Referencias

### 17.1 Documentación de proveedores consultada (vía `WebSearch`)

- SEC — *Developer Resources*: `https://www.sec.gov/about/developer-resources`
- SEC — *EDGAR Application Programming Interfaces*:
  `https://www.sec.gov/search-filings/edgar-application-programming-interfaces`
- SEC — *Accessing EDGAR Data*:
  `https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data`
- SEC — *Financial Statement Data Sets*:
  `https://www.sec.gov/data-research/sec-markets-data/financial-statement-data-sets`
- FINRA — *Equity Short Interest Data*:
  `https://www.finra.org/finra-data/browse-catalog/equity-short-interest/data`
- FINRA — *Short Interest Reporting*: `https://www.finra.org/filing-reporting/regulatory-filing-systems/short-interest`
- FINRA — *Daily Short Sale Volume Files*:
  `https://www.finra.org/finra-data/browse-catalog/short-sale-volume-data/daily-short-sale-volume-files`
- FINRA — *OTC Transparency Data*: `https://www.finra.org/filing-reporting/otc-transparency-data`
- FINRA — *OTC Transparency Data Publication Delay*:
  `https://www.finra.org/rules-guidance/rulebooks/otc-transparency-data-publication-delay`
- FINRA Developer Center — catálogo: `https://developer.finra.org/catalog`
- Polygon/Massive — *Pricing*, *Options*, *Flat Files*: `https://polygon.io/pricing`,
  `https://polygon.io/options`, `https://polygon.io/flat-files`
- Financial Modeling Prep — *Pricing Plans*:
  `https://site.financialmodelingprep.com/pricing-plans`
- EODHD — *Pricing*: `https://eodhd.com/pricing`
- Finnhub — *Pricing*: `https://finnhub.io/pricing`
- Tiingo — *Pricing*: `https://www.tiingo.com/about/pricing`
- Alpha Vantage — planes: recopilación en `https://apicostcalc.com/alpha-vantage.html`
- Alpaca — *Market Data*: `https://alpaca.markets/data`
- Tradier — *Market Data*: `https://docs.tradier.com/docs/market-data`
- ORATS — *Data API* y documentación: `https://orats.com/data-api`, `https://docs.orats.io/`
- Cboe DataShop — *Option EOD Summary*, *Open-Close Volume Summary*, *Academic Discount*:
  `https://datashop.cboe.com/option-eod-summary`,
  `https://datashop.cboe.com/cboe-options-open-close-volume-summary`,
  `https://datashop.cboe.com/academic-discount`
- OptionMetrics — *IvyDB US*: `https://optionmetrics.com/united-states/`
- Databento — *OPRA*: `https://databento.com/datasets/OPRA.PILLAR`
- Nasdaq Data Link — *Sharadar Core US Fundamentals (SF1)*: `https://data.nasdaq.com/databases/SF1`
- Sharadar — *Sharadar Launches Direct* (2026-07): `https://blog.sharadar.com/2026/07/sharadar-launches-direct.html`
- QuantRocket — *Sharadar Data*: `https://www.quantrocket.com/sharadar/`
- Wall Street Horizon — *Earnings Calendar*: `https://www.wallstreethorizon.com/earnings-calendar`
- Zacks Data — *Consensus Data*: `https://zacksdata.com/datasets/consensus-data/`
- WRDS — *OptionMetrics*: `https://wrds-www.wharton.upenn.edu/pages/about/data-vendors/optionmetrics/`
- WRDS — *A Note on IBES Unadjusted Data*:
  `https://wrds-www.wharton.upenn.edu/documents/5/A_Note_on_IBES_Unadjusted_Data_pdf.pdf`
- `github.com/fja05680/sp500` — componentes históricos del S&P 500

### 17.2 Referencias académicas citadas

- **Foster, G., Olsen, C., Shevlin, T. (1984)**. *Earnings Releases, Anomalies, and the
  Behavior of Security Returns*. The Accounting Review 59(4). — origen del SUE
  estandarizado por sigma de sorpresas históricas y del modelo de paseo aleatorio
  estacional.
- **Bernard, V., Thomas, J. (1989, 1990)**. *Post-Earnings-Announcement Drift*. — el efecto
  que sostiene el ángulo B.
- **Diether, K., Malloy, C., Scherbina, A. (2002)**. *Differences of Opinion and the Cross
  Section of Stock Returns*. Journal of Finance 57(5). — origen documentado del problema
  de ajuste por splits y redondeo en I/B/E/S; motiva usar los ficheros *unadjusted*.
- **Livnat, J., Mendenhall, R. (2006)**. *Comparing the Post-Earnings Announcement Drift
  for Surprises Calculated from Analyst and Time Series Forecasts*. Journal of Accounting
  Research 44(1). — el PEAD es mayor con sorpresas basadas en analistas que en series
  temporales. Es la referencia que cuantifica lo que se pierde en el nivel 0.
- **DellaVigna, S., Pollet, J. (2009)**. *Investor Inattention and Friday Earnings
  Announcements*. Journal of Finance 64(2). — algoritmo estándar de conciliación de fechas
  Compustat/IBES y de desplazamiento a la sesión siguiente cuando el anuncio es posterior
  al cierre.
- **Pan, J., Poteshman, A. (2006)**. *The Information in Option Volume for Future Stock
  Prices*. Review of Financial Studies 19(3). — construida sobre datos de apertura/cierre
  por tipo de participante; justifica el interés del Cboe Open-Close Volume Summary.
- **Cremers, M., Weinbaum, D. (2010)**. *Deviations from Put-Call Parity and Stock Return
  Predictability*. Journal of Financial and Quantitative Analysis 45(2). — define el
  `vol_spread` que pide el contrato.
- **Blocher, J., Ringgenberg, M. (2019)**. *Short Covering*. — advierte contra interpretar
  el volumen en corto de Reg SHO como sentimiento bajista.
- **Jame, R., Johnston, R., Markov, S., Wolfe, M. (2016)**. *The Value of Crowdsourced
  Earnings Forecasts*. Journal of Accounting Research 54(4). — contenido informativo
  incremental de Estimize.
- **Bailey, D., López de Prado, M. (2014)**. *The Deflated Sharpe Ratio*. — exigido por
  `stats` del contrato.

---

## 18. Resumen ejecutivo en cinco frases

1. **Los cinco datos bloqueantes del proyecto —fundamentales PIT, calendario con hora,
   insiders, short interest y universo— son gratuitos**, vía SEC EDGAR y FINRA; lo que
   cuesta dinero son las opciones y el consenso.
2. **EDGAR `companyfacts` es simultáneamente la fuente más barata y la más point-in-time**
   de fundamentales, porque conserva todos los vintages con su `filed`; el precio a pagar
   es escribir un parser XBRL serio y aceptar que la historia empieza en ~2010.
3. **`acceptanceDateTime` del 8-K item 2.02, y no `filingDate`, es la base correcta de
   `announced_at`**: es una cota superior conservadora del instante del comunicado, así que
   sesga hacia entrar tarde en lugar de hacia el look-ahead.
4. **Los vintages históricos de consenso de analistas no se compran por menos de cuatro
   cifras al año ni fuera del circuito académico**; las dos salidas honestas son el SUE de
   series temporales de Foster (gratis, señal menor pero real) y arrancar hoy un
   recolector propio *append-only*.
5. **Con ~99 USD/mes de ORATS se desbloquea la mitad más informativa del ángulo B**
   (skew 25Δ, pendiente temporal, `vol_spread`, acumulación de OI desde 2007), siempre que
   el crawler se dirija por ventanas de evento y no por calendario, porque la cuota de
   20 000 peticiones/mes no da para reconstruir el panel completo.
