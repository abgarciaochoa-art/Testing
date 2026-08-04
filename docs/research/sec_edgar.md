# SEC EDGAR como fuente de datos — especificación técnica

**Ámbito.** Este documento es la especificación de implementación de EDGAR para
`earnings_alpha/data/edgar.py` y `earnings_alpha/data/fundamentals.py`. Cubre las tres
cosas que el repo necesita de la SEC y que **ninguna otra fuente gratuita da con calidad
point-in-time**:

1. **Fundamentales trimestrales as-reported con vintages** (ángulo A: `factors/*`)
   → API XBRL `companyfacts` / `companyconcept` / `frames`.
2. **El instante exacto del anuncio de resultados** (ángulo B: `pit.tradable_date`,
   `events/*`) → 8-K con *item 2.02* y su `acceptanceDateTime`.
3. **Transacciones de insiders ya publicadas** (`PreEventFeatures.insider_net_buy_form4`)
   → Form 4 en XML.

Complementa `docs/research/data_sources.md` (§4.1 y §5.2), que sitúa a EDGAR frente a los
proveedores comerciales. Aquí no se repite esa comparación: aquí se especifica **cómo se
llama, qué devuelve exactamente y cómo se convierte en `FundamentalFact` y
`EarningsEvent`** sin introducir look-ahead.

---

## 0. Método, procedencia y honestidad sobre lo verificado

### 0.1 Restricciones del entorno en que se escribió este documento

- El proxy de egress **bloquea `sec.gov`, `data.sec.gov` y `efts.sec.gov`**: devuelven 403
  en el CONNECT. `WebFetch` sobre cualquier URL de `sec.gov` devuelve igualmente
  **HTTP 403**. **No se ha ejecutado ni una sola llamada real contra EDGAR.**
- El presupuesto de `WebSearch` de la sesión estaba **agotado** (200/200) antes de empezar.
- Sí funcionan `WebFetch` y la búsqueda de código sobre GitHub. La investigación se ha
  hecho contra **espejos íntegros de la documentación oficial de la SEC alojados en
  repositorios públicos** (copias literales de la página *EDGAR Application Programming
  Interfaces*) y contra **implementaciones de referencia** cuyo comportamiento coincide
  entre sí.

### 0.2 Etiquetas de procedencia usadas en todo el documento

| Marca | Significado |
|---|---|
| **[verificado]** | Comprobado **ejecutando código** contra los datos semilla de este repo. Certeza total. |
| **[oficial]** | Texto de la documentación oficial de la SEC, leído en un **espejo íntegro** de esa página en GitHub. Fiabilidad muy alta; la redacción es la de la SEC. |
| **[secundario]** | Coincidente entre **≥2 implementaciones de referencia independientes** (edgartools, OpenBB, sec-edgar-api, edgarWebR, secedgar…). Fiabilidad alta para la forma del dato; media para cifras exactas. |
| **[inferencia]** | Razonamiento o **decisión de diseño de este repo**, no un hecho externo. Se marca para que nadie lo cite como si lo dijera la SEC. |
| **[pendiente]** | No se ha podido confirmar sin red. Debe comprobarse en la máquina del usuario. |

### 0.3 Lo que sí se ha verificado ejecutando código

Se ejecutó un script contra `data/seed/` (fuera del árbol del repo, en el scratchpad de la
sesión, por la regla de propiedad de ficheros). Resultados **[verificado]**:

| Comprobación | Resultado |
|---|---|
| `sp500_constituents.csv` | 503 filas, columna `CIK` presente, `int64`, **0 nulos** |
| `normalize_cik` sobre las 503 filas | 503/503 producen exactamente 10 dígitos |
| **CIKs distintos** | **500 CIKs para 503 tickers** — el mapeo **no es 1:1** |
| CIKs compartidos | `NWS`/`NWSA` → 1564708 · `GOOGL`/`GOOG` → 1652044 · `FOX`/`FOXA` → 1754301 |
| Tickers con separador de clase | `BRK.B`, `BF.B` (2 de 503) |
| `sp500_historical_components.csv` | 3482 filas, 1996-01-02 → 2025-08-23; **1128 símbolos crudos → 1126 tras `normalize_ticker`** |
| Solape actual ↔ histórico | 482 de los 503 actuales aparecen en el histórico; **644 símbolos históricos ya no están en la lista de hoy**; unión = **1147** |
| Fracción del panel histórico no resoluble con `company_tickers.json` | **56,1 %** |
| Símbolos con sufijo `Q` de quiebra en el histórico | **27** (`AAMRQ`, `ABKFQ`, `DPHIQ`, `LEHMQ`…) |
| Coste de una carga completa de `companyfacts` a 8 req/s | 503 CIK → **63 s**; 1126 símbolos → **141 s** |

> Nota de reconciliación: `docs/research/data_sources.md` cita 1126 y 644. La diferencia con
> el conteo crudo (1128) son **dos símbolos que colapsan al normalizar** el separador de
> clase de acción. Ambas cifras son correctas; la que debe usarse es la **normalizada**,
> porque es la que verá el código. [verificado]

La última fila es la conclusión operativa más importante de todo el documento y se
desarrolla en §11.3: **descargar EDGAR entero para este universo cuesta minutos, no horas**.
El límite de tasa de la SEC no es un problema para este proyecto.

---

## 1. Modelo mental: las cinco entidades de EDGAR

Antes de cualquier endpoint hay que fijar el vocabulario, porque casi todos los errores de
integración vienen de confundir dos de estas cinco cosas.

| Entidad | Qué es | Forma canónica | Ejemplo |
|---|---|---|---|
| **CIK** | *Central Index Key*: identificador del **emisor** (o del insider). Estable de por vida. | 10 dígitos con ceros a la izquierda en `data.sec.gov`; **sin** ceros en las URLs de `Archives` | `0000320193` / `320193` |
| **Accession number** (`adsh`, `accn`) | Identificador del **envío** (submission). Un envío contiene N documentos. | `##########-##-######` con guiones; **sin guiones** en la ruta de `Archives` | `0000320193-24-000006` |
| **Form type** | Tipo de formulario del envío | cadena, con `/A` para enmiendas | `10-Q`, `8-K`, `4`, `10-K/A` |
| **Document** | Un fichero dentro del envío | nombre de fichero | `aapl-20231230.htm`, `ex-99_1.htm` |
| **Fact** (XBRL) | Un número etiquetado con un concepto y un periodo | ver §5 | `Revenues = 119575000000` |

**Regla de construcción de URL de documento** [secundario, coincidente en ≥3 fuentes]:

```
https://www.sec.gov/Archives/edgar/data/{cik_sin_ceros}/{accn_sin_guiones}/{documento}
```

Ejemplo:
`https://www.sec.gov/Archives/edgar/data/320193/000032019324000006/aapl-20231230.htm`

Y el listado de todos los ficheros de un envío:
`https://www.sec.gov/Archives/edgar/data/320193/000032019324000006/index.json`

> **Trampa nº 1.** `data.sec.gov` exige el CIK **con** ceros (`CIK0000320193.json`);
> `Archives` lo admite **sin** ceros. Mezclar las dos convenciones produce 404 silenciosos.
> `types.normalize_cik()` ya devuelve la forma de 10 dígitos; para `Archives` hay que
> aplicar `.lstrip("0")`.

---

## 2. Política de acceso: cabeceras, tasa y errores

### 2.1 No hay autenticación, pero sí identificación obligatoria

> *"These APIs do not require any authentication or API keys to access."* [oficial]

Lo que sí exige la SEC es que el tráfico automatizado **se declare**:

> *"Please declare your traffic by updating your user agent to include company specific
> information."* [oficial — este texto es literalmente el cuerpo de la página de bloqueo
> que sirve `www.sec.gov` cuando rechaza una petición]

Formato aceptado y ampliamente usado [secundario]:

```
User-Agent: <Nombre o Empresa> <correo de contacto>
# p. ej.  "earnings-alpha research abgarciaochoa@gmail.com"
```

Esto ya está previsto en el repo: `config.PROVIDER_ENV_KEYS["sec"] = ["SEC_USER_AGENT"]`.

**Cabeceras recomendadas completas** [secundario + inferencia]:

```python
HEADERS = {
    "User-Agent": os.environ["SEC_USER_AGENT"],   # obligatorio en www.sec.gov
    "Accept-Encoding": "gzip, deflate",            # companyfacts de una megacap ronda los MB
    "Host": "data.sec.gov",                        # o www.sec.gov / efts.sec.gov
}
```

### 2.2 Los tres hosts no se comportan igual

Hallazgo importante y poco documentado [secundario, confirmado por dos fuentes
independientes que lo probaron en vivo]:

| Host | Sirve | Rigor del `User-Agent` |
|---|---|---|
| `www.sec.gov` | `Archives/*`, `files/company_tickers*.json`, índices | **Estricto**: rechaza UA vacío y también UA de navegador. Exige un contacto real. |
| `data.sec.gov` | `submissions/*`, `api/xbrl/*` | Laxo: acepta casi cualquier UA no vacío. |
| `efts.sec.gov` | búsqueda a texto completo | Laxo. |

**Decisión de diseño** [inferencia]: enviar **siempre** el `SEC_USER_AGENT` correcto a los
tres hosts, y que `EdgarProvider.available()` lance `ProviderUnavailable("sec", ...,
missing_env=["SEC_USER_AGENT"])` si la variable no está. Depender de la laxitud de
`data.sec.gov` es frágil: la SEC puede endurecerla sin aviso, y el fallo se manifestaría
como un 403 masivo en mitad de una carga.

### 2.3 Límite de tasa

- **10 peticiones por segundo**, agregadas sobre `www.sec.gov` + `data.sec.gov` +
  `efts.sec.gov` [oficial: la política de *fair access* de la SEC; secundario: coincidente
  en todas las implementaciones consultadas, incl. `sec-edgar-api`, que se describe como
  *"Automatic rate-limiting to 10 requests per second to conform with SEC fair access
  rules"*].
- Superarlo devuelve **403** (no 429) con un cuerpo HTML que contiene el texto
  *"Please declare your traffic…"*, y puede acabar en **bloqueo temporal de la IP**
  [secundario].
- `Settings.max_requests_per_second = 8.0` ya deja el margen correcto.

> **Trampa nº 2.** El error de rate-limit de la SEC llega como **403 con cuerpo HTML**, no
> como 429 con `Retry-After`. Un cliente que solo mire el código de estado confundirá
> "me han limitado" con "no existe" o "no tengo permiso". El adaptador debe **inspeccionar
> el cuerpo**: si un 403 contiene `"declare your traffic"` o `"Request Rate Threshold"`,
> es limitación y debe lanzarse `RateLimited("sec", retry_after=...)`, no
> `ProviderUnavailable`. Son dos ramas de recuperación distintas: una reintenta con
> backoff, la otra aborta.

### 2.4 CORS y compresión

> *"data.sec.gov does not support Cross Origin Resource Scripting (CORS)."* [oficial]

Irrelevante para un cliente Python, relevante si alguna vez se sirve un tearsheet HTML que
intente llamar a EDGAR desde el navegador: no funcionará, hay que proxyar.

### 2.5 Frescura de los datos

> *"The APIs are updated in real-time as filings are disseminated."* … *"the submissions
> API is updated with a typical processing delay of less than a second; the xbrl APIs are
> updated with a typical processing delay of under a minute."* [oficial]

Y para los volcados masivos:

> republicados **cada noche hacia las 03:00 ET** [oficial]

**Consecuencia para el repo** [inferencia]: `submissions` es apto para un *poller* de
eventos casi en tiempo real (detectar un 8-K 2.02 a los segundos de su aceptación). Los
ZIP masivos **no**: llevan hasta 24 h de retraso y solo sirven para la carga histórica
inicial.

### 2.6 Política de caché por tipo de recurso [inferencia]

`DiskCache` debe tratar estos recursos de forma muy distinta, porque su mutabilidad lo es:

| Recurso | TTL | Razón |
|---|---|---|
| Documento de `Archives/...` (un filing concreto) | **∞** | Un filing aceptado es inmutable. Una enmienda es un envío **nuevo**, con otro `accn`. |
| `full-index/{Y}/QTR{n}/master.idx` de un trimestre **cerrado** | **∞** | Histórico congelado. |
| `daily-index/.../master.YYYYMMDD.idx` de un día **pasado** | **∞** | Idem. |
| `submissions/CIK##########.json` | 1 día | Crece con cada filing nuevo. |
| `api/xbrl/companyfacts/...` | 1 día | Crece y **puede ganar vintages** por reexpresión. |
| `api/xbrl/frames/...` de un periodo cerrado | 7 días | Sigue admitiendo filers rezagados meses después. |
| `efts.sec.gov` (búsqueda) | 1 día | Índice vivo. |

---

## 3. Identidad: de ticker a CIK

### 3.1 `company_tickers.json`

```
https://www.sec.gov/files/company_tickers.json
```

Forma exacta [secundario, coincidente en ≥3 fuentes]: **un objeto cuyas claves son enteros
consecutivos en forma de cadena**, no un array:

```json
{
  "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
  "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corporation"}
}
```

- `cik_str` es un **entero** pese al nombre; hay que pasarlo por `normalize_cik`.
- Tamaño ≈ 50 KB, ≈ 10 000 empresas [secundario].

### 3.2 `company_tickers_exchange.json` — preferible

```
https://www.sec.gov/files/company_tickers_exchange.json
```

Forma columnar [secundario, citado literalmente en el generador de edgartools]:

```json
{"fields": ["cik", "name", "ticker", "exchange"], "data": [[320193, "Apple Inc.", "AAPL", "Nasdaq"], ...]}
```

Aporta el **mercado por instrumento** (`Nasdaq` / `NYSE` / `CBOE` / `OTC` / `null`), que es
justo lo que hace falta para filtrar el universo y para no confundir una clase de acción
cotizada con una no cotizada. **Es el fichero que debe usar `universe.cik_for()`.**

### 3.3 Los cuatro problemas del mapeo ticker↔CIK

Este es el punto donde la mayoría de los pipelines se rompen en silencio.

**(a) Solo refleja el presente.** Ambos ficheros son un *snapshot* de hoy. No hay versión
histórica. Una empresa deslistada o adquirida **desaparece** del fichero, aunque su CIK y
todos sus filings sigan existiendo en EDGAR.

> **Consecuencia para el sesgo de supervivencia.** El universo histórico del repo tiene
> **1126 símbolos normalizados** y **644 de ellos ya no figuran en la lista de hoy**
> — el **56,1 % de la unión del panel** [verificado]. Entre ellos están los **27 símbolos
> con sufijo `Q` de quiebra** (`AAMRQ`, `ABKFQ`, `DPHIQ`, `LEHMQ`…) [verificado], que son
> justamente los que un backtest honesto necesita y los que ningún fichero de hoy contiene.
> Esos 644 **no son resolubles** con `company_tickers.json`. Para ellos hay que
> reconstruir el mapeo por otra vía (§11.2: los índices trimestrales incluyen el nombre de
> empresa y el CIK de **todo** filer histórico). Un `cik_for()` que solo mire el fichero de
> hoy devolverá `None` para más de la mitad del panel histórico y, si el consumidor no
> distingue "no existe" de "no lo sé", el backtest quedará sesgado hacia los supervivientes
> exactamente igual que si se hubiera usado la lista de hoy como universo.

**(b) El mapeo CIK→ticker es uno-a-muchos.** [verificado]: en el propio fichero semilla,
**500 CIKs para 503 tickers**. `GOOGL`/`GOOG` comparten el CIK `1652044`; `NWS`/`NWSA` el
`1564708`; `FOX`/`FOXA` el `1754301`.

> **Trampa nº 3, y es grave.** `companyfacts` es **por emisor, no por clase de acción**.
> Alphabet tiene **una** serie de `Revenues` y **una** de `NetIncomeLoss`, no una por clase.
> Si el panel de fundamentales se une por CIK, `GOOGL` y `GOOG` recibirán **cifras
> idénticas**, lo que en un factor cross-section significa **dos observaciones perfectamente
> colineales**: infla el N efectivo, sesga a la baja los errores estándar del IC y duplica
> la exposición de cualquier cartera por quintiles que las seleccione a las dos.
>
> Decisiones obligatorias [inferencia]:
> - Los datos **por acción** (EPS, precio, capitalización) deben calcularse con el número
>   de acciones y el precio **de la clase concreta**, nunca heredarse del emisor.
> - El panel debe llevar una columna `is_primary_share_class: bool` y el backtest debe
>   poder **colapsar a una clase por emisor**. Sin ese interruptor, cualquier estadístico
>   de significancia calculado sobre el panel está inflado.
> - `EarningsEvent.cik` no identifica un evento: el `event_id` correcto sigue siendo por
>   **ticker** (`types.EarningsEvent.event_id` ya lo hace bien).

**(c) Convención de separador.** EDGAR usa **guion** (`BRK-B`, `BF-B`); el repo usa
**punto** (`BRK.B`, `BF.B`) [verificado: son los 2 casos del fichero semilla].
`types.normalize_ticker()` ya convierte `-`→`.`, así que la normalización debe aplicarse
**a la respuesta de EDGAR**, no al revés.

**(d) Cambios de ticker.** El CIK es estable, el ticker no. `submissions` expone
`formerNames[]` (nombres, no tickers) — útil pero insuficiente. El registro histórico de
cambios de ticker hay que construirlo con la fuente del universo, no con EDGAR.

---

## 4. `submissions/CIK##########.json` — el índice de filings

```
https://data.sec.gov/submissions/CIK0000320193.json
```

> *"Each entity's current filing history"* … contiene *"metadata such as current name,
> former name, and stock exchanges and ticker symbols"* [oficial]

### 4.1 Estructura

```json
{
  "cik": "320193",
  "entityType": "operating",
  "sic": "3571",
  "sicDescription": "Electronic Computers",
  "ownerOrg": "06 Technology",
  "insiderTransactionForOwnerExists": 0,
  "insiderTransactionForIssuerExists": 1,
  "name": "Apple Inc.",
  "tickers": ["AAPL"],
  "exchanges": ["Nasdaq"],
  "ein": "942404110",
  "description": "",
  "website": "",
  "investorWebsite": "",
  "category": "Large accelerated filer",
  "fiscalYearEnd": "0928",
  "stateOfIncorporation": "CA",
  "stateOfIncorporationDescription": "CA",
  "addresses": {"mailing": {...}, "business": {...}},
  "phone": "(408) 996-1010",
  "flags": "",
  "formerNames": [{"name": "...", "from": "...", "to": "..."}],
  "filings": {
    "recent": {
      "accessionNumber": [...],
      "filingDate": [...],
      "reportDate": [...],
      "acceptanceDateTime": [...],
      "act": [...],
      "form": [...],
      "fileNumber": [...],
      "filmNumber": [...],
      "items": [...],
      "size": [...],
      "isXBRL": [...],
      "isInlineXBRL": [...],
      "primaryDocument": [...],
      "primaryDocDescription": [...]
    },
    "files": []
  }
}
```

[oficial para la lista de campos de `filings.recent`; secundario para los campos de
cabecera, coincidentes en ≥4 modelos tipados independientes]

### 4.2 Los arrays son **paralelos**, no una lista de objetos

`filings.recent` es un **struct-of-arrays**: `accessionNumber[i]`, `form[i]`,
`acceptanceDateTime[i]`… describen el **mismo** filing. Es eficiente y es una fuente
constante de errores.

```python
rec = payload["filings"]["recent"]
n = len(rec["accessionNumber"])
assert all(len(rec[k]) == n for k in rec), "arrays desalineados"  # invariante obligatoria
rows = [{k: rec[k][i] for k in rec} for i in range(n)]
```

> **Trampa nº 4.** Un `pd.DataFrame(rec)` funciona **solo** si todos los arrays tienen la
> misma longitud. Si la SEC añade una clave nueva más corta, pandas lanza; si la trunca,
> **desplaza silenciosamente todas las fechas**. El `assert` de longitud debe ser un
> `DataQualityError`, no un `assert` que desaparece con `python -O`.

### 4.3 Paginación: `filings.files[]`

> *"at least one year's of filing or to 1,000 (whichever is more) of the most recent
> filings"* [oficial]. Cuando hay más:
> *"files will contain an array of additional JSON files and the date range for the filings
> each one contains"* [oficial]

```json
"files": [
  {"name": "CIK0000320193-submissions-001.json",
   "filingCount": 1000, "filingFrom": "1994-01-26", "filingTo": "2015-02-09"}
]
```

Se descargan desde `https://data.sec.gov/submissions/{name}` y contienen **solo** el bloque
de arrays (la misma forma que `filings.recent`, sin la cabecera de la entidad) [secundario].

> **Trampa nº 5, específica de este proyecto.** Una empresa del S&P 500 con 30 años de
> historia acumula **decenas de miles** de filings, y la inmensa mayoría son **Form 4 de
> insiders**, no informes financieros. Los 8-K 2.02 antiguos y los 10-Q de 2010 **no están
> en `recent`**: están en los ficheros adicionales. Un pipeline que solo lea `recent`
> obtendrá un histórico de eventos que empieza hace pocos años y **no se dará cuenta**,
> porque la respuesta es perfectamente válida. Hay que paginar siempre por `files[]` y
> **verificar que el `filingFrom` más antiguo cubre el inicio del backtest**; si no, es un
> `InsufficientHistory`, no un panel corto.

### 4.4 Semántica campo a campo (lo que importa para PIT)

| Campo | Tipo | Significado | Uso correcto en el repo |
|---|---|---|---|
| `accessionNumber` | `str` con guiones | ID del envío | `FundamentalFact.accession`, auditoría |
| `filingDate` | `YYYY-MM-DD` | Fecha **administrativa**. EDGAR corta a las **17:30 ET**: lo aceptado después recibe fecha del siguiente día hábil [secundario] | **NO usar como `available_at`** |
| `reportDate` | `YYYY-MM-DD` | Fin del periodo cubierto (`period`) | `EarningsEvent.period_end`, `FundamentalFact.period_end` |
| `acceptanceDateTime` | ISO-8601 en **ET** | Instante real de aceptación | **`available_at` / `announced_at`. Es EL campo.** |
| `form` | `str` | `10-Q`, `8-K`, `4`, `10-K/A`… | filtro primario |
| `items` | `str` | **Solo para 8-K**: códigos separados por coma, `"2.02,9.01"` | detección del anuncio de resultados (§7) |
| `primaryDocument` | `str` | Fichero principal del envío | construir la URL del documento |
| `primaryDocDescription` | `str` | Descripción libre | heurística de respaldo |
| `isXBRL` / `isInlineXBRL` | `0/1` | Si el envío trae XBRL | prefiltro de fundamentales |
| `size` | `int` | bytes del envío | control de coste de descarga |
| `act`, `fileNumber`, `filmNumber` | `str` | Metadatos administrativos | no se usan |

**Las tres fechas que no son la misma** — se repite aquí porque es el error nº 1:

```
period / reportDate   fin del trimestre reportado          NUNCA es fecha de señal
filingDate            fecha administrativa, corte 17:30 ET  NUNCA es available_at
acceptanceDateTime    instante real de aceptación           <- SIEMPRE este
```

> **Por qué `filingDate` rompe el backtest.** Un anuncio AMC a las 16:05 ET recibe
> `filingDate` = ese día. Uno a las 18:00 ET recibe `filingDate` = **día siguiente**. Los
> dos son AMC del mismo día de negociación y deben producir el **mismo** `tradable_date`.
> Con `filingDate`, unos eventos se desplazan un día y otros no, de forma correlacionada
> con la hora del anuncio — que a su vez correlaciona con el tipo de noticia. No es ruido:
> es un sesgo con estructura.
>
> Excepción documentada: los **Formularios 3, 4 y 5** conservan la fecha del día hasta las
> **22:00 ET** [secundario], no las 17:30. Otra razón para no razonar nunca con `filingDate`.

**Zona horaria.** `acceptanceDateTime` viene en hora del este **sin desplazamiento explícito
fiable** [pendiente: confirmar si la cadena trae sufijo de zona o es ET desnuda]. El
adaptador debe localizarlo explícitamente en `America/New_York` y convertir a UTC, **nunca**
dejar que pandas lo interprete como naive/UTC. Un error aquí son 4-5 horas, que en el caso
de un anuncio a las 16:05 ET cruza la medianoche UTC y **cambia el día del evento**.

---

## 5. La API XBRL: `companyconcept`, `companyfacts`, `frames`

### 5.1 `companyconcept` — un concepto, una empresa

```
https://data.sec.gov/api/xbrl/companyconcept/CIK0000320193/us-gaap/AccountsPayableCurrent.json
```

> *"returns all the XBRL disclosures from a single company (CIK) and concept (a taxonomy
> and tag) … with a separate array of facts for each units on measure that the company has
> chosen to disclose"* [oficial]

```json
{
  "cik": 320193,
  "taxonomy": "us-gaap",
  "tag": "AccountsPayableCurrent",
  "label": "Accounts Payable, Current",
  "description": "...",
  "entityName": "Apple Inc.",
  "units": {
    "USD": [ /* array de hechos, ver §5.4 */ ]
  }
}
```

Taxonomías válidas: `us-gaap`, `ifrs-full`, `dei`, `srt` [oficial].

### 5.2 `companyfacts` — todos los conceptos de una empresa

```
https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json
```

> *"all the company concepts data for a company are returned in a single API call"* [oficial]

```json
{
  "cik": 320193,
  "entityName": "Apple Inc.",
  "facts": {
    "dei": {
      "EntityCommonStockSharesOutstanding": {
        "label": "Entity Common Stock, Shares Outstanding",
        "description": "...",
        "units": {"shares": [ ... ]}
      },
      "EntityPublicFloat": {"label": "Entity Public Float", "description": "...",
                            "units": {"USD": [ ... ]}}
    },
    "us-gaap": {
      "AccountsPayable": {
        "label": "Accounts Payable (Deprecated 2009-01-31)",
        "description": "...",
        "units": {"USD": [ ... ]}
      },
      "AccountsPayableCurrent": {"label": "Accounts Payable, Current", "description": "...",
                                 "units": {"USD": [ ... ]}}
    }
  }
}
```

Ruta completa a un hecho: `facts[taxonomía][tag]["units"][unidad][i]`.

Nótese el `label` `"Accounts Payable (Deprecated 2009-01-31)"`: **los tags obsoletos siguen
presentes** con la marca de deprecación en la etiqueta. Es una señal explotable para
construir la tabla de prioridad de conceptos (§6).

**Tamaño**: del orden de **varios MB** por megacap [secundario: ~5 MB citado]. Con 503
empresas son ~2-3 GB si se guarda el JSON crudo. Recomendación [inferencia]: **normalizar a
parquet largo en cuanto se descarga** (`ticker, concept, unit, start, end, val, accn, fy,
fp, form, filed, frame`) y no persistir el JSON.

### 5.3 `frames` — un concepto, todas las empresas, un periodo

```
https://data.sec.gov/api/xbrl/frames/us-gaap/AccountsPayableCurrent/USD/CY2019Q1I.json
```

> *"aggregates one fact for each reporting entity that is last filed that most closely fits
> the calendrical period requested"* [oficial]

```json
{
  "taxonomy": "us-gaap",
  "tag": "AccountsPayableCurrent",
  "ccp": "CY2019Q1I",
  "uom": "USD",
  "label": "Accounts Payable, Current",
  "description": "...",
  "pts": 3388,
  "data": [
    {"accn": "0001555538-19-000006", "cik": 1555538,
     "entityName": "SUNCOKE ENERGY PARTNERS, L.P.", "loc": "US-IL",
     "end": "2019-03-31", "val": 78300000},
    {"accn": "0000011199-19-000012", "cik": 11199,
     "entityName": "BEMIS CO INC", "loc": "US-WI",
     "end": "2019-03-31", "val": 465700000}
  ]
}
```

| Campo | Significado |
|---|---|
| `ccp` | *calendar/culumated period*: el marco solicitado (`CY2019Q1I`) |
| `uom` | unidad de medida |
| `pts` | **número de puntos de datos** (empresas) en el marco |
| `data[].loc` | localización del emisor (`US-IL`) — útil para segmentar |
| `data[].accn` | filing del que salió **ese** hecho |

**Formato del periodo** [oficial, cita literal]:

> *"The period format is CY#### for annual data (duration 365 days +/- 30 days), CY####Q#
> for quarterly data (duration 91 days +/- 30 days), and CY####Q#I for instantaneous data.
> Because company financial calendars can start and end on any month or day and even change
> in length from quarter to quarter to according to the day of the week, the frame data is
> assembled by the dates that best align with a calendar quarter or year. Data users should
> be mindful different reporting start and end dates for facts contained in a frame."*

- `CY2019` → anual, duración 365 ± 30 días
- `CY2019Q1` → trimestral (**flujo**), duración 91 ± 30 días
- `CY2019Q1I` → **instantáneo** (saldo de balance); la `I` final es obligatoria

**Unidades** [oficial, cita literal]:

> *"Where the units of measure specified in the XBRL contains a numerator and a denominator,
> these are separated by "-per-" such as "USD-per-shares". Note that the default unit in
> XBRL is "pure"."*

Unidades que se verán en la práctica: `USD`, `shares`, `USD-per-shares` (EPS), `pure`
(ratios), y ocasionalmente `EUR`, `JPY` para filers extranjeros.

> **Trampa nº 6 — `frames` NO es point-in-time y no debe construir el panel.**
> La propia definición lo dice: *"one fact … that is **last filed**"*. `frames` devuelve el
> valor **más recientemente presentado**, es decir, **la reexpresión**, no lo que se conocía
> en la fecha. Además el marco **se rellena a posteriori** durante meses conforme los filers
> rezagados presentan.
>
> - `frames` **sí** sirve para: exploración, comprobar cobertura de un tag, sanity-check
>   cruzado, y descubrir qué empresas usan qué concepto.
> - `frames` **no** sirve para: el panel de `factors/*`. Ahí, `companyfacts` + filtro por
>   `filed` (§5.5) es la única vía correcta.
>
> Y ojo con el otro sentido del error: `pts` cambia con el tiempo para el **mismo** marco.
> Dos ejecuciones del mismo backtest en fechas distintas darían universos distintos. Eso
> viola el principio de **determinismo** (`ARCHITECTURE.md` §0, principio nº 4).

### 5.4 **Un hecho XBRL, campo a campo** — el corazón de todo

Este es el objeto que aparece en `units[unidad][i]` tanto en `companyconcept` como en
`companyfacts`. Ejemplo real de estructura:

```json
{"start": "2023-10-01", "end": "2023-12-30", "val": 119575000000,
 "accn": "0000320193-24-000006", "fy": 2024, "fp": "Q1", "form": "10-Q",
 "filed": "2024-02-02", "frame": "CY2023Q4"}
```

| Campo | Tipo | Presencia | Significado exacto | Uso en el repo |
|---|---|---|---|---|
| `start` | `YYYY-MM-DD` | **solo hechos de duración** | Inicio del periodo del hecho | validar la duración (§6.4) |
| `end` | `YYYY-MM-DD` | **siempre** | Fin del periodo (duración) **o** la fecha del saldo (instante) | `FundamentalFact.period_end` |
| `val` | número | siempre | El valor. Signo según la convención del tag | `FundamentalFact.value` |
| `accn` | `str` con guiones | siempre | Envío del que procede este hecho | `FundamentalFact.accession` |
| `fy` | `int` | siempre | Año fiscal **del filing que lo publicó** | ⚠ ver trampa nº 7 |
| `fp` | `"Q1"\|"Q2"\|"Q3"\|"FY"` | siempre | Periodo fiscal **del filing que lo publicó** | ⚠ ver trampa nº 7 |
| `form` | `str` | siempre | `10-Q`, `10-K`, `8-K`, `10-K/A`, `20-F`… | `FundamentalFact.form`; filtro |
| `filed` | `YYYY-MM-DD` | siempre | **Fecha de presentación del filing que publicó este hecho** | **la clave del PIT** |
| `frame` | `str` | **opcional** | Presente solo si este hecho fue el elegido para ese marco calendario | ayuda de deduplicación **[secundario]** |

> **Trampa nº 7 — `fy` y `fp` describen el FILING, no la observación.**
> Este es probablemente el error más caro y menos conocido de toda la API, y está
> documentado en vivo por al menos una implementación que lo sufrió [secundario, con
> reproducción numérica]:
>
> > *"The derivation keyed on the SEC `:fy`/`:fp` fields, which describe the **filing**, not
> > the observation period. A Q2 10-Q carries current 3-month, current YTD, and prior-year
> > comparative rows all tagged with the same fy/fp, so the YTD lookup collided
> > (last-write-wins) and subtractions mixed windows — verified live producing **−$13.1B
> > "quarterly revenue" for AAPL**."*
>
> Es decir: un 10-Q de Q2 contiene, **todos con `fy=2024, fp="Q2"`**:
> - el trimestre corriente (3 meses),
> - el acumulado del año (6 meses),
> - **los comparativos del año anterior** (3 y 6 meses de 2023).
>
> **Regla obligatoria:** el periodo de una observación se determina **exclusivamente** por
> `(start, end)`. `fy`/`fp` sirven para agrupar filings y para construir
> `EarningsEvent.fiscal_quarter`, **jamás** para identificar de qué trimestre es un número.
> Un panel construido con `groupby(["fy","fp"])` está mal por construcción.

### 5.5 El algoritmo PIT canónico

Esta es la razón por la que EDGAR gratis vence a proveedores de pago: **`companyfacts`
contiene todos los vintages**. Cuando una empresa reexpresa, el nuevo valor aparece como un
hecho **adicional** para el mismo `(start, end)`, con otro `accn` y otro `filed`. El hecho
antiguo **no se borra**.

Definición operativa, para el concepto `c`, empresa `i`, periodo `p = (start, end)`:

```python
def pit_value(facts, as_of: date) -> float | None:
    """Valor de un concepto conocible en `as_of` (Regla de Oro nº 1 del repo).

    Se selecciona el hecho con `filed` máximo entre los ya presentados en `as_of`.
    Devolver el `filed` mínimo da el valor *as-originally-reported*.
    """
    visible = [f for f in facts if f["filed"] <= as_of.isoformat()]
    if not visible:
        return None                    # el dato aún no era conocible
    return max(visible, key=lambda f: (f["filed"], f["accn"]))["val"]
```

Dos series distintas y ambas legítimas:

| Serie | Selección | Para qué |
|---|---|---|
| **as-originally-reported** | `filed` **mínimo** por `(concept, start, end)` | Backtest honesto. Es lo que el gestor vio. |
| **PIT en `t`** | `filed` **máximo** con `filed ≤ t` | Igual de honesto y más realista: incorpora reexpresiones **ya publicadas** en `t`. |
| **reexpresado** | `filed` **máximo** sin filtro | ❌ Look-ahead puro. Es lo que devuelve `frames`. |

`types.FundamentalFact.is_restated` modela esto: `True` cuando existe otro hecho para el
mismo `(concept, start, end)` con `filed` anterior. **Que la diferencia entre serie original
y serie reexpresada sea observable convierte a las revisiones contables en una señal en sí
misma**, gratis, sin comprar Compustat Point-in-Time.

Desempate cuando dos hechos comparten `filed` (ocurre: un 10-K y su exhibit el mismo día)
[inferencia]: ordenar por `(filed, accn)` y quedarse con el mayor `accn`, que es el envío
posterior del mismo día. Debe ser determinista o el panel no es reproducible.

### 5.6 Lo que `companyfacts` **no** tiene

- **Sin XBRL antes de ~2009.** La obligación se implantó por fases 2009-2011 (grandes
  emisores primero). Cobertura sólida del S&P 500 desde **2010-2011**, irregular en 2009,
  **nula antes** [secundario]. Es el límite duro del ángulo A gratuito, y hay que decirlo en
  el tearsheet, no esconderlo.
- **Sin dimensiones.** `companyfacts` devuelve **solo los hechos consolidados**, sin los
  ejes de segmento (geografía, línea de producto). Para segmentos hay que parsear el XBRL
  del filing o usar el dataset *Financial Statement and Notes*.
- **Sin precios, sin consenso, sin calendario prospectivo.** EDGAR es retrospectivo por
  definición.

---

## 6. Conceptos `us-gaap`: variantes y reconciliación

### 6.1 Por qué no existe "el tag de ingresos"

La adopción de **ASC 606** (2018) partió en dos la taxonomía de ingresos, y las empresas
migraron en años distintos. Además, sectores enteros (bancos, seguros, REITs) usan tags
propios. **No hay un nombre canónico.** Cualquier código que haga
`facts["us-gaap"]["Revenues"]` cubre una fracción del universo y **falla en silencio** en el
resto — devolviendo `NaN` para las empresas que más importan.

La solución es una **tabla de prioridad por concepto lógico**: una lista ordenada de tags, se
toma el primero presente para esa empresa-periodo.

### 6.2 Tabla de prioridad recomendada

Base tomada de una implementación de referencia y ampliada con las variantes que aparecen en
las listas de deduplicación de edgartools [secundario]:

```python
SEC_CONCEPT_MAP: dict[str, list[str]] = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",   # ASC 606, el moderno
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "Revenues",                                              # el genérico clásico
        "SalesRevenueNet",                                       # pre-606
        "SalesRevenueGoodsNet",
        "SalesRevenueServicesNet",
        "RevenueFromContractWithCustomer",
        "TotalRevenues",
        "TotalRevenuesAndGains",                                 # financieras
    ],
    "cogs":             ["CostOfGoodsAndServicesSold", "CostOfRevenue", "CostOfGoodsSold"],
    "gross_profit":     ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income":       ["NetIncomeLoss",                        # atribuible a la matriz
                         "ProfitLoss"],                          # incluye minoritarios
    "eps_diluted":      ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"],
    "eps_basic":        ["EarningsPerShareBasic",  "EarningsPerShareBasicAndDiluted"],
    "shares_diluted":   ["WeightedAverageNumberOfDilutedSharesOutstanding",
                         "WeightedAverageDilutedSharesOutstanding"],
    "total_assets":     ["Assets"],
    "total_equity":     ["StockholdersEquity",
                         "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "total_debt":       ["LongTermDebtAndFinanceLeaseObligations",
                         "LongTermDebtAndCapitalLeaseObligations",
                         "LongTermDebt", "LongTermDebtNoncurrent",
                         "LongTermDebtCurrent", "DebtCurrent", "ShortTermBorrowings"],
    "cash":             ["CashAndCashEquivalentsAtCarryingValue",
                         "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
                         "Cash", "CashAndDueFromBanks"],
    "cfo":              ["NetCashProvidedByUsedInOperatingActivities",
                         "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capex":            ["PaymentsToAcquirePropertyPlantAndEquipment",
                         "PaymentsToAcquireProductiveAssets",
                         "PaymentsToAcquirePropertyPlantAndEquipmentAndIntangibleAssets"],
}
```

Notas de la fuente, que son avisos reales y no adorno [secundario]:

- **Revenue**: *"Recent AAPL filings report revenue under RevenueFromContractWithCustomer…
  rather than the old Revenues concept."*
- **Cash**: *"The first concept is the common operating-company aggregate; the rest keep
  banks and restricted-cash filers from hard-missing."*
- **Total debt**: la lista mezcla **agregados** y **componentes**. Tomar
  `LongTermDebtNoncurrent` como "deuda total" **subestima** el apalancamiento al omitir la
  parte corriente. La propia fuente lo reconoce: *"a future loader should combine current and
  noncurrent concepts instead of treating one leg as total debt."* **Para el factor de
  apalancamiento del repo hay que sumar corriente + no corriente, no caer por la cascada.**

### 6.3 Las reconciliaciones que no son una cascada

Tres casos donde "el primero que exista" está **mal**:

| Concepto lógico | Regla correcta |
|---|---|
| `total_debt` | `LongTermDebtNoncurrent + LongTermDebtCurrent + ShortTermBorrowings`, y usar el agregado (`LongTermDebtAndFinanceLeaseObligations`) **solo** si los componentes faltan. |
| `net_income` | `NetIncomeLoss` (atribuible a la matriz) y `ProfitLoss` (incluye minoritarios) **no son lo mismo**. Para EPS y ROE el correcto es `NetIncomeLoss`. Sustituir uno por otro introduce un salto de nivel en la serie, que un factor de crecimiento lee como una aceleración inexistente. |
| `revenue` | Excluding vs Including **AssessedTax** difieren en impuestos indirectos repercutidos. Mezclar las dos en la misma serie temporal crea un escalón artificial. **La prioridad debe fijarse por empresa y mantenerse estable en toda su historia**, no elegirse trimestre a trimestre. |

> **Trampa nº 8 — la selección de tag debe ser estable por empresa.**
> Si en 2017 una empresa usa `SalesRevenueNet` y en 2019 `RevenueFromContract…`, la cascada
> aplicada trimestre a trimestre produce una serie continua **solo si ambos tags miden lo
> mismo**. Cuando no, aparece un salto en el trimestre de migración, y **todos los factores
> de crecimiento y aceleración del repo dispararán una señal falsa exactamente en esa
> fecha**, para muchas empresas a la vez (porque la migración a ASC 606 fue simultánea).
> Es un factor espurio con estructura temporal común: el peor tipo.
>
> Mitigaciones [inferencia]: (a) resolver el tag por empresa sobre toda su historia y
> registrarlo en una columna `concept_used`; (b) en el trimestre de cambio, exigir que
> ambos tags coexistan y comprobar que difieren <1 %, y si no, marcar
> `DataQualityError` o cortar la serie; (c) reportar en el tearsheet cuántas empresas
> cambiaron de tag por trimestre — un pico coincidente con un pico del factor es un
> diagnóstico, no una casualidad.

### 6.4 Filtros de duración: separar trimestres de acumulados

Consecuencia directa de la trampa nº 7. Con `(start, end)`:

```python
duration = (end - start).days
QUARTER_SPAN_DAYS = (80, 100)     # trimestre: ~91 días
ANNUAL_SPAN_DAYS  = (335, 395)    # anual: ~365 días
# los acumulados de 6 y 9 meses (~182, ~273 días) se descartan explícitamente
```

- Hechos **instantáneos** (balance): no tienen `start`. Se identifican por su ausencia.
- Hechos de **duración** (P&G, flujo de caja): tienen `start` y `end`.

Un filtro de duración es **obligatorio** antes de cualquier agregación. Sin él se mezclan
acumulados de nueve meses con trimestres y el resultado no es ruido: es una serie
sistemáticamente inflada en Q2 y Q3.

### 6.5 Derivación del Q4

Muchas empresas etiquetan el cuarto trimestre **solo dentro del 10-K anual**, como cifra
anual. Hay que calcular `Q4 = FY − (Q1+Q2+Q3)`.

Reglas [inferencia, con la mecánica confirmada en una implementación de referencia]:

1. **Los cuatro sumandos deben venir del mismo vintage**, o la resta mezcla un anual
   reexpresado con trimestres originales y produce basura.
2. `available_at` del Q4 sintético = `max(filed del 10-K, filed del Q3)` — es decir, **la
   fecha del 10-K**, no la del cierre del trimestre. El Q4 se conoce cuando se publica el
   10-K, y eso son 45-90 días después del cierre.
3. Marcar el registro como derivado (`is_derived`), porque no es un hecho reportado y su
   error se acumula.
4. Solo aplica a conceptos de **flujo**. Los saldos de balance del Q4 **sí** están en el
   10-K como hecho instantáneo; restarlos sería un error.

### 6.6 Otras trampas de la taxonomía

- **Enmiendas (`10-K/A`, `10-Q/A`)**: son vintages nuevos legítimos. **No descartarlas**;
  entran por el algoritmo de §5.5 y marcan `is_restated=True`.
- **Ejercicios 52/53 semanas** (retail, distribución): `period_end` se mueve varios días
  año a año, y un año tiene 53 semanas. Alinear por `fy`/`fp` **del emisor**, nunca por mes
  natural, y esperar que la duración de un "trimestre" oscile entre 84 y 98 días — de ahí
  la tolerancia amplia de §6.4.
- **Filers extranjeros** (`20-F`, `40-F`): reportan **semestral** o anualmente y a menudo en
  `ifrs-full`, no `us-gaap`. En el S&P 500 son pocos pero existen; hay que decidir
  explícitamente si se incluyen y registrarlo.
- **Tags obsoletos**: `label` contiene `"(Deprecated YYYY-MM-DD)"`. Útil para ordenar la
  cascada automáticamente.

---

## 7. El 8-K item 2.02: el anuncio de resultados

Es la pieza que alimenta `pit.tradable_date`, que `ARCHITECTURE.md` califica de *"la función
más crítica del repo"*.

### 7.1 Qué es el item 2.02

**Item 2.02 — *Results of Operations and Financial Condition*** [oficial: es el título del
item en el Form 8-K]. Existe desde **agosto de 2004** [secundario]. La empresa *suministra*
(*furnish*, no *file*) el 8-K con el comunicado de prensa adjunto como **exhibit EX-99.1**.

Lista de items del 8-K relevantes para este repo [secundario, coincidente entre fuentes]:

| Item | Título | Relevancia |
|---|---|---|
| **2.02** | **Results of Operations and Financial Condition** | **el anuncio de resultados** |
| 9.01 | Financial Statements and Exhibits | acompaña casi siempre al 2.02 (es el que declara el EX-99.1) |
| 7.01 | Regulation FD Disclosure | a veces lleva el mismo comunicado; usado en anuncios no rutinarios |
| 4.02 | Non-Reliance on Previously Issued Financial Statements | **reexpresión**: evento negativo fuerte, señal por derecho propio |
| 5.02 | Departure/Election of Directors or Officers | ruido en la ventana de evento; conviene marcarlo |
| 2.06 | Material Impairments | deterioros, suele acompañar malos resultados |
| 1.01 / 2.01 | Acuerdos y M&A | contaminan la ventana de evento; excluir o marcar |

### 7.2 Cómo detectarlo (algoritmo)

```python
def earnings_8ks(submissions: dict) -> Iterator[dict]:
    """Localiza los 8-K con item 2.02 recorriendo `recent` y TODOS los `files[]`."""
    for blk in iter_all_filing_blocks(submissions):     # §4.3: recent + files[]
        for row in zip_parallel_arrays(blk):            # §4.2
            if row["form"] != "8-K":                    # ojo: "8-K/A" es otra cosa
                continue
            items = {s.strip() for s in (row["items"] or "").split(",")}
            if "2.02" in items:
                yield row
```

Detalles que importan:

- `items` es una **cadena separada por comas** (`"2.02,9.01"`), no una lista. Un `in`
  aplicado a la cadena cruda daría falsos positivos (`"12.02"` no existe, pero la fragilidad
  sí). **Hay que tokenizar.**
- `form == "8-K"` con igualdad estricta. Un `8-K/A` es una **enmienda**: normalmente no es
  el anuncio original y usarlo como `announced_at` retrasaría el evento días o semanas.
- Un mismo trimestre puede generar **varios** 8-K con 2.02 (preliminar y definitivo, o una
  corrección). Se debe quedar el de **`acceptanceDateTime` mínimo** por
  `(cik, reportDate)`: el primero es el que movió el precio.

### 7.3 De `acceptanceDateTime` a `Session`

```python
acc_et = acceptance_datetime.astimezone(ZoneInfo("America/New_York"))
if   acc_et.time() <  time(9, 30):  session = Session.BMO
elif acc_et.time() >= time(16, 0):  session = Session.AMC
else:                               session = Session.DMH   # baja confianza
```

**Por qué `acceptanceDateTime` es una cota superior segura.** El 8-K se suministra
**después** de que el comunicado haya salido por el hilo de prensa; el desfase típico va de
minutos a un par de horas [inferencia; no se ha localizado un estudio con la distribución
exacta]. Por tanto `acceptanceDateTime ≥ announced_at`:

> Un backtest que use `acceptanceDateTime` **nunca opera antes de que la información fuese
> pública**; como mucho opera un poco tarde. Para una plataforma cuyo principio nº 1 es
> "point-in-time o nada", **el sesgo es del signo correcto**.

El riesgo residual es de **clasificación de sesión**, no de look-ahead:

| Comunicado | 8-K aceptado | Clasificación | ¿Correcta? |
|---|---|---|---|
| 07:00 ET | 07:15 ET | BMO | ✅ |
| 08:30 ET | 09:41 ET | DMH | ❌ era BMO — y el evento se retrasa un día |
| 16:01 ET | 16:35 ET | AMC | ✅ |

El caso malo es el segundo: un anuncio BMO cuyo 8-K se acepta después de la apertura. La
mitigación [inferencia] es **triangular** con el campo `hour`/`time` de un proveedor de
calendario (Finnhub, FMP) y exponer una columna de **acuerdo entre fuentes**, de modo que
`EventBacktest` pueda excluir los eventos en desacuerdo y reportar la sensibilidad. *Si una
estrategia solo funciona incluyendo los eventos ambiguos, la estrategia es un artefacto de
fechas mal puestas.*

### 7.4 Extraer las cifras del comunicado (EX-99.1)

El 8-K 2.02 en sí no trae XBRL con el EPS. Las cifras están en el comunicado adjunto, en
HTML. Estrategia de localización [secundario, es exactamente lo que hace edgartools]:

> *"EdgarTools looks for EX-99, EX-99.1, EX-99.01 exhibits or exhibits with 'RELEASE' in the
> description."*

Y para saber si merece la pena parsearlo:

> *"Item 2.02 is present AND an EX-99.1 exhibit contains parseable tables"* → hay resultados
> estructurados. *"some earnings 8-Ks only have narrative text"* → no todos los traen.

Procedimiento:

1. `index.json` del envío → listar documentos y sus `type`/`description`.
2. Elegir el EX-99.x o el que contenga `RELEASE`.
3. Descargar de `Archives/...` y parsear tablas HTML (`pandas.read_html` + detección de
   escala: *units / thousands / millions*).
4. **Del comunicado también se puede afinar la hora**: los *datelines* de PRNewswire/BusinessWire
   suelen incluir `"4:05 p.m. ET"` o `"after the close of market"`. Es la fuente más
   próxima a `announced_at` real que existe sin pagar.

Nivel de esfuerzo [inferencia]: alto y con baja fiabilidad por la heterogeneidad de formatos.
**Recomendación: no bloquear la v1 en esto.** Para `EarningsEvent.eps_actual` el camino
barato y robusto es el XBRL del 10-Q posterior (`EarningsPerShareDiluted`); para
`eps_estimate` hace falta un proveedor de consenso (`data/estimates.py`). El 8-K aporta lo
que solo él aporta: **el timestamp**.

### 7.5 Límites honestos

- **Retrospectivo**: EDGAR no da fechas futuras. El calendario prospectivo necesita otra
  fuente, y esos eventos deben llevar `is_estimated_date=True`.
- **2004→** para el item 2.02. Para 1996-2004 hay que recurrir a Compustat `RDQ` o a un
  proveedor comercial. Con la semilla del repo (1996→) esto deja **8 años sin timestamps
  fiables**.
- Alguna empresa publica bajo item **7.01** en vez de 2.02, o no presenta 8-K el día del
  comunicado. Raro en el S&P 500, pero hay que **medir** la tasa de cobertura
  (eventos detectados / trimestres esperados por empresa) y tratar los huecos como
  `InsufficientHistory`, no como "esa empresa no publicó resultados".

---

## 8. Form 4: transacciones de insiders

Alimenta `PreEventFeatures.insider_net_buy_form4`. **Nota legal obligatoria**: el Form 4 es
un documento **público ya publicado**; usarlo es análisis de información pública, no acceso
a información privilegiada. Es exactamente la distinción que exige el docstring de módulo de
`events/*` en `ARCHITECTURE.md`.

### 8.1 Plazo de presentación — y por qué define la ventana utilizable

Desde la **Sarbanes-Oxley (2002)**, el Form 4 debe presentarse **antes del final del segundo
día hábil** siguiente a la transacción [secundario].

> **Consecuencia crítica para el ángulo B.** `available_at` de una transacción de insider es
> su `acceptanceDateTime`, **no** su `transactionDate`. Entre ambas hay hasta 2 días
> hábiles. Una feature `insider_net_buy_form4` construida sobre `transactionDate` es
> look-ahead de 1-2 días en cada observación — y precisamente en la ventana pre-evento,
> que es donde el ángulo B busca su señal. Sería un detector que "detecta" porque hace
> trampa.
>
> La feature debe agregar por **fecha de publicación**: "compras netas de insiders
> *publicadas* en `[T-N, T-1]`", no "*realizadas* en `[T-N, T-1]`".

Segundo detalle PIT ya mencionado: los Formularios 3/4/5 conservan `filingDate` del día
hasta las **22:00 ET** [secundario], a diferencia del corte general de 17:30. Otra razón
para usar siempre `acceptanceDateTime`.

### 8.2 Estructura XML (`ownershipDocument`)

Estructura confirmada contra un **filing real** (accession `0000009984-19-000114`, Barnes
Group, CIK 9984) [secundario, fichero real de EDGAR]:

```xml
<?xml version="1.0"?>
<ownershipDocument>
  <schemaVersion>X0306</schemaVersion>
  <documentType>4</documentType>
  <periodOfReport>2019-12-10</periodOfReport>

  <issuer>
    <issuerCik>0000009984</issuerCik>
    <issuerName>BARNES GROUP INC</issuerName>
    <issuerTradingSymbol>B</issuerTradingSymbol>
  </issuer>

  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerCik>0001175893</rptOwnerCik>
      <rptOwnerName>MANGUM MYLLE H</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerAddress>...</reportingOwnerAddress>
    <reportingOwnerRelationship>
      <isDirector>1</isDirector>
      <isOfficer>0</isOfficer>
      <isTenPercentOwner>0</isTenPercentOwner>
      <isOther>0</isOther>
      <!-- <officerTitle>, <otherText> cuando aplica -->
    </reportingOwnerRelationship>
  </reportingOwner>

  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <securityTitle><value>Common Stock</value></securityTitle>
      <transactionDate><value>2019-12-10</value></transactionDate>
      <transactionCoding>
        <transactionFormType>4</transactionFormType>
        <transactionCode>P</transactionCode>
        <equitySwapInvolved>0</equitySwapInvolved>
      </transactionCoding>
      <transactionAmounts>
        <transactionShares><value>34.1689</value></transactionShares>
        <transactionPricePerShare><value>60.8979</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
      <postTransactionAmounts>
        <sharesOwnedFollowingTransaction><value>22291.39</value></sharesOwnedFollowingTransaction>
      </postTransactionAmounts>
      <ownershipNature>
        <directOrIndirectOwnership><value>D</value></directOrIndirectOwnership>
        <!-- <natureOfOwnership> cuando es indirecta -->
      </ownershipNature>
    </nonDerivativeTransaction>
    <!-- <nonDerivativeHolding> : tenencias sin transacción -->
  </nonDerivativeTable>

  <derivativeTable>
    <!-- <derivativeTransaction> con conversionOrExercisePrice, exerciseDate,
         expirationDate, underlyingSecurityTitle, underlyingSecurityShares -->
  </derivativeTable>

  <footnotes><footnote id="F1">...</footnote></footnotes>
  <ownerSignature>...</ownerSignature>
</ownershipDocument>
```

**Tres particularidades de parseo que rompen los parsers ingenuos:**

1. **Casi todo valor va envuelto en `<value>`.** `transactionShares` **no** contiene el
   número: contiene `<value>34.1689</value>`. Y el mismo elemento puede contener **además**
   un `<footnoteId id="F1"/>`. Un `elem.text` devuelve `None` o espacios.
2. **`reportingOwner` puede ser una lista.** Una transacción conjunta declara varios
   propietarios. Los parsers que asumen `dict` fallan o pierden datos.
3. **XML mal formado.** Es habitual encontrar entidades HTML sin escapar y caracteres de
   control en `natureOfOwnership` y en las notas al pie; las implementaciones de referencia
   limpian con regex antes de parsear [secundario].

**Localización del XML**: el envío tiene un `primaryDocument` (a menudo un `.xml`, a veces un
`.html` renderizado). Lo robusto es leer `index.json` del envío y quedarse con el fichero
`.xml` cuyo contenido empiece por `<ownershipDocument>` [secundario].

### 8.3 Códigos de transacción — tabla completa

Reproducida literalmente de una implementación de referencia que la copia del propio Form 4
[secundario, verbatim]:

| Código | Descripción oficial |
|---|---|
| **A** | Grant, award or other acquisition pursuant to Rule 16b-3(d) |
| **C** | Conversion of derivative security |
| **D** | Disposition to the issuer of issuer equity securities pursuant to Rule 16b-3(e) |
| **E** | Expiration of short derivative position |
| **F** | Payment of exercise price or tax liability by delivering or withholding securities incident to the receipt, exercise or vesting of a security issued in accordance with Rule 16b-3 |
| **G** | Bona fide gift |
| **H** | Expiration (or cancellation) of long derivative position with value received |
| **I** | Discretionary transaction in accordance with Rule 16b-3(f) resulting in acquisition or disposition of issuer securities |
| **J** | Other acquisition or disposition (describe transaction) |
| **K** | Transaction in equity swap or instrument with similar characteristics |
| **L** | Small acquisition under Rule 16a-6 |
| **M** | Exercise or conversion of derivative security exempted pursuant to Rule 16b-3 |
| **O** | Exercise of out-of-the-money derivative security |
| **P** | **Open market or private purchase of non-derivative or derivative security** |
| **S** | **Open market or private sale of non-derivative or derivative security** |
| **U** | Disposition pursuant to a tender of shares in a change of control transaction |
| **V** | Transaction voluntarily reported earlier than required |
| **W** | Acquisition or disposition by will or the laws of descent and distribution |
| **X** | Exercise of in-the-money or at-the-money derivative security |
| **Z** | Deposit into or withdrawal from voting trust |

(**K** y **V** proceden de una segunda implementación; el resto es verbatim de la primera.)

### 8.4 Cuáles indican negociación informada — y cuáles son ruido

Esta es la parte que decide si `insider_net_buy_form4` mide algo o mide el calendario de
nóminas de la empresa.

| Código | ¿Discrecional? | Interpretación | Uso en la feature |
|---|---|---|---|
| **P** | **Sí, totalmente** | Compra en mercado abierto con dinero propio. **La señal por excelencia.** | **numerador de compras** |
| **S** | Parcialmente | Venta en mercado abierto. Mucho más ruidosa que P: los insiders venden por diversificación, impuestos, divorcios… | **numerador de ventas, con peso menor** |
| **A** | **No** | Concesión de acciones/RSU por el consejo. Es **retribución**, no una opinión sobre el precio. Ocurre en fechas fijas del calendario. | **excluir** |
| **F** | **No** | Retención de acciones para pagar impuestos al *vestear* RSU. **Mecánico y automático.** | **excluir** |
| **M** | **No (el ejercicio)** | Ejercicio de opciones. Suele ir seguido de una venta inmediata (`M` + `S` el mismo día) por un plan preestablecido. | **excluir el `M`**; el `S` asociado debe **marcarse aparte** |
| **C**, **X**, **O** | No | Conversiones y ejercicios de derivados. Mecánicos. | excluir |
| **G**, **W**, **Z** | No | Donaciones, herencias, trusts. Sin contenido informativo de precio. | excluir |
| **D**, **U** | No | Recompras por el emisor, OPAs. Evento corporativo, no opinión individual. | excluir |
| **I**, **J**, **K**, **L** | Mixto | Poco frecuentes y heterogéneos. | excluir por defecto, contar aparte |
| **V** | — | **No es un tipo de transacción**: es un **modificador** que indica reporte anticipado voluntario. | no filtrar por él |

> **Trampa nº 9 — la más común en las features de insiders.** Si no se filtran los códigos,
> `insider_net_buy_form4` queda **dominada por A, F y M**, que son eventos de retribución
> con **estacionalidad anual fija** (concesiones tras el cierre del ejercicio, vesting
> trimestral). Esa estacionalidad **correlaciona con el calendario de resultados** —ambos
> siguen el año fiscal—, así que la feature parecerá tener poder predictivo en la ventana
> pre-evento **sin contener información alguna sobre el resultado**. Es un falso positivo
> que sobrevive a un backtest ingenuo.
>
> Sanity-check obligatorio [inferencia]: la distribución de códigos del panel debe estar
> dominada en número por **A/F/M**, y las **P** deben ser una minoría clara. Si las `P`
> salen mayoritarias, el filtro está mal puesto.

**Definición recomendada de la feature** [inferencia, alineada con la literatura de
*insider trading* que distingue transacciones "rutinarias" de "oportunistas"]:

```
net_buy_value = Σ(P: shares × price)  −  Σ(S_no_asociado_a_M: shares × price)
insider_net_buy_form4 = net_buy_value / dollar_volume_medio_20d
```

Normalizar por volumen en dólares hace la métrica comparable entre una megacap y una empresa
mediana; sin normalizar, la feature mide tamaño de empresa. Variantes que conviene calcular
en paralelo: separar por rol (`isDirector` / `isOfficer` / `isTenPercentOwner`) y contar
**número de insiders distintos** que compran, que suele ser más informativo que el importe
(varios directivos comprando a la vez es más señal que uno comprando mucho).

### 8.5 Volumen de datos

Una empresa grande genera **cientos de Form 4 al año**. Para 503 empresas × 15 años son
**cientos de miles de filings**, cada uno un XML de pocos KB. A 8 req/s eso son **horas de
descarga**, no minutos: es, con diferencia, la parte más cara de EDGAR en este proyecto.

**Recomendación** [inferencia]: no descargarlos vía `submissions` empresa a empresa. Usar
los **índices diarios** (§10.1), filtrar `Form Type == "4"` y quedarse solo con los CIK del
universo vigente ese día. Un fichero por día de sesión (~250/año) frente a cientos de miles
de peticiones individuales.

---

## 9. Búsqueda a texto completo (EDGAR full-text search)

```
https://efts.sec.gov/LATEST/search-index?q=%22earnings%22&forms=8-K&startdt=2024-01-01&enddt=2024-03-31
```

Es el backend Elasticsearch del buscador web `https://www.sec.gov/edgar/search/`.
**No está en la documentación oficial de las APIs**: es un endpoint no documentado pero
estable y ampliamente usado [secundario].

### 9.1 Parámetros

| Parámetro | Valores |
|---|---|
| `q` | Consulta. **Comillas dobles** (`%22…%22`) para frase exacta. Soporta booleanos. Insensible a mayúsculas. |
| `forms` | Lista separada por comas: `8-K`, `10-K,10-Q` |
| `startdt`, `enddt` | `YYYY-MM-DD`, ambos inclusive |
| `ciks` | CIK de 10 dígitos con ceros, para filtrar por empresa |
| `dateRange` / `category` | `custom` cuando se dan fechas explícitas |
| `from` | desplazamiento de paginación (por defecto 0) |
| `size` | resultados por página, **máximo 100** |
| `entityName`, `locationType`, `locationCode(s)` | filtros adicionales |

Admite GET con query string y también **POST con cuerpo JSON** [secundario].

### 9.2 Respuesta

```json
{
  "hits": {
    "total": {"value": 1234, "relation": "eq"},
    "hits": [
      {
        "_id": "0000320193-24-000006:aapl-20231230.htm",
        "_source": {
          "adsh": "0000320193-24-000006",
          "ciks": ["0000320193"],
          "display_names": ["Apple Inc.  (AAPL)  (CIK 0000320193)"],
          "file_date": "2024-02-02",
          "form": "10-Q",
          "root_form": "10-Q",
          "file_type": "10-Q",
          "file_description": "10-Q",
          "period_ending": "2023-12-30",
          "sics": ["3571"],
          "biz_states": ["CA"], "inc_states": ["CA"],
          "biz_locations": ["Cupertino, CA"],
          "file_num": [...], "film_num": [...]
        }
      }
    ]
  },
  "aggregations": { "entity_filter": {...}, "form_filter": {...}, "sic_filter": {...} }
}
```

**`_id` tiene formato `{accession_con_guiones}:{nombre_de_fichero}`** [secundario,
coincidente en dos fuentes]. Partiéndolo por `:` se obtiene directamente el `adsh` y el
documento concreto que hizo *match* — que es lo que permite ir al EX-99.1 exacto sin
descargar el envío entero.

### 9.3 Límites duros

- **Cobertura: 2001 en adelante.** Los filings anteriores existen en EDGAR pero **no están
  en el índice de texto**. Para un backtest desde 1996 esto deja 5 años fuera.
- **`size ≤ 100`** por llamada.
- **`from + size ≤ 10000`**: techo absoluto de resultados paginables.
- **`total.relation`**:
  - `"eq"` → `total.value` es el recuento **exacto**.
  - `"gte"` → hay **más de 10 000** coincidencias y solo las primeras 10 000 son
    alcanzables.

> **Trampa nº 10, y es de honestidad intelectual antes que técnica.** Cuando
> `relation == "gte"`, `total.value` **no es un recuento**: es el techo. Escribir "10 000
> filings mencionan X" a partir de ese campo es fabricar una estadística. Si un análisis
> necesita el recuento real, hay que **trocear la consulta por rangos de fecha** hasta que
> cada tramo devuelva `relation == "eq"` y sumar. El adaptador debe exponer un flag
> `hit_cap: bool` y cualquier consumidor debe negarse a reportar un total con `hit_cap=True`.

### 9.4 Para qué sirve en este repo

Uso principal [inferencia]: **descubrimiento y control de calidad**, no construcción del
panel.

- Localizar los EX-99.1 de resultados directamente (`forms=8-K` + frases del tipo
  `"reports fourth quarter"`), sin recorrer envíos.
- **Auditar la cobertura del item 2.02**: buscar por frases de comunicado y comparar con los
  eventos detectados por `items`. La diferencia son las empresas que publican bajo 7.01 o
  fuera de patrón.
- Detectar eventos cualitativos para excluir ventanas contaminadas
  (`"restatement"`, `"impairment"`, `"CEO transition"`).

No debe usarse para construir la lista de eventos: `submissions` + `items` es exhaustivo,
determinista y llega hasta 2004, mientras que la búsqueda depende de que el texto contenga
la frase que se le ocurrió a quien escribió la consulta.

---

## 10. Índices de filings y volcados masivos

### 10.1 Índices diarios

```
https://www.sec.gov/Archives/edgar/daily-index/{YYYY}/QTR{n}/master.{YYYYMMDD}.idx
```

Variantes en el mismo directorio: `company.{YYYYMMDD}.idx`, `form.{YYYYMMDD}.idx`,
`master.{YYYYMMDD}.idx`, y `index.json` para listar el directorio [secundario].

### 10.2 Índices trimestrales (histórico completo)

```
https://www.sec.gov/Archives/edgar/full-index/{YYYY}/QTR{n}/master.idx
```

Y en ese mismo directorio: `company.idx`, `form.idx`, `crawler.idx`, `xbrl.idx`, más las
versiones comprimidas `.gz` / `.Z` / `.zip` de cada una, y
`https://www.sec.gov/Archives/edgar/full-index/{YYYY}/index.json` para navegar
[secundario, confirmado con un listado de directorio real].

**Cobertura: 1993→** [secundario].

### 10.3 Formato de `master.idx` — y el error clásico

```
CIK|Company Name|Form Type|Date Filed|Filename
--------------------------------------------------------------------------------
320193|Apple Inc.|10-Q|2024-02-02|edgar/data/320193/0000320193-24-000006.txt
```

- Delimitado por `|`.
- **Dos líneas de cabecera** (títulos + guiones) que hay que saltar; en los ficheros
  antiguos hay además varias líneas de preámbulo.
- `Filename` es una ruta **relativa a `https://www.sec.gov/Archives/`**.

> **Trampa nº 11.** `company.idx` y `form.idx` **no** son de ancho delimitado por `|`: son
> **ficheros de ancho fijo**. Una implementación de referencia documenta el fallo exacto:
> *"The parser split `company.idx` on `|`, but company.idx is a fixed-width file — no line
> ever parsed. Verified live: 0 rows for any quarter."* Es un fallo **silencioso**: devuelve
> cero filas, no una excepción. **Usar siempre `master.idx`**, que es el pipe-delimitado.
> Como referencia de magnitud, 2024Q1 tiene **370 304 filings** [secundario].

### 10.4 Volcados masivos

| Fichero | URL | Contenido |
|---|---|---|
| `companyfacts.zip` | `https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip` | **todos** los `companyfacts` de **todas** las empresas |
| `submissions.zip` | `https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip` | todos los `submissions` |

> republicados **"nightly at approximately 3:00 a.m. ET"** [oficial]

Son de varios GB. **Para 503-1128 empresas no compensan**: la carga por API cuesta
63-141 s [verificado, §0.3], frente a descargar y descomprimir varios GB para descartar el
90 %. Se documentan por completitud y porque sí compensarían si el universo pasara a ser
"todo el mercado US".

### 10.5 Financial Statement Data Sets (DERA)

```
https://www.sec.gov/files/dera/data/financial-statement-data-sets/{YYYY}q{n}.zip
```

ZIP trimestrales desde **2009Q1**, ~60-130 MB cada uno [secundario], con cuatro ficheros
TSV:

| Fichero | Contenido | Campos clave |
|---|---|---|
| `sub.txt` | una fila **por filing** | `adsh`, `cik`, `name`, `form`, `period`, `fy`, `fp`, **`filed`**, **`accepted`** |
| `num.txt` | los valores numéricos | `adsh`, `tag`, `version`, `coreg`, `ddate`, `qtrs`, `uom`, `value` |
| `pre.txt` | presentación (a qué estado y línea pertenece cada tag) | `adsh`, `tag`, `stmt`, `line`, `plabel`, `inpth` |
| `tag.txt` | definiciones de tags, estándar y **personalizados** | `tag`, `version`, `custom`, `datatype`, `iord` |

Dos cosas que esta vía da y `companyfacts` **no**:

1. **`sub.txt` trae `accepted`** además de `filed` — el `acceptanceDateTime` para *todos*
   los filings de un trimestre en una sola descarga.
2. **`num.txt.qtrs`** codifica explícitamente el número de trimestres que cubre el hecho
   (`0` = instantáneo, `1` = trimestre, `4` = anual). Es exactamente el filtro de duración
   de §6.4, **ya resuelto por la SEC** — y sin tener que inferirlo de `(end - start)`.
3. **`tag.txt.custom`** identifica los tags inventados por la empresa, que son los que
   rompen las cascadas de §6.2.

Existe además una variante ampliada, *Financial Statement **and Notes** Data Sets*
(`.../dera/data/financial-statement-and-notes-data-sets/{Y}q{Q}.zip`), que añade el texto de
las notas [secundario].

**Recomendación** [inferencia]: usar FSDS para la **carga histórica inicial** (4 ficheros al
año, ~68 ZIP para 2009-2025) y `companyfacts`/`submissions` para el **mantenimiento
incremental** y para cualquier consulta puntual. Es la combinación más barata en peticiones
y la única que da `qtrs` gratis. Publicación: unas semanas después del cierre del trimestre
[pendiente: confirmar el desfase exacto].

---

## 11. Plan de implementación contra el contrato

### 11.1 Superficie propuesta para `data/edgar.py`

```python
class EdgarProvider(Provider):
    """Cliente de SEC EDGAR: submissions, XBRL, 8-K 2.02 y Form 4.

    No requiere credencial pero sí `SEC_USER_AGENT` (nombre + correo), que la SEC
    exige para identificar el tráfico automatizado. Sin esa variable, `available()`
    es False y cualquier llamada lanza `ProviderUnavailable`.
    """
    name = "sec"

    def available(self) -> bool: ...
    def company_tickers(self) -> pd.DataFrame: ...          # cik, name, ticker, exchange
    def submissions(self, cik: CIK, *, all_pages: bool = True) -> pd.DataFrame: ...
    def company_facts(self, cik: CIK) -> pd.DataFrame: ...   # tabla larga, §5.4
    def company_concept(self, cik: CIK, taxonomy: str, tag: str) -> pd.DataFrame: ...
    def frames(self, taxonomy: str, tag: str, unit: str, period: str) -> pd.DataFrame: ...
    def earnings_8k(self, cik: CIK) -> list[EarningsEvent]: ...
    def form4(self, cik: CIK, start: date, end: date) -> pd.DataFrame: ...
    def full_text_search(self, q: str, **kw) -> pd.DataFrame: ...
```

Y en `data/fundamentals.py`, la capa que aplica §5.5 + §6:

```python
def fundamentals_panel(ciks, concepts, *, as_of: date | None = None,
                       vintage: Literal["original", "pit", "latest"] = "pit") -> pd.DataFrame:
    """Panel de fundamentales con MultiIndex (date, ticker).

    `vintage="original"` -> primer `filed` (as-reported).
    `vintage="pit"`      -> último `filed` <= as_of.
    `vintage="latest"`   -> sin filtro. SOLO para diagnóstico: es look-ahead.
    """
```

Que `vintage="latest"` exista y esté documentado como look-ahead es deliberado: permite
**cuantificar** cuánto del rendimiento de un factor viene de las reexpresiones. Un factor
cuyo Sharpe se desploma al pasar de `latest` a `original` estaba viviendo del futuro.

### 11.2 Reconstrucción del mapeo histórico ticker↔CIK

Como `company_tickers.json` solo tiene el presente (§3.3a), y el **56,1 %** del universo
histórico no está en él [verificado: 644 de 1147 símbolos de la unión], la vía es:

1. Descargar los `full-index/{Y}/QTR{n}/master.idx` del rango del backtest → da
   `(CIK, Company Name, Form Type, Date Filed)` de **todo** filer histórico, incluidos los
   deslistados y los quebrados.
2. Cruzar `Company Name` con la columna `Security` del fichero semilla (fuzzy + normalización
   de sufijos societarios).
3. Los que no casen, resolverlos por el `issuerTradingSymbol` que aparece **dentro de los
   Form 4** de esa empresa (§8.2): es el único sitio de EDGAR donde el ticker histórico
   queda registrado con fecha.

El paso 3 es la parte no obvia y es, que se sepa, la única forma gratuita de recuperar el
ticker histórico de una empresa deslistada desde EDGAR. **[inferencia; validación pendiente
de red].** El resultado debe persistirse **append-only**, igual que
`SP500Universe.refresh()`.

### 11.3 Presupuesto de peticiones [verificado]

| Tarea | Peticiones | Tiempo a 8 req/s |
|---|---|---|
| `company_tickers_exchange.json` | 1 | <1 s |
| `companyfacts` del universo actual (503) | 503 | **63 s** |
| `companyfacts` del universo histórico (1126) | 1126 | **141 s** |
| `submissions` + páginas extra (~3/empresa) | ~2000 | ~4 min |
| Índices diarios, 2004-2025 (~5300 sesiones) | ~5300 | ~11 min |
| Form 4 individuales (vía `submissions`) | ~10⁵-10⁶ | **horas** ⇒ usar índices |

**Conclusión:** salvo los Form 4, **toda la ingesta de EDGAR para este proyecto cabe en
minutos**. El límite de 10 req/s no es una restricción real aquí; la restricción es el
diseño (paginar `files[]`, evitar el fan-out de Form 4).

### 11.4 Estrategia de test sin red

El contrato exige que todo módulo se pruebe sin red. Para EDGAR eso significa **fixtures
grabados** que ejerciten precisamente las trampas de este documento:

| Fixture | Qué debe probar |
|---|---|
| `submissions` con `files[]` no vacío | que se paginan las páginas extra (trampa nº 5) |
| `submissions` con arrays desalineados | que lanza `DataQualityError`, no que desplaza fechas (trampa nº 4) |
| 8-K con `items="2.02,9.01"` a las 16:35 ET | `Session.AMC` y `tradable_date` = sesión siguiente |
| 8-K con `items="2.02"` a las 07:15 ET | `Session.BMO` y `tradable_date` = misma sesión |
| 8-K a las 09:41 ET | `Session.DMH` y marca de baja confianza |
| dos 8-K 2.02 para el mismo `reportDate` | se queda el de `acceptanceDateTime` mínimo |
| `companyfacts` con dos hechos mismo `(start,end)` y distinto `filed` | `original` ≠ `pit`, `is_restated=True` (§5.5) |
| `companyfacts` con 3M, 6M y comparativo del año anterior, todos `fy=2024, fp="Q2"` | que el filtro de duración aísla el trimestre (trampas nº 7 y §6.4) |
| empresa que migra `SalesRevenueNet` → `RevenueFromContract…` | que no aparece un salto espurio (trampa nº 8) |
| Form 4 con `A`, `F`, `M`, `P`, `S` | que la feature solo cuenta `P`/`S` (trampa nº 9) |
| Form 4 con `reportingOwner` como lista | que no se pierden propietarios |
| respuesta 403 con cuerpo `"declare your traffic"` | que lanza `RateLimited`, no `ProviderUnavailable` (trampa nº 2) |
| respuesta `efts` con `relation="gte"` | que marca `hit_cap=True` y no reporta el total (trampa nº 10) |

Los fixtures deben ser **JSON/XML recortados a mano**, no descargas completas: un
`companyfacts` real son megabytes y no cabe razonablemente en el repo.

---

## 12. Checklist de revisión (lo que hay que poder responder "sí" a todo)

1. ¿`available_at` sale de `acceptanceDateTime` y **nunca** de `filingDate` ni de
   `period_end`?
2. ¿`acceptanceDateTime` se localiza explícitamente en `America/New_York` antes de pasar a
   UTC?
3. ¿Se paginan **todas** las entradas de `filings.files[]`, y se comprueba que el rango
   cubre el inicio del backtest?
4. ¿El periodo de una observación se determina por `(start, end)` y **jamás** por `fy`/`fp`?
5. ¿Hay filtro de duración antes de cualquier agregación trimestral?
6. ¿El panel se construye con `companyfacts` filtrado por `filed`, y **no** con `frames`?
7. ¿La selección de tag de ingresos es **estable por empresa** en toda su historia?
8. ¿`total_debt` suma corriente + no corriente en lugar de tomar una sola pata?
9. ¿El Q4 derivado usa los cuatro sumandos del **mismo vintage** y `available_at` del 10-K?
10. ¿`GOOGL`/`GOOG`, `FOX`/`FOXA`, `NWS`/`NWSA` están marcados como clases del mismo emisor,
    y el backtest puede colapsarlas?
11. ¿La feature de Form 4 filtra por código `P`/`S` y agrega por **fecha de publicación**?
12. ¿Un 403 con `"declare your traffic"` produce `RateLimited` y no `ProviderUnavailable`?
13. ¿Se lanza `ProviderUnavailable` si falta `SEC_USER_AGENT`, en vez de intentar la llamada?
14. ¿Se lanza `InsufficientHistory` cuando falta histórico, en vez de devolver un panel corto?

---

## 13. Preguntas abiertas (para `docs/OPEN_QUESTIONS.md`)

1. **[pendiente]** ¿`acceptanceDateTime` incluye desplazamiento de zona explícito en la
   cadena, o es ET desnuda? Determina si `fromisoformat` basta o hay que localizar a mano.
   *Se resuelve con una sola llamada real.*
2. **[pendiente]** Desfase real entre el comunicado de prensa y la aceptación del 8-K. Se
   puede **medir** con el repo: parsear el *dateline* de N EX-99.1 y compararlo con
   `acceptanceDateTime`. El resultado calibra la tasa de error de clasificación de sesión
   de §7.3 y es un resultado publicable en el tearsheet.
3. **[pendiente]** Cobertura real del item 2.02 en el S&P 500 por año: ¿qué fracción de
   trimestres-empresa tiene un 8-K 2.02 identificable? Es el numerador de la fiabilidad de
   todo el ángulo B.
4. **[pendiente]** Desfase de publicación de los Financial Statement Data Sets tras el
   cierre del trimestre.
5. **[inferencia, decisión pendiente]** Política para filers extranjeros del índice
   (`20-F`/`40-F`, taxonomía `ifrs-full`): ¿se incluyen con conceptos mapeados, o se
   excluyen y se documenta el hueco?
6. **[inferencia, decisión pendiente]** ¿El panel de fundamentales se indexa por emisor
   (CIK) o por clase de acción (ticker)? El contrato usa `ticker` en el MultiIndex, lo que
   obliga a duplicar las cifras del emisor entre clases — con el problema de colinealidad
   de §3.3b. Debe decidirse explícitamente y quedar registrado.
7. **[pendiente]** Verificar que el `issuerTradingSymbol` de los Form 4 permite reconstruir
   tickers históricos de empresas deslistadas (§11.2, paso 3).

---

## 14. Resumen de una página

- **EDGAR es la única fuente gratuita realmente point-in-time del proyecto**, y para
  fundamentales es además la *mejor*: `companyfacts` conserva **todos los vintages**, algo
  que normalmente exige comprar Compustat PIT.
- El campo que hace todo el trabajo es **`filed`** (para hechos XBRL) y
  **`acceptanceDateTime`** (para eventos). `filingDate`, `period_end`, `fy` y `fp` **no son
  fechas de señal** y usarlos como tales es la fuente número uno de look-ahead.
- **`frames` no es point-in-time**: devuelve el último valor presentado. Sirve para explorar,
  nunca para construir el panel.
- **`fy`/`fp` describen el filing, no la observación.** El periodo se determina por
  `(start, end)`.
- **No existe "el tag de ingresos"**: hace falta una cascada de conceptos, estable por
  empresa, con reconciliaciones explícitas para deuda y beneficio.
- El **8-K item 2.02** da el timestamp del anuncio desde 2004, gratis y auditable. Su
  `acceptanceDateTime` es una **cota superior segura** de `announced_at`: el sesgo va en la
  dirección correcta.
- El **Form 4** solo informa si se filtran los códigos: **`P` es la señal**, `A`/`F`/`M` son
  retribución con estacionalidad que **imita** poder predictivo.
- El límite de **10 req/s no es una restricción real** para este universo: la ingesta
  completa de fundamentales cabe en **63-141 segundos** [verificado]. Lo caro son los Form 4,
  y se resuelve con los índices diarios en vez del fan-out por empresa.
