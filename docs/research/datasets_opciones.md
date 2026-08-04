# Datasets de cadenas de opciones — búsqueda de fuentes abiertas y gratuitas

**Categoría:** histórico de cadenas de opciones (EOD) sobre acciones USA
**Fecha:** 2026-08-04
**Entorno:** proxy de egress que solo permite la familia `*.githubusercontent.com` para descargas
**Presupuesto:** 0 €

---

## 1. Veredicto (léelo primero)

**El hueco de opciones NO se puede cerrar gratis para el histórico. Es un NO honesto, sin adornos.**

Con más precisión, porque la respuesta tiene dos mitades muy distintas:

| Necesidad | ¿Se puede gratis? | Detalle |
|---|---|---|
| **Backfill histórico** (cadenas EOD de S&P 500 alrededor de resultados, 2008→hoy) | **NO** | No existe ninguna fuente abierta con licencia limpia. Todo lo que cubre ese rango es producto comercial (OptionMetrics, ORATS, DeltaNeutral, OptionsDX, Cboe DataShop, Alpha Vantage premium). Lo que circula "gratis" en GitHub son republicaciones sin licencia. |
| **Recolección hacia adelante** (empezar a capturar desde hoy) | **SÍ, con matices** | Cboe publica la cadena completa con IV y griegas sin autenticación, y yfinance da cadenas sin griegas. Pero solo acumula desde el día que lo enchufes: **cero histórico**. Y el uso de Cboe requiere autorización previa suya. |
| **Control de régimen de volatilidad** (VIX) | **SÍ, sin reservas** | Descargado y verificado. Dominio público (ODC-PDDL-1.0). |

**Consecuencia para el proyecto:** cualquier estrategia del ángulo B que dependa de IV, skew, open interest o volumen de opciones **antes** de la fecha de arranque del proyecto no es backtesteable con presupuesto cero. Hay que o bien (a) reformular esas features para que no dependan de opciones, (b) pagar (la opción más barata razonable es ~50–200 $ únicos, ver §5), o (c) aceptar que esas features solo se validarán *out-of-sample* hacia adelante, empezando a recolectar hoy.

---

## 2. Lo que SÍ se ha descargado y verificado

Todo en `/home/user/Testing/data/external/opciones/`. Total: **9,1 MB** (presupuesto 300 MB).

### 2.1 VIX — índice de volatilidad CBOE ✅ recomendado

| Campo | Valor |
|---|---|
| Fuente | `datasets/finance-vix` (proyecto DataHub/Frictionless) |
| URL | https://github.com/datasets/finance-vix |
| Origen real | CSV oficial de Cboe: `cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv` |
| **Licencia** | **ODC-PDDL-1.0** (dedicación a dominio público), declarada en `datapackage.json` |
| Ficheros | `vix-daily.csv`, `vix-monthly.csv`, `LICENSE_finance-vix.txt` |
| Verificado | **9.241 filas**, columnas `DATE,OPEN,HIGH,LOW,CLOSE`, rango **1990-01-02 → 2026-07-31**, **cero nulos** |
| Mantenimiento | Repo actualizado el 2026-08-01 (activo, 99 estrellas) |

Comprobación de autenticidad: los máximos históricos que aparecen son los reales — 82,69 (2020-03-16, máximo histórico de cierre), 80,86 (2008-11-20), 80,06 (2008-10-27). Media 19,44, mínimo 9,14. Es dato genuino.

**Advertencia de calidad documentada por la propia fuente:** las primeras 506 filas (hasta 2004-06-11) tienen OPEN=HIGH=LOW=CLOSE porque Cboe solo registraba cierres antes de esa fecha. No uses el rango intradía antes de mediados de 2004.

*Matiz de licencia, para ser exactos:* DataHub declara PDDL sobre el paquete. Cboe, en sus propios términos, es restrictivo con sus datos de índices. En la práctica el riesgo es mínimo porque la misma serie la redistribuye la Reserva Federal en FRED (`VIXCLS`) sin restricciones, pero conviene saberlo en vez de asumir que PDDL zanja el tema.

### 2.2 QuantConnect Lean — muestra de opciones ⚠️ solo como fixture

| Campo | Valor |
|---|---|
| Fuente | `QuantConnect/Lean`, carpeta `Data/` |
| **Licencia** | **Apache 2.0** (licencia del repositorio, limpia) |
| Ficheros | `lean_sample/aapl_20140606_quote_american.zip` (8,9 MB), `lean_sample/spx_20210104_quote_european.zip` (39 KB), `lean_sample/spx_universe_20210104.csv`, `lean_sample/LICENSE_Apache2.txt` |

Verificado abriendo los zips:

- **AAPL 2014-06-06**: 2.352 contratos (1.176 calls / 1.176 puts), **190 strikes** distintos (195 → 1050), **13 vencimientos** (2014-06-06 → 2016-01-15). Cada contrato es un CSV con barras de **1 minuto** de bid/ask OHLC + tamaños. Formato Lean: precios en **deci-céntimos** (dividir entre 10.000), tiempo en **ms desde medianoche**.
- **SPX 2021-01-04**: fichero de universo con `strike, right, open/high/low/close, volume, open_interest, implied_volatility, delta, gamma, vega, theta, rho`.

Prueba de autenticidad: en el universo SPX, el call y el put de strike 3200 comparten IV exactamente igual (0,444034), que es justo lo que exige la paridad put-call. Es dato real y bien construido.

**Pero la cobertura es de juguete y no sirve para investigar.** Medida directamente: solo **33** ficheros de universo de opciones sobre acciones y **18** de opciones sobre índices en todo el repo; y datos de minuto solo para **AAPL (2 días: 2014-06-06 y 2014-06-09)** y **TWX (1 día)**. Son fixtures de test unitario de Lean.

**Para qué sirve entonces:** como **fixture de desarrollo** con licencia impecable. Permite escribir y testear el parser de cadenas, el cálculo de moneyness/skew y el pipeline de features de opciones sin datos de licencia dudosa, y sin depender de la red. Ese es su único uso legítimo aquí.

---

## 3. Rechazado por licencia — el caso importante

### 3.1 `philippdubach/options-dataset-hist` (y sus mirrors) — **NO UTILIZABLE**

Es, con diferencia, lo que más se parecía a la solución: **53,4 millones de filas contrato-día**, SPY 2008–2025, IWM 2008–2025, QQQ 2011–2025, con bid/ask **y tamaños**, volumen, open interest, IV y las cinco griegas. Formato Parquet por año, ~1,6 GB. Declarado **MIT**.

Y aun así hay que descartarlo. Estado actual:

- El repositorio **original ha desaparecido**: `philippdubach/options-dataset-hist` y `philippdubach/options-data` devuelven **404**.
- Sobreviven al menos dos mirrors creados por cuentas nuevas: `SaidBahaDev/options-dataset-hist` (ficheros en Git LFS, no servibles por `raw`) y `anahatsingh-ui/options-dataset-hist` (creado 2026-07-16, Parquet como blobs planos, **sí descargable**), que se describe a sí mismo como *"preservation mirror... The original repository and its file host went offline in 2026"*.

Hice una **diligencia acotada**: descargué **un solo fichero** (`iwm/options_2012.parquet`, 13 MB) a un directorio temporal — nunca a `data/external/` — para determinar la procedencia, y lo **borré** al terminar. Lo que encontró:

**a) La procedencia es Alpha Vantage, un endpoint de pago.**

El informe técnico del dataset (`dataset_description.pdf`, 21 páginas) evita deliberadamente nombrar al proveedor. Dice literalmente, y nada más:

> *"The options data was collected from market data providers offering historical option chain snapshots."*

Esa vaguedad es en sí misma la señal. Pero el esquema la delata sin ambigüedad:

- Tabla de opciones: `contract_id, symbol, expiration, strike, type, last, mark, bid, bid_size, ask, ask_size, volume, open_interest, date, implied_volatility, delta, gamma, theta, vega, rho, in_the_money` → es **campo por campo** la respuesta de **Alpha Vantage `HISTORICAL_OPTIONS`**.
- Tabla de subyacentes: `open, high, low, close, adjusted_close, volume, dividend_amount, split_coefficient` → es **campo por campo** `TIME_SERIES_DAILY_ADJUSTED` de Alpha Vantage. Y `adjusted_close`, `dividend_amount` y `split_coefficient` vienen **todos a null**, exactamente como los devuelve un plan que no los incluye.
- La documentación de Alpha Vantage indica que `HISTORICAL_OPTIONS` cubre 15+ años con IV y griegas, y que **requiere plan premium de pago**; el tier gratuito no lo incluye.

Conclusión: es una **redistribución no autorizada de un producto comercial**. La licencia MIT es inválida sobre los datos — Dubach puede licenciar su código, no puede regalar derechos sobre datos que licenció de un tercero cuyos términos prohíben la redistribución. Que el original haya desaparecido de GitHub es coherente con esa lectura.

**b) Además, la metodología declarada es falsa.**

El PDF afirma *"Daily option chain snapshots captured at market close"*. Pero la columna `created_at` muestra que **las 250 sesiones de IWM de todo 2012 se insertaron entre las 15:44:45 y las 15:48:52 del 2025-12-16** — cuatro minutos. No son snapshots diarios capturados al cierre; es un backfill masivo por API en diciembre de 2025.

**c) Y la calidad de los campos derivados es mala.** Medido sobre las 405.804 filas de IWM 2012:

- `in_the_money` vale **0 en el 100 % de las filas**. Contrastado contra el cierre real del subyacente, debería ser 1 en ~202.859 de ellas (50 %). El campo está sencillamente roto.
- **Violación de paridad put-call en la IV**: sobre 202.900 pares (misma fecha/vencimiento/strike), la diferencia mediana de IV entre call y put es de **7,8 puntos porcentuales**, y el **36,3 % de los pares difiere en más de 10 pp**. En datos reales esa diferencia debe ser ~0. La columna de IV no es fiable.
- IV máxima de **7,77 (777 %)**. Coherente con la limitación conocida de Alpha Vantage, cuya IV es poco fiable para vencimientos de ≤3 días.

**Decisión: no se usa, no se descarga, no se cita como fuente.** Los precios bid/ask, volumen y open interest probablemente sí sean reales, pero el problema de licencia es descalificante por sí solo, y las columnas analíticas (IV, griegas, ITM) —que son justamente las que el proyecto necesita— no son de fiar.

---

## 4. Tabla completa de candidatos evaluados

| Dataset | URL | Contenido | Rango | Licencia | ¿Descargable aquí? | Veredicto |
|---|---|---|---|---|---|---|
| **datasets/finance-vix** | github.com/datasets/finance-vix | VIX OHLC diario y mensual | 1990-01-02 → 2026-07-31 | **ODC-PDDL-1.0** | **Sí** | **ÚTIL** — descargado y verificado. Control de régimen de volatilidad. |
| **QuantConnect/Lean** (`Data/`) | github.com/QuantConnect/Lean | Quotes de opciones a 1 min + universos con IV/griegas | AAPL 2 días (2014), TWX 1 día, SPX/otros universos sueltos | **Apache 2.0** | **Sí** | **PARCIAL** — licencia limpia y dato real, pero cobertura de juguete. Solo como fixture de desarrollo. |
| philippdubach/options-dataset-hist + mirrors | (original 404; mirror `anahatsingh-ui/options-dataset-hist`) | 53,4 M filas SPY/QQQ/IWM: bid/ask+tamaños, vol, OI, IV, griegas | 2008 → 2025 | MIT **inválida** sobre los datos | Sí (técnicamente) | **NO LICENCIADO** — redistribución de Alpha Vantage premium. Además IV/ITM defectuosos. Ver §3.1. |
| **DoltHub `post-no-preference/options`** | dolthub.com/repositories/post-no-preference/options | `option_chain`: bid, ask, vol, OI, griegas, equity USA | 2019 → 2024-06, ~2.098 símbolos | Mayoría de DoltHub es Creative Commons — **verificar en la ficha** | **No** (dolthub bloqueado) | **MEJOR CANDIDATO EXTERNO** — el histórico libre más largo que existe. Descargar fuera de este entorno. |
| optiondata.org / historicaldata.net — archivo 2013 gratuito | optiondata.org/resource.html | CSV EOD de **todos** los símbolos USA: bid/ask+tamaños, vol, OI, griegas, IV + precios del subyacente | **ene–jun 2013** (~1,8 GB) | Términos del proveedor; gratis "para probar antes de comprar" | **No** (bloqueado) | **PARCIAL** — el mejor corte gratuito con cobertura S&P 500 completa, pero solo 6 meses. Sirve para un estudio piloto de earnings de 2013. |
| OptionsDX (tier gratuito) | optionsdx.com | Cadenas EOD, selección limitada | desde ~2010, por años | Requiere cuenta gratuita; términos propios, redistribución no permitida | **No** (bloqueado) | **PARCIAL** — utilizable en local para uso propio; no redistribuible dentro del repo. |
| Cboe DataShop | datashop.cboe.com | EOD Summary con IV y griegas; Open-Close Volume | histórico completo | Comercial. Prueba gratis 6 meses de Open-Close; **académico 500 $/año** | **No** (bloqueado) | **DE PAGO** — la vía barata si hay afiliación académica. |
| Cboe `cdn.cboe.com/api/global/delayed_quotes` | (documentado en `simonlin1212/global-stock-data`) | Cadena completa + IV + delta/gamma/vega/theta/rho + 0DTE, sin autenticación | **solo snapshot actual** | Tier C: Cboe exige *"approval in advance"* + *"execution of a license agreement"*; **no redistribuible** | Endpoint bloqueado aquí | **SOLO HACIA ADELANTE** — no da histórico. Ver §5. |
| ebeirne/Black_Scholes_Model | github.com/ebeirne/Black_Scholes_Model | Cadenas yfinance del **S&P 500 completo** (~500 tickers): strike, bid, ask, vol, OI, IV, ITM | **1 solo día: 2025-04-23** | **Sin licencia** en el repo; derivado de Yahoo (uso personal) | Sí | **INÚTIL para el objetivo** — corte transversal de un día, sin licencia. No descargado. |
| slihn/volsurface | github.com/slihn/volsurface | SPX EOD: strike, exp, IV, volumen, OI, delta (2.331 filas) | **1 solo día: 2013-12-31** | Sin licencia explícita; fuente no declarada | Sí | **INÚTIL para el objetivo** — un día. No descargado. |
| gar-c/OptionData | github.com/gar-c/OptionData | *Pipeline* en R para scrapear Cboe a Parquet particionado | n/a | n/a | Sí | **SOLO CÓDIGO** — no contiene datos. |
| Kaggle: SPY EOD Options 2020-2022 | kaggle.com/datasets/kylegraupe/... | Cadenas SPY | 2020 → 2022 | Sin licencia clara; con toda probabilidad derivado de OptionsDX | **No** (kaggle bloqueado) | **DUDOSO** — mismo problema de licencia que §3.1. |
| simonlin1212/global-stock-data | github.com/simonlin1212/global-stock-data | Cliente Python (skill) a fuentes oficiales, con **tiers de cumplimiento** | n/a | Apache 2.0 (código) | Sí | **ÚTIL COMO REFERENCIA** — no trae datos, pero su tabla de tiers documenta los términos de Cboe/FINRA/Yahoo citados literalmente. Confirma de forma independiente el análisis de licencias de §5. |

---

## 5. Recomendación operativa

Ordenada por relación coste/beneficio:

1. **Usa ya el VIX descargado** como control de régimen de volatilidad. Es gratis, es dominio público, cubre 1990–2026 y no tiene asteriscos. Cierra la parte de "régimen" del ángulo B aunque no cierre la de cadenas.

2. **Si hay afiliación académica: Cboe DataShop a 500 $/año.** Es la vía limpia y definitiva. Datos oficiales del exchange, IV y griegas incluidas, licencia en regla.

3. **Si no, y se acepta un piloto acotado: el archivo gratuito de ene–jun 2013 de optiondata.org.** Seis meses, pero cubre *todos* los símbolos USA con el esquema completo. Da para una temporada de resultados entera (Q4-2012 reportado en enero-febrero 2013, y Q1-2013 en abril-mayo). Suficiente para decidir si la señal de opciones merece inversión adicional, antes de gastar nada.

4. **Para histórico largo y gratuito: DoltHub `post-no-preference/options`** (2019 → mediados de 2024, ~2.098 símbolos). Es lo mejor que existe en abierto. Requiere descargarlo fuera de este entorno e **inspeccionar la licencia concreta en su ficha** antes de incorporarlo.

5. **Empieza a recolectar hacia adelante desde hoy**, en paralelo a todo lo anterior. Cada día que pasa sin recolectar es un día de panel que se pierde para siempre. Ojo: para el endpoint de Cboe, sus términos exigen **autorización previa**, así que pídela o quédate en yfinance (sin griegas, calculando la IV por tu cuenta con Black-Scholes).

**Lo que no se debe hacer:** incorporar el dataset de 53 M de filas de §3.1. No es solo un problema legal; sus columnas de IV y griegas —las únicas que aportarían valor sobre un simple bid/ask— están medibly rotas.

---

## 6. Notas sobre el entorno (útiles para futuros agentes)

- `raw.githubusercontent.com` sirve **cualquier repo público** sin restricción de sesión. Es la única vía de descarga real.
- `api.github.com/repos/...` y `api.github.com/search/...` **por curl están bloqueados** ("sessions are bound to their configured repositories"). La búsqueda **sí** funciona a través de las herramientas MCP `search_repositories` / `search_code`, que resultaron ser el instrumento decisivo: `search_code` sobre cabeceras de CSV (p. ej. `"strike" "open_interest" "implied_volatility" extension:csv`) localiza datasets que la búsqueda por nombre de repo no encuentra.
- Los ficheros en **Git LFS no se pueden bajar** por `raw` — devuelve el puntero de texto, no el contenido. Conviene comprobarlo antes de planificar una descarga (`.gitattributes` o los primeros bytes del fichero).
- `WebFetch` recibe **403** de bastantes sitios financieros (optionsdx.com, dolthub.com, optiondata.org); `WebSearch` sí devuelve sus contenidos indexados y fue la vía para documentarlos.
