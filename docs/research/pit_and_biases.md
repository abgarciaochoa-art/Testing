# Sesgos y disciplina point-in-time en backtests de fundamentales y eventos de resultados

**Ámbito.** Este documento es la referencia normativa de la plataforma sobre *cuándo*
un dato puede entrar en una señal. Cubre los dos ángulos del proyecto: los factores
fundamentales cross-section (ángulo A) y la ventana en torno al anuncio de
resultados (ángulo B). No discute qué predice: discute qué se sabía y cuándo. Un
factor con IC de 0.05 bien fechado vale infinitamente más que uno con IC de 0.15
fechado un día antes de tiempo.

**Principio rector** (§0 de `docs/ARCHITECTURE.md`): *point-in-time o nada*. Ningún
dato entra en una señal antes del instante en que fue **públicamente conocible**.
Toda serie fundamental se indexa por `available_at`, nunca por `period_end`.

---

## 0. Cómo leer las cifras de este documento

La credibilidad de una guía de sesgos depende de que distinga entre lo que ha medido
y lo que ha leído. Cada cifra lleva una etiqueta:

| Etiqueta | Significado |
|---|---|
| **[medido]** | Calculado por mí sobre los datos reales del repo (`data/seed/*.csv`). Reproducible con los scripts descritos en §14.4. |
| **[simulado]** | Resultado de un Monte Carlo con supuestos explícitos que se declaran junto a la cifra. No es una estimación empírica del mercado; es la cuantificación del *mecanismo*. |
| **[lit]** | Cifra publicada, con referencia en §17, verificada contra el resumen de la fuente. |
| **[verificar]** | Cifra procedente de fuentes secundarias que **no** he podido contrastar contra el texto original (el proxy de este entorno devuelve 403 en la descarga directa de PDFs académicos y de muchos dominios). Úsese como orden de magnitud, no como parámetro de calibración. |
| **[prior]** | Prior de ingeniería mío, no un número publicado. |

Esta distinción no es cosmética. Calibrar un modelo de costes o un umbral de
significancia contra una cifra inventada produce exactamente el mismo tipo de falso
positivo que este documento intenta evitar.

**Restricción del entorno.** Las APIs financieras (SEC, Nasdaq, FINRA, proveedores
comerciales) están bloqueadas en este contenedor. Las mediciones **[medido]** se han
hecho contra los ficheros semilla del repo, que son datos reales. Las validaciones de
código se han ejecutado en el scratchpad, fuera del árbol del repositorio, porque
este documento es el único fichero que tengo asignado.

---

## 1. El reloj del backtest: las siete fechas

### 1.1 Taxonomía

El error número uno de la literatura aplicada es tratar «la fecha del trimestre»
como si fuera una sola cosa. Son siete, y en un trimestre típico del S&P 500 abarcan
un rango de más de dos meses.

| # | Fecha | Qué es | ¿Es una fecha de señal? |
|---|---|---|---|
| 1 | `period_end` | Cierre del trimestre fiscal reportado | **Nunca.** No es conocible como dato hasta 3-8 semanas después |
| 2 | `announced_at` | Instante del comunicado de resultados (nota de prensa / wire) | Sí, para EPS, ingresos, guidance |
| 3 | `accepted_at` (8-K) | Aceptación del 8-K item 2.02 en EDGAR | Casi siempre ≥ (2); redundante para la señal |
| 4 | `accepted_at` (10-Q/10-K) | Aceptación del informe completo en EDGAR | Sí, y **es la única** para balance y flujos de caja |
| 5 | `disseminated_at` | Instante de difusión pública por EDGAR | = (3)/(4) si la aceptación es antes de 17:30 ET; si no, 06:00 ET del siguiente día hábil |
| 6 | `tradable_date` | Primera **sesión bursátil** en que la noticia es explotable | Es la fecha que entra en el panel |
| 7 | `settlement`/`effective` | Fecha en que un cambio de índice, un split o un short interest surten efecto | Relevante para universo y flujo |

`tradable_date` es la función más crítica del repositorio (`pit.tradable_date`). Un
error de un día en ella convierte cualquier backtest de eventos en look-ahead
(§9, cuantificado).

### 1.2 La consecuencia operativa que casi todo el mundo subestima

**EPS e ingresos** viajan en la nota de prensa (fecha 2). **CFO, capex, activo
corriente, deuda a largo plazo, inventarios y cuentas por cobrar** normalmente
**no**: viajan en el 10-Q (fecha 4). Por tanto:

- SUE, sorpresa de ingresos, PEAD → negociables desde `tradable_date(announced_at)`.
- Accruals de Sloan, F-Score de Piotroski, CFO/NI, FCF yield, apalancamiento →
  negociables desde `tradable_date(disseminated_at(10-Q))`, que llega **típicamente
  entre 2 y 6 semanas después** del anuncio [prior].

Calcular accruals con la fecha del 8-K es look-ahead salvo que se verifique, filing
a filing, que ese 8-K concreto incluía el estado de flujos de caja completo. Muchas
empresas del S&P 500 lo incluyen; muchas otras no. La política del repo es
**verificar, no suponer**: el proveedor de fundamentales debe devolver el
`available_at` real de cada concepto, no un `available_at` común por evento.

```python
# earnings_alpha/data/fundamentals.py (contrato, no implementación)
# CADA FundamentalFact lleva SU PROPIO available_at. Nunca uno por trimestre.
FundamentalFact(ticker="AAPL", concept="EarningsPerShareDiluted", value=2.40,
                period_end=date(2025, 6, 28),
                available_at=datetime(2025, 7, 31, 20, 30, tzinfo=UTC),  # 8-K, 16:30 ET
                fiscal_period="2025Q3", form="8-K")
FundamentalFact(ticker="AAPL", concept="NetCashProvidedByOperatingActivities", value=1.0e10,
                period_end=date(2025, 6, 28),
                available_at=datetime(2025, 8, 1, 14, 5, tzinfo=UTC),    # 10-Q, 10:05 ET
                fiscal_period="2025Q3", form="10-Q")
```

### 1.3 Definición formal de point-in-time

Sea `x` un dato y `avail(x)` el instante en que fue públicamente conocible. Una
señal `s_t` es PIT si y solo si

```
s_t = f({x : avail(x) <= close(t)})
```

donde `close(t)` es el cierre de la sesión `t` si la señal se ejecuta en el cierre, o
la apertura si se ejecuta en la apertura. Dos corolarios operativos:

1. **PIT es una propiedad del conjunto de entrada, no del resultado.** Una señal
   puede parecer causal y no serlo si un paso intermedio (una normalización, un
   `fillna`, un ajuste por splits, un winsorizado global) toca el futuro. §14 da el
   test que lo detecta.
2. **PIT no es «restar N días».** Restar días es un parche que ni corrige el sesgo
   (si el retardo real es mayor que N) ni es gratis (si es menor, se tira alfa).
   Sirve solo como *cinturón de seguridad* encima de un `available_at` real, jamás
   como sustituto.

### 1.4 El invariante que debe cumplir todo panel

Todo panel canónico `(date, ticker)` que alimente un backtest debe poder acompañarse
de una tabla de procedencia con, por cada celda, el `available_at` máximo de los
datos que la produjeron. El invariante es:

```
max(available_at de los inputs de la celda) <= timestamp de decisión de la celda
```

Si no se puede construir esa tabla, no se puede afirmar que el panel es PIT. La
implementación práctica no necesita almacenarla por celda: basta con propagar, por
factor y fecha, el `available_at` máximo (§14.3).

---

## 2. B1 — Look-ahead por usar `period_end` en vez de la fecha de presentación

### 2.1 Cómo se manifiesta

Es la forma canónica y la más destructiva. El investigador tiene una tabla de
fundamentales indexada por trimestre fiscal, la reindexa a fechas diarias con un
`ffill` a partir de `period_end`, y con eso construye el factor. El resultado es que
el 30 de junio la estrategia ya conoce el beneficio del trimestre que cerró ese día
y que no se publicará hasta finales de julio.

```python
# ANTIPATRÓN. No hacer esto nunca.
fund = fund.set_index(["period_end", "ticker"]).sort_index()
panel = fund.unstack("ticker").reindex(trading_days).ffill()   # look-ahead de 3-8 semanas
```

Tres razones por las que este error es tan caro:

1. **Se come el propio salto del anuncio.** El factor ya "sabe" el resultado antes de
   que el mercado lo sepa, así que la posición está colocada antes del salto. El
   salto de resultados es, con mucho, el mayor movimiento idiosincrático del
   trimestre.
2. **Es sistemático, no ruidoso.** Afecta a todos los nombres, todos los trimestres,
   siempre en la dirección de inflar. No se promedia a cero.
3. **Sobrevive a la mayoría de los controles.** Neutralización sectorial, límites de
   peso, costes de transacción, purged CV: nada de eso lo detecta, porque el problema
   está en el input, no en el estimador.

### 2.2 Cuánto infla

El mecanismo se puede cuantificar sin datos de mercado. Modelo **[simulado]**: 400
nombres, 10 años, 4 eventos/año/nombre, volatilidad idiosincrática diaria 1.5%,
salto de resultados con desviación típica 5.5% (≈ 4.4% de movimiento absoluto medio,
consistente con el rango típico del S&P 500 [prior]), cartera long-short neutral con
bruto 1 reponderada a diario, y **cero alfa real**. Se introduce una única
contaminación: el score en `t` tiene correlación `ρ` con el salto que ocurrirá en
`t+1`.

| Fuga `ρ` con el salto futuro | Sharpe anualizado resultante [simulado] |
|---|---|
| 0.00 (limpio) | −0.04 |
| 0.02 | +0.37 |
| 0.05 | +0.76 |
| 0.10 | +1.60 |
| 0.20 | +3.31 |

Lectura: **una correlación del 5% con el retorno del día siguiente basta para
fabricar un Sharpe de 0.76 partiendo de puro ruido.** El desfase `period_end` →
`announced_at` no produce una `ρ` del 5%: produce una `ρ` cercana a la correlación
entre el fundamental y el salto, que para un SUE bien construido es mucho mayor. Con
signo perfecto del salto y 4 eventos al año, el techo del artefacto es **17.6% anual
por nombre** [simulado], con volatilidad diversificable.

Referencias del orden de magnitud en la literatura: los backtests con datos
reexpresados superan sistemáticamente a los equivalentes con datos tal como se
reportaron, y en factores de calidad la diferencia documentada ronda los **100 pb
anuales** [verificar] — y eso es solo la parte de reexpresión, sin el desfase de
fecha, que es mucho mayor.

### 2.3 Cómo se corrige en código

La corrección tiene tres piezas: (a) `available_at` por dato, (b) as-of join
estrictamente hacia atrás, (c) tope de obsolescencia.

```python
# earnings_alpha/pit/asof.py — patrón de referencia (validado, §14.4)
import pandas as pd
from earnings_alpha.errors import LookAheadError

def asof_join(
    signal_dates: pd.DatetimeIndex,
    facts: pd.DataFrame,             # columnas: ticker, available_at, value
    *,
    lag_days: int = 0,               # cinturón de seguridad ADICIONAL, no sustituto
    max_staleness_days: int | None = None,
) -> pd.DataFrame:
    """Une hechos fundamentales a un calendario de decisión respetando PIT.

    Para cada `(date, ticker)` devuelve el ÚLTIMO hecho cuyo `available_at` es
    menor o igual que `date - lag_days`. Antes del primer hecho devuelve NaN: no
    se rellena hacia atrás bajo ninguna circunstancia. Si `max_staleness_days`
    está fijado, un hecho más antiguo que ese umbral se invalida en vez de
    arrastrarse indefinidamente (protege de tickers que dejan de reportar).
    """
    f = facts.copy()
    f["available_at"] = pd.to_datetime(f["available_at"])
    if f["available_at"].isna().any():
        raise LookAheadError("hay hechos sin available_at; no se puede fechar la señal")
    f = f.sort_values("available_at")

    out = []
    for ticker, grp in f.groupby("ticker", sort=False):
        left = pd.DataFrame({"date": pd.DatetimeIndex(signal_dates)})
        left["cutoff"] = left["date"] - pd.Timedelta(days=lag_days)
        m = pd.merge_asof(
            left.sort_values("cutoff"),
            grp[["available_at", "value"]],
            left_on="cutoff", right_on="available_at",
            direction="backward",         # <- lo único que impide el look-ahead
            allow_exact_matches=True,
        )
        m["ticker"] = ticker
        if max_staleness_days is not None:
            stale = (m["date"] - m["available_at"]).dt.days > max_staleness_days
            m.loc[stale, ["value", "available_at"]] = pd.NA
        out.append(m)

    return pd.concat(out).set_index(["date", "ticker"]).sort_index()
```

Resultado de la validación **[medido]**, con tres vintages del mismo hecho
(`2025-02-05: 1.00`, `2025-05-01: 1.10`, `2025-08-14: 0.85`) y fechas de señal
mensuales:

```
2025-01-01  NaN     <- antes del primer vintage: NaN, no relleno hacia atrás
2025-02-01  NaN
2025-03-01  1.00
2025-04-01  1.00
2025-05-01  1.10
...
2025-09-01  0.85
```

El join naïve (aplicar el último valor conocido hoy a toda la historia) produce
`0.85` constante: look-ahead en **8 de 9 fechas**.

### 2.4 Test obligatorio

```python
def test_asof_join_nunca_adelanta():
    facts = pd.DataFrame({"ticker": ["A"], "available_at": [pd.Timestamp("2025-06-10")],
                          "value": [1.0]})
    dates = pd.to_datetime(["2025-06-09", "2025-06-10", "2025-06-11"])
    got = asof_join(dates, facts)["value"].droplevel("ticker")
    assert pd.isna(got.loc["2025-06-09"])     # el día ANTES no existe
    assert got.loc["2025-06-10"] == 1.0       # el día EXACTO sí
    assert got.loc["2025-06-11"] == 1.0
```

---

## 3. El retardo real entre cierre de trimestre y publicación

### 3.1 El marco regulatorio fija el techo, no la mediana

Prácticamente todo el S&P 500 es *large accelerated filer* (flotante ≥ 700 M USD).
Los plazos máximos son [lit]:

| Formulario | Large accelerated | Accelerated | Non-accelerated |
|---|---|---|---|
| 10-K | 60 días naturales | 75 | 90 |
| 10-Q | 40 días naturales | 40 | 45 |
| 8-K (incl. item 2.02) | 4 días hábiles desde el hecho | ídem | ídem |
| Form 4 (insiders) | 2 días hábiles desde la operación | ídem | ídem |
| 13F | 45 días desde el cierre de trimestre | ídem | ídem |

Estos son **topes legales**. La nota de prensa de resultados casi siempre precede al
10-Q, porque el 8-K item 2.02 se dispara con el comunicado, no con el informe.

### 3.2 Distribución típica del retardo en el S&P 500

Compendio de lo que se puede afirmar con confianza:

- La temporada de resultados arranca **2-3 semanas después** del cierre de trimestre
  (los grandes bancos abren), y el grueso se concentra en las **4-6 semanas**
  siguientes [lit].
- Para el S&P 500, la moda del retardo `period_end → announced_at` está en la
  **franja de 20 a 35 días naturales**, con cola derecha hasta el tope de 40 días del
  10-Q y algún rezagado por encima **[prior]**. Los trimestres fiscales de cierre de
  año (10-K) se desplazan a la derecha: 30-60 días.
- El retardo `announced_at → 10-Q disseminated_at` es el que decide cuándo son
  negociables accruals y F-Score: mediana **de 2 a 6 semanas** [prior]. En algunos
  emisores es de horas (presentan a la vez); en otros, casi el mes completo.

**Advertencia metodológica clave:** el retardo es *endógeno*. No es ruido: las
empresas eligen cuándo publicar y la elección correlaciona con el signo de la
noticia (§10). Por eso hay dos errores simétricos:

- Usar el retardo **realizado** de cada empresa-trimestre para fechar el dato es
  correcto (es lo que pasó), pero usarlo para **construir una feature** (p. ej.
  "días de retraso") exige conocer la fecha *esperada* de forma PIT (§8, §10).
- Usar un retardo **fijo** (45 o 90 días) es la política de emergencia cuando no
  hay `available_at`. Es conservadora en media, pero deja look-ahead residual en la
  cola de rezagados y regala alfa en los adelantados.

### 3.3 Regla PIT de disponibilidad en EDGAR

EDGAR introduce una sutileza que casi ningún backtest modela [lit]:

- Una presentación transmitida **después de las 17:30 ET** recibe fecha de
  presentación de las 06:00 ET del **siguiente día hábil** y **no se difunde** hasta
  entonces. Es decir, el 10-Q aceptado a las 18:10 ET del martes no es público hasta
  el miércoles por la mañana.
- **Excepción importante para el ángulo B:** los formularios 3, 4, 5 y 144 (insiders)
  transmitidos entre las 17:30 y las 22:00 ET **sí** se difunden ese mismo día. Un
  Form 4 aceptado a las 19:00 ET del martes es público el martes por la noche y, por
  tanto, negociable en la sesión del miércoles.

```python
# earnings_alpha/pit/edgar_clock.py — regla de disponibilidad
from datetime import datetime, time
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
_INSIDER_FORMS = {"3", "3/A", "4", "4/A", "5", "5/A", "144", "144/A"}

def disseminated_at(accepted_at: datetime, form: str, cal: "TradingCalendar") -> datetime:
    """Instante de difusión pública de una presentación EDGAR.

    Regla del EDGAR Filer Manual: aceptación posterior a las 17:30 ET -> fecha de
    presentación y difusión a las 06:00 ET del siguiente día hábil, salvo los
    formularios de insiders (3/4/5/144), que se difunden hasta las 22:00 ET del
    mismo día.
    """
    local = accepted_at.astimezone(NY)
    if local.time() <= time(17, 30):
        return accepted_at
    if form in _INSIDER_FORMS and local.time() <= time(22, 0):
        return accepted_at
    nxt = cal.next_business_day(local.date())
    return datetime.combine(nxt, time(6, 0), tzinfo=NY)
```

**Nota de dirección del error.** Confundir `filing date` con `available_at` en la
dirección "usar filing date" es *conservador* (llegas tarde), y confundirlo en la
dirección "usar acceptance datetime sin la regla de las 17:30" es *look-ahead* (te
adelantas una sesión). Solo el segundo invalida un backtest, pero el primero cuesta
alfa de forma silenciosa. Se modelan los dos.

### 3.4 Y para los resultados, el reloj no es EDGAR

Para el ángulo B la fuente de verdad **no es el 8-K**: es la nota de prensa. Las
empresas difunden por wire (Business Wire, PR Newswire, su propia web de relación con
inversores) y el 8-K se acepta minutos u horas después. `announced_at` debe ser el
timestamp del wire. Usar el `accepted_at` del 8-K es conservador (llegas tarde) pero
en la práctica suele caer en la misma sesión, así que rara vez cambia el
`tradable_date`. La política del repo:

```
announced_at := timestamp del wire si el proveedor lo da
             := accepted_at del 8-K item 2.02 en caso contrario, marcando
                EarningsEvent.source para poder auditar el downgrade
```

---

## 4. B2 — Reexpresiones (restatements) y por qué manda el primer vintage

### 4.1 Cómo se manifiesta

Un proveedor de fundamentales «estándar» sirve, para el trimestre 2019Q2, el valor
que hoy figura en sus registros. Ese valor puede ser:

- el que se publicó en julio de 2019,
- el corregido en una reexpresión de 2020 tras un error contable,
- el reclasificado por un cambio de norma (ASC 842, ASC 606) que redefinió partidas,
- el reajustado por una operación discontinuada, que reexpresa hacia atrás toda la
  serie de ventas y márgenes.

Los cuatro casos son look-ahead, y **los dos últimos son los más frecuentes y los
menos discutidos**. Las reclasificaciones por operaciones discontinuadas reexpresan
retroactivamente el histórico de ingresos de la empresa; un factor de "aceleración de
ventas" calculado sobre esa serie reexpresada usa una definición de ventas que en su
momento nadie tenía.

### 4.2 Frecuencia y magnitud

- En 2023 se registraron **430 reexpresiones** entre empresas cotizadas
  estadounidenses, un 6% menos que las 458 de 2022 [lit].
- La distinción **Big R** (reemisión, el estado anterior deja de ser fiable) vs
  **little r** (revisión, se corrige en el siguiente informe) importa: la proporción
  de Big R subió al **38% en 2022** desde el 25% en 2021 [lit].
- Sobre ~4.000-5.000 emisores, 430 reexpresiones ≈ **1% de emisor-año**. Suena poco,
  pero el sesgo no es proporcional a la frecuencia: se concentra en las empresas de
  peor calidad contable, que es exactamente la cola que los factores de calidad,
  accruals y F-Score intentan capturar.
- Efecto agregado documentado en factores de calidad: los backtests con datos
  reexpresados superan a los de datos originales en el orden de **100 pb anuales**
  en muestras estadounidenses [verificar].
- Hay una tercera fuente, distinta de la reexpresión y a menudo mayor: la
  **estandarización** del proveedor. Un estudio encuentra que **17 de 30 variables**
  de una base comercial difieren de forma significativa de lo que figura en el 10-K
  [verificar], por normalización de definiciones, redondeos y erratas. Es decir: no
  basta con "usar el primer vintage" del proveedor; hay que saber si el proveedor
  reescribe la definición además del valor.

### 4.3 Cómo se corrige en código

La estructura de datos correcta es **bitemporal**: cada hecho lleva `period_end` (a
qué se refiere) y `available_at` (cuándo se supo), y **los vintages no se sobrescriben,
se añaden**.

```
ticker  concept  period_end   available_at         value   form   is_restated
AAA     Revenue  2024-03-31   2024-04-25 20:05Z    1000.0  8-K    False
AAA     Revenue  2024-03-31   2024-05-02 14:00Z    1000.0  10-Q   False   <- confirma
AAA     Revenue  2024-03-31   2025-02-14 21:30Z     940.0  10-K/A True    <- reexpresión
```

Reglas de la plataforma:

1. **Nunca hacer `UPDATE`.** La caché de fundamentales es *append-only*. Si un
   proveedor devuelve un valor distinto para `(ticker, concept, period_end)` que ya
   estaba, se inserta como vintage nuevo con el `available_at` de la consulta y
   `is_restated=True`; no se pisa el anterior.
2. **El backtest consume `direction="backward"`** sobre `available_at` (§2.3). Eso
   selecciona automáticamente el vintage vigente en cada fecha, incluida la
   reexpresión a partir del día en que se publicó (que es correcto: a partir de ahí,
   el mercado sí la conoce).
3. **La reexpresión es, además, una señal.** `is_restated=True` con una revisión
   negativa material es información negativa sobre calidad contable, negociable
   desde su propio `available_at`. Que no se pueda usar hacia atrás no significa que
   haya que tirarla.
4. **Detección de proveedores que reescriben.** Si el proveedor no da vintages, la
   plataforma los reconstruye por observación: se guarda un *snapshot* fechado de
   cada descarga y se comparan snapshots consecutivos.

```python
# earnings_alpha/data/fundamentals.py — reconstrucción de vintages por observación
def upsert_vintage(store: pd.DataFrame, new: pd.DataFrame, observed_at: datetime) -> pd.DataFrame:
    """Inserta hechos nuevos como vintage adicional; nunca sobrescribe.

    Si `(ticker, concept, period_end)` ya existe con otro valor, se marca el hecho
    entrante como reexpresión y su `available_at` es `observed_at` (lo más
    conservador posible: solo sabemos que a esa hora ya era distinto).
    """
    key = ["ticker", "concept", "period_end"]
    if len(store):
        prev = store.sort_values("available_at").groupby(key, as_index=False).last()
        merged = new.merge(prev[key + ["value"]], on=key, how="left", suffixes=("", "_prev"))
    else:
        merged = new.copy()
        merged["value_prev"] = pd.NA

    changed = merged["value_prev"].notna() & (merged["value"] != merged["value_prev"])
    fresh = merged["value_prev"].isna()
    keep = changed | fresh                       # un valor repetido NO genera vintage nuevo

    add = merged[keep].drop(columns=["value_prev"]).copy()
    add["is_restated"] = changed[keep].to_numpy()
    add["available_at"] = add["available_at"].fillna(observed_at)
    return pd.concat([store, add], ignore_index=True)
```

Comprobación **[medido]** con la secuencia del ejemplo (8-K `1000.0` → 10-Q `1000.0`
→ 10-K/A `940.0`): el almacén queda con **dos** vintages, no tres. La confirmación
del 10-Q no crea registro porque el valor no cambia, y el tercero entra con
`is_restated=True`. Es el comportamiento correcto: un vintage por *cambio*, no por
*observación*.

### 4.4 Test obligatorio

```python
def test_reexpresion_no_contamina_el_pasado():
    facts = pd.DataFrame({
        "ticker": ["A", "A"],
        "available_at": pd.to_datetime(["2024-04-25", "2025-02-14"]),
        "value": [1000.0, 940.0],
    })
    got = asof_join(pd.to_datetime(["2024-06-30", "2025-06-30"]), facts)["value"]
    assert got.loc[("2024-06-30", "A")] == 1000.0   # el valor original
    assert got.loc[("2025-06-30", "A")] == 940.0    # la reexpresión, DESDE su fecha
```

---

## 5. B3 — Sesgo de supervivencia y la lista PIT de constituyentes

### 5.1 Cómo se manifiesta

Se descarga la lista actual de los 503 tickers del S&P 500, se piden 30 años de
precios y se hace el backtest. El universo así construido está condicionado a **haber
sobrevivido y seguir en el índice hoy**: excluye por construcción a Enron, Lehman,
WorldCom, Kodak, Washington Mutual, Circuit City, General Motors (la vieja) y a las
cientos de empresas que salieron por quiebra, fusión o pérdida de relevancia.

### 5.2 Magnitud medida sobre los datos de este repositorio

Todas las cifras siguientes son **[medido]** sobre
`data/seed/sp500_historical_components.csv` (3.482 snapshots, 1996-01-02 →
2025-08-23) y `data/seed/sp500_constituents.csv` (503 filas):

| Métrica | Valor |
|---|---|
| Tickers distintos que aparecen alguna vez en el índice | **1.128** |
| Miembros en el último snapshot | 503 |
| Ratio universo histórico / universo actual | **2.24×** |
| Miembros de 1996-01-02 (469) que siguen en 2025-08-23 | **156 = 33.3%** |
| Salidas del índice por año (29 años completos, 1996-2024): media / mediana / rango | **21.7 / 20.0 / [9, 52]** |
| Tasa de salida anual sobre tamaño medio del índice (473) | **4.59%** |
| Cobertura: fracción de nombre-día PIT reales que la lista de hoy captura | **65.9%** |
| **Nombre-día falsos que asume un backtest con la lista de hoy** | **38.1%** |

Esa última línea es la que hay que interiorizar: un backtest 1996-2025 sobre la
lista actual construye **1.751.446** observaciones nombre-día, de las cuales
**667.649 (38.1%)** corresponden a empresas que en esa fecha **no estaban en el
índice**. No es un sesgo marginal: es más de un tercio de la muestra.

Calibración del efecto sobre el retorno **[simulado sobre parámetros medidos]**: con
una tasa de salida del 4.59% anual, si el nombre saliente rinde peor que el índice en
su último año en

| Bajo rendimiento del saliente en su último año | Sobrestimación del retorno equiponderado |
|---|---|
| −20% | **0.92 pp/año** |
| −30% | **1.38 pp/año** |
| −40% | **1.84 pp/año** |

El rango 0.9-1.8 pp/año coincide con lo que reporta la literatura aplicada para
backtests del S&P 500 sobre constituyentes actuales frente a pertenencia histórica
(**1-2 pp/año** [verificar]), lo cual es una comprobación de consistencia útil: el
mecanismo simple explica el orden de magnitud observado.

Y el efecto es mucho mayor en índices concentrados y en estrategias de rotación: se
ha reportado que para el Dow 30 el CAGR pasa de 11.7% con constituyentes actuales a
6.1% con series que incluyen exclusiones — **5.6 pp** [verificar].

### 5.3 Cómo se corrige en código

```python
# earnings_alpha/universe/sp500.py — membresía PIT con as-of hacia atrás
import pandas as pd
from earnings_alpha.errors import InsufficientHistory, UniverseError
from earnings_alpha.types import Ticker, normalize_ticker

class SP500Universe:
    """Pertenencia point-in-time al S&P 500 desde el fichero histórico de semilla."""

    MAX_STALENESS_DAYS = 10   # ver §5.4: el fichero no es una rejilla diaria

    def __init__(self, snapshots: pd.DataFrame) -> None:
        s = snapshots.drop_duplicates(subset="date", keep="last").sort_values("date")
        # NORMALIZACIÓN OBLIGATORIA: el fichero mezcla convenciones (§7.2)
        s["members"] = s["tickers"].str.split(",").apply(
            lambda xs: frozenset(normalize_ticker(x) for x in xs)
        )
        self._snap = s.reset_index(drop=True)

    def members_on(self, d: "date") -> list[Ticker]:
        ts = pd.Timestamp(d)
        i = self._snap["date"].searchsorted(ts, side="right") - 1
        if i < 0:
            raise InsufficientHistory(
                f"no hay snapshot de constituyentes anterior a {d}; "
                f"el primero es {self._snap['date'].iloc[0].date()}"
            )
        staleness = (ts - self._snap["date"].iloc[i]).days
        if staleness > self.MAX_STALENESS_DAYS:
            raise UniverseError(
                f"snapshot más reciente para {d} tiene {staleness} días de antigüedad "
                f"(> {self.MAX_STALENESS_DAYS}); la membresía no es fiable"
            )
        return sorted(self._snap["members"].iloc[i])
```

Los tres detalles que hacen que esto sea correcto y no decorativo:

1. **`side="right"` y `- 1`.** Selecciona el último snapshot con fecha ≤ `d`. Un
   `searchsorted` con `side="left"` o un `reindex(...).bfill()` mira al futuro.
2. **`InsufficientHistory` antes del primer snapshot.** No se extrapola hacia atrás.
3. **Tope de obsolescencia.** Ver §5.4.

### 5.4 El fichero semilla tampoco es perfecto: sesgo residual medido

Un punto de honestidad que este documento no puede omitir. El fichero histórico del
repo **no elimina** el sesgo de supervivencia, solo lo reduce. Dos defectos medidos:

**(a) Cobertura incompleta hacia atrás.** El número de miembros por snapshot
**[medido]**:

| Año | Mín | Mediana | Máx | Snapshots |
|---|---|---|---|---|
| 1996 | 468 | 468 | 469 | 106 |
| 2000 | 460 | 464 | 466 | 120 |
| 2005 | 455 | 457 | 457 | 99 |
| **2009** | **442** | **444** | 445 | 112 |
| 2015 | 463 | 467 | 471 | 117 |
| 2019 | 496 | 497 | 499 | 27 |
| 2023 | 498 | 503 | 503 | 234 |
| 2025 | 502 | 503 | 503 | 222 |

El S&P 500 siempre ha tenido ~500 componentes. El fichero lista **442-468** en el
periodo 1996-2015, con el mínimo en 2009. Faltan entre **6% y 12%** de los miembros
reales, y faltan precisamente los que sufrieron eventos corporativos que el registro
de cambios no capturó: fusiones y quiebras del ciclo 2007-2009. Es decir, el sesgo
residual **apunta en la misma dirección** que el sesgo original: sobrerrepresenta a
los supervivientes. La forma en U de la cobertura (buena hoy, peor cuanto más atrás,
mínima en 2009) es la firma característica de una historia **reconstruida hacia
atrás** desde la lista actual aplicando un registro incompleto de cambios.

Consecuencia práctica: no se puede afirmar "este backtest está libre de sesgo de
supervivencia". Se puede afirmar "este backtest usa membresía PIT con una cobertura
del X%", y hay que **reportar la cobertura**:

```python
# earnings_alpha/universe/sp500.py
EXPECTED_SIZE = 500

def coverage_report(self) -> pd.DataFrame:
    """Diagnóstico de cobertura por año. Un backtest cuya cobertura media caiga por
    debajo del umbral debe reportarlo en el tearsheet, no silenciarlo."""
    n = self._snap["members"].apply(len)
    df = pd.DataFrame({"date": self._snap["date"], "n": n})
    out = df.groupby(df["date"].dt.year)["n"].agg(["min", "median", "max", "count"])
    out["coverage"] = out["median"] / EXPECTED_SIZE
    return out
```

**(b) La rejilla no es diaria.** Los 3.482 snapshots cubren 29.6 años (≈7.450
sesiones). Distribución de huecos entre snapshots consecutivos **[medido]**: mediana
1 día, media 3.11, p90 7 días, **máximo 88 días** (hueco que termina el 2020-09-18).
En 2019-2022 la frecuencia cae drásticamente (13-27 snapshots por año). Por eso
`members_on` debe:

- hacer as-of **hacia atrás** (nunca `reindex` + `dropna`, que borraría el 70% de las
  fechas, ni `bfill`, que es look-ahead puro),
- **medir y exponer la obsolescencia**: un cambio de índice ocurrido dentro de un
  hueco de 88 días se incorpora al universo hasta 88 días tarde. En ese intervalo se
  negocian nombres que ya habían salido y se ignoran los que ya habían entrado.
  Ninguna de las dos cosas es look-ahead (el error es hacia el pasado, no hacia el
  futuro), pero sí es ruido de universo que hay que declarar.

**(c) Corolario para el módulo `universe`.** El `refresh()` *append-only* del
contrato es la respuesta correcta a largo plazo: a partir de hoy, cada día que la
plataforma se ejecute registra la membresía real de ese día, y esa parte del
histórico sí será PIT sin reconstrucción. La historia pre-2026 arrastra el sesgo
residual medido arriba y hay que tratarla como tal.

---

## 6. B4 — Sesgo de delisting y el retorno de delisting

### 6.1 Cómo se manifiesta

Aunque el universo sea PIT, si la serie de precios de una empresa **termina** el día
antes de que la acción deje de cotizar, el backtest nunca registra la pérdida final.
La posición simplemente "desaparece" de la cartera al último precio observado. Es el
sesgo hermano del de supervivencia y sobrevive a su corrección: se puede tener una
lista PIT impecable y aun así no perder nunca dinero en las quiebras.

Casos donde ocurre:

- **Quiebra / liquidación**: la acción deja de cotizar y el accionista recibe cero o
  casi cero. El último precio observado suele ser ya muy bajo, pero la caída final
  desde ese precio no está en la serie.
- **Exclusión por incumplimiento de requisitos de cotización** (precio mínimo,
  capitalización): el paso al mercado OTC implica una caída y un salto de liquidez.
- **Fusión en efectivo**: aquí el retorno final es *positivo y conocido* (el precio
  de la oferta). Omitirlo también sesga, en la otra dirección.
- **Fusión en acciones**: hay que continuar la posición en el adquirente con la
  ratio de canje.

### 6.2 Cuánto infla

El trabajo fundacional es Shumway (1997): los retornos de delisting correctos no
están disponibles para la mayoría de las acciones excluidas por razones negativas
desde 1962; los que faltan son **mucho más frecuentes cuando la exclusión es por mal
rendimiento** y son **grandes y negativos** en media [lit]. Las magnitudes de
imputación que se han vuelto estándar:

| Mercado | Retorno de delisting a imputar cuando falta y la baja es por rendimiento |
|---|---|
| NYSE / AMEX | **−30%** (Shumway 1997) [verificar en el original] |
| Nasdaq | **−55%** (Shumway & Warther 1999) [lit] |

Shumway & Warther (1999) encuentran además que el sesgo en datos Nasdaq es **4.7
veces mayor** que el documentado para NYSE/AMEX, y que **tras corregirlo desaparece
la evidencia del efecto tamaño en Nasdaq** [lit]. Es el ejemplo canónico de una
anomalía entera que era un artefacto de datos.

Para el S&P 500 el efecto es menor que en small caps (las bajas por precio mínimo son
raras), pero **no es despreciable**: 21.7 salidas al año de media [medido], con picos
de **52 en 2000**, 38 en 1999, 36 en 2007 y 32 en 2008 [medido] — es decir, la
rotación se dispara justo en los años en que el retorno de delisting más importa. Las
que salen por quiebra o rescate (Enron, Lehman, WaMu, Ambac, Delphi) son exactamente
las que un factor de valor o de calidad tendría en cartera.

### 6.3 Cómo se corrige en código

```python
# earnings_alpha/data/prices.py — cierre de posición con retorno de delisting
from enum import StrEnum

class DelistReason(StrEnum):
    MERGER_CASH = "merger_cash"
    MERGER_STOCK = "merger_stock"
    BANKRUPTCY = "bankruptcy"
    LISTING_REQUIREMENTS = "listing_requirements"
    INDEX_REMOVAL_ONLY = "index_removal_only"   # sigue cotizando: NO es delisting
    UNKNOWN = "unknown"

_PERFORMANCE_RELATED = {DelistReason.BANKRUPTCY, DelistReason.LISTING_REQUIREMENTS}
_FALLBACK_DLRET = {"nyse": -0.30, "amex": -0.30, "nasdaq": -0.55}

def delisting_return(
    last_price: float,
    final_value: float | None,
    reason: DelistReason,
    exchange: str,
) -> float:
    """Retorno de la última barra de una serie que termina en exclusión.

    `final_value` es el valor recibido por acción (precio de la oferta en una
    fusión en efectivo, valor de liquidación, primer precio en el mercado OTC de
    destino). Si falta y la baja es por rendimiento, se imputa el valor de
    Shumway (1997) / Shumway & Warther (1999) en vez de dejar la posición
    evaporarse a coste cero, que es el sesgo.
    """
    if final_value is not None:
        return final_value / last_price - 1.0
    if reason in _PERFORMANCE_RELATED:
        return _FALLBACK_DLRET.get(exchange.lower(), -0.30)
    if reason is DelistReason.MERGER_STOCK:
        raise DataQualityError(
            "fusión en acciones sin ratio de canje: la posición debe continuar en el "
            "adquirente, no cerrarse"
        )
    raise DataQualityError(
        f"exclusión de {reason!r} sin valor final conocido; imputar un retorno "
        f"arbitrario aquí sesgaría el backtest en dirección desconocida"
    )
```

Puntos no negociables del diseño:

1. **La razón de la exclusión es obligatoria.** Imputar −30% a una fusión en efectivo
   con prima del +25% es tan sesgado como no imputar nada a una quiebra.
2. **`INDEX_REMOVAL_ONLY` no es una exclusión.** Salir del S&P 500 por reducción de
   capitalización no implica dejar de cotizar. Confundirlas mete pérdidas ficticias.
3. **Sin razón conocida se lanza, no se adivina.** Regla de oro n.º 5 del proyecto.
4. **La barra de delisting se ejecuta.** El backtest debe procesar ese retorno con la
   posición que tuviera, no descartar el nombre el día anterior.

### 6.4 Detección: los nombres que "desaparecen" sin cerrar

```python
def audit_silent_disappearances(prices: pd.DataFrame, universe: SP500Universe) -> pd.DataFrame:
    """Nombres cuya serie de precios termina sin una barra de delisting explícita.

    Cada fila del resultado es una posición que el backtest cerraría a coste cero:
    exactamente el sesgo de delisting. Debe ser una lista vacía antes de producción.
    """
    last_px = prices.groupby("ticker")["date"].max()
    last_member = ...  # última fecha en que el ticker estuvo en el índice
    gap = (last_member - last_px).dt.days
    return gap[gap > 1].sort_values(ascending=False).to_frame("dias_sin_precio")
```

---

## 7. B5 — Identificadores: tickers, CIK, fusiones y spin-offs

### 7.1 El problema general

El ticker **no es un identificador**. Cambia, se recicla y se reasigna. Los
identificadores permanentes de la industria (PERMNO de CRSP, GVKEY de Compustat)
existen precisamente porque *no cambian durante la vida de la emisión ni se
reasignan* [lit]. El repo no tiene acceso a CRSP; su identificador más estable
disponible es el **CIK de la SEC**, que es persistente por emisor pero:

- **no cubre** empresas extranjeras sin registro en la SEC (relevante para ADR),
- **no resuelve** clases de acciones (una empresa, un CIK, varios tickers),
- **cambia de asignación efectiva** en fusiones: la entidad superviviente conserva
  su CIK y la absorbida deja de presentar.

### 7.2 Lo que pasa en los datos de este repositorio [medido]

Tres patologías reales, medidas sobre el fichero histórico de constituyentes:

**(a) La convención de tickers cambia a mitad del fichero.** `BRK-B` aparece del
**2010-02-16 al 2023-09-25**; `BRK.B` aparece del **2023-05-09 al 2025-08-23**. Lo
mismo con `BF-B` / `BF.B` (esta última solo en 3 snapshots de mayo de 2023). No hay
ninguna fecha con las dos formas simultáneas [medido], así que no hay duplicados,
pero **sí hay una discontinuidad**: un join por ticker crudo rompe la serie de
Berkshire el 2023-05-09 y vuelve a romperla el 2023-09-26. El factor de momentum de
esos nombres tendría un hueco de varios meses, y el backtest lo interpretaría como
salida y reentrada al índice.

La corrección es la que ya prescribe `types.normalize_ticker` (punto, no guion), y es
**obligatoria en la carga**, no en el consumo:

```python
members = frozenset(normalize_ticker(x) for x in row["tickers"].split(","))
```

Comprobación **[medido]**: tras normalizar, **cero snapshots** contienen un ticker
duplicado. La normalización es segura en este dataset.

**(b) Identificadores retroactivos: la historia lleva el ticker del final.** El
fichero etiqueta a cada empresa con su **último** ticker conocido durante toda su
vida. Ejemplos medidos:

| Ticker en el fichero | Primera aparición | Última | Ticker real en esas fechas |
|---|---|---|---|
| `ENRNQ` | 1996-01-02 | 2001-11-26 | Enron cotizaba como `ENE` hasta la quiebra |
| `MTLQQ` | 1996-01-02 | 2009-05-29 | General Motors cotizaba como `GM`; `MTLQQ` es Motors Liquidation, la carcasa post-quiebra |
| `LEHMQ` | 1998-01-12 | 2008-09-16 | Lehman Brothers cotizaba como `LEH` |
| `EKDKQ` | 1996-01-02 | 2010-12-17 | Eastman Kodak cotizaba como `EK` |
| `AAMRQ` | 1996-01-02 | 2003-03-10 | AMR Corp cotizaba como `AMR` |
| `WAMUQ` | 1997-07-02 | 2008-09-26 | Washington Mutual cotizaba como `WM` |

Esto tiene **dos consecuencias graves**:

1. **El join con precios falla silenciosamente.** Ningún proveedor sirve precios de
   1996 bajo el símbolo `ENRNQ`. El nombre simplemente no aparecerá en el panel, y el
   universo PIT quedará *de facto* mutilado precisamente en los nombres que
   quebraron — reintroduciendo el sesgo de supervivencia por la puerta de atrás,
   después de haberlo corregido por la puerta principal.
2. **El propio identificador filtra el futuro.** El sufijo `Q` significa "en
   procedimiento concursal". En el snapshot de 1996 la etiqueta `ENRNQ` ya contiene
   la información de que esa empresa acabará quebrando. Un modelo que use como
   feature cualquier cosa derivada del texto del ticker (longitud, sufijo,
   agrupaciones) está mirando al futuro. Suena esotérico; deja de sonarlo en cuanto
   alguien codifica el ticker como variable categórica en un modelo de ML.

Cuidado con la heurística del sufijo: hay **27 tickers de ≥4 letras terminados en Q**
[medido], y entre ellos hay falsos positivos obvios — `HPQ` (Hewlett-Packard),
`CPQ` (Compaq), `LKQ` (LKQ Corp), `NDAQ` (Nasdaq), `MERQ` (Mercury Interactive). El
sufijo Q es una pista para auditar, no un clasificador.

**(c) Reutilización cruzada de símbolos.** `GM` aparece **desde 2013-06-07** (la
nueva General Motors, tras la reestructuración) mientras que la vieja GM aparece como
`MTLQQ` hasta 2009. Son dos entidades distintas con historias distintas. Un panel
indexado por ticker que concatene ambas produce una serie de precios con un salto
imposible.

### 7.3 Cómo se corrige en código

```python
# earnings_alpha/universe/identity.py — mapa PIT de identidad
@dataclass(frozen=True, slots=True)
class SecurityIdentity:
    """Identidad estable de una emisión, independiente del ticker vigente."""
    security_id: str          # clave interna estable: f"{cik}:{share_class}"
    cik: CIK | None
    share_class: str = "common"

@dataclass(frozen=True, slots=True)
class TickerMapping:
    """Un ticker vigente para una identidad durante un intervalo cerrado por la izquierda."""
    security_id: str
    ticker: Ticker
    valid_from: date
    valid_to: date | None     # None = vigente

class IdentityMap:
    def resolve(self, ticker: Ticker, on: date) -> str:
        """security_id del ticker EN esa fecha. Lanza UniverseError si el símbolo
        estaba asignado a otra entidad o a ninguna en `on`."""

    def ticker_on(self, security_id: str, on: date) -> Ticker:
        """Símbolo con el que cotizaba esa identidad en `on`. Es el que hay que pedir
        al proveedor de precios para esa fecha."""
```

Reglas de eventos corporativos:

| Evento | Tratamiento correcto |
|---|---|
| **Cambio de ticker** | Mismo `security_id`, se cierra el `TickerMapping` anterior y se abre uno nuevo. La serie de retornos **no se corta**. |
| **Cambio de CIK** (reincorporación, redomiciliación) | Mismo `security_id`, se registra el CIK nuevo con su `valid_from`. Los filings antiguos siguen bajo el CIK antiguo: el proveedor de EDGAR debe consultar **ambos**. |
| **Fusión en efectivo** | La identidad absorbida termina. Retorno final = precio de oferta (§6.3). No se continúa la serie. |
| **Fusión en acciones** | La identidad absorbida termina; la posición **continúa** en el adquirente con la ratio de canje. El retorno del día de canje es `(ratio × P_adq) / P_abs − 1`. |
| **Spin-off** | La matriz **continúa**; el día ex el precio cae por el valor de la escindida y ese salto **no es un retorno negativo**: hay que ajustarlo o acreditar la posición en la escindida. La escindida es una identidad **nueva**, sin historia. |
| **Split / split inverso** | Solo ajuste de precio y volumen. Nunca un retorno. |
| **Cambio de clase / recapitalización** | Puede crear identidades nuevas (p. ej. Google → GOOG/GOOGL). Se resuelve por `share_class`. |

Regla crítica sobre los spin-offs y el ángulo A: la escindida **no tiene historia
fundamental propia**. Sus estados financieros *pro forma* retroactivos existen, pero
no eran públicos antes del spin-off. Un factor que requiera 8 trimestres de sorpresas
debe lanzar `InsufficientHistory` para ese nombre hasta que los tenga **realmente
publicados**, no *pro forma*.

### 7.4 Test obligatorio

```python
def test_identidad_no_confunde_gm_vieja_y_nueva():
    m = IdentityMap.from_seed()
    assert m.resolve("GM", date(2005, 1, 3)) != m.resolve("GM", date(2015, 1, 5))

def test_normalizacion_elimina_la_discontinuidad_de_brk():
    u = SP500Universe.from_seed()
    antes = set(u.members_on(date(2023, 5, 5)))
    despues = set(u.members_on(date(2023, 5, 10)))
    assert "BRK.B" in antes and "BRK.B" in despues     # sin normalizar fallaría
    assert "BRK-B" not in antes and "BRK-B" not in despues
```

---

## 8. B6 — Sesgo de selección en el calendario de resultados

### 8.1 Cómo se manifiesta

Los proveedores de calendario mezclan dos tipos de fecha:

- **Confirmada**: la empresa ha anunciado públicamente cuándo publicará.
- **Estimada**: el proveedor la deduce del patrón histórico ("el trimestre pasado
  publicó el tercer martes, así que este también").

Un backtest de eventos que no distinga ambos casos comete tres errores distintos:

1. **Look-ahead por revisión de la fecha.** Si el proveedor sirve el calendario "tal
   como está hoy", todas las fechas están confirmadas *a posteriori*. La estrategia
   asume que el 1 de julio ya sabía que la empresa publicaría el 28 de julio, cuando
   en realidad esa fecha se confirmó el 14 de julio. Toda estrategia de
   pre-posicionamiento (`entry_offset` negativo en `EventBacktest`, y todas las
   features de `PreEventFeatures`, que se calculan en `[T-N, T-1]`) queda contaminada:
   el ancla `T` no era conocible en `T-N`.
2. **Sesgo de selección al filtrar.** Si se descartan los eventos con fecha estimada,
   se descarta un subconjunto **no aleatorio**. Las empresas que no confirman con
   antelación tienden a ser las que tienen algo que decidir todavía (§10). Filtrar por
   "solo confirmadas" tira sistemáticamente los eventos malos.
3. **Errores de fecha en la fuente.** Incluso entre las confirmadas hay discrepancias
   entre proveedores. La convención de la literatura es aceptar hasta **un día
   natural** de diferencia entre fuentes distintas y tratar diferencias mayores como
   error de datos [lit]. Un día de diferencia, sin embargo, es exactamente el error
   que §9 muestra que es letal.

### 8.2 Cuánto pesa

- A octubre de 2025, solo el **62%** de los ~11.000 nombres del universo global de un
  proveedor especializado tenía la fecha **confirmada** [lit]. En el S&P 500 el
  porcentaje es sensiblemente mayor, pero en ningún momento es 100%: hay una ventana
  de semanas en la que buena parte del calendario es estimación.
- Los proveedores de fundamentales tienen su propia patología: la fecha de anuncio a
  veces procede de un diario (que la publica **al día siguiente**) y a veces del wire
  (mismo día) [lit], lo que introduce un error de un día **no aleatorio** que
  correlaciona con el tamaño y la época de la muestra.
- Filtros de saneamiento que la literatura aplica de forma estándar [lit], y que la
  plataforma debe implementar: descartar el evento si hay más de un anuncio del mismo
  emisor en la misma fecha, si dista menos de 30 días del anterior, o si cae antes de
  `period_end` o más de 180 días después.

### 8.3 Cómo se corrige en código

El tipo `EarningsEvent` ya lleva el campo necesario:

```python
is_estimated_date: bool = False
```

La disciplina es **guardar el calendario con vintages**, igual que los fundamentales:

```python
# earnings_alpha/data/estimates.py
@dataclass(frozen=True, slots=True)
class CalendarVintage:
    """El calendario TAL COMO ESTABA en `as_of`. Sin esto, ninguna feature
    pre-evento es PIT, porque el ancla temporal del evento no era conocible."""
    ticker: Ticker
    period_end: date
    expected_date: date
    session: Session
    is_estimated: bool
    as_of: date              # cuándo se observó ESTA versión del calendario

def known_event_date(vintages: pd.DataFrame, ticker: Ticker,
                     period_end: date, as_of: date) -> tuple[date, bool]:
    """Fecha de resultados que un operador conocía en `as_of`, y si era estimada.

    Es la función que hace PIT a todo el ángulo B: `PreEventFeatures` mide la ventana
    [T-N, T-1] y por tanto necesita saber cuál era el `T` *esperado* en cada día de
    esa ventana, no el `T` realizado.
    """
    sel = vintages[(vintages.ticker == ticker)
                   & (vintages.period_end == period_end)
                   & (vintages.as_of <= as_of)]
    if sel.empty:
        raise InsufficientHistory(
            f"no hay vintage de calendario para {ticker} {period_end} en {as_of}"
        )
    row = sel.sort_values("as_of").iloc[-1]
    return row.expected_date, bool(row.is_estimated)
```

**Política de la plataforma (obligatoria en `EventBacktest`):**

1. Si `entry_offset < 0` (posicionamiento pre-evento) el backtest **exige** vintages
   de calendario. Sin ellos lanza `LookAheadError`. No hay modo degradado.
2. Si `entry_offset >= 0` (entrada tras el anuncio) basta con la fecha realizada: el
   evento ya ha ocurrido y es de dominio público.
3. `is_estimated_date` **no se usa como filtro por defecto**. Se reporta como
   estratificación: el tearsheet muestra el resultado sobre confirmadas, sobre
   estimadas y sobre todas. Si la señal solo funciona en el subconjunto de
   confirmadas, eso es un hallazgo, no una limpieza.
4. La revisión de la fecha esperada (`expected_date` de hoy vs. la de hace 5 días) es
   **una feature legítima**, PIT y potencialmente informativa (§10).

---

## 9. B7 — La trampa BMO/AMC y el gap overnight

### 9.1 Cómo se manifiesta

Es el error de un día más caro del proyecto y el que motiva que
`pit.tradable_date` sea, según el contrato, "la función más crítica del repo".

- Un anuncio **BMO** (antes de la apertura) es negociable el **mismo día**: hay una
  sesión completa por delante.
- Un anuncio **AMC** (tras el cierre) **no** es negociable ese día. La primera
  ejecución posible es la **apertura siguiente**, y la apertura siguiente ya
  incorpora la noticia: el movimiento ocurre en el **gap** close→open, sin que haya
  habido oportunidad de operar.

Si el backtest supone BMO cuando en realidad era AMC, entra "al cierre del día del
anuncio" a un precio que **todavía no conocía la noticia**, y luego cobra el gap.
Eso no es una estrategia: es una máquina del tiempo.

### 9.2 Cuánto infla

**[simulado]**, con los mismos parámetros de §2.2 (salto de resultados σ = 5.5%,
~20 eventos simultáneos en cartera, 4 eventos/año/nombre), variando la fracción del
movimiento que ocurre en el gap y la habilidad *real* de la señal (correlación
ex-ante con el salto):

| Fracción en el gap | Habilidad real | Sharpe correcto | Sharpe con el error | Inflación | Retorno/evento correcto | Retorno/evento con error |
|---|---|---|---|---|---|---|
| 0.60 | 0.05 | 0.32 | 0.36 | 1.1× | 0.09% | 0.23% |
| 0.60 | 0.20 | 1.21 | 1.41 | 1.2× | 0.36% | 0.89% |
| 0.75 | 0.05 | 0.26 | 0.36 | 1.4× | 0.06% | 0.23% |
| 0.75 | 0.20 | 0.99 | 1.41 | 1.4× | 0.22% | 0.89% |
| **0.90** | **0.05** | **0.14** | **0.36** | **2.6×** | **0.02%** | **0.23%** |
| **0.90** | **0.20** | **0.51** | **1.41** | **2.8×** | **0.09%** | **0.89%** |

La lectura es doble y las dos partes importan:

1. **La inflación del Sharpe crece con la fracción del movimiento que ocurre en el
   gap.** Y esa fracción es alta: el consenso empírico es que el grueso de la
   reacción a resultados se materializa en la reapertura, con los primeros minutos
   concentrando el mayor volumen y el mayor movimiento del día [lit]. Con una
   fracción de 0.90 el Sharpe se multiplica por **2.6-2.8** y el retorno por evento
   por **4-10**.
2. **Y en la dirección contraria: la estrategia correcta gana mucho menos de lo que
   parece.** Con el 90% del movimiento en el gap, una señal con habilidad real 0.20
   —que es mucha— rinde solo 0.09% por evento operable. Buena parte del PEAD
   "documentado" en papers que miden desde el cierre previo **no es capturable**.
   Esto conecta con el resultado de que los beneficios del PEAD se reducen
   drásticamente o desaparecen tras costes de transacción [lit].

**Corolario para `EventBacktest`:** el motor **debe** descomponer el retorno del
evento en gap (close→open) y sesión (open→close), y reportarlos por separado. Un
tearsheet de eventos que solo dé retornos close→close es inauditable.

### 9.3 Cómo se corrige en código

```python
# earnings_alpha/pit/calendar.py — la función más crítica del repo
from datetime import date, datetime, time
from zoneinfo import ZoneInfo
from earnings_alpha.types import Session

NY = ZoneInfo("America/New_York")

def tradable_date(announced_at: datetime, session: Session, cal: "TradingCalendar") -> date:
    """Primera sesión bursátil en que la información del anuncio es explotable.

    Reglas:
      BMO -> la sesión del mismo día (o la siguiente si ese día no es sesión).
      DMH -> la sesión del mismo día (se puede operar el resto de la sesión).
      AMC -> la sesión SIGUIENTE.
      UNKNOWN -> se trata como AMC. Es la política conservadora del repo
                 (types.Session.UNKNOWN): equivocarse hacia tarde cuesta alfa;
                 equivocarse hacia pronto invalida el backtest.

    `announced_at` DEBE llevar tzinfo. La fecha se toma en horario de Nueva York,
    nunca en UTC: un anuncio de las 20:30 ET cae en el día natural UTC siguiente y
    la conversión ingenua desplaza el evento una sesión entera.
    """
    if announced_at.tzinfo is None:
        raise ValueError("announced_at debe llevar tzinfo; sin zona no hay PIT")
    d = announced_at.astimezone(NY).date()
    if session in (Session.BMO, Session.DMH):
        return cal.session_on_or_after(d)
    return cal.next_session(d)          # AMC y UNKNOWN
```

**Casos verificados [medido]** (calendario NYSE 2025 con festivos):

| Anuncio | Sesión | `tradable_date` | Comentario |
|---|---|---|---|
| jue 2025-01-30 16:05 ET | AMC | **2025-01-31** | siguiente sesión |
| vie 2025-01-31 07:00 ET | BMO | **2025-01-31** | mismo día |
| vie 2025-01-31 16:10 ET | AMC | **2025-02-03** | salta el fin de semana |
| sáb 2025-02-01 07:00 ET | BMO | **2025-02-03** | día no hábil → siguiente sesión |
| lun 2025-03-10 16:05 ET | AMC | **2025-03-11** | correcto tras el cambio horario |
| **2025-01-31 01:30 UTC** (= jue 30-ene 20:30 ET) | AMC | **2025-01-31** | correcto |
| ídem, tomando la **fecha UTC** | AMC | **2025-02-03** | **ERROR: dos sesiones de retraso** |

El último par es la trampa que hay que interiorizar: `announced_at` se almacena en
UTC (contrato de `types.EarningsEvent`), pero **la fecha de sesión debe calcularse en
horario de Nueva York**. Un anuncio nocturno ET cae en el día natural UTC siguiente.
En este caso el error es hacia tarde (pierdes la sesión del 31 de enero y entras el
lunes), lo cual "solo" destruye alfa; pero el error simétrico —tratar un anuncio
temprano de la mañana ET como si fuera del día anterior— sería look-ahead directo.

### 9.4 Los cuatro casos borde que hay que codificar

1. **Sesiones de media jornada.** El día después de Acción de Gracias, Nochebuena y
   el 3 de julio el cierre es a las **13:00 ET**. Un anuncio a las 13:30 ET de esos
   días es **AMC**, no DMH. La clasificación BMO/AMC/DMH debe consultar el **horario
   de cierre de esa sesión concreta**, no un 16:00 constante.
2. **`Session.UNKNOWN`.** La política del repo es tratarlo como AMC. Debe estar
   implementada explícitamente y cubierta por un test, porque el valor por defecto
   de un campo no documentado acaba siendo BMO por accidente en algún proveedor.
3. **DMH.** Los anuncios durante la sesión son ~**4.3%** de la muestra [verificar].
   Son raros y suelen ser filtraciones o publicaciones no programadas. Son
   negociables el mismo día pero el retorno close-to-close del día contiene una parte
   pre-anuncio: el motor debe poder excluirlos o tratarlos aparte.
4. **Horario de verano.** La conversión ET↔UTC cambia de offset dos veces al año. Se
   resuelve usando `zoneinfo` con la zona `America/New_York`, jamás un offset fijo de
   −5 o −4 horas.

### 9.5 Modelado del gap en el motor de eventos

```python
# earnings_alpha/backtest/event.py — descomposición obligatoria
def decompose_event_return(bars: pd.DataFrame, t0: date, cal) -> dict[str, float]:
    """Descompone el retorno del evento en gap y sesión.

    `t0` es el `tradable_date`. Para un anuncio AMC:
      gap      = open(t0)  / close(t0 - 1 sesión) - 1   <- NO capturable
      intraday = close(t0) / open(t0)             - 1   <- capturable con orden a la apertura
    Para un anuncio BMO la descomposición es la misma, pero el gap SÍ es
    parcialmente capturable si se estaba posicionado desde el cierre anterior con
    información previa al anuncio.
    """
    prev = cal.shift(t0, -1)
    c_prev = bars.loc[(prev, "close")]
    o0, c0 = bars.loc[(t0, "open")], bars.loc[(t0, "close")]
    return {"gap": o0 / c_prev - 1.0, "intraday": c0 / o0 - 1.0, "total": c0 / c_prev - 1.0}
```

Y la regla de ejecución: **una señal derivada de un anuncio AMC no puede ejecutarse
al cierre del día del anuncio ni a la apertura teórica sin deslizamiento.** El
`CostModel` debe aplicar en la apertura post-evento un spread ampliado: la horquilla
en los primeros minutos tras un anuncio de resultados es un múltiplo de la normal.

---

## 10. B8 — Sesgo de anuncio tardío: las malas noticias llegan tarde y en viernes

### 10.1 Cómo se manifiesta

El calendario de resultados no es exógeno. La dirección elige la fecha, y la elección
está correlacionada con el contenido. Dos regularidades documentadas:

**(a) *Bad news late*.** Las empresas que retrasan su anuncio respecto a la fecha
esperada tienden a publicar peores noticias. Begley y Fischer (1998) encuentran que
las que adelantan (retrasan) publican mejores (peores) noticias; Bagnoli, Kross y
Watts (2002), usando como fecha esperada la **estimación de la propia dirección**,
encuentran evidencia sólida de la hipótesis *bad-news-late* y mucho más débil de
*good-news-early* [lit]. El patrón que documentan —"un día tarde, un centavo de
menos"— es que el mercado castiga a la acción **el día en que se incumple la fecha
esperada**, y sigue castigándola a medida que el retraso se alarga [lit].

**(b) Efecto viernes / inatención del inversor.** DellaVigna y Pollet (2009)
encuentran que los anuncios en viernes tienen una respuesta inmediata **entre un 15%
y un 20% menor** y una respuesta diferida **entre un 60% y un 70% mayor**; el
componente diferido supone el **60%** de la respuesta total en viernes frente al
**40%** en el resto de días [lit].

### 10.2 Por qué es un sesgo y no solo un fenómeno

Tiene tres efectos distintos sobre un backtest, y conviene no mezclarlos:

1. **Sesgo de composición de la muestra.** Si el estudio de eventos filtra por
   cualquier criterio correlacionado con la fecha (p. ej. "solo fechas confirmadas
   con ≥10 días de antelación", §8), el filtro elimina desproporcionadamente los
   eventos malos. El CAR medio de la muestra sube sin que ninguna señal haya
   mejorado.
2. **Look-ahead si la feature de retraso está mal fechada.** "Días de retraso
   respecto a lo esperado" es una feature valiosa **y PIT**, pero solo si
   `expected_date` procede del vintage de calendario vigente en cada día de la
   ventana (§8.3). Si se calcula contra la fecha realizada, la feature es
   idénticamente cero y no dice nada; si se calcula contra la fecha esperada *de
   hoy*, es look-ahead.
3. **Heterogeneidad del horizonte de drift.** El PEAD de un anuncio en viernes es más
   lento. Un backtest con `exit_offset` fijo mide un drift mezclado. La estratificación
   por día de la semana no es cosmética: cambia el horizonte óptimo de salida.

### 10.3 Cómo se corrige — y cómo se explota — en código

```python
# earnings_alpha/events/timing.py
def announcement_timing_features(
    ev: EarningsEvent, vintages: pd.DataFrame, cal: "TradingCalendar", asof: date,
) -> dict[str, float | bool]:
    """Features PIT del *timing* del anuncio, medidas en `asof` (día de la ventana
    pre-evento), no con información posterior.

    Referencias:
      Begley & Fischer (1998); Bagnoli, Kross & Watts (2002) — bad news late.
      DellaVigna & Pollet (2009, JF) — inatención del inversor y anuncios en viernes.
    """
    expected, is_est = known_event_date(vintages, ev.ticker, ev.period_end, asof)
    # referencia histórica del propio emisor: retardo mediano de los 8 trimestres
    # ANTERIORES (nunca del actual)
    hist = _historical_lag_days(ev.ticker, ev.period_end, n=8)   # lanza InsufficientHistory
    normal_date = cal.shift(ev.period_end, int(round(hist.median())))
    return {
        "expected_lag_days": (expected - ev.period_end).days,
        "delay_vs_history_days": cal.session_distance(normal_date, expected),
        "date_revised_later": cal.session_distance(normal_date, expected) > 0,
        "is_estimated_date": is_est,
        "is_friday": expected.weekday() == 4,
        "is_after_thanksgiving_week": ...,
    }
```

Reglas:

- `delay_vs_history_days` usa la mediana de los **8 trimestres anteriores**; con
  menos historia se lanza `InsufficientHistory` (regla de oro n.º 5), no se calcula
  con 2 observaciones.
- `is_friday` se evalúa sobre la fecha **esperada** en `asof`, no sobre la realizada.
- El motor de eventos estratifica por `is_friday` y por `delay_vs_history_days` y
  reporta el drift por estrato. Si el alfa de una señal desaparece al controlar por
  timing, es que la señal era un proxy del timing.

---

## 11. B9 — Supervivencia y reescritura en los datos de consenso

### 11.1 Cómo se manifiesta

El consenso de analistas es el input más frágil de todo el proyecto, porque la base
de datos de estimaciones es un objeto **vivo que se reescribe**. Tres patologías
distintas:

**(a) Reescritura retroactiva del histórico.** Ljungqvist, Malloy y Marston (2009)
compararon **siete descargas completas** de la base de recomendaciones de I/B/E/S
entre 2000 y 2007. Entre el **1.6% y el 21.7%** de las observaciones emparejadas
diferían de una descarga a la siguiente [lit]. Los cambios incluyen alteraciones del
nivel de la recomendación, adiciones y **borrados** de registros, y eliminación de
nombres de analistas. Y no son aleatorios: se agrupan **por reputación del analista,
tamaño y estatus de la casa de bolsa, y audacia de la recomendación** [lit]. El
impacto sobre los resultados es material: cambian la clasificación de las señales, la
rentabilidad de las estrategias basadas en cambios de consenso y la persistencia
medida de la habilidad individual de los analistas [lit].

La implicación es brutal: **una descarga hecha hoy no reproduce lo que un operador
veía en 2015**, ni siquiera para los registros que sobreviven.

**(b) Supervivencia por abandono de cobertura.** Los analistas dejan de cubrir
precisamente los valores que se deterioran, cuando proyectar beneficios se vuelve
difícil tras un cambio de fortuna a peor [lit]. Un panel construido con "empresas con
consenso disponible" excluye desproporcionadamente a las empresas en problemas. Es
el mismo mecanismo que el sesgo de supervivencia del universo, pero opera **dentro**
de un universo ya corregido: aunque el nombre esté en el índice, si no tiene
consenso, se cae de la muestra.

**(c) Definición móvil del "actual".** El EPS "actual" que sirve la base de
estimaciones es un EPS **ajustado**, calculado con la misma base que las
estimaciones (excluyendo extraordinarios según el criterio del proveedor). Ese
criterio cambia, y el "actual" de un trimestre antiguo puede ser recalculado años
después. Comparar un `eps_actual` recalculado con un `eps_estimate` de época produce
una sorpresa que nadie observó.

### 11.2 Cuánto pesa

- **1.6%-21.7%** de registros alterados entre descargas consecutivas [lit]. En una
  serie de 10 años con revisiones acumuladas, la fracción de la historia que difiere
  de lo que se veía en su momento no es marginal.
- Los efectos son **no aleatorios** y concentrados en las recomendaciones más audaces
  y los analistas más reputados [lit], que son exactamente el subconjunto que
  cualquier señal de revisiones intenta explotar.
- Para el ángulo A esto afecta directamente al factor de **momentum de revisiones de
  analistas**, y para el ángulo B al `eps_estimate` que entra en el denominador del
  SUE y en `PreEventFeatures.analyst_revision_drift`.

### 11.3 Cómo se corrige en código

El tipo del repo ya está diseñado para esto: `EstimateSnapshot` lleva `as_of`. La
disciplina:

```python
# earnings_alpha/data/estimates.py
class EstimatesStore:
    """Almacén append-only de fotos de consenso. Nunca sobrescribe una foto previa.

    La única forma de tener consenso PIT sin una base histórica comercial es
    *empezar a fotografiar hoy y no parar*. Toda estimación reconstruida hacia
    atrás desde una descarga actual está sujeta a la reescritura documentada por
    Ljungqvist, Malloy y Marston (2009).
    """

    def snapshot(self, tickers: list[Ticker], as_of: date) -> int:
        """Guarda el consenso vigente hoy con `as_of=hoy`. Idempotente por día."""

    def consensus_at(self, ticker: Ticker, period_end: date, as_of: date) -> EstimateSnapshot:
        """Consenso que un operador veía en `as_of`. Lanza InsufficientHistory si no
        hay ninguna foto anterior a esa fecha: NUNCA devuelve la foto actual."""
        sel = self._df[(self._df.ticker == ticker)
                       & (self._df.period_end == period_end)
                       & (self._df.as_of <= as_of)]
        if sel.empty:
            raise InsufficientHistory(
                f"no hay foto de consenso de {ticker}/{period_end} anterior a {as_of}; "
                f"usar el consenso actual sería look-ahead (Ljungqvist et al., 2009)"
            )
        return _to_snapshot(sel.sort_values("as_of").iloc[-1])

    def coverage_panel(self) -> pd.DataFrame:
        """Panel booleano (date, ticker) de si HABÍA consenso. Necesario para
        distinguir 'sin consenso' de 'no cubierto por la muestra'."""
```

**Mitigaciones cuando no hay consenso PIT** (que es el caso de partida de este
proyecto), en orden de preferencia:

1. **Empezar el registro propio hoy.** El `snapshot()` diario es barato y en dos años
   da una muestra PIT genuina. Es la única solución de verdad.
2. **Sustituir el consenso por un modelo de series temporales de beneficios.** El SUE
   original de Foster, Olsen y Shevlin (1984) usa un modelo autorregresivo estacional
   sobre EPS reportados, no consenso de analistas. Es **completamente PIT** si los
   EPS llevan `available_at`, y no depende de ninguna base reescribible. Es la
   opción por defecto recomendada para el histórico largo del proyecto.
3. **Usar consenso reconstruido, declarándolo.** Si se usa, el tearsheet debe llevar
   una advertencia explícita y el resultado debe reportarse también con la variante
   (2) para ver cuánto del alfa depende del input contaminado.
4. **Reportar la cobertura.** Todo backtest que use consenso debe publicar el
   porcentaje de nombre-día del universo PIT con consenso disponible. Si es del 70%,
   el resultado se refiere a ese 70%, no al S&P 500.

---

## 12. Sesgos de segundo orden que también matan

Ninguno de estos tiene el nombre propio de los anteriores, pero todos he visto que
invaliden trabajos.

| # | Sesgo | Manifestación | Corrección |
|---|---|---|---|
| 12.1 | **Ajuste por splits con factor actual** | Se descarga `adj_close` hoy y se usa como si fuera el precio observable en el pasado. El nivel de precio (relevante para el tick size, la elegibilidad de opciones y los filtros de precio mínimo) queda falseado. | Guardar `close` **sin ajustar** y los factores con su fecha ex. Los retornos se calculan con `adj_close`; los **filtros de precio**, con `close` de época. |
| 12.2 | **Short interest fechado por settlement** | FINRA publica el short interest de una fecha de liquidación **~8 días hábiles después** [lit]. Usarlo desde la fecha de liquidación es look-ahead de 8 sesiones. | `available_at` = fecha de **diseminación** FINRA, no la de liquidación. Afecta a `PreEventFeatures.short_interest_delta`. |
| 12.3 | **13F fechado por cierre de trimestre** | Las posiciones institucionales se conocen hasta **45 días** después del cierre [lit]. | `available_at` = fecha de presentación del 13F. |
| 12.4 | **Universo de opciones reconstruido** | Las cadenas de opciones que existen hoy no son las que existían: strikes y vencimientos se añaden. Un `iv_skew_25delta` calculado interpolando sobre strikes que no cotizaban es ficción. | Guardar la cadena tal como se observó, con su timestamp, e interpolar solo entre strikes con volumen o interés abierto **positivo ese día**. |
| 12.5 | **Cambios de índice fechados por fecha efectiva** | S&P anuncia los cambios con antelación; la media entre anuncio y efectividad es de **4.8 días para altas y 5.8 para bajas** [lit]. El mercado reacciona al **anuncio**. | El universo cambia en la **fecha efectiva** (es cuando el índice cambia), pero cualquier señal sobre el evento de inclusión usa la **fecha de anuncio**. Son dos fechas distintas y hacen falta las dos. |
| 12.6 | **Filtros de liquidez con datos futuros** | Filtrar por "ADV medio de toda la muestra > X" usa volumen futuro. Sesga hacia empresas que crecieron. | ADV **rodante hacia atrás** (60 sesiones), recalculado en cada fecha. |
| 12.7 | **Winsorizado y z-score con momentos globales** | `df.clip(df.quantile(0.01), df.quantile(0.99))` sobre todo el panel usa el futuro. Es el error que detecta el test de §14.1. | Winsorizar y estandarizar **por fecha** (cross-section), como manda §3.6 del contrato. Si hace falta una escala temporal, ventana rodante hacia atrás. |
| 12.8 | **Imputación de faltantes con la media de la serie** | `fillna(mean())` inyecta el futuro en cada hueco. | `ffill` limitado con tope de obsolescencia, o NaN explícito y exclusión de la cross-section ese día. |
| 12.9 | **Sectores GICS actuales aplicados hacia atrás** | La clasificación sectorial cambia (creación del sector Servicios de Comunicación en 2018, reclasificaciones de Inmobiliario en 2016). Neutralizar por el sector de hoy en 2005 mete look-ahead en la neutralización. | Guardar la clasificación con `valid_from`/`valid_to`. Con solo la actual disponible, declararlo como limitación conocida. |
| 12.10 | **Festivos y medias sesiones del calendario** | Un calendario de días hábiles genérico incluye el 3 de enero de 2007 (funeral de Ford, mercado cerrado), el 29-30 de octubre de 2012 (huracán Sandy) y omite las medias sesiones. | Calendario de sesiones **real**, con cierres extraordinarios y horas de cierre por sesión. Sin él, §9.4 no se puede implementar. |
| 12.11 | **Selección del periodo de muestra** | Elegir el inicio del backtest donde la señal funciona. | Muestra fijada **antes** de mirar resultados; reportar el resultado en submuestras que no se eligieron. |
| 12.12 | **Sobreajuste por multiplicidad** | Se prueban 200 variantes y se publica la mejor. No es look-ahead, pero infla igual. | §15, punto 9: Sharpe deflactado, corrección por multiplicidad y registro del número de pruebas. |

---

## 13. Resumen: cuánto infla cada sesgo

Tabla de referencia rápida. Las magnitudes tienen su etiqueta de procedencia; las
columnas «dirección» indican si el error hace que el backtest parezca mejor (↑) o
peor (↓) de lo real.

| Sesgo | Dir. | Magnitud típica | Procedencia |
|---|---|---|---|
| B1 `period_end` en vez de `announced_at` | ↑↑↑ | Sharpe de **0.76 con solo ρ=0.05** de fuga; techo de **17.6%/año por nombre** con signo perfecto del salto | [simulado] |
| B2 Datos reexpresados en vez de originales | ↑ | ~**100 pb/año** en factores de calidad | [verificar] |
| B3 Supervivencia (lista actual) | ↑↑ | **38.1% de nombre-día falsos**; **0.9-1.8 pp/año** de sobrestimación; hasta **5.6 pp** en índices concentrados | [medido] / [verificar] |
| B3-residual Cobertura incompleta del fichero PIT | ↑ | **6-12% de miembros ausentes** en 1996-2015, concentrados en el ciclo 2008-09 | [medido] |
| B4 Delisting sin retorno final | ↑ | Imputación estándar **−30%** (NYSE/AMEX) / **−55%** (Nasdaq); eliminó por completo el efecto tamaño en Nasdaq | [lit] |
| B5 Identificadores retroactivos | ↑ (indirecto) | Reintroduce B3 al fallar el join de precios; **≥6 nombres relevantes** con ticker post-quiebra aplicado a toda su historia | [medido] |
| B6 Calendario estimado vs confirmado | ↑ y sesgo de composición | Solo **62%** de fechas confirmadas en un momento dado | [lit] |
| B7 Confundir AMC con BMO | ↑↑↑ | **2.6-2.8×** el Sharpe con 90% del movimiento en el gap; **4-10×** el retorno por evento | [simulado] |
| B8 Anuncio tardío / viernes | ↑ y sesgo de composición | Viernes: respuesta inmediata **−15/−20%**, diferida **+60/+70%**; diferido = **60%** del total | [lit] |
| B9 Reescritura del consenso | ↑ | **1.6%-21.7%** de registros alterados entre descargas; cambios no aleatorios | [lit] |
| 12.2 Short interest por settlement | ↑↑ | **8 sesiones** de look-ahead | [lit] |
| 12.7 Normalización con momentos globales | ↑↑ | Detectable: `max abs(Δ) = 1.0` en el test de truncamiento | [medido] |

---

## 14. Auditoría en código: el harness anti-look-ahead

Los tests que siguen no comprueban que la señal sea buena. Comprueban que sea
**legal**. Deben correr sin red y formar parte de la suite obligatoria.

### 14.1 La prueba de la línea temporal cortada (*truncation test*)

Es el test más potente y el más barato: si una señal es PIT, calcularla sobre la
muestra completa y sobre la muestra truncada en `T` debe dar **valores idénticos bit
a bit** hasta `T`. Cualquier diferencia es información del futuro filtrándose hacia
atrás.

```python
# tests/test_pit_guard.py
import numpy as np, pandas as pd
from earnings_alpha.errors import LookAheadError

def truncation_test(fn, data: pd.DataFrame, cut: int, tol: float = 1e-12) -> None:
    """Verifica que `fn` no usa información posterior a `cut`.

    `fn` debe ser la pipeline COMPLETA (carga, limpieza, normalización, combinación),
    no solo el cálculo del factor: la mayoría de las fugas viven en los pasos de
    preproceso, no en la fórmula.

    ATENCIÓN al patrón de NaN: comparar solo `max|Δ|` produce un FALSO NEGATIVO en
    la fuga más clásica de todas, `shift(-1)`, porque la única celda que difiere
    pasa de NaN a valor y `max()` ignora los NaN. La comparación de máscaras de
    nulidad no es un extra defensivo: es lo que hace que el test funcione.
    """
    a = fn(data).iloc[:cut]
    b = fn(data.iloc[:cut]).iloc[:cut]

    if (a.isna() != b.isna()).to_numpy().any():
        n = int((a.isna() != b.isna()).to_numpy().sum())
        raise LookAheadError(
            f"el patrón de valores ausentes cambia en el pasado al añadir datos "
            f"futuros ({n} celdas): fuga de horizonte en la frontera de la muestra"
        )

    delta = (a - b).abs().max().max()
    if not (np.isnan(delta) or delta <= tol):
        raise LookAheadError(
            f"la señal cambia en el pasado al añadir datos futuros: max|Δ| = {delta:.3e}"
        )
```

Validación **[medido]** sobre cuatro señales, una limpia y tres con fugas de
naturaleza distinta:

```python
def signal_ok(px):        # solo pasado
    return px.pct_change(20)

def signal_bad_mean(px):  # .mean() sobre TODA la muestra: usa el futuro
    return px.pct_change(20) / px.pct_change().rolling(60).std().mean()

def signal_bad_shift(px): # la fuga clásica de un día
    return px.pct_change(20).shift(-1)

def signal_bad_max(px):   # normalización por el máximo global
    return px.pct_change(20) / px.pct_change(20).abs().max()
```

| Señal | Truncamiento **sin** control de NaN | Truncamiento **con** control de NaN | Futuro aleatorizado (§14.2) |
|---|---|---|---|
| `signal_ok` | pasa | pasa | pasa |
| `signal_bad_mean` | **falla** (`max abs(Δ)`=1.001e+00) | **falla** | **falla** |
| `signal_bad_shift` | **pasa — FALSO NEGATIVO** | **falla** (patrón de NaN) | **falla** (Δ=5.2e−02) |
| `signal_bad_max` | **falla** (`max abs(Δ)`=4.98e−01) | **falla** | **falla** (Δ=4.98e−01) |

Dos lecturas que conviene no perder:

- `signal_bad_mean` es el error que sobrevive a una revisión de código: la ventana
  rodante de 60 días parece cuidadosa y la fuga está en el `.mean()` final, que
  agrega sobre toda la historia.
- `signal_bad_shift` es la razón de que este apartado exista. La versión ingenua del
  test —la que escribe casi todo el mundo— **no detecta un `shift(-1)`**, que es
  literalmente el look-ahead de un día del §9. Cualquier implementación del test que
  no compare las máscaras de nulidad da una falsa sensación de seguridad.

**Cuándo aplicarlo:** en cada factor, en cada feature pre-evento, y en la combinación
final. Con al menos tres puntos de corte distintos, incluido uno dentro de una
temporada de resultados.

### 14.2 La prueba del futuro aleatorizado (*future-shuffle test*)

Complementa a la anterior: en vez de truncar, se **destruye** el futuro y se
comprueba que la señal pasada no cambia.

```python
def future_shuffle_test(fn, data: pd.DataFrame, cut: int, seed: int = 0, tol: float = 1e-12) -> None:
    """Permuta aleatoriamente todo lo posterior a `cut` y verifica que la señal
    anterior a `cut` no se mueve.

    Es complementario del truncamiento, no redundante: mantiene el TAMAÑO de la
    muestra y cambia solo su CONTENIDO futuro. Detecta las fugas de horizonte
    (`shift(-n)`) por diferencia numérica directa, sin depender de que la
    implementación compare máscaras de nulos, que es justo donde falla la versión
    ingenua del truncamiento. Se ejecutan los dos, siempre.
    """
    rng = np.random.default_rng(seed)
    shuffled = data.copy()
    idx = rng.permutation(len(data) - cut) + cut
    shuffled.iloc[cut:] = data.iloc[idx].to_numpy()
    a, b = fn(data).iloc[:cut], fn(shuffled).iloc[:cut]
    if (a.isna() != b.isna()).to_numpy().any():
        raise LookAheadError("el patrón de nulos depende del contenido del futuro")
    delta = (a - b).abs().max().max()
    if not (np.isnan(delta) or delta <= tol):
        raise LookAheadError(f"la señal depende de la forma del futuro: max|Δ| = {delta:.3e}")
```

**Complementariedad verificada [medido].** En la batería de cuatro señales de §14.1,
el barajado detecta las tres fugas, incluida `signal_bad_shift`, que la versión
ingenua del truncamiento deja pasar. La razón por la que se ejecutan **los dos** y no
solo el barajado es que los dos atacan ejes distintos —tamaño de muestra frente a
contenido— y una fuga estadística cuyo efecto sobre el estadístico agregado sea
invariante ante permutación de filas (un `sum()` o un `count()` sobre el eje temporal)
puede sobrevivir al barajado y no al truncamiento. El coste de correr los dos es
despreciable; el coste de elegir mal, no.

### 14.3 El guardián de `available_at`

Instrumentación, no test: se propaga el `available_at` máximo de los inputs a través
del cálculo y se comprueba el invariante de §1.4 en cada frontera de módulo.

```python
# earnings_alpha/pit/guard.py
from contextlib import contextmanager

class PITGuard:
    """Registra el available_at máximo consumido y verifica el invariante PIT.

    Se instala alrededor del cálculo de cada factor. Coste despreciable y detecta
    la clase de errores que ningún test estadístico ve.
    """
    def __init__(self, decision_ts: pd.Timestamp) -> None:
        self.decision_ts = decision_ts
        self.max_seen: pd.Timestamp | None = None
        self.offenders: list[tuple[str, pd.Timestamp]] = []

    def observe(self, source: str, available_at: pd.Timestamp) -> None:
        if pd.isna(available_at):
            raise LookAheadError(f"{source}: available_at nulo; no se puede verificar PIT")
        if available_at > self.decision_ts:
            self.offenders.append((source, available_at))
        if self.max_seen is None or available_at > self.max_seen:
            self.max_seen = available_at

    def assert_clean(self) -> None:
        if self.offenders:
            worst = max(self.offenders, key=lambda x: x[1])
            raise LookAheadError(
                f"{len(self.offenders)} inputs posteriores a la fecha de decisión "
                f"{self.decision_ts}; el peor es {worst[0]} con available_at={worst[1]}"
            )

@contextmanager
def pit_guard(decision_ts):
    g = PITGuard(pd.Timestamp(decision_ts))
    try:
        yield g
    finally:
        g.assert_clean()
```

### 14.4 Reproducibilidad de las mediciones de este documento

Las cifras **[medido]** y **[simulado]** de este documento proceden de cuatro scripts
ejecutados fuera del árbol del repositorio (este documento es el único fichero que
tengo asignado). Lo que hace cada uno:

| Script | Qué calcula | Secciones |
|---|---|---|
| `seed_stats.py` | Tickers distintos, supervivencia 1996→2025, rotación anual, nombre-día falsos, convenciones de ticker | §5.2, §7.2 |
| `seed_stats3.py` | Tamaño del índice por año, solapes `BRK-B`/`BRK.B`, huecos entre snapshots, tickers retroactivos | §5.4, §7.2 |
| `val_pit.py` | `tradable_date` (8 casos incl. la trampa UTC), `asof_join` con 3 vintages, `members_on` as-of | §2.3, §9.3 |
| `val_pit2.py` / `val_pit4.py` | Monte Carlo de fuga de información, calibración del sesgo de supervivencia, inflación BMO/AMC | §2.2, §5.2, §9.2 |
| `val_doc_snippets.py` | `disseminated_at`, `delisting_return`, `upsert_vintage`, `PITGuard`, `future_shuffle_test` | §3.3, §4.3, §6.3, §14.2, §14.3 |
| `val_complement.py` | Falso negativo del truncamiento ingenuo ante `shift(-1)`; complementariedad de los dos detectores | §14.1, §14.2 |

Semillas fijas (`20260803`, `7`, `11`, `1000+i`). **Todos los fragmentos de código de
este documento se han ejecutado y sus aserciones pasan**; los que producen cifras las
tienen transcritas en las tablas correspondientes.

Cuando el módulo `pit` esté implementado, estos scripts deben migrarse a `tests/`
como tests de regresión: las cifras de las tablas de §5.2 y §5.4 son **aserciones
sobre los datos semilla** y detectarán cualquier corrupción o sustitución silenciosa
del fichero.

---

## 15. Checklist de validación pre-producción

Ninguna estrategia pasa a producción sin que las 30 casillas estén marcadas y
firmadas. La checklist está ordenada por coste de descubrir el fallo tarde.

### A. Fechado de los datos

- [ ] **1.** Todo hecho fundamental tiene `available_at` no nulo, y ese
  `available_at` es la fecha de **presentación/difusión**, nunca `period_end`.
- [ ] **2.** Los conceptos que solo aparecen en el 10-Q (CFO, capex, balance) llevan
  el `available_at` del **10-Q**, no el del 8-K. Verificado por muestreo sobre al
  menos 20 empresa-trimestre.
- [ ] **3.** La regla EDGAR de las 17:30 ET está implementada, con la excepción de
  los formularios 3/4/5/144 hasta las 22:00 ET.
- [ ] **4.** Ningún join usa `direction="forward"`, `bfill`, `interpolate` ni
  `reindex` sin `ffill` acotado.
- [ ] **5.** Todos los `ffill` tienen tope de obsolescencia y el tope está
  justificado por la frecuencia real de la fuente.
- [ ] **6.** El calendario de estimaciones y el de resultados se almacenan con
  **vintages** (`as_of`), y `entry_offset < 0` los exige.

### B. Universo e identidad

- [ ] **7.** La pertenencia al índice se resuelve con as-of **hacia atrás** sobre el
  fichero histórico; nunca con la lista actual.
- [ ] **8.** El informe de cobertura por año está adjunto al tearsheet y su mediana
  supera el umbral acordado. Los años por debajo se declaran.
- [ ] **9.** La obsolescencia máxima del snapshot de membresía usada en el backtest
  está reportada (en el fichero semilla llega a **88 días**).
- [ ] **10.** Todos los tickers pasan por `normalize_ticker` **en la carga**.
- [ ] **11.** No hay ningún ticker que resuelva a dos identidades distintas dentro de
  la muestra sin que el mapa de identidad lo separe (caso `GM` / `MTLQQ`).
- [ ] **12.** Los nombres cuya serie de precios termina antes de su salida del índice
  están enumerados y **cada uno** tiene una razón de exclusión y un retorno de
  delisting, o se documenta por qué no.
- [ ] **13.** Las fusiones en acciones continúan la posición en el adquirente; las
  fusiones en efectivo cierran al precio de la oferta.
- [ ] **14.** Los spin-offs no generan retornos negativos ficticios el día ex, y las
  escindidas no heredan historia fundamental *pro forma*.

### C. Eventos y ejecución

- [ ] **15.** `tradable_date` está cubierta por tests para: BMO, AMC, DMH, UNKNOWN,
  fin de semana, festivo, media sesión, cambio de horario de verano y la trampa
  UTC↔ET.
- [ ] **16.** La clasificación BMO/AMC consulta la **hora de cierre de esa sesión**,
  no un 16:00 ET constante.
- [ ] **17.** `Session.UNKNOWN` se trata como AMC, y hay un test que lo fija.
- [ ] **18.** El retorno del evento se reporta **descompuesto** en gap (close→open) y
  sesión (open→close). El tearsheet muestra qué fracción del alfa vive en cada uno.
- [ ] **19.** Ninguna orden derivada de un anuncio AMC se ejecuta al cierre del día
  del anuncio.
- [ ] **20.** El modelo de costes aplica un spread ampliado en la apertura
  post-evento, justificado y no un fijo arbitrario.

### D. Pruebas de fuga

- [ ] **21.** El **test de truncamiento** (§14.1) pasa para cada factor, cada feature
  pre-evento y la combinación final, con ≥3 puntos de corte — y su implementación
  **compara las máscaras de nulos**, sin lo cual no detecta un `shift(-1)` (§14.1).
- [ ] **22.** El **test del futuro aleatorizado** (§14.2) pasa para la pipeline
  completa.
- [ ] **23.** El `PITGuard` (§14.3) está instalado y `assert_clean()` no lanza en
  ninguna fecha del backtest.
- [ ] **24.** Toda normalización (z-score, winsorizado, rank) es **por fecha**; no
  hay ningún `quantile()`, `mean()`, `std()`, `min()` ni `max()` calculado sobre el
  eje temporal completo.
- [ ] **25.** **Prueba del reloj adelantado:** se reejecuta el backtest desplazando
  todas las señales una sesión hacia atrás. Si el resultado *mejora* mucho, hay una
  fuga de un día en alguna parte.

### E. Estadística y credibilidad

- [ ] **26.** Toda métrica de rendimiento va con intervalo de confianza (bootstrap
  estacionario). Un Sharpe sin banda de error no se acepta (§3.8 del contrato).
- [ ] **27.** La validación temporal es **CV purgada con embargo**: se purgan las
  observaciones cuyas etiquetas solapan con el test, y se añade un embargo del orden
  del **1%** de las observaciones [lit]. Con eventos de resultados, el solape es de
  semanas, no de días.
- [ ] **28.** El **número de configuraciones probadas** está registrado, y el Sharpe
  se reporta **deflactado** (Bailey y López de Prado, 2014) y con corrección por
  comparaciones múltiples (Benjamini-Hochberg).
- [ ] **29.** El umbral de significancia es el de multiplicidad, no `t > 2`: la
  referencia para un factor nuevo es **`t > 3.0`** (Harvey, Liu y Zhu, 2016) [lit].
- [ ] **30.** El resultado se reporta **estratificado** por: fecha estimada vs
  confirmada, BMO vs AMC, viernes vs resto, y con/sin los años de peor cobertura del
  universo. Si el alfa vive en un solo estrato, se dice.

### Firma

```
Estrategia: ______________________  Rama/commit: ______________________
Universo:   ______________________  Muestra: ____________ a ____________
Cobertura mediana del universo: ______%   Nº de configuraciones probadas: ______
Sharpe bruto: ______  Sharpe deflactado: ______  IC 95%: [______, ______]
Casillas 1-30 verificadas por: ______________________  Fecha: ____________
```

---

## 16. Preguntas abiertas (para `docs/OPEN_QUESTIONS.md`)

Estas quedan pendientes de verificación en un entorno con red, y las anoto aquí para
que el propietario de ese fichero las traslade:

1. **Distribución empírica del retardo `period_end → announced_at` en el S&P 500.**
   Las cifras de §3.2 están marcadas [prior]. Medirla sobre EDGAR (8-K item 2.02) es
   directo y convertiría §3.2 en [medido]. Debe reportarse por sector y por tamaño,
   y separando trimestres de 10-Q de los de 10-K.
2. **Retardo `announced_at → 10-Q disseminated_at`.** Determina cuándo son negociables
   accruals, F-Score y FCF yield. Es el número que más cambia el diseño del ángulo A.
3. **Fracción de anuncios AMC / BMO / DMH en el S&P 500 actual**, y **fracción del
   movimiento de resultados que ocurre en el gap**. Los dos alimentan directamente la
   tabla de §9.2, que hoy es [simulado] con `gap_frac` como parámetro libre.
4. **Verificación del −30% de Shumway (1997)** contra el texto original: las fuentes
   secundarias consultadas discrepan entre −30% y −55% para NYSE/AMEX.
5. **Completar el fichero histórico de constituyentes** para 1996-2018, donde faltan
   entre el 6% y el 12% de los miembros [medido]. Sin esto, ningún backtest anterior
   a 2019 puede declararse libre de sesgo de supervivencia residual.
6. **Mapa PIT de tickers** que traduzca los identificadores retroactivos del fichero
   semilla (`ENRNQ`, `MTLQQ`, `LEHMQ`, `EKDKQ`, `AAMRQ`, `WAMUQ`, ...) al símbolo
   vigente en cada fecha. Sin él, esos nombres no se pueden unir con precios y el
   sesgo de supervivencia vuelve por la puerta de atrás.
7. **Historia de clasificación GICS con `valid_from`/`valid_to`** (§12.9). Con solo la
   clasificación actual, la neutralización sectorial anterior a 2018 tiene
   look-ahead.

---

## 17. Referencias

**Look-ahead, reexpresiones y datos point-in-time**

- Ljungqvist, A., Malloy, C. J., y Marston, F. C. (2009). *Rewriting History*. **The
  Journal of Finance**, 64(4), 1935-1960.
  https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.2009.01484.x
- Ideagen Audit Analytics (2024). *Financial Restatements: A Twenty-Year Review*
  (datos 2004-2023). Resumen en
  https://www.thecorporatecounsel.net/blog/2025/06/restatements-non-accelerated-filers-lead-the-pack.html
  y https://www.thecaq.org/audit-in-action-what-ten-years-of-restatement-trends-tell-us-about-the-state-of-financial-reporting
- *Lost in Standardization: Revisiting Accounting-Based Return Anomalies*. Columbia
  Business School / CEASA.
  https://business.columbia.edu/sites/default/files-efs/imce-uploads/CEASA/Events%20Page/revisiting_accounting-based_return_anomalies.pdf
  (diferencias entre datos as-filed y datos estandarizados de proveedor)
- Asness, C. S., Frazzini, A., y Pedersen, L. H. (2019). *Quality Minus Junk*.
  **Review of Accounting Studies**, 24(1), 34-112.
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2312432

**Supervivencia y delisting**

- Shumway, T. (1997). *The Delisting Bias in CRSP Data*. **The Journal of Finance**,
  52(1), 327-340.
  https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.1997.tb03818.x
- Shumway, T., y Warther, V. A. (1999). *The Delisting Bias in CRSP's Nasdaq Data and
  Its Implications for the Size Effect*. **The Journal of Finance**, 54(6),
  2361-2379. https://onlinelibrary.wiley.com/doi/abs/10.1111/0022-1082.00192
- Beaver, W., McNichols, M., y Price, R. (2007). *Delisting Returns and Their Effect
  on Accounting-Based Market Anomalies*. **Journal of Accounting and Economics**.
  https://www.researchgate.net/publication/222392568_Delisting_Returns_and_Their_Effect_on_Accounting-Based_Market_Anomalies
- CRSP. *CRSP Calculations* (tratamiento de `DLRET`, `CFACPR`, spin-offs y fusiones).
  http://www.crsp.com/products/documentation/crsp-calculations
- Dimensional Fund Advisors. *Why Worry About Survivorship Bias?*
  https://www.dimensional.com/us-en/insights/why-worry-about-survivorship-bias

**Identificadores y enlace de bases**

- WRDS / NYU Libraries. *Linking Queries* (PERMNO, PERMCO, GVKEY, reutilización de
  tickers y CUSIP). https://guides.nyu.edu/wrds/linking-suite
- CRSP. *US Stock Database Knowledge Base*.
  https://www.crsp.org/wp-content/uploads/CRSP_US_Stock_Database_Knowledge_Base.pdf

**Fechas de anuncio, timing estratégico e inatención**

- DellaVigna, S., y Pollet, J. M. (2009). *Investor Inattention and Friday Earnings
  Announcements*. **The Journal of Finance**, 64(2), 709-749.
  https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.2009.01447.x — versión de
  trabajo: https://eml.berkeley.edu/~sdellavi/wp/earnfr06-12-11NewTitle.pdf
- Begley, J., y Fischer, P. E. (1998). *Is There Information in an Earnings
  Announcement Delay?* **Review of Accounting Studies**, 3, 347-363.
- Bagnoli, M., Kross, W., y Watts, S. G. (2002). *The Information in Management's
  Expected Earnings Report Date: A Day Late, a Penny Short*. **Journal of Accounting
  Research**, 40(5), 1275-1296.
  https://www.researchgate.net/publication/230557124_The_Information_in_Management's_Expected_Report_Date_A_Day_Late_A_Penny_Short
- Boulland, R., y Dessaint, O. (2017). *Announcing the Announcement*. **Journal of
  Banking & Finance**.
  https://www.sciencedirect.com/science/article/abs/pii/S0378426617301097
- Wall Street Horizon. *Earnings Calendar* y *5 Research Findings on Earnings Date
  Timing That Affect Trading* (fechas confirmadas vs. estimadas).
  https://www.wallstreethorizon.com/blog/5-Research-Findings-on-Earnings-Date-Timing-That-Affect-Trading
- Gow, I. *Empirical Research in Accounting: Tools and Methods*, cap. 14
  (Post-earnings announcement drift; tratamiento de `RDQ` e `ANNDATS`).
  https://iangow.github.io/far_book/pead.html

**PEAD, costes y capacidad**

- Chordia, T., Goyal, A., Sadka, G., Sadka, R., y Shivakumar, L. (2009). *Liquidity
  and the Post-Earnings-Announcement Drift*. **Financial Analysts Journal**.
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=972758
- Ng, J., Rusticus, T. O., y Verdi, R. S. (2008). *Implications of Transaction Costs
  for the Post-Earnings Announcement Drift*. **Journal of Accounting Research**,
  46(3), 661-696.
  https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1475-679X.2008.00290.x
- Foster, G., Olsen, C., y Shevlin, T. (1984). *Earnings Releases, Anomalies, and the
  Behavior of Security Returns*. **The Accounting Review**, 59(4), 574-603. (Modelo
  de series temporales de beneficios para el SUE sin consenso de analistas.)

**Estadística de backtests**

- Harvey, C. R., Liu, Y., y Zhu, H. (2016). *…and the Cross-Section of Expected
  Returns*. **The Review of Financial Studies**, 29(1), 5-68.
  https://academic.oup.com/rfs/article/29/1/5/1843824
- Bailey, D. H., y López de Prado, M. (2014). *The Deflated Sharpe Ratio: Correcting
  for Selection Bias, Backtest Overfitting and Non-Normality*. **Journal of Portfolio
  Management**, 40(5). https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
  (Purging, embargo, CPCV.) Resumen:
  https://en.wikipedia.org/wiki/Purged_cross-validation

**Índices y microestructura**

- Greenwood, R., y Sammon, M. (2022). *The Disappearing Index Effect*. NBER Working
  Paper 30748. https://www.nber.org/system/files/working_papers/w30748/w30748.pdf
  (retardo medio anuncio→efectividad: 4.8 días en altas, 5.8 en bajas)
- S&P Dow Jones Indices. *What Happened to the Index Effect?*
  https://www.spglobal.com/spdji/en/documents/research/research-what-happened-to-the-index-effect.pdf

**Normativa y plazos**

- SEC. *Acceleration of Periodic Report Filing Dates* (Release 33-8128) y *Revisions
  to Accelerated Filer Definition* (Release 33-8644).
  https://www.sec.gov/files/rules/final/33-8128.htm ·
  https://www.sec.gov/files/rules/final/33-8644.pdf
- SEC. *EDGAR Filer Manual (Volume II)*, cap. 10 — horarios de aceptación y regla de
  las 17:30 ET. https://www.sec.gov/files/edgar/filermanual/efmvol2-c10.pdf
- FINRA. *Short Interest Reporting* — calendario de fechas de liquidación,
  presentación y diseminación.
  https://www.finra.org/filing-reporting/regulatory-filing-systems/short-interest
- Gibson Dunn. *2026 SEC Filing Deadlines* (Form 4: 2 días hábiles; 13F: 45 días).
  https://www.gibsondunn.com/wp-content/uploads/2025/08/SEC-Filing-Deadline-Calendar-2026.pdf

**Datos del repositorio usados en las mediciones**

- `data/seed/sp500_historical_components.csv` — 3.482 snapshots, 1996-01-02 a
  2025-08-23.
- `data/seed/sp500_constituents.csv` — 503 constituyentes actuales con sector GICS y
  CIK.
