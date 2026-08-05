# RECOLECTOR — captura diaria point-in-time en la máquina del usuario

**Módulo:** `earnings_alpha/collector/` · **Versión:** 1.0.0 (`COLLECTOR_VERSION`)
**Comando:** `python3 -m earnings_alpha.collector`

---

## 1. Qué es y por qué existe

El recolector acumula, desde el día en que se enchufa, los datos que **no se
pueden comprar hacia atrás** con presupuesto cero:

| Dataset | Qué captura | Por qué no se puede backfillear |
|---|---|---|
| `option_chain` | cadena completa por ticker: strikes, bid/ask, último, volumen, OI, IV | no existe histórico gratuito con licencia limpia (`docs/research/datasets_opciones.md` §1) |
| `open_interest` | OI actualizado de la sesión anterior, pasada matinal | el OI diario histórico es producto comercial; y el de ayer solo existe hoy |
| `consensus` | vintage diario `(ticker, period_end, as_of)` del consenso EPS/ingresos | los vintages históricos solo existen en IBES/WRDS o Zacks (`data_sources.md` §6) |
| `earnings_calendar` | fechas previstas de próximos anuncios, con sesión BMO/AMC | la *historia* de fechas previstas (movimientos, confirmaciones) no la publica nadie |
| `short_interest` | publicación quincenal consolidada de FINRA | FINRA solo mantiene **un año rodante** en línea (`data_sources.md` §8.2) |
| `off_exchange` | fichero diario Reg SHO (short volume + volumen off-exchange) | recuperable de archivos, pero archivarlo a diario evita depender de ellos |

**Cada día sin recolectar es un día de panel perdido para siempre.** Las
features de opciones del ángulo B (`vol_spread` de Cremers-Weinbaum,
`iv_skew_25delta`, `oi_buildup`, `put_call_volume_ratio`) son las de mayor
respaldo académico para detectar negociación informada y las únicas que hoy no
se pueden backtestear con presupuesto cero. Este módulo es la salida (c) del
veredicto de `datasets_opciones.md` §1: *validarlas out-of-sample hacia
adelante, empezando a recolectar hoy*.

Toda fila lleva `source`, `captured_at` (UTC exacto), `available_at` y
`collector_version`. Ese sello es lo que permite usar el panel con
`pit.asof_join` sin riesgo de look-ahead.

---

## 2. Las dos pasadas y la hora de captura

### 2.1 Pasada `close` — 16:15–17:00 ET, y siempre a la misma hora

Captura: calendario de próximos resultados, cadenas de opciones (cola de
prioridad, §4) y consenso de esos tickers.

Por qué **después de las 16:15 ET**:

- la subasta de cierre ha terminado y el volumen del día es final;
- las opciones dejan de cotizar a las 16:00 ET (16:15 para algunos productos):
  el bid/ask capturado es el asentado del cierre, no un punto intradía;
- la IV capturada corresponde al nivel de cierre del subyacente, que es el que
  usan todos los modelos aguas abajo.

Por qué **antes de las ~17:00 ET**: pasada esa hora algunos proveedores
recalculan/consolidan y la foto deja de ser "el cierre de hoy". Y por qué **a
hora fija**: dos fotos a horas distintas de días distintos no son comparables
(volumen acumulado, régimen de liquidez); la consistencia horaria es una
propiedad point-in-time del panel, no una manía. El recolector avisa en el log
(`capture_window_warning`) si corre fuera de ventana, pero captura igualmente:
un dato tarde vale más que un hueco.

### 2.2 Pasada `morning` — 08:30–09:15 ET: la trampa del open interest

El OI del cierre de la sesión `T` lo calcula la OCC en su ciclo nocturno y se
disemina **la mañana de `T+1`** (`data_sources.md` §7.3;
`informed_trading.md`, apartado de `oi_buildup`). Consecuencias:

- lo que la pasada `close` de `T` guarda en la columna `open_interest` es el OI
  de `T-1` — correcto como "lo conocible en ese instante", pero no es el OI de `T`;
- la pasada `morning` de `T+1` vuelve a pedir la cadena y guarda **solo** el OI
  en el dataset `open_interest`, con `oi_date = T` y `available_at` = instante
  real de la captura matinal.

Un `oi_buildup` calculado con el OI de `T` como conocido en `T` mete un día
entero de look-ahead en ventanas pre-evento de 5–20 sesiones: es exactamente el
error que esta separación en dos pasadas hace imposible.

La pasada matinal captura además el fichero Reg SHO de la sesión anterior y, si
hay publicación nueva, el short interest quincenal (los días sin publicación la
deduplicación lo convierte en no-op, de modo que se puede intentar a diario).

---

## 3. Instalación en la máquina del usuario

### 3.1 Dependencias y credenciales

```bash
pip install -e ".[providers]"        # añade yfinance al núcleo
# opcional pero recomendado (cadena con IV de ORATS, sandbox gratuito):
export TRADIER_ACCESS_TOKEN="..."    # cuenta gratuita en developer.tradier.com
# opcional: URL de producción en vez del sandbox (por defecto sandbox):
# export TRADIER_API_BASE="https://api.tradier.com/v1"
# opcional, para calendario/consenso con etiqueta BMO/AMC:
export FINNHUB_API_KEY="..."         # plan gratuito: 60 req/min
# alternativa: export FMP_API_KEY="..."
```

Selección automática de fuentes (`--options-source auto`,
`--calendar-provider auto`): Tradier si hay token → yfinance si está instalado
→ sintético (con aviso; solo útil para probar). Calendario: Finnhub → FMP →
sintético.

**Nota de licencias.** yfinance raspa endpoints no oficiales de Yahoo: uso
personal/investigación, no redistribuir los datos crudos. El endpoint público de
Cboe exige autorización previa por escrito y **no** se usa aquí
(`datasets_opciones.md` §4). FINRA y el sandbox de Tradier son de acceso
gratuito con sus términos estándar.

### 3.2 Crontab exacto

El recolector ya sabe saltarse los días sin sesión (festivos NYSE incluidos, vía
`pit.TradingCalendar`), así que el cron puede ser un simple lunes-viernes.

Con **cronie** (Fedora/RHEL/Arch) o cualquier cron que soporte `CRON_TZ`:

```cron
CRON_TZ=America/New_York
# pasada de cierre: 16:30 ET, lunes a viernes
30 16 * * 1-5  cd /ruta/a/Testing && /usr/bin/python3 -m earnings_alpha.collector run --pass close   >> "$HOME/earnings_collector.log" 2>&1
# pasada matinal (open interest + Reg SHO + short interest): 08:45 ET
45  8 * * 1-5  cd /ruta/a/Testing && /usr/bin/python3 -m earnings_alpha.collector run --pass morning >> "$HOME/earnings_collector.log" 2>&1
```

Con el cron de Debian/Ubuntu (que **no** soporta `CRON_TZ`), usar systemd, que
maneja la zona horaria y los cambios de hora de EE. UU. correctamente:

```ini
# ~/.config/systemd/user/earnings-collector-close.service
[Unit]
Description=Recolector earnings-alpha: pasada de cierre
[Service]
Type=oneshot
WorkingDirectory=/ruta/a/Testing
ExecStart=/usr/bin/python3 -m earnings_alpha.collector run --pass close

# ~/.config/systemd/user/earnings-collector-close.timer
[Unit]
Description=16:30 ET, lunes a viernes
[Timer]
OnCalendar=Mon..Fri 16:30 America/New_York
Persistent=true
[Install]
WantedBy=timers.target
```

(duplicar para `morning` con `OnCalendar=Mon..Fri 08:45 America/New_York`), y:

```bash
systemctl --user daemon-reload
systemctl --user enable --now earnings-collector-close.timer earnings-collector-morning.timer
```

`Persistent=true` hace que un portátil apagado a las 16:30 ejecute la pasada al
encenderse: tarde y con aviso de ventana, pero sin hueco. **No** usar horas UTC
fijas en cron: el cambio de hora de EE. UU. las desplazaría una hora dos veces
al año, justo el tipo de inconsistencia que la §2.1 prohíbe.

### 3.3 Verificación de la instalación

```bash
python3 -m earnings_alpha.collector run --dry-run     # sin red: muestra el plan de hoy
python3 -m earnings_alpha.collector status            # particiones, huecos, fallos
python3 -m earnings_alpha.collector verify            # integridad SHA-256 del almacén
```

`--dry-run` no abre la red ni escribe nada; imprime la cola de prioridad, el
coste estimado en peticiones y qué datasets se alimentarían. `--date` sirve para
inspeccionar el plan de otra fecha; los sellos `captured_at` usan **siempre** el
reloj real (un recolector que miente sobre cuándo capturó no es point-in-time),
así que `--date` es para planificar y probar, no para "rellenar" días pasados —
eso es imposible por construcción.

---

## 4. Cola de prioridad y coste en peticiones

### 4.1 La cola

1. **Tickers del S&P 500 con resultados en las próximas 4 semanas**
   (`horizon_days=28`), ordenados por proximidad del anuncio. La ventana cubre
   la ventana pre-evento típica [T-20 sesiones, T-1] de `PreEventFeatures` con
   margen para fechas estimadas que se adelantan.
2. **Muestra rotatoria de línea base** (`baseline=25` por defecto) del resto del
   universo: grupo de control fuera de ventana (un detector que también
   "detecta" fuera de ventana está midiendo ruido) y garantía de que ningún
   ticker pasa más de ~3-4 semanas sin foto (⌈478/25⌉ ≈ 20 días hábiles de ciclo).

La rotación es determinista (función de la fecha), de modo que re-ejecutar la
pasada el mismo día reproduce el mismo plan, y la pasada matinal puede leer del
almacén exactamente qué se fotografió ayer.

### 4.2 Aritmética de coste [verificado por aritmética]

Coste por ticker de una cadena: `1 + max_expiries` peticiones (lista de
vencimientos + una por vencimiento). Con el valor por defecto `--max-expiries 6`:
**7 peticiones/ticker**.

| Escenario | Tickers en cola | Peticiones cadena | Consenso (Finnhub, 1/ticker) | Total | Reloj a 1 req/s (yfinance) |
|---|---|---|---|---|---|
| Semana tranquila | ~40 evento + 25 base = 65 | 455 | 65 | ~520 | ~9 min |
| Media | 100 | 700 | 100 | ~800 | ~14 min |
| Pico de temporada | 250 | 1 750 | 250 | ~2 000 | ~34 min |

El presupuesto por defecto (`--budget 2000`) cubre el pico; si se recorta, la
cola se trunca **por el final** (línea base primero, eventos lejanos después) y
los diferidos quedan registrados. La pasada matinal cuesta lo mismo que la de
cierre en cadenas (repite los mismos tickers) + 1 petición de Reg SHO + 1 de
short interest.

Con Tradier sandbox el límite práctico es su rate limit (~60-120 req/min según
endpoint [verificar]); el `HttpClient` del repo ya aplica una política
conservadora (2 req/s, ráfaga 4).

### 4.3 Tamaño en disco [prior]

Una cadena de 6 vencimientos ronda 300–900 filas-contrato; en parquet comprimido,
~40–120 KB por ticker y día. A 100 tickers/día ≈ 5–12 MB/día ≈ **1,5–3 GB/año**,
dominado por `option_chain`. El resto de datasets es despreciable (<50 MB/año).

---

## 5. Cuándo el panel acumulado cruza el umbral de utilidad

Base: ~503 tickers × 4 anuncios/año ≈ **2 000 eventos/año**; el recolector
fotografía la ventana [T-28, T] de prácticamente todos (la cola prioriza
exactamente eso).

| Antigüedad | Eventos con ventana pre-evento completa | Qué se puede hacer honestamente |
|---|---|---|
| **3 meses** | ~500 | Calibración del pipeline y sanity checks (¿la IV sube hacia el evento? ¿el `vol_spread` medio es ≈0 fuera de ventana?). Potencia solo para efectos grandes: con 250/250 eventos por grupo, el mínimo efecto detectable a 80 % de potencia es d≈0,25 [verificado por aritmética]. Una temporada entera de resultados: suficiente para decidir si la señal merece seguir. |
| **6 meses** | ~1 000 | Primera validación out-of-sample de features individuales (d≈0,18). Dos temporadas: se puede separar temporada fuerte/débil. Aún insuficiente para el modelo conjunto con 15 features y corrección de Benjamini-Hochberg. |
| **12 meses** | ~2 000 | Cross-section completa: neutralización por sector, CV purgada con embargo (`stats.validation`), BH sobre el set completo de features (d≈0,12). Es el umbral en que el panel propio **sustituye** una compra de datos de opciones de ~1 200 USD/año. Cada año adicional añade ~2 000 eventos y un régimen de volatilidad más. |

Dos advertencias honestas: (a) el panel solo contiene el régimen de mercado que
le tocó vivir — un año sin sobresaltos no valida nada sobre crisis; (b) los
primeros ~28 días son de arranque: los eventos de esas semanas tienen ventana
pre-evento incompleta y deben excluirse de cualquier test.

---

## 6. Almacén: layout, idempotencia e integridad

```
data/collector/
  option_chain/date=2026-08-05/part-0000-a79a3886.parquet
  option_chain/date=2026-08-05/_MANIFEST.json
  open_interest/date=2026-08-05/…
  consensus/date=2026-08-05/…
  earnings_calendar/date=2026-08-05/…
  off_exchange/date=2026-08-05/…
  short_interest/date=2026-07-31/…        # particiona por settlement, no por captura
  _log/collection_log.jsonl               # registro append-only de cada intento
```

- **Append-only:** nunca se reescribe un fichero; añadir datos crea part files
  nuevos. Hive-style, legible con pyarrow/duckdb sin este módulo delante.
- **Idempotencia:** cada dataset tiene su clave natural (p. ej.
  `(chain_date, ticker, expiry, right, strike, source)`); re-ejecutar la pasada
  el mismo día aporta 0 filas. **La primera captura del día gana**: una segunda
  foto más tarde sería otro vintage y no debe suplantar a la primera.
- **Integridad:** `_MANIFEST.json` guarda filas y SHA-256 por fichero;
  `read(validate=True)` (el defecto) verifica checksums, recuentos, ficheros
  ausentes y parquet huérfanos (residuo de un proceso interrumpido), y falla con
  `DataQualityError` en vez de servir datos dudosos. `verify` recorre todo el
  almacén y lista los problemas sin detenerse en el primero.
- Un part file por ticker y pasada: más ficheros pequeños a cambio de que un
  proceso interrumpido pierda como mucho un ticker, no la pasada entera.

---

## 7. Fallos, huecos y qué es recuperable

Todo intento (ok/failed/skipped/deferred, con error y nº de reintentos) queda en
`_log/collection_log.jsonl`. Los reintentos usan backoff exponencial con jitter
y respetan `Retry-After` (política de `data.base`). `status` resume: particiones
por dataset, **huecos** (sesiones sin partición, medidas contra el calendario
NYSE) y días con fallos.

| Dataset | ¿Un hueco es recuperable? |
|---|---|
| `option_chain`, `open_interest` | **No.** La cadena de ayer no existe en ninguna fuente gratuita. El hueco es permanente; por eso `Persistent=true` en systemd y por eso el recolector captura aunque llegue tarde. |
| `consensus` | Parcialmente: el nivel actual se recupera, el vintage del día perdido no. |
| `earnings_calendar` | Parcialmente: las fechas futuras se recapturan mañana; el vintage del día perdido no. |
| `short_interest` | Sí, dentro del año rodante de FINRA (después, no). |
| `off_exchange` (Reg SHO) | Sí: FINRA mantiene archivos descargables de años. |

---

## 8. Consumo del panel desde la plataforma

```python
from earnings_alpha.collector import SnapshotStore

store = SnapshotStore("data/collector")
chains = store.read("option_chain", start=date(2026, 8, 1), end=date(2026, 8, 31))
oi     = store.read("open_interest")          # available_at = mañana real de T+1
cons   = store.read("consensus")              # vintages con as_of = fecha de captura
```

Reglas para quien construya features encima:

1. **Únase siempre por `available_at`** (`pit.asof_join`), nunca por
   `chain_date`/`oi_date`: la diferencia es exactamente el día de look-ahead del
   OI que la §2.2 explica.
2. El consenso del recolector lleva `is_point_in_time=True` y pasa
   `require_point_in_time`: es apto para `analyst_revision_drift` **desde la
   fecha de arranque del recolector**, no antes.
3. El `open_interest` de la columna homónima de `option_chain` es el de la
   sesión anterior (conocible al cierre); para el OI del propio día, usar el
   dataset `open_interest`.
4. Short volume (Reg SHO) ≠ short interest: gran parte es hedging de creadores
   de mercado (Blocher y Ringgenberg; `data_sources.md` §8.3). Archivado aquí,
   interpretado en `events.flow`.
