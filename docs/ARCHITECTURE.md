# earnings-alpha — Contrato de arquitectura

Plataforma de investigación de estrategias sobre **fundamentales y eventos de resultados**
para el universo **S&P 500 completo**.

Este documento es el **contrato vinculante** entre módulos. Todo agente que implemente
un módulo debe programar contra estas interfaces y **no modificar** las firmas aquí
definidas. Si una firma resulta insuficiente, se anota en `docs/OPEN_QUESTIONS.md`
y se implementa la alternativa menos invasiva.

---

## 0. Principios no negociables

1. **Point-in-time o nada.** Ningún dato entra en una señal antes del instante en que
   fue *públicamente conocible*. Toda serie fundamental se indexa por
   `available_at` (timestamp de disponibilidad), nunca por `period_end`.
2. **Sin sesgo de supervivencia.** El universo en la fecha `t` es la pertenencia real
   al índice en `t` (`data/seed/sp500_historical_components.csv`), no la lista de hoy.
3. **Fallo explícito.** Un proveedor sin credenciales o sin red lanza
   `ProviderUnavailable`; nunca devuelve datos vacíos silenciosamente ni inventa valores.
4. **Determinismo.** Todo componente con aleatoriedad acepta `seed: int`.
5. **Offline-testable.** Cada módulo debe poder probarse sin red, contra
   `earnings_alpha.data.synthetic`.

---

## 1. Tipos núcleo (`earnings_alpha/types.py`) — YA IMPLEMENTADO, no modificar

```python
Ticker = str          # símbolo normalizado, puntos no guiones: "BRK.B"
CIK = str             # 10 dígitos con ceros a la izquierda: "0000320193"

class Session(StrEnum):        # momento del anuncio respecto a la sesión
    BMO = "bmo"                # before market open
    AMC = "amc"                # after market close
    DMH = "dmh"                # during market hours (raro)
    UNKNOWN = "unknown"

@dataclass(frozen=True)
class EarningsEvent:
    ticker: Ticker
    cik: CIK | None
    period_end: date           # fin del trimestre fiscal reportado
    announced_at: datetime     # UTC, timestamp del anuncio
    session: Session
    fiscal_quarter: str        # "2025Q3"
    eps_actual: float | None
    eps_estimate: float | None
    revenue_actual: float | None
    revenue_estimate: float | None
    source: str
    # event_date = primera sesión de trading en que la noticia es negociable
    # (BMO -> mismo día; AMC/DMH-post-cierre -> siguiente sesión)

@dataclass(frozen=True)
class Bar:
    ticker: Ticker; ts: datetime
    open: float; high: float; low: float; close: float
    volume: float; adj_close: float | None
```

Panel canónico = `pandas.DataFrame` con **MultiIndex `(date, ticker)`** ordenado,
`date` tz-naive normalizado a medianoche UTC. Todas las funciones que devuelven
paneles respetan esta convención.

---

## 2. Mapa de módulos y **propiedad de ficheros**

Cada módulo tiene un único propietario. **No escribas fuera de tus ficheros.**

| Módulo | Ficheros | Responsabilidad |
|---|---|---|
| `types`, `config`, `errors` | `earnings_alpha/{types,config,errors}.py` | núcleo (ya hecho) |
| **universe** | `earnings_alpha/universe/*.py` | pertenencia PIT al S&P 500, updater diario, mapeo ticker↔CIK |
| **pit** | `earnings_alpha/pit/*.py` | calendario de trading, BMO/AMC, as-of joins, vintages |
| **data.base** | `earnings_alpha/data/{base,cache,synthetic}.py` | protocolo Provider, caché, rate-limit, generador sintético |
| **data.market** | `earnings_alpha/data/prices.py` | OHLCV multi-proveedor |
| **data.edgar** | `earnings_alpha/data/{edgar,fundamentals}.py` | SEC XBRL, 8-K 2.02, Form 4 |
| **data.estimates** | `earnings_alpha/data/estimates.py` | calendario de resultados, consenso EPS/ingresos |
| **data.options** | `earnings_alpha/data/options.py` | cadenas, IV, OI, volumen de opciones |
| **data.flow** | `earnings_alpha/data/shortinterest.py` | short interest FINRA, volumen off-exchange (ATS) |
| **factors** | `earnings_alpha/factors/*.py` | ángulo A: factores fundamentales cross-section |
| **events** | `earnings_alpha/events/*.py` | ángulo B: estudio de eventos y detección de flujo informado |
| **signals** | `earnings_alpha/signals/*.py` | transformaciones y combinación de señales |
| **backtest** | `earnings_alpha/backtest/*.py` | motores de backtest y costes |
| **stats** | `earnings_alpha/stats/*.py` | IC, significancia, validación cruzada purgada |
| **reports** | `earnings_alpha/reports/*.py` | tearsheets HTML |
| **cli** | `earnings_alpha/cli.py` | entrypoints |

---

## 3. Interfaces por módulo

### 3.1 `universe`

```python
class UniverseProvider(Protocol):
    def members_on(self, d: date) -> list[Ticker]: ...
    def membership_panel(self, start: date, end: date) -> pd.DataFrame:
        """DataFrame booleano index=date, columns=ticker. True = en el índice ese día."""
    def cik_for(self, t: Ticker, on: date | None = None) -> CIK | None: ...
    def sector_for(self, t: Ticker) -> str | None: ...

class SP500Universe(UniverseProvider):
    """PIT desde data/seed/sp500_historical_components.csv (1996→).
    `refresh()` reconstruye desde fuentes vivas y *append-only* añade el día de hoy;
    jamás reescribe historia ya registrada."""
    def refresh(self, sources: list[str] | None = None) -> RefreshReport: ...
```

Reglas: normalizar tickers (`BRK-B`→`BRK.B`), registrar cambios de ticker,
y para fechas anteriores al primer snapshot lanzar `InsufficientHistory`.

### 3.2 `pit`

```python
class TradingCalendar:
    def sessions(self, start: date, end: date) -> pd.DatetimeIndex: ...
    def next_session(self, d: date) -> date: ...
    def shift(self, d: date, n: int) -> date:  # n negativo = hacia atrás
    def is_session(self, d: date) -> bool: ...

def tradable_date(announced_at: datetime, session: Session, cal: TradingCalendar) -> date:
    """Primera sesión en que la información es explotable. BMO -> mismo día.
    AMC -> siguiente sesión. Es la función más crítica del repo: un error de un día
    aquí convierte cualquier backtest en look-ahead."""

def asof_join(signal: pd.DataFrame, panel: pd.DataFrame, lag_days: int = 0) -> pd.DataFrame:
    """Une por (date,ticker) usando SOLO valores con available_at <= date - lag."""
```

### 3.3 `data.base`

```python
class Provider(Protocol):
    name: str
    def available(self) -> bool: ...      # credenciales + red

class ProviderRegistry:
    def register(self, kind: str, p: Provider, priority: int) -> None: ...
    def resolve(self, kind: str) -> Provider:
        """Devuelve el proveedor disponible de mayor prioridad; si ninguno,
        lanza ProviderUnavailable con la lista de lo que falta (clave/red)."""
```

Caché: `DiskCache(root)` con claves `(kind, provider, params_hash)`, formato parquet,
TTL configurable, y modo `offline=True` que sirve solo de caché.

`synthetic.SyntheticMarket(seed)` genera un panel realista: precios GBM con
volatilidad por sector, eventos de resultados trimestrales con sorpresas
correlacionadas con el drift posterior, run-up de volumen pre-evento en un
subconjunto configurable de "eventos con filtración" (para poder **validar que el
detector detecta**), y datos de opciones/short interest coherentes.

### 3.4 `factors` (ángulo A — continuo todo el año)

```python
class Factor(Protocol):
    name: str; requires: list[str]        # tipos de datos necesarios
    def compute(self, ctx: FactorContext) -> pd.Series:
        """Series con MultiIndex (date,ticker). Valor mayor = más alcista."""

@dataclass
class FactorContext:
    dates: pd.DatetimeIndex; universe: UniverseProvider
    prices: pd.DataFrame; fundamentals: pd.DataFrame
    estimates: pd.DataFrame; events: pd.DataFrame
    calendar: TradingCalendar
```

Factores mínimos a implementar (cada uno con docstring citando la referencia):
SUE (estandarizado por sigma de sorpresas históricas), sorpresa de ingresos,
momentum de revisiones de analistas, drift post-resultados (PEAD),
crecimiento de ventas y su aceleración, tendencia de margen bruto/operativo,
FCF yield, earnings yield, Piotroski F-score, acumulaciones (accruals) de Sloan,
calidad de beneficios (CFO/NI), apalancamiento, y cambios de guidance.

### 3.5 `events` (ángulo B — ventana de resultados)

```python
def event_windows(events: pd.DataFrame, cal: TradingCalendar,
                  pre: int = 30, post: int = 60) -> pd.DataFrame:
    """Expande cada evento a tiempo-evento tau ∈ [-pre, +post] en días de sesión."""

def abnormal_returns(prices, events, model: Literal["market","ff3","mean"] = "market",
                     estimation: tuple[int,int] = (-250, -40)) -> pd.DataFrame:
    """AR/CAR por tau. La ventana de estimación termina ANTES de la ventana de
    evento para no contaminar."""

class PreEventFeatures:
    """Huella de negociación informada en [T-N, T-1]. NO usa información privada:
    solo datos públicos de mercado que reflejan el comportamiento de otros.
    Devuelve un DataFrame indexado por (event_id) con, al menos:
      volume_runup, turnover_zscore, abnormal_volume_5/10/20d,
      order_imbalance_proxy, oi_buildup_calls/puts, put_call_volume_ratio,
      iv_skew_25delta, vol_spread (call_iv - put_iv, Cremers-Weinbaum),
      iv_term_slope, short_interest_delta, off_exchange_share_delta,
      insider_net_buy_form4, pre_event_car_5/10/20d, analyst_revision_drift.
    """
    def compute(self, ctx: EventContext) -> pd.DataFrame: ...

class SurpriseModel:
    """Predice signo/magnitud de la sorpresa y del retorno del evento a partir de
    PreEventFeatures. Validación temporal estricta (purged CV, embargo)."""
    def fit(self, X, y, groups_by_date) -> Self: ...
    def predict(self, X) -> pd.Series: ...
```

**Nota legal/metodológica obligatoria en el docstring del módulo:** todas las
features derivan de datos **públicos** (volumen, precios, cadenas de opciones,
short interest agregado FINRA, Form 4 ya publicados). El objetivo es *detectar la
huella estadística* de negociación informada, no acceder a información privilegiada.

### 3.6 `signals`

`zscore`, `winsorize(q)`, `rank_pct`, `neutralize(by=["sector","size","beta"])`,
`combine(weights|ic_weighted|orthogonalize)`. Todas operan por fecha (cross-section).

### 3.7 `backtest`

```python
class CrossSectionalBacktest:
    def run(self, scores: pd.Series, prices: pd.DataFrame, *,
            n_quantiles: int = 5, rebalance: str = "W-FRI",
            long_short: bool = True, costs: CostModel,
            max_weight: float = 0.02, adv_participation: float = 0.05) -> BacktestResult: ...

class EventBacktest:
    """Entra `entry_offset` sesiones antes/después del evento, sale en `exit_offset`.
    Modela gap overnight de forma explícita (el evento AMC se negocia en el gap)."""
```

`CostModel`: spread por tramo de liquidez + impacto sqrt(participación) + comisión
+ coste de préstamo para cortos. Nada de "0.05% fijo" sin justificar.

### 3.8 `stats`

IC/rank-IC con t de Newey-West, spreads por quintil, Sharpe deflactado
(Bailey–López de Prado), corrección por comparaciones múltiples (Benjamini-Hochberg),
CV purgada con embargo, bootstrap estacionario. **Toda métrica de rendimiento debe
reportar su intervalo de confianza**; un Sharpe sin banda de error no se acepta.

---

## 4. Convenciones

- Python ≥3.11, tipado estricto, `ruff` + `mypy` limpios.
- Docstrings en español; identificadores y API en inglés.
- Cada factor/feature cita su referencia académica en el docstring.
- Tests con `pytest`; todo módulo trae tests que corren **sin red**.
- Nada de secretos en el repo: claves por variable de entorno (`config.Settings`).

## 5. Estado de red del entorno de desarrollo

El contenedor de desarrollo tiene el egress restringido: SEC, Yahoo, Nasdaq, Polygon,
FMP, FINRA y similares devuelven 403 en el CONNECT del proxy. GitHub y PyPI sí son
alcanzables. Por eso:

- Los adaptadores de red se implementan **completos y probados contra fixtures
  grabados/sintéticos**, no contra la API viva.
- La verificación end-to-end contra APIs reales queda pendiente de ejecución en la
  máquina del usuario (ver `docs/OPEN_QUESTIONS.md`).
