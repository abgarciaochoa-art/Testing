"""Infraestructura del módulo `factors`: protocolo, contexto y registro.

Implementa el contrato §3.4 de `docs/ARCHITECTURE.md`:

- `Factor`: protocolo que cumple todo factor. `compute(ctx)` devuelve una
  `pd.Series` con MultiIndex ``(date, ticker)`` donde **valor mayor = más
  alcista**. Los factores cuya lectura natural es "menos es mejor" se devuelven
  con el signo cambiado y lo documentan (`fundamental_factors.md` §1.3).
- `FactorContext`: paquete de datos que recibe cada factor. Sus campos son
  exactamente los del contrato; no se añaden ni renombran.
- `FactorRegistry`: registro con decorador, para que la CLI y los informes
  puedan enumerar e instanciar factores por nombre sin importar cada módulo.

Además contiene la maquinaria común que comparten los factores de evento:

- `spread_event_values`: proyecta valores fechados por evento (un SUE, una
  puntuación de guidance) sobre el panel canónico ``(date, ticker)`` respetando
  la primera sesión negociable, con soporte de horizonte finito y decaimiento.
- `StaticUniverse`: implementación mínima de `UniverseProvider` para tests y
  datos sintéticos (pertenencia constante).
- `context_from_synthetic`: construye un `FactorContext` completo desde
  `data.synthetic.SyntheticMarket`, que es el banco de pruebas offline del repo.

Principios point-in-time que esta capa hace cumplir por construcción:

1. Ningún valor de evento entra en el panel antes de su `tradable_date`
   (`pit.tradable_date`): un anuncio AMC no existe para la señal hasta la
   sesión siguiente.
2. La proyección usa `pit.asof_join` (estrictamente hacia atrás) o aritmética
   de posiciones sobre el calendario de sesiones; nunca un `reindex` con
   relleno bilateral.
3. Antes del primer evento negociable de un ticker el factor es NaN, no cero:
   un cero sería una posición neutral *fabricada* y además burlaría la
   auditoría `pit.assert_no_lookahead(check_first_event=True)`.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from earnings_alpha.errors import (
    ConfigError,
    DataQualityError,
    InsufficientHistory,
    ProviderUnavailable,
)
from earnings_alpha.pit import TradingCalendar, asof_join, get_calendar
from earnings_alpha.types import CIK, Ticker, normalize_ticker

__all__ = [  # noqa: RUF022 - orden temático, no alfabético
    "Factor",
    "FactorContext",
    "FactorRegistry",
    "default_registry",
    "register_factor",
    "StaticUniverse",
    "UniverseLike",
    "build_panel_index",
    "spread_event_values",
    "spread_event_frame",
    "context_from_synthetic",
    "require_columns",
    "empty_panel",
]


# ---------------------------------------------------------------------------
# Protocolos
# ---------------------------------------------------------------------------


@runtime_checkable
class UniverseLike(Protocol):
    """Subconjunto del protocolo `universe.UniverseProvider` que usa `factors`.

    Se declara aquí para no acoplar el módulo a la implementación concreta:
    cualquier objeto con esta forma (incluido `SP500Universe`) sirve.
    """

    def members_on(self, d: dt.date) -> list[Ticker]: ...

    def membership_panel(self, start: dt.date, end: dt.date) -> pd.DataFrame: ...

    def cik_for(self, t: Ticker, on: dt.date | None = None) -> CIK | None: ...

    def sector_for(self, t: Ticker) -> str | None: ...


@runtime_checkable
class Factor(Protocol):
    """Un factor cross-section (contrato §3.4).

    `requires` enumera los tipos de dato que necesita (nomenclatura de
    `data.base.DataKind`: ``"prices"``, ``"fundamentals"``, ``"estimates"``,
    ``"earnings_calendar"``...), lo que permite a un orquestador comprobar
    disponibilidad antes de computar.
    """

    name: str
    requires: list[str]

    def compute(self, ctx: FactorContext) -> pd.Series:
        """Serie con MultiIndex ``(date, ticker)``. Valor mayor = más alcista."""
        ...


@dataclass(slots=True)
class FactorContext:
    """Datos de entrada de un factor (contrato §3.4, campos exactos).

    Convenciones de cada campo:

    - `dates`: sesiones de decisión, normalizadas a medianoche, tz-naive.
    - `prices`: panel canónico ``(date, ticker)`` con al menos ``close``;
      idealmente el de `SyntheticMarket.prices()` o `data.prices`.
    - `fundamentals`: tabla larga por ``(ticker, period_end)`` con
      `available_at` point-in-time (nota de prensa) y `filed_at` (10-Q/10-K).
    - `estimates`: fotos de consenso estilo `types.EstimateSnapshot`, una fila
      por ``(ticker, period_end, as_of)``.
    - `events`: tabla de anuncios con `announced_at`, `session` y, si ya está
      resuelta, `event_date` (primera sesión negociable).
    - `calendar`: calendario NYSE de `pit.get_calendar`.
    """

    dates: pd.DatetimeIndex
    universe: UniverseLike
    prices: pd.DataFrame
    fundamentals: pd.DataFrame
    estimates: pd.DataFrame
    events: pd.DataFrame
    calendar: TradingCalendar

    def require(self, *kinds: str) -> None:
        """Falla con `ProviderUnavailable` si falta alguno de los datos pedidos.

        Es la materialización de la regla "fallo explícito" del contrato §0.3:
        un factor jamás debe computarse sobre una tabla vacía y devolver un
        panel de NaN en silencio.
        """
        tables: dict[str, pd.DataFrame | pd.DatetimeIndex] = {
            "prices": self.prices,
            "fundamentals": self.fundamentals,
            "estimates": self.estimates,
            "events": self.events,
        }
        for kind in kinds:
            if kind not in tables:
                msg = f"tipo de dato desconocido en FactorContext: {kind!r}"
                raise ConfigError(msg)
            table = tables[kind]
            if table is None or len(table) == 0:
                raise ProviderUnavailable(
                    kind,
                    f"FactorContext.{kind} está vacío: el factor no puede computarse "
                    "sin ese dato (contrato §0.3: fallo explícito, no NaN silencioso)",
                )

    def tickers(self) -> list[Ticker]:
        """Universo de símbolos del contexto: los que aparecen en `prices`."""
        return sorted(set(self.prices.index.get_level_values("ticker")))

    def panel_index(self) -> pd.MultiIndex:
        """MultiIndex canónico ``(date, ticker)`` = `dates` x tickers del panel."""
        return build_panel_index(self.dates, self.tickers())


# ---------------------------------------------------------------------------
# Registro
# ---------------------------------------------------------------------------


class FactorRegistry:
    """Registro de factores por nombre, con decorador de registro.

    Ejemplo::

        registry = FactorRegistry()

        @registry.register()
        class MyFactor:
            name = "my_factor"
            requires = ["events"]
            def compute(self, ctx): ...

        factor = registry.create("my_factor")

    El registro guarda *fábricas* (la clase o un callable), no instancias: los
    factores llevan parámetros de construcción (horizonte, base de la sorpresa)
    y cada consumidor debe poder instanciar el suyo.
    """

    def __init__(self) -> None:
        self._factories: dict[str, Callable[..., Factor]] = {}

    # -- registro -----------------------------------------------------------

    def register(
        self,
        name: str | None = None,
        *,
        overwrite: bool = False,
    ) -> Callable[[Callable[..., Factor]], Callable[..., Factor]]:
        """Decorador que registra una clase (o fábrica) de factor.

        Si `name` se omite se usa el atributo de clase ``name``. Registrar dos
        veces el mismo nombre sin `overwrite=True` es `ConfigError`: un choque
        de nombres silencioso haría que "el factor X" significase cosas
        distintas según el orden de importación.
        """

        def _decorator(factory: Callable[..., Factor]) -> Callable[..., Factor]:
            key = name or getattr(factory, "name", None)
            if not key or not isinstance(key, str):
                msg = (
                    f"no se puede registrar {factory!r}: ni se pasó `name` ni la "
                    "clase define un atributo de clase `name`"
                )
                raise ConfigError(msg)
            if not callable(factory):
                msg = f"la fábrica registrada para {key!r} no es invocable"
                raise ConfigError(msg)
            if key in self._factories and not overwrite:
                msg = (
                    f"el factor {key!r} ya está registrado; usa overwrite=True si la "
                    "sustitución es deliberada"
                )
                raise ConfigError(msg)
            self._factories[key] = factory
            return factory

        return _decorator

    # -- consulta -----------------------------------------------------------

    def names(self) -> list[str]:
        """Nombres registrados, ordenados."""
        return sorted(self._factories)

    def get(self, name: str) -> Callable[..., Factor]:
        """Fábrica registrada bajo `name`; `ConfigError` con la lista si no existe."""
        try:
            return self._factories[name]
        except KeyError:
            msg = f"factor desconocido: {name!r}. Registrados: {self.names()}"
            raise ConfigError(msg) from None

    def create(self, name: str, **kwargs: object) -> Factor:
        """Instancia el factor `name` con los parámetros dados."""
        factory = self.get(name)
        instance = factory(**kwargs)
        if not isinstance(instance, Factor):
            msg = (
                f"la fábrica de {name!r} devolvió {type(instance).__name__}, que no "
                "cumple el protocolo Factor (name, requires, compute)"
            )
            raise ConfigError(msg)
        return instance

    def compute(self, name: str, ctx: FactorContext, **kwargs: object) -> pd.Series:
        """Atajo: instancia y computa en una llamada."""
        return self.create(name, **kwargs).compute(ctx)

    def __contains__(self, name: object) -> bool:
        return name in self._factories

    def __len__(self) -> int:
        return len(self._factories)

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __repr__(self) -> str:  # pragma: no cover - cosmético
        return f"FactorRegistry({self.names()})"


default_registry = FactorRegistry()
"""Registro global del paquete: los módulos `surprise`, `revisions` y
`guidance` registran aquí sus factores al importarse."""

register_factor = default_registry.register
"""Decorador de conveniencia sobre `default_registry`."""


# ---------------------------------------------------------------------------
# Universo estático
# ---------------------------------------------------------------------------


class StaticUniverse:
    """`UniverseProvider` mínimo: pertenencia constante para una lista fija.

    Sirve para tests y para el mercado sintético, donde no hay altas ni bajas
    del índice. **No** debe usarse con datos reales multi-año: ahí la
    pertenencia constante es exactamente el sesgo de supervivencia que
    `SP500Universe` existe para evitar (contrato §0.2).
    """

    def __init__(
        self,
        tickers: Sequence[Ticker],
        *,
        sectors: dict[Ticker, str] | pd.Series | None = None,
        ciks: dict[Ticker, CIK] | None = None,
    ) -> None:
        if len(tickers) == 0:
            msg = "StaticUniverse necesita al menos un ticker"
            raise ConfigError(msg)
        self._tickers = sorted({normalize_ticker(t) for t in tickers})
        if isinstance(sectors, pd.Series):
            sectors = {str(k): str(v) for k, v in sectors.items()}
        self._sectors = dict(sectors or {})
        self._ciks = dict(ciks or {})

    def members_on(self, d: dt.date) -> list[Ticker]:  # firma del protocolo
        return list(self._tickers)

    def membership_panel(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        dates = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D", name="date")
        return pd.DataFrame(True, index=dates, columns=pd.Index(self._tickers, name="ticker"))

    def cik_for(self, t: Ticker, on: dt.date | None = None) -> CIK | None:
        return self._ciks.get(normalize_ticker(t))

    def sector_for(self, t: Ticker) -> str | None:
        return self._sectors.get(normalize_ticker(t))


# ---------------------------------------------------------------------------
# Utilidades de panel
# ---------------------------------------------------------------------------


def require_columns(df: pd.DataFrame, cols: Sequence[str], *, name: str) -> None:
    """Falla con `DataQualityError` si a `df` le falta alguna columna."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        msg = f"a `{name}` le faltan columnas obligatorias: {missing}; tiene {list(df.columns)}"
        raise DataQualityError(msg)


def build_panel_index(dates: pd.DatetimeIndex, tickers: Sequence[Ticker]) -> pd.MultiIndex:
    """MultiIndex canónico ``(date, ticker)`` ordenado (contrato §1)."""
    idx = pd.DatetimeIndex(pd.to_datetime(dates)).normalize().unique().sort_values()
    return pd.MultiIndex.from_product(
        [idx.as_unit("ns"), pd.Index(sorted(set(tickers)), dtype=object)],
        names=["date", "ticker"],
    )


def empty_panel(dates: pd.DatetimeIndex, tickers: Sequence[Ticker], name: str) -> pd.Series:
    """Panel canónico todo-NaN. Para factores que declaran "sin dato" honesto."""
    return pd.Series(np.nan, index=build_panel_index(dates, tickers), name=name, dtype=float)


def _positions_on_grid(
    dates: pd.DatetimeIndex,
    when: pd.Series | pd.DatetimeIndex,
    calendar: TradingCalendar | None,
) -> np.ndarray:
    """Posición de cada fecha en la rejilla de sesiones del panel.

    - Fecha dentro del rango: `searchsorted` a la izquierda (una fecha que no
      es sesión se lleva a la primera sesión posterior: la información fechada
      un sábado es negociable el lunes, nunca el viernes anterior).
    - Fecha posterior al final del panel: ``len(grid)`` (el filtro posterior
      la descarta).
    - Fecha anterior al inicio del panel: posición **negativa** contada en
      sesiones reales del calendario, para que una ventana de vida finita que
      empezó antes del panel entre ya parcialmente consumida y no se
      "reinicie" en la primera fecha del panel. Sin calendario no se puede
      contar y esas fechas se marcan como ``-len(grid)-horizonte`` efectivo
      (fuera de todo alcance), lo que las excluye de forma conservadora.
    """
    grid = dates.to_numpy(dtype="datetime64[ns]")
    ts = pd.DatetimeIndex(pd.to_datetime(when)).normalize().to_numpy(dtype="datetime64[ns]")
    pos = np.searchsorted(grid, ts, side="left").astype("int64")
    before = ts < grid[0]
    if before.any():
        if calendar is None:
            pos[before] = np.iinfo(np.int64).min // 4
        else:
            first = pd.Timestamp(grid[0]).date()
            for i in np.flatnonzero(before):
                start = pd.Timestamp(ts[i]).date()
                # Sesiones en [start, primera_fecha_del_panel), contadas con el
                # calendario real: la del propio evento incluida.
                n_before = len(calendar.sessions(start, first)) - 1
                pos[i] = -max(n_before, 0)
    return pos


def spread_event_values(
    values: pd.DataFrame,
    dates: pd.DatetimeIndex,
    tickers: Sequence[Ticker],
    *,
    value_col: str = "value",
    date_col: str = "tradable_date",
    horizon: int | None = None,
    decay: float | None = None,
    include_event_day: bool = True,
    dead_value: float = np.nan,
    max_staleness_days: int | None = None,
    calendar: TradingCalendar | None = None,
    name: str = "factor",
) -> pd.Series:
    """Proyecta valores por evento sobre el panel canónico ``(date, ticker)``.

    Dos regímenes:

    - ``horizon=None`` (paso mantenido): cada valor se arrastra desde su
      `tradable_date` hasta el siguiente evento del mismo ticker, con tope
      opcional de obsolescencia `max_staleness_days` (días naturales) que evita
      que el último evento de un ticker que deja de reportar se propague
      indefinidamente. Implementado con `pit.asof_join`, la única vía admitida
      de unión temporal del repo.
    - ``horizon=H`` (vida finita): el valor vive `H` sesiones y después vale
      `dead_value`. `include_event_day=False` reproduce la convención del PEAD
      de `fundamental_factors.md` §4.1 (``0 < s <= H``): la señal empieza en la
      sesión *siguiente* a la negociable. El decaimiento opcional es
      exponencial, ``w(k) = exp(-k / decay)`` con `k` sesiones desde la primera
      sesión viva.

    En ambos regímenes, **antes del primer evento negociable del ticker el
    panel es NaN** (nunca `dead_value`): así lo exige la auditoría
    `pit.assert_no_lookahead(check_first_event=True)`.

    Parámetros
    ----------
    values:
        DataFrame con columnas ``ticker``, `date_col` (fecha negociable ya
        resuelta con `pit.tradable_date`) y `value_col`. Filas con valor NaN se
        ignoran (no pisan al evento anterior).
    dates:
        Sesiones de decisión (rejilla del panel).
    """
    require_columns(values, ["ticker", date_col, value_col], name="values")
    if horizon is not None and horizon < 1:
        msg = f"horizon debe ser >= 1 o None; recibido {horizon}"
        raise ConfigError(msg)
    if decay is not None and decay <= 0:
        msg = f"decay debe ser > 0 o None; recibido {decay}"
        raise ConfigError(msg)

    grid = pd.DatetimeIndex(pd.to_datetime(dates)).normalize().unique().sort_values().as_unit("ns")
    if len(grid) == 0:
        msg = "la rejilla de fechas está vacía"
        raise InsufficientHistory(msg)
    ticker_list = sorted(set(tickers))
    out = pd.Series(
        np.nan, index=build_panel_index(grid, ticker_list), name=name, dtype=float
    )

    work = values[["ticker", date_col, value_col]].copy()
    work["ticker"] = work["ticker"].astype(str)
    work = work[work["ticker"].isin(set(ticker_list))]
    work = work.dropna(subset=[value_col, date_col])
    if len(work) == 0:
        # Sin ningún evento con valor no hay nada que proyectar; devolver NaN
        # en silencio ocultaría un problema de datos aguas arriba.
        msg = (
            "ningún evento con valor no nulo que proyectar sobre el panel "
            f"(columna {value_col!r})"
        )
        raise InsufficientHistory(msg)

    if horizon is None:
        frame = spread_event_frame(
            work,
            grid,
            ticker_list,
            value_col=value_col,
            date_col=date_col,
            max_staleness_days=max_staleness_days,
            name=name,
        )
        series = frame[name]
        series.name = name
        return series

    # --- horizonte finito --------------------------------------------------
    pos = _positions_on_grid(grid, work[date_col], calendar)
    # Se conservan los eventos cuya ventana viva interseca el panel: los
    # anteriores entran con la vida parcialmente consumida (pos negativa).
    start_offset = 0 if include_event_day else 1
    keep = (pos + start_offset + horizon - 1 >= 0) & (pos < len(grid))
    work = work.loc[keep]
    pos = pos[keep]
    if len(work) == 0:
        msg = "ningún evento con ventana viva dentro del rango de fechas del panel"
        raise InsufficientHistory(msg)

    offsets = np.arange(horizon)
    n_ev = len(work)
    row_pos = (pos[:, None] + start_offset + offsets[None, :]).ravel()
    row_ticker = np.repeat(work["ticker"].to_numpy(), horizon)
    base_vals = np.repeat(work[value_col].to_numpy(dtype=float), horizon)
    if decay is not None:
        weights = np.exp(-offsets / float(decay))
        base_vals = base_vals * np.tile(weights, n_ev)
    # Orden del evento: si dos eventos del mismo ticker solapan ventanas, el
    # más reciente (mayor tradable_date) debe ganar.
    order_key = np.repeat(work[date_col].to_numpy(dtype="datetime64[ns]"), horizon)

    valid = (row_pos >= 0) & (row_pos < len(grid))
    frame = pd.DataFrame(
        {
            "date": grid.to_numpy()[row_pos[valid]],
            "ticker": row_ticker[valid],
            "value": base_vals[valid],
            "order": order_key[valid],
        }
    )
    frame = frame.sort_values(["date", "ticker", "order"], kind="mergesort")
    frame = frame.drop_duplicates(subset=["date", "ticker"], keep="last")
    live = frame.set_index(["date", "ticker"])["value"]

    if not np.isnan(dead_value):
        # `dead_value` (típicamente 0 para PEAD) solo desde la primera sesión
        # viva de cada ticker en adelante; antes, NaN.
        first_live_pos = (
            pd.Series(pos + start_offset, index=work["ticker"].to_numpy())
            .groupby(level=0)
            .min()
        )
        date_pos = pd.Series(np.arange(len(grid)), index=grid)
        pos_level = out.index.get_level_values("date").map(date_pos).to_numpy()
        first_level = (
            out.index.get_level_values("ticker").map(first_live_pos).to_numpy(dtype=float)
        )
        alive = pos_level >= np.where(np.isnan(first_level), np.inf, first_level)
        out[alive] = dead_value

    out.loc[live.index] = live.to_numpy()
    out.name = name
    return out


def spread_event_frame(
    values: pd.DataFrame,
    dates: pd.DatetimeIndex,
    tickers: Sequence[Ticker],
    *,
    value_col: str = "value",
    date_col: str = "tradable_date",
    max_staleness_days: int | None = None,
    name: str = "factor",
) -> pd.DataFrame:
    """Régimen de paso mantenido con columna `available_at` auditable.

    Igual que `spread_event_values(horizon=None)`, pero devuelve un DataFrame
    con el valor **y** la fecha negociable del evento del que procede cada
    celda (columna ``available_at``). Ese acompañante es lo que permite pasar
    el resultado por `pit.assert_no_lookahead` sin reconstruir nada: cada valor
    lleva consigo la prueba de cuándo fue explotable por primera vez.
    """
    require_columns(values, ["ticker", date_col, value_col], name="values")
    grid = pd.DatetimeIndex(pd.to_datetime(dates)).normalize().unique().sort_values().as_unit("ns")
    ticker_list = sorted(set(tickers))
    work = values[["ticker", date_col, value_col]].copy()
    work["ticker"] = work["ticker"].astype(str)
    work = work[work["ticker"].isin(set(ticker_list))]
    work = work.dropna(subset=[value_col, date_col])
    if len(work) == 0:
        msg = (
            "ningún evento con valor no nulo que proyectar sobre el panel "
            f"(columna {value_col!r})"
        )
        raise InsufficientHistory(msg)
    panel = work.rename(columns={date_col: "available_at", value_col: name})
    joined = asof_join(
        grid,
        panel,
        date_col="date",
        ticker_col="ticker",
        available_col="available_at",
        value_cols=[name],
        max_staleness_days=max_staleness_days,
        # "date_start" incluye la coincidencia exacta: un valor fechado en la
        # medianoche de su `tradable_date` está vivo ESA sesión, que es
        # exactamente la semántica de la primera sesión explotable.
        cutoff="date_start",
        keep_available_at=True,
    )
    return joined.reindex(build_panel_index(grid, ticker_list))


# ---------------------------------------------------------------------------
# Contexto desde el mercado sintético
# ---------------------------------------------------------------------------


def context_from_synthetic(
    market: object,
    *,
    include_prehistory: bool = True,
) -> FactorContext:
    """Construye un `FactorContext` completo desde un `SyntheticMarket`.

    `include_prehistory=True` incorpora los trimestres generados antes de
    `start`, imprescindibles para que SUE (13 trimestres, §2.1 de
    `fundamental_factors.md`) tenga histórico desde el primer día del panel.

    Se tipa como `object` y se accede por duck-typing para no imponer una
    dependencia dura de `data.synthetic` a los consumidores de `factors.base`.
    """
    for attr in ("prices", "events", "fundamentals", "estimates", "sectors", "cik_map"):
        if not hasattr(market, attr):
            msg = f"el mercado no expone `{attr}()`: no parece un SyntheticMarket"
            raise ConfigError(msg)
    prices = market.prices()
    events = market.events(include_prehistory=include_prehistory)
    fundamentals = market.fundamentals()
    estimates = market.estimates()
    dates = pd.DatetimeIndex(prices.index.get_level_values("date").unique()).sort_values()
    universe = StaticUniverse(
        list(market.tickers), sectors=market.sectors(), ciks=market.cik_map()
    )
    calendar = getattr(market, "calendar", None) or get_calendar()
    return FactorContext(
        dates=dates,
        universe=universe,
        prices=prices,
        fundamentals=fundamentals,
        estimates=estimates,
        events=events,
        calendar=calendar,
    )
