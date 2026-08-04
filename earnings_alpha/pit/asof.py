"""Resolución point-in-time: fecha negociable de un evento y uniones as-of.

Este es el fichero más delicado del repositorio. Todo lo demás —factores,
estudio de eventos, backtests— consume el instante en que un dato pasó a ser
públicamente conocible. Un error de un día aquí no produce un error del 1%:
produce un backtest que "funciona" porque conoce el futuro.

Tres piezas:

1. `classify_session` / `tradable_date`: del timestamp del anuncio a la primera
   sesión en la que la noticia es explotable.
2. `asof_join`: unión de un panel de decisión `(date, ticker)` con hechos
   fechados por `available_at`, estrictamente hacia atrás y vectorizada.
3. `assert_no_lookahead`: auditoría reutilizable por los tests de otros módulos.

Referencias metodológicas:
  - Foster, Olsen y Shevlin (1984), *Earnings Releases, Anomalies, and the
    Behavior of Security Returns*: la ventana de evento se ancla al día de
    anuncio negociable, no al cierre del trimestre.
  - Bernard y Thomas (1989), *Post-Earnings-Announcement Drift*: el drift se mide
    desde la sesión de anuncio; desplazar el ancla un día cambia el signo de los
    primeros días de CAR.
  - Ball y Brown (1968): el estudio de eventos original, que ya distingue entre
    la fecha del informe y la fecha en que el mercado pudo reaccionar.
  - DellaVigna y Pollet (2009), *Investor Inattention and Friday Earnings
    Announcements*: motivación empírica de tratar con cuidado los anuncios de
    viernes por la tarde, cuya sesión negociable es el lunes.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime
from typing import Literal

import numpy as np
import pandas as pd

from earnings_alpha.errors import (
    CalendarError,
    DataQualityError,
    InsufficientHistory,
    LookAheadError,
)
from earnings_alpha.pit.calendar import (
    TradingCalendar,
    eastern_offsets_for,
    get_calendar,
    utc_to_eastern,
)
from earnings_alpha.types import Session

__all__ = [  # noqa: RUF022 - orden temático, no alfabético
    "parse_session",
    "classify_session",
    "classify_sessions",
    "tradable_date",
    "tradable_dates",
    "asof_join",
    "assert_no_lookahead",
    "to_naive_utc",
    "CutoffPolicy",
]

CutoffPolicy = Literal["date_start", "session_open", "date_end"]
"""Política de corte temporal de `asof_join`.

- ``"date_start"`` (por defecto): el corte es la medianoche UTC de la fecha de
  decisión. Es la política **conservadora**: un hecho publicado a las 21:30 UTC
  del día D no se usa para la señal fechada en D, sino a partir de D+1.
- ``"session_open"``: el corte es la apertura real de la sesión (9:30 ET, o la
  apertura efectiva si hubo apertura tardía). Es la política correcta para una
  estrategia que ejecuta en la apertura, y requiere calendario.
- ``"date_end"``: el corte es el final del día natural de la fecha de decisión.
  **Solo** es válida si la ejecución ocurre en el cierre y el dato se conoció
  durante la sesión; usarla con ejecución en apertura es look-ahead.
"""

# Sinónimos que usan los proveedores para el momento del anuncio. La lista está
# en minúsculas y sin puntuación; `parse_session` normaliza antes de buscar.
_SESSION_SYNONYMS: dict[str, Session] = {
    "bmo": Session.BMO,
    "before market open": Session.BMO,
    "before open": Session.BMO,
    "beforemarket": Session.BMO,
    "pre": Session.BMO,
    "premarket": Session.BMO,
    "pre market": Session.BMO,
    "pre-market": Session.BMO,
    "morning": Session.BMO,
    "am": Session.BMO,
    "amc": Session.AMC,
    "after market close": Session.AMC,
    "after close": Session.AMC,
    "aftermarket": Session.AMC,
    "post": Session.AMC,
    "postmarket": Session.AMC,
    "post market": Session.AMC,
    "post-market": Session.AMC,
    "evening": Session.AMC,
    "pm": Session.AMC,
    "dmh": Session.DMH,
    "during market hours": Session.DMH,
    "during": Session.DMH,
    "intraday": Session.DMH,
    "unknown": Session.UNKNOWN,
    "": Session.UNKNOWN,
    "--": Session.UNKNOWN,
    "n/a": Session.UNKNOWN,
    "na": Session.UNKNOWN,
    "none": Session.UNKNOWN,
    "tns": Session.UNKNOWN,  # "time not supplied", código habitual de Zacks/Nasdaq
}


def parse_session(raw: object) -> Session:
    """Normaliza la etiqueta de sesión de un proveedor a `types.Session`.

    Devuelve `Session.UNKNOWN` ante cualquier valor no reconocido, nulo o vacío.
    Nunca adivina BMO: el valor por defecto silencioso "before market open" es el
    origen clásico de un día entero de look-ahead en los datasets de calendario
    de resultados.
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return Session.UNKNOWN
    if isinstance(raw, Session):
        return raw
    key = str(raw).strip().lower().replace("_", " ").replace(".", "")
    key = " ".join(key.split())
    return _SESSION_SYNONYMS.get(key, Session.UNKNOWN)


def to_naive_utc(values: object, *, assume_utc_if_naive: bool = True) -> pd.DatetimeIndex:
    """Convierte instantes heterogéneos a `DatetimeIndex` tz-naive en UTC.

    Convención del repo (`types.EarningsEvent`): los timestamps se almacenan en
    UTC. Un timestamp sin zona se interpreta como UTC, **jamás** como hora local
    de Nueva York; interpretarlo como local desplazaría el evento 4-5 horas y
    podría convertir un AMC en DMH. Con `assume_utc_if_naive=False` un valor sin
    zona lanza `DataQualityError` en vez de asumir nada.
    """
    if isinstance(values, pd.DatetimeIndex):
        idx = values
    else:
        try:
            idx = pd.DatetimeIndex(pd.to_datetime(values))
        except (TypeError, ValueError):
            # Mezcla de zonas horarias: pandas exige `utc=True` para unificarlas.
            idx = pd.DatetimeIndex(pd.to_datetime(values, utc=True))
    if idx.tz is not None:
        return pd.DatetimeIndex(idx.tz_convert("UTC").tz_localize(None)).as_unit("ns")
    if not assume_utc_if_naive and len(idx) > 0:
        msg = "hay timestamps sin zona horaria; sin zona no hay point-in-time verificable"
        raise DataQualityError(msg)
    # Resolución uniforme en nanosegundos: pandas >=2.2 admite datetime64 en `s`,
    # `ms`, `us` y `ns`, y `merge_asof` rechaza claves de resolución distinta.
    return idx.as_unit("ns")


def _one_to_naive_utc(ts: datetime, *, assume_utc_if_naive: bool = True) -> datetime:
    if ts.tzinfo is None:
        if not assume_utc_if_naive:
            msg = (
                f"announced_at={ts!r} no lleva tzinfo; sin zona horaria no se puede "
                "determinar la sesión de Nueva York de forma verificable"
            )
            raise DataQualityError(msg)
        return ts
    return ts.astimezone(UTC).replace(tzinfo=None)


# ---------------------------------------------------------------------------
# BMO / AMC / DMH
# ---------------------------------------------------------------------------


def classify_session(
    announced_at: datetime,
    cal: TradingCalendar | None = None,
    *,
    assume_utc_if_naive: bool = True,
) -> Session:
    """Deduce BMO/AMC/DMH a partir del instante del anuncio.

    El corte se evalúa **en hora de Nueva York** y contra el horario real de esa
    sesión concreta: en una media sesión el cierre son las 13:00, de modo que un
    anuncio a las 13:30 de Nochebuena es AMC y no DMH (§9.4 de
    `docs/research/pit_and_biases.md`). Simétricamente, el 2002-09-11 el mercado
    abrió a mediodía y un anuncio a las 11:00 sigue siendo BMO.

    Convenciones de frontera:
      - `t < apertura`  -> BMO
      - `apertura <= t < cierre` -> DMH
      - `t >= cierre` -> AMC (a las 16:00:00 en punto la subasta de cierre ya se
        ha ejecutado; clasificarlo como DMH permitiría "operar" un precio que ya
        no existe).
      - Si el día natural no es sesión se devuelve BMO, porque la información se
        digiere antes de la apertura de la siguiente sesión. Es indiferente para
        `tradable_date`: en un día sin mercado, `session_on_or_after` y
        `next_session` coinciden.
    """
    cal = cal or get_calendar()
    ts_utc = _one_to_naive_utc(announced_at, assume_utc_if_naive=assume_utc_if_naive)
    local = utc_to_eastern(ts_utc)
    day = local.date()
    if not cal.is_session(day):
        return Session.BMO
    minute = local.hour * 60 + local.minute
    open_t, close_t = cal.open_time(day), cal.close_time(day)
    if minute < open_t.hour * 60 + open_t.minute:
        return Session.BMO
    if minute < close_t.hour * 60 + close_t.minute:
        return Session.DMH
    return Session.AMC


def classify_sessions(
    announced_at: pd.Series | pd.DatetimeIndex | Sequence[datetime],
    cal: TradingCalendar | None = None,
    *,
    assume_utc_if_naive: bool = True,
) -> pd.Series:
    """Versión vectorizada de `classify_session`. Devuelve una `Series` de `Session`."""
    cal = cal or get_calendar()
    ts = to_naive_utc(announced_at, assume_utc_if_naive=assume_utc_if_naive)
    if len(ts) == 0:
        return pd.Series([], dtype=object, name="session")
    local = ts + eastern_offsets_for(ts, first_year=cal.first_year, last_year=cal.last_year)
    local_dates = pd.DatetimeIndex(local).normalize()
    minutes = (
        pd.DatetimeIndex(local).hour.to_numpy(dtype="int64") * 60
        + pd.DatetimeIndex(local).minute.to_numpy(dtype="int64")
    )
    opens, closes = cal.session_bounds_minutes(local_dates)

    out = np.full(len(ts), Session.BMO, dtype=object)
    is_session = opens >= 0
    out[is_session & (minutes >= closes)] = Session.AMC
    out[is_session & (minutes >= opens) & (minutes < closes)] = Session.DMH
    index = announced_at.index if isinstance(announced_at, pd.Series) else None
    return pd.Series(out, index=index, name="session")


def tradable_date(
    announced_at: datetime,
    session: Session | str,
    cal: TradingCalendar | None = None,
    *,
    assume_utc_if_naive: bool = True,
    verify: bool = False,
) -> date:
    """Primera sesión de trading en que la información del anuncio es explotable.

    Reglas (contrato §3.2):

    ==========  =====================================================
    `session`   fecha negociable
    ==========  =====================================================
    BMO         la sesión del mismo día natural; si ese día no es
                sesión, la siguiente (`session_on_or_after`).
    DMH         igual que BMO: el anuncio ocurre con el mercado ya
                abierto, así que el resto de la sesión es operable.
    AMC         la sesión **siguiente** (`next_session`).
    UNKNOWN     se trata como AMC.
    ==========  =====================================================

    **Política para `UNKNOWN`, y por qué.** Es la decisión conservadora y está
    documentada en `types.Session.UNKNOWN`. El error no es simétrico: si el
    anuncio era en realidad BMO y lo tratamos como AMC, perdemos la primera
    sesión de reacción y el backtest subestima el alfa —un coste de
    oportunidad—. Si era AMC y lo tratamos como BMO, el backtest opera el día del
    anuncio con información que aún no existía: look-ahead directo, y todo
    resultado posterior queda invalidado. Ante la duda se pierde alfa, nunca se
    inventa.

    **La trampa de la fecha UTC.** El día natural se toma tras convertir a hora
    de Nueva York. `announced_at` se almacena en UTC (contrato de
    `types.EarningsEvent`), y un anuncio de las 20:30 ET cae en el día natural
    UTC *siguiente*: leer la fecha directamente del timestamp UTC desplaza el
    evento una sesión entera.

    Parámetros
    ----------
    verify:
        Si es True, comprueba que la etiqueta `session` es coherente con el
        timestamp y lanza `DataQualityError` si no lo es. Se deja desactivado por
        defecto porque muchos proveedores dan un timestamp de fichero (hora de
        carga) junto a una etiqueta BMO/AMC fiable, y en ese caso manda la
        etiqueta.
    """
    cal = cal or get_calendar()
    sess = session if isinstance(session, Session) else parse_session(session)
    ts_utc = _one_to_naive_utc(announced_at, assume_utc_if_naive=assume_utc_if_naive)
    day = utc_to_eastern(ts_utc).date()

    if verify and sess is not Session.UNKNOWN:
        inferred = classify_session(
            announced_at, cal, assume_utc_if_naive=assume_utc_if_naive
        )
        if inferred is not sess:
            msg = (
                f"la etiqueta de sesión {sess.value!r} contradice el timestamp "
                f"{announced_at!r} (hora de Nueva York: {utc_to_eastern(ts_utc)}, "
                f"clasificación inferida: {inferred.value!r})"
            )
            raise DataQualityError(msg)

    if sess in (Session.BMO, Session.DMH):
        return cal.session_on_or_after(day)
    return cal.next_session(day)


def tradable_dates(
    events: pd.DataFrame,
    cal: TradingCalendar | None = None,
    *,
    announced_col: str = "announced_at",
    session_col: str = "session",
    assume_utc_if_naive: bool = True,
) -> pd.Series:
    """`tradable_date` vectorizada sobre una tabla de eventos.

    Devuelve una `Series` de `datetime64[ns]` (medianoche, tz-naive) alineada con
    el índice de `events`, lista para ser el nivel `date` de un panel canónico.
    Si falta la columna de sesión, todas las filas se tratan como `UNKNOWN`, es
    decir, como AMC.
    """
    cal = cal or get_calendar()
    if announced_col not in events.columns:
        msg = f"la tabla de eventos no tiene columna {announced_col!r}"
        raise DataQualityError(msg)
    if len(events) == 0:
        return pd.Series([], dtype="datetime64[ns]", index=events.index, name="tradable_date")

    ts = to_naive_utc(events[announced_col], assume_utc_if_naive=assume_utc_if_naive)
    if pd.isna(ts).any():
        n_bad = int(pd.isna(ts).sum())
        msg = f"{n_bad} eventos sin `announced_at`; no se puede fechar su sesión negociable"
        raise DataQualityError(msg)

    local = ts + eastern_offsets_for(ts, first_year=cal.first_year, last_year=cal.last_year)
    local_dates = pd.DatetimeIndex(local).normalize()

    if session_col in events.columns:
        sessions = events[session_col].map(parse_session)
    else:
        sessions = pd.Series(Session.UNKNOWN, index=events.index)

    same_day = sessions.isin([Session.BMO, Session.DMH]).to_numpy()
    on_or_after = cal.session_on_or_after_array(local_dates).to_numpy(dtype="datetime64[ns]")
    next_sess = cal.next_session_array(local_dates).to_numpy(dtype="datetime64[ns]")
    values = np.where(same_day, on_or_after, next_sess)
    return pd.Series(values, index=events.index, name="tradable_date").astype("datetime64[ns]")


# ---------------------------------------------------------------------------
# as-of join
# ---------------------------------------------------------------------------


def _normalize_left(
    signal: pd.DataFrame | pd.Series | pd.Index,
    panel_tickers: Sequence[str],
    *,
    date_col: str,
    ticker_col: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Lleva el lado izquierdo a columnas planas `date`, `ticker` + resto."""
    if isinstance(signal, pd.DatetimeIndex) or (
        isinstance(signal, pd.Index) and not isinstance(signal, pd.MultiIndex)
    ):
        dates = pd.DatetimeIndex(pd.to_datetime(signal)).normalize().as_unit("ns")
        if len(panel_tickers) == 0:
            msg = "no hay tickers en `panel` con los que expandir el índice de fechas"
            raise InsufficientHistory(msg)
        left = pd.MultiIndex.from_product(
            [dates.unique(), pd.Index(sorted(set(panel_tickers)))],
            names=[date_col, ticker_col],
        ).to_frame(index=False)
        return left, []

    frame = signal.to_frame() if isinstance(signal, pd.Series) else signal.copy()

    if isinstance(frame.index, pd.MultiIndex):
        names = [str(n) for n in frame.index.names]
        if date_col not in names or ticker_col not in names:
            msg = (
                f"el MultiIndex de `signal` debe tener niveles {date_col!r} y "
                f"{ticker_col!r}; tiene {names}"
            )
            raise DataQualityError(msg)
        extra = list(frame.columns)
        left = frame.reset_index()
    else:
        if date_col not in frame.columns or ticker_col not in frame.columns:
            msg = (
                f"`signal` debe traer columnas {date_col!r} y {ticker_col!r}, "
                f"o un MultiIndex con esos niveles; tiene {list(frame.columns)}"
            )
            raise DataQualityError(msg)
        extra = [c for c in frame.columns if c not in (date_col, ticker_col)]
        left = frame.copy()

    left[date_col] = pd.DatetimeIndex(pd.to_datetime(left[date_col])).normalize().as_unit("ns")
    if left[date_col].isna().any():
        n_bad = int(left[date_col].isna().sum())
        msg = f"{n_bad} filas de `signal` sin fecha de decisión: no se pueden fechar"
        raise DataQualityError(msg)
    left[ticker_col] = left[ticker_col].astype(str)
    return left, extra


def _normalize_right(
    panel: pd.DataFrame,
    *,
    ticker_col: str,
    available_col: str,
    value_cols: Sequence[str] | None,
) -> tuple[pd.DataFrame, list[str]]:
    """Lleva el lado derecho a columnas planas `ticker`, `available_at` + valores."""
    right = panel.copy()
    if isinstance(right.index, pd.MultiIndex) or right.index.name in (available_col, ticker_col):
        right = right.reset_index()

    if ticker_col not in right.columns:
        msg = f"`panel` debe traer la columna {ticker_col!r} (o como nivel del índice)"
        raise DataQualityError(msg)
    if available_col not in right.columns:
        msg = (
            f"`panel` debe traer la columna {available_col!r}: sin fecha de "
            "disponibilidad pública no existe point-in-time"
        )
        raise LookAheadError(msg)

    right[available_col] = to_naive_utc(right[available_col])
    if right[available_col].isna().any():
        n_bad = int(right[available_col].isna().sum())
        msg = f"{n_bad} hechos sin `{available_col}`; no se puede fechar la señal"
        raise LookAheadError(msg)
    right[ticker_col] = right[ticker_col].astype(str)

    if value_cols is None:
        cols = [c for c in right.columns if c not in (ticker_col, available_col)]
    else:
        missing = [c for c in value_cols if c not in right.columns]
        if missing:
            msg = f"columnas de valor ausentes en `panel`: {missing}"
            raise DataQualityError(msg)
        cols = list(value_cols)
    if not cols:
        msg = "`panel` no aporta ninguna columna de valor que unir"
        raise DataQualityError(msg)
    return right[[ticker_col, available_col, *cols]], cols


def asof_join(
    signal: pd.DataFrame | pd.Series | pd.Index,
    panel: pd.DataFrame,
    lag_days: int = 0,
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    available_col: str = "available_at",
    value_cols: Sequence[str] | None = None,
    max_staleness_days: int | None = None,
    cutoff: CutoffPolicy = "date_start",
    cal: TradingCalendar | None = None,
    keep_available_at: bool = True,
    suffix: str = "_pit",
) -> pd.DataFrame:
    """Une hechos fechados por `available_at` a un calendario de decisión, sin futuro.

    Para cada par `(date, ticker)` de `signal` se toma el **último** registro de
    `panel` de ese ticker cuyo `available_at` sea `<=` el corte de esa fecha.
    Antes del primer registro el resultado es NaN: no se rellena hacia atrás bajo
    ninguna circunstancia, porque un relleno hacia atrás es literalmente conocer
    el dato antes de que existiera.

    Implementación **vectorizada**: una única llamada a `pd.merge_asof` con
    `by=ticker` y `direction="backward"`, que es lo único que impide el
    look-ahead. No hay bucle por ticker ni por fecha.

    Parámetros
    ----------
    lag_days:
        Cinturón de seguridad **adicional** sobre `available_at`, no un sustituto
        de él. Sirve para modelar la latencia de ingestión de un proveedor, no
        para "arreglar" un dataset sin fechas de disponibilidad.
    cutoff:
        Momento del día que se usa como corte; ver `CutoffPolicy`. Por defecto
        `"date_start"` (medianoche UTC de la fecha), la más conservadora.
    max_staleness_days:
        Si se fija, un hecho más antiguo que ese umbral respecto a la fecha de
        decisión se invalida (NaN) en vez de arrastrarse indefinidamente. Protege
        de tickers que dejan de reportar y cuyo último fundamental seguiría
        propagándose años.
    keep_available_at:
        Conserva en la salida la columna `available_at` del hecho seleccionado.
        Es lo que permite a `assert_no_lookahead` auditar el resultado después.

    Devuelve
    --------
    `DataFrame` con MultiIndex `(date, ticker)` ordenado, las columnas originales
    de `signal` y las columnas de valor de `panel` (sufijadas si colisionan).
    """
    if lag_days < 0:
        msg = f"lag_days debe ser >= 0; recibido {lag_days}"
        raise ValueError(msg)
    if not isinstance(panel, pd.DataFrame):
        msg = "`panel` debe ser un DataFrame con `ticker`, `available_at` y valores"
        raise DataQualityError(msg)
    if len(panel) == 0:
        msg = (
            "`panel` está vacío: no hay ningún hecho con `available_at` que unir. "
            "Un join as-of contra un panel vacío devolvería NaN silenciosos."
        )
        raise InsufficientHistory(msg)

    right, val_cols = _normalize_right(
        panel, ticker_col=ticker_col, available_col=available_col, value_cols=value_cols
    )
    left, extra_cols = _normalize_left(
        signal, right[ticker_col].unique(), date_col=date_col, ticker_col=ticker_col
    )

    if len(left) == 0:
        empty_index = pd.MultiIndex.from_arrays(
            [pd.DatetimeIndex([], name=date_col), pd.Index([], name=ticker_col, dtype=object)]
        )
        cols = [*extra_cols, *val_cols] + ([available_col] if keep_available_at else [])
        return pd.DataFrame(columns=cols, index=empty_index)

    # --- corte temporal -----------------------------------------------------
    if cutoff == "date_start":
        cut = left[date_col]
    elif cutoff == "date_end":
        cut = left[date_col] + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    elif cutoff == "session_open":
        calendar = cal or get_calendar()
        unique_dates = pd.DatetimeIndex(left[date_col].unique())
        opens = pd.Series(
            calendar.opens_utc(unique_dates).as_unit("ns").to_numpy(),
            index=unique_dates,
            name="open_utc",
        )
        cut = left[date_col].map(opens)
        if cut.isna().any():
            msg = "no se pudo resolver la apertura de sesión de alguna fecha de decisión"
            raise CalendarError(msg)
    else:  # pragma: no cover - protegido por el tipo
        msg = f"política de corte desconocida: {cutoff!r}"
        raise ValueError(msg)

    left = left.copy()
    left["__cutoff__"] = (
        pd.DatetimeIndex(cut).as_unit("ns") - pd.Timedelta(days=lag_days)
    )

    # --- colisiones de nombres ---------------------------------------------
    # Una columna homónima en ambos lados haría que `merge_asof` la sufijara con
    # `_x`/`_y` y el resultado dejaría de ser direccionable; se renombra antes.
    collisions = set(val_cols) & set(left.columns)
    if collisions:
        right = right.rename(columns={c: f"{c}{suffix}" for c in collisions})
        val_cols = [f"{c}{suffix}" if c in collisions else c for c in val_cols]
    avail_out = available_col
    if available_col in left.columns:
        avail_out = f"{available_col}{suffix}"
        right = right.rename(columns={available_col: avail_out})

    right = right.sort_values([avail_out, ticker_col], kind="mergesort")
    left_sorted = left.sort_values(["__cutoff__", ticker_col], kind="mergesort")

    merged = pd.merge_asof(
        left_sorted,
        right,
        left_on="__cutoff__",
        right_on=avail_out,
        by=ticker_col,
        direction="backward",
        allow_exact_matches=True,
    )

    # --- tope de obsolescencia --------------------------------------------
    if max_staleness_days is not None:
        age = (merged[date_col] - merged[avail_out]).dt.days
        stale = age.notna() & (age > max_staleness_days)
        if bool(stale.any()):
            for col in val_cols:
                merged.loc[stale, col] = (
                    np.nan if pd.api.types.is_numeric_dtype(merged[col]) else None
                )
            merged.loc[stale, avail_out] = pd.NaT

    keep = [date_col, ticker_col, *extra_cols, *val_cols]
    if keep_available_at:
        keep.append(avail_out)
    out = merged[keep].set_index([date_col, ticker_col]).sort_index()
    out.index.names = [date_col, ticker_col]
    return out


# ---------------------------------------------------------------------------
# Auditoría
# ---------------------------------------------------------------------------


def _cutoff_series(
    dates: pd.Series, *, cutoff: CutoffPolicy, cal: TradingCalendar | None, lag_days: int
) -> pd.Series:
    dates = pd.Series(
        pd.DatetimeIndex(dates).as_unit("ns"), index=getattr(dates, "index", None)
    )
    if cutoff == "date_start":
        base = dates
    elif cutoff == "date_end":
        base = dates + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    else:
        calendar = cal or get_calendar()
        unique_dates = pd.DatetimeIndex(dates.unique()).as_unit("ns")
        opens = pd.Series(
            calendar.opens_utc(unique_dates).as_unit("ns").to_numpy(), index=unique_dates
        )
        base = dates.map(opens)
    return base - pd.Timedelta(days=lag_days)


def _violation_report(bad: pd.DataFrame, sample: int, what: str) -> str:
    head = bad.head(sample).to_string(max_cols=8)
    return f"{what}: {len(bad)} observaciones violan el corte point-in-time.\nMuestra:\n{head}"


def assert_no_lookahead(
    signal_df: pd.DataFrame,
    events_df: pd.DataFrame,
    *,
    cal: TradingCalendar | None = None,
    date_col: str = "date",
    ticker_col: str = "ticker",
    available_col: str = "available_at",
    event_key: str = "event_id",
    lag_days: int = 0,
    cutoff: CutoffPolicy = "date_start",
    check_first_event: bool = False,
    sample: int = 5,
) -> None:
    """Audita un panel de señal y lanza `LookAheadError` si usa datos del futuro.

    Pensada para que la usen los tests de `factors`, `events`, `signals` y
    `backtest`: cualquier módulo que produzca un panel `(date, ticker)` derivado
    de eventos de resultados puede llamarla y obtener un fallo ruidoso en vez de
    un Sharpe demasiado bueno.

    Realiza hasta tres comprobaciones, aplicando las que los datos permitan:

    1. **Auto-consistencia.** Si `signal_df` trae una columna `available_at`
       (por ejemplo porque salió de `asof_join`), verifica
       `available_at <= corte(date) - lag_days` en toda fila con señal no nula.
    2. **Referencia a evento.** Si `signal_df` puede unirse a `events_df` por
       `event_id` o por `(ticker, period_end)`, verifica que la fecha negociable
       del evento referenciado no es posterior a la fecha de la señal. Aquí es
       donde se caza el error clásico: asociar a la sesión del anuncio un evento
       AMC que solo era operable al día siguiente.
    3. **Primera disponibilidad** (`check_first_event=True`). Ninguna señal no
       nula de un ticker puede existir antes de la fecha negociable de su primer
       evento. Caza los rellenos hacia atrás. Es opt-in porque un factor puramente
       de precios puede legítimamente existir antes del primer evento de la tabla.

    Si los datos no permiten ninguna comprobación, lanza `DataQualityError`: una
    auditoría que no comprueba nada y pasa en verde es peor que no auditar.
    """
    calendar = cal or get_calendar()

    sig = signal_df.copy()
    if isinstance(sig.index, pd.MultiIndex) or sig.index.name in (date_col, ticker_col):
        sig = sig.reset_index()
    if date_col not in sig.columns:
        msg = f"`signal_df` no tiene columna ni nivel {date_col!r}"
        raise DataQualityError(msg)
    sig[date_col] = pd.DatetimeIndex(pd.to_datetime(sig[date_col])).normalize()

    value_cols = [
        c
        for c in sig.columns
        if c not in (date_col, ticker_col, available_col, event_key, "period_end")
    ]
    if value_cols:
        has_value = sig[value_cols].notna().any(axis=1)
    else:
        has_value = pd.Series(True, index=sig.index)

    cut = _cutoff_series(sig[date_col], cutoff=cutoff, cal=calendar, lag_days=lag_days)
    checks_run = 0

    # --- 1. auto-consistencia ---------------------------------------------
    if available_col in sig.columns:
        checks_run += 1
        avail = to_naive_utc(sig[available_col])
        bad_mask = (
            has_value.to_numpy()
            & np.asarray(pd.notna(avail))
            & np.asarray(avail > pd.DatetimeIndex(cut).as_unit("ns"))
        )
        if bool(np.any(bad_mask)):
            bad = sig.loc[bad_mask, [date_col, ticker_col, available_col, *value_cols[:3]]]
            raise LookAheadError(
                _violation_report(bad, sample, "available_at posterior al corte de la señal")
            )

    # --- 2. referencia a evento -------------------------------------------
    if events_df is None or len(events_df) == 0:
        if checks_run == 0:
            msg = "`events_df` vacío y `signal_df` sin `available_at`: no hay nada que auditar"
            raise DataQualityError(msg)
        return

    ev = events_df.copy()
    if isinstance(ev.index, pd.MultiIndex):
        ev = ev.reset_index()
    if "tradable_date" in ev.columns:
        ev_avail = pd.DatetimeIndex(pd.to_datetime(ev["tradable_date"])).normalize()
    elif "announced_at" in ev.columns:
        ev_avail = pd.DatetimeIndex(tradable_dates(ev, calendar))
    elif available_col in ev.columns:
        ev_avail = to_naive_utc(ev[available_col])
    else:
        msg = (
            "`events_df` debe traer `tradable_date`, `announced_at` o "
            f"{available_col!r} para poder auditar"
        )
        raise DataQualityError(msg)
    ev = ev.assign(__avail__=ev_avail)

    join_key: list[str] | None = None
    if event_key in sig.columns and event_key in ev.columns:
        join_key = [event_key]
    elif (
        "period_end" in sig.columns
        and "period_end" in ev.columns
        and ticker_col in sig.columns
        and ticker_col in ev.columns
    ):
        sig["period_end"] = pd.DatetimeIndex(pd.to_datetime(sig["period_end"])).normalize()
        ev["period_end"] = pd.DatetimeIndex(pd.to_datetime(ev["period_end"])).normalize()
        join_key = [ticker_col, "period_end"]

    if join_key is not None:
        checks_run += 1
        # Si la tabla de eventos trae la misma clave más de una vez (revisiones
        # del calendario, duplicados del proveedor), se audita contra la fecha
        # negociable MÁS TARDÍA: es la lectura estricta, la que no deja pasar un
        # uso anticipado apoyado en la copia más optimista del evento.
        ref = (
            ev[[*join_key, "__avail__"]]
            .sort_values("__avail__")
            .drop_duplicates(subset=join_key, keep="last")
        )
        merged = sig.merge(ref, on=join_key, how="left", validate="many_to_one")
        cut_m = _cutoff_series(merged[date_col], cutoff=cutoff, cal=calendar, lag_days=lag_days)
        bad_mask = (
            has_value.to_numpy()
            & merged["__avail__"].notna().to_numpy()
            & (merged["__avail__"] > cut_m).to_numpy()
        )
        if bool(np.any(bad_mask)):
            cols = [c for c in (date_col, ticker_col, *join_key, "__avail__") if c in merged]
            bad = merged.loc[bad_mask, cols].rename(columns={"__avail__": "tradable_date"})
            raise LookAheadError(
                _violation_report(
                    bad, sample, "la señal referencia eventos aún no negociables en su fecha"
                )
            )

    # --- 3. primera disponibilidad ----------------------------------------
    if check_first_event and ticker_col in sig.columns and ticker_col in ev.columns:
        checks_run += 1
        first = ev.groupby(ticker_col)["__avail__"].min()
        mapped = sig[ticker_col].map(first)
        bad_mask = (
            has_value.to_numpy()
            & mapped.notna().to_numpy()
            & np.asarray(pd.DatetimeIndex(cut) < pd.DatetimeIndex(mapped))
        )
        if bool(np.any(bad_mask)):
            bad = sig.loc[bad_mask, [date_col, ticker_col, *value_cols[:3]]].assign(
                first_tradable=mapped[bad_mask].to_numpy()
            )
            raise LookAheadError(
                _violation_report(
                    bad, sample, "hay señal antes del primer evento negociable del ticker"
                )
            )

    if checks_run == 0:
        msg = (
            "no se pudo auditar nada: `signal_df` no trae `available_at` ni una clave "
            f"({event_key!r} o ({ticker_col!r}, 'period_end')) que la una a `events_df`"
        )
        raise DataQualityError(msg)
