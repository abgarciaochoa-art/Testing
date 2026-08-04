"""Expansión de eventos de resultados a tiempo-evento sobre el calendario de sesiones.

Este módulo implementa la primera mitad del contrato §3.5 de `docs/ARCHITECTURE.md`:
`event_windows`, que convierte una tabla de anuncios en un panel largo indexado por
``tau`` (días de **sesión** relativos a la fecha negociable del evento), y las
utilidades de saneamiento que comparte con `earnings_alpha.events.eventstudy`.

Decisiones de diseño y sus porqués:

- **`tau` cuenta sesiones, no días naturales.** Un fin de semana o un festivo no son
  "días" de un estudio de eventos; contarlos desplazaría los perfiles AAR/CAAR y
  rompería la comparabilidad entre eventos. Toda la aritmética se hace sobre el
  índice de sesiones de `pit.TradingCalendar`.
- **`tau = 0` es la fecha negociable (`event_date`), no la fecha del anuncio.** Para
  un anuncio AMC la sesión 0 es la *siguiente* al anuncio (`pit.tradable_date`);
  confundir ambas introduce un día entero de look-ahead (regla de oro nº 1 del repo).
- **El solapamiento se expone, nunca se oculta.** Con la ventana por defecto
  ``[-30, +60]`` (91 sesiones) y eventos trimestrales separados ~63 sesiones, las
  ventanas consecutivas del mismo emisor se solapan por construcción
  (`docs/research/validation_methodology.md` §4.4): el mismo día de retorno aparece
  con dos etiquetas `tau` distintas. `event_windows` lo marca fila a fila
  (`n_shared`) y `window_overlap_summary` lo cuantifica evento a evento, para que el
  consumidor decida truncar, purgar o corregir errores estándar — pero nunca pueda
  ignorarlo sin saberlo.

Referencias
-----------
- MacKinlay, A. C. (1997). "Event Studies in Economics and Finance". *Journal of
  Economic Literature* 35(1), 13-39. (Estructura estándar de ventana de evento.)
- Kolari, J., Pynnönen, S. (2010). "Event Study Testing with Cross-sectional
  Correlation of Abnormal Returns". *RFS* 23(11). (Por qué el solapamiento
  transversal debe quedar visible; ver `validation_methodology.md` §4.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from earnings_alpha.errors import CalendarError, DataQualityError
from earnings_alpha.pit import TradingCalendar, get_calendar, tradable_dates

__all__ = [
    "normalize_events",
    "event_windows",
    "window_overlap_summary",
]


# --------------------------------------------------------------------------- #
# Saneamiento de la tabla de eventos                                          #
# --------------------------------------------------------------------------- #


def _build_event_ids(events: pd.DataFrame) -> pd.Series:
    """Construye identificadores estables de evento.

    Prioridad:

    1. Columna `event_id` ya presente (se respeta tal cual).
    2. ``ticker:fiscal_quarter:period_end`` — exactamente el formato de
       `types.EarningsEvent.event_id`, de modo que los identificadores generados
       aquí y los del resto de la plataforma coinciden byte a byte.
    3. ``ticker:event_date`` como último recurso, documentadamente más débil (dos
       anuncios del mismo emisor el mismo día colisionarían y se detectaría en la
       validación de unicidad).
    """
    if "event_id" in events.columns:
        return events["event_id"].astype(str)
    if {"fiscal_quarter", "period_end"}.issubset(events.columns):
        period = pd.to_datetime(events["period_end"]).dt.date
        return pd.Series(
            [
                f"{t}:{q}:{p.isoformat()}"
                for t, q, p in zip(
                    events["ticker"], events["fiscal_quarter"], period, strict=True
                )
            ],
            index=events.index,
            name="event_id",
        )
    dates = pd.to_datetime(events["event_date"]).dt.date
    return pd.Series(
        [f"{t}:{d.isoformat()}" for t, d in zip(events["ticker"], dates, strict=True)],
        index=events.index,
        name="event_id",
    )


def normalize_events(
    events: pd.DataFrame,
    cal: TradingCalendar | None = None,
) -> pd.DataFrame:
    """Lleva una tabla de eventos heterogénea a la forma canónica del módulo.

    Devuelve un DataFrame con columnas ``event_id`` (str, única), ``ticker`` y
    ``event_date`` (Timestamp tz-naive normalizado a medianoche, garantizado sesión
    del calendario), conservando cualquier otra columna de entrada.

    Reglas point-in-time:

    - Si `events` ya trae ``event_date`` se valida que cada fecha sea una sesión.
    - Si no la trae, se deriva de ``announced_at`` + ``session`` con
      `pit.tradable_dates` (BMO/DMH -> misma sesión; AMC/UNKNOWN -> la siguiente),
      que es la única forma legal de fechar un anuncio en este repo.

    Lanza `DataQualityError` ante tabla vacía, columnas imprescindibles ausentes,
    identificadores duplicados o fechas que no son sesión: un estudio de eventos
    sobre una tabla defectuosa debe fallar aquí, no producir CARs desplazados.
    """
    if not isinstance(events, pd.DataFrame):
        msg = f"`events` debe ser un DataFrame; recibido {type(events).__name__}"
        raise DataQualityError(msg)
    if len(events) == 0:
        msg = "la tabla de eventos está vacía: no hay nada que expandir"
        raise DataQualityError(msg)
    if "ticker" not in events.columns:
        msg = "la tabla de eventos no tiene columna 'ticker'"
        raise DataQualityError(msg)

    cal = cal or get_calendar()
    out = events.copy()

    if "event_date" not in out.columns:
        if "announced_at" not in out.columns:
            msg = (
                "la tabla de eventos necesita 'event_date' o, en su defecto, "
                "'announced_at' (+ 'session') para derivar la sesión negociable"
            )
            raise DataQualityError(msg)
        out["event_date"] = tradable_dates(out, cal)

    out["event_date"] = pd.to_datetime(out["event_date"]).dt.normalize()
    if out["event_date"].isna().any():
        n_bad = int(out["event_date"].isna().sum())
        msg = f"{n_bad} eventos con `event_date` nulo tras la normalización"
        raise DataQualityError(msg)

    out["event_id"] = _build_event_ids(out)
    dupes = out["event_id"][out["event_id"].duplicated()]
    if len(dupes) > 0:
        sample = sorted(set(dupes))[:5]
        msg = (
            f"{len(dupes)} identificadores de evento duplicados (p. ej. {sample}): "
            "cada evento debe tener un event_id único y estable"
        )
        raise DataQualityError(msg)

    # Validación vectorizada de que cada fecha es sesión del calendario.
    grid = cal.sessions(cal.first_session, cal.last_session)
    values = out["event_date"].to_numpy(dtype="datetime64[ns]")
    if values.min() < grid.values[0] or values.max() > grid.values[-1]:
        msg = (
            "hay fechas de evento fuera del rango del calendario "
            f"{cal.first_session.isoformat()}..{cal.last_session.isoformat()}"
        )
        raise CalendarError(msg)
    pos = np.searchsorted(grid.values, values)
    ok = grid.values[np.clip(pos, 0, len(grid) - 1)] == values
    if not ok.all():
        bad = out.loc[~ok, ["event_id", "event_date"]].head(5)
        msg = (
            "fechas de evento que no son sesión del calendario (¿festivo o fin de "
            f"semana sin pasar por pit.tradable_date?): {bad.to_dict('records')}"
        )
        raise DataQualityError(msg)

    first = ["event_id", "ticker", "event_date"]
    rest = [c for c in out.columns if c not in first]
    return out[[*first, *rest]].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Expansión a tiempo-evento                                                   #
# --------------------------------------------------------------------------- #


def event_windows(
    events: pd.DataFrame,
    cal: TradingCalendar | None = None,
    pre: int = 30,
    post: int = 60,
) -> pd.DataFrame:
    """Expande cada evento a tiempo-evento ``tau`` en [-pre, +post] días de sesión.

    Contrato §3.5 de `docs/ARCHITECTURE.md`. Cada evento produce exactamente
    ``pre + post + 1`` filas; ``tau = 0`` es la sesión negociable (`event_date`), de
    modo que para un anuncio AMC la ventana empieza a contarse en la sesión
    *siguiente* al anuncio y no hay look-ahead posible.

    Parámetros
    ----------
    events:
        Tabla con al menos ``ticker`` y (``event_date`` | ``announced_at`` +
        ``session``); ver `normalize_events`. `event_id` se genera si falta.
    cal:
        Calendario de sesiones; por defecto `pit.get_calendar()`.
    pre, post:
        Sesiones antes/después del evento, ambas >= 0.

    Devuelve
    --------
    DataFrame largo con columnas:

    - ``event_id``, ``ticker``, ``event_date``: identidad del evento.
    - ``tau``: entero en ``[-pre, +post]``.
    - ``date``: la sesión correspondiente a ese `tau` (Timestamp normalizado).
    - ``n_shared``: número de ventanas de evento **del mismo ticker** que contienen
      esa sesión. ``1`` significa día exclusivo; ``>= 2`` significa que ese retorno
      diario aparecerá varias veces con etiquetas `tau` distintas. Con la ventana
      por defecto y eventos trimestrales (~63 sesiones) esto ocurre siempre
      (`validation_methodology.md` §4.4): el solapamiento se **expone** aquí para
      que el consumidor lo trate (truncado, purga o corrección de errores estándar),
      nunca se oculta.

    Errores
    -------
    `CalendarError` si alguna ventana se sale del calendario (una ventana truncada
    en silencio sesgaría los CAR medios); `DataQualityError` ante entradas
    defectuosas (ver `normalize_events`).
    """
    if pre < 0 or post < 0:
        msg = f"pre y post deben ser no negativos: pre={pre}, post={post}"
        raise ValueError(msg)
    cal = cal or get_calendar()
    base = normalize_events(events, cal)

    grid = cal.sessions(cal.first_session, cal.last_session)
    grid_values = grid.values
    pos = np.searchsorted(grid_values, base["event_date"].to_numpy(dtype="datetime64[ns]"))

    lo, hi = pos - pre, pos + post
    out_of_range = (lo < 0) | (hi >= len(grid_values))
    if out_of_range.any():
        bad = base.loc[out_of_range, "event_id"].head(5).tolist()
        msg = (
            f"la ventana [-{pre}, +{post}] de {int(out_of_range.sum())} eventos se sale "
            f"del calendario {cal.first_session.isoformat()}..{cal.last_session.isoformat()} "
            f"(p. ej. {bad}); una ventana truncada sesga los CAR medios"
        )
        raise CalendarError(msg)

    taus = np.arange(-pre, post + 1, dtype=np.int64)
    width = len(taus)
    n_events = len(base)
    position_matrix = pos[:, None] + taus[None, :]

    frame = pd.DataFrame(
        {
            "event_id": np.repeat(base["event_id"].to_numpy(), width),
            "ticker": np.repeat(base["ticker"].to_numpy(), width),
            "event_date": np.repeat(base["event_date"].to_numpy(), width),
            "tau": np.tile(taus, n_events),
            "date": grid_values[position_matrix.ravel()],
        }
    )
    frame["n_shared"] = (
        frame.groupby(["ticker", "date"], sort=False)["event_id"].transform("size").astype(np.int64)
    )
    return frame


def window_overlap_summary(
    events: pd.DataFrame,
    cal: TradingCalendar | None = None,
    pre: int = 30,
    post: int = 60,
) -> pd.DataFrame:
    """Solapamiento entre ventanas de eventos consecutivos del mismo ticker.

    Dos eventos del mismo emisor separados ``d`` sesiones tienen ventanas
    ``[-pre, +post]`` que comparten ``max(0, pre + post + 1 - d)`` sesiones. Con
    eventos trimestrales (~63 sesiones) y la ventana por defecto (91 sesiones), el
    solapamiento es la norma, no la excepción (`validation_methodology.md` §4.4);
    esta función lo cuantifica para que el consumidor pueda truncar en la siguiente
    presentación o purgar en validación cruzada.

    Devuelve un DataFrame indexado por ``event_id`` con columnas ``ticker``,
    ``event_date``, ``prev_event_id`` / ``next_event_id``, ``sessions_to_prev`` /
    ``sessions_to_next`` (distancia en sesiones, `Int64` anulable) y
    ``overlap_sessions_prev`` / ``overlap_sessions_next`` (sesiones compartidas,
    0 si no hay solapamiento), más el booleano ``overlaps_any``.
    """
    if pre < 0 or post < 0:
        msg = f"pre y post deben ser no negativos: pre={pre}, post={post}"
        raise ValueError(msg)
    cal = cal or get_calendar()
    base = normalize_events(events, cal)

    grid = cal.sessions(cal.first_session, cal.last_session)
    base = base.sort_values(["ticker", "event_date"], kind="stable").reset_index(drop=True)
    base["_pos"] = np.searchsorted(
        grid.values, base["event_date"].to_numpy(dtype="datetime64[ns]")
    )

    window_len = pre + post + 1
    grouped = base.groupby("ticker", sort=False)
    to_prev = grouped["_pos"].diff()
    to_next = -grouped["_pos"].diff(-1)
    prev_id = grouped["event_id"].shift(1)
    next_id = grouped["event_id"].shift(-1)

    overlap_prev = np.maximum(window_len - to_prev.to_numpy(dtype=float), 0.0)
    overlap_next = np.maximum(window_len - to_next.to_numpy(dtype=float), 0.0)
    overlap_prev = np.nan_to_num(overlap_prev, nan=0.0)
    overlap_next = np.nan_to_num(overlap_next, nan=0.0)

    out = pd.DataFrame(
        {
            "ticker": base["ticker"].to_numpy(),
            "event_date": base["event_date"].to_numpy(),
            "prev_event_id": prev_id.to_numpy(),
            "sessions_to_prev": to_prev.astype("Int64").to_numpy(),
            "overlap_sessions_prev": overlap_prev.astype(np.int64),
            "next_event_id": next_id.to_numpy(),
            "sessions_to_next": to_next.astype("Int64").to_numpy(),
            "overlap_sessions_next": overlap_next.astype(np.int64),
        },
        index=pd.Index(base["event_id"].to_numpy(), name="event_id"),
    )
    out["overlaps_any"] = (out["overlap_sessions_prev"] > 0) | (out["overlap_sessions_next"] > 0)
    return out
