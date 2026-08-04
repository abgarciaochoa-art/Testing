"""Módulo point-in-time: calendario de sesiones, fecha negociable y uniones as-of.

Es el cimiento del repositorio. Todo lo que produce señales pasa por aquí para
responder a una única pregunta: *¿qué se podía saber en la fecha t?*

Puntos de entrada:

- `get_calendar()` / `TradingCalendar`: calendario NYSE/NASDAQ con festivos,
  cierres extraordinarios y medias sesiones, sin dependencias externas.
- `tradable_date(announced_at, session, cal)`: primera sesión en que un anuncio
  de resultados es explotable. BMO/DMH -> misma sesión; AMC y UNKNOWN -> la
  siguiente (política conservadora documentada en `types.Session`).
- `classify_session(announced_at, cal)`: deduce BMO/AMC/DMH del timestamp,
  respetando el horario real de cada sesión (incluidas las medias sesiones).
- `asof_join(signal, panel, lag_days)`: unión estrictamente hacia atrás por
  `available_at`, vectorizada.
- `assert_no_lookahead(signal_df, events_df)`: auditoría reutilizable por los
  tests de los demás módulos.
- `first_reported` / `as_restated` / `vintage_asof` / `restatement_magnitude`:
  gestión de vintages y medición del sesgo de reexpresión.
"""

from __future__ import annotations

from earnings_alpha.pit.asof import (
    CutoffPolicy,
    asof_join,
    assert_no_lookahead,
    classify_session,
    classify_sessions,
    parse_session,
    to_naive_utc,
    tradable_date,
    tradable_dates,
)
from earnings_alpha.pit.calendar import (
    DEFAULT_FIRST_YEAR,
    DEFAULT_LAST_YEAR,
    EARLY_CLOSE,
    REGULAR_CLOSE,
    REGULAR_OPEN,
    TradingCalendar,
    easter_sunday,
    eastern_offsets_for,
    eastern_to_utc,
    eastern_utc_offset,
    get_calendar,
    good_friday,
    utc_to_eastern,
)
from earnings_alpha.pit.restatements import (
    FACT_KEY,
    RestatementStats,
    as_restated,
    facts_to_frame,
    first_reported,
    restatement_magnitude,
    revisions,
    upsert_vintage,
    vintage_asof,
)

__all__ = [  # noqa: RUF022 - orden temático, no alfabético
    # calendario
    "TradingCalendar",
    "get_calendar",
    "easter_sunday",
    "good_friday",
    "eastern_utc_offset",
    "eastern_to_utc",
    "utc_to_eastern",
    "eastern_offsets_for",
    "REGULAR_OPEN",
    "REGULAR_CLOSE",
    "EARLY_CLOSE",
    "DEFAULT_FIRST_YEAR",
    "DEFAULT_LAST_YEAR",
    # as-of
    "parse_session",
    "classify_session",
    "classify_sessions",
    "tradable_date",
    "tradable_dates",
    "asof_join",
    "assert_no_lookahead",
    "to_naive_utc",
    "CutoffPolicy",
    # vintages
    "FACT_KEY",
    "facts_to_frame",
    "first_reported",
    "as_restated",
    "vintage_asof",
    "revisions",
    "restatement_magnitude",
    "upsert_vintage",
    "RestatementStats",
]
