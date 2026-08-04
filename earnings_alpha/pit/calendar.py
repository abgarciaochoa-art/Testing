"""Calendario de sesiones de NYSE/NASDAQ, implementado sin dependencias externas.

Por qué no se usa ``pandas_market_calendars`` ni ``exchange_calendars``: son
dependencias opcionales que pueden no estar instaladas, y el calendario es el
cimiento de todo el módulo point-in-time. Un backtest que se cae —o peor, que
usa un calendario aproximado— porque falta un paquete no es aceptable. Aquí se
codifican las reglas del NYSE explícitamente, con sus cierres extraordinarios y
sus medias sesiones, y los tests fijan fechas conocidas.

Alcance temporal: 1990-2035. Las reglas implementadas son las **modernas**
(Washington's Birthday como tercer lunes de febrero desde 1971, Memorial Day
como último lunes de mayo desde 1971, MLK desde 1998, Juneteenth desde 2022),
por lo que el calendario no debe extrapolarse hacia atrás de 1990 sin revisar
las reglas históricas; fuera de rango se lanza `CalendarError` en vez de
devolver una respuesta silenciosamente incorrecta.

Referencias de las reglas y de los cierres excepcionales:
  - NYSE Rule 7.2 (traslado de festivos que caen en sábado/domingo).
  - NYSE, *Historical Closings* (http://www.nyse.com/pdfs/closings.pdf), recogido
    en `exchange_calendars/us_holidays.py` y en
    `pandas_market_calendars/holidays/nyse.py`, que se han usado como fuente de
    verificación cruzada de las fechas ad-hoc.
  - Meeus, J. / algoritmo "Gregoriano anónimo" para la fecha de Pascua
    (Viernes Santo = Pascua - 2 días).

El horario de mercado se maneja en hora de Nueva York con una implementación
propia del horario de verano de EE. UU. (Energy Policy Act de 2005 desde 2007;
regla 1987-2006 antes), de modo que el módulo no depende de que la base de datos
`tzdata` esté presente en el contenedor. `tests/test_pit.py` verifica la
equivalencia con `zoneinfo` cuando esta sí está disponible.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date, datetime, time, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd

from earnings_alpha.errors import CalendarError

__all__ = [  # noqa: RUF022 - orden temático, no alfabético
    "TradingCalendar",
    "get_calendar",
    "easter_sunday",
    "good_friday",
    "eastern_utc_offset",
    "utc_to_eastern",
    "eastern_to_utc",
    "eastern_offsets_for",
    "DEFAULT_FIRST_YEAR",
    "DEFAULT_LAST_YEAR",
    "REGULAR_OPEN",
    "REGULAR_CLOSE",
    "EARLY_CLOSE",
]

# ---------------------------------------------------------------------------
# Constantes de horario
# ---------------------------------------------------------------------------

REGULAR_OPEN: time = time(9, 30)
"""Apertura ordinaria del mercado de contado, hora de Nueva York."""

REGULAR_CLOSE: time = time(16, 0)
"""Cierre ordinario, hora de Nueva York (vigente desde 1974)."""

EARLY_CLOSE: time = time(13, 0)
"""Cierre de media sesión desde 1993 (antes eran las 14:00)."""

DEFAULT_FIRST_YEAR: int = 1990
DEFAULT_LAST_YEAR: int = 2035

_MONDAY, _TUESDAY, _WEDNESDAY, _THURSDAY, _FRIDAY, _SATURDAY, _SUNDAY = range(7)

# Alias de mercados que comparten exactamente el calendario del NYSE.
# NASDAQ publica el mismo cuadro de festivos y de medias sesiones que NYSE;
# las diferencias entre ambos están en la microestructura, no en el calendario.
_CALENDAR_ALIASES: Mapping[str, str] = {
    "XNYS": "XNYS",
    "NYSE": "XNYS",
    "NEW YORK STOCK EXCHANGE": "XNYS",
    "XNAS": "XNYS",
    "NASDAQ": "XNYS",
    "ARCX": "XNYS",
    "NYSE ARCA": "XNYS",
    "BATS": "XNYS",
    "US": "XNYS",
    "US_EQUITIES": "XNYS",
}


# ---------------------------------------------------------------------------
# Utilidades de calendario civil
# ---------------------------------------------------------------------------


def easter_sunday(year: int) -> date:
    """Domingo de Pascua del calendario gregoriano.

    Algoritmo "gregoriano anónimo" (Meeus/Jones/Butcher). Se necesita porque el
    Viernes Santo es el único festivo del NYSE de fecha móvil no anclada a un
    día de la semana de un mes.
    """
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month, day_minus_one = divmod(h + ell - 7 * m + 114, 31)
    return date(year, month, day_minus_one + 1)


def good_friday(year: int) -> date:
    """Viernes Santo: dos días antes del Domingo de Pascua."""
    return easter_sunday(year) - timedelta(days=2)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """N-ésimo `weekday` (0=lunes) del mes indicado, con `n` empezando en 1."""
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """Último `weekday` del mes indicado."""
    last = date(year, 12, 31) if month == 12 else date(year, month + 1, 1) - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _nearest_workday(d: date) -> date:
    """Traslado del NYSE: sábado -> viernes anterior, domingo -> lunes siguiente."""
    if d.weekday() == _SATURDAY:
        return d - timedelta(days=1)
    if d.weekday() == _SUNDAY:
        return d + timedelta(days=1)
    return d


# ---------------------------------------------------------------------------
# Horario de verano de EE. UU. (implementación propia, sin tzdata)
# ---------------------------------------------------------------------------

_EST = timedelta(hours=-5)
_EDT = timedelta(hours=-4)


def _dst_bounds_utc(year: int) -> tuple[datetime, datetime]:
    """Instantes UTC (naive) de inicio y fin del horario de verano del Este.

    Desde 2007 (Energy Policy Act de 2005): segundo domingo de marzo a las 02:00
    hora estándar local (= 07:00 UTC) hasta el primer domingo de noviembre a las
    02:00 hora de verano local (= 06:00 UTC).
    Entre 1987 y 2006: primer domingo de abril hasta el último domingo de octubre,
    con los mismos husos horarios de conmutación.
    """
    if year >= 2007:
        start = _nth_weekday(year, 3, _SUNDAY, 2)
        end = _nth_weekday(year, 11, _SUNDAY, 1)
    else:
        start = _nth_weekday(year, 4, _SUNDAY, 1)
        end = _last_weekday(year, 10, _SUNDAY)
    return (
        datetime(start.year, start.month, start.day, 7),
        datetime(end.year, end.month, end.day, 6),
    )


def eastern_utc_offset(ts_utc: datetime) -> timedelta:
    """Desfase UTC->Este vigente en el instante `ts_utc` (naive, interpretado UTC)."""
    start, end = _dst_bounds_utc(ts_utc.year)
    return _EDT if start <= ts_utc < end else _EST


def utc_to_eastern(ts_utc: datetime) -> datetime:
    """Convierte un instante UTC (naive) a hora de pared de Nueva York (naive)."""
    return ts_utc + eastern_utc_offset(ts_utc)


def eastern_to_utc(local_naive: datetime) -> datetime:
    """Convierte hora de pared de Nueva York (naive) a UTC (naive).

    Estrategia de dos pasos porque el desfase depende del propio instante: se
    supone hora estándar, se corrige y se reevalúa. Las horas de apertura y
    cierre del mercado nunca caen en la ventana ambigua (02:00-03:00 del domingo
    del cambio), de modo que la iteración converge siempre.
    """
    guess_utc = local_naive - _EST
    offset = eastern_utc_offset(guess_utc)
    utc = local_naive - offset
    if eastern_utc_offset(utc) != offset:
        utc = local_naive - eastern_utc_offset(utc)
    return utc


@lru_cache(maxsize=8)
def _dst_transitions(first_year: int, last_year: int) -> np.ndarray:
    """Instantes UTC de conmutación DST, alternando inicio/fin, en orden ascendente.

    Se usa para vectorizar la conversión UTC->Este: `searchsorted` sobre este
    vector devuelve una posición impar exactamente cuando el instante está en
    horario de verano (la última conmutación superada fue un inicio).
    """
    stamps: list[np.datetime64] = []
    for year in range(first_year - 1, last_year + 2):
        start, end = _dst_bounds_utc(year)
        stamps.append(np.datetime64(start, "ns"))
        stamps.append(np.datetime64(end, "ns"))
    return np.array(stamps, dtype="datetime64[ns]")


def eastern_offsets_for(
    ts_utc: pd.DatetimeIndex | pd.Series,
    *,
    first_year: int = DEFAULT_FIRST_YEAR,
    last_year: int = DEFAULT_LAST_YEAR,
) -> pd.TimedeltaIndex:
    """Versión vectorizada de `eastern_utc_offset` para un vector de instantes UTC.

    `ts_utc` debe ser tz-naive y estar expresado en UTC (convención del repo para
    `EarningsEvent.announced_at`).
    """
    idx = pd.DatetimeIndex(pd.Series(ts_utc).to_numpy(dtype="datetime64[ns]"))
    transitions = _dst_transitions(first_year, last_year)
    pos = np.searchsorted(transitions, idx.to_numpy(dtype="datetime64[ns]"), side="right")
    is_dst = (pos % 2) == 1
    hours = np.where(is_dst, -4, -5).astype("int64")
    return pd.TimedeltaIndex(pd.to_timedelta(hours, unit="h"))


# ---------------------------------------------------------------------------
# Tablas de festivos y cierres extraordinarios
# ---------------------------------------------------------------------------

# Cierres completos no recurrentes. Fuente: NYSE Historical Closings.
_ADHOC_CLOSURES: dict[date, str] = {
    date(1994, 4, 27): "Duelo nacional: funeral de Richard Nixon",
    date(2001, 9, 11): "Atentados del 11-S",
    date(2001, 9, 12): "Atentados del 11-S",
    date(2001, 9, 13): "Atentados del 11-S",
    date(2001, 9, 14): "Atentados del 11-S",
    date(2004, 6, 11): "Duelo nacional: funeral de Ronald Reagan",
    date(2007, 1, 2): "Duelo nacional: funeral de Gerald Ford",
    date(2012, 10, 29): "Huracán Sandy",
    date(2012, 10, 30): "Huracán Sandy",
    date(2018, 12, 5): "Duelo nacional: funeral de George H. W. Bush",
    date(2025, 1, 9): "Duelo nacional: funeral de Jimmy Carter",
}

# Cierres tempranos irregulares (fecha -> hora de cierre en Nueva York).
# Importan para clasificar BMO/AMC: un anuncio a las 14:00 del 1997-10-27 es AMC.
_ADHOC_EARLY_CLOSES: dict[date, time] = {
    date(1994, 2, 11): time(14, 30),  # temporal de nieve
    date(1996, 1, 8): time(14, 0),  # ventisca del 96
    date(1997, 10, 27): time(15, 30),  # cortacircuitos: caída del mercado
    date(1997, 12, 26): time(13, 0),
    date(1999, 12, 31): time(13, 0),  # preparativos del efecto 2000
    date(2003, 12, 26): time(13, 0),
    date(2005, 6, 1): time(15, 56),  # avería en los sistemas de comunicación
}

# Aperturas tardías. Afectan a la frontera BMO/DMH: el 2002-09-11 el mercado no
# abrió hasta el mediodía, de modo que un anuncio a las 11:00 sigue siendo BMO.
_ADHOC_LATE_OPENS: dict[date, time] = {
    date(1990, 12, 27): time(9, 31),
    date(1991, 1, 17): time(9, 31),
    date(1991, 2, 25): time(9, 31),
    date(1995, 12, 18): time(10, 30),
    date(1996, 1, 8): time(11, 0),
    date(2001, 9, 17): time(9, 33),
    date(2001, 10, 8): time(9, 31),
    date(2002, 9, 11): time(12, 0),
    date(2003, 3, 20): time(9, 32),
    date(2004, 6, 7): time(9, 32),
    date(2006, 12, 27): time(9, 32),
}


def _regular_holidays(year: int) -> dict[date, str]:
    """Festivos recurrentes del NYSE para un año dado, ya trasladados."""
    out: dict[date, str] = {}

    # Año Nuevo. Regla asimétrica: si cae en domingo se observa el lunes; si cae
    # en sábado NO se observa (el viernes anterior es 31 de diciembre, cierre del
    # ejercicio contable, y el NYSE abre: 1999-12-31, 2004-12-31, 2010-12-31).
    new_year = date(year, 1, 1)
    if new_year.weekday() == _SUNDAY:
        out[date(year, 1, 2)] = "Año Nuevo (observado)"
    elif new_year.weekday() != _SATURDAY:
        out[new_year] = "Año Nuevo"

    # Martin Luther King Jr.: tercer lunes de enero, observado por el NYSE desde 1998.
    if year >= 1998:
        out[_nth_weekday(year, 1, _MONDAY, 3)] = "Martin Luther King Jr."

    # Washington's Birthday / Presidents' Day: tercer lunes de febrero desde 1971.
    out[_nth_weekday(year, 2, _MONDAY, 3)] = "Washington's Birthday"

    # Viernes Santo.
    out[good_friday(year)] = "Viernes Santo"

    # Memorial Day: último lunes de mayo desde 1971.
    out[_last_weekday(year, 5, _MONDAY)] = "Memorial Day"

    # Juneteenth: 19 de junio, observado por el NYSE desde 2022.
    # En 2021 el 19 de junio cayó en sábado y el NYSE abrió el viernes 18: la ley
    # federal se firmó el 17 de junio de 2021 y el NYSE no lo observó ese año.
    if year >= 2022:
        out[_nearest_workday(date(year, 6, 19))] = "Juneteenth"

    # Día de la Independencia.
    out[_nearest_workday(date(year, 7, 4))] = "Día de la Independencia"

    # Labor Day: primer lunes de septiembre.
    out[_nth_weekday(year, 9, _MONDAY, 1)] = "Labor Day"

    # Acción de Gracias: cuarto jueves de noviembre.
    out[_nth_weekday(year, 11, _THURSDAY, 4)] = "Acción de Gracias"

    # Navidad.
    out[_nearest_workday(date(year, 12, 25))] = "Navidad"

    return out


def _regular_early_closes(year: int) -> dict[date, time]:
    """Medias sesiones recurrentes del NYSE para un año dado.

    Reglas vigentes (verificadas contra las tablas de `exchange_calendars` y
    `pandas_market_calendars`):

    - 3 de julio, si cae en lunes, martes o jueves: cierre a las 13:00 (desde 1995).
    - 3 de julio en miércoles (es decir, 4 de julio en jueves): media sesión
      **desde 2013**; antes de 2013 la media sesión era el viernes 5 de julio.
    - Viernes siguiente a Acción de Gracias: 13:00 desde 1993 (14:00 en 1992).
    - Nochebuena, si cae de lunes a jueves: 13:00 desde 1993 (14:00 en 1990-1992).
      Si el 25 cae en sábado, el 24 es festivo completo, no media sesión.
    """
    out: dict[date, time] = {}

    if year >= 1995:
        july3 = date(year, 7, 3)
        if july3.weekday() in (_MONDAY, _TUESDAY, _THURSDAY) or (july3.weekday() == _WEDNESDAY and year >= 2013):
            out[july3] = EARLY_CLOSE
        july5 = date(year, 7, 5)
        if year <= 2012 and july5.weekday() == _FRIDAY:
            out[july5] = EARLY_CLOSE

    if year >= 1992:
        black_friday = _nth_weekday(year, 11, _THURSDAY, 4) + timedelta(days=1)
        out[black_friday] = EARLY_CLOSE if year >= 1993 else time(14, 0)

    christmas_eve = date(year, 12, 24)
    if christmas_eve.weekday() in (_MONDAY, _TUESDAY, _WEDNESDAY, _THURSDAY):
        out[christmas_eve] = EARLY_CLOSE if year >= 1993 else time(14, 0)

    return out


# ---------------------------------------------------------------------------
# Calendario
# ---------------------------------------------------------------------------


def _as_date(value: date | datetime | str | pd.Timestamp | np.datetime64) -> date:
    """Coerción tolerante a `datetime.date`, sin perder el aviso si el tipo es raro."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.date()


class TradingCalendar:
    """Calendario de sesiones bursátiles de EE. UU. (NYSE y NASDAQ).

    Precalcula el vector completo de sesiones del rango soportado y resuelve las
    consultas por búsqueda binaria, de modo que `next_session`/`shift` son O(log n)
    y admiten uso intensivo dentro de bucles de backtest.

    Toda fecha devuelta es un `datetime.date`; los índices devueltos son
    `pd.DatetimeIndex` tz-naive normalizados a medianoche, que es la convención de
    paneles del repo (`docs/ARCHITECTURE.md`, §1).
    """

    def __init__(
        self,
        name: str = "XNYS",
        *,
        first_year: int = DEFAULT_FIRST_YEAR,
        last_year: int = DEFAULT_LAST_YEAR,
    ) -> None:
        canonical = _CALENDAR_ALIASES.get(name.strip().upper())
        if canonical is None:
            msg = (
                f"calendario desconocido: {name!r}; "
                f"soportados: {sorted(set(_CALENDAR_ALIASES))}"
            )
            raise CalendarError(msg)
        if first_year < 1990 or last_year > 2100 or first_year > last_year:
            msg = (
                f"rango de años no soportado: {first_year}-{last_year}; "
                "las reglas implementadas son las del NYSE moderno (>=1990)"
            )
            raise CalendarError(msg)

        self.name = canonical
        self.requested_name = name
        self.first_year = first_year
        self.last_year = last_year

        holidays: dict[date, str] = {}
        early: dict[date, time] = {}
        for year in range(first_year, last_year + 1):
            holidays.update(_regular_holidays(year))
            early.update(_regular_early_closes(year))
        for day, reason in _ADHOC_CLOSURES.items():
            if first_year <= day.year <= last_year:
                holidays[day] = reason
        for day, hour in _ADHOC_EARLY_CLOSES.items():
            if first_year <= day.year <= last_year:
                early[day] = hour

        all_days = pd.date_range(
            start=pd.Timestamp(year=first_year, month=1, day=1),
            end=pd.Timestamp(year=last_year, month=12, day=31),
            freq="D",
        )
        weekdays = all_days[all_days.dayofweek < 5]
        holiday_index = pd.DatetimeIndex(sorted(pd.Timestamp(d) for d in holidays))
        sessions = weekdays.difference(holiday_index)
        sessions.name = "date"

        self._sessions: pd.DatetimeIndex = sessions
        self._sessions_np: np.ndarray = sessions.to_numpy(dtype="datetime64[ns]")
        self._holidays: dict[date, str] = holidays

        # Solo son medias sesiones las que caen en día hábil de mercado.
        session_dates = set(sessions.date)
        self._early_closes: dict[date, time] = {
            d: t for d, t in early.items() if d in session_dates
        }
        self._late_opens: dict[date, time] = {
            d: t
            for d, t in _ADHOC_LATE_OPENS.items()
            if d in session_dates and first_year <= d.year <= last_year
        }

        # Vectores paralelos a `_sessions` con los minutos de apertura y cierre,
        # para clasificar BMO/AMC de forma vectorizada.
        open_minutes = np.full(len(sessions), REGULAR_OPEN.hour * 60 + REGULAR_OPEN.minute, "int64")
        close_minutes = np.full(
            len(sessions), REGULAR_CLOSE.hour * 60 + REGULAR_CLOSE.minute, "int64"
        )
        for d, t in self._late_opens.items():
            open_minutes[self._index_of(d)] = t.hour * 60 + t.minute
        for d, t in self._early_closes.items():
            close_minutes[self._index_of(d)] = t.hour * 60 + t.minute
        self._open_minutes = open_minutes
        self._close_minutes = close_minutes

    # -- representación ----------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover - cosmético
        return (
            f"TradingCalendar({self.name!r}, {self.first_year}-{self.last_year}, "
            f"{len(self._sessions)} sesiones)"
        )

    def __len__(self) -> int:
        return len(self._sessions)

    # -- rango -------------------------------------------------------------

    @property
    def first_session(self) -> date:
        """Primera sesión soportada."""
        return self._sessions[0].date()

    @property
    def last_session(self) -> date:
        """Última sesión soportada."""
        return self._sessions[-1].date()

    def _check_range(self, d: date, *, what: str = "fecha") -> None:
        if d.year < self.first_year or d.year > self.last_year:
            msg = (
                f"{what} {d.isoformat()} fuera del rango del calendario "
                f"{self.first_year}-{self.last_year}"
            )
            raise CalendarError(msg)

    def _index_of(self, d: date) -> int:
        pos = int(np.searchsorted(self._sessions_np, np.datetime64(d, "ns"), side="left"))
        if pos >= len(self._sessions_np) or self._sessions_np[pos] != np.datetime64(d, "ns"):
            msg = f"{d.isoformat()} no es una sesión de {self.name}"
            raise CalendarError(msg)
        return pos

    # -- consultas básicas -------------------------------------------------

    def is_session(self, d: date | datetime | str | pd.Timestamp) -> bool:
        """True si `d` es una sesión bursátil completa o media."""
        day = _as_date(d)
        self._check_range(day)
        target = np.datetime64(day, "ns")
        pos = int(np.searchsorted(self._sessions_np, target, side="left"))
        return pos < len(self._sessions_np) and self._sessions_np[pos] == target

    def is_holiday(self, d: date | datetime | str | pd.Timestamp) -> bool:
        """True si `d` es festivo del NYSE (excluye fines de semana)."""
        return _as_date(d) in self._holidays

    def holiday_name(self, d: date | datetime | str | pd.Timestamp) -> str | None:
        """Motivo del cierre, o None si `d` no es festivo."""
        return self._holidays.get(_as_date(d))

    def sessions(
        self,
        start: date | datetime | str | pd.Timestamp,
        end: date | datetime | str | pd.Timestamp,
    ) -> pd.DatetimeIndex:
        """Sesiones en `[start, end]`, ambos inclusive.

        Devuelve un `pd.DatetimeIndex` tz-naive normalizado a medianoche y con
        nombre ``"date"``, listo para ser el nivel de fecha de un panel canónico.
        """
        s, e = _as_date(start), _as_date(end)
        self._check_range(s, what="fecha inicial")
        self._check_range(e, what="fecha final")
        if s > e:
            msg = f"rango invertido: {s.isoformat()} > {e.isoformat()}"
            raise CalendarError(msg)
        lo = int(np.searchsorted(self._sessions_np, np.datetime64(s, "ns"), side="left"))
        hi = int(np.searchsorted(self._sessions_np, np.datetime64(e, "ns"), side="right"))
        out = self._sessions[lo:hi]
        out.name = "date"
        return out

    def next_session(self, d: date | datetime | str | pd.Timestamp) -> date:
        """Primera sesión **estrictamente posterior** a `d`."""
        day = _as_date(d)
        self._check_range(day)
        pos = int(np.searchsorted(self._sessions_np, np.datetime64(day, "ns"), side="right"))
        if pos >= len(self._sessions_np):
            msg = f"no hay sesión posterior a {day.isoformat()} dentro del calendario"
            raise CalendarError(msg)
        return self._sessions[pos].date()

    def prev_session(self, d: date | datetime | str | pd.Timestamp) -> date:
        """Última sesión **estrictamente anterior** a `d`."""
        day = _as_date(d)
        self._check_range(day)
        pos = int(np.searchsorted(self._sessions_np, np.datetime64(day, "ns"), side="left")) - 1
        if pos < 0:
            msg = f"no hay sesión anterior a {day.isoformat()} dentro del calendario"
            raise CalendarError(msg)
        return self._sessions[pos].date()

    def session_on_or_after(self, d: date | datetime | str | pd.Timestamp) -> date:
        """`d` si es sesión; si no, la siguiente. Es la regla de un anuncio BMO."""
        day = _as_date(d)
        self._check_range(day)
        pos = int(np.searchsorted(self._sessions_np, np.datetime64(day, "ns"), side="left"))
        if pos >= len(self._sessions_np):
            msg = f"no hay sesión en o después de {day.isoformat()} dentro del calendario"
            raise CalendarError(msg)
        return self._sessions[pos].date()

    def session_on_or_before(self, d: date | datetime | str | pd.Timestamp) -> date:
        """`d` si es sesión; si no, la anterior."""
        day = _as_date(d)
        self._check_range(day)
        pos = int(np.searchsorted(self._sessions_np, np.datetime64(day, "ns"), side="right")) - 1
        if pos < 0:
            msg = f"no hay sesión en o antes de {day.isoformat()} dentro del calendario"
            raise CalendarError(msg)
        return self._sessions[pos].date()

    def shift(self, d: date | datetime | str | pd.Timestamp, n: int) -> date:
        """Desplaza `n` sesiones (negativo hacia atrás).

        Semántica cuando `d` no es sesión: para `n > 0` el ancla es la última
        sesión anterior a `d`, y para `n < 0` la primera posterior, de modo que
        `shift(d, 1) == next_session(d)` y `shift(d, -1) == prev_session(d)`
        **para cualquier** `d`. Con `n == 0` sobre un día no bursátil se lanza
        `CalendarError`: no existe "la sesión de hoy" un domingo, y devolver una
        aproximación silenciosa es exactamente el tipo de atajo que introduce
        errores de un día.
        """
        day = _as_date(d)
        self._check_range(day)
        target = np.datetime64(day, "ns")
        left = int(np.searchsorted(self._sessions_np, target, side="left"))
        is_sess = left < len(self._sessions_np) and self._sessions_np[left] == target

        if is_sess:
            idx = left + n
        elif n > 0:
            idx = left + n - 1
        elif n < 0:
            idx = left + n
        else:
            msg = (
                f"shift(..., 0) sobre {day.isoformat()}, que no es sesión de {self.name}; "
                "usa session_on_or_after/session_on_or_before para desambiguar"
            )
            raise CalendarError(msg)

        if idx < 0 or idx >= len(self._sessions_np):
            msg = (
                f"desplazamiento de {n} sesiones desde {day.isoformat()} "
                f"cae fuera del calendario {self.first_year}-{self.last_year}"
            )
            raise CalendarError(msg)
        return self._sessions[idx].date()

    def session_distance(
        self,
        start: date | datetime | str | pd.Timestamp,
        end: date | datetime | str | pd.Timestamp,
    ) -> int:
        """Número de sesiones de `start` a `end` en tiempo-evento.

        Devuelve `indice(end) - indice(start)`, es decir 0 si son la misma sesión
        y negativo si `end` precede a `start`. Ambas deben ser sesiones.
        """
        return self._index_of(_as_date(end)) - self._index_of(_as_date(start))

    def sessions_around(
        self, d: date | datetime | str | pd.Timestamp, pre: int, post: int
    ) -> pd.DatetimeIndex:
        """Ventana de sesiones `[d - pre, d + post]` en tiempo-evento.

        `d` debe ser sesión (típicamente el `tradable_date` de un evento). Si la
        ventana se sale del calendario se lanza `CalendarError` en vez de
        devolverla truncada: una ventana truncada sesga los CAR medios.
        """
        if pre < 0 or post < 0:
            msg = f"pre y post deben ser no negativos: pre={pre}, post={post}"
            raise CalendarError(msg)
        idx = self._index_of(_as_date(d))
        lo, hi = idx - pre, idx + post
        if lo < 0 or hi >= len(self._sessions_np):
            msg = (
                f"ventana [-{pre}, +{post}] alrededor de {_as_date(d).isoformat()} "
                f"se sale del calendario {self.first_year}-{self.last_year}"
            )
            raise CalendarError(msg)
        out = self._sessions[lo : hi + 1]
        out.name = "date"
        return out

    # -- horarios ----------------------------------------------------------

    def open_time(self, d: date | datetime | str | pd.Timestamp) -> time:
        """Hora de apertura de la sesión `d` en horario de Nueva York."""
        day = _as_date(d)
        self._index_of(day)
        return self._late_opens.get(day, REGULAR_OPEN)

    def close_time(self, d: date | datetime | str | pd.Timestamp) -> time:
        """Hora de cierre de la sesión `d` en horario de Nueva York.

        Es la pieza que hace correcta la clasificación AMC: en una media sesión el
        corte son las 13:00, no las 16:00, y un anuncio a las 13:30 de Nochebuena
        es AMC (negociable la sesión siguiente), no DMH.
        """
        day = _as_date(d)
        self._index_of(day)
        return self._early_closes.get(day, REGULAR_CLOSE)

    def is_early_close(self, d: date | datetime | str | pd.Timestamp) -> bool:
        """True si `d` es media sesión (cierre anticipado)."""
        return _as_date(d) in self._early_closes

    def is_late_open(self, d: date | datetime | str | pd.Timestamp) -> bool:
        """True si `d` tuvo apertura retrasada."""
        return _as_date(d) in self._late_opens

    def early_closes(
        self,
        start: date | datetime | str | pd.Timestamp,
        end: date | datetime | str | pd.Timestamp,
    ) -> pd.Series:
        """Serie `fecha -> hora de cierre` de las medias sesiones del rango."""
        s, e = _as_date(start), _as_date(end)
        items = sorted((d, t) for d, t in self._early_closes.items() if s <= d <= e)
        idx = pd.DatetimeIndex([pd.Timestamp(d) for d, _ in items], name="date")
        return pd.Series([t for _, t in items], index=idx, dtype=object, name="close_time")

    def open_utc(self, d: date | datetime | str | pd.Timestamp) -> datetime:
        """Instante UTC (naive) de apertura de la sesión `d`."""
        day = _as_date(d)
        return eastern_to_utc(datetime.combine(day, self.open_time(day)))

    def close_utc(self, d: date | datetime | str | pd.Timestamp) -> datetime:
        """Instante UTC (naive) de cierre de la sesión `d`."""
        day = _as_date(d)
        return eastern_to_utc(datetime.combine(day, self.close_time(day)))

    def opens_utc(self, dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """Versión vectorizada de `open_utc` sobre un índice de sesiones."""
        return pd.DatetimeIndex(
            [pd.Timestamp(self.open_utc(ts.date())) for ts in dates], name="open_utc"
        ).as_unit("ns")

    # -- helpers vectorizados ---------------------------------------------

    def _positions_on_or_after(self, days: np.ndarray) -> np.ndarray:
        return np.searchsorted(self._sessions_np, days, side="left")

    def _positions_after(self, days: np.ndarray) -> np.ndarray:
        return np.searchsorted(self._sessions_np, days, side="right")

    def session_on_or_after_array(self, days: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """`session_on_or_after` aplicada a un vector de fechas normalizadas."""
        arr = pd.DatetimeIndex(days).normalize().to_numpy(dtype="datetime64[ns]")
        pos = self._positions_on_or_after(arr)
        self._raise_if_out_of_bounds(pos, days)
        return pd.DatetimeIndex(self._sessions_np[pos], name="date")

    def next_session_array(self, days: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """`next_session` aplicada a un vector de fechas normalizadas."""
        arr = pd.DatetimeIndex(days).normalize().to_numpy(dtype="datetime64[ns]")
        pos = self._positions_after(arr)
        self._raise_if_out_of_bounds(pos, days)
        return pd.DatetimeIndex(self._sessions_np[pos], name="date")

    def _raise_if_out_of_bounds(self, pos: np.ndarray, days: pd.DatetimeIndex) -> None:
        bad = pos >= len(self._sessions_np)
        if bool(bad.any()):
            first_bad = pd.DatetimeIndex(days)[bad][0]
            msg = (
                f"no hay sesión disponible para {first_bad.date().isoformat()} "
                f"dentro del calendario {self.first_year}-{self.last_year}"
            )
            raise CalendarError(msg)

    def session_bounds_minutes(self, days: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
        """Minutos de apertura y cierre (hora de NY) de cada fecha del vector.

        Para fechas que no son sesión devuelve `(-1, -1)`, lo que permite al
        clasificador distinguir "día sin mercado" sin lanzar.
        """
        arr = pd.DatetimeIndex(days).normalize().to_numpy(dtype="datetime64[ns]")
        pos = np.searchsorted(self._sessions_np, arr, side="left")
        clipped = np.clip(pos, 0, len(self._sessions_np) - 1)
        exact = (pos < len(self._sessions_np)) & (self._sessions_np[clipped] == arr)
        opens = np.where(exact, self._open_minutes[clipped], -1)
        closes = np.where(exact, self._close_minutes[clipped], -1)
        return opens, closes

    # -- diagnóstico -------------------------------------------------------

    def session_counts_by_year(self) -> pd.Series:
        """Sesiones por año natural. Un año bursátil típico tiene 250-253."""
        counts = pd.Series(1, index=self._sessions).groupby(self._sessions.year).sum()
        counts.index.name = "year"
        counts.name = "n_sessions"
        return counts

    def holidays(
        self,
        start: date | datetime | str | pd.Timestamp,
        end: date | datetime | str | pd.Timestamp,
    ) -> pd.Series:
        """Serie `fecha -> motivo` de los festivos del rango."""
        s, e = _as_date(start), _as_date(end)
        items = sorted((d, name) for d, name in self._holidays.items() if s <= d <= e)
        idx = pd.DatetimeIndex([pd.Timestamp(d) for d, _ in items], name="date")
        return pd.Series([n for _, n in items], index=idx, dtype=object, name="holiday")


@lru_cache(maxsize=8)
def _build_calendar(canonical: str, first_year: int, last_year: int) -> TradingCalendar:
    return TradingCalendar(canonical, first_year=first_year, last_year=last_year)


def get_calendar(
    name: str = "XNYS",
    *,
    first_year: int = DEFAULT_FIRST_YEAR,
    last_year: int = DEFAULT_LAST_YEAR,
) -> TradingCalendar:
    """Devuelve un `TradingCalendar` cacheado.

    Construirlo cuesta unos milisegundos y es inmutable, así que se comparte
    entre módulos. El nombre se canonicaliza antes de consultar la caché, de modo
    que `"NYSE"`, `"NASDAQ"`, `"XNYS"` y `"XNAS"` devuelven **el mismo objeto**:
    comparten calendario de festivos y de medias sesiones.
    """
    canonical = _CALENDAR_ALIASES.get(name.strip().upper())
    if canonical is None:
        msg = f"calendario desconocido: {name!r}; soportados: {sorted(set(_CALENDAR_ALIASES))}"
        raise CalendarError(msg)
    return _build_calendar(canonical, first_year, last_year)


def _iter_dates(values: Iterable[object]) -> list[date]:  # pragma: no cover - utilidad
    return [_as_date(v) for v in values]  # type: ignore[arg-type]
