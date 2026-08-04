"""Tests del módulo point-in-time.

Todos corren **sin red**. La estrategia es triple:

1. **Fechas conocidas.** Se fijan festivos, medias sesiones y cierres
   extraordinarios del NYSE que se pueden verificar contra fuentes públicas.
   Si alguien "mejora" el calendario y rompe una de estas fechas, el test lo dice.
2. **Invariantes.** Propiedades que deben cumplirse siempre (`shift(d,1) ==
   next_session(d)`, monotonía de las sesiones, conteos anuales plausibles).
3. **Fuerza bruta contra la implementación vectorizada.** `asof_join` se compara
   con un cálculo elemental por bucle sobre datos aleatorios, y se comprueba
   exhaustivamente que ningún valor devuelto tiene `available_at` posterior al
   corte. Es la garantía de que no hay futuro.
"""

from __future__ import annotations

import datetime as dt
import random

import numpy as np
import pandas as pd
import pytest

from earnings_alpha.errors import (
    CalendarError,
    DataQualityError,
    InsufficientHistory,
    LookAheadError,
)
from earnings_alpha.pit import (
    TradingCalendar,
    as_restated,
    asof_join,
    assert_no_lookahead,
    classify_session,
    classify_sessions,
    eastern_offsets_for,
    eastern_to_utc,
    eastern_utc_offset,
    first_reported,
    get_calendar,
    good_friday,
    parse_session,
    restatement_magnitude,
    revisions,
    tradable_date,
    tradable_dates,
    upsert_vintage,
    utc_to_eastern,
    vintage_asof,
)
from earnings_alpha.types import FundamentalFact, Session

UTC = dt.UTC


@pytest.fixture(scope="module")
def cal() -> TradingCalendar:
    return get_calendar("XNYS")


def et(year: int, month: int, day: int, hour: int, minute: int = 0) -> dt.datetime:
    """Instante aware en UTC a partir de una hora de pared de Nueva York."""
    return eastern_to_utc(dt.datetime(year, month, day, hour, minute)).replace(tzinfo=UTC)


# ===========================================================================
# 1. Calendario: festivos y fechas conocidas
# ===========================================================================


@pytest.mark.parametrize(
    ("day", "why"),
    [
        ("2024-11-28", "Acción de Gracias 2024 (cuarto jueves de noviembre)"),
        ("2024-03-29", "Viernes Santo 2024"),
        ("2025-04-18", "Viernes Santo 2025"),
        ("2023-04-07", "Viernes Santo 2023"),
        ("2024-01-01", "Año Nuevo 2024"),
        ("2024-01-15", "Martin Luther King Jr. 2024"),
        ("2024-02-19", "Presidents' Day 2024"),
        ("2024-05-27", "Memorial Day 2024"),
        ("2024-07-04", "Día de la Independencia 2024"),
        ("2024-09-02", "Labor Day 2024"),
        ("2024-12-25", "Navidad 2024"),
        ("2023-01-02", "Año Nuevo 2023 observado en lunes (1-ene fue domingo)"),
        ("2021-12-24", "Navidad 2021 en sábado -> festivo el viernes 24"),
        ("2015-07-03", "4-jul-2015 en sábado -> festivo el viernes 3"),
        ("2021-07-05", "4-jul-2021 en domingo -> festivo el lunes 5"),
        ("2022-12-26", "Navidad 2022 en domingo -> festivo el lunes 26"),
    ],
)
def test_festivos_conocidos_cierran_el_mercado(cal: TradingCalendar, day: str, why: str) -> None:
    assert not cal.is_session(day), why
    assert cal.is_holiday(day)
    assert cal.holiday_name(day) is not None


def test_viernes_santo_2024_es_29_de_marzo() -> None:
    assert good_friday(2024) == dt.date(2024, 3, 29)
    assert good_friday(2025) == dt.date(2025, 4, 18)
    assert good_friday(2000) == dt.date(2000, 4, 21)


def test_juneteenth_solo_desde_2022(cal: TradingCalendar) -> None:
    # El 19 de junio de 2021 cayó en sábado y el NYSE abrió el viernes 18:
    # la ley federal se firmó el 17 de junio de 2021 y el NYSE no lo observó.
    assert cal.is_session("2021-06-18")
    assert not cal.is_holiday("2021-06-18")
    # Desde 2022 sí, con el traslado habitual sábado->viernes / domingo->lunes.
    assert not cal.is_session("2022-06-20"), "19-jun-2022 en domingo -> lunes 20"
    assert not cal.is_session("2023-06-19")
    assert not cal.is_session("2024-06-19")
    assert not cal.is_session("2025-06-19")
    assert cal.holiday_name("2024-06-19") == "Juneteenth"


def test_mlk_no_se_observaba_antes_de_1998(cal: TradingCalendar) -> None:
    assert cal.is_session("1997-01-20"), "tercer lunes de enero de 1997: NYSE abierto"
    assert not cal.is_session("1998-01-19"), "primer MLK observado por el NYSE"


def test_ano_nuevo_en_sabado_no_se_observa(cal: TradingCalendar) -> None:
    # Regla asimétrica: el viernes anterior es cierre del ejercicio y el NYSE abre.
    for day in ("1999-12-31", "2004-12-31", "2010-12-31", "2021-12-31"):
        assert cal.is_session(day), f"{day} debe ser sesión"
    # En domingo sí se traslada al lunes.
    assert not cal.is_session("2012-01-02")
    assert not cal.is_session("2017-01-02")


@pytest.mark.parametrize(
    ("day", "why"),
    [
        ("1994-04-27", "duelo nacional: funeral de Nixon"),
        ("2001-09-11", "atentados del 11-S"),
        ("2001-09-12", "atentados del 11-S"),
        ("2001-09-13", "atentados del 11-S"),
        ("2001-09-14", "atentados del 11-S"),
        ("2004-06-11", "duelo nacional: funeral de Reagan"),
        ("2007-01-02", "duelo nacional: funeral de Ford"),
        ("2012-10-29", "huracán Sandy"),
        ("2012-10-30", "huracán Sandy"),
        ("2018-12-05", "duelo nacional: funeral de G. H. W. Bush"),
        ("2025-01-09", "duelo nacional: funeral de Carter"),
    ],
)
def test_cierres_extraordinarios(cal: TradingCalendar, day: str, why: str) -> None:
    assert not cal.is_session(day), why


def test_reaperturas_tras_los_cierres_extraordinarios(cal: TradingCalendar) -> None:
    assert cal.is_session("2001-09-17"), "el mercado reabrió el lunes 17 de septiembre"
    assert cal.next_session("2001-09-10") == dt.date(2001, 9, 17)
    assert cal.is_session("2012-10-31"), "el mercado reabrió el miércoles 31 de octubre"
    assert cal.next_session("2012-10-26") == dt.date(2012, 10, 31)
    assert cal.next_session("2006-12-29") == dt.date(2007, 1, 3), "el 2 de enero estuvo cerrado"


@pytest.mark.parametrize(
    ("year", "expected"),
    [
        (2001, 248),  # cuatro sesiones perdidas por el 11-S
        (2012, 250),  # dos sesiones perdidas por Sandy
        (2019, 252),
        (2020, 253),
        (2021, 252),
        (2022, 251),
        (2023, 250),
        (2024, 252),
        (2025, 250),  # una sesión perdida por el funeral de Carter
    ],
)
def test_conteo_anual_de_sesiones_exacto(cal: TradingCalendar, year: int, expected: int) -> None:
    counts = cal.session_counts_by_year()
    assert int(counts.loc[year]) == expected


def test_conteo_anual_de_sesiones_en_rango_plausible(cal: TradingCalendar) -> None:
    counts = cal.session_counts_by_year()
    plausible = counts.loc[1996:2035]
    assert plausible.min() >= 246, f"año con muy pocas sesiones: {plausible.idxmin()}"
    assert plausible.max() <= 254, f"año con demasiadas sesiones: {plausible.idxmax()}"
    # El grueso de los años cae en la horquilla clásica 250-253.
    assert (plausible.between(250, 253)).mean() > 0.85


# ===========================================================================
# 2. Calendario: medias sesiones y aperturas tardías
# ===========================================================================


@pytest.mark.parametrize(
    ("day", "close", "why"),
    [
        ("2024-07-03", dt.time(13, 0), "miércoles antes del 4 de julio, regla post-2013"),
        ("2024-11-29", dt.time(13, 0), "viernes de Acción de Gracias"),
        ("2024-12-24", dt.time(13, 0), "Nochebuena en martes"),
        ("2025-07-03", dt.time(13, 0), "jueves antes del 4 de julio"),
        ("2025-11-28", dt.time(13, 0), "viernes de Acción de Gracias"),
        ("2025-12-24", dt.time(13, 0), "Nochebuena en miércoles"),
        ("2019-07-03", dt.time(13, 0), "regla post-2013 en miércoles"),
        ("2013-07-03", dt.time(13, 0), "primer año de la regla del miércoles"),
        ("2002-07-05", dt.time(13, 0), "regla pre-2013: media sesión el viernes 5"),
        ("1997-12-26", dt.time(13, 0), "media sesión ad-hoc"),
        ("1999-12-31", dt.time(13, 0), "preparativos del efecto 2000"),
        ("2003-12-26", dt.time(13, 0), "media sesión ad-hoc"),
        ("1997-10-27", dt.time(15, 30), "cortacircuitos: cierre a las 15:30"),
        ("2005-06-01", dt.time(15, 56), "avería de sistemas"),
        ("1996-01-08", dt.time(14, 0), "ventisca del 96"),
    ],
)
def test_medias_sesiones(cal: TradingCalendar, day: str, close: dt.time, why: str) -> None:
    assert cal.is_session(day), why
    assert cal.is_early_close(day), why
    assert cal.close_time(day) == close, why


def test_sesiones_completas_que_parecen_medias(cal: TradingCalendar) -> None:
    # Desde 2013 el viernes posterior al 4 de julio es sesión completa.
    assert cal.close_time("2013-07-05") == dt.time(16, 0)
    assert cal.close_time("2019-07-05") == dt.time(16, 0)
    assert cal.close_time("2024-07-05") == dt.time(16, 0)
    # Con el 4 de julio en domingo no hay media sesión el viernes anterior.
    assert cal.close_time("2021-07-02") == dt.time(16, 0)
    # Y con Nochebuena en viernes (Navidad en sábado) el 24 es festivo, no media sesión.
    assert not cal.is_session("2021-12-24")


def test_aperturas_tardias(cal: TradingCalendar) -> None:
    assert cal.open_time("2002-09-11") == dt.time(12, 0), "primer aniversario del 11-S"
    assert cal.is_late_open("2002-09-11")
    assert cal.open_time("1996-01-08") == dt.time(11, 0), "ventisca del 96"
    assert cal.open_time("2024-07-15") == dt.time(9, 30)
    assert not cal.is_late_open("2024-07-15")


def test_early_closes_devuelve_serie_del_rango(cal: TradingCalendar) -> None:
    ser = cal.early_closes("2024-01-01", "2024-12-31")
    assert list(ser.index.date) == [
        dt.date(2024, 7, 3),
        dt.date(2024, 11, 29),
        dt.date(2024, 12, 24),
    ]
    assert set(ser.to_numpy()) == {dt.time(13, 0)}


def test_horarios_en_utc_respetan_el_horario_de_verano(cal: TradingCalendar) -> None:
    # Enero: EST (UTC-5) -> cierre a las 21:00 UTC.
    assert cal.close_utc("2024-01-16") == dt.datetime(2024, 1, 16, 21, 0)
    # Julio: EDT (UTC-4) -> cierre a las 20:00 UTC.
    assert cal.close_utc("2024-07-16") == dt.datetime(2024, 7, 16, 20, 0)
    assert cal.open_utc("2024-07-16") == dt.datetime(2024, 7, 16, 13, 30)
    # Media sesión: cierre a las 13:00 ET = 18:00 UTC en diciembre.
    assert cal.close_utc("2024-12-24") == dt.datetime(2024, 12, 24, 18, 0)


# ===========================================================================
# 3. Horario de verano calculado por nosotros
# ===========================================================================


def test_dst_coincide_con_zoneinfo_en_todo_el_rango() -> None:
    zoneinfo = pytest.importorskip("zoneinfo")
    try:
        ny = zoneinfo.ZoneInfo("America/New_York")
    except Exception:  # pragma: no cover - sin tzdata en el contenedor
        pytest.skip("tzdata no disponible; la implementación propia es la referencia")

    t = dt.datetime(1990, 1, 1)
    mismatches = 0
    while t < dt.datetime(2036, 1, 1):
        ref = t.replace(tzinfo=UTC).astimezone(ny).utcoffset()
        if ref != eastern_utc_offset(t):
            mismatches += 1
        t += dt.timedelta(hours=6)
    assert mismatches == 0


def test_transiciones_dst_exactas() -> None:
    # Regla moderna: segundo domingo de marzo, primer domingo de noviembre.
    assert eastern_utc_offset(dt.datetime(2024, 3, 10, 6, 59)) == dt.timedelta(hours=-5)
    assert eastern_utc_offset(dt.datetime(2024, 3, 10, 7, 0)) == dt.timedelta(hours=-4)
    assert eastern_utc_offset(dt.datetime(2024, 11, 3, 5, 59)) == dt.timedelta(hours=-4)
    assert eastern_utc_offset(dt.datetime(2024, 11, 3, 6, 0)) == dt.timedelta(hours=-5)
    # Regla antigua (1987-2006): primer domingo de abril, último domingo de octubre.
    assert eastern_utc_offset(dt.datetime(2005, 3, 13, 12, 0)) == dt.timedelta(hours=-5)
    assert eastern_utc_offset(dt.datetime(2005, 4, 3, 7, 0)) == dt.timedelta(hours=-4)
    assert eastern_utc_offset(dt.datetime(2005, 10, 30, 6, 0)) == dt.timedelta(hours=-5)


def test_ida_y_vuelta_utc_este() -> None:
    for moment in (
        dt.datetime(2024, 1, 16, 9, 30),
        dt.datetime(2024, 7, 16, 16, 0),
        dt.datetime(2024, 3, 11, 9, 30),
        dt.datetime(2024, 11, 4, 16, 0),
    ):
        assert utc_to_eastern(eastern_to_utc(moment)) == moment


def test_offsets_vectorizados_coinciden_con_el_escalar() -> None:
    idx = pd.date_range("2000-01-01", "2035-12-31", freq="53h")
    vec = eastern_offsets_for(idx)
    ref = pd.TimedeltaIndex([eastern_utc_offset(ts.to_pydatetime()) for ts in idx])
    assert (vec == ref).all()


# ===========================================================================
# 4. Navegación del calendario
# ===========================================================================


def test_next_prev_y_is_session(cal: TradingCalendar) -> None:
    assert cal.next_session("2024-11-27") == dt.date(2024, 11, 29), "salta Acción de Gracias"
    assert cal.prev_session("2024-11-29") == dt.date(2024, 11, 27)
    assert cal.next_session("2024-07-05") == dt.date(2024, 7, 8), "viernes -> lunes"
    assert cal.prev_session("2024-07-08") == dt.date(2024, 7, 5)
    assert cal.is_session("2024-07-05")
    assert not cal.is_session("2024-07-06"), "sábado"


def test_session_on_or_after_y_before(cal: TradingCalendar) -> None:
    assert cal.session_on_or_after("2024-07-05") == dt.date(2024, 7, 5)
    assert cal.session_on_or_after("2024-07-06") == dt.date(2024, 7, 8)
    assert cal.session_on_or_before("2024-07-06") == dt.date(2024, 7, 5)
    assert cal.session_on_or_before("2024-07-05") == dt.date(2024, 7, 5)


def test_shift_es_consistente_con_next_y_prev(cal: TradingCalendar) -> None:
    for day in ("2024-07-05", "2024-07-06", "2024-07-07", "2024-11-28", "2012-10-28"):
        assert cal.shift(day, 1) == cal.next_session(day), day
        assert cal.shift(day, -1) == cal.prev_session(day), day
    assert cal.shift("2024-07-03", 0) == dt.date(2024, 7, 3)
    assert cal.shift("2024-07-03", 5) == dt.date(2024, 7, 11)
    assert cal.shift("2024-07-11", -5) == dt.date(2024, 7, 3)


def test_shift_cero_sobre_dia_no_bursatil_lanza(cal: TradingCalendar) -> None:
    with pytest.raises(CalendarError, match="no es sesión"):
        cal.shift("2024-07-06", 0)


def test_shift_ida_y_vuelta_sobre_sesiones(cal: TradingCalendar) -> None:
    sessions = cal.sessions("2019-01-01", "2021-12-31")
    for n in (1, 3, 21, 63):
        sample = sessions[100:-100:37]
        for ts in sample:
            day = ts.date()
            assert cal.shift(cal.shift(day, n), -n) == day


def test_sessions_devuelve_indice_normalizado(cal: TradingCalendar) -> None:
    idx = cal.sessions("2024-01-01", "2024-12-31")
    assert isinstance(idx, pd.DatetimeIndex)
    assert idx.name == "date"
    assert idx.tz is None
    assert (idx == idx.normalize()).all()
    assert idx.is_monotonic_increasing and idx.is_unique
    assert len(idx) == 252
    assert idx[0].date() == dt.date(2024, 1, 2)
    assert idx[-1].date() == dt.date(2024, 12, 31)


def test_validacion_de_rango(cal: TradingCalendar) -> None:
    with pytest.raises(CalendarError, match="fuera del rango"):
        cal.sessions("1970-01-01", "1975-01-01")
    with pytest.raises(CalendarError, match="fuera del rango"):
        cal.is_session("2099-01-04")
    with pytest.raises(CalendarError, match="invertido"):
        cal.sessions("2024-12-31", "2024-01-01")
    with pytest.raises(CalendarError):
        TradingCalendar("BOLSA DE MADRID")
    with pytest.raises(CalendarError):
        TradingCalendar("XNYS", first_year=1900, last_year=1950)


def test_desplazamiento_fuera_del_calendario_lanza() -> None:
    small = TradingCalendar("XNYS", first_year=2020, last_year=2021)
    with pytest.raises(CalendarError, match="fuera del calendario"):
        small.shift("2021-12-31", 5)
    with pytest.raises(CalendarError):
        small.next_session("2021-12-31")


def test_session_distance_y_ventanas(cal: TradingCalendar) -> None:
    assert cal.session_distance("2024-07-03", "2024-07-08") == 2, "salta el 4 y el fin de semana"
    assert cal.session_distance("2024-07-08", "2024-07-03") == -2
    assert cal.session_distance("2024-07-03", "2024-07-03") == 0
    win = cal.sessions_around("2024-07-08", 2, 2)
    assert [ts.date() for ts in win] == [
        dt.date(2024, 7, 3),
        dt.date(2024, 7, 5),  # el 4 de julio no es sesión
        dt.date(2024, 7, 8),
        dt.date(2024, 7, 9),
        dt.date(2024, 7, 10),
    ]
    with pytest.raises(CalendarError):
        cal.sessions_around("2024-07-06", 2, 2)  # no es sesión


def test_nasdaq_comparte_calendario_con_nyse() -> None:
    assert get_calendar("NASDAQ") is get_calendar("XNYS")
    assert get_calendar("NYSE") is get_calendar("XNYS")


# ===========================================================================
# 5. Clasificación BMO / AMC / DMH
# ===========================================================================


@pytest.mark.parametrize(
    ("moment", "expected", "why"),
    [
        (et(2024, 7, 15, 7, 0), Session.BMO, "antes de la apertura"),
        (et(2024, 7, 15, 9, 29), Session.BMO, "un minuto antes de abrir"),
        (et(2024, 7, 15, 9, 30), Session.DMH, "la apertura ya es sesión"),
        (et(2024, 7, 15, 12, 0), Session.DMH, "media sesión de negociación"),
        (et(2024, 7, 15, 15, 59), Session.DMH, "un minuto antes del cierre"),
        (et(2024, 7, 15, 16, 0), Session.AMC, "la subasta de cierre ya se ejecutó"),
        (et(2024, 7, 15, 20, 0), Session.AMC, "bien entrada la tarde"),
        (et(2024, 7, 13, 10, 0), Session.BMO, "sábado: no hay sesión"),
        (et(2024, 12, 24, 12, 30), Session.DMH, "media sesión, aún abierto"),
        (et(2024, 12, 24, 13, 0), Session.AMC, "media sesión: cierre a las 13:00"),
        (et(2024, 12, 24, 13, 30), Session.AMC, "13:30 en Nochebuena NO es DMH"),
        (et(2002, 9, 11, 11, 0), Session.BMO, "apertura retrasada al mediodía"),
        (et(2002, 9, 11, 12, 30), Session.DMH, "ya abierto tras la apertura tardía"),
    ],
)
def test_classify_session(
    cal: TradingCalendar, moment: dt.datetime, expected: Session, why: str
) -> None:
    assert classify_session(moment, cal) is expected, why


def test_classify_sessions_vectorizado_coincide_con_el_escalar(cal: TradingCalendar) -> None:
    rng = np.random.default_rng(20260803)
    base = pd.Timestamp("2018-01-01")
    stamps = [
        (base + pd.Timedelta(minutes=int(m))).to_pydatetime().replace(tzinfo=UTC)
        for m in rng.integers(0, 8 * 365 * 24 * 60, size=800)
    ]
    vec = classify_sessions(pd.Series(stamps), cal)
    scalar = [classify_session(s, cal) for s in stamps]
    assert list(vec) == scalar


def test_parse_session_normaliza_sinonimos() -> None:
    assert parse_session("BMO") is Session.BMO
    assert parse_session("Before Market Open") is Session.BMO
    assert parse_session("pre-market") is Session.BMO
    assert parse_session("amc") is Session.AMC
    assert parse_session("After Close") is Session.AMC
    assert parse_session("during market hours") is Session.DMH
    assert parse_session(Session.DMH) is Session.DMH
    # Lo desconocido nunca se adivina como BMO: se marca UNKNOWN.
    assert parse_session(None) is Session.UNKNOWN
    assert parse_session("") is Session.UNKNOWN
    assert parse_session("--") is Session.UNKNOWN
    assert parse_session(float("nan")) is Session.UNKNOWN
    assert parse_session("cualquier cosa rara") is Session.UNKNOWN


# ===========================================================================
# 6. tradable_date: la función más crítica del repo
# ===========================================================================


@pytest.mark.parametrize(
    ("moment", "session", "expected", "why"),
    [
        (et(2025, 1, 30, 16, 5), Session.AMC, "2025-01-31", "AMC de jueves -> viernes"),
        (et(2025, 1, 31, 7, 0), Session.BMO, "2025-01-31", "BMO -> misma sesión"),
        (et(2025, 1, 31, 16, 10), Session.AMC, "2025-02-03", "AMC de VIERNES -> LUNES"),
        (et(2025, 2, 1, 7, 0), Session.BMO, "2025-02-03", "sábado -> siguiente sesión"),
        (et(2025, 3, 10, 16, 5), Session.AMC, "2025-03-11", "tras el cambio de hora"),
        (et(2025, 1, 31, 11, 0), Session.DMH, "2025-01-31", "DMH -> misma sesión"),
        (et(2025, 1, 31, 16, 10), Session.UNKNOWN, "2025-02-03", "UNKNOWN se trata como AMC"),
        (et(2024, 11, 27, 16, 5), Session.AMC, "2024-11-29", "AMC víspera de Acción de Gracias"),
        (et(2024, 11, 29, 13, 30), Session.AMC, "2024-12-02", "AMC en media sesión de viernes"),
        (et(2024, 12, 24, 13, 30), Session.AMC, "2024-12-26", "AMC de Nochebuena salta Navidad"),
        (et(2012, 10, 26, 16, 5), Session.AMC, "2012-10-31", "AMC antes del huracán Sandy"),
        (et(2001, 9, 10, 16, 5), Session.AMC, "2001-09-17", "AMC del 10-S: se reabrió el 17"),
        (et(2006, 12, 29, 16, 5), Session.AMC, "2007-01-03", "el 2 de enero de 2007 cerró"),
        (et(2021, 6, 17, 16, 5), Session.AMC, "2021-06-18", "en 2021 el 18 de junio fue sesión"),
        (et(2022, 6, 17, 16, 5), Session.AMC, "2022-06-21", "en 2022 Juneteenth ya cierra"),
    ],
)
def test_tradable_date_casos_conocidos(
    cal: TradingCalendar,
    moment: dt.datetime,
    session: Session,
    expected: str,
    why: str,
) -> None:
    assert tradable_date(moment, session, cal) == dt.date.fromisoformat(expected), why


def test_tradable_date_la_trampa_de_la_fecha_utc(cal: TradingCalendar) -> None:
    """Un anuncio de las 20:30 ET cae en el día natural UTC siguiente.

    Tomar la fecha directamente del timestamp UTC desplaza el evento una sesión
    entera. El resultado correcto es el 31 de enero, no el 3 de febrero.
    """
    moment = dt.datetime(2025, 1, 31, 1, 30, tzinfo=UTC)  # = jue 30-ene 20:30 ET
    assert tradable_date(moment, Session.AMC, cal) == dt.date(2025, 1, 31)
    # Comprobación de que el fallo que se evita es real:
    ingenuo = cal.next_session(moment.date())
    assert ingenuo == dt.date(2025, 2, 3)
    assert ingenuo != tradable_date(moment, Session.AMC, cal)


def test_tradable_date_viernes_amc_siempre_cae_en_la_siguiente_sesion(
    cal: TradingCalendar,
) -> None:
    """Barrido de todos los viernes de 2024: un AMC nunca es negociable ese día."""
    sessions = cal.sessions("2024-01-01", "2024-12-31")
    fridays = [ts.date() for ts in sessions if ts.dayofweek == 4]
    assert len(fridays) >= 50
    for friday in fridays:
        moment = eastern_to_utc(dt.datetime(friday.year, friday.month, friday.day, 16, 30))
        got = tradable_date(moment.replace(tzinfo=UTC), Session.AMC, cal)
        assert got > friday, f"AMC del {friday} no puede negociarse el mismo día"
        assert got == cal.next_session(friday)
        assert got.weekday() == 0 or not cal.is_session(
            friday + dt.timedelta(days=3)
        ), "salvo festivo, el destino es lunes"


def test_tradable_date_unknown_es_igual_de_conservador_que_amc(cal: TradingCalendar) -> None:
    sessions = cal.sessions("2023-01-01", "2024-12-31")
    for ts in sessions[::7]:
        day = ts.date()
        moment = eastern_to_utc(dt.datetime(day.year, day.month, day.day, 18, 0)).replace(
            tzinfo=UTC
        )
        assert tradable_date(moment, Session.UNKNOWN, cal) == tradable_date(
            moment, Session.AMC, cal
        )


def test_tradable_date_nunca_precede_al_anuncio(cal: TradingCalendar) -> None:
    """Invariante duro: la fecha negociable nunca es anterior al día del anuncio."""
    rng = np.random.default_rng(7)
    base = pd.Timestamp("2015-01-01")
    for _ in range(500):
        moment = (base + pd.Timedelta(minutes=int(rng.integers(0, 9 * 365 * 24 * 60)))).to_pydatetime()
        moment = moment.replace(tzinfo=UTC)
        local_day = utc_to_eastern(moment.replace(tzinfo=None)).date()
        for session in Session:
            got = tradable_date(moment, session, cal)
            assert got >= local_day
            assert cal.is_session(got)
            if session in (Session.AMC, Session.UNKNOWN):
                assert got > local_day or not cal.is_session(local_day)


def test_tradable_date_verify_detecta_etiquetas_incoherentes(cal: TradingCalendar) -> None:
    moment = et(2024, 7, 15, 16, 30)  # claramente AMC
    assert tradable_date(moment, Session.AMC, cal, verify=True) == dt.date(2024, 7, 16)
    with pytest.raises(DataQualityError, match="contradice el timestamp"):
        tradable_date(moment, Session.BMO, cal, verify=True)
    # UNKNOWN nunca contradice: no hay nada que verificar.
    assert tradable_date(moment, Session.UNKNOWN, cal, verify=True) == dt.date(2024, 7, 16)


def test_tradable_date_sin_zona_horaria(cal: TradingCalendar) -> None:
    naive = dt.datetime(2025, 1, 31, 21, 10)  # UTC por convención del repo
    assert tradable_date(naive, Session.AMC, cal) == dt.date(2025, 2, 3)
    with pytest.raises(DataQualityError, match="tzinfo"):
        tradable_date(naive, Session.AMC, cal, assume_utc_if_naive=False)


def test_tradable_dates_vectorizado_coincide_con_el_escalar(cal: TradingCalendar) -> None:
    rng = np.random.default_rng(11)
    base = pd.Timestamp("2019-01-01")
    stamps = [base + pd.Timedelta(minutes=int(m)) for m in rng.integers(0, 6 * 365 * 24 * 60, 400)]
    sessions = rng.choice(["bmo", "amc", "dmh", "unknown"], size=400)
    events = pd.DataFrame({"announced_at": stamps, "session": sessions, "ticker": "AAA"})
    vec = tradable_dates(events, cal)
    scalar = [
        tradable_date(ts.to_pydatetime().replace(tzinfo=UTC), parse_session(s), cal)
        for ts, s in zip(stamps, sessions, strict=True)
    ]
    assert [d.date() for d in vec] == scalar


def test_tradable_dates_sin_columna_de_sesion_trata_todo_como_amc(cal: TradingCalendar) -> None:
    events = pd.DataFrame({"announced_at": [pd.Timestamp("2025-01-31 12:00")], "ticker": ["AAA"]})
    got = tradable_dates(events, cal)
    assert got.iloc[0] == pd.Timestamp("2025-02-03")


def test_tradable_dates_falla_si_falta_announced_at(cal: TradingCalendar) -> None:
    with pytest.raises(DataQualityError, match="announced_at"):
        tradable_dates(pd.DataFrame({"ticker": ["AAA"]}), cal)
    with pytest.raises(DataQualityError, match="sin `announced_at`"):
        tradable_dates(pd.DataFrame({"ticker": ["AAA"], "announced_at": [pd.NaT]}), cal)


# ===========================================================================
# 7. asof_join
# ===========================================================================


def test_asof_join_reproduce_la_validacion_de_referencia() -> None:
    """Tres vintages del mismo hecho contra fechas de señal mensuales."""
    facts = pd.DataFrame(
        {
            "ticker": ["A", "A", "A"],
            "available_at": pd.to_datetime(["2025-02-05", "2025-05-01", "2025-08-14"]),
            "value": [1.00, 1.10, 0.85],
        }
    )
    dates = pd.to_datetime([f"2025-{m:02d}-01" for m in range(1, 10)])
    got = asof_join(dates, facts)["value"].droplevel("ticker")

    assert pd.isna(got.loc["2025-01-01"]), "antes del primer vintage: NaN, no relleno atrás"
    assert pd.isna(got.loc["2025-02-01"])
    assert got.loc["2025-03-01"] == 1.00
    assert got.loc["2025-04-01"] == 1.00
    assert got.loc["2025-05-01"] == 1.10, "el día exacto del vintage ya cuenta"
    assert got.loc["2025-07-01"] == 1.10
    assert got.loc["2025-09-01"] == 0.85
    # El join ingenuo (último valor conocido hoy) daría 0.85 en 8 de 9 fechas.
    assert (got == 0.85).sum() == 1


def test_asof_join_nunca_adelanta() -> None:
    facts = pd.DataFrame(
        {"ticker": ["A"], "available_at": [pd.Timestamp("2025-06-10")], "value": [1.0]}
    )
    dates = pd.to_datetime(["2025-06-09", "2025-06-10", "2025-06-11"])
    got = asof_join(dates, facts)["value"].droplevel("ticker")
    assert pd.isna(got.loc["2025-06-09"]), "el día ANTES no existe"
    assert got.loc["2025-06-10"] == 1.0, "el día EXACTO sí"
    assert got.loc["2025-06-11"] == 1.0


def test_asof_join_devuelve_panel_canonico() -> None:
    facts = pd.DataFrame(
        {
            "ticker": ["A", "B"],
            "available_at": pd.to_datetime(["2025-01-02", "2025-01-03"]),
            "value": [1.0, 2.0],
        }
    )
    out = asof_join(pd.to_datetime(["2025-01-05", "2025-01-06"]), facts)
    assert isinstance(out.index, pd.MultiIndex)
    assert list(out.index.names) == ["date", "ticker"]
    assert out.index.is_monotonic_increasing
    assert out.index.get_level_values("date").tz is None


def test_asof_join_es_independiente_por_ticker() -> None:
    facts = pd.DataFrame(
        {
            "ticker": ["A", "B", "A", "B"],
            "available_at": pd.to_datetime(
                ["2025-01-10", "2025-03-10", "2025-05-10", "2025-07-10"]
            ),
            "value": [1.0, 10.0, 2.0, 20.0],
        }
    )
    dates = pd.to_datetime(["2025-02-01", "2025-04-01", "2025-06-01", "2025-08-01"])
    out = asof_join(dates, facts)["value"].unstack("ticker")  # noqa: PD010 - sin agregación
    assert list(out["A"]) == [1.0, 1.0, 2.0, 2.0]
    assert list(out["B"].fillna(-1)) == [-1.0, 10.0, 10.0, 20.0]


def test_asof_join_lag_days_retrasa_la_disponibilidad() -> None:
    facts = pd.DataFrame(
        {"ticker": ["A"], "available_at": [pd.Timestamp("2025-06-10")], "value": [1.0]}
    )
    dates = pd.to_datetime(["2025-06-10", "2025-06-12", "2025-06-13"])
    sin_lag = asof_join(dates, facts, 0)["value"].droplevel("ticker")
    con_lag = asof_join(dates, facts, 3)["value"].droplevel("ticker")
    assert sin_lag.loc["2025-06-10"] == 1.0
    assert pd.isna(con_lag.loc["2025-06-10"])
    assert pd.isna(con_lag.loc["2025-06-12"])
    assert con_lag.loc["2025-06-13"] == 1.0
    with pytest.raises(ValueError, match="lag_days"):
        asof_join(dates, facts, -1)


def test_asof_join_max_staleness_invalida_datos_rancios() -> None:
    facts = pd.DataFrame(
        {"ticker": ["A"], "available_at": [pd.Timestamp("2020-01-02")], "value": [1.0]}
    )
    dates = pd.to_datetime(["2020-02-01", "2021-02-01"])
    got = asof_join(dates, facts, max_staleness_days=180)["value"].droplevel("ticker")
    assert got.loc["2020-02-01"] == 1.0
    assert pd.isna(got.loc["2021-02-01"]), "un fundamental de hace un año no debe arrastrarse"


def test_asof_join_politicas_de_corte() -> None:
    """Un hecho publicado a las 21:30 UTC del día D.

    Con corte a medianoche (conservador) no está disponible ese día; con corte en
    la apertura de la sesión tampoco (la apertura son las 14:30 UTC); con corte a
    fin de día sí, porque el dato se conoció durante la jornada.
    """
    facts = pd.DataFrame(
        {
            "ticker": ["A"],
            "available_at": [pd.Timestamp("2025-06-10 21:30")],
            "value": [1.0],
        }
    )
    dates = pd.to_datetime(["2025-06-10", "2025-06-11"])
    conservador = asof_join(dates, facts, cutoff="date_start")["value"].droplevel("ticker")
    apertura = asof_join(dates, facts, cutoff="session_open")["value"].droplevel("ticker")
    fin_de_dia = asof_join(dates, facts, cutoff="date_end")["value"].droplevel("ticker")

    assert pd.isna(conservador.loc["2025-06-10"])
    assert conservador.loc["2025-06-11"] == 1.0
    assert pd.isna(apertura.loc["2025-06-10"])
    assert apertura.loc["2025-06-11"] == 1.0
    assert fin_de_dia.loc["2025-06-10"] == 1.0


def test_asof_join_acepta_panel_de_senal_con_multiindex() -> None:
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2025-03-01", "2025-06-01"]), ["A", "B"]], names=["date", "ticker"]
    )
    signal = pd.DataFrame({"score": [0.1, 0.2, 0.3, 0.4]}, index=idx)
    facts = pd.DataFrame(
        {
            "ticker": ["A", "B"],
            "available_at": pd.to_datetime(["2025-02-01", "2025-05-01"]),
            "value": [1.0, 2.0],
        }
    )
    out = asof_join(signal, facts)
    assert list(out.columns) == ["score", "value", "available_at"]
    assert out.loc[(pd.Timestamp("2025-03-01"), "A"), "value"] == 1.0
    assert pd.isna(out.loc[(pd.Timestamp("2025-03-01"), "B"), "value"])
    assert out.loc[(pd.Timestamp("2025-06-01"), "B"), "value"] == 2.0
    assert out.loc[(pd.Timestamp("2025-03-01"), "A"), "score"] == 0.1


def test_asof_join_resuelve_colisiones_de_nombres() -> None:
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2025-06-01"]), ["A"]], names=["date", "ticker"]
    )
    signal = pd.DataFrame({"value": [99.0], "available_at": [pd.Timestamp("1999-01-01")]}, index=idx)
    facts = pd.DataFrame(
        {"ticker": ["A"], "available_at": [pd.Timestamp("2025-01-01")], "value": [1.0]}
    )
    out = asof_join(signal, facts)
    assert out.loc[(pd.Timestamp("2025-06-01"), "A"), "value"] == 99.0, "la señal se conserva"
    assert out.loc[(pd.Timestamp("2025-06-01"), "A"), "value_pit"] == 1.0
    assert out.loc[(pd.Timestamp("2025-06-01"), "A"), "available_at_pit"] == pd.Timestamp(
        "2025-01-01"
    )


def test_asof_join_falla_ruidosamente_ante_datos_incompletos() -> None:
    dates = pd.to_datetime(["2025-01-01"])
    with pytest.raises(InsufficientHistory, match="vacío"):
        asof_join(dates, pd.DataFrame(columns=["ticker", "available_at", "value"]))
    with pytest.raises(LookAheadError, match="available_at"):
        asof_join(dates, pd.DataFrame({"ticker": ["A"], "value": [1.0]}))
    with pytest.raises(LookAheadError, match="sin `available_at`"):
        asof_join(
            dates, pd.DataFrame({"ticker": ["A"], "available_at": [pd.NaT], "value": [1.0]})
        )
    with pytest.raises(DataQualityError, match="ticker"):
        asof_join(dates, pd.DataFrame({"available_at": [pd.Timestamp("2024-01-01")], "v": [1.0]}))


def test_asof_join_acepta_timestamps_con_zona() -> None:
    facts = pd.DataFrame(
        {
            "ticker": ["A"],
            "available_at": pd.to_datetime(["2025-06-10T21:30:00Z"]),
            "value": [1.0],
        }
    )
    got = asof_join(pd.to_datetime(["2025-06-10", "2025-06-11"]), facts)["value"]
    assert pd.isna(got.iloc[0])
    assert got.iloc[1] == 1.0


def test_asof_join_nunca_usa_el_futuro_fuerza_bruta() -> None:
    """Comparación exhaustiva contra un cálculo elemental por bucle.

    Se generan hechos y fechas aleatorios, se ejecuta `asof_join` y se verifica
    fila a fila que (a) el valor devuelto coincide con el último hecho disponible
    calculado a mano y (b) ningún `available_at` devuelto supera el corte.
    """
    rng = random.Random(20260803)
    tickers = ["AAA", "BBB", "CCC", "DDD"]
    rows = []
    for ticker in tickers:
        for _ in range(rng.randint(3, 12)):
            day = pd.Timestamp("2020-01-01") + pd.Timedelta(
                minutes=rng.randint(0, 5 * 365 * 24 * 60)
            )
            rows.append({"ticker": ticker, "available_at": day, "value": rng.random()})
    facts = pd.DataFrame(rows)

    dates = pd.to_datetime(
        sorted(
            {
                (pd.Timestamp("2019-06-01") + pd.Timedelta(days=rng.randint(0, 2200))).normalize()
                for _ in range(120)
            }
        )
    )

    for lag in (0, 1, 7):
        out = asof_join(dates, facts, lag)
        assert len(out) == len(dates) * len(tickers)
        for (date_, ticker), row in out.iterrows():
            cutoff = date_ - pd.Timedelta(days=lag)
            candidates = facts[
                (facts["ticker"] == ticker) & (facts["available_at"] <= cutoff)
            ]
            if candidates.empty:
                assert pd.isna(row["value"]), (date_, ticker)
                assert pd.isna(row["available_at"])
            else:
                best = candidates.loc[candidates["available_at"].idxmax()]
                assert row["value"] == pytest.approx(best["value"]), (date_, ticker)
                assert row["available_at"] == best["available_at"]
                # La garantía central: nada del futuro.
                assert row["available_at"] <= cutoff


# ===========================================================================
# 8. assert_no_lookahead
# ===========================================================================


def _events_fixture(cal: TradingCalendar) -> pd.DataFrame:
    events = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "BBB"],
            "period_end": pd.to_datetime(["2024-03-31", "2024-06-30", "2024-03-31"]),
            "announced_at": [
                et(2024, 4, 25, 16, 30),  # AMC -> negociable el 26
                et(2024, 7, 25, 7, 0),  # BMO -> negociable el 25
                et(2024, 4, 26, 16, 5),  # AMC de viernes -> negociable el lunes 29
            ],
            "session": ["amc", "bmo", "amc"],
        }
    )
    events["event_id"] = [
        f"{t}:{p:%Y}Q{((p.month - 1) // 3) + 1}:{p:%Y-%m-%d}"
        for t, p in zip(events["ticker"], events["period_end"], strict=True)
    ]
    events["tradable_date"] = tradable_dates(events, cal)
    return events


def test_assert_no_lookahead_acepta_una_senal_correcta(cal: TradingCalendar) -> None:
    events = _events_fixture(cal)
    assert events.loc[0, "tradable_date"] == pd.Timestamp("2024-04-26")
    assert events.loc[1, "tradable_date"] == pd.Timestamp("2024-07-25")
    assert events.loc[2, "tradable_date"] == pd.Timestamp("2024-04-29")

    signal = pd.DataFrame(
        {
            "date": events["tradable_date"],
            "ticker": events["ticker"],
            "event_id": events["event_id"],
            "sue": [1.2, -0.4, 0.8],
        }
    ).set_index(["date", "ticker"])
    assert_no_lookahead(signal, events, cal=cal)  # no lanza


def test_assert_no_lookahead_caza_el_evento_amc_operado_el_mismo_dia(
    cal: TradingCalendar,
) -> None:
    """El error clásico: anclar la señal al día del anuncio en vez de al negociable."""
    events = _events_fixture(cal)
    signal = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-04-25", "2024-07-25", "2024-04-26"]),
            "ticker": events["ticker"],
            "event_id": events["event_id"],
            "sue": [1.2, -0.4, 0.8],
        }
    ).set_index(["date", "ticker"])
    with pytest.raises(LookAheadError, match="aún no negociables"):
        assert_no_lookahead(signal, events, cal=cal)


def test_assert_no_lookahead_caza_available_at_futuro(cal: TradingCalendar) -> None:
    events = _events_fixture(cal)
    signal = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-05-01", "2024-05-01"]),
            "ticker": ["AAA", "BBB"],
            "available_at": pd.to_datetime(["2024-04-26", "2024-12-31"]),
            "score": [1.0, 2.0],
        }
    ).set_index(["date", "ticker"])
    with pytest.raises(LookAheadError, match="posterior al corte"):
        assert_no_lookahead(signal, events, cal=cal)


def test_assert_no_lookahead_ignora_observaciones_sin_valor(cal: TradingCalendar) -> None:
    """Una fila con NaN no usa nada, así que no puede usar el futuro."""
    events = _events_fixture(cal)
    signal = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-05-01"]),
            "ticker": ["AAA", "AAA"],
            "available_at": pd.to_datetime(["2024-04-26", "2024-04-26"]),
            "score": [np.nan, 1.0],
        }
    ).set_index(["date", "ticker"])
    assert_no_lookahead(signal, events, cal=cal)


def test_assert_no_lookahead_por_ticker_y_period_end(cal: TradingCalendar) -> None:
    events = _events_fixture(cal)
    signal = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-04-26", "2024-04-26"]),
            "ticker": ["AAA", "BBB"],
            "period_end": pd.to_datetime(["2024-03-31", "2024-03-31"]),
            "sue": [1.0, 2.0],
        }
    ).set_index(["date", "ticker"])
    # BBB solo era negociable el 29 de abril: la fila del 26 es look-ahead.
    with pytest.raises(LookAheadError):
        assert_no_lookahead(signal, events.drop(columns="event_id"), cal=cal)


def test_assert_no_lookahead_primera_disponibilidad(cal: TradingCalendar) -> None:
    events = _events_fixture(cal)
    signal = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-15", "2024-05-15"]),
            "ticker": ["AAA", "AAA"],
            "sue": [1.0, 1.0],
        }
    ).set_index(["date", "ticker"])
    # Sin la comprobación opt-in no hay clave que auditar -> error explícito.
    with pytest.raises(DataQualityError, match="no se pudo auditar"):
        assert_no_lookahead(signal, events, cal=cal)
    with pytest.raises(LookAheadError, match="antes del primer evento"):
        assert_no_lookahead(signal, events, cal=cal, check_first_event=True)


def test_assert_no_lookahead_no_pasa_en_verde_sin_auditar_nada(cal: TradingCalendar) -> None:
    signal = pd.DataFrame(
        {"date": pd.to_datetime(["2024-05-01"]), "ticker": ["AAA"], "sue": [1.0]}
    ).set_index(["date", "ticker"])
    with pytest.raises(DataQualityError):
        assert_no_lookahead(signal, pd.DataFrame(), cal=cal)


def test_assert_no_lookahead_valida_la_salida_de_asof_join(cal: TradingCalendar) -> None:
    """Integración: lo que produce `asof_join` debe pasar siempre la auditoría."""
    facts = pd.DataFrame(
        {
            "ticker": ["AAA"] * 3 + ["BBB"] * 3,
            "available_at": pd.to_datetime(
                [
                    "2024-02-01 21:30",
                    "2024-05-02 21:00",
                    "2024-08-01 20:30",
                    "2024-02-15 21:30",
                    "2024-05-16 21:00",
                    "2024-08-15 20:30",
                ]
            ),
            "value": [1.0, 1.1, 1.2, 2.0, 2.1, 2.2],
        }
    )
    dates = cal.sessions("2024-01-01", "2024-12-31")
    panel = asof_join(dates, facts)
    events = _events_fixture(cal)
    assert_no_lookahead(panel.reset_index(), events, cal=cal)

    # Y si alguien "adelanta" la señal un día, la auditoría debe caerse.
    trucado = panel.reset_index()
    trucado["date"] = trucado["date"] - pd.Timedelta(days=40)
    with pytest.raises(LookAheadError):
        assert_no_lookahead(trucado, events, cal=cal)


# ===========================================================================
# 9. Vintages y reexpresiones
# ===========================================================================


def _fact(
    value: float, available: dt.datetime, form: str, *, restated: bool = False, ticker: str = "AAA"
) -> FundamentalFact:
    return FundamentalFact(
        ticker=ticker,
        concept="Revenue",
        value=value,
        period_end=dt.date(2024, 3, 31),
        available_at=available,
        fiscal_period="2024Q1",
        form=form,
        is_restated=restated,
    )


@pytest.fixture
def vintages() -> list[FundamentalFact]:
    return [
        _fact(1000.0, dt.datetime(2024, 4, 25, 20, 30), "8-K"),
        _fact(1000.0, dt.datetime(2024, 5, 2, 21, 0), "10-Q"),
        _fact(940.0, dt.datetime(2025, 2, 14, 21, 30), "10-K/A", restated=True),
        _fact(500.0, dt.datetime(2024, 4, 26, 20, 30), "8-K", ticker="BBB"),
    ]


def test_first_reported_es_el_primer_vintage(vintages: list[FundamentalFact]) -> None:
    first = first_reported(vintages).set_index("ticker")
    assert first.loc["AAA", "value"] == 1000.0
    assert first.loc["AAA", "form"] == "8-K"
    assert first.loc["AAA", "available_at"] == pd.Timestamp("2024-04-25 20:30")
    assert first.loc["BBB", "value"] == 500.0


def test_as_restated_es_el_ultimo_vintage(vintages: list[FundamentalFact]) -> None:
    last = as_restated(vintages).set_index("ticker")
    assert last.loc["AAA", "value"] == 940.0
    assert last.loc["AAA", "form"] == "10-K/A"
    assert last.loc["BBB", "value"] == 500.0


def test_vintage_asof_reconstruye_el_almacen_historico(vintages: list[FundamentalFact]) -> None:
    antes = vintage_asof(vintages, "2024-12-31").set_index("ticker")
    assert antes.loc["AAA", "value"] == 1000.0, "la reexpresión de 2025 aún no existía"
    despues = vintage_asof(vintages, "2025-06-30").set_index("ticker")
    assert despues.loc["AAA", "value"] == 940.0
    # Antes del primer vintage el almacén está vacío, no lleno de ceros.
    assert len(vintage_asof(vintages, "2024-01-01")) == 0


def test_la_reexpresion_no_contamina_el_pasado(vintages: list[FundamentalFact]) -> None:
    """El test obligatorio de §4.3 de la nota de investigación."""
    facts = pd.DataFrame(
        [
            {"ticker": "A", "available_at": pd.Timestamp("2024-05-02"), "value": 1000.0},
            {"ticker": "A", "available_at": pd.Timestamp("2025-02-14"), "value": 940.0},
        ]
    )
    got = asof_join(pd.to_datetime(["2024-06-30", "2025-06-30"]), facts)["value"]
    assert got.loc[(pd.Timestamp("2024-06-30"), "A")] == 1000.0, "el valor original"
    assert got.loc[(pd.Timestamp("2025-06-30"), "A")] == 940.0, "la reexpresión, DESDE su fecha"


def test_revisions_mide_la_correccion(vintages: list[FundamentalFact]) -> None:
    rev = revisions(vintages).set_index("ticker")
    assert rev.loc["AAA", "n_vintages"] == 3
    assert rev.loc["AAA", "first_value"] == 1000.0
    assert rev.loc["AAA", "final_value"] == 940.0
    assert rev.loc["AAA", "abs_revision"] == pytest.approx(-60.0)
    assert rev.loc["AAA", "rel_revision"] == pytest.approx(-0.06)
    assert rev.loc["AAA", "delay_days"] == 295
    assert bool(rev.loc["AAA", "is_revised"])
    assert bool(rev.loc["AAA", "flagged_restated"])
    assert not bool(rev.loc["BBB", "is_revised"]), "un solo vintage no es una revisión"


def test_revisions_no_marca_revision_si_el_valor_se_repite() -> None:
    facts = [
        _fact(1000.0, dt.datetime(2024, 4, 25, 20, 30), "8-K"),
        _fact(1000.0, dt.datetime(2024, 5, 2, 21, 0), "10-Q"),
    ]
    rev = revisions(facts)
    assert rev.loc[0, "n_vintages"] == 2
    assert not bool(rev.loc[0, "is_revised"])


def test_restatement_magnitude(vintages: list[FundamentalFact]) -> None:
    stats = restatement_magnitude(vintages)
    assert stats.n_keys == 2
    assert stats.n_facts == 4
    assert stats.n_revised_keys == 1
    assert stats.revised_share == pytest.approx(0.5)
    assert stats.median_abs_rel == pytest.approx(0.06)
    assert stats.median_signed_rel == pytest.approx(-0.06)
    assert stats.negative_share == pytest.approx(1.0)
    assert stats.median_delay_days == pytest.approx(295.0)
    assert "Revenue" in stats.by_concept.index
    assert stats.as_dict()["n_revised_keys"] == 1


def test_restatement_magnitude_con_muchas_claves() -> None:
    """Magnitud típica sobre una muestra grande y controlada."""
    rng = np.random.default_rng(3)
    rows = []
    for i in range(400):
        ticker = f"T{i:03d}"
        base = 1000.0 + rng.normal(0, 50)
        rows.append(
            {
                "ticker": ticker,
                "concept": "Revenue",
                "period_end": pd.Timestamp("2024-03-31"),
                "available_at": pd.Timestamp("2024-05-01"),
                "value": base,
            }
        )
        if i < 40:  # el 10% se reexpresa, con una revisión del -2%
            rows.append(
                {
                    "ticker": ticker,
                    "concept": "Revenue",
                    "period_end": pd.Timestamp("2024-03-31"),
                    "available_at": pd.Timestamp("2025-02-01"),
                    "value": base * 0.98,
                }
            )
    stats = restatement_magnitude(pd.DataFrame(rows))
    assert stats.n_keys == 400
    assert stats.n_revised_keys == 40
    assert stats.revised_share == pytest.approx(0.10)
    assert stats.median_abs_rel == pytest.approx(0.02, abs=1e-9)
    assert stats.median_signed_rel < 0


def test_restatement_magnitude_falla_si_no_hay_nada_que_medir() -> None:
    with pytest.raises(InsufficientHistory):
        restatement_magnitude(pd.DataFrame(columns=["ticker", "concept", "period_end",
                                                    "available_at", "value"]))
    facts = [_fact(1.0, dt.datetime(2024, 4, 25), "8-K")]
    with pytest.raises(InsufficientHistory):
        restatement_magnitude(facts, concepts=["NoExiste"])


def test_facts_to_frame_exige_available_at() -> None:
    with pytest.raises(DataQualityError, match="available_at"):
        first_reported(
            pd.DataFrame(
                {
                    "ticker": ["A"],
                    "concept": ["Revenue"],
                    "period_end": [pd.Timestamp("2024-03-31")],
                    "available_at": [pd.NaT],
                    "value": [1.0],
                }
            )
        )
    with pytest.raises(DataQualityError, match="faltan columnas"):
        first_reported(pd.DataFrame({"ticker": ["A"], "value": [1.0]}))
    with pytest.raises(InsufficientHistory):
        first_reported([])


def test_upsert_vintage_es_append_only_y_solo_registra_cambios() -> None:
    store = upsert_vintage(
        None, [_fact(1000.0, dt.datetime(2024, 4, 25, 20, 30), "8-K")], dt.datetime(2024, 4, 26)
    )
    assert len(store) == 1

    # Confirmación del mismo valor en el 10-Q: NO genera vintage nuevo.
    store = upsert_vintage(
        store, [_fact(1000.0, dt.datetime(2024, 5, 2, 21, 0), "10-Q")], dt.datetime(2024, 5, 3)
    )
    assert len(store) == 1, "un valor repetido no es información nueva"

    # La reexpresión sí.
    store = upsert_vintage(
        store,
        [_fact(940.0, dt.datetime(2025, 2, 14, 21, 30), "10-K/A", restated=True)],
        dt.datetime(2025, 2, 15),
    )
    assert len(store) == 2
    assert list(store["value"]) == [1000.0, 940.0]
    assert bool(store.iloc[1]["is_restated"])

    # Y el histórico reconstruido respeta el pasado.
    got = asof_join(pd.to_datetime(["2024-06-30", "2025-06-30"]), store, value_cols=["value"])
    assert got.loc[(pd.Timestamp("2024-06-30"), "AAA"), "value"] == 1000.0
    assert got.loc[(pd.Timestamp("2025-06-30"), "AAA"), "value"] == 940.0


def test_upsert_vintage_marca_reexpresion_aunque_el_proveedor_no_lo_diga() -> None:
    store = upsert_vintage(
        None, [_fact(1000.0, dt.datetime(2024, 4, 25), "10-Q")], dt.datetime(2024, 4, 26)
    )
    store = upsert_vintage(
        store, [_fact(900.0, dt.datetime(2024, 4, 25), "10-Q")], dt.datetime(2025, 1, 15)
    )
    assert len(store) == 2
    assert bool(store.iloc[1]["is_restated"]), "el cambio silencioso se marca igualmente"
    # El `available_at` declarado era anterior al vintage previo: se usa el momento
    # en que realmente lo observamos, que es lo único demostrable.
    assert store.iloc[1]["available_at"] == pd.Timestamp("2025-01-15")


def test_upsert_vintage_nunca_fecha_por_delante_de_la_observacion() -> None:
    store = upsert_vintage(
        None, [_fact(1000.0, dt.datetime(2030, 1, 1), "10-Q")], dt.datetime(2024, 4, 26)
    )
    assert store.iloc[0]["available_at"] == pd.Timestamp("2024-04-26")


# ===========================================================================
# 10. Casos frontera adicionales
# ===========================================================================


def test_tradable_date_en_los_bordes_del_calendario() -> None:
    small = TradingCalendar("XNYS", first_year=2020, last_year=2021)
    ultimo = et(2021, 12, 31, 16, 30)
    with pytest.raises(CalendarError, match="no hay sesión posterior"):
        tradable_date(ultimo, Session.AMC, small)
    # BMO del último día sí resuelve: es la propia sesión.
    assert tradable_date(et(2021, 12, 31, 7, 0), Session.BMO, small) == dt.date(2021, 12, 31)
    with pytest.raises(CalendarError, match="fuera del rango"):
        tradable_date(et(2019, 6, 3, 7, 0), Session.BMO, small)


def test_tradable_date_dmh_en_fin_de_semana_cae_en_la_siguiente_sesion(
    cal: TradingCalendar,
) -> None:
    """DMH un sábado es contradictorio; la respuesta segura es la siguiente sesión."""
    got = tradable_date(et(2024, 7, 6, 12, 0), Session.DMH, cal)
    assert got == dt.date(2024, 7, 8)


def test_asof_join_acepta_fechas_como_lista_de_cadenas() -> None:
    facts = pd.DataFrame(
        {"ticker": ["A"], "available_at": [pd.Timestamp("2025-01-15")], "value": [1.0]}
    )
    got = asof_join(pd.Index(["2025-01-10", "2025-02-10"]), facts)["value"].droplevel("ticker")
    assert pd.isna(got.iloc[0])
    assert got.iloc[1] == 1.0


def test_asof_join_cutoff_de_apertura_exige_que_la_fecha_sea_sesion(
    cal: TradingCalendar,
) -> None:
    facts = pd.DataFrame(
        {"ticker": ["A"], "available_at": [pd.Timestamp("2025-01-02")], "value": [1.0]}
    )
    with pytest.raises(CalendarError, match="no es una sesión"):
        asof_join(pd.to_datetime(["2025-01-04"]), facts, cutoff="session_open", cal=cal)


def test_assert_no_lookahead_respeta_lag_days(cal: TradingCalendar) -> None:
    events = _events_fixture(cal)
    signal = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-04-26"]),
            "ticker": ["AAA"],
            "event_id": [events.loc[0, "event_id"]],
            "sue": [1.0],
        }
    ).set_index(["date", "ticker"])
    assert_no_lookahead(signal, events, cal=cal)  # sin lag pasa
    with pytest.raises(LookAheadError):
        assert_no_lookahead(signal, events, cal=cal, lag_days=1)


def test_revisions_con_primer_valor_nulo_no_inventa_un_cociente() -> None:
    facts = pd.DataFrame(
        {
            "ticker": ["A", "A"],
            "concept": ["NetIncome", "NetIncome"],
            "period_end": [pd.Timestamp("2024-03-31")] * 2,
            "available_at": pd.to_datetime(["2024-05-01", "2025-02-01"]),
            "value": [0.0, 25.0],
        }
    )
    rev = revisions(facts)
    assert rev.loc[0, "abs_revision"] == 25.0
    assert pd.isna(rev.loc[0, "rel_revision"]), "no hay revisión relativa sobre un cero"
    assert bool(rev.loc[0, "is_revised"])


def test_upsert_vintage_rechaza_dos_vintages_de_la_misma_clave_en_un_lote() -> None:
    lote = [
        _fact(1000.0, dt.datetime(2024, 4, 25), "8-K"),
        _fact(940.0, dt.datetime(2024, 5, 2), "10-Q"),
    ]
    with pytest.raises(DataQualityError, match="dos vintages de la misma clave"):
        upsert_vintage(None, lote, dt.datetime(2024, 5, 3))


def test_vintage_asof_acepta_marca_temporal_con_zona(vintages: list[FundamentalFact]) -> None:
    got = vintage_asof(vintages, pd.Timestamp("2024-12-31T00:00:00Z")).set_index("ticker")
    assert got.loc["AAA", "value"] == 1000.0


def test_first_reported_desempata_a_favor_del_no_reexpresado() -> None:
    momento = dt.datetime(2024, 5, 2, 21, 0)
    facts = [
        _fact(940.0, momento, "10-K/A", restated=True),
        _fact(1000.0, momento, "10-Q"),
    ]
    first = first_reported(facts)
    assert first.loc[0, "value"] == 1000.0
    assert first.loc[0, "form"] == "10-Q"


def test_assert_no_lookahead_usa_la_fecha_negociable_mas_tardia_ante_duplicados(
    cal: TradingCalendar,
) -> None:
    """Con eventos duplicados manda la lectura estricta, no la más optimista."""
    events = _events_fixture(cal)
    duplicado = events.iloc[[0]].copy()
    duplicado["tradable_date"] = pd.Timestamp("2024-05-10")  # revisión posterior
    events = pd.concat([events, duplicado], ignore_index=True)

    signal = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-04-26"]),
            "ticker": ["AAA"],
            "event_id": [events.loc[0, "event_id"]],
            "sue": [1.0],
        }
    ).set_index(["date", "ticker"])
    with pytest.raises(LookAheadError, match="aún no negociables"):
        assert_no_lookahead(signal, events, cal=cal)
