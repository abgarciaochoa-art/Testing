"""Tests del módulo `universe`, ejecutables **sin red**.

La mayor parte se apoya en los ficheros semilla *reales* del repo
(`data/seed/sp500_historical_components.csv`, 3.482 filas de 1996-01-02 a
2025-08-23, y `data/seed/sp500_constituents.csv`, 503 constituyentes con CIK y
GICS): probar el universo contra un fixture inventado no demostraría nada sobre el
sesgo de supervivencia, que es justo lo que este módulo debe evitar.

Los tests que requieren red real llevan `@pytest.mark.network` y están excluidos por
defecto; se ejecutan con `pytest -m network`.
"""

from __future__ import annotations

import datetime as dt
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from earnings_alpha.errors import (
    DataQualityError,
    InsufficientHistory,
    ProviderUnavailable,
    UniverseError,
)
from earnings_alpha.universe import (
    ConstituentSnapshot,
    GithubConstituentsSource,
    GithubDatasetSource,
    IdentifierMap,
    IdentifierRecord,
    SlickchartsSource,
    SP500Universe,
    WikipediaSource,
    build_source,
    clean_symbol,
    detect_symbol_changes,
    merge_identifier_maps,
    parse_html_tables,
    read_history_csv,
    refresh_history_file,
    ticker_spans,
)
from earnings_alpha.universe.sources import history_frame_to_records

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED_DIR = REPO_ROOT / "data" / "seed"
HISTORY_CSV = SEED_DIR / "sp500_historical_components.csv"
CONSTITUENTS_CSV = SEED_DIR / "sp500_constituents.csv"

FIRST_SNAPSHOT = dt.date(1996, 1, 2)
LAST_SNAPSHOT = dt.date(2025, 8, 23)

# Símbolos que no existían (o no estaban en el índice) en 1996. Si alguno apareciera
# en el universo de 1996, el fichero no sería point-in-time sino la lista de hoy
# proyectada hacia atrás: sesgo de supervivencia en estado puro.
MODERN_TICKERS = [
    "TSLA", "META", "NVDA", "GOOGL", "GOOG", "NFLX", "AMZN", "PLTR", "CRM", "ABNB",
    "UBER", "PYPL", "V", "MA", "BRK.B", "FB", "MRNA", "NOW", "PANW", "AXON", "COIN",
    "TTD", "SMCI", "KVUE", "GEHC", "VLTO", "DAY", "FI", "GEN", "ELV", "RVTY", "WTW",
]

# Miembros reales del índice el 1996-01-02 que hoy ya no existen como tales
# (AMR, Bank of Boston, Chemical Bank, Digital Equipment, Caldor, Amdahl...).
LEGACY_1996_TICKERS = ["AAMRQ", "ABI", "ABS", "ACKH", "AGC", "AHM", "BHMSQ", "BKB", "CMB", "DEC"]

# Los tests de red llevan la marca `network` (declarada en pyproject) y además se
# saltan salvo que se pidan explícitamente. En el contenedor de desarrollo el proxy
# de egress devuelve 403 para Wikipedia y Slickcharts, así que ejecutarlos por
# defecto haría fallar la suite por una condición del entorno, no del código.
requires_network = pytest.mark.skipif(
    os.environ.get("EARNINGS_ALPHA_RUN_NETWORK", "").lower() not in {"1", "true", "yes"},
    reason="requiere red real; exportar EARNINGS_ALPHA_RUN_NETWORK=1 para ejecutarlos",
)


@pytest.fixture(scope="module")
def universe() -> SP500Universe:
    """Universo sobre los ficheros semilla reales (parseo cacheado entre tests)."""
    return SP500Universe(HISTORY_CSV, CONSTITUENTS_CSV)


# ============================================================ carga e integridad


def test_seed_files_present() -> None:
    """Los ficheros semilla existen: sin ellos el resto de tests no prueba nada."""
    assert HISTORY_CSV.exists(), "falta el histórico de composición"
    assert CONSTITUENTS_CSV.exists(), "falta la tabla de constituyentes"


def test_index_shape_and_range(universe: SP500Universe) -> None:
    """El índice cubre 1996→2025 y deduplica las dos fechas repetidas del fichero."""
    idx = universe.index
    raw_rows = len(pd.read_csv(HISTORY_CSV))
    assert raw_rows == 3482, "el fichero semilla ha cambiado de tamaño"
    # 2022-06-21 y 2023-05-14 aparecen dos veces en el CSV; se colapsan a una.
    assert idx.n_snapshots == 3480
    assert universe.first_snapshot == FIRST_SNAPSHOT
    assert universe.last_snapshot == LAST_SNAPSHOT
    assert len(idx.tickers) > 1000, "el histórico debe contener símbolos ya desaparecidos"
    assert idx.matrix.dtype == np.bool_
    assert idx.matrix.shape == (idx.n_snapshots, len(idx.tickers))


def test_snapshot_dates_are_strictly_increasing(universe: SP500Universe) -> None:
    """Monotonía temporal estricta: sin duplicados ni desorden."""
    dates = universe.snapshot_dates()
    assert dates.is_monotonic_increasing
    assert not dates.has_duplicates
    assert dates[0] == pd.Timestamp(FIRST_SNAPSHOT)
    assert dates[-1] == pd.Timestamp(LAST_SNAPSHOT)


def test_member_counts_in_plausible_range(universe: SP500Universe) -> None:
    """El recuento por fecha se mantiene en un rango plausible para el S&P 500.

    Rango observado en la semilla: 442-505. El mínimo (442, en 2009) está por debajo
    de 500 porque el dataset omite algunas clases de acciones secundarias y algunos
    símbolos de empresas ya quebradas; se documenta el hecho en vez de aserciones
    sobre un 500 exacto que el dato no cumple.
    """
    counts = universe.member_counts()
    assert len(counts) == universe.index.n_snapshots
    assert counts.min() >= 440
    assert counts.max() <= 510
    # A partir de 2016 la cobertura del dataset es prácticamente completa.
    modern = counts[counts.index >= "2016-06-01"]
    assert modern.min() >= 478
    assert modern.max() <= 510
    assert 460 <= counts.median() <= 505


def test_no_abrupt_jumps_between_consecutive_snapshots(universe: SP500Universe) -> None:
    """Ningún salto brusco de miembros: delataría un snapshot truncado."""
    counts = universe.member_counts().to_numpy()
    assert np.abs(np.diff(counts)).max() <= 25


def test_validate_reports_no_issues_on_seed(universe: SP500Universe) -> None:
    """La semilla supera las comprobaciones de integridad del propio módulo."""
    assert universe.validate() == []


def test_missing_history_file_raises_provider_unavailable(tmp_path: Path) -> None:
    """Un fichero ausente es indisponibilidad de proveedor, no un panel vacío."""
    u = SP500Universe(tmp_path / "no_existe.csv", CONSTITUENTS_CSV)
    with pytest.raises(ProviderUnavailable):
        _ = u.index
    with pytest.raises(ProviderUnavailable):
        read_history_csv(tmp_path / "tampoco.csv")


def test_history_frame_with_bad_schema_raises(tmp_path: Path) -> None:
    """Un CSV sin las columnas esperadas falla explícitamente."""
    bad = tmp_path / "bad.csv"
    bad.write_text("fecha,miembros\n2020-01-01,AAPL\n", encoding="utf-8")
    with pytest.raises(DataQualityError):
        read_history_csv(bad)


# ================================================================== members_on


def test_members_on_exact_snapshot(universe: SP500Universe) -> None:
    """En una fecha con snapshot se devuelve esa composición, ordenada y sin repetir."""
    members = universe.members_on(FIRST_SNAPSHOT)
    assert members == sorted(members)
    assert len(members) == len(set(members))
    assert len(members) == 469
    assert universe.snapshot_date_for(FIRST_SNAPSHOT) == FIRST_SNAPSHOT


def test_members_on_forward_fills_from_previous_snapshot(universe: SP500Universe) -> None:
    """Una fecha sin snapshot usa el ANTERIOR más próximo, nunca el posterior.

    Se elige un hueco real de 28 días (2019-01-18 → 2019-02-15) en el que además la
    composición cambia (`ATO` entra, `NFX` sale). Si la resolución mirara hacia
    adelante, `ATO` aparecería en el universo del 1 de febrero de 2019, dos semanas
    antes de que su entrada fuera pública.
    """
    prev_snap, next_snap = dt.date(2019, 1, 18), dt.date(2019, 2, 15)
    target = dt.date(2019, 2, 1)
    assert universe.snapshot_date_for(target) == prev_snap
    assert universe.members_on(target) == universe.members_on(prev_snap)
    assert universe.members_on(target) != universe.members_on(next_snap)
    assert not universe.is_member("ATO", target)
    assert universe.is_member("ATO", next_snap)
    assert universe.is_member("NFX", target)
    assert not universe.is_member("NFX", next_snap)


def test_forward_fill_never_looks_ahead(universe: SP500Universe) -> None:
    """Propiedad general: el snapshot usado es siempre ≤ la fecha, y es el más reciente.

    Se comprueba sobre 400 fechas aleatorias (semilla fija) de todo el histórico. Un
    fallo aquí significaría look-ahead en el universo, que contamina cualquier
    backtest construido encima.
    """
    rng = random.Random(20260803)
    dates = universe.snapshot_dates()
    span = (LAST_SNAPSHOT - FIRST_SNAPSHOT).days
    for _ in range(400):
        d = FIRST_SNAPSHOT + dt.timedelta(days=rng.randrange(span + 1))
        used = universe.snapshot_date_for(d)
        assert used <= d
        later = dates[(dates > pd.Timestamp(used)) & (dates <= pd.Timestamp(d))]
        assert len(later) == 0, f"existe un snapshot entre {used} y {d}"
        assert universe.members_on(d) == universe.members_on(used)


def test_members_on_after_last_snapshot_uses_last(universe: SP500Universe) -> None:
    """Después del último snapshot se mantiene la última composición conocida."""
    future = LAST_SNAPSHOT + dt.timedelta(days=200)
    assert universe.snapshot_date_for(future) == LAST_SNAPSHOT
    assert universe.members_on(future) == universe.members_on(LAST_SNAPSHOT)


def test_members_before_first_snapshot_raises(universe: SP500Universe) -> None:
    """Antes del primer snapshot no hay respuesta honesta: `InsufficientHistory`."""
    with pytest.raises(InsufficientHistory):
        universe.members_on(dt.date(1995, 12, 29))
    with pytest.raises(InsufficientHistory):
        universe.membership_panel(dt.date(1990, 1, 1), dt.date(2000, 1, 1))


def test_max_forward_fill_days_guard() -> None:
    """`max_forward_fill_days` convierte un ffill excesivo en error explícito."""
    strict = SP500Universe(HISTORY_CSV, CONSTITUENTS_CSV, max_forward_fill_days=5)
    assert strict.members_on(LAST_SNAPSHOT)
    with pytest.raises(InsufficientHistory):
        strict.members_on(LAST_SNAPSHOT + dt.timedelta(days=60))
    with pytest.raises(UniverseError):
        SP500Universe(HISTORY_CSV, CONSTITUENTS_CSV, max_forward_fill_days=-1)


def test_date_input_forms_are_equivalent(universe: SP500Universe) -> None:
    """`date`, `datetime`, `str` y `Timestamp` dan el mismo resultado."""
    ref = universe.members_on(dt.date(2015, 3, 16))
    assert universe.members_on("2015-03-16") == ref
    assert universe.members_on(dt.datetime(2015, 3, 16, 15, 30)) == ref
    assert universe.members_on(pd.Timestamp("2015-03-16")) == ref
    with pytest.raises(UniverseError):
        universe.members_on("no-es-una-fecha")


def test_is_member_and_counts(universe: SP500Universe) -> None:
    """`is_member` coincide con `members_on` y normaliza el símbolo."""
    d = dt.date(2008, 6, 30)
    members = universe.members_set_on(d)
    assert universe.is_member("AAPL", d) is True
    assert universe.is_member("aapl", d) is True
    assert universe.is_member("TSLA", d) is False
    assert universe.is_member("ESTO.NO.EXISTE", d) is False
    assert universe.n_members_on(d) == len(members)
    assert universe.is_member("BRK-B", dt.date(2020, 1, 2)) is True


# =========================================================== membership_panel


def test_membership_panel_structure(universe: SP500Universe) -> None:
    """Panel booleano `date x ticker`, con fila en `start` y solo cambios después."""
    start, end = dt.date(2005, 1, 1), dt.date(2010, 12, 31)
    panel = universe.membership_panel(start, end)
    assert panel.index.name == "date"
    assert panel.columns.name == "ticker"
    assert all(pd.api.types.is_bool_dtype(d) for d in panel.dtypes)
    assert panel.index[0] == pd.Timestamp(start)
    assert panel.index.is_monotonic_increasing
    assert not panel.index.has_duplicates
    assert panel.index[-1] <= pd.Timestamp(end)
    # Todas las filas menos la primera son fechas de snapshot reales.
    snaps = set(universe.snapshot_dates(start, end))
    assert set(panel.index[1:]).issubset(snaps)


def test_membership_panel_rows_match_members_on(universe: SP500Universe) -> None:
    """Cada fila del panel reproduce exactamente `members_on` de esa fecha."""
    start, end = dt.date(2018, 1, 1), dt.date(2019, 12, 31)
    panel = universe.membership_panel(start, end)
    for day in [panel.index[0], panel.index[5], panel.index[-1]]:
        row = panel.loc[day]
        from_panel = sorted(row.index[row.to_numpy()])
        assert from_panel == universe.members_on(day.date())


def test_membership_panel_daily_grid(universe: SP500Universe) -> None:
    """Con `freq` el panel se densifica por forward-fill sin mirar al futuro."""
    start, end = dt.date(2019, 1, 1), dt.date(2019, 6, 30)
    panel = universe.membership_panel(start, end, freq="B")
    expected = pd.date_range(start, end, freq="B")
    assert panel.index.equals(expected.rename("date"))
    assert all(pd.api.types.is_bool_dtype(d) for d in panel.dtypes)
    for day in (expected[0], expected[40], expected[-1]):
        row = panel.loc[day]
        assert sorted(row.index[row.to_numpy()]) == universe.members_on(day.date())


def test_membership_panel_ticker_subset(universe: SP500Universe) -> None:
    """`tickers=` fija las columnas; un símbolo inexistente da columna toda False."""
    panel = universe.membership_panel(
        dt.date(2020, 1, 1), dt.date(2020, 12, 31), tickers=["AAPL", "brk-b", "NO.EXISTE"]
    )
    assert list(panel.columns) == ["AAPL", "BRK.B", "NO.EXISTE"]
    assert panel["AAPL"].all()
    assert not panel["NO.EXISTE"].any()


def test_membership_panel_inverted_range_raises(universe: SP500Universe) -> None:
    """Un rango invertido es un error de programación, no un panel vacío."""
    with pytest.raises(UniverseError):
        universe.membership_panel(dt.date(2020, 1, 1), dt.date(2019, 1, 1))


def test_membership_panel_full_history_is_efficient(universe: SP500Universe) -> None:
    """El panel completo (3.480 x ~1.125) se construye como vista de la matriz cacheada."""
    import time

    t0 = time.perf_counter()
    panel = universe.membership_panel(FIRST_SNAPSHOT, LAST_SNAPSHOT)
    elapsed = time.perf_counter() - t0
    assert panel.shape[0] == universe.index.n_snapshots
    assert panel.shape[1] == len(universe.index.tickers)
    assert elapsed < 2.0, f"membership_panel tardó {elapsed:.2f}s sobre el histórico completo"


def test_membership_index_is_canonical_multiindex(universe: SP500Universe) -> None:
    """`membership_index` devuelve el MultiIndex (date, ticker) del contrato."""
    mi = universe.membership_index(dt.date(2021, 1, 4), dt.date(2021, 1, 29), freq="B")
    assert isinstance(mi, pd.MultiIndex)
    assert mi.names == ["date", "ticker"]
    day = pd.Timestamp("2021-01-04")
    from_index = sorted(t for d, t in mi if d == day)
    assert from_index == universe.members_on(day.date())


# ============================================================== altas y bajas


def test_changes_schema(universe: SP500Universe) -> None:
    """`changes` devuelve (date, ticker, action) ordenado y con acciones válidas."""
    ch = universe.changes(dt.date(2015, 1, 1), dt.date(2016, 12, 31))
    assert list(ch.columns) == ["date", "ticker", "action"]
    assert set(ch["action"]) <= {"add", "delete"}
    assert ch["date"].is_monotonic_increasing
    assert pd.api.types.is_datetime64_any_dtype(ch["date"])
    assert len(ch) > 0


def test_additions_and_deletions_partition_changes(universe: SP500Universe) -> None:
    """`additions` + `deletions` reconstruyen exactamente `changes`."""
    start, end = dt.date(2012, 1, 1), dt.date(2014, 12, 31)
    adds = universe.additions(start, end)
    dels = universe.deletions(start, end)
    both = universe.changes(start, end)
    assert set(adds["action"]) == {"add"}
    assert set(dels["action"]) == {"delete"}
    assert len(adds) + len(dels) == len(both)


@pytest.mark.parametrize(
    ("day", "ticker", "action"),
    [
        # Tesla entra en el S&P 500 en la reconstitución de diciembre de 2020.
        (dt.date(2020, 12, 21), "TSLA", "add"),
        # Facebook pasa a cotizar como META el 9 de junio de 2022.
        (dt.date(2022, 6, 9), "META", "add"),
        (dt.date(2022, 6, 9), "FB", "delete"),
        # PerkinElmer se renombra Revvity.
        (dt.date(2023, 5, 18), "RVTY", "add"),
        (dt.date(2023, 5, 18), "PKI", "delete"),
        # Google entra en 2006; Netflix en diciembre de 2010; Palantir en 2024.
        (dt.date(2006, 4, 3), "GOOGL", "add"),
        (dt.date(2010, 12, 20), "NFLX", "add"),
        (dt.date(2024, 9, 22), "PLTR", "add"),
    ],
)
def test_known_index_events_are_detected(
    universe: SP500Universe, day: dt.date, ticker: str, action: str
) -> None:
    """Altas y bajas conocidas aparecen en la fecha correcta y solo en esa."""
    ch = universe.changes(day - dt.timedelta(days=45), day + dt.timedelta(days=45))
    hit = ch[(ch["ticker"] == ticker) & (ch["action"] == action)]
    assert len(hit) == 1, f"{ticker}/{action} debería aparecer una vez cerca de {day}"
    assert hit.iloc[0]["date"] == pd.Timestamp(day)


def test_lehman_leaves_the_index_in_september_2008(universe: SP500Universe) -> None:
    """Lehman Brothers (LEHMQ) está en el índice en junio de 2008 y sale tras quebrar."""
    assert universe.is_member("LEHMQ", dt.date(2008, 6, 30))
    dels = universe.deletions(dt.date(2008, 9, 1), dt.date(2008, 10, 31))
    lehman = dels[dels["ticker"] == "LEHMQ"]
    assert len(lehman) == 1
    assert lehman.iloc[0]["date"] == pd.Timestamp("2008-09-17")
    assert not universe.is_member("LEHMQ", dt.date(2008, 12, 31))


def test_first_snapshot_generates_no_additions(universe: SP500Universe) -> None:
    """Los miembros del primer snapshot no son "altas": no se observó su entrada."""
    ch = universe.changes(FIRST_SNAPSHOT, FIRST_SNAPSHOT)
    assert ch.empty
    ch2 = universe.changes(FIRST_SNAPSHOT, dt.date(1996, 1, 31))
    assert (ch2["date"] > pd.Timestamp(FIRST_SNAPSHOT)).all()


def test_changes_before_history_start_raises(universe: SP500Universe) -> None:
    """No se pueden derivar cambios de un periodo no observado."""
    with pytest.raises(InsufficientHistory):
        universe.changes(dt.date(1990, 1, 1), dt.date(2000, 1, 1))


def test_changes_reconstruct_membership(universe: SP500Universe) -> None:
    """Aplicar altas y bajas al universo inicial reproduce el universo final.

    Es el test más fuerte del módulo: si la detección de cambios se saltara o
    inventara un evento, la reconstrucción divergiría del panel.

    Nota de semántica: `changes(start, end)` incluye el cambio ocurrido *en* `start`
    (la diferencia respecto al snapshot anterior), así que la reconstrucción arranca
    del snapshot previo a `start`.
    """
    start, end = dt.date(2016, 1, 4), dt.date(2021, 12, 31)
    dates = universe.snapshot_dates()
    prev = dates[dates < pd.Timestamp(start)][-1].date()
    current = set(universe.members_on(prev))
    for _, row in universe.changes(start, end).iterrows():
        if row["action"] == "add":
            assert row["ticker"] not in current, f"alta duplicada de {row['ticker']}"
            current.add(row["ticker"])
        else:
            assert row["ticker"] in current, f"baja de un no miembro: {row['ticker']}"
            current.discard(row["ticker"])
    assert sorted(current) == universe.members_on(end)


def test_turnover_is_plausible(universe: SP500Universe) -> None:
    """La rotación anual del índice ronda las 20-25 altas; nunca centenares."""
    turnover = universe.turnover(dt.date(1997, 1, 1), dt.date(2024, 12, 31))
    assert turnover.min() >= 1
    assert turnover.max() <= 80
    assert 10 <= turnover.median() <= 45


# ================================================== ausencia de sesgo de supervivencia


def test_1996_universe_has_no_modern_tickers(universe: SP500Universe) -> None:
    """El universo de 1996 no contiene símbolos modernos: prueba de que es PIT.

    Si el fichero fuera la lista de hoy proyectada al pasado, Tesla o Meta aparecerían
    en 1996 y cualquier backtest de los 90 estaría comprando compañías que no cotizaban.
    """
    members_1996 = universe.members_set_on(FIRST_SNAPSHOT)
    intruders = sorted(t for t in MODERN_TICKERS if t in members_1996)
    assert intruders == [], f"símbolos modernos en el universo de 1996: {intruders}"


def test_1996_universe_contains_companies_that_no_longer_exist(
    universe: SP500Universe,
) -> None:
    """Y sí contiene compañías desaparecidas: el histórico no ha sido "limpiado"."""
    members_1996 = universe.members_set_on(FIRST_SNAPSHOT)
    present = [t for t in LEGACY_1996_TICKERS if t in members_1996]
    assert present == LEGACY_1996_TICKERS
    current = set(universe.constituents.index)
    assert not set(LEGACY_1996_TICKERS) & current


def test_universe_overlap_1996_vs_2025_is_partial(universe: SP500Universe) -> None:
    """Solo una minoría del S&P 500 de 1996 sigue en el índice 30 años después."""
    old = universe.members_set_on(FIRST_SNAPSHOT)
    new = universe.members_set_on(LAST_SNAPSHOT)
    overlap = len(old & new) / len(old)
    assert 0.20 < overlap < 0.60, f"solapamiento inesperado 1996↔2025: {overlap:.2%}"


def test_modern_tickers_enter_only_after_their_ipo(universe: SP500Universe) -> None:
    """Cada símbolo moderno aparece por primera vez en su fecha real de entrada."""
    spans = universe.spans()
    assert spans["TSLA"].first_seen == dt.date(2020, 12, 21)
    assert spans["META"].first_seen == dt.date(2022, 6, 9)
    assert spans["FB"].last_seen == dt.date(2022, 6, 8)
    assert spans["GOOGL"].first_seen == dt.date(2006, 4, 3)
    assert spans["AMZN"].first_seen == dt.date(2005, 11, 21)
    assert spans["AAPL"].first_seen == FIRST_SNAPSHOT
    assert spans["LEHMQ"].last_seen == dt.date(2008, 9, 16)


# ================================================= metadatos, CIK y sectores


def test_cik_and_sector_lookup(universe: SP500Universe) -> None:
    """CIK, sector y sub-industria se resuelven con normalización de símbolo."""
    assert universe.cik_for("AAPL") == "0000320193"
    assert universe.cik_for("aapl") == "0000320193"
    assert universe.sector_for("AAPL") == "Information Technology"
    assert universe.sub_industry_for("AAPL") is not None
    assert universe.security_name_for("MMM") == "3M"
    assert universe.date_added_for("MMM") == dt.date(1957, 3, 4)
    # Símbolos con clase de acción, en cualquiera de sus grafías.
    assert universe.cik_for("BRK-B") == universe.cik_for("BRK.B") == "0001067983"
    assert universe.sector_for("BF/B") == "Consumer Staples"


def test_cik_for_unknown_or_delisted_returns_none(universe: SP500Universe) -> None:
    """Sin correspondencia conocida se devuelve None, nunca un CIK inventado."""
    assert universe.cik_for("LEHMQ") is None
    assert universe.cik_for("NO.EXISTE") is None
    assert universe.sector_for("LEHMQ") is None


def test_cik_for_with_date_is_conservative(universe: SP500Universe) -> None:
    """Con fecha, solo se responde si el símbolo pertenecía al índice ese día."""
    assert universe.cik_for("AAPL", on=dt.date(2008, 6, 30)) == "0000320193"
    # Tesla no estaba en el índice en 2008: la respuesta conservadora es None.
    assert universe.cik_for("TSLA", on=dt.date(2008, 6, 30)) is None
    assert universe.cik_for("TSLA", on=dt.date(2021, 6, 30)) is not None


def test_sector_map_and_groups(universe: SP500Universe) -> None:
    """`sector_map` cubre a los miembros y deja NaN los símbolos sin metadato."""
    d = dt.date(2024, 6, 28)
    smap = universe.sector_map(d)
    members = universe.members_on(d)
    assert list(smap.index) == sorted(members)
    assert smap.notna().mean() > 0.90
    groups = universe.members_by_sector(d)
    assert sum(len(v) for v in groups.values()) == len(members)
    assert "Information Technology" in groups


def test_coverage_reports_the_gap_honestly(universe: SP500Universe) -> None:
    """`coverage()` cuantifica lo que la semilla NO tiene (CIK histórico)."""
    cov = universe.coverage()
    assert cov["n_snapshots"] == 3480
    assert cov["n_tickers_current"] == 503
    assert cov["n_tickers_historic"] > 1000
    # Solo los constituyentes actuales traen CIK: la cobertura histórica es parcial.
    assert 0.30 < float(cov["cik_coverage_historic"]) < 0.60
    assert cov["n_cik_collisions"] >= 3


def test_restrict_intersects_with_known_universe(universe: SP500Universe) -> None:
    """`restrict` normaliza y descarta lo que nunca estuvo en el índice."""
    assert universe.restrict(["aapl", "brk-b", "ZZZZ"]) == {"AAPL", "BRK.B"}


def test_repr_is_informative(universe: SP500Universe) -> None:
    """El repr muestra el rango cargado (útil en logs de backtest)."""
    text = repr(universe)
    assert "3480" in text and "1996-01-02" in text and "2025-08-23" in text


# ============================================================== identificadores


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("BRK-B", "BRK.B"),
        ("brk.b", "BRK.B"),
        ("BF/B", "BF.B"),
        (" aapl ", "AAPL"),
        ("RVTY (Previously PKI)", "RVTY"),
        ("BRK B", "BRK.B"),
        ("AZA.A", "AZA.A"),
    ],
)
def test_clean_symbol(raw: str, expected: str) -> None:
    """Normalización de símbolos, incluidas las anotaciones que traen los datasets."""
    assert clean_symbol(raw) == expected


def test_clean_symbol_rejects_empty() -> None:
    """Un símbolo vacío no puede colarse en el universo en silencio."""
    with pytest.raises(UniverseError):
        clean_symbol("   ")
    with pytest.raises(UniverseError):
        clean_symbol("(sin símbolo)")


def test_identifier_map_bidirectional(universe: SP500Universe) -> None:
    """El mapa resuelve ticker→CIK y CIK→tickers sobre la semilla real."""
    ident = universe.identifiers()
    assert len(ident) == 503
    assert ident.cik_for("AAPL") == "0000320193"
    assert ident.tickers_for("0000320193") == ("AAPL",)
    assert ident.tickers_for(320193) == ("AAPL",)
    assert "AAPL" in ident
    assert "NO.EXISTE" not in ident
    frame = ident.to_frame()
    assert frame.index.name == "ticker"
    assert frame.loc["AAPL", "sector"] == "Information Technology"


def test_identifier_map_collisions_are_share_classes(universe: SP500Universe) -> None:
    """Las colisiones CIK→varios tickers son clases múltiples de acciones."""
    collisions = universe.identifiers().collisions()
    families = set(collisions.values())
    assert ("GOOG", "GOOGL") in families
    assert ("NWS", "NWSA") in families
    assert ("FOX", "FOXA") in families


def test_primary_ticker_and_collision_resolution(universe: SP500Universe) -> None:
    """La elección de símbolo primario es determinista y colapsa los secundarios."""
    ident = universe.identifiers()
    cik = ident.cik_for("GOOGL")
    assert cik is not None
    primary = ident.primary_ticker(cik)
    assert primary in ("GOOG", "GOOGL")
    assert ident.primary_ticker(cik) == primary  # determinismo
    mapping = ident.resolve_collisions()
    # Cada colisión aporta exactamente un secundario redirigido al primario.
    assert len(mapping) == len(ident.collisions())
    for secondary, target in mapping.items():
        assert secondary != target
        assert ident.cik_for(secondary) == ident.cik_for(target)


def test_primary_ticker_honours_explicit_preference() -> None:
    """Una preferencia explícita gana a la regla de desempate."""
    records = [
        IdentifierRecord("GOOGL", "0001652044", date_added=dt.date(2006, 4, 3)),
        IdentifierRecord("GOOG", "0001652044", date_added=dt.date(2014, 4, 3)),
    ]
    default = IdentifierMap(records)
    assert default.primary_ticker("0001652044") == "GOOGL"  # entró antes
    forced = IdentifierMap(records, preferred={"0001652044": "GOOG"})
    assert forced.primary_ticker("0001652044") == "GOOG"


def test_identifier_map_records_conflicts_instead_of_overwriting() -> None:
    """Dos CIK para el mismo ticker se registran como conflicto; gana el primero."""
    m = IdentifierMap(
        [
            IdentifierRecord("XYZ", "0000000001", source="a"),
            IdentifierRecord("XYZ", "0000000002", source="b"),
        ]
    )
    assert m.cik_for("XYZ") == "0000000001"
    assert m.conflicts() == (("XYZ", "0000000001", "0000000002", "b"),)


def test_merge_identifier_maps_priority() -> None:
    """Al fusionar, manda el primer mapa y el desacuerdo queda anotado."""
    a = IdentifierMap([IdentifierRecord("AAA", "0000000001", source="a")])
    b = IdentifierMap(
        [
            IdentifierRecord("AAA", "0000000099", source="b"),
            IdentifierRecord("BBB", "0000000002", source="b"),
        ]
    )
    merged = merge_identifier_maps(a, b)
    assert merged.cik_for("AAA") == "0000000001"
    assert merged.cik_for("BBB") == "0000000002"
    assert len(merged.conflicts()) == 1


def test_ticker_spans_detects_reentry() -> None:
    """Un símbolo que sale y vuelve genera dos tramos, no uno continuo."""
    dates = [dt.date(2020, 1, d) for d in (1, 2, 3, 4, 5)]
    sets = [
        frozenset({"A", "B"}),
        frozenset({"A", "B"}),
        frozenset({"A"}),
        frozenset({"A", "B"}),
        frozenset({"A", "B"}),
    ]
    spans = ticker_spans(dates, sets)
    assert spans["A"].n_spans == 1
    assert spans["A"].has_gaps is False
    assert spans["B"].n_spans == 2
    assert spans["B"].has_gaps is True
    assert spans["B"].first_seen == dt.date(2020, 1, 1)
    assert spans["B"].last_seen == dt.date(2020, 1, 5)
    assert spans["B"].n_snapshots == 4
    with pytest.raises(UniverseError):
        ticker_spans(dates, sets[:2])


def test_seed_history_has_reentries(universe: SP500Universe) -> None:
    """El histórico real contiene reentradas: el panel no puede asumir tramos únicos."""
    spans = universe.spans()
    with_gaps = [s.ticker for s in spans.values() if s.has_gaps]
    assert len(with_gaps) > 0


def test_detect_symbol_changes_confirmed_by_cik() -> None:
    """Con CIK coincidente el candidato se marca como concluyente."""
    changes = pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-06-09", "2022-06-09"]),
            "ticker": ["FB", "META"],
            "action": ["delete", "add"],
        }
    )
    ident = IdentifierMap(
        [
            IdentifierRecord("FB", "0001326801"),
            IdentifierRecord("META", "0001326801"),
        ]
    )
    out = detect_symbol_changes(changes, identifiers=ident)
    assert len(out) == 1
    row = out.iloc[0]
    assert (row["old_ticker"], row["new_ticker"]) == ("FB", "META")
    assert row["confirmed_by_cik"] is True or bool(row["confirmed_by_cik"])
    assert row["tier"] == "confirmed"
    assert row["score"] == pytest.approx(1.0)
    assert "cik_match" in row["reasons"]


def test_detect_symbol_changes_share_class_family() -> None:
    """La familia de clases de acciones puntúa aunque no haya CIK."""
    changes = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02"] * 4),
            "ticker": ["NWS", "OTRA", "NWSA", "TERCERA"],
            "action": ["delete", "delete", "add", "add"],
        }
    )
    out = detect_symbol_changes(changes)
    pair = out[(out["old_ticker"] == "NWS") & (out["new_ticker"] == "NWSA")]
    assert len(pair) == 1
    assert "share_class_family" in pair.iloc[0]["reasons"]


def test_detect_symbol_changes_known_table_overrides() -> None:
    """Una tabla curada convierte el candidato en confirmado."""
    changes = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-05-18", "2023-05-18"]),
            "ticker": ["PKI", "RVTY"],
            "action": ["delete", "add"],
        }
    )
    out = detect_symbol_changes(changes, known_changes={"PKI": "RVTY"})
    assert out.iloc[0]["tier"] == "confirmed"
    assert "known_change" in out.iloc[0]["reasons"]


def test_detect_symbol_changes_on_seed_finds_real_renames(universe: SP500Universe) -> None:
    """Sobre el dato real aparecen los renombramientos conocidos como candidatos.

    También aparecen sustituciones auténticas (`TWTR`→`ACGL`): es la limitación
    documentada del método, no un fallo del test.
    """
    cands = universe.symbol_change_candidates(dt.date(2022, 1, 1), dt.date(2024, 1, 1))
    pairs = set(zip(cands["old_ticker"], cands["new_ticker"], strict=True))
    assert ("FB", "META") in pairs
    assert ("ANTM", "ELV") in pairs
    assert ("FISV", "FI") in pairs
    assert set(cands["tier"]) <= {"confirmed", "strong", "candidate"}
    # Sin evidencia de CIK (la semilla no la tiene para símbolos retirados) nada se
    # declara concluyente: el módulo no se atribuye certeza que no tiene.
    assert not cands["confirmed_by_cik"].any()


def test_detect_symbol_changes_validates_input() -> None:
    """Entradas mal formadas fallan; una entrada vacía devuelve el esquema vacío."""
    with pytest.raises(UniverseError):
        detect_symbol_changes(pd.DataFrame({"a": [1]}))
    empty = detect_symbol_changes(
        pd.DataFrame(columns=["date", "ticker", "action"])
    )
    assert empty.empty
    assert "old_ticker" in empty.columns


# ==================================================== fuentes (parsers offline)


WIKIPEDIA_FIXTURE = """
<html><body>
<table id="constituents" class="wikitable sortable">
<tbody>
<tr><th>Symbol</th><th>Security</th><th>GICS Sector</th><th>GICS Sub-Industry</th>
    <th>Headquarters Location</th><th>Date added</th><th>CIK</th><th>Founded</th></tr>
<tr><td><a href="/wiki/MMM">MMM</a></td><td><a href="/wiki/3M">3M</a></td>
    <td>Industrials</td><td>Industrial Conglomerates</td><td>Saint Paul, Minnesota</td>
    <td>1957-03-04<sup class="reference">[1]</sup></td><td>0000066740</td><td>1902</td></tr>
<tr><td>AAPL</td><td>Apple Inc.</td><td>Information Technology</td>
    <td>Technology Hardware, Storage &amp; Peripherals</td><td>Cupertino, California</td>
    <td>1982-11-30</td><td>0000320193</td><td>1977</td></tr>
<tr><td>BRK.B</td><td>Berkshire Hathaway</td><td>Financials</td><td>Multi-Sector Holdings</td>
    <td>Omaha, Nebraska</td><td>2010-02-16</td><td>0001067983</td><td>1839</td></tr>
</tbody>
</table>
<table class="wikitable">
<tbody>
<tr><th rowspan="2">Date</th><th colspan="2">Added</th><th colspan="2">Removed</th>
    <th rowspan="2">Reason</th></tr>
<tr><th>Ticker</th><th>Security</th><th>Ticker</th><th>Security</th></tr>
<tr><td>June 9, 2022</td><td>META</td><td>Meta Platforms</td><td>FB</td>
    <td>Meta Platforms</td><td>Ticker change</td></tr>
</tbody>
</table>
</body></html>
"""

SLICKCHARTS_FIXTURE = """
<html><body><table class="table">
<thead><tr><th>#</th><th>Company</th><th>Symbol</th><th>Weight</th><th>Price</th></tr></thead>
<tbody>
<tr><td>1</td><td>Apple Inc</td><td><a href="/symbol/AAPL">AAPL</a></td>
    <td>7.05%</td><td>212.44</td></tr>
<tr><td>2</td><td>Microsoft Corp</td><td>MSFT</td><td>6.50%</td><td>430.16</td></tr>
<tr><td>3</td><td>Berkshire Hathaway</td><td>BRK.B</td><td>1.70%</td><td>460.00</td></tr>
</tbody></table></body></html>
"""


def test_parse_html_tables_strips_markup() -> None:
    """El parser propio extrae texto limpio, sin enlaces ni referencias `<sup>`."""
    frames = parse_html_tables(WIKIPEDIA_FIXTURE)
    assert len(frames) == 2
    first = frames[0]
    assert list(first.columns)[:3] == ["Symbol", "Security", "GICS Sector"]
    assert first.iloc[0]["Symbol"] == "MMM"
    assert first.iloc[0]["Date added"] == "1957-03-04"  # la referencia [1] desaparece
    assert first.iloc[1]["GICS Sub-Industry"] == "Technology Hardware, Storage & Peripherals"


def test_wikipedia_parser_offline() -> None:
    """`WikipediaSource.parse_html` produce un snapshot con CIK y sector."""
    snap = WikipediaSource.parse_html(WIKIPEDIA_FIXTURE)
    assert snap.tickers == ("AAPL", "BRK.B", "MMM")
    by_ticker = {r.ticker: r for r in snap.records}
    assert by_ticker["AAPL"].cik == "0000320193"
    assert by_ticker["MMM"].sector == "Industrials"
    assert by_ticker["BRK.B"].date_added == dt.date(2010, 2, 16)
    assert snap.is_estimated_date is True
    # Con 3 miembros el snapshot es implausible y `validate` lo dice.
    assert snap.validate(strict=False)
    with pytest.raises(DataQualityError):
        snap.validate(strict=True)


def test_wikipedia_changes_table_offline() -> None:
    """La tabla de cambios se localiza sin confundirla con la de constituyentes.

    La cabecera de dos niveles con `colspan` se expande, de modo que las columnas
    quedan alineadas con las celdas de datos.
    """
    changes = WikipediaSource.parse_changes(WIKIPEDIA_FIXTURE)
    assert len(changes) == 1
    assert "Date" in changes.columns
    values = list(changes.iloc[0].to_numpy())
    assert "META" in values and "FB" in values
    # La tabla de constituyentes tiene "Date added", que NO debe activar la búsqueda.
    only_constituents = WIKIPEDIA_FIXTURE.split("<table class=\"wikitable\">")[0] + "</body></html>"
    assert WikipediaSource.parse_changes(only_constituents).empty


def test_slickcharts_parser_offline() -> None:
    """Slickcharts aporta símbolos y pesos por capitalización."""
    snap = SlickchartsSource.parse_html(SLICKCHARTS_FIXTURE)
    assert snap.tickers == ("AAPL", "BRK.B", "MSFT")
    weights = SlickchartsSource.parse_weights(SLICKCHARTS_FIXTURE)
    assert weights["AAPL"] == pytest.approx(0.0705)
    assert weights.index.name == "ticker"
    assert weights.sum() == pytest.approx(0.1525)


def test_html_without_expected_table_raises() -> None:
    """Un HTML sin la tabla esperada falla explícitamente."""
    with pytest.raises(DataQualityError):
        WikipediaSource.parse_html("<html><body><p>nada</p></body></html>")
    with pytest.raises(DataQualityError):
        SlickchartsSource.parse_html("<html><body><table><tr><td>x</td></tr></table></body></html>")


def test_github_constituents_parser_on_real_seed() -> None:
    """El parser de constituyentes funciona sobre el CSV semilla real (503 filas)."""
    frame = pd.read_csv(CONSTITUENTS_CSV)
    snap = GithubConstituentsSource.parse_frame(frame, source="test")
    assert snap.n_members == 503
    assert snap.validate(strict=False) == []
    by_ticker = {r.ticker: r for r in snap.records}
    assert by_ticker["AAPL"].cik == "0000320193"
    assert by_ticker["BRK.B"].sector == "Financials"


def test_constituent_snapshot_rejects_empty() -> None:
    """Un snapshot vacío es un error de calidad, no un resultado válido."""
    with pytest.raises(DataQualityError):
        ConstituentSnapshot(as_of=dt.date(2024, 1, 1), tickers=(), source="test")


def test_build_source_rejects_unknown_name() -> None:
    """Un nombre de fuente mal escrito falla en vez de no hacer nada."""
    assert isinstance(build_source("github"), GithubDatasetSource)
    assert isinstance(build_source("wikipedia"), WikipediaSource)
    with pytest.raises(UniverseError):
        build_source("wikipedía")


def test_duplicate_dates_keep_last_row(tmp_path: Path) -> None:
    """Ante fechas repetidas gana la última fila, que es la que enlaza con el futuro.

    Reproduce el caso real del fichero semilla en 2022-06-21: la primera fila repite
    el snapshot previo y la segunda incorpora la sustitución (`KDP` entra,
    `UA`/`UAA` salen), que es la que continúa en el snapshot siguiente.
    """
    csv = tmp_path / "dups.csv"
    csv.write_text(
        'date,tickers\n'
        '2022-06-09,"AAA,UA,UAA"\n'
        '2022-06-21,"AAA,UA,UAA"\n'
        '2022-06-21,"AAA,KDP"\n'
        '2022-06-28,"AAA,KDP"\n',
        encoding="utf-8",
    )
    records = history_frame_to_records(read_history_csv(csv))
    assert len(records) == 3
    assert records[1][1] == frozenset({"AAA", "KDP"})


def test_seed_duplicate_date_resolution_matches_next_snapshot(universe: SP500Universe) -> None:
    """En el fichero real, la resolución de 2022-06-21 enlaza con el 2022-06-28."""
    assert universe.is_member("KDP", dt.date(2022, 6, 21))
    assert not universe.is_member("UAA", dt.date(2022, 6, 21))
    assert universe.members_on(dt.date(2022, 6, 21)) == universe.members_on(dt.date(2022, 6, 27))


# ================================================= refresh append-only (offline)


def _synthetic_members(n: int = 460, *, offset: int = 0) -> list[str]:
    """Universo sintético plausible en tamaño, para probar el motor de refresco."""
    return [f"T{i:04d}" for i in range(offset, offset + n)]


def _write_history(path: Path, rows: list[tuple[str, list[str]]]) -> None:
    """Escribe un CSV de composición con el formato del fichero semilla."""
    lines = ["date,tickers"]
    lines += [f'{d},"{",".join(members)}"' for d, members in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class _FakeHistorySource:
    """Fuente histórica en memoria, para probar el refresco sin red."""

    def __init__(self, rows: list[tuple[str, list[str]]], name: str = "fake") -> None:
        self.name = name
        self._rows = rows

    def available(self) -> bool:
        return True

    def fetch_history(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": pd.to_datetime([d for d, _ in self._rows]),
                "tickers": [",".join(m) for _, m in self._rows],
            }
        )

    def fetch_snapshot(self) -> ConstituentSnapshot:
        day, members = self._rows[-1]
        return ConstituentSnapshot(
            as_of=dt.date.fromisoformat(day), tickers=tuple(members), source=self.name
        )


class _DeadSource:
    """Fuente que siempre falla, para probar la política de fallo explícito."""

    name = "dead"

    def available(self) -> bool:
        return False

    def fetch_snapshot(self) -> ConstituentSnapshot:
        raise ProviderUnavailable(self.name, "egress bloqueado en el entorno de prueba")


@pytest.fixture
def history_file(tmp_path: Path) -> Path:
    """Histórico pequeño pero de tamaño plausible, con dos snapshots."""
    base = _synthetic_members()
    path = tmp_path / "hist.csv"
    _write_history(path, [("2025-01-06", base), ("2025-02-03", base)])
    return path


def test_refresh_appends_only_new_dates(history_file: Path) -> None:
    """El refresco añade días nuevos y detecta las altas y bajas que implican."""
    base = _synthetic_members()
    nuevo = [*base[:-1], "ZZZZ"]  # sale T0459, entra ZZZZ
    src = _FakeHistorySource(
        [
            ("2025-01-06", base),
            ("2025-02-03", base),
            ("2025-03-03", nuevo),
        ]
    )
    before = history_file.read_bytes()
    report = refresh_history_file(history_file, [src])

    assert report.n_appended == 1
    assert report.dates_appended == (dt.date(2025, 3, 3),)
    assert report.rows_before == 2
    assert report.rows_after == 3
    assert report.sources_used == ("fake",)
    assert list(report.additions["ticker"]) == ["ZZZZ"]
    assert list(report.deletions["ticker"]) == ["T0459"]
    assert report.changed is True
    assert "fake" in report.summary()

    after = history_file.read_bytes()
    assert after.startswith(before), "el refresco reescribió bytes ya existentes"

    u = SP500Universe(history_file)
    assert u.last_snapshot == dt.date(2025, 3, 3)
    assert u.is_member("ZZZZ", dt.date(2025, 3, 3))
    assert not u.is_member("ZZZZ", dt.date(2025, 2, 3))


def test_refresh_never_rewrites_history(history_file: Path) -> None:
    """Una fuente que discrepa del pasado no lo modifica: solo se reporta."""
    base = _synthetic_members()
    falsificado = _synthetic_members(offset=1000)  # historia completamente distinta
    src = _FakeHistorySource(
        [
            ("2025-01-06", falsificado),
            ("2025-02-03", falsificado),
            ("2025-03-03", base),
        ]
    )
    original = history_file.read_text(encoding="utf-8")
    report = refresh_history_file(history_file, [src])

    assert history_file.read_text(encoding="utf-8").startswith(original)
    assert len(report.divergences) == 2
    assert set(report.divergences["date"].dt.date) == {
        dt.date(2025, 1, 6),
        dt.date(2025, 2, 3),
    }
    assert report.divergences.iloc[0]["only_local"]
    u = SP500Universe(history_file)
    assert u.members_on(dt.date(2025, 1, 6)) == sorted(base)


def test_refresh_refuses_retroactive_insertions(history_file: Path) -> None:
    """Un día intermedio que la fuente conoce y el fichero no NO se inserta."""
    base = _synthetic_members()
    src = _FakeHistorySource(
        [
            ("2025-01-06", base),
            ("2025-01-20", base),  # fecha intermedia ausente del fichero
            ("2025-02-03", base),
        ]
    )
    report = refresh_history_file(history_file, [src])
    assert report.n_appended == 0
    assert report.skipped_dates == (dt.date(2025, 1, 20),)
    assert any("append-only" in w for w in report.warnings)
    u = SP500Universe(history_file)
    assert len(u.snapshot_dates()) == 2


def test_refresh_dry_run_does_not_write(history_file: Path) -> None:
    """`dry_run` calcula el informe completo sin tocar el fichero."""
    base = _synthetic_members()
    src = _FakeHistorySource(
        [("2025-01-06", base), ("2025-02-03", base), ("2025-03-03", base)]
    )
    before = history_file.read_bytes()
    report = refresh_history_file(history_file, [src], dry_run=True)
    assert report.dry_run is True
    assert report.n_appended == 1
    assert report.changed is False
    assert report.rows_after == report.rows_before
    assert history_file.read_bytes() == before


def test_refresh_all_sources_failing_raises(history_file: Path) -> None:
    """Si ninguna fuente responde se lanza `ProviderUnavailable`, no un informe vacío."""
    with pytest.raises(ProviderUnavailable) as exc:
        refresh_history_file(history_file, [_DeadSource()])
    assert "dead" in str(exc.value)


def test_refresh_without_sources_raises(history_file: Path) -> None:
    """Una lista de fuentes vacía es un error de configuración."""
    with pytest.raises(UniverseError):
        refresh_history_file(history_file, [])


def test_refresh_rejects_implausible_snapshot(history_file: Path) -> None:
    """Un snapshot con 3 miembros no se escribe: sería corromper el histórico."""
    base = _synthetic_members()
    src = _FakeHistorySource(
        [("2025-01-06", base), ("2025-02-03", base), ("2025-03-03", ["A", "B", "C"])]
    )
    with pytest.raises(DataQualityError):
        refresh_history_file(history_file, [src])
    assert len(read_history_csv(history_file)) == 2


def test_refresh_handles_missing_trailing_newline(tmp_path: Path) -> None:
    """Un fichero sin salto de línea final no corrompe la última fila al anexar."""
    base = _synthetic_members()
    path = tmp_path / "sin_newline.csv"
    path.write_text(f'date,tickers\n2025-01-06,"{",".join(base)}"', encoding="utf-8")
    src = _FakeHistorySource([("2025-01-06", base), ("2025-02-03", base)])
    refresh_history_file(path, [src])
    records = history_frame_to_records(read_history_csv(path))
    assert [d for d, _ in records] == [dt.date(2025, 1, 6), dt.date(2025, 2, 3)]
    assert records[0][1] == frozenset(base)


def test_refresh_first_source_wins(history_file: Path) -> None:
    """Con dos fuentes que cubren el mismo día manda la de mayor prioridad."""
    base = _synthetic_members()
    primaria = _FakeHistorySource(
        [("2025-02-03", base), ("2025-03-03", [*base[:-1], "PRIM"])], name="primaria"
    )
    secundaria = _FakeHistorySource(
        [("2025-02-03", base), ("2025-03-03", [*base[:-1], "SEC"])], name="secundaria"
    )
    report = refresh_history_file(history_file, [primaria, secundaria])
    assert report.sources_used == ("primaria", "secundaria")
    u = SP500Universe(history_file)
    assert u.is_member("PRIM", dt.date(2025, 3, 3))
    assert not u.is_member("SEC", dt.date(2025, 3, 3))


def test_universe_refresh_delegates_and_invalidates_cache(history_file: Path) -> None:
    """`SP500Universe.refresh` ve el fichero actualizado sin reiniciar el proceso."""
    base = _synthetic_members()
    u = SP500Universe(history_file)
    assert u.last_snapshot == dt.date(2025, 2, 3)
    src = _FakeHistorySource(
        [("2025-02-03", base), ("2025-03-03", base)], name="fake"
    )
    report = u.refresh([src])
    assert report.n_appended == 1
    assert u.last_snapshot == dt.date(2025, 3, 3)


def test_universe_refresh_rejects_unknown_source_name(history_file: Path) -> None:
    """`refresh(["fuente-inventada"])` falla antes de tocar la red."""
    u = SP500Universe(history_file)
    with pytest.raises(UniverseError):
        u.refresh(["fuente-inventada"])


# ================================================================ red real


@pytest.mark.network
@requires_network
def test_github_history_source_live() -> None:
    """`GithubDatasetSource` descarga la serie histórica real desde GitHub."""
    src = GithubDatasetSource()
    assert src.available() is True
    frame = src.fetch_history()
    assert {"date", "tickers"}.issubset(frame.columns)
    assert len(frame) > 2000
    assert frame["date"].min() <= pd.Timestamp("1996-01-02")
    assert frame["date"].is_monotonic_increasing
    snap = src.fetch_snapshot()
    assert 480 <= snap.n_members <= 520
    assert "AAPL" in snap.tickers


@pytest.mark.network
@requires_network
def test_github_constituents_source_live() -> None:
    """`GithubConstituentsSource` descarga la lista vigente con CIK y GICS."""
    snap = GithubConstituentsSource().fetch_snapshot()
    assert 480 <= snap.n_members <= 520
    by_ticker = {r.ticker: r for r in snap.records}
    assert by_ticker["AAPL"].cik == "0000320193"
    assert by_ticker["AAPL"].sector == "Information Technology"


@pytest.mark.network
@requires_network
def test_wikipedia_source_live() -> None:
    """Wikipedia (bloqueada por el proxy de desarrollo; requiere red abierta)."""
    snap = WikipediaSource().fetch_snapshot()
    assert 480 <= snap.n_members <= 520
    assert "AAPL" in snap.tickers


@pytest.mark.network
@requires_network
def test_slickcharts_source_live() -> None:
    """Slickcharts (bloqueada por el proxy de desarrollo; requiere red abierta)."""
    src = SlickchartsSource()
    snap = src.fetch_snapshot()
    assert 480 <= snap.n_members <= 520


@pytest.mark.network
@requires_network
def test_blocked_sources_report_provider_unavailable() -> None:
    """En un entorno con egress restringido el fallo es `ProviderUnavailable`."""
    with pytest.raises(ProviderUnavailable):
        WikipediaSource(url="https://en.wikipedia.org/wiki/No_Existe_404_XYZ").fetch_snapshot()
