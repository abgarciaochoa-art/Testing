"""Tests del núcleo del estudio de eventos (`earnings_alpha.events`).

Cinco familias de propiedades, todas sin red y contra `SyntheticMarket`:

1. **Tiempo-evento.** `event_windows` produce exactamente `pre+post+1` sesiones por
   evento, `tau = 0` es la sesión negociable (AMC -> sesión siguiente, sin
   look-ahead) y el solapamiento entre ventanas consecutivas del mismo emisor se
   expone en vez de ocultarse.
2. **Higiene point-in-time del estudio.** La ventana de estimación termina
   estrictamente antes de la de evento, excluye las sesiones vecinas de otros
   anuncios del mismo emisor y los eventos sin histórico fallan de forma explícita.
3. **Recuperación de la verdad-terreno.** El AR de `tau = 0` recupera la respuesta
   inyectada por unidad de sorpresa (`event_response`), el CAR post-evento recupera
   el PEAD, el CAAR del quintil alto de SUE supera al bajo con significancia y las
   betas de ff3 recuperan las betas verdaderas del generador.
4. **Contrastes.** Patell, BMP y Corrado con fórmulas verificadas a mano en un
   micro-caso determinista, significativos en el día del evento y silenciosos en
   una ventana placebo pre-evento.
5. **Gap overnight.** La partición gap + intradía reproduce exactamente el retorno
   cierre-a-cierre y recupera el `event_gap_share = 0.80` del generador.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from earnings_alpha.data.synthetic import SyntheticMarket
from earnings_alpha.errors import (
    CalendarError,
    ConfigError,
    DataQualityError,
    InsufficientHistory,
)
from earnings_alpha.events import (
    aar_caar,
    abnormal_returns,
    bmp_test,
    caar_by_group,
    car_window,
    corrado_rank_test,
    estimate_rho_bar,
    event_windows,
    normalize_events,
    overnight_decomposition,
    patell_test,
    quantile_groups,
    window_overlap_summary,
)
from earnings_alpha.pit import get_calendar, tradable_date
from earnings_alpha.types import Session

# ------------------------------------------------------------------- fixtures

MKT = {"n_tickers": 28, "start": "2019-01-02", "end": "2023-12-29"}

#: Verdad-terreno del generador usada en los tests de recuperación.
EVENT_RESPONSE = 0.019
PEAD_TOTAL = 0.025
EVENT_GAP_SHARE = 0.80


@pytest.fixture(scope="module")
def mkt() -> SyntheticMarket:
    """Mercado sintético amplio; una sola instancia para todo el módulo."""
    return SyntheticMarket(seed=1234, leak_fraction=0.12, **MKT)


@pytest.fixture(scope="module")
def events(mkt: SyntheticMarket) -> pd.DataFrame:
    return mkt.events()


@pytest.fixture(scope="module")
def prices(mkt: SyntheticMarket) -> pd.DataFrame:
    return mkt.prices()


@pytest.fixture(scope="module")
def market_ret(mkt: SyntheticMarket) -> pd.Series:
    return mkt.market_index()["log_return"]


@pytest.fixture(scope="module")
def ar(
    prices: pd.DataFrame, events: pd.DataFrame, market_ret: pd.Series
) -> pd.DataFrame:
    """Panel AR con modelo de mercado, ventana [-30, +60] y filas de estimación."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return abnormal_returns(
            prices,
            events,
            model="market",
            market=market_ret,
            pre=30,
            post=60,
            include_estimation=True,
        )


@pytest.fixture(scope="module")
def truth(mkt: SyntheticMarket) -> pd.DataFrame:
    return mkt.ground_truth()


# ------------------------------------------------------------ event_windows


def test_event_windows_shape_and_tau(events: pd.DataFrame, mkt: SyntheticMarket) -> None:
    win = event_windows(events, mkt.calendar, pre=30, post=60)
    assert len(win) == len(events) * 91
    assert list(win.columns) == ["event_id", "ticker", "event_date", "tau", "date", "n_shared"]
    per_event = win.groupby("event_id")["tau"]
    assert (per_event.count() == 91).all()
    assert (per_event.min() == -30).all()
    assert (per_event.max() == 60).all()
    # tau = 0 es exactamente la sesión negociable.
    at_zero = win[win["tau"] == 0]
    assert (at_zero["date"] == at_zero["event_date"]).all()
    # Sin identificadores duplicados por (event_id, tau).
    assert not win.duplicated(["event_id", "tau"]).any()


def test_event_windows_dates_follow_the_session_grid(
    events: pd.DataFrame, mkt: SyntheticMarket
) -> None:
    """Cada `tau` es un desplazamiento en sesiones, verificado contra el calendario."""
    cal = mkt.calendar
    win = event_windows(events.head(5), cal, pre=30, post=60)
    for row in win[win["tau"].isin([-30, -1, 0, 1, 60])].itertuples(index=False):
        expected = (
            pd.Timestamp(row.event_date).date()
            if row.tau == 0
            else cal.shift(pd.Timestamp(row.event_date).date(), int(row.tau))
        )
        assert pd.Timestamp(row.date).date() == expected


def test_event_windows_ids_stable_and_deterministic(
    events: pd.DataFrame, mkt: SyntheticMarket
) -> None:
    win1 = event_windows(events, mkt.calendar, pre=5, post=5)
    win2 = event_windows(events, mkt.calendar, pre=5, post=5)
    pd.testing.assert_frame_equal(win1, win2)
    # El event_id respeta el de la tabla de entrada (formato de types.EarningsEvent).
    assert set(win1["event_id"]) == set(events["event_id"])


def test_event_windows_overlap_exposed_not_hidden(
    events: pd.DataFrame, mkt: SyntheticMarket
) -> None:
    """Con la ventana por defecto y eventos trimestrales, el solapamiento es la norma."""
    win = event_windows(events, mkt.calendar, pre=30, post=60)
    assert (win["n_shared"] >= 1).all()
    assert (win["n_shared"] >= 2).any(), "91 sesiones de ventana con eventos cada ~63 deben solapar"
    # Con una ventana corta el solapamiento desaparece por completo.
    narrow = event_windows(events, mkt.calendar, pre=2, post=2)
    assert (narrow["n_shared"] == 1).all()


def test_window_overlap_summary(events: pd.DataFrame, mkt: SyntheticMarket) -> None:
    summ = window_overlap_summary(events, mkt.calendar, pre=30, post=60)
    assert summ.index.name == "event_id"
    assert len(summ) == len(events)
    med = float(summ["sessions_to_prev"].dropna().astype(float).median())
    assert 55 <= med <= 70, f"eventos trimestrales deben distar ~63 sesiones; mediana {med}"
    # Solapamiento coherente con la aritmética pre+post+1-d.
    both = summ.dropna(subset=["sessions_to_prev"])
    expected = np.maximum(91 - both["sessions_to_prev"].astype(int), 0)
    assert (both["overlap_sessions_prev"].to_numpy() == expected.to_numpy()).all()
    assert summ["overlaps_any"].all()
    # El primer evento de cada ticker no tiene predecesor.
    firsts = summ.groupby("ticker")["sessions_to_prev"].apply(lambda s: s.isna().sum())
    assert (firsts == 1).all()
    # Ventana corta: nada solapa.
    narrow = window_overlap_summary(events, mkt.calendar, pre=2, post=2)
    assert not narrow["overlaps_any"].any()


def test_event_windows_from_announced_at_respects_tradable_date() -> None:
    """Sin `event_date`, la fecha sale de pit.tradable_dates: AMC -> sesión siguiente."""
    cal = get_calendar()
    frame = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            # 2021-07-16 es viernes. BMO 12:00 UTC = 08:00 ET; AMC 20:30 UTC = 16:30 ET.
            "announced_at": [
                pd.Timestamp("2021-07-16 12:00:00"),
                pd.Timestamp("2021-07-16 20:30:00"),
            ],
            "session": ["bmo", "amc"],
        }
    )
    win = event_windows(frame, cal, pre=1, post=1)
    zero = win[win["tau"] == 0].set_index("ticker")["date"]
    assert zero["AAA"] == pd.Timestamp("2021-07-16")  # BMO: la misma sesión
    assert zero["BBB"] == pd.Timestamp("2021-07-19")  # AMC: el lunes siguiente
    # event_id de respaldo ticker:fecha cuando no hay fiscal_quarter.
    assert set(win["event_id"]) == {"AAA:2021-07-16", "BBB:2021-07-19"}


def test_event_windows_errors(mkt: SyntheticMarket) -> None:
    cal = mkt.calendar
    with pytest.raises(DataQualityError):
        event_windows(pd.DataFrame(columns=["ticker", "event_date"]), cal)
    dup = pd.DataFrame(
        {"ticker": ["AAA", "AAA"], "event_date": ["2021-07-16", "2021-07-16"]}
    )
    with pytest.raises(DataQualityError):
        event_windows(dup, cal)
    weekend = pd.DataFrame({"ticker": ["AAA"], "event_date": ["2021-07-17"]})  # sábado
    with pytest.raises(DataQualityError):
        event_windows(weekend, cal)
    ok = pd.DataFrame({"ticker": ["AAA"], "event_date": ["2021-07-16"]})
    with pytest.raises(ValueError, match="no negativos"):
        event_windows(ok, cal, pre=-1)
    # Ventana que se sale del calendario: error explícito, nunca truncado.
    edge = pd.DataFrame({"ticker": ["AAA"], "event_date": [cal.first_session]})
    with pytest.raises(CalendarError):
        event_windows(edge, cal, pre=5, post=5)


def test_normalize_events_requires_dates_or_announcements() -> None:
    with pytest.raises(DataQualityError, match="announced_at"):
        normalize_events(pd.DataFrame({"ticker": ["AAA"]}), get_calendar())


# --------------------------------------------------------- abnormal_returns


def test_abnormal_returns_shape_and_internal_consistency(ar: pd.DataFrame) -> None:
    assert ar.attrs["model"] == "market"
    assert ar.attrs["estimation"] == (-250, -40)
    assert ar.attrs["n_events"] > 300
    ev = ar[ar["is_event_window"]]
    per_event = ev.groupby("event_id")["tau"]
    assert (per_event.count() == 91).all()
    assert (per_event.min() == -30).all()
    assert (per_event.max() == 60).all()
    assert np.isfinite(ev[["ret", "expected_ret", "ar", "sar", "car", "scar", "bhar"]]).all().all()
    # AR = ret - esperado; CAR = suma acumulada del AR dentro del evento.
    assert np.allclose(ev["ar"], ev["ret"] - ev["expected_ret"])
    one = ev[ev["event_id"] == ev["event_id"].iloc[0]].sort_values("tau")
    assert np.allclose(one["car"].to_numpy(), one["ar"].cumsum().to_numpy())
    k = np.arange(1, len(one) + 1)
    assert np.allclose(one["scar"].to_numpy(), one["sar"].cumsum().to_numpy() / np.sqrt(k))
    # Las columnas por evento son constantes dentro del evento.
    for col in ("n_estimation", "dof", "resid_std", "alpha", "beta_mkt"):
        assert (ev.groupby("event_id")[col].nunique() == 1).all()


def test_estimation_window_never_touches_event_window(
    ar: pd.DataFrame, prices: pd.DataFrame, events: pd.DataFrame, market_ret: pd.Series
) -> None:
    est = ar[~ar["is_event_window"]]
    ev = ar[ar["is_event_window"]]
    assert len(est) > 0
    # En tiempo-evento: la estimación acaba en -40 < -30 = inicio de la ventana.
    assert int(est["tau"].max()) <= -40
    # En tiempo de calendario, evento a evento.
    est_max = est.groupby("event_id")["date"].max()
    ev_min = ev.groupby("event_id")["date"].min()
    common = est_max.index.intersection(ev_min.index)
    assert (est_max.loc[common] < ev_min.loc[common]).all()
    # Una estimación que pisa la ventana de evento se rechaza de plano.
    with pytest.raises(ConfigError, match="estrictamente anterior"):
        abnormal_returns(
            prices, events, model="market", market=market_ret, estimation=(-100, -5), pre=30
        )


def test_estimation_excludes_neighbor_event_days(
    ar: pd.DataFrame, prices: pd.DataFrame
) -> None:
    """Las sesiones a <=2 de otro anuncio del mismo emisor no entran en la estimación."""
    dates_index = prices.index.get_level_values("date").unique().sort_values()
    est = ar[~ar["is_event_window"]]
    ticker = est["ticker"].iloc[0]
    event_dates = ar.loc[ar["ticker"] == ticker, "event_date"].unique()
    event_pos = np.searchsorted(dates_index.values, np.sort(event_dates))
    sub = est[est["ticker"] == ticker]
    est_pos = np.searchsorted(dates_index.values, sub["date"].to_numpy("datetime64[ns]"))
    dist = np.abs(est_pos[:, None] - event_pos[None, :]).min(axis=1)
    assert (dist > 2).all()
    # Y el recuento de filas de estimación coincide con n_estimation.
    counts = est.groupby("event_id")["ar"].count()
    declared = est.drop_duplicates("event_id").set_index("event_id")["n_estimation"]
    assert (counts == declared.loc[counts.index]).all()


def test_recovers_injected_event_response(ar: pd.DataFrame, truth: pd.DataFrame) -> None:
    """El AR de tau=0 recupera el `event_response` = 0.019 por unidad de sorpresa."""
    ar0 = ar[ar["is_event_window"] & (ar["tau"] == 0)].set_index("event_id")
    z = truth["surprise_z"].reindex(ar0.index).clip(-4, 4)
    slope, intercept = np.polyfit(z.to_numpy(), ar0["ar"].to_numpy(), 1)
    resid = ar0["ar"].to_numpy() - (slope * z.to_numpy() + intercept)
    se = np.sqrt(resid.var(ddof=2) / (len(z) * z.to_numpy().var()))
    assert slope / se > 4.0
    assert 0.6 * EVENT_RESPONSE < slope < 1.5 * EVENT_RESPONSE


def test_recovers_injected_pead_drift(ar: pd.DataFrame, truth: pd.DataFrame) -> None:
    """El CAR de (+1, +60) recupera el drift PEAD total = 0.025 por unidad de sorpresa."""
    car_post = car_window(ar, (1, 60))
    z = truth["surprise_z"].reindex(car_post.index).clip(-4, 4)
    slope, intercept = np.polyfit(z.to_numpy(), car_post.to_numpy(), 1)
    resid = car_post.to_numpy() - (slope * z.to_numpy() + intercept)
    se = np.sqrt(resid.var(ddof=2) / (len(z) * z.to_numpy().var()))
    assert slope / se > 2.0
    assert 0.3 * PEAD_TOTAL < slope < 2.0 * PEAD_TOTAL


def test_aar_caar_profile_and_errors(ar: pd.DataFrame) -> None:
    agg = aar_caar(ar)
    assert agg.index.min() == -30 and agg.index.max() == 60
    assert (agg["n"] > 300).all()
    # El día del evento domina el perfil AAR y es positivo (sorpresa media > 0).
    assert agg["aar"].idxmax() == 0
    assert agg.loc[0, "aar"] > 0
    assert agg.loc[0, "aar_t"] > 1.5
    assert (agg["aar_se"] > 0).all() and (agg["caar_se"] > 0).all()
    # El deflactor de Kolari-Pynnönen ensancha los errores estándar.
    agg_kp = aar_caar(ar, rho_bar=0.05)
    assert (agg_kp["aar_se"] > agg["aar_se"]).all()
    assert abs(agg_kp.loc[0, "aar_t"]) < abs(agg.loc[0, "aar_t"])
    with pytest.raises(ConfigError):
        aar_caar(ar, rho_bar=1.5)


def test_caar_high_sue_quintile_beats_low(ar: pd.DataFrame, events: pd.DataFrame) -> None:
    """El gráfico CAAR por quintil de SUE: Q5 > Q1 con significancia."""
    car_full = car_window(ar, (0, 60))
    sue = events.set_index("event_id")["sue"].reindex(car_full.index)
    groups = quantile_groups(sue.dropna(), q=5)
    by_group = caar_by_group(ar, groups)
    assert by_group.index.names == ["group", "tau"]
    assert by_group.index.get_level_values("group").nunique() == 5
    caar60 = by_group.xs(60, level="tau")["caar"]
    assert caar60["Q5"] > caar60["Q1"]
    # Significancia del spread con t de Welch sobre los CAR por evento.
    q5 = car_full.reindex(groups[groups == "Q5"].index).dropna()
    q1 = car_full.reindex(groups[groups == "Q1"].index).dropna()
    welch = sps.ttest_ind(q5, q1, equal_var=False)
    assert q5.mean() - q1.mean() > 0.03
    assert welch.statistic > 2.5
    with pytest.raises(DataQualityError, match="grupos con menos"):
        caar_by_group(ar, groups.iloc[:3])


def test_quantile_groups_balance_and_errors(events: pd.DataFrame) -> None:
    sue = events.set_index("event_id")["sue"]
    labels = quantile_groups(sue, q=5)
    sizes = labels.value_counts()
    assert len(sizes) == 5
    assert sizes.max() - sizes.min() <= 1
    assert len(labels) == sue.notna().sum()  # los NaN quedan fuera
    with pytest.raises(ConfigError):
        quantile_groups(sue, q=1)
    with pytest.raises(ConfigError):
        quantile_groups(sue, q=5, labels=["a", "b"])
    with pytest.raises(InsufficientHistory):
        quantile_groups(pd.Series([1.0, 2.0, np.nan]), q=5)


# ------------------------------------------------------------- contrastes


def test_patell_and_bmp_micro_formulas() -> None:
    """Verificación a mano de Patell (1976) y BMP (1991) en un caso determinista."""
    frame = pd.DataFrame(
        {
            "event_id": ["e1", "e2", "e3"],
            "ticker": ["A", "B", "C"],
            "tau": [0, 0, 0],
            "is_event_window": [True, True, True],
            "ar": [0.01, 0.02, 0.03],
            "sar": [1.0, 2.0, 3.0],
            "car": [0.01, 0.02, 0.03],
            "dof": [100, 100, 100],
            "n_estimation": [102, 102, 102],
        }
    )
    pt = patell_test(frame, (0, 0))
    # Z = sum(SAR) / sqrt(N * dof/(dof-2)) = 6 / sqrt(3 * 100/98)
    assert pt.statistic == pytest.approx(6.0 / np.sqrt(3.0 * 100.0 / 98.0), rel=1e-12)
    assert pt.mean_car == pytest.approx(0.02)
    bt = bmp_test(frame, (0, 0))
    # t = mean(SCAR) * sqrt(N) / sd(SCAR) = 2 * sqrt(3) / 1
    assert bt.statistic == pytest.approx(2.0 * np.sqrt(3.0), rel=1e-12)
    assert bt.n_events == 3


def test_tests_significant_on_event_day(ar: pd.DataFrame) -> None:
    """Patell, BMP y Corrado detectan la reacción inyectada en (0, +1)."""
    pt = patell_test(ar, (0, 1))
    bt = bmp_test(ar, (0, 1))
    ct = corrado_rank_test(ar, (0, 1))
    assert pt.statistic > 5.0 and pt.p_value < 1e-6
    assert bt.statistic > 2.0 and bt.p_value < 0.05
    assert ct.statistic > 2.5 and ct.p_value < 0.02
    # Patell (ignora la varianza inducida por el evento) debe superar a BMP.
    assert pt.statistic > bt.statistic
    for result in (pt, bt, ct):
        assert result.mean_car > 0
        assert result.significant
        assert result.to_dict()["window"] == (0, 1)


def test_placebo_window_is_quiet(ar: pd.DataFrame) -> None:
    """En (-28, -22), fuera del alcance de cualquier filtración (<=15 sesiones), nada suena."""
    for test in (patell_test, bmp_test, corrado_rank_test):
        result = test(ar, (-28, -22))
        assert abs(result.statistic) < 3.0, f"{result.method} = {result.statistic:.2f} en placebo"


def test_bmp_kolari_pynnonen_shrinks_t(ar: pd.DataFrame) -> None:
    plain = bmp_test(ar, (0, 1))
    rho = estimate_rho_bar(ar)
    assert np.isfinite(rho) and 0.0 <= rho < 0.5
    adjusted = bmp_test(ar, (0, 1), rho_bar=rho)
    assert abs(adjusted.statistic) < abs(plain.statistic)
    assert adjusted.method == "bmp_kolari_pynnonen"
    assert adjusted.rho_bar == rho


def test_corrado_and_rho_bar_require_estimation_rows(ar: pd.DataFrame) -> None:
    event_only = ar[ar["is_event_window"]]
    with pytest.raises(DataQualityError, match="include_estimation"):
        corrado_rank_test(event_only, (0, 1))
    with pytest.raises(DataQualityError, match="include_estimation"):
        estimate_rho_bar(event_only)


def test_car_window_values_and_errors(ar: pd.DataFrame) -> None:
    ev = ar[ar["is_event_window"]]
    eid = ev["event_id"].iloc[0]
    manual = ev.loc[(ev["event_id"] == eid) & ev["tau"].between(0, 3), "ar"].sum()
    assert car_window(ar, (0, 3)).loc[eid] == pytest.approx(manual)
    with pytest.raises(ConfigError, match="excede"):
        car_window(ar, (0, 400))
    with pytest.raises(ConfigError, match="invertida"):
        car_window(ar, (5, 0))


# ------------------------------------------------------------ gap overnight


def test_overnight_gap_identity_and_share(
    prices: pd.DataFrame, events: pd.DataFrame
) -> None:
    gap = overnight_decomposition(prices, events)
    assert gap.index.name == "event_id"
    assert len(gap) == len(events)
    # Identidad exacta en logs: gap + intradía = retorno cierre-a-cierre del panel.
    log_ret = prices["log_return"].unstack("ticker")
    reconstructed = gap["gap_return"] + gap["intraday_return"]
    assert np.allclose(reconstructed, gap["event_day_return"], atol=1e-12)
    panel_ret = pd.Series(
        [log_ret.loc[r.event_date, r.ticker] for r in gap.itertuples(index=False)],
        index=gap.index,
    )
    assert np.allclose(gap["event_day_return"], panel_ret, atol=1e-9)
    # El generador realiza el 80 % del retorno del día del evento en el gap.
    slope = np.polyfit(gap["event_day_return"], gap["gap_return"], 1)[0]
    assert 0.70 < slope < 0.90
    # Ambos tramos se exponen por separado y ninguno es degenerado.
    assert gap["gap_return"].std() > 0 and gap["intraday_return"].std() > 0


def test_overnight_gap_amc_uses_next_session(
    prices: pd.DataFrame, events: pd.DataFrame, mkt: SyntheticMarket
) -> None:
    """Para AMC la sesión del evento es la siguiente al anuncio: el gap contiene la noticia."""
    gap = overnight_decomposition(prices, events)
    sample = events[events["session"] == "amc"].head(10)
    for row in sample.itertuples(index=False):
        expected = tradable_date(
            pd.Timestamp(row.announced_at).to_pydatetime(), Session.AMC, mkt.calendar
        )
        assert pd.Timestamp(gap.loc[row.event_id, "event_date"]).date() == expected
        # La víspera del gap es una sesión estrictamente anterior al event_date.
        assert gap.loc[row.event_id, "prev_close_date"] < gap.loc[row.event_id, "event_date"]


def test_overnight_gap_simple_kind_composition(
    prices: pd.DataFrame, events: pd.DataFrame
) -> None:
    gap = overnight_decomposition(prices, events, returns_kind="simple")
    composed = (1.0 + gap["gap_return"]) * (1.0 + gap["intraday_return"]) - 1.0
    assert np.allclose(composed, gap["event_day_return"], atol=1e-12)
    with pytest.raises(DataQualityError, match="open"):
        overnight_decomposition(prices[["close"]], events)


# ------------------------------------------------- modelos ff3, mean y errores


def test_ff3_recovers_true_betas(
    prices: pd.DataFrame, events: pd.DataFrame, mkt: SyntheticMarket
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ar3 = abnormal_returns(
            prices, events, model="ff3", factors=mkt.factor_returns(), pre=5, post=5
        )
    per_event = ar3.drop_duplicates("event_id")
    assert np.isfinite(per_event[["beta_mkt", "beta_smb", "beta_hml"]]).all().all()
    estimated = per_event.groupby("ticker")["beta_mkt"].mean()
    true_betas = mkt.metadata()["beta_mkt"].reindex(estimated.index)
    assert estimated.corr(true_betas) > 0.8


def test_mean_model_is_constant_expectation(
    prices: pd.DataFrame, events: pd.DataFrame
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        arm = abnormal_returns(prices, events, model="mean", pre=5, post=5)
    ev = arm[arm["is_event_window"]]
    assert (ev.groupby("event_id")["expected_ret"].nunique() == 1).all()
    assert ev["beta_mkt"].isna().all()


def test_input_validation_errors(
    prices: pd.DataFrame, events: pd.DataFrame, market_ret: pd.Series, mkt: SyntheticMarket
) -> None:
    with pytest.raises(ConfigError, match="modelo desconocido"):
        abnormal_returns(prices, events, model="capm")  # type: ignore[arg-type]
    with pytest.raises(DataQualityError, match="market"):
        abnormal_returns(prices, events, model="market")
    with pytest.raises(DataQualityError, match="factors"):
        abnormal_returns(prices, events, model="ff3")
    with pytest.raises(DataQualityError, match="faltan columnas"):
        abnormal_returns(
            prices, events, model="ff3", factors=mkt.factor_returns()[["mkt", "smb"]]
        )
    with pytest.raises(ConfigError, match="invertida"):
        abnormal_returns(prices, events, model="market", market=market_ret, estimation=(-40, -250))
    with pytest.raises(ConfigError, match="min_estimation_obs"):
        abnormal_returns(
            prices, events, model="market", market=market_ret, min_estimation_obs=5
        )
    with pytest.raises(ConfigError, match="no negativos"):
        abnormal_returns(prices, events, model="market", market=market_ret, pre=-1)
    with pytest.raises(DataQualityError, match="MultiIndex"):
        abnormal_returns(prices.reset_index(), events, model="market", market=market_ret)


def test_skipped_events_warn_and_are_recorded(
    prices: pd.DataFrame, events: pd.DataFrame, market_ret: pd.Series
) -> None:
    """Los eventos sin histórico se descartan con aviso y quedan contabilizados."""
    with pytest.warns(UserWarning, match="descartados"):
        result = abnormal_returns(
            prices, events, model="market", market=market_ret, pre=5, post=5
        )
    assert result.attrs["n_skipped"] > 0
    assert len(result.attrs["skipped"]) == result.attrs["n_skipped"]
    assert result.attrs["n_events"] + result.attrs["n_skipped"] == len(events)
    reasons = " ".join(result.attrs["skipped"].values())
    assert "estimación" in reasons
    # Ningún evento descartado aparece en el panel.
    assert not set(result.attrs["skipped"]) & set(result["event_id"])


def test_all_events_skipped_raises(
    prices: pd.DataFrame, events: pd.DataFrame, market_ret: pd.Series
) -> None:
    early = events[events["event_date"] < "2019-12-31"]
    assert len(early) > 0
    with pytest.raises(InsufficientHistory, match="ningún evento"):
        abnormal_returns(prices, early, model="market", market=market_ret)


def test_abnormal_returns_deterministic(
    prices: pd.DataFrame, events: pd.DataFrame, market_ret: pd.Series
) -> None:
    subset = events.iloc[200:260]
    kwargs = {"model": "market", "market": market_ret, "pre": 5, "post": 5}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        first = abnormal_returns(prices, subset, **kwargs)
        second = abnormal_returns(prices, subset, **kwargs)
    pd.testing.assert_frame_equal(first, second)
