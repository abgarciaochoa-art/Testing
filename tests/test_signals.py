"""Tests del módulo `signals`: transformaciones cross-section y combinación.

Corren **sin red**: todo el material realista sale de
`earnings_alpha.data.synthetic.SyntheticMarket`, y el resto son paneles
construidos a mano para poder verificar aritmética exacta.

Los cuatro contratos que estos tests defienden, en orden de importancia:

1. **Invariancia del rango a transformaciones monótonas.** Si `rank_pct` no fuese
   invariante, la rank IC —métrica principal del repo— dependería del
   preprocesado y toda la validación estadística sería arbitraria.
2. **`neutralize` deja exposición ~0.** A precisión de máquina, no "pequeña":
   `fundamental_factors.md` §15.2 fija el listón en 1e-9 y `lstsq` da 1e-15.
3. **La ponderación por IC no usa el futuro.** Test explícito y con dos filos:
   se comprueba que perturbar el futuro no mueve el pasado *y* que el corte está
   justo donde debe (perturbar la última observación admisible sí lo mueve). Un
   test que solo comprobase lo primero pasaría también con pesos constantes.
4. **Robustez con secciones cruzadas pobres.** Fechas de uno o dos nombres
   válidos, fechas constantes, grupos diminutos: NaN explícito o
   `InsufficientHistory`, nunca un número calculado sobre cuatro observaciones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from earnings_alpha.data.synthetic import SyntheticMarket
from earnings_alpha.errors import DataQualityError, InsufficientHistory
from earnings_alpha.pit import get_calendar
from earnings_alpha.signals import (
    align_panel_like,
    check_panel,
    combine,
    condition_factor,
    cross_section_count,
    cross_section_ic,
    demean,
    demedian,
    expanding_ic_weights,
    fixed_weight_combine,
    ic_weighted_combine,
    ic_weighted_weights,
    neutralize,
    observability_dates,
    orthogonalize,
    rank_pct,
    residualize,
    symmetric_orthogonalize,
    winsorize,
    zscore,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FWD_HORIZON = 5


@pytest.fixture(scope="module")
def market() -> SyntheticMarket:
    """Mercado sintético con sección cruzada suficiente para IC (>= 30 nombres)."""
    return SyntheticMarket(seed=4242, n_tickers=60, start="2021-01-04", end="2023-12-29")


@pytest.fixture(scope="module")
def panel(market: SyntheticMarket) -> dict[str, object]:
    """Señales de precio, retornos forward y exposiciones (sector, tamaño, beta).

    Las series se construyen sobre el propio panel ``(date, ticker)`` con
    `groupby(level="ticker")`, sin pasar por formato ancho: así el desplazamiento
    temporal ocurre dentro de cada símbolo y no puede mezclar tickers.
    """
    prices = market.prices().sort_index()
    close = prices["adj_close"]
    by_ticker = close.groupby(level="ticker")
    daily = by_ticker.pct_change()

    signals = pd.DataFrame(
        {
            "momentum": close / by_ticker.shift(21) - 1.0,
            "reversal": -(close / by_ticker.shift(5) - 1.0),
            "lowvol": -daily.groupby(level="ticker").transform(
                lambda s: s.rolling(60).std()
            ),
        }
    )
    forward = (by_ticker.shift(-FWD_HORIZON) / close - 1.0).rename("fwd")

    sectors = market.sectors()
    metadata = market.metadata()
    tickers = signals.index.get_level_values(1)
    exposures = pd.DataFrame(
        {
            "sector": sectors.reindex(tickers).to_numpy(),
            "size": np.log(prices["market_cap"].reindex(signals.index).to_numpy()),
            "beta": metadata["beta_mkt"].reindex(tickers).to_numpy(),
        },
        index=signals.index,
    )
    return {
        "signals": signals,
        "forward": forward,
        "exposures": exposures,
        "sectors": sectors,
        "prices": prices,
    }


@pytest.fixture
def toy() -> pd.Series:
    """Panel diminuto y controlado: 3 fechas x 6 nombres, con casos límite.

    - 2021-01-04: sección cruzada completa y estrictamente creciente.
    - 2021-01-05: solo un valor válido (sección cruzada insuficiente).
    - 2021-01-06: sección cruzada constante (varianza cero).
    """
    dates = pd.to_datetime(["2021-01-04", "2021-01-05", "2021-01-06"])
    index = pd.MultiIndex.from_product([dates, list("abcdef")], names=["date", "ticker"])
    values = np.arange(18.0)
    values[6:11] = np.nan
    values[12:18] = 5.0
    return pd.Series(values, index=index, name="factor")


# ---------------------------------------------------------------------------
# Validación de panel
# ---------------------------------------------------------------------------


def test_check_panel_rejects_inverted_levels() -> None:
    """(ticker, date) en vez de (date, ticker) debe fallar de forma ruidosa."""
    idx = pd.MultiIndex.from_tuples(
        [("AAPL", pd.Timestamp("2021-01-04")), ("MSFT", pd.Timestamp("2021-01-04"))],
        names=["ticker", "date"],
    )
    with pytest.raises(DataQualityError, match="primer nivel"):
        check_panel(pd.Series([1.0, 2.0], index=idx))


def test_check_panel_rejects_duplicates_and_flat_index() -> None:
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2021-01-04"), "AAPL")] * 2, names=["date", "ticker"]
    )
    with pytest.raises(DataQualityError, match="duplicados"):
        check_panel(pd.Series([1.0, 2.0], index=idx))
    with pytest.raises(DataQualityError, match="MultiIndex"):
        check_panel(pd.Series([1.0], index=pd.Index(["AAPL"])))


def test_empty_panel_raises_instead_of_returning_empty(toy: pd.Series) -> None:
    """Regla 5 del proyecto: nada de devolver un panel vacío en silencio."""
    with pytest.raises(InsufficientHistory):
        zscore(toy.iloc[:0])


def test_align_panel_like_broadcasts_static_maps(toy: pd.Series) -> None:
    static = pd.Series(["x", "x", "x", "y", "y", "y"], index=list("abcdef"))
    aligned = align_panel_like(static, toy.index)
    assert isinstance(aligned, pd.Series)
    assert aligned.index.equals(toy.index)
    assert aligned.loc[(pd.Timestamp("2021-01-06"), "d")] == "y"


# ---------------------------------------------------------------------------
# zscore
# ---------------------------------------------------------------------------


def test_zscore_standardizes_each_date_independently(panel: dict[str, object]) -> None:
    signals = panel["signals"]
    assert isinstance(signals, pd.DataFrame)
    z = zscore(signals["momentum"])
    by_date = z.groupby(level=0)
    assert by_date.mean().abs().max() < 1e-10
    std = by_date.std().dropna()
    assert np.allclose(std.to_numpy(), 1.0)


def test_zscore_is_immune_to_a_date_specific_level_shift(toy: pd.Series) -> None:
    """Una estandarización global dejaría rastro del desplazamiento; la de fecha no.

    Es la comprobación operativa de `pit_and_biases.md` §12.7: normalizar con los
    momentos de todo el panel usa la distribución futura de la señal.
    """
    base = toy.dropna()
    shifted = base.copy()
    mask = shifted.index.get_level_values(0) == pd.Timestamp("2021-01-04")
    shifted[mask] = shifted[mask] + 1_000.0
    pd.testing.assert_series_equal(zscore(base, min_obs=2), zscore(shifted, min_obs=2))

    # Y una escala distinta por fecha tampoco deja rastro.
    scaled = base.copy()
    scaled[mask] = scaled[mask] * 50.0
    pd.testing.assert_series_equal(zscore(base, min_obs=2), zscore(scaled, min_obs=2))


def test_zscore_robust_resists_a_single_outlier() -> None:
    dates = pd.to_datetime(["2021-01-04"])
    index = pd.MultiIndex.from_product([dates, [f"t{i}" for i in range(20)]], names=["date", "ticker"])
    values = np.arange(20.0)
    contaminated = values.copy()
    contaminated[-1] = 10_000.0
    clean = pd.Series(values, index=index)
    dirty = pd.Series(contaminated, index=index)

    classic_shift = float(
        (zscore(dirty, min_obs=5).iloc[:19] - zscore(clean, min_obs=5).iloc[:19]).abs().mean()
    )
    robust_shift = float(
        (
            zscore(dirty, robust=True, min_obs=5).iloc[:19]
            - zscore(clean, robust=True, min_obs=5).iloc[:19]
        )
        .abs()
        .mean()
    )
    assert robust_shift < 0.05
    assert classic_shift > 10 * robust_shift


def test_zscore_constant_cross_section(toy: pd.Series) -> None:
    constant_day = pd.Timestamp("2021-01-06")
    default = zscore(toy, min_obs=3)
    assert default.loc[constant_day].isna().all()
    as_zero = zscore(toy, min_obs=3, constant="zero")
    assert (as_zero.loc[constant_day] == 0.0).all()


def test_zscore_clip_bounds() -> None:
    dates = pd.to_datetime(["2021-01-04"])
    index = pd.MultiIndex.from_product([dates, [f"t{i}" for i in range(40)]], names=["date", "ticker"])
    values = np.concatenate([np.zeros(39), [500.0]])
    out = zscore(pd.Series(values, index=index), clip=3.0, min_obs=5)
    assert out.max() <= 3.0 + 1e-12
    assert out.min() >= -3.0 - 1e-12


# ---------------------------------------------------------------------------
# winsorize
# ---------------------------------------------------------------------------


def test_winsorize_clips_to_the_quantiles_and_preserves_order() -> None:
    dates = pd.to_datetime(["2021-01-04"])
    index = pd.MultiIndex.from_product([dates, [f"t{i}" for i in range(100)]], names=["date", "ticker"])
    values = np.arange(100.0)
    s = pd.Series(values, index=index)
    out = winsorize(s, 0.1)
    lo, hi = np.quantile(values, 0.1), np.quantile(values, 0.9)
    assert np.isclose(out.min(), lo)
    assert np.isclose(out.max(), hi)
    # Monótona no decreciente: conserva el orden débil. Los valores recortados
    # quedan empatados en el límite (por eso existe `mode="drop"`), pero el
    # interior de la sección cruzada mantiene exactamente su rango.
    diffs = np.diff(out.to_numpy())
    assert (diffs >= -1e-12).all()
    interior = slice(10, 90)
    np.testing.assert_allclose(
        rank_pct(s).to_numpy()[interior], rank_pct(out).to_numpy()[interior]
    )


def test_winsorize_drop_mode_nulls_the_tails() -> None:
    dates = pd.to_datetime(["2021-01-04"])
    index = pd.MultiIndex.from_product([dates, [f"t{i}" for i in range(100)]], names=["date", "ticker"])
    s = pd.Series(np.arange(100.0), index=index)
    out = winsorize(s, 0.05, mode="drop")
    assert out.isna().sum() > 0
    assert out.max() < 95.0


def test_winsorize_mad_is_more_stable_than_quantiles_on_small_sections() -> None:
    dates = pd.to_datetime(["2021-01-04"])
    index = pd.MultiIndex.from_product([dates, list("abcdefgh")], names=["date", "ticker"])
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 900.0])
    s = pd.Series(values, index=index)
    mad = winsorize(s, method="mad", k=3.0, min_obs=4)
    assert mad.max() < 100.0
    assert np.isclose(mad.iloc[0], 1.0)


# ---------------------------------------------------------------------------
# rank_pct — invariancia monótona (contrato clave)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "func"),
    [
        ("afin", lambda v: 3.7 * v + 12.0),
        ("exp", np.exp),
        ("cubo", lambda v: v**3),
        ("logistica", lambda v: 1.0 / (1.0 + np.exp(-v))),
        ("arcsinh", np.arcsinh),
    ],
)
def test_rank_pct_invariant_to_increasing_monotone_transforms(
    panel: dict[str, object], label: str, func: object
) -> None:
    """`rank_pct(f(x)) == rank_pct(x)` para toda `f` monótona creciente.

    Es la propiedad que sostiene la elección de la rank IC como métrica principal
    (`validation_methodology.md` §2.1): la métrica no puede cambiar porque se
    winsorice al 1 % o al 2.5 %.
    """
    signals = panel["signals"]
    assert isinstance(signals, pd.DataFrame)
    x = signals["reversal"].dropna()
    transformed = pd.Series(func(x.to_numpy()), index=x.index, name=x.name)  # type: ignore[operator]
    assert np.isfinite(transformed.to_numpy()).all(), f"{label} produjo valores no finitos"
    pd.testing.assert_series_equal(rank_pct(x), rank_pct(transformed))


def test_rank_pct_reverses_under_decreasing_transform(panel: dict[str, object]) -> None:
    signals = panel["signals"]
    assert isinstance(signals, pd.DataFrame)
    x = signals["momentum"].dropna()
    forward = rank_pct(x)
    backward = rank_pct(-x)
    assert np.allclose(forward.to_numpy() + backward.to_numpy(), 1.0)


def test_rank_pct_modes_and_bounds(panel: dict[str, object]) -> None:
    signals = panel["signals"]
    assert isinstance(signals, pd.DataFrame)
    x = signals["momentum"].dropna()
    uniform = rank_pct(x)
    assert uniform.min() > 0.0
    assert uniform.max() < 1.0
    centered = rank_pct(x, mode="centered")
    assert np.allclose(centered.to_numpy(), 2.0 * uniform.to_numpy() - 1.0)
    normal = rank_pct(x, mode="normal")
    assert abs(float(normal.groupby(level=0).mean().abs().max())) < 1e-9
    assert np.isfinite(normal.to_numpy()).all()


def test_rank_pct_ties_average(toy: pd.Series) -> None:
    constant_day = pd.Timestamp("2021-01-06")
    out = rank_pct(toy, min_obs=2)
    assert np.allclose(out.loc[constant_day].to_numpy(), 0.5)


# ---------------------------------------------------------------------------
# Política de NaN
# ---------------------------------------------------------------------------


def test_nan_policy_propagate_is_the_default(toy: pd.Series) -> None:
    out = zscore(toy, min_obs=2)
    assert out.isna().to_numpy()[6:11].all()


def test_nan_policy_raise_and_drop(toy: pd.Series) -> None:
    with pytest.raises(DataQualityError, match="ausentes"):
        zscore(toy, min_obs=2, nan_policy="raise")
    dropped = zscore(toy, min_obs=2, nan_policy="drop")
    assert len(dropped) == len(toy) - 5
    assert not dropped.index.isin([(pd.Timestamp("2021-01-05"), "a")]).any()


def test_nan_policy_fill_only_touches_missing_inputs_of_live_dates() -> None:
    """El relleno neutro no puede fabricar una sección cruzada donde no la hay."""
    dates = pd.to_datetime(["2021-01-04", "2021-01-05"])
    index = pd.MultiIndex.from_product([dates, list("abcdef")], names=["date", "ticker"])
    values = np.array([1.0, 2.0, 3.0, 4.0, np.nan, 6.0, 1.0, np.nan, np.nan, np.nan, np.nan, np.nan])
    s = pd.Series(values, index=index)

    neutral = zscore(s, min_obs=3, nan_policy="neutral")
    assert neutral.loc[(pd.Timestamp("2021-01-04"), "e")] == 0.0
    # 2021-01-05 no llega a min_obs: se anula entera, incluidos los huecos.
    assert neutral.loc[pd.Timestamp("2021-01-05")].isna().all()

    zeros = rank_pct(s, min_obs=3, nan_policy="zero")
    assert zeros.loc[(pd.Timestamp("2021-01-04"), "e")] == 0.0


def test_nan_policy_mean_imputation_compresses_dispersion() -> None:
    dates = pd.to_datetime(["2021-01-04"])
    index = pd.MultiIndex.from_product([dates, list("abcdefgh")], names=["date", "ticker"])
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.nan, np.nan])
    s = pd.Series(values, index=index)
    imputed = zscore(s, min_obs=3, nan_policy="mean")
    assert imputed.notna().all()
    # los imputados caen exactamente en el centro
    assert np.allclose(imputed.to_numpy()[6:], 0.0)
    # y la dispersión de los observados se comprime respecto a propagar
    propagated = zscore(s, min_obs=3)
    assert imputed.iloc[:6].abs().max() > propagated.iloc[:6].abs().max()


def test_unknown_nan_policy_raises(toy: pd.Series) -> None:
    with pytest.raises(DataQualityError, match="nan_policy desconocida"):
        zscore(toy, nan_policy="rellenar_con_cero_y_rezar")


# ---------------------------------------------------------------------------
# Secciones cruzadas pobres
# ---------------------------------------------------------------------------


def test_min_obs_nulls_thin_dates_and_keeps_the_rest(toy: pd.Series) -> None:
    out = zscore(toy, min_obs=3)
    assert out.loc[pd.Timestamp("2021-01-05")].isna().all()
    assert out.loc[pd.Timestamp("2021-01-04")].notna().all()
    assert float(cross_section_count(toy).loc[pd.Timestamp("2021-01-05")]) == 1.0


def test_all_dates_below_minimum_raise_insufficient_history(toy: pd.Series) -> None:
    with pytest.raises(InsufficientHistory, match="ninguna sección cruzada"):
        zscore(toy, min_obs=50)


def test_on_insufficient_raise_reports_the_offending_dates(toy: pd.Series) -> None:
    with pytest.raises(InsufficientHistory, match="2021-01-05"):
        zscore(toy, min_obs=3, on_insufficient="raise")


def test_transforms_survive_a_single_valid_name(toy: pd.Series) -> None:
    """Con min_obs=1 la fecha de un solo nombre no revienta: sale NaN o 0, no un error."""
    out = zscore(toy, min_obs=1)
    assert out.loc[(pd.Timestamp("2021-01-05"), "f")] != out.loc[(pd.Timestamp("2021-01-05"), "f")]
    ranked = rank_pct(toy, min_obs=1)
    assert ranked.loc[(pd.Timestamp("2021-01-05"), "f")] == 0.5


# ---------------------------------------------------------------------------
# demean / demedian por grupo
# ---------------------------------------------------------------------------


def test_demedian_zeroes_the_group_median(panel: dict[str, object]) -> None:
    signals = panel["signals"]
    sectors = panel["sectors"]
    assert isinstance(signals, pd.DataFrame)
    out = demedian(signals["momentum"], sectors, min_group=5)
    frame = pd.DataFrame({"v": out, "sector": align_panel_like(sectors, out.index)}).dropna()
    medians = frame.groupby([frame.index.get_level_values(0), "sector"])["v"].median()
    assert medians.abs().max() < 1e-12


def test_demean_zeroes_the_group_mean(panel: dict[str, object]) -> None:
    signals = panel["signals"]
    sectors = panel["sectors"]
    assert isinstance(signals, pd.DataFrame)
    out = demean(signals["momentum"], sectors, min_group=5)
    frame = pd.DataFrame({"v": out, "sector": align_panel_like(sectors, out.index)}).dropna()
    means = frame.groupby([frame.index.get_level_values(0), "sector"])["v"].mean()
    assert means.abs().max() < 1e-10


def test_small_group_policies(toy: pd.Series) -> None:
    groups = pd.Series(["x", "x", "x", "x", "x", "y"], index=list("abcdef"))
    nulled = demedian(toy, groups, min_group=3, min_obs=1)
    assert np.isnan(nulled.loc[(pd.Timestamp("2021-01-04"), "f")])
    pooled = demedian(toy, groups, min_group=3, small_group="pool", min_obs=1)
    assert not np.isnan(pooled.loc[(pd.Timestamp("2021-01-04"), "f")])
    with pytest.raises(InsufficientHistory, match="grupos con menos"):
        demedian(toy, groups, min_group=3, small_group="raise", min_obs=1)


def test_missing_group_label_yields_nan(toy: pd.Series) -> None:
    groups = pd.Series(["x", "x", "x", None, "y", "y"], index=list("abcdef"))
    out = demean(toy, groups, min_obs=1, min_group=1)
    assert np.isnan(out.loc[(pd.Timestamp("2021-01-04"), "d")])
    with pytest.raises(DataQualityError, match="sin etiqueta de grupo"):
        demean(toy, groups, min_obs=1, min_group=1, missing_group="raise")


# ---------------------------------------------------------------------------
# neutralize / residualize (contrato clave)
# ---------------------------------------------------------------------------


def test_neutralize_leaves_zero_exposure_to_sector_and_size(panel: dict[str, object]) -> None:
    """Tras neutralizar, media por sector ~0 y correlación con log(MC) ~0.

    `fundamental_factors.md` §15.2 exige < 1e-9; con `lstsq` se obtiene ~1e-15.
    """
    signals, exposures = panel["signals"], panel["exposures"]
    assert isinstance(signals, pd.DataFrame)
    assert isinstance(exposures, pd.DataFrame)
    raw = zscore(signals["momentum"])
    resid = neutralize(raw, exposures, by=["sector", "size"])
    assert isinstance(resid, pd.Series)

    frame = pd.DataFrame(
        {"r": resid, "sector": exposures["sector"], "size": exposures["size"]}
    ).dropna()
    assert len(frame) > 10_000

    sector_means = frame.groupby([frame.index.get_level_values(0), "sector"])["r"].mean()
    assert sector_means.abs().max() < 1e-9

    corr = frame.groupby(level=0).apply(lambda g: g["r"].corr(g["size"]), include_groups=False)
    assert corr.abs().max() < 1e-9

    # La exposición cruda SÍ estaba ahí: si no, el test no probaría nada.
    raw_frame = pd.DataFrame(
        {"r": raw, "sector": exposures["sector"], "size": exposures["size"]}
    ).dropna()
    raw_sector = raw_frame.groupby([raw_frame.index.get_level_values(0), "sector"])["r"].mean()
    assert raw_sector.abs().max() > 0.1


def test_neutralize_with_weights_zeroes_the_weighted_exposure(panel: dict[str, object]) -> None:
    signals, exposures, prices = panel["signals"], panel["exposures"], panel["prices"]
    assert isinstance(signals, pd.DataFrame)
    assert isinstance(exposures, pd.DataFrame)
    assert isinstance(prices, pd.DataFrame)
    weights = np.sqrt(prices["market_cap"].reindex(signals.index))
    resid = neutralize(zscore(signals["momentum"]), exposures, by=["sector"], weights=weights)
    frame = pd.DataFrame({"r": resid, "w": weights, "sector": exposures["sector"]}).dropna()
    weighted = frame.groupby([frame.index.get_level_values(0), "sector"]).apply(
        lambda g: float(np.average(g["r"], weights=g["w"])), include_groups=False
    )
    assert weighted.abs().max() < 1e-9


def test_neutralize_refuses_dates_without_degrees_of_freedom() -> None:
    """Con tantos nombres como columnas el residuo sería 0 exacto: eso es NaN, no señal."""
    dates = pd.to_datetime(["2021-01-04"])
    tickers = list("abcdef")
    index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    x = pd.Series(np.arange(6.0), index=index)
    # Diseño saturado: un sector por nombre, luego las dummies generan todo R^6.
    exposures = pd.DataFrame({"sector": [f"s{i}" for i in range(6)]}, index=index)

    strict = neutralize(x, exposures, min_obs=4, min_dof=1)
    assert isinstance(strict, pd.Series)
    assert strict.isna().all()

    lax = neutralize(x, exposures, min_obs=4, min_dof=0)
    assert isinstance(lax, pd.Series)
    assert lax.notna().all()
    # Ajuste perfecto: el residuo es cero exacto. Sin `min_dof` esto pasaría por
    # "factor perfectamente neutralizado" cuando lo que queda no es señal alguna.
    assert lax.abs().max() < 1e-9


def test_neutralize_requires_exposures(toy: pd.Series) -> None:
    with pytest.raises(DataQualityError, match="necesita exposiciones"):
        neutralize(toy)


def test_neutralize_missing_exposure_policies(panel: dict[str, object]) -> None:
    signals, exposures = panel["signals"], panel["exposures"]
    assert isinstance(signals, pd.DataFrame)
    assert isinstance(exposures, pd.DataFrame)
    broken = exposures.copy()
    broken.iloc[0, broken.columns.get_loc("size")] = np.nan
    dropped = neutralize(zscore(signals["momentum"]), broken, by=["sector", "size"])
    assert isinstance(dropped, pd.Series)
    assert np.isnan(dropped.iloc[0])
    with pytest.raises(DataQualityError, match="exposición completa"):
        neutralize(
            zscore(signals["momentum"]), broken, by=["sector", "size"], exposure_nan="raise"
        )


def test_residualize_matches_a_manual_least_squares(panel: dict[str, object]) -> None:
    signals, exposures = panel["signals"], panel["exposures"]
    assert isinstance(signals, pd.DataFrame)
    assert isinstance(exposures, pd.DataFrame)
    y = zscore(signals["momentum"])
    x = exposures[["size"]]
    resid = residualize(y, x)

    day = y.index.get_level_values(0).unique()[400]
    sub = pd.DataFrame({"y": y, "size": x["size"]}).loc[day].dropna()
    design = np.column_stack([np.ones(len(sub)), sub["size"].to_numpy()])
    beta, *_ = np.linalg.lstsq(design, sub["y"].to_numpy(), rcond=None)
    expected = sub["y"].to_numpy() - design @ beta
    got = resid.loc[day].reindex(sub.index).to_numpy()
    assert np.allclose(got, expected, atol=1e-10)


def test_residualize_rejects_categorical_regressors(panel: dict[str, object]) -> None:
    signals, exposures = panel["signals"], panel["exposures"]
    assert isinstance(signals, pd.DataFrame)
    assert isinstance(exposures, pd.DataFrame)
    with pytest.raises(DataQualityError, match="no numéricas"):
        residualize(signals["momentum"], exposures[["sector"]])


# ---------------------------------------------------------------------------
# condition_factor y aplicación a DataFrame
# ---------------------------------------------------------------------------


def test_condition_factor_pipeline_is_standardized_and_neutral(panel: dict[str, object]) -> None:
    signals, exposures = panel["signals"], panel["exposures"]
    assert isinstance(signals, pd.DataFrame)
    assert isinstance(exposures, pd.DataFrame)
    out = condition_factor(signals["momentum"], exposures, by=["sector", "size"])
    assert isinstance(out, pd.Series)
    by_date = out.groupby(level=0)
    assert by_date.mean().abs().max() < 1e-10
    assert np.allclose(by_date.std().dropna().to_numpy(), 1.0)
    frame = pd.DataFrame({"r": out, "sector": exposures["sector"]}).dropna()
    means = frame.groupby([frame.index.get_level_values(0), "sector"])["r"].mean()
    assert means.abs().max() < 1e-9


def test_transforms_apply_columnwise_to_frames(panel: dict[str, object]) -> None:
    signals = panel["signals"]
    assert isinstance(signals, pd.DataFrame)
    out = zscore(signals)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == list(signals.columns)
    pd.testing.assert_series_equal(out["lowvol"], zscore(signals["lowvol"]), check_names=False)


def test_transforms_are_truncation_invariant(panel: dict[str, object]) -> None:
    """Recalcular con los datos truncados en T reproduce la columna T del panel completo.

    Es el test mecánico de no look-ahead de `fundamental_factors.md` §15.2
    aplicado al acondicionamiento: si alguna transformación usara un estadístico
    global (media de toda la muestra, cuantil del panel entero), truncar cambiaría
    el resultado del día T.
    """
    signals, exposures = panel["signals"], panel["exposures"]
    assert isinstance(signals, pd.DataFrame)
    assert isinstance(exposures, pd.DataFrame)
    dates = signals.index.get_level_values(0)
    cutoff = pd.Timestamp("2022-06-30")
    mask = dates <= cutoff

    full = condition_factor(signals["momentum"], exposures, by=["sector", "size"])
    truncated = condition_factor(
        signals["momentum"][mask], exposures[mask], by=["sector", "size"]
    )
    assert isinstance(full, pd.Series)
    assert isinstance(truncated, pd.Series)
    pd.testing.assert_series_equal(full[mask], truncated)


# ---------------------------------------------------------------------------
# Coeficiente de información
# ---------------------------------------------------------------------------


def test_cross_section_ic_matches_scipy_spearman(panel: dict[str, object]) -> None:
    scipy_stats = pytest.importorskip("scipy.stats")
    signals, forward = panel["signals"], panel["forward"]
    assert isinstance(signals, pd.DataFrame)
    assert isinstance(forward, pd.Series)
    ic = cross_section_ic(signals[["momentum"]], forward)

    day = ic["momentum"].dropna().index[300]
    sub = pd.DataFrame({"s": signals["momentum"], "r": forward}).loc[day].dropna()
    expected = scipy_stats.spearmanr(sub["s"].to_numpy(), sub["r"].to_numpy()).statistic
    assert np.isclose(ic.loc[day, "momentum"], expected, atol=1e-12)


def test_cross_section_ic_pearson_matches_numpy(panel: dict[str, object]) -> None:
    signals, forward = panel["signals"], panel["forward"]
    assert isinstance(signals, pd.DataFrame)
    assert isinstance(forward, pd.Series)
    ic = cross_section_ic(signals[["lowvol"]], forward, method="pearson")
    day = ic["lowvol"].dropna().index[200]
    sub = pd.DataFrame({"s": signals["lowvol"], "r": forward}).loc[day].dropna()
    expected = np.corrcoef(sub["s"].to_numpy(), sub["r"].to_numpy())[0, 1]
    assert np.isclose(ic.loc[day, "lowvol"], expected, atol=1e-12)


def test_cross_section_ic_respects_the_minimum_cross_section(panel: dict[str, object]) -> None:
    signals, forward = panel["signals"], panel["forward"]
    assert isinstance(signals, pd.DataFrame)
    assert isinstance(forward, pd.Series)
    with pytest.raises(InsufficientHistory, match="mínimo de sección cruzada"):
        cross_section_ic(signals[["momentum"]], forward, min_obs=500)


def test_cross_section_ic_keeps_the_calendar(panel: dict[str, object]) -> None:
    signals, forward = panel["signals"], panel["forward"]
    assert isinstance(signals, pd.DataFrame)
    assert isinstance(forward, pd.Series)
    ic = cross_section_ic(signals, forward)
    dates = pd.DatetimeIndex(signals.index.get_level_values(0).unique()).sort_values()
    assert ic.index.equals(dates)
    assert list(ic.columns) == list(signals.columns)


# ---------------------------------------------------------------------------
# Ponderación por IC — NO look-ahead (contrato clave)
# ---------------------------------------------------------------------------


def _synthetic_ic(n: int = 90, k: int = 3, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n)
    return pd.DataFrame(
        rng.uniform(0.01, 0.09, size=(n, k)), index=dates, columns=["a", "b", "c"][:k]
    )


def test_ic_weights_no_lookahead() -> None:
    """El peso de la fecha `t` no puede depender de ningún IC posterior a `t - h`.

    Doble filo, y hace falta que sea doble: la primera mitad comprueba que
    reescribir el futuro no mueve el pasado; la segunda, que el corte está
    exactamente donde debe estar. Sin la segunda, unos pesos constantes pasarían
    el test con matrícula de honor.
    """
    horizon = 5
    ic = _synthetic_ic()
    base = expanding_ic_weights(ic, horizon=horizon, min_periods=5, warmup="nan")

    target = 40  # posición de la fecha de decisión que auditamos
    rng = np.random.default_rng(99)

    # (a) todo lo que no es utilizable en `target` puede cambiar sin efecto alguno.
    future = ic.copy()
    first_unusable = target - horizon
    future.iloc[first_unusable:] = rng.uniform(0.01, 0.09, size=future.iloc[first_unusable:].shape)
    perturbed = expanding_ic_weights(future, horizon=horizon, min_periods=5, warmup="nan")
    np.testing.assert_array_equal(
        base.iloc[: target + 1].to_numpy(), perturbed.iloc[: target + 1].to_numpy()
    )

    # (b) la última observación utilizable SÍ influye: el corte no es conservador
    #     de más, está donde dice estar.
    tight = ic.copy()
    tight.iloc[first_unusable - 1] = tight.iloc[first_unusable - 1] * np.array([5.0, 0.2, 1.0])
    moved = expanding_ic_weights(tight, horizon=horizon, min_periods=5, warmup="nan")
    assert not np.allclose(base.iloc[target].to_numpy(), moved.iloc[target].to_numpy())
    # y no afecta a la fecha anterior, que aún no la tenía disponible
    np.testing.assert_array_equal(
        base.iloc[: target - 1].to_numpy(), moved.iloc[: target - 1].to_numpy()
    )


def test_ic_weights_horizon_zero_still_excludes_the_same_day() -> None:
    """Ni con horizonte 0 se usa el IC del propio día de decisión."""
    ic = _synthetic_ic(n=40, seed=5)
    weights = expanding_ic_weights(ic, horizon=0, min_periods=3, warmup="nan")
    altered = ic.copy()
    altered.iloc[10] = altered.iloc[10] * 4.0
    moved = expanding_ic_weights(altered, horizon=0, min_periods=3, warmup="nan")
    np.testing.assert_array_equal(weights.iloc[:11].to_numpy(), moved.iloc[:11].to_numpy())
    assert not np.allclose(weights.iloc[11].to_numpy(), moved.iloc[11].to_numpy())


def test_ic_weights_extra_lag_widens_the_exclusion() -> None:
    ic = _synthetic_ic(n=60, seed=3)
    strict = expanding_ic_weights(ic, horizon=1, min_periods=3, extra_lag_days=10, warmup="nan")
    loose = expanding_ic_weights(ic, horizon=1, min_periods=3, extra_lag_days=0, warmup="nan")
    assert strict.notna().any(axis=1).sum() < loose.notna().any(axis=1).sum()


def test_ic_weighted_pipeline_is_truncation_invariant(panel: dict[str, object]) -> None:
    """Extremo a extremo: recortar la muestra en T no cambia ningún peso hasta T."""
    signals, forward = panel["signals"], panel["forward"]
    assert isinstance(signals, pd.DataFrame)
    assert isinstance(forward, pd.Series)
    cutoff = pd.Timestamp("2022-09-30")
    mask = signals.index.get_level_values(0) <= cutoff

    full = ic_weighted_weights(signals, forward, horizon=FWD_HORIZON, min_periods=20)
    truncated = ic_weighted_weights(
        signals[mask], forward[mask], horizon=FWD_HORIZON, min_periods=20
    )
    common = truncated.index
    np.testing.assert_allclose(
        full.reindex(common).to_numpy(), truncated.to_numpy(), rtol=0, atol=0
    )


def test_ic_weights_warmup_policies() -> None:
    ic = _synthetic_ic(n=40, seed=8)
    equal = expanding_ic_weights(ic, horizon=2, min_periods=10, warmup="equal")
    assert np.allclose(equal.iloc[0].to_numpy(), 1.0 / ic.shape[1])
    nan_warmup = expanding_ic_weights(ic, horizon=2, min_periods=10, warmup="nan")
    assert nan_warmup.iloc[0].isna().all()
    with pytest.raises(InsufficientHistory, match="sin IC observable"):
        expanding_ic_weights(ic, horizon=2, min_periods=10, warmup="raise")


def test_ic_weights_only_use_signals_with_enough_history() -> None:
    ic = _synthetic_ic(n=60, k=2, seed=13)
    ic.iloc[:50, 1] = np.nan  # la segunda señal aparece tarde
    weights = expanding_ic_weights(ic, horizon=1, min_periods=10, warmup="nan")
    assert weights.iloc[30]["b"] == 0.0
    assert weights.iloc[30]["a"] > 0.0


def test_ic_weighting_prefers_the_informative_signal(panel: dict[str, object]) -> None:
    """Una señal con IC real debe acabar pesando más que ruido puro.

    El "oráculo" se construye a partir del retorno futuro *a propósito*: aquí no
    se valida una estrategia, se valida que el mecanismo de ponderación detecta
    IC cuando lo hay.
    """
    forward = panel["forward"]
    assert isinstance(forward, pd.Series)
    rng = np.random.default_rng(2024)
    oracle = (forward + rng.normal(0.0, forward.std() * 3.0, len(forward))).rename("oracle")
    noise = pd.Series(rng.normal(size=len(forward)), index=forward.index, name="noise")
    signals = pd.DataFrame({"oracle": oracle, "noise": noise}).dropna()
    aligned = forward.reindex(signals.index)

    weights = ic_weighted_weights(signals, aligned, horizon=FWD_HORIZON, min_periods=30)
    last = weights.iloc[-1]
    assert last["oracle"] > 0.75
    assert last["oracle"] > 3 * last["noise"]


def test_ic_weighted_combine_produces_a_standardized_score(panel: dict[str, object]) -> None:
    signals, forward = panel["signals"], panel["forward"]
    assert isinstance(signals, pd.DataFrame)
    assert isinstance(forward, pd.Series)
    score = ic_weighted_combine(signals, forward, horizon=FWD_HORIZON, min_periods=20)
    by_date = score.groupby(level=0)
    assert by_date.mean().abs().max() < 1e-10
    assert np.allclose(by_date.std().dropna().to_numpy(), 1.0)
    assert score.index.equals(signals.index)


def test_ic_cov_weights_discount_redundancy() -> None:
    """Con dos señales casi idénticas y una independiente, `ic_cov` reparte mejor.

    Grinold-Kahn: ``w ∝ Σ⁻¹·IC``. Si dos columnas son la misma cosa, la
    ponderación ingenua por IC les da peso doble; la que usa Σ no.
    """
    rng = np.random.default_rng(77)
    dates = pd.bdate_range("2021-01-04", periods=120)
    tickers = [f"t{i:02d}" for i in range(40)]
    index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    base = rng.normal(size=len(index))
    twin = base + rng.normal(0.0, 0.05, size=len(index))
    independent = rng.normal(size=len(index))
    signals = pd.DataFrame(
        {"a": base, "a_twin": twin, "b": independent}, index=index
    )
    forward = pd.Series(
        0.3 * base + 0.3 * independent + rng.normal(0.0, 1.0, size=len(index)),
        index=index,
        name="fwd",
    )

    naive = ic_weighted_weights(signals, forward, horizon=1, min_periods=20, method="ic")
    cov = ic_weighted_weights(
        signals, forward, horizon=1, min_periods=20, method="ic_cov", corr_shrinkage=0.05
    )
    pair_naive = float(naive.iloc[-1][["a", "a_twin"]].sum())
    pair_cov = float(cov.iloc[-1][["a", "a_twin"]].sum())
    assert pair_cov < pair_naive
    assert cov.iloc[-1]["b"] > naive.iloc[-1]["b"]


def test_observability_dates_positional_and_with_calendar() -> None:
    dates = pd.DatetimeIndex(pd.bdate_range("2021-01-04", periods=10))
    positional = observability_dates(dates, 3)
    assert positional[0] == dates[3]
    assert pd.isna(positional[-1])

    cal = get_calendar()
    with_cal = observability_dates(dates, 3, cal)
    assert with_cal[0] == pd.Timestamp(cal.shift(dates[0].date(), 3))
    assert not pd.isna(with_cal[-1])  # el calendario sí sabe qué hay después
    with pytest.raises(ValueError, match="horizon"):
        observability_dates(dates, -1)


# ---------------------------------------------------------------------------
# Pesos fijos
# ---------------------------------------------------------------------------


def test_fixed_weight_combine_respects_declared_weights(panel: dict[str, object]) -> None:
    signals = panel["signals"]
    assert isinstance(signals, pd.DataFrame)
    only_momentum = fixed_weight_combine(
        signals, {"momentum": 1.0, "reversal": 0.0, "lowvol": 0.0}, standardize_output=False
    )
    reference = zscore(signals["momentum"], min_obs=20)
    assert isinstance(reference, pd.Series)
    common = only_momentum.dropna().index.intersection(reference.dropna().index)
    np.testing.assert_allclose(
        only_momentum.loc[common].to_numpy(), reference.loc[common].to_numpy(), atol=1e-12
    )


def test_fixed_weight_combine_missing_policies() -> None:
    dates = pd.to_datetime(["2021-01-04"])
    tickers = [f"t{i:02d}" for i in range(40)]
    index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    rng = np.random.default_rng(1)
    frame = pd.DataFrame(
        {"a": rng.normal(size=40), "b": rng.normal(size=40)}, index=index
    )
    frame.iloc[0, 1] = np.nan

    renorm = fixed_weight_combine(frame, standardize_output=False, missing="renormalize")
    skipped = fixed_weight_combine(frame, standardize_output=False, missing="skip")
    strict = fixed_weight_combine(frame, standardize_output=False, missing="nan")

    assert np.isclose(renorm.iloc[0], 2.0 * skipped.iloc[0])
    assert np.isnan(strict.iloc[0])
    assert renorm.iloc[1:].notna().all()


def test_fixed_weight_combine_rejects_bad_weights(panel: dict[str, object]) -> None:
    signals = panel["signals"]
    assert isinstance(signals, pd.DataFrame)
    with pytest.raises(DataQualityError, match="faltan pesos"):
        fixed_weight_combine(signals, {"momentum": 1.0})
    with pytest.raises(DataQualityError, match="inexistentes"):
        fixed_weight_combine(
            signals,
            {"momentum": 1.0, "reversal": 1.0, "lowvol": 1.0, "fantasma": 1.0},
        )
    with pytest.raises(DataQualityError, match="todos los pesos son cero"):
        fixed_weight_combine(signals, {"momentum": 0.0, "reversal": 0.0, "lowvol": 0.0})


# ---------------------------------------------------------------------------
# Ortogonalización
# ---------------------------------------------------------------------------


def test_gram_schmidt_removes_cross_sectional_correlation(panel: dict[str, object]) -> None:
    signals = panel["signals"]
    assert isinstance(signals, pd.DataFrame)
    order = ["momentum", "reversal", "lowvol"]
    ortho = orthogonalize(signals, order=order)
    clean = ortho.dropna()
    assert len(clean) > 10_000

    correlations = clean.groupby(level=0).corr()
    for i, first in enumerate(order):
        for second in order[i + 1 :]:
            pair = correlations.xs(first, level=1)[second]
            assert pair.abs().max() < 1e-9, f"{first} vs {second}"

    # Antes de ortogonalizar la correlación era material: si no, nada se probaría.
    raw_corr = signals.dropna().groupby(level=0).corr()
    assert raw_corr.xs("momentum", level=1)["reversal"].abs().max() > 0.1


def test_gram_schmidt_preserves_the_first_factor(panel: dict[str, object]) -> None:
    signals = panel["signals"]
    assert isinstance(signals, pd.DataFrame)
    ortho = orthogonalize(signals, order=["reversal", "momentum", "lowvol"])
    reference = zscore(signals["reversal"], min_obs=20)
    assert isinstance(reference, pd.Series)
    common = ortho["reversal"].dropna().index.intersection(reference.dropna().index)
    np.testing.assert_allclose(
        ortho["reversal"].loc[common].to_numpy(), reference.loc[common].to_numpy(), atol=1e-12
    )


def test_gram_schmidt_order_matters(panel: dict[str, object]) -> None:
    signals = panel["signals"]
    assert isinstance(signals, pd.DataFrame)
    a = orthogonalize(signals, order=["momentum", "reversal", "lowvol"])
    b = orthogonalize(signals, order=["reversal", "momentum", "lowvol"])
    common = a.dropna().index.intersection(b.dropna().index)
    assert not np.allclose(
        a["momentum"].loc[common].to_numpy(), b["momentum"].loc[common].to_numpy()
    )


def test_orthogonalize_rejects_unknown_order(panel: dict[str, object]) -> None:
    signals = panel["signals"]
    assert isinstance(signals, pd.DataFrame)
    with pytest.raises(DataQualityError, match="inexistentes"):
        orthogonalize(signals, order=["momentum", "no_existe"])
    with pytest.raises(DataQualityError, match="repite"):
        orthogonalize(signals, order=["momentum", "momentum"])


def test_symmetric_orthogonalization_yields_identity_correlation(
    panel: dict[str, object],
) -> None:
    signals = panel["signals"]
    assert isinstance(signals, pd.DataFrame)
    ortho = symmetric_orthogonalize(signals.dropna())
    clean = ortho.dropna()
    correlations = clean.groupby(level=0).corr()
    for i, first in enumerate(ortho.columns):
        for second in list(ortho.columns)[i + 1 :]:
            assert correlations.xs(first, level=1)[second].abs().max() < 1e-9


def test_symmetric_orthogonalization_needs_complete_cross_sections() -> None:
    dates = pd.to_datetime(["2021-01-04"])
    index = pd.MultiIndex.from_product([dates, list("abcd")], names=["date", "ticker"])
    frame = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 1.0, 4.0, 3.0]}, index=index
    )
    with pytest.raises(InsufficientHistory, match="ortogonalización simétrica"):
        symmetric_orthogonalize(frame, min_obs=20)


# ---------------------------------------------------------------------------
# Despachador
# ---------------------------------------------------------------------------


def test_combine_dispatcher_matches_the_explicit_functions(panel: dict[str, object]) -> None:
    signals, forward = panel["signals"], panel["forward"]
    assert isinstance(signals, pd.DataFrame)
    assert isinstance(forward, pd.Series)

    pd.testing.assert_series_equal(
        combine(signals, "weights"), fixed_weight_combine(signals), check_names=False
    )
    pd.testing.assert_series_equal(
        combine(signals, "ic_weighted", forward_returns=forward, horizon=FWD_HORIZON),
        ic_weighted_combine(signals, forward, horizon=FWD_HORIZON),
        check_names=False,
    )
    ortho_score = combine(signals, "orthogonalize", order=["momentum", "reversal", "lowvol"])
    assert isinstance(ortho_score, pd.Series)
    assert ortho_score.notna().sum() > 10_000


def test_combine_rejects_unknown_method_and_missing_returns(panel: dict[str, object]) -> None:
    signals = panel["signals"]
    assert isinstance(signals, pd.DataFrame)
    with pytest.raises(ValueError, match="desconocido"):
        combine(signals, "magia")  # type: ignore[arg-type]
    with pytest.raises(DataQualityError, match="forward_returns"):
        combine(signals, "ic_weighted")


def test_combination_is_robust_to_sparse_dates() -> None:
    """Fechas con uno o dos nombres válidos no rompen la combinación: salen NaN."""
    dates = pd.bdate_range("2021-01-04", periods=8)
    tickers = [f"t{i:02d}" for i in range(30)]
    index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    rng = np.random.default_rng(19)
    frame = pd.DataFrame(
        {"a": rng.normal(size=len(index)), "b": rng.normal(size=len(index))}, index=index
    )
    thin = dates[3]
    thin_rows = frame.index.get_level_values(0) == thin
    frame.loc[thin_rows, :] = np.nan
    frame.loc[(thin, "t00"), :] = [1.0, 2.0]

    score = fixed_weight_combine(frame, min_obs=10)
    assert score.loc[thin].isna().all()
    assert score.loc[dates[0]].notna().all()
