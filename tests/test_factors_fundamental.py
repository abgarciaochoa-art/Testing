"""Tests de los factores contables (`factors.value/quality/growth/accruals`).

Cinco familias de propiedades, todas sin red (contrato §0.5):

1. **Fórmulas exactas con datos construidos a mano.** Cada uno de los 9
   componentes del F-Score de Piotroski verificado por separado (incluida la
   trampa del signo de EQ_OFFER), los accruals de Sloan por balance y por
   flujos, percent accruals, crecimiento sostenible, EV/EBITDA...
2. **Identidades contables.** ``CFO/NI = 1 − PACC`` con ``NI>0`` (§8.1 del
   informe), ``ACC_BS == ACC_CF`` con articulación perfecta y su discrepancia
   como bandera de calidad (Collins-Hribar 2002), y la columna `accruals` del
   generador sintético reproducida por la fórmula pura.
3. **Point-in-time.** El corte exacto de `asof_join` en la frontera de
   `filed_at`, la distinción nota de prensa vs 10-Q, la ausencia de relleno
   hacia atrás, el test mecánico de truncado de §15.2 del informe y la
   sensibilidad a mover un `filed_at` (si el factor no cambia, el as-of join
   está roto).
4. **Requisitos declarados y fallo explícito.** Columna requerida ausente →
   `DataQualityError`; opcional ausente → aviso y degradación documentada;
   historia insuficiente → `InsufficientHistory`, por nombre (NaN) y en el
   agregado (excepción); sectores estructuralmente excluidos → NaN, no cero.
5. **Forma canónica.** Las 26 variantes de factor devuelven un panel
   ``(date, ticker)`` canónico, determinista y con el `name` correcto.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from earnings_alpha.data.synthetic import SyntheticMarket
from earnings_alpha.errors import DataQualityError, InsufficientHistory
from earnings_alpha.factors.accruals import (
    NetOperatingAccruals,
    PercentAccruals,
    SloanAccrualsBalance,
    SloanAccrualsCashFlow,
    accruals_articulation_gap,
    percent_accruals,
    sloan_accruals_balance,
    sloan_accruals_cashflow,
)
from earnings_alpha.factors.growth import (
    AssetGrowth,
    EarningsGrowth,
    SalesAcceleration,
    SalesGrowth,
    SustainableGrowth,
    sales_acceleration,
    sales_growth_ttm,
    sustainable_growth,
)
from earnings_alpha.factors.quality import (
    PIOTROSKI_COMPONENTS,
    CFOToNetIncome,
    EarningsStability,
    GrossProfitability,
    Leverage,
    MarginTrend,
    PiotroskiFScore,
    Profitability,
    piotroski_components_annual,
    trend_t_stat,
)
from earnings_alpha.factors.value import (
    BookToMarket,
    EarningsYield,
    EbitdaToEV,
    FreeCashFlowYield,
    SalesYield,
    enterprise_value,
)
from earnings_alpha.pit import get_calendar
from earnings_alpha.signals import check_panel

# --------------------------------------------------------------------------- utilidades


class StubUniverse:
    """Universo mínimo conforme al protocolo del contrato §3.1 (lo que usan los factores)."""

    def __init__(self, sectors: dict[str, str]) -> None:
        self._sectors = dict(sectors)

    def members_on(self, d: object) -> list[str]:
        return sorted(self._sectors)

    def membership_panel(self, start: object, end: object) -> pd.DataFrame:
        raise NotImplementedError

    def cik_for(self, t: str, on: object | None = None) -> None:
        return None

    def sector_for(self, t: str) -> str | None:
        return self._sectors.get(t)


def make_ctx(
    *,
    fundamentals: pd.DataFrame,
    dates: pd.DatetimeIndex,
    prices: pd.DataFrame | None = None,
    sectors: dict[str, str] | None = None,
    events: pd.DataFrame | None = None,
) -> object:
    """Construye un FactorContext real si `factors.base` ya existe; si no, un duck-type.

    Coordinación con el agente paralelo de `factors/base.py`: los factores de
    esta tarea solo acceden a los atributos del contrato (§3.4), así que ambos
    caminos deben ser equivalentes.
    """
    kwargs = {
        "dates": pd.DatetimeIndex(dates),
        "universe": StubUniverse(sectors or {}),
        "prices": prices if prices is not None else pd.DataFrame(),
        "fundamentals": fundamentals,
        "estimates": pd.DataFrame(),
        "events": events if events is not None else pd.DataFrame(),
        "calendar": get_calendar(),
    }
    try:  # pragma: no cover - depende de si base.py ya está escrito
        from earnings_alpha.factors.base import FactorContext  # type: ignore[import-not-found]

        try:
            return FactorContext(**kwargs)
        except TypeError:
            pass
    except ImportError:
        pass
    return SimpleNamespace(**kwargs)


def make_fund(
    ticker: str,
    n_quarters: int,
    *,
    start_period: str = "2019-03-31",
    avail_lag_days: int = 20,
    filed_lag_days: int = 45,
    **columns: object,
) -> pd.DataFrame:
    """Historia trimestral de juguete con `available_at` (8-K) y `filed_at` (10-Q)."""
    period_ends = pd.date_range(start_period, periods=n_quarters, freq="QE")
    df = pd.DataFrame({"ticker": ticker, "period_end": period_ends})
    df["available_at"] = df["period_end"] + pd.Timedelta(days=avail_lag_days)
    df["filed_at"] = df["period_end"] + pd.Timedelta(days=filed_lag_days)
    for name, value in columns.items():
        df[name] = value
    return df


def make_prices(
    dates: pd.DatetimeIndex, market_caps: dict[str, float]
) -> pd.DataFrame:
    """Panel de precios mínimo: solo `market_cap`, constante por ticker."""
    idx = pd.MultiIndex.from_product(
        [pd.DatetimeIndex(dates), sorted(market_caps)], names=["date", "ticker"]
    )
    mc = [market_caps[t] for _, t in idx]
    return pd.DataFrame({"market_cap": mc}, index=idx)


CAL = get_calendar()

# Ocho trimestres 2019Q1..2020Q4; con filed_lag=45, el último filing es 2021-02-14.
MICRO_DATES = CAL.sessions(pd.Timestamp("2021-02-22"), pd.Timestamp("2021-03-05"))


def micro_value_ctx() -> object:
    """Dos tickers con cifras redondas para verificar los factores de valoración."""
    aaa = make_fund(
        "AAA", 8, net_income=25.0, revenue=250.0, cfo=30.0, capex=5.0,
        total_equity=500.0, ebitda=32.5, total_debt=400.0, cash=100.0,
    )
    bbb = make_fund(
        "BBB", 8, net_income=-225.0, revenue=125.0, cfo=10.0, capex=5.0,
        total_equity=-50.0, ebitda=32.5, total_debt=400.0, cash=5000.0,
    )
    fund = pd.concat([aaa, bbb], ignore_index=True)
    prices = make_prices(MICRO_DATES, {"AAA": 1000.0, "BBB": 1000.0})
    return make_ctx(fundamentals=fund, dates=MICRO_DATES, prices=prices)


@pytest.fixture(scope="module")
def market() -> SyntheticMarket:
    return SyntheticMarket(seed=11, n_tickers=14, start="2020-01-02", end="2022-12-30")


@pytest.fixture(scope="module")
def ctx(market: SyntheticMarket) -> object:
    return make_ctx(
        fundamentals=market.fundamentals(),
        dates=market.sessions[::5],
        prices=market.prices(),
        sectors=market.sectors().to_dict(),
        events=market.events(),
    )


# =========================================================================== #
# 1. Piotroski: los 9 componentes exactos, por separado
# =========================================================================== #

PIOTROSKI_BASE: dict[str, float] = {
    "net_income": 80.0,
    "cfo": 120.0,
    "revenue": 1000.0,
    "cost_of_goods": 600.0,
    "current_assets": 500.0,
    "current_liabilities": 250.0,
    "long_term_debt": 200.0,
    "equity_issuance": 0.0,
    "total_assets": 1100.0,
    "total_assets_prev": 1000.0,
    "total_assets_prev2": 900.0,
    "net_income_prev": 50.0,
    "revenue_prev": 850.0,
    "cost_of_goods_prev": 540.0,
    "current_assets_prev": 400.0,
    "current_liabilities_prev": 250.0,
    "long_term_debt_prev": 300.0,
}


def test_piotroski_base_scores_nine() -> None:
    comps = piotroski_components_annual(**PIOTROSKI_BASE)
    assert set(comps) == {*PIOTROSKI_COMPONENTS, "f_score"}
    assert len(PIOTROSKI_COMPONENTS) == 9
    for name in PIOTROSKI_COMPONENTS:
        assert comps[name] == 1.0, name
    assert comps["f_score"] == 9.0


@pytest.mark.parametrize(
    ("override", "expected_zeros"),
    [
        # ROA <= 0 pierde el punto 1 (ni_prev también negativo para aislar dROA).
        ({"net_income": -10.0, "net_income_prev": -50.0}, {"f_roa"}),
        # CFO negativo pierde el punto 2 y, por definición, también ACCRUAL.
        ({"cfo": -5.0}, {"f_cfo", "f_accrual"}),
        # ROA cae respecto al año previo: pierde solo dROA.
        ({"net_income": 50.0, "net_income_prev": 80.0}, {"f_droa"}),
        # CFO/TA (0.06) por debajo de ROA (0.08): pierde solo ACCRUAL.
        ({"cfo": 60.0}, {"f_accrual"}),
        # El apalancamiento sube: pierde dLEVER.
        ({"long_term_debt": 340.0}, {"f_dlever"}),
        # El ratio corriente cae: pierde dLIQUID.
        ({"current_liabilities": 350.0}, {"f_dliquid"}),
        # LA TRAMPA DEL SIGNO (§6.1): la empresa QUE EMITE pierde el punto.
        ({"equity_issuance": 25.0}, {"f_eq_offer"}),
        # El margen bruto cae: pierde dMARGIN.
        ({"cost_of_goods": 660.0}, {"f_dmargin"}),
        # La rotación cae (cogs ajustado para no arrastrar el margen): pierde dTURN.
        ({"revenue": 900.0, "cost_of_goods": 520.0}, {"f_dturn"}),
    ],
)
def test_piotroski_each_component_flips_exactly(
    override: dict[str, float], expected_zeros: set[str]
) -> None:
    comps = piotroski_components_annual(**{**PIOTROSKI_BASE, **override})
    zeros = {k for k in PIOTROSKI_COMPONENTS if comps[k] == 0.0}
    ones = {k for k in PIOTROSKI_COMPONENTS if comps[k] == 1.0}
    assert zeros == expected_zeros
    assert ones == set(PIOTROSKI_COMPONENTS) - expected_zeros
    assert comps["f_score"] == 9.0 - len(expected_zeros)


def test_piotroski_missing_input_gives_nan_not_zero() -> None:
    comps = piotroski_components_annual(
        **{**PIOTROSKI_BASE, "current_liabilities": float("nan")}
    )
    assert math.isnan(comps["f_dliquid"])
    assert math.isnan(comps["f_score"])  # un F sobre 8 componentes no es comparable
    assert comps["f_roa"] == 1.0  # el resto sigue calculándose


def test_piotroski_scale_invariance() -> None:
    scaled = {
        k: (v * 7.0 if k != "equity_issuance" else v) for k, v in PIOTROSKI_BASE.items()
    }
    base = piotroski_components_annual(**PIOTROSKI_BASE)
    seven = piotroski_components_annual(**scaled)
    assert base == seven


def test_piotroski_panel_range_and_components(ctx: object) -> None:
    factor = PiotroskiFScore(variant="binary")
    with pytest.warns(UserWarning, match="equity_issuance"):
        daily = factor.compute_components(ctx)
    comp_cols = list(PIOTROSKI_COMPONENTS)
    assert list(daily.columns) == [*comp_cols, "f_score"]
    values = daily[comp_cols].stack().dropna().unique()
    assert set(values) <= {0.0, 1.0}
    scores = daily["f_score"].dropna()
    assert len(scores) > 0
    assert scores.isin(range(10)).all()
    # Identidad interna: el F es la suma exacta de los 9 componentes.
    complete = daily.dropna()
    pd.testing.assert_series_equal(
        complete[comp_cols].sum(axis=1), complete["f_score"], check_names=False
    )
    # El generador sintético nunca emite acciones: EQ_OFFER siempre 1 donde hay dato.
    assert (daily["f_eq_offer"].dropna() == 1.0).all()


def test_piotroski_continuous_more_granular(ctx: object) -> None:
    binary = PiotroskiFScore(variant="binary").compute(ctx)
    continuous = PiotroskiFScore(variant="continuous").compute(ctx)
    last_date = binary.index.get_level_values("date").max()
    xs_b = binary.xs(last_date, level="date").dropna()
    xs_c = continuous.xs(last_date, level="date").dropna()
    assert len(xs_c) >= 8  # sección cruzada suficiente para que el test signifique algo
    assert xs_c.nunique() > xs_b.nunique()
    assert (xs_c > 0).all() and (xs_c < 1).all()  # promedio de percentiles


def test_piotroski_structural_nan_financials(ctx: object, market: SyntheticMarket) -> None:
    sectors = market.sectors()
    excluded = sectors[sectors.isin(["Financials", "Real Estate"])].index
    kept = sectors[~sectors.isin(["Financials", "Real Estate"])].index
    assert len(excluded) > 0  # la muestra estratificada incluye ambos sectores
    for factor in [
        PiotroskiFScore(variant="binary"),
        PercentAccruals(),
        GrossProfitability(),
        FreeCashFlowYield(),
        CFOToNetIncome(),
    ]:
        series = factor.compute(ctx)
        by_ticker = series.groupby(level="ticker").count()
        for t in excluded:
            assert by_ticker.get(t, 0) == 0, (factor.name, t)
        assert sum(by_ticker.get(t, 0) for t in kept) > 0, factor.name


# =========================================================================== #
# 2. Identidades contables
# =========================================================================== #


def test_sloan_formulas_agree_when_articulated() -> None:
    """Con articulación perfecta, balance y flujos dan el mismo accrual."""
    d_ca, d_cash, d_cl, d_std, d_txp, dep = 80.0, 40.0, 20.0, 8.0, 4.0, 40.0
    avg_ta = 1050.0
    # NI − CFO construido para igualar el numerador del método de balance.
    numerator = (d_ca - d_cash) - (d_cl - d_std - d_txp) - dep
    ni, cfo = 100.0, 100.0 - numerator
    bs = sloan_accruals_balance(
        delta_current_assets=d_ca,
        delta_cash=d_cash,
        delta_current_liabilities=d_cl,
        delta_short_term_debt=d_std,
        delta_taxes_payable=d_txp,
        depreciation=dep,
        avg_total_assets=avg_ta,
    )
    cf = sloan_accruals_cashflow(net_income=ni, cfo=cfo, avg_total_assets=avg_ta)
    assert bs == pytest.approx(cf)
    assert bs == pytest.approx(numerator / avg_ta)


def test_sloan_formulas_disagree_without_articulation() -> None:
    """Una partida no corriente (p. ej. SBC) rompe la articulación: la
    discrepancia es exactamente la bandera de Collins-Hribar (2002)."""
    bs = sloan_accruals_balance(
        delta_current_assets=80.0,
        delta_cash=40.0,
        delta_current_liabilities=20.0,
        delta_short_term_debt=8.0,
        delta_taxes_payable=0.0,
        depreciation=40.0,
        avg_total_assets=1050.0,
    )
    cf = sloan_accruals_cashflow(net_income=200.0, cfo=230.0, avg_total_assets=1050.0)
    assert bs != pytest.approx(cf)


def test_percent_accruals_defined_with_losses() -> None:
    """El ejemplo verificado del informe §7.3: NI=−200, CFO=50 → PACC=−1.25."""
    assert percent_accruals(net_income=-200.0, cfo=50.0) == pytest.approx(-1.25)
    assert math.isnan(percent_accruals(net_income=0.0, cfo=50.0))


def test_cfo_ni_is_one_minus_pacc_scalar() -> None:
    """Identidad §8.1 (verificada allí: 1.3333 == 1.3333) con NI > 0."""
    ni, cfo = 150.0, 200.0
    assert cfo / ni == pytest.approx(1.0 - percent_accruals(net_income=ni, cfo=cfo))


def test_cfo_ni_is_one_minus_pacc_panel(ctx: object) -> None:
    """La identidad se mantiene en el panel completo: no son dos factores."""
    pacc_factor = PercentAccruals().compute(ctx)  # = −PACC
    cfo_ni = CFOToNetIncome().compute(ctx)
    both = cfo_ni.notna() & pacc_factor.notna()
    assert both.sum() > 100
    lhs = cfo_ni[both]
    rhs = 1.0 + pacc_factor[both]  # 1 − PACC = 1 + (−PACC)
    assert np.allclose(lhs.to_numpy(), rhs.to_numpy(), rtol=1e-10)


def test_synthetic_accruals_column_reproduced(market: SyntheticMarket) -> None:
    """La fórmula pura reproduce la columna `accruals` del generador (implementación
    independiente de la misma definición de Sloan por flujos)."""
    fund = market.fundamentals()
    sub = fund[fund["ticker"] == fund["ticker"].iloc[0]].sort_values("period_end")
    ta_prev = sub["total_assets"].shift(1)
    manual = [
        sloan_accruals_cashflow(
            net_income=ni, cfo=cfo, avg_total_assets=(ta + tp) / 2.0
        )
        for ni, cfo, ta, tp in zip(
            sub["net_income"], sub["cfo"], sub["total_assets"], ta_prev, strict=True
        )
    ]
    expected = sub["accruals"].to_numpy()[1:]
    assert np.allclose(np.array(manual)[1:], expected, rtol=1e-9)


def test_enterprise_value_formula() -> None:
    assert enterprise_value(1000.0, 400.0, 100.0) == pytest.approx(1300.0)
    assert enterprise_value(1000.0, 400.0, 100.0, preferred=50.0, minority=25.0) == pytest.approx(
        1375.0
    )


# =========================================================================== #
# 3. Point-in-time: cortes exactos, truncado y sensibilidad
# =========================================================================== #


def test_asof_boundary_filed_at_bites() -> None:
    """Un 10-Q aceptado el 16-feb a las 08:00 no es usable el 16 con corte a
    medianoche (política conservadora `date_start`): entra el 17. Sin relleno
    hacia atrás: antes del filing el factor es NaN."""
    fund = make_fund("AAA", 4, start_period="2020-03-31", net_income=100.0, cfo=80.0)
    fund.loc[fund.index[-1], "filed_at"] = pd.Timestamp("2021-02-16 08:00:00")
    dates = CAL.sessions(pd.Timestamp("2021-02-08"), pd.Timestamp("2021-02-19"))
    series = PercentAccruals().compute(make_ctx(fundamentals=fund, dates=dates))

    before = series.xs("AAA", level="ticker").loc[: pd.Timestamp("2021-02-16")]
    assert before.isna().all()  # incluye el propio día 16: aún no público al corte
    on_17 = series.loc[(pd.Timestamp("2021-02-17"), "AAA")]
    assert on_17 == pytest.approx(-0.2)  # (400 − 320)/|400| = 0.2; factor = −PACC


def test_shifting_filed_at_moves_the_signal() -> None:
    """Test de sensibilidad de §15.2: si retrasar el filing no mueve la señal,
    el as-of join está roto."""
    fund = make_fund("AAA", 4, start_period="2020-03-31", net_income=100.0, cfo=80.0)
    fund.loc[fund.index[-1], "filed_at"] = pd.Timestamp("2021-02-16 08:00:00")
    dates = CAL.sessions(pd.Timestamp("2021-02-08"), pd.Timestamp("2021-02-26"))
    base = PercentAccruals().compute(make_ctx(fundamentals=fund, dates=dates))
    assert base.loc[(pd.Timestamp("2021-02-17"), "AAA")] == pytest.approx(-0.2)

    delayed = fund.copy()
    delayed.loc[delayed.index[-1], "filed_at"] += pd.Timedelta(days=7)
    moved = PercentAccruals().compute(make_ctx(fundamentals=delayed, dates=dates))
    assert pd.isna(moved.loc[(pd.Timestamp("2021-02-17"), "AAA")])
    assert moved.loc[(pd.Timestamp("2021-02-24"), "AAA")] == pytest.approx(-0.2)


def test_announcement_vs_filing_policies() -> None:
    """SalesYield (nota de prensa) se activa semanas antes que PercentAccruals
    (10-Q) sobre los mismos hechos: la latencia de §1.2 del informe."""
    fund = make_fund(
        "AAA", 4, start_period="2020-03-31",
        avail_lag_days=5, filed_lag_days=40,
        revenue=1000.0, net_income=100.0, cfo=80.0,
    )
    dates = CAL.sessions(pd.Timestamp("2021-01-04"), pd.Timestamp("2021-02-26"))
    prices = make_prices(dates, {"AAA": 8000.0})
    context = make_ctx(fundamentals=fund, dates=dates, prices=prices)

    sales = SalesYield().compute(context).xs("AAA", level="ticker")
    pacc = PercentAccruals().compute(context).xs("AAA", level="ticker")

    # available_at del Q4 = 2021-01-05 00:00 → utilizable ese mismo día (corte
    # a medianoche con igualdad exacta admitida).
    assert pd.isna(sales.loc[pd.Timestamp("2021-01-04")])
    assert sales.loc[pd.Timestamp("2021-01-05")] == pytest.approx(4000.0 / 8000.0)
    # filed_at del Q4 = 2021-02-09 00:00 → los accruals esperan al 10-Q.
    assert pd.isna(pacc.loc[pd.Timestamp("2021-02-08")])
    assert pacc.loc[pd.Timestamp("2021-02-09")] == pytest.approx(-0.2)


@pytest.mark.parametrize(
    ("factory", "avail_col"),
    [(PercentAccruals, "filed_at"), (EarningsYield, "available_at")],
)
def test_truncation_no_lookahead(
    ctx: object, market: SyntheticMarket, factory: type, avail_col: str
) -> None:
    """Test mecánico de §15.2: truncar los datos en T y exigir igualdad con la
    parte <= T del panel completo. Si difiere, hay look-ahead."""
    full = factory().compute(ctx)
    dates = pd.DatetimeIndex(market.sessions[::5])
    cutoff = dates[len(dates) // 2]

    fund = market.fundamentals()
    truncated_fund = fund[pd.to_datetime(fund[avail_col]) <= cutoff]
    truncated_ctx = make_ctx(
        fundamentals=truncated_fund,
        dates=dates[dates <= cutoff],
        prices=market.prices(),
        sectors=market.sectors().to_dict(),
    )
    truncated = factory().compute(truncated_ctx)

    full_head = full[full.index.get_level_values("date") <= cutoff]
    pd.testing.assert_series_equal(full_head, truncated)


# =========================================================================== #
# 4. Requisitos, degradaciones y fallo explícito
# =========================================================================== #


def test_missing_required_column_raises() -> None:
    fund = make_fund("AAA", 8, net_income=100.0)  # sin `cfo`
    with pytest.raises(DataQualityError, match="cfo"):
        PercentAccruals().compute(make_ctx(fundamentals=fund, dates=MICRO_DATES))


def test_empty_fundamentals_raises_insufficient_history() -> None:
    empty = pd.DataFrame(columns=["ticker", "period_end", "available_at", "net_income", "cfo"])
    with pytest.raises(InsufficientHistory):
        PercentAccruals().compute(make_ctx(fundamentals=empty, dates=MICRO_DATES))


def test_missing_prices_raises() -> None:
    fund = make_fund("AAA", 8, net_income=100.0)
    with pytest.raises(DataQualityError, match="prices"):
        EarningsYield().compute(make_ctx(fundamentals=fund, dates=MICRO_DATES))


def test_missing_shares_refuses_totals() -> None:
    """Sin acciones no hay crecimiento por acción, y con totales el factor
    mediría recompras (§3 del informe): fallo explícito, no degradación."""
    fund = make_fund("AAA", 10, revenue=1000.0)
    with pytest.raises(DataQualityError, match="POR ACCIÓN"):
        SalesGrowth().compute(make_ctx(fundamentals=fund, dates=MICRO_DATES))


def test_balance_accruals_optional_taxes_payable() -> None:
    """`taxes_payable` es opcional: sin ella se asume ΔTXP=0 con aviso; con
    ella el resultado cambia exactamente en ΔTXP/activo medio."""
    base_cols = {
        "current_assets": [500.0] * 4 + [520.0, 540.0, 560.0, 580.0],
        "cash": [100.0] * 4 + [110.0, 120.0, 130.0, 140.0],
        "current_liabilities": [300.0] * 4 + [305.0, 310.0, 315.0, 320.0],
        "short_term_debt": [50.0] * 4 + [52.0, 54.0, 56.0, 58.0],
        "depreciation_amortization": 10.0,
        "total_assets": [1000.0] * 4 + [1025.0, 1050.0, 1075.0, 1100.0],
    }
    fund = make_fund("AAA", 8, **base_cols)
    with pytest.warns(UserWarning, match="taxes_payable"):
        without = SloanAccrualsBalance().compute(make_ctx(fundamentals=fund, dates=MICRO_DATES))
    last = (MICRO_DATES[-1], "AAA")
    # ((80−40) − (20−8−0) − 40)/1050 = −12/1050; factor = −ACC_BS.
    assert without.loc[last] == pytest.approx(12.0 / 1050.0)

    fund_txp = make_fund(
        "AAA", 8, **base_cols, taxes_payable=[20.0] * 4 + [22.0, 24.0, 26.0, 28.0]
    )
    with_txp = SloanAccrualsBalance().compute(make_ctx(fundamentals=fund_txp, dates=MICRO_DATES))
    # ΔTXP=8 → ((80−40) − (20−8−8) − 40)/1050 = −4/1050.
    assert with_txp.loc[last] == pytest.approx(4.0 / 1050.0)


def test_articulation_gap_flag() -> None:
    """La discrepancia balance/flujos invalida observaciones (§7.2): con umbral
    estrecho todo cae (InsufficientHistory explícito), con umbral holgado no."""
    fund = make_fund(
        "AAA", 8,
        current_assets=[500.0] * 4 + [520.0, 540.0, 560.0, 580.0],
        cash=[100.0] * 4 + [110.0, 120.0, 130.0, 140.0],
        current_liabilities=[300.0] * 4 + [305.0, 310.0, 315.0, 320.0],
        short_term_debt=[50.0] * 4 + [52.0, 54.0, 56.0, 58.0],
        depreciation_amortization=10.0,
        total_assets=[1000.0] * 4 + [1025.0, 1050.0, 1075.0, 1100.0],
        net_income=50.0,
        cfo=57.5,
    )
    # ACC_CF = (200−230)/1050 ≈ −0.02857; ACC_BS = −12/1050 ≈ −0.01143; gap ≈ 0.0171.
    loose = SloanAccrualsCashFlow(max_articulation_gap=0.05).compute(
        make_ctx(fundamentals=fund, dates=MICRO_DATES)
    )
    assert loose.loc[(MICRO_DATES[-1], "AAA")] == pytest.approx(30.0 / 1050.0)
    with pytest.raises(InsufficientHistory), pytest.warns(UserWarning, match="discrepancia"):
        SloanAccrualsCashFlow(max_articulation_gap=0.01).compute(
            make_ctx(fundamentals=fund, dates=MICRO_DATES)
        )
    gap = accruals_articulation_gap(make_ctx(fundamentals=fund, dates=MICRO_DATES))
    assert gap.loc[(MICRO_DATES[-1], "AAA")] == pytest.approx(18.0 / 1050.0)


def test_all_excluded_sectors_raise_insufficient() -> None:
    """Universo íntegramente financiero → todo NaN → fallo agregado explícito,
    jamás un panel vacío silencioso."""
    fund = make_fund("AAA", 8, net_income=100.0, cfo=80.0)
    context = make_ctx(fundamentals=fund, dates=MICRO_DATES, sectors={"AAA": "Financials"})
    with pytest.raises(InsufficientHistory):
        PercentAccruals().compute(context)


def test_restated_rows_are_discarded() -> None:
    """El backtest usa la cifra tal y como se reportó por primera vez (§1.3)."""
    fund = make_fund("AAA", 4, start_period="2020-03-31", net_income=100.0, cfo=80.0)
    restated = fund.copy()
    restated["net_income"] = 999.0
    restated["is_restated"] = True
    fund["is_restated"] = False
    both = pd.concat([fund, restated], ignore_index=True)
    dates = CAL.sessions(pd.Timestamp("2021-02-22"), pd.Timestamp("2021-02-26"))
    with pytest.warns(UserWarning, match="reexpresadas"):
        series = PercentAccruals().compute(make_ctx(fundamentals=both, dates=dates))
    assert series.dropna().iloc[-1] == pytest.approx(-0.2)  # la cifra original


# =========================================================================== #
# 5. Historia mínima (funciones puras y panel)
# =========================================================================== #


def test_sales_growth_history_requirement() -> None:
    growing = [100.0] * 4 + [125.0] * 4
    assert sales_growth_ttm(growing) == pytest.approx(0.25)
    with pytest.raises(InsufficientHistory):
        sales_growth_ttm(growing[:7])


def test_sales_acceleration_history_requirement() -> None:
    # Serie con crecimiento variable para que la dispersión no sea degenerada.
    r = [100.0 * (1.0 + 0.01 * q + 0.002 * (q % 3)) ** 2 for q in range(13)]
    value = sales_acceleration(r, standardized=True)
    assert math.isfinite(value)
    with pytest.raises(InsufficientHistory):
        sales_acceleration(r[:12], standardized=True)
    raw = sales_acceleration(r[-6:], standardized=False)
    assert math.isfinite(raw)
    with pytest.raises(InsufficientHistory):
        sales_acceleration(r[:5], standardized=False)


def test_trend_t_stat_history_and_noise_ordering() -> None:
    """§9.2 del informe: a igual pendiente real, la serie limpia debe puntuar
    más que la ruidosa (la diferencia simple ordena al revés)."""
    rng = np.random.default_rng(0)
    t_grid = np.arange(8, dtype=float)
    clean = 0.30 + 0.004 * t_grid + rng.normal(0.0, 0.001, 8)
    noisy = 0.30 + 0.004 * t_grid + rng.normal(0.0, 0.010, 8)
    t_clean = trend_t_stat(clean, 8)
    t_noisy = trend_t_stat(noisy, 8)
    assert t_clean > t_noisy > 0
    with pytest.raises(InsufficientHistory):
        trend_t_stat(clean[:7], 8)
    assert trend_t_stat([1.0] * 8, 8) == 0.0  # serie constante: sin tendencia


def test_sustainable_growth_formula() -> None:
    value = sustainable_growth(net_income_ttm=100.0, dividends_ttm=40.0, avg_equity=500.0)
    assert value == pytest.approx(0.12)  # ROE 0.2 × retención 0.6
    assert math.isnan(
        sustainable_growth(net_income_ttm=-10.0, dividends_ttm=0.0, avg_equity=500.0)
    )


def test_panel_min_history_gives_nan_per_name() -> None:
    """Un nombre con 7 trimestres no puntúa en crecimiento; con 8 sí. NaN por
    nombre, nunca un valor calculado sobre menos observaciones (§1.3)."""
    short = make_fund("AAA", 7, revenue=1000.0, shares_diluted=10.0)
    long = make_fund("BBB", 8, revenue=[1000.0] * 4 + [1250.0] * 4, shares_diluted=10.0)
    fund = pd.concat([short, long], ignore_index=True)
    series = SalesGrowth().compute(make_ctx(fundamentals=fund, dates=MICRO_DATES))
    assert series.xs("AAA", level="ticker").isna().all()
    assert series.loc[(MICRO_DATES[-1], "BBB")] == pytest.approx(-0.25)


# =========================================================================== #
# 6. Aritmética exacta de los factores (paneles de juguete)
# =========================================================================== #


def test_value_factors_hand_numbers() -> None:
    context = micro_value_ctx()
    last = MICRO_DATES[-1]

    ey = EarningsYield().compute(context)
    assert ey.loc[(last, "AAA")] == pytest.approx(100.0 / 1000.0)
    # Pérdidas: E/P negativo, finito y ordenable (política (a) de §11.2).
    assert ey.loc[(last, "BBB")] == pytest.approx(-900.0 / 1000.0)
    assert ey.loc[(last, "BBB")] < ey.loc[(last, "AAA")]

    sy = SalesYield().compute(context)
    assert sy.loc[(last, "AAA")] == pytest.approx(1000.0 / 1000.0)

    fcf = FreeCashFlowYield().compute(context)
    assert fcf.loc[(last, "AAA")] == pytest.approx((120.0 - 20.0) / 1000.0)

    bm = BookToMarket().compute(context)
    assert bm.loc[(last, "AAA")] == pytest.approx(500.0 / 1000.0)
    assert pd.isna(bm.loc[(last, "BBB")])  # patrimonio negativo → NaN, no "barata"

    with pytest.warns(UserWarning, match="preferred_equity"):
        ev = EbitdaToEV().compute(context)
    # EV(AAA) = 1000 + 400 − 100 = 1300; EBITDA TTM = 130 → 0.1 exacto.
    assert ev.loc[(last, "AAA")] == pytest.approx(130.0 / 1300.0)
    assert pd.isna(ev.loc[(last, "BBB")])  # EV = 1000 + 400 − 5000 < 0 → NaN


def test_cheaper_firm_scores_higher() -> None:
    """Mismos fundamentales, mitad de capitalización → doble yield."""
    aaa = make_fund("AAA", 8, net_income=25.0)
    bbb = make_fund("BBB", 8, net_income=25.0)
    fund = pd.concat([aaa, bbb], ignore_index=True)
    prices = make_prices(MICRO_DATES, {"AAA": 1000.0, "BBB": 500.0})
    ey = EarningsYield().compute(
        make_ctx(fundamentals=fund, dates=MICRO_DATES, prices=prices)
    )
    last = MICRO_DATES[-1]
    assert ey.loc[(last, "BBB")] == pytest.approx(2.0 * ey.loc[(last, "AAA")])


def test_growth_factors_hand_numbers() -> None:
    fund = make_fund(
        "AAA", 8,
        revenue=[1000.0] * 4 + [1250.0] * 4,
        shares_diluted=10.0,
        eps_diluted=[1.0] * 4 + [1.1] * 4,
        net_income=25.0,
        dividends_paid=10.0,
        total_equity=500.0,
        total_assets=[1000.0] * 4 + [1000.0, 1100.0, 1200.0, 1300.0],
    )
    context = make_ctx(fundamentals=fund, dates=MICRO_DATES)
    last = (MICRO_DATES[-1], "AAA")

    assert SalesGrowth().compute(context).loc[last] == pytest.approx(-0.25)
    assert EarningsGrowth().compute(context).loc[last] == pytest.approx(-0.1)
    assert AssetGrowth().compute(context).loc[last] == pytest.approx(-0.3)
    assert SustainableGrowth().compute(context).loc[last] == pytest.approx(0.12)


def test_per_share_basis_detects_buybacks() -> None:
    """Ingresos totales planos + recompra del 10 % → crecimiento por acción
    positivo → factor negativo. Con totales daría 0: la trampa de §3."""
    fund = make_fund(
        "AAA", 8, revenue=1000.0, shares_diluted=[10.0] * 4 + [9.0] * 4
    )
    series = SalesGrowth().compute(make_ctx(fundamentals=fund, dates=MICRO_DATES))
    value = series.loc[(MICRO_DATES[-1], "AAA")]
    assert value == pytest.approx(-(4000.0 / 9.0 / (4000.0 / 10.0) - 1.0))
    assert value < -0.05


def test_percent_accruals_scale_invariance() -> None:
    fund = make_fund("AAA", 4, start_period="2020-03-31", net_income=100.0, cfo=80.0)
    scaled = fund.copy()
    scaled[["net_income", "cfo"]] *= 7.0
    dates = CAL.sessions(pd.Timestamp("2021-02-22"), pd.Timestamp("2021-02-26"))
    a = PercentAccruals().compute(make_ctx(fundamentals=fund, dates=dates))
    b = PercentAccruals().compute(make_ctx(fundamentals=scaled, dates=dates))
    pd.testing.assert_series_equal(a, b)


def test_leverage_sign_and_nonpositive_ebitda() -> None:
    aaa = make_fund("AAA", 4, start_period="2020-03-31", ebitda=32.5, total_debt=400.0, cash=100.0)
    bbb = make_fund("BBB", 4, start_period="2020-03-31", ebitda=-5.0, total_debt=400.0, cash=100.0)
    fund = pd.concat([aaa, bbb], ignore_index=True)
    dates = CAL.sessions(pd.Timestamp("2021-02-22"), pd.Timestamp("2021-02-26"))
    lev = Leverage().compute(make_ctx(fundamentals=fund, dates=dates))
    last = dates[-1]
    # −NetDebt/EBITDA = −(400−100)/130: signo invertido documentado (§12.1).
    assert lev.loc[(last, "AAA")] == pytest.approx(-300.0 / 130.0)
    assert pd.isna(lev.loc[(last, "BBB")])  # EBITDA ≤ 0 → NaN


def test_gross_profitability_hand_numbers() -> None:
    fund = make_fund(
        "AAA", 4, start_period="2020-03-31",
        revenue=250.0, cost_of_revenue=150.0, total_assets=2000.0,
    )
    dates = CAL.sessions(pd.Timestamp("2021-02-22"), pd.Timestamp("2021-02-26"))
    gpa = GrossProfitability().compute(make_ctx(fundamentals=fund, dates=dates))
    # (1000 − 600) / 2000 = 0.2 (Novy-Marx: bruto sobre activo total).
    assert gpa.loc[(dates[-1], "AAA")] == pytest.approx(0.2)


# =========================================================================== #
# 7. Forma canónica, determinismo y contrato de salida
# =========================================================================== #

ALL_FACTORY_CALLS = [
    EarningsYield,
    FreeCashFlowYield,
    BookToMarket,
    EbitdaToEV,
    SalesYield,
    lambda: PiotroskiFScore(variant="binary"),
    PiotroskiFScore,
    lambda: PiotroskiFScore(variant="binary", frequency="annual"),
    CFOToNetIncome,
    GrossProfitability,
    MarginTrend,
    lambda: MarginTrend(margin="operating"),
    lambda: Profitability("roa"),
    lambda: Profitability("roe"),
    EarningsStability,
    Leverage,
    SalesGrowth,
    SalesAcceleration,
    lambda: SalesAcceleration(standardized=False),
    EarningsGrowth,
    SustainableGrowth,
    AssetGrowth,
    SloanAccrualsBalance,
    lambda: SloanAccrualsBalance(frequency="annual"),
    SloanAccrualsCashFlow,
    PercentAccruals,
    NetOperatingAccruals,
]


@pytest.mark.parametrize("factory", ALL_FACTORY_CALLS)
def test_factor_output_is_canonical(ctx: object, factory: object) -> None:
    factor = factory()
    series = factor.compute(ctx)
    assert isinstance(series, pd.Series)
    check_panel(series, name=factor.name)  # MultiIndex (date, ticker) ordenado y único
    assert series.name == factor.name
    assert series.dtype == np.float64
    assert series.notna().any()  # finalize_factor garantiza no-vacío
    assert not np.isinf(series.dropna()).any()
    dates = series.index.get_level_values("date").unique()
    assert dates.isin(pd.DatetimeIndex(ctx.dates)).all()
    # Protocolo Factor (contrato §3.4): nombre y requisitos declarados.
    assert isinstance(factor.name, str) and factor.name
    assert isinstance(factor.requires, list)
    assert "fundamentals" in factor.requires


def test_factor_is_deterministic(ctx: object) -> None:
    first = PercentAccruals().compute(ctx)
    second = PercentAccruals().compute(ctx)
    pd.testing.assert_series_equal(first, second)


def test_registered_in_default_registry(ctx: object) -> None:
    """Coordinación con `factors/base.py`: si el registro global existe, los
    factores contables están registrados y producen lo mismo que la clase."""
    base = pytest.importorskip("earnings_alpha.factors.base")
    names = base.default_registry.names()
    for expected in (
        "earnings_yield",
        "fcf_yield",
        "book_to_market",
        "ebitda_to_ev",
        "sales_yield",
        "piotroski_f",
        "piotroski_f_cont",
        "cfo_to_ni",
        "gross_profitability",
        "margin_trend_gross",
        "roa",
        "roe",
        "earnings_stability",
        "low_leverage",
        "sales_growth",
        "sales_acceleration",
        "earnings_growth",
        "sustainable_growth",
        "asset_growth",
        "percent_accruals",
        "accruals_cf",
        "accruals_bs",
        "net_operating_accruals",
    ):
        assert expected in names, expected
    via_registry = base.default_registry.compute("percent_accruals", ctx)
    pd.testing.assert_series_equal(via_registry, PercentAccruals().compute(ctx))


def test_annual_variant_updates_less_often(ctx: object) -> None:
    """La variante anual del F-Score solo cambia con cada 10-K: menos valores
    distintos por ticker que la TTM."""
    ttm = PiotroskiFScore(variant="binary", frequency="ttm").compute(ctx)
    annual = PiotroskiFScore(variant="binary", frequency="annual").compute(ctx)
    changes_ttm = ttm.groupby(level="ticker").nunique().sum()
    changes_annual = annual.groupby(level="ticker").nunique().sum()
    assert changes_annual <= changes_ttm
    assert annual.dropna().isin(range(10)).all()
