"""Tests del módulo `factors`: sorpresa, PEAD, revisiones y guidance.

Corren **sin red**. Tres frentes, en orden de importancia:

1. **Contratos matemáticos del informe** (`fundamental_factors.md` §15.2):
   13 trimestres exactos para el modelo estacional (`InsufficientHistory` con
   12), invariancia de escala de `SUE_sigma`, NaN —no inf— con sigma
   degenerada, e ingresos POR ACCIÓN en SURGE (el error del signo invertido).
2. **Point-in-time.** El panel de cada factor pasa `pit.assert_no_lookahead`
   (incluida la comprobación de primera disponibilidad), la auditoría *muerde*
   cuando se desplaza la señal hacia el pasado, y mover un evento de AMC a BMO
   mueve la entrada de la señal exactamente una sesión — el test que demuestra
   que `tradable_date` gobierna de verdad el as-of join.
3. **Verdad-terreno sintética.** El SUE de analistas reproduce exactamente el
   `sue` del generador y recupera la sorpresa latente inyectada
   (`ground_truth`); el PEAD sintético tiene el signo esperado (el generador
   inyecta drift post-anuncio proporcional a la sorpresa).

Además, un bloque contra el dataset REAL `consenso_master.parquet`: parseo,
derivación de sesiones BMO/AMC desde `report_time`, deduplicación entre
fuentes y cálculo del SUE de analistas para una muestra de tickers.
"""

from __future__ import annotations

import datetime as dt
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from earnings_alpha.data.synthetic import SyntheticMarket
from earnings_alpha.errors import (
    ConfigError,
    InsufficientHistory,
    LookAheadError,
    ProviderUnavailable,
)
from earnings_alpha.factors import (
    PEAD,
    AnalystDispersion,
    AnalystSUE,
    DoubleSurprise,
    FactorContext,
    FactorRegistry,
    GuidanceChange,
    NoVintagesWarning,
    RevenueSurprise,
    RevisionDiffusion,
    RevisionMomentum,
    TimeSeriesSUE,
    analyst_sue_events,
    context_from_synthetic,
    default_registry,
    double_surprise,
    foster_surprise_events,
    guidance_surprise,
    has_revision_vintages,
    load_consensus_events,
    revenue_surprise_events,
    score_guidance_actions,
    seasonal_random_walk,
    spread_event_values,
    sue_time_series,
)
from earnings_alpha.pit import assert_no_lookahead, get_calendar
from earnings_alpha.types import SurpriseBasis

SEED = 11


@pytest.fixture(scope="module")
def mkt() -> SyntheticMarket:
    return SyntheticMarket(seed=SEED, n_tickers=24, start="2021-01-04", end="2022-12-30")


@pytest.fixture(scope="module")
def ctx(mkt: SyntheticMarket) -> FactorContext:
    return context_from_synthetic(mkt)


# ---------------------------------------------------------------------------
# 1. Primitivas: contratos matemáticos exactos
# ---------------------------------------------------------------------------


def _drifting_seasonal_series(n: int = 13, *, drift: float = 0.05, seed: int = 3) -> np.ndarray:
    """Serie trimestral con estacionalidad, deriva y ruido reproducible."""
    rng = np.random.default_rng(seed)
    season = np.tile([0.9, 1.0, 1.1, 1.3], (n + 3) // 4 + 1)[:n]
    return season + drift * np.arange(n) + rng.normal(0.0, 0.03, n)


class TestSeasonalRandomWalk:
    def test_twelve_quarters_raise_thirteen_compute(self) -> None:
        """§15.2: con 12 trimestres `InsufficientHistory`; con 13, valor finito."""
        eps = _drifting_seasonal_series(13)
        with pytest.raises(InsufficientHistory):
            seasonal_random_walk(eps[:12])
        ue, drift, sigma = seasonal_random_walk(eps)
        assert np.isfinite(ue) and np.isfinite(drift) and np.isfinite(sigma)
        assert sigma >= 0.0

    def test_nan_inside_window_raises(self) -> None:
        eps = _drifting_seasonal_series(13)
        eps[5] = np.nan
        with pytest.raises(InsufficientHistory):
            seasonal_random_walk(eps)

    def test_pure_drift_has_zero_surprise(self) -> None:
        """Con E_q = a + d·q la expectativa es exacta: UE = 0 y sigma = 0."""
        eps = 1.0 + 0.07 * np.arange(13)
        ue, drift, sigma = seasonal_random_walk(eps)
        assert ue == pytest.approx(0.0, abs=1e-12)
        assert drift == pytest.approx(0.28, abs=1e-12)  # 4 trimestres * 0.07
        assert sigma == pytest.approx(0.0, abs=1e-12)

    def test_sue_sigma_scale_invariance(self) -> None:
        """§15.2: SUE_sigma(k·EPS) == SUE_sigma(EPS) para todo k>0."""
        eps = _drifting_seasonal_series(13)
        base = sue_time_series(eps, basis=SurpriseBasis.SIGMA)
        scaled = sue_time_series(7.0 * eps, basis=SurpriseBasis.SIGMA)
        assert np.isfinite(base)
        assert scaled == pytest.approx(base, rel=1e-12)

    def test_sigma_zero_gives_nan_not_inf_without_floor(self) -> None:
        """§2.2: el modo de fallo sigma→0 no puede degenerar en ±inf silencioso."""
        eps = 1.0 + 0.07 * np.arange(13)
        eps[-1] += 0.01  # un céntimo de sorpresa con serie perfectamente regular
        out = sue_time_series(eps, basis=SurpriseBasis.SIGMA)
        assert np.isnan(out)

    def test_sigma_floor_activates_with_price(self) -> None:
        eps = 1.0 + 0.07 * np.arange(13)
        eps[-1] += 0.01
        out = sue_time_series(eps, basis=SurpriseBasis.SIGMA, price=180.0)
        # suelo = 0.005 * 180 = 0.9 -> SUE = 0.01 / 0.9
        assert out == pytest.approx(0.01 / 0.9, rel=1e-9)

    def test_sue_price_exact_and_requires_price(self) -> None:
        eps = 1.0 + 0.07 * np.arange(13)
        eps[-1] += 0.14
        out = sue_time_series(eps, basis=SurpriseBasis.PRICE, price=180.0)
        assert out == pytest.approx(0.14 / 180.0, rel=1e-9)
        with pytest.raises(ConfigError):
            sue_time_series(eps, basis=SurpriseBasis.PRICE)

    def test_analyst_bases_rejected(self) -> None:
        with pytest.raises(ConfigError):
            sue_time_series(_drifting_seasonal_series(13), basis=SurpriseBasis.ABS_ESTIMATE)


# ---------------------------------------------------------------------------
# 2. Registro de factores
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_default_registry_names(self) -> None:
        expected = {
            "sue_analyst",
            "sue_sigma",
            "sue_price",
            "revenue_surprise",
            "double_surprise",
            "pead",
            "revision_momentum",
            "revision_momentum_1m",
            "revision_momentum_3m",
            "revision_diffusion",
            "analyst_dispersion",
            "guidance_change",
        }
        assert expected <= set(default_registry.names())

    def test_create_and_compute(self, ctx: FactorContext) -> None:
        factor = default_registry.create("sue_analyst", use_snapshots=False)
        out = factor.compute(ctx)
        assert isinstance(out, pd.Series)
        assert list(out.index.names) == ["date", "ticker"]

    def test_unknown_name_raises_with_listing(self) -> None:
        with pytest.raises(ConfigError, match="Registrados"):
            default_registry.create("no_existe")

    def test_duplicate_registration_raises(self) -> None:
        registry = FactorRegistry()

        @registry.register()
        class Dummy:
            name = "dummy"
            requires: ClassVar[list[str]] = []

            def compute(self, ctx: FactorContext) -> pd.Series:  # pragma: no cover
                raise NotImplementedError

        with pytest.raises(ConfigError, match="ya está registrado"):
            registry.register()(Dummy)
        # con overwrite explícito sí se admite
        registry.register(overwrite=True)(Dummy)
        assert "dummy" in registry

    def test_register_without_name_raises(self) -> None:
        registry = FactorRegistry()
        with pytest.raises(ConfigError):
            registry.register()(lambda: None)


# ---------------------------------------------------------------------------
# 3. SUE sobre el mercado sintético
# ---------------------------------------------------------------------------


class TestAnalystSUESynthetic:
    def test_matches_generator_sue_exactly(self, mkt: SyntheticMarket, ctx: FactorContext) -> None:
        """El SUE (base sigma, consenso final) reproduce el `sue` del generador.

        El generador estandariza `eps_surprise` por media y desviación de las 8
        sorpresas previas (min. 6); con los mismos parámetros el cálculo debe
        coincidir a precisión de máquina — cualquier desviación delataría un
        desalineamiento de ventanas u ordenación.
        """
        table = AnalystSUE(use_snapshots=False).event_values(ctx)
        events = mkt.events(include_prehistory=True)
        merged = table.merge(events[["event_id", "sue"]], on="event_id", suffixes=("", "_gen"))
        both = merged.dropna(subset=["sue", "sue_gen"])
        assert len(both) > 100
        assert (both["sue"] - both["sue_gen"]).abs().max() < 1e-10
        # y donde el generador no tiene histórico, nosotros tampoco inventamos
        assert merged["sue"].isna().equals(merged["sue_gen"].isna())

    def test_recovers_injected_surprise(self, mkt: SyntheticMarket, ctx: FactorContext) -> None:
        """El SUE recupera la sorpresa latente `surprise_z` de la verdad-terreno."""
        table = AnalystSUE(use_snapshots=False).event_values(ctx)
        truth = mkt.ground_truth()
        merged = table.dropna(subset=["sue"]).merge(
            truth[["surprise_z"]], left_on="event_id", right_index=True
        )
        assert len(merged) > 100
        corr = merged["sue"].corr(merged["surprise_z"])
        assert corr > 0.5, f"correlación con la sorpresa inyectada demasiado baja: {corr:.3f}"

    def test_pit_snapshot_consensus_is_prior_to_event(self, ctx: FactorContext) -> None:
        """Con fotos as-of, el consenso usado es estrictamente anterior al evento."""
        factor = AnalystSUE(use_snapshots=True)
        table = factor.event_values(ctx)
        est = ctx.estimates
        # el consenso PIT debe existir para los eventos y proceder de fotos
        # con as_of < tradable_date; se verifica reconstruyendo el join
        sample = table.dropna(subset=["consensus"]).head(50)
        for row in sample.itertuples():
            snaps = est[
                (est["ticker"] == row.ticker)
                & (pd.to_datetime(est["period_end"]) == row.period_end)
            ]
            prior = snaps[pd.to_datetime(snaps["as_of"]) < row.tradable_date]
            assert len(prior) > 0
            expected = prior.sort_values("as_of").iloc[-1]["eps_median"]
            assert row.consensus == pytest.approx(expected, rel=1e-12)

    def test_all_bases_compute(self, ctx: FactorContext) -> None:
        for basis in SurpriseBasis:
            table = AnalystSUE(basis=basis, use_snapshots=True).event_values(ctx)
            assert table["sue"].notna().sum() > 0, basis

    def test_insufficient_history_raises(self, ctx: FactorContext) -> None:
        """Con solo 3 eventos por ticker no hay sigma histórica: fallo explícito."""
        short = (
            ctx.events.sort_values(["ticker", "period_end"]).groupby("ticker").head(3)
        )
        with pytest.raises(InsufficientHistory):
            analyst_sue_events(short, basis=SurpriseBasis.SIGMA)


class TestTimeSeriesSUESynthetic:
    def test_panel_computes_and_respects_history(self, ctx: FactorContext) -> None:
        table = TimeSeriesSUE(basis=SurpriseBasis.SIGMA).event_values(ctx)
        by_ticker = table.sort_values(["ticker", "period_end"]).groupby("ticker")
        for _, sub in by_ticker:
            # los 12 primeros eventos de cada ticker no pueden tener SUE
            assert sub["sue"].iloc[:12].isna().all()
        assert table["sue"].notna().sum() > 0

    def test_price_and_sigma_bases_are_monotone_related(self, ctx: FactorContext) -> None:
        """Ambas bases comparten numerador UE: el signo debe coincidir."""
        sig = TimeSeriesSUE(basis=SurpriseBasis.SIGMA).event_values(ctx)
        pric = TimeSeriesSUE(basis=SurpriseBasis.PRICE).event_values(ctx)
        both = pd.DataFrame({"a": sig["sue"], "b": pric["sue"]}).dropna()
        assert len(both) > 50
        assert (np.sign(both["a"]) == np.sign(both["b"])).mean() > 0.99


# ---------------------------------------------------------------------------
# 4. Anti look-ahead
# ---------------------------------------------------------------------------


class TestNoLookahead:
    def test_panel_passes_full_audit(self, mkt: SyntheticMarket, ctx: FactorContext) -> None:
        """El panel del factor pasa las tres comprobaciones de la auditoría."""
        frame = AnalystSUE(use_snapshots=False).compute_frame(ctx)
        events = mkt.events(include_prehistory=True)
        assert_no_lookahead(frame, events, check_first_event=True)

    def test_audit_bites_when_signal_is_shifted_into_past(
        self, mkt: SyntheticMarket, ctx: FactorContext
    ) -> None:
        """Control negativo: adelantar la señal 5 días debe disparar la auditoría.

        Un test de no-look-ahead que nunca falla no demuestra nada; aquí se
        corrompe deliberadamente el panel y se exige el `LookAheadError`.
        """
        frame = AnalystSUE(use_snapshots=False).compute_frame(ctx).reset_index()
        frame["date"] = frame["date"] - pd.Timedelta(days=5)
        with pytest.raises(LookAheadError):
            assert_no_lookahead(frame, mkt.events(include_prehistory=True))

    def test_amc_to_bmo_moves_entry_exactly_one_session(self) -> None:
        """§15.2: cambiar la sesión del anuncio mueve la señal UNA sesión.

        Si el factor no se mueve al cambiar AMC→BMO, el as-of join está roto:
        es la comprobación de que `tradable_date` gobierna la entrada.
        """
        cal = get_calendar()
        rng = np.random.default_rng(5)
        n_ev = 10
        period_ends = pd.date_range("2019-03-31", periods=n_ev, freq="QE")
        announced = [
            pd.Timestamp(cal.session_on_or_after(pe.date() + dt.timedelta(days=35)))
            + pd.Timedelta(hours=21)  # 21:00 UTC = 17:00 ET en verano
            for pe in period_ends
        ]
        eps = 1.0 + 0.02 * np.arange(n_ev) + rng.normal(0, 0.05, n_ev)
        base = pd.DataFrame(
            {
                "ticker": "TEST",
                "period_end": period_ends,
                "announced_at": announced,
                "session": "amc",
                "eps_actual": eps,
                "eps_estimate": eps - rng.normal(0.0, 0.04, n_ev),
            }
        )
        flipped = base.copy()
        flipped.loc[flipped.index[-1], "session"] = "bmo"

        grid = cal.sessions(dt.date(2021, 5, 1), dt.date(2021, 9, 30))

        def first_live(events: pd.DataFrame) -> pd.Timestamp:
            table = analyst_sue_events(events, basis=SurpriseBasis.SIGMA)
            last_sue = table["sue"].iloc[-1]
            assert np.isfinite(last_sue)
            panel = spread_event_values(
                table, grid, ["TEST"], value_col="sue", name="sue"
            ).xs("TEST", level="ticker")
            live = panel[panel == last_sue]
            return live.index[0]

        amc_entry = first_live(base)
        bmo_entry = first_live(flipped)
        assert bmo_entry < amc_entry
        assert pd.Timestamp(cal.next_session(bmo_entry.date())) == amc_entry


# ---------------------------------------------------------------------------
# 5. PEAD
# ---------------------------------------------------------------------------


class TestPEAD:
    def test_signal_lives_exactly_horizon_sessions(self, ctx: FactorContext) -> None:
        horizon = 5
        out = PEAD(horizon=horizon).compute(ctx)
        nonzero = out[out.notna() & (out != 0.0)]
        events_in_panel = ctx.events[
            pd.to_datetime(ctx.events["event_date"]) >= ctx.dates[0]
        ]
        # cada evento del panel enciende exactamente `horizon` sesiones
        assert len(nonzero) == len(events_in_panel) * horizon

    def test_starts_session_after_tradable_date(self, ctx: FactorContext) -> None:
        """Convención `0 < s <= H` (§4.1): el día negociable NO forma parte
        de la ventana; la deriva se mide desde la sesión siguiente."""
        out = PEAD(horizon=5).compute(ctx)
        events = ctx.events.sort_values("event_date")
        cal = ctx.calendar
        sample = events[pd.to_datetime(events["event_date"]) >= ctx.dates[20]].head(10)
        for row in sample.itertuples():
            t0 = pd.Timestamp(row.event_date)
            at_event = out.get((t0, row.ticker), np.nan)
            next_day = pd.Timestamp(cal.next_session(t0.date()))
            if next_day <= ctx.dates[-1]:
                after = out.get((next_day, row.ticker), np.nan)
                if not np.isnan(after):
                    assert after != 0.0
            # en tau=0 el factor no está vivo (0 del evento anterior o NaN)
            assert np.isnan(at_event) or at_event == 0.0

    def test_synthetic_drift_has_expected_sign(
        self, mkt: SyntheticMarket, ctx: FactorContext
    ) -> None:
        """El generador inyecta PEAD real: el factor debe capturarlo con signo +.

        Es el test de §15.2 que caza signos invertidos: sobre datos sintéticos
        con drift proporcional a la sorpresa, el retorno medio de la sesión
        siguiente debe ser mayor bajo señal positiva que bajo negativa.
        """
        out = PEAD(horizon=10).compute(ctx)
        fwd = (
            mkt.prices()["log_return"].unstack("ticker").shift(-1).stack()  # noqa: PD010, PD013
        )
        both = pd.concat([out.rename("pead"), fwd.rename("fwd")], axis=1).dropna()
        live = both[both["pead"] != 0.0]
        assert len(live) > 1000
        spread = live.loc[live["pead"] > 0, "fwd"].mean() - live.loc[
            live["pead"] < 0, "fwd"
        ].mean()
        assert spread > 0.0003, f"spread PEAD sin el signo esperado: {spread:.5f}"
        assert live["pead"].corr(live["fwd"]) > 0.02

    def test_decay_is_monotone_within_window(self, ctx: FactorContext) -> None:
        """Sin estandarizar (el z diario re-escala la sección cruzada y
        rompería la geometría), el decaimiento es exactamente exp(-k/decay)."""
        out = PEAD(horizon=6, decay=2.0, standardize=False).compute(ctx)
        nz = out[out.notna() & (out != 0.0)]
        ticker = nz.index.get_level_values("ticker")[0]
        series = nz.xs(ticker, level="ticker").iloc[:6]
        magnitudes = series.abs().to_numpy()
        assert (np.diff(magnitudes) < 0).all()
        ratios = magnitudes[1:] / magnitudes[:-1]
        assert ratios == pytest.approx(np.exp(-0.5), rel=1e-9)

    def test_zero_after_window_nan_before_first_event(self, ctx: FactorContext) -> None:
        out = PEAD(horizon=5).compute(ctx)
        events = ctx.events.sort_values("event_date")
        first_by_ticker = events.groupby("ticker")["event_date"].min()
        some_ticker = ctx.tickers()[0]
        sub = out.xs(some_ticker, level="ticker")
        first_event = pd.Timestamp(first_by_ticker[some_ticker])
        if first_event > sub.index[0]:
            assert sub.loc[: first_event - pd.Timedelta(days=1)].isna().all()
        # después de eventos, fuera de ventana, el valor es 0 (señal agotada)
        assert (sub.dropna() == 0.0).sum() > 0


# ---------------------------------------------------------------------------
# 6. SURGE y doble sorpresa
# ---------------------------------------------------------------------------


def _buyback_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    """17 trimestres: ingresos totales estancados, recompra fuerte al final.

    Reproduce el experimento del informe (§3): con ingresos POR ACCIÓN la
    sorpresa final es positiva (la recompra concentra los mismos ingresos en
    menos acciones); con ingresos totales, no hay sorpresa positiva que
    encontrar. El error de usar totales es silencioso y de signo.
    """
    n = 17
    period_ends = pd.date_range("2019-03-31", periods=n, freq="QE")
    revenue = 1000.0 * (1.0 + 0.002) ** np.arange(n)  # crecimiento plano
    shares = np.full(n, 100.0)
    shares[-4:] = 100.0 * (0.96 ** np.arange(1, 5))  # recompra 4% por trimestre
    fundamentals = pd.DataFrame(
        {
            "ticker": "BUYB",
            "period_end": period_ends,
            "revenue": revenue,
            "shares_diluted": shares,
        }
    )
    events = pd.DataFrame(
        {
            "ticker": "BUYB",
            "period_end": period_ends,
            "tradable_date": period_ends + pd.Timedelta(days=40),
        }
    )
    return fundamentals, events


class TestRevenueSurprise:
    def test_per_share_vs_total_sign(self) -> None:
        fundamentals, events = _buyback_fixture()
        table = revenue_surprise_events(fundamentals, events)
        last = table.iloc[-1]
        assert last["surge"] > 0.5, "la recompra debe producir sorpresa positiva por acción"

        # el mismo cálculo sobre ingresos TOTALES no ve la recompra
        merged = events.merge(fundamentals, on=["ticker", "period_end"])
        foster_total = foster_surprise_events(merged, value_col="revenue")
        assert abs(foster_total["ue"].iloc[-1]) < 1.0  # sin recompra no hay señal comparable
        table_ps = table["ue"].iloc[-1]
        assert table_ps > foster_total["ue"].iloc[-1] / 1000.0  # escala por acción

    def test_panel_on_synthetic(self, ctx: FactorContext) -> None:
        out = RevenueSurprise().compute(ctx)
        assert out.notna().sum() > 0
        assert list(out.index.names) == ["date", "ticker"]

    def test_zero_shares_is_nan(self) -> None:
        fundamentals, events = _buyback_fixture()
        fundamentals.loc[fundamentals.index[-1], "shares_diluted"] = 0.0
        table = revenue_surprise_events(fundamentals, events)
        assert np.isnan(table["surge"].iloc[-1])


class TestDoubleSurprise:
    def test_agreement_rule(self) -> None:
        idx = pd.MultiIndex.from_product(
            [pd.to_datetime(["2022-01-03"]), list("ABCDE")], names=["date", "ticker"]
        )
        sue = pd.Series([2.0, -1.0, 0.5, np.nan, 1.0], index=idx)
        surge = pd.Series([1.0, -3.0, -0.5, 1.0, np.nan], index=idx)
        out = double_surprise(sue, surge)
        assert out.iloc[0] == pytest.approx(1.0)   # acuerdo +: min(2, 1)
        assert out.iloc[1] == pytest.approx(-1.0)  # acuerdo -: -min(1, 3)
        assert out.iloc[2] == 0.0                  # desacuerdo -> 0
        assert np.isnan(out.iloc[3]) and np.isnan(out.iloc[4])

    def test_factor_on_synthetic(self, ctx: FactorContext) -> None:
        out = DoubleSurprise(
            sue_factor=AnalystSUE(use_snapshots=False)
        ).compute(ctx)
        valid = out.dropna()
        assert len(valid) > 0
        assert (valid == 0.0).sum() > 0  # los desacuerdos existen y puntúan 0


# ---------------------------------------------------------------------------
# 7. Revisiones
# ---------------------------------------------------------------------------


class TestRevisions:
    def test_momentum_computes_on_synthetic_vintages(self, ctx: FactorContext) -> None:
        assert has_revision_vintages(ctx.estimates)
        out = RevisionMomentum(months=1).compute(ctx)
        valid = out.dropna()
        assert len(valid) > 1000
        # walk-down (Richardson-Teoh-Wysocki): el consenso se revisa a la baja
        # de media, así que el momentum medio de revisiones debe ser negativo
        assert valid.mean() < 0.0

    def test_momentum_without_vintages_warns_and_returns_nan(self, ctx: FactorContext) -> None:
        """Caso consenso_master: sin fotos históricas NO se inventan revisiones."""
        final_only = (
            ctx.estimates.sort_values("as_of").groupby(["ticker", "period_end"]).tail(1)
        )
        assert not has_revision_vintages(final_only)
        with pytest.warns(NoVintagesWarning):
            out = RevisionMomentum(months=1).compute(
                FactorContext(
                    dates=ctx.dates,
                    universe=ctx.universe,
                    prices=ctx.prices,
                    fundamentals=ctx.fundamentals,
                    estimates=final_only,
                    events=ctx.events,
                    calendar=ctx.calendar,
                )
            )
        assert out.isna().all()
        assert len(out) == len(ctx.dates) * len(ctx.tickers())

    def test_diffusion_bounded_and_min_revisions(self, ctx: FactorContext) -> None:
        out = RevisionDiffusion(months=3, min_revisions=3).compute(ctx)
        valid = out.dropna()
        assert len(valid) > 0
        assert valid.between(-1.0, 1.0).all()

    def test_dispersion_sign_is_negative(self, ctx: FactorContext) -> None:
        """Diether-Malloy-Scherbina: más dispersión = menos retorno => factor <= 0."""
        out = AnalystDispersion().compute(ctx)
        valid = out.dropna()
        assert len(valid) > 1000
        assert (valid <= 0.0).all()


# ---------------------------------------------------------------------------
# 8. Guidance
# ---------------------------------------------------------------------------


class TestGuidance:
    def test_categorical_scores_ordering(self) -> None:
        actions = pd.Series(["raised", "affirmed", "lowered", "withdrawn", "initiated"])
        scores = score_guidance_actions(actions)
        assert scores[0] > scores[4] > scores[1] > scores[3] > scores[2]

    def test_unknown_label_is_nan_with_warning(self) -> None:
        with pytest.warns(UserWarning, match="no reconocidas"):
            scores = score_guidance_actions(pd.Series(["raised", "quux"]))
        assert scores[0] == 1.0
        assert np.isnan(scores[1])

    def test_missing_table_raises_provider_unavailable(self, ctx: FactorContext) -> None:
        with pytest.raises(ProviderUnavailable):
            GuidanceChange(guidance=None).compute(ctx)

    def test_factor_nan_when_no_data_and_lives_horizon(self, ctx: FactorContext) -> None:
        tickers = ctx.tickers()
        guided, unguided = tickers[0], tickers[1]
        anchor = ctx.dates[30]
        guidance = pd.DataFrame(
            {
                "ticker": [guided],
                "announced_at": [anchor + pd.Timedelta(hours=21)],
                "session": ["amc"],
                "action": ["raised"],
            }
        )
        out = GuidanceChange(guidance=guidance, horizon=10, decay=None).compute(ctx)
        # ticker sin guía: NaN explícito en TODO su historial, jamás 0
        assert out.xs(unguided, level="ticker").isna().all()
        sub = out.xs(guided, level="ticker")
        live = sub.dropna()
        assert len(live) == 10
        assert (live == 1.0).all()
        # AMC: la primera sesión viva es la SIGUIENTE al anuncio
        assert live.index[0] == pd.Timestamp(ctx.calendar.next_session(anchor.date()))
        # fuera de la ventana la guía caduca a NaN, no a 0
        assert sub.loc[live.index[-1] + pd.Timedelta(days=1) :].isna().all()

    def test_guidance_surprise_exact(self) -> None:
        guidance = pd.DataFrame(
            {
                "ticker": ["X"],
                "tradable_date": [pd.Timestamp("2022-03-01")],
                "action": ["raised"],
                "guide_low": [2.0],
                "guide_high": [2.4],
                "consensus_prev": [2.0],
                "price": [100.0],
            }
        )
        gs = guidance_surprise(guidance)
        assert gs.iloc[0] == pytest.approx((2.2 - 2.0) / 100.0)


# ---------------------------------------------------------------------------
# 9. Dataset REAL: consenso_master.parquet
# ---------------------------------------------------------------------------

SAMPLE_TICKERS = ["AAPL", "MSFT", "JNJ", "XOM", "WMT", "KO", "PG", "CAT"]


@pytest.fixture(scope="module")
def real_events() -> pd.DataFrame:
    with pytest.warns(UserWarning, match="fiscal_period_end"):
        return load_consensus_events(tickers=SAMPLE_TICKERS)


class TestConsensusMaster:
    def test_parses_and_dedupes(self, real_events: pd.DataFrame) -> None:
        assert len(real_events) > 400
        assert set(real_events["ticker"]) <= set(SAMPLE_TICKERS)
        # una fila por evento: la deduplicación entre fuentes funcionó
        assert not real_events.duplicated(["ticker", "period_end"]).any()
        assert real_events["event_id"].is_unique
        assert real_events.attrs["n_duplicates_collapsed"] > 0
        assert "pit_warning" in real_events.attrs
        assert "consenso FINAL" in real_events.attrs["pit_warning"]

    def test_sessions_derived_from_report_time(self, real_events: pd.DataFrame) -> None:
        sessions = set(real_events["session"])
        assert "bmo" in sessions and "amc" in sessions
        announced_day = pd.DatetimeIndex(
            pd.to_datetime(real_events["announced_at"])
        ).normalize()
        event_day = pd.DatetimeIndex(real_events["event_date"]).normalize()
        bmo = real_events["session"] == "bmo"
        amc = real_events["session"].isin(["amc", "unknown"])
        # BMO: negociable la misma sesión (o la siguiente si el día no fue sesión)
        assert (event_day[bmo.to_numpy()] >= announced_day[bmo.to_numpy()]).all()
        same_day = (event_day[bmo.to_numpy()] == announced_day[bmo.to_numpy()]).mean()
        assert same_day > 0.95
        # AMC/UNKNOWN: negociable estrictamente después del día del anuncio
        assert (event_day[amc.to_numpy()] > announced_day[amc.to_numpy()]).all()

    def test_session_propagation_across_sources(self, real_events: pd.DataFrame) -> None:
        """Las filas sin `report_time` heredan la sesión de otra fuente del
        mismo (ticker, report_date): sin propagación casi todo sería UNKNOWN."""
        assert real_events.attrs["n_session_filled_across_sources"] > 0
        known = (real_events["session"] != "unknown").mean()
        assert known > 0.5

    def test_analyst_sue_on_real_data(self, real_events: pd.DataFrame) -> None:
        """El SUE de analistas (base sigma histórica) se calcula sobre datos reales."""
        table = analyst_sue_events(real_events, basis=SurpriseBasis.SIGMA)
        coverage = table["sue"].notna().mean()
        assert coverage > 0.7, f"cobertura de SUE real insuficiente: {coverage:.2%}"
        # cada ticker de la muestra obtiene SUE en la mayoría de sus eventos
        per_ticker = table.assign(ok=table["sue"].notna()).groupby("ticker")["ok"].mean()
        assert (per_ticker > 0.5).all()
        # magnitudes plausibles de un estadístico estandarizado
        valid = table["sue"].dropna()
        assert valid.abs().median() < 5.0
        # la sorpresa usada es la del dataset: (A - M) reproducible
        recomputed = real_events["eps_actual"] - real_events["eps_estimate"]
        both = pd.concat([table["surprise"], recomputed.rename("raw")], axis=1).dropna()
        assert (both["surprise"] - both["raw"]).abs().max() < 1e-9

    def test_abs_estimate_matches_surprise_pct(self, real_events: pd.DataFrame) -> None:
        """La base ABS_ESTIMATE reproduce el `surprise_pct` del proveedor (x100)."""
        table = analyst_sue_events(real_events, basis=SurpriseBasis.ABS_ESTIMATE)
        merged = pd.concat(
            [table["sue"], real_events["surprise_pct"], real_events["eps_estimate"]], axis=1
        ).dropna()
        merged = merged[merged["eps_estimate"].abs() > 0.05]  # fuera del suelo
        corr = merged["sue"].corr(merged["surprise_pct"] / 100.0)
        assert corr > 0.99

    def test_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(ProviderUnavailable):
            load_consensus_events(tmp_path / "no_existe.parquet")

    def test_unknown_ticker_raises(self) -> None:
        with pytest.raises(InsufficientHistory):
            load_consensus_events(tickers=["ZZZZZZ"])


# ---------------------------------------------------------------------------
# 10. Utilidades de proyección y contexto
# ---------------------------------------------------------------------------


class TestSpreadAndContext:
    def test_partial_window_before_panel_start(self) -> None:
        """Un evento anterior al panel entra con su vida parcialmente consumida."""
        cal = get_calendar()
        sessions = cal.sessions(dt.date(2021, 6, 1), dt.date(2021, 6, 30))
        grid = sessions[4:]
        events = pd.DataFrame(
            {"ticker": ["AAPL"], "tradable_date": [sessions[2]], "value": [2.0]}
        )
        out = spread_event_values(
            events, grid, ["AAPL"], horizon=5, include_event_day=True, calendar=cal
        ).xs("AAPL", level="ticker")
        # vida = sessions[2..6]; el panel arranca en sessions[4] -> 3 sesiones vivas
        assert out.notna().sum() == 3
        assert out.iloc[:3].eq(2.0).all()
        assert out.iloc[3:].isna().all()

    def test_context_require_explicit_failure(self, ctx: FactorContext) -> None:
        empty_ctx = FactorContext(
            dates=ctx.dates,
            universe=ctx.universe,
            prices=ctx.prices,
            fundamentals=ctx.fundamentals,
            estimates=ctx.estimates,
            events=ctx.events.iloc[0:0],
            calendar=ctx.calendar,
        )
        with pytest.raises(ProviderUnavailable):
            AnalystSUE(use_snapshots=False).compute(empty_ctx)

    def test_panel_is_canonical(self, ctx: FactorContext) -> None:
        out = AnalystSUE(use_snapshots=False).compute(ctx)
        assert list(out.index.names) == ["date", "ticker"]
        assert out.index.is_monotonic_increasing
        dates = out.index.get_level_values("date")
        assert (dates == dates.normalize()).all()
        assert dates.tz is None
