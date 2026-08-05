"""Tests de integración end-to-end de los dos pipelines (`earnings_alpha.pipeline`).

Qué se verifica, en orden de importancia:

1. **Ángulo A recupera IC positivo.** El generador sintético inyecta respuesta
   al anuncio (Ball-Brown 1968) y PEAD (Bernard-Thomas 1989); un pipeline
   correcto —factores PIT, neutralización por fecha, combinación, IC con
   retardo de ejecución— debe recuperar una IC combinada positiva. Si algún
   eslabón introduce un desplazamiento temporal erróneo, la señal se degrada y
   este test lo delata.
2. **Ángulo B distingue los eventos con filtración.** Contra la verdad-terreno
   (`SyntheticMarket.leaked_event_ids()`): el AUC del score de intensidad debe
   ser significativamente > 0,5 y el score medio de los eventos filtrados,
   mayor que el del grupo de control.
3. **Los guardarraíles PIT del orquestador.** Pre-posicionarse sin declarar la
   conocibilidad del calendario es `LookAheadError`; una etiqueta que empieza
   antes de T es `DataQualityError`; y todos los errores de configuración
   fallan ANTES de pagar el cómputo de features (fail-fast).
4. **Fallo explícito, degradación declarada.** Un factor sin fuente lanza
   `ProviderUnavailable`; con ``on_factor_error="skip"`` queda registrado en
   `skipped_factors`, nunca oculto.

Los fixtures son de módulo: cada pipeline pesado se ejecuta una sola vez y
varios tests inspeccionan el mismo resultado. Lo marcado `@pytest.mark.slow`
(pipelines completos con SurpriseModel, cesta de 6 factores, ejemplos vía
subprocess) queda fuera de la pasada rápida.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from earnings_alpha.data.synthetic import SyntheticMarket
from earnings_alpha.errors import (
    ConfigError,
    DataQualityError,
    LookAheadError,
    ProviderUnavailable,
)
from earnings_alpha.pipeline import (
    DEFAULT_CONTINUOUS_FACTORS,
    ContinuousPipelineResult,
    EventPipelineResult,
    run_continuous_pipeline,
    run_event_pipeline,
)
from earnings_alpha.stats import ICSummary

REPO_ROOT = Path(__file__).resolve().parent.parent

# Universos reducidos y rangos cortos: la pasada rápida completa (ambos
# pipelines) debe caber en decenas de segundos, no minutos.
MKT_A = {"n_tickers": 24, "start": "2021-01-04", "end": "2022-12-30"}
MKT_B = {"n_tickers": 20, "start": "2020-01-02", "end": "2022-12-30"}
WIDE = {"n_tickers": 24, "start": "2019-01-02", "end": "2023-12-29"}


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def mkt_a() -> SyntheticMarket:
    """Mercado del ángulo A: sin filtración, con respuesta al anuncio y PEAD."""
    return SyntheticMarket(seed=20260803, **MKT_A)


@pytest.fixture(scope="module")
def res_a(mkt_a: SyntheticMarket) -> ContinuousPipelineResult:
    """Pipeline continuo con la pareja de factores de sorpresa (rápida)."""
    return run_continuous_pipeline(
        mkt_a,
        factors=("sue_analyst", "pead"),
        max_weight=0.25,  # universo de 24 nombres: el tope de 500 nombres no aplica
    )


@pytest.fixture(scope="module")
def mkt_b() -> SyntheticMarket:
    """Mercado del ángulo B con filtración abundante (verdad-terreno conocida)."""
    return SyntheticMarket(seed=1234, leak_fraction=0.35, **MKT_B)


@pytest.fixture(scope="module")
def res_b(mkt_b: SyntheticMarket) -> EventPipelineResult:
    """Pipeline de eventos sin modelo (el panel corto no da para CV purgada)."""
    return run_event_pipeline(
        mkt_b,
        model_mode="skip",
        entry_offsets=(-1, 0, 1),
        exit_offsets=(1, 5),
        # Calendario sintético: público por adelantado, declarable (§8.3).
        calendar_known_in_advance=True,
        # Panel de 3 años: ventana de estimación acortada (sigue terminando
        # estrictamente antes de la de evento) y estudio [-5, +20].
        estimation=(-160, -31),
        pre=5,
        post=20,
    )


# --------------------------------------------------------------------------- #
# Ángulo A: pipeline continuo                                                 #
# --------------------------------------------------------------------------- #


class TestAnguloA:
    def test_recupera_ic_positivo(self, res_a: ContinuousPipelineResult) -> None:
        """El pipeline entero recupera la señal inyectada: IC combinada > 0.

        Es el test central del ángulo A: si cualquier eslabón (proyección PIT
        del factor, neutralización, combinación, retardo de ejecución del
        retorno forward) desplazara mal una fecha, la IC se hundiría.
        """
        s = res_a.ic_combined
        assert isinstance(s, ICSummary)
        assert s.mean > 0.01, f"IC combinada {s.mean:.4f}: la señal inyectada no se recupera"
        assert s.n_periods > 100
        lo, hi = s.ci
        assert lo < s.mean < hi, "el IC de confianza no encierra la media"
        assert np.isfinite(s.t_stat)

    def test_ic_por_factor_con_banda(self, res_a: ContinuousPipelineResult) -> None:
        """Cada factor reporta su IC con banda (regla §3.8: nada sin intervalo)."""
        assert set(res_a.ic_by_factor) == {"sue_analyst_sigma", "pead"}
        for name, s in res_a.ic_by_factor.items():
            lo, hi = s.ci
            assert np.isfinite(lo) and np.isfinite(hi) and lo < hi, name
            assert s.mean > 0.0, f"{name}: IC {s.mean:.4f} <= 0 sobre el sintético"
            assert s.significance.method, f"{name}: sin método de corrección declarado"

    def test_estructura_de_paneles(
        self, mkt_a: SyntheticMarket, res_a: ContinuousPipelineResult
    ) -> None:
        """Paneles canónicos (date, ticker); fechas dentro de las sesiones."""
        for panel in (res_a.factor_panel, res_a.conditioned):
            assert list(panel.index.names) == ["date", "ticker"]
        assert list(res_a.factor_panel.columns) == ["sue_analyst_sigma", "pead"]
        assert res_a.combined.name == "combined_score"
        dates = res_a.combined.index.get_level_values("date").unique()
        assert set(dates).issubset(set(mkt_a.sessions))
        # La señal combinada existe de verdad: sección cruzada no degenerada.
        by_date = res_a.combined.groupby(level="date").count()
        assert int(by_date.max()) >= 12

    def test_neutralizacion_elimina_media_sectorial(
        self, mkt_a: SyntheticMarket, res_a: ContinuousPipelineResult
    ) -> None:
        """Tras neutralizar, la media por sector es ~0 en las fechas pobladas."""
        sectors = mkt_a.sectors()
        cond = res_a.conditioned["sue_analyst_sigma"].dropna()
        frame = cond.reset_index()
        frame["sector"] = frame["ticker"].map(sectors)
        # Fechas con sección cruzada completa (las 24 empresas).
        counts = frame.groupby("date")["sue_analyst_sigma"].count()
        full_dates = counts[counts >= 20].index
        sub = frame[frame["date"].isin(full_dates)]
        assert len(sub) > 0
        by_sector = sub.groupby(["date", "sector"])["sue_analyst_sigma"].mean()
        # Sectores con >= 3 nombres: media residual despreciable frente a sigma 1.
        sizes = sub.groupby(["date", "sector"])["sue_analyst_sigma"].count()
        big = by_sector[sizes >= 3]
        assert float(big.abs().mean()) < 0.35

    def test_backtest_coherente(self, res_a: ContinuousPipelineResult) -> None:
        """El backtest devuelve series consistentes y costes desglosados > 0."""
        bt = res_a.backtest
        assert len(bt.returns) > 50
        assert np.isfinite(bt.returns.to_numpy()).all()
        assert (bt.nav > 0).all()
        totals = bt.total_costs
        assert totals["total"] > 0.0
        for comp in ("spread", "impact", "commission", "borrow"):
            assert totals[comp] >= 0.0
        assert abs(totals["total"] - totals[["spread", "impact", "commission", "borrow"]].sum()) < 1e-12
        summary = bt.summary()
        sr = summary["sharpe"]
        assert sr.ci_low < sr.sharpe_annualized < sr.ci_high
        assert summary["n_silent_delistings"] == 0
        assert (bt.turnover >= 0).all()

    def test_params_reproducibles(self, res_a: ContinuousPipelineResult) -> None:
        """El eco de parámetros permite reproducir la pasada."""
        p = res_a.params
        assert p["factors"] == ["sue_analyst_sigma", "pead"]
        assert p["combine_method"] == "weights"
        assert p["n_tickers"] == 24
        assert p["membership_filter"] == "ok"
        assert res_a.skipped_factors == {}
        assert res_a.ic_errors == {}

    def test_fallo_explicito_configuracion(self, mkt_a: SyntheticMarket) -> None:
        """Errores de configuración: ruidosos y tempranos."""
        with pytest.raises(ConfigError, match="desconocido"):
            run_continuous_pipeline(mkt_a, factors=("factor_inexistente",))
        with pytest.raises(ConfigError, match="vacía"):
            run_continuous_pipeline(mkt_a, factors=())
        with pytest.raises(ConfigError, match="combine_method"):
            run_continuous_pipeline(
                mkt_a, factors=("pead",), combine_method="promedio_magico"  # type: ignore[arg-type]
            )
        with pytest.raises(ConfigError, match="exposiciones no construibles"):
            run_continuous_pipeline(
                mkt_a, factors=("pead",), neutralize_by=("sector", "beta")
            )

    def test_factor_sin_fuente_lanza_o_se_registra(self, mkt_a: SyntheticMarket) -> None:
        """`guidance_change` no tiene fuente sintética: raise por defecto, skip declarado."""
        with pytest.raises(ProviderUnavailable):
            run_continuous_pipeline(mkt_a, factors=("pead", "guidance_change"))
        res = run_continuous_pipeline(
            mkt_a,
            factors=("pead", "guidance_change"),
            on_factor_error="skip",
            max_weight=0.25,
        )
        assert list(res.factor_panel.columns) == ["pead"]
        assert "guidance_change" in res.skipped_factors
        assert res.skipped_factors["guidance_change"].startswith("ProviderUnavailable")


# --------------------------------------------------------------------------- #
# Ángulo B: pipeline de eventos                                               #
# --------------------------------------------------------------------------- #


class TestAnguloB:
    def test_distingue_eventos_con_filtracion(self, res_b: EventPipelineResult) -> None:
        """El score de intensidad separa los eventos filtrados del control.

        Contra la verdad-terreno del generador: AUC > 0,5 con t de
        Hanley-McNeil > 2, y score medio mayor en el grupo filtrado. Es la
        validación de que el detector detecta (docstring de `data.synthetic`).
        """
        d = res_b.detection
        assert d is not None
        assert d["n_leaked"] > 30
        assert 0.15 < d["prevalence"] < 0.55
        assert d["auc"] > 0.55, f"AUC {d['auc']:.3f}: el detector no distingue"
        assert d["auc_t"] > 2.0, f"t={d['auc_t']:.2f}: separación no significativa"
        assert d["mean_score_leaked"] > d["mean_score_other"]

    def test_features_y_scores_alineados(
        self, mkt_b: SyntheticMarket, res_b: EventPipelineResult
    ) -> None:
        """Features por event_id único; scores alineados; degradación declarada."""
        feats = res_b.features
        assert feats.index.name == "event_id"
        assert feats.index.is_unique
        valid_ids = set(mkt_b.events(include_prehistory=True)["event_id"])
        assert set(feats.index).issubset(valid_ids)
        assert res_b.intensity_score.index.equals(feats.index)
        assert res_b.directional_score.index.equals(feats.index)
        assert np.isfinite(res_b.intensity_score.to_numpy(dtype=float)).sum() > 100
        # Fuentes ausentes: anotadas, no silenciadas (el sintético no trae Form 4).
        assert "form4" in feats.attrs.get("missing_sources", [])
        assert res_b.params["missing_sources"] == feats.attrs["missing_sources"]

    def test_etiquetas_pit(self, res_b: EventPipelineResult) -> None:
        """Las etiquetas empiezan en T (w0 >= 0) y cubren la mayoría de eventos."""
        labels = res_b.labels
        assert {"surprise", "surprise_sign", "event_return"}.issubset(labels.columns)
        assert res_b.params["label_window"] == (0, 1)
        assert labels["event_return"].notna().sum() > 150

    def test_rejilla_completa_y_ejecutable(self, res_b: EventPipelineResult) -> None:
        """`run_grid`: 3 entradas x 2 salidas, todas ejecutables, con bandas."""
        grid = res_b.grid
        assert list(grid.index.names) == ["entry_offset", "exit_offset"]
        assert len(grid) == 6
        assert (grid["status"] == "ok").all()
        assert (grid["n_events"] >= 100).all()
        # Media por evento con su banda de bootstrap: nada sin intervalo.
        ok = grid
        assert (ok["mean_net_ci_low"] <= ok["mean_net"]).all()
        assert (ok["mean_net"] <= ok["mean_net_ci_high"]).all()
        # La descomposición gap/intradía existe (los AMC viven en el gap).
        assert ok["holds_event_gap_frac"].between(0.0, 1.0).all()

    def test_caar_por_quintil_ordena_el_pead(self, res_b: EventPipelineResult) -> None:
        """CAAR por quintil de SUE: Q5 > Q1 al final de la ventana (PEAD inyectado)."""
        caar = res_b.caar_by_quintile
        assert caar is not None
        groups = caar.index.get_level_values("group").unique()
        assert {"Q1", "Q5"}.issubset(set(map(str, groups)))
        last_tau = int(caar.index.get_level_values("tau").max())
        assert last_tau == 20
        q1 = float(caar.loc[("Q1", last_tau), "caar"])
        q5 = float(caar.loc[("Q5", last_tau), "caar"])
        assert q5 > q1, f"CAAR Q5 ({q5:+.4f}) no supera a Q1 ({q1:+.4f})"
        assert q5 - q1 > 0.03, "el spread de PEAD inyectado no se recupera"
        # Cada quintil con tamaño honesto (min_group_size del pipeline).
        assert (caar["n"] >= 5).all()

    def test_estudio_de_eventos_estructura(self, res_b: EventPipelineResult) -> None:
        """El panel AR y el agregado AAR/CAAR traen lo que promete el contrato."""
        ar = res_b.event_study
        assert ar is not None
        assert {"event_id", "tau", "ar", "car", "bhar"}.issubset(ar.columns)
        assert int(ar["tau"].min()) == -5 and int(ar["tau"].max()) == 20
        aar = res_b.aar
        assert aar is not None
        assert {"aar", "caar", "caar_se", "caar_t", "n"}.issubset(aar.columns)
        assert (aar["n"] >= 2).all()

    def test_guardarrailes_pit_fail_fast(self, mkt_b: SyntheticMarket) -> None:
        """Los errores de planteamiento fallan ANTES de computar features.

        Pre-posicionamiento sin declarar la conocibilidad del calendario
        (`pit_and_biases.md` §8.3), etiqueta que empieza antes de T
        (tautología), y configuraciones imposibles.
        """
        with pytest.raises(LookAheadError, match="calendar_known_in_advance"):
            run_event_pipeline(
                mkt_b, entry_offsets=(-5,), exit_offsets=(1,), model_mode="skip"
            )
        with pytest.raises(DataQualityError, match="label_window"):
            run_event_pipeline(mkt_b, label_window=(-1, 1), model_mode="skip")
        with pytest.raises(ConfigError, match="model_mode"):
            run_event_pipeline(mkt_b, model_mode="magico")  # type: ignore[arg-type]
        with pytest.raises(ConfigError, match="backtest_score"):
            run_event_pipeline(
                mkt_b, model_mode="skip", backtest_score="bola_de_cristal"  # type: ignore[arg-type]
            )
        with pytest.raises(ConfigError, match="model"):
            run_event_pipeline(mkt_b, model_mode="skip", backtest_score="model")


# --------------------------------------------------------------------------- #
# Pipelines completos (lentos)                                                #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def mkt_wide() -> SyntheticMarket:
    return SyntheticMarket(seed=1234, leak_fraction=0.35, **WIDE)


@pytest.fixture(scope="module")
def res_wide(mkt_wide: SyntheticMarket) -> EventPipelineResult:
    return run_event_pipeline(
        mkt_wide,
        model_mode="evaluate",
        n_splits=4,
        entry_offsets=(-5, -1, 0, 1),
        exit_offsets=(1, 5, 20),
        calendar_known_in_advance=True,
    )


@pytest.mark.slow
class TestPipelinesCompletos:
    def test_angulo_b_modelo_oos_bate_linea_base(
        self, res_wide: EventPipelineResult
    ) -> None:
        """La CV purgada produce un OOS honesto que bate a la línea base."""
        report = res_wide.model_report
        assert report is not None
        assert report.n_splits == 4
        assert report.metrics["auc"] > 0.6, "AUC OOS del SurpriseModel demasiado bajo"
        assert report.beats_baseline, "el modelo no bate a la constante de prevalencia"
        # Cada evento cae exactamente en un test: sin huecos masivos.
        oos = report.oos_prediction
        assert float(np.isfinite(oos.to_numpy(dtype=float)).mean()) > 0.95

    def test_angulo_b_deteccion_significativa(self, res_wide: EventPipelineResult) -> None:
        d = res_wide.detection
        assert d is not None
        assert d["auc"] > 0.6
        assert d["auc_t"] > 3.0

    def test_angulo_b_pre_posicionamiento_y_caar(
        self, res_wide: EventPipelineResult
    ) -> None:
        """Rejilla con entradas pre-anuncio declaradas y CAAR por quintil de SUE."""
        grid = res_wide.grid
        assert len(grid) == 12
        assert (grid["status"] == "ok").all()
        assert res_wide.params["n_grid_trials"] == 12
        caar = res_wide.caar_by_quintile
        assert caar is not None
        last_tau = int(caar.index.get_level_values("tau").max())
        spread = float(caar.loc[("Q5", last_tau), "caar"]) - float(
            caar.loc[("Q1", last_tau), "caar"]
        )
        assert spread > 0.05, f"spread CAAR Q5-Q1 {spread:+.4f} demasiado bajo"

    def test_angulo_b_mercado_limpio_sin_filtracion(self) -> None:
        """Con leak_fraction=0 el pipeline degrada con elegancia: 0 filtrados, AUC NaN."""
        clean = SyntheticMarket(seed=1234, leak_fraction=0.0, **MKT_B)
        res = run_event_pipeline(
            clean,
            model_mode="skip",
            entry_offsets=(0,),
            exit_offsets=(1,),
            event_study=False,
        )
        d = res.detection
        assert d is not None
        assert d["n_leaked"] == 0
        assert np.isnan(d["auc"])
        assert res.caar_by_quintile is None and res.event_study is None

    def test_angulo_a_cesta_completa_ic_weighted(self) -> None:
        """La cesta de 6 factores por defecto con combinación ic_weighted."""
        market = SyntheticMarket(seed=7, n_tickers=40, start="2020-01-02", end="2023-12-29")
        res = run_continuous_pipeline(
            market,
            combine_method="ic_weighted",
            max_weight=0.10,
        )
        assert res.params["factors"] == [
            "sue_analyst_sigma",
            "pead",
            "revenue_surprise",
            "earnings_yield",
            "piotroski_f",
            "accruals_cf",
        ]
        assert len(DEFAULT_CONTINUOUS_FACTORS) == 6
        assert res.skipped_factors == {}
        # Todos los factores con IC estimable o error registrado; ninguno perdido.
        assert len(res.ic_by_factor) + len(res.ic_errors) == 6
        assert np.isfinite(res.ic_combined.mean)
        assert len(res.backtest.returns) > 100
        summary = res.summary()
        assert "PIPELINE CONTINUO" in summary
        assert "Sharpe" in summary


@pytest.mark.slow
class TestEjemplos:
    """Los tres scripts de `examples/` se ejecutan tal cual y terminan bien."""

    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, *args],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=560,
            check=False,
        )

    def test_angulo_a_factores(self) -> None:
        proc = self._run("examples/angulo_a_factores.py", "--rapido")
        assert proc.returncode == 0, proc.stderr[-2000:]
        assert "PIPELINE CONTINUO" in proc.stdout
        assert "Sharpe anualizado" in proc.stdout

    def test_angulo_b_eventos(self) -> None:
        proc = self._run("examples/angulo_b_eventos.py", "--rapido")
        assert proc.returncode == 0, proc.stderr[-2000:]
        assert "PIPELINE DE EVENTOS" in proc.stdout
        assert "AUC del score de intensidad" in proc.stdout
        assert "CAAR por quintil" in proc.stdout

    def test_sorpresas_reales(self) -> None:
        proc = self._run("examples/sorpresas_reales.py")
        assert proc.returncode == 0, proc.stderr[-2000:]
        assert "ADVERTENCIA POINT-IN-TIME" in proc.stdout
        assert "Beat:" in proc.stdout
        assert "SUE DE ANALISTAS" in proc.stdout
