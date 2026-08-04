"""Tests de `events.surprise_model`: el modelo predictivo de sorpresa (contrato §3.5).

Cobertura, en orden de importancia:

1. **El modelo detecta la relación inyectada y bate a las líneas base.** Sobre el
   mercado sintético con filtración (`leak_fraction=0.4`), el clasificador de
   signo de sorpresa supera con claridad a la clase mayoritaria (AUC >> 0,5,
   Brier mejor que la constante de prevalencia) y los modelos de magnitud/retorno
   baten al consenso (R² OOS de Campbell-Thompson > 0 con rank-IC significativo).
2. **Prueba de ausencia de fuga en el pipeline.** Con `leak_fraction=0` la
   sorpresa es aleatoria respecto a las features: el AUC OOS no se distingue de
   0,5, el R² OOS es <= 0 y `beats_baseline` es False. Se añade el placebo de
   etiquetas barajadas: si la CV purgada filtrara el test al train, un modelo
   entrenado sobre etiquetas permutadas mostraría AUC > 0,5.
3. **Validación temporal estricta.** `evaluate` solo funciona con fechas (la CV
   es `stats.validation.PurgedKFold`), no existe ningún parámetro `shuffle`, y
   la purga consume observaciones de verdad (se verifica la contabilidad
   train + test + purgadas = N por fold).
4. **Interfaz única** para ridge y gradient boosting, permutation importance
   FUERA de muestra, calibración de probabilidades con curva y ECE, y los
   fallos explícitos del repo (`DataQualityError`, `InsufficientHistory`,
   `ConfigError`).

Las señales de opciones proceden de `events.options_signals` (módulo de otro
propietario). El generador sintético inyecta buena parte de la huella de
*magnitud* por ese canal, así que los tests que dependen de él se saltan con
aviso si el módulo no está disponible — el resto es incondicional.
"""

from __future__ import annotations

import inspect
import math

import numpy as np
import pandas as pd
import pytest

from earnings_alpha.data.synthetic import LeakSpec, SyntheticConfig, SyntheticMarket
from earnings_alpha.errors import ConfigError, DataQualityError, InsufficientHistory
from earnings_alpha.events.preevent import EventContext, PreEventFeatures
from earnings_alpha.events.surprise_model import (
    SurpriseModel,
    SurpriseModelReport,
    brier_score,
    calibration_curve,
    compare_estimators,
    event_labels,
    expected_calibration_error,
    feature_matrix,
    roc_auc,
    spearman_ic,
)

# --------------------------------------------------------------------------- fixtures

WIDE = {"n_tickers": 40, "start": "2019-01-02", "end": "2023-12-29"}

# Mercado con respuesta de anuncio amplificada: el retorno del día del evento
# tiene un ratio señal/ruido bajo por diseño (jump_vol domina), y este config
# hace la relación inyectada detectable con ~800 eventos sin cambiar el pipeline.
RESPONSE_CONFIG = SyntheticConfig(
    event_response=0.06, leak=LeakSpec(price_drift_total=0.055)
)


def _dataset(mkt: SyntheticMarket) -> tuple[pd.DataFrame, pd.DataFrame]:
    feats = PreEventFeatures().compute(EventContext.from_synthetic(mkt))
    labels = event_labels(mkt.events(), mkt.prices(), market=mkt.market_index())
    labels = labels.dropna(subset=["surprise", "event_return"])
    idx = feats.index.intersection(labels.index)
    return feats.loc[idx], labels.loc[idx]


@pytest.fixture(scope="module")
def leaky() -> SyntheticMarket:
    return SyntheticMarket(seed=1234, leak_fraction=0.4, **WIDE)


@pytest.fixture(scope="module")
def clean() -> SyntheticMarket:
    return SyntheticMarket(seed=1234, leak_fraction=0.0, **WIDE)


@pytest.fixture(scope="module")
def resp() -> SyntheticMarket:
    return SyntheticMarket(seed=1234, leak_fraction=0.4, config=RESPONSE_CONFIG, **WIDE)


@pytest.fixture(scope="module")
def data_leaky(leaky: SyntheticMarket) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _dataset(leaky)


@pytest.fixture(scope="module")
def data_clean(clean: SyntheticMarket) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _dataset(clean)


@pytest.fixture(scope="module")
def data_resp(resp: SyntheticMarket) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _dataset(resp)


@pytest.fixture(scope="module")
def report_sign_leaky(
    leaky: SyntheticMarket, data_leaky: tuple[pd.DataFrame, pd.DataFrame]
) -> SurpriseModelReport:
    """Informe OOS del clasificador de signo sobre el mercado con filtración."""
    x_mat, labels = data_leaky
    model = SurpriseModel(target="surprise", objective="sign", estimator="ridge")
    return model.evaluate(
        x_mat,
        labels["surprise_sign"],
        labels["event_date"],
        timeline=leaky.sessions,
        n_permutation_repeats=5,
    )


def _has_options(x_mat: pd.DataFrame) -> bool:
    """True si las features de opciones traen datos (módulo de otro propietario)."""
    return "vol_spread" in x_mat.columns and bool(
        np.isfinite(x_mat["vol_spread"].to_numpy(dtype=float)).any()
    )


def _skip_without_options(x_mat: pd.DataFrame) -> None:
    if not _has_options(x_mat):
        pytest.skip(
            "sin events.options_signals no hay canal de opciones y la huella de "
            "magnitud inyectada por el generador es indetectable por diseño"
        )


# ===========================================================================
# 1. Etiquetas
# ===========================================================================


def _label_fixture() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Panel manual de un ticker con retornos conocidos para verificar la etiqueta."""
    dates = pd.bdate_range("2021-03-01", periods=6)
    closes = [100.0, 102.0, 101.0, 105.0, 104.0, 106.0]
    rows = [
        {"date": d, "ticker": "ZZ", "close": c, "volume": 1000.0}
        for d, c in zip(dates, closes, strict=True)
    ]
    prices = pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()
    market = pd.Series(0.001, index=dates)  # retorno log constante del índice
    events = pd.DataFrame(
        {
            "event_id": ["ZZ:ev"],
            "ticker": ["ZZ"],
            "event_date": [dates[3]],
            "sue": [1.7],
            "eps_surprise": [-0.25],
        }
    )
    return prices, market, events


class TestEventLabels:
    def test_valor_exacto_del_retorno_de_evento(self) -> None:
        """CAR[0,1] ajustado por mercado calculado a mano, bit a bit."""
        prices, market, events = _label_fixture()
        out = event_labels(events, prices, market=market)
        expected = (
            (math.log(105.0 / 101.0) - 0.001) + (math.log(104.0 / 105.0) - 0.001)
        )
        assert float(out.loc["ZZ:ev", "event_return"]) == pytest.approx(expected, rel=1e-12)
        assert float(out.loc["ZZ:ev", "surprise"]) == pytest.approx(1.7)
        # El signo sale de eps_surprise (beat/miss bruto), no del SUE demeaned.
        assert float(out.loc["ZZ:ev", "surprise_sign"]) == -1.0

    def test_ventana_incompleta_da_nan(self) -> None:
        """Un evento en la última sesión no tiene T+1: etiqueta NaN, no sesgada."""
        prices, market, events = _label_fixture()
        ev = events.assign(event_date=[prices.index.get_level_values("date")[-1]])
        out = event_labels(ev, prices, market=market)
        assert np.isnan(out.loc["ZZ:ev", "event_return"])

    def test_ventana_que_empieza_antes_de_t_es_ilegal(self) -> None:
        """w0 < 0 solaparía la ventana de features: tautología prohibida."""
        prices, market, events = _label_fixture()
        with pytest.raises(DataQualityError):
            event_labels(events, prices, market=market, return_window=(-1, 1))

    def test_columna_de_sorpresa_ausente(self) -> None:
        prices, _, events = _label_fixture()
        with pytest.raises(DataQualityError, match="sorpresa"):
            event_labels(events.drop(columns=["sue"]), prices, surprise_col="sue")

    def test_todo_nan_lanza(self) -> None:
        _, _, events = _label_fixture()
        with pytest.raises(InsufficientHistory):
            event_labels(events.assign(sue=np.nan))

    def test_estructura_sobre_el_sintetico(
        self, leaky: SyntheticMarket, data_leaky: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        _, labels = data_leaky
        assert labels.index.name == "event_id"
        assert labels.index.is_unique
        assert set(np.sign(labels["surprise_sign"].dropna().unique())) <= {-1.0, 0.0, 1.0}
        # El walk-down del generador produce mayoría de beats: la línea base
        # "clase mayoritaria" es exigente de verdad, no un 50 %.
        assert (labels["surprise_sign"] > 0).mean() > 0.6


# ===========================================================================
# 2. Matriz de features
# ===========================================================================


class TestFeatureMatrix:
    def test_excluye_metadatos_y_registra_all_nan(
        self, data_leaky: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        x_raw, _ = data_leaky
        out = feature_matrix(x_raw)
        for col in ("ticker", "event_date", "sector", "cohort", "log_mktcap"):
            assert col not in out.columns
        # El sintético no publica Form 4: esas columnas degradan y se registran.
        assert "insider_net_buy_form4" in out.attrs["dropped_all_nan"]
        assert "insider_net_buy_form4" not in out.columns
        assert all(pd.api.types.is_float_dtype(out[c]) for c in out.columns)

    def test_indice_duplicado_lanza(self, data_leaky: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        x_raw, _ = data_leaky
        dup = pd.concat([x_raw.head(3), x_raw.head(3)])
        with pytest.raises(DataQualityError):
            feature_matrix(dup)

    def test_sin_columnas_numericas_lanza(self) -> None:
        frame = pd.DataFrame({"ticker": ["A", "B"]}, index=["e1", "e2"])
        with pytest.raises(DataQualityError):
            feature_matrix(frame)


# ===========================================================================
# 3. Interfaz fit/predict (contrato §3.5)
# ===========================================================================


class TestInterface:
    def test_misma_interfaz_para_ambos_estimadores(
        self, data_leaky: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        x_mat, labels = data_leaky
        for est in ("ridge", "gbm"):
            model = SurpriseModel(objective="sign", estimator=est)
            fitted = model.fit(x_mat, labels["surprise_sign"], labels["event_date"])
            assert fitted is model
            pred = model.predict(x_mat)
            assert isinstance(pred, pd.Series)
            assert pred.index.equals(x_mat.index)
            assert float(pred.min()) >= 0.0 and float(pred.max()) <= 1.0
            hard = model.predict_class(x_mat)
            assert set(hard.unique()) <= {-1.0, 1.0}

    def test_regresion_devuelve_magnitud(
        self, data_leaky: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        x_mat, labels = data_leaky
        for est in ("ridge", "gbm"):
            model = SurpriseModel(objective="magnitude", estimator=est)
            pred = model.fit(x_mat, labels["surprise"], labels["event_date"]).predict(x_mat)
            assert np.isfinite(pred.to_numpy()).all()
            with pytest.raises(DataQualityError):
                model.predict_class(x_mat)

    def test_predict_sin_fit_lanza(self, data_leaky: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        x_mat, _ = data_leaky
        with pytest.raises(DataQualityError, match="fit"):
            SurpriseModel().predict(x_mat)

    def test_fechas_obligatorias_y_prohibicion_del_kfold(
        self, data_leaky: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        """Sin `groups_by_date` no hay modelo: el K-fold aleatorio está vetado."""
        x_mat, labels = data_leaky
        with pytest.raises(DataQualityError, match="K-fold aleatorio"):
            SurpriseModel().fit(x_mat, labels["surprise_sign"], None)
        # Y no existe ningún parámetro de barajado en la evaluación.
        params = inspect.signature(SurpriseModel.evaluate).parameters
        assert "shuffle" not in params
        assert "random_state" not in params

    def test_y_con_nan_lanza(self, data_leaky: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        x_mat, labels = data_leaky
        y_bad = labels["surprise"].copy()
        y_bad.iloc[0] = np.nan
        with pytest.raises(DataQualityError, match="no finitos"):
            SurpriseModel(objective="magnitude").fit(x_mat, y_bad, labels["event_date"])

    def test_y_desalineada_lanza(self, data_leaky: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        x_mat, labels = data_leaky
        with pytest.raises(DataQualityError, match="no cubre"):
            SurpriseModel().fit(
                x_mat, labels["surprise_sign"].iloc[:-5], labels["event_date"]
            )

    def test_pocos_eventos_lanza(self, data_leaky: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        x_mat, labels = data_leaky
        with pytest.raises(InsufficientHistory):
            SurpriseModel().fit(
                x_mat.head(20), labels["surprise_sign"].head(20), labels["event_date"].head(20)
            )

    def test_clase_unica_lanza(self, data_leaky: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        x_mat, labels = data_leaky
        y_const = pd.Series(1.0, index=x_mat.index)
        with pytest.raises(InsufficientHistory, match="minoritaria"):
            SurpriseModel().fit(x_mat, y_const, labels["event_date"])

    def test_predict_con_columnas_ausentes_lanza(
        self, data_leaky: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        x_mat, labels = data_leaky
        model = SurpriseModel().fit(x_mat, labels["surprise_sign"], labels["event_date"])
        with pytest.raises(DataQualityError, match="faltan"):
            model.predict(x_mat[["volume_runup", "turnover_zscore"]])

    def test_configuraciones_invalidas(self) -> None:
        with pytest.raises(ConfigError):
            SurpriseModel(estimator="random_forest")  # type: ignore[arg-type]
        with pytest.raises(ConfigError):
            SurpriseModel(objective="magnitude", calibration="sigmoid")
        with pytest.raises(ConfigError):
            SurpriseModel(alpha=0.0)
        with pytest.raises(ConfigError):
            SurpriseModel(calibration_fraction=0.9)

    def test_determinismo_del_gbm(self, data_leaky: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        """Regla de oro nº 4: misma semilla, mismas predicciones, bit a bit."""
        x_mat, labels = data_leaky
        args = (x_mat, labels["surprise_sign"], labels["event_date"])
        p1 = SurpriseModel(estimator="gbm", seed=7).fit(*args).predict(x_mat)
        p2 = SurpriseModel(estimator="gbm", seed=7).fit(*args).predict(x_mat)
        pd.testing.assert_series_equal(p1, p2)

    def test_coeficientes_interpretables_solo_lineal(
        self, data_leaky: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        x_mat, labels = data_leaky
        ridge = SurpriseModel(estimator="ridge").fit(
            x_mat, labels["surprise_sign"], labels["event_date"]
        )
        coefs = ridge.coefficients()
        assert set(coefs.index) == set(ridge.feature_names_)
        magnitudes = np.abs(coefs.to_numpy())
        assert (np.diff(magnitudes) <= 1e-12).all(), "ordenados por |coef| descendente"
        gbm = SurpriseModel(estimator="gbm").fit(
            x_mat, labels["surprise_sign"], labels["event_date"]
        )
        with pytest.raises(DataQualityError, match="permutation importance"):
            gbm.coefficients()


# ===========================================================================
# 4. CV purgada: estructura y contabilidad
# ===========================================================================


class TestPurgedEvaluation:
    def test_estructura_del_informe(self, report_sign_leaky: SurpriseModelReport) -> None:
        rep = report_sign_leaky
        assert rep.n_splits == 5
        assert len(rep.purge) == 5
        # Cada evento cae en exactamente un test: cobertura OOS completa.
        assert rep.oos_prediction.notna().all()
        assert rep.oos_prediction.index.equals(rep.oos_target.index)
        assert rep.fold_metrics.shape[0] == 5

    def test_contabilidad_de_la_purga(self, report_sign_leaky: SurpriseModelReport) -> None:
        """train + test + purgadas = N en cada fold, y la purga muerde de verdad.

        Con spans de 250+10 sesiones y eventos trimestrales, los folds interiores
        pierden un tramo sustancial del panel (validation_methodology.md §8.4);
        una purga que no elimina nada es señal de spans mal construidos.
        """
        rep = report_sign_leaky
        n_total = len(rep.oos_prediction)
        fm = rep.fold_metrics
        assert ((fm["n_train"] + fm["n_test"] + fm["n_purged"]) == n_total).all()
        assert fm["n_purged"].max() > 0
        assert rep.purge["purged_pct"].mean() > 5.0

    def test_min_train_inalcanzable_lanza(
        self, leaky: SyntheticMarket, data_leaky: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        x_mat, labels = data_leaky
        with pytest.raises(InsufficientHistory, match="purga"):
            SurpriseModel().evaluate(
                x_mat,
                labels["surprise_sign"],
                labels["event_date"],
                timeline=leaky.sessions,
                min_train=len(x_mat),
                n_permutation_repeats=0,
            )

    def test_evaluate_es_determinista(
        self, leaky: SyntheticMarket, data_leaky: tuple[pd.DataFrame, pd.DataFrame],
        report_sign_leaky: SurpriseModelReport,
    ) -> None:
        x_mat, labels = data_leaky
        rep2 = SurpriseModel(objective="sign", estimator="ridge").evaluate(
            x_mat,
            labels["surprise_sign"],
            labels["event_date"],
            timeline=leaky.sessions,
            n_permutation_repeats=5,
        )
        pd.testing.assert_series_equal(
            report_sign_leaky.oos_prediction, rep2.oos_prediction
        )
        pd.testing.assert_frame_equal(report_sign_leaky.importance, rep2.importance)


# ===========================================================================
# 5. Potencia: con relación inyectada, el modelo bate a las líneas base
# ===========================================================================


class TestBeatsBaselinesWithLeak:
    def test_signo_ridge_bate_a_la_clase_mayoritaria(
        self, report_sign_leaky: SurpriseModelReport
    ) -> None:
        """AUC OOS >> 0,5 y Brier mejor que la constante de prevalencia."""
        rep = report_sign_leaky
        assert rep.metrics["auc"] > 0.62
        assert rep.metrics["auc_t"] > 5.0
        assert rep.metrics["brier"] < rep.baselines["brier"]
        assert rep.metrics["log_loss"] < rep.baselines["log_loss"]
        assert rep.metrics["average_precision"] > rep.baselines["average_precision"] + 0.01
        assert rep.beats_baseline

    def test_la_linea_base_de_mayoria_es_dificil(
        self, report_sign_leaky: SurpriseModelReport
    ) -> None:
        """El walk-down hace que la mayoría ya acierte >60 %: exactitud sin mérito."""
        rep = report_sign_leaky
        assert rep.baselines["accuracy"] > 0.6
        assert rep.metrics["prevalence"] > 0.6

    def test_signo_gbm_misma_interfaz_y_señal(
        self, leaky: SyntheticMarket, data_leaky: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        x_mat, labels = data_leaky
        rep = SurpriseModel(objective="sign", estimator="gbm").evaluate(
            x_mat,
            labels["surprise_sign"],
            labels["event_date"],
            timeline=leaky.sessions,
            n_permutation_repeats=0,
        )
        assert rep.metrics["auc"] > 0.55
        assert rep.metrics["auc_t"] > 2.5
        if _has_options(x_mat):
            assert rep.metrics["auc"] > 0.70
            assert rep.beats_baseline

    def test_magnitud_de_la_sorpresa_bate_al_consenso(
        self, leaky: SyntheticMarket, data_leaky: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        """R² OOS (Campbell-Thompson) > 0 y rank-IC fuerte sobre el SUE realizado."""
        x_mat, labels = data_leaky
        _skip_without_options(x_mat)
        rep = SurpriseModel(
            target="surprise", objective="magnitude", estimator="ridge", alpha=20.0
        ).evaluate(
            x_mat,
            labels["surprise"],
            labels["event_date"],
            timeline=leaky.sessions,
            n_permutation_repeats=0,
        )
        assert rep.metrics["r2_oos"] > 0.01
        assert rep.metrics["spearman_ic"] > 0.15
        assert rep.metrics["spearman_p"] < 1e-8
        assert rep.beats_baseline

    def test_retorno_de_evento_bate_al_consenso(
        self, data_leaky: tuple[pd.DataFrame, pd.DataFrame], request: pytest.FixtureRequest
    ) -> None:
        """La tarea que importa: predecir el retorno, no el beat ya descontado."""
        x_leaky, _ = data_leaky
        _skip_without_options(x_leaky)
        resp: SyntheticMarket = request.getfixturevalue("resp")
        x_mat, labels = request.getfixturevalue("data_resp")
        rep = SurpriseModel(
            target="event_return", objective="magnitude", estimator="ridge", alpha=50.0
        ).evaluate(
            x_mat,
            labels["event_return"],
            labels["event_date"],
            timeline=resp.sessions,
            n_permutation_repeats=0,
        )
        assert rep.metrics["r2_oos"] > 0.0
        assert rep.metrics["spearman_ic"] > 0.10
        assert rep.metrics["spearman_p"] < 1e-4
        assert rep.beats_baseline

    def test_comparacion_de_estimadores_en_los_mismos_folds(
        self, leaky: SyntheticMarket, data_leaky: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        x_mat, labels = data_leaky
        table = compare_estimators(
            x_mat,
            labels["surprise_sign"],
            labels["event_date"],
            base_model=SurpriseModel(objective="sign"),
            timeline=leaky.sessions,
        )
        assert list(table.index) == ["ridge", "gbm"]
        for col in ("auc", "brier", "baseline_brier", "beats_baseline"):
            assert col in table.columns
        assert (table["auc"] > 0.55).all()


# ===========================================================================
# 6. Placebo: sin filtración, el modelo NO bate a la base (no hay fuga)
# ===========================================================================


class TestPlaceboNoLeak:
    def test_signo_placebo(
        self, clean: SyntheticMarket, data_clean: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        """Con leak_fraction=0 la sorpresa es aleatoria: AUC ≈ 0,5, no bate."""
        x_mat, labels = data_clean
        rep = SurpriseModel(objective="sign", estimator="ridge").evaluate(
            x_mat,
            labels["surprise_sign"],
            labels["event_date"],
            timeline=clean.sessions,
            n_permutation_repeats=0,
        )
        band = max(0.08, 3.5 * rep.metrics["auc_se"])
        assert abs(rep.metrics["auc"] - 0.5) < band, (
            f"AUC {rep.metrics['auc']:.3f} lejos de 0,5 sin filtración: fuga en el pipeline"
        )
        assert not rep.beats_baseline

    def test_magnitud_placebo(
        self, clean: SyntheticMarket, data_clean: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        x_mat, labels = data_clean
        rep = SurpriseModel(
            target="surprise", objective="magnitude", estimator="ridge", alpha=20.0
        ).evaluate(
            x_mat,
            labels["surprise"],
            labels["event_date"],
            timeline=clean.sessions,
            n_permutation_repeats=0,
        )
        assert rep.metrics["r2_oos"] < 0.02
        assert rep.metrics["spearman_p"] > 0.01
        assert not rep.beats_baseline

    def test_retorno_placebo(
        self, clean: SyntheticMarket, data_clean: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        x_mat, labels = data_clean
        rep = SurpriseModel(
            target="event_return", objective="magnitude", estimator="ridge", alpha=50.0
        ).evaluate(
            x_mat,
            labels["event_return"],
            labels["event_date"],
            timeline=clean.sessions,
            n_permutation_repeats=0,
        )
        assert abs(rep.metrics["spearman_ic"]) < 0.10
        assert rep.metrics["spearman_p"] > 0.01
        assert rep.metrics["r2_oos"] < 0.02
        assert not rep.beats_baseline

    def test_etiquetas_barajadas(
        self, leaky: SyntheticMarket, data_leaky: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        """Permutar y rompe la relación: si el AUC sobreviviera, la CV filtraría."""
        x_mat, labels = data_leaky
        rng = np.random.default_rng(99)
        y_shuffled = pd.Series(
            rng.permutation(labels["surprise_sign"].to_numpy()), index=labels.index
        )
        rep = SurpriseModel(objective="sign", estimator="ridge").evaluate(
            x_mat,
            y_shuffled,
            labels["event_date"],
            timeline=leaky.sessions,
            n_permutation_repeats=0,
        )
        band = max(0.08, 3.5 * rep.metrics["auc_se"])
        assert abs(rep.metrics["auc"] - 0.5) < band
        assert not rep.beats_baseline


# ===========================================================================
# 7. Permutation importance FUERA de muestra
# ===========================================================================


class TestPermutationImportance:
    def test_cubre_todas_las_features_del_modelo(
        self, report_sign_leaky: SurpriseModelReport,
        data_leaky: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        x_raw, _ = data_leaky
        imp = report_sign_leaky.importance
        expected = set(feature_matrix(x_raw).columns)
        assert set(imp.index) == expected
        assert {"importance_mean", "importance_std", "n_samples"} <= set(imp.columns)
        # 5 folds x 5 repeticiones; algún fold degenerado podría restar.
        assert (imp["n_samples"] >= 15).all()

    def test_las_features_direccionales_importan(
        self, report_sign_leaky: SurpriseModelReport
    ) -> None:
        """La huella inyectada (deriva firmada, volumen) debe aparecer arriba."""
        imp = report_sign_leaky.importance
        flow_candidates = {
            "pre_event_scar_5d",
            "pre_event_scar_10d",
            "pre_event_scar_20d",
            "abnormal_volume_10d",
            "abnormal_volume_20d",
            "turnover_zscore_vs_own_prior_quarters",
            "order_imbalance_bvc_5d",
            "order_imbalance_bvc_10d",
        }
        top10 = set(imp.head(10).index)
        winners = flow_candidates & top10
        assert winners, f"ninguna feature de flujo en el top-10: {sorted(top10)}"
        assert (imp.loc[sorted(winners), "importance_mean"] > 0).all()

    def test_ordenada_descendente(self, report_sign_leaky: SurpriseModelReport) -> None:
        means = report_sign_leaky.importance["importance_mean"].dropna().to_numpy()
        assert (np.diff(means) <= 1e-12).all()


# ===========================================================================
# 8. Calibración de probabilidades
# ===========================================================================


class TestCalibration:
    def test_curva_del_informe(self, report_sign_leaky: SurpriseModelReport) -> None:
        curve = report_sign_leaky.calibration
        assert curve is not None
        assert int(curve["count"].sum()) == int(report_sign_leaky.metrics["n_oos"])
        assert curve["p_mean"].between(0, 1).all()
        assert curve["y_rate"].between(0, 1).all()
        assert 0.0 <= curve.attrs["ece"] <= 1.0
        assert curve.attrs["ece"] == pytest.approx(report_sign_leaky.metrics["ece"])

    def test_prediccion_perfecta_tiene_ece_cero(self) -> None:
        y = np.array([0, 1] * 30)
        assert expected_calibration_error(y, y.astype(float)) == pytest.approx(0.0)

    def test_calibrar_el_gbm_mejora_brier_y_ece(
        self, leaky: SyntheticMarket, data_leaky: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        """El boosting descalibra (Niculescu-Mizil y Caruana 2005); Platt lo corrige."""
        x_mat, labels = data_leaky
        _skip_without_options(x_mat)
        args = (x_mat, labels["surprise_sign"], labels["event_date"])
        kwargs = {"timeline": leaky.sessions, "n_permutation_repeats": 0}
        raw = SurpriseModel(objective="sign", estimator="gbm").evaluate(*args, **kwargs)
        platt = SurpriseModel(
            objective="sign", estimator="gbm", calibration="sigmoid"
        ).evaluate(*args, **kwargs)
        assert platt.metrics["brier"] < raw.metrics["brier"]
        assert platt.metrics["ece"] < raw.metrics["ece"]

    def test_isotonic_produce_probabilidades_validas(
        self, data_leaky: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        x_mat, labels = data_leaky
        model = SurpriseModel(objective="sign", estimator="ridge", calibration="isotonic")
        pred = model.fit(x_mat, labels["surprise_sign"], labels["event_date"]).predict(x_mat)
        assert float(pred.min()) >= 0.0 and float(pred.max()) <= 1.0
        assert model.calibrator_ is not None

    def test_validacion_de_entradas(self) -> None:
        with pytest.raises(DataQualityError):
            brier_score([0, 1, 2], [0.1, 0.2, 0.3])  # etiquetas no binarias
        with pytest.raises(DataQualityError):
            brier_score([0, 1, 0], [0.1, 1.4, 0.3])  # probabilidad > 1
        with pytest.raises(InsufficientHistory):
            calibration_curve([0, 1] * 3, [0.2, 0.8] * 3, n_bins=10)
        with pytest.raises(ConfigError):
            calibration_curve([0, 1] * 30, [0.2, 0.8] * 30, n_bins=1)


# ===========================================================================
# 9. Métricas auxiliares y líneas base
# ===========================================================================


class TestMetricHelpers:
    def test_auc_extremos_y_error_estandar(self) -> None:
        y = np.array([False] * 50 + [True] * 50)
        s = np.arange(100, dtype=float)
        auc, se = roc_auc(s, y)
        assert auc == pytest.approx(1.0)
        auc_inv, _ = roc_auc(-s, y)
        assert auc_inv == pytest.approx(0.0)
        # Hanley-McNeil bajo H0 con n1 = n0 = 50.
        assert se == pytest.approx(math.sqrt(101.0 / (12.0 * 2500.0)))

    def test_auc_una_sola_clase_lanza(self) -> None:
        with pytest.raises(InsufficientHistory):
            roc_auc([0.1, 0.9, 0.5], [True, True, True])

    def test_spearman_ic_guardas(self) -> None:
        with pytest.raises(InsufficientHistory):
            spearman_ic([1.0, 2.0], [2.0, 1.0])
        with pytest.raises(DataQualityError, match="constante"):
            spearman_ic([1.0] * 10, list(range(10)))
        rho, p = spearman_ic(list(range(20)), list(range(20)))
        assert rho == pytest.approx(1.0)
        assert p < 1e-10

    def test_resumen_y_claves_de_lineas_base(
        self, report_sign_leaky: SurpriseModelReport
    ) -> None:
        rep = report_sign_leaky
        assert {"accuracy", "auc", "brier", "log_loss", "average_precision"} <= set(
            rep.baselines
        )
        assert rep.baselines["auc"] == 0.5
        summary = rep.summary()
        assert {"model", "baseline"} == set(summary.columns)
        assert summary.loc["auc", "model"] == pytest.approx(rep.metrics["auc"])
        assert summary.loc["auc", "baseline"] == pytest.approx(0.5)

    def test_lineas_base_de_regresion(
        self, clean: SyntheticMarket, data_clean: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        x_mat, labels = data_clean
        rep = SurpriseModel(target="event_return", objective="magnitude").evaluate(
            x_mat,
            labels["event_return"],
            labels["event_date"],
            timeline=clean.sessions,
            n_permutation_repeats=0,
        )
        assert {"mse", "mse_consensus_zero", "spearman_ic"} <= set(rep.baselines)
        assert rep.baselines["mse_consensus_zero"] > 0.0
        # R² contra el consenso-cero puede ser > 0 solo por la prima media de
        # anuncio; por eso nunca se usa como criterio de habilidad en solitario.
        assert "r2_vs_consensus" in rep.metrics
