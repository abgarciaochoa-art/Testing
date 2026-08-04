"""Tests de `events.flow` y `events.preevent`: detección de huella informada.

Cobertura, en orden de importancia:

1. **No-look-ahead (crítico).** Ninguna feature usa datos de la sesión T ni
   posteriores: se verifica con `pit.assert_no_lookahead` y, sobre todo, con el
   test de perturbación — se corrompen los datos de un emisor desde su fecha de
   evento en adelante y su fila de features debe salir *bit a bit* idéntica.
   Las fuentes con retardo institucional (short interest, off-exchange, Form 4)
   tienen tests de fixture que comprueban el filtrado estricto por `available_at`.
2. **Matemática del informe.** El `n_eff` exacto del AR(1) se contrasta contra la
   tabla verificada por Monte Carlo de `docs/research/informed_trading.md` §3.3 y
   contra una simulación propia; la aproximación asintótica queda demostrada como
   materialmente sesgada. El BVC solo existe ponderado por volumen y se demuestra
   que no es función del retorno (informe §4.2).
3. **El detector detecta.** Contra `SyntheticMarket.leaked_event_ids()` (la
   verdad-terreno): AUC significativamente > 0,5 con filtración inyectada, y AUC
   indistinguible de 0,5 en el placebo con `leak_fraction=0` (informe §12.5).
   Recordatorio de tasa base (informe §12.4): con prevalencia del 2 %, un
   detector con Se=0,80/Sp=0,90 tiene PPV ≈ 0,14 — el 86 % de las alertas serían
   falsos positivos; por eso el output es un score continuo y estas métricas se
   miran junto a la precisión, no en lugar de ella.
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata, spearmanr

from earnings_alpha.data.synthetic import SyntheticMarket
from earnings_alpha.errors import (
    DataQualityError,
    InsufficientHistory,
    LookAheadError,
)
from earnings_alpha.events import flow
from earnings_alpha.events.preevent import (
    DEFAULT_DIRECTIONAL_WEIGHTS,
    DEFAULT_INTENSITY_WEIGHTS,
    METADATA_COLUMNS,
    OPTION_FEATURE_COLUMNS,
    EventContext,
    InformedTradingScore,
    PreEventFeatures,
    cohort_labels,
    positive_predictive_value,
    pre_event_car,
    residualize_features,
)
from earnings_alpha.pit import assert_no_lookahead, get_calendar

try:  # el módulo de opciones pertenece a otro propietario y puede no existir aún
    from earnings_alpha.events import options_signals as _options_signals  # noqa: F401

    HAS_OPTIONS_SIGNALS = True
except ImportError:
    HAS_OPTIONS_SIGNALS = False

# --------------------------------------------------------------------------- fixtures

WIDE = {"n_tickers": 24, "start": "2019-01-02", "end": "2023-12-29"}
SMALL = {"n_tickers": 12, "start": "2020-01-02", "end": "2022-12-30"}


@pytest.fixture(scope="module")
def leaky() -> SyntheticMarket:
    """Mercado amplio con filtración abundante, para los tests de potencia."""
    return SyntheticMarket(seed=1234, leak_fraction=0.35, **WIDE)


@pytest.fixture(scope="module")
def clean() -> SyntheticMarket:
    """Mismo mercado sin filtración alguna: grupo de control / placebo."""
    return SyntheticMarket(seed=1234, leak_fraction=0.0, **WIDE)


@pytest.fixture(scope="module")
def ctx_leaky(leaky: SyntheticMarket) -> EventContext:
    return EventContext.from_synthetic(leaky)


@pytest.fixture(scope="module")
def feats_leaky(ctx_leaky: EventContext) -> pd.DataFrame:
    return PreEventFeatures().compute(ctx_leaky)


@pytest.fixture(scope="module")
def feats_clean(clean: SyntheticMarket) -> pd.DataFrame:
    return PreEventFeatures().compute(EventContext.from_synthetic(clean))


@pytest.fixture(scope="module")
def small() -> SyntheticMarket:
    """Mercado pequeño para tests estructurales y de perturbación."""
    return SyntheticMarket(seed=777, leak_fraction=0.30, **SMALL)


@pytest.fixture(scope="module")
def ctx_small(small: SyntheticMarket) -> EventContext:
    return EventContext.from_synthetic(small)


@pytest.fixture(scope="module")
def feats_small(ctx_small: EventContext) -> pd.DataFrame:
    return PreEventFeatures().compute(ctx_small)


# ------------------------------------------------------------------------- utilidades


def _leak_mask(mkt: SyntheticMarket, feats: pd.DataFrame) -> pd.Series:
    leaked = set(mkt.leaked_event_ids())
    return feats.index.to_series().isin(leaked)


def _welch_t(x: pd.Series, mask: pd.Series) -> float:
    """t de Welch entre eventos filtrados y limpios."""
    a, b = x[mask].dropna(), x[~mask].dropna()
    assert len(a) > 20 and len(b) > 40, "muestra insuficiente para el contraste"
    return float((a.mean() - b.mean()) / math.sqrt(a.var() / len(a) + b.var() / len(b)))


def _auc(score: pd.Series, labels: pd.Series) -> tuple[float, float, int, int]:
    """AUC-ROC por Mann-Whitney y su error estándar bajo H0 (Hanley-McNeil)."""
    s = score.to_numpy(dtype=float)
    finite = np.isfinite(s)
    ranks = rankdata(s[finite])
    pos = labels.to_numpy()[finite]
    n1, n0 = int(pos.sum()), int((~pos).sum())
    auc = float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    se = math.sqrt((n1 + n0 + 1) / (12.0 * n1 * n0))
    return auc, se, n1, n0


def _panel(rows: list[dict]) -> pd.DataFrame:
    """Panel OHLCV canónico (date, ticker) desde filas planas."""
    frame = pd.DataFrame(rows)
    return frame.set_index(["date", "ticker"]).sort_index()


# ===========================================================================
# 1. n_eff exacto de un AR(1) — la corrección central del informe (§3.3)
# ===========================================================================


class TestAR1EffectiveSampleSize:
    @pytest.mark.parametrize(
        ("rho", "k", "expected"),
        [
            (0.3, 10, 5.765),
            (0.6, 5, 1.911),
            (0.6, 20, 5.517),
            (0.8, 5, 1.381),
            (0.8, 10, 1.842),
        ],
    )
    def test_tabla_verificada_del_informe(self, rho: float, k: int, expected: float) -> None:
        """Los valores exactos coinciden con la tabla Monte Carlo del informe §3.3."""
        assert flow.ar1_effective_sample_size(k, rho) == pytest.approx(expected, abs=5e-3)

    def test_casos_limite(self) -> None:
        assert flow.ar1_effective_sample_size(1, 0.9) == 1.0
        assert flow.ar1_effective_sample_size(10, 0.0) == pytest.approx(10.0)
        # Con rho negativo la media es MÁS precisa que bajo iid.
        assert flow.ar1_effective_sample_size(10, -0.4) > 10.0

    def test_la_asintotica_esta_sesgada_y_no_se_usa(self) -> None:
        """La aproximación k(1-ρ)/(1+ρ) puede dar n_eff < 1; la exacta jamás."""
        rho, k = 0.8, 5
        asymptotic = k * (1 - rho) / (1 + rho)
        exact = flow.ar1_effective_sample_size(k, rho)
        assert asymptotic < 1.0  # el sinsentido que el informe señala
        assert exact > 1.0
        assert abs(exact - asymptotic) / exact > 0.5  # sesgo material, no cosmético

    def test_verificacion_monte_carlo(self) -> None:
        """Var(media_k) simulada = sigma²/n_eff con error < 3 % (2·10⁵ obs)."""
        rho, k, n = 0.6, 5, 200_000
        rng = np.random.default_rng(20260804)
        eps = rng.standard_normal(n)
        x = np.empty(n)
        x[0] = eps[0] / math.sqrt(1 - rho * rho)
        for i in range(1, n):
            x[i] = rho * x[i - 1] + eps[i]
        means = pd.Series(x).rolling(k).mean().dropna().to_numpy()[::k]
        n_eff_emp = x.var(ddof=1) / means.var(ddof=1)
        assert n_eff_emp == pytest.approx(flow.ar1_effective_sample_size(k, rho), rel=0.03)

    def test_entradas_invalidas(self) -> None:
        with pytest.raises(DataQualityError):
            flow.ar1_effective_sample_size(0, 0.5)
        with pytest.raises(DataQualityError):
            flow.ar1_effective_sample_size(5, 1.0)


# ===========================================================================
# 2. Features de volumen
# ===========================================================================


class TestTurnoverZscore:
    def test_estructura(self, ctx_small: EventContext) -> None:
        z = flow.turnover_zscore(ctx_small.prices, ctx_small.events, k=5)
        assert z.index.name == "event_id"
        assert z.index.is_unique
        assert z.notna().sum() > 50

    def test_detecta_la_filtracion(self, leaky: SyntheticMarket, ctx_leaky: EventContext) -> None:
        """El run-up de volumen inyectado sube el z de los eventos filtrados."""
        z = flow.turnover_zscore(ctx_leaky.prices, ctx_leaky.events, k=10)
        frame = z.to_frame("z")
        mask = frame.index.to_series().isin(set(leaky.leaked_event_ids()))
        assert _welch_t(frame["z"], mask) > 3.0

    def test_metodos_empirico_y_ar1_coherentes(self, ctx_leaky: EventContext) -> None:
        """Las dos correcciones de autocorrelación del informe §3.3 casi coinciden."""
        ze = flow.turnover_zscore(ctx_leaky.prices, ctx_leaky.events, k=5, method="empirical")
        za = flow.turnover_zscore(ctx_leaky.prices, ctx_leaky.events, k=5, method="ar1")
        both = pd.concat([ze, za], axis=1).dropna()
        assert len(both) > 300
        corr = float(np.corrcoef(both.iloc[:, 0], both.iloc[:, 1])[0, 1])
        assert corr > 0.95

    def test_historia_insuficiente_lanza(self, ctx_small: EventContext) -> None:
        """Con un panel de 40 sesiones ningún evento tiene ventana base: fallo ruidoso."""
        dates = ctx_small.prices.index.get_level_values("date").unique().sort_values()
        short_panel = ctx_small.prices[
            ctx_small.prices.index.get_level_values("date") >= dates[-40]
        ]
        with pytest.raises(InsufficientHistory):
            flow.turnover_zscore(short_panel, ctx_small.events, k=5)

    def test_eventos_vacios_o_mal_formados(self, ctx_small: EventContext) -> None:
        with pytest.raises(DataQualityError):
            flow.turnover_zscore(ctx_small.prices, ctx_small.events.iloc[:0], k=5)
        with pytest.raises(DataQualityError):
            flow.turnover_zscore(
                ctx_small.prices, ctx_small.events.drop(columns=["event_date"]), k=5
            )
        dup = pd.concat([ctx_small.events.head(2)] * 2)
        with pytest.raises(DataQualityError):
            flow.turnover_zscore(ctx_small.prices, dup, k=5)


class TestAbnormalVolumeYRunup:
    def test_abnormal_volume_columnas_y_consistencia(self, ctx_small: EventContext) -> None:
        """`abnormal_volume_5d` es exactamente el zbar de k=5 (informe §3.4)."""
        av = flow.abnormal_volume(ctx_small.prices, ctx_small.events, windows=(5, 10, 20))
        assert list(av.columns) == [
            "abnormal_volume_5d",
            "abnormal_volume_10d",
            "abnormal_volume_20d",
        ]
        z5 = flow.turnover_zscore(ctx_small.prices, ctx_small.events, k=5)
        pd.testing.assert_series_equal(
            av["abnormal_volume_5d"], z5, check_names=False
        )

    def test_runup_detecta_y_esta_bien_normalizado(
        self, leaky: SyntheticMarket, ctx_leaky: EventContext
    ) -> None:
        avr = flow.volume_runup(ctx_leaky.prices, ctx_leaky.events, k=5)
        frame = avr.to_frame("avr")
        mask = frame.index.to_series().isin(set(leaky.leaked_event_ids()))
        assert _welch_t(frame["avr"], mask) > 4.0
        # AVR = 1 es normalidad: la mediana de los eventos limpios debe rondar 1.
        assert 0.85 < float(frame.loc[~mask, "avr"].median()) < 1.15

    def test_chae_baseline_por_emisor(
        self, leaky: SyntheticMarket, ctx_leaky: EventContext
    ) -> None:
        """La corrección de Chae (2005): z frente a los pre-anuncios del propio emisor."""
        feat = flow.turnover_zscore_vs_own_prior_quarters(
            ctx_leaky.prices, ctx_leaky.events, k=5, min_prior=2
        )
        frame = feat.to_frame("v")
        mask = frame.index.to_series().isin(set(leaky.leaked_event_ids()))
        assert _welch_t(frame["v"], mask) > 3.0
        # Los primeros eventos de cada emisor no tienen línea base propia: NaN.
        ev = ctx_leaky.events.sort_values(["ticker", "event_date"])
        primeros = ev.groupby("ticker", sort=False)["event_id"].head(1)
        assert feat.reindex(primeros).isna().all()


class TestSUV:
    def test_estructura_y_potencia(
        self, leaky: SyntheticMarket, ctx_leaky: EventContext
    ) -> None:
        s = flow.suv(ctx_leaky.prices, ctx_leaky.events, windows=(5, 10))
        assert list(s.columns) == ["suv_5d", "suv_10d"]
        mask = s.index.to_series().isin(set(leaky.leaked_event_ids()))
        # El run-up inyectado no lo explica el retorno: el SUV debe subir.
        assert _welch_t(s["suv_10d"], mask) > 2.0


# ===========================================================================
# 3. Proxies de order imbalance con OHLCV
# ===========================================================================


def _bvc_fixture(volume_on_up_days: float, volume_on_down_days: float) -> pd.DataFrame:
    """Panel de un solo ticker: misma senda de precios, distinto reparto de volumen.

    Los últimos 10 días alternan dP = +1 / -1 (retorno acumulado ≈ 0); el volumen
    se concentra en subidas o en bajadas según los argumentos. Si el BVC agregado
    dependiera solo del retorno, ambos repartos darían el mismo valor.
    """
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2021-01-04", periods=80)
    n = len(dates)
    close = np.empty(n)
    close[0] = 100.0
    steps = rng.normal(0.0, 0.8, n)
    for i in range(1, n - 10):
        close[i] = close[i - 1] + steps[i]
    for j, i in enumerate(range(n - 10, n)):
        close[i] = close[i - 1] + (1.0 if j % 2 == 0 else -1.0)
    volume = np.full(n, 1000.0)
    for j, i in enumerate(range(n - 10, n)):
        volume[i] = volume_on_up_days if j % 2 == 0 else volume_on_down_days
    rows = [
        {
            "date": d,
            "ticker": "XX",
            "close": c,
            "high": c + 0.5,
            "low": c - 0.5,
            "volume": v,
        }
        for d, c, v in zip(dates, close, volume, strict=True)
    ]
    return _panel(rows)


_BVC_EVENTS = pd.DataFrame(
    {
        "event_id": ["XX:ev"],
        "ticker": ["XX"],
        "event_date": [pd.Timestamp("2021-01-04") + pd.offsets.BDay(79)],
    }
)


class TestBVC:
    def test_solo_existe_ponderado_por_volumen(self) -> None:
        """Misma senda de dP, distinto volumen -> distinto OIB (informe §4.2).

        La trampa documentada: con barras diarias el BVC de una barra es una
        función determinista del retorno. La única versión implementada pondera
        por volumen entre días, y este test demuestra que esa versión NO es
        función de la senda de precios: separa compra de distribución.
        """
        up = flow.bvc_order_imbalance(_bvc_fixture(10_000.0, 1_000.0), _BVC_EVENTS, k=10)
        down = flow.bvc_order_imbalance(_bvc_fixture(1_000.0, 10_000.0), _BVC_EVENTS, k=10)
        v_up, v_down = float(up.iloc[0]), float(down.iloc[0])
        assert v_up > 0.2, "volumen en subidas debe leer acumulación"
        assert v_down < -0.2, "volumen en bajadas debe leer distribución"
        assert v_up != pytest.approx(v_down)

    def test_direccion_en_eventos_filtrados(
        self, leaky: SyntheticMarket, ctx_leaky: EventContext
    ) -> None:
        """La deriva inyectada firma el OIB en el signo de la sorpresa."""
        oib = flow.bvc_order_imbalance(ctx_leaky.prices, ctx_leaky.events, k=5)
        truth = leaky.ground_truth()
        signed = (oib * truth["surprise_sign"].reindex(oib.index)).to_frame("v")
        mask = signed.index.to_series().isin(set(leaky.leaked_event_ids()))
        assert _welch_t(signed["v"], mask) > 4.0


class TestCLV:
    def test_valor_exacto_y_rango_degenerado(self) -> None:
        """CLV en [-1, 1]; un día con H == L contribuye 0 (informe §4.3)."""
        dates = pd.bdate_range("2021-06-01", periods=3)
        rows = [
            # cierre en el máximo -> CLV = +1, volumen 300
            {"date": dates[0], "ticker": "YY", "close": 11.0, "high": 11.0, "low": 10.0,
             "volume": 300.0},
            # cierre en el mínimo -> CLV = -1, volumen 100
            {"date": dates[1], "ticker": "YY", "close": 10.0, "high": 11.0, "low": 10.0,
             "volume": 100.0},
            # sin rango -> CLV = 0, volumen 600
            {"date": dates[2], "ticker": "YY", "close": 10.5, "high": 10.5, "low": 10.5,
             "volume": 600.0},
        ]
        events = pd.DataFrame(
            {"event_id": ["YY:ev"], "ticker": ["YY"],
             "event_date": [dates[-1] + pd.offsets.BDay(1)]}
        )
        oib = flow.clv_order_imbalance(_panel(rows), events, k=3)
        # (1*300 - 1*100 + 0*600) / 1000 = 0.2
        assert float(oib.iloc[0]) == pytest.approx(0.2)

    def test_acotado_y_direccional(self, leaky: SyntheticMarket, ctx_leaky: EventContext) -> None:
        oib = flow.clv_order_imbalance(ctx_leaky.prices, ctx_leaky.events, k=5)
        finite = oib.dropna()
        assert ((finite >= -1.0) & (finite <= 1.0)).all()
        truth = leaky.ground_truth()
        signed = (oib * truth["surprise_sign"].reindex(oib.index)).to_frame("v")
        mask = signed.index.to_series().isin(set(leaky.leaked_event_ids()))
        assert _welch_t(signed["v"], mask) > 3.0


class TestAmihud:
    def test_control_de_iliquidez(self, ctx_leaky: EventContext, feats_leaky: pd.DataFrame) -> None:
        """ILLIQ > 0 y decreciente en el tamaño/volumen: es la proxy de Amihud (2002)."""
        illiq = feats_leaky["amihud_illiquidity"].dropna()
        assert (illiq > 0).all()
        both = feats_leaky[["amihud_illiquidity", "log_adv"]].dropna()
        rho = float(spearmanr(both["amihud_illiquidity"], both["log_adv"]).statistic)
        assert rho < -0.8


# ===========================================================================
# 4. CAR pre-evento
# ===========================================================================


class TestPreEventCAR:
    def test_estructura_y_formas_de_mercado(self, ctx_small: EventContext) -> None:
        a = pre_event_car(ctx_small.prices, ctx_small.market, ctx_small.events)
        b = pre_event_car(ctx_small.prices, ctx_small.market["log_return"], ctx_small.events)
        pd.testing.assert_frame_equal(a, b)
        for k in (5, 10, 20):
            assert f"pre_event_car_{k}d" in a.columns
            assert f"pre_event_scar_{k}d" in a.columns

    def test_estimacion_no_solapa_deteccion(self, ctx_small: EventContext) -> None:
        """El contrato §3.5 exige que la estimación termine antes del evento."""
        with pytest.raises(DataQualityError):
            pre_event_car(
                ctx_small.prices, ctx_small.market, ctx_small.events,
                estimation=(-250, -10), windows=(20,),
            )
        with pytest.raises(DataQualityError):
            pre_event_car(
                ctx_small.prices, ctx_small.market, ctx_small.events, estimation=(-40, -250)
            )

    def test_min_observaciones_de_estimacion(self, ctx_small: EventContext) -> None:
        """Los eventos sin 120 obs de estimación salen NaN, no estimados a la ligera."""
        car = pre_event_car(ctx_small.prices, ctx_small.market, ctx_small.events)
        ev = flow._normalize_events(ctx_small.events)
        dates = ctx_small.prices.index.get_level_values("date").unique().sort_values()
        pos = pd.DatetimeIndex(dates).get_indexer(pd.DatetimeIndex(ev["event_date"]))
        early = ev.loc[(pos >= 0) & (pos < 130), "event_id"]
        assert len(early) > 0
        assert car.reindex(early)["pre_event_car_20d"].isna().all()

    def test_deriva_anticipada_firmada(
        self, leaky: SyntheticMarket, ctx_leaky: EventContext, feats_leaky: pd.DataFrame
    ) -> None:
        """La deriva inyectada con el signo de la sorpresa aparece en el SCAR."""
        truth = leaky.ground_truth()
        sign = truth["surprise_sign"].reindex(feats_leaky.index)
        signed = feats_leaky["pre_event_scar_5d"] * sign
        mask = _leak_mask(leaky, feats_leaky)
        assert _welch_t(signed, mask) > 5.0


# ===========================================================================
# 5. Fuentes con retardo institucional (fixtures PIT)
# ===========================================================================


def _si_fixture(jump_published_before_t: bool) -> pd.DataFrame:
    """Short interest quincenal con un salto en el último snapshot publicable.

    Si `jump_published_before_t` es False, el salto se publica el mismo día T y
    la feature debe ignorarlo por completo (publicación intradía desconocida ->
    exclusión conservadora del propio día).
    """
    settlements = pd.date_range("2020-06-15", periods=24, freq="SME")
    sir = 0.05 + 0.001 * np.resize([1.0, -1.0], len(settlements))
    sir[-2] = 0.09  # snapshot liquidado 2021-05-31: el salto sospechoso
    rows = []
    for i, (d, v) in enumerate(zip(settlements, sir, strict=True)):
        avail = d + pd.Timedelta(days=12)
        if i == len(settlements) - 2 and not jump_published_before_t:
            avail = pd.Timestamp("2021-06-15")  # publicado exactamente en T
        rows.append(
            {
                "ticker": "AAA",
                "settlement_date": d,
                "available_at": avail,
                "short_percent_shares": v,
            }
        )
    return pd.DataFrame(rows)


_SI_EVENTS = pd.DataFrame(
    {"event_id": ["AAA:ev"], "ticker": ["AAA"], "event_date": [pd.Timestamp("2021-06-15")]}
)


class TestShortInterest:
    def test_salto_publicado_dispara_la_senal(self) -> None:
        v = float(flow.short_interest_delta(_si_fixture(True), _SI_EVENTS).iloc[0])
        # Subida brusca de SIR -> señal bajista -> feature (= -dSI_z) muy negativa.
        assert v < -3.0

    def test_available_at_manda_no_la_settlement_date(self) -> None:
        """El snapshot publicado en T no existe para la señal: PIT estricto (§8.1)."""
        v_pub = float(flow.short_interest_delta(_si_fixture(True), _SI_EVENTS).iloc[0])
        v_late = float(flow.short_interest_delta(_si_fixture(False), _SI_EVENTS).iloc[0])
        assert abs(v_late) < 3.0  # sin el salto, el delta es el ruido alternante
        assert v_pub != pytest.approx(v_late)

    def test_pocos_snapshots_dan_nan(self) -> None:
        si = _si_fixture(True).tail(4)
        with pytest.raises(InsufficientHistory):
            flow.short_interest_delta(si, _SI_EVENTS)

    def test_ciclo_de_liquidacion_por_epoca(self) -> None:
        """T+3 / T+2 / T+1 según la fecha (informe §8.1, trampa 2)."""
        assert flow.settlement_cycle_lag(dt.date(2015, 6, 15)) == 3
        assert flow.settlement_cycle_lag(dt.date(2017, 9, 6)) == 3
        assert flow.settlement_cycle_lag(dt.date(2017, 9, 7)) == 2
        assert flow.settlement_cycle_lag(dt.date(2020, 3, 16)) == 2
        assert flow.settlement_cycle_lag(dt.date(2024, 5, 28)) == 2
        assert flow.settlement_cycle_lag(dt.date(2024, 5, 29)) == 1
        assert flow.settlement_cycle_lag(dt.date(2025, 1, 15)) == 1

    def test_settlement_a_trade_date(self) -> None:
        cal = get_calendar()
        # 2020-01-15 (miércoles), ciclo T+2 -> negociado el lunes 13.
        assert flow.settlement_to_trade_date(dt.date(2020, 1, 15), cal) == dt.date(2020, 1, 13)
        # 2015-01-15 (jueves), ciclo T+3 -> negociado el lunes 12.
        assert flow.settlement_to_trade_date(dt.date(2015, 1, 15), cal) == dt.date(2015, 1, 12)
        # 2025-01-15 (miércoles), ciclo T+1 -> negociado el martes 14.
        assert flow.settlement_to_trade_date(dt.date(2025, 1, 15), cal) == dt.date(2025, 1, 14)


def _offex_fixture(shift_unpublished_week: float = 0.0) -> pd.DataFrame:
    """Cuota off-exchange semanal: 20 semanas, la última publicable con salto a 0,50.

    `shift_unpublished_week` altera una semana aún NO publicada en T: el
    resultado no debe moverse ni un bit (test de la ventana efectiva ≈
    [T-35, T-15] del informe §8.2).
    """
    weeks = pd.date_range("2021-01-08", periods=20, freq="W-FRI")
    share = 0.40 + 0.01 * np.resize([1.0, -1.0], len(weeks))
    share[-3] = 0.50  # week_end 2021-05-28, publicada el 2021-06-11 (< T)
    share[-1] += shift_unpublished_week  # week_end 2021-06-11, publicada tras T
    return pd.DataFrame(
        {
            "ticker": "AAA",
            "week_end": weeks,
            "available_at": weeks + pd.Timedelta(days=14),
            "off_exchange_share": share,
        }
    )


class TestOffExchange:
    def test_valor_manual(self) -> None:
        """z contra la media/std de las 12 semanas previas a la última publicada."""
        out = flow.off_exchange_share_delta(_offex_fixture(), _SI_EVENTS)
        base = 0.40 + 0.01 * np.resize([1.0, -1.0], 20)[5:17]
        expected = (0.50 - base.mean()) / base.std(ddof=1)
        assert float(out.iloc[0]) == pytest.approx(expected, rel=1e-9)

    def test_semana_no_publicada_es_invisible(self) -> None:
        """El retardo de 2 semanas de FINRA hace inobservable la semana previa a T."""
        a = flow.off_exchange_share_delta(_offex_fixture(0.0), _SI_EVENTS)
        b = flow.off_exchange_share_delta(_offex_fixture(0.45), _SI_EVENTS)
        assert float(a.iloc[0]) == float(b.iloc[0])

    def test_historia_corta_lanza(self) -> None:
        with pytest.raises(InsufficientHistory):
            flow.off_exchange_share_delta(_offex_fixture().tail(5), _SI_EVENTS)


def _form4_fixture() -> pd.DataFrame:
    """Form 4 con los casos límite del informe §8.3."""
    rows = [
        # -- historia del insider rutinario: P cada mayo de 2018-2020
        *(
            {
                "ticker": "AAA", "insider_id": "rut", "transaction_code": "P",
                "shares": 100, "transaction_date": f"{y}-05-10",
                "accepted_at": f"{y}-05-11",
            }
            for y in (2018, 2019, 2020)
        ),
        # -- historia del oportunista: meses cambiantes desde 2018
        {"ticker": "AAA", "insider_id": "opp", "transaction_code": "P", "shares": 80,
         "transaction_date": "2018-03-05", "accepted_at": "2018-03-06"},
        {"ticker": "AAA", "insider_id": "opp", "transaction_code": "S", "shares": 60,
         "transaction_date": "2019-11-12", "accepted_at": "2019-11-13"},
        {"ticker": "AAA", "insider_id": "opp", "transaction_code": "P", "shares": 90,
         "transaction_date": "2020-07-01", "accepted_at": "2020-07-02"},
        # -- ventana [T-180, T-1] del evento del 2021-06-15
        {"ticker": "AAA", "insider_id": "ins1", "transaction_code": "P", "shares": 300,
         "transaction_date": "2021-05-01", "accepted_at": "2021-05-03"},
        {"ticker": "AAA", "insider_id": "ins1", "transaction_code": "S", "shares": 100,
         "transaction_date": "2021-04-20", "accepted_at": "2021-04-21"},
        {"ticker": "AAA", "insider_id": "rut", "transaction_code": "P", "shares": 200,
         "transaction_date": "2021-05-10", "accepted_at": "2021-05-11"},
        {"ticker": "AAA", "insider_id": "opp", "transaction_code": "P", "shares": 150,
         "transaction_date": "2021-05-20", "accepted_at": "2021-05-21"},
        {"ticker": "AAA", "insider_id": "opp", "transaction_code": "S", "shares": 50,
         "transaction_date": "2021-04-01", "accepted_at": "2021-04-02"},
        # -- exclusiones obligatorias
        {"ticker": "AAA", "insider_id": "ins1", "transaction_code": "M", "shares": 500,
         "transaction_date": "2021-05-15", "accepted_at": "2021-05-16"},  # ejercicio
        {"ticker": "AAA", "insider_id": "ins1", "transaction_code": "F", "shares": 400,
         "transaction_date": "2021-05-15", "accepted_at": "2021-05-16"},  # impuestos
        {"ticker": "AAA", "insider_id": "ins1", "transaction_code": "A", "shares": 900,
         "transaction_date": "2021-05-15", "accepted_at": "2021-05-16"},  # concesión
        {"ticker": "AAA", "insider_id": "ins1", "transaction_code": "S", "shares": 4000,
         "transaction_date": "2021-06-01", "accepted_at": "2021-06-15"},  # aceptado en T
        {"ticker": "AAA", "insider_id": "ins1", "transaction_code": "P", "shares": 7000,
         "transaction_date": "2020-01-10", "accepted_at": "2020-01-11"},  # fuera de ventana
        {"ticker": "AAA", "insider_id": "plan", "transaction_code": "P", "shares": 9999,
         "transaction_date": "2021-05-25", "accepted_at": "2021-05-26",
         "is_10b5_1": True},  # plan 10b5-1
    ]
    frame = pd.DataFrame(rows)
    frame["is_10b5_1"] = frame.get("is_10b5_1", False)
    return frame


class TestInsiders:
    def test_npr_filtrado_y_clasificacion(self) -> None:
        """Solo P/S, solo aceptados antes de T, ventana de 180 días, y CMP 2012."""
        out = flow.insider_net_buy_form4(_form4_fixture(), _SI_EVENTS)
        row = out.iloc[0]
        # En ventana: ins1 P300 + S100, rut P200, opp P150 + S50.
        # NPR total = (650 - 150) / 800.
        assert float(row["insider_net_buy_form4"]) == pytest.approx((650 - 150) / 800)
        # rut opera cada mayo desde 2018 -> rutinario; su NPR = +1 (solo compra).
        assert float(row["insider_net_buy_form4_routine"]) == pytest.approx(1.0)
        # opp tiene >= 3 años de historia sin patrón mensual -> oportunista.
        assert float(row["insider_net_buy_form4_opportunistic"]) == pytest.approx(
            (150 - 50) / 200
        )

    def test_insider_nuevo_es_unknown_no_oportunista(self) -> None:
        """Sin 3 años de historia la etiqueta es unknown y se excluye del desglose.

        Asumir oportunista por defecto sesgaría la señal (informe §8.3): ins1
        empieza a operar en 2021, así que sus operaciones cuentan en el NPR total
        pero en ningún subtotal.
        """
        f4 = _form4_fixture()
        solo_ins1 = f4[f4["insider_id"] == "ins1"]
        out = flow.insider_net_buy_form4(solo_ins1, _SI_EVENTS)
        row = out.iloc[0]
        assert float(row["insider_net_buy_form4"]) == pytest.approx((300 - 100) / 400)
        assert np.isnan(row["insider_net_buy_form4_opportunistic"])
        assert np.isnan(row["insider_net_buy_form4_routine"])

    def test_sin_transacciones_p_s_lanza(self) -> None:
        f4 = _form4_fixture()
        solo_mecanicas = f4[f4["transaction_code"].isin(["M", "F", "A"])]
        with pytest.raises(DataQualityError):
            flow.insider_net_buy_form4(solo_mecanicas, _SI_EVENTS)

    def test_sin_operaciones_en_ventana_da_nan(self) -> None:
        """La ausencia se declara como NaN, no como un cero neutral fabricado."""
        eventos_lejanos = pd.DataFrame(
            {"event_id": ["AAA:far"], "ticker": ["AAA"],
             "event_date": [pd.Timestamp("2023-06-15")]}
        )
        out = flow.insider_net_buy_form4(_form4_fixture(), eventos_lejanos)
        assert out.iloc[0].isna().all()


# ===========================================================================
# 6. PreEventFeatures: orquestación
# ===========================================================================


CONTRACT_COLUMNS = [
    "volume_runup",
    "turnover_zscore",
    "abnormal_volume_5d",
    "abnormal_volume_10d",
    "abnormal_volume_20d",
    "order_imbalance_proxy",
    "oi_buildup_calls",
    "oi_buildup_puts",
    "put_call_volume_ratio",
    "iv_skew_25delta",
    "vol_spread",
    "iv_term_slope",
    "short_interest_delta",
    "off_exchange_share_delta",
    "insider_net_buy_form4",
    "pre_event_car_5d",
    "pre_event_car_10d",
    "pre_event_car_20d",
    "analyst_revision_drift",
]


class TestPreEventFeatures:
    def test_contrato_y_estructura(self, feats_small: pd.DataFrame) -> None:
        """Todas las columnas del contrato §3.5 y los metadatos están presentes."""
        assert feats_small.index.name == "event_id"
        assert feats_small.index.is_unique
        for col in CONTRACT_COLUMNS:
            assert col in feats_small.columns, f"falta la columna del contrato {col!r}"
        for col in METADATA_COLUMNS:
            assert col in feats_small.columns, f"falta el metadato {col!r}"
        # Las adiciones del informe §14 también.
        for col in (
            "turnover_zscore_vs_own_prior_quarters",
            "suv_10d",
            "order_imbalance_clv_5d",
            "pre_event_scar_5d",
            "amihud_illiquidity",
        ):
            assert col in feats_small.columns

    def test_alias_del_contrato(self, feats_small: pd.DataFrame) -> None:
        """`order_imbalance_proxy` es el BVC ponderado por volumen a k=5."""
        pd.testing.assert_series_equal(
            feats_small["order_imbalance_proxy"],
            feats_small["order_imbalance_bvc_5d"],
            check_names=False,
        )

    def test_available_at_es_la_medianoche_de_t(self, feats_small: pd.DataFrame) -> None:
        assert (feats_small["available_at"] == feats_small["event_date"]).all()

    def test_fuentes_ausentes_degradan_a_nan_documentado(
        self, feats_small: pd.DataFrame
    ) -> None:
        """Sin Form 4 las columnas de insiders son NaN y la fuente queda anotada."""
        missing = feats_small.attrs["missing_sources"]
        assert "form4" in missing
        assert feats_small["insider_net_buy_form4"].isna().all()
        assert "analyst_revisions" in missing
        assert feats_small["analyst_revision_drift"].isna().all()

    def test_opciones_degradan_si_no_hay_modulo(self, feats_small: pd.DataFrame) -> None:
        """Si `events.options_signals` no existe, sus columnas salen NaN (contrato)."""
        if HAS_OPTIONS_SIGNALS:
            pytest.skip("options_signals existe: la degradación no aplica")
        assert "options" in feats_small.attrs["missing_sources"]
        for col in OPTION_FEATURE_COLUMNS:
            assert feats_small[col].isna().all()

    def test_mercado_obligatorio(self, ctx_small: EventContext) -> None:
        ctx = EventContext(events=ctx_small.events, prices=ctx_small.prices, market=None)
        with pytest.raises(DataQualityError):
            PreEventFeatures().compute(ctx)

    def test_sin_historico_lanza(self, ctx_small: EventContext) -> None:
        dates = ctx_small.prices.index.get_level_values("date").unique().sort_values()
        short_panel = ctx_small.prices[
            ctx_small.prices.index.get_level_values("date") >= dates[-40]
        ]
        ctx = EventContext(
            events=ctx_small.events, prices=short_panel, market=ctx_small.market
        )
        with pytest.raises(InsufficientHistory):
            PreEventFeatures().compute(ctx)

    def test_from_synthetic_valida_el_tipo(self) -> None:
        with pytest.raises(DataQualityError):
            EventContext.from_synthetic(object())


# ===========================================================================
# 7. Cohortes y residualización
# ===========================================================================


class TestCohortes:
    def test_fusion_voraz_hasta_el_minimo(self) -> None:
        dates = pd.Series(
            [pd.Timestamp("2021-04-20")] * 30
            + [pd.Timestamp("2021-05-03")] * 5
            + [pd.Timestamp("2021-05-04")] * 5
            + [pd.Timestamp("2021-05-05")] * 12
        )
        labels = cohort_labels(dates, min_size=20)
        assert (labels.iloc[:30] == pd.Timestamp("2021-04-20")).all()
        assert (labels.iloc[30:] == pd.Timestamp("2021-05-03")).all()
        counts = labels.value_counts()
        assert (counts >= 20).all()

    def test_cohortes_del_panel_respetan_el_minimo(self, feats_leaky: pd.DataFrame) -> None:
        counts = feats_leaky["cohort"].value_counts()
        assert (counts >= 20).all() or len(counts) == 1


class TestResidualizacion:
    def test_residuo_ortogonal_a_los_controles(self, ctx_small: EventContext) -> None:
        """Dentro de cada cohorte, el residuo es ⊥ al CAR-20, tamaño y ADV (§12.2)."""
        feats = PreEventFeatures(residualize=True).compute(ctx_small)
        assert feats.attrs["residualized"] is True
        col = "order_imbalance_bvc_5d"
        for _, sub in feats.groupby("cohort"):
            both = sub[[col, "pre_event_car_20d", "log_mktcap", "log_adv"]].dropna()
            if len(both) < 12:
                continue
            for control in ("pre_event_car_20d", "log_mktcap", "log_adv"):
                if both[control].nunique() < 2 or both[col].std() == 0:
                    continue
                corr = float(np.corrcoef(both[col], both[control])[0, 1])
                assert abs(corr) < 1e-6, f"residuo correlado con {control}: {corr}"

    def test_la_infraestructura_permite_medir_la_muerte_de_features(
        self, feats_small: pd.DataFrame
    ) -> None:
        """Residualizar reduce (o mantiene) la varianza: es una proyección OLS.

        Este es el mecanismo con el que se mide qué features "mueren" tras
        controlar por el CAR pre-evento (informe §9.h y §11): se comparan IC con
        y sin residualizar. Aquí se verifica la propiedad algebraica que lo
        sustenta.
        """
        res = residualize_features(feats_small)
        for col in ("order_imbalance_bvc_5d", "suv_10d", "abnormal_volume_5d"):
            paired = pd.concat(
                [feats_small[col], res[col]], axis=1, keys=["raw", "res"]
            ).dropna()
            assert len(paired) > 30
            assert paired["res"].var() <= paired["raw"].var() * (1.0 + 1e-9)

    def test_faltan_controles_lanza(self, feats_small: pd.DataFrame) -> None:
        with pytest.raises(DataQualityError):
            residualize_features(feats_small.drop(columns=["pre_event_car_20d"]))
        with pytest.raises(DataQualityError):
            residualize_features(feats_small.drop(columns=["cohort"]))


# ===========================================================================
# 8. InformedTradingScore: ¿el detector detecta?
# ===========================================================================


class TestInformedTradingScore:
    def test_auc_significativo_con_filtracion(
        self, leaky: SyntheticMarket, feats_leaky: pd.DataFrame
    ) -> None:
        """AUC >> 0,5 sobre la verdad-terreno del generador (informe §12.5)."""
        score = InformedTradingScore().score(feats_leaky)
        auc, se, n1, n0 = _auc(score, _leak_mask(leaky, feats_leaky))
        assert n1 > 100 and n0 > 200, "muestra insuficiente para un AUC estable"
        assert auc > 0.60
        assert (auc - 0.5) / se > 4.0, f"AUC {auc:.3f} no separa (se={se:.3f})"

    def test_placebo_sin_filtracion(
        self, leaky: SyntheticMarket, clean: SyntheticMarket, feats_clean: pd.DataFrame
    ) -> None:
        """Con leak_fraction=0 el AUC no se distingue de 0,5.

        Las pseudo-etiquetas son los event_id que estarían filtrados en el
        mercado gemelo con filtración (misma semilla, misma rejilla de eventos):
        un subconjunto independiente de los datos del mercado limpio, así que
        cualquier AUC lejos de 0,5 delataría una fuga en el pipeline.
        """
        assert clean.leaked_event_ids() == []
        score = InformedTradingScore().score(feats_clean)
        pseudo = feats_clean.index.to_series().isin(set(leaky.leaked_event_ids()))
        auc, se, n1, n0 = _auc(score, pseudo)
        assert n1 > 100 and n0 > 200
        assert abs(auc - 0.5) < max(0.10, 3.5 * se)

    def test_pesos_configurables(self, feats_leaky: pd.DataFrame) -> None:
        solo_volumen = InformedTradingScore(
            weights={"abnormal_volume_5d": 1.0}, min_features=1
        ).score(feats_leaky)
        assert solo_volumen.notna().sum() > 200
        # Con un único peso, el score es el z por cohorte de esa feature: debe
        # correlacionar fuertemente con la feature original.
        both = pd.concat([solo_volumen, feats_leaky["abnormal_volume_5d"]], axis=1).dropna()
        assert float(np.corrcoef(both.iloc[:, 0], both.iloc[:, 1])[0, 1]) > 0.7

    def test_modos_y_pesos_por_defecto(self, feats_leaky: pd.DataFrame) -> None:
        intensity = InformedTradingScore(mode="intensity").score(feats_leaky)
        directional = InformedTradingScore(mode="directional").score(feats_leaky)
        both = pd.concat([intensity, directional], axis=1).dropna()
        assert len(both) > 100
        assert not np.allclose(both.iloc[:, 0], both.iloc[:, 1])
        assert set(DEFAULT_INTENSITY_WEIGHTS) != set(DEFAULT_DIRECTIONAL_WEIGHTS)

    def test_pesos_invalidos(self, feats_leaky: pd.DataFrame) -> None:
        with pytest.raises(DataQualityError):
            InformedTradingScore(weights={}).score(feats_leaky)
        with pytest.raises(DataQualityError):
            InformedTradingScore(weights={"a": 0.0}).score(feats_leaky)
        with pytest.raises(DataQualityError):
            InformedTradingScore(weights={"no_existe": 1.0}).score(feats_leaky)

    def test_ppv_de_la_tasa_base(self) -> None:
        """La aritmética del informe §12.4: PPV ≈ 0,14 con prevalencia del 2 %."""
        ppv = positive_predictive_value(0.02, 0.80, 0.90)
        assert ppv == pytest.approx(0.016 / (0.016 + 0.098), abs=1e-12)
        assert ppv == pytest.approx(0.1404, abs=5e-4)
        with pytest.raises(DataQualityError):
            positive_predictive_value(1.5, 0.8, 0.9)


# ===========================================================================
# 9. Point-in-time: la garantía que valida todo lo demás
# ===========================================================================


class TestNoLookAhead:
    def test_assert_no_lookahead_pasa(
        self, small: SyntheticMarket, feats_small: pd.DataFrame
    ) -> None:
        sig = (
            feats_small.reset_index()[
                ["event_id", "ticker", "event_date", "available_at",
                 "turnover_zscore", "pre_event_car_5d"]
            ]
            .rename(columns={"event_date": "date"})
        )
        assert_no_lookahead(sig, small.events(), cal=small.calendar)

    def test_desplazar_available_at_delata_la_fuga(
        self, small: SyntheticMarket, feats_small: pd.DataFrame
    ) -> None:
        """El protocolo del informe §12.5 (punto 4): +1 sesión debe fallar."""
        sig = (
            feats_small.reset_index()[
                ["event_id", "ticker", "event_date", "available_at", "turnover_zscore"]
            ]
            .rename(columns={"event_date": "date"})
        )
        sig["available_at"] = sig["available_at"] + pd.Timedelta(days=1)
        with pytest.raises(LookAheadError):
            assert_no_lookahead(sig, small.events(), cal=small.calendar)

    def test_perturbar_el_futuro_no_mueve_ni_un_bit(
        self, small: SyntheticMarket, ctx_small: EventContext, feats_small: pd.DataFrame
    ) -> None:
        """CRÍTICO: ninguna feature usa datos de T o posteriores.

        Para una muestra de eventos se multiplican por 25 el volumen y por 3 los
        precios del emisor desde su fecha negociable en adelante y se recalcula
        TODO el pipeline. La fila del evento debe ser idéntica valor a valor y en
        su patrón de NaN: cualquier diferencia significaría que alguna feature
        tocó la sesión T o posteriores.
        """
        events = small.events().set_index("event_id")
        usable = feats_small.dropna(
            subset=["turnover_zscore", "pre_event_car_5d"]
        ).index.tolist()
        assert len(usable) > 20
        sample = usable[:: max(1, len(usable) // 4)][:4]

        meta_cols = ["ticker", "event_date", "available_at", "sector", "cohort"]
        for eid in sample:
            t0 = pd.Timestamp(events.loc[eid, "event_date"])
            tkr = events.loc[eid, "ticker"]
            px = ctx_small.prices.copy()
            mask = (px.index.get_level_values("ticker") == tkr) & (
                px.index.get_level_values("date") >= t0
            )
            assert mask.sum() > 0, "el evento debe tener sesiones futuras que corromper"
            for col, mult in [
                ("volume", 25.0), ("close", 3.0), ("adj_close", 3.0),
                ("high", 3.0), ("low", 3.0), ("open", 3.0),
                ("dollar_volume", 75.0), ("market_cap", 3.0),
            ]:
                if col in px.columns:
                    px.loc[mask, col] = px.loc[mask, col] * mult
            ctx2 = EventContext(
                events=ctx_small.events, prices=px, market=ctx_small.market,
                calendar=ctx_small.calendar, sectors=ctx_small.sectors,
                short_interest=ctx_small.short_interest,
                off_exchange=ctx_small.off_exchange, options=ctx_small.options,
            )
            perturbed = PreEventFeatures().compute(ctx2)
            a = feats_small.loc[eid].drop(meta_cols).astype(float)
            b = perturbed.loc[eid].drop(meta_cols).astype(float)
            av, bv = a.to_numpy(), b.to_numpy()
            assert np.array_equal(np.isnan(av), np.isnan(bv)), f"patrón NaN cambió en {eid}"
            assert np.array_equal(
                np.nan_to_num(av), np.nan_to_num(bv)
            ), f"features de {eid} usan datos >= T"
