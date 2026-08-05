"""Tests de los adaptadores de precios (`earnings_alpha.data.prices`).

**Ninguno de estos tests abre un socket** (salvo los marcados `network`,
excluidos por defecto): todo el tráfico va contra `ScriptedTransport` con los
fixtures de `tests/fixtures/prices/`, construidos contra la forma documentada
de cada API. Todos los fixtures describen el MISMO episodio (split 4:1 el
2020-08-31 y dividendo 0.205 USD ex 2020-09-02, calcado del caso AAPL), de modo
que se puede exigir que los adaptadores de Yahoo, Polygon, Tiingo, Stooq y
Alpaca produzcan **el mismo panel** a partir de formatos distintos.

Cobertura:

1. Matemática del ajuste: factores CRSP ``(close*split + div)/close_prev``,
   `compute_adj_close` (ancla en fin de ventana), `adjust_panel`,
   `total_returns` por eventos y por `adj_close`.
2. Validación de calidad: precios no positivos, OHLC incoherente, saltos
   imposibles, splits sin ajustar, volumen cero prolongado, huecos frente al
   `TradingCalendar`, barras en días no bursátiles, duplicados.
3. Cada proveedor contra su fixture: parsing, zonas horarias, paginación,
   cabeceras de autenticación, símbolos sin datos, credenciales ausentes.
4. Equivalencia entre proveedores sobre el mismo episodio de mercado.
5. Caché por símbolo, registro con prioridades y fallback de la fachada.
"""

from __future__ import annotations

import json
import math
import os
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from earnings_alpha.data.base import (
    DataKind,
    HttpClient,
    HttpResponse,
    ManualClock,
    ProviderRegistry,
    RateLimitPolicy,
    RetryPolicy,
    ScriptedTransport,
)
from earnings_alpha.data.cache import DiskCache
from earnings_alpha.data.prices import (
    DEFAULT_PRICE_PRIORITIES,
    PRICE_COLUMNS,
    PRICES_KIND,
    AlpacaProvider,
    PolygonProvider,
    PriceProvider,
    QualityReport,
    Severity,
    StooqProvider,
    SyntheticProvider,
    TiingoProvider,
    YFinanceProvider,
    _normalize_tickers,
    adjust_panel,
    bars_to_panel,
    compute_adj_close,
    get_bars,
    panel_to_bars,
    register_price_providers,
    total_return_factors,
    total_returns,
    validate_price_panel,
)
from earnings_alpha.data.synthetic import SyntheticMarket
from earnings_alpha.errors import (
    ConfigError,
    DataQualityError,
    InsufficientHistory,
    ProviderUnavailable,
)
from earnings_alpha.pit import get_calendar

# ---------------------------------------------------------------------------
# Utilidades comunes
# ---------------------------------------------------------------------------

FIXTURES = Path(__file__).parent / "fixtures" / "prices"

CAL = get_calendar()

TRUTH: dict[str, Any] = json.loads((FIXTURES / "ground_truth.json").read_text("utf-8"))
DATES = pd.DatetimeIndex([pd.Timestamp(d) for d in TRUTH["dates"]], name="date")
START, END = date(2020, 8, 24), date(2020, 9, 2)
FACTORS = np.array(
    [np.nan if f is None else float(f) for f in TRUTH["total_return_factors"]]
)
ADJ_TRUTH = np.array(TRUTH["adj_close_anchored_end"], dtype=float)

FAST_RATE = RateLimitPolicy(1_000_000.0, burst=1000, note="sin limitación en el test")
NO_BACKOFF = RetryPolicy(max_retries=1, backoff_base_s=0.0, jitter="none")

requires_network = pytest.mark.skipif(
    os.environ.get("EARNINGS_ALPHA_RUN_NETWORK", "").lower() not in {"1", "true", "yes"},
    reason="requiere red real; exportar EARNINGS_ALPHA_RUN_NETWORK=1 para ejecutarlos",
)


def fixture_bytes(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def response(name: str, status: int = 200, content_type: str = "application/json") -> HttpResponse:
    return HttpResponse(
        status,
        fixture_bytes(name),
        {"Content-Type": content_type},
        url=f"https://fixture.local/{name}",
    )


def scripted_http(
    script: Any, provider: str, *, repeat_last: bool = False
) -> tuple[HttpClient, ScriptedTransport]:
    """Cliente HTTP sin red, sin espera real y determinista."""
    clock = ManualClock()
    transport = ScriptedTransport(script, repeat_last=repeat_last, clock=clock)
    http = HttpClient(
        provider,
        transport=transport,
        clock=clock,
        rate=FAST_RATE,
        retry=NO_BACKOFF,
        seed=7,
    )
    return http, transport


def one_ticker_panel(
    closes: Any,
    *,
    ticker: str = "TEST",
    dates: pd.DatetimeIndex | None = None,
    volume: Any = None,
    dividend: Any = None,
    split_factor: Any = None,
) -> pd.DataFrame:
    """Panel canónico de un símbolo para los tests de validación.

    OHLC coherente por construcción (open=close, high=+2 %, low=-2 %);
    `dividend`/`split_factor` a NaN («proveedor sin eventos») salvo que se pasen.
    """
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    if dates is None:
        dates = CAL.sessions(date(2021, 1, 4), date(2021, 12, 31))[:n]
    index = pd.MultiIndex.from_arrays([dates, [ticker] * n], names=["date", "ticker"])
    frame = pd.DataFrame(
        {
            "open": closes,
            "high": closes * 1.02,
            "low": closes * 0.98,
            "close": closes,
            "volume": np.full(n, 1e6) if volume is None else np.asarray(volume, dtype=float),
            "adj_close": closes,
            "dividend": np.nan if dividend is None else np.asarray(dividend, dtype=float),
            "split_factor": (
                np.nan if split_factor is None else np.asarray(split_factor, dtype=float)
            ),
        },
        index=index,
    )
    return frame[list(PRICE_COLUMNS)].astype("float64")


def issue_codes(report: QualityReport, severity: Severity | None = None) -> set[str]:
    return {
        i.code
        for i in report.issues
        if severity is None or i.severity is severity
    }


def assert_canonical(panel: pd.DataFrame) -> None:
    """El contrato §1: MultiIndex (date, ticker) ordenado y columnas completas."""
    assert isinstance(panel.index, pd.MultiIndex)
    assert list(panel.index.names) == ["date", "ticker"]
    assert panel.index.is_monotonic_increasing
    assert list(panel.columns) == list(PRICE_COLUMNS)
    dates = panel.index.get_level_values("date")
    assert dates.tz is None
    assert (dates.normalize() == dates).all()


@pytest.fixture(scope="module")
def market() -> SyntheticMarket:
    return SyntheticMarket(seed=11, n_tickers=6, start="2021-01-04", end="2021-06-30")


# ===========================================================================
# 1. Matemática del ajuste
# ===========================================================================


class TestTotalReturnFactors:
    def test_formula_crsp_exacta(self) -> None:
        """f_t = (close_t*split_t + div_t)/close_{t-1} sobre el episodio de referencia."""
        close = pd.Series(TRUTH["close"], index=DATES, dtype=float)
        div = pd.Series(0.0, index=DATES)
        div.loc[pd.Timestamp(TRUTH["div_day"])] = TRUTH["div_amount"]
        split = pd.Series(1.0, index=DATES)
        split.loc[pd.Timestamp(TRUTH["split_day"])] = TRUTH["split_factor"]
        factors = total_return_factors(close, div, split)
        assert math.isnan(factors.iloc[0])
        np.testing.assert_allclose(factors.to_numpy()[1:], FACTORS[1:], rtol=1e-12)

    def test_split_bien_registrado_no_salta(self) -> None:
        """Con el split en su columna, el factor del día del split es ~3 %, no -74 %."""
        close = pd.Series(TRUTH["close"], index=DATES, dtype=float)
        split = pd.Series(1.0, index=DATES)
        split.loc[pd.Timestamp(TRUTH["split_day"])] = 4.0
        factors = total_return_factors(close, split_factor=split)
        f_split = float(factors.loc[pd.Timestamp(TRUTH["split_day"])])
        assert 1.0 < f_split < 1.05

    def test_sin_eventos_es_cociente_de_cierres(self) -> None:
        close = pd.Series([100.0, 101.0, 99.0])
        factors = total_return_factors(close)
        np.testing.assert_allclose(factors.to_numpy()[1:], [1.01, 99.0 / 101.0])

    def test_nan_en_dividendos_se_rechaza(self) -> None:
        close = pd.Series([100.0, 101.0])
        div = pd.Series([0.0, np.nan])
        with pytest.raises(DataQualityError, match="dividend"):
            total_return_factors(close, div)

    def test_cierre_no_positivo_se_rechaza(self) -> None:
        with pytest.raises(DataQualityError, match="no positivos"):
            total_return_factors(pd.Series([100.0, -1.0]))

    def test_split_no_positivo_se_rechaza(self) -> None:
        close = pd.Series([100.0, 101.0])
        split = pd.Series([1.0, 0.0])
        with pytest.raises(DataQualityError, match="split_factor"):
            total_return_factors(close, split_factor=split)

    def test_serie_vacia(self) -> None:
        with pytest.raises(InsufficientHistory):
            total_return_factors(pd.Series([], dtype=float))


class TestComputeAdjClose:
    def test_ancla_y_cocientes(self) -> None:
        """adj_T == close_T y adj_t/adj_{t-1} == f_t, exactamente."""
        close = pd.Series(TRUTH["close"], index=DATES, dtype=float)
        div = pd.Series(0.0, index=DATES)
        div.loc[pd.Timestamp(TRUTH["div_day"])] = TRUTH["div_amount"]
        split = pd.Series(1.0, index=DATES)
        split.loc[pd.Timestamp(TRUTH["split_day"])] = TRUTH["split_factor"]
        adj = compute_adj_close(close, div, split)
        assert adj.iloc[-1] == pytest.approx(close.iloc[-1], rel=1e-12)
        np.testing.assert_allclose(
            (adj / adj.shift(1)).to_numpy()[1:], FACTORS[1:], rtol=1e-10
        )
        np.testing.assert_allclose(adj.to_numpy(), ADJ_TRUTH, rtol=1e-9)

    def test_sin_eventos_adj_igual_close(self) -> None:
        close = pd.Series([100.0, 101.0, 99.5])
        adj = compute_adj_close(close)
        np.testing.assert_allclose(adj.to_numpy(), close.to_numpy(), rtol=1e-12)


class TestAdjustPanel:
    def test_reancla_un_panel_recortado(self) -> None:
        """Tras recortar la ventana, el ancla debe moverse al nuevo fin."""
        div = np.zeros(8)
        div[7] = TRUTH["div_amount"]
        split = np.ones(8)
        split[5] = TRUTH["split_factor"]
        panel = one_ticker_panel(
            TRUTH["close"], dates=DATES, dividend=div, split_factor=split
        )
        recortado = panel[panel.index.get_level_values("date") <= "2020-08-28"]
        adjusted = adjust_panel(recortado)
        sub = adjusted.xs("TEST", level="ticker")
        assert sub["adj_close"].iloc[-1] == pytest.approx(sub["close"].iloc[-1])
        np.testing.assert_allclose(
            (sub["adj_close"] / sub["adj_close"].shift(1)).to_numpy()[1:],
            FACTORS[1:5],
            rtol=1e-10,
        )

    def test_sin_eventos_no_se_puede_ajustar(self) -> None:
        panel = one_ticker_panel([100.0, 101.0])
        with pytest.raises(DataQualityError, match="dividend"):
            adjust_panel(panel.drop(columns=["dividend"]))
        with pytest.raises(DataQualityError, match="NaN"):
            adjust_panel(panel)  # dividend/split_factor a NaN


class TestTotalReturns:
    @staticmethod
    def _panel() -> pd.DataFrame:
        div = np.zeros(8)
        div[7] = TRUTH["div_amount"]
        split = np.ones(8)
        split[5] = TRUTH["split_factor"]
        panel = one_ticker_panel(
            TRUTH["close"], dates=DATES, dividend=div, split_factor=split
        )
        return adjust_panel(panel)

    def test_events_reproduce_los_factores(self) -> None:
        rets = total_returns(self._panel(), method="events")
        np.testing.assert_allclose(rets.to_numpy()[1:], FACTORS[1:] - 1.0, rtol=1e-10)

    def test_adj_close_y_events_coinciden_por_construccion(self) -> None:
        panel = self._panel()
        by_events = total_returns(panel, method="events")
        by_adj = total_returns(panel, method="adj_close")
        np.testing.assert_allclose(
            by_events.to_numpy()[1:], by_adj.to_numpy()[1:], rtol=1e-9
        )

    def test_auto_prefiere_events(self) -> None:
        panel = self._panel()
        auto = total_returns(panel)
        np.testing.assert_allclose(
            auto.to_numpy()[1:], FACTORS[1:] - 1.0, rtol=1e-10
        )

    def test_log_devuelve_el_logaritmo_del_factor(self) -> None:
        rets = total_returns(self._panel(), method="events", log=True)
        np.testing.assert_allclose(rets.to_numpy()[1:], np.log(FACTORS[1:]), rtol=1e-10)

    def test_events_sin_eventos_falla(self) -> None:
        panel = one_ticker_panel([100.0, 101.0])  # dividend/split NaN
        with pytest.raises(DataQualityError, match="events"):
            total_returns(panel, method="events")

    def test_adj_close_ausente_falla(self) -> None:
        panel = one_ticker_panel([100.0, 101.0])
        panel["adj_close"] = np.nan
        with pytest.raises(DataQualityError, match="adjusted=True"):
            total_returns(panel, method="adj_close")

    def test_indice_no_canonico_falla(self) -> None:
        frame = pd.DataFrame({"close": [1.0]}, index=pd.Index([0]))
        with pytest.raises(DataQualityError, match="MultiIndex"):
            total_returns(frame)

    def test_primer_dia_de_cada_ticker_es_nan(self) -> None:
        panel = pd.concat([self._panel(), one_ticker_panel(
            [50.0, 51.0], ticker="OTRO", dates=DATES[:2],
            dividend=np.zeros(2), split_factor=np.ones(2),
        )]).sort_index()
        rets = total_returns(panel, method="events")
        firsts = rets.groupby(level="ticker").head(1)
        assert firsts.isna().all()


# ===========================================================================
# 2. Interoperabilidad con types.Bar
# ===========================================================================


class TestBarsInterop:
    def test_ida_y_vuelta(self, market: SyntheticMarket) -> None:
        ticker = market.tickers[0]
        bars = market.bars(ticker)[:10]
        panel = bars_to_panel(bars)
        assert_canonical(panel)
        assert panel["dividend"].isna().all()  # Bar no transporta eventos
        vuelta = panel_to_bars(panel)
        assert len(vuelta) == len(bars)
        assert [b.close for b in vuelta] == pytest.approx([b.close for b in bars])
        assert [b.adj_close for b in vuelta] == pytest.approx(
            [b.adj_close for b in bars]
        )
        assert all(b.ticker == ticker for b in vuelta)

    def test_lista_vacia(self) -> None:
        with pytest.raises(InsufficientHistory):
            bars_to_panel([])


# ===========================================================================
# 3. Validación de calidad
# ===========================================================================


class TestValidacionCalidad:
    def test_panel_limpio_no_tiene_hallazgos(self) -> None:
        closes = 100.0 * np.cumprod(1.0 + np.linspace(-0.01, 0.01, 20))
        report = validate_price_panel(one_ticker_panel(closes), calendar=CAL)
        assert report.ok
        assert report.issues == []
        assert "sin hallazgos" in report.describe()

    def test_precio_no_positivo_es_error(self) -> None:
        closes = np.full(10, 100.0)
        closes[4] = -5.0
        report = validate_price_panel(one_ticker_panel(closes), calendar=CAL)
        assert "non_positive_price" in issue_codes(report, Severity.ERROR)
        assert not report.ok
        with pytest.raises(DataQualityError, match="non_positive_price"):
            report.raise_if_errors()

    def test_nan_en_ohlc_es_error(self) -> None:
        closes = np.full(10, 100.0)
        panel = one_ticker_panel(closes)
        panel.iloc[3, panel.columns.get_loc("high")] = np.nan
        report = validate_price_panel(panel, calendar=CAL)
        assert "non_positive_price" in issue_codes(report, Severity.ERROR)

    def test_ohlc_incoherente_es_error(self) -> None:
        panel = one_ticker_panel(np.full(10, 100.0))
        panel.iloc[2, panel.columns.get_loc("close")] = 110.0  # > high (102)
        report = validate_price_panel(panel, calendar=CAL)
        assert "ohlc_inconsistent" in issue_codes(report, Severity.ERROR)

    def test_salto_imposible_es_error(self) -> None:
        closes = np.full(10, 100.0)
        closes[5:] = 450.0  # x4.5: no coincide con un cociente de split típico
        report = validate_price_panel(one_ticker_panel(closes), calendar=CAL)
        assert "impossible_jump" in issue_codes(report, Severity.ERROR)

    def test_movimiento_grande_es_aviso(self) -> None:
        closes = np.full(10, 100.0)
        closes[5:] = 140.0  # +40 %: posible (biotecnología en un evento), revisable
        report = validate_price_panel(one_ticker_panel(closes), calendar=CAL)
        assert "large_move" in issue_codes(report, Severity.WARNING)
        assert report.ok

    def test_split_sin_ajustar_con_eventos_conocidos_es_error(self) -> None:
        """El proveedor afirma «no hubo split» y el precio cae exactamente 2:1."""
        closes = np.full(10, 100.0)
        closes[5:] = 50.0
        panel = one_ticker_panel(
            closes, dividend=np.zeros(10), split_factor=np.ones(10)
        )
        report = validate_price_panel(panel, calendar=CAL)
        assert "unadjusted_split" in issue_codes(report, Severity.ERROR)

    def test_salto_tipo_split_sin_metadatos_es_aviso(self) -> None:
        """Sin columna de splits no se puede saber si el 2:1 es real: aviso."""
        closes = np.full(10, 100.0)
        closes[5:] = 50.0
        report = validate_price_panel(one_ticker_panel(closes), calendar=CAL)
        assert "split_like_jump" in issue_codes(report, Severity.WARNING)
        assert "unadjusted_split" not in issue_codes(report)

    def test_split_bien_registrado_no_es_hallazgo(self) -> None:
        div = np.zeros(8)
        div[7] = TRUTH["div_amount"]
        split = np.ones(8)
        split[5] = TRUTH["split_factor"]
        panel = adjust_panel(
            one_ticker_panel(TRUTH["close"], dates=DATES, dividend=div, split_factor=split)
        )
        report = validate_price_panel(panel, calendar=CAL)
        assert report.issues == []

    def test_volumen_cero_corto_es_aviso(self) -> None:
        vol = np.full(30, 1e6)
        vol[10:15] = 0.0  # 5 sesiones
        panel = one_ticker_panel(np.full(30, 100.0), volume=vol)
        report = validate_price_panel(panel, calendar=CAL)
        assert "zero_volume_run" in issue_codes(report, Severity.WARNING)
        assert report.ok

    def test_volumen_cero_un_mes_es_error(self) -> None:
        vol = np.full(30, 1e6)
        vol[3:24] = 0.0  # 21 sesiones: un mes bursátil sin negociar
        panel = one_ticker_panel(np.full(30, 100.0), volume=vol)
        report = validate_price_panel(panel, calendar=CAL)
        assert "zero_volume_run" in issue_codes(report, Severity.ERROR)

    def test_hueco_pequeno_frente_al_calendario_es_aviso(self) -> None:
        dates = CAL.sessions(date(2021, 1, 4), date(2021, 12, 31))[:20]
        keep = dates.delete(10)  # falta 1 de 20 sesiones (5 %)
        panel = one_ticker_panel(np.full(19, 100.0), dates=keep)
        report = validate_price_panel(panel, calendar=CAL)
        assert "calendar_gaps" in issue_codes(report, Severity.WARNING)
        assert report.ok

    def test_hueco_masivo_es_error(self) -> None:
        dates = CAL.sessions(date(2021, 1, 4), date(2021, 12, 31))[:20]
        keep = dates[[0, 1, 2, 3, 19]]  # faltan 15 de 20 (75 %)
        panel = one_ticker_panel(np.full(5, 100.0), dates=keep)
        report = validate_price_panel(panel, calendar=CAL)
        assert "calendar_gaps" in issue_codes(report, Severity.ERROR)

    def test_barra_en_dia_no_bursatil_es_error(self) -> None:
        """Una barra en sábado delata un error de zona horaria en el adaptador."""
        dates = CAL.sessions(date(2021, 1, 4), date(2021, 1, 15))
        with_saturday = dates.append(pd.DatetimeIndex([pd.Timestamp("2021-01-09")]))
        with_saturday = pd.DatetimeIndex(sorted(with_saturday), name="date")
        panel = one_ticker_panel(np.full(len(with_saturday), 100.0), dates=with_saturday)
        report = validate_price_panel(panel, calendar=CAL)
        assert "not_a_session" in issue_codes(report, Severity.ERROR)

    def test_duplicados_y_desorden_son_error(self) -> None:
        panel = one_ticker_panel(np.full(5, 100.0))
        dup = pd.concat([panel, panel.iloc[[2]]]).sort_index()
        report = validate_price_panel(dup, calendar=CAL)
        assert "duplicate_rows" in issue_codes(report, Severity.ERROR)

        unsorted = panel.iloc[::-1]
        report2 = validate_price_panel(unsorted, calendar=CAL)
        assert "unsorted_index" in issue_codes(report2, Severity.ERROR)

    def test_raise_on_error(self) -> None:
        closes = np.full(10, 100.0)
        closes[4] = -5.0
        with pytest.raises(DataQualityError, match="no supera la validación"):
            validate_price_panel(one_ticker_panel(closes), calendar=CAL, raise_on_error=True)

    def test_columnas_obligatorias(self) -> None:
        panel = one_ticker_panel(np.full(5, 100.0)).drop(columns=["volume"])
        with pytest.raises(DataQualityError, match="volume"):
            validate_price_panel(panel, calendar=CAL)

    def test_indice_no_canonico(self) -> None:
        with pytest.raises(DataQualityError, match="MultiIndex"):
            validate_price_panel(pd.DataFrame({"close": [1.0]}))


# ===========================================================================
# 4. SyntheticProvider
# ===========================================================================


class TestSyntheticProvider:
    def test_siempre_disponible_y_declara_kinds(self, market: SyntheticMarket) -> None:
        provider = SyntheticProvider(market)
        assert provider.available() is True
        assert PRICES_KIND in provider.kinds
        assert isinstance(provider, PriceProvider)

    def test_panel_canonico_y_valido(self, market: SyntheticMarket) -> None:
        provider = SyntheticProvider(market)
        tickers = list(market.tickers[:3])
        panel = provider.get_bars(tickers, date(2021, 2, 1), date(2021, 4, 30))
        assert_canonical(panel)
        assert set(panel.index.get_level_values("ticker")) == set(tickers)
        assert panel["close"].notna().all()
        assert panel["adj_close"].notna().all()
        assert panel["dividend"].notna().all()
        assert panel["split_factor"].notna().all()

    def test_retornos_por_eventos_y_por_adj_coinciden(self, market: SyntheticMarket) -> None:
        """El generador usa la misma fórmula CRSP: ambas vías deben coincidir."""
        provider = SyntheticProvider(market)
        panel = provider.get_bars(list(market.tickers[:2]), date(2021, 2, 1), date(2021, 6, 30))
        by_events = total_returns(panel, method="events")
        by_adj = total_returns(panel, method="adj_close")
        mask = by_events.notna()
        np.testing.assert_allclose(
            by_events[mask].to_numpy(), by_adj[mask].to_numpy(), rtol=1e-8, atol=1e-12
        )

    def test_adjusted_false_deja_adj_a_nan(self, market: SyntheticMarket) -> None:
        provider = SyntheticProvider(market)
        panel = provider.get_bars(
            market.tickers[0], date(2021, 2, 1), date(2021, 3, 31), adjusted=False
        )
        assert panel["adj_close"].isna().all()
        assert panel["close"].notna().all()

    def test_ticker_desconocido_por_defecto_lanza(self, market: SyntheticMarket) -> None:
        provider = SyntheticProvider(market)
        with pytest.raises(InsufficientHistory, match="ZZZTOP"):
            provider.get_bars(
                [market.tickers[0], "ZZZTOP"], date(2021, 2, 1), date(2021, 3, 31)
            )

    def test_on_missing_warn_continua_con_el_resto(self, market: SyntheticMarket) -> None:
        provider = SyntheticProvider(market)
        panel = provider.get_bars(
            [market.tickers[0], "ZZZTOP"],
            date(2021, 2, 1),
            date(2021, 3, 31),
            on_missing="warn",
        )
        assert set(panel.index.get_level_values("ticker")) == {market.tickers[0]}

    def test_ningun_simbolo_con_datos_lanza(self, market: SyntheticMarket) -> None:
        provider = SyntheticProvider(market)
        with pytest.raises(InsufficientHistory):
            provider.get_bars(["ZZZTOP"], date(2021, 2, 1), date(2021, 3, 31))

    def test_ventana_invertida_y_tickers_vacios(self, market: SyntheticMarket) -> None:
        provider = SyntheticProvider(market)
        with pytest.raises(ConfigError, match="invertida"):
            provider.get_bars(market.tickers[0], date(2021, 3, 1), date(2021, 2, 1))
        with pytest.raises(ConfigError, match="vacía"):
            provider.get_bars([], date(2021, 2, 1), date(2021, 3, 1))
        with pytest.raises(ConfigError, match="on_missing"):
            provider.get_bars(
                market.tickers[0], date(2021, 2, 1), date(2021, 3, 1),
                on_missing="ignore",  # type: ignore[arg-type]
            )

    def test_normalizacion_de_tickers(self) -> None:
        assert _normalize_tickers(["brk-b", "BRK.B", "aapl"]) == ["BRK.B", "AAPL"]


# ===========================================================================
# 5. YFinanceProvider
# ===========================================================================


def yfinance_provider(script: Any, **kwargs: Any) -> tuple[YFinanceProvider, ScriptedTransport]:
    http, transport = scripted_http(script, "yfinance")
    return YFinanceProvider(http=http, calendar=CAL, **kwargs), transport


class TestYFinanceProvider:
    def test_desajusta_el_split_y_reconstruye_el_cierre_negociado(self) -> None:
        provider, transport = yfinance_provider([response("yfinance_aapl_chart.json")])
        panel = provider.get_bars("AAPL", START, END)
        assert_canonical(panel)
        sub = panel.xs("AAPL", level="ticker")
        assert list(sub.index) == list(DATES)
        # Yahoo sirve 126.00 (base actual); el cierre negociado el 24-08 fue 504.
        np.testing.assert_allclose(sub["close"].to_numpy(), TRUTH["close"], rtol=1e-9)
        np.testing.assert_allclose(sub["open"].to_numpy(), TRUTH["open"], rtol=1e-9)
        np.testing.assert_allclose(sub["volume"].to_numpy(), TRUTH["volume"], rtol=1e-9)

    def test_eventos_y_adj_close(self) -> None:
        provider, _ = yfinance_provider([response("yfinance_aapl_chart.json")])
        sub = provider.get_bars("AAPL", START, END).xs("AAPL", level="ticker")
        assert sub.loc[TRUTH["split_day"], "split_factor"] == pytest.approx(4.0)
        assert sub.loc[TRUTH["div_day"], "dividend"] == pytest.approx(TRUTH["div_amount"])
        assert (sub["split_factor"].drop(pd.Timestamp(TRUTH["split_day"])) == 1.0).all()
        np.testing.assert_allclose(sub["adj_close"].to_numpy(), ADJ_TRUTH, rtol=1e-8)
        # ancla en el fin de la ventana, no en «hoy»
        assert sub["adj_close"].iloc[-1] == pytest.approx(sub["close"].iloc[-1])

    def test_url_y_parametros(self) -> None:
        provider, transport = yfinance_provider([response("yfinance_aapl_chart.json")])
        provider.get_bars("AAPL", START, END)
        req = transport.calls[0]
        assert req.url.endswith("/v8/finance/chart/AAPL")
        assert req.params is not None
        assert req.params["interval"] == "1d"
        assert req.params["events"] == "div,split"

    def test_clases_de_accion_van_con_guion(self) -> None:
        assert YFinanceProvider._symbol("BRK.B") == "BRK-B"

    def test_simbolo_inexistente(self) -> None:
        provider, _ = yfinance_provider([response("yfinance_not_found.json", status=404)])
        with pytest.raises(InsufficientHistory, match="ZZZTOP"):
            provider.get_bars("ZZZTOP", START, END)

    def test_adjusted_false_no_rellena_adj(self) -> None:
        provider, _ = yfinance_provider([response("yfinance_aapl_chart.json")])
        sub = provider.get_bars("AAPL", START, END, adjusted=False).xs(
            "AAPL", level="ticker"
        )
        assert sub["adj_close"].isna().all()
        # pero el des-ajuste del split se aplica igual: close es el negociado
        np.testing.assert_allclose(sub["close"].to_numpy(), TRUTH["close"], rtol=1e-9)


# ===========================================================================
# 6. PolygonProvider
# ===========================================================================


def polygon_provider(script: Any, **kwargs: Any) -> tuple[PolygonProvider, ScriptedTransport]:
    http, transport = scripted_http(script, "polygon")
    return PolygonProvider(http=http, calendar=CAL, **kwargs), transport


POLYGON_SCRIPT = [
    response("polygon_aggs_aapl_page1.json"),
    response("polygon_aggs_aapl_page2.json"),
    response("polygon_dividends_aapl.json"),
    response("polygon_splits_aapl.json"),
]


class TestPolygonProvider:
    def test_sin_credencial_falla_antes_de_tocar_la_red(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("POLYGON_API_KEY", raising=False)
        provider, transport = polygon_provider(POLYGON_SCRIPT)
        with pytest.raises(ProviderUnavailable) as exc:
            provider.get_bars("AAPL", START, END)
        assert "POLYGON_API_KEY" in exc.value.missing_env
        assert transport.n_calls == 0

    def test_pagina_con_next_url_y_reconstruye_el_ajuste(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("POLYGON_API_KEY", "test-key")
        provider, transport = polygon_provider(POLYGON_SCRIPT)
        panel = provider.get_bars("AAPL", START, END)
        assert_canonical(panel)
        sub = panel.xs("AAPL", level="ticker")
        # aggs crudos (epoch ms medianoche ET) -> fechas de sesión correctas
        assert list(sub.index) == list(DATES)
        np.testing.assert_allclose(sub["close"].to_numpy(), TRUTH["close"], rtol=1e-9)
        assert sub.loc[TRUTH["split_day"], "split_factor"] == pytest.approx(4.0)
        assert sub.loc[TRUTH["div_day"], "dividend"] == pytest.approx(TRUTH["div_amount"])
        np.testing.assert_allclose(sub["adj_close"].to_numpy(), ADJ_TRUTH, rtol=1e-8)
        # 4 peticiones: aggs página 1, aggs página 2 (next_url), dividends, splits
        assert transport.n_calls == 4
        assert "cursor=fixture-cursor-1" in transport.calls[1].url
        assert transport.calls[1].params is None  # next_url ya lleva el cursor
        assert "/v3/reference/dividends" in transport.calls[2].url
        assert "/v3/reference/splits" in transport.calls[3].url

    def test_autenticacion_por_cabecera_no_por_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("POLYGON_API_KEY", "sekret")
        provider, transport = polygon_provider(POLYGON_SCRIPT)
        provider.get_bars("AAPL", START, END)
        for req in transport.calls:
            assert req.headers is not None
            assert req.headers.get("Authorization") == "Bearer sekret"
            assert "sekret" not in req.url
            assert "sekret" not in str(req.params or {})

    def test_pide_los_aggs_sin_ajustar(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("POLYGON_API_KEY", "test-key")
        provider, transport = polygon_provider(POLYGON_SCRIPT)
        provider.get_bars("AAPL", START, END)
        first = transport.calls[0]
        assert first.params is not None
        assert first.params["adjusted"] == "false"
        assert "/v2/aggs/ticker/AAPL/range/1/day/2020-08-24/2020-09-02" in first.url

    def test_adjusted_false_evita_las_peticiones_de_eventos(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("POLYGON_API_KEY", "test-key")
        provider, transport = polygon_provider(POLYGON_SCRIPT[:2])
        panel = provider.get_bars("AAPL", START, END, adjusted=False)
        assert transport.n_calls == 2  # solo las dos páginas de aggs
        assert panel["adj_close"].isna().all()

    def test_sin_resultados_es_simbolo_sin_datos(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("POLYGON_API_KEY", "test-key")
        provider, _ = polygon_provider([response("polygon_aggs_empty.json")])
        with pytest.raises(InsufficientHistory, match="ZZZTOP"):
            provider.get_bars("ZZZTOP", START, END)

    def test_status_error_es_data_quality(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("POLYGON_API_KEY", "test-key")
        body = json.dumps({"status": "ERROR", "error": "Unknown API Key"}).encode()
        provider, _ = polygon_provider([HttpResponse(200, body)])
        with pytest.raises(DataQualityError, match="Unknown API Key"):
            provider.get_bars("AAPL", START, END)


# ===========================================================================
# 7. TiingoProvider
# ===========================================================================


def tiingo_provider(script: Any, **kwargs: Any) -> tuple[TiingoProvider, ScriptedTransport]:
    http, transport = scripted_http(script, "tiingo")
    return TiingoProvider(http=http, calendar=CAL, **kwargs), transport


class TestTiingoProvider:
    def test_sin_credencial_falla_antes_de_tocar_la_red(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("TIINGO_API_KEY", raising=False)
        provider, transport = tiingo_provider([response("tiingo_aapl.json")])
        with pytest.raises(ProviderUnavailable) as exc:
            provider.get_bars("AAPL", START, END)
        assert "TIINGO_API_KEY" in exc.value.missing_env
        assert transport.n_calls == 0

    def test_divcash_splitfactor_y_ajuste(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TIINGO_API_KEY", "test-token")
        provider, transport = tiingo_provider([response("tiingo_aapl.json")])
        panel = provider.get_bars("AAPL", START, END)
        assert_canonical(panel)
        sub = panel.xs("AAPL", level="ticker")
        np.testing.assert_allclose(sub["close"].to_numpy(), TRUTH["close"], rtol=1e-9)
        assert sub.loc[TRUTH["split_day"], "split_factor"] == pytest.approx(4.0)
        assert sub.loc[TRUTH["div_day"], "dividend"] == pytest.approx(TRUTH["div_amount"])
        np.testing.assert_allclose(sub["adj_close"].to_numpy(), ADJ_TRUTH, rtol=1e-8)
        req = transport.calls[0]
        assert req.headers is not None
        assert req.headers.get("Authorization") == "Token test-token"
        assert "/tiingo/daily/AAPL/prices" in req.url

    def test_la_fecha_es_el_literal_sin_conversion_de_zona(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`2020-08-24T00:00:00.000Z` es la fecha de negociación: convertirla a
        Nueva York la retrasaría al 23-08 (off-by-one clásico)."""
        monkeypatch.setenv("TIINGO_API_KEY", "test-token")
        provider, _ = tiingo_provider([response("tiingo_aapl.json")])
        sub = provider.get_bars("AAPL", START, END).xs("AAPL", level="ticker")
        assert sub.index[0] == pd.Timestamp("2020-08-24")
        assert sub.index[-1] == pd.Timestamp("2020-09-02")

    def test_404_es_simbolo_sin_datos(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TIINGO_API_KEY", "test-token")
        provider, _ = tiingo_provider(
            [HttpResponse(404, b'{"detail": "Not found."}')]
        )
        with pytest.raises(InsufficientHistory, match="ZZZTOP"):
            provider.get_bars("ZZZTOP", START, END)

    def test_detail_not_found_con_200_tambien(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TIINGO_API_KEY", "test-token")
        provider, _ = tiingo_provider([response("tiingo_not_found.json")])
        with pytest.raises(InsufficientHistory):
            provider.get_bars("ZZZTOP", START, END)

    def test_cache_por_simbolo_evita_la_segunda_peticion(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("TIINGO_API_KEY", "test-token")
        cache = DiskCache(tmp_path / "cache")
        provider, transport = tiingo_provider([response("tiingo_aapl.json")], cache=cache)
        first = provider.get_bars("AAPL", START, END)
        assert transport.n_calls == 1
        second = provider.get_bars("AAPL", START, END)
        assert transport.n_calls == 1  # servido íntegramente de caché
        pd.testing.assert_frame_equal(first, second)
        # la ventana terminó hace años: la entrada debe ser inmutable (sin TTL)
        meta = cache.stat(
            PRICES_KIND,
            "tiingo",
            {"ticker": "AAPL", "start": START, "end": END, "adjusted": True},
        )
        assert meta is not None
        assert meta.immutable is True


# ===========================================================================
# 8. StooqProvider
# ===========================================================================


def stooq_provider(script: Any, **kwargs: Any) -> tuple[StooqProvider, ScriptedTransport]:
    http, transport = scripted_http(script, "stooq")
    return StooqProvider(http=http, calendar=CAL, **kwargs), transport


class TestStooqProvider:
    def test_serie_ajustada_close_igual_adj(self) -> None:
        provider, transport = stooq_provider(
            [response("stooq_aapl.csv", content_type="text/csv")]
        )
        panel = provider.get_bars("AAPL", START, END)
        assert_canonical(panel)
        sub = panel.xs("AAPL", level="ticker")
        # Stooq solo tiene la serie ajustada: close y adj_close son la misma
        np.testing.assert_allclose(
            sub["close"].to_numpy(), sub["adj_close"].to_numpy(), rtol=1e-12
        )
        np.testing.assert_allclose(sub["adj_close"].to_numpy(), ADJ_TRUTH, atol=1e-3)
        # y no puede afirmar nada sobre eventos: NaN, no 0/1
        assert sub["dividend"].isna().all()
        assert sub["split_factor"].isna().all()
        assert transport.calls[0].params is not None
        assert transport.calls[0].params["s"] == "aapl.us"

    def test_cocientes_coinciden_con_el_retorno_total(self) -> None:
        provider, _ = stooq_provider([response("stooq_aapl.csv", content_type="text/csv")])
        panel = provider.get_bars("AAPL", START, END)
        rets = total_returns(panel, method="adj_close")
        np.testing.assert_allclose(
            rets.to_numpy()[1:], FACTORS[1:] - 1.0, atol=2e-4
        )

    def test_adjusted_false_es_capacidad_inexistente(self) -> None:
        provider, transport = stooq_provider([response("stooq_aapl.csv")])
        with pytest.raises(DataQualityError, match="ajustados"):
            provider.get_bars("AAPL", START, END, adjusted=False)
        assert transport.n_calls == 0

    def test_no_data_es_simbolo_sin_datos(self) -> None:
        provider, _ = stooq_provider(
            [response("stooq_no_data.txt", content_type="text/plain")]
        )
        with pytest.raises(InsufficientHistory, match="ZZZTOP"):
            provider.get_bars("ZZZTOP", START, END)

    def test_simbolo_stooq(self) -> None:
        assert StooqProvider._symbol("BRK.B") == "brk-b.us"


# ===========================================================================
# 9. AlpacaProvider
# ===========================================================================


ALPACA_SCRIPT = [
    response("alpaca_bars_raw_page1.json"),
    response("alpaca_bars_raw_page2.json"),
    response("alpaca_bars_all.json"),
]


def alpaca_provider(script: Any, **kwargs: Any) -> tuple[AlpacaProvider, ScriptedTransport]:
    http, transport = scripted_http(script, "alpaca")
    return AlpacaProvider(http=http, calendar=CAL, **kwargs), transport


def _set_alpaca_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPACA_API_KEY_ID", "key-id")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "key-secret")


class TestAlpacaProvider:
    def test_sin_credenciales_enumera_las_dos_variables(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ALPACA_API_KEY_ID", raising=False)
        monkeypatch.delenv("ALPACA_API_SECRET_KEY", raising=False)
        provider, transport = alpaca_provider(ALPACA_SCRIPT)
        with pytest.raises(ProviderUnavailable) as exc:
            provider.get_bars("AAPL", START, END)
        assert set(exc.value.missing_env) == {"ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY"}
        assert transport.n_calls == 0

    def test_lote_paginado_y_dos_pasadas_raw_all(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_alpaca_env(monkeypatch)
        provider, transport = alpaca_provider(ALPACA_SCRIPT)
        panel = provider.get_bars(
            ["AAPL", "MSFT", "ZZZTOP"], START, END, on_missing="warn"
        )
        assert_canonical(panel)
        assert set(panel.index.get_level_values("ticker")) == {"AAPL", "MSFT"}
        assert transport.n_calls == 3
        c0, c1, c2 = transport.calls
        assert c0.params is not None and c1.params is not None and c2.params is not None
        assert c0.params["symbols"] == "AAPL,MSFT,ZZZTOP"
        assert c0.params["adjustment"] == "raw"
        assert c1.params["page_token"] == "fixture-token-1"
        assert c2.params["adjustment"] == "all"
        assert c0.headers is not None
        assert c0.headers.get("APCA-API-KEY-ID") == "key-id"
        assert c0.headers.get("APCA-API-SECRET-KEY") == "key-secret"

    def test_raw_es_el_negociado_y_all_da_los_cocientes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_alpaca_env(monkeypatch)
        provider, _ = alpaca_provider(ALPACA_SCRIPT)
        panel = provider.get_bars(["AAPL", "MSFT"], START, END)
        aapl = panel.xs("AAPL", level="ticker")
        np.testing.assert_allclose(aapl["close"].to_numpy(), TRUTH["close"], rtol=1e-9)
        # Alpaca no publica los eventos en /bars: desconocido, no 0/1
        assert aapl["dividend"].isna().all()
        assert aapl["split_factor"].isna().all()
        # los COCIENTES del adj_close (lo único contractual) son el retorno total
        ratios = (aapl["adj_close"] / aapl["adj_close"].shift(1)).to_numpy()[1:]
        np.testing.assert_allclose(ratios, FACTORS[1:], rtol=1e-5)
        # sin eventos corporativos, raw == all
        msft = panel.xs("MSFT", level="ticker")
        np.testing.assert_allclose(
            msft["adj_close"].to_numpy(), msft["close"].to_numpy(), rtol=1e-12
        )
        np.testing.assert_allclose(msft["close"].to_numpy(), TRUTH["msft_close"], rtol=1e-9)

    def test_adjusted_false_hace_solo_la_pasada_raw(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_alpaca_env(monkeypatch)
        provider, transport = alpaca_provider(ALPACA_SCRIPT[:2])
        panel = provider.get_bars(["AAPL", "MSFT"], START, END, adjusted=False)
        assert transport.n_calls == 2
        assert panel["adj_close"].isna().all()

    def test_cache_por_simbolo(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _set_alpaca_env(monkeypatch)
        cache = DiskCache(tmp_path / "cache")
        provider, transport = alpaca_provider(ALPACA_SCRIPT, cache=cache)
        first = provider.get_bars(["AAPL", "MSFT"], START, END)
        assert transport.n_calls == 3
        second = provider.get_bars(["AAPL", "MSFT"], START, END)
        assert transport.n_calls == 3  # todo de caché
        pd.testing.assert_frame_equal(first, second)


# ===========================================================================
# 10. Equivalencia entre proveedores
# ===========================================================================


class TestEquivalenciaEntreProveedores:
    """Los fixtures describen el mismo suceso en cinco formatos distintos: los
    adaptadores deben converger al mismo panel (mismos cierres negociados,
    mismos eventos, mismos factores de retorno total)."""

    def _paneles(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, pd.DataFrame]:
        monkeypatch.setenv("POLYGON_API_KEY", "k")
        monkeypatch.setenv("TIINGO_API_KEY", "k")
        _set_alpaca_env(monkeypatch)
        yf, _ = yfinance_provider([response("yfinance_aapl_chart.json")])
        pg, _ = polygon_provider(POLYGON_SCRIPT)
        tg, _ = tiingo_provider([response("tiingo_aapl.json")])
        al, _ = alpaca_provider(ALPACA_SCRIPT)
        return {
            "yfinance": yf.get_bars("AAPL", START, END),
            "polygon": pg.get_bars("AAPL", START, END),
            "tiingo": tg.get_bars("AAPL", START, END),
            "alpaca": al.get_bars(["AAPL", "MSFT"], START, END).loc[
                (slice(None), "AAPL"), :
            ],
        }

    def test_cierre_negociado_identico(self, monkeypatch: pytest.MonkeyPatch) -> None:
        paneles = self._paneles(monkeypatch)
        for name, panel in paneles.items():
            np.testing.assert_allclose(
                panel["close"].to_numpy(),
                TRUTH["close"],
                rtol=1e-9,
                err_msg=f"cierre negociado de {name}",
            )

    def test_mismo_retorno_total_en_todos(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name, panel in self._paneles(monkeypatch).items():
            rets = total_returns(panel, method="adj_close")
            np.testing.assert_allclose(
                rets.to_numpy()[1:],
                FACTORS[1:] - 1.0,
                rtol=1e-5,
                err_msg=f"retorno total de {name}",
            )

    def test_eventos_identicos_donde_se_informan(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paneles = self._paneles(monkeypatch)
        for name in ("yfinance", "polygon", "tiingo"):
            sub = paneles[name].xs("AAPL", level="ticker")
            assert sub.loc[TRUTH["split_day"], "split_factor"] == pytest.approx(
                4.0
            ), name
            assert sub.loc[TRUTH["div_day"], "dividend"] == pytest.approx(
                TRUTH["div_amount"]
            ), name


# ===========================================================================
# 11. Registro y fachada
# ===========================================================================


class _FailingProvider:
    """Proveedor que se declara disponible pero falla al pedirle datos."""

    name = "roto"
    kinds = (PRICES_KIND,)

    def __init__(self, exc: Exception) -> None:
        self.exc = exc
        self.calls = 0

    def available(self) -> bool:
        return True

    def get_bars(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        self.calls += 1
        raise self.exc


class TestRegistroYFachada:
    def test_prioridades_por_defecto(self) -> None:
        reg = register_price_providers(ProviderRegistry())
        names = [p.name for p in reg.providers_for(PRICES_KIND)]
        assert names == ["tiingo", "polygon", "alpaca", "stooq", "yfinance", "synthetic"]
        assert DEFAULT_PRICE_PRIORITIES["synthetic"] == 0
        assert reg.priority_of(PRICES_KIND, "tiingo") == 90

    def test_include_y_priorities(self) -> None:
        reg = register_price_providers(
            ProviderRegistry(),
            include=["stooq", "synthetic"],
            priorities={"synthetic": 99},
        )
        names = [p.name for p in reg.providers_for(PRICES_KIND)]
        assert names == ["synthetic", "stooq"]
        with pytest.raises(ConfigError, match="desconocidos"):
            register_price_providers(ProviderRegistry(), include=["bloomberg"])

    def test_declaran_el_kind_de_corporate_actions_quien_lo_tiene(self) -> None:
        assert DataKind.CORPORATE_ACTIONS.value in TiingoProvider.kinds
        assert DataKind.CORPORATE_ACTIONS.value in PolygonProvider.kinds
        assert DataKind.CORPORATE_ACTIONS.value not in StooqProvider.kinds
        assert DataKind.CORPORATE_ACTIONS.value not in AlpacaProvider.kinds

    def test_fachada_cae_al_sintetico_si_el_primario_falla(
        self, market: SyntheticMarket
    ) -> None:
        reg = ProviderRegistry()
        roto = _FailingProvider(ProviderUnavailable("roto", "se cayó a mitad"))
        reg.register(PRICES_KIND, roto, 100)
        reg.register(PRICES_KIND, SyntheticProvider(market), 0)
        panel = get_bars(
            market.tickers[0], date(2021, 2, 1), date(2021, 3, 31), registry=reg
        )
        assert_canonical(panel)
        assert roto.calls == 1
        assert reg.is_quarantined("roto")  # no se le volverá a preguntar en un rato

    def test_data_quality_no_provoca_fallback(self, market: SyntheticMarket) -> None:
        """Datos mal parseados son un bug reproducible, no una razón para cambiar
        de fuente en silencio (data.base.FALLBACK_ERRORS)."""
        reg = ProviderRegistry()
        roto = _FailingProvider(DataQualityError("parser roto"))
        reg.register(PRICES_KIND, roto, 100)
        reg.register(PRICES_KIND, SyntheticProvider(market), 0)
        with pytest.raises(DataQualityError, match="parser roto"):
            get_bars(market.tickers[0], date(2021, 2, 1), date(2021, 3, 31), registry=reg)

    def test_sin_proveedores_es_error_accionable(self) -> None:
        reg = ProviderRegistry()
        with pytest.raises(ProviderUnavailable, match="prices"):
            get_bars("AAPL", date(2021, 2, 1), date(2021, 3, 31), registry=reg)


# ===========================================================================
# 12. Contra la API real (excluidos por defecto)
# ===========================================================================


@pytest.mark.network
@requires_network
def test_stooq_live() -> None:
    """Stooq real: sin credencial. Verifica forma, calendario y ajuste."""
    provider = StooqProvider()
    panel = provider.get_bars("AAPL", date(2023, 1, 3), date(2023, 3, 31))
    assert_canonical(panel)
    assert len(panel) > 50


@pytest.mark.network
@requires_network
def test_yfinance_live() -> None:
    """Yahoo real: el episodio AAPL de agosto de 2020 debe cuadrar con la
    verdad-terreno de los fixtures (split 4:1 el 31-08, cierre ~499-504 USD
    antes del split)."""
    provider = YFinanceProvider()
    panel = provider.get_bars("AAPL", date(2020, 8, 24), date(2020, 9, 2))
    sub = panel.xs("AAPL", level="ticker")
    assert sub.loc["2020-08-31", "split_factor"] == pytest.approx(4.0)
    assert 480 < sub.loc["2020-08-24", "close"] < 520  # base pre-split


@pytest.mark.network
@requires_network
@pytest.mark.skipif(not os.environ.get("TIINGO_API_KEY"), reason="sin TIINGO_API_KEY")
def test_tiingo_live() -> None:
    provider = TiingoProvider()
    panel = provider.get_bars("AAPL", date(2020, 8, 24), date(2020, 9, 2))
    sub = panel.xs("AAPL", level="ticker")
    assert sub.loc["2020-08-31", "split_factor"] == pytest.approx(4.0)


@pytest.mark.network
@requires_network
@pytest.mark.skipif(not os.environ.get("POLYGON_API_KEY"), reason="sin POLYGON_API_KEY")
def test_polygon_live() -> None:
    provider = PolygonProvider()
    panel = provider.get_bars("AAPL", date(2023, 1, 3), date(2023, 1, 31))
    assert_canonical(panel)


@pytest.mark.network
@requires_network
@pytest.mark.skipif(
    not (os.environ.get("ALPACA_API_KEY_ID") and os.environ.get("ALPACA_API_SECRET_KEY")),
    reason="sin credenciales de Alpaca",
)
def test_alpaca_live() -> None:
    provider = AlpacaProvider(feed="iex")
    panel = provider.get_bars(["AAPL", "MSFT"], date(2023, 1, 3), date(2023, 1, 31))
    assert_canonical(panel)
