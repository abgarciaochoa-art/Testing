"""Tests de los adaptadores de calendario/consenso, opciones y flujo.

**Ningún test abre un socket** (salvo los marcados `network`, excluidos por
defecto): todo el tráfico va contra `ScriptedTransport` con los fixtures de
`tests/fixtures/alt/`, construidos contra la forma documentada de cada API
(`docs/research/data_sources.md`). El dataset externo real
`data/external/consenso/consenso_master.parquet` sí se lee, porque está en el
repo y es parte del contrato del `ExternalConsensusProvider`.

Cobertura:

1. `data.estimates`: esquema canónico, sesiones BMO/AMC, hora nominal,
   proveedores FMP/Finnhub/EODHD/Nasdaq contra fixtures, el proveedor externo
   real (deduplicación, banderas PIT, advertencia de supervivencia), sintético
   con vintages verdaderos y el guardián `require_point_in_time`.
2. `data.options`: matemática BSM (paridad, griegas contra la referencia del
   generador y contra diferencias finitas), `implied_vol` con Newton +
   bisección (ida y vuelta, cotas de no-arbitraje, ruta de bisección),
   forward implícito por paridad put-call, esquema canónico y disponibilidad
   PIT del open interest, y los cuatro proveedores.
3. `data.shortinterest`: calendario quincenal FINRA, ciclo de liquidación
   T+3/T+2/T+1 (paridad con `events.flow`), fecha de publicación como
   `available_at`, guardias anti-look-ahead, Query API de FINRA (OAuth,
   paginación), ficheros Reg SHO y el semanal ATS con su retardo Tier 1.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from earnings_alpha.data.base import (
    HttpClient,
    HttpRequest,
    HttpResponse,
    ProviderRegistry,
    RateLimitPolicy,
    RetryPolicy,
    ScriptedTransport,
)
from earnings_alpha.data.estimates import (
    CALENDAR_COLUMNS,
    CALENDAR_KIND,
    CONSENSUS_COLUMNS,
    ESTIMATES_KIND,
    EODHDEstimatesProvider,
    ExternalConsensusProvider,
    FinnhubEstimatesProvider,
    FMPEstimatesProvider,
    NasdaqEstimatesProvider,
    SyntheticEstimatesProvider,
    calendar_to_events,
    events_to_frame,
    fiscal_quarter_label,
    get_consensus,
    get_earnings_calendar,
    nominal_announced_at,
    register_estimates_providers,
    require_point_in_time,
)
from earnings_alpha.data.options import (
    CHAIN_COLUMNS,
    OPTIONS_KIND,
    ORATSOptionsProvider,
    PolygonOptionsProvider,
    SyntheticOptionsProvider,
    TradierOptionsProvider,
    bs_greeks,
    bs_price,
    chain_availability,
    filter_chain,
    get_option_chain,
    implied_forward,
    implied_forward_table,
    implied_vol,
    normalize_chain,
    occ_symbol,
    parse_occ_symbol,
    register_options_providers,
)
from earnings_alpha.data.shortinterest import (
    FIRST_T1_SETTLEMENT,
    FIRST_T2_SETTLEMENT,
    OFF_EXCHANGE_COLUMNS,
    OFF_EXCHANGE_KIND,
    SHORT_INTEREST_COLUMNS,
    SHORT_INTEREST_KIND,
    SHORT_VOLUME_COLUMNS,
    SHORT_VOLUME_KIND,
    FinraOffExchangeProvider,
    FinraShortInterestProvider,
    RegShoShortVolumeProvider,
    SyntheticFlowProvider,
    ats_publication_date,
    finra_publication_date,
    finra_settlement_dates,
    get_off_exchange,
    get_short_interest,
    off_exchange_share_daily,
    published_asof,
    register_flow_providers,
    settlement_cycle_lag,
    settlement_to_trade_date,
    validate_short_interest,
)
from earnings_alpha.data.synthetic import SyntheticMarket
from earnings_alpha.errors import (
    ConfigError,
    DataQualityError,
    InsufficientHistory,
    LookAheadError,
    ProviderUnavailable,
)
from earnings_alpha.pit import get_calendar, tradable_date
from earnings_alpha.types import Session

# ---------------------------------------------------------------------------
# Utilidades comunes
# ---------------------------------------------------------------------------

FIXTURES = Path(__file__).parent / "fixtures" / "alt"
CAL = get_calendar()

FAST_RATE = RateLimitPolicy(1_000_000.0, burst=1000, note="sin limitación en el test")
NO_BACKOFF = RetryPolicy(max_retries=0, backoff_base_s=0.0, jitter="none")

requires_network = pytest.mark.skipif(
    os.environ.get("EARNINGS_ALPHA_RUN_NETWORK", "").lower() not in {"1", "true", "yes"},
    reason="requiere red real; exportar EARNINGS_ALPHA_RUN_NETWORK=1 para ejecutarlos",
)

ALL_PROVIDER_ENV = [
    "FMP_API_KEY",
    "FINNHUB_API_KEY",
    "EODHD_API_KEY",
    "POLYGON_API_KEY",
    "ORATS_TOKEN",
    "TRADIER_ACCESS_TOKEN",
    "FINRA_API_CLIENT_ID",
    "FINRA_API_CLIENT_SECRET",
]


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Borra todas las credenciales para que la resolución sea determinista."""
    for key in ALL_PROVIDER_ENV:
        monkeypatch.delenv(key, raising=False)


def fixture_bytes(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def response(
    name: str, status: int = 200, content_type: str = "application/json"
) -> HttpResponse:
    return HttpResponse(
        status,
        fixture_bytes(name),
        {"Content-Type": content_type},
        url=f"https://fixture.local/{name}",
    )


def json_response(payload: Any, status: int = 200) -> HttpResponse:
    return HttpResponse(
        status,
        json.dumps(payload).encode("utf-8"),
        {"Content-Type": "application/json"},
        url="https://fixture.local/inline",
    )


def scripted_http(
    provider: str, script: Any, *, repeat_last: bool = False
) -> tuple[HttpClient, ScriptedTransport]:
    transport = ScriptedTransport(script, repeat_last=repeat_last)
    client = HttpClient(
        provider, transport=transport, rate=FAST_RATE, retry=NO_BACKOFF
    )
    return client, transport


@pytest.fixture(scope="module")
def market() -> SyntheticMarket:
    return SyntheticMarket(seed=99, n_tickers=6, start="2021-01-04", end="2021-12-31")


@pytest.fixture(scope="module")
def external() -> ExternalConsensusProvider:
    provider = ExternalConsensusProvider()
    if not provider.available():  # pragma: no cover - el fichero está en el repo
        pytest.skip("falta data/external/consenso/consenso_master.parquet")
    return provider


# ===========================================================================
# 1. estimates: esquema y utilidades
# ===========================================================================


class TestEstimatesSchema:
    def test_fiscal_quarter_label(self) -> None:
        assert fiscal_quarter_label("2020-06-27") == "2020Q2"
        assert fiscal_quarter_label(date(2019, 12, 28)) == "2019Q4"
        assert fiscal_quarter_label(pd.Timestamp("2021-01-02")) == "2021Q1"
        assert fiscal_quarter_label(None) == ""
        assert fiscal_quarter_label(pd.NaT) == ""

    def test_nominal_announced_at_dst(self) -> None:
        # Agosto (EDT, UTC-4): BMO 07:30 ET -> 11:30 UTC; AMC 16:30 -> 20:30 UTC.
        assert nominal_announced_at("2020-08-27", Session.BMO) == dt.datetime(
            2020, 8, 27, 11, 30
        )
        assert nominal_announced_at("2020-08-27", Session.AMC) == dt.datetime(
            2020, 8, 27, 20, 30
        )
        # Enero (EST, UTC-5): la conversión debe cambiar con el DST.
        assert nominal_announced_at("2020-01-28", "bmo") == dt.datetime(
            2020, 1, 28, 12, 30
        )
        # UNKNOWN recibe hora AMC: política conservadora del repo.
        assert nominal_announced_at("2020-08-27", Session.UNKNOWN) == nominal_announced_at(
            "2020-08-27", Session.AMC
        )

    def test_nominal_unknown_is_next_session_tradable(self) -> None:
        ts = nominal_announced_at("2020-08-27", Session.UNKNOWN)
        assert tradable_date(ts, Session.UNKNOWN, CAL) == date(2020, 8, 28)

    def test_require_point_in_time_rejects(self) -> None:
        frame = pd.DataFrame(
            {"is_point_in_time": [True, False], "source": ["a", "b"]}
        )
        with pytest.raises(LookAheadError, match="momentum de revisiones"):
            require_point_in_time(frame)
        ok = pd.DataFrame({"is_point_in_time": [True, True], "source": ["a", "a"]})
        assert require_point_in_time(ok) is ok
        with pytest.raises(DataQualityError):
            require_point_in_time(pd.DataFrame({"x": [1]}))

    def test_events_roundtrip(self, market: SyntheticMarket) -> None:
        provider = SyntheticEstimatesProvider(market)
        frame = provider.earnings_calendar("2021-01-04", "2021-12-31")
        events = calendar_to_events(frame)
        assert len(events) == len(frame)
        back = events_to_frame(events)
        assert list(back.columns) == list(CALENDAR_COLUMNS)
        assert set(back["ticker"]) == set(frame["ticker"])
        merged = back.merge(
            frame, on=["ticker", "period_end"], suffixes=("_b", "_f"), how="inner"
        )
        assert len(merged) == len(frame)
        assert np.allclose(merged["eps_actual_b"], merged["eps_actual_f"])
        assert (merged["session_b"] == merged["session_f"]).all()


# ===========================================================================
# 2. estimates: FMP
# ===========================================================================


class TestFMPEstimates:
    def make(self, monkeypatch: pytest.MonkeyPatch, script: Any) -> tuple[
        FMPEstimatesProvider, ScriptedTransport
    ]:
        monkeypatch.setenv("FMP_API_KEY", "fmp-key")
        client, transport = scripted_http("fmp", script)
        return FMPEstimatesProvider(http=client), transport

    def test_calendar_parses_sessions_and_flags(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider, transport = self.make(monkeypatch, response("fmp_earnings_calendar.json"))
        frame = provider.earnings_calendar("2020-08-24", "2099-01-31")
        assert list(frame.columns) == list(CALENDAR_COLUMNS)
        assert len(frame) == 4
        by_ticker = frame.set_index("ticker")
        assert by_ticker.loc["AAPL", "session"] == "amc"
        assert by_ticker.loc["MSFT", "session"] == "bmo"
        assert by_ticker.loc["BRK.B", "session"] == "unknown"  # "--" jamás se adivina
        assert bool(by_ticker.loc["ZZZT", "is_estimated_date"])
        assert not bool(by_ticker.loc["AAPL", "is_estimated_date"])
        assert by_ticker.loc["AAPL", "surprise"] == pytest.approx(0.645 - 0.5132)
        assert frame["announced_time_is_nominal"].all()
        # La clave viaja como query param `apikey` y la ventana como from/to.
        params = transport.calls[0].params
        assert params["apikey"] == "fmp-key"
        assert params["from"] == "2020-08-24"
        # AMC 2020-08-27 -> negociable el 28: la cadena completa hasta tradable_date.
        aapl = frame[frame["ticker"] == "AAPL"].iloc[0]
        assert tradable_date(aapl["announced_at"], aapl["session"], CAL) == date(2020, 8, 28)

    def test_calendar_ticker_filter_normalizes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider, _ = self.make(monkeypatch, response("fmp_earnings_calendar.json"))
        # El fixture trae "BRK-B"; el filtro se hace tras normalizar a BRK.B.
        frame = provider.earnings_calendar("2020-08-24", "2099-01-31", tickers="BRK-B")
        assert list(frame["ticker"]) == ["BRK.B"]

    def test_missing_credentials(self, clean_env: None) -> None:
        provider = FMPEstimatesProvider()
        with pytest.raises(ProviderUnavailable) as excinfo:
            provider.earnings_calendar("2020-08-24", "2020-08-31")
        assert "FMP_API_KEY" in excinfo.value.missing_env

    def test_empty_calendar_is_insufficient(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider, _ = self.make(monkeypatch, json_response([]))
        with pytest.raises(InsufficientHistory):
            provider.earnings_calendar("2020-08-24", "2020-08-31")

    def test_error_payload_is_data_quality(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider, _ = self.make(
            monkeypatch, json_response({"Error Message": "Invalid API key"})
        )
        with pytest.raises(DataQualityError, match="Invalid API key"):
            provider.earnings_calendar("2020-08-24", "2020-08-31")

    def test_historical_earnings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider, _ = self.make(monkeypatch, response("fmp_earnings_aapl.json"))
        frame = provider.historical_earnings("AAPL")
        assert len(frame) == 3
        assert (frame["session"] == "unknown").all()
        assert (frame["ticker"] == "AAPL").all()
        assert frame["fiscal_quarter"].tolist() == ["2019Q4", "2020Q1", "2020Q2"]

    def test_consensus_pit_flags(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider, _ = self.make(monkeypatch, response("fmp_analyst_estimates_aapl.json"))
        monkeypatch.setattr(provider, "_today", lambda: date(2020, 9, 1))
        frame = provider.consensus("AAPL")
        assert list(frame.columns) == list(CONSENSUS_COLUMNS)
        assert len(frame) == 2
        by_period = frame.set_index(frame["period_end"].dt.date)
        # Periodo futuro capturado hoy = vintage legítimo; periodo pasado, no.
        assert bool(by_period.loc[date(2099, 12, 27), "is_point_in_time"])
        assert not bool(by_period.loc[date(2020, 6, 27), "is_point_in_time"])
        assert (frame["as_of"] == pd.Timestamp("2020-09-01")).all()
        with pytest.raises(LookAheadError):
            require_point_in_time(frame)


# ===========================================================================
# 3. estimates: Finnhub / EODHD / Nasdaq
# ===========================================================================


class TestFinnhubEstimates:
    def make(self, monkeypatch: pytest.MonkeyPatch, script: Any) -> tuple[
        FinnhubEstimatesProvider, ScriptedTransport
    ]:
        monkeypatch.setenv("FINNHUB_API_KEY", "fh-key")
        client, transport = scripted_http("finnhub", script)
        return FinnhubEstimatesProvider(http=client), transport

    def test_calendar_hour_field(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider, transport = self.make(
            monkeypatch, response("finnhub_earnings_calendar.json")
        )
        frame = provider.earnings_calendar("2020-08-24", "2020-09-02")
        assert len(frame) == 3
        by_ticker = frame.set_index("ticker")
        assert by_ticker.loc["AAPL", "session"] == "amc"
        assert by_ticker.loc["MSFT", "session"] == "bmo"
        assert by_ticker.loc["ZM", "session"] == "unknown"  # hour == ""
        # period_end aproximado por trimestre natural (year=2020, quarter=2).
        assert by_ticker.loc["AAPL", "period_end"] == pd.Timestamp("2020-06-30")
        assert by_ticker.loc["AAPL", "fiscal_quarter"] == "2020Q2"
        # La clave va por cabecera, no por query param (no acaba en logs).
        assert transport.calls[0].headers["X-Finnhub-Token"] == "fh-key"
        assert "token" not in (transport.calls[0].params or {})

    def test_consensus_merges_revenue(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider, _ = self.make(
            monkeypatch,
            [response("finnhub_eps_estimate.json"), response("finnhub_revenue_estimate.json")],
        )
        monkeypatch.setattr(provider, "_today", lambda: date(2020, 9, 1))
        frame = provider.consensus("AAPL")
        assert len(frame) == 2
        past = frame[frame["period_end"] == pd.Timestamp("2020-06-30")].iloc[0]
        future = frame[frame["period_end"] == pd.Timestamp("2099-09-30")].iloc[0]
        assert past["eps_mean"] == pytest.approx(0.5132)
        assert past["revenue_mean"] == pytest.approx(52251000000)
        assert past["n_analysts"] == 28
        assert not bool(past["is_point_in_time"])
        assert bool(future["is_point_in_time"])

    def test_bad_payload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider, _ = self.make(monkeypatch, json_response({"foo": "bar"}))
        with pytest.raises(DataQualityError, match="earningsCalendar"):
            provider.earnings_calendar("2020-08-24", "2020-09-02")


class TestEODHDEstimates:
    def test_calendar(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EODHD_API_KEY", "eod-key")
        client, transport = scripted_http("eodhd", response("eodhd_earnings_calendar.json"))
        provider = EODHDEstimatesProvider(http=client)
        frame = provider.earnings_calendar("2020-08-24", "2020-09-02")
        assert len(frame) == 3
        by_ticker = frame.set_index("ticker")
        assert by_ticker.loc["AAPL", "session"] == "amc"     # AfterMarket
        assert by_ticker.loc["MSFT", "session"] == "bmo"     # BeforeMarket
        assert by_ticker.loc["ZM", "session"] == "unknown"   # null
        assert by_ticker.loc["AAPL", "period_end"] == pd.Timestamp("2020-06-27")
        assert transport.calls[0].params["api_token"] == "eod-key"

    def test_consensus_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EODHD_API_KEY", "eod-key")
        provider = EODHDEstimatesProvider()
        with pytest.raises(ProviderUnavailable, match="vintages"):
            provider.consensus("AAPL")


class TestNasdaqEstimates:
    def test_money_parser(self) -> None:
        money = NasdaqEstimatesProvider._money
        assert money("$2.07") == pytest.approx(2.07)
        assert money("($0.15)") == pytest.approx(-0.15)
        assert money("$1,978,350,000,000") == pytest.approx(1.97835e12)
        assert np.isnan(money("N/A"))
        assert np.isnan(money(""))
        assert np.isnan(money(None))

    def test_calendar_day(self) -> None:
        client, transport = scripted_http("nasdaq", response("nasdaq_calendar_day.json"))
        provider = NasdaqEstimatesProvider(http=client)
        frame = provider.earnings_calendar("2020-07-30", "2020-07-30")
        assert len(frame) == 3
        assert len(transport.calls) == 1  # una petición por día natural
        assert transport.calls[0].params["date"] == "2020-07-30"
        assert "Mozilla" in transport.calls[0].headers["User-Agent"]
        by_ticker = frame.set_index("ticker")
        assert by_ticker.loc["AAPL", "session"] == "amc"
        assert by_ticker.loc["TWTR", "session"] == "bmo"
        assert by_ticker.loc["ZZZQ", "session"] == "unknown"
        assert by_ticker.loc["AAPL", "period_end"] == pd.Timestamp("2020-06-30")
        assert by_ticker.loc["TWTR", "eps_estimate"] == pytest.approx(-0.02)
        assert by_ticker.loc["AAPL", "eps_actual"] == pytest.approx(0.65)

    def test_window_guard(self) -> None:
        provider = NasdaqEstimatesProvider()
        with pytest.raises(ConfigError, match="MAX_WINDOW_DAYS"):
            provider.earnings_calendar("2020-01-01", "2020-06-01")

    def test_consensus_unavailable(self) -> None:
        provider = NasdaqEstimatesProvider()
        with pytest.raises(ProviderUnavailable):
            provider.consensus("AAPL")


# ===========================================================================
# 4. estimates: dataset externo real
# ===========================================================================


class TestExternalConsensus:
    def test_available_and_dedupe(self, external: ExternalConsensusProvider) -> None:
        assert external.available()
        table = external.table
        with_period = table[table["fiscal_period_end"].notna()]
        assert not with_period.duplicated(["ticker", "fiscal_period_end"]).any()
        # Las filas sin period_end sobreviven solo si no duplican un evento.
        assert (table["fiscal_period_end"].isna().sum()) > 0
        assert len(table) < 56971  # la deduplicación ha hecho trabajo real

    def test_calendar_real_events(self, external: ExternalConsensusProvider) -> None:
        frame = external.earnings_calendar(
            "2015-01-01", "2020-12-31", tickers=["AAPL", "MSFT"]
        )
        assert list(frame.columns) == list(CALENDAR_COLUMNS)
        assert len(frame) >= 40  # ~4 trimestres x 6 años x 2 tickers
        assert frame["source"].str.startswith("external:").all()
        assert not frame["is_estimated_date"].any()
        assert frame["announced_time_is_nominal"].all()
        assert set(frame["session"]).issubset({"bmo", "amc", "dmh", "unknown"})
        # El calendario debe encadenar con tradable_date sin excepciones.
        sample = frame.head(8)
        for row in sample.itertuples(index=False):
            day = tradable_date(row.announced_at, row.session, CAL)
            assert day >= pd.Timestamp(row.announced_at).date()

    def test_consensus_is_marked_non_pit(
        self, external: ExternalConsensusProvider
    ) -> None:
        cons = external.consensus(["AAPL"])
        assert list(cons.columns) == list(CONSENSUS_COLUMNS)
        assert len(cons) > 50
        assert not cons["is_point_in_time"].any()
        assert (cons["as_of"] == cons["available_at"]).all()
        with pytest.raises(LookAheadError):
            require_point_in_time(cons)

    def test_warnings_documented(self) -> None:
        doc = " ".join((ExternalConsensusProvider.__doc__ or "").split()).lower()
        assert "supervivencia" in doc
        assert "momentum de revisiones" in doc
        assert "is_point_in_time" in doc

    def test_missing_file(self, tmp_path: Path) -> None:
        provider = ExternalConsensusProvider(tmp_path / "no_existe.parquet")
        assert not provider.available()
        with pytest.raises(ProviderUnavailable, match="no existe"):
            provider.earnings_calendar("2020-01-01", "2020-12-31")

    def test_unknown_ticker(self, external: ExternalConsensusProvider) -> None:
        with pytest.raises(InsufficientHistory):
            external.consensus(["ZZZZZZZ"])


# ===========================================================================
# 5. estimates: sintético y registro
# ===========================================================================


class TestSyntheticEstimates:
    def test_calendar_matches_market(self, market: SyntheticMarket) -> None:
        provider = SyntheticEstimatesProvider(market)
        frame = provider.earnings_calendar("2021-01-04", "2021-12-31")
        events = market.events()
        expected = events[
            (events["announced_at"] >= pd.Timestamp("2021-01-04"))
            & (events["announced_at"] < pd.Timestamp("2022-01-01"))
        ]
        assert len(frame) == len(expected)
        assert not frame["announced_time_is_nominal"].any()  # timestamps reales

    def test_consensus_is_true_vintage(self, market: SyntheticMarket) -> None:
        provider = SyntheticEstimatesProvider(market)
        cons = provider.consensus(list(market.tickers)[:3])
        assert cons["is_point_in_time"].all()
        assert require_point_in_time(cons) is cons
        # Todo as_of es estrictamente anterior al anuncio de su periodo.
        cal = provider.earnings_calendar("2021-01-04", "2021-12-31")
        merged = cons.merge(cal[["ticker", "period_end", "announced_at"]],
                            on=["ticker", "period_end"], how="inner")
        assert len(merged) > 0
        assert (merged["as_of"] < merged["announced_at"]).all()

    def test_unknown_ticker(self, market: SyntheticMarket) -> None:
        provider = SyntheticEstimatesProvider(market)
        with pytest.raises(InsufficientHistory):
            provider.earnings_calendar("2021-01-04", "2021-12-31", tickers="NOPE")


class TestEstimatesRegistry:
    def test_register_and_resolve(
        self, clean_env: None, market: SyntheticMarket
    ) -> None:
        reg = ProviderRegistry()
        register_estimates_providers(reg, market=market)
        assert set(reg.kinds()) == {CALENDAR_KIND, ESTIMATES_KIND}
        # Sin credenciales, el mejor disponible es el dataset externo (real).
        provider = reg.resolve(CALENDAR_KIND)
        assert provider.name == "external_consensus"
        statuses = {s.provider: s for s in reg.statuses(CALENDAR_KIND)}
        assert "FINNHUB_API_KEY" in statuses["finnhub"].missing_env
        assert "FMP_API_KEY" in statuses["fmp"].missing_env

    def test_facades(self, clean_env: None, market: SyntheticMarket) -> None:
        reg = ProviderRegistry()
        register_estimates_providers(reg, market=market, include=["synthetic"])
        cal = get_earnings_calendar("2021-01-04", "2021-06-30", registry=reg)
        assert len(cal) > 0
        cons = get_consensus(
            list(market.tickers)[:2], point_in_time_only=True, registry=reg
        )
        assert cons["is_point_in_time"].all()

    def test_include_unknown(self) -> None:
        with pytest.raises(ConfigError):
            register_estimates_providers(ProviderRegistry(), include=["nope"])


# ===========================================================================
# 6. options: matemática BSM
# ===========================================================================


class TestBlackScholes:
    def test_put_call_parity(self) -> None:
        S, r, q = 100.0, 0.03, 0.012
        K = np.array([70.0, 90, 100, 110, 140])
        tau = np.array([0.05, 0.25, 0.5, 1.0, 2.0])
        sigma = np.array([0.2, 0.35, 0.5, 0.25, 0.6])
        call = bs_price(S, K, tau, sigma, r, q, True)
        put = bs_price(S, K, tau, sigma, r, q, False)
        parity = S * np.exp(-q * tau) - K * np.exp(-r * tau)
        np.testing.assert_allclose(call - put, parity, atol=1e-12)

    def test_matches_synthetic_reference(self) -> None:
        """La referencia del generador (`synthetic._bs_greeks`) y esta deben coincidir."""
        from earnings_alpha.data.synthetic import _bs_greeks as syn_bs

        rng = np.random.default_rng(7)
        n = 64
        spot = rng.uniform(20, 400, n)
        strike = spot * rng.uniform(0.7, 1.3, n)
        tau = rng.uniform(0.02, 2.0, n)
        iv = rng.uniform(0.1, 1.2, n)
        div = rng.uniform(0.0, 0.03, n)
        is_call = rng.random(n) > 0.5
        rate = 0.028
        ref_price, ref_delta, ref_gamma, ref_vega = syn_bs(
            spot, strike, tau, iv, rate, div, is_call
        )
        ours = bs_greeks(spot, strike, tau, iv, rate, div, is_call)
        np.testing.assert_allclose(ours.price, ref_price, atol=1e-10)
        np.testing.assert_allclose(ours.delta, ref_delta, atol=1e-10)
        np.testing.assert_allclose(ours.gamma, ref_gamma, atol=1e-10)
        np.testing.assert_allclose(ours.vega, ref_vega, atol=1e-10)

    def test_greeks_against_finite_differences(self) -> None:
        S, K, tau, sigma, r, q = 100.0, 105.0, 0.5, 0.3, 0.03, 0.01
        h = 1e-5
        for is_call in (True, False):
            g = bs_greeks(S, K, tau, sigma, r, q, is_call)
            d_s = (
                bs_price(S + h, K, tau, sigma, r, q, is_call)
                - bs_price(S - h, K, tau, sigma, r, q, is_call)
            ) / (2 * h)
            assert float(g.delta) == pytest.approx(float(d_s), abs=1e-6)
            # Para la segunda derivada un h tan fino cancela dígitos: h más ancho.
            hg = 1e-3
            d2_s = (
                bs_price(S + hg, K, tau, sigma, r, q, is_call)
                - 2 * bs_price(S, K, tau, sigma, r, q, is_call)
                + bs_price(S - hg, K, tau, sigma, r, q, is_call)
            ) / hg**2
            assert float(g.gamma) == pytest.approx(float(d2_s), abs=1e-6)
            d_sigma = (
                bs_price(S, K, tau, sigma + h, r, q, is_call)
                - bs_price(S, K, tau, sigma - h, r, q, is_call)
            ) / (2 * h)
            assert float(g.vega) == pytest.approx(float(d_sigma) / 100.0, abs=1e-8)
            d_tau = (
                bs_price(S, K, tau + h, sigma, r, q, is_call)
                - bs_price(S, K, tau - h, sigma, r, q, is_call)
            ) / (2 * h)
            assert float(g.theta) == pytest.approx(-float(d_tau) / 365.0, abs=1e-8)
            d_r = (
                bs_price(S, K, tau, sigma, r + h, q, is_call)
                - bs_price(S, K, tau, sigma, r - h, q, is_call)
            ) / (2 * h)
            assert float(g.rho) == pytest.approx(float(d_r) / 100.0, abs=1e-8)

    def test_expired_option_is_intrinsic(self) -> None:
        assert float(bs_price(100.0, 90.0, 0.0, 0.3, 0.03, 0.0, True)) == pytest.approx(10.0)
        assert float(bs_price(100.0, 120.0, -0.1, 0.3, 0.03, 0.0, False)) == pytest.approx(20.0)


class TestImpliedVol:
    def test_roundtrip(self) -> None:
        K = np.array([80.0, 95, 100, 105, 120, 150])
        tau = np.array([0.1, 0.25, 0.5, 1.0, 0.6, 2.0])
        sigma = np.array([0.6, 0.3, 0.2, 0.45, 0.35, 0.8])
        for is_call in (True, False):
            price = bs_price(100.0, K, tau, sigma, 0.03, 0.01, is_call)
            iv = implied_vol(price, 100.0, K, tau, 0.03, 0.01, is_call)
            np.testing.assert_allclose(iv, sigma, atol=1e-7)

    def test_bisection_fallback_engages(self) -> None:
        """OTM extremo: la vega inicial es ~0 y Newton necesita el respaldo."""
        price = float(bs_price(100.0, 300.0, 0.02, 3.0, 0.0, 0.0, True))
        iv, diag = implied_vol(
            price, 100.0, 300.0, 0.02, 0.0, 0.0, True, return_diagnostics=True
        )
        assert diag["bisection_steps"] > 0
        assert float(iv) == pytest.approx(3.0, abs=1e-6)

    def test_no_arbitrage_bounds_yield_nan(self) -> None:
        # Por encima de la cota superior de la call (S e^{-q tau}).
        assert np.isnan(float(implied_vol(101.0, 100.0, 100.0, 0.5, 0.0, 0.0, True)))
        # Por debajo del intrínseco descontado.
        assert np.isnan(float(implied_vol(1.0, 100.0, 80.0, 0.5, 0.0, 0.0, True)))
        # Vencida.
        assert np.isnan(float(implied_vol(5.0, 100.0, 100.0, 0.0, 0.0, 0.0, True)))
        # Un NaN de entrada no revienta ni contamina al resto.
        out = implied_vol(
            np.array([np.nan, 8.0]), 100.0, np.array([100.0, 95.0]),
            np.array([0.5, 0.5]), 0.0, 0.0, True,
        )
        assert np.isnan(out[0]) and np.isfinite(out[1])

    def test_scalar_input_scalar_output(self) -> None:
        price = float(bs_price(100.0, 100.0, 0.5, 0.3, 0.03, 0.0, True))
        iv = implied_vol(price, 100.0, 100.0, 0.5, 0.03, 0.0, True)
        assert iv.shape == ()
        assert iv.item() == pytest.approx(0.3, abs=1e-8)


class TestImpliedForward:
    def test_recovers_forward_and_df(self) -> None:
        strikes = np.array([90.0, 95, 100, 105, 110])
        F, DF, tau = 101.3, 0.985, 0.5
        diffs = DF * (F - strikes)
        est = implied_forward(strikes, diffs, np.zeros_like(strikes), tau,
                              rate=0.03, spot=100.0)
        assert est["forward"] == pytest.approx(F, abs=1e-9)
        assert est["discount_factor"] == pytest.approx(DF, abs=1e-9)
        # implied_borrow = r - ln(F/S)/tau
        assert est["implied_borrow"] == pytest.approx(
            0.03 - np.log(F / 100.0) / tau, abs=1e-9
        )

    def test_insufficient_pairs(self) -> None:
        with pytest.raises(DataQualityError, match=">=3"):
            implied_forward([100.0, 105.0], [1.0, 0.5], [0.0, 0.0], 0.5)

    def test_corrupt_slope(self) -> None:
        strikes = np.array([90.0, 100, 110])
        diffs = np.array([1.0, 2.0, 3.0])  # pendiente positiva: imposible
        with pytest.raises(DataQualityError, match="descuento"):
            implied_forward(strikes, diffs, np.zeros(3), 0.5)

    def test_table_from_synthetic_chain(self, market: SyntheticMarket) -> None:
        provider = SyntheticOptionsProvider(market)
        ticker = market.tickers[0]
        chain = provider.chain(ticker, "2021-06-15")
        table = implied_forward_table(chain, rate=market.config.risk_free)
        assert len(table) > 0
        spot = float(chain["spot"].iloc[0])
        q = float(market._meta.loc[ticker, "dividend_yield"])
        expected = spot * np.exp((market.config.risk_free - q) * table["tau_years"])
        np.testing.assert_allclose(table["forward"], expected, rtol=2e-3)


# ===========================================================================
# 7. options: esquema canónico
# ===========================================================================


def _mini_chain(**overrides: Any) -> pd.DataFrame:
    base = {
        "date": ["2020-09-01", "2020-09-01"],
        "ticker": ["AAPL", "AAPL"],
        "expiry": ["2020-09-18", "2020-09-18"],
        "strike": [110.0, 110.0],
        "right": ["call", "put"],
        "bid": [8.1, 1.6],
        "ask": [8.4, 1.8],
        "last": [8.2, 1.7],
        "volume": [100.0, 50.0],
        "open_interest": [1000.0, 600.0],
        "iv": [0.44, 0.45],
        "delta": [0.71, -0.29],
    }
    base.update(overrides)
    return pd.DataFrame(base)


class TestChainSchema:
    def test_normalize_rights_and_order(self) -> None:
        frame = _mini_chain(extra=[1, 2])
        out = normalize_chain(frame)
        assert list(out.columns[: len(CHAIN_COLUMNS)]) == list(CHAIN_COLUMNS)
        assert set(out["right"]) == {"C", "P"}
        assert "extra" in out.columns
        # C antes que P dentro del mismo strike (ordenación estable documentada).
        assert out["right"].tolist() == ["C", "P"]

    def test_normalize_errors(self) -> None:
        with pytest.raises(InsufficientHistory):
            normalize_chain(pd.DataFrame())
        with pytest.raises(DataQualityError, match="faltan columnas"):
            normalize_chain(_mini_chain().drop(columns=["iv"]))
        with pytest.raises(DataQualityError, match="strikes"):
            normalize_chain(_mini_chain(strike=[0.0, 110.0]))
        with pytest.raises(DataQualityError, match="vencimiento"):
            normalize_chain(_mini_chain(expiry=["2020-08-01", "2020-09-18"]))
        with pytest.raises(DataQualityError, match="right"):
            normalize_chain(_mini_chain(right=["call", "X"]))

    def test_chain_availability_pit(self) -> None:
        out = chain_availability(normalize_chain(_mini_chain()), CAL)
        # Cierre del 2020-09-01: 16:15 EDT = 20:15 UTC.
        assert (out["available_at"] == pd.Timestamp("2020-09-01 20:15")).all()
        # OI: pre-apertura de la siguiente sesión, 08:00 EDT = 12:00 UTC del 09-02.
        assert (out["oi_available_at"] == pd.Timestamp("2020-09-02 12:00")).all()
        assert (out["oi_available_at"] > out["available_at"]).all()

    def test_occ_symbol_roundtrip(self) -> None:
        symbol = occ_symbol("AAPL", "2020-09-18", "call", 112.5)
        assert symbol == "AAPL200918C00112500"
        parsed = parse_occ_symbol("O:" + symbol)
        assert parsed == {
            "ticker": "AAPL",
            "expiry": date(2020, 9, 18),
            "right": "C",
            "strike": 112.5,
        }
        with pytest.raises(DataQualityError):
            parse_occ_symbol("garbage")

    def test_filter_chain_counts(self) -> None:
        rows = pd.concat(
            [
                _mini_chain(),                                    # 2 filas buenas
                _mini_chain(bid=[0.0, 0.0]),                      # F1 x2
                _mini_chain(bid=[9.0, 2.0], ask=[8.0, 1.5]),      # F2 x2
                _mini_chain(bid=[1.0, 1.0], ask=[3.0, 3.0]),      # F3 x2
                # mid < 0.05 con horquilla relativa <= 0.5: cae en F4, no en F3.
                _mini_chain(bid=[0.03, 0.03], ask=[0.04, 0.04]),  # F4 x2
                _mini_chain(expiry=["2022-12-16", "2022-12-16"]), # F6 x2
            ],
            ignore_index=True,
        )
        out, counts = filter_chain(rows, max_dropped_fraction=0.99)
        assert len(out) == 2
        assert counts["F1_bid_cero"] == 2
        assert counts["F2_cruce"] == 2
        assert counts["F3_horquilla"] == 2
        assert counts["F4_mid_minimo"] == 2
        assert counts["F6_plazo"] == 2

    def test_filter_chain_excessive_drop_raises(self) -> None:
        rows = pd.concat([_mini_chain(), _mini_chain(bid=[0.0, 0.0])], ignore_index=True)
        with pytest.raises(DataQualityError, match="NaN"):
            filter_chain(rows, max_dropped_fraction=0.25)

    def test_filter_chain_keeps_unquoted_rows(self) -> None:
        rows = _mini_chain(bid=[np.nan, np.nan], ask=[np.nan, np.nan])
        out, counts = filter_chain(rows)
        assert len(out) == 2
        assert counts["sin_cotizacion"] == 2


# ===========================================================================
# 8. options: proveedores
# ===========================================================================


class TestSyntheticOptions:
    def test_chain_schema_and_pit(self, market: SyntheticMarket) -> None:
        provider = SyntheticOptionsProvider(market)
        ticker = market.tickers[0]
        chain = provider.chain(ticker, "2021-06-15")
        for col in CHAIN_COLUMNS:
            assert col in chain.columns
        assert (chain["source"] == "synthetic").all()
        assert (chain["last"] == chain["mid"]).all()
        # OI conocible en la pre-apertura de la sesión siguiente (informe §7.3).
        assert (chain["oi_available_at"].dt.date == date(2021, 6, 16)).all()

    def test_own_iv_recovers_generator_iv(self, market: SyntheticMarket) -> None:
        """El solver propio debe invertir los precios del generador a su IV."""
        provider = SyntheticOptionsProvider(market)
        ticker = market.tickers[0]
        chain = provider.chain(ticker, "2021-06-15")
        q = float(market._meta.loc[ticker, "dividend_yield"])
        r = market.config.risk_free
        # Contratos con vega material: fuera de ahí la IV no es identificable
        # desde el precio (plana en sigma) y ningún solver puede recuperarla.
        sub = chain[(chain["mid"] > 0.10) & (chain["vega"] > 0.03)]
        assert len(sub) > 20
        tau = (sub["expiry"] - sub["date"]).dt.days.clip(lower=1) / 365.0
        iv = implied_vol(
            sub["mid"].to_numpy(),
            sub["spot"].to_numpy(),
            sub["strike"].to_numpy(),
            tau.to_numpy(),
            r,
            q,
            (sub["right"] == "C").to_numpy(),
        )
        np.testing.assert_allclose(iv, sub["iv"].to_numpy(), atol=5e-4)

    def test_unknown_date_raises(self, market: SyntheticMarket) -> None:
        provider = SyntheticOptionsProvider(market)
        with pytest.raises(DataQualityError):
            provider.chain(market.tickers[0], "2021-06-13")  # domingo


class TestPolygonOptions:
    def test_snapshot_pagination(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("POLYGON_API_KEY", "pg-key")
        client, transport = scripted_http(
            "polygon",
            [
                response("polygon_options_snapshot_p1.json"),
                response("polygon_options_snapshot_p2.json"),
            ],
        )
        provider = PolygonOptionsProvider(http=client)
        monkeypatch.setattr(provider, "_today", lambda: date(2026, 8, 5))
        chain = provider.chain("AAPL")
        assert len(chain) == 4
        assert len(transport.calls) == 2
        assert transport.calls[0].headers["Authorization"] == "Bearer pg-key"
        assert set(chain["right"]) == {"C", "P"}
        row = chain[(chain["strike"] == 115.0) & (chain["right"] == "C")].iloc[0]
        assert row["iv"] == pytest.approx(0.4125)
        assert row["bid"] == pytest.approx(6.45)
        assert row["open_interest"] == 15230
        assert row["spot"] == pytest.approx(116.6)
        assert (chain["date"] == pd.Timestamp("2026-08-05")).all()

    def test_historical_reconstruction_with_own_iv(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Polygon histórico: contratos as_of + barras; IV y delta propias."""
        monkeypatch.setenv("POLYGON_API_KEY", "pg-key")
        sigma_true, rate, spot = 0.40, 0.03, 116.5
        asof = pd.Timestamp("2020-09-01")

        def dispatcher(request: HttpRequest) -> HttpResponse:
            url = request.url
            if "/v2/aggs/ticker/AAPL/" in url:
                return response("polygon_underlying_agg.json")
            if "/v3/reference/options/contracts" in url:
                return response("polygon_contracts_asof.json")
            if "/v2/aggs/ticker/O:" in url:
                occ = url.split("/v2/aggs/ticker/O:")[1].split("/")[0]
                meta = parse_occ_symbol(occ)
                tau = (pd.Timestamp(meta["expiry"]) - asof).days / 365.0
                price = float(
                    bs_price(spot, meta["strike"], tau, sigma_true, rate, 0.0,
                             meta["right"] == "C")
                )
                return json_response(
                    {"status": "OK", "results": [{"c": price, "v": 111.0}]}
                )
            raise AssertionError(f"URL inesperada: {url}")

        client, transport = scripted_http("polygon", dispatcher, repeat_last=True)
        provider = PolygonOptionsProvider(http=client, risk_free_rate=rate)
        chain = provider.chain("AAPL", asof)
        # 6 contratos en el fixture; el strike 300 (banda) y el vencimiento 2022
        # (max_days) quedan fuera: 4 contratos reconstruidos.
        assert len(chain) == 4
        assert chain["bid"].isna().all() and chain["open_interest"].isna().all()
        np.testing.assert_allclose(chain["iv"].to_numpy(), sigma_true, atol=1e-6)
        expected_delta = bs_greeks(
            spot,
            chain["strike"].to_numpy(),
            (chain["expiry"] - asof).dt.days.to_numpy() / 365.0,
            chain["iv"].to_numpy(),
            rate,
            0.0,
            (chain["right"] == "C").to_numpy(),
        ).delta
        np.testing.assert_allclose(chain["delta"].to_numpy(), expected_delta, atol=1e-9)

    def test_historical_max_contracts_guard(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("POLYGON_API_KEY", "pg-key")
        client, _ = scripted_http(
            "polygon",
            [
                response("polygon_underlying_agg.json"),
                response("polygon_contracts_asof.json"),
            ],
        )
        provider = PolygonOptionsProvider(http=client, max_contracts=2)
        with pytest.raises(ConfigError, match="max_contracts"):
            provider.chain("AAPL", "2020-09-01")

    def test_missing_credentials(self, clean_env: None) -> None:
        provider = PolygonOptionsProvider()
        with pytest.raises(ProviderUnavailable) as excinfo:
            provider.chain("AAPL")
        assert "POLYGON_API_KEY" in excinfo.value.missing_env


class TestORATSOptions:
    def test_strikes_parse(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ORATS_TOKEN", "orats-key")
        client, transport = scripted_http("orats", response("orats_strikes.json"))
        provider = ORATSOptionsProvider(http=client)
        chain = provider.chain("AAPL", "2020-09-01")
        # 3 filas válidas x 2 lados; la fila sin expirDate se descarta.
        assert len(chain) == 6
        assert transport.calls[0].params["token"] == "orats-key"
        assert transport.calls[0].params["tradeDate"] == "2020-09-01"
        call_row = chain[(chain["strike"] == 110.0) & (chain["right"] == "C")].iloc[0]
        put_row = chain[(chain["strike"] == 110.0) & (chain["right"] == "P")].iloc[0]
        assert call_row["iv"] == pytest.approx(0.4462)
        assert call_row["delta"] == pytest.approx(0.7124)  # delta de ORATS (call)
        assert call_row["open_interest"] == 84210
        assert put_row["volume"] == 16400

    def test_put_delta_from_own_greeks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ORATS_TOKEN", "orats-key")
        client, _ = scripted_http("orats", response("orats_strikes.json"))
        provider = ORATSOptionsProvider(http=client, risk_free_rate=0.03)
        chain = provider.chain("AAPL", "2020-09-01")
        put_row = chain[(chain["strike"] == 110.0) & (chain["right"] == "P")].iloc[0]
        tau = (pd.Timestamp("2020-09-18") - pd.Timestamp("2020-09-01")).days / 365.0
        expected = float(
            bs_greeks(116.5, 110.0, tau, 0.4517, 0.03, 0.0, False).delta
        )
        assert put_row["delta"] == pytest.approx(expected, abs=1e-12)
        assert -1.0 < put_row["delta"] < 0.0


class TestTradierOptions:
    def test_snapshot_chain(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TRADIER_ACCESS_TOKEN", "td-key")
        client, transport = scripted_http(
            "tradier",
            [
                response("tradier_expirations.json"),
                response("tradier_chain_1.json"),
                response("tradier_chain_2.json"),
            ],
        )
        provider = TradierOptionsProvider(http=client)
        monkeypatch.setattr(provider, "_today", lambda: date(2026, 8, 5))
        chain = provider.chain("AAPL")
        assert len(chain) == 3  # 2 del primer vencimiento + 1 (dict suelto) del 2.º
        assert transport.calls[0].headers["Authorization"] == "Bearer td-key"
        assert transport.calls[0].headers["Accept"] == "application/json"
        row = chain[(chain["right"] == "P")].iloc[0]
        assert row["iv"] == pytest.approx(0.419)
        assert row["open_interest"] == 6410

    def test_refuses_historical(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TRADIER_ACCESS_TOKEN", "td-key")
        provider = TradierOptionsProvider()
        monkeypatch.setattr(provider, "_today", lambda: date(2026, 8, 5))
        with pytest.raises(ProviderUnavailable, match="snapshot actual"):
            provider.chain("AAPL", "2020-09-01")


class TestOptionsRegistry:
    def test_fallback_to_synthetic(
        self, clean_env: None, market: SyntheticMarket
    ) -> None:
        reg = ProviderRegistry()
        register_options_providers(reg, market=market)
        assert reg.resolve(OPTIONS_KIND).name == "synthetic"
        statuses = {s.provider: s for s in reg.statuses(OPTIONS_KIND)}
        assert "ORATS_TOKEN" in statuses["orats"].missing_env
        chain = get_option_chain(market.tickers[0], "2021-06-15", registry=reg)
        assert len(chain) > 0
        assert (chain["source"] == "synthetic").all()


# ===========================================================================
# 9. shortinterest: calendario FINRA y ciclo de liquidación
# ===========================================================================


class TestSettlementCalendar:
    def test_cycle_boundaries(self) -> None:
        assert settlement_cycle_lag("2017-09-06") == 3
        assert settlement_cycle_lag(FIRST_T2_SETTLEMENT) == 2
        assert settlement_cycle_lag("2024-05-28") == 2
        assert settlement_cycle_lag(FIRST_T1_SETTLEMENT) == 1
        assert settlement_cycle_lag("1999-06-15") == 3
        assert settlement_cycle_lag("2026-01-15") == 1

    def test_parity_with_events_flow(self) -> None:
        """`events.flow` consume estos datos: ambas implementaciones deben coincidir."""
        from earnings_alpha.events import flow as ev_flow

        probe = [
            date(2000, 3, 15), date(2017, 9, 6), date(2017, 9, 7),
            date(2020, 8, 14), date(2024, 5, 28), date(2024, 5, 29),
            date(2025, 12, 15),
        ]
        for day in probe:
            assert settlement_cycle_lag(day) == ev_flow.settlement_cycle_lag(day), day
            assert settlement_to_trade_date(day, CAL) == ev_flow.settlement_to_trade_date(
                day, CAL
            ), day

    def test_settlement_to_trade_date(self) -> None:
        # 2020-08-14 (viernes, T+2): la negociación es del miércoles 12.
        assert settlement_to_trade_date("2020-08-14", CAL) == date(2020, 8, 12)
        # 2016 (T+3): 2016-06-15 miércoles -> 2016-06-10 viernes.
        assert settlement_to_trade_date("2016-06-15", CAL) == date(2016, 6, 10)

    def test_finra_settlement_dates(self) -> None:
        dates = finra_settlement_dates("2020-08-01", "2020-10-01", CAL)
        # El 15-ago-2020 es sábado -> se retrocede al viernes 14.
        assert dates == [
            date(2020, 8, 14),
            date(2020, 8, 31),
            date(2020, 9, 15),
            date(2020, 9, 30),
        ]

    def test_publication_date_default_lag(self) -> None:
        # 8 sesiones tras el 2020-08-14: 17,18,19,20,21,24,25,26 -> 2020-08-26.
        assert finra_publication_date("2020-08-14", CAL) == date(2020, 8, 26)
        assert finra_publication_date("2020-08-14", CAL, lag_sessions=1) == date(2020, 8, 17)

    def test_publication_official_schedule(self) -> None:
        official = {date(2020, 8, 14): date(2020, 8, 27)}
        assert (
            finra_publication_date("2020-08-14", CAL, official_schedule=official)
            == date(2020, 8, 27)
        )
        with pytest.raises(DataQualityError, match="corrupta"):
            finra_publication_date(
                "2020-08-14", CAL,
                official_schedule={date(2020, 8, 14): date(2020, 8, 10)},
            )

    def test_ats_publication_date(self) -> None:
        assert ats_publication_date("2020-08-28", tier="T1") == date(2020, 9, 11)
        assert ats_publication_date("2020-08-28", tier="T2") == date(2020, 9, 25)
        with pytest.raises(ConfigError):
            ats_publication_date("2020-08-28", tier="X9")


# ===========================================================================
# 10. shortinterest: guardias PIT
# ===========================================================================


class TestFlowGuards:
    def test_validate_short_interest_lookahead(self, market: SyntheticMarket) -> None:
        provider = SyntheticFlowProvider(market)
        frame = provider.short_interest("2021-01-04", "2021-12-31")
        # El panel legítimo pasa.
        assert validate_short_interest(frame) is frame
        # Indexar por la fecha de referencia = look-ahead puro, y debe explotar.
        corrupt = frame.copy()
        corrupt["available_at"] = corrupt["settlement_date"]
        with pytest.raises(LookAheadError, match="look-ahead"):
            validate_short_interest(corrupt)

    def test_published_asof_excludes_unpublished(
        self, market: SyntheticMarket
    ) -> None:
        provider = SyntheticFlowProvider(market)
        frame = provider.short_interest("2021-01-04", "2021-12-31")
        last_settlement = frame["settlement_date"].max()
        published_at = frame.loc[
            frame["settlement_date"] == last_settlement, "available_at"
        ].iloc[0]
        # asof entre la referencia y la publicación: el snapshot NO es conocible.
        asof_before = last_settlement + pd.Timedelta(days=1)
        assert asof_before < published_at
        visible = published_asof(frame, asof_before)
        assert (visible["settlement_date"] < last_settlement).all()
        # asof tras la publicación: ahora sí.
        visible_after = published_asof(frame, published_at + pd.Timedelta(days=1))
        assert last_settlement in set(visible_after["settlement_date"])

    def test_off_exchange_share_daily_and_double_count_trap(self) -> None:
        sv = pd.DataFrame(
            {
                "date": ["2020-09-02", "2020-09-02"],
                "ticker": ["AAPL", "MSFT"],
                "total_volume": [70119912.0, 21031850.0],
                "available_at": [pd.Timestamp("2020-09-03")] * 2,
            }
        )
        idx = pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2020-09-02"), "AAPL"), (pd.Timestamp("2020-09-02"), "MSFT")],
            names=["date", "ticker"],
        )
        consolidated = pd.Series([140239824.0, 42063700.0], index=idx)
        out = off_exchange_share_daily(sv, consolidated)
        np.testing.assert_allclose(out["off_exchange_share"], 0.5)
        # Convenciones mezcladas -> cuotas > 100 % -> error con diagnóstico.
        with pytest.raises(DataQualityError, match="doble conteo"):
            off_exchange_share_daily(sv, consolidated * 0.4)


# ===========================================================================
# 11. shortinterest: proveedores
# ===========================================================================


class TestSyntheticFlow:
    def test_short_interest_schema_and_lag(self, market: SyntheticMarket) -> None:
        provider = SyntheticFlowProvider(market)
        frame = provider.short_interest("2021-01-04", "2021-12-31")
        for col in SHORT_INTEREST_COLUMNS:
            assert col in frame.columns
        settlement = pd.to_datetime(frame["settlement_date"])
        available = pd.to_datetime(frame["available_at"])
        assert (available > settlement).all()
        # El retardo del generador es el configurado (sesiones de publicación).
        lag = market.config.short_publication_lag
        sample = frame.drop_duplicates("settlement_date").head(5)
        for row in sample.itertuples(index=False):
            expected = CAL.shift(pd.Timestamp(row.settlement_date).date(), lag)
            assert pd.Timestamp(row.available_at).date() == expected

    def test_off_exchange_schema_and_lag(self, market: SyntheticMarket) -> None:
        provider = SyntheticFlowProvider(market)
        frame = provider.off_exchange("2021-01-04", "2021-12-31")
        for col in OFF_EXCHANGE_COLUMNS:
            assert col in frame.columns
        delta = pd.to_datetime(frame["available_at"]) - pd.to_datetime(frame["week_end"])
        assert (delta == pd.Timedelta(days=market.config.offex_publication_lag)).all()

    def test_ticker_filter_and_unknown(self, market: SyntheticMarket) -> None:
        provider = SyntheticFlowProvider(market)
        one = provider.short_interest(
            "2021-01-04", "2021-12-31", tickers=market.tickers[0]
        )
        assert set(one["ticker"]) == {market.tickers[0]}
        with pytest.raises(DataQualityError):
            provider.short_interest("2021-01-04", "2021-12-31", tickers="ZZZZZT")


class TestFinraProviders:
    def make_si(
        self, monkeypatch: pytest.MonkeyPatch, script: Any, **kwargs: Any
    ) -> tuple[FinraShortInterestProvider, ScriptedTransport]:
        monkeypatch.setenv("FINRA_API_CLIENT_ID", "cid")
        monkeypatch.setenv("FINRA_API_CLIENT_SECRET", "sec")
        client, transport = scripted_http("finra", script)
        return FinraShortInterestProvider(http=client, calendar=CAL, **kwargs), transport

    def test_short_interest_parse_and_available_at(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider, transport = self.make_si(
            monkeypatch,
            [response("finra_token.json"), response("finra_short_interest.json")],
        )
        frame = provider.short_interest("2020-08-01", "2020-09-10")
        assert len(frame) == 3
        assert list(frame.columns) == list(SHORT_INTEREST_COLUMNS)
        # OAuth: Basic para el token, Bearer para los datos.
        assert transport.calls[0].headers["Authorization"].startswith("Basic ")
        assert transport.calls[1].headers["Authorization"] == (
            "Bearer test-bearer-token-0001"
        )
        body = json.loads(transport.calls[1].body.decode())
        assert body["compareFilters"][0]["fieldName"] == "settlementDate"
        aapl_aug14 = frame[
            (frame["ticker"] == "AAPL")
            & (frame["settlement_date"] == pd.Timestamp("2020-08-14"))
        ].iloc[0]
        assert aapl_aug14["shares_short"] == 39420000
        # available_at = publicación (+8 sesiones), no la settlement date.
        assert aapl_aug14["available_at"] == pd.Timestamp("2020-08-26")
        assert validate_short_interest(frame) is frame

    def test_token_reused_between_calls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider, transport = self.make_si(
            monkeypatch,
            [
                response("finra_token.json"),
                response("finra_short_interest.json"),
                response("finra_short_interest.json"),
            ],
        )
        provider.short_interest("2020-08-01", "2020-09-10")
        provider.short_interest("2020-08-01", "2020-09-10")
        token_calls = [c for c in transport.calls if "oauth2" in c.url]
        assert len(token_calls) == 1

    def test_pagination(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rows = json.loads(fixture_bytes("finra_short_interest.json"))
        provider, transport = self.make_si(
            monkeypatch,
            [
                response("finra_token.json"),
                json_response(rows[:2]),
                json_response(rows[2:]),
            ],
        )
        provider.PAGE_LIMIT = 2  # atributo de instancia; no toca la clase
        frame = provider.short_interest("2020-08-01", "2020-09-10")
        assert len(frame) == 3
        offsets = [
            json.loads(c.body.decode())["offset"]
            for c in transport.calls
            if "oauth2" not in c.url
        ]
        assert offsets == [0, 2]

    def test_missing_credentials(self, clean_env: None) -> None:
        provider = FinraShortInterestProvider()
        with pytest.raises(ProviderUnavailable) as excinfo:
            provider.short_interest("2020-08-01", "2020-09-10")
        assert "FINRA_API_CLIENT_ID" in excinfo.value.missing_env

    def test_weekly_ats(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FINRA_API_CLIENT_ID", "cid")
        monkeypatch.setenv("FINRA_API_CLIENT_SECRET", "sec")
        client, _ = scripted_http(
            "finra",
            [response("finra_token.json"), response("finra_weekly_summary.json")],
        )
        provider = FinraOffExchangeProvider(http=client, calendar=CAL)
        frame = provider.off_exchange("2020-08-24", "2020-08-28")
        # AAPL (ATS + no-ATS) y MSFT (solo ATS); la fila por firma se ignora.
        assert len(frame) == 2
        aapl = frame[frame["ticker"] == "AAPL"].iloc[0]
        assert aapl["ats_volume"] == pytest.approx(98500000)
        assert aapl["non_ats_volume"] == pytest.approx(141200000)
        assert aapl["off_exchange_volume"] == pytest.approx(239700000)
        assert aapl["week_end"] == pd.Timestamp("2020-08-28")
        # Tier 1: publicación 2 semanas después del fin de la semana.
        assert aapl["available_at"] == pd.Timestamp("2020-09-11")
        msft = frame[frame["ticker"] == "MSFT"].iloc[0]
        assert msft["non_ats_volume"] == 0.0


class TestRegSho:
    def test_parse_and_availability(self) -> None:
        client, transport = scripted_http(
            "finra",
            [
                response("regsho_cnms_20200902.txt", content_type="text/plain"),
                HttpResponse(404, b"Not Found", {}, url="https://cdn.finra.org/x"),
            ],
        )
        provider = RegShoShortVolumeProvider(http=client, calendar=CAL)
        frame = provider.daily_short_volume(
            "2020-09-02", "2020-09-03", tickers=["AAPL", "MSFT"]
        )
        assert list(frame.columns) == list(SHORT_VOLUME_COLUMNS)
        assert len(frame) == 2  # el día 404 se omite con aviso
        assert "CNMSshvol20200902.txt" in transport.calls[0].url
        aapl = frame[frame["ticker"] == "AAPL"].iloc[0]
        assert aapl["short_volume"] == 35907771
        assert aapl["total_volume"] == 70119912
        # T+1: el fichero del 2 de septiembre es conocible el 3.
        assert aapl["available_at"] == pd.Timestamp("2020-09-03")

    def test_malformed_file(self) -> None:
        client, _ = scripted_http(
            "finra", response("regsho_malformed.txt", content_type="text/plain")
        )
        provider = RegShoShortVolumeProvider(http=client, calendar=CAL)
        with pytest.raises(DataQualityError, match="malformadas"):
            provider.daily_short_volume("2020-09-02", "2020-09-02")

    def test_all_missing_is_insufficient(self) -> None:
        client, _ = scripted_http(
            "finra",
            HttpResponse(404, b"Not Found", {}, url="https://cdn.finra.org/x"),
            repeat_last=True,
        )
        provider = RegShoShortVolumeProvider(http=client, calendar=CAL)
        with pytest.raises(InsufficientHistory):
            provider.daily_short_volume("2020-09-02", "2020-09-03")


class TestFlowRegistry:
    def test_fallback_and_facades(
        self, clean_env: None, market: SyntheticMarket
    ) -> None:
        reg = ProviderRegistry()
        register_flow_providers(reg, market=market)
        assert set(reg.kinds()) == {
            SHORT_INTEREST_KIND, OFF_EXCHANGE_KIND, SHORT_VOLUME_KIND
        }
        si = get_short_interest("2021-01-04", "2021-06-30", registry=reg)
        assert (si["source"] == "synthetic").all()
        oe = get_off_exchange("2021-01-04", "2021-06-30", registry=reg)
        assert (oe["source"] == "synthetic").all()
        # regsho queda bajo su kind propio, nunca en la cadena de short_interest.
        names_si = [p.name for p in reg.providers_for(SHORT_INTEREST_KIND)]
        assert "regsho" not in names_si
        assert [p.name for p in reg.providers_for(SHORT_VOLUME_KIND)] == ["regsho"]


# ===========================================================================
# 12. Red real (excluidos por defecto)
# ===========================================================================


@pytest.mark.network
@requires_network
class TestNetwork:
    def test_fmp_calendar_live(self) -> None:
        if not os.environ.get("FMP_API_KEY"):
            pytest.skip("sin FMP_API_KEY")
        provider = FMPEstimatesProvider()
        frame = provider.earnings_calendar("2024-01-02", "2024-01-31")
        assert len(frame) > 0
        assert set(frame["session"]).issubset({"bmo", "amc", "dmh", "unknown"})

    def test_finnhub_calendar_live(self) -> None:
        if not os.environ.get("FINNHUB_API_KEY"):
            pytest.skip("sin FINNHUB_API_KEY")
        provider = FinnhubEstimatesProvider()
        frame = provider.earnings_calendar("2024-01-02", "2024-01-31")
        assert len(frame) > 0

    def test_regsho_live(self) -> None:
        provider = RegShoShortVolumeProvider()
        frame = provider.daily_short_volume("2024-01-03", "2024-01-03")
        assert len(frame) > 100
        assert (frame["available_at"] == pd.Timestamp("2024-01-04")).all()
