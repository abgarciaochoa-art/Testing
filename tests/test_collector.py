"""Tests del recolector diario point-in-time (`earnings_alpha.collector`).

**Ninguno de estos tests abre un socket.** Las cadenas de yfinance se sirven con
un cliente falso en memoria, Tradier y FINRA con `ScriptedTransport`, el tiempo
con `ManualWallClock`/`ManualClock`, y el flujo de extremo a extremo con el
mercado sintético (contrato §0.5).

Cobertura:

1. `snapshot`: esquemas canónicos y sellos (`source`, `captured_at` UTC exacto,
   `available_at`, `collector_version`); conversión de símbolo a Yahoo; fallo
   explícito sin dependencia/token; parseo de Tradier, Reg SHO y short interest;
   la pasada matinal de open interest con su `available_at` real.
2. `storage`: particionado Hive por fecha, append-only, idempotencia de
   re-ejecución, `available_at` obligatorio, verificación de integridad
   (SHA-256, huérfanos), particiones múltiples en un solo append y huecos.
3. `scheduler`: orden de la cola de prioridad, rotación determinista de la
   línea base con cobertura completa, truncado por presupuesto, registro de
   fallos y reintentos con backoff medible.
4. `DailyCollector` + CLI: pasada de cierre y matinal sin red, re-ejecución
   idempotente y `--dry-run` que no escribe nada.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Any, NamedTuple

import pandas as pd
import pytest

from earnings_alpha.collector import main as collector_main
from earnings_alpha.collector.scheduler import (
    REASON_BASELINE,
    REASON_EVENT,
    CollectionLog,
    DailyCollector,
    LogEntry,
    RequestBudget,
    build_priority_queue,
    call_with_retries,
    plan_collection,
)
from earnings_alpha.collector.snapshot import (
    CALENDAR_SNAPSHOT_COLUMNS,
    COLLECTOR_VERSION,
    CONSENSUS_SNAPSHOT_COLUMNS,
    OFF_EXCHANGE_COLUMNS,
    OPEN_INTEREST_COLUMNS,
    OPTION_CHAIN_COLUMNS,
    PASS_CLOSE,
    PASS_MORNING,
    SHORT_INTEREST_COLUMNS,
    FinraShortInterestSource,
    RegShoSource,
    SyntheticOptionsSource,
    TradierOptionsSource,
    YFinanceOptionsSource,
    capture_window_warning,
    snapshot_consensus,
    snapshot_earnings_calendar,
    snapshot_off_exchange,
    snapshot_open_interest,
    snapshot_option_chain,
    snapshot_short_interest,
    to_yahoo_symbol,
)
from earnings_alpha.collector.storage import DATASET_SPECS, SnapshotStore
from earnings_alpha.data.base import (
    HttpResponse,
    ManualClock,
    RateLimited,
    RetryPolicy,
    ScriptedTransport,
)
from earnings_alpha.data.cache import ManualWallClock
from earnings_alpha.data.estimates import SyntheticEstimatesProvider
from earnings_alpha.data.synthetic import SyntheticMarket
from earnings_alpha.errors import (
    ConfigError,
    DataQualityError,
    InsufficientHistory,
    ProviderUnavailable,
)
from earnings_alpha.pit import get_calendar

UTC = dt.UTC

# 2026-08-05 es miércoles (sesión NYSE); 16:45 ET = 20:45 UTC (EDT, UTC-4).
CLOSE_CAPTURE_UTC = dt.datetime(2026, 8, 5, 20, 45, tzinfo=UTC)
# Mañana siguiente, 08:45 ET = 12:45 UTC.
MORNING_CAPTURE_UTC = dt.datetime(2026, 8, 6, 12, 45, tzinfo=UTC)


# ===========================================================================
# Fakes sin red
# ===========================================================================


class _FakeYFChain(NamedTuple):
    calls: pd.DataFrame
    puts: pd.DataFrame


def _yf_leg(strikes: list[float], oi: list[int]) -> pd.DataFrame:
    n = len(strikes)
    return pd.DataFrame(
        {
            "contractSymbol": [f"XX{i}" for i in range(n)],
            "strike": strikes,
            "bid": [s * 0.05 for s in strikes],
            "ask": [s * 0.06 for s in strikes],
            "lastPrice": [s * 0.055 for s in strikes],
            "volume": [10 * (i + 1) for i in range(n)],
            "openInterest": oi,
            "impliedVolatility": [0.25 + 0.01 * i for i in range(n)],
        }
    )


class _FakeYFTicker:
    def __init__(self, symbol: str, expiries: tuple[str, ...]) -> None:
        self.symbol = symbol
        self._expiries = expiries
        self.fast_info = {"last_price": 190.0}

    @property
    def options(self) -> tuple[str, ...]:
        return self._expiries

    def option_chain(self, expiry: str) -> _FakeYFChain:
        assert expiry in self._expiries
        return _FakeYFChain(
            calls=_yf_leg([180.0, 190.0, 200.0], [100, 200, 300]),
            puts=_yf_leg([180.0, 190.0, 200.0], [150, 250, 350]),
        )


class _FakeYFClient:
    """Imita el módulo `yfinance` lo justo para el adaptador."""

    def __init__(self, expiries: tuple[str, ...] = ("2026-08-21", "2026-09-18")) -> None:
        self.expiries = expiries
        self.requested: list[str] = []

    def Ticker(self, symbol: str) -> _FakeYFTicker:  # noqa: N802 - API de yfinance
        self.requested.append(symbol)
        return _FakeYFTicker(symbol, self.expiries)


def _json_response(payload: Any) -> HttpResponse:
    return HttpResponse(
        status_code=200,
        content=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        url="scripted://",
    )


def _tradier_transport() -> ScriptedTransport:
    expirations = _json_response({"expirations": {"date": ["2026-08-21", "2026-09-18"]}})

    def chain_for(request: Any) -> HttpResponse:
        expiry = request.params["expiration"]
        options = []
        for right in ("call", "put"):
            for strike in (180.0, 190.0, 200.0):
                options.append(
                    {
                        "symbol": f"AAPL{expiry}{right[0].upper()}{int(strike)}",
                        "option_type": right,
                        "strike": strike,
                        "bid": strike * 0.05,
                        "ask": strike * 0.06,
                        "last": strike * 0.055,
                        "volume": 42,
                        "open_interest": 1000,
                        "expiration_date": expiry,
                        "greeks": {"mid_iv": 0.31, "delta": 0.5},
                    }
                )
        return _json_response({"options": {"option": options}})

    return ScriptedTransport([expirations, chain_for, chain_for])


REGSHO_SAMPLE = (
    "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
    "20260805|AAPL|1000|10|2500|B,Q,N\n"
    "20260805|BRK B|500|0|900|B,Q\n"
    "20260805|MSFT|2000|5|4100|B,Q,N\n"
    "20260805\n"  # pie de fichero con recuento: debe ignorarse
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(scope="module")
def market() -> SyntheticMarket:
    return SyntheticMarket(seed=11, n_tickers=8, start="2025-06-02", end="2026-09-30")


@pytest.fixture()
def wall_close() -> ManualWallClock:
    return ManualWallClock(CLOSE_CAPTURE_UTC)


@pytest.fixture()
def wall_morning() -> ManualWallClock:
    return ManualWallClock(MORNING_CAPTURE_UTC)


def _make_collector(
    tmp_path,
    market: SyntheticMarket,
    wall: ManualWallClock,
    **kwargs: Any,
) -> DailyCollector:
    store = SnapshotStore(tmp_path / "store")
    options = SyntheticOptionsSource(market, wall=wall)
    estimates = SyntheticEstimatesProvider(market)
    defaults: dict[str, Any] = {
        "store": store,
        "universe_tickers": list(market.tickers),
        "estimates_provider": estimates,
        "options_source": options,
        "wall": wall,
        "clock": ManualClock(),
        "baseline_size": 3,
        "budget_requests": 5000,
    }
    defaults.update(kwargs)
    return DailyCollector(**defaults)


# ===========================================================================
# 1. snapshot: sellos y fuentes
# ===========================================================================


class TestSnapshotOptionChain:
    def test_yfinance_schema_and_stamps(self, wall_close: ManualWallClock) -> None:
        client = _FakeYFClient()
        source = YFinanceOptionsSource(client=client)
        frame = snapshot_option_chain(source, "BRK.B", wall=wall_close)

        assert list(frame.columns) == list(OPTION_CHAIN_COLUMNS)
        # símbolo convertido para Yahoo, pero normalizado en la salida
        assert client.requested == ["BRK-B"]
        assert set(frame["ticker"]) == {"BRK.B"}
        assert set(frame["right"]) == {"C", "P"}
        assert (frame["chain_date"] == pd.Timestamp("2026-08-05")).all()
        assert (frame["source"] == "yfinance").all()
        assert (frame["collector_version"] == COLLECTOR_VERSION).all()
        assert (frame["captured_at"] == pd.Timestamp(CLOSE_CAPTURE_UTC)).all()
        # pasada de cierre: available_at == captured_at
        assert (frame["available_at"] == frame["captured_at"]).all()
        # 2 vencimientos x 2 lados x 3 strikes
        assert len(frame) == 12
        assert source.requests_made == 3  # 1 lista + 2 cadenas

    def test_yfinance_missing_dependency(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom() -> Any:
            raise ImportError("no yfinance")

        monkeypatch.setattr(
            "earnings_alpha.collector.snapshot._import_yfinance", boom
        )
        source = YFinanceOptionsSource(client=None)
        with pytest.raises(ProviderUnavailable, match="yfinance"):
            source.fetch_chain("AAPL")

    def test_yfinance_no_expiries_is_explicit(self, wall_close: ManualWallClock) -> None:
        source = YFinanceOptionsSource(client=_FakeYFClient(expiries=()))
        with pytest.raises(InsufficientHistory, match="vencimientos"):
            snapshot_option_chain(source, "AAPL", wall=wall_close)

    def test_tradier_parses_chain(
        self, monkeypatch: pytest.MonkeyPatch, wall_close: ManualWallClock
    ) -> None:
        monkeypatch.setenv("TRADIER_ACCESS_TOKEN", "sandbox-token")
        transport = _tradier_transport()
        source = TradierOptionsSource(transport=transport)
        frame = snapshot_option_chain(source, "AAPL", wall=wall_close, max_expiries=2)

        assert list(frame.columns) == list(OPTION_CHAIN_COLUMNS)
        assert len(frame) == 12
        assert (frame["iv"] == 0.31).all()
        assert source.requests_made == 3
        first = transport.calls[0]
        assert first.url.startswith(TradierOptionsSource.SANDBOX_BASE)
        assert first.headers is not None
        assert first.headers.get("Authorization") == "Bearer sandbox-token"

    def test_tradier_without_token_fails_before_network(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("TRADIER_ACCESS_TOKEN", raising=False)
        source = TradierOptionsSource()
        assert not source.available()
        with pytest.raises(ProviderUnavailable) as excinfo:
            source.fetch_chain("AAPL")
        assert "TRADIER_ACCESS_TOKEN" in excinfo.value.missing_env

    def test_synthetic_uses_last_closed_session(
        self, market: SyntheticMarket, wall_morning: ManualWallClock
    ) -> None:
        # A las 08:45 ET del 2026-08-06 la última sesión cerrada es el 05.
        source = SyntheticOptionsSource(market, wall=wall_morning)
        raw = source.fetch_chain(market.tickers[0])
        assert len(raw) > 0
        assert source.estimated_requests() == 0

    def test_capture_window_warning(self) -> None:
        ok = dt.datetime(2026, 8, 5, 20, 45, tzinfo=UTC)  # 16:45 ET
        late = dt.datetime(2026, 8, 5, 23, 30, tzinfo=UTC)  # 19:30 ET
        assert capture_window_warning(ok, PASS_CLOSE) is None
        warning = capture_window_warning(late, PASS_CLOSE)
        assert warning is not None and "fuera de la ventana" in warning
        with pytest.raises(ConfigError):
            capture_window_warning(ok, "lunch")

    def test_to_yahoo_symbol(self) -> None:
        assert to_yahoo_symbol("BRK.B") == "BRK-B"
        assert to_yahoo_symbol("brk/b") == "BRK-B"
        assert to_yahoo_symbol("AAPL") == "AAPL"


class TestSnapshotOpenInterest:
    def test_morning_pass_has_real_available_at(
        self, market: SyntheticMarket, wall_morning: ManualWallClock
    ) -> None:
        source = SyntheticOptionsSource(market, wall=wall_morning)
        frame = snapshot_open_interest(
            source, market.tickers[0], calendar=get_calendar(), wall=wall_morning
        )
        assert list(frame.columns) == list(OPEN_INTEREST_COLUMNS)
        # el OI capturado la mañana del 06 describe el cierre del 05...
        assert (frame["oi_date"] == pd.Timestamp("2026-08-05")).all()
        # ...pero solo fue conocible en el instante real de la captura matinal.
        assert (frame["available_at"] == pd.Timestamp(MORNING_CAPTURE_UTC)).all()
        assert (frame["available_at"] > frame["oi_date"].dt.tz_localize(UTC)).all()


class TestSnapshotEstimates:
    def test_consensus_as_of_is_capture_date(
        self, market: SyntheticMarket, wall_close: ManualWallClock
    ) -> None:
        provider = SyntheticEstimatesProvider(market)
        frame = snapshot_consensus(provider, list(market.tickers[:4]), wall=wall_close)
        assert list(frame.columns) == list(CONSENSUS_SNAPSHOT_COLUMNS)
        # as_of = fecha de la captura, no la del proveedor (data_sources.md §6.3)
        assert (frame["as_of"] == pd.Timestamp("2026-08-05")).all()
        assert frame["is_point_in_time"].all()
        assert (frame["captured_at"] == pd.Timestamp(CLOSE_CAPTURE_UTC)).all()
        # una única fila por (ticker, period_end)
        assert not frame.duplicated(subset=["ticker", "period_end"]).any()
        # solo trimestres recientes o futuros
        cutoff = pd.Timestamp("2026-08-05") - pd.Timedelta(days=120)
        assert (frame["period_end"] >= cutoff).all()

    def test_calendar_snapshot_stamps(
        self, market: SyntheticMarket, wall_close: ManualWallClock
    ) -> None:
        provider = SyntheticEstimatesProvider(market)
        frame = snapshot_earnings_calendar(
            provider, dt.date(2026, 8, 5), dt.date(2026, 9, 2), wall=wall_close
        )
        assert list(frame.columns) == list(CALENDAR_SNAPSHOT_COLUMNS)
        assert (frame["capture_date"] == pd.Timestamp("2026-08-05")).all()
        assert (frame["available_at"] == pd.Timestamp(CLOSE_CAPTURE_UTC)).all()
        assert len(frame) > 0


class TestSnapshotFlow:
    def test_regsho_parse_and_stamps(self, wall_morning: ManualWallClock) -> None:
        transport = ScriptedTransport(
            [HttpResponse(status_code=200, content=REGSHO_SAMPLE.encode())]
        )
        source = RegShoSource(transport=transport)
        frame = snapshot_off_exchange(source, dt.date(2026, 8, 5), wall=wall_morning)
        assert list(frame.columns) == list(OFF_EXCHANGE_COLUMNS)
        assert set(frame["ticker"]) == {"AAPL", "BRK.B", "MSFT"}
        assert frame.loc[frame["ticker"] == "AAPL", "total_volume"].item() == 2500
        assert (frame["trade_date"] == pd.Timestamp("2026-08-05")).all()
        assert (frame["source"] == "finra_regsho").all()

    def test_regsho_not_yet_published(self, wall_morning: ManualWallClock) -> None:
        transport = ScriptedTransport(
            [HttpResponse(status_code=404), HttpResponse(status_code=404)]
        )
        source = RegShoSource(transport=transport)
        with pytest.raises(InsufficientHistory, match="Reg SHO"):
            snapshot_off_exchange(source, dt.date(2026, 8, 5), wall=wall_morning)

    def test_regsho_wrong_date_aborts(self) -> None:
        wrong = REGSHO_SAMPLE.replace("20260805", "20260804")
        transport = ScriptedTransport(
            [HttpResponse(status_code=200, content=wrong.encode())]
        )
        source = RegShoSource(transport=transport)
        with pytest.raises(DataQualityError, match="fecha"):
            source.fetch_daily(dt.date(2026, 8, 5))

    def test_short_interest_parse(self, wall_morning: ManualWallClock) -> None:
        records = [
            {
                "symbolCode": "AAPL",
                "settlementDate": "2026-07-31",
                "currentShortPositionQuantity": 120_000_000,
                "averageDailyVolumeQuantity": 60_000_000,
                "daysToCoverQuantity": 2.0,
            },
            {
                "symbolCode": "MSFT",
                "settlementDate": "2026-07-31",
                "currentShortPositionQuantity": 80_000_000,
                "averageDailyVolumeQuantity": 40_000_000,
                "daysToCoverQuantity": 2.0,
            },
        ]
        transport = ScriptedTransport([_json_response(records)])
        source = FinraShortInterestSource(transport=transport)
        frame = snapshot_short_interest(source, wall=wall_morning)
        assert list(frame.columns) == list(SHORT_INTEREST_COLUMNS)
        assert (frame["settlement_date"] == pd.Timestamp("2026-07-31")).all()
        assert frame.loc[frame["ticker"] == "AAPL", "short_interest"].item() == 120_000_000

    def test_short_interest_unrecognized_fields(self) -> None:
        transport = ScriptedTransport([_json_response([{"foo": 1, "bar": 2}])])
        source = FinraShortInterestSource(transport=transport)
        with pytest.raises(DataQualityError, match="alias"):
            source.fetch_latest()


# ===========================================================================
# 2. storage: append-only, idempotencia e integridad
# ===========================================================================


def _chain_frame(
    tickers: list[str],
    *,
    chain_date: str = "2026-08-05",
    captured_at: dt.datetime = CLOSE_CAPTURE_UTC,
) -> pd.DataFrame:
    rows = []
    for t in tickers:
        for strike in (180.0, 190.0):
            rows.append(
                {
                    "ticker": t,
                    "chain_date": pd.Timestamp(chain_date),
                    "expiry": pd.Timestamp("2026-08-21"),
                    "right": "C",
                    "strike": strike,
                    "bid": 1.0,
                    "ask": 1.2,
                    "last": 1.1,
                    "volume": 10.0,
                    "open_interest": 100.0,
                    "iv": 0.3,
                    "spot": 190.0,
                    "source": "test",
                    "captured_at": pd.Timestamp(captured_at),
                    "available_at": pd.Timestamp(captured_at),
                    "collector_version": COLLECTOR_VERSION,
                }
            )
    return pd.DataFrame(rows)


class TestSnapshotStore:
    def test_partition_layout(self, tmp_path) -> None:
        store = SnapshotStore(tmp_path)
        results = store.append("option_chain", _chain_frame(["AAPL"]))
        assert len(results) == 1
        result = results[0]
        assert result.rows_appended == 2 and result.rows_duplicated == 0
        partition = tmp_path / "option_chain" / "date=2026-08-05"
        assert partition.is_dir()
        assert (partition / "_MANIFEST.json").exists()
        assert result.path is not None and result.path.parent == partition

    def test_rerun_same_day_is_idempotent(self, tmp_path) -> None:
        store = SnapshotStore(tmp_path)
        frame = _chain_frame(["AAPL", "MSFT"])
        store.append("option_chain", frame)
        # re-ejecución: mismas claves, quizá con precios distintos horas después
        rerun = frame.copy()
        rerun["bid"] = 9.99
        results = store.append("option_chain", rerun)
        assert results[0].rows_appended == 0
        assert results[0].rows_duplicated == len(frame)
        assert results[0].path is None  # no se escribe ni un fichero vacío
        stored = store.read("option_chain")
        assert len(stored) == len(frame)
        # la primera captura del día gana: append-only, sin sobreescritura
        assert (stored["bid"] == 1.0).all()

    def test_append_only_new_rows(self, tmp_path) -> None:
        store = SnapshotStore(tmp_path)
        store.append("option_chain", _chain_frame(["AAPL"]))
        results = store.append("option_chain", _chain_frame(["AAPL", "MSFT"]))
        assert results[0].rows_appended == 2  # solo MSFT
        assert len(store.read("option_chain")) == 4

    def test_available_at_is_mandatory(self, tmp_path) -> None:
        store = SnapshotStore(tmp_path)
        frame = _chain_frame(["AAPL"]).drop(columns=["available_at"])
        with pytest.raises(DataQualityError, match="available_at"):
            store.append("option_chain", frame)
        frame = _chain_frame(["AAPL"])
        frame.loc[0, "available_at"] = pd.NaT
        with pytest.raises(DataQualityError, match="available_at"):
            store.append("option_chain", frame)

    def test_unknown_dataset_rejected(self, tmp_path) -> None:
        store = SnapshotStore(tmp_path)
        with pytest.raises(ConfigError, match="dataset desconocido"):
            store.append("mystery", _chain_frame(["AAPL"]))
        with pytest.raises(DataQualityError, match="vacío"):
            store.append("option_chain", _chain_frame([]))

    def test_read_range_filters_partitions(self, tmp_path) -> None:
        store = SnapshotStore(tmp_path)
        store.append("option_chain", _chain_frame(["AAPL"], chain_date="2026-08-05"))
        store.append("option_chain", _chain_frame(["AAPL"], chain_date="2026-08-07"))
        both = store.read("option_chain")
        assert sorted(both["chain_date"].dt.date.unique()) == [
            dt.date(2026, 8, 5),
            dt.date(2026, 8, 7),
        ]
        only = store.read("option_chain", start=dt.date(2026, 8, 6))
        assert set(only["chain_date"].dt.date) == {dt.date(2026, 8, 7)}
        with pytest.raises(InsufficientHistory):
            store.read("option_chain", start=dt.date(2026, 8, 8))

    def test_integrity_detects_corruption(self, tmp_path) -> None:
        store = SnapshotStore(tmp_path)
        results = store.append("option_chain", _chain_frame(["AAPL"]))
        path = results[0].path
        assert path is not None
        path.write_bytes(path.read_bytes() + b"garbage")
        with pytest.raises(DataQualityError, match="SHA-256"):
            store.read("option_chain")
        report = store.verify()
        assert not report.ok
        assert any(p.kind == "sha256" for p in report.problems)

    def test_integrity_detects_orphan_files(self, tmp_path) -> None:
        store = SnapshotStore(tmp_path)
        results = store.append("option_chain", _chain_frame(["AAPL"]))
        path = results[0].path
        assert path is not None
        orphan = path.with_name("part-9999-deadbeef.parquet")
        orphan.write_bytes(path.read_bytes())
        with pytest.raises(DataQualityError, match="fuera del manifiesto"):
            store.read("option_chain")
        report = store.verify()
        assert any(p.kind == "huérfano" for p in report.problems)

    def test_multi_partition_append_and_cross_day_idempotence(self, tmp_path) -> None:
        """El short interest particiona por settlement: recapturarlo cada mañana
        sin publicación nueva debe ser un no-op aunque cambie el día de captura."""
        store = SnapshotStore(tmp_path)
        base = {
            "short_interest": 1000.0,
            "avg_daily_volume": 500.0,
            "days_to_cover": 2.0,
            "source": "finra_short_interest",
            "collector_version": COLLECTOR_VERSION,
        }
        frame = pd.DataFrame(
            [
                {"ticker": "AAPL", "settlement_date": pd.Timestamp("2026-07-15"), **base},
                {"ticker": "AAPL", "settlement_date": pd.Timestamp("2026-07-31"), **base},
            ]
        )
        frame["captured_at"] = pd.Timestamp(MORNING_CAPTURE_UTC)
        frame["available_at"] = pd.Timestamp(MORNING_CAPTURE_UTC)
        results = store.append("short_interest", frame)
        assert [r.partition_date for r in results] == [
            dt.date(2026, 7, 15),
            dt.date(2026, 7, 31),
        ]
        # a la mañana siguiente se recaptura lo mismo con otro captured_at
        again = frame.copy()
        next_day = MORNING_CAPTURE_UTC + dt.timedelta(days=1)
        again["captured_at"] = pd.Timestamp(next_day)
        again["available_at"] = pd.Timestamp(next_day)
        results2 = store.append("short_interest", again)
        assert all(r.rows_appended == 0 for r in results2)

    def test_missing_sessions_reports_gaps(self, tmp_path) -> None:
        store = SnapshotStore(tmp_path)
        store.append("option_chain", _chain_frame(["AAPL"], chain_date="2026-08-05"))
        store.append("option_chain", _chain_frame(["AAPL"], chain_date="2026-08-07"))
        gaps = store.missing_sessions("option_chain", get_calendar())
        assert gaps == [dt.date(2026, 8, 6)]

    def test_specs_all_partition_and_dedup_within_schema(self) -> None:
        for spec in DATASET_SPECS.values():
            assert "available_at" in spec.required_columns
            assert spec.partition_column in spec.required_columns
            assert set(spec.dedup_keys) <= set(spec.required_columns)


# ===========================================================================
# 3. scheduler: cola, presupuesto, registro, reintentos
# ===========================================================================


def _calendar_frame(entries: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": [t for t, _ in entries],
            "announced_at": [pd.Timestamp(d) for _, d in entries],
        }
    )


class TestPriorityQueue:
    TODAY = dt.date(2026, 8, 5)

    def test_orders_by_event_proximity(self) -> None:
        members = ["AAA", "BBB", "CCC", "DDD", "EEE"]
        calendar = _calendar_frame(
            [("CCC", "2026-08-20 20:30"), ("AAA", "2026-08-07 11:30"), ("EEE", "2026-09-20")]
        )
        queue = build_priority_queue(
            members, calendar, today=self.TODAY, horizon_days=28, baseline_size=2
        )
        events = [p for p in queue if p.reason == REASON_EVENT]
        assert [p.ticker for p in events] == ["AAA", "CCC"]  # EEE fuera del horizonte
        assert events[0].days_to_event == 2
        assert queue[:2] == events  # los eventos van primero
        baseline = [p for p in queue if p.reason == REASON_BASELINE]
        assert len(baseline) == 2
        assert {p.ticker for p in baseline} <= {"BBB", "DDD", "EEE"}

    def test_no_calendar_degrades_to_baseline(self) -> None:
        queue = build_priority_queue(
            ["AAA", "BBB", "CCC"], None, today=self.TODAY, baseline_size=2
        )
        assert all(p.reason == REASON_BASELINE for p in queue)
        assert len(queue) == 2

    def test_deterministic_and_rotating_baseline(self) -> None:
        members = [f"T{i:02d}" for i in range(12)]
        one = build_priority_queue(members, None, today=self.TODAY, baseline_size=5)
        two = build_priority_queue(members, None, today=self.TODAY, baseline_size=5)
        assert [p.ticker for p in one] == [p.ticker for p in two]  # mismo día -> mismo plan
        nxt = build_priority_queue(
            members, None, today=self.TODAY + dt.timedelta(days=1), baseline_size=5
        )
        assert [p.ticker for p in nxt] != [p.ticker for p in one]  # rota entre días
        # la rotación cubre el universo entero en pocos días
        seen: set[str] = set()
        for offset in range(12):
            day = self.TODAY + dt.timedelta(days=offset)
            queue = build_priority_queue(members, None, today=day, baseline_size=5)
            seen.update(p.ticker for p in queue)
        assert seen == set(members)

    def test_budget_truncates_in_priority_order(self) -> None:
        members = [f"T{i:02d}" for i in range(10)]
        calendar = _calendar_frame([("T03", "2026-08-06"), ("T07", "2026-08-10")])
        queue = build_priority_queue(
            members, calendar, today=self.TODAY, baseline_size=8
        )
        budget = RequestBudget(max_requests=26)
        plan = plan_collection(
            queue,
            pass_=PASS_CLOSE,
            today=self.TODAY,
            budget=budget,
            cost_per_ticker=10,
            fixed_cost=5,
        )
        # 5 fijo + 2x10 = 25 <= 26; el tercero ya no cabe
        assert [p.ticker for p in plan.selected] == ["T03", "T07"]
        assert len(plan.deferred) == len(queue) - 2
        assert plan.estimated_requests == 25
        assert budget.remaining == 1
        assert "diferidos por presupuesto: 8" in plan.describe()


class TestCollectionLog:
    def test_record_and_query(self, tmp_path) -> None:
        log = CollectionLog(tmp_path / "log.jsonl")

        def entry(day: str, status: str, ticker: str) -> LogEntry:
            return LogEntry(
                at="2026-08-05T21:00:00+00:00",
                run_id="r1",
                day=day,
                pass_=PASS_CLOSE,
                dataset="option_chain",
                ticker=ticker,
                status=status,
                attempts=1,
                rows=10,
                error="boom" if status == "failed" else "",
            )

        log.record(entry("2026-08-05", "ok", "AAPL"))
        log.record(entry("2026-08-05", "failed", "MSFT"))
        log.record(entry("2026-08-06", "ok", "AAPL"))
        assert len(log.entries()) == 3
        assert len(log.entries(day=dt.date(2026, 8, 5))) == 2
        assert [e.ticker for e in log.failures()] == ["MSFT"]
        assert log.failed_days() == [dt.date(2026, 8, 5)]
        assert "failed=1" in log.summary(dt.date(2026, 8, 5))


class TestRetries:
    def test_backoff_and_retry_after(self) -> None:
        clock = ManualClock()
        calls = {"n": 0}

        def flaky() -> int:
            calls["n"] += 1
            if calls["n"] < 3:
                raise RateLimited("prov", retry_after=5.0)
            return 42

        value, attempts = call_with_retries(
            flaky,
            attempts=3,
            retry=RetryPolicy(max_retries=2, backoff_base_s=1.0, jitter="none"),
            clock=clock,
        )
        assert (value, attempts) == (42, 3)
        # Retry-After (5 s) manda sobre el backoff propio (1 s, 2 s)
        assert clock.sleeps == [5.0, 5.0]

    def test_exhaustion_reraises(self) -> None:
        clock = ManualClock()

        def always_fails() -> None:
            raise RateLimited("prov", retry_after=1.0)

        with pytest.raises(RateLimited):
            call_with_retries(
                always_fails,
                attempts=2,
                retry=RetryPolicy(max_retries=1, backoff_base_s=0.5, jitter="none"),
                clock=clock,
            )
        assert len(clock.sleeps) == 1  # solo se espera entre intentos, no tras el último

    def test_data_quality_error_is_not_retried(self) -> None:
        clock = ManualClock()
        calls = {"n": 0}

        def broken_parser() -> None:
            calls["n"] += 1
            raise DataQualityError("payload malformado")

        with pytest.raises(DataQualityError):
            call_with_retries(broken_parser, attempts=3, clock=clock)
        assert calls["n"] == 1 and clock.sleeps == []


# ===========================================================================
# 4. DailyCollector de extremo a extremo (sin red) y CLI
# ===========================================================================


class TestDailyCollector:
    def test_close_pass_writes_all_datasets(
        self, tmp_path, market: SyntheticMarket, wall_close: ManualWallClock
    ) -> None:
        collector = _make_collector(tmp_path, market, wall_close)
        report = collector.run(pass_=PASS_CLOSE)
        assert report.day == dt.date(2026, 8, 5)
        assert report.success, report.failed
        assert report.rows_by_dataset.get("option_chain", 0) > 0
        assert report.rows_by_dataset.get("consensus", 0) > 0
        assert report.rows_by_dataset.get("earnings_calendar", 0) > 0
        chains = collector.store.read("option_chain")
        assert set(chains["chain_date"].dt.date) == {dt.date(2026, 8, 5)}
        assert set(chains["ticker"]) <= set(market.tickers)
        # el registro contiene una línea ok por ticker capturado
        ok_entries = [e for e in collector.log.entries() if e.status == "ok"]
        assert len(ok_entries) >= len(report.ok)

    def test_close_pass_rerun_is_idempotent(
        self, tmp_path, market: SyntheticMarket, wall_close: ManualWallClock
    ) -> None:
        collector = _make_collector(tmp_path, market, wall_close)
        first = collector.run(pass_=PASS_CLOSE)
        assert sum(first.rows_by_dataset.values()) > 0
        wall_close.advance(dt.timedelta(minutes=7))  # reintento del cron minutos después
        second = collector.run(pass_=PASS_CLOSE)
        assert second.success
        assert sum(second.rows_by_dataset.values()) == 0  # nada duplicado

    def test_morning_pass_after_close(
        self, tmp_path, market: SyntheticMarket, wall_close: ManualWallClock
    ) -> None:
        collector = _make_collector(tmp_path, market, wall_close)
        close_report = collector.run(pass_=PASS_CLOSE)
        assert close_report.success
        close_tickers = set(
            collector.store.read("option_chain", columns=["ticker"], validate=False)["ticker"]
        )
        wall_close.set(MORNING_CAPTURE_UTC)
        report = collector.run(pass_=PASS_MORNING)
        assert report.success, report.failed
        oi = collector.store.read("open_interest")
        assert set(oi["oi_date"].dt.date) == {dt.date(2026, 8, 5)}
        # el OI matinal cubre exactamente lo fotografiado ayer
        assert set(oi["ticker"]) == close_tickers
        # y su available_at es posterior a la captura de cierre (PIT real)
        assert (oi["available_at"] == pd.Timestamp(MORNING_CAPTURE_UTC)).all()

    def test_morning_pass_flow_sources(
        self, tmp_path, market: SyntheticMarket, wall_close: ManualWallClock
    ) -> None:
        regsho_rows = "\n".join(
            f"20260805|{t.replace('.', ' ')}|100|0|300|B,Q" for t in market.tickers[:3]
        )
        regsho = (
            "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
            + regsho_rows
            + "\n"
        )
        si_records = [
            {
                "symbolCode": t,
                "settlementDate": "2026-07-31",
                "currentShortPositionQuantity": 1000,
                "averageDailyVolumeQuantity": 500,
                "daysToCoverQuantity": 2.0,
            }
            for t in market.tickers[:3]
        ]
        collector = _make_collector(
            tmp_path,
            market,
            wall_close,
            off_exchange_source=RegShoSource(
                transport=ScriptedTransport(
                    [HttpResponse(status_code=200, content=regsho.encode())]
                )
            ),
            short_interest_source=FinraShortInterestSource(
                transport=ScriptedTransport([_json_response(si_records)])
            ),
        )
        collector.run(pass_=PASS_CLOSE)
        wall_close.set(MORNING_CAPTURE_UTC)
        report = collector.run(pass_=PASS_MORNING)
        assert report.success, report.failed
        assert report.rows_by_dataset.get("off_exchange", 0) == 3
        assert report.rows_by_dataset.get("short_interest", 0) == 3
        off = collector.store.read("off_exchange")
        assert set(off["trade_date"].dt.date) == {dt.date(2026, 8, 5)}

    def test_non_session_day_is_noop(
        self, tmp_path, market: SyntheticMarket
    ) -> None:
        # 2026-08-08 es sábado
        saturday = ManualWallClock(dt.datetime(2026, 8, 8, 20, 45, tzinfo=UTC))
        collector = _make_collector(tmp_path, market, saturday)
        report = collector.run(pass_=PASS_CLOSE)
        assert report.plan is None
        assert any("no es sesión" in n for n in report.notes)

    def test_dry_run_writes_nothing(
        self, tmp_path, market: SyntheticMarket, wall_close: ManualWallClock
    ) -> None:
        collector = _make_collector(tmp_path, market, wall_close)
        report = collector.run(pass_=PASS_CLOSE, dry_run=True)
        assert report.dry_run and report.plan is not None
        assert len(report.plan.selected) > 0
        assert report.rows_by_dataset == {}
        root = collector.store.root
        assert not any(root.rglob("*.parquet"))
        assert collector.log.entries() == []


class TestCLI:
    def test_dry_run_cli(self, tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
        rc = collector_main(
            [
                "run",
                "--dry-run",
                "--date",
                "2026-08-05",
                "--root",
                str(tmp_path / "store"),
                "--options-source",
                "synthetic",
                "--calendar-provider",
                "synthetic",
                "--budget",
                "100",
                "--baseline",
                "5",
            ]
        )
        out = capsys.readouterr().out
        assert rc == 0
        assert "DRY-RUN" in out
        assert "plan de la pasada 'close'" in out
        assert not any((tmp_path / "store").rglob("*.parquet"))

    def test_status_and_verify_cli(self, tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
        store = SnapshotStore(tmp_path / "store")
        store.append("option_chain", _chain_frame(["AAPL"]))
        assert collector_main(["status", "--root", str(tmp_path / "store")]) == 0
        out = capsys.readouterr().out
        assert "option_chain: 1 particiones" in out
        assert collector_main(["verify", "--root", str(tmp_path / "store")]) == 0
        out = capsys.readouterr().out
        assert "integridad OK" in out
