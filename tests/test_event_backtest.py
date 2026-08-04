"""Tests del backtest dirigido por eventos (`earnings_alpha.backtest.event_engine`).

Cinco familias, todas sin red:

1. **Micro-caso determinista.** Un panel de precios construido a mano permite
   verificar número a número la descomposición gap/intradía, los precios de
   ejecución (apertura vs. cierre), la partición del día del anuncio y la
   bandera `holds_event_gap`.
2. **Point-in-time.** Pre-posicionarse (`entry_offset < 0`) sin declarar la
   conocibilidad del calendario lanza `LookAheadError` (política obligatoria de
   `pit_and_biases.md` §8.3); la entrada en `tau = 0` es en la apertura, nunca al
   cierre del día del anuncio de un AMC; los anuncios DMH se excluyen o retrasan.
3. **Verdad-terreno sintética.** Entrar con el signo de la sorpresa verdadera
   bate con holgura al mismo motor con el score permutado; el quintil alto de SUE
   bate al bajo en el drift posterior; la partición del día del evento recupera
   el `event_gap_share = 0.80` del generador.
4. **Concurrencia y costes.** El límite de posiciones simultáneas se respeta al
   día, los descartes quedan registrados, la utilización se reporta, y los costes
   son monótonos (más costes nunca mejoran el neto; los cortos pagan préstamo).
5. **`run_grid`.** Tabla completa y coherente: todas las combinaciones presentes,
   combinaciones imposibles marcadas (no omitidas), percentiles ordenados,
   contabilidad candidatos = ejecutados + descartados, y columnas por quintil.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from earnings_alpha.backtest.costs import CostModel
from earnings_alpha.backtest.event_engine import (
    SKIP_REASONS,
    EventBacktest,
    run_grid,
    summarize_event_returns,
)
from earnings_alpha.data.synthetic import SyntheticMarket
from earnings_alpha.errors import (
    ConfigError,
    DataQualityError,
    InsufficientHistory,
    LookAheadError,
)
from earnings_alpha.pit import get_calendar

# ------------------------------------------------------------------- fixtures

MKT = {"n_tickers": 32, "start": "2019-01-02", "end": "2022-12-30"}

#: Verdad-terreno del generador usada en los tests de recuperación.
EVENT_GAP_SHARE = 0.80


@pytest.fixture(scope="module")
def mkt() -> SyntheticMarket:
    """Mercado sintético compartido por todo el módulo (una sola generación)."""
    return SyntheticMarket(seed=777, leak_fraction=0.15, **MKT)


@pytest.fixture(scope="module")
def events(mkt: SyntheticMarket) -> pd.DataFrame:
    return mkt.events()


@pytest.fixture(scope="module")
def prices(mkt: SyntheticMarket) -> pd.DataFrame:
    return mkt.prices()


@pytest.fixture(scope="module")
def engine(mkt: SyntheticMarket) -> EventBacktest:
    """Motor sin costes: los tests de señal separan alfa bruta de implementación."""
    return EventBacktest(calendar=mkt.calendar, cost_model=CostModel.zero())


@pytest.fixture(scope="module")
def engine_costs(mkt: SyntheticMarket) -> EventBacktest:
    """Motor con el modelo de costes realista por defecto."""
    return EventBacktest(calendar=mkt.calendar, cost_model=CostModel())


# --------------------------------------------------------- micro-caso a mano

MICRO_CLOSES = [100.0, 101.0, 102.0, 103.0, 104.0, 110.0, 111.0, 112.0, 113.0, 114.0]
MICRO_OPENS = [99.5, 100.5, 101.5, 102.5, 103.5, 108.0, 110.5, 111.5, 112.5, 113.5]


def _micro_panel(tickers: list[str]) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Panel canónico de 10 sesiones (2021-03-01..2021-03-12) con precios a mano.

    El evento se sitúa en la sexta sesión (2021-03-08): el gap del anuncio es
    108/104 - 1 y su tramo intradía 110/108 - 1.
    """
    cal = get_calendar()
    sessions = cal.sessions("2021-03-01", "2021-03-12")
    assert len(sessions) == 10  # sin festivos en ese rango
    index = pd.MultiIndex.from_product([sessions, tickers], names=["date", "ticker"])
    n = len(tickers)
    frame = pd.DataFrame(
        {
            "open": np.repeat(MICRO_OPENS, n),
            "close": np.repeat(MICRO_CLOSES, n),
        },
        index=index,
    )
    return frame, sessions


def _micro_events(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    """Tabla mínima de eventos: (ticker, event_date, session)."""
    return pd.DataFrame(
        {
            "ticker": [r[0] for r in rows],
            "event_date": pd.to_datetime([r[1] for r in rows]),
            "session": [r[2] for r in rows],
        }
    )


def test_micro_entry_after_event_decomposition() -> None:
    """Entrada en la apertura de tau=0, salida al cierre de tau=+1: números exactos."""
    panel, _ = _micro_panel(["AAA"])
    events = _micro_events([("AAA", "2021-03-08", "amc")])
    eng = EventBacktest(cost_model=CostModel.zero())
    res = eng.run(events, panel, entry_offset=0, exit_offset=1, min_events=1)

    assert len(res.trades) == 1
    t = res.trades.iloc[0]
    assert t["entry_at"] == "open" and t["exit_at"] == "close"
    assert t["entry_price"] == pytest.approx(108.0)
    assert t["exit_price"] == pytest.approx(111.0)
    assert t["price_return"] == pytest.approx(111.0 / 108.0 - 1.0)
    # Única noche mantenida: cierre 2021-03-08 (110) -> apertura 2021-03-09 (110.5).
    assert t["gap_return"] == pytest.approx(110.5 / 110.0 - 1.0)
    assert t["intraday_return"] == pytest.approx((110.0 / 108.0) * (111.0 / 110.5) - 1.0)
    # El gap del ANUNCIO (104 -> 108) queda fuera: entrar en tau=0 no lo captura.
    assert not bool(t["holds_event_gap"])
    assert t["event_gap_return"] == pytest.approx(108.0 / 104.0 - 1.0)
    assert t["event_intraday_return"] == pytest.approx(110.0 / 108.0 - 1.0)
    assert t["holding_sessions"] == 1


def test_micro_pre_positioning_captures_event_gap() -> None:
    """Entrada al cierre de tau=-1: la operación sí cruza el gap del anuncio."""
    panel, _ = _micro_panel(["AAA"])
    events = _micro_events([("AAA", "2021-03-08", "amc")])
    eng = EventBacktest(cost_model=CostModel.zero())
    res = eng.run(
        events,
        panel,
        entry_offset=-1,
        exit_offset=0,
        calendar_known_in_advance=True,
        min_events=1,
    )
    t = res.trades.iloc[0]
    assert t["entry_at"] == "close"
    assert t["entry_price"] == pytest.approx(104.0)
    assert t["price_return"] == pytest.approx(110.0 / 104.0 - 1.0)
    assert t["gap_return"] == pytest.approx(108.0 / 104.0 - 1.0)
    assert t["intraday_return"] == pytest.approx(110.0 / 108.0 - 1.0)
    assert bool(t["holds_event_gap"])


def test_micro_identity_multiplicative() -> None:
    """(1+gap)·(1+intradía) == 1+retorno, exacto al épsilon de máquina."""
    panel, _ = _micro_panel(["AAA"])
    events = _micro_events([("AAA", "2021-03-08", "bmo")])
    eng = EventBacktest(cost_model=CostModel.zero())
    for eo, xo, kwargs in [(-3, 2, {"calendar_known_in_advance": True}), (0, 3, {}), (1, 4, {})]:
        res = eng.run(events, panel, entry_offset=eo, exit_offset=xo, min_events=1, **kwargs)
        t = res.trades.iloc[0]
        lhs = (1.0 + t["gap_return"]) * (1.0 + t["intraday_return"])
        assert lhs == pytest.approx(1.0 + t["price_return"], abs=1e-12)


def test_micro_dmh_excluded_and_delayed() -> None:
    """DMH en tau=0: la apertura es pre-anuncio; se excluye o se retrasa al cierre."""
    panel, _ = _micro_panel(["AAA", "BBB"])
    events = _micro_events([("AAA", "2021-03-08", "bmo"), ("BBB", "2021-03-08", "dmh")])

    excl = EventBacktest(cost_model=CostModel.zero(), dmh_policy="exclude")
    res = excl.run(events, panel, entry_offset=0, exit_offset=2, min_events=1)
    assert list(res.trades["ticker"]) == ["AAA"]
    assert res.skipped["reason"].tolist() == ["dmh_excluded"]
    assert set(res.skipped["reason"]) <= set(SKIP_REASONS)

    delay = EventBacktest(cost_model=CostModel.zero(), dmh_policy="delay")
    res2 = delay.run(events, panel, entry_offset=0, exit_offset=2, min_events=1)
    bbb = res2.trades.loc[res2.trades["ticker"] == "BBB"].iloc[0]
    assert bbb["entry_at"] == "close"  # primer precio limpio post-anuncio
    assert bbb["entry_price"] == pytest.approx(110.0)
    aaa = res2.trades.loc[res2.trades["ticker"] == "AAA"].iloc[0]
    assert aaa["entry_at"] == "open"  # el BMO no se ve afectado


def test_micro_same_session_intraday_trade() -> None:
    """entry_offset == exit_offset es legal: apertura -> cierre de la misma sesión."""
    panel, _ = _micro_panel(["AAA"])
    events = _micro_events([("AAA", "2021-03-08", "amc")])
    eng = EventBacktest(cost_model=CostModel.zero())
    res = eng.run(events, panel, entry_offset=0, exit_offset=0, min_events=1)
    t = res.trades.iloc[0]
    assert t["gap_return"] == pytest.approx(0.0)  # ninguna noche mantenida
    assert t["price_return"] == pytest.approx(110.0 / 108.0 - 1.0)
    assert t["intraday_return"] == pytest.approx(t["price_return"])
    assert t["holding_sessions"] == 0


# ----------------------------------------------------------- point-in-time

def test_negative_entry_requires_declaration(
    events: pd.DataFrame, prices: pd.DataFrame, engine: EventBacktest
) -> None:
    """Pre-posicionarse sin vintages de calendario declarados es LookAheadError."""
    with pytest.raises(LookAheadError, match="calendar_known_in_advance"):
        engine.run(events, prices, entry_offset=-5, exit_offset=1)


def test_entry_at_event_is_at_open_of_tradable_session(
    events: pd.DataFrame, prices: pd.DataFrame, engine: EventBacktest
) -> None:
    """En tau=0 se entra en la APERTURA de la sesión negociable, nunca en el cierre
    previo: para un AMC la sesión negociable ya es la siguiente al anuncio."""
    res = engine.run(events, prices, entry_offset=0, exit_offset=5, max_concurrent=None)
    assert (res.trades["entry_at"] == "open").all()
    assert (res.trades["entry_date"] == res.trades["event_date"]).all()
    # Ningún trade de entrada en tau=0 mantiene el gap del anuncio.
    assert not res.trades["holds_event_gap"].any()

    merged = res.trades.merge(
        events[["event_id", "announced_at", "session"]], left_index=True, right_on="event_id"
    )
    amc = merged[merged["session"] == "amc"]
    assert len(amc) > 0
    # La sesión negociable de un AMC es estrictamente posterior al día del anuncio.
    assert (
        pd.to_datetime(amc["entry_date"]).dt.normalize()
        > pd.to_datetime(amc["announced_at"]).dt.normalize()
    ).all()


def test_prices_without_open_fail_explicitly(
    events: pd.DataFrame, prices: pd.DataFrame, engine: EventBacktest
) -> None:
    """Sin columna `open` no hay gap modelable: DataQualityError, no degradación."""
    with pytest.raises(DataQualityError, match="open"):
        engine.run(events, prices.drop(columns=["open"]), entry_offset=0, exit_offset=5)


def test_price_panel_with_session_gap_fails(
    events: pd.DataFrame, prices: pd.DataFrame, engine: EventBacktest
) -> None:
    """Un hueco de sesión en el panel desalinearía los offsets: fallo explícito."""
    dates = prices.index.get_level_values("date").unique().sort_values()
    broken = prices[prices.index.get_level_values("date") != dates[100]]
    with pytest.raises(DataQualityError, match="sesiones del calendario"):
        engine.run(events, broken, entry_offset=0, exit_offset=5)


# ------------------------------------------------- verdad-terreno sintética

def test_true_signal_beats_scrambled(
    mkt: SyntheticMarket, events: pd.DataFrame, prices: pd.DataFrame, engine: EventBacktest
) -> None:
    """Con la sorpresa latente como score, cruzar el anuncio con su signo gana; el
    mismo motor con el score permutado (aleatorio, con semilla) no.

    El calendario sintético es público por adelantado, así que declarar
    `calendar_known_in_advance=True` es legítimo aquí (y solo aquí).
    """
    truth = mkt.ground_truth()
    res = engine.run(
        events,
        prices,
        entry_offset=-5,
        exit_offset=1,
        score=truth["surprise_z"],
        side="signed",
        max_concurrent=None,
        calendar_known_in_advance=True,
        n_boot=200,
    )
    rng = np.random.default_rng(42)
    scrambled = pd.Series(
        rng.permutation(truth["surprise_z"].to_numpy()), index=truth.index
    )
    res_scr = engine.run(
        events,
        prices,
        entry_offset=-5,
        exit_offset=1,
        score=scrambled,
        side="signed",
        max_concurrent=None,
        calendar_known_in_advance=True,
        n_boot=200,
    )
    assert res.summary["n_events"] > 400
    assert res.summary["mean_gross"] > 0.008  # jump ~1.9% por unidad de |z|
    assert abs(res_scr.summary["mean_gross"]) < 0.006
    assert res.summary["mean_gross"] > res_scr.summary["mean_gross"] + 0.008
    # La media reporta su banda y excluye el cero.
    assert res.summary["mean_net_ci_low"] > 0.0
    # Cruzando el anuncio, todas las operaciones mantienen el gap del anuncio.
    assert res.summary["holds_event_gap_frac"] == pytest.approx(1.0)


def test_high_sue_quintile_beats_low_in_drift(
    events: pd.DataFrame, prices: pd.DataFrame, engine: EventBacktest
) -> None:
    """PEAD (Bernard-Thomas): tras el anuncio, el quintil alto de SUE bate al bajo."""
    res = engine.run(
        events,
        prices,
        entry_offset=1,
        exit_offset=40,
        score="sue",
        side="long",
        max_concurrent=None,
        n_boot=200,
    )
    s = res.summary
    assert s["q5_n"] > 50 and s["q1_n"] > 50
    assert s["q5_mean_net"] > s["q1_mean_net"]
    assert s["q_spread_mean"] > 0.0
    assert s["q_spread_ci_low"] > 0.0  # la banda del spread excluye el cero
    assert s["q5_hit_rate"] > s["q1_hit_rate"]


def test_event_day_gap_share_recovered(
    events: pd.DataFrame, prices: pd.DataFrame, engine: EventBacktest
) -> None:
    """La partición del día del anuncio recupera el event_gap_share = 0.80."""
    res = engine.run(events, prices, entry_offset=0, exit_offset=5, max_concurrent=None)
    t = res.trades
    share = t["event_gap_return"].abs() / (
        t["event_gap_return"].abs() + t["event_intraday_return"].abs()
    )
    assert 0.65 < share.median() < 0.92
    # Y el gap del anuncio NO está en el retorno de la operación (entrada en tau=0):
    # el componente nocturno de la tenencia es de noches ordinarias, mucho menor.
    assert t["gap_return"].abs().median() < t["event_gap_return"].abs().median()


def test_decomposition_identity_and_accounting(
    events: pd.DataFrame, prices: pd.DataFrame, engine: EventBacktest
) -> None:
    """Identidad multiplicativa por evento y contabilidad diaria que cuadra."""
    res = engine.run(events, prices, entry_offset=1, exit_offset=30, max_concurrent=None)
    t = res.trades
    lhs = (1.0 + t["gap_return"]) * (1.0 + t["intraday_return"])
    assert np.allclose(lhs, 1.0 + t["price_return"], atol=1e-12)
    assert np.allclose(t["log_gap"] + t["log_intraday"], t["log_total"], atol=1e-12)
    # La suma de P&L diarios reproduce el total por eventos (contabilidad aditiva).
    assert res.daily["portfolio_return"].sum() == pytest.approx(
        res.summary["total_net_pnl"], abs=1e-10
    )
    assert (t["holding_sessions"] == 29).all()


# ------------------------------------------------------------- concurrencia

def test_concurrency_limit_respected(
    events: pd.DataFrame, prices: pd.DataFrame, engine: EventBacktest
) -> None:
    """El límite de posiciones simultáneas se respeta día a día y los descartes
    quedan registrados con su motivo; la utilización del capital se reporta."""
    limited = engine.run(
        events, prices, entry_offset=0, exit_offset=10, score="sue", max_concurrent=5
    )
    assert int(limited.daily["n_active"].max()) <= 5
    assert limited.summary["max_deployed"] <= 1.0 + 1e-12  # 5 huecos de 1/5
    assert limited.summary["n_skipped_concurrency"] > 0
    reasons = set(limited.skipped["reason"])
    assert "concurrency_limit" in reasons and reasons <= set(SKIP_REASONS)
    assert 0.0 < limited.summary["avg_deployed"] <= 1.0
    assert 0.0 <= limited.summary["pct_days_at_limit"] <= 1.0

    unlimited = engine.run(
        events, prices, entry_offset=0, exit_offset=10, score="sue", max_concurrent=None
    )
    assert len(unlimited.trades) > len(limited.trades)
    # Contabilidad: candidatos = ejecutados + descartados, en ambos casos.
    for res in (limited, unlimited):
        assert res.summary["n_candidates"] == len(res.trades) + len(res.skipped)


def test_concurrency_prefers_high_absolute_score(
    events: pd.DataFrame, prices: pd.DataFrame, engine: EventBacktest
) -> None:
    """Con los huecos llenos, se ejecutan los eventos de |score| mayor."""
    limited = engine.run(
        events, prices, entry_offset=0, exit_offset=10, score="sue", max_concurrent=3
    )
    executed = limited.trades["score"].abs()
    dropped_ids = limited.skipped.loc[
        limited.skipped["reason"] == "concurrency_limit", "event_id"
    ]
    sue = events.set_index("event_id")["sue"]
    dropped = sue.reindex(dropped_ids).abs()
    # No es un orden total (depende del instante), pero en media el ejecutado
    # debe tener bastante más |score| que el descartado.
    assert executed.mean() > dropped.mean()


# ------------------------------------------------------------------- costes

def test_costs_monotone_and_shorts_pay_borrow(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    engine: EventBacktest,
    engine_costs: EventBacktest,
) -> None:
    """Más costes nunca mejoran el neto; los cortos devengan préstamo ACT/360."""
    kwargs = {"entry_offset": 1, "exit_offset": 20, "max_concurrent": None, "n_boot": 100}
    free = engine.run(events, prices, **kwargs)
    real = engine_costs.run(events, prices, **kwargs)
    double = EventBacktest(cost_model=CostModel().scaled(2.0)).run(events, prices, **kwargs)

    assert (real.trades["cost_total"] > 0.0).all()
    assert np.allclose(
        real.trades["net_return"],
        real.trades["gross_return"] - real.trades["cost_total"],
        atol=1e-15,
    )
    assert free.summary["mean_net"] > real.summary["mean_net"] > double.summary["mean_net"]
    # Largos no pagan préstamo; cortos sí, proporcional a los días naturales.
    assert (real.trades["cost_borrow"] == 0.0).all()
    shorts = engine_costs.run(events, prices, side="short", **kwargs)
    assert (shorts.trades["cost_borrow"] > 0.0).all()
    assert (shorts.trades["calendar_days"] >= shorts.trades["holding_sessions"]).all()


def test_event_open_execution_pays_wider_spread(
    events: pd.DataFrame, prices: pd.DataFrame
) -> None:
    """La ejecución en la apertura de tau=0 paga el spread ampliado (§9.5)."""
    base = EventBacktest(cost_model=CostModel(), event_open_spread_multiplier=1.0)
    wide = EventBacktest(cost_model=CostModel(), event_open_spread_multiplier=5.0)
    kwargs = {"entry_offset": 0, "exit_offset": 5, "max_concurrent": None, "n_boot": 100}
    r1 = base.run(events, prices, **kwargs)
    r5 = wide.run(events, prices, **kwargs)
    aligned = r1.trades["cost_entry"].align(r5.trades["cost_entry"], join="inner")
    assert (aligned[1] > aligned[0]).all()
    # La salida (cierre de tau=+5) no se ve afectada por el multiplicador.
    exit_aligned = r1.trades["cost_exit"].align(r5.trades["cost_exit"], join="inner")
    assert np.allclose(exit_aligned[0], exit_aligned[1], atol=1e-15)


# ---------------------------------------------------------------- run_grid

@pytest.fixture(scope="module")
def grid(events: pd.DataFrame, prices: pd.DataFrame, engine: EventBacktest) -> pd.DataFrame:
    return run_grid(
        events,
        prices,
        entry_offsets=[-5, 0, 1],
        exit_offsets=[0, 20],
        score="sue",
        side="long",
        engine=engine,
        calendar_known_in_advance=True,
        max_concurrent=20,
        n_boot=100,
    )


def test_grid_is_complete_and_labelled(grid: pd.DataFrame) -> None:
    """Todas las combinaciones aparecen; las imposibles se marcan, no se omiten."""
    assert list(grid.index.names) == ["entry_offset", "exit_offset"]
    assert len(grid) == 6
    assert grid.loc[(1, 0), "status"] == "invalid_offsets"
    assert np.isnan(grid.loc[(1, 0), "mean_net"])
    ok = grid[grid["status"] == "ok"]
    assert len(ok) == 5
    for col in (
        "n_events",
        "hit_rate",
        "mean_net",
        "mean_net_ci_low",
        "mean_net_ci_high",
        "median_net",
        "p01",
        "p05",
        "p95",
        "p99",
        "worst_net",
        "worst_event_id",
        "best_net",
        "mean_gap",
        "mean_intraday",
        "gap_share_log",
        "holds_event_gap_frac",
        "avg_deployed",
        "max_deployed",
        "sharpe_annualized",
        "sharpe_ci_low",
        "sharpe_ci_high",
        "q1_mean_net",
        "q5_mean_net",
        "q_spread_mean",
        "amc_frac",
    ):
        assert col in grid.columns, col


def test_grid_rows_are_internally_coherent(grid: pd.DataFrame) -> None:
    """Percentiles ordenados, hit rate en [0,1], extremos consistentes."""
    ok = grid[grid["status"] == "ok"]
    for _, row in ok.iterrows():
        assert 0.0 <= row["hit_rate"] <= 1.0
        chain = [
            row["worst_net"],
            row["p01"],
            row["p05"],
            row["p10"],
            row["p25"],
            row["median_net"],
            row["p75"],
            row["p90"],
            row["p95"],
            row["p99"],
            row["best_net"],
        ]
        assert all(a <= b + 1e-12 for a, b in zip(chain, chain[1:], strict=True))
        assert row["worst_net"] <= row["mean_net"] <= row["best_net"]
        assert row["n_events"] >= 10
        assert row["mean_net_ci_low"] <= row["mean_net"] <= row["mean_net_ci_high"]


def test_grid_distinguishes_crossing_vs_drift(grid: pd.DataFrame) -> None:
    """La rejilla separa las dos familias de estrategia: solo las entradas
    pre-anuncio mantienen el gap del evento."""
    assert grid.loc[(-5, 0), "holds_event_gap_frac"] == pytest.approx(1.0)
    assert grid.loc[(-5, 20), "holds_event_gap_frac"] == pytest.approx(1.0)
    assert grid.loc[(0, 20), "holds_event_gap_frac"] == pytest.approx(0.0)
    assert grid.loc[(1, 20), "holds_event_gap_frac"] == pytest.approx(0.0)
    # Y las tenencias largas acumulan más componente nocturno ordinario que las cortas.
    assert grid.loc[(0, 20), "avg_holding_sessions"] > grid.loc[(0, 0), "avg_holding_sessions"]


def test_grid_requires_declaration_for_negative_entries(
    events: pd.DataFrame, prices: pd.DataFrame, engine: EventBacktest
) -> None:
    """El barrido con entradas pre-anuncio exige la declaración PIT: no hay modo
    degradado por celda."""
    with pytest.raises(LookAheadError):
        run_grid(
            events,
            prices,
            entry_offsets=[-5, 1],
            exit_offsets=[20],
            engine=engine,
        )


def test_grid_empty_offsets_rejected(events: pd.DataFrame, prices: pd.DataFrame) -> None:
    with pytest.raises(ConfigError):
        run_grid(events, prices, entry_offsets=[], exit_offsets=[1])


def test_grid_marks_insufficient_cells(
    events: pd.DataFrame, prices: pd.DataFrame, engine: EventBacktest
) -> None:
    """Una celda cuya salida se sale del panel se marca, sin matar el barrido."""
    last_year = events[events["event_date"] >= "2022-10-01"]
    table = run_grid(
        last_year,
        prices,
        entry_offsets=[1],
        exit_offsets=[5, 250],
        engine=engine,
        min_events=10,
        n_boot=100,
    )
    assert table.loc[(1, 5), "status"] == "ok"
    assert table.loc[(1, 250), "status"] == "insufficient_events"


# ------------------------------------------------------- errores y contratos

def test_invalid_inputs_raise_config_errors(
    events: pd.DataFrame, prices: pd.DataFrame, engine: EventBacktest
) -> None:
    with pytest.raises(ConfigError, match="no precede"):
        engine.run(events, prices, entry_offset=5, exit_offset=1)
    with pytest.raises(ConfigError, match="entero"):
        engine.run(events, prices, entry_offset=1.5, exit_offset=5)  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="side"):
        engine.run(events, prices, entry_offset=1, exit_offset=5, side="hedged")  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="columna"):
        engine.run(events, prices, entry_offset=1, exit_offset=5, score="no_such_column")
    with pytest.raises(ConfigError, match="per_event_capital"):
        engine.run(events, prices, entry_offset=1, exit_offset=5, per_event_capital=1.5)
    with pytest.raises(ConfigError, match="max_concurrent"):
        engine.run(events, prices, entry_offset=1, exit_offset=5, max_concurrent=0)


def test_too_few_events_is_insufficient_history(
    events: pd.DataFrame, prices: pd.DataFrame, engine: EventBacktest
) -> None:
    """Tres eventos no son un backtest: InsufficientHistory con el recuento real."""
    with pytest.raises(InsufficientHistory, match="mínimo"):
        engine.run(events.head(3), prices, entry_offset=1, exit_offset=5, min_events=10)


def test_summarize_requires_engine_trades() -> None:
    with pytest.raises(DataQualityError, match="faltan columnas"):
        summarize_event_returns(pd.DataFrame({"net_return": [0.1, 0.2]}))


def test_engine_constructor_validation() -> None:
    with pytest.raises(ConfigError):
        EventBacktest(nav_usd=-1.0)
    with pytest.raises(ConfigError):
        EventBacktest(event_open_spread_multiplier=0.5)
    with pytest.raises(ConfigError):
        EventBacktest(dmh_policy="ignore")  # type: ignore[arg-type]


def test_deterministic_repeated_runs(
    events: pd.DataFrame, prices: pd.DataFrame, engine: EventBacktest
) -> None:
    """Misma entrada, mismo resultado, byte a byte (regla 4 del repo)."""
    kwargs = {
        "entry_offset": 0,
        "exit_offset": 10,
        "score": "sue",
        "max_concurrent": 10,
        "n_boot": 100,
    }
    a = engine.run(events, prices, **kwargs)
    b = engine.run(events, prices, **kwargs)
    pd.testing.assert_frame_equal(a.trades, b.trades)
    pd.testing.assert_frame_equal(a.daily, b.daily)
    assert a.summary == b.summary


def test_signed_side_skips_events_without_score(
    events: pd.DataFrame, prices: pd.DataFrame, engine: EventBacktest
) -> None:
    """side='signed' sin score utilizable descarta con motivo, nunca inventa lado."""
    partial = pd.Series(
        np.nan, index=pd.Index(events["event_id"], name="event_id"), dtype=float
    )
    keep = events["event_id"].iloc[: len(events) // 2]
    partial.loc[keep] = 1.0
    res = engine.run(
        events,
        prices,
        entry_offset=1,
        exit_offset=5,
        score=partial,
        side="signed",
        max_concurrent=None,
    )
    assert (res.skipped["reason"] == "score_missing").sum() >= len(events) // 2 - 5
    assert (res.trades["side"] == 1.0).all()
