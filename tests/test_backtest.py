"""Tests del módulo `backtest`: costes, cartera y motor cross-section.

Corren **sin red**. El material realista sale de `SyntheticMarket`; los
contratos de no-look-ahead, universo PIT y delisting se verifican con paneles
deterministas construidos a mano, donde el resultado correcto se conoce a
precisión de máquina.

Los cinco contratos que estos tests defienden:

1. **Señal perfecta ⇒ Sharpe altísimo; señal aleatoria ⇒ ~0.** Es el par de
   sanidad del motor: si la primera falla, el motor destruye información; si la
   segunda falla, la fabrica.
2. **Los costes reducen el retorno de forma monótona** y el bruto no cambia:
   el `CostModel` no puede tocar la construcción de la cartera.
3. **El universo PIT manda en cada rebalanceo**: un no-miembro no puede
   comprarse, y un miembro saliente se vende en el siguiente rebalanceo.
4. **La ejecución retardada no usa el precio de la señal**: un gap de apertura
   posterior a la señal no es capturable. Es el test anti-look-ahead central
   (`validation_methodology.md` §11.1).
5. **Un delisting se liquida y queda auditado** (`pit_and_biases.md` §6.3-6.4),
   nunca se evapora en silencio.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from earnings_alpha.backtest import (
    CostBreakdown,
    CostModel,
    CrossSectionalBacktest,
    SpreadTier,
    apply_participation_limit,
    apply_turnover_limit,
    assign_quantiles,
    balance_legs,
    build_target_weights,
    cap_weights,
    leg_weights,
    one_way_turnover,
    rebalance_schedule,
)
from earnings_alpha.data.synthetic import SyntheticMarket
from earnings_alpha.errors import (
    ConfigError,
    DataQualityError,
    InsufficientHistory,
    UniverseError,
)
from earnings_alpha.pit import get_calendar

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def market() -> SyntheticMarket:
    """Mercado sintético con sección cruzada suficiente para quintiles."""
    return SyntheticMarket(seed=20260804, n_tickers=40, start="2021-01-04", end="2022-12-30")


@pytest.fixture(scope="module")
def prices(market: SyntheticMarket) -> pd.DataFrame:
    return market.prices()


@pytest.fixture(scope="module")
def adj_open(prices: pd.DataFrame) -> pd.DataFrame:
    """Apertura ajustada en formato ancho, para construir señales de previsión."""
    close = prices["close"].unstack(level=-1)
    adjc = prices["adj_close"].unstack(level=-1)
    open_w = prices["open"].unstack(level=-1)
    return open_w * (adjc / close)


def _to_panel_series(frame: pd.DataFrame) -> pd.Series:
    """DataFrame ancho (date x ticker) -> Series canónica (date, ticker)."""
    out = frame.stack()
    out.index.names = ["date", "ticker"]
    return out.astype(float)


def _flat_panel(
    tickers: list[str],
    sessions: pd.DatetimeIndex,
    price: float = 100.0,
    volume: float = 1_000_000.0,
) -> pd.DataFrame:
    """Panel determinista de precios constantes con todas las columnas necesarias."""
    idx = pd.MultiIndex.from_product([sessions, tickers], names=["date", "ticker"])
    n = len(idx)
    return pd.DataFrame(
        {
            "open": np.full(n, price),
            "close": np.full(n, price),
            "adj_close": np.full(n, price),
            "volume": np.full(n, volume),
        },
        index=idx,
    )


SESSIONS = get_calendar().sessions("2022-01-03", "2022-06-30")


# ---------------------------------------------------------------------------
# CostModel
# ---------------------------------------------------------------------------


class TestCostModel:
    def test_spread_tiers_decrecientes_con_liquidez(self) -> None:
        cm = CostModel()
        advs = np.array([5e9, 5e8, 1e8, 2e7, 1e6])
        hs = cm.half_spread_fraction(advs)
        assert np.all(np.diff(hs) >= 0.0), "menos ADV nunca puede abaratar el spread"
        # fronteras exactas de los tramos por defecto
        assert cm.half_spread_fraction(1_000e6)[0] == pytest.approx(1.0e-4)
        assert cm.half_spread_fraction(999e6)[0] == pytest.approx(2.0e-4)
        assert cm.half_spread_fraction(0.0)[0] == pytest.approx(20.0e-4)

    def test_adv_desconocido_paga_el_peor_tramo(self) -> None:
        cm = CostModel()
        worst = cm.spread_tiers[-1].half_spread_bps * 1e-4
        assert cm.half_spread_fraction(np.nan)[0] == pytest.approx(worst)
        assert cm.half_spread_fraction(-5.0)[0] == pytest.approx(worst)

    def test_impacto_escala_con_raiz_de_la_participacion(self) -> None:
        cm = CostModel(impact_coefficient=0.5)
        base = cm.impact_fraction(0.01, 0.02)[0]
        quad = cm.impact_fraction(0.04, 0.02)[0]
        assert quad == pytest.approx(2.0 * base), "×4 participación ⇒ ×2 impacto unitario"
        # coste total de una orden: ×4 tamaño ⇒ ×8 dólares de impacto
        cb1 = CostModel(commission_bps=0.0).execution_costs([0.01], [1e8], [0.02], 1e8)
        cb4 = CostModel(commission_bps=0.0).execution_costs([0.04], [1e8], [0.02], 1e8)
        assert cb4.impact == pytest.approx(8.0 * cb1.impact)

    def test_desglose_aritmetica_exacta(self) -> None:
        cm = CostModel()
        cb = cm.execution_costs([0.1, -0.2], [1e8, 5e7], [0.015, 0.03], 1e7)
        # spread: ambos en el tramo [50M, 250M) = 4 pb de medio spread
        assert cb.spread == pytest.approx(4e-4 * 0.1 + 4e-4 * 0.2)
        # impacto: eta*sigma*sqrt(p)*|dw|, p = |dw|*NAV/ADV
        imp1 = 0.5 * 0.015 * np.sqrt(0.1 * 1e7 / 1e8) * 0.1
        imp2 = 0.5 * 0.03 * np.sqrt(0.2 * 1e7 / 5e7) * 0.2
        assert cb.impact == pytest.approx(imp1 + imp2)
        assert cb.commission == pytest.approx(0.5e-4 * 0.3)
        assert cb.total == pytest.approx(cb.spread + cb.impact + cb.commission)

    def test_operacion_nula_cuesta_cero_incluso_con_nan(self) -> None:
        cm = CostModel()
        cb = cm.execution_costs([0.0, 0.0], [np.nan, np.nan], [np.nan, np.nan], 1e7)
        assert cb.total == 0.0

    def test_zero_y_scaled(self) -> None:
        zero = CostModel.zero()
        assert zero.execution_costs([0.5], [np.nan], [np.nan], 1e7).total == 0.0
        cm = CostModel()
        cb1 = cm.execution_costs([0.1], [1e8], [0.02], 1e7)
        cb2 = cm.scaled(2.0).execution_costs([0.1], [1e8], [0.02], 1e7)
        assert cb2.total == pytest.approx(2.0 * cb1.total)
        assert cm.scaled(0.0).execution_costs([0.1], [1e8], [0.02], 1e7).total == 0.0

    def test_prestamo_solo_sobre_cortos_y_con_overrides(self) -> None:
        cm = CostModel(borrow_bps_annual=100.0, borrow_overrides_bps={"HTB": 500.0})
        cb = cm.holding_costs([0.5, -0.2, -0.1], ["L", "S", "HTB"], calendar_days=3)
        expected = (0.2 * 100e-4 / 360 + 0.1 * 500e-4 / 360) * 3
        assert cb.borrow == pytest.approx(expected)
        assert cb.spread == cb.impact == cb.commission == 0.0
        # sin cortos no hay devengo
        assert cm.holding_costs([0.5, 0.5], ["A", "B"], 3).total == 0.0

    def test_validaciones(self) -> None:
        with pytest.raises(ConfigError):
            CostModel(commission_bps=-1.0)
        with pytest.raises(ConfigError):
            CostModel(spread_tiers=(SpreadTier(10e6, 5.0),))  # no cubre ADV=0
        with pytest.raises(ConfigError):
            CostModel(
                spread_tiers=(SpreadTier(0.0, 5.0), SpreadTier(10e6, 2.0))
            )  # desordenado
        with pytest.raises(ConfigError):
            CostModel().scaled(-1.0)
        with pytest.raises(ConfigError):
            CostModel(borrow_overrides_bps={"X": -3.0})

    def test_suma_de_desgloses(self) -> None:
        total = CostBreakdown(spread=1.0) + CostBreakdown(borrow=2.0, impact=0.5)
        assert total.total == pytest.approx(3.5)


# ---------------------------------------------------------------------------
# Cartera
# ---------------------------------------------------------------------------


class TestPortfolio:
    def test_quantiles_equilibrados_y_deterministas(self) -> None:
        s = pd.Series(np.arange(20, dtype=float), index=[f"T{i:02d}" for i in range(20)])
        q = assign_quantiles(s, 5)
        counts = q.value_counts()
        assert sorted(counts.index) == [1.0, 2.0, 3.0, 4.0, 5.0]
        assert set(counts) == {4}
        assert q["T00"] == 1.0 and q["T19"] == 5.0
        # empates: determinista aunque cambie el orden de entrada
        tied = pd.Series(1.0, index=list("DCBA"))
        assert (
            assign_quantiles(tied, 2)
            .sort_index()
            .equals(assign_quantiles(tied.sort_index(), 2))
        )

    def test_quantiles_nan_y_grupo_pequeno(self) -> None:
        s = pd.Series([1.0, np.nan, 3.0, 4.0, 5.0], index=list("ABCDE"))
        q = assign_quantiles(s, 2)
        assert np.isnan(q["B"])
        by = pd.Series({"A": "g1", "B": "g1", "C": "g1", "D": "g2", "E": "g2"})
        q5 = assign_quantiles(s, 5, by=by)  # ningún grupo llega a 5 nombres
        assert q5.isna().all()

    def test_pesos_por_cubo_sumas_y_neutralidad(self) -> None:
        s = pd.Series(np.arange(10, dtype=float), index=[f"T{i}" for i in range(10)])
        w = build_target_weights(s, n_quantiles=5, long_short=True)
        assert w[w > 0].sum() == pytest.approx(1.0)
        assert w[w < 0].sum() == pytest.approx(-1.0)
        assert (w[["T8", "T9"]] > 0).all() and (w[["T0", "T1"]] < 0).all()
        assert (w[["T3", "T4", "T5", "T6"]] == 0.0).all()
        w_long = build_target_weights(s, n_quantiles=5, long_short=False)
        assert w_long.sum() == pytest.approx(1.0)
        assert (w_long >= 0).all()

    def test_ponderacion_por_score_monotona_en_la_pata(self) -> None:
        s = pd.Series([10.0, 11.0, 12.0, 13.0], index=list("ABCD"))
        w = leg_weights(s, 1.0, weighting="score")
        assert w.sum() == pytest.approx(1.0)
        assert w["D"] > w["C"] > w["B"] > w["A"] > 0
        ws = leg_weights(-s, -1.0, weighting="score")
        assert ws.sum() == pytest.approx(-1.0)
        assert ws["D"] < ws["C"] < ws["B"] < ws["A"] < 0
        # invariancia a transformaciones monótonas (usa rangos, no niveles)
        assert leg_weights(s**3, 1.0, weighting="score").equals(w)

    def test_cap_weights_redistribuye_y_respeta_tope(self) -> None:
        w = pd.Series([0.5, 0.3, 0.1, 0.1], index=list("ABCD"))
        capped = cap_weights(w, 0.35)
        assert capped.max() <= 0.35 + 1e-12
        assert capped.sum() == pytest.approx(1.0), "el exceso se redistribuye, no se pierde"
        # tope inviable: la pata queda desapalancada a n*cap, nunca viola el tope
        tight = cap_weights(w, 0.10)
        assert tight.max() <= 0.10 + 1e-12
        assert tight.sum() == pytest.approx(0.4)

    def test_balance_legs_restaura_neutralidad(self) -> None:
        w = pd.Series([0.6, 0.4, -0.3, -0.3], index=list("ABCD"))
        b = balance_legs(w)
        assert b[b > 0].sum() == pytest.approx(0.6)
        assert b[b < 0].sum() == pytest.approx(-0.6)

    def test_neutralidad_sectorial_exacta(self) -> None:
        rng = np.random.default_rng(11)
        names = [f"T{i:02d}" for i in range(24)]
        sectors = pd.Series((["Tech"] * 12) + (["Energy"] * 8) + (["Utils"] * 4), index=names)
        s = pd.Series(rng.normal(size=24), index=names)
        w = build_target_weights(s, n_quantiles=4, long_short=True, sectors=sectors)
        for sec in ("Tech", "Energy", "Utils"):
            net = w[sectors == sec].sum()
            assert net == pytest.approx(0.0, abs=1e-12), f"exposición neta en {sec}"
        assert w[w > 0].sum() == pytest.approx(1.0)

    def test_limite_de_rotacion(self) -> None:
        prev = pd.Series([0.5, 0.5, 0.0], index=list("ABC"))
        tgt = pd.Series([0.0, 0.5, 0.5], index=list("ABC"))
        assert one_way_turnover(prev, tgt) == pytest.approx(0.5)
        adjusted, lam = apply_turnover_limit(prev, tgt, 0.25)
        assert lam == pytest.approx(0.5)
        assert one_way_turnover(prev, adjusted) == pytest.approx(0.25)
        full, lam_full = apply_turnover_limit(prev, tgt, 5.0)
        assert lam_full == 1.0 and full.equals(tgt)

    def test_limite_de_participacion(self) -> None:
        prev = pd.Series([0.0, 0.0], index=list("AB"))
        tgt = pd.Series([0.3, -0.3], index=list("AB"))
        cap = pd.Series([0.1, np.nan], index=list("AB"))  # NaN = sin límite
        out = apply_participation_limit(prev, tgt, cap)
        assert out["A"] == pytest.approx(0.1)
        assert out["B"] == pytest.approx(-0.3)

    def test_seccion_insuficiente_lanza(self) -> None:
        s = pd.Series([1.0, 2.0], index=list("AB"))
        with pytest.raises(InsufficientHistory):
            build_target_weights(s, n_quantiles=5)

    def test_multiindex_rechazado(self) -> None:
        idx = pd.MultiIndex.from_product([[pd.Timestamp("2022-01-03")], ["A", "B"]])
        s = pd.Series([1.0, 2.0], index=idx)
        with pytest.raises(DataQualityError):
            assign_quantiles(s, 2)


# ---------------------------------------------------------------------------
# Calendario de rebalanceo
# ---------------------------------------------------------------------------


class TestRebalanceSchedule:
    def test_diario_semanal_mensual(self) -> None:
        daily = rebalance_schedule(SESSIONS, "D")
        assert daily.equals(SESSIONS)
        weekly = rebalance_schedule(SESSIONS, "W-FRI")
        assert len(weekly) < len(SESSIONS)
        # cada fecha semanal es la última sesión de su semana (viernes salvo festivo)
        assert (weekly.dayofweek <= 4).all()
        assert pd.Timestamp("2022-04-14") in weekly, "Viernes Santo: la semana acaba en jueves"
        monthly = rebalance_schedule(SESSIONS, "M")
        assert list(monthly[:3]) == [
            pd.Timestamp("2022-01-31"),
            pd.Timestamp("2022-02-28"),
            pd.Timestamp("2022-03-31"),
        ]
        assert rebalance_schedule(SESSIONS, "ME").equals(monthly)

    def test_regla_invalida(self) -> None:
        with pytest.raises(ConfigError):
            rebalance_schedule(SESSIONS, "cada luna llena")


# ---------------------------------------------------------------------------
# Motor: sanidad estadística
# ---------------------------------------------------------------------------


class TestEngineSignalSanity:
    def test_senal_perfecta_sharpe_altisimo(
        self, prices: pd.DataFrame, adj_open: pd.DataFrame
    ) -> None:
        """Previsión perfecta del retorno apertura->apertura del periodo de tenencia.

        Con rebalanceo diario y ejecución en la apertura siguiente, la señal de
        `t` se mantiene de open(t+1) a open(t+2): ese retorno ES la señal.
        """
        fwd = adj_open.shift(-2) / adj_open.shift(-1) - 1.0
        scores = _to_panel_series(fwd)
        bt = CrossSectionalBacktest()
        res = bt.run(
            scores,
            prices,
            n_quantiles=5,
            rebalance="D",
            costs=CostModel.zero(),
            max_weight=0.2,
            adv_participation=None,
        )
        sr = res.sharpe()
        assert sr.sharpe_annualized > 5.0
        assert res.returns.mean() > 0.005
        # y los cubos ordenan: el diagnóstico interno tiene que verlo
        q = res.quantile_returns.mean()
        assert q["q5"] > q["q1"]

    def test_senal_aleatoria_sharpe_cercano_a_cero(self, prices: pd.DataFrame) -> None:
        adjc = prices["adj_close"].unstack(level=-1)
        rng = np.random.default_rng(20260804)
        noise = pd.DataFrame(rng.normal(size=adjc.shape), index=adjc.index, columns=adjc.columns)
        scores = _to_panel_series(noise)
        res = CrossSectionalBacktest().run(
            scores,
            prices,
            rebalance="W-FRI",
            costs=CostModel.zero(),
            max_weight=0.2,
            adv_participation=None,
        )
        sr = res.sharpe()
        assert abs(sr.sharpe_annualized) < 1.5
        assert abs(res.returns.mean()) < 2e-3
        # el IC del ruido es 0: el intervalo de confianza debe contener el 0
        assert sr.ci_low < 0.0 < sr.ci_high

    def test_costes_reducen_retorno_monotonamente(self, prices: pd.DataFrame) -> None:
        adjc = prices["adj_close"].unstack(level=-1)
        rng = np.random.default_rng(7)
        noise = pd.DataFrame(rng.normal(size=adjc.shape), index=adjc.index, columns=adjc.columns)
        scores = _to_panel_series(noise)
        base = CostModel()
        totals: list[float] = []
        gross_totals: list[float] = []
        for factor in (0.0, 1.0, 2.0, 4.0):
            res = CrossSectionalBacktest().run(
                scores,
                prices,
                rebalance="W-FRI",
                costs=base.scaled(factor),
                max_weight=0.2,
                adv_participation=None,
            )
            totals.append(float((1.0 + res.returns).prod()))
            gross_totals.append(float((1.0 + res.gross_returns).prod()))
            assert (res.costs["total"] >= 0.0).all()
        assert all(a > b for a, b in pairwise(totals)), totals
        # el bruto no depende del modelo de costes: los costes no tocan la cartera
        assert gross_totals[0] == pytest.approx(gross_totals[1], rel=1e-12)
        assert gross_totals[0] == pytest.approx(gross_totals[3], rel=1e-12)

    def test_desglose_de_costes_positivo_y_coherente(self, prices: pd.DataFrame) -> None:
        adjc = prices["adj_close"].unstack(level=-1)
        rng = np.random.default_rng(3)
        noise = pd.DataFrame(rng.normal(size=adjc.shape), index=adjc.index, columns=adjc.columns)
        res = CrossSectionalBacktest().run(
            _to_panel_series(noise),
            prices,
            rebalance="W-FRI",
            costs=CostModel(),
            max_weight=0.2,
        )
        assert list(res.costs.columns) == ["spread", "impact", "commission", "borrow", "total"]
        sums = res.total_costs
        assert sums["spread"] > 0 and sums["impact"] > 0 and sums["commission"] > 0
        assert sums["borrow"] > 0, "una cartera long-short devenga préstamo"
        assert sums["total"] == pytest.approx(
            sums[["spread", "impact", "commission", "borrow"]].sum()
        )
        assert res.returns.equals(res.returns)  # sin NaN
        assert not res.returns.isna().any()
        assert (res.turnover > 0).all() and (res.turnover <= 2.0 + 1e-9).all()


# ---------------------------------------------------------------------------
# Motor: universo PIT
# ---------------------------------------------------------------------------


class TestEnginePITUniverse:
    def test_universo_pit_excluye_no_miembros(self, market: SyntheticMarket) -> None:
        px = market.prices()
        adjc = px["adj_close"].unstack(level=-1)
        tickers = list(adjc.columns)
        star, dead = tickers[0], tickers[1]

        # señal fija: `star` siempre arriba, `dead` en el medio, resto por orden
        base = pd.Series(np.linspace(-1.0, 1.0, len(tickers)), index=tickers)
        base[star] = 10.0
        scores_frame = pd.DataFrame(
            np.tile(base.to_numpy(), (len(adjc.index), 1)),
            index=adjc.index,
            columns=tickers,
        )
        scores = _to_panel_series(scores_frame)

        membership = pd.DataFrame(True, index=adjc.index, columns=tickers)
        membership[dead] = False  # nunca miembro
        out_start, out_end = pd.Timestamp("2021-06-01"), pd.Timestamp("2021-09-30")
        membership.loc[out_start:out_end, star] = False  # expulsión temporal

        res = CrossSectionalBacktest(universe=membership).run(
            scores,
            px,
            rebalance="W-FRI",
            costs=CostModel.zero(),
            max_weight=0.5,
            adv_participation=None,
        )
        # un nombre que jamás es miembro jamás se toca
        assert (res.weights[dead] == 0.0).all()
        assert (res.trades[dead] == 0.0).all()

        # el miembro expulsado se vende en el primer rebalanceo dentro de la ventana
        decisions_out = [d for d in res.rebalance_dates if out_start <= d <= out_end]
        assert decisions_out, "la ventana de expulsión debe contener rebalanceos"
        first_exec_out = res.execution_dates[list(res.rebalance_dates).index(decisions_out[0])]
        inside = res.weights.loc[first_exec_out:out_end, star]
        assert (inside == 0.0).all(), "pesos vivos de un no-miembro dentro de la ventana"
        # antes de la expulsión sí estaba en cartera (la señal lo pone arriba)
        before = res.weights.loc[: out_start - pd.Timedelta(days=1), star]
        assert (before > 0).any()
        # y tras volver al índice, vuelve a cartera
        after = res.weights.loc[out_end + pd.Timedelta(days=10) :, star]
        assert (after > 0).any()

    def test_universo_sin_cobertura_lanza(self, market: SyntheticMarket) -> None:
        px = market.prices()
        adjc = px["adj_close"].unstack(level=-1)
        rng = np.random.default_rng(5)
        scores = _to_panel_series(
            pd.DataFrame(rng.normal(size=adjc.shape), index=adjc.index, columns=adjc.columns)
        )
        late = pd.DataFrame(
            True, index=adjc.index[len(adjc) // 2 :], columns=list(adjc.columns)
        )
        with pytest.raises(UniverseError):
            CrossSectionalBacktest(universe=late).run(
                scores,
                px,
                rebalance="W-FRI",
                costs=CostModel.zero(),
                max_weight=0.5,
                adv_participation=None,
            )


# ---------------------------------------------------------------------------
# Motor: no look-ahead en la ejecución
# ---------------------------------------------------------------------------


class TestEngineNoLookahead:
    @staticmethod
    def _gap_setup() -> tuple[pd.DataFrame, pd.Series, pd.Timestamp, pd.Timestamp]:
        """Panel plano donde 'A' salta +30 % en el gap POSTERIOR a la señal.

        La señal existe solo el día `k`; el salto ocurre entre el cierre de `k`
        y la apertura de `k+1`. Un motor honesto ejecuta en la apertura de `k+1`
        (130) y no captura nada; uno con look-ahead compra al cierre de `k`
        (100) y "gana" un 30 %.
        """
        tickers = ["A", "B", "C", "D"]
        panel = _flat_panel(tickers, SESSIONS)
        k = 10
        sig_day, jump_day = SESSIONS[k], SESSIONS[k + 1]
        after = SESSIONS[k + 1 :]
        panel.loc[(after, "A"), ["open", "close", "adj_close"]] = 130.0

        scores = pd.Series(
            [1.0, -1.0, 0.5, -0.5],
            index=pd.MultiIndex.from_product([[sig_day], tickers], names=["date", "ticker"]),
        )
        return panel, scores, sig_day, jump_day

    def test_ejecucion_en_apertura_siguiente_no_captura_el_gap(self) -> None:
        panel, scores, sig_day, jump_day = self._gap_setup()
        res = CrossSectionalBacktest().run(
            scores,
            panel,
            n_quantiles=2,
            rebalance="D",
            costs=CostModel.zero(),
            max_weight=1.0,
            adv_participation=None,
            signal_staleness=0,
            warmup=0,
        )
        # el día de la señal no hay posición: la primera ejecución es el gap day
        assert res.execution_dates[0] == jump_day
        assert res.rebalance_dates[0] == sig_day
        # la posición se toma DESPUÉS del salto: retorno total exactamente 0
        assert float(res.returns.abs().sum()) == pytest.approx(0.0, abs=1e-12)
        assert float(res.nav.iloc[-1]) == pytest.approx(1.0, abs=1e-12)
        # y la cartera queda efectivamente larga de A a partir del gap day
        assert res.weights.loc[jump_day, "A"] > 0

    def test_ejecucion_next_close_tampoco_captura_el_gap(self) -> None:
        panel, scores, _, _ = self._gap_setup()
        res = CrossSectionalBacktest(execution="next_close").run(
            scores,
            panel,
            n_quantiles=2,
            rebalance="D",
            costs=CostModel.zero(),
            max_weight=1.0,
            adv_participation=None,
            signal_staleness=0,
            warmup=0,
        )
        assert float(res.returns.abs().sum()) == pytest.approx(0.0, abs=1e-12)

    def test_el_gap_lo_gana_quien_ya_estaba_dentro(self) -> None:
        """Control del test anterior: una señal un día ANTES sí captura el gap.

        Verifica que el test del gap tiene poder: el retorno está ahí y es
        capturable por quien compra en la apertura de `k` (antes del salto).
        """
        panel, _, sig_day, jump_day = self._gap_setup()
        tickers = ["A", "B", "C", "D"]
        early = SESSIONS[SESSIONS.get_loc(sig_day) - 1]
        scores = pd.Series(
            [1.0, -1.0, 0.5, -0.5],
            index=pd.MultiIndex.from_product([[early], tickers], names=["date", "ticker"]),
        )
        res = CrossSectionalBacktest().run(
            scores,
            panel,
            n_quantiles=2,
            rebalance="D",
            costs=CostModel.zero(),
            max_weight=1.0,
            adv_participation=None,
            signal_staleness=0,
            warmup=0,
        )
        # larga A al 50% de la pata: gap de +30% -> +15% de NAV el gap day
        assert float(res.returns.loc[jump_day]) == pytest.approx(0.15, abs=1e-9)


# ---------------------------------------------------------------------------
# Motor: delisting
# ---------------------------------------------------------------------------


class TestEngineDelisting:
    @staticmethod
    def _delisting_setup() -> tuple[pd.DataFrame, pd.Series, pd.Timestamp]:
        tickers = ["A", "B", "C", "D"]
        panel = _flat_panel(tickers, SESSIONS)
        sig_day = SESSIONS[5]
        dl_day = SESSIONS[40]
        # la serie de A termina en dl_day: no hay filas posteriores
        keep = ~(
            (panel.index.get_level_values("ticker") == "A")
            & (panel.index.get_level_values("date") > dl_day)
        )
        panel = panel[keep]
        scores = pd.Series(
            [1.0, -1.0, 0.5, -0.5],
            index=pd.MultiIndex.from_product([[sig_day], tickers], names=["date", "ticker"]),
        )
        return panel, scores, dl_day

    def _run(self, delist_returns: dict[str, float] | None):
        panel, scores, dl_day = self._delisting_setup()
        res = CrossSectionalBacktest().run(
            scores,
            panel,
            n_quantiles=2,
            rebalance="D",
            costs=CostModel.zero(),
            max_weight=1.0,
            adv_participation=None,
            signal_staleness=0,
            warmup=0,
            delist_returns=delist_returns,
        )
        return res, dl_day

    def test_delisting_liquida_y_queda_auditado(self) -> None:
        res, dl_day = self._run(None)
        # posición viva hasta el último precio, cero después
        assert res.weights.loc[dl_day - pd.Timedelta(days=1) :, "A"].iloc[0] >= 0.0
        after = res.weights.loc[res.weights.index > dl_day, "A"]
        assert (after == 0.0).all()
        assert not res.returns.isna().any()
        # sin retorno de delisting conocido: la auditoría lo registra (§6.4)
        assert ("A", dl_day) in res.silent_delistings
        # sin retorno final, la liquidación al último precio no mueve el NAV
        assert float(res.nav.iloc[-1]) == pytest.approx(1.0, abs=1e-12)

    def test_retorno_de_delisting_se_ejecuta(self) -> None:
        res, dl_day = self._run({"A": -0.30})
        # A pesaba 0.5 (media pata larga): la barra final resta 15% del NAV
        assert float(res.returns.loc[dl_day]) == pytest.approx(-0.15, abs=1e-9)
        assert float(res.nav.iloc[-1]) == pytest.approx(0.85, abs=1e-9)
        assert res.silent_delistings == ()
        # y la posición queda cerrada
        assert (res.weights.loc[res.weights.index > dl_day, "A"] == 0.0).all()


# ---------------------------------------------------------------------------
# Motor: contratos de interfaz y restricciones
# ---------------------------------------------------------------------------


class TestEngineContracts:
    def test_max_weight_respetado_en_ejecucion(self, prices: pd.DataFrame) -> None:
        adjc = prices["adj_close"].unstack(level=-1)
        rng = np.random.default_rng(13)
        scores = _to_panel_series(
            pd.DataFrame(rng.normal(size=adjc.shape), index=adjc.index, columns=adjc.columns)
        )
        res = CrossSectionalBacktest().run(
            scores,
            prices,
            rebalance="W-FRI",
            costs=CostModel.zero(),
            max_weight=0.15,
            adv_participation=None,
        )
        assert float(res.target_weights.abs().max().max()) <= 0.15 + 1e-9

    def test_limite_de_rotacion_en_el_motor(self, prices: pd.DataFrame) -> None:
        adjc = prices["adj_close"].unstack(level=-1)
        rng = np.random.default_rng(17)
        scores = _to_panel_series(
            pd.DataFrame(rng.normal(size=adjc.shape), index=adjc.index, columns=adjc.columns)
        )
        res = CrossSectionalBacktest().run(
            scores,
            prices,
            rebalance="W-FRI",
            costs=CostModel.zero(),
            max_weight=0.2,
            adv_participation=None,
            max_turnover=0.30,
        )
        assert (res.turnover <= 0.30 + 1e-9).all()

    def test_rotacion_semanal_mayor_que_mensual(self, prices: pd.DataFrame) -> None:
        adjc = prices["adj_close"].unstack(level=-1)
        rng = np.random.default_rng(19)
        scores = _to_panel_series(
            pd.DataFrame(rng.normal(size=adjc.shape), index=adjc.index, columns=adjc.columns)
        )
        kw: dict[str, object] = {
            "costs": CostModel.zero(),
            "max_weight": 0.2,
            "adv_participation": None,
        }
        weekly = CrossSectionalBacktest().run(scores, prices, rebalance="W-FRI", **kw)
        monthly = CrossSectionalBacktest().run(scores, prices, rebalance="M", **kw)
        assert weekly.turnover.sum() > monthly.turnover.sum()
        assert len(weekly.rebalance_dates) > len(monthly.rebalance_dates)

    def test_atribucion_sectorial_cierra_con_el_bruto(
        self, market: SyntheticMarket, prices: pd.DataFrame
    ) -> None:
        adjc = prices["adj_close"].unstack(level=-1)
        rng = np.random.default_rng(23)
        scores = _to_panel_series(
            pd.DataFrame(rng.normal(size=adjc.shape), index=adjc.index, columns=adjc.columns)
        )
        res = CrossSectionalBacktest(sectors=market.sectors()).run(
            scores,
            prices,
            rebalance="W-FRI",
            costs=CostModel.zero(),
            max_weight=0.2,
            adv_participation=None,
        )
        assert set(res.sector_attribution.columns) <= set(market.sectors().unique())
        # la suma por sectores reproduce el bruto (aritmético; el término cruzado
        # del día de ejecución es de segundo orden)
        diff = (res.sector_attribution.sum(axis=1) - res.gross_returns).abs()
        assert float(diff.max()) < 5e-4

    def test_neutralidad_sectorial_en_el_motor(
        self, market: SyntheticMarket, prices: pd.DataFrame
    ) -> None:
        adjc = prices["adj_close"].unstack(level=-1)
        rng = np.random.default_rng(29)
        scores = _to_panel_series(
            pd.DataFrame(rng.normal(size=adjc.shape), index=adjc.index, columns=adjc.columns)
        )
        sectors = market.sectors()
        res = CrossSectionalBacktest(sectors=sectors).run(
            scores,
            prices,
            n_quantiles=3,
            rebalance="M",
            costs=CostModel.zero(),
            max_weight=0.5,
            adv_participation=None,
            sector_neutral=True,
        )
        tgt = res.target_weights.iloc[0]
        for sec in sectors.unique():
            names = sectors.index[sectors == sec]
            net = float(tgt.reindex(names).fillna(0.0).sum())
            assert abs(net) < 1e-9, f"exposición neta en {sec}: {net}"

    def test_sin_open_exige_next_close(self) -> None:
        tickers = ["A", "B", "C", "D"]
        panel = _flat_panel(tickers, SESSIONS).drop(columns=["open"])
        scores = pd.Series(
            [1.0, -1.0, 0.5, -0.5],
            index=pd.MultiIndex.from_product([[SESSIONS[5]], tickers], names=["date", "ticker"]),
        )
        with pytest.raises(DataQualityError, match="next_open"):
            CrossSectionalBacktest().run(
                scores,
                panel,
                n_quantiles=2,
                costs=CostModel.zero(),
                max_weight=1.0,
                adv_participation=None,
                warmup=0,
            )
        # con next_close funciona
        res = CrossSectionalBacktest(execution="next_close").run(
            scores,
            panel,
            n_quantiles=2,
            rebalance="D",
            costs=CostModel.zero(),
            max_weight=1.0,
            adv_participation=None,
            signal_staleness=0,
            warmup=0,
        )
        assert float(res.nav.iloc[-1]) == pytest.approx(1.0, abs=1e-12)

    def test_sin_solapamiento_lanza_insufficient_history(self) -> None:
        tickers = ["A", "B", "C", "D"]
        panel = _flat_panel(tickers, SESSIONS)
        outside = pd.Timestamp("2010-01-05")
        scores = pd.Series(
            [1.0, -1.0, 0.5, -0.5],
            index=pd.MultiIndex.from_product([[outside], tickers], names=["date", "ticker"]),
        )
        with pytest.raises(InsufficientHistory):
            CrossSectionalBacktest().run(
                scores,
                panel,
                n_quantiles=2,
                costs=CostModel.zero(),
                max_weight=1.0,
                adv_participation=None,
                warmup=0,
            )

    def test_participacion_sin_volumen_lanza(self) -> None:
        tickers = ["A", "B", "C", "D"]
        panel = _flat_panel(tickers, SESSIONS).drop(columns=["volume"])
        scores = pd.Series(
            [1.0, -1.0, 0.5, -0.5],
            index=pd.MultiIndex.from_product([[SESSIONS[5]], tickers], names=["date", "ticker"]),
        )
        with pytest.raises(DataQualityError, match="adv_participation"):
            CrossSectionalBacktest().run(
                scores,
                panel,
                n_quantiles=2,
                costs=CostModel.zero(),
                max_weight=1.0,
                warmup=0,
            )

    def test_costs_obligatorio_y_tipado(self, prices: pd.DataFrame) -> None:
        adjc = prices["adj_close"].unstack(level=-1)
        scores = _to_panel_series(adjc * 0.0)
        with pytest.raises(ConfigError):
            CrossSectionalBacktest().run(scores, prices, costs="0.05%")  # type: ignore[arg-type]

    def test_resultado_reproducible_y_parametros_echo(self, prices: pd.DataFrame) -> None:
        adjc = prices["adj_close"].unstack(level=-1)
        rng = np.random.default_rng(31)
        scores = _to_panel_series(
            pd.DataFrame(rng.normal(size=adjc.shape), index=adjc.index, columns=adjc.columns)
        )
        kw: dict[str, object] = {
            "rebalance": "W-FRI",
            "costs": CostModel(),
            "max_weight": 0.2,
        }
        r1 = CrossSectionalBacktest().run(scores, prices, **kw)
        r2 = CrossSectionalBacktest().run(scores, prices, **kw)
        pd.testing.assert_series_equal(r1.returns, r2.returns)
        assert r1.params["rebalance"] == "W-FRI"
        assert r1.params["execution"] == "next_open"
        summ = r1.summary()
        assert {"sharpe", "ann_return", "max_drawdown", "total_costs"} <= set(summ)
