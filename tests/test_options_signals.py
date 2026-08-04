"""Tests de `earnings_alpha.events.options_signals` — sin red.

Cobertura, siguiendo `docs/research/options_signals.md` (el informe de opciones):

1. Aritmética Black-76 y las dos verificaciones numéricas del informe (§2.4
   desfase de captura, §3.4 comisión de préstamo) reproducidas con el código
   del módulo.
2. Forward implícito por regresión de paridad: recuperación exacta, absorción
   de una desviación uniforme de paridad (la razón de ser de §2.1) y fallo
   explícito con pocos pares.
3. Filtros de calidad F1..F6 con contabilidad de descartes y degradación del
   ticker-día (informe §2.3).
4. Punto fijo delta↔strike del 25-delta y no-intercambiabilidad con
   Xing-Zhang-Zhao (informe §4.2-§4.3, superficie de verificación del §14).
5. Straddle ATM por parábola y las identidades 0,79788 / 1,2533 (informe §8).
6. Descomposición de varianza del evento con dos vencimientos, caso A y B,
   estructura invertida → NaN, dos anuncios → NaN (informe §7.2-§7.3).
7. Cadenas del `SyntheticMarket`: recuperación del forward, de σ_E (jump_vol
   del generador), signo del skew y crush previsto vs realizado.
8. Pipeline por evento sobre el panel diario: detección de la filtración
   direccional contra la verdad-terreno, invariancia point-in-time al
   perturbar el futuro, gancho de `events.preevent` y fallos explícitos.
9. Humo sobre los fixtures reales LEAN (AAPL 2014-06-06), incluida la
   extracción de la volatilidad del evento con la fecha real de resultados.
"""

from __future__ import annotations

import math
import re
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtr

from earnings_alpha.data.synthetic import SyntheticMarket
from earnings_alpha.errors import DataQualityError, InsufficientHistory
from earnings_alpha.events import options_signals as osig
from earnings_alpha.events.options_signals import (
    ABS_MOVE_FACTOR,
    STRADDLE_TO_ONE_SIGMA,
    ImpliedForward,
    OptionsPreEventFeatures,
    QualityFilterConfig,
    aggregate_chain_daily,
    apply_quality_filters,
    atm_straddle,
    black76_price,
    compute_pre_event_option_features,
    event_variance_decomposition,
    expected_iv_crush,
    implied_forward,
    implied_vol_black76,
    option_stock_ratio,
    skew_25delta,
    skew_xzz,
    vol_spread_cw,
)
from earnings_alpha.events.preevent import (
    OPTION_FEATURE_COLUMNS,
    EventContext,
    PreEventFeatures,
)

LEAN_DIR = Path("/home/user/Testing/data/external/opciones/lean_sample")

# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def leaky() -> SyntheticMarket:
    """Mercado con filtración abundante para los tests de detección."""
    return SyntheticMarket(
        seed=1234, leak_fraction=0.35, n_tickers=18, start="2019-01-02", end="2023-12-29"
    )


@pytest.fixture(scope="module")
def ctx(leaky: SyntheticMarket) -> EventContext:
    return EventContext.from_synthetic(leaky)


@pytest.fixture(scope="module")
def feats(ctx: EventContext) -> pd.DataFrame:
    return OptionsPreEventFeatures().compute(ctx)


@pytest.fixture(scope="module")
def chain_mkt() -> SyntheticMarket:
    """Mercado pequeño para los tests de cadena por contrato."""
    return SyntheticMarket(seed=31, n_tickers=6, start="2021-01-04", end="2022-12-30")


def _welch(a: pd.Series, b: pd.Series) -> float:
    a, b = a.dropna(), b.dropna()
    assert len(a) > 15 and len(b) > 15, "muestra insuficiente para el contraste"
    return float((a.mean() - b.mean()) / math.sqrt(a.var() / len(a) + b.var() / len(b)))


def _doc_chain(
    *,
    forward: float = 100.0,
    tau: float = 30.0 / 365.0,
    rate: float = 0.045,
    vol_spread: float = 0.0,
    grid: float = 5.0,
    lo: float = 70.0,
    hi: float = 130.0,
    spread: float = 0.04,
    expiry: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Cadena sintética sobre la superficie de verificación del informe §14.

    ``IV(k) = 0,35 − 0,55·k + 1,2·k²`` con ``k = ln(K/F)``; las calls cotizan a
    ``IV + vs/2`` y las puts a ``IV − vs/2`` (el mecanismo de Cremers-Weinbaum).
    """
    exp = expiry if expiry is not None else pd.Timestamp("2025-02-21")
    df = math.exp(-rate * tau)
    rows = []
    for strike in np.arange(lo, hi + grid / 2, grid):
        k = math.log(strike / forward)
        iv = 0.35 - 0.55 * k + 1.2 * k * k
        for cp, right in ((1, "C"), (-1, "P")):
            price = black76_price(forward, strike, tau, iv + cp * vol_spread / 2.0, cp, df)
            rows.append(
                {
                    "as_of": pd.Timestamp("2025-01-20"),
                    "ticker": "DOC",
                    "expiry": exp,
                    "right": right,
                    "strike": float(strike),
                    "bid": max(price - spread / 2.0, 0.0),
                    "ask": price + spread / 2.0,
                    "days_to_expiry": round(tau * 365.0),
                    "open_interest": 100.0,
                    "volume": 10.0,
                    "spot": forward * df,
                }
            )
    return pd.DataFrame(rows)


def _fwd(forward: float, tau: float, discount: float = 1.0) -> ImpliedForward:
    return ImpliedForward(
        expiry=pd.Timestamp("2025-02-21"),
        tau=tau,
        forward=forward,
        discount=discount,
        n_pairs=9,
        rmse=0.0,
        implied_rate=-math.log(discount) / tau if discount != 1.0 else 0.0,
        implied_borrow=float("nan"),
    )


# ===========================================================================
# 1. Black-76 y las verificaciones numéricas del informe
# ===========================================================================


class TestBlack76:
    @pytest.mark.parametrize("cp", [1, -1])
    @pytest.mark.parametrize(
        ("strike", "tau", "sigma"), [(90.0, 0.1, 0.25), (100.0, 0.25, 0.35), (115.0, 0.5, 0.6)]
    )
    def test_ida_y_vuelta(self, cp: int, strike: float, tau: float, sigma: float) -> None:
        df = math.exp(-0.03 * tau)
        price = black76_price(100.0, strike, tau, sigma, cp, df)
        iv = implied_vol_black76(price, 100.0, strike, tau, cp, df)
        assert iv == pytest.approx(sigma, abs=1e-8)

    def test_precio_fuera_de_cotas_es_nan(self) -> None:
        """F9 del informe §2.3: sin convergencia → NaN, jamás un default."""
        assert math.isnan(implied_vol_black76(120.0, 100.0, 100.0, 0.1, 1))  # > DF·F
        assert math.isnan(implied_vol_black76(0.0, 100.0, 90.0, 0.1, 1))  # <= intrínseco
        assert math.isnan(implied_vol_black76(5.0, 100.0, 90.0, -0.1, 1))  # tau inválido

    def test_tau_cero_devuelve_intrinseco(self) -> None:
        assert black76_price(100.0, 90.0, 0.0, 0.3, 1) == pytest.approx(10.0)
        assert black76_price(100.0, 90.0, 0.0, 0.3, -1) == pytest.approx(0.0)

    def test_sesgo_por_comision_de_prestamo(self) -> None:
        """Informe §3.4 [verificado]: ``VS ≈ −f·√T/φ(0) = −2,5066·f·√T``."""
        s, r, sigma, tau, f = 100.0, 0.045, 0.35, 30.0 / 365.0, 0.02
        df = math.exp(-r * tau)
        f_true, f_naive = s * math.exp((r - f) * tau), s * math.exp(r * tau)
        call = black76_price(f_true, s, tau, sigma, 1, df)
        put = black76_price(f_true, s, tau, sigma, -1, df)
        vs = implied_vol_black76(call, f_naive, s, tau, 1, df) - implied_vol_black76(
            put, f_naive, s, tau, -1, df
        )
        phi0 = 1.0 / math.sqrt(2.0 * math.pi)
        assert vs == pytest.approx(-f * math.sqrt(tau) / phi0, abs=5e-5)

    def test_sesgo_por_desfase_de_captura(self) -> None:
        """Informe §2.4 [verificado]: ``VS ≈ −ε/(φ(0)·√T)``; 10 pb → 0,88 pts a 30 d."""
        s, r, sigma, tau, eps = 100.0, 0.045, 0.35, 30.0 / 365.0, 0.001
        df = math.exp(-r * tau)
        f_real, f_used = s * math.exp(r * tau), s * (1 + eps) * math.exp(r * tau)
        call = black76_price(f_real, s, tau, sigma, 1, df)
        put = black76_price(f_real, s, tau, sigma, -1, df)
        vs = implied_vol_black76(call, f_used, s, tau, 1, df) - implied_vol_black76(
            put, f_used, s, tau, -1, df
        )
        phi0 = 1.0 / math.sqrt(2.0 * math.pi)
        assert vs == pytest.approx(-eps / (phi0 * math.sqrt(tau)), abs=5e-5)
        assert vs == pytest.approx(-0.0088, abs=3e-4)


# ===========================================================================
# 2. Forward implícito por regresión de paridad (informe §2.1)
# ===========================================================================


class TestImpliedForward:
    def test_recupera_forward_descuento_y_prestamo(self) -> None:
        """Con paridad exacta la regresión es exacta, y ``f = −ln(F·DF/S)/τ``."""
        s, r, borrow, tau = 250.0, 0.04, 0.015, 60.0 / 365.0
        f_true = s * math.exp((r - borrow) * tau)
        df_true = math.exp(-r * tau)
        chain = _doc_chain(forward=f_true, tau=tau, rate=r, lo=200.0, hi=300.0, grid=5.0)
        chain["spot"] = s
        fw = implied_forward(chain, tau=tau, spot=s)
        assert fw is not None
        assert fw.forward == pytest.approx(f_true, rel=2e-4)
        assert fw.discount == pytest.approx(df_true, abs=2e-3)
        assert fw.implied_borrow == pytest.approx(borrow, abs=2e-3)

    def test_pocos_pares_devuelve_none(self) -> None:
        chain = _doc_chain(lo=95.0, hi=105.0, grid=5.0)  # 3 strikes < min_pairs=4
        assert implied_forward(chain, tau=30.0 / 365.0) is None

    def test_absorcion_de_la_desviacion_uniforme(self) -> None:
        """La razón de ser del §2.1: un shift uniforme call-put migra al forward.

        Con el forward VERDADERO el `vol_spread` medido recupera el shift
        inyectado exactamente; con el forward de la regresión de paridad, la
        parte uniforme de la desviación queda absorbida en (F, DF) — que es
        donde el informe §3.4 la quiere, como `implied_borrow` — y el spread
        residual es mucho menor. Ambas cosas se comprueban.
        """
        vs_true, tau = 0.03, 30.0 / 365.0
        chain = _doc_chain(vol_spread=vs_true, tau=tau, spread=0.02)
        true_fwd = {pd.Timestamp("2025-02-21"): _fwd(100.0, tau, math.exp(-0.045 * tau))}
        vs_with_true, n_pairs = vol_spread_cw(chain, true_fwd)
        assert n_pairs >= 3
        assert vs_with_true == pytest.approx(vs_true, abs=3e-3)

        fw_hat = implied_forward(chain, tau=tau)
        assert fw_hat is not None
        assert fw_hat.forward > 100.0  # calls caras → forward implícito mayor
        vs_with_hat, _ = vol_spread_cw(chain, {fw_hat.expiry: fw_hat})
        assert abs(vs_with_hat) < 0.5 * vs_true


# ===========================================================================
# 3. Filtros de calidad (informe §2.3)
# ===========================================================================


class TestQualityFilters:
    def _corrupt_chain(self) -> pd.DataFrame:
        exp = pd.Timestamp("2025-02-21")
        base = {"expiry": exp, "right": "C", "days_to_expiry": 30}
        rows = [
            # 3 contratos limpios
            {**base, "strike": 95.0, "bid": 6.0, "ask": 6.2},
            {**base, "strike": 100.0, "bid": 3.0, "ask": 3.2, "right": "P"},
            {**base, "strike": 105.0, "bid": 1.0, "ask": 1.1},
            # F1: bid cero (x2)
            {**base, "strike": 60.0, "bid": 0.0, "ask": 0.10},
            {**base, "strike": 150.0, "bid": 0.0, "ask": 0.05, "right": "P"},
            # F2: cruce
            {**base, "strike": 90.0, "bid": 5.0, "ask": 4.0},
            # F3: horquilla relativa > 50 %
            {**base, "strike": 110.0, "bid": 1.0, "ask": 2.0},
            # F4: prima < 0,05
            {**base, "strike": 120.0, "bid": 0.03, "ask": 0.04},
            # F5: call por encima de DF·F
            {**base, "strike": 100.0, "bid": 119.0, "ask": 121.0},
            # F6: DTE > 365
            {**base, "strike": 100.0, "bid": 10.0, "ask": 10.4, "days_to_expiry": 400},
        ]
        return pd.DataFrame(rows)

    def test_contabilidad_por_filtro(self) -> None:
        chain = self._corrupt_chain()
        forwards = {pd.Timestamp("2025-02-21"): _fwd(100.0, 30.0 / 365.0)}
        alive, report = apply_quality_filters(chain, forwards=forwards)
        assert report.initial == 10
        assert report.dropped == {
            "F1_bid_cero": 2,
            "F2_cruce": 1,
            "F3_horquilla": 1,
            "F4_prima_minima": 1,
            "F5_no_arbitraje": 1,
            "F6_dte": 1,
        }
        assert report.remaining == 3 == len(alive)
        assert sorted(alive["strike"]) == [95.0, 100.0, 105.0]

    def test_umbral_de_degradacion(self) -> None:
        chain = self._corrupt_chain()
        _, report = apply_quality_filters(chain)  # sin F5 (sin forwards)
        assert report.dropped["F5_no_arbitraje"] == 0
        assert report.dropped_fraction == pytest.approx(0.6)
        assert not report.usable(QualityFilterConfig())
        assert report.usable(QualityFilterConfig(max_drop_fraction=0.75))

    def test_ticker_dia_corrupto_degrada_a_nan(self) -> None:
        """Informe §2.3: >40 % de descartes → el día no genera señal, genera NaN."""
        chain = _doc_chain(spread=0.02)
        corrupt = chain.copy()
        corrupt.loc[corrupt.index[: int(len(corrupt) * 0.55)], "bid"] = 0.0
        agg = aggregate_chain_daily(corrupt)
        row = agg.iloc[0]
        assert row["quality_drop_fraction"] > 0.40
        assert math.isnan(row["vol_spread"])
        assert math.isnan(row["iv_skew_25delta"])
        assert row["n_contracts"] == len(corrupt)


# ===========================================================================
# 4. Skew: punto fijo 25-delta y Xing-Zhang-Zhao (informe §4)
# ===========================================================================


class TestSkew:
    def _smile(self, grid: float = 5.0):
        tau = 30.0 / 365.0
        chain = _doc_chain(tau=tau, grid=grid, spread=0.0001)
        fw = _fwd(100.0, tau, math.exp(-0.045 * tau))
        filtered, _ = apply_quality_filters(chain, forwards={fw.expiry: fw})
        smile = osig._build_smile(filtered, fw)
        assert smile is not None
        return smile

    def test_punto_fijo_25delta_en_la_superficie_del_informe(self) -> None:
        """Informe §4.2 [verificado]: rr25 verdadero = −7,529 pts con malla de 5 $.

        Nuestra columna es ``put − call`` = −rr25 = **+7,529 pts**. El punto
        fijo comete ~0,02 pts; el atajo del strike listado más cercano, ~1,9.
        """
        smile = self._smile(grid=5.0)
        skew = skew_25delta(smile)
        assert skew == pytest.approx(0.07529, abs=2e-3)

        # El atajo ingenuo: strike listado cuyo delta esté más cerca de ±0,25.
        def naive(target: float, side: str) -> float:
            ks = smile.k_call if side == "C" else smile.k_put
            ivs = smile.iv_call if side == "C" else smile.iv_put
            deltas = []
            for k, iv in zip(ks, ivs, strict=True):
                v = iv * math.sqrt(smile.tau)
                deltas.append(float(ndtr((-k + 0.5 * v * v) / v)))
            arr = np.asarray(deltas)
            return float(ivs[int(np.argmin(np.abs(arr - target)))])

        naive_skew = naive(0.75, "P") - naive(0.25, "C")
        assert abs(naive_skew - 0.07529) > 5.0 * abs(skew - 0.07529)

    def test_xzz_y_25delta_no_son_intercambiables(self) -> None:
        """Informe §4.3 [verificado]: XZZ ≈ +3,14 pts vs 25Δ ≈ +7,53 pts.

        Se pasa ``spot = F`` para que el strike 95 caiga exactamente en el
        borde 0,95 de la banda OTM del paper: con ``spot = F·DF`` la selección
        de strikes listados saltaría (correctamente) al siguiente strike y la
        cifra de referencia del informe, evaluada en el continuo, no aplicaría.
        """
        smile = self._smile()
        xzz = skew_xzz(smile, spot=smile.forward)
        s25 = skew_25delta(smile)
        assert xzz == pytest.approx(0.0314, abs=4e-3)
        assert s25 / xzz > 2.0  # misma pendiente, brazos de palanca distintos

    def test_xzz_sin_strikes_en_banda_es_nan(self) -> None:
        smile = self._smile()
        assert math.isnan(skew_xzz(smile, spot=float("nan")))
        assert math.isnan(skew_xzz(smile, spot=500.0))  # ningún strike en banda


# ===========================================================================
# 5. Straddle ATM y movimiento esperado (informe §8)
# ===========================================================================


class TestStraddle:
    def test_identidades_de_conversion(self) -> None:
        """Informe §8.1-§8.2: straddle = 0,79788·F·σ√T y 1σ = 1,2533·straddle."""
        assert pytest.approx(0.7978845608, abs=1e-9) == ABS_MOVE_FACTOR
        assert pytest.approx(1.2533141373, abs=1e-9) == STRADDLE_TO_ONE_SIGMA
        assert pytest.approx(1.0) == ABS_MOVE_FACTOR * STRADDLE_TO_ONE_SIGMA
        f, tau, sigma = 100.0, 30.0 / 365.0, 0.30
        straddle = black76_price(f, f, tau, sigma, 1) + black76_price(f, f, tau, sigma, -1)
        exact = 2.0 * f * (2.0 * ndtr(sigma * math.sqrt(tau) / 2.0) - 1.0)
        assert straddle == pytest.approx(exact, abs=1e-10)
        assert straddle / (f * sigma * math.sqrt(tau)) == pytest.approx(
            ABS_MOVE_FACTOR, abs=1e-3
        )

    def test_parabola_mejor_que_interpolacion_lineal(self) -> None:
        """Informe §8.3 [verificado]: el straddle tiene un mínimo en K≈F, luego
        la recta SIEMPRE sobreestima; la parábola es casi exacta."""
        tau, sigma, rate = 30.0 / 365.0, 0.35, 0.045
        df = math.exp(-rate * tau)
        forward = 102.4  # deliberadamente entre strikes de la malla de 5 $
        rows = []
        strikes = np.arange(80.0, 125.1, 5.0)
        for strike in strikes:
            for cp, right in ((1, "C"), (-1, "P")):
                price = black76_price(forward, float(strike), tau, sigma, cp, df)
                rows.append(
                    {
                        "expiry": pd.Timestamp("2025-02-21"),
                        "right": right,
                        "strike": float(strike),
                        "bid": price,
                        "ask": price,
                    }
                )
        chain = pd.DataFrame(rows)
        fw = _fwd(forward, tau, df)
        truth = black76_price(forward, forward, tau, sigma, 1, df) + black76_price(
            forward, forward, tau, sigma, -1, df
        )
        parabola = atm_straddle(chain, fw)
        assert parabola == pytest.approx(truth, rel=2e-3)

        # Interpolación lineal entre los dos strikes que rodean F: sobreestima.
        pairs = chain.pivot_table(index="strike", columns="right", values="bid")
        stra = (pairs["C"] + pairs["P"]).sort_index()
        below = stra.index[stra.index <= forward][-1]
        above = stra.index[stra.index > forward][0]
        w = (forward - below) / (above - below)
        linear = float((1 - w) * stra[below] + w * stra[above])
        assert linear > truth
        assert abs(parabola - truth) < abs(linear - truth) / 5.0


# ===========================================================================
# 6. Descomposición de varianza del evento (informe §7)
# ===========================================================================


class TestEventVariance:
    def test_caso_a_recuperacion_exacta(self) -> None:
        """Tabla del informe §7.2: recuperación exacta hasta precisión de máquina."""
        sd, se = 0.30, 0.05
        for t1_d, t2_d in [(7, 35), (3, 31), (10, 45), (21, 49)]:
            t1, t2 = t1_d / 365.0, t2_d / 365.0
            iv1 = math.sqrt(sd**2 + se**2 / t1)
            iv2 = math.sqrt(sd**2 + se**2 / t2)
            deco = event_variance_decomposition(iv1, t1, iv2, t2, 1, 1)
            assert deco.sigma_diffusive == pytest.approx(sd, abs=1e-12)
            assert deco.sigma_event == pytest.approx(se, abs=1e-12)

    def test_crush_previsto_36_por_ciento_a_7_dias(self) -> None:
        sd, se, t1 = 0.30, 0.05, 7.0 / 365.0
        iv1 = math.sqrt(sd**2 + se**2 / t1)
        crush = expected_iv_crush(sd, se, iv1, t1)
        assert crush == pytest.approx(0.361, abs=2e-3)
        assert crush == pytest.approx(1.0 - math.sqrt(1.0 - se**2 / (iv1**2 * t1)), abs=1e-9)

    def test_caso_b_con_weeklies(self) -> None:
        """El corto vence antes del anuncio: σ_d = IV_pre, σ_E² = T₂·(IV₂²−IV₁²)."""
        sd, se, t1, t2 = 0.28, 0.06, 10.0 / 365.0, 17.0 / 365.0
        iv1 = sd
        iv2 = math.sqrt(sd**2 + se**2 / t2)
        deco = event_variance_decomposition(iv1, t1, iv2, t2, 0, 1)
        assert deco.sigma_diffusive == pytest.approx(sd, abs=1e-12)
        assert deco.sigma_event == pytest.approx(se, abs=1e-12)

    def test_estructura_invertida_devuelve_nan_no_cero(self) -> None:
        """Informe §7.2: σ_d² negativa → NaN; truncar crearía un suelo espurio."""
        deco = event_variance_decomposition(0.80, 7.0 / 365.0, 0.20, 35.0 / 365.0, 1, 1)
        assert math.isnan(deco.sigma_diffusive)
        assert math.isnan(deco.sigma_event)
        assert not deco.is_valid

    def test_conteos_de_anuncios_incompatibles_nan(self) -> None:
        """Dos anuncios en el vencimiento lejano → la resta da basura → NaN."""
        t1, t2 = 7.0 / 365.0, 100.0 / 365.0
        for n1, n2 in [(1, 2), (0, 0), (0, 2), (2, 2)]:
            deco = event_variance_decomposition(0.45, t1, 0.33, t2, n1, n2)
            assert not deco.is_valid, (n1, n2)


# ===========================================================================
# 7. O/S: unidades y signo de Johnson-So
# ===========================================================================


class TestOptionStock:
    def test_unidades_con_multiplicador_100(self) -> None:
        assert option_stock_ratio(500.0, 1_000_000.0) == pytest.approx(0.05)
        serie = option_stock_ratio(
            pd.Series([100.0, 200.0]), pd.Series([10_000.0, 10_000.0])
        )
        assert list(serie) == [pytest.approx(1.0), pytest.approx(2.0)]

    def test_signo_negativo_documentado_johnson_so(self) -> None:
        """La tarea lo exige explícitamente: el docstring debe advertir que el
        signo predictivo del O/S es NEGATIVO según Johnson-So (2012)."""
        doc = option_stock_ratio.__doc__ or ""
        assert "Johnson y So (2012" in doc
        assert "NEGATIVO" in doc
        assert re.search(r"bajista", doc, flags=re.IGNORECASE)

    def test_direccion_es_menos_magnitud(self, feats: pd.DataFrame) -> None:
        both = feats[["option_stock_magnitude", "option_stock_direction"]].dropna()
        assert len(both) > 50
        assert np.allclose(both["option_stock_direction"], -both["option_stock_magnitude"])


# ===========================================================================
# 8. Cadenas del SyntheticMarket
# ===========================================================================


@pytest.fixture(scope="module")
def sample_days(chain_mkt: SyntheticMarket) -> pd.DataFrame:
    """Cinco ticker-días en T−1 de un anuncio, agregados desde la cadena."""
    daily = chain_mkt.options_daily()
    rows = daily[daily["days_to_earnings"] == 1.0]
    picks = rows.index[:: max(1, len(rows) // 5)][:5]
    chains = [chain_mkt.options_chain(tickers=t, asof=d) for d, t in picks]
    agg = aggregate_chain_daily(
        pd.concat(chains, ignore_index=True), events=chain_mkt.events()
    )
    assert len(agg) == len(picks)
    return agg


class TestChainSynthetic:
    def test_forward_recuperado_de_los_mids(
        self, chain_mkt: SyntheticMarket, sample_days: pd.DataFrame
    ) -> None:
        """La regresión de paridad recupera el forward del generador (<1 %)."""
        for (day, ticker), row in sample_days.iterrows():
            chain = chain_mkt.options_chain(tickers=ticker, asof=day)
            front = chain.loc[chain["expiry"] == chain["expiry"].min(), "forward"].iloc[0]
            assert row["forward_front"] == pytest.approx(front, rel=0.01)

    def test_sigma_evento_recupera_el_jump_del_generador(
        self, chain_mkt: SyntheticMarket, sample_days: pd.DataFrame
    ) -> None:
        """σ_E extraída de dos vencimientos ≈ jump_vol latente (±30 %).

        La tolerancia cubre la contaminación por la pendiente difusiva del
        generador (los dos vencimientos no comparten término difusivo exacto).
        """
        ok = 0
        for (_, ticker), row in sample_days.iterrows():
            jump = float(chain_mkt._meta.loc[ticker, "jump_vol"])
            if np.isfinite(row["sigma_event"]):
                assert row["sigma_event"] == pytest.approx(jump, rel=0.30)
                ok += 1
        assert ok >= 3, "demasiados ticker-días sin descomposición del evento"

    def test_signo_del_skew_y_consistencia(self, sample_days: pd.DataFrame) -> None:
        """Smirk de renta variable: puts caras (skew>0, rr25<0) y magnitudes sanas."""
        skews = sample_days["iv_skew_25delta"].dropna()
        assert len(skews) >= 3
        assert (skews > 0).all()  # put − call > 0: el risk reversal es negativo
        xzz = sample_days["iv_skew_xzz"].dropna()
        assert (xzz > 0).all()
        for _, row in sample_days.iterrows():
            if np.isfinite(row["one_sigma_move"]) and np.isfinite(row["iv_atm_front"]):
                approx = row["iv_atm_front"] * math.sqrt(row["dte_front"] / 365.0)
                assert row["one_sigma_move"] == pytest.approx(approx, rel=0.10)
            if np.isfinite(row["expected_abs_move"]):
                assert row["expected_abs_move"] == pytest.approx(
                    ABS_MOVE_FACTOR * row["sigma_event"], rel=1e-9
                )

    def test_calidad_y_crush_dentro_de_rango(self, sample_days: pd.DataFrame) -> None:
        assert (sample_days["quality_drop_fraction"] < 0.40).all()
        crush = sample_days["iv_crush_expected"].dropna()
        assert len(crush) >= 3
        assert ((crush > 0.0) & (crush < 0.9)).all()


# ===========================================================================
# 9. Pipeline por evento sobre el panel diario
# ===========================================================================


class TestPanelFeatures:
    def test_estructura_del_contrato(self, feats: pd.DataFrame) -> None:
        assert feats.index.name == "event_id"
        assert feats.index.is_unique
        for col in OPTION_FEATURE_COLUMNS:
            assert col in feats.columns, f"falta la columna del contrato {col!r}"
            assert feats[col].notna().sum() > 100, f"{col} no se calcula nunca"
        for col in osig.OPTION_EVENT_FEATURES:
            assert col in feats.columns
        assert (feats["available_at"] == feats["event_date"]).all()

    def test_deteccion_direccional_contra_la_verdad_terreno(
        self, leaky: SyntheticMarket, feats: pd.DataFrame
    ) -> None:
        """Los signos del informe §1 sobre los eventos filtrados del generador.

        Filtración alcista vs bajista: `vol_spread` sube (Cremers-Weinbaum,
        signo +), el skew put-call baja (XZZ, signo −), la cuota de puts baja
        (Pan-Poteshman, signo −) y el OI neto de calls sube (Fodor et al.).
        """
        truth = leaky.ground_truth()
        leaked = truth[truth["is_leaked"]]
        common = feats.index.intersection(leaked.index)
        sign = leaked.loc[common, "surprise_sign"]
        bull, bear = common[sign > 0], common[sign < 0]
        clean = feats.index.difference(leaked.index)

        assert _welch(feats.loc[bull, "vol_spread"], feats.loc[bear, "vol_spread"]) > 5.0
        assert _welch(feats.loc[bull, "iv_skew_25delta"], feats.loc[bear, "iv_skew_25delta"]) < -5.0
        assert (
            _welch(
                feats.loc[bull, "put_call_volume_ratio"],
                feats.loc[bear, "put_call_volume_ratio"],
            )
            < -5.0
        )
        assert _welch(feats.loc[bull, "oi_buildup_net"], feats.loc[bear, "oi_buildup_net"]) > 3.0
        assert _welch(feats.loc[bull, "cavs_20"], feats.loc[clean, "cavs_20"]) > 3.0
        assert _welch(feats.loc[bull, "d_vol_spread_5"], feats.loc[clean, "d_vol_spread_5"]) > 3.0

    def test_event_iv_recupera_el_jump_por_ticker(
        self, leaky: SyntheticMarket, feats: pd.DataFrame
    ) -> None:
        """La σ_E mediana por emisor reproduce el jump_vol latente del generador."""
        per_ticker = feats.groupby("ticker")["event_iv"].median().dropna()
        jump = leaky._meta["jump_vol"]
        both = pd.concat([per_ticker, jump], axis=1).dropna()
        assert len(both) >= 12
        ratio = both["event_iv"] / both["jump_vol"]
        assert float(both.corr().iloc[0, 1]) > 0.95
        assert ((ratio > 0.85) & (ratio < 1.25)).all()

    def test_crush_previsto_contra_el_realizado(
        self, leaky: SyntheticMarket, feats: pd.DataFrame
    ) -> None:
        """El crush previsto en T−1 anticipa el colapso realizado de la IV frontal."""
        daily = leaky.options_daily()
        iv7 = daily["iv_atm_7d"].unstack("ticker")  # noqa: PD010 - sin duplicados que agregar
        events = leaky.events().set_index("event_id")
        pred_real: list[tuple[float, float]] = []
        for eid, pred in feats["iv_crush_expected"].dropna().items():
            if eid not in events.index:
                continue
            day = pd.Timestamp(events.loc[eid, "event_date"])
            pos = iv7.index.get_indexer([day])[0]
            if pos < 1:
                continue
            ticker = events.loc[eid, "ticker"]
            realized = 1.0 - float(iv7.iloc[pos][ticker]) / float(iv7.iloc[pos - 1][ticker])
            pred_real.append((float(pred), realized))
        arr = np.asarray(pred_real)
        assert len(arr) > 200
        corr = float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1])
        assert corr > 0.85
        assert float(np.mean(np.abs(arr[:, 0] - arr[:, 1]))) < 0.05

    def test_event_vol_surprise_exige_historial(self, feats: pd.DataFrame) -> None:
        """Informe §7.4: sin 8 trimestres previos de σ_E, NaN (nunca 2 puntos)."""
        got_some = False
        for _, sub in feats.dropna(subset=["event_iv"]).groupby("ticker"):
            ordered = sub.sort_values("event_date")["event_vol_surprise"]
            assert ordered.iloc[: min(8, len(ordered))].isna().all()
            if ordered.notna().any():
                got_some = True
        assert got_some, "ningún emisor llega a acumular 8 trimestres de σ_E"

    def test_move_ratio_hist_plausible(self, feats: pd.DataFrame) -> None:
        ratios = feats["move_ratio_hist"].dropna()
        assert len(ratios) > 100
        assert (ratios > 0).all()
        assert 0.5 < float(ratios.median()) < 1.8

    def test_perturbar_el_futuro_no_mueve_ni_un_bit(
        self, leaky: SyntheticMarket, ctx: EventContext, feats: pd.DataFrame
    ) -> None:
        """CRÍTICO (point-in-time): datos de T en adelante no tocan ninguna feature."""
        events = leaky.events().set_index("event_id")
        usable = feats.dropna(subset=["vol_spread", "put_call_volume_ratio"]).index
        sample = list(usable[:: max(1, len(usable) // 3)][:3])
        assert len(sample) == 3
        for eid in sample:
            day = pd.Timestamp(events.loc[eid, "event_date"])
            ticker = events.loc[eid, "ticker"]
            options = ctx.options.copy()
            mask = (options.index.get_level_values("ticker") == ticker) & (
                options.index.get_level_values("date") >= day
            )
            assert mask.sum() > 0
            options.loc[mask, :] = options.loc[mask, :] * 9.0
            prices = ctx.prices.copy()
            pmask = (prices.index.get_level_values("ticker") == ticker) & (
                prices.index.get_level_values("date") >= day
            )
            prices.loc[pmask, :] = prices.loc[pmask, :] * 4.0
            perturbed = compute_pre_event_option_features(options, ctx.events, prices=prices)
            a = feats.loc[eid].drop(["ticker", "event_date", "available_at"]).astype(float)
            b = perturbed.loc[eid].drop(["ticker", "event_date", "available_at"]).astype(float)
            av, bv = a.to_numpy(), b.to_numpy()
            assert np.array_equal(np.isnan(av), np.isnan(bv)), f"patrón NaN cambió en {eid}"
            assert np.array_equal(np.nan_to_num(av), np.nan_to_num(bv)), (
                f"features de {eid} usan datos >= T"
            )

    def test_gancho_de_preevent(self, leaky: SyntheticMarket, ctx: EventContext) -> None:
        """`preevent` invoca `compute_pre_event_option_features(options, events)`."""
        block = compute_pre_event_option_features(leaky.options_daily(), leaky.events())
        assert block.index.name == "event_id"
        for col in OPTION_FEATURE_COLUMNS:
            assert col in block.columns
        # Sin precios, las features que los necesitan degradan a NaN documentado.
        assert block["option_stock_magnitude"].isna().all()
        assert block["move_ratio_hist"].isna().all()
        assert block["vol_spread"].notna().sum() > 100

        full = PreEventFeatures().compute(ctx)
        assert "options" not in full.attrs["missing_sources"]
        for col in OPTION_FEATURE_COLUMNS:
            assert full[col].notna().mean() > 0.9, f"{col} no llega desde options_signals"

    def test_fallos_explicitos(self, ctx: EventContext) -> None:
        engine = OptionsPreEventFeatures()
        with pytest.raises(DataQualityError):
            engine.compute(SimpleNamespace(events=ctx.events, options=None))
        with pytest.raises(DataQualityError):
            engine.compute(SimpleNamespace(events=None, options=ctx.options))
        with pytest.raises(DataQualityError):
            engine.compute_from_frames(pd.DataFrame(), ctx.events)
        sin_columnas = pd.DataFrame(
            {"nada": [1.0]},
            index=pd.MultiIndex.from_tuples(
                [(pd.Timestamp("2021-01-04"), "AAA")], names=["date", "ticker"]
            ),
        )
        with pytest.raises(DataQualityError):
            engine.compute_from_frames(sin_columnas, ctx.events)

    def test_eventos_fuera_del_panel_lanzan(self, ctx: EventContext) -> None:
        """Ningún evento calculable → InsufficientHistory, no una tabla de NaN."""
        eventos = pd.DataFrame(
            {
                "event_id": ["X:1", "X:2"],
                "ticker": ["NOEXISTE", "NOEXISTE"],
                "event_date": [pd.Timestamp("1999-01-04"), pd.Timestamp("1999-04-05")],
            }
        )
        with pytest.raises(InsufficientHistory):
            OptionsPreEventFeatures().compute_from_frames(ctx.options, eventos)


# ===========================================================================
# 10. Humo sobre los fixtures reales LEAN (Apache 2.0)
# ===========================================================================


_LEAN_ZIP = LEAN_DIR / "aapl_20140606_quote_american.zip"
_LEAN_PAT = re.compile(
    r"^(\d{8})_(\w+)_minute_quote_\w+_(call|put)_(\d+)_(\d{8})\.csv$"
)


def _parse_lean_chain(zip_path: Path) -> pd.DataFrame | None:
    """Cadena EOD desde un zip LEAN de quotes por minuto.

    Formato LEAN: un CSV por contrato con filas ``ms, bid OHLC, bid_size,
    ask OHLC, ask_size`` y precios en diezmilésimas de dólar; el strike va en
    el nombre del fichero multiplicado por 10 000. Se toma la última fila del
    día con ambos lados cotizados. Devuelve None si el formato no se reconoce.
    """
    rows: list[dict[str, object]] = []
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                m = _LEAN_PAT.match(name)
                if m is None:
                    continue
                day, sym, right, strike_raw, expiry = m.groups()
                quote = None
                for line in reversed(zf.read(name).strip().splitlines()):
                    parts = line.decode().split(",")
                    if len(parts) >= 11 and parts[4] and parts[9]:
                        quote = (float(parts[4]) / 10000.0, float(parts[9]) / 10000.0)
                        break
                if quote is None:
                    continue
                rows.append(
                    {
                        "as_of": pd.Timestamp(day),
                        "ticker": sym.upper(),
                        "expiry": pd.Timestamp(expiry),
                        "right": "C" if right == "call" else "P",
                        "strike": int(strike_raw) / 10000.0,
                        "bid": quote[0],
                        "ask": quote[1],
                        "days_to_expiry": (pd.Timestamp(expiry) - pd.Timestamp(day)).days,
                    }
                )
    except (zipfile.BadZipFile, ValueError, UnicodeDecodeError):
        return None
    if not rows:
        return None
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def lean_chain() -> pd.DataFrame:
    if not _LEAN_ZIP.exists():
        pytest.skip("fixture LEAN no disponible")
    chain = _parse_lean_chain(_LEAN_ZIP)
    if chain is None or len(chain) < 1000:
        pytest.skip("el formato del fixture LEAN no se reconoce")
    return chain


@pytest.mark.skipif(not _LEAN_ZIP.exists(), reason="fixture LEAN no disponible")
class TestLeanSmoke:
    def test_forward_y_niveles_de_iv_realistas(self, lean_chain: pd.DataFrame) -> None:
        """AAPL cerró a 645,57 $ el 2014-06-06: el forward implícito debe salir ahí."""
        agg = aggregate_chain_daily(lean_chain)
        row = agg.iloc[0]
        assert 600.0 < row["forward_front"] < 700.0
        assert 0.97 < row["discount_front"] < 1.03
        assert 0.10 < row["iv_atm_front"] < 0.60
        assert row["vol_spread_pairs"] >= 10
        assert abs(row["vol_spread"]) < 0.05
        assert abs(row["iv_skew_25delta"]) < 0.15
        assert 0.005 < row["straddle_frac"] < 0.15
        assert row["quality_drop_fraction"] < QualityFilterConfig().max_drop_fraction

    def test_sigma_evento_con_la_fecha_real_de_resultados(
        self, lean_chain: pd.DataFrame
    ) -> None:
        """AAPL anunció el 2014-07-22 AMC (negociable el 23): las weeklies del
        18 y el 25 de julio rodean el anuncio y habilitan el caso B del §7.2."""
        events = pd.DataFrame(
            {
                "event_id": ["AAPL:2014Q3:2014-06-28"],
                "ticker": ["AAPL"],
                "event_date": [pd.Timestamp("2014-07-23")],
            }
        )
        agg = aggregate_chain_daily(lean_chain, events=events)
        row = agg.iloc[0]
        assert 0.02 < row["sigma_event"] < 0.12
        assert 0.10 < row["sigma_diffusive"] < 0.50
        assert 0.0 < row["iv_crush_expected"] < 0.6
        assert 0.01 < row["expected_abs_move"] < 0.10
        assert 0.0 < row["event_share"] < 1.0
