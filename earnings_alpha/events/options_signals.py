"""Señales del mercado de opciones en la ventana pre-evento (contrato §3.5).

NOTA LEGAL Y METODOLÓGICA OBLIGATORIA (contrato §3.5)
-----------------------------------------------------
Todas las señales de este módulo derivan de datos **públicos**: cadenas de
opciones cotizadas (bid/ask, volumen, open interest) difundidas por OPRA/OCC y
resúmenes diarios de superficie de volatilidad. El objetivo es detectar la
**huella estadística** que la negociación informada deja en precios y flujo de
opciones observables por cualquier participante, no acceder a información
material no pública. Ninguna función requiere ni admite información privilegiada.

Qué hay aquí (siguiendo `docs/research/options_signals.md`, citado como *el
informe de opciones*)
---------------------------------------------------------------------------
* **Aritmética de valoración**: Black-76 sobre el forward y su inversión robusta
  (`black76_price`, `implied_vol_black76`). Nunca Black-Scholes sobre el spot
  con dividendo estimado: el informe §2.1-§2.4 y §3.4 cuantifica los sesgos de
  dividendo, comisión de préstamo y desfase de captura que eso introduce.
* **Forward implícito por regresión de paridad put-call** (`implied_forward`,
  informe §2.1): la corrección que elimina de raíz el artefacto de préstamo de
  Muravyev-Pearson-Pollet (2025) y el desfase de captura de Wallmeier (2024), y
  que regala la comisión de préstamo implícita como señal.
* **Filtros de calidad de cadena** (`apply_quality_filters`, informe §2.3):
  bid cero, cruces bid/ask, horquilla excesiva, prima ínfima, cotas de no
  arbitraje, DTE fuera de rango. Cada filtro contabiliza sus descartes y un
  ticker-día que pierde más del 40 % de la cadena degrada a NaN, no a un número.
* **Interpolador de sonrisa** (`SmileSlice`): IV por lado OTM en log-moneyness,
  ATM en el forward y localización del 25-delta por **punto fijo delta-strike**
  (`strike_at_delta`, informe §4.2: el atajo del "strike más cercano al 25Δ"
  produce hasta 1,9 puntos de error con mallas de 5 $).
* **Señales**: `vol_spread` de Cremers-Weinbaum ponderado por open interest
  (§3), skew 25-delta y Xing-Zhang-Zhao con su curvatura (§4), ratios put/call
  por volumen y OI (§5), O/S de Roll-Schwartz-Subrahmanyam (§6), descomposición
  de varianza del evento con dos vencimientos (§7), movimiento esperado desde el
  straddle ATM por parábola de tres strikes (§8) y acumulación direccional de
  open interest (§9).
* **`OptionsPreEventFeatures.compute(ctx)`**: tabla por ``event_id`` con las
  columnas del contrato §3.5 y las ampliaciones del informe §12, calculada
  sobre un panel diario ``(date, ticker)`` o sobre cadenas por contrato
  (agregadas primero con `aggregate_chain_daily`).

Convención de signos (contrato §3.4: valor mayor = más alcista NO se impone a
las features individuales; se documenta el signo esperado de cada una)
---------------------------------------------------------------------------
==============================  =========  =====================================
Feature                         Signo      Referencia
==============================  =========  =====================================
``vol_spread``                  positivo   Cremers y Weinbaum (2010)
``iv_skew_25delta``             negativo   put 25Δ − call 25Δ (= −risk reversal);
                                           convención XZZ: más skew = bajista
``put_call_volume_ratio``       negativo   Pan y Poteshman (2006), degradada EOD
``option_stock_direction``      ya orientada  Johnson y So (2012): O/S alto es
                                           BAJISTA; el signo negativo va dentro
``oi_buildup_calls``            positivo   Fodor, Krieger y Doran (2011)
``oi_buildup_puts``             negativo   ídem, efecto menos marcado
``iv_term_slope``               ambiguo    predice magnitud, no signo (§7.4)
``event_iv`` y derivadas        ambiguo    magnitud (Dubinsky-Johannes)
==============================  =========  =====================================

**Sobre `iv_skew_25delta`.** El informe §4.2 define el risk reversal
``rr25 = IV(call 25Δ) − IV(put 25Δ)`` (positivo = alcista). Aquí la columna del
contrato se almacena con la convención opuesta y equivalente
``iv_skew_25delta = IV(put 25Δ) − IV(call 25Δ) = −rr25`` (mayor = puts más
caras = bajista) por dos razones deliberadas: (a) es la convención del panel
sintético y de la mayoría de proveedores ("25-delta skew"), y (b) el consumidor
aguas arriba (`events.preevent.DEFAULT_DIRECTIONAL_WEIGHTS`) la pondera con
peso −1 citando a Xing-Zhang-Zhao. Cambiar aquí el signo rompería el score
compuesto en silencio. El risk reversal del informe es simplemente el negativo.

Invariante point-in-time
------------------------
Ninguna feature usa datos de la sesión T (fecha negociable del anuncio) ni
posteriores. Además (informe §9.4 y §13):

* cotizaciones, IV y volumen de la sesión ``t`` son conocibles al cierre de
  ``t``: la última sesión utilizable es **T−1**;
* el **open interest** de la sesión ``t`` lo publica la OCC la mañana de
  ``t+1``; como el ``available_at`` de la tabla es la medianoche de T, la
  última sesión de OI utilizable es **T−2** (una sesión de margen adicional,
  conservadora y documentada).

Limitaciones declaradas (informe §10.4)
---------------------------------------
Con datos de fin de día no hay volumen firmado ni desglose apertura/cierre:
las señales de volumen (put/call, O/S direccional, `oi_buildup`) son versiones
degradadas de Pan-Poteshman y Ge-Lin-Pearson y deben ponderarse poco. El filtro
F10 del informe (±2 días alrededor de splits y dividendos especiales) requiere
un calendario de acciones corporativas que no llega a este módulo: se documenta
y queda a cargo de la capa de datos.

Referencias principales
-----------------------
- Cremers, M. y Weinbaum, D. (2010). "Deviations from Put-Call Parity and Stock
  Return Predictability". *JFQA* 45(2), 335-367.
- Xing, Y., Zhang, X. y Zhao, R. (2010). "What Does the Individual Option
  Volatility Smirk Tell Us About Future Equity Returns?". *JFQA* 45(3), 641-662.
- Atilgan, Y. (2014). "Volatility spreads and earnings announcement returns".
  *Journal of Banking & Finance* 38, 205-215.
- Pan, J. y Poteshman, A. M. (2006). "The Information in Option Volume for
  Future Stock Prices". *RFS* 19(3), 871-908.
- Roll, R., Schwartz, E. y Subrahmanyam, A. (2010). "O/S: The relative trading
  activity in options and stock". *JFE* 96(1), 1-17.
- Johnson, T. L. y So, E. C. (2012). "The option to stock volume ratio and
  future returns". *JFE* 106(2), 262-286. **Signo predictivo NEGATIVO.**
- Ge, L., Lin, T.-C. y Pearson, N. D. (2016). *JFE* 120(3), 601-622.
- Fodor, A., Krieger, K. y Doran, J. (2011). *FMPM* 25(3), 265-280.
- Dubinsky, A. y Johannes, M. (2006); Dubinsky, Johannes, Kaeck y Seeger
  (2019). "Option Pricing of Earnings Announcement Risks". *RFS* 32(2), 646-687.
- Barth, M. E. y So, E. C. (2014). *The Accounting Review* 89(5), 1579-1607.
- Muravyev, D., Pearson, N. D. y Pollet, J. M. (2025). *JFE* 172, 104047.
- Goncalves-Pinto, L. et al. (2020). *Management Science* 66(9), 3903-3926.
- Wallmeier, M. (2024). *Journal of Futures Markets* 44(5), 854-875.
- Amin, K. I. y Lee, C. M. C. (1997). *Contemporary Accounting Research* 14(2).
"""

from __future__ import annotations

import itertools
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final, Literal

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import ndtr

from earnings_alpha.errors import DataQualityError, InsufficientHistory
from earnings_alpha.events import flow

__all__ = [
    "ABS_MOVE_FACTOR",
    "OPTION_EVENT_FEATURES",
    "STRADDLE_TO_ONE_SIGMA",
    "EventVolDecomposition",
    "FilterReport",
    "ImpliedForward",
    "OptionsPreEventFeatures",
    "QualityFilterConfig",
    "SmileSlice",
    "aggregate_chain_daily",
    "apply_quality_filters",
    "atm_straddle",
    "black76_price",
    "compute_pre_event_option_features",
    "event_variance_decomposition",
    "expected_iv_crush",
    "implied_forward",
    "implied_vol_black76",
    "oi_buildup_delta_weighted",
    "option_stock_ratio",
    "skew_25delta",
    "skew_xzz",
    "strike_at_delta",
    "vol_spread_cw",
]


_DAYS_PER_YEAR: Final[float] = 365.0
_PHI0: Final[float] = 1.0 / math.sqrt(2.0 * math.pi)

ABS_MOVE_FACTOR: Final[float] = math.sqrt(2.0 / math.pi)
"""``E^Q|S_T − F| ≈ 0,79788·F·σ√T`` bajo lognormal (informe §8.1, exacto al 0,05 %)."""

STRADDLE_TO_ONE_SIGMA: Final[float] = math.sqrt(math.pi / 2.0)
"""Factor straddle→sigma = 1,2533 (informe §8.2). La regla retail del
"0,85 × straddle" es incorrecta: subestima la sigma un 32 %."""

OPTION_EVENT_FEATURES: Final[tuple[str, ...]] = (
    # contrato §3.5
    "vol_spread",
    "iv_skew_25delta",
    "put_call_volume_ratio",
    "oi_buildup_calls",
    "oi_buildup_puts",
    "iv_term_slope",
    # ampliaciones del informe §12
    "d_vol_spread_5",
    "d_vol_spread_20",
    "cavs_20",
    "d_iv_skew_5",
    "iv_skew_xzz",
    "iv_curvature",
    "oi_buildup_net",
    "open_ratio_5d",
    "iv_term_slope_raw",
    "option_stock_level",
    "option_stock_magnitude",
    "option_stock_direction",
    "event_iv",
    "iv_crush_expected",
    "event_share",
    "expected_abs_move",
    "event_vol_surprise",
    "move_ratio_hist",
)
"""Columnas de features que produce `OptionsPreEventFeatures.compute` (además de
los metadatos ``ticker``, ``event_date`` y ``available_at``)."""


# ---------------------------------------------------------------------------
# Black-76: precio e inversión
# ---------------------------------------------------------------------------


def black76_price(
    forward: float | np.ndarray,
    strike: float | np.ndarray,
    tau: float | np.ndarray,
    sigma: float | np.ndarray,
    kind: int | np.ndarray,
    discount: float | np.ndarray = 1.0,
) -> float | np.ndarray:
    """Precio Black (1976) sobre el forward. ``kind`` = +1 call, −1 put.

    Se valora sobre el **forward** y no sobre el spot deliberadamente: el
    forward implícito de la cadena (§2.1 del informe de opciones) ya incorpora
    dividendos, comisión de préstamo y el instante de captura, que son las tres
    fuentes de sesgo sistemático de una IV calculada sobre el spot (informe
    §2.4, §3.4; Wallmeier 2024; Muravyev, Pearson y Pollet 2025).
    """
    f = np.asarray(forward, dtype=float)
    k = np.asarray(strike, dtype=float)
    t = np.asarray(tau, dtype=float)
    s = np.asarray(sigma, dtype=float)
    cp = np.asarray(kind, dtype=float)
    df = np.asarray(discount, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = s * np.sqrt(t)
        d1 = (np.log(f / k) + 0.5 * v * v) / v
        d2 = d1 - v
        price = df * cp * (f * ndtr(cp * d1) - k * ndtr(cp * d2))
    intrinsic = df * np.maximum(cp * (f - k), 0.0)
    out = np.where((t <= 0.0) | (v <= 0.0), intrinsic, price)
    return float(out) if out.ndim == 0 else out


def implied_vol_black76(
    price: float,
    forward: float,
    strike: float,
    tau: float,
    kind: int,
    discount: float = 1.0,
    bounds: tuple[float, float] = (1e-4, 5.0),
) -> float:
    """Volatilidad implícita Black-76 por Brent; **NaN si no converge** (F9).

    La política del informe §2.3 (filtro F9) es explícita: si el precio queda
    fuera de las cotas de no arbitraje o el solver no puede acotar la raíz en
    ``[0,01 %, 500 %]``, el resultado es NaN, jamás un valor por defecto — un
    default silencioso fabricaría IVs correlacionadas con la iliquidez.
    """
    if not (
        np.isfinite(price) and np.isfinite(forward) and forward > 0.0
        and strike > 0.0 and tau > 0.0 and discount > 0.0
    ):
        return float("nan")
    lo, hi = bounds
    intrinsic = discount * max(kind * (forward - strike), 0.0)
    upper = discount * (forward if kind > 0 else strike)
    if not intrinsic < price < upper:
        return float("nan")
    f_lo = black76_price(forward, strike, tau, lo, kind, discount) - price
    f_hi = black76_price(forward, strike, tau, hi, kind, discount) - price
    if f_lo * f_hi > 0.0:
        return float("nan")
    try:
        return float(
            brentq(
                lambda s: black76_price(forward, strike, tau, s, kind, discount) - price,
                lo,
                hi,
                xtol=1e-10,
            )
        )
    except (ValueError, RuntimeError):  # pragma: no cover - brentq ya acotado
        return float("nan")


# ---------------------------------------------------------------------------
# Filtros de calidad de la cadena (informe §2.3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class QualityFilterConfig:
    """Umbrales de los filtros F1..F9 del informe de opciones §2.3.

    Son obligatorios, no opcionales: una IV calculada sobre un bid de cero o
    una horquilla del 200 % no es una señal ruidosa, es un sesgo sistemático
    correlacionado con la iliquidez, que es a su vez un factor de riesgo.

    * ``F1 bid > 0`` — un bid de 0 no es un precio, es la ausencia de un precio.
    * ``F2 ask > bid`` — los cruces son datos corruptos.
    * ``F3 (ask−bid)/mid <= max_rel_spread`` — el ruido de horquilla en IV es
      ``(s/2)/vega`` (informe §3.6): con horquillas relativas > 50 % el punto
      medio no informa.
    * ``F4 mid >= min_mid`` — por debajo de 5 centavos el tick domina el precio.
    * ``F5 cotas de no arbitraje sobre el forward`` — call: ``DF·max(F−K,0) <=
      mid <= DF·F``; put: ``DF·max(K−F,0) <= mid <= DF·K``.
    * ``F6 min_dte <= DTE <= max_dte``.
    * F7 (moneyness) y F8 (open interest) los aplica cada señal con su propio
      rango; F9 (convergencia de la IV) lo aplica `implied_vol_black76`.
    * F10 (±2 días de splits/dividendos especiales) requiere acciones
      corporativas que no llegan aquí: responsabilidad de la capa de datos,
      limitación documentada.
    """

    max_rel_spread: float = 0.50
    min_mid: float = 0.05
    min_dte: int = 1
    max_dte: int = 365
    max_drop_fraction: float = 0.40
    """Si los filtros eliminan más de esta fracción de la cadena del ticker-día,
    el ticker-día completo degrada a NaN (informe §2.3)."""


@dataclass(slots=True)
class FilterReport:
    """Contabilidad de descartes por filtro: cuántos contratos eliminó cada uno."""

    initial: int
    dropped: dict[str, int] = field(default_factory=dict)

    @property
    def remaining(self) -> int:
        return self.initial - sum(self.dropped.values())

    @property
    def dropped_fraction(self) -> float:
        if self.initial == 0:
            return 1.0
        return 1.0 - self.remaining / self.initial

    def usable(self, config: QualityFilterConfig) -> bool:
        """False si el ticker-día pierde demasiada cadena y debe degradar a NaN."""
        return self.remaining > 0 and self.dropped_fraction <= config.max_drop_fraction


def apply_quality_filters(
    chain: pd.DataFrame,
    *,
    forwards: Mapping[pd.Timestamp, ImpliedForward] | None = None,
    config: QualityFilterConfig | None = None,
) -> tuple[pd.DataFrame, FilterReport]:
    """Aplica los filtros F1..F6 en orden y devuelve ``(cadena limpia, informe)``.

    ``chain`` es la cadena de **un** ticker-día con columnas ``expiry, right,
    strike, bid, ask`` y, si se quiere F6, ``days_to_expiry``. ``forwards``
    (por vencimiento, de `implied_forward`) habilita el filtro F5 de no
    arbitraje. El informe registra cuántos contratos elimina cada filtro, en
    el orden de aplicación: los filtros posteriores solo cuentan supervivientes
    de los anteriores, exactamente como pide el informe §2.3.
    """
    cfg = config or QualityFilterConfig()
    _require(chain, ["expiry", "right", "strike", "bid", "ask"], "chain")
    report = FilterReport(initial=len(chain))
    alive = chain.copy()
    alive["mid"] = (alive["bid"].astype(float) + alive["ask"].astype(float)) / 2.0

    def _drop(name: str, bad: pd.Series) -> None:
        nonlocal alive
        n = int(bad.sum())
        report.dropped[name] = n
        if n:
            alive = alive.loc[~bad]

    _drop("F1_bid_cero", ~(alive["bid"].astype(float) > 0.0))
    _drop("F2_cruce", ~(alive["ask"].astype(float) > alive["bid"].astype(float)))
    rel = (alive["ask"].astype(float) - alive["bid"].astype(float)) / alive["mid"].where(
        alive["mid"] > 0
    )
    _drop("F3_horquilla", ~(rel <= cfg.max_rel_spread))
    _drop("F4_prima_minima", ~(alive["mid"] >= cfg.min_mid))

    if forwards:
        cp = np.where(_right_sign(alive["right"]) > 0, 1.0, -1.0)
        f_arr = alive["expiry"].map(
            {e: fw.forward for e, fw in forwards.items()}
        ).to_numpy(dtype=float)
        df_arr = alive["expiry"].map(
            {e: fw.discount for e, fw in forwards.items()}
        ).to_numpy(dtype=float)
        k_arr = alive["strike"].to_numpy(dtype=float)
        mid = alive["mid"].to_numpy(dtype=float)
        with np.errstate(invalid="ignore"):
            lower = df_arr * np.maximum(cp * (f_arr - k_arr), 0.0)
            upper = np.where(cp > 0, df_arr * f_arr, df_arr * k_arr)
            bad_arb = np.isfinite(f_arr) & ~((mid >= lower) & (mid <= upper))
        _drop("F5_no_arbitraje", pd.Series(bad_arb, index=alive.index))
    else:
        report.dropped["F5_no_arbitraje"] = 0

    if "days_to_expiry" in alive.columns:
        dte = alive["days_to_expiry"].astype(float)
        _drop("F6_dte", ~((dte >= cfg.min_dte) & (dte <= cfg.max_dte)))
    else:
        report.dropped["F6_dte"] = 0
    return alive, report


# ---------------------------------------------------------------------------
# Forward implícito por regresión de paridad put-call (informe §2.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ImpliedForward:
    """Resultado de la regresión de paridad de un vencimiento.

    ``C(K) − P(K) = DF·F − DF·K`` es una recta en K: la ordenada da ``DF·F`` y
    la pendiente ``−DF``. `implied_rate` es ``−ln(DF)/τ`` y `implied_borrow`
    la comisión de préstamo implícita ``−ln(F·DF/S)/τ`` (identidad exacta:
    con ``F = S·e^{(r−f)τ}`` y ``DF = e^{−rτ}``, ``ln(F·DF/S) = −f·τ``). Es la
    señal gratuita del informe §3.4: comisión alta → retorno bajo, sin el
    retardo quincenal de FINRA.
    """

    expiry: pd.Timestamp
    tau: float
    forward: float
    discount: float
    n_pairs: int
    rmse: float
    implied_rate: float
    implied_borrow: float


def implied_forward(
    chain_expiry: pd.DataFrame,
    *,
    tau: float,
    spot: float | None = None,
    min_pairs: int = 4,
    band: float = 0.10,
) -> ImpliedForward | None:
    """Extrae ``(F, DF)`` de un vencimiento por regresión de paridad (informe §2.1).

    Procedimiento: (1) se casan call y put por strike con ``bid > 0`` y
    ``ask > bid`` en ambas patas; (2) el cruce de signo de ``C−P`` da un ancla
    ``F0`` sin necesitar el spot; (3) se restringe a ``|K/F0 − 1| <= band``
    (donde el valor temporal de ambas patas es máximo y el ejercicio anticipado
    americano es despreciable, informe §3.5), **ensanchando a 2·band si la
    malla de strikes es tan gruesa que no deja `min_pairs` pares** — decisión
    operativa documentada, no del paper; (4) mínimos cuadrados ponderados por
    ``1/(horquilla_call + horquilla_put)`` para que manden los strikes líquidos.

    Ventajas materiales (informe §2.1): elimina el sesgo de dividendo
    (Wallmeier 2024), el de comisión de préstamo (Muravyev-Pearson-Pollet 2025)
    y el desfase de captura opción-acción (§2.4), porque el forward se observa
    en las mismas cotizaciones que después se invierten.

    Devuelve None si no hay pares suficientes o la regresión produce un
    descuento fuera de ``(0,5, 1,02]``: el vencimiento no se usa (fallo
    explícito del consumidor, no un forward inventado).
    """
    if tau <= 0.0:
        return None
    sign = _right_sign(chain_expiry["right"])
    calls = chain_expiry.loc[sign > 0, ["strike", "bid", "ask"]]
    puts = chain_expiry.loc[sign < 0, ["strike", "bid", "ask"]]
    pairs = calls.merge(puts, on="strike", suffixes=("_c", "_p"))
    ok = (
        (pairs["bid_c"] > 0.0) & (pairs["bid_p"] > 0.0)
        & (pairs["ask_c"] > pairs["bid_c"]) & (pairs["ask_p"] > pairs["bid_p"])
    )
    pairs = pairs.loc[ok].sort_values("strike")
    if len(pairs) < min_pairs:
        return None

    k = pairs["strike"].to_numpy(dtype=float)
    mid_c = (pairs["bid_c"] + pairs["ask_c"]).to_numpy(dtype=float) / 2.0
    mid_p = (pairs["bid_p"] + pairs["ask_p"]).to_numpy(dtype=float) / 2.0
    diff = mid_c - mid_p

    # Ancla F0: el strike en que C−P cambia de signo (C−P es decreciente en K).
    below = np.where(diff > 0)[0]
    above = np.where(diff <= 0)[0]
    if len(below) and len(above):
        i, j = below[-1], above[0]
        if i < j and diff[i] != diff[j]:
            f0 = k[i] + (k[j] - k[i]) * diff[i] / (diff[i] - diff[j])
        else:
            f0 = k[int(np.argmin(np.abs(diff)))]
    else:
        f0 = k[int(np.argmin(np.abs(diff)))]

    for width in (band, 2.0 * band):
        keep = np.abs(k / f0 - 1.0) <= width
        if int(keep.sum()) >= min_pairs:
            break
    else:
        return None
    kk, dd = k[keep], diff[keep]
    spread = (
        (pairs["ask_c"] - pairs["bid_c"]) + (pairs["ask_p"] - pairs["bid_p"])
    ).to_numpy(dtype=float)[keep]
    w = 1.0 / np.maximum(spread, 1e-3)

    x = np.column_stack([np.ones(len(kk)), kk])
    wx = x * w[:, None]
    try:
        coef, *_ = np.linalg.lstsq(wx, dd * w, rcond=None)
    except np.linalg.LinAlgError:  # pragma: no cover - malla degenerada
        return None
    intercept, slope = float(coef[0]), float(coef[1])
    discount = -slope
    if not (0.5 < discount <= 1.02) or intercept <= 0.0:
        return None
    fwd = intercept / discount
    if not np.isfinite(fwd) or fwd <= 0.0 or abs(math.log(fwd / f0)) > 0.25:
        return None
    resid = dd - (intercept + slope * kk)
    rmse = float(np.sqrt(np.mean(resid**2)))
    rate = -math.log(discount) / tau
    borrow = float("nan")
    if spot is not None and np.isfinite(spot) and spot > 0.0:
        borrow = -math.log(fwd * discount / spot) / tau
    return ImpliedForward(
        expiry=pd.Timestamp(chain_expiry["expiry"].iloc[0]),
        tau=float(tau),
        forward=float(fwd),
        discount=float(discount),
        n_pairs=int(keep.sum()),
        rmse=rmse,
        implied_rate=float(rate),
        implied_borrow=borrow,
    )


# ---------------------------------------------------------------------------
# Sonrisa por vencimiento e interpolación robusta
# ---------------------------------------------------------------------------


def strike_at_delta(
    grid_k: np.ndarray,
    grid_iv: np.ndarray,
    forward: float,
    tau: float,
    target_delta: float,
) -> float:
    """Strike cuyo delta-call Black-76 es ``target_delta`` (punto fijo, informe §4.2).

    El delta depende de la IV y la IV depende del strike: es un punto fijo que
    se resuelve con Brent sobre K interpolando la IV en log-moneyness a cada
    paso. La convención estándar: la put de delta −0,25 es el strike cuyo
    delta-call vale +0,75.

    El informe §4.2 cuantifica el coste de no hacerlo: con malla de 5 $, el
    atajo "strike listado más cercano al 25Δ" comete 1,9 puntos de vol de error
    en el risk reversal (más que la señal), mientras que el punto fijo comete
    0,02. Y el error del atajo no es aleatorio: depende del nivel de precio de
    la acción, que es persistente — ranking espurio garantizado.

    Devuelve NaN si el objetivo cae fuera de la malla observada (sin
    extrapolar: extrapolar una sonrisa es inventar datos).
    """
    if len(grid_k) < 3 or tau <= 0.0 or not 0.0 < target_delta < 1.0:
        return float("nan")

    def _delta_at(k_strike: float) -> float:
        iv = float(np.interp(math.log(k_strike / forward), grid_k, grid_iv))
        v = iv * math.sqrt(tau)
        if v <= 0.0:
            return float("nan")
        d1 = (math.log(forward / k_strike) + 0.5 * v * v) / v
        return float(ndtr(d1))

    lo = forward * math.exp(grid_k[0])
    hi = forward * math.exp(grid_k[-1])
    d_lo, d_hi = _delta_at(lo), _delta_at(hi)
    # delta-call decrece con K: la raíz existe solo si el objetivo está entre ambos.
    if not (min(d_lo, d_hi) <= target_delta <= max(d_lo, d_hi)):
        return float("nan")
    try:
        return float(brentq(lambda kk: _delta_at(kk) - target_delta, lo, hi, xtol=1e-8))
    except (ValueError, RuntimeError):
        return float("nan")


@dataclass(slots=True)
class SmileSlice:
    """Sonrisa de un vencimiento: IVs propias invertidas sobre el forward implícito.

    ``k_call/iv_call`` y ``k_put/iv_put`` son las mallas por lado en
    log-moneyness ``k = ln(K/F)``, ya filtradas. La interpolación es lineal en
    ``k`` **sin extrapolación** (fuera de la malla → NaN): robusta y sin grados
    de libertad ocultos. Para medidas de un solo lado se usa la pata OTM, que
    es la líquida (puts para ``k<0``, calls para ``k>0``).
    """

    expiry: pd.Timestamp
    tau: float
    forward: float
    discount: float
    k_call: np.ndarray
    iv_call: np.ndarray
    k_put: np.ndarray
    iv_put: np.ndarray

    @property
    def dte(self) -> float:
        return self.tau * _DAYS_PER_YEAR

    def _interp(self, grid_k: np.ndarray, grid_iv: np.ndarray, k: float) -> float:
        if len(grid_k) < 2 or k < grid_k[0] or k > grid_k[-1]:
            return float("nan")
        return float(np.interp(k, grid_k, grid_iv))

    def iv_call_at(self, k: float) -> float:
        return self._interp(self.k_call, self.iv_call, k)

    def iv_put_at(self, k: float) -> float:
        return self._interp(self.k_put, self.iv_put, k)

    def atm_iv(self) -> float:
        """IV ATM en ``K = F``: media de las patas call y put interpoladas.

        Promediar ambas patas cancela la mitad del `vol_spread` y deja el nivel
        de la sonrisa, que es lo que necesitan la estructura temporal (§7) y el
        straddle (§8). Si solo una pata llega al dinero, se usa esa.
        """
        c, p = self.iv_call_at(0.0), self.iv_put_at(0.0)
        if np.isfinite(c) and np.isfinite(p):
            return 0.5 * (c + p)
        return c if np.isfinite(c) else p

    def iv_at_call_delta(self, target_delta: float) -> float:
        """IV en el strike de delta-call objetivo, con la pata OTM correspondiente.

        Para ``Δ_call < 0,5`` el strike cae por encima del forward → pata call;
        para ``Δ_call > 0,5`` cae por debajo → pata put (la put OTM equivalente).
        El strike se localiza por punto fijo sobre la malla OTM combinada.
        """
        grid_k, grid_iv = self._otm_grid()
        if len(grid_k) < 3:
            return float("nan")
        k_star = strike_at_delta(grid_k, grid_iv, self.forward, self.tau, target_delta)
        if not np.isfinite(k_star):
            return float("nan")
        k_log = math.log(k_star / self.forward)
        return self.iv_call_at(k_log) if k_log >= 0.0 else self.iv_put_at(k_log)

    def _otm_grid(self) -> tuple[np.ndarray, np.ndarray]:
        """Malla combinada OTM: puts en ``k<0``, calls en ``k>=0``."""
        ks: list[float] = []
        ivs: list[float] = []
        for k, iv in zip(self.k_put, self.iv_put, strict=True):
            if k < 0.0:
                ks.append(float(k))
                ivs.append(float(iv))
        for k, iv in zip(self.k_call, self.iv_call, strict=True):
            if k >= 0.0:
                ks.append(float(k))
                ivs.append(float(iv))
        if not ks:
            return np.array([]), np.array([])
        order = np.argsort(ks)
        return np.asarray(ks)[order], np.asarray(ivs)[order]


def _build_smile(
    chain_expiry: pd.DataFrame, fwd: ImpliedForward
) -> SmileSlice | None:
    """Invierte las IV de un vencimiento filtrado y construye su `SmileSlice`."""
    sign = _right_sign(chain_expiry["right"])
    sides: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for cp in (1, -1):
        side = chain_expiry.loc[sign == cp]
        if len(side) == 0:
            sides[cp] = (np.array([]), np.array([]))
            continue
        strikes = side["strike"].to_numpy(dtype=float)
        mids = ((side["bid"] + side["ask"]) / 2.0).to_numpy(dtype=float)
        ivs = np.array(
            [
                implied_vol_black76(m, fwd.forward, k, fwd.tau, cp, fwd.discount)
                for k, m in zip(strikes, mids, strict=True)
            ]
        )
        good = np.isfinite(ivs)
        kk = np.log(strikes[good] / fwd.forward)
        order = np.argsort(kk)
        sides[cp] = (kk[order], ivs[good][order])
    if len(sides[1][0]) + len(sides[-1][0]) < 3:
        return None
    return SmileSlice(
        expiry=fwd.expiry,
        tau=fwd.tau,
        forward=fwd.forward,
        discount=fwd.discount,
        k_call=sides[1][0],
        iv_call=sides[1][1],
        k_put=sides[-1][0],
        iv_put=sides[-1][1],
    )


# ---------------------------------------------------------------------------
# Volatility spread de Cremers-Weinbaum (informe §3)
# ---------------------------------------------------------------------------


def vol_spread_cw(
    chain: pd.DataFrame,
    forwards: Mapping[pd.Timestamp, ImpliedForward],
    *,
    dte_range: tuple[float, float] = (20.0, 90.0),
    moneyness_band: tuple[float, float] = (0.90, 1.10),
    min_pairs: int = 3,
    weighting: Literal["oi_mean", "oi_min", "equal"] = "oi_mean",
) -> tuple[float, int]:
    """Volatility spread de Cremers y Weinbaum (2010): ``media ponderada por OI
    de (IV_call − IV_put)`` sobre pares casados por strike y vencimiento.

    Decisiones fijadas donde el paper no fija detalle (informe §3.1,
    **[propuesto, no del paper]**, pregunta abierta §15.1-15.2):

    * ponderación ``(OI_c + OI_p)/2`` por defecto (la reproducción más citada),
      con ``min(OI_c, OI_p)`` como variante de robustez — penaliza pares con
      una pata cascarón — y ``equal`` como degradación documentada cuando la
      cadena no trae open interest;
    * ``20 <= DTE <= 90``: el suelo de 20 controla el sesgo de no-sincronía,
      que escala como ``1/√T`` (informe §2.4: 10 pb de desfase = 1,8 puntos de
      vol a 7 días); el techo de 90 evita series ilíquidas;
    * moneyness en el **forward** ``0,90 <= K/F <= 1,10``: fuera de ahí el
      sesgo de ejercicio anticipado americano explota (informe §3.5: −6,6
      puntos de vol en K/S = 1,30);
    * mínimo de ``3`` pares válidos (informe §3.6: con horquillas de 0,05 $
      hacen falta ≥3 pares para sd < 0,2 puntos). Con menos → NaN, y el umbral
      **no debe relajarse** para ganar cobertura: metería ruido correlacionado
      con la iliquidez.

    Los pares con open interest nulo en ambas patas reciben peso cero (F8).
    Signo esperado: **positivo** (calls relativamente caras → retornos altos).
    Devuelve ``(vol_spread, n_pares)``.
    """
    lo_m, hi_m = moneyness_band
    lo_d, hi_d = dte_range
    diffs: list[float] = []
    weights: list[float] = []
    has_oi = "open_interest" in chain.columns
    for expiry, sub in chain.groupby("expiry", sort=True):
        fwd = forwards.get(pd.Timestamp(expiry))
        if fwd is None or not lo_d <= fwd.tau * _DAYS_PER_YEAR <= hi_d:
            continue
        sign = _right_sign(sub["right"])
        cols = ["strike", "bid", "ask"] + (["open_interest"] if has_oi else [])
        pairs = sub.loc[sign > 0, cols].merge(
            sub.loc[sign < 0, cols], on="strike", suffixes=("_c", "_p")
        )
        if len(pairs) == 0:
            continue
        keep = (pairs["strike"] / fwd.forward).between(lo_m, hi_m)
        for row in pairs.loc[keep].itertuples(index=False):
            iv_c = implied_vol_black76(
                (row.bid_c + row.ask_c) / 2.0, fwd.forward, row.strike, fwd.tau, 1, fwd.discount
            )
            iv_p = implied_vol_black76(
                (row.bid_p + row.ask_p) / 2.0, fwd.forward, row.strike, fwd.tau, -1, fwd.discount
            )
            if not (np.isfinite(iv_c) and np.isfinite(iv_p)):
                continue
            if weighting == "equal" or not has_oi:
                w = 1.0
            elif weighting == "oi_min":
                w = float(min(row.open_interest_c, row.open_interest_p))
            else:
                w = float(row.open_interest_c + row.open_interest_p) / 2.0
            if w <= 0.0:
                continue
            diffs.append(iv_c - iv_p)
            weights.append(w)
    if len(diffs) < min_pairs:
        return float("nan"), len(diffs)
    d = np.asarray(diffs)
    w = np.asarray(weights)
    return float(np.sum(d * w) / np.sum(w)), len(diffs)


# ---------------------------------------------------------------------------
# Skew: 25-delta y Xing-Zhang-Zhao (informe §4)
# ---------------------------------------------------------------------------


def skew_25delta(smile: SmileSlice) -> float:
    """Skew 25-delta: ``IV(put Δ=−0,25) − IV(call Δ=+0,25)`` = −risk reversal.

    Es la medida de mesa que pide el contrato (`iv_skew_25delta`), almacenada
    con la convención put-menos-call: **mayor = puts OTM más caras = bajista**
    (coherente con el peso −1 de `preevent.DEFAULT_DIRECTIONAL_WEIGHTS` y con
    el signo de Xing-Zhang-Zhao). El risk reversal del informe §4.2 es
    exactamente el negativo de este valor; en renta variable el resultado aquí
    es casi siempre **positivo** (smirk).

    Ambos strikes se localizan por punto fijo delta↔strike (`strike_at_delta`);
    la put de −0,25 es el strike de delta-call +0,75. Invariante al nivel de
    precio de la acción, al contrario que las medidas por moneyness fija.
    """
    iv_call_25 = smile.iv_at_call_delta(0.25)
    iv_put_25 = smile.iv_at_call_delta(0.75)
    if not (np.isfinite(iv_call_25) and np.isfinite(iv_put_25)):
        return float("nan")
    return float(iv_put_25 - iv_call_25)


def skew_xzz(
    smile: SmileSlice,
    spot: float,
    *,
    put_moneyness: tuple[float, float] = (0.80, 0.95),
    atm_moneyness: tuple[float, float] = (0.95, 1.05),
) -> float:
    """Smirk de Xing, Zhang y Zhao (2010): ``IV(put OTM) − IV(call ATM)``.

    Selección del paper (JFQA 45(3), 641-662): la put con ``K/S`` más próximo a
    0,95 dentro de la banda OTM [0,80, 0,95] y la call con ``K/S`` más próximo
    a 1,00 dentro de la banda ATM [0,95, 1,05], **ambas del mismo
    vencimiento**. Se seleccionan strikes listados, sin interpolar, que es lo
    que hace el paper.

    Signo del predictor: **negativo** (smirk pronunciado → retornos menores;
    quintil extremo ≈ −10,9 %/año ajustado, cifra de abstract ⚠ pendiente de
    verificación contra el texto completo). El matiz decisivo para el ángulo B
    es Van Buskirk (SSRN 1740513): el skew solo predice crashes **en ventanas
    de anuncio de resultados** — exactamente nuestra ventana.

    No es intercambiable con `skew_25delta` (informe §4.3): miden la misma
    pendiente con brazos de palanca distintos y su cociente depende de la
    curvatura, que crece justo antes de resultados; la diferencia entre ambas
    es la feature `iv_curvature`.
    """
    if not (np.isfinite(spot) and spot > 0.0):
        return float("nan")
    put_k = smile.forward * np.exp(smile.k_put)
    put_m = put_k / spot
    in_put = (put_m >= put_moneyness[0]) & (put_m <= put_moneyness[1])
    call_k = smile.forward * np.exp(smile.k_call)
    call_m = call_k / spot
    in_call = (call_m >= atm_moneyness[0]) & (call_m <= atm_moneyness[1])
    if not (in_put.any() and in_call.any()):
        return float("nan")
    i_put = int(np.argmin(np.abs(put_m[in_put] - put_moneyness[1])))
    i_call = int(np.argmin(np.abs(call_m[in_call] - 1.0)))
    return float(smile.iv_put[in_put][i_put] - smile.iv_call[in_call][i_call])


# ---------------------------------------------------------------------------
# Estructura temporal y volatilidad implícita del evento (informe §7)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EventVolDecomposition:
    """Descomposición ``IV(T)²·T = σ_d²·T + σ_E²·n_eventos`` (Dubinsky-Johannes)."""

    sigma_diffusive: float
    sigma_event: float

    @property
    def is_valid(self) -> bool:
        return bool(np.isfinite(self.sigma_diffusive) and np.isfinite(self.sigma_event))


def event_variance_decomposition(
    iv_front: float,
    tau_front: float,
    iv_back: float,
    tau_back: float,
    events_front: int,
    events_back: int,
) -> EventVolDecomposition:
    """Extrae ``(σ_d, σ_E)`` de dos vencimientos que rodean el anuncio (informe §7.2).

    Modelo de dos componentes (Dubinsky y Johannes 2006; Dubinsky, Johannes,
    Kaeck y Seeger 2019, RFS 32(2)): difusión de volatilidad ``σ_d`` más un
    salto en el anuncio con desviación ``σ_E`` riesgo-neutral, de modo que la
    varianza total es aditiva: ``IV(T)²·T = σ_d²·T + σ_E²·1{anuncio < T}``.

    **Caso A** — ambos vencimientos posteriores al anuncio
    (``events_front == events_back == 1``); restando, ``σ_E²`` se cancela::

        σ_d² = (IV₂²·T₂ − IV₁²·T₁) / (T₂ − T₁)
        σ_E² = T₁·T₂·(IV₁² − IV₂²) / (T₂ − T₁)

    **Caso B** — el corto vence ANTES del anuncio y el largo después
    (``events_front == 0, events_back == 1``), posible con weeklies y más
    limpio porque no supone ``σ_d`` plana::

        σ_d = IV₁          σ_E² = T₂·(IV₂² − IV₁²)

    Trampas que este código respeta (informe §7.2):

    * ``T`` en **años naturales**, consistente en ambas ecuaciones;
    * los vencimientos deben contener un número de anuncios compatible: si el
      lejano contiene dos trimestres (``events_back >= 2``) la resta da basura
      y el resultado es **NaN** — hay que comprobarlo con el calendario de
      eventos, no suponerlo (es lo primero que hace ORATS);
    * si la estructura temporal está invertida por razones ajenas al evento y
      ``σ_d²`` sale negativa, se devuelve **NaN, nunca se trunca a cero**:
      truncar crearía un suelo artificial que se activa justo en días de estrés.
    """
    nan = EventVolDecomposition(float("nan"), float("nan"))
    if not (
        np.isfinite(iv_front) and np.isfinite(iv_back)
        and 0.0 < tau_front < tau_back and iv_front > 0.0 and iv_back > 0.0
    ):
        return nan
    if (events_front, events_back) == (1, 1):
        var_d = (iv_back**2 * tau_back - iv_front**2 * tau_front) / (tau_back - tau_front)
        var_e = tau_front * tau_back * (iv_front**2 - iv_back**2) / (tau_back - tau_front)
    elif (events_front, events_back) == (0, 1):
        var_d = iv_front**2
        var_e = tau_back * (iv_back**2 - iv_front**2)
    else:
        return nan
    if var_d <= 0.0 or var_e < 0.0:
        return nan
    return EventVolDecomposition(math.sqrt(var_d), math.sqrt(var_e))


def expected_iv_crush(
    sigma_diffusive: float, sigma_event: float, iv_front: float, tau_front: float
) -> float:
    """Caída relativa prevista de la IV frontal tras el anuncio (informe §7.3).

    Tras el anuncio la varianza restante es solo difusiva, luego la IV cae de
    ``IV₁`` a ``σ_d``::

        crush = 1 − σ_d/IV₁ = 1 − √(1 − σ_E²/(IV₁²·T₁))

    Con ``σ_d = 30 %`` y ``σ_E = 5 %`` reproduce el rango observado (36 % a
    7 días, 70 % a 1 día) sin calibrar nada. **No es una señal direccional: es
    un componente del coste** para cualquier posición en opciones mantenida a
    través del evento, y un filtro de calidad (crush previsto > 90 % o < 0 →
    cadena corrupta).
    """
    if not (
        np.isfinite(sigma_diffusive) and np.isfinite(iv_front)
        and iv_front > 0.0 and tau_front > 0.0
    ):
        return float("nan")
    crush = 1.0 - sigma_diffusive / iv_front
    if not np.isfinite(sigma_event):
        return float(crush)
    # Las dos expresiones del docstring son la misma identidad algebraica: si
    # los argumentos no la satisfacen es que (σ_d, σ_E, IV₁) no salen de la
    # misma descomposición aditiva, y el resultado honesto es NaN.
    arg = 1.0 - sigma_event**2 / (iv_front**2 * tau_front)
    if arg < 0.0 or abs(crush - (1.0 - math.sqrt(arg))) > 1e-6:
        return float("nan")
    return float(crush)


# ---------------------------------------------------------------------------
# Straddle ATM y movimiento esperado (informe §8)
# ---------------------------------------------------------------------------


def atm_straddle(
    chain_expiry: pd.DataFrame, fwd: ImpliedForward
) -> float:
    """Precio del straddle en ``K = F`` por **parábola de tres strikes** (informe §8.3).

    Identidad exacta (§8.1): ``straddle(K=F) = DF·E^Q|S_T − F|`` — el straddle
    ATM *es* el valor actual del movimiento absoluto esperado riesgo-neutral,
    sin aproximación ni supuesto de modelo.

    ``K = F`` no está listado y la intuición habitual falla: el straddle tiene
    un **mínimo** en ``K ≈ F``, luego la interpolación lineal entre dos strikes
    **siempre sobreestima** (+0,5 % con mallas de 5 $), mientras que la
    parábola por los tres strikes más cercanos es esencialmente exacta
    (−0,01 %). Con menos de tres pares casados se degrada al strike más
    cercano, que es el segundo mejor método (−0,14 %), nunca a la recta.
    """
    sign = _right_sign(chain_expiry["right"])
    pairs = chain_expiry.loc[sign > 0, ["strike", "bid", "ask"]].merge(
        chain_expiry.loc[sign < 0, ["strike", "bid", "ask"]],
        on="strike",
        suffixes=("_c", "_p"),
    )
    if len(pairs) == 0:
        return float("nan")
    pairs = pairs.sort_values("strike")
    k = pairs["strike"].to_numpy(dtype=float)
    straddle = (
        (pairs["bid_c"] + pairs["ask_c"]) / 2.0 + (pairs["bid_p"] + pairs["ask_p"]) / 2.0
    ).to_numpy(dtype=float)
    order = np.argsort(np.abs(k - fwd.forward))
    if len(k) >= 3:
        idx = np.sort(order[:3])
        k3, s3 = k[idx], straddle[idx]
        if len(np.unique(k3)) == 3:
            coefs = np.polyfit(k3 - fwd.forward, s3, 2)
            value = float(np.polyval(coefs, 0.0))
            if np.isfinite(value) and value > 0.0:
                return value
    return float(straddle[order[0]])


# ---------------------------------------------------------------------------
# O/S y open interest (informe §6 y §9)
# ---------------------------------------------------------------------------


def option_stock_ratio(
    option_volume_contracts: float | np.ndarray | pd.Series,
    stock_volume_shares: float | np.ndarray | pd.Series,
    *,
    multiplier: float = 100.0,
) -> float | np.ndarray | pd.Series:
    """O/S de Roll, Schwartz y Subrahmanyam (2010): actividad relativa opciones/acción.

    ``O/S = contratos_de_opciones · 100 / acciones_negociadas``: el × 100
    convierte contratos a acciones equivalentes y **se rompe con contratos
    ajustados por splits** (multiplicador ≠ 100; filtro F10 del informe §2.3, a
    cargo de la capa de datos). El volumen de la acción debe ser el
    **consolidado** (todas las plazas, incluido off-exchange): usar solo la
    plaza primaria infla el ratio de forma correlacionada con la cuota
    off-exchange, que es otra feature de este proyecto.

    **ADVERTENCIA DE SIGNO — el signo predictivo direccional es NEGATIVO, no
    positivo.** Johnson y So (2012, JFE 106(2), 262-286) documentan que el
    decil de O/S más BAJO supera al más ALTO en ≈0,34 % semanal ⚠: un pico de
    O/S es en promedio **bajista**, al contrario de la intuición retail del
    "unusual options activity". Roll, Schwartz y Subrahmanyam (2010) coinciden
    en el signo (O/S alto → retornos anormales post-anuncio menores). El
    mecanismo está en disputa — Johnson-So proponen costes de venta en corto,
    Ge, Lin y Pearson (2016) muestran que lo informativo son las compras de
    calls de apertura (apalancamiento), y Muravyev-Pearson-Pollet (2025)
    añaden el artefacto de comisión de préstamo — así que la dirección debe
    ponderarse poco y vigilarse. Lo robusto es el componente **no direccional**:
    O/S pre-evento alto → reacción de mayor MAGNITUD (predictor de ``|CAR|``).
    Por eso `OptionsPreEventFeatures` separa `option_stock_magnitude` (z, para
    el modelo de magnitud) de `option_stock_direction` (= −z, direccional con
    el signo de Johnson-So ya incorporado).
    """
    denom = np.maximum(np.asarray(stock_volume_shares, dtype=float), 1.0)
    out = np.asarray(option_volume_contracts, dtype=float) * multiplier / denom
    if isinstance(option_volume_contracts, pd.Series):
        return pd.Series(out, index=option_volume_contracts.index)
    return float(out) if out.ndim == 0 else out


def oi_buildup_delta_weighted(
    chain_now: pd.DataFrame,
    chain_prev: pd.DataFrame,
    forwards: Mapping[pd.Timestamp, ImpliedForward],
    *,
    spot: float,
    market_cap: float | None = None,
    otm_delta_band: tuple[float, float] = (0.03, 0.35),
    max_dte: float = 45.0,
) -> tuple[float, float]:
    """Acumulación direccional de OI ponderada por delta (informe §9.1), por lado.

    ``Σ_j |Δ_j| · (OI_j(t) − OI_j(t−k)) · 100 · S / normalizador``, con el delta
    Black-76 sobre el forward implícito y el normalizador la capitalización si
    se conoce (sin ella se normaliza por el nocional total de OI vigente, y se
    está midiendo mezcla de acumulación y tamaño: degradación documentada).

    Restringida a OTM de vencimiento corto (``0,03 < |Δ| < 0,35``,
    ``DTE <= 45``), donde la venta cubierta es menos plausible y el
    apalancamiento del informado es máximo (informe §9.3.a; Hilliard, Hilliard
    y Wu 2026). Aun así **ΔOI es neto y no está firmado** (§9.3): un aumento de
    OI de calls es igual de compatible con un informado comprando calls que con
    venta cubierta; sin el desglose Cboe Open-Close es una feature de baja
    fiabilidad y así debe ponderarse. Trampa PIT (§9.4): el OI de la sesión
    ``t`` se publica la mañana de ``t+1``.

    Referencias: Fodor, Krieger y Doran (2011) — el cambio del OI de calls
    predice con más claridad que el de puts; Ge, Lin y Pearson (2016).
    """
    prev_oi = {
        (pd.Timestamp(r.expiry), float(r.strike), _one_right(r.right)): float(r.open_interest)
        for r in chain_prev.itertuples(index=False)
    }
    build = {1: 0.0, -1: 0.0}
    notional = 0.0
    for row in chain_now.itertuples(index=False):
        fwd = forwards.get(pd.Timestamp(row.expiry))
        if fwd is None or fwd.tau * _DAYS_PER_YEAR > max_dte:
            continue
        cp = _one_right(row.right)
        iv = implied_vol_black76(
            (row.bid + row.ask) / 2.0, fwd.forward, float(row.strike), fwd.tau, cp, fwd.discount
        )
        if not np.isfinite(iv):
            continue
        v = iv * math.sqrt(fwd.tau)
        d1 = (math.log(fwd.forward / float(row.strike)) + 0.5 * v * v) / v
        delta = float(ndtr(d1)) if cp > 0 else float(ndtr(d1)) - 1.0
        if not (otm_delta_band[0] < abs(delta) < otm_delta_band[1]):
            continue
        key = (pd.Timestamp(row.expiry), float(row.strike), cp)
        d_oi = float(row.open_interest) - prev_oi.get(key, 0.0)
        build[cp] += abs(delta) * d_oi * 100.0 * spot
        notional += float(row.open_interest) * 100.0 * spot
    denom = market_cap if market_cap and market_cap > 0.0 else notional
    if not denom or denom <= 0.0:
        return float("nan"), float("nan")
    return build[1] / denom, build[-1] / denom


# ---------------------------------------------------------------------------
# Agregación diaria de cadenas por contrato
# ---------------------------------------------------------------------------


def aggregate_chain_daily(
    chain: pd.DataFrame,
    *,
    events: pd.DataFrame | None = None,
    config: QualityFilterConfig | None = None,
    min_dte_skew: float = 10.0,
    vol_spread_dte: tuple[float, float] = (20.0, 90.0),
    vol_spread_band: tuple[float, float] = (0.90, 1.10),
    weighting: Literal["oi_mean", "oi_min", "equal"] = "oi_mean",
) -> pd.DataFrame:
    """Reduce cadenas por contrato a un panel diario ``(date, ticker)`` de señales.

    Sigue la receta EOD del informe §10.2: por ticker-día, (1) forward
    implícito por vencimiento (§2.1) — los vencimientos sin regresión válida no
    se usan; (2) filtros F1..F6 con contabilidad de descartes — si cae más del
    40 % de la cadena, el ticker-día degrada a NaN (§2.3); (3) IVs propias con
    Black-76 sobre el forward, **nunca la IV del proveedor sin auditar**; (4)
    señales de precios (`vol_spread_cw`, `skew_25delta`, `skew_xzz`,
    curvatura); (5) estructura temporal con el calendario de eventos para
    contar anuncios por vencimiento (§7.2) y crush previsto (§7.3); (6)
    movimiento esperado desde el straddle ATM por parábola (§8.3); (7)
    agregados de volumen y OI por lado.

    ``events`` (con ``ticker, event_date``) permite contar cuántos anuncios
    caen antes de cada vencimiento; sin él se usa la columna
    ``covers_earnings`` si existe, y sin ninguna de las dos la descomposición
    del evento sale NaN (nunca se suponen los conteos).

    Columnas de salida por ``(date, ticker)``: forward y descuento frontales,
    ``implied_borrow_front`` (señal gratuita §3.4), ``vol_spread`` y su nº de
    pares, ``iv_skew_25delta``, ``iv_skew_xzz``, ``iv_curvature``,
    ``iv_atm_front/next`` con sus DTE, ``iv_term_slope_raw`` y de-eventizada
    (``iv_term_slope_ex``), ``sigma_event``, ``sigma_diffusive``,
    ``iv_crush_expected``, ``event_share``, ``straddle_frac``,
    ``one_sigma_move``, ``expected_abs_move`` (la del EVENTO, §8.2.c),
    volúmenes y OI por lado, y diagnósticos de calidad.
    """
    cfg = config or QualityFilterConfig()
    _require(chain, ["as_of", "ticker", "expiry", "right", "strike", "bid", "ask"], "chain")
    if len(chain) == 0:
        msg = "la cadena está vacía: no hay nada que agregar"
        raise DataQualityError(msg)

    event_dates: dict[str, np.ndarray] = {}
    if events is not None:
        ev = flow._normalize_events(events)
        for tkr, sub in ev.groupby("ticker", sort=False):
            event_dates[str(tkr)] = np.sort(
                pd.DatetimeIndex(sub["event_date"]).to_numpy()
            )

    rows: list[dict[str, object]] = []
    for (day, ticker), group in chain.groupby(["as_of", "ticker"], sort=True):
        day_ts = pd.Timestamp(day).normalize()
        spot = (
            float(group["spot"].iloc[0])
            if "spot" in group.columns and np.isfinite(group["spot"].iloc[0])
            else float("nan")
        )
        row: dict[str, object] = {"date": day_ts, "ticker": str(ticker)}
        row.update(_aggregate_one(
            group, day_ts, str(ticker), spot, event_dates, cfg,
            min_dte_skew=min_dte_skew, vol_spread_dte=vol_spread_dte,
            vol_spread_band=vol_spread_band, weighting=weighting,
        ))
        rows.append(row)
    out = pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()
    if not out.select_dtypes(include=[float]).notna().to_numpy().any():
        msg = (
            "aggregate_chain_daily: ningún ticker-día produce señal alguna; "
            "revisa los filtros de calidad y la cobertura de la cadena"
        )
        raise InsufficientHistory(msg)
    return out


def _aggregate_one(
    group: pd.DataFrame,
    day: pd.Timestamp,
    ticker: str,
    spot: float,
    event_dates: Mapping[str, np.ndarray],
    cfg: QualityFilterConfig,
    *,
    min_dte_skew: float,
    vol_spread_dte: tuple[float, float],
    vol_spread_band: tuple[float, float],
    weighting: Literal["oi_mean", "oi_min", "equal"],
) -> dict[str, float]:
    """Señales de un ticker-día. NaN homogéneo si la calidad no da (informe §2.3)."""
    out: dict[str, float] = {c: float("nan") for c in _CHAIN_DAILY_COLUMNS}

    # 1. Forward implícito por vencimiento.
    forwards: dict[pd.Timestamp, ImpliedForward] = {}
    for expiry, sub in group.groupby("expiry", sort=True):
        exp_ts = pd.Timestamp(expiry)
        tau = max((exp_ts.normalize() - day).days, 0) / _DAYS_PER_YEAR
        fwd = implied_forward(sub, tau=tau, spot=spot if np.isfinite(spot) else None)
        if fwd is not None:
            forwards[exp_ts] = fwd
    # 2. Filtros de calidad con contabilidad.
    filtered, report = apply_quality_filters(group, forwards=forwards, config=cfg)
    out["quality_drop_fraction"] = report.dropped_fraction
    out["n_contracts"] = float(report.initial)
    # Agregados de flujo: se calculan SIEMPRE sobre la cadena bruta (el volumen
    # negociado existe aunque la cotización de cierre sea mala).
    sign_all = _right_sign(group["right"])
    if "volume" in group.columns:
        out["call_volume"] = float(group.loc[sign_all > 0, "volume"].sum())
        out["put_volume"] = float(group.loc[sign_all < 0, "volume"].sum())
        out["option_volume"] = out["call_volume"] + out["put_volume"]
    if "open_interest" in group.columns:
        out["call_open_interest"] = float(group.loc[sign_all > 0, "open_interest"].sum())
        out["put_open_interest"] = float(group.loc[sign_all < 0, "open_interest"].sum())
    if not forwards or not report.usable(cfg):
        return out

    expiries = sorted(forwards)
    front = forwards[expiries[0]]
    out["spot"] = spot
    out["forward_front"] = front.forward
    out["discount_front"] = front.discount
    out["implied_rate_front"] = front.implied_rate
    out["implied_borrow_front"] = front.implied_borrow

    # 3-4. Sonrisas e IVs propias.
    smiles: dict[pd.Timestamp, SmileSlice] = {}
    for exp_ts in expiries:
        smile = _build_smile(filtered.loc[filtered["expiry"] == exp_ts], forwards[exp_ts])
        if smile is not None:
            smiles[exp_ts] = smile
    vs, n_pairs = vol_spread_cw(
        filtered, forwards, dte_range=vol_spread_dte,
        moneyness_band=vol_spread_band, weighting=weighting,
    )
    out["vol_spread"] = vs
    out["vol_spread_pairs"] = float(n_pairs)

    skew_exp = next(
        (e for e in expiries if forwards[e].tau * _DAYS_PER_YEAR >= min_dte_skew and e in smiles),
        None,
    )
    if skew_exp is not None:
        out["iv_skew_25delta"] = skew_25delta(smiles[skew_exp])
        # XZZ selecciona strikes por K/S; sin spot en la cadena se usa F·DF como
        # proxy (S = F·DF + VP(dividendos); el error es el VP del dividendo, muy
        # por debajo de la anchura de las bandas de moneyness del paper).
        spot_for_xzz = spot if np.isfinite(spot) else (
            forwards[skew_exp].forward * forwards[skew_exp].discount
        )
        out["iv_skew_xzz"] = skew_xzz(smiles[skew_exp], spot_for_xzz)
        if np.isfinite(out["iv_skew_25delta"]) and np.isfinite(out["iv_skew_xzz"]):
            out["iv_curvature"] = out["iv_skew_25delta"] - out["iv_skew_xzz"]

    atm: dict[pd.Timestamp, float] = {
        e: s.atm_iv() for e, s in smiles.items() if np.isfinite(s.atm_iv())
    }
    usable = [e for e in expiries if e in atm]
    if usable:
        out["iv_atm_front"] = atm[usable[0]]
        out["dte_front"] = forwards[usable[0]].tau * _DAYS_PER_YEAR
    if len(usable) >= 2:
        out["iv_atm_next"] = atm[usable[1]]
        out["dte_next"] = forwards[usable[1]].tau * _DAYS_PER_YEAR
        out["iv_term_slope_raw"] = atm[usable[-1]] - atm[usable[0]]

    # 5. Conteo de anuncios por vencimiento y descomposición del evento.
    counts: dict[pd.Timestamp, int] | None = None
    if ticker in event_dates:
        dates = event_dates[ticker]
        counts = {
            e: int(((dates > np.datetime64(day)) & (dates <= np.datetime64(e))).sum())
            for e in usable
        }
    elif "covers_earnings" in group.columns:
        counts = {}
        for e in usable:
            sub = group.loc[group["expiry"] == e, "covers_earnings"]
            counts[e] = int(bool(sub.astype(bool).any()))
    if counts is not None and len(usable) >= 2:
        pair = next(
            (
                (a, b)
                for a, b in itertools.pairwise(usable)
                if (counts[a], counts[b]) in {(1, 1), (0, 1)}
            ),
            None,
        )
        if pair is not None:
            e1, e2 = pair
            deco = event_variance_decomposition(
                atm[e1], forwards[e1].tau, atm[e2], forwards[e2].tau, counts[e1], counts[e2]
            )
            if deco.is_valid:
                out["sigma_event"] = deco.sigma_event
                out["sigma_diffusive"] = deco.sigma_diffusive
                pillar = e1 if counts[e1] == 1 else e2
                out["iv_crush_expected"] = expected_iv_crush(
                    deco.sigma_diffusive, deco.sigma_event, atm[pillar], forwards[pillar].tau
                )
                out["event_share"] = deco.sigma_event**2 / (
                    atm[pillar] ** 2 * forwards[pillar].tau
                )
                out["expected_abs_move"] = ABS_MOVE_FACTOR * deco.sigma_event
                # Pendiente de-eventizada (§7.4, opción 1): se resta la varianza
                # del evento de cada pilar con conteo 0/1 y se compara corto-largo.
                de_evented: dict[pd.Timestamp, float] = {}
                for e in usable:
                    n_e = counts.get(e)
                    if n_e not in (0, 1):
                        continue
                    var = atm[e] ** 2 - n_e * deco.sigma_event**2 / forwards[e].tau
                    if var > 0.0:
                        de_evented[e] = math.sqrt(var)
                keys = [e for e in usable if e in de_evented]
                if len(keys) >= 2:
                    out["iv_term_slope_ex"] = de_evented[keys[-1]] - de_evented[keys[0]]

    # 6. Straddle ATM del vencimiento frontal utilizable.
    if usable:
        e0 = usable[0]
        straddle = atm_straddle(filtered.loc[filtered["expiry"] == e0], forwards[e0])
        if np.isfinite(straddle):
            frac = straddle / (forwards[e0].discount * forwards[e0].forward)
            out["straddle_frac"] = frac
            out["one_sigma_move"] = STRADDLE_TO_ONE_SIGMA * frac
    return out


_CHAIN_DAILY_COLUMNS: Final[tuple[str, ...]] = (
    "spot",
    "forward_front",
    "discount_front",
    "implied_rate_front",
    "implied_borrow_front",
    "vol_spread",
    "vol_spread_pairs",
    "iv_skew_25delta",
    "iv_skew_xzz",
    "iv_curvature",
    "iv_atm_front",
    "iv_atm_next",
    "dte_front",
    "dte_next",
    "iv_term_slope_raw",
    "iv_term_slope_ex",
    "sigma_event",
    "sigma_diffusive",
    "iv_crush_expected",
    "event_share",
    "expected_abs_move",
    "straddle_frac",
    "one_sigma_move",
    "call_volume",
    "put_volume",
    "option_volume",
    "call_open_interest",
    "put_open_interest",
    "quality_drop_fraction",
    "n_contracts",
)


# ---------------------------------------------------------------------------
# Pipeline por evento
# ---------------------------------------------------------------------------


@dataclass
class OptionsPreEventFeatures:
    """Señales de opciones por evento en ``[T-N, T-1]`` (contrato §3.5).

    ``compute(ctx)`` acepta un `events.preevent.EventContext` (o cualquier
    objeto con ``events`` y ``options``; ``prices`` es opcional) y devuelve un
    DataFrame indexado por ``event_id`` con las columnas de
    `OPTION_EVENT_FEATURES` más los metadatos ``ticker``, ``event_date`` y
    ``available_at`` (medianoche de T). ``options`` puede ser un panel diario
    ``(date, ticker)`` o cadenas por contrato (detectadas por las columnas
    ``strike/right/expiry`` y agregadas con `aggregate_chain_daily`).

    Ventanas point-in-time (todas terminan en T−1; el OI, en T−2 por su retardo
    de publicación OCC — informe §9.4 y §13):

    * ``vol_spread`` y ``iv_skew_25delta``: nivel en T−1 (informe §3.3), con
      cambios ``d_*`` a 5 y 20 sesiones y el spread anormal acumulado ``cavs``
      de Atilgan (2014) contra la mediana de ``[T−250, T−30]`` — el periodo de
      referencia termina en T−30 para no contaminarse con la propia
      acumulación, en paralelo a la ventana de estimación del contrato.
    * ``put_call_volume_ratio``: cuota de puts ``P/(P+C)`` media de
      ``[T−5, T−1]`` z-scoreada contra ``[T−60, T−11]`` **excluyendo ±5
      sesiones de los demás eventos del emisor** (informe §5.1: sin excluirlas
      la referencia contiene los picos estacionales y el z infravalora). El
      nivel bruto nunca se usa en sección cruzada. Versión sin firmar,
      degradada respecto a Pan-Poteshman (2006): ponderar poco.
    * ``option_stock_*``: O/S de Roll-Schwartz-Subrahmanyam con el volumen
      consolidado de la acción; **signo direccional NEGATIVO** (Johnson y So
      2012) — ver `option_stock_ratio`. Necesita ``ctx.prices``; sin él, NaN.
    * ``oi_buildup_calls/puts``: cambio del log-OI por lado entre T−2 y
      T−2−k (k=10). Versión de panel, degradada respecto a la delta-ponderada
      del informe §9.1 (`oi_buildup_delta_weighted`, que exige cadenas y
      capitalización). ``oi_buildup_net = calls − puts``; el ``d_pcr_oi`` de
      Fodor-Krieger-Doran (2011) es exactamente ``−oi_buildup_net`` en esta
      construcción y no se duplica como columna.
    * ``open_ratio_5d``: ``ΔOI_total/volumen_total`` medio de ``[T−6, T−2]``
      (informe §9.3.b): ≈+1 si casi todo el volumen abre posiciones. Firma
      parcial y gratuita del desglose apertura/cierre.
    * ``iv_term_slope``: **de-eventizada** (informe §7.4: la pendiente cruda es
      aritmética del calendario, no señal); la cruda se conserva en
      ``iv_term_slope_raw``.
    * ``event_iv`` (σ_E), ``iv_crush_expected``, ``event_share`` y
      ``expected_abs_move`` (= 0,79788·σ_E): descomposición de varianza en
      T−1 (informe §7.2-§7.3, §8.2.c).
    * ``event_vol_surprise``: σ_E frente a su propia historia; exige
      ``min_history_quarters`` eventos previos con σ_E (informe §7.4); con
      menos, NaN.
    * ``move_ratio_hist``: media, sobre los eventos previos del emisor, de
      ``|retorno del día del evento| / expected_abs_move(T−1)``. Barth y So
      (2014) predicen < 1 en promedio (prima de varianza del evento): el
      ``expected_move`` es riesgo-neutral, no una previsión. Necesita precios.

    Higiene estadística aguas arriba (informe §10.2 paso 8): estas features se
    estandarizan por cohorte y se ortogonalizan contra el retorno pre-evento en
    `events.preevent` (`residualize_features`); aquí no se imputa nada — en
    opciones la ausencia de dato está correlacionada con la iliquidez y la
    iliquidez es un factor: **NaN se queda NaN**.
    """

    detection_window: int = 5
    oi_change_window: int = 10
    base_window: tuple[int, int] = (-60, -11)
    spread_reference: tuple[int, int] = (-250, -30)
    cavs_window: int = 20
    exclusion_halfwidth: int = 5
    min_base_obs: int = 20
    min_reference_obs: int = 60
    min_history_quarters: int = 8
    min_move_history: int = 3
    quality: QualityFilterConfig = field(default_factory=QualityFilterConfig)

    def compute(self, ctx: object) -> pd.DataFrame:
        """Tabla de features por ``event_id`` desde un `EventContext` (o similar)."""
        events = getattr(ctx, "events", None)
        options = getattr(ctx, "options", None)
        prices = getattr(ctx, "prices", None)
        if events is None:
            msg = "el contexto no trae `events`: no hay eventos que caracterizar"
            raise DataQualityError(msg)
        if options is None:
            msg = (
                "el contexto no trae datos de opciones (`options is None`); "
                "OptionsPreEventFeatures no puede degradar: su única fuente son "
                "las opciones"
            )
            raise DataQualityError(msg)
        return self.compute_from_frames(options, events, prices=prices)

    # ------------------------------------------------------------------ núcleo

    def compute_from_frames(
        self,
        options: pd.DataFrame,
        events: pd.DataFrame,
        *,
        prices: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Versión de bajo nivel: panel/cadena de opciones + tabla de eventos."""
        if not isinstance(options, pd.DataFrame) or len(options) == 0:
            msg = "`options` está vacío o no es un DataFrame"
            raise DataQualityError(msg)
        ev = flow._normalize_events(events)
        if {"strike", "right", "expiry"}.issubset(options.columns):
            panel = aggregate_chain_daily(options, events=ev, config=self.quality)
        else:
            panel = self._validate_panel(options)

        mats, dates, tickers = self._panel_matrices(panel)
        n_dates = len(dates)
        col_of = {t: j for j, t in enumerate(tickers)}
        pos_arr = flow._positions(ev["event_date"], dates)
        excl = flow._exclusion_mask(dates, list(tickers), ev, self.exclusion_halfwidth)

        # Volumen de la acción y retornos, alineados a la rejilla de opciones.
        stock_vol = returns = None
        if prices is not None:
            px = flow._check_prices(prices, ["close", "volume"])
            stock_vol = flow._wide(px, "volume").reindex(dates)
            returns = flow._returns_wide(px).reindex(dates)

        n_ev = len(ev)
        feats: dict[str, np.ndarray] = {
            name: np.full(n_ev, np.nan) for name in OPTION_EVENT_FEATURES
        }

        def cell(name: str, pos: int, j: int) -> float:
            mat = mats.get(name)
            if mat is None or not 0 <= pos < n_dates:
                return float("nan")
            return float(mat[pos, j])

        # σ_E y expected_abs_move de cada evento en su propio T−1 (para las
        # features históricas por emisor).
        own_sigma = np.full(n_ev, np.nan)
        own_move = np.full(n_ev, np.nan)
        for i in range(n_ev):
            p = int(pos_arr[i])
            j = col_of.get(ev["ticker"].iloc[i])
            if p >= 1 and j is not None:
                own_sigma[i] = cell("sigma_event", p - 1, j)
                own_move[i] = cell("expected_abs_move", p - 1, j)

        by_ticker: dict[str, list[int]] = {}
        for i, t in enumerate(ev["ticker"]):
            by_ticker.setdefault(str(t), []).append(i)

        k_det = self.detection_window
        k_oi = self.oi_change_window
        for i in range(n_ev):
            ticker = str(ev["ticker"].iloc[i])
            p = int(pos_arr[i])
            j = col_of.get(ticker)
            if p < 1 or j is None:
                continue

            # --- niveles y cambios del spread y del skew --------------------
            feats["vol_spread"][i] = cell("vol_spread", p - 1, j)
            feats["iv_skew_25delta"][i] = cell("iv_skew_25delta", p - 1, j)
            feats["iv_skew_xzz"][i] = cell("iv_skew_xzz", p - 1, j)
            feats["iv_curvature"][i] = cell("iv_curvature", p - 1, j)
            if p - 6 >= 0:
                feats["d_vol_spread_5"][i] = feats["vol_spread"][i] - cell(
                    "vol_spread", p - 6, j
                )
                feats["d_iv_skew_5"][i] = feats["iv_skew_25delta"][i] - cell(
                    "iv_skew_25delta", p - 6, j
                )
            if p - 21 >= 0:
                feats["d_vol_spread_20"][i] = feats["vol_spread"][i] - cell(
                    "vol_spread", p - 21, j
                )
            feats["cavs_20"][i] = self._cavs(mats.get("vol_spread"), p, j)

            # --- put/call por volumen (z de la cuota de puts) ---------------
            feats["put_call_volume_ratio"][i] = self._window_zscore(
                mats.get("put_share"), p, j, excl[ticker], k_det
            )

            # --- O/S (necesita precios) -------------------------------------
            if stock_vol is not None and "option_volume" in mats:
                os_mat = mats.get("log_os")
                if os_mat is None:
                    opt = mats["option_volume"]
                    sv = stock_vol.reindex(columns=list(tickers)).to_numpy(dtype=float)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        os_mat = np.log(
                            np.where(
                                (opt > 0) & (sv > 0), opt * 100.0 / np.maximum(sv, 1.0), np.nan
                            )
                        )
                    mats["log_os"] = os_mat
                lo, hi = p - k_det, p
                if lo >= 0:
                    seg = os_mat[lo:hi, j]
                    if np.isfinite(seg).sum() >= 3:
                        feats["option_stock_level"][i] = float(
                            np.exp(np.nanmean(seg))
                        )
                z = self._window_zscore(os_mat, p, j, excl[ticker], k_det)
                feats["option_stock_magnitude"][i] = z
                feats["option_stock_direction"][i] = -z if np.isfinite(z) else np.nan

            # --- open interest (última sesión utilizable: T−2, §9.4) --------
            oi_c, oi_p = mats.get("call_oi"), mats.get("put_oi")
            if oi_c is not None and oi_p is not None and p - 2 - k_oi >= 0:
                feats["oi_buildup_calls"][i] = _log_change(oi_c, p - 2, k_oi, j)
                feats["oi_buildup_puts"][i] = _log_change(oi_p, p - 2, k_oi, j)
                feats["oi_buildup_net"][i] = (
                    feats["oi_buildup_calls"][i] - feats["oi_buildup_puts"][i]
                )
            feats["open_ratio_5d"][i] = self._open_ratio(mats, p, j)

            # --- estructura temporal y evento -------------------------------
            feats["iv_term_slope"][i] = cell("iv_term_slope_ex", p - 1, j)
            feats["iv_term_slope_raw"][i] = cell("iv_term_slope_raw", p - 1, j)
            feats["event_iv"][i] = own_sigma[i]
            feats["iv_crush_expected"][i] = cell("iv_crush_expected", p - 1, j)
            feats["event_share"][i] = cell("event_share", p - 1, j)
            feats["expected_abs_move"][i] = own_move[i]

        # --- features históricas por emisor (σ_E y move ratio) --------------
        for ticker, idxs in by_ticker.items():
            idxs_sorted = sorted(idxs, key=lambda i: pos_arr[i] if pos_arr[i] >= 0 else 10**9)
            j = col_of.get(ticker)
            for rank, i in enumerate(idxs_sorted):
                p = int(pos_arr[i])
                if p < 0:
                    continue
                prior = [
                    q for q in idxs_sorted[:rank] if 0 <= pos_arr[q] < p
                ]
                sig_prior = own_sigma[[*prior]] if prior else np.array([])
                sig_prior = sig_prior[np.isfinite(sig_prior)]
                if len(sig_prior) >= self.min_history_quarters:
                    tail = sig_prior[-self.min_history_quarters:]
                    sd = float(np.std(tail, ddof=1))
                    if sd > 0.0 and np.isfinite(own_sigma[i]):
                        feats["event_vol_surprise"][i] = (
                            own_sigma[i] - float(np.mean(tail))
                        ) / sd
                if returns is not None and j is not None:
                    ratios: list[float] = []
                    for q in prior:
                        pq = int(pos_arr[q])
                        move = own_move[q]
                        if not (np.isfinite(move) and move > 0.0) or pq >= n_dates:
                            continue
                        r_evt = float(returns.iloc[pq, returns.columns.get_loc(ticker)]) if (
                            ticker in returns.columns
                        ) else float("nan")
                        if np.isfinite(r_evt):
                            ratios.append(abs(r_evt) / move)
                    if len(ratios) >= self.min_move_history:
                        feats["move_ratio_hist"][i] = float(
                            np.mean(ratios[-self.min_history_quarters:])
                        )

        index = pd.Index(ev["event_id"], name="event_id")
        frame = pd.DataFrame(feats, index=index)
        meta = pd.DataFrame(
            {
                "ticker": ev["ticker"].to_numpy(),
                "event_date": ev["event_date"].to_numpy(),
                "available_at": ev["event_date"].to_numpy(),
            },
            index=index,
        )
        out = pd.concat([meta, frame], axis=1)
        flow._raise_if_all_nan(frame, "options_signals")
        out.attrs["detection_window"] = self.detection_window
        out.attrs["oi_change_window"] = self.oi_change_window
        out.attrs["base_window"] = self.base_window
        out.attrs["oi_last_usable_offset"] = -2
        return out

    # ------------------------------------------------------------------ ayudas

    def _validate_panel(self, options: pd.DataFrame) -> pd.DataFrame:
        """Valida un panel diario ``(date, ticker)`` y su contenido mínimo."""
        if not isinstance(options.index, pd.MultiIndex) or [
            str(n) for n in options.index.names
        ] != ["date", "ticker"]:
            msg = (
                "`options` debe ser un panel (date, ticker) o una cadena por "
                "contrato con columnas strike/right/expiry; recibido un DataFrame "
                f"con índice {options.index.names}"
            )
            raise DataQualityError(msg)
        known = {
            "vol_spread", "iv_skew_25delta", "call_volume", "put_volume",
            "call_open_interest", "put_open_interest", "option_volume",
        } | {c for c in options.columns if re.fullmatch(r"iv_atm_\d+d", str(c))}
        if not (set(map(str, options.columns)) & known):
            msg = (
                "el panel de opciones no contiene ninguna columna reconocible "
                f"(esperaba alguna de {sorted(known)}); columnas: "
                f"{list(options.columns)}"
            )
            raise DataQualityError(msg)
        return options

    def _panel_matrices(
        self, panel: pd.DataFrame
    ) -> tuple[dict[str, np.ndarray], pd.DatetimeIndex, list[str]]:
        """Pivota el panel a matrices fechas × tickers y deriva la parte de evento."""
        dates = pd.DatetimeIndex(
            panel.index.get_level_values("date").unique()
        ).sort_values()
        tickers = sorted(panel.index.get_level_values("ticker").unique())

        def wide(col: str) -> np.ndarray | None:
            if col not in panel.columns:
                return None
            return (
                panel[col]  # noqa: PD010 - pivot_table agregaría duplicados en silencio
                .unstack("ticker")
                .reindex(index=dates, columns=tickers)
                .to_numpy(dtype=float)
            )

        mats: dict[str, np.ndarray] = {}
        for name, col in (
            ("vol_spread", "vol_spread"),
            ("iv_skew_25delta", "iv_skew_25delta"),
            ("iv_skew_xzz", "iv_skew_xzz"),
            ("iv_curvature", "iv_curvature"),
            ("call_volume", "call_volume"),
            ("put_volume", "put_volume"),
            ("call_oi", "call_open_interest"),
            ("put_oi", "put_open_interest"),
            ("iv_term_slope_ex", "iv_term_slope_ex"),
            ("iv_term_slope_raw", "iv_term_slope_raw"),
            ("sigma_event", "sigma_event"),
            ("iv_crush_expected", "iv_crush_expected"),
            ("event_share", "event_share"),
            ("expected_abs_move", "expected_abs_move"),
        ):
            mat = wide(col)
            if mat is not None:
                mats[name] = mat

        if "call_volume" in mats and "put_volume" in mats:
            total = mats["call_volume"] + mats["put_volume"]
            with np.errstate(invalid="ignore", divide="ignore"):
                mats["put_share"] = np.where(total > 0, mats["put_volume"] / total, np.nan)
            mats["option_volume"] = total
        elif (ov := wide("option_volume")) is not None:
            mats["option_volume"] = ov

        # Estructura temporal: derivación desde pilares de IV constante si el
        # panel no la trae ya precalculada (ruta de cadena).
        if "sigma_event" not in mats:
            derived = _derive_event_vol_matrices(panel, dates, tickers)
            mats.update(derived)
        if "iv_term_slope_raw" not in mats:
            raw = wide("iv_term_slope")
            if raw is not None:
                mats["iv_term_slope_raw"] = raw
        return mats, dates, tickers

    def _window_zscore(
        self,
        mat: np.ndarray | None,
        pos: int,
        j: int,
        excl: np.ndarray,
        k: int,
    ) -> float:
        """Z de la media de ``[T−k, T−1]`` contra la base ``[T−60, T−11]`` propia.

        La base excluye las ±`exclusion_halfwidth` sesiones de los demás
        eventos del emisor (informe §5.1) y exige `min_base_obs` observaciones
        y sigma positiva; si no, NaN.
        """
        if mat is None:
            return float("nan")
        b0, b1 = self.base_window
        lo_b, hi_b = pos + b0, pos + b1 + 1
        lo_d, hi_d = pos - k, pos
        if lo_b < 0 or lo_d < 0:
            return float("nan")
        det = mat[lo_d:hi_d, j]
        if np.isfinite(det).sum() < max(3, k - 2):
            return float("nan")
        base = mat[lo_b:hi_b, j].copy()
        base[excl[lo_b:hi_b]] = np.nan
        valid = np.isfinite(base)
        if int(valid.sum()) < self.min_base_obs:
            return float("nan")
        mu = float(np.mean(base[valid]))
        sd = float(np.std(base[valid], ddof=1))
        if not np.isfinite(sd) or sd <= 0.0:
            return float("nan")
        return (float(np.nanmean(det)) - mu) / sd

    def _cavs(self, vs: np.ndarray | None, pos: int, j: int) -> float:
        """Spread anormal acumulado de Atilgan (2014) — informe §3.2.b.

        ``avs = vol_spread − mediana_[T−250, T−30]`` (mediana, no media: la
        serie tiene colas gruesas por días con pocos pares) y
        ``cavs = Σ_[T−20, T−1] avs``. Exige `min_reference_obs` observaciones en
        la referencia y ventana de acumulación completa.
        """
        if vs is None:
            return float("nan")
        r0, r1 = self.spread_reference
        lo_r, hi_r = pos + r0, pos + r1 + 1
        lo_a = pos - self.cavs_window
        if lo_r < 0 or lo_a < 0:
            return float("nan")
        ref = vs[lo_r:hi_r, j]
        ref = ref[np.isfinite(ref)]
        if len(ref) < self.min_reference_obs:
            return float("nan")
        acc = vs[lo_a:pos, j]
        if not np.isfinite(acc).all():
            return float("nan")
        return float(np.sum(acc - np.median(ref)))

    def _open_ratio(
        self, mats: Mapping[str, np.ndarray], pos: int, j: int
    ) -> float:
        """``ΔOI_total / volumen_total`` medio de ``[T−6, T−2]`` (informe §9.3.b).

        ≈ +1 → casi todo el volumen del día abrió posiciones nuevas; ≈ −1 →
        casi todo cerró. No dice quién, pero es la mitad gratuita del desglose
        Cboe Open-Close. Respeta el retardo del OI (última sesión: T−2).
        """
        oi_c, oi_p = mats.get("call_oi"), mats.get("put_oi")
        vol = mats.get("option_volume")
        if oi_c is None or oi_p is None or vol is None:
            return float("nan")
        lo, hi = pos - 6, pos - 1  # sesiones T−6 .. T−2
        if lo - 1 < 0:
            return float("nan")
        oi_tot = oi_c[:, j] + oi_p[:, j]
        d_oi = oi_tot[lo:hi] - oi_tot[lo - 1 : hi - 1]
        v = vol[lo:hi, j]
        ok = np.isfinite(d_oi) & np.isfinite(v) & (v > 0)
        if int(ok.sum()) < 3:
            return float("nan")
        ratio = np.clip(d_oi[ok] / v[ok], -1.0, 1.0)
        return float(np.mean(ratio))


def _derive_event_vol_matrices(
    panel: pd.DataFrame, dates: pd.DatetimeIndex, tickers: list[str]
) -> dict[str, np.ndarray]:
    """Descomposición del evento sobre pilares de IV a vencimiento constante.

    Con pilares ``iv_atm_{h}d`` y ``days_to_earnings`` (d): el **post-pilar** es
    el menor ``h >= d`` (contiene el anuncio) y el **pre-pilar** el mayor
    ``h < d`` (no lo contiene). Si hay pre-pilar se aplica el Caso B del
    informe §7.2; si el anuncio cae antes del primer pilar, el Caso A con los
    dos primeros. ``d < 1`` (anuncio ya público) o sin post-pilar → NaN. Las
    varianzas negativas devuelven NaN, no cero (§7.2).

    Con vencimientos constantes la interpretación del crush es aproximada (el
    pilar no es un contrato listado); la ruta de cadenas usa vencimientos
    reales. Devuelve también la pendiente de-eventizada (§7.4, opción 1)::

        iv_ex(h)² = iv(h)² − 1{d <= h}·σ_E²·(365/h)
        iv_term_slope_ex = iv_ex(h_largo) − iv_ex(h_corto)
    """
    horizons: list[int] = sorted(
        int(m.group(1))
        for c in panel.columns
        if (m := re.fullmatch(r"iv_atm_(\d+)d", str(c)))
    )
    if len(horizons) < 2 or "days_to_earnings" not in panel.columns:
        return {}

    def wide(col: str) -> np.ndarray:
        return (
            panel[col]  # noqa: PD010 - pivot_table agregaría duplicados en silencio
            .unstack("ticker")
            .reindex(index=dates, columns=tickers)
            .to_numpy(dtype=float)
        )

    iv = {h: wide(f"iv_atm_{h}d") for h in horizons}
    d = wide("days_to_earnings")
    shape = d.shape
    var_e = np.full(shape, np.nan)
    sig_d = np.full(shape, np.nan)
    crush = np.full(shape, np.nan)
    share = np.full(shape, np.nan)

    with np.errstate(invalid="ignore"):
        for idx, h_post in enumerate(horizons):
            t_post = h_post / _DAYS_PER_YEAR
            if idx == 0:
                # Caso A con los dos primeros pilares (ambos contienen el anuncio).
                h2 = horizons[1]
                t1, t2 = t_post, h2 / _DAYS_PER_YEAR
                mask = (d >= 1.0) & (d <= h_post)
                vd = (iv[h2] ** 2 * t2 - iv[h_post] ** 2 * t1) / (t2 - t1)
                ve = t1 * t2 * (iv[h_post] ** 2 - iv[h2] ** 2) / (t2 - t1)
            else:
                h_pre = horizons[idx - 1]
                mask = (d > h_pre) & (d <= h_post)
                vd = iv[h_pre] ** 2
                ve = t_post * (iv[h_post] ** 2 - iv[h_pre] ** 2)
            ok = mask & (vd > 0.0) & (ve >= 0.0)
            var_e[ok] = ve[ok]
            sig_d[ok] = np.sqrt(vd[ok])
            crush[ok] = 1.0 - np.sqrt(vd[ok]) / iv[h_post][ok]
            share[ok] = ve[ok] / (iv[h_post][ok] ** 2 * t_post)

        sigma_e = np.sqrt(var_e)
        h_short = horizons[1] if len(horizons) >= 3 else horizons[0]
        h_long = horizons[-1]
        parts: dict[int, np.ndarray] = {}
        for h in (h_short, h_long):
            includes = (d >= 1.0) & (d <= h)
            # 0·NaN sería NaN: si el pilar no contiene el anuncio no se resta nada.
            correction = np.where(includes, var_e * (_DAYS_PER_YEAR / h), 0.0)
            v = iv[h] ** 2 - correction
            parts[h] = np.where(v > 0.0, np.sqrt(v), np.nan)
        slope_ex = parts[h_long] - parts[h_short]

    return {
        "sigma_event": sigma_e,
        "sigma_diffusive": sig_d,
        "iv_crush_expected": crush,
        "event_share": share,
        "expected_abs_move": ABS_MOVE_FACTOR * sigma_e,
        "iv_term_slope_ex": slope_ex,
    }


def compute_pre_event_option_features(
    options: pd.DataFrame,
    events: pd.DataFrame,
    prices: pd.DataFrame | None = None,
    **kwargs: object,
) -> pd.DataFrame:
    """Punto de entrada que consume `events.preevent.PreEventFeatures` (contrato §3.5).

    Equivale a ``OptionsPreEventFeatures(**kwargs).compute_from_frames(options,
    events, prices=prices)``. Cuando lo invoca `preevent` solo con
    ``(options, events)``, las features que necesitan precios (O/S y
    ``move_ratio_hist``) salen NaN: degradación documentada, no silenciosa.
    """
    engine = OptionsPreEventFeatures(**kwargs)  # type: ignore[arg-type]
    return engine.compute_from_frames(options, events, prices=prices)


# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------


def _require(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        msg = f"`{name}` no tiene las columnas requeridas {missing}; tiene {list(frame.columns)}"
        raise DataQualityError(msg)


def _right_sign(right: pd.Series) -> np.ndarray:
    """+1 para calls, −1 para puts. Acepta 'C'/'P', 'call'/'put' y variantes."""
    first = right.astype(str).str.strip().str.upper().str[0]
    sign = np.where(first == "C", 1, np.where(first == "P", -1, 0))
    if (sign == 0).any():
        bad = sorted(set(right[sign == 0].astype(str).head(3)))
        msg = f"columna `right` con valores no interpretables (muestra: {bad})"
        raise DataQualityError(msg)
    return sign


def _one_right(right: object) -> int:
    r = str(right).strip().upper()[:1]
    if r == "C":
        return 1
    if r == "P":
        return -1
    msg = f"valor de `right` no interpretable: {right!r}"
    raise DataQualityError(msg)


def _log_change(mat: np.ndarray, pos: int, k: int, j: int) -> float:
    """``ln(x[pos]) − ln(x[pos−k])`` con NaN si algún extremo no es positivo."""
    if pos - k < 0 or pos >= mat.shape[0]:
        return float("nan")
    a, b = mat[pos, j], mat[pos - k, j]
    if not (np.isfinite(a) and np.isfinite(b) and a > 0.0 and b > 0.0):
        return float("nan")
    return float(math.log(a) - math.log(b))
