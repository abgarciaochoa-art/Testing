"""Cadenas de opciones históricas normalizadas y matemática de IV (`data.options`).

Responsabilidad (contrato §2, `data.options`): servir cadenas de opciones de
varios proveedores en un **esquema único** y aportar el cálculo propio de
volatilidad implícita y griegas, de modo que las features del ángulo B
(`vol_spread`, `iv_skew_25delta`, `put_call_volume_ratio`, `oi_buildup`,
`iv_term_slope`) no dependan del modelo opaco de cada proveedor.

Esquema canónico (una fila por contrato y día)::

    date            sesión del snapshot (tz-naive, medianoche)
    ticker          subyacente normalizado ("BRK.B")
    expiry          fecha de vencimiento
    strike          precio de ejercicio
    right           "C" | "P"
    bid, ask, last  precios del contrato (NaN si el proveedor no los da)
    volume          contratos negociados en la sesión
    open_interest   interés abierto (ver disponibilidad más abajo)
    iv              volatilidad implícita anualizada
    delta           delta del contrato

Columnas adicionales permitidas y añadidas cuando hay datos: `mid`, `spot`,
`gamma`, `vega`, `source`, y las dos columnas de disponibilidad PIT.

**Disponibilidad point-in-time** (`docs/research/data_sources.md` §7.3 y
`docs/research/informed_trading.md` §13): el volumen y la IV del día `t` son
conocibles al **cierre de t** (`available_at`), pero el open interest lo
disemina la OCC en su proceso nocturno y no es conocible hasta la
**pre-apertura de t+1** (`oi_available_at`). Con ventanas pre-evento de 5-20
sesiones, ignorar ese día de retardo es un error relativo grande; por eso las
dos columnas viajan con la cadena y `oi_buildup` debe indexarse por
`oi_available_at`, nunca por `date`.

Volatilidad implícita propia
----------------------------
`implied_vol` invierte Black-Scholes-Merton (Black y Scholes 1973; Merton 1973)
con **Newton-Raphson salvaguardado por bisección**: cada iterando mantiene un
corchete `[lo, hi]` con cambio de signo; si el paso de Newton se sale del
corchete o la vega es demasiado pequeña (opciones profundamente ITM/OTM, donde
Newton diverge), se da un paso de bisección. La convergencia está garantizada
porque el precio BSM es estrictamente creciente en sigma. Precios fuera de las
cotas de no-arbitraje devuelven **NaN, nunca un valor por defecto** (filtro F9
de `docs/research/options_signals.md` §2.3). El arranque usa la aproximación
ATM de Brenner y Subrahmanyam (1988), `sigma_0 ≈ precio/(0.398*S*sqrt(tau))`.

`implied_forward` implementa la regresión de paridad put-call de
`docs/research/options_signals.md` §2.1: `C(K) - P(K) = DF*F - DF*K` es una
recta en K cuya ordenada da `DF*F` y cuya pendiente da `-DF`. De ahí salen el
forward implícito, el factor de descuento y la comisión de préstamo implícita
sin estimar dividendos por separado (metodología del nivel forward del Cboe
VIX; evita el sesgo de dividendo documentado por Wallmeier 2024, *Journal of
Futures Markets* 44).

Proveedores: Polygon (snapshot con IV/griegas + reconstrucción histórica),
ORATS (superficie EOD desde 2007), Tradier (solo snapshot actual; inútil para
backtest, se registra como tal) y sintético (`SyntheticMarket.options_chain`).
"""

from __future__ import annotations

import datetime as dt
import logging
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import ndtr

from earnings_alpha.config import Settings, get_settings
from earnings_alpha.data.base import (
    BaseProvider,
    DataKind,
    HttpClient,
    ProviderRegistry,
    get_registry,
)
from earnings_alpha.data.cache import DiskCache
from earnings_alpha.data.synthetic import SyntheticMarket
from earnings_alpha.errors import (
    ConfigError,
    DataQualityError,
    InsufficientHistory,
    ProviderUnavailable,
)
from earnings_alpha.pit import TradingCalendar, eastern_to_utc, get_calendar
from earnings_alpha.types import Ticker, normalize_ticker

__all__ = [  # noqa: RUF022 - orden temático, no alfabético
    # esquema
    "OPTIONS_KIND",
    "CHAIN_COLUMNS",
    "normalize_chain",
    "chain_availability",
    "filter_chain",
    "occ_symbol",
    "parse_occ_symbol",
    # matemática
    "BsGreeks",
    "bs_price",
    "bs_greeks",
    "implied_vol",
    "implied_forward",
    "implied_forward_table",
    # proveedores
    "OptionsProviderBase",
    "SyntheticOptionsProvider",
    "PolygonOptionsProvider",
    "ORATSOptionsProvider",
    "TradierOptionsProvider",
    # registro y fachadas
    "DEFAULT_OPTIONS_PRIORITIES",
    "register_options_providers",
    "get_option_chain",
]

logger = logging.getLogger(__name__)

OPTIONS_KIND = DataKind.OPTIONS.value

CHAIN_COLUMNS: tuple[str, ...] = (
    "date",
    "ticker",
    "expiry",
    "strike",
    "right",
    "bid",
    "ask",
    "last",
    "volume",
    "open_interest",
    "iv",
    "delta",
)
"""Columnas obligatorias del esquema canónico de cadenas."""

DateLike = str | dt.date | dt.datetime | pd.Timestamp

_SQRT_2PI = math.sqrt(2.0 * math.pi)


# ===========================================================================
# 1. Black-Scholes-Merton: precio, griegas e IV
# ===========================================================================


def _phi(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / _SQRT_2PI


def _broadcast(*arrays: Any) -> tuple[np.ndarray, ...]:
    return tuple(np.asarray(a, dtype=float) for a in np.broadcast_arrays(*arrays))


def bs_price(
    spot: Any,
    strike: Any,
    tau: Any,
    sigma: Any,
    rate: Any = 0.0,
    div_yield: Any = 0.0,
    is_call: Any = True,
) -> np.ndarray:
    """Precio Black-Scholes-Merton europeo con dividendo continuo.

    ``call = S e^{-q tau} N(d1) - K e^{-r tau} N(d2)`` con
    ``d1 = (ln(S/K) + (r - q + sigma^2/2) tau) / (sigma sqrt(tau))`` (Black y
    Scholes 1973; Merton 1973). Vectorizado por difusión (*broadcasting*).
    `tau <= 0` devuelve el valor intrínseco (el contrato ya venció).
    """
    s, k, t, v, r, q = _broadcast(spot, strike, tau, sigma, rate, div_yield)
    call = np.broadcast_to(np.asarray(is_call, dtype=bool), s.shape)
    intrinsic = np.where(call, np.maximum(s - k, 0.0), np.maximum(k - s, 0.0))
    live = (t > 0) & (v > 0) & (s > 0) & (k > 0)
    t_ = np.where(live, t, 1.0)
    v_ = np.where(live, v, 1.0)
    sq = np.sqrt(t_)
    d1 = (np.log(s / np.where(live, k, 1.0)) + (r - q + 0.5 * v_ * v_) * t_) / (v_ * sq)
    d2 = d1 - v_ * sq
    disc_r, disc_q = np.exp(-r * t_), np.exp(-q * t_)
    c = s * disc_q * ndtr(d1) - k * disc_r * ndtr(d2)
    p = k * disc_r * ndtr(-d2) - s * disc_q * ndtr(-d1)
    out = np.where(call, c, p)
    return np.where(live, out, intrinsic)


@dataclass(frozen=True, slots=True)
class BsGreeks:
    """Griegas BSM. `vega` por punto porcentual de vol; `theta` por día natural."""

    price: np.ndarray
    delta: np.ndarray
    gamma: np.ndarray
    vega: np.ndarray
    theta: np.ndarray
    rho: np.ndarray


def bs_greeks(
    spot: Any,
    strike: Any,
    tau: Any,
    sigma: Any,
    rate: Any = 0.0,
    div_yield: Any = 0.0,
    is_call: Any = True,
) -> BsGreeks:
    """Griegas analíticas BSM (Hull, *Options, Futures and Other Derivatives*).

    Convenciones de mesa, iguales que `data.synthetic._bs_greeks` donde ambas
    se solapan: `vega` por **punto porcentual** de volatilidad (dPrecio/dSigma
    dividido por 100) y `theta` por **día natural** (dPrecio/dTiempo / 365).
    `rho` por punto porcentual de tipo. `gamma` y `delta` en unidades crudas.
    """
    s, k, t, v, r, q = _broadcast(spot, strike, tau, sigma, rate, div_yield)
    call = np.broadcast_to(np.asarray(is_call, dtype=bool), s.shape)
    live = (t > 0) & (v > 0) & (s > 0) & (k > 0)
    t_ = np.where(live, t, 1.0)
    v_ = np.where(live, v, 1.0)
    sq = np.sqrt(t_)
    d1 = (np.log(s / np.where(live, k, 1.0)) + (r - q + 0.5 * v_ * v_) * t_) / (v_ * sq)
    d2 = d1 - v_ * sq
    disc_r, disc_q = np.exp(-r * t_), np.exp(-q * t_)
    pdf = _phi(d1)

    price = bs_price(s, k, t, v, r, q, call)
    delta = np.where(call, disc_q * ndtr(d1), disc_q * (ndtr(d1) - 1.0))
    gamma = disc_q * pdf / (s * v_ * sq)
    vega = s * disc_q * pdf * sq / 100.0
    theta_common = -s * disc_q * pdf * v_ / (2.0 * sq)
    theta_call = theta_common + q * s * disc_q * ndtr(d1) - r * k * disc_r * ndtr(d2)
    theta_put = theta_common - q * s * disc_q * ndtr(-d1) + r * k * disc_r * ndtr(-d2)
    theta = np.where(call, theta_call, theta_put) / 365.0
    rho = np.where(call, k * t_ * disc_r * ndtr(d2), -k * t_ * disc_r * ndtr(-d2)) / 100.0

    nan = np.full_like(s, np.nan)
    keep = live
    return BsGreeks(
        price=price,
        delta=np.where(keep, delta, np.where(call, (s > k) * 1.0, -(k > s) * 1.0)),
        gamma=np.where(keep, gamma, nan),
        vega=np.where(keep, vega, nan),
        theta=np.where(keep, theta, nan),
        rho=np.where(keep, rho, nan),
    )


def implied_vol(
    price: Any,
    spot: Any,
    strike: Any,
    tau: Any,
    rate: Any = 0.0,
    div_yield: Any = 0.0,
    is_call: Any = True,
    *,
    lo: float = 1e-4,
    hi: float = 5.0,
    tol: float = 1e-9,
    max_iter: int = 100,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Volatilidad implícita BSM por Newton-Raphson con bisección de respaldo.

    Algoritmo (vectorizado; una máscara por elemento):

    1. **Cotas de no-arbitraje.** Un precio por debajo del valor intrínseco
       descontado o por encima de la cota superior (``S e^{-q tau}`` para la
       call, ``K e^{-r tau}`` para la put), o fuera de lo alcanzable con
       ``sigma in [lo, hi]``, devuelve NaN (filtro F9 del informe de opciones:
       *si el solver no converge, es NaN, no un valor por defecto*).
    2. **Arranque** con Brenner-Subrahmanyam (1988):
       ``sigma_0 = precio / (phi(0) * S * sqrt(tau))``, recortado a `[lo, hi]`.
    3. **Newton salvaguardado.** El paso usa la vega analítica; si el iterando
       cae fuera del corchete vigente `[lo_i, hi_i]` o la vega es < 1e-12
       (contratos muy ITM/OTM, donde Newton se dispara), se sustituye por un
       paso de **bisección**. El corchete se actualiza con el signo del error,
       así que se estrecha monótonamente y la convergencia está garantizada.

    Con `return_diagnostics=True` devuelve además contadores (`newton_steps`,
    `bisection_steps`, `converged`) que los tests usan para verificar que la
    ruta de bisección existe y se ejercita.
    """
    p, s, k, t, r, q = _broadcast(price, spot, strike, tau, rate, div_yield)
    call = np.broadcast_to(np.asarray(is_call, dtype=bool), p.shape)
    out_shape = p.shape
    # Se trabaja siempre en 1-D: simplifica las máscaras y admite escalares.
    p, s, k, t, r, q = (np.atleast_1d(a).astype(float) for a in (p, s, k, t, r, q))
    call = np.atleast_1d(call).copy()
    shape = p.shape

    out = np.full(shape, np.nan)
    disc_r, disc_q = np.exp(-r * np.maximum(t, 0.0)), np.exp(-q * np.maximum(t, 0.0))
    lower = np.where(call, np.maximum(s * disc_q - k * disc_r, 0.0),
                     np.maximum(k * disc_r - s * disc_q, 0.0))
    upper = np.where(call, s * disc_q, k * disc_r)
    price_lo = bs_price(s, k, t, lo, r, q, call)
    price_hi = bs_price(s, k, t, hi, r, q, call)

    active = (
        (t > 0)
        & (s > 0)
        & (k > 0)
        & np.isfinite(p)
        & (p >= lower - 1e-12)
        & (p <= upper + 1e-12)
        & (p >= price_lo - 1e-12)
        & (p <= price_hi + 1e-12)
    )
    n_newton = 0
    n_bisect = 0

    lo_arr = np.full(shape, float(lo))
    hi_arr = np.full(shape, float(hi))
    sigma = np.clip(p / (_phi(np.zeros(shape)) * s * np.sqrt(np.maximum(t, 1e-12))), lo, hi)
    sigma = np.where(active, sigma, np.nan)

    for _ in range(max_iter):
        if not active.any():
            break
        model = bs_price(s, k, t, sigma, r, q, call)
        err = model - p
        # Escala de tolerancia: absoluta en precio, relativa al subyacente.
        done = active & (np.abs(err) <= tol * np.maximum(s, 1.0))
        out[done] = sigma[done]
        active &= ~done
        if not active.any():
            break
        # Actualiza el corchete con el signo del error (precio creciente en sigma).
        hi_arr = np.where(active & (err > 0), sigma, hi_arr)
        lo_arr = np.where(active & (err < 0), sigma, lo_arr)
        # Vega por unidad de vol (no por punto): es la derivada exacta de f.
        sq = np.sqrt(np.maximum(t, 1e-12))
        with np.errstate(divide="ignore", invalid="ignore"):
            d1 = (np.log(s / k) + (r - q + 0.5 * sigma * sigma) * np.maximum(t, 1e-12)) / (
                np.maximum(sigma, 1e-12) * sq
            )
        vega_unit = s * disc_q * _phi(d1) * sq
        with np.errstate(divide="ignore", invalid="ignore"):
            newton = sigma - err / vega_unit
        ok_newton = (
            active
            & np.isfinite(newton)
            & (vega_unit > 1e-12)
            & (newton > lo_arr)
            & (newton < hi_arr)
        )
        bisect = active & ~ok_newton
        n_newton += int(ok_newton.sum())
        n_bisect += int(bisect.sum())
        sigma = np.where(ok_newton, newton, sigma)
        sigma = np.where(bisect, 0.5 * (lo_arr + hi_arr), sigma)

    if active.any():
        # Sin converger en max_iter: se entrega el punto medio del corchete si
        # ya es suficientemente estrecho; si no, NaN (F9).
        narrow = active & ((hi_arr - lo_arr) < 1e-6)
        out[narrow] = 0.5 * (lo_arr + hi_arr)[narrow]

    out = out.reshape(out_shape)
    if return_diagnostics:
        diagnostics = {
            "newton_steps": n_newton,
            "bisection_steps": n_bisect,
            "converged": np.isfinite(out),
        }
        return out, diagnostics
    return out


def implied_forward(
    strikes: Any,
    call_mids: Any,
    put_mids: Any,
    tau: float,
    *,
    weights: Any = None,
    rate: float | None = None,
    spot: float | None = None,
) -> dict[str, float]:
    """Forward implícito y factor de descuento por regresión de paridad put-call.

    Para cada strike con call y put cotizadas, ``C(K) - P(K) = DF*F - DF*K``
    es una recta en `K`: la pendiente estima ``-DF`` y la ordenada ``DF*F``
    (`docs/research/options_signals.md` §2.1; construcción estándar del nivel
    forward del Cboe VIX). Con `weights` (p. ej. ``1/(spread_c + spread_p)``)
    la regresión es WLS y mandan los strikes líquidos.

    Devuelve ``{"forward", "discount_factor", "implied_rate", "implied_borrow"}``.
    `implied_borrow = rate - ln(F/spot)/tau` (comisión de préstamo implícita,
    señal en sí misma) solo si se pasan `rate` y `spot`. Con menos de 3 strikes
    o una pendiente no negativa (datos corruptos) lanza `DataQualityError`.
    """
    k = np.asarray(strikes, dtype=float)
    diff = np.asarray(call_mids, dtype=float) - np.asarray(put_mids, dtype=float)
    w = np.ones_like(k) if weights is None else np.asarray(weights, dtype=float)
    valid = np.isfinite(k) & np.isfinite(diff) & np.isfinite(w) & (w > 0)
    k, diff, w = k[valid], diff[valid], w[valid]
    if len(k) < 3 or len(np.unique(k)) < 3:
        msg = f"implied_forward necesita >=3 strikes distintos con call y put; hay {len(k)}"
        raise DataQualityError(msg)
    if tau <= 0:
        msg = f"tau debe ser > 0; recibido {tau}"
        raise DataQualityError(msg)
    sw = np.sqrt(w)
    design = np.column_stack([np.ones_like(k), k])
    coef, *_ = np.linalg.lstsq(design * sw[:, None], diff * sw, rcond=None)
    intercept, slope = float(coef[0]), float(coef[1])
    df_hat = -slope
    if df_hat <= 0 or df_hat > 1.5:
        msg = (
            f"la regresión de paridad da un factor de descuento imposible ({df_hat:.4f}); "
            "la cadena está corrupta o mezcla vencimientos"
        )
        raise DataQualityError(msg)
    forward = intercept / df_hat
    out = {
        "forward": forward,
        "discount_factor": df_hat,
        "implied_rate": -math.log(min(df_hat, 1.0 - 1e-12)) / tau if df_hat < 1 else 0.0,
        "implied_borrow": float("nan"),
    }
    if rate is not None and spot is not None and spot > 0 and forward > 0:
        out["implied_borrow"] = rate - math.log(forward / spot) / tau
    return out


def implied_forward_table(
    chain: pd.DataFrame,
    *,
    rate: float | None = None,
    moneyness_band: float = 0.10,
) -> pd.DataFrame:
    """Tabla ``(date, ticker, expiry) -> forward, DF, implied_borrow``.

    Aplica `implied_forward` por vencimiento casando calls y puts del mismo
    strike, restringido a ``|K/S - 1| <= moneyness_band`` (donde el valor
    temporal de ambas patas es máximo y el ejercicio anticipado americano es
    despreciable, §3.5 del informe de opciones). Los vencimientos sin pares
    suficientes se omiten con un aviso; si ninguno sobrevive,
    `InsufficientHistory`.
    """
    frame = normalize_chain(chain)
    if "mid" not in frame.columns:
        frame = frame.assign(mid=(frame["bid"] + frame["ask"]) / 2.0)
    rows: list[dict[str, Any]] = []
    for (day, ticker, expiry), sub in frame.groupby(["date", "ticker", "expiry"], sort=True):
        calls = sub[sub["right"] == "C"].set_index("strike")["mid"]
        puts = sub[sub["right"] == "P"].set_index("strike")["mid"]
        strikes = calls.index.intersection(puts.index)
        spot = float(sub["spot"].iloc[0]) if "spot" in sub.columns else float("nan")
        if math.isfinite(spot) and spot > 0:
            strikes = strikes[abs(strikes / spot - 1.0) <= moneyness_band]
        tau = max((pd.Timestamp(expiry) - pd.Timestamp(day)).days, 1) / 365.0
        try:
            est = implied_forward(
                strikes.to_numpy(),
                calls.loc[strikes].to_numpy(),
                puts.loc[strikes].to_numpy(),
                tau,
                rate=rate,
                spot=spot if math.isfinite(spot) else None,
            )
        except DataQualityError as exc:
            logger.warning("implied_forward_table: %s/%s %s omitido (%s)",
                           ticker, pd.Timestamp(expiry).date(), pd.Timestamp(day).date(), exc)
            continue
        rows.append(
            {
                "date": day,
                "ticker": ticker,
                "expiry": expiry,
                "tau_years": tau,
                "forward": est["forward"],
                "discount_factor": est["discount_factor"],
                "implied_rate": est["implied_rate"],
                "implied_borrow": est["implied_borrow"],
                "n_pairs": int(len(strikes)),
            }
        )
    if not rows:
        msg = "ningún vencimiento tiene pares call/put suficientes para el forward implícito"
        raise InsufficientHistory(msg)
    return pd.DataFrame(rows).sort_values(["date", "ticker", "expiry"]).reset_index(drop=True)


# ===========================================================================
# 2. Esquema canónico
# ===========================================================================


_RIGHT_MAP = {
    "c": "C", "call": "C", "calls": "C", "C": "C",
    "p": "P", "put": "P", "puts": "P", "P": "P",
}


def normalize_chain(frame: pd.DataFrame) -> pd.DataFrame:
    """Valida y normaliza una cadena al esquema canónico.

    Obliga a las columnas `CHAIN_COLUMNS` (las extra se conservan), normaliza
    `right` a ``{"C","P"}``, tickers al formato del repo y fechas a tz-naive.
    Estructuras rotas (strike <= 0, right desconocido, vencimiento anterior a
    la fecha) son `DataQualityError`: una cadena corrupta que entra en la capa
    de señales produce un `iv_skew` con sesgo, no un error visible.
    """
    if frame is None or len(frame) == 0:
        msg = "la cadena está vacía"
        raise InsufficientHistory(msg)
    missing = [c for c in CHAIN_COLUMNS if c not in frame.columns]
    if missing:
        msg = f"faltan columnas del esquema canónico de cadenas: {missing}"
        raise DataQualityError(msg)
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["expiry"] = pd.to_datetime(out["expiry"]).dt.normalize()
    out["ticker"] = [normalize_ticker(str(t)) for t in out["ticker"]]
    rights = [
        _RIGHT_MAP.get(str(r).strip().lower(), _RIGHT_MAP.get(str(r).strip(), None))
        for r in out["right"]
    ]
    if any(r is None for r in rights):
        bad = sorted({str(r) for r, m in zip(out["right"], rights, strict=True) if m is None})
        msg = f"valores de `right` no reconocidos: {bad} (se espera C/P/call/put)"
        raise DataQualityError(msg)
    out["right"] = rights
    for col in ("strike", "bid", "ask", "last", "volume", "open_interest", "iv", "delta"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if (out["strike"] <= 0).any():
        msg = "hay strikes <= 0; la cadena está corrupta"
        raise DataQualityError(msg)
    if (out["expiry"] < out["date"]).any():
        msg = "hay contratos con vencimiento anterior a la fecha del snapshot"
        raise DataQualityError(msg)
    ordered = [*CHAIN_COLUMNS, *[c for c in out.columns if c not in CHAIN_COLUMNS]]
    return (
        out[ordered]
        .sort_values(["date", "ticker", "expiry", "right", "strike"])
        .reset_index(drop=True)
    )


def chain_availability(
    frame: pd.DataFrame, cal: TradingCalendar | None = None
) -> pd.DataFrame:
    """Añade `available_at` y `oi_available_at` a una cadena canónica.

    - `available_at` = **16:15 ET del propio día** (cierre + margen para el
      print consolidado): momento en que precios, IV y volumen son conocibles.
    - `oi_available_at` = **08:00 ET de la siguiente sesión**: la OCC publica
      el open interest en su ciclo nocturno y está disponible pre-apertura de
      `t+1` (`docs/research/data_sources.md` §7.3). Usar el OI de `t` como
      conocido en `t` es un día entero de look-ahead.
    """
    calendar = cal or get_calendar()
    out = frame.copy()
    unique_days = pd.DatetimeIndex(out["date"].unique())
    close_map = {
        d: pd.Timestamp(eastern_to_utc(dt.datetime.combine(d.date(), dt.time(16, 15))))
        for d in unique_days
    }
    oi_map = {
        d: pd.Timestamp(
            eastern_to_utc(
                dt.datetime.combine(calendar.next_session(d.date()), dt.time(8, 0))
            )
        )
        for d in unique_days
    }
    out["available_at"] = out["date"].map(close_map)
    out["oi_available_at"] = out["date"].map(oi_map)
    return out


def filter_chain(
    chain: pd.DataFrame,
    *,
    max_rel_spread: float = 0.50,
    min_mid: float = 0.05,
    max_days: int = 365,
    moneyness_band: tuple[float, float] | None = None,
    require_open_interest: bool = False,
    max_dropped_fraction: float = 0.40,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Filtros de calidad F1-F8 de `docs/research/options_signals.md` §2.3.

    Devuelve ``(cadena filtrada, recuento de filas eliminadas por filtro)``.
    El informe exige registrar cuántos contratos elimina cada filtro y anular
    el ticker-día cuando la poda supera el 40 %: aquí, si la fracción global
    eliminada supera `max_dropped_fraction`, se lanza `DataQualityError` para
    que el llamante decida (NaN en la señal, no un valor sobre 3 contratos).
    Los filtros F1-F4 solo aplican a filas **con** bid/ask: una fila con
    precios NaN (proveedores históricos sin cotización) no se elimina por
    ellos, se deja pasar y queda documentada en `counts["sin_cotizacion"]`.
    """
    frame = normalize_chain(chain)
    n0 = len(frame)
    counts: dict[str, int] = {}
    has_quote = frame["bid"].notna() & frame["ask"].notna()
    counts["sin_cotizacion"] = int((~has_quote).sum())

    mid = (frame["bid"] + frame["ask"]) / 2.0
    keep = pd.Series(True, index=frame.index)

    f1 = has_quote & (frame["bid"] <= 0)
    counts["F1_bid_cero"] = int(f1.sum())
    keep &= ~f1
    f2 = has_quote & (frame["ask"] <= frame["bid"])
    counts["F2_cruce"] = int((f2 & keep).sum())
    keep &= ~f2
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = (frame["ask"] - frame["bid"]) / mid
    f3 = has_quote & (rel > max_rel_spread)
    counts["F3_horquilla"] = int((f3 & keep).sum())
    keep &= ~f3
    f4 = has_quote & (mid < min_mid)
    counts["F4_mid_minimo"] = int((f4 & keep).sum())
    keep &= ~f4
    days = (frame["expiry"] - frame["date"]).dt.days
    f6 = (days <= 0) | (days > max_days)
    counts["F6_plazo"] = int((f6 & keep).sum())
    keep &= ~f6
    if moneyness_band is not None:
        if "spot" not in frame.columns:
            msg = "el filtro de moneyness necesita la columna `spot` en la cadena"
            raise DataQualityError(msg)
        ratio = frame["strike"] / frame["spot"]
        f7 = ~ratio.between(moneyness_band[0], moneyness_band[1])
        counts["F7_moneyness"] = int((f7 & keep).sum())
        keep &= ~f7
    if require_open_interest:
        f8 = ~(frame["open_interest"] > 0)
        counts["F8_open_interest"] = int((f8 & keep).sum())
        keep &= ~f8

    out = frame[keep].reset_index(drop=True)
    dropped = 1.0 - len(out) / n0
    if dropped > max_dropped_fraction:
        msg = (
            f"los filtros de calidad eliminan el {100 * dropped:.0f}% de la cadena "
            f"(> {100 * max_dropped_fraction:.0f}%); este ticker-día no debe generar "
            f"señal sino NaN. Recuento por filtro: {counts}"
        )
        raise DataQualityError(msg)
    return out, counts


_OCC_RE = re.compile(r"^(?P<root>[A-Z.]{1,6})(?P<date>\d{6})(?P<right>[CP])(?P<strike>\d{8})$")


def occ_symbol(ticker: Ticker, expiry: DateLike, right: str, strike: float) -> str:
    """Símbolo OCC de 21 caracteres: ``AAPL  200918C00112500`` sin los espacios.

    Formato OSI: raíz + AAMMDD + C/P + strike*1000 con 8 dígitos. Es la clave
    de contrato de Polygon (`O:{occ}`), Tradier y los ficheros OPRA.
    """
    r = _RIGHT_MAP.get(str(right).strip().lower())
    if r is None:
        msg = f"right no reconocido para símbolo OCC: {right!r}"
        raise ConfigError(msg)
    day = pd.Timestamp(expiry)
    milli = round(float(strike) * 1000)
    return f"{normalize_ticker(ticker)}{day.strftime('%y%m%d')}{r}{milli:08d}"


def parse_occ_symbol(symbol: str) -> dict[str, Any]:
    """Inversa de `occ_symbol`. Lanza `DataQualityError` si no es un OCC válido."""
    text = symbol.strip().upper().removeprefix("O:").replace(" ", "")
    match = _OCC_RE.match(text)
    if match is None:
        msg = f"símbolo OCC no reconocido: {symbol!r}"
        raise DataQualityError(msg)
    return {
        "ticker": normalize_ticker(match["root"]),
        "expiry": dt.datetime.strptime(match["date"], "%y%m%d").date(),
        "right": match["right"],
        "strike": int(match["strike"]) / 1000.0,
    }


# ===========================================================================
# 3. Base de proveedores
# ===========================================================================


class OptionsProviderBase(BaseProvider):
    """Base de los adaptadores de cadenas.

    API pública: `chain(ticker, asof=None)` -> cadena canónica con columnas de
    disponibilidad PIT. `asof=None` significa "el snapshot más reciente que el
    proveedor pueda servir". Política de fallo idéntica al resto de la capa de
    datos: credenciales ausentes -> `ProviderUnavailable`; sin contratos ->
    `InsufficientHistory`; payload malformado -> `DataQualityError`.
    """

    name = "options_base"
    kinds: tuple[str, ...] = (OPTIONS_KIND,)

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        http: HttpClient | None = None,
        cache: DiskCache | None = None,
        calendar: TradingCalendar | None = None,
    ) -> None:
        super().__init__(settings=settings, http=http)
        self.cache = cache
        self._calendar = calendar

    @property
    def calendar(self) -> TradingCalendar:
        if self._calendar is None:
            self._calendar = get_calendar()
        return self._calendar

    def _credential(self, key: str) -> str:
        value = self.settings.env(key)
        if not value:
            raise ProviderUnavailable(
                self.name, "falta la credencial en el entorno", missing_env=[key]
            )
        return value

    def _require_credentials(self) -> None:
        missing = self.missing_env()
        if missing:
            raise ProviderUnavailable(
                self.name,
                "faltan credenciales para pedir cadenas de opciones",
                missing_env=missing,
            )

    def chain(self, ticker: Ticker, asof: DateLike | None = None) -> pd.DataFrame:
        """Cadena canónica de `ticker` en la sesión `asof` (o la más reciente)."""
        symbol = normalize_ticker(ticker)
        day = None if asof is None else pd.Timestamp(asof).normalize()
        self._require_credentials()
        raw = self._fetch_chain(symbol, day)
        if raw is None or len(raw) == 0:
            when = "snapshot actual" if day is None else day.date().isoformat()
            msg = f"{self.name}: sin contratos para {symbol} en {when}"
            raise InsufficientHistory(msg)
        frame = normalize_chain(raw)
        frame["source"] = self.name
        return chain_availability(frame, self.calendar)

    def _fetch_chain(self, ticker: Ticker, asof: pd.Timestamp | None) -> pd.DataFrame:
        raise NotImplementedError


# ===========================================================================
# 4. Sintético
# ===========================================================================


class SyntheticOptionsProvider(OptionsProviderBase):
    """Cadenas del generador sintético. Sin red ni credenciales.

    `SyntheticMarket.options_chain` valora cada contrato con BSM sobre una
    superficie con smile, término y varianza de evento (Dubinsky-Johannes
    2006), de modo que la inversión con `implied_vol` de este módulo debe
    recuperar la IV publicada: es la verdad-terreno del solver.
    """

    name = "synthetic"
    kinds: tuple[str, ...] = (OPTIONS_KIND,)

    def __init__(
        self,
        market: SyntheticMarket | None = None,
        *,
        settings: Settings | None = None,
        calendar: TradingCalendar | None = None,
        **market_kwargs: Any,
    ) -> None:
        super().__init__(settings=settings, calendar=calendar)
        self._market = market
        self._market_kwargs = dict(market_kwargs)

    @property
    def market(self) -> SyntheticMarket:
        if self._market is None:
            kwargs = {"seed": self.settings.seed, **self._market_kwargs}
            self._market = SyntheticMarket(**kwargs)
        return self._market

    def available(self) -> bool:
        return True

    def _fetch_chain(self, ticker: Ticker, asof: pd.Timestamp | None) -> pd.DataFrame:
        raw = self.market.options_chain(tickers=ticker, asof=asof)
        out = raw.rename(columns={"as_of": "date"})
        out["last"] = out["mid"]
        keep = [
            "date", "ticker", "expiry", "strike", "right", "bid", "ask", "last",
            "volume", "open_interest", "iv", "delta", "mid", "spot", "gamma", "vega",
        ]
        return out[keep]


# ===========================================================================
# 5. Polygon
# ===========================================================================


class PolygonOptionsProvider(OptionsProviderBase):
    """Polygon.io opciones: snapshot con IV/griegas y reconstrucción histórica.

    Endpoints (`docs/research/data_sources.md` §7.2)::

        GET /v3/snapshot/options/{underlying}?limit=250
            -> {"results": [{"details": {"contract_type", "expiration_date",
                "strike_price", "ticker": "O:..."}, "day": {"close", "volume"},
                "greeks": {"delta", ...}, "implied_volatility",
                "last_quote": {"bid", "ask"}, "last_trade": {"price"},
                "open_interest", "underlying_asset": {"price"}}],
                "next_url": ...}
        GET /v3/reference/options/contracts?underlying_ticker=&as_of=&expired=
            -> contratos QUE EXISTÍAN en `as_of` (clave: evita el sesgo de
               mirar hoy la lista de contratos, §7.2)
        GET /v2/aggs/ticker/O:{occ}/range/1/day/{d}/{d}   (barra del contrato)
        GET /v2/aggs/ticker/{t}/range/1/day/{d}/{d}       (cierre del subyacente)

    Modo snapshot (`asof=None`): la cadena viene completa con IV y griegas del
    proveedor. Modo histórico (`asof` pasado): Polygon no versiona el snapshot,
    así que se reconstruye con `contracts as_of` + una barra diaria por
    contrato; **no hay bid/ask ni open interest históricos** (quedan NaN), el
    `last` es el cierre del contrato y la IV y la delta se calculan con
    `implied_vol`/`bs_greeks` propios a partir del cierre del subyacente y de
    `risk_free_rate` (plano; la curva de tipos real es una mejora pendiente
    anotada en el docstring). Cada contrato cuesta una petición: el parámetro
    `max_contracts` corta la reconstrucción antes de quemar la cuota.
    """

    name = "polygon"
    kinds: tuple[str, ...] = (OPTIONS_KIND,)
    BASE = "https://api.polygon.io"

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        http: HttpClient | None = None,
        cache: DiskCache | None = None,
        calendar: TradingCalendar | None = None,
        risk_free_rate: float = 0.03,
        strike_band: float = 0.5,
        max_days: int = 400,
        max_contracts: int = 250,
    ) -> None:
        super().__init__(settings=settings, http=http, cache=cache, calendar=calendar)
        self.risk_free_rate = float(risk_free_rate)
        self.strike_band = float(strike_band)
        self.max_days = int(max_days)
        self.max_contracts = int(max_contracts)

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._credential('POLYGON_API_KEY')}"}

    def _paged(self, url: str, params: dict[str, Any] | None) -> list[dict[str, Any]]:
        headers = self._auth_headers()
        rows: list[dict[str, Any]] = []
        next_url: str | None = url
        next_params = params
        while next_url:
            payload = self.http.get_json(
                next_url, params=next_params, headers=headers, allow_status=(404,)
            )
            if not isinstance(payload, dict):
                msg = f"polygon: respuesta no-objeto en {next_url}"
                raise DataQualityError(msg)
            if payload.get("status") == "ERROR":
                msg = f"polygon: status=ERROR — {payload.get('error') or payload.get('message')}"
                raise DataQualityError(msg)
            rows.extend(payload.get("results") or [])
            next_url = payload.get("next_url")
            next_params = None
        return rows

    def _fetch_chain(self, ticker: Ticker, asof: pd.Timestamp | None) -> pd.DataFrame:
        if asof is None:
            return self._snapshot_chain(ticker)
        return self._historical_chain(ticker, asof)

    def _snapshot_chain(self, ticker: Ticker) -> pd.DataFrame:
        results = self._paged(f"{self.BASE}/v3/snapshot/options/{ticker}", {"limit": 250})
        today = pd.Timestamp(dt.datetime.now(dt.UTC).date())
        rows: list[dict[str, Any]] = []
        for item in results:
            details = item.get("details") or {}
            if not details.get("expiration_date"):
                continue
            quote = item.get("last_quote") or {}
            day = item.get("day") or {}
            greeks = item.get("greeks") or {}
            underlying = item.get("underlying_asset") or {}
            rows.append(
                {
                    "date": today,
                    "ticker": ticker,
                    "expiry": details.get("expiration_date"),
                    "strike": details.get("strike_price"),
                    "right": details.get("contract_type"),
                    "bid": quote.get("bid"),
                    "ask": quote.get("ask"),
                    "last": (item.get("last_trade") or {}).get("price", day.get("close")),
                    "volume": day.get("volume"),
                    "open_interest": item.get("open_interest"),
                    "iv": item.get("implied_volatility"),
                    "delta": greeks.get("delta"),
                    "gamma": greeks.get("gamma"),
                    "vega": greeks.get("vega"),
                    "spot": underlying.get("price"),
                }
            )
        return pd.DataFrame(rows)

    def _underlying_close(self, ticker: Ticker, day: pd.Timestamp) -> float:
        iso = day.date().isoformat()
        rows = self._paged(
            f"{self.BASE}/v2/aggs/ticker/{ticker}/range/1/day/{iso}/{iso}",
            {"adjusted": "false"},
        )
        if not rows:
            msg = f"polygon: sin cierre del subyacente {ticker} en {iso}"
            raise InsufficientHistory(msg)
        return float(rows[0]["c"])

    def _historical_chain(self, ticker: Ticker, asof: pd.Timestamp) -> pd.DataFrame:
        iso = asof.date().isoformat()
        spot = self._underlying_close(ticker, asof)
        contracts = self._paged(
            f"{self.BASE}/v3/reference/options/contracts",
            {
                "underlying_ticker": ticker,
                "as_of": iso,
                "expired": "true",
                "limit": 1000,
            },
        )
        chosen: list[dict[str, Any]] = []
        for c in contracts:
            strike = c.get("strike_price")
            expiry = c.get("expiration_date")
            if strike is None or not expiry:
                continue
            days = (pd.Timestamp(expiry) - asof).days
            if days <= 0 or days > self.max_days:
                continue
            if abs(float(strike) / spot - 1.0) > self.strike_band:
                continue
            chosen.append(c)
        if len(chosen) > self.max_contracts:
            msg = (
                f"polygon: la reconstrucción histórica de {ticker}@{iso} necesita "
                f"{len(chosen)} contratos (> max_contracts={self.max_contracts}); "
                "estrecha strike_band/max_days o sube max_contracts a sabiendas "
                "de que cada contrato es una petición"
            )
            raise ConfigError(msg)
        rows: list[dict[str, Any]] = []
        for c in chosen:
            occ = str(c.get("ticker", "")).removeprefix("O:")
            aggs = self._paged(
                f"{self.BASE}/v2/aggs/ticker/O:{occ}/range/1/day/{iso}/{iso}",
                {"adjusted": "false"},
            )
            close = float(aggs[0]["c"]) if aggs else np.nan
            volume = float(aggs[0]["v"]) if aggs else np.nan
            rows.append(
                {
                    "date": asof,
                    "ticker": ticker,
                    "expiry": c["expiration_date"],
                    "strike": float(c["strike_price"]),
                    "right": c.get("contract_type"),
                    "bid": np.nan,
                    "ask": np.nan,
                    "last": close,
                    "volume": volume,
                    "open_interest": np.nan,
                    "iv": np.nan,
                    "delta": np.nan,
                    "spot": spot,
                }
            )
        frame = pd.DataFrame(rows)
        if not len(frame):
            return frame
        tau = (pd.to_datetime(frame["expiry"]) - asof).dt.days.to_numpy() / 365.0
        is_call = (frame["right"].astype(str).str.lower().str.startswith("c")).to_numpy()
        iv = implied_vol(
            frame["last"].to_numpy(), spot, frame["strike"].to_numpy(), tau,
            self.risk_free_rate, 0.0, is_call,
        )
        frame["iv"] = iv
        greeks = bs_greeks(
            spot, frame["strike"].to_numpy(), tau, iv, self.risk_free_rate, 0.0, is_call
        )
        frame["delta"] = greeks.delta
        frame["gamma"] = greeks.gamma
        frame["vega"] = greeks.vega
        return frame


# ===========================================================================
# 6. ORATS
# ===========================================================================


class ORATSOptionsProvider(OptionsProviderBase):
    """ORATS `hist/strikes`: superficie EOD por strike desde 2007.

    Endpoint (`docs/research/data_sources.md` §7.2)::

        GET https://api.orats.io/datav2/hist/strikes?token=&ticker=&tradeDate=
            -> {"data": [{"ticker", "tradeDate", "expirDate", "strike",
                "stockPrice", "callVolume", "callOpenInterest", "callBidPrice",
                "callAskPrice", "putVolume", "putOpenInterest", "putBidPrice",
                "putAskPrice", "callMidIv", "putMidIv", "delta", "gamma",
                "vega", ...}]}

    Cada fila de ORATS es un par (strike, vencimiento): aquí se separa en dos
    filas canónicas C/P. El `delta` que publica ORATS es el de la **call**; la
    delta de la put se calcula con `bs_greeks` propio a partir de `putMidIv`
    (dividendo continuo 0: sesgo de décimas de delta cerca del dinero,
    homogéneo en sección cruzada). La cuota de ORATS es **mensual**
    (~20.000 req/mes en el plan Delayed): pedir solo fechas dentro de ventanas
    de evento, como aconseja el informe §7.2.
    """

    name = "orats"
    kinds: tuple[str, ...] = (OPTIONS_KIND,)
    BASE = "https://api.orats.io/datav2"

    def __init__(self, *, risk_free_rate: float = 0.03, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.risk_free_rate = float(risk_free_rate)

    def _fetch_chain(self, ticker: Ticker, asof: pd.Timestamp | None) -> pd.DataFrame:
        params: dict[str, Any] = {
            "token": self._credential("ORATS_TOKEN"),
            "ticker": ticker,
        }
        if asof is not None:
            params["tradeDate"] = asof.date().isoformat()
        payload = self.http.get_json(f"{self.BASE}/hist/strikes", params=params)
        if not isinstance(payload, dict) or "data" not in payload:
            msg = "orats: la respuesta no contiene `data`"
            raise DataQualityError(msg)
        rows: list[dict[str, Any]] = []
        for row in payload["data"] or []:
            trade_date = row.get("tradeDate")
            expiry = row.get("expirDate")
            strike = row.get("strike")
            if not trade_date or not expiry or strike is None:
                continue
            spot = row.get("stockPrice")
            tau = max((pd.Timestamp(expiry) - pd.Timestamp(trade_date)).days, 1) / 365.0
            put_iv = row.get("putMidIv")
            if put_iv is not None and spot:
                put_delta = float(
                    bs_greeks(
                        float(spot), float(strike), tau, float(put_iv),
                        self.risk_free_rate, 0.0, False,
                    ).delta
                )
            else:
                put_delta = np.nan
            common = {
                "date": trade_date,
                "ticker": ticker,
                "expiry": expiry,
                "strike": strike,
                "spot": spot,
            }
            rows.append(
                {
                    **common,
                    "right": "C",
                    "bid": row.get("callBidPrice"),
                    "ask": row.get("callAskPrice"),
                    "last": np.nan,
                    "volume": row.get("callVolume"),
                    "open_interest": row.get("callOpenInterest"),
                    "iv": row.get("callMidIv"),
                    "delta": row.get("delta"),
                }
            )
            rows.append(
                {
                    **common,
                    "right": "P",
                    "bid": row.get("putBidPrice"),
                    "ask": row.get("putAskPrice"),
                    "last": np.nan,
                    "volume": row.get("putVolume"),
                    "open_interest": row.get("putOpenInterest"),
                    "iv": put_iv,
                    "delta": put_delta,
                }
            )
        return pd.DataFrame(rows)


# ===========================================================================
# 7. Tradier
# ===========================================================================


class TradierOptionsProvider(OptionsProviderBase):
    """Tradier: cadena del **snapshot actual**, con griegas cortesía de ORATS.

    Endpoints (`docs/research/data_sources.md` §7.2)::

        GET /v1/markets/options/expirations?symbol=&includeAllRoots=true
            -> {"expirations": {"date": ["2025-09-19", ...]}}
        GET /v1/markets/options/chains?symbol=&expiration=&greeks=true
            -> {"options": {"option": [{"strike", "option_type", "bid", "ask",
                "last", "volume", "open_interest", "expiration_date",
                "greeks": {"delta", "gamma", "vega", "mid_iv"}}]}}

    **Veredicto del informe §7.2, que este adaptador hace cumplir:** Tradier no
    ofrece snapshots históricos de cadena; pedir `asof` en el pasado lanza
    `ProviderUnavailable` en vez de devolver en silencio la cadena de hoy con
    fecha de ayer, que sería un look-ahead fabricado por el adaptador.
    """

    name = "tradier"
    kinds: tuple[str, ...] = (OPTIONS_KIND,)
    BASE = "https://api.tradier.com/v1"
    max_expirations: int = 12

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._credential('TRADIER_ACCESS_TOKEN')}",
            "Accept": "application/json",
        }

    def _fetch_chain(self, ticker: Ticker, asof: pd.Timestamp | None) -> pd.DataFrame:
        today = pd.Timestamp(dt.datetime.now(dt.UTC).date())
        if asof is not None and asof.normalize() != today:
            raise ProviderUnavailable(
                self.name,
                "tradier solo sirve el snapshot actual de la cadena; no hay "
                f"histórico para {asof.date().isoformat()} (usa ORATS o Polygon)",
            )
        headers = self._headers()
        payload = self.http.get_json(
            f"{self.BASE}/markets/options/expirations",
            params={"symbol": ticker, "includeAllRoots": "true", "strikes": "false"},
            headers=headers,
        )
        block = (payload or {}).get("expirations") if isinstance(payload, dict) else None
        dates = (block or {}).get("date") if isinstance(block, dict) else None
        if not dates:
            return pd.DataFrame()
        if isinstance(dates, str):
            dates = [dates]
        rows: list[dict[str, Any]] = []
        for expiry in dates[: self.max_expirations]:
            chain = self.http.get_json(
                f"{self.BASE}/markets/options/chains",
                params={"symbol": ticker, "expiration": expiry, "greeks": "true"},
                headers=headers,
            )
            options = (chain or {}).get("options") if isinstance(chain, dict) else None
            listing = (options or {}).get("option") if isinstance(options, dict) else None
            if not listing:
                continue
            if isinstance(listing, dict):
                listing = [listing]
            for opt in listing:
                greeks = opt.get("greeks") or {}
                rows.append(
                    {
                        "date": today,
                        "ticker": ticker,
                        "expiry": opt.get("expiration_date", expiry),
                        "strike": opt.get("strike"),
                        "right": opt.get("option_type"),
                        "bid": opt.get("bid"),
                        "ask": opt.get("ask"),
                        "last": opt.get("last"),
                        "volume": opt.get("volume"),
                        "open_interest": opt.get("open_interest"),
                        "iv": greeks.get("mid_iv"),
                        "delta": greeks.get("delta"),
                        "gamma": greeks.get("gamma"),
                        "vega": greeks.get("vega"),
                    }
                )
        return pd.DataFrame(rows)


# ===========================================================================
# 8. Registro y fachadas
# ===========================================================================


DEFAULT_OPTIONS_PRIORITIES: dict[str, int] = {
    "orats": 40,       # superficie EOD desde 2007: la mejor fuente histórica
    "polygon": 30,     # 2 años de histórico reconstruible + snapshot con IV
    "tradier": 10,     # solo snapshot actual
    "synthetic": 0,    # red de seguridad sin red (contrato §0.5)
}


def register_options_providers(
    registry: ProviderRegistry | None = None,
    *,
    settings: Settings | None = None,
    market: SyntheticMarket | None = None,
    include: Sequence[str] | None = None,
    priorities: Mapping[str, int] | None = None,
) -> ProviderRegistry:
    """Registra los proveedores de cadenas de opciones en el registro."""
    reg = registry or get_registry()
    cfg = settings or get_settings()
    chosen = dict(DEFAULT_OPTIONS_PRIORITIES)
    if include is not None:
        unknown = sorted(set(include) - set(chosen))
        if unknown:
            msg = f"proveedores de opciones desconocidos en include: {unknown}"
            raise ConfigError(msg)
        chosen = {k: v for k, v in chosen.items() if k in set(include)}
    if priorities:
        chosen.update({k: int(v) for k, v in priorities.items() if k in chosen})

    factories: dict[str, Any] = {
        "orats": lambda: ORATSOptionsProvider(settings=cfg),
        "polygon": lambda: PolygonOptionsProvider(settings=cfg),
        "tradier": lambda: TradierOptionsProvider(settings=cfg),
        "synthetic": lambda: SyntheticOptionsProvider(market, settings=cfg),
    }
    for name, priority in chosen.items():
        reg.register(OPTIONS_KIND, factories[name](), priority, replace=True)
    return reg


def get_option_chain(
    ticker: Ticker,
    asof: DateLike | None = None,
    *,
    registry: ProviderRegistry | None = None,
) -> pd.DataFrame:
    """Fachada: cadena canónica del mejor proveedor disponible, con fallback."""
    reg = registry or get_registry()
    when = "latest" if asof is None else pd.Timestamp(asof).date().isoformat()
    return reg.call(
        OPTIONS_KIND,
        lambda p: p.chain(ticker, asof),  # type: ignore[attr-defined]
        description=f"chain[{normalize_ticker(ticker)}@{when}]",
    )
