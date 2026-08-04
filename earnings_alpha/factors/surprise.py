"""Familia de factores de sorpresa de resultados: SUE, SURGE, DOUBLE y PEAD.

Implementa las secciones §2-§4 de `docs/research/fundamental_factors.md` con sus
fórmulas exactas y sus salvaguardas:

- **SUE de series temporales** (Foster 1977; Foster, Olsen y Shevlin 1984;
  Bernard y Thomas 1989, 1990): paseo aleatorio estacional con deriva sobre las
  últimas **13 observaciones trimestrales consecutivas** (con 12 se lanza
  `InsufficientHistory`, verificado en el informe §2.1). Denominadores: sigma de
  las 8 sorpresas estacionales (`SurpriseBasis.SIGMA`, con suelo obligatorio
  `sigma_eff = max(sigma, k·P)` porque sin suelo el cociente explota con
  beneficios estables, §2.2) o el precio previo (`SurpriseBasis.PRICE`, §2.3).
- **SUE de analistas** (Livnat y Mendenhall 2006): sorpresa frente al consenso,
  estandarizada por la sigma de las 8 sorpresas históricas, por el precio o por
  la dispersión de analistas con suelo (§2.5-2.6). Livnat y Mendenhall
  documentan que el drift posterior es mayor con sorpresas de analistas que con
  modelos de series temporales: es la variante primaria cuando hay consenso.
- **SURGE, sorpresa de ingresos** (Jegadeesh y Livnat 2006): mismo modelo de
  expectativas sobre ingresos **por acción** — con ingresos totales el factor
  mide la política de recompras y puede invertir el signo (§3, verificado en el
  informe con una recompra del 10%: +0.126 por acción vs -0.814 en total—).
- **DOUBLE** (Jegadeesh y Livnat 2006, FAJ): mínimo de |z(SUE)| y |z(SURGE)|
  cuando comparten signo, cero cuando discrepan (§3).
- **PEAD** (Ball y Brown 1968; Bernard y Thomas 1989; erosión: Martineau 2021,
  *Rest in Peace PEAD*): el z-score de la sorpresa del último evento, vivo
  durante ``0 < s <= H`` sesiones desde el `tradable_date` y cero después, con
  decaimiento exponencial parametrizable (§4.1). El horizonte por defecto es
  **corto** (5 sesiones): en valores grandes y líquidos el drift clásico está
  documentado como inexistente desde ~2006 (§4.3).

Point-in-time
-------------
Toda proyección a panel pasa por `pit.tradable_date` / `pit.asof_join`
(mediante `factors.base.spread_event_values`): un anuncio AMC no alimenta la
señal hasta la sesión siguiente, y el precio deflactor es el cierre de la
sesión **anterior** a la fecha negociable (`P_{t^-}`, §2.3), nunca el del
propio día del anuncio.

Datos reales
------------
`load_consensus_events` construye la tabla de eventos desde
``data/external/consenso/consenso_master.parquet`` derivando la sesión BMO/AMC
de `report_time`. Lleva su advertencia PIT en el docstring: es el consenso
FINAL previo al anuncio, sin vintages, y con supervivencia parcial de tickers.
"""

from __future__ import annotations

import datetime as dt
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar, Final

import numpy as np
import pandas as pd

from earnings_alpha.config import Settings, get_settings
from earnings_alpha.errors import (
    ConfigError,
    DataQualityError,
    InsufficientHistory,
    ProviderUnavailable,
)
from earnings_alpha.factors.base import (
    FactorContext,
    register_factor,
    require_columns,
    spread_event_frame,
    spread_event_values,
)
from earnings_alpha.pit import (
    TradingCalendar,
    eastern_to_utc,
    get_calendar,
    parse_session,
    tradable_dates,
)
from earnings_alpha.signals import zscore
from earnings_alpha.types import Session, SurpriseBasis, normalize_ticker

__all__ = [  # noqa: RUF022 - orden temático, no alfabético
    # constantes (defaults de fundamental_factors.md §15.3)
    "MIN_QUARTERS_SUE",
    "SUE_WINDOW",
    "SIGMA_FLOOR_FRAC",
    "CONSENSUS_WINDOW_DAYS",
    "PEAD_HORIZON",
    "MAX_QUARTER_GAP_DAYS",
    # primitivas
    "seasonal_random_walk",
    "sue_time_series",
    "attach_tradable_date",
    "price_before_event",
    # tablas por evento
    "analyst_sue_events",
    "foster_surprise_events",
    "revenue_surprise_events",
    "double_surprise",
    # datos reales
    "load_consensus_events",
    "CONSENSUS_PIT_WARNING",
    # factores
    "AnalystSUE",
    "TimeSeriesSUE",
    "RevenueSurprise",
    "DoubleSurprise",
    "PEAD",
]

MIN_QUARTERS_SUE: Final[int] = 13
"""Observaciones trimestrales consecutivas que exige el paseo aleatorio
estacional con deriva (`j=1..8` requiere `E_{q-12}`; §2.1, verificado)."""

SUE_WINDOW: Final[int] = 8
"""Sorpresas históricas del denominador sigma (Foster, Olsen y Shevlin 1984)."""

SUE_MIN_HISTORY: Final[int] = 6
"""Mínimo de sorpresas previas para la sigma de la variante de analistas."""

SIGMA_FLOOR_FRAC: Final[float] = 0.005
"""Suelo del denominador sigma como fracción del precio (§2.2: sin suelo, una
utility con beneficios estables produce SUE explosivos por un céntimo)."""

CONSENSUS_WINDOW_DAYS: Final[int] = 90
"""Antigüedad máxima del consenso PIT respecto al anuncio (§2.5)."""

PEAD_HORIZON: Final[int] = 5
"""Horizonte de vida por defecto del PEAD, en sesiones. Prior corto
deliberado: Martineau (2021) documenta el PEAD como inexistente en valores
grandes desde ~2006 (§4.3)."""

MAX_QUARTER_GAP_DAYS: Final[int] = 100
"""Separación máxima entre cierres trimestrales para considerarlos
consecutivos; un hueco mayor rompe la comparabilidad estacional (§2.2)."""

DEFAULT_STALENESS_DAYS: Final[int] = 120
"""Tope de arrastre (días naturales) de la sorpresa del último evento: pasado
un trimestre y margen, un ticker que deja de reportar decae a NaN."""


# ---------------------------------------------------------------------------
# Primitivas del modelo de expectativas
# ---------------------------------------------------------------------------


def seasonal_random_walk(eps: Sequence[float] | np.ndarray) -> tuple[float, float, float]:
    """Paseo aleatorio estacional con deriva (Foster, Olsen y Shevlin 1984).

    Sobre las últimas 13 observaciones trimestrales consecutivas (la actual
    incluida) calcula::

        delta = (1/8) * sum_{j=1..8} ( E_{q-j} - E_{q-j-4} )
        E_hat = E_{q-4} + delta
        UE    = E_q - E_hat
        sigma = sd_{j=1..8}[ (E_{q-j} - E_{q-j-4}) - delta ]   (ddof=1)

    Devuelve ``(UE, delta, sigma)``.

    Falla con `InsufficientHistory` si hay menos de 13 observaciones o si
    alguna es NaN: el contrato del repo (§1.3 del informe) prohíbe calcular
    sobre menos historia de la exigida o saltarse huecos como si la serie
    fuese consecutiva.
    """
    values = np.asarray(eps, dtype=float)
    if values.ndim != 1:
        msg = f"`eps` debe ser unidimensional; forma recibida {values.shape}"
        raise DataQualityError(msg)
    if len(values) < MIN_QUARTERS_SUE:
        msg = (
            f"el paseo aleatorio estacional con deriva necesita {MIN_QUARTERS_SUE} "
            f"trimestres consecutivos (q..q-12); recibidos {len(values)}"
        )
        raise InsufficientHistory(msg)
    window = values[-MIN_QUARTERS_SUE:]
    if np.isnan(window).any():
        msg = (
            "hay NaN dentro de la ventana de 13 trimestres: la serie no es "
            "consecutiva y el modelo estacional no es aplicable"
        )
        raise InsufficientHistory(msg)
    # window[0] = E_{q-12} ... window[12] = E_q
    seasonal_diffs = window[4:12] - window[0:8]  # (E_{q-j} - E_{q-j-4}), j=8..1
    drift = float(seasonal_diffs.mean())
    expectation = float(window[8]) + drift  # E_{q-4} + delta
    ue = float(window[12]) - expectation
    sigma = float(np.std(seasonal_diffs - drift, ddof=1))
    return ue, drift, sigma


def sue_time_series(
    eps: Sequence[float] | np.ndarray,
    *,
    basis: SurpriseBasis | str = SurpriseBasis.SIGMA,
    price: float | None = None,
    sigma_floor_frac: float = SIGMA_FLOOR_FRAC,
) -> float:
    """SUE escalar de series temporales para una empresa (§2.2-2.3).

    - ``SurpriseBasis.SIGMA``: ``UE / max(sigma, sigma_floor_frac * price)``.
      Adimensional e invariante de escala (verificado en el informe:
      multiplicar la serie por 7 no cambia el resultado). Si no se aporta
      `price` el suelo no puede aplicarse y una sigma degenerada (0) devuelve
      NaN en vez de ±inf.
    - ``SurpriseBasis.PRICE``: ``UE / P_{t^-}`` con `P_{t^-}` el cierre de la
      sesión anterior a la fecha negociable. Requiere `price`. Advertencia del
      informe §2.3: correlaciona con *value* por construcción y debe
      ortogonalizarse contra `earnings_yield` antes de combinar.

    Las variantes de analistas no pasan por aquí: usan `analyst_sue_events`.
    """
    b = SurpriseBasis(basis)
    ue, _, sigma = seasonal_random_walk(eps)
    if b is SurpriseBasis.SIGMA:
        floor = 0.0
        if price is not None:
            if price <= 0:
                msg = f"precio no positivo para el suelo de sigma: {price}"
                raise DataQualityError(msg)
            floor = sigma_floor_frac * price
        # Degeneración relativa: una serie perfectamente estacional produce
        # sigma ~ 1e-16 por redondeo, no 0.0 exacto; dividir por ese residuo
        # fabricaría SUEs de 1e13. El umbral es relativo a la escala de la
        # serie para preservar la invariancia de escala.
        scale = float(np.max(np.abs(np.asarray(eps, dtype=float)[-MIN_QUARTERS_SUE:])))
        degenerate = sigma < 1e-9 * max(scale, np.finfo(float).tiny)
        denom = max(sigma, floor)
        if not np.isfinite(denom) or denom <= 0.0 or (degenerate and floor <= 0.0):
            return float("nan")
        return ue / denom
    if b is SurpriseBasis.PRICE:
        if price is None or price <= 0:
            msg = "SurpriseBasis.PRICE necesita el cierre previo al tradable_date (P_{t^-})"
            raise ConfigError(msg)
        return ue / price
    msg = (
        f"base {b.value!r} no aplicable al SUE de series temporales; "
        "las bases de analistas se calculan con `analyst_sue_events`"
    )
    raise ConfigError(msg)


# ---------------------------------------------------------------------------
# Utilidades sobre tablas de eventos
# ---------------------------------------------------------------------------


def attach_tradable_date(
    events: pd.DataFrame, calendar: TradingCalendar | None = None
) -> pd.DataFrame:
    """Devuelve `events` con columna `tradable_date` (primera sesión explotable).

    Si la tabla ya trae `event_date` (convención de `SyntheticMarket.events` y
    de `load_consensus_events`) se reutiliza; si no, se deriva de
    `announced_at` + `session` con `pit.tradable_dates`, que aplica la política
    conservadora (AMC y UNKNOWN → sesión siguiente).
    """
    out = events.copy()
    require_columns(out, ["ticker"], name="events")
    if "tradable_date" in out.columns:
        out["tradable_date"] = pd.DatetimeIndex(pd.to_datetime(out["tradable_date"])).normalize()
        return out
    if "event_date" in out.columns:
        out["tradable_date"] = pd.DatetimeIndex(pd.to_datetime(out["event_date"])).normalize()
        return out
    require_columns(out, ["announced_at"], name="events")
    out["tradable_date"] = tradable_dates(out, calendar or get_calendar())
    return out


def price_before_event(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    *,
    price_col: str = "close",
    date_col: str = "tradable_date",
) -> pd.Series:
    """Cierre de la sesión **anterior** a la fecha negociable de cada evento.

    Es el `P_{t^-}` del informe (§2.3): la elección PIT-válida y documentada
    una sola vez para deflactar sorpresas. Se usa el `close` sin ajustar, no el
    `adj_close`: el EPS del trimestre y el precio deben compartir base de
    acciones contemporánea.

    Devuelve una Series alineada con el índice de `events`; NaN cuando el
    ticker no está en el panel o no hay sesión previa con precio.
    """
    require_columns(events, ["ticker", date_col], name="events")
    if not isinstance(prices.index, pd.MultiIndex):
        msg = "`prices` debe ser el panel canónico con MultiIndex (date, ticker)"
        raise DataQualityError(msg)
    if price_col not in prices.columns:
        msg = f"`prices` no tiene columna {price_col!r}"
        raise DataQualityError(msg)

    wide = prices[price_col].unstack("ticker")  # noqa: PD010
    grid = wide.index.to_numpy(dtype="datetime64[ns]")
    values = wide.to_numpy(dtype=float)

    ts = (
        pd.DatetimeIndex(pd.to_datetime(events[date_col]))
        .normalize()
        .to_numpy(dtype="datetime64[ns]")
    )
    # side="left" y -1: la última sesión ESTRICTAMENTE anterior al tradable_date.
    row = np.searchsorted(grid, ts, side="left") - 1
    col = wide.columns.get_indexer(events["ticker"].astype(str))
    ok = (row >= 0) & (col >= 0)
    out = np.full(len(events), np.nan)
    out[ok] = values[row[ok], col[ok]]
    return pd.Series(out, index=events.index, name="price_prev")


def _event_ids(events: pd.DataFrame) -> pd.Series:
    """`event_id` estable (mismo formato que `types.EarningsEvent.event_id`)."""
    if "event_id" in events.columns:
        return events["event_id"].astype(str)
    pe = pd.DatetimeIndex(pd.to_datetime(events["period_end"]))
    if "fiscal_quarter" in events.columns:
        fq = events["fiscal_quarter"].astype(str)
    else:
        fq = pd.Series(
            [f"{d.year}Q{(d.month - 1) // 3 + 1}" for d in pe], index=events.index
        )
    return pd.Series(
        [
            f"{t}:{q}:{d.date().isoformat()}"
            for t, q, d in zip(events["ticker"], fq, pe, strict=True)
        ],
        index=events.index,
        name="event_id",
    )


def _pit_consensus(
    events: pd.DataFrame,
    estimates: pd.DataFrame,
    *,
    consensus_window_days: int,
) -> pd.DataFrame:
    """Última foto de consenso estrictamente anterior al `tradable_date`.

    Materializa la trampa 3 de §2.5: el consenso debe leerse de un snapshot con
    ``as_of <= tradable_date - 1``; usar el consenso final revisado tras el
    anuncio es look-ahead puro. Devuelve por evento ``consensus`` (mediana si
    existe, si no media: la mediana es más robusta al analista rezagado) y
    ``dispersion`` (`eps_std`). Fotos más antiguas que `consensus_window_days`
    respecto al `tradable_date` se consideran caducadas → NaN.
    """
    require_columns(estimates, ["ticker", "period_end", "as_of"], name="estimates")
    est = estimates.copy()
    est["period_end"] = pd.DatetimeIndex(pd.to_datetime(est["period_end"])).normalize()
    est["as_of"] = pd.DatetimeIndex(pd.to_datetime(est["as_of"])).normalize()
    value_col = "eps_median" if "eps_median" in est.columns else "eps_mean"
    if value_col not in est.columns:
        msg = "`estimates` debe traer `eps_median` o `eps_mean`"
        raise DataQualityError(msg)
    keep = ["ticker", "period_end", "as_of", value_col]
    if "eps_std" in est.columns:
        keep.append("eps_std")
    est = est[keep].dropna(subset=["as_of"])

    ev = events[["ticker", "period_end", "tradable_date"]].copy()
    ev["period_end"] = pd.DatetimeIndex(pd.to_datetime(ev["period_end"])).normalize()
    ev["__row__"] = np.arange(len(ev))

    merged = ev.merge(est, on=["ticker", "period_end"], how="left")
    # Estrictamente anterior a la sesión negociable: en la práctica equivale a
    # `as_of <= tradable_date - 1 día`.
    merged = merged[merged["as_of"] < merged["tradable_date"]]
    merged = merged[
        merged["as_of"] >= merged["tradable_date"] - pd.Timedelta(days=consensus_window_days)
    ]
    merged = merged.sort_values("as_of").drop_duplicates("__row__", keep="last")

    out = pd.DataFrame(
        {"consensus": np.nan, "dispersion": np.nan, "consensus_as_of": pd.NaT},
        index=events.index,
    )
    rows = merged["__row__"].to_numpy()
    out.iloc[rows, out.columns.get_loc("consensus")] = merged[value_col].to_numpy()
    if "eps_std" in merged.columns:
        out.iloc[rows, out.columns.get_loc("dispersion")] = merged["eps_std"].to_numpy()
    out.iloc[rows, out.columns.get_loc("consensus_as_of")] = merged["as_of"].to_numpy()
    return out


# ---------------------------------------------------------------------------
# SUE de analistas (por evento)
# ---------------------------------------------------------------------------


def analyst_sue_events(
    events: pd.DataFrame,
    *,
    basis: SurpriseBasis | str = SurpriseBasis.SIGMA,
    prices: pd.DataFrame | None = None,
    estimates: pd.DataFrame | None = None,
    window: int = SUE_WINDOW,
    min_history: int = SUE_MIN_HISTORY,
    demean: bool = True,
    consensus_window_days: int = CONSENSUS_WINDOW_DAYS,
    dispersion_floor_frac: float = SIGMA_FLOOR_FRAC,
    abs_floor: float = 0.01,
    calendar: TradingCalendar | None = None,
) -> pd.DataFrame:
    """SUE de analistas por evento (Livnat y Mendenhall 2006), cuatro bases.

    Sorpresa: ``s_q = A_q - M_q`` con `A` el EPS publicado y `M` el consenso.
    Si se aporta `estimates` (fotos as-of estilo `types.EstimateSnapshot`), `M`
    es la **mediana PIT** de la última foto estrictamente anterior al
    `tradable_date` (§2.5, trampa 3). Sin fotos, `M` es el `eps_estimate` de la
    tabla de eventos: válido con `load_consensus_events` porque ese dataset ya
    es el consenso *final previo al anuncio* (véase `CONSENSUS_PIT_WARNING`),
    pero inadmisible con un consenso descargado hoy para eventos pasados.

    Bases (`types.SurpriseBasis`):

    - ``SIGMA``: ``(s_q - mean_8(s)) / sd_8(s)`` sobre las últimas `window`
      sorpresas **anteriores** (mínimo `min_history`); con `demean=False` se
      omite la media. Es la estandarización habitual de la literatura PEAD
      cuando no hay precio (Livnat y Mendenhall 2006, §2 del informe).
    - ``PRICE``: ``s_q / P_{t^-}`` (§2.5; primaria con consenso). Requiere
      `prices`. Correlaciona con *value*: ortogonalizar contra `E/P`.
    - ``ANALYST_DISPERSION``: ``s_q / max(sd_analistas, 0.005 · P_{t^-})``
      (§2.6). Requiere `estimates` con `eps_std` y `prices` para el suelo.
      Advertencia: la dispersión es por sí misma un predictor (Diether, Malloy
      y Scherbina 2002); usar solo con la dispersión como control aparte.
    - ``ABS_ESTIMATE``: ``s_q / max(|M|, abs_floor)``. **Solo reporting**
      (§2.4): con consenso próximo a cero explota y la explosión correlaciona
      con sector y ciclo. Se implementa para reproducir titulares.

    Devuelve un DataFrame alineado por evento con ``event_id, ticker,
    period_end, tradable_date, consensus, surprise, sue``. Eventos sin
    histórico o sin insumos → NaN; si **ningún** evento es computable se lanza
    `InsufficientHistory` (nada de paneles enteros de NaN en silencio).
    """
    b = SurpriseBasis(basis)
    require_columns(events, ["ticker", "period_end", "eps_actual"], name="events")
    ev = attach_tradable_date(events, calendar)
    ev["period_end"] = pd.DatetimeIndex(pd.to_datetime(ev["period_end"])).normalize()

    if estimates is not None and len(estimates) > 0:
        cons = _pit_consensus(ev, estimates, consensus_window_days=consensus_window_days)
        consensus = cons["consensus"]
        dispersion = cons["dispersion"]
    else:
        if "eps_estimate" not in ev.columns:
            msg = (
                "sin panel de `estimates` la tabla de eventos debe traer "
                "`eps_estimate` (consenso final previo al anuncio)"
            )
            raise DataQualityError(msg)
        consensus = ev["eps_estimate"].astype(float)
        dispersion = pd.Series(np.nan, index=ev.index)

    surprise = ev["eps_actual"].astype(float) - consensus

    if b is SurpriseBasis.SIGMA:
        ordered = pd.DataFrame(
            {
                "ticker": ev["ticker"].to_numpy(),
                "period_end": ev["period_end"].to_numpy(),
                "__s__": surprise.to_numpy(dtype=float),
            }
        ).sort_values(["ticker", "period_end"], kind="mergesort")
        grouped = ordered.groupby("ticker", sort=False)["__s__"]
        prior_mean = grouped.transform(
            lambda s: s.shift(1).rolling(window, min_periods=min_history).mean()
        )
        prior_std = grouped.transform(
            lambda s: s.shift(1).rolling(window, min_periods=min_history).std(ddof=1)
        )
        center = prior_mean if demean else 0.0
        raw = (ordered["__s__"] - center) / prior_std.replace(0.0, np.nan)
        sue = pd.Series(raw.sort_index().to_numpy(), index=ev.index)
    elif b is SurpriseBasis.PRICE:
        if prices is None:
            msg = "SurpriseBasis.PRICE necesita el panel de precios para P_{t^-}"
            raise ConfigError(msg)
        p_prev = price_before_event(prices, ev)
        sue = surprise / p_prev.where(p_prev > 0)
    elif b is SurpriseBasis.ANALYST_DISPERSION:
        if prices is None:
            msg = (
                "SurpriseBasis.ANALYST_DISPERSION necesita `prices` para el suelo "
                "0.005·P del denominador (§2.6)"
            )
            raise ConfigError(msg)
        if dispersion.isna().all():
            msg = (
                "SurpriseBasis.ANALYST_DISPERSION necesita `estimates` con `eps_std`: "
                "sin dispersión de analistas la base no está definida"
            )
            raise ProviderUnavailable("estimates", msg)
        p_prev = price_before_event(prices, ev)
        floor = dispersion_floor_frac * p_prev.where(p_prev > 0)
        denom = np.maximum(dispersion, floor)
        sue = surprise / pd.Series(denom, index=ev.index).where(lambda d: d > 0)
    elif b is SurpriseBasis.ABS_ESTIMATE:
        denom = consensus.abs().clip(lower=abs_floor)
        sue = surprise / denom
    else:  # pragma: no cover - SurpriseBasis es exhaustivo
        msg = f"base de sorpresa desconocida: {b!r}"
        raise ConfigError(msg)

    out = pd.DataFrame(
        {
            "event_id": _event_ids(ev),
            "ticker": ev["ticker"].astype(str),
            "period_end": ev["period_end"],
            "tradable_date": ev["tradable_date"],
            "consensus": consensus,
            "surprise": surprise,
            "sue": sue.astype(float),
        },
        index=ev.index,
    )
    if out["sue"].isna().all():
        msg = (
            f"ningún evento con SUE de analistas computable (base {b.value!r}): "
            "revisa histórico de sorpresas, consenso PIT o panel de precios"
        )
        raise InsufficientHistory(msg)
    return out


# ---------------------------------------------------------------------------
# SUE de series temporales y SURGE (por evento)
# ---------------------------------------------------------------------------


def foster_surprise_events(
    events: pd.DataFrame,
    *,
    value_col: str = "eps_actual",
    min_quarters: int = MIN_QUARTERS_SUE,
    max_gap_days: int = MAX_QUARTER_GAP_DAYS,
) -> pd.DataFrame:
    """UE, deriva y sigma del modelo estacional para cada evento de la tabla.

    Aplica `seasonal_random_walk` de forma rodante por ticker sobre la columna
    `value_col` ordenada por `period_end`. Un evento recibe NaN si no tiene 13
    trimestres consecutivos detrás (huecos de más de `max_gap_days` días entre
    cierres rompen la consecutividad, §2.2: discontinuidades estructurales
    contaminan `delta` y `sigma` durante 8 trimestres).

    Si ningún evento de la tabla alcanza el histórico, `InsufficientHistory`.
    """
    require_columns(events, ["ticker", "period_end", value_col], name="events")
    if min_quarters < MIN_QUARTERS_SUE:
        msg = (
            f"min_quarters={min_quarters} < {MIN_QUARTERS_SUE}: el modelo estacional "
            "con j=1..8 exige 13 observaciones; relajarlo produce sesgo, no robustez"
        )
        raise ConfigError(msg)

    work = events[["ticker", "period_end", value_col]].copy()
    work["period_end"] = pd.DatetimeIndex(pd.to_datetime(work["period_end"])).normalize()
    original_index = work.index
    work = work.reset_index(drop=True)
    work["__orig__"] = np.arange(len(work))
    work = work.sort_values(["ticker", "period_end"], kind="mergesort")

    ue = np.full(len(work), np.nan)
    drift = np.full(len(work), np.nan)
    sigma = np.full(len(work), np.nan)

    for _, sub in work.groupby("ticker", sort=False):
        values = sub[value_col].to_numpy(dtype=float)
        ends = sub["period_end"].to_numpy(dtype="datetime64[ns]")
        rows = sub["__orig__"].to_numpy()
        gaps = np.diff(ends) / np.timedelta64(1, "D")
        for i in range(min_quarters - 1, len(sub)):
            lo = i - (min_quarters - 1)
            window = values[lo : i + 1]
            if np.isnan(window).any():
                continue
            if len(gaps) and (gaps[lo:i] > max_gap_days).any():
                continue
            u, d, s = seasonal_random_walk(window)
            ue[rows[i]], drift[rows[i]], sigma[rows[i]] = u, d, s

    if np.isnan(ue).all():
        msg = (
            f"ningún evento con {min_quarters} trimestres consecutivos de "
            f"{value_col!r}: no hay SUE de series temporales computable"
        )
        raise InsufficientHistory(msg)
    base = work.sort_values("__orig__").drop(columns="__orig__")
    base.index = original_index
    return base.assign(ue=ue, drift=drift, sigma=sigma)


def revenue_surprise_events(
    fundamentals: pd.DataFrame,
    events: pd.DataFrame,
    *,
    revenue_col: str = "revenue",
    shares_col: str = "shares_diluted",
    min_quarters: int = MIN_QUARTERS_SUE,
    max_gap_days: int = MAX_QUARTER_GAP_DAYS,
    calendar: TradingCalendar | None = None,
) -> pd.DataFrame:
    """SURGE por evento (Jegadeesh y Livnat 2006), sobre ingresos POR ACCIÓN.

    ::

        R_q     = Ingresos_q / acciones_q
        SURGE_q = ( R_q - R_{q-4} - delta^R ) / sd_8( R_{q-j} - R_{q-j-4} )

    El "por acción" no es cosmético: el informe (§3) verifica numéricamente que
    con una recompra del 10% el factor sobre ingresos totales invierte el signo
    (-0.814 vs +0.126) y pasa a medir la política de recompras (*net share
    issuance*), que es otro factor. Mismo requisito de 13 trimestres que SUE.

    La fecha de la señal es el `tradable_date` del **evento** (los ingresos se
    publican en la nota de prensa 8-K), no el `filed_at` del 10-Q.

    Dónde falla (§3): bancos y aseguradoras ("ingresos" no es homogéneo),
    inmobiliarias (usar rentas) y empresas muy adquisitivas (crecimiento
    inorgánico); neutralizar por sector siempre.
    """
    require_columns(fundamentals, ["ticker", "period_end", revenue_col, shares_col],
                    name="fundamentals")
    require_columns(events, ["ticker", "period_end"], name="events")

    ev = attach_tradable_date(events, calendar)
    ev["period_end"] = pd.DatetimeIndex(pd.to_datetime(ev["period_end"])).normalize()

    fund = fundamentals[["ticker", "period_end", revenue_col, shares_col]].copy()
    fund["period_end"] = pd.DatetimeIndex(pd.to_datetime(fund["period_end"])).normalize()
    shares = fund[shares_col].astype(float)
    fund["rps"] = fund[revenue_col].astype(float) / shares.where(shares > 0)

    merged = ev.merge(
        fund[["ticker", "period_end", "rps"]],
        on=["ticker", "period_end"],
        how="left",
        validate="many_to_one",
    )
    merged.index = ev.index
    if merged["rps"].isna().all():
        msg = (
            "ningún evento casó con fundamentales de ingresos por acción: "
            "revisa las claves (ticker, period_end)"
        )
        raise DataQualityError(msg)

    foster = foster_surprise_events(
        merged, value_col="rps", min_quarters=min_quarters, max_gap_days=max_gap_days
    )
    # Degeneración relativa (misma salvaguarda que `sue_time_series`): una
    # sigma que solo es residuo de redondeo no es un denominador.
    degeneracy = 1e-9 * merged["rps"].abs().clip(lower=np.finfo(float).tiny)
    sigma = foster["sigma"].where(foster["sigma"] > degeneracy)
    surge = foster["ue"] / sigma

    out = pd.DataFrame(
        {
            "event_id": _event_ids(ev),
            "ticker": ev["ticker"].astype(str),
            "period_end": ev["period_end"],
            "tradable_date": ev["tradable_date"],
            "rps": merged["rps"],
            "ue": foster["ue"],
            "surge": surge.astype(float),
        },
        index=ev.index,
    )
    if out["surge"].isna().all():
        msg = "ningún evento con SURGE computable tras aplicar el requisito de histórico"
        raise InsufficientHistory(msg)
    return out


def double_surprise(sue_z: pd.Series, surge_z: pd.Series) -> pd.Series:
    """Señal de doble sorpresa (Jegadeesh y Livnat 2006, FAJ)::

        DOUBLE = sign(SUE) * min(|z(SUE)|, |z(SURGE)|)   si comparten signo
               = 0                                        si discrepan

    Deliberadamente conservadora (§3 del informe): solo puntúa cuando la
    sorpresa de demanda (ingresos) confirma la de beneficios, y la
    discrepancia se anula en lugar de promediarse. NaN si falta cualquiera de
    las dos entradas.
    """
    a, b = sue_z.align(surge_z, join="outer")
    agree = np.sign(a) == np.sign(b)
    value = np.sign(a) * np.minimum(a.abs(), b.abs())
    out = value.where(agree, 0.0)
    out[a.isna() | b.isna()] = np.nan
    out.name = "double_surprise"
    return out


# ---------------------------------------------------------------------------
# Eventos reales: consenso_master.parquet
# ---------------------------------------------------------------------------

CONSENSUS_PIT_WARNING: Final[str] = (
    "ADVERTENCIA PIT de consenso_master.parquet: (1) `eps_estimated` es el "
    "consenso FINAL previo al anuncio, sin historial de revisiones "
    "intra-trimestre — sirve para SUE de analistas y estudios de evento, NO "
    "para momentum de revisiones (reconstruirlo desde el consenso final es "
    "look-ahead de manual); (2) la lista de tickers padece supervivencia "
    "parcial (snapshots de 2022 y 2025): empresas excluidas del índice antes "
    "de esas fechas están infrarrepresentadas, así que toda estadística "
    "agregada sobre este dataset hereda ese sesgo; (3) los ficheros de "
    "proveedores estilo I/B/E/S llegan reexpresados por splits, y el redondeo "
    "a dos decimales puede fabricar sorpresas espurias correlacionadas con la "
    "rentabilidad (Payne y Thomas 2003)."
)

# Claves como `str` y no como `Session`: las columnas Arrow del parquet
# degradan un StrEnum a texto plano al almacenarlo, así que todo el flujo del
# loader trabaja con los valores string canónicos de `types.Session`.
_SESSION_ANNOUNCE_MINUTES: Final[dict[str, int]] = {
    Session.BMO.value: 7 * 60,        # 07:00 ET, antes de la apertura
    Session.DMH.value: 12 * 60 + 30,  # 12:30 ET, sesión abierta
    Session.AMC.value: 17 * 60,       # 17:00 ET, tras el cierre
    Session.UNKNOWN.value: 17 * 60,   # conservador: se trata como AMC
}


def load_consensus_events(
    path: str | Path | None = None,
    *,
    tickers: Sequence[str] | None = None,
    start: dt.date | str | None = None,
    end: dt.date | str | None = None,
    calendar: TradingCalendar | None = None,
    settings: Settings | None = None,
    keep_sources: bool = True,
) -> pd.DataFrame:
    """Tabla de eventos de resultados REALES desde ``consenso_master.parquet``.

    Construye, a partir del dataset externo de consenso (56.971 filas, ~500
    tickers, 1995-2026), una tabla de eventos con el mismo esquema que
    `SyntheticMarket.events()`: ``event_id, ticker, period_end, fiscal_quarter,
    announced_at, session, event_date, eps_actual, eps_estimate, eps_surprise,
    surprise_pct, source, snapshot_date``.

    Derivación de la sesión BMO/AMC
    -------------------------------
    `report_time` (``pre-market`` → BMO, ``post-market`` → AMC, ``intraday`` →
    DMH, ausente → UNKNOWN) se normaliza con `pit.parse_session`, que **nunca
    adivina BMO**: UNKNOWN se trata como AMC (sesión negociable = la
    siguiente), la política conservadora del repo. Como la mayor parte de las
    filas sin `report_time` proceden de una fuente y las filas con él de otra,
    la sesión se **propaga entre fuentes** por clave ``(ticker, report_date)``
    cuando todas las fuentes con etiqueta coinciden; ante etiquetas
    contradictorias se conserva UNKNOWN.

    El instante `announced_at` se sintetiza (07:00 ET para BMO, 12:30 ET para
    DMH, 17:00 ET para AMC/UNKNOWN) porque el dataset solo trae la fecha: la
    hora exacta es ficticia, pero la **fecha y la sesión** —lo único que usa
    `pit.tradable_date`— son las reportadas.

    Deduplicación
    -------------
    El dataset agrega varias fuentes y una misma presentación aparece hasta 4
    veces. Se conserva una fila por ``(ticker, fiscal_period_end)`` con esta
    preferencia: sesión conocida > EPS y consenso no nulos > `snapshot_date`
    más reciente. Las filas sin `fiscal_period_end` (fuentes yfinance) se usan
    para propagar la sesión y después se descartan: sin cierre de trimestre no
    hay clave de evento ni posibilidad de histórico estacional. El recuento de
    descartes queda en ``result.attrs``.

    Advertencia PIT (obligatoria de propagar a todo consumidor)
    -----------------------------------------------------------
    El texto vinculante vive en `CONSENSUS_PIT_WARNING` y se adjunta en
    ``result.attrs["pit_warning"]``. En resumen: (1) `eps_estimated` es el
    consenso **final previo al anuncio, sin vintages** — válido para SUE de
    analistas y estudios de evento, look-ahead de manual si se usa para
    momentum de revisiones —; (2) **supervivencia parcial** de tickers
    (snapshots de 2022 y 2025): las empresas excluidas antes de esas fechas
    están infrarrepresentadas; (3) posibles reexpresiones por split del
    proveedor, que fabrican sorpresas espurias (Payne y Thomas 2003).

    Lanza `ProviderUnavailable` si el fichero no existe y `DataQualityError` /
    `InsufficientHistory` si el filtrado deja la tabla vacía.
    """
    cfg = settings or get_settings()
    file_path = (
        Path(path)
        if path is not None
        else cfg.data_dir / "external" / "consenso" / "consenso_master.parquet"
    )
    if not file_path.exists():
        raise ProviderUnavailable(
            "consenso_master",
            f"no existe {file_path}; el dataset externo de consenso no está en el repo",
        )
    raw = pd.read_parquet(file_path)
    require_columns(
        raw,
        ["ticker", "fiscal_period_end", "report_date", "report_time",
         "eps_reported", "eps_estimated", "surprise", "surprise_pct",
         "source", "snapshot_date"],
        name="consenso_master",
    )

    frame = raw.copy()
    frame["ticker"] = [normalize_ticker(str(t)) for t in frame["ticker"]]
    frame["report_date"] = pd.DatetimeIndex(pd.to_datetime(frame["report_date"])).normalize()
    frame["period_end"] = pd.to_datetime(frame["fiscal_period_end"], errors="coerce")
    frame["session"] = pd.Series(
        [parse_session(v if not pd.isna(v) else None).value for v in frame["report_time"]],
        index=frame.index,
        dtype=object,
    )

    if tickers is not None:
        wanted = {normalize_ticker(t) for t in tickers}
        frame = frame[frame["ticker"].isin(wanted)]
        if len(frame) == 0:
            msg = f"ningún evento en consenso_master para los tickers pedidos: {sorted(wanted)}"
            raise InsufficientHistory(msg)
    if start is not None:
        frame = frame[frame["report_date"] >= pd.Timestamp(start)]
    if end is not None:
        frame = frame[frame["report_date"] <= pd.Timestamp(end)]
    if len(frame) == 0:
        msg = "el filtro de fechas dejó consenso_master vacío"
        raise InsufficientHistory(msg)

    # --- propagación de sesión entre fuentes por (ticker, report_date) -----
    unknown = Session.UNKNOWN.value
    labeled = frame[frame["session"] != unknown]
    per_key = labeled.groupby(["ticker", "report_date"])["session"].agg(
        lambda s: s.iloc[0] if len(set(s)) == 1 else unknown
    )
    key = pd.MultiIndex.from_frame(frame[["ticker", "report_date"]])
    inferred = pd.Series(per_key.reindex(key).to_numpy(), index=frame.index)
    needs = frame["session"] == unknown
    frame.loc[needs, "session"] = inferred[needs].where(inferred[needs].notna(), unknown)
    n_session_filled = int((needs & (frame["session"] != unknown)).sum())

    # --- descartes sin clave de evento -------------------------------------
    n_no_period = int(frame["period_end"].isna().sum())
    frame = frame.dropna(subset=["period_end"])
    if len(frame) == 0:
        msg = "todas las filas seleccionadas carecen de fiscal_period_end"
        raise DataQualityError(msg)

    # --- deduplicación por (ticker, period_end) ----------------------------
    frame["__has_session__"] = frame["session"] != unknown
    frame["__has_eps__"] = frame["eps_reported"].notna() & frame["eps_estimated"].notna()
    frame["__snap__"] = pd.to_datetime(frame["snapshot_date"], errors="coerce")
    n_before = len(frame)
    frame = frame.sort_values(
        ["ticker", "period_end", "__has_session__", "__has_eps__", "__snap__", "source"],
        kind="mergesort",
    ).drop_duplicates(subset=["ticker", "period_end"], keep="last")
    n_dupes = n_before - len(frame)

    # --- announced_at sintético y fecha negociable --------------------------
    cal = calendar or get_calendar()
    minutes = frame["session"].map(_SESSION_ANNOUNCE_MINUTES).astype(int)
    local = frame["report_date"] + pd.to_timedelta(minutes, unit="m")
    unique_local = pd.DatetimeIndex(local.unique())
    to_utc = {ts: eastern_to_utc(ts.to_pydatetime()) for ts in unique_local}
    frame["announced_at"] = pd.DatetimeIndex([to_utc[ts] for ts in local])
    frame["event_date"] = tradable_dates(frame, cal)

    pe = pd.DatetimeIndex(frame["period_end"])
    frame["fiscal_quarter"] = [f"{d.year}Q{(d.month - 1) // 3 + 1}" for d in pe]
    frame["event_id"] = [
        f"{t}:{q}:{d.date().isoformat()}"
        for t, q, d in zip(frame["ticker"], frame["fiscal_quarter"], pe, strict=True)
    ]
    frame = frame.rename(
        columns={"eps_reported": "eps_actual", "eps_estimated": "eps_estimate",
                 "surprise": "eps_surprise"}
    )
    frame["is_estimated_date"] = False

    cols = [
        "event_id", "ticker", "period_end", "fiscal_quarter", "announced_at",
        "session", "event_date", "eps_actual", "eps_estimate", "eps_surprise",
        "surprise_pct", "is_estimated_date",
    ]
    if keep_sources:
        cols += ["source", "snapshot_date"]
    out = (
        frame[cols]
        .sort_values(["ticker", "period_end"], kind="mergesort")
        .reset_index(drop=True)
    )
    out.attrs["pit_warning"] = CONSENSUS_PIT_WARNING
    out.attrs["n_dropped_no_period_end"] = n_no_period
    out.attrs["n_duplicates_collapsed"] = n_dupes
    out.attrs["n_session_filled_across_sources"] = n_session_filled
    if n_no_period:
        warnings.warn(
            f"consenso_master: {n_no_period} filas sin fiscal_period_end descartadas "
            "tras aprovechar su report_time para inferir la sesión",
            stacklevel=2,
        )
    return out


# ---------------------------------------------------------------------------
# Factores registrados
# ---------------------------------------------------------------------------


class AnalystSUE:
    """Factor `sue_analyst`: sorpresa frente al consenso, proyectada a panel.

    Referencia: Livnat y Mendenhall (2006), *Comparing the Post-Earnings
    Announcement Drift for Surprises Calculated from Analyst and Time Series
    Forecasts*, JAR 44(1). Mayor sorpresa estandarizada = más alcista.

    El valor del último evento se arrastra desde su `tradable_date` (PIT vía
    `pit.asof_join`) hasta el siguiente evento, con tope de obsolescencia
    `staleness_days`. La vida útil de la señal es corta (3-10 sesiones, §14
    del informe): el arrastre largo existe para que DOUBLE y los diagnósticos
    tengan panel completo, y el horizonte corto lo aplica `PEAD`.
    """

    requires: ClassVar[list[str]] = ["earnings_calendar", "prices", "estimates"]

    def __init__(
        self,
        basis: SurpriseBasis | str = SurpriseBasis.SIGMA,
        *,
        window: int = SUE_WINDOW,
        min_history: int = SUE_MIN_HISTORY,
        demean: bool = True,
        use_snapshots: bool = True,
        staleness_days: int | None = DEFAULT_STALENESS_DAYS,
    ) -> None:
        self.basis = SurpriseBasis(basis)
        self.window = window
        self.min_history = min_history
        self.demean = demean
        self.use_snapshots = use_snapshots
        self.staleness_days = staleness_days
        self.name = f"sue_analyst_{self.basis.value}"

    def event_values(self, ctx: FactorContext) -> pd.DataFrame:
        """Tabla por evento (`analyst_sue_events`) con los insumos del contexto."""
        ctx.require("events")
        needs_prices = self.basis in (SurpriseBasis.PRICE, SurpriseBasis.ANALYST_DISPERSION)
        if needs_prices:
            ctx.require("prices")
        estimates = None
        if self.use_snapshots and ctx.estimates is not None and len(ctx.estimates) > 0:
            estimates = ctx.estimates
        return analyst_sue_events(
            ctx.events,
            basis=self.basis,
            prices=ctx.prices if needs_prices else None,
            estimates=estimates,
            window=self.window,
            min_history=self.min_history,
            demean=self.demean,
            calendar=ctx.calendar,
        )

    def compute_frame(self, ctx: FactorContext) -> pd.DataFrame:
        """Panel con columnas ``[valor, available_at]``, auditable con
        `pit.assert_no_lookahead`."""
        table = self.event_values(ctx)
        return spread_event_frame(
            table,
            ctx.dates,
            ctx.tickers(),
            value_col="sue",
            max_staleness_days=self.staleness_days,
            name=self.name,
        )

    def compute(self, ctx: FactorContext) -> pd.Series:
        """Serie ``(date, ticker)``; valor mayor = más alcista."""
        return self.compute_frame(ctx)[self.name].rename(self.name)


class TimeSeriesSUE:
    """Factor `sue_time_series`: SUE del paseo aleatorio estacional con deriva.

    Referencias: Foster (1977); Foster, Olsen y Shevlin (1984); Bernard y
    Thomas (1989). No necesita consenso: solo la historia de EPS publicados,
    13 trimestres consecutivos. Es el respaldo por defecto sin datos de
    analistas (§2.7). Con ``basis=SIGMA`` aplica el suelo obligatorio
    ``max(sigma, 0.005·P_{t^-})``; con ``basis=PRICE`` deflacta por el cierre
    previo (§2.3: ortogonalizar contra `E/P` antes de combinar).
    """

    requires: ClassVar[list[str]] = ["earnings_calendar", "prices"]

    def __init__(
        self,
        basis: SurpriseBasis | str = SurpriseBasis.SIGMA,
        *,
        sigma_floor_frac: float = SIGMA_FLOOR_FRAC,
        staleness_days: int | None = DEFAULT_STALENESS_DAYS,
        value_col: str = "eps_actual",
    ) -> None:
        b = SurpriseBasis(basis)
        if b not in (SurpriseBasis.SIGMA, SurpriseBasis.PRICE):
            msg = (
                f"TimeSeriesSUE solo admite SIGMA o PRICE; {b.value!r} es una base "
                "de analistas (usa AnalystSUE)"
            )
            raise ConfigError(msg)
        self.basis = b
        self.sigma_floor_frac = sigma_floor_frac
        self.staleness_days = staleness_days
        self.value_col = value_col
        self.name = f"sue_ts_{b.value}"

    def event_values(self, ctx: FactorContext) -> pd.DataFrame:
        ctx.require("events", "prices")
        ev = attach_tradable_date(ctx.events, ctx.calendar)
        foster = foster_surprise_events(ev, value_col=self.value_col)
        p_prev = price_before_event(ctx.prices, ev)
        if self.basis is SurpriseBasis.PRICE:
            sue = foster["ue"] / p_prev.where(p_prev > 0)
        else:
            floor = self.sigma_floor_frac * p_prev.where(p_prev > 0)
            denom = np.maximum(foster["sigma"], floor)
            sue = foster["ue"] / pd.Series(denom, index=ev.index).where(lambda d: d > 0)
        out = pd.DataFrame(
            {
                "event_id": _event_ids(ev),
                "ticker": ev["ticker"].astype(str),
                "period_end": pd.DatetimeIndex(pd.to_datetime(ev["period_end"])).normalize(),
                "tradable_date": ev["tradable_date"],
                "ue": foster["ue"],
                "sigma": foster["sigma"],
                "sue": sue.astype(float),
            },
            index=ev.index,
        )
        if out["sue"].isna().all():
            msg = "ningún evento con SUE de series temporales computable"
            raise InsufficientHistory(msg)
        return out

    def compute_frame(self, ctx: FactorContext) -> pd.DataFrame:
        table = self.event_values(ctx)
        return spread_event_frame(
            table,
            ctx.dates,
            ctx.tickers(),
            value_col="sue",
            max_staleness_days=self.staleness_days,
            name=self.name,
        )

    def compute(self, ctx: FactorContext) -> pd.Series:
        return self.compute_frame(ctx)[self.name].rename(self.name)


@register_factor()
class RevenueSurprise:
    """Factor `revenue_surprise` (SURGE): sorpresa de ingresos por acción.

    Referencia: Jegadeesh y Livnat (2006), *Revenue Surprises and Stock
    Returns*, JAE 41(1-2). Mayor sorpresa de ingresos = más alcista. Los
    ingresos son menos manipulables que el EPS: una sorpresa de ingresos es
    una sorpresa de demanda, no de tipo impositivo o provisiones (§3).
    """

    name = "revenue_surprise"
    requires: ClassVar[list[str]] = ["earnings_calendar", "fundamentals"]

    def __init__(
        self,
        *,
        revenue_col: str = "revenue",
        shares_col: str = "shares_diluted",
        staleness_days: int | None = DEFAULT_STALENESS_DAYS,
    ) -> None:
        self.revenue_col = revenue_col
        self.shares_col = shares_col
        self.staleness_days = staleness_days

    def event_values(self, ctx: FactorContext) -> pd.DataFrame:
        ctx.require("events", "fundamentals")
        return revenue_surprise_events(
            ctx.fundamentals,
            ctx.events,
            revenue_col=self.revenue_col,
            shares_col=self.shares_col,
            calendar=ctx.calendar,
        )

    def compute_frame(self, ctx: FactorContext) -> pd.DataFrame:
        table = self.event_values(ctx)
        return spread_event_frame(
            table,
            ctx.dates,
            ctx.tickers(),
            value_col="surge",
            max_staleness_days=self.staleness_days,
            name=self.name,
        )

    def compute(self, ctx: FactorContext) -> pd.Series:
        return self.compute_frame(ctx)[self.name].rename(self.name)


@register_factor()
class DoubleSurprise:
    """Factor `double_surprise`: confirmación cruzada SUE ∧ SURGE.

    Referencia: Jegadeesh y Livnat (2006, 2007). El drift posterior es más
    fuerte cuando la sorpresa de ingresos confirma la de beneficios; cuando
    discrepan, la señal se anula (0), no se promedia (§3).

    Los z-scores se calculan por fecha sobre la sección cruzada de los paneles
    arrastrados de SUE y SURGE, y después se combinan celda a celda.
    """

    name = "double_surprise"
    requires: ClassVar[list[str]] = ["earnings_calendar", "fundamentals", "prices"]

    def __init__(
        self,
        *,
        sue_factor: AnalystSUE | TimeSeriesSUE | None = None,
        surge_factor: RevenueSurprise | None = None,
        min_obs: int = 3,
    ) -> None:
        self.sue_factor = sue_factor or AnalystSUE()
        self.surge_factor = surge_factor or RevenueSurprise()
        self.min_obs = min_obs

    def compute(self, ctx: FactorContext) -> pd.Series:
        sue_panel = self.sue_factor.compute(ctx)
        surge_panel = self.surge_factor.compute(ctx)
        sue_z = zscore(sue_panel, min_obs=self.min_obs)
        surge_z = zscore(surge_panel, min_obs=self.min_obs)
        out = double_surprise(sue_z, surge_z)
        return out.rename(self.name)


@register_factor()
class PEAD:
    """Factor `pead`: deriva post-anuncio con vida finita y decaimiento.

    Fórmula (§4.1 del informe)::

        PEAD_{i,t} = z( SUE_{i,q(t)} ) * w(s) * 1{ 0 < s <= H }

    con ``s`` las sesiones transcurridas desde el `tradable_date` y
    ``w(s) = exp(-(s-1)/decay)`` si `decay` no es None (1 en caso contrario).
    El z-score es cross-section por fecha sobre el panel de sorpresas; después
    de la ventana el factor vale **0** (señal agotada = neutral), y antes del
    primer evento negociable de un ticker vale **NaN** (sin información, no
    neutral): esa distinción es la que permite pasar
    `pit.assert_no_lookahead(check_first_event=True)`.

    Referencias: Ball y Brown (1968); Bernard y Thomas (1989): CAR a 60
    sesiones monótono en el decil de SUE. Erosión documentada y vinculante
    para el diseño: Martineau (2021) — PEAD inexistente en valores grandes
    desde ~2006 —, Chordia, Subrahmanyam y Tong (2014), Ng, Rusticus y Verdi
    (2008). Por eso `horizon` por defecto es 5 sesiones y no 60, y cualquier
    backtest debe partirse en subperiodos (§4.3, §13.1).

    `include_event_day=False` reproduce la convención ``0 < s`` del informe:
    la primera sesión viva es la **siguiente** a la negociable, de modo que el
    factor captura deriva y no la reacción del día del anuncio.
    """

    name = "pead"
    requires: ClassVar[list[str]] = ["earnings_calendar", "prices"]

    def __init__(
        self,
        horizon: int = PEAD_HORIZON,
        *,
        decay: float | None = None,
        sue_factor: AnalystSUE | TimeSeriesSUE | None = None,
        include_event_day: bool = False,
        standardize: bool = True,
        min_obs: int = 3,
    ) -> None:
        if horizon < 1:
            msg = f"horizon debe ser >= 1; recibido {horizon}"
            raise ConfigError(msg)
        self.horizon = horizon
        self.decay = decay
        self.sue_factor = sue_factor or AnalystSUE()
        self.include_event_day = include_event_day
        self.standardize = standardize
        self.min_obs = min_obs

    def compute(self, ctx: FactorContext) -> pd.Series:
        table = self.sue_factor.event_values(ctx)
        tickers = ctx.tickers()

        # 1. Panel de sorpresas arrastrado y su z cross-section por fecha.
        sue_panel = spread_event_values(
            table,
            ctx.dates,
            tickers,
            value_col="sue",
            max_staleness_days=DEFAULT_STALENESS_DAYS,
            name="sue",
        )
        scores = zscore(sue_panel, min_obs=self.min_obs) if self.standardize else sue_panel

        # 2. Indicador de vida 1{0 < s <= H} con decaimiento, 0 tras la ventana
        #    y NaN antes del primer evento del ticker.
        indicator = spread_event_values(
            table.assign(__one__=np.where(table["sue"].notna(), 1.0, np.nan)),
            ctx.dates,
            tickers,
            value_col="__one__",
            horizon=self.horizon,
            decay=self.decay,
            include_event_day=self.include_event_day,
            dead_value=0.0,
            calendar=ctx.calendar,
            name="live",
        )
        out = (scores * indicator).astype(float)
        return out.rename(self.name)


def _register_aliases() -> None:
    """Alias de registro con la nomenclatura del informe (§2.7 y §14)."""

    @register_factor("sue_sigma")
    def _sue_sigma(**kwargs: object) -> TimeSeriesSUE:
        return TimeSeriesSUE(basis=SurpriseBasis.SIGMA, **kwargs)  # type: ignore[arg-type]

    @register_factor("sue_price")
    def _sue_price(**kwargs: object) -> TimeSeriesSUE:
        return TimeSeriesSUE(basis=SurpriseBasis.PRICE, **kwargs)  # type: ignore[arg-type]

    @register_factor("sue_analyst")
    def _sue_analyst(**kwargs: object) -> AnalystSUE:
        return AnalystSUE(**kwargs)  # type: ignore[arg-type]


_register_aliases()
