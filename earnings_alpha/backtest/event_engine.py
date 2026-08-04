"""Backtest dirigido por eventos de resultados (contrato §3.7 de `docs/ARCHITECTURE.md`).

El contrato dice: *"Entra `entry_offset` sesiones antes/después del evento, sale en
`exit_offset`. Modela gap overnight de forma explícita (el evento AMC se negocia en
el gap)"*. Este módulo implementa ese motor y su comparador de rejilla
(`run_grid`), que es el entregable con el que se decide entre **cruzar el anuncio**
(entrar antes, cobrar el salto) y **explotar el drift posterior** (entrar después,
cobrar el PEAD).

Decisiones de diseño y sus porqués
----------------------------------

1. **`tau = 0` es la sesión negociable, no la del anuncio.** La tabla de eventos se
   canoniza con `events.window.normalize_events`, que deriva `event_date` vía
   `pit.tradable_date` (BMO/DMH -> misma sesión; AMC/UNKNOWN -> la siguiente). Un
   error de un día aquí es la trampa B7 de `docs/research/pit_and_biases.md` §9:
   con el 90 % del movimiento en el gap infla el Sharpe 2.6-2.8x y el retorno por
   evento 4-10x.

2. **Momento de ejecución explícito.** Cada extremo de la operación es un punto
   `(offset, open|close)`. Con `entry_at="auto"`:

   - `entry_offset >= 0` ejecuta en la **apertura** de esa sesión: para un anuncio
     AMC la apertura de `tau = 0` ya incorpora la noticia (el gap queda fuera de la
     operación, como debe ser); para un BMO, el anuncio precede a la apertura del
     mismo día. Primera ejecución legal en ambos casos.
   - `entry_offset < 0` ejecuta al **cierre** de esa sesión (posicionamiento
     previo al anuncio: el último precio disponible antes del evento).
   - La salida (`exit_at="auto"`) es al cierre de la sesión `exit_offset`.

3. **Política PIT del pre-posicionamiento (obligatoria, `pit_and_biases.md` §8.3).**
   Con `entry_offset < 0` el ancla `T` del evento tiene que haber sido *conocible*
   en `T + entry_offset`, lo que exige vintages de calendario
   (`CalendarVintage`/`known_event_date`). Ese subsistema aún no existe en la
   plataforma, así que el motor **lanza `LookAheadError`** salvo que el llamante
   declare explícitamente `calendar_known_in_advance=True` (legítimo con el
   mercado sintético, cuyo calendario es público por adelantado, o cuando las
   fechas ya fueron resueltas contra vintages aguas arriba). No hay modo degradado
   silencioso.

4. **Gap overnight explícito.** Todo retorno de operación se descompone de forma
   **exacta y multiplicativa** en su componente nocturno y su componente intradía::

       (1 + gap_return) * (1 + intraday_return) = 1 + price_return

   donde `gap_return` compone todos los tramos cierre->apertura mantenidos y
   `intraday_return` los tramos apertura->cierre. Además se reporta, para cada
   evento, la partición del **día del anuncio** (`event_gap_return`,
   `event_intraday_return`, mismas fórmulas que `events.overnight_decomposition`)
   y si la operación mantuvo posición a través del gap del anuncio
   (`holds_event_gap`). Corolario de `pit_and_biases.md` §9.2: "un tearsheet de
   eventos que solo dé retornos close->close es inauditable".

5. **Concurrencia de eventos.** En temporada de resultados decenas de ventanas se
   solapan. El motor impone `max_concurrent` posiciones simultáneas (por defecto
   20, el orden de magnitud usado en las simulaciones de `pit_and_biases.md`
   §2.2/§9.2), asigna a cada evento una fracción fija del NAV
   (`per_event_capital`, por defecto `1/max_concurrent`) y, cuando hay más
   candidatos que huecos, prioriza por |score| descendente con desempate
   determinista. Los descartes quedan en `result.skipped` con su motivo y la
   utilización del capital (`deployed`) se reporta día a día.

6. **Contabilidad por evento, no compuesta.** Cada posición mantiene un número
   fijo de "acciones" (`peso * NAV / precio_entrada`), de modo que la suma de los
   P&L diarios reproduce exactamente `peso * net_return` por evento y el retorno
   total de la cartera es aditivo. Es la contabilidad estándar de los estudios de
   evento con horizonte fijo (Barber y Lyon 1997 discuten las alternativas BHAR
   vs. CAR; aquí el retorno por evento es buy-and-hold y la agregación, aritmética).

7. **Costes reales.** Se reutiliza `backtest.costs.CostModel` (spread por tramo de
   liquidez, impacto raíz cuadrada, comisión, préstamo de cortos), con ADV y
   volatilidad calculados **hacia atrás** (la media móvil termina la sesión
   anterior a la ejecución, `pit_and_biases.md` §12.6). La regla de §9.5 —"el
   CostModel debe aplicar en la apertura post-evento un spread ampliado"— se
   implementa con `event_open_spread_multiplier` (por defecto 3x) sobre el medio
   spread de toda ejecución que ocurra en la apertura de `tau = 0`.

8. **Colas gruesas a la vista.** El resumen por combinación reporta hit rate,
   media con banda de error (bootstrap estacionario sobre los eventos ordenados
   por fecha, `stats.validation`), mediana, percentiles p01..p99, peor y mejor
   evento, asimetría y curtosis cruda: nada de "media y sigma" a secas
   (`validation_methodology.md`: el retorno en ventana de evento tiene curtosis
   altísima). Toda métrica con banda; un Sharpe sin banda no se acepta.

Fuera de alcance de este motor: estrategias con **opciones** a través del evento
(straddles pre-anuncio, IV crush). Requieren un modelo de costes propio (spreads
de opciones, mucho más anchos) y el `iv_crush_expected` de
`docs/research/options_signals.md` §7.3; este motor solo opera el subyacente en
contado. Queda registrado como cuestión abierta de la plataforma.

Referencias
-----------
- Ball, R. y Brown, P. (1968). *An Empirical Evaluation of Accounting Income
  Numbers*. Journal of Accounting Research 6(2). (Respuesta del precio al anuncio.)
- Bernard, V. y Thomas, J. (1989). *Post-Earnings-Announcement Drift: Delayed
  Price Response or Risk Premium?* Journal of Accounting Research 27. (El drift
  que explota `entry_offset > 0`.)
- Barber, B. y Lyon, J. (1997). *Detecting Long-Run Abnormal Stock Returns*.
  Journal of Financial Economics 43. (Retornos buy-and-hold por evento.)
- Kothari, S. P. y Warner, J. (2007). *Econometrics of Event Studies*. Handbook of
  Corporate Finance vol. 1. (Diseño general de backtests de evento.)
- Berkman, H. y Truong, C. (2009). *Event Day 0? After-Hours Earnings
  Announcements*. Journal of Accounting Research 47(1). (Por qué el día negociable
  de un AMC es el siguiente: la base empírica de la regla BMO/AMC.)
- DellaVigna, S. y Pollet, J. (2009). *Investor Inattention and Friday Earnings
  Announcements*. Journal of Finance 64(2). (Heterogeneidad del drift que motiva
  comparar salidas con `run_grid` en vez de fijar una.)
- `docs/research/pit_and_biases.md` §8 (vintages de calendario), §9 (gap
  overnight, multiplicador de spread post-anuncio).
- `docs/research/validation_methodology.md` §11 (costes y bandas de error).
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats as sps

from earnings_alpha.backtest.costs import BPS, CostModel
from earnings_alpha.backtest.portfolio import assign_quantiles
from earnings_alpha.errors import (
    ConfigError,
    DataQualityError,
    InsufficientHistory,
    LookAheadError,
)
from earnings_alpha.events.window import normalize_events
from earnings_alpha.pit import TradingCalendar, get_calendar
from earnings_alpha.stats.performance import raw_kurtosis, sharpe_ratio
from earnings_alpha.stats.validation import stationary_bootstrap

__all__ = [  # noqa: RUF022 - orden temático, no alfabético
    "ExecutionAt",
    "SidePolicy",
    "SKIP_REASONS",
    "EventBacktest",
    "EventBacktestResult",
    "summarize_event_returns",
    "run_grid",
]

ExecutionAt = Literal["auto", "open", "close"]
"""Momento de ejecución dentro de la sesión. `"auto"` aplica la regla del módulo:
entrada en la apertura si `entry_offset >= 0`, al cierre si es negativo; salida
siempre al cierre."""

SidePolicy = Literal["long", "short", "signed"]
"""Lado de la posición por evento: todo largo, todo corto, o el signo del score."""

SKIP_REASONS: tuple[str, ...] = (
    "ticker_not_in_prices",
    "event_date_not_in_prices",
    "window_out_of_price_panel",
    "missing_prices",
    "dmh_excluded",
    "score_missing",
    "score_zero",
    "concurrency_limit",
)
"""Motivos posibles de descarte de un evento; aparecen en `result.skipped["reason"]`.
Ningún evento desaparece en silencio: o se ejecuta o figura aquí con su motivo."""

_PERCENTILES: tuple[int, ...] = (1, 5, 10, 25, 75, 90, 95, 99)


# --------------------------------------------------------------------------- #
# Puntos de ejecución                                                          #
# --------------------------------------------------------------------------- #


def _point_key(pos: int, at: str) -> int:
    """Clave ordenable de un punto de ejecución: la apertura precede al cierre."""
    return 2 * int(pos) + (1 if at == "close" else 0)


def _resolve_execution(
    entry_offset: int,
    exit_offset: int,
    entry_at: ExecutionAt,
    exit_at: ExecutionAt,
) -> tuple[str, str]:
    """Resuelve `"auto"` y valida que la entrada preceda estrictamente a la salida.

    La regla `auto` está razonada en el punto 2 del docstring del módulo: para
    `entry_offset >= 0` la apertura es la primera ejecución legal post-anuncio
    (en un AMC el gap queda fuera de la operación); para `entry_offset < 0`, el
    cierre es el último precio previo al evento.
    """
    for name, off in (("entry_offset", entry_offset), ("exit_offset", exit_offset)):
        if not isinstance(off, (int, np.integer)) or isinstance(off, bool):
            msg = f"{name} debe ser un entero de sesiones; recibido {off!r}"
            raise ConfigError(msg)
    for name, at in (("entry_at", entry_at), ("exit_at", exit_at)):
        if at not in ("auto", "open", "close"):
            msg = f"{name} debe ser 'auto', 'open' o 'close'; recibido {at!r}"
            raise ConfigError(msg)
    e_at = ("open" if entry_offset >= 0 else "close") if entry_at == "auto" else entry_at
    x_at = "close" if exit_at == "auto" else exit_at
    if _point_key(entry_offset, e_at) >= _point_key(exit_offset, x_at):
        msg = (
            f"la entrada ({entry_offset:+d}@{e_at}) no precede estrictamente a la "
            f"salida ({exit_offset:+d}@{x_at}): no hay periodo de tenencia"
        )
        raise ConfigError(msg)
    return e_at, x_at


# --------------------------------------------------------------------------- #
# Matrices de precios                                                          #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _PriceData:
    """Matrices `(n_sesiones, n_tickers)` derivadas del panel canónico de precios."""

    grid: pd.DatetimeIndex
    tickers: tuple[str, ...]
    col: dict[str, int]
    adj_open: np.ndarray
    adj_close: np.ndarray
    ln_open: np.ndarray
    ln_close: np.ndarray
    cum_gap: np.ndarray
    """`cum_gap[t]` = suma de log-retornos nocturnos (cierre->apertura) hasta `t`
    inclusive; la noche `t` es la transición `close[t-1] -> open[t]`."""
    cum_gap_nan: np.ndarray
    cum_intra: np.ndarray
    """`cum_intra[k]` = suma de log-retornos intradía de las sesiones `< k`
    (longitud `n+1`, fila 0 = 0), para sumar rangos cerrados sin casos borde."""
    cum_intra_nan: np.ndarray
    adv_usd: np.ndarray | None
    """ADV en dólares con la media móvil terminando la sesión ANTERIOR (PIT)."""
    sigma_daily: np.ndarray | None


def _prepare_prices(
    prices: pd.DataFrame,
    cal: TradingCalendar,
    *,
    adv_window: int,
    vol_window: int,
) -> _PriceData:
    """Valida el panel canónico y construye las matrices del motor.

    Exigencias (fallo explícito, nunca degradación silenciosa):

    - MultiIndex ``(date, ticker)`` y columnas ``open`` y ``close``. Sin precio de
      apertura **no hay descomposición gap/intradía posible** y el backtest sería
      exactamente el tearsheet inauditable contra el que advierte
      `pit_and_biases.md` §9.2, así que se lanza `DataQualityError`.
    - Las fechas del panel deben ser exactamente las sesiones del calendario entre
      la primera y la última: un hueco desalinearía la aritmética de offsets en
      sesiones (un `T+5` que en realidad es `T+6`).

    Ajuste por dividendos y splits: si el panel trae ``adj_close`` se aplica el
    factor ``adj_close/close`` también a la apertura, lo que asigna el ajuste del
    día ex-dividendo al tramo nocturno (el dividendo se descuenta en la apertura),
    y deja el tramo intradía idéntico al no ajustado. Sin ``adj_close`` se usan
    los precios tal cual (documentado: un split dentro de la ventana rompería el
    retorno; los proveedores del repo siempre sirven `adj_close`).
    """
    if not isinstance(prices, pd.DataFrame) or not isinstance(prices.index, pd.MultiIndex):
        msg = "`prices` debe ser el panel canónico con MultiIndex (date, ticker)"
        raise DataQualityError(msg)
    if list(prices.index.names) != ["date", "ticker"]:
        msg = f"el MultiIndex debe llamarse (date, ticker); recibido {prices.index.names}"
        raise DataQualityError(msg)
    for col in ("open", "close"):
        if col not in prices.columns:
            msg = (
                f"el panel de precios no tiene columna {col!r}; sin apertura y cierre "
                "no se puede descomponer el gap overnight (pit_and_biases.md §9)"
            )
            raise DataQualityError(msg)

    open_w = prices["open"].unstack("ticker").sort_index()
    close_w = prices["close"].unstack("ticker").sort_index()
    grid = open_w.index
    if len(grid) < 2:
        msg = f"el panel de precios tiene {len(grid)} sesiones; se necesitan al menos 2"
        raise InsufficientHistory(msg)

    expected = cal.sessions(grid[0].date(), grid[-1].date())
    if len(expected) != len(grid) or not np.array_equal(expected.values, grid.values):
        missing = expected.difference(grid)[:5].tolist()
        extra = grid.difference(expected)[:5].tolist()
        msg = (
            "las fechas del panel de precios no coinciden con las sesiones del "
            f"calendario en su rango (faltan p. ej. {missing}, sobran {extra}): "
            "la aritmética de offsets en sesiones sería incorrecta"
        )
        raise DataQualityError(msg)

    if "adj_close" in prices.columns:
        adj_close_w = prices["adj_close"].unstack("ticker").sort_index()
        factor = adj_close_w / close_w
    else:
        factor = 1.0
    adj_open = (open_w * factor).to_numpy(dtype=float)
    adj_close = (close_w * factor).to_numpy(dtype=float)

    with np.errstate(invalid="ignore", divide="ignore"):
        ln_open = np.where(adj_open > 0.0, np.log(np.where(adj_open > 0.0, adj_open, 1.0)), np.nan)
        ln_close = np.where(
            adj_close > 0.0, np.log(np.where(adj_close > 0.0, adj_close, 1.0)), np.nan
        )

    n = len(grid)
    ln_gap = np.full_like(ln_open, np.nan)
    ln_gap[1:] = ln_open[1:] - ln_close[:-1]
    ln_intra = ln_close - ln_open

    gap_nan = ~np.isfinite(ln_gap)
    intra_nan = ~np.isfinite(ln_intra)
    cum_gap = np.cumsum(np.where(gap_nan, 0.0, ln_gap), axis=0)
    cum_gap_nan = np.cumsum(gap_nan.astype(np.int64), axis=0)
    cum_intra = np.zeros((n + 1, ln_intra.shape[1]))
    cum_intra[1:] = np.cumsum(np.where(intra_nan, 0.0, ln_intra), axis=0)
    cum_intra_nan = np.zeros((n + 1, ln_intra.shape[1]), dtype=np.int64)
    cum_intra_nan[1:] = np.cumsum(intra_nan.astype(np.int64), axis=0)

    # ADV y sigma HACIA ATRÁS: la ventana termina la sesión anterior a la fecha en
    # que se consultan (shift(1)); usarlos el día de la ejecución sería mirar el
    # propio volumen del evento (pit_and_biases.md §12.6).
    adv_usd: np.ndarray | None = None
    if "dollar_volume" in prices.columns:
        dv = prices["dollar_volume"].unstack("ticker").sort_index()
    elif "volume" in prices.columns:
        dv = (prices["volume"].unstack("ticker").sort_index() * close_w).sort_index()
    else:
        dv = None
    min_p = max(5, adv_window // 2)
    if dv is not None:
        adv_usd = dv.rolling(adv_window, min_periods=min_p).mean().shift(1).to_numpy(dtype=float)

    ret_cc = pd.DataFrame(ln_close, index=grid, columns=open_w.columns).diff()
    sigma = (
        ret_cc.rolling(vol_window, min_periods=max(5, vol_window // 2))
        .std()
        .shift(1)
        .to_numpy(dtype=float)
    )

    tickers = tuple(str(t) for t in open_w.columns)
    return _PriceData(
        grid=grid,
        tickers=tickers,
        col={t: i for i, t in enumerate(tickers)},
        adj_open=adj_open,
        adj_close=adj_close,
        ln_open=ln_open,
        ln_close=ln_close,
        cum_gap=cum_gap,
        cum_gap_nan=cum_gap_nan,
        cum_intra=cum_intra,
        cum_intra_nan=cum_intra_nan,
        adv_usd=adv_usd,
        sigma_daily=sigma,
    )


# --------------------------------------------------------------------------- #
# Resultado                                                                    #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EventBacktestResult:
    """Resultado de una pasada del motor de eventos.

    Attributes
    ----------
    trades:
        Una fila por evento **ejecutado**, indexada por `event_id`. Incluye la
        descomposición exacta gap/intradía (`(1+gap_return)*(1+intraday_return)
        == 1+price_return`), la partición del día del anuncio y el desglose de
        costes. `net_return` y `gross_return` están en fracción del nocional del
        evento y ya llevan el signo de la posición.
    skipped:
        Eventos no ejecutados con su `reason` (ver `SKIP_REASONS`).
    daily:
        Serie diaria de la cartera entre la primera entrada y la última salida:
        `portfolio_return` (fracción del NAV, aditiva), `n_active`, `deployed`
        (capital comprometido, fracción del NAV) y `cum_net`.
    summary:
        Métricas agregadas (las mismas que una fila de `run_grid`).
    params:
        Parámetros efectivos de la pasada, para auditoría y reproducibilidad.
    """

    entry_offset: int
    exit_offset: int
    entry_at: str
    exit_at: str
    side: str
    trades: pd.DataFrame
    skipped: pd.DataFrame
    daily: pd.DataFrame
    summary: dict[str, object]
    params: dict[str, object]

    def summary_series(self) -> pd.Series:
        """El resumen como `pd.Series` (una fila de la tabla de `run_grid`)."""
        return pd.Series(self.summary)


# --------------------------------------------------------------------------- #
# Resumen estadístico por evento                                               #
# --------------------------------------------------------------------------- #


def _bootstrap_mean_ci(
    ordered_returns: np.ndarray,
    *,
    alpha: float,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    """IC de la media por bootstrap estacionario sobre los eventos en orden temporal.

    Los eventos concurrentes comparten días de mercado y por tanto están
    correlacionados en bloques temporales; el bootstrap estacionario sobre la
    serie ordenada por fecha de entrada (longitud de bloque Politis-White) recoge
    ese agrupamiento de forma aproximada — un iid naïf daría bandas demasiado
    estrechas (`stats.validation.stationary_bootstrap`, §9.4). Con menos de 8
    eventos o un estadístico degenerado la banda es NaN **explícita**, nunca una
    banda inventada.
    """
    try:
        boot = stationary_bootstrap(
            ordered_returns, n_boot=n_boot, alpha=alpha, seed=seed
        )
    except (InsufficientHistory, DataQualityError, ValueError, ZeroDivisionError):
        # Muestra corta o serie degenerada (varianza nula): banda NaN explícita.
        return float("nan"), float("nan")
    return boot.ci_low, boot.ci_high


def summarize_event_returns(
    trades: pd.DataFrame,
    *,
    alpha: float = 0.05,
    n_boot: int = 500,
    seed: int = 20260804,
    n_quantiles: int = 5,
) -> dict[str, object]:
    """Bloque de métricas por evento de una tabla `trades` del motor.

    Reporta exactamente lo que exige el contrato de la tarea: hit rate, retorno
    medio por evento **con banda de error**, percentiles (p01..p99: el retorno de
    evento tiene colas gruesas y la media con sigma no lo describe,
    `validation_methodology.md` §2), peor y mejor evento identificados,
    descomposición gap/intradía y resultado por cubo del score.

    Convenciones:

    - `mean_gap` / `mean_intraday` son las medias de los componentes nocturno e
      intradía **con el signo de la posición aplicado**: son los sumandos (en
      composición multiplicativa) del `mean_gross`.
    - `gap_share_log` = Σ(lado·log_gap) / Σ(lado·log_total): fracción del retorno
      logarítmico agregado que ocurrió con el mercado cerrado. Es inestable si el
      agregado es ~0 y en ese caso se reporta NaN, no un cociente absurdo.
    - Los cubos del score (`q1..qQ`) usan `backtest.portfolio.assign_quantiles`
      (rango determinista); si no hay score suficiente, salen NaN.
    """
    required = {
        "entry_date",
        "net_return",
        "gross_return",
        "cost_total",
        "side",
        "log_gap",
        "log_intraday",
        "gap_return",
        "intraday_return",
        "event_gap_return",
        "event_intraday_return",
        "holds_event_gap",
        "holding_sessions",
    }
    if not isinstance(trades, pd.DataFrame) or not required.issubset(trades.columns):
        missing = sorted(required - set(getattr(trades, "columns", [])))
        msg = f"`trades` no es una tabla del motor de eventos; faltan columnas {missing}"
        raise DataQualityError(msg)
    if len(trades) == 0:
        msg = "no hay eventos ejecutados que resumir"
        raise InsufficientHistory(msg)

    ordered = trades.reset_index(names="event_id").sort_values(
        ["entry_date", "event_id"], kind="stable"
    )
    net = ordered["net_return"].to_numpy(dtype=float)
    n = net.size
    ci_low, ci_high = _bootstrap_mean_ci(net, alpha=alpha, n_boot=n_boot, seed=seed)

    log_total = ordered["log_gap"].to_numpy() + ordered["log_intraday"].to_numpy()
    side = ordered["side"].to_numpy(dtype=float)
    signed_gap = float(np.sum(side * ordered["log_gap"].to_numpy()))
    signed_total = float(np.sum(side * log_total))
    gap_share = signed_gap / signed_total if abs(signed_total) > 1e-10 else float("nan")

    worst_pos = int(np.argmin(net))
    best_pos = int(np.argmax(net))
    pcts = np.percentile(net, _PERCENTILES)

    out: dict[str, object] = {
        "n_events": int(n),
        "hit_rate": float(np.mean(net > 0.0)),
        "mean_net": float(np.mean(net)),
        "mean_net_ci_low": ci_low,
        "mean_net_ci_high": ci_high,
        "median_net": float(np.median(net)),
        "std_net": float(np.std(net, ddof=1)) if n > 1 else float("nan"),
        "skew_net": float(sps.skew(net, bias=True)) if n > 2 else float("nan"),
        "kurtosis_raw_net": raw_kurtosis(net) if n > 3 else float("nan"),
    }
    for p, v in zip(_PERCENTILES, pcts, strict=True):
        out[f"p{p:02d}"] = float(v)
    out.update(
        {
            "worst_net": float(net[worst_pos]),
            "worst_event_id": str(ordered["event_id"].iloc[worst_pos]),
            "best_net": float(net[best_pos]),
            "best_event_id": str(ordered["event_id"].iloc[best_pos]),
            "mean_gross": float(ordered["gross_return"].mean()),
            "mean_cost": float(ordered["cost_total"].mean()),
            "mean_gap": float((side * ordered["gap_return"].to_numpy()).mean()),
            "mean_intraday": float((side * ordered["intraday_return"].to_numpy()).mean()),
            "gap_share_log": float(gap_share),
            "holds_event_gap_frac": float(ordered["holds_event_gap"].mean()),
            "mean_abs_event_gap": float(ordered["event_gap_return"].abs().mean()),
            "mean_abs_event_intraday": float(ordered["event_intraday_return"].abs().mean()),
            "avg_holding_sessions": float(ordered["holding_sessions"].mean()),
        }
    )
    if "session" in ordered.columns:
        sess = ordered["session"].astype(str).str.lower()
        out["amc_frac"] = float((sess == "amc").mean())
    else:
        out["amc_frac"] = float("nan")
    if "is_estimated_date" in ordered.columns:
        out["estimated_date_frac"] = float(
            ordered["is_estimated_date"].fillna(False).astype(bool).mean()
        )
    else:
        out["estimated_date_frac"] = float("nan")

    # ------------------------------------------------- resultado por cubo de score
    if "quantile" in ordered.columns:
        quant = ordered["quantile"]
    else:
        quant = pd.Series(np.nan, index=ordered.index)
    for k in range(1, n_quantiles + 1):
        mask = quant == float(k)
        sub = ordered.loc[np.asarray(mask, dtype=bool), "net_return"]
        out[f"q{k}_n"] = int(len(sub))
        out[f"q{k}_mean_net"] = float(sub.mean()) if len(sub) else float("nan")
        out[f"q{k}_hit_rate"] = float((sub > 0).mean()) if len(sub) else float("nan")

    top = ordered.loc[np.asarray(quant == float(n_quantiles), dtype=bool), "net_return"]
    bot = ordered.loc[np.asarray(quant == 1.0, dtype=bool), "net_return"]
    if len(top) >= 5 and len(bot) >= 5:
        spread = float(top.mean() - bot.mean())
        rng = np.random.default_rng(seed)
        t_arr, b_arr = top.to_numpy(), bot.to_numpy()
        reps = np.array(
            [
                np.mean(rng.choice(t_arr, size=t_arr.size, replace=True))
                - np.mean(rng.choice(b_arr, size=b_arr.size, replace=True))
                for _ in range(n_boot)
            ]
        )
        out["q_spread_mean"] = spread
        out["q_spread_ci_low"] = float(np.percentile(reps, 100.0 * alpha / 2.0))
        out["q_spread_ci_high"] = float(np.percentile(reps, 100.0 * (1.0 - alpha / 2.0)))
    else:
        out["q_spread_mean"] = float("nan")
        out["q_spread_ci_low"] = float("nan")
        out["q_spread_ci_high"] = float("nan")
    return out


# --------------------------------------------------------------------------- #
# Motor                                                                        #
# --------------------------------------------------------------------------- #


class EventBacktest:
    """Backtest dirigido por eventos con offsets de entrada/salida en sesiones.

    Contrato §3.7: *"Entra `entry_offset` sesiones antes/después del evento, sale
    en `exit_offset`. Modela gap overnight de forma explícita (el evento AMC se
    negocia en el gap)"*. Los detalles de semántica (puntos de ejecución, política
    PIT del pre-posicionamiento, concurrencia, costes) están en el docstring del
    módulo; los parámetros de construcción fijan lo que no cambia entre pasadas y
    `run()` recibe lo que sí (offsets, score, lado, límites).

    Parameters
    ----------
    calendar:
        Calendario de sesiones; por defecto `pit.get_calendar()`.
    cost_model:
        `backtest.costs.CostModel`. Por defecto el modelo realista con sus
        defaults documentados; `CostModel.zero()` debe pedirse explícitamente.
    nav_usd:
        NAV en dólares usado para convertir pesos en nocionales al calcular la
        participación sobre el ADV (impacto de mercado). 10 M$ por defecto: un
        sleeve institucional pequeño; con NAV mayor los costes de impacto crecen.
    adv_window, vol_window:
        Ventanas (sesiones) de la media de dólar-volumen y de la volatilidad
        diaria, ambas terminando la sesión **anterior** a la ejecución.
    event_open_spread_multiplier:
        Multiplicador del medio spread para ejecuciones en la **apertura de
        `tau = 0`** (la horquilla tras un anuncio es un múltiplo de la normal,
        `pit_and_biases.md` §9.5). 3x por defecto.
    dmh_policy:
        Qué hacer con anuncios DMH (durante la sesión) cuando la entrada caería
        en la apertura de `tau = 0`, que para un DMH sería **anterior al
        anuncio**: `"exclude"` (defecto) los descarta con motivo; `"delay"`
        retrasa la entrada al cierre de `tau = 0` (primer precio limpio).
    seed:
        Semilla de los bootstraps del resumen (regla 4 del repo: determinismo).
    """

    def __init__(
        self,
        *,
        calendar: TradingCalendar | None = None,
        cost_model: CostModel | None = None,
        nav_usd: float = 10_000_000.0,
        adv_window: int = 21,
        vol_window: int = 21,
        event_open_spread_multiplier: float = 3.0,
        dmh_policy: Literal["exclude", "delay"] = "exclude",
        seed: int = 20260804,
    ) -> None:
        if not math.isfinite(nav_usd) or nav_usd <= 0.0:
            msg = f"nav_usd debe ser positivo y finito; recibido {nav_usd!r}"
            raise ConfigError(msg)
        if adv_window < 2 or vol_window < 2:
            msg = f"adv_window y vol_window deben ser >= 2; recibidos {adv_window}, {vol_window}"
            raise ConfigError(msg)
        if not math.isfinite(event_open_spread_multiplier) or event_open_spread_multiplier < 1.0:
            msg = (
                "event_open_spread_multiplier debe ser >= 1 (la apertura post-anuncio "
                f"nunca es más barata que lo normal); recibido {event_open_spread_multiplier!r}"
            )
            raise ConfigError(msg)
        if dmh_policy not in ("exclude", "delay"):
            msg = f"dmh_policy debe ser 'exclude' o 'delay'; recibido {dmh_policy!r}"
            raise ConfigError(msg)
        self.calendar = calendar or get_calendar()
        self.cost_model = cost_model if cost_model is not None else CostModel()
        self.nav_usd = float(nav_usd)
        self.adv_window = int(adv_window)
        self.vol_window = int(vol_window)
        self.event_open_spread_multiplier = float(event_open_spread_multiplier)
        self.dmh_policy = dmh_policy
        self.seed = int(seed)

    # ------------------------------------------------------------------ run

    def run(
        self,
        events: pd.DataFrame,
        prices: pd.DataFrame,
        *,
        entry_offset: int,
        exit_offset: int,
        score: pd.Series | str | None = None,
        side: SidePolicy = "long",
        max_concurrent: int | None = 20,
        per_event_capital: float | None = None,
        calendar_known_in_advance: bool = False,
        entry_at: ExecutionAt = "auto",
        exit_at: ExecutionAt = "auto",
        n_quantiles: int = 5,
        min_events: int = 10,
        alpha: float = 0.05,
        n_boot: int = 500,
    ) -> EventBacktestResult:
        """Ejecuta el backtest para una combinación `(entry_offset, exit_offset)`.

        Parameters
        ----------
        events:
            Tabla de anuncios; se canoniza con `events.window.normalize_events`
            (necesita `ticker` y `event_date`, o `announced_at` + `session` para
            derivarla vía `pit.tradable_date`).
        prices:
            Panel canónico ``(date, ticker)`` con al menos ``open`` y ``close``
            (idealmente también ``adj_close`` y ``dollar_volume``).
        entry_offset, exit_offset:
            Desplazamientos en **sesiones** respecto a `tau = 0` (la sesión
            negociable). Negativo = antes del anuncio.
        score:
            Puntuación por evento: nombre de columna de `events` o `pd.Series`
            indexada por `event_id`. Define los cubos del resumen, la prioridad
            bajo el límite de concurrencia y el lado si `side="signed"`.
            **Responsabilidad PIT del llamante**: con `entry_offset < 0` el score
            debe derivar exclusivamente de datos pre-evento (p. ej.
            `events.PreEventFeatures`); un SUE del propio anuncio es legal solo
            con `entry_offset >= 0`, porque el anuncio ya es público en la
            apertura de `tau = 0`.
        side:
            ``"long"``, ``"short"`` o ``"signed"`` (signo del score).
        max_concurrent:
            Máximo de posiciones simultáneas (`None` = sin límite). El exceso se
            descarta con motivo `"concurrency_limit"`, priorizando |score| alto.
        per_event_capital:
            Fracción del NAV por evento. Por defecto `1/max_concurrent`, o 0.05
            (20 huecos nominales) si no hay límite.
        calendar_known_in_advance:
            Declaración explícita de que la fecha de cada evento era conocible en
            la fecha de entrada. **Obligatoria** si `entry_offset < 0`
            (`pit_and_biases.md` §8.3); sin ella se lanza `LookAheadError`.
        min_events:
            Mínimo de eventos ejecutados; por debajo, `InsufficientHistory` (un
            resumen sobre 3 eventos no es un backtest, es una anécdota).

        Raises
        ------
        LookAheadError
            Pre-posicionamiento sin vintages de calendario ni declaración.
        DataQualityError, ConfigError, InsufficientHistory
            Entradas defectuosas o muestra insuficiente, siempre con mensaje.
        """
        e_at_global, x_at_global = _resolve_execution(entry_offset, exit_offset, entry_at, exit_at)
        if entry_offset < 0 and not calendar_known_in_advance:
            msg = (
                f"entry_offset={entry_offset} entra ANTES del anuncio y la fecha del "
                "evento no era necesariamente conocible entonces: hacen falta vintages "
                "de calendario (pit_and_biases.md §8.3, CalendarVintage/known_event_date). "
                "Si las fechas ya están resueltas point-in-time (o son sintéticas, con "
                "calendario público por adelantado), decláralo con "
                "calendar_known_in_advance=True"
            )
            raise LookAheadError(msg)
        if side not in ("long", "short", "signed"):
            msg = f"side debe ser 'long', 'short' o 'signed'; recibido {side!r}"
            raise ConfigError(msg)
        if max_concurrent is not None and max_concurrent < 1:
            msg = f"max_concurrent debe ser >= 1 o None; recibido {max_concurrent!r}"
            raise ConfigError(msg)
        if per_event_capital is None:
            weight = 1.0 / max_concurrent if max_concurrent is not None else 0.05
        else:
            if not math.isfinite(per_event_capital) or not 0.0 < per_event_capital <= 1.0:
                msg = f"per_event_capital debe estar en (0, 1]; recibido {per_event_capital!r}"
                raise ConfigError(msg)
            weight = float(per_event_capital)
        if min_events < 1:
            msg = f"min_events debe ser >= 1; recibido {min_events!r}"
            raise ConfigError(msg)
        if not 0.0 < alpha < 1.0:
            msg = f"alpha debe estar en (0, 1); recibido {alpha!r}"
            raise ConfigError(msg)

        base = normalize_events(events, self.calendar)
        score_by_event = self._resolve_score(base, score)
        data = _prepare_prices(
            prices, self.calendar, adv_window=self.adv_window, vol_window=self.vol_window
        )

        cand, skipped = self._build_candidates(
            base,
            data,
            score_by_event,
            side=side,
            entry_offset=entry_offset,
            exit_offset=exit_offset,
            entry_at=e_at_global,
            exit_at=x_at_global,
        )
        accepted, conc_skips = self._apply_concurrency(cand, max_concurrent)
        skipped.extend(conc_skips)

        if len(accepted) < min_events:
            reasons = (
                pd.Series([s["reason"] for s in skipped], dtype=object).value_counts().to_dict()
            )
            msg = (
                f"solo {len(accepted)} eventos ejecutables de {len(base)} candidatos "
                f"(mínimo {min_events}); descartes por motivo: {reasons}"
            )
            raise InsufficientHistory(msg)

        trades = self._compute_trades(accepted, data, weight)
        trades["quantile"] = assign_quantiles(trades["score"], n_quantiles)

        daily = self._daily_accounting(trades, data)
        summary = self._summarize(
            trades,
            daily,
            n_candidates=len(base),
            n_skipped=skipped,
            entry_offset=entry_offset,
            exit_offset=exit_offset,
            entry_at=e_at_global,
            exit_at=x_at_global,
            max_concurrent=max_concurrent,
            weight=weight,
            alpha=alpha,
            n_boot=n_boot,
            n_quantiles=n_quantiles,
        )
        skipped_df = pd.DataFrame(
            skipped, columns=["event_id", "ticker", "event_date", "reason"]
        ).reset_index(drop=True)
        params = {
            "entry_offset": int(entry_offset),
            "exit_offset": int(exit_offset),
            "entry_at": e_at_global,
            "exit_at": x_at_global,
            "side": side,
            "max_concurrent": max_concurrent,
            "per_event_capital": weight,
            "nav_usd": self.nav_usd,
            "adv_window": self.adv_window,
            "vol_window": self.vol_window,
            "event_open_spread_multiplier": self.event_open_spread_multiplier,
            "dmh_policy": self.dmh_policy,
            "calendar_known_in_advance": bool(calendar_known_in_advance),
            "n_quantiles": int(n_quantiles),
            "min_events": int(min_events),
            "alpha": float(alpha),
            "n_boot": int(n_boot),
            "seed": self.seed,
        }
        return EventBacktestResult(
            entry_offset=int(entry_offset),
            exit_offset=int(exit_offset),
            entry_at=e_at_global,
            exit_at=x_at_global,
            side=side,
            trades=trades,
            skipped=skipped_df,
            daily=daily,
            summary=summary,
            params=params,
        )

    # ------------------------------------------------------------- score/lado

    @staticmethod
    def _resolve_score(base: pd.DataFrame, score: pd.Series | str | None) -> pd.Series:
        """Score por evento como Series float indexada por `event_id` (NaN = sin score)."""
        ids = pd.Index(base["event_id"], name="event_id")
        if score is None:
            return pd.Series(np.nan, index=ids, name="score")
        if isinstance(score, str):
            if score not in base.columns:
                msg = (
                    f"score={score!r} no es una columna de la tabla de eventos; "
                    f"disponibles p. ej. {sorted(base.columns)[:10]}"
                )
                raise ConfigError(msg)
            return pd.Series(
                pd.to_numeric(base[score], errors="coerce").to_numpy(dtype=float),
                index=ids,
                name="score",
            )
        if isinstance(score, pd.Series):
            if isinstance(score.index, pd.MultiIndex):
                msg = "el score por evento debe indexarse por event_id, no por MultiIndex"
                raise DataQualityError(msg)
            if score.index.has_duplicates:
                msg = "el score tiene event_id duplicados"
                raise DataQualityError(msg)
            numeric = pd.to_numeric(score, errors="coerce")
            return pd.Series(
                numeric.reindex(ids.astype(str)).to_numpy(dtype=float), index=ids, name="score"
            )
        msg = f"score debe ser None, nombre de columna o pd.Series; recibido {type(score)!r}"
        raise ConfigError(msg)

    # --------------------------------------------------------------- candidatos

    def _build_candidates(
        self,
        base: pd.DataFrame,
        data: _PriceData,
        score_by_event: pd.Series,
        *,
        side: SidePolicy,
        entry_offset: int,
        exit_offset: int,
        entry_at: str,
        exit_at: str,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        """Valida evento a evento y separa candidatos ejecutables de descartes.

        Toda la validación de precios ocurre **antes** del límite de concurrencia:
        un evento inejecutable no debe ocupar un hueco de cartera.
        """
        n_dates = len(data.grid)
        grid_values = data.grid.values

        ids = base["event_id"].astype(str).to_numpy()
        tickers = base["ticker"].astype(str).to_numpy()
        event_dates = base["event_date"].to_numpy(dtype="datetime64[ns]")
        if "session" in base.columns:
            sessions_arr = base["session"].astype(str).str.lower().to_numpy()
        else:
            sessions_arr = np.full(len(base), "", dtype=object)
        if "is_estimated_date" in base.columns:
            est_arr = base["is_estimated_date"].fillna(False).astype(bool).to_numpy()
        else:
            est_arr = np.zeros(len(base), dtype=bool)
        scores_arr = score_by_event.to_numpy(dtype=float)
        pos0_arr = np.searchsorted(grid_values, event_dates)

        candidates: list[dict[str, object]] = []
        skipped: list[dict[str, object]] = []

        def skip(i: int, reason: str) -> None:
            skipped.append(
                {
                    "event_id": ids[i],
                    "ticker": tickers[i],
                    "event_date": pd.Timestamp(event_dates[i]),
                    "reason": reason,
                }
            )

        for i in range(len(base)):
            ticker = tickers[i]
            if ticker not in data.col:
                skip(i, "ticker_not_in_prices")
                continue
            j = data.col[ticker]
            pos0 = int(pos0_arr[i])
            if pos0 >= n_dates or grid_values[pos0] != event_dates[i]:
                skip(i, "event_date_not_in_prices")
                continue

            e_at, x_at = entry_at, exit_at
            session = str(sessions_arr[i])
            if session == "dmh" and entry_offset == 0 and e_at == "open":
                # La apertura de tau=0 de un DMH es ANTERIOR al anuncio: entrar ahí
                # sería look-ahead (pit_and_biases.md §9.4, caso 3).
                if self.dmh_policy == "exclude":
                    skip(i, "dmh_excluded")
                    continue
                e_at = "close"
                if _point_key(entry_offset, e_at) >= _point_key(exit_offset, x_at):
                    skip(i, "dmh_excluded")
                    continue

            e_pos = pos0 + entry_offset
            x_pos = pos0 + exit_offset
            if e_pos < 0 or x_pos > n_dates - 1:
                skip(i, "window_out_of_price_panel")
                continue

            sc = float(scores_arr[i])
            if side == "signed":
                if not np.isfinite(sc):
                    skip(i, "score_missing")
                    continue
                if sc == 0.0:
                    skip(i, "score_zero")
                    continue
                sd = 1.0 if sc > 0 else -1.0
            else:
                sd = 1.0 if side == "long" else -1.0

            entry_ln = data.ln_open[e_pos, j] if e_at == "open" else data.ln_close[e_pos, j]
            exit_ln = data.ln_open[x_pos, j] if x_at == "open" else data.ln_close[x_pos, j]
            n_gap_nan = int(data.cum_gap_nan[x_pos, j] - data.cum_gap_nan[e_pos, j])
            ie = e_pos + (1 if e_at == "close" else 0)
            ix = x_pos - (1 if x_at == "open" else 0)
            n_intra_nan = (
                int(data.cum_intra_nan[ix + 1, j] - data.cum_intra_nan[ie, j]) if ie <= ix else 0
            )
            if (
                not np.isfinite(entry_ln)
                or not np.isfinite(exit_ln)
                or n_gap_nan > 0
                or n_intra_nan > 0
            ):
                skip(i, "missing_prices")
                continue

            candidates.append(
                {
                    "event_id": ids[i],
                    "ticker": ticker,
                    "col": j,
                    "event_date": pd.Timestamp(event_dates[i]),
                    "session": session,
                    "is_estimated_date": bool(est_arr[i]),
                    "pos0": pos0,
                    "e_pos": e_pos,
                    "x_pos": x_pos,
                    "e_at": e_at,
                    "x_at": x_at,
                    "ie": ie,
                    "ix": ix,
                    "score": sc,
                    "side": sd,
                }
            )
        return candidates, skipped

    # -------------------------------------------------------------- concurrencia

    @staticmethod
    def _apply_concurrency(
        candidates: list[dict[str, object]],
        max_concurrent: int | None,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        """Aplica el límite de posiciones simultáneas.

        Los candidatos se procesan por instante de entrada; un hueco se libera en
        el instante de salida (una salida al cierre libera capital para una
        entrada en ese mismo cierre: el roll simultáneo es ejecutable). Con más
        candidatos que huecos gana el |score| mayor y el empate se resuelve por
        ticker y `event_id` (determinismo, regla 4 del repo).
        """

        def priority(c: dict[str, object]) -> tuple[int, float, str, str]:
            sc = float(c["score"])
            return (
                _point_key(int(c["e_pos"]), str(c["e_at"])),
                -(abs(sc) if np.isfinite(sc) else 0.0),
                str(c["ticker"]),
                str(c["event_id"]),
            )

        ordered = sorted(candidates, key=priority)
        if max_concurrent is None:
            return ordered, []

        accepted: list[dict[str, object]] = []
        rejected: list[dict[str, object]] = []
        exit_heap: list[int] = []
        for c in ordered:
            entry_key = _point_key(int(c["e_pos"]), str(c["e_at"]))
            exit_key = _point_key(int(c["x_pos"]), str(c["x_at"]))
            while exit_heap and exit_heap[0] <= entry_key:
                heapq.heappop(exit_heap)
            if len(exit_heap) < max_concurrent:
                accepted.append(c)
                heapq.heappush(exit_heap, exit_key)
            else:
                rejected.append(
                    {
                        "event_id": c["event_id"],
                        "ticker": c["ticker"],
                        "event_date": c["event_date"],
                        "reason": "concurrency_limit",
                    }
                )
        return accepted, rejected

    # ------------------------------------------------------------------ trades

    def _compute_trades(
        self,
        accepted: list[dict[str, object]],
        data: _PriceData,
        weight: float,
    ) -> pd.DataFrame:
        """Retornos, descomposición gap/intradía y costes de los eventos aceptados.

        La descomposición es exacta por construcción telescópica: el log-retorno
        total de la tenencia es la suma de los tramos nocturnos
        (`cierre[t-1] -> apertura[t]`) y los intradía (`apertura[t] -> cierre[t]`)
        efectivamente mantenidos, de modo que
        ``(1+gap_return)·(1+intraday_return) == 1+price_return`` al épsilon de
        máquina. Los tests lo verifican operación a operación.
        """
        n = len(accepted)
        cols = np.array([c["col"] for c in accepted], dtype=int)
        pos0 = np.array([c["pos0"] for c in accepted], dtype=int)
        e_pos = np.array([c["e_pos"] for c in accepted], dtype=int)
        x_pos = np.array([c["x_pos"] for c in accepted], dtype=int)
        ie = np.array([c["ie"] for c in accepted], dtype=int)
        ix = np.array([c["ix"] for c in accepted], dtype=int)
        e_open = np.array([c["e_at"] == "open" for c in accepted])
        x_open = np.array([c["x_at"] == "open" for c in accepted])
        side_arr = np.array([c["side"] for c in accepted], dtype=float)
        tickers = [str(c["ticker"]) for c in accepted]

        log_gap = data.cum_gap[x_pos, cols] - data.cum_gap[e_pos, cols]
        has_intra = ie <= ix
        log_intra = np.where(
            has_intra,
            data.cum_intra[np.minimum(ix, len(data.grid) - 1) + 1, cols]
            - data.cum_intra[np.clip(ie, 0, len(data.grid)), cols],
            0.0,
        )
        log_total = log_gap + log_intra
        price_return = np.expm1(log_total)
        gap_return = np.expm1(log_gap)
        intraday_return = np.expm1(log_intra)

        entry_price = np.where(
            e_open, data.adj_open[e_pos, cols], data.adj_close[e_pos, cols]
        )
        exit_price = np.where(x_open, data.adj_open[x_pos, cols], data.adj_close[x_pos, cols])

        # Partición del día del anuncio (events.overnight_decomposition, §9.5):
        # gap = open(tau0)/close(tau0-1) - 1, intradía = close(tau0)/open(tau0) - 1.
        event_gap = np.full(n, np.nan)
        valid_prev = pos0 >= 1
        ev_gap_log = data.cum_gap[pos0, cols] - data.cum_gap[np.maximum(pos0 - 1, 0), cols]
        event_gap[valid_prev] = np.expm1(ev_gap_log[valid_prev])
        event_intraday = np.expm1(data.ln_close[pos0, cols] - data.ln_open[pos0, cols])

        entry_key = 2 * e_pos + (~e_open).astype(int)
        exit_key = 2 * x_pos + (~x_open).astype(int)
        holds_event_gap = (entry_key <= 2 * (pos0 - 1) + 1) & (exit_key >= 2 * pos0) & valid_prev

        # ------------------------------------------------------------- costes
        cm = self.cost_model
        if data.adv_usd is not None:
            adv_entry = data.adv_usd[e_pos, cols]
            adv_exit = data.adv_usd[x_pos, cols]
        else:
            adv_entry = np.full(n, np.nan)
            adv_exit = np.full(n, np.nan)
        if data.sigma_daily is not None:
            sig_entry = data.sigma_daily[e_pos, cols]
            sig_exit = data.sigma_daily[x_pos, cols]
        else:
            sig_entry = np.full(n, np.nan)
            sig_exit = np.full(n, np.nan)

        notional = weight * self.nav_usd

        def participation(adv: np.ndarray) -> np.ndarray:
            known = np.isfinite(adv) & (adv > 0.0)
            if cm.default_adv_usd is not None:
                adv = np.where(known, adv, cm.default_adv_usd)
                known = adv > 0.0
            return np.where(known, notional / np.where(known, adv, 1.0), 1.0)

        mult_entry = np.where(
            e_open & (e_pos == pos0), self.event_open_spread_multiplier, 1.0
        )
        mult_exit = np.where(x_open & (x_pos == pos0), self.event_open_spread_multiplier, 1.0)
        commission = cm.commission_bps * BPS
        cost_entry = (
            cm.half_spread_fraction(adv_entry) * mult_entry
            + cm.impact_fraction(participation(adv_entry), sig_entry)
            + commission
        )
        cost_exit = (
            cm.half_spread_fraction(adv_exit) * mult_exit
            + cm.impact_fraction(participation(adv_exit), sig_exit)
            + commission
        )

        entry_dates = data.grid[e_pos]
        exit_dates = data.grid[x_pos]
        calendar_days = (exit_dates.values - entry_dates.values).astype("timedelta64[D]").astype(
            int
        )
        borrow_rates = cm.borrow_daily_rates(tickers)
        cost_borrow = np.where(side_arr < 0, borrow_rates * calendar_days, 0.0)

        gross = side_arr * price_return
        cost_total = cost_entry + cost_exit + cost_borrow
        net = gross - cost_total

        frame = pd.DataFrame(
            {
                "ticker": tickers,
                "event_date": [c["event_date"] for c in accepted],
                "session": [c["session"] for c in accepted],
                "is_estimated_date": [c["is_estimated_date"] for c in accepted],
                "score": [float(c["score"]) for c in accepted],
                "side": side_arr,
                "weight": weight,
                "entry_date": entry_dates,
                "entry_at": np.where(e_open, "open", "close"),
                "exit_date": exit_dates,
                "exit_at": np.where(x_open, "open", "close"),
                "holding_sessions": x_pos - e_pos,
                "calendar_days": calendar_days,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "price_return": price_return,
                "gap_return": gap_return,
                "intraday_return": intraday_return,
                "log_total": log_total,
                "log_gap": log_gap,
                "log_intraday": log_intra,
                "event_gap_return": event_gap,
                "event_intraday_return": event_intraday,
                "holds_event_gap": holds_event_gap,
                "gross_return": gross,
                "cost_entry": cost_entry,
                "cost_exit": cost_exit,
                "cost_borrow": cost_borrow,
                "cost_total": cost_total,
                "net_return": net,
                "_pos0": pos0,
                "_e_pos": e_pos,
                "_x_pos": x_pos,
                "_col": cols,
            },
            index=pd.Index([c["event_id"] for c in accepted], name="event_id"),
        )
        return frame.sort_values(["entry_date", "ticker"], kind="stable")

    # ------------------------------------------------------------------ diario

    def _daily_accounting(self, trades: pd.DataFrame, data: _PriceData) -> pd.DataFrame:
        """Serie diaria de la cartera con número fijo de acciones por posición.

        Cada posición mantiene `peso·NAV/precio_entrada` acciones, así que el P&L
        del día `t` es `peso·lado·(V_fin(t) − V_ini(t))/precio_entrada`, con
        `V_ini` el cierre previo (si se mantuvo la noche) o el precio de entrada,
        y `V_fin` el cierre del día o el precio de salida. La suma temporal
        reproduce exactamente `peso·net_return` por evento (los costes se cargan
        el día en que se devengan: ejecución en entrada/salida, préstamo a la
        salida). La serie es **aditiva** (sin capitalización intra-estrategia):
        es la contabilidad de horizonte por evento, no la de un NAV compuesto.
        """
        n_dates = len(data.grid)
        pnl = np.zeros(n_dates)
        n_active = np.zeros(n_dates, dtype=int)
        deployed = np.zeros(n_dates)

        e_arr = trades["_e_pos"].to_numpy(dtype=int)
        x_arr = trades["_x_pos"].to_numpy(dtype=int)
        col_arr = trades["_col"].to_numpy(dtype=int)
        side_arr = trades["side"].to_numpy(dtype=float)
        w_arr = trades["weight"].to_numpy(dtype=float)
        entry_p = trades["entry_price"].to_numpy(dtype=float)
        exit_p = trades["exit_price"].to_numpy(dtype=float)
        c_entry = trades["cost_entry"].to_numpy(dtype=float)
        c_exit = trades["cost_exit"].to_numpy(dtype=float)
        c_borrow = trades["cost_borrow"].to_numpy(dtype=float)

        for k in range(len(trades)):
            e, x, j = int(e_arr[k]), int(x_arr[k]), int(col_arr[k])
            side, w = side_arr[k], w_arr[k]
            closes = data.adj_close[:, j]
            for t in range(e, x + 1):
                start = entry_p[k] if t == e else closes[t - 1]
                end = exit_p[k] if t == x else closes[t]
                pnl[t] += w * side * (end - start) / entry_p[k]
            pnl[e] -= w * c_entry[k]
            pnl[x] -= w * (c_exit[k] + c_borrow[k])
            n_active[e : x + 1] += 1
            deployed[e : x + 1] += w

        first = int(trades["_e_pos"].min())
        last = int(trades["_x_pos"].max())
        daily = pd.DataFrame(
            {
                "portfolio_return": pnl[first : last + 1],
                "n_active": n_active[first : last + 1],
                "deployed": deployed[first : last + 1],
            },
            index=data.grid[first : last + 1].rename("date"),
        )
        daily["cum_net"] = daily["portfolio_return"].cumsum()
        return daily

    # ------------------------------------------------------------------ resumen

    def _summarize(
        self,
        trades: pd.DataFrame,
        daily: pd.DataFrame,
        *,
        n_candidates: int,
        n_skipped: list[dict[str, object]],
        entry_offset: int,
        exit_offset: int,
        entry_at: str,
        exit_at: str,
        max_concurrent: int | None,
        weight: float,
        alpha: float,
        n_boot: int,
        n_quantiles: int,
    ) -> dict[str, object]:
        """Bloque de resumen: métricas por evento + métricas de cartera."""
        out: dict[str, object] = {
            "status": "ok",
            "entry_offset": int(entry_offset),
            "exit_offset": int(exit_offset),
            "entry_at": entry_at,
            "exit_at": exit_at,
            "n_candidates": int(n_candidates),
            "n_skipped_total": int(len(n_skipped)),
            "n_skipped_concurrency": int(
                sum(1 for s in n_skipped if s["reason"] == "concurrency_limit")
            ),
        }
        out.update(
            summarize_event_returns(
                trades, alpha=alpha, n_boot=n_boot, seed=self.seed, n_quantiles=n_quantiles
            )
        )

        limit_gross = max_concurrent * weight if max_concurrent is not None else float("nan")
        dep = daily["deployed"].to_numpy(dtype=float)
        out["avg_deployed"] = float(dep.mean())
        out["max_deployed"] = float(dep.max())
        out["pct_days_at_limit"] = (
            float(np.mean(dep >= limit_gross - 1e-12)) if np.isfinite(limit_gross) else float("nan")
        )
        out["total_net_pnl"] = float(
            (trades["weight"] * trades["net_return"]).sum()
        )

        # Sharpe diario anualizado con banda (Mertens); NaN explícito si la serie
        # es demasiado corta o degenerada — nunca una banda inventada.
        returns = daily["portfolio_return"]
        try:
            sr = sharpe_ratio(returns, alpha=alpha)
            out["sharpe_annualized"] = sr.sharpe_annualized
            out["sharpe_ci_low"] = sr.ci_low
            out["sharpe_ci_high"] = sr.ci_high
        except (DataQualityError, InsufficientHistory):
            out["sharpe_annualized"] = float("nan")
            out["sharpe_ci_low"] = float("nan")
            out["sharpe_ci_high"] = float("nan")
        return out


# --------------------------------------------------------------------------- #
# Comparador de rejilla                                                        #
# --------------------------------------------------------------------------- #


def run_grid(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    entry_offsets: Sequence[int],
    exit_offsets: Sequence[int],
    score: pd.Series | str | None = None,
    side: SidePolicy = "long",
    engine: EventBacktest | None = None,
    calendar_known_in_advance: bool = False,
    max_concurrent: int | None = 20,
    per_event_capital: float | None = None,
    entry_at: ExecutionAt = "auto",
    exit_at: ExecutionAt = "auto",
    n_quantiles: int = 5,
    min_events: int = 10,
    alpha: float = 0.05,
    n_boot: int = 500,
) -> pd.DataFrame:
    """Barre combinaciones `(entry_offset, exit_offset)` y devuelve la tabla comparativa.

    Es el entregable con el que se decide **cuándo entrar y cuándo salir**: por
    ejemplo, comparar "cruzar el anuncio" (entrar `T-5`, salir `T+1`: cobra el
    salto, asume el riesgo del gap) contra "explotar el drift" (entrar `T+1`,
    salir `T+60`: renuncia al salto, cobra el PEAD de Bernard y Thomas 1989 sin
    riesgo de anuncio). Cada fila reporta hit rate, retorno medio por evento con
    su banda, percentiles p01..p99 (colas gruesas: la media y la sigma no bastan),
    peor y mejor evento identificados, la descomposición gap/intradía, la
    utilización del capital y el resultado por cubo del score.

    Semántica de filas:

    - ``status == "ok"``: la combinación se ejecutó.
    - ``status == "invalid_offsets"``: la entrada no precede a la salida (p. ej.
      entrar `T+1` y salir `T+0`); las métricas quedan NaN. La fila se conserva
      para que la tabla sea completa y el hueco sea visible, no silencioso.
    - ``status == "insufficient_events"``: menos de `min_events` eventos
      ejecutables (p. ej. una salida `T+60` que se sale del panel de precios).

    Cualquier otra condición de error (pre-posicionamiento sin declarar la
    conocibilidad del calendario, panel de precios defectuoso...) **propaga la
    excepción**: son errores del experimento, no resultados.

    Nota de multiplicidad: una rejilla de `N` combinaciones son `N` pruebas sobre
    los mismos datos. Antes de creerse la mejor celda, pásela por
    `stats.validation.benjamini_hochberg` o `stats.performance.deflated_sharpe_ratio`
    con `n_trials = N` (`validation_methodology.md` §7).
    """
    entry_list = [int(e) for e in entry_offsets]
    exit_list = [int(x) for x in exit_offsets]
    if not entry_list or not exit_list:
        msg = "entry_offsets y exit_offsets no pueden estar vacíos"
        raise ConfigError(msg)
    eng = engine if engine is not None else EventBacktest()

    rows: dict[tuple[int, int], dict[str, object]] = {}
    ok_columns: list[str] | None = None
    for eo in entry_list:
        for xo in exit_list:
            try:
                _resolve_execution(eo, xo, entry_at, exit_at)
            except ConfigError:
                rows[(eo, xo)] = {
                    "status": "invalid_offsets",
                    "entry_offset": eo,
                    "exit_offset": xo,
                }
                continue
            try:
                result = eng.run(
                    events,
                    prices,
                    entry_offset=eo,
                    exit_offset=xo,
                    score=score,
                    side=side,
                    max_concurrent=max_concurrent,
                    per_event_capital=per_event_capital,
                    calendar_known_in_advance=calendar_known_in_advance,
                    entry_at=entry_at,
                    exit_at=exit_at,
                    n_quantiles=n_quantiles,
                    min_events=min_events,
                    alpha=alpha,
                    n_boot=n_boot,
                )
            except InsufficientHistory:
                rows[(eo, xo)] = {
                    "status": "insufficient_events",
                    "entry_offset": eo,
                    "exit_offset": xo,
                }
                continue
            rows[(eo, xo)] = result.summary
            if ok_columns is None:
                ok_columns = list(result.summary.keys())

    index = pd.MultiIndex.from_tuples(rows.keys(), names=["entry_offset", "exit_offset"])
    table = pd.DataFrame(list(rows.values()), index=index)
    if ok_columns is not None:
        extra = [c for c in table.columns if c not in ok_columns]
        table = table.reindex(columns=[*ok_columns, *extra])
    return table
