"""Features de flujo pre-anuncio: la huella observable de la negociación informada.

NOTA LEGAL Y METODOLÓGICA OBLIGATORIA (contrato §3.5)
-----------------------------------------------------
Todas las features de este módulo derivan de datos **públicos**: volumen y precios
consolidados (OHLCV), short interest agregado que publica FINRA, ficheros de
transparencia ATS/OTC de FINRA y formularios Form 4 ya presentados en EDGAR. El
objetivo es detectar la **huella estadística** que la negociación informada deja en
variables observables por cualquiera, no acceder a información material no pública.
Ninguna función de este módulo requiere ni admite información privilegiada.

Diseño general
--------------
Cada función pública recibe la tabla de eventos (``event_id, ticker, event_date``,
donde ``event_date`` es la **primera sesión negociable** del anuncio, según
`pit.tradable_date`) y devuelve una serie o tabla indexada por ``event_id``. Todas
las ventanas se expresan en tiempo-evento ``tau`` (sesiones): la ventana de
detección es ``[T-k, T-1]`` y la ventana base por defecto ``[T-70, T-21]`` (50
sesiones), que termina en ``T-21`` para no solaparse con la detección
(`docs/research/informed_trading.md` §3.3). **Ninguna feature usa datos de la
sesión T ni posteriores**: ese es el invariante que los tests de
`tests/test_preevent.py` verifican perturbando el futuro.

Decisiones tomadas del informe `docs/research/informed_trading.md` (citado aquí
como *el informe*), todas con verificación numérica allí documentada:

* El z-score de turnover corrige la autocorrelación con el tamaño muestral
  efectivo **exacto** de un AR(1) —``n_eff = k² / (k + 2·Σ_{j=1..k-1}(k-j)·ρ^j)``—
  y nunca con la aproximación asintótica ``k(1-ρ)/(1+ρ)``, que para k ∈ {5,10,20}
  está materialmente sesgada y puede devolver ``n_eff < 1`` (informe §3.3: con
  ρ=0,8 y k=10 la hipótesis iid inflaría el z por un factor 2,3).
* El BVC con barras diarias es una función determinista del retorno diario
  estandarizado; solo la versión **ponderada por volumen** aporta información
  (informe §4.2). La versión de barra única no se implementa como feature.
* La línea base de volumen debe ser consciente de Chae (2005): el volumen **cae**
  antes de anuncios programados, así que se ofrece además el z respecto a los
  mismos ``tau`` de los eventos previos del propio emisor
  (`turnover_zscore_vs_own_prior_quarters`, informe §3.6).
* `available_at` manda siempre: el short interest se usa por su **fecha de
  publicación** (~8 días hábiles tras la referencia), la cuota off-exchange llega
  con ~2 semanas de retardo (ventana efectiva ≈ [T-35, T-15]) y los Form 4 por su
  timestamp de aceptación en EDGAR (informe §8 y §13).

Referencias principales
-----------------------
- Ajinkya, B. B., Jain, P. C. (1989). *JAE* — transformación logarítmica del volumen.
- Garfinkel, J. A., Sokobin, J. (2006). *JAR* — turnover ajustado por mercado y SUV.
- Chae, J. (2005). *Journal of Finance* 60(1) — el volumen cae antes de anuncios
  programados.
- Easley, D., López de Prado, M., O'Hara, M. (2012, 2016) — Bulk Volume
  Classification.
- Lee, C. M. C., Ready, M. J. (1991). *JF* 46(2) — referencia intradía del signing.
- Amihud, Y. (2002). *Journal of Financial Markets* 5, 31–56 — iliquidez.
- Boehmer, E., Jones, C. M., Zhang, X. (2008). *JF* 63(2); Christophe, Ferri y
  Angel (2004). *JF* 59(4); Akbas et al. (2017). *Financial Management* — short
  interest.
- Zhu, H. (2014). *RFS* 27(3); Comerton-Forde, C., Putniņš, T. J. (2015). *JFE*
  118(1) — cuota dark/off-exchange.
- Cohen, L., Malloy, C., Pomorski, L. (2012). *JF* 67(3) — insiders rutinarios vs.
  oportunistas; Ke, Huddart y Petroni (2003). *JAE* 35(3) — horizonte de las
  ventas de insiders; SOX §403 — plazo de 2 días hábiles del Form 4.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Sequence
from typing import Final, Literal

import numpy as np
import pandas as pd
from scipy.special import ndtr

from earnings_alpha.errors import DataQualityError, InsufficientHistory
from earnings_alpha.pit import TradingCalendar, get_calendar

__all__ = [
    "DEFAULT_BASE_WINDOW",
    "DEFAULT_DETECTION_WINDOWS",
    "LOG_TURNOVER_CONSTANT",
    "amihud_illiquidity",
    "abnormal_volume",
    "ar1_effective_sample_size",
    "bvc_order_imbalance",
    "clv_order_imbalance",
    "insider_net_buy_form4",
    "off_exchange_share_delta",
    "settlement_cycle_lag",
    "settlement_to_trade_date",
    "short_interest_delta",
    "suv",
    "turnover_zscore",
    "turnover_zscore_vs_own_prior_quarters",
    "volume_runup",
]

DEFAULT_BASE_WINDOW: Final[tuple[int, int]] = (-70, -21)
"""Ventana base por defecto en tiempo-evento: [T-70, T-21], 50 sesiones. Termina en
T-21 para no solaparse con la ventana de detección (informe §3.3)."""

DEFAULT_DETECTION_WINDOWS: Final[tuple[int, ...]] = (5, 10, 20)
"""Ventanas de detección. La principal es k=5 —la huella documentada (tipping de
Irvine, Lipson y Puckett 2007; hackeo de newswires) aparece en [T-5, T-1]—; 10 y 20
son secundarias (informe §4.6 y §10.2)."""

LOG_TURNOVER_CONSTANT: Final[float] = 0.000255
"""Constante de la literatura de eventos de volumen para evitar ln(0) en días sin
negociación (informe §3.1)."""

_MIN_ROLLING_WINDOWS: Final[int] = 8
"""Mínimo de ventanas móviles completas para que la corrección empírica (método a
del informe §3.3) produzca una desviación típica utilizable."""

# Transición de ciclos de liquidación en EE. UU. (informe §8.1, trampa 2):
# T+3 hasta que el T+2 entró en vigor para operaciones del 2017-09-05 (primera
# liquidación T+2: 2017-09-07); T+1 en vigor para operaciones del 2024-05-28
# (primera liquidación T+1: 2024-05-29).
_FIRST_T2_SETTLEMENT: Final[dt.date] = dt.date(2017, 9, 7)
_FIRST_T1_SETTLEMENT: Final[dt.date] = dt.date(2024, 5, 29)


# ---------------------------------------------------------------------------
# Utilidades de validación y preparación
# ---------------------------------------------------------------------------


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    """Falla ruidosamente si faltan columnas: nada de KeyError crípticos aguas abajo."""
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        msg = f"`{name}` no tiene las columnas requeridas {missing}; tiene {list(frame.columns)}"
        raise DataQualityError(msg)


def _normalize_events(events: pd.DataFrame) -> pd.DataFrame:
    """Normaliza la tabla de eventos a ``event_id, ticker, event_date`` (tz-naive).

    Acepta ``tradable_date`` como sinónimo de ``event_date`` (es el nombre que usa
    `pit.tradable_dates`). Exige `event_id` único: la salida de todas las funciones
    del módulo se indexa por él y un duplicado corrompería cualquier join.
    """
    if not isinstance(events, pd.DataFrame):
        msg = f"`events` debe ser un DataFrame; recibido {type(events).__name__}"
        raise DataQualityError(msg)
    ev = events.copy()
    if isinstance(ev.index, pd.MultiIndex) or ev.index.name == "event_id":
        ev = ev.reset_index()
    if "event_date" not in ev.columns and "tradable_date" in ev.columns:
        ev = ev.rename(columns={"tradable_date": "event_date"})
    _require_columns(ev, ["event_id", "ticker", "event_date"], "events")
    if len(ev) == 0:
        msg = "la tabla de eventos está vacía: no hay nada que calcular"
        raise DataQualityError(msg)
    if ev["event_id"].duplicated().any():
        dup = ev.loc[ev["event_id"].duplicated(), "event_id"].head(3).tolist()
        msg = f"`events` tiene event_id duplicados (muestra: {dup})"
        raise DataQualityError(msg)
    ev["event_date"] = pd.DatetimeIndex(pd.to_datetime(ev["event_date"])).normalize()
    return ev.reset_index(drop=True)


def _check_prices(prices: pd.DataFrame, needed: Sequence[str]) -> pd.DataFrame:
    """Valida el panel canónico ``(date, ticker)`` y las columnas necesarias."""
    if not isinstance(prices, pd.DataFrame) or not isinstance(prices.index, pd.MultiIndex):
        msg = "`prices` debe ser un DataFrame con MultiIndex (date, ticker)"
        raise DataQualityError(msg)
    names = [str(n) for n in prices.index.names]
    if names != ["date", "ticker"]:
        msg = f"`prices` debe tener niveles ['date', 'ticker']; tiene {names}"
        raise DataQualityError(msg)
    _require_columns(prices, needed, "prices")
    if len(prices) == 0:
        msg = "el panel de precios está vacío"
        raise InsufficientHistory(msg)
    return prices


def _wide(prices: pd.DataFrame, column: str) -> pd.DataFrame:
    """Pivota una columna del panel a formato ancho fechas × tickers."""
    out = prices[column].unstack("ticker")
    out.index = pd.DatetimeIndex(out.index).normalize()
    return out.sort_index()


def _returns_wide(prices: pd.DataFrame) -> pd.DataFrame:
    """Retornos logarítmicos diarios calculados **internamente** a partir del precio.

    Se recalculan aunque el panel traiga una columna de retornos: así el test de
    no-look-ahead que perturba `close`/`adj_close` no puede quedar enmascarado por
    una columna precalculada inconsistente. Se prefiere `adj_close` (ajustado por
    splits y dividendos) y se degrada a `close` si no existe.
    """
    price_col = "adj_close" if "adj_close" in prices.columns else "close"
    px = _wide(prices, price_col)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(px.where(px > 0)).diff()


def _positions(event_dates: pd.Series, dates: pd.DatetimeIndex) -> np.ndarray:
    """Posición de cada `event_date` en el índice de sesiones; -1 si no está.

    Un evento fuera del panel (prehistoria o posterior al rango) no es un error:
    sus features salen NaN, pero el evento sigue contando para las exclusiones y
    para la línea base por emisor.
    """
    return dates.get_indexer(pd.DatetimeIndex(event_dates))


def _exclusion_mask(
    dates: pd.DatetimeIndex,
    tickers: Sequence[str],
    events: pd.DataFrame,
    halfwidth: int,
) -> dict[str, np.ndarray]:
    """Máscara por ticker de sesiones a excluir de la ventana base.

    El informe §3.3 exige excluir de la base las ventanas ``[-hw, +hw]`` de
    cualquier evento de resultados del mismo emisor: el pico de volumen del
    anuncio anterior contaminaría la media y la sigma "normales". Los días de
    inclusión/exclusión del índice y los splits no están disponibles aquí y quedan
    sin excluir (limitación declarada).
    """
    masks = {t: np.zeros(len(dates), dtype=bool) for t in tickers}
    for row in events.itertuples(index=False):
        mask = masks.get(row.ticker)
        if mask is None:
            continue
        # `searchsorted` en vez de `get_indexer`: un evento anterior al panel aún
        # puede proyectar la cola de su ventana sobre las primeras sesiones.
        p = int(dates.searchsorted(pd.Timestamp(row.event_date)))
        lo = max(p - halfwidth, 0)
        hi = min(p + halfwidth + 1, len(dates))
        if lo < hi:
            mask[lo:hi] = True
    return masks


def _base_slice(pos: int, base_window: tuple[int, int]) -> slice | None:
    """Índices absolutos de la ventana base ``[pos+b0, pos+b1]`` (inclusive)."""
    b0, b1 = base_window
    if b0 > b1 or b1 >= 0:
        msg = f"ventana base inválida {base_window}: debe ser (b0 <= b1 < 0)"
        raise DataQualityError(msg)
    lo, hi = pos + b0, pos + b1 + 1
    if lo < 0:
        return None
    return slice(lo, hi)


def _detection_slice(pos: int, k: int) -> slice | None:
    """Índices absolutos de la ventana de detección ``[T-k, T-1]``."""
    if k < 1:
        msg = f"k debe ser >= 1; recibido {k}"
        raise DataQualityError(msg)
    lo = pos - k
    if lo < 0:
        return None
    return slice(lo, pos)


def _raise_if_all_nan(obj: pd.Series | pd.DataFrame, what: str) -> None:
    """Regla 5 del repo: nada de resultados vacíos silenciosos."""
    values = obj.to_numpy(dtype=float) if isinstance(obj, pd.DataFrame) else obj.to_numpy()
    if not np.isfinite(np.asarray(values, dtype=float)).any():
        msg = (
            f"{what}: ninguna de las {len(obj)} filas es calculable. Lo habitual es "
            "que el panel de precios no cubra la ventana base de ningún evento "
            "(hacen falta al menos ~70 sesiones antes del primer evento)."
        )
        raise InsufficientHistory(msg)


# ---------------------------------------------------------------------------
# Tamaño muestral efectivo de un AR(1) — corrección central del informe
# ---------------------------------------------------------------------------


def ar1_effective_sample_size(k: int, rho: float) -> float:
    """Tamaño muestral efectivo **exacto** de la media de k observaciones de un AR(1).

    Si ``x`` es AR(1) con autocorrelación ``rho``, la varianza de la media de k
    observaciones consecutivas es exactamente
    ``Var(media_k) = (sigma²/k²)·[k + 2·Σ_{j=1..k-1}(k-j)·ρ^j]`` y por tanto::

        n_eff(k, rho) = k² / ( k + 2·Σ_{j=1..k-1} (k-j)·ρ^j )

    **Nunca** usar la aproximación asintótica ``n_eff ≈ k·(1-ρ)/(1+ρ)``: es una
    expresión de k→∞, para k ∈ {5, 10, 20} está materialmente sesgada y puede
    devolver ``n_eff < 1``, que carece de sentido (informe
    `docs/research/informed_trading.md` §3.3, verificado por Monte Carlo: con
    ρ=0,8 y k=10 el valor exacto es 1,842 frente a 1,111 del asintótico y 10 del
    supuesto iid — un z-score iid estaría inflado por un factor ≈ 2,3).

    Parámetros
    ----------
    k:
        Número de observaciones consecutivas promediadas (>= 1).
    rho:
        Autocorrelación de primer orden, en (-1, 1).
    """
    if k < 1:
        msg = f"k debe ser >= 1; recibido {k}"
        raise DataQualityError(msg)
    if not -1.0 < rho < 1.0:
        msg = f"rho debe estar en (-1, 1); recibido {rho}"
        raise DataQualityError(msg)
    if k == 1:
        return 1.0
    j = np.arange(1, k, dtype=float)
    denom = k + 2.0 * float(np.sum((k - j) * rho**j))
    return k * k / denom


# ---------------------------------------------------------------------------
# Motor común de turnover
# ---------------------------------------------------------------------------


def _log_turnover_adjusted(
    prices: pd.DataFrame, *, constant: float = LOG_TURNOVER_CONSTANT
) -> pd.DataFrame:
    """Log-turnover ajustado por mercado: ``x - mediana cross-section del día``.

    ``x[i,t] = ln(volume/shares_outstanding + c)`` (Ajinkya y Jain 1989: el log es
    aproximadamente normal; la constante evita ln(0)). Si el panel no trae
    `shares_outstanding` point-in-time se degrada a ``ln(volume + 1)``: el nivel
    por ticker es distinto, pero el z-score le resta su propia media histórica y
    es invariante a ese nivel mientras las acciones en circulación no cambien
    bruscamente dentro de la ventana (sustituto aceptado por el informe §3.1).

    El ajuste de mercado resta la **mediana** de la sección cruzada de los tickers
    del panel, no la media: en un día de anuncios masivos la media la dominan los
    propios anunciantes (Garfinkel y Sokobin 2006; informe §3.2).
    """
    volume = _wide(prices, "volume")
    if "shares_outstanding" in prices.columns:
        shares = _wide(prices, "shares_outstanding")
        turn = volume / shares.where(shares > 0)
        x = np.log(turn + constant)
    else:
        x = np.log(volume.clip(lower=0.0) + 1.0)
    return x.sub(x.median(axis=1), axis=0)


def _turnover_engine(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    *,
    windows: Sequence[int],
    base_window: tuple[int, int],
    method: Literal["empirical", "ar1"],
    min_base_obs: int,
    exclusion_halfwidth: int,
    constant: float,
) -> pd.DataFrame:
    """Calcula ``zbar[i,k]`` para cada evento y cada k (informe §3.3).

    Devuelve un DataFrame indexado por event_id con una columna por k. NaN cuando
    el evento no tiene sitio en el panel o la base no llega a `min_base_obs`.
    """
    ev = _normalize_events(events)
    px = _check_prices(prices, ["volume"])
    xadj = _log_turnover_adjusted(px, constant=constant)
    dates = pd.DatetimeIndex(xadj.index)
    pos_arr = _positions(ev["event_date"], dates)
    excl = _exclusion_mask(dates, list(xadj.columns), ev, exclusion_halfwidth)

    out = {int(k): np.full(len(ev), np.nan) for k in windows}
    columns = {t: xadj[t].to_numpy() for t in xadj.columns}

    for i, row in enumerate(ev.itertuples(index=False)):
        pos = int(pos_arr[i])
        series = columns.get(row.ticker)
        if pos < 0 or series is None:
            continue
        sl = _base_slice(pos, base_window)
        if sl is None:
            continue
        base = series[sl].copy()
        base[excl[row.ticker][sl]] = np.nan
        valid = np.isfinite(base)
        n_base = int(valid.sum())
        if n_base < min_base_obs:
            continue
        mu = float(np.mean(base[valid]))
        sigma = float(np.std(base[valid], ddof=1))
        if not np.isfinite(sigma) or sigma <= 0.0:
            continue

        for k in windows:
            det = _detection_slice(pos, int(k))
            if det is None:
                continue
            det_vals = series[det]
            if not np.isfinite(det_vals).all():
                continue
            se = _standard_error(base, mu, sigma, int(k), n_base, method)
            if se is None:
                continue
            out[int(k)][i] = (float(np.mean(det_vals)) - mu) / se

    frame = pd.DataFrame(
        {f"turnover_zscore_{k}d": out[int(k)] for k in windows},
        index=pd.Index(ev["event_id"], name="event_id"),
    )
    return frame


def _standard_error(
    base: np.ndarray,
    mu: float,
    sigma: float,
    k: int,
    n_base: int,
    method: Literal["empirical", "ar1"],
) -> float | None:
    """Error estándar de la media de k días bajo autocorrelación (informe §3.3).

    Método (a) *empírico* (recomendado): desviación típica muestral de la media
    móvil de k días sobre la ventana base. Absorbe la autocorrelación sin
    modelarla; requiere ``n_base >= 2k + 10`` y al menos `_MIN_ROLLING_WINDOWS`
    ventanas completas (los huecos por exclusión de eventos previos no se
    concatenan: una ventana con hueco no es una media de k días consecutivos).

    Método (b) *AR(1) exacto* (fallback): ``se = sigma / sqrt(n_eff(k, rho))`` con
    el `ar1_effective_sample_size` **exacto**, jamás el asintótico.
    """
    if method == "empirical" and n_base >= 2 * k + 10:
        rolls = pd.Series(base).rolling(k, min_periods=k).mean().dropna()
        if len(rolls) >= _MIN_ROLLING_WINDOWS:
            se = float(rolls.std(ddof=1))
            if np.isfinite(se) and se > 0.0:
                return se
    # Fallback (o método pedido): rho muestral de primer orden sobre pares
    # consecutivos válidos de la base.
    lead, lag = base[1:], base[:-1]
    pair = np.isfinite(lead) & np.isfinite(lag)
    if int(pair.sum()) < 10:
        return None
    a = lead[pair] - float(np.mean(lead[pair]))
    b = lag[pair] - float(np.mean(lag[pair]))
    denom = math.sqrt(float(np.sum(a * a)) * float(np.sum(b * b)))
    if denom <= 0.0:
        return None
    rho = float(np.clip(float(np.sum(a * b)) / denom, -0.95, 0.95))
    n_eff = ar1_effective_sample_size(k, rho)
    return sigma / math.sqrt(n_eff)


# ---------------------------------------------------------------------------
# Features de volumen
# ---------------------------------------------------------------------------


def turnover_zscore(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    *,
    k: int = 5,
    base_window: tuple[int, int] = DEFAULT_BASE_WINDOW,
    method: Literal["empirical", "ar1"] = "empirical",
    min_base_obs: int = 30,
    exclusion_halfwidth: int = 2,
    constant: float = LOG_TURNOVER_CONSTANT,
) -> pd.Series:
    """Z-score del log-turnover ajustado por mercado en la ventana ``[T-k, T-1]``.

    ``zbar[i,k] = (media_{tau=-k..-1} xadj[i,tau] - mu[i]) / se[i,k]`` con `mu` y
    `sigma` estimados en la ventana base y el **denominador corregido por
    autocorrelación** (informe §3.3): método (a) empírico por defecto, con
    fallback a la forma **exacta** del AR(1) vía `ar1_effective_sample_size`.
    Bajo iid el denominador sería ``sigma/sqrt(k)``; con la persistencia típica
    del volumen (ρ ≈ 0,4–0,8) eso infla el z hasta un factor 2,3 y fabrica
    significancia (verificación Monte Carlo del informe).

    Interpretación: **magnitud, no dirección** (informe §1, fila 1). Y cuidado con
    Chae (2005): el patrón incondicional antes de anuncios programados es que el
    volumen *cae*, así que un z positivo es más anómalo de lo que parece; la
    versión condicionada al emisor está en
    `turnover_zscore_vs_own_prior_quarters`.

    Referencias: Ajinkya y Jain (1989); Garfinkel y Sokobin (2006); Chae (2005).
    """
    frame = _turnover_engine(
        prices,
        events,
        windows=[k],
        base_window=base_window,
        method=method,
        min_base_obs=min_base_obs,
        exclusion_halfwidth=exclusion_halfwidth,
        constant=constant,
    )
    series = frame.iloc[:, 0].rename("turnover_zscore")
    _raise_if_all_nan(series, "turnover_zscore")
    return series


def abnormal_volume(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    *,
    windows: Sequence[int] = DEFAULT_DETECTION_WINDOWS,
    base_window: tuple[int, int] = DEFAULT_BASE_WINDOW,
    method: Literal["empirical", "ar1"] = "empirical",
    min_base_obs: int = 30,
    exclusion_halfwidth: int = 2,
    constant: float = LOG_TURNOVER_CONSTANT,
) -> pd.DataFrame:
    """Volumen anormal a 5/10/20 días como z-scores ``zbar[i,k]`` (informe §3.4).

    Devuelve columnas ``abnormal_volume_{k}d``. Se usa el z-score y no el ratio
    como feature de modelo porque es comparable entre acciones; el ratio
    interpretable está en `volume_runup`. Misma corrección de autocorrelación que
    `turnover_zscore`.
    """
    frame = _turnover_engine(
        prices,
        events,
        windows=windows,
        base_window=base_window,
        method=method,
        min_base_obs=min_base_obs,
        exclusion_halfwidth=exclusion_halfwidth,
        constant=constant,
    )
    frame.columns = [f"abnormal_volume_{k}d" for k in windows]
    _raise_if_all_nan(frame, "abnormal_volume")
    return frame


def volume_runup(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    *,
    k: int = 5,
    base_window: tuple[int, int] = DEFAULT_BASE_WINDOW,
    min_base_obs: int = 30,
    exclusion_halfwidth: int = 2,
) -> pd.Series:
    """Ratio de volumen anormal ``AVR[i,k]`` (informe §3.4), interpretable en niveles.

    ``AVR = media(volume, tau=-k..-1) / mediana(volume, base)``. AVR = 1 es
    normalidad; AVR = 1,8 significa un 80 % de volumen extra. **Mediana** en el
    denominador porque la media del volumen la dominan tres o cuatro días
    atípicos. La versión logarítmica que entra en modelos lineales es
    ``ln(AVR)``; como feature estandarizada úsese `abnormal_volume`.

    Caveat de endogeneidad (Kacperczyk y Pagnotta 2019, informe §2): los
    informados eligen días que *ya* tienen volumen alto para esconderse, así que
    parte del run-up es causa y no consecuencia de su presencia.
    """
    ev = _normalize_events(events)
    px = _check_prices(prices, ["volume"])
    volume = _wide(px, "volume")
    dates = pd.DatetimeIndex(volume.index)
    pos_arr = _positions(ev["event_date"], dates)
    excl = _exclusion_mask(dates, list(volume.columns), ev, exclusion_halfwidth)

    values = np.full(len(ev), np.nan)
    columns = {t: volume[t].to_numpy(dtype=float) for t in volume.columns}
    for i, row in enumerate(ev.itertuples(index=False)):
        pos = int(pos_arr[i])
        series = columns.get(row.ticker)
        if pos < 0 or series is None:
            continue
        sl = _base_slice(pos, base_window)
        det = _detection_slice(pos, k)
        if sl is None or det is None:
            continue
        base = series[sl].copy()
        base[excl[row.ticker][sl]] = np.nan
        valid = np.isfinite(base)
        if int(valid.sum()) < min_base_obs:
            continue
        med = float(np.median(base[valid]))
        det_vals = series[det]
        if med <= 0.0 or not np.isfinite(det_vals).all():
            continue
        values[i] = float(np.mean(det_vals)) / med

    series_out = pd.Series(
        values, index=pd.Index(ev["event_id"], name="event_id"), name="volume_runup"
    )
    _raise_if_all_nan(series_out, "volume_runup")
    return series_out


def turnover_zscore_vs_own_prior_quarters(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    *,
    k: int = 5,
    n_prior: int = 4,
    min_prior: int = 2,
    base_window: tuple[int, int] = DEFAULT_BASE_WINDOW,
    method: Literal["empirical", "ar1"] = "empirical",
    min_base_obs: int = 30,
    exclusion_halfwidth: int = 2,
    constant: float = LOG_TURNOVER_CONSTANT,
) -> pd.Series:
    """Línea base consciente de Chae (2005): z frente a los pre-anuncios del emisor.

    Chae (2005, *JF* 60(1), 413–442) documenta que el volumen acumulado
    **disminuye** antes de anuncios programados —más cuanto mayor es la asimetría
    informativa— porque los *discretionary liquidity traders* posponen sus
    operaciones. Comparar [T-20, T-1] contra una base genérica [T-70, T-21] mide
    sobre todo cuánto se secó el volumen (efecto de liquidez); la línea base
    correcta es la propia ventana pre-anuncio del emisor en trimestres anteriores
    (informe §3.6, "la más importante de las features que faltan")::

        feature[i] = zbar[i,k] - media( zbar[i,k] de hasta n_prior eventos previos )

    NaN si el emisor no tiene al menos `min_prior` eventos previos con z
    calculable. Pásese la tabla de eventos **completa** (con prehistoria si
    existe): cada evento extra con histórico de precios mejora la base.
    """
    frame = _turnover_engine(
        prices,
        events,
        windows=[k],
        base_window=base_window,
        method=method,
        min_base_obs=min_base_obs,
        exclusion_halfwidth=exclusion_halfwidth,
        constant=constant,
    )
    ev = _normalize_events(events)
    z = frame.iloc[:, 0].to_numpy()
    order = ev.sort_values(["ticker", "event_date"]).index

    values = np.full(len(ev), np.nan)
    prior_by_ticker: dict[str, list[float]] = {}
    for idx in order:
        ticker = ev.at[idx, "ticker"]
        history = prior_by_ticker.setdefault(ticker, [])
        z_i = z[idx] if idx < len(z) else np.nan
        usable = [v for v in history[-n_prior:] if np.isfinite(v)]
        if np.isfinite(z_i) and len(usable) >= min_prior:
            values[idx] = z_i - float(np.mean(usable))
        if np.isfinite(z_i):
            history.append(float(z_i))

    out = pd.Series(
        values,
        index=pd.Index(ev["event_id"], name="event_id"),
        name="turnover_zscore_vs_own_prior_quarters",
    )
    _raise_if_all_nan(out, "turnover_zscore_vs_own_prior_quarters")
    return out


def suv(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    *,
    windows: Sequence[int] = DEFAULT_DETECTION_WINDOWS,
    base_window: tuple[int, int] = DEFAULT_BASE_WINDOW,
    min_base_obs: int = 30,
    exclusion_halfwidth: int = 2,
    constant: float = LOG_TURNOVER_CONSTANT,
    standardization: Literal["empirical", "sqrt_k"] = "empirical",
) -> pd.DataFrame:
    """SUV — *Standardized Unexpected Volume* de Garfinkel y Sokobin (informe §3.5).

    Aísla el volumen **no explicado por el movimiento de precio**, candidato
    natural a proxy de actividad informada / divergencia de opinión (Garfinkel y
    Sokobin 2006, *JAR*; Garfinkel 2009). En la ventana base se estima::

        x[i,t] = a + b1·Rpos[i,t] + b2·Rneg[i,t] + e[i,t]
        Rpos = max(r, 0);  Rneg = max(-r, 0)

    (la asimetría en dos coeficientes captura que el volumen reacciona distinto a
    subidas y bajadas de la misma magnitud). Para cada día pre-evento se calcula
    el residuo **fuera de muestra**, se estandariza con la desviación típica
    residual dentro de muestra y se agrega::

        SUV[i,k] = Σ_{tau=-k..-1} SUV[i,tau] / sqrt(k)

    Con ``standardization="empirical"`` (por defecto) el divisor ``sqrt(k)`` se
    sustituye por la desviación típica empírica de la suma móvil de k días de los
    residuos estandarizados en la base, que absorbe la autocorrelación (mismo
    caveat que §3.3 del informe); si la base no da para ello se cae a
    ``sqrt(k)``. Signo sobre el drift posterior: positivo; como feature pre-evento
    el signo debe estimarse, no imponerse (informe §3.5).
    """
    ev = _normalize_events(events)
    px = _check_prices(prices, ["volume"])
    xadj = _log_turnover_adjusted(px, constant=constant)
    returns = _returns_wide(px)
    dates = pd.DatetimeIndex(xadj.index)
    pos_arr = _positions(ev["event_date"], dates)
    excl = _exclusion_mask(dates, list(xadj.columns), ev, exclusion_halfwidth)

    x_cols = {t: xadj[t].to_numpy() for t in xadj.columns}
    r_cols = {t: returns[t].to_numpy() for t in returns.columns}
    out = {int(k): np.full(len(ev), np.nan) for k in windows}

    for i, row in enumerate(ev.itertuples(index=False)):
        pos = int(pos_arr[i])
        x = x_cols.get(row.ticker)
        r = r_cols.get(row.ticker)
        if pos < 0 or x is None or r is None:
            continue
        sl = _base_slice(pos, base_window)
        if sl is None:
            continue
        xb, rb = x[sl].copy(), r[sl]
        xb[excl[row.ticker][sl]] = np.nan
        valid = np.isfinite(xb) & np.isfinite(rb)
        n = int(valid.sum())
        if n < max(min_base_obs, 10):
            continue
        rpos = np.maximum(rb[valid], 0.0)
        rneg = np.maximum(-rb[valid], 0.0)
        design = np.column_stack([np.ones(n), rpos, rneg])
        coef, *_ = np.linalg.lstsq(design, xb[valid], rcond=None)
        resid = xb[valid] - design @ coef
        dof = n - 3
        s_e = math.sqrt(float(np.sum(resid * resid)) / dof) if dof > 0 else 0.0
        if not np.isfinite(s_e) or s_e <= 0.0:
            continue

        # Residuos estandarizados dentro de muestra sobre la rejilla de la base
        # (NaN en huecos): sirven para la estandarización empírica de la suma.
        resid_grid = np.full(len(xb), np.nan)
        fitted_grid = coef[0] + coef[1] * np.maximum(rb, 0.0) + coef[2] * np.maximum(-rb, 0.0)
        resid_grid[valid] = (xb[valid] - fitted_grid[valid]) / s_e

        for k in windows:
            det = _detection_slice(pos, int(k))
            if det is None:
                continue
            xd, rd = x[det], r[det]
            if not (np.isfinite(xd).all() and np.isfinite(rd).all()):
                continue
            fitted = coef[0] + coef[1] * np.maximum(rd, 0.0) + coef[2] * np.maximum(-rd, 0.0)
            suv_days = (xd - fitted) / s_e
            denom = math.sqrt(float(k))
            if standardization == "empirical":
                sums = pd.Series(resid_grid).rolling(int(k), min_periods=int(k)).sum().dropna()
                if len(sums) >= _MIN_ROLLING_WINDOWS:
                    emp = float(sums.std(ddof=1))
                    if np.isfinite(emp) and emp > 0.0:
                        denom = emp
            out[int(k)][i] = float(np.sum(suv_days)) / denom

    frame = pd.DataFrame(
        {f"suv_{k}d": out[int(k)] for k in windows},
        index=pd.Index(ev["event_id"], name="event_id"),
    )
    _raise_if_all_nan(frame, "suv")
    return frame


# ---------------------------------------------------------------------------
# Proxies de desequilibrio de órdenes con OHLCV diario
# ---------------------------------------------------------------------------


def bvc_order_imbalance(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    *,
    k: int = 5,
    base_window: tuple[int, int] = DEFAULT_BASE_WINDOW,
    min_base_obs: int = 30,
) -> pd.Series:
    """Desequilibrio BVC **ponderado por volumen** sobre ``[T-k, T-1]`` (informe §4.2).

    Bulk Volume Classification (Easley, López de Prado y O'Hara 2012, 2016)::

        Vbuy[tau] = V[tau] · Φ( dP[tau] / sigma_dP )
        OIB_bvc[i,k] = Σ V·(2·Φ(dP/sigma_dP) - 1) / Σ V

    **Trampa matemática documentada en el informe (§4.2) y respetada aquí:** con
    barras diarias, el desequilibrio BVC de una sola barra,
    ``2·Φ(dP/sigma) - 1``, es una función determinista y monótona del retorno
    diario estandarizado — un retorno "aplastado" por una sigmoide que no aporta
    nada sobre el CAR. El único valor añadido aparece al **ponderar por volumen**
    entre días, que es la única versión que este módulo expone. Aun así debe
    ortogonalizarse frente a ``pre_event_car_k`` antes de usarse y probarse que
    aporta IC incremental (informe §7.2 y §12.2).

    ``sigma_dP`` se estima **solo con la ventana base**, estrictamente anterior a
    la de detección: estimarla sobre toda la muestra es el look-ahead que
    Andersen y Bondarenko (2014) señalan en las implementaciones de VPIN
    (informe §6.2). Se usa la CDF normal (`scipy.special.ndtr`); la variante t de
    Student del paper original queda descartada por el informe (§4.2) como de
    segundo orden.

    Signo esperado: positivo (Campbell, Ramadorai y Schwartz 2009: el flujo
    institucional es más positivo antes de sorpresas positivas). Precisión
    limitada con datos diarios (informe §4.1: los algoritmos de signing pierden
    fiabilidad justo donde opera el informado; el error atenúa hacia cero).
    """
    ev = _normalize_events(events)
    px = _check_prices(prices, ["volume"])
    price_col = "adj_close" if "adj_close" in px.columns else "close"
    close = _wide(px, price_col)
    volume = _wide(px, "volume")
    dp = close.diff()
    dates = pd.DatetimeIndex(close.index)
    pos_arr = _positions(ev["event_date"], dates)

    dp_cols = {t: dp[t].to_numpy() for t in dp.columns}
    v_cols = {t: volume[t].to_numpy(dtype=float) for t in volume.columns}
    values = np.full(len(ev), np.nan)

    for i, row in enumerate(ev.itertuples(index=False)):
        pos = int(pos_arr[i])
        d = dp_cols.get(row.ticker)
        v = v_cols.get(row.ticker)
        if pos < 0 or d is None or v is None:
            continue
        sl = _base_slice(pos, base_window)
        det = _detection_slice(pos, k)
        if sl is None or det is None:
            continue
        base = d[sl]
        base = base[np.isfinite(base)]
        if len(base) < min_base_obs:
            continue
        sigma_dp = float(np.std(base, ddof=1))
        if not np.isfinite(sigma_dp) or sigma_dp <= 0.0:
            continue
        dd, vd = d[det], v[det]
        if not (np.isfinite(dd).all() and np.isfinite(vd).all()) or float(np.sum(vd)) <= 0.0:
            continue
        oib = 2.0 * ndtr(dd / sigma_dp) - 1.0
        values[i] = float(np.sum(vd * oib) / np.sum(vd))

    out = pd.Series(
        values, index=pd.Index(ev["event_id"], name="event_id"), name="order_imbalance_bvc"
    )
    _raise_if_all_nan(out, "bvc_order_imbalance")
    return out


def clv_order_imbalance(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    *,
    k: int = 5,
) -> pd.Series:
    """Proxy de desequilibrio por *Close Location Value* sobre ``[T-k, T-1]``.

    Con solo OHLCV diario, la posición del cierre dentro del rango del día es un
    proxy que **no** es función del retorno cierre-a-cierre y por tanto aporta
    información linealmente independiente (informe §4.3)::

        CLV = ((C - L) - (H - C)) / (H - L)   en [-1, +1], 0 si H == L
        OIB_clv[i,k] = Σ CLV·V / Σ V

    Es la construcción subyacente a la línea de acumulación/distribución de
    Chaikin. **Aviso de honestidad (informe §4.3): no tiene validación académica
    como proxy de order imbalance.** Su validación pendiente —correlación de
    rangos >= 0,4 contra el OIB de Lee y Ready (1991) sobre un subconjunto con
    tick data— está anotada en `docs/OPEN_QUESTIONS.md`; hasta entonces debe
    tratarse como candidata, no como señal establecida. La regla del tick sobre
    cierres diarios (informe §4.4) NO se implementa: es casi colineal con el
    signo del CAR pre-evento.
    """
    ev = _normalize_events(events)
    px = _check_prices(prices, ["high", "low", "close", "volume"])
    high = _wide(px, "high")
    low = _wide(px, "low")
    close = _wide(px, "close")
    volume = _wide(px, "volume")
    rng_hl = high - low
    clv = ((close - low) - (high - close)).div(rng_hl.where(rng_hl > 0))
    clv = clv.where(rng_hl > 0, 0.0)
    dates = pd.DatetimeIndex(close.index)
    pos_arr = _positions(ev["event_date"], dates)

    clv_cols = {t: clv[t].to_numpy() for t in clv.columns}
    v_cols = {t: volume[t].to_numpy(dtype=float) for t in volume.columns}
    values = np.full(len(ev), np.nan)
    for i, row in enumerate(ev.itertuples(index=False)):
        pos = int(pos_arr[i])
        c = clv_cols.get(row.ticker)
        v = v_cols.get(row.ticker)
        if pos < 0 or c is None or v is None:
            continue
        det = _detection_slice(pos, k)
        if det is None:
            continue
        cd, vd = c[det], v[det]
        if not (np.isfinite(cd).all() and np.isfinite(vd).all()) or float(np.sum(vd)) <= 0.0:
            continue
        values[i] = float(np.sum(cd * vd) / np.sum(vd))

    out = pd.Series(
        values, index=pd.Index(ev["event_id"], name="event_id"), name="order_imbalance_clv"
    )
    _raise_if_all_nan(out, "clv_order_imbalance")
    return out


def amihud_illiquidity(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    *,
    window: tuple[int, int] = DEFAULT_BASE_WINDOW,
    min_obs: int = 30,
) -> pd.Series:
    """Iliquidez de Amihud (2002) sobre la ventana base: **variable de control**.

    ``ILLIQ[i] = media_t( |r[i,t]| / dollar_volume[i,t] ) · 1e6`` sobre
    ``[T-70, T-21]`` por defecto (Amihud 2002, *Journal of Financial Markets* 5,
    31–56). No es una señal de información: entra en el conjunto porque Duarte y
    Young (2009) muestran que las medidas "de información" tipo PIN cotizan por
    su componente de iliquidez (informe §5.4), y sin este control cualquier
    detector redescubriría la prima de iliquidez y la llamaría filtración.
    """
    ev = _normalize_events(events)
    px = _check_prices(prices, ["close", "volume"])
    returns = _returns_wide(px)
    if "dollar_volume" in px.columns:
        dollar = _wide(px, "dollar_volume")
    else:
        dollar = _wide(px, "close") * _wide(px, "volume")
    dates = pd.DatetimeIndex(returns.index)
    pos_arr = _positions(ev["event_date"], dates)

    r_cols = {t: returns[t].to_numpy() for t in returns.columns}
    d_cols = {t: dollar[t].to_numpy(dtype=float) for t in dollar.columns}
    values = np.full(len(ev), np.nan)
    for i, row in enumerate(ev.itertuples(index=False)):
        pos = int(pos_arr[i])
        r = r_cols.get(row.ticker)
        d = d_cols.get(row.ticker)
        if pos < 0 or r is None or d is None:
            continue
        sl = _base_slice(pos, window)
        if sl is None:
            continue
        rr, dd = r[sl], d[sl]
        valid = np.isfinite(rr) & np.isfinite(dd) & (dd > 0)
        if int(valid.sum()) < min_obs:
            continue
        values[i] = float(np.mean(np.abs(rr[valid]) / dd[valid])) * 1e6

    out = pd.Series(
        values, index=pd.Index(ev["event_id"], name="event_id"), name="amihud_illiquidity"
    )
    _raise_if_all_nan(out, "amihud_illiquidity")
    return out


# ---------------------------------------------------------------------------
# Short interest FINRA
# ---------------------------------------------------------------------------


def settlement_cycle_lag(settlement_date: dt.date | pd.Timestamp) -> int:
    """Ciclo de liquidación vigente en una fecha: T+3, T+2 o T+1 (informe §8.1).

    La correspondencia *settlement date* → *trade date* ha cambiado dos veces:
    T+3 hasta que el T+2 entró en vigor (operaciones desde el 2017-09-05; primera
    liquidación T+2 el 2017-09-07) y T+1 desde el 2024-05-28 (primera liquidación
    T+1 el 2024-05-29). Un backtest largo que ignore el cambio desalinea hasta
    dos sesiones la referencia real de cada snapshot de short interest. En las
    fechas exactas de transición la correspondencia es ambigua (conviven dos
    ciclos en liquidación); se aplica el ciclo nuevo desde su primera liquidación.
    """
    d = pd.Timestamp(settlement_date).date()
    if d < _FIRST_T2_SETTLEMENT:
        return 3
    if d < _FIRST_T1_SETTLEMENT:
        return 2
    return 1


def settlement_to_trade_date(
    settlement_date: dt.date | pd.Timestamp, cal: TradingCalendar | None = None
) -> dt.date:
    """Última sesión de negociación cuyas operaciones liquidan en `settlement_date`.

    Aplica el ciclo de `settlement_cycle_lag` retrocediendo en **sesiones** del
    calendario. Las posiciones del snapshot de FINRA reflejan lo negociado hasta
    esta fecha, no hasta la fecha de liquidación.
    """
    calendar = cal or get_calendar()
    ref = calendar.session_on_or_before(pd.Timestamp(settlement_date).date())
    return calendar.shift(ref, -settlement_cycle_lag(ref))


def short_interest_delta(
    short_interest: pd.DataFrame,
    events: pd.DataFrame,
    *,
    n_history: int = 8,
    min_history: int = 6,
) -> pd.Series:
    """Delta estandarizado del short interest, publicado y con el signo del repo.

    Fórmulas del informe §8.1::

        SIR[i,t] = shares_short / shares_outstanding
        dSI[i]   = SIR[último snapshot PUBLICADO antes de T] - SIR[snapshot anterior]
        dSI_z[i] = ( dSI - media(dSI, 8 snapshots previos) ) / std(dSI, 8 previos)
        short_interest_delta = -dSI_z            (mayor = más alcista)

    El signo se invierte porque el efecto económico es negativo: más interés corto
    predice peores retornos y sorpresas más negativas (Boehmer, Jones y Zhang
    2008: decil alto vs. bajo ≈ −1,16 %/mes; Christophe, Ferri y Angel 2004;
    Akbas, Boehmer, Ertürk y Sorescu 2017).

    **Trampa PIT decisiva (informe §8.1, trampa 1), respetada aquí:** el snapshot
    se fecha por su `available_at` = **fecha de publicación de FINRA** (~7.º–8.º
    día hábil tras la fecha de referencia), nunca por la `settlement_date`.
    Indexar por la fecha de referencia mete ~8 días hábiles de look-ahead,
    suficiente para atravesar íntegra la ventana [T-5, T-1]. Solo se usan
    snapshots con ``available_at`` estrictamente anterior a la sesión T (la hora
    de publicación intradía no se conoce; excluir el propio día T es la lectura
    conservadora). La referencia real de negociación de cada snapshot se obtiene
    con `settlement_to_trade_date` (ciclo T+3/T+2/T+1 según época, trampa 2).

    Confusores no filtrables aquí (trampa 5): arbitraje de convertibles, de
    fusiones y cobertura de ETF generan short interest sin contenido informativo.
    Y el *short volume* diario de FINRA NO es sustituto: es flujo dominado por la
    intermediación (FINRA Information Notice 10/05/2019, trampa 4).
    """
    ev = _normalize_events(events)
    if not isinstance(short_interest, pd.DataFrame) or len(short_interest) == 0:
        msg = "`short_interest` está vacío o no es un DataFrame"
        raise DataQualityError(msg)
    si = short_interest.copy()
    _require_columns(si, ["ticker", "settlement_date", "available_at"], "short_interest")
    if "short_percent_shares" in si.columns:
        si["__sir__"] = si["short_percent_shares"].astype(float)
    else:
        _require_columns(si, ["shares_short", "shares_outstanding"], "short_interest")
        shares_out = si["shares_outstanding"].astype(float)
        si["__sir__"] = si["shares_short"].astype(float) / shares_out.where(shares_out > 0)
    si["settlement_date"] = pd.DatetimeIndex(pd.to_datetime(si["settlement_date"])).normalize()
    si["available_at"] = pd.DatetimeIndex(pd.to_datetime(si["available_at"])).normalize()
    si = si.sort_values(["ticker", "settlement_date"])
    si["__dsi__"] = si.groupby("ticker", sort=False)["__sir__"].diff()

    grouped: dict[str, pd.DataFrame] = {
        str(t): sub.reset_index(drop=True) for t, sub in si.groupby("ticker", sort=False)
    }
    values = np.full(len(ev), np.nan)
    for i, row in enumerate(ev.itertuples(index=False)):
        sub = grouped.get(row.ticker)
        if sub is None:
            continue
        published = sub[sub["available_at"] < pd.Timestamp(row.event_date)]
        if len(published) < 2:
            continue
        last = published.iloc[-1]
        d_last = float(last["__dsi__"]) if np.isfinite(last["__dsi__"]) else np.nan
        history = published["__dsi__"].iloc[:-1].tail(n_history)
        history = history[np.isfinite(history)]
        if not np.isfinite(d_last) or len(history) < min_history:
            continue
        std = float(history.std(ddof=1))
        if not np.isfinite(std) or std <= 0.0:
            continue
        values[i] = -(d_last - float(history.mean())) / std

    out = pd.Series(
        values, index=pd.Index(ev["event_id"], name="event_id"), name="short_interest_delta"
    )
    _raise_if_all_nan(out, "short_interest_delta")
    return out


# ---------------------------------------------------------------------------
# Cuota off-exchange / ATS (FINRA OTC Transparency)
# ---------------------------------------------------------------------------


def off_exchange_share_delta(
    off_exchange: pd.DataFrame,
    events: pd.DataFrame,
    *,
    n_baseline_weeks: int = 12,
    min_baseline_weeks: int = 8,
) -> pd.Series:
    """Delta estandarizado de la cuota de volumen off-exchange (informe §8.2).

    ::

        off_share[i,w] = (ats_volume + non_ats_volume) / consolidated_volume
        delta[i] = ( off_share[última semana PUBLICADA antes de T]
                     - media(off_share, 12 semanas previas) )
                   / std(off_share, 12 semanas previas)

    **Ventana efectiva ≈ [T-35, T-15], y hay que decirlo así de claro:** FINRA
    publica el Tier 1 de NMS (todo el S&P 500) con **2 semanas de retardo**, de
    modo que la última semana publicada antes de T cubre negociación de hace 3–5
    semanas. Esta feature **no puede** describir la ventana [T-5, T-1]; es una
    variable de **condicionamiento lento** (régimen de fragmentación del valor),
    no una señal rápida de evento. Cualquier uso que suponga lo contrario es un
    error de diseño, no de datos (informe §8.2, "trampa PIT decisiva").

    Signo esperado: **negativo o nulo**, lo contrario de la intuición popular.
    Zhu (2014, *RFS*): los informados se concentran en el mercado *lit* porque en
    el dark pool el riesgo de no ejecución es máximo justo cuando se tiene
    información; Comerton-Forde y Putniņš (2015, *JFE*): las operaciones dark son
    menos informadas. Por eso no se invierte el signo aquí: la feature se entrega
    tal cual y el peso (si alguno) lo decide el modelo.

    Trampa de doble conteo (informe §8.2): FINRA reporta cada operación **una
    vez**; el volumen consolidado del denominador debe seguir la misma convención
    o la cuota queda inflada al doble. Si la tabla trae `off_exchange_share` ya
    calculada se usa directamente; en otro caso se calcula de los volúmenes y la
    verificación de convención es responsabilidad del proveedor de datos.
    """
    ev = _normalize_events(events)
    if not isinstance(off_exchange, pd.DataFrame) or len(off_exchange) == 0:
        msg = "`off_exchange` está vacío o no es un DataFrame"
        raise DataQualityError(msg)
    off = off_exchange.copy()
    _require_columns(off, ["ticker", "week_end", "available_at"], "off_exchange")
    if "off_exchange_share" in off.columns:
        off["__share__"] = off["off_exchange_share"].astype(float)
    else:
        _require_columns(off, ["ats_volume", "non_ats_volume", "total_volume"], "off_exchange")
        total = off["total_volume"].astype(float)
        off["__share__"] = (
            off["ats_volume"].astype(float) + off["non_ats_volume"].astype(float)
        ) / total.where(total > 0)
    off["week_end"] = pd.DatetimeIndex(pd.to_datetime(off["week_end"])).normalize()
    off["available_at"] = pd.DatetimeIndex(pd.to_datetime(off["available_at"])).normalize()
    off = off.sort_values(["ticker", "week_end"])

    grouped: dict[str, pd.DataFrame] = {
        str(t): sub.reset_index(drop=True) for t, sub in off.groupby("ticker", sort=False)
    }
    values = np.full(len(ev), np.nan)
    for i, row in enumerate(ev.itertuples(index=False)):
        sub = grouped.get(row.ticker)
        if sub is None:
            continue
        published = sub[sub["available_at"] < pd.Timestamp(row.event_date)]
        if len(published) < min_baseline_weeks + 1:
            continue
        last = float(published["__share__"].iloc[-1])
        base = published["__share__"].iloc[:-1].tail(n_baseline_weeks)
        base = base[np.isfinite(base)]
        if not np.isfinite(last) or len(base) < min_baseline_weeks:
            continue
        std = float(base.std(ddof=1))
        if not np.isfinite(std) or std <= 0.0:
            continue
        values[i] = (last - float(base.mean())) / std

    out = pd.Series(
        values, index=pd.Index(ev["event_id"], name="event_id"), name="off_exchange_share_delta"
    )
    _raise_if_all_nan(out, "off_exchange_share_delta")
    return out


# ---------------------------------------------------------------------------
# Insiders — Form 4
# ---------------------------------------------------------------------------


def insider_net_buy_form4(
    form4: pd.DataFrame,
    events: pd.DataFrame,
    *,
    window_days: int = 180,
    lookback_years: int = 3,
) -> pd.DataFrame:
    """Compra neta de insiders (Form 4) en ``[T-180, T-1]``, rutinaria vs. oportunista.

    ``NPR[i] = (buy_shares - sell_shares) / (buy_shares + sell_shares)`` en
    [-1, +1] (informe §8.3), agregando **solo** los Form 4 con
    ``accepted_at`` estrictamente anterior a la sesión T y transacción dentro de
    ``[T - window_days, T-1]`` en días naturales. ``available_at`` es el timestamp
    de **aceptación en EDGAR**, nunca la fecha de la transacción: usar esta última
    mete de 2 días a varios meses de look-ahead (SOX §403 fija el plazo en 2 días
    hábiles desde 2002; las presentaciones tardías existen).

    **Filtro de códigos de transacción — el error más común de la literatura
    aplicada (informe §8.3):** se conservan únicamente ``P`` (compra en mercado
    abierto) y ``S`` (venta en mercado abierto). Sin este filtro la "venta de
    insiders" está dominada por combinaciones M+S (ejercitar y vender) y por F
    (retención para impuestos), que son mecánicas y no informativas. Si la tabla
    trae la casilla ``is_10b5_1`` (obligatoria en el Form 4 desde abril de 2023),
    las operaciones bajo plan preexistente se excluyen.

    **Clasificación rutinario/oportunista (Cohen, Malloy y Pomorski 2012, *JF*
    67(3), 1009–1043):** un insider es *rutinario* en una operación si operó en el
    mismo mes natural en cada uno de los `lookback_years` años anteriores; con
    menos historia que eso la etiqueta es ``unknown`` y la operación se excluye de
    ambos subtotales (asumir oportunista por defecto sesgaría la señal — informe
    §8.3). Toda la clasificación usa solo operaciones **pasadas**: es point-in-time
    por construcción. La señal está en los oportunistas (82 pb/mes VW, 180 pb/mes
    EW en CMP; > 1 %/mes de alfa 4F en Ali y Hirshleifer 2017); los rutinarios
    rinden ≈ 0 y sirven de control.

    **Por qué la ventana es [T-180, T-1] y no [T-30, T-1] (informe §8.3):** los
    blackout periods corporativos prohíben operar desde ~2 semanas antes del
    cierre del trimestre hasta 1–2 días tras el anuncio, y Ke, Huddart y Petroni
    (2003) muestran que las ventas informadas ocurren de 3 a 9 trimestres antes de
    la ruptura, no en los 2 trimestres finales (evitación del riesgo legal). Sobre
    [T-30, T-1] la feature sería casi siempre cero y el "no operó" mezclaría "no
    sabía" con "no podía operar". NPR = NaN cuando no hay operaciones P/S en la
    ventana: la ausencia se declara, no se convierte en un cero neutral.

    Parámetros de la tabla `form4` (formato largo, una fila por transacción):
    ``ticker, transaction_date, accepted_at, transaction_code, shares`` y
    opcionalmente ``insider_id`` (necesario para clasificar; sin él todo es
    ``unknown``) e ``is_10b5_1``.
    """
    ev = _normalize_events(events)
    if not isinstance(form4, pd.DataFrame) or len(form4) == 0:
        msg = "`form4` está vacío o no es un DataFrame"
        raise DataQualityError(msg)
    f4 = form4.copy()
    _require_columns(
        f4, ["ticker", "transaction_date", "accepted_at", "transaction_code", "shares"], "form4"
    )
    f4["transaction_date"] = pd.DatetimeIndex(pd.to_datetime(f4["transaction_date"])).normalize()
    f4["accepted_at"] = pd.DatetimeIndex(pd.to_datetime(f4["accepted_at"]))
    f4["transaction_code"] = f4["transaction_code"].astype(str).str.strip().str.upper()
    f4 = f4[f4["transaction_code"].isin(["P", "S"])]
    if "is_10b5_1" in f4.columns:
        f4 = f4[~f4["is_10b5_1"].fillna(False).astype(bool)]
    if len(f4) == 0:
        msg = (
            "`form4` no contiene ninguna transacción con código P o S tras el filtro; "
            "revisa el mapeo de códigos del proveedor (informe §8.3)"
        )
        raise DataQualityError(msg)
    f4["shares"] = f4["shares"].astype(float).abs()
    has_insider = "insider_id" in f4.columns

    # Historial (ticker, insider) -> conjunto de (año, mes) de sus operaciones P/S,
    # con la fecha de cada una para poder restringir a operaciones pasadas.
    f4 = f4.sort_values("transaction_date").reset_index(drop=True)
    if has_insider:
        month_history: dict[tuple[str, str], list[tuple[pd.Timestamp, int, int]]] = {}
        for r in f4.itertuples(index=False):
            key = (str(r.ticker), str(r.insider_id))
            month_history.setdefault(key, []).append(
                (r.transaction_date, r.transaction_date.year, r.transaction_date.month)
            )

    def _classify(ticker: str, insider: str, when: pd.Timestamp) -> str:
        """Etiqueta PIT de una operación: routine / opportunistic / unknown."""
        history = month_history.get((ticker, insider), [])
        past = [(y, m) for (d, y, m) in history if d < when]
        if not past:
            return "unknown"
        first_year = min(y for y, _ in past)
        if when.year - first_year < lookback_years:
            return "unknown"
        months_by_year = {(y, m) for y, m in past}
        routine = all(
            (when.year - j, when.month) in months_by_year for j in range(1, lookback_years + 1)
        )
        return "routine" if routine else "opportunistic"

    by_ticker: dict[str, pd.DataFrame] = {
        str(t): sub.reset_index(drop=True) for t, sub in f4.groupby("ticker", sort=False)
    }

    cols = {
        "insider_net_buy_form4": np.full(len(ev), np.nan),
        "insider_net_buy_form4_opportunistic": np.full(len(ev), np.nan),
        "insider_net_buy_form4_routine": np.full(len(ev), np.nan),
    }

    def _npr(sub: pd.DataFrame) -> float:
        buys = float(sub.loc[sub["transaction_code"] == "P", "shares"].sum())
        sells = float(sub.loc[sub["transaction_code"] == "S", "shares"].sum())
        total = buys + sells
        return (buys - sells) / total if total > 0 else np.nan

    for i, row in enumerate(ev.itertuples(index=False)):
        sub = by_ticker.get(row.ticker)
        if sub is None:
            continue
        t0 = pd.Timestamp(row.event_date)
        window = sub[
            (sub["accepted_at"] < t0)
            & (sub["transaction_date"] >= t0 - pd.Timedelta(days=window_days))
            & (sub["transaction_date"] < t0)
        ]
        if len(window) == 0:
            continue
        cols["insider_net_buy_form4"][i] = _npr(window)
        if has_insider:
            labels = [
                _classify(str(r.ticker), str(r.insider_id), r.transaction_date)
                for r in window.itertuples(index=False)
            ]
            label_arr = np.asarray(labels)
            opp = window[label_arr == "opportunistic"]
            rut = window[label_arr == "routine"]
            if len(opp):
                cols["insider_net_buy_form4_opportunistic"][i] = _npr(opp)
            if len(rut):
                cols["insider_net_buy_form4_routine"][i] = _npr(rut)

    return pd.DataFrame(cols, index=pd.Index(ev["event_id"], name="event_id"))
