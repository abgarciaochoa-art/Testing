"""Estudio de eventos de resultados: retornos anormales, tests y agregación.

Implementa la segunda mitad del contrato §3.5 de `docs/ARCHITECTURE.md`:
`abnormal_returns` (AR/CAR/BHAR por evento y `tau` bajo los modelos *mean*,
*market* y *ff3*), la agregación AAR/CAAR con errores estándar, los contrastes de
Patell (1976), Boehmer-Musumeci-Poulsen (1991) y el test de rango de Corrado
(1989), la agregación por grupos (quintil de sorpresa, sector, tamaño) y la
descomposición explícita del gap overnight del día del evento.

Principios que gobiernan cada fórmula (con su fuente en `docs/research/`):

- **La ventana de estimación termina ANTES de la ventana de evento.** Se exige
  ``estimation[1] < -pre`` (`ConfigError` si no); el default ``(-250, -40)`` del
  contrato deja 10 sesiones de margen sobre el ``pre=30`` por defecto. Además, de
  la ventana de estimación se excluyen las sesiones ``[-r, +r]`` alrededor de
  cualquier otro anuncio del mismo emisor (``exclude_event_radius``, por defecto
  ±2), porque con eventos cada ~63 sesiones la ventana de estimación del trimestre
  ``q+1`` contiene los saltos de ``q``, ``q-1`` y ``q-2`` y contaminaría beta
  (`validation_methodology.md` §4.3).
- **Patell estandariza con la varianza de predicción fuera de muestra**, no con la
  residual: ``s²[i,tau] = s²_e·(1 + x_tau'(X'X)⁻¹x_tau)``, que para el modelo de
  mercado se reduce a la forma clásica ``s²_e·(1 + 1/L + (r_m,tau - r̄_m)²/Σ(r_m -
  r̄_m)²)`` de `informed_trading.md` §7.1.
- **Alrededor de resultados la varianza aumenta**, así que el t transversal de
  referencia es BMP (varianza empírica de los SCAR en sección cruzada), no el Z de
  Patell, que sobre-rechaza masivamente (`informed_trading.md` §7.1). Se ofrece
  además el ajuste de Kolari-Pynnönen (2010) por correlación transversal media
  `rho_bar` (`validation_methodology.md` §4.2), imprescindible cuando cientos de
  emisores anuncian en la misma semana.
- **Los errores estándar de AAR/CAAR son transversales por `tau`** (desviación
  típica entre eventos / raíz de N), la elección robusta a la varianza inducida por
  el evento, con el mismo deflactor opcional `rho_bar`.
- **El gap overnight se expone por separado.** Para un anuncio AMC, `tau = 0` es la
  sesión *siguiente* y la reacción se negocia mayoritariamente en el gap de
  apertura, no intradía; ocultar esa partición haría parecer capturable un retorno
  que solo existe entre el cierre y la apertura (`ARCHITECTURE.md` §3.7).

Referencias
-----------
- Patell, J. M. (1976). "Corporate Forecasts of Earnings Per Share and Stock Price
  Behavior". *Journal of Accounting Research* 14(2), 246-276.
- Boehmer, E., Musumeci, J., Poulsen, A. B. (1991). "Event-study methodology under
  conditions of event-induced variance". *JFE* 30(2), 253-272.
- Corrado, C. J. (1989). "A nonparametric test for abnormal security-price
  performance in event studies". *JFE* 23(2), 385-395; Corrado, C. J., Zivney,
  T. L. (1992). *JFQA* 27(3), 465-478 (transformación de rango con datos
  faltantes); Campbell, C. J., Wasley, C. E. (1993). *JFE* 33(1) (ventanas
  multi-día).
- Kolari, J., Pynnönen, S. (2010). "Event Study Testing with Cross-sectional
  Correlation of Abnormal Returns". *RFS* 23(11), 3996-4025.
- MacKinlay, A. C. (1997). "Event Studies in Economics and Finance". *JEL* 35(1).
- Barber, B., Lyon, J. (1997). *JFE* 43 (BHAR frente a CAR a horizonte largo).
- Fama, E., French, K. (1993). *JFE* 33 (modelo de tres factores).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats as sps

from earnings_alpha.errors import ConfigError, DataQualityError, InsufficientHistory
from earnings_alpha.events.window import normalize_events
from earnings_alpha.pit import TradingCalendar

__all__ = [
    "abnormal_returns",
    "car_window",
    "aar_caar",
    "caar_by_group",
    "quantile_groups",
    "patell_test",
    "bmp_test",
    "corrado_rank_test",
    "estimate_rho_bar",
    "overnight_decomposition",
    "EventTestResult",
]

Model = Literal["market", "ff3", "mean"]
ReturnsKind = Literal["log", "simple"]

_FF3_COLUMNS = ("mkt", "smb", "hml", "rf")

#: Columnas por evento que se repiten en cada fila `tau` del panel de salida.
_PER_EVENT_COLUMNS = (
    "n_estimation",
    "dof",
    "resid_std",
    "alpha",
    "beta_mkt",
    "beta_smb",
    "beta_hml",
)


# --------------------------------------------------------------------------- #
# Preparación de insumos                                                      #
# --------------------------------------------------------------------------- #


def _returns_matrix(prices: pd.DataFrame, kind: ReturnsKind) -> pd.DataFrame:
    """Matriz ancha fecha x ticker de retornos diarios en la unidad pedida.

    Prioridad de fuentes: ``adj_close`` (retorno **total** por construcción,
    estilo CRSP) > ``log_return`` > ``close`` (retorno de precio). Se documenta
    la elección porque no es inocua: con retornos de precio los días
    ex-dividendo aparecen como retornos negativos espurios, y la columna
    ``log_return`` del panel sintético (`data/synthetic.py`) es exactamente eso
    — ``log(close·split_factor/prev_close)``, que incluye la caída mecánica del
    ex-dividendo. Como los ex-dividendo sintéticos caen sistemáticamente dentro
    de la ventana post-evento, priorizar ``log_return`` doblaba a la baja la
    curva CAAR/PEAD (~21 pb en CAAR(+30) medidos). Por eso ``adj_close`` manda
    cuando existe.
    """
    if not isinstance(prices.index, pd.MultiIndex) or prices.index.nlevels != 2:
        msg = "`prices` debe ser el panel canónico con MultiIndex (date, ticker)"
        raise DataQualityError(msg)
    names = list(prices.index.names)
    if names != ["date", "ticker"]:
        msg = f"el panel de precios debe tener índice ['date','ticker']; tiene {names}"
        raise DataQualityError(msg)

    if "adj_close" in prices.columns and prices["adj_close"].notna().any():
        level = prices["adj_close"].unstack("ticker").sort_index()
        with np.errstate(divide="ignore", invalid="ignore"):
            log_wide = np.log(level).diff()
    elif "log_return" in prices.columns:
        log_wide = prices["log_return"].unstack("ticker").sort_index()
    elif "close" in prices.columns:
        level = prices["close"].unstack("ticker").sort_index()
        with np.errstate(divide="ignore", invalid="ignore"):
            log_wide = np.log(level).diff()
    else:
        msg = (
            "el panel de precios no tiene 'log_return', 'adj_close' ni 'close': "
            "no hay retornos que estudiar"
        )
        raise DataQualityError(msg)

    if not log_wide.index.is_monotonic_increasing or log_wide.index.has_duplicates:
        msg = "el índice de fechas del panel de precios debe ser único y creciente"
        raise DataQualityError(msg)
    if kind == "simple":
        return np.expm1(log_wide)
    return log_wide


def _as_return_series(
    obj: pd.Series | pd.DataFrame,
    kind: ReturnsKind,
    *,
    what: str,
) -> pd.Series:
    """Serie de retornos diarios indexada por fecha, en la unidad pedida.

    Acepta una `Series` de retornos **logarítmicos** o un DataFrame con columna
    ``log_return`` (se usa) o ``close`` (se log-diferencia). La convención de
    entrada es log porque es la del panel canónico; la conversión a simple se hace
    aquí para no mezclar unidades aguas abajo.
    """
    if isinstance(obj, pd.DataFrame):
        if "log_return" in obj.columns:
            series = obj["log_return"]
        elif "close" in obj.columns:
            with np.errstate(divide="ignore", invalid="ignore"):
                series = np.log(obj["close"].astype(float)).diff()
        else:
            msg = f"{what}: el DataFrame necesita columna 'log_return' o 'close'"
            raise DataQualityError(msg)
    elif isinstance(obj, pd.Series):
        series = obj
    else:
        msg = f"{what}: se esperaba Series o DataFrame; recibido {type(obj).__name__}"
        raise DataQualityError(msg)

    series = series.copy()
    series.index = pd.DatetimeIndex(pd.to_datetime(series.index)).normalize()
    series = series.sort_index()
    if series.index.has_duplicates:
        msg = f"{what}: fechas duplicadas en la serie de retornos"
        raise DataQualityError(msg)
    if kind == "simple":
        series = np.expm1(series.astype(float))
    return series.astype(float)


def _design_matrix(
    model: Model,
    dates: pd.DatetimeIndex,
    market: pd.Series | None,
    factors: pd.DataFrame | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Matriz de diseño (con intercepto) y tipo libre `rf` alineadas a `dates`.

    Para ``mean`` es la columna de unos (modelo de retorno medio constante,
    Brown-Warner 1985); para ``market`` es ``[1, r_m]`` (MacKinlay 1997); para
    ``ff3`` es ``[1, mkt-rf, smb, hml]`` y la regresión se hace en exceso de `rf`
    (Fama-French 1993).
    """
    n = len(dates)
    if model == "mean":
        return np.ones((n, 1)), np.zeros(n)
    if model == "market":
        rm = market.reindex(dates).to_numpy(dtype=float)  # type: ignore[union-attr]
        return np.column_stack([np.ones(n), rm]), np.zeros(n)
    fac = factors.reindex(dates)  # type: ignore[union-attr]
    rf = fac["rf"].to_numpy(dtype=float)
    x = np.column_stack(
        [
            np.ones(n),
            fac["mkt"].to_numpy(dtype=float) - rf,
            fac["smb"].to_numpy(dtype=float),
            fac["hml"].to_numpy(dtype=float),
        ]
    )
    return x, rf


# --------------------------------------------------------------------------- #
# Núcleo: retornos anormales                                                  #
# --------------------------------------------------------------------------- #


def abnormal_returns(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    model: Model = "market",
    estimation: tuple[int, int] = (-250, -40),
    *,
    pre: int = 30,
    post: int = 60,
    market: pd.Series | pd.DataFrame | None = None,
    factors: pd.DataFrame | None = None,
    cal: TradingCalendar | None = None,
    returns_kind: ReturnsKind = "log",
    min_estimation_obs: int = 120,
    exclude_event_radius: int | None = 2,
    include_estimation: bool = False,
) -> pd.DataFrame:
    """AR, CAR y BHAR por evento y `tau`, con estandarización de Patell.

    Contrato §3.5 de `docs/ARCHITECTURE.md`. La ventana de estimación
    ``[estimation[0], estimation[1]]`` (en sesiones relativas al evento) termina
    **antes** de la ventana de evento ``[-pre, +post]``; se valida
    ``estimation[1] < -pre`` y se lanza `ConfigError` si no se cumple, porque una
    estimación que pisa la ventana de evento contamina beta con el propio efecto
    que se quiere medir.

    Modelos de retorno esperado
    ---------------------------
    - ``"mean"``: media constante de la ventana de estimación (Brown-Warner 1985).
    - ``"market"``: ``r_i = a + b·r_m + e`` por OLS (MacKinlay 1997); requiere
      `market` (Series de retornos log del índice, o DataFrame con ``log_return`` /
      ``close``).
    - ``"ff3"``: ``r_i - rf = a + b·(mkt-rf) + s·smb + h·hml + e``
      (Fama-French 1993); requiere `factors` con columnas ``mkt, smb, hml, rf``.

    Parámetros adicionales
    ----------------------
    returns_kind:
        ``"log"`` (por defecto, la unidad del panel canónico) o ``"simple"``. El
        BHAR se calcula **siempre** sobre retornos simples compuestos
        (Barber-Lyon 1997), sea cual sea esta elección; en modo log, CAR es el
        logaritmo del ratio buy-and-hold y ambas medidas son casi redundantes —
        la distinción CAR/BHAR solo muerde en modo simple.
    min_estimation_obs:
        Mínimo de observaciones válidas en la ventana de estimación (120 por
        defecto, `informed_trading.md` §7.1). Los eventos que no llegan se
        descartan con un aviso (`warnings.warn`) y quedan registrados en
        ``result.attrs["skipped"]``; si no sobrevive ninguno se lanza
        `InsufficientHistory`. Nada desaparece en silencio.
    exclude_event_radius:
        Excluye de la ventana de estimación las sesiones a distancia <= r de
        cualquier otro anuncio del mismo emisor (por defecto ±2), para no estimar
        sigma y beta sobre los saltos de resultados de trimestres anteriores
        (`validation_methodology.md` §4.3). `None` desactiva la exclusión.
    include_estimation:
        Si True, el panel devuelto incluye también las filas de la ventana de
        estimación (``is_event_window = False``, con `ar` = residuo in-sample y
        `car`/`scar`/`bhar` = NaN). Es un requisito del test de rango de Corrado y
        de `estimate_rho_bar`.

    Devuelve
    --------
    DataFrame largo con una fila por (evento, `tau`) y columnas::

        event_id ticker event_date tau date is_event_window
        ret expected_ret ar sar car scar bhar rank_u
        n_estimation dof resid_std alpha beta_mkt beta_smb beta_hml

    - ``sar``: AR estandarizado por la desviación de **predicción** de Patell
      (1976): ``AR/√(s²_e·(1 + x'(X'X)⁻¹x))``.
    - ``car``/``scar``: acumulados desde ``tau = -pre``; ``scar`` es
      ``Σ SAR/√k`` con `k` el número de sesiones acumuladas
      (`informed_trading.md` §7.1). Para ventanas arbitrarias usar `car_window`.
    - ``rank_u``: rango de Corrado-Zivney ``K/(1+M) - 0.5`` calculado sobre la
      ventana conjunta estimación + evento del propio evento.
    - ``dof`` = L - k (grados de libertad residuales), necesario para la
      corrección ``(dof)/(dof-2)`` del Z de Patell.

    El panel lleva en ``attrs``: ``model``, ``estimation``, ``pre``, ``post``,
    ``returns_kind``, ``n_events``, ``n_skipped`` y ``skipped`` (dict
    event_id -> motivo).
    """
    # ------------------------------------------------------------- validación
    if model not in ("market", "ff3", "mean"):
        msg = f"modelo desconocido: {model!r}; opciones: 'market', 'ff3', 'mean'"
        raise ConfigError(msg)
    est_start, est_end = int(estimation[0]), int(estimation[1])
    if est_start >= est_end:
        msg = f"ventana de estimación invertida o vacía: {estimation}"
        raise ConfigError(msg)
    if pre < 0 or post < 0:
        msg = f"pre y post deben ser no negativos: pre={pre}, post={post}"
        raise ConfigError(msg)
    if est_end >= -pre:
        msg = (
            f"la ventana de estimación {estimation} termina en {est_end}, que no es "
            f"estrictamente anterior al inicio de la ventana de evento (-{pre}): "
            "la estimación contaminada con el evento invalida el estudio"
        )
        raise ConfigError(msg)
    if min_estimation_obs < 10:
        msg = f"min_estimation_obs demasiado bajo ({min_estimation_obs}); mínimo razonable: 10"
        raise ConfigError(msg)
    if model == "market" and market is None:
        msg = (
            "model='market' requiere la serie del índice (`market=`): p. ej. "
            "SyntheticMarket.market_index()['log_return']"
        )
        raise DataQualityError(msg)
    if model == "ff3":
        if factors is None:
            msg = "model='ff3' requiere `factors=` con columnas mkt, smb, hml, rf"
            raise DataQualityError(msg)
        missing = [c for c in _FF3_COLUMNS if c not in factors.columns]
        if missing:
            msg = f"a `factors` le faltan columnas para ff3: {missing}"
            raise DataQualityError(msg)

    rets = _returns_matrix(prices, returns_kind)
    rets_simple = np.expm1(rets) if returns_kind == "log" else rets

    market_series: pd.Series | None = None
    if model == "market":
        market_series = _as_return_series(market, returns_kind, what="market")  # type: ignore[arg-type]

    factors_frame: pd.DataFrame | None = None
    if model == "ff3":
        factors_frame = factors[list(_FF3_COLUMNS)].copy()  # type: ignore[index]
        factors_frame.index = pd.DatetimeIndex(
            pd.to_datetime(factors_frame.index)
        ).normalize()
        factors_frame = factors_frame.sort_index().astype(float)
        if returns_kind == "simple":
            factors_frame = np.expm1(factors_frame)

    base = normalize_events(events, cal)
    dates_index = rets.index
    date_values = dates_index.values
    n_dates = len(date_values)

    # Posiciones de todos los eventos por ticker (para la exclusión §4.3).
    event_pos_by_ticker: dict[str, np.ndarray] = {}
    if exclude_event_radius is not None:
        for ticker, sub in base.groupby("ticker", sort=False):
            p = np.searchsorted(date_values, sub["event_date"].to_numpy("datetime64[ns]"))
            p = p[(p < n_dates)]
            event_pos_by_ticker[str(ticker)] = p

    taus_event = np.arange(-pre, post + 1, dtype=np.int64)
    skipped: dict[str, str] = {}
    chunks: list[pd.DataFrame] = []

    for row in base.itertuples(index=False):
        event_id = str(row.event_id)
        ticker = str(row.ticker)
        event_date = np.datetime64(row.event_date, "ns")

        if ticker not in rets.columns:
            skipped[event_id] = "ticker sin precios en el panel"
            continue
        p = int(np.searchsorted(date_values, event_date))
        if p >= n_dates or date_values[p] != event_date:
            skipped[event_id] = "event_date fuera del panel de precios"
            continue
        if p + est_start < 0:
            skipped[event_id] = (
                f"histórico insuficiente: la ventana de estimación empieza en la "
                f"sesión {p + est_start} del panel"
            )
            continue
        if p - pre < 0 or p + post >= n_dates:
            skipped[event_id] = "ventana de evento incompleta en el panel de precios"
            continue

        col = rets[ticker].to_numpy(dtype=float)
        col_simple = rets_simple[ticker].to_numpy(dtype=float)

        # ----------------------------------------------------- estimación OLS
        est_positions = np.arange(p + est_start, p + est_end + 1)
        if exclude_event_radius is not None:
            other = event_pos_by_ticker.get(ticker, np.empty(0, dtype=np.int64))
            if len(other) > 0:
                dist = np.abs(est_positions[:, None] - other[None, :]).min(axis=1)
                est_positions = est_positions[dist > int(exclude_event_radius)]

        est_dates = pd.DatetimeIndex(date_values[est_positions])
        x_est, rf_est = _design_matrix(model, est_dates, market_series, factors_frame)
        y_est = col[est_positions] - rf_est
        valid = np.isfinite(y_est) & np.all(np.isfinite(x_est), axis=1)
        n_obs = int(valid.sum())
        if n_obs < min_estimation_obs:
            skipped[event_id] = (
                f"solo {n_obs} observaciones válidas en la ventana de estimación "
                f"(mínimo {min_estimation_obs})"
            )
            continue

        xv, yv = x_est[valid], y_est[valid]
        k = xv.shape[1]
        dof = n_obs - k
        if dof <= 4:
            skipped[event_id] = f"grados de libertad insuficientes en la estimación ({dof})"
            continue
        xtx = xv.T @ xv
        try:
            xtx_inv = np.linalg.inv(xtx)
        except np.linalg.LinAlgError:
            skipped[event_id] = "matriz de diseño singular en la ventana de estimación"
            continue
        beta = xtx_inv @ (xv.T @ yv)
        resid = yv - xv @ beta
        s2_e = float(resid @ resid) / dof
        if s2_e <= 0.0:
            skipped[event_id] = "varianza residual nula en la ventana de estimación"
            continue

        # -------------------------------------------------- ventana de evento
        ev_positions = p + taus_event
        ev_dates = pd.DatetimeIndex(date_values[ev_positions])
        x_ev, rf_ev = _design_matrix(model, ev_dates, market_series, factors_frame)
        ret_ev = col[ev_positions]
        expected_ev = x_ev @ beta + rf_ev
        ar_ev = ret_ev - expected_ev
        if not (np.isfinite(ret_ev).all() and np.isfinite(expected_ev).all()):
            skipped[event_id] = "retornos o regresores no finitos en la ventana de evento"
            continue

        # Varianza de predicción de Patell: s2_e * (1 + x'(X'X)^-1 x).
        leverage_ev = np.einsum("ij,jk,ik->i", x_ev, xtx_inv, x_ev)
        sar_ev = ar_ev / np.sqrt(s2_e * (1.0 + leverage_ev))

        car_ev = np.cumsum(ar_ev)
        scar_ev = np.cumsum(sar_ev) / np.sqrt(np.arange(1, len(sar_ev) + 1))

        # BHAR sobre retornos simples compuestos (Barber-Lyon 1997).
        ret_simple_ev = col_simple[ev_positions]
        if returns_kind == "log":
            expected_simple_ev = np.expm1(expected_ev)
        else:
            expected_simple_ev = expected_ev
        bhar_ev = np.cumprod(1.0 + ret_simple_ev) - np.cumprod(1.0 + expected_simple_ev)

        # -------------------------------------- rangos de Corrado-Zivney (U)
        est_valid_positions = est_positions[valid]
        combined_ar = np.concatenate([resid, ar_ev])
        finite = np.isfinite(combined_ar)
        m_i = int(finite.sum())
        rank_u = np.full(len(combined_ar), np.nan)
        if m_i > 0:
            rank_u[finite] = sps.rankdata(combined_ar[finite]) / (1.0 + m_i) - 0.5
        rank_u_est = rank_u[: len(resid)]
        rank_u_ev = rank_u[len(resid) :]

        per_event = {
            "n_estimation": n_obs,
            "dof": dof,
            "resid_std": float(np.sqrt(s2_e)),
            "alpha": float(beta[0]),
            "beta_mkt": float(beta[1]) if model in ("market", "ff3") else np.nan,
            "beta_smb": float(beta[2]) if model == "ff3" else np.nan,
            "beta_hml": float(beta[3]) if model == "ff3" else np.nan,
        }

        ev_frame = pd.DataFrame(
            {
                "event_id": event_id,
                "ticker": ticker,
                "event_date": row.event_date,
                "tau": taus_event,
                "date": ev_dates,
                "is_event_window": True,
                "ret": ret_ev,
                "expected_ret": expected_ev,
                "ar": ar_ev,
                "sar": sar_ev,
                "car": car_ev,
                "scar": scar_ev,
                "bhar": bhar_ev,
                "rank_u": rank_u_ev,
                **per_event,
            }
        )
        if include_estimation:
            est_frame = pd.DataFrame(
                {
                    "event_id": event_id,
                    "ticker": ticker,
                    "event_date": row.event_date,
                    "tau": est_valid_positions - p,
                    "date": pd.DatetimeIndex(date_values[est_valid_positions]),
                    "is_event_window": False,
                    "ret": yv + rf_est[valid],
                    "expected_ret": xv @ beta + rf_est[valid],
                    "ar": resid,
                    "sar": np.nan,
                    "car": np.nan,
                    "scar": np.nan,
                    "bhar": np.nan,
                    "rank_u": rank_u_est,
                    **per_event,
                }
            )
            chunks.append(pd.concat([est_frame, ev_frame], ignore_index=True))
        else:
            chunks.append(ev_frame)

    if not chunks:
        reasons = pd.Series(list(skipped.values())).value_counts().to_dict()
        msg = (
            f"ningún evento de {len(base)} tiene datos suficientes para el estudio "
            f"(motivos: {reasons})"
        )
        raise InsufficientHistory(msg)

    out = pd.concat(chunks, ignore_index=True)
    if skipped:
        warnings.warn(
            f"abnormal_returns: {len(skipped)} de {len(base)} eventos descartados "
            f"(detalle en result.attrs['skipped'])",
            UserWarning,
            stacklevel=2,
        )
    out.attrs.update(
        {
            "model": model,
            "estimation": (est_start, est_end),
            "pre": pre,
            "post": post,
            "returns_kind": returns_kind,
            "n_events": int(out.loc[out["is_event_window"], "event_id"].nunique()),
            "n_skipped": len(skipped),
            "skipped": skipped,
        }
    )
    return out


# --------------------------------------------------------------------------- #
# Ventanas de acumulación                                                     #
# --------------------------------------------------------------------------- #


def _event_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Filas de la ventana de evento, validando la forma del panel."""
    required = {"event_id", "tau", "ar", "sar", "is_event_window"}
    missing = required - set(frame.columns)
    if missing:
        msg = f"el panel de retornos anormales no tiene las columnas {sorted(missing)}"
        raise DataQualityError(msg)
    return frame.loc[frame["is_event_window"]]


def _window_sums(
    frame: pd.DataFrame, window: tuple[int, int], column: str
) -> tuple[pd.Series, pd.DataFrame]:
    """Suma por evento de `column` sobre `tau` en [w0, w1], exigiendo cobertura completa.

    Devuelve (sumas indexadas por event_id, sub-frame por evento con columnas de
    apoyo `dof` y `n_estimation`). Los eventos con la ventana incompleta o con NaN
    se excluyen; si no queda ninguno se lanza `DataQualityError`.
    """
    w0, w1 = int(window[0]), int(window[1])
    if w0 > w1:
        msg = f"ventana invertida: {window}"
        raise ConfigError(msg)
    rows = _event_rows(frame)
    taus = rows["tau"]
    if w0 < int(taus.min()) or w1 > int(taus.max()):
        msg = (
            f"la ventana {window} excede el rango de tau disponible "
            f"[{int(taus.min())}, {int(taus.max())}]"
        )
        raise ConfigError(msg)
    width = w1 - w0 + 1
    sub = rows.loc[(taus >= w0) & (taus <= w1)]
    grouped = sub.groupby("event_id", sort=False)[column]
    counts = grouped.count()
    sums = grouped.sum()
    complete = counts[counts == width].index
    if len(complete) == 0:
        msg = f"ningún evento cubre la ventana {window} con datos completos"
        raise DataQualityError(msg)
    per_event = (
        sub.loc[sub["event_id"].isin(complete), ["event_id", "dof", "n_estimation"]]
        .drop_duplicates("event_id")
        .set_index("event_id")
    )
    return sums.loc[complete], per_event


def car_window(
    frame: pd.DataFrame,
    window: tuple[int, int] = (0, 1),
    *,
    column: str = "ar",
) -> pd.Series:
    """CAR por evento sobre una ventana arbitraria ``[w0, w1]`` de tiempo-evento.

    Suma la columna `column` (por defecto ``ar``) sobre las sesiones de la ventana,
    exigiendo cobertura completa: un evento con sesiones ausentes queda fuera en
    vez de aportar un CAR sesgado a la baja. Devuelve una `Series` indexada por
    ``event_id``.
    """
    sums, _ = _window_sums(frame, window, column)
    sums.name = f"car[{window[0]},{window[1]}]"
    return sums


# --------------------------------------------------------------------------- #
# Agregación AAR / CAAR                                                       #
# --------------------------------------------------------------------------- #


def _kp_factor(n: pd.Series | np.ndarray, rho_bar: float | None) -> np.ndarray:
    """Inflactor de Kolari-Pynnönen ``sqrt(1 + (n-1)·rho_bar)`` del error estándar.

    Con `rho_bar = None` es 1 (errores estándar transversales ingenuos). Ver
    `validation_methodology.md` §4.2: con 473 eventos por trimestre y
    ``rho_bar = 0.05``, ignorarlo infla el t x5.1.
    """
    n_arr = np.asarray(n, dtype=float)
    if rho_bar is None:
        return np.ones_like(n_arr)
    if rho_bar < 0.0 or rho_bar >= 1.0:
        msg = f"rho_bar debe estar en [0, 1); recibido {rho_bar}"
        raise ConfigError(msg)
    return np.sqrt(1.0 + (n_arr - 1.0) * float(rho_bar))


def aar_caar(frame: pd.DataFrame, *, rho_bar: float | None = None) -> pd.DataFrame:
    """AAR y CAAR por `tau` con errores estándar transversales.

    Para cada `tau`: ``AAR = media_i(AR)``, ``CAAR = media_i(CAR)`` (el CAR de cada
    evento acumulado desde ``-pre``), y errores estándar transversales
    ``sd_i/√N``, la elección robusta a la varianza inducida por el evento (espíritu
    BMP). `rho_bar` aplica el inflactor de Kolari-Pynnönen a los SE, porque los
    eventos de la misma fecha comparten shocks (`validation_methodology.md` §1.2a).

    Devuelve un DataFrame indexado por `tau` con columnas ``n, aar, aar_se, aar_t,
    caar, caar_se, caar_t, mean_bhar``.
    """
    rows = _event_rows(frame)
    grouped = rows.groupby("tau", sort=True)
    n = grouped["ar"].count()
    if (n < 2).all():
        msg = "hacen falta al menos 2 eventos por tau para errores estándar transversales"
        raise DataQualityError(msg)
    kp = _kp_factor(n, rho_bar)

    aar = grouped["ar"].mean()
    aar_se = grouped["ar"].std(ddof=1) / np.sqrt(n) * kp
    caar = grouped["car"].mean()
    caar_se = grouped["car"].std(ddof=1) / np.sqrt(n) * kp

    out = pd.DataFrame(
        {
            "n": n,
            "aar": aar,
            "aar_se": aar_se,
            "aar_t": aar / aar_se,
            "caar": caar,
            "caar_se": caar_se,
            "caar_t": caar / caar_se,
            "mean_bhar": grouped["bhar"].mean(),
        }
    )
    out.attrs["rho_bar"] = rho_bar
    return out


def caar_by_group(
    frame: pd.DataFrame,
    groups: pd.Series,
    *,
    rho_bar: float | None = None,
    min_group_size: int = 5,
) -> pd.DataFrame:
    """AAR/CAAR por grupo y `tau`: el insumo del gráfico "CAAR por quintil de SUE".

    `groups` es una `Series` indexada por ``event_id`` con la etiqueta de grupo
    (quintil de sorpresa vía `quantile_groups`, sector GICS, tramo de tamaño...).
    Los eventos sin etiqueta se excluyen y se contabilizan en
    ``result.attrs["n_unlabelled"]``; los grupos con menos de `min_group_size`
    eventos lanzan `DataQualityError` en vez de devolver medias de dos
    observaciones con aspecto de curva.

    Devuelve un DataFrame con MultiIndex ``(group, tau)`` y las mismas columnas que
    `aar_caar`.
    """
    rows = _event_rows(frame).copy()
    labels = groups.reindex(rows["event_id"])
    rows["group"] = labels.to_numpy()
    n_unlabelled = int(rows.loc[rows["group"].isna(), "event_id"].nunique())
    rows = rows.dropna(subset=["group"])
    if len(rows) == 0:
        msg = "ningún evento del panel tiene etiqueta de grupo"
        raise DataQualityError(msg)

    sizes = rows.groupby("group", observed=True)["event_id"].nunique()
    too_small = sizes[sizes < min_group_size]
    if len(too_small) > 0:
        msg = (
            f"grupos con menos de {min_group_size} eventos: {too_small.to_dict()}; "
            "una media por grupo tan pequeña no es una curva CAAR, es ruido"
        )
        raise DataQualityError(msg)

    grouped = rows.groupby(["group", "tau"], sort=True, observed=True)
    n = grouped["ar"].count()
    kp = _kp_factor(n, rho_bar)
    aar = grouped["ar"].mean()
    aar_se = grouped["ar"].std(ddof=1) / np.sqrt(n) * kp
    caar = grouped["car"].mean()
    caar_se = grouped["car"].std(ddof=1) / np.sqrt(n) * kp
    out = pd.DataFrame(
        {
            "n": n,
            "aar": aar,
            "aar_se": aar_se,
            "aar_t": aar / aar_se,
            "caar": caar,
            "caar_se": caar_se,
            "caar_t": caar / caar_se,
            "mean_bhar": grouped["bhar"].mean(),
        }
    )
    out.attrs["rho_bar"] = rho_bar
    out.attrs["n_unlabelled"] = n_unlabelled
    return out


def quantile_groups(
    values: pd.Series,
    q: int = 5,
    *,
    labels: list[str] | None = None,
) -> pd.Series:
    """Asigna cuantiles (por defecto quintiles) a una característica por evento.

    Pensada para el quintil de SUE del gráfico CAAR: ``quantile_groups(sue, 5)``
    devuelve etiquetas ``Q1`` (más bajo) ... ``Q5`` (más alto) como `Categorical`
    ordenado, indexadas como `values`. Los NaN se excluyen (un SUE sin histórico
    suficiente no pertenece a ningún quintil; `InsufficientHistory` ya se aplicó
    aguas arriba al calcularlo). Los empates se rompen por orden de aparición
    (``rank(method="first")``) para que los grupos queden equilibrados.
    """
    if q < 2:
        msg = f"q debe ser >= 2; recibido {q}"
        raise ConfigError(msg)
    clean = values.dropna()
    if len(clean) < q:
        msg = f"solo {len(clean)} valores no nulos para {q} cuantiles"
        raise InsufficientHistory(msg)
    names = labels if labels is not None else [f"Q{i}" for i in range(1, q + 1)]
    if len(names) != q:
        msg = f"se esperaban {q} etiquetas; recibidas {len(names)}"
        raise ConfigError(msg)
    ranked = clean.rank(method="first")
    return pd.qcut(ranked, q, labels=names)


# --------------------------------------------------------------------------- #
# Tests de significancia                                                      #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class EventTestResult:
    """Resultado de un contraste de estudio de eventos.

    `method` no es decorativo: un estadístico sin la etiqueta de su corrección es
    inservible en este panel (`validation_methodology.md` §1.3). `mean_car` es la
    media transversal del CAR de la ventana, en la unidad de retorno del panel.
    """

    statistic: float
    p_value: float
    n_events: int
    window: tuple[int, int]
    method: str
    mean_car: float
    rho_bar: float | None = None
    notes: str = ""

    @property
    def significant(self) -> bool:
        """True al 5 % bilateral."""
        return self.p_value < 0.05

    def to_dict(self) -> dict[str, object]:
        """Vista plana para tablas de resultados."""
        return {
            "statistic": self.statistic,
            "p_value": self.p_value,
            "n_events": self.n_events,
            "window": self.window,
            "method": self.method,
            "mean_car": self.mean_car,
            "rho_bar": self.rho_bar,
        }


def patell_test(frame: pd.DataFrame, window: tuple[int, int] = (0, 1)) -> EventTestResult:
    """Z de Patell (1976) sobre el CAR de la ventana.

    Por evento ``CSAR_i = Σ_{tau∈W} SAR_i,tau / √W`` con el SAR estandarizado por
    la varianza de predicción (`informed_trading.md` §7.1); cada SAR tiene varianza
    ``dof_i/(dof_i - 2)`` (momento de una t de Student con ``dof_i = L_i - k``
    grados de libertad), de modo que::

        Z = Σ_i CSAR_i / sqrt( Σ_i dof_i/(dof_i - 2) )

    **Advertencia de uso** (obligatoria por `informed_trading.md` §7.1): alrededor
    de resultados la varianza aumenta y este test sobre-rechaza masivamente; el
    contraste de referencia del repo es `bmp_test`. Patell se reporta como cota
    superior de significancia y para comparabilidad con la literatura.
    """
    w = int(window[1]) - int(window[0]) + 1
    sar_sums, per_event = _window_sums(frame, window, "sar")
    csar = sar_sums / np.sqrt(w)
    dof = per_event["dof"].astype(float)
    if (dof <= 2).any():
        msg = "hay eventos con dof <= 2: la varianza del SAR no existe"
        raise DataQualityError(msg)
    variances = dof / (dof - 2.0)
    z = float(csar.sum() / np.sqrt(variances.sum()))
    p = float(2.0 * sps.norm.sf(abs(z)))
    mean_car = float(car_window(frame, window).mean())
    return EventTestResult(
        statistic=z,
        p_value=p,
        n_events=len(csar),
        window=(int(window[0]), int(window[1])),
        method="patell",
        mean_car=mean_car,
        notes="sobre-rechaza con varianza inducida por el evento; usar BMP como referencia",
    )


def bmp_test(
    frame: pd.DataFrame,
    window: tuple[int, int] = (0, 1),
    *,
    rho_bar: float | None = None,
) -> EventTestResult:
    """t de Boehmer-Musumeci-Poulsen (1991), robusto a la varianza inducida.

    Sobre los mismos ``SCAR_i`` de Patell::

        t_BMP = mean(SCAR) · √N / sd(SCAR, ddof=1)

    La desviación típica **transversal** de los SCAR absorbe el aumento de varianza
    del propio evento, que es exactamente lo que el Z de Patell ignora. Con
    `rho_bar` (correlación transversal media de los residuos de estimación,
    estimable con `estimate_rho_bar`) se aplica el ajuste de Kolari-Pynnönen
    (2010)::

        t_KP = t_BMP / sqrt(1 + (N-1)·rho_bar)

    (`validation_methodology.md` §4.2). Es el contraste de referencia del repo
    para el ángulo B.
    """
    w = int(window[1]) - int(window[0]) + 1
    sar_sums, _ = _window_sums(frame, window, "sar")
    scar = sar_sums / np.sqrt(w)
    n = len(scar)
    if n < 3:
        msg = f"solo {n} eventos con la ventana completa: BMP necesita al menos 3"
        raise InsufficientHistory(msg)
    sd = float(scar.std(ddof=1))
    if sd <= 0.0:
        msg = "los SCAR son constantes en sección cruzada: no hay contraste posible"
        raise DataQualityError(msg)
    t_stat = float(scar.mean() * np.sqrt(n) / sd)
    method = "bmp"
    if rho_bar is not None:
        t_stat = float(t_stat / _kp_factor(np.array([n]), rho_bar)[0])
        method = "bmp_kolari_pynnonen"
    p = float(2.0 * sps.norm.sf(abs(t_stat)))
    mean_car = float(car_window(frame, window).mean())
    return EventTestResult(
        statistic=t_stat,
        p_value=p,
        n_events=n,
        window=(int(window[0]), int(window[1])),
        method=method,
        mean_car=mean_car,
        rho_bar=rho_bar,
    )


def corrado_rank_test(
    frame: pd.DataFrame, window: tuple[int, int] = (0, 1)
) -> EventTestResult:
    """Test de rango de Corrado (1989) con la transformación de Corrado-Zivney.

    No paramétrico: inmune a la no normalidad y a la asimetría de los AR diarios,
    que alrededor de resultados es severa (colas t de Student, saltos). Para cada
    evento, los AR de la ventana conjunta estimación + evento se convierten en
    rangos ``U_i,t = K_i,t/(1 + M_i) - 0.5`` (Corrado-Zivney 1992, robusta a datos
    faltantes; es la columna ``rank_u`` del panel). Con ``Ū_t`` la media
    transversal del rango en el día de evento `t` y ``N_t`` su número de eventos::

        σ̂² = (1/T) Σ_t N_t · Ū_t²          (sobre TODOS los días, estimación incluida)
        z   = Σ_{t∈W} √N_t · Ū_t / (√W · σ̂)

    (extensión multi-día de Campbell-Wasley 1993). Incluir los días de evento en
    ``σ̂`` es la práctica estándar y hace el test ligeramente conservador bajo H1.

    Requiere que el panel se haya construido con ``include_estimation=True``: sin
    los días de estimación no hay distribución nula de rangos contra la que
    contrastar (`DataQualityError` si faltan).
    """
    required = {"rank_u", "tau", "is_event_window"}
    missing = required - set(frame.columns)
    if missing:
        msg = f"el panel no tiene las columnas {sorted(missing)}"
        raise DataQualityError(msg)
    if not (~frame["is_event_window"]).any():
        msg = (
            "el test de Corrado necesita las filas de la ventana de estimación: "
            "reconstruye el panel con abnormal_returns(..., include_estimation=True)"
        )
        raise DataQualityError(msg)
    w0, w1 = int(window[0]), int(window[1])
    if w0 > w1:
        msg = f"ventana invertida: {window}"
        raise ConfigError(msg)
    width = w1 - w0 + 1

    rows = frame.dropna(subset=["rank_u"])
    by_tau = rows.groupby("tau", sort=True)["rank_u"]
    u_bar = by_tau.mean()
    n_t = by_tau.count().astype(float)

    usable = n_t[n_t >= 2].index
    u_bar = u_bar.loc[usable]
    n_t = n_t.loc[usable]
    if len(u_bar) < 30:
        msg = f"solo {len(u_bar)} días con >= 2 eventos: el test de rango no es fiable"
        raise InsufficientHistory(msg)

    sigma2 = float((n_t * u_bar**2).mean())
    if sigma2 <= 0.0:
        msg = "varianza de rango nula: los rangos son degenerados"
        raise DataQualityError(msg)

    win_taus = [t for t in range(w0, w1 + 1) if t in u_bar.index]
    if len(win_taus) != width:
        msg = f"la ventana {window} tiene días sin datos de rango"
        raise DataQualityError(msg)
    numer = float(np.sum(np.sqrt(n_t.loc[win_taus]) * u_bar.loc[win_taus]))
    z = numer / (np.sqrt(width) * np.sqrt(sigma2))
    p = float(2.0 * sps.norm.sf(abs(z)))
    n_events = int(
        rows.loc[rows["is_event_window"] & rows["tau"].between(w0, w1), "event_id"].nunique()
    )
    mean_car = float(car_window(frame, window).mean())
    return EventTestResult(
        statistic=float(z),
        p_value=p,
        n_events=n_events,
        window=(w0, w1),
        method="corrado_rank",
        mean_car=mean_car,
    )


def estimate_rho_bar(
    frame: pd.DataFrame,
    *,
    max_events: int = 400,
    min_overlap: int = 60,
    seed: int = 0,
) -> float:
    """Correlación transversal media de los residuos de estimación entre eventos.

    Es el ``ρ̄`` del ajuste de Kolari-Pynnönen (`validation_methodology.md` §4.2):
    se calcula sobre la ventana de **estimación** (que por construcción no contiene
    el evento), emparejando los residuos de cada par de eventos por fecha de
    calendario y exigiendo al menos `min_overlap` fechas comunes. Con más de
    `max_events` eventos se toma una submuestra determinista (`seed`) para acotar
    el coste O(N²) del cálculo por pares.

    Requiere un panel construido con ``include_estimation=True``.
    """
    est = frame.loc[~frame["is_event_window"]]
    if len(est) == 0:
        msg = (
            "no hay filas de estimación en el panel: reconstruye con "
            "abnormal_returns(..., include_estimation=True)"
        )
        raise DataQualityError(msg)
    wide = est.pivot_table(index="date", columns="event_id", values="ar", aggfunc="first")
    if wide.shape[1] < 2:
        msg = "hacen falta al menos 2 eventos para estimar rho_bar"
        raise InsufficientHistory(msg)
    if wide.shape[1] > max_events:
        rng = np.random.default_rng(seed)
        keep = rng.choice(wide.columns.to_numpy(), size=max_events, replace=False)
        wide = wide[np.sort(keep)]
    corr = wide.corr(min_periods=min_overlap).to_numpy()
    upper = corr[np.triu_indices_from(corr, k=1)]
    upper = upper[np.isfinite(upper)]
    if len(upper) == 0:
        msg = (
            f"ningún par de eventos comparte >= {min_overlap} fechas de estimación: "
            "no se puede estimar rho_bar"
        )
        raise InsufficientHistory(msg)
    return float(upper.mean())


# --------------------------------------------------------------------------- #
# Gap overnight explícito                                                     #
# --------------------------------------------------------------------------- #


def overnight_decomposition(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    *,
    cal: TradingCalendar | None = None,
    returns_kind: ReturnsKind = "log",
) -> pd.DataFrame:
    """Descompone el retorno del día del evento en gap de apertura + intradía.

    Para cada evento, con ``T = event_date`` (la sesión negociable de
    `pit.tradable_date`) y ``T-1`` la sesión anterior en el panel::

        gap_return       = ln( open_T / close_{T-1} )
        intraday_return  = ln( close_T / open_T )
        event_day_return = gap_return + intraday_return = ln( close_T / close_{T-1} )

    (en modo ``simple``, las mismas magnitudes con ``exp(·) - 1`` y composición
    multiplicativa; la identidad aditiva solo es exacta en logs).

    Por qué importa (contrato §3.7, `EventBacktest`): en un anuncio **AMC** la
    sesión negociable es la *siguiente* y la reacción se realiza mayoritariamente
    en el gap — nadie puede comprar al cierre previo una vez conocida la noticia,
    así que una estrategia solo captura el tramo intradía salvo que ya tuviera la
    posición. En un anuncio **BMO** el gap de apertura del mismo día contiene
    también la reacción (el anuncio precede a la apertura). En ambos casos,
    presentar el retorno del evento sin esta partición hace parecer capturable un
    retorno que en realidad ocurre con el mercado cerrado. Para DMH la reacción es
    intradía y el gap es informativo de la víspera.

    Advertencia de datos: se usan ``open``/``close`` **sin ajustar**; si `T` es
    fecha ex-dividendo el gap incluye la caída mecánica del dividendo (sesgo
    pequeño y a la baja). Los eventos sin cobertura de precios se descartan con
    aviso y quedan en ``result.attrs["skipped"]``.

    Devuelve un DataFrame indexado por ``event_id`` con columnas ``ticker``,
    ``session`` (si la tabla de eventos la trae), ``event_date``,
    ``prev_close_date``, ``gap_return``, ``intraday_return`` y
    ``event_day_return``.
    """
    for col in ("open", "close"):
        if col not in prices.columns:
            msg = f"el panel de precios no tiene columna {col!r}, necesaria para el gap"
            raise DataQualityError(msg)
    base = normalize_events(events, cal)

    open_wide = prices["open"].unstack("ticker").sort_index()
    close_wide = prices["close"].unstack("ticker").sort_index()
    date_values = open_wide.index.values
    n_dates = len(date_values)

    has_session = "session" in base.columns
    skipped: dict[str, str] = {}
    rows: list[dict[str, object]] = []
    for row in base.itertuples(index=False):
        event_id = str(row.event_id)
        ticker = str(row.ticker)
        if ticker not in open_wide.columns:
            skipped[event_id] = "ticker sin precios en el panel"
            continue
        event_date = np.datetime64(row.event_date, "ns")
        p = int(np.searchsorted(date_values, event_date))
        if p >= n_dates or date_values[p] != event_date:
            skipped[event_id] = "event_date fuera del panel de precios"
            continue
        if p == 0:
            skipped[event_id] = "sin sesión previa en el panel: el gap no es calculable"
            continue
        o_t = float(open_wide[ticker].iloc[p])
        c_t = float(close_wide[ticker].iloc[p])
        c_prev = float(close_wide[ticker].iloc[p - 1])
        if not (np.isfinite(o_t) and np.isfinite(c_t) and np.isfinite(c_prev)):
            skipped[event_id] = "precios no finitos en el día del evento o la víspera"
            continue
        if min(o_t, c_t, c_prev) <= 0.0:
            skipped[event_id] = "precios no positivos: retorno logarítmico indefinido"
            continue
        gap_log = float(np.log(o_t / c_prev))
        intra_log = float(np.log(c_t / o_t))
        if returns_kind == "log":
            gap, intra, total = gap_log, intra_log, gap_log + intra_log
        else:
            gap, intra = float(np.expm1(gap_log)), float(np.expm1(intra_log))
            total = float((1.0 + gap) * (1.0 + intra) - 1.0)
        rows.append(
            {
                "event_id": event_id,
                "ticker": ticker,
                "session": getattr(row, "session", None) if has_session else None,
                "event_date": row.event_date,
                "prev_close_date": pd.Timestamp(date_values[p - 1]),
                "gap_return": gap,
                "intraday_return": intra,
                "event_day_return": total,
            }
        )

    if not rows:
        msg = f"ningún evento de {len(base)} tiene precios para descomponer el gap"
        raise InsufficientHistory(msg)
    if skipped:
        warnings.warn(
            f"overnight_decomposition: {len(skipped)} de {len(base)} eventos sin "
            "precios suficientes (detalle en result.attrs['skipped'])",
            UserWarning,
            stacklevel=2,
        )
    out = pd.DataFrame(rows).set_index("event_id")
    if not has_session:
        out = out.drop(columns=["session"])
    out.attrs.update({"returns_kind": returns_kind, "n_skipped": len(skipped), "skipped": skipped})
    return out
