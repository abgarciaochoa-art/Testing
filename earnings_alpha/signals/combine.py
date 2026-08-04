"""Combinación de señales: pesos fijos, ponderación por IC y ortogonalización.

Tercera pata del contrato §3.6. Combinar factores parece trivial —sumar
z-scores— y es donde se pierden la mayoría de los backtests honestos, por dos
razones concretas:

1. **Los factores no son independientes.** `docs/research/fundamental_factors.md`
   §14 documenta los solapamientos medidos: `SUE_price`↔`E/P`, `CFO/NI` ≡
   `1 − PACC`, `F_ACCRUAL` ⊂ F-Score, revisiones de analistas ↔ momentum de
   precio. Sumar z-scores a pesos iguales sobre esa batería concentra el riesgo
   en la dirección repetida y llama "diversificación" a lo contrario. De ahí
   `orthogonalize` (Gram-Schmidt secuencial por fecha).
2. **Ponderar por IC es una invitación al look-ahead.** El IC de la fecha `t` no
   se conoce en `t`: hace falta el retorno futuro de `h` sesiones, así que solo
   es observable en `t + h`. Ponderar con el IC de toda la muestra —o incluso con
   el IC "hasta t" calculado sin descontar el horizonte— produce Sharpes
   fantásticos e irreproducibles. `expanding_ic_weights` fecha cada IC por su
   **fecha de realización** y solo usa las estrictamente anteriores a la fecha de
   decisión.

Piezas
------
* `cross_section_ic`: rank IC (o Pearson) por fecha, con mínimo de sección
  cruzada.
* `expanding_ic_weights`: pesos de ventana expansiva (o EWMA) a partir de una
  serie de ICs, con la contabilidad point-in-time explícita.
* `ic_weighted_weights` / `ic_weighted_combine`: la cadena completa.
* `fixed_weight_combine`: pesos fijos con política declarada para señales
  ausentes.
* `orthogonalize`: Gram-Schmidt secuencial cross-section (y ortogonalización
  simétrica de Löwdin como alternativa sin orden privilegiado).
* `combine`: despachador `weights | ic_weighted | orthogonalize` del contrato.

Referencias
-----------
- Grinold, R. y Kahn, R. (2000), *Active Portfolio Management*, cap. 3-4 y 12:
  ley fundamental de la gestión activa, ``IR ≈ IC·√BR``, y combinación óptima de
  señales ``w ∝ Σ⁻¹·IC`` cuando las señales están correlacionadas.
- Ledoit, O. y Wolf, M. (2004), *A well-conditioned estimator for large
  dimensional covariance matrices*: motivación del encogimiento de la matriz de
  correlación antes de invertirla.
- Löwdin, P.-O. (1950), *On the Non-Orthogonality Problem*: ortogonalización
  simétrica ``X S^{-1/2}``, la transformación ortogonal más cercana a la original.
- Klein, R. y Chow, V. (2013), *Orthogonalized factors and systematic risk
  decomposition*, QREF 53: su uso en carteras de factores.
- Golub, G. y Van Loan, C. (2013), *Matrix Computations* §5.2: Gram-Schmidt
  modificado y su relación con la factorización QR.
- `docs/research/validation_methodology.md` §2 (rank IC como métrica principal,
  mínimo de 30 nombres en la sección cruzada) y §5.5 (cada esquema de
  ponderación probado cuenta como un ensayo más en el `N` del Sharpe deflactado).
- `docs/research/informed_trading.md` §12.2 (ortogonalización obligatoria de las
  features de pre-evento antes de combinarlas).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, cast

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from earnings_alpha.errors import DataQualityError, InsufficientHistory
from earnings_alpha.pit.calendar import TradingCalendar
from earnings_alpha.signals.transforms import (
    InsufficientPolicy,
    NaNPolicy,
    check_panel,
    rank_pct,
    residualize,
    zscore,
)

__all__ = [  # noqa: RUF022 - orden temático, no alfabético
    "cross_section_ic",
    "observability_dates",
    "expanding_ic_weights",
    "ic_weighted_weights",
    "ic_weighted_combine",
    "fixed_weight_combine",
    "orthogonalize",
    "symmetric_orthogonalize",
    "combine",
    "MIN_CROSS_SECTION",
]

MIN_CROSS_SECTION = 30
"""Mínimo de nombres para que la IC de una fecha sea un número y no ruido.

`validation_methodology.md` §2.2: por debajo de 30 activos el estimador de
correlación tiene un sesgo y una varianza que contaminan todo lo que se calcule
después, de modo que la IC de esa fecha es `NaN`, no un número.
"""

_MissingPolicy = Literal["renormalize", "skip", "nan"]


# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------


def _as_signal_frame(signals: pd.Series | pd.DataFrame, name: str = "signals") -> pd.DataFrame:
    """Normaliza la entrada a DataFrame de panel y valida el índice."""
    if isinstance(signals, pd.Series):
        frame = signals.to_frame(name=signals.name if signals.name is not None else "signal")
    elif isinstance(signals, pd.DataFrame):
        frame = signals
    else:
        msg = f"`{name}` debe ser Series o DataFrame con MultiIndex (date, ticker)"
        raise DataQualityError(msg)
    check_panel(frame, name=name)
    if frame.shape[1] == 0:
        msg = f"`{name}` no tiene ninguna columna de señal"
        raise DataQualityError(msg)
    if len(frame) == 0:
        msg = f"`{name}` está vacío: no hay nada que combinar"
        raise InsufficientHistory(msg)
    # Los nombres de señal se normalizan a texto: los pesos, los ICs y el panel
    # se alinean por nombre, y una etiqueta entera frente a su versión en texto
    # produciría una alineación vacía —y una combinación silenciosamente nula—.
    labels = [str(c) for c in frame.columns]
    dupes = sorted({c for c in labels if labels.count(c) > 1})
    if dupes:
        msg = f"`{name}` tiene columnas duplicadas: {dupes}"
        raise DataQualityError(msg)
    out = frame.astype("float64")
    out.columns = pd.Index(labels, name=frame.columns.name)
    return out


def _panel_dates(frame: pd.DataFrame | pd.Series) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(frame.index.get_level_values(0).unique()).sort_values()


def _standardize_panel(
    frame: pd.DataFrame,
    how: Literal["zscore", "rank", "none"],
    *,
    min_obs: int,
    on_insufficient: InsufficientPolicy,
) -> pd.DataFrame:
    """Lleva todas las señales a una escala comparable antes de ponderarlas.

    Sin esto, un peso de 0.5 sobre un factor con desviación típica 3 y otro con
    desviación típica 0.01 no significa "mitad y mitad": significa que el segundo
    factor no existe.
    """
    if how == "none":
        return frame
    if how == "zscore":
        out = zscore(frame, min_obs=min_obs, on_insufficient=on_insufficient)
    elif how == "rank":
        out = rank_pct(frame, mode="normal", min_obs=min_obs, on_insufficient=on_insufficient)
    else:  # pragma: no cover - protegido por el tipo
        msg = f"estandarización desconocida: {how!r}"
        raise ValueError(msg)
    return cast(pd.DataFrame, out)


def _apply_weights(
    panel: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    missing: _MissingPolicy,
    min_signals: int,
) -> pd.Series:
    """Combina el panel con una matriz de pesos por fecha.

    `weights` está indexado por fecha (una fila por fecha de decisión) y tiene
    las mismas columnas que `panel`. Las combinaciones de pesos fijos usan la
    misma fila repetida; la ponderación por IC, una fila distinta cada día.
    """
    cols = list(panel.columns)
    aligned = weights.reindex(columns=cols)
    row_dates = panel.index.get_level_values(0)
    w = aligned.reindex(row_dates).to_numpy(dtype="float64")
    v = panel.to_numpy(dtype="float64")

    usable = np.isfinite(v) & np.isfinite(w)
    n_present = usable.sum(axis=1)
    contrib = np.where(usable, v * w, 0.0).sum(axis=1)
    l1_present = np.where(usable, np.abs(w), 0.0).sum(axis=1)

    if missing == "renormalize":
        with np.errstate(invalid="ignore", divide="ignore"):
            score = np.where(l1_present > 0, contrib / l1_present, np.nan)
    elif missing == "skip":
        score = np.where(n_present > 0, contrib, np.nan)
    elif missing == "nan":
        complete = n_present == len(cols)
        score = np.where(complete, contrib, np.nan)
    else:  # pragma: no cover - protegido por el tipo
        msg = f"política de señales ausentes desconocida: {missing!r}"
        raise ValueError(msg)

    score = np.where(n_present >= max(min_signals, 1), score, np.nan)
    return pd.Series(score, index=panel.index, name="score")


# ---------------------------------------------------------------------------
# Coeficiente de información por fecha
# ---------------------------------------------------------------------------


def cross_section_ic(
    signals: pd.Series | pd.DataFrame,
    forward_returns: pd.Series,
    *,
    method: Literal["spearman", "pearson"] = "spearman",
    min_obs: int = MIN_CROSS_SECTION,
) -> pd.DataFrame:
    """Coeficiente de información de cada señal, fecha a fecha.

    ``IC_t = corr(s_{i,t}, r_{i,t→t+h})`` sobre la sección cruzada de la fecha
    `t`, con `s` **ya conocible en t** (contrato: la señal viene desplazada por
    `pit.asof_join`, aquí no se desplaza nada) y `r` el retorno *forward*.

    Por defecto **Spearman** (rank IC), que es la métrica principal del repo
    (`validation_methodology.md` §2.1): los factores fundamentales tienen colas
    patológicas —un *earnings yield* con beneficio próximo a cero produce |z|>20—
    y la IC de Pearson sobre esos datos mide el efecto de tres observaciones.
    Además la rank IC es invariante a las transformaciones monótonas que aplica
    `signals.transforms`, así que no cambia según se winsorice al 1 % o al 2.5 %.

    Parámetros
    ----------
    forward_returns:
        `Series` ``(date, ticker)`` con el retorno futuro **alineado en la fecha
        de la señal**: el valor de `(t, i)` es el retorno de `i` entre `t` y
        `t+h`. Este módulo no construye ese retorno ni conoce `h`; quien lo
        construya es responsable de no colar el retorno del propio día de
        decisión si la ejecución es en la apertura.
    min_obs:
        Mínimo de pares válidos por fecha. Por debajo, `NaN`.

    Devuelve
    --------
    `DataFrame` indexado por fecha, una columna por señal. Las fechas sin
    sección cruzada suficiente aparecen como `NaN`, no se omiten: la serie de
    ICs debe conservar el calendario para que las medias móviles posteriores
    cuenten bien.
    """
    frame = _as_signal_frame(signals)
    if not isinstance(forward_returns, pd.Series):
        msg = "`forward_returns` debe ser una Series con MultiIndex (date, ticker)"
        raise DataQualityError(msg)
    check_panel(forward_returns, name="forward_returns")

    dates = _panel_dates(frame)
    ret = forward_returns.astype("float64").reindex(frame.index)
    out: dict[str, pd.Series] = {}

    for col in frame.columns:
        pair = pd.DataFrame({"x": frame[col], "y": ret}).dropna()
        if len(pair) == 0:
            out[str(col)] = pd.Series(np.nan, index=dates)
            continue
        day = pd.Index(pair.index.get_level_values(0))
        if method == "spearman":
            grouped = pair.groupby(day)
            xv = grouped["x"].rank()
            yv = grouped["y"].rank()
        elif method == "pearson":
            xv, yv = pair["x"], pair["y"]
        else:  # pragma: no cover - protegido por el tipo
            msg = f"método de correlación desconocido: {method!r}"
            raise ValueError(msg)
        xc = xv - xv.groupby(day).transform("mean")
        yc = yv - yv.groupby(day).transform("mean")
        cov = (xc * yc).groupby(day).sum()
        vx = (xc * xc).groupby(day).sum()
        vy = (yc * yc).groupby(day).sum()
        n = pair["x"].groupby(day).count()
        denom = np.sqrt(vx.to_numpy() * vy.to_numpy())
        with np.errstate(invalid="ignore", divide="ignore"):
            ic = np.where(denom > 0, cov.to_numpy() / denom, np.nan)
        ic = np.where(n.to_numpy() >= min_obs, ic, np.nan)
        out[str(col)] = pd.Series(ic, index=pd.DatetimeIndex(cov.index)).reindex(dates)

    result = pd.DataFrame(out, index=dates)
    result.index.name = "date"
    result.columns = pd.Index([str(c) for c in frame.columns], name="signal")
    if not result.notna().to_numpy().any():
        msg = (
            "ninguna fecha alcanza el mínimo de sección cruzada "
            f"(min_obs={min_obs}) o no hay solape entre señales y retornos: "
            "la serie de IC sería íntegramente NaN"
        )
        raise InsufficientHistory(msg)
    return result


# ---------------------------------------------------------------------------
# Pesos por IC sin mirar al futuro
# ---------------------------------------------------------------------------


def observability_dates(
    dates: pd.DatetimeIndex,
    horizon: int,
    calendar: TradingCalendar | None = None,
) -> pd.DatetimeIndex:
    """Fecha en la que el IC de cada fecha de señal pasa a ser **observable**.

    El IC de `t` mide la correlación con el retorno de `t` a `t+h`; ese retorno
    no existe hasta el cierre de `t+h`. Por tanto el IC de `t` no puede influir
    en ninguna decisión anterior a `t+h`. Esta función materializa ese desfase,
    que es la única defensa real contra el look-ahead en la ponderación por IC.

    Con `calendar`, `horizon` se cuenta en **sesiones de mercado**; sin él, en
    posiciones de la propia rejilla de fechas recibida (que en un panel diario de
    precios *es* la rejilla de sesiones). Las últimas `horizon` fechas no tienen
    fecha de observabilidad dentro de la muestra y salen como `NaT`: su IC nunca
    llega a usarse, que es exactamente lo correcto.
    """
    if horizon < 0:
        msg = f"horizon debe ser >= 0; recibido {horizon}"
        raise ValueError(msg)
    idx = pd.DatetimeIndex(dates)
    if calendar is None:
        shifted = idx.to_series().shift(-horizon)
        return pd.DatetimeIndex(shifted.to_numpy(), name="observable_at")
    values = [calendar.shift(d.date(), horizon) for d in idx]
    return pd.DatetimeIndex(pd.to_datetime(values), name="observable_at")


def _prefix_stats(
    frame: pd.DataFrame, halflife: float | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Media, desviación típica y recuento acumulados fila a fila.

    En la posición `k` devuelve el estadístico calculado **solo** con las filas
    `0..k`. Tanto `expanding` como `ewm` son recursivas hacia delante, de modo
    que una sola pasada sirve para todas las fechas de decisión.
    """
    counts = frame.expanding(min_periods=1).count().to_numpy(dtype="float64")
    if halflife is None:
        mean = frame.expanding(min_periods=1).mean().to_numpy(dtype="float64")
        std = frame.expanding(min_periods=2).std().to_numpy(dtype="float64")
    else:
        ewm = frame.ewm(halflife=halflife, min_periods=1, ignore_na=True)
        mean = ewm.mean().to_numpy(dtype="float64")
        std = frame.ewm(halflife=halflife, min_periods=2, ignore_na=True).std().to_numpy("float64")
    return mean, std, counts


def _signal_corr_prefix(
    panel: pd.DataFrame, dates: pd.DatetimeIndex
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Media acumulada de las matrices de correlación cross-section por fecha.

    La correlación entre señales en la fecha `s` es observable **en `s`**: no
    interviene ningún retorno futuro. Por eso su ventana de acumulación solo
    exige `s < t`, sin el desfase de horizonte que sí necesita el IC.
    """
    k = panel.shape[1]
    mats: list[np.ndarray] = []
    keep: list[pd.Timestamp] = []
    values = panel.to_numpy(dtype="float64")
    day = panel.index.get_level_values(0)
    codes, uniques = pd.factorize(day, sort=True)
    order = np.argsort(codes, kind="stable")
    offsets = np.searchsorted(codes[order], np.arange(len(uniques) + 1))
    for i, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        rows = order[start:stop]
        block = values[rows]
        block = block[np.isfinite(block).all(axis=1)]
        if len(block) < max(k + 2, 5):
            continue
        centered = block - block.mean(axis=0, keepdims=True)
        sd = centered.std(axis=0, ddof=1)
        if not np.all(sd > 0):
            continue
        corr = np.corrcoef(centered, rowvar=False)
        if not np.all(np.isfinite(corr)):
            continue
        mats.append(np.atleast_2d(corr))
        keep.append(pd.Timestamp(uniques[i]))
    if not mats:
        msg = (
            "no hay ninguna fecha con sección cruzada suficiente para estimar la "
            "matriz de correlación entre señales; method='ic_cov' no es aplicable"
        )
        raise InsufficientHistory(msg)
    stack = np.stack(mats)
    cum = np.cumsum(stack, axis=0) / np.arange(1, len(stack) + 1)[:, None, None]
    return pd.DatetimeIndex(keep), cum


def expanding_ic_weights(
    ic: pd.DataFrame,
    *,
    horizon: int = 1,
    calendar: TradingCalendar | None = None,
    observable_at: pd.Series | pd.DatetimeIndex | None = None,
    dates: pd.DatetimeIndex | None = None,
    method: Literal["ic", "ir", "ic_cov", "equal"] = "ic",
    min_periods: int = 12,
    halflife: float | None = None,
    extra_lag_days: int = 0,
    shrinkage: float = 0.0,
    clip_negative: bool = True,
    normalize: Literal["l1", "sum", "none"] = "l1",
    warmup: Literal["equal", "zero", "nan", "raise"] = "equal",
    signals: pd.DataFrame | None = None,
    corr_shrinkage: float = 0.2,
) -> pd.DataFrame:
    """Pesos por IC histórico con ventana expansiva y **sin mirar al futuro**.

    Regla point-in-time que implementa, y que es todo el valor de esta función:
    el peso aplicado en la fecha de decisión `t` se estima únicamente con los
    ICs cuya **fecha de observabilidad** (`observability_dates`) es
    estrictamente anterior a `t - extra_lag_days`. Un IC cuya ventana forward
    cierra justo en `t` **no** se usa: en el mejor de los casos se conocería al
    cierre de `t`, no en el momento de decidir.

    Consecuencia deseada: al principio de la muestra no hay pesos estimables. Eso
    no es un defecto, es la realidad de un inversor que empieza; `warmup`
    decide qué hacer entonces, y la opción por defecto (`"equal"`) es la única
    honesta —pesos iguales, que es lo que se puede hacer sin historia—.

    Parámetros
    ----------
    method:
        - ``"ic"``: peso proporcional al IC medio histórico.
        - ``"ir"``: proporcional a ``media(IC)/sd(IC)``, el *information ratio*
          de la propia serie de ICs. Penaliza señales de IC alto pero
          inestable, que son las que peor sobreviven fuera de muestra.
        - ``"ic_cov"``: combinación de Grinold-Kahn ``w ∝ Σ⁻¹·IC``, con `Σ` la
          correlación media cross-section **entre señales** (requiere pasar
          `signals`). Es la única de las cuatro que descuenta el solapamiento
          entre factores, que en esta batería es grande
          (`fundamental_factors.md` §14).
        - ``"equal"``: pesos iguales; línea base contra la que hay que comparar
          cualquier esquema más sofisticado antes de creérselo.
    min_periods:
        ICs observables mínimos para que una señal reciba peso propio. Con menos,
        su peso es 0 (no participa) hasta que acumule historia.
    halflife:
        Si se indica, la media y la desviación del IC son exponenciales con esa
        semivida (en número de observaciones de IC), no expansivas. Reacciona
        antes al decaimiento de un factor a costa de más varianza.
    shrinkage:
        Encogimiento hacia pesos iguales, en [0, 1]. `0.3` es un valor razonable:
        la estimación del IC medio con 30-60 observaciones es muy ruidosa y el
        encogimiento hacia la línea base rara vez pierde y a menudo gana.
    clip_negative:
        Trunca a cero los pesos negativos. Un IC medio negativo es un hallazgo
        que hay que investigar (`validation_methodology.md` §2.2), no una
        invitación a invertir el signo del factor sobre la marcha: hacerlo dentro
        del propio combinador es sobreajuste puro.
    normalize:
        ``"l1"`` normaliza a ``Σ|w| = 1`` (mantiene la escala de la señal
        combinada aunque haya pesos negativos), ``"sum"`` a ``Σw = 1``,
        ``"none"`` deja los pesos crudos.

    Devuelve
    --------
    `DataFrame` indexado por las fechas de decisión (`dates`, o el índice de
    `ic`), una columna por señal.
    """
    if not isinstance(ic, pd.DataFrame):
        msg = "`ic` debe ser un DataFrame indexado por fecha con una columna por señal"
        raise DataQualityError(msg)
    if not is_datetime64_any_dtype(ic.index):
        msg = "el índice de `ic` debe ser de tipo fecha"
        raise DataQualityError(msg)
    if not 0.0 <= shrinkage <= 1.0:
        msg = f"shrinkage debe estar en [0, 1]; recibido {shrinkage}"
        raise ValueError(msg)
    if not 0.0 <= corr_shrinkage <= 1.0:
        msg = f"corr_shrinkage debe estar en [0, 1]; recibido {corr_shrinkage}"
        raise ValueError(msg)
    if extra_lag_days < 0:
        msg = f"extra_lag_days debe ser >= 0; recibido {extra_lag_days}"
        raise ValueError(msg)

    ic = ic.sort_index()
    cols = list(ic.columns)
    k = len(cols)
    targets = pd.DatetimeIndex(ic.index if dates is None else dates).sort_values()

    if observable_at is None:
        obs = observability_dates(pd.DatetimeIndex(ic.index), horizon, calendar)
    else:
        obs = pd.DatetimeIndex(pd.to_datetime(pd.Series(observable_at).to_numpy()))
        if len(obs) != len(ic):
            msg = f"`observable_at` debe tener {len(ic)} elementos; tiene {len(obs)}"
            raise DataQualityError(msg)

    live = pd.notna(obs)
    ic_live = ic[live]
    obs_live = obs[live]
    order = np.argsort(obs_live.to_numpy(), kind="stable")
    ic_sorted = ic_live.iloc[order]
    obs_sorted = obs_live.to_numpy()[order]

    if len(ic_sorted) == 0:
        mean = std = counts = np.zeros((0, k))
    else:
        mean, std, counts = _prefix_stats(ic_sorted, halflife)

    cut = targets.to_numpy() - np.timedelta64(int(extra_lag_days), "D")
    pos = np.searchsorted(obs_sorted, cut, side="left") - 1
    have = pos >= 0

    mu = np.full((len(targets), k), np.nan)
    sd = np.full((len(targets), k), np.nan)
    cnt = np.zeros((len(targets), k))
    if have.any():
        mu[have] = mean[pos[have]]
        sd[have] = std[pos[have]]
        cnt[have] = counts[pos[have]]

    enough = cnt >= max(min_periods, 1)

    if method == "equal":
        raw = np.where(enough, 1.0, np.nan)
    elif method == "ic":
        raw = np.where(enough, mu, np.nan)
    elif method == "ir":
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(sd > 0, mu / sd, np.nan)
        raw = np.where(enough, ratio, np.nan)
    elif method == "ic_cov":
        if signals is None:
            msg = "method='ic_cov' requiere el panel `signals` para estimar Σ entre señales"
            raise DataQualityError(msg)
        panel = _as_signal_frame(signals).reindex(columns=cols)
        corr_dates, corr_cum = _signal_corr_prefix(panel, _panel_dates(panel))
        corr_pos = np.searchsorted(corr_dates.to_numpy(), targets.to_numpy(), side="left") - 1
        raw = np.full((len(targets), k), np.nan)
        eye = np.eye(k)
        for i in range(len(targets)):
            if corr_pos[i] < 0 or not enough[i].any():
                continue
            sigma = (1.0 - corr_shrinkage) * corr_cum[corr_pos[i]] + corr_shrinkage * eye
            active = enough[i]
            mu_i = np.where(active, mu[i], 0.0)
            sub = sigma[np.ix_(active, active)]
            try:
                sol = np.linalg.solve(sub, mu_i[active])
            except np.linalg.LinAlgError:  # pragma: no cover - Σ encogida es definida positiva
                sol = np.linalg.lstsq(sub, mu_i[active], rcond=None)[0]
            row = np.full(k, np.nan)
            row[active] = sol
            raw[i] = row
    else:  # pragma: no cover - protegido por el tipo
        msg = f"método de ponderación desconocido: {method!r}"
        raise ValueError(msg)

    if clip_negative:
        raw = np.where(np.isnan(raw), np.nan, np.maximum(raw, 0.0))

    # Una señal sin historia suficiente no entra (peso 0). Si NINGUNA la tiene,
    # o si el recorte de negativos deja la fila entera a cero, se aplica `warmup`.
    weights = np.where(np.isnan(raw), 0.0, raw)
    degenerate = ~np.isfinite(raw).any(axis=1) | np.isclose(np.abs(weights).sum(axis=1), 0.0)
    if degenerate.any():
        if warmup == "raise":
            first_bad = targets[degenerate][0]
            msg = (
                f"{int(degenerate.sum())} fechas sin IC observable suficiente "
                f"(la primera, {first_bad.date()}). Sube el tamaño de la muestra, "
                "baja `min_periods` o elige warmup='equal'."
            )
            raise InsufficientHistory(msg)
        if warmup == "equal":
            weights[degenerate] = 1.0
        elif warmup == "nan":
            weights[degenerate] = np.nan
        else:  # "zero"
            weights[degenerate] = 0.0

    weights = _normalize_weights(weights, normalize)
    if shrinkage > 0:
        base = _normalize_weights(np.ones_like(weights), normalize)
        weights = (1.0 - shrinkage) * weights + shrinkage * base
        weights = _normalize_weights(weights, normalize)

    out = pd.DataFrame(weights, index=targets, columns=pd.Index(cols, name="signal"))
    out.index.name = "date"
    return out


def _normalize_weights(w: np.ndarray, how: Literal["l1", "sum", "none"]) -> np.ndarray:
    if how == "none":
        return w
    if how == "l1":
        denom = np.nansum(np.abs(w), axis=1, keepdims=True)
    elif how == "sum":
        denom = np.nansum(w, axis=1, keepdims=True)
    else:  # pragma: no cover - protegido por el tipo
        msg = f"normalización desconocida: {how!r}"
        raise ValueError(msg)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.abs(denom) > 0, w / denom, np.nan)


def ic_weighted_weights(
    signals: pd.Series | pd.DataFrame,
    forward_returns: pd.Series,
    *,
    horizon: int = 1,
    calendar: TradingCalendar | None = None,
    ic_method: Literal["spearman", "pearson"] = "spearman",
    standardize: Literal["zscore", "rank", "none"] = "zscore",
    min_obs: int = 20,
    min_obs_ic: int = MIN_CROSS_SECTION,
    on_insufficient: InsufficientPolicy = "nan",
    **weight_kwargs: object,
) -> pd.DataFrame:
    """Pesos por IC listos para aplicar: estandariza, calcula ICs y los pondera.

    Se expone aparte de `ic_weighted_combine` porque los pesos son el objeto que
    hay que auditar: son ellos los que pueden mirar al futuro, no la suma
    ponderada.
    """
    frame = _as_signal_frame(signals)
    std = _standardize_panel(frame, standardize, min_obs=min_obs, on_insufficient=on_insufficient)
    ic = cross_section_ic(std, forward_returns, method=ic_method, min_obs=min_obs_ic)
    return expanding_ic_weights(
        ic,
        horizon=horizon,
        calendar=calendar,
        dates=_panel_dates(frame),
        signals=std,
        **weight_kwargs,  # type: ignore[arg-type]
    )


def ic_weighted_combine(
    signals: pd.Series | pd.DataFrame,
    forward_returns: pd.Series,
    *,
    horizon: int = 1,
    calendar: TradingCalendar | None = None,
    ic_method: Literal["spearman", "pearson"] = "spearman",
    standardize: Literal["zscore", "rank", "none"] = "zscore",
    standardize_output: bool = True,
    missing: _MissingPolicy = "renormalize",
    min_signals: int = 1,
    min_obs: int = 20,
    min_obs_ic: int = MIN_CROSS_SECTION,
    on_insufficient: InsufficientPolicy = "nan",
    **weight_kwargs: object,
) -> pd.Series:
    """Señal combinada con pesos proporcionales al IC histórico, sin look-ahead.

    Encadena `cross_section_ic` → `expanding_ic_weights` → suma ponderada. La
    propiedad que la hace utilizable en un backtest es la de
    `expanding_ic_weights`: el peso del día `t` solo depende de ICs ya
    realizados antes de `t`. Verificado por el test explícito
    `test_ic_weights_no_lookahead` de `tests/test_signals.py`, que altera todos
    los datos posteriores a una fecha y exige que los pesos anteriores no se
    muevan ni un bit.

    Advertencia de método (`validation_methodology.md` §5.5): cada variante de
    ponderación probada cuenta como un ensayo más en el `N` del Sharpe
    deflactado. Probar `ic`, `ir`, `ic_cov` y `equal` y quedarse con la mejor son
    cuatro ensayos, no uno.
    """
    frame = _as_signal_frame(signals)
    std = _standardize_panel(frame, standardize, min_obs=min_obs, on_insufficient=on_insufficient)
    weights = ic_weighted_weights(
        frame,
        forward_returns,
        horizon=horizon,
        calendar=calendar,
        ic_method=ic_method,
        standardize=standardize,
        min_obs=min_obs,
        min_obs_ic=min_obs_ic,
        on_insufficient=on_insufficient,
        **weight_kwargs,
    )
    score = _apply_weights(std, weights, missing=missing, min_signals=min_signals)
    if standardize_output:
        score = cast(
            pd.Series, zscore(score, min_obs=min_obs, on_insufficient=on_insufficient)
        )
    score.name = "score"
    return score


# ---------------------------------------------------------------------------
# Pesos fijos
# ---------------------------------------------------------------------------


def fixed_weight_combine(
    signals: pd.Series | pd.DataFrame,
    weights: Mapping[str, float] | pd.Series | Sequence[float] | None = None,
    *,
    standardize: Literal["zscore", "rank", "none"] = "zscore",
    standardize_output: bool = True,
    normalize: Literal["l1", "sum", "none"] = "l1",
    missing: _MissingPolicy = "renormalize",
    min_signals: int = 1,
    min_obs: int = 20,
    on_insufficient: InsufficientPolicy = "nan",
) -> pd.Series:
    """Combinación lineal con pesos fijos, declarados por el usuario.

    Es la línea base contra la que hay que comparar cualquier esquema adaptativo:
    con factores de IC parecido y correlación moderada, los pesos iguales son
    difíciles de batir fuera de muestra, porque no gastan grados de libertad en
    estimar nada.

    Parámetros
    ----------
    weights:
        Mapa nombre → peso, `Series` indexada por nombre de señal, o secuencia
        del mismo largo que las columnas. `None` = pesos iguales.
    missing:
        Qué hacer cuando a un nombre le falta alguna señal ese día:
        - ``"renormalize"`` (por defecto): se reparte el peso entre las señales
          presentes, manteniendo la escala del compuesto.
        - ``"skip"``: la señal ausente aporta 0, así que el compuesto se encoge
          hacia el centro. Es defendible —menos información, menos convicción—
          pero introduce una dependencia entre cobertura y tamaño de la apuesta.
        - ``"nan"``: sin todas las señales no hay puntuación.
    """
    frame = _as_signal_frame(signals)
    cols = [str(c) for c in frame.columns]

    if weights is None:
        w = pd.Series(1.0, index=cols)
    elif isinstance(weights, pd.Series):
        w = weights.astype("float64")
        w.index = pd.Index([str(i) for i in w.index])
    elif isinstance(weights, Mapping):
        w = pd.Series({str(kk): float(vv) for kk, vv in weights.items()})
    else:
        arr = np.asarray(weights, dtype="float64")
        if arr.shape != (len(cols),):
            msg = f"`weights` como secuencia debe tener {len(cols)} elementos; tiene {arr.shape}"
            raise DataQualityError(msg)
        w = pd.Series(arr, index=cols)

    missing_cols = [c for c in cols if c not in w.index]
    if missing_cols:
        msg = f"faltan pesos para las señales {missing_cols}; declara uno por señal"
        raise DataQualityError(msg)
    extra = [str(i) for i in w.index if str(i) not in cols]
    if extra:
        msg = f"`weights` declara señales inexistentes en el panel: {extra}"
        raise DataQualityError(msg)
    w = w.reindex(cols)
    if not np.isfinite(w.to_numpy()).all():
        msg = "hay pesos no finitos (NaN o infinito)"
        raise DataQualityError(msg)
    if np.allclose(w.to_numpy(), 0.0):
        msg = "todos los pesos son cero: la combinación no tendría contenido"
        raise DataQualityError(msg)

    normalized = _normalize_weights(w.to_numpy()[None, :], normalize)[0]
    std = _standardize_panel(frame, standardize, min_obs=min_obs, on_insufficient=on_insufficient)
    dates = _panel_dates(frame)
    weight_frame = pd.DataFrame(
        np.repeat(normalized[None, :], len(dates), axis=0), index=dates, columns=cols
    )
    score = _apply_weights(std, weight_frame, missing=missing, min_signals=min_signals)
    if standardize_output:
        score = cast(pd.Series, zscore(score, min_obs=min_obs, on_insufficient=on_insufficient))
    score.name = "score"
    return score


# ---------------------------------------------------------------------------
# Ortogonalización
# ---------------------------------------------------------------------------


def orthogonalize(
    signals: pd.DataFrame,
    order: Sequence[str] | None = None,
    *,
    method: Literal["gram_schmidt", "symmetric"] = "gram_schmidt",
    standardize: bool = True,
    min_obs: int = 20,
    min_dof: int = 5,
    on_insufficient: InsufficientPolicy = "nan",
) -> pd.DataFrame:
    """Ortogonaliza las señales dentro de cada sección cruzada.

    **Gram-Schmidt secuencial** (por defecto): el primer factor de `order` se
    conserva íntegro y cada factor siguiente se sustituye por el residuo de su
    regresión cross-section sobre los ya ortogonalizados. El resultado depende
    del orden, y eso es una característica, no un defecto: el orden expresa la
    prelación económica. La convención del repo es poner primero el factor con
    más respaldo académico y dejar para el final el sospechoso de redundancia
    —`SUE_price` antes que `E/P`, `pre_event_car` antes que las features de
    opciones (`informed_trading.md` §12.2)— de modo que lo que sobrevive al
    final es incremento puro. Si una señal pierde todo su IC tras la
    ortogonalización, no aportaba nada.

    **Simétrica** (`method="symmetric"`, Löwdin 1950): ``Y = X·S^{-1/2}``, la
    rotación ortogonal más próxima a `X` en norma de Frobenius. No privilegia
    ningún orden y reparte la parte común entre todos los factores, pero exige
    sección cruzada completa (sin NaN) y matriz de correlación no singular.

    Devuelve
    --------
    `DataFrame` con las mismas columnas (en el orden pedido) y el mismo índice.
    Con `standardize=True` cada columna sale con media 0 y desviación 1 por
    fecha, lista para combinar linealmente.
    """
    frame = _as_signal_frame(signals)
    cols = list(frame.columns)
    seq = cols if order is None else [str(c) for c in order]
    unknown = [c for c in seq if c not in cols]
    if unknown:
        msg = f"`order` menciona señales inexistentes: {unknown}"
        raise DataQualityError(msg)
    if len(set(seq)) != len(seq):
        msg = f"`order` repite señales: {seq}"
        raise DataQualityError(msg)

    if method == "symmetric":
        return symmetric_orthogonalize(
            frame[seq], min_obs=min_obs, on_insufficient=on_insufficient
        )
    if method != "gram_schmidt":  # pragma: no cover - protegido por el tipo
        msg = f"método de ortogonalización desconocido: {method!r}"
        raise ValueError(msg)

    done = pd.DataFrame(index=frame.index)
    out: dict[str, pd.Series] = {}
    for i, col in enumerate(seq):
        current = frame[col]
        if standardize:
            current = cast(
                pd.Series,
                zscore(current, min_obs=min_obs, on_insufficient=on_insufficient),
            )
        if i > 0:
            current = residualize(
                current,
                done,
                add_intercept=True,
                min_obs=min_obs,
                min_dof=min_dof,
                on_insufficient=on_insufficient,
                nan_policy=NaNPolicy.PROPAGATE,
            )
            if standardize:
                current = cast(
                    pd.Series,
                    zscore(current, min_obs=min_obs, on_insufficient=on_insufficient),
                )
        current.name = col
        out[col] = current
        done[col] = current
    return pd.DataFrame(out, index=frame.index)[seq]


def symmetric_orthogonalize(
    signals: pd.DataFrame,
    *,
    min_obs: int = 20,
    on_insufficient: InsufficientPolicy = "nan",
    ridge: float = 1e-10,
) -> pd.DataFrame:
    """Ortogonalización simétrica de Löwdin, fecha a fecha: ``Y = X · S^{-1/2}``.

    `S` es la matriz de correlación cross-section entre señales de esa fecha.
    `S^{-1/2}` se calcula por descomposición espectral. La transformación es la
    ortogonalización que **menos** deforma el conjunto original (minimiza
    ``‖Y − X‖_F`` entre todas las bases ortonormales del mismo espacio), lo que
    evita tener que justificar un orden de prelación arbitrario
    (Löwdin 1950; Klein y Chow 2013).

    Exige observaciones completas en todas las señales de la fecha: un nombre al
    que le falte un factor no puede rotarse. Esas filas salen `NaN`.
    """
    frame = _as_signal_frame(signals)
    cols = list(frame.columns)
    k = frame.shape[1]
    values = frame.to_numpy(dtype="float64")
    complete = np.isfinite(values).all(axis=1)

    out = np.full_like(values, np.nan)
    day = frame.index.get_level_values(0)
    codes, uniques = pd.factorize(day, sort=True)
    order = np.argsort(codes, kind="stable")
    offsets = np.searchsorted(codes[order], np.arange(len(uniques) + 1))
    n_ok = 0
    for start, stop in zip(offsets[:-1], offsets[1:], strict=True):
        rows = order[start:stop]
        sel = rows[complete[rows]]
        if len(sel) < max(min_obs, k + 2):
            continue
        block = values[sel]
        centered = block - block.mean(axis=0, keepdims=True)
        sd = centered.std(axis=0, ddof=1)
        if not np.all(sd > 0):
            continue
        scaled = centered / sd
        corr = scaled.T @ scaled / (len(sel) - 1)
        eigval, eigvec = np.linalg.eigh(corr)
        if np.min(eigval) <= ridge:
            continue
        inv_sqrt = eigvec @ np.diag(eigval**-0.5) @ eigvec.T
        out[sel] = scaled @ inv_sqrt
        n_ok += 1
    if n_ok == 0:
        msg = (
            "ninguna fecha admite ortogonalización simétrica: hacen falta al menos "
            f"max(min_obs={min_obs}, k+2={k + 2}) observaciones completas y una matriz "
            "de correlación definida positiva"
        )
        raise InsufficientHistory(msg)
    if on_insufficient == "raise" and n_ok < len(uniques):
        msg = f"{len(uniques) - n_ok} fechas sin ortogonalización simétrica posible"
        raise InsufficientHistory(msg)
    return pd.DataFrame(out, index=frame.index, columns=pd.Index(cols))


# ---------------------------------------------------------------------------
# Despachador del contrato
# ---------------------------------------------------------------------------


def combine(
    signals: pd.Series | pd.DataFrame,
    method: Literal["weights", "ic_weighted", "orthogonalize"] = "weights",
    **kwargs: object,
) -> pd.Series:
    """Despachador de combinación del contrato §3.6.

    - ``"weights"`` → `fixed_weight_combine`.
    - ``"ic_weighted"`` → `ic_weighted_combine` (exige `forward_returns`).
    - ``"orthogonalize"`` → `orthogonalize` seguido de una combinación de pesos
      fijos sobre los factores ya ortogonalizados. Ese segundo paso es
      necesario: la ortogonalización produce un panel de factores, no una
      puntuación, y sumarlos sin más volvería a mezclar lo que se acaba de
      separar si no se declara con qué pesos.
    """
    if method == "weights":
        return fixed_weight_combine(signals, **kwargs)  # type: ignore[arg-type]
    if method == "ic_weighted":
        if "forward_returns" not in kwargs:
            msg = "method='ic_weighted' requiere el argumento `forward_returns`"
            raise DataQualityError(msg)
        forward = cast(pd.Series, kwargs.pop("forward_returns"))
        return ic_weighted_combine(signals, forward, **kwargs)  # type: ignore[arg-type]
    if method == "orthogonalize":
        frame = _as_signal_frame(signals)
        ortho_keys = ("order", "min_dof")
        ortho_kwargs = {k: kwargs.pop(k) for k in ortho_keys if k in kwargs}
        shared = {k: kwargs[k] for k in ("min_obs", "on_insufficient") if k in kwargs}
        ortho = orthogonalize(frame, **ortho_kwargs, **shared)  # type: ignore[arg-type]
        kwargs.setdefault("standardize", "none")
        return fixed_weight_combine(ortho, **kwargs)  # type: ignore[arg-type]
    msg = (
        f"método de combinación desconocido: {method!r}. "
        "Válidos: 'weights', 'ic_weighted', 'orthogonalize'"
    )
    raise ValueError(msg)
