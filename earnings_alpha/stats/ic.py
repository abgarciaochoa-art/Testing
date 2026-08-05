"""Coeficiente de información (IC) y su significancia estadística.

Implementa la sección 2 y 3 de `docs/research/validation_methodology.md`, que es a su
vez la especificación de `docs/ARCHITECTURE.md` §3.8.

Contenido:

- IC de Pearson y **rank IC** de Spearman por fecha sobre el corte transversal
  (`cross_sectional_ic`), con mínimo de nombres obligatorio.
- Resumen de la serie temporal de IC: media, `IC_IR`, `t` ingenuo y `t` de
  Newey–West con **selección automática de lags** (`summarize_ic`).
- Error estándar de Newey–West con kernel de Bartlett (`long_run_variance`,
  `mean_significance`) y muestreo disjunto como contraste exacto
  (`disjoint_significance`).
- IC por sector y por tramo de capitalización (`ic_by_group`, `ic_by_size_bucket`).
- Decaimiento del IC por horizonte (`ic_decay`), con retornos *forward* a 1, 5, 10,
  21 y 63 sesiones.

Principio que gobierna el módulo: **ningún `t` se emite sin declarar qué corrección
de dependencia lleva aplicada** (§1.3 del informe). Por eso todo contraste devuelve
un `SignificanceResult` con el campo `method` obligatorio y el número de
observaciones *efectivas* tras corregir la dependencia.

Referencias
-----------
- Newey, W. K., y West, K. D. (1987). *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*.
  **Econometrica** 55(3), 703-708.
- Newey, W. K., y West, K. D. (1994). *Automatic Lag Selection in Covariance Matrix
  Estimation*. **Review of Economic Studies** 61(4), 631-653.
- Bonett, D. G., y Wright, T. A. (2000). *Sample size requirements for estimating
  Pearson, Kendall and Spearman correlations*. **Psychometrika** 65, 23-28.
- Grinold, R. C. (1989). *The Fundamental Law of Active Management*. **JPM** 15(3).
- Kolari, J. W., y Pynnönen, S. (2010). *Event Study Testing with Cross-sectional
  Correlation of Abnormal Returns*. **RFS** 23(11), 3996-4025.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from earnings_alpha.errors import DataQualityError, InsufficientHistory

__all__ = [
    "DEFAULT_HORIZONS",
    "DEFAULT_MIN_NAMES",
    "TRADING_DAYS_PER_YEAR",
    "CorrelationKind",
    "ICSummary",
    "SignificanceResult",
    "cross_sectional_ic",
    "disjoint_significance",
    "effective_n",
    "fisher_ci",
    "fisher_se",
    "forward_returns",
    "ic_by_group",
    "ic_by_size_bucket",
    "ic_decay",
    "long_run_variance",
    "mean_significance",
    "newey_west_lags",
    "restrict_to_universe",
    "screen_ic_criteria",
    "summarize_ic",
]

TRADING_DAYS_PER_YEAR = 252
"""Sesiones bursátiles por año usadas para anualizar. Constante del repo."""

DEFAULT_MIN_NAMES = 30
"""Mínimo de activos en la sección cruzada para que `IC_t` sea un número.

Por debajo de 30 el estimador de correlación tiene un sesgo y una varianza que
invalidan todo lo que se construya encima (§2.2 del informe de validación)."""

DEFAULT_HORIZONS: tuple[int, ...] = (1, 5, 10, 21, 63)
"""Horizontes de decaimiento exigidos por la Etapa 1 del protocolo (§13)."""

CorrelationKind = Literal["pearson", "spearman"]


# --------------------------------------------------------------------------- #
# Resultado de un contraste                                                    #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class SignificanceResult:
    """Resultado de un contraste con su corrección de dependencia declarada.

    Sigue el contrato propuesto en §1.3 del informe de validación. El campo
    `method` no es decorativo: un `t` sin etiqueta de corrección es un `t` inválido,
    porque en este panel la dependencia temporal y transversal infla el estadístico
    por factores de 4 a 8.

    `n_eff` es el número de observaciones *independientes equivalentes*: para
    Newey–West se define como `T · γ̂₀ / Ŝ_NW`, es decir, cuántas observaciones iid
    darían la misma varianza de la media. Si la serie no tiene autocorrelación,
    `n_eff ≈ n_obs`; con solapamiento de horizonte `h`, `n_eff ≈ n_obs/h`.
    """

    estimate: float
    std_error: float
    t_stat: float
    p_value: float
    ci_low: float
    ci_high: float
    n_obs: int
    n_eff: float
    method: str
    nw_lags: int | None = None
    alpha: float = 0.05
    notes: str = ""

    @property
    def significant(self) -> bool:
        """True si el intervalo de confianza excluye el cero."""
        return self.ci_low > 0.0 or self.ci_high < 0.0

    def to_dict(self) -> dict[str, object]:
        """Vista plana, apta para construir un DataFrame de resultados."""
        return {
            "estimate": self.estimate,
            "std_error": self.std_error,
            "t_stat": self.t_stat,
            "p_value": self.p_value,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n_obs": self.n_obs,
            "n_eff": self.n_eff,
            "method": self.method,
            "nw_lags": self.nw_lags,
        }


# --------------------------------------------------------------------------- #
# Newey–West                                                                    #
# --------------------------------------------------------------------------- #


def newey_west_lags(
    n_obs: int,
    horizon: int = 1,
    *,
    rule: Literal["stock_watson", "newey_west_1994"] = "stock_watson",
) -> int:
    """Número de lags del kernel de Bartlett, con suelo por solapamiento.

    Regla vinculante del repo (§3.2 del informe de validación)::

        L = max( ⌊4·(T/100)^(2/9)⌋ , h − 1 )

    El término automático cubre la autocorrelación económica residual; el suelo
    `h−1` cubre la media móvil **determinista** que induce usar retornos *forward*
    de `h` periodos muestreados a cada periodo. Ninguna de las dos reglas estándar
    conoce `h`, y esa es exactamente la razón por la que hace falta el suelo: con
    30 años de datos diarios la regla automática pide `L = 10`, pero una IC a 63
    días arrastra una MA(62).

    Parameters
    ----------
    n_obs:
        Longitud `T` de la serie sobre la que se promedia.
    horizon:
        Horizonte `h` del retorno *forward*, en periodos de muestreo. `h = 1`
        significa que no hay solapamiento.
    rule:
        `"stock_watson"` usa el exponente 2/9 (práctica dominante);
        `"newey_west_1994"` usa 2/25 como semilla de la regla automática original.

    Returns
    -------
    int
        Número de lags, siempre `0 ≤ L ≤ n_obs − 1`.

    Examples
    --------
    >>> newey_west_lags(1260, horizon=1)
    7
    >>> newey_west_lags(1260, horizon=21)
    20
    """
    if n_obs < 2:
        msg = f"se necesitan al menos 2 observaciones para estimar lags; hay {n_obs}"
        raise InsufficientHistory(msg)
    if horizon < 1:
        msg = f"el horizonte debe ser >= 1; recibido {horizon}"
        raise ValueError(msg)
    exponent = 2.0 / 9.0 if rule == "stock_watson" else 2.0 / 25.0
    automatic = int(math.floor(4.0 * (n_obs / 100.0) ** exponent))
    lags = max(automatic, horizon - 1)
    return int(min(max(lags, 0), n_obs - 1))


def long_run_variance(x: np.ndarray | pd.Series, lags: int) -> float:
    """Varianza de largo plazo `Ŝ_NW` con kernel de Bartlett.

    ``γ̂_k = (1/T)·Σ_{t=k+1}^{T} d_t·d_{t−k}``  con ``d_t = x_t − x̄`` y

    ``Ŝ_NW = γ̂_0 + 2·Σ_{k=1}^{L} (1 − k/(L+1))·γ̂_k``.

    Los pesos de Bartlett no son cosméticos: garantizan `Ŝ_NW ≥ 0`. La suma sin
    ponderar (Hansen–Hodrick) puede dar varianzas negativas con `L` grande y `T`
    moderado, y en ese caso no hay error estándar que reportar.
    """
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 2:
        msg = f"la serie tiene {n} observaciones finitas; se necesitan al menos 2"
        raise InsufficientHistory(msg)
    lags = int(min(max(lags, 0), n - 1))
    d = arr - arr.mean()
    gamma0 = float(d @ d) / n
    total = gamma0
    for k in range(1, lags + 1):
        gamma_k = float(d[k:] @ d[:-k]) / n
        weight = 1.0 - k / (lags + 1.0)
        total += 2.0 * weight * gamma_k
    # Bartlett garantiza no negatividad; el clip protege solo del error de redondeo.
    return float(max(total, 0.0))


def effective_n(n: int | float, rho_bar: float) -> float:
    """Observaciones efectivas bajo correlación transversal media `ρ̄`.

    ``n_eff = n / [1 + (n−1)·ρ̄]`` (§1.2(a) del informe; es la misma álgebra que
    Kolari–Pynnönen 2010).

    Es *el* número que gobierna cualquier afirmación de significancia del ángulo B:
    con 473 eventos por trimestre y `ρ̄ = 0,05`, las 473 observaciones valen lo que
    19 independientes, y el `t` ingenuo se infla ×5,1.

    Examples
    --------
    >>> round(effective_n(500, 0.05), 1)
    19.3
    """
    n = float(n)
    if n <= 0:
        msg = f"n debe ser positivo; recibido {n}"
        raise ValueError(msg)
    denom = 1.0 + (n - 1.0) * float(rho_bar)
    if denom <= 0.0:
        msg = (
            f"1 + (n−1)·ρ̄ = {denom:.4f} no es positivo con n={n} y ρ̄={rho_bar}: "
            "una correlación media tan negativa es incompatible con una matriz "
            "de correlaciones válida"
        )
        raise DataQualityError(msg)
    return float(n / denom)


def mean_significance(
    x: np.ndarray | pd.Series,
    *,
    horizon: int = 1,
    lags: int | None = None,
    alpha: float = 0.05,
    label: str = "",
) -> SignificanceResult:
    """Contraste `H₀: E[x] = 0` con error estándar de Newey–West.

    `SE_NW(x̄) = √(Ŝ_NW / T)` y `t_NW = x̄ / SE_NW(x̄)`, con `L` elegido por
    `newey_west_lags(T, horizon)` salvo que se fije explícitamente.

    **Advertencia calibrada (§3.3).** Newey–West con `L ≥ h−1` reduce la inflación
    del `t` de ×7,98 a ×1,25 con `h = 63`, pero el residuo *se estanca*: la tasa de
    rechazo real se estabiliza en ≈ 11,7 % frente al 5 % nominal para todo `h ≥ 10`.
    Un `t_NW = 2,0` sobre datos solapados equivale a un `t` real de ≈ 1,6. Por eso
    con `h > 5` hay que contrastar además con `disjoint_significance`, y por eso los
    umbrales de §14 son altos.
    """
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 3:
        msg = f"solo {n} observaciones finitas: no hay serie que contrastar"
        raise InsufficientHistory(msg)
    if not 0.0 < alpha < 1.0:
        msg = f"alpha debe estar en (0,1); recibido {alpha}"
        raise ValueError(msg)

    lags_used = newey_west_lags(n, horizon) if lags is None else int(lags)
    s_nw = long_run_variance(arr, lags_used)
    mean = float(arr.mean())
    gamma0 = float(((arr - mean) ** 2).mean())

    if s_nw <= 0.0:
        msg = (
            "la varianza de largo plazo estimada es cero: la serie es constante y "
            "no admite contraste"
        )
        raise DataQualityError(msg)

    se = math.sqrt(s_nw / n)
    t_stat = mean / se
    p_value = float(2.0 * stats.norm.sf(abs(t_stat)))
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    n_eff = n * gamma0 / s_nw if s_nw > 0 else float(n)

    notes = label
    if horizon > 5:
        notes = (
            f"{label} | h={horizon}>5: el t_NW sobre datos solapados conserva un "
            "sesgo residual de ~25 % (§3.3); contrastar con muestreo disjunto"
        ).strip(" |")

    return SignificanceResult(
        estimate=mean,
        std_error=se,
        t_stat=t_stat,
        p_value=p_value,
        ci_low=mean - z * se,
        ci_high=mean + z * se,
        n_obs=n,
        n_eff=float(n_eff),
        method="newey_west",
        nw_lags=lags_used,
        alpha=alpha,
        notes=notes,
    )


def disjoint_significance(
    x: np.ndarray | pd.Series,
    horizon: int,
    *,
    alpha: float = 0.05,
    phase: int | Literal["all"] = "all",
) -> SignificanceResult:
    """Contraste sobre submuestras **disjuntas**, exacto bajo solapamiento.

    Tomar una observación de cada `h` elimina por completo el solapamiento de los
    retornos *forward*: la simulación de §3.3 mide una tasa de rechazo de 4,5–5,4 %
    frente al 5 % nominal para todo `h ∈ {1,…,63}`, mientras que Newey–West se queda
    en ≈ 11,7 %. El precio es reducir la muestra a `T/h` observaciones.

    Con `phase="all"` se promedian los `h` desfases posibles (el resultado no
    depende entonces de qué día se empezó a muestrear) y se reporta en `notes` el
    rango entre desfases: si es amplio, la conclusión es frágil.

    **Regla del repo:** con `h > 5` se reportan Newey–West y disjunto; en caso de
    discrepancia manda el disjunto.
    """
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if horizon < 1:
        msg = f"el horizonte debe ser >= 1; recibido {horizon}"
        raise ValueError(msg)
    if n < 3 * horizon:
        msg = (
            f"el muestreo disjunto con h={horizon} dejaría {n // horizon} "
            f"observaciones de {n}: insuficiente para un contraste"
        )
        raise InsufficientHistory(msg)

    phases = range(horizon) if phase == "all" else [int(phase) % horizon]
    estimates: list[float] = []
    ses: list[float] = []
    ts: list[float] = []
    sizes: list[int] = []
    for p in phases:
        sub = arr[p::horizon]
        if sub.size < 3:
            continue
        m = float(sub.mean())
        se = float(sub.std(ddof=1) / math.sqrt(sub.size))
        if se <= 0.0:
            continue
        estimates.append(m)
        ses.append(se)
        ts.append(m / se)
        sizes.append(int(sub.size))
    if not ts:
        msg = "ningún desfase produjo una submuestra utilizable"
        raise InsufficientHistory(msg)

    estimate = float(np.mean(estimates))
    # El error estándar combinado se toma como la media de los SE por desfase: cada
    # submuestra es un contraste válido por separado y promediarlos no reduce la
    # varianza (comparten observaciones a través de los desfases).
    se = float(np.mean(ses))
    t_stat = float(np.mean(ts))
    p_value = float(2.0 * stats.norm.sf(abs(t_stat)))
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    spread = f"t por desfase en [{min(ts):.2f}, {max(ts):.2f}]" if len(ts) > 1 else ""

    return SignificanceResult(
        estimate=estimate,
        std_error=se,
        t_stat=t_stat,
        p_value=p_value,
        ci_low=estimate - z * se,
        ci_high=estimate + z * se,
        n_obs=int(np.mean(sizes)),
        n_eff=float(np.mean(sizes)),
        method="disjoint_sampling",
        nw_lags=None,
        alpha=alpha,
        notes=f"h={horizon}, {len(ts)} desfases; {spread}".strip("; "),
    )


# --------------------------------------------------------------------------- #
# Error estándar de una IC de un solo día (transformación de Fisher)            #
# --------------------------------------------------------------------------- #


def fisher_se(n_names: int, *, kind: CorrelationKind = "spearman", rho: float = 0.0) -> float:
    """Error estándar de `z = arctanh(IC)` para una sola fecha.

    - Pearson: `SE(z) = 1/√(n−3)`, exacto asintóticamente bajo normalidad bivariante.
    - Spearman: `SE(z) = √[(1 + ρ̂²/2)/(n−3)]` (Bonett–Wright 2000).

    Con `n = 475` (el universo típico del repo) el `SE(z)` es ≈ 0,046: **una IC
    diaria individual necesitaría ser ≈ 0,09 para ser significativa**, y las IC de
    factores viables están en 0,01–0,05. Corolario: *la IC de un día no significa
    nada nunca*; toda la evidencia está en la serie temporal.
    """
    if n_names <= 3:
        msg = f"se necesitan al menos 4 activos para el SE de Fisher; hay {n_names}"
        raise InsufficientHistory(msg)
    base = 1.0 / math.sqrt(n_names - 3)
    if kind == "pearson":
        return base
    return math.sqrt((1.0 + rho * rho / 2.0) / (n_names - 3))


def fisher_ci(
    ic: float,
    n_names: int,
    *,
    kind: CorrelationKind = "spearman",
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Intervalo de confianza de una IC de una sola fecha: `tanh(z ± z_{α/2}·SE(z))`."""
    if not -1.0 < ic < 1.0:
        msg = f"la IC debe estar en (-1,1) para transformarla con arctanh; es {ic}"
        raise DataQualityError(msg)
    se = fisher_se(n_names, kind=kind, rho=ic)
    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    center = math.atanh(ic)
    return float(math.tanh(center - z_crit * se)), float(math.tanh(center + z_crit * se))


# --------------------------------------------------------------------------- #
# Preparación de paneles                                                        #
# --------------------------------------------------------------------------- #


def _as_panel_series(obj: pd.Series | pd.DataFrame, name: str, column: str | None = None) -> pd.Series:
    """Normaliza a Series con MultiIndex `(date, ticker)` ordenado."""
    if isinstance(obj, pd.DataFrame):
        if column is not None:
            if column not in obj.columns:
                msg = f"la columna {column!r} no está en el panel {name!r}: {list(obj.columns)}"
                raise DataQualityError(msg)
            obj = obj[column]
        elif obj.shape[1] == 1:
            obj = obj.iloc[:, 0]
        else:
            msg = (
                f"el panel {name!r} tiene {obj.shape[1]} columnas; indica cuál usar "
                "o pasa una Series"
            )
            raise DataQualityError(msg)
    if not isinstance(obj.index, pd.MultiIndex) or obj.index.nlevels != 2:
        msg = (
            f"{name!r} debe tener MultiIndex (date, ticker) de 2 niveles, como exige "
            "el panel canónico de ARCHITECTURE.md §1"
        )
        raise DataQualityError(msg)
    out = obj.copy()
    out.index = out.index.set_names(["date", "ticker"])
    dates = pd.to_datetime(out.index.get_level_values("date")).normalize()
    out.index = pd.MultiIndex.from_arrays(
        [dates, out.index.get_level_values("ticker")], names=["date", "ticker"]
    )
    return out.sort_index()


def restrict_to_universe(panel: pd.Series, membership: pd.DataFrame) -> pd.Series:
    """Filtra un panel `(date, ticker)` a la pertenencia PIT al índice.

    `membership` es el panel booleano `index=date, columns=ticker` que devuelve
    `universe.SP500Universe.membership_panel`. Sin este filtro la IC está sesgada al
    alza por supervivencia (§2.2), que es el error nº 13 de la lista de §15.
    """
    panel = _as_panel_series(panel, "panel")
    flags = membership.copy()
    flags.index = pd.to_datetime(flags.index).normalize()
    # `membership_panel` devuelve por defecto un panel DISPERSO (una fila por
    # snapshot de composición). Casar por igualdad exacta de fechas descartaría
    # en silencio toda sesión del panel sin snapshot ese día: se forward-fillea
    # la pertenencia sobre las fechas del panel (misma semántica PIT que
    # `backtest.engine._membership_matrix`: vale la última composición conocida).
    panel_dates = pd.DatetimeIndex(panel.index.get_level_values("date").unique()).sort_values()
    flags = (
        flags.reindex(flags.index.union(panel_dates))
        .ffill()
        .reindex(panel_dates)
        .fillna(False)
        .astype(bool)
    )
    stacked = flags.stack()
    stacked.index = stacked.index.set_names(["date", "ticker"])
    keep = stacked.reindex(panel.index).fillna(False).astype(bool)
    out = panel[keep.to_numpy()]
    if out.empty:
        msg = (
            "el filtro de universo ha dejado el panel vacío: revisa que las fechas y "
            "los símbolos de `membership` estén normalizados igual que los del panel"
        )
        raise InsufficientHistory(msg)
    return out


def forward_returns(
    prices: pd.DataFrame | pd.Series,
    horizon: int = 1,
    *,
    price_col: str = "adj_close",
    execution_lag: int = 1,
    log: bool = False,
) -> pd.Series:
    """Retorno *forward* de `horizon` sesiones, con retardo de ejecución explícito.

    La señal conocible en `t` se ejecuta como pronto al cierre de `t + execution_lag`
    (Etapa 0 del protocolo: *retardo de ejecución ≥ 1 sesión entre señal y precio*).
    El retorno etiquetado en `t` es por tanto::

        r_t = P(t + lag + h) / P(t + lag) − 1

    Ejecutar al cierre de `t` con datos de `t` es look-ahead, y por eso el valor por
    defecto de `execution_lag` es 1 y no 0.

    Parameters
    ----------
    prices:
        Panel `(date, ticker)` con la columna de precios, o Series ya extraída.
    horizon:
        Número de sesiones del retorno. El eje temporal es el del propio panel, que
        se asume ya restringido a sesiones de mercado.
    price_col:
        Columna de precio. Debe ser **ajustada por splits y dividendos**; usar el
        precio sin ajustar introduce saltos espurios en cada dividendo.
    execution_lag:
        Sesiones entre la fecha de la señal y la de entrada.
    log:
        Si es True devuelve el retorno logarítmico.
    """
    if horizon < 1:
        msg = f"el horizonte debe ser >= 1; recibido {horizon}"
        raise ValueError(msg)
    if execution_lag < 0:
        msg = f"execution_lag no puede ser negativo; recibido {execution_lag}"
        raise ValueError(msg)
    px = _as_panel_series(prices, "prices", column=price_col if isinstance(prices, pd.DataFrame) else None)
    wide = px.unstack("ticker").sort_index()
    if len(wide) <= execution_lag + horizon:
        msg = (
            f"el panel tiene {len(wide)} fechas y se piden {execution_lag + horizon} "
            "hacia adelante: no hay retorno futuro que calcular"
        )
        raise InsufficientHistory(msg)
    entry = wide.shift(-execution_lag)
    exit_ = wide.shift(-(execution_lag + horizon))
    ratio = exit_ / entry.where(entry > 0)
    out = np.log(ratio) if log else ratio - 1.0
    stacked = out.stack()
    stacked.index = stacked.index.set_names(["date", "ticker"])
    stacked = stacked.dropna().sort_index()
    if stacked.empty:
        msg = "no ha quedado ningún retorno forward finito tras alinear el panel"
        raise InsufficientHistory(msg)
    return stacked.rename(f"fwd_{horizon}")


def _aligned(scores: pd.Series | pd.DataFrame, returns: pd.Series | pd.DataFrame) -> pd.DataFrame:
    """Une señal y retorno por `(date, ticker)` y descarta lo incompleto."""
    s = _as_panel_series(scores, "scores")
    r = _as_panel_series(returns, "returns")
    df = pd.DataFrame({"s": s, "r": r.reindex(s.index)})
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if df.empty:
        msg = (
            "señal y retorno no comparten ninguna observación (date, ticker): "
            "revisa la normalización de símbolos y de fechas antes de calcular IC"
        )
        raise InsufficientHistory(msg)
    return df


# --------------------------------------------------------------------------- #
# IC transversal                                                                #
# --------------------------------------------------------------------------- #


def _ic_from_frame(
    df: pd.DataFrame,
    *,
    method: CorrelationKind,
    min_names: int,
) -> pd.DataFrame:
    """Correlación transversal por fecha, vectorizada mediante sumas agrupadas.

    Calcular la correlación con sumas agrupadas en vez de con un `apply` por fecha
    es lo que hace viable recorrer 7.500 fechas × 475 activos sin salirse del
    presupuesto de tiempo de un cribado.
    """
    work = df[["s", "r"]].copy()
    by_date = work.groupby(level="date", observed=True, sort=True)
    if method == "spearman":
        work["s"] = by_date["s"].rank()
        work["r"] = by_date["r"].rank()
        by_date = work.groupby(level="date", observed=True, sort=True)

    work["ss"] = work["s"] * work["s"]
    work["rr"] = work["r"] * work["r"]
    work["sr"] = work["s"] * work["r"]
    grouped = work.groupby(level="date", observed=True, sort=True)
    sums = grouped[["s", "r", "ss", "rr", "sr"]].sum()
    n = grouped.size().astype(float)

    mean_s = sums["s"] / n
    mean_r = sums["r"] / n
    cov = sums["sr"] / n - mean_s * mean_r
    var_s = (sums["ss"] / n - mean_s**2).clip(lower=0.0)
    var_r = (sums["rr"] / n - mean_r**2).clip(lower=0.0)
    denom = np.sqrt(var_s * var_r)
    ic = pd.Series(
        np.where(denom > 0.0, cov / denom.where(denom > 0.0), np.nan),
        index=sums.index,
        dtype=float,
    )
    ic = ic.where(n >= min_names)
    out = pd.DataFrame({"ic": ic, "n_names": n})
    out.index = out.index.set_names("date")
    return out


def cross_sectional_ic(
    scores: pd.Series | pd.DataFrame,
    returns: pd.Series | pd.DataFrame,
    *,
    method: CorrelationKind = "spearman",
    min_names: int = DEFAULT_MIN_NAMES,
    membership: pd.DataFrame | None = None,
    with_counts: bool = False,
) -> pd.Series | pd.DataFrame:
    """Serie temporal de IC transversal, una observación por fecha.

    Fórmula (§2.1): correlación de Pearson entre señal y retorno *forward* dentro
    del corte transversal de cada fecha; **rank IC** si `method="spearman"`,
    idéntica fórmula sobre los rangos con corrección de empates por rango medio.

    La métrica principal del repo es el **rank IC**, por tres razones concretas:
    los factores fundamentales tienen colas patológicas (un *earnings yield* con
    beneficio cercano a cero genera |z| > 20 y la IC de Pearson acaba midiendo tres
    observaciones), el retorno tiene curtosis alta, y la IC de Pearson no es
    invariante a las transformaciones monótonas que aplica `signals`. La de Pearson
    se reporta como diagnóstico secundario: si `IC_pearson >> IC_rank`, el factor
    vive en las colas y su implementabilidad es dudosa.

    Si en una fecha hay menos de `min_names` activos, `IC_t` es `NaN`, **no un
    número**: con `n` pequeño el estimador tiene un sesgo que rompe todo lo
    posterior.

    Parameters
    ----------
    scores:
        Señal conocible en `t`, MultiIndex `(date, ticker)`. Debe venir ya
        desplazada por `pit.asof_join`; este módulo no puede verificarlo.
    returns:
        Retorno *forward*, típicamente de `forward_returns`.
    membership:
        Panel booleano de pertenencia PIT al índice. Si se pasa, se filtra por él.
    with_counts:
        Si es True devuelve un DataFrame con columnas `ic` y `n_names`.

    Raises
    ------
    InsufficientHistory
        Si ninguna fecha alcanza `min_names`. Devolver una serie de NaN en silencio
        sería peor que fallar.
    """
    if method not in ("pearson", "spearman"):
        msg = f"method debe ser 'pearson' o 'spearman'; recibido {method!r}"
        raise ValueError(msg)
    df = _aligned(scores, returns)
    if membership is not None:
        kept = restrict_to_universe(df["s"], membership)
        df = df.loc[kept.index]
    out = _ic_from_frame(df, method=method, min_names=min_names)
    if out["ic"].notna().sum() == 0:
        msg = (
            f"ninguna de las {len(out)} fechas alcanza el mínimo de {min_names} "
            f"activos (máximo observado: {int(out['n_names'].max())}). Baja "
            "`min_names` conscientemente o amplía el universo"
        )
        raise InsufficientHistory(msg)
    out.index.name = "date"
    if with_counts:
        return out
    return out["ic"].rename(f"ic_{method}")


# --------------------------------------------------------------------------- #
# Resumen de la serie de IC                                                     #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ICSummary:
    """Resumen de una serie temporal de IC con su banda de error.

    `ic_ir` es `ĪC / σ_IC` por periodo; `ic_ir_annualized` lo multiplica por
    `√(periodos por año)`. El `t` ingenuo `IC_IR·√T` se reporta **solo** para que se
    vea cuánto infla: con `h = 21` su tasa de rechazo real al nominal 5 % es del
    67,9 % (§3.3). El estadístico que manda es `significance.t_stat`.
    """

    mean: float
    median: float
    std: float
    ic_ir: float
    ic_ir_annualized: float
    t_naive: float
    hit_rate: float
    monthly_positive_fraction: float
    n_periods: int
    horizon: int
    method: str
    periods_per_year: int
    significance: SignificanceResult
    disjoint: SignificanceResult | None = None
    mean_names: float = float("nan")
    by_year: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    @property
    def t_stat(self) -> float:
        """`t` de Newey–West: el único que este repo acepta reportar como *el* `t`."""
        return self.significance.t_stat

    @property
    def ci(self) -> tuple[float, float]:
        """Intervalo de confianza de la IC media."""
        return (self.significance.ci_low, self.significance.ci_high)

    def to_dict(self) -> dict[str, object]:
        """Vista plana para tablas comparativas entre factores."""
        return {
            "mean_ic": self.mean,
            "median_ic": self.median,
            "std_ic": self.std,
            "ic_ir": self.ic_ir,
            "ic_ir_annualized": self.ic_ir_annualized,
            "t_naive": self.t_naive,
            "t_nw": self.significance.t_stat,
            "nw_lags": self.significance.nw_lags,
            "p_value": self.significance.p_value,
            "ci_low": self.significance.ci_low,
            "ci_high": self.significance.ci_high,
            "t_disjoint": None if self.disjoint is None else self.disjoint.t_stat,
            "hit_rate": self.hit_rate,
            "monthly_positive_fraction": self.monthly_positive_fraction,
            "n_periods": self.n_periods,
            "n_eff": self.significance.n_eff,
            "horizon": self.horizon,
            "method": self.method,
            "mean_names": self.mean_names,
        }

    def __str__(self) -> str:  # pragma: no cover - cosmético
        lo, hi = self.ci
        return (
            f"IC({self.method}, h={self.horizon}) = {self.mean:+.4f} "
            f"[{lo:+.4f}, {hi:+.4f}] · t_NW={self.significance.t_stat:+.2f} "
            f"(L={self.significance.nw_lags}) · IC_IR anual={self.ic_ir_annualized:+.2f} "
            f"· T={self.n_periods}"
        )


def summarize_ic(
    ic: pd.Series | pd.DataFrame,
    *,
    horizon: int = 1,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    alpha: float = 0.05,
    method: str = "spearman",
    counts: pd.Series | None = None,
) -> ICSummary:
    """Media, `IC_IR`, `t` de Newey–West e intervalo de confianza de una serie de IC.

    Cumple el mandato de `ARCHITECTURE.md` §3.8: la IC media **nunca** se reporta
    sin su banda de error. Con `horizon > 5` se añade además el contraste con
    muestreo disjunto, que es el que manda en caso de discrepancia (§3.3).

    `monthly_positive_fraction` es el criterio nº 4 de §14.1 (VIVA si ≥ 55 %): mide
    estabilidad, no solo media, y descarta factores cuya IC media la produce un
    puñado de meses.
    """
    if isinstance(ic, pd.DataFrame):
        counts = ic["n_names"] if counts is None and "n_names" in ic.columns else counts
        ic = ic["ic"]
    series = pd.Series(ic).dropna().sort_index()
    if series.size < 3:
        msg = f"la serie de IC tiene {series.size} valores no nulos: nada que resumir"
        raise InsufficientHistory(msg)

    values = series.to_numpy(dtype=float)
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    ic_ir = mean / std if std > 0 else float("nan")
    significance = mean_significance(values, horizon=horizon, alpha=alpha, label=f"IC {method}")

    disjoint: SignificanceResult | None = None
    if horizon > 5:
        try:
            disjoint = disjoint_significance(values, horizon, alpha=alpha)
        except InsufficientHistory:
            disjoint = None

    idx = series.index
    if isinstance(idx, pd.DatetimeIndex):
        monthly = series.resample("ME").mean().dropna()
        monthly_positive = float((monthly > 0).mean()) if len(monthly) else float("nan")
        by_year = series.groupby(idx.year).mean()
        by_year.index.name = "year"
    else:  # pragma: no cover - índice no temporal, uso analítico
        monthly_positive = float("nan")
        by_year = pd.Series(dtype=float)

    return ICSummary(
        mean=mean,
        median=float(np.median(values)),
        std=std,
        ic_ir=float(ic_ir),
        ic_ir_annualized=float(ic_ir * math.sqrt(periods_per_year)) if std > 0 else float("nan"),
        t_naive=float(ic_ir * math.sqrt(values.size)) if std > 0 else float("nan"),
        hit_rate=float((values > 0).mean()),
        monthly_positive_fraction=monthly_positive,
        n_periods=int(values.size),
        horizon=int(horizon),
        method=str(method),
        periods_per_year=int(periods_per_year),
        significance=significance,
        disjoint=disjoint,
        mean_names=float(pd.Series(counts).mean()) if counts is not None else float("nan"),
        by_year=by_year,
    )


# --------------------------------------------------------------------------- #
# IC condicionada: sector, tamaño                                               #
# --------------------------------------------------------------------------- #


def _group_labels(
    index: pd.MultiIndex,
    groups: pd.Series | Mapping[str, str],
) -> pd.Series:
    """Resuelve etiquetas de grupo por `(date, ticker)` o por `ticker`."""
    if isinstance(groups, Mapping) and not isinstance(groups, pd.Series):
        groups = pd.Series(groups, dtype=object)
    if isinstance(groups.index, pd.MultiIndex):
        labels = _as_panel_series(groups, "groups").reindex(index)
    else:
        tickers = index.get_level_values("ticker")
        labels = pd.Series(groups.reindex(tickers).to_numpy(), index=index)
    return labels.rename("group")


def ic_by_group(
    scores: pd.Series | pd.DataFrame,
    returns: pd.Series | pd.DataFrame,
    groups: pd.Series | Mapping[str, str],
    *,
    method: CorrelationKind = "spearman",
    min_names: int = 20,
    horizon: int = 1,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    alpha: float = 0.05,
    min_periods: int = 24,
) -> pd.DataFrame:
    """IC calculada **dentro** de cada grupo (típicamente el sector GICS).

    La IC dentro de sector responde a la pregunta de §14.1 criterio 8: *¿el factor
    ordena empresas comparables, o solo está apostando por sectores?* Una IC global
    fuerte que se desvanece dentro de sector es una apuesta sectorial disfrazada, y
    el umbral del repo es que la IC neutralizada conserve ≥ 60 % de la cruda.

    Ojo con `min_names`: dentro de un sector la sección cruzada es pequeña
    (11 sectores GICS sobre ~475 nombres dan ~43 de media, con sectores de 20). Por
    eso el valor por defecto aquí es 20 y no 30, y la columna `mean_names` del
    resultado permite juzgar cuánta fe merece cada fila.

    Returns
    -------
    pandas.DataFrame
        Una fila por grupo con `mean_ic`, `t_nw`, `nw_lags`, `p_value`, `ci_low`,
        `ci_high`, `ic_ir_annualized`, `hit_rate`, `n_periods` y `mean_names`,
        ordenada por `t_nw` descendente. Los grupos con menos de `min_periods`
        fechas utilizables se omiten y se anotan en el atributo `.attrs["skipped"]`.
    """
    df = _aligned(scores, returns)
    df["group"] = _group_labels(df.index, groups)
    df = df.dropna(subset=["group"])
    if df.empty:
        msg = "ningún activo del panel tiene grupo asignado"
        raise InsufficientHistory(msg)

    rows: list[dict[str, object]] = []
    skipped: dict[str, str] = {}
    for name, sub in df.groupby("group", observed=True):
        table = _ic_from_frame(sub[["s", "r"]], method=method, min_names=min_names)
        usable = table["ic"].dropna()
        if usable.size < min_periods:
            skipped[str(name)] = f"solo {usable.size} fechas con >= {min_names} activos"
            continue
        summary = summarize_ic(
            table.loc[usable.index],
            horizon=horizon,
            periods_per_year=periods_per_year,
            alpha=alpha,
            method=method,
        )
        row = summary.to_dict()
        row["group"] = str(name)
        rows.append(row)

    if not rows:
        msg = (
            "ningún grupo tiene sección cruzada suficiente: baja `min_names` o "
            f"`min_periods` (grupos descartados: {skipped})"
        )
        raise InsufficientHistory(msg)
    out = pd.DataFrame(rows).set_index("group").sort_values("t_nw", ascending=False)
    out.attrs["skipped"] = skipped
    return out


def ic_by_size_bucket(
    scores: pd.Series | pd.DataFrame,
    returns: pd.Series | pd.DataFrame,
    market_cap: pd.Series | pd.DataFrame,
    *,
    n_buckets: int = 5,
    labels: Sequence[str] | None = None,
    method: CorrelationKind = "spearman",
    min_names: int = 20,
    horizon: int = 1,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    alpha: float = 0.05,
    min_periods: int = 24,
) -> pd.DataFrame:
    """IC por tramo de capitalización, con tramos recalculados **cada fecha**.

    Los tramos se asignan por cuantiles de la capitalización dentro de la sección
    cruzada de cada fecha, no con cortes absolutos: un corte fijo en dólares
    convertiría la deriva del mercado en una migración espuria entre tramos a lo
    largo de 30 años.

    Es el diagnóstico que separa un factor real de un factor de tamaño: si la IC
    vive entera en el tramo pequeño, la señal es cara de implementar y su capacidad
    es la del tramo pequeño, no la del índice (§11.1 y Etapa 5 del protocolo).

    `market_cap` debe ser point-in-time (capitalización conocible en `t`), no la
    actual: usar la de hoy para fechas pasadas mete el resultado del propio periodo
    dentro del clasificador.
    """
    if n_buckets < 2:
        msg = f"n_buckets debe ser >= 2; recibido {n_buckets}"
        raise ValueError(msg)
    cap = _as_panel_series(market_cap, "market_cap")
    df = _aligned(scores, returns)
    cap = cap.reindex(df.index)
    df = df[cap.notna()]
    cap = cap.dropna()
    if df.empty:
        msg = "no hay capitalización disponible para ninguna observación del panel"
        raise InsufficientHistory(msg)

    default_labels = [f"Q{i + 1}" for i in range(n_buckets)]
    names = list(labels) if labels is not None else default_labels
    if len(names) != n_buckets:
        msg = f"labels tiene {len(names)} etiquetas para {n_buckets} tramos"
        raise ValueError(msg)

    ranks = cap.groupby(level="date").rank(pct=True, method="first")
    codes = np.ceil(ranks.to_numpy() * n_buckets).astype(int) - 1
    codes = np.clip(codes, 0, n_buckets - 1)
    bucket = pd.Series([names[c] for c in codes], index=cap.index, name="bucket")

    out = ic_by_group(
        df["s"],
        df["r"],
        bucket,
        method=method,
        min_names=min_names,
        horizon=horizon,
        periods_per_year=periods_per_year,
        alpha=alpha,
        min_periods=min_periods,
    )
    order = [n for n in names if n in out.index]
    return out.loc[order]


# --------------------------------------------------------------------------- #
# Decaimiento del IC por horizonte                                              #
# --------------------------------------------------------------------------- #


def ic_decay(
    scores: pd.Series | pd.DataFrame,
    prices: pd.DataFrame | pd.Series | None = None,
    *,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    returns_by_horizon: Mapping[int, pd.Series] | None = None,
    method: CorrelationKind = "spearman",
    min_names: int = DEFAULT_MIN_NAMES,
    price_col: str = "adj_close",
    execution_lag: int = 1,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    alpha: float = 0.05,
    membership: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Perfil de decaimiento: IC frente a retornos a 1, 5, 10, 21 y 63 sesiones.

    Es el punto 3 de la Etapa 1 del protocolo (§13). Lo que se busca en el perfil:

    - **IC creciente con `h` y luego plana**: información sobre el nivel del precio
      que el mercado incorpora en unas semanas. Es el patrón del PEAD.
    - **IC que decae rápido a cero**: la señal es de muy corto plazo y el coste de
      transacción se la come; hay que mirar la Etapa 5 antes que nada más.
    - **IC que cambia de signo**: reversión. El horizonte de la señal está mal
      elegido o la señal captura un efecto de microestructura.

    La columna `ic_per_sqrt_h` normaliza por `√h`. Si la información se acumula como
    un paseo aleatorio con *drift* constante, esa columna es plana; si crece, el
    efecto se concentra en horizontes largos; si cae, en los cortos.

    La columna `t_disjoint` aparece para `h > 5` y **manda sobre `t_nw`** cuando
    discrepan (§3.3): el muestreo disjunto es exacto, Newey–West conserva un sesgo
    residual del 25 % que no desaparece al aumentar los lags.

    Parameters
    ----------
    prices:
        Panel de precios del que derivar los retornos *forward*. Alternativamente se
        pasan ya calculados en `returns_by_horizon`.
    returns_by_horizon:
        Mapa `horizonte -> Series de retorno forward`. Tiene prioridad sobre
        `prices`; útil para reutilizar retornos ya calculados o para tests con IC
        conocida por construcción.
    """
    hs = [int(h) for h in horizons]
    if not hs:
        msg = "hay que pedir al menos un horizonte"
        raise ValueError(msg)
    if any(h < 1 for h in hs):
        msg = f"todos los horizontes deben ser >= 1; recibido {hs}"
        raise ValueError(msg)
    if returns_by_horizon is None and prices is None:
        msg = "hay que pasar `prices` o `returns_by_horizon`"
        raise ValueError(msg)

    rows: list[dict[str, object]] = []
    for h in hs:
        if returns_by_horizon is not None and h in returns_by_horizon:
            fwd = returns_by_horizon[h]
        else:
            assert prices is not None  # garantizado arriba
            fwd = forward_returns(
                prices, h, price_col=price_col, execution_lag=execution_lag
            )
        table = cross_sectional_ic(
            scores,
            fwd,
            method=method,
            min_names=min_names,
            membership=membership,
            with_counts=True,
        )
        summary = summarize_ic(
            table,
            horizon=h,
            periods_per_year=periods_per_year,
            alpha=alpha,
            method=method,
        )
        row = summary.to_dict()
        row["horizon"] = h
        row["ic_per_sqrt_h"] = summary.mean / math.sqrt(h)
        rows.append(row)

    out = pd.DataFrame(rows).set_index("horizon")
    base = out["mean_ic"].iloc[0]
    out["ic_ratio_vs_first"] = out["mean_ic"] / base if base != 0 else np.nan
    cols = [
        "mean_ic",
        "ic_per_sqrt_h",
        "ic_ratio_vs_first",
        "ic_ir",
        "ic_ir_annualized",
        "t_nw",
        "nw_lags",
        "t_disjoint",
        "t_naive",
        "p_value",
        "ci_low",
        "ci_high",
        "hit_rate",
        "n_periods",
        "n_eff",
        "mean_names",
    ]
    return out[[c for c in cols if c in out.columns]]


# --------------------------------------------------------------------------- #
# Cribado según los umbrales del repositorio                                    #
# --------------------------------------------------------------------------- #


def screen_ic_criteria(
    summary: ICSummary,
    *,
    established: bool = False,
    neutralized_mean_ic: float | None = None,
) -> pd.DataFrame:
    """Comprueba los criterios 1-4 y 8 de §14.1 sobre un resumen de IC.

    Devuelve una tabla con el valor observado, los umbrales de VIVA y CUARENTENA y
    el veredicto por criterio. No decide por el usuario: agrega el juicio pero deja
    la evidencia a la vista, que es lo que exige el protocolo.

    `established=True` aplica el umbral `t_NW ≥ 2,0` reservado a factores con
    literatura previa robusta (SUE, PEAD, accruals, Piotroski); un factor nuevo
    necesita `t_NW ≥ 3,0` (§7.3: la postura conservadora ante la disputa
    Harvey–Liu–Zhu vs Chen–Zimmermann).
    """
    t_live = 2.0 if established else 3.0
    t_quar = 1.7 if established else 2.0
    checks: list[tuple[str, float, float, float]] = [
        ("rank_ic_mean", summary.mean, 0.020, 0.010),
        ("t_newey_west", summary.significance.t_stat, t_live, t_quar),
        ("ic_ir_annualized", summary.ic_ir_annualized, 0.30, 0.20),
        ("monthly_positive_fraction", summary.monthly_positive_fraction, 0.55, 0.52),
    ]
    if neutralized_mean_ic is not None and summary.mean != 0:
        checks.append(
            (
                "sector_neutral_retention",
                float(neutralized_mean_ic / summary.mean),
                0.60,
                0.40,
            )
        )
    rows = []
    for name, value, live, quarantine in checks:
        if not np.isfinite(value):
            verdict = "SIN_DATO"
        elif value >= live:
            verdict = "VIVA"
        elif value >= quarantine:
            verdict = "CUARENTENA"
        else:
            verdict = "MUERTA"
        rows.append(
            {
                "criterion": name,
                "value": value,
                "threshold_live": live,
                "threshold_quarantine": quarantine,
                "verdict": verdict,
            }
        )
    return pd.DataFrame(rows).set_index("criterion")
