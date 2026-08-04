"""Métricas de rendimiento, **todas con intervalo de confianza**.

Implementa la sección 5 de `docs/research/validation_methodology.md` y el mandato de
`docs/ARCHITECTURE.md` §3.8: *toda métrica de rendimiento debe reportar su intervalo
de confianza; un Sharpe sin banda de error no se acepta*. Por eso no hay en este
módulo ninguna función que devuelva un `float` suelto de rendimiento: devuelven
`Estimate`, `SharpeResult` o `DSRResult`, que llevan la banda incorporada.

Contenido
---------
- Retorno anualizado, volatilidad, Sharpe, Calmar, *max drawdown* con su duración,
  rotación y capacidad.
- Sharpe **probabilístico** (PSR) y **deflactado** (DSR) de Bailey y López de Prado,
  con la fórmula completa que incorpora asimetría y curtosis.
- `MinTRL` (longitud mínima de registro) y `MinBTL` (longitud mínima de backtest).

Tres trampas de implementación que este módulo evita explícitamente, y que fallan
**en silencio y en la dirección peligrosa**:

1. **Curtosis en exceso en vez de cruda.** `scipy.stats.kurtosis` devuelve el exceso
   por defecto (`fisher=True`). En el PSR hay que usar la **cruda** (3 para la
   normal). Restar 3 de más hace el denominador más pequeño e **infla** la
   significancia. Este módulo calcula siempre `fisher=False` y **rechaza** una
   curtosis < 1, que es matemáticamente imposible para la cruda y por tanto delata
   que se ha pasado el exceso.
2. **Sharpe anualizado dentro de PSR/DSR.** Las fórmulas usan el Sharpe **por
   observación**. Aquí la anualización solo ocurre al reportar.
3. **`E[max SR]` con `N/e` en vez de `N·e`.** La transcripción más difundida de la
   fórmula de Bailey–López de Prado tiene un error de signo en el exponente que
   subestima el umbral entre un 11 % y un 43 %, siempre en la dirección de aceptar
   estrategias falsas. Aquí se usa `N·e`, verificado contra Monte Carlo.

Referencias
-----------
- Lo, A. W. (2002). *The Statistics of Sharpe Ratios*. **FAJ** 58(4), 36-52.
- Bailey, D. H., y López de Prado, M. (2012). *The Sharpe Ratio Efficient Frontier*.
  **Journal of Risk** 15(2), 3-44.
- Bailey, D. H., y López de Prado, M. (2014). *The Deflated Sharpe Ratio: Correcting
  for Selection Bias, Backtest Overfitting and Non-Normality*. **JPM** 40(5), 94-107.
- Bailey, D. H., Borwein, J. M., López de Prado, M., y Zhu, Q. J. (2014).
  *Pseudo-Mathematics and Financial Charlatanism*. **Notices of the AMS** 61(5).
- Opdyke, J. D. (2007). *Comparing Sharpe ratios: So where are the p-values?*
  **Journal of Asset Management** 8, 308-336.
- López de Prado, M., y Lewis, M. J. (2019). *Detection of False Investment
  Strategies Using Unsupervised Learning Methods*. **Quantitative Finance** 19(9).
- Grinold, R. C. (1989); Clarke, de Silva y Thorley (2002) para la ley fundamental.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from earnings_alpha.errors import DataQualityError, InsufficientHistory
from earnings_alpha.stats.ic import TRADING_DAYS_PER_YEAR, mean_significance
from earnings_alpha.stats.validation import (
    EULER_MASCHERONI,
    BootstrapResult,
    stationary_bootstrap,
)

__all__ = [
    "CapacityResult",
    "DSRResult",
    "DrawdownResult",
    "Estimate",
    "SharpeResult",
    "TurnoverResult",
    "annualized_return",
    "annualized_volatility",
    "calmar_ratio",
    "capacity",
    "deflated_sharpe_ratio",
    "drawdown_series",
    "effective_breadth",
    "expected_max_sharpe",
    "fundamental_law_sharpe",
    "max_drawdown",
    "min_backtest_length",
    "min_trl",
    "n_effective_trials",
    "performance_summary",
    "psr",
    "raw_kurtosis",
    "sharpe_ratio",
    "sharpe_standard_error",
    "sharpe_variance_factor",
    "turnover",
]


# --------------------------------------------------------------------------- #
# Contenedores                                                                  #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Estimate:
    """Una métrica con su banda de error. No existe métrica sin banda en este repo."""

    name: str
    value: float
    ci_low: float
    ci_high: float
    std_error: float
    method: str
    alpha: float = 0.05
    n_obs: int = 0
    notes: str = ""

    @property
    def excludes_zero(self) -> bool:
        return self.ci_low > 0.0 or self.ci_high < 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "metric": self.name,
            "value": self.value,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "std_error": self.std_error,
            "method": self.method,
            "n_obs": self.n_obs,
        }

    def __str__(self) -> str:  # pragma: no cover - cosmético
        return (
            f"{self.name} = {self.value:.4f} "
            f"[{self.ci_low:.4f}, {self.ci_high:.4f}] ({self.method})"
        )


# --------------------------------------------------------------------------- #
# Utilidades básicas                                                            #
# --------------------------------------------------------------------------- #


def _clean_returns(returns: pd.Series | np.ndarray | Sequence[float], *, minimum: int = 8) -> np.ndarray:
    arr = np.asarray(
        returns.to_numpy() if isinstance(returns, pd.Series) else returns, dtype=float
    )
    if arr.ndim != 1:
        msg = "se espera una serie unidimensional de retornos por periodo"
        raise DataQualityError(msg)
    arr = arr[np.isfinite(arr)]
    if arr.size < minimum:
        msg = f"solo {arr.size} retornos finitos; se necesitan al menos {minimum}"
        raise InsufficientHistory(msg)
    if np.any(arr <= -1.0):
        n_bad = int(np.sum(arr <= -1.0))
        msg = (
            f"{n_bad} retornos son <= -100 %: la cartera no puede perder más que todo "
            "el capital, hay un error de escala (¿porcentajes en vez de fracciones?)"
        )
        raise DataQualityError(msg)
    return arr


def raw_kurtosis(x: np.ndarray | pd.Series) -> float:
    """Curtosis **cruda** (3 para la normal), que es la que exigen PSR y DSR.

    `scipy.stats.kurtosis` devuelve el **exceso** por defecto. Confundirlas es el
    error nº 3 de la lista de fallos frecuentes, y falla en la dirección peligrosa:
    hace el denominador del PSR más pequeño y por tanto **infla** la significancia.
    """
    return float(stats.kurtosis(np.asarray(x, dtype=float), fisher=False, bias=True))


def _validate_moments(skew: float, kurtosis: float) -> None:
    if not np.isfinite(skew) or not np.isfinite(kurtosis):
        msg = f"asimetría o curtosis no finitas: γ₃={skew}, γ₄={kurtosis}"
        raise DataQualityError(msg)
    if kurtosis < 1.0:
        msg = (
            f"la curtosis recibida es {kurtosis:.4f}, y la curtosis CRUDA no puede ser "
            "menor que 1 (es 3 para la normal). Casi con seguridad se ha pasado la "
            "curtosis en EXCESO, que es lo que devuelve scipy.stats.kurtosis por "
            "defecto: usa `fisher=False` o suma 3. Con el exceso, PSR y DSR salen "
            "inflados"
        )
        raise DataQualityError(msg)


def sharpe_variance_factor(sharpe: float, skew: float = 0.0, kurtosis: float = 3.0) -> float:
    """Factor `√[1 − γ₃·SR̂ + ((γ̂₄−1)/4)·SR̂²]` del error estándar del Sharpe.

    Es el denominador del PSR y el numerador del error estándar de Mertens /
    Christie / Opdyke / Bailey–López de Prado. `SR̂` va en unidades **por
    observación**.

    Verificación de consistencia exacta con Lo (2002): con `γ₃ = 0` y `γ₄ = 3`
    (curtosis cruda de la normal) debe reducirse a `√(1 + SR²/2)`.

    Examples
    --------
    >>> round(sharpe_variance_factor(0.10, 0.0, 3.0), 8)
    1.00249688
    >>> round(math.sqrt(1 + 0.10**2 / 2), 8)
    1.00249688
    """
    _validate_moments(skew, kurtosis)
    inner = 1.0 - skew * sharpe + ((kurtosis - 1.0) / 4.0) * sharpe * sharpe
    if inner <= 0.0:
        msg = (
            f"el factor de varianza del Sharpe es {inner:.4f} <= 0 con SR={sharpe:.4f}, "
            f"γ₃={skew:.3f}, γ₄={kurtosis:.3f}: momentos incompatibles"
        )
        raise DataQualityError(msg)
    return float(math.sqrt(inner))


def sharpe_standard_error(
    sharpe: float,
    n_obs: int,
    *,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    method: Literal["lo", "mertens"] = "mertens",
) -> float:
    """Error estándar del Sharpe **por observación**.

    - Lo (2002), retornos iid normales: `SE = √[(1 + SR̂²/2)/T]`.
    - Mertens / Bailey–López de Prado, con no-normalidad:
      `SE = √[(1 − γ̂₃·SR̂ + ((γ̂₄−1)/4)·SR̂²)/(T−1)]`.

    **Cuánto importa la no-normalidad** (§5.1, con asimetría −0,5 y curtosis cruda
    8): a frecuencia **diaria** la corrección es del 1–4 %; a frecuencia **mensual**
    con Sharpe 2 llega al **26,7 %**. Es decir, reportar el Sharpe mensual (práctica
    común en la industria) hace la corrección *más* necesaria, no menos: la
    agregación no elimina las colas, concentra su efecto en el estadístico.

    Corolario honesto: si el backtest se evalúa a frecuencia diaria, el PSR aporta
    poco sobre el `t` clásico. **Lo que mata las falsas señales por órdenes de
    magnitud es la deflación por número de pruebas** (`deflated_sharpe_ratio`).
    """
    if n_obs < 2:
        msg = f"n_obs debe ser >= 2; recibido {n_obs}"
        raise InsufficientHistory(msg)
    if method == "lo":
        return float(math.sqrt((1.0 + sharpe * sharpe / 2.0) / n_obs))
    factor = sharpe_variance_factor(sharpe, skew, kurtosis)
    return float(factor / math.sqrt(n_obs - 1))


def psr(
    sharpe: float,
    *,
    benchmark: float = 0.0,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Sharpe probabilístico (Bailey y López de Prado, 2012, 2014).

    Probabilidad de que el Sharpe verdadero supere el umbral `SR*`::

        PSR(SR*) = Φ[ (SR̂ − SR*)·√(T − 1) / √(1 − γ₃·SR̂ + ((γ₄ − 1)/4)·SR̂²) ]

    con `γ₃` la asimetría muestral y `γ₄` la curtosis **cruda** (3 para la normal).

    `PSR(0)` es el complementario del p-valor de una cola de `H₀: SR = 0`, ya
    corregido por longitud de muestra, asimetría y curtosis. **Pero no por
    selección**: para eso está el DSR.

    `SR̂` y `SR*` van en unidades **por observación**. Meter un Sharpe anualizado en
    esta fórmula es el error de implementación más común y produce p-valores
    absurdos.
    """
    if n_obs < 2:
        msg = f"n_obs debe ser >= 2; recibido {n_obs}"
        raise InsufficientHistory(msg)
    factor = sharpe_variance_factor(sharpe, skew, kurtosis)
    z = (sharpe - benchmark) * math.sqrt(n_obs - 1) / factor
    return float(stats.norm.cdf(z))


def min_trl(
    sharpe: float,
    *,
    benchmark: float = 0.0,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    alpha: float = 0.05,
    periods_per_year: int | None = None,
) -> float:
    """Longitud mínima de registro (MinTRL) para un `PSR(SR*) = 1 − α`.

    Despejando `T` de la ecuación del PSR::

        MinTRL = 1 + [1 − γ₃·SR̂ + ((γ₄−1)/4)·SR̂²] · (Z_α / (SR̂ − SR*))²

    Valores de referencia con datos diarios, `SR* = 0`, `α = 5 %`, asimetría −0,5 y
    curtosis cruda 8: un Sharpe anual de 0,5 exige **11,0 años**; de 1,0, **2,81
    años**; de 2,0, 0,74 años. Y esto es solo para **una** estrategia: con selección
    entre varias, la exigencia sube (ver `min_backtest_length`).

    Parameters
    ----------
    sharpe:
        Por observación, salvo que se pase `periods_per_year`, en cuyo caso se
        interpreta como **anualizado** y se convierte.

    Examples
    --------
    >>> round(min_trl(1.0, skew=-0.5, kurtosis=8.0, periods_per_year=252))
    709
    """
    if periods_per_year is not None:
        if periods_per_year < 1:
            msg = f"periods_per_year debe ser >= 1; recibido {periods_per_year}"
            raise ValueError(msg)
        sharpe = sharpe / math.sqrt(periods_per_year)
        benchmark = benchmark / math.sqrt(periods_per_year)
    gap = sharpe - benchmark
    if gap <= 0:
        return float("inf")
    factor = sharpe_variance_factor(sharpe, skew, kurtosis)
    z_alpha = float(stats.norm.ppf(1.0 - alpha))
    return float(1.0 + (factor**2) * (z_alpha / gap) ** 2)


def expected_max_sharpe(n_trials: int, mean: float = 0.0, sd: float = 1.0) -> float:
    """Sharpe máximo esperado tras `N` pruebas bajo la hipótesis nula.

    Fórmula de Bailey y López de Prado (2014), del máximo de `N` gaussianas
    independientes::

        E[max SR] = mean + sd·[ (1 − γ)·Z⁻¹(1 − 1/N) + γ·Z⁻¹(1 − 1/(N·e)) ]

    con `γ` la constante de Euler–Mascheroni y `e` el número de Euler.

    .. warning::
       Buena parte de las reproducciones online escriben el segundo término como
       `Z⁻¹[1 − 1/(N·e⁻¹)]`, es decir `N/e`. **Es incorrecto**: es un error de signo
       en el exponente. Verificado contra Monte Carlo del máximo de `N` normales
       estándar, la forma con `N·e` reproduce la simulación con error < 2,5 % en
       todo el rango (`N`=1.000 → 3,2551 frente a 3,2422 medido), mientras que
       `N/e` subestima entre un 11 % y un 43 %, **siempre** en la dirección de
       aceptar estrategias falsas.

    `sd` es la desviación típica de los Sharpe **por observación** de las `N`
    pruebas, es decir `√V[{SR_n}]` en la notación del paper.

    Examples
    --------
    >>> round(expected_max_sharpe(1000), 4)
    3.2551
    >>> round(expected_max_sharpe(100), 4)
    2.5306
    """
    if n_trials < 1:
        msg = f"n_trials debe ser >= 1; recibido {n_trials}"
        raise ValueError(msg)
    if sd < 0:
        msg = f"sd no puede ser negativa; recibida {sd}"
        raise ValueError(msg)
    if n_trials == 1:
        # Con una sola prueba no hay selección: el umbral es el propio centro.
        return float(mean)
    z1 = float(stats.norm.ppf(1.0 - 1.0 / n_trials))
    z2 = float(stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e)))
    return float(mean + sd * ((1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2))


def min_backtest_length(
    n_trials: int, target_sharpe: float = 1.0, *, exact: bool = True
) -> float:
    """Longitud mínima de backtest (MinBTL) en **años** (Bailey et al., 2014).

    Complementaria del DSR: en vez de "¿es significativo?", pregunta *¿cuántos años
    necesito para que `N` pruebas no produzcan un Sharpe objetivo espurio?*::

        MinBTL = ( E[max_N] / SR_objetivo )²        (exacta)
        MinBTL ≲ 2·ln(N) / SR_objetivo²             (cota clásica, √(2 ln N))

    Con `N = 1.000` y `SR = 1,0`: 10,60 años con la forma exacta, 13,82 con la cota.

    **Aplicado a este repo:** hay 29,6 años de composición del índice, que soportan
    del orden de `N ≈ 10⁵` pruebas antes de que un Sharpe 1,0 sea indistinguible del
    ruido. Un backtest de 5 años solo soporta `N ≈ 25`. Corolario: **los backtests
    cortos están prohibidos como evidencia principal**.
    """
    if target_sharpe <= 0:
        msg = f"target_sharpe debe ser positivo; recibido {target_sharpe}"
        raise ValueError(msg)
    if n_trials < 2:
        # Sin selección no hay sesgo de selección que compensar con más historia.
        return 0.0
    e_max = expected_max_sharpe(n_trials) if exact else math.sqrt(2.0 * math.log(n_trials))
    return float((e_max / target_sharpe) ** 2)


def n_effective_trials(
    trial_returns: pd.DataFrame, *, threshold: float = 0.9, min_periods: int = 20
) -> int:
    """Número de pruebas **efectivamente independientes** entre las probadas.

    Si se prueban 500 configuraciones que son variaciones mínimas unas de otras,
    `N = 500` sobre-penaliza el DSR. López de Prado y Lewis (2019) proponen agrupar
    las series de retorno de las pruebas y usar el número de clusters. La regla
    práctica del repo, más simple y defendible:

        `N_eff` = número de clusters obtenidos uniendo pruebas cuya correlación
        supera `threshold` (0,9 por defecto), por cierre transitivo.

    **En caso de duda se reporta el DSR con `N_eff` y con `N` bruto.** Si las
    conclusiones difieren, la señal está en zona gris y no se promociona.
    """
    data = trial_returns.dropna(how="all", axis=1)
    if data.shape[1] == 0:
        msg = "no hay ninguna columna de pruebas"
        raise InsufficientHistory(msg)
    if len(data) < min_periods:
        msg = f"solo {len(data)} periodos: insuficiente para estimar correlaciones"
        raise InsufficientHistory(msg)
    corr = data.corr().to_numpy()
    n = corr.shape[0]
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if np.isfinite(corr[i, j]) and corr[i, j] > threshold:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    return len({find(i) for i in range(n)})


# --------------------------------------------------------------------------- #
# Sharpe                                                                        #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class SharpeResult:
    """Sharpe con banda de error, momentos y su PSR."""

    sharpe_per_period: float
    sharpe_annualized: float
    std_error_per_period: float
    ci_low: float
    ci_high: float
    n_obs: int
    periods_per_year: int
    skew: float
    kurtosis: float
    """Curtosis **cruda** (3 para la normal)."""
    psr_zero: float
    min_track_record: float
    method: str
    alpha: float = 0.05
    bootstrap: BootstrapResult | None = None

    @property
    def significant(self) -> bool:
        """El IC anualizado excluye el cero."""
        return self.ci_low > 0.0 or self.ci_high < 0.0

    def as_estimate(self) -> Estimate:
        return Estimate(
            name="sharpe_annualized",
            value=self.sharpe_annualized,
            ci_low=self.ci_low,
            ci_high=self.ci_high,
            std_error=self.std_error_per_period * math.sqrt(self.periods_per_year),
            method=self.method,
            alpha=self.alpha,
            n_obs=self.n_obs,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "sharpe_annualized": self.sharpe_annualized,
            "sharpe_per_period": self.sharpe_per_period,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "std_error_per_period": self.std_error_per_period,
            "skew": self.skew,
            "kurtosis_raw": self.kurtosis,
            "psr_zero": self.psr_zero,
            "min_track_record_obs": self.min_track_record,
            "n_obs": self.n_obs,
            "method": self.method,
        }

    def __str__(self) -> str:  # pragma: no cover - cosmético
        return (
            f"Sharpe anual = {self.sharpe_annualized:.3f} "
            f"[{self.ci_low:.3f}, {self.ci_high:.3f}] · PSR(0)={self.psr_zero:.4f} "
            f"· T={self.n_obs} ({self.method})"
        )


def sharpe_ratio(
    returns: pd.Series | np.ndarray,
    *,
    risk_free: float | pd.Series | np.ndarray = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    alpha: float = 0.05,
    ci_method: Literal["mertens", "lo", "bootstrap"] = "mertens",
    n_boot: int = 1000,
    block_length: float | None = None,
    seed: int | None = 20260804,
    horizon: int = 1,
) -> SharpeResult:
    """Ratio de Sharpe con intervalo de confianza, PSR y MinTRL.

    El Sharpe se calcula sobre el **exceso** de retorno (`returns − risk_free`) a la
    frecuencia de muestreo, y se anualiza multiplicando por `√(periodos por año)`.
    Las fórmulas de PSR y MinTRL se evalúan siempre en unidades **por observación**.

    `ci_method`:

    - `"mertens"` (por defecto): error estándar con asimetría y curtosis cruda.
    - `"lo"`: error estándar bajo iid normal, `√[(1+SR²/2)/T]`.
    - `"bootstrap"`: bootstrap estacionario, que además recoge la autocorrelación
      de los retornos de la cartera (los factores fundamentales son casi constantes
      entre presentaciones, §1.2(c), y eso autocorrela el retorno de la cartera).
    """
    arr = _clean_returns(returns)
    if isinstance(risk_free, (pd.Series, np.ndarray)):
        rf = np.asarray(
            risk_free.to_numpy() if isinstance(risk_free, pd.Series) else risk_free, dtype=float
        )
        if rf.size != arr.size:
            msg = f"risk_free tiene {rf.size} valores y los retornos {arr.size}"
            raise DataQualityError(msg)
        excess = arr - rf
    else:
        excess = arr - float(risk_free)

    n = excess.size
    mean = float(excess.mean())
    sd = float(excess.std(ddof=1))
    if sd <= 0:
        msg = "la serie de exceso de retorno tiene volatilidad nula"
        raise DataQualityError(msg)
    sr = mean / sd
    skew = float(stats.skew(excess, bias=True))
    kurt = raw_kurtosis(excess)

    boot: BootstrapResult | None = None
    if ci_method == "bootstrap":
        boot = stationary_bootstrap(
            excess,
            statistic=lambda z: float(np.mean(z) / np.std(z, ddof=1))
            if np.std(z, ddof=1) > 0
            else float("nan"),
            n_boot=n_boot,
            alpha=alpha,
            seed=seed,
            block_length=block_length,
            horizon=horizon,
        )
        se = boot.std_error
        lo, hi = boot.ci_low, boot.ci_high
    else:
        se = sharpe_standard_error(
            sr, n, skew=skew, kurtosis=kurt, method="lo" if ci_method == "lo" else "mertens"
        )
        z = float(stats.norm.ppf(1.0 - alpha / 2.0))
        lo, hi = sr - z * se, sr + z * se

    scale = math.sqrt(periods_per_year)
    return SharpeResult(
        sharpe_per_period=float(sr),
        sharpe_annualized=float(sr * scale),
        std_error_per_period=float(se),
        ci_low=float(lo * scale),
        ci_high=float(hi * scale),
        n_obs=int(n),
        periods_per_year=int(periods_per_year),
        skew=skew,
        kurtosis=kurt,
        psr_zero=psr(sr, benchmark=0.0, n_obs=n, skew=skew, kurtosis=kurt),
        min_track_record=min_trl(sr, skew=skew, kurtosis=kurt, alpha=alpha),
        method=f"sharpe_{ci_method}",
        alpha=alpha,
        bootstrap=boot,
    )


@dataclass(frozen=True, slots=True)
class DSRResult:
    """Sharpe deflactado: el PSR evaluado contra el máximo esperado del ruido."""

    dsr: float
    psr_zero: float
    sharpe_per_period: float
    sharpe_annualized: float
    sr_star_per_period: float
    sr_star_annualized: float
    n_trials: int
    sr_variance: float
    n_obs: int
    periods_per_year: int
    skew: float
    kurtosis: float
    notes: str = ""

    @property
    def verdict(self) -> str:
        """Criterio 9 de §14.1: VIVA con DSR ≥ 0,95; CUARENTENA ≥ 0,90."""
        if self.dsr >= 0.95:
            return "VIVA"
        if self.dsr >= 0.90:
            return "CUARENTENA"
        return "MUERTA"

    def to_dict(self) -> dict[str, object]:
        return {
            "dsr": self.dsr,
            "psr_zero": self.psr_zero,
            "sharpe_annualized": self.sharpe_annualized,
            "sr_star_annualized": self.sr_star_annualized,
            "n_trials": self.n_trials,
            "sr_variance": self.sr_variance,
            "n_obs": self.n_obs,
            "verdict": self.verdict,
        }

    def __str__(self) -> str:  # pragma: no cover - cosmético
        return (
            f"DSR = {self.dsr:.4f} (N={self.n_trials}, SR*={self.sr_star_annualized:.2f} "
            f"anual) frente a PSR(0) = {self.psr_zero:.4f} → {self.verdict}"
        )


def deflated_sharpe_ratio(
    returns: pd.Series | np.ndarray,
    *,
    n_trials: int | None = None,
    trial_sharpes: Sequence[float] | np.ndarray | pd.Series | None = None,
    sr_variance: float | None = None,
    trials_annualized: bool = False,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    risk_free: float = 0.0,
) -> DSRResult:
    """Sharpe deflactado (Bailey y López de Prado, 2014). **El corazón del asunto.**

    El DSR es el PSR evaluado contra un umbral `SR*` que no es cero, sino el Sharpe
    máximo que uno esperaría por puro azar tras `N` pruebas::

        SR*₀ = √V · [ (1 − γ)·Z⁻¹(1 − 1/N) + γ·Z⁻¹(1 − 1/(N·e)) ]
        DSR  = PSR(SR*₀)

    con `V` la **varianza de los Sharpe (por observación) de las `N` pruebas
    realizadas**.

    **Cuánto deflacta.** Una estrategia con 5 años diarios y Sharpe anual 1,0
    (`t` clásico 2,24) tiene `PSR(0) = 0,986`, aparentemente sólida al 98,6 %. Con
    solo **100 pruebas** y dispersión moderada cae a **DSR = 0,28**. Con 100 pruebas,
    un Sharpe 1,0 a 5 años no es evidencia de nada.

    **Cómo contar `N` honestamente** (§5.5). `N` no es "el número de backtests que
    guardé". Incluye cada combinación de hiperparámetros, cada variante de
    preprocesado, **cada decisión tomada después de ver un resultado** (incluida
    invertir el signo de una señal), las pruebas de otros agentes sobre el mismo
    panel y las pruebas implícitas en la literatura que inspiró el factor. El
    contrato del repo es leer `N` y `V̂` del registro append-only
    `data/trials/<familia>.jsonl`, nunca elegirlos a mano: **un DSR con un `N`
    inventado es peor que no calcular DSR**, porque da una falsa sensación de rigor.

    Por eso esta función **exige** `trial_sharpes` o `sr_variance` explícitos y no
    asume ningún valor por defecto para la dispersión entre pruebas.

    Parameters
    ----------
    trial_sharpes:
        Sharpes de todas las pruebas de la familia. De aquí salen `N` y `V̂`.
    sr_variance:
        Alternativa a lo anterior: varianza de los Sharpe entre pruebas. Debe ir en
        unidades **por observación** salvo que `trials_annualized=True`.
    n_trials:
        Obligatorio si se pasa `sr_variance` en vez de `trial_sharpes`; puede ser
        mayor que `len(trial_sharpes)` si se conoce el recuento real del registro.
    """
    arr = _clean_returns(returns)
    excess = arr - float(risk_free)
    n = excess.size
    sd = float(excess.std(ddof=1))
    if sd <= 0:
        msg = "la serie de exceso de retorno tiene volatilidad nula"
        raise DataQualityError(msg)
    sr = float(excess.mean()) / sd
    skew = float(stats.skew(excess, bias=True))
    kurt = raw_kurtosis(excess)

    if trial_sharpes is not None:
        trials = np.asarray(
            trial_sharpes.to_numpy() if isinstance(trial_sharpes, pd.Series) else trial_sharpes,
            dtype=float,
        )
        trials = trials[np.isfinite(trials)]
        if trials.size < 2:
            msg = (
                "hacen falta al menos 2 Sharpes de prueba para estimar V̂; con una "
                "sola prueba no hay selección que deflactar"
            )
            raise InsufficientHistory(msg)
        if trials_annualized:
            trials = trials / math.sqrt(periods_per_year)
        variance = float(np.var(trials, ddof=1))
        trials_count = int(n_trials) if n_trials is not None else int(trials.size)
    else:
        if sr_variance is None or n_trials is None:
            msg = (
                "hay que pasar `trial_sharpes`, o bien `sr_variance` y `n_trials`. "
                "El DSR con un N inventado es peor que no calcularlo: el contrato del "
                "repo es leerlos del registro append-only de pruebas"
            )
            raise ValueError(msg)
        variance = float(sr_variance)
        if trials_annualized:
            variance = variance / periods_per_year
        trials_count = int(n_trials)
    if variance < 0:
        msg = f"la varianza entre pruebas no puede ser negativa; recibida {variance}"
        raise DataQualityError(msg)

    sr_star = expected_max_sharpe(trials_count, mean=0.0, sd=math.sqrt(variance))
    scale = math.sqrt(periods_per_year)
    return DSRResult(
        dsr=psr(sr, benchmark=sr_star, n_obs=n, skew=skew, kurtosis=kurt),
        psr_zero=psr(sr, benchmark=0.0, n_obs=n, skew=skew, kurtosis=kurt),
        sharpe_per_period=sr,
        sharpe_annualized=sr * scale,
        sr_star_per_period=float(sr_star),
        sr_star_annualized=float(sr_star * scale),
        n_trials=trials_count,
        sr_variance=variance,
        n_obs=int(n),
        periods_per_year=int(periods_per_year),
        skew=skew,
        kurtosis=kurt,
        notes=(
            "N y V̂ deben proceder del registro append-only de pruebas "
            "(data/trials/<familia>.jsonl), no de una elección a mano."
        ),
    )


# --------------------------------------------------------------------------- #
# Retorno, volatilidad, drawdown, Calmar                                        #
# --------------------------------------------------------------------------- #


def _bootstrap_estimate(
    name: str,
    values: np.ndarray,
    statistic,
    *,
    alpha: float,
    n_boot: int,
    block_length: float | None,
    seed: int | None,
    horizon: int,
    notes: str = "",
) -> Estimate:
    boot = stationary_bootstrap(
        values,
        statistic=statistic,
        n_boot=n_boot,
        alpha=alpha,
        seed=seed,
        block_length=block_length,
        horizon=horizon,
    )
    return Estimate(
        name=name,
        value=boot.value,
        ci_low=boot.ci_low,
        ci_high=boot.ci_high,
        std_error=boot.std_error,
        method=boot.method,
        alpha=alpha,
        n_obs=values.size,
        notes=notes or f"L={boot.block_length:.1f}, B={boot.n_boot}",
    )


def annualized_return(
    returns: pd.Series | np.ndarray,
    *,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    geometric: bool = True,
    alpha: float = 0.05,
    ci_method: Literal["bootstrap", "newey_west"] = "bootstrap",
    n_boot: int = 1000,
    block_length: float | None = None,
    seed: int | None = 20260804,
    horizon: int = 1,
) -> Estimate:
    """Retorno anualizado con intervalo de confianza.

    Geométrico: `(∏(1+r))^(ppy/T) − 1`, que es el que un inversor experimenta.
    Aritmético: `r̄·ppy`, que es el que aparece en las fórmulas de Sharpe. La
    diferencia entre ambos es aproximadamente `σ²/2` y crece con la volatilidad; con
    una estrategia al 20 % anual de volatilidad son 2 puntos porcentuales al año, así
    que declarar cuál se reporta no es una formalidad.

    `ci_method="newey_west"` solo está disponible para la versión aritmética, donde
    el estimador es una media y `mean_significance` aplica directamente.
    """
    arr = _clean_returns(returns)

    if geometric:
        def stat(z: np.ndarray) -> float:
            z = np.asarray(z, dtype=float)
            return float(np.exp(np.log1p(z).mean() * periods_per_year) - 1.0)
    else:
        def stat(z: np.ndarray) -> float:
            return float(np.mean(z) * periods_per_year)

    if ci_method == "newey_west":
        if geometric:
            msg = (
                "Newey–West aplica al retorno medio (aritmético); para el geométrico "
                "usa el bootstrap, que no exige linealidad del estimador"
            )
            raise ValueError(msg)
        res = mean_significance(arr, horizon=horizon, alpha=alpha, label="retorno medio")
        return Estimate(
            name="annualized_return_arithmetic",
            value=res.estimate * periods_per_year,
            ci_low=res.ci_low * periods_per_year,
            ci_high=res.ci_high * periods_per_year,
            std_error=res.std_error * periods_per_year,
            method=res.method,
            alpha=alpha,
            n_obs=res.n_obs,
            notes=f"L={res.nw_lags}, n_eff={res.n_eff:.0f}",
        )

    return _bootstrap_estimate(
        "annualized_return_geometric" if geometric else "annualized_return_arithmetic",
        arr,
        stat,
        alpha=alpha,
        n_boot=n_boot,
        block_length=block_length,
        seed=seed,
        horizon=horizon,
    )


def annualized_volatility(
    returns: pd.Series | np.ndarray,
    *,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    alpha: float = 0.05,
    ci_method: Literal["bootstrap", "chi2"] = "bootstrap",
    n_boot: int = 1000,
    block_length: float | None = None,
    seed: int | None = 20260804,
    horizon: int = 1,
) -> Estimate:
    """Volatilidad anualizada `σ̂·√(ppy)` con intervalo de confianza.

    `ci_method="chi2"` usa el intervalo exacto bajo normalidad iid, que es
    **optimista** con retornos de curtosis alta: el bootstrap estacionario es el
    método por defecto porque no asume ni normalidad ni independencia, y los
    retornos de una estrategia long-short cumplen mal las dos.
    """
    arr = _clean_returns(returns)
    scale = math.sqrt(periods_per_year)

    if ci_method == "chi2":
        n = arr.size
        var = float(arr.var(ddof=1))
        lo_chi = float(stats.chi2.ppf(1.0 - alpha / 2.0, n - 1))
        hi_chi = float(stats.chi2.ppf(alpha / 2.0, n - 1))
        vol = math.sqrt(var) * scale
        return Estimate(
            name="annualized_volatility",
            value=vol,
            ci_low=math.sqrt((n - 1) * var / lo_chi) * scale,
            ci_high=math.sqrt((n - 1) * var / hi_chi) * scale,
            std_error=vol / math.sqrt(2.0 * (n - 1)),
            method="chi2_iid_normal",
            alpha=alpha,
            n_obs=n,
            notes="asume normalidad iid: optimista con curtosis alta",
        )

    return _bootstrap_estimate(
        "annualized_volatility",
        arr,
        lambda z: float(np.std(np.asarray(z, dtype=float), ddof=1) * scale),
        alpha=alpha,
        n_boot=n_boot,
        block_length=block_length,
        seed=seed,
        horizon=horizon,
    )


def drawdown_series(returns: pd.Series | np.ndarray, *, is_equity: bool = False) -> pd.Series:
    """Serie de *drawdown* (valores ≤ 0) frente al máximo acumulado."""
    if is_equity:
        equity = pd.Series(returns, dtype=float)
    else:
        r = pd.Series(returns, dtype=float)
        equity = (1.0 + r).cumprod()
    peak = equity.cummax()
    return (equity / peak - 1.0).rename("drawdown")


@dataclass(frozen=True, slots=True)
class DrawdownResult:
    """*Max drawdown* con su cronología y su duración."""

    max_drawdown: float
    """Negativo por convención: −0,32 significa una caída del 32 %."""
    peak_position: int
    trough_position: int
    recovery_position: int | None
    peak_label: object
    trough_label: object
    recovery_label: object | None
    drawdown_duration: int
    """Periodos de pico a valle."""
    recovery_duration: int | None
    """Periodos de valle a recuperación; `None` si nunca se recuperó."""
    underwater_duration: int | None
    """Periodos de pico a recuperación."""
    longest_underwater: int
    """El episodio bajo agua más largo de toda la muestra, se recuperase o no."""
    ci_low: float
    ci_high: float
    std_error: float
    method: str
    n_obs: int

    @property
    def recovered(self) -> bool:
        return self.recovery_position is not None

    def as_estimate(self) -> Estimate:
        return Estimate(
            name="max_drawdown",
            value=self.max_drawdown,
            ci_low=self.ci_low,
            ci_high=self.ci_high,
            std_error=self.std_error,
            method=self.method,
            n_obs=self.n_obs,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "max_drawdown": self.max_drawdown,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "peak": self.peak_label,
            "trough": self.trough_label,
            "recovery": self.recovery_label,
            "drawdown_duration": self.drawdown_duration,
            "recovery_duration": self.recovery_duration,
            "underwater_duration": self.underwater_duration,
            "longest_underwater": self.longest_underwater,
        }


def max_drawdown(
    returns: pd.Series | np.ndarray,
    *,
    is_equity: bool = False,
    alpha: float = 0.05,
    n_boot: int = 1000,
    block_length: float | None = None,
    seed: int | None = 20260804,
    horizon: int = 1,
) -> DrawdownResult:
    """*Max drawdown*, su cronología completa y su banda de error.

    El *max drawdown* es el estadístico de rendimiento **peor estimado** de todos:
    es un extremo, no una media, y su varianza muestral es enorme. Reportarlo sin
    banda invita a comparar dos estrategias por una diferencia que es ruido puro. La
    banda se obtiene por bootstrap estacionario, que preserva la dependencia serial
    —esencial aquí, porque un drawdown *es* dependencia serial: una racha.

    La duración importa tanto como la magnitud: una caída del 20 % que se recupera en
    dos meses y otra que tarda cuatro años son riesgos distintos para el mismo
    número. Por eso se devuelven `drawdown_duration`, `recovery_duration`,
    `underwater_duration` y `longest_underwater`.
    """
    series = pd.Series(returns, dtype=float).dropna()
    if series.size < 8:
        msg = f"solo {series.size} observaciones para estimar un drawdown"
        raise InsufficientHistory(msg)
    dd = drawdown_series(series, is_equity=is_equity)
    values = dd.to_numpy()
    trough_pos = int(np.argmin(values))
    mdd = float(values[trough_pos])

    equity = pd.Series(series, dtype=float) if is_equity else (1.0 + series).cumprod()
    eq = equity.to_numpy()
    peak_pos = int(np.argmax(eq[: trough_pos + 1])) if trough_pos > 0 else 0
    recovery_pos: int | None = None
    after = np.flatnonzero(eq[trough_pos:] >= eq[peak_pos])
    if after.size:
        recovery_pos = int(trough_pos + after[0])

    underwater = values < -1e-12
    longest = 0
    current = 0
    for flag in underwater:
        current = current + 1 if flag else 0
        longest = max(longest, current)

    arr = series.to_numpy(dtype=float)

    def stat(z: np.ndarray) -> float:
        z = np.asarray(z, dtype=float)
        curve = z if is_equity else np.cumprod(1.0 + z)
        return float(np.min(curve / np.maximum.accumulate(curve) - 1.0))

    boot = stationary_bootstrap(
        arr,
        statistic=stat,
        n_boot=n_boot,
        alpha=alpha,
        seed=seed,
        block_length=block_length,
        horizon=horizon,
    )

    labels = series.index
    return DrawdownResult(
        max_drawdown=mdd,
        peak_position=peak_pos,
        trough_position=trough_pos,
        recovery_position=recovery_pos,
        peak_label=labels[peak_pos],
        trough_label=labels[trough_pos],
        recovery_label=None if recovery_pos is None else labels[recovery_pos],
        drawdown_duration=int(trough_pos - peak_pos),
        recovery_duration=None if recovery_pos is None else int(recovery_pos - trough_pos),
        underwater_duration=None if recovery_pos is None else int(recovery_pos - peak_pos),
        longest_underwater=int(longest),
        ci_low=boot.ci_low,
        ci_high=boot.ci_high,
        std_error=boot.std_error,
        method=boot.method,
        n_obs=int(series.size),
    )


def calmar_ratio(
    returns: pd.Series | np.ndarray,
    *,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    alpha: float = 0.05,
    n_boot: int = 1000,
    block_length: float | None = None,
    seed: int | None = 20260804,
    horizon: int = 1,
) -> Estimate:
    """Calmar = retorno anualizado geométrico / |max drawdown|, con banda.

    Hereda la mala estimación del denominador (`max_drawdown`), así que su banda es
    ancha por construcción. Se reporta porque es la métrica que mira quien asigna
    capital, no porque sea estadísticamente sólida: comparar dos Calmar sin mirar
    sus intervalos es comparar dos ruidos.
    """
    arr = _clean_returns(returns)

    def stat(z: np.ndarray) -> float:
        z = np.asarray(z, dtype=float)
        curve = np.cumprod(1.0 + z)
        dd = float(np.min(curve / np.maximum.accumulate(curve) - 1.0))
        ann = float(np.exp(np.log1p(z).mean() * periods_per_year) - 1.0)
        if dd >= -1e-12:
            return float("nan")
        return ann / abs(dd)

    return _bootstrap_estimate(
        "calmar_ratio",
        arr,
        stat,
        alpha=alpha,
        n_boot=n_boot,
        block_length=block_length,
        seed=seed,
        horizon=horizon,
        notes="denominador extremo: banda ancha por construcción",
    )


# --------------------------------------------------------------------------- #
# Rotación y capacidad                                                          #
# --------------------------------------------------------------------------- #


def _drifted_weights(weights: pd.DataFrame, returns: pd.DataFrame | None) -> pd.DataFrame:
    """Pesos de la cartera justo **antes** de rebalancear, tras la deriva de precios.

    `w_drift_i = w_i·(1 + r_i) / (1 + Σ_j w_j·r_j)`. El denominador es el retorno del
    NAV, no la suma de pesos: así la fórmula sigue siendo correcta para una cartera
    long-short cuyos pesos suman ≈ 0.
    """
    prev = weights.shift(1)
    if returns is None:
        return prev
    r = returns.reindex(index=weights.index, columns=weights.columns).fillna(0.0)
    grown = prev * (1.0 + r)
    nav = 1.0 + (prev * r).sum(axis=1)
    nav = nav.where(nav.abs() > 1e-12, 1.0)
    return grown.div(nav, axis=0)


@dataclass(frozen=True, slots=True)
class TurnoverResult:
    """Rotación de la cartera, en fracción del NAV."""

    per_period: pd.Series = field(repr=False)
    one_way: Estimate
    annualized_one_way: float
    traded_notional_annual: float
    n_rebalances: int

    def to_dict(self) -> dict[str, object]:
        return {
            "one_way_per_rebalance": self.one_way.value,
            "ci_low": self.one_way.ci_low,
            "ci_high": self.one_way.ci_high,
            "annualized_one_way": self.annualized_one_way,
            "traded_notional_annual": self.traded_notional_annual,
            "n_rebalances": self.n_rebalances,
        }


def turnover(
    weights: pd.DataFrame,
    *,
    returns: pd.DataFrame | None = None,
    periods_per_year: float | None = None,
    alpha: float = 0.05,
    n_boot: int = 1000,
    seed: int | None = 20260804,
) -> TurnoverResult:
    """Rotación por rebalanceo y anualizada, con intervalo de confianza.

    Rotación **one-way** de un rebalanceo: `½·Σ_i |w_i(t) − w_i^deriva(t)|`, es decir
    la fracción del NAV que hay que comprar (o, simétricamente, vender). El notional
    total negociado es el doble.

    Restar la **deriva** y no el peso anterior importa: entre rebalanceos las
    posiciones se mueven solas con los precios, y contabilizar esa deriva como
    operación infla la rotación —y con ella los costes— sin motivo. En una cartera
    equiponderada de 500 nombres al 25 % de volatilidad, la diferencia es de varios
    puntos de rotación anual.

    La rotación es la variable que decide la Etapa 5 del protocolo: una señal cuya
    rotación necesaria supera la capacidad a `adv_participation = 0,05` se declara
    **muerta** por implementabilidad, por bueno que sea su `t`.
    """
    if not isinstance(weights, pd.DataFrame) or weights.empty:
        msg = "weights debe ser un DataFrame no vacío (index=fecha, columns=ticker)"
        raise DataQualityError(msg)
    w = weights.fillna(0.0).astype(float)
    drift = _drifted_weights(w, returns).fillna(0.0)
    delta = (w - drift).abs().sum(axis=1)
    per_period = (0.5 * delta).iloc[1:]
    if per_period.size < 2:
        msg = "hacen falta al menos 2 rebalanceos para medir rotación"
        raise InsufficientHistory(msg)

    values = per_period.to_numpy(dtype=float)
    if values.size >= 8:
        boot = stationary_bootstrap(values, n_boot=n_boot, alpha=alpha, seed=seed)
        est = Estimate(
            name="turnover_one_way",
            value=boot.value,
            ci_low=boot.ci_low,
            ci_high=boot.ci_high,
            std_error=boot.std_error,
            method=boot.method,
            alpha=alpha,
            n_obs=values.size,
        )
    else:  # muestra corta: intervalo normal, declarado como tal
        mean = float(values.mean())
        se = float(values.std(ddof=1) / math.sqrt(values.size))
        z = float(stats.norm.ppf(1.0 - alpha / 2.0))
        est = Estimate(
            name="turnover_one_way",
            value=mean,
            ci_low=mean - z * se,
            ci_high=mean + z * se,
            std_error=se,
            method="normal_small_sample",
            alpha=alpha,
            n_obs=values.size,
            notes="menos de 8 rebalanceos: intervalo normal, no bootstrap",
        )

    if periods_per_year is None and isinstance(w.index, pd.DatetimeIndex) and len(w) > 1:
        span_days = (w.index[-1] - w.index[0]).days
        periods_per_year = (
            365.25 * (len(w) - 1) / span_days if span_days > 0 else float(len(w) - 1)
        )
    ppy = float(periods_per_year) if periods_per_year is not None else float("nan")
    return TurnoverResult(
        per_period=per_period,
        one_way=est,
        annualized_one_way=float(est.value * ppy),
        traded_notional_annual=float(2.0 * est.value * ppy),
        n_rebalances=int(per_period.size),
    )


@dataclass(frozen=True, slots=True)
class CapacityResult:
    """Capacidad de la estrategia en dólares de AUM."""

    per_period: pd.Series = field(repr=False)
    median_capacity: Estimate
    p5_capacity: float
    binding_names: pd.Series = field(repr=False, default_factory=lambda: pd.Series(dtype=object))
    adv_participation: float = 0.05

    def to_dict(self) -> dict[str, object]:
        return {
            "median_capacity_usd": self.median_capacity.value,
            "ci_low": self.median_capacity.ci_low,
            "ci_high": self.median_capacity.ci_high,
            "p5_capacity_usd": self.p5_capacity,
            "adv_participation": self.adv_participation,
            "n_rebalances": int(self.per_period.size),
        }


def capacity(
    weights: pd.DataFrame,
    adv: pd.DataFrame,
    *,
    returns: pd.DataFrame | None = None,
    adv_participation: float = 0.05,
    adv_window: int | None = None,
    min_trade: float = 1e-6,
    alpha: float = 0.05,
    n_boot: int = 1000,
    seed: int | None = 20260804,
) -> CapacityResult:
    """Capacidad en AUM, limitada por la participación máxima en el volumen.

    Para cada rebalanceo, la operación en el nombre `i` es `|Δw_i|·AUM` y la
    restricción es `|Δw_i|·AUM ≤ p·ADV_i`, de donde::

        capacidad(t) = mín_i  p·ADV_i / |Δw_i|

    sobre los nombres efectivamente operados. La capacidad de la estrategia es el
    perfil temporal de ese mínimo: se reporta la **mediana con banda** y el
    **percentil 5**, porque la capacidad que importa no es la del día tranquilo sino
    la del día malo.

    `binding_names` identifica en cada fecha el nombre que ata la restricción. Suele
    ser el mismo puñado de valores ilíquidos, y esa información es directamente
    accionable: excluirlos o limitar su peso puede multiplicar la capacidad sin
    tocar la señal.

    `adv` debe ser volumen **en dólares** (precio × volumen), no en acciones. Con
    `adv_window` se toma la mediana móvil de esa ventana, que es más robusta a los
    picos de volumen de los días de resultados —justamente las fechas en las que este
    repo opera— que la media.
    """
    if not 0.0 < adv_participation <= 1.0:
        msg = f"adv_participation debe estar en (0,1]; recibido {adv_participation}"
        raise ValueError(msg)
    w = weights.fillna(0.0).astype(float)
    liquidity = adv.reindex(index=w.index, columns=w.columns).astype(float)
    if adv_window is not None:
        liquidity = liquidity.rolling(adv_window, min_periods=max(2, adv_window // 2)).median()
    if liquidity.isna().all().all():
        msg = "no hay volumen en dólares disponible para ninguna posición"
        raise InsufficientHistory(msg)

    drift = _drifted_weights(w, returns).fillna(0.0)
    trades = (w - drift).abs().iloc[1:]
    liq = liquidity.iloc[1:]
    limit = liq.mul(adv_participation).div(trades.where(trades > min_trade))
    per_period = limit.min(axis=1).dropna()
    if per_period.empty:
        msg = "ningún rebalanceo tiene operaciones por encima de `min_trade`"
        raise InsufficientHistory(msg)
    binding = limit.loc[per_period.index].idxmin(axis=1)

    values = per_period.to_numpy(dtype=float)
    if values.size >= 8:
        boot = stationary_bootstrap(
            values, statistic=lambda z: float(np.median(z)), n_boot=n_boot, alpha=alpha, seed=seed
        )
        est = Estimate(
            name="capacity_usd_median",
            value=boot.value,
            ci_low=boot.ci_low,
            ci_high=boot.ci_high,
            std_error=boot.std_error,
            method=boot.method,
            alpha=alpha,
            n_obs=values.size,
        )
    else:
        med = float(np.median(values))
        est = Estimate(
            name="capacity_usd_median",
            value=med,
            ci_low=float(np.min(values)),
            ci_high=float(np.max(values)),
            std_error=float("nan"),
            method="range_small_sample",
            alpha=alpha,
            n_obs=values.size,
            notes="menos de 8 rebalanceos: se reporta el rango observado",
        )

    return CapacityResult(
        per_period=per_period,
        median_capacity=est,
        p5_capacity=float(np.percentile(values, 5)),
        binding_names=binding,
        adv_participation=float(adv_participation),
    )


# --------------------------------------------------------------------------- #
# Ley fundamental: test de sanidad                                              #
# --------------------------------------------------------------------------- #


def effective_breadth(n_bets: int, rho: float) -> float:
    """Amplitud efectiva `BR_eff = N / [1 + (N−1)·ρ]`.

    Aplicar la ley fundamental con `BR` bruta es el error nº 14 de §15: con
    `BR = 52 × 475 = 24.700` y una IC de 0,03 sale un `IR` de **4,71**, que nadie ha
    obtenido jamás. Las 475 apuestas no son independientes: comparten factores de
    mercado y sector. Con `ρ = 0,05` la amplitud efectiva cae a **19,2** y el `IR` a
    0,95, que sí es una cifra creíble para un factor bueno.
    """
    if n_bets < 1:
        msg = f"n_bets debe ser >= 1; recibido {n_bets}"
        raise ValueError(msg)
    denom = 1.0 + (n_bets - 1) * rho
    if denom <= 0:
        msg = f"1 + (N−1)·ρ = {denom} no es positivo"
        raise DataQualityError(msg)
    return float(n_bets / denom)


def fundamental_law_sharpe(
    ic: float,
    *,
    n_bets: int,
    rebalances_per_year: int,
    rho: float = 0.05,
    transfer_coefficient: float = 0.5,
) -> float:
    """`IR = IC·√BR_eff·TC` de Grinold (1989) y Clarke–de Silva–Thorley (2002).

    **Regla de coherencia del repo (§11.4):** si el Sharpe del backtest supera en más
    de un factor 2 al que predice esta fórmula con `ρ` estimado de los datos, **hay
    un error en el backtest** —look-ahead, costes ausentes o sesgo de
    supervivencia—. Es un test de sanidad barato que atrapa la mayoría de los fallos
    graves antes que cualquier test estadístico, y es el criterio 19 de §14.1.

    `transfer_coefficient` es la correlación entre las posiciones ideales y las
    realmente tomadas tras `max_weight`, `adv_participation` y la prohibición de
    cortos. Un `TC = 0,5` es realista; `TC = 1` solo existe sobre el papel.
    """
    br_eff = effective_breadth(n_bets, rho) * rebalances_per_year
    return float(ic * math.sqrt(br_eff) * transfer_coefficient)


# --------------------------------------------------------------------------- #
# Resumen integrado                                                             #
# --------------------------------------------------------------------------- #


def performance_summary(
    returns: pd.Series | np.ndarray,
    *,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    risk_free: float = 0.0,
    alpha: float = 0.05,
    n_boot: int = 1000,
    seed: int | None = 20260804,
    horizon: int = 1,
    trial_sharpes: Sequence[float] | np.ndarray | None = None,
    n_trials: int | None = None,
    sr_variance: float | None = None,
    weights: pd.DataFrame | None = None,
    adv: pd.DataFrame | None = None,
    asset_returns: pd.DataFrame | None = None,
    adv_participation: float = 0.05,
) -> pd.DataFrame:
    """Tabla de todas las métricas con su intervalo de confianza.

    Es el bloque que consume `reports` para el tearsheet. Cada fila lleva método y
    banda; ninguna métrica sale de aquí como número desnudo, que es exactamente lo
    que prohíbe `ARCHITECTURE.md` §3.8.

    El DSR aparece **solo** si se suministran los Sharpes de las pruebas o su
    recuento: no se inventa un `N`.
    """
    rows: list[dict[str, object]] = []

    ret = annualized_return(
        returns,
        periods_per_year=periods_per_year,
        alpha=alpha,
        n_boot=n_boot,
        seed=seed,
        horizon=horizon,
    )
    vol = annualized_volatility(
        returns,
        periods_per_year=periods_per_year,
        alpha=alpha,
        n_boot=n_boot,
        seed=seed,
        horizon=horizon,
    )
    shp = sharpe_ratio(
        returns,
        risk_free=risk_free,
        periods_per_year=periods_per_year,
        alpha=alpha,
        horizon=horizon,
    )
    shp_boot = sharpe_ratio(
        returns,
        risk_free=risk_free,
        periods_per_year=periods_per_year,
        alpha=alpha,
        ci_method="bootstrap",
        n_boot=n_boot,
        seed=seed,
        horizon=horizon,
    )
    dd = max_drawdown(returns, alpha=alpha, n_boot=n_boot, seed=seed, horizon=horizon)
    cal = calmar_ratio(
        returns, periods_per_year=periods_per_year, alpha=alpha, n_boot=n_boot, seed=seed
    )

    rows.append(ret.to_dict())
    rows.append(vol.to_dict())
    rows.append(shp.as_estimate().to_dict())
    boot_row = shp_boot.as_estimate().to_dict()
    boot_row["metric"] = "sharpe_annualized_bootstrap"
    rows.append(boot_row)
    rows.append(dd.as_estimate().to_dict())
    rows.append(cal.to_dict())
    rows.append(
        {
            "metric": "psr_zero",
            "value": shp.psr_zero,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "std_error": np.nan,
            "method": "bailey_lopez_de_prado_psr",
            "n_obs": shp.n_obs,
        }
    )
    rows.append(
        {
            "metric": "min_track_record_obs",
            "value": shp.min_track_record,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "std_error": np.nan,
            "method": "min_trl",
            "n_obs": shp.n_obs,
        }
    )

    if trial_sharpes is not None or sr_variance is not None:
        dsr = deflated_sharpe_ratio(
            returns,
            n_trials=n_trials,
            trial_sharpes=trial_sharpes,
            sr_variance=sr_variance,
            periods_per_year=periods_per_year,
            risk_free=risk_free,
        )
        rows.append(
            {
                "metric": "deflated_sharpe_ratio",
                "value": dsr.dsr,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "std_error": np.nan,
                "method": f"dsr_N={dsr.n_trials}",
                "n_obs": dsr.n_obs,
            }
        )

    if weights is not None:
        tv = turnover(
            weights,
            returns=asset_returns,
            alpha=alpha,
            n_boot=n_boot,
            seed=seed,
        )
        row = tv.one_way.to_dict()
        rows.append(row)
        rows.append(
            {
                "metric": "turnover_annualized_one_way",
                "value": tv.annualized_one_way,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "std_error": np.nan,
                "method": tv.one_way.method,
                "n_obs": tv.n_rebalances,
            }
        )
        if adv is not None:
            cap = capacity(
                weights,
                adv,
                returns=asset_returns,
                adv_participation=adv_participation,
                alpha=alpha,
                n_boot=n_boot,
                seed=seed,
            )
            rows.append(cap.median_capacity.to_dict())
            rows.append(
                {
                    "metric": "capacity_usd_p5",
                    "value": cap.p5_capacity,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "std_error": np.nan,
                    "method": "empirical_percentile",
                    "n_obs": int(cap.per_period.size),
                }
            )

    return pd.DataFrame(rows).set_index("metric")
