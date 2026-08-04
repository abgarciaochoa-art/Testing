"""Transformaciones cross-section de señales (contrato `ARCHITECTURE.md` §3.6).

Todo lo que hay aquí opera **por fecha, sobre la sección cruzada**, sobre paneles
canónicos con MultiIndex ``(date, ticker)``. Esa restricción no es estética: es la
que separa un preprocesado point-in-time de un look-ahead silencioso. Winsorizar o
estandarizar con los momentos de *todo* el panel usa la distribución futura de la
señal para decidir qué es un outlier hoy; es el error catalogado como §12.7 en
`docs/research/pit_and_biases.md` y el que caza el checklist §24 del mismo
documento ("toda normalización es por fecha").

Contenido
---------

* `zscore`, `winsorize`, `rank_pct`: acondicionamiento básico.
* `demean` / `demedian`: centrado por grupo (sector, sub-industria...).
* `residualize`: residuo de una regresión cross-section por fecha, resuelta por
  `lstsq` (SVD) y no por inversión explícita de ``X'X``.
* `neutralize`: la anterior con construcción automática del diseño
  ``X = [1, dummies(sector), log(MC), beta, ...]`` a partir de exposiciones
  categóricas y numéricas.
* `condition_factor`: la receta canónica §1.4 de
  `docs/research/fundamental_factors.md` (winsorizar → z → neutralizar → z).

Política de NaN
---------------
El repo prohíbe el relleno silencioso. Cada función expone `nan_policy`
(`NaNPolicy`) y el valor por defecto es `"propagate"`: un dato ausente entra como
NaN, se excluye del cálculo de los momentos de esa fecha y **sale como NaN**. Los
rellenos (`"neutral"`, `"zero"`, `"mean"`, `"median"`) existen porque a veces son
la decisión correcta, pero hay que pedirlos por su nombre.

Se distinguen dos causas de NaN, y se gobiernan con parámetros distintos:

1. **Dato ausente en la entrada** → `nan_policy`.
2. **Sección cruzada insuficiente** en esa fecha (`min_obs`, varianza nula, grados
   de libertad insuficientes) → `min_obs` / `on_insufficient`. Un relleno de
   `nan_policy` **nunca** tapa una fecha insuficiente: se seguiría propagando NaN.
   Si *ninguna* fecha del panel supera el mínimo, se lanza `InsufficientHistory`,
   porque devolver un panel entero de NaN en silencio es exactamente lo que la
   regla 5 del proyecto prohíbe.

Referencias
-----------
- Tukey, J. W. (1962), *The Future of Data Analysis*: winsorización como
  estimación robusta frente a colas pesadas.
- Rousseeuw, P. y Croux, C. (1993), *Alternatives to the Median Absolute
  Deviation*, JASA 88: constante 1.4826 que hace la MAD consistente con sigma bajo
  normalidad.
- Blom, G. (1958) y van der Waerden (1952): puntuaciones normales por rango
  (`rank_pct(mode="normal")`).
- Fama, E. y MacBeth, J. (1973), *Risk, Return and Equilibrium*: la regresión
  cross-section por fecha, que es el esqueleto de `residualize`/`neutralize`.
- Grinold, R. y Kahn, R. (2000), *Active Portfolio Management*, cap. 3-4:
  estandarización y neutralización de puntuaciones antes de combinarlas.
- Golub, G. y Van Loan, C. (2013), *Matrix Computations* §5.5: solución por SVD
  de sistemas mal condicionados; con dummies sectoriales casi colineales la
  inversión explícita de ``X'X`` es numéricamente inaceptable.
- `docs/research/fundamental_factors.md` §1.4 y §15.2 (receta y test de
  neutralización a precisión de máquina), §9.4 (por qué el sector es obligatorio).
- `docs/research/pit_and_biases.md` §12.7 y §12.9 (normalización global = futuro;
  sector GICS actual aplicado hacia atrás = look-ahead leve pero real).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from itertools import pairwise
from typing import Literal, cast

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

from earnings_alpha.errors import DataQualityError, InsufficientHistory

__all__ = [  # noqa: RUF022 - orden temático, no alfabético
    "NaNPolicy",
    "InsufficientPolicy",
    "MAD_TO_SIGMA",
    "check_panel",
    "align_panel_like",
    "cross_section_count",
    "zscore",
    "winsorize",
    "rank_pct",
    "demean",
    "demedian",
    "residualize",
    "neutralize",
    "condition_factor",
]

MAD_TO_SIGMA = 1.482602218505602
"""Constante que hace la MAD consistente con sigma bajo normalidad (1/Φ⁻¹(0.75)).

Rousseeuw y Croux (1993). Sin ella, un "z robusto" no es comparable con un z
clásico y los umbrales de winsorización cambian de significado.
"""


class NaNPolicy(StrEnum):
    """Qué hacer con las observaciones ausentes de la sección cruzada.

    El valor por defecto de todas las funciones del módulo es `PROPAGATE`. Los
    rellenos existen, pero hay que solicitarlos explícitamente: un relleno con
    cero no declarado convierte "no sé" en "exactamente la media", que en un
    ranking cross-section es una posición central *fabricada*.
    """

    PROPAGATE = "propagate"
    """NaN entra, NaN sale. La observación no participa en los momentos ni en los
    rangos de su fecha, y tampoco aparece en la salida con un valor inventado."""

    RAISE = "raise"
    """Cualquier NaN en la entrada lanza `DataQualityError`. Útil cuando el
    llamante ya ha filtrado el universo y un NaN indica un bug aguas arriba."""

    DROP = "drop"
    """Las filas con entrada ausente se eliminan del índice de salida. El panel
    resultante deja de ser rectangular; es lo correcto cuando la señal se pasa
    directamente a un constructor de cartera."""

    NEUTRAL = "neutral"
    """Rellena la salida con el valor neutro de la transformación (0 para un
    z-score, 0.5 para un `rank_pct` uniforme, la mediana de la fecha para una
    winsorización). Equivale a decir "sin opinión sobre este nombre"."""

    ZERO = "zero"
    """Rellena la salida con 0.0 literal, sea cual sea la escala. **Ojo**: en
    `rank_pct` uniforme 0.0 es el *peor* rango, no la ausencia de opinión."""

    MEAN = "mean"
    """Imputa la **entrada** con la media de los válidos de esa fecha (o grupo)
    antes de transformar. Reduce la dispersión cross-section y hace indistinguible
    lo imputado de lo observado; se documenta aquí para que sea una decisión y no
    un accidente."""

    MEDIAN = "median"
    """Como `MEAN`, con la mediana. Más robusta a colas, mismo efecto de
    compresión de la dispersión."""


InsufficientPolicy = Literal["nan", "raise"]
"""Qué hacer con una fecha cuya sección cruzada no llega al mínimo exigido.

``"nan"`` (por defecto) anula esa fecha; ``"raise"`` lanza `InsufficientHistory`.
Si *ninguna* fecha llega al mínimo se lanza `InsufficientHistory` en ambos casos.
"""

_DATE = "date"
_TICKER = "ticker"
_OTHER_GROUP = "__other__"


# ---------------------------------------------------------------------------
# Validación y utilidades de panel
# ---------------------------------------------------------------------------


def check_panel(
    x: pd.Series | pd.DataFrame,
    *,
    name: str = "x",
    allow_duplicates: bool = False,
) -> None:
    """Valida que `x` es un panel canónico con MultiIndex ``(date, ticker)``.

    Comprueba tres cosas que, de fallar, corrompen en silencio cualquier cálculo
    cross-section: que hay exactamente dos niveles, que el primero es de tipo
    fecha (invertir el orden de los niveles es el error más común y produce
    "secciones cruzadas" de un solo elemento) y que no hay pares duplicados.
    """
    idx = x.index
    if not isinstance(idx, pd.MultiIndex) or idx.nlevels != 2:
        msg = (
            f"`{name}` debe tener MultiIndex de dos niveles (date, ticker); "
            f"recibido {type(idx).__name__} con {idx.nlevels} nivel(es)"
        )
        raise DataQualityError(msg)
    if not is_datetime64_any_dtype(idx.get_level_values(0)):
        msg = (
            f"el primer nivel del índice de `{name}` debe ser de tipo fecha; "
            f"es {idx.get_level_values(0).dtype}. ¿Están los niveles invertidos "
            "(ticker, date)? Ese error produce secciones cruzadas de un elemento."
        )
        raise DataQualityError(msg)
    if not allow_duplicates and idx.has_duplicates:
        dupes = idx[idx.duplicated()].unique()
        msg = (
            f"`{name}` tiene {len(dupes)} pares (date, ticker) duplicados; "
            f"muestra: {list(dupes[:5])}"
        )
        raise DataQualityError(msg)


def _as_float_series(x: pd.Series, name: str) -> pd.Series:
    if not isinstance(x, pd.Series):
        msg = f"`{name}` debe ser una Series con MultiIndex (date, ticker)"
        raise DataQualityError(msg)
    check_panel(x, name=name)
    if not is_numeric_dtype(x):
        msg = f"`{name}` debe ser numérica; su dtype es {x.dtype}"
        raise DataQualityError(msg)
    if len(x) == 0:
        msg = (
            f"`{name}` está vacía. Transformar un panel vacío devolvería otro panel "
            "vacío sin que nadie se entere; la regla 5 del proyecto lo prohíbe."
        )
        raise InsufficientHistory(msg)
    return x.astype("float64")


def align_panel_like(
    obj: pd.Series | pd.DataFrame | Mapping[str, object] | Sequence[object] | np.ndarray,
    index: pd.MultiIndex,
    *,
    name: str = "by",
) -> pd.Series | pd.DataFrame:
    """Alinea exposiciones o etiquetas de grupo al índice de un panel.

    Acepta cuatro formas, porque en la práctica conviven las cuatro:

    - `Series`/`DataFrame` con MultiIndex ``(date, ticker)``: se reindexa. Es la
      forma **point-in-time correcta**, la única que permite que el sector o el
      tamaño de un nombre cambien con el tiempo.
    - `Series`/`DataFrame` indexado solo por ticker: valor **estático**, se
      difunde a todas las fechas. Cómodo y peligroso: aplicar la clasificación
      GICS de hoy a 2005 mete look-ahead (GICS creó Real Estate en 2016 y
      Communication Services en 2018; `pit_and_biases.md` §12.9). Admisible como
      limitación declarada, nunca como descuido.
    - `Mapping` ticker → valor: idéntico al caso anterior.
    - Array o secuencia de la misma longitud que `index`: se usa tal cual.
    """
    if isinstance(obj, pd.DataFrame):
        if isinstance(obj.index, pd.MultiIndex):
            check_panel(obj, name=name)
            return obj.reindex(index)
        return obj.reindex(index.get_level_values(1)).set_axis(index)
    if isinstance(obj, pd.Series):
        if isinstance(obj.index, pd.MultiIndex):
            check_panel(obj, name=name)
            return obj.reindex(index)
        return pd.Series(
            obj.reindex(index.get_level_values(1)).to_numpy(), index=index, name=obj.name or name
        )
    if isinstance(obj, Mapping):
        mapped = pd.Index(index.get_level_values(1)).map(lambda t: obj.get(t, None))
        return pd.Series(np.asarray(mapped, dtype=object), index=index, name=name)
    arr = np.asarray(obj)
    if arr.ndim != 1 or len(arr) != len(index):
        msg = (
            f"`{name}` como array debe ser unidimensional y de longitud {len(index)}; "
            f"recibido shape={arr.shape}"
        )
        raise DataQualityError(msg)
    return pd.Series(arr, index=index, name=name)


def _group_labels(
    x: pd.Series,
    by: object | None,
    *,
    missing_group: Literal["nan", "pool", "raise"],
) -> np.ndarray | None:
    """Resuelve las etiquetas de grupo alineadas con `x` (o None si no hay grupo)."""
    if by is None:
        return None
    if isinstance(by, str):
        names = [str(n) for n in x.index.names]
        if by not in names:
            msg = (
                f"`by={by!r}` no es un nivel del índice (niveles: {names}); pásalo como "
                "Series, Mapping o array si es una exposición externa"
            )
            raise DataQualityError(msg)
        labels = pd.Series(x.index.get_level_values(by), index=x.index)
    else:
        aligned = align_panel_like(by, x.index, name="by")
        if isinstance(aligned, pd.DataFrame):
            if aligned.shape[1] != 1:
                msg = "`by` como DataFrame debe tener exactamente una columna de etiquetas"
                raise DataQualityError(msg)
            aligned = aligned.iloc[:, 0]
        labels = aligned
    values = labels.to_numpy(dtype=object, copy=True)
    missing = pd.isna(values)
    if missing.any():
        if missing_group == "raise":
            msg = f"{int(missing.sum())} observaciones sin etiqueta de grupo en `by`"
            raise DataQualityError(msg)
        if missing_group == "pool":
            values[missing] = _OTHER_GROUP
        # "nan": se dejan como NaN; groupby(dropna=True) las excluirá y saldrán NaN.
    return values


def _groupers(x: pd.Series, labels: np.ndarray | None) -> list[np.ndarray]:
    dates = x.index.get_level_values(0).to_numpy()
    return [dates] if labels is None else [dates, labels]


def cross_section_count(
    x: pd.Series,
    by: object | None = None,
    *,
    missing_group: Literal["nan", "pool", "raise"] = "nan",
) -> pd.Series:
    """Número de valores no nulos por fecha (o por fecha y grupo).

    Diagnóstico de primera línea: la mayoría de las sorpresas al interpretar un
    factor se explican mirando cuántos nombres tenía de verdad la sección cruzada
    en cada fecha.
    """
    s = _as_float_series(x, "x")
    labels = _group_labels(s, by, missing_group=missing_group)
    counts = s.groupby(_groupers(s, labels)).count()
    counts.name = "n_valid"
    if labels is None:
        counts.index.name = _DATE
    else:
        counts.index.names = [_DATE, "group"]
    return counts


# ---------------------------------------------------------------------------
# Motor común: política de NaN + mínimos por sección cruzada
# ---------------------------------------------------------------------------


def _parse_nan_policy(policy: NaNPolicy | str) -> NaNPolicy:
    try:
        return NaNPolicy(policy)
    except ValueError as exc:
        valid = ", ".join(p.value for p in NaNPolicy)
        msg = f"nan_policy desconocida: {policy!r}. Válidas: {valid}"
        raise DataQualityError(msg) from exc


def _pre_fill(s: pd.Series, groupers: list[np.ndarray], policy: NaNPolicy) -> pd.Series:
    """Aplica la parte de la política que actúa sobre la **entrada**."""
    na = s.isna()
    if policy is NaNPolicy.RAISE and bool(na.any()):
        msg = (
            f"{int(na.sum())} valores ausentes con nan_policy='raise'. "
            "Muestra: " + str(list(s.index[na][:5]))
        )
        raise DataQualityError(msg)
    if policy in (NaNPolicy.MEAN, NaNPolicy.MEDIAN) and bool(na.any()):
        how = "mean" if policy is NaNPolicy.MEAN else "median"
        filler = s.groupby(groupers).transform(how)
        return s.where(~na, filler)
    return s


def _post_fill(
    out: pd.Series,
    input_na: pd.Series,
    policy: NaNPolicy,
    neutral: float | pd.Series,
) -> pd.Series:
    """Aplica la parte de la política que actúa sobre la **salida**.

    Dos salvaguardas:

    1. Solo se rellenan filas cuya *entrada* era nula. Un NaN nacido de la
       transformación (varianza cero, grados de libertad insuficientes) no es un
       dato ausente y no se tapa.
    2. Solo se rellenan fechas que **produjeron alguna señal**. Si una fecha se
       anuló entera por `min_obs`, rellenar sus huecos con el valor neutro
       fabricaría una sección cruzada donde no la hay: quedaría un día en el que
       todos los nombres valen 0 y ninguno vale otra cosa, indistinguible de un
       día real de dispersión nula.
    """
    if policy is NaNPolicy.DROP:
        return out[~input_na.to_numpy()]
    if policy in (NaNPolicy.NEUTRAL, NaNPolicy.ZERO):
        fill_value = 0.0 if policy is NaNPolicy.ZERO else neutral
        produced = (
            out.notna().astype("float64").groupby(out.index.get_level_values(0)).transform("max")
            > 0
        )
        target = input_na.to_numpy() & out.isna().to_numpy() & produced.to_numpy()
        if isinstance(fill_value, pd.Series):
            out = out.where(~target, fill_value)
        else:
            out = out.where(~target, float(fill_value))
    return out


def _enforce_min_obs(
    out: pd.Series,
    n_valid: pd.Series,
    min_obs: int,
    on_insufficient: InsufficientPolicy,
    *,
    what: str,
) -> pd.Series:
    """Anula (o hace fallar) las secciones cruzadas por debajo del mínimo."""
    if min_obs <= 0:
        return out
    short = n_valid.to_numpy() < min_obs
    if not short.any():
        return out
    if not (~short).any():
        msg = (
            f"{what}: ninguna sección cruzada alcanza min_obs={min_obs} "
            f"(máximo observado: {int(np.nanmax(n_valid.to_numpy()))}). "
            "Devolver un panel entero de NaN en silencio está prohibido."
        )
        raise InsufficientHistory(msg)
    if on_insufficient == "raise":
        bad = out.index[short].get_level_values(0).unique()
        msg = (
            f"{what}: {len(bad)} fechas por debajo de min_obs={min_obs} "
            f"(muestra: {[str(d.date()) for d in bad[:5]]})"
        )
        raise InsufficientHistory(msg)
    return out.mask(short)


def _valid_counts(s: pd.Series, groupers: list[np.ndarray]) -> pd.Series:
    """Cuenta de válidos **de la entrada original**, difundida a cada fila.

    Se calcula antes de cualquier imputación: los valores imputados por
    `nan_policy` no cuentan para `min_obs`, que mide cobertura real.
    """
    return s.notna().astype("float64").groupby(groupers).transform("sum")


# ---------------------------------------------------------------------------
# Transformaciones básicas
# ---------------------------------------------------------------------------


def zscore(
    x: pd.Series | pd.DataFrame,
    by: object | None = None,
    *,
    robust: bool = False,
    ddof: int = 1,
    clip: float | None = None,
    nan_policy: NaNPolicy | str = NaNPolicy.PROPAGATE,
    min_obs: int = 3,
    on_insufficient: InsufficientPolicy = "nan",
    constant: Literal["nan", "zero"] = "nan",
    missing_group: Literal["nan", "pool", "raise"] = "nan",
) -> pd.Series | pd.DataFrame:
    """Estandariza la sección cruzada de cada fecha: ``(x - μ_t) / sigma_t``.

    Con `by` se estandariza dentro de cada grupo de la fecha (sector, por
    ejemplo), lo que equivale a una neutralización sectorial de media *y*
    varianza; es más agresivo que `demean` y conviene solo cuando los grupos
    tienen tamaño suficiente (`fundamental_factors.md` §1.5: con 21 nombres en
    Energy el sigma intra-sector es ya ruidoso).

    Parámetros
    ----------
    robust:
        Usa mediana y MAD escalada (`MAD_TO_SIGMA`) en vez de media y desviación
        típica. Recomendado cuando la señal no se ha winsorizado antes: un único
        outlier de |z|>20 —habitual en `E/P` o en accruals— infla sigma y comprime
        toda la sección cruzada hacia cero (`fundamental_factors.md` §1.4).
    clip:
        Recorte simétrico posterior en unidades de z (p. ej. 3.0). Es una
        winsorización expresada en la escala estandarizada.
    constant:
        Qué devolver si la sección cruzada es constante (sigma = 0): `"nan"` por
        defecto —no hay información transversal que estandarizar— o `"zero"`.

    Devuelve
    --------
    Panel de la misma forma; media 0 y desviación típica 1 por fecha (o por
    fecha y grupo) sobre las observaciones válidas.
    """
    if isinstance(x, pd.DataFrame):
        return _columnwise(
            zscore,
            x,
            by,
            robust=robust,
            ddof=ddof,
            clip=clip,
            nan_policy=nan_policy,
            min_obs=min_obs,
            on_insufficient=on_insufficient,
            constant=constant,
            missing_group=missing_group,
        )

    policy = _parse_nan_policy(nan_policy)
    s = _as_float_series(x, "x")
    labels = _group_labels(s, by, missing_group=missing_group)
    groupers = _groupers(s, labels)

    input_na = s.isna()
    n_valid = _valid_counts(s, groupers)
    work = _pre_fill(s, groupers, policy)
    grouped = work.groupby(groupers)

    if robust:
        center = grouped.transform("median")
        dev = (work - center).abs()
        scale = MAD_TO_SIGMA * dev.groupby(groupers).transform("median")
    else:
        center = grouped.transform("mean")
        scale = grouped.transform("std", ddof=ddof)

    degenerate = scale.isna().to_numpy() | np.isclose(scale.to_numpy(), 0.0, atol=0.0)
    safe_scale = scale.mask(degenerate)
    out = (work - center) / safe_scale
    if constant == "zero":
        out = out.mask(degenerate & work.notna().to_numpy(), 0.0)
    if clip is not None:
        if clip <= 0:
            msg = f"clip debe ser > 0; recibido {clip}"
            raise ValueError(msg)
        out = out.clip(-clip, clip)

    out = _enforce_min_obs(out, n_valid, min_obs, on_insufficient, what="zscore")
    out = _post_fill(out, input_na, policy, 0.0)
    out.name = x.name
    return out


def winsorize(
    x: pd.Series | pd.DataFrame,
    q: float = 0.01,
    by: object | None = None,
    *,
    upper_q: float | None = None,
    method: Literal["quantile", "mad", "sigma"] = "quantile",
    k: float = 3.0,
    mode: Literal["clip", "drop"] = "clip",
    nan_policy: NaNPolicy | str = NaNPolicy.PROPAGATE,
    min_obs: int = 5,
    on_insufficient: InsufficientPolicy = "nan",
    missing_group: Literal["nan", "pool", "raise"] = "nan",
) -> pd.Series | pd.DataFrame:
    """Acota las colas de cada sección cruzada.

    Winsorizar **antes** de estandarizar, nunca al revés: al revés, un solo
    outlier infla sigma y comprime el resto de la sección cruzada
    (`fundamental_factors.md` §1.4). El valor por defecto `q=0.01` es el 1 %/99 %
    de la receta canónica del repo.

    Parámetros
    ----------
    q, upper_q:
        Cuantiles inferior y superior. Si `upper_q` es None se usa ``1 - q``
        (recorte simétrico).
    method:
        - ``"quantile"``: recorta a los cuantiles empíricos de la fecha. No
          asume forma de la distribución; con secciones cruzadas pequeñas los
          cuantiles extremos son ruidosos.
        - ``"mad"``: recorta a ``mediana ± k·MAD_escalada``. Robusto y estable
          con pocos nombres (Rousseeuw y Croux 1993).
        - ``"sigma"``: recorta a ``media ± k·sigma``. El menos robusto: los propios
          outliers desplazan los límites.
    mode:
        ``"clip"`` (winsorización propiamente dicha) sustituye el valor extremo
        por el límite; ``"drop"`` lo convierte en NaN (truncamiento). El segundo
        pierde información pero evita acumular masa en los límites, que es lo que
        hace que los deciles extremos de un factor winsorizado estén poblados por
        empates.

    Devuelve
    --------
    Panel de la misma forma, con los extremos acotados. La transformación es
    **monótona no decreciente** dentro de cada fecha, luego no altera el orden ni,
    por tanto, la rank IC.
    """
    if isinstance(x, pd.DataFrame):
        return _columnwise(
            winsorize,
            x,
            q,
            by,
            upper_q=upper_q,
            method=method,
            k=k,
            mode=mode,
            nan_policy=nan_policy,
            min_obs=min_obs,
            on_insufficient=on_insufficient,
            missing_group=missing_group,
        )

    hi_q = 1.0 - q if upper_q is None else upper_q
    if not 0.0 <= q < hi_q <= 1.0:
        msg = f"cuantiles inválidos: q={q}, upper_q={hi_q}; se exige 0 <= q < upper_q <= 1"
        raise ValueError(msg)
    if method in ("mad", "sigma") and k <= 0:
        msg = f"k debe ser > 0; recibido {k}"
        raise ValueError(msg)

    policy = _parse_nan_policy(nan_policy)
    s = _as_float_series(x, "x")
    labels = _group_labels(s, by, missing_group=missing_group)
    groupers = _groupers(s, labels)

    input_na = s.isna()
    n_valid = _valid_counts(s, groupers)
    work = _pre_fill(s, groupers, policy)
    grouped = work.groupby(groupers)

    if method == "quantile":
        lo = grouped.transform("quantile", q)
        hi = grouped.transform("quantile", hi_q)
    elif method == "mad":
        center = grouped.transform("median")
        scale = MAD_TO_SIGMA * (work - center).abs().groupby(groupers).transform("median")
        lo, hi = center - k * scale, center + k * scale
    elif method == "sigma":
        center = grouped.transform("mean")
        scale = grouped.transform("std", ddof=1)
        lo, hi = center - k * scale, center + k * scale
    else:  # pragma: no cover - protegido por el tipo
        msg = f"método de winsorización desconocido: {method!r}"
        raise ValueError(msg)

    if mode == "clip":
        out = work.clip(lower=lo, upper=hi)
    elif mode == "drop":
        out = work.mask((work < lo) | (work > hi))
    else:  # pragma: no cover - protegido por el tipo
        msg = f"modo desconocido: {mode!r}"
        raise ValueError(msg)

    out = _enforce_min_obs(out, n_valid, min_obs, on_insufficient, what="winsorize")
    neutral = work.groupby(groupers).transform("median")
    out = _post_fill(out, input_na, policy, neutral)
    out.name = x.name
    return out


def rank_pct(
    x: pd.Series | pd.DataFrame,
    by: object | None = None,
    *,
    ascending: bool = True,
    method: Literal["average", "min", "max", "first", "dense"] = "average",
    mode: Literal["uniform", "centered", "normal"] = "uniform",
    nan_policy: NaNPolicy | str = NaNPolicy.PROPAGATE,
    min_obs: int = 3,
    on_insufficient: InsufficientPolicy = "nan",
    missing_group: Literal["nan", "pool", "raise"] = "nan",
) -> pd.Series | pd.DataFrame:
    """Rango percentil dentro de la sección cruzada de cada fecha.

    Es la transformación más importante del módulo para este proyecto: descarta
    la magnitud y conserva solo el orden, y por eso es **invariante a cualquier
    transformación monótona creciente** de la señal. Esa invariancia es la razón
    por la que la rank IC es la métrica principal del repo
    (`validation_methodology.md` §2.1): no cambia si se winsoriza al 1 % o al
    2.5 %, mientras que la IC de Pearson sí.

    Parámetros
    ----------
    mode:
        - ``"uniform"``: ``(rango - 0.5) / n`` ∈ (0, 1). Se usa el desplazamiento
          de medio rango para que el resultado sea simétrico y no toque nunca 0
          ni 1, cosa que rompería un `logit` o una `Φ⁻¹` posteriores.
        - ``"centered"``: la anterior reescalada a (-1, 1), media 0.
        - ``"normal"``: ``Φ⁻¹`` del uniforme, es decir puntuaciones normales por
          rango (van der Waerden 1952; Blom 1958). Da una señal gaussiana por
          construcción, ideal para combinar linealmente factores con colas muy
          distintas.
    method:
        Tratamiento de empates, tal cual lo define `pandas.Series.rank`. Con
        factores discretos y muchos empates (el F-score de Piotroski toma 10
        valores) ``"average"`` es lo correcto; ``"first"`` rompería empates por
        orden alfabético de ticker, que es una fuente de sesgo silencioso.
    """
    if isinstance(x, pd.DataFrame):
        return _columnwise(
            rank_pct,
            x,
            by,
            ascending=ascending,
            method=method,
            mode=mode,
            nan_policy=nan_policy,
            min_obs=min_obs,
            on_insufficient=on_insufficient,
            missing_group=missing_group,
        )

    policy = _parse_nan_policy(nan_policy)
    s = _as_float_series(x, "x")
    labels = _group_labels(s, by, missing_group=missing_group)
    groupers = _groupers(s, labels)

    input_na = s.isna()
    n_valid = _valid_counts(s, groupers)
    work = _pre_fill(s, groupers, policy)
    grouped = work.groupby(groupers)

    ranks = grouped.rank(method=method, ascending=ascending)
    counts = grouped.transform("count")
    uniform = (ranks - 0.5) / counts.where(counts > 0)

    if mode == "uniform":
        out, neutral = uniform, 0.5
    elif mode == "centered":
        out, neutral = 2.0 * uniform - 1.0, 0.0
    elif mode == "normal":
        from scipy.stats import norm  # import perezoso: scipy solo si se pide

        out = pd.Series(norm.ppf(uniform.to_numpy()), index=uniform.index)
        neutral = 0.0
    else:  # pragma: no cover - protegido por el tipo
        msg = f"modo de rango desconocido: {mode!r}"
        raise ValueError(msg)

    out = _enforce_min_obs(out, n_valid, min_obs, on_insufficient, what="rank_pct")
    out = _post_fill(out, input_na, policy, neutral)
    out.name = x.name
    return out


def _center_by_group(
    x: pd.Series | pd.DataFrame,
    by: object | None,
    how: Literal["mean", "median"],
    *,
    nan_policy: NaNPolicy | str,
    min_obs: int,
    min_group: int,
    small_group: Literal["nan", "pool", "raise"],
    on_insufficient: InsufficientPolicy,
    missing_group: Literal["nan", "pool", "raise"],
) -> pd.Series | pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return _columnwise(
            _center_by_group,
            x,
            by,
            how,
            nan_policy=nan_policy,
            min_obs=min_obs,
            min_group=min_group,
            small_group=small_group,
            on_insufficient=on_insufficient,
            missing_group=missing_group,
        )

    policy = _parse_nan_policy(nan_policy)
    s = _as_float_series(x, "x")
    labels = _group_labels(s, by, missing_group=missing_group)

    if labels is not None and min_group > 1:
        counts = s.notna().astype("float64").groupby(_groupers(s, labels)).transform("sum")
        small = (counts.to_numpy() < min_group) & pd.notna(labels)
        if small.any():
            if small_group == "raise":
                bad = sorted({str(v) for v in labels[small]})
                msg = (
                    f"grupos con menos de {min_group} observaciones válidas: {bad[:8]}. "
                    "Centrar dentro de un grupo diminuto es ruido, no neutralización."
                )
                raise InsufficientHistory(msg)
            labels = labels.copy()
            # "pool": se centran contra toda la sección cruzada de la fecha;
            # "nan": quedan sin grupo y por tanto sin valor.
            labels[small] = _OTHER_GROUP if small_group == "pool" else None

    groupers = _groupers(s, labels)
    input_na = s.isna()
    n_valid = _valid_counts(s, groupers)
    work = _pre_fill(s, groupers, policy)
    center = work.groupby(groupers).transform(how)
    out = work - center

    out = _enforce_min_obs(out, n_valid, min_obs, on_insufficient, what=f"de{how}")
    out = _post_fill(out, input_na, policy, 0.0)
    out.name = x.name
    return out


def demean(
    x: pd.Series | pd.DataFrame,
    by: object | None = None,
    *,
    nan_policy: NaNPolicy | str = NaNPolicy.PROPAGATE,
    min_obs: int = 2,
    min_group: int = 2,
    small_group: Literal["nan", "pool", "raise"] = "nan",
    on_insufficient: InsufficientPolicy = "nan",
    missing_group: Literal["nan", "pool", "raise"] = "nan",
) -> pd.Series | pd.DataFrame:
    """Resta la media del grupo dentro de cada fecha (``by=None`` → toda la sección).

    Es la neutralización más simple: equivale a `neutralize` con un diseño de
    solo dummies del grupo, sin regresores continuos, y deja la media exactamente
    a cero en cada grupo. Con grupos pequeños es frágil —la media de 21 nombres
    de Energy tiene un error estándar considerable—, por lo que `min_group`
    permite exigir un tamaño mínimo y `small_group` decide qué hacer con los que
    no llegan: anularlos (`"nan"`), agruparlos contra el resto de la sección
    cruzada (`"pool"`) o fallar (`"raise"`).
    """
    return _center_by_group(
        x,
        by,
        "mean",
        nan_policy=nan_policy,
        min_obs=min_obs,
        min_group=min_group,
        small_group=small_group,
        on_insufficient=on_insufficient,
        missing_group=missing_group,
    )


def demedian(
    x: pd.Series | pd.DataFrame,
    by: object | None = None,
    *,
    nan_policy: NaNPolicy | str = NaNPolicy.PROPAGATE,
    min_obs: int = 2,
    min_group: int = 2,
    small_group: Literal["nan", "pool", "raise"] = "nan",
    on_insufficient: InsufficientPolicy = "nan",
    missing_group: Literal["nan", "pool", "raise"] = "nan",
) -> pd.Series | pd.DataFrame:
    """Resta la **mediana** del grupo dentro de cada fecha.

    Preferible a `demean` cuando el grupo tiene pocos nombres o alguno con una
    cifra extrema: la mediana de un sector de 21 nombres no se mueve porque una
    petrolera reporte un accrual absurdo, la media sí. A cambio, la media del
    residuo no es exactamente cero (la mediana no es un proyector lineal), de
    modo que si lo que se necesita es exposición sectorial *nula por
    construcción* hay que usar `demean` o `neutralize`.
    """
    return _center_by_group(
        x,
        by,
        "median",
        nan_policy=nan_policy,
        min_obs=min_obs,
        min_group=min_group,
        small_group=small_group,
        on_insufficient=on_insufficient,
        missing_group=missing_group,
    )


def _columnwise(
    func: object,
    frame: pd.DataFrame,
    *args: object,
    **kwargs: object,
) -> pd.DataFrame:
    """Aplica una transformación columna a columna conservando nombres y orden."""
    check_panel(frame, name="x")
    pieces = {}
    for col in frame.columns:
        result = func(frame[col], *args, **kwargs)  # type: ignore[operator]
        pieces[col] = result
    out = pd.DataFrame(pieces)
    # Con nan_policy="drop" cada columna puede perder filas distintas; el índice
    # resultante es la unión, y las diferencias quedan como NaN de forma visible.
    return out[list(frame.columns)]


# ---------------------------------------------------------------------------
# Regresión cross-section: residualize / neutralize
# ---------------------------------------------------------------------------


def _blocks(index: pd.MultiIndex) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    """Devuelve (orden por fecha, offsets de bloque, fechas únicas)."""
    codes, uniques = pd.factorize(index.get_level_values(0), sort=True)
    order = np.argsort(codes, kind="stable")
    offsets = np.searchsorted(codes[order], np.arange(len(uniques) + 1))
    return order, offsets, uniques


def _lstsq_residual(
    y: np.ndarray, design: np.ndarray, weights: np.ndarray | None
) -> tuple[np.ndarray, int]:
    """Residuo de la proyección de `y` sobre el espacio columna de `design`.

    Se usa `numpy.linalg.lstsq` (SVD, solución de norma mínima) y no
    ``inv(X'X) X'y``: con dummies sectoriales más intercepto el diseño es
    deliberadamente deficiente de rango, `X'X` es singular y cualquier inversión
    explícita produce basura numérica. La SVD devuelve la proyección correcta y
    el residuo es único aunque los coeficientes no lo sean
    (Golub y Van Loan 2013, §5.5; `fundamental_factors.md` §1.4).
    """
    if weights is None:
        beta, _res, rank, _sv = np.linalg.lstsq(design, y, rcond=None)
    else:
        root = np.sqrt(weights)[:, None]
        beta, _res, rank, _sv = np.linalg.lstsq(design * root, y * root[:, 0], rcond=None)
    return y - design @ beta, int(rank)


def residualize(
    y: pd.Series,
    regressors: pd.DataFrame | pd.Series,
    *,
    add_intercept: bool = True,
    weights: pd.Series | None = None,
    min_obs: int = 10,
    min_dof: int = 5,
    nan_policy: NaNPolicy | str = NaNPolicy.PROPAGATE,
    on_insufficient: InsufficientPolicy = "nan",
    standardize: bool = False,
    regressor_nan: Literal["drop", "raise"] = "drop",
) -> pd.Series:
    """Residuo de la regresión cross-section de `y` sobre `regressors`, fecha a fecha.

    ``residuo_t = y_t - X_t (X_t'X_t)⁻ X_t' y_t`` resuelto por SVD. Es el núcleo
    de `neutralize` y de la ortogonalización secuencial de `combine`, y sigue el
    esquema de regresión transversal por fecha de Fama y MacBeth (1973).

    Parámetros
    ----------
    regressors:
        Panel numérico ``(date, ticker)`` con una columna por regresor. Las
        variables categóricas hay que codificarlas antes (o usar `neutralize`,
        que lo hace por ti).
    weights:
        Pesos por observación para mínimos cuadrados ponderados (p. ej. raíz de
        la capitalización, para que el residuo sea neutral en la cartera
        ponderada por valor y no en la equiponderada). Con pesos, la media del
        residuo es cero **en la métrica ponderada**, no en la simple.
    min_dof:
        Grados de libertad mínimos exigidos (``n_válidos - rango(X)``). Sin este
        control, una fecha con tantos nombres como columnas produce un ajuste
        perfecto y un residuo idénticamente cero, que se vería como "señal
        perfectamente neutralizada" cuando en realidad no queda señal alguna.
    regressor_nan:
        Qué hacer si a un nombre le falta alguna exposición: excluirlo de la
        regresión y devolver NaN (`"drop"`, por defecto) o fallar (`"raise"`).

    Devuelve
    --------
    `Series` alineada con `y`. NaN donde no había dato, donde faltaba una
    exposición o donde la sección cruzada no daba para estimar.
    """
    policy = _parse_nan_policy(nan_policy)
    s = _as_float_series(y, "y")
    if isinstance(regressors, pd.Series):
        regressors = regressors.to_frame()
    if not isinstance(regressors, pd.DataFrame) or regressors.shape[1] == 0:
        msg = "`regressors` debe ser un DataFrame con al menos una columna"
        raise DataQualityError(msg)

    x_panel = cast(pd.DataFrame, align_panel_like(regressors, s.index, name="regressors"))
    non_numeric = [c for c in x_panel.columns if not is_numeric_dtype(x_panel[c])]
    if non_numeric:
        msg = (
            f"`regressors` tiene columnas no numéricas: {non_numeric}. "
            "Codifícalas como dummies o usa `neutralize`, que construye el diseño."
        )
        raise DataQualityError(msg)

    input_na = s.isna()
    if policy is NaNPolicy.RAISE and bool(input_na.any()):
        msg = f"{int(input_na.sum())} valores ausentes en `y` con nan_policy='raise'"
        raise DataQualityError(msg)

    reg_values = x_panel.to_numpy(dtype="float64")
    reg_ok = np.isfinite(reg_values).all(axis=1)
    if not reg_ok.all() and regressor_nan == "raise":
        msg = (
            f"{int((~reg_ok).sum())} observaciones con exposiciones ausentes; "
            "con regressor_nan='raise' no se estima"
        )
        raise DataQualityError(msg)

    y_values = s.to_numpy(dtype="float64")
    if policy in (NaNPolicy.MEAN, NaNPolicy.MEDIAN):
        how = "mean" if policy is NaNPolicy.MEAN else "median"
        groupers = _groupers(s, None)
        filler = s.groupby(groupers).transform(how)
        y_values = np.where(np.isnan(y_values), filler.to_numpy(dtype="float64"), y_values)

    usable = np.isfinite(y_values) & reg_ok
    w_values = None
    if weights is not None:
        w_aligned = cast(pd.Series, align_panel_like(weights, s.index, name="weights"))
        w_values = w_aligned.to_numpy(dtype="float64")
        if np.nanmin(w_values, initial=np.inf) < 0:
            msg = "los pesos de `weights` deben ser no negativos"
            raise ValueError(msg)
        usable &= np.isfinite(w_values) & (w_values > 0)

    out = np.full(len(s), np.nan)
    n_used = np.zeros(len(s))
    order, offsets, _dates = _blocks(s.index)

    for start, stop in pairwise(offsets):
        rows = order[start:stop]
        sel = rows[usable[rows]]
        n_used[rows] = len(sel)
        if len(sel) < max(min_obs, 1):
            continue
        block_x = reg_values[sel]
        if add_intercept:
            block_x = block_x - block_x.mean(axis=0, keepdims=True)
            design = np.column_stack([np.ones(len(sel)), block_x])
        else:
            design = block_x
        block_y = y_values[sel]
        block_w = None if w_values is None else w_values[sel]
        resid, rank = _lstsq_residual(block_y, design, block_w)
        if len(sel) - rank < min_dof:
            continue
        out[sel] = resid

    result = pd.Series(out, index=s.index, name=s.name)
    result = _enforce_min_obs(
        result, pd.Series(n_used, index=s.index), min_obs, on_insufficient, what="residualize"
    )
    if standardize:
        result = cast(pd.Series, zscore(result, min_obs=min_obs, on_insufficient="nan"))
    return _post_fill(result, input_na, policy, 0.0)


def _expand_exposures(
    frame: pd.DataFrame,
    categorical: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Separa exposiciones numéricas de las categóricas (estas, como códigos)."""
    numeric = frame.drop(columns=list(categorical))
    codes: dict[str, np.ndarray] = {}
    for col in categorical:
        values = frame[col]
        code, _uniq = pd.factorize(values, sort=True)
        codes[col] = np.asarray(code)
    return numeric, codes


def neutralize(
    x: pd.Series | pd.DataFrame,
    exposures: pd.DataFrame | pd.Series | Mapping[str, object] | None = None,
    *,
    by: Sequence[str] | str | None = None,
    categorical: Sequence[str] | None = None,
    weights: pd.Series | None = None,
    add_intercept: bool = True,
    standardize: bool = False,
    min_obs: int = 20,
    min_dof: int = 5,
    nan_policy: NaNPolicy | str = NaNPolicy.PROPAGATE,
    on_insufficient: InsufficientPolicy = "nan",
    exposure_nan: Literal["drop", "raise"] = "drop",
) -> pd.Series | pd.DataFrame:
    """Elimina de la señal su exposición lineal a sector, tamaño, beta, etc.

    Implementa el paso 4 de la receta canónica (`fundamental_factors.md` §1.4):

    ``residuo = x - X (X'X)⁻ X' x``  con  ``X = [1, dummies(GICS), log(MC), beta]``

    Por qué es obligatorio y no cosmético: sin neutralizar sectorialmente, un
    factor de FCF yield es una apuesta larga en Energía y corta en Tecnología, y
    un factor de apalancamiento es una apuesta corta en Utilities; el propio
    documento de investigación lo señala factor por factor (§9.4, §11). Una IC
    que desaparece al neutralizar por sector era una apuesta sectorial
    disfrazada (`validation_methodology.md` §2.2).

    Parámetros
    ----------
    exposures:
        Panel ``(date, ticker)`` con las columnas de exposición, o un mapa
        estático ticker → valor (típico del sector). Ver `align_panel_like` para
        la advertencia point-in-time del caso estático.
    by:
        Subconjunto de columnas a usar. `None` usa todas.
    categorical:
        Columnas que deben expandirse a variables dummy. Si es `None` se detectan
        automáticamente las columnas no numéricas. Se incluyen **todas** las
        categorías presentes en cada fecha junto al intercepto: el diseño queda
        deficiente de rango a propósito y la SVD de `lstsq` resuelve la
        proyección sin necesidad de elegir una categoría de referencia
        arbitraria.
    weights:
        Pesos para mínimos cuadrados ponderados (ver `residualize`).
    min_obs:
        Mínimo de nombres válidos por fecha. El valor por defecto (20) es
        deliberadamente alto: con 11 sectores el diseño ya tiene 12-14 columnas y
        neutralizar con 15 nombres no deja residuo, deja ruido.

    Devuelve
    --------
    Panel de residuos con la misma forma. Verificable: la media por sector y la
    correlación con `log(MC)` del residuo son ~0 a precisión de máquina
    (`fundamental_factors.md` §15.2).
    """
    if isinstance(x, pd.DataFrame):
        return _columnwise(
            neutralize,
            x,
            exposures,
            by=by,
            categorical=categorical,
            weights=weights,
            add_intercept=add_intercept,
            standardize=standardize,
            min_obs=min_obs,
            min_dof=min_dof,
            nan_policy=nan_policy,
            on_insufficient=on_insufficient,
            exposure_nan=exposure_nan,
        )

    s = _as_float_series(x, "x")
    if exposures is None:
        msg = (
            "`neutralize` necesita exposiciones: pásalas en `exposures` "
            "(sector, log market cap, beta...). Para centrar solo por grupo usa `demean`."
        )
        raise DataQualityError(msg)

    aligned = align_panel_like(exposures, s.index, name="exposures")
    frame = aligned.to_frame() if isinstance(aligned, pd.Series) else aligned
    if by is not None:
        cols = [by] if isinstance(by, str) else list(by)
        missing = [c for c in cols if c not in frame.columns]
        if missing:
            msg = f"columnas de exposición ausentes: {missing}; disponibles: {list(frame.columns)}"
            raise DataQualityError(msg)
        frame = frame[cols]
    if frame.shape[1] == 0:
        msg = "`exposures` no aporta ninguna columna"
        raise DataQualityError(msg)

    if categorical is None:
        cat_cols = [c for c in frame.columns if not is_numeric_dtype(frame[c])]
    else:
        cat_cols = [categorical] if isinstance(categorical, str) else list(categorical)
        missing = [c for c in cat_cols if c not in frame.columns]
        if missing:
            msg = f"columnas categóricas ausentes en `exposures`: {missing}"
            raise DataQualityError(msg)

    numeric, codes = _expand_exposures(frame, cat_cols)
    numeric_values = numeric.to_numpy(dtype="float64") if numeric.shape[1] else None

    exposure_ok = np.ones(len(s), dtype=bool)
    if numeric_values is not None:
        exposure_ok &= np.isfinite(numeric_values).all(axis=1)
    for code in codes.values():
        exposure_ok &= code >= 0
    if not exposure_ok.all() and exposure_nan == "raise":
        msg = (
            f"{int((~exposure_ok).sum())} observaciones sin exposición completa "
            "(sector desconocido o exposición numérica ausente)"
        )
        raise DataQualityError(msg)

    policy = _parse_nan_policy(nan_policy)
    input_na = s.isna()
    if policy is NaNPolicy.RAISE and bool(input_na.any()):
        msg = f"{int(input_na.sum())} valores ausentes en `x` con nan_policy='raise'"
        raise DataQualityError(msg)
    y_values = s.to_numpy(dtype="float64")
    if policy in (NaNPolicy.MEAN, NaNPolicy.MEDIAN):
        how = "mean" if policy is NaNPolicy.MEAN else "median"
        filler = s.groupby(_groupers(s, None)).transform(how)
        y_values = np.where(np.isnan(y_values), filler.to_numpy(dtype="float64"), y_values)

    usable = np.isfinite(y_values) & exposure_ok
    w_values = None
    if weights is not None:
        w_aligned = cast(pd.Series, align_panel_like(weights, s.index, name="weights"))
        w_values = w_aligned.to_numpy(dtype="float64")
        usable &= np.isfinite(w_values) & (w_values > 0)

    out = np.full(len(s), np.nan)
    n_used = np.zeros(len(s))
    order, offsets, _dates = _blocks(s.index)

    for start, stop in pairwise(offsets):
        rows = order[start:stop]
        sel = rows[usable[rows]]
        n_used[rows] = len(sel)
        if len(sel) < max(min_obs, 1):
            continue
        parts: list[np.ndarray] = []
        if add_intercept:
            parts.append(np.ones((len(sel), 1)))
        if numeric_values is not None:
            block = numeric_values[sel]
            if add_intercept:
                # Centrar los regresores continuos no cambia el espacio columna
                # cuando hay intercepto, pero mejora mucho el condicionamiento:
                # log(MC) ~ 24 con desviación 1 daría un número de condición enorme.
                block = block - block.mean(axis=0, keepdims=True)
            parts.append(block)
        for code in codes.values():
            block_codes = code[sel]
            present = np.unique(block_codes)
            dummies = (block_codes[:, None] == present[None, :]).astype("float64")
            parts.append(dummies)
        design = np.column_stack(parts)
        block_w = None if w_values is None else w_values[sel]
        resid, rank = _lstsq_residual(y_values[sel], design, block_w)
        if len(sel) - rank < min_dof:
            continue
        out[sel] = resid

    result = pd.Series(out, index=s.index, name=s.name)
    result = _enforce_min_obs(
        result, pd.Series(n_used, index=s.index), min_obs, on_insufficient, what="neutralize"
    )
    if standardize:
        result = cast(pd.Series, zscore(result, min_obs=min_obs, on_insufficient="nan"))
    return _post_fill(result, input_na, policy, 0.0)


def condition_factor(
    x: pd.Series | pd.DataFrame,
    exposures: pd.DataFrame | pd.Series | Mapping[str, object] | None = None,
    *,
    by: Sequence[str] | str | None = None,
    categorical: Sequence[str] | None = None,
    winsor_q: float = 0.01,
    winsor_method: Literal["quantile", "mad", "sigma"] = "quantile",
    robust_z: bool = False,
    weights: pd.Series | None = None,
    nan_policy: NaNPolicy | str = NaNPolicy.PROPAGATE,
    min_obs: int = 20,
    on_insufficient: InsufficientPolicy = "nan",
) -> pd.Series | pd.DataFrame:
    """Receta canónica de acondicionamiento de un factor (§1.4 de la investigación).

    Encadena, **por fecha**::

        1. winsorizar al 1 % / 99 %          -> winsorize(q)
        2. z-score cross-section             -> zscore
        3. neutralizar por regresión          -> neutralize(by=[sector, size, beta])
        4. re-estandarizar el residuo        -> zscore

    El orden importa: winsorizar después de estandarizar deja que un único
    outlier fije la escala de toda la sección cruzada. Y re-estandarizar al final
    es imprescindible porque la varianza del residuo depende del R² de la
    neutralización, que varía de fecha en fecha: sin el último paso, la señal
    tendría más peso los días en que el modelo de riesgo explica menos.

    Sin `exposures` ejecuta solo los pasos 1-2, que es lo correcto para una señal
    ya neutral por construcción.
    """
    step = winsorize(
        x,
        winsor_q,
        method=winsor_method,
        nan_policy=nan_policy,
        min_obs=min_obs,
        on_insufficient=on_insufficient,
    )
    step = zscore(
        step,
        robust=robust_z,
        nan_policy=NaNPolicy.PROPAGATE,
        min_obs=min_obs,
        on_insufficient=on_insufficient,
    )
    if exposures is None:
        return step
    step = neutralize(
        step,
        exposures,
        by=by,
        categorical=categorical,
        weights=weights,
        standardize=False,
        min_obs=min_obs,
        nan_policy=NaNPolicy.PROPAGATE,
        on_insufficient=on_insufficient,
    )
    return zscore(
        step,
        robust=robust_z,
        nan_policy=NaNPolicy.PROPAGATE,
        min_obs=min_obs,
        on_insufficient=on_insufficient,
    )
