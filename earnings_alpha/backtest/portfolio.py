"""Construcción de carteras cross-section para el backtest (§3.7).

Todo lo que hay aquí opera sobre **una sección cruzada** (una fecha): una
`pd.Series` de puntuaciones indexada por ticker entra, y una `pd.Series` de
pesos objetivo (fracción del NAV, positivos largos, negativos cortos) sale. El
motor (`engine.py`) itera fechas; este módulo no ve el tiempo, y por eso no
puede introducir look-ahead.

Piezas, en el orden en que el motor las aplica:

1. `assign_quantiles` — cubos por rango (global o dentro de grupo/sector).
2. `build_target_weights` — pipeline completo: quantiles → pesos (igual o por
   score) → neutralidad sectorial opcional → tope de peso → re-equilibrio de
   patas para mantener la neutralidad en dólares.
3. `cap_weights` — tope por nombre con redistribución dentro de la pata.
4. `apply_participation_limit` / `apply_turnover_limit` — restricciones sobre
   las **operaciones** (necesitan los pesos previos, así que las aplica el motor
   en la fecha de ejecución).

Convenciones de pesos
---------------------
- Long-short: pata larga suma +1 y pata corta −1 (bruto 2, neto 0); es la
  cartera `Q_alto − Q_bajo` estándar de la literatura de factores (Fama y French
  1993; `validation_methodology.md` §11.1).
- Long-only: pesos suman +1.
- Si el tope de peso hace inviable colocar toda la pata (`n·max_weight < 1`),
  la pata queda **desapalancada** (suma menos de 1) en vez de violar el tope; con
  `long_short` ambas patas se reescalan al bruto común alcanzable para conservar
  la neutralidad en dólares. Nunca se viola `max_weight` en silencio.

Referencias
-----------
- Fama, E. y French, K. (1993). *Common Risk Factors in the Returns on Stocks
  and Bonds*. JFE 33: construcción por cubos de características.
- Patton, A. y Timmermann, A. (2010). *Monotonicity in Asset Returns*. JFE 98:
  por qué los cubos deben ser comparables entre fechas (rango, no niveles).
- Grinold, R. y Kahn, R. (2000). *Active Portfolio Management*, cap. 14: pesos
  proporcionales a la señal y el *transfer coefficient* que las restricciones
  (tope de peso, participación) restan al IR (`validation_methodology.md` §11.4).
- `docs/research/validation_methodology.md` §11.1: la ponderación y la
  neutralización sectorial son decisiones que deben declararse siempre.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from earnings_alpha.errors import ConfigError, DataQualityError, InsufficientHistory

__all__ = [  # noqa: RUF022 - orden temático, no alfabético
    "Weighting",
    "assign_quantiles",
    "leg_weights",
    "build_target_weights",
    "cap_weights",
    "balance_legs",
    "one_way_turnover",
    "apply_turnover_limit",
    "apply_participation_limit",
]

Weighting = str
"""Esquema de ponderación dentro de cada pata: ``"equal"`` o ``"score"``."""

_VALID_WEIGHTING = ("equal", "score")


def _check_cross_section(scores: pd.Series, name: str = "scores") -> pd.Series:
    """Valida una sección cruzada: Series numérica indexada por ticker, sin duplicados."""
    if not isinstance(scores, pd.Series):
        msg = f"`{name}` debe ser una pd.Series indexada por ticker"
        raise DataQualityError(msg)
    if isinstance(scores.index, pd.MultiIndex):
        msg = (
            f"`{name}` tiene MultiIndex: este módulo opera sobre UNA sección "
            "cruzada; extrae la fecha antes (scores.xs(fecha))"
        )
        raise DataQualityError(msg)
    if scores.index.has_duplicates:
        dupes = scores.index[scores.index.duplicated()].unique().tolist()[:5]
        msg = f"`{name}` tiene tickers duplicados; muestra: {dupes}"
        raise DataQualityError(msg)
    return scores.astype(float)


def assign_quantiles(
    scores: pd.Series,
    n_quantiles: int,
    *,
    by: pd.Series | Mapping[str, str] | None = None,
) -> pd.Series:
    """Cubo 1..Q por rango de la puntuación (1 = peor, Q = mejor).

    El cubo del nombre con rango `r` (1..n) es ``ceil(r·Q/n)``: los cubos quedan
    de tamaños lo más iguales posible y cada cubo es no vacío si `n >= Q`. El
    rango usa `method="first"` con el índice ordenado alfabéticamente, de modo
    que los empates se resuelven de forma **determinista** (regla 4 del repo:
    determinismo; un `qcut` con bordes duplicados no lo es).

    Con `by` (p. ej. sector GICS) los cubos se calculan **dentro de cada
    grupo**: es el "quintil dentro de sector" de `validation_methodology.md`
    §11.1. Un grupo con menos de `n_quantiles` nombres no puede llenar todos los
    cubos sin degenerar y sus nombres salen como NaN (excluidos), nunca
    asignados a un cubo calculado sobre dos observaciones.

    Los NaN de entrada salen como NaN. Devuelve una Series `float` (NaN exige
    dtype flotante) con valores enteros 1..Q.
    """
    s = _check_cross_section(scores)
    if n_quantiles < 2:
        msg = f"n_quantiles debe ser >= 2; recibido {n_quantiles}"
        raise ConfigError(msg)

    out = pd.Series(np.nan, index=s.index, dtype=float, name="quantile")
    valid = s.dropna()
    if valid.empty:
        return out

    if by is None:
        groups: list[pd.Series] = [valid]
    else:
        labels = pd.Series(dict(by)) if isinstance(by, Mapping) else by
        labels = labels.reindex(valid.index)
        if labels.isna().any():
            missing = labels.index[labels.isna()].tolist()[:5]
            msg = f"`by` no cubre todos los tickers puntuados; faltan p. ej. {missing}"
            raise DataQualityError(msg)
        groups = [grp for _, grp in valid.groupby(labels, sort=True)]

    for grp in groups:
        n = len(grp)
        if n < n_quantiles:
            continue  # grupo demasiado pequeño: NaN explícito, no un cubo degenerado
        ordered = grp.sort_index()  # empates resueltos por orden alfabético: determinista
        ranks = ordered.rank(method="first")
        buckets = np.ceil(ranks * n_quantiles / n)
        out.loc[ordered.index] = buckets

    return out


def leg_weights(
    scores: pd.Series,
    total: float,
    weighting: Weighting = "equal",
) -> pd.Series:
    """Pesos dentro de una pata que suman `total` (con su signo).

    - ``"equal"``: `total/n` por nombre.
    - ``"score"``: proporcional al **rango** de la puntuación dentro de la pata
      (el mejor nombre pesa más; en la pata corta, el peor). Se usa el rango y no
      el score crudo porque es invariante a transformaciones monótonas y no
      puede producir pesos negativos dentro de la pata con distribuciones
      asimétricas. Con dispersión nula degrada a equiponderado.
    """
    s = _check_cross_section(scores).dropna()
    if s.empty:
        msg = "pata vacía: no hay nombres con puntuación"
        raise InsufficientHistory(msg)
    if weighting not in _VALID_WEIGHTING:
        msg = f"weighting desconocido: {weighting!r}; válidos: {_VALID_WEIGHTING}"
        raise ConfigError(msg)

    if weighting == "equal" or (s.to_numpy() == s.iloc[0]).all():
        raw = pd.Series(1.0, index=s.index)
    else:
        ascending = total >= 0  # pata larga: score alto ⇒ rango alto ⇒ más peso
        raw = s.sort_index().rank(method="first", ascending=ascending)
    return raw / raw.sum() * total


def cap_weights(
    weights: pd.Series,
    max_weight: float,
    *,
    max_iter: int = 200,
) -> pd.Series:
    """Tope de peso por nombre con redistribución dentro de cada signo.

    El exceso sobre `max_weight` (en valor absoluto) se redistribuye
    proporcionalmente entre los nombres del **mismo signo** aún por debajo del
    tope, iterando hasta converger (la redistribución puede empujar a otros por
    encima). Si todos los nombres del signo quedan al tope, la pata se
    desapalanca: suma `n·max_weight` en bruto, y esa reducción es visible en el
    resultado, no silenciosa.
    """
    if not np.isfinite(max_weight) or max_weight <= 0.0:
        msg = f"max_weight debe ser > 0; recibido {max_weight!r}"
        raise ConfigError(msg)
    w = _check_cross_section(weights, "weights").fillna(0.0)

    out = w.copy()
    for sign in (1.0, -1.0):
        leg = out[np.sign(out) == sign]
        if leg.empty:
            continue
        vals = leg.abs()
        for _ in range(max_iter):
            over = vals > max_weight + 1e-15
            if not over.any():
                break
            excess = float((vals[over] - max_weight).sum())
            vals[over] = max_weight
            under = ~over
            room = max_weight - vals[under]
            if float(room.sum()) <= 1e-15:
                break  # pata entera al tope: queda desapalancada, no redistribuible
            # reparto proporcional al hueco disponible; si alguno rebasa, la
            # siguiente iteración lo recorta y redistribuye de nuevo (converge
            # geométricamente al reparto factible).
            vals[under] = vals[under] + room / float(room.sum()) * excess
        vals = vals.clip(upper=max_weight)  # garantía dura tras max_iter
        out.loc[leg.index] = vals * sign
    return out


def balance_legs(weights: pd.Series) -> pd.Series:
    """Reescala las patas larga y corta al bruto común mínimo (neutralidad en dólares).

    Tras un tope de peso las dos patas pueden quedar con brutos distintos (una
    tenía más nombres donde redistribuir); mantener ese desequilibrio convierte
    la cartera en una apuesta direccional no pedida. Se reescala la pata mayor
    hacia abajo (nunca se apalanca hacia arriba, que violaría el tope).
    """
    w = _check_cross_section(weights, "weights").fillna(0.0)
    gross_long = float(w[w > 0].sum())
    gross_short = float(-w[w < 0].sum())
    if gross_long <= 0.0 or gross_short <= 0.0:
        return w
    common = min(gross_long, gross_short)
    out = w.copy()
    out[w > 0] = w[w > 0] * (common / gross_long)
    out[w < 0] = w[w < 0] * (common / gross_short)
    return out


def build_target_weights(
    scores: pd.Series,
    *,
    n_quantiles: int = 5,
    long_short: bool = True,
    weighting: Weighting = "equal",
    max_weight: float | None = None,
    sectors: pd.Series | Mapping[str, str] | None = None,
) -> pd.Series:
    """Pipeline completo: puntuaciones de una fecha → pesos objetivo.

    Cartera de cubos extremos (Fama-French): larga el cubo `Q` y, si
    `long_short`, corta el cubo `1`. Con `sectors` los cubos se forman **dentro
    de cada sector** y cada sector aporta a cada pata un peso proporcional a su
    número de nombres puntuados, de modo que la exposición neta por sector es
    exactamente cero (la neutralización sectorial de `validation_methodology.md`
    §11.1: "si la señal solo funciona globalmente, es una apuesta sectorial").

    Devuelve pesos sobre **todos** los tickers de entrada (cero para los no
    incluidos), listos para alinear con el panel de precios.

    Raises
    ------
    InsufficientHistory
        Si ningún grupo alcanza `n_quantiles` nombres puntuados.
    """
    s = _check_cross_section(scores)
    buckets = assign_quantiles(s, n_quantiles, by=sectors)
    if buckets.dropna().empty:
        msg = (
            f"sección cruzada insuficiente: ningún grupo llega a {n_quantiles} "
            f"nombres puntuados (hay {int(s.notna().sum())} en total)"
        )
        raise InsufficientHistory(msg)

    out = pd.Series(0.0, index=s.index, name="weight")

    if sectors is None:
        group_labels = pd.Series("__all__", index=s.index)
    else:
        group_labels = pd.Series(dict(sectors)) if isinstance(sectors, Mapping) else sectors
        group_labels = group_labels.reindex(s.index)

    scored = buckets.dropna().index
    shares = group_labels.loc[scored].value_counts(normalize=True, sort=False)

    for group, share in shares.items():
        in_group = scored[group_labels.loc[scored] == group]
        top = in_group[buckets.loc[in_group] == n_quantiles]
        if len(top) > 0:
            out.loc[top] = leg_weights(s.loc[top], float(share), weighting)
        if long_short:
            bottom = in_group[buckets.loc[in_group] == 1]
            if len(bottom) > 0:
                out.loc[bottom] = leg_weights(s.loc[bottom], -float(share), weighting)

    if max_weight is not None:
        out = cap_weights(out, max_weight)
        if long_short:
            out = balance_legs(out)
    return out


def one_way_turnover(previous: pd.Series, target: pd.Series) -> float:
    """Rotación one-way: ``½·Σ|target − previous|`` (fracción del NAV a comprar).

    Misma definición que `stats.performance.turnover`; el nocional negociado
    total es el doble.
    """
    prev, tgt = previous.align(target, fill_value=0.0)
    return float(0.5 * (tgt - prev).abs().sum())


def apply_turnover_limit(
    previous: pd.Series,
    target: pd.Series,
    max_turnover: float,
) -> tuple[pd.Series, float]:
    """Limita la rotación de un rebalanceo moviéndose parcialmente hacia el objetivo.

    Si la rotación necesaria supera `max_turnover` (one-way, fracción del NAV),
    se ejecuta la fracción `λ = max_turnover / rotación` de cada operación:
    ``w = previo + λ·(objetivo − previo)``. Escalar todas las operaciones por el
    mismo λ preserva las proporciones del objetivo (maximiza el *transfer
    coefficient* bajo la restricción de rotación, Grinold-Kahn cap. 16).

    Devuelve `(pesos, λ)` con `λ ∈ (0, 1]`.
    """
    if not np.isfinite(max_turnover) or max_turnover < 0.0:
        msg = f"max_turnover debe ser >= 0; recibido {max_turnover!r}"
        raise ConfigError(msg)
    prev, tgt = previous.align(target, fill_value=0.0)
    need = one_way_turnover(prev, tgt)
    if need <= max_turnover or need == 0.0:
        return tgt, 1.0
    lam = max_turnover / need
    return prev + lam * (tgt - prev), lam


def apply_participation_limit(
    previous: pd.Series,
    target: pd.Series,
    max_trade_weight: pd.Series,
) -> pd.Series:
    """Recorta cada operación al máximo peso negociable por nombre.

    `max_trade_weight` es ``participación_máxima · ADV_i / NAV`` en unidades de
    peso; la calcula el motor con el ADV **hacia atrás** conocido en la fecha de
    ejecución. Un límite NaN o infinito no restringe (la penalización por
    liquidez desconocida vive en `CostModel`, que es donde es explícita); un
    límite 0 congela la posición.
    """
    prev, tgt = previous.align(target, fill_value=0.0)
    cap = max_trade_weight.reindex(prev.index)
    cap = cap.where(np.isfinite(cap), np.inf).clip(lower=0.0)
    delta = (tgt - prev).clip(lower=-cap, upper=cap)
    return prev + delta
