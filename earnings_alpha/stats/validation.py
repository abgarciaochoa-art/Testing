"""Validación honesta: CV purgada, CPCV, bootstrap estacionario y multiplicidad.

Implementa las secciones 6 a 10 de `docs/research/validation_methodology.md`.

Piezas
------
- **CV purgada con embargo** (`PurgedKFold`): elimina del entrenamiento toda
  observación cuyo *span de información* solape con el test, y además embarga hacia
  adelante. El span **no es solo la etiqueta**: si las features miran `b` periodos
  atrás y la etiqueta `f` hacia adelante, el span de un evento en `T` es
  `[T−b, T+f]`. Para el ángulo B eso son 310 sesiones, no 60.
- **CPCV** (`CombinatorialPurgedCV`): en vez de una única trayectoria de backtest,
  `φ[N,k] = C(N,k)·k/N` trayectorias, de las que sale la *distribución* del Sharpe
  y por tanto el intervalo de confianza que exige `ARCHITECTURE.md` §3.8.
- **Bootstrap estacionario** de Politis–Romano, con longitud de bloque automática
  de Politis–White. El bootstrap iid sobre series autocorrelacionadas tiene una
  cobertura real del 50 % para un nominal del 95 % con `ρ = 0,8`: no es una
  aproximación aceptable, es un intervalo equivocado.
- **Multiplicidad**: Bonferroni, Holm, Benjamini–Hochberg, Benjamini–Yekutieli y
  Romano–Wolf, más el umbral de Harvey–Liu–Zhu (`t > 3`) y su lectura crítica.
- **PBO por CSCV** (`pbo_cscv`): valida el *procedimiento de búsqueda*, no la
  estrategia. Es complementario del CPCV, no un sustituto.
- **Reality Check de White y SPA de Hansen** para contrastar la mejor de `L`
  estrategias contra un benchmark.

Orden obligatorio de las correcciones (§7.4), porque invertirlo es el error nº 11
de la lista de fallos frecuentes::

    1. corregir dependencia (Newey–West / bootstrap por bloques)
    2. obtener p-valores válidos
    3. corregir multiplicidad (BH para cribar; Romano–Wolf u Holm para confirmar)
    4. deflactar el Sharpe (DSR) con el N del registro de pruebas

Referencias
-----------
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
  (Purga, embargo, CPCV.)
- Bailey, D. H., Borwein, J. M., López de Prado, M., y Zhu, Q. J. (2017).
  *The Probability of Backtest Overfitting*. **Journal of Computational Finance**
  20(4), 39-69.
- Politis, D. N., y Romano, J. P. (1994). *The Stationary Bootstrap*. **JASA**
  89(428), 1303-1313.
- Politis, D. N., y White, H. (2004). *Automatic Block-Length Selection for the
  Dependent Bootstrap*. **Econometric Reviews** 23(1), 53-70, con la corrección de
  Patton, Politis y White (2009).
- Benjamini, Y., y Hochberg, Y. (1995). **JRSS B** 57(1), 289-300.
- Benjamini, Y., y Yekutieli, D. (2001). **Annals of Statistics** 29(4), 1165-1188.
- Holm, S. (1979). **Scandinavian Journal of Statistics** 6(2), 65-70.
- Romano, J. P., y Wolf, M. (2005). **Econometrica** 73(4), 1237-1282; y (2016)
  **Statistics & Probability Letters** 113, 38-40.
- Harvey, C. R., Liu, Y., y Zhu, H. (2016). *…and the Cross-Section of Expected
  Returns*. **RFS** 29(1), 5-68.
- Chen, A. Y., y Zimmermann, T. (2020). *Publication Bias and the Cross-Section of
  Stock Returns*. **RAPS** 10(2), 249-289.
- White, H. (2000). *A Reality Check for Data Snooping*. **Econometrica** 68(5).
- Hansen, P. R. (2005). *A Test for Superior Predictive Ability*. **JBES** 23(4).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

from earnings_alpha.errors import DataQualityError, InsufficientHistory
from earnings_alpha.stats.ic import long_run_variance, newey_west_lags

__all__ = [
    "EULER_MASCHERONI",
    "BootstrapResult",
    "CPCVGeometry",
    "CPCVSplit",
    "CombinatorialPurgedCV",
    "MultipleTestResult",
    "MultipleTestingReport",
    "PBOResult",
    "PurgedKFold",
    "RomanoWolfResult",
    "SPAResult",
    "benjamini_hochberg",
    "benjamini_yekutieli",
    "benjamini_yekutieli_t",
    "block_length_from_autocovariance",
    "bonferroni",
    "bonferroni_t",
    "cpcv_paths",
    "hansen_spa",
    "harvey_liu_zhu_threshold",
    "hlz_equivalent_tests",
    "holm",
    "information_spans",
    "multiple_testing_report",
    "pbo_cscv",
    "politis_white_block_length",
    "purge_and_embargo",
    "romano_wolf",
    "stationary_bootstrap",
    "stationary_bootstrap_indices",
    "white_reality_check",
]

EULER_MASCHERONI = 0.5772156649015329
"""Constante de Euler–Mascheroni `γ`, usada en `c(M)` de Benjamini–Yekutieli y en
`expected_max_sharpe` (módulo `performance`)."""


# =========================================================================== #
# 1. Spans de información, purga y embargo                                     #
# =========================================================================== #


def information_spans(
    reference: Sequence[Any] | pd.Series | pd.DatetimeIndex,
    *,
    lookback: int,
    lookahead: int,
    timeline: Sequence[Any] | pd.DatetimeIndex | None = None,
    ids: Sequence[Any] | None = None,
) -> pd.DataFrame:
    """Construye el span de información `[T − lookback, T + lookahead]`.

    El span es lo que gobierna la purga, y aquí es donde más se falla (§8.2): purgar
    solo la etiqueta deja en el entrenamiento observaciones cuyas *features* ya
    contienen el periodo de test. Para el ángulo B, con `PreEventFeatures` en
    `[T−30, T−1]`, un modelo de mercado estimado en `[T−250, T−40]` y una etiqueta
    `CAR[0, +60]`, el span es `[T−250, T+60]`: **310 sesiones**, un orden de
    magnitud más que el `post = 60` que uno pondría de forma ingenua.

    Parameters
    ----------
    reference:
        Fecha de referencia `T` de cada observación (p.ej. `event_date`).
    lookback, lookahead:
        Periodos hacia atrás y hacia adelante. Si se pasa `timeline` se cuentan en
        **pasos de esa rejilla** (sesiones de mercado, trimestres fiscales…); si no,
        en días naturales cuando las fechas son temporales, o en unidades nativas
        cuando son numéricas.
    timeline:
        Rejilla ordenada sobre la que contar (típicamente
        `TradingCalendar.sessions(...)`). Contar sesiones y no días naturales
        importa: 60 días naturales son ~41 sesiones.
    ids:
        Índice del resultado. Por defecto el de `reference`.
    """
    if lookback < 0 or lookahead < 0:
        msg = f"lookback y lookahead deben ser >= 0; recibidos {lookback}, {lookahead}"
        raise ValueError(msg)
    ref = pd.Series(list(reference) if not isinstance(reference, pd.Series) else reference.to_numpy())
    if ids is not None:
        ref.index = pd.Index(list(ids))
    elif isinstance(reference, pd.Series):
        ref.index = reference.index

    if timeline is not None:
        grid = pd.Index(timeline)
        if not grid.is_monotonic_increasing:
            grid = grid.sort_values()
        pos = grid.searchsorted(ref.to_numpy(), side="left")
        pos = np.clip(pos, 0, len(grid) - 1)
        lo = np.clip(pos - lookback, 0, len(grid) - 1)
        hi = np.clip(pos + lookahead, 0, len(grid) - 1)
        t0 = grid.to_numpy()[lo]
        t1 = grid.to_numpy()[hi]
    elif pd.api.types.is_datetime64_any_dtype(pd.Series(ref)):
        stamps = pd.to_datetime(ref)
        t0 = (stamps - pd.Timedelta(days=lookback)).to_numpy()
        t1 = (stamps + pd.Timedelta(days=lookahead)).to_numpy()
    else:
        values = ref.to_numpy(dtype=float)
        t0 = values - lookback
        t1 = values + lookahead
    return pd.DataFrame({"t0": t0, "t1": t1}, index=ref.index)


def _span_arrays(spans: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extrae `t0` y `t1` validando que el span esté bien formado."""
    if not isinstance(spans, pd.DataFrame) or not {"t0", "t1"} <= set(spans.columns):
        msg = "spans debe ser un DataFrame con columnas 't0' y 't1'"
        raise DataQualityError(msg)
    t0 = spans["t0"].to_numpy()
    t1 = spans["t1"].to_numpy()
    if len(t0) == 0:
        msg = "spans está vacío: no hay observaciones que partir"
        raise InsufficientHistory(msg)
    if np.any(pd.isna(t0)) or np.any(pd.isna(t1)):
        msg = "hay spans con t0 o t1 nulos; una observación sin span no se puede purgar"
        raise DataQualityError(msg)
    if np.any(t1 < t0):
        bad = int(np.sum(t1 < t0))
        msg = f"{bad} spans tienen t1 < t0: el span de información va hacia atrás"
        raise DataQualityError(msg)
    return t0, t1


def _grid_positions(
    spans: pd.DataFrame, timeline: Sequence[Any] | None
) -> tuple[np.ndarray, np.ndarray, int]:
    """Proyecta los spans sobre una rejilla entera, redondeando **hacia fuera**.

    Redondear hacia fuera (t0 al punto anterior, t1 al posterior) hace la purga
    conservadora: ante la duda se elimina más entrenamiento, nunca menos. El error
    en la dirección contraria es look-ahead silencioso.
    """
    t0, t1 = _span_arrays(spans)
    if timeline is not None:
        grid = pd.Index(timeline)
        if not grid.is_monotonic_increasing:
            grid = grid.sort_values()
        grid_values = grid.to_numpy()
    else:
        grid_values = np.unique(np.concatenate([t0, t1]))
    n_grid = len(grid_values)
    g0 = np.clip(np.searchsorted(grid_values, t0, side="right") - 1, 0, n_grid - 1)
    g1 = np.clip(np.searchsorted(grid_values, t1, side="left"), 0, n_grid - 1)
    return g0.astype(np.int64), g1.astype(np.int64), n_grid


def purge_and_embargo(
    g0: np.ndarray,
    g1: np.ndarray,
    test_positions: np.ndarray,
    *,
    embargo_steps: int = 0,
) -> np.ndarray:
    """Índices de entrenamiento tras purgar solapamientos y aplicar embargo.

    **Purga.** Se elimina toda observación `i` para la que exista `j` en test con
    `[t_i0, t_i1] ∩ [t_j0, t_j1] ≠ ∅`. Una observación de entrenamiento cuya
    etiqueta comparte periodo con una de test comparte el **mismo retorno
    realizado**: el modelo ve la respuesta.

    **Embargo.** La purga no basta: una observación de entrenamiento
    inmediatamente posterior al test, aunque no solape, está correlacionada con él
    por la persistencia serial del mercado. Se eliminan además las observaciones
    cuyo span **empieza** en `(t_test_fin, t_test_fin + e]`. El embargo es
    **asimétrico**, solo hacia adelante: hacia atrás ya lo cubre la purga.

    Parameters
    ----------
    g0, g1:
        Posiciones enteras del inicio y fin del span de cada observación.
    test_positions:
        Posiciones (índices en `g0`/`g1`) del conjunto de test.
    embargo_steps:
        Longitud del embargo en pasos de la rejilla.

    Returns
    -------
    numpy.ndarray
        Posiciones de entrenamiento supervivientes, ordenadas.
    """
    n = g0.size
    test_positions = np.asarray(test_positions, dtype=np.int64)
    mask_test = np.zeros(n, dtype=bool)
    mask_test[test_positions] = True
    if not mask_test.any():
        msg = "el conjunto de test está vacío"
        raise InsufficientHistory(msg)

    test_g0 = g0[mask_test]
    test_g1 = g1[mask_test]

    # Purga: i solapa con algún j de test <=> existe j con t0_j <= t1_i y t1_j >= t0_i.
    # Ordenando el test por t0 y acumulando el máximo de t1, la condición se resuelve
    # con una búsqueda binaria por observación en lugar de un producto cartesiano.
    order = np.argsort(test_g0, kind="stable")
    sorted_g0 = test_g0[order]
    running_max_g1 = np.maximum.accumulate(test_g1[order])

    upto = np.searchsorted(sorted_g0, g1, side="right")  # cuántos test tienen t0_j <= t1_i
    max_g1_prefix = np.where(upto > 0, running_max_g1[np.clip(upto - 1, 0, None)], -(2**62))
    overlaps = (upto > 0) & (max_g1_prefix >= g0)

    keep = ~overlaps & ~mask_test
    if embargo_steps > 0:
        # El embargo se aplica tras el fin de CADA segmento de test, no solo
        # tras el último: `CombinatorialPurgedCV` pasa k grupos NO contiguos
        # como un único test, y embargar solo tras `max(test_g1)` dejaría en
        # el entrenamiento las observaciones inmediatamente posteriores a los
        # demás bloques — exactamente la fuga que el embargo debe eliminar.
        # Embargar tras cada `t1` de test es un superconjunto seguro del
        # embargo por bloque: las posiciones interiores a un bloque ya están
        # excluidas por ser test o por la purga.
        ends = np.unique(test_g1)
        pos = np.searchsorted(ends, g0, side="left") - 1
        prev_end = np.where(pos >= 0, ends[np.clip(pos, 0, None)], -(2**62))
        embargoed = (g0 > prev_end) & (g0 <= prev_end + embargo_steps)
        keep &= ~embargoed
    return np.flatnonzero(keep)


def _embargo_steps(embargo: float | int, n_grid: int) -> int:
    """Traduce el embargo a pasos de rejilla. `0 < embargo < 1` = fracción de `T`."""
    if embargo < 0:
        msg = f"el embargo no puede ser negativo; recibido {embargo}"
        raise ValueError(msg)
    if 0 < embargo < 1:
        return int(math.ceil(embargo * n_grid))
    return int(embargo)


@dataclass(slots=True)
class PurgedKFold:
    """Validación cruzada en `k` bloques contiguos, con purga y embargo.

    La `KFold` de scikit-learn asume observaciones intercambiables. En un panel
    financiero con etiquetas que abarcan `h` periodos eso es sencillamente falso, y
    el resultado es un OOS optimista que no significa nada.

    **Coste medido** (§8.4, obs. eliminadas ≈ `2·K·(h + e)`): con `K=5`, `h=21` y
    embargo 12 se pierde el 9,5 % del panel — asumible para el ángulo A. Con
    `K=10`, `h=63` y embargo 35 se pierde el **56,3 %**. Más folds no es gratis. Con
    el span de 310 sesiones del ángulo B, `K=10` es directamente inviable y la
    partición debe hacerse **por trimestre fiscal**, no por día.

    Parameters
    ----------
    n_splits:
        Número de bloques de test contiguos.
    embargo:
        Fracción de la rejilla temporal (si está en `(0,1)`) o número de pasos.
        López de Prado recomienda ≈ 1 % de `T`.
    timeline:
        Rejilla temporal sobre la que medir el embargo (p.ej. las sesiones del
        calendario). Si es `None` se usa la rejilla implícita de los propios spans.
    """

    n_splits: int = 5
    embargo: float = 0.01
    timeline: Sequence[Any] | None = None

    def __post_init__(self) -> None:
        if self.n_splits < 2:
            msg = f"n_splits debe ser >= 2; recibido {self.n_splits}"
            raise ValueError(msg)

    def get_n_splits(self) -> int:
        """Compatibilidad con la API de scikit-learn."""
        return self.n_splits

    def split(self, spans: pd.DataFrame) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Genera `(train_positions, test_positions)` como índices posicionales.

        Los índices son **posicionales respecto al orden de entrada** de `spans`,
        de modo que `X.iloc[train]` / `X.iloc[test]` funcionan directamente aunque
        `spans` no esté ordenado por fecha.
        """
        g0, g1, n_grid = _grid_positions(spans, self.timeline)
        n = g0.size
        if n < self.n_splits:
            msg = f"{n} observaciones para {self.n_splits} folds"
            raise InsufficientHistory(msg)
        steps = _embargo_steps(self.embargo, n_grid)
        order = np.argsort(g0, kind="stable")
        for chunk in np.array_split(order, self.n_splits):
            if chunk.size == 0:  # pragma: no cover - imposible con n >= n_splits
                continue
            test = np.sort(chunk)
            train = purge_and_embargo(g0, g1, test, embargo_steps=steps)
            yield train, test

    def purge_report(self, spans: pd.DataFrame) -> pd.DataFrame:
        """Cuánto cuesta la purga, fold a fold.

        Reportarlo no es cosmético: si la purga se come el 56 % del panel, el
        modelo entrenado no es el mismo modelo, y hay que rediseñar la partición
        antes de interpretar nada.
        """
        rows = []
        total = len(spans)
        for i, (train, test) in enumerate(self.split(spans)):
            purged = total - len(train) - len(test)
            rows.append(
                {
                    "fold": i,
                    "n_train": len(train),
                    "n_test": len(test),
                    "n_purged": purged,
                    "purged_pct": 100.0 * purged / total,
                }
            )
        return pd.DataFrame(rows).set_index("fold")


@dataclass(frozen=True, slots=True)
class CPCVSplit:
    """Un split de CPCV: entrenamiento, test y qué grupos forman el test."""

    train: np.ndarray
    test: np.ndarray
    test_groups: tuple[int, ...]
    split_id: int


@dataclass(frozen=True, slots=True)
class CPCVGeometry:
    """Geometría combinatoria de un diseño CPCV.

    Distinguir `n_splits` de `n_paths` es el error nº 8 de §15: con `N=6, k=2` hay
    **15 splits** pero solo **5 trayectorias**.
    """

    n_groups: int
    n_test_groups: int
    n_splits: int
    n_paths: int
    train_fraction: float

    def __str__(self) -> str:  # pragma: no cover - cosmético
        return (
            f"CPCV(N={self.n_groups}, k={self.n_test_groups}): {self.n_splits} splits, "
            f"{self.n_paths} trayectorias, {100 * self.train_fraction:.0f}% entrenamiento"
        )


def cpcv_paths(n_groups: int, n_test_groups: int) -> CPCVGeometry:
    """Número de splits y de **trayectorias** de un diseño CPCV.

    ``φ[N, k] = C(N, k) · k / N``, que es igual a `C(N−1, k−1)`.

    La CV purgada con `K` folds produce **una sola** trayectoria de backtest; la
    CPCV produce muchas, y de ahí sale la distribución del Sharpe (mediana, p5, p95)
    en vez de un número suelto. Configuración recomendada por el repo: ángulo A
    `N=12, k=2` (66 splits, 11 trayectorias, 83 % de entrenamiento); ángulo B por
    trimestre fiscal `N=10, k=2` (45 splits, 9 trayectorias).

    Examples
    --------
    >>> g = cpcv_paths(10, 2)
    >>> g.n_splits, g.n_paths
    (45, 9)
    >>> g = cpcv_paths(6, 2)
    >>> g.n_splits, g.n_paths
    (15, 5)
    """
    if n_groups < 2:
        msg = f"n_groups debe ser >= 2; recibido {n_groups}"
        raise ValueError(msg)
    if not 1 <= n_test_groups < n_groups:
        msg = f"n_test_groups debe estar en [1, n_groups); recibido {n_test_groups}"
        raise ValueError(msg)
    n_splits = math.comb(n_groups, n_test_groups)
    n_paths = math.comb(n_groups - 1, n_test_groups - 1)
    return CPCVGeometry(
        n_groups=n_groups,
        n_test_groups=n_test_groups,
        n_splits=n_splits,
        n_paths=n_paths,
        train_fraction=(n_groups - n_test_groups) / n_groups,
    )


@dataclass(slots=True)
class CombinatorialPurgedCV:
    """Validación cruzada purgada **combinatoria** (López de Prado, 2018).

    Parte el panel en `n_groups` grupos contiguos y usa **todas** las combinaciones
    de `n_test_groups` grupos como test, purgando y embargando en cada frontera. Las
    predicciones OOS se recombinan en `φ[N,k]` trayectorias completas, cada una con
    su propio Sharpe.

    El resultado no es un Sharpe OOS, es una **distribución** de Sharpes. Eso es lo
    que permite cumplir el mandato de `ARCHITECTURE.md` §3.8 (toda métrica con su
    banda) y lo que hace que el criterio 11 de §14.1 —*p5 del Sharpe entre
    trayectorias > 0*— sea formulable.

    No confundir con el PBO: la CPCV valida **una** estrategia fija; el PBO valida
    **el procedimiento de selección** entre configuraciones (§8.6). El repo exige
    los dos.
    """

    n_groups: int = 12
    n_test_groups: int = 2
    embargo: float = 0.01
    timeline: Sequence[Any] | None = None

    def __post_init__(self) -> None:
        _ = self.geometry  # valida n_groups / n_test_groups al construir

    @property
    def geometry(self) -> CPCVGeometry:
        """Geometría del diseño: splits, trayectorias y fracción de entrenamiento."""
        return cpcv_paths(self.n_groups, self.n_test_groups)

    @property
    def n_splits(self) -> int:
        return self.geometry.n_splits

    @property
    def n_paths(self) -> int:
        return self.geometry.n_paths

    def group_assignment(self, spans: pd.DataFrame) -> np.ndarray:
        """Grupo contiguo (0..N−1) de cada observación, ordenando por `t0`."""
        g0, _, _ = _grid_positions(spans, self.timeline)
        order = np.argsort(g0, kind="stable")
        groups = np.empty(g0.size, dtype=np.int64)
        for gid, chunk in enumerate(np.array_split(order, self.n_groups)):
            groups[chunk] = gid
        return groups

    def split(self, spans: pd.DataFrame) -> Iterator[CPCVSplit]:
        """Genera los `C(N,k)` splits purgados, en orden lexicográfico."""
        g0, g1, n_grid = _grid_positions(spans, self.timeline)
        if g0.size < self.n_groups:
            msg = f"{g0.size} observaciones para {self.n_groups} grupos"
            raise InsufficientHistory(msg)
        groups = self.group_assignment(spans)
        steps = _embargo_steps(self.embargo, n_grid)
        for split_id, combo in enumerate(combinations(range(self.n_groups), self.n_test_groups)):
            test = np.flatnonzero(np.isin(groups, combo))
            if test.size == 0:  # pragma: no cover
                continue
            train = purge_and_embargo(g0, g1, test, embargo_steps=steps)
            yield CPCVSplit(train=train, test=test, test_groups=tuple(combo), split_id=split_id)

    def path_matrix(self) -> np.ndarray:
        """Matriz `(n_paths, n_groups)` con el split que aporta el OOS de cada grupo.

        Cada grupo aparece en el test de exactamente `C(N−1, k−1) = φ[N,k]` splits.
        La trayectoria `p` toma, para cada grupo, el `p`-ésimo de esos splits: así se
        reconstruyen `φ[N,k]` recorridos completos y disjuntos del panel.
        """
        combos = list(combinations(range(self.n_groups), self.n_test_groups))
        matrix = np.full((self.n_paths, self.n_groups), -1, dtype=np.int64)
        for gid in range(self.n_groups):
            owners = [i for i, c in enumerate(combos) if gid in c]
            matrix[:, gid] = owners[: self.n_paths]
        return matrix

    def assemble_paths(
        self,
        predictions: Mapping[int, pd.Series],
        spans: pd.DataFrame,
    ) -> list[pd.Series]:
        """Recombina las predicciones OOS de cada split en trayectorias completas.

        Cada grupo es evaluado OOS por `φ[N,k]` splits distintos; la trayectoria `p`
        toma el `p`-ésimo. El resultado son `n_paths` recorridos que cubren el panel
        entero **una sola vez cada uno**, es decir, `n_paths` backtests OOS legítimos
        en lugar del único que da la CV purgada clásica.

        Parameters
        ----------
        predictions:
            `split_id -> Series` de predicciones OOS, indexada con las etiquetas del
            panel (subconjunto de `spans.index`).
        spans:
            Los mismos spans con los que se generaron los splits.
        """
        matrix = self.path_matrix()
        groups = pd.Series(self.group_assignment(spans), index=spans.index)
        paths: list[pd.Series] = []
        for p in range(self.n_paths):
            pieces: list[pd.Series] = []
            for gid in range(self.n_groups):
                split_id = int(matrix[p, gid])
                series = predictions.get(split_id)
                if series is None:
                    continue
                labels = groups.index[groups.to_numpy() == gid]
                pieces.append(series.reindex(series.index.intersection(labels)))
            if pieces:
                paths.append(pd.concat(pieces).sort_index())
        return paths


# =========================================================================== #
# 2. Bootstrap estacionario de Politis–Romano                                  #
# =========================================================================== #


def block_length_from_autocovariance(
    gamma: Sequence[float] | np.ndarray,
    n_obs: int,
    *,
    weights: Sequence[float] | np.ndarray | None = None,
) -> float:
    """Longitud óptima de bloque a partir de las autocovarianzas de un lado.

    Fórmula de Politis–White (2004) con la corrección de Patton, Politis y White
    (2009) para el bootstrap **estacionario**::

        b_opt = ( 2·Ĝ² / D̂_SB )^(1/3) · T^(1/3)

    con ``Ĝ = Σ_k |k|·γ̂_k`` y ``D̂_SB = 2·(Σ_k γ̂_k)²`` sobre `k ∈ [−M, M]`.

    La dependencia `T^(1/3)` es la firma característica del método: **la longitud de
    bloque crece con la muestra**, no es una constante que se fija una vez.

    Comprobación analítica para un AR(1) con `γ_k = ρ^{|k|}`, que da la forma
    cerrada `b = (2ρ/((1−ρ)(1+ρ)))^{2/3}·T^{1/3}`: con `T = 500` produce 4,4 / 7,7 /
    12,1 / 21,5 para `ρ` = 0,2 / 0,4 / 0,6 / 0,8.
    """
    g = np.asarray(gamma, dtype=float)
    if g.size < 1 or not np.isfinite(g[0]) or g[0] <= 0:
        msg = "gamma[0] (la varianza) debe ser finita y positiva"
        raise DataQualityError(msg)
    if n_obs < 2:
        msg = f"n_obs debe ser >= 2; recibido {n_obs}"
        raise InsufficientHistory(msg)
    k = np.arange(g.size, dtype=float)
    w = np.ones_like(g) if weights is None else np.asarray(weights, dtype=float)
    # Sumas de dos lados: el término k=0 aparece una vez, los k>0 dos veces.
    g_hat = 2.0 * float(np.sum(w[1:] * k[1:] * g[1:]))
    lrv = float(g[0] * w[0] + 2.0 * np.sum(w[1:] * g[1:]))
    d_hat = 2.0 * lrv * lrv
    if d_hat <= 0 or g_hat <= 0:
        return 1.0
    b = (2.0 * g_hat * g_hat / d_hat) ** (1.0 / 3.0) * n_obs ** (1.0 / 3.0)
    return float(max(b, 1.0))


def politis_white_block_length(x: np.ndarray | pd.Series, *, max_block: int | None = None) -> float:
    """Longitud media de bloque estimada de la propia serie (Politis–White 2004).

    Usa el kernel *flat-top* de Politis–Romano y la regla del "primer lag a partir
    del cual `K_N` autocorrelaciones consecutivas son insignificantes" para elegir
    el truncamiento `M`.

    **La sensibilidad a `L` es asimétrica** (§9.3): pasarse de largo es barato
    (con `ρ=0,2`, `L=40` da 0,900 de cobertura frente al 0,927 óptimo), quedarse
    corto es caro (`L=1` da 0,880 y con `ρ=0,8` se desploma a 0,507). **Ante la
    duda, `L` grande**; por eso el resultado se redondea hacia arriba en
    `stationary_bootstrap`.
    """
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 8:
        msg = f"se necesitan al menos 8 observaciones; hay {n}"
        raise InsufficientHistory(msg)
    d = arr - arr.mean()
    var = float(d @ d) / n
    if var <= 0:
        return 1.0

    k_n = max(5, int(math.ceil(math.sqrt(math.log10(n)))))
    m_max = int(math.ceil(math.sqrt(n))) + k_n
    m_max = min(m_max, n - 2)
    acf = np.array([float(d[k:] @ d[:-k]) / n / var for k in range(1, m_max + 1)])
    crit = 2.0 * math.sqrt(math.log10(n) / n)

    m_hat = m_max
    for m in range(1, m_max - k_n + 1):
        window = np.abs(acf[m : m + k_n])
        if np.all(window < crit):
            m_hat = m
            break
    trunc = int(min(2 * m_hat, m_max))
    trunc = max(trunc, 1)

    lags = np.arange(trunc + 1)
    ratio = lags / trunc
    kernel = np.where(ratio <= 0.5, 1.0, np.where(ratio <= 1.0, 2.0 * (1.0 - ratio), 0.0))
    gamma = np.concatenate([[var], acf[:trunc] * var])
    b = block_length_from_autocovariance(gamma, n, weights=kernel)

    ceiling = max_block if max_block is not None else int(math.ceil(min(3.0 * math.sqrt(n), n / 3.0)))
    return float(min(b, max(ceiling, 1)))


def stationary_bootstrap_indices(
    n_obs: int,
    *,
    block_length: float,
    n_boot: int,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Índices del bootstrap estacionario de Politis–Romano (1994).

    Bloques de **longitud aleatoria geométrica** de media `L = 1/p`, lo que preserva
    la estacionariedad de la serie remuestreada (el bootstrap por bloques de
    longitud fija no lo hace)::

        I_1 ~ U{1..T}
        para t = 2..T:
            con prob. p:      I_t ~ U{1..T}            (nuevo bloque)
            con prob. 1 − p:  I_t = (I_{t−1} mod T)+1   (continuar, envolvente)

    El **envolvente** (`mod T`) es esencial: sin él las observaciones del final de la
    muestra se submuestrean y el estimador queda sesgado.

    `block_length = 1` recupera el bootstrap iid, que sobre series
    autocorrelacionadas tiene una cobertura real de 0,50 para un nominal de 0,95
    con `ρ = 0,8`. No usarlo nunca sobre retornos.

    Returns
    -------
    numpy.ndarray
        Matriz `(n_boot, n_obs)` de índices enteros en `[0, n_obs)`.
    """
    if n_obs < 2:
        msg = f"n_obs debe ser >= 2; recibido {n_obs}"
        raise InsufficientHistory(msg)
    if n_boot < 1:
        msg = f"n_boot debe ser >= 1; recibido {n_boot}"
        raise ValueError(msg)
    if block_length < 1:
        msg = f"la longitud media de bloque debe ser >= 1; recibida {block_length}"
        raise ValueError(msg)
    generator = rng if rng is not None else np.random.default_rng(seed)
    p = 1.0 / float(block_length)

    starts = generator.integers(0, n_obs, size=(n_boot, n_obs), dtype=np.int64)
    new_block = generator.random((n_boot, n_obs)) < p
    new_block[:, 0] = True
    positions = np.arange(n_obs, dtype=np.int64)[None, :]
    anchors = np.maximum.accumulate(np.where(new_block, positions, -1), axis=1)
    rows = np.arange(n_boot, dtype=np.int64)[:, None]
    idx = (starts[rows, anchors] + (positions - anchors)) % n_obs
    return idx.astype(np.int64)


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """Estimación puntual con su intervalo de confianza por bootstrap."""

    value: float
    ci_low: float
    ci_high: float
    std_error: float
    block_length: float
    n_boot: int
    alpha: float
    method: str
    replicates: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))

    @property
    def excludes_zero(self) -> bool:
        """Criterio 16 de §14.1: el IC bootstrap al 95 % debe excluir el cero."""
        return self.ci_low > 0.0 or self.ci_high < 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "value": self.value,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "std_error": self.std_error,
            "block_length": self.block_length,
            "n_boot": self.n_boot,
            "method": self.method,
        }


def _percentile_ci(reps: np.ndarray, alpha: float) -> tuple[float, float]:
    lo = float(np.percentile(reps, 100.0 * alpha / 2.0))
    hi = float(np.percentile(reps, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def _bca_ci(
    reps: np.ndarray,
    observed: float,
    jackknife: np.ndarray,
    alpha: float,
) -> tuple[float, float]:
    """Intervalo BCa: corrige sesgo y aceleración del bootstrap de percentiles.

    El bootstrap de percentiles sub-cubre en muestras moderadas (§9.3, lectura 4:
    ≈ 0,92 frente a un nominal de 0,95 con `T = 500` incluso sin dependencia). BCa
    corrige el sesgo mediano y la asimetría; el jackknife se hace **por bloques**
    para no destruir la dependencia serial al estimar la aceleración.
    """
    b = reps.size
    prop = float(np.mean(reps < observed))
    if prop <= 0.0 or prop >= 1.0:
        return _percentile_ci(reps, alpha)
    z0 = float(stats.norm.ppf(prop))
    jack_mean = float(np.mean(jackknife))
    diffs = jack_mean - jackknife
    denom = 6.0 * (float(np.sum(diffs**2)) ** 1.5)
    accel = float(np.sum(diffs**3)) / denom if denom > 0 else 0.0

    z_lo = float(stats.norm.ppf(alpha / 2.0))
    z_hi = float(stats.norm.ppf(1.0 - alpha / 2.0))

    def adjust(z: float) -> float:
        num = z0 + z
        den = 1.0 - accel * num
        if den <= 0:
            return float("nan")
        return float(stats.norm.cdf(z0 + num / den))

    q_lo, q_hi = adjust(z_lo), adjust(z_hi)
    if not (np.isfinite(q_lo) and np.isfinite(q_hi)) or q_lo >= q_hi:
        return _percentile_ci(reps, alpha)
    return (
        float(np.percentile(reps, 100.0 * q_lo)),
        float(np.percentile(reps, 100.0 * q_hi)),
    )


def stationary_bootstrap(
    data: np.ndarray | pd.Series | pd.DataFrame,
    statistic: Callable[[Any], float] | None = None,
    *,
    block_length: float | None = None,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int | None = 20260804,
    method: Literal["percentile", "basic", "bca"] = "percentile",
    horizon: int = 1,
    n_jackknife: int = 40,
) -> BootstrapResult:
    """Intervalo de confianza por bootstrap estacionario sobre el eje temporal.

    Si `data` es un DataFrame se remuestrean **filas completas**. Eso es
    deliberado y es la decisión de diseño de §9.4: al remuestrear un panel, la
    unidad debe ser la **fecha con toda su sección cruzada**, porque remuestrear
    activos destruye la correlación transversal y produce intervalos demasiado
    estrechos. Para el ángulo B el átomo debe ser aún más grueso: la fecha de evento
    con todos sus eventos, o el trimestre fiscal completo.

    Parameters
    ----------
    statistic:
        Función del remuestreo a un escalar. Por defecto, la media.
    block_length:
        Longitud media de bloque. Si es `None` se estima con Politis–White sobre la
        serie (o sobre la media transversal por fecha si `data` es un panel), con
        suelo `L ≥ horizon` cuando hay solapamiento, y se redondea hacia arriba.
    n_boot:
        `B ≥ 1.000` para intervalos, `B ≥ 5.000` para p-valores de cola (§9.3).
    method:
        `"percentile"`, `"basic"` o `"bca"`. Para intervalos críticos se prefiere
        BCa, dado el sesgo liberal documentado del de percentiles.
    """
    if not 0.0 < alpha < 1.0:
        msg = f"alpha debe estar en (0,1); recibido {alpha}"
        raise ValueError(msg)
    func: Callable[[Any], float] = statistic if statistic is not None else (
        lambda z: float(np.nanmean(np.asarray(z, dtype=float)))
    )

    is_frame = isinstance(data, pd.DataFrame)
    if is_frame:
        values: Any = data
        n = len(data)
        reference = data.mean(axis=1).to_numpy(dtype=float)
    else:
        arr = np.asarray(data, dtype=float)
        if arr.ndim != 1:
            msg = "para datos multidimensionales pasa un DataFrame (se remuestrean filas)"
            raise DataQualityError(msg)
        arr = arr[np.isfinite(arr)]
        values = arr
        n = arr.size
        reference = arr
    if n < 8:
        msg = f"se necesitan al menos 8 observaciones para remuestrear; hay {n}"
        raise InsufficientHistory(msg)

    if block_length is None:
        estimated = politis_white_block_length(reference)
        block_length = float(max(math.ceil(estimated), horizon))
    block_length = float(max(block_length, 1.0))

    idx = stationary_bootstrap_indices(n, block_length=block_length, n_boot=n_boot, seed=seed)
    if is_frame:
        frame = data
        reps = np.array([func(frame.iloc[row]) for row in idx], dtype=float)
    else:
        reps = np.array([func(values[row]) for row in idx], dtype=float)
    reps = reps[np.isfinite(reps)]
    if reps.size < max(10, n_boot // 10):
        msg = (
            f"solo {reps.size} de {n_boot} réplicas produjeron un estadístico finito: "
            "el estadístico no es estable bajo remuestreo"
        )
        raise DataQualityError(msg)

    observed = float(func(values))
    se = float(np.std(reps, ddof=1))

    if method == "percentile":
        lo, hi = _percentile_ci(reps, alpha)
    elif method == "basic":
        p_lo, p_hi = _percentile_ci(reps, alpha)
        lo, hi = 2.0 * observed - p_hi, 2.0 * observed - p_lo
    elif method == "bca":
        blocks = np.array_split(np.arange(n), min(n_jackknife, n))
        jack = []
        for b_idx in blocks:
            keep = np.setdiff1d(np.arange(n), b_idx, assume_unique=True)
            if keep.size < 2:
                continue
            jack.append(func(values.iloc[keep] if is_frame else values[keep]))
        lo, hi = _bca_ci(reps, observed, np.asarray(jack, dtype=float), alpha)
    else:  # pragma: no cover - validado por el Literal
        msg = f"método de intervalo desconocido: {method!r}"
        raise ValueError(msg)

    return BootstrapResult(
        value=observed,
        ci_low=float(lo),
        ci_high=float(hi),
        std_error=se,
        block_length=float(block_length),
        n_boot=int(reps.size),
        alpha=alpha,
        method=f"stationary_bootstrap_{method}",
        replicates=reps,
    )


# =========================================================================== #
# 3. Comparaciones múltiples                                                   #
# =========================================================================== #


@dataclass(frozen=True, slots=True)
class MultipleTestResult:
    """Resultado de un procedimiento de multiplicidad sobre una familia."""

    method: str
    alpha: float
    p_values: np.ndarray = field(repr=False)
    adjusted: np.ndarray = field(repr=False)
    rejected: np.ndarray = field(repr=False)
    names: tuple[str, ...] = ()

    @property
    def n_tests(self) -> int:
        return int(self.p_values.size)

    @property
    def n_rejected(self) -> int:
        """Cuántas hipótesis sobreviven a la corrección."""
        return int(np.sum(self.rejected))

    @property
    def survivors(self) -> tuple[str, ...]:
        """Nombres de las señales que sobreviven."""
        if not self.names:
            return tuple(str(i) for i in np.flatnonzero(self.rejected))
        return tuple(n for n, r in zip(self.names, self.rejected, strict=True) if r)

    def to_frame(self) -> pd.DataFrame:
        names = self.names or tuple(str(i) for i in range(self.n_tests))
        return pd.DataFrame(
            {
                "p_value": self.p_values,
                f"p_adj_{self.method}": self.adjusted,
                f"reject_{self.method}": self.rejected,
            },
            index=pd.Index(names, name="signal"),
        )


def _prepare_pvalues(
    p_values: Sequence[float] | np.ndarray | pd.Series | Mapping[str, float],
) -> tuple[np.ndarray, tuple[str, ...]]:
    if isinstance(p_values, Mapping) and not isinstance(p_values, pd.Series):
        p_values = pd.Series(p_values, dtype=float)
    names: tuple[str, ...] = ()
    if isinstance(p_values, pd.Series):
        names = tuple(str(i) for i in p_values.index)
        arr = p_values.to_numpy(dtype=float)
    else:
        arr = np.asarray(p_values, dtype=float)
    if arr.size == 0:
        msg = "no hay p-valores que corregir"
        raise InsufficientHistory(msg)
    if np.any(~np.isfinite(arr)):
        msg = "hay p-valores no finitos; un contraste que no produce p-valor no entra en la familia"
        raise DataQualityError(msg)
    if np.any((arr < 0) | (arr > 1)):
        msg = "hay p-valores fuera de [0,1]"
        raise DataQualityError(msg)
    return arr, names


def bonferroni(
    p_values: Sequence[float] | np.ndarray | pd.Series | Mapping[str, float],
    alpha: float = 0.05,
) -> MultipleTestResult:
    """Bonferroni (FWER, un paso): rechaza si `p ≤ α/M`.

    Controla la probabilidad de **cualquier** falso positivo. Muy conservador si los
    contrastes están correlacionados, que es exactamente el caso de una familia de
    factores fundamentales. Se reporta como referencia del umbral más exigente:
    **no hay razón para usarlo en vez de Holm**, que lo domina uniformemente.
    """
    p, names = _prepare_pvalues(p_values)
    adjusted = np.minimum(p * p.size, 1.0)
    return MultipleTestResult(
        method="bonferroni",
        alpha=alpha,
        p_values=p,
        adjusted=adjusted,
        rejected=adjusted <= alpha,
        names=names,
    )


def holm(
    p_values: Sequence[float] | np.ndarray | pd.Series | Mapping[str, float],
    alpha: float = 0.05,
) -> MultipleTestResult:
    """Holm (FWER, secuencial descendente): `p_(i) ≤ α/(M − i + 1)`.

    **Domina uniformemente a Bonferroni**: nunca rechaza menos, a veces más, con el
    mismo control de FWER. Es la elección para *confirmar* supervivientes cuando no
    se quiere pagar el coste computacional de Romano–Wolf.
    """
    p, names = _prepare_pvalues(p_values)
    m = p.size
    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    factors = m - np.arange(m)
    adj_sorted = np.maximum.accumulate(np.minimum(sorted_p * factors, 1.0))
    adjusted = np.empty_like(adj_sorted)
    adjusted[order] = np.minimum(adj_sorted, 1.0)
    return MultipleTestResult(
        method="holm",
        alpha=alpha,
        p_values=p,
        adjusted=adjusted,
        rejected=adjusted <= alpha,
        names=names,
    )


def benjamini_hochberg(
    p_values: Sequence[float] | np.ndarray | pd.Series | Mapping[str, float],
    alpha: float = 0.05,
) -> MultipleTestResult:
    """Benjamini–Hochberg (FDR, escalonado ascendente).

    Encuentra ``k = max{ i : p_(i) ≤ (i/M)·α }`` y rechaza `H_(1) … H_(k)`. P-valor
    ajustado::

        p_adj^(i) = min_{j ≥ i} min( (M/j)·p_(j) , 1 )

    Controla la **proporción esperada de falsos positivos entre los rechazos**, no
    la probabilidad de tener alguno. Válido bajo independencia y bajo dependencia
    positiva por regresión (PRDS), condición que se cumple razonablemente en una
    familia de factores correlacionados positivamente.

    BH es **adaptativo**: en el rango 1 coincide con Bonferroni, pero en el rango
    `M` el umbral es simplemente `α` (`t = 1,96`). Cuantos más rechazos hay, más
    permisivo se vuelve. Por eso es la elección correcta para **cribar** una familia
    grande, y Holm o Romano–Wolf para **confirmar** los supervivientes.

    El repo exige BH al 10 % dentro de la familia (criterio 13 de §14.1).
    """
    p, names = _prepare_pvalues(p_values)
    m = p.size
    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    ranks = np.arange(1, m + 1)
    scaled = sorted_p * m / ranks
    adj_sorted = np.minimum.accumulate(scaled[::-1])[::-1]
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = np.minimum(adj_sorted, 1.0)
    return MultipleTestResult(
        method="benjamini_hochberg",
        alpha=alpha,
        p_values=p,
        adjusted=adjusted,
        rejected=adjusted <= alpha,
        names=names,
    )


def _harmonic(m: int) -> float:
    """`c(M) = Σ_{j=1}^{M} 1/j`, la constante de Benjamini–Yekutieli."""
    return float(np.sum(1.0 / np.arange(1, m + 1)))


def benjamini_yekutieli(
    p_values: Sequence[float] | np.ndarray | pd.Series | Mapping[str, float],
    alpha: float = 0.05,
) -> MultipleTestResult:
    """Benjamini–Yekutieli: FDR bajo dependencia **arbitraria**.

    Sustituye `α` por `α/c(M)` con ``c(M) = Σ 1/j ≈ ln(M) + γ + 1/(2M)``. Es el
    procedimiento que usan Harvey–Liu–Zhu, y por tanto el que hay que aplicar si se
    quiere reproducir su marco en vez de citarlo de oídas.
    """
    p, names = _prepare_pvalues(p_values)
    base = benjamini_hochberg(p, alpha)
    c_m = _harmonic(p.size)
    adjusted = np.minimum(base.adjusted * c_m, 1.0)
    return MultipleTestResult(
        method="benjamini_yekutieli",
        alpha=alpha,
        p_values=p,
        adjusted=adjusted,
        rejected=adjusted <= alpha,
        names=names,
    )


def bonferroni_t(n_tests: int, alpha: float = 0.05, *, two_sided: bool = True) -> float:
    """Umbral de `t` bajo Bonferroni con aproximación normal.

    Cross-check independiente: Harvey–Liu–Zhu reportan **3,78** para su conjunto de
    316 factores; esta función devuelve 3,7778. La aritmética reproduce su tabla.

    Examples
    --------
    >>> round(bonferroni_t(316), 3)
    3.778
    """
    if n_tests < 1:
        msg = f"n_tests debe ser >= 1; recibido {n_tests}"
        raise ValueError(msg)
    if not 0.0 < alpha < 1.0:
        msg = f"alpha debe estar en (0,1); recibido {alpha}"
        raise ValueError(msg)
    tail = alpha / n_tests
    if two_sided:
        tail /= 2.0
    return float(stats.norm.ppf(1.0 - tail))


def benjamini_yekutieli_t(
    n_tests: int, alpha: float = 0.05, *, rank: int = 1, two_sided: bool = True
) -> float:
    """Umbral de `t` bajo BHY para el rango `i`: `p_(i) ≤ (i/(M·c(M)))·α`.

    Con `M = 316` y `rank = 1` devuelve 4,215, más exigente que el 3,778 de
    Bonferroni: BHY en el rango peor es más estricto porque reparte el `α` entre
    `M·c(M)` en vez de entre `M`.
    """
    if not 1 <= rank <= n_tests:
        msg = f"rank debe estar en [1, n_tests]; recibido {rank}"
        raise ValueError(msg)
    tail = alpha * rank / (n_tests * _harmonic(n_tests))
    if two_sided:
        tail /= 2.0
    return float(stats.norm.ppf(1.0 - tail))


def harvey_liu_zhu_threshold(kind: Literal["new", "established"] = "new") -> float:
    """Umbral de `t` de Harvey–Liu–Zhu (2016) según la postura del repo (§7.3).

    HLZ recopilan 316 factores publicados y argumentan que, dado el *data mining*
    acumulado, un factor nuevo debe superar **`t > 3,0`**. Con ese criterio solo
    **9 de 313** variables sobreviven.

    **Matización cuantitativa obligatoria.** El p-valor bilateral de `t = 3,0` es
    0,0027; bajo Bonferroni con `α = 0,05` eso equivale a `M = 18,5` pruebas. Es
    decir, `t > 3` es el umbral Bonferroni de apenas **19** pruebas: con las 316 de
    su propio conjunto haría falta `t > 3,78`. El 3,0 **no es un listón
    conservador, es el mínimo**.

    **Contrapunto.** Chen y Zimmermann (2020) estiman el sesgo de publicación sobre
    156 carteras y encuentran que los retornos ajustados son solo un 12 % menores,
    con un umbral tan bajo como `t > 1,8`. El repo no resuelve la disputa por
    decreto: exige `t_NW > 3,0` a un factor **nuevo** y `t_NW > 2,0` a uno con
    **literatura previa robusta** (SUE, PEAD, accruals, Piotroski), porque en ese
    caso la hipótesis no fue generada por estos datos.
    """
    if kind == "new":
        return 3.0
    if kind == "established":
        return 2.0
    msg = f"kind debe ser 'new' o 'established'; recibido {kind!r}"
    raise ValueError(msg)


def hlz_equivalent_tests(t_threshold: float = 3.0, alpha: float = 0.05) -> float:
    """A cuántas pruebas de Bonferroni equivale un umbral de `t`.

    ``M_equivalente = α / p(t)``. Con `t = 3` y `α = 5 %` da **18,5**: el famoso
    "t > 3" es el umbral Bonferroni de 19 pruebas, no de 316.

    Examples
    --------
    >>> round(hlz_equivalent_tests(3.0), 1)
    18.5
    """
    p = float(2.0 * stats.norm.sf(abs(t_threshold)))
    if p <= 0:
        return float("inf")
    return float(alpha / p)


# --------------------------------------------------------------------------- #
# Romano–Wolf                                                                   #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class RomanoWolfResult:
    """Resultado del procedimiento *stepdown* de Romano–Wolf."""

    names: tuple[str, ...]
    t_stats: np.ndarray = field(repr=False)
    adjusted: np.ndarray = field(repr=False)
    rejected: np.ndarray = field(repr=False)
    alpha: float = 0.05
    n_boot: int = 0
    block_length: float = 1.0
    two_sided: bool = True

    @property
    def n_rejected(self) -> int:
        return int(np.sum(self.rejected))

    @property
    def survivors(self) -> tuple[str, ...]:
        return tuple(n for n, r in zip(self.names, self.rejected, strict=True) if r)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "t_stat": self.t_stats,
                "p_adj_romano_wolf": self.adjusted,
                "reject_romano_wolf": self.rejected,
            },
            index=pd.Index(self.names, name="signal"),
        )


def romano_wolf(
    contributions: pd.DataFrame,
    *,
    alpha: float = 0.05,
    n_boot: int = 2000,
    block_length: float | None = None,
    seed: int | None = 20260804,
    two_sided: bool = True,
    horizon: int = 1,
) -> RomanoWolfResult:
    """Contraste múltiple *stepdown* de Romano–Wolf con bootstrap por bloques.

    Es el más potente de los procedimientos que controlan FWER porque **estima la
    dependencia entre contrastes por bootstrap** en vez de asumir el peor caso. Para
    una familia de factores correlacionados como la de este repo es estrictamente
    mejor que Bonferroni/Holm: gana potencia sin perder control de FWER. Su coste es
    `B × M` evaluaciones, por lo que se exige en la promoción final y no en el
    cribado (§7.1).

    Parameters
    ----------
    contributions:
        DataFrame `T × M`: cada columna es la serie temporal de la contribución
        periódica de una señal (retorno del *spread*, IC diaria, diferencial frente
        al benchmark). Las filas se remuestrean **conjuntamente**, que es lo que
        preserva la correlación entre señales.
    horizon:
        Horizonte de solapamiento, usado como suelo de la longitud de bloque.

    Notes
    -----
    Los estadísticos se studentizan con el error estándar de Newey–West de cada
    columna, de modo que la dependencia temporal ya está corregida **antes** de
    corregir la multiplicidad, como exige el orden de §7.4.
    """
    if not isinstance(contributions, pd.DataFrame) or contributions.shape[1] < 1:
        msg = "contributions debe ser un DataFrame T x M con al menos una columna"
        raise DataQualityError(msg)
    data = contributions.dropna(how="any")
    n, m = data.shape
    if n < 20:
        msg = f"solo {n} periodos comunes a las {m} señales: insuficiente para bootstrap"
        raise InsufficientHistory(msg)

    values = data.to_numpy(dtype=float)
    means = values.mean(axis=0)
    lags = newey_west_lags(n, horizon)
    omega = np.array([math.sqrt(long_run_variance(values[:, j], lags)) for j in range(m)])
    if np.any(omega <= 0):
        bad = [data.columns[j] for j in np.flatnonzero(omega <= 0)]
        msg = f"varianza de largo plazo nula en {bad}: series constantes"
        raise DataQualityError(msg)
    t_stats = math.sqrt(n) * means / omega

    if block_length is None:
        block_length = float(max(math.ceil(politis_white_block_length(values.mean(axis=1))), horizon))
    idx = stationary_bootstrap_indices(n, block_length=block_length, n_boot=n_boot, seed=seed)
    boot_means = values[idx].mean(axis=1)  # (n_boot, m)
    t_boot = math.sqrt(n) * (boot_means - means) / omega
    if two_sided:
        t_boot = np.abs(t_boot)
        observed = np.abs(t_stats)
    else:
        observed = t_stats

    order = np.argsort(-observed, kind="stable")
    adjusted_sorted = np.empty(m, dtype=float)
    previous = 0.0
    for step in range(m):
        remaining = order[step:]
        max_boot = t_boot[:, remaining].max(axis=1)
        p_step = float(np.mean(max_boot >= observed[order[step]]))
        previous = max(previous, p_step)
        adjusted_sorted[step] = previous
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted

    return RomanoWolfResult(
        names=tuple(str(c) for c in data.columns),
        t_stats=t_stats,
        adjusted=adjusted,
        rejected=adjusted <= alpha,
        alpha=alpha,
        n_boot=n_boot,
        block_length=float(block_length),
        two_sided=two_sided,
    )


# --------------------------------------------------------------------------- #
# Informe integrado de multiplicidad                                            #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class MultipleTestingReport:
    """Cuántas señales de una familia sobreviven a cada corrección.

    Es la respuesta a la pregunta operativa del repo: *he probado `M` señales y `k`
    parecen significativas; ¿cuántas lo son de verdad?*
    """

    table: pd.DataFrame
    n_tests: int
    alpha: float
    n_survivors: dict[str, int]
    survivors: dict[str, tuple[str, ...]]
    notes: str = ""

    def summary(self) -> pd.DataFrame:
        """Tabla resumen: método, umbral conceptual y número de supervivientes."""
        rows = [
            {"method": k, "n_survivors": v, "fraction": v / self.n_tests}
            for k, v in self.n_survivors.items()
        ]
        return pd.DataFrame(rows).set_index("method")

    def __str__(self) -> str:  # pragma: no cover - cosmético
        parts = ", ".join(f"{k}={v}" for k, v in self.n_survivors.items())
        return f"MultipleTestingReport(M={self.n_tests}, α={self.alpha}: {parts})"


def multiple_testing_report(
    results: pd.DataFrame | Mapping[str, float] | pd.Series,
    *,
    alpha: float = 0.05,
    fdr_alpha: float = 0.10,
    t_column: str = "t_stat",
    p_column: str = "p_value",
    established: Sequence[str] = (),
    contributions: pd.DataFrame | None = None,
    n_boot: int = 2000,
    seed: int | None = 20260804,
) -> MultipleTestingReport:
    """Aplica todas las correcciones de multiplicidad a una familia de señales.

    Responde a "¿cuántas sobreviven?" con Bonferroni, Holm, Benjamini–Hochberg (al
    `fdr_alpha` del repo, 10 %), Benjamini–Yekutieli, el umbral `t > 3` de
    Harvey–Liu–Zhu y, si se pasan las series de contribución, Romano–Wolf.

    **Los p-valores que entran aquí deben venir ya corregidos por dependencia**
    (Newey–West o bootstrap por bloques). Meter en BH p-valores calculados con `t`
    ingenuos sobre retornos solapados es aplicar una corrección exquisita a números
    inflados ×4; es el error nº 11 de §15 y el orden correcto está en §7.4.

    Parameters
    ----------
    results:
        DataFrame indexado por nombre de señal con una columna de `t` o de p-valor,
        o un mapa `nombre -> t`. Si solo hay `t`, el p-valor se deriva con la
        aproximación normal bilateral.
    established:
        Señales con literatura previa robusta, a las que se aplica el umbral HLZ de
        `t > 2,0` en vez de `t > 3,0`.
    contributions:
        Panel `T × M` de contribuciones periódicas para Romano–Wolf. Sin él, el
        informe omite esa columna en vez de inventársela.
    """
    if isinstance(results, Mapping) and not isinstance(results, (pd.DataFrame, pd.Series)):
        results = pd.Series(results, dtype=float)
    if isinstance(results, pd.Series):
        frame = results.to_frame(name=t_column)
    else:
        frame = results.copy()
    if p_column not in frame.columns:
        if t_column not in frame.columns:
            msg = f"results debe tener la columna {t_column!r} o {p_column!r}"
            raise DataQualityError(msg)
        frame[p_column] = 2.0 * stats.norm.sf(np.abs(frame[t_column].to_numpy(dtype=float)))
    if t_column not in frame.columns:
        frame[t_column] = stats.norm.isf(frame[p_column].to_numpy(dtype=float) / 2.0)

    frame = frame.dropna(subset=[p_column])
    if frame.empty:
        msg = "no queda ninguna señal con p-valor tras descartar los nulos"
        raise InsufficientHistory(msg)
    names = tuple(str(i) for i in frame.index)
    p = frame[p_column].to_numpy(dtype=float)
    t = frame[t_column].to_numpy(dtype=float)
    m = p.size

    procedures = {
        "uncorrected": MultipleTestResult(
            method="uncorrected",
            alpha=alpha,
            p_values=p,
            adjusted=p,
            rejected=p <= alpha,
            names=names,
        ),
        "bonferroni": bonferroni(pd.Series(p, index=names), alpha),
        "holm": holm(pd.Series(p, index=names), alpha),
        "benjamini_hochberg": benjamini_hochberg(pd.Series(p, index=names), fdr_alpha),
        "benjamini_yekutieli": benjamini_yekutieli(pd.Series(p, index=names), fdr_alpha),
    }

    out = frame[[t_column, p_column]].copy()
    out.columns = ["t_stat", "p_value"]
    n_survivors: dict[str, int] = {}
    survivors: dict[str, tuple[str, ...]] = {}
    for key, res in procedures.items():
        out[f"p_adj_{key}"] = res.adjusted
        out[f"reject_{key}"] = res.rejected
        n_survivors[key] = res.n_rejected
        survivors[key] = res.survivors

    established_set = {str(s) for s in established}
    hlz_threshold = np.array(
        [
            harvey_liu_zhu_threshold("established" if n in established_set else "new")
            for n in names
        ]
    )
    out["hlz_threshold"] = hlz_threshold
    out["reject_hlz"] = np.abs(t) >= hlz_threshold
    n_survivors["harvey_liu_zhu"] = int(out["reject_hlz"].sum())
    survivors["harvey_liu_zhu"] = tuple(np.asarray(names)[out["reject_hlz"].to_numpy()])

    notes = (
        f"Umbral Bonferroni equivalente para M={m}: t={bonferroni_t(m, alpha):.3f}. "
        f"El t>3 de HLZ equivale a Bonferroni con solo {hlz_equivalent_tests():.1f} pruebas."
    )

    if contributions is not None:
        common = [c for c in contributions.columns if str(c) in set(names)]
        if common:
            rw = romano_wolf(
                contributions[common], alpha=alpha, n_boot=n_boot, seed=seed
            )
            rw_frame = rw.to_frame().reindex(out.index)
            out["p_adj_romano_wolf"] = rw_frame["p_adj_romano_wolf"]
            out["reject_romano_wolf"] = rw_frame["reject_romano_wolf"].fillna(False).astype(bool)
            n_survivors["romano_wolf"] = int(out["reject_romano_wolf"].sum())
            survivors["romano_wolf"] = rw.survivors
        else:  # pragma: no cover - configuración degenerada
            notes += " Romano-Wolf omitido: los nombres de `contributions` no casan."

    return MultipleTestingReport(
        table=out,
        n_tests=m,
        alpha=alpha,
        n_survivors=n_survivors,
        survivors=survivors,
        notes=notes,
    )


# =========================================================================== #
# 4. PBO por CSCV                                                              #
# =========================================================================== #


@dataclass(frozen=True, slots=True)
class PBOResult:
    """Probabilidad de sobreajuste de backtest y métricas complementarias."""

    pbo: float
    ci_low: float
    ci_high: float
    n_combinations: int
    n_configs: int
    n_partitions: int
    degradation_slope: float
    probability_of_loss: float
    degradation_ratio: float
    logits: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    sr_is: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    sr_oos: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    notes: str = ""

    @property
    def verdict(self) -> str:
        """Umbral operativo calibrado del repo: VIVA ≤ 0,35; CUARENTENA ≤ 0,50."""
        if self.pbo <= 0.35:
            return "VIVA"
        if self.pbo <= 0.50:
            return "CUARENTENA"
        return "MUERTA"

    def to_dict(self) -> dict[str, object]:
        return {
            "pbo": self.pbo,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n_combinations": self.n_combinations,
            "degradation_slope": self.degradation_slope,
            "probability_of_loss": self.probability_of_loss,
            "degradation_ratio": self.degradation_ratio,
            "verdict": self.verdict,
        }


def pbo_cscv(
    returns: pd.DataFrame | np.ndarray,
    *,
    n_partitions: int = 16,
    alpha: float = 0.10,
    seed: int | None = 20260804,
    n_boot: int = 500,
) -> PBOResult:
    """Probabilidad de sobreajuste de backtest por CSCV (Bailey et al., 2017).

    Algoritmo exacto:

    1. Matriz `M` de dimensión `T × N` con el retorno del periodo `t` de la
       configuración `n`. **Todas las columnas deben cubrir el mismo periodo.**
    2. Partir las `T` filas en `S` subconjuntos contiguos y disjuntos de igual
       tamaño (`S` par).
    3. Para cada una de las `C(S, S/2)` combinaciones: entrenamiento = los `S/2`
       subconjuntos elegidos; test = el complemento; `n*` = la configuración con
       mayor Sharpe in-sample; `ω̄ = rango_OOS(n*)/(N+1)`; `λ = ln[ω̄/(1−ω̄)]`.
    4. `PBO = P[λ ≤ 0]`.

    La **simetría** —evaluar todas las combinaciones, no una partición— es lo que
    hace insesgado el estimador. Usar solo "primera mitad / segunda mitad" da una
    única observación de `λ` y no estima nada.

    **Calibración medida** (§6.3): con paneles de estrategias sin habilidad el
    estimador converge a 0,494 ± 0,164, como debe. Pero **su desviación típica es
    ≈ 0,16**: un PBO de una sola ejecución no es un punto, es una estimación con una
    banda de ±0,3 al 90 %. Reportarlo con tres decimales y sin banda es engañoso.

    El barrido de habilidad muestra que el PBO solo baja de 0,35 con Sharpe
    verdadero > 1,3 y de 0,15 con Sharpe > 1,9. Por eso el umbral operativo del repo
    es **PBO ≤ 0,35**: exigir 0,10 descartaría cualquier alfa realista en acciones.

    Parameters
    ----------
    returns:
        Matriz `T × N` de retornos por configuración probada.
    n_partitions:
        `S`, par. López de Prado sugiere 16; con `S = 16` hay 12.870 combinaciones.
    """
    frame = returns if isinstance(returns, pd.DataFrame) else pd.DataFrame(returns)
    frame = frame.dropna(how="any")
    n_periods, n_configs = frame.shape
    if n_configs < 2:
        msg = f"el CSCV compara configuraciones entre sí; hay {n_configs}"
        raise InsufficientHistory(msg)
    if n_partitions % 2 != 0:
        msg = f"n_partitions debe ser par; recibido {n_partitions}"
        raise ValueError(msg)
    if n_periods < 4 * n_partitions:
        msg = (
            f"{n_periods} periodos para {n_partitions} particiones: cada bloque "
            "tendría menos de 4 observaciones y el Sharpe no significaría nada"
        )
        raise InsufficientHistory(msg)

    values = frame.to_numpy(dtype=float)
    usable = n_periods - (n_periods % n_partitions)
    blocks = values[:usable].reshape(n_partitions, usable // n_partitions, n_configs)
    block_n = float(blocks.shape[1])
    block_sum = blocks.sum(axis=1)  # (S, N)
    block_sumsq = (blocks**2).sum(axis=1)

    combos = list(combinations(range(n_partitions), n_partitions // 2))
    mask = np.zeros((len(combos), n_partitions), dtype=float)
    for i, c in enumerate(combos):
        mask[i, list(c)] = 1.0

    def sharpe(sum_: np.ndarray, sumsq: np.ndarray, count: float) -> np.ndarray:
        mean = sum_ / count
        var = (sumsq - count * mean**2) / (count - 1.0)
        sd = np.sqrt(np.clip(var, 0.0, None))
        return np.where(sd > 0, mean / np.where(sd > 0, sd, 1.0), 0.0)

    total_sum = block_sum.sum(axis=0)
    total_sumsq = block_sumsq.sum(axis=0)
    is_count = block_n * (n_partitions / 2)

    is_sum = mask @ block_sum
    is_sumsq = mask @ block_sumsq
    oos_sum = total_sum[None, :] - is_sum
    oos_sumsq = total_sumsq[None, :] - is_sumsq

    sr_is_all = sharpe(is_sum, is_sumsq, is_count)
    sr_oos_all = sharpe(oos_sum, oos_sumsq, is_count)

    best = np.argmax(sr_is_all, axis=1)
    rows = np.arange(len(combos))
    # Rango del ganador OOS: 1 = peor, N = mejor.
    ranks = np.argsort(np.argsort(sr_oos_all, axis=1, kind="stable"), axis=1) + 1
    rank_star = ranks[rows, best].astype(float)
    omega = rank_star / (n_configs + 1.0)
    logits = np.log(omega / (1.0 - omega))
    pbo = float(np.mean(logits <= 0.0))

    sr_is_star = sr_is_all[rows, best]
    sr_oos_star = sr_oos_all[rows, best]
    if np.std(sr_is_star) > 0:
        slope = float(np.polyfit(sr_is_star, sr_oos_star, 1)[0])
    else:  # pragma: no cover - degenerado
        slope = float("nan")
    prob_loss = float(np.mean(sr_oos_star < 0.0))
    med_is = float(np.median(sr_is_star))
    ratio = float(np.median(sr_oos_star) / med_is) if med_is != 0 else float("nan")

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(combos), size=(n_boot, len(combos)))
    boot_pbo = (logits[draws] <= 0.0).mean(axis=1)
    ci_low = float(np.percentile(boot_pbo, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_pbo, 100 * (1 - alpha / 2)))

    return PBOResult(
        pbo=pbo,
        ci_low=ci_low,
        ci_high=ci_high,
        n_combinations=len(combos),
        n_configs=int(n_configs),
        n_partitions=int(n_partitions),
        degradation_slope=slope,
        probability_of_loss=prob_loss,
        degradation_ratio=ratio,
        logits=logits,
        sr_is=sr_is_star,
        sr_oos=sr_oos_star,
        notes=(
            "El IC se obtiene remuestreando combinaciones, que NO son independientes "
            "(comparten bloques), por lo que subestima la incertidumbre real. La "
            "calibración Monte Carlo de §6.3 mide sd(PBO) ≈ 0,16 bajo ruido puro: "
            "úsala como banda de referencia."
        ),
    )


# =========================================================================== #
# 5. Reality Check de White y SPA de Hansen                                    #
# =========================================================================== #


@dataclass(frozen=True, slots=True)
class SPAResult:
    """Resultado del SPA de Hansen (o del Reality Check de White)."""

    statistic: float
    p_values: dict[str, float]
    best: str
    n_strategies: int
    n_obs: int
    n_boot: int
    block_length: float
    method: str

    @property
    def p_value(self) -> float:
        """P-valor de la variante consistente (o el único, en el Reality Check)."""
        return self.p_values.get("consistent", next(iter(self.p_values.values())))

    @property
    def robust(self) -> bool:
        """True si las tres variantes de `A_l` coinciden en rechazar al 5 %."""
        return len(self.p_values) == 3 and all(v <= 0.05 for v in self.p_values.values())


def _spa_inputs(
    differentials: pd.DataFrame, horizon: int, block_length: float | None, seed: int | None, n_boot: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    data = differentials.dropna(how="any")
    n, m = data.shape
    if n < 20 or m < 1:
        msg = f"SPA necesita al menos 20 periodos y 1 estrategia; hay {n} x {m}"
        raise InsufficientHistory(msg)
    values = data.to_numpy(dtype=float)
    means = values.mean(axis=0)
    lags = newey_west_lags(n, horizon)
    omega = np.array([math.sqrt(long_run_variance(values[:, j], lags)) for j in range(m)])
    omega = np.where(omega > 0, omega, np.nan)
    if np.all(~np.isfinite(omega)):
        msg = "todas las series de diferenciales son constantes"
        raise DataQualityError(msg)
    omega = np.nan_to_num(omega, nan=np.nanmax(omega))
    if block_length is None:
        block_length = float(max(math.ceil(politis_white_block_length(values.mean(axis=1))), horizon))
    idx = stationary_bootstrap_indices(n, block_length=block_length, n_boot=n_boot, seed=seed)
    boot_means = values[idx].mean(axis=1)
    return values, means, omega, boot_means, float(block_length)


def white_reality_check(
    differentials: pd.DataFrame,
    *,
    n_boot: int = 2000,
    block_length: float | None = None,
    seed: int | None = 20260804,
    horizon: int = 1,
) -> SPAResult:
    """Reality Check de White (2000): `H₀: max_l E[f_l] ≤ 0`.

    `f_{l,t}` es el diferencial de rendimiento de la estrategia `l` frente al
    benchmark. `V = max_l √T·f̄_l`, y la nula se impone **recentrando**
    `V*_b = max_l √T·(f̄*_l − f̄_l)`.

    **Debilidad conocida y determinante aquí:** el RC usa la configuración menos
    favorable (todos los `E[f_l] = 0`). Si la familia contiene muchas estrategias
    claramente malas, éstas no cambian `V` pero sí engordan la cola de `V*`, y el
    test pierde potencia dramáticamente. En este repo, donde se probarán decenas de
    variantes y muchas serán obviamente inferiores, **hay que usar `hansen_spa`**;
    el RC se calcula solo como referencia histórica.
    """
    data = differentials.dropna(how="any")
    n = len(data)
    _, means, _, boot_means, block = _spa_inputs(data, horizon, block_length, seed, n_boot)
    stat = float(np.max(math.sqrt(n) * means))
    boot_stat = np.max(math.sqrt(n) * (boot_means - means), axis=1)
    p = float(np.mean(boot_stat > stat))
    return SPAResult(
        statistic=stat,
        p_values={"white_rc": p},
        best=str(data.columns[int(np.argmax(means))]),
        n_strategies=data.shape[1],
        n_obs=n,
        n_boot=n_boot,
        block_length=block,
        method="white_reality_check",
    )


def hansen_spa(
    differentials: pd.DataFrame,
    *,
    n_boot: int = 2000,
    block_length: float | None = None,
    seed: int | None = 20260804,
    horizon: int = 1,
) -> SPAResult:
    """Superior Predictive Ability de Hansen (2005), con las **tres** variantes.

    Corrige las dos debilidades del Reality Check:

    1. **Studentización**: `T^SPA = max(0, max_l √T·f̄_l/ω̂_l)`, lo que reduce la
       influencia de las estrategias erráticas de alta varianza.
    2. **Distribución nula dependiente de la muestra**: en vez de recentrar todo a
       cero, recentra solo las que no son claramente inferiores,
       `g_l = f̄_l · 1{f̄_l ≥ −A_l}`.

    Variantes de `A_l`, todas reportadas porque **si las tres coinciden la
    conclusión es robusta**:

    - `"upper"`: `A_l = 0` (cota superior del p-valor, la más conservadora).
    - `"consistent"`: `A_l = (1/4)·T^(−1/4)·ω̂_l`.
    - `"lower"`: `A_l = ω̂_l·√(2·ln ln T / T)` (la más liberal).

    El criterio 15 de §14.1 exige `p ≤ 0,05` en las tres variantes para promocionar
    un factor a VIVA contra el benchmark FF5+MOM.
    """
    data = differentials.dropna(how="any")
    n = len(data)
    _, means, omega, boot_means, block = _spa_inputs(data, horizon, block_length, seed, n_boot)
    root_t = math.sqrt(n)
    stat = float(max(0.0, np.max(root_t * means / omega)))

    thresholds = {
        "upper": np.zeros_like(omega),
        "consistent": 0.25 * n ** (-0.25) * omega,
        "lower": omega * math.sqrt(2.0 * math.log(math.log(n)) / n) if n > 3 else np.zeros_like(omega),
    }
    p_values: dict[str, float] = {}
    for name, a_l in thresholds.items():
        g = np.where(means >= -a_l, means, 0.0)
        z_boot = root_t * (boot_means - g) / omega
        t_boot = np.maximum(0.0, z_boot.max(axis=1))
        p_values[name] = float(np.mean(t_boot > stat))

    return SPAResult(
        statistic=stat,
        p_values=p_values,
        best=str(data.columns[int(np.argmax(means / omega))]),
        n_strategies=data.shape[1],
        n_obs=n,
        n_boot=n_boot,
        block_length=block,
        method="hansen_spa",
    )
