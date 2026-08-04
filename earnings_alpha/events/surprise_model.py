"""SurpriseModel: modelo predictivo de sorpresa de EPS y de retorno de evento (contrato §3.5).

Qué se predice y por qué son DOS tareas separadas
-------------------------------------------------
1. **Sorpresa de EPS** (signo y magnitud estandarizada, SUE de Foster, Olsen y
   Shevlin 1984): ¿el resultado batirá o fallará el consenso, y por cuánto?
2. **Retorno del evento** (retorno anormal en la ventana ``[T, T+1]``): ¿qué hará
   el precio cuando la noticia se publique?

No son la misma pregunta, y confundirlas es el error económico central de esta
literatura: **lo que mueve el precio no es la sorpresa, sino la sorpresa respecto
a lo ya descontado**. El consenso final está sesgado a la baja de forma
sistemática (*walk-down* de Richardson, Teoh y Wysocki 2004): ~64 % de los
anuncios "baten", el mercado lo sabe, y un modelo que "predice" ese beat modal
tiene una exactitud aparente del 64 % y un alfa de cero. Por eso el módulo:

- entrena las dos tareas con la **misma interfaz** pero como modelos separados;
- compara cada una contra sus líneas base honestas: la **clase mayoritaria**
  (para el signo, que con el walk-down ya acierta ~64 %) y el **consenso de
  analistas** (sorpresa esperada = 0; retorno de evento esperado = 0), de modo
  que batir a la base exige información *incremental* sobre lo descontado;
- reporta el rank-IC de Spearman además del R² OOS: una constante puede ganar al
  consenso-cero solo por la prima de anuncio media (Frazzini y Lamont 2007),
  pero no puede tener IC.

Por qué el K-fold aleatorio está PROHIBIDO aquí
-----------------------------------------------
La `KFold` barajada de scikit-learn asume observaciones intercambiables. En este
panel es falso por construcción (`validation_methodology.md` §8.1-§8.2):

- Las features de `PreEventFeatures` miran hasta 250 sesiones atrás (la ventana
  de estimación del modelo de mercado) y la etiqueta hasta ``f`` sesiones
  adelante: el **span de información** de un evento en ``T`` es ``[T-250, T+f]``
  — 310 sesiones con la etiqueta ``CAR[0,+60]`` del informe, un orden de
  magnitud más que el horizonte "ingenuo".
- Los eventos se agrupan en temporadas de resultados: ~40 % de los anuncios caen
  en tres semanas por trimestre y comparten shocks de mercado y sector. Un fold
  aleatorio pone en entrenamiento eventos cuyo span solapa el test: el modelo
  **ve la respuesta** y el OOS resultante es una ficción optimista.

Por eso `evaluate` usa exclusivamente `stats.validation.PurgedKFold` (purga por
solape de spans + embargo asimétrico hacia adelante, López de Prado 2018) sobre
bloques temporales contiguos, y `fit` exige `groups_by_date` en la firma del
contrato: sin fechas no hay validación legal en este repo.

Piezas
------
- `event_labels`: construye las dos etiquetas por `event_id` (sorpresa
  estandarizada y retorno anormal ajustado por mercado, Brown y Warner 1985).
- `SurpriseModel`: regresión regularizada interpretable (ridge / logística L2
  con winsorización e imputación ajustadas SOLO en entrenamiento) y gradient
  boosting (`HistGradientBoosting*`, Friedman 2001), misma interfaz
  ``fit/predict``. Para el signo, `predict` devuelve una **probabilidad
  continua**, nunca una alerta binaria — la aritmética de tasa base del informe
  `informed_trading.md` §12.4 (PPV ≈ 0,14 con prevalencia del 2 %) hace
  inaceptable cualquier salida binaria.
- `SurpriseModel.evaluate`: CV purgada con embargo, métricas OOS con líneas
  base, **permutation importance estrictamente fuera de muestra** (Breiman
  2001) y curva de calibración con ECE.
- `calibration_curve` / `expected_calibration_error`: fiabilidad de las
  probabilidades; el boosting descalibra de forma documentada (Niculescu-Mizil
  y Caruana 2005) y el modelo ofrece recalibración de Platt (1999) o isotónica
  (Zadrozny y Elkan 2002) ajustada en una cola temporal embargada del
  entrenamiento, jamás en el test.
- `compare_estimators`: los dos estimadores sobre los mismos folds purgados.

Nota legal y metodológica (obligatoria por el contrato §3.5)
------------------------------------------------------------
Todas las features de entrada derivan de datos **públicos** (precios, volumen,
cadenas de opciones, short interest agregado de FINRA, Form 4 ya presentados).
El objetivo es explotar la huella estadística que el comportamiento de otros
participantes deja en variables observables, no acceder a información material
no pública. Ver `earnings_alpha.events.preevent`.

Referencias
-----------
- Ball, R., y Brown, P. (1968). *An Empirical Evaluation of Accounting Income
  Numbers*. **JAR** 6(2), 159-178. (El retorno del evento responde a la sorpresa.)
- Bernard, V. L., y Thomas, J. K. (1989). *Post-Earnings-Announcement Drift:
  Delayed Price Response or Risk Premium?* **JAR** 27, 1-36.
- Foster, G., Olsen, C., y Shevlin, T. (1984). *Earnings Releases, Anomalies,
  and the Behavior of Security Returns*. **The Accounting Review** 59(4). (SUE.)
- Richardson, S., Teoh, S. H., y Wysocki, P. (2004). *The Walk-down to Beatable
  Analyst Forecasts*. **Contemporary Accounting Research** 21(4).
- Brown, S. J., y Warner, J. B. (1985). *Using Daily Stock Returns: The Case of
  Event Studies*. **JFE** 14(1). (El ajuste por mercado basta a 1-2 días.)
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
  (CV purgada con embargo; por qué el K-fold aleatorio miente.)
- Campbell, J. Y., y Thompson, S. B. (2008). *Predicting Excess Stock Returns
  Out of Sample*. **RFS** 21(4). (R² fuera de muestra contra la media histórica.)
- Hoerl, A. E., y Kennard, R. W. (1970). *Ridge Regression*. **Technometrics**
  12(1). Friedman, J. H. (2001). *Greedy Function Approximation: A Gradient
  Boosting Machine*. **Annals of Statistics** 29(5).
- Breiman, L. (2001). *Random Forests*. **Machine Learning** 45. (Permutation
  importance; aquí se calcula sobre el test de cada fold, nunca in-sample.)
- Platt, J. (1999); Zadrozny, B., y Elkan, C. (2002); Niculescu-Mizil, A., y
  Caruana, R. (2005). (Calibración de probabilidades.)
- Hanley, J. A., y McNeil, B. J. (1982). *The Meaning and Use of the Area under
  a ROC Curve*. **Radiology** 143(1). (Error estándar del AUC.)
- Saito, T., y Rehmsmeier, M. (2015). *The Precision-Recall Plot Is More
  Informative than the ROC Plot on Imbalanced Datasets*. **PLoS ONE** 10(3).
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Final, Literal, Self

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from earnings_alpha.errors import ConfigError, DataQualityError, InsufficientHistory
from earnings_alpha.events import flow
from earnings_alpha.events.preevent import METADATA_COLUMNS, _market_returns
from earnings_alpha.stats.validation import PurgedKFold, information_spans

try:  # pragma: no cover - el entorno de tests siempre trae scikit-learn
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
    )
    from sklearn.impute import SimpleImputer
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import average_precision_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:  # pragma: no cover
    msg = (
        "earnings_alpha.events.surprise_model necesita scikit-learn; "
        "instálalo con `pip install 'earnings-alpha[stats]'`"
    )
    raise ImportError(msg) from exc

__all__ = [
    "DEFAULT_FEATURE_LOOKBACK",
    "DEFAULT_LABEL_LOOKAHEAD",
    "MIN_CALIBRATION_EVENTS",
    "MIN_CLASS_COUNT",
    "MIN_FIT_EVENTS",
    "SurpriseModel",
    "SurpriseModelReport",
    "brier_score",
    "calibration_curve",
    "compare_estimators",
    "event_labels",
    "expected_calibration_error",
    "feature_matrix",
    "roc_auc",
    "spearman_ic",
]


MIN_FIT_EVENTS: Final[int] = 30
"""Mínimo de eventos para ajustar un modelo; menos es curve-fitting, no estimación."""

MIN_CLASS_COUNT: Final[int] = 5
"""Mínimo de observaciones de la clase minoritaria en clasificación."""

MIN_CALIBRATION_EVENTS: Final[int] = 20
"""Mínimo de eventos en la cola temporal reservada para el calibrador."""

DEFAULT_FEATURE_LOOKBACK: Final[int] = 250
"""Lookback del span de información: la ventana de estimación del modelo de
mercado de `PreEventFeatures` llega a T-250 (`validation_methodology.md` §8.2)."""

DEFAULT_LABEL_LOOKAHEAD: Final[int] = 10
"""Lookahead por defecto del span: cubre la etiqueta ``CAR[0,+1]`` más un margen
por anuncios AMC. Con etiquetas de horizonte largo (``CAR[0,+60]``) debe subirse
a 60, reproduciendo el span de 310 sesiones del informe."""

_EPS_PROB: Final[float] = 1e-7


# ---------------------------------------------------------------------------
# Matriz de features y etiquetas
# ---------------------------------------------------------------------------


def feature_matrix(
    features: pd.DataFrame, *, exclude: Sequence[str] = METADATA_COLUMNS
) -> pd.DataFrame:
    """Extrae la matriz numérica de features de la tabla de `PreEventFeatures`.

    Excluye los metadatos (`METADATA_COLUMNS`) y cualquier columna no numérica,
    y elimina las columnas íntegramente NaN (fuentes ausentes degradadas), que
    quedan registradas en ``result.attrs['dropped_all_nan']`` — se documenta la
    degradación, no se imputa una constante inventada sobre una fuente que no
    existe.
    """
    if not isinstance(features, pd.DataFrame) or len(features) == 0:
        msg = "`features` debe ser un DataFrame no vacío"
        raise DataQualityError(msg)
    if not features.index.is_unique:
        msg = "`features` tiene event_id duplicados en el índice"
        raise DataQualityError(msg)
    banned = set(exclude)
    cols = [
        c
        for c in features.columns
        if c not in banned and pd.api.types.is_numeric_dtype(features[c])
    ]
    if not cols:
        msg = "no queda ninguna columna numérica de features tras excluir metadatos"
        raise DataQualityError(msg)
    out = features[cols].astype(float)
    all_nan = [c for c in out.columns if not np.isfinite(out[c].to_numpy()).any()]
    out = out.drop(columns=all_nan)
    if out.shape[1] == 0:
        msg = "todas las features son NaN: ninguna fuente produjo datos"
        raise DataQualityError(msg)
    out.attrs["dropped_all_nan"] = list(all_nan)
    return out


def event_labels(
    events: pd.DataFrame,
    prices: pd.DataFrame | None = None,
    *,
    market: pd.DataFrame | pd.Series | None = None,
    surprise_col: str = "sue",
    return_window: tuple[int, int] = (0, 1),
) -> pd.DataFrame:
    """Construye las etiquetas de las dos tareas del contrato §3.5, por `event_id`.

    Columnas devueltas:

    - ``surprise``: la sorpresa estandarizada (`surprise_col`, por defecto el SUE
      de Foster, Olsen y Shevlin 1984 que publica el generador y `factors`).
    - ``surprise_sign``: signo de la sorpresa bruta (``eps_surprise`` si existe;
      si no, el signo de `surprise_col`). Cero = *meet* exacto.
    - ``event_return`` (si se pasa `prices`): retorno anormal acumulado en la
      ventana de sesiones ``[T+w0, T+w1]`` con ``T = event_date`` (la fecha
      *negociable* de `pit.tradable_date`, que para un AMC ya es la sesión
      siguiente e incluye el gap de apertura). Ajuste por mercado simple
      ``r_i - r_m``: para ventanas de 1-2 días rinde igual que el modelo de
      mercado completo (Brown y Warner 1985) y no necesita ventana de estimación.

    Point-in-time: la ventana del retorno debe empezar en ``w0 >= 0``. Una
    etiqueta que empezara antes de T solaparía la ventana de las features y el
    "modelo" se predeciría a sí mismo; se rechaza con `DataQualityError`.

    Los eventos sin cobertura de precios completa en la ventana salen NaN. Si
    ninguna etiqueta es calculable se lanza `InsufficientHistory`.
    """
    ev = flow._normalize_events(events)
    if surprise_col not in ev.columns:
        candidates = [
            c
            for c in ("sue", "surprise_z", "eps_surprise", "surprise", "surprise_pct")
            if c in ev.columns
        ]
        msg = (
            f"`events` no tiene la columna de sorpresa {surprise_col!r}; "
            f"candidatas presentes: {candidates}"
        )
        raise DataQualityError(msg)
    surprise = ev[surprise_col].astype(float)
    sign_col = "eps_surprise" if "eps_surprise" in ev.columns else surprise_col
    sign = np.sign(ev[sign_col].astype(float).to_numpy())

    out = pd.DataFrame(
        {
            "ticker": ev["ticker"].to_numpy(),
            "event_date": ev["event_date"].to_numpy(),
            "surprise": surprise.to_numpy(),
            "surprise_sign": sign,
        },
        index=pd.Index(ev["event_id"], name="event_id"),
    )

    if prices is not None:
        w0, w1 = int(return_window[0]), int(return_window[1])
        if not 0 <= w0 <= w1:
            msg = (
                f"return_window inválida {return_window}: se exige 0 <= w0 <= w1. "
                "Una etiqueta que empieza antes de T solapa la ventana de features "
                "y convierte el modelo en una tautología"
            )
            raise DataQualityError(msg)
        px = flow._check_prices(prices, ["close"])
        returns = flow._returns_wide(px)
        dates = pd.DatetimeIndex(returns.index)
        adjusted = returns
        if market is not None:
            r_m = _market_returns(market).reindex(dates)
            adjusted = returns.sub(r_m, axis=0)
        pos = flow._positions(ev["event_date"], dates)
        r_cols = {t: adjusted[t].to_numpy() for t in adjusted.columns}
        vals = np.full(len(ev), np.nan)
        width = w1 - w0 + 1
        for i, row in enumerate(ev.itertuples(index=False)):
            p = int(pos[i])
            r = r_cols.get(row.ticker)
            if p < 0 or r is None or p + w1 >= len(dates):
                continue
            seg = r[p + w0 : p + w1 + 1]
            if len(seg) == width and np.isfinite(seg).all():
                vals[i] = float(np.sum(seg))
        out["event_return"] = vals

    label_cols = [c for c in ("surprise", "event_return") if c in out.columns]
    flow._raise_if_all_nan(out[label_cols], "event_labels")
    return out


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------


def roc_auc(score: Sequence[float] | np.ndarray | pd.Series,
            labels: Sequence[bool] | np.ndarray | pd.Series) -> tuple[float, float]:
    """AUC-ROC por Mann-Whitney y su error estándar bajo H0 (Hanley-McNeil 1982).

    ``AUC = (Σ rangos_positivos - n1(n1+1)/2) / (n1·n0)`` y, bajo la nula de no
    discriminación, ``se = sqrt((n1+n0+1)/(12·n1·n0))``: es el error estándar
    correcto para contrastar ``AUC = 0,5``, no para un IC alrededor de un AUC
    alto. Las puntuaciones no finitas se descartan.
    """
    s = np.asarray(score, dtype=float)
    y = np.asarray(labels, dtype=bool)
    if s.shape != y.shape:
        msg = f"score y labels con formas distintas: {s.shape} vs {y.shape}"
        raise DataQualityError(msg)
    finite = np.isfinite(s)
    s, y = s[finite], y[finite]
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 < 1 or n0 < 1:
        msg = f"AUC indefinido: {n1} positivos y {n0} negativos"
        raise InsufficientHistory(msg)
    ranks = sp_stats.rankdata(s)
    auc = float((ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))
    se = float(np.sqrt((n1 + n0 + 1) / (12.0 * n1 * n0)))
    return auc, se


def spearman_ic(pred: Sequence[float] | np.ndarray | pd.Series,
                actual: Sequence[float] | np.ndarray | pd.Series) -> tuple[float, float]:
    """Rank-IC de Spearman entre predicción y realización, con su p-valor.

    Es la métrica que una constante no puede tener: separa habilidad direccional
    de la mera captura de la media (prima de anuncio). El p-valor es el nominal
    de `scipy.stats.spearmanr`; para promoción final debe pasar por la
    maquinaria de dependencia y multiplicidad de `earnings_alpha.stats`.
    """
    p = np.asarray(pred, dtype=float)
    a = np.asarray(actual, dtype=float)
    if p.shape != a.shape:
        msg = f"pred y actual con formas distintas: {p.shape} vs {a.shape}"
        raise DataQualityError(msg)
    mask = np.isfinite(p) & np.isfinite(a)
    if int(mask.sum()) < 8:
        msg = f"solo {int(mask.sum())} pares finitos: IC no estimable"
        raise InsufficientHistory(msg)
    if np.ptp(p[mask]) == 0.0 or np.ptp(a[mask]) == 0.0:
        msg = "entrada constante: el rank-IC no está definido"
        raise DataQualityError(msg)
    res = sp_stats.spearmanr(p[mask], a[mask])
    return float(res.statistic), float(res.pvalue)


def brier_score(y_true: Sequence[float] | np.ndarray | pd.Series,
                prob: Sequence[float] | np.ndarray | pd.Series) -> float:
    """Puntuación de Brier: error cuadrático medio de la probabilidad (Brier 1950)."""
    y = _binary_labels(y_true)
    p = _valid_probs(prob, len(y))
    return float(np.mean((p - y) ** 2))


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    """Log-loss con recorte numérico de la probabilidad."""
    q = np.clip(p, _EPS_PROB, 1.0 - _EPS_PROB)
    return float(-np.mean(y * np.log(q) + (1.0 - y) * np.log(1.0 - q)))


def _binary_labels(y_true: Sequence[float] | np.ndarray | pd.Series) -> np.ndarray:
    """Valida etiquetas binarias {0,1} (acepta bool); rechaza cualquier otra cosa."""
    y = np.asarray(y_true)
    if y.dtype == bool:
        return y.astype(float)
    y = y.astype(float)
    uniq = set(np.unique(y[np.isfinite(y)]).tolist())
    if not uniq or not uniq <= {0.0, 1.0}:
        msg = f"y_true debe ser binaria en {{0,1}}; valores vistos: {sorted(uniq)[:6]}"
        raise DataQualityError(msg)
    return y


def _valid_probs(prob: Sequence[float] | np.ndarray | pd.Series, n: int) -> np.ndarray:
    p = np.asarray(prob, dtype=float)
    if p.shape[0] != n:
        msg = f"prob tiene {p.shape[0]} filas y y_true {n}"
        raise DataQualityError(msg)
    if not np.isfinite(p).all():
        msg = "prob contiene valores no finitos"
        raise DataQualityError(msg)
    if p.min() < -1e-9 or p.max() > 1.0 + 1e-9:
        msg = f"prob fuera de [0,1]: rango [{p.min():.4f}, {p.max():.4f}]"
        raise DataQualityError(msg)
    return np.clip(p, 0.0, 1.0)


def calibration_curve(
    y_true: Sequence[float] | np.ndarray | pd.Series,
    prob: Sequence[float] | np.ndarray | pd.Series,
    *,
    n_bins: int = 10,
    strategy: Literal["quantile", "uniform"] = "quantile",
) -> pd.DataFrame:
    """Curva de fiabilidad: probabilidad predicha media frente a frecuencia observada.

    Por bin devuelve ``p_mean`` (probabilidad media predicha), ``y_rate``
    (frecuencia observada de positivos) y ``count``. Un modelo calibrado tiene
    ``y_rate ≈ p_mean`` en cada bin. En ``attrs`` viajan ``ece`` (Expected
    Calibration Error, media de ``|p_mean - y_rate|`` ponderada por ``count``),
    ``brier`` y ``n``.

    ``strategy="quantile"`` (por defecto) reparte las observaciones en bins de
    igual población — con probabilidades concentradas cerca de la prevalencia,
    los bins uniformes dejarían la mayoría vacíos y la curva no mediría nada.
    """
    y = _binary_labels(y_true)
    p = _valid_probs(prob, len(y))
    if n_bins < 2:
        msg = f"n_bins debe ser >= 2; recibido {n_bins}"
        raise ConfigError(msg)
    if len(y) < 2 * n_bins:
        msg = f"{len(y)} observaciones para {n_bins} bins: usa menos bins"
        raise InsufficientHistory(msg)
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1)))
    elif strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:  # pragma: no cover - protegido por el Literal
        msg = f"strategy desconocida: {strategy!r}"
        raise ConfigError(msg)
    if len(edges) < 3:
        # Probabilidades casi constantes: un único bin sigue siendo informativo.
        edges = np.array([p.min() - 1e-9, p.max() + 1e-9])
    bin_id = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)

    rows = []
    for b in range(len(edges) - 1):
        mask = bin_id == b
        n_b = int(mask.sum())
        if n_b == 0:
            continue
        rows.append(
            {
                "bin": b,
                "p_mean": float(p[mask].mean()),
                "y_rate": float(y[mask].mean()),
                "count": n_b,
            }
        )
    curve = pd.DataFrame(rows).set_index("bin")
    weights = curve["count"] / curve["count"].sum()
    curve.attrs["ece"] = float((weights * (curve["p_mean"] - curve["y_rate"]).abs()).sum())
    curve.attrs["brier"] = float(np.mean((p - y) ** 2))
    curve.attrs["n"] = int(len(y))
    curve.attrs["strategy"] = strategy
    return curve


def expected_calibration_error(
    y_true: Sequence[float] | np.ndarray | pd.Series,
    prob: Sequence[float] | np.ndarray | pd.Series,
    *,
    n_bins: int = 10,
    strategy: Literal["quantile", "uniform"] = "quantile",
) -> float:
    """ECE: media ponderada de ``|p_mean - y_rate|`` sobre los bins de la curva."""
    return float(calibration_curve(y_true, prob, n_bins=n_bins, strategy=strategy).attrs["ece"])


# ---------------------------------------------------------------------------
# Preprocesado
# ---------------------------------------------------------------------------


class _Winsorizer(BaseEstimator, TransformerMixin):
    """Winsorización por columna con cuantiles ajustados SOLO en entrenamiento.

    Las features de flujo tienen colas pesadas (Amihud, deltas de short
    interest); sin recorte, unas pocas observaciones extremas dominan la
    regresión lineal. Los cuantiles se estiman en `fit` y se congelan: aplicar
    cuantiles del conjunto completo filtraría información del test al train. El
    gradient boosting es invariante a transformaciones monótonas y no lo usa.
    """

    def __init__(self, quantile: float = 0.01) -> None:
        self.quantile = quantile

    def fit(self, X: Any, y: Any = None) -> _Winsorizer:  # noqa: N803 - API sklearn
        arr = np.asarray(X, dtype=float)
        q = float(self.quantile)
        if not 0.0 <= q < 0.5:
            msg = f"quantile debe estar en [0, 0.5); recibido {q}"
            raise ConfigError(msg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # columnas todo-NaN
            lo = np.nanquantile(arr, q, axis=0)
            hi = np.nanquantile(arr, 1.0 - q, axis=0)
        self.lower_ = np.where(np.isfinite(lo), lo, -np.inf)
        self.upper_ = np.where(np.isfinite(hi), hi, np.inf)
        return self

    def transform(self, X: Any) -> np.ndarray:  # noqa: N803 - API sklearn
        arr = np.asarray(X, dtype=float)
        return np.clip(arr, self.lower_, self.upper_)


def _logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, _EPS_PROB, 1.0 - _EPS_PROB)
    return np.log(q / (1.0 - q))


# ---------------------------------------------------------------------------
# Informe de evaluación
# ---------------------------------------------------------------------------


@dataclass
class SurpriseModelReport:
    """Resultado OOS de la CV purgada de un `SurpriseModel`.

    `metrics` y `baselines` comparten claves comparables (p. ej. ``brier`` del
    modelo frente a ``brier`` de la constante de prevalencia por fold), de modo
    que la comparación contra la línea base es un zip, no una interpretación.
    """

    target: str
    objective: str
    estimator: str
    metrics: dict[str, float]
    baselines: dict[str, float]
    fold_metrics: pd.DataFrame = field(repr=False)
    oos_prediction: pd.Series = field(repr=False)
    oos_target: pd.Series = field(repr=False)
    importance: pd.DataFrame = field(repr=False)
    calibration: pd.DataFrame | None = field(repr=False)
    purge: pd.DataFrame = field(repr=False)
    dropped_features: tuple[str, ...] = ()
    n_splits: int = 0

    @property
    def beats_baseline(self) -> bool:
        """Veredicto de *cribado* (no de promoción) contra la línea base honesta.

        - Signo: AUC significativamente > 0,5 (t de Hanley-McNeil > 2) **y**
          Brier no peor que la constante de prevalencia entrenada por fold. La
          exactitud no entra: con un 64 % de beats, la clase mayoritaria ya
          "acierta" el 64 % y la exactitud premia imitarla.
        - Magnitud: R² OOS contra la media de entrenamiento (Campbell-Thompson
          2008) > 0 **y** rank-IC positivo con p nominal < 0,05.

        La promoción final exige además la maquinaria de §3.8: dependencia,
        multiplicidad y deflación del Sharpe. Esto solo dice "merece seguir".
        """
        if self.objective == "sign":
            return (
                self.metrics.get("auc_t", 0.0) > 2.0
                and self.metrics.get("brier", np.inf) <= self.baselines.get("brier", 0.0)
            )
        return (
            self.metrics.get("r2_oos", -np.inf) > 0.0
            and self.metrics.get("spearman_ic", 0.0) > 0.0
            and self.metrics.get("spearman_p", 1.0) < 0.05
        )

    def summary(self) -> pd.DataFrame:
        """Tabla métrica -> (modelo, línea base) para lectura rápida."""
        keys = sorted(set(self.metrics) | set(self.baselines))
        return pd.DataFrame(
            {
                "model": [self.metrics.get(k, np.nan) for k in keys],
                "baseline": [self.baselines.get(k, np.nan) for k in keys],
            },
            index=pd.Index(keys, name="metric"),
        )


# ---------------------------------------------------------------------------
# El modelo
# ---------------------------------------------------------------------------


@dataclass
class SurpriseModel:
    """Predice signo/magnitud de la sorpresa o el retorno del evento (contrato §3.5).

    Parameters
    ----------
    target:
        Qué etiqueta se modela: ``"surprise"`` (SUE) o ``"event_return"``
        (retorno anormal del evento). Es metadato de reporting: el dato real
        entra por ``y``. Las dos tareas se entrenan como modelos separados
        porque **lo que se negocia es la sorpresa respecto a lo descontado**:
        predecir el beat modal del walk-down no es predecir el retorno.
    objective:
        ``"sign"`` = clasificación binaria (``y > 0``); `predict` devuelve la
        **probabilidad continua** del signo positivo (nunca una alerta binaria;
        aritmética de tasa base del informe §12.4). ``"magnitude"`` = regresión.
    estimator:
        ``"ridge"``: pipeline interpretable winsorización -> imputación mediana
        -> estandarización -> Ridge (Hoerl-Kennard 1970) o logística L2, con
        todos los parámetros ajustados SOLO en entrenamiento; los coeficientes
        por desviación típica salen de `coefficients()`. ``"gbm"``:
        `HistGradientBoosting*` (Friedman 2001), que consume NaN nativamente y
        captura no-linealidades; su interpretación es la permutation importance
        OOS del informe de `evaluate`. Misma interfaz en ambos casos.
    calibration:
        ``"none"`` | ``"sigmoid"`` (Platt 1999) | ``"isotonic"`` (Zadrozny-Elkan
        2002), solo para ``objective="sign"``. El calibrador se ajusta sobre la
        **cola temporal final** del entrenamiento (fracción
        `calibration_fraction`), separada del resto por un embargo de
        `calibration_embargo_days` días naturales: calibrar in-sample produce
        curvas de fiabilidad decorativas.
    seed:
        Determinismo del boosting y de las permutaciones (regla de oro nº 4).

    Validación
    ----------
    `fit(X, y, groups_by_date)` exige las fechas de anuncio (contrato §3.5) y
    `evaluate` solo valida con `stats.validation.PurgedKFold`. **El K-fold
    aleatorio está prohibido** en este módulo: con features que miran 250
    sesiones atrás, etiquetas hacia adelante y eventos apiñados por temporada,
    un fold barajado mete en el entrenamiento observaciones cuyo span de
    información solapa el test — el modelo ve la respuesta y el OOS es ficción
    (López de Prado 2018; `validation_methodology.md` §8.1-§8.2). No existe
    ningún parámetro `shuffle` aquí, deliberadamente.
    """

    target: Literal["surprise", "event_return"] = "surprise"
    objective: Literal["sign", "magnitude"] = "sign"
    estimator: Literal["ridge", "gbm"] = "ridge"
    alpha: float = 1.0
    winsor_quantile: float = 0.01
    learning_rate: float = 0.05
    max_iter: int = 250
    max_depth: int | None = 3
    min_samples_leaf: int = 25
    l2_regularization: float = 1.0
    calibration: Literal["none", "sigmoid", "isotonic"] = "none"
    calibration_fraction: float = 0.25
    calibration_embargo_days: int = 10
    seed: int = 20260804

    # --- estado ajustado (no forma parte de la configuración)
    pipeline_: Any = field(default=None, init=False, repr=False, compare=False)
    calibrator_: Any = field(default=None, init=False, repr=False, compare=False)
    feature_names_: tuple[str, ...] = field(default=(), init=False, repr=False, compare=False)
    dropped_features_: tuple[str, ...] = field(default=(), init=False, repr=False, compare=False)
    train_prevalence_: float = field(default=float("nan"), init=False, repr=False, compare=False)
    train_mean_: float = field(default=float("nan"), init=False, repr=False, compare=False)
    n_train_: int = field(default=0, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.target not in ("surprise", "event_return"):
            msg = f"target debe ser 'surprise' o 'event_return'; recibido {self.target!r}"
            raise ConfigError(msg)
        if self.objective not in ("sign", "magnitude"):
            msg = f"objective debe ser 'sign' o 'magnitude'; recibido {self.objective!r}"
            raise ConfigError(msg)
        if self.estimator not in ("ridge", "gbm"):
            msg = f"estimator debe ser 'ridge' o 'gbm'; recibido {self.estimator!r}"
            raise ConfigError(msg)
        if self.calibration not in ("none", "sigmoid", "isotonic"):
            msg = f"calibration debe ser 'none', 'sigmoid' o 'isotonic'; recibido {self.calibration!r}"
            raise ConfigError(msg)
        if self.calibration != "none" and self.objective != "sign":
            msg = "la calibración de probabilidades solo aplica a objective='sign'"
            raise ConfigError(msg)
        if self.alpha <= 0:
            msg = f"alpha debe ser > 0; recibido {self.alpha}"
            raise ConfigError(msg)
        if not 0.0 < self.calibration_fraction <= 0.5:
            msg = f"calibration_fraction debe estar en (0, 0.5]; recibido {self.calibration_fraction}"
            raise ConfigError(msg)

    # ------------------------------------------------------------------ básicos

    @property
    def is_classifier(self) -> bool:
        """True si la tarea es el signo (clasificación probabilística)."""
        return self.objective == "sign"

    @property
    def is_fitted(self) -> bool:
        return self.pipeline_ is not None

    def _clone(self) -> SurpriseModel:
        """Copia SIN estado ajustado: cada fold de la CV entrena desde cero."""
        return replace(self)

    def _build_pipeline(self) -> Pipeline:
        if self.estimator == "ridge":
            model: Any = (
                LogisticRegression(C=1.0 / self.alpha, solver="lbfgs", max_iter=5000)
                if self.is_classifier
                else Ridge(alpha=self.alpha)
            )
            return Pipeline(
                [
                    ("winsor", _Winsorizer(self.winsor_quantile)),
                    ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                    ("scale", StandardScaler()),
                    ("model", model),
                ]
            )
        kwargs: dict[str, Any] = {
            "learning_rate": self.learning_rate,
            "max_iter": self.max_iter,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "l2_regularization": self.l2_regularization,
            "random_state": self.seed,
            "early_stopping": False,
        }
        model = (
            HistGradientBoostingClassifier(**kwargs)
            if self.is_classifier
            else HistGradientBoostingRegressor(**kwargs)
        )
        return Pipeline([("model", model)])

    # -------------------------------------------------------------- validación

    def _prepare(
        self,
        X: pd.DataFrame,  # noqa: N803 - convención sklearn del contrato
        y: pd.Series | Sequence[float] | np.ndarray,
        dates: pd.Series | Sequence[Any] | np.ndarray | None,
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Valida y alinea features, etiqueta y fechas; falla ruidosamente."""
        x_mat = feature_matrix(X)

        if isinstance(y, pd.Series):
            missing = x_mat.index.difference(y.index)
            if len(missing) > 0:
                msg = (
                    f"y no cubre {len(missing)} event_id de X "
                    f"(muestra: {list(missing[:3])})"
                )
                raise DataQualityError(msg)
            y_ser = y.reindex(x_mat.index).astype(float)
        else:
            arr = np.asarray(y, dtype=float)
            if arr.ndim != 1 or arr.shape[0] != len(x_mat):
                msg = f"y tiene forma {arr.shape} y X {len(x_mat)} filas"
                raise DataQualityError(msg)
            y_ser = pd.Series(arr, index=x_mat.index)
        n_bad = int((~np.isfinite(y_ser.to_numpy())).sum())
        if n_bad:
            msg = (
                f"y contiene {n_bad} valores no finitos; una etiqueta ausente no es "
                "un cero — elimina esas filas antes (p. ej. labels.dropna())"
            )
            raise DataQualityError(msg)

        if dates is None:
            msg = (
                "faltan las fechas de anuncio (`groups_by_date`). Sin fechas no hay "
                "validación temporal purgada, y el K-fold aleatorio está prohibido "
                "en este módulo (validation_methodology.md §8.1)"
            )
            raise DataQualityError(msg)
        if isinstance(dates, pd.Series):
            missing = x_mat.index.difference(dates.index)
            if len(missing) > 0:
                msg = f"`groups_by_date` no cubre {len(missing)} event_id de X"
                raise DataQualityError(msg)
            d_ser = dates.reindex(x_mat.index)
        else:
            arr_d = np.asarray(dates)
            if arr_d.ndim != 1 or arr_d.shape[0] != len(x_mat):
                msg = f"`groups_by_date` tiene forma {arr_d.shape} y X {len(x_mat)} filas"
                raise DataQualityError(msg)
            d_ser = pd.Series(arr_d, index=x_mat.index)
        stamps = pd.DatetimeIndex(pd.to_datetime(d_ser)).normalize()
        if stamps.isna().any():
            msg = "`groups_by_date` contiene fechas nulas"
            raise DataQualityError(msg)
        d_ser = pd.Series(stamps, index=x_mat.index)

        if len(x_mat) < MIN_FIT_EVENTS:
            msg = f"{len(x_mat)} eventos; se exigen al menos {MIN_FIT_EVENTS}"
            raise InsufficientHistory(msg)
        if self.is_classifier:
            pos = int((y_ser > 0).sum())
            neg = len(y_ser) - pos
            if min(pos, neg) < MIN_CLASS_COUNT:
                msg = (
                    f"clase minoritaria con {min(pos, neg)} eventos "
                    f"(positivos={pos}, negativos={neg}); mínimo {MIN_CLASS_COUNT}"
                )
                raise InsufficientHistory(msg)
        return x_mat, y_ser, d_ser

    # ------------------------------------------------------------------ ajuste

    def fit(
        self,
        X: pd.DataFrame,  # noqa: N803 - firma del contrato §3.5
        y: pd.Series | Sequence[float] | np.ndarray,
        groups_by_date: pd.Series | Sequence[Any] | np.ndarray,
    ) -> Self:
        """Ajusta el modelo. Las fechas son obligatorias por contrato.

        Para ``objective="sign"`` la etiqueta se binariza como ``y > 0`` (el
        *meet* exacto cuenta como no-positivo). Con calibración activa, el
        pipeline base se ajusta en la parte temprana del entrenamiento y el
        calibrador en la cola temporal final, con embargo entre ambas — el
        calibrador nunca ve datos usados por el modelo base ni viceversa.
        """
        x_mat, y_ser, d_ser = self._prepare(X, y, groups_by_date)
        self.feature_names_ = tuple(str(c) for c in x_mat.columns)
        self.dropped_features_ = tuple(x_mat.attrs.get("dropped_all_nan", []))
        target = (y_ser > 0).astype(int) if self.is_classifier else y_ser

        self.pipeline_ = self._build_pipeline()
        self.calibrator_ = None
        if self.is_classifier and self.calibration != "none":
            base_idx, cal_idx = self._calibration_split(d_ser, target)
            self.pipeline_.fit(x_mat.iloc[base_idx], target.iloc[base_idx])
            raw = self._raw_prob(x_mat.iloc[cal_idx])
            self.calibrator_ = self._fit_calibrator(raw, target.iloc[cal_idx].to_numpy())
        else:
            self.pipeline_.fit(x_mat, target)

        self.train_prevalence_ = float((y_ser > 0).mean())
        self.train_mean_ = float(y_ser.mean())
        self.n_train_ = int(len(x_mat))
        return self

    def _calibration_split(
        self, dates: pd.Series, target: pd.Series
    ) -> tuple[np.ndarray, np.ndarray]:
        """Partición temporal base/calibración con embargo (posicional)."""
        n = len(dates)
        order = np.argsort(dates.to_numpy(), kind="stable")
        n_cal = max(int(round(n * self.calibration_fraction)), MIN_CALIBRATION_EVENTS)
        if n - n_cal < MIN_FIT_EVENTS:
            msg = (
                f"{n} eventos no bastan para separar {n_cal} de calibración y "
                f"{MIN_FIT_EVENTS} de ajuste base"
            )
            raise InsufficientHistory(msg)
        cal = order[n - n_cal :]
        base = order[: n - n_cal]
        cal_start = dates.iloc[cal].min()
        cutoff = cal_start - pd.Timedelta(days=self.calibration_embargo_days)
        base = base[(dates.iloc[base] <= cutoff).to_numpy()]
        if len(base) < MIN_FIT_EVENTS:
            msg = (
                f"tras el embargo de {self.calibration_embargo_days} días quedan "
                f"{len(base)} eventos base (< {MIN_FIT_EVENTS})"
            )
            raise InsufficientHistory(msg)
        for name, part in (("base", base), ("calibración", cal)):
            if target.iloc[part].nunique() < 2:
                msg = f"la parte de {name} tiene una sola clase: calibración imposible"
                raise InsufficientHistory(msg)
        return np.sort(base), np.sort(cal)

    def _fit_calibrator(self, raw: np.ndarray, y: np.ndarray) -> tuple[str, Any]:
        if self.calibration == "isotonic":
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(raw, y)
            return ("isotonic", iso)
        platt = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000)
        platt.fit(_logit(raw).reshape(-1, 1), y)
        return ("sigmoid", platt)

    def _apply_calibrator(self, raw: np.ndarray) -> np.ndarray:
        kind, cal = self.calibrator_
        if kind == "isotonic":
            return np.asarray(cal.predict(raw), dtype=float)
        proba = cal.predict_proba(_logit(raw).reshape(-1, 1))
        col = list(cal.classes_).index(1)
        return np.asarray(proba[:, col], dtype=float)

    def _raw_prob(self, x_mat: pd.DataFrame) -> np.ndarray:
        proba = self.pipeline_.predict_proba(x_mat)
        col = list(self.pipeline_.classes_).index(1)
        return np.asarray(proba[:, col], dtype=float)

    # -------------------------------------------------------------- predicción

    def _align(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        if self.pipeline_ is None:
            msg = "el modelo no está ajustado: llama a fit() antes de predecir"
            raise DataQualityError(msg)
        if not isinstance(X, pd.DataFrame):
            msg = f"X debe ser un DataFrame; recibido {type(X).__name__}"
            raise DataQualityError(msg)
        missing = [c for c in self.feature_names_ if c not in X.columns]
        if missing:
            msg = f"a X le faltan {len(missing)} features del ajuste (muestra: {missing[:5]})"
            raise DataQualityError(msg)
        return X[list(self.feature_names_)].astype(float)

    def predict(self, X: pd.DataFrame) -> pd.Series:  # noqa: N803 - contrato §3.5
        """Predicción por evento, alineada con el índice de ``X``.

        - ``objective="sign"``: probabilidad **continua** de sorpresa/retorno
          positivo, calibrada si el modelo se ajustó con calibración. Nunca una
          clase dura: el uso legítimo es ponderar exposición (informe §12.4).
        - ``objective="magnitude"``: valor predicho de la etiqueta.
        """
        x_mat = self._align(X)
        if self.is_classifier:
            p = self._raw_prob(x_mat)
            if self.calibrator_ is not None:
                p = self._apply_calibrator(p)
            return pd.Series(
                np.clip(p, 0.0, 1.0), index=x_mat.index, name=f"p_positive_{self.target}"
            )
        values = np.asarray(self.pipeline_.predict(x_mat), dtype=float)
        return pd.Series(values, index=x_mat.index, name=f"pred_{self.target}")

    def predict_class(self, X: pd.DataFrame, *, threshold: float = 0.5) -> pd.Series:  # noqa: N803
        """Clase dura ±1 a partir de la probabilidad; solo para diagnóstico.

        El umbral por defecto de 0,5 sobre una prevalencia del ~64 % reproduce
        casi siempre a la clase mayoritaria: la métrica de mérito del modelo es
        el AUC/Brier de la probabilidad, no la exactitud de esta clase.
        """
        if not self.is_classifier:
            msg = "predict_class solo aplica a objective='sign'"
            raise DataQualityError(msg)
        prob = self.predict(X)
        return pd.Series(
            np.where(prob.to_numpy() >= threshold, 1.0, -1.0),
            index=prob.index,
            name=f"class_{self.target}",
        )

    def coefficients(self) -> pd.Series:
        """Coeficientes del modelo lineal por desviación típica de cada feature.

        Al ir tras `StandardScaler`, cada coeficiente es el efecto de +1σ de la
        feature sobre el logit (signo) o la etiqueta (magnitud): comparable
        entre features. Ordenados por magnitud absoluta. Para ``"gbm"`` no hay
        coeficientes: usa la permutation importance OOS de `evaluate`.
        """
        if self.pipeline_ is None:
            msg = "el modelo no está ajustado: no hay coeficientes"
            raise DataQualityError(msg)
        if self.estimator != "ridge":
            msg = (
                "los coeficientes solo existen para el estimador lineal; para 'gbm' "
                "usa la permutation importance fuera de muestra del informe de evaluate()"
            )
            raise DataQualityError(msg)
        model = self.pipeline_.named_steps["model"]
        coef = np.ravel(model.coef_)
        out = pd.Series(coef, index=list(self.feature_names_), name="coef_per_sd")
        return out.iloc[np.argsort(-np.abs(out.to_numpy()), kind="stable")]

    # -------------------------------------------------------------- evaluación

    def _holdout_score(self, y_true: pd.Series, pred: pd.Series) -> float:
        """Métrica de un conjunto retenido: AUC (signo) o Spearman (magnitud).

        Devuelve NaN cuando la métrica no está definida (una sola clase en el
        fold, predicción constante); el agregado global sigue siendo calculable.
        """
        p = pred.to_numpy(dtype=float)
        if self.is_classifier:
            y_bin = y_true.to_numpy(dtype=float) > 0
            if y_bin.all() or (~y_bin).all():
                return float("nan")
            auc, _ = roc_auc(p, y_bin)
            return auc
        y_arr = y_true.to_numpy(dtype=float)
        mask = np.isfinite(p) & np.isfinite(y_arr)
        # `ptp` y no `std`: la desviación de una constante puede no ser 0.0 exacto
        # por redondeo de la suma, y spearmanr avisaría de entrada constante.
        if int(mask.sum()) < 3 or np.ptp(p[mask]) == 0.0 or np.ptp(y_arr[mask]) == 0.0:
            return float("nan")
        return float(sp_stats.spearmanr(p[mask], y_arr[mask]).statistic)

    def evaluate(
        self,
        X: pd.DataFrame,  # noqa: N803
        y: pd.Series | Sequence[float] | np.ndarray,
        dates: pd.Series | Sequence[Any] | np.ndarray,
        *,
        n_splits: int = 5,
        embargo: float = 0.02,
        timeline: Sequence[Any] | pd.DatetimeIndex | None = None,
        lookback: int = DEFAULT_FEATURE_LOOKBACK,
        lookahead: int = DEFAULT_LABEL_LOOKAHEAD,
        n_permutation_repeats: int = 5,
        n_bins: int = 10,
        min_train: int = MIN_FIT_EVENTS,
    ) -> SurpriseModelReport:
        """Evaluación fuera de muestra con CV purgada y embargo (única permitida).

        Construye el span de información ``[T - lookback, T + lookahead]`` de
        cada evento (`stats.validation.information_spans`) y parte el panel en
        `n_splits` bloques temporales contiguos con `PurgedKFold`: del
        entrenamiento se elimina todo evento cuyo span solape el test, más el
        embargo hacia adelante. **No hay opción de barajado** — ver el docstring
        del módulo para la prohibición del K-fold aleatorio.

        Parameters
        ----------
        timeline:
            Rejilla de sesiones (`TradingCalendar.sessions(...)` o
            `SyntheticMarket.sessions`) sobre la que contar `lookback`,
            `lookahead` y el embargo **en sesiones**. Sin ella se cuentan días
            naturales: 250 sesiones ≈ 362 días naturales, así que en ese caso
            hay que escalar los parámetros al alza — quedarse corto purga de
            menos, y purgar de menos es look-ahead silencioso.
        lookback, lookahead:
            Span de información. Los defaults (250, 10) corresponden a
            `PreEventFeatures` con estimación en ``[-250, -40]`` y a la
            etiqueta ``CAR[0,+1]``; con etiquetas ``CAR[0,+60]`` hay que pasar
            ``lookahead=60`` (el span de 310 sesiones del informe §8.2).
        n_permutation_repeats:
            Repeticiones de la permutation importance **sobre el test de cada
            fold** (jamás in-sample); 0 la desactiva.

        Returns
        -------
        SurpriseModelReport
            Predicciones OOS (cada evento cae en exactamente un test), métricas
            con sus líneas base (clase mayoritaria / prevalencia entrenada por
            fold; consenso = 0), curva de calibración (signo), permutation
            importance agregada y el coste de la purga fold a fold.
        """
        x_mat, y_ser, d_ser = self._prepare(X, y, dates)
        spans = information_spans(
            d_ser, lookback=lookback, lookahead=lookahead, timeline=timeline, ids=x_mat.index
        )
        cv = PurgedKFold(n_splits=n_splits, embargo=embargo, timeline=timeline)
        purge = cv.purge_report(spans)

        y_bin = (y_ser > 0).astype(int)
        oos_pred = pd.Series(np.nan, index=x_mat.index, dtype=float, name="oos_prediction")
        base_const = pd.Series(np.nan, index=x_mat.index, dtype=float)
        fold_rows: list[dict[str, float]] = []
        importance_samples: dict[str, list[float]] = {str(c): [] for c in x_mat.columns}

        for split_id, (train, test) in enumerate(cv.split(spans)):
            if len(train) < min_train:
                msg = (
                    f"fold {split_id}: solo {len(train)} eventos de entrenamiento tras "
                    f"purga y embargo (mínimo {min_train}). El span de {lookback}+"
                    f"{lookahead} pasos es caro (validation_methodology.md §8.4): "
                    "reduce n_splits o amplía el panel"
                )
                raise InsufficientHistory(msg)
            model = self._clone()
            model.fit(x_mat.iloc[train], y_ser.iloc[train], d_ser.iloc[train])
            x_test = x_mat.iloc[test]
            y_test = y_ser.iloc[test]
            pred = model.predict(x_test)
            oos_pred.iloc[test] = pred.to_numpy()
            base_const.iloc[test] = (
                float(y_bin.iloc[train].mean())
                if self.is_classifier
                else float(y_ser.iloc[train].mean())
            )

            score = self._holdout_score(y_test, pred)
            base_score = self._holdout_score(
                y_test, pd.Series(base_const.iloc[test].to_numpy(), index=y_test.index)
            )
            fold_rows.append(
                {
                    "fold": split_id,
                    "n_train": int(len(train)),
                    "n_test": int(len(test)),
                    "n_purged": int(len(x_mat) - len(train) - len(test)),
                    "score": score,
                    "baseline_score": base_score,
                }
            )

            if n_permutation_repeats > 0 and np.isfinite(score):
                rng = np.random.default_rng(self.seed + 7919 * (split_id + 1))
                for col in x_mat.columns:
                    original = x_test[col].to_numpy(copy=True)
                    for _ in range(n_permutation_repeats):
                        x_perm = x_test.copy()
                        x_perm[col] = rng.permutation(original)
                        s_perm = self._holdout_score(y_test, model.predict(x_perm))
                        if np.isfinite(s_perm):
                            importance_samples[str(col)].append(score - s_perm)

        metrics, baselines, calibration = self._oos_metrics(
            y_ser, y_bin, oos_pred, base_const, n_bins=n_bins
        )
        importance = _aggregate_importance(importance_samples)
        return SurpriseModelReport(
            target=self.target,
            objective=self.objective,
            estimator=self.estimator,
            metrics=metrics,
            baselines=baselines,
            fold_metrics=pd.DataFrame(fold_rows).set_index("fold"),
            oos_prediction=oos_pred,
            oos_target=y_ser.rename("oos_target"),
            importance=importance,
            calibration=calibration,
            purge=purge,
            dropped_features=tuple(x_mat.attrs.get("dropped_all_nan", [])),
            n_splits=n_splits,
        )

    def _oos_metrics(
        self,
        y_ser: pd.Series,
        y_bin: pd.Series,
        oos_pred: pd.Series,
        base_const: pd.Series,
        *,
        n_bins: int,
    ) -> tuple[dict[str, float], dict[str, float], pd.DataFrame | None]:
        """Métricas OOS agregadas y sus líneas base emparejadas por clave."""
        pred = oos_pred.to_numpy(dtype=float)
        base = base_const.to_numpy(dtype=float)
        covered = np.isfinite(pred)
        if not covered.all():  # pragma: no cover - PurgedKFold cubre todo el panel
            msg = f"{int((~covered).sum())} eventos sin predicción OOS"
            raise DataQualityError(msg)

        if self.is_classifier:
            yb = y_bin.to_numpy(dtype=float)
            y_bool = yb > 0
            auc, auc_se = roc_auc(pred, y_bool)
            hard = pred >= 0.5
            tpr = float(np.mean(hard[y_bool])) if y_bool.any() else float("nan")
            tnr = float(np.mean(~hard[~y_bool])) if (~y_bool).any() else float("nan")
            majority_hard = base >= 0.5
            metrics = {
                "n_oos": float(len(pred)),
                "prevalence": float(y_bool.mean()),
                "auc": auc,
                "auc_se": auc_se,
                "auc_t": (auc - 0.5) / auc_se,
                "accuracy": float(np.mean(hard == y_bool)),
                "balanced_accuracy": 0.5 * (tpr + tnr),
                "brier": float(np.mean((pred - yb) ** 2)),
                "log_loss": _log_loss(yb, pred),
                "average_precision": float(average_precision_score(y_bool, pred)),
                "ece": expected_calibration_error(y_bool, np.clip(pred, 0.0, 1.0), n_bins=n_bins),
            }
            baselines = {
                "accuracy": float(np.mean(majority_hard == y_bool)),
                "auc": 0.5,
                "brier": float(np.mean((base - yb) ** 2)),
                "log_loss": _log_loss(yb, base),
                # AP de un clasificador sin información = prevalencia (Saito-Rehmsmeier 2015).
                "average_precision": float(y_bool.mean()),
            }
            calibration = calibration_curve(y_bool, np.clip(pred, 0.0, 1.0), n_bins=n_bins)
            return metrics, baselines, calibration

        y_arr = y_ser.to_numpy(dtype=float)
        sse_model = float(np.sum((pred - y_arr) ** 2))
        sse_base = float(np.sum((base - y_arr) ** 2))
        sse_zero = float(np.sum(y_arr**2))
        try:
            rho, pval = spearman_ic(pred, y_arr)
        except DataQualityError:  # predicción degenerada constante: IC indefinido
            rho, pval = float("nan"), float("nan")
        metrics = {
            "n_oos": float(len(pred)),
            "mse": sse_model / len(pred),
            "mae": float(np.mean(np.abs(pred - y_arr))),
            # R² OOS de Campbell-Thompson (2008): contra la media de entrenamiento.
            "r2_oos": 1.0 - sse_model / sse_base if sse_base > 0 else float("nan"),
            # Contra el consenso (sorpresa/retorno esperado = 0). Cuidado: la mera
            # media (prima de anuncio) ya lo bate; por eso el IC acompaña siempre.
            "r2_vs_consensus": 1.0 - sse_model / sse_zero if sse_zero > 0 else float("nan"),
            "spearman_ic": rho,
            "spearman_p": pval,
        }
        baselines = {
            "mse": sse_base / len(pred),
            "mse_consensus_zero": sse_zero / len(pred),
            "spearman_ic": 0.0,
        }
        return metrics, baselines, None


def _aggregate_importance(samples: dict[str, list[float]]) -> pd.DataFrame:
    """Agrega las muestras de permutation importance por feature (media, std, n).

    La importancia de una feature es la **caída de la métrica OOS** al permutarla
    en el test de cada fold: mide dependencia real del modelo desplegado, no la
    atención in-sample (que en árboles sobrestima features de alta cardinalidad).
    Puede ser negativa: permutar ruido a veces mejora la métrica por azar.
    """
    rows = []
    for name, vals in samples.items():
        if vals:
            arr = np.asarray(vals, dtype=float)
            rows.append(
                {
                    "feature": name,
                    "importance_mean": float(arr.mean()),
                    "importance_std": float(arr.std(ddof=1)) if len(arr) > 1 else float("nan"),
                    "n_samples": int(len(arr)),
                }
            )
        else:
            rows.append(
                {
                    "feature": name,
                    "importance_mean": float("nan"),
                    "importance_std": float("nan"),
                    "n_samples": 0,
                }
            )
    frame = pd.DataFrame(rows).set_index("feature")
    return frame.sort_values("importance_mean", ascending=False)


# ---------------------------------------------------------------------------
# Comparación de estimadores
# ---------------------------------------------------------------------------


def compare_estimators(
    X: pd.DataFrame,  # noqa: N803
    y: pd.Series | Sequence[float] | np.ndarray,
    dates: pd.Series | Sequence[Any] | np.ndarray,
    *,
    base_model: SurpriseModel | None = None,
    estimators: Sequence[Literal["ridge", "gbm"]] = ("ridge", "gbm"),
    **evaluate_kwargs: Any,
) -> pd.DataFrame:
    """Compara los estimadores sobre LOS MISMOS folds purgados.

    La comparación es honesta por construcción: mismos spans, misma purga, mismo
    embargo y mismas observaciones de test para todos; solo cambia el estimador.
    Devuelve una fila por estimador con las métricas OOS, las líneas base con
    prefijo ``baseline_`` y el veredicto de cribado ``beats_baseline``.

    La permutation importance se desactiva por defecto (es la parte cara y no
    interviene en la comparación); reactívala con ``n_permutation_repeats=...``.
    """
    base = base_model if base_model is not None else SurpriseModel()
    evaluate_kwargs.setdefault("n_permutation_repeats", 0)
    rows: dict[str, dict[str, float | bool]] = {}
    for est in estimators:
        model = replace(base, estimator=est)
        report = model.evaluate(X, y, dates, **evaluate_kwargs)
        row: dict[str, float | bool] = dict(report.metrics)
        row.update({f"baseline_{k}": v for k, v in report.baselines.items()})
        row["beats_baseline"] = report.beats_baseline
        rows[str(est)] = row
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "estimator"
    return out
