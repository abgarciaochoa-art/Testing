"""Vintages y reexpresiones: separar el dato *tal y como se reportó* del dato de hoy.

Un `FundamentalFact` no es una cifra: es una cifra **con una fecha de
disponibilidad**. La misma tupla `(ticker, concept, period_end)` puede tener
varias observaciones a lo largo del tiempo —vintages— porque:

- el preliminar del 8-K item 2.02 se corrige en el 10-Q definitivo;
- una reexpresión (10-K/A) corrige un error contable posterior;
- una operación discontinuada reclasifica hacia atrás toda la serie de ventas,
  sin que haya habido error alguno.

Los proveedores comerciales sirven casi siempre la **última** versión. Un factor
de calidad calculado sobre datos reexpresados usa información que no existía en
la fecha de la señal; la literatura sitúa ese sesgo en el orden de 100 pb anuales
para factores de calidad y acumulaciones.

Referencias:
  - Sloan (1996), *Do Stock Prices Fully Reflect Information in Accruals and Cash
    Flows about Future Earnings?*: el factor de accruals es especialmente sensible
    a la versión del balance que se use.
  - Piotroski (2000), *Value Investing: The Use of Historical Financial Statement
    Information*: el F-score se construye con estados financieros ya publicados;
    reexpresiones posteriores cambian varias de sus nueve señales.
  - Ball, Gerakos, Linnainmaa y Nikolaev (2016) sobre la sensibilidad de los
    factores contables a la definición y a la vintage de los datos.
  - Ideagen Audit Analytics (2024), *Financial Restatements: A Twenty-Year
    Review*: base de la magnitud típica y de la frecuencia de reexpresión.
  - Kothari (2001) y Lee (2001) sobre el sesgo de "as-restated" en la
    investigación en contabilidad y mercados.

Convenciones de este módulo:
  - clave de hecho: `(ticker, concept, period_end)`;
  - el orden temporal lo da `available_at`, nunca `period_end`;
  - `first_reported` es la serie honesta para un backtest; `as_restated` solo
    sirve para medir cuánto se habría distorsionado el backtest al usarla.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from earnings_alpha.errors import DataQualityError, InsufficientHistory
from earnings_alpha.pit.asof import to_naive_utc
from earnings_alpha.types import FundamentalFact

__all__ = [  # noqa: RUF022 - orden temático, no alfabético
    "FACT_KEY",
    "facts_to_frame",
    "first_reported",
    "as_restated",
    "vintage_asof",
    "revisions",
    "restatement_magnitude",
    "upsert_vintage",
    "RestatementStats",
]

FACT_KEY: tuple[str, str, str] = ("ticker", "concept", "period_end")
"""Clave que identifica un hecho fundamental; sus vintages comparten esta clave."""

_REQUIRED = ("ticker", "concept", "period_end", "available_at", "value")

# Denominador mínimo para una revisión relativa. Por debajo de este valor
# absoluto, el cociente explota y no informa de nada.
_MIN_DENOM = 1e-9


@dataclass(frozen=True, slots=True)
class RestatementStats:
    """Resumen de la magnitud típica de las reexpresiones de un histórico.

    `rel_*` son revisiones relativas (final vs. primer vintage) medidas sobre el
    valor absoluto del primer vintage, que es el denominador honesto: es la cifra
    que un backtest habría usado en su momento.
    """

    n_keys: int
    """Tuplas `(ticker, concept, period_end)` distintas."""
    n_facts: int
    """Observaciones totales (suma de vintages)."""
    n_revised_keys: int
    """Claves con más de un vintage y con cambio efectivo de valor."""
    revised_share: float
    """Fracción de claves revisadas."""
    median_abs_rel: float
    """Mediana de |revisión relativa| entre las claves revisadas."""
    mean_abs_rel: float
    p90_abs_rel: float
    median_signed_rel: float
    """Mediana de la revisión relativa con signo: mide si las correcciones son
    sistemáticamente a la baja (el patrón esperado si hubo inflado inicial)."""
    negative_share: float
    """Fracción de revisiones que bajan la cifra reportada."""
    median_delay_days: float
    """Mediana de días naturales entre el primer vintage y la revisión."""
    max_abs_rel: float
    by_concept: pd.DataFrame = field(repr=False)
    """Desglose por concepto: nº de claves, nº revisadas y mediana de |revisión|."""

    def as_dict(self) -> dict[str, float | int]:
        """Vista plana y serializable (sin el desglose por concepto)."""
        return {
            "n_keys": self.n_keys,
            "n_facts": self.n_facts,
            "n_revised_keys": self.n_revised_keys,
            "revised_share": self.revised_share,
            "median_abs_rel": self.median_abs_rel,
            "mean_abs_rel": self.mean_abs_rel,
            "p90_abs_rel": self.p90_abs_rel,
            "median_signed_rel": self.median_signed_rel,
            "negative_share": self.negative_share,
            "median_delay_days": self.median_delay_days,
            "max_abs_rel": self.max_abs_rel,
        }


# ---------------------------------------------------------------------------
# Normalización de entrada
# ---------------------------------------------------------------------------


def facts_to_frame(facts: Iterable[FundamentalFact] | pd.DataFrame) -> pd.DataFrame:
    """Normaliza `FundamentalFact`s o un DataFrame a la tabla canónica de vintages.

    Columnas garantizadas: `ticker`, `concept`, `period_end` (datetime64),
    `available_at` (datetime64 tz-naive en UTC), `value` (float),
    `is_restated` (bool), y `form`/`accession`/`fiscal_period`/`unit` si vienen.
    """
    if isinstance(facts, pd.DataFrame):
        df = facts.copy()
        if isinstance(df.index, pd.MultiIndex) or df.index.name in _REQUIRED:
            df = df.reset_index()
    else:
        rows = [
            {
                "ticker": f.ticker,
                "concept": f.concept,
                "value": f.value,
                "period_end": f.period_end,
                "available_at": f.available_at,
                "fiscal_period": f.fiscal_period,
                "unit": f.unit,
                "form": f.form,
                "accession": f.accession,
                "is_restated": f.is_restated,
            }
            for f in facts
        ]
        df = pd.DataFrame(rows)

    if len(df) == 0:
        msg = "no hay hechos fundamentales: no se puede construir la serie de vintages"
        raise InsufficientHistory(msg)

    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        msg = f"faltan columnas obligatorias en el histórico de hechos: {missing}"
        raise DataQualityError(msg)

    df["ticker"] = df["ticker"].astype(str)
    df["concept"] = df["concept"].astype(str)
    df["period_end"] = (
        pd.DatetimeIndex(pd.to_datetime(df["period_end"])).normalize().as_unit("ns")
    )

    df["available_at"] = pd.Series(
        to_naive_utc(df["available_at"]).to_numpy(), index=df.index, dtype="datetime64[ns]"
    )
    if df["available_at"].isna().any():
        n_bad = int(df["available_at"].isna().sum())
        msg = (
            f"{n_bad} hechos sin `available_at`: un hecho sin fecha de disponibilidad "
            "pública no es point-in-time y no puede entrar en una señal"
        )
        raise DataQualityError(msg)

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    if "is_restated" not in df.columns:
        df["is_restated"] = False
    df["is_restated"] = df["is_restated"].fillna(False).astype(bool)
    for col in ("form", "accession", "fiscal_period", "unit"):
        if col not in df.columns:
            df[col] = None

    return df.sort_values([*FACT_KEY, "available_at"], kind="mergesort").reset_index(drop=True)


def _ordered(facts: Iterable[FundamentalFact] | pd.DataFrame) -> pd.DataFrame:
    """Tabla canónica ordenada por clave y `available_at` ascendente."""
    return facts_to_frame(facts)


# ---------------------------------------------------------------------------
# Series as-first-reported / as-restated
# ---------------------------------------------------------------------------


def first_reported(facts: Iterable[FundamentalFact] | pd.DataFrame) -> pd.DataFrame:
    """Serie **as-first-reported**: el primer vintage de cada clave.

    Es la única serie admisible en un backtest honesto: reproduce lo que un
    inversor podía leer el día en que el dato se publicó por primera vez.

    Desempate: si dos vintages comparten `available_at`, gana el no reexpresado;
    si aun así hay empate, el primero en el orden estable de entrada.
    """
    df = _ordered(facts)
    order = df.assign(__restated__=df["is_restated"].astype(int)).sort_values(
        [*FACT_KEY, "available_at", "__restated__"], kind="mergesort"
    )
    out = order.groupby(list(FACT_KEY), as_index=False, sort=True).first()
    return out.drop(columns="__restated__").reset_index(drop=True)


def as_restated(facts: Iterable[FundamentalFact] | pd.DataFrame) -> pd.DataFrame:
    """Serie **as-restated**: el último vintage conocido de cada clave.

    Es lo que sirve un proveedor comercial que sobrescribe. **No debe usarse para
    generar señales**; existe para poder cuantificar, comparándola con
    `first_reported`, cuánto sesgo introduciría usarla.
    """
    df = _ordered(facts)
    out = df.groupby(list(FACT_KEY), as_index=False, sort=True).last()
    return out.reset_index(drop=True)


def vintage_asof(
    facts: Iterable[FundamentalFact] | pd.DataFrame,
    as_of: datetime | pd.Timestamp | str,
) -> pd.DataFrame:
    """Vintage vigente de cada clave en el instante `as_of`.

    Es la reconstrucción correcta del almacén de datos tal y como estaba en una
    fecha pasada: incluye las reexpresiones ya publicadas a esa fecha (que sí eran
    conocibles) y excluye las posteriores. Claves cuyo primer vintage es posterior
    a `as_of` simplemente no aparecen: no existían.
    """
    df = _ordered(facts)
    cut = pd.Timestamp(as_of)
    if cut.tzinfo is not None:
        cut = cut.tz_convert("UTC").tz_localize(None)
    visible = df[df["available_at"] <= cut]
    if len(visible) == 0:
        return df.iloc[0:0].copy()
    return visible.groupby(list(FACT_KEY), as_index=False, sort=True).last().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Medición de reexpresiones
# ---------------------------------------------------------------------------


def revisions(
    facts: Iterable[FundamentalFact] | pd.DataFrame,
    *,
    rel_tol: float = 1e-9,
) -> pd.DataFrame:
    """Tabla de revisión por clave: primer vintage frente al último.

    Columnas devueltas
    ------------------
    `ticker`, `concept`, `period_end`, `n_vintages`,
    `first_value`, `first_available_at`, `first_form`,
    `final_value`, `final_available_at`, `final_form`,
    `abs_revision` (final - primero), `rel_revision` (sobre |primero|),
    `delay_days` (días naturales entre ambos vintages),
    `is_revised` (cambio de valor por encima de `rel_tol`),
    `flagged_restated` (algún vintage venía marcado `is_restated`).

    `rel_revision` es NaN cuando el primer vintage es prácticamente cero: una
    revisión relativa sobre un denominador nulo no es informativa y rellenarla
    con un número enorme contaminaría cualquier estadístico agregado.
    """
    df = _ordered(facts)
    grouped = df.groupby(list(FACT_KEY), sort=True)

    first = first_reported(df).set_index(list(FACT_KEY))
    last = as_restated(df).set_index(list(FACT_KEY))
    counts = grouped.size().rename("n_vintages")
    flagged = grouped["is_restated"].any().rename("flagged_restated")

    out = pd.DataFrame(
        {
            "n_vintages": counts,
            "first_value": first["value"],
            "first_available_at": first["available_at"],
            "first_form": first["form"],
            "final_value": last["value"],
            "final_available_at": last["available_at"],
            "final_form": last["form"],
            "flagged_restated": flagged,
        }
    ).reset_index()

    out["abs_revision"] = out["final_value"] - out["first_value"]
    denom = out["first_value"].abs()
    out["rel_revision"] = np.where(
        denom > _MIN_DENOM, out["abs_revision"] / denom.replace(0.0, np.nan), np.nan
    )
    out["delay_days"] = (out["final_available_at"] - out["first_available_at"]).dt.days

    changed = out["abs_revision"].abs() > (denom * rel_tol).clip(lower=_MIN_DENOM)
    out["is_revised"] = (out["n_vintages"] > 1) & changed.fillna(False)
    return out


def restatement_magnitude(
    facts: Iterable[FundamentalFact] | pd.DataFrame,
    *,
    concepts: Sequence[str] | None = None,
    rel_tol: float = 1e-9,
) -> RestatementStats:
    """Mide la magnitud típica de las reexpresiones de un histórico de vintages.

    Sirve para dos cosas: (a) documentar cuánto se aparta la serie reexpresada de
    la original en los datos concretos con los que se está trabajando, en vez de
    citar una cifra genérica de la literatura; (b) detectar proveedores que
    reescriben la historia sin avisar (una fracción de revisión sospechosamente
    alta o baja es diagnóstica).

    Lanza `InsufficientHistory` si no hay ninguna clave que medir: devolver un
    resumen de ceros haría creer que el proveedor es limpio cuando puede que solo
    esté vacío.
    """
    rev = revisions(facts, rel_tol=rel_tol)
    if concepts is not None:
        rev = rev[rev["concept"].isin(list(concepts))]
    if len(rev) == 0:
        msg = "no hay claves (ticker, concept, period_end) que medir"
        raise InsufficientHistory(msg)

    revised = rev[rev["is_revised"]]
    rel = revised["rel_revision"].dropna()
    abs_rel = rel.abs()

    by_concept = (
        rev.groupby("concept")
        .agg(
            n_keys=("is_revised", "size"),
            n_revised=("is_revised", "sum"),
            median_abs_rel=("rel_revision", lambda s: float(s.abs().median()) if len(s) else np.nan),
        )
        .assign(revised_share=lambda d: d["n_revised"] / d["n_keys"])
        .sort_values("n_keys", ascending=False)
    )

    return RestatementStats(
        n_keys=len(rev),
        n_facts=int(rev["n_vintages"].sum()),
        n_revised_keys=len(revised),
        revised_share=float(len(revised) / len(rev)),
        median_abs_rel=float(abs_rel.median()) if len(abs_rel) else float("nan"),
        mean_abs_rel=float(abs_rel.mean()) if len(abs_rel) else float("nan"),
        p90_abs_rel=float(abs_rel.quantile(0.90)) if len(abs_rel) else float("nan"),
        median_signed_rel=float(rel.median()) if len(rel) else float("nan"),
        negative_share=float((rel < 0).mean()) if len(rel) else float("nan"),
        median_delay_days=float(revised["delay_days"].median()) if len(revised) else float("nan"),
        max_abs_rel=float(abs_rel.max()) if len(abs_rel) else float("nan"),
        by_concept=by_concept,
    )


# ---------------------------------------------------------------------------
# Almacén append-only
# ---------------------------------------------------------------------------


def upsert_vintage(
    store: pd.DataFrame | None,
    new: Iterable[FundamentalFact] | pd.DataFrame,
    observed_at: datetime | pd.Timestamp,
    *,
    rel_tol: float = 1e-9,
) -> pd.DataFrame:
    """Inserta hechos nuevos como vintage adicional; **nunca** sobrescribe.

    Reglas:

    1. Una clave que no estaba en el almacén se inserta con su `available_at`.
    2. Una clave que estaba y llega con **el mismo valor** no genera vintage
       nuevo: un proveedor que repite la misma cifra cada día no es información.
    3. Una clave que estaba y llega con **valor distinto** se inserta como vintage
       nuevo, marcado `is_restated=True`. Su `available_at` es el que traiga el
       hecho entrante y, si no trae uno posterior al último conocido, se usa
       `observed_at`: lo más temprano que podemos *demostrar* que el nuevo valor
       era conocible es el momento en que lo observamos.

    Esta es la disciplina que permite reconstruir vintages incluso con
    proveedores que solo sirven la última versión: se observa a diario y se
    registra cada cambio.

    Un lote (`new`) representa **una** observación del proveedor, así que no
    puede traer dos vintages de la misma clave: si los trae, se lanza
    `DataQualityError` en vez de resolver el conflicto adivinando cuál es el
    posterior.
    """
    incoming = _ordered(new)
    dup = incoming.duplicated(subset=list(FACT_KEY))
    if bool(dup.any()):
        sample = incoming.loc[dup, list(FACT_KEY)].head(3).to_dict("records")
        msg = (
            "un lote de `upsert_vintage` no puede traer dos vintages de la misma clave: "
            "cada observación del proveedor es un lote. Claves duplicadas: "
            f"{sample}"
        )
        raise DataQualityError(msg)

    observed = pd.Timestamp(observed_at)
    if observed.tzinfo is not None:
        observed = observed.tz_convert("UTC").tz_localize(None)

    if store is None or len(store) == 0:
        out = incoming.copy()
        out["available_at"] = out["available_at"].clip(upper=observed)
        return out.sort_values([*FACT_KEY, "available_at"], kind="mergesort").reset_index(drop=True)

    current = _ordered(store)
    last = current.groupby(list(FACT_KEY), sort=False).last()

    keys = pd.MultiIndex.from_frame(incoming[list(FACT_KEY)])
    known = keys.isin(last.index)
    prev_value = pd.Series(
        np.where(known, last["value"].reindex(keys).to_numpy(), np.nan), index=incoming.index
    )
    prev_avail = pd.Series(
        pd.to_datetime(
            np.where(known, last["available_at"].reindex(keys).to_numpy(), np.datetime64("NaT"))
        ),
        index=incoming.index,
    )

    denom = prev_value.abs().clip(lower=_MIN_DENOM)
    unchanged = known & ((incoming["value"] - prev_value).abs() <= denom * rel_tol).to_numpy()
    fresh = ~known
    keep = incoming.loc[fresh | ~unchanged].copy()
    if len(keep) == 0:
        return current

    is_restatement = pd.Series(known, index=incoming.index).loc[keep.index]
    keep["is_restated"] = keep["is_restated"].to_numpy() | is_restatement.to_numpy()

    # `available_at` efectivo: el declarado, pero nunca anterior o igual al del
    # vintage previo (haría irreconstruible el orden), ni posterior a la observación.
    declared = keep["available_at"].clip(upper=observed)
    floor = prev_avail.loc[keep.index]
    needs_bump = floor.notna() & (declared <= floor)
    declared = declared.mask(needs_bump, observed)
    keep["available_at"] = declared

    out = pd.concat([current, keep], ignore_index=True)
    return out.sort_values([*FACT_KEY, "available_at"], kind="mergesort").reset_index(drop=True)
