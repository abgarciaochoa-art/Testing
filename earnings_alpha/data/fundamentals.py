"""Capa alta de fundamentales: EDGAR primero, FMP/EODHD como respaldo.

Este módulo convierte los hechos XBRL crudos de `data.edgar` (y, como respaldo,
los estados financieros de FMP/EODHD) en la **tabla canónica de vintages** que
consumen `pit.restatements` (`first_reported` / `vintage_asof`) y
`factors.FactorContext.fundamentals`:

    ticker, concept, period_end, available_at, value,
    fiscal_period, unit, form, accession, is_restated

con `concept` en el vocabulario **lógico** del repo (``revenue``, ``net_income``,
``cfo``…) y `available_at` point-in-time.

Jerarquía de fuentes (y por qué EDGAR va primero)
-------------------------------------------------
1. **EDGAR** (`prioridad 100`): la única fuente gratuita con **todos los
   vintages** de cada cifra (sec_edgar.md §5.5) — lo que normalmente exige
   Compustat Point-in-Time. `available_at` sale de `filed` con el corte
   administrativo de las 17:30 ET (cota superior segura).
2. **FMP** (`prioridad 50`): estados as-reported con `acceptedDate`/`fillingDate`.
   Sin historial de vintages: sirve la última versión conocida, con la fecha del
   filing original. Útil para cifras pre-XBRL (antes de ~2009) y como contraste.
3. **EODHD** (`prioridad 40`): fundamentales con `filing_date` pero **valores
   reexpresados** (el proveedor sobrescribe). La peor calidad PIT de las tres;
   solo respaldo y diagnóstico.

Cada tabla lleva `df.attrs["pit_quality"]` (``"vintages"`` / ``"filing-date"`` /
``"restated"``) para que el consumidor sepa qué grado de honestidad temporal
tiene lo que está usando, y ningún backtest debería mezclar calidades sin
reportarlo.

Decisiones metodológicas implementadas aquí (sec_edgar.md §6):

- **Cascada de tags estable por empresa** (`resolve_company_series`, trampa nº 8):
  el tag de un concepto lógico se elige una vez por empresa sobre toda su
  historia, no trimestre a trimestre; si la empresa migró de tag (ASC 606), las
  dos series se empalman solo si en el periodo de solape difieren menos de
  `merge_tol` (por defecto 1 %), y el empalme queda registrado en `concept_used`
  y `tag_switch`.
- **`total_debt` no es una cascada** (`reconcile_total_debt`, §6.3): se suma
  corriente + no corriente y solo se cae al agregado si faltan los componentes.
- **Q4 derivado** (`derive_q4`, §6.5): muchos emisores solo etiquetan el Q4
  dentro del 10-K anual; `Q4 = FY − (Q1+Q2+Q3)` con los cuatro sumandos del
  mismo vintage, `available_at` del 10-K y marca `is_derived`. Solo conceptos de
  flujo: los saldos de balance del Q4 sí están en el 10-K como hecho instantáneo.
- **Vintage explícito** (`fundamentals_panel`, §11.1): ``original`` (primer
  `filed`, as-reported), ``pit`` (último `filed` ≤ `as_of`) y ``latest`` (sin
  filtro, **look-ahead deliberado** para cuantificar cuánto rendimiento de un
  factor procede de las reexpresiones — un Sharpe que se desploma al pasar de
  `latest` a `original` estaba viviendo del futuro).

Referencias:
- Ball y Brown (1968) y Sloan (1996) motivan por qué las series deben ser
  *as-originally-reported*: la señal contable se define sobre lo que el mercado
  pudo leer, no sobre la historia reescrita.
- SEC, *EDGAR Application Programming Interfaces* (via docs/research/sec_edgar.md).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import Any, Literal, Protocol, runtime_checkable

import pandas as pd

from earnings_alpha.data.base import (
    BaseProvider,
    DataKind,
    HttpClient,
    ProviderRegistry,
    Transport,
    rate_limit_for,
)
from earnings_alpha.data.cache import DiskCache
from earnings_alpha.data.edgar import (
    SEC_CONCEPT_MAP,
    EdgarProvider,
    classify_periods,
    filed_available_at,
    parse_acceptance_datetime,
)
from earnings_alpha.errors import (
    DataQualityError,
    InsufficientHistory,
    ProviderUnavailable,
)
from earnings_alpha.types import CIK, Ticker, normalize_ticker

__all__ = [  # noqa: RUF022 - orden temático, no alfabético
    # tabla canónica
    "CANONICAL_COLUMNS",
    "FLOW_CONCEPTS",
    "INSTANT_CONCEPTS",
    # resolución de tags y reconciliaciones (puras, testables sin red)
    "resolve_company_tag",
    "resolve_company_series",
    "reconcile_total_debt",
    "derive_q4",
    "fundamentals_panel",
    # proveedores
    "EdgarFundamentals",
    "FMPFundamentals",
    "EODHDFundamentals",
    "FundamentalsService",
    "build_default_registry",
]

logger = logging.getLogger(__name__)

CANONICAL_COLUMNS: tuple[str, ...] = (
    "ticker",
    "concept",
    "period_end",
    "available_at",
    "value",
    "fiscal_period",
    "unit",
    "form",
    "accession",
    "is_restated",
    "concept_used",
    "is_derived",
    "source",
)
"""Columnas de la tabla canónica de fundamentales. Superconjunto de lo que exige
`pit.restatements.facts_to_frame` (FACT_KEY + available_at + value); las tres
últimas son metadatos de auditoría de esta capa."""

FLOW_CONCEPTS: frozenset[str] = frozenset(
    {
        "revenue",
        "cogs",
        "gross_profit",
        "operating_income",
        "net_income",
        "eps_diluted",
        "eps_basic",
        "shares_diluted",
        "cfo",
        "capex",
    }
)
"""Conceptos de duración (P&G y flujo de caja): admiten derivación de Q4."""

INSTANT_CONCEPTS: frozenset[str] = frozenset(
    {"total_assets", "total_equity", "total_debt", "cash"}
)
"""Saldos de balance: el Q4 ya está en el 10-K como hecho instantáneo; restarlos
sería un error (sec_edgar.md §6.5, regla 4)."""


# ---------------------------------------------------------------------------
# 1. Resolución de tags estable por empresa (trampa nº 8)
# ---------------------------------------------------------------------------


def _quarterly_slice(facts: pd.DataFrame) -> pd.DataFrame:
    """Hechos utilizables para medir cobertura de un tag: trimestres y anuales
    (duración) o instantáneos, según el tipo de hecho. Excluye acumulados."""
    work = facts if "period_kind" in facts.columns else classify_periods(facts)
    return work[work["period_kind"].isin(("quarter", "annual", "instant"))]


def resolve_company_tag(
    facts: pd.DataFrame,
    logical: str,
    *,
    concept_map: Mapping[str, Sequence[str]] | None = None,
    merge_tol: float = 0.01,
) -> tuple[list[str], pd.DataFrame]:
    """Elige el/los tags us-gaap de un concepto lógico para **una** empresa.

    Reglas (sec_edgar.md §6.2 y trampa nº 8):

    1. La prioridad de la cascada es la documentada en `SEC_CONCEPT_MAP`, pero la
       elección se hace **una vez por empresa sobre toda su historia**, nunca
       trimestre a trimestre: una cascada aplicada por periodo produce series que
       saltan de tag en el trimestre de migración (ASC 606) y disparan señales
       falsas de crecimiento simultáneas en medio universo.
    2. Si un solo tag cubre todos los periodos observados, se usa ese.
    3. Si la empresa **migró** de tag, se empalman el tag antiguo y el moderno
       únicamente si en los periodos de solape difieren menos de `merge_tol`
       relativo; una divergencia mayor es `DataQualityError` (mejor cortar la
       serie que fabricar una aceleración inexistente).

    Devuelve ``(tags_usados_en_orden, hechos_del_concepto)`` con una columna
    `concept_used` (el tag del que salió cada fila) y `tag_switch` (True en las
    filas del tag moderno cuando hubo empalme). Lanza `InsufficientHistory` si
    ningún tag de la cascada aparece para la empresa.
    """
    cmap = dict(concept_map or SEC_CONCEPT_MAP)
    if logical not in cmap:
        msg = f"concepto lógico desconocido: {logical!r}; conocidos: {sorted(cmap)}"
        raise DataQualityError(msg)
    cascade = list(cmap[logical])
    pool = _quarterly_slice(facts)
    present = [t for t in cascade if (pool["concept"] == t).any()]
    if not present:
        msg = (
            f"ningún tag de la cascada de {logical!r} aparece en los hechos de la "
            f"empresa (buscados: {cascade})"
        )
        raise InsufficientHistory(msg)

    def _periods(tag: str) -> set[pd.Timestamp]:
        return set(pool.loc[pool["concept"] == tag, "end"])

    all_periods: set[pd.Timestamp] = set()
    for tag in present:
        all_periods |= _periods(tag)

    # 2. ¿Un solo tag cubre toda la historia? El primero de la cascada gana.
    for tag in present:
        if _periods(tag) >= all_periods:
            out = pool[pool["concept"] == tag].copy()
            out["concept_used"] = tag
            out["tag_switch"] = False
            return [tag], out.reset_index(drop=True)

    # 3. Migración: empalme del primero de la cascada (moderno) con el que
    # cubra los periodos que le faltan, validando el solape.
    primary = present[0]
    primary_periods = _periods(primary)
    remaining = all_periods - primary_periods
    for fallback in present[1:]:
        fb_periods = _periods(fallback)
        if not (remaining & fb_periods):
            continue
        overlap = primary_periods & fb_periods
        if overlap:
            a = (
                pool[(pool["concept"] == primary) & pool["end"].isin(overlap)]
                .groupby("end")["value"]
                .last()
            )
            b = (
                pool[(pool["concept"] == fallback) & pool["end"].isin(overlap)]
                .groupby("end")["value"]
                .last()
            )
            joined = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
            if len(joined):
                denom = joined["a"].abs().clip(lower=1.0)
                rel = ((joined["a"] - joined["b"]).abs() / denom).max()
                if rel > merge_tol:
                    msg = (
                        f"los tags {primary!r} y {fallback!r} de {logical!r} difieren "
                        f"{rel:.2%} en el solape (> {merge_tol:.0%}): no se empalman. "
                        "Mezclarlos crearía un escalón artificial en el trimestre de "
                        "migración (sec_edgar.md trampa nº 8); corta la serie o revisa "
                        "el mapa de conceptos."
                    )
                    raise DataQualityError(msg)
        old_rows = pool[(pool["concept"] == fallback) & pool["end"].isin(remaining)].copy()
        new_rows = pool[pool["concept"] == primary].copy()
        old_rows["concept_used"] = fallback
        old_rows["tag_switch"] = False
        new_rows["concept_used"] = primary
        new_rows["tag_switch"] = True
        out = pd.concat([old_rows, new_rows], ignore_index=True)
        out = out.sort_values(["end", "filed"], kind="mergesort").reset_index(drop=True)
        return [fallback, primary], out

    # Sin solape resoluble: se sirve el tag primario y se registra el hueco.
    out = pool[pool["concept"] == primary].copy()
    out["concept_used"] = primary
    out["tag_switch"] = False
    logger.warning(
        "concepto %s: el tag %s no cubre %d periodos y ningún fallback los aporta",
        logical,
        primary,
        len(remaining),
    )
    return [primary], out.reset_index(drop=True)


def resolve_company_series(
    facts: pd.DataFrame,
    concepts: Sequence[str] | None = None,
    *,
    concept_map: Mapping[str, Sequence[str]] | None = None,
    merge_tol: float = 0.01,
) -> pd.DataFrame:
    """Resuelve varios conceptos lógicos de una empresa con tags estables.

    Aplica `resolve_company_tag` por concepto y devuelve la unión con la columna
    `concept` ya renombrada al vocabulario lógico (el tag crudo queda en
    `concept_used`). `total_debt` no pasa por la cascada: se reconcilia con
    `reconcile_total_debt` (sec_edgar.md §6.3). Los conceptos sin dato en la
    empresa se omiten con un WARNING (el fallo duro por concepto es decisión del
    consumidor; un panel cross-section tolera huecos por empresa).
    """
    cmap = dict(concept_map or SEC_CONCEPT_MAP)
    wanted = list(concepts or cmap.keys())
    pieces: list[pd.DataFrame] = []
    for logical in wanted:
        if logical == "total_debt":
            try:
                debt = reconcile_total_debt(facts)
            except InsufficientHistory:
                logger.warning("empresa sin datos de deuda: se omite total_debt")
                continue
            pieces.append(debt)
            continue
        try:
            _tags, rows = resolve_company_tag(
                facts, logical, concept_map=cmap, merge_tol=merge_tol
            )
        except InsufficientHistory:
            logger.warning("empresa sin tag para %r: se omite", logical)
            continue
        rows = rows.copy()
        rows["concept"] = logical
        pieces.append(rows)
    if not pieces:
        msg = f"la empresa no reporta ninguno de los conceptos pedidos: {wanted}"
        raise InsufficientHistory(msg)
    out = pd.concat(pieces, ignore_index=True)
    return out.sort_values(["concept", "end", "filed"], kind="mergesort").reset_index(
        drop=True
    )


# ---------------------------------------------------------------------------
# 2. Reconciliación de deuda total (sec_edgar.md §6.3)
# ---------------------------------------------------------------------------

_DEBT_COMPONENT_TAGS: tuple[str, ...] = (
    "LongTermDebtNoncurrent",
    "LongTermDebtCurrent",
    "ShortTermBorrowings",
    "DebtCurrent",
)
_DEBT_AGGREGATE_TAGS: tuple[str, ...] = (
    "LongTermDebtAndFinanceLeaseObligations",
    "LongTermDebtAndCapitalLeaseObligations",
    "LongTermDebt",
)


def reconcile_total_debt(facts: pd.DataFrame) -> pd.DataFrame:
    """Deuda total por `(period_end, vintage)`: componentes primero, agregado después.

    Regla (sec_edgar.md §6.3): ``total_debt = LongTermDebtNoncurrent +
    LongTermDebtCurrent + ShortTermBorrowings`` (más `DebtCurrent` si sustituye a
    los dos últimos), y el agregado (`LongTermDebtAndFinanceLeaseObligations`…)
    **solo** cuando faltan los componentes. Tomar `LongTermDebtNoncurrent` como
    "deuda total" omite la parte corriente y subestima sistemáticamente el
    apalancamiento — no es ruido, es sesgo con signo.

    La suma se hace **dentro de cada vintage** (`filed`): mezclar un componente
    reexpresado con otro original fabricaría una cifra que nunca existió. Los
    componentes ausentes en un vintage se tratan como 0 solo si al menos el
    tramo no corriente está presente; si no hay componentes se cae al agregado.
    `concept_used` documenta la vía usada (``sum(...)`` o el tag del agregado).
    """
    pool = _quarterly_slice(facts)
    inst = pool[pool["is_instant"]]
    comp = inst[inst["concept"].isin(_DEBT_COMPONENT_TAGS)]
    aggr = inst[inst["concept"].isin(_DEBT_AGGREGATE_TAGS)]
    if len(comp) == 0 and len(aggr) == 0:
        msg = "sin tags de deuda (componentes ni agregados) en los hechos de la empresa"
        raise InsufficientHistory(msg)

    rows: list[dict[str, Any]] = []
    if len(comp):
        # DebtCurrent duplica ShortTermBorrowings+LongTermDebtCurrent en algunos
        # filers; si coexisten se prefiere el desglose fino.
        for (end, filed), grp in comp.groupby(["end", "filed"]):
            by_tag = grp.groupby("concept")["value"].last()
            if "LongTermDebtNoncurrent" not in by_tag.index:
                continue
            fine = {"LongTermDebtCurrent", "ShortTermBorrowings"} & set(by_tag.index)
            current = (
                sum(by_tag[t] for t in fine)
                if fine
                else float(by_tag.get("DebtCurrent", 0.0))
            )
            used = ["LongTermDebtNoncurrent", *sorted(fine)] if fine else [
                "LongTermDebtNoncurrent",
                *(["DebtCurrent"] if "DebtCurrent" in by_tag.index else []),
            ]
            template = grp.iloc[-1]
            rows.append(
                {
                    **{c: template[c] for c in grp.columns},
                    "value": float(by_tag["LongTermDebtNoncurrent"]) + float(current),
                    "concept": "total_debt",
                    "concept_used": "sum(" + "+".join(used) + ")",
                    "tag_switch": False,
                }
            )
    covered = {(r["end"], r["filed"]) for r in rows}
    for (end, filed), grp in aggr.groupby(["end", "filed"]):
        if (end, filed) in covered:
            continue
        for tag in _DEBT_AGGREGATE_TAGS:
            hit = grp[grp["concept"] == tag]
            if len(hit):
                template = hit.iloc[-1]
                rows.append(
                    {
                        **{c: template[c] for c in grp.columns},
                        "value": float(template["value"]),
                        "concept": "total_debt",
                        "concept_used": tag,
                        "tag_switch": False,
                    }
                )
                break
    if not rows:
        msg = "los tags de deuda existen pero ningún vintage es reconstruible"
        raise InsufficientHistory(msg)
    out = pd.DataFrame(rows)
    return out.sort_values(["end", "filed"], kind="mergesort").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Derivación del Q4 (sec_edgar.md §6.5)
# ---------------------------------------------------------------------------


def derive_q4(resolved: pd.DataFrame) -> pd.DataFrame:
    """Deriva el Q4 de los conceptos de flujo: ``Q4 = FY − (Q1+Q2+Q3)``.

    Muchas empresas etiquetan el cuarto trimestre solo dentro del 10-K, como
    cifra anual. Reglas implementadas (sec_edgar.md §6.5):

    1. **Mismo vintage**: los cuatro sumandos se toman *as-first-reported*
       (primer `filed` de cada periodo). Restar un anual reexpresado de
       trimestres originales mezcla ventanas y produce basura.
    2. **`available_at` del 10-K**: el Q4 sintético se conoce cuando se publica
       el anual — 45-90 días después del cierre —, así que
       ``available_at = max(available_at del FY, available_at del Q3)``.
    3. **Marca `is_derived=True`**: no es un hecho reportado y su error se
       acumula (suma de tres errores de trimestre más el del anual).
    4. Solo **conceptos de flujo** (`FLOW_CONCEPTS`): los saldos de balance del
       Q4 están en el 10-K como hecho instantáneo y restarlos sería un error.

    Espera la salida de `resolve_company_series` (columnas `concept`, `end`,
    `start`, `filed`, `available_at`, `value`, `period_kind`). Devuelve solo las
    filas Q4 derivadas; el llamante decide concatenarlas. Los años cuyo FY ya
    tiene un Q4 reportado explícito no se derivan.
    """
    need = {"concept", "end", "filed", "value", "period_kind", "available_at"}
    missing = need - set(resolved.columns)
    if missing:
        msg = f"derive_q4 necesita columnas {sorted(missing)}: usa resolve_company_series"
        raise DataQualityError(msg)
    flows = resolved[resolved["concept"].isin(FLOW_CONCEPTS)]
    rows: list[dict[str, Any]] = []
    for concept, grp in flows.groupby("concept"):
        # Vintage original de cada periodo (regla 1).
        first = (
            grp.sort_values(["end", "filed"], kind="mergesort")
            .groupby(["period_kind", "end"], as_index=False)
            .first()
        )
        quarters = first[first["period_kind"] == "quarter"].set_index("end")
        annuals = first[first["period_kind"] == "annual"]
        for annual in annuals.itertuples(index=False):
            fy_end = pd.Timestamp(annual.end)
            fy_start = pd.Timestamp(annual.start)
            in_year = quarters.loc[
                (quarters.index > fy_start) & (quarters.index <= fy_end)
            ]
            # ¿Ya hay Q4 reportado? (un trimestre que termina en el cierre anual)
            has_q4 = any(abs((fy_end - q_end).days) <= 5 for q_end in in_year.index)
            if has_q4:
                continue
            prior = in_year.loc[in_year.index < fy_end - pd.Timedelta(days=5)]
            if len(prior) != 3:
                continue  # sin los tres primeros trimestres no hay resta honesta
            q123 = float(prior["value"].sum())
            avail = max(
                pd.Timestamp(annual.available_at), pd.Timestamp(prior["available_at"].max())
            )
            template = {c: getattr(annual, c, None) for c in flows.columns}
            rows.append(
                {
                    **template,
                    "concept": concept,
                    "start": pd.Timestamp(prior.index.max()) + pd.Timedelta(days=1),
                    "end": fy_end,
                    "value": float(annual.value) - q123,
                    "filed": pd.Timestamp(annual.filed),
                    "available_at": avail,
                    "period_kind": "quarter",
                    "is_derived": True,
                }
            )
    if not rows:
        return flows.iloc[0:0].assign(is_derived=pd.Series(dtype=bool))
    return pd.DataFrame(rows).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Panel con vintage explícito (sec_edgar.md §11.1)
# ---------------------------------------------------------------------------


def _to_canonical(
    resolved: pd.DataFrame, ticker: Ticker, *, source: str
) -> pd.DataFrame:
    """Proyecta la salida de `resolve_company_series` a `CANONICAL_COLUMNS`."""
    out = pd.DataFrame(
        {
            "ticker": normalize_ticker(ticker),
            "concept": resolved["concept"].astype(str),
            "period_end": pd.to_datetime(resolved["end"]),
            "available_at": pd.to_datetime(resolved["available_at"]),
            "value": pd.to_numeric(resolved["value"], errors="coerce"),
            "fiscal_period": resolved.get("fiscal_period"),
            "unit": resolved.get("unit", pd.Series(dtype=object)),
            "form": resolved.get("form"),
            "accession": resolved.get("accession"),
            "is_restated": resolved.get(
                "is_restated", pd.Series(False, index=resolved.index)
            ).astype(bool),
            "concept_used": resolved.get("concept_used"),
            "is_derived": resolved.get(
                "is_derived", pd.Series(False, index=resolved.index)
            )
            .fillna(False)
            .astype(bool),
            "source": source,
        }
    )
    if out["fiscal_period"].isna().all():
        pe = pd.DatetimeIndex(out["period_end"])
        out["fiscal_period"] = [
            f"{ts.year}Q{(ts.month - 1) // 3 + 1}" for ts in pe
        ]
    return out.loc[:, list(CANONICAL_COLUMNS)]


def fundamentals_panel(
    facts: pd.DataFrame,
    *,
    as_of: date | datetime | None = None,
    vintage: Literal["original", "pit", "latest"] = "pit",
) -> pd.DataFrame:
    """Selecciona un vintage por `(ticker, concept, period_end)` de la tabla canónica.

    ============  ==========================================================
    `vintage`     selección (sec_edgar.md §5.5 y §11.1)
    ============  ==========================================================
    ``original``  primer `available_at` — *as-originally-reported*, lo que el
                  gestor vio. La serie del backtest honesto.
    ``pit``       último `available_at` ≤ `as_of` — igual de honesto y más
                  realista: incorpora reexpresiones **ya publicadas** en
                  `as_of`. Exige `as_of`.
    ``latest``    último vintage sin filtro. **Look-ahead deliberado**: solo
                  para cuantificar cuánto rendimiento de un factor viene de
                  las reexpresiones (equivale a lo que sirve `frames`).
    ============  ==========================================================

    El desempate con `available_at` idéntico es por `accession` (determinista,
    sec_edgar.md §5.5). Las claves cuyo primer vintage es posterior a `as_of`
    no aparecen: no existían.
    """
    need = {"ticker", "concept", "period_end", "available_at", "value"}
    missing = need - set(facts.columns)
    if missing:
        msg = f"fundamentals_panel necesita columnas {sorted(missing)}"
        raise DataQualityError(msg)
    if len(facts) == 0:
        msg = "tabla de fundamentales vacía: nada que seleccionar"
        raise InsufficientHistory(msg)
    if vintage not in {"original", "pit", "latest"}:
        msg = f"vintage desconocido: {vintage!r} (esperado original|pit|latest)"
        raise DataQualityError(msg)
    work = facts.copy()
    work["available_at"] = pd.to_datetime(work["available_at"])
    work["_accn"] = work.get("accession", pd.Series("", index=work.index)).fillna("")
    work = work.sort_values(
        ["ticker", "concept", "period_end", "available_at", "_accn"],
        kind="mergesort",
    )
    key = ["ticker", "concept", "period_end"]
    if vintage == "original":
        out = work.groupby(key, as_index=False, sort=True).first()
    elif vintage == "latest":
        logger.warning(
            "fundamentals_panel(vintage='latest') es look-ahead deliberado: "
            "solo para diagnóstico, jamás para señales"
        )
        out = work.groupby(key, as_index=False, sort=True).last()
    else:
        if as_of is None:
            msg = "vintage='pit' exige `as_of`: sin fecha de corte no hay point-in-time"
            raise DataQualityError(msg)
        cut = pd.Timestamp(as_of)
        if cut.tzinfo is not None:
            cut = cut.tz_convert("UTC").tz_localize(None)
        if isinstance(as_of, date) and not isinstance(as_of, datetime):
            # Un `date` significa "al cierre de ese día": corte inclusivo.
            cut = cut + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        visible = work[work["available_at"] <= cut]
        if len(visible) == 0:
            msg = f"ningún fundamental era conocible en {as_of}: histórico insuficiente"
            raise InsufficientHistory(msg)
        out = visible.groupby(key, as_index=False, sort=True).last()
    out = out.drop(columns=["_accn"])
    out.attrs["vintage"] = vintage
    out.attrs["as_of"] = None if as_of is None else pd.Timestamp(as_of).isoformat()
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. Proveedores
# ---------------------------------------------------------------------------


@runtime_checkable
class FundamentalsSource(Protocol):
    """Contrato mínimo de una fuente de fundamentales."""

    name: str

    def available(self) -> bool: ...

    def quarterly_facts(
        self, ticker: Ticker, *, cik: CIK | None = None
    ) -> pd.DataFrame:
        """Tabla canónica (`CANONICAL_COLUMNS`) de la empresa, con
        `df.attrs["pit_quality"]` declarando la honestidad temporal."""
        ...


class EdgarFundamentals(BaseProvider):
    """Fundamentales desde EDGAR: la fuente preferida (vintages completos).

    Envuelve `EdgarProvider.company_facts` y aplica la cascada estable por
    empresa, la reconciliación de deuda y la derivación de Q4. `available_at`
    procede de `filed` con el corte administrativo (cota superior segura); si se
    pasa `use_acceptance=True` se afina con el `acceptanceDateTime` de
    `submissions` (una petición extra por empresa).

    `pit_quality="vintages"`: conserva todas las versiones y marca
    `is_restated`; es la única de las tres fuentes apta para medir revisiones.
    """

    name = "sec"
    kinds = (DataKind.FUNDAMENTALS.value,)

    def __init__(
        self,
        *,
        edgar: EdgarProvider | None = None,
        settings: Any = None,
        cache: DiskCache | None = None,
        concepts: Sequence[str] | None = None,
        derive_fourth_quarter: bool = True,
        use_acceptance: bool = False,
    ) -> None:
        super().__init__(settings=settings)
        self.edgar = edgar or EdgarProvider(settings=self.settings, cache=cache)
        self.concepts = list(concepts) if concepts is not None else None
        self.derive_fourth_quarter = bool(derive_fourth_quarter)
        self.use_acceptance = bool(use_acceptance)

    def available(self) -> bool:
        return self.edgar.available()

    def quarterly_facts(
        self, ticker: Ticker, *, cik: CIK | None = None
    ) -> pd.DataFrame:
        resolved_cik = cik or self.edgar.cik_for(ticker)
        if resolved_cik is None:
            raise ProviderUnavailable(
                self.name,
                f"CIK no resoluble para {ticker!r} con el snapshot actual de la SEC "
                "(¿ticker deslistado? el mapeo histórico exige los índices "
                "trimestrales, sec_edgar.md §11.2)",
            )
        raw = self.edgar.company_facts(resolved_cik)
        resolved = resolve_company_series(raw, self.concepts)
        if self.use_acceptance:
            subs = self.edgar.submissions(resolved_cik)
            acc = (
                subs.dropna(subset=["accepted_at"])
                .drop_duplicates(subset=["accession"])
                .set_index("accession")["accepted_at"]
            )
            exact = resolved["accession"].map(acc)
            resolved = resolved.assign(
                available_at=exact.fillna(resolved["available_at"])
            )
        if self.derive_fourth_quarter:
            q4 = derive_q4(resolved)
            if len(q4):
                resolved = pd.concat([resolved, q4], ignore_index=True)
        keep = resolved[
            resolved["period_kind"].isin(("quarter", "instant"))
        ].reset_index(drop=True)
        if len(keep) == 0:
            msg = f"{ticker}: sin hechos trimestrales tras el filtro de duración"
            raise InsufficientHistory(msg)
        out = _to_canonical(keep, ticker, source="sec:companyfacts")
        out.attrs["pit_quality"] = "vintages"
        return out


class FMPFundamentals(BaseProvider):
    """Fundamentales de Financial Modeling Prep (respaldo, sin vintages).

    Endpoints ``/api/v3/{income-statement,balance-sheet-statement,
    cash-flow-statement}/{ticker}?period=quarter``. `available_at` sale de
    `acceptedDate` (instante ET del filing en la SEC, se convierte con la misma
    política que EDGAR) o, en su defecto, de `fillingDate` con el corte de las
    17:30 ET.

    **Limitación PIT documentada** (`pit_quality="filing-date"`): FMP sirve la
    última versión de cada cifra con la fecha del filing *original*; una
    reexpresión sobrescribe el valor sin cambiar la fecha, y no hay forma de
    detectarla desde aquí (`is_restated=False` siempre). Apto como respaldo y
    para el histórico pre-XBRL; inapto para medir revisiones contables.
    """

    name = "fmp"
    kinds = (DataKind.FUNDAMENTALS.value,)
    base_url = "https://financialmodelingprep.com/api/v3"

    _FIELD_MAP: dict[str, dict[str, str]] = {
        "income-statement": {
            "revenue": "revenue",
            "costOfRevenue": "cogs",
            "grossProfit": "gross_profit",
            "operatingIncome": "operating_income",
            "netIncome": "net_income",
            "epsdiluted": "eps_diluted",
            "eps": "eps_basic",
            "weightedAverageShsOutDil": "shares_diluted",
        },
        "balance-sheet-statement": {
            "totalAssets": "total_assets",
            "totalStockholdersEquity": "total_equity",
            "totalDebt": "total_debt",
            "cashAndCashEquivalents": "cash",
        },
        "cash-flow-statement": {
            "operatingCashFlow": "cfo",
            "capitalExpenditure": "capex",
        },
    }

    def __init__(
        self,
        *,
        settings: Any = None,
        http: HttpClient | None = None,
        transport: Transport | None = None,
        cache: DiskCache | None = None,
        limit: int = 400,
    ) -> None:
        super().__init__(settings=settings, http=http)
        if http is None and transport is not None:
            self._http = HttpClient(
                self.name,
                transport=transport,
                rate=rate_limit_for(self.name, self.settings),
                settings=self.settings,
            )
        self.cache = cache
        self.limit = int(limit)

    def _get(self, statement: str, ticker: Ticker) -> list[dict[str, Any]]:
        key = self.settings.env("FMP_API_KEY")
        if not key:
            raise ProviderUnavailable(
                self.name, "falta la clave de API", missing_env=["FMP_API_KEY"]
            )
        url = f"{self.base_url}/{statement}/{normalize_ticker(ticker)}"
        params = {"period": "quarter", "limit": self.limit, "apikey": key}

        def _load() -> list[dict[str, Any]]:
            payload = self.http.get_json(url, params=params)
            if not isinstance(payload, list):
                msg = f"respuesta inesperada de FMP {statement}: {type(payload).__name__}"
                raise DataQualityError(msg)
            return payload

        if self.cache is None:
            return _load()
        return self.cache.fetch(
            DataKind.FUNDAMENTALS.value,
            self.name,
            {"statement": statement, "ticker": normalize_ticker(ticker)},
            _load,
            ttl=86_400.0,
        )

    def quarterly_facts(
        self, ticker: Ticker, *, cik: CIK | None = None
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for statement, fields in self._FIELD_MAP.items():
            for entry in self._get(statement, ticker):
                accepted = entry.get("acceptedDate")
                filled = entry.get("fillingDate") or entry.get("filingDate")
                if accepted:
                    available = parse_acceptance_datetime(accepted)
                elif filled:
                    available = filed_available_at(filled)
                else:
                    # Sin fecha de filing el hecho no es fechable: se descarta
                    # explícitamente en vez de inventar available_at=period_end.
                    continue
                for src, logical in fields.items():
                    value = entry.get(src)
                    if value is None:
                        continue
                    rows.append(
                        {
                            "ticker": normalize_ticker(ticker),
                            "concept": logical,
                            "period_end": pd.Timestamp(entry["date"]),
                            "available_at": pd.Timestamp(available),
                            "value": float(value),
                            "fiscal_period": entry.get("period"),
                            "unit": "USD",
                            "form": None,
                            "accession": None,
                            "is_restated": False,
                            "concept_used": f"fmp:{src}",
                            "is_derived": False,
                            "source": f"fmp:{statement}",
                        }
                    )
        if not rows:
            msg = f"FMP no devuelve estados trimestrales para {ticker!r}"
            raise InsufficientHistory(msg)
        out = pd.DataFrame(rows).loc[:, list(CANONICAL_COLUMNS)]
        out = out.sort_values(
            ["concept", "period_end"], kind="mergesort"
        ).reset_index(drop=True)
        out.attrs["pit_quality"] = "filing-date"
        return out


class EODHDFundamentals(BaseProvider):
    """Fundamentales de EODHD (último respaldo; valores reexpresados).

    Endpoint ``/api/fundamentals/{TICKER}.US`` con los bloques
    ``Financials::{Income_Statement,Balance_Sheet,Cash_Flow}::quarterly``.
    `available_at` sale de `filing_date` con el corte de las 17:30 ET.

    **Limitación PIT documentada** (`pit_quality="restated"`): EODHD sobrescribe
    con la última reexpresión y no conserva la cifra original; es la peor calidad
    temporal de las tres fuentes y solo debe usarse como respaldo de cobertura o
    para diagnóstico. Un backtest construido solo con esta fuente hereda el
    sesgo de reexpresión (look-ahead contable).
    """

    name = "eodhd"
    kinds = (DataKind.FUNDAMENTALS.value,)
    base_url = "https://eodhd.com/api"

    _FIELD_MAP: dict[str, dict[str, str]] = {
        "Income_Statement": {
            "totalRevenue": "revenue",
            "costOfRevenue": "cogs",
            "grossProfit": "gross_profit",
            "operatingIncome": "operating_income",
            "netIncome": "net_income",
        },
        "Balance_Sheet": {
            "totalAssets": "total_assets",
            "totalStockholderEquity": "total_equity",
            "shortLongTermDebtTotal": "total_debt",
            "cashAndEquivalents": "cash",
        },
        "Cash_Flow": {
            "totalCashFromOperatingActivities": "cfo",
            "capitalExpenditures": "capex",
        },
    }

    def __init__(
        self,
        *,
        settings: Any = None,
        http: HttpClient | None = None,
        transport: Transport | None = None,
        cache: DiskCache | None = None,
    ) -> None:
        super().__init__(settings=settings, http=http)
        if http is None and transport is not None:
            self._http = HttpClient(
                self.name,
                transport=transport,
                rate=rate_limit_for(self.name, self.settings),
                settings=self.settings,
            )
        self.cache = cache

    def _payload(self, ticker: Ticker) -> Mapping[str, Any]:
        token = self.settings.env("EODHD_API_KEY")
        if not token:
            raise ProviderUnavailable(
                self.name, "falta la clave de API", missing_env=["EODHD_API_KEY"]
            )
        symbol = f"{normalize_ticker(ticker).replace('.', '-')}.US"
        url = f"{self.base_url}/fundamentals/{symbol}"

        def _load() -> Mapping[str, Any]:
            payload = self.http.get_json(url, params={"api_token": token, "fmt": "json"})
            if not isinstance(payload, Mapping):
                msg = f"respuesta inesperada de EODHD: {type(payload).__name__}"
                raise DataQualityError(msg)
            return payload

        if self.cache is None:
            return _load()
        return self.cache.fetch(
            DataKind.FUNDAMENTALS.value,
            self.name,
            {"symbol": symbol},
            _load,
            ttl=86_400.0,
        )

    def quarterly_facts(
        self, ticker: Ticker, *, cik: CIK | None = None
    ) -> pd.DataFrame:
        payload = self._payload(ticker)
        financials = payload.get("Financials") or {}
        rows: list[dict[str, Any]] = []
        for block, fields in self._FIELD_MAP.items():
            quarterly = (financials.get(block) or {}).get("quarterly") or {}
            for period_end, entry in quarterly.items():
                filing = entry.get("filing_date")
                if not filing:
                    continue  # sin fecha de filing no hay PIT: se descarta
                available = filed_available_at(filing)
                for src, logical in fields.items():
                    value = entry.get(src)
                    if value in (None, "", "None"):
                        continue
                    rows.append(
                        {
                            "ticker": normalize_ticker(ticker),
                            "concept": logical,
                            "period_end": pd.Timestamp(period_end),
                            "available_at": pd.Timestamp(available),
                            "value": float(value),
                            "fiscal_period": None,
                            "unit": str(entry.get("currency_symbol") or "USD"),
                            "form": None,
                            "accession": None,
                            "is_restated": False,
                            "concept_used": f"eodhd:{src}",
                            "is_derived": False,
                            "source": f"eodhd:{block}",
                        }
                    )
        if not rows:
            msg = f"EODHD no devuelve fundamentales trimestrales para {ticker!r}"
            raise InsufficientHistory(msg)
        out = pd.DataFrame(rows).loc[:, list(CANONICAL_COLUMNS)]
        out = out.sort_values(
            ["concept", "period_end"], kind="mergesort"
        ).reset_index(drop=True)
        out.attrs["pit_quality"] = "restated"
        return out


# ---------------------------------------------------------------------------
# 6. Servicio combinado
# ---------------------------------------------------------------------------

_DEFAULT_PRIORITIES: tuple[tuple[str, int], ...] = (("sec", 100), ("fmp", 50), ("eodhd", 40))


def build_default_registry(
    *,
    settings: Any = None,
    cache: DiskCache | None = None,
    registry: ProviderRegistry | None = None,
) -> ProviderRegistry:
    """Registra las tres fuentes de fundamentales con la preferencia del repo.

    EDGAR (100) > FMP (50) > EODHD (40). La preferencia no es negociable a la
    ligera: EDGAR es la única con vintages (`pit_quality="vintages"`), y elegir
    otra fuente cuando EDGAR está disponible degradaría silenciosamente la
    honestidad temporal del panel.
    """
    reg = registry or ProviderRegistry(settings=settings)
    providers: list[BaseProvider] = [
        EdgarFundamentals(settings=settings, cache=cache),
        FMPFundamentals(settings=settings, cache=cache),
        EODHDFundamentals(settings=settings, cache=cache),
    ]
    by_name = {p.name: p for p in providers}
    for name, priority in _DEFAULT_PRIORITIES:
        reg.register(DataKind.FUNDAMENTALS.value, by_name[name], priority)
    return reg


class FundamentalsService:
    """Fachada de fundamentales multi-fuente con preferencia EDGAR.

    `facts(ticker)` resuelve la mejor fuente disponible vía
    `ProviderRegistry.call` (fallback en cadena con cuarentena) y devuelve la
    tabla canónica; `panel(tickers, ...)` concatena varias empresas y aplica
    `fundamentals_panel` con el vintage pedido.

    El registro anota cada intento (`registry.attempts`), de modo que queda
    auditado *de qué fuente* salió cada empresa: si EDGAR estuvo caído y el
    panel se llenó con EODHD, eso debe aparecer en el tearsheet, no descubrirse
    meses después.
    """

    def __init__(
        self,
        registry: ProviderRegistry | None = None,
        *,
        settings: Any = None,
        cache: DiskCache | None = None,
    ) -> None:
        self.registry = registry or build_default_registry(
            settings=settings, cache=cache
        )

    def facts(self, ticker: Ticker, *, cik: CIK | None = None) -> pd.DataFrame:
        """Tabla canónica de la empresa desde la mejor fuente disponible."""
        return self.registry.call(
            DataKind.FUNDAMENTALS.value,
            lambda p: p.quarterly_facts(ticker, cik=cik),  # type: ignore[attr-defined]
            description=f"fundamentals:{ticker}",
        )

    def panel(
        self,
        tickers: Sequence[Ticker],
        *,
        as_of: date | datetime | None = None,
        vintage: Literal["original", "pit", "latest"] = "pit",
        cik_map: Mapping[Ticker, CIK] | None = None,
    ) -> pd.DataFrame:
        """Panel multi-empresa con un vintage por `(ticker, concept, period_end)`.

        Las empresas irresolubles en **todas** las fuentes se acumulan y, si son
        todas, se lanza `ProviderUnavailable`; si son algunas, se registra un
        WARNING con la lista (el consumidor cross-section decide si el hueco es
        tolerable). `df.attrs["missing"]` conserva la lista y
        `df.attrs["pit_quality_by_ticker"]` la calidad por empresa.
        """
        if not tickers:
            msg = "panel de fundamentales sin tickers"
            raise DataQualityError(msg)
        pieces: list[pd.DataFrame] = []
        missing: list[str] = []
        quality: dict[str, str] = {}
        ciks = dict(cik_map or {})
        for ticker in tickers:
            t = normalize_ticker(ticker)
            try:
                table = self.facts(t, cik=ciks.get(t))
            except (ProviderUnavailable, InsufficientHistory) as exc:
                logger.warning("sin fundamentales para %s: %s", t, exc)
                missing.append(t)
                continue
            quality[t] = str(table.attrs.get("pit_quality", "unknown"))
            pieces.append(table)
        if not pieces:
            raise ProviderUnavailable(
                "fundamentals",
                f"ninguna fuente pudo servir fundamentales para {list(tickers)}",
            )
        combined = pd.concat(pieces, ignore_index=True)
        out = fundamentals_panel(combined, as_of=as_of, vintage=vintage)
        out.attrs["missing"] = missing
        out.attrs["pit_quality_by_ticker"] = quality
        out.attrs["vintage"] = vintage
        return out
