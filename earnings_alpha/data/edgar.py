"""Adaptador de SEC EDGAR: identidad, submissions, XBRL, 8-K item 2.02 y Form 4.

Implementa la especificación de `docs/research/sec_edgar.md` contra el contrato de
`docs/ARCHITECTURE.md` (§3.3 data.base). EDGAR es la única fuente **gratuita y
point-in-time** del proyecto: `companyfacts` conserva todos los vintages de cada
cifra (lo que normalmente exige Compustat Point-in-Time) y el `acceptanceDateTime`
del 8-K con item 2.02 da el instante del anuncio de resultados desde 2004.

Decisiones de diseño vinculantes (con la sección del informe que las justifica):

- **`available_at` jamás sale de `filingDate` ni de `period_end`** (sec_edgar.md
  §4.4). Para hechos XBRL la clave PIT es `filed`; como `filed` es una fecha sin
  hora, se convierte a instante con el **corte administrativo de EDGAR**: un filing
  con `filed = D` fue aceptado como muy tarde a las 17:30 ET de D (los Formularios
  3/4/5, a las 22:00 ET). Usar ese corte como `available_at` es una **cota
  superior** del instante real: el error posible es perder unas horas de señal,
  nunca mirar el futuro. Cuando se dispone del `acceptanceDateTime` exacto (vía
  `submissions`), se usa ese instante.
- **`acceptanceDateTime` se interpreta como hora de pared de Nueva York** y se
  convierte a UTC con `pit.calendar.eastern_to_utc` (sec_edgar.md §4.4 y pregunta
  abierta nº 1: la cadena puede traer un sufijo ``Z`` engañoso; un desplazamiento
  explícito distinto de ``Z`` sí se respeta).
- **`fy`/`fp` describen el filing, no la observación** (trampa nº 7): el periodo de
  un hecho se determina exclusivamente por `(start, end)`, y `fiscal_period` se
  deriva del `period_end` del hecho, jamás de `fy`/`fp`.
- **Vintages**: se conservan todas las versiones de un `(concept, unit, start,
  end)` y se marca `is_restated` cuando el valor difiere del primero presentado
  (§5.5). La republicación de un comparativo con el mismo valor **no** es una
  reexpresión.
- **Un 403 de la SEC puede ser limitación de tasa** (trampa nº 2): la SEC no
  devuelve 429 sino 403 con un cuerpo HTML que contiene *"declare your traffic"* /
  *"Request Rate Threshold"*. Este adaptador inspecciona el cuerpo y lanza
  `RateLimited`; cualquier otro 403 es `ProviderUnavailable`.
- **`frames` NO es point-in-time** (trampa nº 6): devuelve el último valor
  presentado (la reexpresión). Se expone solo para exploración y control de
  calidad; el panel de factores debe construirse con `company_facts`.

Nota legal (contrato §3.5): el Form 4 es un documento público ya publicado; su uso
aquí es análisis de información pública, no acceso a información privilegiada.

Referencias académicas:
- Lakonishok y Lee (2001), *Are Insider Trades Informative?*: las compras abiertas
  de insiders predicen retornos; las ventas son mucho más ruidosas.
- Cohen, Malloy y Pomorski (2012), *Decoding Inside Information*: distinguir
  transacciones "rutinarias" de "oportunistas" es lo que da poder predictivo; de
  ahí el filtro de códigos P/S frente a A/F/M (trampa nº 9).
- Foster, Olsen y Shevlin (1984) y Bernard y Thomas (1989): la ventana de evento se
  ancla a la primera sesión negociable del anuncio, que aquí procede del
  `acceptanceDateTime` del 8-K 2.02.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

import numpy as np
import pandas as pd

from earnings_alpha.data.base import (
    DEFAULT_USER_AGENT,
    BaseProvider,
    Clock,
    DataKind,
    HttpClient,
    HttpResponse,
    HttpStatusError,
    Transport,
    rate_limit_for,
)
from earnings_alpha.data.cache import DiskCache
from earnings_alpha.errors import (
    DataQualityError,
    InsufficientHistory,
    ProviderUnavailable,
    RateLimited,
)
from earnings_alpha.pit.asof import classify_session, tradable_date
from earnings_alpha.pit.calendar import (
    TradingCalendar,
    eastern_offsets_for,
    eastern_to_utc,
    get_calendar,
)
from earnings_alpha.types import (
    CIK,
    EarningsEvent,
    Session,
    Ticker,
    normalize_cik,
    normalize_ticker,
)

__all__ = [  # noqa: RUF022 - orden temático, no alfabético
    # constantes y política PIT
    "SEC_WWW", "SEC_DATA", "SEC_FTS",
    "FILING_DAY_CUTOFF_ET", "OWNERSHIP_DAY_CUTOFF_ET", "OWNERSHIP_FORMS",
    "RATE_LIMIT_BODY_MARKERS",
    "SEC_CONCEPT_MAP",
    "QUARTER_SPAN_DAYS", "ANNUAL_SPAN_DAYS",
    # utilidades puras (testables sin red)
    "parse_acceptance_datetime", "filed_available_at", "eastern_wall_to_utc",
    "company_tickers_to_frame", "submissions_to_frame", "companyfacts_to_frame",
    "mark_vintages", "classify_periods", "facts_to_fundamental_facts",
    "detect_earnings_8k", "frame_to_earnings_events",
    # Form 4
    "FORM4_TRANSACTION_CODES", "DISCRETIONARY_BUY_CODES",
    "DISCRETIONARY_SELL_CODES", "ROUTINE_CODES",
    "parse_form4", "form4_to_frame", "insider_net_buys", "form4_code_distribution",
    # proveedor
    "EdgarProvider",
]

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

SEC_WWW = "https://www.sec.gov"
"""Sirve `Archives/*` y `files/company_tickers*.json`. Estricto con el User-Agent."""

SEC_DATA = "https://data.sec.gov"
"""Sirve `submissions/*` y `api/xbrl/*`. Se le envía igualmente el UA declarado."""

SEC_FTS = "https://efts.sec.gov"
"""Búsqueda a texto completo (backend Elasticsearch, no documentado pero estable)."""

FILING_DAY_CUTOFF_ET: time = time(17, 30)
"""Corte administrativo general de EDGAR: lo aceptado después de las 17:30 ET
recibe `filingDate` del siguiente día hábil (sec_edgar.md §4.4). Por tanto, para un
filing con `filed = D`, las 17:30 ET de D son una cota superior del instante real
de aceptación: usarla como `available_at` nunca introduce look-ahead."""

OWNERSHIP_DAY_CUTOFF_ET: time = time(22, 0)
"""Los Formularios 3/4/5 conservan el `filingDate` del día hasta las 22:00 ET
(sec_edgar.md §4.4 y §8.1), no las 17:30. Cota superior específica para ellos."""

OWNERSHIP_FORMS: frozenset[str] = frozenset({"3", "4", "5", "3/A", "4/A", "5/A"})

RATE_LIMIT_BODY_MARKERS: tuple[str, ...] = (
    "declare your traffic",
    "request rate threshold",
    "rate threshold exceeded",
    "undeclared automated tool",
)
"""Frases del cuerpo HTML con que la SEC responde a un exceso de tasa. La SEC no
devuelve 429: devuelve **403 con este cuerpo** (trampa nº 2), y hay que
distinguirlo de un 403 por User-Agent rechazado, que no se arregla reintentando."""

QUARTER_SPAN_DAYS: tuple[int, int] = (80, 100)
"""Duración admisible de un trimestre fiscal (~91 días; los ejercicios de 52/53
semanas oscilan entre 84 y 98 días, sec_edgar.md §6.4 y §6.6)."""

ANNUAL_SPAN_DAYS: tuple[int, int] = (335, 395)
"""Duración admisible de un ejercicio anual (365 ± 30 días, sec_edgar.md §6.4)."""

# TTLs por tipo de recurso (sec_edgar.md §2.6): la mutabilidad del recurso decide.
TTL_SUBMISSIONS_S = 86_400.0
TTL_COMPANYFACTS_S = 86_400.0
TTL_FRAMES_S = 7 * 86_400.0
TTL_TICKERS_S = 86_400.0
TTL_FTS_S = 86_400.0

# ---------------------------------------------------------------------------
# Mapa de sinónimos us-gaap (sec_edgar.md §6.2)
# ---------------------------------------------------------------------------

SEC_CONCEPT_MAP: dict[str, list[str]] = {
    # La prioridad dentro de cada lista está documentada: primero el tag moderno o
    # el agregado canónico, después variantes históricas o sectoriales. La cascada
    # se resuelve POR EMPRESA y de forma estable en toda su historia (trampa nº 8);
    # ver `fundamentals.resolve_company_series`.
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",  # ASC 606 (2018→)
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "Revenues",                                             # genérico clásico
        "SalesRevenueNet",                                      # pre-ASC 606
        "SalesRevenueGoodsNet",
        "SalesRevenueServicesNet",
        "RevenueFromContractWithCustomer",
        "TotalRevenues",
        "TotalRevenuesAndGains",                                # financieras
    ],
    "cogs": ["CostOfGoodsAndServicesSold", "CostOfRevenue", "CostOfGoodsSold"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": [
        # `NetIncomeLoss` es el atribuible a la matriz (el correcto para EPS y
        # ROE); `ProfitLoss` incluye minoritarios. NO son intercambiables
        # (sec_edgar.md §6.3): el fallback solo se usa si el primero falta en toda
        # la historia de la empresa, y queda registrado en `concept_used`.
        "NetIncomeLoss",
        "ProfitLoss",
    ],
    "eps_diluted": ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"],
    "eps_basic": ["EarningsPerShareBasic", "EarningsPerShareBasicAndDiluted"],
    "shares_diluted": [
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageDilutedSharesOutstanding",
    ],
    "total_assets": ["Assets"],
    "total_equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "total_debt": [
        # ⚠ Esta lista mezcla agregados y componentes a propósito, pero NO debe
        # recorrerse como cascada simple: tomar `LongTermDebtNoncurrent` como
        # "deuda total" omite la parte corriente y subestima el apalancamiento
        # (sec_edgar.md §6.3). La reconciliación correcta —sumar corriente + no
        # corriente y caer al agregado solo si faltan los componentes— está en
        # `fundamentals.reconcile_total_debt`.
        "LongTermDebtAndFinanceLeaseObligations",
        "LongTermDebtAndCapitalLeaseObligations",
        "LongTermDebt",
        "LongTermDebtNoncurrent",
        "LongTermDebtCurrent",
        "DebtCurrent",
        "ShortTermBorrowings",
    ],
    "cash": [
        # El primero es el agregado operativo común; el resto evita que bancos y
        # filers con caja restringida queden sin dato (sec_edgar.md §6.2).
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "Cash",
        "CashAndDueFromBanks",
    ],
    "cfo": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
        "PaymentsToAcquirePropertyPlantAndEquipmentAndIntangibleAssets",
    ],
}

# ---------------------------------------------------------------------------
# Conversión de instantes
# ---------------------------------------------------------------------------

_COMPACT_ACCEPTANCE = re.compile(r"^\d{14}$")


def parse_acceptance_datetime(raw: object, *, assume_eastern: bool = True) -> datetime:
    """Convierte un `acceptanceDateTime` de EDGAR a datetime **naive en UTC**.

    Política (sec_edgar.md §4.4 y pregunta abierta nº 1): la SEC sirve el instante
    de aceptación en hora del Este. La cadena puede llegar:

    - en ISO-8601 con sufijo ``Z`` — que en EDGAR es **engañoso**: el instante es
      hora de pared de Nueva York, no UTC. Con `assume_eastern=True` (defecto) se
      interpreta como ET y se convierte con `pit.calendar.eastern_to_utc`;
    - en ISO-8601 con desplazamiento explícito distinto de ``Z`` (p. ej.
      ``-05:00``): se respeta el desplazamiento declarado;
    - en formato compacto ``YYYYMMDDHHMMSS`` (cabecera SGML): ET de pared.

    Equivocarse aquí son 4-5 horas, que para un anuncio de las 16:05 ET cruzan la
    medianoche UTC y cambian el día del evento. Por eso jamás se deja que pandas
    interprete la cadena por su cuenta.
    """
    s = str(raw).strip()
    if not s or s.lower() in {"none", "nan", "nat"}:
        msg = f"acceptanceDateTime vacío o ilegible: {raw!r}"
        raise DataQualityError(msg)
    if _COMPACT_ACCEPTANCE.match(s):
        wall = datetime.strptime(s, "%Y%m%d%H%M%S")  # noqa: DTZ007 - ET de pared
        return eastern_to_utc(wall)
    text = s
    zulu = text.endswith(("Z", "z"))
    if zulu:
        text = text[:-1]
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        msg = f"acceptanceDateTime no ISO-8601: {raw!r}"
        raise DataQualityError(msg) from exc
    if parsed.tzinfo is not None:
        # Desplazamiento explícito no-Z: se confía en él.
        return parsed.astimezone(UTC).replace(tzinfo=None)
    if assume_eastern:
        return eastern_to_utc(parsed)
    return parsed  # ya interpretado como UTC por decisión del llamante


def filed_available_at(filed: date | str | pd.Timestamp, form: str | None = None) -> datetime:
    """Instante `available_at` (naive UTC) para un hecho cuyo único dato es `filed`.

    `filed` es la fecha administrativa del filing que publicó el hecho. EDGAR
    garantiza que un filing con `filed = D` fue aceptado **no más tarde** del corte
    de ese día: 17:30 ET en general, 22:00 ET para los Formularios 3/4/5
    (sec_edgar.md §4.4). Usar el corte como `available_at` es por tanto una cota
    superior del instante real de publicación: el sesgo va en la dirección segura
    (se puede perder señal intradía, nunca inventarla).
    """
    d = pd.Timestamp(filed).date()
    root = str(form or "").strip().upper()
    cutoff = OWNERSHIP_DAY_CUTOFF_ET if root in OWNERSHIP_FORMS else FILING_DAY_CUTOFF_ET
    return eastern_to_utc(datetime.combine(d, cutoff))


def eastern_wall_to_utc(local: pd.Series | pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Convierte un vector de horas de pared de Nueva York (naive) a UTC (naive).

    Versión vectorizada de `pit.calendar.eastern_to_utc`, con la misma estrategia
    de dos pasos: el desfase depende del propio instante, así que se estima en
    hora estándar y se reevalúa. Las horas administrativas de EDGAR nunca caen en
    la ventana ambigua del cambio de hora (02:00-03:00 de madrugada).
    """
    idx = pd.DatetimeIndex(pd.Series(local).to_numpy(dtype="datetime64[ns]"))
    guess_utc = idx + pd.Timedelta(hours=5)
    offsets = eastern_offsets_for(guess_utc)
    utc = pd.DatetimeIndex(idx.to_numpy() - offsets.to_numpy())
    offsets2 = eastern_offsets_for(utc)
    return pd.DatetimeIndex(idx.to_numpy() - offsets2.to_numpy())


# ---------------------------------------------------------------------------
# Identidad: company_tickers
# ---------------------------------------------------------------------------


def company_tickers_to_frame(payload: Mapping[str, Any]) -> pd.DataFrame:
    """Normaliza `company_tickers[_exchange].json` a un DataFrame.

    Admite las dos formas que sirve la SEC (sec_edgar.md §3.1-3.2):

    - columnar: ``{"fields": [...], "data": [[...], ...]}`` (la preferible, trae
      el mercado por instrumento);
    - objeto: ``{"0": {"cik_str": ..., "ticker": ..., "title": ...}, ...}``.

    Devuelve columnas `cik` (10 dígitos), `ticker` (normalizado: `BRK-B`→`BRK.B`),
    `name` y `exchange` (``None`` si la fuente no lo trae). El mapeo CIK→ticker es
    uno-a-muchos (`GOOGL`/`GOOG` comparten emisor): el DataFrame conserva una fila
    por instrumento, no por emisor, y **este fichero solo refleja el presente**:
    una empresa deslistada no aparece aunque sus filings sigan en EDGAR
    (sec_edgar.md §3.3a — sesgo de supervivencia si se usa como universo).
    """
    if "fields" in payload and "data" in payload:
        fields = [str(f) for f in payload["fields"]]
        rows = [dict(zip(fields, row, strict=True)) for row in payload["data"]]
        df = pd.DataFrame(rows)
        df = df.rename(columns={"cik_str": "cik", "title": "name"})
    else:
        entries = [v for _, v in sorted(payload.items(), key=lambda kv: str(kv[0]))]
        if not entries:
            msg = "company_tickers.json vacío"
            raise DataQualityError(msg)
        df = pd.DataFrame(entries).rename(columns={"cik_str": "cik", "title": "name"})
    required = {"cik", "ticker"}
    if not required.issubset(df.columns):
        msg = f"company_tickers sin columnas {sorted(required - set(df.columns))}"
        raise DataQualityError(msg)
    df["cik"] = df["cik"].map(normalize_cik)
    df["ticker"] = df["ticker"].astype(str).map(normalize_ticker)
    if "name" not in df.columns:
        df["name"] = None
    if "exchange" not in df.columns:
        df["exchange"] = None
    return df.loc[:, ["cik", "ticker", "name", "exchange"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# submissions
# ---------------------------------------------------------------------------

_SUBMISSIONS_RENAME = {
    "accessionNumber": "accession",
    "filingDate": "filing_date",
    "reportDate": "report_date",
    "acceptanceDateTime": "acceptance_raw",
    "primaryDocument": "primary_document",
    "primaryDocDescription": "primary_doc_description",
    "isXBRL": "is_xbrl",
    "isInlineXBRL": "is_inline_xbrl",
    "fileNumber": "file_number",
    "filmNumber": "film_number",
}


def _filing_block(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], list[dict[str, Any]]]:
    """Extrae el bloque de arrays paralelos y la lista de páginas extra.

    El JSON principal trae `filings.recent` + `filings.files`; las páginas
    adicionales traen el bloque de arrays directamente en la raíz (sec_edgar.md
    §4.3).
    """
    if "filings" in payload:
        filings = payload["filings"]
        return filings.get("recent", {}), list(filings.get("files", []) or [])
    return payload, []


def _block_rows(block: Mapping[str, Any]) -> pd.DataFrame:
    """Convierte un bloque struct-of-arrays en filas, validando el alineamiento.

    `filings.recent` son **arrays paralelos**: `accessionNumber[i]`, `form[i]`,
    `acceptanceDateTime[i]` describen el mismo filing (sec_edgar.md §4.2). Si la
    SEC añadiera o truncara una clave, un `pd.DataFrame(block)` desplazaría
    silenciosamente las fechas; por eso el desalineamiento es `DataQualityError`
    (trampa nº 4), nunca un `assert` que desaparece con ``python -O``.
    """
    arrays = {k: v for k, v in block.items() if isinstance(v, list)}
    if "accessionNumber" not in arrays:
        msg = "bloque de submissions sin 'accessionNumber': no es un bloque de filings"
        raise DataQualityError(msg)
    n = len(arrays["accessionNumber"])
    bad = {k: len(v) for k, v in arrays.items() if len(v) != n}
    if bad:
        msg = (
            f"arrays de submissions desalineados: accessionNumber tiene {n} filas "
            f"pero {bad} — un DataFrame construido así desplazaría fechas en silencio"
        )
        raise DataQualityError(msg)
    return pd.DataFrame(arrays)


def submissions_to_frame(
    payload: Mapping[str, Any],
    extra_pages: Sequence[Mapping[str, Any]] = (),
    *,
    assume_eastern: bool = True,
) -> pd.DataFrame:
    """Normaliza `submissions/CIK##########.json` (+ páginas extra) a un DataFrame.

    Columnas: `accession`, `form`, `items`, `filing_date`, `report_date`,
    `accepted_at` (**naive UTC**, desde `acceptanceDateTime` interpretado como hora
    de Nueva York), `primary_document`, `is_xbrl`, `is_inline_xbrl`, `size`…

    `filing_date` se conserva SOLO como metadato administrativo: jamás debe usarse
    como `available_at` (sec_edgar.md §4.4 — el corte de las 17:30 ET desplaza los
    filings vespertinos al día siguiente de forma correlacionada con la hora del
    anuncio).
    """
    recent, _files = _filing_block(payload)
    frames = [_block_rows(recent)]
    frames.extend(_block_rows(_filing_block(page)[0]) for page in extra_pages)
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns=_SUBMISSIONS_RENAME)

    for col in ("form", "items", "primary_document"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    df["filing_date"] = pd.to_datetime(df.get("filing_date"), errors="coerce")
    df["report_date"] = pd.to_datetime(df.get("report_date"), errors="coerce")

    raw = df.get("acceptance_raw")
    accepted: list[pd.Timestamp | pd.NaTType] = []
    for value in ([] if raw is None else raw.tolist()):
        text = str(value).strip() if value is not None else ""
        if not text or text.lower() in {"none", "nan", "nat"}:
            accepted.append(pd.NaT)
        else:
            accepted.append(
                pd.Timestamp(parse_acceptance_datetime(text, assume_eastern=assume_eastern))
            )
    df["accepted_at"] = pd.Series(accepted, index=df.index, dtype="datetime64[ns]")
    if "acceptance_raw" in df.columns:
        df = df.drop(columns=["acceptance_raw"])
    df = df.sort_values("filing_date", kind="mergesort", ascending=False)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# companyfacts / companyconcept
# ---------------------------------------------------------------------------

_DEFAULT_TAXONOMIES = ("us-gaap", "dei", "ifrs-full", "srt")

_INSTANT_SENTINEL = pd.Timestamp("1800-01-01")
"""Centinela interno para agrupar hechos instantáneos (sin `start`) sin perder los
grupos en un `groupby` con NaT."""


def _facts_rows(
    cik: str,
    entity: str | None,
    facts: Mapping[str, Any],
    taxonomies: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for taxonomy, tags in facts.items():
        if taxonomies and taxonomy not in taxonomies:
            continue
        for tag, body in tags.items():
            label = body.get("label")
            deprecated = bool(label and "(Deprecated" in str(label))
            for unit, items in (body.get("units") or {}).items():
                for it in items:
                    rows.append(
                        {
                            "cik": cik,
                            "entity": entity,
                            "taxonomy": str(taxonomy),
                            "concept": str(tag),
                            "label": label,
                            "deprecated": deprecated,
                            "unit": str(unit),
                            "start": it.get("start"),
                            "end": it.get("end"),
                            "value": it.get("val"),
                            "accession": it.get("accn"),
                            "fy": it.get("fy"),
                            "fp": it.get("fp"),
                            "form": it.get("form"),
                            "filed": it.get("filed"),
                            "frame": it.get("frame"),
                        }
                    )
    return rows


def companyfacts_to_frame(
    payload: Mapping[str, Any],
    *,
    taxonomies: Sequence[str] = _DEFAULT_TAXONOMIES,
) -> pd.DataFrame:
    """Normaliza `companyfacts` (o `companyconcept`) a una tabla larga de hechos.

    Columnas: `cik`, `taxonomy`, `concept`, `label`, `deprecated`, `unit`,
    `start` (NaT en hechos instantáneos), `end`, `value`, `accession`, `fy`, `fp`,
    `form`, `filed`, `frame`, más las derivadas:

    - `is_instant` / `duration_days`: el periodo de una observación se determina
      **exclusivamente** por `(start, end)`; `fy`/`fp` describen el filing y no la
      observación (trampa nº 7 — un 10-Q de Q2 trae el trimestre, el acumulado y
      los comparativos del año anterior todos con el mismo `fy`/`fp`).
    - `available_at`: instante PIT (naive UTC) desde `filed` con el corte
      administrativo (`filed_available_at`). Si se necesita el instante exacto,
      únase con `submissions` por `accession` (`facts_to_fundamental_facts`).
    - `is_restated` / `first_filed` / `n_vintages`: ver `mark_vintages`.
    """
    if "facts" in payload:
        facts = payload.get("facts") or {}
        if not facts:
            msg = "companyfacts sin bloque `facts`: la empresa no tiene XBRL (pre-2009?)"
            raise DataQualityError(msg)
        cik = normalize_cik(payload.get("cik", 0))
        rows = _facts_rows(cik, payload.get("entityName"), facts, tuple(taxonomies))
    else:
        # companyconcept: un solo (taxonomy, tag) con la misma forma de `units`.
        cik = normalize_cik(payload.get("cik", 0))
        wrapper = {str(payload.get("taxonomy", "us-gaap")): {str(payload.get("tag", "?")): payload}}
        rows = _facts_rows(cik, payload.get("entityName"), wrapper, ())
    if not rows:
        msg = "companyfacts sin hechos en las taxonomías pedidas"
        raise DataQualityError(msg)

    df = pd.DataFrame(rows)
    df["start"] = pd.to_datetime(df["start"], errors="coerce")
    df["end"] = pd.to_datetime(df["end"], errors="coerce")
    if df["end"].isna().any():
        msg = "hechos XBRL sin `end`: no se puede fechar el periodo"
        raise DataQualityError(msg)
    df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
    if df["filed"].isna().any():
        msg = "hechos XBRL sin `filed`: sin fecha de presentación no hay point-in-time"
        raise DataQualityError(msg)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["form"] = df["form"].fillna("").astype(str)
    df["is_instant"] = df["start"].isna()
    df["duration_days"] = (df["end"] - df["start"]).dt.days

    # available_at: corte administrativo del día `filed` (17:30 ET; 22:00 para
    # formularios de insiders), convertido a UTC de forma vectorizada.
    cutoff_minutes = np.where(
        df["form"].str.strip().str.upper().isin(OWNERSHIP_FORMS),
        OWNERSHIP_DAY_CUTOFF_ET.hour * 60 + OWNERSHIP_DAY_CUTOFF_ET.minute,
        FILING_DAY_CUTOFF_ET.hour * 60 + FILING_DAY_CUTOFF_ET.minute,
    )
    wall = pd.DatetimeIndex(df["filed"]) + pd.to_timedelta(cutoff_minutes, unit="m")
    df["available_at"] = pd.Series(
        eastern_wall_to_utc(wall).to_numpy(), index=df.index, dtype="datetime64[ns]"
    )

    # Duplicados exactos: el mismo hecho puede llegar repetido dentro del payload.
    df = df.drop_duplicates(
        subset=["taxonomy", "concept", "unit", "start", "end", "accession", "value"]
    ).reset_index(drop=True)
    return mark_vintages(df)


def mark_vintages(df: pd.DataFrame, *, rel_tol: float = 1e-9) -> pd.DataFrame:
    """Marca los vintages de cada hecho `(taxonomy, concept, unit, start, end)`.

    `companyfacts` conserva todas las versiones: cuando una empresa reexpresa, el
    nuevo valor aparece como hecho adicional para el mismo periodo, con otro
    `accn` y otro `filed`, sin borrar el antiguo (sec_edgar.md §5.5). Este método
    añade:

    - `first_filed`: primer `filed` del periodo (la serie *as-originally-reported*
      se obtiene filtrando `filed == first_filed`).
    - `n_vintages`: número de versiones del periodo.
    - `is_restated`: True si el hecho es posterior al primero **y su valor
      difiere** del primero. La republicación de un comparativo con el mismo valor
      (práctica habitual en cada 10-Q) no es una reexpresión y no se marca.

    Las enmiendas (`10-K/A`, `10-Q/A`) son vintages legítimos y entran por aquí;
    no se descartan (sec_edgar.md §6.6). El desempate con `filed` idéntico es por
    `accession` (el envío posterior del mismo día), determinista.
    """
    out = df.copy()
    skey = out["start"].fillna(_INSTANT_SENTINEL)
    out["_skey"] = skey
    out = out.sort_values(
        ["taxonomy", "concept", "unit", "_skey", "end", "filed", "accession"],
        kind="mergesort",
    )
    grp = out.groupby(["taxonomy", "concept", "unit", "_skey", "end"], sort=False)
    out["first_filed"] = grp["filed"].transform("first")
    out["n_vintages"] = grp["value"].transform("size").astype(int)
    first_value = grp["value"].transform("first")
    later = (out["filed"] > out["first_filed"]) | (
        (out["filed"] == out["first_filed"])
        & (out["accession"] != grp["accession"].transform("first"))
    )
    denom = first_value.abs().clip(lower=1.0)
    changed = (out["value"] - first_value).abs() > denom * rel_tol
    out["is_restated"] = (later & changed).fillna(False)
    out["is_amendment"] = out["form"].str.endswith("/A")
    return out.drop(columns=["_skey"]).reset_index(drop=True)


def classify_periods(df: pd.DataFrame) -> pd.DataFrame:
    """Añade `period_kind` según `(start, end)`: la única fuente fiable del periodo.

    ==============  ======================================================
    `period_kind`   criterio
    ==============  ======================================================
    ``instant``     sin `start` (saldos de balance, dei)
    ``quarter``     duración en `QUARTER_SPAN_DAYS` (80-100 días)
    ``annual``      duración en `ANNUAL_SPAN_DAYS` (335-395 días)
    ``cumulative``  resto (acumulados de 6 y 9 meses, que se descartan
                    explícitamente antes de agregar — sec_edgar.md §6.4)
    ==============  ======================================================

    Sin este filtro, los acumulados YTD del 10-Q se mezclan con los trimestres y
    la serie queda sistemáticamente inflada en Q2 y Q3 (trampas nº 7 y §6.4).
    """
    out = df.copy()
    dur = out["duration_days"]
    kind = np.full(len(out), "cumulative", dtype=object)
    kind[out["is_instant"].to_numpy(dtype=bool)] = "instant"
    q_lo, q_hi = QUARTER_SPAN_DAYS
    a_lo, a_hi = ANNUAL_SPAN_DAYS
    kind[(dur >= q_lo) & (dur <= q_hi)] = "quarter"
    kind[(dur >= a_lo) & (dur <= a_hi)] = "annual"
    out["period_kind"] = kind
    return out


def _fiscal_label(end: pd.Timestamp, period_kind: str) -> str:
    """Etiqueta fiscal derivada del **periodo del hecho**, jamás de `fy`/`fp`.

    `fy`/`fp` describen el filing que publicó el hecho (trampa nº 7): un
    comparativo de 2023 dentro del 10-Q de 2024 llega con `fy=2024`. La etiqueta
    honesta sale del `period_end` de la observación (trimestre natural
    aproximado; los ejercicios desplazados de 52/53 semanas se etiquetan por el
    trimestre natural del cierre, sec_edgar.md §6.6).
    """
    if period_kind == "annual":
        return f"{end.year}FY"
    return f"{end.year}Q{(end.month - 1) // 3 + 1}"


def facts_to_fundamental_facts(
    df: pd.DataFrame,
    ticker: Ticker,
    *,
    submissions: pd.DataFrame | None = None,
    concept_names: Mapping[str, str] | None = None,
) -> list[Any]:
    """Convierte la tabla larga de `companyfacts_to_frame` en `FundamentalFact`s.

    - `available_at` = `acceptanceDateTime` exacto si `submissions` permite unir
      por `accession`; si no, la cota superior de `filed_available_at` (el corte
      de las 17:30/22:00 ET del día `filed`). En ningún caso `filingDate` a
      medianoche ni `period_end` (checklist nº 1 de sec_edgar.md §12).
    - `accession` y `form` se conservan para auditar el origen de cada cifra.
    - `is_restated` viene de `mark_vintages` y **todas** las versiones se emiten:
      el consumidor elige vintage con `pit.restatements.first_reported` /
      `vintage_asof`.
    - `fiscal_period` se deriva del periodo del hecho (jamás de `fy`/`fp`).

    `concept_names` permite renombrar tags crudos a conceptos lógicos
    (p. ej. ``{"Revenues": "revenue"}``); los tags sin entrada conservan su nombre.
    """
    from earnings_alpha.types import FundamentalFact

    work = classify_periods(df) if "period_kind" not in df.columns else df.copy()
    available = work["available_at"].copy()
    if submissions is not None and len(submissions) > 0:
        acc = (
            submissions.dropna(subset=["accepted_at"])
            .drop_duplicates(subset=["accession"])
            .set_index("accession")["accepted_at"]
        )
        exact = work["accession"].map(acc)
        available = exact.fillna(available)

    out: list[FundamentalFact] = []
    names = dict(concept_names or {})
    for row, avail in zip(work.itertuples(index=False), available, strict=True):
        end = pd.Timestamp(row.end)
        out.append(
            FundamentalFact(
                ticker=ticker,
                concept=names.get(row.concept, row.concept),
                value=float(row.value),
                period_end=end.date(),
                available_at=pd.Timestamp(avail).to_pydatetime(),
                fiscal_period=_fiscal_label(end, str(row.period_kind)),
                unit=str(row.unit),
                form=(str(row.form) or None),
                accession=(None if row.accession is None else str(row.accession)),
                is_restated=bool(row.is_restated),
            )
        )
    return out


# ---------------------------------------------------------------------------
# 8-K item 2.02 -> EarningsEvent
# ---------------------------------------------------------------------------


def _tokenize_items(items: str) -> set[str]:
    """Tokeniza el campo `items` (cadena separada por comas, no lista).

    Un ``"2.02" in items`` sobre la cadena cruda es frágil (sec_edgar.md §7.2);
    aquí se tokeniza y se compara por igualdad exacta.
    """
    return {tok.strip() for tok in str(items or "").split(",") if tok.strip()}


def detect_earnings_8k(
    submissions: pd.DataFrame,
    *,
    calendar: TradingCalendar | None = None,
) -> pd.DataFrame:
    """Localiza los anuncios de resultados (8-K con item 2.02) en un `submissions`.

    Reglas (sec_edgar.md §7.2-7.3):

    - `form == "8-K"` **estricto**: un `8-K/A` es una enmienda posterior y usar su
      `acceptanceDateTime` retrasaría el evento días o semanas.
    - `items` se tokeniza y se exige el token exacto ``"2.02"``.
    - Si un mismo `report_date` genera varios 8-K 2.02 (preliminar + definitivo,
      correcciones), se conserva el de `accepted_at` **mínimo**: el primero es el
      que movió el precio.
    - `session` se clasifica con el calendario real de NYSE (medias sesiones
      incluidas) y `tradable` con `pit.tradable_date`.

    **Limitación documentada** (sec_edgar.md §7.3): el 8-K se *suministra* después
    de que el comunicado haya salido por el hilo de prensa, así que
    `accepted_at >= announced_at` real. El backtest nunca opera antes de que la
    información fuese pública —el sesgo tiene el signo seguro—, pero un anuncio
    BMO cuyo 8-K se aceptó tras la apertura queda clasificado DMH: esas filas se
    marcan `low_confidence_session=True` para que `EventBacktest` pueda excluirlas
    o triangularlas con un proveedor de calendario.
    """
    cal = calendar or get_calendar()
    required = {"form", "items", "accepted_at", "report_date", "accession"}
    missing = required - set(submissions.columns)
    if missing:
        msg = f"submissions sin columnas {sorted(missing)}: usa submissions_to_frame"
        raise DataQualityError(msg)

    mask = submissions["form"].str.strip().eq("8-K") & submissions["items"].map(
        lambda s: "2.02" in _tokenize_items(s)
    )
    hits = submissions.loc[mask].copy()
    if len(hits) == 0:
        return pd.DataFrame(
            columns=[
                "accession", "report_date", "accepted_at", "session",
                "tradable", "low_confidence_session", "items", "primary_document",
            ]
        )
    if hits["accepted_at"].isna().any():
        msg = "8-K 2.02 sin acceptanceDateTime: no se puede fechar el anuncio"
        raise DataQualityError(msg)

    # Deduplicación por trimestre reportado: el primero en aceptarse gana.
    hits = hits.sort_values(["accepted_at", "accession"], kind="mergesort")
    with_period = hits["report_date"].notna()
    deduped = hits.loc[with_period].drop_duplicates(subset=["report_date"], keep="first")
    hits = pd.concat([deduped, hits.loc[~with_period]], ignore_index=True)

    sessions: list[Session] = []
    tradables: list[date] = []
    for ts in hits["accepted_at"]:
        when = pd.Timestamp(ts).to_pydatetime()
        sess = classify_session(when, cal)
        sessions.append(sess)
        tradables.append(tradable_date(when, sess, cal))
    hits["session"] = [s.value for s in sessions]
    hits["tradable"] = pd.to_datetime(tradables)
    hits["low_confidence_session"] = [s is Session.DMH for s in sessions]

    cols = [
        "accession", "report_date", "accepted_at", "session",
        "tradable", "low_confidence_session", "items", "primary_document",
    ]
    extra = [c for c in ("filing_date", "size") if c in hits.columns]
    return hits.loc[:, cols + extra].sort_values("accepted_at").reset_index(drop=True)


def frame_to_earnings_events(
    events: pd.DataFrame,
    ticker: Ticker,
    cik: CIK | None = None,
    *,
    source: str = "sec:8-K-2.02",
) -> list[EarningsEvent]:
    """Convierte la tabla de `detect_earnings_8k` en `types.EarningsEvent`.

    `fiscal_quarter` se deriva del `report_date` (trimestre natural del cierre);
    `eps_actual`/`eps_estimate` quedan a ``None``: el 8-K aporta lo que solo él
    aporta —el timestamp—, las cifras llegan del XBRL del 10-Q y del proveedor de
    consenso (sec_edgar.md §7.4).
    """
    out: list[EarningsEvent] = []
    for row in events.itertuples(index=False):
        if pd.isna(row.report_date):
            continue
        pe = pd.Timestamp(row.report_date)
        announced = pd.Timestamp(row.accepted_at).to_pydatetime()
        out.append(
            EarningsEvent(
                ticker=ticker,
                period_end=pe.date(),
                announced_at=announced,
                session=Session(str(row.session)),
                fiscal_quarter=f"{pe.year}Q{(pe.month - 1) // 3 + 1}",
                cik=cik,
                source=source,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Form 4
# ---------------------------------------------------------------------------

FORM4_TRANSACTION_CODES: dict[str, str] = {
    # Tabla completa del propio Formulario 4 (sec_edgar.md §8.3, verbatim).
    "A": "Grant, award or other acquisition pursuant to Rule 16b-3(d)",
    "C": "Conversion of derivative security",
    "D": "Disposition to the issuer pursuant to Rule 16b-3(e)",
    "E": "Expiration of short derivative position",
    "F": "Payment of exercise price or tax liability by delivering/withholding securities",
    "G": "Bona fide gift",
    "H": "Expiration (or cancellation) of long derivative position with value received",
    "I": "Discretionary transaction pursuant to Rule 16b-3(f)",
    "J": "Other acquisition or disposition (describe transaction)",
    "K": "Transaction in equity swap or similar instrument",
    "L": "Small acquisition under Rule 16a-6",
    "M": "Exercise or conversion of derivative security exempted pursuant to Rule 16b-3",
    "O": "Exercise of out-of-the-money derivative security",
    "P": "Open market or private purchase of non-derivative or derivative security",
    "S": "Open market or private sale of non-derivative or derivative security",
    "U": "Disposition pursuant to a tender of shares in a change of control transaction",
    "V": "Transaction voluntarily reported earlier than required",
    "W": "Acquisition or disposition by will or the laws of descent and distribution",
    "X": "Exercise of in-the-money or at-the-money derivative security",
    "Z": "Deposit into or withdrawal from voting trust",
}

DISCRETIONARY_BUY_CODES: frozenset[str] = frozenset({"P"})
"""Compra en mercado abierto con dinero propio: la señal por excelencia (Cohen,
Malloy y Pomorski 2012; sec_edgar.md §8.4)."""

DISCRETIONARY_SELL_CODES: frozenset[str] = frozenset({"S"})
"""Venta en mercado abierto. Mucho más ruidosa que P (diversificación, impuestos);
además, una S el mismo día que una M del mismo filing es la pata de un ejercicio
de opciones preprogramado (10b5-1) y se excluye del neto."""

ROUTINE_CODES: frozenset[str] = frozenset({"A", "C", "D", "F", "G", "M", "O", "U", "W", "X", "Z"})
"""Retribución y mecánica (concesiones, vesting, ejercicios, donaciones,
recompras): sin contenido informativo de precio y con estacionalidad anual fija
que **imita** poder predictivo en la ventana pre-evento (trampa nº 9). `I`, `J`,
`K`, `L` no están aquí: son heterogéneas y se excluyen del neto pero se cuentan
aparte."""

_XML_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_XML_BARE_AMP = re.compile(r"&(?!#\d+;|#x[0-9a-fA-F]+;|[A-Za-z][A-Za-z0-9]*;)")
_XML_KNOWN_ENTITIES = frozenset({"amp", "lt", "gt", "apos", "quot"})
_XML_HTML_ENTITY = re.compile(r"&([A-Za-z][A-Za-z0-9]*);")


def _clean_form4_xml(text: str) -> str:
    """Sanea el XML de un Form 4 antes de parsear.

    Es habitual encontrar entidades HTML sin escapar y caracteres de control en
    `natureOfOwnership` y en las notas al pie (sec_edgar.md §8.2); las
    implementaciones de referencia limpian con regex antes de parsear.
    """
    text = _XML_CTRL.sub("", text)
    text = _XML_BARE_AMP.sub("&amp;", text)

    def _entity(match: re.Match[str]) -> str:
        name = match.group(1)
        return match.group(0) if name in _XML_KNOWN_ENTITIES else f"&amp;{name};"

    return _XML_HTML_ENTITY.sub(_entity, text)


def _el_text(parent: ET.Element | None, *path: str) -> str | None:
    """Texto de un elemento, desenvolviendo `<value>` si existe.

    En el esquema `ownershipDocument` casi todo valor va envuelto:
    `<transactionShares><value>34.1689</value><footnoteId id="F1"/></transactionShares>`.
    Un `elem.text` ingenuo devuelve None o espacios (sec_edgar.md §8.2).
    """
    node = parent
    for step in path:
        if node is None:
            return None
        node = node.find(step)
    if node is None:
        return None
    value = node.find("value")
    target = value if value is not None else node
    text = (target.text or "").strip()
    return text or None


def _el_float(parent: ET.Element | None, *path: str) -> float:
    raw = _el_text(parent, *path)
    if raw is None:
        return float("nan")
    try:
        return float(raw)
    except ValueError:
        return float("nan")


def _el_bool(parent: ET.Element | None, *path: str) -> bool:
    raw = _el_text(parent, *path)
    return str(raw).strip().lower() in {"1", "true", "yes"}


def parse_form4(xml_text: str) -> dict[str, Any]:
    """Parsea el XML `ownershipDocument` de un Form 3/4/5.

    Maneja las tres particularidades que rompen los parsers ingenuos
    (sec_edgar.md §8.2): valores envueltos en `<value>` (con `<footnoteId>`
    conviviendo en el mismo elemento), `reportingOwner` como **lista** (las
    transacciones conjuntas declaran varios propietarios y perderlos sesga el
    recuento de insiders distintos) y XML mal formado (entidades sin escapar,
    caracteres de control).

    Devuelve un dict con `document_type`, `period_of_report`, `issuer`
    (cik/name/ticker), `owners` (lista), `transactions` (lista: tabla
    no-derivada y derivada) y `footnotes`.
    """
    try:
        root = ET.fromstring(_clean_form4_xml(xml_text))
    except ET.ParseError as exc:
        msg = f"XML de Form 4 no parseable ni tras el saneado: {exc}"
        raise DataQualityError(msg) from exc
    if root.tag != "ownershipDocument":
        msg = f"el XML no es un ownershipDocument (raíz {root.tag!r})"
        raise DataQualityError(msg)

    issuer = root.find("issuer")
    issuer_ticker = _el_text(issuer, "issuerTradingSymbol")
    parsed: dict[str, Any] = {
        "schema_version": _el_text(root, "schemaVersion"),
        "document_type": _el_text(root, "documentType"),
        "period_of_report": _el_text(root, "periodOfReport"),
        "issuer": {
            "cik": (
                normalize_cik(_el_text(issuer, "issuerCik"))
                if _el_text(issuer, "issuerCik")
                else None
            ),
            "name": _el_text(issuer, "issuerName"),
            "ticker": normalize_ticker(issuer_ticker) if issuer_ticker else None,
        },
        "owners": [],
        "transactions": [],
        "footnotes": {},
    }

    for owner in root.findall("reportingOwner"):  # puede haber varios
        rel = owner.find("reportingOwnerRelationship")
        raw_cik = _el_text(owner, "reportingOwnerId", "rptOwnerCik")
        parsed["owners"].append(
            {
                "cik": normalize_cik(raw_cik) if raw_cik else None,
                "name": _el_text(owner, "reportingOwnerId", "rptOwnerName"),
                "is_director": _el_bool(rel, "isDirector"),
                "is_officer": _el_bool(rel, "isOfficer"),
                "is_ten_percent_owner": _el_bool(rel, "isTenPercentOwner"),
                "is_other": _el_bool(rel, "isOther"),
                "officer_title": _el_text(rel, "officerTitle"),
            }
        )

    for note in root.findall("footnotes/footnote"):
        parsed["footnotes"][note.get("id", "")] = (note.text or "").strip()

    def _transactions(table_tag: str, row_tag: str, table_name: str) -> None:
        table = root.find(table_tag)
        if table is None:
            return
        for txn in table.findall(row_tag):
            coding = txn.find("transactionCoding")
            amounts = txn.find("transactionAmounts")
            raw_date = _el_text(txn, "transactionDate")
            parsed["transactions"].append(
                {
                    "table": table_name,
                    "security_title": _el_text(txn, "securityTitle"),
                    "transaction_date": (
                        pd.Timestamp(raw_date).date() if raw_date else None
                    ),
                    "form_type": _el_text(coding, "transactionFormType"),
                    "code": _el_text(coding, "transactionCode"),
                    "equity_swap": _el_bool(coding, "equitySwapInvolved"),
                    "shares": _el_float(amounts, "transactionShares"),
                    "price": _el_float(amounts, "transactionPricePerShare"),
                    "acquired_disposed": _el_text(
                        amounts, "transactionAcquiredDisposedCode"
                    ),
                    "shares_owned_after": _el_float(
                        txn, "postTransactionAmounts", "sharesOwnedFollowingTransaction"
                    ),
                    "ownership": _el_text(
                        txn, "ownershipNature", "directOrIndirectOwnership"
                    ),
                }
            )

    _transactions("nonDerivativeTable", "nonDerivativeTransaction", "non_derivative")
    _transactions("derivativeTable", "derivativeTransaction", "derivative")
    return parsed


def form4_to_frame(
    parsed: Mapping[str, Any],
    *,
    accession: str | None = None,
    available_at: datetime | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Convierte un Form 4 parseado en filas de transacción (una por transacción).

    Los propietarios se agregan en columnas (`owner_ciks`/`owner_names` separados
    por ``;``, flags OR de rol): una transacción conjunta de dos insiders es
    **una** transacción, y duplicar la fila por propietario doblaría el importe en
    la agregación.

    `available_at` es el `acceptanceDateTime` del Form 4, **no** la fecha de la
    transacción: entre ambas median hasta 2 días hábiles (plazo Sarbanes-Oxley) y
    agregarlo por `transaction_date` sería look-ahead de 1-2 días justo en la
    ventana pre-evento (sec_edgar.md §8.1).
    """
    owners = list(parsed.get("owners", []))
    txns = list(parsed.get("transactions", []))
    issuer = dict(parsed.get("issuer", {}))
    if not txns:
        return pd.DataFrame(columns=_FORM4_COLUMNS)

    owner_ciks = ";".join(str(o.get("cik") or "") for o in owners)
    owner_names = ";".join(str(o.get("name") or "") for o in owners)
    officer_titles = [o.get("officer_title") for o in owners if o.get("officer_title")]

    rows = []
    for txn in txns:
        rows.append(
            {
                "ticker": issuer.get("ticker"),
                "issuer_cik": issuer.get("cik"),
                "owner_ciks": owner_ciks,
                "owner_names": owner_names,
                "n_owners": len(owners),
                "is_director": any(o.get("is_director") for o in owners),
                "is_officer": any(o.get("is_officer") for o in owners),
                "is_ten_percent_owner": any(o.get("is_ten_percent_owner") for o in owners),
                "officer_title": officer_titles[0] if officer_titles else None,
                **txn,
                "accession": accession,
                "available_at": (pd.Timestamp(available_at) if available_at is not None else pd.NaT),
            }
        )
    df = pd.DataFrame(rows)
    df["value"] = df["shares"] * df["price"]
    df["available_at"] = pd.to_datetime(df["available_at"])
    return df.loc[:, _FORM4_COLUMNS]


_FORM4_COLUMNS = [
    "ticker", "issuer_cik", "owner_ciks", "owner_names", "n_owners",
    "is_director", "is_officer", "is_ten_percent_owner", "officer_title",
    "table", "security_title", "transaction_date", "form_type", "code",
    "equity_swap", "shares", "price", "acquired_disposed", "shares_owned_after",
    "ownership", "accession", "available_at", "value",
]


def form4_code_distribution(transactions: pd.DataFrame) -> pd.Series:
    """Distribución de códigos de transacción, para el sanity-check de trampa nº 9.

    En un panel sano, A/F/M dominan en número y las P son minoría clara. Si las P
    salen mayoritarias, el filtro (o el parseo del código) está mal puesto.
    """
    if "code" not in transactions.columns:
        msg = "el DataFrame de transacciones no tiene columna 'code'"
        raise DataQualityError(msg)
    return transactions["code"].value_counts()


def insider_net_buys(
    transactions: pd.DataFrame,
    *,
    by: str = "available_at",
) -> pd.DataFrame:
    """Agrega transacciones de Form 4 a compras netas discrecionales por día.

    Definición (sec_edgar.md §8.4, alineada con Cohen-Malloy-Pomorski 2012):

    ``net_buy_value = Σ(P: shares × price) − Σ(S no asociada a M: shares × price)``

    - Solo cuentan las **P** (compra en mercado abierto, dinero propio) y las
      **S** no ligadas a un ejercicio: una S con una **M** del mismo filing y
      fecha es la venta preprogramada de un ejercicio de opciones (rutina/10b5-1)
      y se acumula aparte en `sell_linked_value`.
    - `A`/`F`/`M` y demás códigos de retribución/mecánica **se excluyen** del neto
      y se cuentan en `n_routine_trades`: su estacionalidad anual correlaciona con
      el calendario de resultados e imita poder predictivo (trampa nº 9).
    - La clave temporal por defecto es **`available_at`** (aceptación del Form 4),
      no `transaction_date`: "compras *publicadas* en [T-N, T-1]", no
      "*realizadas*". Con `transaction_date` la feature sería look-ahead de 1-2
      días en cada observación (sec_edgar.md §8.1).

    Devuelve un DataFrame indexado por `(ticker, date)` con importes brutos y
    netos, número de operaciones y número de insiders **distintos** comprando
    (más informativo que el importe: varios directivos comprando a la vez es más
    señal que uno comprando mucho). La normalización por volumen en dólares es
    responsabilidad del consumidor (`PreEventFeatures`), que es quien tiene el
    panel de precios.
    """
    if len(transactions) == 0:
        msg = "no hay transacciones de Form 4 que agregar"
        raise InsufficientHistory(msg)
    if by not in transactions.columns:
        msg = f"columna de agregación {by!r} ausente"
        raise DataQualityError(msg)
    df = transactions.copy()
    if by == "available_at" and df[by].isna().any():
        msg = (
            "hay transacciones sin `available_at` (aceptación del Form 4): agregarlas "
            "por fecha de transacción sería look-ahead de 1-2 días"
        )
        raise DataQualityError(msg)

    # S ligada a M: mismo filing (accession) y misma fecha de transacción.
    link_key = df["accession"].fillna(df["owner_ciks"]).astype(str)
    m_keys = set(
        zip(
            link_key[df["code"] == "M"],
            df.loc[df["code"] == "M", "transaction_date"],
            strict=True,
        )
    )
    linked = pd.Series(
        [
            (k, d) in m_keys
            for k, d in zip(link_key, df["transaction_date"], strict=True)
        ],
        index=df.index,
    )
    df["sale_linked_to_exercise"] = (df["code"] == "S") & linked

    df["date"] = pd.to_datetime(df[by]).dt.normalize()
    df["ticker"] = df["ticker"].astype(str)
    value = df["value"].fillna(0.0)

    is_buy = df["code"].isin(DISCRETIONARY_BUY_CODES) & (df["acquired_disposed"] == "A")
    is_sell = (
        df["code"].isin(DISCRETIONARY_SELL_CODES)
        & (df["acquired_disposed"] == "D")
        & ~df["sale_linked_to_exercise"]
    )
    is_linked_sell = (df["code"] == "S") & df["sale_linked_to_exercise"]
    is_routine = df["code"].isin(ROUTINE_CODES)

    def _owners(mask: pd.Series) -> pd.Series:
        return (
            df.loc[mask]
            .groupby(["ticker", "date"])["owner_ciks"]
            .agg(lambda col: len({c for cell in col for c in str(cell).split(";") if c}))
        )

    grouped = df.groupby(["ticker", "date"])
    out = pd.DataFrame(
        {
            "buy_value": value.where(is_buy, 0.0).groupby([df["ticker"], df["date"]]).sum(),
            "sell_value": value.where(is_sell, 0.0).groupby([df["ticker"], df["date"]]).sum(),
            "sell_linked_value": value.where(is_linked_sell, 0.0)
            .groupby([df["ticker"], df["date"]])
            .sum(),
            "n_buy_trades": is_buy.groupby([df["ticker"], df["date"]]).sum().astype(int),
            "n_sell_trades": is_sell.groupby([df["ticker"], df["date"]]).sum().astype(int),
            "n_routine_trades": is_routine.groupby([df["ticker"], df["date"]]).sum().astype(int),
            "n_trades": grouped.size().astype(int),
        }
    )
    out.index = out.index.set_names(["ticker", "date"])
    out["net_buy_value"] = out["buy_value"] - out["sell_value"]
    out["n_distinct_buyers"] = _owners(is_buy).reindex(out.index).fillna(0).astype(int)
    out["n_distinct_sellers"] = _owners(is_sell).reindex(out.index).fillna(0).astype(int)
    return out.sort_index()


# ---------------------------------------------------------------------------
# Proveedor
# ---------------------------------------------------------------------------


class EdgarProvider(BaseProvider):
    """Cliente de SEC EDGAR: identidad, submissions, XBRL, 8-K 2.02 y Form 4.

    No requiere clave de API pero **sí** `SEC_USER_AGENT` (nombre + correo de
    contacto): la SEC exige declarar el tráfico automatizado y `www.sec.gov`
    rechaza los UA genéricos (sec_edgar.md §2.1-2.2). Sin esa variable,
    `available()` es False y cualquier llamada lanza `ProviderUnavailable` antes
    de tocar la red.

    Tasa: 10 req/s agregadas sobre los tres hosts; el token bucket del repo va a
    8 req/s con margen deliberado (`PROVIDER_RATE_LIMITS["sec"]`). Superar la
    cuota devuelve **403 con cuerpo HTML**, no 429: `_raise_403` lo distingue.

    `cache` (opcional): un `DiskCache` con los TTL de sec_edgar.md §2.6 — los
    documentos de `Archives` son inmutables; `submissions`/`companyfacts` caducan
    a diario; `frames` a la semana.
    """

    name = "sec"
    kinds = (
        DataKind.FUNDAMENTALS.value,
        DataKind.FILINGS.value,
        DataKind.INSIDER.value,
        DataKind.EARNINGS_CALENDAR.value,
    )
    probe_url = None  # la disponibilidad se decide por credenciales, no sondeo

    def __init__(
        self,
        *,
        settings: Any = None,
        http: HttpClient | None = None,
        transport: Transport | None = None,
        cache: DiskCache | None = None,
        calendar: TradingCalendar | None = None,
        clock: Clock | None = None,
    ) -> None:
        super().__init__(settings=settings, http=http, clock=clock)
        if http is None and transport is not None:
            self._http = HttpClient(
                self.name,
                transport=transport,
                rate=rate_limit_for(self.name, self.settings),
                settings=self.settings,
                clock=clock,
                user_agent=self.settings.env("SEC_USER_AGENT") or DEFAULT_USER_AGENT,
            )
        self.cache = cache
        self.calendar = calendar or get_calendar()

    # -- infraestructura ------------------------------------------------------

    def _ensure_available(self) -> None:
        missing = self.missing_env()
        if missing:
            raise ProviderUnavailable(
                self.name,
                "la SEC exige un User-Agent identificable (nombre + email de contacto)",
                missing_env=missing,
            )

    def _raise_403(self, url: str, resp: HttpResponse) -> None:
        """Clasifica un 403 de la SEC (trampa nº 2).

        Con cuerpo de limitación (*"declare your traffic"*, *"Request Rate
        Threshold"*) es `RateLimited`: se debe reintentar con backoff. Cualquier
        otro 403 es `ProviderUnavailable` (UA rechazado o egress bloqueado):
        reintentar no arregla nada y acelera el bloqueo de IP.
        """
        body = resp.text.lower()
        if any(marker in body for marker in RATE_LIMIT_BODY_MARKERS):
            raise RateLimited(self.name, None)
        raise ProviderUnavailable(
            self.name,
            f"HTTP 403 en {url}: User-Agent rechazado o egress bloqueado",
            missing_env=self.missing_env(),
        )

    def _request_json(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        allow_404: bool = False,
    ) -> Any:
        self._ensure_available()
        resp = self.http.get(url, params=params, allow_status=(403, 404))
        if resp.status_code == 403:
            self._raise_403(url, resp)
        if resp.status_code == 404:
            if allow_404:
                return None
            raise HttpStatusError(self.name, 404, url, resp.text[:200])
        return resp.json()

    def _fetch_json(
        self,
        url: str,
        *,
        kind: str,
        ttl: float | None,
        immutable: bool = False,
        params: Mapping[str, Any] | None = None,
    ) -> Any:
        if self.cache is None:
            return self._request_json(url, params=params)
        key = {"url": url, **(dict(params) if params else {})}
        return self.cache.fetch(
            kind,
            self.name,
            key,
            lambda: self._request_json(url, params=params),
            ttl=ttl,
            immutable=immutable,
        )

    # -- identidad ------------------------------------------------------------

    def company_tickers(self) -> pd.DataFrame:
        """Mapa instrumento→CIK de **hoy** (`company_tickers_exchange.json`).

        Advertencia de supervivencia (sec_edgar.md §3.3a): el fichero es un
        snapshot del presente; el 56 % del universo histórico del repo no aparece
        en él. Para tickers deslistados hay que reconstruir el mapeo con los
        índices trimestrales y los Form 4 (§11.2); un `None` aquí significa "no
        está hoy", no "no existe".
        """
        payload = self._fetch_json(
            f"{SEC_WWW}/files/company_tickers_exchange.json",
            kind=DataKind.UNIVERSE.value,
            ttl=TTL_TICKERS_S,
        )
        return company_tickers_to_frame(payload)

    def cik_for(self, ticker: Ticker) -> CIK | None:
        """CIK del ticker según el snapshot actual (None si no está hoy)."""
        table = self.company_tickers()
        hit = table.loc[table["ticker"] == normalize_ticker(ticker)]
        return None if len(hit) == 0 else str(hit.iloc[0]["cik"])

    # -- submissions ----------------------------------------------------------

    def submissions(
        self,
        cik: CIK | str | int,
        *,
        all_pages: bool = True,
        require_history_from: date | None = None,
    ) -> pd.DataFrame:
        """Índice completo de filings de un emisor, con paginación.

        `filings.recent` solo cubre ~1 año o 1 000 filings; el resto vive en
        `filings.files[]` y **hay que paginarlo siempre** (trampa nº 5: los 8-K
        de 2010 no están en `recent` y la respuesta corta es perfectamente
        válida). Con `require_history_from` se comprueba que el histórico
        paginado cubre el inicio del backtest y, si no, se lanza
        `InsufficientHistory` en vez de devolver un panel corto.
        """
        cik10 = normalize_cik(cik)
        payload = self._fetch_json(
            f"{SEC_DATA}/submissions/CIK{cik10}.json",
            kind=DataKind.FILINGS.value,
            ttl=TTL_SUBMISSIONS_S,
        )
        pages: list[Mapping[str, Any]] = []
        if all_pages:
            _, files = _filing_block(payload)
            for entry in files:
                name = str(entry.get("name", "")).strip()
                if not name:
                    continue
                pages.append(
                    self._fetch_json(
                        f"{SEC_DATA}/submissions/{name}",
                        kind=DataKind.FILINGS.value,
                        ttl=TTL_SUBMISSIONS_S,
                    )
                )
        df = submissions_to_frame(payload, pages)
        if require_history_from is not None:
            oldest = df["filing_date"].min()
            if pd.isna(oldest) or oldest.date() > require_history_from:
                msg = (
                    f"el histórico de filings de CIK {cik10} empieza en "
                    f"{None if pd.isna(oldest) else oldest.date()} y el backtest exige "
                    f"cobertura desde {require_history_from}"
                )
                raise InsufficientHistory(msg)
        return df

    # -- XBRL -----------------------------------------------------------------

    def company_facts(self, cik: CIK | str | int) -> pd.DataFrame:
        """Todos los hechos XBRL del emisor, en tabla larga con vintages.

        Es la única vía correcta para el panel de factores: conserva todos los
        vintages y permite filtrar por `filed` (algoritmo PIT de sec_edgar.md
        §5.5). Cobertura: sin XBRL antes de ~2009 (§5.6). Los hechos son **por
        emisor, no por clase de acción** (trampa nº 3): `GOOGL` y `GOOG`
        comparten cifras y el consumidor debe poder colapsar a una clase.
        """
        cik10 = normalize_cik(cik)
        payload = self._fetch_json(
            f"{SEC_DATA}/api/xbrl/companyfacts/CIK{cik10}.json",
            kind=DataKind.FUNDAMENTALS.value,
            ttl=TTL_COMPANYFACTS_S,
        )
        return companyfacts_to_frame(payload)

    def company_concept(
        self, cik: CIK | str | int, taxonomy: str, tag: str
    ) -> pd.DataFrame:
        """Un concepto de una empresa (`companyconcept`). 404 = no lo reporta.

        Un 404 aquí es información legítima —"esta empresa no publica ese tag"—
        y se convierte en `InsufficientHistory` explícito, nunca en un DataFrame
        vacío silencioso (Regla de Oro nº 5).
        """
        cik10 = normalize_cik(cik)
        url = f"{SEC_DATA}/api/xbrl/companyconcept/CIK{cik10}/{taxonomy}/{tag}.json"
        payload = self._request_json(url, allow_404=True)
        if payload is None:
            msg = f"CIK {cik10} no reporta {taxonomy}/{tag} (404 de companyconcept)"
            raise InsufficientHistory(msg)
        return companyfacts_to_frame(payload)

    def frames(self, taxonomy: str, tag: str, unit: str, period: str) -> pd.DataFrame:
        """Un concepto, todas las empresas, un periodo calendario (`frames`).

        ⚠ **NO es point-in-time y no debe construir el panel** (trampa nº 6): la
        propia definición de la SEC es *"one fact … that is last filed"*, es
        decir, la reexpresión; además el marco se rellena a posteriori durante
        meses y `pts` cambia entre ejecuciones, lo que viola el determinismo
        (contrato §0.4). Sirve para exploración, cobertura de un tag y
        sanity-checks cruzados. Formato del periodo: ``CY2019``, ``CY2019Q1``,
        ``CY2019Q1I`` (la ``I`` marca hechos instantáneos y es obligatoria para
        saldos de balance).
        """
        if not re.fullmatch(r"CY\d{4}(Q[1-4]I?)?", period):
            msg = f"periodo de frames inválido: {period!r} (esperado CY####, CY####Q#, CY####Q#I)"
            raise DataQualityError(msg)
        payload = self._fetch_json(
            f"{SEC_DATA}/api/xbrl/frames/{taxonomy}/{tag}/{unit}/{period}.json",
            kind=DataKind.FUNDAMENTALS.value,
            ttl=TTL_FRAMES_S,
        )
        rows = payload.get("data") or []
        if not rows:
            msg = f"frames {taxonomy}/{tag}/{unit}/{period} sin datos"
            raise DataQualityError(msg)
        df = pd.DataFrame(rows).rename(columns={"val": "value", "accn": "accession"})
        df["cik"] = df["cik"].map(normalize_cik)
        df["end"] = pd.to_datetime(df["end"])
        if "start" in df.columns:
            df["start"] = pd.to_datetime(df["start"], errors="coerce")
        df.attrs.update(
            {
                "ccp": payload.get("ccp"),
                "uom": payload.get("uom"),
                "pts": payload.get("pts"),
                "taxonomy": payload.get("taxonomy"),
                "tag": payload.get("tag"),
                "not_point_in_time": True,
            }
        )
        return df

    # -- 8-K 2.02 -------------------------------------------------------------

    def earnings_events(
        self,
        cik: CIK | str | int,
        *,
        submissions: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Tabla de anuncios de resultados (8-K 2.02) con sesión y fecha negociable.

        Ver `detect_earnings_8k` para las reglas y la limitación documentada:
        el `acceptanceDateTime` del 8-K es una **cota superior** del instante del
        comunicado de prensa (el 8-K se suministra después de que la nota salga
        por el hilo), así que nunca introduce look-ahead pero puede clasificar
        DMH un anuncio que fue BMO; esas filas quedan `low_confidence_session`.
        Cobertura: el item 2.02 existe desde agosto de 2004 (sec_edgar.md §7.5).
        """
        subs = submissions if submissions is not None else self.submissions(cik)
        return detect_earnings_8k(subs, calendar=self.calendar)

    def earnings_8k(
        self,
        cik: CIK | str | int,
        *,
        ticker: Ticker | None = None,
        submissions: pd.DataFrame | None = None,
    ) -> list[EarningsEvent]:
        """Anuncios de resultados como `types.EarningsEvent`.

        El ticker sale del payload de `submissions` (`tickers[0]`, normalizado) si
        no se pasa explícito; sin ticker resoluble se lanza `DataQualityError`
        porque `EarningsEvent.event_id` se construye por ticker (trampa nº 3: el
        CIK no identifica la clase de acción).
        """
        cik10 = normalize_cik(cik)
        resolved = ticker
        if resolved is None:
            payload = self._fetch_json(
                f"{SEC_DATA}/submissions/CIK{cik10}.json",
                kind=DataKind.FILINGS.value,
                ttl=TTL_SUBMISSIONS_S,
            )
            tickers = payload.get("tickers") or []
            if tickers:
                resolved = normalize_ticker(str(tickers[0]))
        if not resolved:
            msg = (
                f"no se puede resolver el ticker de CIK {cik10}: pásalo explícito "
                "(EarningsEvent se identifica por ticker, no por CIK)"
            )
            raise DataQualityError(msg)
        events = self.earnings_events(cik10, submissions=submissions)
        return frame_to_earnings_events(events, resolved, cik10)

    # -- Archives -------------------------------------------------------------

    @staticmethod
    def archive_url(cik: CIK | str | int, accession: str, document: str = "") -> str:
        """URL de un documento de `Archives` (sec_edgar.md §1).

        `Archives` exige el CIK **sin** ceros a la izquierda y el accession **sin**
        guiones; mezclar convenciones produce 404 silenciosos (trampa nº 1).
        """
        cik_short = normalize_cik(cik).lstrip("0") or "0"
        accn = str(accession).replace("-", "")
        base = f"{SEC_WWW}/Archives/edgar/data/{cik_short}/{accn}"
        return f"{base}/{document}" if document else base

    def filing_index(self, cik: CIK | str | int, accession: str) -> list[dict[str, Any]]:
        """Listado de documentos de un envío (`index.json`)."""
        payload = self._fetch_json(
            self.archive_url(cik, accession, "index.json"),
            kind=DataKind.FILINGS.value,
            ttl=None,
            immutable=True,  # un filing aceptado es inmutable (§2.6)
        )
        directory = payload.get("directory") or {}
        return [dict(item) for item in directory.get("item", [])]

    def fetch_document(self, cik: CIK | str | int, accession: str, document: str) -> bytes:
        """Descarga un documento concreto de un envío (inmutable: caché ∞)."""
        url = self.archive_url(cik, accession, document)

        def _load() -> bytes:
            self._ensure_available()
            resp = self.http.get(url, allow_status=(403,))
            if resp.status_code == 403:
                self._raise_403(url, resp)
            return resp.content

        if self.cache is None:
            return _load()
        return self.cache.fetch(
            DataKind.FILINGS.value,
            self.name,
            {"url": url},
            _load,
            immutable=True,
        )

    # -- Form 4 ---------------------------------------------------------------

    def _form4_xml(self, cik10: str, accession: str, primary_document: str) -> str:
        """Localiza y descarga el XML `ownershipDocument` de un envío de Form 4.

        Lo robusto es leer `index.json` y quedarse con el `.xml` cuyo contenido
        empiece por `<ownershipDocument>` (sec_edgar.md §8.2): el
        `primaryDocument` puede ser una versión renderizada (`xslF345X…/*.html`).
        """
        candidates: list[str] = []
        doc = str(primary_document or "").strip()
        if doc.lower().endswith(".xml") and "/" not in doc:
            candidates.append(doc)
        else:
            for item in self.filing_index(cik10, accession):
                name = str(item.get("name", ""))
                if name.lower().endswith(".xml") and "index" not in name.lower():
                    candidates.append(name)
        for name in candidates:
            text = self.fetch_document(cik10, accession, name).decode("utf-8", "replace")
            if "<ownershipDocument" in text:
                return text
        msg = f"el envío {accession} no contiene un ownershipDocument XML"
        raise DataQualityError(msg)

    def form4(
        self,
        cik: CIK | str | int,
        start: date,
        end: date,
        *,
        include_amendments: bool = False,
        submissions: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Transacciones de insiders **publicadas** en `[start, end]`.

        La ventana filtra por `accepted_at` (fecha de publicación del Form 4), no
        por fecha de transacción: es la única agregación sin look-ahead
        (sec_edgar.md §8.1 — entre transacción y publicación median hasta 2 días
        hábiles de Sarbanes-Oxley, y los Formularios 4 conservan su `filingDate`
        hasta las 22:00 ET).

        Devuelve una fila por transacción con `available_at = accepted_at` del
        filing. Una ventana sin Form 4 devuelve un DataFrame vacío **tipado**: es
        una ausencia verificada contra el índice completo de filings del emisor
        (no un fallo de proveedor), y "ningún insider publicó" es un dato
        legítimo de la feature.

        Advertencia de coste (sec_edgar.md §8.5): para cargas masivas úsense los
        índices diarios, no este fan-out por empresa.
        """
        cik10 = normalize_cik(cik)
        subs = submissions if submissions is not None else self.submissions(cik10)
        forms = {"4"} | ({"4/A"} if include_amendments else set())
        mask = (
            subs["form"].str.strip().isin(forms)
            & subs["accepted_at"].notna()
            & (subs["accepted_at"] >= pd.Timestamp(start))
            & (subs["accepted_at"] <= pd.Timestamp(end) + pd.Timedelta(days=1))
        )
        rows = subs.loc[mask]
        frames_out: list[pd.DataFrame] = []
        for row in rows.itertuples(index=False):
            xml_text = self._form4_xml(cik10, row.accession, row.primary_document)
            parsed = parse_form4(xml_text)
            frames_out.append(
                form4_to_frame(
                    parsed,
                    accession=row.accession,
                    available_at=row.accepted_at,
                )
            )
        if not frames_out:
            return pd.DataFrame(columns=_FORM4_COLUMNS)
        return pd.concat(frames_out, ignore_index=True)

    # -- búsqueda a texto completo -------------------------------------------

    def full_text_search(
        self,
        q: str,
        *,
        forms: str | Sequence[str] | None = None,
        startdt: date | str | None = None,
        enddt: date | str | None = None,
        ciks: CIK | Sequence[CIK] | None = None,
        size: int = 100,
        offset: int = 0,
    ) -> pd.DataFrame:
        """Búsqueda a texto completo (efts.sec.gov). Cobertura: 2001→.

        Uso previsto: descubrimiento y control de calidad (localizar EX-99.1,
        auditar la cobertura del item 2.02), **no** construir la lista de eventos
        (sec_edgar.md §9.4).

        Trampa nº 10: cuando `hits.total.relation == "gte"`, el total **no es un
        recuento** sino el techo de 10 000 resultados paginables. En ese caso
        `df.attrs["hit_cap"] = True` y `df.attrs["total"] = None` (se conserva
        `total_lower_bound`); ningún consumidor debe reportar un total con
        `hit_cap=True` — hay que trocear por fechas hasta obtener `"eq"`.
        """
        if size > 100:
            msg = f"efts limita size a 100 por página; recibido {size}"
            raise DataQualityError(msg)
        params: dict[str, Any] = {"q": q, "from": int(offset), "size": int(size)}
        if forms:
            params["forms"] = forms if isinstance(forms, str) else ",".join(forms)
        if startdt:
            params["startdt"] = str(pd.Timestamp(startdt).date())
        if enddt:
            params["enddt"] = str(pd.Timestamp(enddt).date())
        if ciks:
            values = [ciks] if isinstance(ciks, str) else list(ciks)
            params["ciks"] = ",".join(normalize_cik(c) for c in values)
        payload = self._fetch_json(
            f"{SEC_FTS}/LATEST/search-index",
            kind=DataKind.FILINGS.value,
            ttl=TTL_FTS_S,
            params=params,
        )
        hits = (payload.get("hits") or {}).get("hits") or []
        total = (payload.get("hits") or {}).get("total") or {}
        rows = []
        for hit in hits:
            src = hit.get("_source", {})
            hit_id = str(hit.get("_id", ""))
            accession, _, document = hit_id.partition(":")
            rows.append(
                {
                    "accession": accession,
                    "document": document,
                    "form": src.get("form"),
                    "root_form": src.get("root_form"),
                    "file_date": src.get("file_date"),
                    "period_ending": src.get("period_ending"),
                    "ciks": ";".join(src.get("ciks", []) or []),
                    "display_names": ";".join(src.get("display_names", []) or []),
                }
            )
        df = pd.DataFrame(
            rows,
            columns=[
                "accession", "document", "form", "root_form",
                "file_date", "period_ending", "ciks", "display_names",
            ],
        )
        if len(df):
            df["file_date"] = pd.to_datetime(df["file_date"], errors="coerce")
        relation = str(total.get("relation", "eq"))
        hit_cap = relation != "eq"
        df.attrs.update(
            {
                "relation": relation,
                "hit_cap": hit_cap,
                "total": None if hit_cap else total.get("value"),
                "total_lower_bound": total.get("value"),
            }
        )
        return df
