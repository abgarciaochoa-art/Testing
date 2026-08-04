"""Fuentes vivas de composición del S&P 500 y motor de refresco *append-only*.

Contiene tres fuentes reales y el mecanismo que las integra en el fichero semilla
`data/seed/sp500_historical_components.csv` **sin reescribir jamás historia ya
registrada**.

Por qué append-only
-------------------
La composición histórica de un índice es un dato *point-in-time*: lo que importa no
es cuál fue la composición "verdadera" según la revisión más reciente del proveedor,
sino qué composición era **conocible** en cada fecha. Si un refresco reescribiera
snapshots pasados, cada re-ejecución del mismo backtest daría un resultado distinto
y, peor, incorporaría correcciones retroactivas que nadie pudo conocer en su
momento: es exactamente la forma de look-ahead que documentan

- Bir, Ivković & Sialm sobre bases reconstruidas, y de manera canónica
- Elton, Gruber & Blake (1996), "Survivorship Bias and Mutual Fund Performance",
  *Review of Financial Studies* 9(4),

y que en el caso del S&P 500 estudian Chen, Noronha & Singal (2004) al medir el
efecto de inclusión: un solo día mal fechado desplaza toda la ventana de evento.

Por eso `refresh()` solo **añade** fechas estrictamente posteriores al último
snapshot registrado. Las discrepancias en el solapamiento se reportan como avisos
(`RefreshReport.divergences`) para que un humano decida, nunca se aplican solas.

Estado de red del entorno de desarrollo
---------------------------------------
`GithubDatasetSource` y `GithubConstituentsSource` son **alcanzables** aquí y están
verificadas contra la red real. `WikipediaSource` y `SlickchartsSource` están
implementadas por completo pero su egress está bloqueado en este contenedor: sus
tests de red llevan la marca `@pytest.mark.network` y su *parser* se prueba offline
contra fixtures HTML.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd

from earnings_alpha.errors import (
    DataQualityError,
    ProviderUnavailable,
    RateLimited,
    UniverseError,
)
from earnings_alpha.types import CIK, Ticker, normalize_cik
from earnings_alpha.universe.identifiers import IdentifierRecord, clean_symbol

__all__ = [
    "ConstituentSnapshot",
    "GithubConstituentsSource",
    "GithubDatasetSource",
    "HistoricalUniverseSource",
    "RefreshReport",
    "SlickchartsSource",
    "UniverseSource",
    "WikipediaSource",
    "build_source",
    "default_sources",
    "history_frame_to_records",
    "parse_html_tables",
    "read_history_csv",
    "refresh_history_file",
]

DEFAULT_USER_AGENT = (
    "earnings-alpha/0.1 (investigación académica; contacto vía repositorio del proyecto)"
)

# Repositorio de referencia para la composición histórica diaria (1996→hoy).
GITHUB_HISTORY_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
)
# Lista vigente con metadatos GICS y CIK, misma estructura que el fichero semilla.
GITHUB_CONSTITUENTS_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
)
WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SLICKCHARTS_URL = "https://www.slickcharts.com/sp500"

_MIN_PLAUSIBLE_MEMBERS = 400
_MAX_PLAUSIBLE_MEMBERS = 520


# --------------------------------------------------------------------- tipos


@dataclass(frozen=True, slots=True)
class ConstituentSnapshot:
    """Composición del índice en una fecha, tal y como la publica una fuente.

    `as_of` es la fecha a la que la fuente atribuye la lista. Para fuentes que solo
    publican "la lista de hoy" (Wikipedia, Slickcharts) es la fecha de descarga, y
    por eso `is_estimated_date=True`: no sabemos el día exacto del último cambio.
    """

    as_of: dt.date
    tickers: tuple[Ticker, ...]
    source: str
    is_estimated_date: bool = False
    records: tuple[IdentifierRecord, ...] = ()
    """Metadatos por símbolo cuando la fuente los trae (CIK, sector, nombre)."""

    def __post_init__(self) -> None:
        if not self.tickers:
            msg = f"snapshot vacío desde {self.source!r}"
            raise DataQualityError(msg)

    @property
    def n_members(self) -> int:
        """Número de miembros del snapshot."""
        return len(self.tickers)

    def validate(self, *, strict: bool = True) -> list[str]:
        """Comprueba plausibilidad básica del snapshot.

        Un índice llamado "S&P 500" con 300 o con 900 miembros indica que el parser se
        ha desincronizado con el formato de la fuente, no un cambio de mercado.
        """
        issues: list[str] = []
        if len(set(self.tickers)) != len(self.tickers):
            dups = sorted({t for t in self.tickers if self.tickers.count(t) > 1})
            issues.append(f"símbolos duplicados: {dups[:10]}")
        if not _MIN_PLAUSIBLE_MEMBERS <= self.n_members <= _MAX_PLAUSIBLE_MEMBERS:
            issues.append(
                f"{self.n_members} miembros, fuera del rango plausible "
                f"[{_MIN_PLAUSIBLE_MEMBERS}, {_MAX_PLAUSIBLE_MEMBERS}]"
            )
        if issues and strict:
            msg = f"snapshot de {self.source!r} en {self.as_of}: " + "; ".join(issues)
            raise DataQualityError(msg)
        return issues


@runtime_checkable
class UniverseSource(Protocol):
    """Fuente capaz de devolver la composición vigente del índice."""

    name: str

    def available(self) -> bool:
        """True si la fuente es alcanzable ahora mismo (red y credenciales)."""
        ...

    def fetch_snapshot(self) -> ConstituentSnapshot:
        """Descarga la composición vigente. Lanza `ProviderUnavailable` si no puede."""
        ...


@runtime_checkable
class HistoricalUniverseSource(UniverseSource, Protocol):
    """Fuente que además publica la serie histórica completa de composiciones."""

    def fetch_history(self) -> pd.DataFrame:
        """DataFrame `(date, tickers)` con una fila por snapshot, `date` ordenada."""
        ...


# ------------------------------------------------------------------ utilidades


def _http_get(
    url: str,
    *,
    provider: str,
    timeout: float = 30.0,
    user_agent: str = DEFAULT_USER_AGENT,
    accept: str = "text/html,text/csv,*/*",
) -> str:
    """GET con traducción de fallos al vocabulario de errores del repo.

    Traduce cualquier problema de red o HTTP a `ProviderUnavailable` (o `RateLimited`
    ante un 429), en lugar de propagar excepciones de `requests` que el resto de la
    plataforma no sabe interpretar.
    """
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - requests es dependencia dura
        raise ProviderUnavailable(provider, "falta la dependencia 'requests'") from exc

    headers = {"User-Agent": user_agent, "Accept": accept}
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
    except Exception as exc:  # cualquier fallo de red se traduce a indisponibilidad
        raise ProviderUnavailable(provider, f"error de red al pedir {url}: {exc}") from exc
    if resp.status_code == 429:
        retry = resp.headers.get("Retry-After")
        raise RateLimited(provider, float(retry) if retry and retry.isdigit() else None)
    if resp.status_code >= 400:
        raise ProviderUnavailable(
            provider, f"HTTP {resp.status_code} al pedir {url} (egress bloqueado o URL caducada)"
        )
    if not resp.text.strip():
        raise ProviderUnavailable(provider, f"respuesta vacía de {url}")
    return resp.text


class _TableParser(HTMLParser):
    """Extractor de tablas HTML basado solo en la biblioteca estándar.

    Se evita `pandas.read_html` a propósito: exige `lxml`/`html5lib`, que no son
    dependencias del proyecto y no están instaladas en el entorno de desarrollo. El
    parser acumula el texto de cada `<td>`/`<th>`, ignorando el marcado interno
    (enlaces, `<sup>` de referencias, banderas), que es justo lo que sobra.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[str]]] = []
        self.table_attrs: list[dict[str, str]] = []
        self._table_stack: list[list[list[str]]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._colspan = 1
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self._table_stack.append([])
            self.table_attrs.append({k: (v or "") for k, v in attrs})
        elif tag == "tr" and self._table_stack:
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []
            attr = dict(attrs).get("colspan") or "1"
            self._colspan = int(attr) if attr.isdigit() and 1 <= int(attr) <= 20 else 1
        elif tag in ("sup", "style", "script") and self._cell is not None:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "table" and self._table_stack:
            table = self._table_stack.pop()
            self.tables.append(table)
        elif tag == "tr" and self._row is not None and self._table_stack:
            if self._row:
                self._table_stack[-1].append(self._row)
            self._row = None
        elif tag in ("td", "th") and self._cell is not None and self._row is not None:
            text = re.sub(r"\s+", " ", "".join(self._cell)).strip()
            # `colspan` se expande replicando el texto: mantiene alineadas las
            # columnas de las tablas de Wikipedia con cabecera de dos niveles.
            self._row.extend([text] * self._colspan)
            self._cell = None
            self._colspan = 1
        elif tag in ("sup", "style", "script") and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._cell is not None and not self._skip_depth:
            self._cell.append(data)


def _dedupe_header(header: Sequence[str]) -> list[str]:
    """Nombres de columna únicos: los repetidos reciben sufijo `_2`, `_3`…

    Necesario porque una cabecera con `colspan` genera nombres duplicados
    (`Added`, `Added`) al expandirla.
    """
    seen: dict[str, int] = {}
    out: list[str] = []
    for raw in header:
        name = raw or "col"
        seen[name] = seen.get(name, 0) + 1
        out.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
    return out


def parse_html_tables(html: str) -> list[pd.DataFrame]:
    """Convierte todas las `<table>` del documento en DataFrames.

    La primera fila se toma como cabecera y solo se conservan las filas con el mismo
    número de celdas: las cabeceras de segundo nivel y las filas de sección quedan
    descartadas. Se expande `colspan`; **`rowspan` no se reconstruye**, así que las
    tablas que lo usan intensivamente pueden perder filas — es una limitación
    aceptada a cambio de no depender de `lxml`.

    Devuelve lista vacía si no hay tablas; quien llame decide si eso es un error.
    """
    parser = _TableParser()
    parser.feed(html)
    parser.close()
    frames: list[pd.DataFrame] = []
    for rows in parser.tables:
        if not rows:
            continue
        header = rows[0]
        body = [r for r in rows[1:] if len(r) == len(header)]
        if not body:
            frames.append(pd.DataFrame(rows))
            continue
        frames.append(pd.DataFrame(body, columns=_dedupe_header(header)))
    return frames


def _table_with_columns(frames: Sequence[pd.DataFrame], required: Iterable[str]) -> pd.DataFrame:
    """Primera tabla que contenga todas las columnas exigidas (comparación laxa)."""
    want = {c.lower().strip() for c in required}
    for frame in frames:
        cols = {str(c).lower().strip() for c in frame.columns}
        if want.issubset(cols):
            return frame
    msg = f"ninguna tabla contiene las columnas {sorted(want)}"
    raise DataQualityError(msg)


def _pick_column(frame: pd.DataFrame, *candidates: str) -> str:
    """Nombre real de la primera columna que coincida (sin distinguir mayúsculas)."""
    lower = {str(c).lower().strip(): str(c) for c in frame.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    msg = f"no encuentro ninguna de las columnas {candidates} en {list(frame.columns)}"
    raise DataQualityError(msg)


# ------------------------------------------------------------------- fuentes


@dataclass(slots=True)
class GithubDatasetSource:
    """Composición histórica diaria desde un dataset público alojado en GitHub.

    Por defecto apunta a `fja05680/sp500`, que publica un CSV `date,tickers` con la
    composición del índice desde 1996 — la misma estructura del fichero semilla del
    repo, lo que permite un refresco append-only trivial.

    Es la única fuente **verificada contra la red real** en el entorno de desarrollo:
    `raw.githubusercontent.com` sí es alcanzable a través del proxy de egress.
    """

    url: str = GITHUB_HISTORY_URL
    name: str = "github_sp500_history"
    timeout: float = 30.0
    user_agent: str = DEFAULT_USER_AGENT

    def available(self) -> bool:
        """Comprueba alcanzabilidad real haciendo una descarga corta."""
        try:
            _http_get(self.url, provider=self.name, timeout=self.timeout, accept="text/csv,*/*")
        except (ProviderUnavailable, RateLimited):
            return False
        return True

    def fetch_history(self) -> pd.DataFrame:
        """Descarga la serie histórica completa como DataFrame `(date, tickers)`."""
        text = _http_get(
            self.url, provider=self.name, timeout=self.timeout, accept="text/csv,*/*"
        )
        try:
            frame = pd.read_csv(io.StringIO(text))
        except Exception as exc:  # cualquier fallo de parseo es un problema de calidad
            raise DataQualityError(f"{self.name}: CSV ilegible ({exc})") from exc
        if not {"date", "tickers"}.issubset(frame.columns):
            msg = f"{self.name}: se esperaban columnas (date, tickers), hay {list(frame.columns)}"
            raise DataQualityError(msg)
        frame = frame.loc[:, ["date", "tickers"]].copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        if frame["date"].isna().any():
            msg = f"{self.name}: fechas no parseables en el CSV remoto"
            raise DataQualityError(msg)
        return frame.sort_values("date").reset_index(drop=True)

    def fetch_snapshot(self) -> ConstituentSnapshot:
        """Último snapshot publicado por el dataset."""
        frame = self.fetch_history()
        last = frame.iloc[-1]
        tickers = tuple(sorted({clean_symbol(t) for t in str(last["tickers"]).split(",") if t}))
        snap = ConstituentSnapshot(
            as_of=pd.Timestamp(last["date"]).date(), tickers=tickers, source=self.name
        )
        snap.validate(strict=False)
        return snap


@dataclass(slots=True)
class GithubConstituentsSource:
    """Lista vigente con metadatos GICS y CIK desde `datasets/s-and-p-500-companies`.

    No aporta historia, pero sí lo que el fichero de composición no tiene: sector,
    sub-industria y **CIK**, imprescindibles para el mapa ticker↔CIK y para
    neutralizar factores por sector. Alcanzable en el entorno de desarrollo.
    """

    url: str = GITHUB_CONSTITUENTS_URL
    name: str = "github_sp500_constituents"
    timeout: float = 30.0
    user_agent: str = DEFAULT_USER_AGENT

    def available(self) -> bool:
        """Comprueba alcanzabilidad real de la URL del dataset."""
        try:
            _http_get(self.url, provider=self.name, timeout=self.timeout, accept="text/csv,*/*")
        except (ProviderUnavailable, RateLimited):
            return False
        return True

    def fetch_snapshot(self) -> ConstituentSnapshot:
        """Composición vigente con metadatos por símbolo."""
        text = _http_get(
            self.url, provider=self.name, timeout=self.timeout, accept="text/csv,*/*"
        )
        frame = pd.read_csv(io.StringIO(text))
        return self.parse_frame(frame, source=self.name)

    @staticmethod
    def parse_frame(frame: pd.DataFrame, *, source: str) -> ConstituentSnapshot:
        """Convierte el CSV de constituyentes en snapshot (probado offline)."""
        sym_col = _pick_column(frame, "Symbol", "Ticker")
        records: list[IdentifierRecord] = []
        tickers: list[Ticker] = []
        for _, row in frame.iterrows():
            ticker = clean_symbol(str(row[sym_col]))
            tickers.append(ticker)
            records.append(
                IdentifierRecord(
                    ticker=ticker,
                    cik=_maybe_cik(row, frame),
                    name=_maybe_str(row, frame, "Security", "Name", "Company"),
                    sector=_maybe_str(row, frame, "GICS Sector", "Sector"),
                    sub_industry=_maybe_str(row, frame, "GICS Sub-Industry", "Sub-Industry"),
                    date_added=_maybe_date(row, frame, "Date added", "Date first added"),
                    source=source,
                )
            )
        snap = ConstituentSnapshot(
            as_of=dt.date.today(),
            tickers=tuple(sorted(set(tickers))),
            source=source,
            is_estimated_date=True,
            records=tuple(records),
        )
        snap.validate(strict=False)
        return snap


def _maybe_str(row: pd.Series, frame: pd.DataFrame, *candidates: str) -> str | None:
    """Valor de texto de la primera columna candidata presente, o None."""
    lower = {str(c).lower().strip(): c for c in frame.columns}
    for cand in candidates:
        col = lower.get(cand.lower())
        if col is not None and not pd.isna(row[col]):
            text = str(row[col]).strip()
            if text:
                return text
    return None


def _maybe_cik(row: pd.Series, frame: pd.DataFrame) -> CIK | None:
    """CIK normalizado a 10 dígitos, o None si la columna falta o es ilegible."""
    raw = _maybe_str(row, frame, "CIK", "Cik", "central_index_key")
    if raw is None:
        return None
    digits = re.sub(r"\D", "", raw)
    if not digits:
        return None
    return normalize_cik(digits)


def _maybe_date(row: pd.Series, frame: pd.DataFrame, *candidates: str) -> dt.date | None:
    """Fecha de alta en el índice, o None. Nunca lanza: es un metadato opcional."""
    raw = _maybe_str(row, frame, *candidates)
    if raw is None:
        return None
    ts = pd.to_datetime(raw[:10], errors="coerce")
    return None if ts is pd.NaT or pd.isna(ts) else ts.date()


@dataclass(slots=True)
class WikipediaSource:
    """Lista vigente desde el artículo "List of S&P 500 companies" de Wikipedia.

    Es la fuente de referencia de facto para el snapshot actual (incluye CIK y GICS)
    y publica además una tabla de cambios recientes. **Bloqueada por el proxy de
    egress en el entorno de desarrollo**: implementada al completo, con el *parser*
    probado offline contra un fixture HTML y la descarga marcada `@pytest.mark.network`.
    """

    url: str = WIKIPEDIA_URL
    name: str = "wikipedia"
    timeout: float = 30.0
    user_agent: str = DEFAULT_USER_AGENT

    def available(self) -> bool:
        """True si el artículo es descargable ahora mismo."""
        try:
            _http_get(self.url, provider=self.name, timeout=self.timeout)
        except (ProviderUnavailable, RateLimited):
            return False
        return True

    def fetch_snapshot(self) -> ConstituentSnapshot:
        """Composición vigente con CIK y clasificación GICS."""
        html = _http_get(self.url, provider=self.name, timeout=self.timeout)
        return self.parse_html(html, source=self.name)

    @staticmethod
    def parse_html(html: str, *, source: str = "wikipedia") -> ConstituentSnapshot:
        """Extrae la tabla de constituyentes del HTML (probado offline)."""
        frames = parse_html_tables(html)
        if not frames:
            msg = f"{source}: el HTML no contiene ninguna tabla"
            raise DataQualityError(msg)
        frame = _table_with_columns(frames, ["Symbol"])
        return GithubConstituentsSource.parse_frame(frame, source=source)

    @staticmethod
    def parse_changes(html: str) -> pd.DataFrame:
        """Extrae la tabla de "Selected changes to the list" si está presente.

        Devuelve `(date, added, removed)`. Es información útil para cotejar las altas
        y bajas detectadas por diferencia de snapshots, pero **no sustituye** al
        histórico: Wikipedia solo conserva cambios recientes y su fecha es la del
        anuncio, no siempre la de efectividad.
        """
        for frame in parse_html_tables(html):
            cols = {str(c).lower().strip().split("_")[0] for c in frame.columns}
            # Exigir una columna llamada exactamente "date" evita confundirla con la
            # tabla de constituyentes, que trae "Date added".
            if "date" in cols and ("added" in cols or "removed" in cols):
                out = frame.copy()
                out.columns = [str(c) for c in out.columns]
                return out
        return pd.DataFrame(columns=["date", "added", "removed"])


@dataclass(slots=True)
class SlickchartsSource:
    """Lista vigente y **pesos por capitalización** desde slickcharts.com.

    Aporta algo que ni Wikipedia ni el dataset de GitHub tienen: el peso de cada
    componente en el índice, necesario para construir un benchmark ponderado por
    capitalización coherente con el universo. **Bloqueada por el proxy de egress
    aquí**; el parser se prueba offline.
    """

    url: str = SLICKCHARTS_URL
    name: str = "slickcharts"
    timeout: float = 30.0
    user_agent: str = DEFAULT_USER_AGENT

    def available(self) -> bool:
        """True si la página es descargable ahora mismo."""
        try:
            _http_get(self.url, provider=self.name, timeout=self.timeout)
        except (ProviderUnavailable, RateLimited):
            return False
        return True

    def fetch_snapshot(self) -> ConstituentSnapshot:
        """Composición vigente según slickcharts."""
        html = _http_get(self.url, provider=self.name, timeout=self.timeout)
        return self.parse_html(html, source=self.name)

    @staticmethod
    def parse_html(html: str, *, source: str = "slickcharts") -> ConstituentSnapshot:
        """Extrae símbolos (y nombres) de la tabla de componentes."""
        frames = parse_html_tables(html)
        if not frames:
            msg = f"{source}: el HTML no contiene ninguna tabla"
            raise DataQualityError(msg)
        frame = _table_with_columns(frames, ["Symbol"])
        sym_col = _pick_column(frame, "Symbol")
        records: list[IdentifierRecord] = []
        tickers: list[Ticker] = []
        for _, row in frame.iterrows():
            ticker = clean_symbol(str(row[sym_col]))
            tickers.append(ticker)
            records.append(
                IdentifierRecord(
                    ticker=ticker,
                    name=_maybe_str(row, frame, "Company", "Name"),
                    source=source,
                )
            )
        snap = ConstituentSnapshot(
            as_of=dt.date.today(),
            tickers=tuple(sorted(set(tickers))),
            source=source,
            is_estimated_date=True,
            records=tuple(records),
        )
        snap.validate(strict=False)
        return snap

    @staticmethod
    def parse_weights(html: str) -> pd.Series:
        """Peso en el índice por símbolo, en tanto por uno."""
        frames = parse_html_tables(html)
        frame = _table_with_columns(frames, ["Symbol", "Weight"])
        sym_col = _pick_column(frame, "Symbol")
        w_col = _pick_column(frame, "Weight")
        weights = (
            frame[w_col].astype(str).str.replace("%", "", regex=False).str.strip().astype(float)
            / 100.0
        )
        idx = [clean_symbol(str(s)) for s in frame[sym_col]]
        return pd.Series(weights.to_numpy(), index=pd.Index(idx, name="ticker"), name="weight")


_SOURCE_REGISTRY: dict[str, type] = {
    "github": GithubDatasetSource,
    "github_history": GithubDatasetSource,
    "github_constituents": GithubConstituentsSource,
    "wikipedia": WikipediaSource,
    "slickcharts": SlickchartsSource,
}


def build_source(name: str, **kwargs: object) -> UniverseSource:
    """Instancia una fuente por nombre. Lanza `UniverseError` si el nombre es falso.

    Fallar con nombres desconocidos es deliberado: un `refresh(sources=["wikipedía"])`
    mal escrito no debe degradarse a "no hice nada" en silencio.
    """
    key = name.strip().lower()
    if key not in _SOURCE_REGISTRY:
        msg = f"fuente desconocida {name!r}; disponibles: {sorted(_SOURCE_REGISTRY)}"
        raise UniverseError(msg)
    return _SOURCE_REGISTRY[key](**kwargs)  # type: ignore[return-value]


def default_sources() -> list[UniverseSource]:
    """Fuentes por defecto, en orden de prioridad decreciente.

    Primero la histórica (fecha exacta del snapshot), después las que solo publican
    "la lista de hoy" y por tanto llevan `is_estimated_date=True`.
    """
    return [
        GithubDatasetSource(),
        WikipediaSource(),
        SlickchartsSource(),
    ]


# ------------------------------------------------------- lectura del histórico


def read_history_csv(path: Path | str) -> pd.DataFrame:
    """Lee el CSV de composición histórica `(date, tickers)`.

    Lanza `ProviderUnavailable` si el fichero no existe: la ausencia del dato semilla
    es indisponibilidad de un proveedor, no un DataFrame vacío.
    """
    p = Path(path)
    if not p.exists():
        raise ProviderUnavailable("sp500_seed_history", f"no existe el fichero {p}")
    frame = pd.read_csv(p)
    if not {"date", "tickers"}.issubset(frame.columns):
        msg = f"{p}: se esperaban columnas (date, tickers), hay {list(frame.columns)}"
        raise DataQualityError(msg)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["date"].isna().any():
        msg = f"{p}: hay fechas no parseables"
        raise DataQualityError(msg)
    return frame


def history_frame_to_records(frame: pd.DataFrame) -> list[tuple[dt.date, frozenset[Ticker]]]:
    """Normaliza un DataFrame `(date, tickers)` a `[(fecha, conjunto de símbolos)]`.

    Política ante fechas duplicadas: **gana la última fila del fichero**. Verificado
    contra el dato real — en `2022-06-21` el fichero semilla trae dos filas y solo la
    segunda enlaza con el snapshot siguiente (`KDP` entra, `UA`/`UAA` salen); en
    `2023-05-14` las dos filas son idénticas una vez normalizado `BF-B`→`BF.B`.
    """
    out: dict[dt.date, frozenset[Ticker]] = {}
    # El fichero real tiene ~1,6 millones de celdas de símbolo pero solo ~1.100
    # símbolos distintos: memoizar la normalización reduce el parseo de segundos a
    # décimas sin cambiar el resultado.
    cache: dict[str, Ticker] = {}
    for raw_date, raw_tickers in zip(frame["date"], frame["tickers"], strict=True):
        day = pd.Timestamp(raw_date).date()
        members: set[Ticker] = set()
        for raw in str(raw_tickers).split(","):
            token = raw.strip()
            if not token:
                continue
            clean = cache.get(token)
            if clean is None:
                clean = clean_symbol(token)
                cache[token] = clean
            members.add(clean)
        if not members:
            msg = f"snapshot vacío en {day}"
            raise DataQualityError(msg)
        out[day] = frozenset(members)
    return sorted(out.items())


# --------------------------------------------------------------- informe y refresh


@dataclass(slots=True)
class RefreshReport:
    """Resultado de un refresco append-only del fichero de composición.

    Está pensado para ser auditable: dice qué fuentes se intentaron, cuáles
    fallaron y por qué, qué fechas se añadieron, qué altas y bajas implican, y qué
    discrepancias hay con la historia ya registrada (que **no** se han aplicado).
    """

    path: Path
    sources_tried: tuple[str, ...] = ()
    sources_used: tuple[str, ...] = ()
    sources_failed: dict[str, str] = field(default_factory=dict)
    rows_before: int = 0
    rows_after: int = 0
    dates_appended: tuple[dt.date, ...] = ()
    additions: pd.DataFrame = field(default_factory=lambda: _empty_changes())
    deletions: pd.DataFrame = field(default_factory=lambda: _empty_changes())
    skipped_dates: tuple[dt.date, ...] = ()
    """Fechas que las fuentes conocen, el fichero no registra, y **no** se han
    insertado por ser anteriores al último snapshot: reescribirlas rompería la
    reproducibilidad point-in-time."""
    divergences: pd.DataFrame = field(default_factory=lambda: _empty_divergences())
    warnings: tuple[str, ...] = ()
    dry_run: bool = False

    @property
    def n_appended(self) -> int:
        """Número de snapshots nuevos escritos (o que se habrían escrito en dry-run)."""
        return len(self.dates_appended)

    @property
    def changed(self) -> bool:
        """True si el refresco modificó el fichero."""
        return self.n_appended > 0 and not self.dry_run

    def summary(self) -> str:
        """Resumen legible de una línea por bloque, apto para log o CLI."""
        lines = [
            f"refresh {self.path.name}: {self.rows_before} → {self.rows_after} filas"
            + (" (dry-run)" if self.dry_run else ""),
            f"  fuentes usadas: {', '.join(self.sources_used) or 'ninguna'}",
        ]
        if self.sources_failed:
            for name, why in self.sources_failed.items():
                lines.append(f"  fuente fallida {name}: {why}")
        if self.dates_appended:
            lines.append(
                f"  fechas añadidas: {self.dates_appended[0]} … {self.dates_appended[-1]}"
                f" ({self.n_appended})"
            )
        lines.append(f"  altas: {len(self.additions)}  bajas: {len(self.deletions)}")
        if len(self.divergences):
            lines.append(
                f"  AVISO: {len(self.divergences)} fechas con divergencia respecto a la "
                "historia registrada (no se han reescrito)"
            )
        for w in self.warnings:
            lines.append(f"  aviso: {w}")
        return "\n".join(lines)


def _empty_changes() -> pd.DataFrame:
    """DataFrame vacío con el esquema canónico de cambios de índice."""
    return pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"),
                         "ticker": pd.Series(dtype="object"),
                         "action": pd.Series(dtype="object")})


def _empty_divergences() -> pd.DataFrame:
    """DataFrame vacío con el esquema de divergencias fuente↔fichero."""
    return pd.DataFrame(
        {
            "date": pd.Series(dtype="datetime64[ns]"),
            "source": pd.Series(dtype="object"),
            "only_local": pd.Series(dtype="object"),
            "only_remote": pd.Series(dtype="object"),
        }
    )


def _changes_between(
    previous: frozenset[Ticker], current: frozenset[Ticker], day: dt.date
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Altas y bajas entre dos composiciones consecutivas."""
    adds = [{"date": day, "ticker": t, "action": "add"} for t in sorted(current - previous)]
    dels = [{"date": day, "ticker": t, "action": "delete"} for t in sorted(previous - current)]
    return adds, dels


def refresh_history_file(
    path: Path | str,
    sources: Sequence[UniverseSource] | None = None,
    *,
    dry_run: bool = False,
    validate_snapshots: bool = True,
    check_overlap: bool = True,
    max_overlap_checks: int = 400,
) -> RefreshReport:
    """Añade al fichero de composición los snapshots posteriores al último registrado.

    Garantías:

    - **Append-only real**: el fichero se abre en modo `"a"`; los bytes existentes no
      se tocan. Solo se escriben fechas *estrictamente posteriores* al último
      snapshot registrado.
    - **Sin invención**: si ninguna fuente responde, lanza `ProviderUnavailable` con
      el detalle de cada fallo, en vez de devolver un informe vacío que parezca éxito.
    - **Divergencias visibles**: las diferencias en el tramo solapado se reportan,
      nunca se aplican.

    `dry_run=True` calcula todo el informe sin escribir, que es la forma correcta de
    revisar un refresco antes de tocar un fichero del que dependen los backtests.
    """
    p = Path(path)
    existing = read_history_csv(p)
    records = history_frame_to_records(existing)
    if not records:
        msg = f"{p}: el fichero no contiene snapshots"
        raise DataQualityError(msg)
    last_date, last_members = records[-1]
    by_date = dict(records)

    srcs = list(sources) if sources is not None else default_sources()
    if not srcs:
        msg = "refresh sin fuentes: pasa al menos una o usa las de por defecto"
        raise UniverseError(msg)

    tried: list[str] = []
    used: list[str] = []
    failed: dict[str, str] = {}
    warnings: list[str] = []
    fetched: dict[dt.date, tuple[frozenset[Ticker], str]] = {}
    divergence_rows: list[dict[str, object]] = []
    refused: set[dt.date] = set()

    for src in srcs:
        name = getattr(src, "name", src.__class__.__name__)
        tried.append(name)
        try:
            rows: list[tuple[dt.date, frozenset[Ticker], bool]] = []
            if hasattr(src, "fetch_history"):
                frame = src.fetch_history()  # type: ignore[attr-defined]
                rows = [(d, m, False) for d, m in history_frame_to_records(frame)]
            else:
                snap = src.fetch_snapshot()
                if validate_snapshots:
                    snap.validate(strict=True)
                rows = [(snap.as_of, frozenset(snap.tickers), snap.is_estimated_date)]
        except (ProviderUnavailable, RateLimited, DataQualityError) as exc:
            failed[name] = str(exc)
            continue

        used.append(name)
        n_new = 0
        for day, members, estimated in rows:
            if day <= last_date:
                # La fuente conoce un snapshot que el fichero no registra, pero
                # insertarlo cambiaría retroactivamente la pertenencia de fechas ya
                # usadas en backtests. Se reporta, no se aplica.
                if day not in by_date:
                    refused.add(day)
                continue
            if day in fetched:  # una fuente anterior ya cubrió ese día: manda la primera
                continue
            if estimated and day > dt.date.today():
                warnings.append(f"{name}: snapshot con fecha futura {day}, descartado")
                continue
            fetched[day] = (members, name)
            n_new += 1
        if n_new == 0:
            warnings.append(f"{name}: sin fechas nuevas posteriores a {last_date}")

        if check_overlap:
            overlap = [(d, m) for d, m, _ in rows if d in by_date][-max_overlap_checks:]
            for day, members in overlap:
                local = by_date[day]
                only_local = sorted(local - members)
                only_remote = sorted(members - local)
                if only_local or only_remote:
                    divergence_rows.append(
                        {
                            "date": day,
                            "source": name,
                            "only_local": ",".join(only_local),
                            "only_remote": ",".join(only_remote),
                        }
                    )

    if not used:
        raise ProviderUnavailable(
            "universe_refresh",
            "ninguna fuente disponible: "
            + "; ".join(f"{k}: {v}" for k, v in failed.items()),
        )

    new_dates = sorted(fetched)
    add_rows: list[dict[str, object]] = []
    del_rows: list[dict[str, object]] = []
    prev = last_members
    lines: list[tuple[str, str]] = []
    for day in new_dates:
        members, _src = fetched[day]
        if validate_snapshots and not (
            _MIN_PLAUSIBLE_MEMBERS <= len(members) <= _MAX_PLAUSIBLE_MEMBERS
        ):
            msg = f"snapshot de {day} con {len(members)} miembros, fuera del rango plausible"
            raise DataQualityError(msg)
        adds, dels = _changes_between(prev, members, day)
        add_rows.extend(adds)
        del_rows.extend(dels)
        lines.append((day.isoformat(), ",".join(sorted(members))))
        prev = members

    if lines and not dry_run:
        _append_lines(p, lines)

    skipped = tuple(sorted(refused))
    if skipped:
        warnings.append(
            f"{len(skipped)} fechas anteriores al último snapshot registrado existen en las "
            f"fuentes y NO se han insertado (política append-only): "
            f"{skipped[0]} … {skipped[-1]}"
        )
    additions = pd.DataFrame(add_rows, columns=["date", "ticker", "action"])
    deletions = pd.DataFrame(del_rows, columns=["date", "ticker", "action"])
    for frame in (additions, deletions):
        if not frame.empty:
            frame["date"] = pd.to_datetime(frame["date"])
    divergences = pd.DataFrame(
        divergence_rows, columns=["date", "source", "only_local", "only_remote"]
    )
    if not divergences.empty:
        divergences["date"] = pd.to_datetime(divergences["date"])

    return RefreshReport(
        path=p,
        sources_tried=tuple(tried),
        sources_used=tuple(used),
        sources_failed=failed,
        rows_before=len(records),
        rows_after=len(records) + (0 if dry_run else len(lines)),
        dates_appended=tuple(new_dates),
        additions=additions if not additions.empty else _empty_changes(),
        deletions=deletions if not deletions.empty else _empty_changes(),
        skipped_dates=skipped,
        divergences=divergences if not divergences.empty else _empty_divergences(),
        warnings=tuple(warnings),
        dry_run=dry_run,
    )


def _append_lines(path: Path, lines: Sequence[tuple[str, str]]) -> None:
    """Escribe filas al final del CSV sin reescribir nada de lo anterior.

    Se comprueba y repara el salto de línea final antes de anexar; un fichero sin
    `\\n` final haría que la primera fila nueva se fusionara con la última antigua,
    corrompiendo justo el snapshot más reciente.
    """
    needs_newline = False
    with path.open("rb") as fh:
        if path.stat().st_size:
            fh.seek(-1, os.SEEK_END)
            needs_newline = fh.read(1) != b"\n"
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    for day, tickers in lines:
        writer.writerow([day, tickers])
    with path.open("a", encoding="utf-8", newline="") as fh:
        if needs_newline:
            fh.write("\n")
        fh.write(buf.getvalue())
