"""Universo S&P 500 point-in-time: pertenencia, altas/bajas e identificadores.

Punto de entrada del módulo `universe` (sección 3.1 de `docs/ARCHITECTURE.md`).

    from earnings_alpha.universe import SP500Universe

    u = SP500Universe()
    u.members_on(date(2008, 6, 30))          # composición conocible ese día
    u.membership_panel(start, end)           # panel booleano date x ticker
    u.additions(start, end)                  # altas del índice (efecto de inclusión)
    u.refresh(["github"])                    # append-only, nunca reescribe historia

El principio rector es que el universo en `t` sea la pertenencia **real** en `t`:
todas las consultas por fecha resuelven con el snapshot anterior más próximo y
lanzan `InsufficientHistory` antes del inicio del histórico, en vez de degradarse a
la lista de hoy y contaminar el estudio con sesgo de supervivencia.
"""

from earnings_alpha.universe.constituents import (
    DateLike,
    HistorySnapshotIndex,
    SP500Universe,
    UniverseProvider,
)
from earnings_alpha.universe.identifiers import (
    IdentifierMap,
    IdentifierRecord,
    SymbolChangeCandidate,
    TickerSpan,
    clean_symbol,
    detect_symbol_changes,
    merge_identifier_maps,
    ticker_spans,
)
from earnings_alpha.universe.sources import (
    ConstituentSnapshot,
    GithubConstituentsSource,
    GithubDatasetSource,
    HistoricalUniverseSource,
    RefreshReport,
    SlickchartsSource,
    UniverseSource,
    WikipediaSource,
    build_source,
    default_sources,
    parse_html_tables,
    read_history_csv,
    refresh_history_file,
)

__all__ = [
    "ConstituentSnapshot",
    "DateLike",
    "GithubConstituentsSource",
    "GithubDatasetSource",
    "HistoricalUniverseSource",
    "HistorySnapshotIndex",
    "IdentifierMap",
    "IdentifierRecord",
    "RefreshReport",
    "SP500Universe",
    "SlickchartsSource",
    "SymbolChangeCandidate",
    "TickerSpan",
    "UniverseProvider",
    "UniverseSource",
    "WikipediaSource",
    "build_source",
    "clean_symbol",
    "default_sources",
    "detect_symbol_changes",
    "merge_identifier_maps",
    "parse_html_tables",
    "read_history_csv",
    "refresh_history_file",
    "ticker_spans",
]
