"""Tipos núcleo compartidos por toda la plataforma.

Este módulo es un contrato estable: las firmas y campos aquí definidos no deben
modificarse sin actualizar `docs/ARCHITECTURE.md`, porque el resto de módulos
programa contra ellos.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

__all__ = [
    "Ticker",
    "CIK",
    "Session",
    "SurpriseBasis",
    "EarningsEvent",
    "Bar",
    "FundamentalFact",
    "EstimateSnapshot",
    "normalize_ticker",
    "normalize_cik",
]

# Símbolo normalizado. Convención del repo: punto, no guion ("BRK.B", no "BRK-B").
Ticker = str

# Central Index Key de la SEC, 10 dígitos con ceros a la izquierda.
CIK = str


class Session(StrEnum):
    """Momento del anuncio respecto a la sesión bursátil.

    La distinción es crítica: un anuncio AMC no es negociable hasta la sesión
    siguiente, y confundirlo con BMO introduce un día entero de look-ahead.
    """

    BMO = "bmo"
    """Before market open: antes de la apertura."""

    AMC = "amc"
    """After market close: tras el cierre."""

    DMH = "dmh"
    """During market hours: durante la sesión (infrecuente, suele ser una filtración
    o un anuncio no programado)."""

    UNKNOWN = "unknown"
    """Sin timestamp fiable. El consumidor debe decidir su política; la política por
    defecto del repo es tratarlo como AMC (la más conservadora)."""


class SurpriseBasis(StrEnum):
    """Denominador usado para estandarizar una sorpresa de resultados."""

    SIGMA = "sigma"
    """Desviación típica de las sorpresas históricas (SUE clásico, Foster et al. 1984)."""

    PRICE = "price"
    """Precio por acción al inicio de la ventana; robusto cuando el EPS es cercano a cero."""

    ABS_ESTIMATE = "abs_estimate"
    """Valor absoluto de la estimación; inestable con estimaciones próximas a cero."""

    ANALYST_DISPERSION = "analyst_dispersion"
    """Dispersión entre analistas; requiere estimaciones individuales."""


@dataclass(frozen=True, slots=True)
class EarningsEvent:
    """Un anuncio de resultados trimestrales.

    `announced_at` es el instante del anuncio en UTC. `period_end` es el cierre del
    trimestre fiscal reportado, que puede estar meses atrás: nunca deben confundirse
    al construir señales.
    """

    ticker: Ticker
    period_end: date
    announced_at: datetime
    session: Session
    fiscal_quarter: str
    cik: CIK | None = None
    eps_actual: float | None = None
    eps_estimate: float | None = None
    revenue_actual: float | None = None
    revenue_estimate: float | None = None
    source: str = "unknown"
    is_estimated_date: bool = False
    """True si la fecha del anuncio es una estimación del proveedor y no una fecha
    confirmada. Estos eventos deben poder excluirse de los backtests de evento."""

    @property
    def event_id(self) -> str:
        """Identificador estable de evento, apto como clave de join."""
        return f"{self.ticker}:{self.fiscal_quarter}:{self.period_end.isoformat()}"

    @property
    def eps_surprise(self) -> float | None:
        """Sorpresa bruta de EPS (actual - estimación), sin estandarizar."""
        if self.eps_actual is None or self.eps_estimate is None:
            return None
        return self.eps_actual - self.eps_estimate

    @property
    def revenue_surprise(self) -> float | None:
        """Sorpresa bruta de ingresos, sin estandarizar."""
        if self.revenue_actual is None or self.revenue_estimate is None:
            return None
        return self.revenue_actual - self.revenue_estimate


@dataclass(frozen=True, slots=True)
class Bar:
    """Barra OHLCV. `adj_close` incorpora splits y dividendos cuando el proveedor
    lo suministra; `close` es siempre el precio sin ajustar."""

    ticker: Ticker
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    adj_close: float | None = None


@dataclass(frozen=True, slots=True)
class FundamentalFact:
    """Un dato fundamental con su fecha de disponibilidad pública.

    `available_at` es lo que hace point-in-time a este registro: es el momento en que
    el dato pudo conocerse (fecha de presentación del informe), no el cierre del
    periodo al que se refiere.
    """

    ticker: Ticker
    concept: str
    value: float
    period_end: date
    available_at: datetime
    fiscal_period: str
    unit: str = "USD"
    form: str | None = None
    """Formulario SEC de origen (10-Q, 10-K, 8-K)."""
    accession: str | None = None
    """Número de acceso del filing; permite auditar el origen de cada cifra."""
    is_restated: bool = False
    """True si el valor procede de una reexpresión posterior. Un backtest honesto usa
    el valor tal y como se reportó por primera vez."""


@dataclass(frozen=True, slots=True)
class EstimateSnapshot:
    """Consenso de analistas tal y como estaba en `as_of`.

    Las revisiones de consenso solo son explotables si se conoce la foto histórica;
    usar el consenso final para un evento pasado es una de las formas más comunes de
    look-ahead en esta literatura.
    """

    ticker: Ticker
    period_end: date
    as_of: date
    eps_mean: float | None = None
    eps_median: float | None = None
    eps_std: float | None = None
    eps_high: float | None = None
    eps_low: float | None = None
    n_analysts: int | None = None
    revenue_mean: float | None = None
    revenue_std: float | None = None
    source: str = "unknown"


def normalize_ticker(raw: str) -> Ticker:
    """Normaliza un símbolo al formato canónico del repo.

    Unifica las variantes de clase de acción (`BRK-B`, `BRK/B`, `BRK.B`) en la forma
    con punto, que es la que usan SEC y la mayoría de datasets de índices.
    """
    t = raw.strip().upper()
    for sep in ("-", "/", " "):
        t = t.replace(sep, ".")
    while ".." in t:
        t = t.replace("..", ".")
    return t.strip(".")


def normalize_cik(raw: str | int) -> CIK:
    """Normaliza un CIK a 10 dígitos con ceros a la izquierda, como exige EDGAR."""
    digits = str(raw).strip().lstrip("CIKcik").lstrip(":").strip()
    if not digits.isdigit():
        msg = f"CIK no numérico: {raw!r}"
        raise ValueError(msg)
    return digits.zfill(10)
