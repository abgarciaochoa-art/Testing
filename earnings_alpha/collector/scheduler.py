"""Planificación y orquestación de la recolección diaria (`collector.scheduler`).

El presupuesto de peticiones es la restricción que gobierna todo el diseño:
fotografiar la cadena de opciones de los ~503 tickers del S&P 500 con yfinance
(≈1 símbolo/segundo, `data_sources.md` §3.2) costaría horas y un baneo casi
seguro. La observación que lo resuelve es que **el valor de una cadena está
concentrado alrededor de los anuncios de resultados**: las features del ángulo B
(`oi_buildup`, `vol_spread`, `iv_skew_25delta`…) se calculan en ventanas
[T-30, T+5] en torno al evento. De ahí la cola de prioridad:

1. **Tickers con resultados en las próximas `horizon_days` (4 semanas)** según
   el calendario capturado: prioridad máxima, ordenados por proximidad del
   anuncio. Con ~500 empresas y 4 anuncios/año, en una semana cualquiera hay
   ~40-90 en ventana; en plena temporada de resultados, 150-250.
2. **Una muestra rotatoria del resto** como línea base fuera de ventana: sirve
   de grupo de control para los detectores (una señal que también "detecta"
   fuera de ventana es ruido) y garantiza que ningún ticker pasa meses sin
   fotografiar. La rotación es determinista —función de la fecha, no de un RNG
   con estado— para que un re-arranque el mismo día produzca el mismo plan
   (idempotencia de la pasada completa, no solo del almacén).

Los fallos se reintentan con backoff exponencial y jitter (reutilizando
`data.base.RetryPolicy`; Brooker, *Exponential Backoff and Jitter*, AWS 2015) y
todo intento queda en un registro append-only (`CollectionLog`) del que salen
los informes de días fallidos. Los huecos de panel se miden contra el calendario
bursátil con `SnapshotStore.missing_sessions`: un hueco en `option_chain` es
un día perdido para siempre y debe doler a la vista.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import random
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

import pandas as pd

from earnings_alpha.collector.snapshot import (
    PASS_CLOSE,
    PASS_MORNING,
    FinraShortInterestSource,
    OptionChainSource,
    RegShoSource,
    snapshot_consensus,
    snapshot_earnings_calendar,
    snapshot_off_exchange,
    snapshot_open_interest,
    snapshot_option_chain,
    snapshot_short_interest,
)
from earnings_alpha.collector.storage import SnapshotStore
from earnings_alpha.config import Settings, get_settings
from earnings_alpha.data.base import (
    Clock,
    HttpStatusError,
    RateLimited,
    RetryPolicy,
    SystemClock,
    TransportError,
)
from earnings_alpha.data.cache import SystemWallClock, WallClock
from earnings_alpha.data.estimates import EstimatesProviderBase
from earnings_alpha.errors import (
    ConfigError,
    DataQualityError,
    EarningsAlphaError,
    InsufficientHistory,
    ProviderUnavailable,
)
from earnings_alpha.pit import TradingCalendar, get_calendar, utc_to_eastern
from earnings_alpha.types import Ticker, normalize_ticker

__all__ = [  # noqa: RUF022 - orden temático, no alfabético
    # cola de prioridad
    "REASON_EVENT",
    "REASON_BASELINE",
    "PrioritizedTicker",
    "build_priority_queue",
    # presupuesto y plan
    "RequestBudget",
    "CollectionPlan",
    "plan_collection",
    # registro
    "LogEntry",
    "CollectionLog",
    # reintentos
    "RETRYABLE_ERRORS",
    "call_with_retries",
    # orquestador
    "RunReport",
    "DailyCollector",
]

logger = logging.getLogger(__name__)

T = TypeVar("T")

REASON_EVENT = "earnings_window"
"""El ticker tiene un anuncio dentro del horizonte: prioridad por proximidad."""

REASON_BASELINE = "baseline_rotation"
"""Muestra rotatoria de control fuera de ventana de resultados."""


# ===========================================================================
# 1. Cola de prioridad
# ===========================================================================


@dataclass(frozen=True, slots=True)
class PrioritizedTicker:
    """Un ticker en la cola, con el motivo y la urgencia de su captura."""

    ticker: Ticker
    reason: str
    days_to_event: int | None = None
    next_announcement: dt.date | None = None

    def describe(self) -> str:
        if self.reason == REASON_EVENT and self.next_announcement is not None:
            return (
                f"{self.ticker}: resultados {self.next_announcement.isoformat()} "
                f"(en {self.days_to_event} días)"
            )
        return f"{self.ticker}: línea base rotatoria"


def build_priority_queue(
    members: Sequence[Ticker],
    calendar_frame: pd.DataFrame | None,
    *,
    today: dt.date,
    horizon_days: int = 28,
    baseline_size: int = 25,
) -> list[PrioritizedTicker]:
    """Construye la cola: eventos próximos primero, línea base rotatoria después.

    Parámetros
    ----------
    members:
        Universo vigente (p. ej. `SP500Universe.members_on(today)`).
    calendar_frame:
        Calendario canónico (`data.estimates.CALENDAR_COLUMNS`) o `None` si el
        proveedor de calendario falló: en ese caso todo pasa a línea base — se
        degrada la priorización, nunca la captura.
    horizon_days:
        Ventana hacia delante (naturales) para considerar un anuncio "próximo".
        28 días cubre la ventana pre-evento típica [T-20 sesiones, T-1]
        (`events.PreEventFeatures`) con margen para fechas estimadas que se
        adelantan.
    baseline_size:
        Tamaño de la muestra de control por día. La rotación es circular y
        determinista: el desplazamiento avanza `baseline_size` posiciones por
        día natural, de modo que en ⌈N/baseline_size⌉ días la muestra recorre
        el resto completo del universo.
    """
    if horizon_days < 1:
        msg = f"horizon_days debe ser >= 1; recibido {horizon_days}"
        raise ConfigError(msg)
    if baseline_size < 0:
        msg = f"baseline_size debe ser >= 0; recibido {baseline_size}"
        raise ConfigError(msg)
    universe = sorted({normalize_ticker(t) for t in members})
    if not universe:
        msg = "el universo está vacío: no hay nada que planificar"
        raise ConfigError(msg)

    upcoming: dict[Ticker, dt.date] = {}
    if calendar_frame is not None and len(calendar_frame):
        frame = calendar_frame
        announced = pd.to_datetime(frame["announced_at"])
        event_day = announced.dt.date
        horizon_end = today + dt.timedelta(days=int(horizon_days))
        mask = (event_day >= today) & (event_day <= horizon_end)
        for ticker, day in zip(frame.loc[mask, "ticker"], event_day[mask], strict=True):
            symbol = normalize_ticker(str(ticker))
            if symbol not in universe:
                continue
            if symbol not in upcoming or day < upcoming[symbol]:
                upcoming[symbol] = day

    event_queue = [
        PrioritizedTicker(
            ticker=symbol,
            reason=REASON_EVENT,
            days_to_event=(day - today).days,
            next_announcement=day,
        )
        for symbol, day in upcoming.items()
    ]
    event_queue.sort(key=lambda p: (p.days_to_event, p.ticker))

    rest = [t for t in universe if t not in upcoming]
    baseline: list[PrioritizedTicker] = []
    if rest and baseline_size > 0:
        k = min(int(baseline_size), len(rest))
        offset = (today.toordinal() * k) % len(rest)
        picked = [rest[(offset + i) % len(rest)] for i in range(k)]
        baseline = [PrioritizedTicker(ticker=t, reason=REASON_BASELINE) for t in picked]

    return event_queue + baseline


# ===========================================================================
# 2. Presupuesto y plan
# ===========================================================================


@dataclass(slots=True)
class RequestBudget:
    """Presupuesto de peticiones HTTP de una pasada.

    Es un contador, no un limitador de tasa (eso lo hace el `TokenBucket` de
    `HttpClient`): su función es decidir **cuántos** tickers entran en el plan
    de hoy, no a qué velocidad se piden.
    """

    max_requests: int
    spent: int = 0

    def __post_init__(self) -> None:
        if self.max_requests < 1:
            msg = f"max_requests debe ser >= 1; recibido {self.max_requests}"
            raise ConfigError(msg)

    @property
    def remaining(self) -> int:
        return max(0, self.max_requests - self.spent)

    def try_consume(self, n: int) -> bool:
        """Reserva `n` peticiones; False (sin consumir) si no caben."""
        if n < 0:
            msg = f"no se puede consumir un coste negativo: {n}"
            raise ConfigError(msg)
        if self.spent + n > self.max_requests:
            return False
        self.spent += n
        return True


@dataclass(slots=True)
class CollectionPlan:
    """Plan de una pasada: qué se captura, qué se difiere y a qué coste."""

    pass_: str
    today: dt.date
    queue: list[PrioritizedTicker]
    selected: list[PrioritizedTicker]
    deferred: list[PrioritizedTicker]
    cost_per_ticker: int
    fixed_cost: int
    budget: RequestBudget

    @property
    def estimated_requests(self) -> int:
        return self.fixed_cost + self.cost_per_ticker * len(self.selected)

    def describe(self, *, max_lines: int = 20) -> str:
        """Resumen legible: es lo que imprime el `--dry-run`."""
        n_event = sum(1 for p in self.selected if p.reason == REASON_EVENT)
        n_base = len(self.selected) - n_event
        lines = [
            f"plan de la pasada {self.pass_!r} para {self.today.isoformat()}:",
            (
                f"  cola: {len(self.queue)} tickers "
                f"({sum(1 for p in self.queue if p.reason == REASON_EVENT)} con resultados "
                f"en ventana, {sum(1 for p in self.queue if p.reason == REASON_BASELINE)} "
                "de línea base)"
            ),
            (
                f"  seleccionados: {len(self.selected)} ({n_event} evento + {n_base} base) · "
                f"diferidos por presupuesto: {len(self.deferred)}"
            ),
            (
                f"  coste estimado: {self.estimated_requests} peticiones "
                f"(fijo {self.fixed_cost} + {self.cost_per_ticker}/ticker) "
                f"de un presupuesto de {self.budget.max_requests}"
            ),
        ]
        for prioritized in self.selected[:max_lines]:
            lines.append(f"    - {prioritized.describe()}")
        if len(self.selected) > max_lines:
            lines.append(f"    … y {len(self.selected) - max_lines} más")
        return "\n".join(lines)


def plan_collection(
    queue: Sequence[PrioritizedTicker],
    *,
    pass_: str,
    today: dt.date,
    budget: RequestBudget,
    cost_per_ticker: int,
    fixed_cost: int = 0,
) -> CollectionPlan:
    """Trunca la cola al presupuesto disponible, en orden de prioridad.

    El coste fijo (calendario, consenso batched, Reg SHO…) se reserva primero:
    si ni siquiera cabe, el plan sale vacío y el fallo se ve en el informe en
    vez de descubrirse a mitad de pasada con la cuota agotada.
    """
    selected: list[PrioritizedTicker] = []
    deferred: list[PrioritizedTicker] = []
    fixed_ok = budget.try_consume(fixed_cost)
    for prioritized in queue:
        if fixed_ok and budget.try_consume(cost_per_ticker):
            selected.append(prioritized)
        else:
            deferred.append(prioritized)
    if not fixed_ok:
        logger.warning(
            "el coste fijo (%d) no cabe en el presupuesto (%d); plan vacío",
            fixed_cost,
            budget.max_requests,
        )
    return CollectionPlan(
        pass_=pass_,
        today=today,
        queue=list(queue),
        selected=selected,
        deferred=deferred,
        cost_per_ticker=int(cost_per_ticker),
        fixed_cost=int(fixed_cost),
        budget=budget,
    )


# ===========================================================================
# 3. Registro de ejecución (días fallidos y huecos)
# ===========================================================================


@dataclass(frozen=True, slots=True)
class LogEntry:
    """Una línea del registro: un intento de captura y su desenlace."""

    at: str
    run_id: str
    day: str
    pass_: str
    dataset: str
    ticker: str
    status: str  # "ok" | "failed" | "skipped" | "deferred"
    attempts: int
    rows: int
    error: str = ""

    def to_json(self) -> dict[str, object]:
        return {
            "at": self.at,
            "run_id": self.run_id,
            "day": self.day,
            "pass": self.pass_,
            "dataset": self.dataset,
            "ticker": self.ticker,
            "status": self.status,
            "attempts": self.attempts,
            "rows": self.rows,
            "error": self.error,
        }

    @classmethod
    def from_json(cls, raw: dict[str, object]) -> LogEntry:
        return cls(
            at=str(raw.get("at", "")),
            run_id=str(raw.get("run_id", "")),
            day=str(raw.get("day", "")),
            pass_=str(raw.get("pass", "")),
            dataset=str(raw.get("dataset", "")),
            ticker=str(raw.get("ticker", "")),
            status=str(raw.get("status", "")),
            attempts=int(raw.get("attempts", 0)),  # type: ignore[arg-type]
            rows=int(raw.get("rows", 0)),  # type: ignore[arg-type]
            error=str(raw.get("error", "")),
        )


class CollectionLog:
    """Registro append-only (JSONL) de todos los intentos de captura.

    Un recolector desatendido en la máquina del usuario falla en silencio si
    nadie lo mira; este registro es lo que convierte "creo que lleva meses
    corriendo" en "el 2026-09-03 fallaron 12 tickers por rate limit y el
    2026-09-07 no corrió". Cada línea es un JSON independiente: un fichero
    parcialmente corrupto pierde una línea, no el histórico.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def record(self, entry: LogEntry) -> None:
        # El directorio se crea al primer registro, no al construir: un dry-run
        # no debe dejar ni un directorio vacío como rastro.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(entry.to_json(), ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + os.linesep)
            fh.flush()

    def entries(self, *, day: dt.date | None = None) -> list[LogEntry]:
        if not self.path.exists():
            return []
        out: list[LogEntry] = []
        wanted = day.isoformat() if day is not None else None
        for raw_line in self.path.read_text(encoding="utf-8").splitlines():
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                raw = json.loads(raw_line)
            except ValueError:
                logger.warning("línea ilegible en %s (se ignora): %.80s", self.path, raw_line)
                continue
            entry = LogEntry.from_json(raw)
            if wanted is None or entry.day == wanted:
                out.append(entry)
        return out

    def failures(self, *, day: dt.date | None = None) -> list[LogEntry]:
        return [e for e in self.entries(day=day) if e.status == "failed"]

    def failed_days(self) -> list[dt.date]:
        """Días con al menos un fallo registrado, ordenados."""
        days = {e.day for e in self.entries() if e.status == "failed"}
        return sorted(dt.date.fromisoformat(d) for d in days if d)

    def summary(self, day: dt.date) -> str:
        entries = self.entries(day=day)
        if not entries:
            return f"{day.isoformat()}: sin registro"
        by_status: dict[str, int] = {}
        for e in entries:
            by_status[e.status] = by_status.get(e.status, 0) + 1
        detail = ", ".join(f"{k}={v}" for k, v in sorted(by_status.items()))
        return f"{day.isoformat()}: {len(entries)} intentos ({detail})"


# ===========================================================================
# 4. Reintentos con backoff
# ===========================================================================

RETRYABLE_ERRORS: tuple[type[BaseException], ...] = (
    RateLimited,
    TransportError,
    ProviderUnavailable,
    HttpStatusError,
)
"""Errores que justifican reintentar una captura. `DataQualityError` queda fuera
a propósito: un payload malformado no mejora reintentándolo — es un bug del
parser o un cambio de la API, y debe verse, no taparse (misma política que
`data.base.FALLBACK_ERRORS`)."""


def call_with_retries(
    fn: Callable[[], T],
    *,
    attempts: int = 3,
    retry: RetryPolicy | None = None,
    clock: Clock | None = None,
    rng: random.Random | None = None,
    description: str = "",
) -> tuple[T, int]:
    """Ejecuta `fn` con reintentos y backoff; devuelve `(resultado, intentos)`.

    Respeta `RateLimited.retry_after` cuando el proveedor lo indica (nunca por
    debajo del backoff propio). Agotados los intentos, relanza el último error:
    quien llama decide si el fallo es de un ticker (se registra y se sigue) o de
    la pasada entera.
    """
    if attempts < 1:
        msg = f"attempts debe ser >= 1; recibido {attempts}"
        raise ConfigError(msg)
    policy = retry or RetryPolicy(max_retries=attempts - 1, backoff_base_s=1.0)
    timer = clock or SystemClock()
    randomness = rng or random.Random(0)
    last: BaseException | None = None
    for attempt in range(attempts):
        try:
            return fn(), attempt + 1
        except RETRYABLE_ERRORS as exc:
            last = exc
            if attempt >= attempts - 1:
                break
            delay = policy.delay_for(attempt, randomness)
            retry_after = getattr(exc, "retry_after", None)
            if retry_after:
                delay = min(max(delay, float(retry_after)), policy.max_retry_after_s)
            logger.warning(
                "reintento %d/%d%s en %.1fs tras %s: %s",
                attempt + 1,
                attempts - 1,
                f" [{description}]" if description else "",
                delay,
                type(exc).__name__,
                exc,
            )
            timer.sleep(delay)
    assert last is not None  # invariante del bucle: solo se llega aquí tras fallar
    raise last


# ===========================================================================
# 5. Orquestador diario
# ===========================================================================


@dataclass(slots=True)
class RunReport:
    """Desenlace de una pasada del recolector."""

    run_id: str
    pass_: str
    day: dt.date
    dry_run: bool
    plan: CollectionPlan | None = None
    rows_by_dataset: dict[str, int] = field(default_factory=dict)
    ok: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
    requests_spent: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return not self.failed

    def describe(self) -> str:
        mode = "DRY-RUN (sin red y sin escritura)" if self.dry_run else "ejecución real"
        lines = [
            f"recolector · pasada {self.pass_!r} · {self.day.isoformat()} · {mode} · run {self.run_id}"
        ]
        if self.plan is not None:
            lines.append(self.plan.describe())
        if not self.dry_run:
            lines.append(
                f"  capturas ok: {len(self.ok)} · fallos: {len(self.failed)} · "
                f"omitidas: {len(self.skipped)} · peticiones gastadas≈{self.requests_spent}"
            )
            for dataset, rows in sorted(self.rows_by_dataset.items()):
                lines.append(f"    {dataset}: +{rows} filas")
            for ticker, error in sorted(self.failed.items()):
                lines.append(f"    FALLO {ticker}: {error}")
            for what, reason in sorted(self.skipped.items()):
                lines.append(f"    omitido {what}: {reason}")
        lines.extend(f"  nota: {n}" for n in self.notes)
        return "\n".join(lines)


class DailyCollector:
    """Orquesta las dos pasadas diarias contra el almacén append-only.

    Pasada `close` (16:15–17:00 ET): calendario de próximos resultados,
    cadenas de opciones de la cola de prioridad y consenso de esos tickers.
    Pasada `morning` (08:30–09:15 ET): open interest actualizado de los
    tickers fotografiados ayer, Reg SHO de la sesión anterior y short interest
    quincenal si hay publicación nueva.

    Todos los relojes y fuentes son inyectables; con las fuentes sintéticas la
    clase completa funciona sin red (contrato §0.5).
    """

    def __init__(
        self,
        *,
        store: SnapshotStore,
        universe_tickers: Sequence[Ticker],
        estimates_provider: EstimatesProviderBase,
        options_source: OptionChainSource,
        off_exchange_source: RegShoSource | None = None,
        short_interest_source: FinraShortInterestSource | None = None,
        calendar: TradingCalendar | None = None,
        settings: Settings | None = None,
        wall: WallClock | None = None,
        clock: Clock | None = None,
        log: CollectionLog | None = None,
        horizon_days: int = 28,
        baseline_size: int = 25,
        max_expiries: int = 6,
        max_attempts: int = 3,
        consensus_batch_size: int = 50,
        budget_requests: int = 2000,
    ) -> None:
        self.store = store
        self.universe = sorted({normalize_ticker(t) for t in universe_tickers})
        if not self.universe:
            msg = "el recolector necesita un universo no vacío"
            raise ConfigError(msg)
        self.estimates_provider = estimates_provider
        self.options_source = options_source
        self.off_exchange_source = off_exchange_source
        self.short_interest_source = short_interest_source
        self.settings = settings or get_settings()
        self.calendar = calendar or get_calendar()
        self.wall: WallClock = wall or SystemWallClock()
        self.clock: Clock = clock or SystemClock()
        self.log = log or CollectionLog(self.store.root / "_log" / "collection_log.jsonl")
        self.horizon_days = int(horizon_days)
        self.baseline_size = int(baseline_size)
        self.max_expiries = int(max_expiries)
        self.max_attempts = int(max_attempts)
        self.consensus_batch_size = int(consensus_batch_size)
        self.budget_requests = int(budget_requests)
        self._rng = random.Random(self.settings.seed)

    # -- ayudas ---------------------------------------------------------------

    def _today(self, override: dt.date | None) -> dt.date:
        if override is not None:
            return override
        now = self.wall.now().astimezone(dt.UTC).replace(tzinfo=None)
        return utc_to_eastern(now).date()

    def _per_ticker_cost(self) -> int:
        return max(0, int(self.options_source.estimated_requests(max_expiries=self.max_expiries)))

    def _fixed_cost(self, pass_: str, n_queue: int) -> int:
        if pass_ == PASS_CLOSE:
            # calendario (≈1 petición) + consenso por lotes (≈1 petición/ticker
            # en el peor caso; se acota por la cola completa).
            synthetic = getattr(self.estimates_provider, "name", "") == "synthetic"
            return 0 if synthetic else 1 + n_queue
        cost = 0
        if self.off_exchange_source is not None:
            cost += self.off_exchange_source.estimated_requests()
        if self.short_interest_source is not None:
            cost += self.short_interest_source.estimated_requests()
        return cost

    def _fetch_upcoming_calendar(
        self, today: dt.date, report: RunReport
    ) -> pd.DataFrame | None:
        start = today
        end = today + dt.timedelta(days=self.horizon_days)
        try:
            frame, attempts = call_with_retries(
                lambda: self.estimates_provider.earnings_calendar(start, end),
                attempts=self.max_attempts,
                clock=self.clock,
                rng=self._rng,
                description="earnings_calendar",
            )
        except (InsufficientHistory, *RETRYABLE_ERRORS) as exc:
            report.notes.append(
                f"calendario no disponible ({type(exc).__name__}: {exc}); "
                "la cola se degrada a línea base pura"
            )
            self.log.record(
                self._entry(report, dataset="earnings_calendar", ticker="", status="failed",
                            attempts=self.max_attempts, rows=0, error=str(exc))
            )
            return None
        self.log.record(
            self._entry(report, dataset="earnings_calendar", ticker="", status="ok",
                        attempts=attempts, rows=len(frame))
        )
        return frame

    def _entry(
        self,
        report: RunReport,
        *,
        dataset: str,
        ticker: str,
        status: str,
        attempts: int,
        rows: int,
        error: str = "",
    ) -> LogEntry:
        return LogEntry(
            at=self.wall.now().astimezone(dt.UTC).isoformat(),
            run_id=report.run_id,
            day=report.day.isoformat(),
            pass_=report.pass_,
            dataset=dataset,
            ticker=ticker,
            status=status,
            attempts=attempts,
            rows=rows,
            error=error,
        )

    def _append(self, report: RunReport, dataset: str, frame: pd.DataFrame) -> int:
        results = self.store.append(dataset, frame)
        rows = sum(r.rows_appended for r in results)
        report.rows_by_dataset[dataset] = report.rows_by_dataset.get(dataset, 0) + rows
        return rows

    # -- planes ---------------------------------------------------------------

    def plan(
        self,
        *,
        pass_: str = PASS_CLOSE,
        today: dt.date | None = None,
        calendar_frame: pd.DataFrame | None = None,
        budget: RequestBudget | None = None,
    ) -> CollectionPlan:
        """Plan puro (sin red ni escritura) de la pasada pedida.

        Para la pasada de cierre el `calendar_frame` debería ser el calendario
        recién capturado; sin él, la cola es línea base pura. Determinista para
        una misma fecha y entradas, que es lo que hace reproducible el dry-run.
        """
        day = self._today(today)
        queue = build_priority_queue(
            self.universe,
            calendar_frame,
            today=day,
            horizon_days=self.horizon_days,
            baseline_size=self.baseline_size,
        )
        return plan_collection(
            queue,
            pass_=pass_,
            today=day,
            budget=budget or RequestBudget(self.budget_requests),
            cost_per_ticker=self._per_ticker_cost(),
            fixed_cost=self._fixed_cost(pass_, len(queue)),
        )

    # -- pasadas --------------------------------------------------------------

    def run(
        self,
        *,
        pass_: str = PASS_CLOSE,
        today: dt.date | None = None,
        dry_run: bool = False,
        budget: RequestBudget | None = None,
    ) -> RunReport:
        """Ejecuta (o simula, con `dry_run=True`) una pasada completa."""
        if pass_ not in (PASS_CLOSE, PASS_MORNING):
            msg = f"pasada desconocida: {pass_!r}"
            raise ConfigError(msg)
        day = self._today(today)
        report = RunReport(
            run_id=uuid.uuid4().hex[:12], pass_=pass_, day=day, dry_run=dry_run
        )
        if not self.calendar.is_session(day):
            report.notes.append(
                f"{day.isoformat()} no es sesión bursátil: no hay nada que capturar"
            )
            report.plan = None
            return report
        if pass_ == PASS_CLOSE:
            return self._run_close(report, dry_run=dry_run, budget=budget)
        return self._run_morning(report, dry_run=dry_run, budget=budget)

    # .. pasada de cierre .....................................................

    def _run_close(
        self, report: RunReport, *, dry_run: bool, budget: RequestBudget | None
    ) -> RunReport:
        day = report.day
        calendar_frame: pd.DataFrame | None = None
        if dry_run:
            # Sin red: si el proveedor es offline (sintético) se usa para que el
            # plan muestre prioridades reales; si exige red, se planifica sin él.
            try:
                calendar_frame = self.estimates_provider.earnings_calendar(
                    day, day + dt.timedelta(days=self.horizon_days)
                ) if getattr(self.estimates_provider, "name", "") == "synthetic" else None
            except (InsufficientHistory, *RETRYABLE_ERRORS):
                calendar_frame = None
            if calendar_frame is None:
                report.notes.append(
                    "dry-run sin calendario offline: la cola mostrada es línea base pura"
                )
        else:
            calendar_frame = self._fetch_upcoming_calendar(day, report)

        plan = self.plan(
            pass_=PASS_CLOSE, today=day, calendar_frame=calendar_frame, budget=budget
        )
        report.plan = plan
        if not dry_run:
            for deferred in plan.deferred:
                self.log.record(
                    self._entry(report, dataset="option_chain", ticker=deferred.ticker,
                                status="deferred", attempts=0, rows=0,
                                error="presupuesto agotado")
                )
        if dry_run:
            report.notes.append(
                "dry-run: no se ha hecho ninguna petición ni se ha escrito nada"
            )
            return report

        if calendar_frame is not None and len(calendar_frame):
            try:
                # Se sella el frame ya pedido en la planificación: una sola
                # petición de calendario por pasada.
                snap = snapshot_earnings_calendar(
                    self.estimates_provider,
                    day,
                    day + dt.timedelta(days=self.horizon_days),
                    wall=self.wall,
                    frame=calendar_frame,
                )
                self._append(report, "earnings_calendar", snap)
            except EarningsAlphaError as exc:
                report.skipped["earnings_calendar"] = str(exc)

        # Cadenas de opciones, en orden de prioridad.
        for prioritized in plan.selected:
            ticker = prioritized.ticker
            try:
                frame, attempts = call_with_retries(
                    lambda t=ticker: snapshot_option_chain(
                        self.options_source, t, wall=self.wall, max_expiries=self.max_expiries
                    ),
                    attempts=self.max_attempts,
                    clock=self.clock,
                    rng=self._rng,
                    description=f"option_chain[{ticker}]",
                )
            except InsufficientHistory as exc:
                report.skipped[ticker] = str(exc)
                self.log.record(
                    self._entry(report, dataset="option_chain", ticker=ticker,
                                status="skipped", attempts=1, rows=0, error=str(exc))
                )
                continue
            except (DataQualityError, *RETRYABLE_ERRORS) as exc:
                report.failed[ticker] = f"{type(exc).__name__}: {exc}"
                self.log.record(
                    self._entry(report, dataset="option_chain", ticker=ticker,
                                status="failed", attempts=self.max_attempts, rows=0,
                                error=str(exc))
                )
                continue
            rows = self._append(report, "option_chain", frame)
            report.ok.append(ticker)
            self.log.record(
                self._entry(report, dataset="option_chain", ticker=ticker,
                            status="ok", attempts=attempts, rows=rows)
            )

        # Consenso de los tickers del plan, por lotes.
        selected = [p.ticker for p in plan.selected]
        for i in range(0, len(selected), self.consensus_batch_size):
            batch = selected[i : i + self.consensus_batch_size]
            label = f"consensus[{batch[0]}..{batch[-1]}]"
            try:
                frame, attempts = call_with_retries(
                    lambda b=batch: snapshot_consensus(
                        self.estimates_provider, b, wall=self.wall
                    ),
                    attempts=self.max_attempts,
                    clock=self.clock,
                    rng=self._rng,
                    description=label,
                )
            except InsufficientHistory as exc:
                report.skipped[label] = str(exc)
                self.log.record(
                    self._entry(report, dataset="consensus", ticker=label,
                                status="skipped", attempts=1, rows=0, error=str(exc))
                )
                continue
            except (DataQualityError, *RETRYABLE_ERRORS) as exc:
                report.failed[label] = f"{type(exc).__name__}: {exc}"
                self.log.record(
                    self._entry(report, dataset="consensus", ticker=label,
                                status="failed", attempts=self.max_attempts, rows=0,
                                error=str(exc))
                )
                continue
            rows = self._append(report, "consensus", frame)
            self.log.record(
                self._entry(report, dataset="consensus", ticker=label,
                            status="ok", attempts=attempts, rows=rows)
            )

        report.requests_spent = plan.budget.spent
        return report

    # .. pasada matinal .......................................................

    def _morning_tickers(self, day: dt.date, report: RunReport) -> list[Ticker]:
        """Tickers cuya cadena se fotografió en la sesión anterior.

        Se leen del almacén (fuente de verdad), no se recalcula el plan: si ayer
        el presupuesto truncó la cola, el OI de hoy debe cubrir exactamente lo
        fotografiado, ni más ni menos. Si ayer no hubo captura (hueco), se cae
        al plan determinista de ayer y se deja nota.
        """
        prev = self.calendar.prev_session(day)
        try:
            chains = self.store.read(
                "option_chain", start=prev, end=prev, columns=["ticker"], validate=False
            )
            return sorted(set(chains["ticker"].astype(str)))
        except InsufficientHistory:
            report.notes.append(
                f"sin cadena archivada de {prev.isoformat()}: hueco en el panel; se usa el "
                "plan determinista de esa sesión como mejor aproximación"
            )
            fallback = self.plan(pass_=PASS_MORNING, today=prev, calendar_frame=None)
            return [p.ticker for p in fallback.selected]

    def _run_morning(
        self, report: RunReport, *, dry_run: bool, budget: RequestBudget | None
    ) -> RunReport:
        day = report.day
        prev = self.calendar.prev_session(day)
        tickers = self._morning_tickers(day, report)
        queue = [PrioritizedTicker(ticker=t, reason=REASON_EVENT) for t in tickers]
        plan = plan_collection(
            queue,
            pass_=PASS_MORNING,
            today=day,
            budget=budget or RequestBudget(self.budget_requests),
            cost_per_ticker=self._per_ticker_cost(),
            fixed_cost=self._fixed_cost(PASS_MORNING, len(queue)),
        )
        report.plan = plan
        report.notes.append(
            f"open interest de la sesión {prev.isoformat()} (capturado la mañana de "
            f"{day.isoformat()}, available_at real)"
        )
        if dry_run:
            report.notes.append(
                "dry-run: no se ha hecho ninguna petición ni se ha escrito nada"
            )
            return report

        for prioritized in plan.selected:
            ticker = prioritized.ticker
            try:
                frame, attempts = call_with_retries(
                    lambda t=ticker: snapshot_open_interest(
                        self.options_source,
                        t,
                        calendar=self.calendar,
                        wall=self.wall,
                        max_expiries=self.max_expiries,
                    ),
                    attempts=self.max_attempts,
                    clock=self.clock,
                    rng=self._rng,
                    description=f"open_interest[{ticker}]",
                )
            except InsufficientHistory as exc:
                report.skipped[ticker] = str(exc)
                self.log.record(
                    self._entry(report, dataset="open_interest", ticker=ticker,
                                status="skipped", attempts=1, rows=0, error=str(exc))
                )
                continue
            except (DataQualityError, *RETRYABLE_ERRORS) as exc:
                report.failed[ticker] = f"{type(exc).__name__}: {exc}"
                self.log.record(
                    self._entry(report, dataset="open_interest", ticker=ticker,
                                status="failed", attempts=self.max_attempts, rows=0,
                                error=str(exc))
                )
                continue
            rows = self._append(report, "open_interest", frame)
            report.ok.append(ticker)
            self.log.record(
                self._entry(report, dataset="open_interest", ticker=ticker,
                            status="ok", attempts=attempts, rows=rows)
            )

        if self.off_exchange_source is not None:
            try:
                frame, attempts = call_with_retries(
                    lambda: snapshot_off_exchange(
                        self.off_exchange_source,  # type: ignore[arg-type]
                        prev,
                        wall=self.wall,
                        tickers=self.universe,
                    ),
                    attempts=self.max_attempts,
                    clock=self.clock,
                    rng=self._rng,
                    description="off_exchange",
                )
                rows = self._append(report, "off_exchange", frame)
                self.log.record(
                    self._entry(report, dataset="off_exchange", ticker="",
                                status="ok", attempts=attempts, rows=rows)
                )
            except InsufficientHistory as exc:
                report.skipped["off_exchange"] = str(exc)
                self.log.record(
                    self._entry(report, dataset="off_exchange", ticker="",
                                status="skipped", attempts=1, rows=0, error=str(exc))
                )
            except (DataQualityError, *RETRYABLE_ERRORS) as exc:
                report.failed["off_exchange"] = f"{type(exc).__name__}: {exc}"
                self.log.record(
                    self._entry(report, dataset="off_exchange", ticker="",
                                status="failed", attempts=self.max_attempts, rows=0,
                                error=str(exc))
                )

        if self.short_interest_source is not None:
            try:
                frame, attempts = call_with_retries(
                    lambda: snapshot_short_interest(
                        self.short_interest_source,  # type: ignore[arg-type]
                        wall=self.wall,
                        tickers=self.universe,
                    ),
                    attempts=self.max_attempts,
                    clock=self.clock,
                    rng=self._rng,
                    description="short_interest",
                )
                rows = self._append(report, "short_interest", frame)
                self.log.record(
                    self._entry(report, dataset="short_interest", ticker="",
                                status="ok", attempts=attempts, rows=rows)
                )
            except InsufficientHistory as exc:
                report.skipped["short_interest"] = str(exc)
                self.log.record(
                    self._entry(report, dataset="short_interest", ticker="",
                                status="skipped", attempts=1, rows=0, error=str(exc))
                )
            except (DataQualityError, *RETRYABLE_ERRORS) as exc:
                report.failed["short_interest"] = f"{type(exc).__name__}: {exc}"
                self.log.record(
                    self._entry(report, dataset="short_interest", ticker="",
                                status="failed", attempts=self.max_attempts, rows=0,
                                error=str(exc))
                )

        report.requests_spent = plan.budget.spent
        return report
