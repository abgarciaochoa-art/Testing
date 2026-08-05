"""Recolector diario point-in-time (`earnings_alpha.collector`).

Acumula desde hoy los datos que **no se pueden comprar hacia atrás** con
presupuesto cero (`docs/research/datasets_opciones.md` §1,
`docs/research/data_sources.md` §6-§9): cadenas de opciones completas, vintages
de consenso, calendario de próximos resultados, short interest y volumen
off-exchange. Cada día sin recolectar es un día de panel perdido para siempre;
este módulo está pensado para correr desatendido en la máquina del usuario vía
cron/systemd (instalación exacta en `docs/RECOLECTOR.md`).

Arquitectura en tres capas, cada una probada sin red:

- `snapshot`  — captura y sellado: cada fila lleva `source`, `captured_at`
  (UTC exacto), `available_at` y `collector_version`. Dos pasadas: `close`
  (16:15–17:00 ET) y `morning` (08:30–09:15 ET, open interest actualizado).
- `storage`   — parquet append-only particionado por fecha, deduplicación
  idempotente por clave natural y verificación de integridad (SHA-256 por
  fichero) al leer.
- `scheduler` — cola de prioridad (resultados en ≤4 semanas primero, muestra
  rotatoria de línea base después), presupuesto de peticiones, reintentos con
  backoff y registro append-only de fallos y huecos.

CLI::

    python3 -m earnings_alpha.collector run --dry-run       # sin red: muestra el plan
    python3 -m earnings_alpha.collector run --pass close    # pasada de cierre real
    python3 -m earnings_alpha.collector run --pass morning  # pasada matinal (OI)
    python3 -m earnings_alpha.collector status              # particiones, huecos, fallos
    python3 -m earnings_alpha.collector verify              # integridad del almacén
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from collections.abc import Sequence
from pathlib import Path

from earnings_alpha.collector.scheduler import (
    REASON_BASELINE,
    REASON_EVENT,
    RETRYABLE_ERRORS,
    CollectionLog,
    CollectionPlan,
    DailyCollector,
    LogEntry,
    PrioritizedTicker,
    RequestBudget,
    RunReport,
    build_priority_queue,
    call_with_retries,
    plan_collection,
)
from earnings_alpha.collector.snapshot import (
    CALENDAR_SNAPSHOT_COLUMNS,
    CAPTURE_PASSES,
    COLLECTOR_VERSION,
    CONSENSUS_SNAPSHOT_COLUMNS,
    OFF_EXCHANGE_COLUMNS,
    OPEN_INTEREST_COLUMNS,
    OPTION_CHAIN_COLUMNS,
    PASS_CLOSE,
    PASS_MORNING,
    RECOMMENDED_CAPTURE_WINDOWS_ET,
    SHORT_INTEREST_COLUMNS,
    STAMP_COLUMNS,
    FinraShortInterestSource,
    OptionChainSource,
    RegShoSource,
    SyntheticOptionsSource,
    TradierOptionsSource,
    YFinanceOptionsSource,
    capture_window_warning,
    snapshot_consensus,
    snapshot_earnings_calendar,
    snapshot_off_exchange,
    snapshot_open_interest,
    snapshot_option_chain,
    snapshot_short_interest,
    to_yahoo_symbol,
)
from earnings_alpha.collector.storage import (
    DATASET_SPECS,
    AppendResult,
    DatasetSpec,
    IntegrityProblem,
    IntegrityReport,
    SnapshotStore,
)
from earnings_alpha.config import get_settings
from earnings_alpha.errors import ConfigError, EarningsAlphaError

__all__ = [  # noqa: RUF022 - orden temático, no alfabético
    # versión y pasadas
    "COLLECTOR_VERSION",
    "PASS_CLOSE",
    "PASS_MORNING",
    "CAPTURE_PASSES",
    "RECOMMENDED_CAPTURE_WINDOWS_ET",
    "capture_window_warning",
    # esquemas
    "STAMP_COLUMNS",
    "OPTION_CHAIN_COLUMNS",
    "OPEN_INTEREST_COLUMNS",
    "CONSENSUS_SNAPSHOT_COLUMNS",
    "CALENDAR_SNAPSHOT_COLUMNS",
    "OFF_EXCHANGE_COLUMNS",
    "SHORT_INTEREST_COLUMNS",
    # fuentes
    "OptionChainSource",
    "YFinanceOptionsSource",
    "TradierOptionsSource",
    "SyntheticOptionsSource",
    "RegShoSource",
    "FinraShortInterestSource",
    "to_yahoo_symbol",
    # capturas
    "snapshot_option_chain",
    "snapshot_open_interest",
    "snapshot_consensus",
    "snapshot_earnings_calendar",
    "snapshot_off_exchange",
    "snapshot_short_interest",
    # almacén
    "SnapshotStore",
    "DatasetSpec",
    "DATASET_SPECS",
    "AppendResult",
    "IntegrityProblem",
    "IntegrityReport",
    # planificación
    "REASON_EVENT",
    "REASON_BASELINE",
    "PrioritizedTicker",
    "build_priority_queue",
    "RequestBudget",
    "CollectionPlan",
    "plan_collection",
    "LogEntry",
    "CollectionLog",
    "RETRYABLE_ERRORS",
    "call_with_retries",
    "RunReport",
    "DailyCollector",
    # CLI
    "main",
]


# ===========================================================================
# CLI
# ===========================================================================


def _default_root() -> Path:
    return get_settings().data_dir / "collector"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m earnings_alpha.collector",
        description=(
            "Recolector diario point-in-time: cadenas de opciones, consenso, "
            "calendario, short interest y off-exchange. Instalación en cron: "
            "docs/RECOLECTOR.md"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="ejecuta (o simula) una pasada de captura")
    run.add_argument(
        "--pass",
        dest="pass_",
        choices=[PASS_CLOSE, PASS_MORNING],
        default=PASS_CLOSE,
        help="pasada: 'close' (16:15-17:00 ET) o 'morning' (08:30-09:15 ET)",
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="muestra qué se capturaría sin abrir la red ni escribir nada",
    )
    run.add_argument("--date", type=dt.date.fromisoformat, default=None,
                     help="fecha de la pasada (por defecto: hoy en hora de Nueva York)")
    run.add_argument("--root", type=Path, default=None,
                     help="raíz del almacén (por defecto: data/collector)")
    run.add_argument("--budget", type=int, default=2000,
                     help="presupuesto de peticiones HTTP de la pasada")
    run.add_argument("--horizon-days", type=int, default=28,
                     help="ventana hacia delante del calendario de resultados")
    run.add_argument("--baseline", type=int, default=25,
                     help="tamaño de la muestra rotatoria de línea base")
    run.add_argument("--max-expiries", type=int, default=6,
                     help="vencimientos por cadena de opciones")
    run.add_argument(
        "--options-source",
        choices=["auto", "tradier", "yfinance", "synthetic"],
        default="auto",
        help="fuente de cadenas: auto = tradier si hay token, si no yfinance, si no synthetic",
    )
    run.add_argument(
        "--calendar-provider",
        choices=["auto", "finnhub", "fmp", "synthetic"],
        default="auto",
        help="proveedor de calendario/consenso: auto = por credenciales, con synthetic de red de seguridad",
    )
    run.add_argument("--no-flow", action="store_true",
                     help="desactiva Reg SHO y short interest en la pasada matinal")

    status = sub.add_parser("status", help="particiones, huecos y fallos registrados")
    status.add_argument("--root", type=Path, default=None)

    verify = sub.add_parser("verify", help="verifica la integridad (SHA-256) del almacén")
    verify.add_argument("--root", type=Path, default=None)

    return parser


def _resolve_options_source(choice: str, settings, wall):  # CLI interno, sin anotar
    from earnings_alpha.data.synthetic import SyntheticMarket

    if choice in ("auto", "tradier") and settings.env("TRADIER_ACCESS_TOKEN"):
        return TradierOptionsSource(settings=settings)
    if choice == "tradier":
        raise ConfigError(
            "se pidió --options-source tradier y falta TRADIER_ACCESS_TOKEN en el entorno"
        )
    if choice in ("auto", "yfinance"):
        source = YFinanceOptionsSource(settings=settings)
        if source.available():
            return source
        if choice == "yfinance":
            raise ConfigError(
                "se pidió --options-source yfinance y la dependencia no está instalada "
                "(pip install yfinance)"
            )
    # Red de seguridad sin red: mercado sintético que cubre el presente, con los
    # mismos tickers reales del fichero semilla (contrato §0.5).
    today = dt.date.today()
    market = SyntheticMarket(
        seed=settings.seed,
        n_tickers=40,
        start=today - dt.timedelta(days=300),
        end=today + dt.timedelta(days=45),
    )
    return SyntheticOptionsSource(market, wall=wall, settings=settings)


def _resolve_estimates_provider(choice: str, settings, options_source):  # CLI interno
    from earnings_alpha.data.estimates import (
        FinnhubEstimatesProvider,
        FMPEstimatesProvider,
        SyntheticEstimatesProvider,
    )

    if choice in ("auto", "finnhub") and settings.env("FINNHUB_API_KEY"):
        return FinnhubEstimatesProvider(settings=settings)
    if choice == "finnhub":
        raise ConfigError("se pidió finnhub y falta FINNHUB_API_KEY en el entorno")
    if choice in ("auto", "fmp") and settings.env("FMP_API_KEY"):
        return FMPEstimatesProvider(settings=settings)
    if choice == "fmp":
        raise ConfigError("se pidió fmp y falta FMP_API_KEY en el entorno")
    # Sintético: si la fuente de opciones ya lleva un mercado, se comparte para
    # que calendario y cadenas hablen del mismo mundo.
    market = getattr(options_source, "market", None) if isinstance(
        options_source, SyntheticOptionsSource
    ) else None
    if market is not None:
        return SyntheticEstimatesProvider(market, settings=settings)
    today = dt.date.today()
    return SyntheticEstimatesProvider(
        settings=settings,
        n_tickers=40,
        start=today - dt.timedelta(days=300),
        end=today + dt.timedelta(days=45),
    )


def _universe_tickers(today: dt.date) -> list[str]:
    from earnings_alpha.errors import InsufficientHistory, UniverseError
    from earnings_alpha.universe import SP500Universe

    try:
        return SP500Universe().members_on(today)
    except (InsufficientHistory, UniverseError, OSError) as exc:
        raise ConfigError(
            f"no se pudo resolver el universo S&P 500 para {today.isoformat()}: {exc}"
        ) from exc


def _cmd_run(args: argparse.Namespace) -> int:
    settings = get_settings()
    root = args.root or _default_root()
    store = SnapshotStore(root)
    from earnings_alpha.data.cache import SystemWallClock

    wall = SystemWallClock()
    options_source = _resolve_options_source(args.options_source, settings, wall)
    estimates_provider = _resolve_estimates_provider(
        args.calendar_provider, settings, options_source
    )
    today = args.date or None
    universe = _universe_tickers(today or dt.date.today())

    collector = DailyCollector(
        store=store,
        universe_tickers=universe,
        estimates_provider=estimates_provider,
        options_source=options_source,
        off_exchange_source=None if args.no_flow else RegShoSource(settings=settings),
        short_interest_source=None if args.no_flow else FinraShortInterestSource(
            settings=settings
        ),
        settings=settings,
        wall=wall,
        horizon_days=args.horizon_days,
        baseline_size=args.baseline,
        max_expiries=args.max_expiries,
        budget_requests=args.budget,
    )
    print(
        f"recolector v{COLLECTOR_VERSION} · almacén {root} · "
        f"opciones={options_source.name} · calendario={estimates_provider.name}"
    )
    report = collector.run(pass_=args.pass_, today=today, dry_run=args.dry_run)
    print(report.describe())
    if not args.dry_run and not report.success:
        return 1
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    from earnings_alpha.pit import get_calendar

    root = args.root or _default_root()
    store = SnapshotStore(root)
    print(store.status())
    calendar = get_calendar()
    for dataset in ("option_chain", "open_interest", "off_exchange"):
        gaps = store.missing_sessions(dataset, calendar)
        if gaps:
            shown = ", ".join(d.isoformat() for d in gaps[:10])
            extra = f" (+{len(gaps) - 10} más)" if len(gaps) > 10 else ""
            print(f"  HUECOS en {dataset}: {shown}{extra}")
    log = CollectionLog(root / "_log" / "collection_log.jsonl")
    failed_days = log.failed_days()
    if failed_days:
        print("días con fallos registrados:")
        for day in failed_days[-10:]:
            print(f"  {log.summary(day)}")
    else:
        print("sin fallos registrados")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    root = args.root or _default_root()
    store = SnapshotStore(root)
    report = store.verify()
    print(report.describe())
    return 0 if report.ok else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada del CLI (`python3 -m earnings_alpha.collector`)."""
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else sys.argv[1:])
    try:
        if args.command == "run":
            return _cmd_run(args)
        if args.command == "status":
            return _cmd_status(args)
        if args.command == "verify":
            return _cmd_verify(args)
    except ConfigError as exc:
        print(f"error de configuración: {exc}", file=sys.stderr)
        return 2
    except EarningsAlphaError as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    parser.error(f"orden desconocida: {args.command!r}")  # pragma: no cover
    return 2  # pragma: no cover
