"""CLI de earnings-alpha: entrypoints de investigación (contrato §2, capa cli).

Se instala como ``earnings-alpha`` (ver ``pyproject.toml``) y también funciona
como ``python3 -m earnings_alpha.cli``. Subcomandos::

    earnings-alpha universe show [--date ...]        pertenencia PIT al S&P 500
    earnings-alpha universe refresh [--dry-run ...]  refresco append-only del histórico
    earnings-alpha data status                       credenciales, datasets y caché
    earnings-alpha factors list                      factores registrados
    earnings-alpha factors compute FACTOR            factor + IC sobre el sintético
    earnings-alpha events scan                       huella de negociación informada
    earnings-alpha backtest run --mode cross|event   motores de backtest
    earnings-alpha collector run --dry-run           recolector diario (passthrough)
    earnings-alpha report --output out.html          tearsheet HTML autocontenido

Principios que el CLI hace cumplir:

- **Sin red ni credenciales por defecto.** Todo subcomando de cómputo corre
  contra `data.synthetic.SyntheticMarket` (contrato §0.5): mismo esquema, misma
  semántica point-in-time y verdad-terreno de filtraciones
  (`leaked_event_ids()`) para poder *medir* que el detector detecta.
- **Fallo explícito.** `ProviderUnavailable`, `InsufficientHistory` o
  `LookAheadError` terminan el proceso con código distinto de cero y un mensaje
  accionable; jamás se degradan a una tabla vacía con aspecto de resultado
  (contrato §0.3).
- **Métricas con banda.** Toda métrica de rendimiento impresa o volcada al
  tearsheet lleva su intervalo de confianza (contrato §3.8): Sharpe con IC de
  Mertens y PSR (Bailey y López de Prado 2012), IC media con t de Newey–West
  (1987), retorno medio por evento con bootstrap estacionario.

Códigos de salida: ``0`` éxito, ``1`` error de datos/proveedor
(`EarningsAlphaError`), ``2`` error de configuración o de uso (`ConfigError`,
argumentos inválidos).
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from earnings_alpha.config import PROVIDER_ENV_KEYS, get_settings
from earnings_alpha.errors import ConfigError, EarningsAlphaError

if TYPE_CHECKING:  # pragma: no cover - solo para tipado
    import pandas as pd

    from earnings_alpha.data.synthetic import SyntheticMarket

__all__ = ["LocalCsvUniverseSource", "build_parser", "main"]


# ===========================================================================
# Utilidades de análisis de argumentos
# ===========================================================================


def _parse_date(raw: str) -> dt.date:
    """Fecha ISO para argparse; el error se convierte en mensaje de uso."""
    try:
        return dt.date.fromisoformat(raw)
    except ValueError as exc:
        msg = f"fecha inválida {raw!r}: se espera AAAA-MM-DD"
        raise argparse.ArgumentTypeError(msg) from exc


def _parse_int_list(raw: str) -> list[int]:
    """Lista de enteros separados por comas ("-5,1,3")."""
    try:
        out = [int(tok) for tok in raw.split(",") if tok.strip() != ""]
    except ValueError as exc:
        msg = f"lista de enteros inválida {raw!r}: se espera p. ej. '-5,1,3'"
        raise argparse.ArgumentTypeError(msg) from exc
    if not out:
        msg = f"lista de enteros vacía: {raw!r}"
        raise argparse.ArgumentTypeError(msg)
    return out


def _add_market_options(parser: argparse.ArgumentParser, *, leak_default: float = 0.15) -> None:
    """Opciones comunes del mercado sintético (contrato §0.5: offline-testable)."""
    grp = parser.add_argument_group("mercado sintético")
    grp.add_argument("--seed", type=int, default=None,
                     help="semilla del generador (por defecto: la de config.Settings)")
    grp.add_argument("--tickers", type=int, default=24, metavar="N",
                     help="tamaño del universo sintético (muestra estratificada por sector)")
    grp.add_argument("--start", type=_parse_date, default=dt.date(2020, 1, 2),
                     help="primera sesión del panel (AAAA-MM-DD)")
    grp.add_argument("--end", type=_parse_date, default=dt.date(2022, 12, 30),
                     help="última sesión del panel (AAAA-MM-DD)")
    grp.add_argument("--leak-fraction", type=float, default=leak_default,
                     help="fracción de eventos con huella pre-anuncio inyectada")


def _build_market(args: argparse.Namespace) -> SyntheticMarket:
    """Construye el mercado sintético con las opciones comunes."""
    from earnings_alpha.data.synthetic import SyntheticMarket

    settings = get_settings()
    return SyntheticMarket(
        seed=args.seed if args.seed is not None else settings.seed,
        n_tickers=args.tickers,
        start=args.start,
        end=args.end,
        leak_fraction=getattr(args, "leak_fraction", 0.15),
    )


def _full_registry():
    """Registro de factores con TODOS los módulos importados.

    Los factores fundamentales (`accruals`, `growth`, `quality`, `value`) se
    registran al importarse; sin estos imports `factors list` mentiría por
    omisión.
    """
    from earnings_alpha.factors import accruals, default_registry, growth, quality, value

    _ = (accruals, growth, quality, value)  # importados por su efecto de registro
    return default_registry


def _human_size(n_bytes: int) -> str:
    """Tamaño legible (1 MB = 1e6 bytes; suficiente para diagnóstico)."""
    value = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1000.0 or unit == "GB":
            return f"{value:,.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1000.0
    return f"{value:,.1f} GB"  # pragma: no cover - inalcanzable


# ===========================================================================
# universe
# ===========================================================================


class LocalCsvUniverseSource:
    """Fuente de universo **offline**: snapshots ``(date, tickers)`` de un CSV local.

    Cumple el protocolo `universe.sources.HistoricalUniverseSource` por
    duck-typing (``name``, ``available()``, ``fetch_history()``,
    ``fetch_snapshot()``), lo que permite incorporar al histórico una descarga
    manual —o un fichero preparado por otro proceso— sin abrir la red, con las
    mismas garantías append-only de `refresh_history_file`: nunca se reescribe
    una fecha ya registrada y las divergencias se reportan, no se aplican.
    """

    name = "local_csv"

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def available(self) -> bool:
        """True si el fichero existe (no hay red ni credenciales implicadas)."""
        return self.path.exists()

    def fetch_history(self) -> pd.DataFrame:
        """Histórico completo del CSV, en el esquema canónico ``(date, tickers)``."""
        from earnings_alpha.universe import read_history_csv

        return read_history_csv(self.path)

    def fetch_snapshot(self):
        """Último snapshot del CSV como `ConstituentSnapshot` (fecha exacta)."""
        from earnings_alpha.universe.sources import ConstituentSnapshot

        frame = self.fetch_history().sort_values("date")
        last = frame.iloc[-1]
        tickers = tuple(sorted(t.strip() for t in str(last["tickers"]).split(",") if t.strip()))
        return ConstituentSnapshot(
            as_of=last["date"].date(),
            tickers=tickers,
            source=self.name,
            is_estimated_date=False,
        )


def _cmd_universe_show(args: argparse.Namespace) -> int:
    from collections import Counter

    from earnings_alpha.universe import SP500Universe

    universe = SP500Universe(history_path=args.history)
    day = args.date or universe.last_snapshot
    members = universe.members_on(day)
    snap = universe.snapshot_date_for(day)
    print(f"S&P 500 en {day.isoformat()} (snapshot registrado: {snap.isoformat()})")
    print(f"  miembros: {len(members)}")
    sectors = Counter(universe.sector_for(t) or "desconocido" for t in members)
    print("  por sector GICS (según los constituyentes actuales; los históricos")
    print("  sin correspondencia aparecen como 'desconocido'):")
    for sector, count in sectors.most_common():
        print(f"    {sector:<28} {count:>4}")
    shown = members[: args.limit]
    print(f"  muestra ({len(shown)} de {len(members)}): {', '.join(shown)}")
    return 0


def _cmd_universe_refresh(args: argparse.Namespace) -> int:
    from earnings_alpha.universe import SP500Universe, build_source

    sources: list[object] = []
    if args.from_csv is not None:
        sources.append(LocalCsvUniverseSource(args.from_csv))
    for name in args.source or []:
        sources.append(build_source(name))
    universe = SP500Universe(history_path=args.history)
    report = universe.refresh(sources or None, dry_run=args.dry_run)  # type: ignore[arg-type]
    print(report.summary())
    if len(report.divergences):
        print("divergencias con la historia registrada (NO aplicadas):")
        print(report.divergences.to_string(index=False, max_rows=20))
    return 0


# ===========================================================================
# data status
# ===========================================================================


def _dataset_line(label: str, path: Path) -> str:
    if not path.exists():
        return f"  {label:<26} AUSENTE ({path})"
    size = _human_size(path.stat().st_size)
    rows = ""
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq

            rows = f", {pq.ParquetFile(path).metadata.num_rows:,} filas"
        except Exception:  # diagnóstico, no ruta crítica: el tamaño basta
            rows = ""
    return f"  {label:<26} {path.name} ({size}{rows})"


def _cmd_data_status(args: argparse.Namespace) -> int:
    settings = get_settings()
    print("configuración:")
    print(f"  data_dir  = {settings.data_dir}")
    print(f"  cache_dir = {settings.cache_dir}")
    print(f"  offline   = {settings.offline}")

    print("\nproveedores (solo credenciales; la conectividad real no se sondea aquí):")
    for name in sorted(PROVIDER_ENV_KEYS):
        needed = PROVIDER_ENV_KEYS[name]
        missing = settings.missing_credentials_for(name)
        if name == "synthetic":
            estado = "disponible SIEMPRE (sin credenciales ni red; contrato §0.5)"
        elif not needed:
            estado = "sin credenciales necesarias"
        elif missing:
            estado = f"faltan variables de entorno: {', '.join(missing)}"
        else:
            estado = "credenciales presentes"
        print(f"  {name:<18} {estado}")

    print("\ndatasets locales:")
    external = settings.data_dir / "external"
    for label, path in (
        ("constituyentes semilla", settings.seed_dir / "sp500_constituents.csv"),
        ("composición histórica", settings.seed_dir / "sp500_historical_components.csv"),
        ("consenso de analistas", external / "consenso" / "consenso_master.parquet"),
        ("VIX diario", external / "opciones" / "vix-daily.csv"),
        ("fixtures de opciones", external / "opciones" / "lean_sample"),
    ):
        if path.is_dir():
            n = sum(1 for _ in path.rglob("*") if _.is_file())
            print(f"  {label:<26} {path.name}/ ({n} ficheros)")
        else:
            print(_dataset_line(label, path))

    consenso = external / "consenso" / "consenso_master.parquet"
    if consenso.exists():
        print(
            "  ADVERTENCIA PIT del consenso: es el consenso FINAL previo al anuncio,\n"
            "  sin historial de revisiones intra-trimestre, y la lista de tickers\n"
            "  padece supervivencia parcial (snapshots de 2022 y 2025). Sirve para\n"
            "  SUE de analistas y estudios de evento; NO para momentum de revisiones\n"
            "  (ver factors.surprise.CONSENSUS_PIT_WARNING)."
        )

    print("\ncaché en disco:")
    cache = settings.cache_dir
    if cache.exists():
        files = [p for p in cache.rglob("*") if p.is_file()]
        total = sum(p.stat().st_size for p in files)
        print(f"  {cache}: {len(files)} ficheros, {_human_size(total)}")
    else:
        print(f"  {cache}: vacía (aún no creada)")
    return 0


# ===========================================================================
# factors
# ===========================================================================


def _cmd_factors_list(args: argparse.Namespace) -> int:
    import inspect

    registry = _full_registry()
    names = registry.names()
    print(f"{len(names)} factores registrados (docstring = referencia académica):\n")
    for name in names:
        try:
            factor = registry.create(name)
            requires = ",".join(factor.requires)
            doc = (inspect.getdoc(type(factor)) or "").strip().splitlines()
            first = doc[0] if doc else ""
        except EarningsAlphaError as exc:
            requires, first = "?", f"(no instanciable sin parámetros: {exc})"
        if len(first) > 76:
            first = first[:73] + "..."
        print(f"  {name:<26} [{requires}]")
        if first:
            print(f"      {first}")
    return 0


def _factor_scores(market: SyntheticMarket, factor_name: str) -> pd.Series:
    """Computa un factor del registro sobre el contexto sintético."""
    from earnings_alpha.factors import context_from_synthetic

    registry = _full_registry()
    ctx = context_from_synthetic(market)
    return registry.compute(factor_name, ctx)


def _ic_block(
    market: SyntheticMarket,
    scores: pd.Series,
    *,
    horizon: int,
    min_names: int,
    method: str = "spearman",
):
    """Serie de IC y su resumen Newey–West (`validation_methodology.md` §2-§3)."""
    from earnings_alpha.stats.ic import cross_sectional_ic, forward_returns, summarize_ic

    fwd = forward_returns(market.prices(), horizon=horizon)
    ic = cross_sectional_ic(scores, fwd, method=method, min_names=min_names)  # type: ignore[arg-type]
    summary = summarize_ic(ic, horizon=horizon, method=method)  # type: ignore[arg-type]
    return ic, summary


def _cmd_factors_compute(args: argparse.Namespace) -> int:
    market = _build_market(args)
    scores = _factor_scores(market, args.factor)
    valid = scores.dropna()
    n_dates = scores.index.get_level_values("date").nunique()
    n_tick = scores.index.get_level_values("ticker").nunique()
    print(f"factor {args.factor!r} sobre {market!r}")
    print(
        f"  cobertura: {len(valid):,} observaciones no-NaN de {len(scores):,} "
        f"({100.0 * len(valid) / max(len(scores), 1):.1f}%), "
        f"{n_dates} fechas x {n_tick} tickers"
    )
    _, summary = _ic_block(
        market, scores, horizon=args.horizon, min_names=args.min_names, method=args.method
    )
    print(f"  IC ({args.method}, forward {args.horizon} sesiones, retardo de ejecución 1):")
    print(f"    {summary}")
    lo, hi = summary.ci
    print(
        f"    IC media {summary.mean:+.4f} con IC95% [{lo:+.4f}, {hi:+.4f}] "
        f"(t Newey-West {summary.t_stat:+.2f}; el t ingenuo "
        f"{summary.t_naive:+.2f} se reporta solo para ver cuánto infla)"
    )
    if args.output is not None:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        frame = scores.rename(args.factor).reset_index()
        if out.suffix == ".parquet":
            frame.to_parquet(out, index=False)
        else:
            frame.to_csv(out, index=False)
        print(f"  panel escrito en {out} ({_human_size(out.stat().st_size)})")
    return 0


# ===========================================================================
# events scan
# ===========================================================================


def _cmd_events_scan(args: argparse.Namespace) -> int:
    from earnings_alpha.events.preevent import (
        EventContext,
        InformedTradingScore,
        PreEventFeatures,
        positive_predictive_value,
    )

    market = _build_market(args)
    ctx = EventContext.from_synthetic(market)
    features = PreEventFeatures().compute(ctx)
    scorer = InformedTradingScore(mode=args.mode)
    score = scorer.score(features)

    missing = features.attrs.get("missing_sources", [])
    print(f"escaneo de huella pre-anuncio sobre {market!r}")
    print(
        f"  {len(features)} eventos con features; score calculable en "
        f"{int(score.notna().sum())} (modo {args.mode})"
    )
    if missing:
        print(f"  fuentes ausentes (features NaN, nunca inventadas): {', '.join(missing)}")

    table = features[["ticker", "event_date"]].assign(score=score)
    top = table.sort_values("score", ascending=False).head(args.top)
    print(f"\n  top {len(top)} eventos por score (datos públicos; ver nota legal de")
    print("  earnings_alpha.events e informed_trading.md):")
    for event_id, row in top.iterrows():
        day = row["event_date"]
        day_txt = day.date().isoformat() if hasattr(day, "date") else str(day)
        print(f"    {event_id!s:<28} {row['ticker']:<8} {day_txt}  score {row['score']:+.3f}")

    if args.ground_truth:
        from earnings_alpha.events.surprise_model import roc_auc

        leaked = set(market.leaked_event_ids())
        labels = features.index.to_series().isin(leaked)
        mask = score.notna()
        auc, se = roc_auc(score[mask], labels[mask])
        prevalence = float(labels[mask].mean())
        k = int(labels[mask].sum())
        top_k = set(score[mask].sort_values(ascending=False).head(k).index)
        precision_at_k = len(top_k & leaked) / k if k else float("nan")
        print("\n  validación contra la verdad-terreno del generador:")
        print(f"    eventos con filtración inyectada: {k} de {int(mask.sum())} "
              f"(prevalencia {100.0 * prevalence:.1f}%)")
        print(f"    AUC-ROC {auc:.3f} (SE bajo H0 de no discriminación: {se:.3f}, "
              "Hanley-McNeil 1982)")
        print(f"    precisión@{k}: {precision_at_k:.2f}")
        ppv = positive_predictive_value(0.02, 0.80, 0.90)
        print(
            "    recordatorio de tasa base (informed_trading.md §12.4): con "
            f"prevalencia real ~2%, Se=0.80/Sp=0.90 dan PPV≈{ppv:.2f}; el score "
            "pondera exposición, no dispara alarmas binarias"
        )

    if args.output is not None:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        dump = features.assign(informed_trading_score=score).reset_index()
        if out.suffix == ".parquet":
            dump.to_parquet(out, index=False)
        else:
            dump.to_csv(out, index=False)
        print(f"\n  features + score escritos en {out} ({_human_size(out.stat().st_size)})")
    return 0


# ===========================================================================
# backtest run
# ===========================================================================


def _cross_backtest(market: SyntheticMarket, scores: pd.Series, args: argparse.Namespace):
    from earnings_alpha.backtest import CostModel, CrossSectionalBacktest

    engine = CrossSectionalBacktest(sectors=market.sectors())
    return engine.run(
        scores,
        market.prices(),
        n_quantiles=args.quantiles,
        rebalance=args.rebalance,
        long_short=not args.long_only,
        costs=CostModel(),
    )


def _print_cross_summary(result) -> None:
    summary = result.summary()
    sharpe = summary["sharpe"]
    print("  resultado (neto de costes; CostModel realista, no un 0.05% plano):")
    print(
        f"    Sharpe anualizado   {sharpe.sharpe_annualized:+.3f}  "
        f"IC95% [{sharpe.ci_low:+.3f}, {sharpe.ci_high:+.3f}]  "
        f"PSR(0)={sharpe.psr_zero:.3f}  (método {sharpe.method}, T={sharpe.n_obs})"
    )
    print(f"    retorno anualizado  {100.0 * summary['ann_return']:+.2f}%   "
          f"volatilidad {100.0 * summary['ann_volatility']:.2f}%")
    print(f"    max drawdown        {100.0 * summary['max_drawdown']:+.2f}%   "
          f"retorno total neto {100.0 * summary['total_net_return']:+.2f}%")
    print(f"    rotación one-way    {summary['annual_one_way_turnover']:.2f}x/año   "
          f"rebalanceos {summary['n_rebalances']} (omitidos {summary['n_skipped']})")
    costs = summary["total_costs"]
    print(f"    costes acumulados   total {100.0 * costs['total']:.3f}% del NAV "
          f"(spread {100.0 * costs['spread']:.3f}%, impacto {100.0 * costs['impact']:.3f}%)")
    if summary["n_silent_delistings"]:
        print(f"    AVISO: {summary['n_silent_delistings']} delistings sin retorno final "
              "conocido (auditar result.silent_delistings)")


def _event_grid(market: SyntheticMarket, args: argparse.Namespace) -> pd.DataFrame:
    from earnings_alpha.backtest import run_grid

    events = market.events()
    return run_grid(
        events,
        market.prices(),
        entry_offsets=args.entry_offsets,
        exit_offsets=args.exit_offsets,
        score=args.score,
        side=args.side,
        calendar_known_in_advance=args.calendar_known,
        n_boot=args.n_boot,
    )


_GRID_PRINT_COLS = (
    "status",
    "n_events",
    "hit_rate",
    "mean_net",
    "mean_net_ci_low",
    "mean_net_ci_high",
    "p05",
    "p95",
    "gap_share_log",
)


def _print_grid(grid: pd.DataFrame) -> None:
    cols = [c for c in _GRID_PRINT_COLS if c in grid.columns]
    view = grid[cols]
    print("  rejilla entrada/salida (retornos netos por evento, banda por bootstrap")
    print("  estacionario; percentiles porque el retorno de evento tiene colas gruesas):")
    print("    " + view.to_string(float_format=lambda v: f"{v:+.4f}").replace("\n", "\n    "))
    n_ok = int((grid.get("status") == "ok").sum()) if "status" in grid.columns else len(grid)
    print(
        f"  multiplicidad: {len(grid)} combinaciones = {len(grid)} pruebas sobre los mismos "
        "datos; antes de elegir la mejor celda, Benjamini-Hochberg o Sharpe deflactado "
        f"con n_trials={n_ok} (validation_methodology.md §7)"
    )


def _cmd_backtest_run(args: argparse.Namespace) -> int:
    market = _build_market(args)
    if args.mode == "cross":
        print(f"backtest cross-section de {args.factor!r} sobre {market!r}")
        scores = _factor_scores(market, args.factor)
        result = _cross_backtest(market, scores, args)
        _print_cross_summary(result)
        if args.report is not None:
            path = _write_cross_report(market, scores, result, args)
            print(f"  tearsheet escrito en {path}")
        return 0

    print(f"backtest de eventos (score {args.score!r}, lado {args.side!r}) sobre {market!r}")
    if any(off < 0 for off in args.entry_offsets) and not args.calendar_known:
        print(
            "  nota PIT: hay offsets de entrada negativos (pre-posicionamiento) y no se "
            "ha pasado --calendar-known; el motor lo rechazará con LookAheadError "
            "(pit_and_biases.md §8.3), porque operar antes del anuncio exige declarar "
            "que la fecha se conocía por adelantado"
        )
    grid = _event_grid(market, args)
    _print_grid(grid)
    if args.report is not None:
        path = _write_event_report(market, grid, args)
        print(f"  tearsheet escrito en {path}")
    return 0


# ===========================================================================
# collector (passthrough)
# ===========================================================================


def _cmd_collector(args: argparse.Namespace) -> int:
    """Delega en `earnings_alpha.collector.main` sin duplicar sus opciones.

    El recolector ya trae su propio CLI probado (``run --dry-run``, ``status``,
    ``verify``); duplicar aquí sus flags crearía dos fuentes de verdad. Con
    ``--options-source synthetic --calendar-provider synthetic`` funciona sin
    red ni credenciales.
    """
    from earnings_alpha.collector import main as collector_main

    if not args.collector_args:
        print(
            "uso: earnings-alpha collector run --dry-run [opciones]  (o status|verify);\n"
            "las opciones son las de `python3 -m earnings_alpha.collector`",
            file=sys.stderr,
        )
        return 2
    return collector_main(args.collector_args)


# ===========================================================================
# report (tearsheet completo)
# ===========================================================================


def _hit_rate_metric(grid: pd.DataFrame):
    """Hit rate de la mejor celda 'ok' con IC binomial normal (diagnóstico)."""
    import math as _math

    from earnings_alpha.reports import MetricWithCI

    ok = grid[grid.get("status") == "ok"] if "status" in grid.columns else grid
    if len(ok) == 0 or "mean_net" not in ok.columns:
        return []
    best = ok.sort_values("mean_net", ascending=False).iloc[0]
    entry, exit_ = best.name if isinstance(best.name, tuple) else ("?", "?")
    label = f"mejor celda de la rejilla (entrada T{entry:+d}, salida T{exit_:+d})"
    out = [
        MetricWithCI(
            label=f"retorno neto medio/evento — {label}",
            value=float(best["mean_net"]),
            ci_low=float(best.get("mean_net_ci_low", float("nan"))),
            ci_high=float(best.get("mean_net_ci_high", float("nan"))),
            unit="%",
            note="bootstrap estacionario sobre eventos en orden temporal; OJO multiplicidad",
        )
    ]
    p, n = float(best.get("hit_rate", float("nan"))), int(best.get("n_events", 0))
    if _math.isfinite(p) and n > 0:
        half = 1.96 * _math.sqrt(max(p * (1.0 - p), 0.0) / n)
        out.append(
            MetricWithCI(
                label=f"hit rate — {label}",
                value=p,
                ci_low=max(p - half, 0.0),
                ci_high=min(p + half, 1.0),
                unit="%",
                decimals=2,
                note=f"IC binomial normal, n={n} eventos",
            )
        )
    return out


def _caar_frame(market: SyntheticMarket, args: argparse.Namespace) -> pd.DataFrame:
    """CAAR por tau agrupado por cubo de SUE (Ball y Brown 1968; PEAD de
    Bernard y Thomas 1989 como referencia del drift posterior)."""
    from earnings_alpha.events import aar_caar, abnormal_returns, caar_by_group, quantile_groups

    events = market.events()
    ar = abnormal_returns(
        market.prices(),
        events,
        model="market",
        market=market.market_index(),
        pre=args.pre,
        post=args.post,
    )
    if args.caar_groups <= 1:
        return aar_caar(ar)
    groups = quantile_groups(events.set_index("event_id")["sue"], q=args.caar_groups)
    return caar_by_group(ar, groups)


def _write_cross_report(market, scores, result, args) -> Path:
    from earnings_alpha.reports import ic_metrics, sharpe_metrics, write_tearsheet

    ic, summary = _ic_block(
        market, scores, horizon=args.horizon, min_names=args.min_names
    )
    metrics = [*sharpe_metrics(result.sharpe()), *ic_metrics(summary)]
    return write_tearsheet(
        args.report,
        title=f"earnings-alpha — {args.factor} (cross-section)",
        subtitle=f"backtest {args.rebalance}, {args.quantiles} cubos, mercado sintético",
        returns=result.returns,
        ic=ic,
        ic_summary=summary,
        quantile_returns=result.quantile_returns,
        metrics=metrics,
        params=_echo_params(market, args),
    )


def _write_event_report(market, grid, args) -> Path:
    from earnings_alpha.reports import write_tearsheet

    return write_tearsheet(
        args.report,
        title="earnings-alpha — backtest de eventos",
        subtitle=f"score {args.score}, lado {args.side}, mercado sintético",
        caar=_caar_frame(market, args),
        grid=grid,
        metrics=_hit_rate_metric(grid),
        params=_echo_params(market, args),
    )


def _echo_params(market: SyntheticMarket, args: argparse.Namespace) -> dict[str, object]:
    """Eco de parámetros para la sección de reproducibilidad del tearsheet."""
    out: dict[str, object] = {
        "seed": market.seed,
        "n_tickers": market.n_tickers,
        "start": market.start.isoformat(),
        "end": market.end.isoformat(),
        "leak_fraction": market.leak_fraction,
        "proveedor": "synthetic (sin red ni credenciales; contrato §0.5)",
    }
    for key in (
        "factor",
        "horizon",
        "rebalance",
        "quantiles",
        "score",
        "side",
        "entry_offsets",
        "exit_offsets",
        "calendar_known",
        "pre",
        "post",
        "caar_groups",
    ):
        if hasattr(args, key):
            out[key] = getattr(args, key)
    return out


def _cmd_report(args: argparse.Namespace) -> int:
    from earnings_alpha.reports import ic_metrics, sharpe_metrics, write_tearsheet

    market = _build_market(args)
    print(f"tearsheet completo sobre {market!r}")

    print("  [1/4] factor y serie de IC...")
    scores = _factor_scores(market, args.factor)
    ic, summary = _ic_block(market, scores, horizon=args.horizon, min_names=args.min_names)

    print("  [2/4] backtest cross-section...")
    result = _cross_backtest(market, scores, args)
    sharpe = result.sharpe()

    print("  [3/4] estudio de eventos (CAAR) y rejilla de entrada/salida...")
    caar = _caar_frame(market, args)
    grid = _event_grid(market, args)

    print("  [4/4] render HTML autocontenido...")
    metrics = [
        *sharpe_metrics(sharpe),
        *ic_metrics(summary),
        *_hit_rate_metric(grid),
    ]
    path = write_tearsheet(
        args.output,
        title=args.title,
        subtitle=(
            f"factor {args.factor}, rebalanceo {args.rebalance}, "
            f"rejilla {args.entry_offsets}x{args.exit_offsets} — todo sobre el "
            "proveedor sintético, sin credenciales"
        ),
        returns=result.returns,
        ic=ic,
        ic_summary=summary,
        quantile_returns=result.quantile_returns,
        caar=caar,
        grid=grid,
        metrics=metrics,
        params=_echo_params(market, args),
    )
    size = _human_size(path.stat().st_size)
    print(f"  escrito {path} ({size}); secciones: equity, drawdown, ic, quantiles, "
          "caar, grid, metrics, params")
    return 0


# ===========================================================================
# Parser
# ===========================================================================


def build_parser() -> argparse.ArgumentParser:
    """Construye el parser completo del CLI (público para tests y documentación)."""
    parser = argparse.ArgumentParser(
        prog="earnings-alpha",
        description=(
            "Plataforma de investigación sobre fundamentales y eventos de resultados "
            "del S&P 500. Todos los subcomandos de cómputo funcionan sin red ni "
            "credenciales contra el proveedor sintético."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ------------------------------------------------------------- universe
    uni = sub.add_parser("universe", help="universo S&P 500 point-in-time")
    uni_sub = uni.add_subparsers(dest="subcommand", required=True)

    show = uni_sub.add_parser("show", help="pertenencia PIT en una fecha")
    show.add_argument("--date", type=_parse_date, default=None,
                      help="fecha de consulta (por defecto: último snapshot)")
    show.add_argument("--limit", type=int, default=20, help="tamaño de la muestra de tickers")
    show.add_argument("--history", type=Path, default=None,
                      help="CSV de composición histórica alternativo")
    show.set_defaults(func=_cmd_universe_show)

    refresh = uni_sub.add_parser(
        "refresh",
        help="refresco append-only del histórico (jamás reescribe fechas registradas)",
    )
    refresh.add_argument("--dry-run", action="store_true",
                         help="calcula el informe completo sin escribir nada")
    refresh.add_argument("--source", action="append", default=None, metavar="NAME",
                         help="fuente viva (github, wikipedia, slickcharts); repetible")
    refresh.add_argument("--from-csv", type=Path, default=None,
                         help="fuente OFFLINE: CSV local (date,tickers) descargado a mano")
    refresh.add_argument("--history", type=Path, default=None,
                         help="CSV de composición histórica a actualizar (por defecto el semilla)")
    refresh.set_defaults(func=_cmd_universe_refresh)

    # ------------------------------------------------------------- data
    data = sub.add_parser("data", help="estado de la capa de datos")
    data_sub = data.add_subparsers(dest="subcommand", required=True)
    status = data_sub.add_parser(
        "status", help="credenciales por proveedor, datasets locales y caché"
    )
    status.set_defaults(func=_cmd_data_status)

    # ------------------------------------------------------------- factors
    fac = sub.add_parser("factors", help="factores fundamentales cross-section (ángulo A)")
    fac_sub = fac.add_subparsers(dest="subcommand", required=True)

    flist = fac_sub.add_parser("list", help="factores registrados y sus requisitos de datos")
    flist.set_defaults(func=_cmd_factors_list)

    fcomp = fac_sub.add_parser(
        "compute", help="computa un factor sobre el sintético y reporta su IC con banda"
    )
    fcomp.add_argument("factor", help="nombre del factor (ver `factors list`)")
    _add_market_options(fcomp)
    fcomp.add_argument("--horizon", type=int, default=5,
                       help="horizonte del retorno forward para la IC (sesiones)")
    fcomp.add_argument("--min-names", type=int, default=8,
                       help="mínimo de nombres por fecha para que la IC del día cuente")
    fcomp.add_argument("--method", choices=["spearman", "pearson"], default="spearman",
                       help="rank IC (por defecto) o IC de Pearson (diagnóstico)")
    fcomp.add_argument("--output", type=Path, default=None,
                       help="vuelca el panel (date,ticker,valor) a CSV o parquet")
    fcomp.set_defaults(func=_cmd_factors_compute)

    # ------------------------------------------------------------- events
    evt = sub.add_parser("events", help="ventana de resultados y flujo informado (ángulo B)")
    evt_sub = evt.add_subparsers(dest="subcommand", required=True)
    scan = evt_sub.add_parser(
        "scan",
        help="huella de negociación informada pre-anuncio (solo datos públicos)",
        description=(
            "Calcula PreEventFeatures + InformedTradingScore sobre el mercado "
            "sintético y, con --ground-truth, valida el detector contra los "
            "eventos con filtración inyectada (leaked_event_ids). Todas las "
            "features derivan de datos públicos; ver la nota legal de "
            "earnings_alpha.events."
        ),
    )
    _add_market_options(scan, leak_default=0.25)
    scan.add_argument("--mode", choices=["intensity", "directional"], default="intensity",
                      help="intensity = cuánta actividad anómala; directional = hacia dónde")
    scan.add_argument("--top", type=int, default=10, help="eventos a mostrar por score")
    scan.add_argument("--ground-truth", action="store_true",
                      help="valida contra la verdad-terreno del generador (AUC, precisión@k)")
    scan.add_argument("--output", type=Path, default=None,
                      help="vuelca features + score a CSV o parquet")
    scan.set_defaults(func=_cmd_events_scan)

    # ------------------------------------------------------------- backtest
    bt = sub.add_parser("backtest", help="motores de backtest")
    bt_sub = bt.add_subparsers(dest="subcommand", required=True)
    run = bt_sub.add_parser("run", help="ejecuta un backtest sobre el sintético")
    run.add_argument("--mode", choices=["cross", "event"], default="cross",
                     help="cross = carteras de cubos (ángulo A); event = offsets de sesión (B)")
    _add_market_options(run)
    run.add_argument("--factor", default="sue_analyst",
                     help="factor del registro para --mode cross")
    run.add_argument("--rebalance", default="W-FRI", help="regla de rebalanceo (D, W-FRI, ME)")
    run.add_argument("--quantiles", type=int, default=5, help="número de cubos")
    run.add_argument("--long-only", action="store_true",
                     help="solo pata larga (por defecto long-short)")
    run.add_argument("--horizon", type=int, default=5,
                     help="horizonte de la IC del informe (--report)")
    run.add_argument("--min-names", type=int, default=8,
                     help="mínimo de nombres por fecha para la IC del informe")
    run.add_argument("--score", default="sue",
                     help="columna de events() usada como score en --mode event")
    run.add_argument("--side", choices=["long", "short", "signed"], default="signed",
                     help="lado por evento; signed = el signo del score")
    run.add_argument("--entry-offsets", type=_parse_int_list, default=[1],
                     help="offsets de entrada en sesiones, p. ej. '-5,1' (negativo = antes)")
    run.add_argument("--exit-offsets", type=_parse_int_list, default=[5, 20],
                     help="offsets de salida en sesiones, p. ej. '5,20,60'")
    run.add_argument("--calendar-known", action="store_true",
                     help="declara que la fecha del anuncio se conocía por adelantado "
                          "(obligatorio para pre-posicionarse; pit_and_biases.md §8.3)")
    run.add_argument("--n-boot", type=int, default=500,
                     help="réplicas bootstrap de las bandas por evento")
    run.add_argument("--pre", type=int, default=10, help="sesiones pre-evento del CAAR (--report)")
    run.add_argument("--post", type=int, default=30,
                     help="sesiones post-evento del CAAR (--report)")
    run.add_argument("--caar-groups", type=int, default=3,
                     help="cubos de SUE de las curvas CAAR (--report); 1 = una sola curva")
    run.add_argument("--report", type=Path, default=None,
                     help="además de imprimir, escribe el tearsheet HTML en esta ruta")
    run.set_defaults(func=_cmd_backtest_run)

    # ------------------------------------------------------------- collector
    col = sub.add_parser(
        "collector",
        help="recolector diario point-in-time (passthrough a earnings_alpha.collector)",
    )
    col.add_argument("collector_args", nargs=argparse.REMAINDER,
                     help="argumentos del recolector, p. ej.: run --dry-run "
                          "--options-source synthetic --calendar-provider synthetic")
    col.set_defaults(func=_cmd_collector)

    # ------------------------------------------------------------- report
    rep = sub.add_parser(
        "report",
        help="tearsheet HTML completo (equity, drawdown, IC, quintiles, CAAR, rejilla)",
    )
    _add_market_options(rep)
    rep.add_argument("--output", type=Path, default=Path("earnings_alpha_tearsheet.html"),
                     help="ruta del HTML de salida")
    rep.add_argument("--title", default="earnings-alpha — informe de investigación",
                     help="título del documento")
    rep.add_argument("--factor", default="sue_analyst", help="factor del bloque cross-section")
    rep.add_argument("--horizon", type=int, default=5, help="horizonte de la IC (sesiones)")
    rep.add_argument("--min-names", type=int, default=8,
                     help="mínimo de nombres por fecha para la IC")
    rep.add_argument("--rebalance", default="W-FRI", help="regla de rebalanceo")
    rep.add_argument("--quantiles", type=int, default=5, help="número de cubos")
    rep.add_argument("--long-only", action="store_true", help="solo pata larga")
    rep.add_argument("--score", default="sue", help="score del motor de eventos")
    rep.add_argument("--side", choices=["long", "short", "signed"], default="signed",
                     help="lado por evento en la rejilla")
    rep.add_argument("--entry-offsets", type=_parse_int_list, default=[1, 3],
                     help="offsets de entrada de la rejilla")
    rep.add_argument("--exit-offsets", type=_parse_int_list, default=[10, 30],
                     help="offsets de salida de la rejilla")
    rep.add_argument("--calendar-known", action="store_true",
                     help="declara conocida por adelantado la fecha del anuncio")
    rep.add_argument("--n-boot", type=int, default=300,
                     help="réplicas bootstrap de las bandas por evento")
    rep.add_argument("--pre", type=int, default=10, help="sesiones pre-evento del CAAR")
    rep.add_argument("--post", type=int, default=30, help="sesiones post-evento del CAAR")
    rep.add_argument("--caar-groups", type=int, default=3,
                     help="cubos de SUE de las curvas CAAR; 1 = una sola curva")
    rep.set_defaults(func=_cmd_report)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada del CLI. Devuelve el código de salida del proceso."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else sys.argv[1:])
    try:
        return int(args.func(args))
    except ConfigError as exc:
        print(f"error de configuración: {exc}", file=sys.stderr)
        return 2
    except EarningsAlphaError as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover - módulo de arranque
    raise SystemExit(main())
