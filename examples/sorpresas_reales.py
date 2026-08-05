#!/usr/bin/env python3
"""Sorpresas de resultados REALES del S&P 500 desde el dataset de consenso del repo.

Sin red y sin credenciales: usa `data/external/consenso/consenso_master.parquet`
(56.971 filas agregadas de varias fuentes; ~500 tickers, 1995-2022) a través de
`factors.load_consensus_events`, que canoniza los eventos con la misma forma que
`SyntheticMarket.events()` y deriva la sesión negociable con `pit.tradable_date`
(BMO → misma sesión; AMC/UNKNOWN → la siguiente, la política conservadora).

Muestra:

1. La ADVERTENCIA PIT del dataset (obligatoria de propagar): consenso FINAL
   previo al anuncio sin vintages, y supervivencia parcial de tickers.
2. La distribución de sorpresas (beat/miss/meet, percentiles, beat rate por año:
   la firma del walk-down de Richardson, Teoh y Wysocki 2004).
3. El calendario BMO/AMC: reparto por sesión, día de la semana y mes.
4. El SUE de analistas (Livnat y Mendenhall 2006) calculado con
   `factors.analyst_sue_events` sobre las sorpresas reales.

Uso::

    python3 examples/sorpresas_reales.py
    python3 examples/sorpresas_reales.py --tickers AAPL MSFT KO --desde 2005-01-01
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Ejecutable tal cual, sin instalar el paquete: se añade la raíz del repo.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from earnings_alpha.factors import (  # noqa: E402
    CONSENSUS_PIT_WARNING,
    analyst_sue_events,
    load_consensus_events,
)
from earnings_alpha.types import SurpriseBasis  # noqa: E402


def _seccion(titulo: str) -> None:
    print()
    print("=" * 72)
    print(titulo)
    print("=" * 72)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="*", default=None, help="subconjunto de símbolos")
    parser.add_argument("--desde", default=None, help="primera fecha de anuncio (AAAA-MM-DD)")
    parser.add_argument("--hasta", default=None, help="última fecha de anuncio (AAAA-MM-DD)")
    args = parser.parse_args(argv)

    t0 = time.time()
    print("Cargando eventos reales desde consenso_master.parquet...")
    events = load_consensus_events(
        tickers=args.tickers or None, start=args.desde, end=args.hasta
    )

    # ---------------------------------------------------- 1. advertencia PIT
    _seccion("1. ADVERTENCIA POINT-IN-TIME DEL DATASET (léase antes que nada)")
    print(events.attrs.get("pit_warning", CONSENSUS_PIT_WARNING))
    print(
        f"[deduplicación: {events.attrs.get('n_duplicates_collapsed', '?')} filas "
        f"duplicadas colapsadas; {events.attrs.get('n_dropped_no_period_end', '?')} "
        "filas sin fiscal_period_end descartadas; "
        f"{events.attrs.get('n_session_filled_across_sources', '?')} sesiones "
        "propagadas entre fuentes]"
    )

    # ------------------------------------------------------------ 2. resumen
    _seccion("2. PANORAMA DEL DATASET")
    dates = pd.DatetimeIndex(events["event_date"])
    print(f"Eventos únicos (ticker, trimestre fiscal): {len(events):,}")
    print(f"Símbolos: {events['ticker'].nunique()}")
    print(f"Fechas negociables: {dates.min().date()} → {dates.max().date()}")
    with_both = events.dropna(subset=["eps_actual", "eps_estimate"])
    print(f"Eventos con EPS publicado Y consenso: {len(with_both):,}")

    # ------------------------------------------------------- 3. la sorpresa
    _seccion("3. DISTRIBUCIÓN DE SORPRESAS (EPS publicado - consenso final)")
    surp = with_both["eps_surprise"].astype(float)
    beat = float((surp > 0).mean())
    miss = float((surp < 0).mean())
    meet = float((surp == 0).mean())
    print(f"Beat: {beat:.1%} | Miss: {miss:.1%} | Meet exacto: {meet:.1%}")
    print(
        "El sesgo hacia el beat es la firma del walk-down del consenso "
        "(Richardson, Teoh y Wysocki 2004): los analistas parten optimistas y "
        "terminan justo por debajo de lo que la empresa puede batir."
    )
    pct = with_both["surprise_pct"].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    qs = pct.quantile([0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99])
    print()
    print("Percentiles de la sorpresa porcentual (surprise_pct, %):")
    print(
        "  " + " | ".join(f"p{int(q * 100):02d}={v:+.1f}" for q, v in qs.items())
    )
    print(
        "  (colas gruesas: la media no describe esta distribución; cualquier "
        "denominador cercano a cero la hace explotar — informe §2.4)"
    )
    print()
    by_year = (
        with_both.assign(year=pd.DatetimeIndex(with_both["event_date"]).year)
        .groupby("year")["eps_surprise"]
        .agg(n="size", beat_rate=lambda s: float((s > 0).mean()))
    )
    by_year = by_year[by_year["n"] >= 100]
    print("Beat rate por año (años con >= 100 eventos):")
    for chunk_start in range(0, len(by_year), 7):
        chunk = by_year.iloc[chunk_start : chunk_start + 7]
        print(
            "  "
            + " | ".join(
                f"{y}: {row.beat_rate:.0%} (n={row.n:,.0f})" for y, row in chunk.iterrows()
            )
        )

    # ------------------------------------------------- 4. calendario BMO/AMC
    _seccion("4. CALENDARIO DE ANUNCIOS: SESIÓN, DÍA DE LA SEMANA Y MES")
    sess_counts = events["session"].value_counts()
    total = float(len(events))
    print("Sesión del anuncio (report_time del proveedor, normalizada):")
    for name, label in (
        ("bmo", "BMO (antes de la apertura; negociable el MISMO día)"),
        ("amc", "AMC (tras el cierre; negociable la sesión SIGUIENTE)"),
        ("dmh", "DMH (durante la sesión, raro)"),
        ("unknown", "UNKNOWN (sin etiqueta → tratada como AMC, política conservadora)"),
    ):
        n = int(sess_counts.get(name, 0))
        print(f"  {label:<62} {n:>7,} ({n / total:.1%})")
    print()
    weekdays = ["lunes", "martes", "miércoles", "jueves", "viernes"]
    wd = pd.Categorical(dates.weekday, categories=range(5))
    wd_counts = pd.Series(wd).value_counts().sort_index()
    print("Día de la semana de la sesión negociable:")
    print(
        "  "
        + " | ".join(
            f"{weekdays[i]}: {wd_counts.get(i, 0) / total:.1%}" for i in range(5)
        )
    )
    print()
    month_counts = pd.Series(dates.month).value_counts().sort_index()
    meses = [
        "ene", "feb", "mar", "abr", "may", "jun",
        "jul", "ago", "sep", "oct", "nov", "dic",
    ]
    print("Mes de la sesión negociable (las 4 temporadas de resultados):")
    print(
        "  "
        + " | ".join(
            f"{meses[m - 1]}: {month_counts.get(m, 0) / total:.1%}" for m in range(1, 13)
        )
    )

    # ------------------------------------------------- 5. SUE de analistas
    _seccion("5. SUE DE ANALISTAS (Livnat-Mendenhall 2006) SOBRE SORPRESAS REALES")
    print(
        "SUE = (sorpresa - media de las 8 anteriores) / sd de las 8 anteriores; "
        "cada evento exige histórico previo del propio ticker."
    )
    sue_table = analyst_sue_events(events, basis=SurpriseBasis.SIGMA)
    sue = sue_table["sue"].dropna()
    print(f"Eventos con SUE computable: {len(sue):,} de {len(events):,}")
    # Sin panel de precios no se puede aplicar el suelo de sigma del informe
    # (max(sigma, 0.005·P_{t^-})): un ticker que clava el consenso 8 trimestres
    # seguidos tiene sd ~ 1e-16 y el cociente fabrica SUEs de 1e14. Se recortan
    # del listado y se cuentan a la vista — es exactamente la trampa que el
    # suelo existe para evitar (fundamental_factors.md §2.2).
    degenerate = sue.abs() > 50.0
    if int(degenerate.sum()):
        print(
            f"  ({int(degenerate.sum())} SUE degenerados con |SUE| > 50 excluidos "
            "del listado: sigma casi nula y sin precios para aplicarle el suelo "
            "max(sigma, 0.005·P); con el panel de precios real no ocurriría)"
        )
    sue = sue[~degenerate]
    qs = sue.quantile([0.05, 0.25, 0.50, 0.75, 0.95])
    print(
        "Percentiles del SUE: "
        + " | ".join(f"p{int(q * 100):02d}={v:+.2f}" for q, v in qs.items())
    )
    merged = sue_table.loc[sue.index].merge(
        events[["event_id", "session"]], on="event_id", how="left"
    )
    print()
    print("Los 5 SUE más extremos de cada cola:")
    extreme = pd.concat([merged.nsmallest(5, "sue"), merged.nlargest(5, "sue")])
    for row in extreme.itertuples(index=False):
        print(
            f"  {row.event_id:<28} sesión={row.session:<8} "
            f"consenso={row.consensus:+.2f} sorpresa={row.surprise:+.2f} "
            f"SUE={row.sue:+.2f}"
        )
    print()
    print(
        "Uso legítimo de esta tabla: factor SUE del ángulo A (proyectado a panel "
        "con pit.tradable_date) y quintiles del CAAR del ángulo B. Uso ILEGÍTIMO: "
        "momentum de revisiones — este dataset no tiene vintages intra-trimestre "
        "(ver la advertencia PIT de la sección 1)."
    )

    print()
    print(f"[terminado en {time.time() - t0:.1f}s]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
