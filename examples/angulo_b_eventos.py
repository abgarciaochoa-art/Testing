#!/usr/bin/env python3
"""Ejemplo del ángulo B: ventana de resultados y detección de flujo informado.

Ejecuta el pipeline de eventos completo sobre el mercado sintético del repo,
que inyecta una huella de negociación informada en una fracción conocida de
los anuncios (`leak_fraction`) y publica la verdad-terreno
(`leaked_event_ids()`), de modo que la capacidad del detector es *medible*:

    universo PIT → calendario de anuncios → PreEventFeatures →
    InformedTradingScore → SurpriseModel (CV purgada) →
    EventBacktest.run_grid → estudio de eventos con CAAR por quintil de SUE

Uso::

    python3 examples/angulo_b_eventos.py             # con modelo (CV purgada)
    python3 examples/angulo_b_eventos.py --rapido    # sin modelo, rango corto

Qué mirar en la salida:

- El AUC del score de intensidad contra la verdad-terreno y su t frente a 0,5.
- La aritmética de tasa base (PPV): por qué el output es un score continuo
  para ponderar exposición y no una alerta binaria.
- La rejilla de entrada/salida: "cruzar el anuncio" vs "explotar el drift",
  con el gap overnight modelado explícitamente.
- El CAAR por quintil de SUE: la firma clásica del PEAD (Bernard-Thomas 1989).

Nota legal y metodológica (contrato §3.5): todas las features derivan de datos
PÚBLICOS (precios, volumen, cadenas de opciones, short interest agregado,
Form 4 ya publicados). El objetivo es detectar la huella estadística que el
comportamiento de otros deja en variables observables, no acceder a
información privilegiada. Ver docs/research/informed_trading.md.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ejecutable tal cual, sin instalar el paquete: se añade la raíz del repo.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from earnings_alpha.data.synthetic import SyntheticMarket  # noqa: E402
from earnings_alpha.events.preevent import positive_predictive_value  # noqa: E402
from earnings_alpha.pipeline import run_event_pipeline  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1234, help="semilla del generador")
    parser.add_argument("--tickers", type=int, default=24, help="tamaño del universo")
    parser.add_argument("--inicio", default="2019-01-02", help="primera fecha del panel")
    parser.add_argument("--fin", default="2023-12-29", help="última fecha del panel")
    parser.add_argument(
        "--filtracion", type=float, default=0.30,
        help="fracción de eventos con huella inyectada (verdad-terreno)",
    )
    parser.add_argument(
        "--rapido", action="store_true",
        help="rango corto y sin SurpriseModel (la CV purgada necesita panel largo)",
    )
    args = parser.parse_args(argv)

    if args.rapido:
        args.tickers, args.inicio, args.fin = 20, "2020-01-02", "2022-12-30"

    print(
        f"Generando mercado sintético: {args.tickers} símbolos, "
        f"{args.inicio} → {args.fin}, leak_fraction={args.filtracion} "
        f"(seed={args.seed})..."
    )
    t0 = time.time()
    market = SyntheticMarket(
        seed=args.seed,
        n_tickers=args.tickers,
        start=args.inicio,
        end=args.fin,
        leak_fraction=args.filtracion,
    )
    print("Ejecutando pipeline de eventos (features, scores, rejilla, CAAR)...")

    if args.rapido:
        kwargs: dict[str, object] = {
            "model_mode": "skip",
            "estimation": (-160, -31),
            "pre": 5,
            "post": 20,
        }
    else:
        kwargs = {"model_mode": "evaluate", "n_splits": 4}

    result = run_event_pipeline(
        market,
        # El calendario de resultados sintético es público por adelantado, así
        # que el pre-posicionamiento (entrar antes del anuncio) es declarable.
        # Con datos reales esto exige vintages de calendario (pit_and_biases §8.3).
        entry_offsets=(-5, -1, 0, 1),
        exit_offsets=(1, 5, 20),
        calendar_known_in_advance=True,
        backtest_score="directional",
        side="signed",
        caar_by="sue",
        **kwargs,  # type: ignore[arg-type]
    )
    elapsed = time.time() - t0

    print()
    print(result.summary())

    # ------------------------------------------------ tasa base y PPV (§12.4)
    if result.detection is not None and result.detection["n_leaked"] > 0:
        prev = result.detection["prevalence"]
        ppv = positive_predictive_value(prev, sensitivity=0.80, specificity=0.90)
        print()
        print("Aritmética de tasa base (informed_trading.md §12.4):")
        print(
            f"  con prevalencia {prev:.1%} y un detector Se=0.80/Sp=0.90, "
            f"la PPV sería {ppv:.2f}"
        )
        prev_real = 0.02
        ppv_real = positive_predictive_value(prev_real, 0.80, 0.90)
        print(
            f"  con la prevalencia realista del mundo real (~{prev_real:.0%}), "
            f"la PPV cae a {ppv_real:.2f}: el {1 - ppv_real:.0%} de las alertas "
            "serían falsos positivos"
        )
        print("  → por eso el score pondera exposición y no dispara alertas binarias.")

    print()
    print(f"[terminado en {elapsed:.1f}s]")
    print()
    print("Advertencia de multiplicidad: la rejilla son N ensayos sobre los mismos")
    print("datos; antes de creer la mejor celda, pásala por")
    print("stats.validation.benjamini_hochberg o deflated_sharpe_ratio(n_trials=N).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
