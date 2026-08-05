#!/usr/bin/env python3
"""Ejemplo del ángulo A: factores fundamentales cross-section, de punta a punta.

Ejecuta el pipeline continuo completo sobre el mercado sintético del repo
(`earnings_alpha.data.synthetic.SyntheticMarket`), que no necesita red ni
credenciales y reproduce la misma forma y semántica point-in-time que los
proveedores reales:

    universo PIT → fundamentales → factores → neutralización → combinación
    → backtest cross-section → métricas (IC con t de Newey-West, Sharpe con banda)

Uso::

    python3 examples/angulo_a_factores.py            # configuración por defecto
    python3 examples/angulo_a_factores.py --rapido   # universo y rango reducidos

Qué mirar en la salida:

- La IC de cada factor **con su banda**: una IC media sin intervalo no se
  acepta en este repo (`docs/ARCHITECTURE.md` §3.8).
- El desglose de costes del backtest: spread + impacto + comisión + préstamo,
  nada de "0.05 % fijo".
- La advertencia de multiplicidad final: cada variante probada es un ensayo.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ejecutable tal cual, sin instalar el paquete: se añade la raíz del repo.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from earnings_alpha.data.synthetic import SyntheticMarket  # noqa: E402
from earnings_alpha.pipeline import (  # noqa: E402
    DEFAULT_CONTINUOUS_FACTORS,
    run_continuous_pipeline,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260803, help="semilla del generador")
    parser.add_argument("--tickers", type=int, default=40, help="tamaño del universo")
    parser.add_argument("--inicio", default="2020-01-02", help="primera fecha del panel")
    parser.add_argument("--fin", default="2023-12-29", help="última fecha del panel")
    parser.add_argument(
        "--rapido", action="store_true",
        help="universo y rango reducidos, solo factores de sorpresa (segundos, no minutos)",
    )
    args = parser.parse_args(argv)

    if args.rapido:
        args.tickers, args.inicio, args.fin = 24, "2021-01-04", "2022-12-30"
        factors: tuple[str, ...] = ("sue_analyst", "pead")
    else:
        factors = DEFAULT_CONTINUOUS_FACTORS

    print(
        f"Generando mercado sintético: {args.tickers} símbolos, "
        f"{args.inicio} → {args.fin} (seed={args.seed})..."
    )
    t0 = time.time()
    market = SyntheticMarket(
        seed=args.seed, n_tickers=args.tickers, start=args.inicio, end=args.fin
    )
    print(f"Factores: {', '.join(factors)}")
    print("Ejecutando pipeline continuo (esto tarda unos segundos)...")

    result = run_continuous_pipeline(
        market,
        factors=factors,
        neutralize_by=("sector", "size"),
        combine_method="weights",       # pesos iguales: la línea base honesta
        forward_horizon=5,
        rebalance="W-FRI",
        n_quantiles=5,
        long_short=True,
        max_weight=0.10,                # universo reducido: tope por nombre holgado
    )
    elapsed = time.time() - t0

    print()
    print(result.summary())
    print()
    print(f"[terminado en {elapsed:.1f}s]")
    print()
    print("Recordatorio: este ejemplo corre sobre datos SINTÉTICOS (banco de")
    print("pruebas offline). Con datos reales, el universo debe ser el S&P 500")
    print("point-in-time (universe.SP500Universe) y la lista de factores, la del")
    print("registro completo: earnings_alpha.factors.default_registry.names().")
    return 0


if __name__ == "__main__":
    sys.exit(main())
