#!/usr/bin/env python3
"""Verificador de rentabilidad del PEAD — envoltorio de línea de comandos.

Uso típico en una máquina con salida a internet:

    python3 examples/verificar_pead.py --fuente stooq --desde 2010-01-01 --hasta 2022-12-31

Prueba de humo sin red (mecánica, no veredicto):

    python3 examples/verificar_pead.py --fuente sintetico

El veredicto (VIVA / DEBIL / MUERTA / NO_CONCLUYENTE) sale de criterios
pre-registrados en ``earnings_alpha/verify.py``; ver docs/VERIFICACION.md.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fuente", default="stooq", choices=["stooq", "yfinance", "sintetico"],
                   help="origen de precios (por defecto stooq, gratuito y sin clave)")
    p.add_argument("--desde", default=None, help="fecha inicial AAAA-MM-DD (por defecto 2010-01-01)")
    p.add_argument("--hasta", default=None, help="fecha final AAAA-MM-DD (por defecto 2022-12-31)")
    p.add_argument("--max-tickers", type=int, default=None,
                   help="submuestrear a N tickers (los de más eventos primero; acelera la descarga)")
    p.add_argument("--incluir-pre2010", action="store_true",
                   help="levanta la guarda de supervivencia (el resultado queda etiquetado como sesgado)")
    p.add_argument("--n-boot", type=int, default=500, help="réplicas bootstrap (500 por defecto)")
    p.add_argument("--seed", type=int, default=20260805)
    p.add_argument("--salida", default=None, help="ruta opcional donde guardar el informe en texto")
    args = p.parse_args(argv)

    from earnings_alpha.verify import imprimir_informe, verificar_pead

    if args.fuente != "sintetico":
        n = args.max_tickers or 500
        print(f"[verificar_pead] fuente={args.fuente}: descarga de ~{n} tickers de precios "
              f"diarios; la primera ejecución puede tardar 20-60 min (la caché de disco hace "
              f"que las siguientes sean minutos).", flush=True)

    t0 = time.time()
    resultado = verificar_pead(
        fuente=args.fuente,
        desde=args.desde,
        hasta=args.hasta,
        max_tickers=args.max_tickers,
        incluir_pre2010=args.incluir_pre2010,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    texto = imprimir_informe(resultado)
    print(f"\n[terminado en {time.time() - t0:,.1f}s]")
    if args.salida:
        with open(args.salida, "w", encoding="utf-8") as fh:
            fh.write(texto + "\n")
        print(f"informe guardado en {args.salida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
