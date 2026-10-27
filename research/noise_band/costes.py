"""Noise Band · ¿cuánto pesan los costes? Comparación con el modelo del artículo.

El artículo cobra 0,0035 $ de comisión y 0,001 $ de deslizamiento por acción de SPY y por
lado: con SPY entre 120 y 300 $ (2005-2020) son solo 0,15-0,4 pb por lado. Aquí se prueba:

- costes fijos de 0 a 1 pb por lado;
- el modelo del artículo (0,0045 $ por acción);
- SPY con órdenes a mercado (0,0035 $ de comisión + medio céntimo de horquilla = 0,0085 $
  por acción, ~0,3-0,7 pb por lado);

para el artículo (largo y corto) y dos versiones solo largo, sin palanca y con la del
artículo (volatilidad del 2 %, máximo 4×), desde 2005 y desde 2007 (inicio del artículo).

Salida: resultados/costes.csv

Uso: python research/noise_band/costes.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import motor as nb

OUT = Path(__file__).parent / "resultados"
VARIANTES = {
    "articulo": dict(lados=2, trail_vwap=True, salida_banda=1),
    "largo_articulo": dict(lados=1, trail_vwap=True, salida_banda=1),
    "largo_hasta_cierre": dict(lados=1, trail_vwap=False, salida_banda=0),
}
COSTES = [(f"{c} pb", dict(coste_bps=c)) for c in (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)] + [
    ("artículo: 0,0045 $ por acción", dict(coste_por_accion=0.0045)),
    ("SPY a mercado: 0,0085 $ por acción", dict(coste_por_accion=0.0085)),
]
VENTANAS = {"2005-2020": ("2005-01-24", None), "2007-2020": ("2007-01-03", None),
            "validacion 2005-12": ("2005-01-24", "2013-01-01"), "busqueda 2013-20": ("2013-01-01", None)}


def main() -> None:
    d = nb.cargar()
    tamaños = {"1x": nb.palanca(d), "vol 2 % (máx. 4×)": nb.palanca(d, 0.02)}
    spy = np.r_[d["O"][0, 0], d["cierre"][:-1]] / 10
    print(f"coste del artículo por lado: {0.0045 / spy.max() * 1e4:.2f}-{0.0045 / spy.min() * 1e4:.2f} pb "
          f"(media {np.mean(0.0045 / spy) * 1e4:.2f}); SPY a mercado: {np.mean(0.0085 / spy) * 1e4:.2f} pb de media")
    filas = []
    for nombre, kw in VARIANTES.items():
        for tam, lev in tamaños.items():
            for etiqueta, kc in COSTES:
                r = nb.simular(d, lev=lev, **kw, **kc)
                fila = {"variante": nombre, "tamaño": tam, "costes": etiqueta}
                for vn, (a, b) in VENTANAS.items():
                    eq = r["equity"][r["equity"].index >= pd.Timestamp(a)]
                    if b is not None:
                        eq = eq[eq.index < pd.Timestamp(b)]
                    m = nb.metricas(eq)
                    fila |= {f"{vn} cagr": m["cagr"], f"{vn} peor_caida": m["peor_caida"], f"{vn} sharpe": m["sharpe"]}
                filas.append(fila)
    df = pd.DataFrame(filas)
    df.to_csv(OUT / "costes.csv", index=False, float_format="%.5f")
    pd.set_option("display.width", 250)
    for tam in tamaños:
        x = df[df.tamaño == tam]
        print(f"\n— {tam} —")
        print(x.pivot_table(index="costes", columns="variante", values="2007-2020 cagr", sort=False).round(4).to_string())
        y = x[x.variante == "articulo"].set_index("costes")
        print(y[["2005-2020 cagr", "2007-2020 cagr", "2007-2020 peor_caida", "2007-2020 sharpe", "validacion 2005-12 cagr",
                 "busqueda 2013-20 cagr"]].round(3).to_string())


if __name__ == "__main__":
    main()
