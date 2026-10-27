"""Noise Band · robustez: parámetros vecinos, otra fuente de datos y costes.

Para el artículo (largo y corto) y dos versiones solo largo, sin palanca:

- sigma con 10, 14 (base) y 20 sesiones;
- controles cada 15, 30 (base) y 60 minutos;
- la misma estrategia con HistData en vez de OANDA (nov-2010 → 2018, días comunes);
- costes de 0; 0,5; 1 y 2 pb por lado y los del artículo; la base es SPY con órdenes a
  mercado (0,0085 $ por acción: 0,55 pb por lado de media).

Salida: resultados/robustez.csv

Uso: python research/noise_band/robustez.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import motor as nb

OUT = Path(__file__).parent / "resultados"
HISTDATA = nb.DATOS.with_name("rth_histdata.npz")
BASE = dict(coste_por_accion=0.0085)  # SPY con órdenes a mercado
VARIANTES = {
    "articulo": dict(lados=2, trail_vwap=True, salida_banda=1),
    "largo_articulo": dict(lados=1, trail_vwap=True, salida_banda=1),
    "largo_hasta_cierre": dict(lados=1, trail_vwap=False, salida_banda=0),
}


def controles(cada: int) -> np.ndarray:
    c = np.zeros(nb.VELAS, dtype=np.bool_)
    ultimo = 374 if cada == 15 else 359
    c[29:ultimo + 1:cada] = True  # primera decisión a las 10:00
    return c


def fila(nombre: str, prueba: str, eq: pd.Series, ops: pd.DataFrame, a=None, b=None) -> dict:
    if a is not None:
        eq = eq[(eq.index >= a) & (eq.index <= b)]
    m = nb.metricas(eq, ops)
    return {"variante": nombre, "prueba": prueba, "desde": eq.index[0].date(), "hasta": eq.index[-1].date(),
            "cagr": m["cagr"], "peor_caida": m["peor_caida"], "sharpe": m["sharpe"],
            "operaciones_año": m["operaciones_año"], "ret_medio": m["ret_medio"]}


def main() -> None:
    filas = []
    base = nb.cargar()
    for nombre, kw in VARIANTES.items():
        for n in (10, 14, 20):
            d = base if n == 14 else nb.cargar(n_sigma=n)
            r = nb.simular(d, **BASE, **kw)
            filas.append(fila(nombre, f"sigma de {n} sesiones", r["equity"], r["operaciones"], "2005-01-24", "2020-12-31"))
        for cada in (15, 60):
            r = nb.simular(base, **BASE, controles=controles(cada), **kw)
            filas.append(fila(nombre, f"controles cada {cada} min", r["equity"], r["operaciones"], "2005-01-24",
                              "2020-12-31"))
        for etiqueta, kc in (("costes 0 pb", dict(coste_bps=0.0)), ("costes 0,5 pb", dict(coste_bps=0.5)),
                             ("costes 1 pb", dict(coste_bps=1.0)), ("costes 2 pb", dict(coste_bps=2.0)),
                             ("costes del artículo (0,0045 $ por acción)", dict(coste_por_accion=0.0045))):
            r = nb.simular(base, **kc, **kw)
            filas.append(fila(nombre, etiqueta, r["equity"], r["operaciones"], "2005-01-24", "2020-12-31"))
    # otra fuente de datos, en los días comunes
    h = nb.cargar(ruta=HISTDATA)
    comunes = h["fechas"].intersection(base["fechas"])
    a, b = comunes[20], comunes[-1]
    for nombre, kw in VARIANTES.items():
        for fuente, d in (("OANDA", base), ("HistData", h)):
            r = nb.simular(d, **BASE, **kw)
            filas.append(fila(nombre, f"fuente {fuente} (días comunes)", r["equity"], r["operaciones"], a, b))
    df = pd.DataFrame(filas)
    df.to_csv(OUT / "robustez.csv", index=False, float_format="%.5f")
    pd.set_option("display.width", 200)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
