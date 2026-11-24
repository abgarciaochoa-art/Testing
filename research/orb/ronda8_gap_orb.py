"""Ronda 8: robustez de la regla elegida 'Gap ORB' en índices.

Regla base (fijada antes de esta ronda): rango de 30 min; si |hueco| > 0,2 × ATR(14) (en %), orden
stop en el extremo del rango del lado del hueco (máximo si el hueco es al alza, mínimo si es a la
baja); stop al otro lado del rango; salida al cierre. Una operación por índice y día.
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import datos  # noqa: E402
from motor import metricas  # noqa: E402
from ronda7_hueco import IDX, cartera, operaciones  # noqa: E402

pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 300)
OUT = Path(__file__).parent / "resultados"

# 1) vecindad de parámetros: t y pb netos (todos los índices)
filas = []
for n_or, k in itertools.product((10, 20, 30, 45, 60, 90), (0.1, 0.15, 0.2, 0.25, 0.3, 0.4)):
    ops = {n: operaciones(n, n_or, 0, k) for n in IDX}
    tod = pd.concat([o.r1[o.op] for o in ops.values()])
    us = pd.concat([ops[n].r1[ops[n].op] for n in ("SPX", "NDX", "RUT")])
    filas.append(dict(n_or=n_or, k=k, pb=tod.mean() * 1e4, t=tod.mean() / tod.std() * len(tod) ** .5,
                      pb_us=us.mean() * 1e4, t_us=us.mean() / us.std() * len(us) ** .5))
v = pd.DataFrame(filas)
print("== vecindad: t de las operaciones (todos los índices, coste 1 pb por lado)")
print(v.pivot(index="n_or", columns="k", values="t").round(1))
print("== pb netos por operación, todos")
print(v.pivot(index="n_or", columns="k", values="pb").round(1))
print("== t solo EE. UU.")
print(v.pivot(index="n_or", columns="k", values="t_us").round(1))

# 2) regla base: por índice y año, costes
base = {n: operaciones(n, 30, 0, 0.2, coste=0) for n in IDX}
print("\n== regla base, bruto: pb por operación y año")
t = pd.concat({n: o.r1[o.op] for n, o in base.items()}, axis=1)
print((t.groupby(t.index.year).mean() * 1e4).round(1))
print("acierto:", {n: round((o.r1[o.op] > 0).mean(), 3) for n, o in base.items()})
print("ops/año:", {n: round(o.op.sum() / (len(o) / 252), 0) for n, o in base.items()})
for c in (0.5, 1, 1.5, 2, 3):
    ops = {n: operaciones(n, 30, 0, 0.2, coste=c * 1e-4) for n in IDX}
    for grupo, nombres in (("7 índices", IDX), ("EE. UU.", ["SPX", "NDX", "RUT"]), ("Europa", ["FTSE", "CAC", "DAX", "STOXX"])):
        rd = cartera({n: ops[n] for n in nombres}, riesgo=0.005).sum(1)
        m = metricas(rd.to_numpy(), None)
        print(f"coste {c} pb/lado {grupo:10s}: CAGR {m['cagr']:.1%} Sharpe {m['sharpe']:.2f} maxDD {m['maxdd']:.0%}")

# 3) cartera EE. UU. con 1 pb: año a año y por riesgo
ops = {n: operaciones(n, 30, 0, 0.2, coste=1e-4) for n in ("SPX", "NDX", "RUT")}
for riesgo in (0.0025, 0.005, 0.0075, 0.01):
    c = cartera(ops, riesgo=riesgo, maxlev_pos=4)
    rd = c.sum(1)
    m = metricas(rd.to_numpy(), None)
    lev = pd.concat({n: np.minimum(riesgo / o["dist"], 4) for n, o in ops.items()}, axis=1)
    print(f"EE. UU. riesgo {riesgo:.2%}: CAGR {m['cagr']:.1%} vol {m['vol']:.1%} Sharpe {m['sharpe']:.2f} maxDD {m['maxdd']:.0%}"
          f" palanca media por posición {np.nanmean(lev.to_numpy()):.2f}")
c = cartera(ops, riesgo=0.005, maxlev_pos=4)
rd = c.sum(1)
print((rd.groupby(rd.index.year).apply(lambda x: (1 + x).prod() - 1) * 100).round(1).to_frame("EE. UU. riesgo 0,5 %").T)
rd.to_csv(OUT / "gap_orb_us_diario.csv")
