"""Ronda 10: la familia ORB entera en SPY 1 min (2000 → oct-2025), costes de SPY (0,02 $/acción por lado).
Cuatro periodos: 2000-04, 2005-12, 2013-may 2020, jun 2020-2025.
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import datos  # noqa: E402
from motor import atr_diario, metricas, orb  # noqa: E402

x = datos.indice("SPY")
D = len(x["fechas"])
a = atr_diario(x["O"], x["H"], x["L"], x["C"], x["ultimo"])
prev = x["prev"]
hueco = x["O"][:, 0] / prev - 1
idx = pd.DatetimeIndex(x["fechas"])
P = [("00-04", "2000", "2004-12-31"), ("05-12", "2005", "2012-12-31"), ("13-20", "2013", "2020-05-13"),
     ("20-25", "2020-05-14", "2026")]
filas = []
for n_or, modo, filtro, sf, tR in itertools.product((5, 15, 30, 60), (0, 1, 3), ("ninguno", "hueco0.2", "hueco0.4", "contra_hueco0.2"),
                                                     (1.0, 0.5), (0.0, 1.0, 2.0)):
    if filtro == "ninguno":
        d, pe, ps, rie, ve, mot = orb(x["O"], x["H"], x["L"], x["C"], x["ultimo"].astype(np.int64), n_or, modo, 0, sf, a,
                                      0.0, tR, 389, 0.0, 1e12)
    else:
        k = float(filtro[-3:])
        res = {l: orb(x["O"], x["H"], x["L"], x["C"], x["ultimo"].astype(np.int64), n_or, modo, l, sf, a, 0.0, tR, 389,
                      0.0, 1e12) for l in (1, -1)}
        g = np.sign(hueco) * (-1 if filtro.startswith("contra") else 1)
        d, pe, ps, rie = (np.where(g > 0, res[1][j], res[-1][j]) for j in range(4))
        d = d * (np.abs(hueco) > k * a / prev)
        if modo == 1:  # color: la dirección la da la vela; se exige que coincida con el hueco
            pass
    op = d != 0
    r = np.where(op, d * (ps / pe - 1) - 2 * 0.02 / pe, 0.0)
    w = np.where(op, np.minimum(0.005 / (rie / pe), 4), 0)
    fila = dict(n_or=n_or, entrada={0: "stop", 1: "color", 3: "cierre"}[modo], filtro=filtro, stop=sf, objetivo=tR)
    for nom, i0, i1 in P:
        m = (idx >= i0) & (idx <= i1)
        t = r[m & op]
        fila[f"pb_{nom}"] = t.mean() * 1e4
        fila[f"t_{nom}"] = t.mean() / t.std() * len(t) ** .5
    t = r[op]
    fila["pb_todo"] = t.mean() * 1e4
    fila["t_todo"] = t.mean() / t.std() * len(t) ** .5
    fila["periodos+"] = sum(fila[f"pb_{nom}"] > 0 for nom, *_ in P)
    m = metricas(w * r, None)
    fila.update(cagr=m["cagr"], sharpe=m["sharpe"], maxdd=m["maxdd"], ops_año=op.sum() / (D / 252))
    filas.append(fila)
df = pd.DataFrame(filas)
df.to_csv(Path(__file__).parent / "resultados" / "ronda10_spy_familia.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 300)
print("configuraciones:", len(df), "| con t_todo > 2:", (df.t_todo > 2).sum(), "| positivas en los 4 periodos:", (df["periodos+"] == 4).sum())
print(df.groupby(["filtro"])[["pb_todo", "t_todo", "periodos+"]].mean().round(2))
print(df.groupby(["n_or"])[["pb_todo", "t_todo", "periodos+"]].mean().round(2))
print(df.groupby(["entrada"])[["pb_todo", "t_todo", "periodos+"]].mean().round(2))
print(df.sort_values("t_todo", ascending=False).head(25).round(2).to_string())
