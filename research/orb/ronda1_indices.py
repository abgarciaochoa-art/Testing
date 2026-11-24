"""Ronda 1: rejilla fija de ORB en 7 índices (1 minuto). Sin optimizar: se mira la familia entera.

Costes: 1 pb por lado (futuros/CFD con stops: horquilla + deslizamiento).
Tamaño: (a) 1× nocional; (b) riesgo fijo del 1 % del capital por operación (tamaño = 1 % /
distancia al stop, máximo 4×), como en Zarattini, Barbon y Aziz.
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import datos  # noqa: E402
from motor import atr_diario, metricas, orb  # noqa: E402

COSTE = 1e-4  # por lado
OUT = Path(__file__).parent / "resultados"


def evaluar(x, a, n_or, modo, lado, stop_frac, target_R, corte, coste=COSTE, riesgo=0.01, maxlev=4.0,
            atr_stop=0.0, rmin=0.0, rmax=1e12):
    d, pe, ps, rie, ve, mot = orb(x["O"], x["H"], x["L"], x["C"], x["ultimo"].astype(np.int64), n_or,
                                  modo, lado, stop_frac, a, atr_stop, target_R, corte, rmin, rmax)
    op = d != 0
    r1 = np.zeros(len(d))
    r1[op] = d[op] * (ps[op] / pe[op] - 1) - 2 * coste
    lev = np.zeros(len(d))
    lev[op] = np.minimum(riesgo / (rie[op] / pe[op]), maxlev)
    rr = r1 * lev
    return dict(d=d, r1=r1, rr=rr, lev=lev, mot=mot, op=op)


def resumen(res, fechas):
    op = res["op"]
    años = pd.DatetimeIndex(fechas).year.to_numpy()
    t = res["r1"][op]
    m1 = metricas(res["r1"], fechas)
    mr = metricas(res["rr"], fechas)
    mitad = años <= 2012 if años[0] < 2010 else años <= 2014
    m_a = metricas(res["rr"][mitad], fechas[mitad])
    m_b = metricas(res["rr"][~mitad], fechas[~mitad])
    return dict(ops_año=op.sum() / (len(op) / 252), acierto=(t > 0).mean(), pb_op=t.mean() * 1e4,
                t=t.mean() / t.std() * np.sqrt(len(t)), cagr_1x=m1["cagr"], sh_1x=m1["sharpe"],
                dd_1x=m1["maxdd"], cagr_r1=mr["cagr"], sh_r1=mr["sharpe"], dd_r1=mr["maxdd"],
                lev_medio=res["lev"][op].mean(), cagr_mitad1=m_a["cagr"], cagr_mitad2=m_b["cagr"])


if __name__ == "__main__":
    filas = []
    for nombre in datos.INDICES:
        x = datos.indice(nombre)
        a = atr_diario(x["O"], x["H"], x["L"], x["C"], x["ultimo"])
        for n_or, modo, lado, sf, tR in itertools.product([5, 15, 30, 60], [0, 1], [0, 1], [1.0, 0.5],
                                                          [0, 2, 10]):
            corte = 150 if nombre in ("SPX", "NDX", "RUT") else 180  # hasta 12:00 NY / 3 h tras abrir
            res = evaluar(x, a, n_or, modo, lado, sf, tR, corte)
            filas.append(dict(indice=nombre, n_or=n_or, modo="color" if modo else "ruptura",
                              lado="ambos" if lado == 0 else "largo", stop=sf, objetivo_R=tR,
                              **resumen(res, x["fechas"])))
    df = pd.DataFrame(filas)
    df.to_csv(OUT / "ronda1_indices.csv", index=False)
    pd.set_option("display.width", 250, "display.max_rows", 400)
    f = {c: "{:.1%}".format for c in ["acierto", "cagr_1x", "dd_1x", "cagr_r1", "dd_r1", "cagr_mitad1",
                                       "cagr_mitad2"]}
    print("== media de la familia por índice y longitud del rango (pb por operación neto, CAGR riesgo 1%)")
    print(df.groupby(["indice", "n_or"])[["pb_op", "t", "cagr_r1", "sh_r1"]].mean().unstack().round(2))
    print("\n== por modo/lado")
    print(df.groupby(["indice", "modo", "lado"])[["pb_op", "t", "sh_r1"]].mean().unstack().round(2))
    print("\n== mejores 25 por Sharpe (riesgo 1%)")
    print(df.sort_values("sh_r1", ascending=False).head(25).to_string(formatters=f, float_format="{:.2f}".format))
