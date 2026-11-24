"""Ronda 9: validación fuera de muestra de la regla Gap ORB en SPY 1 min (2000 → oct-2025).

La regla y sus parámetros salieron de los CFD de 2005-2020. Periodos nuevos: 2000-2004 y
jun-2020 → oct-2025. Costes de SPY por lado: 0,02 $ por acción (medio céntimo de horquilla,
1 céntimo de deslizamiento de la orden stop, 0,0035 $ de comisión, redondeado) — en pb:
~1,5 pb en 2003 (SPY a 100 $), ~0,3 pb en 2025 (SPY a 650 $).
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import datos  # noqa: E402
from motor import atr_diario, metricas, orb  # noqa: E402

OUT = Path(__file__).parent / "resultados"


def gap_orb(x, n_or=30, k=0.2, modo=0, coste_usd=0.02, coste_bp=0.0, target_R=0.0, corte=None):
    D = len(x["fechas"])
    a = atr_diario(x["O"], x["H"], x["L"], x["C"], x["ultimo"])
    prev = x["prev"]
    hueco = x["O"][:, 0] / prev - 1
    corte = corte or x["O"].shape[1] - 1
    res = {l: orb(x["O"], x["H"], x["L"], x["C"], x["ultimo"].astype(np.int64), n_or, modo, l, 1.0, a, 0.0,
                  target_R, corte, 0.0, 1e12) for l in (1, -1)}
    g = np.sign(hueco)
    usar = np.abs(hueco) > k * a / prev
    pick = lambda j: np.where(g > 0, res[1][j], res[-1][j])  # noqa: E731
    d = pick(0) * usar
    pe, ps, rie, mot = pick(1), pick(2), pick(3), pick(5)
    op = d != 0
    coste = coste_usd / pe + coste_bp * 1e-4
    out = pd.DataFrame(index=pd.DatetimeIndex(x["fechas"]))
    out["bruto"] = np.where(op, d * (ps / pe - 1), 0.0)
    out["r1"] = np.where(op, out["bruto"] - 2 * coste, 0.0)
    out["dist"] = np.where(op, rie / pe, np.nan)
    out["op"] = op
    out["dir"] = d
    out["motivo"] = np.where(op, mot, 0)
    return out


def con_riesgo(o, riesgo=0.005, maxlev=4.0):
    w = np.minimum(riesgo / o["dist"], maxlev).fillna(0)
    return w * o["r1"], w


def tramo(o, ini, fin, riesgo=0.005):
    m = (o.index >= ini) & (o.index <= fin)
    x = o[m]
    t = x.r1[x.op]
    rd, w = con_riesgo(x, riesgo)
    mm = metricas(rd.to_numpy(), None)
    return dict(n=int(x.op.sum()), ops_año=x.op.sum() / (len(x) / 252), pb_bruto=x.bruto[x.op].mean() * 1e4,
                pb_neto=t.mean() * 1e4, t=t.mean() / t.std() * len(t) ** .5, acierto=(t > 0).mean(),
                cagr=mm["cagr"], sharpe=mm["sharpe"], maxdd=mm["maxdd"], palanca=w[x.op].mean())


if __name__ == "__main__":
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 300)
    x = datos.indice("SPY")
    tramos = [("2000-2004 (nuevo)", "2000-01-01", "2004-12-31"), ("2005-2012 (CFD visto)", "2005-01-01", "2012-12-31"),
              ("2013-may 2020 (CFD visto)", "2013-01-01", "2020-05-13"), ("jun 2020-oct 2025 (nuevo)", "2020-05-14", "2025-12-31"),
              ("todo 2000-2025", "2000-01-01", "2025-12-31")]
    base = gap_orb(x)
    print("== regla base en SPY (rango 30 min, hueco > 0,2 ATR, orden stop, stop al otro lado, al cierre), riesgo 0,5 %")
    print(pd.DataFrame({n: tramo(base, a, b) for n, a, b in tramos}).T.round(3).to_string())
    rd, _ = con_riesgo(base)
    print("\nPor año (%):")
    print((rd.groupby(rd.index.year).apply(lambda r: (1 + r).prod() - 1) * 100).round(1).to_frame().T.to_string())
    print("\n== vecindad en los periodos nuevos (t; pb netos)")
    filas = []
    for n_or, k, modo in itertools.product((15, 30, 60), (0.1, 0.2, 0.3), (0, 3)):
        o = gap_orb(x, n_or, k, modo)
        r_new = tramo(o, "2020-05-14", "2025-12-31")
        r_old = tramo(o, "2000-01-01", "2004-12-31")
        filas.append(dict(n_or=n_or, k=k, entrada="stop" if modo == 0 else "cierre", t_2000_04=r_old["t"],
                          pb_2000_04=r_old["pb_neto"], t_2020_25=r_new["t"], pb_2020_25=r_new["pb_neto"],
                          cagr_2020_25=r_new["cagr"], sh_2020_25=r_new["sharpe"]))
    print(pd.DataFrame(filas).round(2).to_string())
    base.to_csv(OUT / "gap_orb_spy_operaciones.csv")
