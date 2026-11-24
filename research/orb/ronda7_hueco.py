"""Ronda 7: 'Gap ORB' en índices: solo la ruptura del rango en la dirección del hueco de la noche,
cuando el hueco es grande respecto al ATR. Entrada al cierre de vela de 1 min fuera del rango
(siguiente apertura) o con orden stop. Stop al otro lado del rango, salida al cierre u objetivo.
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import datos  # noqa: E402
from motor import atr_diario, metricas, orb  # noqa: E402

IDX = list(datos.INDICES)


def operaciones(nombre, n_or, modo, k_hueco, target_R=0.0, stop_frac=1.0, corte_min=None, coste=1e-4):
    x = datos.indice(nombre)
    D = len(x["fechas"])
    a = atr_diario(x["O"], x["H"], x["L"], x["C"], x["ultimo"])
    prev = x["prev"]
    hueco = x["O"][:, 0] / prev - 1
    corte = corte_min if corte_min else x["O"].shape[1] - 1
    res = {}
    for lado in (1, -1):
        res[lado] = orb(x["O"], x["H"], x["L"], x["C"], x["ultimo"].astype(np.int64), n_or, modo, lado,
                        stop_frac, a, 0.0, target_R, corte, 0.0, 1e12)
    g = np.sign(hueco)
    usar = np.abs(hueco) > k_hueco * a / prev
    out = pd.DataFrame(index=pd.DatetimeIndex(x["fechas"]))
    d = np.where(g > 0, res[1][0], res[-1][0]) * usar
    pe = np.where(g > 0, res[1][1], res[-1][1])
    ps = np.where(g > 0, res[1][2], res[-1][2])
    rie = np.where(g > 0, res[1][3], res[-1][3])
    op = d != 0
    out["r1"] = np.where(op, d * (ps / pe - 1) - 2 * coste, 0.0)
    out["dist"] = np.where(op, rie / pe, np.nan)
    out["op"] = op
    return out


def cartera(ops: dict, riesgo=0.005, maxlev_pos=3.0):
    """Suma de posiciones; cada índice arriesga `riesgo` por operación (tamaño = riesgo/dist, máx maxlev_pos)."""
    r = None
    for n, o in ops.items():
        w = np.minimum(riesgo / o["dist"], maxlev_pos).fillna(0)
        x = (w * o["r1"]).rename(n)
        r = x.to_frame() if r is None else r.join(x, how="outer")
    return r.fillna(0)


def resumen(rd: pd.Series):
    m = metricas(rd.to_numpy(), rd.index)
    a = rd.index.year
    m1 = metricas(rd[a <= 2012].to_numpy(), None)
    m2 = metricas(rd[a > 2012].to_numpy(), None)
    return dict(cagr=m["cagr"], sharpe=m["sharpe"], maxdd=m["maxdd"], cagr_05_12=m1["cagr"], sh_05_12=m1["sharpe"],
                cagr_13_20=m2["cagr"], sh_13_20=m2["sharpe"])


if __name__ == "__main__":
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 300)
    filas = []
    for n_or, modo, k, tR in itertools.product((5, 15, 30, 60), (3, 0), (0.0, 0.1, 0.2, 0.3, 0.5), (0.0, 1.0, 2.0)):
        ops = {n: operaciones(n, n_or, modo, k, tR) for n in IDX}
        fila = dict(n_or=n_or, entrada="cierre" if modo == 3 else "stop", k_hueco=k, objetivo=tR)
        for n, o in ops.items():
            t = o.r1[o.op]
            fila[f"pb_{n}"] = t.mean() * 1e4
        tod = pd.concat([o.r1[o.op] for o in ops.values()])
        fila["pb_todos"] = tod.mean() * 1e4
        fila["t_todos"] = tod.mean() / tod.std() * np.sqrt(len(tod))
        fila["ops_año_idx"] = np.mean([o.op.sum() / (len(o) / 252) for o in ops.values()])
        rd = cartera(ops).sum(1)
        fila.update(resumen(rd))
        filas.append(fila)
    df = pd.DataFrame(filas)
    df.to_csv(Path(__file__).parent / "resultados" / "ronda7_hueco.csv", index=False)
    print(df.sort_values("t_todos", ascending=False).round(2).to_string())
