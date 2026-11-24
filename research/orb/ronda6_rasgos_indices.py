"""Ronda 6: ¿qué rasgos separan las rupturas buenas de las malas en índices?

Operación base: rango de n minutos, entrada al cierre de la vela de 1 min fuera del rango (siguiente
apertura; sin ambigüedad), stop al otro lado del rango, hasta el cierre. Resultado bruto en pb y R.
Rasgos conocidos antes de entrar: ancho del rango / ATR, hueco de la noche (a favor/en contra),
tendencia (cierre de ayer sobre la media de 20 días, a favor/en contra), régimen de volatilidad (ATR
de 14 días / su media de 250), hora de la ruptura, día de la semana, rentabilidad de ayer.
Mitades: 2005-2012 y 2013-2020 (DAX/STOXX: 2010-2014 y 2015-2018).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import datos  # noqa: E402
from motor import atr_diario, orb  # noqa: E402

filas = []
for nombre in datos.INDICES:
    x = datos.indice(nombre)
    D = len(x["fechas"])
    a = atr_diario(x["O"], x["H"], x["L"], x["C"], x["ultimo"])
    cl = x["C"][np.arange(D), x["ultimo"]]
    prev = np.r_[np.nan, cl[:-1]]
    ma20 = pd.Series(cl).rolling(20).mean().shift(1).to_numpy()
    tend = np.sign(prev - ma20)
    regimen = a / pd.Series(a).rolling(250, min_periods=100).mean().to_numpy()
    ayer = np.r_[np.nan, np.nan, cl[1:-1] / cl[:-2] - 1]
    hueco = x["O"][:, 0] / prev - 1
    años = pd.DatetimeIndex(x["fechas"]).year.to_numpy()
    corte_mitad = 2012 if años[0] < 2008 else 2014
    for n_or in (5, 15, 30, 60):
        for modo in (3, 0):
            d, pe, ps, rie, ve, mot = orb(x["O"], x["H"], x["L"], x["C"], x["ultimo"].astype(np.int64), n_or, modo,
                                          0, 1.0, a, 0.0, 0.0, 300, 0.0, 1e12)
            ok = d != 0
            for i in np.where(ok)[0]:
                R = x["H"][i, :n_or].max() - x["L"][i, :n_or].min()
                filas.append(dict(indice=nombre, n_or=n_or, entrada="cierre" if modo == 3 else "stop",
                                  pb=d[i] * (ps[i] / pe[i] - 1) * 1e4, Rmult=d[i] * (ps[i] - pe[i]) / rie[i],
                                  r_atr=R / a[i], hueco_favor=np.sign(hueco[i]) * d[i], abs_hueco=abs(hueco[i]) / (a[i] / prev[i]),
                                  tend_favor=tend[i] * d[i], regimen=regimen[i], hora=ve[i], dow=pd.Timestamp(x["fechas"][i]).dayofweek,
                                  ayer_favor=np.sign(ayer[i]) * d[i], lado=d[i], mitad=1 if años[i] <= corte_mitad else 2))
df = pd.DataFrame(filas)
# (operaciones no se guardan: 4 MB)
pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 300)


def tabla(col, bins=None):
    t = df.copy()
    if bins is not None:
        t[col] = pd.qcut(t.groupby(["indice", "n_or", "entrada"])[col].rank(pct=True), bins, labels=False)
    g = t.groupby(["entrada", "n_or", col, "mitad"]).pb.mean().unstack("mitad").round(2)
    # nº de índices con media positiva en cada mitad
    pos = t.groupby(["entrada", "n_or", col, "mitad", "indice"]).pb.mean().gt(0).groupby(
        ["entrada", "n_or", col, "mitad"]).sum().unstack("mitad")
    g.columns = [f"pb_m{c}" for c in g.columns]
    pos.columns = [f"idx+_m{c}" for c in pos.columns]
    return pd.concat([g, pos], axis=1)


print("== base: pb brutos por índice, rango, entrada y mitad")
print(df.groupby(["entrada", "n_or", "indice", "mitad"]).pb.mean().unstack(["mitad"]).unstack("indice").round(1))
for col, bins in (("r_atr", 5), ("hueco_favor", None), ("tend_favor", None), ("regimen", 4), ("ayer_favor", None),
                  ("lado", None), ("abs_hueco", 4), ("dow", None)):
    print(f"\n== {col}")
    print(tabla(col, bins))
