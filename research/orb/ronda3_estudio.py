"""Ronda 3: anatomía del ORB de 5 min en acciones con volumen relativo alto."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from acciones import cartera, informe, panel, rel_vol, señales  # noqa: E402
from motor import metricas  # noqa: E402

p = panel()
f = p["fechas"]
años = pd.DatetimeIndex(f).year.to_numpy()
rv = rel_vol(p, 1)
s = señales(p, n_or=1, modo=0, atr_stop=0.0, stop_frac=1.0)
pd.set_option("display.width", 250, "display.max_columns", 40)

# 1) por decil de RV (todas las acciones-día con operación): R-múltiplo y pb brutos
op = (s["dire"] != 0) & np.isfinite(rv)
R = s["ret"] / s["dist"]
tab = pd.DataFrame(dict(rv=rv[op], pb=s["ret"][op] * 1e4, R=R[op], lado=s["dire"][op],
                        año=np.broadcast_to(años, rv.shape)[op]))
tab["dec"] = pd.qcut(tab.rv, 10, labels=False)
print("== por decil de volumen relativo (bruto, sin costes)")
print(tab.groupby("dec").agg(rv=("rv", "median"), n=("pb", "size"), pb=("pb", "mean"), R=("R", "mean"),
                             acierto=("pb", lambda x: (x > 0).mean())).round(3))
print("\n== RV alto (>3) por lado y año: pb brutos")
alto = tab[tab.rv > 3]
print(alto.pivot_table(index="año", columns="lado", values="pb", aggfunc=["mean", "size"]).round(2))

# 2) cartera base y variantes
base = dict(K=10, rv_min=1.0, coste=1e-4)
filas = []
def corre(nombre, s_=s, rank=rv, **kw):
    a = {**base, **kw}
    res = cartera(p, s_, rank, rv=rv, **a)
    filas.append(informe(res, f, nombre))
    return res

r0 = corre("base K10 coste1")
corre("base K10 coste2", coste=2e-4)
corre("base K10 coste3", coste=3e-4)
corre("K5", K=5); corre("K20", K=20); corre("K40", K=40)
corre("rv>2", rv_min=2.0); corre("rv>3", rv_min=3.0)
corre("riesgo 0.5%", riesgo=0.005); corre("maxlev 2", maxlev=2.0)
for lado, nom in ((1, "solo largos"), (-1, "solo cortos")):
    corre(nom, señales(p, n_or=1, modo=0, lado=lado, atr_stop=0.0, stop_frac=1.0))
for tR in (1, 2, 3, 5, 10):
    corre(f"objetivo {tR}R", señales(p, n_or=1, modo=0, atr_stop=0.0, stop_frac=1.0, target_R=tR))
for c, nom in ((6, "entrar hasta 10:00"), (12, "hasta 10:30"), (24, "hasta 11:30")):
    corre(nom, señales(p, n_or=1, modo=0, atr_stop=0.0, stop_frac=1.0, corte=c))
for rmin, rmax, nom in ((0, 0.15, "rango<0.15 ATR"), (0.15, 0.3, "rango 0.15-0.3 ATR"), (0.3, 99, "rango>0.3 ATR")):
    corre(nom, señales(p, n_or=1, modo=0, atr_stop=0.0, stop_frac=1.0, rmin=rmin, rmax=rmax))
corre("stop 0.75 rango", señales(p, n_or=1, modo=0, atr_stop=0.0, stop_frac=0.75))
corre("stop 1.5 rango", señales(p, n_or=1, modo=0, atr_stop=0.0, stop_frac=1.5))
corre("rango 10 min", señales(p, n_or=2, modo=0, atr_stop=0.0, stop_frac=1.0), rank=rel_vol(p, 2))
df = pd.DataFrame(filas)
df.to_csv(Path(__file__).parent / "resultados" / "ronda3_estudio.csv", index=False)
print(df.round(3).to_string())
