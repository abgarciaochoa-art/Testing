"""Ronda 4: robustez alrededor de 'rango 5 min, ruptura, stop al otro lado, objetivo ~1R, top-K por RV'."""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from acciones import cartera, informe, panel, rel_vol, señales  # noqa: E402

p = panel()
f = p["fechas"]
rv1 = rel_vol(p, 1)
filas = []
cache = {}
for n_or, tR, sf in itertools.product((1, 2, 3), (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 0), (0.75, 1.0, 1.25)):
    s = señales(p, n_or=n_or, modo=0, atr_stop=0.0, stop_frac=sf, target_R=tR)
    rv = rel_vol(p, n_or)
    for K, coste in itertools.product((5, 10, 20), (0.5e-4, 1e-4, 1.5e-4, 2e-4)):
        res = cartera(p, s, rv, K=K, rv=rv, rv_min=1.0, coste=coste, riesgo=0.005, maxlev=4)
        inf = informe(res, f)
        a = pd.DatetimeIndex(f).year
        inf["sh_19_22"] = res["r"][(a >= 2019) & (a <= 2022)].mean() / res["r"][(a >= 2019) & (a <= 2022)].std() * 252 ** .5
        inf["sh_23_26"] = res["r"][a >= 2023].mean() / res["r"][a >= 2023].std() * 252 ** .5
        filas.append(dict(n_or=n_or * 5, objetivo=tR, stop=sf, K=K, coste_pb=coste * 1e4, **inf))
    print(n_or, tR, sf, flush=True)
df = pd.DataFrame(filas).drop(columns="nombre")
df.to_csv(Path(__file__).parent / "resultados" / "ronda4_robustez.csv", index=False)
