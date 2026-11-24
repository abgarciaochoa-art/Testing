"""Ronda 2: ORB en ~200 acciones grandes de EE. UU., 5 minutos, nov-2018 → sep-2026."""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from acciones import cartera, informe, panel, rel_vol, señales  # noqa: E402

OUT = Path(__file__).parent / "resultados"

if __name__ == "__main__":
    p = panel()
    f = p["fechas"]
    print("panel", p["O"].shape, f[0], f[-1])
    filas = []
    # 1) réplica Zarattini: rango de 5 min, color + ruptura, stop 10 % ATR, hasta el cierre, top 20 por RV
    for n_or in (1, 3, 6):  # 5, 15, 30 minutos
        rv = rel_vol(p, n_or)
        for modo, stop in itertools.product((2, 0), ("atr0.05", "atr0.1", "atr0.2", "rango1", "rango0.5")):
            kw = dict(n_or=n_or, modo=modo)
            if stop.startswith("atr"):
                kw["atr_stop"] = float(stop[3:])
            else:
                kw["atr_stop"] = 0.0
                kw["stop_frac"] = float(stop[5:])
            s = señales(p, **kw)
            for K, rvmin, rk in ((20, 1.0, "rv"), (10, 1.0, "rv"), (5, 1.0, "rv"), (20, 0.0, "azar")):
                rank = rv if rk == "rv" else np.random.default_rng(0).random(rv.shape) * np.isfinite(rv)
                for coste in (1e-4, 2e-4):
                    res = cartera(p, s, rank, K=K, rv=rv, rv_min=rvmin, coste=coste)
                    filas.append(dict(n_or=n_or * 5, modo={2: "color+ruptura", 0: "ruptura"}[modo],
                                      stop=stop, K=K, rank=rk, coste_pb=coste * 1e4,
                                      **informe(res, f)))
            print(filas[-2]["n_or"], filas[-2]["modo"], stop, "ok", flush=True)
    df = pd.DataFrame(filas)
    df.to_csv(OUT / "ronda2_acciones.csv", index=False)
    pd.set_option("display.width", 250, "display.max_rows", 400, "display.max_columns", 40)
    cols = ["n_or", "modo", "stop", "K", "rank", "coste_pb", "cagr", "sharpe", "maxdd", "ops_dia", "lev_medio",
            "pb_op", "acierto"] + [c for c in df.columns if c.isdigit()]
    print(df[df.coste_pb == 1][cols].sort_values("sharpe", ascending=False).to_string(
        float_format="{:.3f}".format))
