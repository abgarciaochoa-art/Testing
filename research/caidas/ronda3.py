"""Ronda 3: carteras con la meta-puntuación (módulos v1+v2), tamaño por convicción y
límite por grupo de pares (motor v2, numba).

Prioridad de las candidatas:
- ``v2``: nº de módulos a favor (desempate: caída más profunda);
- ``v2meta``: nº de módulos (desempate: modelo, en rango del día);
- ``meta``: solo el modelo.

Tamaño por convicción (pre-especificado): peso = clip((módulos - 4) / 2, 0.5, 2):
5 módulos → 0,5 huecos, 6 → 1, 7 → 1,5, ≥8 → 2.

Se elige en la búsqueda (oct-2021 →) y se valida en 2002 → sep-2021.

Uso: python research/caidas/ronda3.py
"""

from __future__ import annotations

import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd

import eventos as ev
import motor_cartera as mc

OUT = Path(__file__).parent / "resultados"
VENTANAS = {"busq": ("2021-10-01", None), "valid": ("2002-01-02", "2021-10-01"), "total": ("2002-01-02", None)}
BASES = {
    "r5≤-7%": lambda d: d.r5 <= -0.07,
    "r10≤-10%": lambda d: d.r10 <= -0.10,
    "r10≤-15%": lambda d: d.r10 <= -0.15,
    "dd20≤-15%": lambda d: d.dd20 <= -0.15,
}


def cargar() -> pd.DataFrame:
    e = ev.cargar()[["t", "j", "r5", "r10", "dd20"]]
    s = pd.read_parquet(ev.SALIDA.with_name("meta.parquet"))
    d = s.merge(e, on=["t", "j"], how="left")
    d["meta_pct"] = d.groupby("t").score_meta.rank(pct=True)
    return d


def main() -> None:
    d = cargar()
    filas, curvas = [], []
    t0 = time.time()
    prep = []
    for b, rank, smin in itertools.product(BASES, ("v2", "v2meta", "meta"), (0, 5, 6, 7)):
        if rank == "meta" and smin:
            continue
        x = d[BASES[b](d).fillna(False)]
        if smin:
            x = x[x.score_v2 >= smin]
        score = {"v2": x.score_v2 + (-x.r10.fillna(0)).clip(0, 0.99),
                 "v2meta": x.score_v2 + x.meta_pct.fillna(0.5) * 0.99,
                 "meta": x.score_meta}[rank].to_numpy(np.float64)
        w_conv = np.clip((x.score_v2.to_numpy() - 4) / 2, 0.5, 2.0)
        prep.append((b, rank, smin, x.t.to_numpy(), x.j.to_numpy(), score, w_conv))
    for (b, rank, smin, t, j, score, w_conv), K, peso, grupo, sal, H, res in itertools.product(
        prep, (3, 5, 8), ("igual", "conviccion"), (0, 1), ("sma5", "sma10", "rsi2"), (10, 20), ("spy", "spy_tend"),
    ):
        w = w_conv if peso == "conviccion" else None
        fila = {"base": b, "rank": rank, "score_min": smin, "K": K, "peso": peso, "max_grupo": grupo,
                "salida": sal, "H": H, "reserva": res, "eventos": len(t)}
        for nombre, (ini, fin) in VENTANAS.items():
            r = mc.simular(t, j, score, peso=w, K=K, H=H, salida=sal, reserva=res, max_grupo=grupo,
                           inicio=ini, fin=fin)
            fila |= {f"{nombre}_{k}": v for k, v in r.items() if k != "equity"}
            if nombre == "total":
                curvas.append(r["equity"].pct_change().fillna(0).to_numpy(np.float32))
        filas.append(fila)
        if len(filas) % 500 == 0:
            print(f"{len(filas)} carteras  {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(filas)
    print(f"{len(df)} carteras en {time.time() - t0:.0f}s")
    df.to_csv(OUT / "ronda3.csv", index=False, float_format="%.5f")
    np.save(ev.SALIDA.with_name("ronda3_retornos.npy"), np.vstack(curvas))


if __name__ == "__main__":
    main()
