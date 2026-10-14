"""Ronda 2: región prometedora de ``combinada.py`` con más salidas y modos de reserva.

Se elige en la ventana de búsqueda (oct-2021 →) y se valida en 2002 → sep-2021. El
Sharpe deflactado usa N = carteras de combinada.py + las de esta ronda.

Uso: python research/caidas/ronda2.py
"""

from __future__ import annotations

import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd

import motor_rapido as mr
from combinada import BASES, puntuar
import eventos as ev

OUT = Path(__file__).parent / "resultados"
VENTANAS = {"busq": ("2021-10-01", None), "valid": ("2002-01-02", "2021-10-01"), "total": ("2002-01-02", None)}


def main() -> None:
    d = puntuar(ev.cargar())
    d = d[d.sobre_sma200 == 1]
    filas, curvas = [], {}
    t0 = time.time()
    for b, rank, smin, K, sal, H, res in itertools.product(
        ("r5≤-7%", "r10≤-10%", "r10≤-15%", "dd20≤-15%"), ("modulos", "limpio", "ridge"), (0, 4, 5, 6, 7),
        (2, 3, 4, 5), ("sma5", "sma10", "rsi2"), (10, 20), ("spy", "spy_tend", None),
    ):
        if (rank == "ridge" and smin) or (rank == "modulos" and smin > 5) or (rank == "limpio" and 0 < smin < 5):
            continue
        x = d[BASES[b](d)]
        col = {"modulos": "score_modulos", "limpio": "score_limpio"}.get(rank)
        if smin:
            x = x[x[col] >= smin]
        score = (x[col] + (-x.r10.fillna(0)).clip(0, 0.99) if col else x.score_ridge)
        fila = {"base": b, "rank": rank, "score_min": smin, "K": K, "salida": sal, "H": H,
                "reserva": res or "efectivo", "eventos": len(x)}
        for w, (ini, fin) in VENTANAS.items():
            r = mr.simular(x.t.to_numpy(), x.j.to_numpy(), score.to_numpy(np.float64), K=K, H=H,
                           salida=sal, reserva=res, inicio=ini, fin=fin)
            fila |= {f"{w}_{k}": v for k, v in r.items() if k != "equity"}
            if w == "total":
                curvas[len(filas)] = r["equity"].pct_change().fillna(0).to_numpy(np.float32)
        filas.append(fila)
    df = pd.DataFrame(filas)
    print(f"{len(df)} carteras en {time.time() - t0:.0f}s")
    df.to_csv(OUT / "ronda2.csv", index=False, float_format="%.5f")
    np.save(ev.SALIDA.with_name("ronda2_retornos.npy"), np.vstack([curvas[i] for i in range(len(df))]))


if __name__ == "__main__":
    main()
