"""Sharpe deflactado con el número EFECTIVO de pruebas independientes.

Muchas de las ~10.700 carteras probadas son variaciones mínimas unas de otras (misma
señal, otra salida...). Siguiendo la regla del repo (``n_effective_trials``: unir
pruebas con correlación > 0,9 por cierre transitivo), pero vectorizado para que
escale: correlación con matrices float32 y componentes conexas con scipy.

Uso: python research/caidas/dsr_efectivo.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

import eventos as ev

OUT = Path(__file__).parent / "resultados"
CACHE = ev.SALIDA.parent
FICHEROS = ("rejilla_retornos.npy", "rejilla_retornos_fase2.npy", "ronda2_retornos.npy", "ronda3_retornos.npy",
            "ronda4_retornos.npy")


def main(umbral: float = 0.9) -> dict:
    mats = [np.load(CACHE / f)[:, 1:] for f in FICHEROS]
    T = min(x.shape[1] for x in mats)
    R = np.vstack([x[:, -T:] for x in mats]).astype(np.float32)
    R = R[R.std(1) > 0]
    Z = (R - R.mean(1, keepdims=True)) / R.std(1, keepdims=True)
    filas = []
    n = len(Z)
    for a in range(0, n, 1000):  # bloques para no materializar n×n densa de golpe en float64
        c = Z[a : a + 1000] @ Z.T / T
        i, j = np.nonzero(c > umbral)
        filas.append((i + a, j))
    ii = np.concatenate([f[0] for f in filas])
    jj = np.concatenate([f[1] for f in filas])
    adj = csr_matrix((np.ones(len(ii), dtype=np.int8), (ii, jj)), shape=(n, n))
    n_eff, etiqueta = connected_components(adj, directed=False)
    sh = R.mean(1) / R.std(1)
    sh_cluster = np.array([sh[etiqueta == k].mean() for k in range(n_eff)])
    res = {"n_bruto": int(n), "n_efectivo": int(n_eff), "umbral_corr": umbral,
           "var_sharpe_bruto": float(np.var(sh, ddof=1)), "var_sharpe_clusters": float(np.var(sh_cluster, ddof=1))}
    (OUT / "dsr_efectivo.json").write_text(json.dumps(res, indent=1))
    print(res)
    return res


if __name__ == "__main__":
    main()
