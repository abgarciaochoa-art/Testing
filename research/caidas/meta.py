"""Meta-puntuación de cada caída candidata: módulos v1+v2 y modelos, eligiendo SOLO
con la ventana de búsqueda (oct-2021 →).

1. ``score_v2`` (explicable): número de módulos a favor. Un grupo de ``modulos.py`` o
   ``modulos_v2.py`` entra si mejora el retorno de la base en las 4 definiciones de
   caída **y en las dos mitades** de la búsqueda (oct-21→mar-24 y abr-24→oct-26).
   La segunda condición filtra efectos de un único episodio.
2. ``score_meta`` (máquina): ridge o gradient boosting sobre los rasgos en rango por
   fecha, con objetivo = retorno de la operación menos la media de ese día. Se elige
   el modelo por validación cruzada DENTRO de la búsqueda (entrenar en una mitad,
   evaluar en la otra, y al revés) y se reentrena con toda la búsqueda.

La validación 2002-2021 no interviene en ninguna decisión: solo se informa.

Uso: python research/caidas/meta.py  -> data/cache/caidas/meta.parquet
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import eventos as ev
from modulos import BASES, MODULOS, resumen
from modulos_v2 import MODULOS_V2

OUT = Path(__file__).parent / "resultados"
CORTE = pd.Timestamp("2021-10-01")
MITAD = pd.Timestamp("2024-04-01")
COSTE = 0.002

RASGOS = ["r1", "r3", "r5", "r10", "r20", "dd20", "dd252", "mom252", "spy5", "spy10", "pares5", "pares10",
          "rel_pares10", "idio10", "gem10", "gem_z", "gem_corr", "pares_corr", "beta", "vol_rel", "vol_rel5",
          "noche1", "noche_min10", "volmax10", "ibs", "volreal_spy", "frac_caen", "sobre_sma200",
          "dias_candidata", "sigma5", "sigma10", "vol60", "dist_sma200", "dist_sma50", "pares_mom252",
          "rel_mom252", "pares_frac_caen", "tam", "spy_dd252", "vol_pico_spy", "sesiones_fin_mes"]


def seleccionar_banderas(d: pd.DataFrame) -> dict:
    todos = {**{f"{m}: {g}": f for m, gs in MODULOS.items() for g, f in gs.items()},
             **{f"{m}: {g}": f for m, gs in MODULOS_V2.items() for g, f in gs.items()}}
    elegidas = {}
    for nombre, f in todos.items():
        ok = True
        for fb in BASES.values():
            x = d[fb(d).fillna(False)]
            sel = f(x).fillna(False).to_numpy(bool)
            for a, b in ((CORTE, MITAD), (MITAD, None)):
                w = (x.fecha >= a).to_numpy() & ((x.fecha < b).to_numpy() if b is not None else True)
                base, grupo = resumen(x[w]), resumen(x[w & sel])
                if not (grupo["n"] >= 30 and grupo["op"] > base["op"]):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            elegidas[nombre] = f
    return elegidas


def matriz(d: pd.DataFrame) -> np.ndarray:
    return d[RASGOS].groupby(d.fecha).rank(pct=True).fillna(0.5).to_numpy(np.float32)


def objetivo(d: pd.DataFrame) -> np.ndarray:
    y = d.op_sma5.clip(-0.3, 0.3)
    return (y - y.groupby(d.fecha).transform("mean")).fillna(0).to_numpy()


def ajustar(tipo: str, X: np.ndarray, y: np.ndarray):
    if tipo == "ridge":
        Xc = np.c_[np.ones(len(X)), X - 0.5]
        lam = 50.0 * len(X) / 1000
        A = Xc.T @ Xc + lam * np.eye(Xc.shape[1])
        A[0, 0] -= lam
        w = np.linalg.solve(A, Xc.T @ y)
        return lambda Z: np.c_[np.ones(len(Z)), Z - 0.5] @ w
    mod = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.03, max_iter=300, min_samples_leaf=400,
                                        l2_regularization=1.0, random_state=0)
    mod.fit(X, y)
    return mod.predict


def calidad(pred: np.ndarray, d: pd.DataFrame) -> dict:
    """Retorno neto medio del quintil superior por fecha y diferencia Q5-Q1."""
    q = pd.Series(pred, index=d.index).groupby(d.fecha).rank(pct=True)
    op = d.op_sma5 - COSTE
    return {"Q5": float(op[q > 0.8].mean()), "Q1": float(op[q <= 0.2].mean()),
            "Q5-Q1": float(op[q > 0.8].mean() - op[q <= 0.2].mean())}


FIN_ENTRENO = pd.Timestamp("2026-10-13")  # ventana de búsqueda/entreno CONGELADA: oct-2021 → este día


def main() -> None:
    import sys
    d = ev.cargar()
    d = d[d.sobre_sma200 == 1].copy()  # todas las carteras finalistas usan el filtro de tendencia
    if "--congelado" in sys.argv:  # uso en vivo: mismas banderas que en la investigación
        nombres = json.loads((OUT / "meta_banderas.json").read_text())["banderas"]
        todos = {**{f"{m}: {g}": f for m, gs in MODULOS.items() for g, f in gs.items()},
                 **{f"{m}: {g}": f for m, gs in MODULOS_V2.items() for g, f in gs.items()}}
        banderas = {k: todos[k] for k in nombres}
    else:
        banderas = seleccionar_banderas(d[d.fecha < FIN_ENTRENO])
    print(f"{len(banderas)} banderas elegidas solo con la búsqueda:")
    for k in banderas:
        print("  -", k)
    d["score_v2"] = sum(f(d).fillna(False).astype(np.int8) for f in banderas.values())

    X, y = matriz(d), objetivo(d)
    busq = ((d.fecha >= CORTE) & (d.fecha < FIN_ENTRENO)).to_numpy()
    h1 = busq & (d.fecha < MITAD).to_numpy()
    h2 = busq & (d.fecha >= MITAD).to_numpy()
    cv = {}
    for tipo in ("ridge", "gbm"):
        r = []
        for tr, te in ((h1, h2), (h2, h1)):
            f = ajustar(tipo, X[tr], y[tr])
            r.append(calidad(f(X[te]), d[te]))
        cv[tipo] = {k: float(np.mean([x[k] for x in r])) for k in r[0]}
    mejor = max(cv, key=lambda k: cv[k]["Q5-Q1"])
    print("CV dentro de la búsqueda:", json.dumps(cv, indent=1), "-> elegido:", mejor)
    f = ajustar(mejor, X[busq], y[busq])
    d["score_meta"] = f(X)
    otro = "gbm" if mejor == "ridge" else "ridge"
    d[f"score_{otro}"] = ajustar(otro, X[busq], y[busq])(X)
    d[f"score_{mejor}"] = d["score_meta"]

    # informe (la validación solo se mira, no decide)
    filas = []
    for b, fb in BASES.items():
        x = d[fb(d).fillna(False)]
        for w, s in (("busq", x.fecha >= CORTE), ("valid", x.fecha < CORTE)):
            for col in ("score_v2", "score_ridge", "score_gbm"):
                if col == "score_v2":
                    g = (x.op_sma5 - COSTE)[s].groupby(x[col][s].clip(upper=10)).mean()
                    filas += [{"base": b, "ventana": w, "score": col, "grupo": int(k), "op": v} for k, v in g.items()]
                else:
                    qq = calidad(x[col][s].to_numpy(), x[s])
                    filas += [{"base": b, "ventana": w, "score": col, "grupo": k, "op": v} for k, v in qq.items()]
    inf = pd.DataFrame(filas)
    inf.to_csv(OUT / "meta_informe.csv", index=False, float_format="%.5f")
    pd.set_option("display.width", 250)
    print((inf.pivot_table(index=["score", "grupo"], columns=["base", "ventana"], values="op") * 100).round(2).to_string())
    d[["t", "j", "fecha", "score_v2", "score_meta", "score_ridge", "score_gbm"]].to_parquet(ev.SALIDA.with_name("meta.parquet"))
    if "--congelado" not in sys.argv:
        (OUT / "meta_banderas.json").write_text(json.dumps({"banderas": list(banderas), "cv": cv, "modelo": mejor},
                                                           ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
