"""Combinación de módulos de mejora en una puntuación y búsqueda de carteras.

Dos puntuaciones, ambas construidas SOLO con la ventana de búsqueda (oct-2021 →):

1. ``score_modulos``: suma de banderas de los módulos que mejoraron la entrada en las
   4 definiciones de caída y en las dos ventanas (``modulos.py``). Es la versión
   explicable: cada punto tiene una razón económica.
2. ``score_ridge``: regresión ridge del retorno de la operación sobre los rasgos
   (winsorizados y en rango 0-1 por fecha), entrenada en la ventana de búsqueda.
   Es la versión "máquina", para ver si combinar muchas señales añade algo.

Luego, rejilla de carteras (motor numba) eligiendo en la ventana de búsqueda y
validando en 2002 → sep-2021 sin tocar nada.

Uso: python research/caidas/combinada.py
"""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import eventos as ev
import motor_rapido as mr

OUT = Path(__file__).parent / "resultados"
CORTE = pd.Timestamp("2021-10-01")
COSTE = 0.002

BANDERAS = {
    "cae con su grupo (rel. pares > -6 %)": lambda d: d.rel_pares10 > -0.06,
    "pares caen ≥7 %": lambda d: d.pares10 <= -0.07,
    "gemela también cae ≥5 %": lambda d: (d.gem_corr >= 0.6) & (d.gem10 <= -0.05),
    "cierra en mínimos (IBS ≤ 0.2)": lambda d: d.ibs <= 0.2,
    "venta general (≥20 % miembros caen)": lambda d: d.frac_caen >= 0.2,
    "momentum 12m ≥ 30 %": lambda d: d.mom252 >= 0.3,
    "beta ≥ 1.3": lambda d: d.beta >= 1.3,
    "sin noticia (no hueco+volumen)": lambda d: ~((d.noche_min10 <= -0.05) & (d.volmax10 >= 3)),
}

def banderas_busqueda() -> dict:
    """Módulos elegidos SOLO con la ventana de búsqueda: el grupo mejora a la base en
    las 4 definiciones de caída de ``modulos.py`` en oct-2021 → hoy. No mira 2002-2021."""
    from modulos import MODULOS
    tablas = [pd.read_csv(OUT / f"modulos_{b}.csv") for b in ("r10_10_tend", "r5_7_tend", "r10_10", "dd20_15")]
    ok = None
    for t in tablas:
        s = t.set_index(["modulo", "grupo"]).busq_mejora > 0
        ok = s if ok is None else ok & s
    return {f"{mo}: {g}": MODULOS[mo][g] for (mo, g), v in ok.items() if v and mo != "BASE"}


RASGOS_RIDGE = ["r1", "r3", "r5", "r10", "r20", "dd20", "dd252", "mom252", "spy5", "spy10", "pares5",
                "pares10", "rel_pares10", "idio10", "gem10", "gem_z", "gem_corr", "pares_corr", "beta",
                "vol_rel", "vol_rel5", "noche1", "noche_min10", "volmax10", "ibs", "volreal_spy",
                "frac_caen", "sobre_sma200", "dias_candidata"]

BASES = {
    "r5≤-5%": lambda d: d.r5 <= -0.05,
    "r5≤-7%": lambda d: d.r5 <= -0.07,
    "r10≤-10%": lambda d: d.r10 <= -0.10,
    "r10≤-15%": lambda d: d.r10 <= -0.15,
    "dd20≤-15%": lambda d: d.dd20 <= -0.15,
}


def puntuar(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    for i, f in enumerate(BANDERAS.values()):
        d[f"b{i}"] = f(d).fillna(False).astype(np.int8)
    d["score_modulos"] = d[[f"b{i}" for i in range(len(BANDERAS))]].sum(axis=1)
    limpias = banderas_busqueda()
    d["score_limpio"] = sum(f(d).fillna(False).astype(np.int8) for f in limpias.values())
    # ridge: rasgos a rango por fecha (robusto a niveles y outliers), entrenado en búsqueda
    X = d[RASGOS_RIDGE].groupby(d.fecha).rank(pct=True).fillna(0.5).to_numpy()
    y = (d.op_sma5.clip(-0.3, 0.3) - d.op_sma5.groupby(d.fecha).transform("mean")).fillna(0).to_numpy()
    tr = (d.fecha >= CORTE).to_numpy()
    Xc = np.c_[np.ones(len(X)), X - 0.5]
    lam = 50.0 * tr.sum() / 1000
    A = Xc[tr].T @ Xc[tr] + lam * np.eye(Xc.shape[1])
    A[0, 0] -= lam
    w = np.linalg.solve(A, Xc[tr].T @ y[tr])
    d["score_ridge"] = Xc @ w
    pesos = pd.Series(w[1:], index=RASGOS_RIDGE).sort_values()
    pesos.to_csv(OUT / "ridge_pesos.csv", float_format="%.6f")
    return d


def por_puntuacion(d: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for b, fb in BASES.items():
        x = d[fb(d)]
        for w, s in (("busq", x.fecha >= CORTE), ("valid", x.fecha < CORTE)):
            g = (x.op_sma5 - COSTE)[s].groupby(x.score_modulos[s].clip(upper=6))
            for sc, v in g.agg(["mean", "size"]).iterrows():
                filas.append({"base": b, "ventana": w, "score": sc, "op": v["mean"], "n": v["size"]})
            q = pd.qcut(x.score_ridge[s].rank(method="first"), 5, labels=False)
            for sc, v in (x.op_sma5 - COSTE)[s].groupby(q).agg(["mean", "size"]).iterrows():
                filas.append({"base": b, "ventana": w, "score": f"ridge_Q{int(sc) + 1}", "op": v["mean"], "n": v["size"]})
    return pd.DataFrame(filas)


def rejilla_carteras(d: pd.DataFrame) -> pd.DataFrame:
    filas = []
    configs = list(itertools.product(
        BASES, ("ninguno", "tendencia"), ("modulos", "ridge", "profundidad"), (0, 3, 4, 5),
        (3, 5, 10), ("sma5", "sma20"), (True, False)))
    t0 = time.time()
    for b, filtro, rank, smin, K, sal, res in configs:
        if rank != "modulos" and smin > 0:
            continue
        x = d[BASES[b](d)]
        if filtro == "tendencia":
            x = x[x.sobre_sma200 == 1]
        if smin:
            x = x[x.score_modulos >= smin]
        score = {"modulos": x.score_modulos + (-x.r10.fillna(0)).clip(0, 0.99),
                 "ridge": x.score_ridge, "profundidad": -x.r10.fillna(0)}[rank].to_numpy(np.float64)
        fila = {"base": b, "filtro": filtro, "rank": rank, "score_min": smin, "K": K, "salida": sal,
                "reserva_spy": res, "eventos": len(x)}
        for w, (ini, fin) in {"busq": ("2021-10-01", None), "valid": ("2002-01-02", "2021-10-01")}.items():
            r = mr.simular(x.t.to_numpy(), x.j.to_numpy(), score, K=K, H=20, salida=sal, reserva=res,
                           inicio=ini, fin=fin)
            fila |= {f"{w}_{k}": v for k, v in r.items() if k != "equity"}
        filas.append(fila)
    print(f"{len(filas)} carteras en {time.time() - t0:.1f}s", flush=True)
    return pd.DataFrame(filas)


def main() -> None:
    d = puntuar(ev.cargar())
    d[["t", "j", "fecha", "score_modulos", "score_ridge"]].to_parquet(ev.SALIDA.with_name("puntuaciones.parquet"))
    pp = por_puntuacion(d)
    pp.to_csv(OUT / "op_por_puntuacion.csv", index=False, float_format="%.5f")
    pd.set_option("display.width", 250)
    print((pp.pivot_table(index="score", columns=["base", "ventana"], values="op") * 100).round(2).to_string())
    print(pd.read_csv(OUT / "ridge_pesos.csv", index_col=0).round(4).T.to_string())
    rc = rejilla_carteras(d)
    rc.to_csv(OUT / "combinada_carteras.csv", index=False, float_format="%.5f")
    spy = {w: mr.simular(np.array([0]), np.array([0]), np.array([0.0]), K=3, reserva=True, inicio=i, fin=f)["cagr"]
           for w, (i, f) in {"busq": ("2021-10-01", None), "valid": ("2002-01-02", "2021-10-01")}.items()}
    (OUT / "combinada_meta.json").write_text(json.dumps({"n_carteras": len(rc), "spy_cagr": spy,
                                                          "banderas": list(BANDERAS)}, ensure_ascii=False, indent=1))
    print("SPY", spy)
    cols = ["base", "filtro", "rank", "score_min", "K", "salida", "reserva_spy", "busq_cagr", "busq_maxdd",
            "valid_cagr", "valid_maxdd", "busq_trades_año"]
    print(rc.sort_values("busq_cagr", ascending=False)[cols].head(25).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
