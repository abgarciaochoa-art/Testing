"""Robustez del semáforo: rejilla de umbrales (sigma × rvol) en las 80 acciones con velas de una
hora (2024-2026). Entrada la primera vez que se cumple (10:30-14:30), stop 0,25 ATR, candado.
Si la ventaja solo apareciera en una casilla, sería azar; si la superficie es suave, no.

Uso: python research/tendencia_intradia/robustez.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent / "bracket_intradia"))
import estudio as B  # noqa: E402
import motor  # noqa: E402
import paneles as T  # noqa: E402
import prueba_semaforo as PS  # noqa: E402
import semaforo as S  # noqa: E402

SIGMAS = [0.5, 1.0, 1.5, 2.0, 3.0]
RVOLS = [0.0, 1.0, 1.5, 2.0, 3.0]


def rejilla(pre: str = "h1", f: float = 0.25, mercado: bool = False) -> pd.DataFrame:
    spy = T.cargar(f"{pre}_SPY")
    datos = []
    for tk in PS.acciones():
        fp = T.PANELES / f"{pre}_{tk}.npz"
        if fp.exists():
            P = T.cargar(f"{pre}_{tk}")
            datos.append((P, S.matrices(P, spy)))
    c, sl = B.COSTES["acciones"]
    filas = []
    for sg_min in SIGMAS:
        for rv_min in RVOLS:
            trozos = []
            for P, M in datos:
                paso = int(P["paso"])
                N = P["C"].shape[1]
                minuto = np.arange(N) * paso + paso
                ventana = ((minuto >= 30) & (minuto <= 300))[None, :]
                ok = (np.abs(np.nan_to_num(M["sigma"])) >= sg_min) & (np.nan_to_num(M["rvol"]) >= rv_min)
                ok &= M["lado_vwap"] == M["dir"]
                if mercado:
                    ok &= M["mercado"] == M["dir"]
                for sgn in (1, -1):
                    sen = ok & (M["dir"] == sgn) & ventana
                    t = motor.simular(P, sen, f * P["atr"], corto=sgn < 0, max_ops=1, cierre=N - 1,
                                      tp=0, candado=(1.8, 1.0), trail=0.8)
                    t["fecha"] = P["fechas"][t["dia"]]
                    trozos.append(t)
            t = {k: np.concatenate([x[k] for x in trozos]) for k in trozos[0]}
            cal = {d: i for i, d in enumerate(sorted(set(t["fecha"].tolist())))}
            t["dia"] = np.array([cal[d] for d in t["fecha"]])
            coste = motor.costes_R(t, c, sl)
            r = motor.resumen(t, coste)
            neto = t["pnl"] - coste
            anio = pd.to_datetime(t["fecha"]).year
            fila = {"sigma_min": sg_min, "rvol_min": rv_min, "mercado": mercado, "ops": r["ops"],
                    "ops_100": 100 * r["ops"] / len(cal) / len(datos), "R_bruto": r["R_bruto"],
                    "R_neto": r["R_neto"], "t": r["t"], "acierto": r["acierto"]}
            for a in sorted(set(anio)):
                fila[f"neto_{a}"] = neto[anio == a].mean()
            filas.append(fila)
    return pd.DataFrame(filas)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    out = pd.concat([rejilla(mercado=False), rejilla(mercado=True)])
    out.to_csv(AQUI / "resultados" / "robustez.csv", index=False, float_format="%.4f")
    for m, g in out.groupby("mercado"):
        print(f"\n== con mercado a favor: {m}  (R neto | t | operaciones por 100 valores y día)")
        a = g.pivot(index="sigma_min", columns="rvol_min", values="R_neto").round(3)
        b = g.pivot(index="sigma_min", columns="rvol_min", values="t").round(1)
        c = g.pivot(index="sigma_min", columns="rvol_min", values="ops_100").round(1)
        print(a.astype(str).add(" | ").add(b.astype(str)).add(" | ").add(c.astype(str)).to_string())
        print(g[[c for c in g.columns if c.startswith("neto_")] + ["sigma_min", "rvol_min"]].round(3).to_string())
