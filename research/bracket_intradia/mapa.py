"""Mapa de "cartas": probabilidad de ganar una horquilla 2:1 según cuánto se ha movido el
precio desde la apertura A FAVOR de la operación (en ATR) en el momento de entrar.

Se entra en todas las velas posibles (largos y cortos) con stop de 0,2 ATR y objetivo 2R; cada
operación se clasifica por el movimiento previo. Si el mercado tiene momento intradía, el
acierto sube cuanto más a favor va el día; si revierte, baja.

Además: probabilidad de un día como el del usuario (5 ganadas de 7 a 2:1) sin ventaja, y
cuántas operaciones hacen falta para distinguir una ventaja real del azar.

Uso: python research/bracket_intradia/mapa.py
"""

from __future__ import annotations

import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import estudio as E  # noqa: E402
import motor  # noqa: E402
import senales as sn  # noqa: E402

CORTES = [-np.inf, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, np.inf]
ETIQ = ["< -0,75", "-0,75 a -0,5", "-0,5 a -0,25", "-0,25 a 0", "0 a 0,25", "0,25 a 0,5",
        "0,5 a 0,75", "> 0,75"]


def _trades(nombre: str, lado: str, f: float = 0.2) -> pd.DataFrame:
    P, _, _ = E.panel(nombre)
    t, _ = E.correr(nombre, "siempre", lado=lado, f_atr=f, tp=2)
    F = sn.rasgos(P)
    j = t["j"] - 1  # vela de la señal
    mov = F["desde_apertura"][t["dia"], j] * (1 if lado == "largo" else -1)
    vw = F["dist_vwap"][t["dia"], j] * (1 if lado == "largo" else -1)
    return pd.DataFrame({"mov": mov, "vwap": vw, "pnl": t["pnl"], "gana": t["pnl"] > 0,
                         "minuto": j * int(P["paso"]), "fecha": t["fecha"]})


def mapa() -> pd.DataFrame:
    filas = []
    fuentes = [(c, n) for c, n in E.todos()] + [("acciones40", a) for a in E.acciones()]
    tablas: dict[str, list] = {}
    for clase, nombre in fuentes:
        for lado in ("largo", "corto"):
            tablas.setdefault(clase, []).append(_trades(nombre, lado))
    for clase, trozos in tablas.items():
        df = pd.concat(trozos)
        df["tramo"] = pd.cut(df["mov"], CORTES, labels=ETIQ)
        g = df.groupby("tramo", observed=True)
        r = g.agg(ops=("pnl", "size"), acierto=("gana", "mean"), R=("pnl", "mean"))
        r["inst"] = clase
        filas.append(r.reset_index())
    return pd.concat(filas, ignore_index=True)


def dia_usuario() -> dict:
    """5 ganadas y 2 perdidas a 2:1 = +8R. ¿Qué probabilidad tiene sin ventaja?"""
    def p_al_menos(k: int, n: int, p: float) -> float:
        return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))

    out = {f"p_5de7_acierto_{p:.2f}": p_al_menos(5, 7, p) for p in (1 / 3, 0.36, 0.40)}
    # operaciones necesarias para que una ventaja de +x R sea distinguible (t = 2)
    sd = np.sqrt((1 / 3) * 4 + (2 / 3) * 1)  # desviación de una horquilla 2:1 ≈ 1,41 R
    for x in (0.02, 0.05, 0.1, 0.2):
        out[f"ops_para_ver_{x}R"] = int(np.ceil((2 * sd / x) ** 2))
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    m = mapa()
    m.to_csv(E.OUT / "mapa.csv", index=False, float_format="%.4f")
    print(m.pivot_table(index="tramo", columns="inst", values="acierto", observed=True)
          .reindex(ETIQ).round(3).to_string())
    print(m.pivot_table(index="tramo", columns="inst", values="R", observed=True)
          .reindex(ETIQ).round(3).to_string())
    print(m.pivot_table(index="tramo", columns="inst", values="ops", observed=True)
          .reindex(ETIQ).to_string())
    u = dia_usuario()
    pd.Series(u).to_csv(E.OUT / "dia_usuario.csv")
    print(u)
