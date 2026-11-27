"""Curvas de R acumulado (neto de costes) de cuatro formas de operar la horquilla, por activo:

- original:   entrar siempre que se esté fuera, largo, stop 0,1 ATR, objetivo 2R, sin límite.
- filtro:     lo mismo, pero solo por encima de la VWAP y de la apertura.
- plan:       a favor del día (largo si va > +0,25 ATR desde la apertura, corto si va < -0,25),
              stop 0,2 ATR, candado (a +1,8R el stop sube a +1R y luego sigue a 0,8R),
              una operación por lado y día.
- plan_corre: el plan sin objetivo ni candado: se deja correr hasta el cierre.

Uso: python research/bracket_intradia/curvas.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import estudio as E  # noqa: E402
import motor  # noqa: E402

FORMAS = {
    "original": [("largo", "siempre", 0.1, dict(tp=2))],
    "filtro": [("largo", "vwap+apertura", 0.1, dict(tp=2))],
    "plan": [(lado, "dia_fuerte", 0.2, dict(tp=0, candado=(1.8, 1.0), trail=0.8, max_ops=1))
             for lado in ("largo", "corto")],
    "plan_corre": [(lado, "dia_fuerte", 0.2, dict(tp=0, max_ops=1)) for lado in ("largo", "corto")],
}


def diario(clase: str, nombres: list[str]) -> pd.DataFrame:
    cols = {}
    for forma, partes in FORMAS.items():
        serie = []
        for nombre in nombres:
            for lado, s, f, kw in partes:
                t, _ = E.correr(nombre, s, lado=lado, f_atr=f, **kw)
                c, sl = E.COSTES[clase]
                neto = t["pnl"] - motor.costes_R(t, c, sl)
                serie.append(pd.Series(neto, index=pd.to_datetime(t["fecha"])))
        todo = pd.concat(serie).groupby(level=0).sum()
        if clase == "acciones":
            todo = todo / len(nombres)  # R medio por acción y día (cartera de 40)
        cols[forma] = todo
    fechas = sorted(set().union(*[set(E.panel(n)[0]["fechas"]) for n in nombres]))
    return pd.DataFrame(cols).reindex(pd.to_datetime(fechas)).fillna(0.0)


def main() -> pd.DataFrame:
    filas = []
    for clase, nombre in E.todos():
        d = diario(clase, [nombre]).cumsum()
        d["inst"] = clase
        filas.append(d)
    d = diario("acciones", E.acciones()).cumsum()
    d["inst"] = "acciones40"
    filas.append(d)
    out = pd.concat(filas).rename_axis("fecha").reset_index()
    out.to_csv(E.OUT / "curvas.csv", index=False, float_format="%.3f")
    return out


if __name__ == "__main__":
    c = main()
    print(c.groupby("inst").tail(1).round(1).to_string())
