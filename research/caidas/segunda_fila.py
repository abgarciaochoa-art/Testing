"""Ronda 6 · umbral ADAPTATIVO: candidatas de segunda fila solo cuando hay pocas compras.

Las candidatas de primera fila son las de D1-técnica. Las de segunda fila (definidas
antes de mirar resultados, con la evidencia de ``ronda5c``) solo pueden entrar cuando la
cartera de caídas tiene menos de ``max_t2`` posiciones; así no desplazan a las buenas y
usan capital que de otro modo estaría ocioso:

- T2a: en tendencia, caída >= 12 % desde el máximo de 20 d, >= 4 módulos y >= 2 técnicas;
- T2b: en tendencia, caída >= 10 %, módulos + técnicas >= 10.

Peso de la segunda fila: medio hueco o uno. Siempre detrás de la primera fila en prioridad.
Reserva contracíclica (``liq100_vg_minvol``); niveles 2 (sin deuda en caídas) y 3 (2x).
Selección por CAGR en la búsqueda; validación 2002-sep21.

Uso: python research/caidas/segunda_fila.py
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

import escalera as es
import motor_v3 as v3
from definitiva import D1_CFG
from ronda5 import componentes_tec
from ronda5b import preparar
from ronda5c import candidatas_d2

OUT = Path(__file__).parent / "resultados"


def candidatas(t2: str | None, peso_t2: float) -> tuple[pd.DataFrame, np.ndarray]:
    comp = componentes_tec()
    x, _, conv = comp["D1"]
    y1, w1 = preparar(x, conv, D1_CFG["kmin"], D1_CFG["uso"], D1_CFG["tam"], D1_CFG["frac"])
    y1 = y1[["t", "j", "prio"]].assign(w=w1)
    if t2 is None:
        return y1, y1.w.to_numpy()
    d = candidatas_d2()
    if t2 == "T2a":
        y2 = d[(d.dd20 <= -0.12) & (d.score_v2 >= 4) & (d.score_tec >= 2)]
    else:
        y2 = d[(d.dd20 <= -0.10) & (d.total >= 10)]
    y2 = y2[["t", "j", "prio"]].assign(prio=lambda z: z.prio - 1000.0, w=-peso_t2)  # siempre detrás
    todo = pd.concat([y1, y2]).sort_values("prio", ascending=False).drop_duplicates(["t", "j"])
    return todo, todo.w.to_numpy()


def main() -> None:
    filas = []
    for (t2, peso_t2, max_t2), (nivel, pal) in itertools.product(
        ((None, 0, 0), ("T2a", 0.5, 1), ("T2a", 0.5, 2), ("T2a", 1.0, 1), ("T2a", 1.0, 2),
         ("T2b", 0.5, 1), ("T2b", 0.5, 2), ("T2b", 1.0, 1), ("T2b", 1.0, 2)),
        (("nivel 2", 1.0), ("nivel 3", 2.0)),
    ):
        y, w = candidatas(t2, peso_t2)
        r = v3.simular(y.t.to_numpy(), y.j.to_numpy(), y.prio.to_numpy(np.float64), peso=w, K=5, frac=0.5, H=20,
                       salida="sma10", reserva_="liq100_vg_minvol", palanca=pal, palanca_off=1.0,
                       vol_objetivo=0.15 if pal > 1 else None, max_t2=max_t2)
        filas.append({"segunda_fila": t2 or "ninguna", "peso_t2": peso_t2, "max_abiertas": max_t2, "nivel": nivel,
                      **es.metricas_ventanas(r["equity"]), "operaciones/año": r["trades_año"],
                      "acierto": r["acierto"]})
    df = pd.DataFrame(filas)
    df.to_csv(OUT / "segunda_fila.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 250)
    cols = ["segunda_fila", "peso_t2", "max_abiertas", "nivel", "CAGR búsqueda", "maxDD búsqueda", "CAGR validación",
            "CAGR 2002-2025", "CAGR total", "maxDD total", "Sharpe", "operaciones/año", "acierto"]
    for nv in ("nivel 2", "nivel 3"):
        print(df[df.nivel == nv].sort_values("CAGR búsqueda", ascending=False)[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
