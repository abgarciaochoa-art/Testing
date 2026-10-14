"""Control de sesgo por datos faltantes en la puntuación de módulos.

Las series de CRSP (sobre todo empresas que luego desaparecieron) no traen apertura,
rango ni volumen, así que no pueden puntuar en 3 de los 11 módulos (IBS, hueco sin
volumen, tamaño). Si eso las apartara de la cartera, la validación se beneficiaría de
saber "quién acabará deslistada". Control: imputar a esas filas la tasa media de cada
módulo afectado y repetir D1-def50. Si el resultado no empeora, no hay beneficio.

Uso: python research/caidas/sesgo_faltantes.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import eventos as ev
import motor as m
import motor_cartera as mc
import ronda3 as r3
from meta import MODULOS, MODULOS_V2

OUT = Path(__file__).parent / "resultados"
DEPENDIENTES = ("IBS", "hueco", "grandes")


def main() -> None:
    p = m.cargar()
    d = r3.cargar()
    e = ev.cargar()
    cand = d[d.dd20 <= -0.15].copy()
    cand["sin_datos"] = p.sin_apertura[cand.t.to_numpy(), cand.j.to_numpy()]
    banderas = json.loads((OUT / "meta_banderas.json").read_text())["banderas"]
    todos = {**{f"{a}: {b}": f for a, gs in MODULOS.items() for b, f in gs.items()},
             **{f"{a}: {b}": f for a, gs in MODULOS_V2.items() for b, f in gs.items()}}
    dep = [b for b in banderas if any(k in b for k in DEPENDIENTES)]
    ee = e[e.sobre_sma200 == 1]
    con = ee[~p.sin_apertura[ee.t.to_numpy(), ee.j.to_numpy()]]
    tasa = {b: float(todos[b](con).fillna(False).mean()) for b in dep}
    cand["score_imp"] = cand.score_v2 + np.where(cand.sin_datos, sum(tasa.values()), 0.0)
    val = cand.fecha < "2021-10-01"
    res = {"fraccion_sin_datos_validacion": float(cand[val].sin_datos.mean()),
           "score_medio_con_datos": float(cand[val & ~cand.sin_datos].score_v2.mean()),
           "score_medio_sin_datos": float(cand[val & cand.sin_datos].score_v2.mean()),
           "tasas_imputadas": tasa}
    for col in ("score_v2", "score_imp"):
        y = cand[cand[col] >= 5]
        prio = (y[col] + y.meta_pct.fillna(0.5) * 0.99).to_numpy(np.float64)
        w = np.minimum(np.clip((y[col].to_numpy() - 4) / 2, 0.5, 2.0), 1.5)
        for nombre, (a, b) in {"validacion": ("2002-01-02", "2021-10-01"), "total": ("2002-01-02", None)}.items():
            r = mc.simular(y.t.to_numpy(), y.j.to_numpy(), prio, peso=w, K=3, H=20, salida="sma10",
                           reserva="spy_tend", inicio=a, fin=b)
            res[f"{col}_{nombre}"] = {"cagr": r["cagr"], "maxdd": r["maxdd"], "sharpe": r["sharpe"]}
    (OUT / "sesgo_faltantes.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
