"""Ronda 5: entradas mejoradas con análisis técnico y reserva SOLO de acciones.

- Reserva: cesta de acciones del S&P 500 (``liq100_dv``: las 100 más líquidas
  ponderadas por volumen en dólares, elegida en la búsqueda; ``ew`` como contraste).
  Sin ETFs ni efectivo.
- ``score_tec``: nº de banderas técnicas a favor (``modulos_tec.py``, elegidas solo con
  la búsqueda): estiramiento bajo EMA5/EMA10/WMA20/VWAP10, Bollinger, estocástico,
  distancia a SMA50, clímax de volumen, OBV.
- Usos: ``filtro`` (exigir score_tec >= k), ``prioridad`` (sumarlo a la puntuación de
  módulos), ``peso`` (convicción con la puntuación conjunta).

Selección en la búsqueda (oct-2021 →), validación 2002-sep21.

Uso: python research/caidas/ronda5.py
"""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import eventos as ev
import finalistas3 as f3
import motor_v3 as v3
import ronda3 as r3
import tecnicos as tc

OUT = Path(__file__).parent / "resultados"
VENTANAS = {"busq": ("2021-10-01", None), "valid": ("2002-01-02", "2021-10-01"),
            "0225": ("2002-01-02", "2026-01-01"), "total": ("2002-01-02", None)}


def score_tec(x: pd.DataFrame) -> np.ndarray:
    reglas = json.loads((OUT / "modulos_tec_elegidas.json").read_text())
    s = np.zeros(len(x))
    for r, op, v in reglas.values():
        col = x[r]
        s += {"<=": col <= v, ">=": col >= v, "<": col < v}[op].fillna(False).to_numpy()
    return s


def componentes_tec() -> dict:
    d = r3.cargar()
    e = ev.cargar()
    tec = tc.cargar()
    out = {}
    for nombre, x, salida, conv, base_col in (("D1", f3.candidatas_d1(d), "sma10", True, "score_v2"),
                                              ("C1", f3.candidatas_c1(d, e), "rsi2", False, "score_limpio")):
        x = x.merge(tec[["t", "j"] + [c for c in tec.columns if c not in ("t", "j") and c not in x.columns]],
                    on=["t", "j"], how="left")
        x["score_tec"] = score_tec(x)
        x["base_score"] = x[base_col]
        out[nombre] = (x, salida, conv)
    return out


def preparar(x: pd.DataFrame, conv: bool, kmin: int, uso: str) -> tuple[pd.DataFrame, np.ndarray | None]:
    y = x[x.score_tec >= kmin].copy()
    prio = y.prio.to_numpy(np.float64)
    if uso in ("prioridad", "peso"):
        prio = prio + y.score_tec.to_numpy()  # la puntuación técnica pesa como los módulos
    w = None
    if conv:
        w = np.minimum(y.peso.to_numpy(), 1.5)
    if uso == "peso":
        total = y.base_score.to_numpy() + y.score_tec.to_numpy()
        ref = np.median(x.base_score + x.score_tec)  # mediana de las candidatas del componente
        w = np.clip(1 + (total - ref) / 4, 0.5, 1.5)
    y = y.assign(prio=prio)
    return y, w


def main() -> None:
    comp = componentes_tec()
    # monotonía de la puntuación técnica (operación real, neta de 20 pb)
    import modulos_tec as mt
    cj = mt.conjuntos()
    for nombre, x in cj.items():
        x["score_tec"] = score_tec(x)
        tabla = x.groupby([x.score_tec.clip(upper=8), np.where(x.ventana == "valid", "valid", "busq")]).ret.agg(
            ["mean", "size"]).unstack()
        print(f"\n{nombre}: retorno neto medio por nº de banderas técnicas (%)")
        print((tabla["mean"] * 100).round(2).T.to_string())
        print(tabla["size"].T.to_string())
    filas, curvas = [], []
    t0 = time.time()
    for (nombre, (x, salida, conv)), kmin, uso, res in itertools.product(
        comp.items(), (0, 1, 2, 3, 4, 5), ("filtro", "prioridad", "peso"), ("liq100_dv", "ew"),
    ):
        y, w = preparar(x, conv, kmin, uso)
        fila = {"componente": nombre, "kmin": kmin, "uso": uso, "reserva": res, "candidatas": len(y)}
        for vn, (a, b) in VENTANAS.items():
            r = v3.simular(y.t.to_numpy(), y.j.to_numpy(), y.prio.to_numpy(np.float64), peso=w, K=5, frac=1 / 3,
                           H=20, salida=salida, reserva_=res, palanca=1.0, inicio=a, fin=b)
            fila |= {f"{vn}_{k}": val for k, val in r.items() if k not in ("equity", "serie_expo", "serie_lev")}
            if vn == "total":
                curvas.append(r["equity"].pct_change().fillna(0).to_numpy(np.float32))
        filas.append(fila)
    df = pd.DataFrame(filas)
    df.to_csv(OUT / "ronda5.csv", index=False, float_format="%.5f")
    np.save(ev.SALIDA.with_name("ronda5_retornos.npy"), np.vstack(curvas))
    print(f"\n{len(df)} carteras en {time.time() - t0:.0f}s")
    pd.set_option("display.width", 260)
    cols = ["componente", "kmin", "uso", "reserva", "candidatas", "busq_cagr", "busq_maxdd", "valid_cagr",
            "valid_maxdd", "0225_cagr", "total_cagr", "total_maxdd", "total_sharpe", "total_trades_año",
            "total_exposicion"]
    for c in ("D1", "C1"):
        g = df[df.componente == c].sort_values("busq_cagr", ascending=False)
        print(f"\n{c} (ordenado por CAGR de búsqueda)")
        print(g[cols].head(10).round(3).to_string(index=False))
        print("referencia sin técnica (kmin=0, filtro):")
        print(g[(g.kmin == 0) & (g.uso == "filtro")][cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
