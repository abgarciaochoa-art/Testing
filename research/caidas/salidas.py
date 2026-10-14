"""Estudio de salidas a nivel de operación para las candidatas de D1 y C1.

Para cada candidata (entrada a la apertura de t+1) se simula cada regla de salida
(ejecución a la apertura siguiente a la señal, máx. 20 sesiones) y se mide el
retorno neto (20 pb ida y vuelta), los días y el retorno POR DÍA invertido (lo que
cuenta para el CAGR cuando el capital rota). Se elige en la búsqueda (oct-2021 →) y
se valida en 2002 → sep-2021.

Uso: python research/caidas/salidas.py
"""

from __future__ import annotations

from pathlib import Path

import numba
import numpy as np
import pandas as pd

import eventos as ev
import finalistas3 as f3
import motor as m
import ronda3 as r3

OUT = Path(__file__).parent / "resultados"
CORTE = pd.Timestamp("2021-10-01")
H = 20
# (nombre, código, parámetro)
REGLAS = [("cierre > SMA5", 0, 5), ("cierre > SMA10", 0, 10), ("cierre > SMA20", 0, 20),
          ("RSI2 ≥ 70", 1, 70), ("RSI2 ≥ 90", 1, 90),
          ("objetivo +5 %", 2, 0.05), ("objetivo +10 %", 2, 0.10),
          ("primer cierre al alza", 3, 0), ("solo tiempo 3 d", 4, 3), ("solo tiempo 5 d", 4, 5),
          ("solo tiempo 10 d", 4, 10), ("solo tiempo 20 d", 4, 20),
          ("trailing 5 % desde máx.", 5, 0.05), ("SMA10 o pérdida -10 %", 6, 0.10)]


@numba.njit(cache=True)
def _salidas(ts, js, O, C, sma5, sma10, sma20, rsi2, codigo, par, H):
    n = len(ts)
    ret = np.full(n, np.nan)
    dias = np.full(n, np.nan)
    T = C.shape[0]
    for e in range(n):
        t, j = ts[e], js[e]
        if t + 1 >= T or np.isnan(O[t + 1, j]):
            continue
        pe = O[t + 1, j]
        maxc = pe
        ult = pe
        hecho = False
        for s in range(t + 1, min(t + 1 + H, T)):
            c = C[s, j]
            if np.isnan(c):
                continue
            ult = c
            maxc = max(maxc, c)
            k = s - t
            if codigo == 0:
                sm = sma5[s, j] if par == 5 else (sma10[s, j] if par == 10 else sma20[s, j])
                cond = c > sm
            elif codigo == 1:
                cond = rsi2[s, j] >= par
            elif codigo == 2:
                cond = c >= pe * (1 + par)
            elif codigo == 3:
                cond = s > t + 1 and c > C[s - 1, j]
            elif codigo == 4:
                cond = k >= par
            elif codigo == 5:
                cond = c < maxc * (1 - par) and maxc > pe
            else:
                cond = c > sma10[s, j] or c <= pe * (1 - par)
            if cond or k >= H:
                px = O[s + 1, j] if s + 1 < T and not np.isnan(O[s + 1, j]) else c
                ret[e] = px / pe - 1
                dias[e] = s + 1 - t
                hecho = True
                break
        if not hecho:
            ret[e] = ult / pe - 1
            dias[e] = H
    return ret, dias


def main() -> None:
    p = m.cargar()
    d = r3.cargar()
    e = ev.cargar()
    conj = {"D1": f3.candidatas_d1(d), "C1": f3.candidatas_c1(d, e)}
    mats = (p.open, p.close, m.sma(5), m.sma(10), m.sma(20), m.rsi(2))
    filas = []
    for nombre, x in conj.items():
        ts, js = x.t.to_numpy().astype(np.int64), x.j.to_numpy().astype(np.int64)
        fechas = pd.DatetimeIndex(p.fechas[ts])
        for regla, cod, par in REGLAS:
            ret, dias = _salidas(ts, js, *mats, cod, float(par), H)
            neto = ret - 0.002
            for w, sel in (("busq", fechas >= CORTE), ("valid", fechas < CORTE)):
                r, dd = neto[sel], dias[sel]
                ok = ~np.isnan(r)
                filas.append({"conjunto": nombre, "regla": regla, "ventana": w, "n": int(ok.sum()),
                              "ret_medio": float(np.nanmean(r)), "dias_medios": float(np.nanmean(dd)),
                              "ret_por_dia": float(np.nansum(r) / np.nansum(dd[ok])),
                              "acierto": float(np.nanmean(r > 0))})
    df = pd.DataFrame(filas)
    df.to_csv(OUT / "salidas.csv", index=False, float_format="%.5f")
    pd.set_option("display.width", 220)
    for c in ("D1", "C1"):
        x = df[df.conjunto == c].pivot(index="regla", columns="ventana",
                                       values=["ret_medio", "dias_medios", "ret_por_dia"])
        x[("ret_medio", "busq")] *= 100
        x[("ret_medio", "valid")] *= 100
        x[("ret_por_dia", "busq")] *= 100
        x[("ret_por_dia", "valid")] *= 100
        print(f"\n{c} (ret. en %, ordenado por retorno/día en la búsqueda)")
        print(x.sort_values(("ret_por_dia", "busq"), ascending=False).round(3).to_string())


if __name__ == "__main__":
    main()
