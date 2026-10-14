"""Entradas con ORDEN LÍMITE (comprar más barato) a nivel de operación.

Señal al cierre de t. Orden de compra para la sesión t+1 (y opcionalmente t+2) con
límite = C_t · (1 - k · ATR14/C_t):
- si la apertura ya está por debajo del límite, se compra a la apertura;
- si no, se compra al límite solo si el mínimo lo PERFORA en al menos 10 pb (tocar el
  límite no basta: cola de órdenes);
- si no se ejecuta, no hay operación (sin efecto en la cartera).
En las series sin apertura/rango (CRSP) se compra a mercado, como antes.
La salida es la de cada componente (D1: > SMA10; C1: RSI2 >= 70; máx. 20 sesiones desde
la entrada), neta de 20 pb.

Mide, por k: tasa de ejecución, retorno de las ejecutadas y retorno ESPERADO por señal
(las no ejecutadas aportan 0), en las dos mitades de la búsqueda y en la validación.

Uso: python research/caidas/limites.py
"""

from __future__ import annotations

from pathlib import Path

import numba
import numpy as np
import pandas as pd

import modulos_tec as mt
import motor as m

OUT = Path(__file__).parent / "resultados"
KS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5)


@numba.njit(cache=True)
def _limite(ts, js, atr, O, H, L, C, sinap, sal, k, dias_validez, H_max):
    n = len(ts)
    ret = np.full(n, np.nan)
    lleno = np.zeros(n, dtype=np.bool_)
    T = C.shape[0]
    for e in range(n):
        t, j = ts[e], js[e]
        if t + 1 >= T or np.isnan(C[t, j]):
            continue
        px = np.nan
        te = -1
        if sinap[t + 1, j] or np.isnan(atr[e]) or k == 0.0:
            if not np.isnan(O[t + 1, j]):
                px = O[t + 1, j]
                te = t + 1
        else:
            lim = C[t, j] * (1 - k * atr[e])
            for s in range(t + 1, min(t + 1 + dias_validez, T)):
                if np.isnan(O[s, j]):
                    continue
                if O[s, j] <= lim:
                    px = O[s, j]
                    te = s
                    break
                if L[s, j] < lim * (1 - 0.001):
                    px = lim
                    te = s
                    break
        if te < 0:
            continue
        lleno[e] = True
        ult = px
        hecho = False
        for s in range(te, min(te + H_max, T)):
            c = C[s, j]
            if np.isnan(c):
                continue
            ult = c
            if sal[s, j] or s - te + 1 >= H_max:
                salida = O[s + 1, j] if s + 1 < T and not np.isnan(O[s + 1, j]) else c
                ret[e] = salida / px - 1
                hecho = True
                break
        if not hecho:
            ret[e] = ult / px - 1
    return ret, lleno


def main() -> None:
    p = m.cargar()
    cj = mt.conjuntos()
    sal = {"D1": np.nan_to_num(p.close > m.sma(10)).astype(bool), "C1": np.nan_to_num(m.rsi(2) >= 70).astype(bool)}
    filas = []
    for nombre, x in cj.items():
        ts, js = x.t.to_numpy().astype(np.int64), x.j.to_numpy().astype(np.int64)
        atr = x.atr_pct.to_numpy(np.float64)
        for k in KS:
            for validez in (1, 2):
                if k == 0.0 and validez == 2:
                    continue
                r, lleno = _limite(ts, js, atr, p.open, p.high, p.low, p.close, p.sin_apertura, sal[nombre],
                                   float(k), validez, 20)
                neto = r - 0.002
                for w in ("busq1", "busq2", "valid"):
                    en = (x.ventana == w).to_numpy()
                    ll = lleno[en]
                    rr = neto[en][ll]
                    filas.append({"componente": nombre, "k_atr": k, "dias_orden": validez, "ventana": w,
                                  "señales": int(en.sum()), "ejecucion": float(ll.mean()),
                                  "ret_ejecutadas": float(np.nanmean(rr)) if len(rr) else np.nan,
                                  "ret_por_señal": float(np.nansum(rr) / en.sum())})
    df = pd.DataFrame(filas)
    df.to_csv(OUT / "limites.csv", index=False, float_format="%.5f")
    pd.set_option("display.width", 220)
    for c in ("D1", "C1"):
        v = df[df.componente == c].pivot_table(index=["k_atr", "dias_orden"], columns="ventana",
                                               values=["ejecucion", "ret_ejecutadas", "ret_por_señal"])
        for col in ("ret_ejecutadas", "ret_por_señal"):
            for w in ("busq1", "busq2", "valid"):
                v[(col, w)] *= 100
        print(f"\n{c}: ejecución, retorno de las ejecutadas (%) y retorno por señal (%)")
        print(v.round(3).to_string())


if __name__ == "__main__":
    main()
