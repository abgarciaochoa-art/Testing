"""Tabla de eventos de caída: una fila por (sesión, acción) candidata con TODOS los
rasgos en t y sus resultados posteriores.

Es la pieza central para iterar rápido: evaluar un módulo de mejora (o combinar
varios, o entrenar un modelo de puntuación más adelante) es un filtro de pandas
sobre esta tabla, no un backtest.

Candidata amplia: miembro PIT, líquida, y (r5 <= -5 % o r10 <= -7 % o dd20 <= -10 %).

Resultados (entrada a la APERTURA de t+1, como en el motor):
- ``f{h}``: retorno bruto hasta el cierre de t+h; ``x{h}``: en exceso sobre la media
  equiponderada de los miembros en la misma ventana.
- ``op_sma5``: retorno de la operación real "salir a la apertura siguiente al primer
  cierre > SMA5, máx. 20 sesiones", y ``op_dias`` su duración (numba).

Uso: python research/caidas/eventos.py  -> data/cache/caidas/eventos.parquet
"""

from __future__ import annotations

import time

import numba
import numpy as np
import pandas as pd

import motor as m
import rasgos as rg

SALIDA = rg.CACHE / "eventos.parquet"
HORIZ = (1, 3, 5, 10, 20)


@numba.njit(cache=True)
def _operaciones(ts, js, O, C, sma5, H):
    n = len(ts)
    ret = np.full(n, np.nan)
    dias = np.full(n, -1)
    T = C.shape[0]
    for e in range(n):
        t, j = ts[e], js[e]
        if t + 1 >= T or np.isnan(O[t + 1, j]):
            continue
        pe = O[t + 1, j]
        ult = pe
        for s in range(t + 1, min(t + 1 + H, T)):
            if not np.isnan(C[s, j]):
                ult = C[s, j]
            if (s - t >= H) or (C[s, j] > sma5[s, j]):
                px = O[s + 1, j] if s + 1 < T and not np.isnan(O[s + 1, j]) else ult
                ret[e] = px / pe - 1
                dias[e] = s + 1 - t
                break
        else_ = ret[e]
        if np.isnan(else_):
            ret[e] = ult / pe - 1
            dias[e] = H
    return ret, dias


def construir() -> None:
    t0 = time.time()
    p = m.cargar()
    R = rg.cargar_rasgos()
    T, N = p.close.shape
    liq = m.liquido()
    with np.errstate(invalid="ignore"):
        cand = p.miembro & liq & ((R["r5"] <= -0.05) | (R["r10"] <= -0.07) | (R["dd20"] <= -0.10))
    cand[: 260] = False
    ts, js = np.nonzero(cand)
    print(f"{len(ts):,} eventos candidatos", flush=True)

    # resultados posteriores
    base = p.miembro & liq
    cols = {}
    for h in HORIZ:
        fwd = np.full((T, N), np.nan)
        fwd[: T - h] = p.close_ff[h:] / p.open[1 : T - h + 1] - 1
        mu = np.nanmean(np.where(base, fwd, np.nan), axis=1)
        cols[f"f{h}"] = fwd[ts, js]
        cols[f"x{h}"] = fwd[ts, js] - mu[ts]
    ret, dias = _operaciones(ts, js, p.open, p.close, m.sma(5), 20)
    cols["op_sma5"], cols["op_dias"] = ret, dias
    mu_op = pd.Series(cols["x5"]).groupby(ts).transform("mean")  # sólo informativo
    del mu_op

    # rasgos en t
    rasgos = {k: v[ts, js] for k, v in R.items() if k != "pares" and v.ndim == 2}
    # derivados: peor noche y mayor volumen relativo de los últimos 10 días
    noche_min10 = pd.DataFrame(R["noche1"]).rolling(10, min_periods=1).min().to_numpy()
    volmax10 = pd.DataFrame(R["vol_rel"]).rolling(10, min_periods=1).max().to_numpy()
    rasgos["noche_min10"] = noche_min10[ts, js]
    rasgos["volmax10"] = volmax10[ts, js]
    dr = rasgos["dias_res"]
    rasgos["dias_res"] = np.where(dr <= 95, dr, np.nan)  # dato de resultados obsoleto -> NaN
    # sesiones consecutivas que la acción lleva siendo candidata
    racha = np.zeros((T, N), dtype=np.int16)
    for t in range(1, T):
        racha[t] = np.where(cand[t], racha[t - 1] + 1, 0)
    rasgos["dias_candidata"] = racha[ts, js]

    df = pd.DataFrame({"t": ts.astype(np.int32), "j": js.astype(np.int32), **rasgos, **cols})
    df["fecha"] = p.fechas[ts]
    df["ticker"] = p.tickers[js]
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    df.to_parquet(SALIDA, index=False)
    print(f"{df.shape} -> {SALIDA} ({time.time() - t0:.0f}s)")


def cargar() -> pd.DataFrame:
    return pd.read_parquet(SALIDA)


if __name__ == "__main__":
    construir()
