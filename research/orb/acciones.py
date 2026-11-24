"""ORB en acciones (5 minutos): señal por acción y día y cartera diaria.

Cartera "Stocks in Play" (Zarattini, Barbon y Aziz 2024): tras el rango de apertura se ordenan
las acciones por volumen relativo (volumen del rango / media del mismo tramo en los 14 días
anteriores) y se operan las K primeras con volumen relativo > rv_min. Tamaño: riesgo `riesgo`
del capital por operación (tamaño = riesgo / distancia al stop), con cada posición limitada a
maxlev / K veces el capital (palanca total ≤ maxlev).
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import datos  # noqa: E402
from motor import atr_diario, metricas, orb  # noqa: E402


@lru_cache(maxsize=1)
def panel():
    a = datos.acciones()
    # quitar el último día (incompleto)
    D = len(a["fechas"]) - 1
    out = {k: a[k][:, :D].astype(np.float64) for k in "OHLCV"}
    C = out["C"]
    for i in range(C.shape[0]):
        c = pd.DataFrame(C[i]).ffill(axis=1).to_numpy()
        hay = ~np.isnan(out["O"][i])
        for k in "OHL":
            out[k][i] = np.where(hay, out[k][i], c)
        out["C"][i] = c
        out["V"][i] = np.nan_to_num(out["V"][i])
    out["fechas"] = a["fechas"][:D]
    out["ultimo"] = a["ultimo"][:D].astype(np.int64)
    out["tickers"] = a["tickers"]
    N = C.shape[0]
    out["atr"] = np.stack([atr_diario(out["O"][i], out["H"][i], out["L"][i], out["C"][i], out["ultimo"])
                           for i in range(N)])
    cl = np.stack([out["C"][i][np.arange(D), out["ultimo"]] for i in range(N)])
    out["cierre"] = cl
    out["prev"] = np.c_[np.full(N, np.nan), cl[:, :-1]]
    return out


def rel_vol(p, n_or, n=14):
    v = p["V"][:, :, :n_or].sum(2)  # N × D
    media = pd.DataFrame(v.T).replace(0, np.nan).rolling(n, min_periods=n).mean().shift(1).to_numpy().T
    return v / media


def señales(p, n_or=1, modo=2, lado=0, stop_frac=1.0, atr_stop=0.1, target_R=0, corte=77, rmin=0.0,
            rmax=1e12):
    """Por acción y día: dirección, rentabilidad bruta (fracción) y distancia relativa al stop."""
    N, D, _ = p["O"].shape
    dire = np.zeros((N, D))
    ret = np.zeros((N, D))
    dist = np.full((N, D), np.nan)
    vent = np.full((N, D), -1)
    for i in range(N):
        d, pe, ps, rie, ve, mot = orb(p["O"][i], p["H"][i], p["L"][i], p["C"][i], p["ultimo"], n_or, modo,
                                      lado, stop_frac, p["atr"][i], atr_stop, target_R, corte, rmin, rmax)
        ok = (d != 0) & np.isfinite(ps) & np.isfinite(pe)
        dire[i, ok] = d[ok]
        ret[i, ok] = d[ok] * (ps[ok] / pe[ok] - 1)
        dist[i, ok] = rie[ok] / pe[ok]
        vent[i, ok] = ve[ok]
    return dict(dire=dire, ret=ret, dist=dist, vent=vent)


def cartera(p, s, rank, K=20, rv=None, rv_min=1.0, riesgo=0.01, maxlev=4.0, coste=1e-4, elegibles=None):
    """rank: N × D (mayor = mejor), conocido tras el rango. Se eligen las K mejores con rv > rv_min
    (aunque luego no se activen). Devuelve rentabilidad diaria y detalles."""
    N, D = rank.shape
    r = np.zeros(D)
    nops = np.zeros(D)
    lev = np.zeros(D)
    ok_base = np.isfinite(rank) & np.isfinite(p["atr"])
    if rv is not None:
        ok_base &= np.nan_to_num(rv) > rv_min
    if elegibles is not None:
        ok_base &= elegibles
    tr_ret, tr_w = [], []
    for d in range(D):
        cand = np.where(ok_base[:, d])[0]
        if len(cand) == 0:
            continue
        top = cand[np.argsort(-rank[cand, d])[:K]]
        act = top[s["dire"][top, d] != 0]
        if len(act) == 0:
            continue
        w = np.minimum(riesgo / s["dist"][act, d], maxlev / K)
        rn = s["ret"][act, d] - 2 * coste
        r[d] = (w * rn).sum()
        nops[d] = len(act)
        lev[d] = w.sum()
        tr_ret.append(rn)
        tr_w.append(w)
    tr = np.concatenate(tr_ret) if tr_ret else np.array([])
    return dict(r=r, nops=nops, lev=lev, trades=tr)


def informe(res, fechas, nombre=""):
    m = metricas(res["r"], fechas)
    t = res["trades"]
    años = pd.DatetimeIndex(fechas).year.to_numpy()
    por_año = {y: np.prod(1 + res["r"][años == y]) - 1 for y in np.unique(años)}
    return dict(nombre=nombre, cagr=m["cagr"], sharpe=m["sharpe"], maxdd=m["maxdd"], vol=m["vol"],
                ops_dia=res["nops"].mean(), lev_medio=res["lev"][res["nops"] > 0].mean(),
                pb_op=t.mean() * 1e4 if len(t) else np.nan, acierto=(t > 0).mean() if len(t) else np.nan,
                **{str(y): v for y, v in por_año.items()})
