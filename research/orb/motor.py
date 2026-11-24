"""Motor de ORB (Opening Range Breakout) sobre matrices días × velas.

Una operación por día como máximo. Todo se cierra en la última vela del día.

Modos de entrada:
  'ruptura' : orden stop en el máximo (largo) / mínimo (corto) del rango de apertura; la primera
              que se toque después del rango. Se llena al nivel o, si la vela abre más allá, a su
              apertura.
  'color+ruptura' (modo 2): dirección por el color de la vela del rango y entrada con orden stop
              en el extremo del rango de ese lado (Zarattini, Barbon y Aziz 2024, "Stocks in Play").
  'color'   : (Zarattini y Aziz 2023) dirección de la vela del rango (cierre > apertura → largo);
              entrada a mercado a la apertura de la vela siguiente al rango.
Stop (en múltiplos del ancho del rango R, medido desde el nivel de entrada):
  stop_frac=1 → stop al otro lado del rango; 0.5 → a mitad; también stop por ATR si atr_stop>0
  (stop = atr_stop × ATR diario de 14 días, a distancia fija de la entrada).
Objetivo: target_R × riesgo (0 = sin objetivo, hasta el cierre).
Si una vela toca stop y objetivo, se asume el stop (conservador). Si la vela de entrada toca
también el stop, se asume que salta.
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def orb(O, H, L, C, ultimo, n_or, modo, lado, stop_frac, atr, atr_stop, target_R, corte,
        rmin, rmax):
    """Devuelve por día: dirección (+1/-1/0), precio de entrada, de salida, riesgo (en precio),
    vela de entrada, motivo de salida (1 stop, 2 objetivo, 3 cierre).

    lado: 0 ambos, 1 solo largos, -1 solo cortos. corte: última vela en la que se puede entrar.
    rmin/rmax: filtrar días por ancho del rango relativo al ATR (R/ATR fuera de [rmin, rmax] → no
    se opera).
    """
    D, B = O.shape
    dire = np.zeros(D)
    pe = np.full(D, np.nan)
    ps = np.full(D, np.nan)
    rie = np.full(D, np.nan)
    ve = np.full(D, -1)
    mot = np.zeros(D)
    for d in range(D):
        u = ultimo[d]
        if u <= n_or + 1:
            continue
        hi = H[d, 0]
        lo = L[d, 0]
        for b in range(1, n_or):
            if H[d, b] > hi:
                hi = H[d, b]
            if L[d, b] < lo:
                lo = L[d, b]
        R = hi - lo
        if R <= 0 or not np.isfinite(R):
            continue
        a = atr[d]
        if np.isfinite(a) and a > 0:
            q = R / a
            if q < rmin or q > rmax:
                continue
        elif rmin > 0 or rmax < 1e9 or atr_stop > 0:
            continue
        s = 0
        entrada = np.nan
        b0 = -1
        if modo == 1:  # color
            o0 = O[d, 0]
            c0 = C[d, n_or - 1]
            if c0 > o0:
                s = 1
            elif c0 < o0:
                s = -1
            if s == 0 or (lado != 0 and s != lado):
                continue
            b0 = n_or
            entrada = O[d, b0]
        elif modo == 3:  # cierre fuera del rango → entrada a la apertura de la vela siguiente
            for b in range(n_or, min(corte, u - 1) + 1):
                if C[d, b] > hi and lado != -1:
                    s = 1
                elif C[d, b] < lo and lado != 1:
                    s = -1
                if s != 0:
                    b0 = b + 1
                    entrada = O[d, b0]
                    break
            if s == 0:
                continue
        else:  # ruptura (modo 0: cualquier lado; modo 2: solo el lado del color de la vela del rango)
            sc = 0
            if modo == 2:
                if C[d, n_or - 1] > O[d, 0]:
                    sc = 1
                elif C[d, n_or - 1] < O[d, 0]:
                    sc = -1
                else:
                    continue
                if lado != 0 and sc != lado:
                    continue
            for b in range(n_or, min(corte, u) + 1):
                arriba = H[d, b] > hi and lado != -1 and sc != -1
                abajo = L[d, b] < lo and lado != 1 and sc != 1
                if arriba and abajo:  # la vela rompe por los dos lados: se supone que toca primero el
                    # nivel más cercano a su apertura (y luego se evalúa el stop en esa misma vela)
                    if hi - O[d, b] <= O[d, b] - lo:
                        abajo = False
                    else:
                        arriba = False
                if arriba:
                    s = 1
                    entrada = max(hi, O[d, b])
                    b0 = b
                    break
                if abajo:
                    s = -1
                    entrada = min(lo, O[d, b])
                    b0 = b
                    break
            if s == 0:
                continue
        # stop
        if atr_stop > 0:
            dist = atr_stop * a
        else:
            if modo == 1:
                ref = lo if s == 1 else hi
                dist = abs(entrada - ref) * stop_frac
            elif modo == 3:  # stop a stop_frac × R desde el extremo roto (1 = el otro lado del rango)
                ref = hi - stop_frac * R if s == 1 else lo + stop_frac * R
                dist = (entrada - ref) * s
            else:
                dist = R * stop_frac
        if dist <= 0:
            continue
        stop = entrada - s * dist
        obj = entrada + s * target_R * dist if target_R > 0 else np.nan
        salida = np.nan
        m = 3
        for b in range(b0, u + 1):
            if s == 1:
                if L[d, b] <= stop:
                    salida = min(stop, O[d, b]) if b > b0 else stop
                    m = 1
                    break
                if target_R > 0 and H[d, b] >= obj:
                    salida = max(obj, O[d, b]) if b > b0 else obj
                    m = 2
                    break
            else:
                if H[d, b] >= stop:
                    salida = max(stop, O[d, b]) if b > b0 else stop
                    m = 1
                    break
                if target_R > 0 and L[d, b] <= obj:
                    salida = min(obj, O[d, b]) if b > b0 else obj
                    m = 2
                    break
        if m == 3:
            salida = C[d, u]
        dire[d] = s
        pe[d] = entrada
        ps[d] = salida
        rie[d] = dist
        ve[d] = b0
        mot[d] = m
    return dire, pe, ps, rie, ve, mot


def atr_diario(O, H, L, C, ultimo, n=14):
    """ATR de n días (true range de la sesión con el cierre anterior), conocido antes de abrir."""
    D = len(O)
    hi = np.nanmax(H, axis=1)
    lo = np.nanmin(L, axis=1)
    cl = C[np.arange(D), ultimo]
    prev = np.r_[np.nan, cl[:-1]]
    tr = np.fmax(hi - lo, np.fmax(np.abs(hi - prev), np.abs(lo - prev)))
    tr[0] = hi[0] - lo[0]
    import pandas as pd
    a = pd.Series(tr).rolling(n, min_periods=n).mean().shift(1).to_numpy()
    return a


def metricas(r: np.ndarray, fechas, anual=252) -> dict:
    """r: rentabilidad diaria de la cuenta (0 los días sin operar)."""
    r = np.nan_to_num(r)
    eq = np.cumprod(1 + r)
    años = len(r) / anual
    cagr = eq[-1] ** (1 / años) - 1 if eq[-1] > 0 else -1
    dd = (eq / np.maximum.accumulate(eq) - 1).min()
    sd = r.std() * np.sqrt(anual)
    sh = r.mean() / r.std() * np.sqrt(anual) if r.std() > 0 else 0
    return dict(cagr=cagr, vol=sd, sharpe=sh, maxdd=dd, calmar=cagr / -dd if dd < 0 else np.nan)
