"""Motor de carteras compilado (numba), equivalente a ``motor.backtest``.

Recibe las entradas como lista dispersa de candidatas por sesión (CSR: ``ptr``,
``cj``, ``cs``) en lugar de matrices booleanas, de modo que cualquier filtro sobre
la tabla de eventos se convierte en estrategia sin recalcular nada. Mismas reglas
que el motor de referencia: señal al cierre de t, ejecución a la apertura de t+1,
tamaño = equity a la apertura / K, salida por condición al cierre o por tiempo,
liquidación de deslistados, reserva opcional en SPY.

``validar()`` comprueba que da el mismo equity que ``motor.backtest``.
"""

from __future__ import annotations

import numba
import numpy as np
import pandas as pd

import motor as m


@numba.njit(cache=True)
def _simular(O, CF, C, ultimo, sal, ptr, cj, cs, K, H, coste, t0, t1, usa_res, r_noche, r_dia, cambio):
    T, N = C.shape
    cash = 1.0
    u = np.zeros(N)
    te = np.full(N, -1)
    pe = np.zeros(N)
    abierta = np.zeros(N, dtype=np.bool_)
    pend_sal = np.zeros(N, dtype=np.bool_)
    pend = np.full(N, -1)
    npend = 0
    eq = np.empty(t1 - t0)
    expo = np.empty(t1 - t0)
    n_tr = 0
    suma_ret = 0.0
    gan = 0
    for t in range(t0, t1):
        if usa_res and t > t0:
            cash *= r_noche[t] * (1 - cambio[t])
        cash_ini = cash
        # salidas a la apertura
        for j in range(N):
            if pend_sal[j]:
                px = O[t, j] if not np.isnan(O[t, j]) else CF[t, j]
                cash += u[j] * px * (1 - coste)
                r = px / pe[j] - 1
                suma_ret += r
                gan += r > 0
                n_tr += 1
                abierta[j] = False
                pend_sal[j] = False
                u[j] = 0.0
        # entradas a la apertura
        if npend > 0:
            eq_ap = cash
            for j in range(N):
                if abierta[j]:
                    px = O[t, j] if not np.isnan(O[t, j]) else CF[t - 1, j]
                    eq_ap += u[j] * px
            hueco = eq_ap / K
            for q in range(npend):
                j = pend[q]
                px = O[t, j]
                if np.isnan(px) or abierta[j]:
                    continue
                imp = min(hueco, cash)
                if imp <= 1e-9:
                    break
                u[j] = imp * (1 - coste) / px
                cash -= imp
                abierta[j] = True
                te[j] = t
                pe[j] = px
            npend = 0
        if usa_res:
            cash -= abs(cash - cash_ini) * 2e-4
            cash *= r_dia[t]
        # cierre: deslistados y valoración
        valor = 0.0
        nab = 0
        for j in range(N):
            if abierta[j]:
                if ultimo[j] <= t and ultimo[j] < T - 1:
                    px = CF[t, j]
                    cash += u[j] * px * (1 - coste)
                    r = px / pe[j] - 1
                    suma_ret += r
                    gan += r > 0
                    n_tr += 1
                    abierta[j] = False
                    u[j] = 0.0
                    continue
                valor += u[j] * CF[t, j]
                nab += 1
        equity = cash + valor
        eq[t - t0] = equity
        expo[t - t0] = valor / equity if equity > 0 else 0.0
        # señales de salida
        nsal = 0
        for j in range(N):
            if abierta[j]:
                if (t - te[j] + 1 >= H) or sal[t, j]:
                    pend_sal[j] = True
                    nsal += 1
        # señales de entrada (candidatas ya ordenadas por puntuación desc.)
        libres = K - (nab - nsal)
        if libres > 0 and t + 1 < t1:
            for q in range(ptr[t], ptr[t + 1]):
                j = cj[q]
                if abierta[j]:
                    continue
                pend[npend] = j
                npend += 1
                if npend >= libres:
                    break
    return eq, expo, n_tr, suma_ret, gan


_CACHE: dict = {}


def _base():
    if "p" not in _CACHE:
        p = m.cargar()
        rj = p.col["SPY"]
        O, CF = p.open, p.close_ff
        r_noche = np.nan_to_num(O[:, rj] / np.roll(CF[:, rj], 1), nan=1.0)
        r_noche[0] = 1.0
        r_dia = np.nan_to_num(CF[:, rj] / O[:, rj], nan=1.0)
        # régimen: SPY por encima de su SMA200 al cierre de t-1 -> reserva en SPY; si no, efectivo
        spy = pd.Series(CF[:, rj])
        reg = np.roll((spy > spy.rolling(200).mean()).to_numpy(), 1)
        reg[0] = False
        cambio_reg = np.r_[0.0, (reg[1:] != reg[:-1]) * 2e-4]
        _CACHE.update(p=p, r_noche=r_noche, r_dia=r_dia,
                      reservas={"spy": (r_noche, r_dia, np.zeros(len(reg))),
                                "spy_tend": (np.where(reg, r_noche, 1.0), np.where(reg, r_dia, 1.0), cambio_reg)},
                      salidas={"sma5": np.nan_to_num(p.close > m.sma(5)).astype(bool),
                               "sma20": np.nan_to_num(p.close > m.sma(20)).astype(bool),
                               "sma10": np.nan_to_num(p.close > m.sma(10)).astype(bool),
                               "rsi2": np.nan_to_num(m.rsi(2) >= 70).astype(bool),
                               "tiempo": np.zeros(p.close.shape, dtype=bool)})
    return _CACHE


def candidatas(t: np.ndarray, j: np.ndarray, score: np.ndarray, T: int):
    """Lista dispersa por sesión, ordenada por puntuación descendente."""
    orden = np.lexsort((-score, t))
    t, j = t[orden], j[orden]
    ptr = np.searchsorted(t, np.arange(T + 1)).astype(np.int64)
    return ptr, j.astype(np.int64), score[orden]


def simular(t, j, score, K=3, H=20, salida="sma5", coste_bps=10.0, reserva="spy",
            inicio="2002-01-02", fin=None) -> dict:
    b = _base()
    p = b["p"]
    T = len(p.fechas)
    ptr, cj, cs = candidatas(np.asarray(t), np.asarray(j), np.asarray(score, dtype=np.float64), T)
    t0 = int(np.searchsorted(p.fechas, np.datetime64(inicio)))
    t1 = T if fin is None else int(np.searchsorted(p.fechas, np.datetime64(fin)))
    if reserva is True:
        reserva = "spy"
    rn, rd, cb = b["reservas"].get(reserva or "spy")
    eq, expo, n_tr, suma, gan = _simular(p.open, p.close_ff, p.close, p.ultimo, b["salidas"][salida],
                                         ptr, cj, cs, K, H, coste_bps / 1e4, t0, t1, bool(reserva),
                                         rn, rd, cb)
    idx = pd.DatetimeIndex(p.fechas[t0:t1])
    eq = pd.Series(eq, idx)
    out = m.metricas(eq)
    años = (idx[-1] - idx[0]).days / 365.25
    out |= {"trades_año": n_tr / años, "acierto": gan / max(n_tr, 1),
            "ret_medio_trade": suma / max(n_tr, 1), "exposicion": float(expo.mean()), "equity": eq}
    return out


def validar() -> None:
    """Mismo resultado que el motor de referencia para una configuración real."""
    import time
    p = m.cargar()
    cfg = dict(tipo="ret", n=10, x=0.15, filtro="tendencia", salida="sma5", h=20, k=3, reserva="SPY")
    ent, score = m.senal(cfg)
    t, j = np.nonzero(ent)
    ref = m.backtest(cfg)
    simular(t, j, score[t, j], K=3, H=20)  # compila
    a = time.time()
    r = simular(t, j, score[t, j], K=3, H=20)
    print(f"numba {time.time() - a:.4f}s   ref equity {ref.equity.iloc[-1]:.6f}   numba {r['equity'].iloc[-1]:.6f}")
    assert abs(ref.equity.iloc[-1] / r["equity"].iloc[-1] - 1) < 1e-9
    del p


if __name__ == "__main__":
    validar()
