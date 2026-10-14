"""Motor de cartera v2 (numba): tamaño por convicción, límite por grupo de pares y
cartera de pares largo/corto.

Largos (``simular``)
--------------------
Mismas reglas de ejecución que ``motor_rapido`` (señal al cierre de t, órdenes a la
apertura de t+1, deslistados liquidados al último precio, reserva opcional en SPY),
y además:

- **Tamaño por convicción**: cada candidata trae un peso ``w`` (1 = hueco normal de
  equity/K). El importe es ``w · equity/K`` limitado por el efectivo disponible; la
  suma nunca supera el 100 % del capital (sin apalancamiento).
- **Límite por grupo de pares** (``max_grupo``): una candidata no entra si ya hay
  ``max_grupo`` posiciones (abiertas o pendientes) que son de su grupo de 10 pares
  (en cualquier dirección: ella en el grupo de la otra o la otra en el suyo).

Pares (``simular_pares``)
-------------------------
Para cada evento (acción rezagada ``i``, gemela ``g``): largo ``i`` y corto ``g`` por el
mismo importe a la apertura de t+1. Se cierra cuando el log-cociente vuelve a su media
de 60 d medida en la entrada, o a las ``H`` sesiones. Coste ``coste_bps`` por pata y
lado, más ``prestamo_anual`` sobre la pata corta. Capital por par = equity · bruto / M.
"""

from __future__ import annotations

import numba
import numpy as np
import pandas as pd

import motor as m
import motor_rapido as mr


@numba.njit(cache=True)
def _en_grupo(a, b, pares_t):
    for q in range(pares_t.shape[1]):
        if pares_t[a, q] == b or pares_t[b, q] == a:
            return True
    return False


@numba.njit(cache=True)
def _simular(O, CF, C, ultimo, sal, ptr, cj, cw, K, H, coste, t0, t1, usa_res, r_noche, r_dia,
             cambio, pares, max_grupo):
    T, N = C.shape
    MAXR = 50000
    reg = np.full((MAXR, 6), np.nan)  # t_entrada, t_salida, j, px_entrada, px_salida, peso_cartera
    nreg = 0
    cash = 1.0
    u = np.zeros(N)
    te = np.full(N, -1)
    pe = np.zeros(N)
    abierta = np.zeros(N, dtype=np.bool_)
    pend_sal = np.zeros(N, dtype=np.bool_)
    pend = np.full(N, -1)
    pend_w = np.zeros(N)
    npend = 0
    eq = np.empty(t1 - t0)
    expo = np.empty(t1 - t0)
    n_tr = 0
    suma_ret = 0.0
    gan = 0
    abiertas = np.full(N, -1)
    peso_ent = np.zeros(N)
    eq_ref = 1.0
    for t in range(t0, t1):
        if usa_res and t > t0:
            cash *= r_noche[t] * (1 - cambio[t])
        cash_ini = cash
        for j in range(N):
            if pend_sal[j]:
                px = O[t, j] if not np.isnan(O[t, j]) else CF[t, j]
                cash += u[j] * px * (1 - coste)
                r = px / pe[j] - 1
                suma_ret += r
                gan += r > 0
                n_tr += 1
                if nreg < MAXR:
                    reg[nreg, 0], reg[nreg, 1], reg[nreg, 2] = te[j], t, j
                    reg[nreg, 3], reg[nreg, 4], reg[nreg, 5] = pe[j], px, peso_ent[j]
                    nreg += 1
                abierta[j] = False
                pend_sal[j] = False
                u[j] = 0.0
        if npend > 0:
            eq_ap = cash
            for j in range(N):
                if abierta[j]:
                    px = O[t, j] if not np.isnan(O[t, j]) else CF[t - 1, j]
                    eq_ap += u[j] * px
            hueco = eq_ap / K
            eq_ref = eq_ap
            for q in range(npend):
                j = pend[q]
                px = O[t, j]
                if np.isnan(px) or abierta[j]:
                    continue
                imp = min(hueco * pend_w[q], cash)
                if imp <= 1e-9:
                    break
                u[j] = imp * (1 - coste) / px
                cash -= imp
                abierta[j] = True
                te[j] = t
                pe[j] = px
                peso_ent[j] = imp / eq_ref
            npend = 0
        if usa_res:
            cash -= abs(cash - cash_ini) * 2e-4
            cash *= r_dia[t]
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
                    if nreg < MAXR:
                        reg[nreg, 0], reg[nreg, 1], reg[nreg, 2] = te[j], t, j
                        reg[nreg, 3], reg[nreg, 4], reg[nreg, 5] = pe[j], px, peso_ent[j]
                        nreg += 1
                    abierta[j] = False
                    u[j] = 0.0
                    continue
                valor += u[j] * CF[t, j]
                abiertas[nab] = j
                nab += 1
        equity = cash + valor
        eq[t - t0] = equity
        expo[t - t0] = valor / equity if equity > 0 else 0.0
        nsal = 0
        for q in range(nab):
            j = abiertas[q]
            if (t - te[j] + 1 >= H) or sal[t, j]:
                pend_sal[j] = True
                nsal += 1
        libres = K - (nab - nsal)
        if libres > 0 and t + 1 < t1:
            for q in range(ptr[t], ptr[t + 1]):
                j = cj[q]
                if abierta[j]:
                    continue
                if max_grupo > 0:
                    n_mismo = 0
                    for a in range(nab):  # posiciones que siguen abiertas
                        h = abiertas[a]
                        if not pend_sal[h] and _en_grupo(j, h, pares[t]):
                            n_mismo += 1
                    for a in range(npend):  # entradas ya elegidas hoy
                        if _en_grupo(j, pend[a], pares[t]):
                            n_mismo += 1
                    if n_mismo >= max_grupo:
                        continue
                pend[npend] = j
                pend_w[npend] = cw[q]
                npend += 1
                if npend >= libres:
                    break
    for j in range(N):  # posiciones abiertas al final (t_salida = -1)
        if abierta[j] and nreg < MAXR:
            reg[nreg, 0], reg[nreg, 1], reg[nreg, 2] = te[j], -1, j
            reg[nreg, 3], reg[nreg, 4], reg[nreg, 5] = pe[j], CF[t1 - 1, j], peso_ent[j]
            nreg += 1
    pendientes = pend[:npend].copy()
    return eq, expo, n_tr, suma_ret, gan, reg[:nreg], pendientes


_PARES: dict = {}


def _pares():
    if "pares" not in _PARES:
        import rasgos as rg
        _PARES["pares"] = np.load(rg.SALIDA)["pares"]
    return _PARES["pares"]


def simular(t, j, score, peso=None, K=3, H=20, salida="sma5", coste_bps=10.0, reserva="spy",
            max_grupo=0, inicio="2002-01-02", fin=None, detalle=False) -> dict:
    b = mr._base()
    p = b["p"]
    T = len(p.fechas)
    t = np.asarray(t)
    j = np.asarray(j)
    score = np.asarray(score, dtype=np.float64)
    peso = np.ones(len(t)) if peso is None else np.asarray(peso, dtype=np.float64)
    orden = np.lexsort((-score, t))
    t, j, peso = t[orden], j[orden].astype(np.int64), peso[orden]
    ptr = np.searchsorted(t, np.arange(T + 1)).astype(np.int64)
    t0 = int(np.searchsorted(p.fechas, np.datetime64(inicio)))
    t1 = T if fin is None else int(np.searchsorted(p.fechas, np.datetime64(fin)))
    rn, rd, cb = b["reservas"].get(reserva or "spy")
    eq, expo, n_tr, suma, gan, reg, pendientes = _simular(
        p.open, p.close_ff, p.close, p.ultimo, b["salidas"][salida], ptr, j, peso, K, H, coste_bps / 1e4,
        t0, t1, bool(reserva), rn, rd, cb, _pares(), max_grupo)
    idx = pd.DatetimeIndex(p.fechas[t0:t1])
    eq = pd.Series(eq, idx)
    out = m.metricas(eq)
    años = (idx[-1] - idx[0]).days / 365.25
    out |= {"trades_año": n_tr / años, "acierto": gan / max(n_tr, 1),
            "ret_medio_trade": suma / max(n_tr, 1), "exposicion": float(expo.mean()), "equity": eq}
    if detalle:
        ops = pd.DataFrame(reg, columns=["t_ent", "t_sal", "j", "px_ent", "px_sal", "peso"])
        ops["ticker"] = p.tickers[ops.j.astype(int)]
        ops["entrada"] = p.fechas[ops.t_ent.astype(int)]
        ops["salida"] = [p.fechas[int(x)] if x >= 0 else pd.NaT for x in ops.t_sal]
        ops["ret"] = ops.px_sal / ops.px_ent - 1
        out["operaciones"] = ops
        out["pendientes"] = [p.tickers[x] for x in pendientes]
    return out


# ----------------------------------------------------------------------- pares L/S
@numba.njit(cache=True)
def _pares_ls(O, CF, L, ptr, ci, cg, cmu, M, H, bruto, coste, prestamo_dia, t0, t1, ultimo):
    T, N = CF.shape
    cash = 1.0
    li = np.full(M, -1)
    lg = np.full(M, -1)
    ul = np.zeros(M)  # unidades largas
    us = np.zeros(M)  # unidades cortas
    ei = np.zeros(M)  # precios de entrada
    eg = np.zeros(M)
    mu = np.zeros(M)
    tin = np.zeros(M, dtype=np.int64)
    pend_sal = np.zeros(M, dtype=np.bool_)
    pend_i = np.full(M, -1)
    pend_g = np.full(M, -1)
    pend_mu = np.zeros(M)
    npend = 0
    eq = np.empty(t1 - t0)
    n_tr = 0
    suma = 0.0
    gan = 0
    for t in range(t0, t1):
        # cierres a la apertura (largo: vender; corto: recomprar)
        for s in range(M):
            if li[s] >= 0 and pend_sal[s]:
                pi = O[t, li[s]] if not np.isnan(O[t, li[s]]) else CF[t, li[s]]
                pg = O[t, lg[s]] if not np.isnan(O[t, lg[s]]) else CF[t, lg[s]]
                cash += ul[s] * pi * (1 - coste) - us[s] * pg * (1 + coste)
                r = (pi / ei[s] - 1) - (pg / eg[s] - 1)
                suma += r
                gan += r > 0
                n_tr += 1
                li[s] = -1
                lg[s] = -1
                pend_sal[s] = False
        # aperturas: importe por pata = equity · bruto / M (el corto financia el largo)
        if npend > 0:
            eq_ap = cash
            for s in range(M):
                if li[s] >= 0:
                    eq_ap += ul[s] * CF[t - 1, li[s]] - us[s] * CF[t - 1, lg[s]]
            imp = eq_ap * bruto / M
            for q in range(npend):
                i, g = pend_i[q], pend_g[q]
                pi, pg = O[t, i], O[t, g]
                if np.isnan(pi) or np.isnan(pg):
                    continue
                for s in range(M):
                    if li[s] < 0:
                        li[s], lg[s] = i, g
                        ul[s] = imp / pi
                        us[s] = imp / pg
                        ei[s], eg[s] = pi, pg
                        cash -= imp * coste * 2
                        mu[s] = pend_mu[q]
                        tin[s] = t
                        break
            npend = 0
        # valoración al cierre y coste de préstamo de la pata corta
        valor = 0.0
        for s in range(M):
            if li[s] >= 0:
                valor += ul[s] * CF[t, li[s]] - us[s] * CF[t, lg[s]]
                cash -= us[s] * CF[t, lg[s]] * prestamo_dia
        eq[t - t0] = cash + valor
        # salidas: el log-cociente vuelve a su media de entrada, o tiempo, o fin de serie
        ocupados = 0
        n_sal = 0
        for s in range(M):
            if li[s] >= 0:
                ocupados += 1
                d = L[t, li[s]] - L[t, lg[s]]
                fin_serie = ultimo[li[s]] <= t or ultimo[lg[s]] <= t
                if d >= mu[s] or t - tin[s] + 1 >= H or fin_serie:
                    pend_sal[s] = True
                if pend_sal[s]:
                    n_sal += 1
        libres = M - (ocupados - n_sal)
        if libres > 0 and t + 1 < t1:
            for q in range(ptr[t], ptr[t + 1]):
                i, g = ci[q], cg[q]
                dup = False
                for s in range(M):
                    if li[s] >= 0 and not pend_sal[s]:
                        if li[s] == i or lg[s] == i or li[s] == g or lg[s] == g:
                            dup = True
                for a in range(npend):
                    if pend_i[a] == i or pend_g[a] == g or pend_i[a] == g or pend_g[a] == i:
                        dup = True
                if dup:
                    continue
                pend_i[npend], pend_g[npend], pend_mu[npend] = i, g, cmu[q]
                npend += 1
                if npend >= libres:
                    break
    return eq, n_tr, suma, gan


def simular_pares(corr_min=0.8, z_max=-2.5, M=10, H=20, bruto=1.0, coste_bps=10.0, prestamo_anual=0.005,
                  inicio="2002-01-02", fin=None, tendencia=False) -> dict:
    import rasgos as rg
    p = m.cargar()
    R = np.load(rg.SALIDA)
    gem = R["pares"][:, :, 0]
    gz, gc = R["gem_z"], R["gem_corr"]
    liq = m.liquido()
    with np.errstate(invalid="ignore"):
        ev = p.miembro & liq & (gc >= corr_min) & (gz <= z_max) & (gem >= 0)
        if tendencia:
            ev &= R["sobre_sma200"] == 1
    t, i = np.nonzero(ev)
    g = gem[t, i]
    ok = p.miembro[t, g] & liq[t, g]
    t, i, g = t[ok], i[ok], g[ok]
    L = np.log(p.close_ff)
    mu = np.empty(len(t))  # media de 60 d del log-cociente con la gemela vigente en t
    for k in range(len(t)):
        a = max(0, t[k] - 59)
        mu[k] = np.nanmean(L[a : t[k] + 1, i[k]] - L[a : t[k] + 1, g[k]])
    score = -gz[t, i]
    orden = np.lexsort((-score, t))
    t, i, g, mu = t[orden], i[orden].astype(np.int64), g[orden].astype(np.int64), mu[orden]
    T = len(p.fechas)
    ptr = np.searchsorted(t, np.arange(T + 1)).astype(np.int64)
    t0 = int(np.searchsorted(p.fechas, np.datetime64(inicio)))
    t1 = T if fin is None else int(np.searchsorted(p.fechas, np.datetime64(fin)))
    eq, n_tr, suma, gan = _pares_ls(p.open, p.close_ff, L, ptr, i, g, mu, M, H, bruto, coste_bps / 1e4,
                                    prestamo_anual / 252, t0, t1, p.ultimo)
    idx = pd.DatetimeIndex(p.fechas[t0:t1])
    eq = pd.Series(eq, idx)
    out = m.metricas(eq)
    out |= {"trades_año": n_tr / ((idx[-1] - idx[0]).days / 365.25), "acierto": gan / max(n_tr, 1),
            "ret_medio_trade": suma / max(n_tr, 1), "equity": eq}
    return out
