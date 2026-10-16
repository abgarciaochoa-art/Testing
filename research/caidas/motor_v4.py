"""Motor v4 (numba): el motor v3 con entrada ESCALONADA y más formas de salir.

Mismas reglas que el v3 (señal al cierre, órdenes a la apertura siguiente, reserva,
deuda, tope de exposición real y mantenimiento de margen). Novedades, todas apagadas
por defecto (con ellas apagadas reproduce el v3 al céntimo):

- **Entrada en dos tramos**: se compra ``w_ini`` del tamaño previsto en la apertura y el
  resto en la apertura siguiente a la sesión ``add_dia`` tras la compra (0 = decidido
  al cierre del día de compra), si se cumple la condición ``add_modo``:
  0 siempre; 1 si el cierre está ``add_umbral`` por DEBAJO del precio medio (promediar
  a la baja); 2 si está ``add_umbral`` por ENCIMA (confirmación); 3 si ``add_cond[t, j]``.
  Si no se cumple ese día, ya no se añade.
- **Salidas por precio** sobre el precio medio de compra:
  al cierre (se vende en la apertura siguiente): objetivo ``obj_c``, stop ``stop_c`` y
  trailing ``trail_c`` desde el máximo cierre; intradía (órdenes límite/stop desde la
  sesión de compra; si la apertura ya las supera, se ejecutan a la apertura):
  objetivo ``obj_i`` y stop ``stop_i``. Si el mismo día se tocan los dos, cuenta el stop.
- **Venta parcial**: al tocar el objetivo se vende ``parcial`` (1 = todo); el resto sigue
  con la salida normal.
- **Añadir según un calendario por candidata** (``add_t``, ``add_f``): en la apertura
  indicada se compra ``add_f`` veces el tamaño previsto, con tope opcional de peso por
  posición (``tope_pos``); y vender según otro calendario (``sal_t``). Lo usa la ronda 8
  (``ronda8.py``).

Registro por posición: retorno NETO de costes con todos los flujos (tramos y parciales).
"""

from __future__ import annotations

import numba
import numpy as np
import pandas as pd

import motor as m
import motor_rapido as mr
import motor_v3 as v3

INF = np.inf


@numba.njit(cache=True)
def _vender(cash, deuda, ingreso):
    pago = min(deuda, ingreso)
    return cash + ingreso - pago, deuda - pago, ingreso - pago


@numba.njit(cache=True)
def _simular(O, Hi, Lo, CF, C, ultimo, sal, ptr, cj, cw, K, frac, H, coste, t0, t1, rn, rd, cambio, activa,
             expo_res, coste_res, pal_t, tasa_dia, max_t2, w_ini, add_modo, add_umbral, add_dia, add_cond,
             obj_c, stop_c, trail_c, obj_i, stop_i, parcial, c_add_t, c_add_f, tope_pos, c_sal_t):
    T, N = C.shape
    MAXR = 60000
    # t_ent, t_sal, j, px_medio, px_salida, peso previsto, ret neto, tramos, parcial hecho
    reg = np.full((MAXR, 9), np.nan)
    nreg = 0
    cash = 1.0
    deuda = 0.0
    u = np.zeros(N)
    te = np.full(N, -1)
    pe = np.zeros(N)
    salida_ = np.zeros(N)  # dinero invertido (incluidos costes) y cobrado por posición
    cobro = np.zeros(N)
    previsto = np.zeros(N)
    peso_prev = np.zeros(N)
    tramos = np.zeros(N, dtype=np.int64)
    decidido = np.zeros(N, dtype=np.bool_)
    hecho_parcial = np.zeros(N, dtype=np.bool_)
    maxc = np.zeros(N)
    abierta = np.zeros(N, dtype=np.bool_)
    pend_sal = np.zeros(N, dtype=np.bool_)
    pend_par = np.zeros(N, dtype=np.bool_)
    pend_add = np.zeros(N, dtype=np.bool_)
    frac_add = np.zeros(N)  # fracción del tamaño previsto que se compra al añadir
    add_t_pos = np.full(N, -1)  # apertura en la que toca añadir según el calendario de la candidata
    add_f_pos = np.zeros(N)
    sal_t_pos = np.full(N, -1)  # apertura en la que toca vender según el calendario de la candidata
    pend_q = np.full(N, -1)
    pend = np.full(N, -1)
    pend_w = np.zeros(N)
    npend = 0
    eq = np.empty(t1 - t0)
    expo = np.empty(t1 - t0)
    lev = np.empty(t1 - t0)
    caja = np.empty(t1 - t0)
    deu = np.empty(t1 - t0)
    npos = np.zeros(t1 - t0, dtype=np.int64)
    n_tr = 0
    suma_ret = 0.0
    gan = 0
    abiertas = np.full(N, -1)
    ruina = False
    for t in range(t0, t1):
        if ruina:
            eq[t - t0] = 0.0
            expo[t - t0] = 0.0
            lev[t - t0] = 0.0
            caja[t - t0] = 0.0
            deu[t - t0] = 0.0
            continue
        if t > t0:
            cash *= rn[t] * (1 - cambio[t])
            deuda *= 1 + tasa_dia
        mov_res = 0.0
        # 1) ventas a la apertura (totales y parciales decididas al cierre anterior)
        for j in range(N):
            if pend_sal[j] or pend_par[j]:
                px = O[t, j] if not np.isnan(O[t, j]) else CF[t, j]
                f = 1.0 if pend_sal[j] else parcial
                ingreso = u[j] * f * px * (1 - coste)
                cash, deuda, neto = _vender(cash, deuda, ingreso)
                mov_res += neto
                cobro[j] += ingreso
                u[j] *= 1 - f
                if pend_par[j] and not pend_sal[j]:
                    hecho_parcial[j] = True
                    pend_par[j] = False
                    continue
                r = cobro[j] / salida_[j] - 1
                suma_ret += r
                gan += r > 0
                n_tr += 1
                if nreg < MAXR:
                    reg[nreg, 0], reg[nreg, 1], reg[nreg, 2] = te[j], t, j
                    reg[nreg, 3], reg[nreg, 4], reg[nreg, 5] = pe[j], px, peso_prev[j]
                    reg[nreg, 6], reg[nreg, 7], reg[nreg, 8] = r, tramos[j], hecho_parcial[j]
                    nreg += 1
                abierta[j] = False
                pend_sal[j] = False
                pend_par[j] = False
                u[j] = 0.0
        # 2) mantenimiento de margen (igual que el v3)
        if deuda > 0:
            valor_m = 0.0
            for j in range(N):
                if abierta[j]:
                    px = O[t, j] if not np.isnan(O[t, j]) else CF[t - 1, j]
                    valor_m += u[j] * px
            eq_m = cash + valor_m - deuda
            bruto_m = valor_m + cash * expo_res[t]
            if eq_m > 0 and bruto_m > 1.2 * pal_t[t] * eq_m and expo_res[t] > 0:
                exceso = bruto_m - pal_t[t] * eq_m
                vender = min(cash, exceso / expo_res[t], deuda)
                cash -= vender
                deuda -= vender
                mov_res -= vender
        # 3) compras a la apertura: segundos tramos y entradas nuevas
        hay_add = False
        for j in range(N):
            if pend_add[j]:
                hay_add = True
                break
        if npend > 0 or hay_add:
            valor_ap = 0.0
            for j in range(N):
                if abierta[j]:
                    px = O[t, j] if not np.isnan(O[t, j]) else CF[t - 1, j]
                    valor_ap += u[j] * px
            eq_ap = cash + valor_ap - deuda
            palanca = pal_t[t]
            for paso in range(2):
                nq = N if paso == 0 else npend
                for q in range(nq):
                    if paso == 0:
                        j = q
                        if not pend_add[j]:
                            continue
                        pend_add[j] = False
                        px = O[t, j]
                        if np.isnan(px) or not abierta[j] or eq_ap <= 0:
                            continue
                        imp = previsto[j] * frac_add[j]
                        if tope_pos < INF:  # tope de peso por posición
                            imp = min(imp, max(0.0, tope_pos * eq_ap - u[j] * px))
                    else:
                        j = pend[q]
                        px = O[t, j]
                        if np.isnan(px) or abierta[j] or eq_ap <= 0:
                            continue
                        imp = eq_ap * frac * pend_w[q]
                        previsto[j] = imp
                        peso_prev[j] = imp / eq_ap
                        imp = imp * w_ini
                    bruto_res = cash * expo_res[t]
                    margen = max(0.0, palanca * eq_ap - (valor_ap + bruto_res))
                    if activa[t]:
                        d = min(imp, margen)
                        v = min(imp - d, cash)
                    else:
                        v = min(imp, cash)
                        d = min(imp - v, max(0.0, palanca * eq_ap - valor_ap - v - (cash - v) * expo_res[t]))
                    imp = v + d
                    if imp <= 1e-9:
                        continue
                    deuda += d
                    cash -= v
                    mov_res -= v
                    unidades = imp * (1 - coste) / px
                    if paso == 0:
                        pe[j] = (pe[j] * u[j] + px * unidades) / (u[j] + unidades)
                        u[j] += unidades
                        salida_[j] += imp
                        tramos[j] = 2
                    else:
                        u[j] = unidades
                        abierta[j] = True
                        te[j] = t
                        pe[j] = px
                        salida_[j] = imp
                        cobro[j] = 0.0
                        tramos[j] = 1
                        decidido[j] = w_ini >= 1.0
                        hecho_parcial[j] = False
                        maxc[j] = px
                        add_t_pos[j] = c_add_t[pend_q[q]]
                        add_f_pos[j] = c_add_f[pend_q[q]]
                        sal_t_pos[j] = c_sal_t[pend_q[q]]
                    valor_ap += imp
            npend = 0
        cash -= abs(mov_res) * coste_res
        cash *= rd[t]
        # 4) órdenes intradía (límite de objetivo y stop) desde la sesión de compra
        if obj_i < INF or stop_i < INF:
            for j in range(N):
                if not abierta[j] or u[j] <= 0 or np.isnan(C[t, j]):
                    continue
                o = O[t, j] if not np.isnan(O[t, j]) else C[t, j]
                hi = Hi[t, j] if not np.isnan(Hi[t, j]) else C[t, j]
                lo = Lo[t, j] if not np.isnan(Lo[t, j]) else C[t, j]
                nivel_s = pe[j] * (1 - stop_i)
                nivel_o = pe[j] * (1 + obj_i)
                f = 0.0
                px = 0.0
                if stop_i < INF and lo <= nivel_s:
                    f = 1.0
                    px = min(o, nivel_s) if te[j] < t else nivel_s
                elif obj_i < INF and hi >= nivel_o and not hecho_parcial[j]:
                    f = parcial
                    px = max(o, nivel_o) if te[j] < t else nivel_o
                if f <= 0.0:
                    continue
                ingreso = u[j] * f * px * (1 - coste)
                cash, deuda, neto = _vender(cash, deuda, ingreso)
                cash -= neto * coste_res  # lo cobrado vuelve a la reserva
                cobro[j] += ingreso
                u[j] *= 1 - f
                if f < 1.0:
                    hecho_parcial[j] = True
                    continue
                r = cobro[j] / salida_[j] - 1
                suma_ret += r
                gan += r > 0
                n_tr += 1
                if nreg < MAXR:
                    reg[nreg, 0], reg[nreg, 1], reg[nreg, 2] = te[j], t, j
                    reg[nreg, 3], reg[nreg, 4], reg[nreg, 5] = pe[j], px, peso_prev[j]
                    reg[nreg, 6], reg[nreg, 7], reg[nreg, 8] = r, tramos[j], hecho_parcial[j]
                    nreg += 1
                abierta[j] = False
                pend_sal[j] = False
                pend_par[j] = False
                pend_add[j] = False
                u[j] = 0.0
        # 5) cierre: deslistados y valoración
        valor = 0.0
        nab = 0
        for j in range(N):
            if abierta[j]:
                if ultimo[j] <= t and ultimo[j] < T - 1:
                    ingreso = u[j] * CF[t, j] * (1 - coste)
                    cash, deuda, neto = _vender(cash, deuda, ingreso)
                    cobro[j] += ingreso
                    r = cobro[j] / salida_[j] - 1
                    suma_ret += r
                    gan += r > 0
                    n_tr += 1
                    if nreg < MAXR:
                        reg[nreg, 0], reg[nreg, 1], reg[nreg, 2] = te[j], t, j
                        reg[nreg, 3], reg[nreg, 4], reg[nreg, 5] = pe[j], CF[t, j], peso_prev[j]
                        reg[nreg, 6], reg[nreg, 7], reg[nreg, 8] = r, tramos[j], hecho_parcial[j]
                        nreg += 1
                    abierta[j] = False
                    pend_add[j] = False
                    u[j] = 0.0
                    continue
                valor += u[j] * CF[t, j]
                abiertas[nab] = j
                nab += 1
        equity = cash + valor - deuda
        if equity <= 0:
            ruina = True
            equity = 0.0
        eq[t - t0] = equity
        expo[t - t0] = valor / equity if equity > 0 else 0.0
        lev[t - t0] = (valor + cash * expo_res[t]) / equity if equity > 0 else 0.0
        caja[t - t0] = cash
        deu[t - t0] = deuda
        npos[t - t0] = nab
        # 6) señales al cierre: salidas, ventas parciales y segundos tramos
        nsal = 0
        for q in range(nab):
            j = abiertas[q]
            c = CF[t, j]
            maxc[j] = max(maxc[j], c)
            rc = c / pe[j] - 1
            if (t - te[j] + 1 >= H) or sal[t, j] or rc <= -stop_c or c <= maxc[j] * (1 - trail_c) or (
                    sal_t_pos[j] == t + 1):
                pend_sal[j] = True
                pend_add[j] = False
                nsal += 1
                continue
            if obj_c < INF and rc >= obj_c and not hecho_parcial[j]:
                if parcial >= 1.0:
                    pend_sal[j] = True
                    pend_add[j] = False
                    nsal += 1
                    continue
                pend_par[j] = True
            if not decidido[j] and t - te[j] == add_dia:
                decidido[j] = True
                ok = add_modo == 0
                if add_modo == 1:
                    ok = rc <= -add_umbral
                elif add_modo == 2:
                    ok = rc >= add_umbral
                elif add_modo == 3:
                    ok = add_cond[t, j]
                pend_add[j] = ok
                frac_add[j] = 1 - w_ini
            if add_t_pos[j] == t + 1:  # calendario de la candidata: añadir en la próxima apertura
                pend_add[j] = True
                frac_add[j] = add_f_pos[j]
                add_t_pos[j] = -1
        libres = K - (nab - nsal)
        if libres > 0 and t + 1 < t1:
            for q in range(ptr[t], ptr[t + 1]):
                j = cj[q]
                if abierta[j]:
                    continue
                w_q = cw[q]
                if w_q < 0:
                    if (nab - nsal) + npend >= max_t2:
                        continue
                    w_q = -w_q
                pend[npend] = j
                pend_w[npend] = w_q
                pend_q[npend] = q
                npend += 1
                if npend >= libres:
                    break
    for j in range(N):
        if abierta[j] and nreg < MAXR:
            r = (cobro[j] + u[j] * CF[t1 - 1, j] * (1 - coste)) / salida_[j] - 1
            reg[nreg, 0], reg[nreg, 1], reg[nreg, 2] = te[j], -1, j
            reg[nreg, 3], reg[nreg, 4], reg[nreg, 5] = pe[j], CF[t1 - 1, j], peso_prev[j]
            reg[nreg, 6], reg[nreg, 7], reg[nreg, 8] = r, tramos[j], hecho_parcial[j]
            nreg += 1
    return eq, expo, lev, n_tr, suma_ret, gan, reg[:nreg], pend[:npend].copy(), deuda, caja, deu, npos


def simular(t, j, score, peso=None, K=3, frac=None, H=20, salida="sma10", coste_bps=10.0, reserva_="spy",
            palanca=1.0, palanca_off=None, vol_objetivo=None, tasa_deuda=0.05, inicio="2002-01-02", fin=None,
            detalle=False, max_t2=0, coste_reserva=None, sal_matriz=None, w_ini=1.0, add_modo=0, add_umbral=0.0,
            add_dia=0, add_cond=None, obj_c=INF, stop_c=INF, trail_c=INF, obj_i=INF, stop_i=INF,
            parcial=1.0, add_t=None, add_f=None, tope_pos=INF, sal_t=None) -> dict:
    """Como ``motor_v3.simular`` más los parámetros de tramos y salidas del docstring del
    módulo. ``sal_matriz`` sustituye a la salida por nombre (matriz T×N de cierres que
    disparan la venta). ``add_t`` (por candidata): sesión en cuya APERTURA se añade a la
    posición ``add_f`` veces su tamaño previsto (-1 = no añadir), con un tope opcional
    del peso de la posición sobre el capital (``tope_pos``). ``sal_t`` (por candidata):
    sesión en cuya apertura se vende si sigue abierta (-1 = nunca)."""
    b = mr._base()
    p = b["p"]
    T = len(p.fechas)
    t = np.asarray(t)
    j = np.asarray(j)
    score = np.asarray(score, dtype=np.float64)
    peso = np.ones(len(t)) if peso is None else np.asarray(peso, dtype=np.float64)
    add_t = np.full(len(t), -1, dtype=np.int64) if add_t is None else np.asarray(add_t, dtype=np.int64)
    add_f = np.broadcast_to(np.asarray(0.0 if add_f is None else add_f, dtype=np.float64), (len(t),))
    orden = np.lexsort((-score, t))
    t, j, peso = t[orden], j[orden].astype(np.int64), peso[orden]
    sal_t = np.full(len(t), -1, dtype=np.int64) if sal_t is None else np.asarray(sal_t, dtype=np.int64)
    add_t, add_f, sal_t = add_t[orden].copy(), add_f[orden].copy(), sal_t[orden].copy()
    ptr = np.searchsorted(t, np.arange(T + 1)).astype(np.int64)
    t0 = int(np.searchsorted(p.fechas, np.datetime64(inicio)))
    t1 = T if fin is None else int(np.searchsorted(p.fechas, np.datetime64(fin)))
    rn, rd, cb, act, ex = v3.reserva(reserva_)
    coste_res = 10e-4 if reserva_.startswith(("mom", "ew", "liq")) else 2e-4
    if coste_reserva is not None:
        coste_res = coste_reserva
    sal = b["salidas"][salida] if sal_matriz is None else sal_matriz
    cond = np.zeros((1, 1), dtype=np.bool_) if add_cond is None else add_cond
    eq, expo, lev, n_tr, suma, gan, reg, pendientes, deuda, caja, deu, npos = _simular(
        p.open, p.high, p.low, p.close_ff, p.close, p.ultimo, sal, ptr, j, peso, K,
        (1.0 / K) if frac is None else frac, H, coste_bps / 1e4, t0, t1, rn, rd, cb, act, ex, coste_res,
        v3.vector_palanca(palanca, palanca_off, vol_objetivo), (1 + tasa_deuda) ** (1 / 252) - 1, max_t2,
        float(w_ini), int(add_modo), float(add_umbral), int(add_dia), cond, float(obj_c), float(stop_c),
        float(trail_c), float(obj_i), float(stop_i), float(parcial), add_t, add_f, float(tope_pos), sal_t)
    idx = pd.DatetimeIndex(p.fechas[t0:t1])
    eq = pd.Series(eq, idx)
    out = m.metricas(eq) if eq.iloc[-1] > 0 else {"cagr": -1.0, "vol": np.nan, "sharpe": np.nan,
                                                  "maxdd": -1.0, "calmar": np.nan}
    años = (idx[-1] - idx[0]).days / 365.25
    out |= {"trades_año": n_tr / años, "acierto": gan / max(n_tr, 1), "ret_medio_trade": suma / max(n_tr, 1),
            "exposicion": float(expo.mean()), "palanca_media": float(lev.mean()), "palanca_max": float(lev.max()),
            "equity": eq, "serie_expo": pd.Series(expo, idx), "serie_lev": pd.Series(lev, idx),
            "serie_reserva": pd.Series(caja, idx), "serie_deuda": pd.Series(deu, idx),
            "serie_posiciones": pd.Series(npos, idx)}
    if detalle:
        ops = pd.DataFrame(reg, columns=["t_ent", "t_sal", "j", "px_ent", "px_sal", "peso", "ret", "tramos",
                                         "parcial"])
        ops["ticker"] = p.tickers[ops.j.astype(int)]
        ops["entrada"] = p.fechas[ops.t_ent.astype(int)]
        ops["salida"] = [p.fechas[int(x)] if x >= 0 else pd.NaT for x in ops.t_sal]
        out |= {"operaciones": ops, "pendientes": [p.tickers[x] for x in pendientes],
                "deuda_final": float(deuda), "exposicion_final": float(expo[-1]), "palanca_final": float(lev[-1])}
    return out


def validar() -> None:
    """Con todo lo nuevo apagado, el v4 debe dar exactamente el mismo equity que el v3."""
    from definitiva import D1_CFG
    from ronda5 import componentes_tec
    from ronda5b import preparar
    x, salida, conv = componentes_tec()["D1"]
    y, w = preparar(x, conv, D1_CFG["kmin"], D1_CFG["uso"], D1_CFG["tam"], D1_CFG["frac"])
    for res, pal in (("liq100_vg_minvol", 1.0), ("liq100_vg_minvol", 2.0), ("spy_tend", 1.0)):
        kw = dict(peso=w, K=5, frac=0.5, H=20, salida=salida, reserva_=res, palanca=pal, palanca_off=1.0,
                  vol_objetivo=0.15 if pal > 1 else None)
        a = v3.simular(y.t.to_numpy(), y.j.to_numpy(), y.prio.to_numpy(np.float64), **kw)
        b = simular(y.t.to_numpy(), y.j.to_numpy(), y.prio.to_numpy(np.float64), **kw)
        dif = float(np.max(np.abs(a["equity"].to_numpy() / b["equity"].to_numpy() - 1)))
        print(res, pal, "v3", round(a["equity"].iloc[-1], 6), "v4", round(b["equity"].iloc[-1], 6), "dif. máx.", dif)


if __name__ == "__main__":
    validar()
