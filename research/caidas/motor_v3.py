"""Motor v3 (numba): reserva configurable y compras de caídas financiadas con deuda.

Mismas reglas de ejecución que los motores anteriores (señal al cierre de t, órdenes
a la apertura de t+1, deslistados liquidados al último precio). Novedades:

- **Reserva arbitraria** (``reservas.py``): rendimientos de noche/día, coste de cambio
  de régimen y si está en un activo de riesgo (``activa``). Mover dinero entre la
  reserva y las acciones cuesta ``coste_res`` sobre lo movido.
- **Tamaño de hueco** = ``fraccion`` × equity × peso (por defecto 1/K).
- **Deuda (overlay)**: la exposición bruta (acciones + reserva) puede llegar a
  ``palanca`` × equity. Con la reserva en riesgo, las compras se financian primero con
  deuda (la reserva sigue invertida) y luego vendiendo reserva; con la reserva fuera
  de riesgo (efectivo/bonos), primero con la reserva y luego con deuda. Las ventas
  devuelven primero la deuda. La deuda paga ``tasa_deuda`` anual, día a día.
- Si el equity llega a 0 la simulación se detiene (ruina).
"""

from __future__ import annotations

import numba
import numpy as np
import pandas as pd

import motor as m
import motor_rapido as mr
import reservas as rs


@numba.njit(cache=True)
def _simular(O, CF, C, ultimo, sal, ptr, cj, cw, K, frac, H, coste, t0, t1, rn, rd, cambio, activa,
             expo_res, coste_res, pal_t, tasa_dia, max_t2):
    T, N = C.shape
    MAXR = 50000
    reg = np.full((MAXR, 6), np.nan)  # t_entrada, t_salida, j, px_entrada, px_salida, peso_cartera
    nreg = 0
    peso_ent = np.zeros(N)
    cash = 1.0  # valor de la reserva (en su activo, o efectivo si no está activa)
    deuda = 0.0
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
    lev = np.empty(t1 - t0)
    caja = np.empty(t1 - t0)  # valor de la reserva al cierre
    deu = np.empty(t1 - t0)  # deuda de las compras de caídas al cierre
    npos = np.zeros(t1 - t0, dtype=np.int64)  # posiciones de caídas abiertas al cierre
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
        # salidas a la apertura: primero se devuelve deuda, el resto va a la reserva
        for j in range(N):
            if pend_sal[j]:
                px = O[t, j] if not np.isnan(O[t, j]) else CF[t, j]
                ingreso = u[j] * px * (1 - coste)
                pago = min(deuda, ingreso)
                deuda -= pago
                cash += ingreso - pago
                mov_res += ingreso - pago  # neto: entradas a la reserva (+), salidas (-)
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
        # mantenimiento de margen: si la exposición bruta real supera un 20 % el tope,
        # se vende reserva a la apertura para devolver deuda hasta volver al tope
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
        # entradas a la apertura
        if npend > 0:
            valor_ap = 0.0
            for j in range(N):
                if abierta[j]:
                    px = O[t, j] if not np.isnan(O[t, j]) else CF[t - 1, j]
                    valor_ap += u[j] * px
            eq_ap = cash + valor_ap - deuda
            hueco = eq_ap * frac
            palanca = pal_t[t]  # decidida con datos hasta el cierre de t-1
            for q in range(npend):
                j = pend[q]
                px = O[t, j]
                if np.isnan(px) or abierta[j] or eq_ap <= 0:
                    continue
                imp = hueco * pend_w[q]
                bruto_res = cash * expo_res[t]  # exposición REAL de la reserva (incluye su palanca interna)
                margen = max(0.0, palanca * eq_ap - (valor_ap + bruto_res))  # bruto adicional permitido
                if activa[t]:
                    # mantener la reserva invertida: deuda hasta el margen, luego vender reserva
                    d = min(imp, margen)
                    v = min(imp - d, cash)
                else:
                    # reserva fuera de riesgo: usarla primero, deuda después
                    v = min(imp, cash)
                    d = min(imp - v, max(0.0, palanca * eq_ap - valor_ap - v - (cash - v) * expo_res[t]))
                imp = v + d
                if imp <= 1e-9:
                    continue
                deuda += d
                cash -= v
                mov_res -= v
                u[j] = imp * (1 - coste) / px
                valor_ap += imp
                abierta[j] = True
                te[j] = t
                pe[j] = px
                peso_ent[j] = imp / eq_ap
            npend = 0
        cash -= abs(mov_res) * coste_res
        cash *= rd[t]
        # cierre: deslistados y valoración
        valor = 0.0
        nab = 0
        for j in range(N):
            if abierta[j]:
                if ultimo[j] <= t and ultimo[j] < T - 1:
                    ingreso = u[j] * CF[t, j] * (1 - coste)
                    pago = min(deuda, ingreso)
                    deuda -= pago
                    cash += ingreso - pago
                    r = CF[t, j] / pe[j] - 1
                    suma_ret += r
                    gan += r > 0
                    n_tr += 1
                    if nreg < MAXR:
                        reg[nreg, 0], reg[nreg, 1], reg[nreg, 2] = te[j], t, j
                        reg[nreg, 3], reg[nreg, 4], reg[nreg, 5] = pe[j], CF[t, j], peso_ent[j]
                        nreg += 1
                    abierta[j] = False
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
        lev[t - t0] = (valor + cash * expo_res[t]) / equity if equity > 0 else 0.0  # exposición bruta real
        caja[t - t0] = cash
        deu[t - t0] = deuda
        npos[t - t0] = nab
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
                w_q = cw[q]
                if w_q < 0:
                    # candidata de SEGUNDA FILA (peso negativo): solo si la cartera de caídas
                    # está casi vacía (menos de max_t2 posiciones abiertas o ya elegidas)
                    if (nab - nsal) + npend >= max_t2:
                        continue
                    w_q = -w_q
                pend[npend] = j
                pend_w[npend] = w_q
                npend += 1
                if npend >= libres:
                    break
    for j in range(N):  # posiciones abiertas al final (t_salida = -1)
        if abierta[j] and nreg < MAXR:
            reg[nreg, 0], reg[nreg, 1], reg[nreg, 2] = te[j], -1, j
            reg[nreg, 3], reg[nreg, 4], reg[nreg, 5] = pe[j], CF[t1 - 1, j], peso_ent[j]
            nreg += 1
    return eq, expo, lev, n_tr, suma_ret, gan, reg[:nreg], pend[:npend].copy(), deuda, caja, deu, npos


_RES: dict = {}


def _regimen() -> np.ndarray:
    if "_reg" not in _RES:
        _RES["_reg"] = rs.regimen_spy()
    return _RES["_reg"]


def reserva(nombre: str):
    if nombre not in _RES:
        _RES.update(rs.cargar())
    return _RES[nombre]


def vector_palanca(palanca: float, palanca_off: float | None = None, vol_objetivo: float | None = None) -> np.ndarray:
    """Palanca máxima por sesión: ``palanca`` con SPY > SMA200 (al cierre anterior) y
    ``palanca_off`` fuera; con ``vol_objetivo``, el exceso sobre 1 se escala por
    min(1, objetivo / vol20 de SPY) (Moreira y Muir, 2017)."""
    reg = _regimen()
    off = palanca if palanca_off is None else palanca_off
    pal = np.where(reg, palanca, off).astype(float)
    if vol_objetivo:
        vol = rs.vol_spy()
        esc = np.nan_to_num(np.minimum(1.0, vol_objetivo / np.where(vol > 0, vol, np.inf)), nan=1.0)
        pal = 1 + (pal - 1) * esc
    return pal


def simular(t, j, score, peso=None, K=3, frac=None, H=20, salida="sma10", coste_bps=10.0, reserva_="spy",
            palanca=1.0, palanca_off=None, vol_objetivo=None, tasa_deuda=0.05, inicio="2002-01-02",
            fin=None, detalle=False, max_t2=0, coste_reserva=None) -> dict:
    """``palanca``: exposición bruta máxima con SPY > SMA200 (al cierre anterior);
    ``palanca_off``: la máxima fuera de tendencia (por defecto, la misma);
    ``vol_objetivo``: si se da, la palanca se reduce cuando la volatilidad de SPY sube;
    ``coste_reserva``: coste de mover dinero entre la reserva y las acciones (por defecto,
    10 pb en cestas de acciones y 2 pb en ETFs)."""
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
    rn, rd, cb, act, ex = reserva(reserva_)
    coste_res = 10e-4 if reserva_.startswith(("mom", "ew", "liq")) else 2e-4
    if coste_reserva is not None:
        coste_res = coste_reserva
    eq, expo, lev, n_tr, suma, gan, reg, pendientes, deuda, caja, deu, npos = _simular(
        p.open, p.close_ff, p.close, p.ultimo, b["salidas"][salida], ptr, j, peso, K,
        (1.0 / K) if frac is None else frac, H, coste_bps / 1e4, t0, t1, rn, rd, cb, act, ex, coste_res,
        vector_palanca(palanca, palanca_off, vol_objetivo), (1 + tasa_deuda) ** (1 / 252) - 1, max_t2)
    idx = pd.DatetimeIndex(p.fechas[t0:t1])
    eq = pd.Series(eq, idx)
    out = m.metricas(eq) if eq.iloc[-1] > 0 else {"cagr": -1.0, "vol": np.nan, "sharpe": np.nan,
                                                  "maxdd": -1.0, "calmar": np.nan}
    años = (idx[-1] - idx[0]).days / 365.25
    out |= {"trades_año": n_tr / años, "acierto": gan / max(n_tr, 1), "ret_medio_trade": suma / max(n_tr, 1),
            "exposicion": float(expo.mean()), "palanca_media": float(lev.mean()),
            "palanca_max": float(lev.max()), "equity": eq,
            "serie_expo": pd.Series(expo, idx), "serie_lev": pd.Series(lev, idx),
            "serie_reserva": pd.Series(caja, idx), "serie_deuda": pd.Series(deu, idx),
            "serie_posiciones": pd.Series(npos, idx)}
    if detalle:
        ops = pd.DataFrame(reg, columns=["t_ent", "t_sal", "j", "px_ent", "px_sal", "peso"])
        ops["ticker"] = p.tickers[ops.j.astype(int)]
        ops["entrada"] = p.fechas[ops.t_ent.astype(int)]
        ops["salida"] = [p.fechas[int(x)] if x >= 0 else pd.NaT for x in ops.t_sal]
        ops["ret"] = ops.px_sal / ops.px_ent - 1
        out |= {"operaciones": ops, "pendientes": [p.tickers[x] for x in pendientes],
                "deuda_final": float(deuda), "exposicion_final": float(expo[-1]), "palanca_final": float(lev[-1])}
    return out


def validar() -> None:
    """Con palanca 1 y la reserva 'spy' debe coincidir con motor_cartera."""
    import motor_cartera as mc
    import ronda3 as r3
    d = r3.cargar()
    x = d[(d.dd20 <= -0.15) & (d.score_v2 >= 5)]
    sc = (x.score_v2 + x.meta_pct.fillna(0.5) * 0.99).to_numpy(np.float64)
    w = np.clip((x.score_v2.to_numpy() - 4) / 2, 0.5, 2.0)
    for res in ("spy", "spy_tend"):
        a = mc.simular(x.t.to_numpy(), x.j.to_numpy(), sc, peso=w, K=3, H=20, salida="sma10", reserva=res)
        b = simular(x.t.to_numpy(), x.j.to_numpy(), sc, peso=w, K=3, H=20, salida="sma10", reserva_=res)
        print(res, "motor_cartera", round(a["equity"].iloc[-1], 6), "motor_v3", round(b["equity"].iloc[-1], 6))


if __name__ == "__main__":
    validar()
