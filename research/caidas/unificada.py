"""Cartera unificada: un solo fondo de capital para las caídas de D1, C1 y E1.

Con capital separado, cada componente deja su parte ociosa en la reserva la mayor parte
del tiempo. Aquí todas las candidatas compiten por los mismos huecos:

- **Universo** = acciones del S&P 500 + 34 ETFs (``etfs.ETFS``).
- **Prioridad** = retorno neto esperado por operación según su origen y puntuación
  (D1: nº de módulos v2; C1: nº de banderas v1; E1: tamaño de la caída), estimado SOLO
  con la ventana de búsqueda (oct-2021 →) con la salida propia de cada origen.
- **Salida propia** de cada posición: D1 > SMA10, C1 y E1 RSI2 ≥ 70 (máx. 20 sesiones).
- **Peso**: convicción de D1 (0,5-1,5 huecos); C1 y E1, 1 hueco.
- Reserva SPY en tendencia / SHY fuera; palanca máxima solo en tendencia.

Uso: python research/caidas/unificada.py
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numba
import numpy as np
import pandas as pd

import componente_etf as ce
import escalera as es
import motor as m
import motor_rapido as mr
import motor_v3 as v3
import salidas as sl
from ronda4 import componentes

OUT = Path(__file__).parent / "resultados"
CORTE = pd.Timestamp("2021-10-01")
E1 = dict(n=5, x=0.05)


@numba.njit(cache=True)
def _simular_u(O, CF, C, ultimo, sal_a, sal_b, ptr, cj, cw, ctipo, K, frac, H, coste, t0, t1, rn, rd, cambio,
               activa, coste_res, palanca_on, palanca_off, regimen, tasa_dia):
    """Igual que motor_v3._simular, pero cada posición usa la salida de su origen
    (tipo 0 -> sal_a, tipo 1 -> sal_b)."""
    T, N = C.shape
    cash = 1.0
    deuda = 0.0
    u = np.zeros(N)
    te = np.full(N, -1)
    pe = np.zeros(N)
    tipo = np.zeros(N, dtype=np.int64)
    abierta = np.zeros(N, dtype=np.bool_)
    pend_sal = np.zeros(N, dtype=np.bool_)
    pend = np.full(N, -1)
    pend_w = np.zeros(N)
    pend_t = np.zeros(N, dtype=np.int64)
    npend = 0
    eq = np.empty(t1 - t0)
    expo = np.empty(t1 - t0)
    n_tr = 0
    suma_ret = 0.0
    gan = 0
    abiertas = np.full(N, -1)
    ruina = False
    for t in range(t0, t1):
        if ruina:
            eq[t - t0] = 0.0
            expo[t - t0] = 0.0
            continue
        if t > t0:
            cash *= rn[t] * (1 - cambio[t])
            deuda *= 1 + tasa_dia
        mov_res = 0.0
        for j in range(N):
            if pend_sal[j]:
                px = O[t, j] if not np.isnan(O[t, j]) else CF[t, j]
                ingreso = u[j] * px * (1 - coste)
                pago = min(deuda, ingreso)
                deuda -= pago
                cash += ingreso - pago
                mov_res += ingreso - pago
                r = px / pe[j] - 1
                suma_ret += r
                gan += r > 0
                n_tr += 1
                abierta[j] = False
                pend_sal[j] = False
                u[j] = 0.0
        if npend > 0:
            valor_ap = 0.0
            for j in range(N):
                if abierta[j]:
                    px = O[t, j] if not np.isnan(O[t, j]) else CF[t - 1, j]
                    valor_ap += u[j] * px
            eq_ap = cash + valor_ap - deuda
            hueco = eq_ap * frac
            palanca = palanca_on if regimen[t] else palanca_off
            for q in range(npend):
                j = pend[q]
                px = O[t, j]
                if np.isnan(px) or abierta[j] or eq_ap <= 0:
                    continue
                imp = hueco * pend_w[q]
                bruto_res = cash if activa[t] else 0.0
                margen = max(0.0, palanca * eq_ap - (valor_ap + bruto_res))
                if activa[t]:
                    d = min(imp, margen)
                    v = min(imp - d, cash)
                else:
                    v = min(imp, cash)
                    d = min(imp - v, max(0.0, palanca * eq_ap - valor_ap - v))
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
                tipo[j] = pend_t[q]
            npend = 0
        cash -= abs(mov_res) * coste_res
        cash *= rd[t]
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
        nsal = 0
        for q in range(nab):
            j = abiertas[q]
            s = sal_a[t, j] if tipo[j] == 0 else sal_b[t, j]
            if (t - te[j] + 1 >= H) or s:
                pend_sal[j] = True
                nsal += 1
        libres = K - (nab - nsal)
        if libres > 0 and t + 1 < t1:
            for q in range(ptr[t], ptr[t + 1]):
                j = cj[q]
                if abierta[j]:
                    continue
                dup = False
                for a in range(npend):
                    if pend[a] == j:
                        dup = True
                if dup:
                    continue
                pend[npend] = j
                pend_w[npend] = cw[q]
                pend_t[npend] = ctipo[q]
                npend += 1
                if npend >= libres:
                    break
    return eq, expo, n_tr, suma_ret, gan


_U: dict = {}


def universo() -> dict:
    """Matrices acciones + ETFs (columnas N..N+E-1 = ETFs)."""
    if not _U:
        b = mr._base()
        p = b["p"]
        D = ce.datos_etf()
        _U.update(
            O=np.hstack([p.open, D["O"]]), CF=np.hstack([p.close_ff, D["CF"]]), C=np.hstack([p.close, D["C"]]),
            ultimo=np.concatenate([p.ultimo, D["ultimo"]]),
            sma10=np.hstack([b["salidas"]["sma10"], D["sal"]["sma10"]]),
            rsi2=np.hstack([b["salidas"]["rsi2"], D["sal"]["rsi2"]]), N=p.close.shape[1])
    return _U


def esperado(x: pd.DataFrame, ret: np.ndarray, cubo: pd.Series) -> pd.Series:
    """Retorno neto medio por cubo en la ventana de búsqueda (lo único que se usa para ordenar)."""
    fechas = pd.DatetimeIndex(m.cargar().fechas[x.t.to_numpy()])
    busq = fechas >= CORTE
    tabla = pd.Series(ret[busq] - 0.002).groupby(cubo.to_numpy()[busq]).mean()
    return cubo.map(tabla).fillna(tabla.min())


def candidatas(incluir_etf: bool = True) -> tuple[pd.DataFrame, dict]:
    comp = componentes()
    p = m.cargar()
    mats = (p.open, p.close, m.sma(5), m.sma(10), m.sma(20), m.rsi(2))
    partes, tablas = [], {}
    # D1: salida SMA10 (código 0, parámetro 10)
    x, _, _ = comp["D1"]
    r, _ = sl._salidas(x.t.to_numpy().astype(np.int64), x.j.to_numpy().astype(np.int64), *mats, 0, 10.0, 20)
    cubo = x.score_v2.clip(upper=8).astype(int)
    e = esperado(x, r, cubo)
    tablas["D1"] = e.groupby(cubo).first().round(4).to_dict()
    partes.append(pd.DataFrame({"t": x.t, "j": x.j, "exp": e + x.meta_pct.fillna(0.5) * 1e-4,
                                "peso": np.minimum(x.peso, 1.5), "tipo": 0, "origen": "D1"}))
    # C1: salida RSI2 >= 70 (código 1, parámetro 70)
    x, _, _ = comp["C1"]
    r, _ = sl._salidas(x.t.to_numpy().astype(np.int64), x.j.to_numpy().astype(np.int64), *mats, 1, 70.0, 20)
    cubo = x.score_limpio.clip(upper=8).astype(int)
    e = esperado(x, r, cubo)
    tablas["C1"] = e.groupby(cubo).first().round(4).to_dict()
    partes.append(pd.DataFrame({"t": x.t, "j": x.j, "exp": e + (-x.r10.fillna(0)).clip(0, 0.99) * 1e-4,
                                "peso": 1.0, "tipo": 1, "origen": "C1"}))
    if incluir_etf:
        c = ce.candidatas(E1["n"], E1["x"], False)
        D = ce.datos_etf()
        r = np.array([np.nan] * len(c))
        rr, _ = sl._salidas(c.t.to_numpy().astype(np.int64), c.j.to_numpy().astype(np.int64), D["O"], D["C"],
                            np.zeros_like(D["C"]), np.zeros_like(D["C"]), np.zeros_like(D["C"]),
                            _rsi2_etf(D["C"]), 1, 70.0, 20)
        r = rr
        cubo = pd.cut(c.prio, [0, 0.08, 0.12, 9], labels=False).fillna(0).astype(int)
        e = esperado(c, r, cubo)
        tablas["E1"] = e.groupby(cubo).first().round(4).to_dict()
        partes.append(pd.DataFrame({"t": c.t, "j": c.j + universo()["N"], "exp": e + c.prio * 1e-4,
                                    "peso": 1.0, "tipo": 1, "origen": "E1"}))
    todo = pd.concat(partes, ignore_index=True)
    todo = todo.sort_values("exp", ascending=False).drop_duplicates(["t", "j"])  # misma acción en D1 y C1
    return todo, tablas


def _rsi2_etf(C: np.ndarray) -> np.ndarray:
    df = pd.DataFrame(C)
    d = df.diff()
    up = d.clip(lower=0).ewm(alpha=0.5, adjust=False, min_periods=2).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=0.5, adjust=False, min_periods=2).mean()
    return (100 - 100 / (1 + up / dn)).to_numpy()


def simular(c: pd.DataFrame, K=8, frac=1 / 3, palanca=1.0, palanca_off=1.0, reserva_="spy_tend_shy",
            coste_bps=10.0, tasa_deuda=0.05, inicio="2002-01-02", fin=None, retraso=0) -> dict:
    U = universo()
    p = m.cargar()
    T = len(p.fechas)
    t = c.t.to_numpy() + retraso
    ok = t < T
    t = t[ok]
    j = c.j.to_numpy()[ok].astype(np.int64)
    s = c.exp.to_numpy()[ok]
    w = c.peso.to_numpy(np.float64)[ok]
    tp = c.tipo.to_numpy()[ok].astype(np.int64)
    orden = np.lexsort((-s, t))
    t, j, w, tp = t[orden], j[orden], w[orden], tp[orden]
    ptr = np.searchsorted(t, np.arange(T + 1)).astype(np.int64)
    t0 = int(np.searchsorted(p.fechas, np.datetime64(inicio)))
    t1 = T if fin is None else int(np.searchsorted(p.fechas, np.datetime64(fin)))
    rn, rd, cb, act, _ = v3.reserva(reserva_)
    eq, expo, n_tr, suma, gan = _simular_u(
        U["O"], U["CF"], U["C"], U["ultimo"], U["sma10"], U["rsi2"], ptr, j, w, tp, K, frac, 20, coste_bps / 1e4,
        t0, t1, rn, rd, cb, act, 2e-4, palanca, palanca_off, v3._regimen(), (1 + tasa_deuda) ** (1 / 252) - 1)
    idx = pd.DatetimeIndex(p.fechas[t0:t1])
    eq = pd.Series(eq, idx)
    out = m.metricas(eq)
    años = (idx[-1] - idx[0]).days / 365.25
    out |= {"trades_año": n_tr / años, "acierto": gan / max(n_tr, 1), "ret_medio_trade": suma / max(n_tr, 1),
            "exposicion": float(expo.mean()), "equity": eq}
    return out


def main() -> None:
    filas = []
    tablas_guardadas = {}
    for incluir in (False, True):
        c, tablas = candidatas(incluir)
        tablas_guardadas[str(incluir)] = tablas
        for K, frac, pal in itertools.product((5, 8, 12), (1 / 3, 1 / 4), (1.0, 1.5, 2.0)):
            r = simular(c, K=K, frac=frac, palanca=pal)
            fila = {"con_ETF": incluir, "K": K, "hueco": round(frac, 3), "palanca": pal,
                    **es.metricas_ventanas(r["equity"]), "operaciones/año": r["trades_año"],
                    "exposición media": r["exposicion"]}
            filas.append(fila)
    df = pd.DataFrame(filas)
    df.to_csv(OUT / "unificada.csv", index=False, float_format="%.4f")
    (OUT / "unificada_esperados.json").write_text(json.dumps(tablas_guardadas, indent=1, default=str))
    pd.set_option("display.width", 260)
    cols = ["con_ETF", "K", "hueco", "palanca", "CAGR búsqueda", "maxDD búsqueda", "CAGR validación",
            "maxDD validación", "CAGR 2002-2025", "CAGR total", "maxDD total", "Sharpe", "operaciones/año",
            "exposición media"]
    print(json.dumps(tablas_guardadas, indent=1, default=str))
    for pal in (1.0, 1.5, 2.0):
        print(df[df.palanca == pal].sort_values("CAGR búsqueda", ascending=False)[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
