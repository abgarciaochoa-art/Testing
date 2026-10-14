"""Reservas candidatas para el capital que no está en compras de caídas.

Cada reserva es un "activo sintético" con valor a la apertura y al cierre de cada
sesión, del que el motor usa dos rendimientos:

- ``rn[t]`` = valor apertura t / valor cierre t-1 (noche, con el activo que se tenía);
- ``rd[t]`` = valor cierre t / valor apertura t (día, con el activo elegido en t);

más ``cambio[t]`` (coste de cambiar de activo en la apertura t) y ``activa[t]``
(True si la reserva está en un activo de riesgo; el motor la usa para decidir si
financia las compras con deuda o con la reserva).

Candidatas (fijadas ANTES de mirar resultados):
- ``spy``: SPY siempre.
- ``spy_tend``: SPY si SPY > SMA200 al cierre anterior; si no, efectivo al 0 %.
- ``spy_tend_shy``: ídem, pero fuera de tendencia en SHY (Tesoro 1-3 años: efectivo
  con interés; antes de jul-2002, efectivo al 0 %).
- ``spy_tend_ief``: ídem con IEF (Tesoro 7-10 años).
- ``mom20`` / ``mom50``: cesta equiponderada de las 20/50 acciones del S&P 500 (miembros
  PIT, líquidas) con mayor momentum 12-1 (Jegadeesh-Titman), formada al cierre de fin
  de mes y comprada a la apertura siguiente; 10 pb sobre lo negociado.
- ``mom20_tend_shy`` / ``mom50_tend_shy``: momentum en tendencia, SHY fuera.
- ``ew``: equiponderada mensual de todos los miembros PIT líquidos.
- ``qqq``: solo como referencia (elegirla sería sesgo retrospectivo).
- ``spy_vg_shy``: SPY gestionado por volatilidad en tendencia (objetivo 15 %, máx. 1,5x),
  SHY fuera (Moreira y Muir, 2017).
- ``spy15_tend_shy`` / ``spy2_tend_shy``: SPY a 1,5x / 2x solo en tendencia, SHY fuera
  ("Leverage for the Long Run", Gayed y Bilello 2016), deuda al 5 %.

Uso: python research/caidas/reservas.py   -> data/cache/caidas/reservas.npz + tabla
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import motor as m

CACHE = m.ROOT / "data" / "cache" / "caidas"
SALIDA = CACHE / "reservas.npz"
COSTE_CESTA = 10e-4
COSTE_ETF = 2e-4


def etf(ticker: str) -> tuple[np.ndarray, np.ndarray]:
    """Apertura y cierre ajustados (dividendos+splits) alineados a las fechas del panel."""
    p = m.cargar()
    if ticker in p.col:
        j = p.col[ticker]
        return p.open[:, j].copy(), p.close_ff[:, j].copy()
    x = pd.read_parquet(CACHE / "prices.parquet", filters=[("ticker", "==", ticker)])
    x = x.drop_duplicates("date").set_index("date").sort_index()
    f = x.adj_close / x.close
    idx = pd.DatetimeIndex(p.fechas)
    c = x.adj_close.reindex(idx).ffill().to_numpy()
    o = (x.open * f).reindex(idx).to_numpy()
    o = np.where(np.isnan(o), np.r_[np.nan, c[:-1]], o)
    return o, c


def cesta(seleccion: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Valor de apertura (antes de rebalancear) y de cierre de una cesta equiponderada
    que se rebalancea a la apertura del día siguiente a cada fin de mes.

    ``seleccion``: matriz booleana T×N con las acciones elegidas al cierre de cada fin
    de mes (solo se miran esas filas)."""
    p = m.cargar()
    O, CF = p.open, p.close_ff
    T, N = CF.shape
    fechas = pd.DatetimeIndex(p.fechas)
    fin_mes = np.flatnonzero(fechas.month != np.roll(fechas.month, -1))
    reb = {int(t) + 1: np.flatnonzero(seleccion[t]) for t in fin_mes if t + 1 < T and seleccion[t].any()}
    v_ap = np.full(T, np.nan)
    v_ci = np.full(T, np.nan)
    u = np.zeros(N)
    valor = 1.0
    efectivo = 0.0
    empezado = False
    for t in range(T):
        if empezado:
            po = np.where(np.isnan(O[t]), CF[t - 1], O[t])
            v_pre = float(np.nansum(u * po)) + efectivo
        else:
            v_pre = valor
        if t in reb:
            elegidas = reb[t]
            po = np.where(np.isnan(O[t]), CF[t - 1] if t else O[t], O[t])
            ok = elegidas[~np.isnan(po[elegidas])]
            actual = u * np.nan_to_num(po)
            objetivo = np.zeros(N)
            objetivo[ok] = v_pre / len(ok)
            coste = COSTE_CESTA * (np.abs(objetivo - actual).sum() + (efectivo if empezado else 0.0))
            v_post = v_pre - coste
            u = np.zeros(N)
            u[ok] = v_post / len(ok) / po[ok]
            efectivo = 0.0
            empezado = True
        v_ap[t] = v_pre
        if empezado:
            # una acción deslistada queda congelada a su último cierre (efectivo hasta el rebalanceo)
            v_ci[t] = float(np.nansum(u * CF[t])) + efectivo
        else:
            v_ci[t] = valor
    return v_ap, v_ci


def componer(riesgo: tuple, defensa: tuple | None, regimen: np.ndarray | None, coste: float):
    """Reserva que está en ``riesgo`` cuando ``regimen`` (decidido al cierre de t-1) y en
    ``defensa`` (o efectivo si None) cuando no."""
    o_r, c_r = riesgo
    T = len(c_r)
    if regimen is None:
        regimen = np.ones(T, dtype=bool)
    if defensa is None:
        o_d, c_d = np.ones(T), np.ones(T)
    else:
        # serie defensiva normalizada a 1 en su primer cierre; antes de existir, efectivo (1)
        o_raw, c_raw = defensa
        primero = int(np.argmax(~np.isnan(c_raw)))
        base = c_raw[primero]
        c_d = np.where(np.isnan(c_raw), np.nan, c_raw / base)
        o_d = np.where(np.isnan(o_raw), np.nan, o_raw / base)
        c_d[:primero] = 1.0
        o_d[: primero + 1] = 1.0
        c_d = pd.Series(c_d).ffill().to_numpy()
        o_d = np.where(np.isnan(o_d), np.r_[1.0, c_d[:-1]], o_d)
    rn = np.ones(T)
    rd = np.ones(T)
    cambio = np.zeros(T)
    for t in range(1, T):
        ant = regimen[t - 1]
        hoy = regimen[t]
        if ant:
            rn[t] = o_r[t] / c_r[t - 1]
        else:
            rn[t] = o_d[t] / c_d[t - 1] if c_d[t - 1] > 0 else 1.0
        rd[t] = c_r[t] / o_r[t] if hoy else (c_d[t] / o_d[t] if o_d[t] > 0 else 1.0)
        if hoy != ant:
            cambio[t] = coste
    rn = np.nan_to_num(rn, nan=1.0)
    rd = np.nan_to_num(rd, nan=1.0)
    return rn, rd, cambio, regimen.astype(bool), regimen.astype(float)


def apalancado(activo: tuple, L: float, tasa: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Serie sintética con exposición L al activo, reajustada a la apertura y al cierre,
    pagando ``tasa`` anual sobre lo prestado (L-1). Base de "Leverage for the Long Run"
    (Gayed y Bilello, 2016) cuando se combina con el filtro de SMA200."""
    o, c = activo
    T = len(c)
    o_l = np.full(T, np.nan)
    c_l = np.full(T, np.nan)
    tasa_dia = (1 + tasa) ** (1 / 252) - 1
    primero = int(np.argmax(~np.isnan(c)))
    c_l[primero] = o_l[primero] = 1.0
    for t in range(primero + 1, T):
        oo = o[t] if not np.isnan(o[t]) else c[t - 1]
        o_l[t] = c_l[t - 1] * (1 + L * (oo / c[t - 1] - 1))
        c_l[t] = o_l[t] * (1 + L * (c[t] / oo - 1)) - c_l[t - 1] * (L - 1) * tasa_dia
    return o_l, c_l


def vol_spy(ventana: int = 20) -> np.ndarray:
    """Volatilidad realizada anualizada de SPY con datos hasta el cierre de t-1."""
    p = m.cargar()
    r = pd.Series(p.close_ff[:, p.col["SPY"]]).pct_change()
    v = (r.rolling(ventana).std() * np.sqrt(252)).to_numpy()
    return np.roll(v, 1)


def gestionada(riesgo: tuple, defensa: tuple, regimen: np.ndarray, objetivo: float = 0.15,
               maximo: float = 1.5, tasa: float = 0.05):
    """Reserva gestionada por volatilidad (Moreira y Muir, 2017): en tendencia, exposición
    e = min(maximo, objetivo / vol20 de SPY) al activo de riesgo (si e < 1, el resto en
    ``defensa``; si e > 1, deuda al ``tasa``); fuera de tendencia, todo en ``defensa``.
    La exposición se decide al cierre de t-1 y se ajusta en la apertura de t (2 pb)."""
    o_r, c_r = riesgo
    T = len(c_r)
    o_raw, c_raw = defensa
    primero = int(np.argmax(~np.isnan(c_raw)))
    base = c_raw[primero]
    c_d = c_raw / base
    o_d = o_raw / base
    c_d[:primero] = 1.0
    o_d[: primero + 1] = 1.0
    c_d = pd.Series(c_d).ffill().to_numpy()
    o_d = np.where(np.isnan(o_d), np.r_[1.0, c_d[:-1]], o_d)
    vol = vol_spy()
    e = np.where(regimen, np.minimum(maximo, objetivo / np.where(vol > 0, vol, np.inf)), 0.0)
    e = np.nan_to_num(e)
    tasa_dia = (1 + tasa) ** (1 / 252) - 1
    rn = np.ones(T)
    rd = np.ones(T)
    cambio = np.zeros(T)
    for t in range(1, T):
        a = e[t - 1]  # exposición mantenida durante la noche
        r_noche_r = o_r[t] / c_r[t - 1] - 1 if not np.isnan(o_r[t]) else 0.0
        r_noche_d = o_d[t] / c_d[t - 1] - 1
        rn[t] = 1 + a * r_noche_r + max(0.0, 1 - a) * r_noche_d
        b = e[t]  # exposición ajustada en la apertura
        cambio[t] = COSTE_ETF * abs(b - a)
        r_dia_r = c_r[t] / o_r[t] - 1 if not np.isnan(o_r[t]) else c_r[t] / c_r[t - 1] - 1
        r_dia_d = c_d[t] / o_d[t] - 1
        rd[t] = 1 + b * r_dia_r + max(0.0, 1 - b) * r_dia_d - max(0.0, b - 1) * tasa_dia
    return np.nan_to_num(rn, nan=1.0), np.nan_to_num(rd, nan=1.0), cambio, e > 0, e


def regimen_spy() -> np.ndarray:
    p = m.cargar()
    spy = pd.Series(p.close_ff[:, p.col["SPY"]])
    reg = np.roll((spy > spy.rolling(200).mean()).to_numpy(), 1)  # decidido al cierre de t-1
    reg[0] = False
    return reg


def construir() -> dict:
    p = m.cargar()
    liq = m.liquido()
    c = p.close
    mom = np.full(c.shape, np.nan)
    mom[252:] = c[231:-21] / c[:-252] - 1  # 12-1: de t-252 a t-21
    base = p.miembro & liq & ~np.isnan(mom)
    sel = {}
    for n in (20, 50):
        s = np.zeros(c.shape, dtype=bool)
        x = np.where(base, mom, -np.inf)
        top = np.argsort(-x, axis=1)[:, :n]
        filas = np.arange(len(c))[:, None]
        s[filas, top] = True
        sel[n] = s & base
    reg = regimen_spy()
    spy = etf("SPY")
    shy, ief = etf("SHY"), etf("IEF")
    cestas = {f"mom{n}": cesta(sel[n]) for n in (20, 50)}
    cestas["ew"] = cesta(base | (p.miembro & liq))
    R = {
        "spy": componer(spy, None, None, 0.0),
        "spy_tend": componer(spy, None, reg, COSTE_ETF),
        "spy_tend_shy": componer(spy, shy, reg, COSTE_ETF),
        "spy_tend_ief": componer(spy, ief, reg, COSTE_ETF),
        "mom20": componer(cestas["mom20"], None, None, 0.0),
        "mom50": componer(cestas["mom50"], None, None, 0.0),
        "mom20_tend_shy": componer(cestas["mom20"], shy, reg, COSTE_CESTA),
        "mom50_tend_shy": componer(cestas["mom50"], shy, reg, COSTE_CESTA),
        "ew": componer(cestas["ew"], None, None, 0.0),
        "qqq": componer(etf("QQQ"), None, None, 0.0),
        "spy15_tend_shy": _escalar(componer(apalancado(spy, 1.5), shy, reg, COSTE_ETF), 1.5),
        "spy_vg_shy": gestionada(spy, shy, reg, 0.15, 1.5),
        "spy2_tend_shy": _escalar(componer(apalancado(spy, 2.0), shy, reg, COSTE_ETF), 2.0),
    }
    np.savez(SALIDA, **{f"{k}__{i}": v for k, arrs in R.items()
                        for i, v in zip(("rn", "rd", "cambio", "activa", "expo"), arrs, strict=True)})
    return R


def _escalar(r: tuple, L: float) -> tuple:
    """Exposición de riesgo de una reserva apalancada L veces cuando está en riesgo."""
    return r[0], r[1], r[2], r[3], r[4] * L


def cargar() -> dict:
    z = np.load(SALIDA)
    claves = sorted({k.split("__")[0] for k in z.files})
    return {k: tuple(z[f"{k}__{i}"] for i in ("rn", "rd", "cambio", "activa", "expo")) for k in claves}


def tabla(R: dict) -> pd.DataFrame:
    p = m.cargar()
    fechas = pd.DatetimeIndex(p.fechas)
    filas = []
    for k, (rn, rd, cb, _, _) in R.items():
        diario = rn * rd * (1 - cb)
        eq = pd.Series(np.cumprod(diario), fechas)
        fila = {"reserva": k}
        for nombre, (a, b) in {"busq": ("2021-10-01", None), "valid": ("2002-01-02", "2021-10-01"),
                               "2002-2025": ("2002-01-02", "2026-01-01"), "total": ("2002-01-02", None)}.items():
            s = eq[(eq.index >= a) & ((eq.index < b) if b else True)]
            mt = m.metricas(s / s.iloc[0])
            fila |= {f"{nombre}_cagr": mt["cagr"], f"{nombre}_maxdd": mt["maxdd"]}
        fila["sharpe_total"] = m.metricas(eq[eq.index >= "2002-01-02"])["sharpe"]
        filas.append(fila)
    return pd.DataFrame(filas)


if __name__ == "__main__":
    R = construir()
    pd.set_option("display.width", 220)
    print(tabla(R).round(3).to_string(index=False))
