"""Motor de Noise Band: ruptura de la "zona de ruido" intradía del S&P 500.

La zona de ruido (Zarattini, Aziz y Barbon, 2024, "Beat the Market: An Effective Intraday
Momentum Strategy for S&P500 ETF (SPY)") es un cono que nace cada día en la apertura y se
abre con las horas. Para cada minuto del día:

- sigma(d, k) = media, en las 14 sesiones anteriores, de |precio(k) / apertura - 1|;
- banda superior = max(apertura, cierre anterior) × (1 + sigma);
- banda inferior = min(apertura, cierre anterior) × (1 - sigma).

Dentro del cono el precio se mueve "por ruido"; si lo rompe, se apuesta a que sigue
(momentum intradía). Convenciones del motor:

- Velas de 1 minuto de la sesión regular (9:30-16:00, hora de Nueva York).
- **Controles** a las HH:00 y HH:30, de 10:00 a 15:30: la decisión se toma con el cierre de
  la vela que acaba a esa hora y se ejecuta a la apertura de la vela siguiente. Con
  ``entrada=1``, en cambio, la entrada es una orden stop en la banda que puede saltar en
  cualquier minuto de 10:00 a 15:30 (al precio de la banda, o a la apertura si salta por
  encima).
- **Stops y objetivos** dentro de la vela: si la apertura ya los supera, a la apertura; si
  no, al nivel. Si en la misma vela se tocan el stop y el objetivo, cuenta el stop.
- Con ``intradia`` se cierra todo en la última vela del día; si no, la posición duerme y
  la noche cuenta (la deuda paga ``tasa`` anual si la palanca pasa de 1).
- Costes: ``coste_bps`` por lado sobre lo negociado (o por acción de SPY, como el artículo).
  El efectivo no rinde.
- La palanca de cada día (``lev``) se decide con datos hasta el día anterior.
"""

from __future__ import annotations

from pathlib import Path

import numba
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATOS = ROOT / "data" / "cache" / "noise_band" / "rth.npz"
VELAS = 390
CONTROLES = np.zeros(VELAS, dtype=np.bool_)
CONTROLES[29:360:30] = True  # velas que cierran a las 10:00, 10:30, ..., 15:30
INF = np.inf


def cargar(n_sigma: int = 14, ruta: Path = DATOS) -> dict:
    """Datos de la sesión regular con el cono, la VWAP y la volatilidad diaria."""
    z = np.load(ruta)
    O, H, L, C, V = z["O"], z["H"], z["L"], z["C"], z["V"]
    D = len(z["fechas"])
    ult = z["ultimo"].astype(np.int64)
    abre = O[:, 0]
    previo = z["previo"].copy()
    previo[0] = abre[0]
    mov = np.abs(C / abre[:, None] - 1)
    sigma = pd.DataFrame(mov).rolling(n_sigma, min_periods=n_sigma).mean().shift(1).to_numpy()
    arriba = np.maximum(abre, previo)[:, None] * (1 + sigma)
    abajo = np.minimum(abre, previo)[:, None] * (1 - sigma)
    tp = (H + L + C) / 3
    vw = np.cumsum(tp * V, axis=1) / np.maximum(np.cumsum(V, axis=1), 1e-12)
    vw = np.where(np.cumsum(V, axis=1) > 0, vw, np.cumsum(tp, axis=1) / np.arange(1, VELAS + 1))
    cierre = C[np.arange(D), ult]
    r_dia = np.r_[np.nan, cierre[1:] / cierre[:-1] - 1]
    vol14 = pd.Series(r_dia).rolling(14, min_periods=14).std().shift(1).to_numpy()
    return dict(fechas=pd.DatetimeIndex(z["fechas"]), O=O, H=H, L=L, C=C, ult=ult, arriba=arriba,
                abajo=abajo, vw=vw, sigma=sigma, cierre=cierre, vol14=vol14,
                ref_arriba=np.maximum(abre, previo), ref_abajo=np.minimum(abre, previo))


def palanca(d: dict, objetivo: float | None = None, maximo: float = 4.0) -> np.ndarray:
    """1 cada día, o min(maximo, objetivo / volatilidad diaria de 14 sesiones) (artículo: 2 %, 4×)."""
    if objetivo is None:
        return np.ones(len(d["fechas"]))
    v = d["vol14"]
    return np.nan_to_num(np.minimum(maximo, objetivo / np.where(v > 0, v, np.inf)), nan=0.0)


@numba.njit(cache=True)
def _simular(O, H, L, C, ult, UB, LB, VW, RA, RB, chk, lev, lados, entrada, trail_vwap, salida_banda, stop_tipo,
             stop_pct, tp_pct, intradia, costes, tasa_dia):
    D = C.shape[0]
    cash = 1.0
    units = 0.0
    lado = 0  # +1 largo, -1 corto
    p_ent = 0.0
    stop = 0.0
    tp = 0.0
    d_ent = 0
    k_ent = 0
    eq = np.empty(D)
    en_mercado = np.zeros(D)
    reg = np.full((50000, 7), np.nan)  # día y minuto de entrada, día y minuto de salida, lado, precio entrada, salida
    nreg = 0
    for d in range(D):
        coste = costes[d]  # por lado, sobre lo negociado ese día
        valido = not np.isnan(UB[d, 0])
        pend = 0  # 0 nada, 1 cerrar, 2 entrar largo, 3 entrar corto, 4 girar a largo, 5 girar a corto
        minutos = 0
        for k in range(ult[d] + 1):
            # 1) órdenes de un control anterior, a la apertura de esta vela
            if pend != 0:
                px = O[d, k]
                if lado != 0 and pend in (1, 4, 5):
                    cash += units * px - coste * abs(units * px)
                    if nreg < 50000:
                        reg[nreg, 0], reg[nreg, 1], reg[nreg, 2], reg[nreg, 3] = d_ent, k_ent, d, k
                        reg[nreg, 4], reg[nreg, 5], reg[nreg, 6] = lado, p_ent, px
                        nreg += 1
                    units = 0.0
                    lado = 0
                if pend >= 2:  # entrar (o girar)
                    nuevo = 1 if pend in (2, 4) else -1
                    eqn = cash
                    units = nuevo * lev[d] * eqn / px
                    cash -= units * px + coste * abs(units * px)
                    lado, p_ent, d_ent, k_ent = nuevo, px, d, k
                    if stop_tipo == 2:
                        stop = RA[d] if nuevo > 0 else RB[d]
                    elif stop_tipo == 3:
                        stop = px * (1 - stop_pct) if nuevo > 0 else px * (1 + stop_pct)
                    tp = px * (1 + tp_pct) if nuevo > 0 else px * (1 - tp_pct)
                pend = 0
            # 2) entrada con orden stop en la banda (cualquier minuto de 10:00 a 15:30)
            if entrada == 1 and lado == 0 and valido and 30 <= k <= 359:
                px = 0.0
                nuevo = 0
                if H[d, k] >= UB[d, k]:
                    px = max(O[d, k], UB[d, k])
                    nuevo = 1
                elif lados == 2 and L[d, k] <= LB[d, k]:
                    px = min(O[d, k], LB[d, k])
                    nuevo = -1
                if nuevo != 0:
                    units = nuevo * lev[d] * cash / px
                    cash -= units * px + coste * abs(units * px)
                    lado, p_ent, d_ent, k_ent = nuevo, px, d, k
                    if stop_tipo == 2:
                        stop = RA[d] if nuevo > 0 else RB[d]
                    elif stop_tipo == 3:
                        stop = px * (1 - stop_pct) if nuevo > 0 else px * (1 + stop_pct)
                    tp = px * (1 + tp_pct) if nuevo > 0 else px * (1 - tp_pct)
            # 3) stops y objetivo dentro de la vela
            if lado != 0:
                s = 0.0
                if stop_tipo == 1:
                    s = LB[d, k] if lado > 0 else UB[d, k]
                elif stop_tipo >= 2:
                    s = stop
                salir = 0.0
                if lado > 0:
                    if stop_tipo > 0 and not np.isnan(s) and L[d, k] <= s:
                        salir = min(O[d, k], s)
                    elif tp_pct < 1e9 and H[d, k] >= tp:
                        salir = max(O[d, k], tp)
                else:
                    if stop_tipo > 0 and not np.isnan(s) and H[d, k] >= s:
                        salir = max(O[d, k], s)
                    elif tp_pct < 1e9 and L[d, k] <= tp:
                        salir = min(O[d, k], tp)
                if salir > 0:
                    cash += units * salir - coste * abs(units * salir)
                    if nreg < 50000:
                        reg[nreg, 0], reg[nreg, 1], reg[nreg, 2], reg[nreg, 3] = d_ent, k_ent, d, k
                        reg[nreg, 4], reg[nreg, 5], reg[nreg, 6] = lado, p_ent, salir
                        nreg += 1
                    units = 0.0
                    lado = 0
            if lado != 0:
                minutos += 1
            # 4) controles a las HH:00 y HH:30 (se ejecuta en la vela siguiente)
            if chk[k] and valido and k + 1 <= ult[d]:
                p = C[d, k]
                if lado == 0:
                    if entrada == 0:
                        if p > UB[d, k]:
                            pend = 2
                        elif lados == 2 and p < LB[d, k]:
                            pend = 3
                elif lado > 0:
                    sale = False
                    if trail_vwap and p < max(UB[d, k], VW[d, k]):
                        sale = True
                    if salida_banda == 1 and p < UB[d, k]:
                        sale = True
                    if salida_banda == 2 and p < LB[d, k]:
                        sale = True
                    if sale:
                        pend = 5 if (lados == 2 and p < LB[d, k]) else 1
                else:
                    sale = False
                    if trail_vwap and p > min(LB[d, k], VW[d, k]):
                        sale = True
                    if salida_banda == 1 and p > LB[d, k]:
                        sale = True
                    if salida_banda == 2 and p > UB[d, k]:
                        sale = True
                    if sale:
                        pend = 4 if p > UB[d, k] else 1
        # 5) cierre del día
        if lado != 0 and intradia:
            px = C[d, ult[d]]
            cash += units * px - coste * abs(units * px)
            if nreg < 50000:
                reg[nreg, 0], reg[nreg, 1], reg[nreg, 2], reg[nreg, 3] = d_ent, k_ent, d, ult[d]
                reg[nreg, 4], reg[nreg, 5], reg[nreg, 6] = lado, p_ent, px
                nreg += 1
            units = 0.0
            lado = 0
        if cash < 0:
            cash *= 1 + tasa_dia  # deuda que duerme
        eq[d] = cash + units * C[d, ult[d]]
        en_mercado[d] = minutos / (ult[d] + 1)
        if eq[d] <= 0:
            eq[d:] = 0.0
            break
    return eq, en_mercado, reg[:nreg]


def costes_dia(d: dict, coste_bps: float = 0.0, coste_por_accion: float | None = None) -> np.ndarray:
    """Coste por lado de cada día, en tanto por uno: fijo en pb, o en dólares por acción de SPY
    (que cotiza a ~1/10 del índice; se usa el cierre de la víspera)."""
    D = len(d["fechas"])
    if coste_por_accion is None:
        return np.full(D, coste_bps / 1e4)
    spy = np.r_[d["O"][0, 0], d["cierre"][:-1]] / 10
    return coste_por_accion / spy


def simular(d: dict, lados: int = 1, entrada: int = 0, trail_vwap: bool = True, salida_banda: int = 1,
            stop_tipo: int = 0, stop_pct: float = 0.0, tp_pct: float = INF, intradia: bool = True,
            coste_bps: float = 2.0, lev: np.ndarray | None = None, tasa: float = 0.05,
            controles: np.ndarray | None = None, coste_por_accion: float | None = None) -> dict:
    """``salida_banda``: 0 ninguna, 1 volver dentro del cono (cruzar la banda propia), 2 romper
    el cono por el otro lado. ``stop_tipo``: 0 ninguno, 1 banda contraria (dinámica), 2
    vértice del cono (max/min de apertura y cierre anterior), 3 ``stop_pct`` desde la entrada.
    ``controles``: velas en las que se decide (por defecto, las que cierran a las HH:00 y HH:30).
    ``coste_por_accion``: en lugar de ``coste_bps``, dólares por acción de SPY y por lado (el
    artículo usa 0,0035 de comisión + 0,001 de deslizamiento); SPY cotiza a ~1/10 del índice."""
    D = len(d["fechas"])
    lev = np.ones(D) if lev is None else lev
    costes = costes_dia(d, coste_bps, coste_por_accion)
    tp = 1e18 if not np.isfinite(tp_pct) else tp_pct
    eq, mercado, reg = _simular(d["O"], d["H"], d["L"], d["C"], d["ult"], d["arriba"], d["abajo"], d["vw"],
                                d["ref_arriba"], d["ref_abajo"], CONTROLES if controles is None else controles,
                                lev.astype(np.float64), lados, entrada, trail_vwap, salida_banda, stop_tipo, stop_pct,
                                tp, intradia, costes.astype(np.float64), (1 + tasa) ** (1 / 252) - 1)
    ops = pd.DataFrame(reg, columns=["d_ent", "k_ent", "d_sal", "k_sal", "lado", "px_ent", "px_sal"])
    ops["ret"] = (ops.lado * (ops.px_sal / ops.px_ent - 1) - costes[ops.d_ent.astype(int)]
                  - costes[ops.d_sal.astype(int)])
    ops["entrada"] = d["fechas"][ops.d_ent.astype(int)]
    ops["dias"] = ops.d_sal - ops.d_ent
    return {"equity": pd.Series(eq, d["fechas"]), "en_mercado": pd.Series(mercado, d["fechas"]), "operaciones": ops}


def metricas(eq: pd.Series, ops: pd.DataFrame | None = None, mercado: pd.Series | None = None) -> dict:
    eq = eq / eq.iloc[0]
    r = eq.pct_change().dropna()
    años = (eq.index[-1] - eq.index[0]).days / 365.25
    dd = (eq / eq.cummax() - 1).min()
    cagr = eq.iloc[-1] ** (1 / años) - 1 if eq.iloc[-1] > 0 else -1.0
    out = {"cagr": cagr, "vol": r.std() * np.sqrt(252), "sharpe": r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0,
           "peor_caida": dd, "calmar": cagr / abs(dd) if dd < 0 else np.nan}
    if ops is not None:
        o = ops[(ops.entrada >= eq.index[0]) & (ops.entrada <= eq.index[-1])]
        out |= {"operaciones_año": len(o) / años, "acierto": (o.ret > 0).mean() if len(o) else np.nan,
                "ret_medio": o.ret.mean() if len(o) else np.nan}
    if mercado is not None:
        m = mercado.loc[eq.index]
        out |= {"dias_con_posicion": (m > 0).mean(), "tiempo_en_mercado": m.mean()}
    return out
