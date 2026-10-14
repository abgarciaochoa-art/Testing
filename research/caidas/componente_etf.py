"""Componente E1: compra de caídas rápidas de ETFs (sectores, países, estilos,
materias primas) con el motor v3.

Configuración elegida SOLO con la búsqueda (oct-2021 →) entre una rejilla pequeña:
caída de n sesiones (3 o 5) ≥ x (5 %, 8 %, 12 %), con o sin filtro de tendencia,
salida (SMA5, SMA10, RSI2 ≥ 70), K (3 o 5); prioridad a la caída más profunda; reserva
SPY en tendencia / SHY fuera; sin deuda. Después se valida en 2002 → sep-2021.

Uso: python research/caidas/componente_etf.py
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

import motor as m
import motor_v3 as v3
from etfs import ETFS, panel_etfs

OUT = Path(__file__).parent / "resultados"
VENTANAS = {"busq": ("2021-10-01", None), "valid": ("2002-01-02", "2021-10-01"),
            "0225": ("2002-01-02", "2026-01-01"), "total": ("2002-01-02", None)}
_C: dict = {}


def datos_etf() -> dict:
    if not _C:
        O, CF = panel_etfs()
        T = len(CF)
        C = CF.copy()
        primero = np.argmax(~np.isnan(CF), axis=0)
        for e in range(C.shape[1]):
            C[: primero[e], e] = np.nan
        ultimo = np.where(~np.isnan(C).all(0), T - 1 - np.argmax(~np.isnan(C[::-1]), axis=0), -1)
        df = pd.DataFrame(C)
        d = df.diff()
        up = d.clip(lower=0).ewm(alpha=0.5, adjust=False, min_periods=2).mean()
        dn = (-d.clip(upper=0)).ewm(alpha=0.5, adjust=False, min_periods=2).mean()
        with np.errstate(invalid="ignore"):
            sal = {"sma5": np.nan_to_num(C > df.rolling(5).mean().to_numpy()).astype(bool),
                   "sma10": np.nan_to_num(C > df.rolling(10).mean().to_numpy()).astype(bool),
                   "rsi2": np.nan_to_num((100 - 100 / (1 + up / dn)).to_numpy() >= 70).astype(bool)}
        _C.update(O=O, CF=pd.DataFrame(CF).ffill().to_numpy(), C=C, ultimo=ultimo, sal=sal,
                  sma200=df.rolling(200).mean().to_numpy())
    return _C


def candidatas(n: int, x: float, tendencia: bool) -> pd.DataFrame:
    D = datos_etf()
    C = D["C"]
    r = np.full(C.shape, np.nan)
    r[n:] = C[n:] / C[:-n] - 1
    with np.errstate(invalid="ignore"):
        ev = r <= -x
        if tendencia:
            ev &= C > D["sma200"]
    t, j = np.nonzero(ev)
    return pd.DataFrame({"t": t, "j": j, "prio": -r[t, j]})


def simular(c: pd.DataFrame, K=3, frac=None, H=20, salida="sma5", reserva_="spy_tend_shy", palanca=1.0,
            palanca_off=1.0, inicio="2002-01-02", fin=None, coste_bps=10.0) -> dict:
    """Mismo motor v3 con las matrices de ETFs en lugar de acciones."""
    D = datos_etf()
    p = m.cargar()
    T = len(p.fechas)
    t, j, score = c.t.to_numpy(), c.j.to_numpy(), c.prio.to_numpy(np.float64)
    orden = np.lexsort((-score, t))
    t, j = t[orden], j[orden].astype(np.int64)
    ptr = np.searchsorted(t, np.arange(T + 1)).astype(np.int64)
    t0 = int(np.searchsorted(p.fechas, np.datetime64(inicio)))
    t1 = T if fin is None else int(np.searchsorted(p.fechas, np.datetime64(fin)))
    rn, rd, cb, act, ex = v3.reserva(reserva_)
    eq, expo, lev, n_tr, suma, gan, _, _, _ = v3._simular(
        D["O"], D["CF"], D["C"], D["ultimo"], D["sal"][salida], ptr, j, np.ones(len(t)), K,
        (1.0 / K) if frac is None else frac, H, coste_bps / 1e4, t0, t1, rn, rd, cb, act, ex, 2e-4,
        v3.vector_palanca(palanca, palanca_off), (1.05) ** (1 / 252) - 1, 0)
    idx = pd.DatetimeIndex(p.fechas[t0:t1])
    eq = pd.Series(eq, idx)
    out = m.metricas(eq)
    años = (idx[-1] - idx[0]).days / 365.25
    out |= {"trades_año": n_tr / años, "acierto": gan / max(n_tr, 1), "ret_medio_trade": suma / max(n_tr, 1),
            "exposicion": float(expo.mean()), "equity": eq}
    return out


def main() -> None:
    filas = []
    for n, x, tend, sal, K in itertools.product((3, 5), (0.05, 0.08, 0.12), (False, True),
                                                 ("sma5", "sma10", "rsi2"), (3, 5)):
        c = candidatas(n, x, tend)
        fila = {"n": n, "x": x, "tendencia": tend, "salida": sal, "K": K, "eventos": len(c)}
        for w, (a, b) in VENTANAS.items():
            r = simular(c, K=K, salida=sal, inicio=a, fin=b)
            fila |= {f"{w}_{k}": v for k, v in r.items() if k != "equity"}
        filas.append(fila)
    df = pd.DataFrame(filas)
    df.to_csv(OUT / "componente_etf.csv", index=False, float_format="%.5f")
    pd.set_option("display.width", 240)
    cols = ["n", "x", "tendencia", "salida", "K", "busq_cagr", "busq_maxdd", "valid_cagr", "valid_maxdd",
            "total_cagr", "total_maxdd", "total_sharpe", "total_trades_año", "total_exposicion"]
    print(df.sort_values("busq_cagr", ascending=False)[cols].head(12).round(3).to_string(index=False))
    print("referencia reserva sola (spy_tend_shy): ver reservas.py")
    print(ETFS)


if __name__ == "__main__":
    main()
