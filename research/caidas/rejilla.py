"""Rejilla de variantes de "comprar caídas" con registro completo de pruebas.

Cada configuración se ejecuta sobre 2002-2026 con costes de 10 pb por lado. Se
guardan métricas in-sample (2002-2015), out-of-sample (2016-2026) y totales, y la
serie diaria de retornos de TODAS las pruebas (para PBO y Sharpe deflactado: el
número de pruebas es el tamaño real de esta rejilla, no el de las que se enseñan).

Uso: python research/caidas/rejilla.py [--rapido]
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import motor as m

OUT = Path(__file__).parent / "resultados"


def configuraciones() -> list[dict]:
    cfgs = []
    # Familia A: caída rápida (desde el máximo de pocas sesiones), mantener semanas/meses.
    for n, x, f, (sal, h), k, md in itertools.product(
        (10, 20, 40, 60), (0.15, 0.2, 0.25, 0.3, 0.4), ("ninguno", "tendencia", "mercado"),
        (("tiempo", 20), ("tiempo", 60), ("tiempo", 126), ("sma20", 60)), (5, 10, 20), (None, 2),
    ):
        cfgs.append(dict(familia="A_caida_rapida", tipo="dd", n=n, x=x, filtro=f, salida=sal, h=h, k=k, max_dia=md))
    # Familia B: reversión de corto plazo (días), salida en el rebote.
    cortas = ([dict(tipo="rsi", n=2, x=x) for x in (5, 10)]
              + [dict(tipo="racha", n=n) for n in (3, 4, 5)]
              + [dict(tipo="ret", n=1, x=0.05), dict(tipo="ret", n=5, x=0.1)]
              + [dict(tipo="z", n=20, x=2.5)])
    for s, f, (sal, h), k in itertools.product(
        cortas, ("ninguno", "tendencia", "mercado"),
        (("tiempo", 5), ("tiempo", 10), ("sma5", 10), ("rsi2", 10), ("sma5", 20)), (10, 20),
    ):
        cfgs.append(dict(familia="B_reversion_corta", **s, filtro=f, salida=sal, h=h, k=k))
    # Familia C: caída lenta y profunda (desde el máximo anual), mantener meses.
    for n, x, f, h, k in itertools.product(
        (126, 252), (0.3, 0.4, 0.5), ("ninguno", "tendencia", "mercado"), (63, 126, 252), (10, 20),
    ):
        cfgs.append(dict(familia="C_caida_lenta", tipo="dd", n=n, x=x, filtro=f, salida="tiempo", h=h, k=k))
    for i, c in enumerate(cfgs):
        c["id"] = i
    return cfgs


def configuraciones_fase2() -> list[dict]:
    """Fase 2 (decidida tras ver la fase 1): señales con filtro de tendencia, más
    concentradas y con el efectivo ocioso en SPY. Se numeran a partir de 10000 y
    cuentan en el total de pruebas del Sharpe deflactado."""
    cfgs = []
    for (n, x), f, (sal, h), k, res in itertools.product(
        ((3, 0.07), (3, 0.1), (5, 0.07), (5, 0.1), (5, 0.15), (10, 0.1), (10, 0.15), (10, 0.2)),
        ("tendencia", "ambos"), (("sma5", 10), ("sma5", 20), ("sma20", 20), ("tiempo", 10)),
        (3, 5, 10), (None, "SPY"),
    ):
        cfgs.append(dict(familia="D_ret_tendencia", tipo="ret", n=n, x=x, filtro=f, salida=sal, h=h, k=k, reserva=res))
    for n, x, f, (sal, h), k, res in itertools.product(
        (20, 40), (0.15, 0.2, 0.25, 0.3), ("tendencia", "ambos"),
        (("sma20", 60), ("tiempo", 60), ("tiempo", 20)), (3, 5, 10), (None, "SPY"),
    ):
        cfgs.append(dict(familia="E_dd_tendencia", tipo="dd", n=n, x=x, filtro=f, salida=sal, h=h, k=k, reserva=res))
    for s, k, res in itertools.product(
        [dict(tipo="rsi", n=2, x=5, salida="sma5", h=10), dict(tipo="racha", n=4, salida="sma5", h=10),
         dict(tipo="z", n=20, x=2.5, salida="sma5", h=10)], (3, 5, 10), (None, "SPY")):
        cfgs.append(dict(familia="F_corta_tendencia", **s, filtro="tendencia", k=k, reserva=res))
    for i, c in enumerate(cfgs):
        c["id"] = 10000 + i
    return cfgs


def ejecutar(cfg: dict) -> tuple[dict, np.ndarray]:
    r = m.backtest(cfg)
    eq = r.equity
    fila = {**cfg}
    for nombre, sel in (("is", eq.index < "2016-01-01"), ("oos", eq.index >= "2015-12-31"), ("tot", slice(None))):
        e = eq[sel]
        tr = r.trades[(r.trades.entrada >= e.index[0]) & (r.trades.entrada <= e.index[-1])]
        for k, v in m.metricas(e, tr, r.exposicion[sel]).items():
            fila[f"{nombre}_{k}"] = float(v)
    return fila, eq.pct_change().fillna(0).to_numpy(np.float32)


def precalcular() -> None:
    m.cargar(), m.liquido(), m.regimen_mercado(), m.sma(200), m.sma(5), m.sma(20), m.rsi(2)
    for n in (10, 20, 40, 60, 126, 252):
        m.maximo(n)
    m.rachas_bajistas(), m.retorno(1), m.retorno(3), m.retorno(5), m.retorno(10), m.desv(20)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    fase2 = "--fase2" in sys.argv
    cfgs = configuraciones_fase2() if fase2 else configuraciones()
    if "--rapido" in sys.argv:
        cfgs = cfgs[::40]
    print(f"{len(cfgs)} configuraciones", flush=True)
    precalcular()
    t = time.time()
    filas, rets = [], []
    with Pool(4) as pool:
        for i, (fila, r) in enumerate(pool.imap(ejecutar, cfgs, chunksize=4)):
            filas.append(fila)
            rets.append(r)
            if i % 50 == 0:
                print(f"{i}/{len(cfgs)}  {time.time() - t:.0f}s", flush=True)
    df = pd.DataFrame(filas)
    sufijo = ("_fase2" if fase2 else "") + ("_rapido" if "--rapido" in sys.argv else "")
    df.to_csv(OUT / f"rejilla{sufijo}.csv", index=False, float_format="%.5f")
    np.save(m.ROOT / "data" / "cache" / "caidas" / f"rejilla_retornos{sufijo}.npy", np.vstack(rets))
    (OUT / f"rejilla{sufijo}_meta.json").write_text(json.dumps(
        {"n_pruebas": len(cfgs), "coste_bps_lado": 10, "is": "2002-01-02..2015-12-31",
         "oos": "2016-01-01..fin", "segundos": round(time.time() - t)}, indent=1))
    print(df.sort_values("is_cagr", ascending=False).head(15)[
        ["id", "familia", "tipo", "n", "x", "filtro", "salida", "h", "k", "is_cagr", "oos_cagr", "tot_maxdd"]].to_string())


if __name__ == "__main__":
    main()
