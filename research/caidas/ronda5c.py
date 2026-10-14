"""Ronda 5c: componente D2 = más entradas, con puntuación CONJUNTA (módulos v2 + técnicas).

D1 exige caída >= 15 % desde el máximo de 20 d y >= 5 módulos. D2 acepta caídas desde
el 10 % si la suma de módulos v2 (11) y banderas técnicas (11) es alta, porque esa suma
ordena monótonamente el retorno de la operación en la búsqueda y en la validación.

Rejilla (selección por CAGR en la búsqueda, oct-2021 →; validación 2002-sep21):
- caída mínima desde máx. 20 d: 10 %, 12 %, 15 %;
- puntuación conjunta mínima T: 9, 10, 11, 12;
- tamaño: base (1 hueco) o convicción w = clip(1 + (total - T) / 3, 0.5, 1.5);
- hueco 1/3 o 1/2, hasta 5 posiciones; salida > SMA10 (máx. 20 d);
- palanca máx. 1 / 1,5 / 2 solo en tendencia (SPY como señal), escalada por volatilidad;
- reserva solo de acciones: ``liq100_tend_minvol``.

Uso: python research/caidas/ronda5c.py
"""

from __future__ import annotations

import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd

import eventos as ev
import motor_v3 as v3
import ronda3 as r3
import tecnicos as tc
from ronda5 import score_tec

OUT = Path(__file__).parent / "resultados"
VENTANAS = {"busq": ("2021-10-01", None), "valid": ("2002-01-02", "2021-10-01"),
            "0225": ("2002-01-02", "2026-01-01"), "total": ("2002-01-02", None)}


def candidatas_d2() -> pd.DataFrame:
    d = r3.cargar()  # eventos en tendencia con score_v2 y meta_pct
    tec = tc.cargar()
    x = d.merge(tec, on=["t", "j"], how="left")
    x["score_tec"] = score_tec(x)
    x["total"] = x.score_v2 + x.score_tec
    x["prio"] = x.total + x.meta_pct.fillna(0.5) * 0.99
    return x


def main() -> None:
    x0 = candidatas_d2()
    filas, curvas = [], []
    t0 = time.time()
    for dd, T, tam, frac, pal in itertools.product((0.10, 0.12, 0.15), (9, 10, 11, 12), ("base", "conviccion"),
                                                    (1 / 3, 1 / 2), (1.0, 1.5, 2.0)):
        y = x0[(x0.dd20 <= -dd) & (x0.total >= T)]
        w = np.ones(len(y)) if tam == "base" else np.clip(1 + (y.total.to_numpy() - T) / 3, 0.5, 1.5)
        w = np.minimum(w, 0.5 / frac)
        fila = {"caida_min": dd, "T": T, "tamaño": tam, "hueco": round(frac, 3), "palanca": pal,
                "candidatas": len(y)}
        for vn, (a, b) in VENTANAS.items():
            r = v3.simular(y.t.to_numpy(), y.j.to_numpy(), y.prio.to_numpy(np.float64), peso=w, K=5, frac=frac,
                           H=20, salida="sma10", reserva_="liq100_tend_minvol", palanca=pal, palanca_off=1.0,
                           vol_objetivo=0.15 if pal > 1 else None, inicio=a, fin=b)
            fila |= {f"{vn}_{k}": val for k, val in r.items() if k not in ("equity", "serie_expo", "serie_lev")}
            if vn == "total":
                curvas.append(r["equity"].pct_change().fillna(0).to_numpy(np.float32))
        filas.append(fila)
    df = pd.DataFrame(filas)
    df.to_csv(OUT / "ronda5c.csv", index=False, float_format="%.5f")
    np.save(ev.SALIDA.with_name("ronda5c_retornos.npy"), np.vstack(curvas))
    print(f"{len(df)} carteras en {time.time() - t0:.0f}s")
    pd.set_option("display.width", 260)
    cols = ["caida_min", "T", "tamaño", "hueco", "palanca", "candidatas", "busq_cagr", "busq_maxdd", "valid_cagr",
            "valid_maxdd", "0225_cagr", "total_cagr", "total_maxdd", "total_sharpe", "total_trades_año"]
    for pal in (1.0, 1.5, 2.0):
        g = df[df.palanca == pal].sort_values("busq_cagr", ascending=False)
        print(f"\nD2 · palanca {pal} (top 6 por CAGR de búsqueda)")
        print(g[cols].head(6).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
