"""Ronda 5b: más capital en las mejores entradas técnicas (reserva solo de acciones).

Sobre D1 y C1 con la puntuación técnica (``ronda5.score_tec``):
- uso de la técnica: filtro (score_tec >= k) y prioridad (se suma a la de módulos);
- tamaño por convicción técnica (pre-especificado): w_tec = clip((score_tec - 2) / 3, 0.5, 2)
  (≤3 banderas: medio hueco; 5: un hueco; ≥8: dos huecos), multiplicado por la
  convicción de módulos de D1; ninguna posición > 50 % del equity;
- hueco de 1/3 o 1/2 del equity, hasta 5 posiciones;
- palanca máxima 1, 1,5 o 2 solo con SPY > SMA200 (SPY solo como SEÑAL, no se compra),
  escalada por la volatilidad de SPY (objetivo 15 %); deuda al 5 %;
- reserva: ``liq100_dv`` (las 100 acciones más líquidas, ponderadas por volumen en
  dólares). Sin ETFs ni efectivo.

Selección por CAGR en la búsqueda (oct-2021 →); validación 2002-sep21.

Uso: python research/caidas/ronda5b.py
"""

from __future__ import annotations

import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd

import eventos as ev
import motor_v3 as v3
from ronda5 import componentes_tec

OUT = Path(__file__).parent / "resultados"
VENTANAS = {"busq": ("2021-10-01", None), "valid": ("2002-01-02", "2021-10-01"),
            "0225": ("2002-01-02", "2026-01-01"), "total": ("2002-01-02", None)}


def preparar(x: pd.DataFrame, conv: bool, kmin: int, uso: str, tam: str, frac: float):
    y = x[x.score_tec >= kmin].copy()
    prio = y.prio.to_numpy(np.float64)
    if uso == "prioridad":
        prio = prio + y.score_tec.to_numpy()
    w = np.minimum(y.peso.to_numpy(), 1.5) if conv else np.ones(len(y))
    if tam == "conviccion_tec":
        w = w * np.clip((y.score_tec.to_numpy() - 2) / 3, 0.5, 2.0)
    w = np.minimum(w, 0.5 / frac)  # ninguna posición > 50 % del equity
    return y.assign(prio=prio), w


def main() -> None:
    comp = componentes_tec()
    filas, curvas = [], []
    t0 = time.time()
    for (nombre, (x, salida, conv)), (kmin, uso), tam, frac, pal in itertools.product(
        comp.items(), ((0, "filtro"), (2, "filtro"), (2, "prioridad"), (3, "filtro"), (0, "prioridad")),
        ("base", "conviccion_tec"), (1 / 3, 1 / 2), (1.0, 1.5, 2.0),
    ):
        y, w = preparar(x, conv, kmin, uso, tam, frac)
        fila = {"componente": nombre, "kmin": kmin, "uso": uso, "tamaño": tam, "hueco": round(frac, 3),
                "palanca": pal}
        for vn, (a, b) in VENTANAS.items():
            r = v3.simular(y.t.to_numpy(), y.j.to_numpy(), y.prio.to_numpy(np.float64), peso=w, K=5, frac=frac,
                           H=20, salida=salida, reserva_="liq100_dv", palanca=pal, palanca_off=1.0,
                           vol_objetivo=0.15 if pal > 1 else None, inicio=a, fin=b)
            fila |= {f"{vn}_{k}": val for k, val in r.items() if k not in ("equity", "serie_expo", "serie_lev")}
            if vn == "total":
                curvas.append(r["equity"].pct_change().fillna(0).to_numpy(np.float32))
        filas.append(fila)
    df = pd.DataFrame(filas)
    df.to_csv(OUT / "ronda5b.csv", index=False, float_format="%.5f")
    np.save(ev.SALIDA.with_name("ronda5b_retornos.npy"), np.vstack(curvas))
    print(f"{len(df)} carteras en {time.time() - t0:.0f}s")
    pd.set_option("display.width", 270)
    cols = ["componente", "kmin", "uso", "tamaño", "hueco", "palanca", "busq_cagr", "busq_maxdd", "valid_cagr",
            "valid_maxdd", "0225_cagr", "total_cagr", "total_maxdd", "total_sharpe", "total_palanca_media"]
    for c in ("D1", "C1"):
        for pal in (1.0, 1.5, 2.0):
            g = df[(df.componente == c) & (df.palanca == pal)].sort_values("busq_cagr", ascending=False)
            print(f"\n{c} · palanca {pal} (top 4 por CAGR de búsqueda)")
            print(g[cols].head(4).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
