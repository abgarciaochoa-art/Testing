"""Ronda 4: reserva, apalancamiento puntual (overlay) y capacidad.

Componentes (sin tocar sus reglas de señal): D1 (caída >=15 % desde máx. 20 d, >=5
módulos, convicción con tope 50 %) y C1 (caída >=7 % en 5 d, >=6 módulos v1, K=5).

Dimensiones:
- reserva: spy, spy_tend_shy, spy_tend_ief (las momentum se descartaron en
  ``reservas.py``: fallan en la validación);
- palanca máxima: 1,0 · 1,25 · 1,5 · 2,0 (deuda al 5 % anual), en todo momento o solo
  con SPY > SMA200 (fuera de tendencia, 1,0);
- capacidad: (K=3, hueco 1/3), (K=5, hueco 1/5), (K=5, hueco 1/3), (K=8, hueco 1/4).

Reglas de selección fijadas antes de mirar: máximo CAGR en la búsqueda con caída
máxima en la búsqueda >= -30 % (A, prudente) o >= -40 % (B, agresiva).

Uso: python research/caidas/ronda4.py
"""

from __future__ import annotations

import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd

import eventos as ev
import finalistas3 as f3
import motor_v3 as v3
import ronda3 as r3

OUT = Path(__file__).parent / "resultados"
VENTANAS = {"busq": ("2021-10-01", None), "valid": ("2002-01-02", "2021-10-01"),
            "0225": ("2002-01-02", "2026-01-01"), "total": ("2002-01-02", None)}


def componentes() -> dict:
    d = r3.cargar()
    e = ev.cargar()
    d1 = f3.candidatas_d1(d)
    d1["peso"] = np.minimum(d1.peso, 1.5)  # tope 50 % con hueco 1/3 (se reescala abajo)
    c1 = f3.candidatas_c1(d, e)
    return {"D1": (d1, "sma10", True), "C1": (c1, "rsi2", False)}


def main() -> None:
    comp = componentes()
    filas, curvas = [], []
    t0 = time.time()
    for (nombre, (x, salida, conv)), res, (pal, pal_off), (K, frac) in itertools.product(
        comp.items(), ("spy", "spy_tend_shy", "spy_tend_ief"),
        ((1.0, 1.0), (1.25, 1.25), (1.5, 1.5), (2.0, 2.0), (1.25, 1.0), (1.5, 1.0), (2.0, 1.0)),
        ((3, 1 / 3), (5, 1 / 5), (5, 1 / 3), (8, 1 / 4)),
    ):
        w = x.peso.to_numpy() if conv else None
        if w is not None:
            w = np.minimum(w, 0.5 / frac)  # ninguna posición > 50 % del equity
        fila = {"componente": nombre, "reserva": res, "palanca": pal, "palanca_off": pal_off, "K": K,
                "hueco": round(frac, 3)}
        for v, (a, b) in VENTANAS.items():
            r = v3.simular(x.t.to_numpy(), x.j.to_numpy(), x.prio.to_numpy(np.float64), peso=w, K=K, frac=frac,
                           H=20, salida=salida, reserva_=res, palanca=pal, palanca_off=pal_off, inicio=a, fin=b)
            fila |= {f"{v}_{k}": val for k, val in r.items() if k != "equity"}
            if v == "total":
                curvas.append(r["equity"].pct_change().fillna(0).to_numpy(np.float32))
        filas.append(fila)
    df = pd.DataFrame(filas)
    df.to_csv(OUT / "ronda4.csv", index=False, float_format="%.5f")
    np.save(ev.SALIDA.with_name("ronda4_retornos.npy"), np.vstack(curvas))
    print(f"{len(df)} carteras en {time.time() - t0:.0f}s")
    pd.set_option("display.width", 260)
    cols = ["componente", "reserva", "palanca", "palanca_off", "K", "hueco", "busq_cagr", "busq_maxdd", "valid_cagr", "valid_maxdd",
            "0225_cagr", "total_cagr", "total_maxdd", "total_sharpe", "total_palanca_media"]
    for regla, lim in (("A (DD búsq >= -30 %)", -0.30), ("B (DD búsq >= -40 %)", -0.40)):
        for c in ("D1", "C1"):
            g = df[(df.componente == c) & (df.busq_maxdd >= lim)].sort_values("busq_cagr", ascending=False)
            print(f"\nRegla {regla} · {c}")
            print(g[cols].head(5).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
