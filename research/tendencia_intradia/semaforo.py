"""Semáforo de tendencia intradía: reglas simples, calculables en cada vela y replicables en
cualquier plataforma (ver ``semaforo.pine``).

En cada vela cerrada se miden cuatro cosas:
  dirección   signo de (precio − apertura del día)
  sigma       (precio / apertura − 1) dividido por el movimiento absoluto medio desde la
              apertura a esa misma hora en los 14 días anteriores. |sigma| < 1: dentro del
              ruido normal del valor a esa hora.
  rvol        volumen acumulado del día / media de los 20 días anteriores a la misma hora
  vwap        el precio está del lado de la dirección respecto a la VWAP del día
  mercado     el SPY (o el índice) va en la misma dirección desde su apertura (opcional; en
              las pruebas no mejora y por defecto no se usa)

Estados (con signo: positivo alcista, negativo bajista):
  0 rojo         |sigma| < 1: sin tendencia; el movimiento cabe en el ruido normal
  1 ámbar        |sigma| ≥ 1, pero sin participación (rvol < 2) o con el precio al otro lado
                 de la VWAP: se mueve, pero no hay ventaja en seguirlo
  2 verde        |sigma| ≥ 1, rvol ≥ 2 y del lado bueno de la VWAP ("valor en juego")
  3 verde fuerte lo mismo con rvol ≥ 3

Los umbrales salen de la rejilla de ``robustez.py``: el volumen relativo es lo que separa los
movimientos que siguen de los que no; el tamaño del movimiento y el mercado añaden poco.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

UMBRALES = {"sigma": 1.0, "rvol": 2.0, "sigma_f": 1.0, "rvol_f": 3.0, "dias_ruido": 14,
            "dias_rvol": 20}
NOMBRES = {0: "rojo", 1: "ámbar", 2: "verde", 3: "verde fuerte"}


def matrices(P: dict, mercado: dict | None = None, u: dict = UMBRALES) -> dict:
    """Rasgos del semáforo en cada vela (días × velas), con datos hasta su cierre."""
    O, H, L, C, V = (P[k] for k in "OHLCV")
    o0 = O[:, :1]
    mov = C / o0 - 1
    base = pd.DataFrame(np.abs(mov)).rolling(u["dias_ruido"]).mean().shift().to_numpy()
    sigma = mov / np.where(base > 0, base, np.nan)
    cv = np.cumsum(V, 1)
    media = pd.DataFrame(cv).rolling(u["dias_rvol"]).mean().shift().to_numpy()
    rvol = cv / np.where(media > 0, media, np.nan)
    tp = (H + L + C) / 3
    vwap = np.where(cv > 0, np.cumsum(tp * V, 1) / np.where(cv > 0, cv, 1),
                    np.cumsum(tp, 1) / np.arange(1, C.shape[1] + 1))
    M = {"sigma": sigma, "rvol": rvol, "lado_vwap": np.sign(C - vwap), "dir": np.sign(C - o0)}
    if mercado is not None:
        fm = pd.to_datetime(mercado["fechas"])
        mk = pd.DataFrame(np.sign(mercado["C"] - mercado["O"][:, :1]), index=fm)
        M["mercado"] = mk.reindex(pd.to_datetime(P["fechas"])).to_numpy()
    return M


def estado(M: dict, u: dict = UMBRALES, usar_mercado: bool = False) -> np.ndarray:
    s, rv, d = M["sigma"], np.nan_to_num(M["rvol"]), M["dir"]
    fuera = np.abs(s) >= u["sigma"]
    vw = M["lado_vwap"] == d
    e = np.where(fuera, 1, 0)
    e = np.where(fuera & (rv >= u["rvol"]) & vw, 2, e)
    fuerte = (np.abs(s) >= u["sigma_f"]) & (rv >= u["rvol_f"]) & vw
    if usar_mercado and "mercado" in M:
        fuerte &= M["mercado"] == d
    e = np.where(fuerte, 3, e)
    e = np.where(np.isnan(s), 0, e)
    return (e * d).astype(int)
