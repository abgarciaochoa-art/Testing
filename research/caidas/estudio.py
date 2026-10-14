"""Estudio de eventos: ¿qué tipo de caída predice rebote, y a qué horizonte?

Para cada definición de caída se mide el retorno desde la apertura de t+1 hasta el
cierre de t+h, **en exceso sobre la media equiponderada de los miembros** en la
misma ventana (aísla el efecto "caída" del movimiento del mercado). Se promedia
primero por fecha (cada día pesa igual) y la t se calcula con Newey-West (h lags),
porque las ventanas se solapan.

Uso: python research/caidas/estudio.py  -> resultados/estudio_eventos.csv
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

import motor as m

OUT = Path(__file__).parent / "resultados"
HORIZONTES = (5, 10, 20, 60, 126)
PERIODOS = {"2002-2008": ("2002-01-01", "2009-01-01"), "2009-2015": ("2009-01-01", "2016-01-01"),
            "2016-2026": ("2016-01-01", "2027-01-01"), "total": ("2002-01-01", "2027-01-01")}


def nw_t(x: np.ndarray, lags: int) -> float:
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 30:
        return np.nan
    e = x - x.mean()
    v = e @ e / n
    for L in range(1, min(lags, n - 1) + 1):
        v += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    return x.mean() / np.sqrt(v / n)


def forward_excess(h: int) -> np.ndarray:
    p = m.cargar()
    T = len(p.fechas)
    fwd = np.full(p.close.shape, np.nan)
    # entrada apertura t+1, salida cierre t+h
    fwd[: T - h] = p.close_ff[h:] / p.open[1 : T - h + 1] - 1
    base = p.miembro & m.liquido()
    mean = np.nanmean(np.where(base, fwd, np.nan), axis=1, keepdims=True)
    return fwd - mean


SENALES = (
    [dict(tipo="dd", n=n, x=x) for n, x in itertools.product((20, 60, 126, 252), (0.1, 0.2, 0.3, 0.5))]
    + [dict(tipo="racha", n=n) for n in (3, 4, 5, 6, 7)]
    + [dict(tipo="rsi", n=2, x=x) for x in (2, 5, 10)] + [dict(tipo="rsi", n=14, x=x) for x in (20, 30)]
    + [dict(tipo="ret", n=n, x=x) for n, x in ((1, 0.05), (1, 0.1), (5, 0.1), (5, 0.15), (20, 0.15), (20, 0.25))]
    + [dict(tipo="z", n=20, x=x) for x in (2.0, 2.5, 3.0)]
)
FILTROS = ("ninguno", "tendencia", "mercado")


def etiqueta(s: dict) -> str:
    return "_".join(f"{k}={v}" for k, v in s.items())


def main() -> None:
    OUT.mkdir(exist_ok=True)
    p = m.cargar()
    fechas = pd.DatetimeIndex(p.fechas)
    fx = {h: forward_excess(h) for h in HORIZONTES}
    filas = []
    for s, f in itertools.product(SENALES, FILTROS):
        ent, _ = m.senal({**s, "filtro": f})
        n_ev = ent.sum(axis=1)
        for per, (a, b) in PERIODOS.items():
            sel = (fechas >= a) & (fechas < b)
            fila = {"senal": etiqueta(s), "filtro": f, "periodo": per,
                    "eventos": int(n_ev[sel].sum()), "dias_con_evento": int((n_ev[sel] > 0).sum())}
            for h in HORIZONTES:
                x = np.where(ent[sel], fx[h][sel], np.nan)
                with np.errstate(invalid="ignore"):
                    diaria = np.nanmean(x, axis=1)  # media por fecha
                fila[f"exc_{h}d"] = np.nanmean(diaria)
                fila[f"t_{h}d"] = nw_t(diaria, h)
            filas.append(fila)
        print(etiqueta(s), f, "ok", flush=True)
    df = pd.DataFrame(filas)
    df.to_csv(OUT / "estudio_eventos.csv", index=False, float_format="%.5f")
    print(df[df.periodo == "total"].sort_values("t_20d", ascending=False).head(25).to_string())


if __name__ == "__main__":
    main()
