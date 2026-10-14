"""Estrategia propia de pares ("gemelas", tipo KO/PEP): cuando una acción se separa a
la baja de su par más correlacionado, ¿converge el spread?

Para todos los miembros con gemela de correlación >= c (252 d, recalculada cada mes,
PIT) y spread z (60 d, log-cociente) <= -z, se mide a h días desde la apertura de t+1:

- ``larga``: retorno en exceso de la rezagada sobre el equiponderado (solo largos);
- ``spread``: rezagada - gemela (largo/corto, neutral al sector).

Se evalúan ventana de BÚSQUEDA (oct-2021 →) y VALIDACIÓN (2002 → sep-2021).

Uso: python research/caidas/gemelas.py
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

import motor as m
import rasgos as rg
from estudio import nw_t

OUT = Path(__file__).parent / "resultados"
CORTE = np.datetime64("2021-10-01")


def main() -> None:
    p = m.cargar()
    R = rg.cargar_rasgos()
    T, N = p.close.shape
    gem = R["pares"][:, :, 0]
    base = p.miembro & m.liquido()
    filas = []
    fwd, exc, spr = {}, {}, {}
    for h in (5, 10, 20, 60):
        f = np.full((T, N), np.nan)
        f[: T - h] = p.close_ff[h:] / p.open[1 : T - h + 1] - 1
        mu = np.nanmean(np.where(base, f, np.nan), axis=1, keepdims=True)
        fwd[h], exc[h] = f, f - mu
        fg = np.take_along_axis(f, np.maximum(gem, 0), 1)
        spr[h] = np.where(gem >= 0, f - fg, np.nan)
    busq = (p.fechas >= CORTE)[:, None]
    for c, z, filtro in itertools.product((0.6, 0.7, 0.8), (-1.5, -2.0, -2.5, -3.0), ("todas", "tendencia")):
        with np.errstate(invalid="ignore"):
            ev = base & (R["gem_corr"] >= c) & (R["gem_z"] <= z)
            if filtro == "tendencia":
                ev &= R["sobre_sma200"] == 1
        for nombre, sel in (("busq", ev & busq), ("valid", ev & ~busq)):
            fila = {"corr_min": c, "z_max": z, "filtro": filtro, "ventana": nombre, "eventos": int(sel.sum())}
            for h in (5, 10, 20, 60):
                for k, mat in (("larga", exc[h]), ("spread", spr[h])):
                    with np.errstate(invalid="ignore"):
                        diaria = np.nanmean(np.where(sel, mat, np.nan), axis=1)
                    fila[f"{k}{h}"] = np.nanmean(diaria)
                    fila[f"t_{k}{h}"] = nw_t(diaria, h)
            filas.append(fila)
    df = pd.DataFrame(filas)
    df.to_csv(OUT / "gemelas.csv", index=False, float_format="%.5f")
    pd.set_option("display.width", 250)
    cols = ["corr_min", "z_max", "filtro", "ventana", "eventos", "larga5", "larga20", "t_larga20",
            "spread5", "spread20", "t_spread20", "spread60", "t_spread60"]
    r = df[cols].copy()
    for c in cols[5:]:
        if not c.startswith("t_"):
            r[c] = r[c] * 100
    print(r.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
