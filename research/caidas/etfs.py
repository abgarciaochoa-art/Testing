"""¿Rebotan las caídas rápidas de ETFs (sectores, países, estilos, materias primas)?

Una caída de un ETF es, por construcción, una caída de GRUPO: el tipo de caída que en
acciones rebota mejor (``modulos.py``). Lista fijada por clase de activo antes de mirar
resultados (ETFs líquidos con historia desde 2002 o antes, o lo antes posible):

- sectores SPDR: XLB XLE XLF XLI XLK XLP XLU XLV XLY
- industrias: IBB SOXX IYT IYR IGV ITB
- países y regiones: EWA EWC EWG EWH EWJ EWS EWT EWU EWY EWZ EEM EFA
- tamaño/estilo: IWM MDY IJR IWD IWF
- materias primas: GLD SLV

Estudio de eventos: retorno desde la apertura de t+1 hasta el cierre de t+h, en exceso
sobre el retorno medio incondicional del mismo ETF en la misma ventana (búsqueda
oct-2021 → / validación 2002 → sep-2021), con t de Newey-West sobre la media diaria.

Uso: python research/caidas/etfs.py
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

import motor as m
import reservas as rs
from estudio import nw_t

OUT = Path(__file__).parent / "resultados"
ETFS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "IBB", "SOXX", "IYT", "IYR", "IGV", "ITB",
        "EWA", "EWC", "EWG", "EWH", "EWJ", "EWS", "EWT", "EWU", "EWY", "EWZ", "EEM", "EFA",
        "IWM", "MDY", "IJR", "IWD", "IWF", "GLD", "SLV"]
CORTE = np.datetime64("2021-10-01")
INICIO = np.datetime64("2002-01-02")


def panel_etfs() -> tuple[np.ndarray, np.ndarray]:
    """Matrices apertura/cierre ajustados T×E alineadas al panel."""
    O, C = [], []
    for e in ETFS:
        o, c = rs.etf(e)
        O.append(o)
        C.append(c)
    return np.column_stack(O), np.column_stack(C)


def main() -> None:
    p = m.cargar()
    O, C = panel_etfs()
    T = len(C)
    df_c = pd.DataFrame(C)
    sma200 = df_c.rolling(200).mean().to_numpy()
    filas = []
    fwd = {}
    for h in (5, 10, 20):
        f = np.full(C.shape, np.nan)
        f[: T - h] = C[h:] / O[1 : T - h + 1] - 1
        fwd[h] = f
    fechas = p.fechas
    for n, x, filtro in itertools.product((3, 5, 10), (0.03, 0.05, 0.08, 0.12), ("ninguno", "tendencia")):
        r = np.full(C.shape, np.nan)
        r[n:] = C[n:] / C[:-n] - 1
        with np.errstate(invalid="ignore"):
            ev = r <= -x
            if filtro == "tendencia":
                ev &= C > sma200
        for w, sel in (("busq", fechas >= CORTE), ("valid", (fechas >= INICIO) & (fechas < CORTE))):
            fila = {"n": n, "x": x, "filtro": filtro, "ventana": w, "eventos": int(ev[sel].sum())}
            for h in (5, 10, 20):
                base = np.nanmean(fwd[h][sel], axis=0)  # media incondicional por ETF en la ventana
                exc = np.where(ev[sel], fwd[h][sel] - base, np.nan)
                with np.errstate(invalid="ignore"):
                    diaria = np.nanmean(exc, axis=1)
                fila[f"exc{h}"] = float(np.nanmean(diaria))
                fila[f"t{h}"] = nw_t(diaria, h)
            filas.append(fila)
    df = pd.DataFrame(filas)
    df.to_csv(OUT / "etfs_eventos.csv", index=False, float_format="%.5f")
    pd.set_option("display.width", 220)
    v = df.pivot_table(index=["n", "x", "filtro"], columns="ventana", values=["eventos", "exc5", "t5", "exc10", "t10"])
    for c in ("exc5", "exc10"):
        v[(c, "busq")] *= 100
        v[(c, "valid")] *= 100
    print(v.round(2).to_string())


if __name__ == "__main__":
    main()
