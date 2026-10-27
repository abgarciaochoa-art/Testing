"""Noise Band como complemento intradía del nivel 1 de caídas (``CAI_N1``).

El nivel 1 está siempre invertido en acciones. Noise Band se superpone con futuros del
S&P 500 solo dentro del día (no duerme, así que no hay palanca por la noche): la
rentabilidad diaria de la mezcla es la del nivel 1 más ``peso`` × la de Noise Band, con
el mismo capital como garantía. Pesos fijados de antemano: 0,5 y 1.

Ventana común: 24-ene-2005 → 13-may-2020 (fin de los datos intradía), con el calendario del
nivel 1: en los 23 días sin datos intradía Noise Band no opera. Toda la ventana está
fuera de la muestra con la que se eligieron las reglas del nivel 1; los parámetros de
Noise Band son los del artículo, que los eligió con 2007-2024.

Salida: resultados/{combinar.csv, combinar_crisis.csv, combinar_anual.csv, combinar.png}

Uso: python research/noise_band/combinar.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import graficas as gr
import motor as nb

OUT = Path(__file__).parent / "resultados"
DIARIO = Path(__file__).resolve().parents[1] / "caidas" / "dossier" / "datos" / "diario.csv"
COSTES = {"artículo (0,0045 $ por acción)": dict(coste_por_accion=0.0045),
          "SPY a mercado (0,0085 $ por acción)": dict(coste_por_accion=0.0085),
          "1 pb": dict(coste_bps=1.0), "2 pb": dict(coste_bps=2.0)}
BASE = "SPY a mercado (0,0085 $ por acción)"
CRISIS = [("Crisis financiera", "2007-10-09", "2009-03-09"), ("Flash crash", "2010-04-23", "2010-07-02"),
          ("Deuda de EE. UU. y del euro", "2011-04-29", "2011-10-03"), ("China y petróleo", "2015-05-21", "2016-02-11"),
          ("Cuarto trimestre de 2018", "2018-09-20", "2018-12-24"), ("Covid", "2020-02-19", "2020-03-23")]


def metricas(r: pd.Series) -> dict:
    eq = (1 + r).cumprod()
    años = (r.index[-1] - r.index[0]).days / 365.25
    dd = (eq / eq.cummax() - 1).min()
    cagr = eq.iloc[-1] ** (1 / años) - 1
    return {"cagr": cagr, "peor_caida": dd, "sharpe": r.mean() / r.std() * np.sqrt(252), "calmar": cagr / abs(dd)}


def main() -> None:
    d = nb.cargar()
    base = pd.read_csv(DIARIO, index_col=0, parse_dates=True)["CAI_N1"]
    complementos = {
        "artículo, largo y corto, 1×": dict(lados=2, lev=nb.palanca(d)),
        "artículo, largo y corto, volatilidad 2 % (máx. 4×)": dict(lados=2, lev=nb.palanca(d, 0.02)),
        "solo largo con las salidas del artículo, 1×": dict(lados=1, lev=nb.palanca(d)),
        "solo largo hasta el cierre, 1×": dict(lados=1, trail_vwap=False, salida_banda=0, lev=nb.palanca(d)),
    }
    # calendario del nivel 1: los 23 días sin datos intradía (entre ellos 9 y 16 de marzo de 2020,
    # con la apertura suspendida) cuentan para el nivel 1 y Noise Band no opera en ellos
    comun = base.index[(base.index >= "2005-01-24") & (base.index <= d["fechas"][-1])]
    series = {}
    filas = []
    for nombre, kw in complementos.items():
        for c, kc in COSTES.items():
            r = nb.simular(d, **kc, **kw)["equity"].pct_change().reindex(comun).fillna(0.0)
            for peso in (0.0, 0.5, 1.0):
                mezcla = base[comun] + peso * r[comun]
                filas.append({"complemento": nombre, "costes": c, "peso": peso, **metricas(mezcla),
                              "corr_mensual_con_N1": ((1 + r[comun]).resample("ME").prod().corr(
                                  (1 + base[comun]).resample("ME").prod()))})
                if c == BASE:
                    series[(nombre, peso)] = mezcla
            if c == BASE:
                series[(nombre, "solo")] = r[comun]
    df = pd.DataFrame(filas)
    df = df[~((df.peso == 0) & (df.duplicated(["peso"])))]
    df.to_csv(OUT / "combinar.csv", index=False, float_format="%.5f")

    principal = "artículo, largo y corto, 1×"
    curvas = {"nivel 1 solo": series[(principal, 0.0)], "nivel 1 + Noise Band (peso 0,5)": series[(principal, 0.5)],
              "nivel 1 + Noise Band (peso 1)": series[(principal, 1.0)], "Noise Band solo (1×)": series[(principal, "solo")]}
    eqs = {k: (1 + v).cumprod() for k, v in curvas.items()}
    crisis = pd.DataFrame({k: [e.asof(pd.Timestamp(b)) / e.asof(pd.Timestamp(a)) - 1 for _, a, b in CRISIS]
                           for k, e in eqs.items()}, index=[n for n, _, _ in CRISIS])
    crisis.to_csv(OUT / "combinar_crisis.csv", float_format="%.4f")
    anual = pd.DataFrame({k: e.resample("YE").last() for k, e in eqs.items()})
    anual = anual / pd.concat([pd.DataFrame({k: [1.0] for k in eqs}), anual.iloc[:-1]]).to_numpy() - 1
    anual.index = anual.index.year
    anual.to_csv(OUT / "combinar_anual.csv", float_format="%.4f")

    pd.set_option("display.width", 220)
    print(df.round(4).to_string(index=False))
    print((crisis * 100).round(1).to_string())
    print((anual * 100).round(1).to_string())

    gr.estilo()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11.5, 7.6), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    colores = {"nivel 1 solo": (gr.AZUL, 2.0), "nivel 1 + Noise Band (peso 0,5)": (gr.AQUA, 1.8),
               "nivel 1 + Noise Band (peso 1)": (gr.NARANJA, 2.0), "Noise Band solo (1×)": (gr.GRIS, 1.6)}
    gr.curvas(a1, {k: (e, *colores[k]) for k, e in eqs.items()})
    for k, e in eqs.items():
        if k != "Noise Band solo (1×)":
            a2.plot(e.index, (e / e.cummax() - 1) * 100, color=colores[k][0], lw=1.2)
    a1.set_title("Nivel 1 de caídas con Noise Band intradía encima: 100.000 $ en 2005, costes de SPY a mercado")
    a2.set_ylabel("caída desde máximos (%)")
    a2.grid(axis="y")
    fig.tight_layout()
    fig.subplots_adjust(right=0.78)
    fig.savefig(OUT / "combinar.png", dpi=130)


if __name__ == "__main__":
    main()
