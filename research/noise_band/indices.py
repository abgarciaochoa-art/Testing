"""Noise Band en otros índices de EE. UU.: Nasdaq-100 y Russell 2000 (OANDA, 2005 → may-2020).

Las mismas reglas que en el S&P 500, sin tocar nada: el artículo, la fuerza ≥ 0,3 y la
puntuación aprendida **solo con operaciones del S&P 500** y en dos mitades (cada mitad del
índice se opera con el modelo entrenado con el S&P 500 de la otra mitad): fuera de muestra por
índice y por fechas. Largo y corto, sin palanca. Costes por acción de su ETF a mercado (QQQ ≈
Nasdaq-100 / 40, IWM ≈ Russell 2000 / 10; el de QQQ fue caro en 2005-2009, ~2 pb por lado) y,
para comparar con el S&P 500, 0,55 pb por lado fijos. También la cartera de los tres a partes
iguales.

Salida: resultados/ronda2_indices.csv

Uso: python research/noise_band/indices.py (antes, ``datos.py`` prepara los paneles)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import datos as dt
import elegida as el
import estadistica as es
import eventos as ev
import modelos as mo
import motor as nb
import rasgos as rg

OUT = Path(__file__).parent / "resultados"
INDICES = {"S&P 500": ("SPX500_USD", 10.0), "Nasdaq-100": ("NAS100_USD", 40.0), "Russell 2000": ("US2000_USD", 10.0)}
COSTES = {"ETF a mercado (0,0085 $ por acción)": dict(coste_por_accion=0.0085),
          "0,55 pb por lado": dict(coste_por_accion=None, coste_bps=0.55)}
VENTANAS = {"total": ("2005-01-24", None), "validacion": mo.VALIDACION, "busqueda": mo.BUSQUEDA}


def main() -> None:
    spx = nb.cargar()
    y_spx = ev.tabla(spx, dict(np.load(rg.SALIDA)))
    series, filas = {}, []
    for indice, (inst, divisor) in INDICES.items():
        d = nb.cargar(ruta=dt.ruta_panel(inst), divisor=divisor)
        R = rg.construir(d)
        # se entrena con las operaciones del S&P 500 y se predice con los rasgos de este índice
        punt = mo.cruzado(d, y_spx, R, "lineal", mo.RASGOS, "filtro", mo.DOS_MITADES)
        reglas = {"artículo": (None, None), "fuerza ≥ 0,3": el.filtro_fuerza(d), "puntuación del S&P 500": punt}
        for costes, kc in COSTES.items():
            for regla, (a, b) in reglas.items():
                r = nb.simular(d, lados=2, mult_largo=a, mult_corto=b, **kc)
                ret = r["equity"].pct_change().fillna(0.0)
                series[(indice, costes, regla)] = ret
                filas.append({"indice": indice, "costes": costes, "regla": regla,
                              **el.fila_metricas(ret, VENTANAS, r["operaciones"])})
    for costes in COSTES:
        for regla in ("artículo", "fuerza ≥ 0,3", "puntuación del S&P 500"):
            cartera = pd.concat([series[(i, costes, regla)] for i in INDICES], axis=1).fillna(0.0).mean(1)
            filas.append({"indice": "los tres a partes iguales", "costes": costes, "regla": regla,
                          **el.fila_metricas(cartera, VENTANAS)})
    df = pd.DataFrame(filas)
    df.to_csv(OUT / "ronda2_indices.csv", index=False, float_format="%.5f")
    pd.set_option("display.width", 250)
    cols = ["indice", "costes", "regla", "total_cagr", "total_peor_caida", "total_sharpe", "validacion_sharpe",
            "busqueda_sharpe", "total_operaciones_año", "total_ret_medio_pb"]
    print(df[cols].round(3).to_string(index=False))
    spx_r = es.tramo(series[("S&P 500", "0,55 pb por lado", "artículo")], *VENTANAS["total"])
    for i in ("Nasdaq-100", "Russell 2000"):
        x = es.tramo(series[(i, "0,55 pb por lado", "artículo")], *VENTANAS["total"])
        print(f"correlación diaria del artículo, {i} con el S&P 500: {x.corr(spx_r):.2f}")


if __name__ == "__main__":
    main()
