"""Noise Band · ronda 1: el artículo, solo largo con stop y objetivo, y dejar correr.

Variantes fijadas de antemano (todas con los parámetros del artículo: sigma de 14 días,
controles a las HH:00 y HH:30, cono ajustado por el hueco de la noche):

- ``articulo``: largo y corto; sale si vuelve dentro del cono o cruza la VWAP (stop móvil);
  cierra al final del día.
- ``largo_articulo``: lo mismo, solo largo.
- ``largo_stop_banda``: solo largo; stop dentro de la vela en la banda inferior del cono, sin
  objetivo, cierre al final del día.
- ``largo_stop_vertice``: ídem con el stop en el vértice del cono (max(apertura, cierre
  anterior)).
- ``largo_stop_objetivo``: stop en la banda inferior y objetivo de +2 % (casi nunca se toca).
- ``largo_hasta_cierre``: solo largo, sin stop, hasta el cierre (control).
- ``largo_dejar_correr``: solo largo y la posición duerme: sale cuando en un control el
  precio rompe el cono por abajo; sin objetivo.
- ``largo_dejar_correr_stop``: ídem con la salida como orden stop en la banda inferior.
- ``*_continua``: la entrada es una orden stop en la banda superior, en cualquier minuto.

Referencias: siempre largo de apertura a cierre, comprar y mantener el índice (sin
dividendos) y SPY con dividendos (panel diario).

Tamaño: 1× (todo el capital cuando hay posición) y el del artículo (volatilidad objetivo del
2 % diario con la de 14 días, máximo 4×). Costes por lado: 0; 0,5; 1 y 2 pb; los del artículo
(0,0045 $ por acción de SPY: 0,13-0,67 pb) y, como base, SPY con órdenes a mercado (0,0035 $
de comisión + medio céntimo de horquilla = 0,0085 $ por acción: 0,25-1,3 pb, 0,55 de media).
Ventanas: búsqueda 2013 → may-2020 (en dos mitades), validación 2005-2012 y desde 2007 (el
inicio del artículo).

Salida: resultados/{ronda1.csv, ronda1_anual.csv, ronda1_operaciones.csv, curvas.png}

Uso: python research/noise_band/ronda1.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import graficas as gr
import motor as nb

OUT = Path(__file__).parent / "resultados"
ROOT = Path(__file__).resolve().parents[2]
VENTANAS = {"total": ("2005-01-24", None), "desde_2007": ("2007-01-03", None),
            "validacion": ("2005-01-24", "2013-01-01"), "busqueda": ("2013-01-01", None),
            "busqueda_1": ("2013-01-01", "2017-01-01"), "busqueda_2": ("2017-01-01", None)}
COSTES = {"0 pb": dict(coste_bps=0.0), "0,5 pb": dict(coste_bps=0.5), "1 pb": dict(coste_bps=1.0),
          "2 pb": dict(coste_bps=2.0), "artículo": dict(coste_por_accion=0.0045),
          "SPY a mercado": dict(coste_por_accion=0.0085)}
BASE = "SPY a mercado"

VARIANTES = {
    "articulo": dict(lados=2, trail_vwap=True, salida_banda=1),
    "largo_articulo": dict(lados=1, trail_vwap=True, salida_banda=1),
    "largo_stop_banda": dict(lados=1, trail_vwap=False, salida_banda=0, stop_tipo=1),
    "largo_stop_vertice": dict(lados=1, trail_vwap=False, salida_banda=0, stop_tipo=2),
    "largo_stop_objetivo": dict(lados=1, trail_vwap=False, salida_banda=0, stop_tipo=1, tp_pct=0.02),
    "largo_hasta_cierre": dict(lados=1, trail_vwap=False, salida_banda=0),
    "largo_dejar_correr": dict(lados=1, trail_vwap=False, salida_banda=2, intradia=False),
    "largo_dejar_correr_stop": dict(lados=1, trail_vwap=False, salida_banda=0, stop_tipo=1, intradia=False),
    "largo_stop_banda_continua": dict(lados=1, entrada=1, trail_vwap=False, salida_banda=0, stop_tipo=1),
    "largo_dejar_correr_continua": dict(lados=1, entrada=1, trail_vwap=False, salida_banda=2, intradia=False),
}


def tramo(s: pd.Series, a, b) -> pd.Series:
    x = s[s.index >= pd.Timestamp(a)]
    return x[x.index < pd.Timestamp(b)] if b is not None else x


def referencias(d: dict) -> dict[str, pd.Series]:
    D = len(d["fechas"])
    intradia = d["cierre"] / d["O"][:, 0] - 1
    siempre = pd.Series(np.cumprod(1 + intradia - 2 * nb.costes_dia(d, **COSTES[BASE])), d["fechas"])
    indice = pd.Series(d["cierre"] / d["cierre"][0], d["fechas"])
    # el motor del estudio de caídas se llama igual que este: se carga por su ruta
    spec = importlib.util.spec_from_file_location("motor_caidas", ROOT / "research" / "caidas" / "motor.py")
    mc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mc)
    p = mc.cargar()
    spy = pd.Series(p.close[:, p.col["SPY"]], pd.DatetimeIndex(p.fechas)).reindex(d["fechas"]).ffill()
    assert len(spy) == D
    return {"siempre largo de apertura a cierre": siempre, "índice comprar y mantener (sin dividendos)": indice,
            "SPY con dividendos": spy / spy.iloc[0]}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = nb.cargar()
    tamaños = {"1x": nb.palanca(d), "vol 2 % (máx. 4×)": nb.palanca(d, 0.02)}
    filas, curvas, anual, ops_base = [], {}, {}, []
    for nombre, kw in VARIANTES.items():
        for tam, lev in tamaños.items():
            for c, kc in COSTES.items():
                r = nb.simular(d, lev=lev, **kw, **kc)
                fila = {"variante": nombre, "tamaño": tam, "costes": c}
                for vn, (a, b) in VENTANAS.items():
                    eq = tramo(r["equity"], a, b)
                    m = nb.metricas(eq, r["operaciones"], r["en_mercado"])
                    fila |= {f"{vn}_{k}": v for k, v in m.items()}
                filas.append(fila)
                if c == BASE:
                    curvas[(nombre, tam)] = r["equity"]
                    if tam == "1x":
                        ops_base.append(r["operaciones"].assign(variante=nombre))
    for nombre, eq in referencias(d).items():
        fila = {"variante": nombre, "tamaño": "1x", "costes": BASE}
        for vn, (a, b) in VENTANAS.items():
            fila |= {f"{vn}_{k}": v for k, v in nb.metricas(tramo(eq, a, b)).items()}
        filas.append(fila)
        curvas[(nombre, "1x")] = eq
    df = pd.DataFrame(filas)
    df.to_csv(OUT / "ronda1.csv", index=False, float_format="%.5f")
    for (nombre, tam), eq in curvas.items():
        a = eq.resample("YE").last()
        anual[f"{nombre} · {tam}"] = a / np.r_[eq.iloc[0], a.to_numpy()[:-1]] - 1
    an = pd.DataFrame(anual)
    an.index = an.index.year
    an.to_csv(OUT / "ronda1_anual.csv", float_format="%.4f")
    pd.concat(ops_base).to_csv(OUT / "ronda1_operaciones.csv", index=False, float_format="%.6f")

    pd.set_option("display.width", 250)
    cols = ["variante", "tamaño", "total_cagr", "desde_2007_cagr", "total_peor_caida", "total_sharpe", "validacion_cagr",
            "busqueda_cagr", "busqueda_1_cagr", "busqueda_2_cagr", "total_operaciones_año", "total_acierto",
            "total_ret_medio", "total_dias_con_posicion", "total_tiempo_en_mercado"]
    for tam in tamaños:
        x = df[(df.tamaño == tam) & (df.costes == BASE)][cols]
        print(f"\n— {tam}, costes: {BASE} —")
        print(x.round(4).to_string(index=False))
    print("\n— sensibilidad a costes (CAGR total, 1×) —")
    print(df[df.tamaño == "1x"].pivot_table(index="variante", columns="costes", values="total_cagr",
                                            sort=False).round(4).to_string())
    print("\n— sensibilidad a costes (CAGR total, vol 2 %) —")
    print(df[df.tamaño != "1x"].pivot_table(index="variante", columns="costes", values="total_cagr",
                                            sort=False).round(4).to_string())

    gr.estilo()
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    sel = {"Noise Band · artículo (largo y corto)": ("articulo", gr.NARANJA, 1.9),
           "Noise Band · solo largo, salidas del artículo": ("largo_articulo", gr.AQUA, 1.9),
           "Noise Band · solo largo, stop en la banda inferior": ("largo_stop_banda", gr.VIOLETA, 1.9),
           "Noise Band · solo largo, dejar correr": ("largo_dejar_correr", gr.AZUL, 1.9),
           "SPY con dividendos": ("SPY con dividendos", gr.GRIS, 2.4),
           "siempre largo de apertura a cierre": ("siempre largo de apertura a cierre", "#b8b7ae", 1.4)}
    gr.curvas(ax, {k: (curvas[(n, "1x")], c, w) for k, (n, c, w) in sel.items()})
    ax.axvline(pd.Timestamp("2013-01-01"), color=gr.EJE, ls=":", lw=1)
    ax.text(pd.Timestamp("2012-11-01"), ax.get_ylim()[1] * 0.92, "validación ←", ha="right", fontsize=8.4,
            color=gr.TINTA2)
    ax.text(pd.Timestamp("2013-03-01"), ax.get_ylim()[1] * 0.92, "→ búsqueda", ha="left", fontsize=8.4,
            color=gr.TINTA2)
    ax.set_title("Noise Band: 100.000 $ en 2005, sin palanca, con costes de SPY a mercado")
    fig.tight_layout()
    fig.subplots_adjust(right=0.70)
    fig.savefig(OUT / "curvas.png", dpi=130)


if __name__ == "__main__":
    main()
