"""Catálogo de las estrategias buenas del estudio: ranking por CAGR, familias y gráficas.

Una FAMILIA es un mismo concepto con distinta palanca o reserva. Todas se recalculan
aquí con el mismo motor (v3; v4 para la entrada en dos tramos), los mismos datos
corregidos y 100.000 $ el 2 de enero de 2002, hasta el último dato:

- Caídas Técnicas (solo acciones): solo caídas (sin reserva), niveles 1, 2 y 3, la de
  caídas a 1,5× de la ronda 5 y el nivel 2 en dos tramos.
- Combinada D1 + C1 (dos componentes al 50 %, rebalanceo mensual): la de la ronda 3,
  la técnica de la ronda 5 y la escalera de la ronda 4 a 1×, 1,5× y 2×.
- Módulos y Convicción (D1, ronda 3) con tres reservas, y Caída de 5 Días (C1, ronda 2).

CAGR 2002-2026 y fuera de muestra (2002 → sep 2021, lo que las reglas no vieron).

Salida: resultados/estrategias_buenas/{ranking.csv, ranking.png, familia_*.png}

Uso: python research/caidas/estrategias_buenas.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

import eventos as ev
import fichas as fi
import finalistas3 as f3
import motor as m
import motor_v3 as v3
import motor_v4 as v4
import ronda3 as r3
import ronda4_final as r4f
from combinar import mezclar
from definitiva import D1_CFG, cartera, curva
from ronda5 import componentes_tec
from ronda5b import preparar

OUT = Path(__file__).parent / "resultados" / "estrategias_buenas"
CAPITAL = 100_000
CORTE = pd.Timestamp("2021-10-01")
# colores: paleta validada (slots 1-3 por familia; dentro de cada familia, rampa ordinal
# validada de claro a oscuro según la palanca); SPY en gris de referencia
AZUL = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
NARANJA = ["#ff9873", "#f26e3b", "#d25117", "#ac3c00", "#842c00"]
AQUA = ["#44c992", "#009b6a", "#006c49"]
VIOLETA = "#6250d6"
GRIS = "#898781"
FAMILIAS = {"Caídas Técnicas": "#2a78d6", "Combinada D1 + C1": "#eb6834",
            "Módulos de origen (rondas 2-3)": "#1baf7a"}
TINTA, TINTA2, REJILLA, EJE, FONDO = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def construir() -> list[dict]:
    """Cada estrategia: nombre, familia, palanca, color, estilo y curva de valor."""
    p = m.cargar()
    n = len(p.fechas)
    v3._RES["efectivo"] = (np.ones(n), np.ones(n), np.zeros(n), np.zeros(n, dtype=bool), np.zeros(n))
    comp = componentes_tec()
    cfg = fi.configuraciones()
    E = []

    def nueva(nombre, corto, familia, palanca, color, eq, estilo="-", nota=""):
        E.append(dict(nombre=nombre, corto=corto, familia=familia, palanca=palanca, color=color, estilo=estilo,
                      nota=nota, eq=eq / eq.iloc[0]))

    # Caídas Técnicas
    fam = "Caídas Técnicas"
    nueva("Solo Caídas Técnicas (resto en efectivo)", "Solo caídas", fam, "1× (efectivo)", AZUL[0],
          fi.simular(cfg["pura"], "2002-01-02")["equity"], nota="usa efectivo")
    nueva("Nivel 1 · reserva de acciones sin deuda", "Nivel 1", fam, "1×", AZUL[1],
          curva(comp, "D1", D1_CFG, 1.0, reserva="liq100_tend_minvol")["equity"])
    nueva("Nivel 2 · reserva contracíclica", "Nivel 2", fam, "hasta 1,5× (reserva)", AZUL[2],
          curva(comp, "D1", D1_CFG, 1.0, reserva="liq100_vg_minvol")["equity"], nota="recomendada")
    x, salida, conv = comp["D1"]
    y, w = preparar(x, conv, D1_CFG["kmin"], D1_CFG["uso"], D1_CFG["tam"], D1_CFG["frac"])
    tramos = v4.simular(y.t.to_numpy(), y.j.to_numpy(), y.prio.to_numpy(np.float64), peso=w, K=5, frac=0.5, H=20,
                        salida=salida, reserva_="liq100_vg_minvol", palanca=1.0, palanca_off=1.0, w_ini=0.5,
                        add_modo=1, add_umbral=0.03)["equity"]
    nueva("Nivel 2 en dos tramos (menos riesgo)", "Nivel 2 dos tramos", fam, "hasta 1,5× (reserva)", AZUL[2], tramos, estilo="--")
    nueva("Caídas a 1,5× (ronda 5)", "Caídas a 1,5×", fam, "hasta 1,5× (caídas)", AZUL[3],
          curva(comp, "D1", D1_CFG, 1.5, reserva="liq100_tend_minvol")["equity"])
    nueva("Nivel 3 · caídas a 2×", "Nivel 3", fam, "hasta 2×", AZUL[4],
          curva(comp, "D1", D1_CFG, 2.0, reserva="liq100_vg_minvol")["equity"])

    # Combinada D1 + C1
    fam = "Combinada D1 + C1"
    d = r3.cargar()
    e = ev.cargar()
    d1 = f3.candidatas_d1(d)
    c1 = f3.candidatas_c1(d, e)
    sim = lambda z, **kw: v3.simular(z.t.to_numpy(), z.j.to_numpy(), z.prio.to_numpy(np.float64), H=20, **kw)["equity"]  # noqa: E731
    eq_d1def50 = sim(d1, peso=np.minimum(d1.peso.to_numpy(), 1.5), K=3, salida="sma10", reserva_="spy_tend")
    eq_c1 = sim(c1, K=5, salida="rsi2", reserva_="spy_tend")
    nueva("Combinada ronda 3 · 1× (SPY y efectivo)", "Ronda 3 · 1×", fam, "1×", NARANJA[0],
          mezclar({"D1": eq_d1def50, "C1": eq_c1}, {"D1": 0.5, "C1": 0.5}), nota="usa SPY y efectivo")
    nueva("Combinada técnica ronda 5 · 1× (solo acciones)", "Técnica · 1×", fam, "1×", NARANJA[1],
          cartera(comp, "D1-técnica + C1-técnica", 1.0, reserva="liq100_tend_minvol")[0])
    for k, (clave, palanca) in enumerate((("Nivel 1 · máx. 1x (sin deuda)", "1×"),
                                          ("Nivel 2 · máx. 1,5x gestionada", "hasta 1,5×"),
                                          ("Nivel 3 · máx. 2x gestionada", "hasta 2×"))):
        nueva(f"Escalera ronda 4 · {palanca.replace('hasta ', '')} (SPY y SHY)",
              f"Escalera · {palanca.replace('hasta ', '')}", fam, palanca, NARANJA[2 + k],
              r4f.cartera(r4f.NIVELES[clave]), nota="usa SPY y SHY")

    # Módulos de origen (rondas 2-3)
    fam = "Módulos de origen (rondas 2-3)"
    nueva("D1-def50 · SPY en tendencia, tope 50 %", "D1-def50", fam, "1×", AQUA[0], eq_d1def50, nota="usa SPY y efectivo")
    nueva("D1-def · SPY en tendencia", "D1-def", fam, "1×", AQUA[1],
          sim(d1, peso=d1.peso.to_numpy(), K=3, salida="sma10", reserva_="spy_tend"), nota="usa SPY y efectivo")
    nueva("D1 · SPY siempre", "D1 · SPY siempre", fam, "1× (siempre invertida)", AQUA[2],
          sim(d1, peso=d1.peso.to_numpy(), K=3, salida="sma10", reserva_="spy"), nota="usa SPY")
    nueva("C1 · caída de 5 días con banderas (ronda 2)", "C1", fam, "1×", VIOLETA, eq_c1, nota="usa SPY y efectivo")

    spy = m.benchmark("SPY")
    nueva("SPY (comprar y mantener)", "SPY", "referencia", "1×", GRIS, spy)
    return E


def metricas(E: list[dict]) -> pd.DataFrame:
    filas = []
    for x in E:
        eq = x["eq"]
        val = eq[eq.index < CORTE]
        mt, mv = m.metricas(eq), m.metricas(val / val.iloc[0])
        filas.append({"estrategia": x["nombre"], "familia": x["familia"], "palanca": x["palanca"], "nota": x["nota"],
                      "CAGR 2002-2026": mt["cagr"], "CAGR fuera de muestra": mv["cagr"], "peor caída": mt["maxdd"],
                      "Sharpe": mt["sharpe"], "valor final de 100.000 $": CAPITAL * eq.iloc[-1] / eq.iloc[0]})
    return pd.DataFrame(filas).sort_values("CAGR 2002-2026", ascending=False).reset_index(drop=True)


def estilo() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.facecolor": FONDO, "figure.facecolor": FONDO,
        "savefig.facecolor": FONDO, "axes.edgecolor": EJE, "axes.labelcolor": TINTA2, "xtick.color": TINTA2,
        "ytick.color": TINTA2, "grid.color": REJILLA, "grid.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round"})


def dinero(v: float, _=None) -> str:
    if v >= 1e6:
        return f"{v / 1e6:,.0f} M$".replace(",", ".") if v >= 1e7 else f"{v / 1e6:,.1f} M$".replace(".", ",")
    return f"{v / 1e3:,.0f} mil $".replace(",", ".")


def ranking(df: pd.DataFrame, colores: dict) -> None:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig, ax = plt.subplots(figsize=(12.5, 0.42 * len(df) + 2.0))
    yy = np.arange(len(df))[::-1]
    cols = (("Al año\n2002-2026", "CAGR 2002-2026", 1.13), ("Al año fuera\nde muestra", "CAGR fuera de muestra", 1.30),
            ("Peor\ncaída", "peor caída", 1.45))
    for i, (_, r) in zip(yy, df.iterrows(), strict=True):
        ax.barh(i, r["CAGR 2002-2026"], height=0.62, color=colores[r.estrategia], edgecolor=FONDO, linewidth=1)
        ax.plot(r["CAGR fuera de muestra"], i, marker="o", ms=7, mfc=FONDO, mec=TINTA, mew=1.6, zorder=3)
        for _, campo, xc in cols:
            v = r[campo]
            ax.text(xc, i, f"{v * 100:.1f} %".replace(".", ",") if campo != "peor caída" else f"{v * 100:.0f} %",
                    transform=ax.get_yaxis_transform(), va="center", ha="right", fontsize=9.2,
                    color=TINTA if campo == "CAGR 2002-2026" else TINTA2, clip_on=False)
    for titulo, _, xc in cols:
        ax.text(xc, len(df) - 0.35, titulo, transform=ax.get_yaxis_transform(), va="bottom", ha="right", fontsize=8.6,
                color=TINTA2, clip_on=False)
    ax.set_yticks(yy, df.estrategia, fontsize=9.2, color=TINTA)
    ax.set_xlim(0, df["CAGR 2002-2026"].max() * 1.04)
    ax.set_ylim(-0.6, len(df) - 0.4)
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v * 100:.0f} %"))
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Estrategias buenas ordenadas por rentabilidad anual, 2002-2026", pad=26)
    leyenda = [Patch(color=c, label=f) for f, c in FAMILIAS.items()] + [Patch(color=GRIS, label="SPY (referencia)"),
               Line2D([], [], marker="o", ls="", mfc=FONDO, mec=TINTA, mew=1.6, ms=7,
                      label="fuera de muestra (2002 → sep 2021)")]
    ax.legend(handles=leyenda, loc="upper center", bbox_to_anchor=(0.62, -0.06), ncol=3, frameon=False, fontsize=8.8)
    fig.text(0.01, 0.008, "Barra: rentabilidad anual 2002-2026 con 100.000 $ al inicio. Punto: rentabilidad anual en el "
             "periodo que las reglas no vieron.", fontsize=8, color=TINTA2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.subplots_adjust(right=0.66)
    fig.savefig(OUT / "ranking.png", dpi=150)
    plt.close(fig)


def familia(E: list[dict], nombres: list[str], con_caidas: list[str], titulo: str, archivo: str, nota: str) -> None:
    sel = [x for x in E if x["nombre"] in nombres] + [x for x in E if x["familia"] == "referencia"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11.5, 7.8), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    inicio, fin = sel[0]["eq"].index[0], sel[0]["eq"].index[-1]
    finales = []
    for x in sel:
        v = CAPITAL * x["eq"]
        es_spy = x["familia"] == "referencia"
        a1.plot(v.index, v.values, color=x["color"], lw=1.3 if es_spy else 1.9, ls=x["estilo"], zorder=2 if es_spy else 3)
        if es_spy or x["nombre"] in con_caidas:
            dd = v / v.cummax() - 1
            a2.plot(dd.index, dd.values, color=x["color"], lw=1.0 if es_spy else 1.2, ls=x["estilo"],
                    zorder=2 if es_spy else 3)
        finales.append([np.log10(v.iloc[-1]), x, v.iloc[-1]])
    a1.set_yscale("log")
    a1.set_xlim(inicio, fin)
    a1.yaxis.set_major_formatter(mtick.FuncFormatter(dinero))
    a1.yaxis.set_minor_formatter(mtick.NullFormatter())
    a1.grid(axis="y", which="major")
    a1.set_title(titulo, pad=22)
    a1.text(0, 1.015, nota, transform=a1.transAxes, fontsize=8.8, color=TINTA2, va="bottom")
    # etiquetas al final de cada línea, separadas y unidas a su línea con un trazo de su color
    finales.sort(key=lambda z: z[0])
    lo, hi = (np.log10(v) for v in a1.get_ylim())
    paso = (hi - lo) * 0.05
    pos = []
    for f_ in finales:
        pos.append(f_[0] if not pos else max(f_[0], pos[-1] + paso))
    x_lab = fin + pd.Timedelta(days=330)
    for yv, (y0, x, valor) in zip(pos, finales, strict=True):
        a1.plot([fin, fin + pd.Timedelta(days=90), x_lab - pd.Timedelta(days=40)], [10 ** y0, 10 ** yv, 10 ** yv],
                color=x["color"], lw=1.0, ls=x["estilo"], clip_on=False)
        a1.text(x_lab, 10 ** yv, f"{x['corto']}  {dinero(valor)}", va="center", fontsize=8.6, color=TINTA,
                clip_on=False)
    for a in (a1, a2):
        a.axvline(CORTE, color=EJE, lw=1, ls=(0, (2, 3)))
    a1.text(CORTE, 10 ** (lo + (hi - lo) * 0.03), "← fuera de muestra  ", fontsize=8.2, color=TINTA2, ha="right")
    a1.text(CORTE, 10 ** (lo + (hi - lo) * 0.03), "  búsqueda →", fontsize=8.2, color=TINTA2)
    a2.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v * 100:.0f} %"))
    a2.set_title("Caída desde el máximo (" + ", ".join(
        x["corto"] for x in sel if x["nombre"] in con_caidas) + " y SPY)", fontsize=10, fontweight="normal",
        color=TINTA2, pad=6)
    a2.grid(axis="y")
    a2.xaxis.set_major_locator(mdates.YearLocator(4))
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.subplots_adjust(right=0.80)
    fig.savefig(OUT / archivo, dpi=150)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    estilo()
    E = construir()
    df = metricas(E)
    df.to_csv(OUT / "ranking.csv", index=False, float_format="%.5f")
    ranking(df, {x["nombre"]: x["color"] if x["familia"] == "referencia" else FAMILIAS[x["familia"]] for x in E})
    por_fam = lambda f: [x["nombre"] for x in E if x["familia"] == f]  # noqa: E731
    familia(E, por_fam("Caídas Técnicas"), ["Nivel 1 · reserva de acciones sin deuda", "Nivel 2 · reserva contracíclica",
                                             "Nivel 3 · caídas a 2×"],
            "Familia Caídas Técnicas: 100.000 $ desde 2002 (escala logarítmica)", "familia_caidas_tecnicas.png",
            "Más oscuro = más palanca. Discontinua: nivel 2 comprando en dos tramos.")
    familia(E, por_fam("Combinada D1 + C1"), ["Escalera ronda 4 · 1× (SPY y SHY)", "Escalera ronda 4 · 1,5× (SPY y SHY)",
                                               "Escalera ronda 4 · 2× (SPY y SHY)"],
            "Familia Combinada D1 + C1: 100.000 $ desde 2002 (escala logarítmica)", "familia_combinada.png",
            "Más oscuro = más palanca. La escalera de la ronda 4 usa SPY y SHY como reserva.")
    familia(E, por_fam("Módulos de origen (rondas 2-3)"), ["D1-def50 · SPY en tendencia, tope 50 %", "D1 · SPY siempre",
                                                            "C1 · caída de 5 días con banderas (ronda 2)"],
            "Rondas 2 y 3, módulos de origen de la caída: 100.000 $ desde 2002 (escala logarítmica)",
            "familia_modulos.png",
            "Verde: D1 con tres reservas (más oscuro = más tiempo invertido). Violeta: C1. Usan SPY o efectivo.")
    pd.set_option("display.width", 220)
    print(df.drop(columns=["nota"]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
