"""Estilo común de las gráficas de Noise Band (la misma paleta validada que el estudio de caídas)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd

AZUL, NARANJA, AQUA, VIOLETA, GRIS = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#898781"
TINTA, TINTA2, REJILLA, EJE, FONDO = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def estilo() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.facecolor": FONDO, "figure.facecolor": FONDO,
        "savefig.facecolor": FONDO, "axes.edgecolor": EJE, "axes.labelcolor": TINTA2, "xtick.color": TINTA2,
        "ytick.color": TINTA2, "grid.color": REJILLA, "grid.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left"})


def dinero(v: float, _=None) -> str:
    if v >= 1e6:
        return f"{v / 1e6:.1f} M$".replace(".", ",")
    return f"{v / 1e3:.0f} mil $"


def curvas(ax, series: dict[str, tuple[pd.Series, str, float]], capital: float = 100_000) -> None:
    """Valor en dólares en escala logarítmica, con la etiqueta al final de cada línea
    (separadas para que no se pisen)."""
    finales = []
    for nombre, (eq, color, grosor) in series.items():
        v = capital * eq / eq.iloc[0]
        ax.plot(v.index, v.values, color=color, lw=grosor)
        finales.append([float(v.iloc[-1]), nombre, color, v.index[-1]])
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(mtick.LogLocator(base=10, subs=(1.0, 1.5, 2.0, 3.0, 5.0, 7.0)))
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(dinero))
    ax.yaxis.set_minor_locator(mtick.NullLocator())
    finales.sort(key=lambda x: x[0])
    ultimo = None
    for f in finales:  # separación mínima de un 9 % en escala logarítmica
        if ultimo is not None and f[0] < ultimo * 1.09:
            f[0] = ultimo * 1.09
        ultimo = f[0]
        ax.annotate(f[1], xy=(f[3], f[0]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.6,
                    color=f[2], annotation_clip=False)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
