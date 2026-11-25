"""Gap ORB con 30.000 $ propios + 70.000 $ prestados (operar como si hubiera 100.000 $), ene-2015 → oct-2025."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import datos  # noqa: E402
from ronda9_spy_oos import gap_orb  # noqa: E402

o = gap_orb(datos.indice("SPY"), 15, 0.4)
o = o[o.index >= "2015-01-01"]
PROPIO, PREST, INT = 30_000, 70_000, 0.058  # interés IBKR aprox. (referencia + 1,5 %)


def sim(riesgo, cap, prestamo=0.0, interes=0.0, fijo=None):
    """riesgo y cap sobre el capital propio; fijo: tamaño calculado sobre un capital fijo (100k) en vez del propio."""
    eq, out, quiebra = PROPIO, [], False
    for r1, dist, op in zip(o.r1, o.dist, o.op):
        if op and eq > 0:
            base = fijo if fijo else eq
            nocional = min(riesgo * base / dist, cap * eq)
            eq += nocional * r1
        eq -= prestamo * interes / 252
        if eq <= 0:
            quiebra, eq = True, 0
        out.append(eq)
    return pd.Series(out, index=o.index), quiebra


esc = [
    ("A · Solo tus 30k, riesgo 1,5 %, máx. 4× (cuenta de margen IBKR)", 0.015, 4, 0, 0, None),
    ("B · 'Como 100k' con margen intradía IBKR: riesgo 5 %, máx. 4× tus 30k", 0.05, 4, 0, 0, None),
    ("C · 'Como 100k' con futuros (MES): riesgo 5 %, máx. 13×, sin intereses", 0.05, 13.3, 0, 0, None),
    ("D · Igual que C pero con préstamo real de 70k al 5,8 %", 0.05, 13.3, PREST, INT, None),
    ("E · Tamaño fijo de 100k (sin bajar al perder), con préstamo", 0.015, 13.3, PREST, INT, 100_000),
]
filas, curvas = [], {}
for nom, r, cap, p, i, fijo in esc:
    e, q = sim(r, cap, p, i, fijo)
    años = (e.index[-1] - e.index[0]).days / 365.25
    dd = (e / e.cummax() - 1)
    anual = e.groupby(e.index.year).last() / e.groupby(e.index.year).last().shift(1, fill_value=PROPIO) - 1
    filas.append(dict(escenario=nom, final=e.iloc[-1], pnl=e.iloc[-1] - PROPIO,
                      cagr=(e.iloc[-1] / PROPIO) ** (1 / años) - 1 if e.iloc[-1] > 0 else -1, peor_caida=dd.min(),
                      minimo=e.min(), peor_año=anual.min(), dias_bajo_25k=(e < 25_000).sum(), quiebra=q))
    curvas[nom] = e
tab = pd.DataFrame(filas)
pd.set_option("display.width", 250)
print(tab.round(3).to_string())
tab.to_csv(Path(__file__).parent / "resultados" / "apalancado_30k.csv", index=False)

COL = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac"]
fig, ax = plt.subplots(figsize=(12.5, 6.2), facecolor="white")
for (nom, e), c in zip(curvas.items(), COL):
    ax.plot(e.index, e / 1000, color=c, lw=2, label=nom.split(" · ")[0] + " · " + nom.split(" · ")[1][:70])
    ax.annotate(f"{nom[0]}: {e.iloc[-1]/1000:,.0f} k$".replace(",", "."), (e.index[-1], e.iloc[-1] / 1000),
                xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color="#191919")
ax.axhline(25, color="#dc3c3d", lw=1, ls=":")
ax.text(o.index[-900], 22.5, "25 k$: mínimo para day trading con acciones (regla PDT)", color="#666666", fontsize=8.5)
ax.set_yscale("log")
ax.set_yticks([10, 25, 50, 100, 200, 400, 800, 1600])
ax.set_yticklabels(["10", "25", "50", "100", "200", "400", "800", "1.600"])
ax.set_ylabel("Capital propio (miles de $, escala log)", color="#666666")
ax.set_title("Gap ORB · 30.000 \\$ propios (+70.000 \\$ de palanca) · ene-2015 → oct-2025", loc="left", fontsize=12.5,
             fontweight="bold")
ax.grid(axis="y", color="#e6e6e6")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=8.5, loc="upper left")
ax.set_xlim(o.index[0], o.index[-1] + pd.Timedelta(days=380))
out = Path(__file__).parent / "graficos" / "apalancado_30k.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(out)
