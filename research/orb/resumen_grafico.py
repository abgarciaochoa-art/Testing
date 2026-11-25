"""Resumen gráfico del Gap ORB en SPY: 100.000 $ desde ene-2015 hasta oct-2025, por tamaño y operativa."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import datos  # noqa: E402
from motor import atr_diario, orb  # noqa: E402
from ronda9_spy_oos import gap_orb  # noqa: E402

INI, CAP = "2015-01-01", 100_000
x = datos.indice("SPY")
o = gap_orb(x, 15, 0.4)
coste = o.bruto - o.r1  # coste ida y vuelta por operación
idx = o.index >= INI


def curva(r1, dist, op, riesgo):
    w = np.minimum(riesgo / dist, 4.0).where(op, 0).fillna(0)
    r = (w * r1)[idx]
    return CAP * (1 + r).cumprod()


def cagr(eq):
    años = (eq.index[-1] - eq.index[0]).days / 365.25
    return (eq.iloc[-1] / CAP) ** (1 / años) - 1


def maxdd(eq):
    return (eq / eq.cummax() - 1).min()


# ORB clásico (sin filtro de hueco, cualquier lado, mismo rango y stop)
a = atr_diario(x["O"], x["H"], x["L"], x["C"], x["ultimo"])
d, pe, ps, rie, _, _ = orb(x["O"], x["H"], x["L"], x["C"], x["ultimo"].astype(np.int64), 15, 0, 0, 1.0, a, 0.0, 0.0,
                           389, 0.0, 1e12)
opc = pd.Series(d != 0, index=o.index)
r1c = pd.Series(np.where(d != 0, d * (ps / pe - 1) - 2 * 0.02 / pe, 0.0), index=o.index)
distc = pd.Series(np.where(d != 0, rie / pe, np.nan), index=o.index)
D = len(x["fechas"])
cl = pd.Series(x["C"][np.arange(D), x["ultimo"]], index=o.index)[idx]
bh = CAP * cl / cl.iloc[0]

curvas = {f"Riesgo {r:.1%}".replace(".", ","): curva(o.r1, o.dist, o.op, r) for r in (0.005, 0.01, 0.015, 0.02)}
R = 0.015
cond = [
    ("Tamaño", "Riesgo 0,5 %/op.", curvas["Riesgo 0,5%"]),
    ("Tamaño", "Riesgo 1 %/op.", curvas["Riesgo 1,0%"]),
    ("Tamaño", "Riesgo 1,5 %/op. (recomendado)", curvas["Riesgo 1,5%"]),
    ("Tamaño", "Riesgo 2 %/op.", curvas["Riesgo 2,0%"]),
    ("Operativa (riesgo 1,5 %)", "Solo largos", curva(o.r1, o.dist, o.op & (o.dir > 0), R)),
    ("Operativa (riesgo 1,5 %)", "Solo cortos", curva(o.r1, o.dist, o.op & (o.dir < 0), R)),
    ("Operativa (riesgo 1,5 %)", "Costes dobles", curva(o.r1 - coste / 2 * 2 * o.op, o.dist, o.op, R)),
    ("Referencia", "ORB clásico sin filtro de hueco", curva(r1c, distc, opc, R)),
    ("Referencia", "Comprar y mantener SPY (sin dividendos)", bh),
]
filas = [dict(grupo=g, nombre=n, cagr=cagr(e), pnl=e.iloc[-1] - CAP, dd=maxdd(e)) for g, n, e in cond]
tab = pd.DataFrame(filas)
tab.to_csv(Path(__file__).parent / "resultados" / "resumen_2015_2025.csv", index=False)
print(tab.round(3).to_string())

# ---------- figura ----------
BLUE, ORANGE, AQUA, YEL = "#257de0", "#f47230", "#00b894", "#f2a600"
TXT, TXT2, GRID = "#191919", "#666666", "#e6e6e6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": TXT, "axes.labelcolor": TXT2,
                     "xtick.color": TXT2, "ytick.color": TXT2, "axes.edgecolor": GRID})
fig = plt.figure(figsize=(13, 9.6), facecolor="white")
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.05], hspace=0.38)
fig.suptitle("Gap ORB en SPY · 100.000 $ de ene-2015 a oct-2025 (10,8 años, costes incluidos)",
             x=0.06, ha="left", fontsize=14, fontweight="bold")

ax = fig.add_subplot(gs[0])
for (nom, eq), col in zip(curvas.items(), (BLUE, ORANGE, AQUA, YEL)):
    ax.plot(eq.index, eq / 1000, color=col, lw=2, label=nom)
    ax.annotate(f"{nom.replace('Riesgo ', '')}: {eq.iloc[-1]/1000:,.0f} k$".replace(",", "."),
                (eq.index[-1], eq.iloc[-1] / 1000), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=9, color=TXT)
ax.plot(bh.index, bh / 1000, color="#9c9c9c", lw=1.5, ls="--", label="Comprar y mantener SPY (1×)")
ax.set_ylabel("Capital (miles de $)")
ax.set_title("Evolución del capital según el riesgo por operación", loc="left", fontsize=11, color=TXT)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="upper left", fontsize=9)
ax.set_xlim(bh.index[0], bh.index[-1] + pd.Timedelta(days=420))

ax2 = fig.add_subplot(gs[1])
y = np.arange(len(tab))[::-1]
cols = [BLUE if g != "Referencia" else "#9c9c9c" for g in tab.grupo]
ax2.barh(y, tab.cagr * 100, color=cols, height=0.62, edgecolor="white", linewidth=2)
for yi, r in zip(y, tab.itertuples()):
    txt = f"{r.cagr:+.1%}/año · PnL {r.pnl/1000:+,.0f} k$ · peor caída {r.dd:.0%}".replace(".", ",", 1)
    xpos = max(r.cagr * 100, 0) + 0.4
    ax2.text(xpos, yi, txt, va="center", fontsize=9, color=TXT)
ax2.set_yticks(y)
ax2.set_yticklabels(tab.nombre, fontsize=9.5)
ax2.axvline(0, color=TXT2, lw=1)
ax2.set_xlabel("CAGR (% anual)")
ax2.set_xlim(min(tab.cagr.min() * 100 - 1, -2), tab.cagr.max() * 100 + 17)
ax2.set_title("CAGR y beneficio por condición (azul = Gap ORB · gris = referencias)", loc="left", fontsize=11, color=TXT)
ax2.grid(axis="x", color=GRID, lw=0.8)
ax2.spines[["top", "right", "left"]].set_visible(False)
for b in (3.5, 6.5):
    ax2.axhline(len(tab) - 1 - b, color=GRID, lw=1)
fig.text(0.06, 0.015, "Regla: hueco > 0,4×ATR(14) → orden stop en el extremo del rango 9:30-9:45 del lado del hueco; stop en el "
         "otro extremo; salida al cierre. Tamaño = riesgo / distancia al stop, máx. 4×. Costes 0,02 $/acción por lado.",
         fontsize=8.5, color=TXT2)
out = Path(__file__).parent / "graficos" / "resumen_gap_orb_2015_2025.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(out)
