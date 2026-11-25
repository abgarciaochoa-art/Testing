"""Gap ORB con futuros (MES), 30.000 $ propios, sin préstamo: riesgo por operación de 1,5 % a 5 %."""
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
CAP0, MAXLEV = 30_000, 13.3


def sim(riesgo):
    eq, out, levs = CAP0, [], []
    for r1, dist, op in zip(o.r1, o.dist, o.op):
        if op:
            lev = min(riesgo / dist, MAXLEV)
            levs.append(lev)
            eq *= 1 + lev * r1
        out.append(eq)
    return pd.Series(out, index=o.index), np.array(levs)


filas, curvas = [], {}
for r in (0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05):
    e, lev = sim(r)
    años = (e.index[-1] - e.index[0]).days / 365.25
    fin = e.groupby(e.index.year).last()
    anual = fin / fin.shift(1, fill_value=CAP0) - 1
    nom = f"{r*100:.1f} %".replace(".", ",").replace(",0 %", " %")
    filas.append(dict(riesgo=nom, final=e.iloc[-1], pnl=e.iloc[-1] - CAP0, cagr=(e.iloc[-1] / CAP0) ** (1 / años) - 1,
                      peor_caida=(e / e.cummax() - 1).min(), peor_año=anual.min(), años_neg=(anual < 0).sum(),
                      minimo=e.min(), lev_media=lev.mean(), lev_p95=np.percentile(lev, 95),
                      mes_medio=(lev.mean() * 30_000 / 34_000)))
    curvas[nom] = e
tab = pd.DataFrame(filas)
pd.set_option("display.width", 250)
print(tab.round(3).to_string())
tab.to_csv(Path(__file__).parent / "resultados" / "futuros_riesgo_30k.csv", index=False)

COL = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#6a4ce2", "#ec4647"]
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor="white", gridspec_kw=dict(width_ratios=[1.6, 1], wspace=0.28))
for (nom, e), c in zip(curvas.items(), COL):
    ax.plot(e.index, e / 1000, color=c, lw=1.8, label=f"Riesgo {nom}")
    if nom in ("3,5 %", "4 %"):
        continue
    ax.annotate(f"{nom}: {e.iloc[-1]/1000:.0f} k$", (e.index[-1], e.iloc[-1] / 1000),
                xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.5, color="#191919")
ax.set_yscale("log")
ax.set_yticks([20, 30, 50, 100, 200, 400, 800])
ax.set_yticklabels(["20", "30", "50", "100", "200", "400", "800"])
ax.set_ylabel("Capital (miles de $, escala log)", color="#666666")
ax.set_title("Capital con 30.000 \\$ en futuros MES, sin préstamo", loc="left", fontsize=11.5, fontweight="bold")
ax.grid(axis="y", color="#e6e6e6")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=8.5, loc="upper left")
ax.set_xlim(o.index[0], o.index[-1] + pd.Timedelta(days=500))

x = -tab.peor_caida * 100
y = tab.cagr * 100
ax2.plot(x, y, color="#9c9c9c", lw=1, zorder=1)
for (xi, yi, nom, c) in zip(x, y, tab.riesgo, COL):
    ax2.scatter(xi, yi, s=70, color=c, edgecolor="white", linewidth=2, zorder=2)
    ax2.annotate(nom, (xi, yi), xytext=(7, -3), textcoords="offset points", fontsize=9, color="#191919")
ax2.set_xlabel("Peor caída (%)", color="#666666")
ax2.set_ylabel("CAGR (% anual)", color="#666666")
ax2.set_title("Rentabilidad frente a peor caída", loc="left", fontsize=11.5, fontweight="bold")
ax2.grid(color="#e6e6e6")
ax2.spines[["top", "right"]].set_visible(False)
fig.text(0.07, -0.02, "Gap ORB en SPY, ene-2015 → oct-2025, costes incluidos. Riesgo = % del capital perdido si salta el stop; "
         "palanca máx. 13×. Sin intereses (no se pasa la noche dentro).", fontsize=8.5, color="#666666")
out = Path(__file__).parent / "graficos" / "futuros_riesgo_30k.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(out)
