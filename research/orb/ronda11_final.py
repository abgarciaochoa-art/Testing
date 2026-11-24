"""Ronda 11: estudio final de la regla candidata en SPY 1 min (2000 → oct-2025).

GAP ORB: si el hueco de la noche es mayor que k × ATR(14) (en %), se pone una orden stop en el
extremo del rango de los primeros n minutos del lado del hueco; stop al otro lado del rango;
salida al cierre. Costes 0,02 $/acción por lado.
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import datos  # noqa: E402
from motor import metricas  # noqa: E402
from ronda9_spy_oos import con_riesgo, gap_orb, tramo  # noqa: E402

OUT = Path(__file__).parent / "resultados"
pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 300)
x = datos.indice("SPY")
P = [("2000-04", "2000-01-01", "2004-12-31"), ("2005-12", "2005-01-01", "2012-12-31"),
     ("2013-20", "2013-01-01", "2020-05-13"), ("2020-25", "2020-05-14", "2025-12-31"), ("TODO", "2000-01-01", "2025-12-31")]

# 1) vecindad
filas = []
for n_or, k in itertools.product((10, 15, 20, 30, 45), (0.3, 0.35, 0.4, 0.5, 0.6, 0.8)):
    o = gap_orb(x, n_or, k)
    f = dict(n_or=n_or, k=k)
    for nom, a, b in P:
        r = tramo(o, a, b)
        f[f"t_{nom}"] = r["t"]
        f[f"pb_{nom}"] = r["pb_neto"]
    f["ops_año"] = tramo(o, "2000", "2026")["ops_año"]
    filas.append(f)
v = pd.DataFrame(filas)
print("== t de las operaciones, 2000-2025")
print(v.pivot(index="n_or", columns="k", values="t_TODO").round(2))
print("== pb netos por operación, 2000-2025")
print(v.pivot(index="n_or", columns="k", values="pb_TODO").round(1))
print("== nº de periodos (de 4) con pb > 0")
v["per+"] = sum((v[f"pb_{n}"] > 0).astype(int) for n, *_ in P[:4])
print(v.pivot(index="n_or", columns="k", values="per+"))
print("== operaciones al año")
print(v.pivot(index="n_or", columns="k", values="ops_año").round(0))

# 2) regla elegida
N_OR, K = 15, 0.4
o = gap_orb(x, N_OR, K)
print(f"\n== regla elegida: rango {N_OR} min, hueco > {K} ATR")
print(pd.DataFrame({n: tramo(o, a, b) for n, a, b in P}).T.round(3).to_string())
t = o.r1[o.op]
gan, per = t[t > 0], t[t <= 0]
racha = (t <= 0).astype(int).groupby((t > 0).cumsum()).sum().max()
print(f"acierto {len(gan)/len(t):.1%}; ganancia media {gan.mean()*1e4:.1f} pb, pérdida media {per.mean()*1e4:.1f} pb; "
      f"ratio {-gan.mean()/per.mean():.2f}; peor racha de pérdidas seguidas {racha}")
print("salidas por stop: %.0f %%" % (100 * (o.motivo[o.op] == 1).mean()))
print("largos: n=%d pb %.1f | cortos: n=%d pb %.1f" % ((o.dir == 1).sum(), o.r1[o.dir == 1].mean() * 1e4,
                                                      (o.dir == -1).sum(), o.r1[o.dir == -1].mean() * 1e4))
# 3) permutación: misma selección de días, dirección al azar
rng = np.random.default_rng(1)
base_t = t.mean() / t.std() * len(t) ** .5
o_l = gap_orb(x, N_OR, 0.0)  # todas las direcciones a favor del hueco con k=0
xs = {}
from motor import atr_diario, orb  # noqa: E402
a = atr_diario(x["O"], x["H"], x["L"], x["C"], x["ultimo"])
res = {l: orb(x["O"], x["H"], x["L"], x["C"], x["ultimo"].astype(np.int64), N_OR, 0, l, 1.0, a, 0.0, 0.0, 389, 0.0, 1e12)
       for l in (1, -1)}
hueco = x["O"][:, 0] / x["prev"] - 1
usar = np.abs(hueco) > K * a / x["prev"]
ts = []
for _ in range(2000):
    g = rng.choice([-1, 1], len(hueco))
    d = np.where(g > 0, res[1][0], res[-1][0]) * usar
    pe = np.where(g > 0, res[1][1], res[-1][1]); ps = np.where(g > 0, res[1][2], res[-1][2])
    op = d != 0
    r = d[op] * (ps[op] / pe[op] - 1) - 2 * 0.02 / pe[op]
    ts.append(r.mean() / r.std() * len(r) ** .5)
ts = np.array(ts)
print(f"permutación (dirección al azar en los mismos días): t real {base_t:.2f}; p = {(ts >= base_t).mean():.4f}; "
      f"t medio al azar {ts.mean():.2f}")
# 4) costes
for c in (0.0, 0.01, 0.02, 0.04, 0.06):
    oc = gap_orb(x, N_OR, K, coste_usd=c)
    r = tramo(oc, "2000", "2026")
    r2 = tramo(oc, "2020-05-14", "2026")
    print(f"coste {c:.2f} $/acción por lado: pb neto {r['pb_neto']:.1f} t {r['t']:.2f} | 2020-25 pb {r2['pb_neto']:.1f}")
# 5) riesgo por operación
print("\n== tamaño: riesgo por operación (tamaño = riesgo / distancia al stop, máx. 4x)")
for riesgo in (0.005, 0.01, 0.015, 0.02, 0.025):
    rd, w = con_riesgo(o, riesgo)
    m = metricas(rd.to_numpy(), None)
    m2 = metricas(rd[rd.index >= "2020-05-14"].to_numpy(), None)
    print(f"riesgo {riesgo:.1%}: CAGR {m['cagr']:.1%} vol {m['vol']:.1%} Sharpe {m['sharpe']:.2f} peor caída {m['maxdd']:.0%} "
          f"palanca media {w[o.op].mean():.2f}x (máx {w.max():.1f}x) | 2020-25 CAGR {m2['cagr']:.1%} caída {m2['maxdd']:.0%}")
rd, _ = con_riesgo(o, 0.015)
print("\nPor año con riesgo 1,5 % (%):")
print((rd.groupby(rd.index.year).apply(lambda r: (1 + r).prod() - 1) * 100).round(1).to_frame("%").T.to_string())
rd.to_csv(OUT / "gap_orb_spy_diario_riesgo1.5.csv")
o.to_csv(OUT / "gap_orb_spy_operaciones.csv")
