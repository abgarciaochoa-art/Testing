"""Finalistas de la ronda de módulos (puntuaciones LIMPIAS: solo vieron oct-2021 →).

C1 · explicable: caída >=7 % en 5 d en tendencia con >=6 módulos a favor, K=5,
     salida RSI2 > 70 (máx. 20 d), reserva en SPY solo si SPY > SMA200.
C2 · máximo CAGR: caída >=15 % en 10 d en tendencia, prioridad por ridge, K=2,
     salida > SMA10 (máx. 10 d), misma reserva.

Pruebas: costes 20/30 pb, universo de miembros actuales (sesgo de supervivencia),
DSR con N = todas las carteras probadas en el proyecto, años naturales y figura.

Uso: python research/caidas/finalistas2.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import eventos as ev
import motor as m
import motor_rapido as mr
from combinada import puntuar

sys.path.insert(0, str(m.ROOT))
from earnings_alpha.stats.performance import deflated_sharpe_ratio  # noqa: E402

OUT = Path(__file__).parent / "resultados"
CACHE = ev.SALIDA.parent

FINAL = {
    "C1 · explicable (≥6 módulos, K=5)": dict(
        filtro=lambda d: (d.r5 <= -0.07) & (d.sobre_sma200 == 1) & (d.score_limpio >= 6),
        score=lambda d: d.score_limpio + (-d.r10.fillna(0)).clip(0, 0.99), K=5, salida="rsi2", H=20),
    "C2 · máximo CAGR (ridge, K=2)": dict(
        filtro=lambda d: (d.r10 <= -0.15) & (d.sobre_sma200 == 1),
        score=lambda d: d.score_ridge, K=2, salida="sma10", H=10),
}


def main() -> None:
    d = puntuar(ev.cargar())
    p = m.cargar()
    actuales = p.miembro[-1]
    R = np.vstack([np.load(CACHE / "rejilla_retornos.npy"), np.load(CACHE / "rejilla_retornos_fase2.npy"),
                   np.load(CACHE / "ronda2_retornos.npy")])
    n_combinada = len(pd.read_csv(OUT / "combinada_carteras.csv"))
    sh = R[:, 1:].mean(1) / R[:, 1:].std(1)
    spy = m.benchmark("SPY")
    curvas = {"SPY": spy / spy.iloc[0]}
    filas, anual = [], {}
    for nombre, c in FINAL.items():
        x = d[c["filtro"](d).fillna(False)]
        kw = dict(K=c["K"], H=c["H"], salida=c["salida"], reserva="spy_tend")
        sc = c["score"](x).to_numpy(np.float64)

        def run(sel=None, **extra):
            y = x if sel is None else x[sel]
            s = sc if sel is None else sc[sel.to_numpy()]
            return mr.simular(y.t.to_numpy(), y.j.to_numpy(), s, **(kw | extra))

        tot = run()
        eq = tot["equity"]
        ret = eq.pct_change().fillna(0).to_numpy()[1:]
        dsr = deflated_sharpe_ratio(ret, trial_sharpes=sh, n_trials=len(sh) + n_combinada)
        fila = {"estrategia": nombre}
        for w, (a, b) in {"búsqueda oct21-oct26": ("2021-10-01", None),
                          "validación 2002-sep21": ("2002-01-02", "2021-10-01")}.items():
            r = run(inicio=a, fin=b)
            fila |= {f"CAGR {w}": r["cagr"], f"maxDD {w}": r["maxdd"]}
        fila |= {"CAGR total": tot["cagr"], "maxDD total": tot["maxdd"], "Sharpe": tot["sharpe"],
                 "vol": tot["vol"], "operaciones/año": tot["trades_año"], "acierto": tot["acierto"],
                 "ret. medio op.": tot["ret_medio_trade"], "% tiempo en acciones": tot["exposicion"]}
        for bps in (20, 30):
            fila[f"CAGR coste {bps}pb"] = run(coste_bps=bps)["cagr"]
        fila["CAGR sin las que luego salieron del índice"] = run(sel=pd.Series(actuales[x.j.to_numpy()], index=x.index))["cagr"]
        fila["CAGR reserva SPY siempre"] = run(reserva="spy")["cagr"]
        fila["CAGR sin reserva (efectivo)"] = run(reserva=None)["cagr"]
        fila |= {"DSR": dsr.dsr, "N pruebas": dsr.n_trials, "PSR(0)": dsr.psr_zero}
        filas.append(fila)
        curvas[nombre] = eq / eq.iloc[0]
        anual[nombre] = curvas[nombre].resample("YE").last()
        print(nombre, "ok", flush=True)
    anual["SPY"] = curvas["SPY"].resample("YE").last()
    a = pd.DataFrame(anual)
    a = a.pct_change().fillna(a.iloc[0] - 1)
    a.index = a.index.year
    df = pd.DataFrame(filas).set_index("estrategia")
    df.T.to_csv(OUT / "finalistas2.csv", float_format="%.4f")
    a.to_csv(OUT / "finalistas2_anual.csv", float_format="%.4f")
    (OUT / "finalistas2.json").write_text(json.dumps({"tabla": df.T.to_dict(), "anual": a.to_dict()},
                                                     ensure_ascii=False, indent=1, default=float))
    pd.set_option("display.width", 200)
    print(df.T.round(3).to_string())
    print((a * 100).round(1).to_string())
    prev = pd.read_csv(OUT / "finalistas.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(11, 6))
    for k, v in curvas.items():
        ax.plot(v.index, v.values, lw=2.4 if k == "SPY" else 1.5, color="black" if k == "SPY" else None, label=k)
    del prev
    ax.set_yscale("log")
    ax.axvline(pd.Timestamp("2021-10-01"), color="grey", ls=":")
    ax.text(pd.Timestamp("2021-12-01"), ax.get_ylim()[0] * 1.3, "← validación | búsqueda →", color="grey", fontsize=8)
    ax.set_title("Compra de caídas con módulos de mejora — 1 $ en 2002 (log), costes 10 pb/lado")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "finalistas2.png", dpi=130)


if __name__ == "__main__":
    main()
