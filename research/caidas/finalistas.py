"""Evaluación a fondo de las variantes del centro de la meseta "SPY + compra en caída".

Se eligen por estar en el centro de una región de parámetros donde casi todos los
vecinos baten a SPY (no por ser el máximo de la rejilla), y se someten a: costes de
10/20/30 pb, universo de miembros actuales (sesgo de supervivencia), Sharpe
deflactado con N = todas las pruebas, alfa/beta frente a SPY con t de Newey-West,
y retornos por año natural.

Uso: python research/caidas/finalistas.py
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

import motor as m
from estudio import nw_t

sys.path.insert(0, str(m.ROOT))
from earnings_alpha.stats.performance import deflated_sharpe_ratio  # noqa: E402

OUT = Path(__file__).parent / "resultados"
CACHE = m.ROOT / "data" / "cache" / "caidas"

BASE = dict(tipo="ret", n=10, filtro="tendencia", reserva="SPY")
FINALISTAS = {
    "F1 · cae ≥15 % en 10d · sale >SMA5 · K=3": {**BASE, "x": 0.15, "salida": "sma5", "h": 20, "k": 3},
    "F2 · cae ≥15 % en 10d · sale >SMA5 · K=5": {**BASE, "x": 0.15, "salida": "sma5", "h": 20, "k": 5},
    "F3 · cae ≥20 % en 10d · sale >SMA20 · K=3": {**BASE, "x": 0.20, "salida": "sma20", "h": 20, "k": 3},
    "F4 · cae ≥10 % en 10d · sale >SMA5 · K=3": {**BASE, "x": 0.10, "salida": "sma5", "h": 10, "k": 3},
}


def main() -> None:
    R = np.vstack([np.load(CACHE / "rejilla_retornos.npy"), np.load(CACHE / "rejilla_retornos_fase2.npy")])
    sh = R[:, 1:].mean(1) / R[:, 1:].std(1)
    spy = m.benchmark("SPY")
    spy_r = spy.pct_change().fillna(0)
    filas, curvas, anual = [], {"SPY": spy / spy.iloc[0]}, {}
    for nombre, c in FINALISTAS.items():
        r = m.backtest(c)
        eq = r.equity
        ret = eq.pct_change().fillna(0)
        x = spy_r.reindex(ret.index).fillna(0)
        beta = np.cov(ret, x)[0, 1] / x.var()
        resid = (ret - beta * x).to_numpy()
        fila = {"estrategia": nombre}
        for per, sel in (("2002-2015", eq.index < "2016-01-01"), ("2016-2026", eq.index >= "2015-12-31"),
                         ("total", slice(None))):
            mt = m.metricas(eq[sel])
            fila |= {f"cagr {per}": mt["cagr"], f"maxdd {per}": mt["maxdd"]}
        tot = m.metricas(eq, r.trades, r.exposicion)
        fila |= {"sharpe": tot["sharpe"], "vol": tot["vol"], "operaciones/año": tot["trades_año"],
                 "acierto": tot["acierto"], "ret. medio op.": tot["ret_medio_trade"],
                 "días medios": tot["dias_medios"], "% tiempo en acciones": tot["exposicion"],
                 "beta SPY": beta, "alfa anual": resid.mean() * 252, "t alfa (NW)": nw_t(resid, 10)}
        for bps in (20, 30):
            fila[f"cagr coste {bps}pb"] = m.metricas(m.backtest(c, coste_bps=bps).equity)["cagr"]
        fila["cagr universo actual (sesgado)"] = m.metricas(m.backtest({**c, "universo": "actual"}).equity)["cagr"]
        dsr = deflated_sharpe_ratio(ret.to_numpy()[1:], trial_sharpes=sh)
        fila |= {"DSR (N=%d)" % len(sh): dsr.dsr, "PSR(0)": dsr.psr_zero}
        filas.append(fila)
        curvas[nombre] = eq / eq.iloc[0]
        anual[nombre] = (eq / eq.iloc[0]).resample("YE").last()
        r.trades.assign(salida=m.cargar().fechas[r.trades.t_sal]).drop(columns=["j"]).to_csv(
            OUT / f"operaciones_{nombre.split(' ')[0]}.csv", index=False, float_format="%.4f")
        print(nombre, "ok", flush=True)
    anual["SPY"] = (spy / spy.iloc[0]).resample("YE").last()
    a = pd.DataFrame(anual)
    a = a.pct_change().fillna(a.iloc[0] - 1)  # el primer año parte de 1
    a.index = a.index.year
    df = pd.DataFrame(filas).set_index("estrategia")
    df.T.to_csv(OUT / "finalistas.csv", float_format="%.4f")
    a.to_csv(OUT / "finalistas_anual.csv", float_format="%.4f")
    (OUT / "finalistas.json").write_text(json.dumps(
        {"tabla": df.T.to_dict(), "anual": a.to_dict(), "cfg": FINALISTAS}, ensure_ascii=False, indent=1,
        default=float))
    print(df.T.round(3).to_string())
    print((a * 100).round(1).to_string())

    fig, ax = plt.subplots(figsize=(11, 6))
    for k, v in curvas.items():
        ax.plot(v.index, v.values, lw=2.4 if k == "SPY" else 1.3, color="black" if k == "SPY" else None, label=k)
    ax.set_yscale("log"); ax.axvline(pd.Timestamp("2016-01-01"), color="grey", ls=":")
    ax.set_title("SPY + compra de caídas rápidas en tendencia — 1 $ en 2002 (log), costes 10 pb/lado")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(OUT / "finalistas.png", dpi=130)


if __name__ == "__main__":
    main()
