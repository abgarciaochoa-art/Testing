"""Validación final de la escalera de la ronda 4: D1 + C1 al 50 % (capital separado,
rebalanceo mensual), hasta 5 huecos de 1/3, palanca máxima solo en tendencia.

Niveles fijados de antemano (no elegidos por la validación). La palanca es la
exposición bruta REAL máxima (acciones + exposición a SPY de la reserva, incluida su
palanca interna):
1. máx. 1x, sin deuda: reserva SPY en tendencia / SHY fuera;
2. máx. 1,5x: reserva gestionada por volatilidad (objetivo 15 %, hasta 1,5x SPY) y el
   exceso de palanca de las compras escalado por la misma regla (Moreira y Muir, 2017);
3. máx. 2x (el límite estándar de margen): ídem.
Variantes sin gestión por volatilidad (reserva SPY/SHY por tendencia) como contraste.

Para cada nivel: ventanas, costes x2/x3, entrada 1 día tarde, deuda al 7 %, los peores
episodios de caída, años naturales y Sharpe deflactado (N bruto y N efectivo).

Uso: python research/caidas/ronda4_final.py
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

import escalera as es
import motor as m
import motor_v3 as v3
from combinar import mezclar
from ronda4 import componentes

sys.path.insert(0, str(m.ROOT))
from earnings_alpha.stats.performance import deflated_sharpe_ratio  # noqa: E402

OUT = Path(__file__).parent / "resultados"
NIVELES = {
    "Nivel 1 · máx. 1x (sin deuda)": dict(reserva_="spy_tend_shy", palanca=1.0, vol_objetivo=None),
    "Nivel 2 · máx. 1,5x gestionada": dict(reserva_="spy_vg_shy", palanca=1.5, vol_objetivo=0.15),
    "Nivel 3 · máx. 2x gestionada": dict(reserva_="spy_vg_shy", palanca=2.0, vol_objetivo=0.15),
    "contraste · máx. 1,5x sin gestión": dict(reserva_="spy_tend_shy", palanca=1.5, vol_objetivo=None),
    "contraste · máx. 2x sin gestión": dict(reserva_="spy_tend_shy", palanca=2.0, vol_objetivo=None),
}


def cartera(nivel: dict, coste_bps: float = 10.0, tasa: float = 0.05, retraso: int = 0,
            con_bruto: bool = False):
    comp = componentes()
    cc = {}
    brutos = []
    for c in ("D1", "C1"):
        x, salida, conv = comp[c]
        if retraso:
            x = x.assign(t=x.t + retraso)
            x = x[x.t < len(m.cargar().fechas)]
        cfg = dict(palanca_off=1.0, K=5, frac=1 / 3, coste_bps=coste_bps, tasa_deuda=tasa) | nivel
        w = np.minimum(x.peso.to_numpy(), 0.5 / cfg["frac"]) if conv else None
        r = v3.simular(x.t.to_numpy(), x.j.to_numpy(), x.prio.to_numpy(np.float64), peso=w, H=20,
                       salida=salida, **cfg)
        cc[c] = r["equity"]
        brutos.append(r["serie_lev"])
    eq = mezclar(cc, {"D1": 0.5, "C1": 0.5})
    if con_bruto:
        return eq, 0.5 * (brutos[0] + brutos[1])
    return eq


def episodios(eq: pd.Series, n: int = 3) -> list[dict]:
    """Los n peores episodios de caída (pico → valle → recuperación)."""
    dd = eq / eq.cummax() - 1
    out = []
    usado = pd.Series(False, index=eq.index)
    for _ in range(n):
        d = dd[~usado]
        if d.empty or d.min() >= 0:
            break
        valle = d.idxmin()
        pico = eq[:valle].idxmax()
        rec = eq[valle:][eq[valle:] >= eq[pico]]
        fin = rec.index[0] if len(rec) else eq.index[-1]
        out.append({"pico": str(pico.date()), "valle": str(valle.date()), "recupera": str(fin.date()),
                    "caida": float(dd[valle])})
        usado[(eq.index >= pico) & (eq.index <= fin)] = True
    return out


def main() -> None:
    ef = json.loads((OUT / "dsr_efectivo.json").read_text())
    cache = m.ROOT / "data" / "cache" / "caidas"
    mats = [np.load(cache / f) for f in ("rejilla_retornos.npy", "rejilla_retornos_fase2.npy",
                                          "ronda2_retornos.npy", "ronda3_retornos.npy", "ronda4_retornos.npy")]
    T = min(x.shape[1] for x in mats)
    sh = np.concatenate([x[:, -T + 1:].mean(1) / x[:, -T + 1:].std(1) for x in mats])
    n_bruto = sum(len(x) for x in mats) + 720 + 300  # + combinada.py y pruebas de la ronda 4 sin serie guardada
    filas, curvas, eps = [], {}, {}
    for nombre, pal in NIVELES.items():
        eq, bruto = cartera(pal, con_bruto=True)
        if not nombre.startswith("contraste"):
            curvas[nombre] = eq / eq.iloc[0]
        fila = {"nivel": nombre, **es.metricas_ventanas(eq), "exposición bruta media": float(bruto.mean()),
                "exposición bruta p95": float(bruto.quantile(0.95)), "exposición bruta máx.": float(bruto.max())}
        fila["CAGR coste 20pb"] = m.metricas(cartera(pal, coste_bps=20))["cagr"]
        fila["CAGR coste 30pb"] = m.metricas(cartera(pal, coste_bps=30))["cagr"]
        fila["CAGR entrando 1 día tarde"] = m.metricas(cartera(pal, retraso=1))["cagr"]
        fila["CAGR deuda al 7 %"] = m.metricas(cartera(pal, tasa=0.07))["cagr"]
        ret = eq.pct_change().fillna(0).to_numpy()[1:]
        fila["DSR (N bruto)"] = deflated_sharpe_ratio(ret, trial_sharpes=sh, n_trials=n_bruto).dsr
        fila["DSR (N efectivo)"] = deflated_sharpe_ratio(ret, sr_variance=ef["var_sharpe_clusters"],
                                                         n_trials=ef["n_efectivo"]).dsr
        eps[nombre] = episodios(eq)
        filas.append(fila)
        print(nombre, "ok", flush=True)
    spy = m.benchmark("SPY")
    curvas["SPY"] = spy / spy.iloc[0]
    eps["SPY"] = episodios(spy)
    df = pd.DataFrame(filas).set_index("nivel")
    df.T.to_csv(OUT / "ronda4_final.csv")
    anual = pd.DataFrame({k: v.resample("YE").last() for k, v in curvas.items()})
    anual = anual.pct_change().fillna(anual.iloc[0] - 1)
    anual.index = anual.index.year
    anual.to_csv(OUT / "ronda4_final_anual.csv", float_format="%.4f")
    (OUT / "ronda4_final_episodios.json").write_text(json.dumps(eps, ensure_ascii=False, indent=1))
    pd.set_option("display.width", 220)
    print(df.T.round(3).to_string())
    print((anual * 100).round(1).to_string())
    print(json.dumps(eps, ensure_ascii=False, indent=1))

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for k, v in curvas.items():
        kw = dict(lw=2.4, color="black") if k == "SPY" else dict(lw=1.4)
        ax.plot(v.index, v.values, label=k, **kw)
        ax2.plot(v.index, (v / v.cummax() - 1).values * 100, **kw)
    ax.set_yscale("log")
    for a in (ax, ax2):
        a.axvline(pd.Timestamp("2021-10-01"), color="grey", ls=":")
        a.grid(alpha=0.3)
    ax.set_title("Ronda 4 — D1 + C1 (50/50): escalera de riesgo frente a SPY — 1 $ en 2002 (log), costes 10 pb/lado")
    ax.legend(fontsize=8)
    ax2.set_ylabel("caída desde máximo (%)")
    fig.tight_layout()
    fig.savefig(OUT / "ronda4_final.png", dpi=130)


if __name__ == "__main__":
    main()
