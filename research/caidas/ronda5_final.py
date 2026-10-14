"""Cartera final de la ronda 5: D1 + C1 (50 %) con entradas técnicas y reserva SOLO de
acciones (sin ETFs ni efectivo).

Para cada nivel de palanca (1, 1,5, 2; solo con SPY > SMA200 y escalada por la
volatilidad de SPY, que se usa solo como señal), la configuración de cada componente
(filtro/prioridad técnica, tamaño, hueco) se toma AUTOMÁTICAMENTE como la de mayor CAGR
en la búsqueda en ``ronda5b.csv``. Reservas: ``liq100_dv`` (siempre las 100 más
líquidas) y ``liq100_tend_minvol`` (las 50 de menor volatilidad fuera de tendencia).

Pruebas: ventanas, costes x2/x3, entrada 1 día tarde, deuda al 7 %, exposición bruta
real, episodios de caída, años naturales, Sharpe deflactado (N bruto y efectivo).

Uso: python research/caidas/ronda5_final.py
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
from ronda4_final import episodios
from ronda5 import componentes_tec
from ronda5b import preparar

sys.path.insert(0, str(m.ROOT))
from earnings_alpha.stats.performance import deflated_sharpe_ratio  # noqa: E402

OUT = Path(__file__).parent / "resultados"
SALIDAS = {"D1": "sma10", "C1": "rsi2"}


def elegir(rb: pd.DataFrame, comp: str, pal: float) -> dict:
    g = rb[(rb.componente == comp) & (rb.palanca == pal)].sort_values("busq_cagr", ascending=False).iloc[0]
    return {"kmin": int(g.kmin), "uso": g.uso, "tam": g["tamaño"], "frac": float(g.hueco)}


def cartera(comp_tec: dict, cfgs: dict, pal: float, reserva: str, coste_bps=10.0, tasa=0.05, retraso=0):
    cc, brutos = {}, []
    for c in ("D1", "C1"):
        x, salida, conv = comp_tec[c]
        k = cfgs[c]
        y, w = preparar(x, conv, k["kmin"], k["uso"], k["tam"], 1 / 3 if abs(k["frac"] - 0.333) < 0.01 else 0.5)
        if retraso:
            y = y.assign(t=y.t + retraso)
            ok = (y.t < len(m.cargar().fechas)).to_numpy()
            y, w = y[ok], w[ok]
        frac = 1 / 3 if abs(k["frac"] - 0.333) < 0.01 else 0.5
        r = v3.simular(y.t.to_numpy(), y.j.to_numpy(), y.prio.to_numpy(np.float64), peso=w, K=5, frac=frac, H=20,
                       salida=salida, reserva_=reserva, palanca=pal, palanca_off=1.0,
                       vol_objetivo=0.15 if pal > 1 else None, coste_bps=coste_bps, tasa_deuda=tasa)
        cc[c] = r["equity"]
        brutos.append(r["serie_lev"])
    return mezclar(cc, {"D1": 0.5, "C1": 0.5}), 0.5 * (brutos[0] + brutos[1])


def main() -> None:
    rb = pd.read_csv(OUT / "ronda5b.csv")
    comp_tec = componentes_tec()
    ef = json.loads((OUT / "dsr_efectivo.json").read_text())
    cache = m.ROOT / "data" / "cache" / "caidas"
    mats = [np.load(cache / f) for f in ("rejilla_retornos.npy", "rejilla_retornos_fase2.npy", "ronda2_retornos.npy",
                                          "ronda3_retornos.npy", "ronda4_retornos.npy", "ronda5_retornos.npy",
                                          "ronda5b_retornos.npy")]
    T = min(x.shape[1] for x in mats)
    sh = np.concatenate([x[:, -T + 1:].mean(1) / x[:, -T + 1:].std(1) for x in mats])
    n_bruto = sum(len(x) for x in mats) + 720 + 400
    filas, curvas, eps, elecciones = [], {}, {}, {}
    for reserva in ("liq100_dv", "liq100_tend_minvol"):
        for pal in (1.0, 1.5, 2.0):
            cfgs = {c: elegir(rb, c, pal) for c in ("D1", "C1")}
            nombre = f"{'líquidas siempre' if reserva == 'liq100_dv' else 'líquidas / baja vol.'} · máx. {pal:g}x"
            elecciones[nombre] = cfgs
            eq, bruto = cartera(comp_tec, cfgs, pal, reserva)
            curvas[nombre] = eq / eq.iloc[0]
            fila = {"cartera": nombre, **es.metricas_ventanas(eq),
                    "exposición bruta media": float(bruto.mean()), "exposición bruta máx.": float(bruto.max())}
            fila["CAGR coste 30pb"] = m.metricas(cartera(comp_tec, cfgs, pal, reserva, coste_bps=30)[0])["cagr"]
            fila["CAGR 1 día tarde"] = m.metricas(cartera(comp_tec, cfgs, pal, reserva, retraso=1)[0])["cagr"]
            if pal > 1:
                fila["CAGR deuda 7 %"] = m.metricas(cartera(comp_tec, cfgs, pal, reserva, tasa=0.07)[0])["cagr"]
            ret = eq.pct_change().fillna(0).to_numpy()[1:]
            fila["DSR (N bruto)"] = deflated_sharpe_ratio(ret, trial_sharpes=sh, n_trials=n_bruto).dsr
            fila["DSR (N efectivo)"] = deflated_sharpe_ratio(ret, sr_variance=ef["var_sharpe_clusters"],
                                                             n_trials=ef["n_efectivo"]).dsr
            eps[nombre] = episodios(eq)
            filas.append(fila)
            print(nombre, "ok", flush=True)
    spy = m.benchmark("SPY")
    curvas["SPY (referencia)"] = spy / spy.iloc[0]
    df = pd.DataFrame(filas).set_index("cartera")
    df.T.to_csv(OUT / "ronda5_final.csv")
    anual = pd.DataFrame({k: v.resample("YE").last() for k, v in curvas.items()})
    anual = anual.pct_change().fillna(anual.iloc[0] - 1)
    anual.index = anual.index.year
    anual.to_csv(OUT / "ronda5_final_anual.csv", float_format="%.4f")
    (OUT / "ronda5_final.json").write_text(json.dumps({"elecciones": elecciones, "episodios": eps},
                                                      ensure_ascii=False, indent=1, default=str))
    pd.set_option("display.width", 250)
    print(json.dumps(elecciones, ensure_ascii=False, indent=1))
    print(df.T.round(3).to_string())
    print((anual * 100).round(1).to_string())
    print(json.dumps(eps, ensure_ascii=False, indent=1)[:2500])

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for k, v in curvas.items():
        if "liq100" in k or k.startswith("líquidas siempre · máx. 1.5"):
            continue
        kw = dict(lw=2.4, color="black") if k.startswith("SPY") else dict(lw=1.3)
        ax.plot(v.index, v.values, label=k, **kw)
        ax2.plot(v.index, (v / v.cummax() - 1).values * 100, **kw)
    ax.set_yscale("log")
    for a in (ax, ax2):
        a.axvline(pd.Timestamp("2021-10-01"), color="grey", ls=":")
        a.grid(alpha=0.3)
    ax.set_title("Ronda 5 — entradas técnicas, solo acciones (sin ETFs ni efectivo) — 1 $ en 2002 (log)")
    ax.legend(fontsize=7)
    ax2.set_ylabel("caída desde máximo (%)")
    fig.tight_layout()
    fig.savefig(OUT / "ronda5_final.png", dpi=130)


if __name__ == "__main__":
    main()
