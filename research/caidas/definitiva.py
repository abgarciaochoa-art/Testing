"""Estrategia definitiva de la ronda 5: D1 con ENTRADAS TÉCNICAS, solo acciones.

Regla de entrada (señal al cierre, compra a la apertura siguiente):
1. la acción es miembro del S&P 500, líquida y está sobre su SMA200;
2. cae >= 15 % desde su máximo de 20 sesiones;
3. tiene >= 5 de los 11 módulos de origen de la caída (``meta.py``);
4. tiene >= 2 de las 11 banderas técnicas (``modulos_tec.py``): estirada bajo
   EMA5/EMA10/WMA20/VWAP10, Bollinger, estocástico, SMA50, clímax de volumen, OBV.
Prioridad: módulos + modelo; hasta 5 posiciones de 1/2 hueco de convicción (medio hueco
con 5 módulos, un hueco con >= 6; nunca > 50 % del capital). Salida: cierre > SMA10
(máx. 20 sesiones).
Reserva (sin ETFs ni efectivo): las 100 acciones más líquidas ponderadas por volumen en
dólares con SPY > SMA200; las 50 de menor volatilidad cuando no (SPY solo como señal).
Palanca máxima (exposición bruta real) 1 / 1,5 / 2, solo en tendencia y escalada por la
volatilidad de SPY.

Comparación: D1-técnica + C1-técnica al 50 %. Pruebas de estrés completas.

Uso: python research/caidas/definitiva.py
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
RESERVA = "liq100_tend_minvol"
D1_CFG = dict(kmin=2, uso="filtro", tam="base", frac=0.5)
C1_CFG = dict(kmin=2, uso="prioridad", tam="base", frac=1 / 3)


def curva(comp_tec, nombre, cfg, pal, coste_bps=10.0, tasa=0.05, retraso=0, detalle=False):
    x, salida, conv = comp_tec[nombre]
    y, w = preparar(x, conv, cfg["kmin"], cfg["uso"], cfg["tam"], cfg["frac"])
    if retraso:
        y = y.assign(t=y.t + retraso)
        ok = (y.t < len(m.cargar().fechas)).to_numpy()
        y, w = y[ok], w[ok]
    return v3.simular(y.t.to_numpy(), y.j.to_numpy(), y.prio.to_numpy(np.float64), peso=w, K=5, frac=cfg["frac"],
                      H=20, salida=salida, reserva_=RESERVA, palanca=pal, palanca_off=1.0,
                      vol_objetivo=0.15 if pal > 1 else None, coste_bps=coste_bps, tasa_deuda=tasa, detalle=detalle)


def cartera(comp_tec, tipo, pal, **kw):
    if tipo == "D1-técnica":
        r = curva(comp_tec, "D1", D1_CFG, pal, **kw)
        return r["equity"], r["serie_lev"]
    a = curva(comp_tec, "D1", D1_CFG, pal, **kw)
    b = curva(comp_tec, "C1", C1_CFG, pal, **kw)
    return mezclar({"D1": a["equity"], "C1": b["equity"]}, {"D1": 0.5, "C1": 0.5}), 0.5 * (a["serie_lev"] + b["serie_lev"])


def main() -> None:
    comp_tec = componentes_tec()
    ef = json.loads((OUT / "dsr_efectivo.json").read_text())
    cache = m.ROOT / "data" / "cache" / "caidas"
    mats = [np.load(cache / f) for f in ("rejilla_retornos.npy", "rejilla_retornos_fase2.npy", "ronda2_retornos.npy",
                                          "ronda3_retornos.npy", "ronda4_retornos.npy", "ronda5_retornos.npy",
                                          "ronda5b_retornos.npy", "ronda5c_retornos.npy")]
    T = min(x.shape[1] for x in mats)
    sh = np.concatenate([x[:, -T + 1:].mean(1) / x[:, -T + 1:].std(1) for x in mats])
    n_bruto = sum(len(x) for x in mats) + 720 + 500
    filas, curvas, eps = [], {}, {}
    for tipo, pal in (("D1-técnica", 1.0), ("D1-técnica", 1.5), ("D1-técnica", 2.0),
                      ("D1-técnica + C1-técnica", 1.0), ("D1-técnica + C1-técnica", 2.0)):
        nombre = f"{tipo} · máx. {pal:g}x"
        eq, bruto = cartera(comp_tec, tipo, pal)
        curvas[nombre] = eq / eq.iloc[0]
        fila = {"cartera": nombre, **es.metricas_ventanas(eq), "exposición bruta media": float(bruto.mean()),
                "exposición bruta máx.": float(bruto.max())}
        fila["CAGR coste 20pb"] = m.metricas(cartera(comp_tec, tipo, pal, coste_bps=20)[0])["cagr"]
        fila["CAGR coste 30pb"] = m.metricas(cartera(comp_tec, tipo, pal, coste_bps=30)[0])["cagr"]
        fila["CAGR 1 día tarde"] = m.metricas(cartera(comp_tec, tipo, pal, retraso=1)[0])["cagr"]
        if pal > 1:
            fila["CAGR deuda 7 %"] = m.metricas(cartera(comp_tec, tipo, pal, tasa=0.07)[0])["cagr"]
        ret = eq.pct_change().fillna(0).to_numpy()[1:]
        fila["DSR (N bruto)"] = deflated_sharpe_ratio(ret, trial_sharpes=sh, n_trials=n_bruto).dsr
        fila["DSR (N efectivo)"] = deflated_sharpe_ratio(ret, sr_variance=ef["var_sharpe_clusters"],
                                                         n_trials=ef["n_efectivo"]).dsr
        eps[nombre] = episodios(eq)
        filas.append(fila)
        print(nombre, "ok", flush=True)
    r1 = curva(comp_tec, "D1", D1_CFG, 1.0, detalle=True)
    ops = r1["operaciones"]
    ops.drop(columns=["t_ent", "t_sal", "j"]).to_csv(OUT / "operaciones_definitiva.csv", index=False,
                                                       float_format="%.4f")
    spy = m.benchmark("SPY")
    curvas["SPY (referencia)"] = spy / spy.iloc[0]
    df = pd.DataFrame(filas).set_index("cartera")
    df.T.to_csv(OUT / "definitiva.csv")
    anual = pd.DataFrame({k: v.resample("YE").last() for k, v in curvas.items()})
    anual = anual.pct_change().fillna(anual.iloc[0] - 1)
    anual.index = anual.index.year
    anual.to_csv(OUT / "definitiva_anual.csv", float_format="%.4f")
    (OUT / "definitiva.json").write_text(json.dumps({"D1": D1_CFG, "C1": C1_CFG, "reserva": RESERVA,
                                                     "episodios": eps}, ensure_ascii=False, indent=1, default=str))
    pd.set_option("display.width", 250)
    print(df.T.round(3).to_string())
    print((anual * 100).round(1).to_string())
    for k, v in eps.items():
        print(k, [(x["pico"], x["valle"], round(x["caida"], 3)) for x in v])
    print("operaciones D1-técnica (sin deuda):", len(ops), "· acierto", round((ops.ret > 0).mean(), 3),
          "· retorno medio", round(ops.ret.mean(), 4))

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for k, v in curvas.items():
        if "C1" in k:
            continue
        kw = dict(lw=2.4, color="black") if k.startswith("SPY") else dict(lw=1.4)
        ax.plot(v.index, v.values, label=k, **kw)
        ax2.plot(v.index, (v / v.cummax() - 1).values * 100, **kw)
    ax.set_yscale("log")
    for a in (ax, ax2):
        a.axvline(pd.Timestamp("2021-10-01"), color="grey", ls=":")
        a.grid(alpha=0.3)
    ax.set_title("Definitiva — compra de caídas con entradas técnicas, solo acciones — 1 $ en 2002 (log), costes 10 pb")
    ax.legend(fontsize=8)
    ax2.set_ylabel("caída desde máximo (%)")
    fig.tight_layout()
    fig.savefig(OUT / "definitiva.png", dpi=130)


if __name__ == "__main__":
    main()
