"""Finalistas de la ronda 3 (datos corregidos, módulos v1+v2, motor v2).

D1     : regla de selección pre-fijada (máximo CAGR en búsqueda con maxDD >= -30 %).
         Caída >=15 % desde el máximo de 20 d, en tendencia, >=5 módulos a favor,
         prioridad módulos + modelo, K=3 con tamaño por convicción, salida > SMA10
         (máx. 20 d), efectivo en SPY.
D1-def : idéntica con la reserva en SPY solo si SPY > SMA200 (si no, efectivo).
C1     : finalista explicable de la ronda 2, para comparar.

Pruebas: ventanas, costes x2/x3, entrada con 1 día de retraso, sin las que luego
salieron del índice, Sharpe deflactado (N = todas las carteras probadas en el
proyecto), bootstrap estacionario del CAGR, años naturales, operaciones y señales
del último día.

Uso: python research/caidas/finalistas3.py
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
import motor_cartera as mc
import ronda3 as r3

sys.path.insert(0, str(m.ROOT))
from earnings_alpha.stats.performance import deflated_sharpe_ratio  # noqa: E402

OUT = Path(__file__).parent / "resultados"
CACHE = ev.SALIDA.parent
VENTANAS = {"búsqueda oct21-oct26": ("2021-10-01", None), "validación 2002-sep21": ("2002-01-02", "2021-10-01"),
            "total 2002-2026": ("2002-01-02", None), "2002-2025": ("2002-01-02", "2026-01-01")}


def candidatas_d1(d: pd.DataFrame) -> pd.DataFrame:
    x = d[(d.dd20 <= -0.15) & (d.score_v2 >= 5)].copy()
    x["prio"] = x.score_v2 + x.meta_pct.fillna(0.5) * 0.99
    x["peso"] = np.clip((x.score_v2 - 4) / 2, 0.5, 2.0)
    return x


def candidatas_c1(d: pd.DataFrame, e: pd.DataFrame) -> pd.DataFrame:
    """C1 de la ronda 2 (puntuación de 13 banderas limpias v1)."""
    from combinada import puntuar
    p = puntuar(e)
    x = p[(p.r5 <= -0.07) & (p.sobre_sma200 == 1) & (p.score_limpio >= 6)].copy()
    x["prio"] = x.score_limpio + (-x.r10.fillna(0)).clip(0, 0.99)
    x["peso"] = 1.0
    return x


ESTRATEGIAS = {
    "D1 · máx. CAGR (reserva SPY)": dict(sel="d1", K=3, H=20, salida="sma10", reserva="spy", conv=True),
    "D1-def · reserva SPY solo en tendencia": dict(sel="d1", K=3, H=20, salida="sma10", reserva="spy_tend", conv=True),
    "D1-def50 · ídem con tope 50 % por posición": dict(sel="d1", K=3, H=20, salida="sma10", reserva="spy_tend",
                                                       conv=True, tope=0.5),
    "C1 · ronda 2 (referencia)": dict(sel="c1", K=5, H=20, salida="rsi2", reserva="spy_tend", conv=False),
}


def bootstrap_cagr(ret: np.ndarray, n: int = 1000, bloque: int = 20, seed: int = 7) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    T = len(ret)
    out = np.empty(n)
    for k in range(n):
        idx = []
        while len(idx) < T:  # bootstrap estacionario (bloques de longitud geométrica)
            a = rng.integers(T)
            L = rng.geometric(1 / bloque)
            idx.extend(range(a, min(a + L, T)))
        r = ret[np.array(idx[:T])]
        out[k] = np.prod(1 + r) ** (252 / T) - 1
    return float(np.quantile(out, 0.05)), float(np.quantile(out, 0.95))


def main() -> None:
    d = r3.cargar()
    e = ev.cargar()
    p = m.cargar()
    actuales = p.miembro[-1]
    fuentes = {"d1": candidatas_d1(d), "c1": candidatas_c1(d, e)}
    mats = [np.load(CACHE / f) for f in ("rejilla_retornos.npy", "rejilla_retornos_fase2.npy",
                                          "ronda2_retornos.npy", "ronda3_retornos.npy")]
    sh = np.concatenate([x[:, 1:].mean(1) / x[:, 1:].std(1) for x in mats])
    n_pruebas = sum(len(x) for x in mats) + 720  # + carteras de combinada.py (sin serie guardada)
    spy = m.benchmark("SPY")
    curvas = {"SPY": spy / spy.iloc[0]}
    filas, ops_d1 = [], None
    señales = {}
    for nombre, c in ESTRATEGIAS.items():
        x = fuentes[c["sel"]]

        def run(y=x, **extra):
            kw = dict(K=c["K"], H=c["H"], salida=c["salida"], reserva=c["reserva"]) | extra
            w = y.peso.to_numpy() if c["conv"] else None
            if w is not None and c.get("tope"):
                w = np.minimum(w, c["tope"] * c["K"])  # peso w ocupa w/K del capital
            return mc.simular(y.t.to_numpy(), y.j.to_numpy(), y.prio.to_numpy(np.float64), peso=w, **kw)

        fila = {"estrategia": nombre}
        for w, (a, b) in VENTANAS.items():
            r = run(inicio=a, fin=b)
            fila |= {f"CAGR {w}": r["cagr"], f"maxDD {w}": r["maxdd"]}
        tot = run(detalle=True)
        eq = tot["equity"]
        ret = eq.pct_change().fillna(0).to_numpy()[1:]
        fila |= {"Sharpe": tot["sharpe"], "vol": tot["vol"], "operaciones/año": tot["trades_año"],
                 "acierto": tot["acierto"], "ret. medio op.": tot["ret_medio_trade"],
                 "% tiempo en acciones": tot["exposicion"]}
        for bps in (20, 30):
            fila[f"CAGR coste {bps}pb"] = run(coste_bps=bps)["cagr"]
        tarde = x.copy()
        tarde["t"] = tarde.t + 1  # la señal se ejecuta un día más tarde (apertura de t+2)
        tarde = tarde[tarde.t < len(p.fechas)]
        fila["CAGR entrando 1 día tarde"] = run(y=tarde)["cagr"]
        fila["CAGR sin las que luego salieron del índice"] = run(y=x[actuales[x.j.to_numpy()]])["cagr"]
        lo, hi = bootstrap_cagr(ret)
        fila["CAGR IC90 bootstrap"] = f"{lo:.1%} … {hi:.1%}"
        dsr = deflated_sharpe_ratio(ret, trial_sharpes=sh, n_trials=n_pruebas)
        ef = json.loads((OUT / "dsr_efectivo.json").read_text())
        dsr_ef = deflated_sharpe_ratio(ret, sr_variance=ef["var_sharpe_clusters"], n_trials=ef["n_efectivo"])
        fila |= {"DSR (N bruto)": dsr.dsr, "N bruto": n_pruebas, "DSR (N efectivo)": dsr_ef.dsr,
                 "N efectivo (corr>0,9)": ef["n_efectivo"]}
        curvas[nombre] = eq / eq.iloc[0]
        filas.append(fila)
        ops = tot["operaciones"]
        abiertas = ops[ops.t_sal < 0]
        señales[nombre] = {"fecha_datos": str(p.fechas[-1]),
                           "posiciones_abiertas": abiertas[["ticker", "entrada", "px_ent", "px_sal", "ret", "peso"]]
                           .assign(entrada=lambda z: z.entrada.astype(str)).to_dict("records"),
                           "ordenes_proxima_apertura": tot["pendientes"]}
        if nombre.startswith("D1-def50"):
            ops_d1 = ops
        print(nombre, "ok", flush=True)

    df = pd.DataFrame(filas).set_index("estrategia")
    df.T.to_csv(OUT / "finalistas3.csv")
    anual = pd.DataFrame({k: v.resample("YE").last() for k, v in curvas.items()})
    anual = anual.pct_change().fillna(anual.iloc[0] - 1)
    anual.index = anual.index.year
    anual.to_csv(OUT / "finalistas3_anual.csv", float_format="%.4f")
    ops_d1.drop(columns=["t_ent", "t_sal", "j"]).to_csv(OUT / "operaciones_D1def.csv", index=False, float_format="%.4f")
    (OUT / "senales_ultimo_dia.json").write_text(json.dumps(señales, ensure_ascii=False, indent=1, default=str))
    pd.set_option("display.width", 220)
    print(df.T.to_string())
    print((anual * 100).round(1).to_string())
    print(json.dumps(señales, ensure_ascii=False, indent=1, default=str)[:3000])

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for k, v in curvas.items():
        kw = dict(lw=2.4, color="black") if k == "SPY" else dict(lw=1.4)
        ax.plot(v.index, v.values, label=k, **kw)
        ax2.plot(v.index, (v / v.cummax() - 1).values * 100, **kw)
    ax.set_yscale("log")
    for a in (ax, ax2):
        a.axvline(pd.Timestamp("2021-10-01"), color="grey", ls=":")
        a.grid(alpha=0.3)
    ax.set_title("Compra de caídas con módulos v1+v2 y tamaño por convicción — 1 $ en 2002 (log), costes 10 pb/lado")
    ax.legend(fontsize=8)
    ax2.set_ylabel("caída desde máximo (%)")
    fig.tight_layout()
    fig.savefig(OUT / "finalistas3.png", dpi=130)


if __name__ == "__main__":
    main()
