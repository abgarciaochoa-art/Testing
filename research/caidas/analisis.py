"""Selección honesta, validación y figuras de las estrategias de caídas.

Regla de selección (fijada antes de mirar el OOS): para cada familia se elige la
configuración con mayor **Sharpe in-sample (2002-2015)**; además, la de mayor CAGR
in-sample con caída máxima in-sample mejor que -50 %. Después se informa el
2016-2026 sin tocar nada.

Validación:
- Sharpe deflactado (Bailey y López de Prado) con N = TODAS las pruebas de las dos
  fases de la rejilla y su dispersión real de Sharpes.
- PBO por CSCV sobre la matriz completa de retornos diarios.
- Costes al doble (20 pb por lado).
- Sesgo de supervivencia: misma estrategia sobre los miembros ACTUALES (lo que
  haría un backtest ingenuo) frente a la pertenencia point-in-time.

Uso: python research/caidas/analisis.py
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

sys.path.insert(0, str(m.ROOT))
from earnings_alpha.stats.performance import deflated_sharpe_ratio  # noqa: E402
from earnings_alpha.stats.validation import pbo_cscv  # noqa: E402

OUT = Path(__file__).parent / "resultados"
CACHE = m.ROOT / "data" / "cache" / "caidas"
CLAVES = ["familia", "tipo", "n", "x", "filtro", "salida", "h", "k", "max_dia", "reserva"]


def cfg_de(fila: pd.Series) -> dict:
    c = {}
    for k in CLAVES:
        if k in fila and pd.notna(fila[k]):
            v = fila[k]
            c[k] = int(v) if k in ("n", "h", "k", "max_dia") else v
    return c


def nombre(c: dict) -> str:
    base = {"dd": f"cae ≥{c.get('x', 0):.0%} desde máx. {c['n']}d",
            "ret": f"cae ≥{c.get('x', 0):.0%} en {c['n']}d",
            "rsi": f"RSI{c['n']} ≤ {c.get('x')}", "racha": f"{c['n']} cierres bajistas",
            "z": f"z{c['n']} ≤ -{c.get('x')}"}[c["tipo"]]
    sal = {"tiempo": f"sale a {c['h']}d", "sma5": f"sale > SMA5 (máx {c['h']}d)",
           "sma20": f"sale > SMA20 (máx {c['h']}d)", "rsi2": f"sale RSI2>70 (máx {c['h']}d)"}[c["salida"]]
    extra = f", K={c['k']}" + (", reserva SPY" if c.get("reserva") else "") + (
        f", filtro {c['filtro']}" if c.get("filtro") != "ninguno" else "")
    return f"{base}; {sal}{extra}"


def main() -> None:
    d1 = pd.read_csv(OUT / "rejilla.csv")
    d2 = pd.read_csv(OUT / "rejilla_fase2.csv")
    d = pd.concat([d1, d2], ignore_index=True)
    R = np.vstack([np.load(CACHE / "rejilla_retornos.npy"), np.load(CACHE / "rejilla_retornos_fase2.npy")])
    fechas = m.benchmark("SPY").index
    n_pruebas = len(d)
    sh_diarios = R[:, 1:].mean(1) / R[:, 1:].std(1)

    # ---------------- selección (solo in-sample)
    sel = {}
    for fam, g in d.groupby("familia"):
        sel[f"mejor Sharpe IS · {fam}"] = g.loc[g.is_sharpe.idxmax()]
    g = d[d.is_maxdd > -0.5]
    sel["mejor CAGR IS con DD IS > -50 %"] = g.loc[g.is_cagr.idxmax()]

    spy = m.benchmark("SPY")
    ew = m.equiponderado_pit()
    filas, curvas = [], {"SPY": spy / spy.iloc[0], "Equiponderado S&P 500 (PIT)": ew / ew.iloc[0]}
    for etiqueta, fila in sel.items():
        c = cfg_de(fila)
        r = m.backtest(c)
        r20 = m.backtest(c, coste_bps=20)
        rsurv = m.backtest({**c, "universo": "actual"})
        ret = r.equity.pct_change().dropna()
        oos = ret[ret.index >= "2016-01-01"]
        dsr = deflated_sharpe_ratio(ret.to_numpy(), trial_sharpes=sh_diarios)
        dsr_oos = deflated_sharpe_ratio(oos.to_numpy(), trial_sharpes=sh_diarios)
        spy_r = spy.pct_change().reindex(ret.index).fillna(0)
        beta = np.cov(ret, spy_r)[0, 1] / spy_r.var()
        alfa = (ret.mean() - beta * spy_r.mean()) * 252
        tot = m.metricas(r.equity, r.trades, r.exposicion)
        e_is = r.equity[r.equity.index < "2016-01-01"]
        e_oos = r.equity[r.equity.index >= "2015-12-31"]
        filas.append({
            "seleccion": etiqueta, "id": int(fila["id"]), "estrategia": nombre(c), "cfg": c,
            "cagr_is": m.metricas(e_is)["cagr"], "cagr_oos": m.metricas(e_oos)["cagr"],
            **{f"tot_{k}": v for k, v in tot.items()},
            "cagr_coste20": m.metricas(r20.equity)["cagr"],
            "cagr_universo_actual": m.metricas(rsurv.equity)["cagr"],
            "maxdd_oos": m.metricas(e_oos)["maxdd"],
            "beta_spy": beta, "alfa_anual": alfa,
            "dsr_total": dsr.dsr, "dsr_oos": dsr_oos.dsr, "psr0_oos": dsr_oos.psr_zero,
        })
        curvas[f"#{int(fila['id'])} {nombre(c)}"] = r.equity / r.equity.iloc[0]
        print(etiqueta, "ok", flush=True)

    res = pd.DataFrame(filas)
    # ---------------- PBO del procedimiento de búsqueda (todas las pruebas)
    pbo = pbo_cscv(pd.DataFrame(R[:, 1:].T, index=fechas[1:]), n_partitions=16)

    # ---------------- años naturales de los seleccionados
    anual = pd.DataFrame({k: v.resample("YE").last().pct_change() for k, v in curvas.items()})
    anual.iloc[0] = pd.DataFrame({k: v.resample("YE").last() for k, v in curvas.items()}).iloc[0] - 1
    anual.index = anual.index.year

    res.drop(columns="cfg").to_csv(OUT / "seleccion.csv", index=False, float_format="%.4f")
    anual.to_csv(OUT / "anual.csv", float_format="%.4f")
    resumen = {
        "n_pruebas": n_pruebas,
        "pbo": pbo.pbo, "pbo_ic": [pbo.ci_low, pbo.ci_high],
        "pbo_prob_perdida_oos": pbo.probability_of_loss, "pbo_pendiente_degradacion": pbo.degradation_slope,
        "spy": m.metricas(spy), "equiponderado_pit": m.metricas(ew),
        "spy_oos": m.metricas(spy[spy.index >= "2015-12-31"]),
        "spy_is": m.metricas(spy[spy.index < "2016-01-01"]),
        "familias": d.groupby("familia")[["is_cagr", "oos_cagr", "tot_cagr", "tot_maxdd", "tot_sharpe"]]
        .median().round(4).to_dict("index"),
        "pct_supera_spy_tot": float((d.tot_cagr > m.metricas(spy)["cagr"]).mean()),
        "seleccion": [{k: (v if not isinstance(v, (np.floating, np.integer)) else float(v)) for k, v in f.items()}
                      for f in filas],
    }
    (OUT / "resumen.json").write_text(json.dumps(resumen, indent=1, ensure_ascii=False, default=float))
    print(res.drop(columns=["cfg"]).round(3).T.to_string())
    print("PBO", pbo.pbo, pbo.ci_low, pbo.ci_high)

    # ---------------- figuras
    fig, ax = plt.subplots(figsize=(11, 6))
    for k, v in curvas.items():
        ax.plot(v.index, v.values, lw=2.2 if k in ("SPY",) else 1.3, label=k[:90],
                color="black" if k == "SPY" else None, ls="--" if k.startswith("Equi") else "-")
    ax.set_yscale("log"); ax.axvline(pd.Timestamp("2016-01-01"), color="grey", ls=":")
    ax.text(pd.Timestamp("2016-03-01"), ax.get_ylim()[0] * 1.2, "fuera de muestra →", color="grey")
    ax.set_title("Crecimiento de 1 $ (escala log), costes 10 pb/lado"); ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(OUT / "curvas.png", dpi=130)

    fig, ax = plt.subplots(figsize=(7, 6))
    for fam, g in d.groupby("familia"):
        ax.scatter(g.is_cagr * 100, g.oos_cagr * 100, s=6, alpha=0.5, label=fam)
    lim = [-10, 30]
    ax.plot(lim, lim, color="grey", lw=0.8); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.axhline(resumen["spy_oos"]["cagr"] * 100, color="black", ls="--", lw=0.8)
    ax.set_xlabel("CAGR 2002-2015 (%)"); ax.set_ylabel("CAGR 2016-2026 (%)")
    ax.set_title(f"{n_pruebas} variantes: dentro vs fuera de muestra"); ax.legend(fontsize=7, markerscale=3)
    ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(OUT / "is_vs_oos.png", dpi=130)


if __name__ == "__main__":
    main()
