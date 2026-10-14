"""Evaluación de módulos de mejora de la entrada sobre la tabla de eventos.

Para una señal base (por defecto: r10 <= -10 % y precio > SMA200), cada módulo parte
los eventos en grupos según el ORIGEN de la caída (mercado, grupo de pares, gemela,
propia), su NATURALEZA (noticia/hueco con volumen, resultados, gradual) o el RÉGIMEN.
Se mide en cada grupo:

- ``op``: retorno medio de la operación real (entrada apertura t+1, salida al primer
  cierre > SMA5, máx. 20 d) NETO de 20 pb de ida y vuelta.
- ``x5``/``x10``: exceso sobre el equiponderado a 5/10 días.
- ``t``: t de Newey-West de ``op`` promediando primero por fecha.

Dos ventanas: BÚSQUEDA (oct-2021 → hoy, donde se eligen cosas) y VALIDACIÓN
(2002 → sep-2021, que no se usa para elegir). Un módulo "mejora" si su grupo
favorable bate a la base en las dos.

Además, ``cuantiles``: cada rasgo continuo en quintiles dentro de la base, para no
descartar ideas por haber elegido mal los umbrales.

Uso: python research/caidas/modulos.py [base]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import eventos as ev
from estudio import nw_t

OUT = Path(__file__).parent / "resultados"
CORTE = pd.Timestamp("2021-10-01")
COSTE = 0.002

BASES = {
    "r10_10_tend": lambda d: (d.r10 <= -0.10) & (d.sobre_sma200 == 1),
    "r5_7_tend": lambda d: (d.r5 <= -0.07) & (d.sobre_sma200 == 1),
    "r10_10": lambda d: d.r10 <= -0.10,
    "dd20_15": lambda d: d.dd20 <= -0.15,
}

# (módulo, grupo) -> condición. Cada módulo es una partición interpretable.
MODULOS = {
    "origen: mercado (SPY 10d)": {
        "SPY cae ≥5 %": lambda d: d.spy10 <= -0.05,
        "SPY -5..-2 %": lambda d: (d.spy10 > -0.05) & (d.spy10 <= -0.02),
        "SPY plano/sube": lambda d: d.spy10 > -0.02,
    },
    "origen: grupo de pares (10d)": {
        "pares caen ≥7 %": lambda d: d.pares10 <= -0.07,
        "pares -7..-2 %": lambda d: (d.pares10 > -0.07) & (d.pares10 <= -0.02),
        "pares planos/suben": lambda d: d.pares10 > -0.02,
    },
    "origen: separación de sus pares": {
        "rel. pares ≤ -12 %": lambda d: d.rel_pares10 <= -0.12,
        "rel. pares -12..-6 %": lambda d: (d.rel_pares10 > -0.12) & (d.rel_pares10 <= -0.06),
        "rel. pares > -6 % (cae con su grupo)": lambda d: d.rel_pares10 > -0.06,
    },
    "origen: componente propia (idio/total)": {
        "≥80 % propia": lambda d: d.idio10 / d.r10 >= 0.8,
        "50-80 % propia": lambda d: (d.idio10 / d.r10 >= 0.5) & (d.idio10 / d.r10 < 0.8),
        "<50 % propia (sistémica)": lambda d: d.idio10 / d.r10 < 0.5,
    },
    "gemela (par nº1, corr ≥ 0.6)": {
        "gemela sube (diverge)": lambda d: (d.gem_corr >= 0.6) & (d.gem10 > 0),
        "gemela -5..0 %": lambda d: (d.gem_corr >= 0.6) & (d.gem10 <= 0) & (d.gem10 > -0.05),
        "gemela también cae ≥5 %": lambda d: (d.gem_corr >= 0.6) & (d.gem10 <= -0.05),
        "sin gemela clara": lambda d: ~(d.gem_corr >= 0.6),
    },
    "spread con gemela (z 60d)": {
        "z ≤ -3": lambda d: (d.gem_corr >= 0.6) & (d.gem_z <= -3),
        "z -3..-2": lambda d: (d.gem_corr >= 0.6) & (d.gem_z > -3) & (d.gem_z <= -2),
        "z > -2": lambda d: (d.gem_corr >= 0.6) & (d.gem_z > -2),
    },
    "naturaleza: noticia (hueco + volumen)": {
        "hueco ≤-5 % con vol ≥3×": lambda d: (d.noche_min10 <= -0.05) & (d.volmax10 >= 3),
        "hueco ≤-5 % sin volumen": lambda d: (d.noche_min10 <= -0.05) & (d.volmax10 < 3),
        "sin hueco (caída gradual)": lambda d: d.noche_min10 > -0.05,
    },
    "naturaleza: resultados": {
        "resultados en ≤10 d": lambda d: d.dias_res <= 10,
        "resultados hace >10 d": lambda d: d.dias_res > 10,
        "sin dato": lambda d: d.dias_res.isna(),
    },
    "naturaleza: sorpresa de resultados (≤10 d)": {
        "sorpresa negativa": lambda d: (d.dias_res <= 10) & (d.sorpresa < 0),
        "sorpresa positiva": lambda d: (d.dias_res <= 10) & (d.sorpresa >= 0),
    },
    "día de entrada: IBS": {
        "cierra en mínimos (IBS ≤0.2)": lambda d: d.ibs <= 0.2,
        "IBS 0.2-0.6": lambda d: (d.ibs > 0.2) & (d.ibs <= 0.6),
        "cierra arriba (IBS >0.6, giro)": lambda d: d.ibs > 0.6,
    },
    "volumen (5d vs 60d)": {
        "volumen ≥2×": lambda d: d.vol_rel5 >= 2,
        "1.2-2×": lambda d: (d.vol_rel5 >= 1.2) & (d.vol_rel5 < 2),
        "<1.2× (sin pánico)": lambda d: d.vol_rel5 < 1.2,
    },
    "régimen: volatilidad SPY (20d)": {
        "vol ≥25 %": lambda d: d.volreal_spy >= 0.25,
        "15-25 %": lambda d: (d.volreal_spy >= 0.15) & (d.volreal_spy < 0.25),
        "<15 %": lambda d: d.volreal_spy < 0.15,
    },
    "régimen: amplitud (% miembros r5≤-5 %)": {
        "≥20 % (venta general)": lambda d: d.frac_caen >= 0.2,
        "5-20 %": lambda d: (d.frac_caen >= 0.05) & (d.frac_caen < 0.2),
        "<5 % (caída aislada)": lambda d: d.frac_caen < 0.05,
    },
    "calidad: momentum 12m": {
        "mom12 ≥ 30 %": lambda d: d.mom252 >= 0.3,
        "0-30 %": lambda d: (d.mom252 >= 0) & (d.mom252 < 0.3),
        "<0": lambda d: d.mom252 < 0,
    },
    "calidad: beta": {
        "beta ≥1.3": lambda d: d.beta >= 1.3,
        "0.8-1.3": lambda d: (d.beta >= 0.8) & (d.beta < 1.3),
        "<0.8 (defensiva)": lambda d: d.beta < 0.8,
    },
    "secuencia: días seguidos cayendo": {
        "1er día": lambda d: d.dias_candidata <= 1,
        "2-4 días": lambda d: (d.dias_candidata >= 2) & (d.dias_candidata <= 4),
        "≥5 días": lambda d: d.dias_candidata >= 5,
    },
}

CONTINUOS = ["spy10", "pares10", "rel_pares10", "idio10", "gem10", "gem_z", "gem_corr", "pares_corr",
             "beta", "vol_rel", "vol_rel5", "noche1", "noche_min10", "gap10", "ibs", "volreal_spy",
             "frac_caen", "mom252", "dd252", "dd20", "r1", "r3", "r10", "r20", "dias_res", "sorpresa",
             "dias_candidata"]


def resumen(d: pd.DataFrame) -> dict:
    if len(d) < 30:
        return {"n": len(d), "dias": d.fecha.nunique(), "op": np.nan, "x5": np.nan, "x10": np.nan,
                "acierto": np.nan, "t": np.nan}
    diaria = (d.op_sma5 - COSTE).groupby(d.fecha).mean()
    return {"n": len(d), "dias": d.fecha.nunique(), "op": (d.op_sma5 - COSTE).mean(),
            "x5": d.x5.mean(), "x10": d.x10.mean(), "acierto": (d.op_sma5 > COSTE).mean(),
            "t": nw_t(diaria.to_numpy(), 5)}


def evaluar(d: pd.DataFrame) -> pd.DataFrame:
    ventanas = {"busq": d.fecha >= CORTE, "valid": d.fecha < CORTE}
    filas = []
    base = {w: resumen(d[s]) for w, s in ventanas.items()}
    filas.append({"modulo": "BASE", "grupo": "todos",
                  **{f"{w}_{k}": v for w in ventanas for k, v in base[w].items()}})
    for mod, grupos in MODULOS.items():
        for g, f in grupos.items():
            sel = f(d).fillna(False).to_numpy(bool)
            fila = {"modulo": mod, "grupo": g}
            for w, s in ventanas.items():
                fila |= {f"{w}_{k}": v for k, v in resumen(d[sel & s.to_numpy()]).items()}
            filas.append(fila)
    out = pd.DataFrame(filas)
    for w in ventanas:
        out[f"{w}_mejora"] = out[f"{w}_op"] - base[w]["op"]
    out["mejora_ambas"] = (out.busq_mejora > 0) & (out.valid_mejora > 0)
    return out


def cuantiles(d: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for c in CONTINUOS:
        x = d[c]
        if x.notna().sum() < 500:
            continue
        q = pd.qcut(x.rank(method="first"), 5, labels=False)
        for w, s in {"busq": d.fecha >= CORTE, "valid": d.fecha < CORTE}.items():
            g = (d.op_sma5 - COSTE)[s].groupby(q[s]).mean()
            filas.append({"rasgo": c, "ventana": w, **{f"Q{int(i) + 1}": v for i, v in g.items()},
                          "Q5-Q1": g.iloc[-1] - g.iloc[0] if len(g) == 5 else np.nan,
                          "corte_Q1": x.quantile(0.2), "corte_Q5": x.quantile(0.8)})
    return pd.DataFrame(filas)


def main() -> None:
    nombre = sys.argv[1] if len(sys.argv) > 1 else "r10_10_tend"
    d = ev.cargar()
    d = d[BASES[nombre](d).fillna(False)].copy()
    res = evaluar(d)
    res.to_csv(OUT / f"modulos_{nombre}.csv", index=False, float_format="%.5f")
    cq = cuantiles(d)
    cq.to_csv(OUT / f"cuantiles_{nombre}.csv", index=False, float_format="%.5f")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    cols = ["modulo", "grupo", "busq_n", "busq_op", "busq_t", "valid_n", "valid_op", "valid_t", "mejora_ambas"]
    r = res[cols].copy()
    for c in ("busq_op", "valid_op"):
        r[c] = (r[c] * 100).round(2)
    print(r.round(2).to_string(index=False))
    piv = cq.pivot(index="rasgo", columns="ventana", values="Q5-Q1") * 100
    print((piv.round(2)).sort_values("valid").to_string())


if __name__ == "__main__":
    main()
