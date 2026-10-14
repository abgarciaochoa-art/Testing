"""Evaluación de los rasgos técnicos (``tecnicos.py``) como mejoras de la ENTRADA de D1 y C1.

Para cada componente se toman sus candidatas reales y el retorno de la operación con su
propia salida (D1: > SMA10; C1: RSI2 >= 70; máx. 20 sesiones; neto de 20 pb).

Para cada rasgo se prueban 4 banderas con cortes calculados SOLO en la búsqueda:
quintil inferior (Q1), quintil superior (Q5), por debajo y por encima de la mediana.
Una bandera se ELIGE si su retorno medio supera al de la base del componente en las dos
mitades de la búsqueda (oct-21→mar-24 y abr-24→oct-26) y en los DOS componentes, con
al menos 30 operaciones en cada celda. La validación 2002-sep21 solo se informa.

Uso: python research/caidas/modulos_tec.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import eventos as ev
import finalistas3 as f3
import motor as m
import ronda3 as r3
import salidas as sl
import tecnicos as tc

OUT = Path(__file__).parent / "resultados"
CORTE = pd.Timestamp("2021-10-01")
MITAD = pd.Timestamp("2024-04-01")
RASGOS = [c for c in tc.cargar().columns if c not in ("t", "j")]


def conjuntos() -> dict[str, pd.DataFrame]:
    p = m.cargar()
    d = r3.cargar()
    e = ev.cargar()
    tec = tc.cargar()
    mats = (p.open, p.close, m.sma(5), m.sma(10), m.sma(20), m.rsi(2))
    out = {}
    for nombre, x, cod, par in (("D1", f3.candidatas_d1(d), 0, 10.0), ("C1", f3.candidatas_c1(d, e), 1, 70.0)):
        x = x.drop(columns=[c for c in RASGOS if c in x.columns]).merge(tec, on=["t", "j"], how="left")
        r, dias = sl._salidas(x.t.to_numpy().astype(np.int64), x.j.to_numpy().astype(np.int64), *mats, cod, par, 20)
        x["ret"] = r - 0.002
        x["dias"] = dias
        x["fecha"] = pd.DatetimeIndex(p.fechas[x.t.to_numpy()])
        x["ventana"] = np.where(x.fecha < CORTE, "valid", np.where(x.fecha < MITAD, "busq1", "busq2"))
        out[nombre] = x[x.ret.notna()].copy()
    return out


def banderas_candidatas(x_busq: pd.DataFrame) -> dict:
    """Cortes calculados SOLO con la búsqueda (unión de D1 y C1)."""
    b = {}
    for r in RASGOS:
        v = x_busq[r].dropna()
        if len(v) < 200:
            continue
        q20, q50, q80 = np.quantile(v, [0.2, 0.5, 0.8])
        b[f"{r} ≤ {q20:.4g} (Q1)"] = (r, "<=", q20)
        b[f"{r} ≥ {q80:.4g} (Q5)"] = (r, ">=", q80)
        b[f"{r} < {q50:.4g} (mitad baja)"] = (r, "<", q50)
        b[f"{r} ≥ {q50:.4g} (mitad alta)"] = (r, ">=", q50)
    return b


def aplicar(x: pd.DataFrame, regla) -> np.ndarray:
    r, op, v = regla
    s = x[r]
    return {"<=": s <= v, ">=": s >= v, "<": s < v}[op].fillna(False).to_numpy()


def main() -> None:
    cj = conjuntos()
    busq = pd.concat([x[x.ventana != "valid"] for x in cj.values()])
    cand = banderas_candidatas(busq)
    filas = []
    for nombre, regla in cand.items():
        fila = {"bandera": nombre, "rasgo": regla[0]}
        ok = True
        for comp, x in cj.items():
            sel = aplicar(x, regla)
            for w in ("busq1", "busq2", "valid"):
                en = (x.ventana == w).to_numpy()
                base = x.ret[en].mean()
                g = x.ret[en & sel]
                fila[f"{comp}_{w}_n"] = int(len(g))
                fila[f"{comp}_{w}_mejora"] = (g.mean() - base) if len(g) else np.nan
                if w != "valid" and (len(g) < 30 or not g.mean() > base):
                    ok = False
        fila["elegida"] = ok
        filas.append(fila)
    df = pd.DataFrame(filas)
    df["valid_ok"] = (df.D1_valid_mejora > 0) & (df.C1_valid_mejora > 0)
    df.to_csv(OUT / "modulos_tec.csv", index=False, float_format="%.5f")
    # correlación de rango rasgo-retorno por ventana (descriptivo)
    corr = []
    for comp, x in cj.items():
        for r in RASGOS:
            fila = {"componente": comp, "rasgo": r}
            for w in ("busq1", "busq2", "valid"):
                y = x[x.ventana == w][[r, "ret"]].dropna()
                fila[w] = y[r].rank().corr(y.ret.rank()) if len(y) > 50 else np.nan
            corr.append(fila)
    cdf = pd.DataFrame(corr)
    cdf.to_csv(OUT / "modulos_tec_corr.csv", index=False, float_format="%.4f")
    elegidas = df[df.elegida]
    (OUT / "modulos_tec_elegidas.json").write_text(json.dumps(
        {k: [cand[k][0], cand[k][1], float(cand[k][2])] for k in elegidas.bandera}, ensure_ascii=False, indent=1))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    cols = ["bandera", "D1_busq1_mejora", "D1_busq2_mejora", "C1_busq1_mejora", "C1_busq2_mejora",
            "D1_valid_mejora", "C1_valid_mejora", "D1_valid_n", "C1_valid_n", "valid_ok"]
    v = elegidas[cols].copy()
    for c in cols[1:7]:
        v[c] = (v[c] * 100).round(2)
    print(f"{len(df)} banderas candidatas, {len(elegidas)} elegidas con la búsqueda; "
          f"{int(elegidas.valid_ok.sum())} de ellas mejoran también en la validación (D1 y C1)")
    print(v.to_string(index=False))
    print("\nCorrelación de rango rasgo-retorno (búsqueda 1 / búsqueda 2 / validación):")
    pv = cdf.pivot(index="rasgo", columns="componente", values=["busq1", "busq2", "valid"]).round(3)
    print(pv.to_string())


if __name__ == "__main__":
    main()
