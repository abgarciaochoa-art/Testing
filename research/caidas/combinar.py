"""Combinación de estrategias en una cartera mayor (cada una con su propio capital).

Se combinan las curvas de D1-def50 (caídas rápidas con módulos v2, convicción) y C1
(caídas de 5 días con módulos v1, K=5) con rebalanceo MENSUAL a pesos fijos (el
rebalanceo diario sería irrealista), y se mide si la mezcla mejora el riesgo/retorno.

Uso: python research/caidas/combinar.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import eventos as ev
import finalistas3 as f3
import motor as m
import motor_cartera as mc
import ronda3 as r3

OUT = Path(__file__).parent / "resultados"


def curva(x: pd.DataFrame, c: dict) -> pd.Series:
    w = x.peso.to_numpy() if c["conv"] else None
    if w is not None and c.get("tope"):
        w = np.minimum(w, c["tope"] * c["K"])
    r = mc.simular(x.t.to_numpy(), x.j.to_numpy(), x.prio.to_numpy(np.float64), peso=w, K=c["K"], H=c["H"],
                   salida=c["salida"], reserva=c["reserva"])
    return r["equity"]


def mezclar(curvas: dict[str, pd.Series], pesos: dict[str, float]) -> pd.Series:
    """Rebalanceo al inicio de cada mes a los pesos objetivo; entre medias, deriva."""
    rets = pd.DataFrame({k: v.pct_change().fillna(0) for k, v in curvas.items()})
    mes = rets.index.to_period("M")
    valor = 1.0
    out = []
    for _, bloque in rets.groupby(mes):
        crec = (1 + bloque).cumprod()
        v = sum(pesos[k] * crec[k] for k in pesos) * valor
        out.append(v)
        valor = float(v.iloc[-1])
    return pd.concat(out)


def main() -> None:
    d = r3.cargar()
    e = ev.cargar()
    curvas = {
        "D1-def50": curva(f3.candidatas_d1(d), f3.ESTRATEGIAS["D1-def50 · ídem con tope 50 % por posición"]),
        "C1": curva(f3.candidatas_c1(d, e), f3.ESTRATEGIAS["C1 · ronda 2 (referencia)"]),
    }
    corr = pd.DataFrame({k: v.pct_change() for k, v in curvas.items()}).corr().iloc[0, 1]
    # correlación solo de la parte activa (exceso sobre SPY con reserva en tendencia)
    filas = []
    for wd in (1.0, 0.75, 0.5, 0.25, 0.0):
        eq = mezclar(curvas, {"D1-def50": wd, "C1": 1 - wd})
        fila = {"peso D1-def50": wd, "peso C1": 1 - wd}
        for nombre, (a, b) in {"validación 2002-sep21": ("2002-01-02", "2021-10-01"),
                               "búsqueda oct21-oct26": ("2021-10-01", None),
                               "2002-2025": ("2002-01-02", "2026-01-01"), "total": ("2002-01-02", None)}.items():
            s = eq[(eq.index >= a) & ((eq.index < b) if b else True)]
            mt = m.metricas(s)
            fila |= {f"CAGR {nombre}": mt["cagr"], f"maxDD {nombre}": mt["maxdd"]}
            if nombre == "total":
                fila["Sharpe total"] = mt["sharpe"]
        filas.append(fila)
    df = pd.DataFrame(filas)
    df.to_csv(OUT / "combinacion.csv", index=False, float_format="%.4f")
    (OUT / "combinacion.json").write_text(json.dumps({"correlacion_diaria": corr}, indent=1))
    pd.set_option("display.width", 250)
    print(f"correlación diaria D1-def50 vs C1: {corr:.2f}")
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
