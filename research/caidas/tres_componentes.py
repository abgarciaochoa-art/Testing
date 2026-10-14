"""Cartera de tres componentes: D1 (caídas rápidas de acciones con módulos v2),
C1 (caídas de 5 días con módulos v1) y E1 (caídas de ETFs), cada uno con su capital,
reserva SPY en tendencia / SHY fuera, hasta 5 huecos de 1/3 (E1: 3 huecos) y la misma
palanca máxima solo en tendencia. Pesos fijados de antemano: 1/3 cada uno (frente a
D1+C1 al 50 %). Rebalanceo mensual.

Uso: python research/caidas/tres_componentes.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import componente_etf as ce
import escalera as es
from combinar import mezclar
from ronda4 import componentes

OUT = Path(__file__).parent / "resultados"
E1 = dict(n=5, x=0.05, tendencia=False, salida="rsi2", K=3)  # elegido en la búsqueda (componente_etf.py)


def curvas(pal: float, coste_bps: float = 10.0) -> dict[str, pd.Series]:
    comp = componentes()
    cfg = dict(reserva_="spy_tend_shy", palanca=pal, palanca_off=1.0, K=5, frac=1 / 3, coste_bps=coste_bps)
    out = {c: es.curva_componente(*comp[c], cfg) for c in ("D1", "C1")}
    out["E1"] = ce.simular(ce.candidatas(E1["n"], E1["x"], E1["tendencia"]), K=E1["K"], salida=E1["salida"],
                           palanca=pal, palanca_off=1.0, coste_bps=coste_bps)["equity"]
    return out


def main() -> None:
    filas = []
    corr = None
    for pal in (1.0, 1.5, 2.0):
        cc = curvas(pal)
        if corr is None:
            rets = pd.DataFrame({k: v.pct_change() for k, v in cc.items()}).dropna()
            corr = rets.corr().round(2).to_dict()
        for nombre, pesos in (("D1+C1 (50/50)", {"D1": 0.5, "C1": 0.5}),
                              ("D1+C1+E1 (1/3 c.u.)", {"D1": 1 / 3, "C1": 1 / 3, "E1": 1 / 3})):
            eq = mezclar(cc, pesos)
            filas.append({"cartera": nombre, "palanca": pal, **es.metricas_ventanas(eq)})
        filas.append({"cartera": "E1 sola", "palanca": pal, **es.metricas_ventanas(cc["E1"])})
    df = pd.DataFrame(filas)
    df.to_csv(OUT / "tres_componentes.csv", index=False, float_format="%.4f")
    (OUT / "tres_componentes.json").write_text(json.dumps({"correlacion_diaria": corr, "E1": E1}, indent=1))
    pd.set_option("display.width", 250)
    print("correlación diaria (palanca 1):", corr)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
