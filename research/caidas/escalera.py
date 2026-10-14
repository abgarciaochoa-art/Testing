"""Escalera de riesgo: cartera combinada (D1 + C1 al 50 %, rebalanceo mensual) con
distintos niveles de apalancamiento puntual, y comparación con SPY apalancado.

Niveles (piezas definidas de antemano, no elegidas por la validación):
- base: componentes sin deuda, reserva SPY solo en tendencia (efectivo fuera);
- IEF-1.0 / IEF-1.5 / IEF-2.0: reserva SPY en tendencia e IEF fuera; palanca máxima
  1,0/1,5/2,0 solo con SPY > SMA200 (1,0 fuera); capacidad K=5 con huecos de 1/3;
- SHY-x: ídem con SHY (Tesoro 1-3 años, casi efectivo) fuera de tendencia, y con la
  deuda al 7 % como sensibilidad;
- regla A / regla B: lo que eligen las reglas de ``ronda4.py`` para cada componente.
Referencia: SPY con apalancamiento constante 1,5x y 2x (deuda al 5 %).

Uso: python research/caidas/escalera.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import motor as m
import motor_v3 as v3
from combinar import mezclar
from ronda4 import componentes

OUT = Path(__file__).parent / "resultados"
VENTANAS = {"búsqueda": ("2021-10-01", None), "validación": ("2002-01-02", "2021-10-01"),
            "2002-2025": ("2002-01-02", "2026-01-01"), "total": ("2002-01-02", None)}

NIVELES = {
    "base (sin deuda, ronda 3)": {"D1": dict(reserva_="spy_tend", palanca=1.0, palanca_off=1.0, K=3, frac=1 / 3),
                                  "C1": dict(reserva_="spy_tend", palanca=1.0, palanca_off=1.0, K=5, frac=1 / 5)},
    "IEF-1.0": {c: dict(reserva_="spy_tend_ief", palanca=1.0, palanca_off=1.0, K=5, frac=1 / 3) for c in ("D1", "C1")},
    "IEF-1.5": {c: dict(reserva_="spy_tend_ief", palanca=1.5, palanca_off=1.0, K=5, frac=1 / 3) for c in ("D1", "C1")},
    "IEF-2.0": {c: dict(reserva_="spy_tend_ief", palanca=2.0, palanca_off=1.0, K=5, frac=1 / 3) for c in ("D1", "C1")},
    "SHY-1.0": {c: dict(reserva_="spy_tend_shy", palanca=1.0, palanca_off=1.0, K=5, frac=1 / 3) for c in ("D1", "C1")},
    "SHY-1.5": {c: dict(reserva_="spy_tend_shy", palanca=1.5, palanca_off=1.0, K=5, frac=1 / 3) for c in ("D1", "C1")},
    "SHY-2.0": {c: dict(reserva_="spy_tend_shy", palanca=2.0, palanca_off=1.0, K=5, frac=1 / 3) for c in ("D1", "C1")},
    "SHY-1.5 deuda 7 %": {c: dict(reserva_="spy_tend_shy", palanca=1.5, palanca_off=1.0, K=5, frac=1 / 3,
                                 tasa_deuda=0.07) for c in ("D1", "C1")},
    "SHY-2.0 deuda 7 %": {c: dict(reserva_="spy_tend_shy", palanca=2.0, palanca_off=1.0, K=5, frac=1 / 3,
                                 tasa_deuda=0.07) for c in ("D1", "C1")},
    "regla A": {"D1": dict(reserva_="spy", palanca=1.25, palanca_off=1.0, K=3, frac=1 / 3),
                "C1": dict(reserva_="spy_tend_shy", palanca=1.5, palanca_off=1.5, K=8, frac=1 / 4)},
    "regla B": {"D1": dict(reserva_="spy", palanca=2.0, palanca_off=2.0, K=5, frac=1 / 3),
                "C1": dict(reserva_="spy_tend_shy", palanca=2.0, palanca_off=2.0, K=5, frac=1 / 3)},
}


def curva_componente(x, salida, conv, cfg) -> pd.Series:
    w = x.peso.to_numpy() if conv else None
    if w is not None:
        w = np.minimum(w, 0.5 / cfg["frac"])
    r = v3.simular(x.t.to_numpy(), x.j.to_numpy(), x.prio.to_numpy(np.float64), peso=w, H=20, salida=salida, **cfg)
    return r["equity"]


def spy_apalancado(L: float, tasa: float = 0.05) -> pd.Series:
    s = m.benchmark("SPY")
    r = s.pct_change().fillna(0)
    diario = L * r - (L - 1) * ((1 + tasa) ** (1 / 252) - 1)
    return (1 + diario).cumprod()


def metricas_ventanas(eq: pd.Series) -> dict:
    out = {}
    for nombre, (a, b) in VENTANAS.items():
        s = eq[(eq.index >= a) & ((eq.index < b) if b else True)]
        mt = m.metricas(s / s.iloc[0])
        out |= {f"CAGR {nombre}": mt["cagr"], f"maxDD {nombre}": mt["maxdd"]}
    mt = m.metricas(eq)
    out |= {"Sharpe": mt["sharpe"], "vol": mt["vol"]}
    return out


def main() -> None:
    comp = componentes()
    filas, curvas = [], {}
    for nivel, cfgs in NIVELES.items():
        cc = {c: curva_componente(*comp[c], cfgs[c]) for c in ("D1", "C1")}
        eq = mezclar(cc, {"D1": 0.5, "C1": 0.5})
        curvas[nivel] = eq / eq.iloc[0]
        filas.append({"nivel": nivel, **metricas_ventanas(eq)})
        print(nivel, "ok", flush=True)
    for L in (1.0, 1.5, 2.0):
        eq = spy_apalancado(L)
        curvas[f"SPY {L}x"] = eq
        filas.append({"nivel": f"SPY {L}x", **metricas_ventanas(eq)})
    df = pd.DataFrame(filas).set_index("nivel")
    df.to_csv(OUT / "escalera.csv", float_format="%.4f")
    anual = pd.DataFrame({k: v.resample("YE").last() for k, v in curvas.items()})
    anual = anual.pct_change().fillna(anual.iloc[0] - 1)
    anual.index = anual.index.year
    anual.to_csv(OUT / "escalera_anual.csv", float_format="%.4f")
    (OUT / "escalera.json").write_text(json.dumps({k: {c: {kk: (float(vv) if not isinstance(vv, str) else vv)
                                                            for kk, vv in cfg.items()} for c, cfg in v.items()}
                                                   for k, v in NIVELES.items()}, ensure_ascii=False, indent=1))
    pd.set_option("display.width", 250)
    print(df.round(3).to_string())
    print((anual[[c for c in anual.columns if not c.startswith("SPY 1.5") and not c.startswith("SPY 2")]] * 100)
          .round(1).to_string())


if __name__ == "__main__":
    main()
