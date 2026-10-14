"""Módulos v2: nuevas preguntas sobre la caída, con la misma metodología que
``modulos.py`` (operación real neta de 20 pb; búsqueda oct-2021 →, validación 2002 →
sep-2021; un grupo "mejora" si bate a la base).

- tamaño de la caída **en desviaciones típicas** de la propia acción (con la
  volatilidad previa a la caída): -10 % en una eléctrica no es lo mismo que en TSLA;
- distancia a la SMA200 (¿la caída llega al soporte de la tendencia?);
- tendencia de su grupo de pares (12 m) y liderazgo dentro del grupo;
- fracción de sus 10 pares que caen ≥5 %;
- tamaño/liquidez relativa dentro del índice;
- estado del mercado: SPY en corrección (caída desde su máximo anual) y pico de
  volatilidad (vol 20 d / vol 120 d);
- calendario: sesiones hasta fin de mes (flujos de cambio de mes);
- volatilidad propia previa.

Uso: python research/caidas/modulos_v2.py [base]
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import eventos as ev
from modulos import BASES, CORTE, resumen

OUT = Path(__file__).parent / "resultados"

MODULOS_V2 = {
    "tamaño en sigmas (r10 / vol previa)": {
        "≤ -4σ (extrema)": lambda d: d.sigma10 <= -4,
        "-4..-2.5σ": lambda d: (d.sigma10 > -4) & (d.sigma10 <= -2.5),
        "> -2.5σ (normal para ella)": lambda d: d.sigma10 > -2.5,
    },
    "distancia a SMA200": {
        "0-5 % sobre (toca soporte)": lambda d: (d.dist_sma200 >= 0) & (d.dist_sma200 < 0.05),
        "5-15 % sobre": lambda d: (d.dist_sma200 >= 0.05) & (d.dist_sma200 < 0.15),
        "≥15 % sobre (muy extendida)": lambda d: d.dist_sma200 >= 0.15,
        "bajo la SMA200": lambda d: d.dist_sma200 < 0,
    },
    "tendencia del grupo de pares (12 m)": {
        "pares +20 % o más": lambda d: d.pares_mom252 >= 0.2,
        "pares 0..20 %": lambda d: (d.pares_mom252 >= 0) & (d.pares_mom252 < 0.2),
        "pares en negativo": lambda d: d.pares_mom252 < 0,
    },
    "liderazgo dentro del grupo (mom - mom pares)": {
        "líder (+20 pp)": lambda d: d.rel_mom252 >= 0.2,
        "en línea": lambda d: (d.rel_mom252 > -0.1) & (d.rel_mom252 < 0.2),
        "rezagada (-10 pp)": lambda d: d.rel_mom252 <= -0.1,
    },
    "amplitud en su grupo (% pares caen ≥5 %)": {
        "≥60 % del grupo": lambda d: d.pares_frac_caen >= 0.6,
        "20-60 %": lambda d: (d.pares_frac_caen >= 0.2) & (d.pares_frac_caen < 0.6),
        "<20 % (sola en su grupo)": lambda d: d.pares_frac_caen < 0.2,
    },
    "tamaño/liquidez en el índice": {
        "grandes (tercio superior)": lambda d: d.tam >= 2 / 3,
        "medianas": lambda d: (d.tam >= 1 / 3) & (d.tam < 2 / 3),
        "pequeñas (tercio inferior)": lambda d: d.tam < 1 / 3,
    },
    "mercado: SPY desde su máximo anual": {
        "SPY en corrección ≥10 %": lambda d: d.spy_dd252 <= -0.10,
        "SPY -10..-3 %": lambda d: (d.spy_dd252 > -0.10) & (d.spy_dd252 <= -0.03),
        "SPY cerca de máximos": lambda d: d.spy_dd252 > -0.03,
    },
    "mercado: pico de volatilidad (20d/120d)": {
        "vol se dispara (≥1.5×)": lambda d: d.vol_pico_spy >= 1.5,
        "1-1.5×": lambda d: (d.vol_pico_spy >= 1) & (d.vol_pico_spy < 1.5),
        "vol cayendo (<1×)": lambda d: d.vol_pico_spy < 1,
    },
    "calendario: sesiones hasta fin de mes": {
        "últimas 3 sesiones": lambda d: d.sesiones_fin_mes <= 2,
        "resto": lambda d: d.sesiones_fin_mes > 2,
    },
    "volatilidad propia previa (anual)": {
        "≥45 %": lambda d: d.vol60 >= 0.45,
        "25-45 %": lambda d: (d.vol60 >= 0.25) & (d.vol60 < 0.45),
        "<25 % (tranquila)": lambda d: d.vol60 < 0.25,
    },
    "forma: peso del último día en la caída (r1/r10)": {
        "≥50 % en el último día (desplome)": lambda d: d.r1 / d.r10 >= 0.5,
        "20-50 %": lambda d: (d.r1 / d.r10 >= 0.2) & (d.r1 / d.r10 < 0.5),
        "<20 % (goteo)": lambda d: d.r1 / d.r10 < 0.2,
    },
}


def evaluar(d: pd.DataFrame) -> pd.DataFrame:
    ventanas = {"busq": d.fecha >= CORTE, "valid": d.fecha < CORTE}
    base = {w: resumen(d[s]) for w, s in ventanas.items()}
    filas = [{"modulo": "BASE", "grupo": "todos", **{f"{w}_{k}": v for w in ventanas for k, v in base[w].items()}}]
    for mod, grupos in MODULOS_V2.items():
        for g, f in grupos.items():
            sel = f(d).fillna(False).to_numpy(bool)
            fila = {"modulo": mod, "grupo": g}
            for w, s in ventanas.items():
                fila |= {f"{w}_{k}": v for k, v in resumen(d[sel & s.to_numpy()]).items()}
            filas.append(fila)
    out = pd.DataFrame(filas)
    for w in ventanas:
        out[f"{w}_mejora"] = out[f"{w}_op"] - base[w]["op"]
    return out


def main() -> None:
    d0 = ev.cargar()
    tablas = []
    for nombre, f in BASES.items():
        d = d0[f(d0).fillna(False)]
        r = evaluar(d)
        r["base"] = nombre
        r.to_csv(OUT / f"modulos_v2_{nombre}.csv", index=False, float_format="%.5f")
        tablas.append(r)
    t = pd.concat(tablas)
    t["b"] = (t.busq_mejora * 100).round(2)
    t["v"] = (t.valid_mejora * 100).round(2)
    pv = t.pivot_table(index=["modulo", "grupo"], columns="base", values=["b", "v"], sort=False)
    pv["ok_busq"] = (pv["b"] > 0).sum(axis=1)
    pv["ok_ambas"] = ((pv["b"] > 0) & (pv["v"] > 0)).sum(axis=1)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    print(pv.to_string())


if __name__ == "__main__":
    main()
