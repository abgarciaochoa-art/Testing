"""Noise Band · ronda 2: filtros, tamaño, salidas, palanca y momentum de la última media hora.

Sobre la estrategia del artículo (S&P 500, largo y corto, costes de SPY a mercado), variantes
fijadas antes de ver sus resultados (columna ``origen``):

- **del estudio de rasgos** (``eventos.py``): el único rasgo elegido es la ``fuerza`` de la
  ruptura. Filtro: solo entrar si la fuerza supera el percentil 40, 50 o 60 de las
  operaciones de la búsqueda, y tamaño 0,5× / 1,5× por debajo / encima de la mediana;
- **de la literatura**: doble tamaño con el mercado ≥ 5 % bajo su máximo anual (Baltussen et
  al., 2021: la cobertura de opciones refuerza el momentum en las caídas); tamaño según la
  volatilidad frente a su mediana de un año; momentum de la última media hora (Gao et al.,
  2018: a las 15:30, a favor de la rentabilidad del día; se cierra al cierre), sola y sumada;
- **a favor del hueco** de la noche (idea conocida, *gap and go*): no pasó el estudio de
  rasgos en la búsqueda y ya se había visto su fuerza en la validación, así que para ella la
  validación no es limpia;
- **mirado en las dos ventanas** (no cuenta como prueba): evitar el 20 % de rupturas con más
  volumen o con más volatilidad de los minutos en la última media hora (efecto en U de los
  quintiles de ``eventos.py``);
- **salidas:** solo la banda o solo la VWAP; **palanca:** volatilidad objetivo del 1,5 / 2 /
  2,5 % y máximo de 2 / 3 / 4×; y **solo largo** de lo anterior que se pueda.

Una variante es elegible si mejora a la base (mismo tamaño) en la búsqueda (2013 → may-2020) y
en sus dos mitades: por CAGR (lo fijado al principio) y, como la CAGR premia a quien solo
añade exposición, también por Sharpe. La validación (2005-2012) solo se mira: diferencia diaria
con la base (t de Newey-West) y diferencia de Sharpe (t de Ledoit y Wolf).

Salida: resultados/{ronda2.csv, ronda2_anual.csv} y las series de cada prueba para el Sharpe
deflactado (data/cache/noise_band/pruebas_ronda2.npz).

Uso: python research/noise_band/ronda2.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import estadistica as es
import motor as nb
import rasgos as rg

OUT = Path(__file__).parent / "resultados"
SERIES = nb.DATOS.with_name("pruebas_ronda2.npz")
COSTES = dict(coste_por_accion=0.0085)
VENTANAS = {"busqueda": ("2013-01-01", None), "busqueda_1": ("2013-01-01", "2017-01-01"),
            "busqueda_2": ("2017-01-01", None), "validacion": ("2005-01-24", "2013-01-01"),
            "total": ("2005-01-24", None)}
RASGOS_ = "estudio de rasgos (búsqueda)"
LITERATURA = "literatura"
VISTA = "vista en la validación"
MIRADO = "mirado en las dos ventanas (no cuenta)"


def multiplicadores(d: dict, f) -> tuple[np.ndarray, np.ndarray]:
    """``f(lado) -> días × 12`` (multiplicador en cada control) → matrices días × 390."""
    D = len(d["fechas"])
    ml, mc = np.ones((D, nb.VELAS)), np.ones((D, nb.VELAS))
    ml[:, rg.KS] = np.nan_to_num(f(+1), nan=0.0)
    mc[:, rg.KS] = np.nan_to_num(f(-1), nan=0.0)
    return ml, mc


def ultima_media_hora(d: dict) -> pd.Series:
    """A las 15:30, a favor de la rentabilidad desde el cierre anterior; se cierra al cierre."""
    C, O, ult = d["C"], d["O"], d["ult"]
    D = len(d["fechas"])
    previo = np.r_[np.nan, d["cierre"][:-1]]
    direc = np.sign(C[:, 359] / previo - 1)
    c = nb.costes_dia(d, **COSTES)
    r = np.where(ult >= 360, direc * (C[np.arange(D), ult] / O[:, 360] - 1) - 2 * c, 0.0)
    return pd.Series(np.nan_to_num(r), d["fechas"])


def main() -> None:
    d = nb.cargar()
    R = dict(np.load(rg.SALIDA))
    ev = pd.read_csv(OUT / "ronda2_eventos.csv", parse_dates=["fecha"])
    busq = ev[ev.fecha >= pd.Timestamp("2013-01-01")]
    cortes = {p: float(busq.fuerza.quantile(p / 100)) for p in (40, 50, 60)}
    p80 = {r: float(busq[r].quantile(0.8)) for r in ("vrel30", "volmin30")}
    fuerza = lambda s: R["fuerza_l"] if s > 0 else R["fuerza_c"]  # noqa: E731
    v14 = pd.Series(d["vol14"])
    vol_rel = np.repeat((v14 / v14.rolling(252, min_periods=120).median()).to_numpy()[:, None], len(rg.KS), 1)

    reglas = {
        "base": (None, "artículo"),
        **{f"fuerza ≥ p{p}": ((lambda c: lambda s: (fuerza(s) >= c).astype(float))(c), RASGOS_)
           for p, c in cortes.items()},
        "tamaño por fuerza (0,5× / 1,5×)": (lambda s: np.where(fuerza(s) >= cortes[50], 1.5, 0.5), RASGOS_),
        "doble con el mercado en caída (≥ 5 %)": (lambda s: np.where(R["dd252"] <= -0.05, 2.0, 1.0), LITERATURA),
        "tamaño por volatilidad relativa": (lambda s: np.clip(vol_rel, 0.5, 2.0), LITERATURA),
        "fuerza ≥ p50 y doble en caída": (
            lambda s: (fuerza(s) >= cortes[50]) * np.where(R["dd252"] <= -0.05, 2.0, 1.0), LITERATURA),
        "a favor del hueco": (lambda s: (R["hueco"] * s >= 0).astype(float), VISTA),
        "fuerza ≥ p50 y a favor del hueco": (
            lambda s: ((fuerza(s) >= cortes[50]) & (R["hueco"] * s >= 0)).astype(float), VISTA),
        "sin volumen extremo (vrel30 < p80)": (lambda s: (np.nan_to_num(R["vrel30"]) < p80["vrel30"]).astype(float),
                                               MIRADO),
        "sin minutos agitados (volmin30 < p80)": (
            lambda s: (np.nan_to_num(R["volmin30"]) < p80["volmin30"]).astype(float), MIRADO),
    }
    salidas = {"salida del artículo": dict(trail_vwap=1, salida_banda=1),
               "solo banda": dict(trail_vwap=0, salida_banda=1), "solo VWAP": dict(trail_vwap=2, salida_banda=0)}
    tamaños = {"1x": nb.palanca(d), "vol 2 % (máx. 4×)": nb.palanca(d, 0.02)}
    extra_palanca = {"vol 2 % (máx. 2×)": nb.palanca(d, 0.02, 2.0), "vol 2 % (máx. 3×)": nb.palanca(d, 0.02, 3.0),
                     "vol 1,5 % (máx. 4×)": nb.palanca(d, 0.015), "vol 2,5 % (máx. 4×)": nb.palanca(d, 0.025)}
    u30 = ultima_media_hora(d)

    pruebas: dict[tuple, pd.Series] = {}
    origen: dict[str, str] = {k: o for k, (_, o) in reglas.items()}
    for lados in (2, 1):
        for nombre, (f, _) in reglas.items():
            ml, mc = (None, None) if f is None else multiplicadores(d, f)
            for tam, lev in tamaños.items():
                r = nb.simular(d, lados=lados, lev=lev, mult_largo=ml, mult_corto=mc, **COSTES)
                pruebas[(lados, nombre, "salida del artículo", tam)] = r["equity"]
        for sal, kw in list(salidas.items())[1:]:
            for tam, lev in tamaños.items():
                pruebas[(lados, "base", sal, tam)] = nb.simular(d, lados=lados, lev=lev, **kw, **COSTES)["equity"]
    for tam, lev in extra_palanca.items():
        pruebas[(2, "base", "salida del artículo", tam)] = nb.simular(d, lados=2, lev=lev, **COSTES)["equity"]
    pruebas[(2, "última media hora sola", "al cierre", "1x")] = (1 + u30).cumprod()
    for tam in tamaños:
        base = pruebas[(2, "base", "salida del artículo", tam)].pct_change().fillna(0.0)
        pruebas[(2, "base + última media hora", "salida del artículo", tam)] = (1 + base + u30).cumprod()
    origen |= {"última media hora sola": LITERATURA, "base + última media hora": LITERATURA}

    filas = []
    for (lados, nombre, sal, tam), eq in pruebas.items():
        ref = pruebas[(lados, "base", "salida del artículo", tam if tam in tamaños else "vol 2 % (máx. 4×)")]
        r, rb = eq.pct_change().fillna(0.0), ref.pct_change().fillna(0.0)
        fila = {"lados": "largo y corto" if lados == 2 else "solo largo", "variante": nombre, "salida": sal,
                "tamaño": tam, "origen": "artículo" if nombre == "base" else origen[nombre]}
        mb = {}
        for vn, (a, b) in VENTANAS.items():
            m, mb[vn] = nb.metricas(es.tramo(eq, a, b)), nb.metricas(es.tramo(ref, a, b))
            fila |= {f"{vn}_cagr": m["cagr"], f"{vn}_peor_caida": m["peor_caida"], f"{vn}_sharpe": m["sharpe"]}
            if vn in ("busqueda", "validacion"):
                x, xb = es.tramo(r, a, b).to_numpy(), es.tramo(rb, a, b).to_numpy()
                igual = np.abs(x - xb).sum() == 0
                fila[f"{vn}_t_dif"] = np.nan if igual else es.nw_t(x - xb)
                fila[f"{vn}_t_sharpe"] = np.nan if igual else es.t_sharpe(x, xb)
        mitades = ("busqueda", "busqueda_1", "busqueda_2")
        fila["elegible_cagr"] = all(fila[f"{v}_cagr"] > mb[v]["cagr"] + 1e-9 for v in mitades)
        fila["elegible_sharpe"] = all(fila[f"{v}_sharpe"] > mb[v]["sharpe"] + 1e-9 for v in mitades)
        filas.append(fila)
    df = pd.DataFrame(filas)
    df.to_csv(OUT / "ronda2.csv", index=False, float_format="%.5f")
    rets = pd.DataFrame({" · ".join(map(str, k)): eq.pct_change().fillna(0.0) for k, eq in pruebas.items()})
    np.savez(SERIES, nombres=np.array(rets.columns), fechas=rets.index.to_numpy(), r=rets.to_numpy())
    anual = {}
    for (lados, nombre, sal, tam), eq in pruebas.items():
        a = eq.resample("YE").last()
        anual[f"{'L/C' if lados == 2 else 'L'} · {nombre} · {sal} · {tam}"] = a / np.r_[eq.iloc[0], a.to_numpy()[:-1]] - 1
    an = pd.DataFrame(anual)
    an.index = an.index.year
    an.to_csv(OUT / "ronda2_anual.csv", float_format="%.4f")

    pd.set_option("display.width", 280)
    pd.set_option("display.max_colwidth", 40)
    cols = ["lados", "variante", "salida", "tamaño", "total_cagr", "total_peor_caida", "total_sharpe",
            "busqueda_sharpe", "busqueda_1_sharpe", "busqueda_2_sharpe", "validacion_sharpe", "validacion_cagr",
            "busqueda_t_sharpe", "validacion_t_sharpe", "elegible_cagr", "elegible_sharpe"]
    print(f"cortes de fuerza (búsqueda): {cortes}; p80 de volumen: {p80}; {len(pruebas)} pruebas")
    print(df[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
