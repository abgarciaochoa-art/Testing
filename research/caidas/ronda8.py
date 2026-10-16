"""Ronda 8 · en la cartera del nivel 2: añadir a las posiciones que siguen cayendo y,
como contraste, salir de las que van perdiendo.

Las reglas de añadir son las que ``gestion.py`` eligió SOLO con la búsqueda (FDR 10 % y
mismo signo en sus dos mitades), más su unión y el modelo ridge del recorrido. Cada
regla se prueba con dos tamaños: +50 % del tamaño previsto (con la posición limitada al
60 % del capital) y +100 % (límite 75 %). Lo añadido sale de la reserva, sin deuda nueva
en las compras. Las de salir son el espejo (vender en la apertura siguiente).

Significancia de cada cambio: diferencia diaria de rentabilidad frente a la estrategia
actual (mismas candidatas, mismo día), media anualizada y t de Newey-West (10 retardos)
en la búsqueda, sus mitades y la validación; para las elegidas, intervalo del 90 % de
la diferencia de CAGR en la validación con bootstrap estacionario por bloques pareado.

Regla fijada antes de mirar la validación: una variante es ELEGIBLE si supera a la
actual en CAGR en la búsqueda y en sus dos mitades; en cada familia se queda la de más
CAGR en la búsqueda. Se añaden, solo como contexto, el test pareado de los tamaños de la
ronda 7 y el vecindario de la salida por el modelo (montado después de ver la validación).

Uso: python research/caidas/ronda8.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

import gestion as ge
import motor as m
import motor_v4 as v4
from definitiva import D1_CFG
from estudio import nw_t
from ronda5 import componentes_tec
from ronda5b import preparar

OUT = Path(__file__).parent / "resultados"
CORTE = pd.Timestamp("2021-10-01")
MITAD = pd.Timestamp("2024-04-01")
VENTANAS = {"búsqueda": (CORTE, None), "búsqueda 1ª mitad": (CORTE, MITAD), "búsqueda 2ª mitad": (MITAD, None),
            "validación": (pd.Timestamp("2002-01-02"), CORTE), "total": (pd.Timestamp("2002-01-02"), None)}
BASE = dict(K=5, frac=0.5, H=20, reserva_="liq100_vg_minvol", palanca=1.0, palanca_off=1.0)
TAMAÑOS = (("+50 % (tope 60 %)", 0.5, 0.60), ("+100 % (tope 75 %)", 1.0, 0.75))


def reglas(g: pd.DataFrame) -> tuple[list, list]:
    """(nombre, día, máscara sobre la tabla de gestión) para añadir y para salir."""
    g = g.copy()
    g["pred"], _ = ge.ridge(g)
    b2 = (g.fecha >= CORTE) & (g.k == 2)
    q20, q80 = g.loc[b2, "pred"].quantile([0.2, 0.8])
    d2 = [("pierde ≥5 % al cierre del día 2", g.r <= -0.05),
          ("pierde ≥1 ATR al cierre del día 2", g.r <= -g.atr),
          ("cierra bajo el mínimo de la señal con volumen ≥1,5× (día 2)", (g.bajo_min_senal > 0.5) & (g.vol_rel >= 1.5)),
          ("día de caída ≥3 % con volumen ≥1,5× (día 2)", (g.ret_dia <= -0.03) & (g.vol_rel >= 1.5)),
          ("en pérdidas y cierra en lo alto del día (día 2)", (g.r < 0) & (g.ibs >= 0.7))]
    union = np.zeros(len(g), dtype=bool)
    for _, msk in d2:
        union |= msk.fillna(False).to_numpy()
    añadir = [(n, 2, msk) for n, msk in d2]
    añadir += [("pierde ≥5 % al cierre del día 5", 5, g.r <= -0.05),
               ("pierde ≥8 % al cierre del día 5", 5, g.r <= -0.08),
               ("pierde ≥2 ATR al cierre del día 5", 5, g.r <= -2 * g.atr),
               ("cualquiera de las cinco del día 2", 2, pd.Series(union, index=g.index)),
               ("modelo: quintil alto del exceso previsto (día 2)", 2, g.pred >= q80)]
    salir = [("salir si sigue en pérdidas al cierre del día 5", 5, g.r < 0),
             ("salir si pierde ≥5 % al cierre del día 2", 2, g.r <= -0.05),
             ("salir si el modelo prevé el quintil bajo (día 2)", 2, g.pred <= q20)]
    return [(n, k, msk.fillna(False).to_numpy() & (g.k == k).to_numpy()) for n, k, msk in añadir], \
        [(n, k, msk.fillna(False).to_numpy() & (g.k == k).to_numpy()) for n, k, msk in salir]


def vecindario(g: pd.DataFrame) -> list:
    """Sensibilidad de la salida por el modelo (montada DESPUÉS de ver la validación: solo
    describe si el resultado es una meseta o un pico; no cuenta como prueba)."""
    g = g.copy()
    g["pred"], _ = ge.ridge(g)
    out = []
    for k, q in ((2, 0.1), (2, 0.3), (2, 0.4), (1, 0.2), (3, 0.2)):
        corte = g.loc[(g.fecha >= CORTE) & (g.k == k), "pred"].quantile(q)
        out.append((f"salir si el modelo prevé el {q:.0%} más bajo (día {k})", k,
                    ((g.pred <= corte) & (g.k == k)).to_numpy()))
    alto = g.loc[(g.fecha >= CORTE) & (g.k == 2), "pred"].quantile(0.8)
    out.append(("combinada: salir del quintil bajo y añadir +100 % al alto (día 2)", 2,
                ((g.pred >= alto) & (g.k == 2)).to_numpy()))
    return out


def calendario(y: pd.DataFrame, g: pd.DataFrame, k: int, msk: np.ndarray) -> np.ndarray:
    """Apertura (absoluta) en la que actuar para cada candidata, o -1."""
    sel = g.loc[msk, ["t", "j"]].drop_duplicates()
    dia = pd.Series(sel.t.to_numpy() + 1 + k, index=pd.MultiIndex.from_frame(sel))
    idx = pd.MultiIndex.from_frame(y[["t", "j"]])
    return dia.reindex(idx).fillna(-1).astype(np.int64).to_numpy()


def ventanas(eq: pd.Series) -> dict:
    out = {}
    for nombre, (a, b) in VENTANAS.items():
        s = eq[(eq.index >= a) & ((eq.index < b) if b is not None else True)]
        mt = m.metricas(s / s.iloc[0])
        out |= {f"CAGR {nombre}": mt["cagr"], f"maxDD {nombre}": mt["maxdd"]}
    out["Sharpe"] = m.metricas(eq)["sharpe"]
    return out


def pareado(eq: pd.Series, eq_base: pd.Series) -> dict:
    d = (eq.pct_change() - eq_base.pct_change()).fillna(0)
    out = {}
    for nombre in ("búsqueda", "búsqueda 1ª mitad", "búsqueda 2ª mitad", "validación", "total"):
        a, b = VENTANAS[nombre]
        x = d[(d.index >= a) & ((d.index < b) if b is not None else True)].to_numpy()
        out |= {f"dif. anual {nombre}": x.mean() * 252, f"t NW {nombre}": nw_t(x, 10)}
    return out


def bootstrap(eq: pd.Series, eq_base: pd.Series, a, b, n: int = 2000, bloque: int = 20, seed: int = 11):
    """IC 90 % de la diferencia de CAGR remuestreando los MISMOS días en las dos curvas."""
    ra = eq.pct_change().fillna(0)
    rb = eq_base.pct_change().fillna(0)
    sel = (ra.index >= a) & ((ra.index < b) if b is not None else True)
    xa, xb = ra[sel].to_numpy(), rb[sel].to_numpy()
    T = len(xa)
    rng = np.random.default_rng(seed)
    dif = np.empty(n)
    for k in range(n):
        idx = []
        while len(idx) < T:
            s = rng.integers(T)
            L = rng.geometric(1 / bloque)
            idx.extend(range(s, min(s + L, T)))
        idx = np.array(idx[:T])
        dif[k] = np.prod(1 + xa[idx]) ** (252 / T) - np.prod(1 + xb[idx]) ** (252 / T)
    return float(np.quantile(dif, 0.05)), float(np.quantile(dif, 0.95)), float((dif <= 0).mean())


def main() -> None:
    t0 = time.time()
    x, salida, conv = componentes_tec()["D1"]
    y, w = preparar(x, conv, D1_CFG["kmin"], D1_CFG["uso"], D1_CFG["tam"], D1_CFG["frac"])
    y = y.reset_index(drop=True)
    g = pd.read_parquet(ge.TABLA)
    añadir, salir = reglas(g)
    args = (y.t.to_numpy(), y.j.to_numpy(), y.prio.to_numpy(np.float64))

    def correr(**kw):
        kw = {"peso": w} | kw
        r = v4.simular(*args, salida=salida, detalle=True, **(BASE | kw))
        ops = r["operaciones"]
        c = ops[ops.t_sal >= 0]
        return r["equity"], {"operaciones/año": r["trades_año"], "acierto": (c.ret > 0).mean(), "ret. medio": c.ret.mean(),
                             "añadidas": int((ops.tramos == 2).sum()), "en caídas (media)": r["exposicion"]}

    eq_base, st_base = correr()
    filas = [{"familia": "base", "regla": "actual (de golpe, sin añadir, SMA10)", "tamaño": "", **ventanas(eq_base),
              **st_base}]
    curvas = {"base": eq_base}
    casos = [("añadir " + tn, n, dict(add_t=calendario(y, g, k, msk), add_f=f, tope_pos=tope))
             for n, k, msk in añadir for tn, f, tope in TAMAÑOS]
    casos += [("salir", n, dict(sal_t=calendario(y, g, k, msk))) for n, k, msk in salir]
    casos.append(("referencia ronda 7", "dos tramos: mitad y la 2ª si cae ≥3 %", dict(w_ini=0.5, add_modo=1,
                                                                                      add_umbral=0.03)))
    import ronda7 as r7
    c7 = r7.candidatas()
    for nombre, col in (("tamaño al entrar por beta (terciles)", "beta"), ("tamaño al entrar por el modelo (terciles)",
                                                                           "meta_pct"),
                        ("tamaño al entrar por puntuación de rasgos (terciles)", "compuesta")):
        pw = pd.Series(r7.por_terciles(c7, col), index=pd.MultiIndex.from_frame(c7[["t", "j"]]))
        casos.append(("referencia ronda 7", nombre, dict(peso=pw.reindex(pd.MultiIndex.from_frame(y[["t", "j"]])).to_numpy())))
    casos.append(("referencia ronda 7", "tamaño fijo: 50 % por operación", dict(peso=np.full(len(y), 1.0))))
    vec = vecindario(g)
    for nombre, k, msk in vec[:-1]:
        casos.append(("vecindario (descriptivo)", nombre, dict(sal_t=calendario(y, g, k, msk))))
    nombre, k, msk = vec[-1]
    q20 = [x for x in salir if "modelo" in x[0]][0]
    casos.append(("vecindario (descriptivo)", nombre, dict(sal_t=calendario(y, g, q20[1], q20[2]),
                                                           add_t=calendario(y, g, k, msk), add_f=1.0, tope_pos=0.75)))
    for familia, nombre, kw in casos:
        eq, st = correr(**kw)
        fila = {"familia": familia, "regla": nombre, "tamaño": familia.replace("añadir ", "") if familia.startswith(
            "añadir") else "", **ventanas(eq), **st, **pareado(eq, eq_base)}
        filas.append(fila)
        curvas[f"{familia} | {nombre}"] = eq
        print(f"{time.time() - t0:5.0f}s {familia:22s} {nombre:62s} búsq {fila['CAGR búsqueda']:.3f} "
              f"valid {fila['CAGR validación']:.3f} t_valid {fila['t NW validación']:+.2f} DD {fila['maxDD total']:.3f}",
              flush=True)
    df = pd.DataFrame(filas)
    b = df.iloc[0]
    df["elegible"] = ((df["CAGR búsqueda"] > b["CAGR búsqueda"]) & (df["CAGR búsqueda 1ª mitad"] > b[
        "CAGR búsqueda 1ª mitad"]) & (df["CAGR búsqueda 2ª mitad"] > b["CAGR búsqueda 2ª mitad"]))
    df.loc[0, "elegible"] = False
    df.loc[df.familia.isin(["referencia ronda 7", "vecindario (descriptivo)"]), "elegible"] = False
    df["elegida"] = False
    for _, gr in df[df.elegible].groupby("familia"):
        df.loc[gr["CAGR búsqueda"].idxmax(), "elegida"] = True
    ic = []
    for i, fila in df.iterrows():
        if fila.elegida or (fila.elegible and fila["t NW búsqueda"] >= 2) or fila.familia == "vecindario (descriptivo)":
            lo, hi, p0 = bootstrap(curvas[f"{fila.familia} | {fila.regla}"], eq_base, *VENTANAS["validación"])
            ic.append((i, lo, hi, p0))
    for i, lo, hi, p0 in ic:
        df.loc[i, ["IC90 dif. CAGR valid. (inf)", "IC90 dif. CAGR valid. (sup)", "P(dif ≤ 0) valid."]] = [lo, hi, p0]
    df.to_csv(OUT / "ronda8.csv", index=False, float_format="%.5f")
    pd.set_option("display.width", 280)
    pd.set_option("display.max_colwidth", 60)
    cols = ["familia", "regla", "CAGR búsqueda", "CAGR búsqueda 1ª mitad", "CAGR búsqueda 2ª mitad", "CAGR validación",
            "CAGR total", "maxDD total", "Sharpe", "añadidas", "dif. anual búsqueda", "t NW búsqueda",
            "dif. anual validación", "t NW validación", "elegible", "elegida"]
    print(df[cols].round(4).to_string(index=False))
    extra = [c for c in df.columns if c.startswith(("IC90", "P(dif"))]
    if extra:
        print(df.loc[df[extra[0]].notna(), ["familia", "regla"] + extra].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
