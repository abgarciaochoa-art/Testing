"""Ronda 7 · cómo comprar, cuánto y cuándo vender (cartera del nivel 2).

Sobre la estrategia final (D1 con entradas técnicas, reserva contracíclica, sin deuda
en las compras), cambia UNA cosa cada vez:

A. Salidas: solo por tiempo; otras medias; condiciones (RSI2, primer cierre al alza,
   cierre sobre el máximo de ayer); objetivos y stops en % (al cierre o con órdenes
   intradía); trailing; venta parcial; tope de sesiones.
B. Entrada: de golpe o en dos tramos (siempre, solo si sigue cayendo o solo si rebota).
C. Tamaño: fijo, por módulos (actual), por beta, por volatilidad, por la puntuación de
   rasgos de ``anatomia.py``, por el modelo y por el hueco de apertura.
D. Filtros y prioridad con la puntuación de rasgos.
E. Perfiles de riesgo: combinaciones montadas después de ver A-D (solo descriptivas).

La puntuación de rasgos usa SOLO los elegidos en la búsqueda por ``anatomia.py``
(|t| >= 2,5 y mismo signo en sus dos mitades), sin los del modelo (ya están en la
prioridad) ni el hueco de apertura (se conoce al comprar): media de percentiles con el
signo de la búsqueda. Los cortes de terciles y quintiles también salen de la búsqueda.

Regla fijada antes de mirar la validación: una variante es ELEGIBLE si supera a la
base en CAGR en la búsqueda y en sus dos mitades; dentro de cada familia se queda la de
más CAGR en la búsqueda. La validación 2002 → sep-2021 dice si se sostiene.

Uso: python research/caidas/ronda7.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import motor as m
import motor_rapido as mr
import motor_v4 as v4
from anatomia import EVENTOS
from definitiva import D1_CFG
from ronda5 import componentes_tec
from ronda5b import preparar

OUT = Path(__file__).parent / "resultados"
CORTE = pd.Timestamp("2021-10-01")
MITAD = pd.Timestamp("2024-04-01")
VENTANAS = {"búsqueda": (CORTE, None), "búsqueda 1ª mitad": (CORTE, MITAD), "búsqueda 2ª mitad": (MITAD, None),
            "validación": ("2002-01-02", CORTE), "2002-2025": ("2002-01-02", "2026-01-01"), "total": ("2002-01-02", None)}
MODELO = {"score_meta", "score_gbm", "meta_pct", "score_ridge"}
BASE = dict(K=5, frac=0.5, H=20, salida="sma10", reserva_="liq100_vg_minvol", palanca=1.0, palanca_off=1.0)


def candidatas() -> pd.DataFrame:
    """Candidatas del nivel 2 con sus rasgos y la puntuación de rasgos (solo búsqueda)."""
    x, _, conv = componentes_tec()["D1"]
    y, w = preparar(x, conv, D1_CFG["kmin"], D1_CFG["uso"], D1_CFG["tam"], D1_CFG["frac"])
    y = y[["t", "j", "prio", "score_v2", "meta_pct"]].assign(w=w)
    ev = pd.read_parquet(EVENTOS)
    disc = pd.read_csv(OUT / "anatomia" / "discriminadores.csv")
    elegidos = disc[disc.elegido_en_búsqueda & ~disc.rasgo.isin(MODELO)]
    cols = sorted(set(elegidos.rasgo) | {"beta", "atr_pct", "hueco_entrada", "fecha"})
    y = y.merge(ev[["t", "j"] + [c for c in cols if c not in ("meta_pct",)]], on=["t", "j"], how="left")
    busq = (y.fecha >= CORTE).to_numpy()
    partes = []
    for _, r in elegidos[elegidos.rasgo != "hueco_entrada"].iterrows():
        v = y[r.rasgo].to_numpy(np.float64)
        ref = np.sort(v[busq & ~np.isnan(v)])
        pct = np.searchsorted(ref, v, side="right") / len(ref)
        pct = np.where(np.isnan(v), np.nan, pct)
        partes.append(pct if r["IC búsqueda"] > 0 else 1 - pct)
    y["compuesta"] = np.nanmean(np.vstack(partes), axis=0)
    y.attrs["rasgos_compuesta"] = [
        (r.rasgo, "+" if r["IC búsqueda"] > 0 else "-") for _, r in elegidos.iterrows() if r.rasgo != "hueco_entrada"]
    return y


def cortes(y: pd.DataFrame, col: str, qs) -> np.ndarray:
    v = y.loc[y.fecha >= CORTE, col].dropna().to_numpy()
    return np.quantile(v, qs)


def por_terciles(y: pd.DataFrame, col: str, pesos=(0.5, 0.75, 1.0)) -> np.ndarray:
    c = cortes(y, col, [1 / 3, 2 / 3])
    v = y[col].to_numpy(np.float64)
    k = np.searchsorted(c, v, side="right")
    return np.where(np.isnan(v), pesos[1], np.asarray(pesos)[np.clip(k, 0, 2)])


def matrices_salida() -> dict[str, np.ndarray]:
    b = mr._base()
    p = b["p"]
    cf = p.close_ff
    prev = np.vstack([np.full((1, cf.shape[1]), np.nan), cf[:-1]])
    prev_hi = np.vstack([np.full((1, cf.shape[1]), np.nan), p.high[:-1]])
    ema5 = pd.DataFrame(p.close).ewm(span=5, adjust=False, min_periods=5).mean().to_numpy()
    with np.errstate(invalid="ignore"):
        extra = {"primer cierre al alza": np.nan_to_num(p.close > prev).astype(bool),
                 "cierre sobre el máximo de ayer": np.nan_to_num(p.close > prev_hi).astype(bool),
                 "cierre sobre EMA5": np.nan_to_num(p.close > ema5).astype(bool)}
    return {**b["salidas"], **extra}


def variantes(y: pd.DataFrame) -> list[tuple[str, str, dict]]:
    w = y.w.to_numpy()
    q80_hueco = cortes(y, "hueco_entrada", [0.8])[0]
    q20_comp = cortes(y, "compuesta", [0.2])[0]
    q20_beta = cortes(y, "beta", [0.2])[0]
    atr = y.atr_pct.to_numpy(np.float64)
    med_atr = cortes(y, "atr_pct", [0.5])[0]
    hueco = y.hueco_entrada.to_numpy(np.float64)
    V = [("base", "actual: SMA10, máx. 20 sesiones, de golpe, tamaño por módulos", {})]
    # A · salidas
    for h in (3, 5, 10, 20):
        V.append(("A · salida", f"solo tiempo: {h} sesiones", dict(salida="tiempo", H=h)))
    for s in ("sma5", "sma20"):
        V.append(("A · salida", f"cierre > {s.upper()}", dict(salida=s)))
    for h in (10, 30):
        V.append(("A · salida", f"cierre > SMA10, máx. {h} sesiones", dict(H=h)))
    V.append(("A · salida", "RSI2 ≥ 70", dict(salida="rsi2")))
    for s in ("primer cierre al alza", "cierre sobre el máximo de ayer", "cierre sobre EMA5"):
        V.append(("A · salida", s, dict(salida=s)))
    for x in (0.05, 0.08, 0.12):
        V.append(("A · salida", f"SMA10 u objetivo +{x:.0%} al cierre", dict(obj_c=x)))
        V.append(("A · salida", f"SMA10 u objetivo +{x:.0%} intradía (orden límite)", dict(obj_i=x)))
    for x in (0.08, 0.12, 0.20):
        V.append(("A · salida", f"SMA10 o stop -{x:.0%} al cierre", dict(stop_c=x)))
        V.append(("A · salida", f"SMA10 o stop -{x:.0%} intradía", dict(stop_i=x)))
    for x in (0.05, 0.10):
        V.append(("A · salida", f"SMA10 o trailing {x:.0%} desde el máximo", dict(trail_c=x)))
    for x in (0.05, 0.08):
        V.append(("A · salida", f"vende la mitad en +{x:.0%} intradía, el resto con SMA10", dict(obj_i=x, parcial=0.5)))
    # B · entrada en tramos
    for d in (0, 1):
        V.append(("B · entrada", f"mitad y mitad: la 2ª al día {d + 2} siempre", dict(w_ini=0.5, add_modo=0, add_dia=d)))
    for u in (0.0, 0.03, 0.06):
        V.append(("B · entrada", f"mitad y la 2ª solo si sigue cayendo ≥{u:.0%}", dict(w_ini=0.5, add_modo=1, add_umbral=u)))
    V.append(("B · entrada", "mitad y la 2ª si sigue cayendo ≥3 % (decide el día 2)",
              dict(w_ini=0.5, add_modo=1, add_umbral=0.03, add_dia=1)))
    V.append(("B · entrada", "2/3 y el resto si sigue cayendo ≥3 %", dict(w_ini=2 / 3, add_modo=1, add_umbral=0.03)))
    for u in (0.0, 0.02):
        V.append(("B · entrada", f"mitad y la 2ª solo si rebota ≥{u:.0%}", dict(w_ini=0.5, add_modo=2, add_umbral=u)))
    # C · tamaño
    V += [("C · tamaño", "fijo: 30 % del capital por operación", dict(peso=np.full(len(y), 0.6))),
          ("C · tamaño", "fijo: 50 % del capital por operación", dict(peso=np.full(len(y), 1.0))),
          ("C · tamaño", "por beta (terciles: 25/37,5/50 %)", dict(peso=por_terciles(y, "beta"))),
          ("C · tamaño", "por puntuación de rasgos (terciles)", dict(peso=por_terciles(y, "compuesta"))),
          ("C · tamaño", "por el modelo (terciles)", dict(peso=por_terciles(y, "meta_pct"))),
          ("C · tamaño", "módulos × puntuación de rasgos",
           dict(peso=np.minimum(1.0, w * por_terciles(y, "compuesta", (0.75, 1.0, 1.25))))),
          ("C · tamaño", "módulos × volatilidad (más a las más volátiles)",
           dict(peso=np.minimum(1.0, w * np.clip(np.nan_to_num(atr / med_atr, nan=1.0), 0.67, 1.5)))),
          ("C · tamaño", "módulos ÷ volatilidad (menos a las más volátiles)",
           dict(peso=np.minimum(1.0, w * np.clip(np.nan_to_num(med_atr / atr, nan=1.0), 0.67, 1.5)))),
          ("C · tamaño", "mitad si abre con subida ≥ quintil alto del hueco",
           dict(peso=np.where(hueco > q80_hueco, 0.5 * w, w)))]
    # D · filtros y prioridad
    prio2 = y.score_v2.to_numpy() + 0.5 * y.meta_pct.fillna(0.5).to_numpy() + 0.5 * np.nan_to_num(y.compuesta, nan=0.5)
    V += [("D · selección", "prioridad: módulos + modelo + puntuación de rasgos", dict(score=prio2)),
          ("D · selección", "sin el quintil bajo de la puntuación de rasgos",
           dict(filtro=~(y.compuesta.to_numpy() < q20_comp))),
          ("D · selección", "sin el quintil de beta más baja", dict(filtro=~(y.beta.to_numpy() < q20_beta))),
          ("D · selección", "no comprar si abre con subida ≥ quintil alto del hueco",
           dict(peso=np.where(hueco > q80_hueco, 0.0, w)))]
    # E · perfiles de riesgo: combinaciones montadas DESPUÉS de ver A-D, solo descriptivas
    # (no cuentan como prueba fuera de muestra ni entran en la elección)
    tramo = dict(w_ini=0.5, add_modo=1, add_umbral=0.03)
    V += [("E · perfiles (descriptivo)", "stop -20 % al cierre + mitad y la 2ª si cae ≥3 %", dict(stop_c=0.20, **tramo)),
          ("E · perfiles (descriptivo)", "2/3 y el resto si cae ≥3 % + stop -20 % al cierre",
           dict(stop_c=0.20, w_ini=2 / 3, add_modo=1, add_umbral=0.03)),
          ("E · perfiles (descriptivo)", "50 % fijo + stop -20 % al cierre", dict(peso=np.full(len(y), 1.0), stop_c=0.20)),
          ("E · perfiles (descriptivo)", "50 % fijo en dos tramos (2ª si cae ≥3 %)", dict(peso=np.full(len(y), 1.0), **tramo)),
          ("E · perfiles (descriptivo)", "50 % fijo en dos tramos + stop -20 % al cierre",
           dict(peso=np.full(len(y), 1.0), stop_c=0.20, **tramo))]
    return V


def correr(y: pd.DataFrame, kw: dict, sal: dict[str, np.ndarray]) -> dict:
    kw = dict(kw)
    filtro = kw.pop("filtro", np.ones(len(y), dtype=bool))
    score = kw.pop("score", y.prio.to_numpy(np.float64))
    peso = kw.pop("peso", y.w.to_numpy())
    cfg = BASE | kw
    if cfg["salida"] not in mr._base()["salidas"]:
        cfg["sal_matriz"] = sal[cfg["salida"]]
    r = v4.simular(y.t.to_numpy()[filtro], y.j.to_numpy()[filtro], np.asarray(score)[filtro],
                   peso=np.asarray(peso)[filtro], detalle=True, **cfg)
    eq = r["equity"]
    fila = {}
    for nombre, (a, b) in VENTANAS.items():
        s = eq[(eq.index >= pd.Timestamp(a)) & ((eq.index < pd.Timestamp(b)) if b else True)]
        mt = m.metricas(s / s.iloc[0])
        fila |= {f"CAGR {nombre}": mt["cagr"], f"maxDD {nombre}": mt["maxdd"]}
    ops = r["operaciones"]
    c = ops[ops.t_sal >= 0]
    fila |= {"Sharpe": m.metricas(eq)["sharpe"], "operaciones/año": r["trades_año"], "acierto": (c.ret > 0).mean(),
             "ret. medio": c.ret.mean(), "sesiones": (c.t_sal - c.t_ent).mean(), "en caídas (media)": r["exposicion"],
             "exposición total media": r["palanca_media"]}
    return fila | {"_eq": eq}


def main() -> None:
    t0 = time.time()
    y = candidatas()
    sal = matrices_salida()
    filas, curvas = [], []
    for familia, nombre, kw in variantes(y):
        fila = {"familia": familia, "variante": nombre, **correr(y, kw, sal)}
        curvas.append(fila.pop("_eq").pct_change().fillna(0).to_numpy(np.float32))
        filas.append(fila)
        print(f"{time.time() - t0:5.0f}s {familia:14s} {nombre:60s} búsq {fila['CAGR búsqueda']:.3f} "
              f"valid {fila['CAGR validación']:.3f} DD {fila['maxDD total']:.3f}", flush=True)
    df = pd.DataFrame(filas)
    base = df.iloc[0]
    df["elegible"] = ((df["CAGR búsqueda"] > base["CAGR búsqueda"]) & (df["CAGR búsqueda 1ª mitad"] > base[
        "CAGR búsqueda 1ª mitad"]) & (df["CAGR búsqueda 2ª mitad"] > base["CAGR búsqueda 2ª mitad"]))
    df.loc[0, "elegible"] = False
    df.loc[df.familia.str.startswith("E"), "elegible"] = False
    df["elegida"] = False
    for _, g in df[df.elegible].groupby("familia"):
        df.loc[g["CAGR búsqueda"].idxmax(), "elegida"] = True
    df["mejora valid. (pp)"] = (df["CAGR validación"] - base["CAGR validación"]) * 100
    df["mejora maxDD total (pp)"] = (df["maxDD total"] - base["maxDD total"]) * 100
    df.to_csv(OUT / "ronda7.csv", index=False, float_format="%.5f")
    np.save(m.ROOT / "data" / "cache" / "caidas" / "ronda7_retornos.npy", np.vstack(curvas))
    (OUT / "ronda7_rasgos.json").write_text(json.dumps(y.attrs["rasgos_compuesta"], ensure_ascii=False))
    pd.set_option("display.width", 260)
    pd.set_option("display.max_colwidth", 62)
    cols = ["familia", "variante", "CAGR búsqueda", "CAGR búsqueda 1ª mitad", "CAGR búsqueda 2ª mitad",
            "CAGR validación", "CAGR total", "maxDD total", "maxDD validación", "Sharpe", "operaciones/año",
            "acierto", "ret. medio", "sesiones", "en caídas (media)", "elegible", "elegida"]
    print(df[cols].round(3).to_string(index=False))
    print("rasgos de la puntuación:", y.attrs["rasgos_compuesta"])


if __name__ == "__main__":
    main()
