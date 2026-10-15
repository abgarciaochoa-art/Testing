"""Ronda 7 · anatomía de las operaciones y rasgos que separan ganadoras de perdedoras.

Universo: las 4.703 candidatas de la estrategia final (D1 con entradas técnicas: caída
>= 15 % desde el máximo de 20 d, en tendencia, >= 5 módulos y >= 2 banderas técnicas).
Cada candidata se evalúa como operación AISLADA (sin la cartera), con las reglas reales:
compra a la apertura de t+1, venta a la apertura siguiente al primer cierre > SMA10 o a
las 20 sesiones, 10 pb por lado. Así hay muchas más operaciones que las 687 que la
cartera llegó a hacer, y el resultado de una no depende de las demás.

Calcula:
1. Recorrido: retorno al cierre de cada sesión 1..20 desde la compra (aunque ya se haya
   vendido), máximo a favor (MFE) y en contra (MAE) con máximos y mínimos del día,
   y en qué sesión se tocó el peor punto.
2. Rasgos (los ~90 de la tabla de eventos, los técnicos y algunos de contexto nuevos,
   todos conocidos al cierre de la señal; el hueco de apertura se conoce al comprar):
   correlación de rangos (IC) con el retorno, AUC de ganar, y retorno por quintiles con
   cortes de la búsqueda. El t usa errores agrupados por mes (las caídas llegan en
   racimos). Regla de selección fijada antes de mirar la validación: |t| >= 2,5 en la
   búsqueda y el mismo signo en sus dos mitades.

Salida: resultados/anatomia/{camino.csv, discriminadores.csv, quintiles.csv, resumen.json}
y la tabla de eventos con rasgos y recorrido en data/cache/caidas/anatomia_eventos.parquet.

Uso: python research/caidas/anatomia.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numba
import numpy as np
import pandas as pd
from scipy.stats import rankdata

import eventos as ev
import motor as m
from definitiva import D1_CFG
from ronda5 import componentes_tec
from ronda5b import preparar

OUT = Path(__file__).parent / "resultados" / "anatomia"
EVENTOS = m.ROOT / "data" / "cache" / "caidas" / "anatomia_eventos.parquet"  # tabla completa (fuera de git)
CORTE = pd.Timestamp("2021-10-01")
MITAD = pd.Timestamp("2024-04-01")
H = 20
COSTE = 10e-4
VENTANAS = {"búsqueda": (CORTE, None), "búsqueda 1ª mitad": (CORTE, MITAD), "búsqueda 2ª mitad": (MITAD, None),
            "validación": (None, CORTE)}
NO_RASGO = {"t", "j", "fecha", "ticker", "prio", "peso", "base_score", "sobre_sma200"}


@numba.njit(cache=True)
def _recorrido(ts, js, O, Hi, Lo, C, sma10, H):
    n = len(ts)
    T = C.shape[0]
    camino = np.full((n, H), np.nan)
    ret = np.full(n, np.nan)
    ses = np.full(n, -1)
    mfe = np.full(n, np.nan)
    mae = np.full(n, np.nan)
    dia_mae = np.full(n, -1)
    for e in range(n):
        t, j = ts[e], js[e]
        if t + 1 >= T or np.isnan(O[t + 1, j]):
            continue
        pe = O[t + 1, j]
        for k in range(1, H + 1):
            if t + k < T and not np.isnan(C[t + k, j]):
                camino[e, k - 1] = C[t + k, j] / pe - 1
        alto, bajo, d_bajo = 0.0, 0.0, 0
        ult = pe
        for s in range(t + 1, min(t + 1 + H, T)):
            if np.isnan(C[s, j]):  # deslistada: se vende al último cierre
                ret[e] = ult / pe - 1
                ses[e] = s - t - 1
                break
            ult = C[s, j]
            h = Hi[s, j] if not np.isnan(Hi[s, j]) else C[s, j]
            lo = Lo[s, j] if not np.isnan(Lo[s, j]) else C[s, j]
            alto = max(alto, h / pe - 1)
            if lo / pe - 1 < bajo:
                bajo = lo / pe - 1
                d_bajo = s - t
            if (s - t >= H) or (C[s, j] > sma10[s, j]):
                px = O[s + 1, j] if s + 1 < T and not np.isnan(O[s + 1, j]) else C[s, j]
                ret[e] = px / pe - 1
                ses[e] = s + 1 - (t + 1)
                break
        if np.isnan(ret[e]):
            ret[e] = ult / pe - 1
            ses[e] = min(H, T - 1 - t)
        mfe[e] = alto
        mae[e] = bajo
        dia_mae[e] = d_bajo
    return camino, ret, ses, mfe, mae, dia_mae


def candidatas() -> pd.DataFrame:
    """Candidatas D1-técnica con todos los rasgos y algunos de contexto nuevos."""
    p = m.cargar()
    x, _, conv = componentes_tec()["D1"]
    y, w = preparar(x, conv, D1_CFG["kmin"], D1_CFG["uso"], D1_CFG["tam"], D1_CFG["frac"])
    y = y.assign(peso_motor=w)
    e = ev.cargar()
    extra = [c for c in e.columns if c not in y.columns and not re.fullmatch(r"[fx]\d+", c)
             and c not in ("op_sma5", "op_dias", "fecha", "ticker")]
    y = y.merge(e[["t", "j"] + extra], on=["t", "j"], how="left")
    t, j = y.t.to_numpy(), y.j.to_numpy()
    T = len(p.fechas)
    sig = np.minimum(t + 1, T - 1)
    y["fecha"] = pd.DatetimeIndex(p.fechas[t])
    y["ticker"] = p.tickers[j]
    # contexto nuevo (todo conocido al cierre de t; el hueco, en la apertura de la compra)
    y["n_cand_dia"] = y.groupby("t").t.transform("size")
    y["n_d1_dia"] = y.t.map(x.groupby("t").size())
    y["hueco_entrada"] = np.where(t + 1 < T, p.open[sig, j] / p.close[t, j] - 1, np.nan)
    y["precio"] = p.precio_real[t, j]
    dv = pd.DataFrame(p.raw_close * p.volume).rolling(60, min_periods=20).mean().to_numpy()
    y["log_dv60"] = np.log(np.where(dv[t, j] > 0, dv[t, j], np.nan))
    y["dia_semana"] = y.fecha.dt.dayofweek
    y["mes"] = y.fecha.dt.month
    y = y.sort_values(["j", "t"])
    y["sesiones_desde_anterior"] = y.groupby("j").t.diff()
    return y.sort_values(["t", "j"]).reset_index(drop=True)


def resultados(y: pd.DataFrame) -> pd.DataFrame:
    p = m.cargar()
    sma10 = pd.DataFrame(p.close).rolling(10).mean().to_numpy()
    camino, ret, ses, mfe, mae, dia_mae = _recorrido(y.t.to_numpy(np.int64), y.j.to_numpy(np.int64), p.open,
                                                     p.high, p.low, p.close, sma10, H)
    y = y.copy()
    y["ret"] = (1 + ret) * (1 - COSTE) ** 2 - 1
    y["gana"] = (y.ret > 0).astype(float).where(y.ret.notna())
    y["sesiones"] = ses
    y["mfe"], y["mae"], y["dia_mae"] = mfe, mae, dia_mae
    for k in range(H):
        y[f"c{k + 1}"] = camino[:, k]
    # la operación anterior en la misma acción ya cerrada al cierre de t (sin mirar al futuro)
    y = y.sort_values(["j", "t"])
    fin_prev = (y.t + y.sesiones).groupby(y.j).shift(1)
    gano_prev = y.groupby("j").gana.shift(1)
    y["anterior_gano"] = np.where(fin_prev < y.t, gano_prev, np.nan)
    return y.sort_values(["t", "j"]).reset_index(drop=True)


def en_ventana(y: pd.DataFrame, a, b) -> pd.Series:
    ok = pd.Series(True, index=y.index)
    if a is not None:
        ok &= y.fecha >= a
    if b is not None:
        ok &= y.fecha < b
    return ok & y.ret.notna()


def ic_agrupado(xv: np.ndarray, yv: np.ndarray, grupos: np.ndarray) -> tuple[float, float, int]:
    """Correlación de rangos y su t con errores agrupados (pendiente de rangos estandarizados)."""
    ok = ~np.isnan(xv) & ~np.isnan(yv)
    n = int(ok.sum())
    if n < 30 or np.nanstd(xv[ok]) == 0:
        return np.nan, np.nan, n
    zx = rankdata(xv[ok])
    zy = rankdata(yv[ok])
    zx = (zx - zx.mean()) / zx.std()
    zy = (zy - zy.mean()) / zy.std()
    ic = float(np.mean(zx * zy))
    res = zy - ic * zx
    s = pd.Series(zx * res).groupby(grupos[ok]).sum().to_numpy()
    se = np.sqrt(np.sum(s ** 2)) / np.sum(zx ** 2)
    return ic, ic / se if se > 0 else np.nan, n


def auc(xv: np.ndarray, g: np.ndarray) -> float:
    ok = ~np.isnan(xv) & ~np.isnan(g)
    x, g = xv[ok], g[ok] > 0.5
    n1, n0 = g.sum(), (~g).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(x)
    return float((r[g].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def discriminadores(y: pd.DataFrame, rasgos: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    mes = y.fecha.dt.to_period("M").astype(str).to_numpy()
    filas, quint = [], []
    busq = en_ventana(y, CORTE, None)
    for r in rasgos:
        xv = y[r].to_numpy(np.float64)
        fila = {"rasgo": r, "cobertura": float(np.mean(~np.isnan(xv)))}
        for nombre, (a, b) in VENTANAS.items():
            ok = en_ventana(y, a, b).to_numpy()
            ic, tt, n = ic_agrupado(xv[ok], y.ret.to_numpy()[ok], mes[ok])
            fila |= {f"IC {nombre}": ic, f"t {nombre}": tt, f"n {nombre}": n,
                     f"AUC {nombre}": auc(xv[ok], y.gana.to_numpy()[ok])}
        # quintiles con cortes de la búsqueda
        xb = xv[busq.to_numpy() & ~np.isnan(xv)]
        if len(np.unique(xb)) >= 5:
            cortes = np.unique(np.quantile(xb, [0.2, 0.4, 0.6, 0.8]))
            q = np.where(np.isnan(xv), -1, np.searchsorted(cortes, xv, side="right"))
            for nombre in ("búsqueda", "validación"):
                a, b = VENTANAS[nombre]
                ok = en_ventana(y, a, b).to_numpy() & (q >= 0)
                g = pd.DataFrame({"q": q[ok], "ret": y.ret.to_numpy()[ok], "gana": y.gana.to_numpy()[ok]})
                for k, s in g.groupby("q"):
                    quint.append({"rasgo": r, "ventana": nombre, "quintil": int(k) + 1, "n": len(s),
                                  "ret_medio": s.ret.mean(), "acierto": s.gana.mean(),
                                  "desde": float(cortes[k - 1]) if k > 0 else np.nan,
                                  "hasta": float(cortes[k]) if k < len(cortes) else np.nan})
        filas.append(fila)
    df = pd.DataFrame(filas)
    sig = np.sign(df["IC búsqueda"])
    df["elegido_en_búsqueda"] = ((df["t búsqueda"].abs() >= 2.5) & (np.sign(df["IC búsqueda 1ª mitad"]) == sig)
                                 & (np.sign(df["IC búsqueda 2ª mitad"]) == sig))
    df["se_mantiene_en_validación"] = df.elegido_en_búsqueda & (np.sign(df["IC validación"]) == sig) & (
        df["t validación"].abs() >= 2.0)
    return df, pd.DataFrame(quint)


def camino(y: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for nombre, (a, b) in {"todas": (None, None), "búsqueda": (CORTE, None), "validación": (None, CORTE)}.items():
        ok = en_ventana(y, a, b)
        for grupo, sel in (("todas", ok), ("ganadoras", ok & (y.gana == 1)), ("perdedoras", ok & (y.gana == 0))):
            z = y[sel]
            for k in range(1, H + 1):
                c = z[f"c{k}"].dropna()
                filas.append({"ventana": nombre, "grupo": grupo, "sesion": k, "n": len(c), "media": c.mean(),
                              "mediana": c.median(), "p25": c.quantile(0.25), "p75": c.quantile(0.75),
                              "p10": c.quantile(0.10), "p90": c.quantile(0.90), "en_ganancia": (c > 0).mean()})
    return pd.DataFrame(filas)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    y = resultados(candidatas())
    rasgos = [c for c in y.columns if c not in NO_RASGO and not re.fullmatch(r"c\d+", c)
              and c not in ("ret", "gana", "sesiones", "mfe", "mae", "dia_mae", "peso_motor")
              and pd.api.types.is_numeric_dtype(y[c])]
    disc, quint = discriminadores(y, rasgos)
    disc.to_csv(OUT / "discriminadores.csv", index=False, float_format="%.5f")
    quint.to_csv(OUT / "quintiles.csv", index=False, float_format="%.5f")
    cam = camino(y)
    cam.to_csv(OUT / "camino.csv", index=False, float_format="%.5f")
    y.to_parquet(EVENTOS)

    resumen = {}
    for nombre, (a, b) in {"todas": (None, None), "búsqueda": (CORTE, None), "validación": (None, CORTE)}.items():
        z = y[en_ventana(y, a, b)]
        g, pz = z[z.gana == 1], z[z.gana == 0]
        resumen[nombre] = {
            "n": len(z), "acierto": z.gana.mean(), "ret_medio": z.ret.mean(), "ret_mediana": z.ret.median(),
            "sesiones_media": z.sesiones.mean(), "mfe_medio": z.mfe.mean(), "mae_medio": z.mae.mean(),
            "mae_mediano": z.mae.median(), "ganadoras_mae_mediano": g.mae.median(),
            "ganadoras_que_pasaron_por_-5%": (g.mae <= -0.05).mean(),
            "perdedoras_mfe_mediano": pz.mfe.median(), "perdedoras_que_llegaron_a_+3%": (pz.mfe >= 0.03).mean(),
            "dia_mae_mediano": z.dia_mae.median(), "peor_mae": z.mae.min(),
        }
    (OUT / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=1, default=float))

    pd.set_option("display.width", 250)
    print(json.dumps(resumen, ensure_ascii=False, indent=1, default=float))
    cols = ["rasgo", "IC búsqueda", "t búsqueda", "IC búsqueda 1ª mitad", "IC búsqueda 2ª mitad", "IC validación",
            "t validación", "AUC validación", "elegido_en_búsqueda", "se_mantiene_en_validación"]
    d = disc.sort_values("t búsqueda", key=np.abs, ascending=False)
    print(d[cols].head(40).round(3).to_string(index=False))
    print("elegidos en búsqueda:", int(disc.elegido_en_búsqueda.sum()), "· se mantienen en validación:",
          int(disc.se_mantiene_en_validación.sum()), "de", len(disc))
    v = disc.sort_values("t validación", key=np.abs, ascending=False)
    print(v[cols].head(25).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
