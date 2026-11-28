"""Anatomía del día de tendencia y comparación con un modelo de aprendizaje automático.

1) Frecuencia de los días de tendencia (cuerpo ≥ 60 % del rango) en las 80 acciones.
2) Huella temprana: a las 10:30 y 11:30, cómo son los rasgos en los días que acaban en tendencia
   frente al resto.
3) Predicción limpia: ¿el resto de la sesión (desde la hora de control hasta el cierre) es una
   tendencia en la dirección del día? Se define así: el cierre queda en el 30 % extremo del
   rango posterior a la hora de control, del lado de la dirección. Precisión del semáforo.
4) ¿Un modelo de árboles con todos los rasgos supera al semáforo? Entrenado con
   ene-2024 → jun-2025 y medido en jul-2025 → nov-2026.

Uso: python research/tendencia_intradia/anatomia.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent / "bracket_intradia"))
import evaluacion as E  # noqa: E402
import paneles as T  # noqa: E402
import prueba_semaforo as PS  # noqa: E402
import rasgos as R  # noqa: E402

OUT = AQUI / "resultados"


def resto_tendencia(P: dict, j: int) -> tuple[np.ndarray, np.ndarray]:
    """Dirección y bandera de "el resto del día es tendencia" desde la vela j."""
    H, L, C = P["H"][:, j + 1:], P["L"][:, j + 1:], P["C"]
    pt = C[:, j]
    hi = np.maximum(H.max(1), pt)
    lo = np.minimum(L.min(1), pt)
    pos = (C[:, -1] - lo) / np.maximum(hi - lo, 1e-12)
    return pos, (hi - lo) / P["atr"]


def anatomia() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = E.conjunto("acc_h1")
    base = df[df.cp == 60]
    frec = pd.Series({"dias_tendencia_alcista": (base.dia == 1).mean(),
                      "dias_tendencia_bajista": (base.dia == -1).mean(),
                      "dias_sin_tendencia": (base.dia == 0).mean()})
    filas = []
    for cp in (60, 120):
        g = df[df.cp == cp].copy()
        g["final"] = g.dia
        tend = g.dia != 0
        acuerdo = np.sign(g.ret_ap) == np.sign(g.dia)
        filas.append({"cp": cp, "grupo": "día de tendencia", "n": int(tend.sum()),
                      "va_a_favor_%": (acuerdo[tend]).mean(),
                      "|sigma|_mediana": g.ruido[tend].abs().median(),
                      "rvol_mediana": g.rvol[tend].median(),
                      "lado_vwap_ok_%": (np.sign(g.vwap) == np.sign(g.dia))[tend].mean(),
                      "rango_ATR_mediana": g.rango[tend].median()})
        filas.append({"cp": cp, "grupo": "resto", "n": int((~tend).sum()),
                      "va_a_favor_%": np.nan,
                      "|sigma|_mediana": g.ruido[~tend].abs().median(),
                      "rvol_mediana": g.rvol[~tend].median(),
                      "lado_vwap_ok_%": np.nan,
                      "rango_ATR_mediana": g.rango[~tend].median()})
    return frec.to_frame("valor"), pd.DataFrame(filas)


def prediccion_limpia() -> pd.DataFrame:
    """Precisión del semáforo para "el resto del día sigue en tendencia"."""
    filas = []
    for tk, P, Est in PS.fuentes("acc_h1"):
        for cp in (60, 120, 180):
            j = cp // 60 - 1
            pos, rng = resto_tendencia(P, j)
            d = np.sign(P["C"][:, j] - P["O"][:, 0])
            sigue = np.where(d > 0, pos >= 0.7, pos <= 0.3)
            filas.append(pd.DataFrame({"cp": cp, "estado": np.abs(Est[:, j]), "sigue": sigue,
                                       "rango_resto": rng, "valor": tk}))
    df = pd.concat(filas)
    out = df.groupby(["cp", "estado"]).agg(n=("sigue", "size"), p_sigue=("sigue", "mean"),
                                           rango_resto=("rango_resto", "median")).reset_index()
    out["estado"] = out.estado.map({0: "rojo", 1: "ámbar", 2: "verde", 3: "verde fuerte"})
    return out


def modelo() -> pd.DataFrame:
    df = E.conjunto("acc_h1")
    feats = [f for f in R.DIRECCIONALES + R.SIN_DIRECCION if df[f].notna().mean() > 0.5]
    filas = []
    for cp in (60, 120):
        g = df[df.cp == cp].dropna(subset=["y_cierre"]).copy()
        g["y"] = g.y_cierre.clip(-3, 3)
        feats_cp = [f for f in feats if g[f].nunique() > 5]
        tr = g[g.fecha < "2025-07-01"]
        te = g[g.fecha >= "2025-07-01"].copy()
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03, max_leaf_nodes=15,
                                          min_samples_leaf=400, l2_regularization=1.0)
        m.fit(tr[feats_cp], tr.y)
        te["pred"] = m.predict(te[feats_cp])
        ic = te.pred.rank().corr(te.y_cierre.rank())
        for q in (0.9, 0.95, 0.98):
            corte = te.pred.abs().quantile(q)
            sel = te[te.pred.abs() >= corte]
            x = np.sign(sel.pred) * sel.y_cierre
            pdia = x.groupby(sel.fecha).mean()
            filas.append({"cp": cp, "regla": f"modelo, {100 * (1 - q):.0f} % más extremo",
                          "n": len(sel), "ventaja_ATR": x.mean(), "acierto": (x > 0).mean(),
                          "t": pdia.mean() / pdia.std() * np.sqrt(len(pdia)), "ic_total": ic})
        # el semáforo en el mismo periodo de prueba
        for nombre, msk in (("semáforo verde (σ≥1, rvol≥1,5, VWAP)",
                             (te.ruido.abs() >= 1) & (te.rvol >= 1.5) & (np.sign(te.vwap) == np.sign(te.ret_ap))),
                            ("rvol ≥ 2 y σ ≥ 1", (te.ruido.abs() >= 1) & (te.rvol >= 2))):
            sel = te[msk & (te.ret_ap != 0)]
            x = np.sign(sel.ret_ap) * sel.y_cierre
            pdia = x.groupby(sel.fecha).mean()
            filas.append({"cp": cp, "regla": nombre, "n": len(sel), "ventaja_ATR": x.mean(),
                          "acierto": (x > 0).mean(),
                          "t": pdia.mean() / pdia.std() * np.sqrt(len(pdia)), "ic_total": np.nan})
    return pd.DataFrame(filas)


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    frec, huella = anatomia()
    print(frec.round(3)); print(huella.round(3).to_string())
    pl = prediccion_limpia(); print(pl.round(3).to_string())
    mo = modelo(); print(mo.round(3).to_string())
    frec.to_csv(OUT / "frecuencia_dias.csv")
    huella.to_csv(OUT / "huella.csv", index=False, float_format="%.4f")
    pl.to_csv(OUT / "resto_del_dia.csv", index=False, float_format="%.4f")
    mo.to_csv(OUT / "modelo_vs_semaforo.csv", index=False, float_format="%.4f")
