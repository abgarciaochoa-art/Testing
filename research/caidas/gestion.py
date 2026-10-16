"""Ronda 8 · gestión DENTRO de la operación: ¿salir antes de las perdedoras o añadir a
las buenas, según lo que ha pasado desde la compra?

Universo: las 4.701 candidatas del nivel 2 (``anatomia.py``) evaluadas como operación
aislada con las reglas reales. Para cada operación que sigue abierta al cierre del día
k desde la compra (k = 1, 2, 3, 5; la venta normal llega después de la apertura k+1):

- ``restante`` = lo que gana desde la apertura k+1 hasta la venta normal (SMA10 / 20 s.);
- ``reserva`` = lo que da la reserva contracíclica en ese mismo tramo (coste de
  oportunidad real: el dinero que se libera o que se añade sale de ella);
- ``exceso`` = restante - reserva. Salir antes conviene si el exceso medio es NEGATIVO
  en las operaciones que cumplen la condición; añadir, si es POSITIVO y supera los
  costes de ida y vuelta (~0,4 %: 10 pb compra + 10 pb venta + 2 movimientos de reserva).

Condiciones fijadas ANTES de mirar resultados (todas con datos hasta el cierre del día
k): pérdida acumulada (en % y en ATR), cierre bajo el mínimo del día de la señal (con y
sin volumen), día de caída con volumen, cae y el mercado no, cae y su gemela no,
rebote fallido, seguir en pérdidas, confirmación del rebote, cae con el mercado, cae
sin volumen, cierre en lo alto del día estando en pérdidas, y beta alta en pérdidas.

Significancia: t con errores agrupados por mes de la señal; corrección de
Benjamini-Hochberg (FDR 10 %) sobre todas las pruebas en la búsqueda. Una regla se
ELIGE si es significativa con FDR en la búsqueda y tiene el mismo signo en sus dos
mitades; la validación (2002 → sep 2021) dice si se sostiene. Además, un modelo ridge
con todo el recorrido (entrenado en la búsqueda) mide cuánta predictibilidad hay.

Salida: resultados/gestion/{condiciones.csv, modelo.json, deciles_modelo.csv} y la
tabla por operación y día en data/cache/caidas/gestion_eventos.parquet.

Uso: python research/caidas/gestion.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

import motor as m
import motor_v3 as v3
import rasgos as rg
from anatomia import CORTE, EVENTOS, MITAD

OUT = Path(__file__).parent / "resultados" / "gestion"
TABLA = m.ROOT / "data" / "cache" / "caidas" / "gestion_eventos.parquet"
DIAS = (1, 2, 3, 5)
COSTE = 10e-4
COSTE_RES = 10e-4
COSTE_AÑADIR = 2 * COSTE + 2 * COSTE_RES
VENTANAS = {"búsqueda": (CORTE, None), "búsqueda 1ª mitad": (CORTE, MITAD), "búsqueda 2ª mitad": (MITAD, None),
            "validación": (None, CORTE)}


def reserva_aperturas() -> np.ndarray:
    """Valor de la reserva contracíclica en la APERTURA de cada sesión (motor v3)."""
    rn, rd, cb, _, _ = v3.reserva("liq100_vg_minvol")
    T = len(rn)
    vo = np.empty(T)
    vc = 1.0
    for u in range(T):
        vo[u] = vc * rn[u] * (1 - cb[u])
        vc = vo[u] * rd[u]
    return vo


def tabla() -> pd.DataFrame:
    """Una fila por (operación, día k) con el recorrido hasta el cierre de k y el resultado."""
    p = m.cargar()
    T = len(p.fechas)
    ev = pd.read_parquet(EVENTOS)
    ev = ev[ev.ret.notna()].reset_index(drop=True)
    t, j = ev.t.to_numpy(), ev.j.to_numpy()
    O, C, Hi, Lo, CF, V = p.open, p.close, p.high, p.low, p.close_ff, p.volume
    pe = O[np.minimum(t + 1, T - 1), j]
    bruto = (1 + ev.ret.to_numpy()) / (1 - COSTE) ** 2 - 1
    px_sal = pe * (1 + bruto)
    ses = ev.sesiones.to_numpy()
    x_sal = t + 1 + ses  # apertura de la venta normal
    vo = reserva_aperturas()
    spy = p.col["SPY"]
    pares = np.load(rg.SALIDA)["pares"]
    gem = pares[t, j, 0]
    peers = pares[t, j, :]
    del pares
    vm = pd.DataFrame(V).rolling(20, min_periods=10).mean().shift(1).to_numpy()[t, j]
    filas = []
    bajo = np.zeros(len(ev))
    alto = np.zeros(len(ev))
    for k in range(1, max(DIAS) + 1):
        i = np.minimum(t + k, T - 1)
        c = C[i, j]
        lo = np.where(np.isnan(Lo[i, j]), c, Lo[i, j])
        hi = np.where(np.isnan(Hi[i, j]), c, Hi[i, j])
        bajo = np.minimum(bajo, lo / pe - 1)
        alto = np.maximum(alto, hi / pe - 1)
        if k not in DIAS:
            continue
        a = np.minimum(t + 1 + k, T - 1)  # apertura en la que se actuaría
        activo = (ses > k) & (x_sal < T) & (t + 1 + k < T) & ~np.isnan(O[a, j]) & ~np.isnan(c)
        prev = CF[i - 1, j]
        with np.errstate(invalid="ignore", divide="ignore"):
            g_ret = np.where(gem >= 0, CF[i, np.maximum(gem, 0)] / CF[t, np.maximum(gem, 0)] - 1, np.nan)
            vals = CF[i[:, None], np.maximum(peers, 0)] / CF[t[:, None], np.maximum(peers, 0)] - 1
            pares_ret = np.nanmean(np.where(peers >= 0, vals, np.nan), axis=1)
            d = pd.DataFrame({
                "e": np.arange(len(ev)), "t": t, "j": j, "fecha": ev.fecha, "k": k, "activo": activo,
                "r": c / pe - 1, "minimo": bajo.copy(), "maximo": alto.copy(), "ret_dia": c / prev - 1,
                "vol_rel": V[i, j] / vm, "ibs": (c - lo) / np.where(hi > lo, hi - lo, np.nan),
                "bajo_min_senal": (c < Lo[t, j]).astype(float), "spy": CF[i, spy] / O[t + 1, spy] - 1,
                "propio": c / C[t, j] - 1, "gemela": g_ret, "pares": pares_ret,
                "atr": ev.atr_pct.to_numpy(), "beta": ev.beta.to_numpy(),
                "restante": px_sal / O[a, j] - 1, "reserva": vo[np.minimum(x_sal, T - 1)] / vo[a] - 1,
                "ret_total": ev.ret.to_numpy(), "sesiones": ses})
        for col in ("caida_atr", "dist_ema5", "hueco_entrada", "vol_rel5", "r1", "r3", "meta_pct", "sigma10", "gem3"):
            d[col] = ev[col].to_numpy()
        filas.append(d[d.activo])
    df = pd.concat(filas, ignore_index=True)
    df["exceso"] = df.restante - df.reserva
    df["rel_gemela"] = df.propio - df.gemela
    df["rel_pares"] = df.propio - df.pares
    df["mes"] = df.fecha.dt.to_period("M").astype(str)
    return df


def condiciones(df: pd.DataFrame) -> list[tuple[str, str, pd.Series]]:
    """(familia, nombre, máscara). Salir si el exceso es negativo; añadir si es positivo."""
    q80_beta = df.loc[df.fecha >= CORTE, "beta"].quantile(0.8)
    C = []
    for x in (0.03, 0.05, 0.08, 0.12):
        C.append(("pérdida", f"va perdiendo ≥{x:.0%}", df.r <= -x))
    for mm in (1, 2, 3):
        C.append(("pérdida", f"va perdiendo ≥{mm} ATR", df.r <= -mm * df.atr))
    C += [("mínimos", "cierra bajo el mínimo del día de la señal", df.bajo_min_senal > 0.5),
          ("mínimos", "bajo el mínimo de la señal y con volumen ≥1,5×", (df.bajo_min_senal > 0.5) & (df.vol_rel >= 1.5)),
          ("volumen", "día de caída ≥3 % con volumen ≥1,5×", (df.ret_dia <= -0.03) & (df.vol_rel >= 1.5)),
          ("contexto", "cae ≥3 % y el mercado no", (df.r <= -0.03) & (df.spy >= 0)),
          ("contexto", "va ≥5 % peor que su gemela", (df.rel_gemela <= -0.05) & (df.r < 0)),
          ("contexto", "va ≥5 % peor que sus pares", (df.rel_pares <= -0.05) & (df.r < 0)),
          ("rebote", "rebote fallido (llegó a +3 % y está en pérdidas)", (df.maximo >= 0.03) & (df.r < 0)),
          ("rebote", "sigue en pérdidas", df.r < 0),
          ("rebote", "confirma: gana ≥2 %", df.r >= 0.02),
          ("rebote", "confirma: gana ≥4 %", df.r >= 0.04),
          ("contexto", "cae ≥3 % junto con el mercado (SPY ≤ -2 %)", (df.r <= -0.03) & (df.spy <= -0.02)),
          ("volumen", "cae ≥3 % sin volumen (≤1×)", (df.r <= -0.03) & (df.vol_rel <= 1.0)),
          ("rebote", "en pérdidas pero cierra en lo alto del día", (df.r < 0) & (df.ibs >= 0.7)),
          ("convicción", "beta alta (quintil alto) y en pérdidas", (df.beta >= q80_beta) & (df.r < 0))]
    return C


def t_agrupado(x: np.ndarray, grupos: np.ndarray) -> tuple[float, float]:
    """Media y su t con errores agrupados (cada grupo aporta la suma de sus desviaciones)."""
    n = len(x)
    if n < 20:
        return np.nan, np.nan
    mu = x.mean()
    s = pd.Series(x - mu).groupby(grupos).sum().to_numpy()
    se = np.sqrt(np.sum(s ** 2)) / n
    return float(mu), float(mu / se) if se > 0 else np.nan


def en_ventana(df: pd.DataFrame, a, b) -> pd.Series:
    ok = pd.Series(True, index=df.index)
    if a is not None:
        ok &= df.fecha >= a
    if b is not None:
        ok &= df.fecha < b
    return ok


def evaluar(df: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for familia, nombre, mask in condiciones(df):
        for k in DIAS:
            sel_k = (df.k == k) & mask.fillna(False)
            fila = {"familia": familia, "condición": nombre, "día": k}
            for vn, (a, b) in VENTANAS.items():
                base = (df.k == k) & en_ventana(df, a, b)
                z = df[sel_k & en_ventana(df, a, b)]
                mu, tt = t_agrupado(z.exceso.to_numpy(), z.mes.to_numpy())
                mu_a, tt_a = t_agrupado(z.exceso.to_numpy() - COSTE_AÑADIR, z.mes.to_numpy())
                fila |= {f"n {vn}": len(z), f"% de abiertas {vn}": len(z) / max(base.sum(), 1),
                         f"exceso {vn}": mu, f"t {vn}": tt, f"t añadir {vn}": tt_a,
                         f"restante {vn}": z.restante.mean() if len(z) else np.nan,
                         f"acaban ganando {vn}": (z.ret_total > 0).mean() if len(z) else np.nan}
            filas.append(fila)
    res = pd.DataFrame(filas)
    # FDR de Benjamini-Hochberg sobre las pruebas de la búsqueda (bilateral)
    p = 2 * norm.sf(np.abs(res["t búsqueda"].fillna(0).to_numpy()))
    orden = np.argsort(p)
    m_ = len(p)
    q = np.empty(m_)
    acum = 1.0
    for rango, idx in reversed(list(enumerate(orden, start=1))):
        acum = min(acum, p[idx] * m_ / rango)
        q[idx] = acum
    res["p búsqueda"] = p
    res["q FDR búsqueda"] = q
    signo = np.sign(res["exceso búsqueda"])
    mismo = (np.sign(res["exceso búsqueda 1ª mitad"]) == signo) & (np.sign(res["exceso búsqueda 2ª mitad"]) == signo)
    res["elegida"] = (res["q FDR búsqueda"] <= 0.10) & mismo & (res["n búsqueda"] >= 30)
    res["acción"] = np.where(signo < 0, "salir", np.where(res["t añadir búsqueda"] >= 2, "añadir", "nada"))
    res["se_mantiene"] = res.elegida & (np.sign(res["exceso validación"]) == signo) & (res["t validación"].abs() >= 2)
    return res


X_COLS = ["r", "minimo", "maximo", "ret_dia", "vol_rel", "ibs", "bajo_min_senal", "spy", "rel_gemela", "rel_pares",
          "atr", "beta", "caida_atr", "dist_ema5", "hueco_entrada", "vol_rel5", "r1", "r3", "meta_pct", "sigma10", "k"]


def ridge(df: pd.DataFrame, lam: float = 50.0) -> tuple[np.ndarray, np.ndarray]:
    """Exceso restante previsto por un ridge entrenado SOLO con la búsqueda (rasgos
    recortados al 1-99 % y estandarizados con la búsqueda). Devuelve (predicción, pesos)."""
    busq = (df.fecha >= CORTE).to_numpy()
    X = df[X_COLS].to_numpy(np.float64)
    med = np.nanmedian(X[busq], axis=0)
    X = np.where(np.isnan(X), med, X)
    lo, hi = np.nanpercentile(X[busq], 1, axis=0), np.nanpercentile(X[busq], 99, axis=0)
    X = np.clip(X, lo, hi)
    mu, sd = X[busq].mean(0), X[busq].std(0) + 1e-12
    X = (X - mu) / sd
    y = df.exceso.to_numpy()
    yb = np.clip(y[busq], *np.percentile(y[busq], [1, 99]))
    A = X[busq].T @ X[busq] + lam * np.eye(X.shape[1])
    w = np.linalg.solve(A, X[busq].T @ (yb - yb.mean()))
    return X @ w, w


def modelo(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Ridge sobre el recorrido y los rasgos de entrada; entrenado en la búsqueda."""
    X_cols = X_COLS
    z = df.copy()
    busq = (z.fecha >= CORTE).to_numpy()
    lam = 50.0
    pred, w = ridge(z, lam)
    y = z.exceso.to_numpy()
    z["pred"] = pred
    out = {"coeficientes": dict(zip(X_cols, map(float, w), strict=True)), "lambda": lam}
    for vn, (a, b) in VENTANAS.items():
        s = en_ventana(z, a, b).to_numpy()
        rx, ry = rankdata(pred[s]), rankdata(y[s])
        zx = (rx - rx.mean()) / rx.std()
        zy = (ry - ry.mean()) / ry.std()
        ic = float(np.mean(zx * zy))
        res = zy - ic * zx
        g = pd.Series(zx * res).groupby(z.mes.to_numpy()[s]).sum().to_numpy()
        se = np.sqrt(np.sum(g ** 2)) / np.sum(zx ** 2)
        out[f"IC {vn}"] = ic
        out[f"t {vn}"] = ic / se
    cortes = np.quantile(pred[busq], np.linspace(0.1, 0.9, 9))
    z["decil"] = np.searchsorted(cortes, pred) + 1
    dec = []
    for vn in ("búsqueda", "validación"):
        a, b = VENTANAS[vn]
        s = en_ventana(z, a, b)
        for d_, g in z[s].groupby("decil"):
            mu_, tt_ = t_agrupado(g.exceso.to_numpy(), g.mes.to_numpy())
            dec.append({"ventana": vn, "decil": int(d_), "n": len(g), "exceso": mu_, "t": tt_,
                        "acaban ganando": (g.ret_total > 0).mean()})
    return out, pd.DataFrame(dec)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = tabla()
    df.to_parquet(TABLA)
    res = evaluar(df)
    res.to_csv(OUT / "condiciones.csv", index=False, float_format="%.5f")
    mod, dec = modelo(df)
    (OUT / "modelo.json").write_text(json.dumps(mod, ensure_ascii=False, indent=1))
    dec.to_csv(OUT / "deciles_modelo.csv", index=False, float_format="%.5f")
    pd.set_option("display.width", 260)
    pd.set_option("display.max_colwidth", 50)
    print("operaciones abiertas por día:", df.groupby("k").size().to_dict())
    base = df.groupby(["k", df.fecha >= CORTE]).exceso.mean().unstack()
    print("exceso medio de TODAS las abiertas (validación / búsqueda):")
    print(base.round(4).to_string())
    cols = ["condición", "día", "n búsqueda", "exceso búsqueda", "t búsqueda", "q FDR búsqueda",
            "exceso búsqueda 1ª mitad", "exceso búsqueda 2ª mitad", "n validación", "exceso validación",
            "t validación", "acaban ganando validación", "acción", "elegida", "se_mantiene"]
    print(res.sort_values("t búsqueda")[cols].round(4).to_string(index=False))
    print(json.dumps({k: v for k, v in mod.items() if k != "coeficientes"}, ensure_ascii=False, indent=1))
    print(dec.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
