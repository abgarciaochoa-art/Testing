"""Noise Band · ronda 2: qué separa las rupturas que ganan de las que pierden.

Toma las operaciones de la estrategia del artículo (largo y corto, sin palanca, costes de
SPY a mercado) y les pone los rasgos de ``rasgos.py`` en el control en que se decidieron,
orientados al lado de la operación (× +1 largo, × -1 corto; ``fuerza`` es la de su lado).
Añade ``n_previas`` (operaciones ya hechas ese día) y ``lado``.

Las operaciones pierden poco muchas veces y ganan mucho pocas (acierto ~36 %), así que lo que
cuenta es el **retorno medio**, no el orden: para cada rasgo se mide la pendiente del retorno
neto (pb) sobre el percentil del rasgo (≈ diferencia de retorno medio entre el más alto y el
más bajo), con su t con errores agrupados por mes, en la búsqueda (2013 → may-2020) y sus dos
mitades (2013-2016, 2017-2020) y en la validación (2005-2012); también la correlación de
rangos (informativa). FDR del 10 % (Benjamini-Hochberg) sobre la pendiente en la búsqueda;
quintiles con los cortes de la búsqueda. Se elige un rasgo si en la búsqueda pasa el FDR y su
pendiente tiene el mismo signo en las dos mitades; la validación solo se mira.

Salida: resultados/{ronda2_eventos.csv, ronda2_discriminadores.csv, ronda2_quintiles.csv}

Uso: python research/noise_band/eventos.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

import motor as nb
import rasgos as rg

OUT = Path(__file__).parent / "resultados"
COSTES = dict(coste_por_accion=0.0085)
VENTANAS = {"búsqueda": ("2013-01-01", None), "búsqueda 2013-16": ("2013-01-01", "2017-01-01"),
            "búsqueda 2017-20": ("2017-01-01", None), "validación": ("2005-01-24", "2013-01-01")}
RASGOS = ["hora", "fuerza", "mov", "hueco", "hueco_abs", "sigma_rel", "vol5", "vol14", "vol60", "vol5_60",
          "dd252", "tend50", "tend200", "ret1", "ret1_dia", "vrel", "vrel30", "vwap", "rango", "volmin30",
          "dia_semana", "vencimiento", "fin_mes", "inicio_mes", "n_previas", "lado"]


def tabla(d: dict | None = None, R: dict | None = None, **kw) -> pd.DataFrame:
    d = nb.cargar() if d is None else d
    R = dict(np.load(rg.SALIDA)) if R is None else R
    r = nb.simular(d, lados=kw.pop("lados", 2), **(COSTES | kw))
    o = r["operaciones"].copy()
    o = o[o.entrada >= pd.Timestamp("2005-01-24")]
    dd, kd = o.d_ent.astype(int).to_numpy(), o.k_ent.astype(int).to_numpy() - 1
    c = (kd - 29) // 30
    ok = (kd >= 29) & ((kd - 29) % 30 == 0) & (c < len(rg.KS))
    o, dd, c = o[ok].copy(), dd[ok], c[ok]
    lado = o.lado.to_numpy()
    for nombre, v in R.items():
        if nombre in ("fuerza_l", "fuerza_c"):
            continue
        x = v[dd, c]
        o[nombre] = x * lado if nombre in rg.CON_SIGNO else x
    o["fuerza"] = np.where(lado > 0, R["fuerza_l"][dd, c], R["fuerza_c"][dd, c])
    o["hueco_abs"] = np.abs(R["hueco"][dd, c])
    o["n_previas"] = o.groupby("d_ent").cumcount()
    o["gana"] = (o.ret > 0).astype(float)
    o["fecha"] = o.entrada
    return o.reset_index(drop=True)


def ic_agrupado(xv: np.ndarray, yv: np.ndarray, grupos: np.ndarray) -> tuple[float, float, int]:
    """Correlación de rangos y su t con errores agrupados (como en el estudio de caídas)."""
    ok = ~np.isnan(xv) & ~np.isnan(yv)
    n = int(ok.sum())
    if n < 30 or np.nanstd(xv[ok]) == 0:
        return np.nan, np.nan, n
    zx, zy = rankdata(xv[ok]), rankdata(yv[ok])
    zx, zy = (zx - zx.mean()) / zx.std(), (zy - zy.mean()) / zy.std()
    ic = float(np.mean(zx * zy))
    s = pd.Series(zx * (zy - ic * zx)).groupby(grupos[ok]).sum().to_numpy()
    se = np.sqrt(np.sum(s ** 2)) / np.sum(zx ** 2)
    return ic, ic / se if se > 0 else np.nan, n


def pendiente(xv: np.ndarray, yv: np.ndarray, grupos: np.ndarray) -> tuple[float, float]:
    """Pendiente de y sobre el percentil de x (centrado) y su t con errores agrupados."""
    ok = ~np.isnan(xv) & ~np.isnan(yv)
    if ok.sum() < 30 or np.nanstd(xv[ok]) == 0:
        return np.nan, np.nan
    p = (rankdata(xv[ok]) - 0.5) / ok.sum() - 0.5
    y = yv[ok] - yv[ok].mean()
    b = float(np.sum(p * y) / np.sum(p ** 2))
    s = pd.Series(p * (y - b * p)).groupby(grupos[ok]).sum().to_numpy()
    se = np.sqrt(np.sum(s ** 2)) / np.sum(p ** 2)
    return b, b / se if se > 0 else np.nan


def en(y: pd.DataFrame, a, b) -> np.ndarray:
    m = (y.fecha >= pd.Timestamp(a)).to_numpy()
    return m & (y.fecha < pd.Timestamp(b)).to_numpy() if b is not None else m


def discriminadores(y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mes = y.fecha.dt.to_period("M").astype(str).to_numpy()
    filas, quint = [], []
    for r in RASGOS:
        xv = y[r].to_numpy(np.float64)
        fila = {"rasgo": r}
        for nombre, (a, b) in VENTANAS.items():
            ok = en(y, a, b)
            ic, t_ic, n = ic_agrupado(xv[ok], y.ret.to_numpy()[ok], mes[ok])
            b, t = pendiente(xv[ok], y.ret.to_numpy()[ok] * 1e4, mes[ok])
            fila |= {f"pendiente {nombre}": b, f"t {nombre}": t, f"IC {nombre}": ic, f"t IC {nombre}": t_ic,
                     f"n {nombre}": n}
        busq = en(y, *VENTANAS["búsqueda"]) & ~np.isnan(xv)
        cortes = np.unique(np.quantile(xv[busq], [0.2, 0.4, 0.6, 0.8]))
        q = np.where(np.isnan(xv), -1, np.searchsorted(cortes, xv, side="right"))
        for nombre in ("búsqueda", "validación"):
            ok = en(y, *VENTANAS[nombre]) & (q >= 0)
            g = pd.DataFrame({"q": q[ok], "ret": y.ret.to_numpy()[ok], "gana": y.gana.to_numpy()[ok]})
            for k, s in g.groupby("q"):
                quint.append({"rasgo": r, "ventana": nombre, "grupo": int(k) + 1, "n": len(s), "ret_medio": s.ret.mean(),
                              "acierto": s.gana.mean(), "desde": float(cortes[k - 1]) if k > 0 else np.nan,
                              "hasta": float(cortes[k]) if k < len(cortes) else np.nan})
        filas.append(fila)
    df = pd.DataFrame(filas)
    # FDR del 10 % en la búsqueda (Benjamini-Hochberg sobre p bilaterales)
    p = 2 * norm.sf(df["t búsqueda"].abs().fillna(0))
    orden = np.argsort(p)
    m = len(p)
    pasa = np.zeros(m, dtype=bool)
    k_max = max([i + 1 for i, j in enumerate(orden) if p[j] <= 0.10 * (i + 1) / m], default=0)
    pasa[orden[:k_max]] = True
    df["p búsqueda"] = p
    df["pasa_fdr"] = pasa
    sig = np.sign(df["pendiente búsqueda"])
    df["elegido"] = pasa & (np.sign(df["pendiente búsqueda 2013-16"]) == sig) & (
        np.sign(df["pendiente búsqueda 2017-20"]) == sig)
    df["se_mantiene"] = df.elegido & (np.sign(df["pendiente validación"]) == sig)
    return df, pd.DataFrame(quint)


def main() -> None:
    y = tabla()
    y.to_csv(OUT / "ronda2_eventos.csv", index=False, float_format="%.6f")
    df, quint = discriminadores(y)
    df.to_csv(OUT / "ronda2_discriminadores.csv", index=False, float_format="%.5f")
    quint.to_csv(OUT / "ronda2_quintiles.csv", index=False, float_format="%.6f")
    pd.set_option("display.width", 250)
    print(f"{len(y)} operaciones ({(y.fecha < '2013-01-01').sum()} en la validación); retorno neto medio "
          f"{y.ret.mean() * 1e4:.2f} pb, acierto {y.gana.mean():.1%}")
    cols = ["rasgo", "pendiente búsqueda", "t búsqueda", "pendiente búsqueda 2013-16", "pendiente búsqueda 2017-20",
            "pendiente validación", "t validación", "IC búsqueda", "pasa_fdr", "elegido", "se_mantiene"]
    print(df[cols].sort_values("t búsqueda", key=np.abs, ascending=False).round(3).to_string(index=False))
    for r in df[df.elegido].rasgo:
        q = quint[quint.rasgo == r].pivot_table(index="grupo", columns="ventana", values="ret_medio") * 1e4
        print(f"\n{r}: retorno neto medio por quintil (pb)\n{q.round(1).T.to_string()}")


if __name__ == "__main__":
    main()
