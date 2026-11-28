"""¿Qué rasgos identifican la tendencia intradía? Evaluación por hora de control.

Para cada conjunto de datos, hora de control y rasgo:
  ic        correlación de rangos (Spearman) entre el rasgo y el movimiento hasta el cierre
  ventaja   media de signo(rasgo) × movimiento hasta el cierre, en ATR: lo que ganaría quien
            entra a esa hora en la dirección del rasgo y cierra al final del día (bruto)
  acierto   % de días en que el signo del rasgo acierta la dirección hasta el cierre
  q5-q1     movimiento medio hasta el cierre del 20 % más alcista menos el del 20 % más bajista
  ic por año / mitad, para ver si es estable

Conjuntos:
  acc_h1    80 acciones, velas de 1 hora, ene-2024 → nov-2026 (mercado = SPY)
  acc_m5    80 acciones, velas de 5 minutos, sep → nov 2026
  etf_h1    SPY, QQQ, IWM, SMH, XLK (1 hora)
  NAS100    Nasdaq-100, 1 minuto, 2005 → 2020
  BTC       bitcóin, 1 minuto, 2022 → 2026
  NVDA_1m   Nvidia (Dukascopy), 1 minuto

Uso: python research/tendencia_intradia/evaluacion.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent / "bracket_intradia"))
import paneles as T  # noqa: E402
import rasgos as R  # noqa: E402

OUT = AQUI / "resultados"
OUT.mkdir(exist_ok=True)
ETFS = ["SPY", "QQQ", "IWM", "SMH", "XLK"]
CP = {"h1": [60, 120, 180, 240, 300, 360],
      "m5": [15, 30, 45, 60, 90, 120, 180, 240, 300, 360],
      "m1": [15, 30, 45, 60, 90, 120, 180, 240, 300, 360]}


def tickers() -> list[str]:
    return sorted(p.stem[3:] for p in T.PANELES.glob("h1_*.npz") if p.stem[3:] not in ETFS)


def _mercado(prefijo: str, cp: int) -> pd.Series:
    P = T.cargar(f"{prefijo}_SPY")
    return R.en_hora(P, cp)["ret_ap"]


def conjunto(nombre: str) -> pd.DataFrame:
    """Todas las filas (día × valor × hora de control) de un conjunto."""
    cache = T.CACHE / "filas" / f"{nombre}.parquet"  # fuera del repo: data/cache está ignorado
    cache.parent.mkdir(exist_ok=True)
    if cache.exists():
        return pd.read_parquet(cache)
    filas = []
    if nombre in ("acc_h1", "acc_m5", "etf_h1"):
        pre = "h1" if nombre.endswith("h1") else "m5"
        lista = ETFS if nombre == "etf_h1" else tickers()
        mk = {cp: _mercado(pre, cp) for cp in CP[pre]}
        for tk in lista:
            f = T.PANELES / f"{pre}_{tk}.npz"
            if not f.exists():
                continue
            P = T.cargar(f"{pre}_{tk}")
            Q = R.precalcular(P)
            for cp in CP[pre]:
                d = R.en_hora(P, cp, Q, mercado=None if tk == "SPY" else mk[cp])
                d["valor"] = tk
                filas.append(d)
    else:
        import estudio as B  # el de la horquilla (research/bracket_intradia): paneles de 1 minuto
        nom = {"NAS100": "NAS100_oanda", "BTC": "BTC", "NVDA_1m": "dukas_NVDA",
               "XAU": "XAU_oanda"}[nombre]
        P = B.preparar(nom)
        Q = R.precalcular(P)
        for cp in CP["m1"]:
            d = R.en_hora(P, cp, Q)
            d["valor"] = nombre
            filas.append(d)
    df = pd.concat(filas).rename_axis("fecha").reset_index()
    df.to_parquet(cache)
    return df


def _ic(a: pd.Series, b: pd.Series) -> float:
    m = a.notna() & b.notna()
    if m.sum() < 30:
        return np.nan
    return a[m].rank().corr(b[m].rank())


def evaluar(df: pd.DataFrame, nombre: str, y: str = "y_cierre") -> pd.DataFrame:
    filas = []
    df = df.copy()
    df["anio"] = df["fecha"].dt.year
    fechas = np.sort(df["fecha"].unique())
    corte = fechas[len(fechas) // 2]
    for cp, g in df.groupby("cp"):
        for r in R.DIRECCIONALES:
            x = g[r]
            if x.notna().sum() < 30:
                continue
            m = x.notna() & g[y].notna() & (x != 0)
            s, yy = np.sign(x[m]), g[y][m]
            q = x[m].rank(pct=True)
            fila = {"conjunto": nombre, "cp": cp, "rasgo": r, "n": int(m.sum()),
                    "ic": _ic(x, g[y]), "ventaja": float((s * yy).mean()),
                    "acierto": float((np.sign(yy) == s).mean()),
                    "q5_q1": float(yy[q > 0.8].mean() - yy[q <= 0.2].mean()),
                    "ic_1a": _ic(x[g.fecha < corte], g[y][g.fecha < corte]),
                    "ic_2a": _ic(x[g.fecha >= corte], g[y][g.fecha >= corte])}
            # t de la ventaja agrupando por fecha (los valores del mismo día se mueven juntos)
            por_dia = (s * yy).groupby(g.loc[m, "fecha"]).mean()
            fila["t"] = float(por_dia.mean() / por_dia.std() * np.sqrt(len(por_dia))) if len(por_dia) > 2 else np.nan
            for a, ga in g[m].groupby("anio"):
                fila[f"ic_{a}"] = _ic(ga[r], ga[y])
            filas.append(fila)
    return pd.DataFrame(filas)


CONJUNTOS = ["acc_h1", "acc_m5", "etf_h1", "NAS100", "BTC", "NVDA_1m"]

if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    todos = []
    for c in sys.argv[1:] or CONJUNTOS:
        df = conjunto(c)
        ev = evaluar(df, c)
        todos.append(ev)
        print(f"[{c}] {df['fecha'].nunique()} días, {len(df)} filas", flush=True)
    ev = pd.concat(todos)
    ev.to_csv(OUT / "rasgos.csv", index=False, float_format="%.4f")
