"""Panel de precios REALES del S&P 500 con pertenencia point-in-time.

Fuente: release pública de `github.com/Johnbrick123/sp500-data` (Yahoo + CRSP +
Tiingo para los nombres deslistados). Incluye las quiebras con su trayectoria real
hasta el final (Lehman cierra en 0,13 $, Washington Mutual en 0,16 $), que es lo que
importa para una estrategia que compra caídas: las empresas que caen y NO se
recuperan son precisamente las que un dataset de "miembros actuales" borra.

Este script:
1. descarga (si falta) precios y pertenencia a `data/cache/caidas/`;
2. mide la cobertura de precios de los miembros PIT por fecha;
3. contrasta la pertenencia con la semilla del repo (`data/seed/...`);
4. limpia picos imposibles (subida >150 % revertida al día siguiente);
5. guarda matrices fecha×ticker en `panel.npz`.

Uso: python research/caidas/datos.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache" / "caidas"
BASE_URL = "https://github.com/Johnbrick123/sp500-data/releases/download/data"
START = "2000-01-01"  # dos años de historia para las ventanas largas antes de 2002


def descargar() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    for nombre in ("prices.parquet", "membership_intervals.parquet"):
        destino = CACHE / nombre
        if not destino.exists():
            print(f"descargando {nombre} ...")
            subprocess.run(
                ["curl", "-sS", "-L", "--fail", "-o", str(destino), f"{BASE_URL}/{nombre}"],
                check=True,
            )


def cobertura(prices: pd.DataFrame, miembros: pd.DataFrame) -> pd.DataFrame:
    rango = prices.dropna(subset=["adj_close"]).groupby("ticker").date.agg(["min", "max"])
    filas = []
    for d in pd.date_range("1996-06-30", prices.date.max(), freq="YE"):
        mem = miembros[(miembros.start <= d) & (miembros.end.isna() | (miembros.end > d))].ticker
        r = rango.reindex(mem)
        filas.append({"fecha": d.date(), "miembros": len(mem),
                      "con_precio_pct": round(100 * ((r["min"] <= d) & (r["max"] >= d)).mean(), 1)})
    return pd.DataFrame(filas)


def contraste_semilla(miembros: pd.DataFrame) -> pd.DataFrame:
    """Compara el número y la identidad de miembros con la semilla diaria del repo."""
    semilla = pd.read_csv(ROOT / "data" / "seed" / "sp500_historical_components.csv")
    semilla["date"] = pd.to_datetime(semilla["date"])
    base = lambda t: t.split("-")[0].replace(".", "-").upper()  # noqa: E731
    filas = []
    for d in pd.to_datetime(["2003-06-30", "2008-06-30", "2013-06-28", "2018-06-29", "2024-06-28"]):
        fila = semilla[semilla.date <= d].iloc[-1]
        s = {t.strip().replace(".", "-").upper() for t in fila["tickers"].split(",")}
        m = {base(t) for t in miembros[(miembros.start <= d) & (miembros.end.isna() | (miembros.end > d))].ticker}
        filas.append({"fecha": d.date(), "semilla": len(s), "dataset": len(m),
                      "coinciden": len(s & m), "jaccard": round(len(s & m) / len(s | m), 3)})
    return pd.DataFrame(filas)


def construir() -> None:
    descargar()
    prices = pd.read_parquet(CACHE / "prices.parquet")
    miembros = pd.read_parquet(CACHE / "membership_intervals.parquet")
    for c in ("start", "end"):
        miembros[c] = pd.to_datetime(miembros[c])

    cob = cobertura(prices, miembros)
    sem = contraste_semilla(miembros)
    print(cob.to_string(index=False))
    print(sem.to_string(index=False))

    etfs = ["SPY", "RSP", "QQQ"]
    tickers = sorted(set(miembros.ticker) | set(etfs))
    p = prices[prices.ticker.isin(tickers) & (prices.date >= START)].copy()
    p = p.drop_duplicates(["date", "ticker"], keep="last")

    # Factor de ajuste por dividendos+splits aplicado también a open/high/low.
    f = (p.adj_close / p.close).where(p.close > 0)
    for c in ("open", "high", "low"):
        p[f"{c}_adj"] = p[c] * f

    # Precio REAL negociado (para filtros de precio). Convención por fuente, medida en
    # las fechas de split: en Yahoo `close` viene ajustado por splits (sin salto en el
    # split), en CRSP/Tiingo/Wiki es el precio real (salta 1/f). Real en Yahoo = close ×
    # producto de los splits POSTERIORES a la fecha. Sin esto, AAPL cotizaba a 0,31 $ en
    # 2003 y NVDA a 0,21 $, y un filtro "precio >= 5 $" las expulsaba durante años.
    p = p.sort_values(["ticker", "date"])
    fs = p["stock_splits"].where(~p["stock_splits"].isin([0, 1]) & p["stock_splits"].notna(), 1.0)
    desde_t = fs[::-1].groupby(p["ticker"][::-1]).cumprod()[::-1]  # producto de splits >= t
    posteriores = desde_t / fs
    p["precio_real"] = np.where(p["source"].str.startswith("yahoo"), p["close"] * posteriores, p["close"])

    def ancho(col: str) -> pd.DataFrame:
        return p.pivot(index="date", columns="ticker", values=col).sort_index()

    close = ancho("adj_close")
    fechas = close.index[close.notna().sum(axis=1) > 300]  # solo sesiones con mercado completo
    close = close.loc[fechas]
    tick = close.columns
    mats = {"close": close}
    for src, dst in (("open_adj", "open"), ("high_adj", "high"), ("low_adj", "low"),
                     ("close", "raw_close"), ("precio_real", "precio_real"), ("volume", "volume")):
        mats[dst] = ancho(src).reindex(index=fechas, columns=tick)

    # Limpieza de picos: subida >150 % que revierte >60 % al día siguiente -> NaN.
    c = mats["close"].to_numpy()
    r = c[1:] / c[:-1] - 1
    pico = np.zeros_like(c, dtype=bool)
    pico[1:-1] = (r[:-1] > 1.5) & (r[1:] < -0.6)
    n_picos = int(pico.sum())
    for k in mats:
        if k not in ("volume", "precio_real"):
            mats[k] = mats[k].mask(pico)
    print(f"picos eliminados: {n_picos}")

    # Pertenencia PIT diaria (entrada permitida solo si es miembro ese día).
    miembro = pd.DataFrame(False, index=fechas, columns=tick)
    for row in miembros.itertuples():
        if row.ticker in miembro.columns:
            fin = row.end if pd.notna(row.end) else fechas[-1] + pd.Timedelta(days=1)
            miembro.loc[(fechas >= row.start) & (fechas < fin), row.ticker] = True

    # open/high/low ausentes en algunas fuentes (CRSP): se marca para que los rasgos
    # intradía (hueco nocturno, IBS) queden como no disponibles, y se rellenan con close
    # (la ejecución "a la apertura" pasa a ser al cierre de esa sesión: más tarde, nunca antes).
    sin_apertura = (mats["open"].isna() | mats["high"].isna() | mats["low"].isna()) & mats["close"].notna()
    mats["sin_apertura"] = sin_apertura.astype(float)
    for k in ("open", "high", "low"):
        mats[k] = mats[k].fillna(mats["close"].where(mats[k].isna()))

    np.savez_compressed(
        CACHE / "panel.npz",
        fechas=fechas.values.astype("datetime64[D]"),
        tickers=np.array(tick, dtype=object),
        miembro=miembro.to_numpy(),
        **{k: v.to_numpy(dtype=np.float64) for k, v in mats.items()},
    )
    meta = {
        "fuente": BASE_URL,
        "fechas": [str(fechas[0].date()), str(fechas[-1].date())],
        "n_tickers": len(tick),
        "picos_eliminados": n_picos,
        "cobertura": cob.astype(str).to_dict("records"),
        "contraste_semilla": sem.astype(str).to_dict("records"),
    }
    (CACHE / "panel_meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print(f"panel: {len(fechas)} sesiones x {len(tick)} tickers -> {CACHE / 'panel.npz'}")


if __name__ == "__main__":
    sys.exit(construir())
