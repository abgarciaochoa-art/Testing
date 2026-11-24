"""Datos intradía para el estudio de Opening Range Breakout (ORB).

Índices (velas de 1 minuto, sesión regular de cada mercado), repositorio público
``FutureSharks/financial-data``:
  - OANDA SPX500_USD, NAS100_USD, US2000_USD (CFD del S&P 500, Nasdaq-100, Russell 2000),
    2005 → may-2020, UTC. Sesión 9:30-16:00 Nueva York (390 velas).
  - OANDA UK100_GBP (FTSE 100) y FR40_EUR (CAC 40): sesión 8:00-16:30 Londres / 9:00-17:30 París.
  - HistData GRXEUR (DAX) y ETXEUR (Euro Stoxx 50), 2010-2018, hora EST fija (UTC-5).
    Sesión 9:00-17:30 de Fráncfort (510 velas).

Acciones (velas de 5 minutos, 9:30-16:00, ajustadas por splits y dividendos), repositorio
público ``fatherpetrek-afk/QFin`` (descargadas de Polygon/Futu): ~200 valores grandes del
S&P 500 (las 100 mayores de hoy + ~100 del resto), sep-2018 → nov-2026. OJO: universo de hoy,
con sesgo de supervivencia; se compensa en parte usando solo reglas simétricas (largos y cortos)
y comparando con comprar y mantener los mismos valores.

Uso: python research/orb/datos.py   (crea data/cache/orb/*.npz)
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache" / "orb"
FD = CACHE / "fd" / "pyfinancialdata" / "data"
QF = CACHE / "probe_QFin" / "cs_intraday" / "data"

INDICES = {
    # nombre: (fuente, carpeta, zona, apertura (h, m), velas)
    "SPX": ("oanda", "SPX500_USD", "America/New_York", (9, 30), 390),
    "NDX": ("oanda", "NAS100_USD", "America/New_York", (9, 30), 390),
    "RUT": ("oanda", "US2000_USD", "America/New_York", (9, 30), 390),
    "FTSE": ("oanda", "UK100_GBP", "Europe/London", (8, 0), 510),
    "CAC": ("oanda", "FR40_EUR", "Europe/Paris", (9, 0), 510),
    "DAX": ("histdata", "GRXEUR", "Europe/Berlin", (9, 0), 510),
    "STOXX": ("histdata", "ETXEUR", "Europe/Berlin", (9, 0), 510),
}
# SPY (ETF) en velas de 1 minuto 2000 → oct-2025, hora de Nueva York, etiqueta = inicio de la vela
# (repositorio público prsdro/spy, spy_contents/spy/*.csv). Sirve para validar después de 2020.
EXTRA = {"SPY": ("spy", "spy_contents/spy", "America/New_York", (9, 30), 390)}


def descargar() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    if not (CACHE / "fd").exists():
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                        "https://github.com/FutureSharks/financial-data", str(CACHE / "fd")], check=True)
        sets = [f"pyfinancialdata/data/{'currencies/oanda' if s == 'oanda' else 'stocks/histdata'}/{c}"
                for s, c, *_ in INDICES.values()]
        subprocess.run(["git", "-C", str(CACHE / "fd"), "sparse-checkout", "set", *sets], check=True)
    if not (CACHE / "probe_QFin").exists():
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                        "https://github.com/fatherpetrek-afk/QFin", str(CACHE / "probe_QFin")], check=True)
        subprocess.run(["git", "-C", str(CACHE / "probe_QFin"), "sparse-checkout", "set",
                        "cs_intraday/data/5m", "cs_intraday/data/5m_flat"], check=True)


def a_sesion(t: pd.Series, o, h, l, c, apertura, velas, paso=1) -> dict:  # noqa: E741
    """Velas con hora local (inicio de vela) → matrices días × velas de la sesión."""
    m = ((t.dt.hour * 60 + t.dt.minute - (apertura[0] * 60 + apertura[1])) // paso).to_numpy()
    ok = (m >= 0) & (m < velas)
    dia = t.dt.normalize().dt.tz_localize(None).to_numpy()[ok] if t.dt.tz is not None else \
        t.dt.normalize().to_numpy()[ok]
    m = m[ok]
    fechas, fila = np.unique(dia, return_inverse=True)
    D = len(fechas)
    out = {k: np.full((D, velas), np.nan) for k in "OHLC"}
    for k, s in zip("OHLC", (o, h, l, c), strict=True):
        out[k][fila, m] = np.asarray(s, dtype=float)[ok]
    real = ~np.isnan(out["C"])
    C = pd.DataFrame(out["C"]).ffill(axis=1).to_numpy()
    primera = np.argmax(real, axis=1)
    abre = out["O"][np.arange(D), primera]
    C = np.where(np.isnan(C), abre[:, None], C)
    for k in "OHL":
        out[k] = np.where(real, out[k], C)
    out["C"] = C
    ultimo = velas - 1 - np.argmax(real[:, ::-1], axis=1)
    validos = (real.sum(1) >= 0.5 * velas) & (primera < max(5 // paso, 1))
    return dict(fechas=fechas[validos], O=out["O"][validos], H=out["H"][validos],
                L=out["L"][validos], C=out["C"][validos], ultimo=ultimo[validos])


def indice(nombre: str) -> dict:
    ruta = CACHE / f"idx_{nombre}.npz"
    if ruta.exists():
        return dict(np.load(ruta))
    fuente, carpeta, zona, ap, velas = {**INDICES, **EXTRA}[nombre]
    if fuente == "spy":
        partes = [pd.read_csv(f) for f in sorted((CACHE / "probe_spy" / carpeta).glob("*.csv"))]
        d = pd.concat(partes, ignore_index=True).drop_duplicates("timestamp").sort_values("timestamp")
        t = pd.to_datetime(d.timestamp)
        x = a_sesion(t, d.open, d.high, d.low, d.close, ap, velas)
    elif fuente == "oanda":
        partes = [pd.read_csv(f) for f in sorted((FD / "currencies" / "oanda" / carpeta).glob("*/*.csv"))]
        d = pd.concat(partes, ignore_index=True).drop_duplicates("time").sort_values("time")
        t = pd.to_datetime(d.time).dt.tz_localize("UTC").dt.tz_convert(zona)
        x = a_sesion(t, d.open, d.high, d.low, d.close, ap, velas)
    else:  # HistData: EST fija (UTC-5), sin horario de verano
        partes = [pd.read_csv(f, sep=";", header=None, names=["t", "o", "h", "l", "c", "v"])
                  for f in sorted((FD / "stocks" / "histdata" / carpeta).glob("*.csv"))]
        d = pd.concat(partes, ignore_index=True).drop_duplicates("t").sort_values("t")
        t = (pd.to_datetime(d.t, format="%Y%m%d %H%M%S") + pd.Timedelta(hours=5)).dt.tz_localize("UTC") \
            .dt.tz_convert(zona)
        x = a_sesion(t, d.o, d.h, d.l, d.c, ap, velas)
    # cierre del día anterior (última vela real) para huecos
    D = len(x["fechas"])
    cierre = x["C"][np.arange(D), x["ultimo"]]
    x["prev"] = np.r_[np.nan, cierre[:-1]]
    np.savez_compressed(ruta, **x)
    return x


def acciones() -> dict:
    """Panel de acciones: arrays (N, D, 78) float32 de O, H, L, C, V; velas de 9:30 a 15:55."""
    ruta = CACHE / "acciones_5m.npz"
    if ruta.exists():
        z = np.load(ruta, allow_pickle=True)
        return {k: z[k] for k in z.files}
    files = sorted((QF / "5m").glob("*.parquet")) + sorted((QF / "5m_flat").glob("*.parquet"))
    tablas, nombres = [], []
    for f in files:
        d = pd.read_parquet(f, columns=["time", "open", "high", "low", "close", "volume"])
        d["inicio"] = d.time - pd.Timedelta(minutes=5)  # la vela 9:35 es la de 9:30-9:35
        d = d[(d.inicio.dt.hour * 60 + d.inicio.dt.minute >= 570) & (d.inicio.dt.hour < 16)]
        tablas.append(d)
        nombres.append(f.stem)
    fechas = np.unique(np.concatenate([t.inicio.dt.normalize().to_numpy() for t in tablas]))
    # quedarse con días en los que cotiza al menos la mitad del universo
    N, D, B = len(tablas), len(fechas), 78
    arr = {k: np.full((N, D, B), np.nan, dtype=np.float32) for k in "OHLCV"}
    pos = {f: i for i, f in enumerate(fechas)}
    for i, d in enumerate(tablas):
        fi = np.array([pos[x] for x in d.inicio.dt.normalize().to_numpy()])
        b = ((d.inicio.dt.hour * 60 + d.inicio.dt.minute - 570) // 5).to_numpy()
        for k, col in zip("OHLCV", ("open", "high", "low", "close", "volume"), strict=True):
            arr[k][i, fi, b] = d[col].to_numpy(dtype=np.float32)
    cuenta = (~np.isnan(arr["C"][:, :, 0])).sum(0)
    ok = cuenta >= 0.5 * N
    for k in arr:
        arr[k] = arr[k][:, ok]
    fechas = fechas[ok]
    # última vela del día (cierre anticipado a las 13:00)
    ultimo = np.where((~np.isnan(arr["C"])).sum(0).max(0) <= 42, 41, 77)  # noqa
    vivo = (~np.isnan(arr["C"])).any(0)  # (D, 78) alguna acción tiene vela
    ultimo = 77 - np.argmax(vivo[:, ::-1], axis=1)
    out = dict(tickers=np.array(nombres), fechas=fechas, ultimo=ultimo, **arr)
    np.savez(ruta, **out)
    return out


if __name__ == "__main__":
    descargar()
    for n in INDICES:
        x = indice(n)
        r = x["C"][np.arange(len(x["fechas"])), x["ultimo"]] / x["O"][:, 0] - 1
        print(f"{n}: {len(x['fechas'])} días {x['fechas'][0]} → {x['fechas'][-1]}; "
              f"|apertura→cierre| medio {np.nanmean(np.abs(r)) * 1e4:.0f} pb")
    a = acciones()
    print("acciones:", a["C"].shape, a["fechas"][0], "→", a["fechas"][-1])
