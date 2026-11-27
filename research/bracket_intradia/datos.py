"""Pasa todas las fuentes a un formato común: matrices días × velas de la sesión regular de
Nueva York (9:30-16:00), con apertura, máximo, mínimo, cierre y volumen, más los rasgos
diarios que se conocen ANTES de abrir (ATR de 14 días, medias de 20 y 50, cierre previo).

Fuentes (ver ``descargar.py``):
- ``dukas_*``: Dukascopy, CFD de acciones de EE. UU. y oro, 1 minuto (precio de compra).
- ``BTC``: Binance BTCUSDT, 1 minuto, 2022 → nov-2026.
- ``XAU_oanda`` y ``NAS100_oanda``: OANDA, 1 minuto, 2005/2006 → may-2020 (repositorio
  público FutureSharks/financial-data).
- ``yahoo2m_*``: Yahoo, velas de 2 minutos de los últimos ~30 días hábiles, 40 valores.

Para el oro y el bitcóin (que cotizan casi todo el día) se usa la misma ventana de 9:30 a
16:00 de Nueva York: es la de más volumen y la que se opera con las acciones.

Uso: python research/bracket_intradia/datos.py
"""

from __future__ import annotations

import json
import lzma
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache" / "bracket"
RAW = CACHE / "raw"
PANELES = CACHE / "paneles"
OANDA = ROOT / "data" / "cache" / "fsdata" / "pyfinancialdata" / "data" / "currencies" / "oanda"
NY = "America/New_York"
ESCALA_DUKAS = {"BTCUSD": 10.0}  # el resto: precio × 1000


# ---------------------------------------------------------------- lectores por fuente
def leer_dukas(inst: str) -> pd.DataFrame:
    filas = []
    div = ESCALA_DUKAS.get(inst, 1000.0)
    for f in sorted((RAW / inst).glob("*.bi5")):
        b = f.read_bytes()
        if not b:
            continue
        raw = lzma.decompress(b)
        a = np.frombuffer(raw, dtype=">i4").reshape(-1, 6)
        v = np.frombuffer(raw, dtype=">f4").reshape(-1, 6)[:, 5]
        t0 = pd.Timestamp(f.stem, tz="UTC")
        df = pd.DataFrame({"t": t0 + pd.to_timedelta(a[:, 0], unit="s"),
                           "o": a[:, 1] / div, "c": a[:, 2] / div, "l": a[:, 3] / div,
                           "h": a[:, 4] / div, "v": v.astype(float)})
        filas.append(df[df["v"] > 0])  # Dukascopy repite el último precio con volumen 0
    return pd.concat(filas, ignore_index=True)


def leer_binance(simbolo: str = "BTCUSDT") -> pd.DataFrame:
    trozos = []
    for f in sorted((RAW / simbolo).glob("*.csv")):
        d = pd.read_csv(f, header=None, usecols=[0, 1, 2, 3, 4, 5])
        d.columns = ["t", "o", "h", "l", "c", "v"]
        unidad = np.where(d["t"] > 1e14, 1e-6, 1e-3)  # desde 2025 Binance da microsegundos
        d["t"] = pd.to_datetime((d["t"] * unidad).round().astype("int64"), unit="s", utc=True)
        trozos.append(d)
    return pd.concat(trozos, ignore_index=True)


def leer_oanda(inst: str) -> pd.DataFrame:
    trozos = [pd.read_csv(f) for f in sorted((OANDA / inst).rglob("*.csv"))]
    d = pd.concat(trozos, ignore_index=True)
    d = d.rename(columns={"time": "t", "open": "o", "high": "h", "low": "l", "close": "c",
                          "volume": "v"})
    d["t"] = pd.to_datetime(d["t"], utc=True)
    return d.sort_values("t").drop_duplicates("t")


def leer_yahoo(f: Path) -> pd.DataFrame:
    r = json.loads(f.read_text())["chart"]["result"][0]
    q = r["indicators"]["quote"][0]
    d = pd.DataFrame({"t": pd.to_datetime(r["timestamp"], unit="s", utc=True),
                      "o": q["open"], "h": q["high"], "l": q["low"], "c": q["close"],
                      "v": q["volume"]})
    return d.dropna(subset=["o", "h", "l", "c"])


def yahoo_diario(ticker: str) -> pd.DataFrame | None:
    f = CACHE / "yahoo_diario" / f"{ticker}.json"
    if not f.exists():
        return None
    r = json.loads(f.read_text())["chart"]["result"][0]
    q = r["indicators"]["quote"][0]
    d = pd.DataFrame({k: q[k] for k in ("open", "high", "low", "close")},
                     index=pd.to_datetime(r["timestamp"], unit="s", utc=True)
                     .tz_convert(NY).normalize().tz_localize(None))
    return d.dropna()


# ---------------------------------------------------------------- a matrices de sesión
def a_sesion(d: pd.DataFrame, paso: int = 1, min_real: float = 0.8) -> dict:
    """Velas en UTC → matrices días × (390/paso) de 9:30 a 16:00 de Nueva York."""
    n = 390 // paso
    t = d["t"].dt.tz_convert(NY)
    m = (t.dt.hour * 60 + t.dt.minute - 570).to_numpy()
    ok = (m >= 0) & (m < 390)
    dia = t.dt.tz_localize(None).dt.normalize().to_numpy()[ok]
    col = m[ok] // paso
    fechas, fila = np.unique(dia, return_inverse=True)
    D = len(fechas)
    M = {k: np.full((D, n), np.nan) for k in "OHLCV"}
    for k, src in zip("OHLCV", "ohlcv", strict=True):
        M[k][fila, col] = d[src].to_numpy(float)[ok]
    real = ~np.isnan(M["C"])
    # días válidos: casi todas las velas y cotización en los primeros 5 minutos
    valido = (real.mean(1) >= min_real) & real[:, : max(1, 5 // paso)].any(1)
    for k in "OHLCV":
        M[k] = M[k][valido]
    fechas, real = fechas[valido], real[valido]
    C = pd.DataFrame(M["C"]).ffill(axis=1).bfill(axis=1).to_numpy()
    vacio = np.isnan(M["O"])
    for k in "OHL":
        M[k] = np.where(vacio, C, M[k])
    M["C"] = C
    M["V"] = np.nan_to_num(M["V"])
    return {"fechas": fechas, "paso": paso, **M}


def ajustar_splits(P: dict) -> list:
    """Los CFD de Dukascopy no ajustan los splits (NVDA 10:1 en 2024): se detectan por el
    salto de la noche y se dividen los precios anteriores."""
    hechos = []
    O, C = P["O"], P["C"]
    for i in range(1, len(O)):
        r = C[i - 1, -1] / O[i, 0]
        for k in (2, 3, 4, 5, 10, 15, 20):
            if abs(r / k - 1) < 0.08:
                for key in "OHLC":
                    P[key][:i] /= k
                P["V"][:i] *= k
                hechos.append((str(P["fechas"][i])[:10], k))
                break
    return hechos


def rasgos_diarios(P: dict, diario: pd.DataFrame | None = None) -> None:
    """ATR(14), medias de 20 y 50 y cierre previo, todos conocidos antes de la apertura."""
    if diario is None:
        diario = pd.DataFrame({"open": P["O"][:, 0], "high": P["H"].max(1),
                               "low": P["L"].min(1), "close": P["C"][:, -1]},
                              index=pd.to_datetime(P["fechas"]))
    prev = diario["close"].shift()
    tr = np.maximum(diario["high"], prev) - np.minimum(diario["low"], prev)
    tabla = pd.DataFrame({
        "atr": tr.rolling(14).mean().shift(), "sma20": diario["close"].rolling(20).mean().shift(),
        "sma50": diario["close"].rolling(50).mean().shift(), "prev": prev,
        "ret5": (diario["close"] / diario["close"].shift(5) - 1).shift(),
    })
    tabla = tabla.reindex(pd.to_datetime(P["fechas"]))
    for k in tabla:
        P[k] = tabla[k].to_numpy()


def guardar(nombre: str, P: dict) -> None:
    PANELES.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PANELES / f"{nombre}.npz", **P)
    ok = ~np.isnan(P["atr"])
    print(f"{nombre:16s} {len(P['fechas']):5d} días {str(P['fechas'][0])[:10]} → "
          f"{str(P['fechas'][-1])[:10]}  ATR medio {np.nanmean(P['atr'][ok] / P['prev'][ok]):.2%}")


def cargar(nombre: str) -> dict:
    z = np.load(PANELES / f"{nombre}.npz", allow_pickle=True)
    return {k: z[k] for k in z.files}


def main() -> None:
    for inst in sorted(p.name for p in RAW.iterdir() if p.is_dir() and p.name != "BTCUSDT"):
        d = leer_dukas(inst)
        if len(d) < 1000:
            continue
        P = a_sesion(d)
        s = ajustar_splits(P) if inst.endswith("USUSD") else []
        if s:
            print("  splits", inst, s)
        rasgos_diarios(P)
        guardar("dukas_" + inst.replace("USUSD", "").replace("USD", "") if inst != "XAUUSD"
                else "dukas_XAU", P)
    P = a_sesion(leer_binance())
    rasgos_diarios(P)
    guardar("BTC", P)
    for inst, nombre in (("XAU_USD", "XAU_oanda"), ("NAS100_USD", "NAS100_oanda")):
        P = a_sesion(leer_oanda(inst))
        rasgos_diarios(P)
        guardar(nombre, P)
    for f in sorted((CACHE / "yahoo_2m").glob("*.json")):
        P = a_sesion(leer_yahoo(f), paso=2)
        if len(P["fechas"]) < 10:
            continue
        rasgos_diarios(P, yahoo_diario(f.stem))
        guardar("yahoo2m_" + f.stem.replace("=", "").replace("-", ""), P)


if __name__ == "__main__":
    main()
