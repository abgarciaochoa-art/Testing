"""Paneles días × velas de la sesión de Nueva York a partir de Yahoo (1 hora y 5 minutos).

- ``h1_TICKER``: 7 velas por día (9:30, 10:30, …, 15:30; la última de 30 minutos), ene-2024 →.
- ``m5_TICKER``: 78 velas de 5 minutos por día, últimos 60 días.

Los rasgos diarios (ATR de 14 días, medias de 20 y 50, cierre previo) salen del panel de una
hora (2 años) y se copian al de 5 minutos, para que este no pierda sus primeros 50 días.

Uso: python research/tendencia_intradia/paneles.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "bracket_intradia"))
import datos as base  # noqa: E402

CACHE = ROOT / "data" / "cache" / "tendencia"
PANELES = CACHE / "paneles"
NY = "America/New_York"


def leer(f: Path) -> pd.DataFrame:
    r = json.loads(f.read_text())["chart"]["result"][0]
    q = r["indicators"]["quote"][0]
    d = pd.DataFrame({"t": pd.to_datetime(r["timestamp"], unit="s", utc=True),
                      "o": q["open"], "h": q["high"], "l": q["low"], "c": q["close"],
                      "v": q["volume"]})
    return d.dropna(subset=["o", "h", "l", "c"])


def a_sesion(d: pd.DataFrame, paso: int) -> dict:
    """Como ``base.a_sesion`` pero con la última vela parcial (15:30-16:00 en 1 hora)."""
    n = -(-390 // paso)
    t = d["t"].dt.tz_convert(NY)
    m = (t.dt.hour * 60 + t.dt.minute - 570).to_numpy()
    ok = (m >= 0) & (m < 390)
    dia = t.dt.tz_localize(None).dt.normalize().to_numpy()[ok]
    col = m[ok] // paso
    fechas, fila = np.unique(dia, return_inverse=True)
    M = {k: np.full((len(fechas), n), np.nan) for k in "OHLCV"}
    for k, src in zip("OHLCV", "ohlcv", strict=True):
        M[k][fila, col] = d[src].to_numpy(float)[ok]
    real = ~np.isnan(M["C"])
    valido = (real.mean(1) >= (1.0 if paso >= 60 else 0.9)) & real[:, 0]
    for k in "OHLCV":
        M[k] = M[k][valido]
    fechas = fechas[valido]
    C = pd.DataFrame(M["C"]).ffill(axis=1).to_numpy()
    vacio = np.isnan(M["O"])
    for k in "OHL":
        M[k] = np.where(vacio, C, M[k])
    M["C"] = C
    M["V"] = np.nan_to_num(M["V"])
    return {"fechas": fechas, "paso": paso, **M}


def diario_de(P: dict) -> pd.DataFrame:
    return pd.DataFrame({"open": P["O"][:, 0], "high": P["H"].max(1), "low": P["L"].min(1),
                         "close": P["C"][:, -1]}, index=pd.to_datetime(P["fechas"]))


def main() -> None:
    PANELES.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted((CACHE / "yahoo_1h").glob("*.json")):
        tk = f.stem
        h1 = a_sesion(leer(f), 60)
        if len(h1["fechas"]) < 120:
            continue
        diario = diario_de(h1)
        base.rasgos_diarios(h1, diario)
        np.savez_compressed(PANELES / f"h1_{tk}.npz", **h1)
        f5 = CACHE / "yahoo_5m" / f"{tk}.json"
        if f5.exists():
            m5 = a_sesion(leer(f5), 5)
            # el diario de 1 hora cubre los días del panel de 5 minutos (mismas sesiones)
            base.rasgos_diarios(m5, diario.reindex(diario.index.union(pd.to_datetime(m5["fechas"])))
                                .ffill())
            np.savez_compressed(PANELES / f"m5_{tk}.npz", **m5)
        n += 1
    print(n, "valores")


def cargar(nombre: str) -> dict:
    z = np.load(PANELES / f"{nombre}.npz", allow_pickle=True)
    return {k: z[k] for k in z.files}


if __name__ == "__main__":
    main()
