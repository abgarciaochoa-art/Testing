"""Ranking de liquidez del S&P 500 (Yahoo, últimos ~6 meses de velas diarias).

Por cada acción: volumen medio en dólares, rango diario medio (máximo-mínimo / cierre previo)
y ATR(14) en %. Lo que importa para la horquilla no es solo el volumen: es cuántas "R" se mueve
al día. Salida: research/bracket_intradia/resultados/liquidez.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "cache" / "bracket" / "yahoo_diario"
OUT = Path(__file__).resolve().parent / "resultados"


def cargar(f: Path) -> pd.DataFrame | None:
    try:
        r = json.loads(f.read_text())["chart"]["result"][0]
        q = r["indicators"]["quote"][0]
        df = pd.DataFrame({k: q[k] for k in ("open", "high", "low", "close", "volume")},
                          index=pd.to_datetime(r["timestamp"], unit="s"))
        return df.dropna()
    except Exception:  # noqa: BLE001
        return None


def main() -> pd.DataFrame:
    filas = []
    for f in sorted(SRC.glob("*.json")):
        df = cargar(f)
        if df is None or len(df) < 60:
            continue
        df = df.iloc[-126:]
        prev = df["close"].shift()
        tr = np.maximum(df["high"], prev) - np.minimum(df["low"], prev)
        rango = (df["high"] - df["low"]) / prev
        dv = df["close"] * df["volume"]
        filas.append({
            "ticker": f.stem, "precio": df["close"].iloc[-1],
            "dolar_vol_m": dv.median() / 1e6,
            "rango_medio_pct": 100 * rango.mean(),
            "atr14_pct": 100 * (tr / prev).rolling(14).mean().iloc[-1],
            "dias_rango_mayor_1_5pct": 100 * (rango > 0.015).mean(),
            "dias_rango_mayor_3pct": 100 * (rango > 0.03).mean(),
        })
    t = pd.DataFrame(filas).sort_values("dolar_vol_m", ascending=False).reset_index(drop=True)
    OUT.mkdir(exist_ok=True)
    t.to_csv(OUT / "liquidez.csv", index=False, float_format="%.3f")
    return t


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    t = main()
    print(t.head(30).to_string())
    print(t[t.ticker.isin(["SPY", "QQQ", "GLD", "GC=F", "BTC-USD"])].to_string())
