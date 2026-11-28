"""Datos para el estudio de tendencia intradía (Yahoo):

- velas de 1 hora de los últimos ~730 días (enero 2024 →) de las 80 acciones del S&P 500
  con más volumen en dólares, más SPY, QQQ, IWM, SMH y XLK (mercado y sectores);
- velas de 5 minutos de los últimos 60 días de los mismos valores.

Se guardan tal cual en ``data/cache/tendencia/yahoo_{1h,5m}/TICKER.json``.

Uso: python research/tendencia_intradia/descargar.py
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "bracket_intradia"))
import descargar as base  # noqa: E402

CACHE = ROOT / "data" / "cache" / "tendencia"
ETFS = ["SPY", "QQQ", "IWM", "SMH", "XLK"]


def universo(n: int = 80) -> list[str]:
    liq = pd.read_csv(ROOT / "research" / "bracket_intradia" / "resultados" / "liquidez.csv")
    acc = [t for t in liq["ticker"] if t not in ("BTC-USD", "GOOG", "SPY", "QQQ", "GLD", "GC=F")]
    return [t.replace(".", "-") for t in acc[:n]] + ETFS


def bajar(ticker: str, intervalo: str, rango: str) -> None:
    destino = CACHE / f"yahoo_{intervalo}" / f"{ticker}.json"
    if destino.exists():
        return
    destino.parent.mkdir(parents=True, exist_ok=True)
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval={intervalo}"
           f"&range={rango}")
    try:
        b = base._get(url, intentos=4)
    except RuntimeError:  # p. ej. valores que cotizan desde hace menos de 2 años
        b = base._get(url.replace("range=730d", "range=1y"), intentos=2) if "730d" in url else None
    if b:
        destino.write_bytes(b)


if __name__ == "__main__":
    u = universo()
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(lambda t: bajar(t, "1h", "730d"), u))
        list(ex.map(lambda t: bajar(t, "5m", "60d"), u))
    for iv in ("1h", "5m"):
        print(iv, len(list((CACHE / f"yahoo_{iv}").glob("*.json"))), "de", len(u))
