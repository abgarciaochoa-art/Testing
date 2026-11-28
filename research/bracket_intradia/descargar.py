"""Descarga velas de 1 minuto de Dukascopy (CFD de acciones de EE. UU., índice y oro) y de
Binance (BTCUSDT), y velas de Yahoo (diarias para el ranking de liquidez; 1 minuto de los
últimos días para contrastar precios y volumen).

Dukascopy: un fichero por instrumento y día, ``{INST}/{AAAA}/{MM-1}/{DD}/BID_candles_min_1.bi5``
(LZMA; registros de 24 bytes: segundos desde las 00:00 UTC, apertura, cierre, mínimo, máximo
como enteros y volumen como float32). Se guardan tal cual en ``data/cache/bracket/raw``.

Uso: python research/bracket_intradia/descargar.py [dukascopy|binance|yahoo] ...
"""

from __future__ import annotations

import io
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache" / "bracket"
RAW = CACHE / "raw"

DUKAS = ["NVDAUSUSD", "XAUUSD", "USA500IDXUSD", "TSLAUSUSD", "AMDUSUSD", "METAUSUSD",
         "AAPLUSUSD", "MSFTUSUSD", "AMZNUSUSD"]  # por prioridad
INICIO = date(2022, 1, 3)
FIN = date(2026, 11, 26)

_s = requests.Session()
_s.headers["User-Agent"] = "Mozilla/5.0"


def _get(url: str, intentos: int = 12) -> bytes | None:
    espera = 2.0
    for _ in range(intentos):
        try:
            r = _s.get(url, timeout=25)
            if r.status_code == 200:
                return r.content
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(espera)
        espera = min(espera * 1.6, 60)
    raise RuntimeError(f"sin respuesta: {url}")


def _dia_dukas(inst: str, d: date) -> str:
    destino = RAW / inst / f"{d.isoformat()}.bi5"
    if destino.exists():
        return "ya"
    url = (f"https://datafeed.dukascopy.com/datafeed/{inst}/{d.year}/{d.month - 1:02d}/"
           f"{d.day:02d}/BID_candles_min_1.bi5")
    b = _get(url)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_bytes(b or b"")  # vacío = no hay fichero (festivo o sin datos)
    return "ok" if b else "vacío"


def dukascopy(instrumentos: list[str], hilos: int = 3) -> None:
    """Dukascopy limita mucho la velocidad (503 y cortes): pocos hilos, del día más reciente
    hacia atrás y un instrumento detrás de otro, para que una descarga parcial ya sirva."""
    dias = [FIN - timedelta(n) for n in range((FIN - INICIO).days + 1)]
    dias = [d for d in dias if d.weekday() < 5]
    tareas = [(i, d) for i in instrumentos for d in dias
              if not (RAW / i / f"{d.isoformat()}.bi5").exists()]
    hechas, t0 = 0, time.time()
    with ThreadPoolExecutor(hilos) as ex:
        futs = {ex.submit(_dia_dukas, i, d): (i, d) for i, d in tareas}
        for f in as_completed(futs):
            hechas += 1
            try:
                f.result()
            except Exception as e:  # noqa: BLE001
                print("ERROR", futs[f], e, flush=True)
            if hechas % 100 == 0:
                print(f"{hechas}/{len(tareas)} en {time.time() - t0:.0f}s", flush=True)


def binance(simbolo: str = "BTCUSDT") -> None:
    """Ficheros mensuales de velas de 1 minuto de data.binance.vision (y diarios del mes en curso)."""
    destino = RAW / simbolo
    destino.mkdir(parents=True, exist_ok=True)
    m = date(INICIO.year, INICIO.month, 1)
    while m <= FIN:
        f = destino / f"{m:%Y-%m}.csv"
        if not f.exists():
            url = (f"https://data.binance.vision/data/spot/monthly/klines/{simbolo}/1m/"
                   f"{simbolo}-1m-{m:%Y-%m}.zip")
            b = _get(url)
            if b:
                with zipfile.ZipFile(io.BytesIO(b)) as z:
                    f.write_bytes(z.read(z.namelist()[0]))
            else:  # mes en curso: ficheros diarios
                trozos = []
                d = m
                while d.month == m.month and d <= FIN:
                    url = (f"https://data.binance.vision/data/spot/daily/klines/{simbolo}/1m/"
                           f"{simbolo}-1m-{d:%Y-%m-%d}.zip")
                    bd = _get(url, intentos=2) if d < FIN else None
                    if bd:
                        with zipfile.ZipFile(io.BytesIO(bd)) as z:
                            trozos.append(z.read(z.namelist()[0]))
                    d += timedelta(1)
                if trozos:
                    f.write_bytes(b"".join(trozos))
            print(simbolo, m, "ok" if f.exists() else "sin datos", flush=True)
        m = date(m.year + (m.month == 12), m.month % 12 + 1, 1)


def yahoo_diario(tickers: list[str], rango: str = "6mo") -> None:
    destino = CACHE / "yahoo_diario"
    destino.mkdir(parents=True, exist_ok=True)

    def uno(t: str) -> None:
        f = destino / f"{t}.json"
        if f.exists():
            return
        url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{t.replace('.', '-')}"
               f"?interval=1d&range={rango}")
        b = _get(url, intentos=4)
        if b:
            f.write_bytes(b)

    with ThreadPoolExecutor(4) as ex:
        list(ex.map(uno, tickers))


def yahoo_minuto(tickers: list[str]) -> None:
    """1 minuto de Yahoo: solo ~30 días hacia atrás, en tramos de 7 días."""
    destino = CACHE / "yahoo_1m"
    destino.mkdir(parents=True, exist_ok=True)
    fin = int(time.time())
    for t in tickers:
        for k in range(4):
            f = destino / f"{t}_{k}.json"
            if f.exists():
                continue
            p2 = fin - k * 7 * 86400
            p1 = p2 - 7 * 86400
            url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{t}?interval=1m"
                   f"&period1={p1}&period2={p2}")
            b = _get(url, intentos=4)
            if b:
                f.write_bytes(b)


if __name__ == "__main__":
    que = sys.argv[1] if len(sys.argv) > 1 else "dukascopy"
    if que == "dukascopy":
        dukascopy(sys.argv[2:] or DUKAS)
    elif que == "binance":
        binance()
    elif que == "yahoo":
        import pandas as pd

        sp = pd.read_csv(ROOT / "data" / "seed" / "sp500_constituents.csv")
        col = [c for c in sp.columns if c.lower() in ("symbol", "ticker")][0]
        yahoo_diario([*sp[col].astype(str).tolist(), "SPY", "QQQ", "GLD", "GC=F", "BTC-USD"])
        yahoo_minuto(["NVDA", "TSLA", "AAPL", "AMD", "MSFT", "AMZN", "META", "GOOGL", "AVGO",
                      "SPY", "GC=F", "BTC-USD"])
