"""Rasgos de ANÁLISIS TÉCNICO en cada caída candidata (para mejorar las entradas).

Cada indicador se calcula sobre el panel completo (solo con datos <= t) y se guarda su
valor en las filas de la tabla de eventos (``eventos.parquet``), en
``data/cache/caidas/eventos_tec.parquet``. Donde la fuente no da apertura, máximo,
mínimo o volumen (CRSP), los indicadores que los necesitan quedan como no disponibles.

Medias móviles
- ``dist_ema{5,10,20,50}``, ``dist_sma{50,100}``, ``dist_wma20`` (WMA lineal): C/media - 1.
- ``kelt20``: (C - EMA20) / ATR14 (posición en el canal de Keltner).
- ``ema20_50_previo``: 1 si EMA20 > EMA50 diez sesiones antes (tendencia antes de caer).
- ``pend_sma50``, ``pend_sma200``: variación de la media en 20 sesiones (fuerza de la tendencia).

Volumen / VWAP
- ``dist_vwap10``, ``dist_vwap20``: C / VWAP (precio típico × volumen) - 1: cuánto está
  el precio por debajo del coste medio de quien compró en las últimas 2-4 semanas.
- ``mfi14`` (Money Flow Index), ``cmf20`` (Chaikin Money Flow), ``obv_pend20`` (pendiente
  del OBV normalizada), ``vol_bajista10`` (fracción del volumen de 10 d en días bajistas),
  ``vol_climax`` (volumen de hoy / máximo de los 60 d previos).

Volatilidad
- ``atr_pct`` (ATR14 / C), ``caida_atr`` ((máx. 20 d - C) / ATR14), ``atr_ratio``
  (ATR5 / ATR20: expansión), ``bb_z20`` (z de Bollinger), ``bb_ancho`` (ancho de bandas).

Osciladores
- ``rsi14``, ``rsi2``, ``estoc14`` (%K), ``macd_hist`` (histograma / C) y ``macd_giro``
  (1 si el histograma sube hoy).

Velas y precio
- ``sombra_inf`` ((min(O,C) - L) / (H - L): martillo), ``cuerpo`` (|C-O|/(H-L)),
  ``giro_clave`` (1 si hace mínimo inferior y cierra por encima del cierre anterior),
  ``dist_min252`` (C / mínimo de 52 semanas - 1).

Uso: python research/caidas/tecnicos.py
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

import eventos as ev
import motor as m

SALIDA = ev.SALIDA.with_name("eventos_tec.parquet")


def _ema(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.ewm(span=n, adjust=False, min_periods=n).mean()


def _wilder(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def _wma(x: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros_like(x)
    peso_total = n * (n + 1) / 2
    for k in range(n):  # peso n para hoy, 1 para hace n-1 sesiones
        sh = np.full_like(x, np.nan)
        sh[k:] = x[: len(x) - k]
        out += (n - k) * sh
    return out / peso_total


def construir() -> pd.DataFrame:
    t0 = time.time()
    p = m.cargar()
    e = ev.cargar()[["t", "j"]]
    ts, js = e.t.to_numpy(), e.j.to_numpy()
    sin = p.sin_apertura
    C = pd.DataFrame(p.close)
    H = pd.DataFrame(np.where(sin, np.nan, p.high))
    L = pd.DataFrame(np.where(sin, np.nan, p.low))
    O = pd.DataFrame(np.where(sin, np.nan, p.open))
    V = pd.DataFrame(p.volume)
    out = {}

    def guardar(nombre: str, mat) -> None:
        a = mat.to_numpy() if isinstance(mat, pd.DataFrame) else mat
        out[nombre] = a[ts, js].astype(np.float32)

    # ---- volatilidad
    cprev = C.shift(1)
    tr = pd.concat([(H - L), (H - cprev).abs(), (L - cprev).abs()]).groupby(level=0).max()
    atr14 = _wilder(tr, 14)
    atr5, atr20 = _wilder(tr, 5), _wilder(tr, 20)
    with np.errstate(invalid="ignore", divide="ignore"):
        guardar("atr_pct", atr14 / C)
        guardar("caida_atr", (C.rolling(20, min_periods=10).max() - C) / atr14)
        guardar("atr_ratio", atr5 / atr20)
        sma20, std20 = C.rolling(20).mean(), C.rolling(20).std()
        guardar("bb_z20", (C - sma20) / std20)
        guardar("bb_ancho", 4 * std20 / sma20)
    print(f"volatilidad {time.time() - t0:.0f}s", flush=True)
    # ---- medias móviles
    ema = {n: _ema(C, n) for n in (5, 10, 12, 20, 26, 50)}
    with np.errstate(invalid="ignore", divide="ignore"):
        for n in (5, 10, 20, 50):
            guardar(f"dist_ema{n}", C / ema[n] - 1)
        for n in (50, 100):
            guardar(f"dist_sma{n}", C / C.rolling(n).mean() - 1)
        guardar("dist_wma20", C.to_numpy() / _wma(C.to_numpy(), 20) - 1)
        guardar("kelt20", (C - ema[20]) / atr14)
        guardar("ema20_50_previo", ((ema[20] > ema[50]).astype(float)).shift(10))
        s50, s200 = C.rolling(50).mean(), C.rolling(200).mean()
        guardar("pend_sma50", s50 / s50.shift(20) - 1)
        guardar("pend_sma200", s200 / s200.shift(20) - 1)
    print(f"medias {time.time() - t0:.0f}s", flush=True)
    # ---- osciladores
    guardar("rsi14", m.rsi(14))
    guardar("rsi2", m.rsi(2))
    with np.errstate(invalid="ignore", divide="ignore"):
        ll, hh = L.rolling(14, min_periods=10).min(), H.rolling(14, min_periods=10).max()
        guardar("estoc14", (C - ll) / (hh - ll))
        macd = ema[12] - ema[26]
        hist = macd - _ema(macd, 9)
        guardar("macd_hist", hist / C)
        guardar("macd_giro", (hist > hist.shift(1)).astype(float).where(hist.notna()))
    # ---- volumen y VWAP
    with np.errstate(invalid="ignore", divide="ignore"):
        tp = ((H + L + C) / 3).where(H.notna(), C)
        for n in (10, 20):
            vwap = (tp * V).rolling(n, min_periods=n // 2).sum() / V.rolling(n, min_periods=n // 2).sum()
            guardar(f"dist_vwap{n}", C / vwap - 1)
        flujo = tp * V
        sube = tp > tp.shift(1)
        pos = flujo.where(sube, 0.0).rolling(14).sum()
        neg = flujo.where(~sube, 0.0).rolling(14).sum()
        guardar("mfi14", 100 - 100 / (1 + pos / neg))
        clv = ((C - L) - (H - C)) / (H - L)
        guardar("cmf20", (clv * V).rolling(20).sum() / V.rolling(20).sum())
        signo = np.sign(C.diff())
        obv = (signo * V).fillna(0).cumsum()
        guardar("obv_pend20", (obv - obv.shift(20)) / (20 * V.rolling(20).mean()))
        baja = (C < C.shift(1)).astype(float)
        guardar("vol_bajista10", (V * baja).rolling(10).sum() / V.rolling(10).sum())
        guardar("vol_climax", V / V.shift(1).rolling(60, min_periods=30).max())
    print(f"volumen {time.time() - t0:.0f}s", flush=True)
    # ---- velas y precio
    with np.errstate(invalid="ignore", divide="ignore"):
        rango = (H - L).where((H - L) > 0)
        guardar("sombra_inf", (np.minimum(O, C) - L) / rango)
        guardar("cuerpo", (C - O).abs() / rango)
        guardar("giro_clave", ((L < L.shift(1)) & (C > C.shift(1))).astype(float).where(L.notna()))
        guardar("dist_min252", C / C.rolling(252, min_periods=120).min() - 1)
    df = pd.DataFrame({"t": ts.astype(np.int32), "j": js.astype(np.int32), **out})
    df.to_parquet(SALIDA, index=False)
    print(f"{df.shape} -> {SALIDA} ({time.time() - t0:.0f}s)")
    return df


def cargar() -> pd.DataFrame:
    return pd.read_parquet(SALIDA)


if __name__ == "__main__":
    construir()
