"""Semáforo de tendencia intradía, versión autónoma para usar con tus propios datos.

Solo necesita pandas y un DataFrame de velas intradía (cualquier tamaño de vela de 1 a 60
minutos) con índice de fecha y hora en Nueva York y columnas open, high, low, close, volume,
solo de la sesión regular (9:30-16:00). Devuelve, para cada vela cerrada:

  sigma   movimiento desde la apertura frente al movimiento medio a esa hora (14 días)
  rvol    volumen acumulado frente a la media de 20 días a esa hora
  vwap    VWAP del día
  estado  0 rojo, 1 ámbar, 2 verde, 3 verde fuerte (con signo: + alcista, − bajista)
  entrada +1/−1 en la primera vela del día en que el estado llega a verde entre las 10:00 y
          las 14:30 (la señal de la prueba operativa); 0 en el resto

Ejemplo:
    import yfinance as yf
    df = yf.download("NVDA", interval="5m", period="60d", prepost=False)
    df = df.tz_convert("America/New_York").rename(columns=str.lower)
    print(semaforo(df).tail())

Reglas (ver ``semaforo.py`` y el informe):
  rojo          |sigma| < 1
  ámbar         |sigma| ≥ 1 sin volumen relativo ≥ 2 o con el precio al otro lado de la VWAP
  verde         |sigma| ≥ 1, rvol ≥ 2 y el precio del lado de la dirección respecto a la VWAP
  verde fuerte  lo mismo con rvol ≥ 3
Gestión probada: stop a 0,25 ATR de 14 días; a +1,8R el stop sube a +1R y después sigue a
0,8R del máximo; todo cerrado al final del día; una entrada por valor y día.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def semaforo(df: pd.DataFrame, dias_ruido: int = 14, dias_vol: int = 20, sigma_min: float = 1.0,
             rvol_min: float = 2.0, rvol_fuerte: float = 3.0) -> pd.DataFrame:
    d = df.copy()
    d["dia"] = d.index.normalize()
    d["k"] = d.groupby("dia").cumcount()                    # n.º de vela dentro del día
    apertura = d.groupby("dia")["open"].transform("first")
    d["mov"] = d["close"] / apertura - 1
    # movimiento absoluto medio a esa misma vela en los días anteriores (sin el de hoy)
    tabla = d.pivot_table(index="dia", columns="k", values="mov").abs()
    base = tabla.rolling(dias_ruido, min_periods=dias_ruido).mean().shift()
    d["base"] = base.stack().reindex(pd.MultiIndex.from_arrays([d["dia"], d["k"]])).to_numpy()
    d["sigma"] = d["mov"] / d["base"]
    # volumen acumulado frente a la media de días anteriores a la misma vela
    d["vol_ac"] = d.groupby("dia")["volume"].cumsum()
    tv = d.pivot_table(index="dia", columns="k", values="vol_ac")
    media = tv.rolling(dias_vol, min_periods=dias_vol).mean().shift()
    d["vol_base"] = media.stack().reindex(pd.MultiIndex.from_arrays([d["dia"], d["k"]])).to_numpy()
    d["rvol"] = d["vol_ac"] / d["vol_base"]
    tp = (d["high"] + d["low"] + d["close"]) / 3
    d["vwap"] = (tp * d["volume"]).groupby(d["dia"]).cumsum() / d["vol_ac"].replace(0, np.nan)
    direccion = np.sign(d["close"] - apertura)
    fuera = d["sigma"].abs() >= sigma_min
    lado = np.sign(d["close"] - d["vwap"]) == direccion
    est = np.where(fuera, 1, 0)
    est = np.where(fuera & (d["rvol"] >= rvol_min) & lado, 2, est)
    est = np.where(fuera & (d["rvol"] >= rvol_fuerte) & lado, 3, est)
    est = np.where(d["sigma"].isna(), 0, est)
    d["estado"] = (est * direccion).astype(int)
    # señal: primera vela del día con verde entre las 10:00 y las 14:30 (minuto de cierre)
    paso = int(round((d.index.to_series().diff().dropna().dt.total_seconds().median()) / 60))
    minuto_cierre = (d.index.hour * 60 + d.index.minute - 570) + paso
    ventana = (minuto_cierre >= 30) & (minuto_cierre <= 300)
    verde = (np.abs(d["estado"]) >= 2) & ventana
    primera = verde & ~verde.groupby(d["dia"]).shift(fill_value=False).groupby(d["dia"]).cummax()
    d["entrada"] = np.where(primera, np.sign(d["estado"]), 0).astype(int)
    return d[["open", "high", "low", "close", "volume", "vwap", "sigma", "rvol", "estado",
              "entrada"]]


if __name__ == "__main__":
    # comprobación: con los datos de 5 minutos del estudio da lo mismo que semaforo.py
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import paneles as T
    import semaforo as S

    P = T.cargar("m5_NVDA")
    filas = []
    for i, f in enumerate(P["fechas"]):
        idx = pd.Timestamp(f) + pd.Timedelta(minutes=570) + pd.to_timedelta(np.arange(78) * 5, "min")
        filas.append(pd.DataFrame({"open": P["O"][i], "high": P["H"][i], "low": P["L"][i],
                                   "close": P["C"][i], "volume": P["V"][i]}, index=idx))
    df = pd.concat(filas)
    a = semaforo(df)["estado"].to_numpy().reshape(len(P["fechas"]), 78)
    b = S.estado(S.matrices(P))
    print("velas iguales:", (a == b).mean().round(4), "de", a.size)
    print(semaforo(df).loc["2026-11-05"].iloc[4:14].round(3).to_string())
