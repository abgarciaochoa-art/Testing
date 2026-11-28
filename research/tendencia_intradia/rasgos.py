"""Rasgos de tendencia intradía calculados en una hora de control (todo con información hasta
esa hora) y objetivos (lo que pasa después).

Todos los rasgos con dirección son positivos si apuntan al alza. Los precios se miden en ATR
de 14 días (conocido antes de abrir) para que sean comparables entre valores y épocas.

Rasgos:
  ret_ap      movimiento desde la apertura                     (P - apertura) / ATR
  vwap        distancia a la VWAP del día                       (P - VWAP) / ATR
  frac_vwap   fracción de velas cerradas sobre la VWAP (−1..1)
  orb         posición frente al rango de apertura (30 min, o la 1.ª vela si es mayor):
              (P - centro) / medio rango, recortado a ±3
  ruido       movimiento desde la apertura en sigmas: frente al movimiento medio absoluto a
              esa hora en los 14 días previos (la "noise area" de Zarattini et al.)
  eficiencia  (P - apertura) / suma de |cambios| desde la apertura (−1..1): limpieza del camino
  regresion   pendiente de la recta de precios desde la apertura × R² (signo de la pendiente)
  pos_rango   dónde está el precio dentro del rango del día hasta ahora (−1 mínimo, +1 máximo)
  vol_neto    (volumen de velas alcistas − bajistas) / volumen total desde la apertura
  ema         pendiente de una media exponencial de 60 minutos en los últimos 30 / ATR
  di          (DI+ − DI−)/(DI+ + DI−) de Wilder (14 velas) en el día; solo velas ≤ 5 min
  primeros30  movimiento de la primera media hora (o primera vela) / ATR
  hueco       (apertura − cierre previo) / ATR
  ayer        movimiento de ayer (cierre a cierre) / ATR
  tendencia_d (cierre previo − media de 20) / ATR
  mercado     ret_ap del SPY en la misma hora (en ATR del SPY)
  relativa    ret_ap − mercado
Sin dirección (para interacciones):
  adx         ADX de Wilder (14 velas) en el día; solo velas ≤ 5 min
  rvol        volumen acumulado / media de los 20 días previos a la misma hora
  rango       (máximo − mínimo del día hasta ahora) / ATR

Objetivos:
  y_cierre    (cierre − P) / ATR
  y_60        (precio 60 min después − P) / ATR
  dia         tipo de día completo: +1 si (cierre − apertura) ≥ 0,6 × rango, −1 si ≤ −0,6, 0 si no
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DIRECCIONALES = ["ret_ap", "vwap", "frac_vwap", "orb", "ruido", "eficiencia", "regresion",
                 "pos_rango", "vol_neto", "ema", "di", "primeros30", "hueco", "ayer",
                 "tendencia_d", "mercado", "relativa"]
SIN_DIRECCION = ["adx", "rvol", "rango"]


def _ema_dia(C: np.ndarray, n: int) -> np.ndarray:
    a = 2 / (n + 1)
    E = np.empty_like(C)
    E[:, 0] = C[:, 0]
    for k in range(1, C.shape[1]):
        E[:, k] = a * C[:, k] + (1 - a) * E[:, k - 1]
    return E


def _wilder(P: dict, n: int = 14) -> tuple[np.ndarray, np.ndarray]:
    """DI y ADX de Wilder reiniciados cada día (velas del día solamente)."""
    H, L, C = P["H"], P["L"], P["C"]
    D, N = C.shape
    prevC = np.concatenate([P["O"][:, :1], C[:, :-1]], 1)
    tr = np.maximum(H, prevC) - np.minimum(L, prevC)
    up = np.diff(H, axis=1, prepend=H[:, :1])
    dn = -np.diff(L, axis=1, prepend=L[:, :1])
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    a = 1 / n
    s_tr, s_p, s_m = (np.zeros((D, N)) for _ in range(3))
    s_tr[:, 0], s_p[:, 0], s_m[:, 0] = tr[:, 0], pdm[:, 0], mdm[:, 0]
    for k in range(1, N):
        s_tr[:, k] = (1 - a) * s_tr[:, k - 1] + a * tr[:, k]
        s_p[:, k] = (1 - a) * s_p[:, k - 1] + a * pdm[:, k]
        s_m[:, k] = (1 - a) * s_m[:, k - 1] + a * mdm[:, k]
    with np.errstate(invalid="ignore", divide="ignore"):
        pdi, mdi = s_p / s_tr, s_m / s_tr
        di = (pdi - mdi) / (pdi + mdi)
        dx = np.abs(di)
    adx = np.zeros((D, N))
    adx[:, 0] = np.nan_to_num(dx[:, 0])
    for k in range(1, N):
        adx[:, k] = (1 - a) * adx[:, k - 1] + a * np.nan_to_num(dx[:, k])
    return np.nan_to_num(di), adx


def precalcular(P: dict) -> dict:
    """Matrices días × velas que se reutilizan en todas las horas de control."""
    O, H, L, C, V = (P[k] for k in "OHLCV")
    paso = int(P["paso"])
    tp = (H + L + C) / 3
    cv = np.cumsum(V, 1)
    vwap = np.where(cv > 0, np.cumsum(tp * V, 1) / np.where(cv > 0, cv, 1),
                    np.cumsum(tp, 1) / np.arange(1, C.shape[1] + 1))
    camino = np.abs(np.diff(np.concatenate([O[:, :1], C], 1), axis=1)).cumsum(1)
    signo_v = np.sign(C - O)
    Q = {"vwap": vwap, "cv": cv, "camino": camino,
         "sobre": np.cumsum(np.sign(C - vwap), 1) / np.arange(1, C.shape[1] + 1),
         "vneto": np.cumsum(signo_v * V, 1) / np.where(cv > 0, cv, 1),
         "Hmax": np.maximum.accumulate(H, 1), "Lmin": np.minimum.accumulate(L, 1),
         "ema": _ema_dia(C, max(2, 60 // paso))}
    if paso <= 5:
        Q["di"], Q["adx"] = _wilder(P)
    return Q


def en_hora(P: dict, cp: int, Q: dict | None = None, mercado: pd.Series | None = None) -> pd.DataFrame:
    """Rasgos y objetivos de cada día a ``cp`` minutos de la apertura (velas cerradas)."""
    Q = Q or precalcular(P)
    O, H, L, C = (P[k] for k in "OHLC")
    D, N = C.shape
    paso = int(P["paso"])
    j = cp // paso - 1
    if j < 0 or j >= N - 1:
        raise ValueError("hora de control fuera de la sesión")
    atr = P["atr"]
    o0, pt = O[:, 0], C[:, j]
    f = {}
    f["ret_ap"] = (pt - o0) / atr
    f["vwap"] = (pt - Q["vwap"][:, j]) / atr
    f["frac_vwap"] = Q["sobre"][:, j]
    nor = max(1, 30 // paso)
    if j >= nor:
        hi, lo = H[:, :nor].max(1), L[:, :nor].min(1)
        f["orb"] = np.clip((pt - (hi + lo) / 2) / np.maximum((hi - lo) / 2, 1e-9), -3, 3)
        f["primeros30"] = (C[:, nor - 1] - o0) / atr
    else:
        f["orb"] = np.full(D, np.nan)
        f["primeros30"] = np.full(D, np.nan)
    mov = np.abs(pt / o0 - 1)
    media14 = pd.Series(mov).rolling(14).mean().shift().to_numpy()
    f["ruido"] = (pt / o0 - 1) / media14
    f["eficiencia"] = (pt - o0) / np.maximum(Q["camino"][:, j], 1e-12)
    if j >= 2:
        k = np.arange(j + 1)
        y = C[:, : j + 1]
        km, ym = k.mean(), y.mean(1, keepdims=True)
        sxy = ((k - km) * (y - ym)).sum(1)
        sxx = ((k - km) ** 2).sum()
        syy = ((y - ym) ** 2).sum(1)
        r2 = np.where(syy > 0, sxy**2 / (sxx * np.where(syy > 0, syy, 1)), 0)
        f["regresion"] = np.sign(sxy) * r2
    else:
        f["regresion"] = np.full(D, np.nan)
    rng = Q["Hmax"][:, j] - Q["Lmin"][:, j]
    f["pos_rango"] = np.where(rng > 0, 2 * (pt - Q["Lmin"][:, j]) / np.where(rng > 0, rng, 1) - 1, 0)
    f["vol_neto"] = Q["vneto"][:, j]
    k30 = max(1, 30 // paso)
    f["ema"] = (Q["ema"][:, j] - Q["ema"][:, max(0, j - k30)]) / atr if paso <= 30 else np.full(D, np.nan)
    f["di"] = Q["di"][:, j] if "di" in Q else np.full(D, np.nan)
    f["adx"] = Q["adx"][:, j] if "adx" in Q else np.full(D, np.nan)
    f["hueco"] = (o0 - P["prev"]) / atr
    prev2 = np.concatenate([[np.nan], P["prev"][:-1]])
    f["ayer"] = (P["prev"] - prev2) / atr
    f["tendencia_d"] = (P["prev"] - P["sma20"]) / atr
    base = Q["cv"][:, j]
    rv = pd.Series(base).rolling(20).mean().shift().to_numpy()
    f["rvol"] = np.where(rv > 0, base / np.where(rv > 0, rv, 1), np.nan)
    f["rango"] = rng / atr
    # objetivos
    f["y_cierre"] = (C[:, -1] - pt) / atr
    k60 = j + max(1, 60 // paso)
    f["y_60"] = (C[:, k60] - pt) / atr if k60 <= N - 1 else np.full(D, np.nan)
    cuerpo = (C[:, -1] - o0) / np.maximum(H.max(1) - L.min(1), 1e-12)
    f["dia"] = np.where(cuerpo >= 0.6, 1, np.where(cuerpo <= -0.6, -1, 0))
    df = pd.DataFrame(f, index=pd.to_datetime(P["fechas"]))
    if mercado is not None:
        df["mercado"] = mercado.reindex(df.index).to_numpy()
        df["relativa"] = df["ret_ap"] - df["mercado"]
    else:
        df["mercado"] = np.nan
        df["relativa"] = np.nan
    df["cp"] = cp
    ok = np.isfinite(atr) & (atr > 0)
    return df[ok]
