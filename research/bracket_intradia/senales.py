"""Señales de entrada (largos) y rasgos intradía. Todo se calcula con información hasta el
cierre de cada vela (la entrada es a la apertura de la siguiente)."""

from __future__ import annotations

import numpy as np


def _rolling_max(X: np.ndarray, w: int) -> np.ndarray:
    """Máximo de las w velas ANTERIORES (sin incluir la actual)."""
    D, N = X.shape
    out = np.full((D, N), np.nan)
    for j in range(1, N):
        out[:, j] = X[:, max(0, j - w):j].max(1)
    return out


def _rolling_min(X: np.ndarray, w: int) -> np.ndarray:
    return -_rolling_max(-X, w)


def rasgos(P: dict) -> dict:
    O, H, L, C, V = (P[k] for k in "OHLCV")
    D, N = C.shape
    paso = int(P["paso"])
    atr = P["atr"][:, None]
    tipico = (H + L + C) / 3
    cv = np.cumsum(V, 1)
    vwap = np.where(cv > 0, np.cumsum(tipico * V, 1) / np.where(cv > 0, cv, 1),
                    np.cumsum(tipico, 1) / np.arange(1, N + 1))
    apertura = O[:, :1]
    # volumen relativo: acumulado hasta la vela frente a la media de los 20 días previos
    rvol = np.full((D, N), np.nan)
    for d in range(20, D):
        base = cv[d - 20:d].mean(0)
        rvol[d] = np.where(base > 0, cv[d] / np.where(base > 0, base, 1), np.nan)
    def m(minutos: int) -> int:
        return max(1, minutos // paso)
    ret15 = (C - np.roll(C, m(15), 1)) / atr
    ret15[:, : m(15)] = np.nan
    return {
        "vwap": vwap, "apertura": apertura, "rvol": rvol,
        "desde_apertura": (C - apertura) / atr,          # en ATR
        "dist_vwap": (C - vwap) / atr,
        "ret15": ret15,
        "max30": _rolling_max(H, m(30)), "min30": _rolling_min(L, m(30)),
        "tendencia_diaria": ((P["prev"] > P["sma20"]) & (P["sma20"] > P["sma50"]))[:, None]
        & np.ones((1, N), bool),
        "bajista_diaria": ((P["prev"] < P["sma20"]) & (P["sma20"] < P["sma50"]))[:, None]
        & np.ones((1, N), bool),
        "hueco": ((O[:, 0] - P["prev"]) / P["atr"])[:, None] * np.ones((1, N)),
        "minuto": np.arange(N)[None, :] * paso * np.ones((D, 1)),
    }


def senales(P: dict, F: dict | None = None) -> dict[str, np.ndarray]:
    F = F or rasgos(P)
    H, L, C = P["H"], P["L"], P["C"]
    D, N = C.shape
    paso = int(P["paso"])
    minuto = F["minuto"]
    sobre_vwap = C > F["vwap"]
    sobre_ap = C > F["apertura"]
    td = F["tendencia_diaria"]
    out: dict[str, np.ndarray] = {}
    out["siempre"] = np.ones((D, N), bool)
    out["tendencia_diaria"] = td.copy()
    out["sobre_vwap"] = sobre_vwap
    out["sobre_apertura"] = sobre_ap
    out["vwap+apertura"] = sobre_vwap & sobre_ap
    out["tendencia_total"] = td & sobre_vwap & sobre_ap
    out["dia_fuerte"] = F["desde_apertura"] > 0.25
    out["dia_debil_compra"] = F["desde_apertura"] < -0.25         # comprar caídas (reversión)
    out["bajo_vwap_compra"] = F["dist_vwap"] < -0.15                # comprar lejos bajo la VWAP
    for k in (5, 15, 30):
        nk = max(1, k // paso)
        orh = H[:, :nk].max(1, keepdims=True)
        s = (C > orh) & (minuto >= k)
        out[f"orb{k}"] = s
    out["nuevo_max30"] = (C > F["max30"]) & (minuto >= 30)
    out["nuevo_max30+vwap"] = out["nuevo_max30"] & sobre_vwap
    # retroceso a la VWAP en tendencia y rebote: el mínimo de las últimas 5 velas tocó la
    # VWAP (a menos de 0,05 ATR) y la vela cierra por encima del máximo de la anterior
    toco = np.zeros((D, N), bool)
    cerca = (L - F["vwap"]) / P["atr"][:, None] < 0.05
    for j in range(N):
        toco[:, j] = cerca[:, max(0, j - max(1, 5 // paso) + 1):j + 1].any(1)
    rebote = C > np.roll(H, 1, 1)
    out["retroceso_vwap"] = td & sobre_vwap & toco & rebote & (minuto >= 15)
    out["retroceso_vwap_sin_td"] = sobre_vwap & toco & rebote & (minuto >= 15)
    rv = np.nan_to_num(F["rvol"]) > 1.3
    out["vwap+apertura+rvol"] = out["vwap+apertura"] & rv
    out["orb5+rvol"] = out["orb5"] & rv
    out["hueco_alcista"] = (F["hueco"] > 0.1) & sobre_vwap
    return out


def espejo(P: dict) -> dict:
    """Serie invertida (−precio): las señales de compra calculadas sobre ella son las de
    venta en corto sobre la original (por encima de la VWAP ↔ por debajo, etc.)."""
    Q = dict(P)
    Q["O"], Q["H"], Q["L"], Q["C"] = -P["O"], -P["L"], -P["H"], -P["C"]
    for k in ("prev", "sma20", "sma50"):
        Q[k] = -P[k]
    Q["ret5"] = -P["ret5"]
    return Q


VENTANAS = {"9:30-10:30": (0, 60), "10:30-12:00": (60, 150), "12:00-14:00": (150, 270),
            "14:00-15:55": (270, 390)}


def ventana(P: dict, nombre: str) -> np.ndarray:
    a, b = VENTANAS[nombre]
    N = P["C"].shape[1]
    minuto = np.arange(N) * int(P["paso"])
    return ((minuto >= a) & (minuto < b))[None, :]
