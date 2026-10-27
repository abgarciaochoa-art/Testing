"""Utilidades estadísticas de Noise Band: tramos de fechas, t de Newey-West, diferencia de
Sharpe y grupos de pruebas correladas (número efectivo de pruebas)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def tramo(s: pd.Series, a, b) -> pd.Series:
    x = s[s.index >= pd.Timestamp(a)]
    return x[x.index < pd.Timestamp(b)] if b is not None else x


def _largo_plazo(x: np.ndarray, lags: int) -> np.ndarray:
    """Covarianza de largo plazo (Newey-West, pesos de Bartlett) de las columnas de ``x``."""
    x = x - x.mean(0)
    n = len(x)
    psi = x.T @ x / n
    for k in range(1, lags + 1):
        g = x[k:].T @ x[:-k] / n
        psi += (1 - k / (lags + 1)) * (g + g.T)
    return psi


def nw_t(x: np.ndarray, lags: int = 10) -> float:
    """t de la media de ``x`` con errores de Newey-West."""
    x = np.asarray(x, dtype=float)
    v = float(_largo_plazo(x[:, None], lags)[0, 0])
    return float(x.mean() / np.sqrt(v / len(x))) if v > 0 else np.nan


def t_sharpe(a: np.ndarray, b: np.ndarray, lags: int = 10) -> float:
    """t de la diferencia de Sharpe entre dos series del mismo calendario (Ledoit y Wolf,
    2008: método delta con covarianza de Newey-West). > 0 si ``a`` tiene más Sharpe."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    v = np.column_stack([a, b, a ** 2, b ** 2])
    m = v.mean(0)
    s1, s2 = m[2] - m[0] ** 2, m[3] - m[1] ** 2
    if s1 <= 0 or s2 <= 0:
        return np.nan
    dif = m[0] / np.sqrt(s1) - m[1] / np.sqrt(s2)
    g = np.array([m[2] / s1 ** 1.5, -m[3] / s2 ** 1.5, -m[0] / (2 * s1 ** 1.5), m[1] / (2 * s2 ** 1.5)])
    var = float(g @ _largo_plazo(v, lags) @ g) / len(a)
    return float(dif / np.sqrt(var)) if var > 0 else np.nan


def grupos_correlados(rets: np.ndarray, umbral: float = 0.9) -> np.ndarray:
    """Grupo de cada prueba (columnas de ``rets``): se unen las que tienen una correlación
    diaria > ``umbral``, por cierre transitivo. Es la regla del repo para el número efectivo
    de pruebas (``n_effective_trials``) y la del estudio de caídas (``dsr_efectivo.py``)."""
    unido = csr_matrix(np.corrcoef(rets, rowvar=False) > umbral)
    return connected_components(unido, directed=False)[1]
