"""Rasgos de Noise Band en cada control (10:00, 10:30, …, 15:30), solo con lo que se sabe
en ese momento.

Para el día d y el control c (vela k = 29 + 30·c, que cierra a la hora del control):

- ``hora``: c (0 = 10:00, 11 = 15:30).
- ``fuerza_l`` / ``fuerza_c``: cuánto ha roto el cono, en unidades de sigma: (precio / vértice
  - 1) / sigma - 1 por arriba (y su espejo por abajo); > 0 = fuera del cono.
- ``mov``: (precio / apertura - 1) / sigma, con signo.
- ``hueco``: (apertura / cierre anterior - 1) / movimiento típico del día (sigma al cierre).
- ``sigma_rel``: movimiento típico de 14 sesiones / el de 60 (¿se está abriendo el cono?).
- ``vol5``, ``vol14``, ``vol60`` y ``vol5_60``: volatilidad de los cierres diarios anteriores
  (anualizada) y su cociente.
- ``dd252``: caída del índice desde su máximo de 252 sesiones (cierre anterior).
- ``tend50``, ``tend200``: cierre anterior / SMA50 y SMA200 - 1.
- ``ret1``, ``ret1_dia``: rentabilidad del día anterior, cierre a cierre y apertura a cierre.
- ``vrel``: volumen (cotizaciones) acumulado desde la apertura / el mismo a esa hora en las 14
  sesiones anteriores; ``vrel30``: ídem de la última media hora.
- ``vwap``: (precio - VWAP) / precio / sigma, con signo.
- ``rango``: (máximo - mínimo desde la apertura) / apertura / sigma.
- ``volmin30``: volatilidad de los minutos de la última media hora / la misma franja en las 14
  sesiones anteriores.
- Calendario: ``dia_semana`` (0 = lunes), ``vencimiento`` (tercer viernes del mes),
  ``fin_mes`` (última sesión del mes), ``inicio_mes`` (primera).

Los rasgos con signo se orientan a cada lado en ``eventos.py`` y ``ronda2.py`` (× +1 largo,
× -1 corto). Salida: ``data/cache/noise_band/rasgos.npz`` (cada rasgo, días × 12).

Uso: python research/noise_band/rasgos.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import motor as nb

SALIDA = nb.DATOS.with_name("rasgos.npz")
KS = np.arange(29, 360, 30)  # velas de los controles
CON_SIGNO = ("mov", "hueco", "tend50", "tend200", "ret1", "ret1_dia", "vwap")


def _previo(x: pd.Series | np.ndarray, n: int, f: str = "mean") -> np.ndarray:
    """Estadístico de las n sesiones anteriores (sin el día actual)."""
    return getattr(pd.DataFrame(np.asarray(x)).rolling(n, min_periods=n), f)().shift(1).to_numpy()


def construir(d: dict | None = None) -> dict[str, np.ndarray]:
    d = nb.cargar() if d is None else d
    V = d["V"]
    O, H, L, C = d["O"], d["H"], d["L"], d["C"]
    D = len(d["fechas"])
    abre = O[:, 0]
    sig = d["sigma"][:, KS]
    px = C[:, KS]
    R = {"hora": np.tile(np.arange(len(KS), dtype=float), (D, 1))}
    with np.errstate(invalid="ignore", divide="ignore"):
        R["fuerza_l"] = (px / d["ref_arriba"][:, None] - 1) / sig - 1
        R["fuerza_c"] = (1 - px / d["ref_abajo"][:, None]) / sig - 1
        R["mov"] = (px / abre[:, None] - 1) / sig
        cierre = d["cierre"]
        previo = np.r_[np.nan, cierre[:-1]]
        sig_cierre = d["sigma"][:, -1]
        R["hueco"] = np.repeat(((abre / previo - 1) / sig_cierre)[:, None], len(KS), 1)
        mov_cierre = np.abs(cierre / abre - 1)
        R["sigma_rel"] = np.repeat((_previo(mov_cierre, 14) / _previo(mov_cierre, 60))[:, 0][:, None], len(KS), 1)
        r = pd.Series(cierre).pct_change().to_numpy()
        for n in (5, 14, 60):
            R[f"vol{n}"] = np.repeat((_previo(r, n, "std")[:, 0] * np.sqrt(252))[:, None], len(KS), 1)
        R["vol5_60"] = R["vol5"] / R["vol60"]
        maximo = pd.Series(cierre).rolling(252, min_periods=60).max().shift(1).to_numpy()
        R["dd252"] = np.repeat((previo / maximo - 1)[:, None], len(KS), 1)
        for n in (50, 200):
            media = pd.Series(cierre).rolling(n, min_periods=n).mean().shift(1).to_numpy()
            R[f"tend{n}"] = np.repeat((previo / media - 1)[:, None], len(KS), 1)
        R["ret1"] = np.repeat(np.r_[np.nan, r[:-1]][:, None], len(KS), 1)
        dia = cierre / abre - 1
        R["ret1_dia"] = np.repeat(np.r_[np.nan, dia[:-1]][:, None], len(KS), 1)
        acum = np.cumsum(V, axis=1)[:, KS]
        R["vrel"] = acum / _previo(acum, 14)
        v30 = acum - np.cumsum(V, axis=1)[:, KS - 29] + V[:, KS - 29]
        R["vrel30"] = v30 / _previo(v30, 14)
        R["vwap"] = (px - d["vw"][:, KS]) / px / sig
        hmax = np.maximum.accumulate(H, axis=1)[:, KS]
        lmin = np.minimum.accumulate(L, axis=1)[:, KS]
        R["rango"] = (hmax - lmin) / abre[:, None] / sig
        rmin = np.diff(np.log(C), axis=1, prepend=np.log(abre)[:, None])
        v_ult = np.stack([rmin[:, k - 29:k + 1].std(1) for k in KS], 1)
        R["volmin30"] = v_ult / _previo(v_ult, 14)
    f = d["fechas"]
    R["dia_semana"] = np.repeat(np.asarray(f.dayofweek, dtype=float)[:, None], len(KS), 1)
    tercer_viernes = (f.dayofweek == 4) & (f.day >= 15) & (f.day <= 21)
    R["vencimiento"] = np.repeat(np.asarray(tercer_viernes, dtype=float)[:, None], len(KS), 1)
    mes = f.to_period("M")
    # el último día de los datos es fin de mes solo si el siguiente día hábil cambia de mes
    ultimo = np.r_[mes[1:] != mes[:-1], (f[-1] + pd.offsets.BDay(1)).month != f[-1].month]
    primero = np.r_[True, mes[1:] != mes[:-1]]
    R["fin_mes"] = np.repeat(ultimo.astype(float)[:, None], len(KS), 1)
    R["inicio_mes"] = np.repeat(primero.astype(float)[:, None], len(KS), 1)
    return R


def main() -> None:
    R = construir()
    np.savez(SALIDA, **R)
    print(f"{len(R)} rasgos, {R['hora'].shape[0]} días × {R['hora'].shape[1]} controles → {SALIDA}")
    for k, v in R.items():
        print(f"  {k:12s} cobertura {np.mean(~np.isnan(v)):.1%}  mediana {np.nanmedian(v):.3f}")


if __name__ == "__main__":
    main()
