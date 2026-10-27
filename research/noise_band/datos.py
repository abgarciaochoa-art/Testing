"""Datos intradía del S&P 500 para Noise Band (velas de 1 minuto, sesión regular).

Fuente: el repositorio público ``FutureSharks/financial-data`` (se clona solo con las
carpetas necesarias en ``data/cache/noise_band/financial-data``):

- **OANDA SPX500_USD**: CFD sobre el S&P 500, velas de 1 minuto en UTC (precio medio entre
  compra y venta), enero 2005 → mayo 2020. Es la serie principal.
- **HistData SPXUSD**: 1 minuto en hora de Nueva York, 2010-2018. Solo para contrastar.
- **OANDA NAS100_USD y US2000_USD** (Nasdaq-100 y Russell 2000), mismo formato y periodo, para
  probar la estrategia en otros índices de EE. UU. (``rth_NAS100_USD.npz``, ``rth_US2000_USD.npz``).

Se pasa a hora de Nueva York y se queda la sesión regular: 390 velas por día, de 9:30 a
15:59 (la vela de las 15:59 cierra a las 16:00). Las velas sin cotización se rellenan con
el último cierre del día. Se descartan los días con menos de 200 velas reales o sin
cotización en los primeros 5 minutos; los de cierre anticipado (13:00) se conservan y
terminan en su última vela real.

Comprobaciones: rentabilidad apertura → cierre y noche contra SPY diario (``data/cache/
caidas/panel.npz``) y contra HistData.

Salida: ``data/cache/noise_band/rth.npz`` con fechas, O/H/L/C y volumen en ticks (días × 390),
velas reales, última vela de cada día y cierre del día anterior (y ``rth_histdata.npz`` con
HistData, para contrastar; sin volumen, así que su VWAP pondera igual cada minuto).

Uso: python research/noise_band/datos.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache" / "noise_band"
FUENTE = CACHE / "financial-data"
OANDA_DIR = FUENTE / "pyfinancialdata" / "data" / "currencies" / "oanda"
OANDA = OANDA_DIR / "SPX500_USD"
OTROS = {"NAS100_USD": "QQQ", "US2000_USD": None}  # otros índices de EE. UU. (y su ETF para contrastar)
HISTDATA = FUENTE / "pyfinancialdata" / "data" / "stocks" / "histdata" / "SPXUSD"
SALIDA = CACHE / "rth.npz"
SALIDA_HISTDATA = CACHE / "rth_histdata.npz"  # HistData 2010-2018, para contrastar
VELAS = 390  # 9:30 → 15:59


def descargar() -> None:
    if OANDA.exists() and HISTDATA.exists():
        return
    CACHE.mkdir(parents=True, exist_ok=True)
    if not FUENTE.exists():
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                        "https://github.com/FutureSharks/financial-data", str(FUENTE)], check=True)
    subprocess.run(["git", "-C", str(FUENTE), "sparse-checkout", "set",
                    "pyfinancialdata/data/currencies/oanda/SPX500_USD",
                    *[f"pyfinancialdata/data/currencies/oanda/{k}" for k in OTROS],
                    "pyfinancialdata/data/stocks/histdata/SPXUSD"], check=True)


def a_sesion(t: pd.Series, o, h, l, c, v=None) -> dict:  # noqa: E741
    """Velas con hora de Nueva York → matrices días × 390 de la sesión regular."""
    m = (t.dt.hour * 60 + t.dt.minute - (9 * 60 + 30)).to_numpy()
    ok = (m >= 0) & (m < VELAS)
    dia = t.dt.normalize().dt.tz_localize(None).to_numpy()[ok]
    m = m[ok]
    fechas, fila = np.unique(dia, return_inverse=True)
    D = len(fechas)
    out = {k: np.full((D, VELAS), np.nan) for k in ("O", "H", "L", "C")}
    for k, serie in (("O", o), ("H", h), ("L", l), ("C", c)):
        out[k][fila, m] = np.asarray(serie, dtype=float)[ok]
    V = np.zeros((D, VELAS))
    if v is not None:
        V[fila, m] = np.asarray(v, dtype=float)[ok]
    real = ~np.isnan(out["C"])
    # rellenar velas vacías con el último cierre del día (y antes de la primera, con su apertura)
    C = pd.DataFrame(out["C"]).ffill(axis=1).to_numpy()
    primera = np.argmax(real, axis=1)
    abre = out["O"][np.arange(D), primera]
    C = np.where(np.isnan(C), abre[:, None], C)
    for k in ("O", "H", "L"):
        out[k] = np.where(real, out[k], C)
    out["C"] = C
    ultimo = VELAS - 1 - np.argmax(real[:, ::-1], axis=1)
    validos = (real.sum(1) >= 200) & (primera < 5)
    return dict(fechas=fechas[validos], O=out["O"][validos], H=out["H"][validos], L=out["L"][validos],
                C=out["C"][validos], V=V[validos], real=real[validos], ultimo=ultimo[validos])


def ruta_panel(instrumento: str) -> Path:
    return SALIDA if instrumento == "SPX500_USD" else CACHE / f"rth_{instrumento}.npz"


def cargar_oanda(instrumento: str = "SPX500_USD") -> dict:
    partes = [pd.read_csv(f) for f in sorted((OANDA_DIR / instrumento).glob("*/*.csv"))]
    d = pd.concat(partes, ignore_index=True).drop_duplicates("time").sort_values("time")
    t = pd.to_datetime(d.time).dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    return a_sesion(t, d.open, d.high, d.low, d.close, d.volume)


def cargar_histdata() -> dict:
    partes = [pd.read_csv(f, sep=";", header=None, names=["t", "o", "h", "l", "c", "v"])
              for f in sorted(HISTDATA.glob("*.csv"))]
    d = pd.concat(partes, ignore_index=True).drop_duplicates("t").sort_values("t")
    t = pd.to_datetime(d.t, format="%Y%m%d %H%M%S").dt.tz_localize("America/New_York", ambiguous="NaT",
                                                                    nonexistent="NaT")
    d, t = d[t.notna().to_numpy()], t[t.notna()]
    return a_sesion(t, d.o, d.h, d.l, d.c)


def contrastar(x: dict, etf: str = "SPY") -> pd.DataFrame:
    """Rentabilidad apertura → cierre y de la noche frente al ETF diario."""
    import sys
    sys.path.insert(0, str(ROOT / "research" / "caidas"))
    import motor as mc

    p = mc.cargar()
    j = p.col[etf]
    spy = pd.DataFrame({"o": p.open[:, j], "c": p.close[:, j]}, index=pd.DatetimeIndex(p.fechas))
    D = len(x["fechas"])
    cierre = x["C"][np.arange(D), x["ultimo"]]
    cfd = pd.DataFrame({"o": x["O"][:, 0], "c": cierre}, index=pd.DatetimeIndex(x["fechas"]))
    z = cfd.join(spy, rsuffix="_spy", how="inner")
    z["dia"], z["dia_spy"] = z.c / z.o - 1, z.c_spy / z.o_spy - 1
    z["noche"], z["noche_spy"] = z.o / z.c.shift() - 1, z.o_spy / z.c_spy.shift() - 1
    return z.dropna()


def main() -> None:
    descargar()
    x = cargar_oanda()
    D = len(x["fechas"])
    cierre = x["C"][np.arange(D), x["ultimo"]]
    previo = np.r_[np.nan, cierre[:-1]]
    np.savez(SALIDA, fechas=x["fechas"], O=x["O"], H=x["H"], L=x["L"], C=x["C"], V=x["V"], real=x["real"],
             ultimo=x["ultimo"], previo=previo)
    anticipado = int((x["ultimo"] < 300).sum())
    print(f"OANDA: {D} sesiones de {pd.Timestamp(x['fechas'][0]).date()} a {pd.Timestamp(x['fechas'][-1]).date()}, "
          f"{x['real'].mean():.1%} de velas con cotización, {anticipado} cierres anticipados")
    z = contrastar(x)
    print(f"frente a SPY ({len(z)} días comunes): correlación apertura→cierre {z.dia.corr(z.dia_spy):.4f}, "
          f"noche {z.noche.corr(z.noche_spy):.4f}; diferencia media apertura→cierre "
          f"{(z.dia - z.dia_spy).mean() * 1e4:+.2f} pb (desv. {(z.dia - z.dia_spy).std() * 1e4:.1f} pb)")
    peores = (z.dia - z.dia_spy).abs().sort_values(ascending=False).head(5)
    print("días con más diferencia:", [(str(k.date()), round(v * 100, 2)) for k, v in peores.items()])
    h = cargar_histdata()
    Dh = len(h["fechas"])
    ch_cierre = h["C"][np.arange(Dh), h["ultimo"]]
    np.savez(SALIDA_HISTDATA, fechas=h["fechas"], O=h["O"], H=h["H"], L=h["L"], C=h["C"], V=h["V"],
             real=h["real"], ultimo=h["ultimo"], previo=np.r_[np.nan, ch_cierre[:-1]])
    ch = pd.Series(h["C"][np.arange(Dh), h["ultimo"]] / h["O"][:, 0] - 1, pd.DatetimeIndex(h["fechas"]))
    co = pd.Series(cierre / x["O"][:, 0] - 1, pd.DatetimeIndex(x["fechas"]))
    comun = ch.index.intersection(co.index)
    print(f"frente a HistData ({len(comun)} días comunes): correlación apertura→cierre "
          f"{ch[comun].corr(co[comun]):.4f}")
    for inst, etf in OTROS.items():
        y = cargar_oanda(inst)
        Dy = len(y["fechas"])
        cy = y["C"][np.arange(Dy), y["ultimo"]]
        np.savez(ruta_panel(inst), fechas=y["fechas"], O=y["O"], H=y["H"], L=y["L"], C=y["C"], V=y["V"],
                 real=y["real"], ultimo=y["ultimo"], previo=np.r_[np.nan, cy[:-1]])
        texto = f"{inst}: {Dy} sesiones, {y['real'].mean():.1%} de velas con cotización"
        if etf is not None:
            zz = contrastar(y, etf)
            texto += f"; frente a {etf}: correlación apertura→cierre {zz.dia.corr(zz.dia_spy):.4f}"
        dia_y = pd.Series(cy / y["O"][:, 0] - 1, pd.DatetimeIndex(y["fechas"]))
        com = dia_y.index.intersection(co.index)
        print(texto + f"; con el S&P 500 dentro del día: {dia_y[com].corr(co[com]):.3f}")
    # forma del "cono": movimiento medio absoluto desde la apertura, por minuto del día
    mov = np.abs(x["C"] / x["O"][:, [0]] - 1).mean(0)
    print("movimiento medio desde la apertura (%):",
          {f"{9 + (30 + k) // 60}:{(30 + k) % 60:02d}": round(mov[k] * 100, 3) for k in (0, 29, 89, 179, 299, 389)})


if __name__ == "__main__":
    main()
