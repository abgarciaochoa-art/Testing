"""Rasgos point-in-time para entender POR QUÉ cae una acción (módulos de mejora).

Todo se calcula una vez sobre el panel completo y se guarda en
``data/cache/caidas/rasgos.npz`` (matrices fecha×ticker, float32). Cada valor en ``t``
usa solo información disponible al cierre de ``t``.

Grupos de pares ("sector estadístico")
--------------------------------------
No hay sectores GICS históricos para las empresas que salieron del índice, y usar los
de hoy introduciría sesgo de supervivencia. En su lugar, al cierre de cada mes se
calcula la correlación de retornos diarios de los últimos 252 días entre los miembros
de ese momento, y cada acción recibe sus **10 pares más correlacionados** (que se usan
durante todo el mes siguiente). El par nº 1 es su "gemela" (KO↔PEP, HD↔LOW, V↔MA…).

Rasgos (N = 5 y 10 sesiones salvo indicación)
---------------------------------------------
- ``r{N}``: retorno propio de N sesiones; ``spy{N}``: el del SPY.
- ``pares{N}``: retorno medio de los 10 pares; ``rel_pares{N}`` = r - pares.
- ``beta``: beta a SPY (252 d); ``idio{N}`` = r - beta·spy (caída propia).
- ``gem_corr``: correlación con la gemela; ``gem{N}``: retorno de la gemela;
  ``gem_z``: z-score (60 d) del log-cociente de precios acción/gemela.
- ``vol_rel``: volumen de hoy / media de 20 d; ``vol_rel5``: media 5 d / media 60 d.
- ``gap{N}``: parte de la caída de N días que se produjo en aperturas (noticias
  nocturnas) frente al total; ``noche1``: retorno de la última noche.
- ``dias_res``: sesiones desde la última publicación de resultados conocida
  (``consenso_master.parquet``); NaN si no hay dato.
- ``sorpresa``: sorpresa de BPA (%) de esa última publicación.
- ``volreal_spy``: volatilidad realizada anualizada del SPY a 20 d (régimen).
- ``frac_caen``: fracción de miembros con r5 <= -5 % (amplitud de la caída).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

import motor as m

CACHE = m.ROOT / "data" / "cache" / "caidas"
SALIDA = CACHE / "rasgos.npz"
N_PARES = 10


def _ret(c: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(c, np.nan)
    out[n:] = c[n:] / c[:-n] - 1
    return out


def pares_mensuales(p: m.Panel) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Para cada sesión, índices de los 10 pares (T×N×10), su correlación media, la
    gemela (T×N) y la correlación con ella. -1 = sin pares."""
    T, N = p.close.shape
    r1 = np.full((T, N), np.nan)
    r1[1:] = p.close[1:] / p.close[:-1] - 1
    fechas = pd.DatetimeIndex(p.fechas)
    fin_mes = np.flatnonzero(fechas.month != np.roll(fechas.month, -1))  # último día de cada mes
    pares = np.full((T, N, N_PARES), -1, dtype=np.int32)
    gem_corr = np.full((T, N), np.nan, dtype=np.float32)
    pares_corr = np.full((T, N), np.nan, dtype=np.float32)
    beta = np.full((T, N), np.nan, dtype=np.float32)
    spy = r1[:, p.col["SPY"]]
    for a, b in zip(fin_mes[:-1], fin_mes[1:], strict=True):
        if a < 252:
            continue
        ventana = r1[a - 251 : a + 1]
        vivos = np.flatnonzero(p.miembro[a] & (np.isnan(ventana).sum(0) < 30))
        x = np.nan_to_num(ventana[:, vivos])
        x = (x - x.mean(0)) / (x.std(0) + 1e-12)
        corr = (x.T @ x) / len(x)
        np.fill_diagonal(corr, -np.inf)
        orden = np.argsort(-corr, axis=1)[:, :N_PARES]
        bloque = slice(a + 1, b + 1)  # se usan durante el mes siguiente
        pares[bloque, vivos, :] = vivos[orden][None, :, :]
        top = np.take_along_axis(corr, orden, 1)
        gem_corr[bloque, vivos] = top[:, 0]
        pares_corr[bloque, vivos] = top.mean(1)
        s = np.nan_to_num(spy[a - 251 : a + 1])
        y = np.nan_to_num(ventana[:, vivos])
        beta[bloque, vivos] = ((y - y.mean(0)) * (s - s.mean())[:, None]).mean(0) / s.var()
    return pares, gem_corr, pares_corr, beta


def media_pares(x: np.ndarray, pares: np.ndarray) -> np.ndarray:
    T, N = x.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    validos = pares[:, :, 0] >= 0
    tt = np.arange(T)[:, None, None]
    vals = x[tt, np.maximum(pares, 0)]  # T×N×10
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(np.where(pares >= 0, vals, np.nan), axis=2)
    out[validos] = mu[validos]
    return out


def gemela_z(p: m.Panel, pares: np.ndarray, ventana: int = 60) -> np.ndarray:
    """z-score del log(precio_i / precio_gemela) en 60 d, con la gemela vigente en t."""
    T, N = p.close.shape
    L = np.log(p.close_ff)
    out = np.full((T, N), np.nan, dtype=np.float32)
    gem = pares[:, :, 0]
    # bloques con la misma gemela (cambia a fin de mes)
    cambios = np.flatnonzero(np.any(gem[1:] != gem[:-1], axis=1)) + 1
    limites = np.concatenate([[0], cambios, [T]])
    for a, b in zip(limites[:-1], limites[1:], strict=True):
        g = gem[a]
        ok = np.flatnonzero(g >= 0)
        if not ok.size or a < ventana:
            continue
        filas = np.arange(a - ventana + 1, b)
        d = L[filas][:, ok] - L[filas][:, g[ok]]
        df = pd.DataFrame(d)
        z = ((df - df.rolling(ventana).mean()) / df.rolling(ventana).std()).to_numpy()
        out[a:b, ok] = z[ventana - 1 :]
    return out


def resultados(p: m.Panel) -> tuple[np.ndarray, np.ndarray]:
    """Sesiones desde la última publicación de resultados y su sorpresa (%)."""
    T, N = p.close.shape
    dias = np.full((T, N), np.nan, dtype=np.float32)
    sorpresa = np.full((T, N), np.nan, dtype=np.float32)
    c = pd.read_parquet(m.ROOT / "data/external/consenso/consenso_master.parquet",
                        columns=["ticker", "report_date", "report_time", "surprise_pct"])
    c["report_date"] = pd.to_datetime(c["report_date"])
    # publicación tras el cierre (amc) => el mercado reacciona la sesión siguiente
    amc = c["report_time"].astype(str).str.lower().str.contains("post|amc|after")
    c.loc[amc, "report_date"] += pd.Timedelta(days=1)
    fechas = p.fechas.astype("datetime64[ns]")
    base = {t.split("-")[0]: j for j, t in enumerate(p.tickers)}
    for t, g in c.groupby("ticker"):
        j = p.col.get(t, base.get(t))
        if j is None:
            continue
        idx = np.searchsorted(fechas, g["report_date"].to_numpy(dtype="datetime64[ns]"))
        ev = np.zeros(T + 1)
        s = np.full(T + 1, np.nan)
        ok = idx < T
        ev[idx[ok]] = 1
        s[idx[ok]] = g["surprise_pct"].to_numpy()[ok]
        ultimo = np.maximum.accumulate(np.where(ev[:T] > 0, np.arange(T), -1))
        tiene = ultimo >= 0
        dias[tiene, j] = (np.arange(T) - ultimo)[tiene]
        sorpresa[tiene, j] = s[ultimo[tiene]]
    return dias, sorpresa


def construir() -> None:
    t0 = time.time()
    p = m.cargar()
    c = p.close
    R = {}
    pares, gem_corr, pares_corr, beta = pares_mensuales(p)
    print(f"pares {time.time() - t0:.0f}s", flush=True)
    spy = c[:, p.col["SPY"]]
    for n in (1, 3, 5, 10, 20):
        r = _ret(c, n)
        R[f"r{n}"] = r
        s = np.full(len(c), np.nan)
        s[n:] = spy[n:] / spy[:-n] - 1
        R[f"spy{n}"] = np.repeat(s[:, None], c.shape[1], 1)
        if n in (3, 5, 10, 20):
            R[f"pares{n}"] = media_pares(r, pares)
            R[f"rel_pares{n}"] = r - R[f"pares{n}"]
            R[f"idio{n}"] = r - beta * s[:, None]
            R[f"gem{n}"] = np.take_along_axis(r, np.maximum(pares[:, :, 0], 0), 1)
            R[f"gem{n}"][pares[:, :, 0] < 0] = np.nan
    print(f"retornos {time.time() - t0:.0f}s", flush=True)
    R["beta"], R["gem_corr"], R["pares_corr"] = beta, gem_corr, pares_corr
    R["gem_z"] = gemela_z(p, pares)
    v = pd.DataFrame(p.volume)
    R["vol_rel"] = (v / v.rolling(20, min_periods=10).mean().shift(1)).to_numpy()
    R["vol_rel5"] = (v.rolling(5).mean() / v.rolling(60, min_periods=30).mean().shift(5)).to_numpy()
    noche = np.full_like(c, np.nan)
    noche[1:] = p.open[1:] / c[:-1] - 1
    noche[p.sin_apertura] = np.nan  # sin apertura real (CRSP): no hay hueco medible
    R["noche1"] = noche
    for n in (5, 10):
        ln = pd.DataFrame(np.log1p(noche)).rolling(n).sum().to_numpy()
        lt = np.log1p(_ret(c, n))
        with np.errstate(invalid="ignore", divide="ignore"):
            R[f"gap{n}"] = np.where(lt < -0.02, ln / lt, np.nan)
    R["dias_res"], R["sorpresa"] = resultados(p)
    print(f"resultados {time.time() - t0:.0f}s", flush=True)
    sr = pd.Series(spy).pct_change()
    R["volreal_spy"] = np.repeat((sr.rolling(20).std() * np.sqrt(252)).to_numpy()[:, None], c.shape[1], 1)
    with np.errstate(invalid="ignore"):
        caen = np.where(p.miembro, R["r5"] <= -0.05, np.nan)
    R["frac_caen"] = np.repeat(np.nanmean(caen, 1)[:, None], c.shape[1], 1)
    with np.errstate(invalid="ignore"):
        R["sobre_sma200"] = (c > m.sma(200)).astype(np.float32)
        R["dd20"] = c / m.maximo(20) - 1
        R["dd252"] = c / m.maximo(252) - 1
        R["mom252"] = _ret(c, 252)
    R["ibs"] = np.where(p.sin_apertura, np.nan, m.ibs())

    # ---- módulos v2 ----------------------------------------------------------------
    r1 = pd.DataFrame(_ret(c, 1))
    for n in (5, 10):
        # volatilidad diaria de 60 sesiones medida ANTES de la caída (termina en t-n)
        vol_previa = r1.rolling(60, min_periods=40).std().shift(n).to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            R[f"sigma{n}"] = R[f"r{n}"] / (vol_previa * np.sqrt(n))  # caída en desviaciones típicas
    R["vol60"] = (r1.rolling(60, min_periods=40).std().shift(10) * np.sqrt(252)).to_numpy()
    with np.errstate(invalid="ignore"):
        R["dist_sma200"] = c / m.sma(200) - 1
        R["dist_sma50"] = c / m.sma(50) - 1
    R["pares_mom252"] = media_pares(R["mom252"], pares)
    R["rel_mom252"] = R["mom252"] - R["pares_mom252"]
    with np.errstate(invalid="ignore"):
        R["pares_frac_caen"] = media_pares(np.where(np.isnan(R["r10"]), np.nan, (R["r10"] <= -0.05) * 1.0), pares)
    dv = pd.DataFrame(p.raw_close * p.volume).rolling(60, min_periods=20).mean()
    R["tam"] = dv.where(p.miembro).rank(axis=1, pct=True).to_numpy()  # tamaño/liquidez relativo (0-1)
    spy_s = pd.Series(spy)
    spy_dd = (spy_s / spy_s.rolling(252, min_periods=60).max() - 1).to_numpy()
    R["spy_dd252"] = np.repeat(spy_dd[:, None], c.shape[1], 1)
    vr = (sr.rolling(20).std() / sr.rolling(120).std()).to_numpy()
    R["vol_pico_spy"] = np.repeat(vr[:, None], c.shape[1], 1)  # >1: la volatilidad se dispara
    mes = pd.DatetimeIndex(p.fechas).to_period("M")
    pos = pd.Series(np.arange(len(mes))).groupby(np.asarray(mes)).transform("max").to_numpy()
    R["sesiones_fin_mes"] = np.repeat((pos - np.arange(len(mes)))[:, None].astype(float), c.shape[1], 1)
    np.savez(SALIDA, pares=pares, **{k: np.asarray(v, dtype=np.float32) for k, v in R.items()})
    print(f"{len(R)} rasgos -> {SALIDA}  ({time.time() - t0:.0f}s)")


def cargar_rasgos() -> dict[str, np.ndarray]:
    z = np.load(SALIDA)
    return {k: z[k] for k in z.files}


if __name__ == "__main__":
    construir()
