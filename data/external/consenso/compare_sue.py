"""Compara el SUE de series temporales (Foster, Olsen y Shevlin 1984) con el SUE
basado en el consenso de analistas, usando el panel `consenso_master.parquet`.

Responde a la pregunta operativa: *si renuncio al consenso de analistas y uso solo
el historico de EPS reportado, cuanto de la senal pierdo?*

Uso:  python3 compare_sue.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE = os.path.dirname(os.path.abspath(__file__))


def build(source: str = "alphavantage/rico3223") -> pd.DataFrame:
    df = pd.read_parquet(os.path.join(BASE, "consenso_master.parquet"))
    d = df[df.source == source].copy()
    d = d.dropna(subset=["eps_reported", "fiscal_period_end"])
    d = d.sort_values(["ticker", "fiscal_period_end"]).reset_index(drop=True)

    g = d.groupby("ticker")["eps_reported"]
    d["dy"] = d.eps_reported - g.shift(4)  # paseo aleatorio estacional
    gd = d.groupby("ticker")["dy"]
    # drift y sigma calculados SOLO con informacion anterior al trimestre t (shift(1))
    d["drift"] = gd.transform(lambda s: s.shift(1).rolling(8, min_periods=6).mean())
    d["sigma"] = gd.transform(lambda s: s.shift(1).rolling(8, min_periods=6).std())

    d["sue_ts"] = (d.dy - d.drift) / d.sigma
    d["sue_an"] = (d.eps_reported - d.eps_estimated) / d.sigma
    d["year"] = d.report_date.str[:4].astype(int)
    return d


def report(v: pd.DataFrame, label: str) -> None:
    v = v.replace([np.inf, -np.inf], np.nan).dropna(subset=["sue_ts", "sue_an"])
    if len(v) < 100:
        print(f"[{label}] muestra insuficiente ({len(v)})")
        return
    rho, p = spearmanr(v.sue_ts, v.sue_an)
    same_sign = (np.sign(v.sue_ts) == np.sign(v.sue_an)).mean()
    dts = pd.qcut(v.sue_ts.rank(method="first"), 10, labels=False)
    dan = pd.qcut(v.sue_an.rank(method="first"), 10, labels=False)
    top = (dts[dan == 9] == 9).mean()
    bot = (dts[dan == 0] == 0).mean()
    print(f"[{label}] n={len(v):,} tickers={v.ticker.nunique()}")
    print(f"    Spearman(sue_ts, sue_an) = {rho:.3f}  (p={p:.2g})")
    print(f"    mismo signo de sorpresa  = {same_sign*100:.1f}%")
    print(f"    mismo decil              = {(dts==dan).mean()*100:.1f}%   +-1 decil = {((dts-dan).abs()<=1).mean()*100:.1f}%")
    print(f"    solapamiento decil 10 (largo) = {top*100:.1f}%   decil 1 (corto) = {bot*100:.1f}%")
    print(f"    sorpresas positivas: analista {(v.sue_an>0).mean()*100:.1f}%  vs  series temp. {(v.sue_ts>0).mean()*100:.1f}%")


def main() -> None:
    d = build()
    report(d, "1996-2022 completo")
    for a, b in [(1996, 2004), (2005, 2013), (2014, 2022)]:
        report(d[(d.year >= a) & (d.year <= b)], f"{a}-{b}")


if __name__ == "__main__":
    main()
