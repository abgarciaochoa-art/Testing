"""Consolida todas las fuentes de consenso descargadas en un unico panel.

Salida: consenso_master.parquet  (una fila por ticker x trimestre x fuente)

Columnas:
  ticker, fiscal_period_end, report_date, report_time, eps_reported,
  eps_estimated, surprise, surprise_pct, source, snapshot_date

`source` identifica el mirror y `snapshot_date` la fecha en que ese mirror
capturo los datos (critico: el consenso de Alpha Vantage NO es point-in-time,
se reescribe entre snapshots; ver docs/research/datasets_consenso.md).
"""

from __future__ import annotations

import glob
import json
import os

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
COLS = [
    "ticker",
    "fiscal_period_end",
    "report_date",
    "report_time",
    "eps_reported",
    "eps_estimated",
    "surprise",
    "surprise_pct",
    "source",
    "snapshot_date",
]


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def load_rico() -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(os.path.join(BASE, "av_earnings_rico3223", "*.csv"))):
        t = os.path.basename(f)[len("data_") : -len(".csv")]
        d = pd.read_csv(f)
        d["ticker"] = t
        rows.append(d)
    d = pd.concat(rows, ignore_index=True)
    return pd.DataFrame(
        {
            "ticker": d.ticker,
            "fiscal_period_end": d.fiscalDateEnding.astype(str).str[:10],
            "report_date": d.reportedDate.astype(str).str[:10],
            "report_time": None,
            "eps_reported": _num(d.reportedEPS),
            "eps_estimated": _num(d.estimatedEPS),
            "surprise": _num(d.surprise),
            "surprise_pct": _num(d.surprisePercentage),
            "source": "alphavantage/rico3223",
            "snapshot_date": "2022-11-30",
        }
    )


def _av_json_dir(dirn: str, strip: str, source: str, fallback_snap: str) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(os.path.join(BASE, dirn, "*.json"))):
        t = os.path.basename(f).replace(strip, "")
        raw = json.load(open(f))
        snap = fallback_snap
        if "data" in raw and isinstance(raw["data"], dict):
            snap = (raw.get("timestamp") or fallback_snap)[:10]
            raw = raw["data"]
        for r in raw.get("quarterlyEarnings", []):
            rows.append(
                {
                    "ticker": t,
                    "fiscal_period_end": str(r.get("fiscalDateEnding"))[:10],
                    "report_date": str(r.get("reportedDate"))[:10],
                    "report_time": r.get("reportTime"),
                    "eps_reported": r.get("reportedEPS"),
                    "eps_estimated": r.get("estimatedEPS"),
                    "surprise": r.get("surprise"),
                    "surprise_pct": r.get("surprisePercentage"),
                    "source": source,
                    "snapshot_date": snap,
                }
            )
    d = pd.DataFrame(rows)
    for c in ["eps_reported", "eps_estimated", "surprise", "surprise_pct"]:
        d[c] = _num(d[c])
    return d


def load_turingplanet() -> pd.DataFrame:
    d = pd.read_csv(os.path.join(BASE, "av_earnings_turingplanet", "quarterly_earnings.csv"))
    return pd.DataFrame(
        {
            "ticker": d.symbol,
            "fiscal_period_end": d.fiscalDateEnding.astype(str).str[:10],
            "report_date": d.reportedDate.astype(str).str[:10],
            "report_time": d.reportTime,
            "eps_reported": _num(d.reportedEPS),
            "eps_estimated": _num(d.estimatedEPS),
            "surprise": _num(d.surprise),
            "surprise_pct": _num(d.surprisePercentage),
            "source": "alphavantage/turingplanet",
            "snapshot_date": "2024-03-31",
        }
    )


def load_yf(dirn: str, datecol: str, est: str, act: str, sur: str, source: str, snap: str) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(os.path.join(BASE, dirn, "*.csv"))):
        t = os.path.basename(f).replace("_earnings.csv", "")
        d = pd.read_csv(f)
        d["ticker"] = t
        rows.append(d)
    d = pd.concat(rows, ignore_index=True)
    ts = pd.to_datetime(d[datecol], utc=True, errors="coerce")
    # yfinance da la hora exacta del anuncio en la zona del mercado -> BMO/AMC
    local = ts.dt.tz_convert("America/New_York")
    rt = pd.Series("unknown", index=d.index, dtype=object)
    rt[local.dt.hour < 9] = "pre-market"
    rt[(local.dt.hour >= 16)] = "post-market"
    rt[(local.dt.hour >= 9) & (local.dt.hour < 16)] = "intraday"
    return pd.DataFrame(
        {
            "ticker": d.ticker,
            "fiscal_period_end": pd.NA,
            "report_date": local.dt.strftime("%Y-%m-%d"),
            "report_time": rt,
            "eps_reported": _num(d[act]),
            "eps_estimated": _num(d[est]),
            "surprise": _num(d[act]) - _num(d[est]),
            "surprise_pct": _num(d[sur]),
            "source": source,
            "snapshot_date": snap,
        }
    )


def main() -> None:
    parts = [
        load_rico(),
        _av_json_dir("av_earnings_tim7en_2025", "_earnings.json", "alphavantage/tim7en", "2025-09-11"),
        _av_json_dir("av_earnings_pulse_2026", "_EARNINGS.json", "alphavantage/earning-pulse", "2026-02-28"),
        load_turingplanet(),
        load_yf("yf_earnings_dates_chinar", "EarningsDate", "EPS Estimate", "Reported EPS", "Surprise(%)",
                "yfinance/chinar-byte", "2025-12-31"),
        load_yf("yf_earnings_dates_rithvik", "Earnings Date", "eps_estimate", "eps_actual", "eps_surprise_pct",
                "yfinance/rithvik-konda", "2026-03-31"),
    ]
    df = pd.concat(parts, ignore_index=True)[COLS]
    df = df[df.report_date.notna() & (df.report_date != "nan") & (df.report_date != "NaT")]
    df = df.sort_values(["ticker", "report_date", "source"]).reset_index(drop=True)
    out = os.path.join(BASE, "consenso_master.parquet")
    df.to_parquet(out, index=False)
    print(f"escrito {out}: {df.shape}")
    print(df.groupby("source").agg(
        filas=("ticker", "size"),
        tickers=("ticker", "nunique"),
        desde=("report_date", "min"),
        hasta=("report_date", "max"),
        con_estimado=("eps_estimated", lambda s: f"{s.notna().mean()*100:.1f}%"),
    ).to_string())


if __name__ == "__main__":
    main()
