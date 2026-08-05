"""Genera los fixtures de tests/fixtures/prices/.

Los valores son sintéticos pero internamente coherentes: ocho sesiones NYSE
(2020-08-24 .. 2020-09-02) con un split 4:1 el 2020-08-31 y un dividendo de
0.205 USD con fecha ex 2020-09-02, replicando el episodio AAPL de agosto de
2020. Cada fichero sigue la FORMA documentada de la API correspondiente
(verificada contra la documentación pública; véanse los docstrings de
earnings_alpha/data/prices.py).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

NY = ZoneInfo("America/New_York")

DATES = [
    "2020-08-24",
    "2020-08-25",
    "2020-08-26",
    "2020-08-27",
    "2020-08-28",
    "2020-08-31",  # split 4:1 efectivo
    "2020-09-01",
    "2020-09-02",  # ex-dividendo 0.205
]

# OHLCV tal y como se negoció (base de acciones de cada día).
OPEN = [505.00, 504.00, 501.00, 508.00, 503.00, 127.50, 130.00, 135.00]
HIGH = [515.00, 508.00, 510.00, 512.00, 506.00, 131.00, 135.00, 138.00]
LOW = [500.00, 496.00, 499.00, 498.00, 495.00, 126.00, 129.50, 127.00]
CLOSE = [504.00, 500.00, 508.00, 502.00, 500.00, 129.00, 134.00, 131.30]
VOLUME = [80e6, 60e6, 55e6, 50e6, 46e6, 220e6, 150e6, 190e6]

SPLIT_DAY = "2020-08-31"
SPLIT_FACTOR = 4.0
DIV_DAY = "2020-09-02"
DIV_AMOUNT = 0.205

SPLIT_IDX = DATES.index(SPLIT_DAY)
DIV_IDX = DATES.index(DIV_DAY)


def split_multiplier(i: int) -> float:
    """Multiplicador que devuelve la base actual (post-split) desde la de i."""
    return SPLIT_FACTOR if i < SPLIT_IDX else 1.0


def dividend_on(i: int) -> float:
    return DIV_AMOUNT if i == DIV_IDX else 0.0


def split_on(i: int) -> float:
    return SPLIT_FACTOR if i == SPLIT_IDX else 1.0


def total_return_factors() -> list[float]:
    out = [float("nan")]
    for i in range(1, len(DATES)):
        out.append((CLOSE[i] * split_on(i) + dividend_on(i)) / CLOSE[i - 1])
    return out


def adj_close_anchored_end() -> list[float]:
    """adj_close CRSP anclado al último cierre de la ventana."""
    fac = total_return_factors()
    cum = [1.0]
    for f in fac[1:]:
        cum.append(cum[-1] * f)
    anchor = CLOSE[-1] / cum[-1]
    return [c * anchor for c in cum]


def epoch_open_s(d: str) -> int:
    """Epoch (s) de la apertura 09:30 ET, como los timestamps diarios de Yahoo."""
    dt = datetime.fromisoformat(d + "T09:30:00").replace(tzinfo=NY)
    return int(dt.timestamp())


def epoch_midnight_et_ms(d: str) -> int:
    """Epoch (ms) de la medianoche ET, como el campo `t` de Polygon."""
    dt = datetime.fromisoformat(d + "T00:00:00").replace(tzinfo=NY)
    return int(dt.timestamp() * 1000)


# ---------------------------------------------------------------------------
# Yahoo Finance v8 chart: precios retro-ajustados por splits (base actual)
# ---------------------------------------------------------------------------
yf_open = [OPEN[i] / split_multiplier(i) for i in range(8)]
yf_high = [HIGH[i] / split_multiplier(i) for i in range(8)]
yf_low = [LOW[i] / split_multiplier(i) for i in range(8)]
yf_close = [CLOSE[i] / split_multiplier(i) for i in range(8)]
yf_volume = [VOLUME[i] * split_multiplier(i) for i in range(8)]

# adjclose de Yahoo: split+dividendo, ancla en el último día conocido (aquí,
# el fin de la ventana). El adaptador NO lo usa; se incluye por fidelidad.
adj = adj_close_anchored_end()

split_epoch = epoch_open_s(SPLIT_DAY)
div_epoch = epoch_open_s(DIV_DAY)

yfinance_chart = {
    "chart": {
        "result": [
            {
                "meta": {
                    "currency": "USD",
                    "symbol": "AAPL",
                    "exchangeName": "NMS",
                    "instrumentType": "EQUITY",
                    "firstTradeDate": 345479400,
                    "regularMarketTime": epoch_open_s(DATES[-1]) + 6 * 3600 + 30 * 60,
                    "gmtoffset": -14400,
                    "timezone": "EDT",
                    "exchangeTimezoneName": "America/New_York",
                    "regularMarketPrice": yf_close[-1],
                    "priceHint": 2,
                    "dataGranularity": "1d",
                    "range": "",
                    "validRanges": ["1d", "5d", "1mo", "3mo", "6mo", "1y", "max"],
                },
                "timestamp": [epoch_open_s(d) for d in DATES],
                "events": {
                    "dividends": {
                        str(div_epoch): {"amount": DIV_AMOUNT, "date": div_epoch}
                    },
                    "splits": {
                        str(split_epoch): {
                            "date": split_epoch,
                            "numerator": 4,
                            "denominator": 1,
                            "splitRatio": "4:1",
                        }
                    },
                },
                "indicators": {
                    "quote": [
                        {
                            "open": [round(x, 4) for x in yf_open],
                            "high": [round(x, 4) for x in yf_high],
                            "low": [round(x, 4) for x in yf_low],
                            "close": [round(x, 4) for x in yf_close],
                            "volume": [int(v) for v in yf_volume],
                        }
                    ],
                    "adjclose": [{"adjclose": [round(x, 6) for x in adj]}],
                },
            }
        ],
        "error": None,
    }
}
(OUT / "yfinance_aapl_chart.json").write_text(
    json.dumps(yfinance_chart, indent=2) + "\n", encoding="utf-8"
)

yfinance_not_found = {
    "chart": {
        "result": None,
        "error": {
            "code": "Not Found",
            "description": "No data found, symbol may be delisted",
        },
    }
}
(OUT / "yfinance_not_found.json").write_text(
    json.dumps(yfinance_not_found, indent=2) + "\n", encoding="utf-8"
)

# ---------------------------------------------------------------------------
# Polygon v2 aggs (adjusted=false) en dos páginas + splits/dividends v3
# ---------------------------------------------------------------------------


def polygon_bar(i: int) -> dict:
    return {
        "v": VOLUME[i],
        "vw": round((HIGH[i] + LOW[i] + CLOSE[i]) / 3.0, 4),
        "o": OPEN[i],
        "c": CLOSE[i],
        "h": HIGH[i],
        "l": LOW[i],
        "t": epoch_midnight_et_ms(DATES[i]),
        "n": int(VOLUME[i] / 200),
    }


page1 = {
    "ticker": "AAPL",
    "queryCount": 5,
    "resultsCount": 5,
    "adjusted": False,
    "results": [polygon_bar(i) for i in range(5)],
    "status": "OK",
    "request_id": "fixture-aggs-page1",
    "count": 5,
    "next_url": (
        "https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/"
        "2020-08-24/2020-09-02?cursor=fixture-cursor-1"
    ),
}
page2 = {
    "ticker": "AAPL",
    "queryCount": 3,
    "resultsCount": 3,
    "adjusted": False,
    "results": [polygon_bar(i) for i in range(5, 8)],
    "status": "OK",
    "request_id": "fixture-aggs-page2",
    "count": 3,
}
(OUT / "polygon_aggs_aapl_page1.json").write_text(
    json.dumps(page1, indent=2) + "\n", encoding="utf-8"
)
(OUT / "polygon_aggs_aapl_page2.json").write_text(
    json.dumps(page2, indent=2) + "\n", encoding="utf-8"
)

polygon_dividends = {
    "results": [
        {
            "cash_amount": DIV_AMOUNT,
            "currency": "USD",
            "declaration_date": "2020-07-30",
            "dividend_type": "CD",
            "ex_dividend_date": DIV_DAY,
            "frequency": 4,
            "id": "fixture-div-1",
            "pay_date": "2020-09-10",
            "record_date": "2020-09-07",
            "ticker": "AAPL",
        }
    ],
    "status": "OK",
    "request_id": "fixture-dividends",
}
(OUT / "polygon_dividends_aapl.json").write_text(
    json.dumps(polygon_dividends, indent=2) + "\n", encoding="utf-8"
)

polygon_splits = {
    "results": [
        {
            "execution_date": SPLIT_DAY,
            "id": "fixture-split-1",
            "split_from": 1,
            "split_to": 4,
            "ticker": "AAPL",
        }
    ],
    "status": "OK",
    "request_id": "fixture-splits",
}
(OUT / "polygon_splits_aapl.json").write_text(
    json.dumps(polygon_splits, indent=2) + "\n", encoding="utf-8"
)

polygon_empty = {
    "ticker": "ZZZTOP",
    "queryCount": 0,
    "resultsCount": 0,
    "adjusted": False,
    "status": "OK",
    "request_id": "fixture-aggs-empty",
    "count": 0,
}
(OUT / "polygon_aggs_empty.json").write_text(
    json.dumps(polygon_empty, indent=2) + "\n", encoding="utf-8"
)

# ---------------------------------------------------------------------------
# Tiingo EOD: as-traded + divCash/splitFactor por día
# ---------------------------------------------------------------------------
tiingo_rows = []
for i, d in enumerate(DATES):
    mult = split_multiplier(i)
    tiingo_rows.append(
        {
            "date": d + "T00:00:00.000Z",
            "close": CLOSE[i],
            "high": HIGH[i],
            "low": LOW[i],
            "open": OPEN[i],
            "volume": int(VOLUME[i]),
            "adjClose": round(adj[i], 6),
            "adjHigh": round(HIGH[i] / mult, 4),
            "adjLow": round(LOW[i] / mult, 4),
            "adjOpen": round(OPEN[i] / mult, 4),
            "adjVolume": int(VOLUME[i] * mult),
            "divCash": dividend_on(i),
            "splitFactor": split_on(i),
        }
    )
(OUT / "tiingo_aapl.json").write_text(
    json.dumps(tiingo_rows, indent=2) + "\n", encoding="utf-8"
)

tiingo_not_found = {
    "detail": "Not found. Error: Ticker 'ZZZTOP' not found or is not supported."
}
(OUT / "tiingo_not_found.json").write_text(
    json.dumps(tiingo_not_found, indent=2) + "\n", encoding="utf-8"
)

# ---------------------------------------------------------------------------
# Stooq: CSV ya ajustado por splits y dividendos (base actual, sin factores)
# ---------------------------------------------------------------------------
lines = ["Date,Open,High,Low,Close,Volume"]
fac = total_return_factors()
# Serie de retorno total en base post-split: cocientes = factores, ancla al final.
stooq_close = adj_close_anchored_end()
for i, d in enumerate(DATES):
    mult = split_multiplier(i)
    scale = stooq_close[i] / CLOSE[i]
    lines.append(
        f"{d},{OPEN[i] * scale:.4f},{HIGH[i] * scale:.4f},"
        f"{LOW[i] * scale:.4f},{stooq_close[i]:.4f},{int(VOLUME[i] * mult)}"
    )
(OUT / "stooq_aapl.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
(OUT / "stooq_no_data.txt").write_text("No data\n", encoding="utf-8")

# ---------------------------------------------------------------------------
# Alpaca /v2/stocks/bars: raw + all, con paginación en el raw
# ---------------------------------------------------------------------------
MSFT_CLOSE = [210.0, 212.0, 214.0, 213.0, 215.0, 216.0, 218.0, 217.0]


def alpaca_bar(i: int, o: float, h: float, lo: float, c: float, v: float) -> dict:
    return {
        "t": DATES[i] + "T04:00:00Z",
        "o": o,
        "h": h,
        "l": lo,
        "c": c,
        "v": int(v),
        "n": int(v / 200),
        "vw": round((h + lo + c) / 3.0, 4),
    }


aapl_raw = [alpaca_bar(i, OPEN[i], HIGH[i], LOW[i], CLOSE[i], VOLUME[i]) for i in range(8)]
msft_raw = [
    alpaca_bar(i, MSFT_CLOSE[i] - 1.0, MSFT_CLOSE[i] + 2.0, MSFT_CLOSE[i] - 3.0,
               MSFT_CLOSE[i], 30e6)
    for i in range(8)
]

# adjustment=all: split+dividendo. Alpaca ancla a su historia completa; aquí el
# ancla coincide con el fin de la ventana (no hay eventos posteriores), y lo
# único contractual son los cocientes.
aapl_all = [
    alpaca_bar(
        i,
        round(OPEN[i] * adj[i] / CLOSE[i], 6),
        round(HIGH[i] * adj[i] / CLOSE[i], 6),
        round(LOW[i] * adj[i] / CLOSE[i], 6),
        round(adj[i], 6),
        VOLUME[i] * split_multiplier(i),
    )
    for i in range(8)
]
msft_all = msft_raw  # sin eventos corporativos: raw == all

alpaca_raw_page1 = {
    "bars": {"AAPL": aapl_raw[:5], "MSFT": msft_raw[:5]},
    "next_page_token": "fixture-token-1",
}
alpaca_raw_page2 = {
    "bars": {"AAPL": aapl_raw[5:], "MSFT": msft_raw[5:]},
    "next_page_token": None,
}
alpaca_all_page = {
    "bars": {"AAPL": aapl_all, "MSFT": msft_all},
    "next_page_token": None,
}
(OUT / "alpaca_bars_raw_page1.json").write_text(
    json.dumps(alpaca_raw_page1, indent=2) + "\n", encoding="utf-8"
)
(OUT / "alpaca_bars_raw_page2.json").write_text(
    json.dumps(alpaca_raw_page2, indent=2) + "\n", encoding="utf-8"
)
(OUT / "alpaca_bars_all.json").write_text(
    json.dumps(alpaca_all_page, indent=2) + "\n", encoding="utf-8"
)

# ---------------------------------------------------------------------------
# Verdad-terreno para los tests
# ---------------------------------------------------------------------------
truth = {
    "dates": DATES,
    "open": OPEN,
    "high": HIGH,
    "low": LOW,
    "close": CLOSE,
    "volume": VOLUME,
    "split_day": SPLIT_DAY,
    "split_factor": SPLIT_FACTOR,
    "div_day": DIV_DAY,
    "div_amount": DIV_AMOUNT,
    "total_return_factors": [None if i == 0 else round(f, 12) for i, f in enumerate(fac)],
    "adj_close_anchored_end": [round(x, 10) for x in adj],
    "msft_close": MSFT_CLOSE,
}
(OUT / "ground_truth.json").write_text(
    json.dumps(truth, indent=2) + "\n", encoding="utf-8"
)

readme = """# Fixtures de precios (`tests/fixtures/prices/`)

Fixtures **construidos a mano contra la forma documentada** de cada API de
precios (el contenedor de desarrollo no alcanza las APIs financieras, contrato
`docs/ARCHITECTURE.md` §5, así que no son grabaciones de tráfico real). La forma
de cada respuesta está verificada contra la documentación pública del proveedor
(2026-08) y citada en el docstring del adaptador correspondiente en
`earnings_alpha/data/prices.py`.

Todos los ficheros describen el MISMO episodio de mercado sintético, calcado del
split de AAPL de agosto de 2020: ocho sesiones NYSE (2020-08-24 → 2020-09-02),
un **split 4:1 efectivo el 2020-08-31** y un **dividendo de 0.205 USD con fecha
ex 2020-09-02**. Como todos los proveedores describen el mismo suceso en su
propio formato (Yahoo retro-ajustado, Polygon crudo + eventos v3, Tiingo con
`divCash`/`splitFactor` diarios, Stooq ya ajustado, Alpaca raw/all), los tests
pueden exigir que los seis adaptadores produzcan **exactamente el mismo panel**.

`ground_truth.json` contiene la serie as-traded y los factores de retorno total
CRSP `(close_t*split_t + div_t)/close_{t-1}` calculados una sola vez; es la
referencia contra la que se comprueban los adaptadores.

Regenerables con `python3 tests/fixtures/prices/_make_fixtures.py` (determinista, sin red).

| Fichero | API imitada |
|---|---|
| `yfinance_aapl_chart.json` | Yahoo `v8/finance/chart` (precios en base actual + events) |
| `yfinance_not_found.json` | Yahoo `chart.error` para símbolo inexistente (HTTP 404) |
| `polygon_aggs_aapl_page{1,2}.json` | Polygon `/v2/aggs` `adjusted=false`, paginado con `next_url` |
| `polygon_{dividends,splits}_aapl.json` | Polygon `/v3/reference/*` |
| `polygon_aggs_empty.json` | Polygon aggs sin resultados |
| `tiingo_aapl.json` | Tiingo `/tiingo/daily/{t}/prices` (as-traded + divCash/splitFactor) |
| `tiingo_not_found.json` | Tiingo detail «Not found» |
| `stooq_aapl.csv` | Stooq `q/d/l` CSV ya ajustado |
| `stooq_no_data.txt` | Stooq sin datos |
| `alpaca_bars_raw_page{1,2}.json` | Alpaca `/v2/stocks/bars` `adjustment=raw`, paginado |
| `alpaca_bars_all.json` | Alpaca `adjustment=all` |
| `ground_truth.json` | serie as-traded y factores CRSP de referencia |
"""
(OUT / "README.md").write_text(readme, encoding="utf-8")

print("fixtures escritos en", OUT)
for p in sorted(OUT.iterdir()):
    print(" ", p.name, p.stat().st_size, "bytes")
