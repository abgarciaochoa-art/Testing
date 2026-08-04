"""Adaptadores de precios OHLCV multi-proveedor (contrato §2, módulo `data.market`).

Este módulo implementa la interfaz común de precios diarios para el proyecto:

    get_bars(tickers, start, end, adjusted=True) -> panel MultiIndex (date, ticker)

con seis proveedores: Tiingo, Polygon, Alpaca, Stooq, Yahoo (yfinance) y el
generador sintético del repo. Todos devuelven el **panel canónico** del contrato
(`docs/ARCHITECTURE.md` §1): MultiIndex ``(date, ticker)`` ordenado, fechas
tz-naive normalizadas a medianoche, y las columnas

==================  ==========================================================
Columna             Semántica
==================  ==========================================================
``open/high/low``   precios de la sesión **tal y como se negociaron** (sin
                    retro-ajustar), cuando el proveedor lo permite
``close``           cierre sin ajustar (el precio que un gestor vio ese día)
``volume``          acciones negociadas ese día, en las acciones de ese día
``adj_close``       cierre de **retorno total** (splits + dividendos),
                    retro-ajustado y anclado al último cierre de la ventana
``dividend``        importe bruto por acción en la fecha ex-dividendo; ``0.0``
                    significa "se sabe que no hubo"; ``NaN`` significa "el
                    proveedor no informa de dividendos"
``split_factor``    multiplicador de acciones efectivo ese día (``4.0`` para un
                    split 4:1, ``0.125`` para un contrasplit 1:8); ``1.0`` = no
                    hubo; ``NaN`` = el proveedor no informa de splits
==================  ==========================================================

Invariante de ajuste (la identidad que une todas las columnas)
--------------------------------------------------------------
El **factor de retorno total** de la sesión ``t`` es

    f_t = (close_t * split_factor_t + dividend_t) / close_{t-1}

que es la definición del *holding period return* de CRSP (CRSP, *Data
Descriptions Guide*: retorno con dividendos y ajuste por cambios de capital).
`adj_close` se construye de modo que ``adj_close_t / adj_close_{t-1} == f_t``
exactamente, con ancla ``adj_close_T == close_T`` en el último día de la
ventana. Es la misma construcción que usa `data.synthetic.SyntheticMarket`, de
modo que un backtest puede cambiar de proveedor sin cambiar de matemática.

Por qué `close` crudo y no solo `adjClose` (resumen de
`docs/research/data_sources.md` §3.1): el `adj_close` retro-ajustado con la
historia conocida **hoy** es correcto para retornos, pero un filtro
``precio > 5 USD``, un *earnings yield* o un umbral de liquidez calculados sobre
él usan información de splits futuros que en ``t`` no existía. Por eso la
interfaz entrega ambas columnas y por eso se prefiere el proveedor que da
`close` crudo + factores fechados (Tiingo, Polygon) sobre el que solo da
`adjClose` (Yahoo).

Nota PIT sobre el ancla: `adj_close` se ancla al **final de la ventana pedida**,
no a "hoy", para que la misma petición devuelva los mismos números dentro de un
año (un ancla en "hoy" cambia con cada split posterior). Los niveles de
`adj_close` no son precios negociables de ningún día: solo sus *cocientes*
tienen significado.

Validación de calidad
---------------------
`validate_price_panel` aplica los filtros clásicos de la literatura de datos de
precios (Ince y Porter, 2006, *Individual equity return data from Thomson
Datastream: Handle with care!*, Journal of Financial Research 29(4), que
documentan retornos ficticios de ±300% por errores de ajuste): precios no
positivos, OHLC incoherente, saltos imposibles, saltos con pinta de split sin
split registrado, volumen cero prolongado, huecos frente al `TradingCalendar` y
barras en días no bursátiles. Cada hallazgo tiene gravedad: los ``error``
lanzan `DataQualityError` (desde `get_bars`) y los ``warning`` se registran en
el log.

Red y fixtures
--------------
El contenedor de desarrollo tiene bloqueadas las APIs financieras (contrato §5),
así que estos adaptadores están escritos contra la documentación pública de cada
API (formas de respuesta verificadas vía WebSearch/WebFetch, 2026-08; véanse las
URLs en cada clase) y probados contra fixtures grabados en
`tests/fixtures/prices/` con `ScriptedTransport`. La verificación contra la API
viva queda para la máquina del usuario (`@pytest.mark.network`).
"""

from __future__ import annotations

import io
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from enum import StrEnum
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from earnings_alpha.config import Settings, get_settings
from earnings_alpha.data.base import (
    BaseProvider,
    Clock,
    DataKind,
    HttpClient,
    ProviderRegistry,
    get_registry,
)
from earnings_alpha.data.cache import DiskCache
from earnings_alpha.data.synthetic import SyntheticMarket
from earnings_alpha.errors import (
    CalendarError,
    ConfigError,
    DataQualityError,
    EarningsAlphaError,
    InsufficientHistory,
    ProviderUnavailable,
)
from earnings_alpha.pit import TradingCalendar, get_calendar
from earnings_alpha.types import Bar, Ticker, normalize_ticker

__all__ = [  # noqa: RUF022 - orden temático, no alfabético
    # contrato
    "PRICE_COLUMNS",
    "PRICES_KIND",
    "PriceProvider",
    "get_bars",
    "register_price_providers",
    # ajuste y retornos
    "total_return_factors",
    "compute_adj_close",
    "adjust_panel",
    "total_returns",
    # interoperabilidad con types.Bar
    "panel_to_bars",
    "bars_to_panel",
    # calidad
    "Severity",
    "QualityIssue",
    "QualityReport",
    "validate_price_panel",
    # proveedores
    "PriceProviderBase",
    "SyntheticProvider",
    "YFinanceProvider",
    "PolygonProvider",
    "TiingoProvider",
    "StooqProvider",
    "AlpacaProvider",
    "DEFAULT_PRICE_PRIORITIES",
]

logger = logging.getLogger(__name__)

PRICES_KIND: str = DataKind.PRICES.value
"""Tipo de dato bajo el que se registran estos proveedores en el registro."""

PRICE_COLUMNS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adj_close",
    "dividend",
    "split_factor",
)
"""Columnas del panel canónico de precios, en su orden contractual."""

DateLike = date | datetime | str | pd.Timestamp

_NY_TZ = "America/New_York"


# ===========================================================================
# 1. Utilidades de fechas y montaje del panel
# ===========================================================================


def _as_date(value: DateLike, label: str = "fecha") -> date:
    """Convierte a `datetime.date`; lanza `ConfigError` si no es interpretable."""
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip()[:10])
        except ValueError as exc:
            msg = f"{label} no interpretable como fecha: {value!r}"
            raise ConfigError(msg) from exc
    msg = f"{label} debe ser date/datetime/str/Timestamp; recibido {type(value).__name__}"
    raise ConfigError(msg)


def _check_window(start: DateLike, end: DateLike) -> tuple[date, date]:
    s = _as_date(start, "start")
    e = _as_date(end, "end")
    if s > e:
        msg = f"ventana invertida: start={s.isoformat()} > end={e.isoformat()}"
        raise ConfigError(msg)
    return s, e


def _normalize_tickers(tickers: Ticker | Sequence[Ticker]) -> list[Ticker]:
    """Normaliza y deduplica preservando el orden. Lanza si la lista queda vacía."""
    raw = [tickers] if isinstance(tickers, str) else list(tickers)
    seen: dict[str, None] = {}
    for t in raw:
        norm = normalize_ticker(str(t))
        if norm:
            seen.setdefault(norm, None)
    if not seen:
        msg = "la lista de tickers está vacía tras normalizar"
        raise ConfigError(msg)
    return list(seen)


def _epochs_to_dates(values: Sequence[float], unit: str, tz: str) -> pd.DatetimeIndex:
    """Epochs UTC -> fecha de sesión en la zona del mercado, tz-naive a medianoche.

    La conversión a la zona de la bolsa importa: una barra diaria de Polygon
    lleva ``t`` = medianoche ET en milisegundos, que en UTC es 04:00/05:00 del
    mismo día; interpretarla como UTC directo funcionaría hoy, pero una fuente
    que timestampe al cierre (21:00 UTC) seguiría siendo el mismo día en ET y el
    día siguiente no. Normalizar en la zona del mercado es lo único robusto.
    """
    idx = pd.to_datetime(list(values), unit=unit, utc=True)
    return pd.DatetimeIndex(idx.tz_convert(tz).normalize().tz_localize(None), name="date")


def _iso_to_dates(values: Sequence[str], tz: str | None) -> pd.DatetimeIndex:
    """Timestamps ISO -> fecha de sesión tz-naive.

    Con ``tz=None`` se toma la fecha del literal sin conversión de zona: es lo
    correcto para Tiingo, cuyos ``"2020-08-31T00:00:00.000Z"`` son la *fecha de
    negociación* etiquetada a medianoche UTC (convertirla a Nueva York la
    retrasaría un día: off-by-one clásico).
    """
    idx = pd.to_datetime(list(values), utc=True)
    if tz is not None:
        idx = idx.tz_convert(tz)
    return pd.DatetimeIndex(idx.tz_localize(None).normalize(), name="date")


def _clip_window(frame: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    return frame[(frame.index >= lo) & (frame.index <= hi)]


def _bar_frame(
    dates: pd.DatetimeIndex,
    data: Mapping[str, Sequence[Any] | np.ndarray | pd.Series | float],
) -> pd.DataFrame:
    """Construye el frame por-ticker con las columnas canónicas y dtype float."""
    frame = pd.DataFrame(index=pd.DatetimeIndex(dates, name="date"))
    for col in PRICE_COLUMNS:
        value = data.get(col, np.nan)
        if np.isscalar(value) or value is None:
            frame[col] = float("nan") if value is None else float(value)  # type: ignore[arg-type]
        else:
            frame[col] = pd.to_numeric(pd.Series(list(value), index=frame.index), errors="coerce")
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame.astype("float64")


def _assemble_panel(frames: Mapping[Ticker, pd.DataFrame]) -> pd.DataFrame:
    """Apila frames por-ticker en el panel canónico `(date, ticker)` ordenado."""
    parts: list[pd.DataFrame] = []
    for ticker, frame in frames.items():
        sub = frame.copy()
        sub.index = pd.MultiIndex.from_arrays(
            [sub.index, [ticker] * len(sub)], names=["date", "ticker"]
        )
        parts.append(sub)
    panel = pd.concat(parts).sort_index()
    return panel[list(PRICE_COLUMNS)]


# ===========================================================================
# 2. Ajuste por splits/dividendos y retorno total
# ===========================================================================


def _require_no_nan(series: pd.Series, name: str) -> pd.Series:
    if series.isna().any():
        n = int(series.isna().sum())
        msg = (
            f"la serie {name!r} tiene {n} NaN: el ajuste de retorno total exige conocer "
            "dividendos y splits en cada sesión. Un proveedor que no los informa "
            "(Stooq, Alpaca) no puede pasar por compute_adj_close; usa su adj_close "
            "nativo o cambia de proveedor."
        )
        raise DataQualityError(msg)
    return series.astype("float64")


def total_return_factors(
    close: pd.Series,
    dividend: pd.Series | None = None,
    split_factor: pd.Series | None = None,
) -> pd.Series:
    """Factor bruto de retorno total por sesión de **un** símbolo.

    Fórmula (CRSP, *Data Descriptions Guide*, holding period return)::

        f_t = (close_t * split_t + div_t) / close_{t-1}

    donde ``close`` es el cierre sin ajustar, ``split_t`` el multiplicador de
    acciones efectivo en ``t`` (4.0 en un split 4:1: multiplicar por él devuelve
    el cierre a la base de acciones de ``t-1``) y ``div_t`` el dividendo bruto
    por acción con fecha ex en ``t``. La primera observación es NaN.

    Los NaN en `dividend`/`split_factor` se rechazan con `DataQualityError`:
    tratarlos como 0/1 en silencio produciría un "retorno total" que no lo es.
    """
    if len(close) == 0:
        msg = "serie de cierres vacía: no hay factores que calcular"
        raise InsufficientHistory(msg)
    px = _require_no_nan(close, "close")
    if (px <= 0).any():
        msg = "hay cierres no positivos: los factores de retorno no están definidos"
        raise DataQualityError(msg)
    div = (
        pd.Series(0.0, index=px.index)
        if dividend is None
        else _require_no_nan(dividend, "dividend")
    )
    split = (
        pd.Series(1.0, index=px.index)
        if split_factor is None
        else _require_no_nan(split_factor, "split_factor")
    )
    if (split <= 0).any():
        msg = "split_factor debe ser > 0 en todas las sesiones"
        raise DataQualityError(msg)
    factors = (px * split + div) / px.shift(1)
    return factors.rename("total_return_factor")


def compute_adj_close(
    close: pd.Series,
    dividend: pd.Series | None = None,
    split_factor: pd.Series | None = None,
) -> pd.Series:
    """Cierre ajustado por retorno total de **un** símbolo, anclado al final.

    Retro-ajuste estándar (CRSP / misma construcción que
    `SyntheticMarket._price_panel`): se acumulan los factores de
    `total_return_factors` y se reescala para que el último valor coincida con
    el último cierre sin ajustar. Garantiza exactamente::

        adj_t / adj_{t-1} == f_t          (retorno total)
        adj_T == close_T                  (ancla en el fin de la ventana)

    El ancla en el fin de la **ventana pedida** (y no en "hoy") hace el
    resultado reproducible: la misma petición devuelve los mismos números
    aunque después haya más splits.
    """
    factors = total_return_factors(close, dividend, split_factor)
    cum = factors.fillna(1.0).cumprod()
    adj = cum / cum.iloc[-1] * float(close.iloc[-1])
    return adj.rename("adj_close")


def adjust_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Recalcula `adj_close` de un panel canónico a partir de sus eventos.

    Exige columnas `dividend` y `split_factor` sin NaN (véase
    `total_return_factors`). Útil para re-anclar un panel recortado: tras un
    ``panel.loc[:fecha]`` el ancla original deja de estar en la ventana.
    """
    for col in ("close", "dividend", "split_factor"):
        if col not in panel.columns:
            msg = f"el panel no tiene la columna {col!r}; no se puede ajustar"
            raise DataQualityError(msg)
    out = panel.copy()
    parts: list[pd.Series] = []
    for _, group in out.groupby(level="ticker", sort=False):
        parts.append(compute_adj_close(group["close"], group["dividend"], group["split_factor"]))
    out["adj_close"] = pd.concat(parts)
    return out


def total_returns(
    panel: pd.DataFrame,
    *,
    method: Literal["auto", "adj_close", "events"] = "auto",
    log: bool = False,
) -> pd.Series:
    """Retorno total por `(date, ticker)` de un panel canónico.

    Métodos:

    - ``"events"``: fórmula exacta ``(close*split + div)/close_prev - 1``.
      Requiere `dividend` y `split_factor` sin NaN.
    - ``"adj_close"``: cociente de `adj_close` consecutivos. Idéntico al
      anterior **por construcción** cuando `adj_close` salió de
      `compute_adj_close`; es la única opción con proveedores sin eventos
      (Stooq, Alpaca).
    - ``"auto"``: ``"events"`` si hay eventos completos, si no ``"adj_close"``.

    Con ``log=True`` devuelve el logaritmo del factor en vez de ``f - 1``.
    La primera observación de cada símbolo es NaN. Nunca se rellenan NaN en
    silencio: un retorno inventado es peor que un hueco visible.
    """
    if not isinstance(panel.index, pd.MultiIndex) or list(panel.index.names) != ["date", "ticker"]:
        msg = "total_returns espera el panel canónico con MultiIndex (date, ticker)"
        raise DataQualityError(msg)
    has_events = (
        "dividend" in panel.columns
        and "split_factor" in panel.columns
        and not panel["dividend"].isna().any()
        and not panel["split_factor"].isna().any()
    )
    chosen = method
    if method == "auto":
        chosen = "events" if has_events else "adj_close"
    if chosen == "events":
        if not has_events:
            msg = (
                "method='events' requiere dividend y split_factor completos; este panel "
                "no los tiene (proveedor sin eventos). Usa method='adj_close'."
            )
            raise DataQualityError(msg)
        grouped = panel.groupby(level="ticker", sort=False)
        prev = grouped["close"].shift(1)
        factors = (panel["close"] * panel["split_factor"] + panel["dividend"]) / prev
    else:
        if "adj_close" not in panel.columns or panel["adj_close"].isna().all():
            msg = (
                "method='adj_close' requiere la columna adj_close con valores; pide el "
                "panel con adjusted=True"
            )
            raise DataQualityError(msg)
        grouped = panel.groupby(level="ticker", sort=False)
        factors = panel["adj_close"] / grouped["adj_close"].shift(1)
    out = np.log(factors) if log else factors - 1.0
    return out.rename("total_return")


# ===========================================================================
# 3. Interoperabilidad con types.Bar
# ===========================================================================


def panel_to_bars(panel: pd.DataFrame) -> list[Bar]:
    """Convierte un panel canónico en una lista de `types.Bar`."""
    bars: list[Bar] = []
    for (ts, ticker), row in panel.iterrows():
        adj = row.get("adj_close")
        bars.append(
            Bar(
                ticker=str(ticker),
                ts=pd.Timestamp(ts).to_pydatetime(),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                adj_close=None if adj is None or pd.isna(adj) else float(adj),
            )
        )
    return bars


def bars_to_panel(bars: Sequence[Bar]) -> pd.DataFrame:
    """Convierte una secuencia de `types.Bar` en el panel canónico.

    `dividend` y `split_factor` quedan a NaN («desconocido»): `Bar` no
    transporta eventos corporativos.
    """
    if not bars:
        msg = "secuencia de barras vacía: nada que convertir"
        raise InsufficientHistory(msg)
    rows = {
        "open": [b.open for b in bars],
        "high": [b.high for b in bars],
        "low": [b.low for b in bars],
        "close": [b.close for b in bars],
        "volume": [b.volume for b in bars],
        "adj_close": [float("nan") if b.adj_close is None else b.adj_close for b in bars],
        "dividend": [float("nan")] * len(bars),
        "split_factor": [float("nan")] * len(bars),
    }
    index = pd.MultiIndex.from_arrays(
        [
            pd.DatetimeIndex([pd.Timestamp(b.ts).normalize() for b in bars]),
            [b.ticker for b in bars],
        ],
        names=["date", "ticker"],
    )
    return pd.DataFrame(rows, index=index).sort_index()[list(PRICE_COLUMNS)].astype("float64")


# ===========================================================================
# 4. Validación de calidad
# ===========================================================================


class Severity(StrEnum):
    """Gravedad de un hallazgo de calidad."""

    WARNING = "warning"
    """Sospechoso pero posible: se registra en el log y no interrumpe."""

    ERROR = "error"
    """Incompatible con un backtest honesto: `DataQualityError`."""


@dataclass(frozen=True, slots=True)
class QualityIssue:
    """Un hallazgo de la validación de un panel de precios."""

    code: str
    severity: Severity
    ticker: Ticker | None
    message: str
    count: int = 1
    examples: tuple[str, ...] = ()
    """Hasta tres fechas ISO de ejemplo, para localizar el problema sin abrir el panel."""

    def describe(self) -> str:
        where = self.ticker or "<panel>"
        sample = f" (p. ej. {', '.join(self.examples)})" if self.examples else ""
        return f"[{self.severity}] {where}: {self.code} x{self.count} — {self.message}{sample}"


@dataclass(slots=True)
class QualityReport:
    """Resultado de `validate_price_panel`."""

    issues: list[QualityIssue] = field(default_factory=list)
    n_rows: int = 0
    n_tickers: int = 0

    @property
    def errors(self) -> list[QualityIssue]:
        return [i for i in self.issues if i.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[QualityIssue]:
        return [i for i in self.issues if i.severity is Severity.WARNING]

    @property
    def ok(self) -> bool:
        """True si no hay ningún hallazgo de gravedad `error`."""
        return not self.errors

    def describe(self) -> str:
        head = f"panel de {self.n_rows} filas y {self.n_tickers} tickers"
        if not self.issues:
            return f"{head}: sin hallazgos"
        lines = [f"{head}: {len(self.errors)} errores, {len(self.warnings)} avisos"]
        lines.extend(f"  - {i.describe()}" for i in self.issues)
        return "\n".join(lines)

    def raise_if_errors(self) -> None:
        """Lanza `DataQualityError` con el detalle si hay errores."""
        if self.errors:
            detail = "\n".join(f"  - {i.describe()}" for i in self.errors)
            msg = f"el panel de precios no supera la validación de calidad:\n{detail}"
            raise DataQualityError(msg)

    def log_warnings(self, log: logging.Logger | None = None) -> None:
        target = log or logger
        for issue in self.warnings:
            target.warning("calidad de precios: %s", issue.describe())


# Cocientes de split habituales en renta variable USA. Un salto de precio que
# coincide con uno de ellos y no viene acompañado de un split registrado es la
# firma clásica de un split sin ajustar (Ince y Porter, 2006).
_SPLIT_RATIOS: tuple[float, ...] = (1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 20.0)


def _looks_like_split(factor: float, tolerance: float) -> bool:
    if not np.isfinite(factor) or factor <= 0:
        return False
    return any(
        abs(factor / r - 1.0) <= tolerance or abs(factor * r - 1.0) <= tolerance
        for r in _SPLIT_RATIOS
    )


def _run_lengths(mask: np.ndarray) -> np.ndarray:
    """Longitudes de las rachas de True en `mask` (vector booleano)."""
    if not mask.any():
        return np.array([], dtype=int)
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    delta = np.diff(padded)
    starts = np.where(delta == 1)[0]
    ends = np.where(delta == -1)[0]
    return ends - starts


def _sample_dates(index: pd.Index, mask: np.ndarray, k: int = 3) -> tuple[str, ...]:
    picked = index[np.asarray(mask, dtype=bool)][:k]
    return tuple(pd.Timestamp(d).date().isoformat() for d in picked)


def validate_price_panel(
    panel: pd.DataFrame,
    *,
    calendar: TradingCalendar | None = None,
    max_jump: float = 0.35,
    hard_jump: float = 3.0,
    split_ratio_tolerance: float = 0.06,
    zero_volume_warn_run: int = 5,
    zero_volume_error_run: int = 21,
    max_gap_fraction: float = 0.10,
    raise_on_error: bool = False,
) -> QualityReport:
    """Valida un panel canónico de precios y clasifica los hallazgos por gravedad.

    Filtros, en la línea de Ince y Porter (2006):

    - **precios no positivos** o NaN en OHLC -> ``error``;
    - **OHLC incoherente** (``high < low``, cierre/apertura fuera de
      ``[low, high]`` más allá de una tolerancia de redondeo) -> ``error``;
    - **saltos**: se calcula el factor de retorno total (usando `dividend` y
      `split_factor` cuando existen, de modo que un split *bien registrado* no
      salta). Un factor que coincide con un cociente de split típico y ese día
      el proveedor afirma "no hubo split" -> ``error``
      («split sin ajustar»); si el proveedor **no informa** de splits (columna
      NaN) el mismo hallazgo es ``warning``, porque puede ser un split real
      sin metadatos. Factor más allá de ``1 + hard_jump`` (o su inverso) ->
      ``error``; movimiento mayor que `max_jump` -> ``warning``;
    - **volumen cero prolongado**: racha >= `zero_volume_warn_run` sesiones ->
      ``warning``; >= `zero_volume_error_run` -> ``error`` (un mes sin negociar
      dentro del S&P 500 es un error de datos, no iliquidez);
    - **huecos frente al calendario**: sesiones NYSE ausentes entre la primera
      y la última barra de cada símbolo -> ``warning``; más del
      `max_gap_fraction` de la ventana -> ``error``. Barras fechadas en días
      **no bursátiles** -> ``error`` (delatan un error de zona horaria);
    - **duplicados** en `(date, ticker)` y un índice sin ordenar -> ``error``
      (rompen `pit.asof_join`).

    Devuelve un `QualityReport`; con ``raise_on_error=True`` lanza
    `DataQualityError` si hay errores. Los avisos nunca lanzan: quedan en el
    informe y en el log de quien llame a `log_warnings`.
    """
    if not isinstance(panel.index, pd.MultiIndex) or list(panel.index.names) != ["date", "ticker"]:
        msg = "validate_price_panel espera el panel canónico con MultiIndex (date, ticker)"
        raise DataQualityError(msg)
    for col in ("open", "high", "low", "close", "volume"):
        if col not in panel.columns:
            msg = f"el panel no tiene la columna obligatoria {col!r}"
            raise DataQualityError(msg)

    report = QualityReport(
        n_rows=len(panel),
        n_tickers=panel.index.get_level_values("ticker").nunique(),
    )
    add = report.issues.append

    dup = panel.index.duplicated()
    if dup.any():
        add(
            QualityIssue(
                "duplicate_rows",
                Severity.ERROR,
                None,
                "hay claves (date, ticker) repetidas",
                count=int(dup.sum()),
            )
        )
    if not panel.index.is_monotonic_increasing:
        add(
            QualityIssue(
                "unsorted_index",
                Severity.ERROR,
                None,
                "el índice no está ordenado; el contrato exige (date, ticker) ordenado",
            )
        )

    cal = calendar
    for ticker, group in panel.groupby(level="ticker", sort=False):
        sub = group.droplevel("ticker").sort_index()
        dates = sub.index
        px = sub[["open", "high", "low", "close"]]

        # --- precios no positivos o ausentes -------------------------------
        bad_px = (px <= 0).any(axis=1) | px.isna().any(axis=1)
        if bad_px.any():
            add(
                QualityIssue(
                    "non_positive_price",
                    Severity.ERROR,
                    str(ticker),
                    "precios <= 0 o NaN en OHLC",
                    count=int(bad_px.sum()),
                    examples=_sample_dates(dates, bad_px.to_numpy()),
                )
            )

        # --- coherencia OHLC ------------------------------------------------
        tol = 1e-4
        with np.errstate(invalid="ignore"):
            incoherent = (
                (sub["high"] < sub["low"])
                | (sub["close"] > sub["high"] * (1 + tol))
                | (sub["close"] < sub["low"] * (1 - tol))
                | (sub["open"] > sub["high"] * (1 + tol))
                | (sub["open"] < sub["low"] * (1 - tol))
            ) & ~bad_px
        if incoherent.any():
            add(
                QualityIssue(
                    "ohlc_inconsistent",
                    Severity.ERROR,
                    str(ticker),
                    "high/low no envuelven a open/close",
                    count=int(incoherent.sum()),
                    examples=_sample_dates(dates, incoherent.to_numpy()),
                )
            )

        # --- saltos ---------------------------------------------------------
        close = sub["close"]
        split_known = "split_factor" in sub.columns and sub["split_factor"].notna().all()
        split_eff = (
            sub["split_factor"].astype(float)
            if split_known
            else pd.Series(1.0, index=dates)
        )
        div_eff = (
            sub["dividend"].astype(float).fillna(0.0)
            if "dividend" in sub.columns
            else pd.Series(0.0, index=dates)
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            factors = (close * split_eff + div_eff) / close.shift(1)
        fvals = factors.to_numpy()
        valid = np.isfinite(fvals)
        valid[0:1] = False
        splitlike = np.array(
            [valid[i] and _looks_like_split(fvals[i], split_ratio_tolerance)
             and not (0.999 < fvals[i] < 1.001)
             for i in range(len(fvals))]
        )
        hard_hi, hard_lo = 1.0 + hard_jump, 1.0 / (1.0 + hard_jump)
        hard = valid & ~splitlike & ((fvals > hard_hi) | (fvals < hard_lo))
        soft = (
            valid
            & ~splitlike
            & ~hard
            & (np.abs(fvals - 1.0) > max_jump)
        )
        if splitlike.any():
            severity = Severity.ERROR if split_known else Severity.WARNING
            detail = (
                "salto igual a un cociente de split sin split registrado ese día"
                if split_known
                else "salto igual a un cociente de split; el proveedor no informa de splits"
            )
            add(
                QualityIssue(
                    "unadjusted_split" if split_known else "split_like_jump",
                    severity,
                    str(ticker),
                    detail,
                    count=int(splitlike.sum()),
                    examples=_sample_dates(dates, splitlike),
                )
            )
        if hard.any():
            add(
                QualityIssue(
                    "impossible_jump",
                    Severity.ERROR,
                    str(ticker),
                    f"retorno total fuera de [{hard_lo:.3f}, {hard_hi:.1f}] en un día",
                    count=int(hard.sum()),
                    examples=_sample_dates(dates, hard),
                )
            )
        if soft.any():
            add(
                QualityIssue(
                    "large_move",
                    Severity.WARNING,
                    str(ticker),
                    f"movimiento diario > {max_jump:.0%} (posible, pero revisable)",
                    count=int(soft.sum()),
                    examples=_sample_dates(dates, soft),
                )
            )

        # --- volumen cero prolongado ---------------------------------------
        vol = sub["volume"].to_numpy(dtype=float)
        zero = np.isfinite(vol) & (vol <= 0)
        runs = _run_lengths(zero)
        if len(runs) and runs.max() >= zero_volume_warn_run:
            longest = int(runs.max())
            severity = (
                Severity.ERROR if longest >= zero_volume_error_run else Severity.WARNING
            )
            add(
                QualityIssue(
                    "zero_volume_run",
                    severity,
                    str(ticker),
                    f"racha de {longest} sesiones con volumen cero",
                    count=longest,
                    examples=_sample_dates(dates, zero),
                )
            )

        # --- huecos frente al calendario ------------------------------------
        if cal is None:
            cal = get_calendar()
        try:
            expected = cal.sessions(dates[0].date(), dates[-1].date())
        except CalendarError:
            expected = None
        if expected is not None and len(expected):
            extra = dates.difference(expected)
            if len(extra):
                add(
                    QualityIssue(
                        "not_a_session",
                        Severity.ERROR,
                        str(ticker),
                        "barras fechadas en días no bursátiles (¿zona horaria?)",
                        count=len(extra),
                        examples=tuple(
                            pd.Timestamp(d).date().isoformat() for d in extra[:3]
                        ),
                    )
                )
            missing = expected.difference(dates)
            if len(missing):
                frac = len(missing) / len(expected)
                severity = Severity.ERROR if frac > max_gap_fraction else Severity.WARNING
                add(
                    QualityIssue(
                        "calendar_gaps",
                        severity,
                        str(ticker),
                        f"faltan {len(missing)} de {len(expected)} sesiones ({frac:.1%})",
                        count=len(missing),
                        examples=tuple(
                            pd.Timestamp(d).date().isoformat() for d in missing[:3]
                        ),
                    )
                )

    if raise_on_error:
        report.raise_if_errors()
    return report


# ===========================================================================
# 5. Interfaz común y base de proveedores
# ===========================================================================


@runtime_checkable
class PriceProvider(Protocol):
    """Contrato de un proveedor de precios (contrato §2, `data.market`)."""

    name: str
    kinds: tuple[str, ...]

    def available(self) -> bool: ...

    def get_bars(
        self,
        tickers: Ticker | Sequence[Ticker],
        start: DateLike,
        end: DateLike,
        adjusted: bool = True,
    ) -> pd.DataFrame:
        """Panel canónico `(date, ticker)` con las columnas `PRICE_COLUMNS`."""
        ...


class _TickerMissing(EarningsAlphaError):
    """Señal interna: el proveedor no tiene datos de este símbolo.

    No forma parte del contrato público; existe para que la envoltura de caché
    pueda distinguir "sin datos" (no se cachea, se reintenta otro día) de un
    panel válido, sin cachear jamás un vacío (contrato §0.3).
    """

    def __init__(self, provider: str, ticker: str) -> None:
        self.ticker = ticker
        super().__init__(f"{provider}: sin datos para {ticker}")


class PriceProviderBase(BaseProvider):
    """Base común de los adaptadores de precios.

    Aporta: normalización de tickers y fechas, bucle por símbolo con caché
    opcional (`DiskCache`, claves ``(prices, proveedor, hash(params))``),
    política explícita ante símbolos sin datos y validación de calidad del
    panel final. Las subclases solo implementan `_fetch_bars` (o `_fetch_all`
    si la API admite lotes, como Alpaca).

    Política de fallo (contrato §0.3):

    - sin credenciales -> `ProviderUnavailable` **antes** de tocar la red;
    - ningún símbolo con datos -> `InsufficientHistory` (jamás un panel vacío);
    - símbolos sin datos con ``on_missing="raise"`` (por defecto) ->
      `InsufficientHistory` enumerándolos; con ``"warn"`` se registran en el
      log y se sigue con el resto;
    - panel con errores de calidad -> `DataQualityError` (con `validate=True`).
    """

    name = "prices_base"
    kinds: tuple[str, ...] = (PRICES_KIND,)

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        http: HttpClient | None = None,
        clock: Clock | None = None,
        cache: DiskCache | None = None,
        calendar: TradingCalendar | None = None,
        validate: bool = True,
    ) -> None:
        super().__init__(settings=settings, http=http, clock=clock)
        self.cache = cache
        self._calendar = calendar
        self.validate_default = bool(validate)

    # -- utilidades ----------------------------------------------------------

    @property
    def calendar(self) -> TradingCalendar:
        if self._calendar is None:
            self._calendar = get_calendar()
        return self._calendar

    def _credential(self, key: str) -> str:
        value = self.settings.env(key)
        if not value:
            raise ProviderUnavailable(
                self.name, "falta la credencial en el entorno", missing_env=[key]
            )
        return value

    def _require_credentials(self) -> None:
        missing = self.missing_env()
        if missing:
            raise ProviderUnavailable(
                self.name,
                "faltan credenciales para pedir precios",
                missing_env=missing,
            )

    # -- API pública ---------------------------------------------------------

    def get_bars(
        self,
        tickers: Ticker | Sequence[Ticker],
        start: DateLike,
        end: DateLike,
        adjusted: bool = True,
        *,
        validate: bool | None = None,
        on_missing: Literal["raise", "warn"] = "raise",
    ) -> pd.DataFrame:
        """Panel OHLCV canónico para `tickers` en `[start, end]` (inclusive).

        Con ``adjusted=True`` (por defecto) la columna `adj_close` viene
        rellena (retorno total, ancla en el fin de la ventana); con ``False``
        queda a NaN y, en los proveedores que lo permiten, se ahorran las
        peticiones de eventos corporativos.
        """
        if on_missing not in ("raise", "warn"):
            msg = f"on_missing debe ser 'raise' o 'warn'; recibido {on_missing!r}"
            raise ConfigError(msg)
        symbols = _normalize_tickers(tickers)
        s, e = _check_window(start, end)
        self._require_credentials()

        frames, missing = self._fetch_all(symbols, s, e, bool(adjusted))
        if missing:
            detail = (
                f"{self.name}: sin datos para {len(missing)} de {len(symbols)} símbolos "
                f"en [{s.isoformat()}, {e.isoformat()}]: {', '.join(sorted(missing))}"
            )
            if not frames:
                raise InsufficientHistory(detail)
            if on_missing == "raise":
                raise InsufficientHistory(
                    detail + ". Usa on_missing='warn' para continuar sin ellos."
                )
            logger.warning("%s (se continúa con el resto)", detail)
        if not frames:
            msg = (
                f"{self.name}: ningún símbolo devolvió datos en "
                f"[{s.isoformat()}, {e.isoformat()}]"
            )
            raise InsufficientHistory(msg)

        panel = _assemble_panel(frames)
        do_validate = self.validate_default if validate is None else bool(validate)
        if do_validate:
            report = validate_price_panel(panel, calendar=self._calendar)
            report.log_warnings(logger)
            report.raise_if_errors()
        return panel

    # -- ganchos para subclases ---------------------------------------------

    def _fetch_all(
        self, tickers: list[Ticker], start: date, end: date, adjusted: bool
    ) -> tuple[dict[Ticker, pd.DataFrame], list[Ticker]]:
        """Bucle por símbolo con caché. Las APIs por lotes lo sobrescriben."""
        frames: dict[Ticker, pd.DataFrame] = {}
        missing: list[Ticker] = []
        for ticker in tickers:
            frame = self._fetch_one_cached(ticker, start, end, adjusted)
            if frame is None or frame.empty:
                missing.append(ticker)
            else:
                frames[ticker] = frame
        return frames, missing

    def _fetch_one_cached(
        self, ticker: Ticker, start: date, end: date, adjusted: bool
    ) -> pd.DataFrame | None:
        if self.cache is None:
            return self._fetch_bars(ticker, start, end, adjusted)
        params = self._cache_params(ticker, start, end, adjusted)

        def loader() -> pd.DataFrame:
            frame = self._fetch_bars(ticker, start, end, adjusted)
            if frame is None or frame.empty:
                # No se cachea un vacío: mañana el proveedor puede tenerlo.
                raise _TickerMissing(self.name, ticker)
            return frame

        try:
            return self.cache.fetch(PRICES_KIND, self.name, params, loader)
        except _TickerMissing:
            return None

    def _cache_params(
        self, ticker: Ticker, start: date, end: date, adjusted: bool
    ) -> dict[str, Any]:
        """Clave de caché. `end` activa la inmutabilidad automática de la caché
        cuando la ventana ya está cerrada (`ImmutabilityPolicy`)."""
        return {"ticker": ticker, "start": start, "end": end, "adjusted": bool(adjusted)}

    def _fetch_bars(
        self, ticker: Ticker, start: date, end: date, adjusted: bool
    ) -> pd.DataFrame | None:
        """Frame por-ticker indexado por fecha, o `None` si el símbolo no existe."""
        raise NotImplementedError


# ===========================================================================
# 6. SyntheticProvider — envuelve SyntheticMarket, siempre disponible
# ===========================================================================


class SyntheticProvider(PriceProviderBase):
    """Precios del generador sintético del repo. Sin red, sin credenciales.

    Envuelve `data.synthetic.SyntheticMarket` con la interfaz `get_bars`, de
    modo que cualquier módulo puede ejecutarse de extremo a extremo sin
    conectividad (contrato §0.5). Se registra con prioridad 0: solo gana cuando
    no hay ningún proveedor real disponible.

    El panel sintético ya trae `dividend` y `split_factor` coherentes con
    `adj_close` (misma fórmula CRSP que `compute_adj_close`), así que sirve de
    verdad-terreno para probar el ajuste de los adaptadores reales.
    """

    name = "synthetic"
    kinds: tuple[str, ...] = (PRICES_KIND, DataKind.CORPORATE_ACTIONS.value)

    def __init__(
        self,
        market: SyntheticMarket | None = None,
        *,
        settings: Settings | None = None,
        calendar: TradingCalendar | None = None,
        validate: bool = True,
        **market_kwargs: Any,
    ) -> None:
        cfg = settings or get_settings()
        super().__init__(settings=cfg, calendar=calendar, validate=validate)
        self._market = market
        self._market_kwargs = dict(market_kwargs)

    @property
    def market(self) -> SyntheticMarket:
        """Mercado sintético perezoso (construirlo cuesta; solo si se usa)."""
        if self._market is None:
            kwargs = {"seed": self.settings.seed, **self._market_kwargs}
            self._market = SyntheticMarket(**kwargs)
        return self._market

    def available(self) -> bool:
        """Siempre: no hay credenciales ni red de por medio."""
        return True

    def _fetch_all(
        self, tickers: list[Ticker], start: date, end: date, adjusted: bool
    ) -> tuple[dict[Ticker, pd.DataFrame], list[Ticker]]:
        known = set(self.market.tickers)
        wanted = [t for t in tickers if t in known]
        missing = [t for t in tickers if t not in known]
        if not wanted:
            return {}, missing
        panel = self.market.prices(tickers=wanted, start=start, end=end)
        frames: dict[Ticker, pd.DataFrame] = {}
        for ticker in wanted:
            try:
                sub = panel.xs(ticker, level="ticker")
            except KeyError:
                missing.append(ticker)
                continue
            frames[ticker] = _bar_frame(
                pd.DatetimeIndex(sub.index, name="date"),
                {
                    "open": sub["open"],
                    "high": sub["high"],
                    "low": sub["low"],
                    "close": sub["close"],
                    "volume": sub["volume"],
                    "adj_close": sub["adj_close"] if adjusted else np.nan,
                    "dividend": sub["dividend"],
                    "split_factor": sub["split_factor"],
                },
            )
        return frames, missing

    def _fetch_bars(
        self, ticker: Ticker, start: date, end: date, adjusted: bool
    ) -> pd.DataFrame | None:  # pragma: no cover - _fetch_all lo sustituye
        frames, _ = self._fetch_all([ticker], start, end, adjusted)
        return frames.get(ticker)


# ===========================================================================
# 7. YFinanceProvider — Yahoo Finance v8 chart (no oficial)
# ===========================================================================


class YFinanceProvider(PriceProviderBase):
    """Yahoo Finance vía el endpoint no oficial ``v8/finance/chart``.

    Endpoint (documentado en `docs/research/data_sources.md` §3.2)::

        GET https://query2.finance.yahoo.com/v8/finance/chart/{symbol}
            ?period1=&period2=&interval=1d&events=div,split

    Respuesta: ``chart.result[0]`` con ``timestamp`` (epoch s),
    ``indicators.quote[0].{open,high,low,close,volume}``,
    ``indicators.adjclose[0].adjclose`` y ``events.{dividends,splits}``.
    Un símbolo desconocido devuelve 404 con ``chart.error.code = "Not Found"``.

    **Advertencia PIT** (data_sources.md §3.2): Yahoo entrega los precios
    **retro-ajustados por splits con toda la historia conocida hoy**, y los
    dividendos en la misma base. Este adaptador **des-ajusta** usando los
    eventos de split de la propia respuesta, reconstruyendo el cierre tal y
    como se negoció: si el split cae dentro de la ventana pedida, `close`,
    `volume` y `dividend` vuelven a su base original y `split_factor` registra
    el evento. Limitación irreducible: un split *posterior* a `end` reescala la
    ventana entera de forma uniforme (los retornos no se ven afectados, los
    niveles sí); por eso, y por el sesgo de supervivencia de Yahoo (los
    deslistados desaparecen), este proveedor se registra con prioridad baja.

    `adj_close` se recalcula localmente con `compute_adj_close` (ancla en el
    fin de la ventana) en vez de usar el ``adjclose`` de Yahoo, cuyo ancla es
    "hoy" y cambia con cada split/dividendo futuro: la misma petición debe
    devolver los mismos números siempre.
    """

    name = "yfinance"
    kinds: tuple[str, ...] = (PRICES_KIND, DataKind.CORPORATE_ACTIONS.value)
    BASE = "https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"

    @staticmethod
    def _symbol(ticker: Ticker) -> str:
        """Símbolo Yahoo: clases de acción con guion (``BRK.B`` -> ``BRK-B``)."""
        return ticker.replace(".", "-")

    def _fetch_bars(
        self, ticker: Ticker, start: date, end: date, adjusted: bool
    ) -> pd.DataFrame | None:
        period1 = int(datetime.combine(start, time.min, tzinfo=UTC).timestamp())
        period2 = int(
            datetime.combine(end + timedelta(days=1), time.min, tzinfo=UTC).timestamp()
        )
        payload = self.http.get_json(
            self.BASE.format(symbol=self._symbol(ticker)),
            params={
                "period1": period1,
                "period2": period2,
                "interval": "1d",
                "events": "div,split",
                "includeAdjustedClose": "true",
            },
            allow_status=(404,),
        )
        chart = payload.get("chart") if isinstance(payload, dict) else None
        if not isinstance(chart, dict):
            msg = f"yfinance: respuesta sin bloque 'chart' para {ticker}"
            raise DataQualityError(msg)
        results = chart.get("result")
        if chart.get("error") or not results:
            logger.info("yfinance sin datos para %s: %s", ticker, chart.get("error"))
            return None
        r0 = results[0]
        timestamps = r0.get("timestamp") or []
        if not timestamps:
            return None
        indicators = r0.get("indicators") or {}
        quote = (indicators.get("quote") or [{}])[0]
        tz = (r0.get("meta") or {}).get("exchangeTimezoneName") or _NY_TZ
        dates = _epochs_to_dates(timestamps, "s", tz)

        frame = pd.DataFrame(
            {
                col: pd.to_numeric(pd.Series(quote.get(col) or [], dtype="object"),
                                   errors="coerce").to_numpy()
                if quote.get(col)
                else np.full(len(dates), np.nan)
                for col in ("open", "high", "low", "close", "volume")
            },
            index=dates,
        )
        # Yahoo intercala filas nulas (sesiones sin consolidar) y duplica la
        # última barra cuando el mercado está abierto: se limpian ambas.
        frame = frame[frame["close"].notna()]
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        if frame.empty:
            return None

        events = r0.get("events") or {}
        splits: list[tuple[pd.Timestamp, float]] = []
        for item in (events.get("splits") or {}).values():
            num = float(item.get("numerator", 0) or 0)
            den = float(item.get("denominator", 0) or 0)
            if num > 0 and den > 0:
                day = _epochs_to_dates([item["date"]], "s", tz)[0]
                splits.append((day, num / den))

        # Des-ajuste: antes de cada split la serie de Yahoo está dividida por
        # el factor; se multiplica de vuelta para recuperar la base original.
        multiplier = pd.Series(1.0, index=frame.index)
        for day, factor in splits:
            multiplier[multiplier.index < day] *= factor
        for col in ("open", "high", "low", "close"):
            frame[col] = frame[col] * multiplier
        frame["volume"] = frame["volume"] / multiplier

        dividend = pd.Series(0.0, index=frame.index)
        for item in (events.get("dividends") or {}).values():
            amount = float(item.get("amount", 0) or 0)
            if amount <= 0:
                continue
            day = _epochs_to_dates([item["date"]], "s", tz)[0]
            pos = frame.index.searchsorted(day, side="left")
            if pos < len(frame.index):
                target = frame.index[pos]
                dividend[target] += amount * float(multiplier[target])
        split_col = pd.Series(1.0, index=frame.index)
        for day, factor in splits:
            pos = frame.index.searchsorted(day, side="left")
            if pos < len(frame.index):
                split_col.iloc[pos] = factor

        frame = _clip_window(frame, start, end)
        if frame.empty:
            return None
        dividend = dividend.loc[frame.index]
        split_col = split_col.loc[frame.index]
        adj = (
            compute_adj_close(frame["close"], dividend, split_col)
            if adjusted
            else np.nan
        )
        return _bar_frame(
            pd.DatetimeIndex(frame.index),
            {
                "open": frame["open"],
                "high": frame["high"],
                "low": frame["low"],
                "close": frame["close"],
                "volume": frame["volume"],
                "adj_close": adj,
                "dividend": dividend,
                "split_factor": split_col,
            },
        )


# ===========================================================================
# 8. PolygonProvider — aggs v2 sin ajustar + splits/dividends v3
# ===========================================================================


class PolygonProvider(PriceProviderBase):
    """Polygon.io: barras diarias crudas + eventos corporativos fechados.

    Endpoints (verificados contra la documentación pública, 2026-08; véase
    `docs/research/data_sources.md` §3.2)::

        GET /v2/aggs/ticker/{t}/range/1/day/{from}/{to}?adjusted=false&sort=asc
            -> {"results": [{"t": epoch_ms, "o","h","l","c","v","vw","n"}, ...],
                "resultsCount": N, "status": "OK"|"DELAYED", "next_url": ...}
        GET /v3/reference/splits?ticker=&execution_date.gte=&execution_date.lte=
            -> {"results": [{"execution_date", "split_from", "split_to"}, ...]}
        GET /v3/reference/dividends?ticker=&ex_dividend_date.gte=&lte=
            -> {"results": [{"cash_amount", "ex_dividend_date", ...}, ...]}

    ``adjusted=false`` da el cierre **tal y como se negoció** —lo que este repo
    prefiere (§3.1)— y los endpoints de referencia dan los eventos con fecha,
    con lo que el ajuste PIT se reconstruye localmente (`compute_adj_close`).
    El `split_factor` del panel es ``split_to / split_from`` (4:1 -> 4.0).
    La paginación usa ``next_url`` tal cual (ya lleva el cursor); la
    autenticación va por cabecera ``Authorization: Bearer`` para que la clave
    no acabe en URLs de logs ni en claves de caché.

    Plan gratuito: 5 req/min (`PROVIDER_RATE_LIMITS`); una carga de 500
    símbolos con ajuste son ~1 500 peticiones: horas. Con plan de pago, subir
    `EARNINGS_ALPHA_RPS_POLYGON`.
    """

    name = "polygon"
    kinds: tuple[str, ...] = (PRICES_KIND, DataKind.CORPORATE_ACTIONS.value)
    BASE = "https://api.polygon.io"

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._credential('POLYGON_API_KEY')}"}

    def _paged_json(
        self, url: str, params: dict[str, Any] | None, headers: dict[str, str]
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        next_url: str | None = url
        next_params: dict[str, Any] | None = params
        while next_url:
            payload = self.http.get_json(
                next_url, params=next_params, headers=headers, allow_status=(404,)
            )
            if not isinstance(payload, dict):
                msg = f"polygon: respuesta no-objeto en {next_url}"
                raise DataQualityError(msg)
            status = payload.get("status")
            if status == "ERROR":
                msg = f"polygon: status=ERROR — {payload.get('error') or payload.get('message')}"
                raise DataQualityError(msg)
            rows.extend(payload.get("results") or [])
            next_url = payload.get("next_url")
            next_params = None  # next_url ya incorpora el cursor
        return rows

    def _fetch_bars(
        self, ticker: Ticker, start: date, end: date, adjusted: bool
    ) -> pd.DataFrame | None:
        headers = self._auth_headers()
        # Polygon usa el punto en clases de acción ("BRK.B"): sin remapeo.
        url = (
            f"{self.BASE}/v2/aggs/ticker/{ticker}/range/1/day/"
            f"{start.isoformat()}/{end.isoformat()}"
        )
        rows = self._paged_json(
            url, {"adjusted": "false", "sort": "asc", "limit": 50000}, headers
        )
        if not rows:
            return None
        dates = _epochs_to_dates([r["t"] for r in rows], "ms", _NY_TZ)
        frame = _bar_frame(
            dates,
            {
                "open": [r.get("o") for r in rows],
                "high": [r.get("h") for r in rows],
                "low": [r.get("l") for r in rows],
                "close": [r.get("c") for r in rows],
                "volume": [r.get("v") for r in rows],
            },
        )
        frame = _clip_window(frame, start, end)
        if frame.empty:
            return None
        if not adjusted:
            return frame

        dividend = pd.Series(0.0, index=frame.index)
        split_col = pd.Series(1.0, index=frame.index)
        for item in self._paged_json(
            f"{self.BASE}/v3/reference/dividends",
            {
                "ticker": ticker,
                "ex_dividend_date.gte": start.isoformat(),
                "ex_dividend_date.lte": end.isoformat(),
                "limit": 1000,
            },
            headers,
        ):
            amount = float(item.get("cash_amount", 0) or 0)
            ex_date = item.get("ex_dividend_date")
            if amount <= 0 or not ex_date:
                continue
            pos = frame.index.searchsorted(pd.Timestamp(ex_date), side="left")
            if pos < len(frame.index):
                dividend.iloc[pos] += amount
        for item in self._paged_json(
            f"{self.BASE}/v3/reference/splits",
            {
                "ticker": ticker,
                "execution_date.gte": start.isoformat(),
                "execution_date.lte": end.isoformat(),
                "limit": 1000,
            },
            headers,
        ):
            frm = float(item.get("split_from", 0) or 0)
            to = float(item.get("split_to", 0) or 0)
            when = item.get("execution_date")
            if frm <= 0 or to <= 0 or not when:
                continue
            pos = frame.index.searchsorted(pd.Timestamp(when), side="left")
            if pos < len(frame.index):
                split_col.iloc[pos] *= to / frm

        out = frame.copy()
        out["dividend"] = dividend
        out["split_factor"] = split_col
        out["adj_close"] = compute_adj_close(out["close"], dividend, split_col)
        return out[list(PRICE_COLUMNS)]


# ===========================================================================
# 9. TiingoProvider — EOD con divCash y splitFactor por día
# ===========================================================================


class TiingoProvider(PriceProviderBase):
    """Tiingo EOD: cierre crudo + ``divCash``/``splitFactor`` diarios.

    Endpoint (forma verificada contra la documentación pública, 2026-08;
    `docs/research/data_sources.md` §3.2)::

        GET https://api.tiingo.com/tiingo/daily/{ticker}/prices
            ?startDate=&endDate=&format=json&resampleFreq=daily
        Authorization: Token <TIINGO_API_KEY>

    Respuesta: lista de objetos ``{"date": "2020-08-31T00:00:00.000Z",
    "open","high","low","close","volume", "adjOpen",...,"adjClose","adjVolume",
    "divCash", "splitFactor"}``. Los campos sin prefijo son **as-traded** y
    ``divCash``/``splitFactor`` llevan fecha, que es exactamente la propiedad
    que hace a Tiingo la mejor opción PIT del nivel de ~50 USD/mes (§3.2):
    permite reconstruir el ajuste tal y como era en ``t``.

    ``date`` etiqueta la *fecha de negociación* a medianoche UTC: se toma el
    literal, **sin** convertir a hora de Nueva York (hacerlo restaría un día).
    `adj_close` se recalcula con `compute_adj_close` (ancla en el fin de la
    ventana) en vez de usar el ``adjClose`` de Tiingo, anclado a su historia
    completa: mismos cocientes, niveles reproducibles.

    Un ticker desconocido responde 404 -> símbolo sin datos.
    """

    name = "tiingo"
    kinds: tuple[str, ...] = (PRICES_KIND, DataKind.CORPORATE_ACTIONS.value)
    BASE = "https://api.tiingo.com"

    @staticmethod
    def _symbol(ticker: Ticker) -> str:
        """Tiingo usa guion en clases de acción (``BRK.B`` -> ``BRK-B``)."""
        return ticker.replace(".", "-")

    def _fetch_bars(
        self, ticker: Ticker, start: date, end: date, adjusted: bool
    ) -> pd.DataFrame | None:
        response = self.http.get(
            f"{self.BASE}/tiingo/daily/{self._symbol(ticker)}/prices",
            params={
                "startDate": start.isoformat(),
                "endDate": end.isoformat(),
                "format": "json",
                "resampleFreq": "daily",
            },
            headers={
                "Authorization": f"Token {self._credential('TIINGO_API_KEY')}",
                "Accept": "application/json",
            },
            allow_status=(404,),
        )
        if response.status_code == 404:
            return None
        rows = response.json()
        if isinstance(rows, dict):
            detail = str(rows.get("detail", rows))
            if "not found" in detail.lower():
                return None
            msg = f"tiingo: respuesta inesperada para {ticker}: {detail[:200]}"
            raise DataQualityError(msg)
        if not rows:
            return None
        dates = _iso_to_dates([r["date"] for r in rows], tz=None)
        frame = _bar_frame(
            dates,
            {
                "open": [r.get("open") for r in rows],
                "high": [r.get("high") for r in rows],
                "low": [r.get("low") for r in rows],
                "close": [r.get("close") for r in rows],
                "volume": [r.get("volume") for r in rows],
                "dividend": [r.get("divCash", 0.0) for r in rows],
                "split_factor": [r.get("splitFactor", 1.0) for r in rows],
            },
        )
        frame = _clip_window(frame, start, end)
        if frame.empty:
            return None
        if adjusted:
            frame["adj_close"] = compute_adj_close(
                frame["close"], frame["dividend"], frame["split_factor"]
            )
        return frame[list(PRICE_COLUMNS)]


# ===========================================================================
# 10. StooqProvider — CSV ya ajustado, segunda opinión gratuita
# ===========================================================================


class StooqProvider(PriceProviderBase):
    """Stooq: CSV diario sin credencial, precios **ya ajustados**.

    Endpoint (`docs/research/data_sources.md` §3.2)::

        GET https://stooq.com/q/d/l/?s={symbol}.us&i=d&d1=YYYYMMDD&d2=YYYYMMDD
        -> "Date,Open,High,Low,Close,Volume" (o el literal "No data")

    Stooq sirve la serie ajustada por splits y dividendos **sin** publicar los
    factores, así que el cierre como-se-negoció no es recuperable: `close` y
    `adj_close` son la misma serie ajustada, y `dividend`/`split_factor` van a
    NaN («desconocido»). Pedir ``adjusted=False`` lanza `DataQualityError`: es
    una capacidad que esta fuente no tiene, y fingirla devolviendo la serie
    ajustada como si fuera cruda es exactamente el error de §3.1.

    Veredicto del informe: excelente **segunda opinión** para detectar
    discrepancias del proveedor primario; nunca fuente primaria.
    """

    name = "stooq"
    kinds: tuple[str, ...] = (PRICES_KIND,)
    BASE = "https://stooq.com/q/d/l/"

    @staticmethod
    def _symbol(ticker: Ticker) -> str:
        """Símbolo Stooq: minúsculas, guion y sufijo de mercado (``brk-b.us``)."""
        return f"{ticker.lower().replace('.', '-')}.us"

    def get_bars(
        self,
        tickers: Ticker | Sequence[Ticker],
        start: DateLike,
        end: DateLike,
        adjusted: bool = True,
        *,
        validate: bool | None = None,
        on_missing: Literal["raise", "warn"] = "raise",
    ) -> pd.DataFrame:
        if not adjusted:
            msg = (
                "stooq solo distribuye precios ya ajustados y sin factores: el cierre "
                "como-se-negoció (adjusted=False) no es reconstruible. Usa tiingo o "
                "polygon para cierres crudos."
            )
            raise DataQualityError(msg)
        return super().get_bars(
            tickers, start, end, adjusted, validate=validate, on_missing=on_missing
        )

    def _fetch_bars(
        self, ticker: Ticker, start: date, end: date, adjusted: bool
    ) -> pd.DataFrame | None:
        response = self.http.get(
            self.BASE,
            params={
                "s": self._symbol(ticker),
                "d1": start.strftime("%Y%m%d"),
                "d2": end.strftime("%Y%m%d"),
                "i": "d",
            },
        )
        text = response.text.strip()
        if (
            not text
            or text.lower().startswith("no data")
            or "<html" in text[:200].lower()
        ):
            return None
        table = pd.read_csv(io.StringIO(text))
        table.columns = [str(c).strip().lower() for c in table.columns]
        if "date" not in table.columns or "close" not in table.columns:
            msg = f"stooq: CSV sin columnas Date/Close para {ticker}: {list(table.columns)}"
            raise DataQualityError(msg)
        dates = pd.DatetimeIndex(pd.to_datetime(table["date"]), name="date").normalize()
        frame = _bar_frame(
            dates,
            {
                "open": table.get("open"),
                "high": table.get("high"),
                "low": table.get("low"),
                "close": table.get("close"),
                "volume": table.get("volume"),
            },
        )
        frame = _clip_window(frame, start, end)
        if frame.empty:
            return None
        # La serie ya es de retorno total; el cierre crudo no existe aquí.
        frame["adj_close"] = frame["close"]
        return frame[list(PRICE_COLUMNS)]


# ===========================================================================
# 11. AlpacaProvider — /v2/stocks/bars por lotes, raw + all
# ===========================================================================


class AlpacaProvider(PriceProviderBase):
    """Alpaca Market Data v2: barras diarias por lotes de símbolos.

    Endpoint (forma verificada contra la documentación pública, 2026-08;
    `docs/research/data_sources.md` §3.2)::

        GET https://data.alpaca.markets/v2/stocks/bars
            ?symbols=AAPL,MSFT&timeframe=1Day&start=&end=&adjustment=raw
            &feed=sip&limit=10000
        APCA-API-KEY-ID / APCA-API-SECRET-KEY

    Respuesta: ``{"bars": {"AAPL": [{"t": "2020-08-03T04:00:00Z", "o","h","l",
    "c","v","n","vw"}, ...]}, "next_page_token": ...}``. Los símbolos sin datos
    simplemente **no aparecen** en ``bars``. La paginación repite la petición
    con ``page_token``.

    Estrategia de ajuste: una pasada con ``adjustment=raw`` para el OHLCV
    como-se-negoció y, si ``adjusted=True``, una segunda con
    ``adjustment=all`` cuyo cierre se usa como `adj_close`. Alpaca no publica
    los eventos en este endpoint, así que `dividend`/`split_factor` quedan a
    NaN y el ancla de `adj_close` es la que aplique Alpaca (su historia
    completa), no el fin de la ventana: los *cocientes* —lo único con
    significado— son igual de válidos.

    Profundidad: el histórico SIP arranca hacia 2016 (§3.2); para 1996-2015
    esta fuente no sirve. `feed` es configurable («sip» exige plan de pago;
    con el gratuito, «iex»).
    """

    name = "alpaca"
    kinds: tuple[str, ...] = (PRICES_KIND,)
    BASE = "https://data.alpaca.markets/v2/stocks/bars"
    batch_size = 50

    def __init__(self, *, feed: str = "sip", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.feed = feed

    def _auth_headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self._credential("ALPACA_API_KEY_ID"),
            "APCA-API-SECRET-KEY": self._credential("ALPACA_API_SECRET_KEY"),
        }

    def _request_bars(
        self, symbols: Sequence[str], start: date, end: date, adjustment: str
    ) -> dict[str, list[dict[str, Any]]]:
        headers = self._auth_headers()
        out: dict[str, list[dict[str, Any]]] = {}
        for i in range(0, len(symbols), self.batch_size):
            chunk = symbols[i : i + self.batch_size]
            params: dict[str, Any] = {
                "symbols": ",".join(chunk),
                "timeframe": "1Day",
                "start": f"{start.isoformat()}T00:00:00Z",
                "end": f"{end.isoformat()}T23:59:59Z",
                "adjustment": adjustment,
                "feed": self.feed,
                "limit": 10000,
            }
            while True:
                payload = self.http.get_json(self.BASE, params=params, headers=headers)
                if not isinstance(payload, dict):
                    msg = "alpaca: respuesta no-objeto en /v2/stocks/bars"
                    raise DataQualityError(msg)
                for symbol, rows in (payload.get("bars") or {}).items():
                    out.setdefault(symbol, []).extend(rows or [])
                token = payload.get("next_page_token")
                if not token:
                    break
                params = {**params, "page_token": token}
        return out

    @staticmethod
    def _rows_to_frame(rows: list[dict[str, Any]], start: date, end: date) -> pd.DataFrame:
        dates = _iso_to_dates([r["t"] for r in rows], tz=_NY_TZ)
        frame = _bar_frame(
            dates,
            {
                "open": [r.get("o") for r in rows],
                "high": [r.get("h") for r in rows],
                "low": [r.get("l") for r in rows],
                "close": [r.get("c") for r in rows],
                "volume": [r.get("v") for r in rows],
            },
        )
        return _clip_window(frame, start, end)

    def _fetch_all(
        self, tickers: list[Ticker], start: date, end: date, adjusted: bool
    ) -> tuple[dict[Ticker, pd.DataFrame], list[Ticker]]:
        frames: dict[Ticker, pd.DataFrame] = {}
        remaining: list[Ticker] = []
        for ticker in tickers:
            if self.cache is None:
                remaining.append(ticker)
                continue
            hit = self.cache.get(
                PRICES_KIND, self.name, self._cache_params(ticker, start, end, adjusted)
            )
            if hit is None:
                remaining.append(ticker)
            else:
                frames[ticker] = hit

        if remaining:
            # Alpaca usa el punto en clases de acción ("BRK.B"): sin remapeo.
            raw = self._request_bars(remaining, start, end, "raw")
            adj = self._request_bars(remaining, start, end, "all") if adjusted else {}
            for ticker in remaining:
                rows = raw.get(ticker)
                if not rows:
                    continue
                frame = self._rows_to_frame(rows, start, end)
                if frame.empty:
                    continue
                if adjusted:
                    adj_rows = adj.get(ticker) or []
                    if adj_rows:
                        adj_frame = self._rows_to_frame(adj_rows, start, end)
                        frame["adj_close"] = adj_frame["close"].reindex(frame.index)
                frame = frame[list(PRICE_COLUMNS)]
                frames[ticker] = frame
                if self.cache is not None:
                    self.cache.write(
                        PRICES_KIND,
                        self.name,
                        self._cache_params(ticker, start, end, adjusted),
                        frame,
                    )
        missing = [t for t in tickers if t not in frames]
        return frames, missing

    def _fetch_bars(
        self, ticker: Ticker, start: date, end: date, adjusted: bool
    ) -> pd.DataFrame | None:  # pragma: no cover - _fetch_all lo sustituye
        frames, _ = self._fetch_all([ticker], start, end, adjusted)
        return frames.get(ticker)


# ===========================================================================
# 12. Registro y fachada
# ===========================================================================

DEFAULT_PRICE_PRIORITIES: dict[str, int] = {
    # docs/research/data_sources.md §15.1 (prices_daily). EODHD no está
    # implementado; Alpaca no figura en la lista diaria del informe y se coloca
    # entre EODHD (70) y Stooq (40) por dar cierre crudo pero poca profundidad.
    "tiingo": 90,
    "polygon": 80,
    "alpaca": 50,
    "stooq": 40,
    "yfinance": 20,
    "synthetic": 0,
}
"""Prioridades por defecto para `kind="prices"` (mayor = se intenta antes)."""


def register_price_providers(
    registry: ProviderRegistry | None = None,
    *,
    settings: Settings | None = None,
    cache: DiskCache | None = None,
    market: SyntheticMarket | None = None,
    include: Sequence[str] | None = None,
    priorities: Mapping[str, int] | None = None,
) -> ProviderRegistry:
    """Registra los proveedores de precios en un `ProviderRegistry`.

    Se registran **todos** (tengan o no credenciales): la disponibilidad se
    evalúa al resolver, y así el mensaje de `ProviderUnavailable` enumera qué
    variable de entorno falta para cada candidato. `synthetic` va con prioridad
    0 como red de seguridad sin red (contrato §0.5).

    Parámetros: `include` restringe el conjunto (nombres de
    `DEFAULT_PRICE_PRIORITIES`), `priorities` sobrescribe prioridades, `cache`
    y `market` se comparten entre los proveedores creados.
    """
    reg = registry or get_registry()
    cfg = settings or get_settings()
    chosen = dict(DEFAULT_PRICE_PRIORITIES)
    if include is not None:
        unknown = sorted(set(include) - set(chosen))
        if unknown:
            msg = f"proveedores de precios desconocidos en include: {unknown}"
            raise ConfigError(msg)
        chosen = {k: v for k, v in chosen.items() if k in set(include)}
    if priorities:
        chosen.update({k: int(v) for k, v in priorities.items() if k in chosen})

    factories: dict[str, Any] = {
        "tiingo": lambda: TiingoProvider(settings=cfg, cache=cache),
        "polygon": lambda: PolygonProvider(settings=cfg, cache=cache),
        "alpaca": lambda: AlpacaProvider(settings=cfg, cache=cache),
        "stooq": lambda: StooqProvider(settings=cfg, cache=cache),
        "yfinance": lambda: YFinanceProvider(settings=cfg, cache=cache),
        "synthetic": lambda: SyntheticProvider(market, settings=cfg),
    }
    for name, priority in chosen.items():
        reg.register(PRICES_KIND, factories[name](), priority, replace=True)
    return reg


def get_bars(
    tickers: Ticker | Sequence[Ticker],
    start: DateLike,
    end: DateLike,
    adjusted: bool = True,
    *,
    registry: ProviderRegistry | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Fachada: panel de precios del mejor proveedor disponible, con fallback.

    Resuelve `kind="prices"` contra el registro (el global si no se pasa otro)
    y delega en `ProviderRegistry.call`, que ante un fallo *en ejecución*
    (429 persistente, timeout, credencial caducada) pone al proveedor en
    cuarentena y prueba el siguiente. Un `DataQualityError` **no** provoca
    fallback: datos que llegan mal parseados son un bug reproducible, no una
    razón para cambiar de fuente en silencio (`data.base.FALLBACK_ERRORS`).
    """
    reg = registry or get_registry()
    s, e = _check_window(start, end)
    return reg.call(
        PRICES_KIND,
        lambda p: p.get_bars(tickers, s, e, adjusted, **kwargs),  # type: ignore[attr-defined]
        description=f"get_bars[{s.isoformat()}..{e.isoformat()}]",
    )
