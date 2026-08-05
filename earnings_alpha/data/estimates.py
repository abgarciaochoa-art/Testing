"""Calendario de resultados y consenso de analistas (`data.estimates`).

Este módulo es el adaptador del contrato §2 (`data.estimates`): sirve **dos**
tipos de dato distintos que la gente confunde y que aquí se separan de forma
explícita (`docs/research/data_sources.md` §6.1):

1. **Calendario de resultados** (`kind="earnings_calendar"`): fecha —y, cuando
   la fuente lo da, hora BMO/AMC— de cada anuncio, con el consenso *en el
   anuncio* (`eps_estimate`) y el resultado (`eps_actual`). Es lo que alimenta
   `pit.tradable_date`, el SUE de analistas y los estudios de evento.
2. **Consenso** (`kind="estimates"`): fotos `(ticker, period_end, as_of)` del
   consenso. **Toda fila lleva `as_of` y una bandera `is_point_in_time`**: solo
   es point-in-time la foto capturada en vivo (el `as_of` es el día real de la
   captura). Un consenso final adosado a un evento pasado **no** lo es, y los
   factores que necesiten vintages (momentum de revisiones,
   `analyst_revision_drift`) deben rechazarlo con `require_point_in_time`.

Proveedores implementados
-------------------------
- `FMPEstimatesProvider`      — FMP `/stable/earnings-calendar`, `/stable/earnings`,
  `/stable/analyst-estimates` (`docs/research/data_sources.md` §5.3, §6.2.e).
- `FinnhubEstimatesProvider`  — Finnhub `/calendar/earnings` (campo `hour` con
  BMO/AMC/DMH), `/stock/eps-estimate`, `/stock/revenue-estimate`.
- `EODHDEstimatesProvider`    — EODHD `/api/calendar/earnings`.
- `NasdaqEstimatesProvider`   — API web no oficial de Nasdaq (solo calendario,
  ventana reciente).
- `ExternalConsensusProvider` — sirve el dataset real del repo
  `data/external/consenso/consenso_master.parquet` (ver su docstring: es la
  fuente histórica offline, con advertencias PIT serias y explícitas).
- `SyntheticEstimatesProvider`— envuelve `data.synthetic.SyntheticMarket`
  (vintages con `as_of` genuino; red de seguridad sin red, contrato §0.5).

Convención de sesión y hora nominal
-----------------------------------
Muchas fuentes dan la **fecha** del anuncio y una etiqueta de sesión, pero no el
timestamp. Para poder construir `types.EarningsEvent.announced_at` sin inventar
precisión se usa una hora **nominal** de Nueva York por sesión (07:30 BMO,
12:30 DMH, 16:30 AMC/UNKNOWN) convertida a UTC, y se marca la columna
`announced_time_is_nominal=True`. La etiqueta `UNKNOWN` conserva su valor: es
`pit.tradable_date` quien aplica la política conservadora (UNKNOWN -> AMC ->
siguiente sesión), documentada en `types.Session.UNKNOWN`. Jamás se adivina BMO:
un BMO inventado es un día entero de look-ahead (DellaVigna y Pollet 2009,
*Journal of Finance*, y `docs/research/data_sources.md` §5.4-5.5).

Referencias
-----------
- DellaVigna, S., Pollet, J. (2009). "Investor Inattention and Friday Earnings
  Announcements". *Journal of Finance* 64(2): la corrección de fecha/hora que
  `tradable_date` implementa.
- Richardson, Teoh y Wysocki (2004): walk-down del consenso; por qué la sorpresa
  media es positiva y el consenso final no es un vintage.
- Diether, Malloy y Scherbina (2002): por qué los vintages mal fechados arruinan
  las medidas de dispersión y de sorpresa.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from earnings_alpha.config import Settings, get_settings
from earnings_alpha.data.base import (
    BaseProvider,
    DataKind,
    HttpClient,
    ProviderRegistry,
    RateLimitPolicy,
    get_registry,
)
from earnings_alpha.data.cache import DiskCache
from earnings_alpha.data.synthetic import SyntheticMarket
from earnings_alpha.errors import (
    ConfigError,
    DataQualityError,
    InsufficientHistory,
    LookAheadError,
    ProviderUnavailable,
)
from earnings_alpha.pit import eastern_to_utc, parse_session
from earnings_alpha.types import EarningsEvent, Session, Ticker, normalize_ticker

__all__ = [  # noqa: RUF022 - orden temático, no alfabético
    # esquema
    "CALENDAR_KIND",
    "ESTIMATES_KIND",
    "CALENDAR_COLUMNS",
    "CONSENSUS_COLUMNS",
    "NOMINAL_SESSION_TIMES_ET",
    "fiscal_quarter_label",
    "nominal_announced_at",
    "calendar_to_events",
    "events_to_frame",
    "require_point_in_time",
    # proveedores
    "EstimatesProviderBase",
    "FMPEstimatesProvider",
    "FinnhubEstimatesProvider",
    "EODHDEstimatesProvider",
    "NasdaqEstimatesProvider",
    "ExternalConsensusProvider",
    "SyntheticEstimatesProvider",
    # registro y fachadas
    "DEFAULT_ESTIMATES_PRIORITIES",
    "register_estimates_providers",
    "get_earnings_calendar",
    "get_consensus",
]

logger = logging.getLogger(__name__)

CALENDAR_KIND = DataKind.EARNINGS_CALENDAR.value
ESTIMATES_KIND = DataKind.ESTIMATES.value

CALENDAR_COLUMNS: tuple[str, ...] = (
    "ticker",
    "period_end",
    "announced_at",
    "session",
    "fiscal_quarter",
    "eps_actual",
    "eps_estimate",
    "revenue_actual",
    "revenue_estimate",
    "surprise",
    "surprise_pct",
    "is_estimated_date",
    "announced_time_is_nominal",
    "source",
)
"""Esquema canónico del calendario de resultados. Una fila por anuncio."""

CONSENSUS_COLUMNS: tuple[str, ...] = (
    "ticker",
    "period_end",
    "as_of",
    "available_at",
    "eps_mean",
    "eps_median",
    "eps_std",
    "eps_high",
    "eps_low",
    "n_analysts",
    "revenue_mean",
    "revenue_std",
    "is_point_in_time",
    "source",
)
"""Esquema canónico del consenso. Una fila por ``(ticker, period_end, as_of)``."""

NOMINAL_SESSION_TIMES_ET: dict[Session, dt.time] = {
    Session.BMO: dt.time(7, 30),
    Session.DMH: dt.time(12, 30),
    Session.AMC: dt.time(16, 30),
    Session.UNKNOWN: dt.time(16, 30),
}
"""Hora de pared de Nueva York asignada cuando la fuente solo da fecha+sesión.

UNKNOWN recibe la hora AMC porque la política del repo (`types.Session.UNKNOWN`,
`pit.tradable_date`) trata lo desconocido como AMC: ante la duda se pierde la
primera sesión, nunca se inventa una."""


DateLike = str | dt.date | dt.datetime | pd.Timestamp


# ===========================================================================
# 1. Utilidades de esquema
# ===========================================================================


def _as_date(value: DateLike, label: str = "fecha") -> dt.date:
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        msg = f"{label} no interpretable: {value!r}"
        raise ConfigError(msg) from exc
    if pd.isna(ts):
        msg = f"{label} no interpretable: {value!r}"
        raise ConfigError(msg)
    return ts.date()


def _check_window(start: DateLike, end: DateLike) -> tuple[dt.date, dt.date]:
    s, e = _as_date(start, "start"), _as_date(end, "end")
    if s > e:
        msg = f"ventana invertida: start={s} > end={e}"
        raise ConfigError(msg)
    return s, e


def _normalize_tickers(tickers: Ticker | Sequence[Ticker] | None) -> list[Ticker] | None:
    if tickers is None:
        return None
    raw = [tickers] if isinstance(tickers, str) else list(tickers)
    out: list[Ticker] = []
    for t in raw:
        norm = normalize_ticker(str(t))
        if norm and norm not in out:
            out.append(norm)
    if not out:
        msg = "la selección de símbolos está vacía"
        raise ConfigError(msg)
    return out


def fiscal_quarter_label(period_end: DateLike | None) -> str:
    """Etiqueta ``"2020Q2"`` del trimestre **natural** en que cae `period_end`.

    Es una convención de etiquetado, no una afirmación sobre el año fiscal del
    emisor: una empresa con cierre fiscal en julio tendrá etiquetas desplazadas
    respecto a su numeración interna. Para claves de join eso es irrelevante
    (la clave real es `period_end`); para mostrar, es honesto y determinista.
    Devuelve ``""`` si `period_end` es nulo.
    """
    if period_end is None:
        return ""
    ts = pd.Timestamp(period_end)
    if pd.isna(ts):
        return ""
    return f"{ts.year}Q{(ts.month - 1) // 3 + 1}"


def nominal_announced_at(report_date: DateLike, session: Session | str) -> dt.datetime:
    """Timestamp UTC nominal de un anuncio del que solo se conoce fecha y sesión.

    Usa `NOMINAL_SESSION_TIMES_ET` y la conversión DST-correcta de
    `pit.eastern_to_utc`. El resultado es tz-naive en UTC (convención de
    `types.EarningsEvent`). El consumidor sabe que la hora es nominal por la
    columna `announced_time_is_nominal` del calendario.
    """
    sess = session if isinstance(session, Session) else parse_session(session)
    day = _as_date(report_date, "report_date")
    local = dt.datetime.combine(day, NOMINAL_SESSION_TIMES_ET[sess])
    return eastern_to_utc(local)


def _empty_calendar() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=object) for c in CALENDAR_COLUMNS})


def _finalize_calendar(frame: pd.DataFrame, provider: str) -> pd.DataFrame:
    """Normaliza tipos y orden del calendario; vacío -> `InsufficientHistory`."""
    if frame is None or len(frame) == 0:
        msg = f"{provider}: el calendario de resultados está vacío para la ventana pedida"
        raise InsufficientHistory(msg)
    out = frame.copy()
    missing = [c for c in CALENDAR_COLUMNS if c not in out.columns]
    if missing:
        msg = f"{provider}: faltan columnas del calendario canónico: {missing}"
        raise DataQualityError(msg)
    out["ticker"] = [normalize_ticker(str(t)) for t in out["ticker"]]
    out["period_end"] = pd.to_datetime(out["period_end"], errors="coerce")
    out["announced_at"] = pd.to_datetime(out["announced_at"])
    if getattr(out["announced_at"].dt, "tz", None) is not None:
        out["announced_at"] = out["announced_at"].dt.tz_convert("UTC").dt.tz_localize(None)
    out["session"] = [
        s.value if isinstance(s, Session) else parse_session(s).value for s in out["session"]
    ]
    for col in ("eps_actual", "eps_estimate", "revenue_actual", "revenue_estimate",
                "surprise", "surprise_pct"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["is_estimated_date"] = out["is_estimated_date"].astype(bool)
    out["announced_time_is_nominal"] = out["announced_time_is_nominal"].astype(bool)
    out = out[list(CALENDAR_COLUMNS)]
    return out.sort_values(["announced_at", "ticker"]).reset_index(drop=True)


def _finalize_consensus(frame: pd.DataFrame, provider: str) -> pd.DataFrame:
    """Normaliza el consenso; vacío -> `InsufficientHistory`."""
    if frame is None or len(frame) == 0:
        msg = f"{provider}: no hay filas de consenso para la selección pedida"
        raise InsufficientHistory(msg)
    out = frame.copy()
    missing = [c for c in CONSENSUS_COLUMNS if c not in out.columns]
    if missing:
        msg = f"{provider}: faltan columnas del consenso canónico: {missing}"
        raise DataQualityError(msg)
    out["ticker"] = [normalize_ticker(str(t)) for t in out["ticker"]]
    out["period_end"] = pd.to_datetime(out["period_end"], errors="coerce")
    out["as_of"] = pd.to_datetime(out["as_of"]).dt.normalize()
    out["available_at"] = pd.to_datetime(out["available_at"])
    for col in ("eps_mean", "eps_median", "eps_std", "eps_high", "eps_low",
                "revenue_mean", "revenue_std"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["n_analysts"] = pd.to_numeric(out["n_analysts"], errors="coerce")
    out["is_point_in_time"] = out["is_point_in_time"].astype(bool)
    if (out["available_at"] < out["as_of"]).any():
        msg = f"{provider}: hay filas con available_at anterior a as_of; el vintage está corrupto"
        raise DataQualityError(msg)
    out = out[list(CONSENSUS_COLUMNS)]
    return out.sort_values(["ticker", "period_end", "as_of"]).reset_index(drop=True)


def require_point_in_time(consensus: pd.DataFrame) -> pd.DataFrame:
    """Rechaza consenso no point-in-time. Para factores de **revisiones**.

    Un factor de momentum de revisiones (`analyst_revision_drift`) calculado con
    consenso final es look-ahead con apariencia de alfa: el "drift" medido
    incluiría revisiones posteriores a la fecha de la señal. Este guardián
    convierte ese error silencioso en un `LookAheadError` ruidoso, que es la
    política del repo (contrato §0.1).
    """
    if "is_point_in_time" not in consensus.columns:
        msg = "el consenso no declara `is_point_in_time`; no se puede verificar el vintage"
        raise DataQualityError(msg)
    bad = consensus[~consensus["is_point_in_time"].astype(bool)]
    if len(bad):
        sources = sorted(set(bad.get("source", pd.Series(dtype=object)).astype(str)))
        msg = (
            f"{len(bad)} de {len(consensus)} filas de consenso no son point-in-time "
            f"(fuentes: {sources}). Estas filas sirven para SUE y estudios de evento, "
            "no para momentum de revisiones. Filtra con `is_point_in_time` o usa un "
            "proveedor con vintages reales."
        )
        raise LookAheadError(msg)
    return consensus


def calendar_to_events(frame: pd.DataFrame) -> list[EarningsEvent]:
    """Convierte el calendario canónico en `types.EarningsEvent`.

    Las filas sin `period_end` (algunas fuentes no lo dan) **no** se pueden
    convertir —`EarningsEvent.period_end` es obligatorio y `event_id` depende de
    él— y se omiten dejando constancia en el log; siguen disponibles en el
    DataFrame para quien pueda reconstruir el trimestre por otra vía.
    """
    dropped = int(frame["period_end"].isna().sum())
    if dropped:
        logger.warning(
            "calendar_to_events: %d filas sin period_end omitidas (siguen en el DataFrame)",
            dropped,
        )
    events: list[EarningsEvent] = []
    for row in frame.dropna(subset=["period_end"]).itertuples(index=False):
        events.append(
            EarningsEvent(
                ticker=row.ticker,
                period_end=pd.Timestamp(row.period_end).date(),
                announced_at=pd.Timestamp(row.announced_at).to_pydatetime(),
                session=Session(row.session),
                fiscal_quarter=row.fiscal_quarter or fiscal_quarter_label(row.period_end),
                eps_actual=None if pd.isna(row.eps_actual) else float(row.eps_actual),
                eps_estimate=None if pd.isna(row.eps_estimate) else float(row.eps_estimate),
                revenue_actual=None if pd.isna(row.revenue_actual) else float(row.revenue_actual),
                revenue_estimate=(
                    None if pd.isna(row.revenue_estimate) else float(row.revenue_estimate)
                ),
                source=str(row.source),
                is_estimated_date=bool(row.is_estimated_date),
            )
        )
    return events


def events_to_frame(events: Sequence[EarningsEvent]) -> pd.DataFrame:
    """Convierte `EarningsEvent` al calendario canónico (inversa de arriba)."""
    rows = [
        {
            "ticker": ev.ticker,
            "period_end": pd.Timestamp(ev.period_end),
            "announced_at": pd.Timestamp(ev.announced_at),
            "session": ev.session.value,
            "fiscal_quarter": ev.fiscal_quarter,
            "eps_actual": ev.eps_actual,
            "eps_estimate": ev.eps_estimate,
            "revenue_actual": ev.revenue_actual,
            "revenue_estimate": ev.revenue_estimate,
            "surprise": ev.eps_surprise,
            "surprise_pct": (
                100.0 * ev.eps_surprise / abs(ev.eps_estimate)
                if ev.eps_surprise is not None and ev.eps_estimate not in (None, 0)
                else np.nan
            ),
            "is_estimated_date": ev.is_estimated_date,
            "announced_time_is_nominal": False,
            "source": ev.source,
        }
        for ev in events
    ]
    if not rows:
        return _empty_calendar()
    return _finalize_calendar(pd.DataFrame(rows), "events_to_frame")


# ===========================================================================
# 2. Base común de proveedores
# ===========================================================================


class EstimatesProviderBase(BaseProvider):
    """Base de los adaptadores de calendario/consenso.

    API pública:

    - `earnings_calendar(start, end, tickers=None)` -> calendario canónico.
    - `consensus(tickers, ...)` -> consenso canónico con `as_of` y bandera
      `is_point_in_time`.

    Política de fallo (contrato §0.3): sin credenciales ->
    `ProviderUnavailable` antes de tocar la red; ventana sin eventos ->
    `InsufficientHistory`; payload malformado -> `DataQualityError`. Nunca un
    DataFrame vacío silencioso.
    """

    name = "estimates_base"
    kinds: tuple[str, ...] = (CALENDAR_KIND, ESTIMATES_KIND)

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        http: HttpClient | None = None,
        cache: DiskCache | None = None,
    ) -> None:
        super().__init__(settings=settings, http=http)
        self.cache = cache

    # -- credenciales ---------------------------------------------------------

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
                "faltan credenciales para pedir calendario/consenso",
                missing_env=missing,
            )

    def _today(self) -> dt.date:
        """Fecha de captura para los `as_of` de consenso en vivo (UTC)."""
        return dt.datetime.now(dt.UTC).date()

    # -- API pública ----------------------------------------------------------

    def earnings_calendar(
        self,
        start: DateLike,
        end: DateLike,
        tickers: Ticker | Sequence[Ticker] | None = None,
    ) -> pd.DataFrame:
        """Calendario canónico de anuncios en ``[start, end]`` (inclusive)."""
        s, e = _check_window(start, end)
        wanted = _normalize_tickers(tickers)
        self._require_credentials()
        out = _finalize_calendar(self._fetch_calendar(s, e, wanted), self.name)
        if wanted is not None:
            # Tras normalizar: el "BRK-B" de un proveedor debe casar con "BRK.B".
            out = out[out["ticker"].isin(wanted)].reset_index(drop=True)
            if not len(out):
                msg = (
                    f"{self.name}: ninguno de los símbolos pedidos ({wanted}) tiene "
                    f"anuncios en [{s}, {e}]"
                )
                raise InsufficientHistory(msg)
        return out

    def consensus(
        self,
        tickers: Ticker | Sequence[Ticker],
        *,
        start: DateLike | None = None,
        end: DateLike | None = None,
    ) -> pd.DataFrame:
        """Consenso canónico. `start`/`end` acotan `period_end` si se dan."""
        wanted = _normalize_tickers(tickers)
        assert wanted is not None  # tickers es obligatorio aquí
        self._require_credentials()
        out = _finalize_consensus(self._fetch_consensus(wanted), self.name)
        if start is not None:
            out = out[out["period_end"] >= pd.Timestamp(_as_date(start))]
        if end is not None:
            out = out[out["period_end"] <= pd.Timestamp(_as_date(end))]
        out = out.reset_index(drop=True)
        if not len(out):
            msg = f"{self.name}: el filtro de periodos deja el consenso vacío"
            raise InsufficientHistory(msg)
        return out

    # -- ganchos --------------------------------------------------------------

    def _fetch_calendar(
        self, start: dt.date, end: dt.date, tickers: list[Ticker] | None
    ) -> pd.DataFrame:
        raise NotImplementedError

    def _fetch_consensus(self, tickers: list[Ticker]) -> pd.DataFrame:
        raise NotImplementedError


# ===========================================================================
# 3. FMP
# ===========================================================================


class FMPEstimatesProvider(EstimatesProviderBase):
    """Financial Modeling Prep: calendario amplio + consenso actual.

    Endpoints (`docs/research/data_sources.md` §5.3 y §6.2.e)::

        GET /stable/earnings-calendar?from=&to=&apikey=
            -> [{"symbol", "date", "epsActual", "epsEstimated",
                 "revenueActual", "revenueEstimated", "time"?, "lastUpdated"?}]
        GET /stable/earnings?symbol=&apikey=
            -> histórico por símbolo con el consenso EN el anuncio
        GET /stable/analyst-estimates?symbol=&period=quarter&apikey=
            -> [{"date" (period_end), "epsAvg", "epsHigh", "epsLow",
                 "numAnalystsEps"?, "revenueAvg", "revenueHigh", "revenueLow"}]

    Advertencias PIT, ambas del informe §6.2.e:

    - El campo `time` (bmo/amc) no siempre viene; sin él la sesión queda
      `UNKNOWN` y `tradable_date` la manda a la sesión siguiente (conservador).
    - `analyst-estimates` es el consenso **actual**: solo las filas de periodos
      futuros son un vintage legítimo (`as_of` = día de la captura). Las filas
      de periodos pasados se marcan `is_point_in_time=False`.
    - Los eventos con fecha futura son estimaciones del proveedor:
      `is_estimated_date=True`.
    """

    name = "fmp"
    kinds: tuple[str, ...] = (CALENDAR_KIND, ESTIMATES_KIND)
    BASE = "https://financialmodelingprep.com"

    def _get(self, path: str, params: dict[str, Any]) -> Any:
        params = dict(params)
        params["apikey"] = self._credential("FMP_API_KEY")
        return self.http.get_json(f"{self.BASE}{path}", params=params)

    @staticmethod
    def _rows(payload: Any, context: str) -> list[dict[str, Any]]:
        if isinstance(payload, dict) and "Error Message" in payload:
            msg = f"fmp: {payload['Error Message']}"
            raise DataQualityError(msg)
        if not isinstance(payload, list):
            msg = f"fmp: se esperaba una lista JSON en {context}, llegó {type(payload).__name__}"
            raise DataQualityError(msg)
        return payload

    def _calendar_rows_to_frame(self, rows: list[dict[str, Any]]) -> pd.DataFrame:
        today = self._today()
        out: list[dict[str, Any]] = []
        for row in rows:
            symbol, when = row.get("symbol"), row.get("date")
            if not symbol or not when:
                continue
            day = _as_date(when, "fmp.date")
            session = parse_session(row.get("time"))
            period = row.get("fiscalDateEnding") or row.get("period") or None
            eps_est = row.get("epsEstimated")
            eps_act = row.get("epsActual", row.get("eps"))
            surprise = (
                float(eps_act) - float(eps_est)
                if eps_act is not None and eps_est is not None
                else np.nan
            )
            out.append(
                {
                    "ticker": symbol,
                    "period_end": pd.Timestamp(period) if period else pd.NaT,
                    "announced_at": nominal_announced_at(day, session),
                    "session": session.value,
                    "fiscal_quarter": fiscal_quarter_label(period) if period else "",
                    "eps_actual": eps_act,
                    "eps_estimate": eps_est,
                    "revenue_actual": row.get("revenueActual", row.get("revenue")),
                    "revenue_estimate": row.get("revenueEstimated"),
                    "surprise": surprise,
                    "surprise_pct": (
                        100.0 * surprise / abs(float(eps_est))
                        if not pd.isna(surprise) and eps_est not in (None, 0)
                        else np.nan
                    ),
                    "is_estimated_date": day > today,
                    "announced_time_is_nominal": True,
                    "source": "fmp",
                }
            )
        return pd.DataFrame(out) if out else _empty_calendar()

    def _fetch_calendar(
        self, start: dt.date, end: dt.date, tickers: list[Ticker] | None
    ) -> pd.DataFrame:
        rows = self._rows(
            self._get(
                "/stable/earnings-calendar",
                {"from": start.isoformat(), "to": end.isoformat()},
            ),
            "earnings-calendar",
        )
        return self._calendar_rows_to_frame(rows)

    def historical_earnings(self, ticker: Ticker) -> pd.DataFrame:
        """Histórico de un símbolo con el consenso EN el anuncio (para SUE).

        Es el endpoint que el informe §6.2.e identifica como "esto sí es el
        consenso en el anuncio". No trae hora: sesión `UNKNOWN`, hora nominal.
        """
        self._require_credentials()
        symbol = normalize_ticker(ticker)
        rows = self._rows(self._get("/stable/earnings", {"symbol": symbol}), "earnings")
        frame = self._calendar_rows_to_frame(rows)
        if len(frame):
            frame["ticker"] = symbol
        return _finalize_calendar(frame, self.name)

    def _fetch_consensus(self, tickers: list[Ticker]) -> pd.DataFrame:
        today = pd.Timestamp(self._today())
        frames: list[pd.DataFrame] = []
        for symbol in tickers:
            rows = self._rows(
                self._get(
                    "/stable/analyst-estimates",
                    {"symbol": symbol, "period": "quarter", "page": 0, "limit": 100},
                ),
                "analyst-estimates",
            )
            for row in rows:
                period = row.get("date")
                if not period:
                    continue
                period_ts = pd.Timestamp(_as_date(period, "fmp.period"))
                frames.append(
                    pd.DataFrame(
                        [
                            {
                                "ticker": symbol,
                                "period_end": period_ts,
                                "as_of": today,
                                "available_at": today,
                                "eps_mean": row.get("epsAvg"),
                                "eps_median": np.nan,
                                "eps_std": np.nan,
                                "eps_high": row.get("epsHigh"),
                                "eps_low": row.get("epsLow"),
                                "n_analysts": row.get(
                                    "numAnalystsEps", row.get("numberAnalystsEstimatedEps")
                                ),
                                "revenue_mean": row.get("revenueAvg"),
                                "revenue_std": np.nan,
                                # Solo la foto de un periodo aún no cerrado es un
                                # vintage legítimo con as_of = hoy (informe §6.3,
                                # salida 2). Un periodo pasado es consenso final.
                                "is_point_in_time": bool(period_ts >= today),
                                "source": "fmp",
                            }
                        ]
                    )
                )
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ===========================================================================
# 4. Finnhub
# ===========================================================================


class FinnhubEstimatesProvider(EstimatesProviderBase):
    """Finnhub: la fuente gratuita con etiqueta BMO/AMC explícita.

    Endpoints (`docs/research/data_sources.md` §5.3)::

        GET /api/v1/calendar/earnings?from=&to=[&symbol=]
            -> {"earningsCalendar": [{"date", "epsActual", "epsEstimate",
                "hour" in {"bmo","amc","dmh",""}, "quarter", "year",
                "revenueActual", "revenueEstimate", "symbol"}]}
        GET /api/v1/stock/eps-estimate?symbol=&freq=quarterly
            -> {"data": [{"epsAvg","epsHigh","epsLow","numberAnalysts","period"}]}
        GET /api/v1/stock/revenue-estimate?symbol=&freq=quarterly

    Autenticación por cabecera ``X-Finnhub-Token`` (no por query param: así la
    clave no acaba en URLs de logs ni en claves de caché).

    `period_end` se aproxima como el fin del trimestre **natural**
    ``(year, quarter)`` que reporta Finnhub; para emisores con año fiscal
    desplazado es una aproximación y se documenta como tal — la clave PIT
    (fecha y sesión del anuncio) no se ve afectada.
    """

    name = "finnhub"
    kinds: tuple[str, ...] = (CALENDAR_KIND, ESTIMATES_KIND)
    BASE = "https://finnhub.io/api/v1"

    def _get(self, path: str, params: dict[str, Any]) -> Any:
        headers = {"X-Finnhub-Token": self._credential("FINNHUB_API_KEY")}
        return self.http.get_json(f"{self.BASE}{path}", params=params, headers=headers)

    @staticmethod
    def _quarter_end(year: Any, quarter: Any) -> pd.Timestamp:
        try:
            y, q = int(year), int(quarter)
        except (TypeError, ValueError):
            return pd.NaT
        if not 1 <= q <= 4:
            return pd.NaT
        month = 3 * q
        return pd.Timestamp(year=y, month=month, day=1) + pd.offsets.MonthEnd(0)

    def _fetch_calendar(
        self, start: dt.date, end: dt.date, tickers: list[Ticker] | None
    ) -> pd.DataFrame:
        params: dict[str, Any] = {"from": start.isoformat(), "to": end.isoformat()}
        if tickers is not None and len(tickers) == 1:
            params["symbol"] = tickers[0]
        payload = self._get("/calendar/earnings", params)
        if not isinstance(payload, dict) or "earningsCalendar" not in payload:
            msg = "finnhub: la respuesta no contiene `earningsCalendar`"
            raise DataQualityError(msg)
        today = self._today()
        out: list[dict[str, Any]] = []
        for row in payload["earningsCalendar"] or []:
            symbol, when = row.get("symbol"), row.get("date")
            if not symbol or not when:
                continue
            day = _as_date(when, "finnhub.date")
            session = parse_session(row.get("hour"))
            period = self._quarter_end(row.get("year"), row.get("quarter"))
            eps_act, eps_est = row.get("epsActual"), row.get("epsEstimate")
            surprise = (
                float(eps_act) - float(eps_est)
                if eps_act is not None and eps_est is not None
                else np.nan
            )
            out.append(
                {
                    "ticker": symbol,
                    "period_end": period,
                    "announced_at": nominal_announced_at(day, session),
                    "session": session.value,
                    "fiscal_quarter": fiscal_quarter_label(period) if not pd.isna(period) else "",
                    "eps_actual": eps_act,
                    "eps_estimate": eps_est,
                    "revenue_actual": row.get("revenueActual"),
                    "revenue_estimate": row.get("revenueEstimate"),
                    "surprise": surprise,
                    "surprise_pct": (
                        100.0 * surprise / abs(float(eps_est))
                        if not pd.isna(surprise) and eps_est not in (None, 0)
                        else np.nan
                    ),
                    "is_estimated_date": day > today,
                    "announced_time_is_nominal": True,
                    "source": "finnhub",
                }
            )
        return pd.DataFrame(out) if out else _empty_calendar()

    def _fetch_consensus(self, tickers: list[Ticker]) -> pd.DataFrame:
        today = pd.Timestamp(self._today())
        rows: list[dict[str, Any]] = []
        for symbol in tickers:
            eps = self._get("/stock/eps-estimate", {"symbol": symbol, "freq": "quarterly"})
            rev = self._get("/stock/revenue-estimate", {"symbol": symbol, "freq": "quarterly"})
            rev_by_period = {
                r.get("period"): r for r in (rev.get("data") or []) if isinstance(r, dict)
            } if isinstance(rev, dict) else {}
            for row in (eps.get("data") or []) if isinstance(eps, dict) else []:
                period = row.get("period")
                if not period:
                    continue
                period_ts = pd.Timestamp(_as_date(period, "finnhub.period"))
                rev_row = rev_by_period.get(period, {})
                rows.append(
                    {
                        "ticker": symbol,
                        "period_end": period_ts,
                        "as_of": today,
                        "available_at": today,
                        "eps_mean": row.get("epsAvg"),
                        "eps_median": np.nan,
                        "eps_std": np.nan,
                        "eps_high": row.get("epsHigh"),
                        "eps_low": row.get("epsLow"),
                        "n_analysts": row.get("numberAnalysts"),
                        "revenue_mean": rev_row.get("revenueAvg"),
                        "revenue_std": np.nan,
                        "is_point_in_time": bool(period_ts >= today),
                        "source": "finnhub",
                    }
                )
        return pd.DataFrame(rows) if rows else pd.DataFrame()


# ===========================================================================
# 5. EODHD
# ===========================================================================


class EODHDEstimatesProvider(EstimatesProviderBase):
    """EODHD: calendario con `before_after_market` y estimación por evento.

    Endpoint (`docs/research/data_sources.md` §5.3)::

        GET https://eodhd.com/api/calendar/earnings?from=&to=&fmt=json
            [&symbols=AAPL.US]&api_token=
            -> {"earnings": [{"code": "AAPL.US", "report_date", "date"
                (period_end), "before_after_market" in
                {"BeforeMarket","AfterMarket",null}, "actual", "estimate",
                "difference", "percent"}]}

    EODHD **no** publica vintages de consenso: `consensus()` lanza
    `ProviderUnavailable` con ese motivo para que el registro pase al siguiente
    candidato en vez de servir un consenso inventado.
    """

    name = "eodhd"
    kinds: tuple[str, ...] = (CALENDAR_KIND, ESTIMATES_KIND)
    BASE = "https://eodhd.com/api"

    def _fetch_calendar(
        self, start: dt.date, end: dt.date, tickers: list[Ticker] | None
    ) -> pd.DataFrame:
        params: dict[str, Any] = {
            "from": start.isoformat(),
            "to": end.isoformat(),
            "fmt": "json",
            "api_token": self._credential("EODHD_API_KEY"),
        }
        if tickers is not None:
            params["symbols"] = ",".join(f"{t}.US" for t in tickers)
        payload = self.http.get_json(f"{self.BASE}/calendar/earnings", params=params)
        if not isinstance(payload, dict) or "earnings" not in payload:
            msg = "eodhd: la respuesta no contiene `earnings`"
            raise DataQualityError(msg)
        today = self._today()
        out: list[dict[str, Any]] = []
        for row in payload["earnings"] or []:
            code, when = row.get("code"), row.get("report_date")
            if not code or not when:
                continue
            symbol = str(code).split(".")[0]
            day = _as_date(when, "eodhd.report_date")
            session = parse_session(row.get("before_after_market"))
            period = row.get("date") or None
            eps_act, eps_est = row.get("actual"), row.get("estimate")
            out.append(
                {
                    "ticker": symbol,
                    "period_end": pd.Timestamp(period) if period else pd.NaT,
                    "announced_at": nominal_announced_at(day, session),
                    "session": session.value,
                    "fiscal_quarter": fiscal_quarter_label(period) if period else "",
                    "eps_actual": eps_act,
                    "eps_estimate": eps_est,
                    "revenue_actual": np.nan,
                    "revenue_estimate": np.nan,
                    "surprise": row.get("difference"),
                    "surprise_pct": row.get("percent"),
                    "is_estimated_date": day > today,
                    "announced_time_is_nominal": True,
                    "source": "eodhd",
                }
            )
        return pd.DataFrame(out) if out else _empty_calendar()

    def _fetch_consensus(self, tickers: list[Ticker]) -> pd.DataFrame:
        raise ProviderUnavailable(
            self.name,
            "EODHD no expone vintages de consenso; usa el calendario para el "
            "consenso en el anuncio o un proveedor con snapshots",
        )


# ===========================================================================
# 6. Nasdaq (API web, no oficial)
# ===========================================================================


class NasdaqEstimatesProvider(EstimatesProviderBase):
    """Calendario de la web de Nasdaq. Gratuito, sin clave, **no oficial**.

    Endpoint (`docs/research/data_sources.md` §5.3)::

        GET https://api.nasdaq.com/api/calendar/earnings?date=YYYY-MM-DD
            -> {"data": {"rows": [{"symbol", "time" in {"time-pre-market",
                "time-after-hours", "time-not-supplied"}, "epsForecast":
                "$2.07"/"($0.15)", "noOfEsts", "fiscalQuarterEnding":
                "Jun/2020", "eps"?, "surprise"?}]},
                "status": {"rCode": 200}}

    Particularidades: una petición **por día natural** (la ventana se limita a
    `MAX_WINDOW_DAYS` para no bombardear un endpoint no documentado), cifras
    monetarias con `$`, comas y negativos entre paréntesis, y cabeceras de
    navegador (el endpoint rechaza clientes sin `User-Agent` "real"). Solo
    cubre una ventana reciente: es la fuente de *triangulación* de sesión, no
    la base histórica. `consensus()` -> `ProviderUnavailable`.
    """

    name = "nasdaq"
    kinds: tuple[str, ...] = (CALENDAR_KIND, ESTIMATES_KIND)
    BASE = "https://api.nasdaq.com/api/calendar/earnings"
    MAX_WINDOW_DAYS = 45
    RATE = RateLimitPolicy(1.0, burst=2, note="endpoint web no documentado; cortesía")
    HEADERS = {
        "Accept": "application/json, text/plain, */*",
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0 Safari/537.36"
        ),
    }

    @property
    def http(self) -> HttpClient:
        if self._http is None:
            self._http = HttpClient(self.name, rate=self.RATE, settings=self.settings)
        return self._http

    @staticmethod
    def _money(raw: Any) -> float:
        """``"$1,234.56"`` -> 1234.56; ``"($0.15)"`` -> -0.15; vacío -> NaN."""
        if raw is None:
            return float("nan")
        text = str(raw).strip()
        if not text or text.lower() in {"n/a", "--", "-"}:
            return float("nan")
        negative = text.startswith("(") and text.endswith(")")
        text = text.strip("()").replace("$", "").replace(",", "").strip()
        try:
            value = float(text)
        except ValueError:
            return float("nan")
        return -value if negative else value

    @staticmethod
    def _fiscal_period(raw: Any) -> pd.Timestamp:
        """``"Jun/2020"`` -> último día de junio de 2020."""
        if not raw:
            return pd.NaT
        try:
            return pd.Timestamp(dt.datetime.strptime(str(raw), "%b/%Y")) + pd.offsets.MonthEnd(0)
        except ValueError:
            return pd.NaT

    _SESSION_BY_TIME = {
        "time-pre-market": Session.BMO,
        "time-after-hours": Session.AMC,
        "time-not-supplied": Session.UNKNOWN,
    }

    def _fetch_calendar(
        self, start: dt.date, end: dt.date, tickers: list[Ticker] | None
    ) -> pd.DataFrame:
        n_days = (end - start).days + 1
        if n_days > self.MAX_WINDOW_DAYS:
            msg = (
                f"nasdaq: la ventana pedida ({n_days} días) supera MAX_WINDOW_DAYS="
                f"{self.MAX_WINDOW_DAYS}; este endpoint exige una petición por día y "
                "no está pensado para backfill histórico"
            )
            raise ConfigError(msg)
        today = self._today()
        out: list[dict[str, Any]] = []
        for offset in range(n_days):
            day = start + dt.timedelta(days=offset)
            payload = self.http.get_json(
                self.BASE, params={"date": day.isoformat()}, headers=self.HEADERS
            )
            if not isinstance(payload, dict) or "data" not in payload:
                msg = f"nasdaq: respuesta sin `data` para {day}"
                raise DataQualityError(msg)
            data = payload.get("data") or {}
            for row in (data.get("rows") or []):
                symbol = row.get("symbol")
                if not symbol:
                    continue
                session = self._SESSION_BY_TIME.get(
                    str(row.get("time", "")).strip(), Session.UNKNOWN
                )
                period = self._fiscal_period(row.get("fiscalQuarterEnding"))
                eps_est = self._money(row.get("epsForecast"))
                eps_act = self._money(row.get("eps"))
                surprise = (
                    eps_act - eps_est
                    if not (np.isnan(eps_act) or np.isnan(eps_est))
                    else np.nan
                )
                out.append(
                    {
                        "ticker": symbol,
                        "period_end": period,
                        "announced_at": nominal_announced_at(day, session),
                        "session": session.value,
                        "fiscal_quarter": (
                            fiscal_quarter_label(period) if not pd.isna(period) else ""
                        ),
                        "eps_actual": eps_act,
                        "eps_estimate": eps_est,
                        "revenue_actual": np.nan,
                        "revenue_estimate": np.nan,
                        "surprise": surprise,
                        "surprise_pct": (
                            100.0 * surprise / abs(eps_est)
                            if not pd.isna(surprise) and eps_est not in (0.0,)
                            else np.nan
                        ),
                        "is_estimated_date": day > today,
                        "announced_time_is_nominal": True,
                        "source": "nasdaq",
                    }
                )
        return pd.DataFrame(out) if out else _empty_calendar()

    def _fetch_consensus(self, tickers: list[Ticker]) -> pd.DataFrame:
        raise ProviderUnavailable(
            self.name,
            "la API web de Nasdaq solo publica el calendario; no hay consenso con as_of",
        )


# ===========================================================================
# 7. ExternalConsensusProvider — el dataset real del repo
# ===========================================================================


class ExternalConsensusProvider(EstimatesProviderBase):
    """Sirve `data/external/consenso/consenso_master.parquet` como fuente real.

    Es el histórico de calendario + consenso-en-el-anuncio del repo: 56.971
    filas, ~500 tickers, 1995-2026, con `ticker, fiscal_period_end,
    report_date, report_time, eps_reported, eps_estimated, surprise,
    surprise_pct, source, snapshot_date`.

    **ADVERTENCIAS PIT — leer antes de usar, y son vinculantes:**

    1. **Es el consenso FINAL previo al anuncio, sin historial de revisiones
       intra-trimestre.** No existe la senda `(as_of, eps_mean)` día a día:
       solo la última foto. Por eso *todo* el consenso de este proveedor sale
       con ``is_point_in_time=False`` y ``as_of = report_date`` (el único día
       en que esa cifra fue con certeza el consenso vigente). Sirve para SUE
       de analistas y estudios de evento; **NO sirve para momentum de
       revisiones** ni para `analyst_revision_drift` — `require_point_in_time`
       lo rechaza, y debe rechazarlo.
    2. **Supervivencia parcial.** La lista de tickers procede de snapshots
       tomados en 2022-2026 (columna `snapshot_date`): las empresas que
       salieron del índice o desaparecieron antes de esos snapshots están
       infrarrepresentadas. Cualquier corte transversal histórico construido
       SOLO con este fichero hereda sesgo de supervivencia; el universo PIT
       debe venir siempre de `universe.SP500Universe`, y este dataset solo
       aporta los atributos del evento.
    3. **Hora nominal.** `report_time` ∈ {pre-market, post-market, intraday}
       existe solo en ~18 % de las filas; el resto queda `UNKNOWN` y
       `tradable_date` lo trata como AMC (conservador). La hora dentro de la
       sesión es nominal (`announced_time_is_nominal=True`).

    Deduplicación: hay hasta 4 filas por ``(ticker, fiscal_period_end)``
    procedentes de snapshots distintos. Se conserva una sola: primero la que
    trae `report_time` (más información de sesión), y a igualdad, la de
    `snapshot_date` más reciente. Las filas sin `fiscal_period_end` (~9 %,
    fuentes yfinance) se conservan con `period_end=NaT` salvo que dupliquen un
    ``(ticker, report_date)`` ya cubierto.
    """

    name = "external_consensus"
    kinds: tuple[str, ...] = (CALENDAR_KIND, ESTIMATES_KIND)

    def __init__(
        self,
        path: Path | str | None = None,
        *,
        settings: Settings | None = None,
    ) -> None:
        super().__init__(settings=settings)
        self.path = Path(
            path
            if path is not None
            else self.settings.data_dir / "external" / "consenso" / "consenso_master.parquet"
        )
        self._table: pd.DataFrame | None = None

    def available(self) -> bool:
        return self.path.is_file()

    def _require_file(self) -> None:
        if not self.path.is_file():
            raise ProviderUnavailable(
                self.name,
                f"no existe el fichero {self.path}; el dataset externo de consenso "
                "no está en el repo",
            )

    _REQUIRED = (
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
    )

    @property
    def table(self) -> pd.DataFrame:
        """Tabla deduplicada y normalizada (se carga y depura una sola vez)."""
        if self._table is None:
            self._require_file()
            raw = pd.read_parquet(self.path)
            missing = [c for c in self._REQUIRED if c not in raw.columns]
            if missing:
                msg = f"external_consensus: faltan columnas en el parquet: {missing}"
                raise DataQualityError(msg)
            frame = raw.copy()
            frame["ticker"] = [normalize_ticker(str(t)) for t in frame["ticker"]]
            frame["report_date"] = pd.to_datetime(frame["report_date"], errors="raise")
            frame["fiscal_period_end"] = pd.to_datetime(
                frame["fiscal_period_end"], errors="coerce"
            )
            frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"], errors="coerce")
            # -- deduplicación (ver docstring) -------------------------------
            frame["_has_time"] = frame["report_time"].notna().astype(int)
            with_period = frame[frame["fiscal_period_end"].notna()]
            with_period = (
                with_period.sort_values(
                    ["ticker", "fiscal_period_end", "_has_time", "snapshot_date"]
                )
                .groupby(["ticker", "fiscal_period_end"], as_index=False, sort=False)
                .tail(1)
            )
            covered = set(
                zip(
                    with_period["ticker"],
                    with_period["report_date"],
                    strict=True,
                )
            )
            no_period = frame[frame["fiscal_period_end"].isna()]
            no_period = no_period[
                [
                    (t, d) not in covered
                    for t, d in zip(no_period["ticker"], no_period["report_date"], strict=True)
                ]
            ]
            no_period = (
                no_period.sort_values(["ticker", "report_date", "_has_time", "snapshot_date"])
                .groupby(["ticker", "report_date"], as_index=False, sort=False)
                .tail(1)
            )
            table = pd.concat([with_period, no_period], ignore_index=True)
            table = table.drop(columns="_has_time")
            self._table = table.sort_values(["ticker", "report_date"]).reset_index(drop=True)
        return self._table

    _SESSION_BY_REPORT_TIME = {
        "pre-market": Session.BMO,
        "post-market": Session.AMC,
        "intraday": Session.DMH,
    }

    def _fetch_calendar(
        self, start: dt.date, end: dt.date, tickers: list[Ticker] | None
    ) -> pd.DataFrame:
        table = self.table
        mask = (table["report_date"] >= pd.Timestamp(start)) & (
            table["report_date"] <= pd.Timestamp(end)
        )
        if tickers is not None:
            mask &= table["ticker"].isin(tickers)
        sub = table[mask]
        if not len(sub):
            return _empty_calendar()
        sessions = [
            self._SESSION_BY_REPORT_TIME.get(
                str(t).strip().lower() if isinstance(t, str) else "", Session.UNKNOWN
            )
            for t in sub["report_time"]
        ]
        out = pd.DataFrame(
            {
                "ticker": sub["ticker"].to_numpy(),
                "period_end": sub["fiscal_period_end"].to_numpy(),
                "announced_at": [
                    nominal_announced_at(d, s)
                    for d, s in zip(sub["report_date"], sessions, strict=True)
                ],
                "session": [s.value for s in sessions],
                "fiscal_quarter": [
                    fiscal_quarter_label(p) if not pd.isna(p) else ""
                    for p in sub["fiscal_period_end"]
                ],
                "eps_actual": sub["eps_reported"].to_numpy(),
                "eps_estimate": sub["eps_estimated"].to_numpy(),
                "revenue_actual": np.nan,
                "revenue_estimate": np.nan,
                "surprise": sub["surprise"].to_numpy(),
                "surprise_pct": sub["surprise_pct"].to_numpy(),
                "is_estimated_date": False,
                "announced_time_is_nominal": True,
                "source": ("external:" + sub["source"].astype(str)).to_numpy(),
            }
        )
        return out

    def _fetch_consensus(self, tickers: list[Ticker]) -> pd.DataFrame:
        table = self.table
        sub = table[table["ticker"].isin(tickers) & table["fiscal_period_end"].notna()]
        if not len(sub):
            return pd.DataFrame()
        sub = sub[sub["eps_estimated"].notna()]
        if not len(sub):
            return pd.DataFrame()
        return pd.DataFrame(
            {
                "ticker": sub["ticker"].to_numpy(),
                "period_end": sub["fiscal_period_end"].to_numpy(),
                # El único día en que esta cifra fue con certeza el consenso
                # vigente es el del anuncio. Cualquier as_of anterior sería
                # inventar un vintage que el dataset no contiene.
                "as_of": sub["report_date"].to_numpy(),
                "available_at": sub["report_date"].to_numpy(),
                "eps_mean": sub["eps_estimated"].to_numpy(),
                "eps_median": np.nan,
                "eps_std": np.nan,
                "eps_high": np.nan,
                "eps_low": np.nan,
                "n_analysts": np.nan,
                "revenue_mean": np.nan,
                "revenue_std": np.nan,
                "is_point_in_time": False,
                "source": ("external:" + sub["source"].astype(str)).to_numpy(),
            }
        )


# ===========================================================================
# 8. Sintético
# ===========================================================================


class SyntheticEstimatesProvider(EstimatesProviderBase):
    """Calendario y consenso del generador sintético. Sin red, sin credenciales.

    Es el único proveedor cuyo consenso es **point-in-time de verdad**: los
    `as_of` de `SyntheticMarket.estimates()` son la senda de revisiones
    generada (walk-down de Richardson-Teoh-Wysocki), estrictamente anterior al
    anuncio. Por eso se usa para probar `analyst_revision_drift` y el guardián
    `require_point_in_time`.
    """

    name = "synthetic"
    kinds: tuple[str, ...] = (CALENDAR_KIND, ESTIMATES_KIND)

    def __init__(
        self,
        market: SyntheticMarket | None = None,
        *,
        settings: Settings | None = None,
        **market_kwargs: Any,
    ) -> None:
        super().__init__(settings=settings)
        self._market = market
        self._market_kwargs = dict(market_kwargs)

    @property
    def market(self) -> SyntheticMarket:
        if self._market is None:
            kwargs = {"seed": self.settings.seed, **self._market_kwargs}
            self._market = SyntheticMarket(**kwargs)
        return self._market

    def available(self) -> bool:
        return True

    def _fetch_calendar(
        self, start: dt.date, end: dt.date, tickers: list[Ticker] | None
    ) -> pd.DataFrame:
        events = self.market.events(include_prehistory=True)
        announced = pd.to_datetime(events["announced_at"])
        mask = (announced >= pd.Timestamp(start)) & (
            announced < pd.Timestamp(end) + pd.Timedelta(days=1)
        )
        if tickers is not None:
            known = set(self.market.tickers)
            unknown = sorted(set(tickers) - known)
            if unknown:
                msg = f"synthetic: símbolos fuera del universo sintético: {unknown}"
                raise InsufficientHistory(msg)
            mask &= events["ticker"].isin(tickers)
        sub = events[mask]
        if not len(sub):
            return _empty_calendar()
        return pd.DataFrame(
            {
                "ticker": sub["ticker"].to_numpy(),
                "period_end": sub["period_end"].to_numpy(),
                "announced_at": sub["announced_at"].to_numpy(),
                "session": sub["session"].to_numpy(),
                "fiscal_quarter": sub["fiscal_quarter"].to_numpy(),
                "eps_actual": sub["eps_actual"].to_numpy(),
                "eps_estimate": sub["eps_estimate"].to_numpy(),
                "revenue_actual": sub["revenue_actual"].to_numpy(),
                "revenue_estimate": sub["revenue_estimate"].to_numpy(),
                "surprise": sub["eps_surprise"].to_numpy(),
                "surprise_pct": sub["surprise_pct"].to_numpy(),
                "is_estimated_date": sub["is_estimated_date"].to_numpy(),
                "announced_time_is_nominal": False,
                "source": "synthetic",
            }
        )

    def _fetch_consensus(self, tickers: list[Ticker]) -> pd.DataFrame:
        known = set(self.market.tickers)
        unknown = sorted(set(tickers) - known)
        if unknown:
            msg = f"synthetic: símbolos fuera del universo sintético: {unknown}"
            raise InsufficientHistory(msg)
        est = self.market.estimates(tickers=tickers)
        if not len(est):
            return pd.DataFrame()
        return pd.DataFrame(
            {
                "ticker": est["ticker"].to_numpy(),
                "period_end": est["period_end"].to_numpy(),
                "as_of": est["as_of"].to_numpy(),
                "available_at": est["available_at"].to_numpy(),
                "eps_mean": est["eps_mean"].to_numpy(),
                "eps_median": est["eps_median"].to_numpy(),
                "eps_std": est["eps_std"].to_numpy(),
                "eps_high": est["eps_high"].to_numpy(),
                "eps_low": est["eps_low"].to_numpy(),
                "n_analysts": est["n_analysts"].to_numpy(),
                "revenue_mean": est["revenue_mean"].to_numpy(),
                "revenue_std": est["revenue_std"].to_numpy(),
                "is_point_in_time": True,
                "source": "synthetic",
            }
        )


# ===========================================================================
# 9. Registro y fachadas
# ===========================================================================


DEFAULT_ESTIMATES_PRIORITIES: dict[str, int] = {
    # El dataset externo es real, gratuito y offline: manda para el histórico.
    "external_consensus": 50,
    # Finnhub da la etiqueta BMO/AMC explícita (la clave de tradable_date).
    "finnhub": 40,
    "fmp": 30,
    "eodhd": 20,
    "nasdaq": 10,
    # Red de seguridad sin red (contrato §0.5).
    "synthetic": 0,
}


def register_estimates_providers(
    registry: ProviderRegistry | None = None,
    *,
    settings: Settings | None = None,
    market: SyntheticMarket | None = None,
    external_path: Path | str | None = None,
    include: Sequence[str] | None = None,
    priorities: Mapping[str, int] | None = None,
) -> ProviderRegistry:
    """Registra los proveedores de calendario y consenso.

    Cada proveedor se registra bajo **ambos** kinds (`earnings_calendar` y
    `estimates`); la disponibilidad se evalúa al resolver, de modo que el
    mensaje de `ProviderUnavailable` enumera qué variable de entorno falta para
    cada candidato (contrato §3.3).
    """
    reg = registry or get_registry()
    cfg = settings or get_settings()
    chosen = dict(DEFAULT_ESTIMATES_PRIORITIES)
    if include is not None:
        unknown = sorted(set(include) - set(chosen))
        if unknown:
            msg = f"proveedores de estimaciones desconocidos en include: {unknown}"
            raise ConfigError(msg)
        chosen = {k: v for k, v in chosen.items() if k in set(include)}
    if priorities:
        chosen.update({k: int(v) for k, v in priorities.items() if k in chosen})

    factories: dict[str, Any] = {
        "external_consensus": lambda: ExternalConsensusProvider(external_path, settings=cfg),
        "finnhub": lambda: FinnhubEstimatesProvider(settings=cfg),
        "fmp": lambda: FMPEstimatesProvider(settings=cfg),
        "eodhd": lambda: EODHDEstimatesProvider(settings=cfg),
        "nasdaq": lambda: NasdaqEstimatesProvider(settings=cfg),
        "synthetic": lambda: SyntheticEstimatesProvider(market, settings=cfg),
    }
    for name, priority in chosen.items():
        provider = factories[name]()
        reg.register(CALENDAR_KIND, provider, priority, replace=True)
        reg.register(ESTIMATES_KIND, provider, priority, replace=True)
    return reg


def get_earnings_calendar(
    start: DateLike,
    end: DateLike,
    tickers: Ticker | Sequence[Ticker] | None = None,
    *,
    registry: ProviderRegistry | None = None,
) -> pd.DataFrame:
    """Fachada: calendario del mejor proveedor disponible, con fallback."""
    reg = registry or get_registry()
    s, e = _check_window(start, end)
    return reg.call(
        CALENDAR_KIND,
        lambda p: p.earnings_calendar(s, e, tickers),  # type: ignore[attr-defined]
        description=f"earnings_calendar[{s.isoformat()}..{e.isoformat()}]",
    )


def get_consensus(
    tickers: Ticker | Sequence[Ticker],
    *,
    start: DateLike | None = None,
    end: DateLike | None = None,
    point_in_time_only: bool = False,
    registry: ProviderRegistry | None = None,
) -> pd.DataFrame:
    """Fachada: consenso del mejor proveedor disponible, con fallback.

    Con ``point_in_time_only=True`` aplica `require_point_in_time` al
    resultado: es el modo que deben usar los factores de revisiones.
    """
    reg = registry or get_registry()
    out = reg.call(
        ESTIMATES_KIND,
        lambda p: p.consensus(tickers, start=start, end=end),  # type: ignore[attr-defined]
        description="consensus",
    )
    if point_in_time_only:
        require_point_in_time(out)
    return out
