"""Captura point-in-time de datos perecederos (`collector.snapshot`).

Este módulo fotografía **hoy** lo que dentro de un año no se podrá comprar a
ningún precio razonable:

1. **Cadenas de opciones completas** por subyacente (strikes, bid/ask, volumen,
   open interest, IV). No existe fuente histórica gratuita con licencia limpia
   (`docs/research/datasets_opciones.md` §1); la única vía con presupuesto cero
   es recolectar hacia adelante desde hoy (§5 del mismo documento).
2. **Consenso de analistas y calendario de próximos resultados.** Los vintages
   de consenso son el dato más caro del proyecto
   (`docs/research/data_sources.md` §6): capturarlos a diario construye, gratis,
   el panel `(ticker, period_end, as_of)` que de otro modo cuesta cuatro cifras
   al año (§6.3, salida 2).
3. **Short interest y volumen off-exchange** de FINRA, que solo mantiene un año
   rodante en línea (`data_sources.md` §8.2): lo que no se archiva al salir se
   pierde.

Hora de captura: por qué tras el cierre (16:15–17:00 ET) y siempre a la misma
--------------------------------------------------------------------------------
Una cadena de opciones fotografiada a las 11:00 y otra a las 15:45 no son
comparables: el volumen acumulado es distinto, la IV incorpora otro nivel del
subyacente y el bid/ask refleja otro régimen de liquidez intradía. Para que el
panel resultante sea una **serie** —y no una colección de fotos a horas
aleatorias— la pasada principal (`PASS_CLOSE`) debe ejecutarse:

- **después de las 16:15 ET**, cuando el subasta de cierre ha terminado, el
  volumen del día es final y las cotizaciones de opciones (que cierran a las
  16:00/16:15 ET) están asentadas;
- **antes de las ~17:00 ET**, para no mezclarse con recálculos vespertinos de
  los proveedores y para que la foto siga siendo "el cierre de hoy", no "la
  noche de hoy".

La consistencia horaria es en sí misma una propiedad point-in-time: si la hora
de captura varía, `captured_at` deja de ser intercambiable entre días y
cualquier feature construida sobre el panel hereda ese ruido.

La trampa del open interest (pasada matinal separada)
-----------------------------------------------------
El open interest que corresponde al cierre de la sesión `T` lo calcula la OCC en
su ciclo nocturno y **se disemina la mañana de `T+1`**
(`docs/research/data_sources.md` §7.3; `docs/research/informed_trading.md`
§sobre `oi_buildup`). Lo que un proveedor muestra durante la sesión `T` es el OI
de `T-1`. Por eso:

- la pasada de cierre guarda la cadena con el OI **que era conocible en ese
  momento** (el de `T-1`, correcto como dato point-in-time de la foto);
- una pasada matinal separada (`PASS_MORNING`, 08:30–09:15 ET) captura el OI ya
  actualizado y lo guarda en el dataset `open_interest` con `oi_date = T-1`
  (la sesión cuyo cierre describe) y `available_at = captured_at` **real** de la
  mañana de `T`. Usar como `available_at` el cierre de `T-1` sería un día entero
  de look-ahead, exactamente el error que el contrato §0.1 prohíbe.

`captured_at` es una cota superior conservadora de la disponibilidad pública
(la OCC publica hacia las 06:00 ET; nosotros registramos cuándo lo vimos). El
sesgo es del signo seguro: nunca se afirma haber sabido algo antes de tiempo.

Toda captura registra `source`, `captured_at` (UTC exacto, tz-aware),
`available_at` y `collector_version`, de modo que el panel acumulado es
auditable fila a fila.

Referencias
-----------
- `docs/research/datasets_opciones.md` §1, §5: el histórico de opciones no se
  puede comprar gratis; recolectar desde hoy es la única salida con coste cero.
- `docs/research/data_sources.md` §6.3: diseño del recolector de consenso
  (`as_of` = fecha de captura, payload del proveedor, append-only).
- `docs/research/data_sources.md` §7.3 y §8: PIT de open interest y calendario
  de publicación de FINRA.
- Cremers, M., Weinbaum, D. (2010). "Deviations from Put-Call Parity and Stock
  Return Predictability", *JFQA* 45(2): el `vol_spread` exige la cadena
  completa (call y put del mismo strike/vencimiento), no agregados: por eso se
  captura la cadena entera y no un resumen.
- Richardson, Teoh y Wysocki (2004): el walk-down del consenso; por qué el
  vintage diario (y no el consenso final) es el dato con valor.
"""

from __future__ import annotations

import datetime as dt
import importlib
import logging
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from earnings_alpha.config import Settings, get_settings
from earnings_alpha.data.base import (
    HttpClient,
    Transport,
)
from earnings_alpha.data.cache import SystemWallClock, WallClock
from earnings_alpha.data.estimates import (
    CALENDAR_COLUMNS,
    CONSENSUS_COLUMNS,
    EstimatesProviderBase,
)
from earnings_alpha.data.synthetic import SyntheticMarket
from earnings_alpha.errors import (
    ConfigError,
    DataQualityError,
    InsufficientHistory,
    ProviderUnavailable,
)
from earnings_alpha.pit import TradingCalendar, get_calendar, utc_to_eastern
from earnings_alpha.types import Ticker, normalize_ticker

__all__ = [  # noqa: RUF022 - orden temático, no alfabético
    # versión y pasadas
    "COLLECTOR_VERSION",
    "PASS_CLOSE",
    "PASS_MORNING",
    "CAPTURE_PASSES",
    "RECOMMENDED_CAPTURE_WINDOWS_ET",
    "capture_window_warning",
    # esquemas
    "STAMP_COLUMNS",
    "OPTION_CHAIN_COLUMNS",
    "OPEN_INTEREST_COLUMNS",
    "CONSENSUS_SNAPSHOT_COLUMNS",
    "CALENDAR_SNAPSHOT_COLUMNS",
    "OFF_EXCHANGE_COLUMNS",
    "SHORT_INTEREST_COLUMNS",
    "to_yahoo_symbol",
    # fuentes de cadenas
    "OptionChainSource",
    "YFinanceOptionsSource",
    "TradierOptionsSource",
    "SyntheticOptionsSource",
    # fuentes de flujo
    "RegShoSource",
    "FinraShortInterestSource",
    # capturas
    "snapshot_option_chain",
    "snapshot_open_interest",
    "snapshot_consensus",
    "snapshot_earnings_calendar",
    "snapshot_off_exchange",
    "snapshot_short_interest",
]

logger = logging.getLogger(__name__)

COLLECTOR_VERSION = "1.0.0"
"""Versión del recolector. Viaja en cada fila para poder auditar, meses después,
con qué código se capturó cada dato y reprocesar solo lo afectado por un bug."""

PASS_CLOSE = "close"
"""Pasada principal, tras el cierre (16:15–17:00 ET): cadena de opciones,
consenso y calendario."""

PASS_MORNING = "morning"
"""Pasada matinal (08:30–09:15 ET): open interest actualizado de la sesión
anterior, Reg SHO del día anterior y short interest quincenal si hay publicación."""

CAPTURE_PASSES: tuple[str, ...] = (PASS_CLOSE, PASS_MORNING)

RECOMMENDED_CAPTURE_WINDOWS_ET: dict[str, tuple[dt.time, dt.time]] = {
    PASS_CLOSE: (dt.time(16, 15), dt.time(17, 0)),
    PASS_MORNING: (dt.time(8, 30), dt.time(9, 15)),
}
"""Ventanas recomendadas en hora de Nueva York. Ver el docstring del módulo para
la justificación; `capture_window_warning` avisa (sin abortar) fuera de ellas."""

STAMP_COLUMNS: tuple[str, ...] = (
    "source",
    "captured_at",
    "available_at",
    "collector_version",
)
"""Columnas de auditoría presentes en TODOS los datasets del recolector."""

OPTION_CHAIN_COLUMNS: tuple[str, ...] = (
    "ticker",
    "chain_date",
    "expiry",
    "right",
    "strike",
    "bid",
    "ask",
    "last",
    "volume",
    "open_interest",
    "iv",
    "spot",
    *STAMP_COLUMNS,
)
"""Cadena de opciones de la pasada de cierre. `chain_date` es la sesión a la que
se atribuye la foto. ATENCIÓN: `open_interest` aquí es el OI *conocible en el
momento de la captura* (el del cierre anterior, ciclo nocturno de la OCC); el OI
del propio `chain_date` llega en el dataset `open_interest` de la pasada matinal."""

OPEN_INTEREST_COLUMNS: tuple[str, ...] = (
    "ticker",
    "oi_date",
    "expiry",
    "right",
    "strike",
    "open_interest",
    *STAMP_COLUMNS,
)
"""Open interest de la pasada matinal. `oi_date` = sesión cuyo cierre describe el
OI (la anterior a la captura); `available_at` = instante real de la captura
matinal. Es la materialización de la regla PIT de `data_sources.md` §7.3."""

CONSENSUS_SNAPSHOT_COLUMNS: tuple[str, ...] = (
    *CONSENSUS_COLUMNS,
    "captured_at",
    "collector_version",
)
"""Consenso canónico (`data.estimates.CONSENSUS_COLUMNS`) + sello del recolector.
`as_of` es SIEMPRE la fecha de la captura, no la que diga el proveedor
(`data_sources.md` §6.3): es lo que convierte la serie en vintages de verdad."""

CALENDAR_SNAPSHOT_COLUMNS: tuple[str, ...] = (
    *CALENDAR_COLUMNS,
    "capture_date",
    "captured_at",
    "available_at",
    "collector_version",
)
"""Calendario canónico (`data.estimates.CALENDAR_COLUMNS`) + sello. Se captura a
diario aunque cambie poco: la *historia* de fechas previstas (cuándo se movió un
anuncio) es en sí una señal y no se puede reconstruir a posteriori."""

OFF_EXCHANGE_COLUMNS: tuple[str, ...] = (
    "ticker",
    "trade_date",
    "short_volume",
    "short_exempt_volume",
    "total_volume",
    "market",
    *STAMP_COLUMNS,
)
"""Fichero diario Reg SHO de FINRA (`data_sources.md` §8.3): la única serie
diaria y gratuita de flujo off-exchange. `trade_date` es la sesión del dato."""

SHORT_INTEREST_COLUMNS: tuple[str, ...] = (
    "ticker",
    "settlement_date",
    "short_interest",
    "avg_daily_volume",
    "days_to_cover",
    *STAMP_COLUMNS,
)
"""Short interest quincenal consolidado de FINRA. `available_at` = captura (la
diseminación real es ~7º día hábil tras el settlement; capturar a diario y
deduplicar garantiza que `available_at` sea una cota superior honesta)."""


# ===========================================================================
# 1. Utilidades de sellado
# ===========================================================================


def _ensure_utc(ts: dt.datetime, label: str) -> dt.datetime:
    """Normaliza un instante a tz-aware UTC; naive o en otra zona son errores."""
    if ts.tzinfo is None:
        msg = f"{label} debe ser tz-aware (UTC); recibido naive: {ts!r}"
        raise ConfigError(msg)
    return ts.astimezone(dt.UTC)


def _session_date_et(captured_at: dt.datetime) -> dt.date:
    """Fecha de calendario en Nueva York del instante de captura.

    Es la fecha a la que se atribuye la foto: una captura a las 16:30 ET son las
    20:30/21:30 UTC del mismo día, pero el criterio correcto es la pared de ET,
    no la de UTC, porque el día bursátil se define allí.
    """
    utc_naive = _ensure_utc(captured_at, "captured_at").replace(tzinfo=None)
    return utc_to_eastern(utc_naive).date()


def capture_window_warning(captured_at: dt.datetime, pass_: str) -> str | None:
    """Mensaje de aviso si la captura cae fuera de la ventana recomendada.

    No aborta: un cron que arranca con retraso debe capturar de todos modos
    (cada día perdido es irrecuperable), pero el desvío queda registrado para
    que el usuario corrija el horario.
    """
    if pass_ not in RECOMMENDED_CAPTURE_WINDOWS_ET:
        msg = f"pasada desconocida: {pass_!r}; válidas: {CAPTURE_PASSES}"
        raise ConfigError(msg)
    lo, hi = RECOMMENDED_CAPTURE_WINDOWS_ET[pass_]
    utc_naive = _ensure_utc(captured_at, "captured_at").replace(tzinfo=None)
    et = utc_to_eastern(utc_naive)
    if lo <= et.time() <= hi:
        return None
    return (
        f"captura de la pasada {pass_!r} a las {et.time().isoformat('minutes')} ET, "
        f"fuera de la ventana recomendada {lo.isoformat('minutes')}-{hi.isoformat('minutes')} ET; "
        "la foto es válida pero no es comparable hora a hora con las demás"
    )


def _stamp(
    frame: pd.DataFrame,
    *,
    source: str,
    captured_at: dt.datetime,
    available_at: dt.datetime | None = None,
) -> pd.DataFrame:
    """Añade las columnas de auditoría a un DataFrame de captura."""
    cap = _ensure_utc(captured_at, "captured_at")
    avail = cap if available_at is None else _ensure_utc(available_at, "available_at")
    out = frame.copy()
    out["source"] = str(source)
    out["captured_at"] = pd.Timestamp(cap)
    out["available_at"] = pd.Timestamp(avail)
    out["collector_version"] = COLLECTOR_VERSION
    return out


def to_yahoo_symbol(ticker: Ticker) -> str:
    """Convierte el símbolo canónico del repo al formato de Yahoo.

    El repo usa punto (`BRK.B`, convención SEC); Yahoo usa guion (`BRK-B`).
    """
    return normalize_ticker(ticker).replace(".", "-")


def _numeric(series: Any, index: pd.Index | None = None) -> pd.Series:
    return pd.to_numeric(pd.Series(series, index=index), errors="coerce")


# ===========================================================================
# 2. Fuentes de cadenas de opciones
# ===========================================================================


@runtime_checkable
class OptionChainSource(Protocol):
    """Contrato mínimo de una fuente de cadenas de opciones.

    `fetch_chain` devuelve la cadena cruda (sin sellos) con al menos las
    columnas `expiry, right, strike, bid, ask, last, volume, open_interest, iv,
    spot`. El sellado (`captured_at`, `available_at`, versión) lo hace
    `snapshot_option_chain`, no la fuente: así el sello es homogéneo aunque las
    fuentes sean heterogéneas.
    """

    name: str

    def available(self) -> bool: ...

    def fetch_chain(self, ticker: Ticker, *, max_expiries: int | None = None) -> pd.DataFrame: ...

    def estimated_requests(self, *, max_expiries: int | None = None) -> int:
        """Peticiones HTTP que costará una cadena; para el presupuesto del scheduler."""
        ...


_RAW_CHAIN_COLUMNS = (
    "expiry",
    "right",
    "strike",
    "bid",
    "ask",
    "last",
    "volume",
    "open_interest",
    "iv",
    "spot",
)


def _finalize_raw_chain(frame: pd.DataFrame, provider: str, ticker: Ticker) -> pd.DataFrame:
    """Normaliza la cadena cruda de cualquier fuente al esquema común.

    Vacía -> `InsufficientHistory` (contrato §0.3: nada de DataFrames vacíos
    silenciosos que meses después parezcan "ese día no había opciones").
    """
    if frame is None or len(frame) == 0:
        msg = f"{provider}: la cadena de opciones de {ticker} llegó vacía"
        raise InsufficientHistory(msg)
    out = frame.copy()
    missing = [c for c in _RAW_CHAIN_COLUMNS if c not in out.columns]
    if missing:
        msg = f"{provider}: faltan columnas de la cadena cruda: {missing}"
        raise DataQualityError(msg)
    out["expiry"] = pd.to_datetime(out["expiry"], errors="coerce")
    out["right"] = out["right"].astype(str).str.upper().str[0]
    bad_right = ~out["right"].isin(["C", "P"])
    if bad_right.any():
        msg = f"{provider}: lados de opción no interpretables: {sorted(out.loc[bad_right, 'right'].unique())}"
        raise DataQualityError(msg)
    for col in ("strike", "bid", "ask", "last", "volume", "open_interest", "iv", "spot"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["expiry", "strike"])
    if len(out) == 0:
        msg = f"{provider}: la cadena de {ticker} no tiene ninguna fila con expiry y strike válidos"
        raise DataQualityError(msg)
    if (out["strike"] <= 0).any():
        msg = f"{provider}: strikes no positivos en la cadena de {ticker}"
        raise DataQualityError(msg)
    return out[list(_RAW_CHAIN_COLUMNS)].reset_index(drop=True)


def _import_yfinance() -> Any:  # pragma: no cover - envoltura trivial de import
    """Importa yfinance; separado para poder simular su ausencia en tests."""
    return importlib.import_module("yfinance")


class YFinanceOptionsSource:
    """Cadenas de opciones vía yfinance (Yahoo Finance no oficial).

    Ventajas: sin credenciales y con la cadena completa (bid/ask, volumen, OI e
    IV). Limitaciones documentadas (`data_sources.md` §3.2-yfinance): sin
    griegas, IV de calidad media, límite de tasa no documentado y cambiante
    (~1 símbolo/segundo es lo prudente; la política `yfinance` de
    `PROVIDER_RATE_LIMITS` ya lo impone al resto del repo, pero yfinance usa su
    propio transporte, así que la cortesía aquí la aporta el scheduler al
    espaciar tickers).

    El cliente es inyectable (`client=`) para poder probar sin red con un objeto
    que imite `yfinance.Ticker`.
    """

    name = "yfinance"

    def __init__(
        self,
        *,
        client: Any | None = None,
        max_expiries: int = 6,
        settings: Settings | None = None,
    ) -> None:
        if max_expiries < 1:
            msg = f"max_expiries debe ser >= 1; recibido {max_expiries}"
            raise ConfigError(msg)
        self.settings = settings or get_settings()
        self.max_expiries = int(max_expiries)
        self._client = client
        self.requests_made = 0

    def _load_client(self) -> Any:
        if self._client is None:
            try:
                self._client = _import_yfinance()
            except ImportError as exc:
                raise ProviderUnavailable(
                    self.name,
                    "falta la dependencia opcional 'yfinance' "
                    "(instala con: pip install 'earnings-alpha[providers]' o pip install yfinance)",
                ) from exc
        return self._client

    def available(self) -> bool:
        if self._client is not None:
            return True
        return importlib.util.find_spec("yfinance") is not None

    def estimated_requests(self, *, max_expiries: int | None = None) -> int:
        n = int(max_expiries or self.max_expiries)
        return 1 + n  # lista de vencimientos + una petición por vencimiento

    def fetch_chain(self, ticker: Ticker, *, max_expiries: int | None = None) -> pd.DataFrame:
        client = self._load_client()
        symbol = to_yahoo_symbol(ticker)
        n_exp = int(max_expiries or self.max_expiries)
        try:
            tk = client.Ticker(symbol)
            expiries: Sequence[str] = tuple(tk.options or ())
            self.requests_made += 1
        except ProviderUnavailable:
            raise
        except Exception as exc:  # yfinance lanza tipos variados y no documentados
            raise ProviderUnavailable(
                self.name, f"fallo al listar vencimientos de {ticker}: {type(exc).__name__}: {exc}"
            ) from exc
        if not expiries:
            msg = f"{self.name}: {ticker} no tiene vencimientos de opciones listados"
            raise InsufficientHistory(msg)

        spot = self._spot(tk)
        blocks: list[pd.DataFrame] = []
        for expiry in list(expiries)[:n_exp]:
            try:
                chain = tk.option_chain(expiry)
                self.requests_made += 1
            except Exception as exc:
                raise ProviderUnavailable(
                    self.name,
                    f"fallo al pedir la cadena de {ticker} vencimiento {expiry}: "
                    f"{type(exc).__name__}: {exc}",
                ) from exc
            for right, leg in (("C", chain.calls), ("P", chain.puts)):
                if leg is None or len(leg) == 0:
                    continue
                blocks.append(
                    pd.DataFrame(
                        {
                            "expiry": expiry,
                            "right": right,
                            "strike": _numeric(leg.get("strike")),
                            "bid": _numeric(leg.get("bid")),
                            "ask": _numeric(leg.get("ask")),
                            "last": _numeric(leg.get("lastPrice")),
                            "volume": _numeric(leg.get("volume")),
                            "open_interest": _numeric(leg.get("openInterest")),
                            "iv": _numeric(leg.get("impliedVolatility")),
                            "spot": spot,
                        }
                    )
                )
        raw = pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()
        return _finalize_raw_chain(raw, self.name, ticker)

    @staticmethod
    def _spot(tk: Any) -> float:
        """Último precio del subyacente; NaN si el cliente no lo expone."""
        try:
            info = getattr(tk, "fast_info", None)
            if info is not None:
                value = info["last_price"] if "last_price" in info else None
                if value is not None:
                    return float(value)
        except Exception:  # noqa: BLE001 - el spot es opcional, la cadena no
            logger.debug("no se pudo leer el spot de yfinance; se deja NaN")
        return float("nan")


class TradierOptionsSource:
    """Cadenas de opciones vía la API de Tradier (sandbox por defecto).

    Tradier sirve la cadena completa con IV y griegas *cortesía de ORATS*
    (`data_sources.md` §7.2-Tradier). No tiene histórico —solo el snapshot
    vivo—, que es exactamente lo que un recolector necesita. El sandbox
    (`sandbox.tradier.com`) es gratuito con una cuenta de desarrollador y sirve
    datos con 15 minutos de retraso: para una foto de cierre tomada a las
    16:30 ET, el retraso es irrelevante.

    Requiere `TRADIER_ACCESS_TOKEN`; la URL base se puede fijar con
    `TRADIER_API_BASE` (p. ej. `https://api.tradier.com/v1` para producción).
    El transporte es inyectable para probar sin red.
    """

    name = "tradier"
    SANDBOX_BASE = "https://sandbox.tradier.com/v1"
    PRODUCTION_BASE = "https://api.tradier.com/v1"

    def __init__(
        self,
        *,
        http: HttpClient | None = None,
        transport: Transport | None = None,
        base_url: str | None = None,
        max_expiries: int = 6,
        settings: Settings | None = None,
    ) -> None:
        if max_expiries < 1:
            msg = f"max_expiries debe ser >= 1; recibido {max_expiries}"
            raise ConfigError(msg)
        self.settings = settings or get_settings()
        self.base_url = (
            base_url or self.settings.env("TRADIER_API_BASE") or self.SANDBOX_BASE
        ).rstrip("/")
        self.max_expiries = int(max_expiries)
        self._http = http
        self._transport = transport
        self.requests_made = 0

    def _client(self) -> HttpClient:
        if self._http is None:
            token = self.settings.env("TRADIER_ACCESS_TOKEN")
            if not token:
                raise ProviderUnavailable(
                    self.name,
                    "falta el token de Tradier (sandbox gratuito en developer.tradier.com)",
                    missing_env=["TRADIER_ACCESS_TOKEN"],
                )
            self._http = HttpClient.for_provider(
                self.name,
                settings=self.settings,
                transport=self._transport,
                default_headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/json",
                },
            )
        return self._http

    def available(self) -> bool:
        return self._http is not None or bool(self.settings.env("TRADIER_ACCESS_TOKEN"))

    def estimated_requests(self, *, max_expiries: int | None = None) -> int:
        n = int(max_expiries or self.max_expiries)
        return 1 + n

    def fetch_chain(self, ticker: Ticker, *, max_expiries: int | None = None) -> pd.DataFrame:
        http = self._client()
        symbol = normalize_ticker(ticker)
        n_exp = int(max_expiries or self.max_expiries)

        payload = http.get_json(
            f"{self.base_url}/markets/options/expirations",
            params={"symbol": symbol, "includeAllRoots": "true"},
        )
        self.requests_made += 1
        expiries = self._parse_expirations(payload)
        if not expiries:
            msg = f"{self.name}: {ticker} no tiene vencimientos de opciones listados"
            raise InsufficientHistory(msg)

        blocks: list[pd.DataFrame] = []
        for expiry in expiries[:n_exp]:
            chain_payload = http.get_json(
                f"{self.base_url}/markets/options/chains",
                params={"symbol": symbol, "expiration": expiry, "greeks": "true"},
            )
            self.requests_made += 1
            rows = self._parse_chain(chain_payload)
            if rows:
                blocks.append(pd.DataFrame(rows))
        raw = pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()
        return _finalize_raw_chain(raw, self.name, ticker)

    @staticmethod
    def _parse_expirations(payload: Any) -> list[str]:
        if not isinstance(payload, dict):
            msg = f"tradier: respuesta de vencimientos no es un objeto JSON: {type(payload).__name__}"
            raise DataQualityError(msg)
        block = payload.get("expirations")
        if block in (None, "null"):
            return []
        dates = block.get("date") if isinstance(block, dict) else None
        if dates is None:
            return []
        if isinstance(dates, str):
            return [dates]
        return [str(d) for d in dates]

    @staticmethod
    def _parse_chain(payload: Any) -> list[dict[str, Any]]:
        if not isinstance(payload, dict):
            msg = f"tradier: respuesta de cadena no es un objeto JSON: {type(payload).__name__}"
            raise DataQualityError(msg)
        block = payload.get("options")
        if block in (None, "null"):
            return []
        options = block.get("option") if isinstance(block, dict) else None
        if options is None:
            return []
        if isinstance(options, dict):
            options = [options]
        rows: list[dict[str, Any]] = []
        for opt in options:
            greeks = opt.get("greeks") or {}
            side = str(opt.get("option_type", "")).lower()
            rows.append(
                {
                    "expiry": opt.get("expiration_date"),
                    "right": "C" if side.startswith("c") else ("P" if side.startswith("p") else "?"),
                    "strike": opt.get("strike"),
                    "bid": opt.get("bid"),
                    "ask": opt.get("ask"),
                    "last": opt.get("last"),
                    "volume": opt.get("volume"),
                    "open_interest": opt.get("open_interest"),
                    "iv": greeks.get("mid_iv", greeks.get("smv_vol")),
                    "spot": np.nan,
                }
            )
        return rows


class SyntheticOptionsSource:
    """Cadenas del generador sintético: la red de seguridad sin red (contrato §0.5).

    Sirve la cadena de la **última sesión ya cerrada** respecto al reloj de
    pared inyectado: antes de las 16:00 ET de un día de sesión, la "última foto
    posible" es la del cierre anterior, igual que en el mundo real. Ese detalle
    es lo que permite probar la pasada matinal de open interest sin introducir
    un look-ahead de un día en el propio banco de pruebas.
    """

    name = "synthetic"

    def __init__(
        self,
        market: SyntheticMarket | None = None,
        *,
        wall: WallClock | None = None,
        max_expiries: int = 4,
        settings: Settings | None = None,
        **market_kwargs: Any,
    ) -> None:
        self.settings = settings or get_settings()
        self.wall: WallClock = wall or SystemWallClock()
        self.max_expiries = int(max_expiries)
        self._market = market
        self._market_kwargs = dict(market_kwargs)
        self.requests_made = 0

    @property
    def market(self) -> SyntheticMarket:
        if self._market is None:
            kwargs = {"seed": self.settings.seed, **self._market_kwargs}
            self._market = SyntheticMarket(**kwargs)
        return self._market

    def available(self) -> bool:
        return True

    def estimated_requests(self, *, max_expiries: int | None = None) -> int:
        return 0  # sin red: no consume presupuesto de peticiones

    def _last_closed_session(self) -> pd.Timestamp:
        now_utc = _ensure_utc(self.wall.now(), "wall.now()").replace(tzinfo=None)
        et = utc_to_eastern(now_utc)
        cutoff = pd.Timestamp(et.date())
        sessions = self.market.sessions
        if et.time() < dt.time(16, 0):
            candidates = sessions[sessions < cutoff]
        else:
            candidates = sessions[sessions <= cutoff]
        if len(candidates) == 0:
            msg = (
                f"synthetic: no hay ninguna sesión cerrada antes de {et.isoformat()} "
                f"en el panel [{self.market.start}..{self.market.end}]"
            )
            raise InsufficientHistory(msg)
        return candidates[-1]

    def fetch_chain(self, ticker: Ticker, *, max_expiries: int | None = None) -> pd.DataFrame:
        symbol = normalize_ticker(ticker)
        if symbol not in self.market.tickers:
            msg = f"synthetic: {symbol} no pertenece al universo sintético"
            raise InsufficientHistory(msg)
        asof = self._last_closed_session()
        n_exp = int(max_expiries or self.max_expiries)
        chain = self.market.options_chain(symbol, asof=asof, n_expiries=n_exp)
        raw = pd.DataFrame(
            {
                "expiry": chain["expiry"],
                "right": chain["right"],
                "strike": chain["strike"],
                "bid": chain["bid"],
                "ask": chain["ask"],
                "last": chain["mid"],
                "volume": chain["volume"],
                "open_interest": chain["open_interest"],
                "iv": chain["iv"],
                "spot": chain["spot"],
            }
        )
        return _finalize_raw_chain(raw, self.name, symbol)


# ===========================================================================
# 3. Fuentes de flujo (FINRA)
# ===========================================================================


class RegShoSource:
    """Fichero diario Reg SHO de FINRA: volumen en corto y total off-exchange.

    `data_sources.md` §8.3 y §9.3: es la única serie diaria y gratuita de flujo
    direccional, y el proxy correcto (T+1) para `off_exchange_share_delta`, en
    lugar del dato ATS semanal que llega con dos semanas de retraso. Se intenta
    primero el CDN actual y después el host histórico; ambos sin autenticación.

    Advertencia de interpretación (obligatoria aguas abajo): short volume ≠
    short interest; gran parte es hedging de creadores de mercado (Blocher y
    Ringgenberg). Aquí solo se archiva; la interpretación es de `events.flow`.
    """

    name = "finra_regsho"
    URL_TEMPLATES: tuple[str, ...] = (
        "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{yyyymmdd}.txt",
        "http://regsho.finra.org/CNMSshvol{yyyymmdd}.txt",
    )

    def __init__(
        self,
        *,
        http: HttpClient | None = None,
        transport: Transport | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._http = http
        self._transport = transport
        self.requests_made = 0

    def _client(self) -> HttpClient:
        if self._http is None:
            self._http = HttpClient(
                "finra", settings=self.settings, transport=self._transport
            )
        return self._http

    def available(self) -> bool:
        return True  # sin credenciales; la red se comprueba al pedir

    def estimated_requests(self) -> int:
        return 1

    def fetch_daily(self, trade_date: dt.date) -> pd.DataFrame:
        """Descarga y parsea el consolidado NMS del día `trade_date` (crudo)."""
        http = self._client()
        stamp = trade_date.strftime("%Y%m%d")
        last_status: int | None = None
        for template in self.URL_TEMPLATES:
            url = template.format(yyyymmdd=stamp)
            resp = http.get(url, allow_status=(404,))
            self.requests_made += 1
            if resp.status_code == 404:
                last_status = 404
                continue
            return self._parse(resp.text, trade_date)
        msg = (
            f"{self.name}: no hay fichero Reg SHO para {trade_date.isoformat()} "
            f"(HTTP {last_status}); FINRA lo publica tras el cierre o T+1 — "
            "si la fecha es festivo o muy reciente, es normal"
        )
        raise InsufficientHistory(msg)

    @staticmethod
    def _parse(text: str, trade_date: dt.date) -> pd.DataFrame:
        expected = trade_date.strftime("%Y%m%d")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines or not lines[0].startswith("Date|Symbol|ShortVolume"):
            msg = "finra_regsho: cabecera inesperada; el formato del fichero ha cambiado"
            raise DataQualityError(msg)
        rows: list[dict[str, Any]] = []
        for ln in lines[1:]:
            parts = ln.split("|")
            if len(parts) < 6:
                continue  # línea de recuento final u otro pie de fichero
            if parts[0] != expected:
                msg = (
                    f"finra_regsho: el fichero contiene la fecha {parts[0]} y se pidió "
                    f"{expected}; se aborta para no archivar el día equivocado"
                )
                raise DataQualityError(msg)
            rows.append(
                {
                    "ticker": normalize_ticker(parts[1]),
                    "trade_date": trade_date,
                    "short_volume": parts[2],
                    "short_exempt_volume": parts[3],
                    "total_volume": parts[4],
                    "market": parts[5],
                }
            )
        if not rows:
            msg = f"finra_regsho: el fichero de {trade_date.isoformat()} no contiene filas de datos"
            raise DataQualityError(msg)
        out = pd.DataFrame(rows)
        for col in ("short_volume", "short_exempt_volume", "total_volume"):
            out[col] = pd.to_numeric(out[col], errors="coerce")
        return out


class FinraShortInterestSource:
    """Short interest quincenal consolidado vía la Query API de FINRA.

    Endpoint documentado en `data_sources.md` §8.2. El acceso anónimo funciona
    con cuota reducida [verificar]; con credenciales del FINRA Developer Center
    (gratuitas) la cuota sube. Los nombres de campo de la respuesta varían entre
    versiones de la API, así que el parser acepta los alias conocidos y falla
    con `DataQualityError` si no reconoce ninguno (mejor un fallo ruidoso que
    archivar columnas vacías).
    """

    name = "finra_short_interest"
    ENDPOINT = "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"

    _SYMBOL_KEYS = ("symbolCode", "issueSymbolIdentifier", "symbol")
    _SETTLEMENT_KEYS = ("settlementDate", "settlement_date")
    _SI_KEYS = ("currentShortPositionQuantity", "shortInterest", "currentShortPosition")
    _ADV_KEYS = ("averageDailyVolumeQuantity", "averageDailyVolume")
    _DTC_KEYS = ("daysToCoverQuantity", "daysToCover")

    def __init__(
        self,
        *,
        http: HttpClient | None = None,
        transport: Transport | None = None,
        settings: Settings | None = None,
        limit: int = 20000,
    ) -> None:
        self.settings = settings or get_settings()
        self._http = http
        self._transport = transport
        self.limit = int(limit)
        self.requests_made = 0

    def _client(self) -> HttpClient:
        if self._http is None:
            self._http = HttpClient(
                "finra", settings=self.settings, transport=self._transport
            )
        return self._http

    def available(self) -> bool:
        return True

    def estimated_requests(self) -> int:
        return 1

    def fetch_latest(self, settlement_date: dt.date | None = None) -> pd.DataFrame:
        """Última publicación (o la del `settlement_date` pedido), cruda."""
        http = self._client()
        body: dict[str, Any] = {"limit": self.limit}
        if settlement_date is not None:
            body["compareFilters"] = [
                {
                    "compareType": "EQUAL",
                    "fieldName": "settlementDate",
                    "fieldValue": settlement_date.isoformat(),
                }
            ]
        import json as _json

        resp = http.post(
            self.ENDPOINT,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            body=_json.dumps(body).encode("utf-8"),
        )
        self.requests_made += 1
        payload = resp.json()
        if not isinstance(payload, list):
            msg = f"{self.name}: se esperaba una lista JSON; llegó {type(payload).__name__}"
            raise DataQualityError(msg)
        if not payload:
            when = settlement_date.isoformat() if settlement_date else "la última publicación"
            msg = f"{self.name}: sin registros para {when}"
            raise InsufficientHistory(msg)
        return self._normalize(payload)

    @classmethod
    def _pick(cls, record: dict[str, Any], keys: Sequence[str]) -> Any:
        for k in keys:
            if k in record:
                return record[k]
        return None

    @classmethod
    def _normalize(cls, records: list[dict[str, Any]]) -> pd.DataFrame:
        sample = records[0]
        if cls._pick(sample, cls._SYMBOL_KEYS) is None or cls._pick(
            sample, cls._SETTLEMENT_KEYS
        ) is None:
            msg = (
                f"{cls.name}: no se reconocen los campos de símbolo/fecha en la respuesta "
                f"(claves presentes: {sorted(sample)[:12]}); actualizar los alias del parser"
            )
            raise DataQualityError(msg)
        rows = [
            {
                "ticker": normalize_ticker(str(cls._pick(r, cls._SYMBOL_KEYS))),
                "settlement_date": cls._pick(r, cls._SETTLEMENT_KEYS),
                "short_interest": cls._pick(r, cls._SI_KEYS),
                "avg_daily_volume": cls._pick(r, cls._ADV_KEYS),
                "days_to_cover": cls._pick(r, cls._DTC_KEYS),
            }
            for r in records
        ]
        out = pd.DataFrame(rows)
        out["settlement_date"] = pd.to_datetime(out["settlement_date"], errors="coerce").dt.date
        if out["settlement_date"].isna().any():
            msg = f"{cls.name}: fechas de liquidación no interpretables"
            raise DataQualityError(msg)
        for col in ("short_interest", "avg_daily_volume", "days_to_cover"):
            out[col] = pd.to_numeric(out[col], errors="coerce")
        return out


# ===========================================================================
# 4. Capturas selladas
# ===========================================================================


def snapshot_option_chain(
    source: OptionChainSource,
    ticker: Ticker,
    *,
    wall: WallClock | None = None,
    max_expiries: int | None = None,
) -> pd.DataFrame:
    """Fotografía la cadena de `ticker` y la sella (pasada de cierre).

    `chain_date` = fecha de sesión (ET) del instante de captura;
    `available_at` = `captured_at`. El open interest incluido es el conocible en
    ese momento (el del cierre anterior); el OI del propio día llega por
    `snapshot_open_interest` a la mañana siguiente.
    """
    clock = wall or SystemWallClock()
    raw = source.fetch_chain(ticker, max_expiries=max_expiries)
    captured = clock.now()
    warning = capture_window_warning(captured, PASS_CLOSE)
    if warning:
        logger.warning("%s: %s", ticker, warning)
    out = raw.copy()
    out.insert(0, "ticker", normalize_ticker(ticker))
    out.insert(1, "chain_date", pd.Timestamp(_session_date_et(captured)))
    out = _stamp(out, source=source.name, captured_at=captured)
    return out[list(OPTION_CHAIN_COLUMNS)]


def snapshot_open_interest(
    source: OptionChainSource,
    ticker: Ticker,
    *,
    calendar: TradingCalendar | None = None,
    wall: WallClock | None = None,
    max_expiries: int | None = None,
) -> pd.DataFrame:
    """Captura matinal del open interest actualizado (pasada `morning`).

    Vuelve a pedir la cadena y conserva solo las columnas de OI. `oi_date` es la
    sesión **anterior** a la fecha ET de la captura: el OI que un proveedor
    muestra por la mañana describe el cierre de ayer, tras el ciclo nocturno de
    la OCC (`data_sources.md` §7.3). `available_at` = `captured_at` real de la
    mañana: es la cota superior honesta de cuándo el dato fue público.
    """
    clock = wall or SystemWallClock()
    cal = calendar or get_calendar()
    raw = source.fetch_chain(ticker, max_expiries=max_expiries)
    captured = clock.now()
    warning = capture_window_warning(captured, PASS_MORNING)
    if warning:
        logger.warning("%s: %s", ticker, warning)
    et_date = _session_date_et(captured)
    oi_date = cal.prev_session(et_date)
    out = raw[["expiry", "right", "strike", "open_interest"]].copy()
    out.insert(0, "ticker", normalize_ticker(ticker))
    out.insert(1, "oi_date", pd.Timestamp(oi_date))
    out = _stamp(out, source=source.name, captured_at=captured)
    return out[list(OPEN_INTEREST_COLUMNS)]


def snapshot_consensus(
    provider: EstimatesProviderBase,
    tickers: Sequence[Ticker],
    *,
    wall: WallClock | None = None,
    min_period_end_lag_days: int = 120,
) -> pd.DataFrame:
    """Foto diaria del consenso vigente para los trimestres aún no anunciados.

    Aplica el diseño de `data_sources.md` §6.3 (salida 2):

    - se conserva **una** fila por `(ticker, period_end)`: la más reciente que
      declare el proveedor;
    - `as_of` se sobreescribe con la **fecha de la captura** — no la que diga el
      proveedor — porque lo que este panel registra es "qué consenso era visible
      hoy", que es la definición de vintage;
    - `is_point_in_time=True`: capturado en vivo, es de los pocos consensos del
      repo que satisface `require_point_in_time` de pleno derecho;
    - se filtran trimestres con `period_end` anterior a
      `captura - min_period_end_lag_days`: ya anunciados con seguridad, su
      "consenso actual" es una mezcla post-anuncio sin valor de vintage.
    """
    clock = wall or SystemWallClock()
    frame = provider.consensus(tickers)
    captured = clock.now()
    capture_date = _session_date_et(captured)
    cutoff = pd.Timestamp(capture_date - dt.timedelta(days=int(min_period_end_lag_days)))
    frame = frame[frame["period_end"] >= cutoff]
    if len(frame) == 0:
        msg = (
            f"{provider.name}: sin consenso para trimestres posteriores a "
            f"{cutoff.date().isoformat()}; nada que capturar"
        )
        raise InsufficientHistory(msg)
    frame = (
        frame.sort_values("as_of")
        .groupby(["ticker", "period_end"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    out = frame.copy()
    out["as_of"] = pd.Timestamp(capture_date)
    out["is_point_in_time"] = True
    cap = pd.Timestamp(_ensure_utc(captured, "captured_at"))
    out["available_at"] = cap
    out["captured_at"] = cap
    out["collector_version"] = COLLECTOR_VERSION
    return out[list(CONSENSUS_SNAPSHOT_COLUMNS)].sort_values(
        ["ticker", "period_end"]
    ).reset_index(drop=True)


def snapshot_earnings_calendar(
    provider: EstimatesProviderBase,
    start: dt.date,
    end: dt.date,
    *,
    wall: WallClock | None = None,
    tickers: Sequence[Ticker] | None = None,
) -> pd.DataFrame:
    """Foto diaria del calendario de próximos resultados en ``[start, end]``.

    Se guarda entera cada día, aunque cambie poco: la trayectoria de una fecha
    prevista (confirmación, adelanto, retraso) no puede reconstruirse después, y
    los retrasos de anuncio son en sí informativos. `capture_date` es la columna
    de partición: cada día produce su propio vintage del calendario.
    """
    clock = wall or SystemWallClock()
    frame = provider.earnings_calendar(start, end, tickers)
    captured = clock.now()
    out = frame.copy()
    out["capture_date"] = pd.Timestamp(_session_date_et(captured))
    out = _stamp(out, source=str(out["source"].iloc[0]), captured_at=captured)
    # `source` por fila puede diferir del proveedor de fachada; se conserva la
    # columna original del frame y solo se añade el sello temporal.
    out["source"] = frame["source"].to_numpy()
    return out[list(CALENDAR_SNAPSHOT_COLUMNS)]


def snapshot_off_exchange(
    source: RegShoSource,
    trade_date: dt.date,
    *,
    wall: WallClock | None = None,
    tickers: Sequence[Ticker] | None = None,
) -> pd.DataFrame:
    """Captura el fichero Reg SHO del día `trade_date` (pasada matinal de T+1).

    `available_at = captured_at`: FINRA publica el fichero tras el cierre o en
    T+1 (`data_sources.md` §8.3); registrar el instante en que lo vimos es la
    cota superior conservadora correcta.
    """
    clock = wall or SystemWallClock()
    raw = source.fetch_daily(trade_date)
    captured = clock.now()
    if tickers is not None:
        wanted = {normalize_ticker(t) for t in tickers}
        raw = raw[raw["ticker"].isin(wanted)]
        if len(raw) == 0:
            msg = (
                f"{source.name}: ninguno de los {len(wanted)} símbolos pedidos aparece en el "
                f"fichero Reg SHO de {trade_date.isoformat()}"
            )
            raise InsufficientHistory(msg)
    out = raw.copy()
    out["trade_date"] = pd.Timestamp(trade_date)
    out = _stamp(out, source=source.name, captured_at=captured)
    return out[list(OFF_EXCHANGE_COLUMNS)].sort_values(["ticker"]).reset_index(drop=True)


def snapshot_short_interest(
    source: FinraShortInterestSource,
    *,
    settlement_date: dt.date | None = None,
    wall: WallClock | None = None,
    tickers: Sequence[Ticker] | None = None,
) -> pd.DataFrame:
    """Captura la publicación de short interest (última, o la del settlement dado).

    Pensada para ejecutarse en **cada** pasada matinal: los días sin publicación
    nueva, la deduplicación del almacén (clave `settlement_date, ticker, source`)
    descarta las filas repetidas y la operación es un no-op idempotente. Así el
    `available_at` de una publicación queda fijado por la primera mañana en que
    se vio — la aproximación honesta a la fecha de diseminación de FINRA.
    """
    clock = wall or SystemWallClock()
    raw = source.fetch_latest(settlement_date)
    captured = clock.now()
    if tickers is not None:
        wanted = {normalize_ticker(t) for t in tickers}
        raw = raw[raw["ticker"].isin(wanted)]
        if len(raw) == 0:
            msg = f"{source.name}: ningún símbolo pedido aparece en la publicación"
            raise InsufficientHistory(msg)
    out = raw.copy()
    out["settlement_date"] = pd.to_datetime(out["settlement_date"])
    out = _stamp(out, source=source.name, captured_at=captured)
    return out[list(SHORT_INTEREST_COLUMNS)].sort_values(
        ["settlement_date", "ticker"]
    ).reset_index(drop=True)
