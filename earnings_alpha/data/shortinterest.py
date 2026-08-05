"""Short interest FINRA y volumen off-exchange/ATS (`data.flow`).

Este módulo adapta las tres fuentes de flujo con **retardo institucional de
publicación** descritas en `docs/research/informed_trading.md` §8 y
`docs/research/data_sources.md` §8-§9:

1. **Short interest quincenal** (FINRA `consolidatedShortInterest`): foto de
   posiciones cortas en la *settlement date* del día 15 y del último día hábil
   de cada mes.
2. **Transparencia OTC/ATS semanal** (FINRA `weeklySummary`): volumen ejecutado
   fuera de bolsa por valor y semana, con retardo de publicación de 2 semanas
   (NMS Tier 1) o 4 (resto).
3. **Volumen en corto diario Reg SHO** (ficheros `CNMSshvol`): la única serie
   diaria y gratuita de flujo off-exchange, con un día de latencia.

La regla que gobierna el módulo — y que sus tests verifican expresamente — es
la fila correspondiente de la tabla PIT del informe (§13): **nunca se indexa
por la fecha del hecho económico; siempre por la fecha de publicabilidad.**

`available_at` del short interest
---------------------------------
El snapshot se refiere a la *settlement date*, pero las firmas reportan hasta
las 18:00 ET del 2.º día hábil posterior y **FINRA compila y publica en torno
al 7.º-8.º día hábil posterior** (informe §8.1, trampa 1). Indexar por la fecha
de referencia adelanta ~8 días hábiles de información: suficiente para
atravesar entera la ventana [T-5, T-1] de un anuncio. Aquí
``available_at = finra_publication_date(settlement)`` (por defecto, la sesión
de bolsa +8 tras la referencia — aproximación conservadora), y el parámetro
`official_schedule` permite inyectar la tabla oficial *Short Interest Reporting
Deadlines* de FINRA cuando se disponga de ella, que es lo que el informe
recomienda como implementación definitiva (calcular festivos a mano es una
fuente de errores de un día, justo el error que el contrato §0.1 prohíbe).

Ciclo de liquidación variable
-----------------------------
La correspondencia *settlement date* → *trade date* ha cambiado dos veces
(informe §8.1, trampa 2): T+3 hasta la primera liquidación T+2 del 2017-09-07,
T+2 hasta la primera liquidación T+1 del 2024-05-29, T+1 después.
`settlement_cycle_lag` y `settlement_to_trade_date` implementan el mapeo; los
tests comprueban que coinciden con la implementación homóloga de
`events.flow`, que consume estos datos.

Advertencia sobre el short volume diario (informe §8.1, trampa 4): el fichero
Reg SHO es **flujo dominado por la intermediación** (FINRA Information Notice
10/05/2019); no es proxy de posicionamiento direccional. Su uso legítimo aquí
es el del §9.3 del informe de fuentes: `TotalVolume` = volumen off-exchange
del día, que dividido por el volumen consolidado da la cuota off-exchange con
un día de latencia (frente a las 2-4 semanas del dato ATS).

Referencias
-----------
- Boehmer, Jones y Zhang (2008), "Which Shorts Are Informed?", *JF* 63(2).
- Akbas, Boehmer, Ertürk y Sorescu (2017), *Financial Management* 46(2).
- Zhu (2014), "Do Dark Pools Harm Price Discovery?", *RFS* 27(3); y
  Comerton-Forde y Putniņš (2015), *JFE* 118(1): por qué el signo esperado de
  la cuota dark es negativo o nulo, no positivo.
- FINRA, *Short Interest Reporting*; *OTC Transparency User Guide*;
  *Information Notice 10/05/2019*.
"""

from __future__ import annotations

import base64
import datetime as dt
import logging
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Final, TypeAlias

import numpy as np
import pandas as pd

from earnings_alpha.config import Settings, get_settings
from earnings_alpha.data.base import (
    BaseProvider,
    DataKind,
    HttpClient,
    ProviderRegistry,
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
from earnings_alpha.pit import TradingCalendar, get_calendar
from earnings_alpha.types import Ticker, normalize_ticker

__all__ = [  # noqa: RUF022 - orden temático, no alfabético
    # esquema
    "SHORT_INTEREST_KIND",
    "OFF_EXCHANGE_KIND",
    "SHORT_VOLUME_KIND",
    "SHORT_INTEREST_COLUMNS",
    "OFF_EXCHANGE_COLUMNS",
    "SHORT_VOLUME_COLUMNS",
    # calendario FINRA
    "FIRST_T2_SETTLEMENT",
    "FIRST_T1_SETTLEMENT",
    "settlement_cycle_lag",
    "settlement_to_trade_date",
    "finra_settlement_dates",
    "finra_publication_date",
    "ats_publication_date",
    # guardias PIT
    "validate_short_interest",
    "validate_off_exchange",
    "published_asof",
    "off_exchange_share_daily",
    # proveedores
    "FlowProviderBase",
    "SyntheticFlowProvider",
    "FinraShortInterestProvider",
    "FinraOffExchangeProvider",
    "RegShoShortVolumeProvider",
    # registro y fachadas
    "DEFAULT_FLOW_PRIORITIES",
    "register_flow_providers",
    "get_short_interest",
    "get_off_exchange",
]

logger = logging.getLogger(__name__)

SHORT_INTEREST_KIND = DataKind.SHORT_INTEREST.value
OFF_EXCHANGE_KIND = DataKind.OFF_EXCHANGE.value
SHORT_VOLUME_KIND = "short_volume"
"""Kind propio del fichero diario Reg SHO. No está en `DataKind` a propósito:
es una variable de flujo distinta del stock de short interest y confundirlas
es la trampa 4 del informe §8.1."""

SHORT_INTEREST_COLUMNS: tuple[str, ...] = (
    "settlement_date",
    "ticker",
    "shares_short",
    "shares_outstanding",
    "short_percent_shares",
    "avg_daily_volume",
    "days_to_cover",
    "available_at",
    "source",
)
"""Esquema canónico del short interest quincenal. `shares_outstanding` puede
venir NaN (FINRA no lo publica; el denominador PIT debe salir de otra fuente,
informe §8.1 trampa 3)."""

OFF_EXCHANGE_COLUMNS: tuple[str, ...] = (
    "week_start",
    "week_end",
    "ticker",
    "ats_volume",
    "non_ats_volume",
    "off_exchange_volume",
    "available_at",
    "source",
)
"""Esquema canónico del volumen off-exchange semanal. La cuota
(`off_exchange_share`) NO forma parte del esquema del proveedor: exige un
volumen consolidado externo y con la convención de conteo correcta
(`off_exchange_share_daily` y la trampa de doble conteo del informe §8.2)."""

SHORT_VOLUME_COLUMNS: tuple[str, ...] = (
    "date",
    "ticker",
    "short_volume",
    "short_exempt_volume",
    "total_volume",
    "market",
    "available_at",
    "source",
)

DateLike: TypeAlias = str | dt.date | dt.datetime | pd.Timestamp

# Primeras liquidaciones de cada ciclo (informe §8.1; SEC T+2 desde operaciones
# del 2017-09-05, T+1 desde operaciones del 2024-05-28). Idénticas a las de
# `events.flow`; el test de paridad entre ambos módulos lo verifica.
FIRST_T2_SETTLEMENT: Final[dt.date] = dt.date(2017, 9, 7)
FIRST_T1_SETTLEMENT: Final[dt.date] = dt.date(2024, 5, 29)

DEFAULT_PUBLICATION_LAG_SESSIONS: Final[int] = 8
"""Sesiones entre la settlement date y la difusión pública del short interest.

FINRA documenta la publicación "en torno al 7.º-8.º día hábil" posterior
(`docs/research/data_sources.md` §8.1, marcado [verificar] porque FINRA ha
ajustado este calendario en el pasado). Se toma 8 —el extremo conservador—
porque equivocarse hacia tarde pierde un día de señal y equivocarse hacia
pronto fabrica look-ahead."""


# ===========================================================================
# 1. Calendario FINRA
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


def settlement_cycle_lag(settlement_date: DateLike) -> int:
    """Ciclo de liquidación vigente en la fecha: 3 (T+3), 2 (T+2) o 1 (T+1).

    Sigue `docs/research/informed_trading.md` §8.1 (trampa 2): T+3 hasta la
    primera liquidación T+2 (2017-09-07, operaciones desde el 2017-09-05) y
    T+1 desde el 2024-05-29 (operaciones desde el 2024-05-28). En los días de
    transición conviven dos ciclos en liquidación; se aplica el ciclo nuevo
    desde su primera liquidación, igual que `events.flow.settlement_cycle_lag`
    (el test de paridad entre ambos módulos protege la coherencia).
    """
    d = _as_date(settlement_date, "settlement_date")
    if d < FIRST_T2_SETTLEMENT:
        return 3
    if d < FIRST_T1_SETTLEMENT:
        return 2
    return 1


def settlement_to_trade_date(
    settlement_date: DateLike, cal: TradingCalendar | None = None
) -> dt.date:
    """Última sesión cuyas operaciones liquidan en `settlement_date`.

    Las posiciones del snapshot reflejan lo negociado hasta esta fecha, no
    hasta la fecha de liquidación: es la fecha que debe usarse para alinear el
    snapshot con series de precios o de eventos.
    """
    calendar = cal or get_calendar()
    ref = calendar.session_on_or_before(_as_date(settlement_date, "settlement_date"))
    return calendar.shift(ref, -settlement_cycle_lag(ref))


def finra_settlement_dates(
    start: DateLike, end: DateLike, cal: TradingCalendar | None = None
) -> list[dt.date]:
    """Fechas de referencia del ciclo quincenal de FINRA en ``[start, end]``.

    FINRA exige reportar posiciones a la liquidación del **día 15** (o el día
    hábil anterior si el 15 no lo es) y del **último día hábil** del mes
    (`docs/research/data_sources.md` §8.1). Se usa el calendario bursátil como
    aproximación de los días hábiles de FINRA (coinciden salvo rarezas
    documentadas; la tabla oficial de deadlines manda cuando esté cargada).
    """
    s, e = _check_window(start, end)
    calendar = cal or get_calendar()
    out: list[dt.date] = []
    cursor = dt.date(s.year, s.month, 1)
    while cursor <= e:
        mid = calendar.session_on_or_before(dt.date(cursor.year, cursor.month, 15))
        last_dom = (
            pd.Timestamp(cursor) + pd.offsets.MonthEnd(0)
        ).date()
        eom = calendar.session_on_or_before(last_dom)
        for day in (mid, eom):
            if s <= day <= e and day not in out:
                out.append(day)
        cursor = (pd.Timestamp(cursor) + pd.offsets.MonthBegin(1)).date()
    return sorted(out)


def finra_publication_date(
    settlement_date: DateLike,
    cal: TradingCalendar | None = None,
    *,
    lag_sessions: int = DEFAULT_PUBLICATION_LAG_SESSIONS,
    official_schedule: Mapping[dt.date, dt.date] | None = None,
) -> dt.date:
    """Fecha de difusión pública del snapshot de short interest.

    **Este es el `available_at` del short interest, sin excepciones** (informe
    §8.1, trampa 1). Por defecto: la sesión `lag_sessions` (=8) posterior a la
    settlement date. Si se pasa `official_schedule` (la tabla *Short Interest
    Reporting Deadlines* de FINRA, `settlement -> dissemination`), esa tabla
    manda y el cálculo aproximado solo cubre las fechas que falten en ella.
    """
    calendar = cal or get_calendar()
    ref = _as_date(settlement_date, "settlement_date")
    if official_schedule is not None and ref in official_schedule:
        published = _as_date(official_schedule[ref], "dissemination_date")
        if published <= ref:
            msg = (
                f"la tabla oficial dice que {ref} se publicó el {published}, que no es "
                "posterior a la referencia; la tabla está corrupta"
            )
            raise DataQualityError(msg)
        return published
    anchor = calendar.session_on_or_before(ref)
    return calendar.shift(anchor, int(lag_sessions))


def ats_publication_date(week_end: DateLike, *, tier: str = "T1") -> dt.date:
    """Fecha de publicación del dato semanal OTC/ATS de FINRA.

    NMS **Tier 1** (todo el S&P 500): la semana `w` se publica **2 semanas**
    después de su fin; el resto de NMS y OTC, 4 semanas (informe §8.2 y
    `data_sources.md` §9.2). Consecuencia que el docstring de la feature debe
    repetir: para un evento en `T`, la última semana conocible cubre
    negociación de hace ~3-5 semanas; `off_exchange_share_delta` es una
    variable de régimen con ventana efectiva ≈ [T-35, T-15], no una feature de
    [T-5, T-1].
    """
    tier_norm = str(tier).strip().upper()
    if tier_norm not in {"T1", "T2", "OTC"}:
        msg = f"tier no reconocido: {tier!r} (se espera T1, T2 u OTC)"
        raise ConfigError(msg)
    lag_days = 14 if tier_norm == "T1" else 28
    return _as_date(week_end, "week_end") + dt.timedelta(days=lag_days)


# ===========================================================================
# 2. Guardias PIT y utilidades
# ===========================================================================


def validate_short_interest(frame: pd.DataFrame) -> pd.DataFrame:
    """Valida el panel de short interest; la violación PIT es `LookAheadError`.

    Comprueba el esquema, que `shares_short >= 0` y —lo crítico— que
    ``available_at > settlement_date`` en **todas** las filas: un snapshot
    "publicado" en su fecha de referencia es exactamente el look-ahead de ~8
    días hábiles contra el que este módulo existe.
    """
    missing = [c for c in SHORT_INTEREST_COLUMNS if c not in frame.columns]
    if missing:
        msg = f"faltan columnas del esquema de short interest: {missing}"
        raise DataQualityError(msg)
    if len(frame) == 0:
        msg = "el panel de short interest está vacío"
        raise InsufficientHistory(msg)
    settlement = pd.to_datetime(frame["settlement_date"])
    available = pd.to_datetime(frame["available_at"])
    bad = available <= settlement
    if bad.any():
        rows = frame.loc[bad, ["ticker", "settlement_date", "available_at"]].head(3)
        msg = (
            f"{int(bad.sum())} filas tienen available_at <= settlement_date: usar el "
            "short interest antes de su publicación es look-ahead puro "
            f"(informe informed_trading.md §8.1). Ejemplos:\n{rows.to_string(index=False)}"
        )
        raise LookAheadError(msg)
    if (pd.to_numeric(frame["shares_short"], errors="coerce") < 0).any():
        msg = "hay shares_short negativos; el panel está corrupto"
        raise DataQualityError(msg)
    return frame


def validate_off_exchange(frame: pd.DataFrame) -> pd.DataFrame:
    """Valida el panel semanal ATS; publicación dentro de la semana es error PIT."""
    missing = [c for c in OFF_EXCHANGE_COLUMNS if c not in frame.columns]
    if missing:
        msg = f"faltan columnas del esquema off-exchange: {missing}"
        raise DataQualityError(msg)
    if len(frame) == 0:
        msg = "el panel off-exchange está vacío"
        raise InsufficientHistory(msg)
    week_end = pd.to_datetime(frame["week_end"])
    available = pd.to_datetime(frame["available_at"])
    bad = available <= week_end
    if bad.any():
        msg = (
            f"{int(bad.sum())} filas tienen available_at <= week_end; FINRA publica el "
            "dato ATS con 2-4 semanas de retraso y usarlo antes es look-ahead "
            "(data_sources.md §9.2)"
        )
        raise LookAheadError(msg)
    return frame


def published_asof(
    frame: pd.DataFrame,
    asof: DateLike,
    *,
    available_col: str = "available_at",
) -> pd.DataFrame:
    """Filas **públicamente conocibles** estrictamente antes de la sesión `asof`.

    Es el filtro que toda feature debe aplicar antes de mirar estos paneles.
    Estricto (`<`, no `<=`) por la misma razón que `events.flow`: la hora de
    publicación dentro del día no está garantizada, y asumir "disponible desde
    la apertura de su propio día" es la mitad de un día de look-ahead.
    """
    if available_col not in frame.columns:
        msg = f"el panel no tiene columna {available_col!r}"
        raise DataQualityError(msg)
    cutoff = pd.Timestamp(_as_date(asof, "asof"))
    return frame[pd.to_datetime(frame[available_col]) < cutoff]


def off_exchange_share_daily(
    short_volume: pd.DataFrame,
    consolidated_volume: pd.Series,
) -> pd.DataFrame:
    """Cuota off-exchange **diaria** = `TotalVolume` Reg SHO / volumen consolidado.

    Es el proxy del §9.3 de `data_sources.md`: la fracción de negociación que
    ocurre fuera de bolsa, con **un día** de latencia en vez de las 2-4 semanas
    del dato ATS. `consolidated_volume` debe ser una Series con MultiIndex
    ``(date, ticker)`` (panel canónico de precios) y contar cada operación
    **una sola vez**, igual que FINRA.

    **Trampa de doble conteo (informe §8.2):** si las convenciones difieren,
    la cuota sale sistemáticamente inflada o por encima del 100 %. Cuotas
    > 1 en más del 1 % de las filas lanzan `DataQualityError` con ese
    diagnóstico en vez de dejar pasar un panel corrupto.
    """
    missing = [c for c in ("date", "ticker", "total_volume") if c not in short_volume.columns]
    if missing:
        msg = f"faltan columnas en el panel de short volume: {missing}"
        raise DataQualityError(msg)
    if not isinstance(consolidated_volume.index, pd.MultiIndex):
        msg = "consolidated_volume debe tener MultiIndex (date, ticker)"
        raise DataQualityError(msg)
    sv = short_volume.copy()
    sv["date"] = pd.to_datetime(sv["date"]).dt.normalize()
    key = pd.MultiIndex.from_arrays([sv["date"], sv["ticker"]], names=["date", "ticker"])
    consolidated = consolidated_volume.reindex(key).to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = sv["total_volume"].to_numpy(dtype=float) / consolidated
    out = sv[["date", "ticker", "total_volume", "available_at"]].copy()
    out = out.rename(columns={"total_volume": "off_exchange_volume"})
    out["consolidated_volume"] = consolidated
    out["off_exchange_share"] = share
    finite = np.isfinite(share)
    if finite.any():
        frac_over = float((share[finite] > 1.0).mean())
        if frac_over > 0.01:
            msg = (
                f"el {100 * frac_over:.1f}% de las cuotas off-exchange supera el 100%: "
                "casi con seguridad el volumen consolidado y el de FINRA usan "
                "convenciones de conteo distintas (doble conteo, informe §8.2). "
                "Verifica la convención del proveedor de precios antes de dividir."
            )
            raise DataQualityError(msg)
    return out.reset_index(drop=True)


# ===========================================================================
# 3. Base de proveedores
# ===========================================================================


class FlowProviderBase(BaseProvider):
    """Base de los adaptadores de flujo (short interest y off-exchange)."""

    name = "flow_base"
    kinds: tuple[str, ...] = (SHORT_INTEREST_KIND, OFF_EXCHANGE_KIND)

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        http: HttpClient | None = None,
        cache: DiskCache | None = None,
        calendar: TradingCalendar | None = None,
    ) -> None:
        super().__init__(settings=settings, http=http)
        self.cache = cache
        self._calendar = calendar

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
                "faltan credenciales para pedir datos de flujo",
                missing_env=missing,
            )

    # -- API pública ----------------------------------------------------------

    def short_interest(
        self,
        start: DateLike,
        end: DateLike,
        tickers: Ticker | Sequence[Ticker] | None = None,
    ) -> pd.DataFrame:
        """Panel quincenal canónico, validado por `validate_short_interest`."""
        s, e = _check_window(start, end)
        wanted = _normalize_tickers(tickers)
        self._require_credentials()
        frame = self._fetch_short_interest(s, e, wanted)
        if frame is None or len(frame) == 0:
            msg = f"{self.name}: sin short interest en [{s}, {e}] para la selección pedida"
            raise InsufficientHistory(msg)
        if wanted is not None:
            frame = frame[frame["ticker"].isin(wanted)]
            if not len(frame):
                msg = f"{self.name}: ninguno de los símbolos pedidos tiene short interest"
                raise InsufficientHistory(msg)
        frame = frame.sort_values(["settlement_date", "ticker"]).reset_index(drop=True)
        return validate_short_interest(frame)

    def off_exchange(
        self,
        start: DateLike,
        end: DateLike,
        tickers: Ticker | Sequence[Ticker] | None = None,
    ) -> pd.DataFrame:
        """Panel semanal canónico, validado por `validate_off_exchange`."""
        s, e = _check_window(start, end)
        wanted = _normalize_tickers(tickers)
        self._require_credentials()
        frame = self._fetch_off_exchange(s, e, wanted)
        if frame is None or len(frame) == 0:
            msg = f"{self.name}: sin datos off-exchange en [{s}, {e}]"
            raise InsufficientHistory(msg)
        if wanted is not None:
            frame = frame[frame["ticker"].isin(wanted)]
            if not len(frame):
                msg = f"{self.name}: ninguno de los símbolos pedidos tiene datos off-exchange"
                raise InsufficientHistory(msg)
        frame = frame.sort_values(["week_end", "ticker"]).reset_index(drop=True)
        return validate_off_exchange(frame)

    # -- ganchos --------------------------------------------------------------

    def _fetch_short_interest(
        self, start: dt.date, end: dt.date, tickers: list[Ticker] | None
    ) -> pd.DataFrame:
        raise NotImplementedError

    def _fetch_off_exchange(
        self, start: dt.date, end: dt.date, tickers: list[Ticker] | None
    ) -> pd.DataFrame:
        raise NotImplementedError


# ===========================================================================
# 4. Sintético
# ===========================================================================


class SyntheticFlowProvider(FlowProviderBase):
    """Flujo del generador sintético. Sin red ni credenciales.

    `SyntheticMarket` ya genera el short interest con el calendario quincenal
    y el retardo de publicación de FINRA, y el panel semanal ATS con su
    retardo Tier 1; este adaptador solo recorta y garantiza el esquema. La
    columna extra `short_percent_shares` del generador se conserva (es la
    verdad-terreno de los tests de `dSI`).
    """

    name = "synthetic"
    kinds: tuple[str, ...] = (SHORT_INTEREST_KIND, OFF_EXCHANGE_KIND)

    def __init__(
        self,
        market: SyntheticMarket | None = None,
        *,
        settings: Settings | None = None,
        calendar: TradingCalendar | None = None,
        **market_kwargs: Any,
    ) -> None:
        super().__init__(settings=settings, calendar=calendar)
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

    def _fetch_short_interest(
        self, start: dt.date, end: dt.date, tickers: list[Ticker] | None
    ) -> pd.DataFrame:
        frame = self.market.short_interest(tickers=tickers)
        frame = frame[
            (frame["settlement_date"] >= pd.Timestamp(start))
            & (frame["settlement_date"] <= pd.Timestamp(end))
        ].copy()
        frame["source"] = "synthetic"
        return frame[[*SHORT_INTEREST_COLUMNS, "short_interest_delta"]]

    def _fetch_off_exchange(
        self, start: dt.date, end: dt.date, tickers: list[Ticker] | None
    ) -> pd.DataFrame:
        frame = self.market.off_exchange(tickers=tickers)
        frame = frame[
            (frame["week_end"] >= pd.Timestamp(start))
            & (frame["week_end"] <= pd.Timestamp(end))
        ].copy()
        frame["source"] = "synthetic"
        keep = [*OFF_EXCHANGE_COLUMNS, "total_volume", "off_exchange_share",
                "off_exchange_share_delta"]
        return frame[keep]


# ===========================================================================
# 5. FINRA Query API (short interest y ATS semanal)
# ===========================================================================


class _FinraQueryMixin:
    """OAuth2 *client credentials* y paginación de la FINRA Query API.

    Registro **gratuito** en el FINRA Developer Center; credenciales por
    variables de entorno ``FINRA_API_CLIENT_ID`` / ``FINRA_API_CLIENT_SECRET``
    (no están en `config.PROVIDER_ENV_KEYS`, así que el proveedor las declara
    él mismo vía `required_env`). El token se pide con Basic auth al endpoint
    EWS y caduca; se renueva con margen usando el reloj inyectable.
    Endpoints y nombres de campo según `docs/research/data_sources.md` §8.2 y
    §9.4, marcados allí [verificar] contra la API viva.
    """

    TOKEN_URL = (
        "https://ews.fip.finra.org/fip/rest/ews/oauth2/access_token"
        "?grant_type=client_credentials"
    )
    DATA_BASE = "https://api.finra.org/data/group/otcMarket/name"
    PAGE_LIMIT = 5000
    ENV_KEYS = ("FINRA_API_CLIENT_ID", "FINRA_API_CLIENT_SECRET")

    _token: str | None = None
    _token_expires: float = 0.0

    def required_env(self) -> list[str]:
        return list(self.ENV_KEYS)

    def _bearer(self) -> str:
        # self es un BaseProvider en las clases que mezclan esto.
        clock = self.clock  # type: ignore[attr-defined]
        if self._token is not None and clock.time() < self._token_expires:
            return self._token
        client_id = self._credential("FINRA_API_CLIENT_ID")  # type: ignore[attr-defined]
        secret = self._credential("FINRA_API_CLIENT_SECRET")  # type: ignore[attr-defined]
        basic = base64.b64encode(f"{client_id}:{secret}".encode()).decode("ascii")
        payload = self.http.post(  # type: ignore[attr-defined]
            self.TOKEN_URL,
            headers={"Authorization": f"Basic {basic}", "Accept": "application/json"},
        ).json()
        token = payload.get("access_token") if isinstance(payload, dict) else None
        if not token:
            msg = "finra: el endpoint de token no devolvió `access_token`"
            raise DataQualityError(msg)
        expires_in = float(payload.get("expires_in", 1800.0))
        self._token = str(token)
        # Margen del 10 % para no operar con un token a punto de caducar.
        self._token_expires = clock.time() + 0.9 * expires_in
        return self._token

    def _query(
        self,
        dataset: str,
        compare_filters: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """POST paginado a un dataset de la Query API; devuelve todas las filas."""
        import json as _json

        rows: list[dict[str, Any]] = []
        offset = 0
        while True:
            body: dict[str, Any] = {"limit": self.PAGE_LIMIT, "offset": offset}
            if compare_filters:
                body["compareFilters"] = compare_filters
            response = self.http.post(  # type: ignore[attr-defined]
                f"{self.DATA_BASE}/{dataset}",
                headers={
                    "Authorization": f"Bearer {self._bearer()}",
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
                body=_json.dumps(body).encode("utf-8"),
            )
            payload = response.json()
            if not isinstance(payload, list):
                msg = f"finra: se esperaba una lista JSON de {dataset}, llegó "
                msg += type(payload).__name__
                raise DataQualityError(msg)
            rows.extend(payload)
            if len(payload) < self.PAGE_LIMIT:
                break
            offset += self.PAGE_LIMIT
        return rows


class FinraShortInterestProvider(_FinraQueryMixin, FlowProviderBase):
    """Short interest consolidado quincenal de la FINRA Query API.

    Dataset ``consolidatedShortInterest`` (`data_sources.md` §8.2); campos
    relevantes: ``issueSymbolIdentifier``, ``settlementDate``,
    ``currentShortPositionQuantity``, ``averageDailyVolumeQuantity``,
    ``daysToCoverQuantity``. FINRA **no** publica `shares_outstanding`: queda
    NaN y el consumidor debe traer un denominador point-in-time propio
    (informe §8.1, trampa 3). Solo hay un año rodante en línea: hay que
    archivar cada publicación (política *append-only* del repo).

    ``available_at = finra_publication_date(settlement)`` — la sesión +8. La
    tabla oficial de deadlines puede inyectarse vía `official_schedule`.
    """

    name = "finra"
    kinds: tuple[str, ...] = (SHORT_INTEREST_KIND,)

    def __init__(
        self,
        *,
        official_schedule: Mapping[dt.date, dt.date] | None = None,
        publication_lag_sessions: int = DEFAULT_PUBLICATION_LAG_SESSIONS,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.official_schedule = dict(official_schedule or {})
        self.publication_lag_sessions = int(publication_lag_sessions)

    def _fetch_short_interest(
        self, start: dt.date, end: dt.date, tickers: list[Ticker] | None
    ) -> pd.DataFrame:
        filters = [
            {"compareType": "GTE", "fieldName": "settlementDate",
             "fieldValue": start.isoformat()},
            {"compareType": "LTE", "fieldName": "settlementDate",
             "fieldValue": end.isoformat()},
        ]
        raw = self._query("consolidatedShortInterest", filters)
        rows: list[dict[str, Any]] = []
        for item in raw:
            symbol = item.get("issueSymbolIdentifier")
            settlement = item.get("settlementDate")
            if not symbol or not settlement:
                continue
            settle_day = _as_date(settlement, "settlementDate")
            rows.append(
                {
                    "settlement_date": pd.Timestamp(settle_day),
                    "ticker": normalize_ticker(str(symbol)),
                    "shares_short": item.get("currentShortPositionQuantity"),
                    "shares_outstanding": np.nan,
                    "short_percent_shares": np.nan,
                    "avg_daily_volume": item.get("averageDailyVolumeQuantity"),
                    "days_to_cover": item.get("daysToCoverQuantity"),
                    "available_at": pd.Timestamp(
                        finra_publication_date(
                            settle_day,
                            self.calendar,
                            lag_sessions=self.publication_lag_sessions,
                            official_schedule=self.official_schedule or None,
                        )
                    ),
                    "source": "finra",
                }
            )
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(rows)
        frame["shares_short"] = pd.to_numeric(frame["shares_short"], errors="coerce")
        frame["avg_daily_volume"] = pd.to_numeric(frame["avg_daily_volume"], errors="coerce")
        frame["days_to_cover"] = pd.to_numeric(frame["days_to_cover"], errors="coerce")
        return frame

    def _fetch_off_exchange(
        self, start: dt.date, end: dt.date, tickers: list[Ticker] | None
    ) -> pd.DataFrame:
        raise ProviderUnavailable(
            self.name,
            "este adaptador sirve short interest; el semanal ATS lo sirve "
            "FinraOffExchangeProvider",
        )


class FinraOffExchangeProvider(_FinraQueryMixin, FlowProviderBase):
    """Volumen semanal OTC/ATS por valor (FINRA OTC Transparency).

    Dataset ``weeklySummary`` (`data_sources.md` §9.4): filas por
    ``(símbolo, semana, summaryTypeCode)`` con
    ``totalWeeklyShareQuantity``; ``ATS_W_SMBL`` es el agregado de dark pools
    y ``OTC_W_SMBL`` el de internalizadores (non-ATS). FINRA cuenta cada
    operación **una sola vez** — el denominador consolidado debe contarse
    igual (trampa de doble conteo, informe §8.2).

    ``available_at = ats_publication_date(week_end, tier)``: +2 semanas para
    Tier 1, +4 para el resto. Para eventos, esto convierte la cuota dark en
    una variable de régimen (ventana efectiva ≈ [T-35, T-15]); el informe §8.2
    exige decirlo en el docstring de la feature y aquí queda dicho también.
    """

    name = "finra_ats"
    kinds: tuple[str, ...] = (OFF_EXCHANGE_KIND,)

    _ATS_CODES: ClassVar[frozenset[str]] = frozenset({"ATS_W_SMBL"})
    _NON_ATS_CODES: ClassVar[frozenset[str]] = frozenset({"OTC_W_SMBL"})

    def _fetch_off_exchange(
        self, start: dt.date, end: dt.date, tickers: list[Ticker] | None
    ) -> pd.DataFrame:
        filters = [
            {"compareType": "GTE", "fieldName": "weekStartDate",
             "fieldValue": start.isoformat()},
            {"compareType": "LTE", "fieldName": "weekStartDate",
             "fieldValue": end.isoformat()},
        ]
        raw = self._query("weeklySummary", filters)
        rows: list[dict[str, Any]] = []
        for item in raw:
            symbol = item.get("issueSymbolIdentifier")
            week_start = item.get("weekStartDate")
            code = str(item.get("summaryTypeCode", ""))
            if not symbol or not week_start:
                continue
            if code not in self._ATS_CODES | self._NON_ATS_CODES:
                continue
            rows.append(
                {
                    "ticker": normalize_ticker(str(symbol)),
                    "week_start": pd.Timestamp(_as_date(week_start, "weekStartDate")),
                    "is_ats": code in self._ATS_CODES,
                    "shares": pd.to_numeric(
                        pd.Series([item.get("totalWeeklyShareQuantity")]), errors="coerce"
                    ).iloc[0],
                    "tier": str(item.get("tierIdentifier", "T1") or "T1"),
                }
            )
        if not rows:
            return pd.DataFrame()
        table = pd.DataFrame(rows)
        grouped = (
            table.pivot_table(
                index=["ticker", "week_start", "tier"],
                columns="is_ats",
                values="shares",
                aggfunc="sum",
            )
            .rename(columns={True: "ats_volume", False: "non_ats_volume"})
            .reset_index()
        )
        for col in ("ats_volume", "non_ats_volume"):
            if col not in grouped.columns:
                grouped[col] = 0.0
        grouped[["ats_volume", "non_ats_volume"]] = grouped[
            ["ats_volume", "non_ats_volume"]
        ].fillna(0.0)
        grouped["week_end"] = grouped["week_start"] + pd.Timedelta(days=4)
        grouped["off_exchange_volume"] = grouped["ats_volume"] + grouped["non_ats_volume"]
        grouped["available_at"] = [
            pd.Timestamp(ats_publication_date(we, tier=tier))
            for we, tier in zip(grouped["week_end"], grouped["tier"], strict=True)
        ]
        grouped["source"] = "finra_ats"
        return grouped[[*OFF_EXCHANGE_COLUMNS, "tier"]]

    def _fetch_short_interest(
        self, start: dt.date, end: dt.date, tickers: list[Ticker] | None
    ) -> pd.DataFrame:
        raise ProviderUnavailable(
            self.name,
            "este adaptador sirve el semanal ATS; el short interest lo sirve "
            "FinraShortInterestProvider",
        )


# ===========================================================================
# 6. Reg SHO: volumen en corto diario
# ===========================================================================


class RegShoShortVolumeProvider(BaseProvider):
    """Ficheros diarios Reg SHO de FINRA (`CNMSshvol{YYYYMMDD}.txt`). Sin clave.

    Formato pipa (`data_sources.md` §8.3)::

        Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market
        20200902|AAPL|35907771|123456|70119912|B,Q,N

    **Advertencia obligatoria (informe §8.1, trampa 4 y FINRA Information
    Notice 10/05/2019):** el volumen en corto diario es flujo dominado por la
    intermediación (creadores de mercado y mayoristas venden en corto como
    fontanería de inventario) y **no** mide posicionamiento bajista. Sus dos
    usos legítimos aquí: (a) `TotalVolume` = volumen off-exchange del día, el
    numerador del proxy diario de cuota dark (`off_exchange_share_daily`,
    §9.3); (b) diagnóstico. No entra en el registro bajo `short_interest`.

    ``available_at`` = la **siguiente sesión** al día del fichero: FINRA lo
    cuelga tras el cierre o en T+1 ([verificar]); asumir T+1 pierde como mucho
    una tarde y asumir T podría fabricar look-ahead intradía.
    """

    name = "regsho"
    kinds: tuple[str, ...] = (SHORT_VOLUME_KIND,)
    BASE = "https://cdn.finra.org/equity/regsho/daily"
    _HEADER = "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market"

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        http: HttpClient | None = None,
        cache: DiskCache | None = None,
        calendar: TradingCalendar | None = None,
    ) -> None:
        super().__init__(settings=settings, http=http)
        self.cache = cache
        self._calendar = calendar

    @property
    def calendar(self) -> TradingCalendar:
        if self._calendar is None:
            self._calendar = get_calendar()
        return self._calendar

    def _parse_file(self, text: str, day: dt.date) -> pd.DataFrame:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines or lines[0].replace(" ", "") != self._HEADER:
            head = lines[0][:80] if lines else "<vacío>"
            msg = f"regsho: cabecera inesperada en el fichero de {day}: {head!r}"
            raise DataQualityError(msg)
        rows: list[dict[str, Any]] = []
        malformed = 0
        for i, line in enumerate(lines[1:], start=2):
            parts = line.split("|")
            if len(parts) != 6:
                # Los ficheros llevan a veces una línea de cierre; cualquier otra
                # línea corta es corrupción y debe verse.
                if i == len(lines):
                    continue
                malformed += 1
                continue
            date_raw, symbol, short_v, exempt_v, total_v, market = parts
            try:
                file_day = dt.datetime.strptime(date_raw, "%Y%m%d").date()
                rows.append(
                    {
                        "date": pd.Timestamp(file_day),
                        "ticker": normalize_ticker(symbol),
                        "short_volume": float(short_v),
                        "short_exempt_volume": float(exempt_v),
                        "total_volume": float(total_v),
                        "market": market,
                    }
                )
            except ValueError:
                malformed += 1
        if malformed:
            msg = f"regsho: {malformed} líneas malformadas en el fichero de {day}"
            raise DataQualityError(msg)
        frame = pd.DataFrame(rows)
        if len(frame):
            available = pd.Timestamp(self.calendar.next_session(day))
            frame["available_at"] = available
            frame["source"] = "regsho"
        return frame

    def daily_short_volume(
        self,
        start: DateLike,
        end: DateLike,
        tickers: Ticker | Sequence[Ticker] | None = None,
    ) -> pd.DataFrame:
        """Panel diario canónico (`SHORT_VOLUME_COLUMNS`) para ``[start, end]``."""
        s, e = _check_window(start, end)
        wanted = _normalize_tickers(tickers)
        frames: list[pd.DataFrame] = []
        for day in self.calendar.sessions(s, e):
            stamp = day.strftime("%Y%m%d")
            response = self.http.get(
                f"{self.BASE}/CNMSshvol{stamp}.txt", allow_status=(404,)
            )
            if response.status_code == 404:
                # Fichero aún no publicado (o festivo raro): no es un error duro.
                logger.warning("regsho: sin fichero para %s", day.date())
                continue
            frame = self._parse_file(response.text, day.date())
            if len(frame):
                frames.append(frame)
        if not frames:
            msg = f"regsho: ningún fichero diario disponible en [{s}, {e}]"
            raise InsufficientHistory(msg)
        out = pd.concat(frames, ignore_index=True)
        if wanted is not None:
            out = out[out["ticker"].isin(wanted)]
            if not len(out):
                msg = "regsho: ninguno de los símbolos pedidos aparece en los ficheros"
                raise InsufficientHistory(msg)
        return (
            out[list(SHORT_VOLUME_COLUMNS)]
            .sort_values(["date", "ticker"])
            .reset_index(drop=True)
        )


# ===========================================================================
# 7. Registro y fachadas
# ===========================================================================


DEFAULT_FLOW_PRIORITIES: dict[str, int] = {
    "finra": 40,
    "finra_ats": 40,
    "synthetic": 0,
}


def register_flow_providers(
    registry: ProviderRegistry | None = None,
    *,
    settings: Settings | None = None,
    market: SyntheticMarket | None = None,
    include: Sequence[str] | None = None,
    priorities: Mapping[str, int] | None = None,
) -> ProviderRegistry:
    """Registra los proveedores de flujo.

    `finra` sirve `short_interest`; `finra_ats` sirve `off_exchange`;
    `synthetic` sirve ambos con prioridad 0. `regsho` **no** se registra en la
    cadena de fallback: su esquema (volumen diario) no es intercambiable con
    el quincenal/semanal y confundirlos es exactamente la trampa 4 del informe
    §8.1 — se registra bajo el kind propio `short_volume`.
    """
    reg = registry or get_registry()
    cfg = settings or get_settings()
    chosen = dict(DEFAULT_FLOW_PRIORITIES)
    if include is not None:
        unknown = sorted(set(include) - set(chosen) - {"regsho"})
        if unknown:
            msg = f"proveedores de flujo desconocidos en include: {unknown}"
            raise ConfigError(msg)
        chosen = {k: v for k, v in chosen.items() if k in set(include)}
    if priorities:
        chosen.update({k: int(v) for k, v in priorities.items() if k in chosen})

    synthetic = SyntheticFlowProvider(market, settings=cfg)
    if "finra" in chosen:
        reg.register(SHORT_INTEREST_KIND, FinraShortInterestProvider(settings=cfg),
                     chosen["finra"], replace=True)
    if "finra_ats" in chosen:
        reg.register(OFF_EXCHANGE_KIND, FinraOffExchangeProvider(settings=cfg),
                     chosen["finra_ats"], replace=True)
    if "synthetic" in chosen:
        reg.register(SHORT_INTEREST_KIND, synthetic, chosen["synthetic"], replace=True)
        reg.register(OFF_EXCHANGE_KIND, synthetic, chosen["synthetic"], replace=True)
    if include is None or "regsho" in include:
        reg.register(SHORT_VOLUME_KIND, RegShoShortVolumeProvider(settings=cfg), 10,
                     replace=True)
    return reg


def get_short_interest(
    start: DateLike,
    end: DateLike,
    tickers: Ticker | Sequence[Ticker] | None = None,
    *,
    registry: ProviderRegistry | None = None,
) -> pd.DataFrame:
    """Fachada: short interest del mejor proveedor disponible, con fallback."""
    reg = registry or get_registry()
    s, e = _check_window(start, end)
    return reg.call(
        SHORT_INTEREST_KIND,
        lambda p: p.short_interest(s, e, tickers),  # type: ignore[attr-defined]
        description=f"short_interest[{s.isoformat()}..{e.isoformat()}]",
    )


def get_off_exchange(
    start: DateLike,
    end: DateLike,
    tickers: Ticker | Sequence[Ticker] | None = None,
    *,
    registry: ProviderRegistry | None = None,
) -> pd.DataFrame:
    """Fachada: panel semanal off-exchange con fallback."""
    reg = registry or get_registry()
    s, e = _check_window(start, end)
    return reg.call(
        OFF_EXCHANGE_KIND,
        lambda p: p.off_exchange(s, e, tickers),  # type: ignore[attr-defined]
        description=f"off_exchange[{s.isoformat()}..{e.isoformat()}]",
    )
