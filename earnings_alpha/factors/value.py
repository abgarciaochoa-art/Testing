"""Factores de valoración (*value*) sobre fundamentales point-in-time.

Implementa la §11 de ``docs/research/fundamental_factors.md``:

- ``EarningsYield``   — E/P sobre beneficio TTM.
- ``FreeCashFlowYield`` — (CFO − CAPEX) TTM / capitalización.
- ``BookToMarket``    — patrimonio neto / capitalización.
- ``EbitdaToEV``      — EV/EBITDA **invertido** (EBITDA TTM / EV).
- ``SalesYield``      — ingresos TTM / capitalización (P/S invertido).

Reglas comunes (contrato §3.4 y ``fundamental_factors.md`` §1.3, §11.2):

1. **Siempre yield, nunca múltiplo.** ``E/P`` es continuo y ordenable en todo el
   dominio, incluido beneficio negativo; ``P/E`` es discontinuo en cero. Todos
   los factores de este módulo devuelven la forma *yield* y **mayor = más
   barato = más alcista**.
2. **Point-in-time.** El numerador entra por ``pit.asof_join`` con
   ``available_at``; el denominador (capitalización) es diario. La asimetría es
   deliberada: el yield se mueve a diario con el precio y a saltos con los
   filings.
3. **NaN estructural, no cero.** Donde el concepto no existe (FCF en bancos) el
   valor es NaN explícito y el nombre queda fuera del ranking.

Este módulo aloja además la **infraestructura compartida** por ``quality.py``,
``growth.py`` y ``accruals.py`` (historia trimestral con huecos detectados, TTM,
disponibilidad por nota de prensa o por filing, exclusión sectorial, unión
as-of a la rejilla diaria). Si ``factors/base.py`` consolida esta maquinaria,
este bloque puede migrar allí sin cambiar la API pública de los factores.

Coordinación con ``factors/base.py`` (escrito por otro agente en paralelo): las
clases de este módulo son implementaciones **estructurales** del protocolo
``Factor`` del contrato (``docs/ARCHITECTURE.md`` §3.4): atributos ``name`` y
``requires`` y método ``compute(ctx) -> pd.Series``. No heredan de nada de
``base.py`` y solo acceden a los atributos de ``FactorContext`` fijados por el
contrato (``dates``, ``universe``, ``prices``, ``fundamentals``, ``calendar``),
de modo que funcionan con cualquier implementación conforme del contexto.

Referencias
-----------
- Gray, W. R. y Vogel, J. (2012). *Analyzing Valuation Measures: A Performance
  Horse-Race over the Past 40 Years*. JPM 39(1), 112-121.
- Hou, K., Xue, C. y Zhang, L. (2020). *Replicating Anomalies*. RFS 33(5).
- Lakonishok, J., Shleifer, A. y Vishny, R. W. (1994). *Contrarian Investment,
  Extrapolation, and Risk*. JF 49(5), 1541-1578.
- Barbee, W. C., Mukherji, S. y Raines, G. A. (1996). *Do Sales-Price and
  Debt-Equity Explain Stock Returns Better than Book-Market and Firm Size?*.
  Financial Analysts Journal 52(2), 56-60.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np
import pandas as pd

from earnings_alpha.errors import DataQualityError, InsufficientHistory
from earnings_alpha.pit import asof_join, to_naive_utc

if TYPE_CHECKING:  # pragma: no cover - solo para tipado
    from earnings_alpha.pit import TradingCalendar

__all__ = [  # noqa: RUF022 - orden temático
    # factores
    "EarningsYield",
    "FreeCashFlowYield",
    "BookToMarket",
    "EbitdaToEV",
    "SalesYield",
    "FACTORS",
    # utilidades puras
    "enterprise_value",
    # infraestructura compartida por los módulos de factores contables
    "FundamentalFactor",
    "FactorContextLike",
    "Availability",
    "DEFAULT_EXCLUDE_SECTORS",
    "DEFAULT_MAX_STALENESS_DAYS",
    "QUARTER_GAP_DAYS",
    "quarterly_history",
    "shift_q",
    "ttm_sum",
    "grouped_rolling",
    "to_daily",
    "apply_sector_exclusion",
    "market_cap_panel",
    "finalize_factor",
    "positive_only",
]

# ---------------------------------------------------------------------------
# Contrato local (duck typing del FactorContext del contrato §3.4)
# ---------------------------------------------------------------------------


class FactorContextLike(Protocol):
    """Atributos de ``FactorContext`` (contrato §3.4) que consume este módulo.

    Se declara localmente porque ``factors/base.py`` se escribe en paralelo:
    programar contra el protocolo estructural permite que estos factores
    funcionen con la implementación real del contexto sin importarla.
    """

    dates: pd.DatetimeIndex
    universe: object
    prices: pd.DataFrame
    fundamentals: pd.DataFrame
    estimates: pd.DataFrame
    events: pd.DataFrame
    calendar: "TradingCalendar"


try:  # Coordinación con factors/base.py, escrito por otro agente en paralelo:
    # si el módulo ya existe se re-exporta su FactorContext real; si aún no,
    # se programa contra el protocolo estructural `FactorContextLike` de arriba.
    from earnings_alpha.factors.base import FactorContext  # noqa: F401
except ImportError:  # pragma: no cover - depende del orden de integración
    FactorContext = None  # type: ignore[assignment]

Availability = Literal["announcement", "filing"]
"""Política de fechado point-in-time del dato fundamental.

- ``"announcement"``: la nota de prensa (8-K item 2.02). Válida para partidas de
  la cuenta de resultados (ingresos, COGS, beneficio), que el comunicado trae.
- ``"filing"``: la aceptación del 10-Q/10-K en EDGAR. **Obligatoria** para
  flujos de caja y balance: la mayoría de comunicados no los incluyen completos
  y usar el 8-K sería look-ahead (``fundamental_factors.md`` §1.2).
"""

DEFAULT_EXCLUDE_SECTORS: frozenset[str] = frozenset({"Financials", "Real Estate"})
"""Sectores GICS donde los factores de capital circulante / margen / FCF son
estructuralmente indefinidos (§1.5 y §7.4 del informe): 107 de 503 nombres."""

QUARTER_GAP_DAYS: tuple[int, int] = (75, 115)
"""Separación admisible en días entre cierres trimestrales consecutivos. Fuera
de este rango la cadena se considera rota (trimestre ausente o cambio de
ejercicio fiscal) y ningún desplazamiento ni TTM cruza la rotura."""

DEFAULT_MAX_STALENESS_DAYS: int = 400
"""Antigüedad máxima de un dato trimestral arrastrado por el as-of join. Un
emisor que deja de reportar no debe seguir puntuando con su último filing
indefinidamente (véase ``pit.asof_join``)."""


# ---------------------------------------------------------------------------
# Infraestructura compartida (quality.py, growth.py y accruals.py la importan)
# ---------------------------------------------------------------------------


def positive_only(x: pd.Series) -> pd.Series:
    """Devuelve ``x`` con los valores no positivos convertidos en NaN.

    Es el guardarraíl estándar de los denominadores: un activo total, un
    patrimonio o una capitalización no positivos hacen el ratio ininterpretable
    y el contrato exige NaN explícito, nunca un cociente con signo cambiado.
    """
    return x.where(x > 0)


def _availability_series(df: pd.DataFrame, *, availability: Availability, factor: str) -> pd.Series:
    """Resuelve la columna de disponibilidad pública según la política pedida.

    Con ``availability="filing"`` se prefiere ``filed_at`` (aceptación del
    10-Q/10-K). Si no existe, se degrada **documentadamente** a ``available_at``
    bajo el supuesto de que el proveedor ya fecha sus hechos por el filing (es
    el caso de EDGAR); si el proveedor fechase por nota de prensa, esta
    degradación adelantaría la disponibilidad de partidas de balance y flujos, y
    por eso se emite un aviso.

    Con ``availability="announcement"`` se prefiere ``available_at`` y la
    degradación a ``filed_at`` es conservadora (solo retrasa la señal).
    """
    if availability == "filing":
        preferred, fallback = "filed_at", "available_at"
        note = (
            "las partidas de balance y flujos solo son públicas con el 10-Q/10-K; "
            "se usa `available_at` bajo el supuesto de que el proveedor ya fecha "
            "por el filing. Si `available_at` fuese la nota de prensa, esto "
            "ADELANTARÍA la disponibilidad real (fundamental_factors.md §1.2)."
        )
    elif availability == "announcement":
        preferred, fallback = "available_at", "filed_at"
        note = "se usa `filed_at`, que es posterior: la señal solo puede retrasarse (conservador)."
    else:  # pragma: no cover - protegido por el tipo
        msg = f"availability desconocida: {availability!r}"
        raise ValueError(msg)

    if preferred in df.columns:
        col = preferred
    elif fallback in df.columns:
        warnings.warn(
            f"{factor}: `fundamentals` no trae `{preferred}`; {note}",
            UserWarning,
            stacklevel=3,
        )
        col = fallback
    else:
        msg = (
            f"{factor}: `fundamentals` no trae `available_at` ni `filed_at`; "
            "sin fecha de disponibilidad pública no existe point-in-time"
        )
        raise DataQualityError(msg)
    return pd.Series(to_naive_utc(df[col]), index=df.index)


def quarterly_history(
    ctx: FactorContextLike,
    columns: Sequence[str],
    *,
    optional: Sequence[str] = (),
    availability: Availability = "filing",
    factor: str,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Prepara la historia trimestral de ``ctx.fundamentals`` para un factor.

    Devuelve ``(frame, faltantes_opcionales)`` donde ``frame`` está ordenado por
    ``(ticker, period_end)`` y trae:

    - las columnas requeridas (``columns``) y las opcionales presentes;
    - ``__avail__``: instante de disponibilidad pública según ``availability``;
    - ``__run__``: identificador de cadena de trimestres **consecutivos**
      (separación dentro de ``QUARTER_GAP_DAYS``). Los desplazamientos y los TTM
      de ``shift_q``/``ttm_sum`` jamás cruzan una rotura de cadena.

    Política de fallo explícito:

    - falta una columna **requerida** → ``DataQualityError`` (error de
      fontanería de datos, no de disponibilidad);
    - falta una columna **opcional** → se devuelve en ``faltantes_opcionales``
      y el factor decide su degradación documentada;
    - ``fundamentals`` vacío o sin ninguna fila fechable → ``InsufficientHistory``;
    - filas con ``is_restated == True`` se **descartan** (el backtest usa la
      cifra tal y como se reportó por primera vez, §1.3 del informe).
    """
    fund = getattr(ctx, "fundamentals", None)
    if not isinstance(fund, pd.DataFrame):
        msg = f"{factor}: el contexto no trae `fundamentals` como DataFrame"
        raise DataQualityError(msg)
    if len(fund) == 0:
        msg = f"{factor}: `fundamentals` está vacío; no hay historia con la que calcular"
        raise InsufficientHistory(msg)

    base_cols = ["ticker", "period_end"]
    missing = [c for c in [*base_cols, *columns] if c not in fund.columns]
    if missing:
        msg = (
            f"{factor}: faltan columnas requeridas en `fundamentals`: {missing}. "
            "El factor declara sus requisitos y falla explícitamente en vez de "
            "degradarse en silencio."
        )
        raise DataQualityError(msg)

    present_optional = [c for c in optional if c in fund.columns]
    missing_optional = tuple(c for c in optional if c not in fund.columns)

    keep = list(
        dict.fromkeys(
            [
                *base_cols,
                *columns,
                *present_optional,
                *(c for c in ("available_at", "filed_at", "fiscal_period", "form", "is_restated") if c in fund.columns),
            ]
        )
    )
    df = fund[keep].copy()

    if "is_restated" in df.columns:
        restated = df["is_restated"].fillna(False).astype(bool)
        if bool(restated.any()):
            warnings.warn(
                f"{factor}: se descartan {int(restated.sum())} filas reexpresadas "
                "(is_restated=True); el factor usa la cifra reportada originalmente",
                UserWarning,
                stacklevel=2,
            )
            df = df[~restated]
        df = df.drop(columns="is_restated")

    df["ticker"] = df["ticker"].astype(str)
    df["period_end"] = pd.DatetimeIndex(pd.to_datetime(df["period_end"])).normalize()
    df["__avail__"] = _availability_series(df, availability=availability, factor=factor)

    n_undated = int(df["__avail__"].isna().sum())
    if n_undated:
        warnings.warn(
            f"{factor}: {n_undated} filas de `fundamentals` sin fecha de disponibilidad; "
            "se descartan (un hecho sin fecha pública no es utilizable point-in-time)",
            UserWarning,
            stacklevel=2,
        )
        df = df[df["__avail__"].notna()]
    if len(df) == 0:
        msg = f"{factor}: ninguna fila de `fundamentals` tiene fecha de disponibilidad"
        raise InsufficientHistory(msg)

    # Primera versión reportada de cada trimestre: ante duplicados gana la de
    # disponibilidad más temprana (semántica de `pit.first_reported`).
    df = (
        df.sort_values(["ticker", "period_end", "__avail__"], kind="mergesort")
        .drop_duplicates(["ticker", "period_end"], keep="first")
        .sort_values(["ticker", "period_end"], kind="mergesort")
        .reset_index(drop=True)
    )

    gap = df.groupby("ticker", sort=False)["period_end"].diff().dt.days
    lo, hi = QUARTER_GAP_DAYS
    df["__run__"] = (gap.isna() | (gap < lo) | (gap > hi)).cumsum()
    return df, missing_optional


def shift_q(frame: pd.DataFrame, values: pd.Series | str, periods: int = 1) -> pd.Series:
    """Desplaza ``periods`` trimestres hacia atrás dentro de cada cadena consecutiva.

    Un desplazamiento que cruzaría una rotura de cadena (trimestre ausente,
    cambio de ejercicio) devuelve NaN: comparar `q` con un "hace 4 trimestres"
    que en realidad está a 6 no es una comparación interanual.
    """
    s = frame[values] if isinstance(values, str) else values
    return s.groupby([frame["ticker"], frame["__run__"]], sort=False).shift(periods)


def grouped_rolling(
    frame: pd.DataFrame,
    values: pd.Series | str,
    *,
    window: int,
    how: Literal["sum", "std", "mean"] = "sum",
    min_periods: int | None = None,
) -> pd.Series:
    """Estadístico móvil por cadena de trimestres consecutivos, alineado a ``frame``.

    ``min_periods`` por defecto es ``window``: nunca se calcula un TTM o una
    volatilidad sobre menos observaciones de las exigidas (§1.3 del informe).
    """
    s = frame[values] if isinstance(values, str) else values
    roller = (
        s.astype("float64")
        .groupby([frame["ticker"], frame["__run__"]], sort=False)
        .rolling(window=window, min_periods=min_periods or window)
    )
    if how == "sum":
        out = roller.sum()
    elif how == "std":
        out = roller.std(ddof=1)
    elif how == "mean":
        out = roller.mean()
    else:  # pragma: no cover - protegido por el tipo
        msg = f"estadístico móvil desconocido: {how!r}"
        raise ValueError(msg)
    return out.droplevel([0, 1]).reindex(frame.index)


def ttm_sum(frame: pd.DataFrame, values: pd.Series | str) -> pd.Series:
    """Suma de los últimos 4 trimestres consecutivos (*trailing twelve months*).

    NaN si la ventana no tiene exactamente 4 trimestres consecutivos válidos.
    """
    return grouped_rolling(frame, values, window=4, how="sum")


def apply_sector_exclusion(
    quarterly: pd.DataFrame,
    value_cols: Sequence[str],
    ctx: FactorContextLike,
    exclude: frozenset[str] | set[str],
    *,
    factor: str,
) -> pd.DataFrame:
    """Pone a NaN los valores de los tickers de sectores estructuralmente excluidos.

    NaN, **no cero**: un cero imputado es una posición central fabricada en el
    ranking; un NaN excluye el nombre (§1.3 del informe). Si el contexto no
    puede resolver sectores se avisa y no se excluye nada: la degradación queda
    documentada en el aviso.
    """
    if not exclude:
        return quarterly
    sector_for = getattr(getattr(ctx, "universe", None), "sector_for", None)
    if not callable(sector_for):
        warnings.warn(
            f"{factor}: el contexto no permite resolver sectores "
            "(`universe.sector_for` ausente); no se aplica la exclusión "
            f"estructural de {sorted(exclude)} y esos nombres entrarán con "
            "valores no interpretables",
            UserWarning,
            stacklevel=2,
        )
        return quarterly
    sectors: dict[str, str | None] = {}
    for t in quarterly["ticker"].unique():
        try:
            sectors[t] = sector_for(t)
        except Exception:  # noqa: BLE001 - un universo parcial no debe tumbar el factor
            sectors[t] = None
    mask = quarterly["ticker"].map(sectors).isin(exclude)
    if bool(mask.any()):
        quarterly = quarterly.copy()
        quarterly.loc[mask, list(value_cols)] = np.nan
    return quarterly


def _drop_superseded(panel: pd.DataFrame) -> pd.DataFrame:
    """Elimina filas que en su fecha de disponibilidad ya estaban superadas.

    Caso raro pero real: un trimestre antiguo presentado *después* de que el
    trimestre siguiente ya fuese público (filing tardío). ``merge_asof`` escoge
    el registro de mayor ``available_at``, que sería el periodo viejo; aquí se
    descarta para que siempre gane el periodo más reciente ya conocido.
    """
    p = panel.sort_values(["ticker", "available_at", "period_end"], kind="mergesort")
    newest = p.groupby("ticker", sort=False)["period_end"].cummax()
    return p[p["period_end"] >= newest]


def to_daily(
    ctx: FactorContextLike,
    quarterly: pd.DataFrame,
    value_cols: Sequence[str],
    *,
    factor: str,
    max_staleness_days: int | None = DEFAULT_MAX_STALENESS_DAYS,
) -> pd.DataFrame:
    """Proyecta valores trimestrales a la rejilla diaria vía ``pit.asof_join``.

    Devuelve un DataFrame con MultiIndex ``(date, ticker)`` sobre el producto de
    ``ctx.dates`` y los tickers con historia. Antes del primer dato disponible el
    valor es NaN (jamás se rellena hacia atrás), y un dato más viejo que
    ``max_staleness_days`` se invalida en vez de arrastrarse indefinidamente.
    """
    dates = getattr(ctx, "dates", None)
    if dates is None:
        msg = f"{factor}: el contexto no trae `dates` (rejilla de decisión)"
        raise DataQualityError(msg)
    idx = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    if len(idx) == 0:
        msg = f"{factor}: `dates` está vacío; no hay rejilla de decisión"
        raise DataQualityError(msg)

    panel = quarterly[["ticker", "period_end", "__avail__", *value_cols]].rename(
        columns={"__avail__": "available_at"}
    )
    panel = _drop_superseded(panel)
    return asof_join(
        idx,
        panel[["ticker", "available_at", *value_cols]],
        value_cols=list(value_cols),
        max_staleness_days=max_staleness_days,
        keep_available_at=False,
    )


def market_cap_panel(ctx: FactorContextLike, *, factor: str) -> pd.Series:
    """Capitalización bursátil diaria ``(date, ticker)`` desde ``ctx.prices``.

    Usa la columna ``market_cap`` si existe; si no, ``close * shares_outstanding``
    (equivalente cuando el número de acciones del panel de precios es el vigente
    en cada fecha). Sin ninguna de las dos, fallo explícito.
    """
    px = getattr(ctx, "prices", None)
    if not isinstance(px, pd.DataFrame) or len(px) == 0:
        msg = f"{factor}: necesita `prices` (capitalización diaria) y el contexto no lo trae"
        raise DataQualityError(msg)
    if not isinstance(px.index, pd.MultiIndex) or list(px.index.names) != ["date", "ticker"]:
        msg = f"{factor}: `prices` debe ser un panel canónico con MultiIndex (date, ticker)"
        raise DataQualityError(msg)
    if "market_cap" in px.columns:
        mc = px["market_cap"]
    elif {"close", "shares_outstanding"} <= set(px.columns):
        mc = px["close"] * px["shares_outstanding"]
    else:
        msg = (
            f"{factor}: `prices` no trae `market_cap` ni el par "
            "(`close`, `shares_outstanding`) con el que derivarla"
        )
        raise DataQualityError(msg)
    out = mc.astype("float64")
    out.name = "market_cap"
    return out


def finalize_factor(values: pd.Series, *, name: str) -> pd.Series:
    """Acondiciona la salida de un factor y aplica el fallo explícito agregado.

    - dtype float64, índice ``(date, ticker)`` ordenado, ``Series.name`` fijado;
    - ±inf → NaN (un denominador degenerado es NaN explícito, no un infinito);
    - si **ninguna** observación es finita lanza ``InsufficientHistory``: un
      panel enteramente NaN devuelto en silencio es exactamente lo que el
      contrato prohíbe.
    """
    s = pd.Series(values).astype("float64").replace([np.inf, -np.inf], np.nan)
    s.name = name
    s = s.sort_index()
    if not bool(s.notna().any()):
        msg = (
            f"{name}: ningún valor calculable en toda la rejilla. Causas típicas: "
            "historia trimestral insuficiente para el requisito del factor, "
            "todas las fechas anteriores a la primera disponibilidad pública, o "
            "universo compuesto íntegramente por sectores excluidos."
        )
        raise InsufficientHistory(msg)
    return s


class FundamentalFactor:
    """Base ligera de los factores contables de este paquete.

    Es una implementación **estructural** del protocolo ``Factor`` del contrato
    (§3.4): atributos ``name`` y ``requires`` y método ``compute``. No depende
    de ``factors/base.py`` (en desarrollo paralelo); cualquier registro o ABC
    que ese módulo introduzca puede envolver estas clases sin modificarlas.
    """

    name: str = "fundamental_factor"
    requires: list[str] = ["fundamentals"]  # noqa: RUF012 - contrato §3.4 pide list[str]
    availability: Availability = "filing"
    exclude_sectors: frozenset[str] = frozenset()

    def __init__(self, *, max_staleness_days: int | None = DEFAULT_MAX_STALENESS_DAYS) -> None:
        self.max_staleness_days = max_staleness_days

    def compute(self, ctx: FactorContextLike) -> pd.Series:  # pragma: no cover - abstracto
        """Serie con MultiIndex ``(date, ticker)``; valor mayor = más alcista."""
        raise NotImplementedError

    # ------------------------------------------------------------------ apoyo

    def _daily_values(
        self,
        ctx: FactorContextLike,
        quarterly: pd.DataFrame,
        value_cols: Sequence[str],
    ) -> pd.DataFrame:
        """Exclusión sectorial + proyección diaria, en el orden correcto."""
        q = apply_sector_exclusion(quarterly, value_cols, ctx, self.exclude_sectors, factor=self.name)
        return to_daily(
            ctx, q, value_cols, factor=self.name, max_staleness_days=self.max_staleness_days
        )


# ---------------------------------------------------------------------------
# Utilidades puras
# ---------------------------------------------------------------------------


def enterprise_value(
    market_cap: float,
    total_debt: float,
    cash: float,
    preferred: float = 0.0,
    minority: float = 0.0,
) -> float:
    """Valor de empresa: ``MC + deuda + preferentes + minoritarios − caja``.

    Fórmula de ``fundamental_factors.md`` §11.1. Es la base de ``EbitdaToEV``;
    se expone como función pura para que los tests la verifiquen con números de
    juguete.
    """
    return market_cap + total_debt + preferred + minority - cash


# ---------------------------------------------------------------------------
# Factores
# ---------------------------------------------------------------------------


class EarningsYield(FundamentalFactor):
    """Earnings yield: ``E/P = beneficio neto TTM / capitalización``.

    Referencias: la regla *yield, nunca múltiplo* y la política con pérdidas
    están en ``fundamental_factors.md`` §11.2; magnitudes conservadoras en Hou,
    Xue y Zhang (2020), RFS 33(5). La lógica contraria (value como explotación
    de la extrapolación) es Lakonishok, Shleifer y Vishny (1994), JF 49(5).

    - **Signo:** mayor yield = más barato = más alcista. Con pérdidas se aplica
      la política (a) del informe: el ``E/P`` negativo se conserva y ordena de
      forma natural (todas las pérdidas al fondo del ranking).
    - **PIT:** el beneficio es cuenta de resultados y viene en la nota de prensa
      → ``availability = "announcement"`` (``available_at`` del 8-K).
    - **Requisitos:** ``fundamentals[net_income]`` (4 trimestres consecutivos
      para el TTM) y ``prices`` con capitalización. Sin 4 trimestres → NaN para
      ese nombre; capitalización no positiva → NaN.
    - **Dónde falla:** financieras (mejor ``P/B``/``P/TBV``, véase
      ``BookToMarket``); empresas con beneficio cercano a cero (yield diminuto
      pero continuo). No se excluye ningún sector: E/P está definido en todos.
    """

    name = "earnings_yield"
    requires: list[str] = ["fundamentals", "prices"]  # noqa: RUF012
    availability: Availability = "announcement"

    def compute(self, ctx: FactorContextLike) -> pd.Series:
        q, _ = quarterly_history(
            ctx, ["net_income"], availability=self.availability, factor=self.name
        )
        q["value"] = ttm_sum(q, "net_income")
        daily = self._daily_values(ctx, q, ["value"])
        mc = market_cap_panel(ctx, factor=self.name).reindex(daily.index)
        return finalize_factor(daily["value"] / positive_only(mc), name=self.name)


class SalesYield(FundamentalFactor):
    """Sales yield: ``ingresos TTM / capitalización`` (P/S invertido).

    Referencia: Barbee, Mukherji y Raines (1996), FAJ 52(2), documentan que el
    ratio ventas/precio explica la sección cruzada al menos tan bien como
    book-to-market y tamaño en su muestra.

    - **Signo:** mayor = más barato = más alcista.
    - **PIT:** los ingresos vienen en la nota de prensa → ``"announcement"``.
    - **Requisitos:** ``fundamentals[revenue]`` (4 trimestres) y ``prices``.
    - **Dónde falla:** en Financieras "ingresos" no es un concepto homogéneo
      (intereses brutos vs netos según proveedor, §3 del informe); no se excluye
      estructuralmente pero la neutralización sectorial es imprescindible. Los
      márgenes heterogéneos hacen el ratio incomparable ENTRE sectores: sin
      neutralizar, es una apuesta corta en software y larga en distribución.
    """

    name = "sales_yield"
    requires: list[str] = ["fundamentals", "prices"]  # noqa: RUF012
    availability: Availability = "announcement"

    def compute(self, ctx: FactorContextLike) -> pd.Series:
        q, _ = quarterly_history(ctx, ["revenue"], availability=self.availability, factor=self.name)
        q["value"] = ttm_sum(q, "revenue")
        daily = self._daily_values(ctx, q, ["value"])
        mc = market_cap_panel(ctx, factor=self.name).reindex(daily.index)
        return finalize_factor(daily["value"] / positive_only(mc), name=self.name)


class FreeCashFlowYield(FundamentalFactor):
    """FCF yield: ``(CFO − CAPEX) TTM / capitalización``.

    Referencias: ``fundamental_factors.md`` §11; Hou, Xue y Zhang (2020)
    reportan para *cash flow-to-price* con cortes NYSE y ponderación por
    capitalización 0.49-0.77 % mensual en el decil extremo [verificar en el
    original, §11.3 del informe].

    - **Signo:** mayor = más alcista.
    - **PIT:** CFO y CAPEX solo están en el estado de flujos → ``"filing"``
      (``filed_at`` del 10-Q/10-K). Usar la nota de prensa aquí es el
      look-ahead más caro del informe (§1.2).
    - **Convención de signos del dato:** ``capex`` se espera **positivo**
      (inversión desembolsada), como lo entrega EDGAR/Compustat; un proveedor
      que lo dé negativo debe corregirse aguas arriba.
    - **Requisitos:** ``fundamentals[cfo, capex]`` (4 trimestres) y ``prices``.
    - **Dónde falla (NaN estructural):** Financieras — el CFO incorpora la
      variación de la cartera crediticia y de trading, y el capex es
      irrelevante — e Inmobiliarias — la amortización deprime NI y el FCF de un
      REIT exige FFO/AFFO —. Ambos sectores quedan en NaN (§11.4). Capex
      cíclico: correlación estructural con ``-asset_growth`` (§11.4); no
      sumarlos como independientes.
    """

    name = "fcf_yield"
    requires: list[str] = ["fundamentals", "prices"]  # noqa: RUF012
    availability: Availability = "filing"
    exclude_sectors: frozenset[str] = DEFAULT_EXCLUDE_SECTORS

    def compute(self, ctx: FactorContextLike) -> pd.Series:
        q, _ = quarterly_history(
            ctx, ["cfo", "capex"], availability=self.availability, factor=self.name
        )
        q["value"] = ttm_sum(q, "cfo") - ttm_sum(q, "capex")
        daily = self._daily_values(ctx, q, ["value"])
        mc = market_cap_panel(ctx, factor=self.name).reindex(daily.index)
        return finalize_factor(daily["value"] / positive_only(mc), name=self.name)


class BookToMarket(FundamentalFactor):
    """Book-to-market: ``patrimonio neto / capitalización``.

    Referencias: Fama y French (1992, 1993) definen HML sobre este ratio;
    Lakonishok, Shleifer y Vishny (1994) dan la interpretación conductual.
    Es además el ratio de valoración recomendado por el informe para
    Financieras (§11.4), donde FCF y EV no son interpretables; por eso este
    factor **no excluye ningún sector**.

    - **Signo:** mayor = más barato = más alcista.
    - **PIT:** el patrimonio es balance → ``"filing"``.
    - **Requisitos:** ``fundamentals[total_equity]`` (última foto trimestral,
      sin TTM: es un stock) y ``prices``.
    - **Patrimonio negativo → NaN**, siguiendo la convención de Fama-French:
      una empresa con fondos propios negativos no es "infinitamente barata",
      es un caso degenerado del ratio.
    """

    name = "book_to_market"
    requires: list[str] = ["fundamentals", "prices"]  # noqa: RUF012
    availability: Availability = "filing"

    def compute(self, ctx: FactorContextLike) -> pd.Series:
        q, _ = quarterly_history(
            ctx, ["total_equity"], availability=self.availability, factor=self.name
        )
        q["value"] = positive_only(q["total_equity"].astype("float64"))
        daily = self._daily_values(ctx, q, ["value"])
        mc = market_cap_panel(ctx, factor=self.name).reindex(daily.index)
        return finalize_factor(daily["value"] / positive_only(mc), name=self.name)


class EbitdaToEV(FundamentalFactor):
    """EV/EBITDA invertido: ``EBITDA TTM / EV``.

    Referencia: Gray y Vogel (2012), JPM 39(1), encuentran que ``EBIT/TEV`` es
    la más robusta de las métricas de valoración habituales en su carrera de
    caballos de 40 años [verificar en el original, §11.3 del informe]. Frente a
    ``E/P``, el EV neutraliza la estructura de capital: a igualdad de negocio,
    una empresa más apalancada no parece más barata (§11.3).

    - **Signo:** mayor = más barato = más alcista. Con EBITDA negativo el
      cociente es negativo y ordenable (misma lógica continua que ``E/P``).
    - **EV ≤ 0 → NaN**: con caja superior a capitalización más deuda el ratio
      se vuelve discontinuo y deja de ser ordenable.
    - **PIT:** deuda y caja son balance → ``"filing"``.
    - **Requisitos:** ``ebitda`` (o ``operating_income`` +
      ``depreciation_amortization``, equivalencia contable estándar), deuda
      total (o corto + largo plazo) y ``cash``. Opcionales
      ``preferred_equity`` y ``minority_interest``: si faltan se asumen 0 con
      aviso (degradación documentada; en el S&P 500 ambas partidas son
      minoritarias pero no siempre nulas).
    - **Dónde falla (NaN estructural):** Financieras — la deuda de un banco es
      materia prima, no financiación, y el EV no es interpretable (§11.4)—.
    """

    name = "ebitda_to_ev"
    requires: list[str] = ["fundamentals", "prices"]  # noqa: RUF012
    availability: Availability = "filing"
    exclude_sectors: frozenset[str] = frozenset({"Financials"})

    def compute(self, ctx: FactorContextLike) -> pd.Series:
        fund = getattr(ctx, "fundamentals", None)
        if not isinstance(fund, pd.DataFrame):
            msg = f"{self.name}: el contexto no trae `fundamentals` como DataFrame"
            raise DataQualityError(msg)
        cols = set(fund.columns)

        if "ebitda" in cols:
            ebitda_inputs = ["ebitda"]
        elif {"operating_income", "depreciation_amortization"} <= cols:
            ebitda_inputs = ["operating_income", "depreciation_amortization"]
        else:
            msg = (
                f"{self.name}: faltan `ebitda` o el par (`operating_income`, "
                "`depreciation_amortization`) en `fundamentals`"
            )
            raise DataQualityError(msg)

        if "total_debt" in cols:
            debt_inputs = ["total_debt"]
        elif {"short_term_debt", "long_term_debt"} <= cols:
            debt_inputs = ["short_term_debt", "long_term_debt"]
        else:
            msg = (
                f"{self.name}: faltan `total_debt` o el par (`short_term_debt`, "
                "`long_term_debt`) en `fundamentals`"
            )
            raise DataQualityError(msg)

        q, missing_opt = quarterly_history(
            ctx,
            [*ebitda_inputs, *debt_inputs, "cash"],
            optional=("preferred_equity", "minority_interest"),
            availability=self.availability,
            factor=self.name,
        )
        if missing_opt:
            warnings.warn(
                f"{self.name}: sin columnas {list(missing_opt)}; se asumen 0 en el EV "
                "(degradación documentada: infravalora el EV de emisores con "
                "preferentes o minoritarios relevantes)",
                UserWarning,
                stacklevel=2,
            )

        ebitda_q = (
            q["ebitda"]
            if "ebitda" in q.columns
            else q["operating_income"] + q["depreciation_amortization"]
        )
        debt_q = (
            q["total_debt"]
            if "total_debt" in q.columns
            else q["short_term_debt"] + q["long_term_debt"]
        )
        preferred = q["preferred_equity"] if "preferred_equity" in q.columns else 0.0
        minority = q["minority_interest"] if "minority_interest" in q.columns else 0.0

        q["ebitda_ttm"] = ttm_sum(q, ebitda_q)
        q["net_claims"] = debt_q + preferred + minority - q["cash"]

        daily = self._daily_values(ctx, q, ["ebitda_ttm", "net_claims"])
        mc = market_cap_panel(ctx, factor=self.name).reindex(daily.index)
        ev = mc + daily["net_claims"]
        return finalize_factor(daily["ebitda_ttm"] / positive_only(ev), name=self.name)


FACTORS: tuple[type[FundamentalFactor], ...] = (
    EarningsYield,
    FreeCashFlowYield,
    BookToMarket,
    EbitdaToEV,
    SalesYield,
)
"""Factores exportados por este módulo, en orden de la tarea (§3.4)."""


def _register() -> None:
    """Registra los factores en el `default_registry` de `factors.base`.

    Import en tiempo de ejecución con try/except: `base.py` lo escribe otro
    agente en paralelo y este módulo debe funcionar también sin él.
    """
    try:
        from earnings_alpha.factors.base import register_factor
    except ImportError:  # pragma: no cover - base.py aún no integrado
        return
    for cls in FACTORS:
        register_factor()(cls)


_register()
