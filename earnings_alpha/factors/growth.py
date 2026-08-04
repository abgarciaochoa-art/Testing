"""Factores de crecimiento sobre fundamentales point-in-time.

Implementa la §10 de ``docs/research/fundamental_factors.md``:

- ``SalesGrowth``       — crecimiento interanual de ventas TTM **por acción**,
  con signo NEGATIVO (contrarian).
- ``SalesAcceleration`` — aceleración del crecimiento trimestral interanual,
  factor **exploratorio**.
- ``EarningsGrowth``    — crecimiento interanual del BPA TTM, signo NEGATIVO.
- ``SustainableGrowth`` — crecimiento sostenible ``ROE × (1 − payout)``.
- ``AssetGrowth``       — crecimiento interanual del activo, signo NEGATIVO.

El punto contraintuitivo, documentado en §10.2 del informe y que este módulo
respeta a rajatabla: **el nivel de crecimiento pasado predice rentabilidades
BAJAS**. Lakonishok, Shleifer y Vishny (1994) muestran que los valores
*glamour* (alto crecimiento pasado) rinden menos porque el inversor típico
extrapola; Cooper, Gulen y Schill (2008) documentan que el crecimiento del
activo es un predictor negativo que **mantiene su poder incluso en large caps**
— relevante para el S&P 500—. Por eso ``SalesGrowth``, ``EarningsGrowth`` y
``AssetGrowth`` devuelven el crecimiento **con el signo cambiado** (mayor valor
del factor = menos crecimiento pasado = más alcista), y lo que no es defendible
es meterlos con signo positivo "porque crecer es bueno".

Por acción, no totales: sobre una serie con recompras del 10 %, el SURGE sobre
ingresos totales y sobre ingresos por acción tienen **signos opuestos**
(verificado numéricamente en §3 del informe). Con totales, el factor mide en
buena parte la política de recompra (*net share issuance*), que es otro factor.
Por eso este módulo exige el número de acciones y falla explícitamente sin él.

Referencias
-----------
- Lakonishok, J., Shleifer, A. y Vishny, R. W. (1994). *Contrarian Investment,
  Extrapolation, and Risk*. JF 49(5), 1541-1578.
- Cooper, M. J., Gulen, H. y Schill, M. J. (2008). *Asset Growth and the
  Cross-Section of Stock Returns*. JF 63(4), 1609-1651.
- Jegadeesh, N. y Livnat, J. (2006). *Revenue Surprises and Stock Returns*.
  JAE 41(1-2), 147-171 (base de la construcción por acción).
- Higgins, R. C. (1977). *How Much Growth Can a Firm Afford?*. Financial
  Management 6(3), 7-16 (crecimiento sostenible).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd

from earnings_alpha.errors import DataQualityError, InsufficientHistory
from earnings_alpha.factors.value import (
    DEFAULT_MAX_STALENESS_DAYS,
    Availability,
    FactorContextLike,
    FundamentalFactor,
    finalize_factor,
    grouped_rolling,
    positive_only,
    quarterly_history,
    shift_q,
    ttm_sum,
)

__all__ = [  # noqa: RUF022 - orden temático
    "SalesGrowth",
    "SalesAcceleration",
    "EarningsGrowth",
    "SustainableGrowth",
    "AssetGrowth",
    "FACTORS",
    "MIN_QUARTERS_GROWTH",
    "MIN_QUARTERS_ACCEL_STD",
    "MIN_QUARTERS_ACCEL_RAW",
    "sales_growth_ttm",
    "sales_acceleration",
    "sustainable_growth",
]

MIN_QUARTERS_GROWTH = 8
"""``g_TTM = RevTTM_q / RevTTM_{q-4} - 1`` necesita los trimestres ``q-7..q``."""

MIN_QUARTERS_ACCEL_RAW = 6
"""``a = g_q - g_{q-1}`` con ``g_q = r_q/r_{q-4} - 1`` necesita ``r_{q-5}..r_q``."""

MIN_QUARTERS_ACCEL_STD = 13
"""``a_std = a / sd(g_{q-8..q-1})`` necesita ``r_{q-12}``: 13 trimestres
consecutivos, verificado en §10.1 del informe (con 12 debe lanzar
``InsufficientHistory``)."""


# ---------------------------------------------------------------------------
# Funciones puras (una empresa, una serie trimestral ordenada antiguo → reciente)
# ---------------------------------------------------------------------------


def sales_growth_ttm(revenue_per_share: Sequence[float]) -> float:
    """Crecimiento interanual de ventas TTM por acción: ``g = TTM_q/TTM_{q-4} − 1``.

    ``fundamental_factors.md`` §10.1. La serie debe estar en ingresos **por
    acción** y ordenada del trimestre más antiguo al más reciente; se usan las
    últimas ``MIN_QUARTERS_GROWTH`` observaciones.

    - Menos de 8 observaciones → ``InsufficientHistory`` (nunca un crecimiento
      calculado sobre menos trimestres de los exigidos).
    - TTM base no positivo → NaN (una tasa de crecimiento desde base negativa
      no es interpretable).
    """
    r = np.asarray(list(revenue_per_share), dtype=float)
    if len(r) < MIN_QUARTERS_GROWTH:
        msg = (
            f"sales_growth_ttm exige {MIN_QUARTERS_GROWTH} trimestres consecutivos y hay "
            f"{len(r)} (fundamental_factors.md §10.1)"
        )
        raise InsufficientHistory(msg)
    ttm_now = float(r[-4:].sum())
    ttm_prev = float(r[-8:-4].sum())
    if not math.isfinite(ttm_prev) or ttm_prev <= 0 or not math.isfinite(ttm_now):
        return math.nan
    return ttm_now / ttm_prev - 1.0


def sales_acceleration(
    revenue_per_share: Sequence[float], *, standardized: bool = True
) -> float:
    """Aceleración del crecimiento trimestral interanual (§10.1 del informe).

    ``g_q = r_q / r_{q-4} − 1``;  ``a = g_q − g_{q-1}`` (segunda diferencia);
    ``a_std = a / sd(g_{q-8..q-1})`` con ``ddof=1``.

    - ``standardized=True`` exige 13 trimestres (``MIN_QUARTERS_ACCEL_STD``,
      verificado: con 12 lanza ``InsufficientHistory``); la versión bruta exige
      6 (``MIN_QUARTERS_ACCEL_RAW``).
    - Denominadores no positivos o dispersión nula → NaN.
    """
    r = np.asarray(list(revenue_per_share), dtype=float)
    needed = MIN_QUARTERS_ACCEL_STD if standardized else MIN_QUARTERS_ACCEL_RAW
    if len(r) < needed:
        msg = (
            f"sales_acceleration(standardized={standardized}) exige {needed} trimestres "
            f"consecutivos y hay {len(r)} (fundamental_factors.md §10.1)"
        )
        raise InsufficientHistory(msg)

    def growth(idx: int) -> float:
        base = r[idx - 4]
        if not math.isfinite(base) or base <= 0:
            return math.nan
        return float(r[idx] / base - 1.0)

    last = len(r) - 1
    a = growth(last) - growth(last - 1)
    if not standardized:
        return a
    g_hist = np.array([growth(last - j) for j in range(1, 9)], dtype=float)
    if np.isnan(g_hist).any():
        return math.nan
    sd = float(np.std(g_hist, ddof=1))
    if sd <= 0 or math.isnan(a):
        return math.nan
    return a / sd


def sustainable_growth(
    *, net_income_ttm: float, dividends_ttm: float, avg_equity: float
) -> float:
    """Crecimiento sostenible de Higgins (1977): ``g = ROE × (1 − payout)``.

    ``ROE = NI_TTM / patrimonio medio``; ``payout = dividendos_TTM / NI_TTM``.
    NaN si el beneficio TTM no es positivo (payout indefinido) o el patrimonio
    medio no es positivo.
    """
    ni = float(net_income_ttm)
    div = float(dividends_ttm)
    eq = float(avg_equity)
    if not math.isfinite(ni) or ni <= 0 or not math.isfinite(eq) or eq <= 0:
        return math.nan
    roe = ni / eq
    retention = 1.0 - div / ni
    return roe * retention


# ---------------------------------------------------------------------------
# Apoyo de panel
# ---------------------------------------------------------------------------


def _per_share(q: pd.DataFrame, numerator: pd.Series | str, *, factor: str) -> pd.Series:
    """Convierte una partida a "por acción" con las acciones de su propio trimestre.

    Preferencia: ``shares_diluted``; degradación documentada a
    ``shares_outstanding``. Sin ninguna de las dos se falla explícitamente:
    calcular crecimiento sobre totales mide política de recompra, no demanda
    (§3 y §10.1 del informe), y ese error es silencioso.
    """
    if "shares_diluted" in q.columns:
        shares = q["shares_diluted"].astype("float64")
    elif "shares_outstanding" in q.columns:
        warnings.warn(
            f"{factor}: sin `shares_diluted`; se usan `shares_outstanding` "
            "(degradación documentada: ignora la dilución potencial)",
            UserWarning,
            stacklevel=3,
        )
        shares = q["shares_outstanding"].astype("float64")
    else:  # pragma: no cover - protegido por quarterly_history con required dinámico
        msg = f"{factor}: sin columna de acciones no hay magnitudes por acción"
        raise DataQualityError(msg)
    s = q[numerator] if isinstance(numerator, str) else numerator
    return s / positive_only(shares)


def _shares_columns(ctx: FactorContextLike, *, factor: str) -> list[str]:
    """Resuelve qué columna de acciones existe, fallando explícitamente si ninguna."""
    fund = getattr(ctx, "fundamentals", None)
    if not isinstance(fund, pd.DataFrame):
        msg = f"{factor}: el contexto no trae `fundamentals` como DataFrame"
        raise DataQualityError(msg)
    if "shares_diluted" in fund.columns:
        return ["shares_diluted"]
    if "shares_outstanding" in fund.columns:
        return ["shares_outstanding"]
    msg = (
        f"{factor}: `fundamentals` no trae `shares_diluted` ni `shares_outstanding`; "
        "el crecimiento debe medirse POR ACCIÓN (fundamental_factors.md §3: con "
        "totales, una recompra del 10 % invierte el signo del factor)"
    )
    raise DataQualityError(msg)


# ---------------------------------------------------------------------------
# Factores
# ---------------------------------------------------------------------------


class SalesGrowth(FundamentalFactor):
    """Crecimiento interanual de ventas TTM por acción, con signo NEGATIVO.

    Referencias: Lakonishok, Shleifer y Vishny (1994) — los valores *glamour*,
    definidos entre otras cosas por alto crecimiento de ventas pasado, rinden
    menos; la estrategia contraria explota la extrapolación del inversor
    típico, no un mayor riesgo—. Construcción por acción de Jegadeesh y Livnat
    (2006). Véase la cabecera del módulo.

    - **Signo (documentado como exige el contrato):** el factor devuelve
      ``−g_TTM``. Mayor valor = menos crecimiento pasado = más alcista.
    - **PIT:** ``"announcement"`` (ingresos y acciones vienen en la nota de
      prensa).
    - **Historia mínima:** 8 trimestres consecutivos; con menos, NaN para ese
      nombre (la función pura ``sales_growth_ttm`` lanza ``InsufficientHistory``
      con menos de 8, y el panel entero sin historia también).
    - **Dónde falla (§10.4):** empresas muy adquisitivas (crecimiento
      inorgánico); Energía y Materiales (el crecimiento es el precio de la
      materia prima: neutralización sectorial imprescindible); Financieras
      ("ventas" no homogéneo). Estacionalidad extrema: por eso siempre TTM
      interanual, jamás secuencial.
    """

    name = "sales_growth"
    requires: list[str] = ["fundamentals"]  # noqa: RUF012
    availability: Availability = "announcement"

    def compute(self, ctx: FactorContextLike) -> pd.Series:
        shares_cols = _shares_columns(ctx, factor=self.name)
        q, _ = quarterly_history(
            ctx, ["revenue", *shares_cols], availability=self.availability, factor=self.name
        )
        rps = _per_share(q, "revenue", factor=self.name)
        rps_ttm = ttm_sum(q, rps)
        base = positive_only(shift_q(q, rps_ttm, 4))
        q["value"] = -(rps_ttm / base - 1.0)
        daily = self._daily_values(ctx, q, ["value"])
        return finalize_factor(daily["value"], name=self.name)


class SalesAcceleration(FundamentalFactor):
    """Aceleración del crecimiento de ventas por acción — factor EXPLORATORIO.

    Honestidad sobre la evidencia (§10.3 del informe): la aceleración es
    popular entre practicantes (CANSLIM, "Earnings & Sales Acceleration") pero
    **no tiene un artículo de referencia en JF/JFE/JAR/RFS** que la aísle como
    factor con rentabilidades ajustadas al riesgo replicadas. En consecuencia:

    1. este docstring la marca como exploratoria;
    2. **no** se le asigna un prior de IC, porque no existe;
    3. debe validarse con el arsenal completo de ``stats`` (CV purgada con
       embargo, Sharpe deflactado, Benjamini-Hochberg junto al resto de
       candidatos): es exactamente el tipo de señal que produce falsos
       positivos evaluada aislada;
    4. si se combina, debe entrar **ortogonalizada contra el nivel** de
       crecimiento (racional en §10.3: nivel negativo por extrapolación,
       aceleración positiva por infrarreacción; sin ortogonalizar se cancelan).

    - **Signo:** positivo (aceleración = más alcista), *sujeto a validación
      interna*.
    - **PIT:** ``"announcement"``.
    - **Historia mínima:** 13 trimestres consecutivos en la variante
      estandarizada (verificado en §10.1: con 12, ``InsufficientHistory`` en la
      función pura y NaN por nombre en el panel); 6 en la bruta.
    """

    requires: list[str] = ["fundamentals"]  # noqa: RUF012
    availability: Availability = "announcement"

    def __init__(
        self,
        *,
        standardized: bool = True,
        max_staleness_days: int | None = DEFAULT_MAX_STALENESS_DAYS,
    ) -> None:
        super().__init__(max_staleness_days=max_staleness_days)
        self.standardized = bool(standardized)
        self.name = "sales_acceleration" if standardized else "sales_acceleration_raw"

    def compute(self, ctx: FactorContextLike) -> pd.Series:
        shares_cols = _shares_columns(ctx, factor=self.name)
        q, _ = quarterly_history(
            ctx, ["revenue", *shares_cols], availability=self.availability, factor=self.name
        )
        rps = _per_share(q, "revenue", factor=self.name)
        base = positive_only(shift_q(q, rps, 4))
        g_q = rps / base - 1.0
        accel = g_q - shift_q(q, g_q, 1)
        if self.standardized:
            # sd de g en q-8..q-1: se desplaza una posición y se toma la ventana de 8.
            g_hist_sd = grouped_rolling(q, shift_q(q, g_q, 1), window=8, how="std")
            accel = accel / g_hist_sd.where(g_hist_sd > 0)
        q["value"] = accel
        daily = self._daily_values(ctx, q, ["value"])
        return finalize_factor(daily["value"], name=self.name)


class EarningsGrowth(FundamentalFactor):
    """Crecimiento interanual del BPA TTM, con signo NEGATIVO (contrarian).

    Referencia: Lakonishok, Shleifer y Vishny (1994) — la sobre-extrapolación
    del crecimiento pasado que penaliza a los *glamour* se mide allí también
    con crecimiento de beneficios pasado—. La evidencia directa más fuerte es
    para ventas y activos (§10.2); el signo negativo aquí es coherente con esa
    familia y debe revalidarse internamente antes de ponderarlo en una
    combinación.

    - **Signo (documentado):** el factor devuelve ``−g_E``. Mayor valor = menos
      crecimiento pasado de BPA = más alcista.
    - **Construcción:** BPA TTM por acción del propio trimestre (columna
      ``eps_diluted``; degradación documentada a ``net_income/acciones``);
      ``g_E = E_TTM_q / E_TTM_{q-4} − 1`` con base positiva. Base TTM no
      positiva → NaN: una tasa de crecimiento desde pérdidas no es
      interpretable (misma lógica que ``SurpriseBasis.ABS_ESTIMATE``, §2.4).
    - **PIT:** ``"announcement"``.
    - **Historia mínima:** 8 trimestres consecutivos.
    - **Aviso:** con base positiva pequeña el cociente explota; el winsorizado
      cross-section de la receta canónica (§1.4) es obligatorio aguas abajo.
    """

    name = "earnings_growth"
    requires: list[str] = ["fundamentals"]  # noqa: RUF012
    availability: Availability = "announcement"

    def compute(self, ctx: FactorContextLike) -> pd.Series:
        fund = getattr(ctx, "fundamentals", None)
        if not isinstance(fund, pd.DataFrame):
            msg = f"{self.name}: el contexto no trae `fundamentals` como DataFrame"
            raise DataQualityError(msg)
        if "eps_diluted" in fund.columns:
            q, _ = quarterly_history(
                ctx, ["eps_diluted"], availability=self.availability, factor=self.name
            )
            eps = q["eps_diluted"].astype("float64")
        else:
            shares_cols = _shares_columns(ctx, factor=self.name)
            warnings.warn(
                f"{self.name}: sin `eps_diluted`; se deriva como net_income/acciones "
                "(degradación documentada)",
                UserWarning,
                stacklevel=2,
            )
            q, _ = quarterly_history(
                ctx,
                ["net_income", *shares_cols],
                availability=self.availability,
                factor=self.name,
            )
            eps = _per_share(q, "net_income", factor=self.name)
        eps_ttm = ttm_sum(q, eps)
        base = positive_only(shift_q(q, eps_ttm, 4))
        q["value"] = -(eps_ttm / base - 1.0)
        daily = self._daily_values(ctx, q, ["value"])
        return finalize_factor(daily["value"], name=self.name)


class SustainableGrowth(FundamentalFactor):
    """Crecimiento sostenible: ``g = ROE_TTM × (1 − payout)`` (Higgins 1977).

    Referencia: Higgins (1977), *How Much Growth Can a Firm Afford?*, Financial
    Management 6(3). Es el crecimiento financiable con fondos internos sin
    alterar la estructura de capital: una medida de capacidad, no de
    crecimiento realizado, y por eso NO lleva el signo contrarian de §10.2 —
    su contenido es de rentabilidad retenida, emparentado con el pilar de
    *profitability* de Asness, Frazzini y Pedersen (2019).

    - **Signo:** positivo (más capacidad de crecimiento autofinanciado = más
      alcista). **Aviso de solapamiento:** correlaciona con ROE por
      construcción; medir antes de combinar con ``quality.Profitability``.
    - **PIT:** ``"filing"`` (el patrimonio es balance).
    - **NaN explícito:** beneficio TTM no positivo (payout indefinido) o
      patrimonio medio no positivo.
    - **Historia mínima:** 8 trimestres (patrimonio medio interanual).
    """

    name = "sustainable_growth"
    requires: list[str] = ["fundamentals"]  # noqa: RUF012
    availability: Availability = "filing"

    def compute(self, ctx: FactorContextLike) -> pd.Series:
        q, _ = quarterly_history(
            ctx,
            ["net_income", "dividends_paid", "total_equity"],
            availability=self.availability,
            factor=self.name,
        )
        ni_ttm = positive_only(ttm_sum(q, "net_income"))
        div_ttm = ttm_sum(q, "dividends_paid")
        equity = q["total_equity"].astype("float64")
        avg_equity = positive_only((equity + shift_q(q, equity, 4)) / 2.0)
        roe = ni_ttm / avg_equity
        retention = 1.0 - div_ttm / ni_ttm
        q["value"] = roe * retention
        daily = self._daily_values(ctx, q, ["value"])
        return finalize_factor(daily["value"], name=self.name)


class AssetGrowth(FundamentalFactor):
    """Crecimiento interanual del activo total, con signo NEGATIVO.

    Referencia: Cooper, Gulen y Schill (2008), JF 63(4), 1609-1651 — la tasa de
    crecimiento anual del activo es un predictor **negativo**, económica y
    estadísticamente significativo, que **mantiene su capacidad predictiva en
    valores de gran capitalización** y sobrevive al control por book-to-market,
    tamaño, momentum y accruals. Uno de los pocos factores del informe que no
    se degrada en nuestro universo (§10.2).

    - **Signo (documentado):** el factor devuelve ``−(TA_q/TA_{q−4} − 1)``.
      Mayor valor = balance más disciplinado = más alcista.
    - **PIT:** ``"filing"`` (el activo es balance).
    - **Historia mínima:** 5 trimestres consecutivos (nivel y su interanual).
    - **Dónde falla:** Financieras — el crecimiento del balance de un banco es
      su negocio, no una decisión de inversión comparable [§10.4 recomienda
      neutralización sectorial; aquí se mantienen con esa condición]. **Aviso
      de solapamiento (§11.4):** correlaciona con ``FCF yield`` (capex alto ⇒
      activo creciendo y FCF bajo); no sumarlos como independientes.
    """

    name = "asset_growth"
    requires: list[str] = ["fundamentals"]  # noqa: RUF012
    availability: Availability = "filing"

    def compute(self, ctx: FactorContextLike) -> pd.Series:
        q, _ = quarterly_history(
            ctx, ["total_assets"], availability=self.availability, factor=self.name
        )
        ta = q["total_assets"].astype("float64")
        base = positive_only(shift_q(q, ta, 4))
        q["value"] = -(ta / base - 1.0)
        daily = self._daily_values(ctx, q, ["value"])
        return finalize_factor(daily["value"], name=self.name)


FACTORS: tuple[type[FundamentalFactor], ...] = (
    SalesGrowth,
    SalesAcceleration,
    EarningsGrowth,
    SustainableGrowth,
    AssetGrowth,
)
"""Factores exportados por este módulo."""


def _register() -> None:
    """Registra los factores en el `default_registry` de `factors.base`.

    Import en tiempo de ejecución con try/except: `base.py` lo escribe otro
    agente en paralelo y este módulo debe funcionar también sin él.
    """
    try:
        from earnings_alpha.factors.base import register_factor
    except ImportError:  # pragma: no cover - base.py aún no integrado
        return
    for cls in (SalesGrowth, EarningsGrowth, SustainableGrowth, AssetGrowth):
        register_factor()(cls)

    @register_factor("sales_acceleration")
    def _accel_std(**kwargs: object) -> SalesAcceleration:
        return SalesAcceleration(standardized=True, **kwargs)  # type: ignore[arg-type]

    @register_factor("sales_acceleration_raw")
    def _accel_raw(**kwargs: object) -> SalesAcceleration:
        return SalesAcceleration(standardized=False, **kwargs)  # type: ignore[arg-type]


_register()
