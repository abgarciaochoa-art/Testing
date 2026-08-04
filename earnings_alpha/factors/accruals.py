"""Factores de accruals (devengos) sobre fundamentales point-in-time.

Implementa la §7 de ``docs/research/fundamental_factors.md``:

- ``SloanAccrualsBalance``  — accruals de Sloan (1996) por **balance**, con la
  fórmula original exacta documentada abajo.
- ``SloanAccrualsCashFlow`` — accruals por **estado de flujos** (Collins y
  Hribar 2002), el método primario recomendado para el S&P 500.
- ``PercentAccruals``       — *percent accruals* de Hafzalla, Lundholm y
  Van Winkle (2011), el factor primario del informe.
- ``NetOperatingAccruals``  — accruals operativos netos: variación de los
  activos operativos netos (Richardson, Sloan, Soliman y Tuna 2005;
  Hirshleifer, Hou, Teoh y Zhang 2004).

El mecanismo económico (Sloan 1996): el componente de devengo del beneficio es
menos persistente que el componente de caja (β_ACC ≈ 0.765 vs β_CF ≈ 0.855 en
la regresión de persistencia [verificar contra el original, §7.4 del informe]),
pero el mercado fija precios como si ambos persistieran igual. Ese diferencial
**es** el factor.

**Signo — el error más caro del módulo.** Accruals altos = malo. El contrato
(§3.4) exige valor mayor = más alcista, así que TODOS los factores de este
módulo devuelven la medida **con el signo cambiado** (``−PACC``, ``−ACC_CF``,
``−ACC_BS``, ``−ΔNOA``). Un signo invertido aquí produce un backtest con Sharpe
negativo simétrico que se lee como "el factor no funciona" cuando funciona al
revés (§7.4).

**Disponibilidad point-in-time.** ``available_at`` de un accrual es la
aceptación del **10-Q/10-K** (``filed_at``), nunca la nota de prensa: la
mayoría de comunicados de resultados no incluyen estado de flujos completo, y
en el S&P 500 la mediana del desfase anuncio → filing ronda las 4-6 semanas
(§1.2, §7.4). Es el desfase más fácil de olvidar y el más caro.

Referencias
-----------
- Sloan, R. G. (1996). *Do Stock Prices Fully Reflect Information in Accruals
  and Cash Flows About Future Earnings?*. The Accounting Review 71(3), 289-315.
- Collins, D. W. y Hribar, P. (2002). *Errors in Estimating Accruals:
  Implications for Empirical Research*. JAR 40(1), 105-134.
- Hafzalla, N., Lundholm, R. y Van Winkle, E. M. (2011). *Percent Accruals*.
  The Accounting Review 86(1), 209-236.
- Richardson, S. A., Sloan, R. G., Soliman, M. T. y Tuna, I. (2005). *Accrual
  Reliability, Earnings Persistence and Stock Prices*. JAE 39(3), 437-485.
- Hirshleifer, D., Hou, K., Teoh, S. H. y Zhang, Y. (2004). *Do Investors
  Overvalue Firms with Bloated Balance Sheets?*. JAE 38, 297-331.
"""

from __future__ import annotations

import math
import warnings
from typing import Literal

import pandas as pd

from earnings_alpha.errors import DataQualityError, InsufficientHistory
from earnings_alpha.factors.value import (
    DEFAULT_EXCLUDE_SECTORS,
    DEFAULT_MAX_STALENESS_DAYS,
    Availability,
    FactorContextLike,
    FundamentalFactor,
    finalize_factor,
    positive_only,
    quarterly_history,
    shift_q,
    to_daily,
    ttm_sum,
)

__all__ = [  # noqa: RUF022 - orden temático
    "SloanAccrualsBalance",
    "SloanAccrualsCashFlow",
    "PercentAccruals",
    "NetOperatingAccruals",
    "FACTORS",
    "sloan_accruals_balance",
    "sloan_accruals_cashflow",
    "percent_accruals",
    "accruals_articulation_gap",
]

Frequency = Literal["ttm", "annual"]
"""``"ttm"`` (por defecto): deltas interanuales refrescados cada trimestre.
``"annual"``: solo al cierre de ejercicio fiscal (filas Q4/10-K), la
formulación literal de Sloan (1996)."""


# ---------------------------------------------------------------------------
# Funciones puras — la fórmula exacta, verificable con números de juguete
# ---------------------------------------------------------------------------


def sloan_accruals_balance(
    *,
    delta_current_assets: float,
    delta_cash: float,
    delta_current_liabilities: float,
    delta_short_term_debt: float,
    delta_taxes_payable: float,
    depreciation: float,
    avg_total_assets: float,
) -> float:
    """Accruals de Sloan (1996) por el método de balance — fórmula exacta.

    .. code-block:: text

        ACC_BS = [ (ΔCA − ΔCash) − (ΔCL − ΔSTD − ΔTXP) − DEP ] / ((TA_t + TA_{t-1})/2)

    con la correspondencia Compustat del artículo (§7.1 del informe):
    ``ΔCA``=ACT, ``ΔCash``=CHE, ``ΔCL``=LCT, ``ΔSTD``=DLC, ``ΔTXP``=TXP,
    ``DEP``=DP, deflactor = activo total medio (AT). La lógica: variación del
    capital circulante operativo neto —excluyendo caja y financiación a corto—
    menos la amortización.

    Activo medio no positivo → NaN.
    """
    if not math.isfinite(avg_total_assets) or avg_total_assets <= 0:
        return math.nan
    numerator = (
        (delta_current_assets - delta_cash)
        - (delta_current_liabilities - delta_short_term_debt - delta_taxes_payable)
        - depreciation
    )
    return numerator / avg_total_assets


def sloan_accruals_cashflow(
    *, net_income: float, cfo: float, avg_total_assets: float
) -> float:
    """Accruals por estado de flujos (Collins y Hribar 2002) — fórmula exacta.

    .. code-block:: text

        ACC_CF = ( NI − CFO ) / ((TA_t + TA_{t-1})/2)

    con ``NI`` = resultado antes de extraordinarios (Compustat IB) y ``CFO`` =
    flujo operativo del estado de flujos (OANCF) menos partidas extraordinarias
    y discontinuadas (XIDOC); el proveedor de la columna ``cfo`` debe respetar
    esa exclusión. Activo medio no positivo → NaN.
    """
    if not math.isfinite(avg_total_assets) or avg_total_assets <= 0:
        return math.nan
    return (net_income - cfo) / avg_total_assets


def percent_accruals(*, net_income: float, cfo: float) -> float:
    """Percent accruals (Hafzalla, Lundholm y Van Winkle 2011) — fórmula exacta.

    .. code-block:: text

        PACC = ( NI − CFO ) / |NI|

    El valor absoluto del denominador mantiene la medida definida y ordenable
    con pérdidas (verificado en §7.3 del informe: ``NI=−200, CFO=50`` →
    ``PACC=−1.25``), que es donde el accrual escalado por activos falla.
    ``NI == 0`` → NaN.
    """
    if not math.isfinite(net_income) or net_income == 0.0:
        return math.nan
    return (net_income - cfo) / abs(net_income)


# ---------------------------------------------------------------------------
# Apoyo de panel
# ---------------------------------------------------------------------------


def _restrict_annual(q: pd.DataFrame, *, factor: str) -> pd.DataFrame:
    """Restringe el panel trimestral a cierres de ejercicio fiscal (Q4/10-K)."""
    if "fiscal_period" in q.columns:
        annual = q["fiscal_period"].astype(str).str.endswith("Q4")
    elif "form" in q.columns:
        annual = q["form"].astype(str).eq("10-K")
    else:
        msg = (
            f"{factor}: frequency='annual' necesita `fiscal_period` o `form` para "
            "identificar el cierre de ejercicio"
        )
        raise DataQualityError(msg)
    out = q[annual.to_numpy()].reset_index(drop=True)
    if len(out) == 0:
        msg = f"{factor}: no hay cierres de ejercicio fiscal en `fundamentals`"
        raise InsufficientHistory(msg)
    return out


def _balance_accruals_quarterly(
    q: pd.DataFrame, *, has_taxes_payable: bool, factor: str
) -> pd.Series:
    """ACC_BS interanual sobre el panel trimestral (deltas a 4 trimestres)."""
    ca = q["current_assets"].astype("float64")
    cash = q["cash"].astype("float64")
    cl = q["current_liabilities"].astype("float64")
    std = q["short_term_debt"].astype("float64")
    ta = q["total_assets"].astype("float64")

    d_ca = ca - shift_q(q, ca, 4)
    d_cash = cash - shift_q(q, cash, 4)
    d_cl = cl - shift_q(q, cl, 4)
    d_std = std - shift_q(q, std, 4)
    if has_taxes_payable:
        txp = q["taxes_payable"].astype("float64")
        d_txp = txp - shift_q(q, txp, 4)
    else:
        d_txp = pd.Series(0.0, index=q.index)
        warnings.warn(
            f"{factor}: sin `taxes_payable`; se asume ΔTXP = 0, la convención "
            "Compustat cuando TXP falta (degradación documentada: sobreestima "
            "los accruals de empresas cuyo pasivo fiscal creció)",
            UserWarning,
            stacklevel=3,
        )
    dep_ttm = ttm_sum(q, "depreciation_amortization")
    avg_ta = positive_only((ta + shift_q(q, ta, 4)) / 2.0)
    return ((d_ca - d_cash) - (d_cl - d_std - d_txp) - dep_ttm) / avg_ta


def _cashflow_accruals_quarterly(q: pd.DataFrame) -> pd.Series:
    """ACC_CF interanual sobre el panel trimestral (TTM y activo medio)."""
    ta = q["total_assets"].astype("float64")
    avg_ta = positive_only((ta + shift_q(q, ta, 4)) / 2.0)
    return (ttm_sum(q, "net_income") - ttm_sum(q, "cfo")) / avg_ta


_BALANCE_COLUMNS = [
    "current_assets",
    "cash",
    "current_liabilities",
    "short_term_debt",
    "depreciation_amortization",
    "total_assets",
]


def accruals_articulation_gap(
    ctx: FactorContextLike, *, frequency: Frequency = "ttm"
) -> pd.Series:
    """Bandera de calidad de datos: ``|ACC_BS − ACC_CF|`` diario por nombre.

    Collins y Hribar (2002) demuestran que la discrepancia entre ambos métodos
    señala años con fusiones, desinversiones o conversión de divisa que rompen
    la articulación balance ↔ resultados — actividad que contamina también SUE
    y el crecimiento de ventas—. La regla del repo (§7.2): usar esta serie como
    bandera de exclusión o ``DataQualityError`` cuando exceda un umbral.
    """
    name = "accruals_articulation_gap"
    q, missing_opt = quarterly_history(
        ctx,
        [*_BALANCE_COLUMNS, "net_income", "cfo"],
        optional=("taxes_payable",),
        availability="filing",
        factor=name,
    )
    if frequency == "annual":
        q = _restrict_annual(q, factor=name)
    acc_bs = _balance_accruals_quarterly(
        q, has_taxes_payable="taxes_payable" not in missing_opt, factor=name
    )
    acc_cf = _cashflow_accruals_quarterly(q)
    q["value"] = (acc_bs - acc_cf).abs()
    daily = to_daily(ctx, q, ["value"], factor=name)
    return finalize_factor(daily["value"], name=name)


# ---------------------------------------------------------------------------
# Factores
# ---------------------------------------------------------------------------


class SloanAccrualsBalance(FundamentalFactor):
    """Accruals de Sloan (1996) por balance, factor = ``−ACC_BS``.

    Referencia: Sloan (1996), The Accounting Review 71(3). Fórmula exacta en la
    función pura ``sloan_accruals_balance`` (este factor la aplica con deltas
    interanuales sobre el panel trimestral, o anuales con
    ``frequency="annual"``).

    **Advertencia de método (§7.2):** Collins y Hribar (2002) demuestran que el
    método de balance está contaminado por fusiones, adquisiciones,
    desinversiones y conversión de divisa — y el S&P 500 está poblado por las
    empresas más adquisitivas y multinacionales del mercado—. Se implementa
    como *control* y para construir la bandera de discrepancia
    (``accruals_articulation_gap``); el primario es ``PercentAccruals`` y el
    secundario ``SloanAccrualsCashFlow``.

    - **Signo (documentado):** devuelve ``−ACC_BS``; mayor = menos devengo =
      más alcista.
    - **PIT:** ``"filing"`` — balance y amortización exigen el 10-Q/10-K.
    - **Requisitos:** ``current_assets, cash, current_liabilities,
      short_term_debt, depreciation_amortization, total_assets``. Opcional
      ``taxes_payable``: si falta, ``ΔTXP = 0`` con aviso (convención Compustat
      documentada).
    - **Historia mínima:** 5 trimestres consecutivos (deltas interanuales) y 4
      para la amortización TTM.
    - **Dónde falla (NaN estructural, §7.4):** Financieras — el capital
      circulante operativo no existe — e Inmobiliarias — la amortización
      enorme empuja los accruals mecánicamente a muy negativos sin contenido
      informativo—.
    """

    requires: list[str] = ["fundamentals"]  # noqa: RUF012
    availability: Availability = "filing"
    exclude_sectors: frozenset[str] = DEFAULT_EXCLUDE_SECTORS

    def __init__(
        self,
        *,
        frequency: Frequency = "ttm",
        max_staleness_days: int | None = None,
    ) -> None:
        if frequency not in ("ttm", "annual"):
            msg = f"frequency debe ser 'ttm' o 'annual'; recibido {frequency!r}"
            raise ValueError(msg)
        if max_staleness_days is None:
            max_staleness_days = 550 if frequency == "annual" else DEFAULT_MAX_STALENESS_DAYS
        super().__init__(max_staleness_days=max_staleness_days)
        self.frequency: Frequency = frequency
        self.name = "accruals_bs" if frequency == "ttm" else "accruals_bs_annual"

    def compute(self, ctx: FactorContextLike) -> pd.Series:
        q, missing_opt = quarterly_history(
            ctx,
            _BALANCE_COLUMNS,
            optional=("taxes_payable",),
            availability=self.availability,
            factor=self.name,
        )
        if self.frequency == "annual":
            q = _restrict_annual(q, factor=self.name)
        acc_bs = _balance_accruals_quarterly(
            q, has_taxes_payable="taxes_payable" not in missing_opt, factor=self.name
        )
        q["value"] = -acc_bs
        daily = self._daily_values(ctx, q, ["value"])
        return finalize_factor(daily["value"], name=self.name)


class SloanAccrualsCashFlow(FundamentalFactor):
    """Accruals por estado de flujos (Collins-Hribar 2002), factor = ``−ACC_CF``.

    Referencia: Collins y Hribar (2002), JAR 40(1) — el método de flujos evita
    la contaminación por M&A y divisa del método de balance; es el método
    **recomendado para el S&P 500** (§7.2 del informe, con verificación
    numérica de una discrepancia del 43 % en un caso sin siquiera modelar una
    adquisición)—. Fórmula exacta en ``sloan_accruals_cashflow``.

    - **Signo (documentado):** devuelve ``−ACC_CF``; mayor = beneficio más
      respaldado por caja = más alcista.
    - **PIT:** ``"filing"`` — el CFO no viene en la nota de prensa (§7.4:
      "Nunca del 8-K").
    - **Requisitos:** ``net_income, cfo, total_assets``; 8 trimestres
      consecutivos (TTM + activo medio interanual).
    - ``max_articulation_gap``: si se fija, las observaciones cuya discrepancia
      ``|ACC_BS − ACC_CF|`` exceda el umbral se invalidan a NaN (regla de §7.2;
      exige las columnas de balance). Por defecto desactivado.
    - **Dónde falla (NaN estructural):** Financieras e Inmobiliarias (§7.4).
      Utilities: sesgo por amortización intensiva, se mantienen con
      neutralización. Empresas con adquisiciones grandes: usar la bandera.
    """

    requires: list[str] = ["fundamentals"]  # noqa: RUF012
    availability: Availability = "filing"
    exclude_sectors: frozenset[str] = DEFAULT_EXCLUDE_SECTORS

    def __init__(
        self,
        *,
        frequency: Frequency = "ttm",
        max_articulation_gap: float | None = None,
        max_staleness_days: int | None = None,
    ) -> None:
        if frequency not in ("ttm", "annual"):
            msg = f"frequency debe ser 'ttm' o 'annual'; recibido {frequency!r}"
            raise ValueError(msg)
        if max_articulation_gap is not None and max_articulation_gap <= 0:
            msg = f"max_articulation_gap debe ser > 0; recibido {max_articulation_gap}"
            raise ValueError(msg)
        if max_staleness_days is None:
            max_staleness_days = 550 if frequency == "annual" else DEFAULT_MAX_STALENESS_DAYS
        super().__init__(max_staleness_days=max_staleness_days)
        self.frequency: Frequency = frequency
        self.max_articulation_gap = max_articulation_gap
        self.name = "accruals_cf" if frequency == "ttm" else "accruals_cf_annual"

    def compute(self, ctx: FactorContextLike) -> pd.Series:
        required = ["net_income", "cfo", "total_assets"]
        if self.max_articulation_gap is not None:
            required = list(dict.fromkeys([*required, *_BALANCE_COLUMNS]))
        q, missing_opt = quarterly_history(
            ctx,
            required,
            optional=("taxes_payable",),
            availability=self.availability,
            factor=self.name,
        )
        if self.frequency == "annual":
            q = _restrict_annual(q, factor=self.name)
        acc_cf = _cashflow_accruals_quarterly(q)
        if self.max_articulation_gap is not None:
            acc_bs = _balance_accruals_quarterly(
                q, has_taxes_payable="taxes_payable" not in missing_opt, factor=self.name
            )
            gap = (acc_bs - acc_cf).abs()
            flagged = gap > self.max_articulation_gap
            if bool(flagged.any()):
                warnings.warn(
                    f"{self.name}: {int(flagged.sum())} observaciones invalidadas por "
                    f"discrepancia balance/flujos > {self.max_articulation_gap} "
                    "(Collins-Hribar 2002: señala M&A o divisa que contamina el accrual)",
                    UserWarning,
                    stacklevel=2,
                )
            acc_cf = acc_cf.where(~flagged)
        q["value"] = -acc_cf
        daily = self._daily_values(ctx, q, ["value"])
        return finalize_factor(daily["value"], name=self.name)


class PercentAccruals(FundamentalFactor):
    """Percent accruals (Hafzalla-Lundholm-Van Winkle 2011), factor = ``−PACC``.

    Referencia: Hafzalla, Lundholm y Van Winkle (2011), The Accounting Review
    86(1) — rentabilidades de cobertura significativamente mayores que la
    medida tradicional, con la mejora concentrada en el lado largo; no depende
    de partidas extraordinarias e identifica mal valorados igual de bien con
    pérdidas que con beneficios, donde el accrual escalado por activos falla—.
    Es el factor de accruals **primario** del informe (§7.3). Fórmula exacta en
    ``percent_accruals``.

    - **Signo (documentado):** devuelve ``−PACC``; mayor = menos devengo por
      unidad de beneficio = más alcista. En el generador sintético, cuyo drift
      responde a la sorpresa y no al devengo, este signo no es contrastable con
      IC; el test estructural compara contra la fórmula pura.
    - **Identidad (§8.1):** con ``NI > 0``, ``CFO/NI = 1 − PACC``. No combinar
      con ``quality.CFOToNetIncome`` a pesos independientes: son el mismo
      factor con dos escalas.
    - **PIT:** ``"filing"``.
    - **Requisitos:** ``net_income, cfo``; 4 trimestres consecutivos (TTM).
      ``NI_TTM == 0`` → NaN.
    - **Dónde falla (NaN estructural):** Financieras e Inmobiliarias (§7.4).
    """

    requires: list[str] = ["fundamentals"]  # noqa: RUF012
    availability: Availability = "filing"
    exclude_sectors: frozenset[str] = DEFAULT_EXCLUDE_SECTORS

    def __init__(
        self,
        *,
        frequency: Frequency = "ttm",
        max_staleness_days: int | None = None,
    ) -> None:
        if frequency not in ("ttm", "annual"):
            msg = f"frequency debe ser 'ttm' o 'annual'; recibido {frequency!r}"
            raise ValueError(msg)
        if max_staleness_days is None:
            max_staleness_days = 550 if frequency == "annual" else DEFAULT_MAX_STALENESS_DAYS
        super().__init__(max_staleness_days=max_staleness_days)
        self.frequency: Frequency = frequency
        self.name = "percent_accruals" if frequency == "ttm" else "percent_accruals_annual"

    def compute(self, ctx: FactorContextLike) -> pd.Series:
        q, _ = quarterly_history(
            ctx, ["net_income", "cfo"], availability=self.availability, factor=self.name
        )
        if self.frequency == "annual":
            q = _restrict_annual(q, factor=self.name)
        ni_ttm = ttm_sum(q, "net_income")
        cfo_ttm = ttm_sum(q, "cfo")
        denom = ni_ttm.abs().where(ni_ttm.abs() > 0)
        q["value"] = -((ni_ttm - cfo_ttm) / denom)
        daily = self._daily_values(ctx, q, ["value"])
        return finalize_factor(daily["value"], name=self.name)


class NetOperatingAccruals(FundamentalFactor):
    """Accruals operativos netos: factor = ``−ΔNOA / TA_{t−1}``.

    Referencias: Richardson, Sloan, Soliman y Tuna (2005), JAE 39(3) — la
    variación de los activos operativos netos generaliza el accrual de capital
    circulante de Sloan a los devengos no corrientes, que son los de menor
    fiabilidad y mayor mispricing—; Hirshleifer, Hou, Teoh y Zhang (2004), JAE
    38 — los balances "hinchados" (NOA altos) predicen rentabilidades bajas—.

    Construcción sobre las identidades de balance:

    .. code-block:: text

        NOA  = (TA − Caja) − (Pasivo total − Deuda total)
        ΔNOA = NOA_t − NOA_{t−4}          (interanual; anual con frequency="annual")
        factor = − ΔNOA / TA_{t−4}

    donde ``TA − Caja`` son los activos operativos y ``Pasivo total − Deuda``
    los pasivos operativos (la deuda es financiación, no operación).

    - **Signo (documentado):** invertido; mayor = balance más disciplinado =
      más alcista.
    - **PIT:** ``"filing"`` (todo es balance).
    - **Requisitos:** ``total_assets, cash, total_liabilities, total_debt`` (o
      ``short_term_debt`` + ``long_term_debt``); 5 trimestres consecutivos.
    - **Dónde falla (NaN estructural):** Financieras — la distinción
      operativo/financiero no existe en un banco — e Inmobiliarias. **Aviso de
      solapamiento:** ΔNOA = accruals totales ≈ crecimiento del activo operativo;
      correlaciona con ``growth.AssetGrowth`` por construcción.
    """

    requires: list[str] = ["fundamentals"]  # noqa: RUF012
    availability: Availability = "filing"
    exclude_sectors: frozenset[str] = DEFAULT_EXCLUDE_SECTORS

    def __init__(
        self,
        *,
        frequency: Frequency = "ttm",
        max_staleness_days: int | None = None,
    ) -> None:
        if frequency not in ("ttm", "annual"):
            msg = f"frequency debe ser 'ttm' o 'annual'; recibido {frequency!r}"
            raise ValueError(msg)
        if max_staleness_days is None:
            max_staleness_days = 550 if frequency == "annual" else DEFAULT_MAX_STALENESS_DAYS
        super().__init__(max_staleness_days=max_staleness_days)
        self.frequency: Frequency = frequency
        self.name = (
            "net_operating_accruals" if frequency == "ttm" else "net_operating_accruals_annual"
        )

    def compute(self, ctx: FactorContextLike) -> pd.Series:
        fund = getattr(ctx, "fundamentals", None)
        if not isinstance(fund, pd.DataFrame):
            msg = f"{self.name}: el contexto no trae `fundamentals` como DataFrame"
            raise DataQualityError(msg)
        cols = set(fund.columns)
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

        q, _ = quarterly_history(
            ctx,
            ["total_assets", "cash", "total_liabilities", *debt_inputs],
            availability=self.availability,
            factor=self.name,
        )
        if self.frequency == "annual":
            q = _restrict_annual(q, factor=self.name)
        debt = (
            q["total_debt"]
            if "total_debt" in q.columns
            else q["short_term_debt"] + q["long_term_debt"]
        ).astype("float64")
        ta = q["total_assets"].astype("float64")
        noa = (ta - q["cash"]) - (q["total_liabilities"] - debt)
        d_noa = noa - shift_q(q, noa, 4)
        q["value"] = -(d_noa / positive_only(shift_q(q, ta, 4)))
        daily = self._daily_values(ctx, q, ["value"])
        return finalize_factor(daily["value"], name=self.name)


FACTORS: tuple[type[FundamentalFactor], ...] = (
    SloanAccrualsBalance,
    SloanAccrualsCashFlow,
    PercentAccruals,
    NetOperatingAccruals,
)
"""Factores exportados por este módulo. Recomendación del informe (§7.3):
``PercentAccruals`` primario, ``SloanAccrualsCashFlow`` secundario y control,
``SloanAccrualsBalance`` solo como control y bandera de articulación."""


def _register() -> None:
    """Registra los factores en el `default_registry` de `factors.base`.

    Import en tiempo de ejecución con try/except: `base.py` lo escribe otro
    agente en paralelo y este módulo debe funcionar también sin él. Se registran
    con el nombre de la variante TTM por defecto; la anual se obtiene pasando
    ``frequency="annual"`` a la fábrica.
    """
    try:
        from earnings_alpha.factors.base import register_factor
    except ImportError:  # pragma: no cover - base.py aún no integrado
        return

    @register_factor("percent_accruals")
    def _percent(**kwargs: object) -> PercentAccruals:
        return PercentAccruals(**kwargs)  # type: ignore[arg-type]

    @register_factor("accruals_cf")
    def _cashflow(**kwargs: object) -> SloanAccrualsCashFlow:
        return SloanAccrualsCashFlow(**kwargs)  # type: ignore[arg-type]

    @register_factor("accruals_bs")
    def _balance(**kwargs: object) -> SloanAccrualsBalance:
        return SloanAccrualsBalance(**kwargs)  # type: ignore[arg-type]

    @register_factor("net_operating_accruals")
    def _noa(**kwargs: object) -> NetOperatingAccruals:
        return NetOperatingAccruals(**kwargs)  # type: ignore[arg-type]


_register()
