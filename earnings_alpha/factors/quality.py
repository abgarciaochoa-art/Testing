"""Factores de calidad (*quality*) sobre fundamentales point-in-time.

Implementa las §§6, 8, 9 y 12.1 de ``docs/research/fundamental_factors.md``:

- ``PiotroskiFScore``   — F-Score con los **9 componentes exactos** de
  Piotroski (2000), cada uno verificable por separado, en variante binaria
  {0..9} y continua (rank-percentil promedio, la recomendada por el informe).
- ``CFOToNetIncome``    — calidad de beneficios ``CFO/NI`` (diagnóstico).
- ``GrossProfitability``— GPA de Novy-Marx (2013).
- ``MarginTrend``       — tendencia de margen por t de la pendiente OLS sobre
  TTM (la variante que el informe verifica numéricamente en §9.2).
- ``Profitability``     — nivel de ROA / ROE TTM.
- ``EarningsStability`` — estabilidad del beneficio (−sd del ROA trimestral).
- ``Leverage``          — apalancamiento con el signo invertido
  (``−NetDebt/EBITDA``).

Todos devuelven ``pd.Series`` con MultiIndex ``(date, ticker)`` y la convención
del contrato: **valor mayor = más alcista**. Los factores cuya lectura natural
es "menos es mejor" (apalancamiento, volatilidad del beneficio) se devuelven
con el signo cambiado y lo documentan.

Disponibilidad point-in-time: los factores que necesitan balance o estado de
flujos (F-Score, CFO/NI, GPA, ROA/ROE, estabilidad, apalancamiento) se fechan
por ``filed_at`` (aceptación del 10-Q/10-K); la tendencia de margen, que solo
usa cuenta de resultados, por la nota de prensa (``available_at``). Véase
``fundamental_factors.md`` §1.2: en el S&P 500 el desfase anuncio → filing
ronda las 4-6 semanas y confundirlos es look-ahead.

Referencias
-----------
- Piotroski, J. D. (2000). *Value Investing: The Use of Historical Financial
  Statement Information to Separate Winners from Losers*. JAR 38 (Supl.), 1-41.
- Novy-Marx, R. (2013). *The Other Side of Value: The Gross Profitability
  Premium*. JFE 108(1), 1-28.
- Asness, C. S., Frazzini, A. y Pedersen, L. H. (2019). *Quality Minus Junk*.
  Review of Accounting Studies 24(1), 34-112.
- Campbell, J. Y., Hilscher, J. y Szilagyi, J. (2008). *In Search of Distress
  Risk*. JF 63(6), 2899-2939.
- Lev, B. y Thiagarajan, S. R. (1993). *Fundamental Information Analysis*.
  JAR 31(2), 190-215.
"""

from __future__ import annotations

import math
import warnings
from typing import Literal

import numpy as np
import pandas as pd

from earnings_alpha.errors import DataQualityError, InsufficientHistory
from earnings_alpha.factors.value import (
    DEFAULT_EXCLUDE_SECTORS,
    DEFAULT_MAX_STALENESS_DAYS,
    Availability,
    FactorContextLike,
    FundamentalFactor,
    apply_sector_exclusion,
    finalize_factor,
    grouped_rolling,
    positive_only,
    quarterly_history,
    shift_q,
    to_daily,
    ttm_sum,
)

__all__ = [  # noqa: RUF022 - orden temático
    "PiotroskiFScore",
    "CFOToNetIncome",
    "GrossProfitability",
    "MarginTrend",
    "Profitability",
    "EarningsStability",
    "Leverage",
    "FACTORS",
    "PIOTROSKI_COMPONENTS",
    "piotroski_components_annual",
    "trend_t_stat",
]

PIOTROSKI_COMPONENTS: tuple[str, ...] = (
    "f_roa",
    "f_cfo",
    "f_droa",
    "f_accrual",
    "f_dlever",
    "f_dliquid",
    "f_eq_offer",
    "f_dmargin",
    "f_dturn",
)
"""Los nueve indicadores canónicos de Piotroski (2000), en el orden del artículo:
rentabilidad (4), apalancamiento/liquidez/origen de fondos (3) y eficiencia (2)."""

_TREND_T_CAP = 100.0
"""Tope del estadístico t con ajuste perfecto (varianza residual nula)."""

_DEFAULT_ISSUANCE_TOLERANCE = 0.0025
"""Emisión neta admisible como fracción del activo medio antes de perder el punto
``EQ_OFFER``. Es un umbral de ingeniería [prior]: absorbe el goteo de la
retribución en acciones sin dejar pasar una ampliación de capital genuina."""


# ---------------------------------------------------------------------------
# Utilidades puras (verificables con números de juguete)
# ---------------------------------------------------------------------------


def _num(x: object) -> float:
    """Convierte a float tratando None como NaN (entrada tolerante de las puras)."""
    if x is None:
        return math.nan
    return float(x)  # type: ignore[arg-type]


def _flag(condition: bool, *inputs: float) -> float:
    """1.0/0.0 según ``condition``; NaN si algún insumo es NaN."""
    if any(math.isnan(v) for v in inputs):
        return math.nan
    return 1.0 if condition else 0.0


def piotroski_components_annual(
    *,
    net_income: float,
    cfo: float,
    revenue: float,
    cost_of_goods: float,
    current_assets: float,
    current_liabilities: float,
    long_term_debt: float,
    equity_issuance: float,
    total_assets: float,
    total_assets_prev: float,
    total_assets_prev2: float,
    net_income_prev: float,
    revenue_prev: float,
    cost_of_goods_prev: float,
    current_assets_prev: float,
    current_liabilities_prev: float,
    long_term_debt_prev: float,
    issuance_tolerance: float = 0.0,
) -> dict[str, float]:
    """Los 9 componentes del F-Score para **un** ejercicio, con la fórmula exacta.

    Piotroski (2000), tabla 1. Convenciones que el informe (§6.1) marca como
    trampas habituales y que aquí se respetan al pie de la letra:

    - ``ROA_t = NI_t / TA_{t-1}``: deflactado por el activo **al inicio del
      ejercicio**, no por el medio ni el de cierre. Análogamente la rotación.
    - ``LEV_t = DeudaLP_t / ((TA_t + TA_{t-1})/2)`` y el punto se otorga si el
      apalancamiento **baja** (``LEV_t < LEV_{t-1}``); nótese que ``LEV_{t-1}``
      necesita ``TA_{t-2}``, de ahí el parámetro ``total_assets_prev2``.
    - ``EQ_OFFER``: el punto es para la empresa que **no emitió** acciones
      ordinarias. Varias fuentes secundarias lo invierten; el razonamiento de
      Piotroski es explícito: captar capital externo con la cotización deprimida
      señala incapacidad de generar fondos internos.

    Cada componente vale 1.0, 0.0 o NaN (insumo ausente o denominador
    estructuralmente inválido: ``TA <= 0``, ``CL <= 0``, ``Ventas <= 0``).
    ``f_score`` es la suma de los nueve, o NaN si alguno lo es: un F-Score de 7
    calculado sobre 7 componentes no es comparable con un 7 sobre 9.

    Parámetros
    ----------
    issuance_tolerance:
        Emisión neta máxima (en fracción del activo medio del ejercicio) que no
        pierde el punto ``EQ_OFFER``. Por defecto 0: cualquier emisión positiva
        pierde el punto, que es la lectura literal del artículo.
    """
    ni = _num(net_income)
    cfo_ = _num(cfo)
    rev = _num(revenue)
    cogs = _num(cost_of_goods)
    ca = _num(current_assets)
    cl = _num(current_liabilities)
    ltd = _num(long_term_debt)
    iss = _num(equity_issuance)
    ta = _num(total_assets)
    ta1 = _num(total_assets_prev)
    ta2 = _num(total_assets_prev2)
    ni1 = _num(net_income_prev)
    rev1 = _num(revenue_prev)
    cogs1 = _num(cost_of_goods_prev)
    ca1 = _num(current_assets_prev)
    cl1 = _num(current_liabilities_prev)
    ltd1 = _num(long_term_debt_prev)

    ta0 = ta1 if not math.isnan(ta1) and ta1 > 0 else math.nan  # TA inicio ejercicio t
    ta0_prev = ta2 if not math.isnan(ta2) and ta2 > 0 else math.nan  # TA inicio ejercicio t-1

    roa = ni / ta0 if not math.isnan(ta0) else math.nan
    cfo_ta = cfo_ / ta0 if not math.isnan(ta0) else math.nan
    roa_prev = ni1 / ta0_prev if not math.isnan(ta0_prev) else math.nan
    droa = roa - roa_prev

    avg_ta_t = (ta + ta1) / 2.0
    avg_ta_prev = (ta1 + ta2) / 2.0
    lev = ltd / avg_ta_t if avg_ta_t > 0 else math.nan
    lev_prev = ltd1 / avg_ta_prev if avg_ta_prev > 0 else math.nan
    dlever = lev - lev_prev

    cr = ca / cl if cl > 0 else math.nan
    cr_prev = ca1 / cl1 if cl1 > 0 else math.nan
    dliquid = cr - cr_prev

    gm = (rev - cogs) / rev if rev > 0 else math.nan
    gm_prev = (rev1 - cogs1) / rev1 if rev1 > 0 else math.nan
    dmargin = gm - gm_prev

    at = rev / ta0 if not math.isnan(ta0) else math.nan
    at_prev = rev1 / ta0_prev if not math.isnan(ta0_prev) else math.nan
    dturn = at - at_prev

    iss_rel = iss / avg_ta_t if avg_ta_t > 0 else math.nan

    out = {
        "f_roa": _flag(roa > 0, roa),
        "f_cfo": _flag(cfo_ta > 0, cfo_ta),
        "f_droa": _flag(droa > 0, droa),
        "f_accrual": _flag(cfo_ta > roa, cfo_ta, roa),
        "f_dlever": _flag(dlever < 0, dlever),
        "f_dliquid": _flag(dliquid > 0, dliquid),
        "f_eq_offer": _flag(iss_rel <= issuance_tolerance, iss_rel),
        "f_dmargin": _flag(dmargin > 0, dmargin),
        "f_dturn": _flag(dturn > 0, dturn),
    }
    values = [out[k] for k in PIOTROSKI_COMPONENTS]
    out["f_score"] = math.nan if any(math.isnan(v) for v in values) else float(sum(values))
    return out


def trend_t_stat(values: object, window: int = 8) -> float:
    """t de la pendiente OLS de las últimas ``window`` observaciones.

    Es el estadístico de tendencia de margen recomendado por el informe (§9.2):
    frente a la diferencia simple extremo a extremo, normalizar por el error
    estándar penaliza el ruido automáticamente (verificado numéricamente allí:
    con idéntica pendiente real, la serie limpia obtiene t ≈ 35 y la ruidosa
    t ≈ 4.6, mientras la diferencia simple ordena al revés).

    - Menos de ``window`` observaciones → ``InsufficientHistory``.
    - NaN dentro de la ventana → NaN.
    - Ajuste perfecto (varianza residual nula): serie constante → 0.0; recta
      exacta con pendiente no nula → ``±_TREND_T_CAP`` (t infinito acotado,
      documentado; el winsorizado cross-section posterior lo absorbe).
    """
    y = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    if window < 4:
        msg = f"window debe ser >= 4 para que la t tenga grados de libertad; recibido {window}"
        raise ValueError(msg)
    if len(y) < window:
        msg = (
            f"trend_t_stat necesita al menos {window} observaciones TTM y hay {len(y)} "
            "(fundamental_factors.md §9.2)"
        )
        raise InsufficientHistory(msg)
    w = y[-window:]
    if np.isnan(w).any():
        return math.nan
    x = np.arange(window, dtype=float) - (window - 1) / 2.0
    sxx = float((x * x).sum())
    beta = float((x * w).sum()) / sxx
    resid = w - w.mean() - beta * x
    s2 = float((resid * resid).sum()) / (window - 2)
    if s2 <= 1e-24:
        return 0.0 if beta == 0.0 else math.copysign(_TREND_T_CAP, beta)
    se = math.sqrt(s2 / sxx)
    return beta / se


def _binary_signal(x: pd.Series, op: Literal["gt", "lt", "le"], threshold: float = 0.0) -> pd.Series:
    """Indicador 1/0 vectorizado con NaN propagado (versión panel de ``_flag``)."""
    valid = x.notna()
    if op == "gt":
        hit = x > threshold
    elif op == "lt":
        hit = x < threshold
    else:
        hit = x <= threshold
    return pd.Series(
        np.where(valid.to_numpy(), hit.to_numpy(dtype=float), np.nan), index=x.index
    )


# ---------------------------------------------------------------------------
# Piotroski F-Score
# ---------------------------------------------------------------------------


class PiotroskiFScore(FundamentalFactor):
    """F-Score de Piotroski (2000) con los 9 componentes exactos.

    Referencia: Piotroski (2000), JAR 38 (Supl.), 1-41. La lista canónica es
    ``{ROA, CFO, dROA, ACCRUAL, dLEVER, dLIQUID, EQ_OFFER, dMARGIN, dTURN}``
    (§6.1 del informe, que documenta las dos corrupciones que circulan: omitir
    ``dROA`` y —peor— invertir el signo de ``EQ_OFFER``).

    Variantes (§6.3 del informe):

    - ``variant="binary"``: el F entero ∈ {0..9}. Sobre 503 nombres produce
      empates masivos (verificado por simulación en el informe: ~146 nombres en
      la moda); útil como filtro, pobre como ranking.
    - ``variant="continuous"`` (por defecto, la recomendada): cada indicador
      binario se sustituye por el rank-percentil cross-section de su magnitud
      subyacente (``ROA``, ``dROA``, ``CFO/TA``, ``CFO/TA − ROA``, ``−dLEV``,
      ``dCR``, ``−emisión_neta/TA``, ``dGM``, ``dAT``) y se promedian los nueve.
      Conserva la lógica de Piotroski y recupera toda la granularidad.

    Frecuencia (§6.4): ``frequency="ttm"`` (por defecto) refresca cada
    trimestre con ventanas móviles de 4 trimestres; ``"annual"`` puntúa solo al
    cierre de cada ejercicio fiscal (filas Q4 / 10-K), que es la formulación
    literal del artículo. En ambos casos la comparación es interanual.

    - **Signo:** mayor F = fundamentales más sólidos = más alcista.
    - **PIT:** ``"filing"`` — ``EQ_OFFER`` y el CFO exigen el estado de flujos,
      que no viene en el 8-K (§6.4).
    - **Historia mínima:** 9 trimestres consecutivos (los deltas interanuales de
      ratios que se deflactan por el activo inicial necesitan ``TA_{q-8}``).
      Nombres con menos → NaN; panel entero sin historia → ``InsufficientHistory``.
    - **Requisitos:** ``net_income, cfo, total_assets, current_assets,
      current_liabilities, long_term_debt, revenue, cost_of_revenue``. Para
      ``EQ_OFFER``: ``equity_issuance`` o, en su defecto, el proxy documentado
      ``Δpaid_in_capital − stock_compensation + buybacks``; sin ninguno de los
      dos, el componente queda NaN (y el F estricto también) con aviso.
    - **Dónde falla (NaN estructural, §6.4):** Financieras e Inmobiliarias —
      ratio corriente sin sentido en un banco, margen bruto inexistente,
      rotación no comparable: 4 de los 9 componentes se caen—. Utilities:
      ``dLEVER`` premia mecánicamente la fase del ciclo regulatorio; se
      mantienen pero exigen neutralización sectorial.
    - **Aviso de solapamiento (§8.2.4):** ``F_ACCRUAL`` es el signo de
      ``−accruals``; combinar F-Score con el factor de accruals duplica esa
      exposición parcialmente.
    - La letra pequeña de Piotroski: el efecto original se concentra en value
      pequeño, ilíquido y poco cubierto — lo contrario del S&P 500 (§6.2). El
      IC esperable aquí es pequeño (tabla §14 del informe).
    """

    requires: list[str] = ["fundamentals"]  # noqa: RUF012
    availability: Availability = "filing"
    exclude_sectors: frozenset[str] = DEFAULT_EXCLUDE_SECTORS

    _MAGNITUDES: tuple[str, ...] = (
        "m_roa",
        "m_cfo",
        "m_droa",
        "m_accrual",
        "m_dlever",
        "m_dliquid",
        "m_eq_offer",
        "m_dmargin",
        "m_dturn",
    )

    def __init__(
        self,
        *,
        variant: Literal["continuous", "binary"] = "continuous",
        frequency: Literal["ttm", "annual"] = "ttm",
        issuance_tolerance: float = _DEFAULT_ISSUANCE_TOLERANCE,
        max_staleness_days: int | None = None,
    ) -> None:
        if variant not in ("continuous", "binary"):
            msg = f"variant debe ser 'continuous' o 'binary'; recibido {variant!r}"
            raise ValueError(msg)
        if frequency not in ("ttm", "annual"):
            msg = f"frequency debe ser 'ttm' o 'annual'; recibido {frequency!r}"
            raise ValueError(msg)
        if max_staleness_days is None:
            # Un score anual sigue vigente hasta el siguiente 10-K (~1 año + retraso).
            max_staleness_days = 550 if frequency == "annual" else DEFAULT_MAX_STALENESS_DAYS
        super().__init__(max_staleness_days=max_staleness_days)
        self.variant = variant
        self.frequency = frequency
        self.issuance_tolerance = float(issuance_tolerance)
        self.name = "piotroski_f" if variant == "binary" else "piotroski_f_cont"

    # ------------------------------------------------------------------ núcleo

    def _issuance_ttm(self, q: pd.DataFrame) -> pd.Series | None:
        """Emisión neta de acciones ordinarias del año móvil, o None si no hay datos.

        Preferencia: columna directa ``equity_issuance``. Proxy documentado en su
        defecto: ``Δpaid_in_capital − stock_compensation + buybacks`` por
        trimestre (la variación del capital desembolsado que no explica ni la
        retribución en acciones ni las recompras es, por eliminación, emisión).
        """
        if "equity_issuance" in q.columns:
            return ttm_sum(q, "equity_issuance")
        needed = {"paid_in_capital", "stock_compensation", "buybacks"}
        if needed <= set(q.columns):
            warnings.warn(
                f"{self.name}: sin `equity_issuance`; EQ_OFFER se infiere con el proxy "
                "Δpaid_in_capital − stock_compensation + buybacks (degradación documentada)",
                UserWarning,
                stacklevel=3,
            )
            issuance_q = (
                q["paid_in_capital"]
                - shift_q(q, "paid_in_capital", 1)
                - q["stock_compensation"]
                + q["buybacks"]
            )
            return ttm_sum(q, issuance_q)
        warnings.warn(
            f"{self.name}: sin `equity_issuance` ni el trío (`paid_in_capital`, "
            "`stock_compensation`, `buybacks`); el componente EQ_OFFER queda en NaN "
            "y el F-Score estricto también (degradación documentada)",
            UserWarning,
            stacklevel=3,
        )
        return None

    def _quarterly_magnitudes(self, ctx: FactorContextLike) -> pd.DataFrame:
        """Panel trimestral con las 9 magnitudes continuas subyacentes al F-Score."""
        q, _ = quarterly_history(
            ctx,
            [
                "net_income",
                "cfo",
                "total_assets",
                "current_assets",
                "current_liabilities",
                "long_term_debt",
                "revenue",
                "cost_of_revenue",
            ],
            optional=("equity_issuance", "paid_in_capital", "stock_compensation", "buybacks"),
            availability=self.availability,
            factor=self.name,
        )
        ni_ttm = ttm_sum(q, "net_income")
        cfo_ttm = ttm_sum(q, "cfo")
        rev_ttm = ttm_sum(q, "revenue")
        cogs_ttm = ttm_sum(q, "cost_of_revenue")

        ta = q["total_assets"].astype("float64")
        ta4 = shift_q(q, ta, 4)
        ta8 = shift_q(q, ta, 8)
        ta0 = positive_only(ta4)  # activo al inicio del año móvil
        ta0_prev = positive_only(ta8)

        roa = ni_ttm / ta0
        cfo_ta = cfo_ttm / ta0
        roa_prev = shift_q(q, ni_ttm, 4) / ta0_prev

        avg_ta = positive_only((ta + ta4) / 2.0)
        avg_ta_prev = positive_only((ta4 + ta8) / 2.0)
        lev = q["long_term_debt"] / avg_ta
        lev_prev = shift_q(q, "long_term_debt", 4) / avg_ta_prev

        cr = q["current_assets"] / positive_only(q["current_liabilities"].astype("float64"))
        gm = (rev_ttm - cogs_ttm) / positive_only(rev_ttm)
        at = rev_ttm / ta0
        at_prev = shift_q(q, rev_ttm, 4) / ta0_prev

        issuance = self._issuance_ttm(q)
        iss_rel = issuance / avg_ta if issuance is not None else pd.Series(np.nan, index=q.index)

        q["m_roa"] = roa
        q["m_cfo"] = cfo_ta
        q["m_droa"] = roa - roa_prev
        q["m_accrual"] = cfo_ta - roa
        q["m_dlever"] = -(lev - lev_prev)  # mayor = desapalanca = mejor
        q["m_dliquid"] = cr - shift_q(q, cr, 4)
        q["m_eq_offer"] = -iss_rel  # mayor = no emite (o recompra) = mejor
        q["m_dmargin"] = gm - shift_q(q, gm, 4)
        q["m_dturn"] = at - at_prev

        if self.frequency == "annual":
            if "fiscal_period" in q.columns:
                annual = q["fiscal_period"].astype(str).str.endswith("Q4")
            elif "form" in q.columns:
                annual = q["form"].astype(str).eq("10-K")
            else:
                msg = (
                    f"{self.name}: frequency='annual' necesita `fiscal_period` o `form` "
                    "para identificar el cierre de ejercicio"
                )
                raise DataQualityError(msg)
            q = q[annual.to_numpy()].reset_index(drop=True)
            if len(q) == 0:
                msg = f"{self.name}: no hay cierres de ejercicio fiscal en `fundamentals`"
                raise InsufficientHistory(msg)
        return q

    def _binary_components(self, q: pd.DataFrame) -> pd.DataFrame:
        """Traduce magnitudes a los 9 indicadores binarios exactos del artículo."""
        comp = pd.DataFrame(index=q.index)
        comp["f_roa"] = _binary_signal(q["m_roa"], "gt")
        comp["f_cfo"] = _binary_signal(q["m_cfo"], "gt")
        comp["f_droa"] = _binary_signal(q["m_droa"], "gt")
        comp["f_accrual"] = _binary_signal(q["m_accrual"], "gt")
        # m_dlever = −ΔLEV: el punto exige ΔLEV < 0, es decir, −ΔLEV > 0.
        comp["f_dlever"] = _binary_signal(q["m_dlever"], "gt")
        comp["f_dliquid"] = _binary_signal(q["m_dliquid"], "gt")
        # m_eq_offer = −emisión/TA: el punto exige emisión <= tolerancia.
        comp["f_eq_offer"] = _binary_signal(-q["m_eq_offer"], "le", self.issuance_tolerance)
        comp["f_dmargin"] = _binary_signal(q["m_dmargin"], "gt")
        comp["f_dturn"] = _binary_signal(q["m_dturn"], "gt")
        f = comp[list(PIOTROSKI_COMPONENTS)].sum(axis=1)
        f[comp[list(PIOTROSKI_COMPONENTS)].isna().any(axis=1)] = np.nan
        comp["f_score"] = f
        return comp

    # ------------------------------------------------------------------ API

    def compute_components(self, ctx: FactorContextLike) -> pd.DataFrame:
        """Panel diario ``(date, ticker)`` con los 9 componentes binarios y ``f_score``.

        Es la vía para verificar cada componente por separado (tests) y para
        usar el F como filtro (mitigación 2 de §6.3 del informe).
        """
        q = self._quarterly_magnitudes(ctx)
        comps = self._binary_components(q)
        cols = [*PIOTROSKI_COMPONENTS, "f_score"]
        q = pd.concat([q, comps], axis=1)
        q = apply_sector_exclusion(q, cols, ctx, self.exclude_sectors, factor=self.name)
        return to_daily(ctx, q, cols, factor=self.name, max_staleness_days=self.max_staleness_days)

    def compute(self, ctx: FactorContextLike) -> pd.Series:
        if self.variant == "binary":
            daily = self.compute_components(ctx)
            return finalize_factor(daily["f_score"], name=self.name)

        # Variante continua: rank-percentil cross-section de cada magnitud.
        from earnings_alpha.signals import rank_pct  # import local: evita ciclos

        q = self._quarterly_magnitudes(ctx)
        cols = list(self._MAGNITUDES)
        q = apply_sector_exclusion(q, cols, ctx, self.exclude_sectors, factor=self.name)
        daily = to_daily(ctx, q, cols, factor=self.name, max_staleness_days=self.max_staleness_days)
        ranks = pd.DataFrame(
            {c: rank_pct(daily[c], mode="uniform") for c in cols}, index=daily.index
        )
        score = ranks.mean(axis=1)
        score[ranks.isna().any(axis=1)] = np.nan
        return finalize_factor(score, name=self.name)


# ---------------------------------------------------------------------------
# Calidad de beneficios y rentabilidad
# ---------------------------------------------------------------------------


class CFOToNetIncome(FundamentalFactor):
    """Calidad de beneficios ``CFO/NI`` (TTM) — **diagnóstico**, no factor primario.

    Referencia: la identidad de ``fundamental_factors.md`` §8.1, verificada
    algebraica y numéricamente allí: con ``NI > 0``,
    ``CFO/NI = 1 − PACC`` — es una transformación monótona de los *percent
    accruals* de Hafzalla, Lundholm y Van Winkle (2011). **No son dos factores**:
    combinarlo con ``accruals.PercentAccruals`` con pesos independientes duplica
    la exposición y subestima el riesgo (§8.2).

    Con ``NI <= 0`` el ratio invierte el signo (una empresa con pérdidas que
    genera caja saldría negativa, es decir, "mala") y por eso aquí es **NaN**
    en ese dominio, siguiendo la recomendación 2 de §8.2: se reporta el ratio
    interpretable restringido a ``NI > 0`` y se deja la ordenación completa a
    ``−PACC``.

    - **Signo:** mayor conversión a caja = más alcista.
    - **PIT:** ``"filing"`` (necesita el estado de flujos).
    - **Dónde falla (NaN estructural):** Financieras e Inmobiliarias, como todos
      los derivados de accruals (§8.3).
    """

    name = "cfo_to_ni"
    requires: list[str] = ["fundamentals"]  # noqa: RUF012
    availability: Availability = "filing"
    exclude_sectors: frozenset[str] = DEFAULT_EXCLUDE_SECTORS

    def compute(self, ctx: FactorContextLike) -> pd.Series:
        q, _ = quarterly_history(
            ctx, ["net_income", "cfo"], availability=self.availability, factor=self.name
        )
        ni_ttm = ttm_sum(q, "net_income")
        cfo_ttm = ttm_sum(q, "cfo")
        q["value"] = cfo_ttm / positive_only(ni_ttm)
        daily = self._daily_values(ctx, q, ["value"])
        return finalize_factor(daily["value"], name=self.name)


class GrossProfitability(FundamentalFactor):
    """Gross profitability (GPA): ``(Ventas − COGS) TTM / activo total``.

    Referencia: Novy-Marx (2013), JFE 108(1), 1-28. El beneficio **bruto**
    escalado por **activo total** tiene aproximadamente el mismo poder
    predictivo cross-section que el book-to-market, y controlar por
    rentabilidad mejora el rendimiento de las estrategias value *especialmente
    entre los valores más grandes y líquidos* — de los pocos hallazgos del
    informe que no se degradan en nuestro universo (§9.1).

    El numerador es deliberadamente el beneficio bruto: es la línea de la
    cuenta de resultados menos contaminada por contabilidad discrecional;
    cuanto más abajo se baja, más "limpia" parece la cifra y más ruido
    discrecional contiene (Novy-Marx 2013, §9.1 del informe).

    - **Signo:** mayor = más rentable = más alcista.
    - **PIT:** ``"filing"`` (el denominador es balance).
    - **Requisitos:** ``revenue, cost_of_revenue, total_assets``; 4 trimestres.
    - **Dónde falla (NaN estructural):** Financieras (sin COGS) e
      Inmobiliarias; 107 nombres fuera (§9.4). El nivel de margen es casi puro
      sector: neutralización sectorial obligatoria aguas abajo.
    """

    name = "gross_profitability"
    requires: list[str] = ["fundamentals"]  # noqa: RUF012
    availability: Availability = "filing"
    exclude_sectors: frozenset[str] = DEFAULT_EXCLUDE_SECTORS

    def compute(self, ctx: FactorContextLike) -> pd.Series:
        q, _ = quarterly_history(
            ctx,
            ["revenue", "cost_of_revenue", "total_assets"],
            availability=self.availability,
            factor=self.name,
        )
        gp_ttm = ttm_sum(q, "revenue") - ttm_sum(q, "cost_of_revenue")
        q["value"] = gp_ttm / positive_only(q["total_assets"].astype("float64"))
        daily = self._daily_values(ctx, q, ["value"])
        return finalize_factor(daily["value"], name=self.name)


class MarginTrend(FundamentalFactor):
    """Tendencia de margen: t de la pendiente OLS del margen TTM.

    Referencia: ``fundamental_factors.md`` §9.2, variante (b) — recomendada y
    verificada numéricamente allí: la diferencia simple extremo a extremo
    califica *mejor* a la serie ruidosa (exactamente al revés de lo deseable),
    mientras que el t de la pendiente penaliza el ruido automáticamente. El
    contenido económico entronca con las señales de margen de Lev y Thiagarajan
    (1993) y Abarbanell y Bushee (1998).

    - ``margin="gross"`` (por defecto): ``GM = (Ventas − COGS)/Ventas``.
    - ``margin="operating"``: ``OM = EBIT/Ventas`` (requiere
      ``operating_income``).

    - **Signo:** margen mejorando = más alcista.
    - **PIT:** ``"announcement"`` — solo usa cuenta de resultados, que sí viene
      en la nota de prensa.
    - **Historia mínima:** ``window`` observaciones TTM consecutivas
      (``window=8`` por defecto → 11 trimestres brutos, §9.2 y §15.3
      ``MIN_TTM_TREND``). Nombres con menos → NaN.
    - **Dónde falla (NaN estructural):** Financieras e Inmobiliarias (sin COGS).
      Semis y materiales: la tendencia mide fase de su ciclo propio de 2-4
      años; neutralizar a nivel sub-industria donde haya nombres (§9.4).
    """

    requires: list[str] = ["fundamentals"]  # noqa: RUF012
    availability: Availability = "announcement"
    exclude_sectors: frozenset[str] = DEFAULT_EXCLUDE_SECTORS

    def __init__(
        self,
        *,
        margin: Literal["gross", "operating"] = "gross",
        window: int = 8,
        max_staleness_days: int | None = DEFAULT_MAX_STALENESS_DAYS,
    ) -> None:
        if margin not in ("gross", "operating"):
            msg = f"margin debe ser 'gross' u 'operating'; recibido {margin!r}"
            raise ValueError(msg)
        if window < 4:
            msg = f"window debe ser >= 4; recibido {window}"
            raise ValueError(msg)
        super().__init__(max_staleness_days=max_staleness_days)
        self.margin = margin
        self.window = int(window)
        self.name = f"margin_trend_{margin}"

    def compute(self, ctx: FactorContextLike) -> pd.Series:
        if self.margin == "gross":
            cols = ["revenue", "cost_of_revenue"]
        else:
            cols = ["revenue", "operating_income"]
        q, _ = quarterly_history(ctx, cols, availability=self.availability, factor=self.name)
        rev_ttm = ttm_sum(q, "revenue")
        if self.margin == "gross":
            num_ttm = rev_ttm - ttm_sum(q, "cost_of_revenue")
        else:
            num_ttm = ttm_sum(q, "operating_income")
        margin_ttm = (num_ttm / positive_only(rev_ttm)).to_numpy(dtype=float)

        out = np.full(len(q), np.nan)
        for positions in q.groupby(["ticker", "__run__"], sort=False).indices.values():
            pos = np.sort(np.asarray(positions))
            series = margin_ttm[pos]
            for i in range(self.window - 1, len(pos)):
                window_vals = series[i - self.window + 1 : i + 1]
                if np.isnan(window_vals).any():
                    continue
                out[pos[i]] = trend_t_stat(window_vals, self.window)
        q["value"] = out
        daily = self._daily_values(ctx, q, ["value"])
        return finalize_factor(daily["value"], name=self.name)


class Profitability(FundamentalFactor):
    """Nivel de rentabilidad TTM: ROA (``NI/activo medio``) o ROE (``NI/patrimonio medio``).

    Referencias: Haugen y Baker (1996) y Fama y French (2006) documentan la
    rentabilidad esperada como predictor positivo; Asness, Frazzini y Pedersen
    (2019), *Quality Minus Junk*, RAST 24(1), la integran como pilar de calidad.

    - **Signo:** más rentable = más alcista.
    - **PIT:** ``"filing"`` (el denominador es balance).
    - **Requisitos:** ``net_income`` + ``total_assets`` (ROA) o ``total_equity``
      (ROE); 8 trimestres para el promedio interanual del denominador.
    - **Sectores:** definido también para Financieras (es el sustituto que el
      informe recomienda allí donde caen los factores de margen, §6.4); no se
      excluye ninguno, pero el nivel es fuertemente sectorial: neutralizar.
    - ROE con patrimonio medio no positivo → NaN (no un ROE "infinito").
    """

    requires: list[str] = ["fundamentals"]  # noqa: RUF012
    availability: Availability = "filing"

    def __init__(
        self,
        metric: Literal["roa", "roe"] = "roa",
        *,
        max_staleness_days: int | None = DEFAULT_MAX_STALENESS_DAYS,
    ) -> None:
        if metric not in ("roa", "roe"):
            msg = f"metric debe ser 'roa' o 'roe'; recibido {metric!r}"
            raise ValueError(msg)
        super().__init__(max_staleness_days=max_staleness_days)
        self.metric = metric
        self.name = metric

    def compute(self, ctx: FactorContextLike) -> pd.Series:
        denom_col = "total_assets" if self.metric == "roa" else "total_equity"
        q, _ = quarterly_history(
            ctx, ["net_income", denom_col], availability=self.availability, factor=self.name
        )
        ni_ttm = ttm_sum(q, "net_income")
        denom = q[denom_col].astype("float64")
        avg_denom = positive_only((denom + shift_q(q, denom, 4)) / 2.0)
        q["value"] = ni_ttm / avg_denom
        daily = self._daily_values(ctx, q, ["value"])
        return finalize_factor(daily["value"], name=self.name)


class EarningsStability(FundamentalFactor):
    """Estabilidad del beneficio: ``−sd(ROA trimestral)`` sobre una ventana móvil.

    Referencia: Asness, Frazzini y Pedersen (2019), *Quality Minus Junk*, RAST
    24(1) — la baja volatilidad de la rentabilidad contable forma parte del
    pilar de seguridad de la calidad. El informe (§8.2.3) la señala como la
    dimensión de calidad que mejor **complementa** a los accruals, con la que no
    es redundante [prior].

    - **Signo:** el signo se invierte — beneficio más estable = más alcista —
      y queda documentado aquí como exige el contrato.
    - **PIT:** ``"filing"`` (el denominador del ROA es balance).
    - **Historia mínima:** ``window`` ROAs trimestrales consecutivos (por
      defecto 8 → 9 trimestres brutos). Nombres con menos → NaN.
    - **Sectores:** definido para todos; el nivel de volatilidad es sectorial
      (energía vs staples), neutralizar aguas abajo.
    """

    name = "earnings_stability"
    requires: list[str] = ["fundamentals"]  # noqa: RUF012
    availability: Availability = "filing"

    def __init__(
        self,
        *,
        window: int = 8,
        max_staleness_days: int | None = DEFAULT_MAX_STALENESS_DAYS,
    ) -> None:
        if window < 4:
            msg = f"window debe ser >= 4 para una desviación típica con sentido; recibido {window}"
            raise ValueError(msg)
        super().__init__(max_staleness_days=max_staleness_days)
        self.window = int(window)

    def compute(self, ctx: FactorContextLike) -> pd.Series:
        q, _ = quarterly_history(
            ctx, ["net_income", "total_assets"], availability=self.availability, factor=self.name
        )
        ta = q["total_assets"].astype("float64")
        avg_ta = positive_only((ta + shift_q(q, ta, 1)) / 2.0)
        roa_q = q["net_income"] / avg_ta
        q["value"] = -grouped_rolling(q, roa_q, window=self.window, how="std")
        daily = self._daily_values(ctx, q, ["value"])
        return finalize_factor(daily["value"], name=self.name)


class Leverage(FundamentalFactor):
    """Apalancamiento con el signo invertido: ``−(Deuda neta / EBITDA TTM)``.

    Referencia: Campbell, Hilscher y Szilagyi (2008), *In Search of Distress
    Risk*, JF 63(6): desde 1981 las acciones en dificultades financieras han
    ofrecido rentabilidades anormalmente **bajas** con más volatilidad y beta —
    más riesgo y menos rentabilidad—. Menor apalancamiento = más alcista, y por
    eso este factor devuelve **``−NetDebt/EBITDA``** (§12.1 del informe): el
    signo invertido está documentado aquí como exige el contrato.

    - **PIT:** ``"filing"`` (deuda y caja son balance).
    - **``EBITDA TTM <= 0`` → NaN**: el cociente cambia de signo sin sentido
      económico (mismo problema que ``CFO/NI`` con ``NI<0``, §12.1).
    - **Dónde falla — de forma total (§12.1):** el apalancamiento es el factor
      más estructuralmente sectorial de todos. Financieras → NaN (un banco gana
      dinero endeudándose para prestar). Utilities se mantienen pero su
      apalancamiento alto y estable es el modelo de negocio: sin neutralización
      sectorial este factor es una posición corta en Utilities.
    - **Requisitos:** ``ebitda`` (o ``operating_income`` +
      ``depreciation_amortization``), deuda total (o corto + largo) y ``cash``.
    """

    name = "low_leverage"
    requires: list[str] = ["fundamentals"]  # noqa: RUF012
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

        q, _ = quarterly_history(
            ctx,
            [*ebitda_inputs, *debt_inputs, "cash"],
            availability=self.availability,
            factor=self.name,
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
        ebitda_ttm = positive_only(ttm_sum(q, ebitda_q))
        net_debt = debt_q - q["cash"]
        q["value"] = -(net_debt / ebitda_ttm)
        daily = self._daily_values(ctx, q, ["value"])
        return finalize_factor(daily["value"], name=self.name)


FACTORS: tuple[type[FundamentalFactor], ...] = (
    PiotroskiFScore,
    CFOToNetIncome,
    GrossProfitability,
    MarginTrend,
    Profitability,
    EarningsStability,
    Leverage,
)
"""Factores exportados por este módulo."""


def _register() -> None:
    """Registra los factores en el `default_registry` de `factors.base`.

    Import en tiempo de ejecución con try/except: `base.py` lo escribe otro
    agente en paralelo y este módulo debe funcionar también sin él. Las clases
    parametrizadas (variante, métrica, margen) se registran mediante fábricas
    con el nombre que declara cada configuración.
    """
    try:
        from earnings_alpha.factors.base import register_factor
    except ImportError:  # pragma: no cover - base.py aún no integrado
        return
    for cls in (CFOToNetIncome, GrossProfitability, EarningsStability, Leverage):
        register_factor()(cls)

    @register_factor("piotroski_f")
    def _piotroski_binary(**kwargs: object) -> PiotroskiFScore:
        return PiotroskiFScore(variant="binary", **kwargs)  # type: ignore[arg-type]

    @register_factor("piotroski_f_cont")
    def _piotroski_continuous(**kwargs: object) -> PiotroskiFScore:
        return PiotroskiFScore(variant="continuous", **kwargs)  # type: ignore[arg-type]

    @register_factor("margin_trend_gross")
    def _margin_trend_gross(**kwargs: object) -> MarginTrend:
        return MarginTrend(margin="gross", **kwargs)  # type: ignore[arg-type]

    @register_factor("margin_trend_operating")
    def _margin_trend_operating(**kwargs: object) -> MarginTrend:
        return MarginTrend(margin="operating", **kwargs)  # type: ignore[arg-type]

    @register_factor("roa")
    def _roa(**kwargs: object) -> Profitability:
        return Profitability("roa", **kwargs)  # type: ignore[arg-type]

    @register_factor("roe")
    def _roe(**kwargs: object) -> Profitability:
        return Profitability("roe", **kwargs)  # type: ignore[arg-type]


_register()
