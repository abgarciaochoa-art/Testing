"""Estadística de validación: IC, significancia, rendimiento y multiplicidad.

Punto de entrada del módulo `stats` (`docs/ARCHITECTURE.md` §3.8), especificado en
detalle por `docs/research/validation_methodology.md`.

Dos reglas gobiernan todo lo que hay aquí:

1. **Ninguna cifra de significancia sin declarar su corrección de dependencia.** En
   este panel el `t` ingenuo se equivoca por factores de 4 a 8, no por decimales: el
   solapamiento de retornos *forward* lo infla ×√h (×7,98 con `h = 63`) y el
   agrupamiento transversal de las fechas de resultados ×5,1 con una correlación
   media de solo 0,05. Por eso todo contraste devuelve un `SignificanceResult` con
   el campo `method` obligatorio.
2. **Ninguna métrica de rendimiento sin intervalo de confianza.** Un Sharpe sin
   banda de error no se acepta.

Mapa rápido
-----------

=================================  =========================================
Qué quieres saber                   Con qué
=================================  =========================================
¿La señal ordena el corte?          `cross_sectional_ic`, `summarize_ic`
¿Es una apuesta sectorial?          `ic_by_group`, `ic_by_size_bucket`
¿A qué horizonte vive?              `ic_decay`
¿Cuánto vale su `t` de verdad?      `mean_significance`, `disjoint_significance`
¿Gana dinero, y con qué banda?      `sharpe_ratio`, `performance_summary`
¿Sobrevive al número de pruebas?    `deflated_sharpe_ratio`, `pbo_cscv`
¿Y a la multiplicidad?              `multiple_testing_report`, `romano_wolf`
¿Se sostiene fuera de muestra?      `PurgedKFold`, `CombinatorialPurgedCV`
¿Cuál es su banda robusta?          `stationary_bootstrap`
¿Es implementable?                  `turnover`, `capacity`
=================================  =========================================

El orden de las correcciones **no es negociable** (§7.4): primero dependencia,
después p-valores, después multiplicidad, y por último deflación por selección.
Corregir multiplicidad sobre `t` inflados es aplicar una corrección exquisita a
números que están mal por un factor 4.
"""

from earnings_alpha.stats.ic import (
    DEFAULT_HORIZONS,
    DEFAULT_MIN_NAMES,
    TRADING_DAYS_PER_YEAR,
    CorrelationKind,
    ICSummary,
    SignificanceResult,
    cross_sectional_ic,
    disjoint_significance,
    effective_n,
    fisher_ci,
    fisher_se,
    forward_returns,
    ic_by_group,
    ic_by_size_bucket,
    ic_decay,
    long_run_variance,
    mean_significance,
    newey_west_lags,
    restrict_to_universe,
    screen_ic_criteria,
    summarize_ic,
)
from earnings_alpha.stats.performance import (
    CapacityResult,
    DrawdownResult,
    DSRResult,
    Estimate,
    SharpeResult,
    TurnoverResult,
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    capacity,
    deflated_sharpe_ratio,
    drawdown_series,
    effective_breadth,
    expected_max_sharpe,
    fundamental_law_sharpe,
    max_drawdown,
    min_backtest_length,
    min_trl,
    n_effective_trials,
    performance_summary,
    psr,
    raw_kurtosis,
    sharpe_ratio,
    sharpe_standard_error,
    sharpe_variance_factor,
    turnover,
)
from earnings_alpha.stats.validation import (
    EULER_MASCHERONI,
    BootstrapResult,
    CombinatorialPurgedCV,
    CPCVGeometry,
    CPCVSplit,
    MultipleTestingReport,
    MultipleTestResult,
    PBOResult,
    PurgedKFold,
    RomanoWolfResult,
    SPAResult,
    benjamini_hochberg,
    benjamini_yekutieli,
    benjamini_yekutieli_t,
    block_length_from_autocovariance,
    bonferroni,
    bonferroni_t,
    cpcv_paths,
    hansen_spa,
    harvey_liu_zhu_threshold,
    hlz_equivalent_tests,
    holm,
    information_spans,
    multiple_testing_report,
    pbo_cscv,
    politis_white_block_length,
    purge_and_embargo,
    romano_wolf,
    stationary_bootstrap,
    stationary_bootstrap_indices,
    white_reality_check,
)

__all__ = [  # noqa: RUF022 - orden temático, no alfabético
    # --- constantes -------------------------------------------------------
    "TRADING_DAYS_PER_YEAR",
    "DEFAULT_MIN_NAMES",
    "DEFAULT_HORIZONS",
    "EULER_MASCHERONI",
    # --- IC y significancia ----------------------------------------------
    "SignificanceResult",
    "ICSummary",
    "CorrelationKind",
    "cross_sectional_ic",
    "summarize_ic",
    "forward_returns",
    "restrict_to_universe",
    "ic_by_group",
    "ic_by_size_bucket",
    "ic_decay",
    "screen_ic_criteria",
    "newey_west_lags",
    "long_run_variance",
    "mean_significance",
    "disjoint_significance",
    "effective_n",
    "fisher_se",
    "fisher_ci",
    # --- rendimiento ------------------------------------------------------
    "Estimate",
    "SharpeResult",
    "DSRResult",
    "DrawdownResult",
    "TurnoverResult",
    "CapacityResult",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "sharpe_standard_error",
    "sharpe_variance_factor",
    "raw_kurtosis",
    "psr",
    "min_trl",
    "expected_max_sharpe",
    "deflated_sharpe_ratio",
    "min_backtest_length",
    "n_effective_trials",
    "max_drawdown",
    "drawdown_series",
    "calmar_ratio",
    "turnover",
    "capacity",
    "effective_breadth",
    "fundamental_law_sharpe",
    "performance_summary",
    # --- validación -------------------------------------------------------
    "PurgedKFold",
    "CombinatorialPurgedCV",
    "CPCVGeometry",
    "CPCVSplit",
    "cpcv_paths",
    "information_spans",
    "purge_and_embargo",
    "BootstrapResult",
    "stationary_bootstrap",
    "stationary_bootstrap_indices",
    "politis_white_block_length",
    "block_length_from_autocovariance",
    "MultipleTestResult",
    "MultipleTestingReport",
    "RomanoWolfResult",
    "bonferroni",
    "holm",
    "benjamini_hochberg",
    "benjamini_yekutieli",
    "bonferroni_t",
    "benjamini_yekutieli_t",
    "harvey_liu_zhu_threshold",
    "hlz_equivalent_tests",
    "romano_wolf",
    "multiple_testing_report",
    "PBOResult",
    "pbo_cscv",
    "SPAResult",
    "white_reality_check",
    "hansen_spa",
]
