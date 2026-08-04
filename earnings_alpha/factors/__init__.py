"""Módulo `factors`: ángulo A, factores fundamentales cross-section (contrato §3.4).

Todo factor cumple el protocolo `Factor` y devuelve una `pd.Series` con
MultiIndex ``(date, ticker)`` donde **valor mayor = más alcista**; los factores
"menos es mejor" se devuelven con el signo cambiado y lo documentan. La
proyección de eventos a panel es point-in-time por construcción
(`pit.tradable_date` + `pit.asof_join`), y cada factor cita su referencia
académica en el docstring (`docs/research/fundamental_factors.md`).

Submódulos:

- `base`: protocolo `Factor`, `FactorContext`, `FactorRegistry` con decorador,
  utilidades de proyección evento→panel y `context_from_synthetic`.
- `surprise`: familia SUE (series temporales y analistas), sorpresa de
  ingresos SURGE, doble sorpresa y PEAD; además `load_consensus_events`, el
  constructor de eventos REALES desde ``consenso_master.parquet`` (con su
  advertencia PIT en `CONSENSUS_PIT_WARNING`).
- `revisions`: momentum de revisiones, difusión y dispersión de analistas;
  detecta la ausencia de vintages (`NoVintagesWarning`) y no la disimula.
- `guidance`: cambios de guidance categóricos → numéricos, con NaN explícito
  donde no hay dato.

Uso típico::

    from earnings_alpha.factors import context_from_synthetic, default_registry
    ctx = context_from_synthetic(SyntheticMarket(seed=7))
    sue = default_registry.compute("sue_analyst", ctx)
"""

from __future__ import annotations

from earnings_alpha.factors.base import (
    Factor,
    FactorContext,
    FactorRegistry,
    StaticUniverse,
    UniverseLike,
    build_panel_index,
    context_from_synthetic,
    default_registry,
    empty_panel,
    register_factor,
    require_columns,
    spread_event_frame,
    spread_event_values,
)
from earnings_alpha.factors.guidance import (
    GUIDANCE_SCORES,
    GuidanceAction,
    GuidanceChange,
    guidance_events,
    guidance_surprise,
    score_guidance_actions,
)
from earnings_alpha.factors.revisions import (
    AnalystDispersion,
    NoVintagesWarning,
    RevisionDiffusion,
    RevisionMomentum,
    analyst_dispersion_panel,
    has_revision_vintages,
    revision_diffusion_panel,
    revision_momentum_panel,
)
from earnings_alpha.factors.surprise import (
    CONSENSUS_PIT_WARNING,
    CONSENSUS_WINDOW_DAYS,
    MIN_QUARTERS_SUE,
    PEAD,
    PEAD_HORIZON,
    SIGMA_FLOOR_FRAC,
    SUE_WINDOW,
    AnalystSUE,
    DoubleSurprise,
    RevenueSurprise,
    TimeSeriesSUE,
    analyst_sue_events,
    attach_tradable_date,
    double_surprise,
    foster_surprise_events,
    load_consensus_events,
    price_before_event,
    revenue_surprise_events,
    seasonal_random_walk,
    sue_time_series,
)

__all__ = [  # noqa: RUF022 - orden temático, no alfabético
    # base
    "Factor",
    "FactorContext",
    "FactorRegistry",
    "default_registry",
    "register_factor",
    "StaticUniverse",
    "UniverseLike",
    "build_panel_index",
    "spread_event_values",
    "spread_event_frame",
    "empty_panel",
    "require_columns",
    "context_from_synthetic",
    # surprise
    "MIN_QUARTERS_SUE",
    "SUE_WINDOW",
    "SIGMA_FLOOR_FRAC",
    "CONSENSUS_WINDOW_DAYS",
    "PEAD_HORIZON",
    "seasonal_random_walk",
    "sue_time_series",
    "attach_tradable_date",
    "price_before_event",
    "analyst_sue_events",
    "foster_surprise_events",
    "revenue_surprise_events",
    "double_surprise",
    "load_consensus_events",
    "CONSENSUS_PIT_WARNING",
    "AnalystSUE",
    "TimeSeriesSUE",
    "RevenueSurprise",
    "DoubleSurprise",
    "PEAD",
    # revisions
    "NoVintagesWarning",
    "has_revision_vintages",
    "revision_momentum_panel",
    "revision_diffusion_panel",
    "analyst_dispersion_panel",
    "RevisionMomentum",
    "RevisionDiffusion",
    "AnalystDispersion",
    # guidance
    "GuidanceAction",
    "GUIDANCE_SCORES",
    "score_guidance_actions",
    "guidance_surprise",
    "guidance_events",
    "GuidanceChange",
]
