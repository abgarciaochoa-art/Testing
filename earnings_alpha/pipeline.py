"""Pipelines end-to-end de la plataforma: ángulo A (continuo) y ángulo B (evento).

Este módulo es la capa de orquestación sobre los módulos ya existentes; no
implementa ninguna matemática nueva. Encadena, en el orden del contrato
(`docs/ARCHITECTURE.md` §2-§3), los pasos de cada ángulo de investigación:

- **Ángulo A** (`run_continuous_pipeline`): universo PIT → fundamentales →
  factores (`factors.default_registry`) → neutralización (`signals.condition_factor`)
  → combinación (`signals.combine`) → backtest cross-section
  (`backtest.CrossSectionalBacktest`) → métricas (`stats.cross_sectional_ic` +
  `stats.summarize_ic`, con t de Newey-West y banda de error obligatoria).
- **Ángulo B** (`run_event_pipeline`): universo PIT → calendario de anuncios →
  `events.PreEventFeatures` → `events.InformedTradingScore` →
  `events.SurpriseModel` (CV purgada con embargo, la única validación admitida)
  → `backtest.EventBacktest` vía `run_grid` → estudio de eventos con CAAR por
  quintil (`events.abnormal_returns` + `events.caar_by_group`).

Principios que el orquestador hace cumplir (y no puede delegar):

1. **Point-in-time o nada.** Todas las proyecciones evento→panel pasan por
   `pit.tradable_date`/`pit.asof_join` dentro de cada factor; los retornos
   forward llevan retardo de ejecución (`stats.forward_returns`,
   ``execution_lag=1``); el pre-posicionamiento del motor de eventos exige la
   declaración explícita `calendar_known_in_advance` (`pit_and_biases.md` §8.3)
   y este módulo **no** la hace en nombre del usuario: quien pre-posiciona,
   declara.
2. **Fallo explícito.** Un factor sin datos lanza (`ProviderUnavailable`,
   `InsufficientHistory`); con ``on_factor_error="skip"`` el descarte queda
   registrado en ``result.skipped_factors``, nunca oculto.
3. **Ninguna métrica sin banda de error.** Los resúmenes reutilizan
   `stats.ICSummary` y `stats.SharpeResult`, que la llevan de serie.
4. **Determinismo.** Toda la aleatoriedad vive en los componentes subyacentes,
   que ya aceptan semilla; el orquestador no introduce ninguna propia.

Nota de multiplicidad (obligatoria): la rejilla de `run_grid` son N ensayos
sobre los mismos datos, y cada variante de combinación de factores es un ensayo
más. Antes de promover el mejor punto, pásese por
`stats.validation.benjamini_hochberg` o `stats.performance.deflated_sharpe_ratio`
con el `n_trials` real (`validation_methodology.md` §7).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

# Los submódulos de factores fundamentales se importan por su efecto de registro
# en `default_registry`; `earnings_alpha.factors` solo registra surprise,
# revisions y guidance al importarse.
import earnings_alpha.factors.accruals
import earnings_alpha.factors.growth
import earnings_alpha.factors.quality
import earnings_alpha.factors.value  # noqa: F401 - registro de factores
from earnings_alpha.backtest import (
    CostModel,
    CrossSectionalBacktest,
    EventBacktest,
    run_grid,
)
from earnings_alpha.backtest.engine import BacktestResult
from earnings_alpha.errors import (
    ConfigError,
    DataQualityError,
    InsufficientHistory,
    LookAheadError,
    ProviderUnavailable,
)
from earnings_alpha.events import (
    aar_caar,
    abnormal_returns,
    caar_by_group,
    normalize_events,
    quantile_groups,
)
from earnings_alpha.events.preevent import (
    EventContext,
    InformedTradingScore,
    PreEventFeatures,
)
from earnings_alpha.events.surprise_model import (
    SurpriseModel,
    SurpriseModelReport,
    event_labels,
    roc_auc,
)
from earnings_alpha.factors import (
    Factor,
    FactorContext,
    context_from_synthetic,
    default_registry,
)
from earnings_alpha.pit import get_calendar
from earnings_alpha.signals import (
    condition_factor,
    fixed_weight_combine,
    ic_weighted_combine,
    winsorize,
    zscore,
)
from earnings_alpha.stats import (
    DEFAULT_MIN_NAMES,
    ICSummary,
    cross_sectional_ic,
    forward_returns,
    summarize_ic,
)
from earnings_alpha.universe import UniverseProvider

__all__ = [
    "DEFAULT_CONTINUOUS_FACTORS",
    "ContinuousPipelineResult",
    "EventPipelineResult",
    "run_continuous_pipeline",
    "run_event_pipeline",
]


DEFAULT_CONTINUOUS_FACTORS: tuple[str, ...] = (
    "sue_analyst",
    "pead",
    "revenue_surprise",
    "earnings_yield",
    "piotroski_f",
    "accruals_cf",
)
"""Cesta por defecto del ángulo A: sorpresa de analistas (Livnat-Mendenhall
2006), PEAD (Bernard-Thomas 1989), sorpresa de ingresos (Jegadeesh-Livnat
2006), earnings yield (Basu 1977), Piotroski (2000) F-score y accruals por
flujo de caja (Sloan 1996; Hribar-Collins 2002). Cada factor cita su
referencia completa en su propio docstring."""


# ---------------------------------------------------------------------------
# Resultados
# ---------------------------------------------------------------------------


@dataclass
class ContinuousPipelineResult:
    """Resultado completo del pipeline continuo (ángulo A).

    Todos los paneles usan el MultiIndex canónico ``(date, ticker)``. Ninguna
    métrica viene sin banda de error: `ic_by_factor` e `ic_combined` son
    `stats.ICSummary` (t de Newey-West, IC de la media) y el Sharpe del
    backtest es un `stats.SharpeResult` dentro de ``backtest.summary()``.
    """

    factor_panel: pd.DataFrame
    """Factores crudos, una columna por factor."""
    conditioned: pd.DataFrame
    """Factores tras winsorización, z-score y neutralización por fecha."""
    combined: pd.Series
    """Puntuación combinada final (la señal que entra al backtest)."""
    forward: pd.Series
    """Retorno forward usado para la IC (con retardo de ejecución de 1 sesión)."""
    ic_by_factor: dict[str, ICSummary]
    ic_combined: ICSummary
    backtest: BacktestResult
    skipped_factors: dict[str, str] = field(default_factory=dict)
    """Factores descartados (solo con ``on_factor_error="skip"``) y su motivo."""
    ic_errors: dict[str, str] = field(default_factory=dict)
    """Factores cuya IC no fue estimable (p. ej. panel demasiado corto)."""
    params: dict[str, object] = field(default_factory=dict)

    def summary(self) -> str:
        """Resumen legible en español, apto para imprimir en un terminal."""
        lines: list[str] = []
        lines.append("=" * 72)
        lines.append("PIPELINE CONTINUO (ángulo A): factores fundamentales cross-section")
        lines.append("=" * 72)
        lines.append(
            f"Universo: {self.params.get('n_tickers', '?')} símbolos | "
            f"panel: {self.params.get('start', '?')} → {self.params.get('end', '?')} | "
            f"horizonte IC: {self.params.get('forward_horizon', '?')} sesiones"
        )
        lines.append("")
        lines.append("IC por factor (rank-IC medio, t de Newey-West, IC 95%):")
        for name, s in self.ic_by_factor.items():
            lo, hi = s.ci
            lines.append(
                f"  {name:<24} IC={s.mean:+.4f}  t_NW={s.t_stat:+5.2f}  "
                f"[{lo:+.4f}, {hi:+.4f}]  ({s.n_periods} fechas)"
            )
        for name, msg in self.ic_errors.items():
            lines.append(f"  {name:<24} IC no estimable: {msg}")
        for name, msg in self.skipped_factors.items():
            lines.append(f"  {name:<24} DESCARTADO: {msg}")
        s = self.ic_combined
        lo, hi = s.ci
        lines.append("")
        lines.append(
            f"Señal combinada ({self.params.get('combine_method', '?')}): "
            f"IC={s.mean:+.4f}  t_NW={s.t_stat:+5.2f}  [{lo:+.4f}, {hi:+.4f}]  "
            f"fracción de meses positivos={s.monthly_positive_fraction:.0%}"
        )
        bt = self.backtest.summary()
        sr = bt["sharpe"]
        lines.append("")
        lines.append("Backtest cross-section (neto de costes):")
        lines.append(
            f"  Sharpe anualizado = {sr.sharpe_annualized:+.2f} "
            f"[{sr.ci_low:+.2f}, {sr.ci_high:+.2f}]  (PSR₀={sr.psr_zero:.2f})"
        )
        lines.append(
            f"  retorno anual = {bt['ann_return']:+.2%} | vol anual = {bt['ann_volatility']:.2%} "
            f"| máx. drawdown = {bt['max_drawdown']:.2%}"
        )
        lines.append(
            f"  rotación one-way anual = {bt['annual_one_way_turnover']:.2f}x | "
            f"rebalanceos = {bt['n_rebalances']} (omitidos: {bt['n_skipped']})"
        )
        costs = bt["total_costs"]
        lines.append(
            "  costes acumulados (fracción del NAV): "
            + ", ".join(f"{k}={v:.4%}" for k, v in costs.items())
        )
        if bt["n_silent_delistings"]:
            lines.append(
                f"  ¡AUDITORÍA!: {bt['n_silent_delistings']} delistings sin retorno final "
                "conocido (revisar backtest.silent_delistings antes de creer nada)"
            )
        lines.append("")
        lines.append(
            "Advertencia de multiplicidad: cada variante probada (factores, pesos, "
            "horizontes) es un ensayo; deflactar con stats.deflated_sharpe_ratio."
        )
        return "\n".join(lines)


@dataclass
class EventPipelineResult:
    """Resultado completo del pipeline de eventos (ángulo B)."""

    features: pd.DataFrame
    """Tabla de `events.PreEventFeatures`, indexada por ``event_id``."""
    intensity_score: pd.Series
    """`InformedTradingScore(mode="intensity")`: ¿cuánta actividad anómala hay?"""
    directional_score: pd.Series
    """`InformedTradingScore(mode="directional")`: ¿hacia dónde apunta? (mayor = alcista)."""
    labels: pd.DataFrame
    """Etiquetas por evento (`surprise`, `surprise_sign`, `event_return`)."""
    model_report: SurpriseModelReport | None
    """Informe OOS de la CV purgada del `SurpriseModel`; None si no se evaluó."""
    model: SurpriseModel | None
    """El modelo (ajustado sobre todo el panel si ``model_mode="fit"``)."""
    grid: pd.DataFrame
    """Tabla comparativa de `backtest.run_grid` (entry_offset x exit_offset)."""
    event_study: pd.DataFrame | None
    """Panel AR/CAR por (evento, tau) de `events.abnormal_returns`."""
    aar: pd.DataFrame | None
    """AAR/CAAR agregado por tau con errores estándar transversales."""
    caar_by_quintile: pd.DataFrame | None
    """CAAR por (quintil, tau): el insumo del gráfico canónico del PEAD."""
    caar_groups: pd.Series | None
    """Etiqueta de quintil por event_id usada en `caar_by_quintile`."""
    detection: dict[str, float] | None
    """Métricas del detector contra la verdad-terreno (solo bancos sintéticos)."""
    params: dict[str, object] = field(default_factory=dict)

    def summary(self) -> str:
        """Resumen legible en español, apto para imprimir en un terminal."""
        lines: list[str] = []
        lines.append("=" * 72)
        lines.append("PIPELINE DE EVENTOS (ángulo B): ventana de resultados")
        lines.append("=" * 72)
        n_feat = len(self.features)
        n_cols = self.features.shape[1]
        lines.append(f"Eventos con features: {n_feat} ({n_cols} columnas)")
        missing = self.features.attrs.get("missing_sources", [])
        if missing:
            lines.append(f"  fuentes ausentes (features NaN, degradación declarada): {missing}")
        if self.detection is not None:
            d = self.detection
            lines.append("")
            lines.append("Detección de huella informada (vs verdad-terreno sintética):")
            lines.append(
                f"  eventos={d['n_events']:.0f} | con filtración={d['n_leaked']:.0f} "
                f"(prevalencia={d['prevalence']:.1%})"
            )
            if np.isfinite(d.get("auc", float("nan"))):
                lines.append(
                    f"  AUC del score de intensidad = {d['auc']:.3f} "
                    f"(se₀={d['auc_se']:.3f}, t vs 0.5 = {d['auc_t']:+.2f})"
                )
                lines.append(
                    f"  score medio: filtrados={d['mean_score_leaked']:+.3f} "
                    f"vs resto={d['mean_score_other']:+.3f}"
                )
            lines.append(
                "  Recordatorio de tasa base (informed_trading.md §12.4): con "
                "prevalencia baja la PPV se hunde; el score pondera exposición, "
                "no dispara alertas binarias."
            )
        if self.model_report is not None:
            r = self.model_report
            lines.append("")
            lines.append(
                f"SurpriseModel [{r.target}/{r.objective}/{r.estimator}] — "
                f"CV purgada de {r.n_splits} folds (OOS honesto):"
            )
            for k in sorted(set(r.metrics) | set(r.baselines)):
                m = r.metrics.get(k, float("nan"))
                b = r.baselines.get(k, float("nan"))
                lines.append(f"  {k:<16} modelo={m:+.4f}   línea base={b:+.4f}")
            verdict = "SÍ" if r.beats_baseline else "NO"
            lines.append(f"  ¿Bate a la línea base honesta (cribado)? {verdict}")
        lines.append("")
        ok = self.grid[self.grid["status"] == "ok"] if "status" in self.grid.columns else self.grid
        lines.append(
            f"Rejilla de backtest de eventos: {len(self.grid)} combinaciones "
            f"({len(ok)} ejecutables) — ¡{len(self.grid)} ensayos para la multiplicidad!"
        )
        if len(ok) > 0:
            cols = [
                c
                for c in ("n_events", "hit_rate", "mean_net", "mean_net_ci_low",
                          "mean_net_ci_high", "gap_share_log")
                if c in ok.columns
            ]
            best = ok.sort_values("mean_net", ascending=False).head(3)
            for (eo, xo), row in best[cols].iterrows():
                lines.append(
                    f"  entrada T{eo:+d} → salida T{xo:+d}: n={row['n_events']:.0f}, "
                    f"hit={row['hit_rate']:.0%}, media neta={row['mean_net']:+.4%} "
                    f"[{row['mean_net_ci_low']:+.4%}, {row['mean_net_ci_high']:+.4%}]"
                )
        if self.caar_by_quintile is not None:
            lines.append("")
            lines.append("CAAR por quintil (tau final de la ventana):")
            tail = self.caar_by_quintile.groupby(level="group", observed=True).tail(1)
            for (group, tau), row in tail.iterrows():
                lines.append(
                    f"  {group}: CAAR[{tau:+d}] = {row['caar']:+.4f} "
                    f"(t={row['caar_t']:+.2f}, n={row['n']:.0f})"
                )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Construcción de contextos
# ---------------------------------------------------------------------------


def _factor_context(source: object, *, include_prehistory: bool) -> FactorContext:
    """`FactorContext` desde un `SyntheticMarket` (o compatible) o passthrough."""
    if isinstance(source, FactorContext):
        return source
    return context_from_synthetic(source, include_prehistory=include_prehistory)


def _event_context(source: object) -> EventContext:
    """`EventContext` desde un `SyntheticMarket` (o compatible) o passthrough.

    Duck-typing deliberado (mismo criterio que `factors.context_from_synthetic`):
    cualquier objeto con ``events()``, ``prices()`` y ``market_index()`` sirve;
    las fuentes opcionales (short interest, off-exchange, opciones) se toman si
    existen y, si no, sus features saldrán NaN con la fuente anotada en
    ``attrs['missing_sources']`` — degradación declarada, nunca silenciosa.
    """
    if isinstance(source, EventContext):
        return source
    for attr in ("events", "prices", "market_index"):
        if not hasattr(source, attr):
            msg = (
                f"la fuente no expone `{attr}()`: se esperaba un SyntheticMarket, "
                "un objeto compatible o un EventContext ya construido"
            )
            raise ConfigError(msg)
    def _optional(name: str) -> pd.DataFrame | None:
        return getattr(source, name)() if hasattr(source, name) else None

    return EventContext(
        events=source.events(include_prehistory=True),  # type: ignore[attr-defined]
        prices=source.prices(),  # type: ignore[attr-defined]
        market=source.market_index(),  # type: ignore[attr-defined]
        calendar=getattr(source, "calendar", None) or get_calendar(),
        sectors=source.sectors() if hasattr(source, "sectors") else None,
        short_interest=_optional("short_interest"),
        off_exchange=_optional("off_exchange"),
        options=_optional("options_daily"),
    )


def _build_exposures(
    ctx: FactorContext,
    index: pd.MultiIndex,
    neutralize_by: Sequence[str],
) -> pd.DataFrame:
    """Exposiciones estándar para la neutralización: sector y tamaño.

    - ``sector``: sector GICS del `UniverseProvider` (dummies en `neutralize`).
    - ``size``: ``ln(market_cap)`` si el panel lo trae; si no, ``ln(ADV_21)``
      (media móvil hacia atrás del dólar-volumen, PIT por construcción). El
      tamaño es el control obligatorio de `fundamental_factors.md` §1.4.

    Para exposiciones adicionales (beta, momentum...) el llamante debe pasar
    su propio `exposures=` a `run_continuous_pipeline`.
    """
    known = {"sector", "size"}
    unknown = [c for c in neutralize_by if c not in known]
    if unknown:
        msg = (
            f"exposiciones no construibles automáticamente: {unknown}. "
            "Pásalas ya calculadas con el argumento `exposures=` "
            f"(automáticas: {sorted(known)})"
        )
        raise ConfigError(msg)
    tickers_level = index.get_level_values("ticker")
    cols: dict[str, pd.Series] = {}
    if "sector" in neutralize_by:
        sec_map = {t: ctx.universe.sector_for(t) for t in set(tickers_level)}
        cols["sector"] = pd.Series(
            [sec_map.get(t) for t in tickers_level], index=index, dtype=object
        )
    if "size" in neutralize_by:
        if "market_cap" in ctx.prices.columns:
            mc = ctx.prices["market_cap"].reindex(index).astype(float)
            cols["size"] = np.log(mc.where(mc > 0))
        elif "dollar_volume" in ctx.prices.columns:
            dv = ctx.prices["dollar_volume"].astype(float)
            adv = (
                dv.groupby(level="ticker")
                .transform(lambda s: s.rolling(21, min_periods=10).mean())
                .reindex(index)
            )
            cols["size"] = np.log(adv.where(adv > 0))
        else:
            msg = (
                "no se puede construir la exposición `size`: el panel de precios "
                "no trae ni `market_cap` ni `dollar_volume`. Pasa `exposures=` o "
                "quita 'size' de `neutralize_by`"
            )
            raise DataQualityError(msg)
    return pd.DataFrame(cols)


# ---------------------------------------------------------------------------
# Ángulo A: pipeline continuo
# ---------------------------------------------------------------------------


def run_continuous_pipeline(
    source: object,
    *,
    factors: Sequence[str | Factor] | None = None,
    factor_params: Mapping[str, Mapping[str, object]] | None = None,
    exposures: pd.DataFrame | None = None,
    neutralize_by: Sequence[str] = ("sector", "size"),
    combine_method: Literal["weights", "ic_weighted"] = "weights",
    combine_weights: Mapping[str, float] | None = None,
    forward_horizon: int = 5,
    rebalance: str = "W-FRI",
    n_quantiles: int = 5,
    long_short: bool = True,
    costs: CostModel | None = None,
    max_weight: float = 0.05,
    adv_participation: float | None = 0.05,
    capital: float = 10_000_000.0,
    min_names: int | None = None,
    neutralize_min_obs: int | None = None,
    include_prehistory: bool = True,
    on_factor_error: Literal["raise", "skip"] = "raise",
) -> ContinuousPipelineResult:
    """Pipeline end-to-end del ángulo A: de los datos crudos a las métricas.

    Cadena completa (contrato §2): **universo PIT** (el `UniverseProvider` del
    contexto; con `SyntheticMarket`, un `StaticUniverse` documentadamente solo
    apto para bancos de pruebas) → **fundamentales** (tabla point-in-time del
    contexto, indexada por `available_at`) → **factores** (por nombre del
    `factors.default_registry` o instancias `Factor`) → **neutralización**
    (`signals.condition_factor`: winsorizar → z-score → residualizar contra
    sector/tamaño → re-estandarizar, todo por fecha) → **combinación**
    (`signals.fixed_weight_combine` o `ic_weighted_combine`, esta última con
    pesos estrictamente pasados: el IC de `t` no es observable en `t`) →
    **backtest cross-section** (`CrossSectionalBacktest`, ejecución en la
    apertura siguiente, costes desglosados) → **métricas** (`ICSummary` por
    factor y combinada, `SharpeResult` con banda).

    Parameters
    ----------
    source:
        `data.synthetic.SyntheticMarket` (o compatible por duck-typing) o un
        `factors.FactorContext` ya construido con datos reales.
    factors:
        Nombres del registro (`factors.default_registry.names()`) o instancias
        `Factor`. Por defecto `DEFAULT_CONTINUOUS_FACTORS`.
    factor_params:
        Parámetros de construcción por nombre de factor, p. ej.
        ``{"sue_analyst": {"basis": "price"}}``.
    exposures:
        Panel ``(date, ticker)`` de exposiciones para la neutralización. Si es
        None se construyen `sector` (universo) y `size` (ln market cap o ln ADV).
    neutralize_by:
        Columnas de exposición a neutralizar. ``()`` desactiva la
        neutralización (solo winsorizar + z-score); hacerlo con datos reales
        deja apuestas sectoriales disfrazadas de factor
        (`fundamental_factors.md` §1.4).
    combine_method, combine_weights:
        ``"weights"`` (pesos fijos; None = iguales, la línea base difícil de
        batir) o ``"ic_weighted"`` (ventana expansiva PIT).
    forward_horizon:
        Sesiones del retorno forward de la IC (con retardo de ejecución 1).
    min_names:
        Mínimo de nombres por fecha para que la IC de esa fecha sea un número.
        None = ``min(30, max(6, 4·n/5))``: 30 es el mínimo de
        `validation_methodology.md` §2.1 y solo se relaja en universos
        sintéticos reducidos, donde n < 30 por construcción.
    neutralize_min_obs:
        Mínimo de nombres por fecha para neutralizar. None =
        ``min(20, max(6, n//2))`` (20 es el default deliberadamente alto de
        `signals.neutralize`; véase su docstring).
    on_factor_error:
        ``"raise"`` (defecto, regla de fallo explícito) o ``"skip"``: el factor
        que falla queda excluido y **registrado** en
        ``result.skipped_factors`` — descartado a la vista, no en silencio.

    Returns
    -------
    ContinuousPipelineResult
        Paneles intermedios, IC con bandas y `BacktestResult` completo.
    """
    ctx = _factor_context(source, include_prehistory=include_prehistory)
    tickers = ctx.tickers()
    n_names = len(tickers)
    if n_names < 2:
        msg = f"solo {n_names} símbolos en el contexto: no hay sección cruzada"
        raise InsufficientHistory(msg)
    eff_min_names = (
        int(min_names)
        if min_names is not None
        else min(DEFAULT_MIN_NAMES, max(6, (4 * n_names) // 5))
    )
    eff_neut_obs = (
        int(neutralize_min_obs)
        if neutralize_min_obs is not None
        else min(20, max(6, n_names // 2))
    )
    eff_combine_obs = min(20, max(4, n_names // 2))

    # ------------------------------------------------------------- factores
    chosen: Sequence[str | Factor] = factors if factors is not None else DEFAULT_CONTINUOUS_FACTORS
    if len(chosen) == 0:
        msg = "la lista de factores está vacía"
        raise ConfigError(msg)
    params_by_name = {k: dict(v) for k, v in (factor_params or {}).items()}
    instances: list[Factor] = []
    for f in chosen:
        if isinstance(f, str):
            instances.append(default_registry.create(f, **params_by_name.get(f, {})))
        else:
            instances.append(f)

    raw: dict[str, pd.Series] = {}
    skipped: dict[str, str] = {}
    for factor in instances:
        try:
            raw[factor.name] = factor.compute(ctx)
        except (InsufficientHistory, ProviderUnavailable, DataQualityError) as exc:
            if on_factor_error == "raise":
                raise
            skipped[factor.name] = f"{type(exc).__name__}: {exc}"
    if not raw:
        msg = (
            "ningún factor pudo computarse; motivos: "
            + "; ".join(f"{k}: {v}" for k, v in skipped.items())
        )
        raise InsufficientHistory(msg)
    factor_panel = pd.DataFrame(raw).sort_index()

    # -------------------------------------------------------- neutralización
    if len(neutralize_by) > 0:
        expo = (
            exposures
            if exposures is not None
            else _build_exposures(ctx, factor_panel.index, neutralize_by)  # type: ignore[arg-type]
        )
        conditioned = condition_factor(
            factor_panel,
            expo,
            by=list(neutralize_by),
            min_obs=eff_neut_obs,
        )
    else:
        conditioned = zscore(winsorize(factor_panel))
    if not isinstance(conditioned, pd.DataFrame):  # una sola columna
        conditioned = conditioned.to_frame()

    # ------------------------------------------------- retorno forward (IC)
    price_col = "adj_close" if "adj_close" in ctx.prices.columns else "close"
    forward = forward_returns(
        ctx.prices, horizon=forward_horizon, price_col=price_col, execution_lag=1
    )

    # ----------------------------------------------------------- combinación
    if combine_method == "weights":
        combined = fixed_weight_combine(
            conditioned,
            dict(combine_weights) if combine_weights is not None else None,
            min_obs=eff_combine_obs,
        )
    elif combine_method == "ic_weighted":
        combined = ic_weighted_combine(
            conditioned,
            forward,
            horizon=forward_horizon,
            min_obs=eff_combine_obs,
            min_obs_ic=eff_min_names,
        )
    else:
        msg = f"combine_method desconocido: {combine_method!r} ('weights' | 'ic_weighted')"
        raise ConfigError(msg)
    combined = combined.rename("combined_score")

    # ------------------------------------------------------------- backtest
    sectors = pd.Series({t: ctx.universe.sector_for(t) for t in tickers}, name="sector")
    engine = CrossSectionalBacktest(
        universe=ctx.universe, sectors=sectors, capital=capital
    )
    backtest = engine.run(
        combined,
        ctx.prices,
        n_quantiles=n_quantiles,
        rebalance=rebalance,
        long_short=long_short,
        costs=costs if costs is not None else CostModel(),
        max_weight=max_weight,
        adv_participation=adv_participation,
    )

    # -------------------------------------------------------------- métricas
    dates = pd.DatetimeIndex(ctx.dates).sort_values()
    membership: pd.DataFrame | None = None
    try:
        membership = ctx.universe.membership_panel(dates[0].date(), dates[-1].date())
    except Exception as exc:
        # Sin panel de pertenencia la IC no se filtra; se anota en params para
        # que el consumidor sepa que la IC puede llevar sesgo de supervivencia.
        membership = None
        membership_note = f"membership_panel no disponible: {exc}"
    else:
        membership_note = "ok"

    ic_by_factor: dict[str, ICSummary] = {}
    ic_errors: dict[str, str] = {}
    for name in conditioned.columns:
        try:
            ic_series = cross_sectional_ic(
                conditioned[name], forward, min_names=eff_min_names, membership=membership
            )
            ic_by_factor[name] = summarize_ic(ic_series, horizon=forward_horizon)
        except (InsufficientHistory, DataQualityError) as exc:
            ic_errors[name] = str(exc)

    ic_combined_series = cross_sectional_ic(
        combined, forward, min_names=eff_min_names, membership=membership
    )
    ic_combined = summarize_ic(ic_combined_series, horizon=forward_horizon)

    params: dict[str, object] = {
        "factors": [f.name for f in instances],
        "skipped_factors": dict(skipped),
        "neutralize_by": list(neutralize_by),
        "combine_method": combine_method,
        "combine_weights": dict(combine_weights) if combine_weights else None,
        "forward_horizon": int(forward_horizon),
        "rebalance": rebalance,
        "n_quantiles": int(n_quantiles),
        "long_short": bool(long_short),
        "max_weight": float(max_weight),
        "capital": float(capital),
        "min_names": eff_min_names,
        "neutralize_min_obs": eff_neut_obs,
        "n_tickers": n_names,
        "start": str(dates[0].date()),
        "end": str(dates[-1].date()),
        "membership_filter": membership_note,
        "price_col_forward": price_col,
    }
    return ContinuousPipelineResult(
        factor_panel=factor_panel,
        conditioned=conditioned,
        combined=combined,
        forward=forward,
        ic_by_factor=ic_by_factor,
        ic_combined=ic_combined,
        backtest=backtest,
        skipped_factors=skipped,
        ic_errors=ic_errors,
        params=params,
    )


# ---------------------------------------------------------------------------
# Ángulo B: pipeline de eventos
# ---------------------------------------------------------------------------


def _label_column(model: SurpriseModel) -> str:
    """Columna de `event_labels` que corresponde al target/objective del modelo."""
    if model.target == "event_return":
        return "event_return"
    return "surprise_sign" if model.objective == "sign" else "surprise"


def run_event_pipeline(
    source: object,
    *,
    features_engine: PreEventFeatures | None = None,
    residualize: bool = False,
    intensity: InformedTradingScore | None = None,
    directional: InformedTradingScore | None = None,
    surprise_model: SurpriseModel | None = None,
    model_mode: Literal["evaluate", "fit", "skip"] = "evaluate",
    n_splits: int = 4,
    embargo: float = 0.02,
    label_window: tuple[int, int] = (0, 1),
    surprise_col: str = "sue",
    entry_offsets: Sequence[int] = (0, 1),
    exit_offsets: Sequence[int] = (1, 5, 20),
    backtest_score: Literal["directional", "intensity", "model"] | pd.Series = "directional",
    side: Literal["long", "short", "signed"] = "signed",
    engine: EventBacktest | None = None,
    calendar_known_in_advance: bool = False,
    max_concurrent: int | None = 20,
    min_events: int = 10,
    event_study: bool = True,
    pre: int = 10,
    post: int = 30,
    estimation: tuple[int, int] = (-250, -40),
    min_estimation_obs: int = 120,
    caar_by: str | pd.Series = "sue",
    caar_quantiles: int = 5,
    min_group_size: int = 5,
    leaked_event_ids: Sequence[str] | None = None,
    universe: UniverseProvider | None = None,
) -> EventPipelineResult:
    """Pipeline end-to-end del ángulo B: de los anuncios a la rejilla y el CAAR.

    Cadena completa (contrato §3.5 y §3.7): **universo PIT + calendario** (la
    tabla de eventos se canoniza con `events.normalize_events`, que deriva la
    sesión negociable vía `pit.tradable_date`: BMO → misma sesión, AMC/UNKNOWN
    → la siguiente) → **PreEventFeatures** (huella en [T-N, T-1], solo datos
    públicos) → **InformedTradingScore** (intensidad para detectar,
    direccional para operar) → **SurpriseModel** (CV purgada con embargo — el
    K-fold aleatorio está prohibido en `events.surprise_model`) →
    **EventBacktest.run_grid** (rejilla entrada x salida con gap overnight
    explícito) → **estudio de eventos** (`abnormal_returns` modelo de mercado +
    `caar_by_group` por quintil).

    Puntos PIT que este orquestador impone y no negocia:

    - Solo el score **pre-evento** (directional/intensity, derivados de
      `PreEventFeatures`) es legal con ``entry_offset < 0``; y aun entonces el
      motor exige `calendar_known_in_advance=True`, que aquí se expone tal
      cual y **por defecto es False**: pre-posicionarse sin declarar la
      conocibilidad del calendario lanza `LookAheadError`
      (`pit_and_biases.md` §8.3).
    - Con ``backtest_score="model"`` se usan las predicciones **OOS** de la CV
      purgada (cada evento predicho por un modelo que jamás lo vio). Aviso
      metodológico: esas predicciones son cross-temporales (folds posteriores
      predicen eventos anteriores); sirven como diagnóstico de investigación,
      no como NAV negociable — para eso hay que reentrenar walk-forward.
      Las probabilidades de signo se centran en 0.5 para que ``side="signed"``
      tenga signo.
    - La etiqueta empieza en ``label_window[0] >= 0``: una etiqueta anterior a
      T solaparía la ventana de features (`event_labels` lo rechaza).

    Parameters
    ----------
    source:
        `SyntheticMarket` (o compatible) o un `events.preevent.EventContext`.
    universe:
        `UniverseProvider` opcional. Si se pasa, los eventos se filtran a los
        emisores que pertenecían al índice en su `event_date` (pertenencia
        PIT); los descartados quedan contados en
        ``params['n_events_outside_universe']``. Con eventos reales derivados
        de la lista ACTUAL de constituyentes (p. ej. `load_consensus_events`)
        omitir este filtro incluye toda la historia pre-inclusión — sesgo de
        selección hacia futuros miembros — y queda anotado en
        ``params['universe_filter']``.
    model_mode:
        ``"evaluate"`` (CV purgada completa → `SurpriseModelReport`),
        ``"fit"`` (ajuste sobre todo el panel, sin OOS — solo coeficientes) o
        ``"skip"`` (sin modelo; paneles cortos donde la CV purgada no cabe).
    surprise_col:
        Columna de sorpresa estandarizada de la tabla de eventos (``"sue"`` en
        el generador sintético; con eventos reales de `load_consensus_events`
        habría que construirla antes, p. ej. con `factors.analyst_sue_events`).
    caar_by:
        Variable de agrupación del CAAR: columna de la tabla de eventos
        (``"sue"``), ``"intensity"`` / ``"directional"`` / ``"model"`` (los
        scores del pipeline) o una Series por event_id ya construida.
    leaked_event_ids:
        Verdad-terreno opcional para las métricas de detección. Si es None y
        `source` expone ``leaked_event_ids()`` (el banco sintético), se toma de
        ahí. Con datos reales no existe y `detection` sale None.

    Returns
    -------
    EventPipelineResult
    """
    # ------------------------------------------------ validación fail-fast
    # Ambas condiciones las re-validan los módulos subyacentes (`event_labels`
    # y `EventBacktest.run`); comprobarlas aquí evita pagar el cómputo de
    # features antes de descubrir un experimento mal planteado.
    w0, w1 = int(label_window[0]), int(label_window[1])
    if not 0 <= w0 <= w1:
        msg = (
            f"label_window inválida {label_window}: se exige 0 <= w0 <= w1. Una "
            "etiqueta que empieza antes de T solapa la ventana de features y "
            "convierte el modelo en una tautología (events.event_labels)"
        )
        raise DataQualityError(msg)
    if len(entry_offsets) == 0 or len(exit_offsets) == 0:
        msg = "entry_offsets y exit_offsets no pueden estar vacíos"
        raise ConfigError(msg)
    if min(int(e) for e in entry_offsets) < 0 and not calendar_known_in_advance:
        msg = (
            f"entry_offsets={list(entry_offsets)} incluye entradas ANTES del "
            "anuncio y la fecha del evento no era necesariamente conocible "
            "entonces (pit_and_biases.md §8.3). Si las fechas están resueltas "
            "point-in-time (o son sintéticas, con calendario público por "
            "adelantado), decláralo con calendar_known_in_advance=True"
        )
        raise LookAheadError(msg)
    if model_mode not in ("evaluate", "fit", "skip"):
        msg = f"model_mode desconocido: {model_mode!r} ('evaluate' | 'fit' | 'skip')"
        raise ConfigError(msg)
    if not isinstance(backtest_score, pd.Series) and backtest_score not in (
        "directional",
        "intensity",
        "model",
    ):
        msg = (
            f"backtest_score desconocido: {backtest_score!r} "
            "('directional' | 'intensity' | 'model' | Series por event_id)"
        )
        raise ConfigError(msg)
    if backtest_score == "model" and model_mode != "evaluate":
        msg = (
            "backtest_score='model' exige model_mode='evaluate' (solo las "
            "predicciones OOS de la CV purgada son admisibles como score)"
        )
        raise ConfigError(msg)

    ctx = _event_context(source)
    cal = ctx.calendar or get_calendar()
    events = normalize_events(ctx.events, cal)
    prices = ctx.prices

    # ----------------------------------------------- filtro PIT de universo
    # Sin este filtro, una tabla de eventos construida desde la lista ACTUAL
    # de miembros (p. ej. `load_consensus_events` sobre consenso_master)
    # incluye toda la historia PRE-inclusión de los constituyentes de hoy:
    # el universo efectivo pasa a ser "empresas que ACABARÁN entrando al
    # índice", una selección condicionada al futuro que infla el retorno
    # medio por evento y el hit rate. El filtro exige pertenencia en la
    # propia `event_date` (la sesión negociable del anuncio).
    n_events_outside_universe = 0
    if universe is not None:
        members_cache: dict[object, frozenset[str]] = {}
        keep_mask = np.zeros(len(events), dtype=bool)
        for i, (tkr, d) in enumerate(
            zip(events["ticker"].astype(str), events["event_date"], strict=True)
        ):
            day = pd.Timestamp(d).date()
            members = members_cache.get(day)
            if members is None:
                members = frozenset(str(t) for t in universe.members_on(day))
                members_cache[day] = members
            keep_mask[i] = tkr in members
        n_events_outside_universe = int((~keep_mask).sum())
        events = events.loc[keep_mask].copy()
        if len(events) == 0:
            msg = (
                "el filtro de universo PIT ha descartado todos los eventos: "
                "revisa que los tickers y fechas del calendario casen con el "
                "UniverseProvider"
            )
            raise InsufficientHistory(msg)

    # ------------------------------------------------------------- features
    feats = (
        features_engine
        if features_engine is not None
        else PreEventFeatures(residualize=residualize)
    ).compute(ctx)

    # --------------------------------------------------------------- scores
    intensity_score = (
        intensity if intensity is not None else InformedTradingScore(mode="intensity")
    ).score(feats)
    directional_score = (
        directional if directional is not None else InformedTradingScore(mode="directional")
    ).score(feats)

    # -------------------------------------------------------------- etiquetas
    labels = event_labels(
        events,
        prices,
        market=ctx.market,
        surprise_col=surprise_col,
        return_window=label_window,
    )

    # ---------------------------------------------------------------- modelo
    model: SurpriseModel | None = None
    report: SurpriseModelReport | None = None
    if model_mode != "skip":
        model = surprise_model if surprise_model is not None else SurpriseModel(
            target="surprise", objective="sign", estimator="ridge"
        )
        ycol = _label_column(model)
        usable = labels.dropna(subset=[ycol])
        idx = feats.index.intersection(usable.index)
        if len(idx) == 0:
            msg = (
                "ningún evento tiene a la vez features y etiqueta "
                f"({ycol!r}): no hay nada que modelar"
            )
            raise InsufficientHistory(msg)
        x_mat = feats.loc[idx]
        y = usable.loc[idx, ycol]
        dates_by_event = usable.loc[idx, "event_date"]
        timeline = pd.DatetimeIndex(
            prices.index.get_level_values("date").unique()
        ).sort_values()
        lookahead = max(10, int(label_window[1]))
        if model_mode == "evaluate":
            report = model.evaluate(
                x_mat,
                y,
                dates_by_event,
                n_splits=n_splits,
                embargo=embargo,
                timeline=timeline,
                lookahead=lookahead,
            )
        else:
            model.fit(x_mat, y, dates_by_event)

    # ------------------------------------------------------ score del backtest
    if isinstance(backtest_score, pd.Series):
        score_series = backtest_score.astype(float)
    elif backtest_score == "directional":
        score_series = directional_score
    elif backtest_score == "intensity":
        score_series = intensity_score
    elif backtest_score == "model":
        if report is None:
            msg = (
                "backtest_score='model' exige model_mode='evaluate' (solo las "
                "predicciones OOS de la CV purgada son admisibles como score)"
            )
            raise ConfigError(msg)
        score_series = report.oos_prediction.astype(float)
        if model is not None and model.is_classifier:
            # Probabilidad de signo positivo → score con signo alrededor de 0.5.
            score_series = score_series - 0.5
    else:
        msg = (
            f"backtest_score desconocido: {backtest_score!r} "
            "('directional' | 'intensity' | 'model' | Series por event_id)"
        )
        raise ConfigError(msg)
    score_series = score_series.rename("score")

    # ----------------------------------------------------------------- rejilla
    bt_engine = engine if engine is not None else EventBacktest(calendar=cal)
    grid = run_grid(
        events,
        prices,
        entry_offsets=entry_offsets,
        exit_offsets=exit_offsets,
        score=score_series,
        side=side,
        engine=bt_engine,
        calendar_known_in_advance=calendar_known_in_advance,
        max_concurrent=max_concurrent,
        min_events=min_events,
    )

    # -------------------------------------------------------- estudio de eventos
    ar_frame: pd.DataFrame | None = None
    aar_frame: pd.DataFrame | None = None
    caar_frame: pd.DataFrame | None = None
    groups: pd.Series | None = None
    if event_study:
        ar_frame = abnormal_returns(
            prices,
            events,
            model="market",
            estimation=estimation,
            pre=pre,
            post=post,
            market=ctx.market["log_return"]
            if isinstance(ctx.market, pd.DataFrame) and "log_return" in ctx.market.columns
            else ctx.market,
            cal=cal,
            min_estimation_obs=min_estimation_obs,
        )
        aar_frame = aar_caar(ar_frame)
        studied_ids = pd.Index(ar_frame["event_id"].unique(), name="event_id")
        if isinstance(caar_by, pd.Series):
            values = caar_by.astype(float)
        elif caar_by == "intensity":
            values = intensity_score
        elif caar_by == "directional":
            values = directional_score
        elif caar_by == "model":
            if report is None:
                msg = "caar_by='model' exige model_mode='evaluate'"
                raise ConfigError(msg)
            values = report.oos_prediction.astype(float)
        elif caar_by in events.columns:
            values = pd.Series(
                events[caar_by].to_numpy(dtype=float),
                index=pd.Index(events["event_id"], name="event_id"),
            )
        else:
            msg = (
                f"caar_by={caar_by!r} no es columna de la tabla de eventos ni un "
                "score del pipeline ('intensity'|'directional'|'model')"
            )
            raise ConfigError(msg)
        # Cuantiles SOLO sobre los eventos presentes en el estudio: cuantilar
        # sobre eventos sin AR produciría grupos fantasma bajo `min_group_size`.
        values = values.reindex(studied_ids).dropna()
        groups = quantile_groups(values, caar_quantiles)
        caar_frame = caar_by_group(ar_frame, groups, min_group_size=min_group_size)

    # -------------------------------------------------------------- detección
    detection: dict[str, float] | None = None
    leaked: Sequence[str] | None = leaked_event_ids
    if leaked is None and hasattr(source, "leaked_event_ids"):
        leaked = source.leaked_event_ids()  # type: ignore[attr-defined]
    if leaked is not None:
        leaked_set = set(str(x) for x in leaked)
        finite = intensity_score[np.isfinite(intensity_score.to_numpy(dtype=float))]
        is_leaked = finite.index.to_series().isin(leaked_set).to_numpy()
        n_pos = int(is_leaked.sum())
        n_tot = len(finite)
        detection = {
            "n_events": float(n_tot),
            "n_leaked": float(n_pos),
            "prevalence": float(n_pos / n_tot) if n_tot else float("nan"),
            "auc": float("nan"),
            "auc_se": float("nan"),
            "auc_t": float("nan"),
            "mean_score_leaked": float(finite[is_leaked].mean()) if n_pos else float("nan"),
            "mean_score_other": (
                float(finite[~is_leaked].mean()) if n_tot > n_pos else float("nan")
            ),
        }
        if 0 < n_pos < n_tot:
            auc, se = roc_auc(finite, is_leaked)
            detection["auc"] = auc
            detection["auc_se"] = se
            detection["auc_t"] = (auc - 0.5) / se if se > 0 else float("nan")

    params: dict[str, object] = {
        "residualize": bool(residualize),
        "model_mode": model_mode,
        "n_splits": int(n_splits),
        "embargo": float(embargo),
        "label_window": tuple(int(x) for x in label_window),
        "surprise_col": surprise_col,
        "entry_offsets": [int(x) for x in entry_offsets],
        "exit_offsets": [int(x) for x in exit_offsets],
        "backtest_score": (
            "<series>" if isinstance(backtest_score, pd.Series) else backtest_score
        ),
        "side": side,
        "calendar_known_in_advance": bool(calendar_known_in_advance),
        "event_study": bool(event_study),
        "pre": int(pre),
        "post": int(post),
        "estimation": tuple(int(x) for x in estimation),
        "caar_by": "<series>" if isinstance(caar_by, pd.Series) else caar_by,
        "caar_quantiles": int(caar_quantiles),
        "n_events_input": len(events),
        "n_events_features": len(feats),
        "missing_sources": list(feats.attrs.get("missing_sources", [])),
        "n_grid_trials": len(grid),
        # Auditoría del filtro PIT de universo: sin `universe` los eventos
        # entran TAL CUAL y, si proceden de la lista actual de miembros del
        # índice, el resultado hereda sesgo de selección hacia futuros
        # incluidos (aviso, no error, para no romper el banco sintético).
        "universe_filter": (
            type(universe).__name__
            if universe is not None
            else "SIN FILTRO: eventos no restringidos a pertenencia PIT"
        ),
        "n_events_outside_universe": int(n_events_outside_universe),
    }
    return EventPipelineResult(
        features=feats,
        intensity_score=intensity_score,
        directional_score=directional_score,
        labels=labels,
        model_report=report,
        model=model,
        grid=grid,
        event_study=ar_frame,
        aar=aar_frame,
        caar_by_quintile=caar_frame,
        caar_groups=groups,
        detection=detection,
        params=params,
    )
