"""Factor de cambios de guidance: de la acción categórica al panel numérico.

Implementa la §12.2 de `docs/research/fundamental_factors.md`:

- **Rogers y Van Buskirk (2013)**, *Bundled forecasts in empirical accounting
  research*, JAE 55(1): la gran mayoría de las guías se publican **agrupadas
  con el anuncio de resultados**; el retorno del día del anuncio mezcla la
  sorpresa del trimestre cerrado con la sorpresa de guidance del siguiente, y
  a menudo domina la segunda.
- **Ng, Tuna y Verdi (2013)**, *Management forecast credibility and
  underreaction to news*, RAST 18(4): la deriva posterior a una guía es mayor
  en emisores históricamente creíbles — existe un análogo del PEAD para
  guidance —.
- **Anilowski, Feng y Skinner (2007)**, JAE 44(1-2): la guía a la baja
  agregada se asocia con noticias de beneficios agregadas; señal de *timing*
  más que de sección cruzada.
- La línea de trabajo de Call y coautores sobre la rigidez del hábito de guiar
  respalda tratar el **abandono** de la guía como evento raro y muy
  informativo.

Restricción de datos, vinculante (§12.2)
----------------------------------------
No existe fuente gratuita de guidance estructurada y point-in-time; la vía
realista es parsear los exhibits EX-99.1 de los 8-K item 2.02 (problema NLP
anotado en `docs/OPEN_QUESTIONS.md`). Por eso este módulo define el **esquema**
de la tabla de guidance y el factor sobre ella, y falla con
`ProviderUnavailable` cuando no se aporta tabla: fabricar guidance o devolver
ceros sería inventar datos. Para el ticker presente en la tabla pero sin guía
en una fecha, el factor es **NaN explícito, nunca 0**: cero es "guía
confirmada", NaN es "no sabemos" — confundirlos convierte ausencia de dato en
una posición neutral fabricada (§1.3 del informe).

Esquema de la tabla de guidance
-------------------------------
Columnas mínimas: ``ticker``, ``announced_at`` (UTC) o ``tradable_date`` ya
resuelta, y ``action`` (etiqueta categórica). Opcionales para la sorpresa
numérica `GS`: ``guide_low``, ``guide_high``, ``consensus_prev`` (consenso
vigente ANTES de la guía) y ``price`` (o un panel de precios del que tomar el
cierre previo).
"""

from __future__ import annotations

import warnings
from enum import StrEnum
from typing import ClassVar, Final

import numpy as np
import pandas as pd

from earnings_alpha.errors import ConfigError, DataQualityError, ProviderUnavailable
from earnings_alpha.factors.base import (
    FactorContext,
    register_factor,
    require_columns,
    spread_event_frame,
    spread_event_values,
)
from earnings_alpha.factors.surprise import attach_tradable_date, price_before_event
from earnings_alpha.pit import TradingCalendar

__all__ = [  # noqa: RUF022 - orden temático, no alfabético
    "GuidanceAction",
    "GUIDANCE_SCORES",
    "score_guidance_actions",
    "guidance_surprise",
    "guidance_events",
    "GuidanceChange",
]


class GuidanceAction(StrEnum):
    """Acciones de guidance reconocidas, normalizadas.

    Las etiquetas cubren el ciclo de vida completo de la práctica de guiar,
    incluido su abandono (`STOPPED`), que la literatura señala como el evento
    más informativo por infrecuente.
    """

    RAISED = "raised"
    """Sube el rango guiado."""
    LOWERED = "lowered"
    """Baja el rango guiado."""
    AFFIRMED = "affirmed"
    """Reitera la guía vigente."""
    INITIATED = "initiated"
    """Empieza a guiar (inicia cobertura de guía)."""
    WITHDRAWN = "withdrawn"
    """Retira una guía vigente sin sustituirla (típico en shocks: COVID-2020)."""
    STOPPED = "stopped"
    """Abandona la práctica de guiar (deja de guiar de forma declarada)."""


GUIDANCE_SCORES: Final[dict[GuidanceAction, float]] = {
    GuidanceAction.RAISED: 1.0,
    GuidanceAction.INITIATED: 0.25,
    GuidanceAction.AFFIRMED: 0.0,
    GuidanceAction.STOPPED: -0.5,
    GuidanceAction.WITHDRAWN: -0.75,
    GuidanceAction.LOWERED: -1.0,
}
"""Mapa categórico → numérico, en la escala "mayor = más alcista" del contrato.

El orden relativo está anclado en la evidencia (§12.2): subir la guía es la
noticia positiva fuerte; iniciarla es levemente positivo (compromiso de
transparencia); reiterar es neutral; dejar de guiar y retirar la guía son
negativos —la retirada más que el abandono declarado, porque suele responder a
incertidumbre aguda—; bajarla es la noticia negativa fuerte. Las magnitudes
intermedias (0.25, -0.5, -0.75) son priors de ingeniería **[prior]**, no
cifras publicadas, y deben calibrarse con el arsenal de `stats` antes de usar
el factor en producción."""

_ACTION_SYNONYMS: Final[dict[str, GuidanceAction]] = {
    "raised": GuidanceAction.RAISED,
    "raise": GuidanceAction.RAISED,
    "raises": GuidanceAction.RAISED,
    "up": GuidanceAction.RAISED,
    "upward": GuidanceAction.RAISED,
    "increase": GuidanceAction.RAISED,
    "increased": GuidanceAction.RAISED,
    "lowered": GuidanceAction.LOWERED,
    "lower": GuidanceAction.LOWERED,
    "lowers": GuidanceAction.LOWERED,
    "down": GuidanceAction.LOWERED,
    "downward": GuidanceAction.LOWERED,
    "cut": GuidanceAction.LOWERED,
    "reduced": GuidanceAction.LOWERED,
    "affirmed": GuidanceAction.AFFIRMED,
    "affirm": GuidanceAction.AFFIRMED,
    "affirms": GuidanceAction.AFFIRMED,
    "reaffirmed": GuidanceAction.AFFIRMED,
    "reiterated": GuidanceAction.AFFIRMED,
    "maintained": GuidanceAction.AFFIRMED,
    "unchanged": GuidanceAction.AFFIRMED,
    "initiated": GuidanceAction.INITIATED,
    "initiate": GuidanceAction.INITIATED,
    "initiates": GuidanceAction.INITIATED,
    "new": GuidanceAction.INITIATED,
    "issued": GuidanceAction.INITIATED,
    "withdrawn": GuidanceAction.WITHDRAWN,
    "withdraw": GuidanceAction.WITHDRAWN,
    "withdraws": GuidanceAction.WITHDRAWN,
    "pulled": GuidanceAction.WITHDRAWN,
    "suspended": GuidanceAction.WITHDRAWN,
    "stopped": GuidanceAction.STOPPED,
    "stop": GuidanceAction.STOPPED,
    "discontinued": GuidanceAction.STOPPED,
    "ceased": GuidanceAction.STOPPED,
}


def score_guidance_actions(actions: pd.Series) -> pd.Series:
    """Convierte etiquetas de acción de guidance en puntuaciones numéricas.

    Acepta `GuidanceAction` o texto libre de proveedor (se normaliza contra un
    diccionario de sinónimos). Etiqueta no reconocida o ausente → **NaN
    explícito** con aviso, jamás 0: un 0 significaría "guía reiterada", que es
    información, no ausencia de ella.
    """
    def _score(raw: object) -> float:
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            return np.nan
        if isinstance(raw, GuidanceAction):
            return GUIDANCE_SCORES[raw]
        key = str(raw).strip().lower().replace("_", " ").replace("-", " ")
        key = " ".join(key.split())
        action = _ACTION_SYNONYMS.get(key)
        if action is None:
            try:
                action = GuidanceAction(key)
            except ValueError:
                return np.nan
        return GUIDANCE_SCORES[action]

    out = actions.map(_score).astype(float)
    unknown = out.isna() & actions.notna()
    if bool(unknown.any()):
        sample = sorted({str(v) for v in actions[unknown].unique()})[:5]
        warnings.warn(
            f"{int(unknown.sum())} etiquetas de guidance no reconocidas -> NaN "
            f"(muestra: {sample}); amplía _ACTION_SYNONYMS si son legítimas",
            stacklevel=2,
        )
    out.name = "guidance_score"
    return out


def guidance_surprise(
    guidance: pd.DataFrame,
    *,
    prices: pd.DataFrame | None = None,
    calendar: TradingCalendar | None = None,
) -> pd.Series:
    """Sorpresa de guidance ``GS = (G_mid - C_prev) / P`` (§12.2).

    ``G_mid = (G_low + G_high)/2`` es el punto medio del rango guiado y
    ``C_prev`` el consenso vigente **antes** de la guía — usar el consenso
    posterior sería mirar la reacción de los analistas a la propia guía —.
    El deflactor es la columna ``price`` si existe; si no, el cierre previo al
    `tradable_date` tomado del panel `prices`. `GS` es el análogo de
    `SUE_analyst` mirando hacia delante y a menudo domina el retorno del
    anuncio (Rogers y Van Buskirk 2013: guías *bundled*).

    Filas sin rango o sin consenso previo → NaN explícito.
    """
    require_columns(guidance, ["ticker", "guide_low", "guide_high", "consensus_prev"],
                    name="guidance")
    work = attach_tradable_date(guidance, calendar)
    mid = (work["guide_low"].astype(float) + work["guide_high"].astype(float)) / 2.0
    if "price" in work.columns:
        px = work["price"].astype(float)
    elif prices is not None:
        px = price_before_event(prices, work)
    else:
        msg = (
            "guidance_surprise necesita una columna `price` o un panel de "
            "precios del que tomar el cierre previo"
        )
        raise ConfigError(msg)
    gs = (mid - work["consensus_prev"].astype(float)) / px.where(px > 0)
    gs.name = "guidance_surprise"
    return gs


def guidance_events(
    guidance: pd.DataFrame,
    *,
    prices: pd.DataFrame | None = None,
    calendar: TradingCalendar | None = None,
) -> pd.DataFrame:
    """Normaliza una tabla de guidance a eventos con puntuación y fecha PIT.

    Devuelve columnas ``ticker, tradable_date, action_score, gs, value`` donde
    `value` es `action_score` y, si hay datos numéricos, `gs` queda disponible
    como alternativa. La fecha negociable se deriva con la misma política que
    los anuncios de resultados (las guías van *bundled* con ellos: Rogers y
    Van Buskirk 2013), reutilizando `announced_at`/`session` vía
    `pit.tradable_date`; UNKNOWN se trata como AMC.
    """
    require_columns(guidance, ["ticker", "action"], name="guidance")
    work = attach_tradable_date(guidance, calendar)
    work["action_score"] = score_guidance_actions(work["action"])
    has_range = {"guide_low", "guide_high", "consensus_prev"} <= set(work.columns)
    if has_range:
        work["gs"] = guidance_surprise(work, prices=prices, calendar=calendar)
    else:
        work["gs"] = np.nan
    work["value"] = work["action_score"]
    cols = ["ticker", "tradable_date", "action_score", "gs", "value"]
    extra = [c for c in ("event_id", "period_end", "session") if c in work.columns]
    return work[[*extra, *cols]]


@register_factor()
class GuidanceChange:
    """Factor `guidance_change`: cambios de guidance como señal numérica.

    Referencias: Rogers y Van Buskirk (2013); Ng, Tuna y Verdi (2013);
    Anilowski, Feng y Skinner (2007). Mayor puntuación (guía al alza) = más
    alcista; véase `GUIDANCE_SCORES` para la escala y sus priors.

    Comportamiento con datos ausentes, en orden de severidad:

    - **Sin tabla de guidance** (``guidance=None``): `ProviderUnavailable`.
      No hay fuente PIT gratuita (§12.2) y este factor no la inventa; el proxy
      barato documentado en el informe —revisión del consenso FY1 en
      ``[+1, +5]`` sesiones tras el anuncio— vive en `factors.revisions`.
    - **Ticker sin ninguna guía**: NaN en todo su historial.
    - **Ticker con guías, fechas fuera de la ventana de vida**: NaN (no 0):
      la guía caduca, no se convierte en "confirmada perpetua".
    - **Etiqueta no reconocida**: NaN con aviso (`score_guidance_actions`).

    `value_col` elige entre la puntuación categórica (``"action_score"``,
    por defecto) y la sorpresa numérica de guidance (``"gs"``) cuando la tabla
    trae rangos y consenso previo.
    """

    name = "guidance_change"
    requires: ClassVar[list[str]] = ["guidance"]

    def __init__(
        self,
        guidance: pd.DataFrame | None = None,
        *,
        value_col: str = "action_score",
        horizon: int = 20,
        decay: float | None = 10.0,
        prices_for_gs: bool = True,
    ) -> None:
        if value_col not in ("action_score", "gs"):
            msg = f"value_col debe ser 'action_score' o 'gs'; recibido {value_col!r}"
            raise ConfigError(msg)
        if horizon < 1:
            msg = f"horizon debe ser >= 1; recibido {horizon}"
            raise ConfigError(msg)
        self.guidance = guidance
        self.value_col = value_col
        self.horizon = horizon
        self.decay = decay
        self.prices_for_gs = prices_for_gs

    def event_values(self, ctx: FactorContext) -> pd.DataFrame:
        if self.guidance is None or len(self.guidance) == 0:
            raise ProviderUnavailable(
                "guidance",
                "no hay tabla de guidance point-in-time; no existe fuente gratuita "
                "estructurada (fundamental_factors.md §12.2) y el factor no inventa "
                "datos. Alternativa documentada: parsear EX-99.1 de los 8-K 2.02 "
                "(docs/OPEN_QUESTIONS.md) o usar el proxy de revisiones post-anuncio",
            )
        prices = ctx.prices if (self.prices_for_gs and len(ctx.prices)) else None
        table = guidance_events(self.guidance, prices=prices, calendar=ctx.calendar)
        if table[self.value_col].isna().all():
            msg = (
                f"la tabla de guidance no produjo ningún valor en {self.value_col!r}; "
                "revisa las etiquetas de acción o los campos numéricos"
            )
            raise DataQualityError(msg)
        return table

    def compute_frame(self, ctx: FactorContext) -> pd.DataFrame:
        """Panel paso-mantenido con `available_at`, auditable; sin decaimiento."""
        table = self.event_values(ctx)
        return spread_event_frame(
            table,
            ctx.dates,
            ctx.tickers(),
            value_col=self.value_col,
            max_staleness_days=None,
            name=self.name,
        )

    def compute(self, ctx: FactorContext) -> pd.Series:
        """Serie ``(date, ticker)``: puntuación viva `horizon` sesiones.

        Tras la ventana el valor vuelve a NaN (la guía caduca); antes de la
        primera guía del ticker es NaN. El decaimiento exponencial refleja el
        horizonte 5-20 sesiones del informe (§14).
        """
        table = self.event_values(ctx)
        return spread_event_values(
            table,
            ctx.dates,
            ctx.tickers(),
            value_col=self.value_col,
            horizon=self.horizon,
            decay=self.decay,
            include_event_day=True,
            dead_value=np.nan,
            calendar=ctx.calendar,
            name=self.name,
        )
