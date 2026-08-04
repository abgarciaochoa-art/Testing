"""Factores de revisiones de analistas: momentum, difusión y dispersión.

Implementa la §5 (revisiones) y la §2.6 (dispersión) de
`docs/research/fundamental_factors.md`:

- **Momentum de revisiones** (Chan, Jegadeesh y Lakonishok 1996, *Momentum
  Strategies*, JF 51(5)): suma móvil de las variaciones del consenso, cada una
  deflactada por el precio **de su propio momento** — deflactar por el precio
  actual reintroduciría el momentum de precio (§5.1). Ventanas de 1 y 3 meses
  (21 y 63 sesiones): el IC de este factor es máximo a un mes y se agota hacia
  los 6 (§5.5).
- **Difusión de revisiones** (§5.2): ``(N_up - N_down) / N_total`` en ventana
  móvil, acotado en [-1, 1]. Sin deflactor, inmune a errores de ajuste por
  split. Regla de higiene: mínimo 3 revisiones en la ventana.
- **Dispersión de analistas** (Diether, Malloy y Scherbina 2002, *Differences
  of Opinion and the Cross Section of Stock Returns*, JF 57(5)): mayor
  dispersión predice **menor** rentabilidad, así que el factor se devuelve con
  el **signo cambiado** para cumplir el contrato "mayor = más alcista".

Requisito de datos, no negociable
---------------------------------
El momentum y la difusión de revisiones exigen **fotos históricas del
consenso** (`types.EstimateSnapshot`: varias filas `as_of` por
``(ticker, period_end)``). El dataset real del repo,
``consenso_master.parquet``, trae únicamente el consenso *final* previo a cada
anuncio — sin vintages —, y reconstruir revisiones desde un consenso final es
look-ahead de manual (§5.5 del informe y advertencia PIT del dataset). Por eso
estos factores **detectan** la ausencia de vintages y devuelven un panel de
NaN emitiendo `NoVintagesWarning`; inventar revisiones está prohibido.

Point-in-time
-------------
Una foto de consenso fechada `as_of` se considera explotable a partir de la
**sesión siguiente** a `as_of` (no se conoce la hora del día en que el
agregador la consolidó; asumir la propia sesión sería la política optimista).
Los precios deflactores son los de la última sesión conocida en el momento de
la revisión.
"""

from __future__ import annotations

import warnings
from typing import ClassVar, Final

import numpy as np
import pandas as pd

from earnings_alpha.errors import ConfigError, DataQualityError
from earnings_alpha.factors.base import (
    FactorContext,
    build_panel_index,
    register_factor,
    require_columns,
)

__all__ = [  # noqa: RUF022 - orden temático, no alfabético
    "NoVintagesWarning",
    "has_revision_vintages",
    "revision_momentum_panel",
    "revision_diffusion_panel",
    "analyst_dispersion_panel",
    "RevisionMomentum",
    "RevisionDiffusion",
    "AnalystDispersion",
    "SESSIONS_PER_MONTH",
]

SESSIONS_PER_MONTH: Final[int] = 21
"""Sesiones por mes natural (convención de la literatura de momentum)."""

DEFAULT_STALENESS_SESSIONS: Final[int] = 130
"""Sesiones sin ninguna foto de consenso tras las cuales el factor decae a
NaN: ~6 meses; un consenso más viejo ya no describe a la empresa."""


class NoVintagesWarning(UserWarning):
    """El panel de estimaciones no contiene fotos históricas (vintages).

    Sin al menos dos `as_of` por ``(ticker, period_end)`` no existen
    revisiones observables y el factor devuelve NaN. Es la situación de
    ``consenso_master.parquet`` (consenso final, sin historial): véase
    `factors.surprise.CONSENSUS_PIT_WARNING`.
    """


# ---------------------------------------------------------------------------
# Detección y preparación
# ---------------------------------------------------------------------------


def has_revision_vintages(estimates: pd.DataFrame) -> bool:
    """True si alguna clave ``(ticker, period_end)`` tiene ≥2 fotos `as_of`.

    Es la comprobación que separa un histórico de consenso genuino (apto para
    momentum de revisiones) de un volcado de consensos finales tipo
    ``consenso_master``, con el que solo puede calcularse SUE de analistas.
    """
    require_columns(estimates, ["ticker", "period_end", "as_of"], name="estimates")
    if len(estimates) == 0:
        return False
    counts = estimates.groupby(["ticker", "period_end"])["as_of"].nunique()
    return bool((counts >= 2).any())


def _warn_and_nan(
    dates: pd.DatetimeIndex, tickers: list[str], name: str, reason: str
) -> pd.Series:
    warnings.warn(
        f"{name}: {reason}; se devuelve un panel de NaN — no se inventan "
        "revisiones (véase CONSENSUS_PIT_WARNING en factors.surprise)",
        NoVintagesWarning,
        stacklevel=3,
    )
    return pd.Series(
        np.nan, index=build_panel_index(dates, tickers), name=name, dtype=float
    )


def _grid_and_tickers(
    dates: pd.DatetimeIndex, tickers: list[str]
) -> tuple[pd.DatetimeIndex, list[str]]:
    grid = pd.DatetimeIndex(pd.to_datetime(dates)).normalize().unique().sort_values().as_unit("ns")
    if len(grid) == 0:
        msg = "la rejilla de fechas está vacía"
        raise DataQualityError(msg)
    return grid, sorted(set(tickers))


def _snapshot_deltas(estimates: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Variaciones del consenso entre fotos consecutivas del mismo periodo.

    Devuelve columnas ``ticker, as_of, prev_as_of, delta``. Restringir la
    diferencia al mismo ``(ticker, period_end)`` evita el salto mecánico de
    nivel cuando el periodo objetivo rueda de un trimestre al siguiente, que
    no es una revisión sino un cambio de base.
    """
    est = estimates[["ticker", "period_end", "as_of", value_col]].copy()
    est["as_of"] = pd.DatetimeIndex(pd.to_datetime(est["as_of"])).normalize()
    est["period_end"] = pd.DatetimeIndex(pd.to_datetime(est["period_end"])).normalize()
    est = est.dropna(subset=["as_of", value_col])
    est = est.sort_values(["ticker", "period_end", "as_of"], kind="mergesort")
    grouped = est.groupby(["ticker", "period_end"], sort=False)[value_col]
    est["delta"] = grouped.diff()
    est["prev_as_of"] = est.groupby(["ticker", "period_end"], sort=False)["as_of"].shift(1)
    out = est.dropna(subset=["delta"])[["ticker", "as_of", "prev_as_of", "delta"]]
    return out.reset_index(drop=True)


def _coverage_mask(
    estimates: pd.DataFrame,
    grid: pd.DatetimeIndex,
    tickers: list[str],
    *,
    staleness_sessions: int,
) -> np.ndarray:
    """Matriz booleana (n_fechas x n_tickers): hay consenso vivo en esa fecha.

    Falsa antes de la primera foto observable del ticker y cuando la última
    foto observada tiene más de `staleness_sessions` sesiones de antigüedad.
    """
    n, m = len(grid), len(tickers)
    last_seen = np.full((n, m), -(10**9), dtype=np.int64)
    est = estimates.dropna(subset=["as_of"])
    obs_pos = np.searchsorted(
        grid.to_numpy(dtype="datetime64[ns]"),
        pd.DatetimeIndex(pd.to_datetime(est["as_of"]))
        .normalize()
        .to_numpy(dtype="datetime64[ns]"),
        side="right",  # explotable desde la sesión SIGUIENTE a as_of
    )
    col_of = {t: j for j, t in enumerate(tickers)}
    cols = np.array([col_of.get(t, -1) for t in est["ticker"].astype(str)], dtype=np.int64)
    ok = (cols >= 0) & (obs_pos < n)
    np.maximum.at(last_seen, (obs_pos[ok], cols[ok]), obs_pos[ok])
    last_seen = np.maximum.accumulate(last_seen, axis=0)
    ages = np.arange(n)[:, None] - last_seen
    return (last_seen >= 0) & (ages <= staleness_sessions)


def _accumulate(
    grid: pd.DatetimeIndex,
    tickers: list[str],
    when: pd.Series,
    ticker_col: pd.Series,
    values: np.ndarray,
) -> np.ndarray:
    """Suma `values` en la celda (primera sesión posterior a `when`, ticker)."""
    n, m = len(grid), len(tickers)
    out = np.zeros((n, m), dtype=float)
    pos = np.searchsorted(
        grid.to_numpy(dtype="datetime64[ns]"),
        pd.DatetimeIndex(pd.to_datetime(when)).normalize().to_numpy(dtype="datetime64[ns]"),
        side="right",
    )
    col_of = {t: j for j, t in enumerate(tickers)}
    cols = np.array([col_of.get(t, -1) for t in ticker_col.astype(str)], dtype=np.int64)
    ok = (cols >= 0) & (pos < n) & np.isfinite(values)
    np.add.at(out, (pos[ok], cols[ok]), values[ok])
    return out


def _rolling_sum(matrix: np.ndarray, window: int) -> np.ndarray:
    """Suma móvil por columnas con ventana en sesiones (expansiva al inicio)."""
    cum = np.cumsum(matrix, axis=0)
    out = cum.copy()
    if window < len(matrix):
        out[window:] = cum[window:] - cum[:-window]
    return out


# ---------------------------------------------------------------------------
# Paneles
# ---------------------------------------------------------------------------


def revision_momentum_panel(
    estimates: pd.DataFrame,
    prices: pd.DataFrame,
    dates: pd.DatetimeIndex,
    tickers: list[str],
    *,
    months: int = 3,
    value_col: str = "eps_mean",
    staleness_sessions: int = DEFAULT_STALENESS_SESSIONS,
    name: str | None = None,
) -> pd.Series:
    """Momentum de revisiones del consenso deflactado por precio (§5.1).

    ::

        REV_{i,t} = sum_{as_of en (t - 21·months, t]}  ΔF_{i,as_of} / P_{i,as_of}

    con ``ΔF`` la variación entre fotos consecutivas del consenso del mismo
    ``(ticker, period_end)`` y ``P`` el cierre de la última sesión conocida en
    el momento de la revisión — **no** el precio actual: Chan, Jegadeesh y
    Lakonishok (1996) deflactan cada revisión por el precio de su propio
    momento precisamente para no reintroducir momentum de precio.

    Semántica del panel:

    - Sin vintages en `estimates` → `NoVintagesWarning` y panel de NaN
      (situación de ``consenso_master.parquet``).
    - Ticker con cobertura viva y ventana sin revisiones → **0** (el consenso
      se mantuvo: información real, no ausencia de dato).
    - Ticker sin foto previa, o con la última foto más vieja que
      `staleness_sessions` → NaN.
    """
    if months < 1:
        msg = f"months debe ser >= 1; recibido {months}"
        raise ConfigError(msg)
    grid, ticker_list = _grid_and_tickers(dates, tickers)
    factor_name = name or f"revision_momentum_{months}m"
    require_columns(estimates, ["ticker", "period_end", "as_of", value_col], name="estimates")

    if not has_revision_vintages(estimates):
        return _warn_and_nan(
            grid,
            ticker_list,
            factor_name,
            "el panel de estimaciones no trae fotos históricas del consenso "
            "(una única `as_of` por (ticker, period_end))",
        )
    require_columns(prices, ["close"], name="prices")

    deltas = _snapshot_deltas(estimates, value_col)
    if len(deltas) == 0:
        return _warn_and_nan(
            grid, ticker_list, factor_name, "no hay ninguna variación de consenso observable"
        )

    # Precio en el momento de la revisión: última sesión <= as_of.
    wide_close = prices["close"].unstack("ticker").reindex(grid)  # noqa: PD010
    price_grid = wide_close.to_numpy(dtype=float)
    pos_price = np.searchsorted(
        grid.to_numpy(dtype="datetime64[ns]"),
        pd.DatetimeIndex(deltas["as_of"]).to_numpy(dtype="datetime64[ns]"),
        side="right",
    ) - 1
    col_idx = wide_close.columns.get_indexer(deltas["ticker"].astype(str))
    ok = (pos_price >= 0) & (col_idx >= 0)
    px = np.full(len(deltas), np.nan)
    px[ok] = price_grid[pos_price[ok], col_idx[ok]]
    scaled = deltas["delta"].to_numpy(dtype=float) / np.where(px > 0, px, np.nan)

    window = SESSIONS_PER_MONTH * months
    matrix = _accumulate(grid, ticker_list, deltas["as_of"], deltas["ticker"], scaled)
    rolled = _rolling_sum(matrix, window)
    alive = _coverage_mask(
        estimates, grid, ticker_list, staleness_sessions=staleness_sessions
    )
    rolled = np.where(alive, rolled, np.nan)
    out = pd.Series(
        rolled.reshape(-1),
        index=build_panel_index(grid, ticker_list),
        name=factor_name,
        dtype=float,
    )
    return out


def revision_diffusion_panel(
    estimates: pd.DataFrame,
    dates: pd.DatetimeIndex,
    tickers: list[str],
    *,
    months: int = 3,
    value_col: str = "eps_mean",
    min_revisions: int = 3,
    tol: float = 1e-6,
    staleness_sessions: int = DEFAULT_STALENESS_SESSIONS,
    name: str | None = None,
) -> pd.Series:
    """Índice de difusión de revisiones (§5.2)::

        DIFF_{i,t} = ( N_up - N_down ) / N_total     en (t - 21·months, t]

    acotado en [-1, +1]. No necesita deflactor y es inmune a errores de ajuste
    por split; su granularidad es pobre con pocos analistas, pero en el
    S&P 500 la cobertura la resuelve (§5.2: es de los pocos factores que
    mejora, no empeora, en nuestro universo).

    **Aproximación documentada:** el índice canónico cuenta revisiones de
    analistas individuales; con fotos de consenso solo son observables los
    movimientos del consenso agregado, que es lo que aquí se cuenta. La regla
    de higiene del informe se conserva: menos de `min_revisions` movimientos
    en la ventana → NaN. Cambios de magnitud ≤ `tol` no cuentan como revisión
    (ruido numérico del agregador, no una opinión nueva).

    Sin vintages → `NoVintagesWarning` y panel de NaN, como el momentum.
    """
    if months < 1:
        msg = f"months debe ser >= 1; recibido {months}"
        raise ConfigError(msg)
    if min_revisions < 1:
        msg = f"min_revisions debe ser >= 1; recibido {min_revisions}"
        raise ConfigError(msg)
    grid, ticker_list = _grid_and_tickers(dates, tickers)
    factor_name = name or f"revision_diffusion_{months}m"
    require_columns(estimates, ["ticker", "period_end", "as_of", value_col], name="estimates")

    if not has_revision_vintages(estimates):
        return _warn_and_nan(
            grid,
            ticker_list,
            factor_name,
            "el panel de estimaciones no trae fotos históricas del consenso",
        )

    deltas = _snapshot_deltas(estimates, value_col)
    if len(deltas) == 0:
        return _warn_and_nan(
            grid, ticker_list, factor_name, "no hay ninguna variación de consenso observable"
        )
    d = deltas["delta"].to_numpy(dtype=float)
    ups = (d > tol).astype(float)
    downs = (d < -tol).astype(float)
    counted = ups + downs

    window = SESSIONS_PER_MONTH * months
    up_m = _rolling_sum(_accumulate(grid, ticker_list, deltas["as_of"], deltas["ticker"], ups),
                        window)
    down_m = _rolling_sum(
        _accumulate(grid, ticker_list, deltas["as_of"], deltas["ticker"], downs), window
    )
    total_m = _rolling_sum(
        _accumulate(grid, ticker_list, deltas["as_of"], deltas["ticker"], counted), window
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        diff = (up_m - down_m) / total_m
    diff = np.where(total_m >= min_revisions, diff, np.nan)
    alive = _coverage_mask(
        estimates, grid, ticker_list, staleness_sessions=staleness_sessions
    )
    diff = np.where(alive, diff, np.nan)
    return pd.Series(
        diff.reshape(-1),
        index=build_panel_index(grid, ticker_list),
        name=factor_name,
        dtype=float,
    )


def analyst_dispersion_panel(
    estimates: pd.DataFrame,
    dates: pd.DatetimeIndex,
    tickers: list[str],
    *,
    prices: pd.DataFrame | None = None,
    floor_frac: float = 0.005,
    abs_floor: float = 0.01,
    staleness_sessions: int = DEFAULT_STALENESS_SESSIONS,
    signed: bool = True,
    name: str = "analyst_dispersion",
) -> pd.Series:
    """Dispersión de analistas, con el signo del contrato del repo.

    Diether, Malloy y Scherbina (2002) documentan que **mayor dispersión de
    previsiones predice menor rentabilidad** (proxy de diferencias de opinión
    con restricciones al corto). Como el contrato §3.4 exige "valor mayor =
    más alcista", el factor devuelve por defecto ``-dispersión``
    (`signed=True`); con `signed=False` devuelve la dispersión cruda para
    usarla como control (imprescindible si `SUE_disp` entra en la combinación,
    §2.6 del informe).

    ::

        disp_{i,t} = eps_std_{i,t} / max( |consenso_{i,t}|, suelo )

    con el consenso y su desviación tomados de la última foto **observable**
    en `t` (as-of estricto: fotos de `as_of >= t` no se usan). El suelo del
    denominador es ``max(abs_floor, floor_frac · P_t)`` si hay precios, o
    `abs_floor` si no: sin suelo, un consenso ≈ 0 (biotech, cíclicas en el
    suelo del ciclo) fabrica dispersiones infinitas correlacionadas con el
    sector, la misma patología que `SurpriseBasis.ABS_ESTIMATE` (§2.4).

    A diferencia del momentum, la dispersión sí es computable con una única
    foto por periodo — es un nivel, no una revisión — así que **no** exige
    vintages; sí exige `eps_std`.
    """
    grid, ticker_list = _grid_and_tickers(dates, tickers)
    require_columns(estimates, ["ticker", "period_end", "as_of", "eps_std"], name="estimates")
    value_col = "eps_median" if "eps_median" in estimates.columns else "eps_mean"
    if value_col not in estimates.columns:
        msg = "`estimates` debe traer `eps_median` o `eps_mean` para el denominador"
        raise DataQualityError(msg)

    est = estimates[["ticker", "as_of", value_col, "eps_std"]].copy()
    est["as_of"] = pd.DatetimeIndex(pd.to_datetime(est["as_of"])).normalize()
    est = est.dropna(subset=["as_of", "eps_std"]).sort_values("as_of", kind="mergesort")

    n, m = len(grid), len(ticker_list)
    col_of = {t: j for j, t in enumerate(ticker_list)}
    grid_np = grid.to_numpy(dtype="datetime64[ns]")
    # Observable desde la sesión siguiente a `as_of` (misma política que las
    # revisiones); la última foto gana dentro de cada celda.
    pos = np.searchsorted(grid_np, est["as_of"].to_numpy(dtype="datetime64[ns]"), side="right")
    cols = np.array([col_of.get(t, -1) for t in est["ticker"].astype(str)], dtype=np.int64)
    ok = (cols >= 0) & (pos < n)

    std_grid = np.full((n, m), np.nan)
    mean_grid = np.full((n, m), np.nan)
    std_grid[pos[ok], cols[ok]] = est["eps_std"].to_numpy(dtype=float)[ok]
    mean_grid[pos[ok], cols[ok]] = est[value_col].to_numpy(dtype=float)[ok]

    # Arrastre hacia delante con tope de obsolescencia, columna a columna.
    filled_std = pd.DataFrame(std_grid).ffill(limit=staleness_sessions).to_numpy()
    filled_mean = pd.DataFrame(mean_grid).ffill(limit=staleness_sessions).to_numpy()

    floor = np.full((n, m), abs_floor, dtype=float)
    if prices is not None:
        require_columns(prices, ["close"], name="prices")
        wide_close = prices["close"].unstack("ticker").reindex(grid)  # noqa: PD010
        aligned = wide_close.reindex(columns=ticker_list).to_numpy(dtype=float)
        with np.errstate(invalid="ignore"):
            floor = np.fmax(floor, floor_frac * aligned)

    denom = np.fmax(np.abs(filled_mean), floor)
    with np.errstate(invalid="ignore", divide="ignore"):
        disp = filled_std / denom
    values = -disp if signed else disp
    return pd.Series(
        values.reshape(-1),
        index=build_panel_index(grid, ticker_list),
        name=name,
        dtype=float,
    )


# ---------------------------------------------------------------------------
# Factores registrados
# ---------------------------------------------------------------------------


@register_factor()
class RevisionMomentum:
    """Factor `revision_momentum`: variación del consenso deflactada por precio.

    Referencia: Chan, Jegadeesh y Lakonishok (1996). Mayor revisión al alza =
    más alcista. Factor **rápido** (vida media 1-3 meses, §5.5); debe
    ortogonalizarse contra momentum de precio antes de combinar, o se estará
    comprando momentum con etiqueta fundamental. Sin vintages en
    `ctx.estimates` (caso `consenso_master`) emite `NoVintagesWarning` y
    devuelve NaN.
    """

    name = "revision_momentum"
    requires: ClassVar[list[str]] = ["estimates", "prices"]

    def __init__(self, months: int = 3, *, value_col: str = "eps_mean") -> None:
        self.months = months
        self.value_col = value_col
        self.name = f"revision_momentum_{months}m"

    def compute(self, ctx: FactorContext) -> pd.Series:
        ctx.require("estimates", "prices")
        return revision_momentum_panel(
            ctx.estimates,
            ctx.prices,
            ctx.dates,
            ctx.tickers(),
            months=self.months,
            value_col=self.value_col,
            name=self.name,
        )


@register_factor()
class RevisionDiffusion:
    """Factor `revision_diffusion`: amplitud de las revisiones del consenso.

    Referencia: §5.2 del informe (índice de difusión clásico de la literatura
    de revisiones; Chan, Jegadeesh y Lakonishok 1996 para el fenómeno de
    fondo). Mayor difusión al alza = más alcista. Sin vintages →
    `NoVintagesWarning` y NaN.
    """

    name = "revision_diffusion"
    requires: ClassVar[list[str]] = ["estimates"]

    def __init__(
        self, months: int = 3, *, min_revisions: int = 3, value_col: str = "eps_mean"
    ) -> None:
        self.months = months
        self.min_revisions = min_revisions
        self.value_col = value_col
        self.name = f"revision_diffusion_{months}m"

    def compute(self, ctx: FactorContext) -> pd.Series:
        ctx.require("estimates")
        return revision_diffusion_panel(
            ctx.estimates,
            ctx.dates,
            ctx.tickers(),
            months=self.months,
            min_revisions=self.min_revisions,
            value_col=self.value_col,
            name=self.name,
        )


@register_factor()
class AnalystDispersion:
    """Factor `analyst_dispersion`: -dispersión de previsiones (signo DMS).

    Referencia: Diether, Malloy y Scherbina (2002). El signo negativo cumple
    el contrato "mayor = más alcista": las empresas con analistas de acuerdo
    puntúan arriba. Documentado aquí porque un signo invertido produce un
    backtest especular que se lee como "el factor no funciona" cuando funciona
    al revés (§7.4 del informe).
    """

    name = "analyst_dispersion"
    requires: ClassVar[list[str]] = ["estimates", "prices"]

    def __init__(self, *, floor_frac: float = 0.005, abs_floor: float = 0.01) -> None:
        self.floor_frac = floor_frac
        self.abs_floor = abs_floor

    def compute(self, ctx: FactorContext) -> pd.Series:
        ctx.require("estimates")
        prices = ctx.prices if ctx.prices is not None and len(ctx.prices) else None
        return analyst_dispersion_panel(
            ctx.estimates,
            ctx.dates,
            ctx.tickers(),
            prices=prices,
            floor_frac=self.floor_frac,
            abs_floor=self.abs_floor,
        )


def _register_aliases() -> None:
    @register_factor("revision_momentum_1m")
    def _rev1(**kwargs: object) -> RevisionMomentum:
        return RevisionMomentum(months=1, **kwargs)  # type: ignore[arg-type]

    @register_factor("revision_momentum_3m")
    def _rev3(**kwargs: object) -> RevisionMomentum:
        return RevisionMomentum(months=3, **kwargs)  # type: ignore[arg-type]


_register_aliases()
