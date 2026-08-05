"""Motor de backtest cross-section (contrato `ARCHITECTURE.md` §3.7).

`CrossSectionalBacktest.run` transforma una señal en panel `(date, ticker)` en
una cartera de cubos larga-corta rebalanceada a la frecuencia pedida, con las
cuatro defensas que separan un backtest honesto de uno decorativo:

1. **Retardo de ejecución.** La señal fechada en `t` es conocible, como pronto,
   al cierre de `t`; la ejecución por defecto es en la **apertura de la sesión
   siguiente** (`execution="next_open"`). Ejecutar al cierre de `t` con datos de
   `t` es look-ahead (`validation_methodology.md` §11.1, Etapa 0 del protocolo),
   y por eso ni siquiera se ofrece como opción.
2. **Universo point-in-time.** En cada fecha de decisión solo son elegibles los
   miembros del índice **ese día** (`universe`), nunca la lista de hoy
   (`pit_and_biases.md` §B1); un nombre que sale del universo se vende en el
   siguiente rebalanceo.
3. **Delisting ejecutado, no evaporado.** Una serie de precios que termina con
   la posición viva se liquida ejecutando su última barra, aplicando el retorno
   de delisting si se conoce (`delist_returns`), y queda **registrada** en
   `BacktestResult.silent_delistings` si no: es la auditoría de
   `pit_and_biases.md` §6.4, porque dejar evaporarse posiciones al último precio
   es exactamente el sesgo de delisting (Shumway 1997).
4. **Costes desglosados.** Spread por tramo de liquidez, impacto √participación,
   comisión y préstamo de los cortos (`costs.CostModel`), con ADV y volatilidad
   calculados **hacia atrás** (`pit_and_biases.md` §12.6: un ADV de muestra
   completa usa volumen futuro).

Contabilidad del día de ejecución (con `execution="next_open"`)::

    cierre t-1 ──(overnight, pesos viejos)──► apertura t ──(trade)──► cierre t
    1 + r_t = (1 + Σ w·r_on)·(1 + Σ w'·r_id) − costes_t

Los pesos son fracciones del NAV corriente y derivan con los precios entre
rebalanceos (dividir por `1 + r_p` los mantiene como fracciones del NAV); los
costes se restan aritméticamente del retorno del día, y el efecto de segundo
orden de pagar el coste sobre un NAV ya movido (≤ coste², sub-punto-básico) se
ignora de forma declarada.

La atribución sectorial reparte `Σ w·r` por sector con la aproximación
aritmética del retorno bruto (la diferencia con el compuesto del día de
ejecución es el término cruzado `(Σw·r_on)·(Σw'·r_id)`, de segundo orden).

Referencias
-----------
- Fama, E. y French, K. (1993). *Common Risk Factors...*. JFE 33 (carteras por
  cubos de características).
- Shumway, T. (1997). *The Delisting Bias in CRSP Data*. JF 52.
- Grinold, R. y Kahn, R. (2000). *Active Portfolio Management*, cap. 14 y 16.
- `docs/research/validation_methodology.md` §11 (construcción, ponderación,
  retardo de ejecución) y `docs/research/pit_and_biases.md` §6, §12.6.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from earnings_alpha.backtest.costs import CostBreakdown, CostModel
from earnings_alpha.backtest.portfolio import assign_quantiles, build_target_weights
from earnings_alpha.errors import (
    ConfigError,
    DataQualityError,
    InsufficientHistory,
    UniverseError,
)
from earnings_alpha.signals import check_panel

__all__ = [  # noqa: RUF022 - orden temático, no alfabético
    "ExecutionTiming",
    "rebalance_schedule",
    "BacktestResult",
    "CrossSectionalBacktest",
]

ExecutionTiming = Literal["next_open", "next_close"]
"""Cuándo se ejecuta la señal fechada en `t`:

- ``"next_open"`` (defecto): apertura de la sesión siguiente. Es el supuesto
  realista para una señal calculada tras el cierre.
- ``"next_close"``: cierre de la sesión siguiente; más conservador todavía (un
  día entero de retardo). No existe ejecución al cierre de `t`: sería look-ahead.
"""

_COST_COLS = ("spread", "impact", "commission", "borrow")


def rebalance_schedule(sessions: pd.DatetimeIndex, rule: str) -> pd.DatetimeIndex:
    """Fechas de decisión: la última sesión de cada periodo de `rule`.

    `rule` admite ``"D"`` (diaria), ``"W-FRI"`` y demás anclas semanales
    (``"W"`` equivale a ``"W-FRI"``, la convención del repo), ``"M"``/``"ME"``
    (última sesión del mes) y ``"Q"``/``"QE"`` (del trimestre). La decisión en
    la última sesión del periodo con ejecución en la siguiente reproduce el
    "señal del viernes, ejecución el lunes" estándar.
    """
    if not isinstance(sessions, pd.DatetimeIndex) or len(sessions) == 0:
        msg = "sessions debe ser un DatetimeIndex no vacío"
        raise ConfigError(msg)
    r = rule.strip().upper()
    if r in {"D", "B", "DAILY"}:
        return sessions
    if r in {"W", "WEEKLY"}:
        r = "W-FRI"
    elif r in {"M", "ME", "MONTHLY"}:
        r = "M"
    elif r in {"Q", "QE", "QUARTERLY"}:
        r = "Q"
    try:
        periods = sessions.to_period(r)
    except (ValueError, pd.errors.ParserError) as exc:  # alias no reconocido
        msg = (
            f"regla de rebalanceo no reconocida: {rule!r}; usa 'D', 'W-FRI' "
            "(u otra ancla semanal), 'M' o 'Q'"
        )
        raise ConfigError(msg) from exc
    codes = periods.asi8
    last_of_period = np.r_[codes[1:] != codes[:-1], True]
    return sessions[last_of_period]


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Resultado completo de un backtest cross-section.

    Todas las series diarias empiezan en la **primera fecha de ejecución** (antes
    no hay cartera y un tramo de ceros diluiría el Sharpe). Los pesos son fin de
    día, fracción del NAV; `trades` y `turnover` viven en las fechas de
    ejecución. Los costes están en fracción del NAV del día, desglosados en las
    cuatro componentes del `CostModel`.
    """

    returns: pd.Series
    """Retorno diario neto de costes."""
    gross_returns: pd.Series
    """Retorno diario bruto (incluye retornos de delisting; sin costes)."""
    weights: pd.DataFrame
    """Pesos fin de día (fracción del NAV), tras deriva, trades y delistings."""
    trades: pd.DataFrame
    """Operación ejecutada por nombre (Δ peso) en cada fecha de ejecución."""
    target_weights: pd.DataFrame
    """Pesos justo después de operar (objetivo ya restringido), por fecha de ejecución."""
    turnover: pd.Series
    """Rotación one-way de cada rebalanceo (½·Σ|Δw|)."""
    costs: pd.DataFrame
    """Costes diarios en fracción del NAV: spread, impact, commission, borrow, total."""
    sector_attribution: pd.DataFrame
    """Contribución bruta aritmética al retorno por sector GICS."""
    quantile_returns: pd.DataFrame
    """Retorno bruto equiponderado diario de cada cubo (diagnóstico Q1..Qn)."""
    rebalance_dates: pd.DatetimeIndex
    """Fechas de decisión efectivamente ejecutadas."""
    execution_dates: pd.DatetimeIndex
    """Fechas de ejecución (decisión + 1 sesión)."""
    skipped_rebalances: tuple[tuple[pd.Timestamp, str], ...]
    """Decisiones omitidas y su motivo (sin señal, sección insuficiente...)."""
    silent_delistings: tuple[tuple[str, pd.Timestamp], ...]
    """Liquidaciones forzosas SIN retorno de delisting conocido: la lista de
    auditoría de `pit_and_biases.md` §6.4. Debe revisarse antes de creer el
    resultado; cada entrada es una posición cerrada al último precio observado."""
    held_gap_days: int
    """Días nombre-fecha en que una posición viva no tuvo precio (hueco de datos)
    y se mantuvo con retorno 0. Debe ser ~0 con datos limpios."""
    params: dict[str, object] = field(default_factory=dict)
    """Eco de los parámetros del backtest, para reproducibilidad del informe."""

    # ------------------------------------------------------------- derivados

    @property
    def nav(self) -> pd.Series:
        """Curva de NAV (base 1.0) compuesta con los retornos netos."""
        out = (1.0 + self.returns).cumprod()
        out.name = "nav"
        return out

    @property
    def total_costs(self) -> pd.Series:
        """Coste total acumulado por componente (fracción del NAV, suma diaria)."""
        return self.costs[[*_COST_COLS, "total"]].sum()

    def sharpe(self, **kwargs: object):  # -> SharpeResult (import diferido)
        """Sharpe **neto** con banda de error (delegado a `stats.performance`).

        Regla §3.8 del contrato: ninguna métrica sin intervalo de confianza.
        """
        from earnings_alpha.stats.performance import sharpe_ratio

        return sharpe_ratio(self.returns, **kwargs)  # type: ignore[arg-type]

    def summary(self) -> dict[str, object]:
        """Resumen de cabecera: rendimiento, riesgo, rotación y costes.

        El Sharpe se devuelve como `SharpeResult` completo (con IC y PSR), no
        como número desnudo.
        """
        from earnings_alpha.stats.ic import TRADING_DAYS_PER_YEAR
        from earnings_alpha.stats.performance import drawdown_series

        r = self.returns.to_numpy()
        years = len(r) / TRADING_DAYS_PER_YEAR
        dd = drawdown_series(self.returns)
        return {
            "sharpe": self.sharpe(),
            "ann_return": float(np.mean(r) * TRADING_DAYS_PER_YEAR),
            "ann_volatility": float(np.std(r, ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR)),
            "max_drawdown": float(dd.min()),
            "total_net_return": float(np.prod(1.0 + r) - 1.0),
            "annual_one_way_turnover": float(self.turnover.sum() / years) if years > 0 else 0.0,
            "total_costs": self.total_costs.to_dict(),
            "n_rebalances": len(self.rebalance_dates),
            "n_skipped": len(self.skipped_rebalances),
            "n_silent_delistings": len(self.silent_delistings),
        }


class CrossSectionalBacktest:
    """Backtest vectorizado de carteras de cubos sobre una señal cross-section.

    Parameters
    ----------
    universe:
        Universo point-in-time: un `UniverseProvider` (se usa
        `membership_panel`) o directamente un DataFrame booleano
        `index=date, columns=ticker`. `None` = todos los nombres del panel son
        elegibles siempre (solo aceptable en tests o universos ya filtrados).
    sectors:
        `ticker -> sector GICS`, para la neutralidad sectorial opcional y la
        atribución. Sin él, la atribución agrupa todo en ``"(sin sector)"``.
    execution:
        Ver `ExecutionTiming`. Por defecto, apertura de la sesión siguiente.
    capital:
        AUM en dólares. Convierte pesos en nocionales: el impacto y el límite de
        participación **escalan con él** (un backtest de $10M y uno de $1B no se
        parecen). El defecto, $10M, es un libro pequeño institucional.
    """

    def __init__(
        self,
        *,
        universe: object | None = None,
        sectors: pd.Series | Mapping[str, str] | None = None,
        execution: ExecutionTiming = "next_open",
        capital: float = 10_000_000.0,
    ) -> None:
        if execution not in ("next_open", "next_close"):
            msg = f"execution debe ser 'next_open' o 'next_close'; recibido {execution!r}"
            raise ConfigError(msg)
        if not math.isfinite(capital) or capital <= 0.0:
            msg = f"capital debe ser positivo; recibido {capital!r}"
            raise ConfigError(msg)
        if (
            universe is not None
            and not isinstance(universe, pd.DataFrame)
            and not hasattr(universe, "membership_panel")
        ):
            msg = (
                "universe debe ser un DataFrame booleano (date x ticker) o un "
                "UniverseProvider con membership_panel()"
            )
            raise ConfigError(msg)
        self.universe = universe
        self.sectors = (
            pd.Series(dict(sectors)) if isinstance(sectors, Mapping) else sectors
        )
        self.execution: ExecutionTiming = execution
        self.capital = float(capital)

    # ------------------------------------------------------------------ run

    def run(
        self,
        scores: pd.Series,
        prices: pd.DataFrame,
        *,
        n_quantiles: int = 5,
        rebalance: str = "W-FRI",
        long_short: bool = True,
        costs: CostModel,
        max_weight: float = 0.02,
        adv_participation: float | None = 0.05,
        weighting: str = "equal",
        sector_neutral: bool = False,
        max_turnover: float | None = None,
        signal_staleness: int = 5,
        min_names: int | None = None,
        adv_window: int = 21,
        vol_window: int = 21,
        warmup: int | None = None,
        delist_returns: Mapping[str, float] | None = None,
    ) -> BacktestResult:
        """Ejecuta el backtest. Firma del contrato §3.7 más parámetros opcionales.

        Parameters
        ----------
        scores:
            Señal en panel `(date, ticker)`. La fila fechada `t` debe ser
            conocible al cierre de `t` (responsabilidad del productor de la
            señal; este motor añade el retardo de ejecución).
        prices:
            Panel `(date, ticker)` con al menos `close`; `adj_close` para
            retornos totales (muy recomendado), `open` para `execution=
            "next_open"`, y `volume` o `dollar_volume` para ADV.
        n_quantiles, rebalance, long_short, costs, max_weight, adv_participation:
            Contrato §3.7. `adv_participation=None` desactiva el límite de
            participación (los costes de impacto siguen aplicando).
        weighting:
            ``"equal"`` o ``"score"`` dentro de cada pata (§11.1: la ponderación
            se declara siempre).
        sector_neutral:
            Cubos dentro de sector y patas equilibradas por sector (exposición
            neta sectorial cero). Requiere `sectors` en el constructor.
        max_turnover:
            Tope one-way por rebalanceo (fracción del NAV); se ejecuta la
            fracción proporcional del movimiento.
        signal_staleness:
            Sesiones máximas de antigüedad de la señal en la fecha de decisión;
            más vieja ⇒ el rebalanceo se omite (y queda registrado).
        min_names:
            Mínimo de nombres elegibles para rebalancear. Por defecto
            `max(2·n_quantiles, 4)`; para investigación real debe ser ≥ 30.
        adv_window, vol_window:
            Ventanas **hacia atrás** (terminan en la sesión previa a la
            ejecución) del ADV y la sigma diaria usados por costes y límites
            (`pit_and_biases.md` §12.6).
        warmup:
            Sesiones iniciales sin decisiones, para que ADV y sigma existan.
            Por defecto `max(adv_window, vol_window)`.
        delist_returns:
            `ticker -> retorno de la barra final` para series que terminan con
            posición viva (fusión en efectivo: prima; quiebra: −30 %/−55 % de
            Shumway si no hay valor final; ver `pit_and_biases.md` §6.3). Sin
            entrada, la liquidación al último precio queda en
            `silent_delistings` para auditoría.
        """
        # ------------------------------------------------------- validación
        if not isinstance(costs, CostModel):
            msg = f"costs debe ser un CostModel; recibido {type(costs).__name__}"
            raise ConfigError(msg)
        if n_quantiles < 2:
            msg = f"n_quantiles debe ser >= 2; recibido {n_quantiles}"
            raise ConfigError(msg)
        if not 0.0 < max_weight <= 1.0:
            msg = f"max_weight debe estar en (0, 1]; recibido {max_weight!r}"
            raise ConfigError(msg)
        if adv_participation is not None and adv_participation <= 0.0:
            msg = f"adv_participation debe ser > 0 o None; recibido {adv_participation!r}"
            raise ConfigError(msg)
        if signal_staleness < 0:
            msg = f"signal_staleness debe ser >= 0; recibido {signal_staleness}"
            raise ConfigError(msg)
        if sector_neutral and self.sectors is None:
            msg = "sector_neutral=True exige sectors en el constructor"
            raise ConfigError(msg)
        if not isinstance(prices, pd.DataFrame) or "close" not in prices.columns:
            msg = "prices debe ser un panel (date, ticker) con columna 'close'"
            raise DataQualityError(msg)
        check_panel(prices["close"], name="prices")
        check_panel(scores, name="scores")

        # ------------------------------------------------- matrices anchas
        close = prices["close"].unstack(level=-1).sort_index()
        if "adj_close" in prices.columns and prices["adj_close"].notna().any():
            adjc = prices["adj_close"].unstack(level=-1).sort_index()
        else:
            adjc = close
        dates = adjc.index
        tickers = adjc.columns
        n_dates, n_names = len(dates), len(tickers)
        if n_dates < 3:
            msg = f"panel de precios con solo {n_dates} fechas: no hay backtest posible"
            raise InsufficientHistory(msg)

        has_open = "open" in prices.columns and prices["open"].notna().any()
        if self.execution == "next_open" and not has_open:
            msg = (
                "execution='next_open' requiere la columna 'open' en prices; "
                "usa execution='next_close' si solo hay cierres"
            )
            raise DataQualityError(msg)
        if has_open:
            open_w = prices["open"].unstack(level=-1).reindex(index=dates, columns=tickers)
            adjo = open_w * (adjc / close)
            adjo_v = adjo.to_numpy(dtype=float)
        else:
            adjo_v = np.full((n_dates, n_names), np.nan)

        if "dollar_volume" in prices.columns:
            dollar_vol = prices["dollar_volume"].unstack(level=-1).reindex(
                index=dates, columns=tickers
            )
        elif "volume" in prices.columns:
            vol_w = prices["volume"].unstack(level=-1).reindex(index=dates, columns=tickers)
            dollar_vol = vol_w * close
        else:
            dollar_vol = None
        if dollar_vol is None and adv_participation is not None:
            msg = (
                "el límite de participación sobre ADV requiere 'volume' o "
                "'dollar_volume' en prices; pasa adv_participation=None para "
                "desactivarlo de forma explícita"
            )
            raise DataQualityError(msg)

        # `rcc_raw` (sin puentear) alimenta la sigma de costes y la auditoría de
        # huecos. Los retornos de TENENCIA del bucle diario (`rcc`, `ron`) se
        # calculan sobre precios FORWARD-FILLEADOS: una posición viva que
        # atraviesa un hueco de cotización (suspensión, halt) debe realizar en
        # la sesión de reapertura el movimiento acumulado contra el último
        # precio conocido — con `pct_change` sin puentear, la reapertura da NaN
        # (precio previo NaN), el bucle lo convierte en retorno 0 y el
        # movimiento del hueco (típicamente una caída severa) desaparece del
        # NAV para siempre. El puenteo DIFIERE el P&L; no lo destruye. La
        # elegibilidad y `can_trade` siguen usando `valid_price` sin ffill.
        rcc_raw_df = adjc.pct_change(fill_method=None)
        rcc_raw = rcc_raw_df.to_numpy(dtype=float)
        adjc_f = adjc.ffill()
        rcc = adjc_f.pct_change(fill_method=None).to_numpy(dtype=float)
        adjc_v = adjc.to_numpy(dtype=float)
        if has_open:
            ron = (adjo / adjc_f.shift(1) - 1.0).to_numpy(dtype=float)
            rid = (adjc / adjo - 1.0).to_numpy(dtype=float)
        else:
            ron = rid = np.full((n_dates, n_names), np.nan)

        # ADV y sigma HACIA ATRÁS: la ventana termina en la sesión previa
        # (shift(1)) a aquella en que se usan (`pit_and_biases.md` §12.6).
        if dollar_vol is not None:
            adv_m = (
                dollar_vol.rolling(adv_window, min_periods=max(3, adv_window // 3))
                .mean()
                .shift(1)
                .to_numpy(dtype=float)
            )
        else:
            adv_m = None
        sig_m = (
            rcc_raw_df.rolling(vol_window, min_periods=max(5, vol_window // 3))
            .std()
            .shift(1)
            .to_numpy(dtype=float)
        )

        scores_w = scores.unstack(level=-1).reindex(index=dates, columns=tickers)
        s_vals = scores_w.to_numpy(dtype=float)

        # ------------------------------------------------------ pertenencia
        membership, row_known = self._membership_matrix(dates, tickers)

        # ---------------------------------------------------------- sectores
        if self.sectors is not None:
            sec_series = self.sectors.reindex(tickers).fillna("(sin sector)").astype(str)
        else:
            sec_series = pd.Series("(sin sector)", index=tickers, dtype=object)
        sec_cat = pd.Categorical(sec_series)
        sector_names = list(sec_cat.categories)
        onehot = (np.asarray(sec_cat.codes)[:, None] == np.arange(len(sector_names))[None, :])
        onehot = onehot.astype(float)

        # --------------------------------------------------- fechas de decisión
        effective_warmup = max(adv_window, vol_window) if warmup is None else warmup
        if effective_warmup < 0:
            msg = f"warmup debe ser >= 0; recibido {warmup!r}"
            raise ConfigError(msg)
        if effective_warmup >= n_dates - 1:
            msg = (
                f"warmup de {effective_warmup} sesiones no deja fechas de decisión "
                f"en un panel de {n_dates}"
            )
            raise InsufficientHistory(msg)
        decisions = rebalance_schedule(dates[effective_warmup:], rebalance)
        min_eligible = max(2 * n_quantiles, 4) if min_names is None else min_names

        rows_with_scores = np.flatnonzero(np.isfinite(s_vals).any(axis=1))
        valid_price = np.isfinite(adjc_v)
        date_pos = {ts: i for i, ts in enumerate(dates)}
        tickers_arr = np.asarray(tickers)

        targets: dict[int, np.ndarray] = {}
        buckets_at: dict[int, np.ndarray] = {}
        decision_of: dict[int, pd.Timestamp] = {}
        skipped: list[tuple[pd.Timestamp, str]] = []

        for ts in decisions:
            d = date_pos[ts]
            e = d + 1
            if e >= n_dates:
                skipped.append((ts, "sin sesión de ejecución posterior en el panel"))
                continue
            pos = int(np.searchsorted(rows_with_scores, d, side="right")) - 1
            if pos < 0:
                skipped.append((ts, "sin puntuaciones hasta la fecha"))
                continue
            s_row = int(rows_with_scores[pos])
            if d - s_row > signal_staleness:
                skipped.append(
                    (ts, f"señal caducada: {d - s_row} sesiones > staleness {signal_staleness}")
                )
                continue
            if membership is not None:
                if not row_known[d]:
                    msg = (
                        f"el universo no cubre la fecha de decisión {ts.date()}: "
                        "no hay snapshot de pertenencia en o antes de esa fecha"
                    )
                    raise UniverseError(msg)
                eligible = membership[d].copy()
            else:
                eligible = np.ones(n_names, dtype=bool)
            eligible &= valid_price[d] & np.isfinite(s_vals[s_row])
            n_elig = int(eligible.sum())
            if n_elig < max(min_eligible, n_quantiles):
                skipped.append((ts, f"sección cruzada insuficiente: {n_elig} nombres"))
                continue

            idx = np.flatnonzero(eligible)
            cs = pd.Series(s_vals[s_row][idx], index=pd.Index(tickers_arr[idx], name="ticker"))
            sec_for_cs = sec_series.loc[cs.index] if sector_neutral else None
            try:
                w_target = build_target_weights(
                    cs,
                    n_quantiles=n_quantiles,
                    long_short=long_short,
                    weighting=weighting,
                    max_weight=max_weight,
                    sectors=sec_for_cs,
                )
            except InsufficientHistory as exc:
                skipped.append((ts, str(exc)))
                continue
            vec = np.zeros(n_names)
            vec[idx] = w_target.to_numpy(dtype=float)
            targets[e] = vec
            decision_of[e] = ts
            bvec = np.full(n_names, np.nan)
            bvec[idx] = assign_quantiles(cs, n_quantiles, by=sec_for_cs).to_numpy(dtype=float)
            buckets_at[e] = bvec

        if not targets:
            reasons = "; ".join(f"{ts.date()}: {why}" for ts, why in skipped[:5])
            msg = (
                "ningún rebalanceo ejecutable: revisa el solapamiento de fechas "
                f"entre señal y precios. Primeros motivos: {reasons or 'sin decisiones'}"
            )
            raise InsufficientHistory(msg)

        # -------------------------------------------------------- bucle diario
        dl_map = dict(delist_returns) if delist_returns else {}
        last_valid = np.where(valid_price, np.arange(n_dates)[:, None], -1).max(axis=0)
        gap_days = np.diff(dates.to_numpy()).astype("timedelta64[D]").astype(int)
        ticker_list = [str(t) for t in tickers_arr]

        ret_net = np.zeros(n_dates)
        ret_gross = np.zeros(n_dates)
        cost_mat = np.zeros((n_dates, 4))
        w_end_mat = np.zeros((n_dates, n_names))
        trade_mat = np.zeros((n_dates, n_names))
        sector_mat = np.zeros((n_dates, len(sector_names)))
        q_mat = np.full((n_dates, n_quantiles), np.nan)
        turnover_vals: list[float] = []
        exec_dates: list[pd.Timestamp] = []
        executed_decisions: list[pd.Timestamp] = []
        post_trade_rows: list[np.ndarray] = []
        silent: list[tuple[str, pd.Timestamp]] = []
        held_gaps = 0

        first_exec = min(targets)
        w = np.zeros(n_names)
        nav = 1.0
        active_bucket = np.full(n_names, np.nan)

        for t in range(first_exec, n_dates):
            cb = CostBreakdown()
            contrib = np.zeros(n_names)

            held = w != 0.0
            held_gaps += int((held & ~np.isfinite(rcc_raw[t]) & (last_valid > t)).sum())

            if t in targets:
                if self.execution == "next_open":
                    # Posición viva sin 'open' hoy pero con cierre: el retorno
                    # cierre-a-cierre se atribuye al tramo overnight de la
                    # posición vieja (r_id queda 0 y `can_trade` ya impide
                    # operar el nombre). Sin esto, el retorno del día
                    # desaparecería del backtest sin rastro en la auditoría.
                    r_on_f = np.where(
                        np.isfinite(ron[t]),
                        ron[t],
                        np.where(np.isfinite(rcc[t]), rcc[t], 0.0),
                    )
                    port_on = float(w @ r_on_f)
                    self._check_solvent(1.0 + port_on, dates[t], "overnight")
                    w_open = w * (1.0 + r_on_f) / (1.0 + port_on)
                    can_trade = np.isfinite(adjo_v[t])
                    tgt = np.where(can_trade, targets[t], w_open)
                    delta = self._constrain_trades(
                        tgt - w_open, adv_m, t, nav, adv_participation, max_turnover
                    )
                    w_new = w_open + delta
                    cb = cb + costs.execution_costs(
                        delta,
                        adv_m[t] if adv_m is not None else None,
                        sig_m[t],
                        self.capital * nav,
                    )
                    r_id_f = np.where(np.isfinite(rid[t]), rid[t], 0.0)
                    port_id = float(w_new @ r_id_f)
                    self._check_solvent(1.0 + port_id, dates[t], "intradía")
                    gross_t = (1.0 + port_on) * (1.0 + port_id) - 1.0
                    w_after = w_new * (1.0 + r_id_f) / (1.0 + port_id)
                    contrib = w * r_on_f + w_new * r_id_f
                    q_source = rid[t]
                else:  # next_close
                    r_f = np.where(np.isfinite(rcc[t]), rcc[t], 0.0)
                    port = float(w @ r_f)
                    self._check_solvent(1.0 + port, dates[t], "sesión")
                    gross_t = port
                    w_drift = w * (1.0 + r_f) / (1.0 + port)
                    can_trade = valid_price[t]
                    tgt = np.where(can_trade, targets[t], w_drift)
                    delta = self._constrain_trades(
                        tgt - w_drift, adv_m, t, nav, adv_participation, max_turnover
                    )
                    w_new = w_drift + delta
                    cb = cb + costs.execution_costs(
                        delta,
                        adv_m[t] if adv_m is not None else None,
                        sig_m[t],
                        self.capital * nav,
                    )
                    w_after = w_new
                    contrib = w * r_f
                    q_source = None  # la posición se toma al cierre: sin retorno hoy

                trade_mat[t] = delta
                turnover_vals.append(float(0.5 * np.abs(delta).sum()))
                exec_dates.append(dates[t])
                executed_decisions.append(decision_of[t])
                post_trade_rows.append(w_new.copy())
                active_bucket = buckets_at[t]
            else:
                r_f = np.where(np.isfinite(rcc[t]), rcc[t], 0.0)
                port = float(w @ r_f)
                self._check_solvent(1.0 + port, dates[t], "sesión")
                gross_t = port
                w_after = w * (1.0 + r_f) / (1.0 + port)
                contrib = w * r_f
                q_source = rcc[t]

            # ---- delisting: series que terminan con posición viva (§6.3-6.4).
            # El final de la MUESTRA no es un delisting: solo cuentan las series
            # que terminan antes que el propio panel.
            dying = (w_after != 0.0) & (last_valid <= t) & (last_valid < n_dates - 1)
            if dying.any():
                dl_total = 0.0
                for i in np.flatnonzero(dying):
                    tkr = str(tickers_arr[i])
                    dlr = dl_map.get(tkr)
                    if dlr is None:
                        silent.append((tkr, dates[t]))
                    else:
                        dl_total += w_after[i] * float(dlr)
                        contrib[i] += w_after[i] * float(dlr)
                    w_after[i] = 0.0
                if dl_total != 0.0:
                    gross_t += dl_total
                    self._check_solvent(1.0 + dl_total, dates[t], "delisting")
                    w_after = w_after / (1.0 + dl_total)

            # ---- préstamo de los cortos que pasan la noche (ACT/360)
            if t < n_dates - 1:
                hold = costs.holding_costs(w_after, ticker_list, int(gap_days[t]))
                cb = cb + hold

            net_t = gross_t - cb.total
            nav *= 1.0 + net_t
            if nav <= 0.0:
                msg = f"NAV agotado el {dates[t].date()}: retorno neto {net_t:.4f}"
                raise DataQualityError(msg)

            ret_net[t] = net_t
            ret_gross[t] = gross_t
            cost_mat[t] = (cb.spread, cb.impact, cb.commission, cb.borrow)
            w_end_mat[t] = w_after
            sector_mat[t] = contrib @ onehot

            if q_source is not None and np.isfinite(active_bucket).any():
                for q in range(1, n_quantiles + 1):
                    mask = (active_bucket == q) & np.isfinite(q_source)
                    if mask.any():
                        q_mat[t, q - 1] = float(np.mean(q_source[mask]))

            w = w_after

        # ------------------------------------------------------------ salida
        sl = slice(first_exec, n_dates)
        idx = dates[sl]
        exec_idx = pd.DatetimeIndex(exec_dates, name="date")
        costs_df = pd.DataFrame(cost_mat[sl], index=idx, columns=list(_COST_COLS))
        costs_df["total"] = costs_df.sum(axis=1)

        params: dict[str, object] = {
            "n_quantiles": n_quantiles,
            "rebalance": rebalance,
            "long_short": long_short,
            "max_weight": max_weight,
            "adv_participation": adv_participation,
            "weighting": weighting,
            "sector_neutral": sector_neutral,
            "max_turnover": max_turnover,
            "signal_staleness": signal_staleness,
            "min_names": min_eligible,
            "adv_window": adv_window,
            "vol_window": vol_window,
            "warmup": effective_warmup,
            "execution": self.execution,
            "capital": self.capital,
            "cost_model": costs,
        }
        return BacktestResult(
            returns=pd.Series(ret_net[sl], index=idx, name="net_return"),
            gross_returns=pd.Series(ret_gross[sl], index=idx, name="gross_return"),
            weights=pd.DataFrame(w_end_mat[sl], index=idx, columns=tickers),
            trades=pd.DataFrame(trade_mat[sl], index=idx, columns=tickers),
            target_weights=pd.DataFrame(
                np.vstack(post_trade_rows), index=exec_idx, columns=tickers
            ),
            turnover=pd.Series(turnover_vals, index=exec_idx, name="one_way_turnover"),
            costs=costs_df,
            sector_attribution=pd.DataFrame(sector_mat[sl], index=idx, columns=sector_names),
            quantile_returns=pd.DataFrame(
                q_mat[sl], index=idx, columns=[f"q{q}" for q in range(1, n_quantiles + 1)]
            ),
            rebalance_dates=pd.DatetimeIndex(executed_decisions, name="date"),
            execution_dates=exec_idx,
            skipped_rebalances=tuple(skipped),
            silent_delistings=tuple(silent),
            held_gap_days=held_gaps,
            params=params,
        )

    # ------------------------------------------------------------- auxiliares

    @staticmethod
    def _check_solvent(factor: float, when: pd.Timestamp, leg: str) -> None:
        if factor <= 0.0 or not math.isfinite(factor):
            msg = (
                f"el retorno {leg} del {when.date()} aniquila el NAV "
                f"(factor {factor:.4f}); el backtest no puede continuar"
            )
            raise DataQualityError(msg)

    def _constrain_trades(
        self,
        delta: np.ndarray,
        adv_m: np.ndarray | None,
        t: int,
        nav: float,
        adv_participation: float | None,
        max_turnover: float | None,
    ) -> np.ndarray:
        """Aplica límite de participación sobre ADV y tope de rotación al vector de trades.

        El límite de participación es duro y por nombre; el de rotación escala
        todas las operaciones por el mismo factor (preserva las proporciones del
        objetivo). Un ADV desconocido no restringe aquí: su castigo, explícito,
        vive en `CostModel` (participación 1.0 y peor tramo).
        """
        out = delta.copy()
        if adv_participation is not None and adv_m is not None:
            cap = adv_participation * adv_m[t] / (self.capital * nav)
            cap = np.where(np.isfinite(cap) & (cap >= 0.0), cap, np.inf)
            out = np.clip(out, -cap, cap)
        if max_turnover is not None:
            need = float(0.5 * np.abs(out).sum())
            if need > max_turnover > 0.0:
                out = out * (max_turnover / need)
            elif max_turnover == 0.0:
                out = np.zeros_like(out)
        return out

    # -------------------------------------------------------------- universo

    def _membership_matrix(
        self, dates: pd.DatetimeIndex, tickers: pd.Index
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Matriz booleana de pertenencia PIT alineada al panel, o None si no hay universo.

        Cada fila usa el último snapshot **en o antes** de la fecha (ffill);
        `row_known` marca las fechas cubiertas por algún snapshot: decidir sobre
        una fecha no cubierta lanza `UniverseError` en vez de asumir pertenencia.
        """
        if self.universe is None:
            return None, None
        if isinstance(self.universe, pd.DataFrame):
            mem = self.universe
        else:
            mem = self.universe.membership_panel(  # type: ignore[attr-defined]
                dates[0].date(), dates[-1].date()
            )
        if not isinstance(mem, pd.DataFrame) or mem.empty:
            msg = "el universo devolvió un panel de pertenencia vacío"
            raise UniverseError(msg)
        mem = mem.sort_index()
        aligned = mem.reindex(index=dates, method="ffill")
        row_known = aligned.notna().any(axis=1).to_numpy()
        matrix = (
            aligned.reindex(columns=tickers)
            .fillna(False)
            .astype(bool)
            .to_numpy()
        )
        return matrix, row_known
