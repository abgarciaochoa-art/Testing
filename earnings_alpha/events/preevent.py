"""PreEventFeatures: detección de la huella de negociación informada pre-anuncio.

NOTA LEGAL Y METODOLÓGICA OBLIGATORIA (contrato §3.5)
-----------------------------------------------------
Todas las features de este módulo derivan de datos **públicos**: precios y
volúmenes consolidados, cadenas de opciones, short interest agregado de FINRA,
transparencia ATS/OTC de FINRA y formularios Form 4 ya presentados en EDGAR. El
objetivo es detectar la **huella estadística** que la negociación informada deja
en variables observables por cualquier participante, no acceder a información
material no pública. Ninguna feature requiere ni admite información privilegiada.

Qué hay aquí
------------
* `EventContext`: contenedor de datos de entrada (eventos, precios, mercado y
  fuentes opcionales de flujo/opciones), con `from_synthetic()` para el banco de
  pruebas offline.
* `PreEventFeatures`: orquesta las features de flujo (`events.flow`), el CAR
  pre-evento (modelo de mercado con estandarización de Patell) y las señales de
  opciones (delegadas a `earnings_alpha.events.options_signals` si el módulo
  existe; si no, degradan a NaN). Devuelve un DataFrame por `event_id` con
  metadatos (ticker, fecha negociable, sector, tamaño, liquidez) y la
  **residualización por cohorte** como opción.
* `InformedTradingScore`: agregación estandarizada en sección cruzada (por
  cohorte de fecha y sector) con pesos configurables.
* `residualize_features` y `cohort_labels`: la infraestructura de ortogonalización
  del informe (§12.2), expuesta por separado para poder **medir** qué features
  mueren tras controlar por el CAR pre-evento, tamaño, liquidez y sector — el
  informe documenta que le ocurre a más señales de las que la literatura primaria
  sugiere (§9.h, §11 fila 4).

La aritmética de tasa base que evita autoengaños (informe §12.4)
----------------------------------------------------------------
Con prevalencia real de filtración ``p``, sensibilidad ``Se`` y especificidad
``Sp``, el valor predictivo positivo es ``PPV = p·Se / (p·Se + (1-p)·(1-Sp))``.
Con ``p = 0,02``, ``Se = 0,80`` y ``Sp = 0,90``::

    PPV = 0,016 / (0,016 + 0,098) ≈ 0,14

Es decir: **el 86 % de las alertas de un detector "excelente" serían falsos
positivos**. Consecuencias de diseño respetadas aquí: (a) la salida es un score
**continuo**, nunca una alerta binaria; (b) la evaluación debe reportar
precision–recall además de AUC-ROC, engañosa con clases desbalanceadas; (c) el
uso legítimo del score es **ponderar la exposición** en la cartera de eventos,
no "señalar filtraciones". Ver `positive_predictive_value`.

Invariante point-in-time
------------------------
Ninguna feature usa datos de la sesión T (la fecha negociable del anuncio) ni
posteriores: ventanas de detección [T-k, T-1], ventanas de estimación que
terminan en T-40, y fuentes con retardo institucional filtradas por su
`available_at` real. La columna `available_at` del resultado es la medianoche de
T: todo lo usado era público antes de la apertura de T. Los tests lo verifican
con `pit.assert_no_lookahead` y perturbando los datos desde T en adelante.

Referencias principales
-----------------------
- Patell, J. M. (1976). *JAR* 14(2) — estandarización del AR.
- Boehmer, E., Musumeci, J., Poulsen, A. B. (1991). *JFE* 30(2) — varianza
  inducida por el evento (para contrastes en sección cruzada).
- Frazzini, A., Lamont, O. A. (2007). NBER WP 13090 — prima de anuncio: el CAR
  pre-evento hereda >60 pb/mes solo por anunciar; benchmark correcto = cartera
  de anunciantes.
- Kyle (1985); Glosten y Milgrom (1985) — marco teórico de la huella.
- Kacperczyk, M., Pagnotta, E. (2019). *RFS* — los informados eligen días de alto
  volumen y los spreads se ESTRECHAN: el ensanchamiento de spread no es detector.
- Akey, Grégoire y Martineau (2022). *JFE* 143(3) — buena parte de la revelación
  va por cotizaciones, no operaciones: techo estructural de un detector OHLCV.
- Xie (2026). *JAR* — incluso la filtración real deja una huella más débil de lo
  que sugiere la sorpresa.
- Cohen, Malloy y Pomorski (2012); Cremers y Weinbaum (2010); Xing, Zhang y Zhao
  (2010); Johnson y So (2012) — signos de las señales agregadas.
- López de Prado (2018) — validación temporal purgada del `SurpriseModel` aguas
  abajo.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final, Literal

import numpy as np
import pandas as pd

from earnings_alpha.errors import DataQualityError, InsufficientHistory
from earnings_alpha.events import flow
from earnings_alpha.pit import TradingCalendar
from earnings_alpha.types import Ticker

try:  # pragma: no cover - depende de si el módulo de opciones ya existe
    from earnings_alpha.events import options_signals as _options_signals
except ImportError:  # pragma: no cover
    _options_signals = None  # type: ignore[assignment]

__all__ = [
    "DEFAULT_DIRECTIONAL_WEIGHTS",
    "DEFAULT_INTENSITY_WEIGHTS",
    "EventContext",
    "InformedTradingScore",
    "METADATA_COLUMNS",
    "OPTION_FEATURE_COLUMNS",
    "PreEventFeatures",
    "cohort_labels",
    "positive_predictive_value",
    "pre_event_car",
    "residualize_features",
]


METADATA_COLUMNS: Final[tuple[str, ...]] = (
    "ticker",
    "event_date",
    "available_at",
    "sector",
    "cohort",
    "log_mktcap",
    "log_adv",
    "realized_vol_60d",
)
"""Columnas de metadatos del DataFrame de features: no son señales y ninguna
transformación (residualización, score) debe tratarlas como tales."""

OPTION_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "oi_buildup_calls",
    "oi_buildup_puts",
    "put_call_volume_ratio",
    "iv_skew_25delta",
    "vol_spread",
    "iv_term_slope",
)
"""Columnas de opciones del contrato §3.5. Las produce
`earnings_alpha.events.options_signals` (módulo de otro propietario); si no está
disponible o no hay datos de opciones, salen como NaN — degradación documentada,
no silenciosa (quedan listadas en ``result.attrs['missing_sources']``)."""


def positive_predictive_value(
    prevalence: float, sensitivity: float, specificity: float
) -> float:
    """Precisión (PPV) de un detector binario dada la prevalencia (informe §12.4).

    ``PPV = p·Se / (p·Se + (1-p)·(1-Sp))``. Con ``p=0,02, Se=0,80, Sp=0,90`` da
    ≈ 0,14: el 86 % de las alertas serían falsos positivos. Es la aritmética que
    justifica que el output de este módulo sea un score continuo para ponderar
    exposición y no una alerta binaria.
    """
    for name, v in (("prevalence", prevalence), ("sensitivity", sensitivity),
                    ("specificity", specificity)):
        if not 0.0 <= v <= 1.0:
            msg = f"{name} debe estar en [0, 1]; recibido {v}"
            raise DataQualityError(msg)
    true_pos = prevalence * sensitivity
    false_pos = (1.0 - prevalence) * (1.0 - specificity)
    if true_pos + false_pos == 0.0:
        return float("nan")
    return true_pos / (true_pos + false_pos)


# ---------------------------------------------------------------------------
# Contexto de evento
# ---------------------------------------------------------------------------


@dataclass
class EventContext:
    """Datos de entrada de `PreEventFeatures.compute` (contrato §3.5).

    Obligatorios: `events` (con ``event_id, ticker, event_date``), `prices`
    (panel canónico ``(date, ticker)`` con OHLCV) y `market` (serie del índice
    para el modelo de mercado del CAR). El resto son fuentes opcionales: si
    faltan, sus features salen NaN y la fuente queda registrada en
    ``attrs['missing_sources']`` del resultado — nunca se inventa un valor.

    `events` debería incluir la prehistoria del emisor cuando exista: los eventos
    sin precios no reciben fila de features, pero mejoran las exclusiones de la
    ventana base y la línea base por emisor de Chae (§3.6 del informe).
    """

    events: pd.DataFrame
    prices: pd.DataFrame
    market: pd.DataFrame | pd.Series | None = None
    calendar: TradingCalendar | None = None
    sectors: pd.Series | Mapping[Ticker, str] | None = None
    short_interest: pd.DataFrame | None = None
    off_exchange: pd.DataFrame | None = None
    form4: pd.DataFrame | None = None
    options: pd.DataFrame | None = None
    analyst_revisions: pd.Series | None = None
    """`analyst_revision_drift` por event_id, si `data.estimates` la ha calculado.
    Su construcción queda fuera de este módulo (informe §14)."""

    @classmethod
    def from_synthetic(cls, market: object) -> EventContext:
        """Construye el contexto completo desde un `data.synthetic.SyntheticMarket`.

        Toma los eventos **con prehistoria** (mejor línea base por emisor), el
        panel OHLCV, el índice de mercado y las tablas de flujo con sus retardos
        de publicación realistas. El generador no publica Form 4 sintéticos, así
        que las features de insiders salen NaN sobre este contexto.
        """
        from earnings_alpha.data.synthetic import SyntheticMarket

        if not isinstance(market, SyntheticMarket):
            msg = f"se esperaba un SyntheticMarket; recibido {type(market).__name__}"
            raise DataQualityError(msg)
        return cls(
            events=market.events(include_prehistory=True),
            prices=market.prices(),
            market=market.market_index(),
            calendar=market.calendar,
            sectors=market.sectors(),
            short_interest=market.short_interest(),
            off_exchange=market.off_exchange(),
            options=market.options_daily(),
        )


# ---------------------------------------------------------------------------
# CAR pre-evento (modelo de mercado + Patell)
# ---------------------------------------------------------------------------


def _market_returns(market: pd.DataFrame | pd.Series) -> pd.Series:
    """Extrae la serie de retornos logarítmicos del índice de mercado."""
    if isinstance(market, pd.Series):
        out = market.astype(float)
    elif isinstance(market, pd.DataFrame):
        if "log_return" in market.columns:
            out = market["log_return"].astype(float)
        elif "close" in market.columns:
            close = market["close"].astype(float)
            out = np.log(close.where(close > 0)).diff()
        else:
            msg = "`market` debe traer una columna 'log_return' o 'close'"
            raise DataQualityError(msg)
    else:
        msg = f"`market` debe ser Series o DataFrame; recibido {type(market).__name__}"
        raise DataQualityError(msg)
    out.index = pd.DatetimeIndex(out.index).normalize()
    return out


def pre_event_car(
    prices: pd.DataFrame,
    market: pd.DataFrame | pd.Series,
    events: pd.DataFrame,
    *,
    windows: Sequence[int] = flow.DEFAULT_DETECTION_WINDOWS,
    estimation: tuple[int, int] = (-250, -40),
    min_estimation_obs: int = 120,
) -> pd.DataFrame:
    """CAR y SCAR pre-evento por modelo de mercado (informe §7.1).

    Modelo de mercado estimado por OLS en la ventana ``estimation`` (por defecto
    ``(-250, -40)``, alineada con el contrato §3.5: la estimación **termina antes**
    de la ventana de detección para no contaminarse con la propia huella)::

        r[i,t] = a + b·r_m[t] + e[i,t]
        AR[i,tau] = r[i,tau] - (a + b·r_m[tau])
        CAR[i,k]  = Σ_{tau=-k..-1} AR[i,tau]          -> pre_event_car_{k}d

    Estandarización de Patell (1976), que corrige el error de estimación de (a, b)
    y hace comparables eventos con distinta volatilidad::

        s²[i,tau] = s²_e·(1 + 1/L + (r_m[tau] - r̄_m)² / Σ(r_m - r̄_m)²)
        SCAR[i,k] = Σ SAR[i,tau] / sqrt(k)            -> pre_event_scar_{k}d

    Para contrastes en sección cruzada úsese el test de Boehmer, Musumeci y
    Poulsen (1991): alrededor de resultados la varianza aumenta y el t de Patell
    puro sobre-rechaza masivamente (informe §7.1).

    Exige al menos `min_estimation_obs` observaciones válidas (120 por defecto);
    si ningún evento las alcanza se lanza `InsufficientHistory`.

    **Advertencia de confusores (informe §7.2):** el CAR pre-evento es la feature
    más contaminada del conjunto — momentum de corto plazo, prima de anuncio
    (Frazzini y Lamont 2007: >60 pb/mes solo por anunciar) y sobre-extrapolación
    con reversión. Por eso el resto de features de flujo se ortogonaliza contra
    él (`residualize_features`) y no al revés.
    """
    ev = flow._normalize_events(events)
    px = flow._check_prices(prices, ["close"])
    returns = flow._returns_wide(px)
    r_m = _market_returns(market).reindex(pd.DatetimeIndex(returns.index))
    dates = pd.DatetimeIndex(returns.index)
    pos_arr = flow._positions(ev["event_date"], dates)
    e0, e1 = estimation
    if not (e0 < e1 < 0):
        msg = f"ventana de estimación inválida {estimation}: se exige e0 < e1 < 0"
        raise DataQualityError(msg)
    max_k = max(int(k) for k in windows)
    if e1 >= -max_k:
        msg = (
            f"la ventana de estimación {estimation} solapa la de detección "
            f"[-{max_k}, -1]; debe terminar antes (contrato §3.5)"
        )
        raise DataQualityError(msg)

    rm_arr = r_m.to_numpy(dtype=float)
    r_cols = {t: returns[t].to_numpy() for t in returns.columns}
    out: dict[str, np.ndarray] = {}
    for k in windows:
        out[f"pre_event_car_{k}d"] = np.full(len(ev), np.nan)
        out[f"pre_event_scar_{k}d"] = np.full(len(ev), np.nan)

    for i, row in enumerate(ev.itertuples(index=False)):
        pos = int(pos_arr[i])
        r = r_cols.get(row.ticker)
        if pos < 0 or r is None or pos + e0 < 0:
            continue
        est = slice(pos + e0, pos + e1 + 1)
        ri, mi = r[est], rm_arr[est]
        valid = np.isfinite(ri) & np.isfinite(mi)
        big_l = int(valid.sum())
        if big_l < min_estimation_obs:
            continue
        x, y = mi[valid], ri[valid]
        xbar = float(np.mean(x))
        sxx = float(np.sum((x - xbar) ** 2))
        if sxx <= 0.0:
            continue
        beta = float(np.sum((x - xbar) * (y - float(np.mean(y)))) / sxx)
        alpha = float(np.mean(y)) - beta * xbar
        resid = y - alpha - beta * x
        s2e = float(np.sum(resid * resid)) / (big_l - 2)
        if not np.isfinite(s2e) or s2e <= 0.0:
            continue

        det_all = flow._detection_slice(pos, max_k)
        if det_all is None:
            continue
        rd, md = r[det_all], rm_arr[det_all]
        if not (np.isfinite(rd).all() and np.isfinite(md).all()):
            continue
        ar = rd - alpha - beta * md
        s2_tau = s2e * (1.0 + 1.0 / big_l + (md - xbar) ** 2 / sxx)
        sar = ar / np.sqrt(s2_tau)
        for k in windows:
            kk = int(k)
            out[f"pre_event_car_{k}d"][i] = float(np.sum(ar[-kk:]))
            out[f"pre_event_scar_{k}d"][i] = float(np.sum(sar[-kk:])) / np.sqrt(kk)

    frame = pd.DataFrame(out, index=pd.Index(ev["event_id"], name="event_id"))
    flow._raise_if_all_nan(frame, "pre_event_car")
    return frame


# ---------------------------------------------------------------------------
# Cohortes y residualización
# ---------------------------------------------------------------------------


def cohort_labels(event_dates: pd.Series, *, min_size: int = 20) -> pd.Series:
    """Etiquetas de cohorte temporal para la estandarización cross-section.

    Los eventos de resultados se agrupan en el tiempo (~40 % de los eventos en
    tres semanas por trimestre) y los niveles calendario no son comparables entre
    temporadas (informe §12.1). La cohorte natural es la fecha de evento; cuando
    una fecha tiene menos de `min_size` eventos, se fusiona con las fechas
    siguientes (avance cronológico voraz) hasta alcanzar el mínimo — el
    equivalente operativo de "misma semana, o ±5 días hábiles" del informe. La
    última cohorte, si queda corta, se fusiona con la anterior.

    Devuelve una Series de Timestamps (fecha de inicio de la cohorte) alineada
    con `event_dates`.
    """
    if min_size < 2:
        msg = f"min_size debe ser >= 2; recibido {min_size}"
        raise DataQualityError(msg)
    dates = pd.DatetimeIndex(pd.to_datetime(event_dates)).normalize()
    counts = pd.Series(dates).value_counts().sort_index()
    mapping: dict[pd.Timestamp, pd.Timestamp] = {}
    bucket_start: pd.Timestamp | None = None
    bucket_count = 0
    for day, n in counts.items():
        if bucket_start is None:
            bucket_start, bucket_count = day, 0
        mapping[day] = bucket_start
        bucket_count += int(n)
        if bucket_count >= min_size:
            bucket_start, bucket_count = None, 0
    if bucket_start is not None and bucket_count < min_size and len(counts) > 1:
        # La cola huérfana se fusiona con la cohorte anterior.
        starts = sorted(set(mapping.values()))
        if len(starts) > 1:
            last_start = starts[-1]
            prev_start = starts[-2]
            for day, start in mapping.items():
                if start == last_start:
                    mapping[day] = prev_start
    labels = pd.Series(dates, index=event_dates.index).map(mapping)
    labels.name = "cohort"
    return labels


def residualize_features(
    features: pd.DataFrame,
    *,
    feature_cols: Sequence[str] | None = None,
    car_col: str = "pre_event_car_20d",
    control_cols: Sequence[str] = ("log_mktcap", "log_adv"),
    sector_col: str = "sector",
    cohort_col: str = "cohort",
    min_rows_over_params: int = 3,
) -> pd.DataFrame:
    """Ortogonalización obligatoria por cohorte de fecha (informe §12.2).

    Para cada cohorte y cada feature ``f`` se estima por OLS::

        f[i] = g0 + g1·pre_event_car_20[i] + g2·ln(mktcap[i]) + g3·ln(ADV[i])
             + Σ_s g_s·sector_dummy[s,i] + u[i]

    y se devuelve el residuo ``u``. Una feature que pierde todo su IC tras esto
    es un duplicado del CAR pre-evento y debe eliminarse — es exactamente el test
    de la fila 4 de la tabla de evidencia contraria del informe (§11): la
    aparente predictibilidad del flujo de opciones y de volumen desaparece al
    controlar por el retorno pre-anuncio. Esta función existe por separado para
    que ese experimento sea *medible*: calcúlese el IC con y sin residualizar.

    Los propios controles (`car_col`, `control_cols`) y las columnas de
    metadatos no se residualizan. Filas sin controles completos, o cohortes con
    menos de ``n_parámetros + min_rows_over_params`` observaciones válidas,
    devuelven NaN (nunca un residuo sobreajustado ni el valor original
    disfrazado de residuo).
    """
    if cohort_col not in features.columns:
        msg = f"`features` necesita la columna {cohort_col!r}; usa `cohort_labels`"
        raise DataQualityError(msg)
    controls = [car_col, *control_cols]
    missing = [c for c in controls if c not in features.columns]
    if missing:
        msg = f"faltan columnas de control {missing} en `features`"
        raise DataQualityError(msg)

    skip = set(METADATA_COLUMNS) | set(controls)
    if feature_cols is None:
        feature_cols = [
            c
            for c in features.columns
            if c not in skip and pd.api.types.is_numeric_dtype(features[c])
        ]
    out = features.copy()
    ctrl = features[controls].astype(float)
    sector = (
        features[sector_col].astype(str)
        if sector_col in features.columns
        else pd.Series("_all_", index=features.index)
    )

    for _, idx in features.groupby(features[cohort_col], sort=False).groups.items():
        sub_ctrl = ctrl.loc[idx]
        sub_sector = sector.loc[idx]
        base_valid = sub_ctrl.notna().all(axis=1)
        dummies = pd.get_dummies(sub_sector, drop_first=True, dtype=float)
        design_full = pd.concat([sub_ctrl, dummies], axis=1)
        for col in feature_cols:
            y = features.loc[idx, col].astype(float)
            valid = base_valid & y.notna()
            n = int(valid.sum())
            sub_design = design_full.loc[valid]
            # Dummies sin variación dentro de las filas válidas se descartan.
            keep = [c for c in sub_design.columns if sub_design[c].nunique() > 1]
            design = np.column_stack(
                [np.ones(n), sub_design[keep].to_numpy(dtype=float)]
            ) if n else np.empty((0, 1))
            out.loc[idx, col] = np.nan
            if n < design.shape[1] + min_rows_over_params:
                continue
            yv = y.loc[valid].to_numpy(dtype=float)
            coef, *_ = np.linalg.lstsq(design, yv, rcond=None)
            resid = yv - design @ coef
            out.loc[idx[valid.to_numpy()], col] = resid
    return out


# ---------------------------------------------------------------------------
# Orquestador
# ---------------------------------------------------------------------------


@dataclass
class PreEventFeatures:
    """Huella de negociación informada en [T-N, T-1] por evento (contrato §3.5).

    `compute(ctx)` devuelve un DataFrame indexado por ``event_id`` con metadatos
    (`METADATA_COLUMNS`) y, al menos, las columnas del contrato: `volume_runup`,
    `turnover_zscore`, `abnormal_volume_5/10/20d`, `order_imbalance_proxy` (alias
    del BVC ponderado por volumen a 5 días), `pre_event_car_5/10/20d`,
    `short_interest_delta`, `off_exchange_share_delta`, `insider_net_buy_form4`
    (con desglose oportunista/rutinario) y las columnas de opciones
    (`OPTION_FEATURE_COLUMNS`), más las adiciones del informe (§14):
    `turnover_zscore_vs_own_prior_quarters`, `suv_{k}d`, `order_imbalance_clv_{k}d`,
    `pre_event_scar_{k}d` y `amihud_illiquidity` como control.

    Política de degradación explícita: `events`, `prices` y `market` son
    obligatorios (sin ellos, `DataQualityError`); las fuentes con retardo
    institucional (short interest, off-exchange, Form 4, opciones, revisiones de
    analistas) son opcionales y sus columnas salen NaN cuando faltan, con la
    fuente anotada en ``result.attrs['missing_sources']``. Las señales de
    opciones se delegan en `earnings_alpha.events.options_signals` (módulo de
    otro propietario) importado con try/except: si no existe todavía, NaN.

    Con ``residualize=True`` cada feature (salvo el propio CAR y los controles)
    se ortogonaliza por cohorte contra ``pre_event_car_20d``, ``ln(mktcap)``,
    ``ln(ADV)`` y sector (`residualize_features`, informe §12.2). El informe
    documenta que algunas features mueren tras esto; la opción existe para poder
    medirlo, no para decidirlo a priori.
    """

    windows: tuple[int, ...] = flow.DEFAULT_DETECTION_WINDOWS
    runup_k: int = 5
    base_window: tuple[int, int] = flow.DEFAULT_BASE_WINDOW
    estimation: tuple[int, int] = (-250, -40)
    residualize: bool = False
    min_cohort: int = 20
    turnover_method: Literal["empirical", "ar1"] = "empirical"
    insider_window_days: int = 180
    missing_sources_: list[str] = field(default_factory=list, init=False, repr=False)

    def compute(self, ctx: EventContext) -> pd.DataFrame:
        """Calcula la tabla de features por evento a partir del contexto."""
        if ctx.market is None:
            msg = (
                "EventContext.market es obligatorio: sin serie de mercado no hay "
                "retorno anormal pre-evento ni control de residualización"
            )
            raise DataQualityError(msg)
        events_all = flow._normalize_events(ctx.events)
        prices = flow._check_prices(ctx.prices, ["close", "volume"])
        self.missing_sources_ = []

        dates = pd.DatetimeIndex(prices.index.get_level_values("date").unique()).sort_values()
        pos = flow._positions(events_all["event_date"], dates)
        in_panel = pos >= 0
        if not bool(in_panel.any()):
            msg = "ningún evento cae dentro del panel de precios"
            raise InsufficientHistory(msg)
        ids_in = pd.Index(events_all.loc[in_panel, "event_id"], name="event_id")

        meta = self._metadata(ctx, events_all, prices, dates, pos)
        blocks: list[pd.DataFrame] = [meta]

        # --- flujo de precios y volumen (obligatorio) ----------------------
        kw = {"base_window": self.base_window}
        blocks.append(
            flow.volume_runup(prices, events_all, k=self.runup_k, **kw).to_frame()
        )
        blocks.append(
            flow.turnover_zscore(
                prices, events_all, k=self.runup_k, method=self.turnover_method, **kw
            ).to_frame()
        )
        blocks.append(
            flow.abnormal_volume(
                prices, events_all, windows=self.windows, method=self.turnover_method, **kw
            )
        )
        blocks.append(
            flow.turnover_zscore_vs_own_prior_quarters(
                prices, events_all, k=self.runup_k, method=self.turnover_method, **kw
            ).to_frame()
        )
        blocks.append(flow.suv(prices, events_all, windows=self.windows, **kw))
        oib = {
            f"order_imbalance_bvc_{k}d": flow.bvc_order_imbalance(
                prices, events_all, k=int(k), **kw
            )
            for k in self.windows
        }
        oib.update(
            {
                f"order_imbalance_clv_{k}d": flow.clv_order_imbalance(
                    prices, events_all, k=int(k)
                )
                for k in self.windows
            }
        )
        blocks.append(pd.DataFrame(oib))
        blocks.append(
            flow.amihud_illiquidity(prices, events_all, window=self.base_window).to_frame()
        )
        blocks.append(
            pre_event_car(
                prices, ctx.market, events_all,
                windows=self.windows, estimation=self.estimation,
            )
        )

        # --- fuentes con retardo institucional (opcionales) ----------------
        blocks.append(
            self._optional_block(
                "short_interest",
                ctx.short_interest,
                lambda src: flow.short_interest_delta(src, events_all).to_frame(),
                ["short_interest_delta"],
                ids_in,
            )
        )
        blocks.append(
            self._optional_block(
                "off_exchange",
                ctx.off_exchange,
                lambda src: flow.off_exchange_share_delta(src, events_all).to_frame(),
                ["off_exchange_share_delta"],
                ids_in,
            )
        )
        blocks.append(
            self._optional_block(
                "form4",
                ctx.form4,
                lambda src: flow.insider_net_buy_form4(
                    src, events_all, window_days=self.insider_window_days
                ),
                [
                    "insider_net_buy_form4",
                    "insider_net_buy_form4_opportunistic",
                    "insider_net_buy_form4_routine",
                ],
                ids_in,
            )
        )
        blocks.append(self._options_block(ctx, events_all, ids_in))

        # --- ensamblado -----------------------------------------------------
        frame = pd.concat([b.reindex(ids_in) for b in blocks], axis=1)
        first_k = int(self.windows[0])
        frame["order_imbalance_proxy"] = frame[f"order_imbalance_bvc_{first_k}d"]
        if ctx.analyst_revisions is not None:
            frame["analyst_revision_drift"] = pd.Series(ctx.analyst_revisions).reindex(ids_in)
        else:
            frame["analyst_revision_drift"] = np.nan
            self.missing_sources_.append("analyst_revisions")

        feature_cols = [c for c in frame.columns if c not in METADATA_COLUMNS]
        if not frame[feature_cols].notna().to_numpy().any():
            msg = (
                "ninguna feature es calculable para ningún evento: el panel no "
                "cubre las ventanas base/estimación de ningún anuncio"
            )
            raise InsufficientHistory(msg)

        if self.residualize:
            car_cols = [c for c in frame.columns if c.startswith("pre_event_")]
            to_resid = [
                c for c in feature_cols
                if c not in car_cols and pd.api.types.is_numeric_dtype(frame[c])
            ]
            frame = residualize_features(frame, feature_cols=to_resid)

        frame.attrs["missing_sources"] = list(self.missing_sources_)
        frame.attrs["residualized"] = bool(self.residualize)
        frame.attrs["windows"] = tuple(int(k) for k in self.windows)
        return frame

    # ------------------------------------------------------------------ ayudas

    def _metadata(
        self,
        ctx: EventContext,
        events_all: pd.DataFrame,
        prices: pd.DataFrame,
        dates: pd.DatetimeIndex,
        pos: np.ndarray,
    ) -> pd.DataFrame:
        """Metadatos por evento: identidad, cohorte, tamaño, liquidez y volatilidad.

        `available_at` es la medianoche de la sesión T: todas las ventanas
        terminan en T-1, así que lo aquí calculado era público antes de la
        apertura de T. `log_mktcap` usa el cierre de T-1 (público), `log_adv` la
        media del dólar-volumen de la ventana base y `realized_vol_60d` la
        volatilidad realizada anualizada de [T-60, T-1] (control adicional que el
        informe §12.2 recomienda junto a los obligatorios).
        """
        returns = flow._returns_wide(prices)
        if "dollar_volume" in prices.columns:
            dollar = flow._wide(prices, "dollar_volume")
        else:
            dollar = flow._wide(prices, "close") * flow._wide(prices, "volume")
        if "market_cap" in prices.columns:
            mktcap = flow._wide(prices, "market_cap")
        elif "shares_outstanding" in prices.columns:
            mktcap = flow._wide(prices, "close") * flow._wide(prices, "shares_outstanding")
        else:
            mktcap = None

        sector_map: Mapping[str, str] | pd.Series
        if ctx.sectors is not None:
            sector_map = pd.Series(dict(ctx.sectors)) if isinstance(
                ctx.sectors, Mapping
            ) else ctx.sectors
        elif "sector" in events_all.columns:
            sector_map = events_all.set_index("ticker")["sector"]
            sector_map = sector_map[~sector_map.index.duplicated()]
        else:
            sector_map = pd.Series(dtype=object)

        r_cols = {t: returns[t].to_numpy() for t in returns.columns}
        d_cols = {t: dollar[t].to_numpy(dtype=float) for t in dollar.columns}
        m_cols = (
            {t: mktcap[t].to_numpy(dtype=float) for t in mktcap.columns}
            if mktcap is not None
            else {}
        )

        n = len(events_all)
        log_mktcap = np.full(n, np.nan)
        log_adv = np.full(n, np.nan)
        rvol = np.full(n, np.nan)
        for i, row in enumerate(events_all.itertuples(index=False)):
            p = int(pos[i])
            if p <= 0:
                continue
            mc = m_cols.get(row.ticker)
            if mc is not None and np.isfinite(mc[p - 1]) and mc[p - 1] > 0:
                log_mktcap[i] = math.log(mc[p - 1])
            dv = d_cols.get(row.ticker)
            sl = flow._base_slice(p, self.base_window)
            if dv is not None and sl is not None:
                window = dv[sl]
                window = window[np.isfinite(window) & (window > 0)]
                if len(window) >= 20:
                    log_adv[i] = float(np.log(np.mean(window)))
            rr = r_cols.get(row.ticker)
            if rr is not None and p >= 60:
                seg = rr[p - 60 : p]
                seg = seg[np.isfinite(seg)]
                if len(seg) >= 40:
                    rvol[i] = float(np.std(seg, ddof=1)) * np.sqrt(252.0)

        meta = pd.DataFrame(
            {
                "ticker": events_all["ticker"].to_numpy(),
                "event_date": events_all["event_date"].to_numpy(),
                "available_at": events_all["event_date"].to_numpy(),
                "sector": events_all["ticker"].map(sector_map).to_numpy(dtype=object),
                "log_mktcap": log_mktcap,
                "log_adv": log_adv,
                "realized_vol_60d": rvol,
            },
            index=pd.Index(events_all["event_id"], name="event_id"),
        )
        in_panel = pos >= 0
        meta_in = meta.loc[np.asarray(in_panel)]
        meta_in = meta_in.assign(
            cohort=cohort_labels(meta_in["event_date"], min_size=self.min_cohort).to_numpy()
        )
        # Reordena para que `cohort` quede junto a los metadatos canónicos.
        ordered = [c for c in METADATA_COLUMNS if c in meta_in.columns]
        return meta_in[ordered]

    def _optional_block(
        self,
        source_name: str,
        source: pd.DataFrame | None,
        builder,
        columns: Sequence[str],
        ids_in: pd.Index,
    ) -> pd.DataFrame:
        """Calcula un bloque opcional o lo degrada a NaN registrando la fuente."""
        if source is None:
            self.missing_sources_.append(source_name)
            return pd.DataFrame(np.nan, index=ids_in, columns=list(columns))
        block = builder(source)
        block.columns = list(columns)
        return block

    def _options_block(
        self, ctx: EventContext, events_all: pd.DataFrame, ids_in: pd.Index
    ) -> pd.DataFrame:
        """Señales de opciones vía `events.options_signals`, degradadas a NaN.

        El módulo de opciones pertenece a otro propietario (contrato §2). Se
        importa con try/except ImportError; si no existe, o no expone una función
        de features pre-evento reconocible, o falla sobre estos datos, las
        columnas del contrato salen NaN y la degradación queda anotada en
        ``missing_sources`` (y avisada con `warnings.warn` si hubo un error real).
        """
        nan_block = pd.DataFrame(np.nan, index=ids_in, columns=list(OPTION_FEATURE_COLUMNS))
        if _options_signals is None or ctx.options is None:
            self.missing_sources_.append("options")
            return nan_block
        fn = None
        for name in (
            "compute_pre_event_option_features",
            "pre_event_option_features",
            "compute_option_features",
        ):
            fn = getattr(_options_signals, name, None)
            if callable(fn):
                break
        if not callable(fn):
            self.missing_sources_.append("options")
            return nan_block
        try:
            block = fn(ctx.options, events_all)
        except Exception as exc:  # noqa: BLE001 - degradación documentada del contrato
            warnings.warn(
                f"options_signals falló y las columnas de opciones degradan a NaN: {exc!r}",
                stacklevel=2,
            )
            self.missing_sources_.append("options")
            return nan_block
        if not isinstance(block, pd.DataFrame):
            self.missing_sources_.append("options")
            return nan_block
        keep = [c for c in OPTION_FEATURE_COLUMNS if c in block.columns]
        out = nan_block.copy()
        if keep:
            out[keep] = block[keep].reindex(ids_in)
        else:
            self.missing_sources_.append("options")
        return out


# ---------------------------------------------------------------------------
# Score compuesto
# ---------------------------------------------------------------------------


DEFAULT_INTENSITY_WEIGHTS: Final[dict[str, float]] = {
    "turnover_zscore_vs_own_prior_quarters": 1.0,  # Chae (2005), informe §3.6
    "abnormal_volume_5d": 1.0,
    "abnormal_volume_10d": 0.75,
    "suv_10d": 0.75,  # Garfinkel y Sokobin (2006)
    "volume_runup": 0.5,
    "pre_event_scar_5d": 1.0,  # |SCAR|: deriva anticipada en cualquier signo
    "pre_event_scar_10d": 0.5,
    "order_imbalance_bvc_5d": 0.5,  # |OIB|: flujo direccional en cualquier signo
}
"""Pesos por defecto del modo *intensity*: ¿hay actividad pre-anuncio anómala?
Las columnas de `INTENSITY_ABSOLUTE_FEATURES` entran en valor absoluto."""

INTENSITY_ABSOLUTE_FEATURES: Final[frozenset[str]] = frozenset(
    {"pre_event_scar_5d", "pre_event_scar_10d", "order_imbalance_bvc_5d"}
)

DEFAULT_DIRECTIONAL_WEIGHTS: Final[dict[str, float]] = {
    "order_imbalance_bvc_5d": 1.0,  # Campbell, Ramadorai y Schwartz (2009)
    "order_imbalance_clv_5d": 0.5,  # proxy CLV, pendiente de validación (§4.3)
    "pre_event_scar_5d": 0.5,  # anticipación, con reversión parcial (§7.2)
    "suv_10d": 0.5,  # Garfinkel y Sokobin (2006), signo sobre el drift
    "short_interest_delta": 1.0,  # ya orientada mayor = alcista (§8.1)
    "insider_net_buy_form4_opportunistic": 1.0,  # Cohen, Malloy y Pomorski (2012)
    "vol_spread": 1.0,  # Cremers y Weinbaum (2010), positivo
    "iv_skew_25delta": -1.0,  # Xing, Zhang y Zhao (2010), negativo
    "put_call_volume_ratio": -0.5,  # Johnson y So (2012): ¡negativo, contraintuitivo!
}
"""Pesos por defecto del modo *directional* (convención del repo: mayor = más
alcista). Los signos siguen la tabla resumen del informe §1; el volumen bruto
queda a peso cero porque predice magnitud, no dirección."""


@dataclass
class InformedTradingScore:
    """Score compuesto de huella informada: z-scores por cohorte, pesos configurables.

    Cada feature se estandariza **dentro de su cohorte de fecha** (informe §12.1:
    la temporada de resultados concentra los eventos y los niveles calendario no
    son comparables) y, si hay columna de sector y el grupo es suficiente, se le
    resta además la media de su (cohorte, sector). El score es la media ponderada
    de los z disponibles, renormalizando los pesos por evento sobre las features
    no-NaN; los eventos con menos de `min_features` features válidas salen NaN.

    Dos modos:

    - ``intensity`` (por defecto): ¿cuánta actividad anómala hay? Las features
      direccionales entran en valor absoluto. Es el modo para *detección* de
      eventos con huella (el que se valida contra `leaked_event_ids()` del
      mercado sintético).
    - ``directional``: ¿hacia dónde apunta la huella? Convención mayor = más
      alcista, con los signos de la tabla del informe §1.

    Recordatorio de tasa base (informe §12.4, ver `positive_predictive_value`):
    con prevalencia del 2 %, incluso Se=0,80/Sp=0,90 da PPV ≈ 0,14. El score se
    usa para **ponderar exposición**, no como alarma binaria.
    """

    weights: Mapping[str, float] | None = None
    mode: Literal["intensity", "directional"] = "intensity"
    min_features: int = 3
    by_sector: bool = True
    min_sector_group: int = 6
    min_cohort: int = 20

    def effective_weights(self) -> dict[str, float]:
        """Pesos vigentes: los del usuario o los del modo por defecto."""
        if self.weights is not None:
            out = {str(k): float(v) for k, v in self.weights.items()}
            if not out or all(v == 0.0 for v in out.values()):
                msg = "los pesos del score están vacíos o son todos cero"
                raise DataQualityError(msg)
            return out
        return dict(
            DEFAULT_INTENSITY_WEIGHTS if self.mode == "intensity" else DEFAULT_DIRECTIONAL_WEIGHTS
        )

    def score(self, features: pd.DataFrame) -> pd.Series:
        """Calcula el score por event_id a partir de la tabla de `PreEventFeatures`."""
        if not isinstance(features, pd.DataFrame) or len(features) == 0:
            msg = "`features` está vacío o no es un DataFrame"
            raise DataQualityError(msg)
        weights = self.effective_weights()
        present = [c for c in weights if c in features.columns]
        if not present:
            msg = (
                f"ninguna de las columnas ponderadas {sorted(weights)} está en la "
                f"tabla de features"
            )
            raise DataQualityError(msg)

        if "cohort" in features.columns:
            cohort = features["cohort"]
        elif "event_date" in features.columns:
            cohort = cohort_labels(features["event_date"], min_size=self.min_cohort)
        else:
            msg = "`features` necesita una columna 'cohort' o 'event_date'"
            raise DataQualityError(msg)
        sector = features["sector"] if "sector" in features.columns else None

        z_block: dict[str, pd.Series] = {}
        absolute = INTENSITY_ABSOLUTE_FEATURES if self.mode == "intensity" else frozenset()
        for col in present:
            x = features[col].astype(float)
            if col in absolute:
                x = x.abs()
            z = x.groupby(cohort).transform(_group_zscore)
            if self.by_sector and sector is not None:
                sec = sector.astype(object).where(sector.notna(), "_unknown_").astype(str)
                group_mean = z.groupby([cohort, sec]).transform("mean")
                group_size = z.groupby([cohort, sec]).transform("count")
                demeaned = z - group_mean
                # Solo se demeana por sector cuando el grupo (cohorte, sector)
                # tiene tamaño suficiente; con grupos pequeños el z de cohorte
                # se conserva tal cual.
                z = demeaned.where(group_size >= self.min_sector_group, z)
            z_block[col] = z

        z_frame = pd.DataFrame(z_block)
        w = np.array([weights[c] for c in z_frame.columns])
        valid = z_frame.notna().to_numpy()
        n_valid = valid.sum(axis=1)
        z_vals = np.nan_to_num(z_frame.to_numpy(dtype=float), nan=0.0)
        denom = (np.abs(w)[None, :] * valid).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            raw = (z_vals * w[None, :]).sum(axis=1) / denom
        raw[n_valid < self.min_features] = np.nan

        out = pd.Series(raw, index=features.index, name="informed_trading_score")
        if not np.isfinite(out.to_numpy()).any():
            msg = (
                "el score no es calculable para ningún evento: ninguna fila tiene "
                f">= {self.min_features} features ponderadas no nulas"
            )
            raise InsufficientHistory(msg)
        return out


def _group_zscore(s: pd.Series) -> pd.Series:
    """Z-score dentro del grupo; NaN si el grupo no da para una sigma honesta."""
    valid = s.dropna()
    if len(valid) < 3:
        return pd.Series(np.nan, index=s.index)
    sd = float(valid.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0.0:
        return pd.Series(np.nan, index=s.index)
    return (s - float(valid.mean())) / sd
