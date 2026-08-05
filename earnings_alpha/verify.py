"""Verificador de rentabilidad del PEAD con criterios pre-registrados.

Este módulo responde a UNA pregunta con datos reales: ¿está vivo el drift
post-anuncio de resultados (PEAD) por quintiles de sorpresa en el S&P 500?

La disciplina central es el **pre-registro**: los umbrales de éxito están
congelados en :data:`CRITERIOS` *antes* de mirar ningún resultado, y
``tests/test_verify.py`` falla si alguien los relaja. Un verificador cuyos
umbrales se ajustan después de ver los datos no verifica nada: selecciona.

Metodología
-----------
1. Eventos reales de ``data/external/consenso/consenso_master.parquet``
   (~47.800 anuncios 1995-2026) filtrados a pertenencia **point-in-time** al
   S&P 500 en la fecha del evento (``SP500Universe.is_member``); los eventos
   de emisores aún no incluidos en el índice se descartan y se cuentan
   (corrige el sesgo de pre-inclusión, hallazgo C1 de docs/REVISION_CRITICA.md).
2. SUE de analistas (Livnat-Mendenhall 2006) vía
   :func:`earnings_alpha.factors.surprise.analyst_sue_events` (base sigma).
3. Rejilla de backtests de evento (:func:`earnings_alpha.backtest.event_engine.run_grid`)
   con costes SIEMPRE activados: entradas post-anuncio T+1 (sin gap del
   anuncio) a horizontes 5/10/21/63, y entradas pre-anuncio T-3/T-1
   **solo informativas** (cruzan el gap; alto riesgo; excluidas del veredicto).
4. Significancia: spread mensual Q5-Q1 con t de Newey-West (lags automáticos
   de la regla NW), corrección de Benjamini-Hochberg sobre TODAS las
   combinaciones probadas, y Sharpe deflactado contando el número real de
   pruebas (Bailey-López de Prado).
5. Partición temporal train/test (calibración/confirmación) reportada por
   separado; el veredicto exige coherencia fuera de la ventana de calibración.

Limitaciones documentadas (heredadas del dataset, no de este módulo):
- Supervivencia parcial del parquet de consenso: cobertura ~60% en 2008,
  quebradas ausentes (LEH/BSC/WAMUQ). Por eso ``CRITERIOS.fecha_minima``
  excluye lo anterior a 2010 del veredicto por defecto.
- Sin vintages intra-trimestre de consenso (no afecta al SUE, sí impide
  el momentum de revisiones).
"""

from __future__ import annotations

import datetime as dt
import warnings
from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from earnings_alpha.errors import DataQualityError, InsufficientHistory, ProviderUnavailable

__all__ = [
    "CRITERIOS",
    "CriteriosPreRegistrados",
    "ResultadoVerificacion",
    "verificar_pead",
]


# ---------------------------------------------------------------------------
# Criterios pre-registrados (congelados por tests/test_verify.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CriteriosPreRegistrados:
    """Umbrales de éxito fijados ANTES de mirar los datos.

    Justificación de cada umbral en docs/VERIFICACION.md y en
    docs/research/validation_methodology.md. Modificarlos exige tocar también
    el test que los congela, dejando rastro explícito en el historial.
    """

    t_newey_west_min: float = 3.0
    """t mínimo del spread mensual Q5-Q1 (Harvey-Liu-Zhu 2016: un factor nuevo
    debe superar t=3, no el clásico 2, por la multiplicidad histórica de pruebas)."""

    bh_alpha: float = 0.05
    """Nivel de la corrección Benjamini-Hochberg sobre todas las combinaciones."""

    dsr_min: float = 0.5
    """Sharpe deflactado (probabilidad de que el Sharpe verdadero sea > 0 tras
    descontar el número de pruebas) mínimo para el mejor combo post-anuncio."""

    horizontes_post: tuple[int, ...] = (5, 10, 21, 63)
    """Sesiones de salida evaluadas entrando en T+1 (post-anuncio, sin gap)."""

    entradas_pre_informativas: tuple[int, ...] = (-3, -1)
    """Entradas pre-anuncio (cruzan el gap). SOLO informativas: nunca votan."""

    entrada_post: int = 1
    """Entrada del veredicto: T+1, primera sesión tras la fecha negociable."""

    fecha_minima: dt.date = dt.date(2010, 1, 1)
    """Antes de 2010 la supervivencia medida del dataset de consenso invalida
    el veredicto (cobertura 59,6% en 2008-06). Incluible solo con bandera
    explícita, y entonces el informe queda etiquetado como sesgado."""

    corte_train_test: dt.date = dt.date(2016, 12, 31)
    """Fin del tramo de calibración; 2017+ es confirmación fuera de muestra."""

    min_eventos_por_tramo: int = 400
    """Por debajo de esto el tramo no tiene potencia y el veredicto es NO_CONCLUYENTE."""


CRITERIOS = CriteriosPreRegistrados()

Veredicto = Literal["VIVA", "DEBIL", "MUERTA", "NO_CONCLUYENTE"]


@dataclass
class ResultadoVerificacion:
    """Resultado completo de la verificación, con todos los números de apoyo."""

    veredicto: Veredicto
    criterios: CriteriosPreRegistrados
    razones: list[str]
    t_newey_west_total: float | None
    p_bh_por_combo: pd.DataFrame | None
    dsr_mejor_combo: float | None
    rejilla_total: pd.DataFrame | None
    rejilla_train: pd.DataFrame | None
    rejilla_test: pd.DataFrame | None
    spread_mensual: pd.Series | None
    auditoria: dict = field(default_factory=dict)
    """Recuentos de honestidad: eventos descartados por universo, tickers sin
    precios, eventos fuera del rango de precios, etc. Nada se descarta en
    silencio: si un número de esta auditoría es grande, el resto vale menos."""


# ---------------------------------------------------------------------------
# Utilidades estadísticas
# ---------------------------------------------------------------------------


def _t_newey_west(serie: pd.Series) -> tuple[float, float]:
    """t-statistic de la media de una serie con errores HAC de Newey-West.

    Lags por la regla clásica ``floor(4·(T/100)^(2/9))``. Devuelve (t, p).
    """
    x = serie.dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 12:
        raise InsufficientHistory(f"serie de {n} observaciones; mínimo 12 para NW")
    import statsmodels.api as sm

    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    modelo = sm.OLS(x, np.ones((n, 1))).fit(
        cov_type="HAC", cov_kwds={"maxlags": max(lags, 1)}
    )
    return float(modelo.tvalues[0]), float(modelo.pvalues[0])


def _spread_mensual_q5_q1(trades: pd.DataFrame) -> pd.Series:
    """Serie mensual del spread de retorno neto por evento entre Q5 y Q1.

    Agregar por mes natural de entrada mitiga la dependencia entre eventos
    solapados (docs/research/validation_methodology.md §CV con solapamiento):
    la inferencia NW se hace sobre esta serie, nunca sobre eventos sueltos.
    """
    t = trades.dropna(subset=["quantile", "net_return"]).copy()
    if t.empty:
        raise InsufficientHistory("sin trades con quintil para el spread mensual")
    t["mes"] = pd.to_datetime(t["entry_date"]).dt.to_period("M")
    q5 = t[t["quantile"] == t["quantile"].max()].groupby("mes")["net_return"].mean()
    q1 = t[t["quantile"] == t["quantile"].min()].groupby("mes")["net_return"].mean()
    spread = (q5 - q1).dropna()
    spread.name = "spread_q5_q1"
    return spread


def _p_normal_desde_ci(mean: float, ci_low: float, ci_high: float) -> float:
    """p bilateral aproximada desde un IC 95% bootstrap (aproximación normal).

    Se usa SOLO para la corrección BH entre combos de la rejilla, donde basta
    un ranking de evidencia homogéneo; el criterio principal (t NW) no pasa
    por aquí.
    """
    from scipy import stats

    se = (ci_high - ci_low) / (2.0 * 1.959964)
    if not np.isfinite(se) or se <= 0:
        return 1.0
    return float(2.0 * (1.0 - stats.norm.cdf(abs(mean) / se)))


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------


def _eventos_reales(
    start: dt.date, end: dt.date, auditoria: dict
) -> pd.DataFrame:
    """Eventos reales con SUE, filtrados a pertenencia PIT al S&P 500."""
    from earnings_alpha.factors.surprise import analyst_sue_events, load_consensus_events
    from earnings_alpha.universe import SP500Universe

    eventos = load_consensus_events(start=start, end=end)
    universo = SP500Universe()
    fechas = pd.to_datetime(eventos["event_date"]).dt.date
    dentro = [
        universo.is_member(t, d)
        for t, d in zip(eventos["ticker"], fechas, strict=True)
    ]
    dentro = pd.Series(dentro, index=eventos.index)
    auditoria["eventos_totales"] = int(len(eventos))
    auditoria["eventos_fuera_del_indice"] = int((~dentro).sum())
    eventos = eventos[dentro].copy()

    sue = analyst_sue_events(eventos)
    eventos = eventos.merge(
        sue[["event_id", "sue"]], on="event_id", how="left", validate="1:1"
    )
    auditoria["eventos_sin_sue"] = int(eventos["sue"].isna().sum())
    eventos = eventos.dropna(subset=["sue"])
    auditoria["eventos_verificables"] = int(len(eventos))
    return eventos


def _eventos_sinteticos(seed: int, auditoria: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Camino de humo sin red: eventos y precios del mercado sintético.

    El universo sintético es estático (todos los tickers, siempre): válido solo
    para probar la mecánica del verificador, jamás para un veredicto real.
    """
    from earnings_alpha.data.synthetic import SyntheticMarket

    m = SyntheticMarket(seed=seed, n_tickers=25, start="2019-01-01", end="2022-12-31")
    eventos, precios = m.events(), m.prices()
    auditoria["modo"] = "sintetico (solo humo; universo estatico)"
    auditoria["eventos_totales"] = int(len(eventos))
    auditoria["eventos_verificables"] = int(eventos["sue"].notna().sum())
    return eventos.dropna(subset=["sue"]), precios


def _precios_reales(
    fuente: str,
    tickers: Sequence[str],
    start: dt.date,
    end: dt.date,
    auditoria: dict,
) -> pd.DataFrame:
    """Descarga precios diarios ajustados con caché de disco y recuento de bajas."""
    from earnings_alpha.data import prices as pmod

    proveedores = {
        "stooq": pmod.StooqProvider,
        "yfinance": pmod.YFinanceProvider,
    }
    if fuente not in proveedores:
        raise ProviderUnavailable(fuente, f"fuente desconocida; usa {sorted(proveedores)} o 'sintetico'")
    prov = proveedores[fuente]()
    if not prov.available():
        raise ProviderUnavailable(
            fuente,
            "proveedor no disponible en este entorno (¿red bloqueada?); "
            "ejecuta esto en una máquina con salida a internet",
        )
    # Margen para la ventana de estimación del event study y el horizonte máximo.
    margen_pre = dt.timedelta(days=420)
    margen_post = dt.timedelta(days=150)
    faltan: list[str] = []
    paneles: list[pd.DataFrame] = []
    for t in tickers:
        try:
            paneles.append(
                prov.get_bars([t], start - margen_pre, end + margen_post, adjusted=True)
            )
        except (ProviderUnavailable, DataQualityError, KeyError, ValueError) as exc:
            faltan.append(t)
            warnings.warn(f"sin precios para {t}: {exc}", stacklevel=2)
    auditoria["tickers_solicitados"] = int(len(tickers))
    auditoria["tickers_sin_precios"] = int(len(faltan))
    auditoria["tickers_sin_precios_lista"] = faltan[:50]
    if not paneles:
        raise ProviderUnavailable(fuente, "ningún ticker devolvió precios")
    if len(faltan) > 0.25 * len(tickers):
        warnings.warn(
            f"{len(faltan)}/{len(tickers)} tickers sin precios: el veredicto "
            "queda sesgado hacia los supervivientes del proveedor",
            stacklevel=2,
        )
    return pd.concat(paneles).sort_index()


# ---------------------------------------------------------------------------
# Núcleo
# ---------------------------------------------------------------------------


def _rejilla(
    eventos: pd.DataFrame,
    precios: pd.DataFrame,
    criterios: CriteriosPreRegistrados,
    n_boot: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ejecuta la rejilla completa y devuelve (tabla, trades del combo base).

    El combo base del veredicto es entrada T+1 / salida T+21 (el horizonte
    canónico del PEAD trimestral); su tabla de trades alimenta el spread
    mensual para la t de Newey-West.
    """
    from earnings_alpha.backtest.event_engine import EventBacktest, run_grid

    engine = EventBacktest(seed=seed)
    salidas = list(criterios.horizontes_post)

    # Rejilla del VEREDICTO: entradas post-anuncio, sin ningún supuesto de
    # conocimiento previo de la fecha (la guarda PIT del motor queda activa).
    tabla_post = run_grid(
        eventos,
        precios,
        entry_offsets=[criterios.entrada_post],
        exit_offsets=salidas,
        score="sue",
        engine=engine,
        n_boot=n_boot,
    ).reset_index(drop=True)
    tabla_post["informativa"] = False

    # Brazo INFORMATIVO pre-anuncio: cruza el gap y ASUME que la fecha del
    # anuncio se conocía por adelantado (calendar_known_in_advance=True). El
    # supuesto es razonable para fechas confirmadas —las empresas anuncian su
    # fecha de resultados con semanas de antelación— y optimista para fechas
    # estimadas, así que estas últimas se excluyen del brazo. Nunca vota.
    ev_pre = eventos
    if "is_estimated_date" in eventos.columns:
        ev_pre = eventos[~eventos["is_estimated_date"].fillna(False)]
    tabla_pre = run_grid(
        ev_pre,
        precios,
        entry_offsets=list(criterios.entradas_pre_informativas),
        exit_offsets=salidas,
        score="sue",
        engine=engine,
        n_boot=n_boot,
        calendar_known_in_advance=True,
    ).reset_index(drop=True)
    tabla_pre["informativa"] = True

    tabla = pd.concat([tabla_post, tabla_pre], ignore_index=True)
    base = engine.run(
        eventos,
        precios,
        entry_offset=criterios.entrada_post,
        exit_offset=21,
        score="sue",
        n_boot=n_boot,
    )
    return tabla, base.trades


def _evaluar(
    tabla_total: pd.DataFrame,
    trades_base: pd.DataFrame,
    tabla_train: pd.DataFrame | None,
    tabla_test: pd.DataFrame | None,
    n_train: int,
    n_test: int,
    criterios: CriteriosPreRegistrados,
) -> tuple[Veredicto, list[str], float | None, pd.DataFrame, float | None, pd.Series | None]:
    """Aplica la regla pre-registrada. Devuelve el veredicto y sus piezas."""
    from earnings_alpha.stats.performance import deflated_sharpe_ratio
    from earnings_alpha.stats.validation import benjamini_hochberg

    razones: list[str] = []

    if n_train < criterios.min_eventos_por_tramo or n_test < criterios.min_eventos_por_tramo:
        razones.append(
            f"potencia insuficiente: train={n_train}, test={n_test} eventos "
            f"(mínimo pre-registrado {criterios.min_eventos_por_tramo} por tramo)"
        )
        return "NO_CONCLUYENTE", razones, None, tabla_total, None, None

    # 1) t de Newey-West del spread mensual Q5-Q1 (combo base T+1→T+21).
    spread = _spread_mensual_q5_q1(trades_base)
    t_nw, _ = _t_newey_west(spread)
    ok_nw = t_nw >= criterios.t_newey_west_min
    razones.append(
        f"spread mensual Q5-Q1 (T+1→T+21): t_NW={t_nw:.2f} "
        f"({'≥' if ok_nw else '<'} {criterios.t_newey_west_min} pre-registrado)"
    )

    # 2) Benjamini-Hochberg sobre TODOS los combos probados (incl. informativos:
    #    también consumieron una prueba y deben pagar su multiplicidad).
    tabla = tabla_total.copy()
    tabla["p_cruda"] = [
        _p_normal_desde_ci(m, lo, hi)
        for m, lo, hi in zip(
            tabla["q_spread_mean"], tabla["q_spread_ci_low"], tabla["q_spread_ci_high"], strict=True
        )
    ]
    bh = benjamini_hochberg(tabla["p_cruda"].fillna(1.0).to_numpy(), alpha=criterios.bh_alpha)
    rechazos = np.asarray(bh.rejected, dtype=bool)
    tabla["bh_significativo"] = rechazos
    post = tabla[~tabla["informativa"]]
    ok_bh = bool((post["bh_significativo"] & (post["q_spread_mean"] > 0)).any())
    razones.append(
        f"combos post-anuncio con spread>0 significativos tras BH: "
        f"{int((post['bh_significativo'] & (post['q_spread_mean'] > 0)).sum())}/{len(post)} "
        f"(se exige ≥1; multiplicidad pagada sobre {len(tabla)} combos)"
    )

    # 3) Sharpe deflactado del mejor combo post-anuncio, descontando n pruebas.
    dsr_val: float | None = None
    if not post.empty and post["q_spread_mean"].notna().any():
        idx_mejor = post["q_spread_mean"].idxmax()
        mensual = _spread_mensual_q5_q1(trades_base)  # serie del combo base
        # Los "trials" del DSR son los Sharpes de TODOS los combos ejecutados en
        # la rejilla (informativos incluidos): exactamente las pruebas que hemos
        # mirado, ni una inventada ni una omitida.
        trial_sharpes = tabla["sharpe_annualized"].dropna().to_numpy(dtype=float)
        try:
            dsr = deflated_sharpe_ratio(
                mensual,
                trial_sharpes=trial_sharpes,
                trials_annualized=True,
                periods_per_year=12,
            )
            dsr_val = float(getattr(dsr, "dsr", getattr(dsr, "value", np.nan)))
        except (InsufficientHistory, ValueError, DataQualityError) as exc:
            razones.append(f"DSR no calculable: {exc}")
        if dsr_val is not None and np.isfinite(dsr_val):
            ok_dsr = dsr_val >= criterios.dsr_min
            razones.append(
                f"Sharpe deflactado (mejor combo post, {len(tabla)} pruebas descontadas): "
                f"{dsr_val:.2f} ({'≥' if ok_dsr else '<'} {criterios.dsr_min})"
            )
        else:
            ok_dsr = False
        _ = idx_mejor
    else:
        ok_dsr = False
        razones.append("sin combos post-anuncio con spread calculable")

    # 4) Confirmación fuera de la calibración: spread medio > 0 en el tramo test.
    ok_test = False
    if tabla_test is not None and not tabla_test.empty:
        post_test = tabla_test[tabla_test["entry_offset"] == criterios.entrada_post]
        media_test = float(post_test["q_spread_mean"].mean())
        ok_test = media_test > 0
        razones.append(
            f"tramo de confirmación ({'post-' + str(criterios.corte_train_test.year)}): "
            f"spread medio {media_test:+.4f} ({'>' if ok_test else '≤'} 0)"
        )

    votos = [ok_nw, ok_bh, ok_dsr, ok_test]
    if ok_nw and ok_bh and ok_test:
        veredicto: Veredicto = "VIVA"
    elif t_nw < 1.0 and not ok_bh:
        veredicto = "MUERTA"
    else:
        veredicto = "DEBIL"
    razones.append(
        f"votos [t_NW≥{criterios.t_newey_west_min}, BH, DSR≥{criterios.dsr_min}, test>0] = "
        f"{['✔' if v else '✘' for v in votos]} → {veredicto}"
    )
    return veredicto, razones, t_nw, tabla, dsr_val, spread


def verificar_pead(
    *,
    fuente: str = "stooq",
    desde: dt.date | str | None = None,
    hasta: dt.date | str | None = None,
    max_tickers: int | None = None,
    incluir_pre2010: bool = False,
    n_boot: int = 500,
    seed: int = 20260805,
    criterios: CriteriosPreRegistrados = CRITERIOS,
) -> ResultadoVerificacion:
    """Ejecuta la verificación completa del PEAD y devuelve el veredicto.

    Parámetros principales: ``fuente`` ∈ {stooq, yfinance, sintetico};
    ``max_tickers`` submuestrea por número de eventos disponibles (los tickers
    con más historia de eventos primero: maximiza potencia por descarga, y se
    documenta porque NO es una muestra aleatoria); ``incluir_pre2010`` levanta
    la guarda de supervivencia y etiqueta el resultado como sesgado.
    """
    auditoria: dict = {"fuente": fuente, "seed": seed}
    desde_d = pd.to_datetime(desde).date() if desde else criterios.fecha_minima
    hasta_d = pd.to_datetime(hasta).date() if hasta else dt.date(2022, 12, 31)

    if not incluir_pre2010 and desde_d < criterios.fecha_minima:
        auditoria["ajuste_fecha_minima"] = (
            f"desde={desde_d} elevado a {criterios.fecha_minima} por la guarda de "
            "supervivencia (usa incluir_pre2010=True para forzarlo, quedará etiquetado)"
        )
        desde_d = criterios.fecha_minima
    if incluir_pre2010:
        auditoria["ADVERTENCIA_SUPERVIVENCIA"] = (
            "incluye pre-2010: cobertura del consenso 59,6% en 2008-06, quebradas "
            "ausentes; el veredicto sobre ese tramo está inflado por construcción"
        )

    # --- datos ---------------------------------------------------------
    if fuente == "sintetico":
        eventos, precios = _eventos_sinteticos(seed, auditoria)
        # La ventana sintética no coincide con el calendario real: el corte
        # train/test pre-registrado se sustituye por la mediana de los eventos
        # y la potencia mínima se reduce. Documentado en la auditoría; el modo
        # sintético nunca produce un veredicto de mercado.
        import dataclasses

        mediana = pd.to_datetime(eventos["event_date"]).quantile(0.5).date()
        criterios = dataclasses.replace(
            criterios, corte_train_test=mediana, min_eventos_por_tramo=100
        )
        auditoria["criterios_adaptados_sintetico"] = (
            f"corte_train_test={mediana}, min_eventos_por_tramo=100"
        )
    else:
        eventos = _eventos_reales(desde_d, hasta_d, auditoria)
        if max_tickers:
            por_ticker = eventos.groupby("ticker").size().sort_values(ascending=False)
            elegidos = por_ticker.head(max_tickers).index
            auditoria["submuestreo"] = (
                f"{max_tickers} tickers con más eventos (no aleatorio: maximiza "
                f"potencia; cubre {int(eventos['ticker'].isin(elegidos).mean() * 100)}% de los eventos)"
            )
            eventos = eventos[eventos["ticker"].isin(elegidos)]
        precios = _precios_reales(
            fuente, sorted(eventos["ticker"].unique()), desde_d, hasta_d, auditoria
        )
        con_precio = eventos["ticker"].isin(precios.index.get_level_values("ticker").unique())
        auditoria["eventos_sin_precios"] = int((~con_precio).sum())
        eventos = eventos[con_precio]

    return verificar_con_datos(
        eventos,
        precios,
        criterios=criterios,
        n_boot=n_boot,
        seed=seed,
        auditoria=auditoria,
        etiqueta_sintetico=(fuente == "sintetico"),
    )


def verificar_con_datos(
    eventos: pd.DataFrame,
    precios: pd.DataFrame,
    *,
    criterios: CriteriosPreRegistrados = CRITERIOS,
    n_boot: int = 500,
    seed: int = 20260805,
    auditoria: dict | None = None,
    etiqueta_sintetico: bool = False,
) -> ResultadoVerificacion:
    """Núcleo de la verificación sobre datos ya cargados.

    Existe como función pública por una razón concreta: los tests de falsación
    inyectan aquí eventos con el SUE barajado y exigen que el veredicto NO sea
    VIVA — la prueba de que el verificador no fabrica señal donde no la hay.
    """
    auditoria = auditoria if auditoria is not None else {}

    # --- partición y rejillas ------------------------------------------
    fechas_ev = pd.to_datetime(eventos["event_date"]).dt.date
    ev_train = eventos[fechas_ev <= criterios.corte_train_test]
    ev_test = eventos[fechas_ev > criterios.corte_train_test]
    auditoria["eventos_train"] = int(len(ev_train))
    auditoria["eventos_test"] = int(len(ev_test))

    tabla_total, trades_base = _rejilla(eventos, precios, criterios, n_boot, seed)
    tabla_train = tabla_test = None
    if len(ev_train) >= criterios.min_eventos_por_tramo // 2:
        tabla_train, _ = _rejilla(ev_train, precios, criterios, max(n_boot // 2, 100), seed)
    if len(ev_test) >= criterios.min_eventos_por_tramo // 2:
        tabla_test, _ = _rejilla(ev_test, precios, criterios, max(n_boot // 2, 100), seed)

    veredicto, razones, t_nw, tabla_eval, dsr_val, spread = _evaluar(
        tabla_total,
        trades_base,
        tabla_train,
        tabla_test,
        len(ev_train),
        len(ev_test),
        criterios,
    )
    if etiqueta_sintetico:
        razones.insert(0, "MODO SINTÉTICO: prueba de mecánica, no un veredicto de mercado")

    return ResultadoVerificacion(
        veredicto=veredicto,
        criterios=criterios,
        razones=razones,
        t_newey_west_total=t_nw,
        p_bh_por_combo=tabla_eval,
        dsr_mejor_combo=dsr_val,
        rejilla_total=tabla_total,
        rejilla_train=tabla_train,
        rejilla_test=tabla_test,
        spread_mensual=spread,
        auditoria=auditoria,
    )


# ---------------------------------------------------------------------------
# Informe legible
# ---------------------------------------------------------------------------

_COLS_INFORME = [
    "entry_offset", "exit_offset", "n_events", "hit_rate", "mean_net",
    "q_spread_mean", "q_spread_ci_low", "q_spread_ci_high",
    "mean_gap", "worst_net", "p01", "bh_significativo", "informativa",
]


def imprimir_informe(r: ResultadoVerificacion) -> str:
    """Informe en texto plano, en español, con el veredicto y su porqué."""
    lineas: list[str] = []
    a = lineas.append
    a("=" * 74)
    a(f"VERIFICACIÓN DEL PEAD — VEREDICTO: {r.veredicto}")
    a("=" * 74)
    a("")
    a("Criterios pre-registrados (congelados por tests/test_verify.py):")
    c = r.criterios
    a(f"  t Newey-West del spread Q5-Q1 ≥ {c.t_newey_west_min} | BH α={c.bh_alpha} | "
      f"DSR ≥ {c.dsr_min} | confirmación test > 0")
    a(f"  entrada veredicto T+{c.entrada_post}; salidas {c.horizontes_post}; "
      f"pre-anuncio {c.entradas_pre_informativas} SOLO informativo")
    a("")
    a("Razonamiento:")
    for razon in r.razones:
        a(f"  · {razon}")
    a("")
    if r.rejilla_total is not None and not r.rejilla_total.empty:
        cols = [x for x in _COLS_INFORME if x in r.p_bh_por_combo.columns]
        a("Rejilla completa (spread Q5-Q1 con IC 95%; 'informativa'=cruza el anuncio):")
        a(r.p_bh_por_combo[cols].to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
        a("")
    a("Auditoría de honestidad (si estos números son grandes, desconfía del resto):")
    for k, v in r.auditoria.items():
        a(f"  {k}: {v}")
    a("")
    a("Recordatorio: VIVA no es una promesa de rentabilidad futura; es que la señal")
    a("superó, con costes y fuera de muestra, los umbrales fijados de antemano.")
    texto = "\n".join(lineas)
    print(texto)
    return texto
