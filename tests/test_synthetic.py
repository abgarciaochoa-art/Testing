"""Tests del generador de mercado sintético (`earnings_alpha.data.synthetic`).

Se comprueban cuatro familias de propiedades:

1. **Determinismo estricto.** Misma semilla, mismo panel byte a byte. Sin esto, el
   resto de módulos no podría escribir tests reproducibles contra este generador.
2. **Forma y rangos.** Índices canónicos, ausencia de NaN, coherencia OHLC, fechas
   de disponibilidad posteriores al periodo, identidades contables exactas.
3. **Estructura estadística declarada.** Colas gruesas, heterogeneidad de
   volatilidad por sector, estacionalidad semanal y persistencia del volumen,
   respuesta a la sorpresa, PEAD, walk-down del consenso, ramp y colapso de IV.
4. **Verdad-terreno de la filtración.** Los eventos marcados como filtrados
   muestran un run-up de volumen y una deriva de precio firmada significativamente
   mayores que el resto; y con `leak_fraction=0` no hay señal alguna que detectar.
   Esta última pareja de tests es la que convierte al generador en un banco de
   pruebas del detector y no en un simple mock.

Todo corre sin red.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from earnings_alpha.data.synthetic import (
    SECTOR_PROFILES,
    LeakSpec,
    SyntheticConfig,
    SyntheticMarket,
    make_synthetic_market,
)
from earnings_alpha.errors import (
    ConfigError,
    DataQualityError,
    InsufficientHistory,
)
from earnings_alpha.pit import (
    asof_join,
    assert_no_lookahead,
    facts_to_frame,
    first_reported,
    get_calendar,
    tradable_date,
)
from earnings_alpha.types import Session, normalize_ticker

# --------------------------------------------------------------------------- fixtures

SMALL = {"n_tickers": 14, "start": "2020-01-02", "end": "2022-12-30"}
WIDE = {"n_tickers": 26, "start": "2019-01-02", "end": "2023-12-29"}


@pytest.fixture(scope="module")
def market() -> SyntheticMarket:
    """Mercado pequeño para tests estructurales."""
    return SyntheticMarket(seed=4242, **SMALL)


@pytest.fixture(scope="module")
def leaky() -> SyntheticMarket:
    """Mercado amplio con filtraciones, para los tests de potencia estadística."""
    return SyntheticMarket(seed=99, leak_fraction=0.15, **WIDE)


@pytest.fixture(scope="module")
def clean() -> SyntheticMarket:
    """Mismo tamaño, sin ninguna filtración: es el grupo de control."""
    return SyntheticMarket(seed=99, leak_fraction=0.0, **WIDE)


# ------------------------------------------------------------------------- utilidades


def _event_window_stats(mkt: SyntheticMarket) -> pd.DataFrame:
    """Estadísticos pre-evento por evento: run-up de volumen y CAR firmado.

    Replica lo que haría un detector honesto: volumen anormal frente a la *mediana*
    de una ventana base anterior (robusta a los picos de vencimiento trimestral) y
    retorno acumulado en la ventana previa, firmado por el signo de la sorpresa.
    """
    events = mkt.events()
    truth = mkt.ground_truth()
    log_volume = np.log(mkt.wide("volume"))
    returns = mkt.wide("log_return")
    sessions = returns.index
    position = {ts: i for i, ts in enumerate(sessions)}

    rows: list[dict[str, object]] = []
    for event in events.itertuples(index=False):
        pos = position.get(pd.Timestamp(event.event_date))
        if pos is None or pos < 50:
            continue
        truth_row = truth.loc[event.event_id]
        col_v = log_volume[event.ticker]
        col_r = returns[event.ticker]
        base = col_v.iloc[pos - 45 : pos - 25].median()
        rows.append(
            {
                "event_id": event.event_id,
                "ticker": event.ticker,
                "is_leaked": bool(truth_row.is_leaked),
                "sign": float(truth_row.surprise_sign),
                "runup_5d": col_v.iloc[pos - 5 : pos].mean() - base,
                "runup_10d": col_v.iloc[pos - 10 : pos].mean() - base,
                "runup_far": col_v.iloc[pos - 25 : pos - 16].mean() - base,
                "signed_car_10d": float(truth_row.surprise_sign) * col_r.iloc[pos - 10 : pos].sum(),
                "signed_car_far": float(truth_row.surprise_sign)
                * col_r.iloc[pos - 25 : pos - 16].sum(),
            }
        )
    frame = pd.DataFrame(rows)
    assert len(frame) > 100, "muestra insuficiente para los tests estadísticos"
    return frame


@pytest.fixture(scope="module")
def leaky_stats(leaky: SyntheticMarket) -> pd.DataFrame:
    return _event_window_stats(leaky)


@pytest.fixture(scope="module")
def clean_stats(clean: SyntheticMarket) -> pd.DataFrame:
    return _event_window_stats(clean)


def _fingerprint(frame: pd.DataFrame) -> bytes:
    """Huella byte a byte de un panel, insensible al orden de las columnas."""
    return frame.sort_index(axis=1).to_csv().encode("utf-8")


def _panels(mkt: SyntheticMarket) -> dict[str, pd.DataFrame]:
    return {
        "prices": mkt.prices(),
        "fundamentals": mkt.fundamentals(),
        "events": mkt.events(),
        "estimates": mkt.estimates(),
        "options_daily": mkt.options_daily(),
        "options_chain": mkt.options_chain(),
        "short_interest": mkt.short_interest(),
        "off_exchange": mkt.off_exchange(),
        "ground_truth": mkt.ground_truth(),
    }


# ---------------------------------------------------------------------- determinismo


def test_misma_semilla_mismo_panel_byte_a_byte() -> None:
    uno = SyntheticMarket(seed=17, **SMALL)
    otro = SyntheticMarket(seed=17, **SMALL)
    izquierda, derecha = _panels(uno), _panels(otro)
    for nombre in izquierda:
        assert _fingerprint(izquierda[nombre]) == _fingerprint(derecha[nombre]), nombre
    assert uno.leaked_event_ids() == otro.leaked_event_ids()


def test_semillas_distintas_producen_paneles_distintos() -> None:
    uno = SyntheticMarket(seed=17, **SMALL)
    otro = SyntheticMarket(seed=18, **SMALL)
    assert _fingerprint(uno.prices()) != _fingerprint(otro.prices())
    assert _fingerprint(uno.events()) != _fingerprint(otro.events())
    assert uno.tickers == otro.tickers, "el universo no debe depender de la semilla"


def test_llamadas_repetidas_son_estables_y_aisladas(market: SyntheticMarket) -> None:
    primera = market.prices()
    primera.loc[:, "close"] = -1.0  # un consumidor descuidado no debe corromper el estado
    segunda = market.prices()
    assert (segunda["close"] > 0).all()
    pd.testing.assert_frame_equal(segunda, market.prices())


def test_leak_fraction_cambia_la_verdad_terreno() -> None:
    poco = SyntheticMarket(seed=5, leak_fraction=0.05, **SMALL)
    mucho = SyntheticMarket(seed=5, leak_fraction=0.40, **SMALL)
    assert len(poco.leaked_event_ids()) < len(mucho.leaked_event_ids())


# ---------------------------------------------------------------- construcción y API


def test_protocolo_provider(market: SyntheticMarket) -> None:
    """Debe satisfacer estructuralmente el protocolo `data.base.Provider`."""
    assert market.name == "synthetic"
    assert market.available() is True
    assert isinstance(market.kinds, tuple) and market.kinds
    pytest.importorskip("earnings_alpha.data.base")
    from earnings_alpha.data.base import KNOWN_KINDS, Provider

    assert isinstance(market, Provider)
    assert set(market.kinds).issubset(KNOWN_KINDS)


def test_universo_estratificado_por_sector() -> None:
    mkt = SyntheticMarket(seed=1, n_tickers=33, start="2021-01-04", end="2022-12-30")
    sectores = mkt.sectors()
    assert len(mkt.tickers) == 33
    assert sectores.nunique() >= 9, "la muestra debe cubrir casi todos los sectores GICS"
    assert set(sectores).issubset(SECTOR_PROFILES)


def test_tickers_y_ciks_reales(market: SyntheticMarket) -> None:
    assert all(t == normalize_ticker(t) for t in market.tickers)
    ciks = market.cik_map()
    assert set(ciks) == set(market.tickers)
    assert all(len(c) == 10 and c.isdigit() for c in ciks.values())


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"leak_fraction": 1.5}, ConfigError),
        ({"leak_fraction": -0.01}, ConfigError),
        ({"n_tickers": 0}, ConfigError),
        ({"n_tickers": 100_000}, ConfigError),
        ({"start": "2022-01-03", "end": "2021-01-04"}, ConfigError),
        ({"start": "2022-01-03", "end": "2022-02-01"}, InsufficientHistory),
        ({"seed": 1.5}, ConfigError),
    ],
)
def test_construccion_invalida(kwargs: dict, error: type[Exception]) -> None:
    base = {"seed": 3, "n_tickers": 8, "start": "2021-01-04", "end": "2022-12-30"}
    with pytest.raises(error):
        SyntheticMarket(**{**base, **kwargs})


def test_configuracion_invalida_lanza() -> None:
    with pytest.raises(ConfigError):
        SyntheticConfig(volume_ar=1.0).validated()
    with pytest.raises(ConfigError):
        SyntheticConfig(t_df_idio=1.5).validated()
    with pytest.raises(ConfigError):
        SyntheticConfig(leak=LeakSpec(window_days=(9, 3))).validated()


def test_seleccion_explicita_de_tickers() -> None:
    mkt = make_synthetic_market(
        seed=2, tickers=["AAPL", "JPM", "XOM"], start="2021-01-04", end="2022-12-30"
    )
    assert set(mkt.tickers) == {"AAPL", "JPM", "XOM"}
    with pytest.raises(ConfigError):
        SyntheticMarket(seed=2, tickers=["NO.EXISTE"], start="2021-01-04", end="2022-12-30")


# ------------------------------------------------------------------------- precios


def test_panel_de_precios_es_canonico(market: SyntheticMarket) -> None:
    panel = market.prices()
    assert panel.index.names == ["date", "ticker"]
    assert panel.index.is_monotonic_increasing
    assert not panel.index.has_duplicates
    assert not panel.isna().to_numpy().any()

    fechas = panel.index.get_level_values("date").unique()
    calendario = get_calendar(first_year=2019, last_year=2024)
    esperadas = calendario.sessions(market.start, market.end)
    assert list(fechas) == list(esperadas)
    assert (fechas.dayofweek < 5).all()
    assert set(panel.index.get_level_values("ticker").unique()) == set(market.tickers)


def test_ohlc_coherente_y_positivo(market: SyntheticMarket) -> None:
    panel = market.prices()
    assert (panel[["open", "high", "low", "close", "adj_close"]] > 0).to_numpy().all()
    assert (panel["high"] >= panel[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (panel["low"] <= panel[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (panel["high"] >= panel["low"]).all()
    assert (panel["volume"] > 0).all()
    assert np.allclose(panel["volume"] % 100, 0.0), "el volumen se redondea a lotes de 100"


def test_retornos_con_colas_gruesas(market: SyntheticMarket) -> None:
    retornos = market.prices()["log_return"]
    curtosis = stats.kurtosis(retornos.to_numpy(), fisher=True)
    assert curtosis > 1.0, f"colas demasiado finas: exceso de curtosis {curtosis:.2f}"
    assert abs(stats.skew(retornos.to_numpy())) < 3.0


def test_volatilidad_heterogenea_por_sector(leaky: SyntheticMarket) -> None:
    retornos = leaky.wide("log_return")
    vol = retornos.std() * np.sqrt(252)
    por_sector = vol.groupby(leaky.sectors()).median()
    defensivos = [s for s in ("Utilities", "Consumer Staples") if s in por_sector.index]
    agresivos = [
        s for s in ("Information Technology", "Consumer Discretionary", "Energy")
        if s in por_sector.index
    ]
    assert defensivos and agresivos
    assert por_sector[defensivos].max() < por_sector[agresivos].min()
    assert vol.min() > 0.10 and vol.max() < 0.75


def test_volumen_con_estacionalidad_semanal(leaky: SyntheticMarket) -> None:
    log_volumen = np.log(leaky.wide("volume"))
    media_diaria = log_volumen.mean(axis=1)
    por_dia = media_diaria.groupby(media_diaria.index.dayofweek).mean()
    assert len(por_dia) == 5
    # El patrón configurado por defecto es martes alto y viernes bajo.
    assert por_dia.loc[1] > por_dia.loc[4]
    grupos = [media_diaria[media_diaria.index.dayofweek == d].to_numpy() for d in range(5)]
    assert stats.f_oneway(*grupos).pvalue < 0.01


def test_log_volumen_es_persistente(market: SyntheticMarket) -> None:
    log_volumen = np.log(market.wide("volume"))
    autocorrelacion = log_volumen.apply(lambda serie: serie.autocorr(1))
    assert autocorrelacion.min() > 0.3
    assert autocorrelacion.median() > 0.5


def test_adj_close_recoge_los_dividendos(market: SyntheticMarket) -> None:
    close = market.wide("close")
    adj = market.wide("adj_close")
    dividendos = market.wide("dividend")
    pagadores = dividendos.sum()[lambda s: s > 0].index
    assert len(pagadores) > 0
    assert np.allclose(adj.iloc[-1], close.iloc[-1])
    # El ajuste hacia atrás deja el precio ajustado por debajo del cierre bruto.
    assert (adj[pagadores].iloc[0] < close[pagadores].iloc[0]).all()
    retorno_total = np.log(adj[pagadores].iloc[-1] / adj[pagadores].iloc[0])
    retorno_precio = np.log(close[pagadores].iloc[-1] / close[pagadores].iloc[0])
    assert (retorno_total > retorno_precio).all()


def test_splits_dejan_continuo_el_precio_ajustado() -> None:
    """Con splits activados, `close` salta pero `adj_close` no: es la serie operable."""
    mkt = SyntheticMarket(
        seed=3,
        n_tickers=12,
        start="2019-01-02",
        end="2023-12-29",
        config=SyntheticConfig(
            enable_splits=True, split_price_threshold=150.0, split_probability=0.5
        ),
    )
    factor = mkt.wide("split_factor")
    ocurrencias = (factor > 1.0).to_numpy()
    assert ocurrencias.sum() > 0, "el escenario debe producir algún split"
    close = mkt.wide("close")
    ajustado = mkt.wide("adj_close")
    salto_bruto = (close / close.shift(1)).to_numpy()[ocurrencias]
    salto_ajustado = np.abs(np.log(ajustado / ajustado.shift(1)).to_numpy()[ocurrencias])
    assert salto_bruto.max() < 0.75, "el cierre sin ajustar debe reflejar el split"
    assert salto_ajustado.max() < 0.25, "el precio ajustado no debe saltar con el split"
    assert (mkt.prices()["split_factor"] >= 1.0).all()


def test_submuestreo_no_altera_valores(market: SyntheticMarket) -> None:
    completo = market.prices()
    trozo = market.prices(tickers=[market.tickers[0]], start="2021-06-01", end="2021-12-31")
    fechas = completo.index.get_level_values("date")
    esperado = completo[
        (completo.index.get_level_values("ticker") == market.tickers[0])
        & (fechas >= pd.Timestamp("2021-06-01"))
        & (fechas <= pd.Timestamp("2021-12-31"))
    ]
    pd.testing.assert_frame_equal(trozo, esperado)


def test_seleccion_vacia_o_desconocida_falla(market: SyntheticMarket) -> None:
    with pytest.raises(DataQualityError):
        market.prices(tickers=["ZZZZ"])
    with pytest.raises(InsufficientHistory):
        market.prices(start="1999-01-04", end="1999-12-31")


def test_barras_tipadas(market: SyntheticMarket) -> None:
    barras = market.bars(market.tickers[0])
    assert len(barras) == len(market.sessions)
    primera = barras[0]
    assert primera.ticker == market.tickers[0]
    assert primera.low <= primera.close <= primera.high


def test_indice_de_mercado_y_factores(market: SyntheticMarket) -> None:
    indice = market.market_index()
    assert list(indice.index) == list(market.sessions)
    assert (indice["close"] > 0).all()
    factores = market.factor_returns()
    for columna in ("mkt", "smb", "hml", "rf"):
        assert columna in factores.columns
    assert any(c.startswith("sector::") for c in factores.columns)
    assert abs(factores["mkt"].std() * np.sqrt(252) - 0.16) < 0.06


# -------------------------------------------------------------------- fundamentales


def test_identidad_contable_exacta(leaky: SyntheticMarket) -> None:
    f = leaky.fundamentals()
    descuadre = (f["total_assets"] - f["total_liabilities"] - f["total_equity"]).abs()
    assert (descuadre / f["total_assets"]).max() < 1e-9


def test_patrimonio_por_arrastre(leaky: SyntheticMarket) -> None:
    f = leaky.fundamentals().sort_values(["ticker", "period_end"])
    grupo = f.groupby("ticker", sort=False)
    variacion = grupo["total_equity"].diff()
    esperado = (
        f["net_income"] - f["dividends_paid"] + f["stock_compensation"] - f["buybacks"]
    )
    mascara = variacion.notna()
    escala = f["total_equity"].abs().clip(lower=1.0)
    assert ((variacion[mascara] - esperado[mascara]).abs() / escala[mascara]).max() < 1e-9


def test_flujo_de_caja_explica_la_tesoreria(leaky: SyntheticMarket) -> None:
    f = leaky.fundamentals().sort_values(["ticker", "period_end"])
    variacion = f.groupby("ticker", sort=False)["cash"].diff()
    mascara = variacion.notna()
    total = f["cfo"] + f["cfi"] + f["cff"]
    assert np.allclose(variacion[mascara], f["change_in_cash"][mascara])
    escala = f["total_assets"].clip(lower=1.0)
    assert ((variacion[mascara] - total[mascara]).abs() / escala[mascara]).max() < 1e-9


def test_magnitudes_fundamentales_plausibles(leaky: SyntheticMarket) -> None:
    f = leaky.fundamentals()
    assert (f["revenue"] > 0).all()
    assert (f["total_assets"] > 0).all()
    assert np.allclose(f["gross_profit"], f["revenue"] - f["cost_of_revenue"])
    assert np.allclose(f["fcf"], f["cfo"] - f["capex"])
    assert f["net_margin"].between(-1.0, 0.6).all()
    assert 0.02 < f["net_margin"].median() < 0.25
    assert (f["eps_diluted"] < 0).mean() < 0.15, "demasiadas empresas en pérdidas"
    assert (f["shares_outstanding"] > 0).all()
    assert f["accruals"].abs().median() < 0.10


def test_fundamentales_son_point_in_time(leaky: SyntheticMarket) -> None:
    f = leaky.fundamentals()
    assert (f["available_at"] > f["period_end"]).all()
    assert (f["filed_at"] >= f["available_at"]).all()
    retardo = (f["available_at"] - f["period_end"]).dt.days
    assert retardo.min() >= 15
    assert retardo.max() <= 60
    assert 20 <= retardo.median() <= 45


def test_hechos_largos_compatibles_con_vintages(market: SyntheticMarket) -> None:
    hechos = market.fundamental_facts(["revenue", "net_income"])
    esperadas = {
        "ticker",
        "concept",
        "period_end",
        "available_at",
        "value",
        "fiscal_period",
        "unit",
        "form",
        "accession",
        "is_restated",
    }
    assert esperadas.issubset(hechos.columns)
    assert set(hechos["concept"]) == {"revenue", "net_income"}
    assert (hechos["available_at"] > hechos["period_end"]).all()
    # El formato debe ser digerible por el módulo de vintages sin conversión previa.
    canonico = facts_to_frame(hechos)
    assert len(canonico) == len(hechos)
    assert len(first_reported(hechos)) == len(hechos), "sin reexpresiones, un vintage por hecho"
    with pytest.raises(DataQualityError):
        market.fundamental_facts(["no_existe"])


def test_asof_join_sobre_fundamentales_no_mira_al_futuro(market: SyntheticMarket) -> None:
    """El panel debe poder unirse con la maquinaria PIT del repo sin look-ahead."""
    fundamentales = market.fundamentals()[["ticker", "available_at", "revenue", "net_income"]]
    fechas = market.sessions[::40]
    senal = pd.MultiIndex.from_product(
        [fechas, list(market.tickers)], names=["date", "ticker"]
    ).to_frame(index=False)
    unido = asof_join(senal, fundamentales, lag_days=1)
    assert len(unido) == len(fechas) * len(market.tickers)
    assert unido["revenue"].notna().mean() > 0.9
    assert_no_lookahead(unido, market.events(), lag_days=1)


# ------------------------------------------------------------------------- eventos


def test_un_evento_por_ticker_y_trimestre(leaky: SyntheticMarket) -> None:
    eventos = leaky.events()
    assert not eventos.duplicated(["ticker", "period_end"]).any()
    assert eventos["event_id"].is_unique
    assert eventos["event_date"].is_monotonic_increasing
    por_ticker = eventos.groupby("ticker").size()
    assert por_ticker.min() >= 4 * 4


def test_event_id_coincide_con_el_tipo_del_contrato(market: SyntheticMarket) -> None:
    eventos = market.events()
    objetos = market.earnings_events()
    assert len(objetos) == len(eventos)
    assert [e.event_id for e in objetos] == list(eventos["event_id"])
    primero = objetos[0]
    assert isinstance(primero.session, Session)
    assert primero.eps_surprise == pytest.approx(primero.eps_actual - primero.eps_estimate)


def test_fecha_negociable_replicable_con_pit(market: SyntheticMarket) -> None:
    """La sesión negociable debe coincidir con la que calcula `pit.tradable_date`."""
    calendario = market.calendar
    eventos = market.events()
    recalculada = [
        tradable_date(pd.Timestamp(fila.announced_at).to_pydatetime(), fila.session, calendario)
        for fila in eventos.itertuples(index=False)
    ]
    assert list(pd.to_datetime(recalculada)) == list(eventos["event_date"])
    # Un anuncio AMC nunca es negociable el mismo día natural.
    amc = eventos[eventos["session"] == "amc"]
    assert (amc["event_date"] > amc["announced_at"].dt.normalize()).all()


def test_eventos_dentro_del_panel_de_precios(market: SyntheticMarket) -> None:
    eventos = market.events()
    assert set(eventos["event_date"]).issubset(set(market.sessions))
    con_historia = market.events(include_prehistory=True)
    assert len(con_historia) > len(eventos)
    assert (con_historia["period_end"].min() < eventos["period_end"].min())


def test_reparto_bmo_amc(leaky: SyntheticMarket) -> None:
    sesiones = leaky.events()["session"]
    assert set(sesiones).issubset({"bmo", "amc"})
    assert 0.25 < (sesiones == "amc").mean() < 0.80


def test_sorpresas_mayoritariamente_positivas(leaky: SyntheticMarket) -> None:
    eventos = leaky.events()
    tasa = (eventos["eps_surprise"] > 0).mean()
    assert 0.55 < tasa < 0.82, f"tasa de batidas irreal: {tasa:.2%}"
    assert eventos["sue"].notna().mean() > 0.7


def test_eps_coherente_con_los_fundamentales(market: SyntheticMarket) -> None:
    eventos = market.events()
    f = market.fundamentals()[["ticker", "period_end", "net_income", "shares_diluted"]]
    unido = eventos.merge(f, on=["ticker", "period_end"], validate="one_to_one")
    assert np.allclose(unido["eps_actual"], unido["net_income"] / unido["shares_diluted"])


def test_eventos_no_exponen_la_etiqueta_de_filtracion(leaky: SyntheticMarket) -> None:
    """La verdad-terreno vive en `ground_truth()`, nunca en los feeds observables.

    Si la etiqueta viajase con los datos, cualquier modelo entrenado sobre ellos
    estaría contaminado y las métricas del detector no significarían nada.
    """
    prohibidas = {
        "is_leaked",
        "leak_intensity",
        "leak_window_days",
        "surprise_z",
        "next_event_z",
        "jump_noise",
    }
    for panel in (leaky.events(), leaky.prices(), leaky.options_daily(), leaky.estimates()):
        assert not prohibidas & set(panel.columns)


# --------------------------------------------------------------------- estimaciones


def test_consenso_estrictamente_anterior_al_anuncio(market: SyntheticMarket) -> None:
    estimaciones = market.estimates()
    eventos = market.events(include_prehistory=True)[["event_id", "announced_at"]]
    unido = estimaciones.merge(eventos, on="event_id")
    assert (unido["as_of"] < unido["announced_at"]).all()
    assert (unido["available_at"] == unido["as_of"]).all()
    conteo = estimaciones.groupby("event_id").size()
    assert (conteo == conteo.iloc[0]).all()


def test_consenso_final_es_el_del_evento(market: SyntheticMarket) -> None:
    ultimo = market.estimates().groupby("event_id").last()
    eventos = market.events(include_prehistory=True).set_index("event_id")
    assert np.allclose(ultimo["eps_mean"], eventos["eps_estimate"].reindex(ultimo.index))


def test_walk_down_del_consenso(leaky: SyntheticMarket) -> None:
    """El consenso arranca por encima del resultado y termina por debajo (batible)."""
    estimaciones = leaky.estimates()
    eventos = leaky.events(include_prehistory=True).set_index("event_id")
    grupo = estimaciones.groupby("event_id")
    primero, ultimo = grupo.first(), grupo.last()
    real = eventos["eps_actual"].reindex(primero.index)
    sesgo_inicial = (primero["eps_mean"] - real).mean()
    sesgo_final = (ultimo["eps_mean"] - real).mean()
    assert sesgo_inicial > 0 > sesgo_final
    assert (primero["eps_mean"] - ultimo["eps_mean"]).mean() > 0


def test_la_senda_del_consenso_converge(market: SyntheticMarket) -> None:
    estimaciones = market.estimates()
    final = estimaciones.groupby("event_id")["eps_mean"].transform("last")
    dias = (
        estimaciones.groupby("event_id")["as_of"].transform("max") - estimaciones["as_of"]
    ).dt.days
    desviacion = (estimaciones["eps_mean"] - final).abs().groupby(dias).mean()
    correlacion = stats.spearmanr(desviacion.index.to_numpy(), desviacion.to_numpy()).statistic
    assert correlacion > 0.95, "la dispersión debe crecer al alejarse del evento"
    assert desviacion.iloc[-1] > 3 * desviacion.iloc[1]


def test_dispersion_y_numero_de_analistas(market: SyntheticMarket) -> None:
    estimaciones = market.estimates()
    assert (estimaciones["eps_std"] > 0).all()
    assert (estimaciones["eps_high"] > estimaciones["eps_low"]).all()
    assert (estimaciones["n_analysts"] >= 3).all()


# ---------------------------------------------------------------- respuesta y drift


def test_la_sorpresa_explica_parte_del_retorno_del_evento(leaky: SyntheticMarket) -> None:
    """Coeficiente de respuesta al beneficio (Ball y Brown 1968): positivo y parcial."""
    retornos = leaky.wide("log_return")
    posicion = {ts: i for i, ts in enumerate(retornos.index)}
    verdad = leaky.ground_truth()
    x, y = [], []
    for evento in leaky.events().itertuples(index=False):
        pos = posicion.get(pd.Timestamp(evento.event_date))
        if pos is None:
            continue
        x.append(np.clip(verdad.loc[evento.event_id, "surprise_z"], -4, 4))
        y.append(retornos[evento.ticker].iloc[pos])
    ajuste = stats.linregress(x, y)
    assert ajuste.slope > 0
    assert ajuste.slope / ajuste.stderr > 4.0
    assert 0.02 < ajuste.rvalue**2 < 0.60, "la relación debe ser parcial, no determinista"


def test_existe_drift_posterior_al_anuncio(leaky: SyntheticMarket) -> None:
    """PEAD (Bernard y Thomas 1989): la sorpresa sigue pagando tras el anuncio.

    Se mide sobre retorno anormal (mercado y sector descontados con las cargas
    verdaderas), como en la literatura: el retorno bruto a 20 sesiones está dominado
    por los factores comunes y ahogaría un drift de esta magnitud.
    """
    retornos = leaky.wide("log_return")
    factores = leaky.factor_returns()
    meta = leaky.metadata()
    posicion = {ts: i for i, ts in enumerate(retornos.index)}
    verdad = leaky.ground_truth()
    x, y = [], []
    for evento in leaky.events().itertuples(index=False):
        pos = posicion.get(pd.Timestamp(evento.event_date))
        if pos is None or pos + 21 >= len(retornos):
            continue
        ventana = slice(pos + 1, pos + 21)
        sector = factores[f"sector::{meta.loc[evento.ticker, 'sector']}"]
        anormal = (
            retornos[evento.ticker].iloc[ventana].sum()
            - float(meta.loc[evento.ticker, "beta_mkt"]) * factores["mkt"].iloc[ventana].sum()
            - float(meta.loc[evento.ticker, "beta_sector"]) * sector.iloc[ventana].sum()
        )
        x.append(np.clip(verdad.loc[evento.event_id, "surprise_z"], -4, 4))
        y.append(anormal)
    ajuste = stats.linregress(x, y)
    assert ajuste.slope > 0
    assert ajuste.slope / ajuste.stderr > 2.5


# ------------------------------------------------------------------------ opciones


def test_cadena_de_opciones_bien_formada(market: SyntheticMarket) -> None:
    cadena = market.options_chain(tickers=market.tickers[:3], asof=market.sessions[-30])
    assert len(cadena) == 3 * 4 * 9 * 2
    assert set(cadena["right"]) == {"C", "P"}
    assert (cadena["iv"] > 0).all() and (cadena["iv"] < 3.0).all()
    assert (cadena["strike"] > 0).all()
    assert (cadena["bid"] <= cadena["mid"] + 1e-9).all()
    assert (cadena["mid"] <= cadena["ask"] + 1e-9).all()
    assert (cadena["open_interest"] >= 0).all() and (cadena["volume"] >= 0).all()
    assert (cadena["days_to_expiry"] > 0).all()

    calls = cadena[cadena["right"] == "C"]
    puts = cadena[cadena["right"] == "P"]
    assert calls["delta"].between(0.0, 1.0).all()
    assert puts["delta"].between(-1.0, 0.0).all()
    assert (cadena["gamma"] > 0).all() and (cadena["vega"] > 0).all()


def test_superficie_con_smile_y_estructura_temporal(market: SyntheticMarket) -> None:
    cadena = market.options_chain(asof=market.sessions[-30])
    cadena = cadena.assign(moneyness=cadena["strike"] / cadena["spot"])
    corto = cadena[cadena["days_to_expiry"] == cadena["days_to_expiry"].min()]
    atm = corto[(corto["moneyness"] - 1.0).abs() < 0.02]["iv"].mean()
    alas = corto[(corto["moneyness"] - 1.0).abs() > 0.12]["iv"].mean()
    assert alas > atm, "la superficie debe tener smile"
    # Estructura temporal: la IV no es plana en el plazo.
    por_plazo = cadena.groupby("days_to_expiry")["iv"].mean()
    assert por_plazo.std() > 1e-3


def test_agregados_de_la_cadena_coinciden_con_el_panel(market: SyntheticMarket) -> None:
    dia = market.sessions[-30]
    cadena = market.options_chain(asof=dia)
    panel = market.options_daily().xs(dia, level="date")
    resumen = cadena.groupby(["ticker", "right"])[["volume", "open_interest"]].sum()
    for ticker in market.tickers:
        assert resumen.loc[(ticker, "C"), "volume"] == pytest.approx(
            panel.loc[ticker, "call_volume"]
        )
        assert resumen.loc[(ticker, "P"), "open_interest"] == pytest.approx(
            panel.loc[ticker, "put_open_interest"]
        )


def test_iv_sube_hacia_el_evento_y_colapsa_despues(leaky: SyntheticMarket) -> None:
    """Varianza de evento aditiva (Dubinsky y Johannes 2006): ramp y luego IV crush.

    El colapso debe ocurrir en tau=0 y no en tau=+1: en la sesión negociable la
    noticia ya es pública desde la apertura, tanto si se anunció BMO ese día como si
    fue AMC la víspera. Un generador que retrasase el crush un día induciría al
    módulo de eventos a modelar mal el calendario de opciones.
    """
    iv = leaky.wide("iv_atm_30d", panel="options")
    posicion = {ts: i for i, ts in enumerate(iv.index)}
    lejos, vispera, dia, siguiente = [], [], [], []
    for evento in leaky.events().itertuples(index=False):
        pos = posicion.get(pd.Timestamp(evento.event_date))
        if pos is None or pos < 30 or pos + 2 >= len(iv):
            continue
        columna = iv[evento.ticker]
        lejos.append(columna.iloc[pos - 25])
        vispera.append(columna.iloc[pos - 1])
        dia.append(columna.iloc[pos])
        siguiente.append(columna.iloc[pos + 1])
    lejos, vispera, dia, siguiente = map(np.asarray, (lejos, vispera, dia, siguiente))
    assert vispera.mean() > lejos.mean()
    assert stats.ttest_rel(vispera, lejos).statistic > 4.0
    assert dia.mean() < vispera.mean()
    assert stats.ttest_rel(dia, vispera).statistic < -6.0
    assert siguiente.mean() < vispera.mean()


def test_panel_diario_de_opciones_coherente(market: SyntheticMarket) -> None:
    panel = market.options_daily()
    assert panel.index.names == ["date", "ticker"]
    assert (panel["iv_atm_30d"] > 0).all()
    assert (panel["call_volume"] >= 0).all() and (panel["put_volume"] >= 0).all()
    assert (panel["put_call_volume_ratio"] > 0).all()
    assert panel["vol_spread"].abs().max() < 0.5
    assert (panel["days_to_earnings"] >= 0).all()
    assert not panel.isna().to_numpy().any()


def test_fecha_de_cadena_invalida(market: SyntheticMarket) -> None:
    with pytest.raises(DataQualityError):
        market.options_chain(asof="2021-01-01")  # festivo


# ------------------------------------------------------------------ short y off-exchange


def test_short_interest_quincenal_y_publicado_con_retardo(market: SyntheticMarket) -> None:
    si = market.short_interest()
    assert (si["available_at"] > si["settlement_date"]).all()
    assert (si["shares_short"] > 0).all()
    assert (si["shares_short"] < si["shares_outstanding"]).all()
    assert (si["days_to_cover"] > 0).all()
    fechas = si["settlement_date"].drop_duplicates().sort_values()
    por_mes = fechas.groupby([fechas.dt.year, fechas.dt.month]).size()
    # Los meses de los extremos pueden quedar truncados por el rango del panel.
    assert set(por_mes.iloc[1:-1].unique()) == {2}
    assert si["short_percent_shares"].between(0.0005, 0.6).all()


def test_off_exchange_semanal_con_retardo_de_publicacion(market: SyntheticMarket) -> None:
    ox = market.off_exchange()
    assert (ox["week_end"].dt.dayofweek == 4).all()
    assert (ox["week_end"] - ox["week_start"] == pd.Timedelta(days=4)).all()
    assert ox["off_exchange_share"].between(0.05, 0.9).all()
    assert (ox["off_exchange_volume"] <= ox["total_volume"]).all()
    assert np.allclose(
        ox["ats_volume"] + ox["non_ats_volume"], ox["off_exchange_volume"], atol=2.0
    )
    retardo = (ox["available_at"] - ox["week_end"]).dt.days
    assert retardo.min() >= 14, "la transparencia ATS de FINRA no es inmediata"


# ------------------------------------------------------- verdad-terreno de filtración


def test_verdad_terreno_consistente(leaky: SyntheticMarket) -> None:
    eventos = set(leaky.events()["event_id"])
    filtrados = leaky.leaked_event_ids()
    assert filtrados == sorted(filtrados)
    assert set(filtrados).issubset(eventos)
    assert len(set(filtrados)) == len(filtrados)

    verdad = leaky.ground_truth()
    assert verdad.index.is_unique
    assert eventos.issubset(set(verdad.index))
    marcados = verdad.loc[verdad["is_leaked"]]
    assert set(marcados.index) == set(filtrados)
    assert (marcados["leak_window_days"] >= 1).all()
    assert (marcados["leak_intensity"] > 0).all()
    limpios = verdad.loc[~verdad["is_leaked"]]
    assert (limpios["leak_window_days"] == 0).all()
    assert (limpios["leak_intensity"] == 0.0).all()
    assert set(verdad["surprise_sign"]) <= {-1.0, 1.0}


def test_fraccion_de_filtraciones_cercana_a_la_pedida(leaky: SyntheticMarket) -> None:
    n_eventos = len(leaky.events())
    observada = len(leaky.leaked_event_ids()) / n_eventos
    esperada = leaky.leak_fraction
    tolerancia = 3.5 * np.sqrt(esperada * (1 - esperada) / n_eventos)
    assert abs(observada - esperada) < tolerancia, f"{observada:.3f} frente a {esperada:.3f}"


def test_filtrados_muestran_run_up_de_volumen(leaky_stats: pd.DataFrame) -> None:
    """Prueba central: la huella de volumen existe y es estadísticamente detectable."""
    filtrados = leaky_stats.loc[leaky_stats["is_leaked"], "runup_5d"]
    limpios = leaky_stats.loc[~leaky_stats["is_leaked"], "runup_5d"]
    assert len(filtrados) >= 30
    prueba = stats.ttest_ind(filtrados, limpios, equal_var=False)
    assert filtrados.mean() - limpios.mean() > 0.15
    assert prueba.statistic > 4.0
    assert prueba.pvalue < 1e-3
    # Robusto a la distribución: también con un contraste no paramétrico.
    assert stats.mannwhitneyu(filtrados, limpios, alternative="greater").pvalue < 1e-3


def test_filtrados_muestran_deriva_de_precio_firmada(leaky_stats: pd.DataFrame) -> None:
    filtrados = leaky_stats.loc[leaky_stats["is_leaked"], "signed_car_10d"]
    limpios = leaky_stats.loc[~leaky_stats["is_leaked"], "signed_car_10d"]
    prueba = stats.ttest_ind(filtrados, limpios, equal_var=False)
    assert filtrados.mean() > limpios.mean()
    assert prueba.statistic > 3.0
    assert prueba.pvalue < 0.01


def test_la_huella_esta_localizada_en_la_ventana(leaky_stats: pd.DataFrame) -> None:
    """Fuera de la ventana de filtración no debe haber diferencia entre grupos.

    Es el contraste de especificidad: si el generador separase a los dos grupos en
    cualquier horizonte, el detector podría "acertar" por un artefacto global en vez
    de por la huella pre-anuncio.
    """
    for columna in ("runup_far", "signed_car_far"):
        filtrados = leaky_stats.loc[leaky_stats["is_leaked"], columna]
        limpios = leaky_stats.loc[~leaky_stats["is_leaked"], columna]
        prueba = stats.ttest_ind(filtrados, limpios, equal_var=False)
        assert abs(prueba.statistic) < 3.0, f"{columna}: t={prueba.statistic:.2f}"


def test_flujo_de_opciones_sesgado_en_los_filtrados(leaky: SyntheticMarket) -> None:
    vol_spread = leaky.wide("vol_spread", panel="options")
    ratio = leaky.wide("put_call_volume_ratio", panel="options")
    posicion = {ts: i for i, ts in enumerate(vol_spread.index)}
    verdad = leaky.ground_truth()
    filas = []
    for evento in leaky.events().itertuples(index=False):
        pos = posicion.get(pd.Timestamp(evento.event_date))
        if pos is None or pos < 50:
            continue
        fila = verdad.loc[evento.event_id]
        signo = float(fila.surprise_sign)
        filas.append(
            {
                "is_leaked": bool(fila.is_leaked),
                "vs": signo * vol_spread[evento.ticker].iloc[pos - 5 : pos].mean(),
                "pc": signo
                * np.log(
                    ratio[evento.ticker].iloc[pos - 5 : pos].mean()
                    / ratio[evento.ticker].iloc[pos - 45 : pos - 25].median()
                ),
            }
        )
    datos = pd.DataFrame(filas)
    vs = stats.ttest_ind(
        datos.loc[datos["is_leaked"], "vs"], datos.loc[~datos["is_leaked"], "vs"], equal_var=False
    )
    assert vs.statistic > 4.0, "el diferencial call-put debe inclinarse hacia la sorpresa"
    pc = stats.ttest_ind(
        datos.loc[datos["is_leaked"], "pc"], datos.loc[~datos["is_leaked"], "pc"], equal_var=False
    )
    assert pc.statistic < -4.0, "el flujo debe desplazarse al lado favorecido"


def test_off_exchange_sube_antes_de_los_eventos_filtrados(leaky: SyntheticMarket) -> None:
    semanal = leaky.off_exchange()
    cuota = semanal.pivot(  # noqa: PD010 - clave única, no hay nada que agregar
        index="week_end", columns="ticker", values="off_exchange_share"
    )
    verdad = leaky.ground_truth()
    filas = []
    for evento in leaky.events().itertuples(index=False):
        dia = pd.Timestamp(evento.event_date)
        semanas = cuota.index[cuota.index < dia]
        if len(semanas) < 14:
            continue
        serie = cuota[evento.ticker]
        filas.append(
            {
                "is_leaked": bool(verdad.loc[evento.event_id, "is_leaked"]),
                "delta": serie.loc[semanas[-1]] - serie.loc[semanas[-12:-3]].median(),
            }
        )
    datos = pd.DataFrame(filas)
    prueba = stats.ttest_ind(
        datos.loc[datos["is_leaked"], "delta"],
        datos.loc[~datos["is_leaked"], "delta"],
        equal_var=False,
    )
    assert prueba.statistic > 3.5
    assert prueba.pvalue < 0.01


# ------------------------------------------------------------- ausencia de filtración


def test_sin_filtracion_no_hay_verdad_terreno(clean: SyntheticMarket) -> None:
    assert clean.leaked_event_ids() == []
    verdad = clean.ground_truth()
    assert not verdad["is_leaked"].any()
    assert (verdad["leak_intensity"] == 0.0).all()


def test_sin_filtracion_no_hay_senal_detectable(clean_stats: pd.DataFrame) -> None:
    """Contraste de la hipótesis nula: sin inyección, nada que detectar.

    Es tan importante como el test de potencia: si el generador dejara una huella
    residual en los eventos limpios, cualquier detector mediría una precisión
    inflada y el banco de pruebas dejaría de servir para calibrar falsos positivos.
    """
    assert not clean_stats["is_leaked"].any()
    for columna in ("runup_5d", "runup_10d", "signed_car_10d"):
        prueba = stats.ttest_1samp(clean_stats[columna], 0.0)
        assert abs(prueba.statistic) < 3.0, f"{columna}: t={prueba.statistic:.2f}"


def test_particion_aleatoria_no_separa_al_mercado_limpio(clean_stats: pd.DataFrame) -> None:
    """Una etiqueta falsa del 15 % no debe distinguirse: fija el listón del detector."""
    rng = np.random.default_rng(0)
    falsa = rng.random(len(clean_stats)) < 0.15
    prueba = stats.ttest_ind(
        clean_stats.loc[falsa, "runup_5d"], clean_stats.loc[~falsa, "runup_5d"], equal_var=False
    )
    assert abs(prueba.statistic) < 3.0


def test_ventanas_de_filtracion_caben_en_el_panel(leaky: SyntheticMarket) -> None:
    verdad = leaky.ground_truth()
    marcados = verdad.loc[verdad["is_leaked"]]
    sesiones = leaky.sessions
    for event_id, fila in marcados.iterrows():
        pos = sesiones.get_loc(pd.Timestamp(fila["event_date"]))
        assert pos - int(fila["leak_window_days"]) >= 0, event_id
    minimo, maximo = leaky.config.leak.window_days
    assert marcados["leak_window_days"].between(minimo, maximo).all()


def test_intensidad_de_la_filtracion_modula_la_huella(leaky_stats: pd.DataFrame) -> None:
    """A mayor amplitud configurada, mayor huella: la inyección es la causa."""
    suave = SyntheticMarket(
        seed=99,
        leak_fraction=0.15,
        config=SyntheticConfig(leak=LeakSpec(volume_log_amp=0.05, price_drift_total=0.002)),
        **WIDE,
    )
    stats_suave = _event_window_stats(suave)
    fuerte_gap = (
        leaky_stats.loc[leaky_stats["is_leaked"], "runup_5d"].mean()
        - leaky_stats.loc[~leaky_stats["is_leaked"], "runup_5d"].mean()
    )
    suave_gap = (
        stats_suave.loc[stats_suave["is_leaked"], "runup_5d"].mean()
        - stats_suave.loc[~stats_suave["is_leaked"], "runup_5d"].mean()
    )
    assert fuerte_gap > 3 * suave_gap


# ------------------------------------------------------------------------ metadatos


def test_metadatos_exponen_la_verdad_del_modelo(market: SyntheticMarket) -> None:
    meta = market.metadata()
    assert set(meta.index) == set(market.tickers)
    for columna in ("beta_mkt", "beta_smb", "beta_hml", "idio_vol", "sector", "jump_vol"):
        assert columna in meta.columns
    assert meta["idio_vol"].between(0.05, 1.0).all()
    assert meta["beta_mkt"].between(0.0, 2.5).all()
    assert set(meta["fy_end_month"]).issubset({1, 6, 9, 12})


def test_betas_verdaderas_son_recuperables(leaky: SyntheticMarket) -> None:
    """Regresando contra los factores verdaderos debe recuperarse la beta declarada."""
    retornos = leaky.wide("log_return")
    factores = leaky.factor_returns()
    meta = leaky.metadata()
    errores = []
    for ticker in leaky.tickers:
        diseño = np.column_stack(
            [
                np.ones(len(factores)),
                factores["mkt"].to_numpy(),
                factores["smb"].to_numpy(),
                factores["hml"].to_numpy(),
                factores[f"sector::{meta.loc[ticker, 'sector']}"].to_numpy(),
            ]
        )
        coef, *_ = np.linalg.lstsq(diseño, retornos[ticker].to_numpy(), rcond=None)
        errores.append(abs(coef[1] - float(meta.loc[ticker, "beta_mkt"])))
    assert np.median(errores) < 0.15


def test_calendarios_fiscales_desplazados(leaky: SyntheticMarket) -> None:
    eventos = leaky.events()
    meses = pd.DatetimeIndex(eventos["period_end"]).month.unique()
    assert len(meses) > 4, "debe haber ejercicios fiscales no naturales"
    etiquetas = eventos["fiscal_quarter"]
    assert etiquetas.str.match(r"^\d{4}Q[1-4]$").all()
    # Cada empresa recorre los cuatro trimestres fiscales.
    por_ticker = eventos.groupby("ticker")["fiscal_quarter"].apply(lambda s: s.str[-1].nunique())
    assert (por_ticker == 4).all()


def test_dias_hasta_resultados_decrecen(market: SyntheticMarket) -> None:
    dias = market.wide("days_to_earnings", panel="options")
    ticker = market.tickers[0]
    serie = dias[ticker]
    saltos = serie.diff()
    # Entre anuncios la cuenta atrás decrece; solo salta hacia arriba al pasar uno.
    assert (saltos <= 0).mean() > 0.9
    assert serie.min() == 0.0


def test_fechas_de_evento_son_sesiones_y_no_festivos(leaky: SyntheticMarket) -> None:
    calendario = leaky.calendar
    for fecha in leaky.events()["event_date"].drop_duplicates():
        dia = pd.Timestamp(fecha).date()
        assert calendario.is_session(dia)
        assert not calendario.is_holiday(dia)
        assert isinstance(dia, dt.date)
