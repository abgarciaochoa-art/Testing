"""Tests del verificador de PEAD.

Tres funciones de estos tests son de contrato, no de cobertura:

1. **Congelar los criterios pre-registrados**: si alguien relaja un umbral en
   ``earnings_alpha/verify.py`` sin tocar este fichero, la suite se pone roja.
   Ese es el mecanismo de pre-registro: cambiar las reglas deja rastro.
2. **Sensibilidad**: sobre el mercado sintético, que INYECTA drift post-anuncio
   por construcción, el veredicto no puede ser MUERTA.
3. **Falsación**: con el SUE barajado (sin relación real señal-retorno), el
   veredicto no puede ser VIVA. Un verificador que aprueba ruido no verifica.
"""

from __future__ import annotations

import dataclasses
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from earnings_alpha.verify import (
    CRITERIOS,
    CriteriosPreRegistrados,
    verificar_con_datos,
    verificar_pead,
)


# ---------------------------------------------------------------------------
# 1. Criterios congelados
# ---------------------------------------------------------------------------


def test_criterios_pre_registrados_congelados():
    """Los umbrales del veredicto son EXACTAMENTE los pre-registrados.

    Si este test te molesta porque quieres cambiar un umbral: ese es su
    trabajo. Cámbialo aquí Y en verify.py en el mismo commit, con la
    justificación en el mensaje, y asume que el cambio es visible.
    """
    assert CRITERIOS.t_newey_west_min == 3.0
    assert CRITERIOS.bh_alpha == 0.05
    assert CRITERIOS.dsr_min == 0.5
    assert CRITERIOS.horizontes_post == (5, 10, 21, 63)
    assert CRITERIOS.entradas_pre_informativas == (-3, -1)
    assert CRITERIOS.entrada_post == 1
    assert CRITERIOS.fecha_minima == dt.date(2010, 1, 1)
    assert CRITERIOS.corte_train_test == dt.date(2016, 12, 31)
    assert CRITERIOS.min_eventos_por_tramo == 400
    # La dataclass es inmutable: nadie ajusta umbrales en caliente.
    with pytest.raises(dataclasses.FrozenInstanceError):
        CRITERIOS.t_newey_west_min = 1.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 2. Sensibilidad sobre drift inyectado
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def resultado_sintetico():
    return verificar_pead(fuente="sintetico", n_boot=80, seed=20260805)


def test_sintetico_termina_y_estructura(resultado_sintetico):
    r = resultado_sintetico
    assert r.veredicto in {"VIVA", "DEBIL", "MUERTA", "NO_CONCLUYENTE"}
    assert r.rejilla_total is not None and len(r.rejilla_total) == 12
    post = r.rejilla_total[~r.rejilla_total["informativa"]]
    assert len(post) == 4
    assert set(post["exit_offset"]) == set(CRITERIOS.horizontes_post)
    assert (post["entry_offset"] == CRITERIOS.entrada_post).all()
    # La auditoría de honestidad existe y cuenta lo que debe.
    assert "eventos_totales" in r.auditoria
    assert "modo" in r.auditoria and "sintetico" in r.auditoria["modo"]
    # El modo sintético queda etiquetado como no-veredicto en las razones.
    assert any("SINTÉTICO" in x for x in r.razones)


def test_sintetico_con_drift_no_es_muerta(resultado_sintetico):
    """El generador inyecta PEAD real: declararlo MUERTA sería un falso negativo."""
    assert resultado_sintetico.veredicto != "MUERTA"
    post = resultado_sintetico.rejilla_total[~resultado_sintetico.rejilla_total["informativa"]]
    assert (post["q_spread_mean"] > 0).all(), "el spread Q5-Q1 inyectado debe ser positivo"


def test_costes_activados(resultado_sintetico):
    """Los costes no son opcionales: mean_cost > 0 en todos los combos."""
    tabla = resultado_sintetico.rejilla_total
    assert (tabla["mean_cost"] > 0).all()
    assert (tabla["mean_net"] < tabla["mean_gross"]).all()


# ---------------------------------------------------------------------------
# 3. Falsación con señal barajada
# ---------------------------------------------------------------------------


def test_sue_barajado_no_es_viva():
    """Con el SUE permutado entre eventos no hay señal: VIVA sería una fuga.

    Este test es la garantía de que el pipeline de verificación no filtra
    información (quintiles, partición, bootstrap): si con ruido puro el
    veredicto saliera VIVA, habría look-ahead en alguna pieza.
    """
    from earnings_alpha.data.synthetic import SyntheticMarket

    m = SyntheticMarket(seed=11, n_tickers=25, start="2019-01-01", end="2022-12-31")
    eventos = m.events().dropna(subset=["sue"]).copy()
    rng = np.random.default_rng(0)
    eventos["sue"] = rng.permutation(eventos["sue"].to_numpy())

    mediana = pd.to_datetime(eventos["event_date"]).quantile(0.5).date()
    criterios = dataclasses.replace(
        CRITERIOS, corte_train_test=mediana, min_eventos_por_tramo=100
    )
    r = verificar_con_datos(
        eventos, m.prices(), criterios=criterios, n_boot=80, seed=1, etiqueta_sintetico=True
    )
    assert r.veredicto != "VIVA", (
        "señal barajada aprobada: hay una fuga de información en el verificador"
    )


# ---------------------------------------------------------------------------
# Unidades estadísticas
# ---------------------------------------------------------------------------


def test_t_newey_west_recupera_media_conocida():
    from earnings_alpha.verify import _t_newey_west

    rng = np.random.default_rng(3)
    # Media claramente positiva con ruido: t alto, p pequeña.
    s = pd.Series(0.02 + 0.01 * rng.standard_normal(120))
    t, p = _t_newey_west(s)
    assert t > 10 and p < 1e-6
    # Ruido puro: |t| pequeño.
    s0 = pd.Series(0.01 * rng.standard_normal(120))
    t0, _ = _t_newey_west(s0)
    assert abs(t0) < 3


def test_p_desde_ci_coherente():
    from earnings_alpha.verify import _p_normal_desde_ci

    # IC estrecho lejos de cero -> p minúscula; IC que cruza cero -> p grande.
    assert _p_normal_desde_ci(0.05, 0.04, 0.06) < 1e-6
    assert _p_normal_desde_ci(0.001, -0.02, 0.022) > 0.5
    # IC degenerado -> p=1 (sin evidencia, nunca evidencia infinita).
    assert _p_normal_desde_ci(0.05, 0.05, 0.05) == 1.0
