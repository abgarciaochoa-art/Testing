"""Dossier portátil de las estrategias de caídas: datos y métricas en un marco común.

Sirve para comparar y combinar estas estrategias con las de otros estudios con las mismas
reglas de juego (ver ``dossier/DOSSIER.md``). Recalcula las 15 estrategias buenas y SPY
con ``estrategias_buenas.construir`` (mismo motor, mismos datos, 2002-01-02 → último dato)
y escribe en ``dossier/datos/``:

- ``diario.csv``: rentabilidad diaria neta de costes (una columna por estrategia);
- ``mensual.csv`` y ``anual.csv``: rentabilidad por mes y por año natural;
- ``metricas.csv``: métricas estándar en las ventanas estándar del marco común;
- ``crisis.csv``: rentabilidad en las caídas de SPY de referencia (pico → mínimo) y en
  los rebotes;
- ``correlaciones_mensuales.csv`` y ``correlaciones_diarias.csv``;
- ``exposicion.csv``: dónde está el dinero y cómo son las operaciones (familia Caídas
  Técnicas);
- ``estres.csv``: niveles 1-3 con costes más altos, reserva más cara, un día de retraso,
  deuda al 7 % y sin el modelo de desempate;
- ``estrategias.json``: ficha de cada estrategia en formato máquina (reglas, parámetros,
  estado y métricas).

Uso: python research/caidas/dossier.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import estrategias_buenas as eb
import fichas as fi
import motor as m
import motor_v3 as v3
import motor_v4 as v4
import reservas as rs
from definitiva import D1_CFG, curva
from ronda5 import componentes_tec
from ronda5b import preparar

OUT = Path(__file__).parent / "dossier" / "datos"
CORTE = pd.Timestamp("2021-10-01")
MITAD = pd.Timestamp("2024-04-01")

# identificador estable de cada estrategia (columna de los CSV)
IDS = {
    "Nivel 3 · caídas a 2×": "CAI_N3",
    "Caídas a 1,5× (ronda 5)": "CAI_15X",
    "Nivel 2 · reserva contracíclica": "CAI_N2",
    "Escalera ronda 4 · 2× (SPY y SHY)": "CAI_ESC_2X",
    "Nivel 1 · reserva de acciones sin deuda": "CAI_N1",
    "Escalera ronda 4 · 1,5× (SPY y SHY)": "CAI_ESC_15X",
    "Nivel 2 en dos tramos (menos riesgo)": "CAI_N2_TRAMOS",
    "Combinada técnica ronda 5 · 1× (solo acciones)": "CAI_COMB_TEC",
    "D1 · SPY siempre": "CAI_D1_SPY",
    "Escalera ronda 4 · 1× (SPY y SHY)": "CAI_ESC_1X",
    "D1-def · SPY en tendencia": "CAI_D1_DEF",
    "D1-def50 · SPY en tendencia, tope 50 %": "CAI_D1_DEF50",
    "Combinada ronda 3 · 1× (SPY y efectivo)": "CAI_COMB_R3",
    "C1 · caída de 5 días con banderas (ronda 2)": "CAI_C1",
    "Solo Caídas Técnicas (resto en efectivo)": "CAI_SOLO",
    "SPY (comprar y mantener)": "SPY",
}

# caídas de SPY de referencia (cierre del pico → cierre del mínimo) y rebotes
CRISIS = [
    ("Fin de la burbuja puntocom", "2002-01-04", "2002-10-09"),
    ("Crisis financiera", "2007-10-09", "2009-03-09"),
    ("Flash crash", "2010-04-23", "2010-07-02"),
    ("Deuda de EE. UU. y del euro", "2011-04-29", "2011-10-03"),
    ("China y petróleo", "2015-05-21", "2016-02-11"),
    ("Cuarto trimestre de 2018", "2018-09-20", "2018-12-24"),
    ("Covid", "2020-02-19", "2020-03-23"),
    ("Inflación y tipos", "2022-01-03", "2022-10-12"),
    ("Corrección de 2023", "2023-07-31", "2023-10-27"),
    ("Aranceles", "2025-02-19", "2025-04-08"),
    ("Rebote: 12 meses desde el mínimo de 2009", "2009-03-09", "2010-03-09"),
    ("Rebote: 12 meses desde el mínimo del Covid", "2020-03-23", "2021-03-23"),
    ("Rebote: 12 meses desde el mínimo de 2022", "2022-10-12", "2023-10-12"),
]

NIVELES = {"CAI_N1": (1.0, "liq100_tend_minvol"), "CAI_N2": (1.0, "liq100_vg_minvol"),
           "CAI_N3": (2.0, "liq100_vg_minvol")}


def ventanas(fin: pd.Timestamp) -> dict:
    return {
        "total": ("2002-01-02", None),
        "fuera_de_muestra": ("2002-01-02", CORTE),
        "2002_2025": ("2002-01-02", "2026-01-01"),
        "busqueda": (CORTE, None),
        "busqueda_mitad_1": (CORTE, MITAD),
        "busqueda_mitad_2": (MITAD, None),
        "ultimos_10_años": (fin - pd.DateOffset(years=10), None),
        "ultimos_5_años": (fin - pd.DateOffset(years=5), None),
        "ultimo_año": (fin - pd.DateOffset(years=1), None),
    }


def _tramo(s: pd.Series, a, b) -> pd.Series:
    x = s[s.index >= pd.Timestamp(a)]
    return x[x.index < pd.Timestamp(b)] if b is not None else x


def _mensual(eq: pd.Series) -> pd.Series:
    v = eq.resample("ME").last()
    return v / np.r_[eq.iloc[0], v.to_numpy()[:-1]] - 1


def estandar(eq: pd.Series, spy: pd.Series) -> dict:
    """Métricas del marco común sobre un tramo de la curva de valor (base = primer cierre)."""
    eq = eq / eq.iloc[0]
    sp = spy.reindex(eq.index).ffill()
    sp = sp / sp.iloc[0]
    r, rs_ = eq.pct_change().dropna(), sp.pct_change().dropna()
    años = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = eq.iloc[-1] ** (1 / años) - 1
    cagr_spy = sp.iloc[-1] ** (1 / años) - 1
    vol = r.std() * np.sqrt(252)
    abajo = np.sqrt((np.minimum(r, 0) ** 2).mean()) * np.sqrt(252)
    pico = eq.cummax()
    dd = eq / pico - 1
    valle = dd.idxmin()
    f_pico = eq.loc[:valle].idxmax()
    rec = eq.loc[valle:]
    rec = rec[rec >= pico.loc[valle]]
    ult_pico = pd.Series(eq.index.where((eq >= pico).to_numpy()), index=eq.index).ffill()
    bajo_agua = (eq.index.to_series() - ult_pico).dt.days
    rm, rm_spy = _mensual(eq), _mensual(sp)
    q5 = r.quantile(0.05)
    return {
        "desde": eq.index[0].date(), "hasta": eq.index[-1].date(), "años": años,
        "cagr": cagr, "cagr_spy": cagr_spy, "exceso_cagr_spy": cagr - cagr_spy,
        "volatilidad": vol, "sharpe_rf0": r.mean() / r.std() * np.sqrt(252),
        "sortino_rf0": r.mean() * 252 / abajo if abajo > 0 else np.nan,
        "peor_caida": dd.min(), "calmar": cagr / abs(dd.min()) if dd.min() < 0 else np.nan,
        "fecha_pico": f_pico.date(), "fecha_valle": valle.date(),
        "fecha_recuperacion": rec.index[0].date() if len(rec) else None,
        "max_dias_bajo_maximo": int(bajo_agua.max()), "ulcer": float(np.sqrt((dd ** 2).mean())),
        "tiempo_caida_mas_10": float((dd < -0.10).mean()), "tiempo_caida_mas_20": float((dd < -0.20).mean()),
        "meses_positivos": float((rm > 0).mean()), "mes_mediano": float(rm.median()),
        "peor_mes": float(rm.min()), "mejor_mes": float(rm.max()),
        "asimetria_mensual": float(rm.skew()), "curtosis_mensual": float(rm.kurt()),
        "var5_diario": float(q5), "cvar5_diario": float(r[r <= q5].mean()),
        "beta_spy": float(np.cov(r, rs_)[0, 1] / rs_.var()), "corr_spy_diaria": float(r.corr(rs_)),
        "corr_spy_mensual": float(rm.corr(rm_spy)),
    }


def nivel(comp: dict, pal: float, reserva: str, coste_bps: float = 10.0, tasa: float = 0.05,
          coste_reserva: float | None = None, retraso: int = 0, prio=None, detalle: bool = False) -> dict:
    """Igual que ``definitiva.curva`` para D1-técnica, con coste de mover la reserva y
    desempate configurables."""
    x, salida, conv = comp["D1"]
    if prio is not None:
        x = x.assign(prio=prio(x))
    y, w = preparar(x, conv, D1_CFG["kmin"], D1_CFG["uso"], D1_CFG["tam"], D1_CFG["frac"])
    if retraso:
        y = y.assign(t=y.t + retraso)
        ok = (y.t < len(m.cargar().fechas)).to_numpy()
        y, w = y[ok], w[ok]
    return v3.simular(y.t.to_numpy(), y.j.to_numpy(), y.prio.to_numpy(np.float64), peso=w, K=5, frac=D1_CFG["frac"],
                      H=20, salida=salida, reserva_=reserva, palanca=pal, palanca_off=1.0,
                      vol_objetivo=0.15 if pal > 1 else None, coste_bps=coste_bps, tasa_deuda=tasa,
                      coste_reserva=coste_reserva, detalle=detalle)


def _cesta(nombre: str) -> tuple[np.ndarray, np.ndarray]:
    """Valor de apertura y cierre de una cesta guardada como reserva siempre invertida."""
    rn, rd = v3.reserva(nombre)[:2]
    c = np.cumprod(rn * rd)
    return np.r_[1.0, c[:-1]] * rn, c


def gestionada_banda(riesgo: tuple, defensa: tuple, regimen: np.ndarray, banda: float, coste: float,
                     exacto: bool = True, objetivo: float = 0.15, maximo: float = 1.5, tasa: float = 0.05) -> tuple:
    """Como ``reservas.gestionada`` (contracíclica), pero la exposición e al activo de
    riesgo solo se ajusta si el objetivo se aleja más de ``banda`` de la actual (o cambia el
    régimen). Con ``exacto``, cada ajuste paga ``coste`` por lado sobre lo negociado en las
    dos cestas (|Δe| en las 100 líquidas + |Δ max(0, 1 - e)| en las de baja volatilidad);
    si no, ``coste`` por unidad de |Δe|, como la original (banda 0, 2 pb, no exacto)."""
    o_r, c_r = riesgo
    T = len(c_r)
    o_raw, c_raw = defensa
    primero = int(np.argmax(~np.isnan(c_raw)))
    c_d = c_raw / c_raw[primero]
    o_d = o_raw / c_raw[primero]
    c_d[:primero] = 1.0
    o_d[: primero + 1] = 1.0
    c_d = pd.Series(c_d).ffill().to_numpy()
    o_d = np.where(np.isnan(o_d), np.r_[1.0, c_d[:-1]], o_d)
    vol = rs.vol_spy()
    obj = np.nan_to_num(np.where(regimen, np.minimum(maximo, objetivo / np.where(vol > 0, vol, np.inf)), 0.0))
    e = obj.copy()
    for t in range(1, T):
        if regimen[t] == regimen[t - 1] and abs(obj[t] - e[t - 1]) <= banda:
            e[t] = e[t - 1]
    tasa_dia = (1 + tasa) ** (1 / 252) - 1
    rn, rd, cambio = np.ones(T), np.ones(T), np.zeros(T)
    for t in range(1, T):
        a, b = e[t - 1], e[t]  # exposición de la noche y la ajustada en la apertura
        r_noche_r = o_r[t] / c_r[t - 1] - 1 if not np.isnan(o_r[t]) else 0.0
        rn[t] = 1 + a * r_noche_r + max(0.0, 1 - a) * (o_d[t] / c_d[t - 1] - 1)
        cambio[t] = coste * (abs(b - a) + (abs(max(0.0, 1 - b) - max(0.0, 1 - a)) if exacto else 0.0))
        r_dia_r = c_r[t] / o_r[t] - 1 if not np.isnan(o_r[t]) else c_r[t] / c_r[t - 1] - 1
        rd[t] = 1 + b * r_dia_r + max(0.0, 1 - b) * (c_d[t] / o_d[t] - 1) - max(0.0, b - 1) * tasa_dia
    return rs._expo_total((np.nan_to_num(rn, nan=1.0), np.nan_to_num(rd, nan=1.0), cambio, e > 0, e))


def reservas_caras() -> dict:
    """Variantes de las reservas con los costes reales de una cesta de acciones (estrés).

    En la contracíclica del estudio, cada ajuste diario de exposición entre las 100
    líquidas y las 50 de baja volatilidad cuesta 2 pb por unidad de |Δe|, como si fuera un
    ETF. Con cestas de acciones se paga 10 pb por lado sobre lo negociado en las dos
    (``_real``); una implementación razonable solo ajusta si la exposición se desvía más
    de 0,10 (``_real_banda``; banda fijada de antemano, sin probar otras); ``_x2`` dobla
    el coste real. En la del nivel 1 el cambio de régimen ya cuesta lo real (20 pb: vender
    una cesta y comprar la otra); ``_x2`` lo sube a 40 pb.

    Devuelve lo negociado al año en la contracíclica (en veces el valor de la reserva)."""
    liq, mv = _cesta("liq100_dv"), _cesta("minvol50")
    reg = rs.regimen_spy()
    control = gestionada_banda(liq, mv, reg, banda=0.0, coste=2e-4, exacto=False)
    for a, b in zip(control, v3.reserva("liq100_vg_minvol"), strict=True):
        assert np.allclose(a, b, rtol=0, atol=1e-9), "la reserva reconstruida no coincide"
    control1 = rs._siempre_invertida(rs.componer(liq, mv, reg, 2 * rs.COSTE_CESTA))
    for a, b in zip(control1, v3.reserva("liq100_tend_minvol"), strict=True):
        assert np.allclose(a, b, rtol=0, atol=1e-9), "la reserva del nivel 1 reconstruida no coincide"
    v3._RES["liq100_vg_minvol_real"] = real = gestionada_banda(liq, mv, reg, banda=0.0, coste=10e-4)
    v3._RES["liq100_vg_minvol_real_banda"] = banda = gestionada_banda(liq, mv, reg, banda=0.10, coste=10e-4)
    v3._RES["liq100_vg_minvol_x2"] = gestionada_banda(liq, mv, reg, banda=0.0, coste=20e-4)
    v3._RES["liq100_tend_minvol_x2"] = rs._siempre_invertida(rs.componer(liq, mv, reg, 4 * rs.COSTE_CESTA))
    t0 = int(np.searchsorted(m.cargar().fechas, np.datetime64("2002-01-02")))
    años = (len(reg) - t0) / 252
    return {nombre: float(r[2][t0:].sum() / 10e-4 / años) for nombre, r in (("sin banda", real), ("banda 0,10", banda))}


def desempate_ema5(x: pd.DataFrame) -> pd.Series:
    """Sin el modelo: a igualdad de módulos, antes la más estirada bajo su EMA5."""
    return x.score_v2 + 0.99 * (-x.dist_ema5).groupby(x.t).rank(pct=True).fillna(0.5)


def estres(comp: dict) -> tuple[pd.DataFrame, dict]:
    rotacion = reservas_caras()
    filas = []
    for clave, (pal, res) in NIVELES.items():
        contra = res == "liq100_vg_minvol"
        res7 = "liq100_vg_minvol_7" if contra else res
        escenarios = {"base": {}}
        if contra:
            escenarios |= {
                "reserva con su coste real de cesta (10 pb por lado sobre lo negociado)": dict(
                    reserva="liq100_vg_minvol_real"),
                "ídem, ajustando solo si la exposición se desvía más de 0,10": dict(
                    reserva="liq100_vg_minvol_real_banda"),
            }
        if pal > 1:
            escenarios["con la reserva del nivel 1 (deuda solo en las compras)"] = dict(reserva="liq100_tend_minvol")
        escenarios |= {
            "compras y ventas a 20 pb por lado (×2)": dict(coste_bps=20.0),
            "compras y ventas a 30 pb por lado (×3)": dict(coste_bps=30.0),
            "mover dinero entre reserva y compras a 20 pb (×2)": dict(coste_reserva=20e-4),
            "gestión de la reserva a ×2 de su coste real": dict(reserva=res + "_x2"),
            "todo a ×2 a la vez (compras, mover reserva y gestión de la reserva)": dict(
                coste_bps=20.0, coste_reserva=20e-4, reserva=res + "_x2"),
            "entrar 1 día tarde": dict(retraso=1),
            "deuda al 7 %": dict(tasa=0.07, reserva=res7),
            "sin el modelo: desempate por distancia a la EMA5": dict(prio=desempate_ema5),
            "sin el modelo: desempate neutro": dict(prio=lambda x: x.score_v2 + 0.495),
        }
        base = None
        for nombre, kw in escenarios.items():
            kw = dict(kw)
            eq = nivel(comp, pal, kw.pop("reserva", res), **kw)["equity"]
            mt = m.metricas(eq)
            val = eq[eq.index < CORTE]
            mv = m.metricas(val / val.iloc[0])
            fila = {"estrategia": clave, "escenario": nombre, "cagr_total": mt["cagr"],
                    "cagr_fuera_de_muestra": mv["cagr"], "peor_caida": mt["maxdd"], "sharpe_rf0": mt["sharpe"]}
            if base is None:
                base = fila
                ref = curva(comp, "D1", D1_CFG, pal, reserva=res)["equity"]
                assert np.allclose(eq.to_numpy(), ref.to_numpy(), rtol=1e-12), "la base no reproduce definitiva.curva"
            fila["dif_cagr_total"] = fila["cagr_total"] - base["cagr_total"]
            fila["dif_cagr_fuera_de_muestra"] = fila["cagr_fuera_de_muestra"] - base["cagr_fuera_de_muestra"]
            filas.append(fila)
    # modo prudente (dos tramos): con la reserva a su coste real y con la del nivel 1
    x, salida, conv = comp["D1"]
    y, w = preparar(x, conv, D1_CFG["kmin"], D1_CFG["uso"], D1_CFG["tam"], D1_CFG["frac"])
    base = None
    for nombre, res in (("base", "liq100_vg_minvol"),
                        ("reserva con su coste real de cesta (10 pb por lado sobre lo negociado)", "liq100_vg_minvol_real"),
                        ("ídem, ajustando solo si la exposición se desvía más de 0,10", "liq100_vg_minvol_real_banda"),
                        ("con la reserva del nivel 1 (sin deuda)", "liq100_tend_minvol")):
        eq = v4.simular(y.t.to_numpy(), y.j.to_numpy(), y.prio.to_numpy(np.float64), peso=w, K=5, frac=0.5, H=20,
                        salida=salida, reserva_=res, palanca=1.0, palanca_off=1.0, w_ini=0.5, add_modo=1,
                        add_umbral=0.03)["equity"]
        mt = m.metricas(eq)
        val = eq[eq.index < CORTE]
        fila = {"estrategia": "CAI_N2_TRAMOS", "escenario": nombre, "cagr_total": mt["cagr"],
                "cagr_fuera_de_muestra": m.metricas(val / val.iloc[0])["cagr"], "peor_caida": mt["maxdd"],
                "sharpe_rf0": mt["sharpe"]}
        base = base or fila
        fila["dif_cagr_total"] = fila["cagr_total"] - base["cagr_total"]
        fila["dif_cagr_fuera_de_muestra"] = fila["cagr_fuera_de_muestra"] - base["cagr_fuera_de_muestra"]
        filas.append(fila)
    return pd.DataFrame(filas), rotacion


def exposicion(comp: dict) -> pd.DataFrame:
    """Dónde está el dinero y cómo son las operaciones (familia Caídas Técnicas)."""
    cfg = fi.configuraciones()
    x, salida, conv = comp["D1"]
    y, w = preparar(x, conv, D1_CFG["kmin"], D1_CFG["uso"], D1_CFG["tam"], D1_CFG["frac"])
    runs = {
        "CAI_SOLO": fi.simular(cfg["pura"], "2002-01-02"),
        "CAI_N1": nivel(comp, 1.0, "liq100_tend_minvol", detalle=True),
        "CAI_N2": nivel(comp, 1.0, "liq100_vg_minvol", detalle=True),
        "CAI_N2_TRAMOS": v4.simular(y.t.to_numpy(), y.j.to_numpy(), y.prio.to_numpy(np.float64), peso=w, K=5,
                                    frac=0.5, H=20, salida=salida, reserva_="liq100_vg_minvol", palanca=1.0,
                                    palanca_off=1.0, w_ini=0.5, add_modo=1, add_umbral=0.03, detalle=True),
        "CAI_15X": nivel(comp, 1.5, "liq100_tend_minvol", detalle=True),
        "CAI_N3": nivel(comp, 2.0, "liq100_vg_minvol", detalle=True),
    }
    filas = []
    for clave, r in runs.items():
        ops = r["operaciones"]
        c = ops[ops.t_sal >= 0]
        # motor v4: retorno neto de la operación con sus tramos; v3: neto de 10 pb por lado
        neto = c.ret if "tramos" in c.columns else c.px_sal / c.px_ent * (1 - 1e-3) ** 2 - 1
        años = (r["equity"].index[-1] - r["equity"].index[0]).days / 365.25
        lev, npos = r["serie_lev"], r["serie_posiciones"]
        filas.append({
            "estrategia": clave, "capital_en_caidas_medio": float(r["serie_expo"].mean()),
            "capital_en_caidas_p95": float(r["serie_expo"].quantile(0.95)),
            "dias_con_alguna_caida": float((npos > 0).mean()), "posiciones_medias": float(npos.mean()),
            "exposicion_bruta_media": float(lev.mean()), "exposicion_bruta_p95": float(lev.quantile(0.95)),
            "exposicion_bruta_max": float(lev.max()), "operaciones": len(c), "operaciones_año": len(c) / años,
            "acierto": float((neto > 0).mean()), "ret_medio_neto": float(neto.mean()),
            "ret_mediano_neto": float(neto.median()), "sesiones_medias": float((c.t_sal - c.t_ent).mean()),
            "peso_medio_entrada": float(c.peso.mean()),
        })
    return pd.DataFrame(filas)


# ----------------------------------------------------------------- fichas en formato máquina
CONVENCIONES = {
    "datos": "release pública Johnbrick123/sp500-data (Yahoo + CRSP + Tiingo + Wiki): ~1.040 tickers con "
             "pertenencia point-in-time al S&P 500 (intervalos [alta, baja)), quiebras incluidas; 2002-01-02 → "
             "2026-10-12 (6.234 sesiones)",
    "precios": "cierre ajustado por splits y dividendos (rentabilidad total) en ~83 % de las series; el resto, sin "
               "ajuste de dividendos (conservador); filtro de 5 $ con el precio REAL",
    "universo": "miembros del S&P 500 en esa fecha y líquidos: precio real >= 5 $ y volumen medio en dólares de 20 "
                "sesiones >= 5 M$ (si la fuente no da volumen, no se excluye)",
    "senal": "al cierre de t, solo con datos <= t",
    "ejecucion": "compras y ventas a la apertura de t+1; deslistadas, al último cierre",
    "costes": "10 pb por lado en cada compra y venta de acciones; 10 pb por mover dinero entre la reserva de "
              "acciones y las compras (2 pb con ETF); 10 pb sobre lo negociado en el rebalanceo mensual de las "
              "cestas; 20 pb por cambio de régimen de la reserva del nivel 1; 2 pb por unidad en los ajustes "
              "diarios de la reserva contracíclica (optimista: ver estres.csv)",
    "deuda": "5 % anual, día a día; el tope de palanca cuenta la exposición real (incluida la palanca interna de la "
             "reserva); si la exposición supera un 20 % el tope, se vende reserva para devolver deuda",
    "regimen": "tendencia = SPY > SMA200 al cierre de t-1; en las versiones solo acciones SPY es solo una señal",
    "rentabilidad": "neta de costes, sobre el valor de la cartera al cierre; sin impuestos",
    "ventanas": "búsqueda (donde se eligió todo) = 2021-10-01 → fin, en dos mitades con corte 2024-04-01; "
                "fuera de muestra (validación, nunca usada para elegir) = 2002-01-02 → 2021-09-30",
}

BLOQUES = {
    "rasgos": {
        "dd20": "C / máximo de los cierres de 20 sesiones - 1",
        "r5, r10": "retorno de 5 / 10 sesiones",
        "sobre_sma200": "C > SMA200",
        "beta": "beta a SPY con 252 sesiones, recalculada a fin de mes",
        "idio10": "r10 - beta · retorno de SPY en 10 sesiones",
        "pares": "las 10 acciones más correlacionadas (252 sesiones, miembros de ese momento, recalculado a fin de "
                 "mes y usado el mes siguiente); la nº 1 es la gemela",
        "gem10, gem_corr": "retorno de 10 sesiones de la gemela y su correlación",
        "gem_z": "z (60 sesiones) del log(precio / precio de la gemela)",
        "noche_min10": "peor hueco de apertura (O_t / C_t-1 - 1) de las 10 últimas sesiones",
        "volmax10": "máximo de 10 sesiones de (volumen / media de 20 sesiones previas)",
        "ibs": "(C - L) / (H - L) del día de la señal",
        "mom252": "retorno de 252 sesiones",
        "dist_sma200": "C / SMA200 - 1",
        "pares_frac_caen": "fracción de sus 10 pares con r10 <= -5 %",
        "tam": "percentil del volumen medio en dólares de 60 sesiones entre los miembros",
        "spy_dd252": "SPY / máximo de 252 sesiones de SPY - 1",
        "spy10": "retorno de SPY en 10 sesiones",
        "pares10": "retorno medio de los 10 pares en 10 sesiones",
        "rel_pares10": "r10 - pares10",
        "volreal_spy": "volatilidad realizada anualizada de SPY en 20 sesiones",
        "frac_caen": "fracción de miembros con r5 <= -5 %",
        "dist_ema5, dist_ema10": "C / EMA(5 | 10) - 1 (EMA con span n)",
        "dist_wma20": "C / media ponderada lineal de 20 sesiones - 1",
        "dist_vwap10": "C / VWAP de 10 sesiones - 1 (precio típico (H+L+C)/3 × volumen)",
        "bb_z20": "(C - SMA20) / desviación típica de 20 sesiones",
        "estoc14": "(C - mínimo de 14) / (máximo de 14 - mínimo de 14)",
        "dist_sma50": "C / SMA50 - 1",
        "vol_climax": "volumen de hoy / máximo del volumen de las 60 sesiones previas",
        "obv_pend20": "(OBV - OBV de hace 20) / (20 × volumen medio de 20)",
        "meta_pct": "percentil, entre las candidatas del día, de un gradient boosting sobre 41 rasgos en rango "
                    "entrenado SOLO con la búsqueda (solo desempata; ver estres.csv para quitarlo)",
    },
    "modulos_score_v2": [
        "componente propia de la caída entre el 50 % y el 80 %: 0,5 <= idio10 / r10 < 0,8",
        "gemela también cae: gem_corr >= 0,6 y gem10 <= -5 %",
        "sin divergencia con la gemela: gem_corr >= 0,6 y gem_z > -2",
        "hueco sin noticia: noche_min10 <= -5 % y volmax10 < 3",
        "cierra en mínimos: ibs <= 0,2",
        "momentum 12 meses fuerte: mom252 >= 30 %",
        "muy extendida sobre su SMA200: dist_sma200 >= 15 %",
        "cae con su grupo: pares_frac_caen >= 60 %",
        "grande del índice: tam >= 2/3",
        "SPY en corrección: spy_dd252 <= -10 %",
        "SPY algo caído: -10 % < spy_dd252 <= -3 %",
    ],
    "banderas_tecnicas": [
        "dist_ema5 <= -5,319 %", "dist_ema5 < -3,449 %", "dist_ema10 < -5,533 %", "dist_wma20 < -7,219 %",
        "dist_vwap10 < -6,401 %", "bb_z20 < -1,663", "estoc14 < 0,1087", "dist_sma50 <= -12,58 %",
        "dist_sma50 < -7,131 %", "vol_climax >= 0,609", "obv_pend20 <= -0,3111",
    ],
    "banderas_c1": [
        "spy10 <= -5 %", "pares10 <= -7 %", "rel_pares10 > -6 %", "0,5 <= idio10 / r10 < 0,8", "idio10 / r10 < 0,5",
        "gem_corr >= 0,6 y gem10 <= -5 %", "gem_corr >= 0,6 y gem_z > -2", "noche_min10 <= -5 % y volmax10 < 3",
        "ibs <= 0,2", "volreal_spy >= 25 %", "frac_caen >= 20 %", "mom252 >= 30 %", "beta >= 1,3",
    ],
    "entradas": {
        "D1_tecnica": "miembro líquido, C > SMA200, dd20 <= -15 %, score_v2 >= 5 (de 11 módulos) y >= 2 de las 11 "
                      "banderas técnicas; prioridad = score_v2 + 0,99 · meta_pct; peso = clip((score_v2 - 4) / 2, "
                      "0,5, 2), con tope 1 → posición = 50 % × peso del capital (25 % con 5 módulos, 50 % con >= 6)",
        "D1_r3": "miembro líquido, C > SMA200, dd20 <= -15 %, score_v2 >= 5; prioridad = score_v2 + 0,99 · meta_pct; "
                 "peso = clip((score_v2 - 4) / 2, 0,5, 2) sobre un hueco de 1/3 (K=3); def50: peso <= 1,5",
        "C1": "miembro líquido, C > SMA200, r5 <= -7 %, >= 6 de las 13 banderas_c1; prioridad = nº de banderas + "
              "clip(-r10, 0, 0,99); hueco 1/K, peso 1",
        "C1_tecnica": "C1 con >= 2 banderas técnicas, que suman también a la prioridad",
    },
    "salidas": {
        "sma10": "vender a la apertura siguiente al primer cierre > SMA10 (se mira desde el cierre del día de "
                 "compra) o tras 20 sesiones en cartera",
        "rsi2": "ídem con RSI(2) de Wilder >= 70",
    },
    "reservas": {
        "liq100_dv": "100 miembros más líquidos (volumen en dólares de 60 sesiones), ponderados por ese volumen; "
                     "se elige al cierre de fin de mes y se rebalancea a la apertura siguiente",
        "minvol50": "50 miembros líquidos con menor volatilidad de 252 sesiones, equiponderados, mensual",
        "liq100_tend_minvol": "liq100_dv si SPY > SMA200, minvol50 si no (siempre en acciones)",
        "liq100_vg_minvol": "contracíclica: en tendencia, exposición e = min(1,5; 15 % / vol20 de SPY) a liq100_dv "
                            "(si e < 1, el resto en minvol50; si e > 1, deuda al 5 %); fuera de tendencia, todo en "
                            "minvol50; e se decide al cierre de t-1 y se ajusta a la apertura de t",
        "spy, spy_tend": "SPY siempre / SPY en tendencia y efectivo al 0 % fuera (usan ETF y efectivo)",
        "spy_tend_shy": "SPY en tendencia, SHY fuera (usa ETF)",
        "spy_vg_shy": "SPY gestionado por volatilidad en tendencia (15 %, máx. 1,5×), SHY fuera (usa ETF)",
        "efectivo": "efectivo al 0 %",
    },
}

FICHAS = {
    "CAI_N2": dict(estado="central; con los costes reales de su reserva empata con el nivel 1 fuera de muestra",
                   veredicto="sí", ronda=6, solo_acciones=True, entrada="D1_tecnica",
                   K=5, hueco=0.5, salida="sma10", reserva="liq100_vg_minvol", palanca_caidas="1× (no se pide deuda para las compras)",
                   deuda="solo dentro de la reserva, en calma y en tendencia (máx. 1,5× en total)",
                   codigo="definitiva.curva(componentes_tec(), 'D1', D1_CFG, 1.0, reserva='liq100_vg_minvol')"),
    "CAI_N1": dict(estado="recomendada por sencillez: sin deuda y, con costes reales, igual que el nivel 2 fuera de "
                          "muestra", veredicto="sí", ronda=6, solo_acciones=True, entrada="D1_tecnica",
                   K=5, hueco=0.5, salida="sma10", reserva="liq100_tend_minvol", palanca_caidas="1×", deuda="ninguna",
                   codigo="definitiva.curva(componentes_tec(), 'D1', D1_CFG, 1.0, reserva='liq100_tend_minvol')"),
    "CAI_N3": dict(estado="alternativa agresiva", veredicto="quizás", ronda=6, solo_acciones=True,
                   entrada="D1_tecnica", K=5, hueco=0.5, salida="sma10", reserva="liq100_vg_minvol",
                   palanca_caidas="hasta 2× en tendencia, el exceso sobre 1 escalado por min(1, 15 % / vol20 de SPY); "
                                  "1× fuera de tendencia",
                   deuda="reserva y compras (exposición real máx. 2,38×)",
                   codigo="definitiva.curva(componentes_tec(), 'D1', D1_CFG, 2.0, reserva='liq100_vg_minvol')"),
    "CAI_N2_TRAMOS": dict(estado="modo prudente", veredicto="quizás", ronda=7, solo_acciones=True,
                          entrada="D1_tecnica (primera mitad) + segunda mitad a la apertura siguiente si el cierre "
                                  "del día de compra queda >= 3 % por debajo del precio de compra",
                          K=5, hueco=0.5, salida="sma10", reserva="liq100_vg_minvol", palanca_caidas="1×",
                          deuda="como el nivel 2",
                          codigo="motor_v4.simular(..., reserva_='liq100_vg_minvol', w_ini=0.5, add_modo=1, "
                                 "add_umbral=0.03)"),
    "CAI_15X": dict(estado="vigente: la palanca más sencilla (deuda solo en las compras, reserva del nivel 1, costes "
                           "reales); entre los niveles 1 y 3", veredicto="quizás", ronda=5, solo_acciones=True,
                    entrada="D1_tecnica", K=5, hueco=0.5, salida="sma10", reserva="liq100_tend_minvol",
                    palanca_caidas="hasta 1,5× en tendencia, escalado por volatilidad; 1× fuera",
                    deuda="solo en las compras",
                    codigo="definitiva.curva(componentes_tec(), 'D1', D1_CFG, 1.5, reserva='liq100_tend_minvol')"),
    "CAI_SOLO": dict(estado="motor de señales (sin reserva); base para combinar con otras estrategias",
                     veredicto="no como cartera (de media, 77 % del capital en efectivo)", ronda=5, solo_acciones=False,
                     entrada="D1_tecnica", K=5, hueco=0.5, salida="sma10", reserva="efectivo", palanca_caidas="1×",
                     deuda="ninguna", codigo="fichas.simular(fichas.configuraciones()['pura'], '2002-01-02')"),
    "CAI_COMB_TEC": dict(estado="descartada: C1 diluye a D1", veredicto="no", ronda=5, solo_acciones=True,
                         entrada="50 % D1_tecnica (K=5, hueco 1/2) + 50 % C1_tecnica (K=5, hueco 1/3), rebalanceo "
                                 "mensual", K=5, hueco=None, salida="sma10 (D1) / rsi2 (C1)",
                         reserva="liq100_tend_minvol", palanca_caidas="1×", deuda="ninguna",
                         codigo="definitiva.cartera(componentes_tec(), 'D1-técnica + C1-técnica', 1.0, "
                                "reserva='liq100_tend_minvol')"),
    "CAI_ESC_1X": dict(estado="histórica (usa ETF)", veredicto="no", ronda=4, solo_acciones=False,
                       entrada="50 % D1_r3 + 50 % C1, cada una K=5 y hueco 1/3, rebalanceo mensual", K=5,
                       hueco=1 / 3, salida="sma10 (D1) / rsi2 (C1)", reserva="spy_tend_shy", palanca_caidas="1×",
                       deuda="ninguna", codigo="ronda4_final.cartera(NIVELES['Nivel 1 · máx. 1x (sin deuda)'])"),
    "CAI_ESC_15X": dict(estado="histórica (usa ETF)", veredicto="no", ronda=4, solo_acciones=False,
                        entrada="como CAI_ESC_1X", K=5, hueco=1 / 3, salida="sma10 / rsi2", reserva="spy_vg_shy",
                        palanca_caidas="hasta 1,5× gestionada por volatilidad (15 %)", deuda="compras y reserva",
                        codigo="ronda4_final.cartera(NIVELES['Nivel 2 · máx. 1,5x gestionada'])"),
    "CAI_ESC_2X": dict(estado="histórica (usa ETF)", veredicto="no", ronda=4, solo_acciones=False,
                       entrada="como CAI_ESC_1X", K=5, hueco=1 / 3, salida="sma10 / rsi2", reserva="spy_vg_shy",
                       palanca_caidas="hasta 2× gestionada por volatilidad (15 %)", deuda="compras y reserva",
                       codigo="ronda4_final.cartera(NIVELES['Nivel 3 · máx. 2x gestionada'])"),
    "CAI_COMB_R3": dict(estado="histórica (usa SPY y efectivo)", veredicto="no", ronda=3, solo_acciones=False,
                        entrada="50 % D1_r3 def50 (K=3) + 50 % C1 (K=5), rebalanceo mensual", K=None, hueco=None,
                        salida="sma10 / rsi2", reserva="spy_tend", palanca_caidas="1×", deuda="ninguna",
                        codigo="combinar.mezclar({'D1': D1-def50, 'C1': C1}, {'D1': 0.5, 'C1': 0.5})"),
    "CAI_D1_SPY": dict(estado="histórica (usa SPY)", veredicto="no", ronda=3, solo_acciones=False, entrada="D1_r3",
                       K=3, hueco=1 / 3, salida="sma10", reserva="spy", palanca_caidas="1×", deuda="ninguna",
                       codigo="motor_v3.simular(d1, K=3, salida='sma10', reserva_='spy')"),
    "CAI_D1_DEF": dict(estado="histórica (usa SPY y efectivo)", veredicto="no", ronda=3, solo_acciones=False,
                       entrada="D1_r3", K=3, hueco=1 / 3, salida="sma10", reserva="spy_tend", palanca_caidas="1×",
                       deuda="ninguna", codigo="motor_v3.simular(d1, K=3, salida='sma10', reserva_='spy_tend')"),
    "CAI_D1_DEF50": dict(estado="histórica (usa SPY y efectivo)", veredicto="no", ronda=3, solo_acciones=False,
                         entrada="D1_r3 con peso <= 1,5 (ninguna posición > 50 %)", K=3, hueco=1 / 3, salida="sma10",
                         reserva="spy_tend", palanca_caidas="1×", deuda="ninguna",
                         codigo="motor_v3.simular(d1, peso=min(peso, 1.5), K=3, salida='sma10', reserva_='spy_tend')"),
    "CAI_C1": dict(estado="histórica (usa SPY y efectivo)", veredicto="no", ronda=2, solo_acciones=False,
                   entrada="C1", K=5, hueco=0.2, salida="rsi2", reserva="spy_tend", palanca_caidas="1×",
                   deuda="ninguna", codigo="motor_v3.simular(c1, K=5, salida='rsi2', reserva_='spy_tend')"),
    "SPY": dict(estado="referencia", veredicto="—", ronda=None, solo_acciones=False, entrada="comprar y mantener",
                K=None, hueco=None, salida=None, reserva=None, palanca_caidas="1×", deuda="ninguna",
                codigo="motor.benchmark('SPY')"),
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    E = eb.construir()
    eqs = {IDS[x["nombre"]]: x["eq"] for x in E}
    assert len(eqs) == len(IDS), set(IDS) - {x["nombre"] for x in E}
    df = pd.DataFrame(eqs)
    assert df.notna().all().all(), "curvas con huecos"
    spy = df["SPY"]
    fin = df.index[-1]

    diario = df.pct_change().fillna(0.0)
    diario.index.name = "fecha"
    diario.to_csv(OUT / "diario.csv", float_format="%.7f")
    mensual = pd.DataFrame({k: _mensual(v / v.iloc[0]) for k, v in df.items()})
    mensual.index = mensual.index.to_period("M").astype(str)
    mensual.index.name = "mes"
    mensual.to_csv(OUT / "mensual.csv", float_format="%.5f")
    anual = df.resample("YE").last()
    anual = anual / pd.concat([df.iloc[[0]], anual.iloc[:-1]]).to_numpy() - 1
    anual.index = anual.index.year
    anual.index.name = "año"
    anual.to_csv(OUT / "anual.csv", float_format="%.5f")

    filas = []
    for clave, eq in df.items():
        for vn, (a, b) in ventanas(fin).items():
            x = _tramo(eq, a, b)
            filas.append({"estrategia": clave, "ventana": vn, **estandar(x, _tramo(spy, a, b))})
    metricas = pd.DataFrame(filas)
    metricas.to_csv(OUT / "metricas.csv", index=False, float_format="%.5f")

    crisis = pd.DataFrame({clave: [eq.asof(pd.Timestamp(b)) / eq.asof(pd.Timestamp(a)) - 1 for _, a, b in CRISIS]
                           for clave, eq in df.items()})
    crisis.insert(0, "hasta", [b for _, _, b in CRISIS])
    crisis.insert(0, "desde", [a for _, a, _ in CRISIS])
    crisis.insert(0, "tramo", [n for n, _, _ in CRISIS])
    crisis.to_csv(OUT / "crisis.csv", index=False, float_format="%.5f")
    mensual.corr().to_csv(OUT / "correlaciones_mensuales.csv", float_format="%.3f")
    diario.iloc[1:].corr().to_csv(OUT / "correlaciones_diarias.csv", float_format="%.3f")

    comp = componentes_tec()
    expo = exposicion(comp)
    expo.to_csv(OUT / "exposicion.csv", index=False, float_format="%.5f")
    est, rotacion = estres(comp)
    est.to_csv(OUT / "estres.csv", index=False, float_format="%.5f")
    print("rotación anual de la exposición de la reserva contracíclica (suma de |Δe|):", rotacion)

    dsr = pd.read_csv(Path(__file__).parent / "resultados" / "definitiva.csv", index_col=0)
    columnas_dsr = {"CAI_N1": "Nivel 1 · sin deuda (máx. 1x)",
                    "CAI_N2": "Nivel 2 · deuda solo en la reserva en calma (máx. 1,5x)",
                    "CAI_N3": "Nivel 3 · reserva contracíclica + caídas (máx. 2x)",
                    "CAI_15X": "ref · ronda 5, caídas hasta 1,5x", "CAI_COMB_TEC": "ref · D1 + C1 técnicas, sin deuda"}
    nombres = {v: k for k, v in IDS.items()}
    familia = {IDS[x["nombre"]]: x["familia"] for x in E}
    salida = []
    for clave, ficha in FICHAS.items():
        mt = metricas[metricas.estrategia == clave].set_index("ventana")
        resumen = {vn: {k: (float(mt.loc[vn, k]) if isinstance(mt.loc[vn, k], (int, float, np.floating)) else str(mt.loc[vn, k]))
                        for k in ("cagr", "peor_caida", "sharpe_rf0", "volatilidad", "calmar")}
                   for vn in ("total", "fuera_de_muestra", "2002_2025", "busqueda", "ultimos_5_años", "ultimo_año")}
        extra = {}
        if clave in columnas_dsr:
            extra = {"dsr_n_bruto": float(dsr.loc["DSR (N bruto)", columnas_dsr[clave]]),
                     "dsr_n_efectivo": float(dsr.loc["DSR (N efectivo)", columnas_dsr[clave]])}
        salida.append({"id": clave, "nombre": nombres[clave], "familia": familia[clave], **ficha,
                       "metricas": resumen, **extra})
    (OUT / "estrategias.json").write_text(json.dumps(
        {"convenciones": CONVENCIONES, "bloques": BLOQUES, "estrategias": salida,
         "pruebas_totales": {"carteras_simuladas": 11000, "n_bruto_dsr": 10470, "n_efectivo_dsr": 205,
                             "nota": "todas las variantes probadas en las ocho rondas cuentan en la corrección por "
                                     "minería de datos"}},
        ensure_ascii=False, indent=1, default=str))

    pd.set_option("display.width", 250)
    tot = metricas[metricas.ventana.isin(["total", "fuera_de_muestra"])].pivot_table(
        index="estrategia", columns="ventana", values=["cagr", "peor_caida", "sharpe_rf0", "sortino_rf0", "calmar"])
    print(tot.round(3).sort_values(("cagr", "total"), ascending=False).to_string())
    print(crisis.set_index("tramo").drop(columns=["desde", "hasta"])[["CAI_N2", "CAI_N1", "CAI_N3", "CAI_SOLO",
                                                                       "SPY"]].round(3).to_string())
    print(expo.round(3).to_string())
    print(est.round(4).to_string())


if __name__ == "__main__":
    main()
