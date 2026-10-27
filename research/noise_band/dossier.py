"""Dossier portátil de Noise Band: datos y métricas en el marco común del dossier de caídas.

Para comparar y combinar Noise Band con otras estrategias con las mismas reglas (ver
``dossier/DOSSIER.md``). Recalcula las versiones con ``elegida.versiones`` (mismo motor,
mismos datos, siempre fuera de muestra) y escribe en ``dossier/datos/``:

- ``diario.csv``: rentabilidad diaria neta de costes de cada versión (prefijo ``NB_``), del
  nivel 1 de caídas con Noise Band encima (``CAI_N1_NB_*``) y de ``CAI_N1`` y ``SPY``, en las
  sesiones de NYSE del 21-ene-2005 al 13-may-2020 (el primer día vale 0);
- ``mensual.csv`` y ``anual.csv``;
- ``metricas.csv``: las métricas estándar del dossier de caídas (``estandar``, sección 2.3) en
  las ventanas de Noise Band;
- ``crisis.csv``: los tramos de crisis del dossier de caídas que caen dentro de estos datos;
- ``correlaciones_mensuales.csv``;
- ``exposicion_diaria.csv``: fracción de la sesión con posición y palanca de cada versión;
- ``operaciones.csv``: cada operación de las versiones principales, para comprobar una réplica;
- ``entrenamiento.csv``: las operaciones del artículo con sus 25 rasgos y su retorno, con las
  que se rehace el modelo de la puntuación;
- ``puntuacion_coeficientes.csv`` y ``puntuacion_percentiles.csv``: el modelo para operar;
- ``ejemplo_puntuacion.csv``: una ruptura real calculada paso a paso;
- ``estrategias.json``: ficha de cada versión en formato máquina.

Uso: python research/noise_band/dossier.py (después de ``ronda2.py``, ``modelos.py`` y
``elegida.py``; usa ``research/caidas/dossier/datos/diario.csv``)
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import elegida as el
import eventos as ev
import modelos as mo
import motor as nb
import rasgos as rg

OUT = Path(__file__).parent / "dossier" / "datos"
GRAFICAS = OUT.parent / "graficas"
RESULTADOS = Path(__file__).parent / "resultados"
CAIDAS = Path(__file__).resolve().parents[1] / "caidas" / "dossier" / "datos" / "diario.csv"
INICIO = "2005-01-21"  # última sesión antes de la primera con cono (hacen falta 14 sesiones)
VOL = "vol 2 % (máx. 4×)"
IDS = {
    ("artículo", "1x"): "NB_ART", ("artículo", VOL): "NB_ART_VOL", ("artículo (solo largo)", "1x"): "NB_ART_L",
    ("fuerza ≥ 0,3", "1x"): "NB_F03", ("fuerza ≥ 0,3", VOL): "NB_F03_VOL",
    ("fuerza ≥ 0,3 (solo largo)", "1x"): "NB_F03_L",
    ("puntuación, dos mitades", "1x"): "NB_PUNT", ("puntuación, dos mitades", VOL): "NB_PUNT_VOL",
    ("puntuación, dos mitades (solo largo)", "1x"): "NB_PUNT_L",
    ("puntuación, año a año", "1x"): "NB_PUNT_AA", ("puntuación, año a año", VOL): "NB_PUNT_AA_VOL",
}
COMBOS = {"CAI_N1_NB_ART": ("NB_ART", 1.0), "CAI_N1_NB_F03": ("NB_F03", 1.0), "CAI_N1_NB_PUNT": ("NB_PUNT", 1.0),
          "CAI_N1_NB_PUNT_05": ("NB_PUNT", 0.5), "CAI_N1_NB_PUNT_VOL": ("NB_PUNT_VOL", 1.0),
          "CAI_N1_NB_PUNT_L": ("NB_PUNT_L", 1.0), "CAI_N1_NB_PUNT_AA": ("NB_PUNT_AA", 1.0),
          "CAI_N1_NB_PUNT_AA_VOL": ("NB_PUNT_AA_VOL", 1.0)}
# cada ventana empieza en el cierre de la sesión anterior a su primer día (como base), para que
# cuente la rentabilidad de ese primer día (el 24-ene-2005 el nivel 1 hizo un -6,1 %)
VENTANAS = {"total": (INICIO, None), "validacion": (INICIO, "2013-01-01"),
            "busqueda": ("2012-12-31", None), "busqueda_mitad_1": ("2012-12-31", "2017-01-01"),
            "busqueda_mitad_2": ("2016-12-30", None), "desde_2008": ("2007-12-31", None)}
CRISIS = [("Crisis financiera", "2007-10-09", "2009-03-09"), ("Flash crash", "2010-04-23", "2010-07-02"),
          ("Deuda de EE. UU. y del euro", "2011-04-29", "2011-10-03"),
          ("China y petróleo", "2015-05-21", "2016-02-11"), ("Cuarto trimestre de 2018", "2018-09-20", "2018-12-24"),
          ("Covid (hasta el mínimo)", "2020-02-19", "2020-03-23"),
          ("Rebote: 12 meses desde el mínimo de 2009", "2009-03-09", "2010-03-09")]
NOMBRES = {
    "NB_ART": "Noise Band del artículo, largo y corto, 1×",
    "NB_ART_VOL": "Noise Band del artículo, largo y corto, volatilidad 2 % (máx. 4×)",
    "NB_ART_L": "Noise Band del artículo, solo largo, 1×",
    "NB_F03": "Noise Band con fuerza ≥ 0,3 sigmas, largo y corto, 1×",
    "NB_F03_VOL": "Noise Band con fuerza ≥ 0,3 sigmas, largo y corto, volatilidad 2 % (máx. 4×)",
    "NB_F03_L": "Noise Band con fuerza ≥ 0,3 sigmas, solo largo, 1×",
    "NB_PUNT": "Noise Band con puntuación (dos mitades), largo y corto, 1×",
    "NB_PUNT_VOL": "Noise Band con puntuación (dos mitades), largo y corto, volatilidad 2 % (máx. 4×)",
    "NB_PUNT_L": "Noise Band con puntuación (dos mitades), solo largo, 1×",
    "NB_PUNT_AA": "Noise Band con puntuación reentrenada año a año (desde 2008), largo y corto, 1×",
    "NB_PUNT_AA_VOL": "Noise Band con puntuación reentrenada año a año (desde 2008), volatilidad 2 % (máx. 4×)",
    "CAI_N1_NB_ART": "Nivel 1 de caídas + NB_ART encima (peso 1)",
    "CAI_N1_NB_F03": "Nivel 1 de caídas + NB_F03 encima (peso 1)",
    "CAI_N1_NB_PUNT": "Nivel 1 de caídas + NB_PUNT encima (peso 1)",
    "CAI_N1_NB_PUNT_05": "Nivel 1 de caídas + NB_PUNT encima (peso 0,5)",
    "CAI_N1_NB_PUNT_VOL": "Nivel 1 de caídas + NB_PUNT_VOL encima (peso 1)",
    "CAI_N1_NB_PUNT_L": "Nivel 1 de caídas + NB_PUNT_L encima (peso 1)",
    "CAI_N1_NB_PUNT_AA": "Nivel 1 de caídas + NB_PUNT_AA encima (peso 1, desde 2008)",
    "CAI_N1_NB_PUNT_AA_VOL": "Nivel 1 de caídas + NB_PUNT_AA_VOL encima (peso 1, desde 2008)",
    "CAI_N1": "Nivel 1 de caídas (referencia)", "SPY": "SPY con dividendos (referencia)",
}
# las de año a año solo operan desde 2008: sus ventanas que empiezan antes no se miden
SOLO_DESDE_2008 = {"NB_PUNT_AA", "NB_PUNT_AA_VOL", "CAI_N1_NB_PUNT_AA", "CAI_N1_NB_PUNT_AA_VOL"}


def _tramo(s, a, b):
    x = s[s.index >= pd.Timestamp(a)]
    return x[x.index < pd.Timestamp(b)] if b is not None else x


def _mensual(eq: pd.Series) -> pd.Series:
    v = eq.resample("ME").last()
    return v / np.r_[eq.iloc[0], v.to_numpy()[:-1]] - 1


def estandar(eq: pd.Series, spy: pd.Series) -> dict:
    """Las métricas del marco común, idénticas a ``research/caidas/dossier.py::estandar`` (se
    copian porque los módulos de los dos estudios se llaman igual y no se pueden importar juntos)."""
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


def hora(k: np.ndarray) -> list[str]:
    """Vela k (0 = la que abre a las 9:30) → hora de Nueva York a la que abre."""
    m = 9 * 60 + 30 + np.asarray(k, dtype=int)
    return [f"{x // 60:02d}:{x % 60:02d}" for x in m]


def operaciones(d: dict, r: dict, lev: np.ndarray, clave: str) -> pd.DataFrame:
    o = r["operaciones"]
    de, ds = o.d_ent.astype(int).to_numpy(), o.d_sal.astype(int).to_numpy()
    ks = o.k_sal.astype(int).to_numpy()
    # la salida del cierre se ejecuta al cierre de la última vela (16:00), no a su apertura
    cierre = ks == d["ult"][ds]
    # el control acaba con la vela k_ent - 1 y se ejecuta a la apertura de la vela k_ent, que
    # empieza a la hora del control; igual en la salida
    return pd.DataFrame({
        "estrategia": clave, "fecha": d["fechas"][de].date, "control_entrada": hora(o.k_ent.astype(int)),
        "lado": np.where(o.lado > 0, "largo", "corto"), "precio_entrada": o.px_ent.round(4),
        "control_salida": np.where(cierre, "16:00", hora(ks)),
        "motivo_salida": np.where(cierre, "cierre del día", "control (stop móvil o giro)"),
        "precio_salida": o.px_sal.round(4), "palanca": lev[de].round(4), "ret_neto": o.ret.round(8),
    })


def ejemplo(d: dict, y: pd.DataFrame) -> pd.DataFrame:
    """Dos rupturas reales puntuadas paso a paso con el modelo final: la última de la historia
    que la puntuación deja pasar y la última que no."""
    t = y[ev.en(y, "2005-01-24", None)]
    X = t[mo.RASGOS].to_numpy(np.float64)
    coef = pd.read_csv(RESULTADOS / "puntuacion_coeficientes.csv").set_index("rasgo").coeficiente_pb
    P = mo.Percentiles(X)
    punt = coef["(constante)"] + P(X) @ coef[mo.RASGOS].to_numpy()
    corte = coef["(corte: entrar si la puntuación ≥)"]
    salida = []
    for nombre, i in (("ejemplo 1: entra", np.flatnonzero(punt >= corte)[-1]),
                      ("ejemplo 2: no entra", np.flatnonzero(punt < corte)[-1])):
        salida.append(_ejemplo(d, t.iloc[i], P, coef).assign(ejemplo=nombre))
    return pd.concat(salida)[["ejemplo", "concepto", "valor", "percentil_centrado", "coeficiente_pb",
                              "contribucion_pb"]]


def _ejemplo(d: dict, fila: pd.Series, P, coef: pd.Series) -> pd.DataFrame:
    dd = int(fila.d_ent)
    kd = int(fila.k_ent) - 1
    x = fila[mo.RASGOS].to_numpy(np.float64)[None, :]
    p = P(x)[0]
    contrib = coef[mo.RASGOS].to_numpy() * p
    total = coef["(constante)"] + contrib.sum()
    contexto = {"fecha": str(d["fechas"][dd].date()), "control": hora([kd + 1])[0],
                "ruptura": "al alza (largo)" if fila.lado > 0 else "a la baja (corto)", "apertura": d["O"][dd, 0],
                "cierre_anterior": d["cierre"][dd - 1], "precio_en_el_control": d["C"][dd, kd],
                "sigma_en_el_control": d["sigma"][dd, kd], "banda_superior": d["arriba"][dd, kd],
                "banda_inferior": d["abajo"][dd, kd], "vwap_en_el_control": d["vw"][dd, kd]}
    filas = [{"concepto": k, "valor": v} for k, v in contexto.items()]
    filas += [{"concepto": r, "valor": float(x[0, j]), "percentil_centrado": float(p[j]),
               "coeficiente_pb": float(coef[r]), "contribucion_pb": float(contrib[j])} for j, r in enumerate(mo.RASGOS)]
    filas += [{"concepto": "(constante)", "contribucion_pb": float(coef["(constante)"])},
              {"concepto": "puntuación = constante + suma de contribuciones", "contribucion_pb": float(total)},
              {"concepto": "corte", "valor": float(coef["(corte: entrar si la puntuación ≥)"])},
              {"concepto": "decisión", "valor": "entrar" if total >= coef["(corte: entrar si la puntuación ≥)"]
               else "no entrar"},
              {"concepto": "retorno neto real de la operación del artículo (pb)", "valor": float(fila.ret * 1e4)}]
    return pd.DataFrame(filas)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    GRAFICAS.mkdir(parents=True, exist_ok=True)
    d = nb.cargar()
    v = el.versiones(d, tamaños=tuple(el.TAMAÑOS), esquemas=("dos mitades", "año a año"), lados=(2, 1))
    cai = pd.read_csv(CAIDAS, index_col=0, parse_dates=True)
    cal = cai.index[(cai.index >= INICIO) & (cai.index <= d["fechas"][-1])]
    # las 23 sesiones sin datos intradía cuentan como días sin operar (0); el 19-feb y el 22-nov
    # de 2018 (NYSE cerrada, el CFD abierto) no tienen operaciones
    diario = pd.DataFrame({clave: v[k]["equity"].pct_change().reindex(cal).fillna(0.0) for k, clave in IDS.items()})
    for clave, (base, peso) in COMBOS.items():
        diario[clave] = cai["CAI_N1"].reindex(cal) + peso * diario[base]
    diario["CAI_N1"], diario["SPY"] = cai["CAI_N1"].reindex(cal), cai["SPY"].reindex(cal)
    diario.iloc[0] = 0.0
    diario.index.name = "fecha"
    diario.to_csv(OUT / "diario.csv", float_format="%.7f")
    eqs = (1 + diario).cumprod()
    mensual = pd.DataFrame({k: _mensual(e) for k, e in eqs.items()})
    mensual.index = mensual.index.to_period("M").astype(str)
    mensual.index.name = "mes"
    mensual.to_csv(OUT / "mensual.csv", float_format="%.5f")
    anual = (1 + diario).resample("YE").prod() - 1
    anual.index = anual.index.year
    anual.index.name = "año"
    anual.to_csv(OUT / "anual.csv", float_format="%.5f")

    filas = []
    for clave, eq in eqs.items():
        for vn, (a, b) in VENTANAS.items():
            if clave in SOLO_DESDE_2008 and pd.Timestamp(a) < pd.Timestamp("2007-12-31"):
                continue
            filas.append({"estrategia": clave, "ventana": vn, **estandar(_tramo(eq, a, b), _tramo(eqs["SPY"], a, b))})
    metricas = pd.DataFrame(filas)
    metricas.to_csv(OUT / "metricas.csv", index=False, float_format="%.5f")
    crisis = pd.DataFrame({k: [e.asof(pd.Timestamp(b)) / e.asof(pd.Timestamp(a)) - 1 for _, a, b in CRISIS]
                           for k, e in eqs.items() if k not in SOLO_DESDE_2008})
    crisis.insert(0, "hasta", [b for _, _, b in CRISIS])
    crisis.insert(0, "desde", [a for _, a, _ in CRISIS])
    crisis.insert(0, "tramo", [n for n, _, _ in CRISIS])
    crisis.to_csv(OUT / "crisis.csv", index=False, float_format="%.5f")
    mensual.iloc[1:].corr().to_csv(OUT / "correlaciones_mensuales.csv", float_format="%.3f")

    expo, ops = {}, []
    for (nombre, tam), clave in IDS.items():
        lev = nb.palanca(d, el.TAMAÑOS[tam])
        tiempo = v[(nombre, tam)]["en_mercado"]
        expo[f"{clave}_tiempo_con_posicion"] = tiempo.reindex(cal).fillna(0.0)
        expo[f"{clave}_palanca"] = pd.Series(np.where(tiempo > 0, lev, 0.0), d["fechas"]).reindex(cal).fillna(0.0)
        if clave in ("NB_ART", "NB_F03", "NB_PUNT", "NB_PUNT_AA", "NB_PUNT_VOL"):
            ops.append(operaciones(d, v[(nombre, tam)], lev, clave))
    expo = pd.DataFrame(expo)
    expo.index.name = "fecha"
    expo.to_csv(OUT / "exposicion_diaria.csv", float_format="%.4f")
    pd.concat(ops).to_csv(OUT / "operaciones.csv", index=False)

    R = dict(np.load(rg.SALIDA))
    y = ev.tabla(d, R)
    t = y[ev.en(y, "2005-01-24", None)]
    ent = pd.DataFrame({"fecha": t.fecha.dt.date, "control": hora(t.k_ent.astype(int)),
                        "lado": t.lado.astype(int), **{r: t[r] for r in mo.RASGOS if r != "lado"},
                        "ret_neto": t.ret, "ret_neto_pb": t.ret * 1e4})
    ent.to_csv(OUT / "entrenamiento.csv", index=False, float_format="%.6f")
    for f in ("puntuacion_coeficientes.csv", "puntuacion_percentiles.csv"):
        shutil.copy(RESULTADOS / f, OUT / f)
    ej = ejemplo(d, y)
    ej.to_csv(OUT / "ejemplo_puntuacion.csv", index=False, float_format="%.6f")
    for f in ("curvas.png", "combinar.png", "ronda2_curvas.png", "ronda2_cono.png", "ronda2_combinar.png"):
        shutil.copy(RESULTADOS / f, GRAFICAS / f)

    dsr = pd.read_csv(RESULTADOS / "ronda2_dsr.csv").set_index("version")
    clave_dsr = {"NB_ART": "artículo · 1x", "NB_ART_VOL": "artículo · vol 2 % (máx. 4×)",
                 "NB_ART_L": "artículo (solo largo) · 1x", "NB_F03": "fuerza ≥ 0,3 · 1x",
                 "NB_F03_VOL": "fuerza ≥ 0,3 · vol 2 % (máx. 4×)", "NB_F03_L": "fuerza ≥ 0,3 (solo largo) · 1x",
                 "NB_PUNT": "puntuación, dos mitades · 1x", "NB_PUNT_VOL": "puntuación, dos mitades · vol 2 % (máx. 4×)",
                 "NB_PUNT_L": "puntuación, dos mitades (solo largo) · 1x", "NB_PUNT_AA": "puntuación, año a año · 1x",
                 "NB_PUNT_AA_VOL": "puntuación, año a año · vol 2 % (máx. 4×)"}
    fichas = []
    for clave in diario.columns:
        mt = metricas[metricas.estrategia == clave].set_index("ventana")
        resumen = {vn: {k: float(mt.loc[vn, k]) for k in ("cagr", "peor_caida", "sharpe_rf0", "volatilidad", "calmar")}
                   for vn in mt.index}
        extra = {}
        if clave in clave_dsr:
            fila = dsr.loc[clave_dsr[clave]]
            extra = {"dsr_n_bruto": float(fila.dsr), "dsr_n_efectivo": float(fila.dsr_ef),
                     "n_pruebas": int(fila.pruebas), "n_pruebas_efectivas": int(fila.pruebas_efectivas)}
        fichas.append({"id": clave, "nombre": NOMBRES[clave], "metricas": resumen, **extra})
    (OUT / "estrategias.json").write_text(json.dumps({
        "convenciones": {
            "datos": "OANDA SPX500_USD (CFD sobre el S&P 500, precio medio), velas de 1 minuto, 3-ene-2005 → "
                     "13-may-2020; sesión regular 9:30-16:00 de Nueva York (390 velas); repositorio público "
                     "FutureSharks/financial-data",
            "instrumento_para_operar": "futuros MES/ES (o SPY); la posición se abre y se cierra en el mismo día",
            "senal": "en los controles de las 10:00, 10:30, ..., 15:30, con el cierre de la vela que acaba a esa hora",
            "ejecucion": "a la apertura del minuto siguiente; la salida del final del día, al cierre de las 16:00",
            "costes": "0,0085 $ por acción de SPY y por lado (comisión 0,0035 + media horquilla 0,005), SPY = índice "
                      "/ 10: 0,25-1,3 pb por lado, 0,55 de media (los futuros cuestan ~0,4-0,6 pb)",
            "dinero": "el capital no rinde nada mientras no hay posición (en la combinación con el nivel 1, el "
                      "capital está siempre en las acciones del nivel 1 y Noise Band va encima con futuros)",
            "ventanas": "búsqueda 2013-01-01 → 2020-05-13 (mitades con corte 2017-01-01) y validación 2005-01-24 → "
                        "2012-12-31; la puntuación siempre se mide fuera de muestra (dos mitades o año a año)",
            "dias_sin_datos": "23 sesiones sin datos intradía cuentan como días sin operar",
        },
        "estrategias": fichas,
        "pruebas_totales": {"n_bruto_dsr": 128, "n_efectivo_dsr": 21,
                            "nota": "todas las pruebas de las rondas 1 y 2 con costes de SPY a mercado; efectivas = "
                                    "grupos con correlación diaria > 0,9"},
    }, ensure_ascii=False, indent=1, default=str))

    pd.set_option("display.width", 250)
    tot = metricas[metricas.ventana.isin(["total", "desde_2008"])].pivot_table(
        index="estrategia", columns="ventana", values=["cagr", "peor_caida", "sharpe_rf0", "calmar"])
    print(tot.round(3).to_string())
    print(ej.to_string(index=False))


if __name__ == "__main__":
    main()
