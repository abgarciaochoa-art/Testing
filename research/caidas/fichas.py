"""Fichas de las estrategias principales: 100.000 $ a 1 año, a 5 años y desde 2002.

Cada estrategia se simula DESDE CERO en cada ventana (100.000 $ en caja a la apertura
del primer día, sin posiciones heredadas) con el motor v3 y las reglas de ejecución de
todo el estudio (señal al cierre, orden a la apertura siguiente, 10 pb por lado, deuda
al 5 %). Para cada estrategia y ventana se guarda:

- el valor de la cartera en $ y la caída desde máximo (diario a 1 y 5 años; semanal
  desde 2002, con la peor caída de cada semana);
- dónde está el dinero al cierre, en % del capital: compras de caídas, reserva en
  acciones, reserva en SPY, caja libre y deuda (lo que pasa de 100 % es deuda);
- las operaciones de caídas (retornos NETOS de costes) y sus estadísticas: número, por
  año, acierto, retorno medio y mediano, sesiones en cartera, mejor y peor, peso;
- la rentabilidad por año natural.

Estrategias: la de referencia de cada ronda (reproducidas con los datos ya corregidos),
la parte de caídas sola (sin reserva) y SPY como comparación.

Salida: resultados/fichas/{estrategias,resumen,anual}.json y {series,operaciones}.csv

Uso: python research/caidas/fichas.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import escalera as es
import eventos as ev
import motor as m
import motor_v3 as v3
import ronda3 as r3
from definitiva import D1_CFG
from finalistas3 import candidatas_c1, candidatas_d1
from ronda5 import componentes_tec
from ronda5b import preparar

OUT = Path(__file__).parent / "resultados" / "fichas"
CAPITAL = 100_000.0
COSTE = 10e-4  # por lado
H = 20

_TEC = ("caída ≥15 % desde el máximo de 20 sesiones, sobre su SMA200, ≥5 de 11 módulos de origen de la caída "
        "y ≥2 de 11 banderas técnicas (estirada bajo EMA5/EMA10/WMA20/VWAP10, Bollinger, estocástico, SMA50, "
        "clímax de volumen, OBV); hasta 5 posiciones de 25-50 % del capital según convicción; vende cuando el "
        "cierre supera la SMA10 o a las 20 sesiones")
ESTRATEGIAS = [
    dict(clave="n2", corto="Nivel 2", ronda="Ronda 6 · recomendada",
         nombre="Nivel 2 · caídas técnicas + reserva contracíclica (máx. 1,5x)",
         reglas=f"Compra a la apertura las acciones del S&P 500 (miembros en esa fecha y líquidas) con {_TEC}. "
                "El resto, en una reserva solo de acciones: con SPY sobre su SMA200, hasta 1,5x en las 100 más "
                "líquidas según la volatilidad de SPY (objetivo 15 %) y lo que falte en las 50 de menor "
                "volatilidad; fuera de tendencia, todo en esas 50.",
         reserva="Acciones: 100 más líquidas / 50 de menor volatilidad",
         deuda="Hasta 1,5x, solo en la reserva y con el mercado en calma (5 %)", solo_acciones="sí"),
    dict(clave="n3", corto="Nivel 3", ronda="Ronda 6",
         nombre="Nivel 3 · como el nivel 2 y caídas con palanca (máx. 2x)",
         reglas="Las mismas compras y la misma reserva que el nivel 2; además, en tendencia las compras de caídas "
                "pueden llevar la exposición total hasta 2x (menos si sube la volatilidad de SPY).",
         reserva="Acciones: 100 más líquidas / 50 de menor volatilidad",
         deuda="Hasta 2x en tendencia (5 %)", solo_acciones="sí"),
    dict(clave="n1", corto="Nivel 1", ronda="Ronda 5",
         nombre="Nivel 1 · caídas técnicas + reserva de acciones, sin deuda",
         reglas=f"Compra a la apertura las acciones del S&P 500 (miembros en esa fecha y líquidas) con {_TEC}. "
                "El resto, en las 100 acciones más líquidas con SPY sobre su SMA200 y en las 50 de menor "
                "volatilidad si no.",
         reserva="Acciones: 100 más líquidas / 50 de menor volatilidad", deuda="Sin deuda", solo_acciones="sí"),
    dict(clave="pura", corto="Solo caídas", ronda="Componente",
         nombre="Solo las compras de caídas (sin reserva: el resto en efectivo)",
         reglas=f"Las compras de los niveles 1-3 ({_TEC}) y nada más: el dinero que no está en caídas queda en "
                "efectivo al 0 %. Muestra cuánto tiempo estaría el capital parado sin reserva.",
         reserva="Efectivo al 0 %", deuda="Sin deuda", solo_acciones="no (efectivo)"),
    dict(clave="d1", corto="D1 ronda 3", ronda="Ronda 3",
         nombre="D1-def50 · módulos de origen de la caída + convicción",
         reglas="Caída ≥15 % desde el máximo de 20 sesiones, sobre su SMA200 y ≥5 de 11 módulos de origen de la "
                "caída; prioridad a más módulos y al modelo; hasta 3 posiciones de 1/6 a 1/2 del capital según "
                "convicción; vende cuando el cierre supera la SMA10 o a las 20 sesiones.",
         reserva="SPY con SPY sobre su SMA200; efectivo si no", deuda="Sin deuda", solo_acciones="no (SPY y efectivo)"),
    dict(clave="c1", corto="C1 ronda 2", ronda="Ronda 2",
         nombre="C1 · caída de 5 sesiones con banderas limpias",
         reglas="Caída ≥7 % en 5 sesiones, sobre su SMA200 y ≥6 de 13 banderas de origen de la caída; prioridad a "
                "más banderas y a la caída de 10 sesiones; hasta 5 posiciones de 1/5; vende con RSI(2) ≥ 70 o a "
                "las 20 sesiones.",
         reserva="SPY con SPY sobre su SMA200; efectivo si no", deuda="Sin deuda", solo_acciones="no (SPY y efectivo)"),
    dict(clave="f1", corto="F1 ronda 1", ronda="Ronda 1",
         nombre="F1 · cae ≥15 % en 10 sesiones",
         reglas="Caída ≥15 % en 10 sesiones y sobre su SMA200; prioridad a la caída mayor; hasta 3 posiciones de "
                "1/3; vende cuando el cierre supera la SMA5 o a las 20 sesiones.",
         reserva="SPY siempre", deuda="Sin deuda", solo_acciones="no (SPY)"),
    dict(clave="spy", corto="SPY", ronda="Referencia", nombre="SPY · comprar y mantener el S&P 500",
         reglas="Compra SPY (el S&P 500 con dividendos reinvertidos) el primer día y lo mantiene.",
         reserva="—", deuda="Sin deuda", solo_acciones="no (ETF)"),
]


def configuraciones() -> dict[str, dict]:
    d = r3.cargar()
    e = ev.cargar()
    ent, score = m.senal({"tipo": "ret", "n": 10, "x": 0.15, "filtro": "tendencia"})
    t, j = np.nonzero(ent)
    vacio = np.zeros(0, np.int64)
    cfg = {"spy": dict(t=vacio, j=vacio, score=np.zeros(0), reserva_="spy"),
           "f1": dict(t=t, j=j, score=score[t, j], K=3, salida="sma5", reserva_="spy")}
    c1 = candidatas_c1(d, e)
    cfg["c1"] = dict(t=c1.t.to_numpy(), j=c1.j.to_numpy(), score=c1.prio.to_numpy(np.float64), K=5, salida="rsi2",
                     reserva_="spy_tend")
    d1 = candidatas_d1(d)
    cfg["d1"] = dict(t=d1.t.to_numpy(), j=d1.j.to_numpy(), score=d1.prio.to_numpy(np.float64),
                     peso=np.minimum(d1.peso.to_numpy(), 0.5 * 3), K=3, salida="sma10", reserva_="spy_tend")
    x, salida, conv = componentes_tec()["D1"]
    y, w = preparar(x, conv, D1_CFG["kmin"], D1_CFG["uso"], D1_CFG["tam"], D1_CFG["frac"])
    tec = dict(t=y.t.to_numpy(), j=y.j.to_numpy(), score=y.prio.to_numpy(np.float64), peso=w, K=5,
               frac=D1_CFG["frac"], salida=salida)
    cfg["pura"] = tec | dict(reserva_="efectivo", coste_reserva=0.0)
    cfg["n1"] = tec | dict(reserva_="liq100_tend_minvol")
    cfg["n2"] = tec | dict(reserva_="liq100_vg_minvol")
    cfg["n3"] = tec | dict(reserva_="liq100_vg_minvol", palanca=2.0, palanca_off=1.0, vol_objetivo=0.15)
    return cfg


def composicion(nombre: str) -> dict[str, np.ndarray]:
    """Reparto de la reserva por sesión (fracciones de su valor): acciones, SPY, efectivo
    y deuda interna (la reserva contracíclica llega a 1,5x en las 100 líquidas)."""
    ex = v3.reserva(nombre)[4]
    uno, cero = np.ones(len(ex)), np.zeros(len(ex))
    return {"spy": dict(acc=cero, spy=uno, libre=cero, deu=cero),
            "spy_tend": dict(acc=cero, spy=ex, libre=1 - ex, deu=cero),
            "efectivo": dict(acc=cero, spy=cero, libre=uno, deu=cero),
            "liq100_tend_minvol": dict(acc=uno, spy=cero, libre=cero, deu=cero),
            "liq100_vg_minvol": dict(acc=ex, spy=cero, libre=cero, deu=ex - 1)}[nombre]


def simular(c: dict, inicio) -> dict:
    kw = dict(c)
    t, j, score = kw.pop("t"), kw.pop("j"), kw.pop("score")
    return v3.simular(t, j, score, H=H, inicio=inicio, detalle=True, **kw)


def ficha(clave: str, ventana: str, r: dict, comp: dict) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    p = m.cargar()
    eq = r["equity"]
    idx = eq.index
    t0 = int(np.searchsorted(p.fechas, idx[0].to_datetime64()))
    sl = slice(t0, t0 + len(idx))
    eqn = eq.to_numpy()
    valor = eqn * CAPITAL
    pico = np.maximum.accumulate(np.r_[CAPITAL, valor])[1:]
    dd = valor / pico - 1
    res_v, deu_v = r["serie_reserva"].to_numpy(), r["serie_deuda"].to_numpy()
    reparto = {"caidas": r["serie_expo"].to_numpy(), "res_acciones": res_v * comp["acc"][sl] / eqn,
               "res_spy": res_v * comp["spy"][sl] / eqn, "libre": res_v * comp["libre"][sl] / eqn,
               "deuda": (deu_v + res_v * comp["deu"][sl]) / eqn}
    resto = reparto["caidas"] + reparto["res_acciones"] + reparto["res_spy"] + reparto["libre"] - reparto["deuda"]
    assert np.abs(resto - 1).max() < 1e-6, (clave, ventana, np.abs(resto - 1).max())
    npos = r["serie_posiciones"].to_numpy()

    años = (idx[-1] - idx[0]).days / 365.25
    ret_d = np.diff(np.r_[1.0, eqn]) / np.r_[1.0, eqn][:-1]
    valle = int(np.argmin(dd))
    previo = np.r_[CAPITAL, valor[: valle + 1]]
    i_pico = int(np.argmax(previo)) - 1  # -1: el pico es el capital inicial
    rec = np.flatnonzero(valor[valle:] >= pico[valle])
    ult_max = np.maximum.accumulate(np.where(dd >= 0, np.arange(len(idx)), 0))
    max_racha = int(np.max(np.where(dd < 0, (idx - idx[ult_max]).days, 0)))  # días naturales bajo el máximo
    fila = {"clave": f"{clave}_{ventana}", "estrategia": clave, "ventana": ventana,
            "inicio": idx[0].date(), "fin": idx[-1].date(), "capital_inicial": CAPITAL,
            "valor_final": valor[-1], "ganancia": valor[-1] - CAPITAL, "rent_total": eqn[-1] - 1,
            "cagr": eqn[-1] ** (1 / años) - 1, "vol": ret_d.std() * np.sqrt(252),
            "sharpe": ret_d.mean() / ret_d.std() * np.sqrt(252), "maxdd": dd[valle],
            "fecha_pico": idx[0].date() if i_pico < 0 else idx[i_pico].date(), "fecha_valle": idx[valle].date(),
            "fecha_recuperacion": idx[valle + rec[0]].date() if len(rec) else "",
            "max_dias_bajo_maximo": max_racha,
            "pct_caidas_medio": reparto["caidas"].mean(), "pct_res_acciones_medio": reparto["res_acciones"].mean(),
            "pct_res_spy_medio": reparto["res_spy"].mean(), "pct_libre_medio": reparto["libre"].mean(),
            "deuda_media": reparto["deuda"].mean(), "deuda_max": reparto["deuda"].max(),
            "bruta_media": 1 + reparto["deuda"].mean() - reparto["libre"].mean(),
            "bruta_max": (1 + reparto["deuda"] - reparto["libre"]).max(),
            "posiciones_medias": npos.mean(), "max_posiciones": int(npos.max()),
            "pct_dias_sin_posiciones": (npos == 0).mean()}

    ops = r["operaciones"].copy()
    ops["ret_neto"] = (1 + ops.ret) * (1 - COSTE) ** 2 - 1
    abierta = ops.t_sal < 0
    ops["sesiones"] = np.where(abierta, len(p.fechas) - ops.t_ent, ops.t_sal - ops.t_ent).astype(int)
    deslist = (~abierta) & (p.ultimo[ops.j.astype(int)] == ops.t_sal) & (ops.t_sal < len(p.fechas) - 1)
    ops["motivo"] = np.select([abierta, deslist, ops.sesiones >= H], ["abierta", "deslistada", "límite de 20 sesiones"],
                              "señal de salida")
    pos_prev = (ops.t_ent - t0 - 1).astype(int).to_numpy()
    base = np.where(pos_prev >= 0, valor[np.clip(pos_prev, 0, None)], CAPITAL)
    ops["ganancia_usd"] = ops.peso * base * ops.ret_neto
    c = ops[~abierta]
    gana, pierde = c.ret_neto[c.ret_neto > 0], c.ret_neto[c.ret_neto <= 0]
    if len(c):
        fila |= {"operaciones": len(c), "abiertas_final": int(abierta.sum()), "op_año": len(c) / años,
                 "acierto": (c.ret_neto > 0).mean(), "ret_medio": c.ret_neto.mean(), "ret_mediana": c.ret_neto.median(),
                 "ganancia_media": gana.mean() if len(gana) else np.nan,
                 "perdida_media": pierde.mean() if len(pierde) else np.nan,
                 "mejor_op": c.ret_neto.max(), "peor_op": c.ret_neto.min(),
                 "sesiones_media": c.sesiones.mean(), "sesiones_mediana": c.sesiones.median(),
                 "dias_naturales_media": (c.salida - c.entrada).dt.days.mean(), "peso_medio": c.peso.mean(),
                 "pct_salida_tiempo": (c.motivo == "límite de 20 sesiones").mean(),
                 "ganancia_ops_usd": c.ganancia_usd.sum()}
    else:
        fila |= {"operaciones": 0, "abiertas_final": int(abierta.sum())}

    serie = pd.DataFrame({"fecha": idx, "valor": valor, "dd": dd, **reparto, "posiciones": npos})
    if ventana == "ini":  # semanal: cierre de la semana y su peor caída
        g = serie.groupby(idx.to_period("W-FRI"))
        serie = g.last().assign(dd=g.dd.min().to_numpy()).reset_index(drop=True)
    serie.insert(0, "ventana", ventana)
    serie.insert(0, "estrategia", clave)

    s_val = pd.Series(valor, idx)
    anual = s_val.groupby(idx.year).last()
    ret_a = anual / anual.shift(1).fillna(CAPITAL) - 1
    año_ini = idx[0].year
    parcial = [(a == año_ini and (idx[0].month > 1 or idx[0].day > 7)) or (a == idx[-1].year and idx[-1].month < 12)
               for a in anual.index]
    anual = pd.DataFrame({"clave": [f"{clave}_{ventana}_{a}" for a in anual.index], "estrategia": clave,
                          "ventana": ventana, "año": anual.index, "ret": ret_a.to_numpy(),
                          "parcial": np.where(parcial, "sí", "no")})

    ops = ops.reset_index(drop=True)
    tabla = pd.DataFrame({"id": [f"{clave}_{ventana}_{k:04d}" for k in range(len(ops))], "estrategia": clave,
                          "ventana": ventana, "ticker": ops.ticker, "entrada": ops.entrada.dt.date,
                          "salida": [x.date() if pd.notna(x) else "" for x in ops.salida], "sesiones": ops.sesiones,
                          "peso": ops.peso, "ret": ops.ret_neto, "ganancia_usd": ops.ganancia_usd,
                          "motivo": ops.motivo})
    return fila, serie, anual, tabla


def a_json(df: pd.DataFrame, nombre: str) -> None:
    """Filas como objetos JSON (NaN -> null, fechas como texto)."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).replace({"nan": None, "None": None})
    filas = json.loads(df.to_json(orient="records", force_ascii=False, double_precision=10))
    (OUT / nombre).write_text(json.dumps(filas, ensure_ascii=False))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p = m.cargar()
    n = len(p.fechas)
    v3._RES["efectivo"] = (np.ones(n), np.ones(n), np.zeros(n), np.zeros(n, dtype=bool), np.zeros(n))
    fechas = pd.DatetimeIndex(p.fechas)
    ventanas = {"1a": fechas[-1] - pd.DateOffset(years=1), "5a": fechas[-1] - pd.DateOffset(years=5),
                "ini": pd.Timestamp("2002-01-02")}
    cfg = configuraciones()
    filas, series, anuales, tablas, meta = [], [], [], [], []
    for orden, info in enumerate(ESTRATEGIAS):
        clave = info["clave"]
        comp = composicion(cfg[clave]["reserva_"])
        for ventana, inicio in ventanas.items():
            r = simular(cfg[clave], inicio)
            fila, serie, anual, tabla = ficha(clave, ventana, r, comp)
            filas.append(fila | {"nombre": info["nombre"], "corto": info["corto"]})
            series.append(serie)
            anuales.append(anual)
            tablas.append(tabla)
            if ventana == "ini":
                mv = es.metricas_ventanas(r["equity"])
                meta.append(info | {"orden": orden, "cagr_validacion": mv["CAGR validación"],
                                    "maxdd_validacion": mv["maxDD validación"], "cagr_busqueda": mv["CAGR búsqueda"],
                                    "maxdd_busqueda": mv["maxDD búsqueda"]})
        print(clave, "ok", flush=True)
    res = pd.DataFrame(filas)
    spy = res[res.estrategia == "spy"].set_index("ventana")
    res["spy_valor_final"] = res.ventana.map(spy.valor_final)
    res["spy_rent_total"] = res.ventana.map(spy.rent_total)
    res["spy_cagr"] = res.ventana.map(spy.cagr)
    res["exceso_cagr"] = res.cagr - res.spy_cagr
    a_json(res, "resumen.json")
    a_json(pd.DataFrame(meta), "estrategias.json")
    ser = pd.concat(series, ignore_index=True)
    ser["fecha"] = pd.to_datetime(ser.fecha).dt.date
    ser["valor"] = ser.valor.round(0).astype(np.int64)
    for col in ("dd", "caidas", "res_acciones", "res_spy", "libre", "deuda"):
        ser[col] = ser[col].round(4)
    ser.to_csv(OUT / "series.csv", index=False)
    a_json(pd.concat(anuales, ignore_index=True), "anual.json")
    ops = pd.concat(tablas, ignore_index=True)
    ops["ganancia_usd"] = ops.ganancia_usd.round(0)
    ops.to_csv(OUT / "operaciones.csv", index=False, float_format="%.5f")

    pd.set_option("display.width", 250)
    cols = ["clave", "inicio", "valor_final", "rent_total", "cagr", "maxdd", "sharpe", "operaciones", "op_año",
            "acierto", "ret_medio", "ret_mediana", "sesiones_media", "pct_caidas_medio", "pct_libre_medio",
            "deuda_media", "bruta_max", "pct_dias_sin_posiciones"]
    print(res[cols].round(3).to_string(index=False))
    print({f.name: f.stat().st_size for f in sorted(OUT.iterdir())})


if __name__ == "__main__":
    main()
