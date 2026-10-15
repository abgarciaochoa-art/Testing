"""Qué esperar del nivel 2 (reserva contracíclica, compras de caídas sin deuda).

Con la simulación continua desde 2002 (100 % = el capital del día):
- meses: rentabilidad, frente a SPY, operaciones cerradas, meses sin ninguna compra;
- rachas: operaciones perdedoras seguidas, meses negativos seguidos, periodos sin
  compras, tiempo por debajo del máximo;
- episodios de caída: los 8 más profundos, con fechas de pico, mínimo y recuperación;
- rentabilidades a 1, 3 y 5 años desde cualquier semana (percentiles), en la validación
  (2002 → sep 2021, lo que las reglas no vieron) y en todo el periodo;
- el recorrido de las operaciones que la cartera hizo de verdad (con ``anatomia.py``).

Salida: resultados/esperar/{mensual.csv, episodios.csv, rodantes.csv, camino_cartera.csv,
operaciones_mae.csv, resumen.json}

Uso: python research/caidas/que_esperar.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import motor as m
import motor_v4 as v4
from anatomia import EVENTOS
from definitiva import D1_CFG
from ronda5 import componentes_tec
from ronda5b import preparar

OUT = Path(__file__).parent / "resultados" / "esperar"
CORTE = pd.Timestamp("2021-10-01")
H = 20


def simular() -> dict:
    x, salida, conv = componentes_tec()["D1"]
    y, w = preparar(x, conv, D1_CFG["kmin"], D1_CFG["uso"], D1_CFG["tam"], D1_CFG["frac"])
    return v4.simular(y.t.to_numpy(), y.j.to_numpy(), y.prio.to_numpy(np.float64), peso=w, K=5, frac=0.5, H=20,
                      salida=salida, reserva_="liq100_vg_minvol", palanca=1.0, palanca_off=1.0, detalle=True)


def episodios(eq: pd.Series, spy: pd.Series, n: int = 8) -> pd.DataFrame:
    filas = []
    pico_v, pico_f, valle_v, valle_f = eq.iloc[0], eq.index[0], eq.iloc[0], eq.index[0]
    for f, v in eq.items():
        if v >= pico_v:
            if valle_v < pico_v * 0.97:
                filas.append((pico_f, valle_f, f, valle_v / pico_v - 1))
            pico_v, pico_f, valle_v, valle_f = v, f, v, f
        elif v < valle_v:
            valle_v, valle_f = v, f
    if valle_v < pico_v * 0.97:
        filas.append((pico_f, valle_f, pd.NaT, valle_v / pico_v - 1))
    df = pd.DataFrame(filas, columns=["pico", "valle", "recuperacion", "caida"]).sort_values("caida").head(n)
    df["dias_hasta_valle"] = (df.valle - df.pico).dt.days
    df["dias_hasta_recuperar"] = (df.recuperacion - df.valle).dt.days
    df["spy_en_el_tramo"] = [spy.loc[a:b].min() / spy.loc[:a].iloc[-1] - 1 for a, b in zip(df.pico, df.valle,
                                                                                               strict=True)]
    return df.reset_index(drop=True)


def rodantes(eq: pd.Series, spy: pd.Series) -> pd.DataFrame:
    sem_e = eq.resample("W-FRI").last()
    sem_s = spy.resample("W-FRI").last()
    filas = []
    for años in (1, 3, 5):
        k = 52 * años
        for nombre, s in (("nivel 2", sem_e), ("SPY", sem_s)):
            r = (s.shift(-k) / s) ** (1 / años) - 1
            r = r.dropna()
            for ventana, sel in (("validación", r.index + pd.DateOffset(years=años) <= CORTE), ("todo", r.index >= r.index[0])):
                z = r[sel]
                filas.append({"horizonte_años": años, "serie": nombre, "ventana": ventana, "n": len(z),
                              **{f"p{q}": z.quantile(q / 100) for q in (5, 10, 25, 50, 75, 90, 95)},
                              "negativos": (z < 0).mean(), "peor": z.min(), "mejor": z.max()})
    return pd.DataFrame(filas)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    r = simular()
    eq = r["equity"]
    spy = m.benchmark("SPY", inicio=np.datetime64(eq.index[0]))
    spy = spy / spy.iloc[0]
    ops = r["operaciones"]
    c = ops[ops.t_sal >= 0].copy()
    c["mes"] = pd.to_datetime(c.salida).dt.to_period("M")
    ent_mes = pd.to_datetime(ops.entrada).dt.to_period("M").value_counts()

    mensual = pd.DataFrame({"ret": eq.resample("ME").last().pct_change(), "ret_spy": spy.resample("ME").last().pct_change()})
    mensual.iloc[0] = [eq.resample("ME").last().iloc[0] - 1, spy.resample("ME").last().iloc[0] - 1]
    per = mensual.index.to_period("M")
    mensual["compras"] = per.map(ent_mes).fillna(0).astype(int)
    mensual["ventas"] = per.map(c.mes.value_counts()).fillna(0).astype(int)
    mensual["ventana"] = np.where(mensual.index < CORTE, "validación", "búsqueda")
    mensual.index = per.astype(str)
    mensual.index.name = "mes"
    mensual.to_csv(OUT / "mensual.csv", float_format="%.5f")

    epi = episodios(eq, spy)
    epi.to_csv(OUT / "episodios.csv", index=False, float_format="%.5f")
    rod = rodantes(eq, spy)
    rod.to_csv(OUT / "rodantes.csv", index=False, float_format="%.5f")

    # rachas de operaciones y de meses
    gana = (c.sort_values("t_sal").ret > 0).to_numpy()
    rach, mx = 0, 0
    for g in gana:
        rach = 0 if g else rach + 1
        mx = max(mx, rach)
    rachas = pd.Series((~gana).astype(int)).groupby((gana).cumsum()).sum()
    neg = (mensual.ret < 0).to_numpy()
    rm, mxm = 0, 0
    for g in neg:
        rm = rm + 1 if g else 0
        mxm = max(mxm, rm)
    # periodos sin ninguna posición de caídas
    npos = r["serie_posiciones"]
    vacio = (npos == 0).to_numpy()
    largo, mx_vacio, ini_v, mejor_ini = 0, 0, None, None
    for k, v_ in enumerate(vacio):
        if v_:
            if largo == 0:
                ini_v = npos.index[k]
            largo += 1
            if largo > mx_vacio:
                mx_vacio, mejor_ini = largo, ini_v
        else:
            largo = 0
    dd = eq / eq.cummax() - 1
    val = mensual[mensual.ventana == "validación"]
    resumen = {
        "meses_positivos": float((mensual.ret > 0).mean()), "meses_positivos_validacion": float((val.ret > 0).mean()),
        "mes_mediano": float(mensual.ret.median()), "mes_medio": float(mensual.ret.mean()),
        "peor_mes": float(mensual.ret.min()), "peor_mes_fecha": str(mensual.ret.idxmin()),
        "mejor_mes": float(mensual.ret.max()), "mejor_mes_fecha": str(mensual.ret.idxmax()),
        "meses_mejor_que_spy": float((mensual.ret > mensual.ret_spy).mean()),
        "max_meses_negativos_seguidos": int(mxm),
        "meses_sin_compras": float((mensual.compras == 0).mean()),
        "compras_por_mes_mediana": float(mensual.compras.median()), "compras_por_mes_p90": float(mensual.compras.quantile(0.9)),
        "max_perdedoras_seguidas": int(mx), "perdedoras_seguidas_p90": float(rachas[rachas > 0].quantile(0.9)),
        "max_sesiones_sin_posiciones": int(mx_vacio), "desde_sin_posiciones": str(mejor_ini.date()) if mejor_ini is not None else "",
        "dias_sin_posiciones": float(vacio.mean()), "tiempo_con_caida_mas_de_10": float((dd < -0.10).mean()),
        "tiempo_con_caida_mas_de_20": float((dd < -0.20).mean()),
    }

    # recorrido de las operaciones que hizo la cartera
    ev = pd.read_parquet(EVENTOS)
    hechas = c.assign(t=c.t_ent.astype(int) - 1, j=c.j.astype(int))[["t", "j", "ret"]].rename(columns={"ret": "ret_cartera"})
    z = hechas.merge(ev, on=["t", "j"], how="left")
    filas = []
    for grupo, sel in (("todas", z.ret_cartera.notna()), ("ganadoras", z.ret_cartera > 0), ("perdedoras", z.ret_cartera <= 0)):
        g = z[sel]
        for k in range(1, H + 1):
            col = g[f"c{k}"].dropna()
            filas.append({"grupo": grupo, "sesion": k, "n": len(col), "media": col.mean(), "mediana": col.median(),
                          "p25": col.quantile(0.25), "p75": col.quantile(0.75), "p10": col.quantile(0.1),
                          "p90": col.quantile(0.9), "en_ganancia": (col > 0).mean()})
    pd.DataFrame(filas).to_csv(OUT / "camino_cartera.csv", index=False, float_format="%.5f")
    z.assign(entrada=pd.DatetimeIndex(m.cargar().fechas[z.t.to_numpy() + 1]).date)[
        ["entrada", "ticker", "ret_cartera", "mae", "mfe", "sesiones", "dia_mae"]].to_csv(
        OUT / "operaciones_mae.csv", index=False, float_format="%.5f")
    g, pz = z[z.ret_cartera > 0], z[z.ret_cartera <= 0]
    resumen |= {"operaciones": len(z), "acierto": float((z.ret_cartera > 0).mean()),
                "mae_mediano": float(z.mae.median()), "mae_p10": float(z.mae.quantile(0.1)),
                "mfe_mediano": float(z.mfe.median()), "ganadoras_mae_mediano": float(g.mae.median()),
                "ganadoras_pasaron_por_-5": float((g.mae <= -0.05).mean()),
                "ganadoras_pasaron_por_-10": float((g.mae <= -0.10).mean()),
                "perdedoras_llegaron_a_+3": float((pz.mfe >= 0.03).mean()),
                "dia_peor_punto_mediano": float(z.dia_mae.median()),
                "en_ganancia_dia1": float((z.c1 > 0).mean()), "en_ganancia_dia3": float((z.c3 > 0).mean())}
    (OUT / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=1))
    pd.set_option("display.width", 220)
    print(json.dumps(resumen, ensure_ascii=False, indent=1))
    print(epi.to_string())
    print(rod.round(3).to_string())


if __name__ == "__main__":
    main()
