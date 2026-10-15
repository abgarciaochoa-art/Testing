"""Datos del panel «Cómo se comporta la estrategia» (ronda 7).

Lee los resultados de ``anatomia.py``, ``ronda7.py`` y ``que_esperar.py`` (hay que
ejecutarlos antes) y los deja en resultados/comportamiento/ con claves simples para la
página: rasgos con nombre legible, quintiles, variantes de cartera, recorrido de las
operaciones, peor/mejor punto de cada una, meses, rentabilidades a 1-5 años, episodios
de caída y el resumen de qué esperar.

Uso: python research/caidas/exportar_comportamiento.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import ronda7 as r7
from anatomia import EVENTOS

RES = Path(__file__).parent / "resultados"
OUT = RES / "comportamiento"
CORTE = pd.Timestamp("2021-10-01")

# rasgo: (nombre, qué mide, cómo leerlo)
RASGOS = {
    "beta": ("Beta frente al mercado", "Cuánto se mueve con SPY (252 sesiones)",
             "Más beta, mejor: las acciones movidas rebotan más"),
    "sigma10": ("Caída en desviaciones típicas", "Caída de 10 sesiones medida con su propia volatilidad",
                "Caídas normales para esa acción, mejor; desmesuradas, peor"),
    "caida_atr": ("Caída medida en ATR", "Distancia al máximo de 20 sesiones en rangos diarios medios",
                  "Cuanto más desmesurada para su rango, peor"),
    "hueco_entrada": ("Hueco de la apertura de compra", "Apertura del día de compra frente al cierre de la señal",
                      "Si abre aún más abajo, mejor; si abre con subida, peor (se conoce al comprar)"),
    "r1": ("Caída del último día", "Retorno de la sesión de la señal", "Cuanto más cae ese día, mejor"),
    "r3": ("Caída de los últimos 3 días", "Retorno de las 3 últimas sesiones", "Cuanto más rápida, mejor"),
    "dist_ema5": ("Distancia a la EMA5", "Precio frente a su media exponencial de 5 sesiones",
                  "Cuanto más estirada por debajo, mejor"),
    "macd_giro": ("MACD ya girando al alza", "El histograma del MACD sube respecto al día anterior",
                  "Si ya giró, peor: conviene comprar antes del giro"),
    "vol_rel5": ("Volumen alto varios días", "Volumen medio de 5 sesiones frente al de 60",
                 "Volumen alto sostenido (noticia), peor"),
    "gem3": ("Su gemela también cae", "Retorno de 3 sesiones de la acción más correlacionada con ella",
             "Si la gemela también cae (movimiento de sector), mejor"),
    "tam": ("Tamaño de la empresa", "Puesto por volumen en dólares entre los miembros del índice",
            "Cambia de signo: en 2021-26 mejor las grandes; en 2002-21, las pequeñas"),
    "log_dv60": ("Volumen en dólares", "Volumen medio en dólares de 60 sesiones", "Cambia de signo, como el tamaño"),
    "precio": ("Precio de la acción", "Precio sin ajustar", "Cambia de signo entre periodos"),
    "dist_min252": ("Distancia al mínimo anual", "Precio frente al mínimo de 52 semanas",
                    "Muy por encima de su mínimo (tendencia fuerte), mejor"),
    "atr_pct": ("Volatilidad diaria", "Rango diario medio (ATR14) en % del precio", "Más volátil, mejor, salvo extremos"),
    "meta_pct": ("Modelo de ranking", "Probabilidad del modelo de la ronda 3 (ya se usa para ordenar)",
                 "Más alta, mejor"),
    "score_tec": ("Banderas técnicas a favor", "Cuántas de las 11 banderas técnicas cumple", "Más banderas, mejor"),
    "score_v2": ("Módulos de origen a favor", "Cuántos de los 11 módulos de la caída cumple", "Más módulos, mejor"),
    "mom252": ("Momento de 12 meses", "Retorno de las 252 sesiones previas", "Solo funciona en 2024-26"),
    "dd20": ("Profundidad de la caída", "Caída desde el máximo de 20 sesiones", "Sin relación estable"),
    "vol_climax": ("Clímax de volumen", "Volumen del día frente al máximo de 60 sesiones", "Sin relación estable"),
    "spy_dd252": ("Caída del mercado", "Caída de SPY desde su máximo de 252 sesiones", "Cambia de signo"),
    "volreal_spy": ("Volatilidad del mercado", "Volatilidad de SPY a 20 sesiones", "Cambia de signo"),
    "n_cand_dia": ("Caídas el mismo día", "Candidatas que da la estrategia esa misma sesión", "Sin relación estable"),
    "dias_res": ("Días desde los resultados", "Sesiones desde la última publicación de resultados", "Sin relación"),
    "sorpresa": ("Sorpresa de beneficios", "Sorpresa del BPA en los últimos resultados", "Sin relación"),
    "compuesta": ("Puntuación de rasgos", "Media de los 12 rasgos elegidos en la búsqueda (sin el modelo)",
                  "Separa bien operaciones, pero no mejora la cartera"),
}
CLAVES_VARIANTES = {
    "actual: SMA10, máx. 20 sesiones, de golpe, tamaño por módulos": "base",
    "mitad y la 2ª solo si sigue cayendo ≥3%": "tramos_cae3",
    "SMA10 o stop -20% al cierre": "stop20_cierre",
    "SMA10 o stop -8% intradía": "stop8_intradia",
    "fijo: 50 % del capital por operación": "fijo50",
    "solo tiempo: 5 sesiones": "tiempo5",
    "SMA10 u objetivo +5% intradía (orden límite)": "objetivo5_intradia",
    "SMA10 o trailing 5% desde el máximo": "trailing5",
    "módulos × volatilidad (más a las más volátiles)": "elegida_tamano",
}


def compuesta_quintiles(ev: pd.DataFrame) -> pd.DataFrame:
    """Retorno por quintil de la puntuación de rasgos (cortes de la búsqueda)."""
    y = r7.candidatas()[["t", "j", "compuesta"]].merge(ev[["t", "j", "ret", "gana", "fecha"]], on=["t", "j"])
    y = y[y.ret.notna()]
    cortes = np.quantile(y.loc[y.fecha >= CORTE, "compuesta"].dropna(), [0.2, 0.4, 0.6, 0.8])
    y["q"] = np.searchsorted(cortes, y.compuesta, side="right") + 1
    filas = []
    for ventana, sel in (("búsqueda", y.fecha >= CORTE), ("validación", y.fecha < CORTE)):
        for q, g in y[sel].groupby("q"):
            filas.append({"rasgo": "compuesta", "ventana": ventana, "quintil": int(q), "n": len(g),
                          "ret_medio": g.ret.mean(), "acierto": g.gana.mean()})
    return pd.DataFrame(filas)


def grupos_discretos(ev: pd.DataFrame) -> pd.DataFrame:
    """Rasgos con pocos valores: el giro del MACD (no/sí) y los módulos (5 frente a 8 o más),
    guardados como quintil 1 y 5 para la tabla."""
    filas = []
    z = ev[ev.ret.notna()]
    for rasgo, bajo, alto in (("macd_giro", z.macd_giro == 0, z.macd_giro == 1),
                              ("score_v2", z.score_v2 <= 5, z.score_v2 >= 8)):
        for ventana, sel in (("búsqueda", z.fecha >= CORTE), ("validación", z.fecha < CORTE)):
            for q, g in ((1, z[sel & bajo]), (5, z[sel & alto])):
                filas.append({"rasgo": rasgo, "ventana": ventana, "quintil": q, "n": len(g),
                              "ret_medio": g.ret.mean(), "acierto": g.gana.mean()})
    return pd.DataFrame(filas)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ev = pd.read_parquet(EVENTOS)
    disc = pd.read_csv(RES / "anatomia" / "discriminadores.csv").set_index("rasgo")
    quin = pd.read_csv(RES / "anatomia" / "quintiles.csv")
    quin = pd.concat([quin[quin.rasgo.isin(RASGOS)], compuesta_quintiles(ev), grupos_discretos(ev)],
                     ignore_index=True)
    quin["clave"] = quin.rasgo + "_" + quin.ventana.map({"búsqueda": "busq", "validación": "valid"}) + "_q" + \
        quin.quintil.astype(str)
    quin[["clave", "rasgo", "ventana", "quintil", "n", "ret_medio", "acierto"]].to_csv(
        OUT / "quintiles.csv", index=False, float_format="%.5f")

    filas = []
    for orden, (r, (nombre, mide, lectura)) in enumerate(RASGOS.items()):
        q = quin[quin.rasgo == r].set_index(["ventana", "quintil"])
        fila = {"clave": r, "orden": orden, "nombre": nombre, "mide": mide, "lectura": lectura}
        if r in disc.index:
            d = disc.loc[r]
            estado = ("se mantiene" if d.se_mantiene_en_validación else
                      "no se mantiene" if d.elegido_en_búsqueda else "no elegido en la búsqueda")
            fila |= {"ic_busq": d["IC búsqueda"], "t_busq": d["t búsqueda"], "ic_b1": d["IC búsqueda 1ª mitad"],
                     "ic_b2": d["IC búsqueda 2ª mitad"], "ic_valid": d["IC validación"], "t_valid": d["t validación"],
                     "estado": estado}
        else:
            fila["estado"] = "resumen de los elegidos"
        for v, k in (("búsqueda", "busq"), ("validación", "valid")):
            if (v, 1) in q.index and (v, 5) in q.index:
                fila |= {f"q1_{k}": q.loc[(v, 1), "ret_medio"], f"q5_{k}": q.loc[(v, 5), "ret_medio"],
                         f"q1_acierto_{k}": q.loc[(v, 1), "acierto"], f"q5_acierto_{k}": q.loc[(v, 5), "acierto"]}
        filas.append(fila)
    (OUT / "rasgos.json").write_text(json.dumps(filas, ensure_ascii=False, default=float))

    v = pd.read_csv(RES / "ronda7.csv")
    var = pd.DataFrame({
        "id": [CLAVES_VARIANTES.get(n, f"v{i:02d}") for i, n in enumerate(v.variante)], "orden": range(len(v)),
        "familia": v.familia, "variante": v.variante, "cagr_busq": v["CAGR búsqueda"],
        "cagr_b1": v["CAGR búsqueda 1ª mitad"], "cagr_b2": v["CAGR búsqueda 2ª mitad"],
        "cagr_valid": v["CAGR validación"], "cagr_total": v["CAGR total"], "maxdd": v["maxDD total"],
        "sharpe": v.Sharpe, "ops_año": v["operaciones/año"], "acierto": v.acierto, "ret_medio": v["ret. medio"],
        "sesiones": v.sesiones, "en_caidas": v["en caídas (media)"],
        "dif_valid": v["CAGR validación"] - v["CAGR validación"].iloc[0], "dif_dd": v["maxDD total"] - v["maxDD total"].iloc[0],
        "elegible": np.where(v.elegible, "sí", "no"), "elegida": np.where(v.elegida, "sí", "no")})
    (OUT / "variantes.json").write_text(var.to_json(orient="records", force_ascii=False, double_precision=6))

    E = RES / "esperar"
    resumen = json.loads((E / "resumen.json").read_text())
    base = var.iloc[0]
    resumen |= {"clave": "nivel2", "ret_medio_op": float(base.ret_medio), "sesiones_media": float(base.sesiones),
                "ops_año": float(base.ops_año), "n_candidatas": int(ev.ret.notna().sum()),
                "n_rasgos": len(disc), "n_elegidos": int(disc.elegido_en_búsqueda.sum()),
                "n_se_mantienen": int(disc.se_mantiene_en_validación.sum()), "n_variantes": len(var) - 1}
    (OUT / "esperar.json").write_text(json.dumps([resumen], ensure_ascii=False))
    rod = pd.read_csv(E / "rodantes.csv")
    rod["clave"] = rod.horizonte_años.astype(str) + "a_" + rod.serie.map({"nivel 2": "n2", "SPY": "spy"}) + "_" + \
        rod.ventana.map({"validación": "valid", "todo": "todo"})
    rod.to_csv(OUT / "rodantes.csv", index=False, float_format="%.5f")
    epi = pd.read_csv(E / "episodios.csv")
    epi.insert(0, "id", [f"e{k + 1}" for k in range(len(epi))])
    epi["recuperacion"] = epi.recuperacion.fillna("")
    epi.to_csv(OUT / "episodios.csv", index=False, float_format="%.5f")
    mens = pd.read_csv(E / "mensual.csv")
    mens.to_csv(OUT / "mensual.csv", index=False, float_format="%.5f")
    cam = pd.read_csv(E / "camino_cartera.csv")
    cam.insert(0, "clave", cam.grupo + "_" + cam.sesion.astype(str))
    cam.to_csv(OUT / "camino.csv", index=False, float_format="%.5f")
    mae = pd.read_csv(E / "operaciones_mae.csv")
    mae.insert(0, "id", [f"o{k:04d}" for k in range(len(mae))])
    mae.to_csv(OUT / "operaciones_mae.csv", index=False, float_format="%.5f")
    print({f.name: f.stat().st_size for f in sorted(OUT.iterdir())})
    print(var[["id", "familia", "variante", "cagr_valid", "maxdd"]].head(3).to_string())


if __name__ == "__main__":
    main()
