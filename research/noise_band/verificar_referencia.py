"""Comprueba que la implementación de referencia del dossier (``dossier/referencia.py``),
escrita solo con las reglas de ``dossier/DOSSIER.md``, da lo mismo que el motor del estudio:

1. **velas:** pasar las de OANDA a CSV y volver a leerlas con ``referencia.sesiones``;
2. **operaciones** de NB_ART y NB_F03 frente a ``dossier/datos/operaciones.csv``, y su
   rentabilidad diaria frente a ``dossier/datos/diario.csv``;
3. **puntuación:** los dos ejemplos de ``dossier/datos/ejemplo_puntuacion.csv`` y las
   operaciones con el modelo final frente al motor con el mismo modelo;
4. **reentrenar:** los coeficientes frente a ``dossier/datos/puntuacion_coeficientes.csv``.

Uso: python research/noise_band/verificar_referencia.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

import eventos as ev
import modelos as mo
import motor as nb
import rasgos as rg

DOSSIER = Path(__file__).parent / "dossier"
_spec = importlib.util.spec_from_file_location("referencia", DOSSIER / "referencia.py")
ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ref)
CLAVES = ["fecha", "control_entrada", "lado", "control_salida"]


def comparar(a: pd.DataFrame, b: pd.DataFrame, nombre: str) -> None:
    a, b = a.assign(fecha=a.fecha.astype(str)), b.assign(fecha=b.fecha.astype(str))
    m = a.merge(b, on=CLAVES, how="outer", suffixes=("_ref", "_esperada"), indicator=True)
    ambos = m[m._merge == "both"]
    dp = np.abs(ambos.precio_entrada_ref - ambos.precio_entrada_esperada).max()
    dr = np.abs(ambos.ret_neto_ref - ambos.ret_neto_esperada).max()
    print(f"{nombre}: {len(a)} operaciones de referencia, {len(b)} esperadas, {len(ambos)} iguales en fecha, "
          f"horas y lado ({len(ambos) / max(len(a), len(b)):.2%}); máx. diferencia de precio {dp:.1e}, de retorno "
          f"{dr:.1e}")


def main() -> None:
    z = dict(np.load(nb.DATOS))  # en memoria: cada acceso a un .npz lo vuelve a leer del disco
    fechas = pd.DatetimeIndex(z["fechas"])
    # 1) velas → CSV → sesiones (60 días)
    filas = []
    for d in range(60):
        for k in np.flatnonzero(z["real"][d]):
            filas.append((fechas[d] + pd.Timedelta(minutes=9 * 60 + 30 + int(k)), z["O"][d, k], z["H"][d, k],
                          z["L"][d, k], z["C"][d, k], z["V"][d, k]))
    velas = pd.DataFrame(filas, columns=["fecha_hora", "open", "high", "low", "close", "volume"])
    S1 = ref.sesiones(velas)
    iguales = all(np.allclose(S1[c], z[c][:60]) for c in ("O", "H", "L", "C", "V")) and (
        S1["ultima"] == z["ultimo"][:60]).all()
    print(f"1) velas de 60 días leídas desde CSV: {len(S1['fechas'])} sesiones, matrices idénticas: {iguales}")

    # 2) operaciones del artículo y de la fuerza con todos los datos
    S = {c: z[c] for c in ("O", "H", "L", "C", "V")} | {"ultima": z["ultimo"].astype(int), "fechas": fechas}
    P = ref.preparar(S)
    esperadas = pd.read_csv(DOSSIER / "datos" / "operaciones.csv")
    diario = pd.read_csv(DOSSIER / "datos" / "diario.csv", index_col=0, parse_dates=True)
    for version in ("NB_ART", "NB_F03"):
        ops, valor = ref.simular(P, version)
        comparar(ops, esperadas[esperadas.estrategia == version], f"2) {version}")
        r = valor.pct_change().reindex(diario.index).fillna(0.0)["2005-01-24":]
        x = diario[version]["2005-01-24":]
        print(f"   rentabilidad diaria: correlación {r.corr(x):.6f}, máx. diferencia {np.abs(r - x).max():.1e}")

    # 3) puntuación: los ejemplos y las operaciones con el modelo final
    ej = pd.read_csv(DOSSIER / "datos" / "ejemplo_puntuacion.csv")
    R = ref.rasgos_controles(P)
    punt = ref.Puntuacion()
    for nombre, g in ej.groupby("ejemplo"):
        v = g.set_index("concepto").valor
        d = fechas.get_loc(pd.Timestamp(v["fecha"]))
        c = ref.CONTROLES.index(int(v["control"][:2]) * 60 + int(v["control"][3:]) - (9 * 60 + 30) - 1)
        lado = 1 if "largo" in v["ruptura"] else -1
        x = ref.rasgos_de(R, d, c, lado)
        dif = max(abs(x[r] - float(v[r])) for r in ref.RASGOS if not np.isnan(x[r]))
        p = punt(x)
        esperada = float(g.set_index("concepto").contribucion_pb["puntuación = constante + suma de contribuciones"])
        print(f"3) {nombre}: máx. diferencia en los 25 rasgos {dif:.1e}; puntuación {p:.4f} (esperada {esperada:.4f})")
    ops_ref, _ = ref.simular(P, "NB_PUNT", puntuacion=punt)
    d = nb.cargar()
    Rm = dict(np.load(rg.SALIDA))
    y = ev.tabla(d, Rm)
    e = mo.entrenar(y, Rm, ("2005-01-24", None), "lineal", mo.RASGOS)
    ml, mc = (np.ones((len(d["fechas"]), nb.VELAS)) for _ in range(2))
    ml[:, rg.KS], mc[:, rg.KS] = mo.multiplicador(e, +1, "filtro"), mo.multiplicador(e, -1, "filtro")
    o = nb.simular(d, lados=2, mult_largo=ml, mult_corto=mc, coste_por_accion=0.0085)["operaciones"]
    motor_ops = pd.DataFrame({"fecha": d["fechas"][o.d_ent.astype(int)].date,
                              "control_entrada": [ref.hora(k) for k in o.k_ent.astype(int)],
                              "lado": np.where(o.lado > 0, "largo", "corto"), "precio_entrada": o.px_ent,
                              "control_salida": [("16:00" if k == d["ult"][j] else ref.hora(k))
                                                 for j, k in zip(o.d_sal.astype(int), o.k_sal.astype(int),
                                                                 strict=True)],
                              "ret_neto": o.ret})
    comparar(ops_ref, motor_ops, "3) NB_PUNT con el modelo final (dentro de muestra)")

    # 4) reentrenar desde entrenamiento.csv
    nuevo = ref.reentrenar()
    viejo = pd.read_csv(DOSSIER / "datos" / "puntuacion_coeficientes.csv").set_index("rasgo").coeficiente_pb
    dif = (nuevo - viejo.reindex(nuevo.index)).abs()
    print(f"4) reentrenar: máx. diferencia en los coeficientes {dif[ref.RASGOS].max():.1e} pb; constante "
          f"{nuevo['(constante)']:.4f} ({viejo['(constante)']:.4f}); corte {nuevo['(corte: entrar si la puntuación ≥)']:.4f} "
          f"({viejo['(corte: entrar si la puntuación ≥)']:.4f}); penalización {nuevo['(penalización del ridge)']:.1f}")


if __name__ == "__main__":
    main()
