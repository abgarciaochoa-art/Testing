"""Prueba del semáforo de tendencia.

A) Deriva hasta el cierre según el estado del semáforo en cada hora de control: si el estado
   identifica la tendencia, el precio debe seguir en su dirección (ventaja > 0) en verde y no
   en rojo.
B) Operativa: entrar la primera vez que el semáforo se pone en un estado (entre las 10:00 y las
   14:00), en su dirección, con stop de k·ATR y salida al cierre o con candado; una operación
   por valor y día. Costes como en el estudio de la horquilla.

Uso: python research/tendencia_intradia/prueba_semaforo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent / "bracket_intradia"))
import estudio as B  # noqa: E402  (paneles de 1 minuto y costes)
import motor  # noqa: E402
import paneles as T  # noqa: E402
import semaforo as S  # noqa: E402

OUT = AQUI / "resultados"
ETFS = ["SPY", "QQQ", "IWM", "SMH", "XLK"]
CP = [30, 60, 90, 120, 180, 240, 300]


def acciones() -> list[str]:
    return sorted(p.stem[3:] for p in T.PANELES.glob("h1_*.npz") if p.stem[3:] not in ETFS)


def fuentes(nombre: str):
    """(nombre del valor, panel, estados) de cada valor del conjunto."""
    if nombre in ("acc_h1", "acc_m5", "etf_h1"):
        pre = "h1" if nombre.endswith("h1") else "m5"
        spy = T.cargar(f"{pre}_SPY")
        for tk in (ETFS if nombre == "etf_h1" else acciones()):
            f = T.PANELES / f"{pre}_{tk}.npz"
            if f.exists():
                P = T.cargar(f"{pre}_{tk}")
                M = S.matrices(P, None if tk == "SPY" else spy)
                yield tk, P, S.estado(M)
    else:
        nom = {"NAS100": "NAS100_oanda", "BTC": "BTC", "NVDA_1m": "dukas_NVDA"}[nombre]
        P = B.preparar(nom)
        yield nombre, P, S.estado(S.matrices(P), usar_mercado=False)


def deriva(nombre: str) -> pd.DataFrame:
    filas = []
    for tk, P, E in fuentes(nombre):
        paso = int(P["paso"])
        C, atr = P["C"], P["atr"]
        for cp in CP:
            j = cp // paso - 1
            if j < 0 or j >= C.shape[1] - 1 or cp % paso:
                continue
            e = E[:, j]
            y = (C[:, -1] - C[:, j]) / atr
            d = np.sign(C[:, j] - P["O"][:, 0])
            filas.append(pd.DataFrame({"fecha": pd.to_datetime(P["fechas"]), "valor": tk, "cp": cp,
                                       "estado": np.abs(e), "dir": d, "y": y}))
    df = pd.concat(filas)
    df = df[np.isfinite(df.y) & (df.dir != 0)]
    df["g"] = df.dir * df.y
    fechas = np.sort(df.fecha.unique())
    corte = fechas[len(fechas) // 2]
    out = []
    for (cp, e), g in df.groupby(["cp", "estado"]):
        pdia = g.groupby("fecha").g.mean()
        out.append({"conjunto": nombre, "cp": cp, "estado": S.NOMBRES[e], "n": len(g),
                    "por_dia": len(g) / df.fecha.nunique() / max(1, df.valor.nunique()) * 100,
                    "ventaja": g.g.mean(), "acierto": (g.g > 0).mean(),
                    "t": pdia.mean() / pdia.std() * np.sqrt(len(pdia)) if len(pdia) > 2 else np.nan,
                    "mitad1": g[g.fecha < corte].g.mean(), "mitad2": g[g.fecha >= corte].g.mean()})
    return pd.DataFrame(out)


SALIDAS = {"cierre": dict(tp=0), "candado": dict(tp=0, candado=(1.8, 1.0), trail=0.8),
           "2:1": dict(tp=2)}


def operativa(nombre: str) -> pd.DataFrame:
    clase = "BTC" if nombre == "BTC" else "NAS100" if nombre == "NAS100" else "acciones"
    c, sl = B.COSTES[clase]
    filas = []
    for umbral in (0, 1, 2, 3):
        for f in (0.15, 0.25, 0.5):
            for sal, kw in SALIDAS.items():
                trozos, nvalores, todas = [], 0, set()
                for tk, P, E in fuentes(nombre):
                    nvalores += 1
                    todas.update(P["fechas"].tolist())
                    paso = int(P["paso"])
                    N = P["C"].shape[1]
                    minuto = np.arange(N) * paso + paso  # minuto del cierre de cada vela
                    ventana = (minuto >= 30) & (minuto <= 270)
                    for lado, sg in (("largo", 1), ("corto", -1)):
                        sen = ((E * sg >= umbral) if umbral > 0 else np.ones_like(E, bool)) & ventana[None, :]
                        if umbral == 0:  # referencia: primera vela que cierra desde las 10:00, a favor del día
                            primera = minuto == minuto[minuto >= 30].min()
                            sen = (np.sign(P["C"] - P["O"][:, :1]) == sg) & primera[None, :]
                        t = motor.simular(P, sen, f * P["atr"], corto=(sg < 0), max_ops=1,
                                          cierre=N - 1, **kw)
                        t["fecha"] = P["fechas"][t["dia"]]
                        trozos.append(t)
                t = {k: np.concatenate([x[k] for x in trozos]) for k in trozos[0]}
                if len(t["pnl"]) == 0:
                    continue
                cal = {d: i for i, d in enumerate(sorted(set(t["fecha"].tolist())))}
                t["dia"] = np.array([cal[d] for d in t["fecha"]])
                coste = motor.costes_R(t, c, sl)
                r = motor.resumen(t, coste)
                fechas = np.sort(np.unique(t["fecha"]))
                corte = fechas[len(fechas) // 2]
                m1 = t["fecha"] < corte
                neto = t["pnl"] - coste
                filas.append({"conjunto": nombre, "umbral": S.NOMBRES[umbral],
                              "f_atr": f, "salida": sal, "ops": r["ops"],
                              "ops_por_100_valores_dia": 100 * r["ops"] / len(todas) / nvalores,
                              "acierto": r["acierto"], "R_bruto": r["R_bruto"], "R_neto": r["R_neto"],
                              "t": r["t"], "coste_R": float(coste.mean()),
                              "neto_mitad1": neto[m1].mean(), "neto_mitad2": neto[~m1].mean()})
    df = pd.DataFrame(filas)
    df.loc[df.umbral == "rojo", "umbral"] = "a favor del día desde las 10:00"
    return df


# Nvidia de Dukascopy no entra: su volumen es el del CFD y no sirve para el volumen relativo
CONJ = ["acc_h1", "acc_m5", "etf_h1", "NAS100", "BTC"]

if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    a = pd.concat([deriva(c) for c in CONJ])
    a.to_csv(OUT / "semaforo_deriva.csv", index=False, float_format="%.4f")
    print(a[a.cp.isin([60, 120])].round(3).to_string())
    b = pd.concat([operativa(c) for c in CONJ])
    b.to_csv(OUT / "semaforo_operativa.csv", index=False, float_format="%.4f")
    print(b[(b.f_atr == 0.25)].round(3).to_string())
