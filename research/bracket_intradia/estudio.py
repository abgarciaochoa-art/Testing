"""Estudio completo de la horquilla intradía. Escribe tablas en ``resultados/``.

Bloques:
 1. base       la horquilla sola (entrar siempre): ¿hay ventaja sin filtro?
 2. filtros    señales de entrada (largos y cortos), por mitades de la muestra
 3. stop       tamaño del stop (fracción del ATR o % fijo) y costes
 4. salidas    objetivo 1/1,5/2/3 R, candado, dejar correr
 5. horario    por franja de entrada
 6. gestion    límite de operaciones y de pérdidas por día

Uso: python research/bracket_intradia/estudio.py [bloque ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datos  # noqa: E402
import motor  # noqa: E402
import senales as sn  # noqa: E402

OUT = Path(__file__).resolve().parent / "resultados"
OUT.mkdir(exist_ok=True)

# costes por lado (fracción del precio) y deslizamiento extra en salidas a mercado
COSTES = {
    "NVDA": (1e-4, 1e-4), "acciones": (1e-4, 1e-4),      # broker barato (≈1-2 céntimos/acción)
    "BTC": (5e-4, 2e-4),                                  # futuros perpetuos, orden a mercado
    "XAU": (1.5e-4, 1e-4), "NAS100": (0.5e-4, 0.5e-4),
}
COSTES_ALTOS = {"NVDA": (3e-4, 1e-4), "acciones": (3e-4, 1e-4), "BTC": (10e-4, 2e-4),
                "XAU": (3e-4, 1e-4), "NAS100": (1e-4, 1e-4)}
LARGAS = {"NVDA": "dukas_NVDA", "BTC": "BTC", "NAS100": "NAS100_oanda", "XAU": "XAU_oanda"}


def preparar(nombre: str) -> dict:
    P = datos.cargar(nombre)
    ok = ~np.isnan(P["atr"]) & ~np.isnan(P["sma50"]) & (P["atr"] > 0)
    if nombre == "dukas_NVDA":  # hueco de descarga: solo el tramo continuo más reciente
        f = pd.to_datetime(P["fechas"])
        ok &= f >= pd.Timestamp("2025-01-01")
    for k in ("O", "H", "L", "C", "V", "atr", "sma20", "sma50", "prev", "ret5", "fechas"):
        P[k] = P[k][ok]
    return P


_CACHE: dict = {}


def panel(nombre: str):
    if nombre not in _CACHE:
        P = preparar(nombre)
        F = sn.rasgos(P)
        Q = sn.espejo(P)
        _CACHE[nombre] = (P, sn.senales(P, F), sn.senales(Q))
    return _CACHE[nombre]


def acciones() -> list[str]:
    return sorted(p.stem for p in datos.PANELES.glob("yahoo2m_*.npz")
                  if p.stem not in ("yahoo2m_BTCUSD", "yahoo2m_GCF", "yahoo2m_GLD",
                                    "yahoo2m_SPY", "yahoo2m_QQQ"))


def correr(nombre: str, senal: str, *, lado: str = "largo", f_atr: float | None = 0.1,
           pct: float | None = None, ventana: str | None = None, **kw) -> tuple[dict, int]:
    P, SL, SC = panel(nombre)
    S = (SL if lado == "largo" else SC)[senal]
    if ventana:
        S = S & sn.ventana(P, ventana)
    R = f_atr * P["atr"] if pct is None else pct * P["C"]
    t = motor.simular(P, S, R, corto=(lado == "corto"), **kw)
    t["fecha"] = P["fechas"][t["dia"]]
    return t, len(P["fechas"])


def correr_acciones(senal: str, **kw) -> tuple[dict, int]:
    """Las 40 acciones de Yahoo (2 minutos) juntas; el t se agrupa por fecha."""
    trozos, fechas = [], set()
    for nombre in acciones():
        t, _ = correr(nombre, senal, **kw)
        trozos.append(t)
        fechas.update(panel(nombre)[0]["fechas"].tolist())
    t = {k: np.concatenate([x[k] for x in trozos]) for k in trozos[0]}
    cal = {f: i for i, f in enumerate(sorted(fechas))}
    t["dia"] = np.array([cal[f] for f in t["fecha"]], dtype=int)
    return t, len(cal) * len(trozos)


def fila(t: dict, D: int, clase: str, altos: bool = False, **extra) -> dict:
    c, s = (COSTES_ALTOS if altos else COSTES)[clase]
    coste = motor.costes_R(t, c, s)
    r = motor.resumen(t, coste, D=D)
    if r["ops"] == 0:
        return {**extra, "ops": 0}
    r["coste_R"] = float(np.mean(coste))
    r["stop_pct"] = float(np.mean(t["R"] / t["entrada"]))
    return {**extra, **r}


def mitades(t: dict, D: int, clase: str, **extra) -> list[dict]:
    """Toda la muestra y cada mitad por fechas (para ver si la señal es estable)."""
    out = [fila(t, D, clase, tramo="todo", **extra)]
    if len(t["pnl"]) == 0:
        return out
    corte = np.sort(np.unique(t["fecha"]))[len(np.unique(t["fecha"])) // 2]
    for nom, m in (("1a_mitad", t["fecha"] < corte), ("2a_mitad", t["fecha"] >= corte)):
        sub = {k: v[m] for k, v in t.items()}
        out.append(fila(sub, D // 2, clase, tramo=nom, **extra))
    return out


SENALES = ["siempre", "tendencia_diaria", "sobre_vwap", "sobre_apertura", "vwap+apertura",
           "tendencia_total", "dia_fuerte", "dia_debil_compra", "bajo_vwap_compra", "orb5",
           "orb15", "orb30", "nuevo_max30", "nuevo_max30+vwap", "retroceso_vwap",
           "retroceso_vwap_sin_td", "vwap+apertura+rvol", "orb5+rvol", "hueco_alcista"]


def todos():
    yield from LARGAS.items()


def bloque_base() -> pd.DataFrame:
    filas = []
    for clase, nombre in todos():
        for f in (0.05, 0.1, 0.2, 0.3):
            for tp in (1, 1.5, 2, 3):
                t, D = correr(nombre, "siempre", f_atr=f, tp=tp)
                filas.append(fila(t, D, clase, inst=clase, f_atr=f, tp=tp,
                                  teorico=1 / (1 + tp)))
    for f in (0.05, 0.1, 0.2):
        for tp in (1, 2, 3):
            t, D = correr_acciones("siempre", f_atr=f, tp=tp)
            filas.append(fila(t, D, "acciones", inst="acciones40", f_atr=f, tp=tp,
                              teorico=1 / (1 + tp)))
    return pd.DataFrame(filas)


def bloque_filtros(f: float = 0.1, tp: float = 2.0) -> pd.DataFrame:
    filas = []
    for lado in ("largo", "corto"):
        for clase, nombre in todos():
            for s in SENALES:
                t, D = correr(nombre, s, lado=lado, f_atr=f, tp=tp)
                filas += mitades(t, D, clase, inst=clase, lado=lado, senal=s)
        for s in SENALES:
            t, D = correr_acciones(s, lado=lado, f_atr=f, tp=tp)
            filas += mitades(t, D, "acciones", inst="acciones40", lado=lado, senal=s)
    return pd.DataFrame(filas)


ELEGIDAS = {  # una de tendencia/momento y una de reversión por activo, más la base
    "largo": ["siempre", "vwap+apertura", "orb30", "dia_fuerte", "bajo_vwap_compra"],
    "corto": ["siempre", "vwap+apertura", "orb30", "dia_fuerte", "bajo_vwap_compra"],
}


def bloque_stop() -> pd.DataFrame:
    filas = []
    for clase, nombre in todos():
        for lado in ("largo", "corto"):
            for s in ELEGIDAS[lado]:
                for f in (0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5):
                    t, D = correr(nombre, s, lado=lado, f_atr=f, tp=2)
                    for altos in (False, True):
                        filas.append(fila(t, D, clase, altos, inst=clase, lado=lado, senal=s,
                                          f_atr=f, costes="altos" if altos else "bajos"))
        # stop fijo en %: la idea original (0,50 $ sobre 145 $ = 0,345 %)
        for pct in (0.0025, 0.00345, 0.005, 0.01):
            t, D = correr(nombre, "siempre", pct=pct, f_atr=None, tp=2)
            filas.append(fila(t, D, clase, inst=clase, lado="largo", senal="siempre",
                              pct=pct, costes="bajos"))
    return pd.DataFrame(filas)


SALIDAS = {
    "1:1": dict(tp=1), "1,5:1": dict(tp=1.5), "2:1": dict(tp=2), "3:1": dict(tp=3),
    "candado 1,8→1 y dinámico 0,8": dict(tp=0, candado=(1.8, 1.0), trail=0.8),
    "a 1R stop a 0, objetivo 3": dict(tp=3, candado=(1.0, 0.0)),
    "a 1R stop a 0, dinámico 1": dict(tp=0, candado=(1.0, 0.0), trail=1.0),
    "dinámico 1R desde la entrada": dict(tp=0, candado=(0.001, -0.999), trail=1.0),
    "sin objetivo (cierre)": dict(tp=0),
}


def bloque_salidas(f: float = 0.1) -> pd.DataFrame:
    filas = []
    for clase, nombre in todos():
        for lado in ("largo", "corto"):
            for s in ELEGIDAS[lado]:
                for nom, kw in SALIDAS.items():
                    t, D = correr(nombre, s, lado=lado, f_atr=f, **kw)
                    filas += mitades(t, D, clase, inst=clase, lado=lado, senal=s, salida=nom)
    for lado in ("largo", "corto"):
        for s in ELEGIDAS[lado]:
            for nom, kw in SALIDAS.items():
                t, D = correr_acciones(s, lado=lado, f_atr=f, **kw)
                filas += mitades(t, D, "acciones", inst="acciones40", lado=lado, senal=s,
                                 salida=nom)
    return pd.DataFrame(filas)


def bloque_horario(f: float = 0.1) -> pd.DataFrame:
    filas = []
    for clase, nombre in todos():
        for lado in ("largo", "corto"):
            for s in ("siempre", "vwap+apertura", "bajo_vwap_compra"):
                for v in sn.VENTANAS:
                    t, D = correr(nombre, s, lado=lado, f_atr=f, tp=2, ventana=v)
                    filas.append(fila(t, D, clase, inst=clase, lado=lado, senal=s, ventana=v))
    for lado in ("largo", "corto"):
        for s in ("siempre", "vwap+apertura", "bajo_vwap_compra"):
            for v in sn.VENTANAS:
                t, D = correr_acciones(s, lado=lado, f_atr=f, tp=2, ventana=v)
                filas.append(fila(t, D, "acciones", inst="acciones40", lado=lado, senal=s,
                                  ventana=v))
    return pd.DataFrame(filas)


def bloque_gestion(f: float = 0.1) -> pd.DataFrame:
    filas = []
    for clase, nombre in todos():
        for lado, s in (("largo", "vwap+apertura"), ("corto", "vwap+apertura"),
                        ("largo", "bajo_vwap_compra"), ("largo", "siempre")):
            for max_ops in (1, 3, 7, 99):
                for max_perd in (2, 3, 99):
                    t, D = correr(nombre, s, lado=lado, f_atr=f, tp=2, max_ops=max_ops,
                                  max_perd=max_perd)
                    c, sl = COSTES[clase]
                    neto = t["pnl"] - motor.costes_R(t, c, sl)
                    dia = np.bincount(t["dia"], weights=neto, minlength=D)
                    filas.append(dict(inst=clase, lado=lado, senal=s, max_ops=max_ops,
                                      max_perd=max_perd, ops_dia=len(neto) / D,
                                      R_dia=dia.mean(), sd_dia=dia.std(),
                                      sharpe_anual=dia.mean() / dia.std() * np.sqrt(252)
                                      if dia.std() > 0 else np.nan,
                                      p_dia_negativo=(dia < 0).mean(),
                                      peor_dia=dia.min(), mejor_dia=dia.max()))
    return pd.DataFrame(filas)


COMBO_SALIDAS = {k: SALIDAS[k] for k in ("2:1", "3:1", "candado 1,8→1 y dinámico 0,8",
                                          "a 1R stop a 0, dinámico 1", "sin objetivo (cierre)")}


def bloque_combo() -> pd.DataFrame:
    """Lo que mejor salió por separado, combinado: filtro de momento, stop más ancho y dejar
    correr; largos, cortos y los dos lados a favor del día."""
    filas = []
    for clase, nombre in [*todos(), ("acciones", None)]:
        for s in ("vwap+apertura", "dia_fuerte", "orb30", "orb5"):
            for f in (0.1, 0.2, 0.3, 0.5):
                for nom, kw in COMBO_SALIDAS.items():
                    for max_ops in (1, 99):
                        ts = {}
                        for lado in ("largo", "corto"):
                            if nombre is None:
                                ts[lado], D = correr_acciones(s, lado=lado, f_atr=f,
                                                              max_ops=max_ops, **kw)
                            else:
                                ts[lado], D = correr(nombre, s, lado=lado, f_atr=f,
                                                     max_ops=max_ops, **kw)
                        ts["ambos"] = {k: np.concatenate([ts["largo"][k], ts["corto"][k]])
                                       for k in ts["largo"]}
                        for lado, t in ts.items():
                            filas += mitades(t, D, clase, inst=clase if nombre else "acciones40",
                                             lado=lado, senal=s, f_atr=f, salida=nom,
                                             max_ops=max_ops)
    return pd.DataFrame(filas)


BLOQUES = {"base": bloque_base, "filtros": bloque_filtros, "stop": bloque_stop,
           "salidas": bloque_salidas, "horario": bloque_horario, "gestion": bloque_gestion,
           "combo": bloque_combo}

if __name__ == "__main__":
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 400)
    for b in sys.argv[1:] or BLOQUES:
        df = BLOQUES[b]()
        df.to_csv(OUT / f"{b}.csv", index=False, float_format="%.4f")
        print(f"[{b}] {len(df)} filas → resultados/{b}.csv", flush=True)
