"""Noise Band: implementación de referencia, escrita a partir de DOSSIER.md (sección 4).

Python legible (numpy y pandas; scikit-learn solo para reentrenar la puntuación), sin el motor
del estudio. Sirve para entender las reglas al detalle y para comprobar una réplica: con los
datos del estudio da las mismas operaciones que ``datos/operaciones.csv`` (lo comprueba
``research/noise_band/verificar_referencia.py``).

Uso:
    python referencia.py velas.csv --version NB_ART     # o NB_F03 o NB_PUNT; --solo-largo; --palanca vol
    python referencia.py --reentrenar                    # rehace el modelo con datos/entrenamiento.csv

``velas.csv``: una fila por minuto con las columnas fecha_hora (hora de Nueva York, la de
apertura de la vela: AAAA-MM-DD HH:MM), open, high, low, close y volume. Escribe
``operaciones_referencia.csv`` (una fila por operación) y muestra la rentabilidad anual.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).parent
VELAS = 390                              # 9:30-16:00 en velas de 1 minuto
CONTROLES = list(range(29, 360, 30))     # velas que acaban a las 10:00, 10:30, ..., 15:30
N_SIGMA = 14                             # sesiones para el cono
COSTE_POR_ACCION = 0.0085                # $ por acción de SPY y por lado (SPY ≈ índice / 10)
RASGOS = ["hora", "fuerza", "mov", "hueco", "hueco_abs", "sigma_rel", "vol5", "vol14", "vol60", "vol5_60",
          "dd252", "tend50", "tend200", "ret1", "ret1_dia", "vrel", "vrel30", "vwap", "rango", "volmin30",
          "dia_semana", "vencimiento", "fin_mes", "inicio_mes", "lado"]
CON_SIGNO = {"mov", "hueco", "tend50", "tend200", "ret1", "ret1_dia", "vwap"}


def hora(k: int) -> str:
    """Hora a la que abre la vela k (la 0, a las 9:30)."""
    m = 9 * 60 + 30 + int(k)
    return f"{m // 60:02d}:{m % 60:02d}"


# ------------------------------------------------------------------ 4.1 datos y sesión
def sesiones(velas: pd.DataFrame) -> dict:
    """Velas de 1 minuto → matrices días × 390 de la sesión regular (reglas de 4.1)."""
    t = pd.to_datetime(velas["fecha_hora"])
    m = (t.dt.hour * 60 + t.dt.minute - (9 * 60 + 30)).to_numpy()
    ok = (m >= 0) & (m < VELAS)
    fechas, fila = np.unique(t.dt.normalize().to_numpy()[ok], return_inverse=True)
    D = len(fechas)
    M = {}
    for c, col in (("O", "open"), ("H", "high"), ("L", "low"), ("C", "close"), ("V", "volume")):
        M[c] = np.full((D, VELAS), np.nan)
        M[c][fila, m[ok]] = velas[col].to_numpy(float)[ok]
    real = ~np.isnan(M["C"])
    primera = real.argmax(1)
    apertura = M["O"][np.arange(D), primera]
    # vela sin cotización = el último cierre del día (antes de la primera, su apertura)
    C = pd.DataFrame(M["C"]).ffill(axis=1).to_numpy()
    C = np.where(np.isnan(C), apertura[:, None], C)
    S = {c: np.where(real, M[c], C) for c in ("O", "H", "L")}
    S["C"], S["V"] = C, np.nan_to_num(M["V"])
    S["ultima"] = VELAS - 1 - real[:, ::-1].argmax(1)       # última vela con cotización
    validos = (real.sum(1) >= 200) & (primera < 5)           # se descartan los días incompletos
    S = {k: v[validos] for k, v in S.items()}
    S["fechas"] = pd.DatetimeIndex(fechas[validos])
    return S


# ------------------------------------------------------------------ 4.2-4.8 cono, VWAP, tamaño y costes
def preparar(S: dict) -> dict:
    O, H, L, C, V = S["O"], S["H"], S["L"], S["C"], S["V"]
    D = len(S["fechas"])
    apertura = O[:, 0]
    cierre = C[np.arange(D), S["ultima"]]
    anterior = np.r_[apertura[0], cierre[:-1]]
    # sigma(d, k): media de |precio(k) / apertura - 1| en las 14 sesiones anteriores
    sigma = pd.DataFrame(np.abs(C / apertura[:, None] - 1)).rolling(N_SIGMA).mean().shift(1).to_numpy()
    tope, suelo = np.maximum(apertura, anterior), np.minimum(apertura, anterior)
    tipico = (H + L + C) / 3
    vol_acum = np.cumsum(V, 1)
    vwap = np.where(vol_acum > 0, np.cumsum(tipico * V, 1) / np.maximum(vol_acum, 1e-12),
                    np.cumsum(tipico, 1) / np.arange(1, VELAS + 1))
    vol14 = pd.Series(cierre).pct_change().rolling(14).std().shift(1).to_numpy()
    return dict(S, apertura=apertura, cierre=cierre, anterior=anterior, sigma=sigma, tope=tope, suelo=suelo,
                UB=tope[:, None] * (1 + sigma), LB=suelo[:, None] * (1 - sigma), vwap=vwap, vol14=vol14,
                coste=COSTE_POR_ACCION / (anterior / 10))   # por lado, en tanto por uno


def palanca(P: dict, tipo: str = "1x") -> np.ndarray:
    """1× o volatilidad objetivo: min(4; 2 % / volatilidad diaria de 14 sesiones); sin dato, 0."""
    if tipo == "1x":
        return np.ones(len(P["fechas"]))
    v = P["vol14"]
    return np.nan_to_num(np.minimum(4.0, 0.02 / np.where(v > 0, v, np.inf)), nan=0.0)


# ------------------------------------------------------------------ 4.9 fuerza
def fuerza(P: dict, d: int, k: int, lado: int) -> float:
    p, s = P["C"][d, k], P["sigma"][d, k]
    return (p / P["tope"][d] - 1) / s - 1 if lado > 0 else (1 - p / P["suelo"][d]) / s - 1


# ------------------------------------------------------------------ 4.10 puntuación
def rasgos_controles(P: dict) -> dict:
    """Los rasgos sin orientar en los 12 controles de cada día (matrices días × 12)."""
    H, L, C, V = P["H"], P["L"], P["C"], P["V"]
    D, K = len(P["fechas"]), np.array(CONTROLES)
    apertura, cierre = P["apertura"], P["cierre"]
    previo = np.r_[np.nan, cierre[:-1]]

    def antes(x, n, f="mean"):  # estadístico de las n sesiones anteriores (columna a columna)
        return getattr(pd.DataFrame(np.asarray(x)).rolling(n), f)().shift(1).to_numpy()

    def cada_control(x):
        return np.repeat(np.asarray(x, dtype=float).reshape(-1, 1), len(K), 1)

    sig, px = P["sigma"][:, K], C[:, K]
    R = {"hora": np.tile(np.arange(len(K), dtype=float), (D, 1)),
         "fuerza_l": (px / P["tope"][:, None] - 1) / sig - 1, "fuerza_c": (1 - px / P["suelo"][:, None]) / sig - 1,
         "mov": (px / apertura[:, None] - 1) / sig,
         "hueco": cada_control((apertura / previo - 1) / P["sigma"][:, -1])}
    R["hueco_abs"] = np.abs(R["hueco"])
    mov_dia = np.abs(cierre / apertura - 1)
    R["sigma_rel"] = cada_control(antes(mov_dia, 14)[:, 0] / antes(mov_dia, 60)[:, 0])
    r = pd.Series(cierre).pct_change().to_numpy()
    for n in (5, 14, 60):
        R[f"vol{n}"] = cada_control(antes(r, n, "std")[:, 0] * np.sqrt(252))
    R["vol5_60"] = R["vol5"] / R["vol60"]
    maximo = pd.Series(cierre).rolling(252, min_periods=60).max().shift(1).to_numpy()
    R["dd252"] = cada_control(previo / maximo - 1)
    for n in (50, 200):
        R[f"tend{n}"] = cada_control(previo / pd.Series(cierre).rolling(n).mean().shift(1).to_numpy() - 1)
    R["ret1"] = cada_control(np.r_[np.nan, r[:-1]])
    R["ret1_dia"] = cada_control(np.r_[np.nan, (cierre / apertura - 1)[:-1]])
    acum = np.cumsum(V, 1)
    R["vrel"] = acum[:, K] / antes(acum[:, K], 14)
    v30 = acum[:, K] - np.c_[np.zeros(D), acum[:, K[1:] - 30]]   # las 30 velas que acaban en el control
    R["vrel30"] = v30 / antes(v30, 14)
    R["vwap"] = (px - P["vwap"][:, K]) / px / sig
    R["rango"] = (np.maximum.accumulate(H, 1)[:, K] - np.minimum.accumulate(L, 1)[:, K]) / apertura[:, None] / sig
    rmin = np.diff(np.log(C), axis=1, prepend=np.log(apertura)[:, None])
    v_min = np.stack([rmin[:, k - 29:k + 1].std(1) for k in K], 1)
    R["volmin30"] = v_min / antes(v_min, 14)
    f = P["fechas"]
    mes = f.to_period("M")
    siguiente = np.r_[mes[1:] != mes[:-1], (f[-1] + pd.offsets.BDay(1)).month != f[-1].month]
    R["dia_semana"] = cada_control(f.dayofweek)
    R["vencimiento"] = cada_control((f.dayofweek == 4) & (f.day >= 15) & (f.day <= 21))
    R["fin_mes"] = cada_control(siguiente)
    R["inicio_mes"] = cada_control(np.r_[True, mes[1:] != mes[:-1]])
    return R


def rasgos_de(R: dict, d: int, c: int, lado: int) -> dict:
    """Los 25 rasgos de una ruptura (día d, control c, lado ±1), orientados al lado."""
    x = {}
    for r in RASGOS:
        if r == "fuerza":
            x[r] = R["fuerza_l" if lado > 0 else "fuerza_c"][d, c]
        elif r == "lado":
            x[r] = float(lado)
        else:
            x[r] = R[r][d, c] * (lado if r in CON_SIGNO else 1)
    return x


class Puntuacion:
    """puntuación = constante + Σ coeficiente × percentil centrado (respecto al entrenamiento)."""

    def __init__(self, carpeta: Path = AQUI / "datos"):
        coef = pd.read_csv(carpeta / "puntuacion_coeficientes.csv").set_index("rasgo")["coeficiente_pb"]
        self.coef = coef[RASGOS].to_numpy(float)
        self.constante = float(coef["(constante)"])
        self.corte = float(coef["(corte: entrar si la puntuación ≥)"])
        ent = pd.read_csv(carpeta / "entrenamiento.csv")
        self.valores = {r: np.sort(ent[r].dropna().to_numpy(float)) for r in RASGOS}

    def percentil(self, r: str, x: float) -> float:
        if np.isnan(x):
            return 0.0
        v = self.valores[r]
        return (np.searchsorted(v, x, "left") + np.searchsorted(v, x, "right")) / 2 / len(v) - 0.5

    def __call__(self, rasgos: dict) -> float:
        return self.constante + float(np.dot([self.percentil(r, rasgos[r]) for r in RASGOS], self.coef))


def reentrenar(carpeta: Path = AQUI / "datos") -> pd.Series:
    """4.10.4 con las operaciones de ``entrenamiento.csv``: ridge sobre percentiles centrados,
    penalización por validación cruzada dejando fuera un año, corte = mediana de las predicciones."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import LeaveOneGroupOut

    ent = pd.read_csv(carpeta / "entrenamiento.csv")
    X = ent[RASGOS].to_numpy(float)
    y = ent.ret_neto_pb.clip(*ent.ret_neto_pb.quantile([0.01, 0.99])).to_numpy()
    orden = [np.sort(c[~np.isnan(c)]) for c in X.T]
    Q = np.zeros_like(X)
    for j, v in enumerate(orden):
        rango = (np.searchsorted(v, X[:, j], "left") + np.searchsorted(v, X[:, j], "right")) / 2
        Q[:, j] = np.where(np.isnan(X[:, j]), 0.0, rango / len(v) - 0.5)
    años = pd.to_datetime(ent.fecha).dt.year
    m = RidgeCV(alphas=np.logspace(-1, 5, 25), cv=list(LeaveOneGroupOut().split(Q, y, años))).fit(Q, y)
    return pd.Series({**dict(zip(RASGOS, m.coef_, strict=True)), "(constante)": m.intercept_,
                      "(corte: entrar si la puntuación ≥)": float(np.median(m.predict(Q))),
                      "(penalización del ridge)": m.alpha_})


# ------------------------------------------------------------------ 4.4-4.6 simulación
def simular(P: dict, version: str = "NB_ART", lados: int = 2, tipo_palanca: str = "1x",
            puntuacion: Puntuacion | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Operaciones y curva de valor (1 al empezar). ``version``: NB_ART, NB_F03 o NB_PUNT."""
    D = len(P["fechas"])
    lev = palanca(P, tipo_palanca)
    R = rasgos_controles(P) if version == "NB_PUNT" else None
    puntuacion = puntuacion if puntuacion is not None or version != "NB_PUNT" else Puntuacion()

    def pasa(d: int, c: int, lado: int) -> bool:
        if version == "NB_ART":
            return True
        if version == "NB_F03":
            return fuerza(P, d, CONTROLES[c], lado) >= 0.3
        return puntuacion(rasgos_de(R, d, c, lado)) >= puntuacion.corte

    caja, valor, ops = 1.0, np.ones(D), []
    for d in range(D):
        valor[d] = caja
        if np.isnan(P["UB"][d, 0]):           # aún no hay 14 sesiones para el cono
            continue
        ult, coste = P["ultima"][d], P["coste"][d]
        lado, unidades, pendiente = 0, 0.0, None
        for k in range(ult + 1):
            # 1) lo decidido en el control anterior, a la apertura de esta vela
            if pendiente is not None:
                px = P["O"][d, k]
                if lado != 0:                     # cerrar (también al girar)
                    caja += unidades * px - coste * abs(unidades * px)
                    ops[-1] |= {"control_salida": hora(k), "precio_salida": px}
                    lado, unidades = 0, 0.0
                if pendiente != 0:                # entrar o girar
                    unidades = pendiente * lev[d] * caja / px
                    caja -= unidades * px + coste * abs(unidades * px)
                    lado = pendiente
                    ops.append({"fecha": P["fechas"][d].date(), "control_entrada": hora(k),
                                "lado": "largo" if lado > 0 else "corto", "precio_entrada": px,
                                "palanca": lev[d], "coste": coste})
                pendiente = None
            # 2) controles de las 10:00 a las 15:30 (se ejecuta en la vela siguiente)
            if k in CONTROLES and k + 1 <= ult:
                c, p = CONTROLES.index(k), P["C"][d, k]
                UB, LB, VW = P["UB"][d, k], P["LB"][d, k], P["vwap"][d, k]
                if lado == 0:
                    if p > UB and pasa(d, c, +1):
                        pendiente = +1
                    elif lados == 2 and p < LB and pasa(d, c, -1):
                        pendiente = -1
                elif lado > 0 and p < max(UB, VW):      # stop móvil del largo
                    pendiente = -1 if lados == 2 and p < LB and pasa(d, c, -1) else 0
                elif lado < 0 and p > min(LB, VW):      # stop móvil del corto
                    pendiente = +1 if p > UB and pasa(d, c, +1) else 0
        # 3) al cierre, se cierra todo
        if lado != 0:
            px = P["C"][d, ult]
            caja += unidades * px - coste * abs(unidades * px)
            ops[-1] |= {"control_salida": "16:00", "precio_salida": px}
        valor[d] = caja
    ops = pd.DataFrame(ops)
    if len(ops):
        signo = np.where(ops.lado == "largo", 1, -1)
        ops["ret_neto"] = signo * (ops.precio_salida / ops.precio_entrada - 1) - 2 * ops.coste
        ops = ops.drop(columns="coste")
    return ops, pd.Series(valor, P["fechas"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("velas", nargs="?", type=Path)
    ap.add_argument("--version", default="NB_ART", choices=["NB_ART", "NB_F03", "NB_PUNT"])
    ap.add_argument("--solo-largo", action="store_true")
    ap.add_argument("--palanca", default="1x", choices=["1x", "vol"])
    ap.add_argument("--reentrenar", action="store_true")
    a = ap.parse_args()
    if a.reentrenar:
        print(reentrenar().round(4).to_string())
        return
    P = preparar(sesiones(pd.read_csv(a.velas)))
    ops, valor = simular(P, a.version, 1 if a.solo_largo else 2, a.palanca)
    ops.to_csv("operaciones_referencia.csv", index=False)
    anual = valor.resample("YE").last()
    print((anual / np.r_[1.0, anual.to_numpy()[:-1]] - 1).round(4).to_string())
    print(f"{len(ops)} operaciones → operaciones_referencia.csv")


if __name__ == "__main__":
    main()
