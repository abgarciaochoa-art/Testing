"""Noise Band · ronda 2: ¿ayuda meter todos los indicadores a la vez?

Con las operaciones de la estrategia del artículo (largo y corto, sin palanca, costes de SPY a
mercado) de una ventana, un modelo aprende qué retorno neto esperar de cada ruptura según los
rasgos de ``rasgos.py`` en el control en que se decide: fuerza de la ruptura, hora, hueco de
la noche, volatilidad de los 5, 14 y 60 días previos, caída desde máximos, tendencia,
rentabilidad del día anterior, volumen relativo del día y de la última media hora, distancia
a la VWAP, rango del día, volatilidad de los minutos, calendario y lado. En la otra ventana,
que no ha visto, se usa para **filtrar** (entrar solo si la predicción supera su mediana en el
entrenamiento) o para el **tamaño** (1,5× por encima de la mediana, 0,5× por debajo). Fijado
de antemano:

- **lineal**: ridge sobre los rasgos en percentiles (del entrenamiento), con la penalización
  elegida por validación cruzada dejando fuera un año cada vez;
- **árboles**: gradient boosting poco profundo (3 niveles, 150 árboles, ≥ 80 operaciones por
  hoja), que puede captar interacciones y efectos en U (como el del volumen);
- los dos **sin la fuerza**, para ver si el resto de indicadores aporta algo por sí solo;
- la referencia de **un solo rasgo**: la fuerza, con el corte en su mediana del entrenamiento.

Se entrena en la búsqueda (2013 → may-2020) y se mide en la validación (2005-2012), y al
revés. El retorno que se aprende se recorta al 1 % y al 99 % del entrenamiento.

Con eso se montan dos curvas **fuera de muestra** de toda la historia:

- **dos mitades**: cada mitad se opera con el modelo entrenado en la otra (2005 → 2020);
- **año a año**: cada año se opera con el modelo entrenado con todos los años anteriores
  (desde 2005; la primera prueba es 2008), como se usaría en real.

Salida: resultados/{ronda2_modelos.csv, ronda2_modelos_rasgos.csv, ronda2_cruzado.csv,
ronda2_cruzado_anual.csv} y las series de cada prueba para el Sharpe deflactado
(data/cache/noise_band/pruebas_modelos.npz).

Uso: python research/noise_band/modelos.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import LeaveOneGroupOut

import estadistica as es
import eventos as ev
import motor as nb
import rasgos as rg

OUT = Path(__file__).parent / "resultados"
SERIES = nb.DATOS.with_name("pruebas_modelos.npz")
COSTES = dict(coste_por_accion=0.0085)
BUSQUEDA, VALIDACION = ("2013-01-01", None), ("2005-01-24", "2013-01-01")
PLIEGUES = {"búsqueda → validación": (BUSQUEDA, VALIDACION), "validación → búsqueda": (VALIDACION, BUSQUEDA)}
RASGOS = [r for r in ev.RASGOS if r != "n_previas"]  # depende del camino del día: no está en las matrices
SIN_FUERZA = [r for r in RASGOS if r != "fuerza"]


def matriz(R: dict, lado: int, nombres: list[str]) -> np.ndarray:
    """Rasgos de cada control orientados a ``lado``, como en ``eventos.tabla`` → (días · 12) × rasgos."""
    cols = []
    for n in nombres:
        if n == "fuerza":
            x = R["fuerza_l"] if lado > 0 else R["fuerza_c"]
        elif n == "hueco_abs":
            x = np.abs(R["hueco"])
        elif n == "lado":
            x = np.full(R["hora"].shape, float(lado))
        else:
            x = R[n] * lado if n in rg.CON_SIGNO else R[n]
        cols.append(x.reshape(-1))
    return np.column_stack(cols)


class Percentiles:
    """Lleva cada rasgo a su percentil (rango medio) en el entrenamiento, centrado; sin dato → 0."""

    def __init__(self, X: np.ndarray):
        self.orden = [np.sort(c[~np.isnan(c)]) for c in X.T]

    def __call__(self, X: np.ndarray) -> np.ndarray:
        out = np.zeros_like(X, dtype=float)
        for j, s in enumerate(self.orden):
            if len(s):
                rango = (np.searchsorted(s, X[:, j], "left") + np.searchsorted(s, X[:, j], "right")) / 2
                out[:, j] = np.where(np.isnan(X[:, j]), 0.0, rango / len(s) - 0.5)
        return out


def entrenar(y: pd.DataFrame, R: dict, ventana: tuple, tipo: str, nombres: list[str], cuantil: float = 0.5,
             escala: float = 1.0) -> dict:
    """Ajusta el modelo con las operaciones de ``ventana`` y predice cada control de cada día.
    ``cuantil``: corte de la predicción (0,5 = la mediana de las operaciones del entrenamiento);
    ``escala``: multiplica la penalización que elige la validación cruzada del ridge."""
    t = y[ev.en(y, *ventana)]
    X = t[nombres].to_numpy(np.float64)
    obj = (t.ret * 1e4).clip(*np.quantile(t.ret * 1e4, [0.01, 0.99])).to_numpy()
    if tipo == "fuerza":
        pred = lambda Z: Z[:, 0]  # noqa: E731
        modelo = None
    elif tipo == "lineal":
        P = Percentiles(X)
        cv = list(LeaveOneGroupOut().split(X, obj, t.fecha.dt.year))
        modelo = RidgeCV(alphas=np.logspace(-1, 5, 25), cv=cv).fit(P(X), obj)
        if escala != 1.0:
            modelo = Ridge(alpha=modelo.alpha_ * escala).fit(P(X), obj)
        pred = lambda Z: modelo.predict(P(Z))  # noqa: E731
    else:
        modelo = HistGradientBoostingRegressor(max_depth=3, max_iter=150, learning_rate=0.03, min_samples_leaf=80,
                                               l2_regularization=1.0, random_state=0).fit(X, obj)
        pred = modelo.predict
    D = R["hora"].shape
    return {"corte": float(np.quantile(pred(X), cuantil)), "largo": pred(matriz(R, +1, nombres)).reshape(D),
            "corto": pred(matriz(R, -1, nombres)).reshape(D), "modelo": modelo, "pred": pred, "nombres": nombres}


def multiplicador(e: dict, lado: int, regla: str) -> np.ndarray:
    """Días × 12: ``filtro`` (1 si la predicción supera la mediana, si no 0) o ``tamaño`` (1,5 / 0,5)."""
    p = e["largo"] if lado > 0 else e["corto"]
    arriba = np.nan_to_num(p, nan=-np.inf) >= e["corte"]
    return arriba.astype(float) if regla == "filtro" else np.where(arriba, 1.5, 0.5)


MODELOS = {"un rasgo: fuerza": ("fuerza", ["fuerza"]), "lineal, todos los rasgos": ("lineal", RASGOS),
           "árboles, todos los rasgos": ("árboles", RASGOS), "lineal, sin la fuerza": ("lineal", SIN_FUERZA),
           "árboles, sin la fuerza": ("árboles", SIN_FUERZA)}
DOS_MITADES = [(BUSQUEDA, VALIDACION), (VALIDACION, BUSQUEDA)]
AÑO_A_AÑO = [(("2005-01-24", f"{a}-01-01"), (f"{a}-01-01", f"{a + 1}-01-01")) for a in range(2008, 2021)]


def cruzado(d: dict, y: pd.DataFrame, R: dict, tipo: str, cols: list[str], regla: str,
            partes: list[tuple], **kw) -> tuple[np.ndarray, np.ndarray]:
    """Multiplicadores (días × 390) en los que cada tramo de ``partes`` usa el modelo entrenado
    con su ventana de entrenamiento; fuera de los tramos no se opera."""
    D, f = len(d["fechas"]), d["fechas"]
    m = {+1: np.zeros((D, len(rg.KS))), -1: np.zeros((D, len(rg.KS)))}
    for entreno, (a, b) in partes:
        e = entrenar(y, R, entreno, tipo, cols, **kw)
        dias = (f >= pd.Timestamp(a)) & ((f < pd.Timestamp(b)) if b else True)
        for lado in (+1, -1):
            m[lado][dias] = multiplicador(e, lado, regla)[dias]
    ml, mc = np.ones((D, nb.VELAS)), np.ones((D, nb.VELAS))
    ml[:, rg.KS], mc[:, rg.KS] = m[+1], m[-1]
    return ml, mc


def curvas_fuera_de_muestra(d: dict, y: pd.DataFrame, R: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Base, un rasgo y los modelos con todos los rasgos, operados siempre fuera de muestra."""
    tamaños = {"1x": nb.palanca(d), "vol 2 % (máx. 4×)": nb.palanca(d, 0.02)}
    ventanas = {"total": ("2005-01-24", None), "validacion": VALIDACION, "busqueda": BUSQUEDA,
                "desde_2008": ("2008-01-01", None)}
    series, filas = {}, []
    for tam, lev in tamaños.items():
        series[("base", "—", tam)] = nb.simular(d, lados=2, lev=lev, **COSTES)["equity"].pct_change().fillna(0.0)
    for nombre in ("un rasgo: fuerza", "lineal, todos los rasgos", "árboles, todos los rasgos"):
        tipo, cols = MODELOS[nombre]
        for regla in ("filtro", "tamaño"):
            for esquema, partes in (("dos mitades", DOS_MITADES), ("año a año", AÑO_A_AÑO)):
                ml, mc = cruzado(d, y, R, tipo, cols, regla, partes)
                for tam, lev in tamaños.items():
                    r = nb.simular(d, lados=2, lev=lev, mult_largo=ml, mult_corto=mc, **COSTES)
                    series[(f"{nombre} · {regla}", esquema, tam)] = r["equity"].pct_change().fillna(0.0)
    for (nombre, esquema, tam), r in series.items():
        rb = series[("base", "—", tam)]
        rf = series.get(("un rasgo: fuerza · filtro", esquema, tam), rb)
        fila = {"regla": nombre, "esquema": esquema, "tamaño": tam}
        for vn, (a, b) in ventanas.items():
            if esquema == "año a año" and vn != "desde_2008":
                continue
            x, xb, xf = es.tramo(r, a, b), es.tramo(rb, a, b), es.tramo(rf, a, b)
            m = nb.metricas((1 + x).cumprod())
            fila |= {f"{vn}_cagr": m["cagr"], f"{vn}_peor_caida": m["peor_caida"], f"{vn}_sharpe": m["sharpe"],
                     f"{vn}_t_sharpe_base": es.t_sharpe(x.to_numpy(), xb.to_numpy()),
                     f"{vn}_t_sharpe_fuerza": es.t_sharpe(x.to_numpy(), xf.to_numpy())}
        filas.append(fila)
    anual = pd.DataFrame({f"{n} · {e} · {t}": (1 + r).resample("YE").prod() - 1
                          for (n, e, t), r in series.items() if t == "1x"})
    anual.index = anual.index.year
    return pd.DataFrame(filas), anual, series


def main() -> None:
    d = nb.cargar()
    R = dict(np.load(rg.SALIDA))
    y = ev.tabla(d, R)
    base = nb.simular(d, lados=2, **COSTES)
    rb = base["equity"].pct_change().fillna(0.0)
    filas, imp, guardar = [], [], {}
    for pliegue, (entreno, prueba) in PLIEGUES.items():
        for nombre, (tipo, cols) in MODELOS.items():
            e = entrenar(y, R, entreno, tipo, cols)
            for regla in ("filtro", "tamaño"):
                ml, mc = (np.ones((len(d["fechas"]), nb.VELAS)) for _ in range(2))
                ml[:, rg.KS], mc[:, rg.KS] = multiplicador(e, +1, regla), multiplicador(e, -1, regla)
                r = nb.simular(d, lados=2, mult_largo=ml, mult_corto=mc, **COSTES)
                ret = r["equity"].pct_change().fillna(0.0)
                guardar[(f"{pliegue} · {nombre} · {regla}", *prueba)] = ret
                ops = r["operaciones"]
                fila = {"pliegue": pliegue, "modelo": nombre, "regla": regla}
                for vn, (a, b) in (("entrenamiento", entreno), ("prueba", prueba)):
                    x, xb = es.tramo(ret, a, b), es.tramo(rb, a, b)
                    o = ops[(ops.entrada >= pd.Timestamp(a)) & ((ops.entrada < pd.Timestamp(b)) if b else True)]
                    m, mb = nb.metricas((1 + x).cumprod()), nb.metricas((1 + xb).cumprod())
                    fila |= {f"{vn}_sharpe": m["sharpe"], f"{vn}_sharpe_base": mb["sharpe"], f"{vn}_cagr": m["cagr"],
                             f"{vn}_cagr_base": mb["cagr"], f"{vn}_peor_caida": m["peor_caida"],
                             f"{vn}_t_sharpe": es.t_sharpe(x.to_numpy(), xb.to_numpy()),
                             f"{vn}_operaciones": len(o), f"{vn}_ret_medio_pb": o.ret.mean() * 1e4}
                filas.append(fila)
            # qué rasgos usa: coeficientes del lineal (pb del percentil más bajo al más alto) e
            # importancia por permutación de los árboles en la ventana que no ha visto
            if tipo == "lineal":
                imp += [{"pliegue": pliegue, "modelo": nombre, "rasgo": c, "medida": "coeficiente (pb)", "valor": v}
                        for c, v in zip(cols, e["modelo"].coef_, strict=True)]
            elif tipo == "árboles":
                t = y[ev.en(y, *prueba)]
                pi = permutation_importance(e["modelo"], t[cols].to_numpy(np.float64), t.ret.to_numpy() * 1e4,
                                            n_repeats=10, random_state=0, scoring="neg_mean_squared_error")
                imp += [{"pliegue": pliegue, "modelo": nombre, "rasgo": c, "medida": "importancia en la prueba",
                         "valor": v} for c, v in zip(cols, pi.importances_mean, strict=True)]
    df = pd.DataFrame(filas)
    df.to_csv(OUT / "ronda2_modelos.csv", index=False, float_format="%.5f")
    im = pd.DataFrame(imp)
    im.to_csv(OUT / "ronda2_modelos_rasgos.csv", index=False, float_format="%.5f")

    cz, anual, series = curvas_fuera_de_muestra(d, y, R)
    for (nombre, esquema, tam), r in series.items():
        if nombre != "base":
            guardar[(f"{nombre} · {esquema} · {tam}", "2005-01-24" if esquema == "dos mitades" else "2008-01-01",
                     None)] = r
    np.savez(SERIES, nombres=np.array([k[0] for k in guardar]), desde=np.array([k[1] for k in guardar]),
             hasta=np.array([k[2] or "" for k in guardar]), fechas=d["fechas"].to_numpy(),
             r=np.column_stack([v.to_numpy() for v in guardar.values()]))
    cz.to_csv(OUT / "ronda2_cruzado.csv", index=False, float_format="%.5f")
    anual.to_csv(OUT / "ronda2_cruzado_anual.csv", float_format="%.4f")

    pd.set_option("display.width", 250)
    cols = ["pliegue", "modelo", "regla", "entrenamiento_sharpe", "entrenamiento_sharpe_base", "prueba_sharpe",
            "prueba_sharpe_base", "prueba_t_sharpe", "prueba_cagr", "prueba_cagr_base", "prueba_peor_caida",
            "prueba_operaciones", "prueba_ret_medio_pb"]
    print(df[cols].round(3).to_string(index=False))
    for (p, m), g in im.groupby(["pliegue", "modelo"], sort=False):
        g = g.reindex(g.valor.abs().sort_values(ascending=False).index).head(6)
        print(f"\n{p} · {m}: " + ", ".join(f"{r.rasgo} {r.valor:+.2f}" for r in g.itertuples()))
    print("\n— curvas fuera de muestra —")
    print(cz.round(3).to_string(index=False))
    print((anual * 100).round(1).to_string())


if __name__ == "__main__":
    main()
