"""Noise Band · ronda 2: las versiones elegidas, a prueba.

Tres versiones de Noise Band (largo y corto, salidas del artículo, costes de SPY a mercado),
siempre fuera de muestra:

- **artículo**: la base;
- **fuerza**: solo se entra si el precio supera la banda en al menos 0,3 sigmas (la mediana
  de las rupturas: 0,30 en la búsqueda y 0,31 en la validación). Es un cono de entrada un
  30 % más ancho con la salida del artículo;
- **puntuación**: el modelo lineal con todos los rasgos de ``modelos.py`` (entrar solo si la
  puntuación de la ruptura supera su mediana), operado en dos mitades (cada una con el modelo
  entrenado en la otra) o año a año (cada año con el modelo entrenado con todos los
  anteriores; desde 2008).

Pruebas:

1. **ancho del cono de entrada**: de 1 a 2 sigmas (fuerza mínima de 0 a 1), en la búsqueda y
   en la validación;
2. **robustez**: sigma de 10 y 20 sesiones, costes del artículo y de 1 pb, corte del modelo
   en el 40 % y el 60 % y penalización del ridge ×10 y ÷10;
3. **Sharpe deflactado** con todas las pruebas de las rondas 1 y 2 (N contado y N efectivo:
   grupos de pruebas con correlación > 0,9, la regla del repo);
4. **encima del nivel 1 de caídas**, como en ``combinar.py``;
5. la **puntuación para operar**: el modelo entrenado con toda la historia (coeficientes y
   tabla de percentiles).

Salida: resultados/{ronda2_versiones.csv, ronda2_versiones_anual.csv, ronda2_cono.csv,
ronda2_robustez.csv, ronda2_dsr.csv, ronda2_combinar.csv, ronda2_combinar_crisis.csv,
ronda2_combinar_anual.csv, puntuacion_coeficientes.csv, puntuacion_percentiles.csv,
ronda2_curvas.png, ronda2_cono.png, ronda2_combinar.png}

Uso: python research/noise_band/elegida.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneGroupOut

import combinar as cb
import estadistica as es
import eventos as ev
import graficas as gr
import modelos as mo
import motor as nb
import rasgos as rg
import ronda1 as r1
import ronda2 as r2
from earnings_alpha.stats.performance import deflated_sharpe_ratio

OUT = Path(__file__).parent / "resultados"
COSTES = dict(coste_por_accion=0.0085)
UMBRAL = 0.3  # fuerza mínima: sigmas por encima de la banda
TAMAÑOS = {"1x": None, "vol 2 % (máx. 4×)": 0.02}
PRINCIPALES = ["artículo · 1x", "fuerza ≥ 0,3 · 1x", "puntuación, dos mitades · 1x",
               "puntuación, dos mitades (solo largo) · 1x", "artículo · vol 2 % (máx. 4×)",
               "puntuación, dos mitades · vol 2 % (máx. 4×)"]
VENTANAS = {"total": ("2005-01-24", None), "validacion": mo.VALIDACION, "busqueda": mo.BUSQUEDA,
            "busqueda_1": ("2013-01-01", "2017-01-01"), "busqueda_2": ("2017-01-01", None),
            "desde_2008": ("2008-01-01", None)}


def filtro_fuerza(d: dict, umbral: float = UMBRAL) -> tuple[np.ndarray, np.ndarray]:
    """Días × 390: 1 si el precio supera la banda en al menos ``umbral`` sigmas (cada lado)."""
    with np.errstate(invalid="ignore"):
        fl = (d["C"] / d["ref_arriba"][:, None] - 1) / d["sigma"] - 1
        fc = (1 - d["C"] / d["ref_abajo"][:, None]) / d["sigma"] - 1
    return (fl >= umbral).astype(float), (fc >= umbral).astype(float)


def versiones(d: dict, costes: dict = COSTES, tamaños=("1x",), esquemas=("dos mitades",), lados=(2,),
              **kw) -> dict:
    """Las tres versiones, fuera de muestra (``lados``: 2 largo y corto, 1 solo largo, con el mismo
    modelo). ``kw`` va al modelo."""
    R = rg.construir(d)
    y = ev.tabla(d, R, **costes)
    fl, fc = filtro_fuerza(d)
    partes = {"dos mitades": mo.DOS_MITADES, "año a año": mo.AÑO_A_AÑO}
    mults = {f"puntuación, {e}": mo.cruzado(d, y, R, "lineal", mo.RASGOS, "filtro", partes[e], **kw)
             for e in esquemas}
    out = {}
    for ld, tam in ((ld, tam) for ld in lados for tam in tamaños):
        lev, sufijo = nb.palanca(d, TAMAÑOS[tam]), "" if ld == 2 else " (solo largo)"
        sim = lambda ml=None, mc=None, ld=ld, lev=lev: nb.simular(  # noqa: E731
            d, lados=ld, lev=lev, mult_largo=ml, mult_corto=mc, **costes)
        out[(f"artículo{sufijo}", tam)] = sim()
        out[(f"fuerza ≥ {UMBRAL:.1f}{sufijo}".replace(".", ","), tam)] = sim(fl, fc)
        for nombre, (ml, mc) in mults.items():
            out[(f"{nombre}{sufijo}", tam)] = sim(ml, mc)
    return out


def fila_metricas(r: pd.Series, ventanas: dict, ops: pd.DataFrame | None = None) -> dict:
    fila = {}
    for vn, (a, b) in ventanas.items():
        x = es.tramo(r, a, b)
        m = nb.metricas((1 + x).cumprod())
        fila |= {f"{vn}_cagr": m["cagr"], f"{vn}_peor_caida": m["peor_caida"], f"{vn}_sharpe": m["sharpe"]}
        if ops is not None:
            o = ops[(ops.entrada >= pd.Timestamp(a)) & ((ops.entrada < pd.Timestamp(b)) if b else True)]
            fila |= {f"{vn}_operaciones_año": len(o) / (len(x) / 252), f"{vn}_ret_medio_pb": o.ret.mean() * 1e4}
    return fila


def cono(d: dict) -> pd.DataFrame:
    filas = []
    for u in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0):
        ml, mc = filtro_fuerza(d, u) if u > 0 else (None, None)
        r = nb.simular(d, lados=2, mult_largo=ml, mult_corto=mc, **COSTES)
        ret = r["equity"].pct_change().fillna(0.0)
        filas.append({"fuerza_minima": u, "ancho_cono_entrada": 1 + u,
                      **fila_metricas(ret, {k: VENTANAS[k] for k in ("total", "busqueda", "validacion")},
                                      r["operaciones"])})
    return pd.DataFrame(filas)


def robustez(base: dict) -> pd.DataFrame:
    casos = {"base": (base, COSTES, {}), "sigma de 10 sesiones": (nb.cargar(n_sigma=10), COSTES, {}),
             "sigma de 20 sesiones": (nb.cargar(n_sigma=20), COSTES, {}),
             "costes del artículo (0,0045 $ por acción)": (base, dict(coste_por_accion=0.0045), {}),
             "costes de 1 pb por lado": (base, dict(coste_por_accion=None, coste_bps=1.0), {}),
             "corte del modelo en el 40 % (entra más)": (base, COSTES, dict(cuantil=0.4)),
             "corte del modelo en el 60 % (entra menos)": (base, COSTES, dict(cuantil=0.6)),
             "penalización del ridge ×10": (base, COSTES, dict(escala=10.0)),
             "penalización del ridge ÷10": (base, COSTES, dict(escala=0.1))}
    filas = []
    for caso, (d, costes, kw) in casos.items():
        for (nombre, tam), r in versiones(d, costes, **kw).items():
            if kw and not nombre.startswith("puntuación"):
                continue
            ret = r["equity"].pct_change().fillna(0.0)
            filas.append({"prueba": caso, "version": nombre, "tamaño": tam,
                          **fila_metricas(ret, {k: VENTANAS[k] for k in ("total", "busqueda", "validacion")})})
    return pd.DataFrame(filas)


def todas_las_pruebas(d: dict, extra: dict[str, pd.Series]) -> tuple[pd.DataFrame, dict]:
    """Series diarias de todas las pruebas de las rondas 1 y 2 (sin repetidas) y la ventana en
    que se juzgó cada una."""
    cols, ventanas = {}, {}
    for nombre, kw in r1.VARIANTES.items():
        for tam, obj in TAMAÑOS.items():
            cols[f"ronda 1 · {nombre} · {tam}"] = nb.simular(d, lev=nb.palanca(d, obj), **kw, **COSTES)[
                "equity"].pct_change().fillna(0.0)
            ventanas[f"ronda 1 · {nombre} · {tam}"] = VENTANAS["total"]
    z = np.load(r2.SERIES, allow_pickle=True)
    for i, n in enumerate(z["nombres"]):
        cols[f"ronda 2 · {n}"] = pd.Series(z["r"][:, i], d["fechas"])
        ventanas[f"ronda 2 · {n}"] = VENTANAS["total"]
    z = np.load(mo.SERIES, allow_pickle=True)
    for i, n in enumerate(z["nombres"]):
        cols[f"modelos · {n}"] = pd.Series(z["r"][:, i], d["fechas"])
        ventanas[f"modelos · {n}"] = (str(z["desde"][i]), str(z["hasta"][i]) or None)
    for n, r in extra.items():
        cols[n], ventanas[n] = r, VENTANAS["total"]
    df = pd.DataFrame(cols)
    x = es.tramo(df, *VENTANAS["total"]).to_numpy()
    repetida = np.zeros(df.shape[1], dtype=bool)
    for j in range(df.shape[1]):  # la misma regla con dos nombres cuenta una vez
        repetida[j] = any(np.abs(x[:, j] - x[:, i]).max() < 1e-12 for i in range(j) if not repetida[i])
    return df.loc[:, ~repetida], ventanas


def dsr(df: pd.DataFrame, ventanas: dict, candidatas: dict[str, tuple[pd.Series, tuple]]) -> pd.DataFrame:
    """Sharpe deflactado con N = todas las pruebas y con N efectivo (grupos de pruebas con
    correlación > 0,9 y la varianza de los Sharpe medios de cada grupo, como en caídas)."""
    sh = np.array([es.tramo(df[c], *ventanas[c]).mean() / es.tramo(df[c], *ventanas[c]).std() for c in df])
    n = len(sh)
    grupo = es.grupos_correlados(es.tramo(df, *VENTANAS["total"]).to_numpy())
    n_ef = int(grupo.max()) + 1
    var_ef = float(np.var([sh[grupo == k].mean() for k in range(n_ef)], ddof=1))
    filas = []
    for nombre, (r, v) in candidatas.items():
        x = es.tramo(r, *v).to_numpy()
        a = deflated_sharpe_ratio(x, trial_sharpes=sh, n_trials=n)
        b = deflated_sharpe_ratio(x, sr_variance=var_ef, n_trials=n_ef)
        filas.append({"version": nombre, "desde": v[0], "sharpe": a.sharpe_annualized, "psr_0": a.psr_zero,
                      "pruebas": n, "sharpe_max_esperado": a.sr_star_annualized, "dsr": a.dsr,
                      "pruebas_efectivas": n_ef, "sharpe_max_esperado_ef": b.sr_star_annualized, "dsr_ef": b.dsr})
    return pd.DataFrame(filas)


def combinar_n1(series: dict[str, pd.Series], fin) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    n1 = pd.read_csv(cb.DIARIO, index_col=0, parse_dates=True)["CAI_N1"]
    comun = n1.index[(n1.index >= "2005-01-24") & (n1.index <= fin)]
    n1 = n1[comun]
    filas, mezclas = [], {"nivel 1 solo": n1}
    for nombre, r in series.items():
        r = r.reindex(comun).fillna(0.0)
        for peso in (0.5, 1.0):
            m = n1 + peso * r
            fila = {"complemento": nombre, "peso": peso,
                    "corr_mensual_con_N1": (1 + r).resample("ME").prod().corr((1 + n1).resample("ME").prod())}
            for vn in ("total", "desde_2008"):
                if vn == "total" and "año a año" in nombre:
                    continue
                fila |= {f"{vn}_{k}": v for k, v in cb.metricas(es.tramo(m, *VENTANAS[vn])).items()}
            filas.append(fila)
            mezclas[f"nivel 1 + {nombre} (peso {f'{peso:g}'.replace('.', ',')})"] = m
    solo = {"complemento": "ninguno", "peso": 0.0}
    for vn in ("total", "desde_2008"):
        solo |= {f"{vn}_{k}": v for k, v in cb.metricas(es.tramo(n1, *VENTANAS[vn])).items()}
    df = pd.DataFrame([solo, *filas])
    principales = ["nivel 1 solo"] + [f"nivel 1 + {k} (peso 1)" for k in PRINCIPALES]
    eqs = {k: (1 + mezclas[k]).cumprod() for k in principales}
    crisis = pd.DataFrame({k: [e.asof(pd.Timestamp(b)) / e.asof(pd.Timestamp(a)) - 1 for _, a, b in cb.CRISIS]
                           for k, e in eqs.items()}, index=[n for n, _, _ in cb.CRISIS])
    anual = pd.DataFrame({k: (1 + mezclas[k]).resample("YE").prod() - 1 for k in principales})
    anual.index = anual.index.year
    return df, crisis, anual, mezclas


def puntuacion_final(y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """El modelo lineal entrenado con todas las operaciones (2005 → may-2020), para operar."""
    t = y[ev.en(y, "2005-01-24", None)]
    X = t[mo.RASGOS].to_numpy(np.float64)
    obj = (t.ret * 1e4).clip(*np.quantile(t.ret * 1e4, [0.01, 0.99])).to_numpy()
    P = mo.Percentiles(X)
    cv = list(LeaveOneGroupOut().split(X, obj, t.fecha.dt.year))
    m = RidgeCV(alphas=np.logspace(-1, 5, 25), cv=cv).fit(P(X), obj)
    corte = float(np.median(m.predict(P(X))))
    coef = pd.DataFrame({"rasgo": mo.RASGOS, "con_signo_del_lado": [r in rg.CON_SIGNO for r in mo.RASGOS],
                         "coeficiente_pb": m.coef_})
    coef = pd.concat([coef, pd.DataFrame([{"rasgo": "(constante)", "coeficiente_pb": m.intercept_},
                                          {"rasgo": "(corte: entrar si la puntuación ≥)", "coeficiente_pb": corte},
                                          {"rasgo": "(penalización del ridge)", "coeficiente_pb": m.alpha_}])])
    q = np.linspace(0, 1, 101)
    pct = pd.DataFrame({r: np.nanquantile(X[:, j], q) for j, r in enumerate(mo.RASGOS)}, index=np.round(q * 100))
    pct.index.name = "percentil"
    return coef, pct


def grafica_curvas(series: dict, ruta: Path) -> None:
    gr.estilo()
    colores = {"artículo": (gr.GRIS, 1.6), "fuerza ≥ 0,3": (gr.AZUL, 1.8), "puntuación": (gr.NARANJA, 2.0)}
    clave = {"artículo": "artículo", "fuerza ≥ 0,3": "fuerza ≥ 0,3", "puntuación": "puntuación, dos mitades"}
    fig, ejes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=False)
    for ax, tam in zip(ejes, TAMAÑOS, strict=True):
        s = {k: ((1 + es.tramo(series[(clave[k], tam)], "2005-01-24", None)).cumprod(), *c)
             for k, c in colores.items()}
        gr.curvas(ax, s)
        ax.legend(handles=[plt.Line2D([], [], color=c, lw=w) for c, w in colores.values()], labels=list(colores),
                  loc="upper left", frameon=False, fontsize=8.6)
        ax.set_title("sin palanca (1×)" if tam == "1x" else "volatilidad objetivo 2 % (máx. 4×)", fontsize=10.5)
    fig.suptitle("Noise Band fuera de muestra (cada mitad con el modelo de la otra): 100.000 $ en 2005, costes de "
                 "SPY a mercado", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.subplots_adjust(right=0.9, wspace=0.36)
    fig.savefig(ruta, dpi=130)
    plt.close(fig)


def grafica_cono(c: pd.DataFrame, ruta: Path) -> None:
    gr.estilo()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
    for vn, color, etiqueta in (("busqueda", gr.AZUL, "búsqueda 2013-2020"),
                                ("validacion", gr.NARANJA, "validación 2005-2012")):
        a1.plot(c.ancho_cono_entrada, c[f"{vn}_sharpe"], color=color, lw=2, marker="o", ms=4, label=etiqueta)
        a2.plot(c.ancho_cono_entrada, c[f"{vn}_ret_medio_pb"], color=color, lw=2, marker="o", ms=4, label=etiqueta)
    for ax, t in ((a1, "Sharpe (sin palanca)"), (a2, "retorno neto medio por operación (pb)")):
        ax.axvline(1 + UMBRAL, color=gr.EJE, lw=1, ls="--")
        ax.set_xlabel("ancho del cono de entrada (× sigma); la salida no cambia")
        ax.set_title(t, fontsize=10.5)
        ax.grid(axis="y")
        ax.legend(frameon=False, fontsize=8.6)
    a1.annotate("elegido: 1,3", xy=(1 + UMBRAL, a1.get_ylim()[0]), xytext=(4, 6), textcoords="offset points",
                fontsize=8.6, color=gr.TINTA2)
    fig.suptitle("Noise Band: exigir que la ruptura sea más clara", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(ruta, dpi=130)
    plt.close(fig)


def grafica_combinar(mezclas: dict, ruta: Path) -> None:
    gr.estilo()
    elegidas = {"nivel 1 solo": (gr.AZUL, 2.0), "nivel 1 + artículo · 1x (peso 1)": (gr.GRIS, 1.6),
                "nivel 1 + puntuación, dos mitades · 1x (peso 1)": (gr.NARANJA, 2.0),
                "nivel 1 + puntuación, dos mitades · vol 2 % (máx. 4×) (peso 1)": (gr.VIOLETA, 1.8)}
    eqs = {k: (1 + mezclas[k]).cumprod() for k in elegidas}
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11.5, 7.6), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    gr.curvas(a1, {k.replace(" (peso 1)", ""): (e, *elegidas[k]) for k, e in eqs.items()})
    a1.legend(handles=[plt.Line2D([], [], color=c, lw=w) for c, w in elegidas.values()],
              labels=[k.replace(" (peso 1)", "") for k in elegidas], loc="upper left", frameon=False, fontsize=8.6)
    for k, e in eqs.items():
        a2.plot(e.index, (e / e.cummax() - 1) * 100, color=elegidas[k][0], lw=1.2)
    a1.set_title("Nivel 1 de caídas con Noise Band intradía encima: 100.000 $ en 2005, costes de SPY a mercado")
    a2.set_ylabel("caída desde máximos (%)")
    a2.grid(axis="y")
    fig.tight_layout()
    fig.subplots_adjust(right=0.74)
    fig.savefig(ruta, dpi=130)
    plt.close(fig)


def main() -> None:
    d = nb.cargar()
    v = versiones(d, tamaños=tuple(TAMAÑOS), esquemas=("dos mitades", "año a año"), lados=(2, 1))
    series = {k: r["equity"].pct_change().fillna(0.0) for k, r in v.items()}
    filas = []
    for (nombre, tam), r in v.items():
        ref = series[("artículo (solo largo)" if "solo largo" in nombre else "artículo", tam)]
        fila = {"version": nombre, "tamaño": tam, **fila_metricas(series[(nombre, tam)], VENTANAS, r["operaciones"])}
        for vn in ("total", "validacion", "busqueda", "desde_2008"):
            x, xb = es.tramo(series[(nombre, tam)], *VENTANAS[vn]), es.tramo(ref, *VENTANAS[vn])
            fila[f"{vn}_t_sharpe"] = np.nan if nombre.startswith("artículo") else es.t_sharpe(x.to_numpy(),
                                                                                                xb.to_numpy())
        fila["tiempo_en_mercado"] = float(es.tramo(r["en_mercado"], *VENTANAS["total"]).mean())
        filas.append(fila)
    ver = pd.DataFrame(filas)
    ver.to_csv(OUT / "ronda2_versiones.csv", index=False, float_format="%.5f")
    anual = pd.DataFrame({f"{n} · {t}": (1 + r).resample("YE").prod() - 1 for (n, t), r in series.items()})
    anual.index = anual.index.year
    anual.to_csv(OUT / "ronda2_versiones_anual.csv", float_format="%.4f")

    c = cono(d)
    c.to_csv(OUT / "ronda2_cono.csv", index=False, float_format="%.5f")
    rob = robustez(d)
    rob.to_csv(OUT / "ronda2_robustez.csv", index=False, float_format="%.5f")

    extra = {f"elegida · cono {u:g}": nb.simular(d, lados=2, mult_largo=ml, mult_corto=mc, **COSTES)[
        "equity"].pct_change().fillna(0.0) for u in c.fuerza_minima if u > 0 for ml, mc in [filtro_fuerza(d, u)]}
    pr, vent = todas_las_pruebas(d, extra)
    cand = {f"{n} · {t}": (r, VENTANAS["desde_2008" if "año a año" in n else "total"]) for (n, t), r in series.items()}
    ds = dsr(pr, vent, cand)
    ds.to_csv(OUT / "ronda2_dsr.csv", index=False, float_format="%.5f")

    nb_series = {f"{n} · {t}": r for (n, t), r in series.items()}
    comb, crisis, an_comb, mezclas = combinar_n1(nb_series, d["fechas"][-1])
    comb.to_csv(OUT / "ronda2_combinar.csv", index=False, float_format="%.5f")
    crisis.to_csv(OUT / "ronda2_combinar_crisis.csv", float_format="%.4f")
    an_comb.to_csv(OUT / "ronda2_combinar_anual.csv", float_format="%.4f")

    coef, pct = puntuacion_final(ev.tabla(d))
    coef.to_csv(OUT / "puntuacion_coeficientes.csv", index=False, float_format="%.5f")
    pct.to_csv(OUT / "puntuacion_percentiles.csv", float_format="%.6f")

    grafica_curvas(series, OUT / "ronda2_curvas.png")
    grafica_cono(c, OUT / "ronda2_cono.png")
    grafica_combinar(mezclas, OUT / "ronda2_combinar.png")

    pd.set_option("display.width", 260)
    pd.set_option("display.max_colwidth", 60)
    print(ver.round(3).T.to_string())
    print(c.round(3).to_string(index=False))
    print(rob[["prueba", "version", "total_cagr", "total_sharpe", "busqueda_sharpe", "validacion_sharpe",
               "total_peor_caida"]].round(3).to_string(index=False))
    print(f"\n{pr.shape[1]} pruebas distintas")
    print(ds.round(3).to_string(index=False))
    print(comb.round(3).to_string(index=False))
    print((crisis * 100).round(1).to_string())
    print((an_comb * 100).round(1).to_string())
    print(coef.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
