"""Ronda 5: ORB solo en días de reacción a resultados (acciones realmente "en juego").

Día de evento: la fecha de publicación si es antes de abrir (o sin hora) y el día hábil siguiente
si es después del cierre (o sin hora), y además volumen relativo de los 5 primeros minutos > 2
(conocido a las 9:35). Fechas: data/external/consenso/consenso_master.parquet (hasta mar-2026).
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from acciones import cartera, informe, panel, rel_vol, señales  # noqa: E402


def eventos(p, rv, rv_min=2.0):
    d = pd.read_parquet(ROOT / "data/external/consenso/consenso_master.parquet")
    d["report_date"] = pd.to_datetime(d.report_date)
    fechas = pd.DatetimeIndex(p["fechas"])
    pos = {t: i for i, t in enumerate(p["tickers"])}
    ev = np.zeros(rv.shape, bool)
    for t, fecha, hora in d[["ticker", "report_date", "report_time"]].itertuples(index=False):
        if t not in pos:
            continue
        j = fechas.searchsorted(fecha)
        if j >= len(fechas):
            continue
        cands = []
        if hora != "post-market" and fechas[j] == fecha:
            cands.append(j)
        if hora != "pre-market":
            cands.append(j + 1 if fechas[j] == fecha else j)
        for c in cands:
            if c < len(fechas):
                ev[pos[t], c] = True
    return ev & (np.nan_to_num(rv) > rv_min)


if __name__ == "__main__":
    p = panel()
    f = p["fechas"]
    rv = rel_vol(p, 1)
    ev = eventos(p, rv)
    hueco = p["O"][:, :, 0] / p["prev"] - 1
    print("eventos:", ev.sum(), "por año:", pd.Series(np.broadcast_to(pd.DatetimeIndex(f).year, ev.shape)[ev]).value_counts().sort_index().to_dict())
    print("hueco medio absoluto en eventos: %.1f%%" % (np.nanmean(np.abs(hueco[ev])) * 100))
    # deriva de apertura a cierre en la dirección del hueco
    cierre = p["cierre"]
    oc = cierre / p["O"][:, :, 0] - 1
    g = np.sign(hueco)
    for umbral in (0, 0.02, 0.04):
        m = ev & (np.abs(hueco) > umbral)
        x = (g * oc)[m]
        print(f"apertura→cierre a favor del hueco (|hueco|>{umbral:.0%}): {np.nanmean(x)*1e4:.1f} pb, t={np.nanmean(x)/np.nanstd(x)*np.sqrt(m.sum()):.2f}, n={m.sum()}")
    filas = []
    for n_or, modo, lado_h, sf, tR in itertools.product((1, 3, 6), (3, 0), ("ambos", "hueco"), (1.0, 0.5),
                                                        (0, 1, 2)):
        s = señales(p, n_or=n_or, modo=modo, atr_stop=0.0, stop_frac=sf, target_R=tR)
        if lado_h == "hueco":
            s = dict(s)
            m = s["dire"] != g
            for k in ("dire", "ret"):
                s[k] = np.where(m, 0, s[k])
        tr = s["ret"][ev & (s["dire"] != 0)]
        fila = dict(n_or=n_or * 5, entrada="cierre+sig. vela" if modo == 3 else "orden stop", lado=lado_h,
                    stop=sf, objetivo=tR, n=len(tr), pb_bruto=tr.mean() * 1e4,
                    t=tr.mean() / tr.std() * np.sqrt(len(tr)), acierto=(tr > 0).mean())
        res = cartera(p, s, rv, K=10, rv=rv, rv_min=0, coste=2e-4, riesgo=0.01, maxlev=4, elegibles=ev)
        inf = informe(res, f)
        filas.append({**fila, **{k: inf[k] for k in ("cagr", "sharpe", "maxdd", "lev_medio")},
                      **{k: v for k, v in inf.items() if k.isdigit()}})
    df = pd.DataFrame(filas)
    df.to_csv(Path(__file__).parent / "resultados" / "ronda5_resultados.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 200)
    print(df.sort_values("t", ascending=False).round(3).to_string())
