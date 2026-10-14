"""Ronda 6 · estudio de eventos de COMPLEMENTOS para cuando hay pocas caídas que comprar.

Todos sobre acciones (miembros PIT y líquidas, salvo las expulsadas, que por definición
acaban de dejar el índice). Retorno desde la apertura de t+1 hasta el cierre de t+h, en
exceso sobre la media equiponderada de los miembros líquidos en la misma ventana; se
promedia por fecha y la t es de Newey-West (h lags).

1. **Salto con noticia** (continuación, la cara opuesta de "la caída con noticia sigue"):
   hueco al alza >= g % con volumen >= 3x y cierre por encima de la apertura.
2. **Máximo de 52 semanas** (anclaje, George y Hwang 2004): cierre en máximo anual con
   volumen >= 1,5x; variante "cerca del máximo" (>= 98 %).
3. **Expulsada del S&P 500** (rebote tras la venta forzada de los indexados): primera
   sesión fuera del índice, si la acción sigue cotizando.
4. **Retroceso suave de una líder**: momentum 12 m >= 50 %, a 3-8 % de su máximo de
   20 d, RSI2 < 10 y en tendencia.

Uso: python research/caidas/complementos.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import motor as m
from estudio import nw_t

OUT = Path(__file__).parent / "resultados"
CORTE = np.datetime64("2021-10-01")
MITAD = np.datetime64("2024-04-01")
INICIO = np.datetime64("2002-01-02")
HORIZ = (5, 20, 60, 120)


def adelante(p: m.Panel) -> dict[int, np.ndarray]:
    T = len(p.fechas)
    base = p.miembro & m.liquido()
    entrada = np.where(np.isnan(p.open), p.close_ff, p.open)
    out = {}
    for h in HORIZ:
        f = np.full(p.close.shape, np.nan)
        f[: T - h] = p.close_ff[h:] / entrada[1 : T - h + 1] - 1
        mu = np.nanmean(np.where(base, f, np.nan), axis=1, keepdims=True)
        out[h] = f - mu
    return out


def resumir(nombre: str, ev: np.ndarray, fx: dict, fechas: np.ndarray) -> list[dict]:
    filas = []
    for w, sel in (("busq1", (fechas >= CORTE) & (fechas < MITAD)), ("busq2", fechas >= MITAD),
                   ("valid", (fechas >= INICIO) & (fechas < CORTE))):
        fila = {"evento": nombre, "ventana": w, "eventos": int(ev[sel].sum())}
        for h in HORIZ:
            x = np.where(ev[sel], fx[h][sel], np.nan)
            with np.errstate(invalid="ignore"):
                diaria = np.nanmean(x, axis=1)
            fila[f"exc{h}"] = float(np.nanmean(diaria))
            fila[f"t{h}"] = nw_t(diaria, h)
        filas.append(fila)
    return filas


def main() -> None:
    p = m.cargar()
    import rasgos as rg
    R = rg.cargar_rasgos()
    liq = m.liquido()
    base = p.miembro & liq
    fx = adelante(p)
    fechas = p.fechas
    C = p.close
    filas = []
    with np.errstate(invalid="ignore", divide="ignore"):
        noche = R["noche1"]
        volr = R["vol_rel"]
        intradia = C / p.open - 1
        for g in (0.03, 0.05, 0.08):
            ev = base & (noche >= g) & (volr >= 3) & (intradia >= 0)
            filas += resumir(f"salto con noticia ≥{g:.0%}", ev, fx, fechas)
            filas += resumir(f"salto con noticia ≥{g:.0%} · tendencia", ev & (R["sobre_sma200"] == 1), fx, fechas)
        max252 = m.maximo(252)
        nuevo_max = C >= max252
        filas += resumir("máximo 52 sem. con volumen ≥1,5x", base & nuevo_max & (volr >= 1.5), fx, fechas)
        filas += resumir("máximo 52 sem.", base & nuevo_max, fx, fechas)
        filas += resumir("a ≥98 % del máximo 52 sem.", base & (C >= 0.98 * max252), fx, fechas)
        lider = (R["mom252"] >= 0.5) & (R["dd20"] <= -0.03) & (R["dd20"] >= -0.08) & (m.rsi(2) < 10)
        filas += resumir("retroceso suave de líder", base & lider & (R["sobre_sma200"] == 1), fx, fechas)
    # expulsadas: primera sesión fuera del índice (si sigue cotizando al menos 20 sesiones)
    miembros = pd.read_parquet(m.ROOT / "data/cache/caidas/membership_intervals.parquet")
    miembros["end"] = pd.to_datetime(miembros["end"])
    ev = np.zeros(C.shape, dtype=bool)
    fe = pd.DatetimeIndex(fechas)
    for row in miembros.dropna(subset=["end"]).itertuples():
        j = p.col.get(row.ticker)
        if j is None:
            continue
        t = int(fe.searchsorted(row.end))
        if t + 21 < len(fe) and not np.isnan(C[t + 20, j]) and not p.miembro[t, j]:
            ev[t, j] = True
    filas += resumir("expulsada del S&P 500", ev, fx, fechas)
    df = pd.DataFrame(filas)
    df.to_csv(OUT / "complementos_eventos.csv", index=False, float_format="%.5f")
    pd.set_option("display.width", 250)
    v = df.pivot_table(index="evento", columns="ventana", values=["eventos", "exc20", "t20", "exc60", "t60"],
                       sort=False)
    for c in ("exc20", "exc60"):
        for w in ("busq1", "busq2", "valid"):
            v[(c, w)] *= 100
    print(v.round(2).to_string())
    v2 = df.pivot_table(index="evento", columns="ventana", values=["exc5", "exc120"], sort=False) * 100
    print(v2.round(2).to_string())


if __name__ == "__main__":
    main()
