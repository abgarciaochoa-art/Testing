"""Señales del último día de datos para la estrategia recomendada (D1-def50).

Muestra:
1. posiciones que la estrategia tendría abiertas al último cierre (simulación
   continua desde 2002, no una cartera "desde cero");
2. órdenes para la próxima apertura;
3. candidatas del último día con sus módulos a favor (la explicación de cada entrada).

Uso:
    python research/caidas/senales_hoy.py              # con los datos en caché
    python research/caidas/senales_hoy.py --refrescar  # descarga precios nuevos y
        reconstruye panel, rasgos, eventos y puntuaciones (≈4 min). Las banderas y el
        modelo quedan CONGELADOS (ventana de entrenamiento fija), ver meta.py.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).parent
sys.path.insert(0, str(AQUI))

import eventos as ev  # noqa: E402
import motor as m  # noqa: E402
import motor_cartera as mc  # noqa: E402

CONFIG = dict(K=3, H=20, salida="sma10", reserva="spy_tend", tope=0.5)


def refrescar() -> None:
    cache = m.ROOT / "data" / "cache" / "caidas"
    for f in ("prices.parquet", "membership_intervals.parquet"):
        (cache / f).unlink(missing_ok=True)
    for paso in ("datos.py", "rasgos.py", "eventos.py"):
        subprocess.run([sys.executable, str(AQUI / paso)], check=True)
    subprocess.run([sys.executable, str(AQUI / "meta.py"), "--congelado"], check=True)


def main() -> None:
    if "--refrescar" in sys.argv:
        refrescar()
    from meta import MODULOS, MODULOS_V2  # noqa: F401  (para nombrar los módulos)
    import ronda3 as r3

    d = r3.cargar()
    x = d[(d.dd20 <= -0.15) & (d.score_v2 >= 5)].copy()
    x["prio"] = x.score_v2 + x.meta_pct.fillna(0.5) * 0.99
    w = np.minimum(np.clip((x.score_v2.to_numpy() - 4) / 2, 0.5, 2.0), CONFIG["tope"] * CONFIG["K"])
    r = mc.simular(x.t.to_numpy(), x.j.to_numpy(), x.prio.to_numpy(np.float64), peso=w, K=CONFIG["K"],
                   H=CONFIG["H"], salida=CONFIG["salida"], reserva=CONFIG["reserva"], detalle=True)
    p = m.cargar()
    ultimo = str(p.fechas[-1])
    ops = r["operaciones"]
    abiertas = ops[ops.t_sal < 0][["ticker", "entrada", "px_ent", "px_sal", "ret", "peso"]]
    abiertas = abiertas.rename(columns={"px_sal": "px_ultimo", "ret": "ret_latente", "peso": "peso_cartera"})

    # candidatas del último día y sus módulos
    banderas = json.loads((AQUI / "resultados" / "meta_banderas.json").read_text())["banderas"]
    todos = {**{f"{a}: {b}": f for a, gs in MODULOS.items() for b, f in gs.items()},
             **{f"{a}: {b}": f for a, gs in MODULOS_V2.items() for b, f in gs.items()}}
    e = ev.cargar()
    hoy = e[(e.t == len(p.fechas) - 1) & (e.sobre_sma200 == 1)].merge(
        d[["t", "j", "score_v2", "meta_pct"]], on=["t", "j"], how="left")
    hoy = hoy[hoy.dd20 <= -0.15].sort_values(["score_v2", "meta_pct"], ascending=False)
    filas = []
    for _, fila in hoy.head(15).iterrows():
        df1 = fila.to_frame().T.infer_objects()
        a_favor = [b.split(": ", 1)[1] for b in banderas if bool(todos[b](df1).fillna(False).iloc[0])]
        filas.append({"ticker": fila.ticker, "caída desde máx. 20d": f"{fila.dd20:.1%}",
                      "módulos a favor": int(fila.score_v2), "entra (≥5)": bool(fila.score_v2 >= 5),
                      "detalle": "; ".join(a_favor)})
    cand = pd.DataFrame(filas)
    salida = {"fecha_datos": ultimo, "config": CONFIG,
              "posiciones_abiertas": abiertas.assign(entrada=abiertas.entrada.astype(str)).to_dict("records"),
              "ordenes_proxima_apertura": r["pendientes"], "candidatas_ultimo_dia": filas}
    (AQUI / "resultados" / "senales_ultimo_dia.json").write_text(
        json.dumps(salida, ensure_ascii=False, indent=1, default=str))
    # componente C1 (50 % del capital en la cartera combinada)
    import finalistas3 as f3
    c1 = f3.candidatas_c1(d, e)
    cfg1 = f3.ESTRATEGIAS["C1 · ronda 2 (referencia)"]
    r1 = mc.simular(c1.t.to_numpy(), c1.j.to_numpy(), c1.prio.to_numpy(np.float64), K=cfg1["K"], H=cfg1["H"],
                    salida=cfg1["salida"], reserva=cfg1["reserva"], detalle=True)
    o1 = r1["operaciones"]
    ab1 = o1[o1.t_sal < 0][["ticker", "entrada", "px_ent", "px_sal", "ret", "peso"]]
    salida["C1"] = {"posiciones_abiertas": ab1.assign(entrada=ab1.entrada.astype(str)).to_dict("records"),
                    "ordenes_proxima_apertura": r1["pendientes"]}
    (AQUI / "resultados" / "senales_ultimo_dia.json").write_text(
        json.dumps(salida, ensure_ascii=False, indent=1, default=str))
    reg = mr_reserva(p)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 140)
    print(f"Datos hasta {ultimo}  ·  reserva: {'SPY' if reg else 'efectivo (SPY bajo su SMA200)'}")
    print("\n=== Componente D1-def50 (50 % del capital) ===")
    print(f"configuración {CONFIG}")
    print("\nPosiciones abiertas al cierre:")
    print(abiertas.round({"px_ent": 2, "px_ultimo": 2, "ret_latente": 3, "peso_cartera": 3}).to_string(index=False) if len(abiertas) else "  (ninguna)")
    print("\nÓrdenes para la próxima apertura:", r["pendientes"] or "ninguna")
    print("\nCandidatas del último día (caída ≥15 % desde máx. 20d, en tendencia):")
    print(cand.to_string(index=False) if len(cand) else "  (ninguna)")
    print("\n=== Componente C1 (50 % del capital) ===")
    print(ab1.round(3).to_string(index=False) if len(ab1) else "  posiciones abiertas: ninguna")
    print("Órdenes para la próxima apertura:", r1["pendientes"] or "ninguna")


def mr_reserva(p) -> bool:
    """¿La reserva está en SPY (SPY > SMA200 al último cierre)?"""
    spy = pd.Series(p.close_ff[:, p.col["SPY"]])
    return bool(spy.iloc[-1] > spy.rolling(200).mean().iloc[-1])


if __name__ == "__main__":
    main()
