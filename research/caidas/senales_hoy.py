"""Señales del último día de datos para la cartera recomendada (ronda 4).

Cartera: D1 + C1 al 50 % (capital separado), hasta 5 huecos de 1/3 cada uno. Niveles
(exposición bruta REAL máxima: acciones + SPY de la reserva):
  1 · máx. 1x, sin deuda: reserva SPY en tendencia / SHY fuera;
  2 · máx. 1,5x: reserva gestionada por volatilidad (SPY/SHY) y palanca de las compras
      escalada por volatilidad;
  3 · máx. 2x (por defecto; el límite estándar de margen): ídem.

Muestra, para cada componente: posiciones que tendría abiertas al último cierre
(simulación continua desde 2002), órdenes para la próxima apertura, estado de la
reserva y, para las candidatas de D1, qué módulos cumple (la explicación de la entrada).

Uso:
    python research/caidas/senales_hoy.py [--nivel 1|2|3]   (por defecto 3)
    python research/caidas/senales_hoy.py --refrescar   # descarga precios nuevos y
        reconstruye panel, rasgos, eventos, puntuaciones (banderas y modelo CONGELADOS)
        y reservas (≈5 min).
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
import motor_v3 as v3  # noqa: E402
import reservas as rs  # noqa: E402

NIVELES = {
    1: dict(reserva_="spy_tend_shy", palanca=1.0, vol_objetivo=None),
    2: dict(reserva_="spy_vg_shy", palanca=1.5, vol_objetivo=0.15),
    3: dict(reserva_="spy_vg_shy", palanca=2.0, vol_objetivo=0.15),
}


def refrescar() -> None:
    cache = m.ROOT / "data" / "cache" / "caidas"
    for f in ("prices.parquet", "membership_intervals.parquet"):
        (cache / f).unlink(missing_ok=True)
    for paso in ("datos.py", "rasgos.py", "eventos.py"):
        subprocess.run([sys.executable, str(AQUI / paso)], check=True)
    subprocess.run([sys.executable, str(AQUI / "meta.py"), "--congelado"], check=True)
    subprocess.run([sys.executable, str(AQUI / "reservas.py")], check=True)


def main() -> None:
    if "--refrescar" in sys.argv:
        refrescar()
    nivel = int(sys.argv[sys.argv.index("--nivel") + 1]) if "--nivel" in sys.argv else 3
    from meta import MODULOS, MODULOS_V2
    from ronda4 import componentes

    p = m.cargar()
    comp = componentes()
    cfg = dict(palanca_off=1.0, K=5, frac=1 / 3) | NIVELES[nivel]
    salida = {"fecha_datos": str(p.fechas[-1]), "nivel": nivel, "config": cfg, "componentes": {}}
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 150)
    reg = rs.regimen_spy()
    vol = rs.vol_spy()
    tend_manana = bool(pd.Series(p.close_ff[:, p.col["SPY"]]).iloc[-1]
                       > pd.Series(p.close_ff[:, p.col["SPY"]]).rolling(200).mean().iloc[-1])
    print(f"Datos hasta {p.fechas[-1]} · nivel {nivel} · {cfg}")
    print(f"SPY sobre su SMA200 al último cierre: {tend_manana} · vol. 20d de SPY: {vol[-1]:.1%}")
    if cfg["reserva_"] == "spy_vg_shy":
        e = min(1.5, 0.15 / vol[-1]) if reg[-1] else 0.0
        print(f"Reserva gestionada: exposición a SPY {e:.2f}x (resto en SHY)")
    for c in ("D1", "C1"):
        x, sal, conv = comp[c]
        w = np.minimum(x.peso.to_numpy(), 0.5 / cfg["frac"]) if conv else None
        r = v3.simular(x.t.to_numpy(), x.j.to_numpy(), x.prio.to_numpy(np.float64), peso=w, H=20, salida=sal,
                       detalle=True, **cfg)
        ops = r["operaciones"]
        ab = ops[ops.t_sal < 0][["ticker", "entrada", "px_ent", "px_sal", "ret", "peso"]].rename(
            columns={"px_sal": "px_ultimo", "ret": "ret_latente", "peso": "peso_entrada"})
        salida["componentes"][c] = {
            "posiciones_abiertas": ab.assign(entrada=ab.entrada.astype(str)).to_dict("records"),
            "ordenes_proxima_apertura": r["pendientes"],
            "deuda_sobre_capital": r["deuda_final"] / r["equity"].iloc[-1],
            "exposicion_acciones": r["exposicion_final"]}
        print(f"\n=== Componente {c} (50 % del capital) ===")
        print(ab.round({"px_ent": 2, "px_ultimo": 2, "ret_latente": 3, "peso_entrada": 3}).to_string(index=False)
              if len(ab) else "Posiciones abiertas: ninguna")
        print("Órdenes para la próxima apertura:", r["pendientes"] or "ninguna",
              f"· exposición en acciones {r['exposicion_final']:.0%}"
              f" · deuda {r['deuda_final'] / r['equity'].iloc[-1]:.0%} del capital del componente")

    # explicación de las candidatas de D1 del último día
    banderas = json.loads((AQUI / "resultados" / "meta_banderas.json").read_text())["banderas"]
    todos = {**{f"{a}: {b}": f for a, gs in MODULOS.items() for b, f in gs.items()},
             **{f"{a}: {b}": f for a, gs in MODULOS_V2.items() for b, f in gs.items()}}
    e = ev.cargar()
    d1 = comp["D1"][0]
    hoy = e[(e.t == len(p.fechas) - 1) & (e.sobre_sma200 == 1) & (e.dd20 <= -0.15)]
    hoy = hoy.merge(d1[["t", "j"]].assign(entra=True), on=["t", "j"], how="left")
    from ronda3 import cargar as cargar_meta
    sc = cargar_meta()[["t", "j", "score_v2"]]
    hoy = hoy.merge(sc, on=["t", "j"], how="left").sort_values("score_v2", ascending=False)
    filas = []
    for _, fila in hoy.head(15).iterrows():
        df1 = fila.to_frame().T.infer_objects()
        a_favor = [b.split(": ", 1)[1] for b in banderas if bool(todos[b](df1).fillna(False).iloc[0])]
        filas.append({"ticker": fila.ticker, "caída desde máx. 20d": f"{fila.dd20:.1%}",
                      "módulos a favor": int(fila.score_v2) if pd.notna(fila.score_v2) else 0,
                      "entra (≥5)": bool(fila.score_v2 >= 5) if pd.notna(fila.score_v2) else False,
                      "detalle": "; ".join(a_favor)})
    salida["candidatas_D1_ultimo_dia"] = filas
    (AQUI / "resultados" / "senales_ultimo_dia.json").write_text(
        json.dumps(salida, ensure_ascii=False, indent=1, default=str))
    print("\nCandidatas D1 del último día (caída ≥15 % desde máx. 20d, en tendencia):")
    print(pd.DataFrame(filas).to_string(index=False) if filas else "  (ninguna)")


if __name__ == "__main__":
    main()
