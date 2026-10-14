"""Señales del último día de datos para la estrategia definitiva (ronda 5).

D1 con ENTRADAS TÉCNICAS, solo acciones (sin ETFs ni efectivo):
- entra: miembro líquido sobre su SMA200, caída >= 15 % desde el máximo de 20 d,
  >= 5 módulos de origen de la caída y >= 2 banderas técnicas; compra a la apertura;
- hasta 5 posiciones de 1/2 hueco de convicción; sale con cierre > SMA10 (máx. 20 d);
- reserva (solo acciones; SPY solo como señal):
  nivel 1 · sin deuda: 100 acciones más líquidas (SPY > SMA200) o 50 de menor volatilidad;
  nivel 2 · CONTRACÍCLICA (por defecto): en calma y en tendencia, hasta 1,5x las 100
            líquidas según la volatilidad de SPY (objetivo 15 %); el resto y fuera de
            tendencia, en las 50 de menor volatilidad;
  nivel 3 · contracíclica + compras de caídas con palanca hasta 2x en tendencia.

Muestra posiciones abiertas (simulación continua desde 2002), órdenes para la próxima
apertura, estado de la reserva y, para cada candidata del último día, qué módulos y qué
banderas técnicas cumple (la explicación de la entrada).

Uso:
    python research/caidas/senales_hoy.py [--nivel 1|2|3]
    python research/caidas/senales_hoy.py --refrescar   # descarga precios nuevos y
        reconstruye panel, rasgos, eventos, puntuaciones (CONGELADAS), técnicos y reservas.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

AQUI = Path(__file__).parent
sys.path.insert(0, str(AQUI))

import motor as m  # noqa: E402
import reservas as rs  # noqa: E402

NIVELES = {1: (1.0, "liq100_tend_minvol"), 2: (1.0, "liq100_vg_minvol"), 3: (2.0, "liq100_vg_minvol")}


def refrescar() -> None:
    cache = m.ROOT / "data" / "cache" / "caidas"
    for f in ("prices.parquet", "membership_intervals.parquet"):
        (cache / f).unlink(missing_ok=True)
    for paso in ("datos.py", "rasgos.py", "eventos.py"):
        subprocess.run([sys.executable, str(AQUI / paso)], check=True)
    subprocess.run([sys.executable, str(AQUI / "meta.py"), "--congelado"], check=True)
    for paso in ("tecnicos.py", "reservas.py"):
        subprocess.run([sys.executable, str(AQUI / paso)], check=True)


def main() -> None:
    if "--refrescar" in sys.argv:
        refrescar()
    nivel = int(sys.argv[sys.argv.index("--nivel") + 1]) if "--nivel" in sys.argv else 2
    pal, reserva = NIVELES[nivel]
    import definitiva as df_
    from meta import MODULOS, MODULOS_V2
    from ronda5 import componentes_tec

    p = m.cargar()
    comp_tec = componentes_tec()
    r = df_.curva(comp_tec, "D1", df_.D1_CFG, pal, detalle=True, reserva=reserva)
    ops = r["operaciones"]
    ab = ops[ops.t_sal < 0][["ticker", "entrada", "px_ent", "px_sal", "ret", "peso"]].rename(
        columns={"px_sal": "px_ultimo", "ret": "ret_latente", "peso": "peso_entrada"})
    reg = rs.regimen_spy()
    spy = pd.Series(p.close_ff[:, p.col["SPY"]])
    en_tendencia = bool(spy.iloc[-1] > spy.rolling(200).mean().iloc[-1])
    print(f"Datos hasta {p.fechas[-1]} · nivel {nivel} (caídas hasta {pal:g}x en tendencia · reserva {reserva})")
    if reserva == "liq100_vg_minvol":
        e = min(1.5, 0.15 / rs.vol_spy()[-1]) if reg[-1] else 0.0
        print(f"SPY sobre su SMA200: {en_tendencia} · vol. 20d de SPY {rs.vol_spy()[-1]:.1%} → reserva: "
              f"{e:.2f}x en las 100 acciones más líquidas" + (f" y {1 - e:.2f}x en las 50 de baja vol." if e < 1 else ""))
    else:
        print(f"SPY sobre su SMA200: {en_tendencia} → reserva en "
              f"{'las 100 acciones más líquidas' if reg[-1] else 'las 50 acciones de menor volatilidad'}")
    print("\nPosiciones abiertas al cierre:")
    print(ab.round({"px_ent": 2, "px_ultimo": 2, "ret_latente": 3, "peso_entrada": 3}).to_string(index=False)
          if len(ab) else "  (ninguna)")
    print("Órdenes para la próxima apertura:", r["pendientes"] or "ninguna",
          f"· exposición en compras {r['exposicion_final']:.0%}"
          f" · deuda de las compras {r['deuda_final'] / r['equity'].iloc[-1]:.0%} del capital"
          " (la reserva contracíclica lleva su propia palanca, indicada arriba)")

    # explicación de las candidatas del último día
    banderas = json.loads((AQUI / "resultados" / "meta_banderas.json").read_text())["banderas"]
    tecnicas = json.loads((AQUI / "resultados" / "modulos_tec_elegidas.json").read_text())
    todos = {**{f"{a}: {b}": f for a, gs in MODULOS.items() for b, f in gs.items()},
             **{f"{a}: {b}": f for a, gs in MODULOS_V2.items() for b, f in gs.items()}}
    import eventos as ev
    x = comp_tec["D1"][0]
    hoy = x[x.t == len(p.fechas) - 1]
    e = ev.cargar()
    hoy = hoy.merge(e[["t", "j"] + [c for c in e.columns if c not in hoy.columns]], on=["t", "j"], how="left")
    hoy = hoy.sort_values(["score_v2", "score_tec"], ascending=False)
    filas = []
    for _, fila in hoy.iterrows():
        df1 = fila.to_frame().T.infer_objects()
        mods = [b.split(": ", 1)[1] for b in banderas if bool(todos[b](df1).fillna(False).iloc[0])]
        tecs = [n.split(" (")[0] for n, (rasgo, op, v) in tecnicas.items()
                if pd.notna(fila[rasgo]) and {"<=": fila[rasgo] <= v, ">=": fila[rasgo] >= v, "<": fila[rasgo] < v}[op]]
        filas.append({"ticker": fila.ticker, "módulos": int(fila.score_v2), "técnicas": int(fila.score_tec),
                      "entra": bool(fila.score_tec >= df_.D1_CFG["kmin"]),
                      "módulos a favor": "; ".join(mods), "técnicas a favor": "; ".join(tecs)})
    salida = {"fecha_datos": str(p.fechas[-1]), "nivel": nivel, "palanca_caidas": pal, "reserva": reserva,
              "reserva_en_tendencia": en_tendencia,
              "posiciones_abiertas": ab.assign(entrada=ab.entrada.astype(str)).to_dict("records"),
              "ordenes_proxima_apertura": r["pendientes"], "candidatas_ultimo_dia": filas}
    (AQUI / "resultados" / "senales_ultimo_dia.json").write_text(
        json.dumps(salida, ensure_ascii=False, indent=1, default=str))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 120)
    print("\nCandidatas del último día (caída ≥15 % desde máx. 20d, en tendencia, ≥5 módulos):")
    print(pd.DataFrame(filas).to_string(index=False) if filas else "  (ninguna)")


if __name__ == "__main__":
    main()
