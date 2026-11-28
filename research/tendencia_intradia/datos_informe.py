"""Reúne en un JSON las cifras y series del informe de tendencia intradía."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent / "bracket_intradia"))
import evaluacion as E  # noqa: E402
import paneles as T  # noqa: E402
import semaforo as S  # noqa: E402

R = AQUI / "resultados"


def _cond(g: pd.DataFrame, valores: pd.Series, bins, labels) -> list[dict]:
    d = g[g.ret_ap.notna() & (g.ret_ap != 0) & g.y_cierre.notna()].copy()
    d["x"] = np.sign(d.ret_ap) * d.y_cierre
    d["b"] = pd.cut(valores.loc[d.index], bins, labels=labels)
    out = []
    for b, gg in d.groupby("b", observed=True):
        pdia = gg.groupby("fecha").x.mean()
        out.append({"tramo": str(b), "n": len(gg), "ventaja": gg.x.mean(),
                    "acierto": (gg.x > 0).mean(),
                    "t": pdia.mean() / pdia.std() * np.sqrt(len(pdia))})
    return out


def ejemplo(tk: str, fecha: str) -> dict:
    P = T.cargar(f"m5_{tk}")
    M = S.matrices(P)
    Est = S.estado(M)
    i = list(pd.to_datetime(P["fechas"])).index(pd.Timestamp(fecha))
    o0 = P["O"][i, 0]
    base = np.abs(P["C"][i] / o0 - 1) / np.abs(M["sigma"][i])
    tp = (P["H"][i] + P["L"][i] + P["C"][i]) / 3
    vw = np.cumsum(tp * P["V"][i]) / np.cumsum(P["V"][i])
    return {"valor": tk, "fecha": fecha, "apertura": o0, "close": np.round(P["C"][i], 2).tolist(),
            "vwap": np.round(vw, 2).tolist(),
            "banda_sup": np.round(o0 * (1 + base), 2).tolist(),
            "banda_inf": np.round(o0 * (1 - base), 2).tolist(),
            "estado": Est[i].tolist(), "rvol": np.round(M["rvol"][i], 2).tolist(),
            "sigma": np.round(M["sigma"][i], 2).tolist(), "atr": float(P["atr"][i])}


def main() -> dict:
    out = {}
    ev = pd.read_csv(R / "rasgos.csv")
    sel = ((ev.rasgo == "ret_ap") & ev.cp.isin([60, 120, 180, 240, 300, 360])) | \
          ((ev.conjunto == "acc_h1") & ev.cp.isin([60, 120]))
    out["ic"] = ev[sel][["conjunto", "cp", "rasgo", "ic", "ventaja", "t", "n"]].round(4).to_dict("records")
    h1 = E.conjunto("acc_h1")
    g = h1[h1.cp == 60]
    out["cond_rvol"] = _cond(g, g.rvol, [0, 0.8, 1.2, 1.6, 2, 3, 99],
                             ["< 0,8", "0,8–1,2", "1,2–1,6", "1,6–2", "2–3", "≥ 3"])
    out["cond_sigma"] = _cond(g, g.ruido.abs(), [0, 0.5, 1, 1.5, 2, 3, 99],
                              ["< 0,5", "0,5–1", "1–1,5", "1,5–2", "2–3", "≥ 3"])
    out["frecuencia"] = pd.read_csv(R / "frecuencia_dias.csv", index_col=0)["valor"].to_dict()
    out["huella"] = pd.read_csv(R / "huella.csv").round(4).to_dict("records")
    out["resto"] = pd.read_csv(R / "resto_del_dia.csv").round(4).to_dict("records")
    de = pd.read_csv(R / "semaforo_deriva.csv")
    out["deriva"] = de[de.cp.isin([60, 120])].round(4).to_dict("records")
    op = pd.read_csv(R / "semaforo_operativa.csv")
    out["operativa"] = op[op.f_atr == 0.25].round(4).to_dict("records")
    out["robustez"] = pd.read_csv(R / "robustez.csv").round(4).to_dict("records")
    out["modelo"] = pd.read_csv(R / "modelo_vs_semaforo.csv").round(4).to_dict("records")
    ops = pd.read_csv(R / "operaciones_verde_h1.csv", parse_dates=["fecha"])
    out["por_anio"] = ops.groupby(ops.fecha.dt.year).neto.agg(["size", "mean"]).reset_index() \
        .rename(columns={"fecha": "anio"}).round(4).to_dict("records")
    q = ops.groupby(ops.fecha.dt.to_period("Q")).neto.agg(["size", "mean"]).reset_index()
    q["fecha"] = q.fecha.astype(str)
    out["por_trimestre"] = q.round(4).to_dict("records")
    out["por_hora"] = ops.groupby("entrada_min").neto.agg(["size", "mean"]).reset_index().round(4).to_dict("records")
    vv = ops.groupby("valor").neto.agg(["size", "mean"]).reset_index()
    out["por_valor_n"] = {"NVDA": int(vv.set_index("valor").loc["NVDA", "size"]),
                          "media": float(vv["size"].mean()), "valores": int(len(vv)),
                          "dias": int(ops.fecha.nunique())}
    out["ejemplos"] = [ejemplo("NVDA", "2026-11-05"), ejemplo("NVDA", "2026-11-06"),
                       ejemplo("NVDA", "2026-11-03")]
    return out


if __name__ == "__main__":
    d = main()
    destino = Path(sys.argv[1]) if len(sys.argv) > 1 else R / "informe.json"
    destino.write_text(json.dumps(d, ensure_ascii=False, separators=(",", ":"), default=float))
    print(destino, destino.stat().st_size // 1024, "KB")
    print(d["frecuencia"], d["por_anio"], d["por_hora"], d["por_valor_n"])
    print(pd.DataFrame(d["cond_rvol"]).round(3).to_string()); print(pd.DataFrame(d["cond_sigma"]).round(3).to_string())
