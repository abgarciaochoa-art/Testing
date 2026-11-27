"""Motor de la horquilla intradía: una posición a la vez, stop y objetivo fijos en múltiplos
de R (la distancia al stop), con candado opcional (al llegar a +X R el stop sube a +Y R) y
stop dinámico opcional (a T R del máximo alcanzado). Todo se cierra al final de la sesión.

Convenciones (conservadoras):
- La señal se evalúa al cierre de la vela j; se entra a la apertura de la vela j+1.
- Dentro de una vela, si se tocan el stop y el objetivo, cuenta el stop (se anota como
  "ambigua"). Si la vela abre ya más allá del stop, se sale a la apertura (hueco).
- El objetivo se llena a su precio (orden limitada); el stop a su precio o peor.
- Los cambios de stop (candado, dinámico) se calculan con el máximo de la vela; si esa misma
  vela cierra por debajo del nuevo stop, se sale al stop (el precio lo cruzó al bajar desde el
  máximo). Si solo el mínimo lo perforó y cerró por encima, no se sabe el orden y se sigue.
- Los cortos se simulan con la serie de precios invertida (−precio).

Salida: una fila por operación con el resultado en R bruto (sin costes). Los costes se
aplican luego (``costes_R``) porque dependen del broker: un coste por lado en % del precio
se convierte en R dividiendo por el tamaño del stop.
"""

from __future__ import annotations

import numpy as np
from numba import njit

# códigos de salida
STOP, OBJETIVO, CIERRE, CANDADO = 0, 1, 2, 3


@njit(cache=True)
def _simular(O, H, L, C, senal, R, tp, cand_x, cand_y, trail, max_ops, max_perd,
             espera, ultima_entrada, cierre_vela):
    D, N = O.shape
    cap = D * 200
    out = np.zeros((cap, 9))  # dia, j_entrada, j_salida, entrada, salida, R, pnl_R, motivo, mfe_R
    k = 0
    for d in range(D):
        ops = 0
        perd = 0
        libre_desde = 0
        j = 0
        while j < min(ultima_entrada, cierre_vela):
            if ops >= max_ops or perd >= max_perd:
                break
            if j < libre_desde or not senal[d, j] or not (R[d, j] > 0):
                j += 1
                continue
            # entrada a la apertura de la vela siguiente
            je = j + 1
            ent = O[d, je]
            r = R[d, j]
            stop = ent - r
            obj = ent + tp * r if tp > 0 else 1e18
            maximo = ent
            candado_activo = False
            salida = -1.0
            motivo = -1
            jx = je
            amb = False
            while jx <= cierre_vela:
                o, h, l = O[d, jx], H[d, jx], L[d, jx]
                # 1) stop (incluye hueco a la apertura)
                if l <= stop:
                    salida = min(stop, o)
                    motivo = CANDADO if candado_activo else STOP
                    if h >= obj:
                        amb = True
                    break
                # 2) objetivo
                if h >= obj:
                    salida = max(obj, o)
                    motivo = OBJETIVO
                    break
                # 3) actualizar máximo y stop para la vela siguiente
                if h > maximo:
                    maximo = h
                if cand_x > 0 and maximo >= ent + cand_x * r:
                    nuevo = ent + cand_y * r
                    if nuevo > stop:
                        stop = nuevo
                        candado_activo = True
                if trail > 0 and candado_activo:
                    nuevo = maximo - trail * r
                    if nuevo > stop:
                        stop = nuevo
                # si la vela cierra por debajo del stop recién subido, el precio lo cruzó
                # después de marcar el máximo: se sale al stop en esta misma vela
                if candado_activo and C[d, jx] <= stop:
                    salida = stop
                    motivo = CANDADO
                    break
                jx += 1
            if motivo < 0:
                jx = cierre_vela
                salida = C[d, jx]
                motivo = CIERRE
            pnl = (salida - ent) / r
            if k < cap:
                out[k, 0] = d
                out[k, 1] = je
                out[k, 2] = jx
                out[k, 3] = ent
                out[k, 4] = salida
                out[k, 5] = r
                out[k, 6] = pnl
                out[k, 7] = motivo + (10 if amb else 0)
                out[k, 8] = (maximo - ent) / r
                k += 1
            ops += 1
            if pnl < 0:
                perd += 1
            libre_desde = jx + 1 + espera
            j = jx + 1
    return out[:k]


def simular(P: dict, senal: np.ndarray, R: np.ndarray, *, tp: float = 2.0,
            candado: tuple[float, float] | None = None, trail: float = 0.0,
            max_ops: int = 99, max_perd: int = 99, espera: int = 0,
            ultima_entrada: int | None = None, corto: bool = False):
    """Simula la horquilla. ``R`` (días × velas o días) es la distancia al stop en precio.

    ``ultima_entrada`` y ``espera`` van en velas. Devuelve un dict de arrays por operación.
    """
    O, H, L, C = (P[k] for k in "OHLC")
    if corto:
        O, H, L, C = -O, -L, -H, -C
    D, N = O.shape
    if R.ndim == 1:
        R = np.repeat(R[:, None], N, axis=1)
    cierre = N - 1 - max(1, 5 // int(P["paso"]))  # cerrar ~5 minutos antes del final
    ue = cierre - 1 if ultima_entrada is None else min(ultima_entrada, cierre - 1)
    cx, cy = candado if candado else (0.0, 0.0)
    a = _simular(np.ascontiguousarray(O), np.ascontiguousarray(H), np.ascontiguousarray(L),
                 np.ascontiguousarray(C), np.ascontiguousarray(senal), np.ascontiguousarray(R),
                 float(tp), float(cx), float(cy), float(trail), int(max_ops), int(max_perd),
                 int(espera), int(ue), int(cierre))
    t = {"dia": a[:, 0].astype(int), "j": a[:, 1].astype(int), "jx": a[:, 2].astype(int),
         "entrada": np.abs(a[:, 3]), "salida": np.abs(a[:, 4]), "R": a[:, 5], "pnl": a[:, 6],
         "motivo": (a[:, 7] % 10).astype(int), "ambigua": a[:, 7] >= 10, "mfe": a[:, 8]}
    return t


def costes_R(t: dict, por_lado: float, desliz_stop: float = 0.0) -> np.ndarray:
    """Coste de cada operación en R. ``por_lado``: fracción del precio por cada orden
    (comisión + medio diferencial). ``desliz_stop``: fracción extra en las salidas a mercado
    (stop, candado y cierre)."""
    c = 2 * por_lado * t["entrada"] / t["R"]
    a_mercado = t["motivo"] != OBJETIVO
    return c + a_mercado * desliz_stop * t["entrada"] / t["R"]


def resumen(t: dict, coste: np.ndarray | float = 0.0, D: int | None = None) -> dict:
    n = len(t["pnl"])
    if n == 0:
        return {"ops": 0}
    neto = t["pnl"] - coste
    gana = t["pnl"] > 0
    dias = D or (t["dia"].max() + 1)
    m = neto.mean()
    # t agrupado por día: las operaciones del mismo día comparten régimen
    por_dia = np.bincount(t["dia"], weights=neto)
    cuenta = np.bincount(t["dia"])
    por_dia = por_dia[cuenta > 0]
    k = len(por_dia)
    tt = por_dia.mean() / por_dia.std(ddof=1) * np.sqrt(k) if k > 2 and por_dia.std() > 0 else np.nan
    pb = np.bincount(t["dia"], weights=t["pnl"])[cuenta > 0]
    tb = pb.mean() / pb.std(ddof=1) * np.sqrt(k) if k > 2 and pb.std() > 0 else np.nan
    return {
        "ops": n, "ops_dia": n / dias, "acierto": gana.mean(),
        "R_bruto": t["pnl"].mean(), "t_bruto": tb, "R_neto": m, "t": tt,
        "objetivo%": (t["motivo"] == OBJETIVO).mean(), "stop%": (t["motivo"] == STOP).mean(),
        "cierre%": (t["motivo"] == CIERRE).mean(), "ambiguas%": t["ambigua"].mean(),
        "min_medios": (t["jx"] - t["j"] + 1).mean(),
    }
