"""Motor de backtest de estrategias de "comprar caídas" sobre el S&P 500 real.

Convenciones (todas pensadas para que el backtest sea ejecutable de verdad):

- La señal se calcula con el cierre de la sesión ``t`` (solo datos <= t).
- Entradas y salidas se ejecutan a la **apertura de t+1** (precio ajustado por
  dividendos/splits). Si no hay apertura ese día, la entrada se descarta y la salida
  se hace al cierre.
- Solo se compra lo que es miembro del S&P 500 el día ``t`` (pertenencia PIT). Una
  posición abierta se mantiene aunque la empresa salga del índice; si deja de
  cotizar (fusión, quiebra), se liquida al último precio disponible, que en las
  quiebras es el precio real de quiebra.
- Cartera de ``K`` huecos: cada entrada recibe ``equity/K`` (equity al cierre de t).
  El efectivo no remunera (conservador).
- Costes: ``coste_bps`` por lado sobre el nominal negociado.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "data" / "cache" / "caidas" / "panel.npz"
INICIO = np.datetime64("2002-01-02")
CORTE_OOS = np.datetime64("2016-01-01")


# --------------------------------------------------------------------------- datos
@dataclass
class Panel:
    fechas: np.ndarray
    tickers: np.ndarray
    miembro: np.ndarray
    close: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    raw_close: np.ndarray
    precio_real: np.ndarray
    volume: np.ndarray
    sin_apertura: np.ndarray
    close_ff: np.ndarray = field(init=False)
    ultimo: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.close_ff = pd.DataFrame(self.close).ffill().to_numpy()
        self.sin_apertura = self.sin_apertura > 0.5
        valido = ~np.isnan(self.close)
        T = len(self.fechas)
        # último índice con precio de cada ticker (para liquidar deslistados)
        self.ultimo = np.where(valido.any(0), T - 1 - np.argmax(valido[::-1], axis=0), -1)
        self.col = {t: i for i, t in enumerate(self.tickers)}


@lru_cache(maxsize=1)
def cargar() -> Panel:
    z = np.load(PANEL, allow_pickle=True)
    return Panel(**{k: z[k] for k in ("fechas", "tickers", "miembro", "close", "open",
                                       "high", "low", "raw_close", "precio_real", "volume",
                                       "sin_apertura")})


# ------------------------------------------------------------------- indicadores
def _df(a: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(a)


@lru_cache(maxsize=64)
def maximo(n: int) -> np.ndarray:
    return _df(cargar().close).rolling(n, min_periods=max(5, n // 2)).max().to_numpy()


@lru_cache(maxsize=64)
def sma(n: int) -> np.ndarray:
    return _df(cargar().close).rolling(n, min_periods=n).mean().to_numpy()


@lru_cache(maxsize=16)
def desv(n: int) -> np.ndarray:
    return _df(cargar().close).rolling(n, min_periods=n).std().to_numpy()


@lru_cache(maxsize=16)
def rsi(n: int) -> np.ndarray:
    c = _df(cargar().close)
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return (100 - 100 / (1 + up / dn)).to_numpy()


@lru_cache(maxsize=16)
def retorno(n: int) -> np.ndarray:
    c = cargar().close
    out = np.full_like(c, np.nan)
    out[n:] = c[n:] / c[:-n] - 1
    return out


@lru_cache(maxsize=1)
def rachas_bajistas() -> np.ndarray:
    c = cargar().close
    out = np.zeros_like(c)
    for t in range(1, len(c)):
        baja = c[t] < c[t - 1]
        out[t] = np.where(baja, out[t - 1] + 1, 0)
    return out


@lru_cache(maxsize=1)
def ibs() -> np.ndarray:
    p = cargar()
    rango = p.high - p.low
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(rango > 0, (p.close - p.low) / rango, 0.5)


@lru_cache(maxsize=1)
def liquido() -> np.ndarray:
    """Precio REAL >= 5 $ y volumen en dólares medio de 20 días >= 5 M$.

    Si la fuente no informa volumen (CRSP: la mayoría de deslistadas), NO se excluye:
    son miembros del S&P 500 y excluirlas reintroduce sesgo de supervivencia. El
    volumen en dólares es close × volume en la convención de cada fuente (ambos
    ajustados en Yahoo, ambos reales en el resto), por lo que es el real.
    """
    p = cargar()
    dv = _df(p.raw_close * p.volume).rolling(20, min_periods=10).mean().to_numpy()
    sin_volumen = _df(p.volume).rolling(20, min_periods=1).count().to_numpy() == 0
    return (np.nan_to_num(p.precio_real) >= 5) & ((np.nan_to_num(dv) >= 5e6) | sin_volumen)


@lru_cache(maxsize=4)
def regimen_mercado(n: int = 200) -> np.ndarray:
    """SPY por encima de su media de ``n`` sesiones (vector por fecha)."""
    p = cargar()
    spy = pd.Series(p.close[:, p.col["SPY"]])
    return (spy > spy.rolling(n).mean()).to_numpy()


# ---------------------------------------------------------------------- señales
def senal(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (entrada bool T×N, puntuación T×N: mayor = más atractiva)."""
    tipo = cfg["tipo"]
    with np.errstate(invalid="ignore", divide="ignore"):
        if tipo == "dd":  # caída desde el máximo de N sesiones
            dd = cargar().close / maximo(cfg["n"]) - 1
            ent, score = dd <= -cfg["x"], -dd
        elif tipo == "racha":  # n cierres bajistas consecutivos
            r = rachas_bajistas()
            ent, score = r >= cfg["n"], -retorno(cfg["n"])
        elif tipo == "rsi":
            v = rsi(cfg["n"])
            ent, score = v <= cfg["x"], -v
        elif tipo == "ret":  # retorno de N sesiones <= -x
            v = retorno(cfg["n"])
            ent, score = v <= -cfg["x"], -v
        elif tipo == "z":  # distancia a la media en desviaciones típicas
            z = (cargar().close - sma(cfg["n"])) / desv(cfg["n"])
            ent, score = z <= -cfg["x"], -z
        else:
            raise ValueError(tipo)
        if cfg.get("ibs"):
            ent = ent & (ibs() <= cfg["ibs"])
        filtro = cfg.get("filtro", "ninguno")
        if filtro in ("tendencia", "ambos"):
            ent = ent & (cargar().close > sma(200))
        if filtro in ("mercado", "ambos"):
            ent = ent & regimen_mercado()[:, None]
    p = cargar()
    if cfg.get("universo") == "actual":  # SOLO para medir el sesgo de supervivencia
        ent = ent & p.miembro[-1][None, :] & liquido() & ~np.isnan(p.close)
    else:
        ent = ent & p.miembro & liquido() & ~np.isnan(p.close)
    return ent, np.nan_to_num(score, nan=-np.inf)


def salida_senal(cfg: dict) -> np.ndarray | None:
    """Condición de salida evaluada al cierre (además del límite de tiempo)."""
    s = cfg.get("salida", "tiempo")
    c = cargar().close
    with np.errstate(invalid="ignore"):
        if s == "tiempo":
            return None
        if s == "sma5":
            return c > sma(5)
        if s == "rsi2":
            return rsi(2) >= 70
        if s == "sma20":
            return c > sma(20)
    raise ValueError(s)


# ---------------------------------------------------------------------- backtest
@dataclass
class Resultado:
    equity: pd.Series
    trades: pd.DataFrame
    exposicion: pd.Series


def backtest(cfg: dict, coste_bps: float = 10.0, inicio=INICIO, fin=None) -> Resultado:
    p = cargar()
    ent, score = senal(cfg)
    sal = salida_senal(cfg)
    K, H = cfg["k"], cfg["h"]
    objetivo = cfg.get("objetivo")
    stop = cfg.get("stop")
    max_dia = cfg.get("max_dia")  # tope de entradas nuevas por sesión (escalonar en crashes)
    coste = coste_bps / 1e4
    t0 = int(np.searchsorted(p.fechas, inicio))
    t1 = len(p.fechas) if fin is None else int(np.searchsorted(p.fechas, fin))

    cash = 1.0
    pos: dict[int, list] = {}  # j -> [unidades, t_entrada, precio_entrada]
    pend_ent: list[int] = []
    pend_sal: set[int] = set()
    eq = np.empty(t1 - t0)
    expo = np.empty(t1 - t0)
    trades = []
    O, C, CF = p.open, p.close, p.close_ff
    # Reserva: el efectivo ocioso se mantiene en SPY (coste 2 pb por lado sobre el neto).
    reserva = cfg.get("reserva")
    if reserva:
        rj = p.col[reserva]
        r_noche = np.nan_to_num(O[:, rj] / np.roll(CF[:, rj], 1), nan=1.0)
        r_dia = np.nan_to_num(CF[:, rj] / O[:, rj], nan=1.0)
        r_noche[0] = 1.0

    for t in range(t0, t1):
        if reserva and t > t0:
            cash *= r_noche[t]
        cash_ini = cash
        # --- apertura: salidas y luego entradas
        for j in list(pend_sal):
            u, te, pe = pos.pop(j)
            px = O[t, j] if not np.isnan(O[t, j]) else CF[t, j]
            cash += u * px * (1 - coste)
            trades.append((te, t, j, px / pe - 1, t - te))
        pend_sal.clear()
        if pend_ent:
            # tamaño = equity a precios de apertura / K (no el del cierre anterior)
            eq_ap = cash + sum(u * (O[t, j] if not np.isnan(O[t, j]) else CF[t - 1, j])
                               for j, (u, _, _) in pos.items())
            objetivo_hueco = eq_ap / K
        for j in pend_ent:
            px = O[t, j]
            if np.isnan(px) or j in pos:
                continue
            importe = min(objetivo_hueco, cash)
            if importe <= 1e-9:
                break
            u = importe * (1 - coste) / px
            cash -= importe
            pos[j] = [u, t, px]
        pend_ent = []
        if reserva:
            cash -= abs(cash - cash_ini) * 2e-4
            cash *= r_dia[t]

        # --- cierre: liquidar deslistados, valorar
        for j in list(pos):
            if p.ultimo[j] <= t and np.isnan(C[min(t + 1, len(C) - 1), j]) and p.ultimo[j] < len(C) - 1:
                u, te, pe = pos.pop(j)
                px = CF[t, j]
                cash += u * px * (1 - coste)
                trades.append((te, t, j, px / pe - 1, t - te))
        valor = sum(u * CF[t, j] for j, (u, _, _) in pos.items())
        equity = cash + valor
        eq[t - t0] = equity
        expo[t - t0] = valor / equity if equity > 0 else 0.0

        # --- señales de salida
        for j, (u, te, pe) in pos.items():
            dias = t - te + 1
            r = CF[t, j] / pe - 1
            if (dias >= H or (sal is not None and sal[t, j])
                    or (objetivo is not None and r >= objetivo)
                    or (stop is not None and r <= -stop)):
                pend_sal.add(j)

        # --- señales de entrada
        libres = K - (len(pos) - len(pend_sal))
        if max_dia:
            libres = min(libres, max_dia)
        if libres > 0 and t + 1 < t1:
            cand = np.flatnonzero(ent[t])
            if cand.size:
                cand = cand[[j not in pos for j in cand]]
                cand = cand[np.argsort(-score[t, cand], kind="stable")[:libres]]
                pend_ent = [int(j) for j in cand]

    for j, (u, te, pe) in pos.items():  # posiciones abiertas al final, a precio de cierre
        trades.append((te, t1 - 1, j, CF[t1 - 1, j] / pe - 1, t1 - 1 - te))
    idx = pd.DatetimeIndex(p.fechas[t0:t1])
    tr = pd.DataFrame(trades, columns=["t_ent", "t_sal", "j", "ret", "dias"])
    if len(tr):
        tr["entrada"] = p.fechas[tr.t_ent]
        tr["ticker"] = p.tickers[tr.j]
    return Resultado(pd.Series(eq, idx), tr, pd.Series(expo, idx))


# ---------------------------------------------------------------------- métricas
def metricas(eq: pd.Series, trades: pd.DataFrame | None = None, expo: pd.Series | None = None) -> dict:
    r = eq.pct_change().dropna()
    años = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / años) - 1
    vol = r.std() * np.sqrt(252)
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan
    dd = (eq / eq.cummax() - 1).min()
    out = {"cagr": cagr, "vol": vol, "sharpe": sharpe, "maxdd": dd,
           "calmar": cagr / abs(dd) if dd < 0 else np.nan}
    if trades is not None and len(trades):
        out.update({"trades_año": len(trades) / años, "acierto": (trades.ret > 0).mean(),
                    "ret_medio_trade": trades.ret.mean(), "dias_medios": trades.dias.mean()})
    if expo is not None:
        out["exposicion"] = expo.mean()
    return out


def benchmark(ticker: str, inicio=INICIO, fin=None) -> pd.Series:
    p = cargar()
    t0 = int(np.searchsorted(p.fechas, inicio))
    t1 = len(p.fechas) if fin is None else int(np.searchsorted(p.fechas, fin))
    return pd.Series(p.close_ff[t0:t1, p.col[ticker]], pd.DatetimeIndex(p.fechas[t0:t1])).dropna()


def equiponderado_pit(inicio=INICIO, fin=None) -> pd.Series:
    """Equiponderado diario de los miembros PIT (rebalanceo diario, sin costes)."""
    p = cargar()
    t0 = int(np.searchsorted(p.fechas, inicio))
    t1 = len(p.fechas) if fin is None else int(np.searchsorted(p.fechas, fin))
    r = np.full(p.close.shape, np.nan)
    r[1:] = p.close_ff[1:] / p.close_ff[:-1] - 1
    m = p.miembro[:-1] & ~np.isnan(p.close[:-1])
    rr = np.where(m, r[1:], np.nan)
    diaria = np.nanmean(rr, axis=1)
    diaria = np.concatenate([[0.0], diaria])[t0:t1]
    diaria[0] = 0.0
    return pd.Series(np.cumprod(1 + np.nan_to_num(diaria)), pd.DatetimeIndex(p.fechas[t0:t1]))
