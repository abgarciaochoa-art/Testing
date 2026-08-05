"""Tearsheet HTML autocontenido: curvas, tablas y métricas con banda de error.

Este módulo implementa la capa `reports` del contrato (`docs/ARCHITECTURE.md`,
mapa de módulos §2): convierte los artefactos numéricos de `backtest`, `stats` y
`events` en un único documento HTML **autocontenido**:

- CSS inline y SVG dibujado a mano; **cero** referencias a hosts externos (sin
  CDNs, sin fuentes web, sin imágenes remotas). El fichero resultante puede
  archivarse junto al backtest y seguirá abriéndose igual dentro de diez años,
  que es el estándar de reproducibilidad del repo.
- Tema claro y oscuro: variables CSS con `prefers-color-scheme` y un conmutador
  manual (atributo ``data-theme`` en la raíz) que puede forzar cualquiera de los
  dos.
- Regla §3.8 del contrato: **toda métrica de rendimiento se reporta con su
  intervalo de confianza**. La tabla de métricas exige `MetricWithCI`; un número
  desnudo no tiene sitio en ella.

Secciones (cada una con ``id`` estable, ver `SECTION_IDS`):

======================  =====================================================
``equity``              curva de equity (NAV base 1.0)
``drawdown``            drawdown desde máximos (área)
``ic``                  IC por periodo (barras por signo) + resumen Newey–West
``quantiles``           retornos por quintil: acumulado y media anualizada
``caar``                CAAR por tau con banda ±1,96·SE (Kolari–Pynnönen si el
                        productor aplicó su inflactor, `events.aar_caar`)
``grid``                REJILLA de entrada/salida de `EventBacktest.run_grid`
``metrics``             métricas con intervalos de confianza (`MetricWithCI`)
``params``              eco de parámetros para reproducibilidad
======================  =====================================================

Referencias de las convenciones estadísticas que el documento presenta:
Newey y West (1987) para el `t` de la IC media; Bailey y López de Prado (2014)
para el Sharpe probabilístico/deflactado; Kolari y Pynnönen (2010) para los SE
del CAAR con correlación transversal; y la advertencia de multiplicidad de la
rejilla sigue `docs/research/validation_methodology.md` §7 (Benjamini–Hochberg
o Sharpe deflactado con `n_trials` = celdas de la rejilla).
"""

from __future__ import annotations

import datetime as dt
import html as _html
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from earnings_alpha.errors import DataQualityError

__all__ = [  # noqa: RUF022 - orden temático, no alfabético
    "SECTION_IDS",
    "MetricWithCI",
    "sharpe_metrics",
    "ic_metrics",
    "render_tearsheet",
    "write_tearsheet",
]

#: Identificadores estables de sección; los tests y los consumidores (CLI) los
#: usan para comprobar que el documento contiene lo que debe.
SECTION_IDS: tuple[str, ...] = (
    "equity",
    "drawdown",
    "ic",
    "quantiles",
    "caar",
    "grid",
    "metrics",
    "params",
)

# Paleta fija de series: tonos de luminosidad media, legibles sobre fondo claro
# y oscuro sin cambiar de color (el resto del documento sí usa variables CSS).
_SERIES_COLORS: tuple[str, ...] = (
    "#3b82f6",  # azul
    "#f59e0b",  # ámbar
    "#10b981",  # verde
    "#ef4444",  # rojo
    "#8b5cf6",  # violeta
    "#14b8a6",  # teal
    "#e879f9",  # fucsia
    "#94a3b8",  # gris azulado
)
_POS_RGB = "16,185,129"
_NEG_RGB = "239,68,68"

_MAX_POINTS = 600
"""Puntos máximos por polilínea: por encima se submuestrea de forma uniforme
(el submuestreo es solo visual; ninguna métrica se calcula sobre la serie
submuestreada)."""


# ---------------------------------------------------------------------------
# Métricas con intervalo de confianza
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MetricWithCI:
    """Una métrica de rendimiento con su intervalo de confianza.

    Es la unidad de la sección ``metrics`` y la materialización de la regla
    §3.8 del contrato: *"Toda métrica de rendimiento debe reportar su intervalo
    de confianza; un Sharpe sin banda de error no se acepta"*. `ci_low`/`ci_high`
    pueden ser ``None`` solo para magnitudes descriptivas sin banda definida
    (un recuento, una fecha); en ese caso la celda muestra un guion explícito,
    nunca una banda inventada.
    """

    label: str
    value: float
    ci_low: float | None = None
    ci_high: float | None = None
    unit: str = ""
    """``""`` (adimensional), ``"%"`` (se multiplica por 100) o texto libre
    que se añade como sufijo (p. ej. ``"x"``)."""
    note: str = ""
    decimals: int = 3

    def __post_init__(self) -> None:
        if not self.label:
            msg = "MetricWithCI necesita una etiqueta no vacía"
            raise DataQualityError(msg)
        if (self.ci_low is None) != (self.ci_high is None):
            msg = f"métrica {self.label!r}: el IC necesita ambos extremos o ninguno"
            raise DataQualityError(msg)
        if (
            self.ci_low is not None
            and self.ci_high is not None
            and math.isfinite(self.ci_low)
            and math.isfinite(self.ci_high)
            and self.ci_low > self.ci_high
        ):
            msg = (
                f"métrica {self.label!r}: ci_low={self.ci_low} > ci_high={self.ci_high}; "
                "un intervalo invertido es un error del productor, no un formato"
            )
            raise DataQualityError(msg)


def sharpe_metrics(sharpe: object, *, label: str = "Sharpe anualizado") -> list[MetricWithCI]:
    """Convierte un `stats.performance.SharpeResult` (duck-typing) en métricas.

    Devuelve el Sharpe anualizado con su IC (Mertens/Lo/bootstrap, según cómo se
    calculó) y el PSR frente a cero de Bailey y López de Prado (2012): la
    probabilidad de que el Sharpe verdadero sea positivo dado lo observado.
    """
    for attr in ("sharpe_annualized", "ci_low", "ci_high", "psr_zero"):
        if not hasattr(sharpe, attr):
            msg = f"el objeto no parece un SharpeResult: falta el atributo {attr!r}"
            raise DataQualityError(msg)
    return [
        MetricWithCI(
            label=label,
            value=float(sharpe.sharpe_annualized),  # type: ignore[attr-defined]
            ci_low=float(sharpe.ci_low),  # type: ignore[attr-defined]
            ci_high=float(sharpe.ci_high),  # type: ignore[attr-defined]
            note=f"método {getattr(sharpe, 'method', '?')}, T={getattr(sharpe, 'n_obs', '?')}",
        ),
        MetricWithCI(
            label="PSR(Sharpe > 0)",
            value=float(sharpe.psr_zero),  # type: ignore[attr-defined]
            note="Sharpe probabilístico, Bailey y López de Prado (2012)",
        ),
    ]


def ic_metrics(summary: object) -> list[MetricWithCI]:
    """Convierte un `stats.ic.ICSummary` (o su `to_dict()`) en métricas.

    El `t` que acompaña a la IC media es el de Newey–West (1987), el único que el
    repo acepta reportar como *el* `t` (`validation_methodology.md` §3).
    """
    if isinstance(summary, Mapping):
        d = dict(summary)
    elif hasattr(summary, "to_dict"):
        d = dict(summary.to_dict())  # type: ignore[call-arg]
    else:
        msg = "ic_metrics espera un ICSummary o un mapping con sus claves"
        raise DataQualityError(msg)
    for key in ("mean_ic", "ci_low", "ci_high", "t_nw"):
        if key not in d:
            msg = f"resumen de IC sin la clave {key!r}"
            raise DataQualityError(msg)
    horizon = d.get("horizon", "?")
    return [
        MetricWithCI(
            label=f"IC media (h={horizon})",
            value=float(d["mean_ic"]),
            ci_low=float(d["ci_low"]),
            ci_high=float(d["ci_high"]),
            decimals=4,
            note=(
                f"t_NW={float(d['t_nw']):+.2f} (L={d.get('nw_lags', '?')}), "
                f"T={d.get('n_periods', '?')} periodos"
            ),
        ),
        MetricWithCI(
            label="IC IR anualizado",
            value=float(d.get("ic_ir_annualized", float("nan"))),
            decimals=2,
            note="ĪC/σ_IC·√(periodos/año); diagnóstico, no contraste",
        ),
    ]


# ---------------------------------------------------------------------------
# Utilidades de formato
# ---------------------------------------------------------------------------


def _esc(text: object) -> str:
    """Escapa texto arbitrario para HTML."""
    return _html.escape(str(text), quote=True)


def _fmt_value(v: float | None, unit: str = "", decimals: int = 3) -> str:
    """Formatea un valor con su unidad; NaN/None -> guion largo explícito."""
    if v is None:
        return "&mdash;"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return _esc(v)
    if not math.isfinite(x):
        return "&mdash;"
    if unit == "%":
        return f"{100.0 * x:+.{max(decimals - 1, 0)}f}%"
    suffix = _esc(unit) if unit else ""
    return f"{x:+.{decimals}f}{suffix}"


def _fmt_axis(v: float, unit: str) -> str:
    """Etiqueta de eje: compacta, sin signo forzado."""
    if unit == "%":
        pct = 100.0 * v
        return f"{pct:.0f}%" if abs(pct) >= 1 or pct == 0 else f"{pct:.1f}%"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) >= 10:
        return f"{v:.0f}"
    return f"{v:.2f}".rstrip("0").rstrip(".") or "0"


def _downsample(s: pd.Series, max_points: int = _MAX_POINTS) -> pd.Series:
    """Submuestreo uniforme para el trazo SVG (solo visual)."""
    clean = s.dropna()
    if len(clean) <= max_points:
        return clean
    pos = np.unique(np.linspace(0, len(clean) - 1, max_points).round().astype(int))
    return clean.iloc[pos]


def _nice_ticks(lo: float, hi: float, target: int = 5) -> list[float]:
    """Marcas 'bonitas' (1-2-2.5-5·10^k) que cubren [lo, hi]."""
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return [0.0]
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    span = hi - lo
    raw = span / max(target, 1)
    mag = 10.0 ** math.floor(math.log10(raw))
    step = 10.0 * mag
    for mult in (1.0, 2.0, 2.5, 5.0, 10.0):
        if span / (mult * mag) <= target:
            step = mult * mag
            break
    first = math.ceil(lo / step) * step
    out: list[float] = []
    v = first
    while v <= hi + 1e-9 * span:
        out.append(round(v, 12))
        v += step
    return out or [lo]


def _x_float(index: pd.Index, x_kind: Literal["time", "numeric"]) -> np.ndarray:
    """Convierte el eje X (fechas o números) a flotantes para escalar."""
    if x_kind == "time":
        return pd.DatetimeIndex(index).asi8.astype(float)
    return np.asarray(index, dtype=float)


# ---------------------------------------------------------------------------
# Motor SVG (a mano, sin dependencias)
# ---------------------------------------------------------------------------

_W, _H = 900, 280
_ML, _MR, _MT, _MB = 62, 14, 12, 34  # márgenes izq/dcha/arriba/abajo


class _Frame:
    """Marco de coordenadas de un gráfico: escala datos -> píxeles SVG."""

    def __init__(
        self, x_lo: float, x_hi: float, y_lo: float, y_hi: float, height: int = _H
    ) -> None:
        if x_lo == x_hi:
            x_lo, x_hi = x_lo - 0.5, x_hi + 0.5
        if y_lo == y_hi:
            y_lo, y_hi = y_lo - 0.5, y_hi + 0.5
        pad = 0.05 * (y_hi - y_lo)
        self.x_lo, self.x_hi = x_lo, x_hi
        self.y_lo, self.y_hi = y_lo - pad, y_hi + pad
        self.h = height
        self.plot_w = _W - _ML - _MR
        self.plot_h = height - _MT - _MB

    def x(self, v: float) -> float:
        return _ML + (v - self.x_lo) / (self.x_hi - self.x_lo) * self.plot_w

    def y(self, v: float) -> float:
        return _MT + (self.y_hi - v) / (self.y_hi - self.y_lo) * self.plot_h

    def contains_y(self, v: float) -> bool:
        return self.y_lo <= v <= self.y_hi


def _points(frame: _Frame, xs: np.ndarray, ys: np.ndarray) -> str:
    return " ".join(f"{frame.x(x):.1f},{frame.y(y):.1f}" for x, y in zip(xs, ys, strict=True))


def _y_grid(frame: _Frame, unit: str) -> str:
    parts: list[str] = []
    for tick in _nice_ticks(frame.y_lo, frame.y_hi):
        if not frame.contains_y(tick):
            continue
        yy = frame.y(tick)
        parts.append(
            f'<line class="grid" x1="{_ML}" y1="{yy:.1f}" x2="{_W - _MR}" y2="{yy:.1f}"/>'
            f'<text class="tick" x="{_ML - 6}" y="{yy + 3.5:.1f}" text-anchor="end">'
            f"{_esc(_fmt_axis(tick, unit))}</text>"
        )
    return "".join(parts)


def _x_axis_time(frame: _Frame, index: pd.DatetimeIndex, height: int) -> str:
    n_labels = min(6, len(index))
    if n_labels < 1:
        return ""
    pos = np.unique(np.linspace(0, len(index) - 1, n_labels).round().astype(int))
    parts: list[str] = []
    for p in pos:
        ts = index[int(p)]
        xx = frame.x(float(ts.value))
        parts.append(
            f'<text class="tick" x="{xx:.1f}" y="{height - _MB + 16}" '
            f'text-anchor="middle">{ts.strftime("%Y-%m")}</text>'
        )
    return "".join(parts)


def _x_axis_numeric(frame: _Frame, height: int) -> str:
    parts: list[str] = []
    for tick in _nice_ticks(frame.x_lo, frame.x_hi, target=8):
        if not frame.x_lo <= tick <= frame.x_hi:
            continue
        xx = frame.x(tick)
        parts.append(
            f'<text class="tick" x="{xx:.1f}" y="{height - _MB + 16}" '
            f'text-anchor="middle">{_esc(_fmt_axis(tick, ""))}</text>'
        )
    return "".join(parts)


def _zero_line(frame: _Frame) -> str:
    if not frame.contains_y(0.0):
        return ""
    yy = frame.y(0.0)
    return f'<line class="zero" x1="{_ML}" y1="{yy:.1f}" x2="{_W - _MR}" y2="{yy:.1f}"/>'


def _legend(labels: Sequence[str]) -> str:
    if len(labels) <= 1:
        return ""
    items = "".join(
        f'<span class="key"><span class="swatch" '
        f'style="background:{_SERIES_COLORS[i % len(_SERIES_COLORS)]}"></span>{_esc(lab)}</span>'
        for i, lab in enumerate(labels)
    )
    return f'<div class="legend">{items}</div>'


def _figure(svg_body: str, caption: str, height: int, legend: str = "") -> str:
    cap = f"<figcaption>{caption}</figcaption>" if caption else ""
    return (
        '<figure class="chart">'
        f'<svg viewBox="0 0 {_W} {height}" role="img" preserveAspectRatio="xMidYMid meet">'
        f"{svg_body}</svg>{legend}{cap}</figure>"
    )


def _line_chart(
    series: Mapping[str, pd.Series],
    *,
    caption: str = "",
    y_unit: str = "",
    bands: Mapping[str, tuple[pd.Series, pd.Series]] | None = None,
    zero_line: bool = False,
    fill_to_zero: str | None = None,
    x_kind: Literal["time", "numeric"] = "time",
    height: int = _H,
) -> str:
    """Gráfico de líneas SVG con bandas opcionales y submuestreo visual."""
    cleaned = {str(k): _downsample(v) for k, v in series.items() if v is not None}
    cleaned = {k: v for k, v in cleaned.items() if len(v) >= 2}
    if not cleaned:
        msg = "gráfico de líneas sin ninguna serie con al menos 2 puntos finitos"
        raise DataQualityError(msg)
    bands = bands or {}

    xs_all: list[np.ndarray] = []
    ys_all: list[np.ndarray] = []
    for name, s in cleaned.items():
        xs_all.append(_x_float(s.index, x_kind))
        ys_all.append(s.to_numpy(dtype=float))
        if name in bands:
            lo, hi = bands[name]
            ys_all.append(lo.dropna().to_numpy(dtype=float))
            ys_all.append(hi.dropna().to_numpy(dtype=float))
    x_cat = np.concatenate(xs_all)
    y_cat = np.concatenate([y[np.isfinite(y)] for y in ys_all])
    if y_cat.size == 0:
        msg = "gráfico de líneas sin valores finitos"
        raise DataQualityError(msg)
    y_lo, y_hi = float(y_cat.min()), float(y_cat.max())
    if zero_line or fill_to_zero:
        y_lo, y_hi = min(y_lo, 0.0), max(y_hi, 0.0)
    frame = _Frame(float(x_cat.min()), float(x_cat.max()), y_lo, y_hi, height)

    body: list[str] = [_y_grid(frame, y_unit)]
    first = next(iter(cleaned.values()))
    if x_kind == "time":
        body.append(_x_axis_time(frame, pd.DatetimeIndex(first.index), height))
    else:
        body.append(_x_axis_numeric(frame, height))
    if zero_line or fill_to_zero:
        body.append(_zero_line(frame))

    for i, (name, s) in enumerate(cleaned.items()):
        color = _SERIES_COLORS[i % len(_SERIES_COLORS)]
        xs = _x_float(s.index, x_kind)
        ys = s.to_numpy(dtype=float)
        keep = np.isfinite(ys)
        xs, ys = xs[keep], ys[keep]
        if name in bands:
            # Las bandas se reindexan sobre la serie ya submuestreada para que
            # polígono y trazo compartan exactamente las mismas abscisas.
            lo = bands[name][0].astype(float).reindex(s.index).to_numpy(dtype=float)
            hi = bands[name][1].astype(float).reindex(s.index).to_numpy(dtype=float)
            ok = np.isfinite(lo) & np.isfinite(hi)
            if ok.sum() >= 2:
                bx = _x_float(s.index, x_kind)[ok]
                pts = _points(frame, np.concatenate([bx, bx[::-1]]),
                              np.concatenate([lo[ok], hi[ok][::-1]]))
                body.append(f'<polygon class="band" style="fill:{color}" points="{pts}"/>')
        if fill_to_zero == name:
            zx = np.concatenate([[xs[0]], xs, [xs[-1]]])
            zy = np.concatenate([[0.0], ys, [0.0]])
            body.append(
                f'<polygon class="area" style="fill:{color}" '
                f'points="{_points(frame, zx, zy)}"/>'
            )
        body.append(
            f'<polyline class="sline" style="stroke:{color}" '
            f'points="{_points(frame, xs, ys)}"/>'
        )
    return _figure("".join(body), _esc(caption), height, _legend(list(cleaned)))


def _date_bar_chart(
    values: pd.Series,
    *,
    caption: str = "",
    y_unit: str = "",
    mean_line: bool = True,
    height: int = _H,
) -> str:
    """Barras finas por fecha coloreadas por signo (serie de IC por periodo)."""
    s = _downsample(values.astype(float))
    if len(s) < 2:
        msg = "hacen falta al menos 2 periodos para el gráfico de barras temporales"
        raise DataQualityError(msg)
    ys = s.to_numpy(dtype=float)
    y_lo, y_hi = min(float(np.nanmin(ys)), 0.0), max(float(np.nanmax(ys)), 0.0)
    xs = np.arange(len(s), dtype=float)
    frame = _Frame(-0.5, len(s) - 0.5, y_lo, y_hi, height)
    slot = frame.plot_w / len(s)
    bar_w = max(min(slot * 0.8, 12.0), 0.7)

    body: list[str] = [_y_grid(frame, y_unit), _zero_line(frame)]
    y0 = frame.y(0.0)
    for x, v in zip(xs, ys, strict=True):
        if not math.isfinite(v):
            continue
        yy = frame.y(v)
        top, hgt = (yy, y0 - yy) if v >= 0 else (y0, yy - y0)
        rgb = _POS_RGB if v >= 0 else _NEG_RGB
        body.append(
            f'<rect x="{frame.x(x) - bar_w / 2:.1f}" y="{top:.1f}" width="{bar_w:.1f}" '
            f'height="{max(hgt, 0.5):.1f}" style="fill:rgba({rgb},0.85)"/>'
        )
    if mean_line:
        mu = float(np.nanmean(ys))
        if frame.contains_y(mu):
            yy = frame.y(mu)
            body.append(
                f'<line class="meanline" x1="{_ML}" y1="{yy:.1f}" '
                f'x2="{_W - _MR}" y2="{yy:.1f}"/>'
            )
    idx = pd.DatetimeIndex(s.index)
    n_labels = min(6, len(idx))
    for p in np.unique(np.linspace(0, len(idx) - 1, n_labels).round().astype(int)):
        body.append(
            f'<text class="tick" x="{frame.x(float(p)):.1f}" y="{height - _MB + 16}" '
            f'text-anchor="middle">{idx[int(p)].strftime("%Y-%m")}</text>'
        )
    return _figure("".join(body), _esc(caption), height)


def _category_bar_chart(
    values: pd.Series,
    *,
    errors: pd.Series | None = None,
    caption: str = "",
    y_unit: str = "",
    height: int = 250,
) -> str:
    """Barras por categoría (quintiles) con bigotes de error opcionales."""
    s = values.astype(float)
    if len(s) < 2:
        msg = "hacen falta al menos 2 categorías para el gráfico de barras"
        raise DataQualityError(msg)
    ys = s.to_numpy(dtype=float)
    err = errors.reindex(s.index).to_numpy(dtype=float) if errors is not None else None
    y_lo = min(float(np.nanmin(ys - (err if err is not None else 0.0))), 0.0)
    y_hi = max(float(np.nanmax(ys + (err if err is not None else 0.0))), 0.0)
    frame = _Frame(-0.6, len(s) - 0.4, y_lo, y_hi, height)
    slot = frame.plot_w / len(s)
    bar_w = min(slot * 0.55, 90.0)

    body: list[str] = [_y_grid(frame, y_unit), _zero_line(frame)]
    y0 = frame.y(0.0)
    for i, (label, v) in enumerate(s.items()):
        if not math.isfinite(v):
            continue
        cx = frame.x(float(i))
        yy = frame.y(v)
        top, hgt = (yy, y0 - yy) if v >= 0 else (y0, yy - y0)
        rgb = _POS_RGB if v >= 0 else _NEG_RGB
        body.append(
            f'<rect x="{cx - bar_w / 2:.1f}" y="{top:.1f}" width="{bar_w:.1f}" '
            f'height="{max(hgt, 0.5):.1f}" rx="2" style="fill:rgba({rgb},0.75)"/>'
        )
        if err is not None and math.isfinite(err[i]):
            e_lo, e_hi = frame.y(v - err[i]), frame.y(v + err[i])
            body.append(
                f'<line class="whisker" x1="{cx:.1f}" y1="{e_lo:.1f}" '
                f'x2="{cx:.1f}" y2="{e_hi:.1f}"/>'
                f'<line class="whisker" x1="{cx - 5:.1f}" y1="{e_lo:.1f}" '
                f'x2="{cx + 5:.1f}" y2="{e_lo:.1f}"/>'
                f'<line class="whisker" x1="{cx - 5:.1f}" y1="{e_hi:.1f}" '
                f'x2="{cx + 5:.1f}" y2="{e_hi:.1f}"/>'
            )
        body.append(
            f'<text class="tick" x="{cx:.1f}" y="{height - _MB + 16}" '
            f'text-anchor="middle">{_esc(label)}</text>'
        )
    return _figure("".join(body), _esc(caption), height)


# ---------------------------------------------------------------------------
# Tablas
# ---------------------------------------------------------------------------


def _metric_table(metrics: Sequence[MetricWithCI]) -> str:
    rows: list[str] = []
    for m in metrics:
        if m.ci_low is None or m.ci_high is None:
            ci = "&mdash;"
        else:
            ci = (
                f"[{_fmt_value(m.ci_low, m.unit, m.decimals)}, "
                f"{_fmt_value(m.ci_high, m.unit, m.decimals)}]"
            )
        rows.append(
            "<tr>"
            f"<td>{_esc(m.label)}</td>"
            f'<td class="num">{_fmt_value(m.value, m.unit, m.decimals)}</td>'
            f'<td class="num">{ci}</td>'
            f'<td class="note">{_esc(m.note) if m.note else "&mdash;"}</td>'
            "</tr>"
        )
    return (
        '<div class="scroll"><table><thead><tr><th>Métrica</th><th class="num">Valor</th>'
        '<th class="num">IC 95%</th><th>Nota</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _kv_table(params: Mapping[str, object]) -> str:
    rows = "".join(
        f'<tr><td>{_esc(k)}</td><td class="num">{_esc(v)}</td></tr>'
        for k, v in params.items()
    )
    return (
        '<div class="scroll"><table><thead><tr><th>Parámetro</th><th class="num">Valor</th>'
        f"</tr></thead><tbody>{rows}</tbody></table></div>"
    )


_GRID_DETAIL_COLS: tuple[tuple[str, str, int], ...] = (
    # (columna del run_grid, unidad, decimales)
    ("status", "", 0),
    ("n_events", "", 0),
    ("hit_rate", "%", 2),
    ("mean_net", "%", 3),
    ("mean_net_ci_low", "%", 3),
    ("mean_net_ci_high", "%", 3),
    ("median_net", "%", 3),
    ("p05", "%", 3),
    ("p95", "%", 3),
    ("mean_cost", "%", 3),
    ("gap_share_log", "", 2),
    ("avg_holding_sessions", "", 1),
)


def _grid_cell_style(v: float, vmax: float) -> str:
    """Color de fondo de la matriz de calor: alfa proporcional a |valor|."""
    if not math.isfinite(v) or vmax <= 0:
        return ""
    alpha = min(abs(v) / vmax, 1.0) * 0.55
    rgb = _POS_RGB if v >= 0 else _NEG_RGB
    return f"background:rgba({rgb},{alpha:.2f})"


def _grid_section_html(grid: pd.DataFrame) -> str:
    """Matriz de calor entrada x salida + tabla detallada del `run_grid`."""
    g = grid.reset_index() if isinstance(grid.index, pd.MultiIndex) else grid.copy()
    for col in ("entry_offset", "exit_offset"):
        if col not in g.columns:
            msg = (
                f"la rejilla no tiene la columna {col!r}: se espera la salida de "
                "`backtest.event_engine.run_grid`"
            )
            raise DataQualityError(msg)
    if "status" not in g.columns:
        g["status"] = "ok"

    entries = sorted(g["entry_offset"].astype(int).unique())
    exits = sorted(g["exit_offset"].astype(int).unique())
    mean_net = g.set_index(["entry_offset", "exit_offset"]).get("mean_net")
    vmax = (
        float(mean_net.abs().max())
        if mean_net is not None and mean_net.notna().any()
        else 0.0
    )
    status = g.set_index(["entry_offset", "exit_offset"])["status"]

    head = "".join(f'<th class="num">salida T{x:+d}</th>' for x in exits)
    body_rows: list[str] = []
    for e in entries:
        cells: list[str] = [f"<th>entrada T{e:+d}</th>"]
        for x in exits:
            key = (e, x)
            st = str(status.get(key, "ausente"))
            v = float(mean_net.get(key, float("nan"))) if mean_net is not None else float("nan")
            if st == "ok" and math.isfinite(v):
                cells.append(
                    f'<td class="num" style="{_grid_cell_style(v, vmax)}">'
                    f"{_fmt_value(v, '%', 3)}</td>"
                )
            else:
                cells.append(f'<td class="num muted" title="{_esc(st)}">{_esc(st)}</td>')
        body_rows.append(f"<tr>{''.join(cells)}</tr>")
    heat = (
        '<div class="scroll"><table class="heat"><thead><tr><th></th>'
        f"{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table></div>"
    )

    present = [(c, u, d) for c, u, d in _GRID_DETAIL_COLS if c in g.columns]
    dhead = "<th>entrada</th><th>salida</th>" + "".join(
        f'<th class="num">{_esc(c)}</th>' for c, _, _ in present
    )
    drows: list[str] = []
    for _, row in g.sort_values(["entry_offset", "exit_offset"]).iterrows():
        cells = [
            f'<td class="num">T{int(row["entry_offset"]):+d}</td>',
            f'<td class="num">T{int(row["exit_offset"]):+d}</td>',
        ]
        for col, unit, dec in present:
            val = row[col]
            if col == "status":
                cells.append(f"<td>{_esc(val)}</td>")
            elif col == "n_events":
                cells.append(
                    f'<td class="num">{int(val) if pd.notna(val) else "&mdash;"}</td>'
                )
            else:
                cells.append(f'<td class="num">{_fmt_value(val, unit, dec)}</td>')
        drows.append(f"<tr>{''.join(cells)}</tr>")
    detail = (
        f'<div class="scroll"><table><thead><tr>{dhead}</tr></thead>'
        f"<tbody>{''.join(drows)}</tbody></table></div>"
    )
    warning = (
        '<p class="warn">Una rejilla de N combinaciones son N pruebas sobre los mismos '
        "datos: antes de creer la mejor celda, aplicar Benjamini&ndash;Hochberg o el Sharpe "
        "deflactado con n_trials = N (validation_methodology.md &sect;7; Bailey y "
        "L&oacute;pez de Prado 2014).</p>"
    )
    return heat + detail + warning


# ---------------------------------------------------------------------------
# Secciones
# ---------------------------------------------------------------------------


def _section(sec_id: str, title: str, body: str, subtitle: str = "") -> str:
    sub = f'<p class="section-sub">{subtitle}</p>' if subtitle else ""
    return f'<section id="{sec_id}"><h2>{_esc(title)}</h2>{sub}{body}</section>'


def _equity_sections(
    equity: pd.Series | None,
    returns: pd.Series | None,
    drawdown: pd.Series | None,
) -> tuple[str, str]:
    """Secciones de equity y drawdown; deriva lo que falte de lo que haya."""
    nav = equity
    if nav is None and returns is not None:
        nav = (1.0 + returns.astype(float).dropna()).cumprod().rename("nav")
    if nav is None:
        return "", ""
    nav = nav.astype(float).dropna()
    if len(nav) < 2:
        msg = "la curva de equity necesita al menos 2 puntos"
        raise DataQualityError(msg)
    dd = drawdown if drawdown is not None else nav / nav.cummax() - 1.0
    total = float(nav.iloc[-1] / nav.iloc[0] - 1.0)
    eq_html = _section(
        "equity",
        "Curva de equity",
        _line_chart(
            {"NAV neto": nav},
            caption=f"NAV base 1.0, neto de costes. Retorno total {total * 100.0:+.1f}%.",
        ),
    )
    dd_html = _section(
        "drawdown",
        "Drawdown",
        _line_chart(
            {"drawdown": dd.astype(float)},
            caption=f"Caída desde máximos. Máximo drawdown {float(dd.min()) * 100.0:+.1f}%.",
            y_unit="%",
            fill_to_zero="drawdown",
        ),
    )
    return eq_html, dd_html


def _ic_section(ic: pd.Series | None, ic_summary: object | None) -> str:
    if ic is None:
        return ""
    s = ic.astype(float).dropna()
    if len(s) < 2:
        msg = "la serie de IC necesita al menos 2 periodos"
        raise DataQualityError(msg)
    s.index = pd.DatetimeIndex(s.index)
    plotted = s
    note = ""
    if len(s) > 150:
        plotted = s.resample("ME").mean().dropna()
        note = " (agregada a media mensual para el trazo; las métricas usan la serie completa)"
    chart = _date_bar_chart(
        plotted,
        caption=(
            "IC por periodo, coloreada por signo; la línea discontinua es la media"
            + note
            + "."
        ),
    )
    extra = ""
    if ic_summary is not None:
        extra = _metric_table(ic_metrics(ic_summary))
    return _section(
        "ic",
        "IC por periodo",
        chart + extra,
        subtitle=(
            "Correlación de rangos señal-retorno forward por fecha; el t de la media "
            "es Newey&ndash;West (1987), no el IC_IR&middot;&radic;T ingenuo "
            "(validation_methodology.md &sect;3)."
        ),
    )


def _quantile_section(quantile_returns: pd.DataFrame | None) -> str:
    if quantile_returns is None:
        return ""
    q = quantile_returns.astype(float)
    q = q.loc[:, q.notna().any(axis=0)]
    if q.shape[1] < 2 or len(q) < 3:
        msg = "los retornos por quintil necesitan >= 2 cubos y >= 3 fechas"
        raise DataQualityError(msg)
    cum = {str(c): (1.0 + q[c].fillna(0.0)).cumprod() for c in q.columns}
    lines = _line_chart(
        cum,
        caption="Retorno acumulado bruto equiponderado de cada cubo (Q1 = peor señal).",
    )
    n = q.notna().sum()
    mean_ann = q.mean() * 252.0
    se_ann = q.std(ddof=1) * 252.0 / np.sqrt(n.clip(lower=1))
    bars = _category_bar_chart(
        mean_ann,
        errors=1.96 * se_ann,
        y_unit="%",
        caption=(
            "Media anualizada por cubo con bigote ±1,96·SE iid (diagnóstico visual; "
            "la inferencia formal, con dependencia serial, está en la tabla de métricas)."
        ),
    )
    return _section("quantiles", "Retornos por quintil", lines + bars)


def _caar_section(caar: pd.DataFrame | None) -> str:
    if caar is None:
        return ""
    if "caar" not in caar.columns:
        msg = "el panel CAAR debe traer la columna 'caar' (salida de events.aar_caar)"
        raise DataQualityError(msg)
    groups: dict[str, pd.DataFrame] = {}
    if isinstance(caar.index, pd.MultiIndex) and caar.index.nlevels >= 2:
        for g, sub in caar.groupby(level=0, sort=True):
            groups[str(g)] = sub.droplevel(0)
    else:
        groups["todos los eventos"] = caar
    series: dict[str, pd.Series] = {}
    bands: dict[str, tuple[pd.Series, pd.Series]] = {}
    for name, sub in groups.items():
        sub = sub.sort_index()
        series[name] = sub["caar"].astype(float)
        if "caar_se" in sub.columns:
            se = sub["caar_se"].astype(float)
            bands[name] = (sub["caar"] - 1.96 * se, sub["caar"] + 1.96 * se)
    chart = _line_chart(
        series,
        bands=bands,
        x_kind="numeric",
        y_unit="%",
        zero_line=True,
        caption=(
            "CAAR por tau (sesiones respecto al anuncio) con banda ±1,96·SE transversal; "
            "si el productor pasó rho_bar, el SE lleva el inflactor de Kolari&ndash;"
            "Pynn&ouml;nen (2010) por correlación entre eventos contemporáneos."
        ),
    )
    return _section("caar", "CAAR por tau", chart)


def _theme_css() -> str:
    light = (
        "--bg:#f6f7f9;--card:#ffffff;--fg:#1a2333;--muted:#5b6779;--grid:#d7dce4;"
        "--border:#e2e6ec;--accent:#3b82f6;--warn-bg:#fdf3d7;--warn-fg:#7a5b12;"
    )
    dark = (
        "--bg:#11151c;--card:#1a202b;--fg:#e6eaf2;--muted:#94a0b4;--grid:#2b3342;"
        "--border:#29303d;--accent:#60a5fa;--warn-bg:#332a12;--warn-fg:#e8c96a;"
    )
    return f"""
:root {{ {light} }}
:root[data-theme="dark"] {{ {dark} }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ {dark} }} }}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; background: var(--bg); color: var(--fg);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
}}
header {{
  display: flex; flex-wrap: wrap; gap: 12px; align-items: baseline;
  justify-content: space-between; padding: 22px 28px 6px;
}}
h1 {{ margin: 0; font-size: 1.45rem; }}
h2 {{ margin: 0 0 8px; font-size: 1.12rem; }}
.subtitle, .meta, .section-sub, figcaption, .note, .muted {{ color: var(--muted); }}
.subtitle {{ margin: 4px 0 0; }}
.meta {{ font-size: 0.85rem; }}
nav {{ padding: 4px 28px 12px; display: flex; flex-wrap: wrap; gap: 8px; }}
nav a {{
  color: var(--accent); text-decoration: none; font-size: 0.85rem;
  border: 1px solid var(--border); border-radius: 999px; padding: 3px 10px;
  background: var(--card);
}}
main {{ padding: 0 28px 40px; display: grid; gap: 18px; max-width: 1080px; margin: 0 auto; }}
section {{
  background: var(--card); border: 1px solid var(--border); border-radius: 10px;
  padding: 18px 20px;
}}
.section-sub {{ margin: 0 0 10px; font-size: 0.88rem; }}
figure.chart {{ margin: 8px 0; }}
figure.chart svg {{ width: 100%; height: auto; display: block; }}
figcaption {{ font-size: 0.82rem; margin-top: 4px; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 12px; font-size: 0.82rem; margin-top: 6px; }}
.swatch {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; }}
.grid {{ stroke: var(--grid); stroke-width: 1; }}
.zero {{ stroke: var(--muted); stroke-width: 1; stroke-dasharray: 4 3; }}
.meanline {{ stroke: var(--fg); stroke-width: 1.2; stroke-dasharray: 6 4; opacity: 0.8; }}
.whisker {{ stroke: var(--fg); stroke-width: 1.2; opacity: 0.75; }}
.tick {{ fill: var(--muted); font-size: 11px; }}
.sline {{ fill: none; stroke-width: 1.8; }}
.band {{ opacity: 0.14; stroke: none; }}
.area {{ opacity: 0.25; stroke: none; }}
.scroll {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.86rem; }}
th, td {{ padding: 5px 9px; border-bottom: 1px solid var(--border); text-align: left; }}
th {{ color: var(--muted); font-weight: 600; white-space: nowrap; }}
td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
td.note {{ font-size: 0.8rem; }}
table.heat td, table.heat th {{ text-align: center; }}
.warn {{
  background: var(--warn-bg); color: var(--warn-fg); border-radius: 6px;
  padding: 8px 12px; font-size: 0.82rem; margin: 10px 0 0;
}}
button#theme-toggle {{
  background: var(--card); color: var(--fg); border: 1px solid var(--border);
  border-radius: 999px; padding: 3px 12px; cursor: pointer; font-size: 0.82rem;
}}
footer {{ text-align: center; padding: 12px 0 26px; font-size: 0.78rem; color: var(--muted); }}
"""


_TOGGLE_JS = """
(function () {
  var root = document.documentElement;
  var btn = document.getElementById("theme-toggle");
  if (!btn) { return; }
  var order = ["auto", "light", "dark"];
  function current() { return root.getAttribute("data-theme") || "auto"; }
  btn.addEventListener("click", function () {
    var next = order[(order.indexOf(current()) + 1) % order.length];
    if (next === "auto") { root.removeAttribute("data-theme"); }
    else { root.setAttribute("data-theme", next); }
    btn.textContent = "tema: " + next;
  });
})();
"""


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------


def render_tearsheet(
    *,
    title: str,
    subtitle: str = "",
    equity: pd.Series | None = None,
    returns: pd.Series | None = None,
    drawdown: pd.Series | None = None,
    ic: pd.Series | None = None,
    ic_summary: object | None = None,
    quantile_returns: pd.DataFrame | None = None,
    caar: pd.DataFrame | None = None,
    grid: pd.DataFrame | None = None,
    metrics: Sequence[MetricWithCI] | None = None,
    params: Mapping[str, object] | None = None,
    generated_at: dt.datetime | None = None,
) -> str:
    """Compone el tearsheet HTML completo y lo devuelve como cadena.

    Cada sección se incluye solo si su insumo está presente; un tearsheet sin
    ningún insumo es un error (`DataQualityError`), nunca un documento vacío
    con aspecto de informe.

    Parameters
    ----------
    equity:
        Curva de NAV (base 1.0) indexada por fecha. Si falta y hay `returns`,
        se compone como ``prod(1+r)``.
    returns:
        Retornos diarios netos; alternativa a `equity`.
    drawdown:
        Serie de drawdown; si falta se deriva de la curva de equity
        (``nav/cummax(nav) - 1``).
    ic:
        Serie de IC por periodo (índice de fechas), p. ej. la salida de
        `stats.ic.cross_sectional_ic`.
    ic_summary:
        `stats.ic.ICSummary` (o su `to_dict()`): añade la tabla con la IC media,
        su IC 95% y el t de Newey–West.
    quantile_returns:
        Panel diario de retornos brutos por cubo (columnas ``q1..qn``), p. ej.
        `BacktestResult.quantile_returns`.
    caar:
        Salida de `events.aar_caar` (índice tau) o de `events.caar_by_group`
        (MultiIndex ``(grupo, tau)``): curva(s) CAAR con banda ±1,96·SE.
    grid:
        Salida de `backtest.event_engine.run_grid`: la REJILLA de entrada/salida
        del motor de eventos, como matriz de calor + tabla detallada.
    metrics:
        Métricas de cabecera; **cada una con su intervalo de confianza**
        (`MetricWithCI`, regla §3.8 del contrato).
    params:
        Eco de parámetros (semilla, universo, fechas...) para reproducibilidad.
    """
    if not title:
        msg = "el tearsheet necesita un título"
        raise DataQualityError(msg)

    eq_html, dd_html = _equity_sections(equity, returns, drawdown)
    sections: list[tuple[str, str]] = []
    if eq_html:
        sections.append(("equity", eq_html))
    if dd_html:
        sections.append(("drawdown", dd_html))
    ic_html = _ic_section(ic, ic_summary)
    if ic_html:
        sections.append(("ic", ic_html))
    q_html = _quantile_section(quantile_returns)
    if q_html:
        sections.append(("quantiles", q_html))
    caar_html = _caar_section(caar)
    if caar_html:
        sections.append(("caar", caar_html))
    if grid is not None:
        sections.append(
            (
                "grid",
                _section(
                    "grid",
                    "Rejilla de entrada/salida (EventBacktest.run_grid)",
                    _grid_section_html(grid),
                    subtitle=(
                        "Cada celda es una pasada del motor de eventos con esos offsets "
                        "de sesión; mean_net es el retorno neto medio por evento."
                    ),
                ),
            )
        )
    if metrics is not None:
        ms = list(metrics)
        if not ms:
            msg = "la lista de métricas está vacía; omite el argumento si no hay métricas"
            raise DataQualityError(msg)
        for m in ms:
            if not isinstance(m, MetricWithCI):
                msg = (
                    f"cada métrica debe ser MetricWithCI; recibido {type(m).__name__}. "
                    "Regla §3.8: ninguna métrica sin su intervalo de confianza declarado"
                )
                raise DataQualityError(msg)
        sections.append(
            ("metrics", _section("metrics", "Métricas con intervalos de confianza",
                                 _metric_table(ms)))
        )
    if params:
        sections.append(("params", _section("params", "Parámetros", _kv_table(params))))

    if not sections:
        msg = (
            "no hay ninguna sección que renderizar: pasa al menos una de "
            "equity/returns, ic, quantile_returns, caar, grid, metrics o params"
        )
        raise DataQualityError(msg)

    _titles = {
        "equity": "equity",
        "drawdown": "drawdown",
        "ic": "IC",
        "quantiles": "quintiles",
        "caar": "CAAR",
        "grid": "rejilla",
        "metrics": "métricas",
        "params": "parámetros",
    }
    nav = "".join(
        f'<a href="#{sec_id}">{_esc(_titles.get(sec_id, sec_id))}</a>'
        for sec_id, _ in sections
    )
    ts = (generated_at or dt.datetime.now(dt.UTC)).strftime("%Y-%m-%d %H:%M UTC")
    body = "".join(h for _, h in sections)
    sub = f'<p class="subtitle">{_esc(subtitle)}</p>' if subtitle else ""
    return (
        "<!DOCTYPE html>\n"
        '<html lang="es">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{_esc(title)}</title>\n"
        f"<style>{_theme_css()}</style>\n</head>\n<body>\n"
        "<header><div>"
        f"<h1>{_esc(title)}</h1>{sub}"
        "</div>"
        f'<div class="meta">generado {ts} &middot; '
        '<button id="theme-toggle" type="button">tema: auto</button></div>'
        "</header>\n"
        f"<nav>{nav}</nav>\n"
        f"<main>{body}</main>\n"
        "<footer>Documento autocontenido: CSS y SVG inline, sin scripts, fuentes ni "
        "im&aacute;genes de hosts externos. earnings-alpha.</footer>\n"
        f"<script>{_TOGGLE_JS}</script>\n"
        "</body>\n</html>\n"
    )


def write_tearsheet(path: Path | str, **kwargs: object) -> Path:
    """Renderiza el tearsheet y lo escribe en `path` (UTF-8). Devuelve la ruta.

    Crea los directorios intermedios si no existen; los `kwargs` son los de
    `render_tearsheet`.
    """
    out = Path(path)
    html_text = render_tearsheet(**kwargs)  # type: ignore[arg-type]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_text, encoding="utf-8")
    return out
