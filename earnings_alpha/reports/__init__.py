"""Módulo `reports`: tearsheets HTML autocontenidos (contrato §2, capa reports).

Convierte los artefactos numéricos de `backtest`, `stats` y `events` en un único
documento HTML archivable: CSS inline, SVG dibujado a mano, **sin ninguna
referencia a hosts externos** y con tema claro/oscuro. La regla que gobierna la
capa es la §3.8 del contrato: toda métrica de rendimiento viaja con su intervalo
de confianza (`MetricWithCI`).

Uso típico::

    from earnings_alpha.reports import render_tearsheet, write_tearsheet, MetricWithCI

    html = render_tearsheet(
        title="SUE analistas - semanal",
        returns=result.returns,
        ic=ic_series,
        ic_summary=ic_summary,
        quantile_returns=result.quantile_returns,
        caar=caar_frame,
        grid=grid_table,
        metrics=[*sharpe_metrics(result.sharpe())],
        params=result.params,
    )
"""

from earnings_alpha.reports.tearsheet import (
    SECTION_IDS,
    MetricWithCI,
    ic_metrics,
    render_tearsheet,
    sharpe_metrics,
    write_tearsheet,
)

__all__ = [  # noqa: RUF022 - orden temático, no alfabético
    "SECTION_IDS",
    "MetricWithCI",
    "sharpe_metrics",
    "ic_metrics",
    "render_tearsheet",
    "write_tearsheet",
]
