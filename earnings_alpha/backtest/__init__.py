"""Módulo `backtest`: motores de backtest y modelo de costes (contrato §3.7).

Dos ángulos, dos motores:

- **Cross-section (ángulo A)** — `CrossSectionalBacktest`: carteras de cubos
  sobre una señal `(date, ticker)`, con rebalanceo diario/semanal/mensual,
  retardo de ejecución (señal al cierre → apertura siguiente), universo
  point-in-time por rebalanceo, liquidación de delistings y costes desglosados
  (`CostModel`: spread por tramo de liquidez + impacto √participación +
  comisión + préstamo de cortos; cada parámetro con fuente).
- **Evento (ángulo B)** — `EventBacktest` (`earnings_alpha.backtest.event_engine`):
  entra/sale en offsets de sesión alrededor del anuncio con el gap overnight
  modelado de forma explícita, y `run_grid` para comparar puntos de entrada.

Ejemplo mínimo::

    from earnings_alpha.backtest import CostModel, CrossSectionalBacktest

    bt = CrossSectionalBacktest(universe=membership, sectors=sectors)
    res = bt.run(scores, prices, rebalance="W-FRI", costs=CostModel())
    res.summary()["sharpe"]          # SharpeResult con intervalo de confianza
    res.costs[["spread", "impact"]]  # desglose diario de costes
    res.silent_delistings            # auditoría obligada antes de creer nada
"""

from __future__ import annotations

from earnings_alpha.backtest.costs import (
    BPS,
    DEFAULT_SPREAD_TIERS,
    CostBreakdown,
    CostModel,
    SpreadTier,
)
from earnings_alpha.backtest.engine import (
    BacktestResult,
    CrossSectionalBacktest,
    ExecutionTiming,
    rebalance_schedule,
)
from earnings_alpha.backtest.event_engine import (
    EventBacktest,
    EventBacktestResult,
    run_grid,
    summarize_event_returns,
)
from earnings_alpha.backtest.portfolio import (
    apply_participation_limit,
    apply_turnover_limit,
    assign_quantiles,
    balance_legs,
    build_target_weights,
    cap_weights,
    leg_weights,
    one_way_turnover,
)

__all__ = [  # noqa: RUF022 - orden temático, no alfabético
    # costes
    "BPS",
    "SpreadTier",
    "DEFAULT_SPREAD_TIERS",
    "CostBreakdown",
    "CostModel",
    # cartera
    "assign_quantiles",
    "leg_weights",
    "build_target_weights",
    "cap_weights",
    "balance_legs",
    "one_way_turnover",
    "apply_turnover_limit",
    "apply_participation_limit",
    # motor cross-section
    "ExecutionTiming",
    "rebalance_schedule",
    "BacktestResult",
    "CrossSectionalBacktest",
    # motor de eventos
    "EventBacktest",
    "EventBacktestResult",
    "summarize_event_returns",
    "run_grid",
]
