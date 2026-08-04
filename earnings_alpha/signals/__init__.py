"""Módulo `signals`: acondicionamiento y combinación de señales (contrato §3.6).

Es la capa que va de un factor crudo —un SUE, un run-up de volumen, un
`vol_spread` de opciones— a una puntuación comparable entre nombres y entre
fechas, apta para entrar en `backtest`. Dos submódulos:

- `transforms`: `zscore`, `winsorize`, `rank_pct`, `demean`/`demedian`,
  `residualize`, `neutralize` y la receta canónica `condition_factor`.
- `combine`: `fixed_weight_combine`, `ic_weighted_combine` (ventana expansiva
  point-in-time), `orthogonalize` (Gram-Schmidt cross-section) y el despachador
  `combine`.

Dos invariantes gobiernan todo el módulo y conviene tenerlos presentes antes de
usarlo:

1. **Todo se calcula por fecha, sobre la sección cruzada.** Nada mira más allá
   del día que se está transformando. Estandarizar o winsorizar con los momentos
   de todo el panel es usar la distribución futura de la señal para decidir qué
   es un outlier hoy (`docs/research/pit_and_biases.md` §12.7).
2. **Ningún NaN se rellena sin declararlo.** Cada función expone `nan_policy`
   (ver `NaNPolicy`) y el valor por defecto es propagar. Una fecha con sección
   cruzada insuficiente sale como NaN —o lanza `InsufficientHistory` si ninguna
   fecha llega al mínimo—, nunca como un número calculado sobre cuatro nombres.

La ponderación por IC merece un aviso propio: el IC de la fecha `t` requiere el
retorno futuro de `h` sesiones y por tanto **no es observable en `t`**.
`combine.expanding_ic_weights` fecha cada IC por su fecha de realización y solo
usa los estrictamente anteriores a la fecha de decisión; el test
`test_ic_weights_no_lookahead` lo verifica alterando el futuro y exigiendo que
el pasado no se mueva.

Ejemplo
-------
>>> from earnings_alpha.signals import condition_factor, combine
>>> score = condition_factor(raw_factor, exposures, by=["sector", "size"])  # doctest: +SKIP
>>> mixed = combine(panel, "weights", weights={"sue": 0.6, "pead": 0.4})     # doctest: +SKIP
"""

from __future__ import annotations

from earnings_alpha.signals.combine import (
    MIN_CROSS_SECTION,
    combine,
    cross_section_ic,
    expanding_ic_weights,
    fixed_weight_combine,
    ic_weighted_combine,
    ic_weighted_weights,
    observability_dates,
    orthogonalize,
    symmetric_orthogonalize,
)
from earnings_alpha.signals.transforms import (
    MAD_TO_SIGMA,
    InsufficientPolicy,
    NaNPolicy,
    align_panel_like,
    check_panel,
    condition_factor,
    cross_section_count,
    demean,
    demedian,
    neutralize,
    rank_pct,
    residualize,
    winsorize,
    zscore,
)

__all__ = [  # noqa: RUF022 - orden temático, no alfabético
    # políticas y constantes
    "NaNPolicy",
    "InsufficientPolicy",
    "MAD_TO_SIGMA",
    "MIN_CROSS_SECTION",
    # utilidades de panel
    "check_panel",
    "align_panel_like",
    "cross_section_count",
    # transformaciones
    "zscore",
    "winsorize",
    "rank_pct",
    "demean",
    "demedian",
    "residualize",
    "neutralize",
    "condition_factor",
    # combinación
    "cross_section_ic",
    "observability_dates",
    "expanding_ic_weights",
    "ic_weighted_weights",
    "ic_weighted_combine",
    "fixed_weight_combine",
    "orthogonalize",
    "symmetric_orthogonalize",
    "combine",
]
