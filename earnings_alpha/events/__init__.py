"""Ángulo B: estudio de la ventana de resultados y detección de flujo informado.

Este paquete implementa el contrato §3.5 de `docs/ARCHITECTURE.md`. Núcleo del
estudio de eventos:

- `event_windows` / `window_overlap_summary` (`events.window`): expansión de cada
  anuncio a tiempo-evento `tau` en días de **sesión**, con `event_id` estable y el
  solapamiento entre ventanas consecutivas del mismo emisor expuesto, nunca
  oculto.
- `abnormal_returns` (`events.eventstudy`): AR/CAR/BHAR por evento y `tau` bajo
  los modelos *mean*, *market* y *ff3*, con la ventana de estimación terminando
  estrictamente antes de la de evento.
- `aar_caar`, `caar_by_group`, `quantile_groups`: agregación con errores estándar
  (el insumo del gráfico CAAR por quintil de SUE).
- `patell_test`, `bmp_test`, `corrado_rank_test`, `estimate_rho_bar`: contrastes
  de significancia paramétricos y de rango, con el ajuste de Kolari-Pynnönen por
  correlación transversal.
- `overnight_decomposition`: partición explícita del retorno del día del evento en
  gap de apertura + intradía (crítica para anuncios AMC, cuya reacción se negocia
  en el gap de la sesión siguiente).

**Nota legal y metodológica (obligatoria por el contrato §3.5).** Todo lo que este
paquete calcula —y todo lo que calcularán sus detectores de negociación informada
(`PreEventFeatures`)— deriva exclusivamente de **datos públicos**: precios y
volúmenes consolidados, cadenas de opciones, short interest agregado de FINRA y
formularios Form 4 ya presentados en EDGAR. El objetivo es detectar la *huella
estadística* que el comportamiento de otros participantes deja en variables
observables, no acceder a información material no pública. Ninguna señal de este
paquete requiere ni admite información privilegiada. Ver
`docs/research/informed_trading.md`.
"""

from earnings_alpha.events.eventstudy import (
    EventTestResult,
    aar_caar,
    abnormal_returns,
    bmp_test,
    caar_by_group,
    car_window,
    corrado_rank_test,
    estimate_rho_bar,
    overnight_decomposition,
    patell_test,
    quantile_groups,
)
from earnings_alpha.events.window import (
    event_windows,
    normalize_events,
    window_overlap_summary,
)

__all__ = [  # noqa: RUF022 - orden temático, no alfabético
    # tiempo-evento
    "event_windows",
    "window_overlap_summary",
    "normalize_events",
    # retornos anormales
    "abnormal_returns",
    "car_window",
    "aar_caar",
    "caar_by_group",
    "quantile_groups",
    # contrastes
    "patell_test",
    "bmp_test",
    "corrado_rank_test",
    "estimate_rho_bar",
    "EventTestResult",
    # gap overnight
    "overnight_decomposition",
]
