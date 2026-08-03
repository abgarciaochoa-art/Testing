"""earnings-alpha: investigación de estrategias sobre fundamentales y eventos de resultados.

Dos ángulos complementarios sobre el mismo universo (S&P 500 completo, point-in-time):

- **Ángulo A (continuo)**: factores fundamentales cross-section evaluados todo el año
  (`earnings_alpha.factors`).
- **Ángulo B (evento)**: ventana en torno a la presentación de resultados y detección
  de la huella de negociación informada previa (`earnings_alpha.events`).
"""

from earnings_alpha.types import CIK, Bar, EarningsEvent, Session, Ticker

__version__ = "0.1.0"
__all__ = ["CIK", "Bar", "EarningsEvent", "Session", "Ticker", "__version__"]
