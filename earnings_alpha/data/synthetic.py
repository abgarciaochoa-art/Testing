"""Generador de mercado sintético: banco de pruebas offline de toda la plataforma.

Cumple dos funciones que el contrato (`docs/ARCHITECTURE.md`, secciones 0.5 y 3.3)
considera irrenunciables:

1. **Probar sin red.** El entorno de desarrollo no alcanza ninguna API financiera.
   Todo módulo -- factores, estudio de eventos, backtest, estadística -- se valida
   contra `SyntheticMarket`, que produce paneles con la misma forma, los mismos
   nombres de columna y la misma semántica point-in-time que los proveedores reales.
2. **Validar que el detector detecta.** El generador inyecta, en una fracción
   conocida de los eventos (`leak_fraction`), la huella estadística de negociación
   informada previa al anuncio: run-up de volumen, deriva de precio en el signo de
   la sorpresa, sesgo en el flujo de opciones y aumento de la cuota off-exchange.
   `leaked_event_ids()` devuelve la verdad-terreno, de modo que la precisión y el
   recall de `events.PreEventFeatures` + `SurpriseModel` son *medibles* y no una
   cuestión de fe. En el resto de eventos no hay ninguna inyección: son el grupo de
   control con el que se calibra la tasa de falsos positivos.

Estructura del modelo
---------------------

* **Precios.** Modelo de factores con mercado, tamaño y valor (Fama-French 1993) más
  un factor sectorial y un residuo idiosincrásico. Las innovaciones son t de Student
  estandarizadas (colas gruesas, Fama 1965; Mandelbrot 1963) y la volatilidad del
  mercado sigue un proceso log-AR(1) que genera agrupamiento de volatilidad
  (Bollerslev 1986). La volatilidad idiosincrásica es heterogénea por sector.
* **Volumen.** Log-volumen AR(1) -- la persistencia del volumen es un hecho
  estilizado clásico (Gallant, Rossi y Tauchen 1992) -- con estacionalidad semanal,
  bump de vencimiento trimestral y la relación volumen-volatilidad (Karpoff 1987).
* **Fundamentales.** Cuenta de resultados, flujo de caja y balance construidos por
  arrastre (*roll-forward*) de modo que la identidad ``activo = pasivo + patrimonio``
  se cumple **exactamente** en cada trimestre y el flujo de caja explica la variación
  de tesorería. Eso hace que Piotroski (2000) F-score, los accruals de Sloan (1996) y
  la calidad de beneficios CFO/NI tengan significado sobre estos datos.
* **Eventos.** Un anuncio por trimestre fiscal, con retardo realista sobre el cierre
  de trimestre, asignación BMO/AMC persistente por empresa y consenso de analistas
  que se revisa a la baja a lo largo del trimestre hacia un objetivo batible
  (*walk-down*, Richardson, Teoh y Wysocki 2004). La sorpresa explica parcialmente
  el retorno del día del evento (coeficiente de respuesta, Ball y Brown 1968) y el
  drift posterior (PEAD, Bernard y Thomas 1989).
* **Opciones.** Superficie con smile y estructura temporal en la que la varianza del
  anuncio entra de forma aditiva (Dubinsky y Johannes 2006): eso produce, sin
  imponerlo a mano, la subida de IV según se acerca el evento y su colapso el día en
  que la noticia se publica (*IV crush*). El diferencial IV call - IV put replica el
  `vol_spread` de Cremers y Weinbaum (2010).

Lo que este generador **no** expone
-----------------------------------
Ningún panel observable (precios, eventos, estimaciones, opciones) contiene la
etiqueta de filtración ni la sorpresa latente. La verdad-terreno vive únicamente en
`ground_truth()` y `leaked_event_ids()`. Es deliberado: si la etiqueta viajase con el
feed, cualquier modelo entrenado sobre él estaría contaminado y sus métricas no
significarían nada. Por la misma razón el panel diario de opciones publica los *días*
que faltan para el anuncio -- el calendario de resultados es público por adelantado --
pero jamás su magnitud.
* **Flujo.** Short interest quincenal con el retardo de publicación de FINRA y cuota
  off-exchange semanal con el retardo de la transparencia ATS.

Nota sobre `available_at`
-------------------------
Todas las tablas de hechos (fundamentales, estimaciones, short interest,
off-exchange) llevan `available_at`: el instante en que el dato fue *públicamente
conocible*. No coincide con `period_end` ni con la fecha de referencia del dato, y
usarlo mal es la vía más rápida a un backtest con look-ahead. El generador aplica
retardos de publicación realistas precisamente para que los tests de otros módulos
los sufran: en particular, la cuota off-exchange de la semana previa a un evento no
es observable antes del propio evento, y una estrategia que la use en tiempo real es
irrealizable aunque el detector la encuentre.

Determinismo
------------
Misma semilla -> mismo panel, byte a byte. Cada bloque estocástico consume un flujo
independiente derivado por BLAKE2b de ``(seed, etiqueta...)``; no se usa `hash()` de
Python, cuya aleatorización por proceso rompería la reproducibilidad entre
ejecuciones.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Final, TypeAlias

import numpy as np
import pandas as pd
from scipy.signal import lfilter
from scipy.special import ndtr

from earnings_alpha.config import Settings, get_settings
from earnings_alpha.errors import (
    ConfigError,
    DataQualityError,
    InsufficientHistory,
    ProviderUnavailable,
)
from earnings_alpha.pit import (
    TradingCalendar,
    eastern_to_utc,
    get_calendar,
    tradable_dates,
)
from earnings_alpha.types import (
    CIK,
    Bar,
    EarningsEvent,
    Session,
    Ticker,
    normalize_cik,
    normalize_ticker,
)

__all__ = [
    "SECTOR_PROFILES",
    "LeakSpec",
    "SectorProfile",
    "SyntheticConfig",
    "SyntheticMarket",
    "make_synthetic_market",
]

DateLike: TypeAlias = dt.date | dt.datetime | str | pd.Timestamp

# Días naturales de un trimestre; se usa como denominador de los ratios de capital
# circulante (DSO, DIO, DPO) expresados en días.
_QUARTER_DAYS: Final[float] = 91.25
_TRADING_DAYS: Final[float] = 252.0
_SQRT_TRADING: Final[float] = math.sqrt(_TRADING_DAYS)
_MIN_IV: Final[float] = 0.05
_MAX_IV: Final[float] = 3.0


# ---------------------------------------------------------------------------
# Perfiles sectoriales
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SectorProfile:
    """Parámetros estructurales de un sector GICS.

    Los valores son órdenes de magnitud plausibles para el S&P 500 (beta y
    volatilidad de los últimos ciclos, márgenes y estacionalidad típicos del
    sector). No pretenden replicar ninguna empresa concreta: su función es que los
    factores cross-section tengan dispersión sectorial realista y que la
    neutralización por sector (`signals.neutralize`) tenga algo que neutralizar.
    """

    beta: float
    """Carga sobre el factor de mercado."""
    idio_vol: float
    """Volatilidad idiosincrásica anualizada."""
    factor_vol: float
    """Volatilidad anualizada del factor sectorial."""
    revenue_growth: float
    """Crecimiento anual medio de ingresos."""
    gross_margin: float
    opex_ratio: float
    """Gastos operativos (sin amortización) sobre ingresos."""
    da_ratio: float
    """Amortización trimestral sobre ingresos; fija la tasa efectiva sobre el
    inmovilizado neto al inicio de la simulación."""
    capex_ratio: float
    """Capex sobre ingresos."""
    sbc_ratio: float
    """Retribución en acciones sobre ingresos."""
    tax_rate: float
    payout: float
    """Fracción del beneficio distribuida en dividendos."""
    buyback_ratio: float
    """Fracción del flujo de caja libre remanente dedicada a recompras."""
    price_to_sales: float
    equity_ratio: float
    """Patrimonio sobre activo total."""
    ppe_to_revenue: float
    """Inmovilizado neto en múltiplos de ingresos trimestrales."""
    goodwill_to_revenue: float
    dso: float
    dio: float
    dpo: float
    seasonality: tuple[float, float, float, float]
    """Factores multiplicativos de ingresos por trimestre natural (media 1)."""
    jump_vol: float
    """Desviación típica del salto de precio el día del anuncio."""
    short_base: float
    """Interés corto medio como fracción del capital."""
    offex_base: float
    """Cuota media de volumen negociado fuera de bolsa."""
    option_liquidity: float
    """Escala relativa del volumen y el interés abierto de opciones."""


SECTOR_PROFILES: Final[dict[str, SectorProfile]] = {
    "Information Technology": SectorProfile(
        beta=1.15, idio_vol=0.250, factor_vol=0.105, revenue_growth=0.12,
        gross_margin=0.58, opex_ratio=0.38, da_ratio=0.045, capex_ratio=0.055,
        sbc_ratio=0.055, tax_rate=0.17, payout=0.14, buyback_ratio=0.55,
        price_to_sales=5.4, equity_ratio=0.46, ppe_to_revenue=1.1,
        goodwill_to_revenue=1.6, dso=58.0, dio=32.0, dpo=45.0,
        seasonality=(0.94, 0.97, 1.00, 1.09), jump_vol=0.061,
        short_base=0.023, offex_base=0.405, option_liquidity=1.45,
    ),
    "Health Care": SectorProfile(
        beta=0.86, idio_vol=0.210, factor_vol=0.083, revenue_growth=0.070,
        gross_margin=0.60, opex_ratio=0.42, da_ratio=0.050, capex_ratio=0.045,
        sbc_ratio=0.030, tax_rate=0.16, payout=0.28, buyback_ratio=0.45,
        price_to_sales=3.1, equity_ratio=0.40, ppe_to_revenue=1.3,
        goodwill_to_revenue=2.1, dso=62.0, dio=78.0, dpo=52.0,
        seasonality=(0.97, 1.00, 0.99, 1.04), jump_vol=0.054,
        short_base=0.026, offex_base=0.398, option_liquidity=1.05,
    ),
    "Financials": SectorProfile(
        beta=1.07, idio_vol=0.195, factor_vol=0.094, revenue_growth=0.050,
        gross_margin=0.84, opex_ratio=0.55, da_ratio=0.030, capex_ratio=0.020,
        sbc_ratio=0.020, tax_rate=0.22, payout=0.32, buyback_ratio=0.60,
        price_to_sales=2.7, equity_ratio=0.30, ppe_to_revenue=0.7,
        goodwill_to_revenue=1.0, dso=45.0, dio=4.0, dpo=30.0,
        seasonality=(1.00, 1.01, 0.99, 1.00), jump_vol=0.045,
        short_base=0.021, offex_base=0.385, option_liquidity=0.95,
    ),
    "Consumer Discretionary": SectorProfile(
        beta=1.21, idio_vol=0.242, factor_vol=0.099, revenue_growth=0.070,
        gross_margin=0.38, opex_ratio=0.26, da_ratio=0.040, capex_ratio=0.048,
        sbc_ratio=0.022, tax_rate=0.22, payout=0.18, buyback_ratio=0.55,
        price_to_sales=1.9, equity_ratio=0.34, ppe_to_revenue=1.6,
        goodwill_to_revenue=0.8, dso=26.0, dio=88.0, dpo=62.0,
        seasonality=(0.92, 0.95, 0.97, 1.16), jump_vol=0.066,
        short_base=0.038, offex_base=0.412, option_liquidity=1.15,
    ),
    "Communication Services": SectorProfile(
        beta=1.06, idio_vol=0.234, factor_vol=0.099, revenue_growth=0.080,
        gross_margin=0.53, opex_ratio=0.28, da_ratio=0.130, capex_ratio=0.085,
        sbc_ratio=0.045, tax_rate=0.18, payout=0.20, buyback_ratio=0.50,
        price_to_sales=3.3, equity_ratio=0.45, ppe_to_revenue=2.3,
        goodwill_to_revenue=1.9, dso=52.0, dio=12.0, dpo=48.0,
        seasonality=(0.96, 0.99, 0.99, 1.06), jump_vol=0.063,
        short_base=0.028, offex_base=0.402, option_liquidity=1.20,
    ),
    "Industrials": SectorProfile(
        beta=1.05, idio_vol=0.187, factor_vol=0.083, revenue_growth=0.050,
        gross_margin=0.33, opex_ratio=0.20, da_ratio=0.040, capex_ratio=0.040,
        sbc_ratio=0.015, tax_rate=0.21, payout=0.30, buyback_ratio=0.45,
        price_to_sales=2.1, equity_ratio=0.33, ppe_to_revenue=1.5,
        goodwill_to_revenue=1.2, dso=55.0, dio=72.0, dpo=58.0,
        seasonality=(0.96, 1.01, 0.99, 1.04), jump_vol=0.049,
        short_base=0.024, offex_base=0.392, option_liquidity=0.85,
    ),
    "Consumer Staples": SectorProfile(
        beta=0.63, idio_vol=0.140, factor_vol=0.066, revenue_growth=0.035,
        gross_margin=0.40, opex_ratio=0.26, da_ratio=0.035, capex_ratio=0.038,
        sbc_ratio=0.012, tax_rate=0.22, payout=0.55, buyback_ratio=0.35,
        price_to_sales=1.7, equity_ratio=0.31, ppe_to_revenue=1.7,
        goodwill_to_revenue=1.5, dso=32.0, dio=68.0, dpo=64.0,
        seasonality=(0.96, 1.01, 1.01, 1.02), jump_vol=0.038,
        short_base=0.022, offex_base=0.388, option_liquidity=0.70,
    ),
    "Energy": SectorProfile(
        beta=1.14, idio_vol=0.273, factor_vol=0.138, revenue_growth=0.055,
        gross_margin=0.29, opex_ratio=0.08, da_ratio=0.110, capex_ratio=0.115,
        sbc_ratio=0.008, tax_rate=0.25, payout=0.42, buyback_ratio=0.40,
        price_to_sales=1.3, equity_ratio=0.48, ppe_to_revenue=4.2,
        goodwill_to_revenue=0.3, dso=38.0, dio=30.0, dpo=52.0,
        seasonality=(1.00, 1.02, 1.01, 0.97), jump_vol=0.052,
        short_base=0.030, offex_base=0.386, option_liquidity=0.90,
    ),
    "Utilities": SectorProfile(
        beta=0.50, idio_vol=0.133, factor_vol=0.072, revenue_growth=0.030,
        gross_margin=0.46, opex_ratio=0.16, da_ratio=0.140, capex_ratio=0.220,
        sbc_ratio=0.006, tax_rate=0.16, payout=0.65, buyback_ratio=0.05,
        price_to_sales=2.4, equity_ratio=0.30, ppe_to_revenue=6.5,
        goodwill_to_revenue=0.5, dso=42.0, dio=22.0, dpo=48.0,
        seasonality=(1.08, 0.93, 1.06, 0.93), jump_vol=0.031,
        short_base=0.026, offex_base=0.372, option_liquidity=0.55,
    ),
    "Real Estate": SectorProfile(
        beta=0.96, idio_vol=0.187, factor_vol=0.088, revenue_growth=0.045,
        gross_margin=0.62, opex_ratio=0.24, da_ratio=0.280, capex_ratio=0.150,
        sbc_ratio=0.014, tax_rate=0.05, payout=0.75, buyback_ratio=0.10,
        price_to_sales=6.5, equity_ratio=0.42, ppe_to_revenue=9.0,
        goodwill_to_revenue=0.4, dso=20.0, dio=2.0, dpo=26.0,
        seasonality=(0.99, 1.00, 1.01, 1.00), jump_vol=0.036,
        short_base=0.032, offex_base=0.378, option_liquidity=0.50,
    ),
    "Materials": SectorProfile(
        beta=1.10, idio_vol=0.203, factor_vol=0.094, revenue_growth=0.040,
        gross_margin=0.26, opex_ratio=0.10, da_ratio=0.070, capex_ratio=0.070,
        sbc_ratio=0.010, tax_rate=0.21, payout=0.35, buyback_ratio=0.35,
        price_to_sales=1.6, equity_ratio=0.40, ppe_to_revenue=2.6,
        goodwill_to_revenue=0.9, dso=48.0, dio=76.0, dpo=54.0,
        seasonality=(0.99, 1.03, 1.00, 0.98), jump_vol=0.050,
        short_base=0.027, offex_base=0.390, option_liquidity=0.65,
    ),
}

_DEFAULT_SECTOR: Final[str] = "Industrials"


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LeakSpec:
    """Amplitud de cada canal de la huella pre-anuncio.

    Los valores por defecto están calibrados para que la huella sea *detectable con
    potencia razonable* (t de dos muestras del orden de 5-10 con unos pocos cientos
    de eventos) sin ser trivial evento a evento: el ratio señal/ruido de un evento
    aislado es cercano a 1, que es el régimen interesante para un detector.
    """

    window_days: tuple[int, int] = (5, 15)
    """Rango del número de sesiones que dura la huella antes del evento."""
    intensity: tuple[float, float] = (0.6, 1.6)
    """Rango uniforme del multiplicador de intensidad por evento."""
    volume_log_amp: float = 0.40
    """Incremento máximo del log-volumen en el pico del run-up."""
    price_drift_total: float = 0.035
    """Deriva acumulada de precio en la ventana, con el signo de la sorpresa."""
    option_volume_log_amp: float = 0.55
    """Incremento máximo del log-volumen de opciones del lado favorecido."""
    oi_buildup: float = 0.35
    """Incremento relativo del interés abierto del lado favorecido."""
    vol_spread_shift: float = 0.022
    """Desplazamiento de (IV call - IV put) en el signo de la sorpresa."""
    skew_shift: float = 0.020
    """Desplazamiento del skew 25-delta en el signo contrario a la sorpresa."""
    offex_share_shift: float = 0.030
    """Incremento absoluto de la cuota off-exchange en las semanas afectadas."""
    short_interest_shift: float = 0.22
    """Incremento relativo del interés corto cuando la sorpresa es negativa."""
    revision_shift: float = 0.55
    """Deriva extra del consenso en la ventana, en unidades de dispersión."""


@dataclass(frozen=True, slots=True)
class SyntheticConfig:
    """Parámetros del generador. Todo es ajustable; los defaults son plausibles."""

    # --- factores de mercado
    market_vol: float = 0.16
    market_drift: float = 0.07
    vol_persistence: float = 0.965
    vol_of_vol: float = 0.30
    smb_vol: float = 0.075
    hml_vol: float = 0.085
    risk_free: float = 0.025
    t_df_market: float = 5.0
    t_df_sector: float = 6.0
    t_df_idio: float = 4.0
    max_abs_log_return: float = 0.35
    """Recorte del retorno logarítmico diario; evita precios absurdos con t(4)."""

    # --- respuesta a resultados
    event_response: float = 0.019
    """Retorno del día del evento por unidad de sorpresa estandarizada."""
    pead_total: float = 0.025
    """Drift acumulado a 60 sesiones por unidad de sorpresa estandarizada. El orden
    de magnitud es el de Bernard y Thomas (1989): 4-6 % entre deciles extremos."""
    pead_horizon: int = 60
    pead_decay: float = 25.0
    event_gap_share: float = 0.80
    """Fracción del retorno del día del evento que se realiza en el gap de apertura."""

    # --- volumen
    volume_ar: float = 0.86
    volume_sigma: float = 0.20
    """Innovación del log-volumen; con `volume_ar` da una desviación estacionaria
    cercana a 0.39, es decir, oscilaciones típicas de +-50 % del volumen medio."""
    volume_dow: tuple[float, float, float, float, float] = (
        0.02, 0.03, 0.01, -0.01, -0.05,
    )
    """Efecto de día de la semana sobre el log-volumen (lunes a viernes)."""
    volume_witching: float = 0.35
    """Bump de log-volumen el tercer viernes de marzo, junio, septiembre y diciembre."""
    volume_abs_return: float = 3.2
    """Sensibilidad del log-volumen al retorno absoluto estandarizado."""
    volume_event_profile: tuple[float, ...] = (1.15, 0.62, 0.36, 0.20, 0.10)
    """Perfil del pico de volumen desde tau=0 en adelante."""

    # --- fundamentales
    history_quarters: int = 12
    """Trimestres generados antes de `start` para que SUE y crecimientos existan."""
    growth_persistence: float = 0.72
    growth_sigma: float = 0.016
    margin_persistence: float = 0.86
    margin_sigma: float = 0.0055
    revenue_noise: float = 0.020
    debt_rate: float = 0.045
    min_cash_ratio: float = 0.15
    """Tesorería mínima en fracción de ingresos trimestrales; por debajo se dispone
    de la línea de crédito (aumenta deuda a corto y caja a la vez)."""

    # --- calendario de resultados
    report_delay_mean: float = 28.0
    report_delay_sd: float = 6.0
    report_delay_jitter: float = 2.5
    report_delay_bounds: tuple[int, int] = (18, 48)
    amc_probability: float = 0.55
    session_flip_probability: float = 0.07
    filing_lag_days: tuple[int, int] = (1, 24)
    """Retardo del 10-Q/10-K respecto a la nota de prensa."""

    # --- consenso
    estimate_snapshots: int = 13
    estimate_horizon_days: int = 91
    walk_down: float = 0.55
    """Sesgo inicial del consenso en unidades de dispersión (optimismo de partida)."""
    surprise_mean_z: float = 0.35
    """Sesgo medio de la sorpresa estandarizada: cerca del 64 % de eventos baten."""
    surprise_df: float = 4.0
    estimate_noise: float = 0.12
    analysts_range: tuple[int, int] = (6, 42)

    # --- opciones
    iv_premium: float = 1.08
    """Prima de riesgo de varianza: IV implícita sobre volatilidad realizada."""
    iv_noise: float = 0.10
    iv_ar: float = 0.94
    iv_short_long_gap: float = 0.035
    iv_term_tau: float = 0.30
    smile_skew: float = -0.085
    smile_curvature: float = 0.045
    vol_spread_sigma: float = 0.010
    option_expiries: int = 4
    option_strikes: int = 9
    put_call_volume_ratio: float = 0.88

    # --- flujo
    short_interest_ar: float = 0.90
    short_interest_sigma: float = 0.14
    short_publication_lag: int = 8
    """Sesiones entre la fecha de liquidación y la publicación de FINRA."""
    offex_ar: float = 0.72
    offex_sigma: float = 0.022
    offex_publication_lag: int = 14
    """Días naturales de retardo de la transparencia ATS de FINRA (Tier 1)."""

    # --- dividendos y splits
    dividend_lag_days: int = 55
    """Días entre el cierre de trimestre y la fecha ex-dividendo."""
    enable_splits: bool = False
    split_price_threshold: float = 400.0
    split_probability: float = 0.35
    """Probabilidad trimestral de split una vez superado el umbral de precio."""

    leak: LeakSpec = LeakSpec()

    def validated(self) -> SyntheticConfig:
        """Comprueba los invariantes numéricos y devuelve la propia configuración."""
        if not 0.0 < self.market_vol < 2.0:
            msg = f"market_vol fuera de rango: {self.market_vol}"
            raise ConfigError(msg)
        if not 0.0 <= self.volume_ar < 1.0:
            msg = f"volume_ar debe estar en [0, 1): {self.volume_ar}"
            raise ConfigError(msg)
        if not 0.0 <= self.vol_persistence < 1.0:
            msg = f"vol_persistence debe estar en [0, 1): {self.vol_persistence}"
            raise ConfigError(msg)
        if self.t_df_idio <= 2.0 or self.t_df_market <= 2.0 or self.t_df_sector <= 2.0:
            msg = "los grados de libertad de la t deben ser > 2 para que exista varianza"
            raise ConfigError(msg)
        if self.estimate_snapshots < 2:
            msg = "hacen falta al menos 2 snapshots de consenso para medir revisiones"
            raise ConfigError(msg)
        if self.history_quarters < 4:
            msg = "history_quarters < 4 deja sin histórico a los factores de crecimiento"
            raise ConfigError(msg)
        lo, hi = self.report_delay_bounds
        if not 0 < lo < hi:
            msg = f"report_delay_bounds inválido: {self.report_delay_bounds}"
            raise ConfigError(msg)
        wlo, whi = self.leak.window_days
        if not 1 <= wlo <= whi:
            msg = f"leak.window_days inválido: {self.leak.window_days}"
            raise ConfigError(msg)
        return self


# ---------------------------------------------------------------------------
# Utilidades deterministas
# ---------------------------------------------------------------------------


def _stream(seed: int, *labels: object) -> np.random.Generator:
    """Generador independiente y reproducible para un bloque del modelo.

    La entropía se deriva por BLAKE2b de la semilla y las etiquetas. Se evita
    deliberadamente `hash()`: su aleatorización por proceso (PYTHONHASHSEED) haría
    que dos ejecuciones con la misma semilla difirieran.
    """
    payload = "|".join([str(seed), *(str(x) for x in labels)]).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=16).digest()
    return np.random.default_rng(np.random.SeedSequence(int.from_bytes(digest, "big")))


def _student_t(rng: np.random.Generator, df: float, size: tuple[int, ...] | int) -> np.ndarray:
    """t de Student estandarizada a varianza unidad (colas gruesas sin recalibrar sigma)."""
    scale = math.sqrt(df / (df - 2.0))
    return np.asarray(rng.standard_t(df, size=size)) / scale


def _ar1(
    rng: np.random.Generator,
    n_steps: int,
    n_series: int,
    phi: float,
    sigma: float,
    *,
    df: float | None = None,
) -> np.ndarray:
    """Simula procesos AR(1) estacionarios de media cero, forma ``(n_steps, n_series)``.

    El estado inicial se extrae de la distribución estacionaria, de modo que la serie
    no arranca en cero ni necesita periodo de calentamiento.
    """
    if n_steps <= 0:
        return np.zeros((0, n_series))
    eps = (
        rng.standard_normal((n_steps, n_series))
        if df is None
        else _student_t(rng, df, (n_steps, n_series))
    )
    eps = eps * sigma
    stationary_sd = sigma / math.sqrt(1.0 - phi * phi)
    x0 = rng.standard_normal(n_series) * stationary_sd
    zi = (phi * x0).reshape(1, n_series)
    out, _ = lfilter([1.0], [1.0, -phi], eps, axis=0, zi=zi)
    return np.asarray(out)


def _as_date(value: DateLike) -> dt.date:
    """Normaliza cualquier representación de fecha a `datetime.date`."""
    ts = pd.Timestamp(value)
    if ts is pd.NaT or pd.isna(ts):
        msg = f"fecha no interpretable: {value!r}"
        raise ConfigError(msg)
    return ts.date()


def _month_end(year: int, month: int) -> dt.date:
    """Último día natural del mes indicado."""
    return (pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)).date()


def _fiscal_label(period_end: dt.date, fy_end_month: int) -> str:
    """Etiqueta ``AAAAQn`` del trimestre fiscal que cierra en `period_end`.

    Para ejercicios desplazados se usa la convención mayoritaria: el año fiscal toma
    el nombre del año natural en el que cae la mayor parte del ejercicio (un cierre
    en enero de 2025 pertenece al ejercicio 2024).
    """
    months_since_start = (period_end.month - fy_end_month - 1) % 12
    quarter = months_since_start // 3 + 1
    fy_end_year = period_end.year if period_end.month <= fy_end_month else period_end.year + 1
    if fy_end_month <= 5:
        fy_end_year -= 1
    return f"{fy_end_year}Q{quarter}"


def _quarter_ends(fy_end_month: int, first: dt.date, last: dt.date) -> list[dt.date]:
    """Cierres de trimestre fiscal dentro de ``[first, last]``."""
    out: list[dt.date] = []
    year = first.year - 1
    while year <= last.year + 1:
        for month in range(1, 13):
            if (month - fy_end_month) % 3 != 0:
                continue
            end = _month_end(year, month)
            if first <= end <= last:
                out.append(end)
        year += 1
    return sorted(out)


def _third_friday(year: int, month: int) -> dt.date:
    """Tercer viernes del mes: fecha de vencimiento estándar de opciones en EE. UU."""
    first = dt.date(year, month, 1)
    offset = (4 - first.weekday()) % 7
    return first + dt.timedelta(days=offset + 14)


def _bs_greeks(
    spot: np.ndarray,
    strike: np.ndarray,
    tau: np.ndarray,
    iv: np.ndarray,
    rate: float,
    div_yield: np.ndarray,
    is_call: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precio y griegas Black-Scholes-Merton (Black y Scholes 1973; Merton 1973).

    Devuelve ``(precio, delta, gamma, vega)``. `vega` se expresa por punto porcentual
    de volatilidad, que es la convención de las mesas de opciones.
    """
    sqrt_tau = np.sqrt(tau)
    fwd = spot * np.exp((rate - div_yield) * tau)
    vol_tau = np.maximum(iv * sqrt_tau, 1e-8)
    d1 = (np.log(fwd / strike) + 0.5 * vol_tau * vol_tau) / vol_tau
    d2 = d1 - vol_tau
    disc = np.exp(-rate * tau)
    nd1, nd2 = ndtr(d1), ndtr(d2)
    call = disc * (fwd * nd1 - strike * nd2)
    put = disc * (strike * (1.0 - nd2) - fwd * (1.0 - nd1))
    price = np.where(is_call, call, put)
    carry = np.exp(-div_yield * tau)
    delta = np.where(is_call, carry * nd1, carry * (nd1 - 1.0))
    pdf = np.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
    gamma = carry * pdf / (spot * vol_tau)
    vega = spot * carry * pdf * sqrt_tau / 100.0
    return price, delta, gamma, vega


# ---------------------------------------------------------------------------
# Selección del universo
# ---------------------------------------------------------------------------


def _load_seed_universe(path: Path) -> pd.DataFrame:
    """Lee los constituyentes semilla y devuelve ``ticker, sector, cik``.

    Usa símbolos y sectores GICS reales (`data/seed/sp500_constituents.csv`) para
    que los tests de otros módulos manejen tickers con los que se puede razonar y
    para que la distribución sectorial sea la del índice y no una invención.
    """
    if not path.exists():
        raise ProviderUnavailable(
            "synthetic",
            f"no existe el fichero semilla de constituyentes {path}; "
            "el generador necesita la lista de símbolos y sectores GICS",
        )
    raw = pd.read_csv(path)
    missing = {"Symbol", "GICS Sector"} - set(raw.columns)
    if missing:
        msg = f"{path}: faltan columnas {sorted(missing)}"
        raise DataQualityError(msg)
    frame = pd.DataFrame(
        {
            "ticker": [normalize_ticker(str(s)) for s in raw["Symbol"]],
            "sector": [str(s).strip() for s in raw["GICS Sector"]],
        }
    )
    if "CIK" in raw.columns:
        frame["cik"] = [normalize_cik(int(c)) for c in raw["CIK"]]
    else:
        frame["cik"] = [normalize_cik(i + 1) for i in range(len(frame))]
    frame = frame.drop_duplicates("ticker").sort_values("ticker").reset_index(drop=True)
    frame["sector"] = frame["sector"].where(frame["sector"].isin(SECTOR_PROFILES), _DEFAULT_SECTOR)
    return frame


def _stratified_sample(frame: pd.DataFrame, n_tickers: int) -> pd.DataFrame:
    """Muestra estratificada por sector, determinista y estable al crecer `n_tickers`.

    Se recorre en rondas la lista de sectores (orden alfabético) tomando el siguiente
    símbolo de cada uno. Así los once sectores GICS aparecen desde tamaños de
    universo pequeños, que es lo que necesitan los tests de neutralización sectorial.
    """
    if n_tickers <= 0:
        msg = f"n_tickers debe ser >= 1; recibido {n_tickers}"
        raise ConfigError(msg)
    if n_tickers > len(frame):
        msg = (
            f"n_tickers={n_tickers} supera los {len(frame)} símbolos disponibles "
            "en el fichero semilla"
        )
        raise ConfigError(msg)
    buckets = {sec: list(sub.index) for sec, sub in frame.groupby("sector", sort=True)}
    order: list[int] = []
    cursor = 0
    while len(order) < n_tickers:
        progressed = False
        for sec in sorted(buckets):
            bucket = buckets[sec]
            if cursor < len(bucket):
                order.append(bucket[cursor])
                progressed = True
                if len(order) == n_tickers:
                    break
        if not progressed:  # pragma: no cover - imposible tras validar el tamaño
            break
        cursor += 1
    return frame.loc[sorted(order)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Mercado sintético
# ---------------------------------------------------------------------------


class SyntheticMarket:
    """Mercado sintético completo, coherente y reproducible.

    Todos los paneles se generan de forma perezosa la primera vez que se piden y se
    memorizan; los métodos públicos devuelven copias para que un consumidor
    descuidado no pueda corromper el estado interno y romper el determinismo.

    Ejemplo
    -------
    >>> mkt = SyntheticMarket(seed=7, n_tickers=12, start="2021-01-04", end="2022-12-30")
    >>> px = mkt.prices()
    >>> px.index.names
    FrozenList(['date', 'ticker'])
    >>> set(mkt.leaked_event_ids()).issubset(set(mkt.events()["event_id"]))
    True
    """

    name: str = "synthetic"
    """Nombre del proveedor (protocolo `data.base.Provider`)."""

    kinds: tuple[str, ...] = (
        "prices",
        "corporate_actions",
        "fundamentals",
        "estimates",
        "earnings_calendar",
        "options",
        "short_interest",
        "off_exchange",
    )
    """Tipos de dato que sirve, en la nomenclatura de `data.base.DataKind`.

    Declararlos permite registrar el generador en el `ProviderRegistry` como
    proveedor de última prioridad: cualquier módulo puede así ejecutarse de extremo
    a extremo sin red, resolviendo contra datos sintéticos en vez de fallar."""

    def __init__(
        self,
        seed: int = 20260803,
        n_tickers: int = 60,
        start: DateLike = "2019-01-02",
        end: DateLike = "2024-12-31",
        *,
        leak_fraction: float = 0.15,
        config: SyntheticConfig | None = None,
        calendar: TradingCalendar | None = None,
        settings: Settings | None = None,
        tickers: Sequence[Ticker] | None = None,
    ) -> None:
        if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
            msg = f"seed debe ser un entero; recibido {seed!r}"
            raise ConfigError(msg)
        if not 0.0 <= leak_fraction <= 1.0:
            msg = f"leak_fraction debe estar en [0, 1]; recibido {leak_fraction}"
            raise ConfigError(msg)

        self.seed = int(seed)
        self.leak_fraction = float(leak_fraction)
        self.config = (config or SyntheticConfig()).validated()
        self.settings = settings or get_settings()
        self.start = _as_date(start)
        self.end = _as_date(end)
        if self.start >= self.end:
            msg = f"rango vacío o invertido: {self.start} >= {self.end}"
            raise ConfigError(msg)

        self.calendar = calendar or get_calendar(
            first_year=max(1990, self.start.year - 5), last_year=max(self.end.year + 2, 2035)
        )
        sessions = self.calendar.sessions(self.start, self.end)
        if len(sessions) < 60:
            msg = (
                f"solo {len(sessions)} sesiones entre {self.start} y {self.end}: "
                "el generador necesita al menos 60 para que los paneles derivados "
                "(volatilidad realizada, run-up, ventanas de evento) tengan sentido"
            )
            raise InsufficientHistory(msg)
        self.sessions: pd.DatetimeIndex = sessions

        seed_frame = _load_seed_universe(self.settings.seed_dir / "sp500_constituents.csv")
        if tickers is not None:
            wanted = [normalize_ticker(t) for t in tickers]
            selected = seed_frame[seed_frame["ticker"].isin(wanted)].reset_index(drop=True)
            unknown = sorted(set(wanted) - set(selected["ticker"]))
            if unknown:
                msg = f"símbolos ausentes del fichero semilla: {unknown}"
                raise ConfigError(msg)
            if len(selected) < 2:
                msg = "hacen falta al menos 2 símbolos para tener sección cruzada"
                raise ConfigError(msg)
        else:
            selected = _stratified_sample(seed_frame, n_tickers)
        self._universe = selected
        self.tickers: tuple[Ticker, ...] = tuple(selected["ticker"])
        self.n_tickers = len(self.tickers)

    # ------------------------------------------------------------------ básico

    def __repr__(self) -> str:  # pragma: no cover - cosmético
        return (
            f"SyntheticMarket(seed={self.seed}, n_tickers={self.n_tickers}, "
            f"start={self.start.isoformat()}, end={self.end.isoformat()}, "
            f"leak_fraction={self.leak_fraction})"
        )

    def available(self) -> bool:
        """Siempre disponible: no necesita credenciales ni red (protocolo `Provider`)."""
        return True

    def sectors(self) -> pd.Series:
        """Sector GICS de cada símbolo, indexado por ticker."""
        return pd.Series(
            self._universe["sector"].to_numpy(),
            index=pd.Index(self.tickers, name="ticker"),
            name="sector",
        )

    def cik_map(self) -> dict[Ticker, CIK]:
        """Correspondencia ticker -> CIK tomada del fichero semilla."""
        return dict(zip(self.tickers, self._universe["cik"], strict=True))

    def metadata(self) -> pd.DataFrame:
        """Parámetros verdaderos de cada empresa (betas, volatilidad, tamaño...).

        Es verdad-terreno del modelo generador: permite comprobar que un estimador
        de betas o de volatilidad idiosincrásica recupera los valores reales, algo
        imposible con datos de mercado auténticos.
        """
        return self._meta.copy()

    # --------------------------------------------------------------- metadatos

    @cached_property
    def _meta(self) -> pd.DataFrame:
        """Parámetros por empresa derivados del sector más dispersión idiosincrásica."""
        rng = _stream(self.seed, "meta")
        n = self.n_tickers
        sectors = self._universe["sector"].to_numpy()
        prof = [SECTOR_PROFILES[s] for s in sectors]

        log_mcap = rng.normal(math.log(28e9), 1.05, n)
        mcap = np.exp(log_mcap)
        price0 = np.exp(rng.normal(math.log(85.0), 0.55, n)).clip(9.0, 900.0)
        shares = mcap / price0

        beta = np.array([p.beta for p in prof]) * rng.normal(1.0, 0.13, n)
        idio_vol = np.array([p.idio_vol for p in prof]) * np.exp(rng.normal(0.0, 0.20, n))
        beta_sector = rng.normal(1.0, 0.22, n).clip(0.2, 2.0)

        # Cargas de tamaño y valor: las pequeñas cargan positivo en SMB, las de
        # múltiplo bajo cargan positivo en HML (Fama y French 1993).
        size_rank = pd.Series(log_mcap).rank(pct=True).to_numpy()
        beta_smb = (0.55 - 1.10 * size_rank) + rng.normal(0.0, 0.18, n)
        ps_ratio = np.array([p.price_to_sales for p in prof]) * np.exp(rng.normal(0.0, 0.30, n))
        value_rank = pd.Series(-ps_ratio).rank(pct=True).to_numpy()
        beta_hml = (value_rank - 0.5) * 0.9 + rng.normal(0.0, 0.20, n)

        revenue0 = mcap / ps_ratio / 4.0  # ingresos trimestrales
        growth = np.array([p.revenue_growth for p in prof]) + rng.normal(0.0, 0.030, n)
        gross_margin = (
            np.array([p.gross_margin for p in prof]) + rng.normal(0.0, 0.032, n)
        ).clip(0.10, 0.93)
        opex_ratio = (
            np.array([p.opex_ratio for p in prof]) * np.exp(rng.normal(0.0, 0.09, n))
        ).clip(0.05, 0.85)
        equity_ratio = (
            np.array([p.equity_ratio for p in prof]) + rng.normal(0.0, 0.055, n)
        ).clip(0.18, 0.72)

        fy_choice = rng.random(n)
        fy_end_month = np.where(
            fy_choice < 0.78, 12, np.where(fy_choice < 0.87, 9, np.where(fy_choice < 0.94, 6, 1))
        ).astype(int)

        cfg = self.config
        delay = rng.normal(cfg.report_delay_mean, cfg.report_delay_sd, n)
        delay = np.clip(np.round(delay), *cfg.report_delay_bounds).astype(int)

        jump_vol = np.array([p.jump_vol for p in prof]) * np.exp(rng.normal(0.0, 0.16, n))
        option_liq = np.array([p.option_liquidity for p in prof]) * np.exp(
            rng.normal(0.0, 0.35, n)
        )
        # El volumen medio se ancla al tamaño: rotación diaria del 0.4 % al 1.2 %.
        turnover = np.exp(rng.normal(math.log(0.0065), 0.40, n)).clip(0.0015, 0.030)
        base_volume = shares * turnover

        meta = pd.DataFrame(
            {
                "sector": sectors,
                "cik": self._universe["cik"].to_numpy(),
                "beta_mkt": beta,
                "beta_smb": beta_smb,
                "beta_hml": beta_hml,
                "beta_sector": beta_sector,
                "idio_vol": idio_vol,
                "price0": price0,
                "shares0": shares,
                "market_cap0": mcap,
                "revenue0": revenue0,
                "revenue_growth": growth,
                "gross_margin": gross_margin,
                "opex_ratio": opex_ratio,
                "equity_ratio": equity_ratio,
                "price_to_sales": ps_ratio,
                "fy_end_month": fy_end_month,
                "report_delay": delay,
                "amc_preference": rng.random(n) < cfg.amc_probability,
                "jump_vol": jump_vol,
                "option_liquidity": option_liq,
                "base_volume": base_volume,
                "turnover": turnover,
                "tax_rate": np.array([p.tax_rate for p in prof]) * rng.normal(1.0, 0.10, n),
                "payout": np.array([p.payout for p in prof]).clip(0.0, 0.95),
                "buyback_ratio": np.array([p.buyback_ratio for p in prof]),
                "capex_ratio": np.array([p.capex_ratio for p in prof])
                * np.exp(rng.normal(0.0, 0.18, n)),
                "da_ratio": np.array([p.da_ratio for p in prof]),
                "sbc_ratio": np.array([p.sbc_ratio for p in prof]),
                "ppe_to_revenue": np.array([p.ppe_to_revenue for p in prof])
                * np.exp(rng.normal(0.0, 0.20, n)),
                "goodwill_to_revenue": np.array([p.goodwill_to_revenue for p in prof]),
                "dso": np.array([p.dso for p in prof]) * rng.normal(1.0, 0.12, n),
                "dio": np.array([p.dio for p in prof]) * rng.normal(1.0, 0.15, n),
                "dpo": np.array([p.dpo for p in prof]) * rng.normal(1.0, 0.12, n),
                "short_base": np.array([p.short_base for p in prof])
                * np.exp(rng.normal(0.0, 0.45, n)),
                "offex_base": np.array([p.offex_base for p in prof]) + rng.normal(0.0, 0.020, n),
                "surprise_scale": np.exp(rng.normal(math.log(0.045), 0.35, n)),
                "n_analysts": rng.integers(cfg.analysts_range[0], cfg.analysts_range[1], n),
            },
            index=pd.Index(self.tickers, name="ticker"),
        )
        meta["dividend_yield"] = (meta["payout"] / meta["price_to_sales"]).clip(0.0, 0.075)
        return meta

    @cached_property
    def _sector_names(self) -> tuple[str, ...]:
        return tuple(sorted(set(self._universe["sector"])))

    @cached_property
    def _quarter_grid(self) -> dict[Ticker, list[dt.date]]:
        """Cierres de trimestre fiscal por empresa, incluido el histórico previo."""
        first = self.start - dt.timedelta(days=round(self.config.history_quarters * 91.3) + 40)
        last = self.end
        return {
            ticker: _quarter_ends(int(self._meta.loc[ticker, "fy_end_month"]), first, last)
            for ticker in self.tickers
        }

    # ------------------------------------------------------------ fundamentales

    @cached_property
    def _fundamentals_core(self) -> pd.DataFrame:
        """Estados financieros trimestrales con identidad contable exacta.

        El balance se construye por arrastre y todas las partidas se mueven a la vez
        a través del estado de flujos (método indirecto), de modo que
        ``ΔActivo = ΔPasivo + ΔPatrimonio`` se cumple por construcción y no por
        ajuste posterior. Es lo que permite que Piotroski (2000) y Sloan (1996) se
        calculen sobre estos datos sin resultados absurdos.
        """
        cfg = self.config
        frames: list[pd.DataFrame] = []
        for ticker in self.tickers:
            meta = self._meta.loc[ticker]
            profile = SECTOR_PROFILES[str(meta["sector"])]
            rng = _stream(self.seed, "fundamentals", ticker)
            quarters = self._quarter_grid[ticker]
            n_q = len(quarters)

            growth = _ar1(rng, n_q, 1, cfg.growth_persistence, cfg.growth_sigma)[:, 0]
            growth += float(meta["revenue_growth"]) / 4.0
            margin_drift = _ar1(rng, n_q, 1, cfg.margin_persistence, cfg.margin_sigma)[:, 0]
            opex_drift = _ar1(rng, n_q, 1, cfg.margin_persistence, cfg.margin_sigma)[:, 0]
            rev_noise = rng.normal(0.0, cfg.revenue_noise, n_q)
            capex_noise = rng.normal(1.0, 0.18, n_q).clip(0.4, 2.0)
            other_noise = rng.normal(0.0, 0.03, (n_q, 2))
            wc_noise = rng.normal(0.0, 0.06, (n_q, 4))
            debt_noise = rng.normal(0.0, 0.020, n_q)
            other_income_noise = rng.normal(0.0, 0.004, n_q)

            revenue0 = float(meta["revenue0"])
            level = revenue0 / profile.seasonality[(quarters[0].month - 1) // 3]
            # Se retrocede el nivel para que el trimestre inicial sea coherente con
            # el tamaño actual tras crecer durante todo el histórico.
            level /= float(np.prod(1.0 + growth[: min(n_q, cfg.history_quarters)]))

            gross_margin = float(meta["gross_margin"])
            opex_ratio = float(meta["opex_ratio"])
            cogs_ratio = 1.0 - gross_margin
            # Tasa de amortización sobre inmovilizado calibrada para que la
            # amortización del primer trimestre sea la fracción de ingresos del
            # sector: así los sectores intensivos en capital no devoran el margen.
            da_rate = float(meta["da_ratio"]) / max(float(meta["ppe_to_revenue"]), 0.05)

            revenue_q0 = level * profile.seasonality[(quarters[0].month - 1) // 3]
            receivables = revenue_q0 * float(meta["dso"]) / _QUARTER_DAYS
            inventory = revenue_q0 * cogs_ratio * float(meta["dio"]) / _QUARTER_DAYS
            payables = revenue_q0 * cogs_ratio * float(meta["dpo"]) / _QUARTER_DAYS
            accrued = revenue_q0 * opex_ratio * 40.0 / _QUARTER_DAYS
            ppe = revenue_q0 * float(meta["ppe_to_revenue"])
            goodwill = revenue_q0 * float(meta["goodwill_to_revenue"])
            other_assets = revenue_q0 * 0.35
            cash = revenue_q0 * 0.55

            assets = cash + receivables + inventory + ppe + goodwill + other_assets
            liabilities_target = assets * (1.0 - float(meta["equity_ratio"]))
            operating_liab = payables + accrued
            residual = max(liabilities_target - operating_liab, 0.02 * assets)
            other_liabilities = 0.25 * residual
            total_debt = 0.75 * residual
            short_debt, long_debt = 0.20 * total_debt, 0.80 * total_debt
            retained = 0.55 * (assets - operating_liab - residual)
            paid_in = assets - (operating_liab + residual) - retained

            shares = float(meta["shares0"])
            price_proxy = float(meta["price0"])
            quarterly_drift = math.exp((cfg.market_drift + float(meta["revenue_growth"]) * 0.3) / 4)
            ni_smooth = revenue_q0 * 0.08
            rows: list[dict[str, object]] = []

            for qi, period_end in enumerate(quarters):
                level *= 1.0 + growth[qi]
                season = profile.seasonality[(period_end.month - 1) // 3]
                revenue = max(level * season * (1.0 + rev_noise[qi]), 1e6)
                margin = min(max(gross_margin + margin_drift[qi], 0.05), 0.95)
                opex_rate = min(max(opex_ratio + opex_drift[qi], 0.02), 0.92)

                cogs = revenue * (1.0 - margin)
                gross_profit = revenue - cogs
                opex = revenue * opex_rate
                d_and_a = ppe * da_rate
                operating_income = gross_profit - opex - d_and_a
                interest = (short_debt + long_debt) * cfg.debt_rate / 4.0
                other_income = revenue * other_income_noise[qi]
                pretax = operating_income - interest + other_income
                tax_rate = float(meta["tax_rate"])
                tax = pretax * tax_rate if pretax > 0 else pretax * tax_rate * 0.30
                net_income = pretax - tax
                sbc = revenue * float(meta["sbc_ratio"])
                capex = revenue * float(meta["capex_ratio"]) * capex_noise[qi]

                new_receivables = revenue * float(meta["dso"]) / _QUARTER_DAYS * (
                    1.0 + wc_noise[qi, 0]
                )
                new_inventory = cogs * float(meta["dio"]) / _QUARTER_DAYS * (1.0 + wc_noise[qi, 1])
                new_payables = cogs * float(meta["dpo"]) / _QUARTER_DAYS * (1.0 + wc_noise[qi, 2])
                new_accrued = opex * 40.0 / _QUARTER_DAYS * (1.0 + wc_noise[qi, 3])
                new_other_assets = max(other_assets * (1.0 + other_noise[qi, 0]), 0.0)
                new_other_liab = max(other_liabilities * (1.0 + other_noise[qi, 1]), 0.0)

                d_receivables = new_receivables - receivables
                d_inventory = new_inventory - inventory
                d_payables = new_payables - payables
                d_accrued = new_accrued - accrued
                d_other_assets = new_other_assets - other_assets
                d_other_liab = new_other_liab - other_liabilities

                cfo = (
                    net_income
                    + d_and_a
                    + sbc
                    - d_receivables
                    - d_inventory
                    - d_other_assets
                    + d_payables
                    + d_accrued
                    + d_other_liab
                )
                cfi = -capex
                fcf = cfo - capex

                ni_smooth = 0.85 * ni_smooth + 0.15 * max(net_income, 0.0)
                dividends = float(meta["payout"]) * max(ni_smooth, 0.0)
                buybacks = max(fcf - dividends, 0.0) * float(meta["buyback_ratio"])
                issuance = 0.0

                # La variación de deuda se aplica primero y se recalcula a partir del
                # saldo resultante: si algún recorte actuase, el flujo de financiación
                # debe reflejar el cambio *realmente* aplicado o la identidad contable
                # dejaría de cumplirse en silencio.
                new_short = max(short_debt + (short_debt + long_debt) * debt_noise[qi] * 0.35, 0.0)
                new_long = max(long_debt + (short_debt + long_debt) * debt_noise[qi] * 0.65, 0.0)
                d_debt = (new_short + new_long) - (short_debt + long_debt)

                cff = d_debt + issuance - buybacks - dividends
                d_cash = cfo + cfi + cff
                new_cash = cash + d_cash
                min_cash = cfg.min_cash_ratio * revenue
                if new_cash < min_cash:
                    draw = min_cash - new_cash
                    d_debt += draw
                    new_short += draw
                    cff += draw
                    d_cash += draw
                    new_cash = min_cash

                short_debt, long_debt = new_short, new_long
                cash = new_cash
                receivables, inventory = new_receivables, new_inventory
                payables, accrued = new_payables, new_accrued
                other_assets, other_liabilities = new_other_assets, new_other_liab
                # `d_and_a = ppe * da_rate` con `da_rate < 1`, así que el inmovilizado
                # neto nunca puede volverse negativo y no hace falta recortarlo.
                ppe = ppe + capex - d_and_a

                retained += net_income - dividends
                paid_in += sbc + issuance - buybacks

                total_assets = cash + receivables + inventory + ppe + goodwill + other_assets
                total_liabilities = payables + accrued + short_debt + long_debt + other_liabilities
                total_equity = paid_in + retained

                price_proxy *= quarterly_drift
                shares = max(shares + (sbc - buybacks + issuance) / price_proxy, 1e5)
                diluted = shares * 1.02

                rows.append(
                    {
                        "ticker": ticker,
                        "period_end": period_end,
                        "fiscal_period": _fiscal_label(
                            period_end, int(meta["fy_end_month"])
                        ),
                        "revenue": revenue,
                        "cost_of_revenue": cogs,
                        "gross_profit": gross_profit,
                        "operating_expenses": opex,
                        "depreciation_amortization": d_and_a,
                        "operating_income": operating_income,
                        "ebitda": operating_income + d_and_a,
                        "interest_expense": interest,
                        "pretax_income": pretax,
                        "income_tax": tax,
                        "net_income": net_income,
                        "stock_compensation": sbc,
                        "cfo": cfo,
                        "cfi": cfi,
                        "cff": cff,
                        "capex": capex,
                        "fcf": fcf,
                        "dividends_paid": dividends,
                        "buybacks": buybacks,
                        "change_in_cash": d_cash,
                        "cash": cash,
                        "receivables": receivables,
                        "inventory": inventory,
                        "ppe_net": ppe,
                        "goodwill": goodwill,
                        "other_assets": other_assets,
                        "total_assets": total_assets,
                        "payables": payables,
                        "accrued_liabilities": accrued,
                        "short_term_debt": short_debt,
                        "long_term_debt": long_debt,
                        "total_debt": short_debt + long_debt,
                        "other_liabilities": other_liabilities,
                        "total_liabilities": total_liabilities,
                        "paid_in_capital": paid_in,
                        "retained_earnings": retained,
                        "total_equity": total_equity,
                        "shares_outstanding": shares,
                        "shares_diluted": diluted,
                        "eps_diluted": net_income / diluted,
                        "dividend_per_share": dividends / shares,
                    }
                )
            frames.append(pd.DataFrame(rows))

        out = pd.concat(frames, ignore_index=True)
        out["current_assets"] = out["cash"] + out["receivables"] + out["inventory"]
        out["current_liabilities"] = (
            out["payables"] + out["accrued_liabilities"] + out["short_term_debt"]
        )
        out["gross_margin"] = out["gross_profit"] / out["revenue"]
        out["operating_margin"] = out["operating_income"] / out["revenue"]
        out["net_margin"] = out["net_income"] / out["revenue"]
        prev_assets = out.groupby("ticker", sort=False)["total_assets"].shift(1)
        avg_assets = (out["total_assets"] + prev_assets) / 2.0
        out["roa"] = out["net_income"] / avg_assets
        out["roe"] = out["net_income"] / out["total_equity"]
        # Accruals de balance de Sloan (1996): la parte no monetaria del beneficio.
        out["accruals"] = (out["net_income"] - out["cfo"]) / avg_assets
        out["earnings_quality"] = out["cfo"] / out["net_income"].replace(0.0, np.nan)
        out["leverage"] = out["total_debt"] / out["total_assets"]
        out["asset_turnover"] = out["revenue"] / avg_assets
        out["period_end"] = pd.to_datetime(out["period_end"])
        return out.sort_values(["ticker", "period_end"]).reset_index(drop=True)

    # ----------------------------------------------------------------- eventos

    @cached_property
    def _event_dates(self) -> pd.DataFrame:
        """Fechas y horas de anuncio, y su primera sesión negociable."""
        cfg = self.config
        rows: list[dict[str, object]] = []
        for ticker in self.tickers:
            meta = self._meta.loc[ticker]
            rng = _stream(self.seed, "eventdates", ticker)
            quarters = self._quarter_grid[ticker]
            n_q = len(quarters)
            jitter = np.round(rng.normal(0.0, cfg.report_delay_jitter, n_q)).astype(int)
            jitter = np.clip(jitter, -6, 6)
            flips = rng.random(n_q) < cfg.session_flip_probability
            bmo_minutes = rng.integers(6 * 60 + 45, 8 * 60 + 30, n_q)
            amc_minutes = rng.integers(16 * 60 + 5, 17 * 60 + 15, n_q)
            filing_lag = rng.integers(cfg.filing_lag_days[0], cfg.filing_lag_days[1], n_q)
            base_delay = int(meta["report_delay"])
            prefers_amc = bool(meta["amc_preference"])

            for qi, period_end in enumerate(quarters):
                delay = int(np.clip(base_delay + jitter[qi], *cfg.report_delay_bounds))
                day = period_end + dt.timedelta(days=delay)
                day = self.calendar.session_on_or_after(day)
                is_amc = prefers_amc != bool(flips[qi])
                minutes = int(amc_minutes[qi] if is_amc else bmo_minutes[qi])
                local = dt.datetime(day.year, day.month, day.day) + dt.timedelta(minutes=minutes)
                announced = eastern_to_utc(local)
                fy_end_month = int(meta["fy_end_month"])
                label = _fiscal_label(period_end, fy_end_month)
                is_annual = label.endswith("Q4")
                rows.append(
                    {
                        "ticker": ticker,
                        "cik": str(meta["cik"]),
                        "period_end": period_end,
                        "fiscal_quarter": label,
                        "announced_at": announced,
                        "session": (Session.AMC if is_amc else Session.BMO).value,
                        "form": "10-K" if is_annual else "10-Q",
                        "filed_at": announced + dt.timedelta(days=int(filing_lag[qi])),
                        "report_delay_days": delay,
                    }
                )
        frame = pd.DataFrame(rows)
        frame["period_end"] = pd.to_datetime(frame["period_end"])
        frame["announced_at"] = pd.to_datetime(frame["announced_at"])
        frame["filed_at"] = pd.to_datetime(frame["filed_at"])
        frame["event_date"] = tradable_dates(frame, self.calendar)
        frame = frame[frame["event_date"] <= pd.Timestamp(self.end)].reset_index(drop=True)
        frame["event_id"] = [
            f"{t}:{q}:{pd.Timestamp(pe).date().isoformat()}"
            for t, q, pe in zip(
                frame["ticker"], frame["fiscal_quarter"], frame["period_end"], strict=True
            )
        ]
        return frame.sort_values(["announced_at", "ticker"]).reset_index(drop=True)

    @cached_property
    def _events_core(self) -> pd.DataFrame:
        """Eventos con resultado real, consenso final y sorpresa estandarizada."""
        cfg = self.config
        fundamentals = self._fundamentals_core
        dates = self._event_dates
        keys = ["ticker", "period_end"]
        merged = dates.merge(
            fundamentals[[*keys, "eps_diluted", "revenue", "net_income", "shares_diluted"]],
            on=keys,
            how="inner",
            validate="one_to_one",
        )
        merged = merged.rename(columns={"eps_diluted": "eps_actual", "revenue": "revenue_actual"})

        # Un flujo aleatorio por empresa: la senda de sorpresas de un ticker no
        # depende de cuántos otros tickers haya en el universo.
        by_ticker_order = merged.sort_values(["ticker", "period_end"])
        z_values = np.empty(len(by_ticker_order))
        cursor = 0
        for ticker, sub in by_ticker_order.groupby("ticker", sort=True):
            rng = _stream(self.seed, "surprise", ticker)
            n_ev = len(sub)
            z_values[cursor : cursor + n_ev] = cfg.surprise_mean_z + _student_t(
                rng, cfg.surprise_df, n_ev
            )
            cursor += n_ev
        merged["surprise_z"] = pd.Series(z_values, index=by_ticker_order.index).reindex(
            merged.index
        )

        scale = merged["ticker"].map(self._meta["surprise_scale"]).to_numpy()
        eps_scale = np.maximum(np.abs(merged["eps_actual"].to_numpy()) * scale, 0.01)
        merged["eps_surprise"] = merged["surprise_z"].to_numpy() * eps_scale
        merged["eps_estimate"] = merged["eps_actual"] - merged["eps_surprise"]
        merged["eps_surprise_sigma"] = eps_scale

        rev_rng = _stream(self.seed, "revenue_surprise")
        rev_z = 0.6 * merged["surprise_z"].to_numpy() + 0.8 * rev_rng.standard_normal(len(merged))
        merged["revenue_surprise"] = rev_z * merged["revenue_actual"].to_numpy() * 0.018
        merged["revenue_estimate"] = merged["revenue_actual"] - merged["revenue_surprise"]
        merged["surprise_pct"] = merged["eps_surprise"] / merged["eps_estimate"].abs().clip(
            lower=0.01
        )
        merged["n_analysts"] = merged["ticker"].map(self._meta["n_analysts"]).to_numpy()
        merged["source"] = "synthetic"
        merged["is_estimated_date"] = False

        # Componente del salto del anuncio no explicada por la sorpresa: es lo que
        # impide que un modelo recupere la relación de forma trivial.
        jump_noise = np.empty(len(by_ticker_order))
        cursor = 0
        for ticker, sub in by_ticker_order.groupby("ticker", sort=True):
            rng = _stream(self.seed, "eventnoise", ticker)
            jump_noise[cursor : cursor + len(sub)] = rng.standard_normal(len(sub))
            cursor += len(sub)
        merged["jump_noise"] = pd.Series(jump_noise, index=by_ticker_order.index).reindex(
            merged.index
        )

        merged = merged.sort_values(["event_date", "ticker"]).reset_index(drop=True)
        # SUE clásico (Foster, Olsen y Shevlin 1984): sorpresa estandarizada por la
        # desviación típica de las ocho sorpresas anteriores de la propia empresa.
        by_ticker = merged.sort_values(["ticker", "period_end"])
        grouped = by_ticker.groupby("ticker", sort=False)["eps_surprise"]
        prior_mean = grouped.transform(lambda s: s.shift(1).rolling(8, min_periods=6).mean())
        prior_std = grouped.transform(lambda s: s.shift(1).rolling(8, min_periods=6).std())
        sue = (by_ticker["eps_surprise"] - prior_mean) / prior_std.replace(0.0, np.nan)
        merged["sue"] = sue.reindex(merged.index)
        return merged

    @cached_property
    def _leaks(self) -> pd.DataFrame:
        """Verdad-terreno de la filtración: qué eventos la tienen y con qué forma.

        Solo se filtran eventos cuya ventana pre-anuncio cae íntegramente dentro del
        panel de precios: sin sesiones previas no hay dónde inyectar la huella, y un
        evento marcado como filtrado pero sin run-up observable envenenaría cualquier
        medida de recall.
        """
        events = self._events_core
        spec = self.config.leak
        rng = _stream(self.seed, "leaks", self.leak_fraction)
        n_ev = len(events)
        draw = rng.random(n_ev)
        window = rng.integers(spec.window_days[0], spec.window_days[1] + 1, n_ev)
        intensity = rng.uniform(spec.intensity[0], spec.intensity[1], n_ev)

        pos = self.sessions.get_indexer(pd.DatetimeIndex(events["event_date"]))
        has_room = (pos >= 0) & (pos - window >= 0)
        is_leaked = (draw < self.leak_fraction) & has_room
        sign = np.sign(events["surprise_z"].to_numpy())
        sign = np.where(sign == 0.0, 1.0, sign)

        return pd.DataFrame(
            {
                "event_id": events["event_id"].to_numpy(),
                "ticker": events["ticker"].to_numpy(),
                "event_date": events["event_date"].to_numpy(),
                "event_pos": pos,
                "is_leaked": is_leaked,
                "leak_window_days": np.where(is_leaked, window, 0),
                "leak_intensity": np.where(is_leaked, intensity, 0.0),
                "surprise_sign": sign,
                "surprise_z": events["surprise_z"].to_numpy(),
            }
        )

    # ------------------------------------------------------------------ precios

    @cached_property
    def _factor_returns(self) -> pd.DataFrame:
        """Retornos diarios de los factores comunes: mercado, tamaño, valor y sector."""
        cfg = self.config
        n_days = len(self.sessions)
        rng = _stream(self.seed, "factors")

        log_vol = _ar1(rng, n_days, 1, cfg.vol_persistence, cfg.vol_of_vol * 0.1)[:, 0]
        vol_mult = np.exp(log_vol - 0.5 * float(np.var(log_vol)))
        daily_mkt_vol = cfg.market_vol / _SQRT_TRADING
        mkt = (
            cfg.market_drift / _TRADING_DAYS
            + daily_mkt_vol * vol_mult * _student_t(rng, cfg.t_df_market, n_days)
        )
        smb = cfg.smb_vol / _SQRT_TRADING * _student_t(rng, cfg.t_df_market, n_days)
        hml = cfg.hml_vol / _SQRT_TRADING * _student_t(rng, cfg.t_df_market, n_days)

        data = {
            "mkt": mkt,
            "smb": smb,
            "hml": hml,
            "rf": np.full(n_days, cfg.risk_free / _TRADING_DAYS),
            "vol_multiplier": vol_mult,
        }
        for sector in self._sector_names:
            profile = SECTOR_PROFILES[sector]
            srng = _stream(self.seed, "sectorfactor", sector)
            shock = _student_t(srng, cfg.t_df_sector, n_days)
            data[f"sector::{sector}"] = profile.factor_vol / _SQRT_TRADING * vol_mult * shock
        return pd.DataFrame(data, index=self.sessions)

    @cached_property
    def _event_effects(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Matrices ``(n_días, n_tickers)`` de efectos de evento sobre retorno y volumen.

        Separadas del ruido base para que la inyección sea auditable: el efecto del
        anuncio, el PEAD y la huella de filtración son sumandos explícitos. La
        tercera matriz marca el día del anuncio, que es el que se negocia con gap.
        """
        cfg = self.config
        n_days, n_names = len(self.sessions), self.n_tickers
        ret = np.zeros((n_days, n_names))
        logvol = np.zeros((n_days, n_names))
        is_event = np.zeros((n_days, n_names), dtype=bool)
        col = {t: i for i, t in enumerate(self.tickers)}

        table = self._event_table
        pead_w = np.exp(-np.arange(cfg.pead_horizon) / cfg.pead_decay)
        pead_w /= pead_w.sum()
        profile = np.asarray(cfg.volume_event_profile)
        spec = cfg.leak
        jump_scale_by_ticker = self._meta["jump_vol"].to_dict()

        for row in table.itertuples(index=False):
            pos = int(row.event_pos)
            if pos < 0:
                continue
            j = col[row.ticker]
            z = float(np.clip(row.surprise_z, -4.0, 4.0))
            ret[pos, j] += (
                cfg.event_response * z + jump_scale_by_ticker[row.ticker] * row.jump_noise
            )
            is_event[pos, j] = True

            end = min(pos + 1 + cfg.pead_horizon, n_days)
            take = end - (pos + 1)
            if take > 0:
                ret[pos + 1 : end, j] += cfg.pead_total * z * pead_w[:take]

            take_v = min(len(profile), n_days - pos)
            logvol[pos : pos + take_v, j] += profile[:take_v]

            if row.is_leaked:
                width = int(row.leak_window_days)
                kappa = float(row.leak_intensity)
                sign = float(row.surprise_sign)
                ramp = np.linspace(1.0 / width, 1.0, width)
                sl = slice(pos - width, pos)
                logvol[sl, j] += spec.volume_log_amp * kappa * ramp
                ret[sl, j] += sign * spec.price_drift_total * kappa * ramp / ramp.sum()
        return ret, logvol, is_event

    @cached_property
    def _event_table(self) -> pd.DataFrame:
        """Eventos con su verdad-terreno de filtración adjunta (uso interno)."""
        leak_cols = [
            "event_id",
            "event_pos",
            "is_leaked",
            "leak_window_days",
            "leak_intensity",
            "surprise_sign",
        ]
        return self._events_core.merge(
            self._leaks[leak_cols], on="event_id", how="left", validate="one_to_one"
        )

    @cached_property
    def _price_panel(self) -> pd.DataFrame:
        """Panel OHLCV completo, incluidos dividendos, `adj_close` y splits opcionales."""
        cfg = self.config
        n_days, n_names = len(self.sessions), self.n_tickers
        meta = self._meta
        factors = self._factor_returns
        rng = _stream(self.seed, "idio")

        vol_mult = factors["vol_multiplier"].to_numpy()[:, None] ** 0.7
        idio_sd = (meta["idio_vol"].to_numpy() / _SQRT_TRADING)[None, :]
        idio = _student_t(rng, cfg.t_df_idio, (n_days, n_names)) * idio_sd * vol_mult

        sector_mat = np.column_stack(
            [factors[f"sector::{sec}"].to_numpy() for sec in meta["sector"]]
        )
        total = (
            meta["beta_mkt"].to_numpy()[None, :] * factors["mkt"].to_numpy()[:, None]
            + meta["beta_smb"].to_numpy()[None, :] * factors["smb"].to_numpy()[:, None]
            + meta["beta_hml"].to_numpy()[None, :] * factors["hml"].to_numpy()[:, None]
            + meta["beta_sector"].to_numpy()[None, :] * sector_mat
            + idio
        )
        event_ret, event_logvol, is_event = self._event_effects
        total = np.clip(total + event_ret, -cfg.max_abs_log_return, cfg.max_abs_log_return)

        # --- dividendos: importe por acción fijado por el fundamental del trimestre
        div_matrix = self._dividend_matrix()
        prev_close = np.empty((n_days, n_names))
        close = np.empty((n_days, n_names))
        level = meta["price0"].to_numpy().astype(float)
        split_factor = np.ones((n_days, n_names))
        split_rng = _stream(self.seed, "splits")

        for i in range(n_days):
            prev_close[i] = level
            level = np.maximum(level * np.exp(total[i]) - div_matrix[i], 1.0)
            if (
                cfg.enable_splits
                and self.sessions[i].month in (2, 5, 8, 11)
                and self.sessions[i].day <= 3
            ):
                candidate = (level > cfg.split_price_threshold) & (
                    split_rng.random(n_names) < cfg.split_probability
                )
                if candidate.any():
                    factor = np.where(candidate, 2.0, 1.0)
                    level = level / factor
                    split_factor[i] = factor
            close[i] = level

        # Retorno de precio ajustado por split (el split no es un retorno): incluye
        # la caída por dividendo, igual que una serie de cierres sin ajustar.
        returns = np.log(close * split_factor / prev_close)

        # --- volumen: AR(1) sobre el logaritmo, con estacionalidad semanal
        vrng = _stream(self.seed, "volume")
        noise = _ar1(vrng, n_days, n_names, cfg.volume_ar, cfg.volume_sigma)
        dow = self.sessions.dayofweek.to_numpy()
        dow_effect = np.asarray(cfg.volume_dow)[np.clip(dow, 0, 4)][:, None]
        witching = np.array(
            [
                1.0 if (d.month in (3, 6, 9, 12) and d.date() == _third_friday(d.year, d.month))
                else 0.0
                for d in self.sessions
            ]
        )[:, None]
        log_volume = (
            np.log(meta["base_volume"].to_numpy())[None, :]
            + noise
            + dow_effect
            + cfg.volume_witching * witching
            + cfg.volume_abs_return * np.minimum(np.abs(returns), 0.25)
            + event_logvol
        )
        # Corrección lognormal con la varianza *estacionaria* del AR(1), para que el
        # volumen medio coincida con `base_volume` y no con su exponencial sesgada.
        stationary_var = cfg.volume_sigma**2 / (1.0 - cfg.volume_ar**2)
        volume = np.exp(log_volume - 0.5 * stationary_var)
        volume = np.maximum(np.round(volume / 100.0) * 100.0, 100.0)

        # --- OHLC coherente con el cierre y el gap de apertura
        orng = _stream(self.seed, "ohlc")
        gap_share = np.where(is_event, cfg.event_gap_share, 0.35)
        gap_noise = orng.normal(0.0, 0.35, (n_days, n_names)) * idio_sd
        open_px = np.maximum((prev_close - div_matrix), 0.5) * np.exp(
            gap_share * returns + gap_noise
        )
        open_px = np.where(split_factor > 1.0, close, open_px)
        body_hi = np.maximum(open_px, close)
        body_lo = np.minimum(open_px, close)
        wick_scale = (np.abs(returns) * 0.45 + idio_sd * 0.55) * (1.0 + 0.8 * is_event)
        high = body_hi * np.exp(orng.standard_exponential((n_days, n_names)) * wick_scale)
        low = body_lo * np.exp(-orng.standard_exponential((n_days, n_names)) * wick_scale)
        low = np.minimum(low, body_lo)
        high = np.maximum(high, body_hi)

        # --- adj_close: retorno total reconstruido hacia atrás (dividendos y splits)
        total_ret_factor = (close * split_factor + div_matrix) / np.maximum(prev_close, 1e-9)
        total_ret_factor[0] = 1.0
        cum_total = np.cumprod(total_ret_factor, axis=0)
        adj = cum_total / cum_total[-1] * close[-1]

        index = pd.MultiIndex.from_product(
            [self.sessions, list(self.tickers)], names=["date", "ticker"]
        )
        panel = pd.DataFrame(
            {
                "open": open_px.reshape(-1),
                "high": high.reshape(-1),
                "low": low.reshape(-1),
                "close": close.reshape(-1),
                "volume": volume.reshape(-1),
                "adj_close": adj.reshape(-1),
                "dividend": div_matrix.reshape(-1),
                "split_factor": split_factor.reshape(-1),
                "log_return": returns.reshape(-1),
            },
            index=index,
        )
        shares = self._shares_panel()
        panel["shares_outstanding"] = shares.reshape(-1)
        panel["dollar_volume"] = panel["close"] * panel["volume"]
        panel["market_cap"] = panel["close"] * panel["shares_outstanding"]
        return panel

    def wide(self, field: str, *, panel: str = "prices") -> pd.DataFrame:
        """Matriz ``fecha x ticker`` de un campo del panel de precios o de opciones.

        Atajo para el consumidor habitual (`factors`, `events`, `backtest`), que casi
        siempre necesita la forma ancha para operar en sección cruzada.
        """
        source = {"prices": self._price_panel, "options": self._options_daily}.get(panel)
        if source is None:
            msg = f"panel desconocido: {panel!r}; use 'prices' u 'options'"
            raise ConfigError(msg)
        if field not in source.columns:
            msg = f"campo {field!r} inexistente en el panel {panel!r}: {list(source.columns)}"
            raise DataQualityError(msg)
        return self._wide(source[field])

    def _wide(self, column: pd.Series) -> pd.DataFrame:
        """Pasa una columna del panel ``(date, ticker)`` a matriz ``fecha x ticker``.

        Se usa `unstack` y no `pivot_table` a propósito: la clave del panel es única,
        aquí no hay nada que agregar y una agregación silenciosa enmascararía un
        duplicado en el índice, que sí sería un error.
        """
        return column.unstack("ticker")[list(self.tickers)]  # noqa: PD010

    def _dividend_matrix(self) -> np.ndarray:
        """Importe por acción abonado en cada sesión ex-dividendo."""
        matrix = np.zeros((len(self.sessions), self.n_tickers))
        lag = dt.timedelta(days=self.config.dividend_lag_days)
        col = {t: i for i, t in enumerate(self.tickers)}
        fundamentals = self._fundamentals_core
        pos_of = {ts: i for i, ts in enumerate(self.sessions)}
        for row in fundamentals.itertuples(index=False):
            if row.dividend_per_share <= 0.0:
                continue
            ex_day = row.period_end.date() + lag
            if not (self.start <= ex_day <= self.end):
                continue
            ts = pd.Timestamp(self.calendar.session_on_or_after(ex_day))
            idx = pos_of.get(ts)
            if idx is not None:
                matrix[idx, col[row.ticker]] += float(row.dividend_per_share)
        return matrix

    def _shares_panel(self) -> np.ndarray:
        """Acciones en circulación conocidas en cada sesión.

        El recuento de acciones de un trimestre se hace público con la nota de
        prensa, no al cierre del trimestre: la serie diaria arrastra el **último
        valor publicado**. De lo contrario la capitalización de mercado -- y con ella
        cualquier factor de tamaño construida sobre ella -- usaría durante semanas un
        recuento que todavía no se conocía.
        """
        efectivo = self._fundamentals_core.merge(
            self._event_dates[["ticker", "period_end", "announced_at"]],
            on=["ticker", "period_end"],
            how="inner",
            validate="one_to_one",
        )
        efectivo["effective_date"] = pd.DatetimeIndex(efectivo["announced_at"]).normalize()
        wide = efectivo.pivot_table(
            index="effective_date",
            columns="ticker",
            values="shares_outstanding",
            aggfunc="last",
        ).reindex(columns=list(self.tickers))
        daily = wide.reindex(wide.index.union(self.sessions)).ffill().reindex(self.sessions)
        return daily.bfill().to_numpy()

    # ------------------------------------------------------------ estimaciones

    @cached_property
    def _estimates_panel(self) -> pd.DataFrame:
        """Serie temporal del consenso con walk-down y revisiones.

        El consenso arranca por encima del resultado que acabará publicándose y
        desciende hacia un objetivo batible; en los eventos filtrados, además, la
        parte final de la senda deriva en el signo de la sorpresa, que es lo que mide
        `analyst_revision_drift` en `events.PreEventFeatures`.
        """
        cfg = self.config
        spec = cfg.leak
        table = self._event_table
        n_snap = cfg.estimate_snapshots
        n_ev_total = len(table)

        offsets = np.linspace(cfg.estimate_horizon_days, 1, n_snap)
        weight = (offsets / cfg.estimate_horizon_days) ** 1.25

        # Ruido y sesgo inicial por empresa (un flujo por ticker, independiente del
        # tamaño del universo), reordenados al orden de `table`.
        path_noise = np.empty((n_snap, n_ev_total))
        start_bias = np.empty(n_ev_total)
        dispersion_scale = np.empty(n_ev_total)
        positions = np.empty(n_ev_total, dtype=int)
        cursor = 0
        for _ticker, sub in table.sort_values(["ticker", "period_end"]).groupby(
            "ticker", sort=True
        ):
            rng = _stream(self.seed, "estimates", _ticker)
            n_ev = len(sub)
            block = slice(cursor, cursor + n_ev)
            path_noise[:, block] = _ar1(rng, n_snap, n_ev, 0.55, cfg.estimate_noise)
            start_bias[block] = cfg.walk_down * rng.uniform(0.4, 1.6, n_ev)
            dispersion_scale[block] = rng.uniform(0.5, 1.3, n_ev)
            positions[block] = table.index.get_indexer(sub.index)
            cursor += n_ev
        order = np.argsort(positions)
        path_noise = path_noise[:, order]
        start_bias, dispersion_scale = start_bias[order], dispersion_scale[order]

        sigma = table["eps_surprise_sigma"].to_numpy()[None, :]
        final = table["eps_estimate"].to_numpy()[None, :]
        wcol = weight[:, None]
        values = final + sigma * (start_bias[None, :] * wcol + path_noise * wcol)

        leaked = table["is_leaked"].to_numpy(dtype=bool)[None, :]
        window = table["leak_window_days"].to_numpy(dtype=float)[None, :]
        in_window = (offsets[:, None] <= window * 1.45) & leaked
        span = np.where(in_window.any(axis=0), np.max(np.where(in_window, offsets[:, None], 0), 0), 1.0)
        remaining = np.where(in_window, offsets[:, None] / np.maximum(span[None, :], 1.0), 0.0)
        values = values - (
            table["surprise_sign"].to_numpy()[None, :]
            * spec.revision_shift
            * table["leak_intensity"].to_numpy()[None, :]
            * sigma
            * remaining
        )
        values[-1, :] = final[0, :]

        dispersion = sigma * dispersion_scale[None, :] * (1.0 + 0.5 * wcol)
        analysts = table["n_analysts"].to_numpy(dtype=float)[None, :] - (offsets / 30).astype(int)[
            :, None
        ]
        as_of = (
            pd.DatetimeIndex(table["announced_at"]).normalize().to_numpy()[None, :]
            - (offsets.astype("int64") * np.timedelta64(1, "D"))[:, None]
        )
        rev_est = table["revenue_estimate"].to_numpy()[None, :] * (
            1.0 + 0.004 * wcol * start_bias[None, :]
        )

        out = pd.DataFrame(
            {
                "ticker": np.tile(table["ticker"].to_numpy(), n_snap),
                "event_id": np.tile(table["event_id"].to_numpy(), n_snap),
                "period_end": np.tile(
                    pd.DatetimeIndex(table["period_end"]).to_numpy(), n_snap
                ),
                "as_of": as_of.reshape(-1),
                "available_at": as_of.reshape(-1),
                "eps_mean": values.reshape(-1),
                "eps_median": (values - 0.05 * sigma).reshape(-1),
                "eps_std": dispersion.reshape(-1),
                "eps_high": (values + 1.9 * dispersion).reshape(-1),
                "eps_low": (values - 1.9 * dispersion).reshape(-1),
                "n_analysts": np.maximum(analysts, 3.0).reshape(-1).astype(int),
                "revenue_mean": rev_est.reshape(-1),
                "revenue_std": np.tile(table["revenue_actual"].to_numpy() * 0.012, n_snap),
                "source": "synthetic",
            }
        )
        return out.sort_values(["ticker", "period_end", "as_of"]).reset_index(drop=True)

    # ---------------------------------------------------------------- opciones

    @cached_property
    def _options_daily(self) -> pd.DataFrame:
        """Resumen diario de la superficie de volatilidad y del flujo de opciones.

        La varianza del anuncio entra de forma aditiva en la varianza total hasta el
        vencimiento (Dubinsky y Johannes 2006): ``iv(T)^2 * T = sigma^2 * T + jump^2``
        cuando el anuncio cae antes del vencimiento. De ahí salen, sin imponerlas a
        mano, la subida de IV al acercarse el evento y su colapso el día en que la
        noticia se publica.

        `days_to_earnings` vale 9999 cuando no queda ningún anuncio dentro del panel;
        es un centinela explícito y no un NaN, para que el panel no tenga huecos.
        Deliberadamente **no** se publica ninguna magnitud del anuncio futuro: el
        calendario de resultados es público por adelantado, la sorpresa no.
        """
        cfg = self.config
        prices = self._price_panel
        meta = self._meta
        n_days = len(self.sessions)

        # La volatilidad de referencia es la *difusiva*: se excluye el retorno del
        # día del anuncio, porque el salto de resultados se modela aparte como
        # varianza de evento. Si no se excluyera, la volatilidad realizada saltaría
        # justo después del anuncio y taparía por completo el colapso de IV.
        returns = self._wide(prices["log_return"])
        _, _, is_event_flag = self._event_effects
        diffusive = returns.mask(pd.DataFrame(is_event_flag, index=returns.index,
                                              columns=returns.columns))
        realized = diffusive.rolling(20, min_periods=5).std() * _SQRT_TRADING
        realized = realized.bfill().ffill().to_numpy()

        rng = _stream(self.seed, "iv")
        iv_noise = _ar1(rng, n_days, self.n_tickers, cfg.iv_ar, cfg.iv_noise * 0.1)
        base_vol = np.maximum(realized * cfg.iv_premium * np.exp(iv_noise), 0.05)

        days_to_event, event_pos_flag = self._days_to_event_matrix()
        jump = meta["jump_vol"].to_numpy()[None, :]
        horizons = {"7d": 7.0, "30d": 30.0, "60d": 60.0, "90d": 90.0}
        surface: dict[str, np.ndarray] = {}
        for label, days in horizons.items():
            tau = days / 365.0
            term = base_vol + cfg.iv_short_long_gap * (math.exp(-tau / cfg.iv_term_tau) - 0.5)
            term = np.maximum(term, 0.05)
            # La varianza del anuncio solo cuenta si el anuncio *todavía no* ha
            # ocurrido. En `event_date` la noticia ya es pública desde la apertura
            # (BMO ese día, AMC la víspera), así que el colapso de IV se produce en
            # tau=0, no al día siguiente.
            includes_event = (days_to_event <= days) & (days_to_event >= 1)
            total_var = term**2 * tau + np.where(includes_event, jump**2, 0.0)
            surface[label] = np.clip(np.sqrt(total_var / tau), _MIN_IV, _MAX_IV)

        vs_rng = _stream(self.seed, "volspread")
        vol_spread = _ar1(vs_rng, n_days, self.n_tickers, 0.80, cfg.vol_spread_sigma * 0.6)
        skew_rng = _stream(self.seed, "skew")
        skew = 0.055 + _ar1(skew_rng, n_days, self.n_tickers, 0.85, 0.006)

        orng = _stream(self.seed, "optvolume")
        liq = meta["option_liquidity"].to_numpy()[None, :]
        base_opt_volume = (
            self._wide(prices["volume"]).to_numpy() * 0.0016 * liq
        )
        opt_noise = _ar1(orng, n_days, self.n_tickers, 0.75, 0.35)
        event_boost = np.where(
            days_to_event <= 5, 0.55 * (6 - np.maximum(days_to_event, 0)) / 6, 0.0
        )
        total_volume = np.maximum(base_opt_volume * np.exp(opt_noise + event_boost), 10.0)
        pc_ratio = cfg.put_call_volume_ratio * np.exp(
            orng.normal(0.0, 0.22, (n_days, self.n_tickers))
        )
        call_volume = total_volume / (1.0 + pc_ratio)
        put_volume = total_volume - call_volume
        oi_noise = _ar1(orng, n_days, self.n_tickers, 0.97, 0.08)
        call_oi = call_volume * 7.5 * np.exp(oi_noise)
        put_oi = put_volume * 7.5 * np.exp(oi_noise)

        leak_call, leak_put, leak_oi_c, leak_oi_p, leak_vs, leak_skew = self._option_leak_matrices()
        call_volume *= np.exp(leak_call)
        put_volume *= np.exp(leak_put)
        call_oi *= 1.0 + leak_oi_c
        put_oi *= 1.0 + leak_oi_p
        vol_spread = vol_spread + leak_vs
        skew = skew + leak_skew

        index = pd.MultiIndex.from_product(
            [self.sessions, list(self.tickers)], names=["date", "ticker"]
        )
        panel = pd.DataFrame(
            {
                "iv_atm_7d": surface["7d"].reshape(-1),
                "iv_atm_30d": surface["30d"].reshape(-1),
                "iv_atm_60d": surface["60d"].reshape(-1),
                "iv_atm_90d": surface["90d"].reshape(-1),
                "base_vol": base_vol.reshape(-1),
                "iv_term_slope": (surface["90d"] - surface["30d"]).reshape(-1),
                "iv_skew_25delta": skew.reshape(-1),
                "vol_spread": vol_spread.reshape(-1),
                "call_volume": np.round(call_volume).reshape(-1),
                "put_volume": np.round(put_volume).reshape(-1),
                "call_open_interest": np.round(call_oi).reshape(-1),
                "put_open_interest": np.round(put_oi).reshape(-1),
                "days_to_earnings": days_to_event.reshape(-1),
                "realized_vol_20d": realized.reshape(-1),
                "is_event_date": event_pos_flag.reshape(-1),
            },
            index=index,
        )
        panel["option_volume"] = panel["call_volume"] + panel["put_volume"]
        panel["put_call_volume_ratio"] = panel["put_volume"] / panel["call_volume"].clip(lower=1.0)
        panel["put_call_oi_ratio"] = panel["put_open_interest"] / panel[
            "call_open_interest"
        ].clip(lower=1.0)
        return panel

    def _days_to_event_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        """Días naturales hasta el siguiente anuncio y marca del día del anuncio.

        El calendario de resultados se anuncia con semanas de antelación, así que
        conocer la *fecha* del próximo anuncio no es información privilegiada. La
        magnitud de la sorpresa, en cambio, no se expone en ningún panel diario.
        """
        n_days, n_names = len(self.sessions), self.n_tickers
        days = np.full((n_days, n_names), 9999.0)
        flag = np.zeros((n_days, n_names))
        col = {t: i for i, t in enumerate(self.tickers)}
        session_values = self.sessions.to_numpy().astype("datetime64[D]").astype(int)
        for ticker, sub in self._events_core.groupby("ticker", sort=False):
            j = col[ticker]
            ev_days = np.sort(
                pd.DatetimeIndex(sub["event_date"]).to_numpy().astype("datetime64[D]").astype(int)
            )
            idx = np.searchsorted(ev_days, session_values, side="left")
            valid = idx < len(ev_days)
            days[valid, j] = ev_days[idx[valid]] - session_values[valid]
            pos = self.sessions.get_indexer(pd.DatetimeIndex(sub["event_date"]))
            flag[pos[pos >= 0], j] = 1.0
        return days, flag

    def _option_leak_matrices(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Huella de la filtración en el flujo de opciones, por canal."""
        spec = self.config.leak
        shape = (len(self.sessions), self.n_tickers)
        call_v, put_v = np.zeros(shape), np.zeros(shape)
        call_oi, put_oi = np.zeros(shape), np.zeros(shape)
        vspread, skew = np.zeros(shape), np.zeros(shape)
        col = {t: i for i, t in enumerate(self.tickers)}
        for row in self._leaks.itertuples(index=False):
            if not row.is_leaked:
                continue
            j = col[row.ticker]
            width, kappa, sign = int(row.leak_window_days), row.leak_intensity, row.surprise_sign
            pos = int(row.event_pos)
            ramp = np.linspace(1.0 / width, 1.0, width)
            sl = slice(pos - width, pos)
            bullish = sign > 0
            call_v[sl, j] += spec.option_volume_log_amp * kappa * ramp * (1.0 if bullish else 0.25)
            put_v[sl, j] += spec.option_volume_log_amp * kappa * ramp * (0.25 if bullish else 1.0)
            call_oi[sl, j] += spec.oi_buildup * kappa * ramp * (1.0 if bullish else 0.2)
            put_oi[sl, j] += spec.oi_buildup * kappa * ramp * (0.2 if bullish else 1.0)
            vspread[sl, j] += sign * spec.vol_spread_shift * kappa * ramp
            skew[sl, j] -= sign * spec.skew_shift * kappa * ramp
        return call_v, put_v, call_oi, put_oi, vspread, skew

    # -------------------------------------------------------------------- flujo

    @cached_property
    def _short_interest(self) -> pd.DataFrame:
        """Interés corto quincenal con el calendario y el retardo de FINRA."""
        cfg = self.config
        settlements: list[dt.date] = []
        cursor = dt.date(self.start.year, self.start.month, 1)
        while cursor <= self.end:
            mid = dt.date(cursor.year, cursor.month, 15)
            eom = _month_end(cursor.year, cursor.month)
            for day in (mid, eom):
                if self.start <= day <= self.end:
                    settlements.append(self.calendar.session_on_or_before(day))
            cursor = (pd.Timestamp(cursor) + pd.offsets.MonthBegin(1)).date()
        settlements = sorted(set(settlements))
        n_settle = len(settlements)
        if n_settle == 0:
            msg = "el rango no contiene ninguna fecha de liquidación de short interest"
            raise InsufficientHistory(msg)

        rng = _stream(self.seed, "shortinterest")
        noise = _ar1(rng, n_settle, self.n_tickers, cfg.short_interest_ar, cfg.short_interest_sigma)
        base = self._meta["short_base"].to_numpy()[None, :]
        ratio = np.clip(base * np.exp(noise), 0.001, 0.45)

        leak_mat = np.zeros((n_settle, self.n_tickers))
        col = {t: i for i, t in enumerate(self.tickers)}
        settle_ts = pd.DatetimeIndex(settlements)
        for row in self._leaks.itertuples(index=False):
            if not row.is_leaked or row.surprise_sign >= 0:
                continue
            event_day = pd.Timestamp(row.event_date)
            window_start = event_day - pd.Timedelta(days=int(row.leak_window_days) * 2)
            hit = (settle_ts >= window_start) & (settle_ts < event_day)
            leak_mat[hit, col[row.ticker]] += (
                self.config.leak.short_interest_shift * float(row.leak_intensity)
            )
        ratio = ratio * (1.0 + leak_mat)

        shares = self._shares_panel()
        pos = self.sessions.get_indexer(settle_ts)
        shares_at = shares[pos]
        adv = (
            self._wide(self._price_panel["volume"]).rolling(20, min_periods=5).mean().to_numpy()[pos]
        )
        shares_short = ratio * shares_at
        available = [
            pd.Timestamp(self.calendar.shift(d, cfg.short_publication_lag)) for d in settlements
        ]
        index = pd.MultiIndex.from_product(
            [settle_ts, list(self.tickers)], names=["settlement_date", "ticker"]
        )
        frame = pd.DataFrame(
            {
                "shares_short": np.round(shares_short).reshape(-1),
                "shares_outstanding": shares_at.reshape(-1),
                "short_percent_shares": ratio.reshape(-1),
                "avg_daily_volume": adv.reshape(-1),
            },
            index=index,
        ).reset_index()
        avail_map = dict(zip(settle_ts, available, strict=True))
        frame["available_at"] = frame["settlement_date"].map(avail_map)
        frame["days_to_cover"] = frame["shares_short"] / frame["avg_daily_volume"].clip(lower=1.0)
        prev = frame.sort_values(["ticker", "settlement_date"]).groupby("ticker", sort=False)[
            "short_percent_shares"
        ].shift(1)
        frame["short_interest_delta"] = frame["short_percent_shares"] - prev.reindex(frame.index)
        return frame.sort_values(["settlement_date", "ticker"]).reset_index(drop=True)

    @cached_property
    def _off_exchange(self) -> pd.DataFrame:
        """Volumen semanal negociado fuera de bolsa (ATS y no-ATS) con su retardo."""
        cfg = self.config
        volume = self._wide(self._price_panel["volume"])
        weeks = volume.resample("W-FRI").sum()
        week_end = weeks.index
        n_weeks = len(week_end)
        if n_weeks == 0:  # pragma: no cover - imposible con >= 60 sesiones
            msg = "el rango no contiene ninguna semana completa"
            raise InsufficientHistory(msg)

        rng = _stream(self.seed, "offexchange")
        noise = _ar1(rng, n_weeks, self.n_tickers, cfg.offex_ar, cfg.offex_sigma)
        base = self._meta["offex_base"].to_numpy()[None, :]
        share = np.clip(base + noise, 0.15, 0.75)

        col = {t: i for i, t in enumerate(self.tickers)}
        for row in self._leaks.itertuples(index=False):
            if not row.is_leaked:
                continue
            event_day = pd.Timestamp(row.event_date)
            window_start = event_day - pd.Timedelta(days=int(row.leak_window_days) * 7 // 5 + 1)
            hit = (week_end >= window_start) & (week_end < event_day + pd.Timedelta(days=1))
            share[hit, col[row.ticker]] += (
                self.config.leak.offex_share_shift * float(row.leak_intensity)
            )
        share = np.clip(share, 0.05, 0.90)

        total = weeks.to_numpy()
        off = total * share
        ats = off * 0.62
        index = pd.MultiIndex.from_product(
            [week_end, list(self.tickers)], names=["week_end", "ticker"]
        )
        frame = pd.DataFrame(
            {
                "total_volume": total.reshape(-1),
                "off_exchange_volume": np.round(off).reshape(-1),
                "ats_volume": np.round(ats).reshape(-1),
                "non_ats_volume": np.round(off - ats).reshape(-1),
                "off_exchange_share": share.reshape(-1),
            },
            index=index,
        ).reset_index()
        frame["week_start"] = frame["week_end"] - pd.Timedelta(days=4)
        frame["available_at"] = frame["week_end"] + pd.Timedelta(days=cfg.offex_publication_lag)
        prev = frame.sort_values(["ticker", "week_end"]).groupby("ticker", sort=False)[
            "off_exchange_share"
        ].shift(1)
        frame["off_exchange_share_delta"] = frame["off_exchange_share"] - prev.reindex(frame.index)
        cols = [
            "week_start",
            "week_end",
            "ticker",
            "total_volume",
            "off_exchange_volume",
            "ats_volume",
            "non_ats_volume",
            "off_exchange_share",
            "off_exchange_share_delta",
            "available_at",
        ]
        return frame[cols].sort_values(["week_end", "ticker"]).reset_index(drop=True)

    # ------------------------------------------------------------------- API

    def prices(
        self,
        tickers: Ticker | Sequence[Ticker] | None = None,
        start: DateLike | None = None,
        end: DateLike | None = None,
    ) -> pd.DataFrame:
        """Panel OHLCV con MultiIndex ``(date, ticker)``.

        Columnas: ``open, high, low, close, volume, adj_close, dividend,
        split_factor, log_return, shares_outstanding, dollar_volume, market_cap``.
        El submuestreo no altera valores: el panel se genera una vez y se recorta.
        """
        panel = self._price_panel
        if tickers is not None:
            wanted = self._resolve_tickers(tickers)
            panel = panel[panel.index.get_level_values("ticker").isin(wanted)]
        if start is not None or end is not None:
            lo = pd.Timestamp(_as_date(start)) if start is not None else self.sessions[0]
            hi = pd.Timestamp(_as_date(end)) if end is not None else self.sessions[-1]
            dates = panel.index.get_level_values("date")
            panel = panel[(dates >= lo) & (dates <= hi)]
        if len(panel) == 0:
            msg = (
                "la selección de precios está vacía: revisa el rango de fechas "
                f"({start} a {end}) frente al panel disponible "
                f"({self.sessions[0].date()} a {self.sessions[-1].date()})"
            )
            raise InsufficientHistory(msg)
        return panel.copy()

    def bars(self, ticker: Ticker) -> list[Bar]:
        """Serie de `types.Bar` de un símbolo, útil para probar adaptadores."""
        sub = self.prices(tickers=ticker).droplevel("ticker")
        return [
            Bar(
                ticker=normalize_ticker(ticker),
                ts=ts.to_pydatetime(),
                open=float(r.open),
                high=float(r.high),
                low=float(r.low),
                close=float(r.close),
                volume=float(r.volume),
                adj_close=float(r.adj_close),
            )
            for ts, r in zip(sub.index, sub.itertuples(index=False), strict=True)
        ]

    def market_index(self) -> pd.DataFrame:
        """Índice de mercado sintético (equivalente a un ETF sobre el índice).

        Necesario para `events.abnormal_returns(model="market")`: sin serie de
        mercado no hay retorno anormal que calcular.
        """
        factors = self._factor_returns
        level = 100.0 * np.exp(np.cumsum(factors["mkt"].to_numpy()))
        prev = np.concatenate([[100.0], level[:-1]])
        return pd.DataFrame(
            {
                "close": level,
                "open": prev * np.exp(0.35 * factors["mkt"].to_numpy()),
                "log_return": factors["mkt"].to_numpy(),
                "volume": 8e8 * np.exp(factors["vol_multiplier"].to_numpy() - 1.0),
            },
            index=self.sessions,
        )

    def factor_returns(self) -> pd.DataFrame:
        """Retornos verdaderos de los factores comunes (mercado, SMB, HML, sector, rf)."""
        return self._factor_returns.copy()

    def fundamentals(self, tickers: Ticker | Sequence[Ticker] | None = None) -> pd.DataFrame:
        """Estados financieros trimestrales con `available_at` point-in-time.

        `available_at` es el instante de la nota de prensa (8-K de resultados), que es
        cuando las cifras se hacen públicas; `filed_at` es la presentación posterior
        del 10-Q/10-K. Un factor conservador debe usar `filed_at`; uno que explote la
        nota de prensa, `available_at`. Nunca `period_end`.
        """
        core = self._fundamentals_core
        dates = self._event_dates[
            ["ticker", "period_end", "announced_at", "filed_at", "form", "event_date", "event_id"]
        ]
        merged = core.merge(dates, on=["ticker", "period_end"], how="inner", validate="one_to_one")
        merged = merged.rename(columns={"announced_at": "available_at"})
        if tickers is not None:
            merged = merged[merged["ticker"].isin(self._resolve_tickers(tickers))]
        first = ["ticker", "period_end", "fiscal_period", "available_at", "filed_at", "form"]
        rest = [c for c in merged.columns if c not in first]
        out = merged[[*first, *rest]]
        return out.sort_values(["ticker", "period_end"]).reset_index(drop=True)

    def fundamental_facts(self, concepts: Sequence[str] | None = None) -> pd.DataFrame:
        """Los mismos fundamentales en formato largo de vintages (`pit.restatements`).

        Columnas ``ticker, concept, period_end, available_at, value, fiscal_period,
        unit, form, accession, is_restated``. Es el formato que consumen
        `first_reported`/`as_restated`, de modo que el módulo de vintages puede
        probarse sin red.
        """
        default = (
            "revenue",
            "net_income",
            "eps_diluted",
            "cfo",
            "capex",
            "total_assets",
            "total_equity",
            "total_debt",
            "gross_profit",
            "operating_income",
        )
        chosen = list(concepts) if concepts is not None else list(default)
        wide = self.fundamentals()
        missing = [c for c in chosen if c not in wide.columns]
        if missing:
            msg = f"conceptos inexistentes en el panel fundamental: {missing}"
            raise DataQualityError(msg)
        long = wide.melt(
            id_vars=["ticker", "period_end", "available_at", "fiscal_period", "form"],
            value_vars=chosen,
            var_name="concept",
            value_name="value",
        )
        long["unit"] = np.where(long["concept"].str.startswith("eps"), "USD/share", "USD")
        long["accession"] = [
            f"SYN-{pd.Timestamp(p).strftime('%Y%m%d')}-{t}"
            for t, p in zip(long["ticker"], long["period_end"], strict=True)
        ]
        long["is_restated"] = False
        cols = [
            "ticker",
            "concept",
            "period_end",
            "available_at",
            "value",
            "fiscal_period",
            "unit",
            "form",
            "accession",
            "is_restated",
        ]
        return long[cols].sort_values(["ticker", "concept", "period_end"]).reset_index(drop=True)

    def events(self, *, include_prehistory: bool = False) -> pd.DataFrame:
        """Tabla de anuncios de resultados.

        Por defecto se devuelven solo los eventos cuya sesión negociable cae dentro
        del panel de precios: un evento sin precios no es un evento operable y
        contaminaría cualquier estudio de eventos. `include_prehistory=True` añade
        los trimestres anteriores a `start`, que existen para que SUE y los factores
        de crecimiento tengan histórico desde el primer día.
        """
        frame = self._events_core
        if not include_prehistory:
            frame = frame[frame["event_date"] >= pd.Timestamp(self.start)]
        cols = [
            "event_id",
            "ticker",
            "cik",
            "period_end",
            "fiscal_quarter",
            "announced_at",
            "session",
            "event_date",
            "eps_actual",
            "eps_estimate",
            "eps_surprise",
            "revenue_actual",
            "revenue_estimate",
            "revenue_surprise",
            "surprise_pct",
            "sue",
            "n_analysts",
            "report_delay_days",
            "is_estimated_date",
            "source",
        ]
        return frame[cols].sort_values(["event_date", "ticker"]).reset_index(drop=True)

    def earnings_events(self, *, include_prehistory: bool = False) -> list[EarningsEvent]:
        """Los mismos eventos como `types.EarningsEvent`, para probar adaptadores."""
        frame = self.events(include_prehistory=include_prehistory)
        return [
            EarningsEvent(
                ticker=row.ticker,
                period_end=pd.Timestamp(row.period_end).date(),
                announced_at=pd.Timestamp(row.announced_at).to_pydatetime(),
                session=Session(row.session),
                fiscal_quarter=row.fiscal_quarter,
                cik=row.cik,
                eps_actual=float(row.eps_actual),
                eps_estimate=float(row.eps_estimate),
                revenue_actual=float(row.revenue_actual),
                revenue_estimate=float(row.revenue_estimate),
                source="synthetic",
                is_estimated_date=False,
            )
            for row in frame.itertuples(index=False)
        ]

    def estimates(self, tickers: Ticker | Sequence[Ticker] | None = None) -> pd.DataFrame:
        """Histórico de consenso: una fila por ``(ticker, period_end, as_of)``.

        Todos los `as_of` son **estrictamente anteriores** al anuncio. El consenso
        parte optimista y desciende hacia un objetivo batible (walk-down), que es el
        patrón documentado en Richardson, Teoh y Wysocki (2004); por eso la sorpresa
        media del panel es positiva y no centrada en cero.
        """
        frame = self._estimates_panel
        if tickers is not None:
            frame = frame[frame["ticker"].isin(self._resolve_tickers(tickers))]
        return frame.reset_index(drop=True)

    def options_daily(self) -> pd.DataFrame:
        """Panel diario ``(date, ticker)`` con el resumen de la superficie y el flujo."""
        return self._options_daily.copy()

    def options_chain(
        self,
        tickers: Ticker | Sequence[Ticker] | None = None,
        asof: DateLike | Sequence[DateLike] | None = None,
        *,
        n_expiries: int | None = None,
        n_strikes: int | None = None,
    ) -> pd.DataFrame:
        """Cadena de opciones para las fechas y símbolos indicados.

        Se construye a partir de los parámetros latentes del panel diario, de modo
        que los agregados de la cadena (volumen e interés abierto por lado) coinciden
        exactamente con los del panel: son la misma realidad vista con dos
        granularidades. El reparto entre strikes usa el método del mayor resto, así
        que las sumas cuadran al contrato.

        Sin argumentos devuelve la cadena de la última sesión para todos los
        símbolos. Pedir muchas fechas y muchos símbolos a la vez multiplica filas:
        ``n_fechas * n_símbolos * n_vencimientos * n_strikes * 2``.
        """
        cfg = self.config
        n_exp = int(n_expiries or cfg.option_expiries)
        n_str = int(n_strikes or cfg.option_strikes)
        if n_exp < 1 or n_str < 1:
            msg = "n_expiries y n_strikes deben ser >= 1"
            raise ConfigError(msg)
        if n_str % 2 == 0:
            n_str += 1  # se exige un strike central

        names = list(self.tickers) if tickers is None else self._resolve_tickers(tickers)
        if asof is None:
            days = [self.sessions[-1]]
        elif isinstance(asof, (str, dt.date, dt.datetime, pd.Timestamp)):
            days = [pd.Timestamp(_as_date(asof))]
        else:
            days = [pd.Timestamp(_as_date(d)) for d in asof]
        unknown_days = [d for d in days if d not in self.sessions]
        if unknown_days:
            msg = f"fechas que no son sesión del panel: {[str(d.date()) for d in unknown_days]}"
            raise DataQualityError(msg)

        daily = self._options_daily
        prices = self._price_panel
        rows: list[pd.DataFrame] = []
        rate = cfg.risk_free
        for day in days:
            expiries = self._expiries_after(day.date(), n_exp)
            for ticker in names:
                key = (day, ticker)
                params = daily.loc[key]
                spot = float(prices.loc[key, "close"])
                div_yield = float(self._meta.loc[ticker, "dividend_yield"])
                rows.append(
                    self._chain_for(
                        ticker=ticker,
                        day=day,
                        spot=spot,
                        expiries=expiries,
                        params=params,
                        n_strikes=n_str,
                        rate=rate,
                        div_yield=div_yield,
                    )
                )
        out = pd.concat(rows, ignore_index=True)
        return out.sort_values(["as_of", "ticker", "expiry", "right", "strike"]).reset_index(
            drop=True
        )

    def short_interest(self, tickers: Ticker | Sequence[Ticker] | None = None) -> pd.DataFrame:
        """Interés corto quincenal (calendario FINRA) con su fecha de publicación.

        `settlement_date` es la fecha a la que se refiere el dato y `available_at`
        la de su difusión, unas ocho sesiones después. Usar la primera como fecha de
        señal es look-ahead puro: el mercado no conocía la cifra ese día.
        """
        frame = self._short_interest
        if tickers is not None:
            frame = frame[frame["ticker"].isin(self._resolve_tickers(tickers))]
        return frame.reset_index(drop=True)

    def off_exchange(self, tickers: Ticker | Sequence[Ticker] | None = None) -> pd.DataFrame:
        """Volumen semanal fuera de bolsa (ATS y no-ATS) con su fecha de publicación.

        La transparencia de FINRA publica la semana con dos semanas de retraso para
        valores NMS Tier 1. Es una restricción real: la subida de cuota off-exchange
        de la semana previa a un anuncio existe en el dato, pero no era observable
        antes del anuncio. El generador la reproduce para que ningún backtest la use
        como si lo fuera.
        """
        frame = self._off_exchange
        if tickers is not None:
            frame = frame[frame["ticker"].isin(self._resolve_tickers(tickers))]
        return frame.reset_index(drop=True)

    def leaked_event_ids(self) -> list[str]:
        """Identificadores de los eventos con huella pre-anuncio inyectada.

        Es la verdad-terreno del banco de pruebas: con ella se calculan precisión,
        recall y curva ROC de cualquier detector, algo imposible con datos reales,
        donde el conjunto de eventos con negociación informada es inobservable.
        """
        leaks = self._leaks
        ids = leaks.loc[leaks["is_leaked"], "event_id"]
        return sorted(ids.tolist())

    def ground_truth(self) -> pd.DataFrame:
        """Verdad-terreno completa por evento, indexada por `event_id`.

        Contiene la marca de filtración, su intensidad y anchura, el signo y la
        magnitud latente de la sorpresa. Se mantiene **fuera** de `events()` a
        propósito: si la etiqueta viajara con el feed, cualquier modelo entrenado
        sobre él estaría contaminado.
        """
        leaks = self._leaks.set_index("event_id")
        cols = [
            "ticker",
            "event_date",
            "is_leaked",
            "leak_window_days",
            "leak_intensity",
            "surprise_sign",
            "surprise_z",
        ]
        return leaks[cols].copy()

    # ------------------------------------------------------------------ ayudas

    def _resolve_tickers(self, tickers: Ticker | Sequence[Ticker]) -> list[Ticker]:
        """Normaliza y valida una selección de símbolos."""
        if isinstance(tickers, str):
            wanted = [normalize_ticker(tickers)]
        elif isinstance(tickers, Iterable):
            wanted = [normalize_ticker(t) for t in tickers]
        else:  # pragma: no cover - protegido por el tipo
            msg = f"selección de símbolos no interpretable: {tickers!r}"
            raise ConfigError(msg)
        known = set(self.tickers)
        unknown = [t for t in wanted if t not in known]
        if unknown:
            msg = f"símbolos fuera del universo sintético: {sorted(unknown)}"
            raise DataQualityError(msg)
        if not wanted:
            msg = "la selección de símbolos está vacía"
            raise DataQualityError(msg)
        return wanted

    def _expiries_after(self, day: dt.date, n_expiries: int) -> list[dt.date]:
        """Los siguientes vencimientos mensuales estándar (tercer viernes)."""
        out: list[dt.date] = []
        year, month = day.year, day.month
        while len(out) < n_expiries:
            candidate = _third_friday(year, month)
            if candidate > day:
                out.append(candidate)
            month += 1
            if month > 12:
                month, year = 1, year + 1
        return out

    def _chain_for(
        self,
        *,
        ticker: Ticker,
        day: pd.Timestamp,
        spot: float,
        expiries: Sequence[dt.date],
        params: pd.Series,
        n_strikes: int,
        rate: float,
        div_yield: float,
    ) -> pd.DataFrame:
        """Construye la cadena de un ``(ticker, fecha)`` a partir del panel diario."""
        cfg = self.config
        increment = 2.5 if spot < 50 else (5.0 if spot < 200 else 10.0)
        centre = round(spot / increment) * increment
        offsets = np.arange(-(n_strikes // 2), n_strikes // 2 + 1)
        strikes = np.maximum(centre + offsets * increment, increment / 2.0)

        days_to_event = float(params["days_to_earnings"])
        base_vol = float(params["base_vol"])
        jump = float(self._meta.loc[ticker, "jump_vol"])
        skew_25 = float(params["iv_skew_25delta"])
        vol_spread = float(params["vol_spread"])

        n_exp = len(expiries)
        tau_days = np.array([max((e - day.date()).days, 1) for e in expiries], dtype=float)
        tau = tau_days / 365.0
        # Estructura temporal de la componente difusiva: el corto plazo cotiza por
        # encima o por debajo del largo según el régimen, con reversión exponencial.
        term = np.maximum(
            base_vol + cfg.iv_short_long_gap * (np.exp(-tau / cfg.iv_term_tau) - 0.5), 0.05
        )
        covers_event = (days_to_event >= 1.0) & (days_to_event <= tau_days)
        atm_iv = np.sqrt((term**2 * tau + np.where(covers_event, jump**2, 0.0)) / tau)
        fwd = spot * np.exp((rate - div_yield) * tau)

        grid_strike = np.tile(strikes, n_exp)
        grid_tau = np.repeat(tau, n_strikes)
        grid_tau_days = np.repeat(tau_days, n_strikes)
        grid_atm = np.repeat(atm_iv, n_strikes)
        grid_fwd = np.repeat(fwd, n_strikes)
        grid_covers = np.repeat(covers_event, n_strikes)

        # Moneyness estandarizada, acotada: en vencimientos muy cortos los strikes
        # listados quedan a muchas sigmas y una parábola sin acotar dispararía las
        # alas hasta volatilidades absurdas.
        moneyness = np.clip(
            np.log(grid_strike / grid_fwd) / np.maximum(grid_atm * np.sqrt(grid_tau), 1e-6),
            -3.0,
            3.0,
        )
        smile = grid_atm * (
            1.0
            + (cfg.smile_skew - skew_25 * 0.8) * moneyness
            + cfg.smile_curvature * moneyness**2
        )
        smile = np.clip(smile, _MIN_IV, _MAX_IV)
        # Peso de liquidez: se concentra en torno al dinero y decae con el plazo.
        weight = np.exp(-0.5 * (moneyness / 0.9) ** 2) * np.exp(-grid_tau_days / 120.0)

        liquidity = max(float(self._meta.loc[ticker, "option_liquidity"]), 0.2)
        blocks: list[pd.DataFrame] = []
        for right in ("C", "P"):
            is_call = right == "C"
            iv = np.clip(smile + (0.5 if is_call else -0.5) * vol_spread, _MIN_IV, _MAX_IV)
            price, delta, gamma, vega = _bs_greeks(
                np.full(len(grid_strike), spot),
                grid_strike,
                grid_tau,
                iv,
                rate,
                np.full(len(grid_strike), div_yield),
                np.full(len(grid_strike), is_call, dtype=bool),
            )
            side_volume = float(params["call_volume" if is_call else "put_volume"])
            side_oi = float(params["call_open_interest" if is_call else "put_open_interest"])
            spread = np.maximum(0.02, price * 0.02 / liquidity)
            blocks.append(
                pd.DataFrame(
                    {
                        "as_of": day,
                        "ticker": ticker,
                        "expiry": np.repeat(pd.DatetimeIndex(expiries).to_numpy(), n_strikes),
                        "right": right,
                        "strike": grid_strike,
                        "spot": spot,
                        "forward": grid_fwd,
                        "tau_years": grid_tau,
                        "days_to_expiry": grid_tau_days.astype(int),
                        "iv": iv,
                        "mid": np.maximum(price, 0.01),
                        "bid": np.maximum(price - spread / 2.0, 0.0),
                        "ask": np.maximum(price + spread / 2.0, 0.02),
                        "delta": delta,
                        "gamma": gamma,
                        "vega": vega,
                        "volume": _largest_remainder(weight, side_volume),
                        "open_interest": _largest_remainder(weight, side_oi),
                        "days_to_earnings": days_to_event,
                        "covers_earnings": grid_covers,
                    }
                )
            )
        return pd.concat(blocks, ignore_index=True)


def _largest_remainder(weights: np.ndarray, total: float) -> np.ndarray:
    """Reparte `total` entre `weights` en enteros cuya suma es exactamente el total.

    Método del mayor resto: evita que la suma de la cadena difiera del agregado
    diario por errores de redondeo, que es la clase de inconsistencia que hace que un
    test de coherencia entre granularidades falle por motivos triviales.
    """
    target = round(total)
    if target <= 0:
        return np.zeros(len(weights))
    raw = weights / weights.sum() * target
    base = np.floor(raw)
    remainder = target - int(base.sum())
    if remainder > 0:
        order = np.argsort(-(raw - base))
        base[order[:remainder]] += 1.0
    return base


def make_synthetic_market(seed: int = 20260803, **kwargs: object) -> SyntheticMarket:
    """Atajo para construir un `SyntheticMarket` desde configuración por palabras clave."""
    return SyntheticMarket(seed=seed, **kwargs)  # type: ignore[arg-type]
