"""Modelo de costes de transacción del backtest (contrato `ARCHITECTURE.md` §3.7).

El contrato es explícito: *"spread por tramo de liquidez + impacto
sqrt(participación) + comisión + coste de préstamo para cortos. Nada de '0.05 %
fijo' sin justificar"*. Este módulo implementa exactamente esas cuatro patas, con
cada parámetro documentado y con fuente, porque los costes son uno de los tres
cuellos de botella reales del proyecto (`docs/research/validation_methodology.md`
§11.3: "los costes de transacción no aparecen en ningún t").

Las cuatro componentes
----------------------

1. **Medio spread por tramo de liquidez.** Cruzar el spread cuesta la mitad del
   spread cotizado en cada sentido. El spread efectivo decrece con el volumen en
   dólares (ADV): es el hecho empírico más robusto de la microestructura
   (Hasbrouck 2009; Corwin y Schultz 2012; Abdi y Ranaldo 2017). Los tramos por
   defecto están calibrados para el S&P 500 moderno: las megacaps cotizan con
   spreads de 1-3 pb (medio spread ~1 pb) y la cola menos líquida del índice con
   10-40 pb. El tramo inferior es deliberadamente punitivo: recoge liquidez
   degradada (crisis, salidas del índice).

2. **Impacto de mercado ~ raíz de la participación.** La "ley de la raíz
   cuadrada": el coste por unidad negociada es ``η · σ_diaria · √(Q/ADV)``.
   Es la forma funcional medida por Almgren, Thum, Hauptmann y Li (2005) —su
   exponente estimado, 0.6, es indistinguible de 0.5 en el rango relevante—,
   racionalizada por Gatheral (2010) y Tóth et al. (2011), y recomendada por
   Grinold y Kahn (2000, cap. 16). El coeficiente por defecto ``η = 0.5`` se
   calibra con Frazzini, Israel y Moskowitz (2018): impacto mediano medido de
   ~9 pb con participación ~1.6 % del ADV y σ diaria ~1.6 %
   (``η = 9e-4 / (0.016·√0.016) ≈ 0.45``), redondeado de forma conservadora.

3. **Comisión.** Las comisiones institucionales en EE. UU. son ~0.3-0.5 centavos
   por acción (ITG/Virtu *Global Cost Review*), es decir, 0.3-0.5 pb sobre un
   precio de $100. Frazzini, Israel y Moskowitz (2018) las documentan como
   componente menor frente al impacto. Por defecto **0.5 pb** del nocional.

4. **Coste de préstamo para cortos.** D'Avolio (2002): el 91 % del mercado es
   *general collateral* con comisión media ~17 pb/año, y las *specials* superan
   el 4 %/año. El S&P 500 es casi todo GC, de ahí el defecto de **25 pb/año**;
   `borrow_overrides_bps` permite marcar nombres *hard-to-borrow* con su tasa
   real (p. ej. estimada vía `implied_borrow` de
   `docs/research/options_signals.md` §3.4). El devengo usa convención ACT/360,
   la habitual del préstamo de valores en EE. UU., sobre días naturales: un fin
   de semana devenga tres días.

Todas las funciones son vectorizadas (aceptan `np.ndarray`) y **nunca devuelven
NaN**: un dato de liquidez ausente se penaliza de forma explícita y conservadora
(peor tramo de spread; participación 1.0 o `default_adv_usd`; `default_sigma_daily`),
en vez de propagar silencio o inventar un coste cero.

Referencias
-----------
- Almgren, R., Thum, C., Hauptmann, E. y Li, H. (2005). *Direct Estimation of
  Equity Market Impact*. Risk 18(7), 58-62.
- Grinold, R. y Kahn, R. (2000). *Active Portfolio Management*, 2ª ed., cap. 16
  (modelo de costes con impacto ~ raíz del tamaño).
- Gatheral, J. (2010). *No-Dynamic-Arbitrage and Market Impact*. Quantitative
  Finance 10(7), 749-759.
- Tóth, B. et al. (2011). *Anomalous Price Impact and the Critical Nature of
  Liquidity in Financial Markets*. Physical Review X 1, 021006.
- Frazzini, A., Israel, R. y Moskowitz, T. (2018). *Trading Costs*. SSRN 3229719.
- Hasbrouck, J. (2009). *Trading Costs and Returns for U.S. Equities*. Journal of
  Finance 64(3), 1445-1477.
- Corwin, S. y Schultz, P. (2012). *A Simple Way to Estimate Bid-Ask Spreads from
  Daily High and Low Prices*. Journal of Finance 67(2), 719-760.
- D'Avolio, G. (2002). *The Market for Borrowing Stock*. Journal of Financial
  Economics 66(2-3), 271-306.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

import numpy as np

from earnings_alpha.errors import ConfigError

__all__ = [  # noqa: RUF022 - orden temático, no alfabético
    "BPS",
    "SpreadTier",
    "DEFAULT_SPREAD_TIERS",
    "CostBreakdown",
    "CostModel",
]

BPS = 1e-4
"""Un punto básico en fracción (1 pb = 0.01 %)."""


@dataclass(frozen=True, slots=True)
class SpreadTier:
    """Un tramo de liquidez: medio spread aplicable a partir de un ADV mínimo.

    `min_adv_usd` es el límite inferior del tramo en dólares de volumen medio
    diario; el tramo aplica a `adv >= min_adv_usd` (hasta el tramo superior).
    """

    min_adv_usd: float
    half_spread_bps: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.min_adv_usd) or self.min_adv_usd < 0.0:
            msg = f"min_adv_usd inválido: {self.min_adv_usd!r}"
            raise ConfigError(msg)
        if not math.isfinite(self.half_spread_bps) or self.half_spread_bps < 0.0:
            msg = f"half_spread_bps inválido: {self.half_spread_bps!r}"
            raise ConfigError(msg)


DEFAULT_SPREAD_TIERS: tuple[SpreadTier, ...] = (
    SpreadTier(min_adv_usd=1_000e6, half_spread_bps=1.0),
    SpreadTier(min_adv_usd=250e6, half_spread_bps=2.0),
    SpreadTier(min_adv_usd=50e6, half_spread_bps=4.0),
    SpreadTier(min_adv_usd=10e6, half_spread_bps=8.0),
    SpreadTier(min_adv_usd=0.0, half_spread_bps=20.0),
)
"""Tramos por defecto para el S&P 500 (medio spread, pb sobre el nocional).

Calibración: las megacaps (ADV > $1B: AAPL, MSFT...) cotizan con spreads
cotizados de 1-3 pb (medio spread ≈ 1 pb); el grueso del índice, entre 3 y 8 pb
de spread; y la cola menos líquida o en condiciones degradadas, decenas de pb.
Órdenes de magnitud consistentes con los costes efectivos medidos por Frazzini,
Israel y Moskowitz (2018) y con los estimadores de Hasbrouck (2009) y
Corwin-Schultz (2012). Para universos small-cap estos tramos serían optimistas
y deben sustituirse.
"""


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    """Coste de una operación (o de un día) desglosado, en fracción del NAV.

    `spread`, `impact` y `commission` se devengan al ejecutar; `borrow` se
    devenga por mantener cortos abiertos. `total` es la suma simple: las cuatro
    componentes están en las mismas unidades (fracción del NAV del día).
    """

    spread: float = 0.0
    impact: float = 0.0
    commission: float = 0.0
    borrow: float = 0.0

    @property
    def total(self) -> float:
        """Coste total en fracción del NAV."""
        return self.spread + self.impact + self.commission + self.borrow

    def __add__(self, other: CostBreakdown) -> CostBreakdown:
        return CostBreakdown(
            spread=self.spread + other.spread,
            impact=self.impact + other.impact,
            commission=self.commission + other.commission,
            borrow=self.borrow + other.borrow,
        )


@dataclass(frozen=True, slots=True)
class CostModel:
    """Modelo de costes: spread por tramo + impacto √participación + comisión + préstamo.

    Los defaults están documentados (con fuente) en el docstring del módulo y en
    cada atributo. El modelo es inmutable; `scaled()` produce variantes
    proporcionales —útil para las curvas de sensibilidad a costes— y `zero()` un
    modelo sin costes para separar señal bruta de implementación.
    """

    spread_tiers: tuple[SpreadTier, ...] = DEFAULT_SPREAD_TIERS
    """Tramos de medio spread por ADV en dólares, ordenados de más a menos líquido."""

    impact_coefficient: float = 0.5
    """`η` de la ley de la raíz: coste = η·σ_diaria·√(participación) por unidad
    negociada. Calibrado con Frazzini-Israel-Moskowitz (2018): η ≈ 0.45 en
    grandes capitalizaciones; 0.5 como redondeo conservador. Almgren et al.
    (2005) estiman la misma forma funcional con exponente 0.6."""

    commission_bps: float = 0.5
    """Comisión en pb del nocional. ~0.3-0.5 c/acción institucionales (ITG/Virtu
    Global Cost Review) ≈ 0.3-0.5 pb sobre un precio de $100."""

    borrow_bps_annual: float = 25.0
    """Comisión anual de préstamo para cortos, en pb. D'Avolio (2002): general
    collateral ~17 pb/año de media; 25 pb es un defecto conservador para el
    S&P 500, casi todo GC."""

    borrow_overrides_bps: Mapping[str, float] = field(default_factory=dict)
    """Tasas anuales por nombre (pb) para *specials* / hard-to-borrow. Puede
    alimentarse del `implied_borrow` de opciones (options_signals.md §3.4)."""

    borrow_day_count: int = 360
    """Convención de devengo del préstamo (ACT/360, estándar del stock loan en
    EE. UU.). El devengo es sobre días naturales: el fin de semana cuenta."""

    default_sigma_daily: float = 0.02
    """Volatilidad diaria a usar cuando la estimada no está disponible. 2 %/día
    (~32 % anual) está por encima de la mediana del S&P 500: penaliza, no regala."""

    default_adv_usd: float | None = None
    """ADV a imputar cuando falta. `None` (defecto) = tratar la operación como
    participación 1.0 (un día entero de volumen) y peor tramo de spread: la
    alternativa explícitamente conservadora a inventar liquidez."""

    def __post_init__(self) -> None:
        if not self.spread_tiers:
            msg = "spread_tiers no puede estar vacío"
            raise ConfigError(msg)
        advs = [t.min_adv_usd for t in self.spread_tiers]
        if any(b >= a for a, b in zip(advs, advs[1:], strict=True)):
            msg = (
                "spread_tiers debe ir ordenado por min_adv_usd estrictamente "
                f"descendente; recibido {advs}"
            )
            raise ConfigError(msg)
        if advs[-1] != 0.0:
            msg = (
                "el último tramo debe cubrir min_adv_usd = 0 para que todo ADV "
                f"tenga tramo; el último recibido empieza en {advs[-1]}"
            )
            raise ConfigError(msg)
        for name in ("impact_coefficient", "commission_bps", "borrow_bps_annual"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                msg = f"{name} debe ser finito y >= 0; recibido {value!r}"
                raise ConfigError(msg)
        if self.borrow_day_count <= 0:
            msg = f"borrow_day_count debe ser positivo; recibido {self.borrow_day_count!r}"
            raise ConfigError(msg)
        if not math.isfinite(self.default_sigma_daily) or self.default_sigma_daily <= 0.0:
            msg = f"default_sigma_daily debe ser > 0; recibido {self.default_sigma_daily!r}"
            raise ConfigError(msg)
        if self.default_adv_usd is not None and (
            not math.isfinite(self.default_adv_usd) or self.default_adv_usd <= 0.0
        ):
            msg = f"default_adv_usd debe ser > 0 o None; recibido {self.default_adv_usd!r}"
            raise ConfigError(msg)
        for ticker, bps in self.borrow_overrides_bps.items():
            if not math.isfinite(bps) or bps < 0.0:
                msg = f"borrow_overrides_bps[{ticker!r}] inválido: {bps!r}"
                raise ConfigError(msg)

    # ------------------------------------------------------------- constructores

    @classmethod
    def zero(cls) -> CostModel:
        """Modelo sin costes (backtest bruto). Explícito, nunca el defecto."""
        return cls(
            spread_tiers=(SpreadTier(min_adv_usd=0.0, half_spread_bps=0.0),),
            impact_coefficient=0.0,
            commission_bps=0.0,
            borrow_bps_annual=0.0,
        )

    def scaled(self, factor: float) -> CostModel:
        """Variante con las cuatro componentes multiplicadas por `factor`.

        Sirve para trazar la curva rendimiento-neto vs. nivel de costes (y para
        el test de monotonía: más costes nunca pueden mejorar el neto). `factor`
        escala los *precios* de los costes, no los parámetros de liquidez.
        """
        if not math.isfinite(factor) or factor < 0.0:
            msg = f"factor de escala inválido: {factor!r}"
            raise ConfigError(msg)
        tiers = tuple(
            SpreadTier(min_adv_usd=t.min_adv_usd, half_spread_bps=t.half_spread_bps * factor)
            for t in self.spread_tiers
        )
        return replace(
            self,
            spread_tiers=tiers,
            impact_coefficient=self.impact_coefficient * factor,
            commission_bps=self.commission_bps * factor,
            borrow_bps_annual=self.borrow_bps_annual * factor,
            borrow_overrides_bps={k: v * factor for k, v in self.borrow_overrides_bps.items()},
        )

    # ------------------------------------------------------------------ spread

    def half_spread_fraction(self, adv_usd: np.ndarray | float) -> np.ndarray:
        """Medio spread (fracción del nocional) según el tramo de ADV.

        ADV ausente (NaN) o no positivo cae al **peor tramo**: no saber la
        liquidez de un nombre no puede abaratarlo.
        """
        adv = np.atleast_1d(np.asarray(adv_usd, dtype=float))
        known = np.isfinite(adv) & (adv > 0.0)
        # Peor tramo por defecto (cubre también el ADV desconocido); recorriendo
        # los tramos de menos a más líquido, la última asignación que aplica es
        # la del tramo con el suelo más alto que el ADV alcanza.
        out = np.full(adv.shape, self.spread_tiers[-1].half_spread_bps * BPS)
        for tier in reversed(self.spread_tiers):
            out = np.where(known & (adv >= tier.min_adv_usd), tier.half_spread_bps * BPS, out)
        return out

    # ------------------------------------------------------------------ impacto

    def impact_fraction(
        self,
        participation: np.ndarray | float,
        sigma_daily: np.ndarray | float,
    ) -> np.ndarray:
        """Coste de impacto por unidad negociada: ``η·σ·√(participación)``.

        `participation` = nocional de la operación / ADV en dólares, acotada
        inferiormente en 0. Sigma ausente → `default_sigma_daily`.
        """
        p = np.clip(np.atleast_1d(np.asarray(participation, dtype=float)), 0.0, None)
        p = np.where(np.isfinite(p), p, 1.0)
        sig = np.atleast_1d(np.asarray(sigma_daily, dtype=float))
        sig = np.where(np.isfinite(sig) & (sig > 0.0), sig, self.default_sigma_daily)
        return self.impact_coefficient * sig * np.sqrt(p)

    # -------------------------------------------------------------- ejecución

    def execution_costs(
        self,
        trade_weights: np.ndarray | Sequence[float],
        adv_usd: np.ndarray | Sequence[float] | None,
        sigma_daily: np.ndarray | Sequence[float] | None,
        nav_usd: float,
    ) -> CostBreakdown:
        """Coste de ejecutar un vector de operaciones, en fracción del NAV.

        Parameters
        ----------
        trade_weights:
            Operación por nombre como fracción del NAV (`Δw`; el signo se ignora).
        adv_usd:
            ADV en dólares por nombre, **calculado hacia atrás** (la media móvil
            debe terminar antes de la ejecución: `pit_and_biases.md` §12.6).
            `None` = desconocido para todos.
        sigma_daily:
            Volatilidad diaria por nombre (fracción), también hacia atrás.
        nav_usd:
            NAV en dólares; convierte pesos en nocionales para la participación.

        Notes
        -----
        - Nombres sin ADV: peor tramo de spread y participación `1.0` (o la que
          implique `default_adv_usd` si se configuró). Conservador y explícito.
        - El resultado nunca contiene NaN; una operación de tamaño 0 cuesta 0.
        """
        dw = np.abs(np.atleast_1d(np.asarray(trade_weights, dtype=float)))
        dw = np.where(np.isfinite(dw), dw, 0.0)
        n = dw.shape[0]
        if nav_usd <= 0.0 or not math.isfinite(nav_usd):
            msg = f"nav_usd debe ser positivo y finito; recibido {nav_usd!r}"
            raise ConfigError(msg)

        if adv_usd is None:
            adv = np.full(n, np.nan)
        else:
            adv = np.asarray(adv_usd, dtype=float)
            if adv.shape != dw.shape:
                msg = f"adv_usd tiene forma {adv.shape}, esperada {dw.shape}"
                raise ConfigError(msg)
        if sigma_daily is None:
            sig = np.full(n, np.nan)
        else:
            sig = np.asarray(sigma_daily, dtype=float)
            if sig.shape != dw.shape:
                msg = f"sigma_daily tiene forma {sig.shape}, esperada {dw.shape}"
                raise ConfigError(msg)

        adv_known = np.isfinite(adv) & (adv > 0.0)
        if self.default_adv_usd is not None:
            adv = np.where(adv_known, adv, self.default_adv_usd)
            adv_known = adv > 0.0

        notional = dw * nav_usd
        participation = np.where(adv_known, notional / np.where(adv_known, adv, 1.0), 1.0)
        participation = np.where(dw > 0.0, participation, 0.0)

        spread = float(np.sum(self.half_spread_fraction(adv) * dw))
        impact = float(np.sum(self.impact_fraction(participation, sig) * dw))
        commission = float(np.sum(self.commission_bps * BPS * dw))
        return CostBreakdown(spread=spread, impact=impact, commission=commission)

    # ---------------------------------------------------------------- préstamo

    def borrow_daily_rates(self, tickers: Sequence[str]) -> np.ndarray:
        """Tasa de préstamo por **día natural** (fracción) para cada nombre.

        `borrow_bps_annual / day_count` con los overrides por nombre aplicados.
        El motor la multiplica por el nocional corto y por los días naturales
        hasta la siguiente sesión (ACT/360: el viernes devenga 3 días).
        """
        base = self.borrow_bps_annual
        rates = np.array(
            [self.borrow_overrides_bps.get(t, base) for t in tickers],
            dtype=float,
        )
        return rates * BPS / float(self.borrow_day_count)

    def holding_costs(
        self,
        weights: np.ndarray | Sequence[float],
        tickers: Sequence[str],
        calendar_days: int = 1,
    ) -> CostBreakdown:
        """Coste de mantener la cartera un periodo: préstamo sobre los cortos.

        `weights` son los pesos fin de día (fracción del NAV); solo los negativos
        devengan. `calendar_days` son los días naturales hasta la siguiente
        sesión (1 entre diario y diario, 3 sobre un fin de semana).
        """
        if calendar_days < 0:
            msg = f"calendar_days debe ser >= 0; recibido {calendar_days!r}"
            raise ConfigError(msg)
        w = np.atleast_1d(np.asarray(weights, dtype=float))
        w = np.where(np.isfinite(w), w, 0.0)
        if len(tickers) != w.shape[0]:
            msg = f"tickers ({len(tickers)}) y weights ({w.shape[0]}) no cuadran"
            raise ConfigError(msg)
        short_gross = np.abs(np.minimum(w, 0.0))
        rates = self.borrow_daily_rates(tickers)
        return CostBreakdown(borrow=float(np.sum(short_gross * rates) * calendar_days))
