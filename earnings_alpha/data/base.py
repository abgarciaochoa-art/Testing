"""Núcleo de la capa de datos: proveedores, registro con fallback y cliente HTTP.

Este módulo es la puerta por la que entra **todo** dato externo al repositorio. Su
responsabilidad no es hablar con ninguna API concreta —eso lo hacen
`data.prices`, `data.edgar`, `data.estimates`, `data.options` y `data.shortinterest`—
sino imponer tres invariantes que el resto del sistema da por supuestas:

1. **Fallo explícito y accionable.** Un proveedor sin credenciales o sin red lanza
   `ProviderUnavailable` diciendo *exactamente* qué falta: qué variable de entorno
   no está definida o qué comprobación de conectividad ha fallado, proveedor por
   proveedor. Nunca se devuelve un panel vacío que más adelante se interprete como
   "ese trimestre no hubo datos" (contrato §0.3).
2. **Cortesía con las fuentes.** Todo tráfico pasa por un *token bucket* con la
   tasa del proveedor. La SEC publica un límite duro de **10 peticiones por
   segundo agregadas** sobre `www.sec.gov`, `data.sec.gov` y `efts.sec.gov`, y
   superarlo devuelve 429 y puede acarrear un bloqueo temporal de IP
   (`docs/research/data_sources.md` §4.1); `Settings.max_requests_per_second = 8.0`
   deja margen deliberado.
3. **Testabilidad sin red.** El transporte, el reloj y la fuente de aleatoriedad son
   inyectables. Los tests de este módulo —y los de cualquier adaptador que lo use—
   corren sin abrir un socket: `ScriptedTransport` simula 200/429/403/timeout y
   `ManualClock` permite *medir* el backoff y la limitación de tasa en tiempo
   simulado, de forma determinista y en microsegundos.

Sobre el modelo de errores
--------------------------
`earnings_alpha.errors` es un contrato cerrado. Este módulo añade tres excepciones
**derivadas de `EarningsAlphaError`** para hechos que no existían en él y que sería
incorrecto forzar dentro de los tipos existentes:

- `TransportError` / `TransportTimeout`: fallo *antes* de tener una respuesta HTTP
  (DNS, TLS, socket, timeout). Es una condición reintentable; convertirla de
  inmediato en `ProviderUnavailable` impediría distinguir "no hay red" de
  "el reintento número 3 tampoco funcionó".
- `HttpStatusError`: respuesta HTTP con un código que no es ni éxito ni una de las
  condiciones ya tipadas (429 → `RateLimited`, 401/403 → `ProviderUnavailable`).

En la **frontera pública** —lo que ve quien llama a `HttpClient.get(...)`— solo se
propagan errores del contrato: agotados los reintentos, un timeout o un 5xx
persistente se traducen a `ProviderUnavailable`, y un 429 persistente a
`RateLimited`. `HttpStatusError` es lo único nuevo que escapa, y solo para códigos
(404, 400, 422…) cuyo significado depende del adaptador.

Referencias
-----------
- SEC EDGAR, *Accessing EDGAR Data* / *Fair Access*: `User-Agent` identificable con
  correo de contacto y ≤10 req/s agregadas.
- AWS Architecture Blog, *Exponential Backoff and Jitter* (Brooker, 2015): el
  backoff exponencial **sin** jitter sincroniza a los clientes y reproduce la
  tormenta que pretende evitar; por eso el jitter es "full" por defecto.
- RFC 9110 §10.2.3 (`Retry-After`) y §15.5.29 (429): el servidor puede indicar
  cuánto esperar, en segundos o como fecha HTTP; se respeta siempre que sea
  razonable.
"""

from __future__ import annotations

import json
import logging
import random
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from enum import StrEnum
from typing import Any, Protocol, TypeVar, runtime_checkable

from earnings_alpha.config import PROVIDER_ENV_KEYS, Settings, get_settings
from earnings_alpha.errors import (
    ConfigError,
    DataQualityError,
    EarningsAlphaError,
    ProviderUnavailable,
    RateLimited,
)

__all__ = [  # noqa: RUF022 - orden temático, no alfabético
    # relojes
    "Clock",
    "SystemClock",
    "ManualClock",
    # limitación de tasa
    "RateLimitPolicy",
    "TokenBucket",
    "PROVIDER_RATE_LIMITS",
    "rate_limit_for",
    # transporte
    "HttpRequest",
    "HttpResponse",
    "Transport",
    "RequestsTransport",
    "ScriptedTransport",
    "TransportError",
    "TransportTimeout",
    "HttpStatusError",
    # cliente
    "RetryPolicy",
    "HttpStats",
    "HttpClient",
    "parse_retry_after",
    # proveedores
    "DataKind",
    "KNOWN_KINDS",
    "Provider",
    "ProviderStatus",
    "BaseProvider",
    "StaticProvider",
    "ProviderAttempt",
    "ProviderRegistry",
    "get_registry",
    "set_registry",
]

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ===========================================================================
# 1. Reloj inyectable
# ===========================================================================


@runtime_checkable
class Clock(Protocol):
    """Fuente de tiempo y de espera.

    Existe para que la limitación de tasa y el backoff sean **medibles**: con
    `ManualClock` un test comprueba que se esperaron 2.0 s sin esperar 2.0 s.
    """

    def time(self) -> float:
        """Segundos monótonos desde un origen arbitrario."""

    def sleep(self, seconds: float) -> None:
        """Bloquea `seconds` segundos (o los simula)."""


class SystemClock:
    """Reloj real. Usa `time.monotonic`, inmune a saltos del reloj de pared."""

    __slots__ = ()

    def time(self) -> float:
        import time as _time

        return _time.monotonic()

    def sleep(self, seconds: float) -> None:
        import time as _time

        if seconds > 0:
            _time.sleep(seconds)

    def __repr__(self) -> str:  # pragma: no cover - cosmético
        return "SystemClock()"


class ManualClock:
    """Reloj simulado: `sleep` avanza el tiempo en vez de bloquear.

    Registra cada espera en `sleeps`, lo que permite afirmar sobre la *secuencia*
    de backoff y no solo sobre su total. Es seguro entre hilos.
    """

    __slots__ = ("_lock", "now", "sleeps")

    def __init__(self, start: float = 0.0) -> None:
        self.now = float(start)
        self.sleeps: list[float] = []
        self._lock = threading.Lock()

    def time(self) -> float:
        with self._lock:
            return self.now

    def sleep(self, seconds: float) -> None:
        if seconds <= 0:
            return
        with self._lock:
            self.now += float(seconds)
            self.sleeps.append(float(seconds))

    @property
    def total_slept(self) -> float:
        """Suma de todas las esperas simuladas."""
        return float(sum(self.sleeps))

    def __repr__(self) -> str:  # pragma: no cover - cosmético
        return f"ManualClock(now={self.now:.6f}, n_sleeps={len(self.sleeps)})"


# ===========================================================================
# 2. Limitación de tasa (token bucket)
# ===========================================================================


@dataclass(frozen=True, slots=True)
class RateLimitPolicy:
    """Política de tasa de un proveedor.

    `requests_per_second` es el caudal sostenido y `burst` la capacidad del cubo:
    cuántas peticiones seguidas se admiten tras un periodo de inactividad. Un
    `burst` de 1 impone espaciado uniforme —lo más cortés—; un `burst` alto agota
    la ventana del proveedor de golpe.
    """

    requests_per_second: float = 8.0
    burst: int = 1
    note: str = ""

    def __post_init__(self) -> None:
        if self.requests_per_second <= 0:
            msg = f"requests_per_second debe ser > 0; recibido {self.requests_per_second}"
            raise ConfigError(msg)
        if self.burst < 1:
            msg = f"burst debe ser >= 1; recibido {self.burst}"
            raise ConfigError(msg)


class TokenBucket:
    """Cubo de fichas con reserva anticipada, seguro entre hilos.

    El cubo se rellena a `rate` fichas por segundo hasta `capacity`. `acquire`
    permite que el saldo quede **negativo**: en ese caso devuelve —y opcionalmente
    espera— el tiempo necesario para que la deuda se amortice. Esa reserva es lo
    que da equidad (FIFO aproximado) cuando varios hilos compiten: cada uno se
    lleva su hueco en la cola en vez de despertar todos a la vez sobre la misma
    ficha, que es el patrón que en la práctica dispara los 429 de la SEC.

    Ejemplo con `rate=10`, `capacity=1`: las peticiones salen en t = 0.0, 0.1,
    0.2… y **ninguna ventana de un segundo contiene más de 10**.
    """

    __slots__ = ("_clock", "_last", "_lock", "_tokens", "capacity", "rate")

    def __init__(
        self,
        rate: float,
        capacity: float | None = None,
        *,
        clock: Clock | None = None,
    ) -> None:
        if rate <= 0:
            msg = f"la tasa debe ser > 0; recibido {rate}"
            raise ConfigError(msg)
        self.rate = float(rate)
        self.capacity = float(capacity if capacity is not None else max(1.0, rate))
        if self.capacity <= 0:
            msg = f"la capacidad debe ser > 0; recibida {capacity}"
            raise ConfigError(msg)
        self._clock: Clock = clock or SystemClock()
        self._tokens = self.capacity
        self._last = self._clock.time()
        self._lock = threading.Lock()

    @classmethod
    def from_policy(cls, policy: RateLimitPolicy, *, clock: Clock | None = None) -> TokenBucket:
        return cls(policy.requests_per_second, float(policy.burst), clock=clock)

    @property
    def tokens(self) -> float:
        """Fichas disponibles ahora mismo (puede ser negativo si hay deuda)."""
        with self._lock:
            return self._refill_locked()

    def _refill_locked(self) -> float:
        now = self._clock.time()
        elapsed = now - self._last
        if elapsed > 0:
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
            self._last = now
        return self._tokens

    def reserve(self, tokens: float = 1.0) -> float:
        """Reserva `tokens` fichas y devuelve los segundos que hay que esperar.

        No duerme: quien llama decide. Separarlo de `acquire` permite reservar
        dentro del cerrojo y esperar fuera, que es lo que evita serializar a todos
        los hilos en el cerrojo.
        """
        if tokens <= 0:
            return 0.0
        with self._lock:
            available = self._refill_locked()
            self._tokens = available - tokens
            if self._tokens >= 0:
                return 0.0
            return -self._tokens / self.rate

    def acquire(self, tokens: float = 1.0) -> float:
        """Reserva y espera si hace falta. Devuelve los segundos esperados."""
        wait = self.reserve(tokens)
        if wait > 0:
            self._clock.sleep(wait)
        return wait

    def __repr__(self) -> str:  # pragma: no cover - cosmético
        return f"TokenBucket(rate={self.rate}, capacity={self.capacity})"


# Tasas por proveedor. Cifras y advertencias de `docs/research/data_sources.md`
# (§4.1 SEC, §3.x precios, §5.x estimaciones). Cuando el proveedor no documenta
# límite se aplica un valor conservador y se anota como tal: el coste de ser
# demasiado cortés es tiempo; el de no serlo, un bloqueo de IP.
PROVIDER_RATE_LIMITS: dict[str, RateLimitPolicy] = {
    # Límite duro publicado: 10 req/s agregadas en www./data./efts.sec.gov.
    "sec": RateLimitPolicy(8.0, burst=8, note="SEC publica 10 req/s; se deja margen"),
    "edgar": RateLimitPolicy(8.0, burst=8, note="alias de 'sec'; comparte cuota de IP"),
    # Yahoo no documenta límite y devuelve 429 sistemáticos en bucles largos.
    "yfinance": RateLimitPolicy(1.0, burst=1, note="no documentado; ≤1 símbolo/s"),
    "stooq": RateLimitPolicy(1.0, burst=1, note="no documentado; conservador"),
    # Polygon: plan gratuito 5 req/min. Los planes de pago no tienen tope.
    "polygon": RateLimitPolicy(5.0 / 60.0, burst=5, note="plan gratuito: 5 req/min"),
    # FMP: Starter 300 req/min. Se usa el escalón más bajo de pago.
    "fmp": RateLimitPolicy(5.0, burst=5, note="Starter 300 req/min"),
    # EODHD de pago: 1 000 req/min; se deja la mitad.
    "eodhd": RateLimitPolicy(8.0, burst=8, note="planes de pago: 1 000 req/min"),
    # Finnhub gratuito: 60 req/min.
    "finnhub": RateLimitPolicy(1.0, burst=1, note="plan gratuito: 60 req/min"),
    # Alpha Vantage gratuito: 5 req/min (y 25 req/día).
    "alphavantage": RateLimitPolicy(5.0 / 60.0, burst=5, note="plan gratuito: 5 req/min"),
    "tiingo": RateLimitPolicy(2.0, burst=4, note="~1 000 req/día gratis"),
    "nasdaq_data_link": RateLimitPolicy(5.0, burst=5, note="50 000 llamadas/día"),
    "alpaca": RateLimitPolicy(3.0, burst=6, note="~200 req/min en el plan básico"),
    "tradier": RateLimitPolicy(2.0, burst=4, note="conservador"),
    "orats": RateLimitPolicy(2.0, burst=4, note="cuota mensual, no por minuto"),
    "finra": RateLimitPolicy(2.0, burst=2, note="acceso anónimo: cuota reducida"),
    "synthetic": RateLimitPolicy(1e6, burst=1000, note="sin red: no limita"),
}


def rate_limit_for(provider: str, settings: Settings | None = None) -> RateLimitPolicy:
    """Política de tasa efectiva de un proveedor.

    Precedencia: variable de entorno `EARNINGS_ALPHA_RPS_<PROVEEDOR>` (para poder
    subir el caudal cuando se contrata un plan de pago sin tocar código) → tabla
    `PROVIDER_RATE_LIMITS` → techo global `Settings.max_requests_per_second`.
    """
    cfg = settings or get_settings()
    key = f"EARNINGS_ALPHA_RPS_{provider.upper().replace('-', '_')}"
    raw = cfg.env(key)
    if raw:
        try:
            rps = float(raw)
        except ValueError as exc:
            msg = f"{key}={raw!r} no es un número de peticiones por segundo"
            raise ConfigError(msg) from exc
        base = PROVIDER_RATE_LIMITS.get(provider)
        burst = base.burst if base else max(1, int(rps))
        return RateLimitPolicy(rps, burst=burst, note=f"sobrescrito por {key}")
    policy = PROVIDER_RATE_LIMITS.get(provider)
    if policy is not None:
        return policy
    return RateLimitPolicy(
        cfg.max_requests_per_second,
        burst=max(1, int(cfg.max_requests_per_second)),
        note="techo global de Settings",
    )


# ===========================================================================
# 3. Transporte
# ===========================================================================


class TransportError(EarningsAlphaError):
    """Fallo de red antes de obtener una respuesta HTTP (DNS, TLS, socket)."""

    def __init__(self, provider: str, reason: str) -> None:
        self.provider = provider
        self.reason = reason
        super().__init__(f"{provider}: fallo de transporte: {reason}")


class TransportTimeout(TransportError):  # noqa: N818 - nomenclatura de errors.py
    """La petición excedió el tiempo máximo."""


class HttpStatusError(EarningsAlphaError):
    """Respuesta HTTP con un código que no es éxito ni una condición ya tipada.

    429 se traduce a `RateLimited` y 401/403 a `ProviderUnavailable`; este error
    cubre el resto (404, 400, 422…), cuyo significado solo conoce el adaptador que
    hizo la llamada: un 404 de EDGAR puede ser "esta empresa no tiene ese
    concepto", que es un dato legítimo, no un fallo.
    """

    def __init__(self, provider: str, status_code: int, url: str, body: str = "") -> None:
        self.provider = provider
        self.status_code = status_code
        self.url = url
        self.body = body
        snippet = body[:200].replace("\n", " ") if body else ""
        detail = f"{provider}: HTTP {status_code} en {url}"
        if snippet:
            detail += f" — {snippet}"
        super().__init__(detail)


@dataclass(frozen=True, slots=True)
class HttpRequest:
    """Petición tal y como llega al transporte (útil para grabar y auditar)."""

    method: str
    url: str
    params: Mapping[str, Any] | None = None
    headers: Mapping[str, str] | None = None
    timeout: float = 30.0
    body: bytes | None = None
    at: float = 0.0
    """Instante del reloj inyectado en que se emitió; permite medir la tasa real."""


@dataclass(frozen=True, slots=True)
class HttpResponse:
    """Respuesta HTTP mínima, desacoplada de `requests`."""

    status_code: int
    content: bytes = b""
    headers: Mapping[str, str] = field(default_factory=dict)
    url: str = ""
    encoding: str = "utf-8"

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    @property
    def text(self) -> str:
        return self.content.decode(self.encoding or "utf-8", errors="replace")

    def header(self, name: str, default: str | None = None) -> str | None:
        """Búsqueda de cabecera insensible a mayúsculas (RFC 9110 §5.1)."""
        lowered = name.lower()
        for k, v in self.headers.items():
            if k.lower() == lowered:
                return v
        return default

    def json(self) -> Any:
        """Cuerpo como JSON. Un cuerpo no-JSON es un `DataQualityError`."""
        try:
            return json.loads(self.text)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            msg = f"la respuesta de {self.url or '<sin url>'} no es JSON válido: {exc}"
            raise DataQualityError(msg) from exc


@runtime_checkable
class Transport(Protocol):
    """Capa que ejecuta una petición. Inyectable para probar sin red."""

    def send(self, request: HttpRequest) -> HttpResponse:
        """Ejecuta la petición o lanza `TransportError`/`TransportTimeout`."""


class RequestsTransport:
    """Transporte real sobre `requests.Session` (conexiones reutilizadas).

    Traduce las excepciones de `requests` al vocabulario del repo; nunca deja
    escapar un tipo de la librería, para que los adaptadores no dependan de ella.
    """

    def __init__(self, provider: str = "http", *, verify: bool | str = True) -> None:
        self.provider = provider
        self.verify = verify
        self._session: Any | None = None
        self._requests: Any | None = None

    def _get_session(self) -> tuple[Any, Any]:
        if self._session is None:
            try:
                import requests
            except ImportError as exc:  # pragma: no cover - dependencia dura
                raise ProviderUnavailable(
                    self.provider, "falta la dependencia 'requests'"
                ) from exc
            self._requests = requests
            self._session = requests.Session()
        return self._session, self._requests

    def send(self, request: HttpRequest) -> HttpResponse:
        session, requests = self._get_session()
        try:
            resp = session.request(
                request.method,
                request.url,
                params=dict(request.params) if request.params else None,
                headers=dict(request.headers) if request.headers else None,
                data=request.body,
                timeout=request.timeout,
                verify=self.verify,
            )
        except requests.Timeout as exc:
            raise TransportTimeout(self.provider, f"timeout tras {request.timeout}s") from exc
        except requests.TooManyRedirects as exc:
            raise TransportError(self.provider, f"demasiadas redirecciones: {exc}") from exc
        except requests.RequestException as exc:
            raise TransportError(self.provider, f"{type(exc).__name__}: {exc}") from exc
        return HttpResponse(
            status_code=int(resp.status_code),
            content=resp.content,
            headers=dict(resp.headers),
            url=str(resp.url),
            encoding=resp.encoding or "utf-8",
        )

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None


ScriptItem = HttpResponse | BaseException | Callable[[HttpRequest], HttpResponse]


class ScriptedTransport:
    """Transporte falso guionizado. **Nunca abre un socket.**

    Cada elemento de `script` se consume en orden y puede ser:

    - un `HttpResponse` (se devuelve tal cual),
    - una excepción ya instanciada (se lanza: sirve para simular timeouts),
    - un invocable `(HttpRequest) -> HttpResponse` (respuesta dependiente de la
      petición, p. ej. paginación).

    Con `repeat_last=True` el último elemento se repite indefinidamente, que es lo
    cómodo para probar "siempre 429". Todas las peticiones quedan en `calls` con
    el instante del reloj inyectado, de modo que un test puede **medir la tasa
    efectiva** en tiempo simulado.
    """

    def __init__(
        self,
        script: Sequence[ScriptItem] | ScriptItem,
        *,
        repeat_last: bool = False,
        clock: Clock | None = None,
    ) -> None:
        items: Sequence[ScriptItem]
        if isinstance(script, HttpResponse | BaseException) or callable(script):
            items = [script]  # type: ignore[list-item]
        else:
            items = list(script)
        if not items:
            msg = "ScriptedTransport necesita al menos una respuesta guionizada"
            raise ConfigError(msg)
        self.script: list[ScriptItem] = list(items)
        self.repeat_last = repeat_last
        self.clock: Clock = clock or SystemClock()
        self.calls: list[HttpRequest] = []
        self._i = 0
        self._lock = threading.Lock()

    @property
    def n_calls(self) -> int:
        return len(self.calls)

    def send(self, request: HttpRequest) -> HttpResponse:
        with self._lock:
            self.calls.append(
                HttpRequest(
                    method=request.method,
                    url=request.url,
                    params=dict(request.params) if request.params else None,
                    headers=dict(request.headers) if request.headers else None,
                    timeout=request.timeout,
                    body=request.body,
                    at=self.clock.time(),
                )
            )
            if self._i >= len(self.script):
                if not self.repeat_last:
                    msg = (
                        f"guion agotado: se han pedido {len(self.calls)} respuestas y el "
                        f"guion tiene {len(self.script)}"
                    )
                    raise ConfigError(msg)
                item = self.script[-1]
            else:
                item = self.script[self._i]
                self._i += 1
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            return item(request)
        return item

    def close(self) -> None:  # pragma: no cover - simetría con RequestsTransport
        return None


# ===========================================================================
# 4. Reintentos
# ===========================================================================


def parse_retry_after(value: str | None, *, now: datetime | None = None) -> float | None:
    """Interpreta la cabecera `Retry-After` (RFC 9110 §10.2.3).

    Admite las dos formas del estándar: número de segundos, o fecha HTTP. Devuelve
    `None` si la cabecera falta o es ininteligible, para que quien llama pueda caer
    en su propio backoff.
    """
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    reference = now or datetime.now(UTC)
    return max(0.0, (when - reference).total_seconds())


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Política de reintentos con backoff exponencial y jitter.

    El jitter no es un adorno: con backoff puramente exponencial, N clientes que
    reciben el mismo 429 reintentan en el mismo instante y reproducen la avalancha
    (Brooker, *Exponential Backoff and Jitter*, AWS 2015). Modos:

    - ``"full"``: espera ~ U(0, d). Máxima dispersión, el recomendado.
    - ``"equal"``: espera ~ d/2 + U(0, d/2). Garantiza un mínimo de espera.
    - ``"none"``: espera d exacta. Solo para tests que quieran comparar contra la
      fórmula cerrada.
    """

    max_retries: int = 4
    backoff_base_s: float = 0.5
    backoff_factor: float = 2.0
    backoff_max_s: float = 60.0
    jitter: str = "full"
    retry_statuses: frozenset[int] = frozenset({408, 425, 429, 500, 502, 503, 504})
    respect_retry_after: bool = True
    max_retry_after_s: float = 120.0
    """Tope al `Retry-After` que se obedece. Un servidor puede pedir 3 600 s; en un
    backtest interactivo eso equivale a colgarse, así que por encima del tope se
    considera al proveedor no disponible y se deja decidir al registro."""

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            msg = f"max_retries debe ser >= 0; recibido {self.max_retries}"
            raise ConfigError(msg)
        if self.backoff_base_s < 0:
            msg = f"backoff_base_s debe ser >= 0; recibido {self.backoff_base_s}"
            raise ConfigError(msg)
        if self.backoff_factor < 1:
            msg = f"backoff_factor debe ser >= 1; recibido {self.backoff_factor}"
            raise ConfigError(msg)
        if self.jitter not in {"full", "equal", "none"}:
            msg = f"jitter debe ser 'full', 'equal' o 'none'; recibido {self.jitter!r}"
            raise ConfigError(msg)

    def deterministic_delay(self, attempt: int) -> float:
        """Retardo **sin** jitter del intento `attempt` (0 = primer reintento)."""
        raw = self.backoff_base_s * (self.backoff_factor**attempt)
        return float(min(self.backoff_max_s, raw))

    def delay_for(self, attempt: int, rng: random.Random) -> float:
        """Retardo efectivo, con jitter, del intento `attempt`."""
        base = self.deterministic_delay(attempt)
        if base <= 0 or self.jitter == "none":
            return base
        if self.jitter == "full":
            return rng.uniform(0.0, base)
        return base / 2.0 + rng.uniform(0.0, base / 2.0)

    def should_retry_status(self, status: int) -> bool:
        return status in self.retry_statuses


# ===========================================================================
# 5. Cliente HTTP
# ===========================================================================


@dataclass(slots=True)
class HttpStats:
    """Contadores de un `HttpClient`. Sirven para auditar el coste de una carga."""

    requests: int = 0
    """Peticiones enviadas al transporte, reintentos incluidos."""
    successes: int = 0
    retries: int = 0
    rate_limited: int = 0
    transport_errors: int = 0
    slept_s: float = 0.0
    """Segundos esperados por limitación de tasa **más** backoff."""
    throttled_s: float = 0.0
    """Solo lo esperado por el token bucket."""
    bytes_in: int = 0
    by_status: dict[int, int] = field(default_factory=dict)

    def record_status(self, status: int) -> None:
        self.by_status[status] = self.by_status.get(status, 0) + 1


DEFAULT_USER_AGENT = (
    "earnings-alpha/0.1 (investigación académica; contacto vía repositorio del proyecto)"
)


class HttpClient:
    """Cliente HTTP compartido: tasa, reintentos y traducción de errores.

    Todo adaptador de red del repositorio debe usarlo en vez de llamar a
    `requests` directamente, por tres razones: la cuota de la SEC es **por IP** y
    solo un cubo de fichas común la respeta; el backoff correcto es delicado
    (jitter, `Retry-After`, tope); y el mapa de códigos HTTP a errores del repo
    tiene que ser único.

    Traducción de códigos en la frontera pública:

    ==========  ======================================================
    Código      Resultado
    ==========  ======================================================
    2xx         `HttpResponse`
    429         reintento; agotados, `RateLimited(provider, retry_after)`
    401 / 403   `ProviderUnavailable` **sin reintentar** (credencial
                inválida o `User-Agent` rechazado: reintentar no arregla
                nada y quema cuota)
    408/425/5xx reintento; agotados, `ProviderUnavailable`
    timeout     reintento; agotados, `ProviderUnavailable`
    resto       `HttpStatusError` (salvo que esté en `allow_status`)
    ==========  ======================================================
    """

    def __init__(
        self,
        provider: str = "http",
        *,
        transport: Transport | None = None,
        rate: RateLimitPolicy | None = None,
        retry: RetryPolicy | None = None,
        user_agent: str | None = None,
        timeout: float | None = None,
        default_headers: Mapping[str, str] | None = None,
        clock: Clock | None = None,
        seed: int | None = None,
        settings: Settings | None = None,
        bucket: TokenBucket | None = None,
    ) -> None:
        cfg = settings or get_settings()
        self.provider = provider
        self.settings = cfg
        self.clock: Clock = clock or SystemClock()
        self.rate = rate or rate_limit_for(provider, cfg)
        self.retry = retry or RetryPolicy(max_retries=cfg.max_retries)
        self.timeout = float(timeout if timeout is not None else cfg.request_timeout_s)
        self.user_agent = user_agent or DEFAULT_USER_AGENT
        self.transport: Transport = transport or RequestsTransport(provider)
        self.bucket = bucket or TokenBucket.from_policy(self.rate, clock=self.clock)
        self.stats = HttpStats()
        self._rng = random.Random(cfg.seed if seed is None else seed)
        self._headers: dict[str, str] = {
            "User-Agent": self.user_agent,
            "Accept-Encoding": "gzip, deflate",
        }
        if default_headers:
            self._headers.update({str(k): str(v) for k, v in default_headers.items()})

    # -- construcción --------------------------------------------------------

    @classmethod
    def for_provider(
        cls,
        provider: str,
        *,
        settings: Settings | None = None,
        require_credentials: bool = True,
        transport: Transport | None = None,
        clock: Clock | None = None,
        user_agent: str | None = None,
        **kwargs: Any,
    ) -> HttpClient:
        """Cliente con la política del proveedor y sus credenciales verificadas.

        Con `require_credentials=True` (por defecto) comprueba las variables de
        entorno declaradas en `config.PROVIDER_ENV_KEYS` y lanza
        `ProviderUnavailable` **enumerándolas** si falta alguna: es preferible
        fallar aquí, con un mensaje que dice qué exportar, que recibir un 403
        opaco a mitad de una carga de 500 tickers.

        Para la SEC, además, el `User-Agent` sale de `SEC_USER_AGENT`: la fuente
        exige nombre y correo de contacto identificables y rechaza los genéricos.
        """
        cfg = settings or get_settings()
        if require_credentials and provider in PROVIDER_ENV_KEYS:
            missing = cfg.missing_credentials_for(provider)
            if missing:
                raise ProviderUnavailable(
                    provider,
                    "faltan credenciales para construir el cliente HTTP",
                    missing_env=missing,
                )
        ua = user_agent
        if ua is None and provider in {"sec", "edgar"}:
            ua = cfg.env("SEC_USER_AGENT") or None
        return cls(
            provider,
            transport=transport,
            rate=rate_limit_for(provider, cfg),
            user_agent=ua,
            settings=cfg,
            clock=clock,
            **kwargs,
        )

    # -- API ------------------------------------------------------------------

    def get(self, url: str, **kwargs: Any) -> HttpResponse:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> HttpResponse:
        return self.request("POST", url, **kwargs)

    def get_text(self, url: str, **kwargs: Any) -> str:
        return self.get(url, **kwargs).text

    def get_bytes(self, url: str, **kwargs: Any) -> bytes:
        return self.get(url, **kwargs).content

    def get_json(self, url: str, **kwargs: Any) -> Any:
        headers = dict(kwargs.pop("headers", None) or {})
        headers.setdefault("Accept", "application/json")
        return self.get(url, headers=headers, **kwargs).json()

    def request(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        body: bytes | None = None,
        timeout: float | None = None,
        allow_status: Iterable[int] = (),
        cost: float = 1.0,
    ) -> HttpResponse:
        """Ejecuta una petición con tasa, reintentos y traducción de errores.

        Parámetros
        ----------
        allow_status:
            Códigos no-2xx que se devuelven en vez de lanzar. Pensado para el 404
            de EDGAR ("esta empresa no publica ese concepto"), que es información
            legítima y no un fallo.
        cost:
            Fichas del cubo que consume la petición. Algunos proveedores ponderan
            los endpoints (EODHD cobra varios créditos por llamada); así se modela.
        """
        allowed = frozenset(allow_status)
        merged: dict[str, str] = dict(self._headers)
        if headers:
            merged.update({str(k): str(v) for k, v in headers.items()})
        req_timeout = float(timeout if timeout is not None else self.timeout)
        last_error: BaseException | None = None
        last_retry_after: float | None = None

        for attempt in range(self.retry.max_retries + 1):
            waited = self.bucket.acquire(cost)
            if waited:
                self.stats.throttled_s += waited
                self.stats.slept_s += waited

            request = HttpRequest(
                method=method.upper(),
                url=url,
                params=params,
                headers=merged,
                timeout=req_timeout,
                body=body,
                at=self.clock.time(),
            )
            self.stats.requests += 1
            try:
                resp = self.transport.send(request)
            except TransportError as exc:
                self.stats.transport_errors += 1
                last_error = exc
                if attempt < self.retry.max_retries:
                    self._sleep_backoff(attempt, reason=str(exc))
                    continue
                break

            self.stats.record_status(resp.status_code)
            self.stats.bytes_in += len(resp.content)

            if resp.ok or resp.status_code in allowed:
                self.stats.successes += 1
                return resp

            if resp.status_code in (401, 403):
                # Reintentar una credencial inválida no la vuelve válida, y con la
                # SEC un 403 repetido acelera el bloqueo de IP.
                missing = self._missing_env()
                reason = (
                    f"HTTP {resp.status_code} en {url}: "
                    + (
                        "credenciales rechazadas o ausentes"
                        if resp.status_code == 401
                        else "acceso denegado (credencial, User-Agent o egress bloqueado)"
                    )
                )
                raise ProviderUnavailable(self.provider, reason, missing_env=missing)

            retry_after = (
                parse_retry_after(resp.header("Retry-After"))
                if self.retry.respect_retry_after
                else None
            )
            if resp.status_code == 429:
                self.stats.rate_limited += 1
                last_retry_after = retry_after
            if self.retry.should_retry_status(resp.status_code):
                last_error = HttpStatusError(
                    self.provider, resp.status_code, url, resp.text[:200]
                )
                if attempt < self.retry.max_retries:
                    if retry_after is not None and retry_after > self.retry.max_retry_after_s:
                        # El servidor pide una espera desmedida: no se bloquea el
                        # proceso, se declara al proveedor no disponible.
                        break
                    self._sleep_backoff(
                        attempt,
                        retry_after=retry_after,
                        reason=f"HTTP {resp.status_code}",
                    )
                    continue
                break

            raise HttpStatusError(self.provider, resp.status_code, url, resp.text[:400])

        # Reintentos agotados: se traduce al vocabulario del contrato.
        attempts = self.retry.max_retries + 1
        if isinstance(last_error, HttpStatusError) and last_error.status_code == 429:
            raise RateLimited(self.provider, last_retry_after)
        if isinstance(last_error, TransportTimeout):
            raise ProviderUnavailable(
                self.provider,
                f"sin respuesta tras {attempts} intentos ({last_error})",
                missing_env=self._missing_env(),
            ) from last_error
        if isinstance(last_error, TransportError):
            raise ProviderUnavailable(
                self.provider,
                f"sin conectividad tras {attempts} intentos ({last_error})",
                missing_env=self._missing_env(),
            ) from last_error
        if isinstance(last_error, HttpStatusError):
            raise ProviderUnavailable(
                self.provider,
                f"HTTP {last_error.status_code} persistente tras {attempts} intentos en {url}",
            ) from last_error
        msg = f"la petición a {url} no produjo respuesta ni error"  # pragma: no cover
        raise ProviderUnavailable(self.provider, msg)  # pragma: no cover

    # -- interno --------------------------------------------------------------

    def _missing_env(self) -> list[str]:
        if self.provider not in PROVIDER_ENV_KEYS:
            return []
        return self.settings.missing_credentials_for(self.provider)

    def _sleep_backoff(
        self, attempt: int, *, retry_after: float | None = None, reason: str = ""
    ) -> None:
        delay = self.retry.delay_for(attempt, self._rng)
        if retry_after is not None:
            # `Retry-After` es una instrucción del servidor: manda sobre el
            # backoff local, pero nunca lo acorta por debajo de él.
            delay = min(max(delay, retry_after), self.retry.max_retry_after_s)
        self.stats.retries += 1
        logger.debug(
            "reintento %d/%d de %s en %.3fs (%s)",
            attempt + 1,
            self.retry.max_retries,
            self.provider,
            delay,
            reason,
        )
        if delay > 0:
            self.stats.slept_s += delay
            self.clock.sleep(delay)

    def close(self) -> None:
        close = getattr(self.transport, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> HttpClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - cosmético
        return (
            f"HttpClient(provider={self.provider!r}, "
            f"rps={self.rate.requests_per_second:g}, burst={self.rate.burst})"
        )


# ===========================================================================
# 6. Proveedores
# ===========================================================================


class DataKind(StrEnum):
    """Tipos de dato que un proveedor puede servir.

    Es un `StrEnum`: donde el contrato pide `kind: str` se puede pasar
    indistintamente `"prices"` o `DataKind.PRICES`. La lista no es cerrada —el
    registro acepta cualquier cadena— pero usar estos valores evita el error
    clásico de registrar `"price"` y resolver `"prices"`.
    """

    PRICES = "prices"
    """OHLCV diario o intradía, con factores de ajuste si el proveedor los da."""
    CORPORATE_ACTIONS = "corporate_actions"
    """Splits y dividendos, necesarios para ajustar precios point-in-time."""
    FUNDAMENTALS = "fundamentals"
    """Hechos contables con `available_at` (XBRL de EDGAR, `filingDate` de FMP)."""
    ESTIMATES = "estimates"
    """Consenso de analistas y sus revisiones históricas."""
    EARNINGS_CALENDAR = "earnings_calendar"
    """Fechas de anuncio, confirmadas o estimadas, con etiqueta BMO/AMC."""
    OPTIONS = "options"
    """Cadenas: strikes, IV, open interest, volumen."""
    SHORT_INTEREST = "short_interest"
    """Short interest quincenal (FINRA/exchanges)."""
    OFF_EXCHANGE = "off_exchange"
    """Volumen off-exchange y ATS (FINRA), proxy de flujo institucional."""
    INSIDER = "insider"
    """Form 3/4/5 ya publicados."""
    FILINGS = "filings"
    """Índices y documentos de EDGAR (8-K 2.02, 10-Q, 10-K)."""
    UNIVERSE = "universe"
    """Composición del índice."""
    NEWS = "news"
    """Titulares fechados."""


KNOWN_KINDS: frozenset[str] = frozenset(k.value for k in DataKind)


@dataclass(frozen=True, slots=True)
class ProviderStatus:
    """Diagnóstico de disponibilidad de un proveedor.

    `reason` y `missing_env` son lo que convierte un `ProviderUnavailable` en algo
    accionable: el usuario debe leer qué exportar o qué host desbloquear, no un
    "no disponible" a secas.
    """

    provider: str
    available: bool
    missing_env: tuple[str, ...] = ()
    reason: str = ""
    priority: int | None = None

    def describe(self) -> str:
        if self.available:
            head = "disponible"
        elif self.missing_env:
            head = f"faltan variables de entorno: {', '.join(self.missing_env)}"
        else:
            head = self.reason or "no disponible"
        if self.reason and self.missing_env and not self.available:
            head = f"{head} ({self.reason})"
        prio = "" if self.priority is None else f" [prioridad {self.priority}]"
        return f"{self.provider}{prio}: {head}"


@runtime_checkable
class Provider(Protocol):
    """Contrato mínimo de un proveedor de datos (contrato §3.3).

    `kinds` declara qué tipos sirve; el registro lo usa para rechazar registros
    incoherentes en el momento del registro y no en mitad de un backtest.
    """

    name: str
    kinds: tuple[str, ...]

    def available(self) -> bool:
        """True si el proveedor puede servir ahora: credenciales **y** red."""


class BaseProvider:
    """Base cómoda para proveedores reales: credenciales, HTTP y diagnóstico.

    Implementa `available()` en dos pasos, del más barato al más caro:

    1. **Credenciales**: variables de entorno declaradas en `PROVIDER_ENV_KEYS`.
       Es una comprobación local, instantánea.
    2. **Conectividad** (opcional): si la subclase define `probe_url`, una petición
       ligera cuyo resultado se memoriza `probe_ttl_s` segundos. Sin memoización,
       resolver un proveedor dentro de un bucle por ticker dispararía una petición
       de sondeo por iteración y agotaría la cuota antes de empezar.
    """

    name: str = "base"
    kinds: tuple[str, ...] = ()
    probe_url: str | None = None
    probe_ttl_s: float = 300.0

    def __init__(
        self,
        *,
        name: str | None = None,
        kinds: Sequence[str] | None = None,
        settings: Settings | None = None,
        http: HttpClient | None = None,
        clock: Clock | None = None,
    ) -> None:
        if name is not None:
            self.name = name
        if kinds is not None:
            self.kinds = tuple(str(k) for k in kinds)
        self.settings = settings or get_settings()
        self.clock: Clock = clock or SystemClock()
        self._http = http
        self._probe: tuple[float, bool, str] | None = None

    # -- credenciales ---------------------------------------------------------

    def required_env(self) -> list[str]:
        """Variables de entorno que el proveedor necesita."""
        return list(PROVIDER_ENV_KEYS.get(self.name, []))

    def missing_env(self) -> list[str]:
        return [k for k in self.required_env() if not self.settings.env(k)]

    # -- disponibilidad -------------------------------------------------------

    def check_connectivity(self) -> tuple[bool, str]:
        """Sondeo de red. Devuelve `(ok, motivo)`; se memoriza en `status()`."""
        if self.probe_url is None:
            return True, ""
        try:
            self.http.get(self.probe_url, timeout=min(10.0, self.http.timeout))
        except ProviderUnavailable as exc:
            return False, str(exc)
        except (RateLimited, TransportError, HttpStatusError) as exc:
            return False, f"sondeo fallido: {exc}"
        return True, ""

    def status(self) -> ProviderStatus:
        """Diagnóstico completo, apto para el mensaje de error del registro."""
        missing = self.missing_env()
        if missing:
            return ProviderStatus(
                self.name,
                False,
                tuple(missing),
                reason="credenciales ausentes en el entorno",
            )
        if self.probe_url is None:
            return ProviderStatus(self.name, True)
        now = self.clock.time()
        if self._probe is not None and now - self._probe[0] < self.probe_ttl_s:
            _, ok, reason = self._probe
        else:
            ok, reason = self.check_connectivity()
            self._probe = (now, ok, reason)
        return ProviderStatus(self.name, ok, reason=reason)

    def available(self) -> bool:
        return self.status().available

    def invalidate_probe(self) -> None:
        """Olvida el sondeo memorizado (tras un fallo en tiempo de ejecución)."""
        self._probe = None

    # -- HTTP -----------------------------------------------------------------

    @property
    def http(self) -> HttpClient:
        """Cliente HTTP perezoso con la política de tasa del proveedor."""
        if self._http is None:
            self._http = HttpClient(
                self.name,
                rate=rate_limit_for(self.name, self.settings),
                settings=self.settings,
                clock=self.clock,
                user_agent=(
                    self.settings.env("SEC_USER_AGENT")
                    if self.name in {"sec", "edgar"}
                    else None
                ),
            )
        return self._http

    def __repr__(self) -> str:  # pragma: no cover - cosmético
        return f"{type(self).__name__}(name={self.name!r}, kinds={self.kinds!r})"


@dataclass
class StaticProvider:
    """Proveedor de disponibilidad fija.

    Pensado para tests y para envolver fuentes que no dependen de red (el
    generador `synthetic`, o un fichero semilla ya en el repositorio).
    """

    name: str
    kinds: tuple[str, ...] = ()
    is_available: bool = True
    reason: str = ""
    missing_env: tuple[str, ...] = ()
    payload: Any = None

    def available(self) -> bool:
        return self.is_available

    def status(self) -> ProviderStatus:
        return ProviderStatus(
            self.name,
            self.is_available,
            tuple(self.missing_env),
            reason=self.reason,
        )


@dataclass(frozen=True, slots=True)
class ProviderAttempt:
    """Registro de un intento de uso de un proveedor dentro de una cadena.

    La cadena de fallback es silenciosa por diseño (el objetivo es que la llamada
    salga adelante), y eso es peligroso: si el proveedor primario lleva un mes
    caído, los datos vienen del secundario y nadie se entera. Estos registros
    quedan en `ProviderRegistry.attempts` y se escriben en el log a nivel WARNING.
    """

    kind: str
    provider: str
    priority: int
    ok: bool
    at: float
    reason: str = ""
    error_type: str = ""

    def describe(self) -> str:
        estado = "ok" if self.ok else f"fallo ({self.error_type}): {self.reason}"
        return f"[{self.kind}] {self.provider} (prioridad {self.priority}) -> {estado}"


# Errores que, lanzados en tiempo de ejecución, justifican pasar al siguiente
# proveedor de la cadena. `DataQualityError` queda fuera a propósito: unos datos
# que llegan pero no cuadran suelen indicar un fallo del *parser*, y taparlo
# saltando de proveedor convierte un bug reproducible en un misterio.
FALLBACK_ERRORS: tuple[type[BaseException], ...] = (
    ProviderUnavailable,
    RateLimited,
    TransportError,
    HttpStatusError,
)


class ProviderRegistry:
    """Registro de proveedores por tipo de dato, con prioridad y fallback.

    Convención de prioridad: **mayor número = se intenta antes**. Los empates se
    resuelven por orden de registro, de modo que el registro es determinista, que
    es requisito para que dos ejecuciones del mismo backtest usen las mismas
    fuentes.

    Dos niveles de fallback, deliberadamente distintos:

    - `resolve(kind)` elige el proveedor disponible de mayor prioridad
      *antes* de llamarlo, y si ninguno lo está lanza `ProviderUnavailable`
      enumerando, proveedor a proveedor, qué falta.
    - `call(kind, op)` ejecuta la operación y, si el proveedor elegido falla **en
      tiempo de ejecución** (429, 5xx, timeout, credencial caducada), lo pone en
      cuarentena, lo registra y prueba el siguiente. La cuarentena evita el peor
      caso: N tickers x un proveedor caído = N timeouts completos.
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        clock: Clock | None = None,
        failure_cooldown_s: float = 300.0,
        max_attempts_log: int = 500,
    ) -> None:
        self.settings = settings or get_settings()
        self.clock: Clock = clock or SystemClock()
        self.failure_cooldown_s = float(failure_cooldown_s)
        self.max_attempts_log = int(max_attempts_log)
        self._by_kind: dict[str, list[tuple[int, int, Provider]]] = {}
        self._seq = 0
        self._quarantine: dict[str, float] = {}
        self._attempts: list[ProviderAttempt] = []
        self._lock = threading.RLock()

    # -- registro -------------------------------------------------------------

    @staticmethod
    def _norm(kind: str) -> str:
        return str(kind).strip().lower()

    def register(
        self,
        kind: str,
        p: Provider,
        priority: int = 0,
        *,
        strict_kinds: bool = True,
        replace: bool = False,
    ) -> None:
        """Registra `p` como proveedor de `kind` con la prioridad dada.

        Valida el pato: `name` y `available()` son obligatorios. Con
        `strict_kinds=True`, si el proveedor declara `kinds` y `kind` no está en
        ellos, se rechaza: es una errata de programación y debe verse al arrancar,
        no seis horas después.
        """
        k = self._norm(kind)
        if not k:
            msg = "el `kind` no puede estar vacío"
            raise ConfigError(msg)
        name = getattr(p, "name", None)
        if not isinstance(name, str) or not name:
            msg = f"el proveedor {p!r} no expone un atributo `name` no vacío"
            raise ConfigError(msg)
        if not callable(getattr(p, "available", None)):
            msg = f"el proveedor {name!r} no implementa `available()`"
            raise ConfigError(msg)
        declared = tuple(getattr(p, "kinds", ()) or ())
        if strict_kinds and declared and k not in {self._norm(x) for x in declared}:
            msg = (
                f"el proveedor {name!r} declara kinds={declared} y no incluye {k!r}; "
                "usa strict_kinds=False si el registro es intencionado"
            )
            raise ConfigError(msg)

        with self._lock:
            entries = self._by_kind.setdefault(k, [])
            existing = [e for e in entries if getattr(e[2], "name", None) == name]
            if existing and not replace:
                msg = f"{name!r} ya está registrado para kind={k!r}; usa replace=True"
                raise ConfigError(msg)
            if existing:
                entries[:] = [e for e in entries if getattr(e[2], "name", None) != name]
            self._seq += 1
            entries.append((-int(priority), self._seq, p))
            entries.sort(key=lambda e: (e[0], e[1]))

    def unregister(self, kind: str, name: str) -> bool:
        """Elimina un proveedor. Devuelve True si estaba registrado."""
        k = self._norm(kind)
        with self._lock:
            entries = self._by_kind.get(k)
            if not entries:
                return False
            before = len(entries)
            entries[:] = [e for e in entries if getattr(e[2], "name", None) != name]
            return len(entries) != before

    def clear(self, kind: str | None = None) -> None:
        with self._lock:
            if kind is None:
                self._by_kind.clear()
                self._quarantine.clear()
            else:
                self._by_kind.pop(self._norm(kind), None)

    def kinds(self) -> list[str]:
        with self._lock:
            return sorted(self._by_kind)

    def providers_for(self, kind: str) -> list[Provider]:
        """Proveedores registrados para `kind`, de mayor a menor prioridad."""
        with self._lock:
            return [p for _, _, p in self._by_kind.get(self._norm(kind), [])]

    def priority_of(self, kind: str, name: str) -> int | None:
        with self._lock:
            for neg, _, p in self._by_kind.get(self._norm(kind), []):
                if getattr(p, "name", None) == name:
                    return -neg
        return None

    # -- cuarentena -----------------------------------------------------------

    def quarantine(self, name: str, seconds: float | None = None) -> None:
        """Aparta un proveedor durante `seconds` (por defecto, el cooldown)."""
        secs = self.failure_cooldown_s if seconds is None else float(seconds)
        if secs <= 0:
            return
        with self._lock:
            self._quarantine[name] = self.clock.time() + secs

    def is_quarantined(self, name: str) -> bool:
        with self._lock:
            until = self._quarantine.get(name)
            if until is None:
                return False
            if self.clock.time() >= until:
                del self._quarantine[name]
                return False
            return True

    def release(self, name: str) -> None:
        with self._lock:
            self._quarantine.pop(name, None)

    # -- diagnóstico ----------------------------------------------------------

    def _status_of(self, p: Provider, priority: int) -> ProviderStatus:
        name = getattr(p, "name", repr(p))
        if self.is_quarantined(name):
            with self._lock:
                until = self._quarantine.get(name, self.clock.time())
            left = max(0.0, until - self.clock.time())
            return ProviderStatus(
                name, False, reason=f"en cuarentena {left:.0f}s tras un fallo", priority=priority
            )
        status_fn = getattr(p, "status", None)
        if callable(status_fn):
            try:
                st = status_fn()
            except Exception as exc:
                return ProviderStatus(
                    name, False, reason=f"status() falló: {exc}", priority=priority
                )
            if isinstance(st, ProviderStatus):
                return ProviderStatus(
                    st.provider, st.available, st.missing_env, st.reason, priority
                )
        try:
            ok = bool(p.available())
        except Exception as exc:
            return ProviderStatus(
                name, False, reason=f"available() falló: {exc}", priority=priority
            )
        missing: tuple[str, ...] = ()
        if not ok and name in PROVIDER_ENV_KEYS:
            missing = tuple(self.settings.missing_credentials_for(name))
        reason = "" if ok else ("credenciales o conectividad ausentes" if not missing else "")
        return ProviderStatus(name, ok, missing, reason, priority)

    def statuses(self, kind: str) -> list[ProviderStatus]:
        """Diagnóstico de todos los candidatos de `kind`, en orden de prioridad."""
        with self._lock:
            entries = list(self._by_kind.get(self._norm(kind), []))
        return [self._status_of(p, -neg) for neg, _, p in entries]

    def describe(self, kind: str | None = None) -> str:
        """Informe legible del registro, para el CLI y los mensajes de error."""
        kinds = [self._norm(kind)] if kind else self.kinds()
        lines: list[str] = []
        for k in kinds:
            lines.append(f"{k}:")
            sts = self.statuses(k)
            if not sts:
                lines.append("  (sin proveedores registrados)")
            lines.extend(f"  - {s.describe()}" for s in sts)
        return "\n".join(lines)

    @property
    def attempts(self) -> tuple[ProviderAttempt, ...]:
        """Bitácora de intentos de la cadena de fallback (los más recientes al final)."""
        with self._lock:
            return tuple(self._attempts)

    def _log_attempt(self, attempt: ProviderAttempt) -> None:
        with self._lock:
            self._attempts.append(attempt)
            if len(self._attempts) > self.max_attempts_log:
                del self._attempts[: len(self._attempts) - self.max_attempts_log]

    # -- resolución -----------------------------------------------------------

    def _unavailable(self, kind: str, statuses: Sequence[ProviderStatus]) -> ProviderUnavailable:
        k = self._norm(kind)
        if not statuses:
            reason = (
                f"no hay ningún proveedor registrado para kind={k!r}. "
                f"Kinds con proveedores: {self.kinds() or 'ninguno'}"
            )
            return ProviderUnavailable(k, reason)
        detalle = "\n".join(f"  - {s.describe()}" for s in statuses)
        missing: list[str] = []
        for s in statuses:
            missing.extend(e for e in s.missing_env if e not in missing)
        reason = f"ningún proveedor puede servir kind={k!r}. Candidatos evaluados:\n{detalle}"
        return ProviderUnavailable(k, reason, missing_env=missing)

    def resolve(self, kind: str) -> Provider:
        """Proveedor disponible de mayor prioridad para `kind`.

        Si ninguno está disponible lanza `ProviderUnavailable` cuyo mensaje lista
        **cada candidato** con lo que le falta: variables de entorno concretas o el
        motivo de conectividad. Es el mensaje que el usuario tiene que poder
        convertir en un `export` sin abrir el código.
        """
        statuses: list[ProviderStatus] = []
        with self._lock:
            entries = list(self._by_kind.get(self._norm(kind), []))
        for neg, _, p in entries:
            st = self._status_of(p, -neg)
            statuses.append(st)
            if st.available:
                return p
        raise self._unavailable(kind, statuses)

    def iter_available(self, kind: str) -> Iterator[Provider]:
        """Itera los proveedores disponibles de `kind`, de mayor a menor prioridad."""
        with self._lock:
            entries = list(self._by_kind.get(self._norm(kind), []))
        for neg, _, p in entries:
            if self._status_of(p, -neg).available:
                yield p

    def call(
        self,
        kind: str,
        op: Callable[[Provider], T],
        *,
        fallback_errors: tuple[type[BaseException], ...] = FALLBACK_ERRORS,
        description: str = "",
        cooldown_s: float | None = None,
    ) -> T:
        """Ejecuta `op` sobre el mejor proveedor, con fallback en cadena.

        Recorre los candidatos por prioridad. Si el elegido falla con uno de
        `fallback_errors` **durante la llamada**, se registra el fallo, se pone al
        proveedor en cuarentena y se prueba el siguiente. Cualquier otra excepción
        se propaga sin más: un `KeyError` del *parser* es un bug, no una razón para
        cambiar de fuente.

        Si todos fallan, lanza `ProviderUnavailable` con el detalle de cada uno,
        encadenando la última excepción como `__cause__`.
        """
        k = self._norm(kind)
        with self._lock:
            entries = list(self._by_kind.get(k, []))
        statuses: list[ProviderStatus] = []
        last_exc: BaseException | None = None

        for neg, _, provider in entries:
            priority = -neg
            st = self._status_of(provider, priority)
            if not st.available:
                statuses.append(st)
                self._log_attempt(
                    ProviderAttempt(
                        kind=k,
                        provider=st.provider,
                        priority=priority,
                        ok=False,
                        at=self.clock.time(),
                        reason=st.describe(),
                        error_type="no disponible",
                    )
                )
                continue
            name = getattr(provider, "name", repr(provider))
            try:
                result = op(provider)
            except fallback_errors as exc:
                last_exc = exc
                statuses.append(
                    ProviderStatus(
                        name,
                        False,
                        tuple(getattr(exc, "missing_env", ()) or ()),
                        reason=f"falló en ejecución: {exc}",
                        priority=priority,
                    )
                )
                self._log_attempt(
                    ProviderAttempt(
                        kind=k,
                        provider=name,
                        priority=priority,
                        ok=False,
                        at=self.clock.time(),
                        reason=str(exc),
                        error_type=type(exc).__name__,
                    )
                )
                logger.warning(
                    "proveedor %s falló para kind=%s%s (%s: %s); se prueba el siguiente",
                    name,
                    k,
                    f" [{description}]" if description else "",
                    type(exc).__name__,
                    exc,
                )
                self.quarantine(name, cooldown_s)
                invalidate = getattr(provider, "invalidate_probe", None)
                if callable(invalidate):
                    invalidate()
                continue
            self._log_attempt(
                ProviderAttempt(
                    kind=k,
                    provider=name,
                    priority=priority,
                    ok=True,
                    at=self.clock.time(),
                    reason=description,
                )
            )
            return result

        exc_final = self._unavailable(k, statuses)
        if last_exc is not None:
            raise exc_final from last_exc
        raise exc_final

    def __repr__(self) -> str:  # pragma: no cover - cosmético
        n = sum(len(v) for v in self._by_kind.values())
        return f"ProviderRegistry(kinds={len(self._by_kind)}, providers={n})"


_REGISTRY: ProviderRegistry | None = None
_REGISTRY_LOCK = threading.Lock()


def get_registry() -> ProviderRegistry:
    """Registro global del proceso.

    Los adaptadores (`data.prices`, `data.edgar`…) se registran aquí al
    importarse, y el resto del sistema resuelve contra él. Los tests deben usar
    una instancia propia —`ProviderRegistry()`— o `set_registry(None)` para no
    contaminarse entre sí.
    """
    global _REGISTRY
    with _REGISTRY_LOCK:
        if _REGISTRY is None:
            _REGISTRY = ProviderRegistry()
        return _REGISTRY


def set_registry(registry: ProviderRegistry | None) -> None:
    """Sustituye (o borra, con `None`) el registro global. Uso previsto: tests."""
    global _REGISTRY
    with _REGISTRY_LOCK:
        _REGISTRY = registry
