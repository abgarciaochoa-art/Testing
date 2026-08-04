"""Capa de datos: proveedores, caché y generador sintético.

Este paquete concentra **todo** el acceso a datos externos. La regla que lo
gobierna es la del contrato §0.3: un proveedor sin credenciales o sin red lanza
`ProviderUnavailable`; nunca devuelve un panel vacío que aguas abajo se confunda
con "ese trimestre no hubo datos".

Piezas del núcleo (este módulo reexporta lo público de `base` y `cache`):

- `Provider` / `BaseProvider` / `ProviderRegistry`: qué proveedor sirve qué tipo de
  dato, con prioridades y fallback en cadena registrado.
- `HttpClient`: limitación de tasa por *token bucket*, reintentos con backoff
  exponencial y jitter, respeto de `Retry-After` y traducción de códigos HTTP a los
  errores del repositorio.
- `DiskCache`: caché en disco con claves `(kind, provider, hash de params)`,
  parquet para paneles, TTL, distinción inmutable/mutable y modo offline.

Los adaptadores concretos viven en módulos hermanos y se importan **bajo demanda**,
nunca aquí: `prices`, `edgar`, `fundamentals`, `estimates`, `options`,
`shortinterest` y `synthetic`. Importarlos desde este `__init__` obligaría a tener
todas sus dependencias instaladas para usar cualquier parte de la capa de datos.
"""

from __future__ import annotations

from earnings_alpha.data.base import (
    KNOWN_KINDS,
    PROVIDER_RATE_LIMITS,
    BaseProvider,
    Clock,
    DataKind,
    HttpClient,
    HttpRequest,
    HttpResponse,
    HttpStats,
    HttpStatusError,
    ManualClock,
    Provider,
    ProviderAttempt,
    ProviderRegistry,
    ProviderStatus,
    RateLimitPolicy,
    RequestsTransport,
    RetryPolicy,
    ScriptedTransport,
    StaticProvider,
    SystemClock,
    TokenBucket,
    Transport,
    TransportError,
    TransportTimeout,
    get_registry,
    parse_retry_after,
    rate_limit_for,
    set_registry,
)
from earnings_alpha.data.cache import (
    CACHE_FORMAT_VERSION,
    CacheKey,
    CacheMeta,
    CacheMiss,
    CacheSlot,
    CacheStats,
    DiskCache,
    ImmutabilityPolicy,
    ManualWallClock,
    SystemWallClock,
    WallClock,
    cached,
    canonicalize,
    params_hash,
)

__all__ = [  # noqa: RUF022 - orden temático, no alfabético
    # proveedores
    "Provider",
    "BaseProvider",
    "StaticProvider",
    "ProviderStatus",
    "ProviderAttempt",
    "ProviderRegistry",
    "get_registry",
    "set_registry",
    "DataKind",
    "KNOWN_KINDS",
    # HTTP
    "HttpClient",
    "HttpRequest",
    "HttpResponse",
    "HttpStats",
    "HttpStatusError",
    "Transport",
    "RequestsTransport",
    "ScriptedTransport",
    "TransportError",
    "TransportTimeout",
    "RateLimitPolicy",
    "RetryPolicy",
    "TokenBucket",
    "PROVIDER_RATE_LIMITS",
    "rate_limit_for",
    "parse_retry_after",
    # relojes
    "Clock",
    "SystemClock",
    "ManualClock",
    "WallClock",
    "SystemWallClock",
    "ManualWallClock",
    # caché
    "DiskCache",
    "CacheKey",
    "CacheMeta",
    "CacheMiss",
    "CacheSlot",
    "CacheStats",
    "CACHE_FORMAT_VERSION",
    "ImmutabilityPolicy",
    "cached",
    "canonicalize",
    "params_hash",
]
