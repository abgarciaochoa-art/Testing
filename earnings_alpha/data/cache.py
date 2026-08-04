"""Caché en disco de respuestas de proveedores: parquet, TTL y modo offline.

Por qué una caché es aquí una pieza *metodológica* y no una optimización
--------------------------------------------------------------------------
Un backtest sobre el S&P 500 completo con veinte años de fundamentales toca
cientos de miles de respuestas de proveedores con cuota. Sin caché, cada
re-ejecución vuelve a pedirlas, tarda horas y —lo grave— **puede recibir datos
distintos**: los proveedores reexpresan, corrigen y rellenan hacia atrás. Un
resultado que no se puede reproducir bit a bit no es un resultado.

De ahí las tres decisiones de diseño de este módulo:

1. **Clave `(kind, provider, hash(params))`.** El `provider` forma parte de la
   clave a propósito: los precios de Polygon y los de yfinance para el mismo
   ticker y rango **no** son intercambiables (ajustes distintos), y mezclarlos en
   una misma entrada produciría series con saltos artificiales.
2. **Inmutable vs mutable.** Un histórico ya cerrado —digamos, los precios de 2019
   pedidos hoy— no volverá a cambiar salvo reexpresión, y cachearlo con caducidad
   solo genera tráfico inútil. Los días recientes, en cambio, **sí** cambian: una
   barra del día en curso se consolida horas después del cierre, y el consenso de
   analistas se revisa a diario. Guardar lo reciente sin TTL congela un dato
   provisional dentro del backtest. `ImmutabilityPolicy` decide a partir de la
   fecha final de la petición, con un margen de días configurable.
3. **Modo offline.** `DiskCache(offline=True)` convierte cualquier fallo de caché
   en `ProviderUnavailable` y **nunca** invoca al proveedor. Es lo que permite
   afirmar "este backtest se ejecutó exactamente sobre estos datos": si algo no
   estaba cacheado, el proceso falla en vez de salir a la red y traer una versión
   distinta de la historia.

Formatos
--------
- `DataFrame`/`Series` → **parquet** (pyarrow). Preserva dtypes, MultiIndex
  `(date, ticker)`, resolución temporal y zonas horarias, que es exactamente lo
  que CSV destruye: un `date` que vuelve como texto o un `datetime64[ns]` que
  vuelve como `[us]` rompe `pd.merge_asof` en `pit.asof_join`.
- `bytes` → fichero binario (cuerpos crudos de EDGAR, ZIP masivos).
- Cualquier otro objeto serializable → **JSON** con etiquetas de tipo, para que
  `date`, `datetime`, `tuple`, `set` y `bytes` sobrevivan al viaje de ida y vuelta.

Nunca se usa `pickle`: una caché es un fichero que se comparte y se conserva meses,
y deserializar pickle es ejecutar código arbitrario.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import math
import os
import re
import shutil
import threading
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Any, Protocol, TypeVar, runtime_checkable

import numpy as np
import pandas as pd

from earnings_alpha.config import Settings, get_settings
from earnings_alpha.errors import (
    ConfigError,
    DataQualityError,
    EarningsAlphaError,
    ProviderUnavailable,
)

__all__ = [  # noqa: RUF022 - orden temático, no alfabético
    "CACHE_FORMAT_VERSION",
    "CacheMiss",
    "CacheKey",
    "CacheMeta",
    "CacheStats",
    "CacheSlot",
    "DiskCache",
    "ImmutabilityPolicy",
    "WallClock",
    "SystemWallClock",
    "ManualWallClock",
    "canonicalize",
    "params_hash",
    "cached",
]

logger = logging.getLogger(__name__)

T = TypeVar("T")

CACHE_FORMAT_VERSION = 1
"""Versión del formato de metadatos. Al subirla, las entradas antiguas se ignoran
(fallo de caché limpio) en vez de leerse mal: una caché con un esquema viejo que se
interpreta con el nuevo es una fuente silenciosa de datos incorrectos."""

_MAX_INLINE_SEQ = 32
"""Longitud a partir de la cual una secuencia en los parámetros se resume por
digest. Un `params` con 500 tickers haría el fichero de metadatos ilegible y la
clave gigantesca sin ganar unicidad."""

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.=-]+")


# ===========================================================================
# 1. Reloj de pared inyectable
# ===========================================================================


@runtime_checkable
class WallClock(Protocol):
    """Fuente de la hora **de pared** en UTC.

    La caducidad se compara contra `written_at`, que es un timestamp persistido:
    debe medirse con hora de calendario, no con el reloj monótono de
    `data.base.Clock` (que reinicia en cada proceso).
    """

    def now(self) -> datetime:
        """Instante actual, tz-aware en UTC."""


class SystemWallClock:
    """Reloj real en UTC."""

    __slots__ = ()

    def now(self) -> datetime:
        return datetime.now(UTC)


class ManualWallClock:
    """Reloj de pared simulado, para probar TTL sin esperar.

    `advance()` acepta segundos o `timedelta`.
    """

    __slots__ = ("_now",)

    def __init__(self, start: datetime | None = None) -> None:
        base = start or datetime(2026, 1, 2, 12, 0, tzinfo=UTC)
        self._now = base if base.tzinfo else base.replace(tzinfo=UTC)

    def now(self) -> datetime:
        return self._now

    def advance(self, delta: timedelta | float) -> datetime:
        step = delta if isinstance(delta, timedelta) else timedelta(seconds=float(delta))
        self._now = self._now + step
        return self._now

    def set(self, when: datetime) -> None:
        self._now = when if when.tzinfo else when.replace(tzinfo=UTC)

    def __repr__(self) -> str:  # pragma: no cover - cosmético
        return f"ManualWallClock({self._now.isoformat()})"


# ===========================================================================
# 2. Canonicalización de parámetros y clave
# ===========================================================================


class CacheMiss(EarningsAlphaError):  # noqa: N818 - nomenclatura de errors.py
    """No hay entrada válida en caché para la clave pedida.

    Es una condición **normal** en modo conectado (se llama al proveedor y se
    cachea) y un **error** en modo offline, donde se traduce a
    `ProviderUnavailable`.
    """

    def __init__(self, kind: str, provider: str, params_hash: str, reason: str = "") -> None:
        self.kind = kind
        self.provider = provider
        self.params_hash = params_hash
        self.reason = reason
        detail = f"fallo de caché para kind={kind!r} provider={provider!r} hash={params_hash[:12]}"
        if reason:
            detail += f": {reason}"
        super().__init__(detail)


def _digest(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _seq_digest(items: Sequence[Any]) -> dict[str, Any]:
    dumped = json.dumps(items, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return {"__seq__": {"len": len(items), "sha256": _digest(dumped)[:32]}}


def canonicalize(obj: Any) -> Any:
    """Convierte un parámetro en una estructura JSON determinista y con tipo.

    Las etiquetas (`__date__`, `__tuple__`, `__bytes__`…) no son decorativas:
    evitan colisiones que producirían **datos equivocados**. Sin ellas,
    `{"end": date(2024,1,1)}` y `{"end": "2024-01-01"}` compartirían clave, y una
    petición de precios hasta esa fecha podría servir la respuesta de otra.

    Las secuencias largas se resumen por digest: la clave debe ser corta y estable,
    no un volcado de 500 tickers.

    Un objeto no serializable lanza `ConfigError`. No se recurre a `repr()`: el
    `repr` por defecto incluye la dirección de memoria, así que la misma petición
    generaría una clave distinta en cada proceso y la caché nunca acertaría.
    """
    if obj is None or isinstance(obj, bool | int | str):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj):
            return {"__float__": "nan"}
        if math.isinf(obj):
            return {"__float__": "inf" if obj > 0 else "-inf"}
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return canonicalize(float(obj))
    if isinstance(obj, np.datetime64):
        return canonicalize(pd.Timestamp(obj))
    if isinstance(obj, datetime):  # incluye pd.Timestamp
        return {"__datetime__": obj.isoformat()}
    if isinstance(obj, date):
        return {"__date__": obj.isoformat()}
    if isinstance(obj, timedelta):
        return {"__timedelta__": obj.total_seconds()}
    if isinstance(obj, bytes | bytearray):
        raw = bytes(obj)
        if len(raw) > _MAX_INLINE_SEQ:
            return {"__bytes_sha256__": _digest(raw.hex())[:32], "len": len(raw)}
        return {"__bytes__": base64.b64encode(raw).decode("ascii")}
    if isinstance(obj, Path):
        return {"__path__": str(obj)}
    if isinstance(obj, pd.Timedelta):  # pragma: no cover - cubierto por timedelta
        return {"__timedelta__": obj.total_seconds()}
    if isinstance(obj, pd.Index | pd.Series):
        return canonicalize(list(obj.tolist()))
    if isinstance(obj, pd.DataFrame):
        msg = (
            "un DataFrame no puede ser parámetro de caché: la clave debe ser pequeña y "
            "estable. Pasa las columnas o el rango que lo describen."
        )
        raise ConfigError(msg)
    if isinstance(obj, Mapping):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            key = k if isinstance(k, str) else f"{type(k).__name__}:{canonicalize(k)}"
            out[str(key)] = canonicalize(v)
        return dict(sorted(out.items()))
    if isinstance(obj, frozenset | set):
        items = sorted(
            (canonicalize(v) for v in obj),
            key=lambda x: json.dumps(x, sort_keys=True, ensure_ascii=False),
        )
        return {"__set__": items if len(items) <= _MAX_INLINE_SEQ else _seq_digest(items)}
    if isinstance(obj, tuple):
        items = [canonicalize(v) for v in obj]
        return {"__tuple__": items if len(items) <= _MAX_INLINE_SEQ else _seq_digest(items)}
    if isinstance(obj, list):
        items = [canonicalize(v) for v in obj]
        return items if len(items) <= _MAX_INLINE_SEQ else _seq_digest(items)
    cache_key = getattr(obj, "cache_key", None)
    if callable(cache_key):
        return canonicalize(cache_key())
    msg = (
        f"parámetro de caché no serializable: {type(obj).__name__}. Implementa "
        "`cache_key()` en el objeto o pasa un valor primitivo; usar `repr()` daría "
        "claves distintas en cada proceso."
    )
    raise ConfigError(msg)


def params_hash(kind: str, provider: str, params: Mapping[str, Any] | None) -> str:
    """SHA-256 de `(versión, kind, provider, params canonicalizados)`."""
    payload = {
        "v": CACHE_FORMAT_VERSION,
        "kind": str(kind),
        "provider": str(provider),
        "params": canonicalize(dict(params or {})),
    }
    return _digest(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


def _safe(name: str) -> str:
    cleaned = _SAFE_NAME.sub("_", str(name).strip())
    return cleaned or "_"


@dataclass(frozen=True, slots=True)
class CacheKey:
    """Clave de caché: `(kind, provider, hash de params)` (contrato §3.3)."""

    kind: str
    provider: str
    params_hash: str

    @classmethod
    def build(
        cls, kind: str, provider: str, params: Mapping[str, Any] | None = None
    ) -> CacheKey:
        return cls(str(kind), str(provider), params_hash(kind, provider, params))

    @property
    def relative_dir(self) -> Path:
        """Subdirectorio; el prefijo de dos dígitos evita directorios con 10⁵ ficheros."""
        return Path(_safe(self.kind)) / _safe(self.provider) / self.params_hash[:2]

    @property
    def stem(self) -> str:
        return self.params_hash

    def __str__(self) -> str:  # pragma: no cover - cosmético
        return f"{self.kind}/{self.provider}/{self.params_hash[:12]}"


# ===========================================================================
# 3. Metadatos
# ===========================================================================

_SUFFIX_BY_PAYLOAD = {
    "dataframe": ".parquet",
    "series": ".parquet",
    "bytes": ".bin",
    "json": ".json",
}


@dataclass(frozen=True, slots=True)
class CacheMeta:
    """Metadatos persistidos junto a cada entrada.

    `written_at` es el timestamp de escritura y la referencia de la caducidad;
    `immutable=True` significa "este dato ya no puede cambiar" y anula el TTL.
    """

    kind: str
    provider: str
    params_hash: str
    payload: str
    written_at: datetime
    ttl_s: float | None = None
    immutable: bool = False
    rows: int | None = None
    n_bytes: int = 0
    tags: tuple[str, ...] = ()
    params: dict[str, Any] = field(default_factory=dict)
    series_name: str | None = None
    version: int = CACHE_FORMAT_VERSION
    note: str = ""

    @property
    def key(self) -> CacheKey:
        return CacheKey(self.kind, self.provider, self.params_hash)

    @property
    def suffix(self) -> str:
        return _SUFFIX_BY_PAYLOAD.get(self.payload, ".bin")

    @property
    def expires_at(self) -> datetime | None:
        """Instante de caducidad; `None` si la entrada es inmutable o sin TTL."""
        if self.immutable or self.ttl_s is None:
            return None
        return self.written_at + timedelta(seconds=float(self.ttl_s))

    def is_expired(self, now: datetime) -> bool:
        exp = self.expires_at
        return exp is not None and now >= exp

    def age_s(self, now: datetime) -> float:
        return (now - self.written_at).total_seconds()

    def to_json(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "kind": self.kind,
            "provider": self.provider,
            "params_hash": self.params_hash,
            "payload": self.payload,
            "written_at": self.written_at.astimezone(UTC).isoformat(),
            "ttl_s": self.ttl_s,
            "immutable": self.immutable,
            "expires_at": (self.expires_at.isoformat() if self.expires_at else None),
            "rows": self.rows,
            "n_bytes": self.n_bytes,
            "tags": list(self.tags),
            "params": self.params,
            "series_name": self.series_name,
            "note": self.note,
        }

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> CacheMeta:
        written = datetime.fromisoformat(str(raw["written_at"]))
        if written.tzinfo is None:
            written = written.replace(tzinfo=UTC)
        return cls(
            kind=str(raw["kind"]),
            provider=str(raw["provider"]),
            params_hash=str(raw["params_hash"]),
            payload=str(raw["payload"]),
            written_at=written,
            ttl_s=(None if raw.get("ttl_s") is None else float(raw["ttl_s"])),
            immutable=bool(raw.get("immutable", False)),
            rows=(None if raw.get("rows") is None else int(raw["rows"])),
            n_bytes=int(raw.get("n_bytes", 0)),
            tags=tuple(raw.get("tags", ()) or ()),
            params=dict(raw.get("params", {}) or {}),
            series_name=raw.get("series_name"),
            version=int(raw.get("version", 0)),
            note=str(raw.get("note", "")),
        )


@dataclass(slots=True)
class CacheStats:
    """Contadores de uso. Un `hit_rate` bajo suele delatar parámetros inestables
    (un `datetime.now()` colándose en la clave) más que una caché pequeña."""

    hits: int = 0
    stale_hits: int = 0
    misses: int = 0
    expired: int = 0
    writes: int = 0
    errors: int = 0
    invalidations: int = 0
    bytes_written: int = 0

    @property
    def lookups(self) -> int:
        return self.hits + self.stale_hits + self.misses

    @property
    def hit_rate(self) -> float:
        return 0.0 if self.lookups == 0 else (self.hits + self.stale_hits) / self.lookups


# ===========================================================================
# 4. Política de inmutabilidad
# ===========================================================================


@dataclass(frozen=True, slots=True)
class ImmutabilityPolicy:
    """Decide si una petición devuelve datos ya cerrados (inmutables).

    Heurística: si los parámetros describen una ventana que **termina** hace más de
    `settled_after_days` días naturales, el resultado ya no cambiará de forma
    material y se cachea sin caducidad. Si la ventana llega hasta hoy —o no tiene
    fin declarado— el resultado es mutable y recibe TTL.

    El margen por defecto (5 días) no es arbitrario: cubre el fin de semana más el
    retraso típico de consolidación de un proveedor EOD, y en el lado de eventos
    absorbe las correcciones de última hora del calendario de resultados. Es una
    heurística conservadora, y por eso `write(immutable=...)` siempre puede
    imponerse a ella cuando el adaptador sabe más (p. ej. un 10-K de 2015 ya
    presentado es inmutable con independencia de las fechas del `params`).
    """

    settled_after_days: int = 5
    end_keys: tuple[str, ...] = ("end", "end_date", "to", "until", "through", "as_of")
    mutable_kinds: frozenset[str] = frozenset({"earnings_calendar", "estimates", "news"})
    """Tipos que se revisan hacia atrás aunque la ventana esté cerrada: el consenso
    y el calendario de resultados se corrigen a posteriori, así que jamás se marcan
    inmutables automáticamente."""

    def end_of(self, params: Mapping[str, Any] | None) -> date | None:
        """Extrae la fecha final declarada en los parámetros, si la hay."""
        if not params:
            return None
        for k in self.end_keys:
            if k not in params:
                continue
            value = params[k]
            parsed = _as_date(value)
            if parsed is not None:
                return parsed
        return None

    def classify(
        self,
        kind: str,
        params: Mapping[str, Any] | None,
        *,
        today: date | None = None,
    ) -> bool:
        """True si la entrada puede considerarse inmutable."""
        if str(kind).strip().lower() in self.mutable_kinds:
            return False
        end = self.end_of(params)
        if end is None:
            return False
        ref = today or datetime.now(UTC).date()
        return end < ref - timedelta(days=self.settled_after_days)


def _as_date(value: Any) -> date | None:
    """Interpreta un valor como fecha; `None` si no lo es de forma inequívoca."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, np.datetime64 | pd.Timestamp):
        return pd.Timestamp(value).date()
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.strip()).date()
        except ValueError:
            return None
    return None


# ===========================================================================
# 5. La caché
# ===========================================================================


@dataclass
class CacheSlot:
    """Hueco de caché devuelto por `DiskCache.entry()`.

    Uso típico::

        with cache.entry("prices", "polygon", params) as slot:
            if not slot.hit:
                slot.store(descargar())
        panel = slot.value
    """

    key: CacheKey
    hit: bool
    value: Any = None
    meta: CacheMeta | None = None
    _cache: DiskCache | None = None
    _ttl: float | timedelta | None = None
    _immutable: bool | None = None
    _tags: tuple[str, ...] = ()
    _allow_empty: bool = False
    _params: dict[str, Any] = field(default_factory=dict)

    def store(self, value: Any) -> None:
        """Guarda `value` en la caché y lo deja disponible en `slot.value`."""
        if self._cache is None:  # pragma: no cover - defensivo
            msg = "el slot no está asociado a ninguna caché"
            raise ConfigError(msg)
        self.meta = self._cache.write(
            self.key.kind,
            self.key.provider,
            self._params,
            value,
            ttl=self._ttl,
            immutable=self._immutable,
            tags=self._tags,
            allow_empty=self._allow_empty,
        )
        self.value = value


class DiskCache:
    """Caché en disco con claves `(kind, provider, params_hash)`.

    Parámetros
    ----------
    root:
        Directorio raíz. Por defecto `Settings.cache_dir` (`data/cache`).
    offline:
        Si es True, **ninguna** operación sale a la red: un fallo de caché es
        `ProviderUnavailable`, y `fetch()` jamás invoca al proveedor. Por defecto
        toma el valor de `Settings.offline` (que a su vez lee
        `EARNINGS_ALPHA_OFFLINE`).
    serve_stale_offline:
        En modo offline, servir entradas caducadas en vez de fallar. Es lo correcto
        por defecto: el TTL existe para provocar un refresco por red que offline no
        puede hacer, y una entrada caducada sigue siendo el dato exacto con el que
        se ejecutó el backtest. Se contabiliza aparte (`stats.stale_hits`).
    enabled:
        `False` desactiva la caché por completo (equivalente a `--no-cache`), sin
        que quien llama tenga que cambiar de código.
    """

    def __init__(
        self,
        root: Path | str | None = None,
        *,
        offline: bool | None = None,
        default_ttl_s: float | None = None,
        settings: Settings | None = None,
        clock: WallClock | None = None,
        immutability: ImmutabilityPolicy | None = None,
        serve_stale_offline: bool = True,
        enabled: bool = True,
    ) -> None:
        cfg = settings or get_settings()
        self.settings = cfg
        self.root = Path(root) if root is not None else Path(cfg.cache_dir)
        self.offline = bool(cfg.offline if offline is None else offline)
        self.default_ttl_s = float(
            default_ttl_s if default_ttl_s is not None else cfg.cache_ttl_days * 86_400.0
        )
        self.clock: WallClock = clock or SystemWallClock()
        self.immutability = immutability or ImmutabilityPolicy()
        self.serve_stale_offline = bool(serve_stale_offline)
        self.enabled = bool(enabled)
        self.stats = CacheStats()
        self._lock = threading.RLock()
        self.root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_settings(cls, settings: Settings | None = None, **kwargs: Any) -> DiskCache:
        cfg = settings or get_settings()
        return cls(cfg.cache_dir, settings=cfg, **kwargs)

    # -- rutas ---------------------------------------------------------------

    def key_for(
        self, kind: str, provider: str, params: Mapping[str, Any] | None = None
    ) -> CacheKey:
        return CacheKey.build(kind, provider, params)

    def _dir(self, key: CacheKey) -> Path:
        return self.root / key.relative_dir

    def _meta_path(self, key: CacheKey) -> Path:
        return self._dir(key) / f"{key.stem}.meta.json"

    def _payload_path(self, key: CacheKey, meta: CacheMeta) -> Path:
        return self._dir(key) / f"{key.stem}{meta.suffix}"

    def path_of(
        self, kind: str, provider: str, params: Mapping[str, Any] | None = None
    ) -> Path | None:
        """Ruta del fichero de datos de una entrada existente, o `None`."""
        key = self.key_for(kind, provider, params)
        meta = self._read_meta(key)
        return None if meta is None else self._payload_path(key, meta)

    # -- metadatos ------------------------------------------------------------

    def _read_meta(self, key: CacheKey) -> CacheMeta | None:
        path = self._meta_path(key)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            meta = CacheMeta.from_json(raw)
        except (OSError, ValueError, KeyError) as exc:
            logger.warning("metadatos de caché ilegibles en %s: %s", path, exc)
            self.stats.errors += 1
            self._remove(key, None)
            return None
        if meta.version != CACHE_FORMAT_VERSION:
            logger.info(
                "entrada de caché con formato v%s (actual v%s): se ignora",
                meta.version,
                CACHE_FORMAT_VERSION,
            )
            return None
        return meta

    def stat(
        self, kind: str, provider: str, params: Mapping[str, Any] | None = None
    ) -> CacheMeta | None:
        """Metadatos de la entrada, aunque esté caducada. `None` si no existe."""
        return self._read_meta(self.key_for(kind, provider, params))

    def has(
        self,
        kind: str,
        provider: str,
        params: Mapping[str, Any] | None = None,
        *,
        fresh: bool = True,
    ) -> bool:
        """¿Existe la entrada? Con `fresh=True`, además, sin caducar."""
        meta = self.stat(kind, provider, params)
        if meta is None:
            return False
        return not (fresh and meta.is_expired(self.clock.now()))

    # -- lectura --------------------------------------------------------------

    def read(
        self,
        kind: str,
        provider: str,
        params: Mapping[str, Any] | None = None,
        *,
        allow_stale: bool = False,
    ) -> Any:
        """Devuelve el valor cacheado o lanza.

        Lanza `CacheMiss` si no hay entrada válida; en **modo offline** lanza
        `ProviderUnavailable`, porque ahí un fallo de caché no tiene remedio: no se
        puede salir a la red a buscarlo.
        """
        key = self.key_for(kind, provider, params)
        if not self.enabled:
            self.stats.misses += 1
            raise self._miss(key, "la caché está desactivada (enabled=False)")

        meta = self._read_meta(key)
        if meta is None:
            self.stats.misses += 1
            raise self._miss(key, "no hay entrada")

        now = self.clock.now()
        stale = meta.is_expired(now)
        if stale and not (allow_stale or (self.offline and self.serve_stale_offline)):
            self.stats.expired += 1
            self.stats.misses += 1
            age = meta.age_s(now)
            raise self._miss(
                key,
                f"entrada caducada (edad {age / 3600:.1f} h, ttl {meta.ttl_s} s)",
            )

        path = self._payload_path(key, meta)
        try:
            value = _load_payload(path, meta)
        except (OSError, ValueError, DataQualityError) as exc:
            logger.warning("entrada de caché corrupta en %s: %s", path, exc)
            self.stats.errors += 1
            self.stats.misses += 1
            self._remove(key, meta)
            raise self._miss(key, f"entrada corrupta: {exc}") from exc

        if stale:
            self.stats.stale_hits += 1
            logger.info(
                "modo offline: se sirve una entrada caducada de %s/%s (edad %.1f h)",
                kind,
                provider,
                meta.age_s(now) / 3600.0,
            )
        else:
            self.stats.hits += 1
        return value

    def get(
        self,
        kind: str,
        provider: str,
        params: Mapping[str, Any] | None = None,
        *,
        default: Any = None,
        allow_stale: bool = False,
    ) -> Any:
        """Como `read` pero devuelve `default` en vez de lanzar `CacheMiss`.

        En modo offline **sí** lanza `ProviderUnavailable`: devolver `None` allí
        induciría a quien llama a intentar la red, que es justo lo que el modo
        offline prohíbe.
        """
        try:
            return self.read(kind, provider, params, allow_stale=allow_stale)
        except CacheMiss:
            return default

    def _miss(self, key: CacheKey, reason: str) -> EarningsAlphaError:
        if self.offline:
            return ProviderUnavailable(
                key.provider,
                (
                    f"modo offline y {reason} para kind={key.kind!r} "
                    f"(hash {key.params_hash[:12]}). El modo offline no sale a la red: "
                    "ejecuta primero la carga con conectividad o desactiva offline."
                ),
            )
        return CacheMiss(key.kind, key.provider, key.params_hash, reason)

    # -- escritura ------------------------------------------------------------

    def write(
        self,
        kind: str,
        provider: str,
        params: Mapping[str, Any] | None,
        value: Any,
        *,
        ttl: float | timedelta | None = None,
        immutable: bool | None = None,
        tags: Iterable[str] = (),
        allow_empty: bool = False,
        note: str = "",
    ) -> CacheMeta:
        """Guarda `value` de forma atómica y devuelve sus metadatos.

        `immutable=None` delega en `ImmutabilityPolicy`. Un valor `None` o un panel
        vacío se rechazan con `DataQualityError` salvo `allow_empty=True`: cachear
        un vacío lo convierte en permanente y, meses después, es indistinguible de
        "ese trimestre no hubo datos" (contrato §0.3).
        """
        key = self.key_for(kind, provider, params)
        if not self.enabled:
            return CacheMeta(
                kind=str(kind),
                provider=str(provider),
                params_hash=key.params_hash,
                payload="json",
                written_at=self.clock.now(),
                note="caché desactivada: no se ha escrito nada",
            )
        if value is None:
            msg = (
                f"no se cachea `None` para {key}: un proveedor que no tiene el dato debe "
                "lanzar ProviderUnavailable o InsufficientHistory, no devolver None"
            )
            raise DataQualityError(msg)
        if isinstance(value, pd.DataFrame | pd.Series) and len(value) == 0 and not allow_empty:
            msg = (
                f"no se cachea un panel vacío para {key}: pásalo con allow_empty=True si "
                "de verdad el proveedor no tiene filas para esa petición"
            )
            raise DataQualityError(msg)

        immutable_flag = (
            self.immutability.classify(kind, params, today=self.clock.now().date())
            if immutable is None
            else bool(immutable)
        )
        ttl_s: float | None
        if immutable_flag:
            ttl_s = None
        elif ttl is None:
            ttl_s = self.default_ttl_s
        elif isinstance(ttl, timedelta):
            ttl_s = ttl.total_seconds()
        else:
            ttl_s = float(ttl)
        if ttl_s is not None and ttl_s < 0:
            msg = f"el TTL no puede ser negativo; recibido {ttl_s}"
            raise ConfigError(msg)

        payload, series_name, rows = _classify_payload(value)
        directory = self._dir(key)
        directory.mkdir(parents=True, exist_ok=True)
        data_path = directory / f"{key.stem}{_SUFFIX_BY_PAYLOAD[payload]}"

        with self._lock:
            n_bytes = _atomic_write(data_path, value, payload)
            meta = CacheMeta(
                kind=str(kind),
                provider=str(provider),
                params_hash=key.params_hash,
                payload=payload,
                written_at=self.clock.now(),
                ttl_s=ttl_s,
                immutable=immutable_flag,
                rows=rows,
                n_bytes=n_bytes,
                tags=tuple(str(t) for t in tags),
                params=canonicalize(dict(params or {})),
                series_name=series_name,
                note=note,
            )
            _atomic_write_text(
                self._meta_path(key),
                json.dumps(meta.to_json(), ensure_ascii=False, indent=2, sort_keys=True),
            )
            self.stats.writes += 1
            self.stats.bytes_written += n_bytes
        return meta

    # -- envoltura de proveedor ----------------------------------------------

    def fetch(
        self,
        kind: str,
        provider: str,
        params: Mapping[str, Any] | None,
        loader: Callable[[], T],
        *,
        ttl: float | timedelta | None = None,
        immutable: bool | None = None,
        tags: Iterable[str] = (),
        refresh: bool = False,
        allow_empty: bool = False,
    ) -> T:
        """Devuelve el valor cacheado o lo produce con `loader` y lo cachea.

        Es la envoltura que deben usar los adaptadores. En **modo offline**,
        `loader` no se invoca **nunca**: si no hay entrada válida se lanza
        `ProviderUnavailable`. Esa garantía es lo que hace reproducible un backtest
        ejecutado con `offline=True`.
        """
        if self.enabled and not refresh:
            try:
                return self.read(kind, provider, params)  # type: ignore[no-any-return]
            except CacheMiss:
                pass
        elif self.offline and refresh:
            key = self.key_for(kind, provider, params)
            raise ProviderUnavailable(
                str(provider),
                f"modo offline: no se puede refrescar {key} porque no se sale a la red",
            )
        if self.offline:
            # `read` ya habría lanzado ProviderUnavailable si la caché estuviera
            # activa; aquí solo se llega con la caché desactivada.
            key = self.key_for(kind, provider, params)
            raise ProviderUnavailable(
                str(provider),
                f"modo offline sin caché activa: no hay forma de obtener {key}",
            )
        value = loader()
        self.write(
            kind,
            provider,
            params,
            value,
            ttl=ttl,
            immutable=immutable,
            tags=tags,
            allow_empty=allow_empty,
        )
        return value

    @contextmanager
    def entry(
        self,
        kind: str,
        provider: str,
        params: Mapping[str, Any] | None = None,
        *,
        ttl: float | timedelta | None = None,
        immutable: bool | None = None,
        tags: Iterable[str] = (),
        allow_empty: bool = False,
        refresh: bool = False,
    ) -> Iterator[CacheSlot]:
        """Context manager para envolver una llamada de proveedor.

        Alternativa a `fetch` cuando la producción del valor no cabe en un
        `lambda` (varias peticiones, paginación, validaciones intermedias).
        """
        key = self.key_for(kind, provider, params)
        slot = CacheSlot(
            key=key,
            hit=False,
            _cache=self,
            _ttl=ttl,
            _immutable=immutable,
            _tags=tuple(str(t) for t in tags),
            _allow_empty=allow_empty,
            _params=dict(params or {}),
        )
        if not refresh:
            try:
                slot.value = self.read(kind, provider, params)
                slot.hit = True
                slot.meta = self._read_meta(key)
            except CacheMiss:
                slot.hit = False
        elif self.offline:
            raise ProviderUnavailable(
                str(provider),
                f"modo offline: no se puede refrescar {key} porque no se sale a la red",
            )
        yield slot

    # -- invalidación ---------------------------------------------------------

    def iter_meta(
        self, kind: str | None = None, provider: str | None = None
    ) -> Iterator[CacheMeta]:
        """Recorre los metadatos de las entradas, filtrando por `kind`/`provider`."""
        base = self.root
        if kind is not None:
            base = base / _safe(kind)
            if provider is not None:
                base = base / _safe(provider)
        if not base.exists():
            return
        for path in sorted(base.rglob("*.meta.json")):
            try:
                meta = CacheMeta.from_json(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError, KeyError):
                self.stats.errors += 1
                continue
            if meta.version != CACHE_FORMAT_VERSION:
                continue
            if kind is not None and meta.kind != kind:
                continue
            if provider is not None and meta.provider != provider:
                continue
            yield meta

    def invalidate(
        self,
        *,
        kind: str | None = None,
        provider: str | None = None,
        params: Mapping[str, Any] | None = None,
        tags: Iterable[str] = (),
        written_before: datetime | None = None,
        expired_only: bool = False,
        include_immutable: bool = False,
        predicate: Callable[[CacheMeta], bool] | None = None,
    ) -> int:
        """Elimina entradas selectivamente. Devuelve cuántas se han borrado.

        Los filtros se combinan con AND. Las entradas **inmutables** están
        protegidas salvo `include_immutable=True`: son el histórico ya cerrado, lo
        más caro de reconstruir, y borrarlas por un `invalidate(kind="prices")`
        destinado a refrescar la última semana sería un error caro.

        Casos de uso reales: un proveedor corrige su histórico
        (`invalidate(provider="fmp", include_immutable=True)`), se cambia el
        *parser* de un tipo (`invalidate(kind="fundamentals")`), o hay que
        recuperar espacio (`invalidate(expired_only=True)`).
        """
        if params is not None:
            if kind is None or provider is None:
                msg = "para invalidar por `params` hay que indicar también `kind` y `provider`"
                raise ConfigError(msg)
            key = self.key_for(kind, provider, params)
            meta = self._read_meta(key)
            if meta is None or (meta.immutable and not include_immutable):
                return 0
            self._remove(key, meta)
            self.stats.invalidations += 1
            return 1

        wanted_tags = {str(t) for t in tags}
        now = self.clock.now()
        removed = 0
        for meta in list(self.iter_meta(kind, provider)):
            if meta.immutable and not include_immutable:
                continue
            if wanted_tags and not wanted_tags.issubset(set(meta.tags)):
                continue
            if written_before is not None and meta.written_at >= written_before:
                continue
            if expired_only and not meta.is_expired(now):
                continue
            if predicate is not None and not predicate(meta):
                continue
            self._remove(meta.key, meta)
            removed += 1
        self.stats.invalidations += removed
        return removed

    def purge_expired(self) -> int:
        """Borra todas las entradas mutables ya caducadas."""
        return self.invalidate(expired_only=True)

    def clear(self) -> int:
        """Vacía la caché por completo, inmutables incluidos."""
        n = sum(1 for _ in self.iter_meta())
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.stats.invalidations += n
        return n

    def size_bytes(self) -> int:
        """Bytes ocupados por los ficheros de datos (sin contar metadatos)."""
        return sum(m.n_bytes for m in self.iter_meta())

    def _remove(self, key: CacheKey, meta: CacheMeta | None) -> None:
        directory = self._dir(key)
        candidates = (
            [directory / f"{key.stem}{meta.suffix}"]
            if meta is not None
            else [directory / f"{key.stem}{suf}" for suf in set(_SUFFIX_BY_PAYLOAD.values())]
        )
        for path in [*candidates, self._meta_path(key)]:
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:  # pragma: no cover - permisos
                logger.warning("no se pudo borrar %s: %s", path, exc)

    def __repr__(self) -> str:  # pragma: no cover - cosmético
        return (
            f"DiskCache(root={str(self.root)!r}, offline={self.offline}, "
            f"ttl={self.default_ttl_s:g}s, enabled={self.enabled})"
        )


# ===========================================================================
# 6. Serialización de la carga útil
# ===========================================================================

_SERIES_COL = "__value__"


def _classify_payload(value: Any) -> tuple[str, str | None, int | None]:
    """Devuelve `(formato, nombre_de_serie, filas)` para `value`."""
    if isinstance(value, pd.DataFrame):
        return "dataframe", None, len(value)
    if isinstance(value, pd.Series):
        name = None if value.name is None else str(value.name)
        return "series", name, len(value)
    if isinstance(value, bytes | bytearray):
        return "bytes", None, None
    return "json", None, (len(value) if isinstance(value, list | dict | tuple) else None)


def _atomic_write(path: Path, value: Any, payload: str) -> int:
    """Escribe la carga útil de forma atómica y devuelve su tamaño en bytes.

    Se escribe a un temporal y se hace `os.replace`, que es atómico dentro del
    mismo sistema de ficheros. Sin esto, un proceso interrumpido a mitad de
    escritura dejaría un parquet truncado que la siguiente ejecución leería como
    "datos"; y un fallo silencioso en la caché es un fallo silencioso en el panel.
    """
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    try:
        if payload == "dataframe":
            _write_parquet(value, tmp)
        elif payload == "series":
            frame = value.to_frame(name=_SERIES_COL if value.name is None else value.name)
            _write_parquet(frame, tmp)
        elif payload == "bytes":
            tmp.write_bytes(bytes(value))
        else:
            tmp.write_text(
                json.dumps(
                    {"v": CACHE_FORMAT_VERSION, "value": canonicalize(value)},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        size = tmp.stat().st_size
        os.replace(tmp, path)
        return int(size)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    try:
        frame.to_parquet(path, engine="pyarrow", index=True)
    except Exception as exc:  # pyarrow lanza tipos de excepción propios
        msg = (
            f"el DataFrame no es serializable a parquet ({type(exc).__name__}: {exc}). "
            "Suele deberse a una columna `object` con tipos mezclados; normaliza los "
            "dtypes antes de cachear."
        )
        raise DataQualityError(msg) from exc


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _load_payload(path: Path, meta: CacheMeta) -> Any:
    if not path.exists():
        msg = f"falta el fichero de datos {path.name}"
        raise ValueError(msg)
    if meta.payload == "dataframe":
        return pd.read_parquet(path, engine="pyarrow")
    if meta.payload == "series":
        frame = pd.read_parquet(path, engine="pyarrow")
        if frame.shape[1] != 1:  # pragma: no cover - defensivo
            msg = f"se esperaba una única columna para una Series; hay {frame.shape[1]}"
            raise ValueError(msg)
        series = frame.iloc[:, 0]
        return series.rename(meta.series_name)
    if meta.payload == "bytes":
        return path.read_bytes()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "value" not in raw:
        msg = "el JSON cacheado no tiene la envoltura esperada"
        raise ValueError(msg)
    return _decanonicalize(raw["value"])


def _decanonicalize(obj: Any) -> Any:
    """Inversa de `canonicalize` para las etiquetas de tipo reversibles.

    Las secuencias resumidas por digest (`__seq__`) **no** son reversibles, pero
    solo aparecen en `params`, que es informativo; los valores cacheados se
    serializan enteros.
    """
    if isinstance(obj, list):
        return [_decanonicalize(v) for v in obj]
    if not isinstance(obj, dict):
        return obj
    if len(obj) == 1:
        tag, payload = next(iter(obj.items()))
        if tag == "__date__":
            return date.fromisoformat(payload)
        if tag == "__datetime__":
            return datetime.fromisoformat(payload)
        if tag == "__timedelta__":
            return timedelta(seconds=float(payload))
        if tag == "__bytes__":
            return base64.b64decode(payload)
        if tag == "__path__":
            return Path(payload)
        if tag == "__tuple__":
            if isinstance(payload, list):
                return tuple(_decanonicalize(v) for v in payload)
            return payload
        if tag == "__set__":
            if isinstance(payload, list):
                return {_decanonicalize(v) for v in payload}
            return payload
        if tag == "__float__":
            return float(payload)
    return {k: _decanonicalize(v) for k, v in obj.items()}


# ===========================================================================
# 7. Decorador
# ===========================================================================


def cached(
    cache: DiskCache | str | Callable[..., DiskCache],
    kind: str,
    provider: str | None = None,
    *,
    ttl: float | timedelta | None = None,
    immutable: bool | None = None,
    tags: Iterable[str] = (),
    ignore: Sequence[str] = (),
    key_fn: Callable[[dict[str, Any]], Mapping[str, Any]] | None = None,
    allow_empty: bool = False,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorador que envuelve una llamada de proveedor con la caché.

    Los parámetros de la clave son los argumentos de la función (con sus valores
    por defecto aplicados, de modo que llamarla posicional o por nombre da la
    **misma** clave). `self` se excluye siempre; `ignore` permite excluir más
    (un cliente HTTP, un flag de verbosidad… cualquier cosa que no cambie el dato).

    `cache` admite tres formas para poder decorar también métodos:

    - una instancia `DiskCache`,
    - el **nombre de un atributo** de `self` (`cached("cache", ...)`),
    - un invocable que recibe `self` (o nada, en funciones sueltas).

    `provider=None` toma `self.name`, que es lo natural en un `BaseProvider`.

    Ejemplo::

        class PolygonPrices(BaseProvider):
            name = "polygon"

            @cached("cache", DataKind.PRICES, ttl=3600)
            def daily_bars(self, ticker: str, start: date, end: date) -> pd.DataFrame:
                ...
    """
    import inspect

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        sig = inspect.signature(fn)
        skip = {"self", "cls", *ignore}

        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            instance = bound.arguments.get("self")
            params: dict[str, Any] = {
                k: v for k, v in bound.arguments.items() if k not in skip
            }
            if key_fn is not None:
                params = dict(key_fn(params))

            store = _resolve_cache(cache, instance)
            prov = provider or getattr(instance, "name", None) or fn.__module__.rsplit(".", 1)[-1]
            return store.fetch(
                kind,
                str(prov),
                params,
                lambda: fn(*args, **kwargs),
                ttl=ttl,
                immutable=immutable,
                tags=tags,
                allow_empty=allow_empty,
            )

        wrapper.__wrapped__ = fn  # type: ignore[attr-defined]
        return wrapper

    return decorator


def _resolve_cache(
    cache: DiskCache | str | Callable[..., DiskCache], instance: Any
) -> DiskCache:
    if isinstance(cache, DiskCache):
        return cache
    if isinstance(cache, str):
        if instance is None:
            msg = f"`cached({cache!r})` solo vale en métodos: no hay `self` de donde leerlo"
            raise ConfigError(msg)
        store = getattr(instance, cache, None)
        if not isinstance(store, DiskCache):
            msg = f"el atributo {cache!r} de {type(instance).__name__} no es un DiskCache"
            raise ConfigError(msg)
        return store
    if callable(cache):
        store = cache(instance) if instance is not None else cache()
        if not isinstance(store, DiskCache):
            msg = "el invocable de `cached` no devolvió un DiskCache"
            raise ConfigError(msg)
        return store
    msg = f"`cache` debe ser DiskCache, nombre de atributo o invocable; es {type(cache).__name__}"
    raise ConfigError(msg)
