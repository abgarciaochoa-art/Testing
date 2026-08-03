"""Configuración global: rutas, credenciales y política de caché.

Las credenciales se leen exclusivamente de variables de entorno. El repo no contiene
ni debe contener claves.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from earnings_alpha.errors import ConfigError

__all__ = ["Settings", "get_settings", "PROVIDER_ENV_KEYS"]

# Variable de entorno que aporta la credencial de cada proveedor.
PROVIDER_ENV_KEYS: dict[str, list[str]] = {
    "polygon": ["POLYGON_API_KEY"],
    "fmp": ["FMP_API_KEY"],
    "eodhd": ["EODHD_API_KEY"],
    "finnhub": ["FINNHUB_API_KEY"],
    "tiingo": ["TIINGO_API_KEY"],
    "alphavantage": ["ALPHAVANTAGE_API_KEY"],
    "alpaca": ["ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY"],
    "tradier": ["TRADIER_ACCESS_TOKEN"],
    "orats": ["ORATS_TOKEN"],
    "nasdaq_data_link": ["NASDAQ_DATA_LINK_API_KEY"],
    # SEC EDGAR no usa clave, pero exige un User-Agent identificable con email.
    "sec": ["SEC_USER_AGENT"],
    # yfinance y stooq no requieren credencial.
    "yfinance": [],
    "stooq": [],
    "synthetic": [],
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@dataclass(frozen=True, slots=True)
class Settings:
    """Configuración efectiva del proceso."""

    repo_root: Path = field(default_factory=_repo_root)
    data_dir: Path = field(default_factory=lambda: _repo_root() / "data")
    cache_dir: Path = field(default_factory=lambda: _repo_root() / "data" / "cache")
    seed_dir: Path = field(default_factory=lambda: _repo_root() / "data" / "seed")

    offline: bool = False
    """Si es True, la caché nunca sale a la red: un fallo de caché es un error.
    Útil para garantizar reproducibilidad de un backtest ya ejecutado."""

    cache_ttl_days: int = 1
    """Caducidad por defecto de las entradas de caché mutables (precios recientes,
    calendarios). Los datos históricos inmutables se cachean sin caducidad."""

    max_requests_per_second: float = 8.0
    """Techo global de tasa. SEC EDGAR exige <=10 req/s; se deja margen."""

    request_timeout_s: float = 30.0
    max_retries: int = 4
    seed: int = 20260803
    """Semilla por defecto para todo componente estocástico."""

    def env(self, name: str) -> str | None:
        return os.environ.get(name)

    def credentials_for(self, provider: str) -> dict[str, str]:
        """Devuelve las credenciales del proveedor.

        Lanza `ConfigError` si el proveedor es desconocido, y devuelve un dict
        incompleto si faltan variables: es el proveedor quien decide si puede operar
        de forma degradada o debe lanzar `ProviderUnavailable`.
        """
        if provider not in PROVIDER_ENV_KEYS:
            msg = f"proveedor desconocido: {provider!r}"
            raise ConfigError(msg)
        out: dict[str, str] = {}
        for key in PROVIDER_ENV_KEYS[provider]:
            val = os.environ.get(key)
            if val:
                out[key] = val
        return out

    def missing_credentials_for(self, provider: str) -> list[str]:
        """Variables de entorno que el proveedor necesita y no están definidas."""
        if provider not in PROVIDER_ENV_KEYS:
            msg = f"proveedor desconocido: {provider!r}"
            raise ConfigError(msg)
        return [k for k in PROVIDER_ENV_KEYS[provider] if not os.environ.get(k)]

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.cache_dir, self.seed_dir):
            d.mkdir(parents=True, exist_ok=True)


_SETTINGS: Settings | None = None


def get_settings(**overrides: object) -> Settings:
    """Devuelve la configuración del proceso.

    Con `overrides` construye una instancia nueva sin tocar la global, lo que permite
    a los tests aislarse por completo.
    """
    global _SETTINGS
    if overrides:
        base = Settings()
        merged = {f: getattr(base, f) for f in base.__slots__}
        merged.update(overrides)  # type: ignore[arg-type]
        return Settings(**merged)  # type: ignore[arg-type]
    if _SETTINGS is None:
        env_offline = os.environ.get("EARNINGS_ALPHA_OFFLINE", "").lower() in {"1", "true", "yes"}
        _SETTINGS = Settings(offline=env_offline)
    return _SETTINGS
