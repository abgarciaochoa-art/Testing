"""Jerarquía de errores.

El principio del repo es el fallo explícito: un proveedor sin credenciales, un
histórico insuficiente o una violación point-in-time deben lanzar, nunca degradarse
a datos vacíos o valores inventados que contaminen un backtest en silencio.
"""

from __future__ import annotations

__all__ = [
    "EarningsAlphaError",
    "ProviderUnavailable",
    "RateLimited",
    "InsufficientHistory",
    "LookAheadError",
    "UniverseError",
    "CalendarError",
    "DataQualityError",
    "ConfigError",
]


class EarningsAlphaError(Exception):
    """Base de todos los errores de la plataforma."""


class ProviderUnavailable(EarningsAlphaError):
    """Un proveedor de datos no puede servir la petición.

    Se usa tanto para falta de credenciales como para falta de conectividad; el
    mensaje debe distinguir ambos casos para que el usuario sepa qué arreglar.
    """

    def __init__(self, provider: str, reason: str, *, missing_env: list[str] | None = None) -> None:
        self.provider = provider
        self.reason = reason
        self.missing_env = missing_env or []
        detail = f"proveedor {provider!r} no disponible: {reason}"
        if self.missing_env:
            detail += f" (faltan variables de entorno: {', '.join(self.missing_env)})"
        super().__init__(detail)


class RateLimited(EarningsAlphaError):
    """El proveedor ha aplicado limitación de tasa."""

    def __init__(self, provider: str, retry_after: float | None = None) -> None:
        self.provider = provider
        self.retry_after = retry_after
        super().__init__(
            f"{provider} ha limitado la tasa"
            + (f"; reintentar en {retry_after:.1f}s" if retry_after else "")
        )


class InsufficientHistory(EarningsAlphaError):
    """No hay histórico suficiente para calcular lo solicitado.

    Ejemplo típico: un SUE necesita 8 trimestres de sorpresas previas; con menos, el
    valor correcto es NaN y no un número calculado sobre 2 observaciones.
    """


class LookAheadError(EarningsAlphaError):
    """Se ha detectado uso de información no disponible en la fecha de la señal.

    Es un error de programación, no una condición de datos: indica que algún join o
    desplazamiento temporal está mal construido.
    """


class UniverseError(EarningsAlphaError):
    """Problema al resolver la pertenencia al índice."""


class CalendarError(EarningsAlphaError):
    """Fecha fuera del rango del calendario o sesión inexistente."""


class DataQualityError(EarningsAlphaError):
    """Los datos recibidos no superan las validaciones de calidad."""


class ConfigError(EarningsAlphaError):
    """Configuración ausente o inconsistente."""
