"""Tests del núcleo de la capa de datos (`earnings_alpha.data.base` y `.cache`).

**Ninguno de estos tests abre un socket.** Todo el tráfico va contra
`ScriptedTransport`, y el tiempo se mide con `ManualClock` / `ManualWallClock`, de
modo que las afirmaciones sobre limitación de tasa, backoff y caducidad son
exactas y no dependen de la carga de la máquina: un test que comprueba un backoff
de 8 segundos durmiendo 8 segundos es un test que nadie ejecuta.

Cobertura:

1. Cubo de fichas: espaciado, ráfaga, recarga, atomicidad entre hilos.
2. Política de reintentos: backoff exponencial, jitter, `Retry-After`.
3. `HttpClient`: 200 / 429 / 401 / 403 / 404 / 5xx / timeout, traducción a los
   errores del contrato, tasa efectiva y cabeceras.
4. `ProviderRegistry`: prioridad, mensaje de indisponibilidad con las variables de
   entorno que faltan, fallback en cadena y cuarentena.
5. `DiskCache`: ida y vuelta de un panel con MultiIndex, TTL, inmutabilidad,
   invalidación selectiva, corrupción y modo offline.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from earnings_alpha.config import get_settings
from earnings_alpha.data.base import (
    BaseProvider,
    DataKind,
    HttpClient,
    HttpRequest,
    HttpResponse,
    HttpStatusError,
    ManualClock,
    ProviderRegistry,
    ProviderStatus,
    RateLimitPolicy,
    RetryPolicy,
    ScriptedTransport,
    StaticProvider,
    TokenBucket,
    TransportError,
    TransportTimeout,
    get_registry,
    parse_retry_after,
    rate_limit_for,
    set_registry,
)
from earnings_alpha.data.cache import (
    CACHE_FORMAT_VERSION,
    CacheMiss,
    DiskCache,
    ImmutabilityPolicy,
    ManualWallClock,
    cached,
    canonicalize,
    params_hash,
)
from earnings_alpha.errors import (
    ConfigError,
    DataQualityError,
    ProviderUnavailable,
    RateLimited,
)

# ---------------------------------------------------------------------------
# Utilidades comunes
# ---------------------------------------------------------------------------

FAST_RATE = RateLimitPolicy(1_000_000.0, burst=1000, note="sin limitación en el test")
"""Política que desactiva de hecho el cubo de fichas, para los tests que miden
backoff y no tasa: mezclar ambas esperas haría ambiguo el resultado."""


def make_panel(n_dates: int = 3, tickers: tuple[str, ...] = ("AAPL", "BRK.B")) -> pd.DataFrame:
    """Panel canónico `(date, ticker)` como los que circulan por el repositorio."""
    dates = pd.date_range("2024-01-02", periods=n_dates, freq="B").as_unit("ns")
    index = pd.MultiIndex.from_product([dates, list(tickers)], names=["date", "ticker"])
    rng = np.random.default_rng(20260803)
    n = len(index)
    return pd.DataFrame(
        {
            "close": rng.normal(100, 5, n).round(4),
            "volume": rng.integers(1_000, 10_000, n).astype("int64"),
            "adj_close": rng.normal(100, 5, n).round(4),
        },
        index=index,
    ).sort_index()


def client(
    script: Any,
    *,
    provider: str = "polygon",
    retry: RetryPolicy | None = None,
    rate: RateLimitPolicy | None = None,
    repeat_last: bool = False,
    seed: int = 7,
) -> tuple[HttpClient, ScriptedTransport, ManualClock]:
    """Cliente HTTP totalmente instrumentado y sin red."""
    clock = ManualClock()
    transport = ScriptedTransport(script, repeat_last=repeat_last, clock=clock)
    http = HttpClient(
        provider,
        transport=transport,
        clock=clock,
        rate=rate or FAST_RATE,
        retry=retry or RetryPolicy(max_retries=3, backoff_base_s=1.0, jitter="none"),
        seed=seed,
    )
    return http, transport, clock


# ===========================================================================
# 1. Cubo de fichas
# ===========================================================================


class TestTokenBucket:
    def test_espaciado_uniforme_con_burst_1(self) -> None:
        """Con `burst=1` las peticiones salen a intervalos exactos de 1/tasa."""
        clock = ManualClock()
        bucket = TokenBucket(10.0, 1.0, clock=clock)
        stamps = []
        for _ in range(25):
            bucket.acquire()
            stamps.append(clock.time())
        deltas = np.diff(stamps)
        assert stamps[0] == 0.0
        assert np.allclose(deltas, 0.1)

    def test_nunca_supera_n_por_segundo(self) -> None:
        """La propiedad que de verdad importa: ninguna ventana de 1 s excede la tasa.

        Es la comprobación que protege la cuota de la SEC (10 req/s agregadas): un
        espaciado medio correcto no basta si el cliente concentra ráfagas.
        """
        clock = ManualClock()
        bucket = TokenBucket(10.0, 1.0, clock=clock)
        stamps = []
        for _ in range(60):
            bucket.acquire()
            stamps.append(clock.time())
        peor = max(sum(1 for t in stamps if x <= t < x + 1.0) for x in stamps)
        assert peor <= 10

    def test_burst_inicial_y_luego_estrangulamiento(self) -> None:
        """Un cubo lleno admite `burst` peticiones inmediatas y luego regula."""
        clock = ManualClock()
        bucket = TokenBucket(5.0, 5.0, clock=clock)
        for _ in range(5):
            assert bucket.acquire() == 0.0
        assert clock.time() == 0.0
        assert bucket.acquire() == pytest.approx(0.2)
        assert clock.time() == pytest.approx(0.2)

    def test_recarga_no_supera_la_capacidad(self) -> None:
        """Tras una pausa larga el cubo se llena, pero no acumula crédito infinito."""
        clock = ManualClock()
        bucket = TokenBucket(2.0, 3.0, clock=clock)
        bucket.acquire()
        clock.sleep(1_000.0)
        assert bucket.tokens == pytest.approx(3.0)
        for _ in range(3):
            assert bucket.acquire() == 0.0
        assert bucket.acquire() > 0.0

    def test_ventana_deslizante_con_burst(self) -> None:
        """Con ráfaga, el techo por ventana es `tasa + burst`, no más."""
        clock = ManualClock()
        bucket = TokenBucket(4.0, 4.0, clock=clock)
        stamps = []
        for _ in range(40):
            bucket.acquire()
            stamps.append(clock.time())
        peor = max(sum(1 for t in stamps if x <= t < x + 1.0) for x in stamps)
        assert peor <= 4 + 4

    def test_reserva_atomica_entre_hilos(self) -> None:
        """20 hilos reservando a la vez reparten huecos distintos, sin doble gasto.

        Se prueba `reserve` (que no duerme) para que el resultado sea determinista:
        si el cerrojo fallara, dos hilos obtendrían la misma espera y el cliente
        emitiría dos peticiones en el mismo instante.
        """
        clock = ManualClock()
        bucket = TokenBucket(10.0, 5.0, clock=clock)
        esperas: list[float] = []
        lock = threading.Lock()

        def worker() -> None:
            w = bucket.reserve(1.0)
            with lock:
                esperas.append(round(w, 6))

        hilos = [threading.Thread(target=worker) for _ in range(20)]
        for h in hilos:
            h.start()
        for h in hilos:
            h.join()

        esperadas = [0.0] * 5 + [round(k / 10.0, 6) for k in range(1, 16)]
        assert sorted(esperas) == sorted(esperadas)

    def test_parametros_invalidos(self) -> None:
        with pytest.raises(ConfigError):
            TokenBucket(0.0)
        with pytest.raises(ConfigError):
            RateLimitPolicy(-1.0)
        with pytest.raises(ConfigError):
            RateLimitPolicy(1.0, burst=0)


class TestRateLimitTable:
    def test_sec_por_debajo_del_limite_publicado(self) -> None:
        """La SEC publica 10 req/s; el repositorio debe quedarse por debajo."""
        assert PROVIDER_SEC.requests_per_second < 10.0

    def test_proveedor_desconocido_usa_el_techo_global(self) -> None:
        cfg = get_settings(max_requests_per_second=3.0)
        policy = rate_limit_for("un_proveedor_inventado", cfg)
        assert policy.requests_per_second == 3.0

    def test_override_por_variable_de_entorno(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Contratar un plan de pago no debe exigir tocar código."""
        monkeypatch.setenv("EARNINGS_ALPHA_RPS_POLYGON", "50")
        assert rate_limit_for("polygon").requests_per_second == 50.0
        monkeypatch.setenv("EARNINGS_ALPHA_RPS_POLYGON", "no-es-un-numero")
        with pytest.raises(ConfigError):
            rate_limit_for("polygon")


PROVIDER_SEC = rate_limit_for("sec", get_settings())


# ===========================================================================
# 2. Reintentos
# ===========================================================================


class TestRetryPolicy:
    def test_backoff_exponencial_con_tope(self) -> None:
        policy = RetryPolicy(backoff_base_s=0.5, backoff_factor=2.0, backoff_max_s=4.0)
        secuencia = [policy.deterministic_delay(i) for i in range(6)]
        assert secuencia == [0.5, 1.0, 2.0, 4.0, 4.0, 4.0]

    def test_jitter_full_dentro_del_intervalo(self) -> None:
        import random

        policy = RetryPolicy(backoff_base_s=1.0, jitter="full")
        rng = random.Random(0)
        muestras = [policy.delay_for(2, rng) for _ in range(200)]
        assert all(0.0 <= d <= 4.0 for d in muestras)
        assert min(muestras) < 1.0 < max(muestras)

    def test_jitter_equal_garantiza_un_minimo(self) -> None:
        import random

        policy = RetryPolicy(backoff_base_s=1.0, jitter="equal")
        rng = random.Random(0)
        muestras = [policy.delay_for(1, rng) for _ in range(200)]
        assert all(1.0 <= d <= 2.0 for d in muestras)

    def test_jitter_desconocido(self) -> None:
        with pytest.raises(ConfigError):
            RetryPolicy(jitter="gaussiano")

    @pytest.mark.parametrize(
        ("cabecera", "esperado"),
        [("5", 5.0), ("0", 0.0), (" 12 ", 12.0), (None, None), ("", None), ("mañana", None)],
    )
    def test_parse_retry_after_numerico(self, cabecera: str | None, esperado: float | None) -> None:
        assert parse_retry_after(cabecera) == esperado

    def test_parse_retry_after_fecha_http(self) -> None:
        """RFC 9110 admite `Retry-After` como fecha; hay proveedores que la usan."""
        ahora = datetime(2026, 8, 4, 12, 0, 0, tzinfo=UTC)
        valor = "Tue, 04 Aug 2026 12:00:30 GMT"
        assert parse_retry_after(valor, now=ahora) == pytest.approx(30.0)
        # Una fecha ya pasada no debe producir esperas negativas.
        assert parse_retry_after("Tue, 04 Aug 2026 11:00:00 GMT", now=ahora) == 0.0


# ===========================================================================
# 3. Cliente HTTP
# ===========================================================================


class TestHttpClientBasico:
    def test_200_devuelve_cuerpo_y_cuenta_estadisticas(self) -> None:
        http, transport, _ = client([HttpResponse(200, b'{"a": 1}', {"Content-Type": "json"})])
        assert http.get_json("https://api.test/x") == {"a": 1}
        assert transport.n_calls == 1
        assert http.stats.requests == 1
        assert http.stats.successes == 1
        assert http.stats.retries == 0
        assert http.stats.by_status == {200: 1}

    def test_user_agent_y_parametros_llegan_al_transporte(self) -> None:
        http, transport, _ = client([HttpResponse(200, b"ok")])
        http.get("https://api.test/x", params={"symbol": "AAPL"}, headers={"Accept": "text/csv"})
        enviado = transport.calls[0]
        assert enviado.method == "GET"
        assert enviado.params == {"symbol": "AAPL"}
        assert enviado.headers is not None
        assert "earnings-alpha" in enviado.headers["User-Agent"]
        assert enviado.headers["Accept"] == "text/csv"

    def test_user_agent_configurable(self) -> None:
        clock = ManualClock()
        transport = ScriptedTransport([HttpResponse(200, b"ok")], clock=clock)
        http = HttpClient(
            "sec",
            transport=transport,
            clock=clock,
            rate=FAST_RATE,
            user_agent="Ana Pérez ana@example.com",
        )
        http.get("https://data.sec.gov/x")
        assert transport.calls[0].headers["User-Agent"] == "Ana Pérez ana@example.com"

    def test_json_invalido_es_error_de_calidad(self) -> None:
        http, _, _ = client([HttpResponse(200, b"<html>not json</html>")])
        with pytest.raises(DataQualityError):
            http.get_json("https://api.test/x")

    def test_timeout_por_peticion_se_propaga(self) -> None:
        http, transport, _ = client([HttpResponse(200, b"ok")])
        http.get("https://api.test/x", timeout=2.5)
        assert transport.calls[0].timeout == 2.5


class TestHttpClientErrores:
    def test_429_con_retry_after_se_respeta_exactamente(self) -> None:
        """`Retry-After` manda sobre el backoff local (RFC 9110 §10.2.3)."""
        http, transport, clock = client(
            [
                HttpResponse(429, b"slow down", {"Retry-After": "7"}),
                HttpResponse(200, b"ok"),
            ]
        )
        assert http.get("https://api.test/x").text == "ok"
        assert clock.sleeps == [7.0]
        assert transport.n_calls == 2
        assert http.stats.rate_limited == 1

    def test_retry_after_nunca_acorta_el_backoff(self) -> None:
        """Un `Retry-After: 0` no debe convertir el reintento en una tormenta."""
        http, _, clock = client(
            [HttpResponse(429, b"", {"Retry-After": "0"}), HttpResponse(200, b"ok")],
            retry=RetryPolicy(max_retries=2, backoff_base_s=1.5, jitter="none"),
        )
        http.get("https://api.test/x")
        assert clock.sleeps == [1.5]

    def test_429_persistente_lanza_rate_limited(self) -> None:
        http, transport, _ = client(
            HttpResponse(429, b"", {"Retry-After": "3"}),
            repeat_last=True,
            retry=RetryPolicy(max_retries=2, backoff_base_s=1.0, jitter="none"),
        )
        with pytest.raises(RateLimited) as exc:
            http.get("https://api.test/x")
        assert exc.value.retry_after == 3.0
        assert exc.value.provider == "polygon"
        assert transport.n_calls == 3  # intento inicial + 2 reintentos

    def test_403_no_reintenta_y_dice_que_falta(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Reintentar un 403 no lo arregla y, con la SEC, acelera el bloqueo de IP."""
        monkeypatch.delenv("SEC_USER_AGENT", raising=False)
        http, transport, clock = client(
            HttpResponse(403, b"Forbidden"), provider="sec", repeat_last=True
        )
        with pytest.raises(ProviderUnavailable) as exc:
            http.get("https://data.sec.gov/x")
        assert transport.n_calls == 1
        assert clock.sleeps == []
        assert "SEC_USER_AGENT" in str(exc.value)
        assert exc.value.missing_env == ["SEC_USER_AGENT"]

    def test_401_es_proveedor_no_disponible(self) -> None:
        http, transport, _ = client(HttpResponse(401, b"unauthorized"), repeat_last=True)
        with pytest.raises(ProviderUnavailable) as exc:
            http.get("https://api.test/x")
        assert "401" in str(exc.value)
        assert transport.n_calls == 1

    def test_404_es_httpstatuserror_salvo_que_se_permita(self) -> None:
        """El 404 lo interpreta el adaptador: en EDGAR suele ser un dato legítimo."""
        http, _, _ = client([HttpResponse(404, b"no such concept")])
        with pytest.raises(HttpStatusError) as exc:
            http.get("https://data.sec.gov/x")
        assert exc.value.status_code == 404

        http2, _, _ = client([HttpResponse(404, b"no such concept")])
        resp = http2.get("https://data.sec.gov/x", allow_status={404})
        assert resp.status_code == 404
        assert resp.text == "no such concept"

    def test_5xx_reintenta_y_luego_declara_no_disponible(self) -> None:
        http, transport, clock = client(
            HttpResponse(503, b"maintenance"),
            repeat_last=True,
            retry=RetryPolicy(max_retries=3, backoff_base_s=1.0, backoff_factor=2.0, jitter="none"),
        )
        with pytest.raises(ProviderUnavailable) as exc:
            http.get("https://api.test/x")
        assert transport.n_calls == 4
        assert clock.sleeps == [1.0, 2.0, 4.0]
        assert "503" in str(exc.value)

    def test_5xx_transitorio_acaba_en_exito(self) -> None:
        http, transport, clock = client(
            [
                HttpResponse(500, b"boom"),
                HttpResponse(502, b"boom"),
                HttpResponse(200, b"por fin"),
            ]
        )
        assert http.get("https://api.test/x").text == "por fin"
        assert transport.n_calls == 3
        assert clock.sleeps == [1.0, 2.0]
        assert http.stats.retries == 2

    def test_retry_after_desmedido_no_bloquea_el_proceso(self) -> None:
        """Un `Retry-After: 3600` no puede colgar un backtest interactivo."""
        http, transport, clock = client(
            HttpResponse(503, b"", {"Retry-After": "3600"}),
            repeat_last=True,
            retry=RetryPolicy(max_retries=3, backoff_base_s=1.0, max_retry_after_s=120.0),
        )
        with pytest.raises(ProviderUnavailable):
            http.get("https://api.test/x")
        assert transport.n_calls == 1
        assert clock.sleeps == []

    def test_timeout_reintenta_y_traduce_a_proveedor_no_disponible(self) -> None:
        http, transport, clock = client(
            TransportTimeout("polygon", "timeout tras 30.0s"),
            repeat_last=True,
            retry=RetryPolicy(max_retries=2, backoff_base_s=0.5, jitter="none"),
        )
        with pytest.raises(ProviderUnavailable) as exc:
            http.get("https://api.test/x")
        assert transport.n_calls == 3
        assert clock.sleeps == [0.5, 1.0]
        assert "3 intentos" in str(exc.value)
        assert http.stats.transport_errors == 3

    def test_timeout_transitorio_se_recupera(self) -> None:
        http, transport, _ = client(
            [TransportTimeout("polygon", "timeout"), HttpResponse(200, b"ok")]
        )
        assert http.get("https://api.test/x").text == "ok"
        assert transport.n_calls == 2

    def test_error_de_red_generico(self) -> None:
        http, _, _ = client(
            TransportError("polygon", "DNS no resuelve"),
            repeat_last=True,
            retry=RetryPolicy(max_retries=1, backoff_base_s=0.1, jitter="none"),
        )
        with pytest.raises(ProviderUnavailable) as exc:
            http.get("https://api.test/x")
        assert "sin conectividad" in str(exc.value)


class TestHttpClientTasaYJitter:
    def test_el_cliente_respeta_su_propia_tasa(self) -> None:
        """10 peticiones a 5 req/s con ráfaga 1 ocupan al menos 1,8 s simulados."""
        http, transport, clock = client(
            HttpResponse(200, b"ok"),
            repeat_last=True,
            rate=RateLimitPolicy(5.0, burst=1),
        )
        for _ in range(10):
            http.get("https://api.test/x")
        assert clock.time() == pytest.approx(1.8)
        instantes = [c.at for c in transport.calls]
        peor = max(sum(1 for t in instantes if x <= t < x + 1.0) for x in instantes)
        assert peor <= 5
        assert http.stats.throttled_s == pytest.approx(1.8)

    def test_el_backoff_no_se_confunde_con_la_tasa(self) -> None:
        """Las esperas por tasa y por backoff se contabilizan por separado.

        Y no se suman de más: mientras se espera el backoff, el cubo se recarga, de
        modo que el reintento no paga *otra vez* el peaje de la tasa.
        """
        http, _, clock = client(
            [HttpResponse(500, b""), HttpResponse(200, b"ok")],
            rate=RateLimitPolicy(10.0, burst=1),
            retry=RetryPolicy(max_retries=2, backoff_base_s=2.0, jitter="none"),
        )
        http.get("https://api.test/x")
        assert clock.sleeps == [2.0]
        assert http.stats.throttled_s == 0.0
        assert http.stats.slept_s == pytest.approx(2.0)

    def test_la_tasa_se_contabiliza_cuando_de_verdad_estrangula(self) -> None:
        http, _, _ = client(
            HttpResponse(200, b"ok"),
            repeat_last=True,
            rate=RateLimitPolicy(10.0, burst=1),
        )
        for _ in range(3):
            http.get("https://api.test/x")
        assert http.stats.throttled_s == pytest.approx(0.2)
        assert http.stats.retries == 0

    def test_jitter_es_determinista_con_la_misma_semilla(self) -> None:
        """La reproducibilidad exige que hasta el jitter sea determinista."""

        def secuencia(seed: int) -> list[float]:
            http, _, clock = client(
                HttpResponse(500, b""),
                repeat_last=True,
                retry=RetryPolicy(max_retries=3, backoff_base_s=1.0, jitter="full"),
                seed=seed,
            )
            with pytest.raises(ProviderUnavailable):
                http.get("https://api.test/x")
            return clock.sleeps

        a, b, c = secuencia(11), secuencia(11), secuencia(12)
        assert a == b
        assert a != c
        assert all(0.0 <= d <= 4.0 for d in a)


class TestHttpClientConstruccion:
    def test_for_provider_exige_credenciales(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("POLYGON_API_KEY", raising=False)
        with pytest.raises(ProviderUnavailable) as exc:
            HttpClient.for_provider("polygon", settings=get_settings())
        assert exc.value.missing_env == ["POLYGON_API_KEY"]
        assert "POLYGON_API_KEY" in str(exc.value)

    def test_for_provider_con_credenciales(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("POLYGON_API_KEY", "clave-de-prueba")
        http = HttpClient.for_provider(
            "polygon", settings=get_settings(), transport=ScriptedTransport(HttpResponse(200))
        )
        assert http.provider == "polygon"
        assert http.rate.requests_per_second == rate_limit_for("polygon").requests_per_second

    def test_for_provider_sec_toma_el_user_agent_del_entorno(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SEC_USER_AGENT", "Investigación ana@example.com")
        transport = ScriptedTransport([HttpResponse(200, b"ok")])
        http = HttpClient.for_provider("sec", settings=get_settings(), transport=transport)
        http.get("https://data.sec.gov/x")
        assert transport.calls[0].headers["User-Agent"] == "Investigación ana@example.com"

    def test_puede_saltarse_la_verificacion_de_credenciales(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("POLYGON_API_KEY", raising=False)
        http = HttpClient.for_provider(
            "polygon", settings=get_settings(), require_credentials=False
        )
        assert http.provider == "polygon"

    def test_guion_agotado_es_error_de_configuracion_del_test(self) -> None:
        """Salvaguarda del propio arnés: si el guion se queda corto, se ve."""
        http, _, _ = client([HttpResponse(200, b"ok")])
        http.get("https://api.test/x")
        with pytest.raises(ConfigError):
            http.get("https://api.test/x")

    def test_transporte_dependiente_de_la_peticion(self) -> None:
        """Un elemento invocable permite simular paginación sin red."""

        def responder(req: HttpRequest) -> HttpResponse:
            pagina = (req.params or {}).get("page", 1)
            cuerpo = json.dumps({"page": pagina, "next": pagina < 2}).encode()
            return HttpResponse(200, cuerpo)

        http, _, _ = client(responder, repeat_last=True)
        assert http.get_json("https://api.test/x", params={"page": 1})["next"] is True
        assert http.get_json("https://api.test/x", params={"page": 2})["next"] is False


# ===========================================================================
# 4. Proveedores y registro
# ===========================================================================


class SondeoProvider(BaseProvider):
    """Proveedor con sondeo de conectividad, para probar la memoización."""

    name = "polygon"
    kinds = ("prices",)
    probe_url = "https://api.polygon.io/v3/reference/tickers"


class TestBaseProvider:
    def test_credenciales_ausentes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("POLYGON_API_KEY", raising=False)
        p = SondeoProvider(settings=get_settings())
        estado = p.status()
        assert not estado.available
        assert estado.missing_env == ("POLYGON_API_KEY",)
        assert "POLYGON_API_KEY" in estado.describe()

    def test_sondeo_se_memoriza(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Sin memoización, resolver dentro de un bucle por ticker gastaría la cuota."""
        monkeypatch.setenv("POLYGON_API_KEY", "k")
        clock = ManualClock()
        transport = ScriptedTransport(HttpResponse(200, b"ok"), repeat_last=True, clock=clock)
        http = HttpClient("polygon", transport=transport, clock=clock, rate=FAST_RATE)
        p = SondeoProvider(settings=get_settings(), http=http, clock=clock)
        for _ in range(5):
            assert p.available()
        assert transport.n_calls == 1
        clock.sleep(p.probe_ttl_s + 1)
        assert p.available()
        assert transport.n_calls == 2

    def test_sondeo_fallido_marca_no_disponible(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("POLYGON_API_KEY", "k")
        clock = ManualClock()
        transport = ScriptedTransport(
            TransportTimeout("polygon", "timeout"), repeat_last=True, clock=clock
        )
        http = HttpClient(
            "polygon",
            transport=transport,
            clock=clock,
            rate=FAST_RATE,
            retry=RetryPolicy(max_retries=0),
        )
        p = SondeoProvider(settings=get_settings(), http=http, clock=clock)
        estado = p.status()
        assert not estado.available
        assert "sin respuesta" in estado.reason


class TestProviderRegistry:
    def test_prioridad_mayor_primero(self) -> None:
        reg = ProviderRegistry(clock=ManualClock())
        bajo = StaticProvider("stooq", ("prices",))
        alto = StaticProvider("polygon", ("prices",))
        reg.register("prices", bajo, priority=10)
        reg.register("prices", alto, priority=100)
        assert reg.resolve("prices").name == "polygon"
        assert [p.name for p in reg.providers_for("prices")] == ["polygon", "stooq"]

    def test_empate_resuelto_por_orden_de_registro(self) -> None:
        """El determinismo también aplica a la elección de fuente."""
        reg = ProviderRegistry(clock=ManualClock())
        reg.register("prices", StaticProvider("a", ("prices",)), priority=5)
        reg.register("prices", StaticProvider("b", ("prices",)), priority=5)
        assert reg.resolve("prices").name == "a"

    def test_salta_los_no_disponibles(self) -> None:
        reg = ProviderRegistry(clock=ManualClock())
        reg.register(
            "prices",
            StaticProvider("polygon", ("prices",), is_available=False),
            priority=100,
        )
        reg.register("prices", StaticProvider("yfinance", ("prices",)), priority=1)
        assert reg.resolve("prices").name == "yfinance"

    def test_mensaje_enumera_lo_que_falta_en_cada_candidato(self) -> None:
        """El mensaje debe poder convertirse en un `export` sin abrir el código."""
        reg = ProviderRegistry(clock=ManualClock())
        reg.register(
            "prices",
            StaticProvider(
                "polygon",
                ("prices",),
                is_available=False,
                missing_env=("POLYGON_API_KEY",),
            ),
            priority=100,
        )
        reg.register(
            "prices",
            StaticProvider(
                "tiingo",
                ("prices",),
                is_available=False,
                missing_env=("TIINGO_API_KEY",),
            ),
            priority=50,
        )
        reg.register(
            "prices",
            StaticProvider(
                "yfinance", ("prices",), is_available=False, reason="egress bloqueado (403)"
            ),
            priority=10,
        )
        with pytest.raises(ProviderUnavailable) as exc:
            reg.resolve("prices")
        mensaje = str(exc.value)
        assert "POLYGON_API_KEY" in mensaje
        assert "TIINGO_API_KEY" in mensaje
        assert "egress bloqueado" in mensaje
        assert "prioridad 100" in mensaje
        assert set(exc.value.missing_env) == {"POLYGON_API_KEY", "TIINGO_API_KEY"}

    def test_kind_sin_proveedores(self) -> None:
        reg = ProviderRegistry(clock=ManualClock())
        reg.register("prices", StaticProvider("polygon", ("prices",)))
        with pytest.raises(ProviderUnavailable) as exc:
            reg.resolve(DataKind.OPTIONS)
        assert "no hay ningún proveedor registrado" in str(exc.value)
        assert "prices" in str(exc.value)

    def test_registro_incoherente_falla_pronto(self) -> None:
        reg = ProviderRegistry(clock=ManualClock())
        with pytest.raises(ConfigError, match="kinds"):
            reg.register("options", StaticProvider("polygon", ("prices",)))
        reg.register("options", StaticProvider("polygon", ("prices",)), strict_kinds=False)
        assert reg.resolve("options").name == "polygon"

    def test_duplicado_requiere_replace(self) -> None:
        reg = ProviderRegistry(clock=ManualClock())
        reg.register("prices", StaticProvider("polygon", ("prices",)), priority=1)
        with pytest.raises(ConfigError, match="ya está registrado"):
            reg.register("prices", StaticProvider("polygon", ("prices",)), priority=2)
        reg.register("prices", StaticProvider("polygon", ("prices",)), priority=2, replace=True)
        assert reg.priority_of("prices", "polygon") == 2

    def test_objeto_sin_protocolo(self) -> None:
        reg = ProviderRegistry(clock=ManualClock())
        with pytest.raises(ConfigError):
            reg.register("prices", object())  # type: ignore[arg-type]

    def test_unregister_y_clear(self) -> None:
        reg = ProviderRegistry(clock=ManualClock())
        reg.register("prices", StaticProvider("polygon", ("prices",)))
        assert reg.unregister("prices", "polygon") is True
        assert reg.unregister("prices", "polygon") is False
        reg.register("prices", StaticProvider("polygon", ("prices",)))
        reg.clear()
        assert reg.kinds() == []

    def test_normaliza_el_kind(self) -> None:
        reg = ProviderRegistry(clock=ManualClock())
        reg.register("  PRICES ", StaticProvider("polygon", ("prices",)))
        assert reg.resolve(DataKind.PRICES).name == "polygon"

    def test_available_que_revienta_no_tumba_la_resolucion(self) -> None:
        """Un proveedor con un `available()` roto es un proveedor no disponible."""

        class Roto:
            name = "roto"
            kinds = ("prices",)

            def available(self) -> bool:
                raise RuntimeError("boom")

        reg = ProviderRegistry(clock=ManualClock())
        reg.register("prices", Roto(), priority=100)
        reg.register("prices", StaticProvider("yfinance", ("prices",)), priority=1)
        assert reg.resolve("prices").name == "yfinance"


class TestFallbackEnCadena:
    def test_pasa_al_siguiente_y_lo_registra(self) -> None:
        """Si el prioritario falla en ejecución, se usa el siguiente y queda rastro."""
        reg = ProviderRegistry(clock=ManualClock())
        reg.register("prices", StaticProvider("polygon", ("prices",)), priority=100)
        reg.register("prices", StaticProvider("yfinance", ("prices",)), priority=10)

        def op(p: Any) -> str:
            if p.name == "polygon":
                raise RateLimited("polygon", 30.0)
            return f"datos de {p.name}"

        assert reg.call("prices", op, description="barras diarias") == "datos de yfinance"
        intentos = reg.attempts
        assert [a.provider for a in intentos] == ["polygon", "yfinance"]
        assert intentos[0].ok is False
        assert intentos[0].error_type == "RateLimited"
        assert intentos[1].ok is True
        assert "polygon" in intentos[0].describe()

    def test_si_todos_fallan_se_agrega_el_diagnostico(self) -> None:
        reg = ProviderRegistry(clock=ManualClock())
        reg.register("prices", StaticProvider("polygon", ("prices",)), priority=100)
        reg.register("prices", StaticProvider("yfinance", ("prices",)), priority=10)

        def op(p: Any) -> str:
            raise ProviderUnavailable(p.name, "el host no responde")

        with pytest.raises(ProviderUnavailable) as exc:
            reg.call("prices", op)
        mensaje = str(exc.value)
        assert "polygon" in mensaje
        assert "yfinance" in mensaje
        assert isinstance(exc.value.__cause__, ProviderUnavailable)
        assert len(reg.attempts) == 2

    def test_los_errores_de_programacion_se_propagan(self) -> None:
        """Un bug del *parser* no debe disfrazarse de cambio de proveedor."""
        reg = ProviderRegistry(clock=ManualClock())
        reg.register("prices", StaticProvider("polygon", ("prices",)), priority=100)
        reg.register("prices", StaticProvider("yfinance", ("prices",)), priority=10)

        def op(p: Any) -> str:
            raise KeyError("results")

        with pytest.raises(KeyError):
            reg.call("prices", op)
        assert len(reg.attempts) == 0

    def test_cuarentena_evita_repetir_el_fallo(self) -> None:
        """N tickers contra un proveedor caído no deben ser N timeouts."""
        clock = ManualClock()
        reg = ProviderRegistry(clock=clock, failure_cooldown_s=300.0)
        reg.register("prices", StaticProvider("polygon", ("prices",)), priority=100)
        reg.register("prices", StaticProvider("yfinance", ("prices",)), priority=10)
        llamadas: list[str] = []

        def op(p: Any) -> str:
            llamadas.append(p.name)
            if p.name == "polygon":
                raise ProviderUnavailable("polygon", "timeout")
            return "ok"

        for _ in range(4):
            assert reg.call("prices", op) == "ok"
        assert llamadas.count("polygon") == 1
        assert reg.is_quarantined("polygon")

        clock.sleep(301.0)
        assert not reg.is_quarantined("polygon")
        reg.call("prices", op)
        assert llamadas.count("polygon") == 2

    def test_release_levanta_la_cuarentena(self) -> None:
        reg = ProviderRegistry(clock=ManualClock())
        reg.quarantine("polygon", 100.0)
        assert reg.is_quarantined("polygon")
        reg.release("polygon")
        assert not reg.is_quarantined("polygon")

    def test_describe_es_legible(self) -> None:
        reg = ProviderRegistry(clock=ManualClock())
        reg.register(
            "prices",
            StaticProvider("polygon", ("prices",), is_available=False, missing_env=("K",)),
            priority=100,
        )
        texto = reg.describe()
        assert "prices:" in texto
        assert "polygon" in texto
        assert "K" in texto

    def test_registro_global_aislable(self) -> None:
        try:
            reg = ProviderRegistry(clock=ManualClock())
            set_registry(reg)
            assert get_registry() is reg
        finally:
            set_registry(None)


def test_provider_status_describe_disponible() -> None:
    assert "disponible" in ProviderStatus("x", True).describe()


# ===========================================================================
# 5. Canonicalización de parámetros
# ===========================================================================


class TestCanonicalizacion:
    def test_orden_de_las_claves_no_cambia_el_hash(self) -> None:
        a = params_hash("prices", "polygon", {"ticker": "AAPL", "start": date(2024, 1, 1)})
        b = params_hash("prices", "polygon", {"start": date(2024, 1, 1), "ticker": "AAPL"})
        assert a == b

    def test_kind_y_provider_forman_parte_de_la_clave(self) -> None:
        """Los precios de dos proveedores no son intercambiables: claves distintas."""
        p = {"ticker": "AAPL"}
        assert params_hash("prices", "polygon", p) != params_hash("prices", "yfinance", p)
        assert params_hash("prices", "polygon", p) != params_hash("options", "polygon", p)

    def test_fecha_y_cadena_no_colisionan(self) -> None:
        """Sin etiquetas de tipo, `date(2024,1,1)` y `"2024-01-01"` compartirían entrada."""
        a = params_hash("prices", "p", {"end": date(2024, 1, 1)})
        b = params_hash("prices", "p", {"end": "2024-01-01"})
        assert a != b

    def test_tupla_lista_y_conjunto_se_distinguen(self) -> None:
        h = [
            params_hash("k", "p", {"x": [1, 2]}),
            params_hash("k", "p", {"x": (1, 2)}),
            params_hash("k", "p", {"x": {1, 2}}),
        ]
        assert len(set(h)) == 3

    def test_conjunto_es_estable_pese_al_orden_de_iteracion(self) -> None:
        assert params_hash("k", "p", {"x": {"b", "a"}}) == params_hash("k", "p", {"x": {"a", "b"}})

    def test_secuencias_largas_se_resumen_por_digest(self) -> None:
        """Un `params` con 500 tickers no puede hacer ilegibles los metadatos."""
        muchos = [f"T{i}" for i in range(500)]
        canon = canonicalize({"tickers": muchos})
        assert "__seq__" in canon["tickers"]
        assert canon["tickers"]["__seq__"]["len"] == 500
        otros = [*muchos[:-1], "DISTINTO"]
        assert params_hash("k", "p", {"tickers": muchos}) != params_hash(
            "k", "p", {"tickers": otros}
        )

    def test_tipos_de_numpy_y_pandas(self) -> None:
        assert canonicalize(np.int64(3)) == 3
        assert canonicalize(np.float64(1.5)) == 1.5
        assert canonicalize(np.bool_(True)) is True
        assert canonicalize(pd.Timestamp("2024-01-01")) == {"__datetime__": "2024-01-01T00:00:00"}
        assert canonicalize(float("nan")) == {"__float__": "nan"}

    def test_objeto_no_serializable_es_error(self) -> None:
        class Opaco:
            pass

        with pytest.raises(ConfigError, match="no serializable"):
            canonicalize({"x": Opaco()})

    def test_objeto_con_cache_key_propia(self) -> None:
        class ConClave:
            def cache_key(self) -> dict[str, Any]:
                return {"tipo": "universo", "n": 503}

        assert canonicalize(ConClave()) == {"n": 503, "tipo": "universo"}

    def test_dataframe_no_puede_ser_parametro(self) -> None:
        with pytest.raises(ConfigError, match="DataFrame"):
            canonicalize({"panel": pd.DataFrame({"a": [1]})})

    def test_datakind_es_una_cadena(self) -> None:
        assert params_hash(DataKind.PRICES, "p", {}) == params_hash("prices", "p", {})


# ===========================================================================
# 6. Caché en disco
# ===========================================================================


@pytest.fixture
def wall() -> ManualWallClock:
    return ManualWallClock(datetime(2026, 8, 4, 12, 0, tzinfo=UTC))


@pytest.fixture
def cache(tmp_path: Path, wall: ManualWallClock) -> DiskCache:
    return DiskCache(tmp_path / "cache", clock=wall, default_ttl_s=3600.0, offline=False)


class TestCacheRoundTrip:
    def test_panel_multiindex_ida_y_vuelta_exacta(self, cache: DiskCache) -> None:
        """El panel canónico debe volver **idéntico**: dtypes, nombres y resolución.

        Un `datetime64[ns]` que vuelve como `[us]`, o un `date` que vuelve como
        texto, rompe `pd.merge_asof` en `pit.asof_join` y con él todo el
        point-in-time.
        """
        panel = make_panel(4)
        params = {"tickers": ("AAPL", "BRK.B"), "start": date(2024, 1, 2)}
        cache.write("prices", "polygon", params, panel)
        vuelto = cache.read("prices", "polygon", params)

        pd.testing.assert_frame_equal(vuelto, panel)
        assert vuelto.index.names == ["date", "ticker"]
        assert vuelto.index.levels[0].dtype == panel.index.levels[0].dtype
        assert vuelto["volume"].dtype == np.dtype("int64")

    def test_series_conserva_el_nombre(self, cache: DiskCache) -> None:
        panel = make_panel()
        serie = panel["close"].rename("sue")
        cache.write("factors", "synthetic", {"f": "sue"}, serie)
        vuelta = cache.read("factors", "synthetic", {"f": "sue"})
        assert isinstance(vuelta, pd.Series)
        assert vuelta.name == "sue"
        pd.testing.assert_series_equal(vuelta, serie)

    def test_series_sin_nombre(self, cache: DiskCache) -> None:
        serie = pd.Series([1.0, 2.0], index=pd.Index(["A", "B"], name="ticker"))
        cache.write("factors", "synthetic", {"f": 1}, serie)
        vuelta = cache.read("factors", "synthetic", {"f": 1})
        assert vuelta.name is None
        pd.testing.assert_series_equal(vuelta, serie)

    def test_zona_horaria_sobrevive(self, cache: DiskCache) -> None:
        """`announced_at` viaja en UTC; perder la zona movería eventos de sesión."""
        frame = pd.DataFrame(
            {
                "announced_at": pd.date_range("2024-01-02 21:05", periods=3, tz="UTC"),
                "ticker": ["AAPL", "MSFT", "NVDA"],
            }
        )
        cache.write("earnings_calendar", "fmp", {"q": "2024Q1"}, frame)
        vuelto = cache.read("earnings_calendar", "fmp", {"q": "2024Q1"})
        assert str(vuelto["announced_at"].dtype).endswith("UTC]")
        pd.testing.assert_frame_equal(vuelto, frame)

    def test_json_con_tipos_ricos(self, cache: DiskCache) -> None:
        valor = {
            "fecha": date(2024, 3, 31),
            "instante": datetime(2024, 5, 2, 20, 30, tzinfo=UTC),
            "tupla": (1, 2, 3),
            "anidado": {"lista": [1, 2, {"x": date(2020, 1, 1)}]},
        }
        cache.write("filings", "sec", {"cik": "0000320193"}, valor)
        vuelto = cache.read("filings", "sec", {"cik": "0000320193"})
        assert vuelto["fecha"] == date(2024, 3, 31)
        assert vuelto["instante"] == datetime(2024, 5, 2, 20, 30, tzinfo=UTC)
        assert vuelto["tupla"] == (1, 2, 3)
        assert vuelto["anidado"]["lista"][2]["x"] == date(2020, 1, 1)

    def test_bytes_crudos(self, cache: DiskCache) -> None:
        crudo = b"\x00\x01binario\xff" * 10
        cache.write("filings", "sec", {"accession": "0000320193-24-000081"}, crudo)
        assert cache.read("filings", "sec", {"accession": "0000320193-24-000081"}) == crudo

    def test_los_metadatos_registran_el_momento_de_escritura(
        self, cache: DiskCache, wall: ManualWallClock
    ) -> None:
        panel = make_panel()
        meta = cache.write("prices", "polygon", {"t": "AAPL"}, panel, tags=["carga-inicial"])
        assert meta.written_at == wall.now()
        assert meta.rows == len(panel)
        assert meta.n_bytes > 0
        assert meta.tags == ("carga-inicial",)
        assert meta.version == CACHE_FORMAT_VERSION

        en_disco = cache.stat("prices", "polygon", {"t": "AAPL"})
        assert en_disco is not None
        assert en_disco.written_at == meta.written_at
        assert en_disco.params == {"t": "AAPL"}

    def test_escritura_atomica_no_deja_temporales(self, cache: DiskCache) -> None:
        cache.write("prices", "polygon", {"t": "AAPL"}, make_panel())
        cache.write("prices", "polygon", {"t": "AAPL"}, make_panel(5))
        assert list(cache.root.rglob("*.tmp-*")) == []
        assert len(cache.read("prices", "polygon", {"t": "AAPL"})) == 10


class TestCacheTTL:
    def test_expiracion_por_ttl(self, cache: DiskCache, wall: ManualWallClock) -> None:
        cache.write("estimates", "fmp", {"t": "AAPL"}, {"eps": 1.5}, ttl=3600)
        assert cache.has("estimates", "fmp", {"t": "AAPL"})
        wall.advance(3599)
        assert cache.read("estimates", "fmp", {"t": "AAPL"}) == {"eps": 1.5}
        wall.advance(2)
        assert not cache.has("estimates", "fmp", {"t": "AAPL"})
        with pytest.raises(CacheMiss, match="caducada"):
            cache.read("estimates", "fmp", {"t": "AAPL"})
        assert cache.stats.expired == 1

    def test_ttl_por_defecto_de_la_configuracion(self, tmp_path: Path) -> None:
        wall = ManualWallClock()
        cfg = get_settings(cache_ttl_days=2)
        c = DiskCache(tmp_path, settings=cfg, clock=wall)
        meta = c.write("estimates", "fmp", {"t": "AAPL"}, {"eps": 1.0})
        assert meta.ttl_s == pytest.approx(2 * 86_400.0)

    def test_ttl_como_timedelta(self, cache: DiskCache, wall: ManualWallClock) -> None:
        cache.write("news", "finnhub", {"t": "AAPL"}, [1], ttl=timedelta(minutes=30))
        wall.advance(timedelta(minutes=31))
        assert not cache.has("news", "finnhub", {"t": "AAPL"})

    def test_inmutable_no_caduca_nunca(self, cache: DiskCache, wall: ManualWallClock) -> None:
        """El histórico cerrado es lo más caro de reconstruir: no debe caducar."""
        panel = make_panel()
        meta = cache.write("prices", "polygon", {"end": date(2015, 12, 31)}, panel, immutable=True)
        assert meta.ttl_s is None
        assert meta.expires_at is None
        wall.advance(timedelta(days=3650))
        assert cache.has("prices", "polygon", {"end": date(2015, 12, 31)})
        pd.testing.assert_frame_equal(
            cache.read("prices", "polygon", {"end": date(2015, 12, 31)}), panel
        )

    def test_allow_stale_sirve_lo_caducado_bajo_demanda(
        self, cache: DiskCache, wall: ManualWallClock
    ) -> None:
        cache.write("estimates", "fmp", {"t": "AAPL"}, {"eps": 1.0}, ttl=60)
        wall.advance(120)
        assert cache.get("estimates", "fmp", {"t": "AAPL"}) is None
        assert cache.read("estimates", "fmp", {"t": "AAPL"}, allow_stale=True) == {"eps": 1.0}
        assert cache.stats.stale_hits == 1


class TestInmutabilidad:
    def test_ventana_cerrada_es_inmutable(self) -> None:
        pol = ImmutabilityPolicy(settled_after_days=5)
        hoy = date(2026, 8, 4)
        assert pol.classify("prices", {"end": date(2024, 12, 31)}, today=hoy) is True
        assert pol.classify("prices", {"end": date(2026, 8, 3)}, today=hoy) is False
        assert pol.classify("prices", {"start": date(2020, 1, 1)}, today=hoy) is False

    def test_el_consenso_nunca_es_inmutable(self) -> None:
        """Estimaciones y calendario se corrigen a posteriori aunque la ventana esté cerrada."""
        pol = ImmutabilityPolicy()
        hoy = date(2026, 8, 4)
        assert pol.classify("estimates", {"end": date(2020, 1, 1)}, today=hoy) is False
        assert pol.classify("earnings_calendar", {"end": date(2020, 1, 1)}, today=hoy) is False

    def test_la_cache_aplica_la_politica_por_defecto(
        self, cache: DiskCache, wall: ManualWallClock
    ) -> None:
        antiguo = cache.write("prices", "polygon", {"end": date(2019, 6, 30)}, make_panel())
        reciente = cache.write("prices", "polygon", {"end": wall.now().date()}, make_panel())
        assert antiguo.immutable is True
        assert antiguo.ttl_s is None
        assert reciente.immutable is False
        assert reciente.ttl_s == 3600.0

    def test_fechas_en_texto_iso_tambien_valen(self) -> None:
        pol = ImmutabilityPolicy()
        assert pol.end_of({"end": "2024-01-05"}) == date(2024, 1, 5)
        assert pol.end_of({"end": "no es fecha"}) is None
        assert pol.end_of(None) is None


class TestCacheFetchYEnvoltorios:
    def test_fetch_llama_al_proveedor_una_sola_vez(self, cache: DiskCache) -> None:
        llamadas = {"n": 0}

        def loader() -> pd.DataFrame:
            llamadas["n"] += 1
            return make_panel()

        p = {"t": "AAPL", "end": date(2026, 8, 3)}
        a = cache.fetch("prices", "polygon", p, loader)
        b = cache.fetch("prices", "polygon", p, loader)
        assert llamadas["n"] == 1
        pd.testing.assert_frame_equal(a, b)
        assert cache.stats.hits == 1

    def test_refresh_fuerza_la_relectura(self, cache: DiskCache) -> None:
        llamadas = {"n": 0}

        def loader() -> dict[str, int]:
            llamadas["n"] += 1
            return {"n": llamadas["n"]}

        cache.fetch("news", "finnhub", {"t": "AAPL"}, loader)
        vuelto = cache.fetch("news", "finnhub", {"t": "AAPL"}, loader, refresh=True)
        assert llamadas["n"] == 2
        assert vuelto == {"n": 2}

    def test_context_manager(self, cache: DiskCache) -> None:
        panel = make_panel()
        with cache.entry("prices", "polygon", {"t": "AAPL"}) as slot:
            assert slot.hit is False
            slot.store(panel)
        with cache.entry("prices", "polygon", {"t": "AAPL"}) as slot2:
            assert slot2.hit is True
            pd.testing.assert_frame_equal(slot2.value, panel)
            assert slot2.meta is not None

    def test_decorador_sobre_un_metodo(self, tmp_path: Path, wall: ManualWallClock) -> None:
        class Proveedor:
            name = "polygon"

            def __init__(self, store: DiskCache) -> None:
                self.cache = store
                self.n = 0

            @cached("cache", DataKind.PRICES, ttl=600)
            def bars(self, ticker: str, start: date, end: date) -> pd.DataFrame:
                self.n += 1
                return make_panel()

        store = DiskCache(tmp_path, clock=wall)
        p = Proveedor(store)
        a = p.bars("AAPL", date(2024, 1, 1), date(2024, 1, 5))
        b = p.bars("AAPL", start=date(2024, 1, 1), end=date(2024, 1, 5))
        assert p.n == 1, "posicional y por nombre deben dar la misma clave"
        pd.testing.assert_frame_equal(a, b)
        p.bars("MSFT", date(2024, 1, 1), date(2024, 1, 5))
        assert p.n == 2
        assert store.stat("prices", "polygon", {"ticker": "MSFT", "start": date(2024, 1, 1),
                                                "end": date(2024, 1, 5)}) is not None

    def test_decorador_ignora_parametros_irrelevantes(
        self, tmp_path: Path, wall: ManualWallClock
    ) -> None:
        store = DiskCache(tmp_path, clock=wall)
        llamadas = {"n": 0}

        @cached(store, "prices", "yfinance", ignore=("verbose",))
        def bars(ticker: str, verbose: bool = False) -> dict[str, str]:
            llamadas["n"] += 1
            return {"t": ticker}

        bars("AAPL", verbose=False)
        bars("AAPL", verbose=True)
        assert llamadas["n"] == 1

    def test_decorador_con_cache_mal_configurada(self, tmp_path: Path) -> None:
        @cached("cache", "prices", "yfinance")
        def bars(ticker: str) -> dict[str, str]:
            return {"t": ticker}

        with pytest.raises(ConfigError):
            bars("AAPL")


class TestCacheRechazos:
    def test_no_se_cachea_none(self, cache: DiskCache) -> None:
        with pytest.raises(DataQualityError, match="None"):
            cache.write("prices", "polygon", {"t": "AAPL"}, None)

    def test_no_se_cachea_un_panel_vacio(self, cache: DiskCache) -> None:
        """Un vacío cacheado es indistinguible, meses después, de 'no hubo datos'."""
        vacio = make_panel().iloc[0:0]
        with pytest.raises(DataQualityError, match="vacío"):
            cache.write("prices", "polygon", {"t": "ZZZZ"}, vacio)
        meta = cache.write("prices", "polygon", {"t": "ZZZZ"}, vacio, allow_empty=True)
        assert meta.rows == 0
        assert len(cache.read("prices", "polygon", {"t": "ZZZZ"})) == 0

    def test_dataframe_no_serializable(self, cache: DiskCache) -> None:
        mezclado = pd.DataFrame({"a": [1, "texto", None]})
        with pytest.raises(DataQualityError, match="parquet"):
            cache.write("prices", "polygon", {"t": "AAPL"}, mezclado)
        assert list(cache.root.rglob("*.tmp-*")) == []

    def test_ttl_negativo(self, cache: DiskCache) -> None:
        with pytest.raises(ConfigError):
            cache.write("news", "finnhub", {"t": "A"}, [1], ttl=-5)


class TestCacheCorrupcion:
    def test_parquet_corrupto_se_trata_como_fallo_y_se_borra(self, cache: DiskCache) -> None:
        params = {"t": "AAPL"}
        cache.write("prices", "polygon", params, make_panel())
        ruta = cache.path_of("prices", "polygon", params)
        assert ruta is not None
        ruta.write_bytes(b"esto no es parquet")

        with pytest.raises(CacheMiss, match="corrupta"):
            cache.read("prices", "polygon", params)
        assert cache.stat("prices", "polygon", params) is None
        assert not ruta.exists()
        assert cache.stats.errors == 1

    def test_metadatos_ilegibles(self, cache: DiskCache) -> None:
        params = {"t": "AAPL"}
        cache.write("prices", "polygon", params, make_panel())
        key = cache.key_for("prices", "polygon", params)
        meta_path = cache.root / key.relative_dir / f"{key.stem}.meta.json"
        meta_path.write_text("{ roto", encoding="utf-8")
        with pytest.raises(CacheMiss):
            cache.read("prices", "polygon", params)

    def test_version_de_formato_distinta_se_ignora(self, cache: DiskCache) -> None:
        """Leer un esquema viejo con el código nuevo daría datos mal interpretados."""
        params = {"t": "AAPL"}
        cache.write("prices", "polygon", params, make_panel())
        key = cache.key_for("prices", "polygon", params)
        meta_path = cache.root / key.relative_dir / f"{key.stem}.meta.json"
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
        raw["version"] = CACHE_FORMAT_VERSION + 1
        meta_path.write_text(json.dumps(raw), encoding="utf-8")
        assert cache.get("prices", "polygon", params) is None

    def test_falta_el_fichero_de_datos(self, cache: DiskCache) -> None:
        params = {"t": "AAPL"}
        cache.write("prices", "polygon", params, make_panel())
        ruta = cache.path_of("prices", "polygon", params)
        assert ruta is not None
        ruta.unlink()
        with pytest.raises(CacheMiss):
            cache.read("prices", "polygon", params)


class TestInvalidacion:
    def _poblar(self, cache: DiskCache) -> None:
        cache.write("prices", "polygon", {"t": "AAPL"}, make_panel(), tags=["diario"])
        cache.write("prices", "polygon", {"t": "MSFT"}, make_panel(), tags=["diario"])
        cache.write("prices", "yfinance", {"t": "AAPL"}, make_panel(), tags=["control"])
        cache.write("fundamentals", "sec", {"cik": "0000320193"}, {"x": 1})
        cache.write(
            "prices", "polygon", {"t": "NVDA", "end": date(2015, 1, 1)}, make_panel()
        )  # inmutable por política

    def test_por_kind_y_proveedor(self, cache: DiskCache) -> None:
        self._poblar(cache)
        assert cache.invalidate(kind="prices", provider="polygon") == 2
        assert cache.get("prices", "polygon", {"t": "AAPL"}) is None
        assert cache.get("prices", "yfinance", {"t": "AAPL"}) is not None
        assert cache.get("fundamentals", "sec", {"cik": "0000320193"}) is not None

    def test_los_inmutables_estan_protegidos(self, cache: DiskCache) -> None:
        """Un refresco de la última semana no puede llevarse veinte años de histórico."""
        self._poblar(cache)
        params = {"t": "NVDA", "end": date(2015, 1, 1)}
        assert cache.stat("prices", "polygon", params).immutable is True
        cache.invalidate(kind="prices")
        assert cache.get("prices", "polygon", params) is not None
        assert cache.invalidate(kind="prices", include_immutable=True) >= 1
        assert cache.get("prices", "polygon", params) is None

    def test_por_etiqueta(self, cache: DiskCache) -> None:
        self._poblar(cache)
        assert cache.invalidate(tags=["control"]) == 1
        assert cache.get("prices", "yfinance", {"t": "AAPL"}) is None
        assert cache.get("prices", "polygon", {"t": "AAPL"}) is not None

    def test_por_params_exactos(self, cache: DiskCache) -> None:
        self._poblar(cache)
        assert cache.invalidate(kind="prices", provider="polygon", params={"t": "AAPL"}) == 1
        assert cache.get("prices", "polygon", {"t": "AAPL"}) is None
        assert cache.get("prices", "polygon", {"t": "MSFT"}) is not None

    def test_por_params_exige_kind_y_provider(self, cache: DiskCache) -> None:
        with pytest.raises(ConfigError):
            cache.invalidate(params={"t": "AAPL"})

    def test_por_antiguedad(self, cache: DiskCache, wall: ManualWallClock) -> None:
        cache.write("news", "finnhub", {"d": 1}, [1])
        corte = wall.advance(timedelta(days=1))
        cache.write("news", "finnhub", {"d": 2}, [2])
        assert cache.invalidate(kind="news", written_before=corte) == 1
        assert cache.get("news", "finnhub", {"d": 2}, allow_stale=True) == [2]

    def test_purga_de_caducados(self, cache: DiskCache, wall: ManualWallClock) -> None:
        cache.write("news", "finnhub", {"d": 1}, [1], ttl=60)
        cache.write("news", "finnhub", {"d": 2}, [2], ttl=100_000)
        wall.advance(120)
        assert cache.purge_expired() == 1
        assert cache.get("news", "finnhub", {"d": 2}) == [2]

    def test_predicado_libre(self, cache: DiskCache) -> None:
        self._poblar(cache)
        n = cache.invalidate(predicate=lambda m: m.provider == "sec")
        assert n == 1
        assert cache.get("fundamentals", "sec", {"cik": "0000320193"}) is None

    def test_clear_y_tamano(self, cache: DiskCache) -> None:
        self._poblar(cache)
        assert cache.size_bytes() > 0
        assert cache.clear() == 5
        assert list(cache.iter_meta()) == []
        assert cache.size_bytes() == 0


class TestModoOffline:
    def test_un_fallo_de_cache_es_error_y_no_se_llama_al_proveedor(
        self, tmp_path: Path, wall: ManualWallClock
    ) -> None:
        """La garantía central del modo offline: no se sale a la red, punto."""
        store = DiskCache(tmp_path, clock=wall, offline=True)
        llamadas = {"n": 0}

        def loader() -> pd.DataFrame:
            llamadas["n"] += 1
            return make_panel()

        with pytest.raises(ProviderUnavailable) as exc:
            store.fetch("prices", "polygon", {"t": "AAPL"}, loader)
        assert llamadas["n"] == 0
        assert "offline" in str(exc.value)
        assert "polygon" in str(exc.value)

    def test_lo_cacheado_se_sirve_igual(self, tmp_path: Path, wall: ManualWallClock) -> None:
        panel = make_panel()
        conectada = DiskCache(tmp_path, clock=wall)
        conectada.write("prices", "polygon", {"t": "AAPL"}, panel)

        offline = DiskCache(tmp_path, clock=wall, offline=True)
        pd.testing.assert_frame_equal(offline.read("prices", "polygon", {"t": "AAPL"}), panel)

        def loader() -> pd.DataFrame:  # pragma: no cover - no debe ejecutarse
            raise AssertionError("el modo offline no puede invocar al proveedor")

        pd.testing.assert_frame_equal(
            offline.fetch("prices", "polygon", {"t": "AAPL"}, loader), panel
        )

    def test_caducado_se_sirve_por_defecto(self, tmp_path: Path, wall: ManualWallClock) -> None:
        """Sin red no hay refresco posible; servir el dato exacto del backtest es lo correcto."""
        conectada = DiskCache(tmp_path, clock=wall)
        conectada.write("estimates", "fmp", {"t": "AAPL"}, {"eps": 1.0}, ttl=60)
        wall.advance(3600)

        offline = DiskCache(tmp_path, clock=wall, offline=True)
        assert offline.read("estimates", "fmp", {"t": "AAPL"}) == {"eps": 1.0}
        assert offline.stats.stale_hits == 1

        estricta = DiskCache(tmp_path, clock=wall, offline=True, serve_stale_offline=False)
        with pytest.raises(ProviderUnavailable, match="caducada"):
            estricta.read("estimates", "fmp", {"t": "AAPL"})

    def test_get_tambien_lanza_offline(self, tmp_path: Path, wall: ManualWallClock) -> None:
        """Devolver `None` induciría a quien llama a intentar la red."""
        store = DiskCache(tmp_path, clock=wall, offline=True)
        with pytest.raises(ProviderUnavailable):
            store.get("prices", "polygon", {"t": "AAPL"})

    def test_refresh_esta_prohibido(self, tmp_path: Path, wall: ManualWallClock) -> None:
        store = DiskCache(tmp_path, clock=wall, offline=True)
        store.write("prices", "polygon", {"t": "AAPL"}, make_panel())
        with pytest.raises(ProviderUnavailable, match="refrescar"):
            store.fetch("prices", "polygon", {"t": "AAPL"}, lambda: make_panel(), refresh=True)
        with (
            pytest.raises(ProviderUnavailable, match="refrescar"),
            store.entry("prices", "polygon", {"t": "AAPL"}, refresh=True),
        ):
            pass  # pragma: no cover

    def test_offline_desde_la_configuracion(self, tmp_path: Path) -> None:
        cfg = get_settings(offline=True, cache_dir=tmp_path)
        store = DiskCache.from_settings(cfg, clock=ManualWallClock())
        assert store.offline is True
        with pytest.raises(ProviderUnavailable):
            store.read("prices", "polygon", {"t": "AAPL"})

    def test_corrupcion_en_offline_es_error(self, tmp_path: Path, wall: ManualWallClock) -> None:
        conectada = DiskCache(tmp_path, clock=wall)
        conectada.write("prices", "polygon", {"t": "AAPL"}, make_panel())
        ruta = conectada.path_of("prices", "polygon", {"t": "AAPL"})
        assert ruta is not None
        ruta.write_bytes(b"basura")

        offline = DiskCache(tmp_path, clock=wall, offline=True)
        with pytest.raises(ProviderUnavailable, match="corrupta"):
            offline.read("prices", "polygon", {"t": "AAPL"})


class TestCacheDesactivada:
    def test_enabled_false_siempre_llama_al_proveedor(
        self, tmp_path: Path, wall: ManualWallClock
    ) -> None:
        store = DiskCache(tmp_path, clock=wall, enabled=False)
        llamadas = {"n": 0}

        def loader() -> dict[str, int]:
            llamadas["n"] += 1
            return {"n": llamadas["n"]}

        assert store.fetch("news", "finnhub", {"t": "A"}, loader) == {"n": 1}
        assert store.fetch("news", "finnhub", {"t": "A"}, loader) == {"n": 2}
        assert list(store.root.rglob("*.meta.json")) == []


# ===========================================================================
# 7. Integración de las tres piezas
# ===========================================================================


class ProveedorFalso(BaseProvider):
    """Proveedor de precios que descarga por HTTP y cachea, como los reales."""

    kinds = ("prices",)

    def __init__(self, name: str, http: HttpClient, store: DiskCache) -> None:
        super().__init__(name=name, settings=get_settings(), http=http)
        self.cache = store

    def required_env(self) -> list[str]:
        return []

    def bars(self, ticker: str, end: date) -> pd.DataFrame:
        params = {"ticker": ticker, "end": end}
        return self.cache.fetch(
            DataKind.PRICES,
            self.name,
            params,
            lambda: self._descargar(ticker),
        )

    def _descargar(self, ticker: str) -> pd.DataFrame:
        payload = self.http.get_json(f"https://api.test/bars/{ticker}")
        return pd.DataFrame(payload["rows"]).set_index(
            pd.MultiIndex.from_product(
                [pd.to_datetime(payload["dates"]).as_unit("ns"), [ticker]],
                names=["date", "ticker"],
            )
        )


def _cuerpo_barras() -> bytes:
    return json.dumps(
        {"dates": ["2024-01-02", "2024-01-03"], "rows": [{"close": 1.0}, {"close": 2.0}]}
    ).encode()


class TestIntegracion:
    def test_cadena_completa_con_fallback_y_cache(
        self, tmp_path: Path, wall: ManualWallClock
    ) -> None:
        """El prioritario da 429 sin remedio; se cae al secundario y se cachea el resultado."""
        clock = ManualClock()
        store = DiskCache(tmp_path, clock=wall)

        primario = ProveedorFalso(
            "polygon",
            HttpClient(
                "polygon",
                transport=ScriptedTransport(
                    HttpResponse(429, b"", {"Retry-After": "1"}), repeat_last=True, clock=clock
                ),
                clock=clock,
                rate=FAST_RATE,
                retry=RetryPolicy(max_retries=1, backoff_base_s=0.1, jitter="none"),
            ),
            store,
        )
        secundario = ProveedorFalso(
            "yfinance",
            HttpClient(
                "yfinance",
                transport=ScriptedTransport(
                    HttpResponse(200, _cuerpo_barras()), repeat_last=True, clock=clock
                ),
                clock=clock,
                rate=FAST_RATE,
            ),
            store,
        )

        reg = ProviderRegistry(clock=clock)
        reg.register(DataKind.PRICES, primario, priority=100)
        reg.register(DataKind.PRICES, secundario, priority=10)

        panel = reg.call(DataKind.PRICES, lambda p: p.bars("AAPL", date(2024, 1, 3)))
        assert list(panel.columns) == ["close"]
        assert panel.index.names == ["date", "ticker"]

        fallos = [a for a in reg.attempts if not a.ok]
        assert fallos and fallos[0].provider == "polygon"
        assert fallos[0].error_type == "RateLimited"

        # La segunda llamada ni siquiera toca el transporte: sale de la caché.
        antes = secundario.http.stats.requests
        reg.call(DataKind.PRICES, lambda p: p.bars("AAPL", date(2024, 1, 3)))
        assert secundario.http.stats.requests == antes

    def test_backtest_reproducible_en_offline(self, tmp_path: Path, wall: ManualWallClock) -> None:
        """Lo cargado con red se replica sin red y sin proveedor alguno."""
        clock = ManualClock()
        conectada = DiskCache(tmp_path, clock=wall)
        proveedor = ProveedorFalso(
            "yfinance",
            HttpClient(
                "yfinance",
                transport=ScriptedTransport(
                    HttpResponse(200, _cuerpo_barras()), repeat_last=True, clock=clock
                ),
                clock=clock,
                rate=FAST_RATE,
            ),
            conectada,
        )
        original = proveedor.bars("AAPL", date(2024, 1, 3))

        offline_store = DiskCache(tmp_path, clock=wall, offline=True)
        sin_red = ProveedorFalso(
            "yfinance",
            HttpClient(
                "yfinance",
                transport=ScriptedTransport(
                    TransportError("yfinance", "no hay red"), repeat_last=True, clock=clock
                ),
                clock=clock,
                rate=FAST_RATE,
            ),
            offline_store,
        )
        pd.testing.assert_frame_equal(sin_red.bars("AAPL", date(2024, 1, 3)), original)

        with pytest.raises(ProviderUnavailable):
            sin_red.bars("MSFT", date(2024, 1, 3))
