# Ejemplos ejecutables

Tres scripts de demostración de la plataforma, **ejecutables tal cual** desde la
raíz del repo (o desde cualquier sitio: cada script añade la raíz a `sys.path`).
Ninguno necesita red ni credenciales: los dos primeros corren sobre el mercado
sintético (`earnings_alpha.data.synthetic.SyntheticMarket`) y el tercero sobre
el dataset REAL de consenso incluido en `data/external/consenso/`.

```bash
python3 examples/angulo_a_factores.py --rapido    # ~5 s
python3 examples/angulo_b_eventos.py --rapido     # ~5 s
python3 examples/sorpresas_reales.py              # ~2 s
```

Sin `--rapido`, los dos primeros usan un universo y un rango mayores (decenas de
segundos) y el ángulo B añade el `SurpriseModel` con validación cruzada purgada.

## `angulo_a_factores.py` — factores fundamentales cross-section

Pipeline continuo completo (`earnings_alpha.pipeline.run_continuous_pipeline`):

    universo PIT → fundamentales → factores → neutralización (sector, tamaño)
    → combinación → backtest cross-section → métricas

Salida: IC por factor y de la señal combinada (rank-IC medio con t de
Newey-West e intervalo de confianza — en este repo no existe métrica sin
banda), y el backtest neto con Sharpe ± IC, drawdown, rotación y el desglose de
costes (spread por tramo de liquidez, impacto √participación, comisión,
préstamo de cortos).

Factores por defecto (cada uno cita su referencia en su docstring): SUE de
analistas (Livnat-Mendenhall 2006), PEAD (Bernard-Thomas 1989), sorpresa de
ingresos (Jegadeesh-Livnat 2006), earnings yield (Basu 1977), Piotroski (2000)
y accruals de Sloan (1996) por flujo de caja.

## `angulo_b_eventos.py` — ventana de resultados y flujo informado

Pipeline de eventos completo (`earnings_alpha.pipeline.run_event_pipeline`):

    universo PIT → calendario de anuncios → PreEventFeatures →
    InformedTradingScore → SurpriseModel (CV purgada con embargo) →
    EventBacktest.run_grid → estudio de eventos con CAAR por quintil de SUE

El mercado sintético inyecta una huella de negociación informada en una
fracción conocida de los anuncios y publica la verdad-terreno
(`leaked_event_ids()`), así que la salida incluye el **AUC medido** del
detector — algo imposible con datos reales, donde el conjunto de eventos con
negociación informada es inobservable. También imprime la aritmética de tasa
base (PPV) que justifica que el score pondere exposición en vez de disparar
alertas binarias.

Nota legal y metodológica (contrato §3.5): todas las features derivan de datos
públicos; el objetivo es detectar la huella estadística del comportamiento de
otros participantes, no acceder a información privilegiada.

## `sorpresas_reales.py` — sorpresas REALES del S&P 500 (1996-2026)

Usa `factors.load_consensus_events` sobre
`data/external/consenso/consenso_master.parquet` (56.971 filas, ~500 tickers)
para construir eventos reales y mostrar:

1. La **advertencia PIT** del dataset, que todo consumidor debe propagar:
   consenso *final* previo al anuncio (sin vintages → prohibido para momentum
   de revisiones) y supervivencia parcial de tickers (snapshots de 2022/2025).
2. La distribución de sorpresas: beat/miss/meet, percentiles con colas gruesas
   y el beat rate por año (del ~47 % en 1997 al ~80 % reciente: el *walk-down*
   del consenso, Richardson-Teoh-Wysocki 2004).
3. El calendario BMO/AMC: reparto por sesión (UNKNOWN se trata como AMC, la
   política conservadora del repo), día de la semana y las cuatro temporadas
   de resultados.
4. El SUE de analistas (Livnat-Mendenhall 2006) sobre las sorpresas reales,
   incluida la demostración práctica de por qué el suelo de sigma
   ``max(sigma, 0.005·P)`` es obligatorio (sin él, una sigma degenerada
   fabrica SUEs de 10¹⁴).

## Reproducibilidad

Todos los componentes estocásticos aceptan `seed` y los scripts la exponen como
argumento (`--seed`): misma semilla → misma salida, byte a byte. Las cifras de
los ejemplos son **sintéticas o históricas**; nada aquí es una recomendación de
inversión.
