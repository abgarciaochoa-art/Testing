# Tendencia intradía: cómo identificarla y cómo replicarlo

Pregunta: a una hora dada (10:00, 10:30, 11:30…), ¿se puede saber si una acción va a seguir en
la dirección que lleva hasta el cierre? Se prueban unos 20 rasgos y se construye un "semáforo"
simple y replicable.

Informe visual: `informe.html` (también publicado como artifact). Indicador para TradingView:
`semaforo.pine`. Versión en Python para tus datos: `replicar.py`.

## Conclusiones

1. **Días de tendencia** (cuerpo ≥ 60 % del rango): el 32 % de los días en las 80 acciones más
   líquidas del S&P 500 (2024-2026), el 16,8 % alcistas y el 15,4 % bajistas.
2. **Lo recorrido no predice lo que queda.** Movimiento desde la apertura, VWAP, rango de
   apertura, eficiencia, regresión, media de 60 min, posición en el rango, volumen neto, hueco,
   día anterior, tendencia de 20 días, mercado y fuerza relativa: en las acciones todos tienen
   una correlación de ±0,03 con el movimiento hasta el cierre. Por la tarde es algo negativa.
   En el Nasdaq-100 (2005-2020) y el bitcóin hay algo de continuación por la mañana.
3. **Un modelo de árboles con todos los rasgos tampoco funciona.** Entrenado hasta junio de 2025
   y medido después, no supera a una regla simple (correlación fuera de muestra de 0,02).
4. **Lo que separa la tendencia del ruido es que el valor esté en juego.** Seguir la dirección
   del día desde las 10:30 hasta el cierre gana, en ATR:

   | Volumen relativo | Resultado |
   |---|---|
   | menos de 2 | 0 |
   | de 2 a 3 | +0,03 |
   | 3 o más | +0,06 (55 % de acierto) |

   El tamaño del movimiento (sigma frente a la banda de ruido) ayuda menos, solo por encima de
   2-3 sigmas.
5. **Semáforo** (`semaforo.py`, `replicar.py`, `semaforo.pine`):

   | Estado | Regla |
   |---|---|
   | rojo | \|sigma\| < 1 |
   | ámbar | se mueve, pero sin rvol ≥ 2 o al otro lado de la VWAP |
   | verde | \|sigma\| ≥ 1, rvol ≥ 2 y del lado de la VWAP |
   | verde fuerte | lo mismo con rvol ≥ 3 |

   En las 80 acciones, entrando la primera vez que se pone verde (10:00-14:30), con stop de
   0,25 ATR y candado (a +1,8R, stop a +1R, luego dinámico a 0,8R), el resultado neto por
   operación es:

   | Señal | Neto | t | Operaciones |
   |---|---|---|---|
   | Verde | +0,076R | 2,0 | ~6 por 100 valores y día |
   | Verde fuerte | +0,12R | 2,1 | ~2 por 100 valores y día |
   | A favor del día, sin filtro | −0,006R | — | — |

6. **La superficie de umbrales es suave.** La ventaja sube con el volumen relativo exigido en
   todas las filas de sigma (`robustez.py`). El filtro de mercado (SPY a favor) no mejora.
7. **Pero se ha estrechado.** Por años, el verde da +0,165R en 2024, +0,028R en 2025 y
   +0,028R en 2026. 7 de 12 trimestres son positivos.
8. **Otros mercados y matices:**
   - Bitcóin: el verde funciona dejando correr hasta el cierre (+0,41R neto, t 2,7).
   - Nasdaq-100: no funciona. En el índice, los movimientos grandes con mucho volumen revierten.
   - Hora: las señales de 10:30-11:30 rinden más; las de después de las 12:30, nada.
   - Nvidia casi nunca se pone verde (31 veces en casi 3 años): ya mueve tanto volumen que
     doblarlo es raro. El semáforo sirve para buscar cada día qué acción está en juego.
9. **El verde predice sobre todo el recorrido.** El resto del día se mueve un 45-90 % más que en
   rojo, con una inclinación direccional modesta. Por eso rinde más con stop corto y dejando
   correr que esperando al cierre sin stop.

## Datos

| Conjunto | Fuente | Vela | Periodo |
|---|---|---|---|
| `acc_h1` | Yahoo, 80 acciones más líquidas + SPY | 1 hora | ene-2024 → nov-2026 (706 sesiones) |
| `acc_m5` | Yahoo, mismas acciones | 5 min | sep → nov 2026 (59 sesiones) |
| `etf_h1` | Yahoo, SPY, QQQ, IWM, SMH, XLK | 1 hora | ene-2024 → nov-2026 |
| `NAS100`, `BTC` | paneles del estudio de la horquilla | 1 min | 2005-2020 / 2022-2026 |
| `NVDA_1m` | Dukascopy (CFD) | 1 min | 2026 |

`NVDA_1m` sirve para los rasgos de precio, pero no para el volumen relativo, porque su volumen
es el del CFD. Nvidia se analiza con Yahoo, dentro de las 80.

## Archivos

| Archivo | Qué hace |
|---|---|
| `descargar.py` | Yahoo de 1 hora (730 días) y de 5 minutos (60 días) de 85 valores |
| `paneles.py` | matrices días × velas y rasgos diarios (ATR, medias) |
| `rasgos.py` | los 20 rasgos y los objetivos en una hora de control |
| `evaluacion.py` | correlación, ventaja, acierto y estabilidad por rasgo, hora y conjunto (`resultados/rasgos.csv`) |
| `semaforo.py` | reglas del semáforo sobre paneles |
| `prueba_semaforo.py` | deriva por estado y operativa con stop, candado y costes |
| `robustez.py` | rejilla de umbrales sigma × volumen relativo |
| `anatomia.py` | frecuencia y huella de los días de tendencia, resto del día y comparación con el modelo de árboles |
| `replicar.py` | versión autónoma en pandas; coincide con `semaforo.py` en el 100 % de las velas |
| `semaforo.pine` | indicador para TradingView (Pine v5); sin compilar en este entorno |
| `datos_informe.py` | JSON del informe |

## Reproducir

```bash
python research/tendencia_intradia/descargar.py
python research/tendencia_intradia/paneles.py
python research/tendencia_intradia/evaluacion.py
python research/tendencia_intradia/prueba_semaforo.py
python research/tendencia_intradia/robustez.py
python research/tendencia_intradia/anatomia.py
python research/tendencia_intradia/replicar.py      # comprobación de la versión autónoma
```

Los conjuntos `NAS100`, `BTC` y `NVDA_1m` necesitan antes los paneles de
`research/bracket_intradia` (ver su README).

## Límites

- **Velas de 1 hora:** el grueso del estudio las usa, así que las entradas son a las 10:30,
  11:30… Con velas de 5 minutos solo hay 59 sesiones.
- **Universo de hoy:** las 80 acciones son las más líquidas ahora (sesgo de supervivencia leve).
- **Sin noticias:** no se distinguen los días con noticias o resultados.
- **Pruebas múltiples:** unas 200 combinaciones de rasgo, hora y conjunto. Las t de 2 a 3 del
  verde son evidencia moderada.
