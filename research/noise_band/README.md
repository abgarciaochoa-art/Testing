# Noise Band: romper el "cono de ruido" intradía del S&P 500

Cada día nace en la apertura un cono que se abre con las horas: el movimiento "normal" desde
la apertura a esa hora del día. Mientras el precio está dentro, se considera ruido; si lo
rompe, se apuesta a que sigue en esa dirección (momentum intradía). La idea viene de
Zarattini, Aziz y Barbon (2024), "Beat the Market: An Effective Intraday Momentum Strategy
for S&P500 ETF (SPY)", que la llaman *Noise Area*.

> **Estado (ronda 1, datos 2005 → may-2020; costes de SPY con órdenes a mercado, ~0,55 pb
> por lado).**
> - **La versión del artículo (largo y corto) es buena, pero no tanto como se publicó.** Con
>   su palanca (volatilidad del 2 %, máximo 4×), desde 2007 da un **11,0 % al año**, con una
>   peor caída del -34 %, frente al 7,5 % de SPY (y su -55 %). Sin palanca, un 6,8 % con una
>   peor caída de solo el -12 % (Sharpe 0,77 frente a 0,50). Gana sobre todo en los pánicos:
>   +60 % en 2008 y +24 % en 2018.
> - **La réplica es correcta.** Con los costes del propio artículo (0,0045 $ por acción de
>   SPY: 0,29 pb por lado de media) sale un 14,1 % al año en 2007-2020, y sin costes, un
>   17,7 %. El ~20 % publicado incluye 2020-2024, que aquí no hay, y supone costes aún más
>   bajos que los de una orden a mercado.
> - **Los costes lo deciden todo:** cada 0,1 pb por lado le quita ~1,4 puntos al año con
>   palanca. Con 1 pb por lado se queda en un 4,6 %; con 2 pb pierde dinero.
> - **Solo largo con stop y objetivo** (la idea original): entre el 1,4 % y el 4 % al año sin
>   palanca y entre el 2 % y el 6,5 % con ella; el stop y el objetivo no ayudan (el objetivo
>   casi nunca se toca) y sin cortos se pierde lo mejor: la ganancia en las crisis.
> - **Dejar correr varios días:** inconsistente (-1,0 % al año en 2005-2012, +5,3 % en
>   2013-2020) y con caídas del -35 % (2008: -26 %).
> - **Como complemento intradía del nivel 1 de caídas** (futuros, sin dormir): correlación
>   mensual de **-0,47**; sube el nivel 1 del 16,6 % al **23,7 %** al año y baja su peor caída
>   del -45 % al **-29 %** (2005 → may-2020). En la crisis de 2008 se queda en -4 % en vez de
>   -42 %; en el Covid no ayudó.
> - **Evidencia C** (marco del dossier): positiva y estable en sus parámetros vecinos, pero
>   los parámetros son los del artículo (elegidos con 2007-2024), depende de costes muy bajos
>   y **faltan datos de 2020 a 2026** para validarla después de su publicación.

## Datos

| Comprobación | Resultado |
|---|---|
| Fuente | repositorio público `FutureSharks/financial-data`: OANDA SPX500_USD (CFD sobre el S&P 500, precio medio), velas de 1 minuto en UTC |
| Periodo | 3-ene-2005 → 13-may-2020: 3.846 sesiones (la red de este entorno no deja descargar fuentes más recientes) |
| Sesión | 9:30-16:00 de Nueva York: 390 velas por día; 95,5 % con cotización, el resto se rellena con el último cierre |
| Días descartados | 23 sin cotización al abrir o con menos de 200 velas (entre ellos el 9 y el 16 de marzo de 2020, con la apertura suspendida) |
| Frente a SPY diario | correlación de 0,988 en la rentabilidad de apertura a cierre y de 0,979 en la de la noche; diferencia media +0,2 pb |
| Frente a HistData (SPXUSD, otra fuente) | correlación de 0,998 (2010-2018) |
| Zona horaria | comprobada con el pico de actividad: 14:30 UTC en invierno y 13:30 en verano (las 9:30 de Nueva York) |
| El cono | movimiento medio desde la apertura: 0,22 % a las 10:00, 0,34 % a las 11:00, 0,50 % a las 14:30 y 0,62 % al cierre |

El CFD sigue a los futuros: en días de pánico su "apertura" es más realista que la oficial
del índice (que mezcla precios viejos de acciones que aún no han abierto).

## Reglas (las del artículo)

Para cada día d y cada minuto k de la sesión:

1. **sigma(d, k)** = media, en las 14 sesiones anteriores, de |precio a esa hora / apertura - 1|.
2. **Banda superior** = max(apertura, cierre anterior) × (1 + sigma); **banda inferior** =
   min(apertura, cierre anterior) × (1 - sigma). Así el cono absorbe el hueco de la noche.
3. **Controles** a las 10:00, 10:30, …, 15:30: si el precio está sobre la banda superior,
   largo; bajo la inferior, corto. Se ejecuta a la apertura del minuto siguiente.
4. **Salida** en un control si el precio vuelve dentro del cono o cruza la VWAP del día (stop
   móvil: para un largo, max(banda superior, VWAP)); todo se cierra al final del día.
5. **Tamaño:** 1× (todo el capital cuando hay posición) o el del artículo: volatilidad objetivo
   del 2 % diario con la de las 14 sesiones anteriores, máximo 4×.
6. **Costes por lado:** base, SPY con órdenes a mercado: 0,0035 $ de comisión + medio céntimo
   de horquilla = 0,0085 $ por acción (SPY cotiza a ~1/10 del índice: 0,25-1,3 pb, 0,55 de
   media). Se prueban también los del artículo (0,0045 $ por acción) y 0-2 pb fijos.

**Variantes solo largo** (la idea original): entrada igual, solo por arriba, y como salida
un stop dentro de la vela en la banda inferior o en el vértice del cono (max(apertura,
cierre anterior)), con o sin objetivo de +2 %, o hasta el cierre. **Dejar correr:** la
posición duerme y se sale cuando el precio rompe el cono por abajo. Todas fijadas de
antemano (`ronda1.py`), sin optimizar nada.

## Resultados sin palanca (costes de SPY a mercado)

| Variante | 2005-2020 | Desde 2007 | Validación 2005-12 | Búsqueda 2013-20 | Mitad 2013-16 | Mitad 2017-20 | Peor caída | Sharpe | Operaciones al año | Acierto |
|---|---|---|---|---|---|---|---|---|---|---|
| Artículo (largo y corto) | 5,8 % | 6,8 % | 9,4 % | 2,1 % | -0,3 % | 5,1 % | -12 % | 0,77 | 237 | 36 % |
| Solo largo, salidas del artículo | 2,8 % | 3,2 % | 4,3 % | 1,3 % | 0,9 % | 1,9 % | -7 % | 0,63 | 116 | 39 % |
| Solo largo, stop en la banda inferior | 1,8 % | 2,6 % | 1,0 % | 2,7 % | 2,8 % | 2,6 % | -13 % | 0,32 | 83 | 54 % |
| Solo largo, stop en el vértice del cono | 2,3 % | 2,8 % | 2,2 % | 2,4 % | 2,2 % | 2,7 % | -8 % | 0,44 | 89 | 45 % |
| Solo largo, stop en la banda inferior y objetivo +2 % | 1,4 % | 2,0 % | 0,3 % | 2,5 % | 2,8 % | 2,2 % | -13 % | 0,26 | 83 | 54 % |
| Solo largo, sin stop, hasta el cierre | 3,2 % | 4,1 % | 4,3 % | 2,0 % | 3,7 % | 0,0 % | -9 % | 0,52 | 82 | 55 % |
| Solo largo, dejar correr (varios días) | 2,2 % | 3,1 % | -1,0 % | 5,3 % | 1,7 % | 9,8 % | -35 % | 0,24 | 46 | 42 % |
| Solo largo, dejar correr con stop en la banda | 0,3 % | 1,1 % | -2,9 % | 3,3 % | 1,5 % | 5,4 % | -25 % | 0,08 | 63 | 46 % |
| Solo largo, stop en la banda, entrada en cualquier minuto | 1,9 % | 2,6 % | 1,6 % | 2,1 % | 2,9 % | 1,2 % | -16 % | 0,28 | 108 | 52 % |
| Solo largo, dejar correr, entrada en cualquier minuto | 2,1 % | 2,8 % | -1,1 % | 5,1 % | 2,3 % | 8,5 % | -46 % | 0,22 | 53 | 43 % |
| Referencia: siempre largo de apertura a cierre | -0,9 % | -0,6 % | -3,6 % | 1,9 % | 3,6 % | -0,0 % | -45 % | 0,01 | — | — |
| Referencia: índice, comprar y mantener (sin dividendos) | 6,0 % | 5,3 % | 2,5 % | 9,3 % | 11,3 % | 6,9 % | -57 % | 0,40 | — | — |
| Referencia: SPY con dividendos | 8,1 % | 7,5 % | 4,7 % | 11,5 % | 13,5 % | 8,9 % | -55 % | 0,50 | — | — |

**Con la palanca del artículo** (volatilidad del 2 %, máximo 4×):

| Variante | 2005-2020 | Desde 2007 | Validación 2005-12 | Búsqueda 2013-20 | Mitad 2013-16 | Mitad 2017-20 | Peor caída | Sharpe |
|---|---|---|---|---|---|---|---|---|
| Artículo (largo y corto) | 9,1 % | **11,0 %** | 11,6 % | 6,6 % | -0,0 % | 15,7 % | -34 % | 0,66 |
| Solo largo, salidas del artículo | 4,8 % | 5,5 % | 5,2 % | 4,4 % | 3,3 % | 6,3 % | -12 % | 0,59 |
| Solo largo, stop en el vértice del cono | 4,3 % | 5,7 % | 2,2 % | 6,7 % | 6,8 % | 6,9 % | -18 % | 0,48 |
| Solo largo, sin stop, hasta el cierre | 4,0 % | 6,5 % | 2,5 % | 5,5 % | 10,4 % | 0,2 % | -29 % | 0,39 |
| Solo largo, dejar correr (varios días) | 0,1 % | 2,4 % | -4,2 % | 3,7 % | -7,2 % | 18,5 % | -51 % | 0,11 |

- Estar siempre comprado **dentro del día** no gana nada: la prima de las acciones se cobra
  de noche. Noise Band sí elige bien cuándo estar dentro, pero su ventaja por operación es
  pequeña (+3,6 pb antes de costes; t de Newey-West de 4,6).
- La versión del artículo pasó cuatro años planos (2013-2016) y gana en los pánicos.

## ¿Fallamos? Por qué el artículo da más (costes, 2007-2020)

| Costes por lado | Artículo 1× | Artículo con su palanca | Peor caída con palanca | Sharpe con palanca | Solo largo, artículo, 1× |
|---|---|---|---|---|---|
| 0 pb | 9,5 % | **17,7 %** | -27 % | 1,18 | 4,5 % |
| 0,2 pb | 8,5 % | 14,9 % | -31 % | 1,02 | 4,0 % |
| 0,3 pb | 8,0 % | 13,6 % | -32 % | 0,94 | 3,8 % |
| 0,5 pb | 7,0 % | 10,9 % | -36 % | 0,78 | 3,3 % |
| 0,75 pb | 5,7 % | 7,7 % | -40 % | 0,57 | 2,7 % |
| 1 pb | 4,5 % | 4,6 % | -47 % | 0,37 | 2,1 % |
| Los del artículo (0,0045 $ por acción, 0,29 pb de media) | 8,1 % | **14,1 %** | -31 % | 0,97 | 3,8 % |
| SPY a mercado (0,0085 $ por acción, 0,55 pb de media) | 6,8 % | **11,0 %** | -34 % | 0,78 | 3,2 % |

1. **La réplica funciona:** las bandas, entradas y salidas coinciden hechas a mano, y con los
   costes del artículo sale un 14,1 % al año en 2007-2020 (17,7 % sin costes).
2. **Los costes del artículo son muy bajos:** 0,001 $ de deslizamiento por acción supone
   casi comprar al precio medio; con una orden a mercado se paga medio céntimo de horquilla.
   Con unas 240 operaciones al año, esa diferencia pesa ~3 puntos al año con palanca.
3. **El periodo:** el artículo llega hasta 2024 e incluye años de mucho movimiento (2020 y
   2022), y aquí los datos terminan en mayo de 2020. Su ~20 % no se puede comprobar sin ellos.
4. **Pequeñas diferencias de ejecución:** aquí se opera el minuto siguiente al control con el
   CFD (que sigue a los futuros) y la VWAP usa el número de cotizaciones como volumen.
5. **Un error mío al principio:** la primera pasada usó 1 pb por lado (el coste de una acción
   suelta, no el de SPY o los futuros), y eso la dejaba en un 4,6 % al año.

## Año a año (costes de SPY a mercado)

| Año | Artículo 1× | Artículo, palanca | Solo largo, artículo | Solo largo hasta el cierre | Solo largo, dejar correr | SPY |
|---|---|---|---|---|---|---|
| 2005 | -3,3 % | -8,8 % | -1,6 % | -4,8 % | -6,7 % | +5,3 % |
| 2006 | +1,7 % | +4,8 % | +2,3 % | -0,1 % | +1,3 % | +15,8 % |
| 2007 | +1,8 % | +11,4 % | +0,6 % | +6,2 % | +7,7 % | +5,1 % |
| 2008 | **+59,9 %** | **+58,3 %** | +15,3 % | +21,0 % | -25,8 % | -36,8 % |
| 2009 | +12,0 % | +13,4 % | +9,7 % | +7,2 % | +16,0 % | +26,4 % |
| 2010 | -2,3 % | -5,9 % | +1,2 % | -0,3 % | +4,7 % | +15,1 % |
| 2011 | +11,2 % | +19,4 % | +0,7 % | +2,7 % | -1,7 % | +1,9 % |
| 2012 | +4,6 % | +11,0 % | +6,8 % | +4,0 % | +2,2 % | +16,0 % |
| 2013 | +0,8 % | +2,9 % | +1,7 % | +6,5 % | +13,9 % | +31,7 % |
| 2014 | +0,9 % | +4,7 % | -1,5 % | -1,6 % | +2,9 % | +14,0 % |
| 2015 | +4,2 % | +15,2 % | +1,7 % | +5,2 % | -4,4 % | +1,2 % |
| 2016 | -6,8 % | -19,7 % | +1,8 % | +5,1 % | -1,2 % | +12,0 % |
| 2017 | -2,8 % | -10,7 % | +1,1 % | +0,8 % | +3,9 % | +21,7 % |
| 2018 | **+23,6 %** | **+60,5 %** | +4,7 % | -4,0 % | +12,5 % | -4,6 % |
| 2019 | +0,3 % | +5,7 % | +1,1 % | +2,5 % | +10,1 % | +31,2 % |
| 2020 (hasta 13-may) | -2,4 % | +5,9 % | -0,9 % | +0,7 % | +6,2 % | -12,0 % |

## Robustez (`robustez.py`; sin palanca, costes de SPY a mercado)

| Prueba | Artículo | Solo largo, artículo | Solo largo hasta el cierre |
|---|---|---|---|
| sigma de 10 / 14 / 20 sesiones | 5,3 / 5,8 / 6,7 % | 3,0 / 2,8 / 3,3 % | 3,7 / 3,2 / 3,3 % |
| Controles cada 15 / 30 / 60 minutos | 3,8 / 5,8 / 4,3 % | 2,1 / 2,8 / 2,2 % | 2,7 / 3,2 / 3,1 % |
| OANDA frente a HistData (2010-2018, días comunes) | 4,0 % / 3,1 % | 2,1 % / 1,7 % | 2,3 % / 2,1 % |

- La ventana de sigma está en meseta; controlar cada 15 minutos multiplica las operaciones y
  los costes; cada 30 (el artículo) es lo mejor.
- Con la otra fuente de datos sale del mismo orden (algo peor: HistData no trae volumen y su
  VWAP pondera igual cada minuto).

## Como complemento del nivel 1 de caídas (`combinar.py`)

El nivel 1 está siempre invertido en acciones. Noise Band se pondría encima con futuros, solo
dentro del día (no duerme, así que no hay palanca por la noche): la rentabilidad de la mezcla
es la del nivel 1 más el peso × la de Noise Band. Ventana 24-ene-2005 → 13-may-2020, con el
calendario del nivel 1 (en los 23 días sin datos intradía Noise Band no opera).

| Mezcla (costes de SPY a mercado) | CAGR | Peor caída | Sharpe | Calmar |
|---|---|---|---|---|
| Nivel 1 solo | 16,6 % | -45 % | 0,74 | 0,37 |
| + Noise Band artículo (largo y corto) 1×, peso 0,5 | 20,2 % | -33 % | 0,86 | 0,60 |
| + Noise Band artículo (largo y corto) 1×, peso 1 | **23,7 %** | **-29 %** | **0,96** | **0,83** |
| + Noise Band artículo con su palanca, peso 1 | 27,8 % | -29 % | 1,02 | 0,96 |
| + solo largo con las salidas del artículo, peso 1 | 19,6 % | -39 % | 0,81 | 0,50 |
| + solo largo hasta el cierre, peso 1 | 19,7 % | -40 % | 0,79 | 0,50 |

- Con los costes del artículo, la mezcla de peso 1 sin palanca llega al 25,2 %; con 1 pb por
  lado, al 21,1 %; con 2 pb, al 15,5 % (por debajo del nivel 1 solo, aunque con mucha menos
  caída).
- La versión con palanca, encima del nivel 1, puede llevar la exposición dentro del día hasta
  5× el capital: es la más rentable sobre el papel y la más exigente de margen.
- La correlación mensual con el nivel 1 es de -0,47 en la versión del artículo y de ~0 en las
  solo largo: **los cortos son los que protegen**.

| Tramo de crisis | Nivel 1 solo | + Noise Band (peso 0,5) | + Noise Band (peso 1) | Noise Band solo (1×) |
|---|---|---|---|---|
| Crisis financiera (oct-07 → mar-09) | -41,7 % | -24,9 % | **-4,0 %** | +61,4 % |
| Flash crash (2010) | -10,6 % | -10,5 % | -10,4 % | +0,1 % |
| Deuda de EE. UU. y del euro (2011) | -15,5 % | -12,1 % | -8,6 % | +7,8 % |
| China y petróleo (2015-16) | +4,7 % | +6,3 % | +7,9 % | +2,9 % |
| Cuarto trimestre de 2018 | -29,0 % | -25,3 % | -21,5 % | +10,0 % |
| Covid (feb-mar 2020) | -26,5 % | -27,6 % | -28,6 % | -2,9 % |

## Conclusiones

1. **No fallamos en la réplica:** con los costes del artículo da un 14,1 % al año en
   2007-2020 (17,7 % sin costes); el resto de la distancia con lo publicado está en los años
   2020-2024 y en unos costes muy optimistas.
2. **Con costes realistas es una estrategia correcta, no espectacular:** 11 % al año con su
   palanca (SPY: 7,5 %), con menos caída que SPY, ganando en los pánicos y pasando años
   planos en los mercados tranquilos.
3. **La versión solo largo con stop y objetivo no aporta:** quitar los cortos le quita lo
   mejor, y el stop y el objetivo no mejoran la salida al cierre.
4. **Dejar correr varios días** cambia de signo entre ventanas: no es robusto.
5. **Su mejor uso es combinarla:** gana cuando la compra de caídas pierde. Como capa intradía
   con futuros sobre el nivel 1, mejora rentabilidad y caída a la vez, si los costes se
   mantienen por debajo de ~1 pb por lado.

## Siguientes pasos

1. **Datos de jun-2020 a oct-2026** (futuros ES/MES o SPY en velas de 1 minuto) para
   validarla después de su publicación. Esta sesión no puede descargarlos: hace falta
   permitir un dominio de datos en la red del entorno o subir los archivos.
2. **Costes reales de futuros** (MES/ES: tick de 0,25 puntos y comisión) y ejecución al precio
   del control.
3. **Decidir si se aceptan cortos** (con futuros): son los que dan la protección.
4. Si aguanta, añadirla al marco del dossier como capa intradía (`NB_LC`) y probarla en papel
   junto al nivel 1.

## Reproducir

```bash
python research/noise_band/datos.py      # clona los datos (≈330 MB) y prepara la sesión regular
python research/noise_band/ronda1.py     # variantes, costes y ventanas
python research/noise_band/costes.py     # comparación fina de costes con el modelo del artículo
python research/noise_band/robustez.py   # sigma, controles y otra fuente
python research/noise_band/combinar.py   # como complemento del nivel 1 (usa research/caidas/dossier/datos/diario.csv)
```

El motor (`motor.py`, numba) tarda ~6 ms por backtest de 15 años en velas de 1 minuto.
