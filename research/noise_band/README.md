# Noise Band: romper el "cono de ruido" intradía del S&P 500

Cada día nace en la apertura un cono que se abre con las horas: el movimiento "normal" desde
la apertura a esa hora del día. Mientras el precio está dentro, se considera ruido; si lo
rompe, se apuesta a que sigue en esa dirección (momentum intradía). La idea viene de
Zarattini, Aziz y Barbon (2024), "Beat the Market: An Effective Intraday Momentum Strategy
for S&P500 ETF (SPY)", que la llaman *Noise Area*.

> **Estado (ronda 1, datos 2005 → may-2020).**
> - **Sola, no compensa:** con costes realistas (1 pb por lado, futuros del S&P o SPY) y sin
>   palanca, la versión del artículo da un 3,6 % al año y las variantes solo largo, entre el
>   1 % y el 2,4 %, frente al 8,1 % de SPY. La ventaja antes de costes es real (+3,6 pb por
>   operación, t de Newey-West 4,6), pero con ~240 operaciones al año los costes se la comen:
>   sin costes y con la palanca del artículo daría un 16,7 %, y con 1 pb, un 2,8 %.
> - **Solo largo con stop y objetivo** (la idea original): el stop y el objetivo no mejoran
>   nada (el objetivo casi nunca se toca), y quitar los cortos quita lo mejor de la
>   estrategia: gana en los desplomes.
> - **Dejar correr varios días:** inconsistente (-1,3 % al año en 2005-2012, +4,7 % en
>   2013-2020) y con caídas del -35 % (2008: -26 %).
> - **Lo prometedor: como complemento intradía del nivel 1 de caídas.** La versión del
>   artículo (largo y corto) tiene una correlación mensual de **-0,46** con el nivel 1 y gana
>   justo en sus peores momentos (2008: +58 %; cuarto trimestre de 2018: +9 %). Superpuesta
>   con futuros solo dentro del día, sube el nivel 1 del 16,6 % al **21,1 %** al año y baja su
>   peor caída del -45 % al **-29 %** (2005 → may-2020, 1 pb por lado). Depende de que los
>   costes no pasen de ~1 pb por lado y de dos años (2008 y 2018); en el Covid no ayudó.
> - **Evidencia C** (marco del dossier): positiva, pero los parámetros son los del artículo
>   (elegidos con 2007-2024, que cubre casi toda nuestra muestra), es muy sensible a los
>   costes y **faltan datos de 2020 a 2026** para validarla después de su publicación.

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
6. **Costes:** 1 pb por lado como base (futuros ES/MES o SPY con un bróker barato); se prueba
   0; 0,5 y 2.

**Variantes solo largo** (la idea original): entrada igual, solo por arriba, y como salida
un stop dentro de la vela en la banda inferior o en el vértice del cono (max(apertura,
cierre anterior)), con o sin objetivo de +2 %, o hasta el cierre. **Dejar correr:** la
posición duerme y se sale cuando el precio rompe el cono por abajo. Todas fijadas de
antemano (`ronda1.py`), sin optimizar nada.

## Resultados (1×, 1 pb por lado)

| Variante | 2005-2020 | Validación 2005-12 | Búsqueda 2013-20 | Mitad 2013-16 | Mitad 2017-20 | Peor caída | Sharpe | Operaciones al año | Acierto | Días con posición |
|---|---|---|---|---|---|---|---|---|---|---|
| Artículo (largo y corto) | 3,6 % | 7,8 % | -0,7 % | -2,9 % | 2,0 % | -19 % | 0,49 | 237 | 33 % | 62 % |
| Solo largo, salidas del artículo | 1,8 % | 3,5 % | -0,1 % | -0,4 % | 0,4 % | -7 % | 0,40 | 116 | 35 % | 33 % |
| Solo largo, stop en la banda inferior | 1,1 % | 0,5 % | 1,7 % | 1,9 % | 1,5 % | -14 % | 0,20 | 83 | 53 % | 33 % |
| Solo largo, stop en el vértice del cono | 1,5 % | 1,7 % | 1,4 % | 1,3 % | 1,6 % | -9 % | 0,30 | 89 | 44 % | 33 % |
| Solo largo, stop en la banda inferior y objetivo +2 % | 0,6 % | -0,2 % | 1,5 % | 1,9 % | 1,1 % | -14 % | 0,14 | 83 | 53 % | 33 % |
| Solo largo, sin stop, hasta el cierre | 2,4 % | 3,8 % | 1,0 % | 2,7 % | -1,0 % | -11 % | 0,40 | 82 | 54 % | 33 % |
| Solo largo, dejar correr (varios días) | 1,8 % | -1,3 % | 4,7 % | 1,1 % | 9,2 % | -35 % | 0,21 | 46 | 42 % | 72 % |
| Solo largo, dejar correr con stop en la banda | -0,3 % | -3,3 % | 2,5 % | 0,8 % | 4,6 % | -27 % | 0,02 | 63 | 45 % | 53 % |
| Solo largo, stop en la banda, entrada en cualquier minuto | 0,9 % | 0,9 % | 0,8 % | 1,7 % | -0,2 % | -17 % | 0,15 | 108 | 51 % | 43 % |
| Solo largo, dejar correr, entrada en cualquier minuto | 1,6 % | -1,5 % | 4,5 % | 1,7 % | 7,9 % | -47 % | 0,19 | 53 | 43 % | 79 % |
| Referencia: siempre largo de apertura a cierre | -3,1 % | -5,0 % | -1,2 % | 0,8 % | -3,4 % | -50 % | -0,13 | — | — | — |
| Referencia: índice, comprar y mantener (sin dividendos) | 6,0 % | 2,5 % | 9,3 % | 11,3 % | 6,9 % | -57 % | 0,40 | — | — | — |
| Referencia: SPY con dividendos | 8,1 % | 4,7 % | 11,5 % | 13,5 % | 8,9 % | -55 % | 0,50 | — | — | — |

- Estar siempre comprado **dentro del día** pierde dinero (-3,1 % al año): la prima de las
  acciones se gana de noche. Noise Band elige bien los días y las horas (más que eso), pero
  su ventaja es pequeña.
- Con la palanca del artículo (volatilidad del 2 %, máximo 4×) todas empeoran con 1 pb: el
  artículo, 2,8 % con una peor caída del -47 %.

## Los costes deciden (CAGR 2005-2020)

| Variante | 0 pb | 0,5 pb | 1 pb | 2 pb |
|---|---|---|---|---|
| Artículo, 1× | 8,6 % | 6,1 % | 3,6 % | -1,2 % |
| Artículo, palanca del artículo | **16,7 %** | 9,5 % | 2,8 % | -9,5 % |
| Solo largo, salidas del artículo, 1× | 4,2 % | 3,0 % | 1,8 % | -0,6 % |
| Solo largo, sin stop, hasta el cierre, 1× | 4,1 % | 3,3 % | 2,4 % | 0,8 % |
| Solo largo, dejar correr, 1× | 2,7 % | 2,3 % | 1,8 % | 0,8 % |

Sin costes y con su palanca, el artículo reproduce el orden de lo publicado (en torno a un
20 % al año en 2007-2024, según sus autores). Cada 0,5 pb por lado le quita ~2,5 puntos al año sin palanca y ~7 con
ella. Para que tenga sentido hay que operar futuros (ES o MES) con un coste total por lado de
0,5-1 pb.

## Año a año

| Año | Artículo 1× | Artículo, palanca | Solo largo, artículo | Solo largo hasta el cierre | Solo largo, dejar correr | SPY |
|---|---|---|---|---|---|---|
| 2005 | -4,8 % | -13,4 % | -2,4 % | -5,4 % | -7,1 % | +5,3 % |
| 2006 | -0,2 % | -1,7 % | +1,3 % | -0,8 % | +0,9 % | +15,8 % |
| 2007 | -0,5 % | +5,2 % | -0,5 % | +5,5 % | +7,3 % | +5,1 % |
| 2008 | **+57,8 %** | **+55,1 %** | +14,6 % | +20,5 % | -26,0 % | -36,8 % |
| 2009 | +11,6 % | +12,5 % | +9,5 % | +7,1 % | +15,9 % | +26,4 % |
| 2010 | -3,3 % | -8,2 % | +0,7 % | -0,7 % | +4,5 % | +15,1 % |
| 2011 | +9,4 % | +15,4 % | -0,1 % | +2,1 % | -2,1 % | +1,9 % |
| 2012 | +2,8 % | +5,8 % | +5,9 % | +3,4 % | +1,9 % | +16,0 % |
| 2013 | -1,5 % | -4,4 % | +0,4 % | +5,6 % | +13,4 % | +31,7 % |
| 2014 | -1,5 % | -3,3 % | -2,7 % | -2,4 % | +2,4 % | +14,0 % |
| 2015 | +1,2 % | +7,2 % | +0,4 % | +4,2 % | -4,9 % | +1,2 % |
| 2016 | -9,4 % | -26,1 % | +0,4 % | +4,1 % | -1,8 % | +12,0 % |
| 2017 | -5,5 % | -20,2 % | -0,3 % | -0,2 % | +3,4 % | +21,7 % |
| 2018 | **+19,8 %** | **+47,1 %** | +3,0 % | -5,1 % | +11,9 % | -4,6 % |
| 2019 | -2,5 % | -3,2 % | -0,3 % | +1,5 % | +9,5 % | +31,2 % |
| 2020 (hasta 13-may) | -3,6 % | +4,0 % | -1,4 % | +0,4 % | +6,0 % | -12,0 % |

Es una estrategia de **crisis**: gana mucho en los años de pánico (2008, 2011, 2018) y pierde
poco a poco en los tranquilos (2016-2017: -9 % y -6 % sin palanca).

## Robustez (`robustez.py`; artículo, 1×, 1 pb)

| Prueba | Artículo | Solo largo, artículo | Solo largo hasta el cierre |
|---|---|---|---|
| sigma de 10 / 14 / 20 sesiones | 3,1 / 3,6 / 4,5 % | 2,0 / 1,8 / 2,2 % | 2,9 / 2,4 / 2,5 % |
| Controles cada 15 / 30 / 60 minutos | 0,8 / 3,6 / 2,7 % | 0,6 / 1,8 / 1,4 % | 1,9 / 2,4 / 2,4 % |
| OANDA frente a HistData (2010-2018, días comunes) | 1,5 % / 0,5 % | 0,8 % / 0,4 % | 1,4 % / 1,2 % |

- La ventana de sigma está en meseta; controlar cada 15 minutos multiplica las operaciones y
  los costes.
- Con la otra fuente de datos sale del mismo orden (algo peor: HistData no trae volumen y su
  VWAP pondera igual cada minuto).
- La ventaja diaria antes de costes es significativa: t de Newey-West de 4,6 (largo y corto)
  y 4,2 (solo largo); con 1 pb, 2,0 y 1,8.

## Como complemento del nivel 1 de caídas (`combinar.py`)

El nivel 1 está siempre invertido en acciones. Noise Band se pondría encima con futuros, solo
dentro del día (no duerme, así que no hay palanca por la noche): la rentabilidad de la mezcla
es la del nivel 1 más el peso × la de Noise Band. Ventana 24-ene-2005 → 13-may-2020, con el
calendario del nivel 1 (en los 23 días sin datos intradía Noise Band no opera).

| Mezcla (1 pb por lado) | CAGR | Peor caída | Sharpe | Calmar |
|---|---|---|---|---|
| Nivel 1 solo | 16,6 % | -45 % | 0,74 | 0,37 |
| + Noise Band artículo (largo y corto), peso 0,5 | 18,9 % | -33 % | 0,82 | 0,57 |
| + Noise Band artículo (largo y corto), peso 1 | **21,1 %** | **-29 %** | **0,88** | **0,73** |
| + Noise Band artículo con su palanca, peso 1 | 20,3 % | -29 % | 0,80 | 0,70 |
| + solo largo con las salidas del artículo, peso 1 | 18,4 % | -39 % | 0,77 | 0,47 |
| + solo largo hasta el cierre, peso 1 | 18,9 % | -40 % | 0,76 | 0,47 |

- Con 0,5 pb por lado la mezcla de peso 1 llega al 24,0 % (peor caída -29 %); con 2 pb baja al
  15,5 %, por debajo del nivel 1 solo, aunque con mucha menos caída.
- La correlación mensual con el nivel 1 es de -0,46 en la versión del artículo y de ~0 en las
  solo largo: **los cortos son los que protegen**.

| Tramo de crisis | Nivel 1 solo | + Noise Band (peso 0,5) | + Noise Band (peso 1) | Noise Band solo |
|---|---|---|---|---|
| Crisis financiera (oct-07 → mar-09) | -41,7 % | -25,6 % | **-5,7 %** | +58,5 % |
| Flash crash (2010) | -10,6 % | -10,6 % | -10,6 % | -0,2 % |
| Deuda de EE. UU. y del euro (2011) | -15,5 % | -12,4 % | -9,2 % | +7,1 % |
| China y petróleo (2015-16) | +4,7 % | +5,2 % | +5,6 % | +0,7 % |
| Cuarto trimestre de 2018 | -29,0 % | -25,6 % | -22,2 % | +9,0 % |
| Covid (feb-mar 2020) | -26,5 % | -27,7 % | -29,0 % | -3,4 % |

## Conclusiones

1. **Noise Band sola no bate a comprar y mantener SPY** con costes realistas en 2005-2020.
   Lo publicado depende de costes casi nulos y de mucha palanca.
2. **La versión solo largo con stop y objetivo no aporta**: el objetivo casi nunca salta, los
   stops no mejoran la salida al cierre y sin cortos se pierde la protección en las crisis.
3. **Dejar correr la posición varios días** cambia de signo entre ventanas: no es robusto.
4. **Su valor está en combinarla**: gana cuando la compra de caídas pierde. Como capa
   intradía con futuros sobre el nivel 1, mejora rentabilidad y caída a la vez, siempre que
   el coste por lado no pase de ~1 pb.

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
python research/noise_band/robustez.py   # sigma, controles y otra fuente
python research/noise_band/combinar.py   # como complemento del nivel 1 (usa research/caidas/dossier/datos/diario.csv)
```

El motor (`motor.py`, numba) tarda ~6 ms por backtest de 15 años en velas de 1 minuto.
