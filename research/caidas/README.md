# Comprar caídas en el S&P 500: qué funciona, qué no y por qué

Investigación sobre **acciones reales del S&P 500 con pertenencia point-in-time**
(2002-01 → 2026-10), con costes de 10 pb por lado y ejecución a la apertura siguiente.
Se probaron **~11.000 carteras** en ocho rondas, y todas cuentan en las
correcciones por minería de datos.

> **Estado actual (ronda 6).** Estrategia **D1 con entradas técnicas, solo acciones**
> (sin ETFs ni efectivo) y **reserva contracíclica**. Compra caídas ≥15 % desde el
> máximo de 20 días de acciones en tendencia, con ≥5 módulos de origen de la caída y ≥2
> banderas técnicas. El resto del capital está siempre en acciones: en calma y en
> tendencia, hasta 1,5× las 100 más líquidas (según la volatilidad de SPY, usado solo
> como señal), y el resto y fuera de tendencia, las 50 de menor volatilidad. La deuda
> trabaja en calma en la reserva y en volatilidad en las caídas.
>
> | | Nivel 1 · sin deuda | **Nivel 2 · contracíclica (máx. 1,5×)** | Nivel 3 · + caídas hasta 2× | SPY |
> |---|---|---|---|---|
> | CAGR validación 2002-sep21 | 20,4 % | **21,8 %** | 25,8 % | 8,9 % |
> | CAGR 2002-2025 | 21,4 % | **23,7 %** | 27,2 % | 9,7 % |
> | CAGR 2002-2026 | 25,8 % | **27,7 %** | 32,7 % | 10,0 % |
> | Caída máxima | -45 % | **-45 %** | -49 % | -55 % |
> | Sharpe | 0,97 | **0,99** | 0,94 | 0,60 |
>
> Sharpe deflactado: 0,59 con N bruto (~10.500) y 0,997 con N efectivo (205). Es el
> mejor del proyecto, pero según la regla del repo sigue siendo **zona gris:
> prometedora y explicable, no demostrada**.
>
> **Ronda 7 (comportamiento):** ninguna salida, tamaño, entrada escalonada ni filtro por
> rasgos mejora la estrategia fuera de muestra; se queda como está. Comprar en dos tramos
> es la mejor forma de bajar el riesgo (caída máxima -36 % en vez de -45 %, ~2 puntos
> menos al año). Ver la sección 8.
>
> **Ronda 8 (gestión dentro de la operación):** cortar perdedoras empeora de forma
> significativa (-5,6 puntos al año fuera de muestra si se vende lo que sigue en
> pérdidas al 5.º día) y añadir a las que siguen cayendo, aunque tiene ventaja por
> dólar, apenas mueve la cartera. Ver la sección 9.

## Datos: por qué fiarse de ellos

| Comprobación | Resultado |
|---|---|
| Fuente | release pública `Johnbrick123/sp500-data` (Yahoo + CRSP + Tiingo + Wiki), 1.200 tickers |
| Pertenencia | intervalos [alta, baja) por ticker, verificados contra anuncios de S&P |
| Cobertura de precios de los miembros de cada fecha | **≈100 % desde 2001** (70 % en 1996, por eso empezamos en 2002) |
| Quiebras incluidas con su trayectoria real | Lehman cierra en 0,13 $, WaMu en 0,16 $, Bear Stearns, Countrywide… |
| Contraste con la semilla del repo (`data/seed/`) | Jaccard 0,83-0,97 (sobre todo diferencias de nomenclatura) |
| Picos imposibles limpiados | 19 (subida >150 % revertida al día siguiente) |
| Conciliación de motores | coste 0: equity = producto de operaciones (1e-14); numba = referencia (exacto) |

### Auditoría de la ronda 3: dos sesgos encontrados y corregidos

1. **Precio ajustado por splits en el filtro de 5 $.** En las series de Yahoo, `close`
   viene ajustado por splits: AAPL figuraba a 0,31 $ en 2003, NVDA a 0,21 $ y AMZN a
   1,77 $. El filtro "precio ≥ 5 $" las expulsaba durante años (a NVDA hasta ~2019).
   Ahora se usa el **precio real**: en Yahoo, `close` × splits posteriores; en CRSP,
   Tiingo y Wiki el precio ya es el real. La convención de cada fuente se midió en
   las fechas de split.
2. **Deslistadas sin volumen.** Las series de CRSP, casi todas de empresas que luego
   desaparecieron, no traen volumen, y el filtro de liquidez las trataba como
   ilíquidas. **No se podían comprar.** Eso reintroducía sesgo de supervivencia en
   2002-2015: el 29 % de los miembros-día de 2003 y el 10 % de los de 2008.
   Corrección: un miembro sin dato de volumen no se excluye. Además, los rasgos
   intradía (hueco nocturno, IBS) quedan como no disponibles donde la fuente no da
   apertura ni rango.

Con esto, el universo negociable de 2003 pasa del 69 % al 97 % de los miembros.
**Todo el informe está recalculado con los datos corregidos.** Efecto: las estrategias
ingenuas de la ronda 1 empeoran claramente, y las de módulos apenas cambian (C1 incluso
mejora). Los módulos sirven justo para evitar las caídas que acaban mal.

## 1. Estudio de eventos: qué caída predice rebote (`estudio.py`)

Exceso sobre la media equiponderada de los miembros, con entrada a la apertura de t+1.
La t es de Newey-West.

| Caída | Eventos | +5 d | +20 d | +60 d | Lectura |
|---|---|---|---|---|---|
| -30 % desde el máx. de **20 d**, en tendencia (> SMA200) | 305 | +1,0 % | +4,0 % | **+5,5 %** (t 1,1) | rebote, positivo en los 3 subperiodos |
| -25 % en 20 d, en tendencia | 523 | +1,4 % (t 2,2) | +3,3 % | +2,0 % | igual idea |
| RSI(2) ≤ 5 | 185.956 | +0,08 % (t 2,0) | +0,06 % | +0,18 % | real pero diminuto: < costes |
| 4 cierres bajistas seguidos | 150.037 | +0,08 % (t 2,0) | 0,0 % | +0,07 % | ídem |
| -30 % desde el máx. de **252 d** (caída lenta) | 405.367 | -0,08 % | -0,34 % | **-1,3 %** (t -2,4) | **los perdedores lentos siguen perdiendo** |

**Importa la velocidad de la caída, no su tamaño.** Una caída brusca de una empresa que
venía subiendo suele ser una sobrerreacción y revierte. Una caída lenta desde máximos
anuales suele reflejar un deterioro y continúa: es el efecto momentum.

## 2. Ronda 1: rejilla de carteras (`rejilla.py`, `analisis.py`)

2.478 variantes. Elección con 2002-2015 y validación con 2016-2026.

| Familia | CAGR mediano 02-15 | CAGR mediano 16-26 | DD mediano |
|---|---|---|---|
| A. Caída rápida (dd 10-60 d), efectivo sin señal | 0,0 % | 4,3 % | -70 % |
| B. Reversión corta (RSI2, rachas, z-score) | 2,3 % | 2,4 % | -56 % |
| C. Caída lenta (dd 126-252 d) | -3,2 % | 7,6 % | -75 % |
| D. Caída de N días en tendencia | 3,3 % | 9,5 % | -55 % |
| E. Caída desde máximo en tendencia | 5,3 % | 10,5 % | -59 % |
| SPY | 6,2 % | 15,1 % | -55 % |

- Solo el **7,5 %** de las variantes bate a SPY.
- El ranking dentro de muestra apenas predice el de fuera: Spearman 0,18; PBO = 0,40.
- **Comprar caídas a secas no funciona.** Concentrado, acumula las quiebras de 2008
  (caídas máximas del -70 % al -90 %).

### SPY + caídas rápidas en tendencia (`finalistas.py`)

| | F1 X=15 % SMA5 K=3 | F2 X=15 % K=5 | F3 X=20 % SMA20 K=3 | F4 X=10 % K=3 | SPY |
|---|---|---|---|---|---|
| CAGR 2002-2015 | 10,1 % | 9,6 % | 9,2 % | 5,8 % | 6,2 % |
| CAGR 2016-2026 | 19,0 % | 18,5 % | 16,5 % | 22,3 % | 15,1 % |
| CAGR total | 13,9 % | 13,4 % | 12,3 % | 12,7 % | 10,0 % |
| Caída máxima | -52 % | -55 % | -56 % | -59 % | -55 % |
| CAGR con universo de hoy (**sesgado**) | 23,2 % | 21,5 % | 24,3 % | 23,7 % | |

La diferencia entre la penúltima fila y el CAGR total es lo que inventa el sesgo de
supervivencia: hasta 12 puntos de CAGR.

## 3. Ronda 2: ¿por qué cae? Módulos de origen de la caída

Metodología, igual en las rondas 2 y 3:

- **Búsqueda** solo en oct-2021 → oct-2026; **validación** en 2002 → sep-2021, que no
  se usa para elegir.
- **Rasgos PIT** (`rasgos.py`, 56 matrices, ~80 s). Grupo de pares = las 10 acciones más
  correlacionadas (252 d, recalculado cada mes, sin sesgo de supervivencia). Ejemplos:
  HD→LOW, V→MA, XOM→CVX, JPM→GS, DAL→UAL, KO→PEP.
- **Tabla de eventos** (`eventos.py`): **554.471 caídas** candidatas con rasgos y retorno
  posterior. Evaluar un módulo es un filtro de milisegundos.
- **Motores compilados** (`motor_rapido.py`, `motor_cartera.py`, numba): 0,015 s por
  backtest de 25 años, idénticos al motor de referencia.

**Qué mejora o empeora la entrada.** Se cumple en las 4 definiciones de caída y en las
dos ventanas (`modulos.py`, `modulos_v2.py`):

| Mejora la entrada | Empeora la entrada |
|---|---|
| sus **pares caen también**; ≥60 % de su grupo cae | cae **sola** en su grupo (<20 %) |
| su **gemela también cae**; sin divergencia (z > -2) | su gemela sube; spread z ≤ -3 |
| **venta general**; SPY en corrección ≥10 % | caída aislada; SPY en máximos |
| cierra en mínimos (IBS ≤ 0,2) | día de giro (IBS > 0,6) |
| momentum 12 m ≥ 30 %, **líder de su grupo**, ≥15 % sobre su SMA200 | caída lenta (goteo), rezagada |
| grandes del índice, beta alta, volatilidad propia ≥45 % | pequeñas, tranquilas (<25 % vol.) |
| hueco nocturno **sin** volumen anómalo | **noticia**: hueco ≤ -5 % con volumen ≥3× |
| | caída **extrema para esa acción** (≤ -4σ) |

**Lectura económica.** Una acción que cae arrastrada por su sector o por el mercado
suele sufrir presión de flujos (ventas forzadas, rebalanceos), y eso revierte. Una que
cae sola, con noticia o fuera de lo normal para ella, lleva información y sigue cayendo,
como en la deriva post-resultados (Da, Liu y Schaumburg, 2014). Las líderes de su grupo
rebotan más que las rezagadas.

**Pares tipo KO/PEP** (`gemelas.py`). Con correlación ≥ 0,8 y z ≤ -2/-2,5, el spread
converge +0,4-0,6 % en 20 d (t ≈ 3 en las dos ventanas). Pero como cartera largo-corto
(`motor_cartera.simular_pares`) da un 0-3 % anual (Sharpe 0,05-0,3): los costes de
cuatro patas y el préstamo de la corta se comen la ventaja. **Descartada como cartera.**
Su información sí se usa, a través de los módulos de la gemela.

### Finalistas de la ronda 2 (`finalistas2.py`)

| | C1 explicable | C2 concentrada | SPY |
|---|---|---|---|
| Regla | cae ≥7 % en 5 d en tendencia, ≥6 módulos v1, K=5, sale RSI2>70 | cae ≥15 % en 10 d, ridge, K=2, sale >SMA10 | |
| CAGR validación 2002-sep21 | **16,7 %** | 15,4 % | 8,9 % |
| CAGR total | 19,3 % | 20,1 % | 10,0 % |
| Caída máxima | -40 % | -38 % | -55 % |
| Sharpe | 0,91 | 0,82 | 0,60 |
| CAGR con costes de 30 pb | 14,1 % | 14,7 % | |

## 4. Ronda 3: módulos v2, meta-puntuación y construcción de cartera

- **Puntuación explicable `score_v2`** (`meta.py`): 11 módulos elegidos **solo con la
  búsqueda**, exigiendo que mejoren en las 4 bases y en las **dos mitades** de la
  búsqueda, para filtrar efectos de un solo episodio. Retorno neto por operación según
  el número de módulos (base: -10 % en 10 d en tendencia):

  | módulos a favor | 0 | 2 | 4 | 5 | 6 | 7 | 8 |
  |---|---|---|---|---|---|---|---|
  | búsqueda 2021-26 | -0,6 % | -0,1 % | +0,4 % | +0,3 % | +1,3 % | +1,2 % | +0,8 % |
  | **validación 2002-21** | -0,1 % | +0,1 % | +0,5 % | +0,8 % | **+1,2 %** | **+1,7 %** | **+1,9 %** |

- **Modelos** (ridge y gradient boosting sobre 41 rasgos, elegidos por validación
  cruzada dentro de la búsqueda). Separan mejor dentro de la búsqueda, pero en la
  validación quedan **por debajo de la simple cuenta de módulos**. Las reglas
  explicables generalizan mejor, así que el modelo solo se usa para desempatar.
- **Construcción de cartera** (`ronda3.py`, 5.184 carteras):
  - el **tamaño por convicción** (más capital con más módulos) mejora CAGR, caída y
    Sharpe en las dos ventanas;
  - el **límite por grupo de pares** empeora el CAGR sin reducir la caída: las mejores
    oportunidades son justo las caídas de sector entero;
  - el **94 %** de las 5.184 carteras bate a SPY en la validación (en la ronda 1, el 7,5 %).

### Finalistas de la ronda 3 (`finalistas3.py`)

D1 sale de la regla fijada de antemano: máximo CAGR en la búsqueda con caída máxima
≥ -30 %. Regla: cae ≥15 % desde el máximo de 20 d, en tendencia, ≥5 módulos;
prioridad por módulos y modelo; K=3 con convicción; sale > SMA10 (máx. 20 d).

| | D1 (reserva SPY) | D1-def50 (SPY solo en tendencia, tope 50 %) | C1 | **Combinada 50/50** | SPY |
|---|---|---|---|---|---|
| CAGR validación 2002-sep21 | 18,0 % | 17,1 % | 16,7 % | 17,1 % | 8,9 % |
| CAGR 2002-2025 | 18,1 % | 17,0 % | 17,3 % | **17,4 %** | 9,7 % |
| CAGR total 2002-2026 | 22,1 % | 20,8 % | 19,3 % | **20,3 %** | 10,0 % |
| Caída máxima | -54 % | -39 % | -40 % | **-35 %** | -55 % |
| Sharpe | 0,87 | 0,92 | 0,91 | **0,97** | 0,60 |
| CAGR costes 30 pb | 17,9 % | 16,7 % | 14,1 % | | |
| CAGR entrando **1 día tarde** | 16,8 % | 15,6 % | 16,5 % | | |
| IC 90 % bootstrap del CAGR | 13,5-31,8 % | 13,5-29,2 % | 12,4-26,3 % | | |
| DSR (N bruto 10.686 / N efectivo 205) | 0,37 / 0,99 | 0,45 / 0,99 | 0,44 / 0,99 | | |

- **Vecindario de D1:** las 15 variantes que cambian un solo parámetro dan un 16,9 % de
  mediana en la validación (p10: 14,2 %). Es una meseta.
- **2026 es un año extremo** (+149 % en D1-def50), por compras de caídas en memoria,
  almacenamiento y óptica (SNDK, LITE, MU, STX). Los precios son coherentes, pero por
  eso se informa también 2002-2025.
- **Concentración:** con K=3 y convicción ×2, una posición podía pesar el 67 %. El tope
  del 50 % casi no cuesta CAGR; uno del 35 % cuesta ~2,5 puntos.
- **Combinar D1-def50 y C1** (`combinar.py`, rebalanceo mensual; correlación diaria
  0,79) mantiene el CAGR y baja la caída máxima a -35 %.
- **Control de datos faltantes** (`sesgo_faltantes.py`). El 15,6 % de las candidatas de
  la validación vienen de series sin apertura ni volumen (CRSP) y no pueden puntuar en
  3 módulos. Si se les imputa la tasa media de esos módulos, D1-def50 pasa del 17,1 % al
  17,8 % en la validación. La falta de datos no favorecía el resultado; era algo
  conservadora.

## 5. Ronda 4: más CAGR con reserva, capacidad y apalancamiento

**Diagnóstico.** La compra de caídas ocupa de media solo el ~20-30 % del capital; el
resto espera en la reserva. Por eso hay dos palancas: rendir más con la reserva y meter
más capital en cada oportunidad buena. El estudio de salidas (`salidas.py`) muestra que
**el cuello de botella son las oportunidades, no el capital**. La salida más rápida
(SMA5) rinde más por día, pero empeora la cartera: no hay suficientes candidatas buenas
para reinvertir. D1 sigue saliendo con SMA10 y C1 con RSI2.

**Reservas** (`reservas.py`, fijadas de antemano; CAGR 2002-2026 / validación):

| Reserva | CAGR total | Validación | Caída máx. | Lectura |
|---|---|---|---|---|
| SPY | 10,0 % | 8,9 % | -55 % | referencia |
| SPY en tendencia, efectivo fuera | 7,5 % | 6,7 % | -24 % | |
| SPY en tendencia, **SHY** fuera | 8,0 % | 7,3 % | -24 % | efectivo con interés: +0,5 puntos |
| SPY en tendencia, IEF fuera | 8,2 % | 7,9 % | -26 % | ayudado por 20 años de bajada de tipos; peor en 2022 |
| Cesta momentum top-20 (12-1) | 9,9 % | **6,1 %** | -66 % | 25,7 % en 2021-26: apuesta de régimen, **descartada** |
| SPY gestionado por volatilidad (15 %, máx. 1,5×) | 8,5 % | 7,7 % | **-19 %** | Moreira y Muir (2017) |
| SPY 2× solo en tendencia | 10,5 % | 8,9 % | -43 % | Gayed y Bilello (2016): no compensa aquí |

**Motor v3** (`motor_v3.py`): reserva configurable y compras financiadas con deuda
(5 % anual; con 7 % el CAGR baja ~0,3 puntos). El tope de palanca cuenta la exposición
**real** (incluida la palanca interna de la reserva), y el cambio de régimen se ejecuta
en la apertura siguiente a la señal. El motor anterior lo aplicaba en el mismo cierre,
lo que inflaba ~0,4 puntos las versiones con reserva por tendencia.

**Qué funcionó y qué no** (`ronda4.py`, `escalera.py`, `tres_componentes.py`, `unificada.py`):

- ✅ **Capacidad:** hasta 5 huecos de 1/3 por componente, sin deuda: +1-2 puntos de CAGR
  con la misma caída máxima (19,8 % → 21,0 % con SHY; 21,7 % con IEF).
- ✅ **Palanca solo en tendencia, aplicada a las compras de caídas:** sube el CAGR
  manteniendo el Sharpe en ~0,93. Apalancar la reserva rinde menos, y apalancar SPY
  empeora el Sharpe (0,47 a 2×).
- ✅ **Gestión por volatilidad:** recorta 1-2 puntos de caída máxima con el mismo CAGR.
- ➖ **Caídas de ETFs (E1, `etfs.py`, `componente_etf.py`):** rebotan en las dos ventanas
  (+1,1-2,4 % a 5 días, t ≈ 3). Como tercer componente diversifica, pero diluye el CAGR.
- ❌ **Un único fondo de capital** para todas las señales: concentra el capital en las
  ventas generales y empeora la caída máxima (-52 % frente a -34 %). Descartado.
- ❌ **Cesta de momentum como reserva:** falla en la validación.

### Escalera final (`ronda4_final.py`)

| D1 + C1 (50/50) | Validación | 2002-25 | Total | Caída máx. | Exposición real media / p95 / máx. | Costes ×3 | 1 día tarde |
|---|---|---|---|---|---|---|---|
| Nivel 1 · máx. 1× | 18,7 % | 18,2 % | 21,0 % | -34 % | 0,86 / 1,00 / 1,00 | 16,0 % | 17,0 % |
| Nivel 2 · máx. 1,5× | 21,3 % | 20,9 % | 24,7 % | -45 % | 1,10 / 1,50 / 1,69 | 18,4 % | 20,0 % |
| **Nivel 3 · máx. 2×** | **23,0 %** | **22,8 %** | **26,8 %** | -46 % | 1,17 / 1,86 / 2,28 | 20,0 % | 21,2 % |
| SPY | 8,9 % | 9,7 % | 10,0 % | -55 % | | | |

Los tres niveles se definieron antes de mirar la validación. En 2008 pierden un -3,5/-3,8 %
(SPY, -37 %). Su peor episodio es el crash del Covid (feb-mar 2020), que llega en
tendencia alcista y con palanca. Con palanca, la exposición puede superar
momentáneamente el tope por la subida de precios tras la entrada (máx. 2,28×).

### Señales del último día (`senales_hoy.py`)

`python research/caidas/senales_hoy.py [--nivel 1|2|3] [--refrescar]` muestra, para cada
componente, las posiciones abiertas (simulación continua desde 2002), las órdenes de la
próxima apertura, la exposición y la deuda. También muestra el estado de la reserva y,
para cada candidata de D1, **qué módulos cumple**. Con `--refrescar` descarga precios
nuevos y lo recalcula todo con las banderas y el modelo **congelados**.

## 6. Ronda 5: entradas técnicas y solo acciones (sin ETFs ni efectivo)

**Requisitos:** sin ETFs ni efectivo. La reserva pasa a ser una cesta de acciones del
S&P 500 y SPY se usa solo como **señal** de régimen, nunca se compra.

**Rasgos técnicos** (`tecnicos.py`, 32 indicadores en las 554.471 caídas, 30 s):
- medias: distancia a EMA5/10/20/50, SMA50/100 y **WMA20**; canal de Keltner; pendiente
  de SMA50/200; EMA20 sobre EMA50 antes de la caída;
- **VWAP de 10 y 20 días** (precio típico × volumen);
- volatilidad: ATR%, caída en ATR, ATR5/ATR20, z y ancho de Bollinger;
- osciladores: RSI14, RSI2, estocástico %K, histograma y giro del MACD;
- volumen: MFI, Chaikin Money Flow, pendiente del OBV, volumen en días bajistas,
  clímax de volumen;
- velas: sombra inferior (martillo), cuerpo, giro clave; distancia al mínimo anual.

**Qué mejora la entrada** (`modulos_tec.py`). Se probaron 128 banderas (cuantiles con
cortes de la búsqueda). **11 salen elegidas solo con la búsqueda**, porque mejoran en
sus dos mitades y en D1 y C1, y **las 11 mejoran también en la validación 2002-21, en
los dos componentes**:

| Bandera (cortes de la búsqueda) | Mejora D1 valid. | Mejora C1 valid. |
|---|---|---|
| precio ≥5,3 % bajo la **EMA5** | +0,90 pp | +0,80 pp |
| precio ≥3,4 % bajo la EMA5 | +0,75 pp | +0,32 pp |
| precio ≥5,5 % bajo la **EMA10** | +0,63 pp | +0,54 pp |
| precio ≥7,2 % bajo la **WMA20** | +0,57 pp | +0,67 pp |
| precio ≥6,4 % bajo la **VWAP de 10 días** | +0,80 pp | +0,71 pp |
| z de **Bollinger** < -1,66 | +0,45 pp | +0,08 pp |
| **estocástico** %K < 0,11 | +0,85 pp | +0,20 pp |
| precio ≥12,6 % bajo la **SMA50** | +0,63 pp | +1,71 pp |
| precio ≥7,1 % bajo la SMA50 | +0,24 pp | +0,21 pp |
| **clímax de volumen** (≥61 % del máximo de 60 d) | +0,51 pp | +0,09 pp |
| **OBV** cayendo con fuerza (Q1) | +0,38 pp | +0,21 pp |

**Lectura:** las mejores compras son las acciones **más estiradas por debajo de sus
medias cortas y de su VWAP**, es decir, por debajo del precio medio de quien compró en
las últimas dos semanas. Mejor aún si hay **capitulación** (volumen de clímax, OBV
desplomándose). Con 8 o más banderas, la operación media de D1 gana +3,0 % (búsqueda)
y +3,1 % (validación), frente a +0,1 % y +0,6 % con ninguna.

**Lo que no funcionó:**
- **Órdenes límite para comprar más barato** (`limites.py`). Con un límite 0,25-0,5·ATR
  bajo el cierre, las operaciones ejecutadas ganan +0,6-1,7 puntos más, pero se pierde
  el 30-65 % de las señales (las que no bajan intradía suelen ser las que rebotan
  fuerte). Por señal, comprar a mercado en la apertura rinde más. Descartadas.
- **Ampliar a caídas menores** (≥10-12 %) con la puntuación conjunta (`ronda5c.py`):
  entran más operaciones, pero con menos ventaja, y desplazan a las buenas.
- **Mezclar D1 con C1**: con el filtro técnico, D1 tiene tanta ventaja que C1 diluye
  (22,9 % frente a 25,8 % sin deuda).

**Reserva solo de acciones** (`reservas.py`; CAGR total / validación / caída máx.):
equiponderada de miembros 9,9 % / 10,1 % / -60 %; 50 más líquidas 10,1 % / 8,1 % /
-61 %; **100 más líquidas ponderadas por volumen en dólares** 11,7 % / 9,2 % / -63 %;
50 de menor volatilidad 9,1 % / 9,9 % / -36 %; **líquidas en tendencia y baja
volatilidad fuera** 10,5 % / 8,9 % / **-42 %**.

### Estrategia definitiva (`definitiva.py`)

Reglas de entrada (señal al cierre, compra a la apertura siguiente): miembro líquido
sobre su SMA200, caída ≥15 % desde el máximo de 20 d, ≥5 módulos de origen y **≥2
banderas técnicas**. Hasta 5 posiciones de **medio capital** con convicción (medio
hueco con 5 módulos, uno con ≥6; nunca más del 50 %). Salida: cierre > SMA10 (máx. 20
sesiones). Palanca solo en tendencia, escalada por volatilidad, con **mantenimiento**:
si la exposición supera un 20 % el tope, se vende reserva para devolver deuda.

| D1-técnica | Validación | 2002-25 | **Total** | Caída máx. | Sharpe | Exposición máx. | Costes ×3 | 1 día tarde | Deuda 7 % |
|---|---|---|---|---|---|---|---|---|---|
| Máx. 1× | 20,4 % | 21,4 % | **25,8 %** | -45 % | 0,97 | 1,00 | 21,4 % | 19,0 % | — |
| **Máx. 1,5×** | 23,0 % | 23,6 % | **28,3 %** | -45 % | 0,93 | 1,76 | 23,2 % | 22,2 % | 28,2 % |
| Máx. 2× | 24,4 % | 24,8 % | **30,7 %** | -48 % | 0,91 | 2,48 | 24,9 % | 23,5 % | 30,4 % |
| *D1 + C1 técnicas, 1×* | 19,0 % | 19,9 % | 22,9 % | -41 % | 0,93 | | | | |
| SPY | 8,9 % | 9,7 % | 10,0 % | -55 % | 0,60 | | | | |

- 688 operaciones en 25 años, 69 % de acierto, +2,0 % de media por operación.
- **Peores episodios:** 2008 (-45 % sin deuda, frente al -55 % de SPY), finales de 2018
  y el Covid (-30 %).
- **La ejecución importa:** entrando un día tarde, el nivel sin deuda baja del 25,8 % al
  19,0 %. Las acciones más estiradas rebotan rápido.

### Señales del último día (`senales_hoy.py`)

`python research/caidas/senales_hoy.py [--nivel 1|2|3] [--refrescar]` (nivel 2, el de la
reserva contracíclica, por defecto) muestra las posiciones abiertas (simulación continua desde 2002), las órdenes
de la próxima apertura, la exposición, la deuda y el estado de la reserva. Para cada
candidata, muestra **qué módulos y qué banderas técnicas cumple**.

## 7. Ronda 6: complementos para cuando hay pocas caídas

**Problema:** en mercados tranquilos y alcistas apenas hay caídas que comprar. El
capital queda en la reserva pasiva y la capacidad de deuda sin usar.

**Complementos probados y descartados** (`complementos.py`, `reservas.py`,
`segunda_fila.py`; misma metodología, solo acciones):

| Idea | Fundamento | Resultado |
|---|---|---|
| Saltos al alza con noticia | cara opuesta de "la caída con noticia sigue" | negativo en la búsqueda 1, sin significación en la validación |
| Máximos de 52 semanas | anclaje (George y Hwang, 2004) | exceso **negativo** en las grandes del S&P (t = -1,9) |
| Expulsadas del S&P 500 | rebote tras la venta forzada | +24 % a 60 d en la validación, pero t 1,4 y negativo en la búsqueda (60 casos) |
| Retrocesos suaves de líderes | caídas del 3-8 % en momentum fuerte | +1,4 % en la búsqueda 1, ~0 en el resto |
| Caídas del propio índice | reversión del mercado | +1,3 % a 20 d en la validación, inestable en la búsqueda |
| Estacionalidad por acción | Heston y Sadka (2008) | 7,8 % de CAGR como reserva (peor que la equiponderada) |
| Volumen anómalo | Gervais et al. (2001) | 12,9 % en la validación, 5,5 % en la búsqueda: efecto diluido |
| Segunda fila solo con la cartera vacía | aprovechar el capital ocioso | empeora: las caídas de peor calidad son las que acaban mal |

**Lo que sí funciona: reserva contracíclica** (`liq100_vg_minvol`). En calma y en
tendencia, la reserva se apalanca hasta 1,5× en las 100 acciones más líquidas, según la
volatilidad de SPY (objetivo 15 %, Moreira y Muir 2017). Cuando la volatilidad sube o
se pierde la tendencia, rota hacia las 50 de menor volatilidad. Es el complemento
natural de la compra de caídas: las caídas aparecen con volatilidad alta (la reserva se
desapalanca) y la calma deja la deuda libre para la reserva.

| Reserva sola (solo acciones) | Búsqueda | Validación | Total | Caída máx. | Sharpe |
|---|---|---|---|---|---|
| 100 líquidas siempre | 21,9 % | 9,2 % | 11,7 % | -63 % | 0,59 |
| líquidas / baja vol. por tendencia | 17,1 % | 8,9 % | 10,5 % | -42 % | 0,63 |
| **contracíclica** | **21,7 %** | **11,1 %** | **13,2 %** | **-41 %** | **0,70** |

### Escalera actual (`definitiva.py`)

| D1-técnica | Validación | 2002-25 | **Total** | Caída máx. | Sharpe | Exposición media / máx. | Costes ×3 | 1 día tarde | Deuda 7 % |
|---|---|---|---|---|---|---|---|---|---|
| Nivel 1 · sin deuda | 20,4 % | 21,4 % | **25,8 %** | -45 % | 0,97 | 1,00 / 1,00 | 21,4 % | 19,0 % | — |
| **Nivel 2 · contracíclica** | 21,8 % | 23,7 % | **27,7 %** | -45 % | **0,99** | 1,19 / 1,50 | 23,2 % | 20,8 % | 27,2 % |
| Nivel 3 · + caídas hasta 2× | 25,8 % | 27,2 % | **32,7 %** | -49 % | 0,94 | 1,33 / 2,38 | 26,8 % | 25,4 % | 31,8 % |
| SPY | 8,9 % | 9,7 % | 10,0 % | -55 % | 0,60 | | | | |

En la prueba "deuda 7 %", la deuda interna de la reserva también va al 7 %.

### Fichas con 100.000 $ (`fichas.py`)

Cada estrategia simulada DESDE CERO (100.000 $ en caja, sin posiciones heredadas) a 1 año
(13-oct-2025), 5 años (12-oct-2021) y desde 2002, con valor diario en $, dónde está el
dinero (caídas, reserva, caja libre, deuda), caída desde máximo, rentabilidad por año y
todas las operaciones (retornos netos de costes). Salida en `resultados/fichas/`.

| 100.000 $ en… | 1 año | 5 años | Desde 2002 | Caída máx. (2002) | En caídas / caja libre (media) |
|---|---|---|---|---|---|
| Nivel 3 | 382.988 $ | 1.281.393 $ | 109,9 M$ (32,7 %) | -49 % | 31 % / 0 % (deuda media 33 %) |
| **Nivel 2** | 281.719 $ | 965.406 $ | 42,6 M$ (27,7 %) | -45 % | 23 % / 0 % (deuda media 19 %) |
| Nivel 1 | 303.468 $ | 832.372 $ | 29,3 M$ (25,8 %) | -45 % | 23 % / 0 % |
| Solo caídas (resto en efectivo) | 329.083 $ | 522.026 $ | 6,1 M$ (18,1 %) | -34 % | 23 % / **77 %** |
| D1 ronda 3 (SPY / efectivo) | 278.172 $ | 504.661 $ | 9,9 M$ (20,4 %) | -41 % | 21 % / 15 % |
| C1 ronda 2 (SPY / efectivo) | 180.492 $ | 376.310 $ | 7,1 M$ (18,8 %) | -40 % | 22 % / 14 % |
| F1 ronda 1 (SPY) | 151.880 $ | 243.903 $ | 2,5 M$ (13,9 %) | -52 % | 18 % / 0 % |
| SPY | 119.049 $ | 190.953 $ | 1,06 M$ (10,0 %) | -55 % | — |

Las compras de caídas solo ocupan de media un 23 % del capital y el 65 % de los días no
hay ninguna abierta: sin reserva, tres cuartas partes del dinero estarían paradas. En el
último año la reserva apenas aportó (los días buenos coincidieron con las compras y moverla
cuesta), por eso "solo caídas" gana a 1 año; a 5 años y desde 2002 la reserva suma mucho.
Ojo: 1 y 5 años están dentro de la muestra con la que se eligieron las reglas.

## 8. Ronda 7: cómo se comporta (rasgos, tamaño, entrada y salida)

**Anatomía** (`anatomia.py`). Las 4.701 candidatas del nivel 2 evaluadas como operación
aislada con las reglas reales (no solo las 687 que hizo la cartera):

| | Búsqueda 2021-26 | Validación 2002-21 |
|---|---|---|
| Acierto · retorno medio · mediano | 67 % · +2,0 % · +2,6 % | 74 % · +2,3 % · +3,3 % |
| Sesiones en cartera | 6,5 | 6,2 |
| Peor punto intradía (mediana) · sesión mediana del peor punto | -5,0 % · 2 | -3,8 % · 2 |
| Ganadoras que pasaron por -5 % | 31 % | 26 % |
| Perdedoras que llegaron a +3 % | 39 % | 37 % |

**Rasgos que separan ganadoras de perdedoras.** 104 rasgos; IC de rangos con errores
agrupados por mes. Se eligen con la búsqueda (|t| ≥ 2,5 y mismo signo en sus dos
mitades): salen 17 y **14 se mantienen en la validación**. Retorno medio del quintil
bajo → alto en la validación:

| Rasgo | Lectura | IC búsqueda / validación | Q1 → Q5 (validación) |
|---|---|---|---|
| Beta frente a SPY | más beta, mejor | +0,19 / +0,17 | +0,0 % → +5,2 % |
| Hueco de apertura en la compra | si abre aún más abajo, mejor | -0,15 / -0,14 | +4,6 % → +1,3 % |
| Caída medida en ATR | desmesurada para su rango, peor | -0,18 / -0,10 | +2,6 % → +1,0 % |
| Caída del último día | cuanto más cae ese día, mejor | -0,12 / -0,12 | +2,6 % → +1,4 % |
| Distancia a la EMA5 | más estirada, mejor | -0,13 / -0,14 | +3,0 % → +1,6 % |
| Volumen alto 5 días (noticia) | peor | -0,18 / -0,08 | +2,7 % → +1,7 % |
| Gemela cayendo también | mejor | -0,11 / -0,10 | +2,1 % → +1,4 % |
| Modelo de la ronda 3 | más alto, mejor | +0,22 / +0,06 | +1,4 % → +3,3 % |

No se mantienen el tamaño, el volumen en dólares y el precio (en 2021-26 ganan las
grandes; en 2002-21, las pequeñas). Sin relación estable: profundidad de la caída,
contexto de mercado, resultados empresariales y su sorpresa.

**Cartera** (`motor_v4.py`, `ronda7.py`; 55 variantes, una cosa cada vez; elegible si
supera a la actual en la búsqueda y en sus dos mitades):

| Variante (nivel 2) | Validación | 2002-2026 | Caída máx. | Sharpe |
|---|---|---|---|---|
| **Actual** (de golpe, SMA10, tamaño por módulos) | **21,8 %** | **27,7 %** | -45 % | **0,99** |
| Salida solo por tiempo (5 sesiones) | 9,7 % | 16,4 % | -61 % | 0,67 |
| Stop -8 % intradía | 13,1 % | 16,8 % | -60 % | 0,69 |
| Trailing 5 % | 15,8 % | 19,0 % | -57 % | 0,76 |
| Objetivo +5 % intradía | 19,9 % | 23,7 % | -50 % | 0,91 |
| Stop de catástrofe -20 % al cierre | 21,2 % | 26,4 % | -39 % | 0,96 |
| Dos tramos (2.ª mitad si cae ≥3 % más) | 20,0 % | 24,7 % | **-36 %** | 0,96 |
| 50 % fijo por operación | 22,4 % | 28,6 % | -42 % | 0,97 |
| Tamaño por volatilidad (el elegido en la búsqueda) | 20,7 % | 27,1 % | -42 % | 0,97 |
| Sin el quintil bajo de la puntuación de rasgos | 21,4 % | 27,5 % | -44 % | 0,99 |

- **Por qué los rasgos no mejoran la cartera:** hasta el peor quintil gana de media
  +1,4 % en ~6 sesiones, más que la reserva en ese tiempo. El límite son las
  oportunidades, no el capital: filtrar deja dinero parado.
- **Los stops venden a las ganadoras:** una de cada cuatro pasa por -5 % antes de ganar.
- La única variante elegida por la regla (tamaño por volatilidad) **no aguanta** en la
  validación (20,7 % frente a 21,8 %). Se mantiene la estrategia como está.

**Qué esperar** (`que_esperar.py`, simulación continua desde 2002): 64 % de meses en
positivo (mediana +2,2 %; peor mes -24 %, oct-2018); hasta 4 meses negativos seguidos y 7
operaciones perdedoras seguidas; en el 37 % de los meses no hay compras y la racha más
larga sin ninguna posición fue de 180 sesiones (2014-15). Fuera de muestra, a 12 meses:
mediana +22,7 %, 1 de cada 10 por debajo de -6 %; a 3 años, el 3 % de los tramos en
negativo; a 5 años, ninguno. Hay caídas del 25-30 % cada 2-3 años, a veces mucho peores
que las de SPY (2018: -30 % frente a -17 %; 2023: -25 % frente a -9 %).

## 9. Ronda 8: gestión dentro de la operación (salir de perdedoras, añadir tamaño)

**Pregunta:** con lo que ha pasado desde la compra, ¿se puede saber qué operaciones
conviene cerrar antes o reforzar? (`gestion.py`). Para cada operación aún abierta al
cierre del día k (1, 2, 3, 5) se mide el **exceso restante**: lo que gana desde la
apertura siguiente hasta la venta normal, menos lo que da la reserva en ese mismo
tramo. 21 condiciones × 4 días = 84 pruebas, t con errores agrupados por mes, FDR del
10 % en la búsqueda y mismo signo en sus dos mitades; luego, validación.

| Condición al cierre del día k | Búsqueda | Validación | Acaban ganando |
|---|---|---|---|
| Todas las abiertas, día 2 | +0,8 % | +1,4 % | 70 % |
| Va perdiendo ≥5 % (día 2) | **+2,6 %** (t 3,0) | **+2,8 %** (t 2,8) | 34 % |
| Va perdiendo ≥8 % (día 5) | **+3,4 %** (t 4,8) | **+4,3 %** (t 2,1) | 10 % |
| Día de caída ≥3 % con volumen ≥1,5× (día 2) | **+3,7 %** (t 2,6) | **+3,2 %** (t 3,1) | 43 % |
| Cierra bajo el mínimo del día de la señal (día 2) | +1,7 % (t 2,2) | +2,3 % (t 3,3) | 52 % |
| Sigue en pérdidas (día 5) | +1,2 % (t 1,7) | +2,2 % (t 2,3) | 36 % |
| Confirma: gana ≥4 % (día 3) | -0,6 % (t -1,0) | +0,1 % (t 0,3) | 89 % |

- **Ninguna** de las 84 pruebas da un exceso restante significativamente negativo: no
  hay ninguna señal para cortar perdedoras. Las que van perdiendo casi nunca acaban
  ganando, pero sí recuperan parte, y venderlas fija la pérdida en el peor momento.
- **Añadir a las que siguen cayendo** tiene ventaja real por dólar añadido (13 reglas
  elegidas, 10 se mantienen en la validación). Un ridge con todo el recorrido predice el
  exceso fuera de muestra (IC 0,18, t 4,7), pero su peor decil no pierde (≈0 %).

**En la cartera del nivel 2** (`ronda8.py`; motor v4 con calendario de compra/venta por
candidata; diferencia diaria frente a la actual con t de Newey-West e IC del 90 % por
bootstrap pareado):

| Cambio | Validación | Dif. anual valid. (t) | Dif. anual búsq. (t) | Caída máx. |
|---|---|---|---|---|
| **Actual** | **21,8 %** | | | -45 % |
| Salir si sigue en pérdidas al día 5 | 15,3 % | **-5,6 pp (-2,8)** | -7,3 pp (-2,0) | -45 % |
| Salir si pierde ≥5 % al día 2 | 21,4 % | -0,3 pp (-0,4) | -0,4 pp (-0,2) | -47 % |
| Añadir +50 % si pierde ≥5 % al día 2 | 21,9 % | +0,2 pp (0,6) | -1,0 pp (-1,1) | -46 % |
| Añadir +100 % si día de caída con volumen | 22,0 % | +0,2 pp (1,3) | +0,5 pp (0,4) | -45 % |
| Añadir +100 % al quintil alto del modelo | 22,2 % | +0,5 pp (1,0) | +2,4 pp (1,3) | -45 % |
| Salir del quintil bajo del modelo (día 2) | 23,7 % | +1,6 pp (1,7) | +2,4 pp (1,4) | -45 % |
| · vecindario: 10 % / 30 % / 40 % más bajo, día 2 | 22,5 / 22,4 / 21,1 % | +0,6 / +0,6 / -0,5 pp | | |
| · vecindario: 20 % más bajo, día 1 / día 3 | 20,4 / 19,8 % | -1,2 / -1,6 pp | | |
| Tamaño al entrar: fijo 50 % / por modelo / por beta | 22,4 / 21,4 / 21,4 % | +1,0 / 0,0 / -0,1 pp (t ≤ 0,7) | | |

- **Cortar perdedoras empeora** de forma significativa.
- **Añadir apenas cambia la cartera:** en 25 años solo hay 9-61 ocasiones en
  posiciones que la cartera tiene, y con tope de peso por posición (60-75 %).
- La única mejora cercana a la significancia (vender a partir del 3.er día el quintil
  bajo del modelo: recicla huecos para caídas nuevas, P(no mejora) = 5 % en el
  bootstrap) **es un pico**: sus vecinas dan entre -1,6 y +0,6 puntos. Se descarta.
- **Conclusión:** con estas reglas no se gana rentabilidad sin subir la exposición.
  El límite está en el número de oportunidades: cada operación ya rinde mucho más que
  el capital parado.

## Reproducir

```bash
pip install -e ".[research]"
python research/caidas/datos.py        # descarga + panel (≈1 min)
python research/caidas/estudio.py      # estudio de eventos
python research/caidas/rejilla.py && python research/caidas/rejilla.py --fase2
python research/caidas/analisis.py && python research/caidas/finalistas.py
python research/caidas/rasgos.py       # 56 rasgos PIT (≈80 s)
python research/caidas/eventos.py      # 554k eventos (≈20 s)
for b in r10_10_tend r5_7_tend r10_10 dd20_15; do python research/caidas/modulos.py $b; done
python research/caidas/modulos_v2.py && python research/caidas/gemelas.py
python research/caidas/combinada.py && python research/caidas/ronda2.py && python research/caidas/finalistas2.py
python research/caidas/meta.py && python research/caidas/ronda3.py   # ronda 3 (≈3 min)
python research/caidas/dsr_efectivo.py && python research/caidas/finalistas3.py && python research/caidas/combinar.py
python research/caidas/sesgo_faltantes.py
# ronda 4
python research/caidas/reservas.py && python research/caidas/ronda4.py && python research/caidas/escalera.py
python research/caidas/salidas.py && python research/caidas/etfs.py && python research/caidas/componente_etf.py
python research/caidas/tres_componentes.py && python research/caidas/unificada.py
python research/caidas/dsr_efectivo.py && python research/caidas/ronda4_final.py
# ronda 5 (solo acciones, entradas técnicas)
python research/caidas/tecnicos.py && python research/caidas/modulos_tec.py && python research/caidas/limites.py
python research/caidas/ronda5.py && python research/caidas/ronda5b.py && python research/caidas/ronda5c.py
python research/caidas/dsr_efectivo.py && python research/caidas/ronda5_final.py && python research/caidas/definitiva.py
# ronda 6 (complementos)
python research/caidas/complementos.py && python research/caidas/reservas.py && python research/caidas/segunda_fila.py
python research/caidas/definitiva.py
python research/caidas/fichas.py       # fichas con 100.000 $ a 1 año, 5 años y desde 2002
# ronda 7 (comportamiento)
python research/caidas/anatomia.py && python research/caidas/motor_v4.py && python research/caidas/ronda7.py
python research/caidas/que_esperar.py && python research/caidas/exportar_comportamiento.py
# ronda 8 (gestión dentro de la operación)
python research/caidas/gestion.py && python research/caidas/ronda8.py
python research/caidas/senales_hoy.py [--nivel 1|2|3] [--refrescar]
```

Los datos crudos, el panel, los rasgos, los eventos y las matrices de retornos van a
`data/cache/caidas/`, que está fuera de git.

## Siguientes pasos

1. **Paper trading:** ejecutar `senales_hoy.py --refrescar` cada día (por ejemplo con una
   rutina programada) y registrar las señales. Es la única validación realmente fuera
   de muestra que queda.
2. Validar en otro universo (S&P 400 / Russell 1000) si se consiguen datos PIT.
3. Cruzar con la deriva post-resultados (PEAD) del resto del repo como tercera pata de
   la cartera combinada.
