# Comprar caídas en el S&P 500: qué funciona, qué no y por qué

Investigación sobre **acciones reales del S&P 500 con pertenencia point-in-time**
(2002-01 → 2026-10), con costes de 10 pb por lado y ejecución a la apertura siguiente.
Se probaron **~10.700 carteras** en tres rondas, y todas cuentan en las correcciones
por minería de datos.

> **Estado actual (ronda 3).** La mejor propuesta es una cartera combinada, al 50 %
> entre D1-def50 y C1.
>
> | | Combinada | SPY |
> |---|---|---|
> | CAGR 2002-2026 | **20,3 %** | 10,0 % |
> | CAGR 2002-2025 | 17,4 % | 9,7 % |
> | Caída máxima | **-35 %** | -55 % |
> | Sharpe | **0,97** | 0,60 |
>
> El Sharpe deflactado supera el listón del repo (0,95) con el número efectivo de
> pruebas independientes (205), pero no con el bruto (10.686). Según la regla del repo
> es **zona gris: prometedora y explicable, no demostrada**.

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

### Señales del último día (`senales_hoy.py`)

Muestra las posiciones abiertas de cada componente, las órdenes de la próxima apertura
y, para cada candidata, **qué módulos cumple**. Con `--refrescar` descarga precios
nuevos y recalcula todo con las banderas y el modelo **congelados**.

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
python research/caidas/senales_hoy.py [--refrescar]
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
