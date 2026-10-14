# Comprar caídas en el S&P 500: qué funciona, qué no y por qué

Investigación sobre **acciones reales del S&P 500 con pertenencia point-in-time**
(2002-01 → 2026-10), costes de 10 pb por lado y ejecución a la apertura siguiente.
Se probaron **2.478 variantes** y todas cuentan en las correcciones por minería de datos.

## Datos: por qué fiarse de ellos

| Comprobación | Resultado |
|---|---|
| Fuente | release pública `Johnbrick123/sp500-data` (Yahoo + CRSP + Tiingo), 1.200 tickers |
| Pertenencia | intervalos [alta, baja) por ticker, verificados contra anuncios de S&P |
| Cobertura de precios de los miembros de cada fecha | **≈100 % desde 2001** (70 % en 1996, por eso empezamos en 2002) |
| Quiebras incluidas con su trayectoria real | Lehman cierra en 0,13 $, WaMu en 0,16 $, Bear Stearns, Countrywide… |
| Contraste con la semilla del repo (`data/seed/`) | Jaccard 0,83-0,97 (las diferencias son sobre todo de nomenclatura de tickers) |
| Picos imposibles limpiados | 19 (subida >150 % revertida al día siguiente) |
| Conciliación del motor | con coste 0, equity final = producto de los retornos de las operaciones (error ~1e-14) |
| Reserva en SPY sin señales | reproduce el CAGR de SPY al decimal |

**El sesgo de supervivencia importa muchísimo en esta familia de estrategias.** Repetida
sobre los miembros *actuales* del índice (lo que haría un backtest ingenuo con la lista
de hoy), la misma estrategia pasa del 15,7 % al **21,6 %** de CAGR. Las empresas que caen
y no vuelven son justo las que un universo sesgado borra.

## 1. Estudio de eventos: qué caída predice rebote (`estudio.py`)

Exceso sobre la media equiponderada de los miembros, con entrada a la apertura de t+1.
La t es de Newey-West.

| Caída | Eventos | +5 d | +20 d | +60 d | Lectura |
|---|---|---|---|---|---|
| -30 % desde el máx. de **20 d**, en tendencia (> SMA200) | 270 | +1,4 % | +4,5 % | **+10,0 %** (t 1,9) | rebote fuerte, positivo en los 3 subperiodos |
| -25 % en 20 d, en tendencia | 472 | +1,5 % (t 2,1) | +3,3 % | +2,6 % | igual idea |
| RSI(2) ≤ 5 | 171.711 | +0,09 % (t 2,2) | +0,09 % | +0,24 % | real pero diminuto: < costes |
| 4 cierres bajistas seguidos | 138.444 | +0,09 % (t 2,1) | +0,06 % | +0,19 % | ídem |
| -30 % desde el máx. de **252 d** (caída lenta) | 365.060 | -0,09 % | -0,31 % | **-1,1 %** (t -2,1) | **los perdedores lentos siguen perdiendo** (momentum) |

**Idea central: importa la *velocidad* de la caída, no su tamaño.** Una caída brusca de
una empresa que venía subiendo suele ser una sobrerreacción (venta forzada, noticia
puntual) y revierte. Una caída lenta desde máximos anuales suele reflejar un deterioro
de fundamentales y continúa (es el efecto momentum de Jegadeesh-Titman). La reversión
de 1-5 días existe, pero en el S&P 500 vale ~0,1 %, menos que el coste de ida y vuelta.

## 2. Rejilla de carteras (`rejilla.py`)

| Familia | Variantes | CAGR mediano 02-15 | CAGR mediano 16-26 | DD mediano |
|---|---|---|---|---|
| A. Caída rápida (dd 10-60 d), en efectivo cuando no hay señal | 1.440 | 1,7 % | 3,9 % | -67 % |
| B. Reversión corta (RSI2, rachas, z-score) | 315 | 3,1 % | 2,3 % | -56 % |
| C. Caída lenta (dd 126-252 d) | 36 | -1,9 % | 7,7 % | -69 % |
| D. Caída de N días **en tendencia** (fase 2) | 384 | 5,6 % | 9,4 % | -54 % |
| E. Caída desde máx. **en tendencia** (fase 2) | 288 | 6,4 % | 10,1 % | -57 % |
| SPY (referencia) | — | 6,2 % | 15,1 % | -55 % |

- Solo el **10 %** de las variantes bate a SPY en el periodo completo.
- El ranking dentro de muestra apenas predice el de fuera: Spearman de 0,12 en CAGR.
  **PBO = 0,42**, es decir, un 42 % de probabilidad de que la mejor configuración
  dentro de muestra quede por debajo de la mediana fuera de ella (`resultados/is_vs_oos.png`).
- Comprar caídas **en solitario** deja el dinero parado la mayor parte del tiempo.
  Cuando se concentra para aprovecharlo (K = 5), acumula quiebras de 2008, con caídas
  máximas del -75 % al -92 %.

## 3. La estrategia que sí aguanta: SPY + compra de caídas rápidas en tendencia (`finalistas.py`)

**Regla:** al cierre, si una acción del S&P 500 está por encima de su SMA200 y ha caído
≥ X % en las últimas 10 sesiones, se compra a la apertura siguiente con 1/K del capital
(prioridad a la caída mayor). Se vende cuando cierra por encima de su SMA5 (o SMA20),
con un máximo de 20 sesiones. **El capital que no está en esas posiciones está en SPY.**

No es el máximo de la rejilla. Está en el centro de una meseta: con X entre 10 % y 20 %,
K = 3, 5 o 10 y cualquiera de las tres salidas, el CAGR mediano es del 11-13 %.

| | F1 X=15 % SMA5 K=3 | F2 X=15 % SMA5 K=5 | F3 X=20 % SMA20 K=3 | F4 X=10 % SMA5 K=3 | SPY |
|---|---|---|---|---|---|
| CAGR 2002-2015 | 13,3 % | 11,3 % | 13,3 % | 14,5 % | 6,2 % |
| CAGR 2016-2026 (fuera de muestra) | 18,8 % | 18,3 % | 16,5 % | 21,8 % | 15,1 % |
| **CAGR total** | **15,7 %** | **14,3 %** | **14,7 %** | **17,7 %** | 10,0 % |
| Caída máxima | -50 % | -50 % | -49 % | -56 % | -55 % |
| Sharpe | 0,70 | 0,70 | 0,67 | 0,71 | 0,60 |
| Operaciones/año | 36 | 43 | **7** | 115 | — |
| Acierto / ret. medio por operación | 63 % / +1,0 % | 64 % / +1,0 % | 71 % / +3,6 % | 64 % / +0,6 % | — |
| Alfa anual vs SPY (t Newey-West) | 5,3 % (2,2) | 4,0 % (2,2) | 4,3 % (1,8) | 7,8 % (2,3) | — |
| CAGR con costes de 20 pb | 13,0 % | 12,3 % | 14,1 % | 9,0 % | — |
| CAGR con costes de 30 pb | 10,3 % | 10,4 % | **13,6 %** | 0,9 % | — |
| CAGR con universo de hoy (sesgado) | 21,6 % | 20,4 % | 22,2 % | 25,3 % | — |
| Sharpe deflactado (N = 2.478) | 0,62 | 0,62 | 0,56 | 0,66 | — |

Retornos por año natural: `resultados/finalistas_anual.csv`. Operaciones una a una:
`resultados/operaciones_F*.csv`. Curvas de capital: `resultados/finalistas.png`.

### Lectura honesta

- **Por qué funciona:** la estrategia es SPY (beta ≈ 1,1) más un satélite que compra
  sobrerreacciones bruscas en empresas con tendencia alcista y las vende en cuanto
  rebotan (3-14 días). El alfa del satélite es de 4-8 % anual y es coherente con el
  estudio de eventos.
- **No reduce el riesgo:** la caída máxima es la de la bolsa (-50 % en 2008, -18/-28 %
  en 2022).
- **Estadística:** el alfa tiene t ≈ 2,2. Tras corregir por las 2.478 pruebas, el Sharpe
  deflactado es 0,56-0,66, por debajo del 0,95 que exigiría el repo para darla por
  demostrada. Es una **hipótesis plausible y explicable, no un hecho probado**.
- **Costes:** F4 (la de mayor CAGR) depende por completo de ejecutar barato. **F3 es la
  más robusta**: 7 operaciones al año y un 13,6 % de CAGR incluso con 30 pb por lado.
- **Concentración:** con K = 3 cada posición pesa un 33 % mientras dura. El riesgo
  idiosincrático por operación es alto.

## Reproducir

```bash
python research/caidas/datos.py        # descarga + panel (≈1 min)
python research/caidas/estudio.py      # estudio de eventos (≈3 min)
python research/caidas/rejilla.py      # fase 1: 1.788 variantes (≈4 min, 4 núcleos)
python research/caidas/rejilla.py --fase2   # fase 2: 690 variantes
python research/caidas/analisis.py     # selección IS, DSR, PBO, figuras
python research/caidas/finalistas.py   # finalistas a fondo
```

Los datos crudos y las matrices de retornos van a `data/cache/caidas/`, que está fuera de git.

## Siguientes pasos

1. Separar el satélite de su base SPY: estudiar si el alfa se concentra en sectores o
   alrededor de resultados trimestrales, cruzando con `consenso_master.parquet` (¿la
   caída es por resultados o no?).
2. Probar el apalancamiento de la meseta (por ejemplo, 1,3× sobre SPY + satélite) frente
   a su caída máxima.
3. Validar en un universo independiente (S&P 400 / Russell 1000) antes de dar crédito
   al alfa.
