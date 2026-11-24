# Opening Range Breakout (ORB) intradía: índices y acciones

Pregunta: ¿hay alguna versión del ORB, con reglas claras y sencillas, que tenga ventaja
estadística y dé un 10-20 % al año?

> **Respuesta corta: sí, una — el "Gap ORB" en SPY (o en el futuro del S&P 500) — y solo si se usa
> la palanca intradía (≈3,5×).** El ORB genérico no funciona ni en índices ni en acciones grandes.
>
> | Gap ORB en SPY, 2000 → oct-2025, costes incluidos | Riesgo 1 %/op. | Riesgo 1,5 %/op. |
> |---|---|---|
> | Rentabilidad anual | 8,1 % | **10,0 %** |
> | Volatilidad | 12 % | 15 % |
> | Sharpe | 0,71 | 0,70 |
> | Peor caída | -19 % | **-21 %** |
> | Rentabilidad anual jun-2020 → oct-2025 (fuera de muestra) | 14,8 % | **15,7 %** |
> | Peor caída en 2020-2025 | -15 % | -17 % |
>
> 55 operaciones al año (≈1 por semana), acierto del 39 %, la ganancia media es el doble que la
> pérdida media, +6,1 pb netos por operación (t = 3,4), positiva en los cuatro periodos
> (2000-04, 2005-12, 2013-20, 2020-25). Una prueba de permutación (misma selección de días con
> dirección al azar) da p = 0,0025.

## La regla (Gap ORB)

1. **Hueco** = apertura de las 9:30 / cierre de ayer − 1. **ATR** = media de 14 días del rango
   verdadero diario, en % del cierre de ayer.
2. Solo se opera si **|hueco| > 0,4 × ATR** (≈1 día de cada 5).
3. **Rango de apertura** = máximo y mínimo de 9:30 a 9:45 (15 minutos).
4. A las 9:45 se pone **una sola orden stop del lado del hueco**: si el hueco es al alza, compra
   stop en el máximo del rango; si es a la baja, venta stop en el mínimo. El otro lado no se opera.
5. **Stop de pérdidas** en el otro extremo del rango. **Sin objetivo**: si no salta el stop, se
   cierra al cierre (orden MOC o a las 15:59). Si la orden no se activa en todo el día, se cancela.
6. **Tamaño:** arriesgar un 1-1,5 % del capital hasta el stop (acciones = 1,5 % × capital /
   (entrada − stop)), con un máximo de 4× el capital (el límite intradía de una cuenta de margen
   de EE. UU.; o con futuros MES/ES).

## Cómo se llegó (en orden)

| Ronda | Qué | Resultado |
|---|---|---|
| 1 (`ronda1_indices.py`) | ORB clásico (rango 5/15/30/60 min, ruptura o color de la vela, stops, objetivos) en 7 índices (CFD de S&P 500, Nasdaq-100, Russell 2000, FTSE, CAC, DAX, Euro Stoxx; 1 min, 2005-2020) | **Sin ventaja** neto de 1 pb por lado en ningún índice. |
| 2-4 (`ronda2_acciones.py` …) | ORB "Stocks in Play" (Zarattini, Barbon y Aziz 2024) en 198 acciones grandes del S&P 500, 5 min, nov-2018 → sep-2026: rango de 5 min, top-K por volumen relativo | Aparentemente bueno (30-40 % al año con palanca)… |
| 3-4 | …pero | Corregido un sesgo (vela que rompe por los dos lados) baja; y con entrada **sin ambigüedad** (cierre de vela fuera del rango → siguiente apertura) **desaparece**: toda la ganancia (+2,5-3 pb) está dentro de la vela de ruptura, que en real se come el deslizamiento de la orden stop. Con velas de 5 min no se puede validar; con acciones grandes no hay ventaja creíble. Elegir por volumen relativo sí ayuda frente a elegir al azar. |
| 5 (`ronda5_resultados.py`) | ORB solo en días de reacción a resultados (3.718 eventos) | Sin ventaja con entrada realista. |
| 6 (`ronda6_rasgos_indices.py`) | Rasgos de las rupturas en los 7 índices | El único rasgo estable: **romper a favor del hueco de la noche**, y mejor cuanto mayor el hueco (positivo en las dos mitades y en 6-7 de 7 índices). El ORB en contra del hueco o sin hueco pierde. |
| 7-8 (`ronda7_hueco.py`, `ronda8_gap_orb.py`) | Gap ORB en los 7 índices CFD | t de 3 a 7 en toda la vecindad de parámetros; 19 % al año la cartera de 7 índices. **Pero:** |
| 9 (`ronda9_spy_oos.py`) | Misma regla en **SPY real** (1 min, 2000 → oct-2025) | Rinde la mitad que en el CFD en los mismos años. Las mismas señales (99,5 % de coincidencia), pero el CFD de OANDA da +3,8 pb más por operación, casi todo en ~20 días de pánico (oct-2008, mar-2020) en los que en SPY salta el stop y en el CFD no. **Los CFD inflan las rupturas intradía** (cotizaciones medias, a veces viejas). |
| 10 (`ronda10_spy_familia.py`) | La familia entera (288 variantes) en SPY | Sin filtro, el ORB **pierde** (t ≈ −5,5 de media). Con hueco > 0,4 ATR a favor, todas las variantes mejoran; es el único filtro positivo en los 4 periodos. |
| 11 (`ronda11_final.py`) | Estudio de la regla elegida | Ver arriba y abajo. |

## Gap ORB en SPY: detalle

**Vecindad de parámetros (t de las operaciones, 2000-2025, costes incluidos).** Ningún valor
aislado: todo el bloque es positivo y casi todo positivo en los 4 periodos.

| rango \ k | 0,30 | 0,35 | 0,40 | 0,50 | 0,60 | 0,80 |
|---|---|---|---|---|---|---|
| 10 min | 2,5 | 2,5 | 3,1 | 2,5 | 2,0 | 1,4 |
| 15 min | 3,1 | 2,9 | **3,4** | 2,9 | 2,5 | 1,5 |
| 20 min | 3,8 | 3,8 | 3,9 | 3,0 | 2,6 | 1,7 |
| 30 min | 2,7 | 2,8 | 3,1 | 2,6 | 2,2 | 1,3 |
| 45 min | 2,5 | 2,4 | 2,6 | 2,3 | 1,9 | 1,0 |

**Por periodo** (riesgo 0,5 % por operación):

| Periodo | Operaciones | pb brutos | pb netos | t | Acierto |
|---|---|---|---|---|---|
| 2000-2004 (nunca visto al diseñar) | 184 | 12,2 | 8,6 | 1,4 | 37 % |
| 2005-2012 | 404 | 11,1 | 7,8 | 2,0 | 38 % |
| 2013-may 2020 | 477 | 5,0 | 3,2 | 1,2 | 39 % |
| jun 2020-oct 2025 (nunca visto al diseñar) | 353 | 7,9 | 7,0 | 2,2 | 43 % |

**Costes:** 0,02 $ por acción y lado (horquilla, 1 céntimo de deslizamiento del stop y
comisión). Con el doble (0,04 $) sigue en +3,9 pb (t 2,2); con el triple, casi a cero. En 2020-25,
con SPY a 300-650 $, el coste pesa poco (+5,2 pb aun con 0,06 $).

**Largos y cortos:** los dos ganan (+7,6 y +4,3 pb).

**Año a año con riesgo 1,5 % (%):**

| 2000 | 2001 | 2002 | 2003 | 2004 | 2005 | 2006 | 2007 | 2008 | 2009 | 2010 | 2011 | 2012 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| -3,7 | 26,5 | 11,1 | 15,2 | 0,4 | -9,7 | -6,5 | 20,4 | 19,7 | 19,9 | 36,3 | 9,0 | -3,7 |

| 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025* |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| -3,0 | 2,8 | 9,8 | -3,1 | 6,4 | 22,8 | 8,9 | 29,1 | 38,8 | 31,7 | -3,4 | 10,0 | -6,2 |

\* hasta el 21-oct-2025. 8 años negativos de 26, ninguno peor que -10 %.

## Lo que hay que saber antes de usarla

- **La ventaja es modesta.** Sharpe ~0,7. El 10-20 % sale de la palanca (≈3,6× de media en los
  días que opera), no de un acierto alto. Con riesgo 1,5 % cabe esperar caídas del 20-25 % y
  rachas de hasta 13 pérdidas seguidas.
- **Selección:** el filtro del hueco salió de los CFD (2005-2020) y el umbral 0,4 y el rango de
  15 min se fijaron mirando SPY entero. Lo más limpio es 2000-04 y 2020-25, positivos en toda la
  vecindad, y la permutación. Aun así, contando las ~300 variantes probadas, la evidencia es
  **moderada, no concluyente**. Empezar pequeño y comparar con este backtest.
- **Datos:** SPY de un repositorio público (`prsdro/spy`, velas de 1 min, precios negociados).
  Unos pocos días de pánico deciden mucho; un dato malo en esos días cambia el resultado.
- **Ejecución:** el stop debe estar en el mercado desde la entrada (orden OCO). Con 4× de palanca
  un hueco intradía grande puede saltar el stop (deslizamiento mayor que el previsto).
- **Lo que no funcionó** (para no repetirlo): ORB sin filtro en cualquier índice o acción;
  "Stocks in Play" en acciones grandes con velas de 5 min; ORB en días de resultados; objetivos de
  beneficio (no mejoran en SPY); el ORB a favor del hueco en acciones individuales (allí revierte).

## Aviso sobre los CFD de OANDA (afecta también a `research/noise_band`)

Las mismas señales de ruptura ganan bastante más en el CFD SPX500 de OANDA que en SPY (ronda 9),
y la diferencia está en los días de pánico. Los resultados de Noise Band, que usan ese CFD,
deberían repetirse con `datos.indice("SPY")`.

## Datos y reproducción

`python research/orb/datos.py` descarga (repositorios públicos de GitHub) y prepara en
`data/cache/orb/`: índices CFD (FutureSharks/financial-data), acciones de 5 min
(fatherpetrek-afk/QFin) y SPY de 1 min (prsdro/spy). Después, cada `rondaN_*.py` se ejecuta solo
y deja su tabla en `resultados/`. El motor (`motor.py`, numba) evalúa una operación por día con
velas de 1 o 5 min; si una vela toca stop y objetivo se supone el stop, y si una vela rompe el
rango por los dos lados se supone que toca primero el lado más cercano a su apertura.
