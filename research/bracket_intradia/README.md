# Horquilla intradía: stop fijo, objetivo 2:1 y candado

La idea: entrar en un valor muy líquido (Nvidia, por ejemplo, a 145 $), arriesgar 0,50 $
(stop en 144,50) y vender en +1 $ (objetivo 2:1). Una variante es el "candado": al llegar a
+1,8R, el stop sube a +1R y se deja correr. Aquí se prueba con velas de 1 minuto y costes
realistas, y se buscan las entradas, stops y salidas que dan ventaja.

Informe visual: `informe.html` (también publicado como artifact).

## Conclusiones

1. **La horquilla sola no da ventaja.** Si el precio no tiene deriva, P(tocar +k·R antes que
   −R) = 1/(1+k): un 33 % a 2:1. Entrando a cualquier hora, el acierto real se pega a la teoría
   en los cinco mercados (Nvidia 33,8 %, Nasdaq-100 35,0 %, bitcóin 35,1 %, oro 35,6 %) y el R
   bruto queda en ±0,03. Con costes, todos pierden.
2. **Los costes mandan.** En R, el coste es inversamente proporcional al stop. Con 0,1 ATR
   (el stop de 0,50 $ en Nvidia a 145 $), son 0,08R en Nvidia, 0,10R en el Nasdaq-100, 0,33R
   en el oro y 0,44R en el bitcóin (0,05 % por lado). Con 0,2 ATR, se quedan en la mitad.
3. **La ventaja viene de operar a favor del día.** Sirven tres señales: precio por encima de
   la VWAP y de la apertura, el día ya va ≥ 0,25 ATR a favor, o rompe el rango de los primeros
   30 minutos. Dan +0,03 a +0,09R brutos a 2:1 en el bitcóin y el Nasdaq-100, en las dos mitades
   de cada muestra. Comprar caídas bajo la VWAP resta (salvo en el oro, que revierte un poco).
   La tendencia de días (media de 20 sobre la de 50) no aporta dentro del día.
4. **Dejar correr gana a 2:1 fijo.** Con la señal "día ≥ 0,25 ATR a favor", el R bruto por
   operación sube así:

   | Mercado | 2:1 fijo | Candado | Sin objetivo, hasta el cierre |
   |---|---|---|---|
   | Bitcóin | +0,07 | +0,12 | +0,20 |
   | Nasdaq-100 | +0,03 | +0,05 | +0,09 |
   | Nvidia | +0,06 | +0,07 | +0,24 |

   El acierto baja al 20–30 %.
5. **Nvidia y las 40 acciones (2026): ganan los cortos.** El periodo incluye la caída de
   noviembre. Solo largo pierde aunque el filtro sea bueno. Hay que seguir al día en los dos
   sentidos, o no operar los días que van en contra.
6. **La versión original pierde mucho.** Largo siempre, stop 0,1 ATR, 2:1 y sin límite son
   unas 13 operaciones al día pagando costes. En Nvidia son −289R en 192 sesiones. El plan
   ("día a favor", ambos lados, stop 0,2 ATR, una operación por lado y día, dejar correr) da
   +35R en Nvidia, +201R en el Nasdaq-100 (2005-2020), −143R en el bitcóin y −511R en el oro.
7. **Un día de 5 ganadas de 7 a 2:1** tiene un 4,5 % de probabilidad sin ninguna ventaja, más o
   menos una vez al mes. Para distinguir una ventaja de +0,05R del azar hacen falta unas 3.200
   operaciones; para +0,1R, unas 800.

## Límites

- **Nvidia:** solo 192 sesiones útiles (feb → nov 2026). Dukascopy limita mucho la descarga.
- **40 acciones:** solo 30 sesiones (Yahoo da 60 días de velas de 2 minutos), en plena caída.
- **Nasdaq-100 y oro:** terminan en mayo de 2020.
- Las cifras positivas de Nvidia y las acciones tienen t de 1-1,5: no son prueba. Lo robusto
  es lo que se repite en el bitcóin (5 años) y el Nasdaq-100 (15 años).

## Datos

| Panel | Fuente | Vela | Periodo |
|---|---|---|---|
| `dukas_NVDA` | Dukascopy (CFD, precio de compra) | 1 min | dic-2025 → nov-2026 (192 sesiones tras el arranque de las medias) |
| `yahoo2m_*` | Yahoo, 40 valores más líquidos | 2 min | 15-oct → 25-nov-2026 |
| `BTC` | Binance BTCUSDT | 1 min | 2022 → nov-2026 |
| `NAS100_oanda`, `XAU_oanda` | OANDA vía FutureSharks/financial-data | 1 min | 2005/2006 → may-2020 |

Todos se recortan a la sesión de Nueva York (9:30-16:00). Nvidia de Dukascopy coincide con
Yahoo (diferencia media del cierre diario del 0,07 %). Los splits se ajustan automáticamente.

## Reglas del motor (`motor.py`)

- La señal se evalúa al cierre de la vela y la entrada es a la apertura de la siguiente.
- Una sola posición a la vez por mercado y lado. Todo se cierra 5 minutos antes del final.
- Si una vela toca el stop y el objetivo a la vez, cuenta como stop (< 1 % de los casos).
- El objetivo se llena a su precio. El stop se llena a su precio o a la apertura si hay hueco,
  más un deslizamiento fijo.
- El candado y el stop dinámico se recalculan con el máximo de cada vela. Si la vela cierra
  por debajo del nuevo stop, se sale a ese stop.
- Comprobado con un paseo aleatorio sin deriva: el acierto sale en 1/(1+k) y la esperanza en
  ≈ 0R para 1:1, 2:1, 3:1 y el candado.
- Costes por lado más deslizamiento en las salidas a mercado:

  | Mercado | Por lado | Deslizamiento |
  |---|---|---|
  | Acciones | 0,01 % | 0,01 % |
  | Bitcóin | 0,05 % | 0,02 % |
  | Oro | 0,015 % | 0,01 % |
  | Nasdaq-100 | 0,005 % | 0,005 % |

  En `estudio.py` hay una variante de costes altos.

## Reproducir

```bash
pip install numba scikit-learn matplotlib
python research/bracket_intradia/descargar.py binance      # BTCUSDT 1 min
python research/bracket_intradia/descargar.py yahoo        # diario S&P 500 + 2 min / 1 min recientes
python research/bracket_intradia/descargar.py dukascopy    # lento: ~1 día por segundo con 5 hilos, a ratos mucho menos
git clone --depth 1 --filter=blob:none --sparse https://github.com/FutureSharks/financial-data data/cache/fsdata
git -C data/cache/fsdata sparse-checkout set pyfinancialdata/data/currencies/oanda/XAU_USD pyfinancialdata/data/currencies/oanda/NAS100_USD
python research/bracket_intradia/datos.py       # paneles en data/cache/bracket/paneles
python research/bracket_intradia/liquidez.py    # ranking de liquidez
python research/bracket_intradia/estudio.py     # base, filtros, stop, salidas, horario, gestion, combo
python research/bracket_intradia/mapa.py        # acierto según el movimiento previo del día
python research/bracket_intradia/curvas.py      # R acumulado de cuatro formas de operar
```

Tablas en `resultados/`: una fila por combinación, con acierto, R bruto, R neto, coste en R,
t agrupado por día y, donde aplica, cada mitad de la muestra.
