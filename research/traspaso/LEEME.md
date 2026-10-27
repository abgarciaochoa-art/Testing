# Traspaso de estrategias: compra de caídas y Noise Band

Todo lo que hace falta para que otro chat (u otra persona) entienda, replique, compare y
combine las dos estrategias de este estudio. Generado el 27-oct-2026 desde la rama
`claude/trading-earnings-reports-p6qt2c`.

## Qué hay en el paquete

| Carpeta o archivo | Contenido |
|---|---|
| `PROMPT_OTRO_CHAT.md` | el texto para pegar en el otro chat (resume las dos estrategias y pide el trabajo por fases) |
| `LEEME.md` | este índice |
| `1_caidas/DOSSIER.md` | compra de caídas en acciones del S&P 500: reglas exactas, marco común de comparación (sección 2), teoría, resultados, historia de 8 rondas, lo que falló, cómo combinar |
| `1_caidas/datos/` | rentabilidad diaria de 15 estrategias y SPY (2002-2026), métricas estándar, crisis, correlaciones, exposición diaria del motor, estrés y fichas en JSON |
| `1_caidas/graficas/` | ranking y curvas por familia |
| `2_noise_band/DOSSIER.md` | Noise Band, momentum intradía del S&P 500: reglas exactas minuto a minuto, la puntuación con sus 25 rasgos y un ejemplo resuelto, resultados en el mismo marco, historia, lo que falló, cómo usarla como capa |
| `2_noise_band/datos/` | rentabilidad diaria de 11 versiones y 8 mezclas con el nivel 1 (2005 → may-2020), métricas, crisis, cada operación, las operaciones de entrenamiento con sus rasgos, el modelo y dos ejemplos paso a paso |
| `2_noise_band/referencia.py` | implementación de referencia de Noise Band en Python legible, escrita solo con su dossier: da las mismas operaciones que el motor del estudio |
| `2_noise_band/graficas/` | curvas de las versiones, ancho del cono y nivel 1 con Noise Band encima |
| `codigo/` | copia del código de los dos estudios para leerlo o ejecutarlo: `pyproject.toml`, el paquete `earnings_alpha` y los `.py` y README de `research/caidas/`, `research/noise_band/` y `research/traspaso/`. Para ejecutarlo: `pip install -e ".[research]"` y cada `datos.py` descarga sus datos de fuentes públicas (la release `Johnbrick123/sp500-data` y el repositorio `FutureSharks/financial-data` de GitHub) |

## Orden de lectura

1. `PROMPT_OTRO_CHAT.md`: el resumen de todo en una página.
2. `1_caidas/DOSSIER.md`, secciones 1 (resumen), 2 (el marco común: es el estándar para
   comparar), 4 (reglas) y 9 (cómo combinar).
3. `2_noise_band/DOSSIER.md`, secciones 1, 2 (en qué cambia el marco), 4 (reglas) y 9.
4. El resto de cada dossier cuando haga falta: resultados (5), historia (6), lo que falló (7)
   y lecciones de método (8).
5. Para el detalle de cada ronda, los README de `codigo/research/caidas/` y
   `codigo/research/noise_band/`; para la lógica exacta, su código.

## Lo esencial

| | Compra de caídas (`CAI_`) | Noise Band (`NB_`) |
|---|---|---|
| Qué hace | compra acciones del S&P 500 en tendencia que caen de golpe sin noticia propia y vende al rebotar; el resto del dinero, en una reserva de acciones | sigue las rupturas del "cono" de movimiento normal del índice dentro del día, largo y corto, y cierra cada día |
| Datos | diarios, 2002-2026, miembros del índice en cada fecha | velas de 1 minuto, 2005 → may-2020 |
| Instrumento | acciones | futuros MES/ES (o SPY) |
| Recomendada | `CAI_N1`: 20,4 % al año fuera de muestra, peor caída -45 % | `NB_PUNT`: 7,9 % al año sin palanca, peor caída -6,5 %, Sharpe 1,18 (sola apenas ocupa capital) |
| Evidencia | B | C |
| Juntas | | `CAI_N1_NB_PUNT`: 26,0 % al año, peor caída -28,7 %, Sharpe 1,04 (nivel 1 solo en la misma ventana: 16,6 %, -45 %, 0,74) |

## Cómo comprobar una réplica

- **Caídas:** con los mismos datos, la curva de `CAI_N1` debe coincidir con
  `1_caidas/datos/diario.csv`; `1_caidas/DOSSIER.md` sección 4.8 da el plan diario.
- **Noise Band:** la lista de operaciones de `NB_ART` debe coincidir con
  `2_noise_band/datos/operaciones.csv` (fecha, hora, lado, precios), los dos ejemplos de
  `2_noise_band/datos/ejemplo_puntuacion.csv` deben salir iguales y la rentabilidad diaria
  debe tener una correlación > 0,99 con `2_noise_band/datos/diario.csv`. La implementación
  de referencia (`2_noise_band/referencia.py`) ya lo cumple: mismas 3.627 operaciones del
  artículo y mismas 2.410 de la fuerza que el motor del estudio.

## Formato de los datos

El de la sección 2.7 del dossier de caídas: `diario.csv` con la columna `fecha` (sesiones de
NYSE) y una columna por estrategia con la rentabilidad simple diaria neta en tanto por uno
(el primer día vale 0). Las de Noise Band cubren del 21-ene-2005 al 13-may-2020; las de año a
año (`*_AA`) no operan antes de 2008.

## Regenerar

```bash
python research/caidas/dossier.py        # datos del dossier de caídas
python research/noise_band/dossier.py    # datos del dossier de Noise Band
python research/traspaso/empaquetar.py   # el zip
```
