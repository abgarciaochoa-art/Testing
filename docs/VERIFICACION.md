# Cómo verificar si el PEAD da dinero — manual del verificador

Este documento responde a la pregunta: **¿cómo sabemos si la estrategia tiene
ganancia o no?** La respuesta es un protocolo, no una opinión:
`examples/verificar_pead.py` ejecuta la batería completa contra datos reales y
emite un veredicto según criterios fijados **antes** de mirar los resultados.

## El comando (en una máquina con internet)

```bash
git clone <repo> && cd Testing
git checkout claude/trading-earnings-reports-p6qt2c
pip install -e ".[providers,stats]"

# Verificación completa 2010-2022 (primera vez: 20-60 min de descarga; con caché, minutos)
python3 examples/verificar_pead.py --fuente stooq --desde 2010-01-01 --hasta 2022-12-31

# Versión rápida para un primer vistazo (100 tickers con más eventos)
python3 examples/verificar_pead.py --fuente stooq --max-tickers 100

# Prueba de humo sin red (mecánica, no veredicto)
python3 examples/verificar_pead.py --fuente sintetico
```

## Qué hace exactamente

1. **Eventos reales**: ~47.800 anuncios de resultados del S&P 500 (1995-2026)
   del dataset de consenso, **filtrados a pertenencia point-in-time al índice**
   en la fecha de cada evento (un evento de TSLA de 2015 no cuenta: TSLA no
   entró al índice hasta dic-2020).
2. **SUE de analistas** (Livnat-Mendenhall): sorpresa estandarizada por la
   sigma de las sorpresas históricas del propio emisor.
3. **Rejilla de backtests** con costes SIEMPRE activados (spread por liquidez +
   impacto + comisión): entrada T+1 tras el anuncio con salidas a 5/10/21/63
   sesiones (el veredicto), y entradas T-3/T-1 pre-anuncio **solo informativas**
   (cruzan el gap del anuncio; el motor exige declarar que la fecha se conocía
   por adelantado, y las fechas estimadas se excluyen de ese brazo).
4. **Estadística seria**: spread mensual Q5-Q1 con t de Newey-West; corrección
   de Benjamini-Hochberg sobre las 12 combinaciones probadas; Sharpe deflactado
   descontando esas 12 pruebas; partición 2010-2016 (calibración) / 2017-2022
   (confirmación) reportada por separado.

## Los criterios pre-registrados (congelados por `tests/test_verify.py`)

| Criterio | Umbral | Por qué |
|---|---|---|
| t Newey-West del spread mensual Q5-Q1 | ≥ 3,0 | Harvey-Liu-Zhu (2016): tras décadas de minería de factores, t=2 ya no protege; un factor nuevo debe superar 3 |
| Significancia tras Benjamini-Hochberg | ≥ 1 combo post-anuncio con α=0,05 | 12 combinaciones probadas ⇒ alguna sale "significativa" por azar; BH lo descuenta |
| Sharpe deflactado | ≥ 0,5 | Bailey-López de Prado: descuenta el sesgo de haber elegido el mejor de N intentos |
| Confirmación fuera de calibración | spread medio > 0 en 2017-2022 | Una señal que solo existe en el tramo de calibración está sobreajustada |
| Potencia mínima | ≥ 400 eventos por tramo | Por debajo, el resultado es ruido con formato de tabla |

**Regla del veredicto** (fijada de antemano):
- **VIVA** = t_NW ✔ y BH ✔ y confirmación ✔
- **MUERTA** = t_NW < 1 y ningún combo sobrevive a BH
- **DEBIL** = todo lo demás (evidencia parcial: típicamente merece más datos, no más capital)
- **NO_CONCLUYENTE** = potencia insuficiente

Cambiar un umbral exige editar `verify.py` **y** el test que lo congela en el
mismo commit: el pre-registro deja rastro o no es pre-registro.

## Cómo leer el resultado

- El bloque **Razonamiento** lista cada voto con su número. El veredicto nunca
  es un misterio: son cuatro comprobaciones que puedes re-derivar a mano.
- La **rejilla** muestra por combinación: spread Q5-Q1 con IC 95%, hit rate,
  peor evento y percentil 1 (las colas importan más que la media), y la
  descomposición gap/intradía. Las filas `informativa=True` cruzan el anuncio:
  compara su retorno extra contra su `worst_net` antes de enamorarte de ellas.
- La **auditoría de honestidad** cuenta todo lo descartado (eventos fuera del
  índice, tickers sin precios, eventos sin SUE). Si esos números son grandes,
  el veredicto vale menos — está impreso justo debajo para que no se olvide.

## Limitaciones que el veredicto NO corrige

- **Supervivencia del dataset de consenso** (medida, no sospechada): cobertura
  59,6% en jun-2008, quebradas ausentes. Por eso pre-2010 está excluido por
  defecto (`--incluir-pre2010` lo fuerza y etiqueta el resultado como sesgado).
- **Precios de proveedores gratuitos**: Stooq/yfinance pueden carecer de
  algunos deslistados; la auditoría cuenta cuántos tickers se pierden y avisa
  si superan el 25%.
- **Sin señales de opciones ni revisiones de analistas**: este verificador
  responde SOLO a la pregunta del PEAD. Las demás señales tienen su propio
  camino (recolector propio, ver docs/OPCIONES_COMO_CONSEGUIR_LOS_DATOS.md).
- **VIVA ≠ promesa**: significa que la señal superó, con costes y fuera de la
  muestra de calibración, los umbrales fijados de antemano. El paso siguiente
  sensato tras un VIVA es paper trading con el recolector corriendo, no
  apalancarse.
