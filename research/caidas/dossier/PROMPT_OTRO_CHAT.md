Hola. Vengo de otro chat en el que investigué estrategias de compra de caídas en acciones
del S&P 500. Te adjunto su dossier: DOSSIER.md, que es la referencia y conviene leer
entero, y la carpeta datos/ con CSV y JSON. Si no puedes abrir el zip o ejecutar código,
trabaja con las tablas de DOSSIER.md.

Objetivo: juntar esas estrategias con las tuyas, compararlas todas con las mismas reglas y
decidir cuáles combinar en una sola cartera.

LO ESENCIAL DEL DOSSIER
- Universo: acciones que eran miembros del S&P 500 en cada fecha (point-in-time, incluidas
  las que quebraron), de 2002-01-02 a 2026-10-12. Señal al cierre, compra y venta a la
  apertura siguiente, 10 pb por lado, dividendos incluidos, sin impuestos.
- Las reglas se eligieron con oct-2021 → oct-2026 y se validaron hacia atrás con 2002 →
  sep-2021. Esa ventana ("fuera de muestra") es la que cuenta. 2021-2026 está dentro de
  muestra, y 2024-2026 fue extraordinario.
- Estrategias vigentes, todas solo con acciones (CAGR fuera de muestra / peor caída):
  · CAI_N1, nivel 1, la recomendada, sin deuda: 20,4 % / -45 % (25,8 % en 2002-2026).
  · CAI_15X, caídas a 1,5× (deuda solo en las compras): 23,0 % / -45 %.
  · CAI_N3, caídas a 2×: 24,4 % con costes reales / -48 a -50 %.
  · CAI_N2_TRAMOS, modo prudente: 18,5 % con costes reales / -35 %.
  · Referencia, SPY: 8,9 % / -55 %.
- CAI_SOLO es solo el "motor" de compras de caídas, sin reserva. Usa de media el 23 % del
  capital, tiene alguna posición el 35 % de los días y su correlación mensual con SPY es
  0,21. El resto del dinero puede esperar en otra estrategia: es la pieza natural para
  combinar con las tuyas.
- Nivel de evidencia B: sólido en histórico (unas 11.000 carteras probadas, con corrección
  por minería de datos), pero todavía sin paper trading.

MIS PREFERENCIAS
- La cartera final, solo con acciones: sin ETFs ni efectivo (SPY solo como señal o
  referencia). Si una estrategia tuya usa ETFs o efectivo, dilo y propón su versión solo
  acciones si existe.
- Prefiero reglas claras y robustas fuera de muestra a un CAGR más alto que solo existe
  dentro de muestra.

QUÉ TE PIDO, POR FASES
1. Entender. Resume en unas 10 líneas cómo funcionan CAI_N1 y CAI_SOLO y el marco común
   (sección 2 del dossier). Lista tus estrategias con: universo, periodo, ejecución,
   costes, dividendos, ventana con la que elegiste las reglas y cuántas variantes
   probaste. Señala en qué se aparta cada una del marco (lista de la sección 2.6).
2. Fichas. Rellena la plantilla de la sección 2.5 para cada estrategia tuya, con reglas
   exactas y nivel de evidencia (sección 2.4). Si alguna hay que recalcularla con las
   reglas del marco (point-in-time, apertura siguiente, 10 pb, dividendos), hazlo o dime
   qué datos faltan.
3. Mismas métricas. Calcula las de la sección 2.3 en las ventanas de la 2.2 y exporta la
   rentabilidad diaria neta de cada estrategia en el formato de datos/diario.csv
   (sección 2.7), con un prefijo propio en el identificador (por ejemplo OTR_).
4. Tabla única. Todas las estrategias de los dos chats en una tabla ordenada por CAGR fuera
   de muestra (la ventana fuera de muestra de cada una y, si se puede, la común), con peor
   caída, Sharpe, Calmar, correlación con SPY, si es solo acciones y nivel de evidencia.
5. Combinar (sección 9). Correlaciones mensuales con las nuestras y comportamiento en los
   tramos de datos/crisis.csv. Prueba primero tus estrategias siempre invertidas como
   reserva del motor CAI_SOLO, con la fórmula de la sección 9.3 y
   datos/exposicion_diaria.csv; después, mezclas por pesos (50/50 y pesos inversos a la
   volatilidad, con rebalanceo mensual). Fija las reglas de mezcla antes de mirar
   resultados, valídalas fuera de muestra y cuenta todas las que pruebes.
6. Propuesta. Una o dos carteras finales con reglas claras, qué esperar (meses positivos,
   peor caída, años malos) y los próximos pasos.

REGLAS
- Ninguna cifra vale sin validación fuera de muestra; marca lo que esté dentro de la
  muestra con la que se eligieron las reglas.
- Si comparas estrategias con convenciones distintas, dilo y cuantifica el efecto.
- Si algo no se puede calcular con lo que tienes, dilo y pídeme lo que falta; no lo
  supongas.

CÓMO RESPONDER
En español, simple y con tablas. Termina siempre con un "Resumen por temas": una tabla de
tres columnas (Tema | Qué descubrimos | ¿Nos ayuda?), corta y bien explicada.

Haz solo la fase 1 y espera mi visto bueno antes de seguir.
