Hola. Vengo de otro chat en el que investigué dos estrategias sobre el S&P 500. Te paso todo
lo necesario para que las entiendas tan bien como allí, las puedas replicar paso a paso y las
comparemos y combinemos con las tuyas bajo las mismas reglas.

QUÉ TE ADJUNTO (traspaso_estrategias.zip)
- LEEME.md: índice y orden de lectura.
- 1_caidas/: DOSSIER.md, datos/ y graficas/ de la compra de caídas en acciones del S&P 500
  (identificadores CAI_).
- 2_noise_band/: DOSSIER.md, datos/ y graficas/ de Noise Band, momentum intradía del índice
  (identificadores NB_), y de su mezcla con el nivel 1 de caídas (CAI_N1_NB_*); y
  referencia.py, una implementación en Python legible escrita solo con su dossier que da las
  mismas operaciones que el motor del estudio.
- codigo/: el código de los dos estudios (y sus README con la historia de cada ronda), por si
  quieres leer la lógica exacta o ejecutarlo; los datos se descargan de fuentes públicas.
Los dos DOSSIER.md son la referencia y conviene leerlos enteros: tienen las reglas exactas, el
marco común de métricas, la historia, lo que falló y los próximos pasos. Si no puedes abrir el
zip o ejecutar código, trabaja con sus tablas.

ESTRATEGIA 1 · COMPRA DE CAÍDAS (CAI_), dossier 1_caidas
- Compra acciones del S&P 500 (miembros en cada fecha, point-in-time) en tendencia que caen
  ≥ 15 % en 20 sesiones cuando la caída parece forzada y no informativa (11 "módulos de
  origen" y 11 banderas técnicas); compra a la apertura siguiente y vende en el primer cierre
  sobre su SMA10 (máximo 20 sesiones). El dinero que no está en caídas espera en una reserva
  de acciones (las 100 más líquidas con SPY > SMA200; las 50 menos volátiles si no).
- Marco: 2002-2026, señal al cierre, ejecución a la apertura siguiente, 10 pb por lado,
  dividendos. Reglas elegidas con oct-2021 → oct-2026; fuera de muestra = 2002 → sep-2021.
- Recomendada, CAI_N1 (sin deuda): 20,4 % al año fuera de muestra (25,8 % en 2002-2026), peor
  caída -45 %, Sharpe 0,97. Alternativas: CAI_15X (23,0 %), CAI_N3 (24,4 % con costes reales),
  CAI_N2_TRAMOS (18,5 % y -35 %). CAI_SOLO es solo el motor (23 % del capital de media):
  la pieza para combinar con otras. Evidencia B (sin paper trading).
- Paso a paso: secciones 4.1-4.9 de su dossier (4.8 = plan de acción diario).

ESTRATEGIA 2 · NOISE BAND (NB_), dossier 2_noise_band
- Cada día nace en la apertura un "cono" con el movimiento normal desde la apertura a cada
  hora (media de las 14 sesiones anteriores). A las 10:00, 10:30, ..., 15:30, si el precio
  sale del cono se va largo (por arriba) o corto (por abajo); se sale si vuelve dentro o
  cruza la VWAP, y todo se cierra al cierre: nunca duerme. Es la Noise Area de Zarattini,
  Aziz y Barbon (2024).
- Lo que añadimos, siempre fuera de muestra: NB_F03 (entrar solo si la ruptura pasa la banda
  en ≥ 0,3 sigmas) y NB_PUNT (una puntuación lineal con 25 indicadores: fuerza de la
  ruptura, hueco de la noche, caída del mercado, lo que hizo ayer, volatilidad...).
- Marco distinto: velas de 1 minuto de un CFD sobre el S&P 500, 2005 → may-2020 (no hay datos
  posteriores); para operar, futuros MES/ES; costes de SPY a mercado (~0,55 pb por lado);
  búsqueda 2013-2020 y validación 2005-2012; la puntuación se mide entrenando en una mitad y
  probando en la otra, o año a año (NB_PUNT_AA).
- Resultados 2005 → may-2020, sin palanca: artículo (NB_ART) 5,8 % al año, peor caída -12 %,
  Sharpe 0,77; NB_F03 7,1 %, -8,5 %, 1,02; NB_PUNT 7,9 %, -6,5 %, 1,18. Con volatilidad
  objetivo del 2 % (máx. 4×), NB_PUNT_VOL: 15,3 %, -14 %, 1,20. Sola apenas ocupa capital
  (tiene posición el 17-26 % del tiempo de sesión): compárala por Sharpe o como capa.
- Evidencia C: Sharpe deflactado 0,93 con 128 pruebas y 0,95 con 21 efectivas (zona gris);
  parámetros de la base tomados del artículo; faltan datos de 2020-2026; necesita cortos.
- Paso a paso: sección 4 de su dossier. Para comprobar tu réplica: datos/operaciones.csv
  (cada operación), datos/ejemplo_puntuacion.csv (dos rupturas puntuadas paso a paso) y
  referencia.py.

CÓMO ENCAJAN (la cartera recomendada hasta ahora)
- Noise Band gana cuando la compra de caídas pierde: en la crisis de 2008, nivel 1 -42 % y
  Noise Band +51 %; correlación mensual -0,39.
- No compiten por el capital: el nivel 1 está siempre en acciones y Noise Band va encima con
  futuros solo dentro del día. Fórmula: r = r_CAI_N1 + peso × r_NB.
- CAI_N1_NB_PUNT (peso 1), 2005 → may-2020: 26,0 % al año, peor caída -28,7 %, Sharpe 1,04,
  un solo año negativo (2018: -0,2 %); el nivel 1 solo en la misma ventana: 16,6 %, -45 %,
  0,74. Con palanca en Noise Band (CAI_N1_NB_PUNT_VOL): 34,9 %, -34 %, 1,25.

MIS PREFERENCIAS
- Cartera solo con acciones: sin ETFs ni efectivo parado (SPY solo como señal o referencia).
  Futuros, solo como capa dentro del día como en Noise Band. Aún no he decidido si acepto
  cortos ni cuánta palanca: muéstrame las dos opciones.
- Prefiero reglas claras y robustas fuera de muestra a un CAGR más alto que solo existe dentro
  de la muestra.

QUÉ TE PIDO, POR FASES
1. Entender. Resume en unas 15 líneas cómo funcionan CAI_N1, CAI_SOLO, NB_PUNT y la mezcla
   CAI_N1_NB_PUNT, y los dos marcos (sección 2 de cada dossier). Para demostrar que entiendes
   Noise Band, explica con tus palabras el ejemplo 1 de 2_noise_band/datos/
   ejemplo_puntuacion.csv. Lista tus estrategias con universo, periodo, ejecución, costes,
   dividendos, ventana con la que elegiste las reglas y cuántas variantes probaste, y di en
   qué se aparta cada una de cada marco (lista 2.6 del dossier de caídas y 2.5 del de Noise
   Band).
2. Fichas. Rellena la plantilla de la sección 2.5 del dossier de caídas para cada estrategia
   tuya, con reglas exactas y nivel de evidencia (sección 2.4).
3. Mismas métricas. Las de la sección 2.3 del dossier de caídas en sus ventanas (2.2) y,
   además, en la ventana común con Noise Band (21-ene-2005 → 13-may-2020). Exporta la
   rentabilidad diaria neta de cada estrategia tuya en el formato de datos/diario.csv
   (sección 2.7), con un prefijo propio (por ejemplo OTR_).
4. Tabla única. Todas las estrategias de los dos chats (CAI_, NB_, CAI_N1_NB_ y las tuyas),
   ordenadas por su cifra fuera de muestra, diciendo la ventana de cada una, con peor caída,
   Sharpe, Calmar, correlación con SPY, si es solo acciones y nivel de evidencia.
5. Combinar. (a) Tus estrategias siempre invertidas como reserva del motor CAI_SOLO (sección
   9.3 del dossier de caídas, con datos/exposicion_diaria.csv); (b) Noise Band como capa
   encima de tus estrategias (sección 9.1 del de Noise Band); (c) mezclas por pesos (50/50 y
   pesos inversos a la volatilidad, con rebalanceo mensual). Fija las reglas de mezcla antes
   de mirar resultados, valídalas fuera de muestra y cuenta todas las que pruebes.
6. Propuesta. Una o dos carteras finales con reglas claras, qué esperar (meses positivos, peor
   caída, años malos) y los próximos pasos.

REGLAS
- Ninguna cifra vale sin validación fuera de muestra; marca lo que esté dentro de la muestra
  con la que se eligieron las reglas.
- Si comparas estrategias con convenciones distintas (diaria frente a intradía, acciones
  frente a futuros, ventanas distintas), dilo y cuantifica el efecto.
- Si algo no se puede calcular con lo que tienes, dilo y pídeme lo que falta; no lo supongas.

CÓMO RESPONDER
En español, simple y con tablas. Termina siempre con un "Resumen por temas": una tabla de tres
columnas (Tema | Qué descubrimos | ¿Nos ayuda?), corta y bien explicada.

Haz solo la fase 1 y espera mi visto bueno antes de seguir.
